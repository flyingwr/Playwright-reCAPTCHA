from __future__ import annotations

import asyncio
import functools
import io
import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import pydub
import speech_recognition
from playwright.async_api import Page, Request, Response

from playwright_recaptcha.errors import (
    RecaptchaNotFoundError,
    RecaptchaRateLimitError,
    RecaptchaSolveError,
)
from playwright_recaptcha.recaptchav2.recaptcha_box import AsyncRecaptchaBox


class AsyncSolver:
    """
    A class used to solve reCAPTCHA v2 asynchronously.

    Parameters
    ----------
    page : Page
        The playwright page to solve the reCAPTCHA on.
    attempts : int, optional
        The number of solve attempts, by default 3.
    has_checkbox : bool, optional
        True if the page has a reCAPTCHA checkbox to be selected

    Attributes
    ----------
    token : Optional[str]
        The g-recaptcha-response token.
    userverify_retrieved : bool
        True if the solver has received a response for reCAPTCHA userverify request
    waiting_for_uvresp : bool
        True if the page is waiting for reCAPTCHA userverify response

    Methods
    -------
    close() -> None
        Remove the userverify response listener.
    solve_recaptcha(attempts: Optional[int] = None) -> str
        Solve the reCAPTCHA and return the g-recaptcha-response token.

    Raises
    ------
    RecaptchaNotFoundError
        If the reCAPTCHA was not found.
    RecaptchaRateLimitError
        If the reCAPTCHA rate limit has been exceeded.
    RecaptchaSolveError
        If the reCAPTCHA could not be solved.
    """

    def __init__(self, page: Page, attempts: int = 3, has_checkbox: bool = True) -> None:
        self._page = page
        self._attempts = attempts
        self.has_checkbox = has_checkbox

        self.token: Optional[str] = None

        self.userverify_retrieved: bool = False
        self.waiting_for_uvresp: bool = False

        self._page.on("request", self._handle_request)
        self._page.on("response", self._extract_token)

    def __repr__(self) -> str:
        return f"AsyncSolver(page={self._page!r}, attempts={self._attempts!r})"

    async def __aenter__(self) -> AsyncSolver:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _convert_audio_to_text(self, audio_url: str) -> Optional[str]:
        """
        Convert the reCAPTCHA audio to text.

        Parameters
        ----------
        audio_url : str
            The reCAPTCHA audio URL.

        Returns
        -------
        Optional[str]
            The reCAPTCHA audio text.
        """
        loop = asyncio.get_event_loop()
        response = await self._page.request.get(audio_url)

        wav_audio = io.BytesIO()
        mp3_audio = io.BytesIO(await response.body())

        with ThreadPoolExecutor() as executor:
            audio = await loop.run_in_executor(
                executor, pydub.AudioSegment.from_mp3, mp3_audio
            )

            await loop.run_in_executor(
                executor, functools.partial(audio.export, wav_audio, format="wav")
            )

            recognizer = speech_recognition.Recognizer()

            with speech_recognition.AudioFile(wav_audio) as source:
                audio_data = await loop.run_in_executor(
                    executor, recognizer.record, source
                )

            try:
                return await loop.run_in_executor(
                    executor, recognizer.recognize_google, audio_data
                )
            except speech_recognition.UnknownValueError:
                return None

    async def _random_delay(self) -> None:
        """Delay the execution for a random amount of time between 1 and 4 seconds."""
        await self._page.wait_for_timeout(random.randint(1000, 4000))

    async def _handle_request(self, request: Request):
        if re.search("/recaptcha/(api2|enterprise)/userverify", request.url) is not None:
            self.waiting_for_uvresp = True

    async def _extract_token(self, response: Response) -> None:
        """
        Extract the g-recaptcha-response token from the userverify response.

        Parameters
        ----------
        response : Response
            The response to extract the g-recaptcha-response token from.
        """
        if re.search("/recaptcha/(api2|enterprise)/userverify", response.url) is None:
            return
        
        token_match = re.search('"uvresp","(.*?)"', await response.text())

        if token_match is not None:
            self.token = token_match.group(1)

        self.waiting_for_uvresp = False
        self.userverify_retrieved = True

    async def _solve(self, recaptcha_box: AsyncRecaptchaBox) -> None:
        """
        Click the reCAPTCHA checkbox.

        Parameters
        ----------
        recaptcha_box : AsyncRecaptchaBox
            The reCAPTCHA box.
        """
        if self.has_checkbox:
            await recaptcha_box.checkbox.click(force=True)

        while recaptcha_box.frames_are_attached():
            if await recaptcha_box.is_solved() if self.has_checkbox else self.userverify_retrieved:
                if self.token is None:
                    raise RecaptchaSolveError

                break

            if (
                await recaptcha_box.audio_challenge_is_visible()
                or await recaptcha_box.audio_challenge_button.is_visible()
                and await recaptcha_box.audio_challenge_button.is_enabled()
            ):
                break

            await self._page.wait_for_timeout(250)

    async def _get_audio_url(self, recaptcha_box: AsyncRecaptchaBox) -> str:
        """
        Get the reCAPTCHA audio URL.

        Parameters
        ----------
        recaptcha_box : AsyncRecaptchaBox
            The reCAPTCHA box.

        Returns
        -------
        str
            The reCAPTCHA audio URL.

        Raises
        ------
        RecaptchaRateLimitError
            If the reCAPTCHA rate limit has been exceeded.
        """
        if await recaptcha_box.audio_challenge_button.is_visible():
            await recaptcha_box.audio_challenge_button.click(force=True)

        while True:
            if await recaptcha_box.rate_limit_is_visible():
                raise RecaptchaRateLimitError

            if await recaptcha_box.audio_challenge_is_visible():
                break

            await self._page.wait_for_timeout(250)

        return await recaptcha_box.audio_download_button.get_attribute("href")

    async def _submit_audio_text(
        self, recaptcha_box: AsyncRecaptchaBox, text: str
    ) -> None:
        """
        Submit the reCAPTCHA audio text.

        Parameters
        ----------
        recaptcha_box : AsyncRecaptchaBox
            The reCAPTCHA box.
        text : str
            The reCAPTCHA audio text.

        Raises
        ------
        RecaptchaRateLimitError
            If the reCAPTCHA rate limit has been exceeded.
        """
        await recaptcha_box.audio_challenge_textbox.fill(text)

        async with self._page.expect_response(
            re.compile("/recaptcha/(api2|enterprise)/userverify")
        ):
            await recaptcha_box.audio_challenge_verify_button.click()

        while recaptcha_box.frames_are_attached():
            if await recaptcha_box.rate_limit_is_visible():
                raise RecaptchaRateLimitError

            if (
                await recaptcha_box.new_challenge_button.is_disabled()
                or await recaptcha_box.solve_failure_is_visible()
                or await recaptcha_box.is_solved()
            ):
                break

            await self._page.wait_for_timeout(250)

    async def close(self) -> None:
        """Remove the userverify response listener."""
        while self.waiting_for_uvresp:
            await asyncio.sleep(0)

        try:
            self._page.remove_listener("request", self._handle_request)
            self._page.remove_listener("response", self._extract_token)
        except KeyError:
            pass

    async def solve_recaptcha(self, attempts: Optional[int] = None) -> str:
        """
        Solve the reCAPTCHA and return the g-recaptcha-response token.

        Parameters
        ----------
        attempts : Optional[int], optional
            The number of solve attempts, by default 3.

        Returns
        -------
        str
            The g-recaptcha-response token.

        Raises
        ------
        RecaptchaNotFoundError
            If the reCAPTCHA was not found.
        RecaptchaRateLimitError
            If the reCAPTCHA rate limit has been exceeded.
        RecaptchaSolveError
            If the reCAPTCHA could not be solved.
        """
        self.token = None
        self.userverify_retrieved = False

        attempts = attempts or self._attempts
        recaptcha_box = await AsyncRecaptchaBox.from_frames(self._page.frames, self.has_checkbox)

        if (
            await recaptcha_box.checkbox.is_hidden() if self.has_checkbox else True
            and await recaptcha_box.audio_challenge_button.is_disabled()
        ):
            raise RecaptchaNotFoundError

        if await recaptcha_box.checkbox.is_visible() if self.has_checkbox else True:
            await self._solve(recaptcha_box)

            if self.token is not None:
                return self.token

        while attempts > 0:
            await self._random_delay()
            url = await self._get_audio_url(recaptcha_box)
            text = await self._convert_audio_to_text(url)

            if text is None:
                await recaptcha_box.new_challenge_button.click()
                attempts -= 1
                continue

            await self._random_delay()
            await self._submit_audio_text(recaptcha_box, text)

            if (
                recaptcha_box.frames_are_detached()
                or await recaptcha_box.new_challenge_button.is_disabled()
                or await recaptcha_box.is_solved()
            ):
                if self.token is None:
                    raise RecaptchaSolveError

                return self.token

            await recaptcha_box.new_challenge_button.click()
            attempts -= 1

        raise RecaptchaSolveError
