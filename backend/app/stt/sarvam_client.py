from __future__ import annotations

import base64
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlencode

import websockets

# Protocol verified live against docs.sarvam.ai/api-reference/speech-to-text/transcribe/
# realtime/ws, 2026-08-20 (spec §0 rule 1: fetch and verify, don't invent). Saaras v3 is
# the endpoint's default and only currently-accepted `model` value.
SARVAM_WS_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
MODEL = "saaras:v3-realtime"


@dataclass
class TranscriptEvent:
    kind: Literal["partial", "final"]
    text: str
    language: str | None
    utterance_idx: int


@dataclass
class SarvamError:
    code: str
    message: str
    is_fatal: bool


class SarvamSTTClient:
    """Thin async wrapper around Sarvam's realtime STT WebSocket protocol.

    The browser never talks to Sarvam directly (spec §4, §15: the key must never reach
    the client bundle) — this class is what app.main's `/ws/transcribe` endpoint uses
    server-side, reading SARVAM_API_KEY from the process environment (Space Secrets in
    production, backend/.env locally via app.env_bootstrap).
    """

    def __init__(
        self, language_code: str = "auto", sample_rate: int = 16000, api_key: str | None = None
    ):
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.api_key = api_key or os.environ["SARVAM_API_KEY"]
        self._ws: websockets.ClientConnection | None = None

    async def __aenter__(self) -> "SarvamSTTClient":
        params = {
            "language_code": self.language_code,
            "model": MODEL,
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
        }
        url = f"{SARVAM_WS_URL}?{urlencode(params)}"
        self._ws = await websockets.connect(
            url, additional_headers={"API-SUBSCRIPTION-KEY": self.api_key}
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._ws is not None:
            await self._ws.close()

    async def send_audio(self, pcm_bytes: bytes) -> None:
        assert self._ws is not None, "use as an async context manager"
        await self._ws.send(
            json.dumps({"event": "audio_input", "audio": base64.b64encode(pcm_bytes).decode("ascii")})
        )

    async def end(self) -> None:
        assert self._ws is not None, "use as an async context manager"
        await self._ws.send(json.dumps({"event": "end"}))

    async def events(self) -> AsyncIterator[TranscriptEvent | SarvamError]:
        """Yields transcript and error events, silently absorbing the protocol's other
        event types (session.begin, vad.speech_start/end, config.updated, pong) — this
        pipeline doesn't need them, and a future caller that does can extend this."""
        assert self._ws is not None, "use as an async context manager"
        async for raw in self._ws:
            data = json.loads(raw)
            event = data.get("event")
            if event == "transcript.partial":
                yield TranscriptEvent("partial", data["text"], data.get("language"), data.get("utterance_idx", 0))
            elif event == "transcript.final":
                yield TranscriptEvent("final", data["text"], data.get("language"), data.get("utterance_idx", 0))
            elif event == "error":
                is_fatal = bool(data.get("is_fatal"))
                yield SarvamError(data.get("code", "unknown"), data.get("message", ""), is_fatal)
                if is_fatal:
                    return
            elif event == "session.end":
                return
