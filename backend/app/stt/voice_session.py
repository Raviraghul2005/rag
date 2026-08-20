from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket

from app.generation.generator import Generator
from app.guardrails.grounding import GroundingVerifier
from app.models.pipeline import PipelineRequest
from app.pipeline import run_pipeline
from app.retrieval.retriever import Retriever
from app.stt.sarvam_client import SarvamError, SarvamSTTClient, TranscriptEvent
from app.stt.speculative import SpeculativeRetrievalCache
from config.loader import GuardrailsConfig

logger = logging.getLogger(__name__)


async def _speculate(
    partial_text: str, retriever: Retriever, cache: SpeculativeRetrievalCache
) -> None:
    """Fires retrieval for a partial transcript in the background (spec §9). Errors
    are stored in the cache rather than raised — a broken speculative attempt must
    never surface on the final path, only show up as a logged miss/error stat."""
    try:
        result = await asyncio.to_thread(retriever.retrieve, partial_text)
        cache.store(partial_text, result)
    except Exception as exc:
        logger.warning("speculative retrieval failed for partial transcript", exc_info=True)
        cache.store(partial_text, exc)


async def run_voice_session(
    client_ws: WebSocket,
    retriever: Retriever,
    generator: Generator,
    grounding_verifier: GroundingVerifier | None,
    guardrails: GuardrailsConfig,
    language_code: str = "auto",
) -> None:
    """Bridges the browser's audio WebSocket to Sarvam's realtime STT and drives
    speculative retrieval off Sarvam's partial transcripts (spec §9). The browser
    never talks to Sarvam directly (spec §4, §15) — SARVAM_API_KEY lives only in this
    process's environment, and this function is the only thing that opens a
    connection to Sarvam.

    Wire protocol to the browser (JSON text frames out, raw PCM16 binary frames in):
      in:  binary frames of 16kHz mono linear16 PCM audio; a text frame "__end__" to
           close the utterance
      out: {"type": "partial_transcript", "text": ...}
           {"type": "final_transcript", "text": ...}
           {"type": "answer", "payload": <PipelineResponse>}
           {"type": "stt_error", "message": ...}
    """
    cache = SpeculativeRetrievalCache()
    speculative_tasks: dict[str, asyncio.Task] = {}

    async with SarvamSTTClient(language_code=language_code) as sarvam:

        async def forward_audio() -> None:
            while True:
                message = await client_ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                audio_bytes = message.get("bytes")
                if audio_bytes:
                    await sarvam.send_audio(audio_bytes)
                elif message.get("text") == "__end__":
                    await sarvam.end()
                    return

        async def handle_transcripts() -> None:
            async for event in sarvam.events():
                if isinstance(event, SarvamError):
                    logger.warning("sarvam error: %s (code=%s)", event.message, event.code)
                    await client_ws.send_json({"type": "stt_error", "message": event.message})
                    if event.is_fatal:
                        return
                    continue
                await _handle_transcript_event(
                    event, client_ws, retriever, generator, grounding_verifier,
                    guardrails, cache, speculative_tasks,
                )

        forward_task = asyncio.create_task(forward_audio())
        transcript_task = asyncio.create_task(handle_transcripts())
        try:
            await asyncio.gather(forward_task, transcript_task)
        finally:
            for task in (*speculative_tasks.values(), forward_task, transcript_task):
                if not task.done():
                    task.cancel()


async def _handle_transcript_event(
    event: TranscriptEvent,
    client_ws: WebSocket,
    retriever: Retriever,
    generator: Generator,
    grounding_verifier: GroundingVerifier | None,
    guardrails: GuardrailsConfig,
    cache: SpeculativeRetrievalCache,
    speculative_tasks: dict[str, asyncio.Task],
) -> None:
    if event.kind == "partial":
        await client_ws.send_json({"type": "partial_transcript", "text": event.text})
        if cache.should_speculate(event.text) and event.text not in speculative_tasks:
            speculative_tasks[event.text] = asyncio.create_task(
                _speculate(event.text, retriever, cache)
            )
        return

    # event.kind == "final"
    await client_ws.send_json({"type": "final_transcript", "text": event.text})

    # Give a speculative task that's still in flight a brief moment to land rather
    # than discarding a near-certain hit — spec §9's "reuse the cached retrieval" only
    # helps if the final transcript usually arrives after the matching partial's
    # retrieval has actually finished, which a slow speculative call could still miss.
    task = speculative_tasks.get(event.text)
    if task is not None and not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.05)
        except (TimeoutError, Exception):
            pass

    cached_result = cache.resolve(event.text)
    request = PipelineRequest(transcript=event.text, language=event.language)
    response = await run_pipeline(
        request, retriever, generator, grounding_verifier, guardrails,
        cached_retrieval=cached_result,
    )
    await client_ws.send_json({"type": "answer", "payload": response.model_dump()})
