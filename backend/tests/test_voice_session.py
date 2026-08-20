from __future__ import annotations

import asyncio

import pytest

from app.models.generation import GeneratedAnswer
from app.models.retrieval import RetrievalResult, ScoredChunk
from app.stt.sarvam_client import SarvamError, TranscriptEvent
from app.stt.speculative import SpeculativeRetrievalCache
from app.stt.voice_session import _handle_transcript_event, _speculate
from config.loader import GuardrailsConfig


def _chunk(chunk_id="c1", fused=0.03):
    return ScoredChunk(
        chunk_id=chunk_id, text="दिल्ली भारत की राजधानी है।", dense_score=0.9,
        sparse_score=0.5, fused_score=fused, language="hi", doc_id="d1",
    )


class FakeRetriever:
    def __init__(self, results):
        self._results = results
        self.calls = 0

    def retrieve(self, query):
        self.calls += 1
        return RetrievalResult(query=query, results=self._results)


class FakeGenerator:
    def __init__(self, answer):
        self._answer = answer

    async def generate(self, question, context):
        return self._answer


class FakeWebSocket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, data):
        self.sent.append(data)


def _guardrails(**overrides) -> GuardrailsConfig:
    base = dict(
        tau_abs=0.01, tau_margin=0.01, grounding_threshold=None,
        enable_input=True, enable_retrieval_gate=True, enable_grounding=False,
    )
    base.update(overrides)
    return GuardrailsConfig(**base)


@pytest.mark.asyncio
async def test_speculate_stores_result_on_success():
    retriever = FakeRetriever([_chunk()])
    cache = SpeculativeRetrievalCache()

    await _speculate("भारत की राजधानी", retriever, cache)

    resolved = cache.resolve("भारत की राजधानी")
    assert resolved is not None
    assert retriever.calls == 1


@pytest.mark.asyncio
async def test_speculate_stores_exception_on_failure():
    class BrokenRetriever:
        def retrieve(self, query):
            raise RuntimeError("index corrupted")

    cache = SpeculativeRetrievalCache()
    await _speculate("भारत", BrokenRetriever(), cache)

    assert cache.resolve("भारत") is None
    assert cache.stats.errors == 1


@pytest.mark.asyncio
async def test_partial_event_forwards_text_and_schedules_speculation():
    ws = FakeWebSocket()
    retriever = FakeRetriever([_chunk()])
    generator = FakeGenerator(None)
    cache = SpeculativeRetrievalCache()
    tasks: dict = {}

    event = TranscriptEvent(kind="partial", text="भारत की राजधानी क्या", language="hi-IN", utterance_idx=0)
    await _handle_transcript_event(event, ws, retriever, generator, None, _guardrails(), cache, tasks)

    assert ws.sent == [{"type": "partial_transcript", "text": "भारत की राजधानी क्या"}]
    assert "भारत की राजधानी क्या" in tasks
    await tasks["भारत की राजधानी क्या"]  # let the background speculation finish
    assert cache.resolve("भारत की राजधानी क्या") is not None


@pytest.mark.asyncio
async def test_final_event_runs_pipeline_and_sends_answer():
    ws = FakeWebSocket()
    retriever = FakeRetriever([_chunk()])
    generator = FakeGenerator(
        GeneratedAnswer(answer="दिल्ली", cited_chunk_ids=["c1"], sufficient=True, provider="groq")
    )
    cache = SpeculativeRetrievalCache()
    tasks: dict = {}

    event = TranscriptEvent(kind="final", text="भारत की राजधानी क्या है?", language="hi-IN", utterance_idx=0)
    await _handle_transcript_event(event, ws, retriever, generator, None, _guardrails(), cache, tasks)

    kinds = [m["type"] for m in ws.sent]
    assert kinds == ["final_transcript", "answer"]
    assert ws.sent[1]["payload"]["answer"] == "दिल्ली"
    assert retriever.calls == 1  # no speculative hit was queued, so a normal retrieve ran


@pytest.mark.asyncio
async def test_final_event_reuses_speculative_hit_without_recalling_retriever():
    ws = FakeWebSocket()
    retriever = FakeRetriever([_chunk()])
    generator = FakeGenerator(
        GeneratedAnswer(answer="दिल्ली", cited_chunk_ids=["c1"], sufficient=True, provider="groq")
    )
    cache = SpeculativeRetrievalCache()
    tasks: dict = {}
    text = "भारत की राजधानी क्या है?"

    # Simulate a partial that already resolved before the final arrives.
    tasks[text] = asyncio.create_task(_speculate(text, retriever, cache))
    await tasks[text]
    assert retriever.calls == 1

    event = TranscriptEvent(kind="final", text=text, language="hi-IN", utterance_idx=0)
    await _handle_transcript_event(event, ws, retriever, generator, None, _guardrails(), cache, tasks)

    assert retriever.calls == 1  # still 1 — the final path reused the cached result
    assert cache.stats.hits == 1


@pytest.mark.asyncio
async def test_non_fatal_sarvam_error_is_just_logged_by_caller():
    # The event itself carries is_fatal; _handle_transcript_event only handles
    # TranscriptEvent — SarvamError routing lives in run_voice_session's loop, this
    # just confirms the dataclass shape callers rely on.
    error = SarvamError(code="bad_audio", message="unsupported sample rate", is_fatal=False)
    assert error.is_fatal is False
    assert error.code == "bad_audio"
