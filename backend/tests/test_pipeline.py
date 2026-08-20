from __future__ import annotations

import pytest

from app.models.generation import GeneratedAnswer
from app.models.pipeline import PipelineRequest
from app.models.retrieval import RetrievalResult, ScoredChunk
from app.pipeline import run_pipeline
from config.loader import GuardrailsConfig


def _chunk(chunk_id: str, fused: float, text: str = "context text") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id, text=text, dense_score=0.9, sparse_score=0.5,
        fused_score=fused, language="hi", doc_id="d1",
    )


class FakeRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query):
        return RetrievalResult(query=query, results=self._results)


class FakeGenerator:
    def __init__(self, answer: GeneratedAnswer):
        self._answer = answer
        self.calls = 0

    async def generate(self, question, context):
        self.calls += 1
        return self._answer


class FakeGroundingVerifier:
    def __init__(self, probability=None, raises=False):
        self.probability = probability
        self.raises = raises

    def entailment_probability(self, premise, hypothesis):
        if self.raises:
            raise RuntimeError("model crashed")
        return self.probability


def _guardrails(**overrides) -> GuardrailsConfig:
    base = dict(
        tau_abs=0.01, tau_margin=0.01, grounding_threshold=0.5,
        enable_input=True, enable_retrieval_gate=True, enable_grounding=True,
    )
    base.update(overrides)
    return GuardrailsConfig(**base)


def _sufficient_answer(**overrides) -> GeneratedAnswer:
    base = dict(answer="Delhi", cited_chunk_ids=["c1"], sufficient=True, provider="groq")
    base.update(overrides)
    return GeneratedAnswer(**base)


@pytest.mark.asyncio
async def test_input_guardrail_blocks_before_retrieval():
    retriever = FakeRetriever([_chunk("c1", 0.03)])
    generator = FakeGenerator(_sufficient_answer())

    response = await run_pipeline(
        PipelineRequest(transcript="How to make a bomb"),
        retriever, generator, FakeGroundingVerifier(), _guardrails(),
    )
    assert response.outcome == "refused"
    assert response.refusal_reason == "unsafe_content"
    assert generator.calls == 0


@pytest.mark.asyncio
async def test_retrieval_gate_blocks_low_margin():
    results = [_chunk("c1", 0.02), _chunk("c2", 0.019), _chunk("c3", 0.0195)]
    generator = FakeGenerator(_sufficient_answer())

    response = await run_pipeline(
        PipelineRequest(transcript="भारत की राजधानी क्या है?"),
        FakeRetriever(results), generator, FakeGroundingVerifier(), _guardrails(),
    )
    assert response.outcome == "refused"
    assert response.refusal_reason == "low_margin_confidence"
    assert generator.calls == 0


@pytest.mark.asyncio
async def test_retrieval_gate_skipped_when_uncalibrated():
    # tau_abs=None means "not calibrated yet" (spec §11.2) — gate must not silently
    # apply some default threshold, it must be skipped entirely.
    results = [_chunk("c1", 0.0001)]  # would fail any real threshold
    generator = FakeGenerator(_sufficient_answer())

    response = await run_pipeline(
        PipelineRequest(transcript="भारत की राजधानी क्या है?"),
        FakeRetriever(results), generator, FakeGroundingVerifier(0.9), _guardrails(tau_abs=None),
    )
    assert response.outcome == "answered"


@pytest.mark.asyncio
async def test_both_providers_down_returns_unavailable_with_passages():
    results = [_chunk("c1", 0.03), _chunk("c2", 0.005)]
    generator = FakeGenerator(
        GeneratedAnswer(answer=None, cited_chunk_ids=[], sufficient=False, provider="none")
    )

    response = await run_pipeline(
        PipelineRequest(transcript="भारत की राजधानी क्या है?"),
        FakeRetriever(results), generator, FakeGroundingVerifier(), _guardrails(),
    )
    assert response.outcome == "unavailable"
    assert response.refusal_reason == "generation_unavailable"
    assert response.cited_chunk_ids == ["c1", "c2"]


@pytest.mark.asyncio
async def test_insufficient_context_is_refused():
    results = [_chunk("c1", 0.03)]
    generator = FakeGenerator(
        GeneratedAnswer(answer=None, cited_chunk_ids=[], sufficient=False, provider="groq")
    )

    response = await run_pipeline(
        PipelineRequest(transcript="भारत की राजधानी क्या है?"),
        FakeRetriever(results), generator, FakeGroundingVerifier(), _guardrails(),
    )
    assert response.outcome == "refused"
    assert response.refusal_reason == "insufficient_context"


@pytest.mark.asyncio
async def test_grounding_gate_blocks_ungrounded_answer():
    results = [_chunk("c1", 0.03)]
    generator = FakeGenerator(_sufficient_answer())

    response = await run_pipeline(
        PipelineRequest(transcript="भारत की राजधानी क्या है?"),
        FakeRetriever(results), generator, FakeGroundingVerifier(probability=0.1), _guardrails(),
    )
    assert response.outcome == "refused"
    assert response.refusal_reason == "ungrounded"
    assert response.grounding_score == 0.1


@pytest.mark.asyncio
async def test_grounding_failure_degrades_to_unverified_not_a_block():
    results = [_chunk("c1", 0.03)]
    generator = FakeGenerator(_sufficient_answer())

    response = await run_pipeline(
        PipelineRequest(transcript="भारत की राजधानी क्या है?"),
        FakeRetriever(results), generator, FakeGroundingVerifier(raises=True), _guardrails(),
    )
    assert response.outcome == "answered"
    assert response.sufficient is True
    assert response.grounding_score is None


@pytest.mark.asyncio
async def test_grounding_skipped_when_disabled():
    results = [_chunk("c1", 0.03)]
    generator = FakeGenerator(_sufficient_answer())

    response = await run_pipeline(
        PipelineRequest(transcript="भारत की राजधानी क्या है?"),
        FakeRetriever(results), generator, None, _guardrails(enable_grounding=False),
    )
    assert response.outcome == "answered"


@pytest.mark.asyncio
async def test_full_happy_path_populates_all_scores():
    results = [_chunk("c1", 0.03), _chunk("c2", 0.005)]
    generator = FakeGenerator(_sufficient_answer(reasoning_tokens=12))

    response = await run_pipeline(
        PipelineRequest(transcript="भारत की राजधानी क्या है?"),
        FakeRetriever(results), generator, FakeGroundingVerifier(probability=0.9), _guardrails(),
    )
    assert response.outcome == "answered"
    assert response.answer == "Delhi"
    assert response.cited_chunk_ids == ["c1"]
    assert response.retrieval_top_score == 0.03
    assert response.grounding_score == 0.9
    assert response.provider == "groq"
    assert set(response.timings.stages) >= {"input_guardrails", "retrieve", "retrieval_gate", "generate", "grounding"}


@pytest.mark.asyncio
async def test_cached_retrieval_skips_calling_retriever():
    class TrackingRetriever(FakeRetriever):
        def __init__(self, results):
            super().__init__(results)
            self.calls = 0

        def retrieve(self, query):
            self.calls += 1
            return super().retrieve(query)

    retriever = TrackingRetriever([_chunk("c1", 0.03)])
    cached = RetrievalResult(query="cached", results=[_chunk("c_cached", 0.03)])
    generator = FakeGenerator(_sufficient_answer(cited_chunk_ids=["c_cached"]))

    response = await run_pipeline(
        PipelineRequest(transcript="भारत की राजधानी क्या है?"),
        retriever, generator, FakeGroundingVerifier(0.9), _guardrails(),
        cached_retrieval=cached,
    )
    assert retriever.calls == 0
    assert response.cited_chunk_ids == ["c_cached"]
    assert "retrieve" in response.timings.stages
