from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.generation.generator import Generator, _enforce_answer_cap, _parse_output
from config.loader import GenerationConfig, HarnessConfig, StageTimeoutMs


def _make_response(content: str | None, reasoning_tokens: int = 0, completion_tokens: int = 10):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            completion_tokens=completion_tokens,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        ),
    )


class FakeClient:
    """Stands in for AsyncOpenAI. `responses` is a list of either a response object to
    return or an Exception instance to raise, consumed in order across calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    class _Completions:
        def __init__(self, outer):
            self._outer = outer

        async def create(self, **kwargs):
            self._outer.calls += 1
            item = self._outer._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    @property
    def chat(self):
        return SimpleNamespace(completions=self._Completions(self))


def _config(**overrides) -> GenerationConfig:
    base = dict(
        primary="groq", primary_model="openai/gpt-oss-20b", primary_reasoning_effort="low",
        failover="cerebras", failover_model="gpt-oss-120b",
        max_tokens=5, max_tokens_request=50, context_token_budget=1200,
    )
    base.update(overrides)
    return GenerationConfig(**base)


def _harness(**overrides) -> HarnessConfig:
    base = dict(
        max_retries=1,
        stage_timeout_ms=StageTimeoutMs(encode=100, retrieve=100, rerank=100, generate=2000),
        total_deadline_ms=5000, breaker_failure_threshold=2, breaker_cooldown_s=30,
    )
    base.update(overrides)
    return HarnessConfig(**base)


# --- _parse_output ---


def test_parse_output_clean_json():
    answer, cited, sufficient, ok = _parse_output(
        '{"answer": "Delhi", "cited_chunk_ids": ["c1"], "sufficient": true}'
    )
    assert (answer, cited, sufficient, ok) == ("Delhi", ["c1"], True, True)


def test_parse_output_extracts_json_from_prose_wrapper():
    answer, cited, sufficient, ok = _parse_output(
        'Sure, here it is:\n{"answer": "Delhi", "cited_chunk_ids": [], "sufficient": true}\nHope that helps!'
    )
    assert ok is True
    assert answer == "Delhi"


def test_parse_output_unparseable_is_honest_failure_not_a_guess():
    answer, cited, sufficient, ok = _parse_output("not json at all")
    assert (answer, cited, sufficient, ok) == (None, [], False, False)


def test_parse_output_none_content():
    assert _parse_output(None) == (None, [], False, False)


# --- _enforce_answer_cap ---


def test_enforce_answer_cap_truncates_over_budget():
    answer, truncated = _enforce_answer_cap("one two three four five six seven", max_tokens=3)
    assert answer == "one two three"
    assert truncated is True


def test_enforce_answer_cap_leaves_short_answer_untouched():
    answer, truncated = _enforce_answer_cap("one two", max_tokens=5)
    assert (answer, truncated) == ("one two", False)


def test_enforce_answer_cap_none_passthrough():
    assert _enforce_answer_cap(None, max_tokens=5) == (None, False)


# --- Generator ---


@pytest.mark.asyncio
async def test_generate_success_on_primary():
    groq = FakeClient([_make_response('{"answer": "Delhi", "cited_chunk_ids": ["c1"], "sufficient": true}', reasoning_tokens=8)])
    cerebras = FakeClient([])
    gen = Generator(_config(), _harness(), groq=groq, cerebras=cerebras)

    result = await gen.generate("capital?", [])
    assert result.answer == "Delhi"
    assert result.provider == "groq"
    assert result.reasoning_tokens == 8
    assert cerebras.calls == 0


@pytest.mark.asyncio
async def test_generate_falls_over_to_cerebras_when_groq_exhausts_retries():
    groq = FakeClient([RuntimeError("groq down"), RuntimeError("groq down again")])
    cerebras = FakeClient([_make_response('{"answer": "Mumbai", "cited_chunk_ids": [], "sufficient": true}')])
    gen = Generator(_config(), _harness(max_retries=1), groq=groq, cerebras=cerebras)

    result = await gen.generate("q", [])
    assert result.answer == "Mumbai"
    assert result.provider == "cerebras"
    assert gen.groq_breaker.state.value == "closed"  # one failed request, threshold is 2


@pytest.mark.asyncio
async def test_generate_both_providers_down_returns_honest_unavailable():
    groq = FakeClient([RuntimeError("down")] * 2)
    cerebras = FakeClient([RuntimeError("down")] * 2)
    gen = Generator(_config(), _harness(max_retries=1), groq=groq, cerebras=cerebras)

    result = await gen.generate("q", [])
    assert result.sufficient is False
    assert result.answer is None
    assert result.provider == "none"


@pytest.mark.asyncio
async def test_generate_skips_groq_when_breaker_open():
    groq = FakeClient([])
    cerebras = FakeClient([_make_response('{"answer": "ok", "cited_chunk_ids": [], "sufficient": true}')])
    gen = Generator(_config(), _harness(), groq=groq, cerebras=cerebras)
    gen.groq_breaker.record_failure()
    gen.groq_breaker.record_failure()  # threshold 2 -> open

    result = await gen.generate("q", [])
    assert groq.calls == 0
    assert result.provider == "cerebras"


@pytest.mark.asyncio
async def test_generate_enforces_answer_cap_end_to_end():
    groq = FakeClient([_make_response('{"answer": "one two three four five six", "cited_chunk_ids": [], "sufficient": true}')])
    gen = Generator(_config(max_tokens=3), _harness(), groq=groq, cerebras=FakeClient([]))

    result = await gen.generate("q", [])
    assert result.answer == "one two three"
    assert result.truncated is True
