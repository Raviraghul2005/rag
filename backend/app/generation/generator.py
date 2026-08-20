from __future__ import annotations

import json
import re

from openai import AsyncOpenAI

from app.chunking.base import word_spans
from app.generation.prompt import build_messages, trim_context
from app.generation.providers import cerebras_client, groq_client
from app.harness.circuit_breaker import CircuitBreaker
from app.harness.retry import RetryExhausted, retry_with_backoff
from app.models.generation import GeneratedAnswer
from app.models.retrieval import ScoredChunk
from config.loader import GenerationConfig, HarnessConfig

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_output(content: str | None) -> tuple[str | None, list[str], bool, bool]:
    """Returns (answer, cited_chunk_ids, sufficient, parse_ok). A model that ignores the
    system prompt's JSON-only instruction (wraps it in prose, markdown fences, etc.)
    gets one fallback attempt via regex before being treated as a parse failure — which
    the caller turns into an honest refusal, never a guess."""
    if not content:
        return None, [], False, False
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(content)
        if not match:
            return None, [], False, False
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, [], False, False

    if not isinstance(data, dict):
        return None, [], False, False
    answer = data.get("answer")
    cited = data.get("cited_chunk_ids")
    if not isinstance(cited, list):
        cited = []
    sufficient = bool(data.get("sufficient", False))
    return (answer if isinstance(answer, str) else None), [str(c) for c in cited], sufficient, True


def _enforce_answer_cap(answer: str | None, max_tokens: int) -> tuple[str | None, bool]:
    """Truncates to the spec's reported answer-length cap using the same whitespace-word
    proxy as elsewhere in the codebase (app/chunking/base.py, generation/prompt.py) —
    needed because max_tokens_request intentionally exceeds max_tokens to leave room for
    a reasoning model's invisible thinking tokens (see config/default.yaml)."""
    if answer is None:
        return None, False
    spans = word_spans(answer)
    if len(spans) <= max_tokens:
        return answer, False
    cutoff = spans[max_tokens - 1][1]
    return answer[:cutoff].rstrip(), True


class Generator:
    """Primary/failover generation with per-provider retry (backoff+jitter) and circuit
    breaker (spec §12.3-4). Groq is primary, Cerebras is failover; a request falls
    through to Cerebras only if Groq's breaker is open or Groq's retries are exhausted
    for this request — not on every request.
    """

    def __init__(
        self,
        config: GenerationConfig,
        harness_config: HarnessConfig,
        groq: AsyncOpenAI | None = None,
        cerebras: AsyncOpenAI | None = None,
    ):
        self.config = config
        self.max_retries = harness_config.max_retries
        self.groq = groq or groq_client()
        self.cerebras = cerebras or cerebras_client()
        self.groq_breaker = CircuitBreaker(
            harness_config.breaker_failure_threshold, harness_config.breaker_cooldown_s
        )
        self.cerebras_breaker = CircuitBreaker(
            harness_config.breaker_failure_threshold, harness_config.breaker_cooldown_s
        )

    async def generate(self, question: str, context: list[ScoredChunk]) -> GeneratedAnswer:
        trimmed = trim_context(context, self.config.context_token_budget)
        messages = build_messages(question, trimmed)

        if self.groq_breaker.allow_request():
            result = await self._try_provider(
                self.groq, self.config.primary_model, messages, provider="groq",
                breaker=self.groq_breaker, reasoning_effort=self.config.primary_reasoning_effort,
            )
            if result is not None:
                return result

        if self.cerebras_breaker.allow_request():
            result = await self._try_provider(
                self.cerebras, self.config.failover_model, messages, provider="cerebras",
                breaker=self.cerebras_breaker, reasoning_effort=None,
            )
            if result is not None:
                return result

        # Both providers unavailable (breaker open or retries exhausted on both) — the
        # degradation ladder's floor (spec §12.6): honest unavailability, never a guess.
        return GeneratedAnswer(
            answer=None, cited_chunk_ids=[], sufficient=False, provider="none", parse_ok=False
        )

    async def _try_provider(
        self,
        client: AsyncOpenAI,
        model: str,
        messages: list[dict],
        provider: str,
        breaker: CircuitBreaker,
        reasoning_effort: str | None,
    ) -> GeneratedAnswer | None:
        try:
            response = await retry_with_backoff(
                lambda: self._call(client, model, messages, reasoning_effort),
                max_retries=self.max_retries,
            )
        except RetryExhausted:
            breaker.record_failure()
            return None

        breaker.record_success()
        msg = response.choices[0].message
        answer, cited, sufficient, parse_ok = _parse_output(msg.content)
        answer, truncated = _enforce_answer_cap(answer, self.config.max_tokens)

        usage = response.usage
        details = usage.completion_tokens_details if usage else None
        return GeneratedAnswer(
            answer=answer if parse_ok else None,
            cited_chunk_ids=cited,
            sufficient=(sufficient and parse_ok),
            provider=provider,
            reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
            completion_tokens=usage.completion_tokens if usage else None,
            parse_ok=parse_ok,
            truncated=truncated,
        )

    async def _call(self, client: AsyncOpenAI, model: str, messages: list[dict], reasoning_effort: str | None):
        kwargs = {}
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=self.config.max_tokens_request,
            response_format={"type": "json_object"},
            **kwargs,
        )
