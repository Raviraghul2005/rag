from __future__ import annotations

import asyncio
import logging

from app.generation.generator import Generator
from app.guardrails.grounding import GroundingVerifier
from app.guardrails.grounding_gate import grounding_gate
from app.guardrails.input_guardrail import run_input_guardrails
from app.guardrails.retrieval_gate import retrieval_confidence_gate
from app.models.pipeline import PipelineRequest, PipelineResponse, StageTimings
from app.models.retrieval import RetrievalResult
from app.retrieval.retriever import Retriever
from app.timing import stage_timer
from config.loader import GuardrailsConfig

logger = logging.getLogger(__name__)


def _log_outcome(request: PipelineRequest, response: PipelineResponse) -> None:
    """Structured JSON log per request (spec §12.7) — logging_setup.configure_logging()
    renders `extra_fields` as top-level JSON keys, so every field needed for the
    latency tables and abstention curve is queryable from the log stream directly,
    without re-running requests."""
    logger.info(
        "pipeline_request",
        extra={
            "extra_fields": {
                "request_id": request.request_id,
                "transcript": request.transcript,
                "language": request.language,
                "outcome": response.outcome,
                "refusal_reason": response.refusal_reason,
                "strategy_used": response.strategy_used,
                "sufficient": response.sufficient,
                "provider": response.provider,
                "stage_timings_ms": response.timings.stages,
                "total_ms": response.timings.total_ms,
                "retrieval_top_score": response.retrieval_top_score,
                "retrieval_margin": response.retrieval_margin,
                "grounding_score": response.grounding_score,
                "cited_chunk_ids": response.cited_chunk_ids,
            }
        },
    )


def _refused(
    request: PipelineRequest,
    timings: StageTimings,
    reason: str,
    strategy_used: str | None = None,
    **scores: float | None,
) -> PipelineResponse:
    response = PipelineResponse(
        request_id=request.request_id,
        answer=None,
        sufficient=False,
        timings=timings,
        outcome="refused",
        refusal_reason=reason,
        strategy_used=strategy_used,
        **scores,
    )
    _log_outcome(request, response)
    return response


async def run_pipeline(
    request: PipelineRequest,
    retriever: Retriever,
    generator: Generator,
    grounding_verifier: GroundingVerifier | None,
    guardrails: GuardrailsConfig,
    cached_retrieval: RetrievalResult | None = None,
    strategy_used: str | None = None,
) -> PipelineResponse:
    """Orchestrates the measured window (spec §13.1): input guardrails -> retrieval ->
    retrieval-confidence gate -> generation -> grounding verification. Every stage's
    wall time is recorded via time.perf_counter() (app/timing.py), never wall-clock.

    `cached_retrieval`, when given, is a speculative-retrieval hit from
    app.stt.speculative (spec §9: "on final transcript, if it matches the last
    speculative transcript, reuse the cached retrieval — retrieval latency is then
    effectively zero"). The retrieve stage still gets a timing entry so the latency
    table shows the near-zero cost honestly, rather than omitting the stage and
    implying it didn't happen.
    """
    timings = StageTimings()

    with stage_timer(timings, "input_guardrails"):
        input_outcome = (
            run_input_guardrails(request.transcript, request.language)
            if guardrails.enable_input
            else None
        )
    if input_outcome is not None and not input_outcome.allowed:
        return _refused(request, timings, input_outcome.reason, strategy_used=strategy_used)

    with stage_timer(timings, "retrieve"):
        if cached_retrieval is not None:
            retrieval_result = cached_retrieval
        else:
            # retriever.retrieve() is sync, CPU-bound (ONNX encode + FAISS search) —
            # run off-thread so it doesn't block the event loop for other concurrent
            # requests/WebSocket sessions (matters once the STT bridge, app/stt/
            # voice_session.py, has multiple live connections).
            retrieval_result = await asyncio.to_thread(retriever.retrieve, request.transcript)

    top_score = retrieval_result.results[0].fused_score if retrieval_result.results else None
    margin = None
    if retrieval_result.results and len(retrieval_result.results) > 1:
        rest = [r.fused_score for r in retrieval_result.results[1:5]]
        margin = top_score - (sum(rest) / len(rest))

    with stage_timer(timings, "retrieval_gate"):
        gate_outcome = None
        if guardrails.enable_retrieval_gate and guardrails.tau_abs is not None:
            gate_outcome = retrieval_confidence_gate(
                retrieval_result.results, guardrails.tau_abs, guardrails.tau_margin
            )
    if gate_outcome is not None and not gate_outcome.allowed:
        return _refused(
            request, timings, gate_outcome.reason, strategy_used=strategy_used,
            retrieval_top_score=top_score, retrieval_margin=margin,
        )

    with stage_timer(timings, "generate"):
        generated = await generator.generate(request.transcript, retrieval_result.results)

    if generated.provider == "none":
        # Degradation ladder floor (spec §12.6): both LLM providers down. Cite the
        # retrieved passages anyway — an honest "couldn't generate, here's what we
        # found" beats an empty refusal when the corpus genuinely had relevant text.
        response = PipelineResponse(
            request_id=request.request_id,
            answer=None,
            cited_chunk_ids=[r.chunk_id for r in retrieval_result.results],
            cited_chunks=retrieval_result.results,
            sufficient=False,
            timings=timings,
            outcome="unavailable",
            refusal_reason="generation_unavailable",
            retrieval_top_score=top_score,
            retrieval_margin=margin,
            provider="none",
            strategy_used=strategy_used,
        )
        _log_outcome(request, response)
        return response

    if not generated.sufficient or generated.answer is None:
        return _refused(
            request, timings, "insufficient_context", strategy_used=strategy_used,
            retrieval_top_score=top_score, retrieval_margin=margin,
        )

    grounding_score = None
    if guardrails.enable_grounding and grounding_verifier is not None and guardrails.grounding_threshold is not None:
        try:
            with stage_timer(timings, "grounding"):
                # Sync, CPU-bound PyTorch inference (~420ms measured — see
                # app/guardrails/grounding.py) — off-thread for the same reason as the
                # retrieve stage above: don't block the event loop for that long.
                grounding_outcome, grounding_score = await asyncio.to_thread(
                    grounding_gate,
                    grounding_verifier, generated, retrieval_result.results, guardrails.grounding_threshold,
                )
        except Exception:
            # Degradation ladder (spec §12.6): grounding model fails -> serve the
            # answer flagged unverified, don't block the response on a broken guardrail.
            logger.warning("grounding verification failed, serving answer unverified", exc_info=True)
            grounding_outcome = None

        if grounding_outcome is not None and not grounding_outcome.allowed:
            return _refused(
                request, timings, grounding_outcome.reason, strategy_used=strategy_used,
                retrieval_top_score=top_score, retrieval_margin=margin, grounding_score=grounding_score,
            )

    cited_chunk_id_set = set(generated.cited_chunk_ids)
    cited_chunks = [r for r in retrieval_result.results if r.chunk_id in cited_chunk_id_set]

    response = PipelineResponse(
        request_id=request.request_id,
        answer=generated.answer,
        cited_chunk_ids=generated.cited_chunk_ids,
        cited_chunks=cited_chunks,
        sufficient=True,
        timings=timings,
        outcome="answered",
        retrieval_top_score=top_score,
        retrieval_margin=margin,
        grounding_score=grounding_score,
        provider=generated.provider,
        reasoning_tokens=generated.reasoning_tokens,
        strategy_used=strategy_used,
    )
    _log_outcome(request, response)
    return response
