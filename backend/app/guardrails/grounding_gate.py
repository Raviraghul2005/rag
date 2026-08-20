from __future__ import annotations

from app.guardrails.grounding import GroundingVerifier
from app.models.generation import GeneratedAnswer
from app.models.guardrails import GuardrailOutcome
from app.models.retrieval import ScoredChunk


def grounding_gate(
    verifier: GroundingVerifier,
    generated: GeneratedAnswer,
    context: list[ScoredChunk],
    threshold: float,
) -> tuple[GuardrailOutcome, float]:
    """Does the retrieved context actually entail the generated answer (spec §11.3)?
    Returns (outcome, entailment_probability) so the probability gets logged even when
    the gate passes — the number itself is part of the eval package, not just the
    pass/fail decision.

    Premise is the cited chunks' text when the model cited any: entailment-checking
    against what it *claims* it used is the honest comparison. If it claimed
    sufficiency but cited nothing, that's itself suspicious, so this falls back to the
    full retrieved context rather than skipping the check — a model that fabricates an
    answer with no citations should still fail entailment against real context, not
    get a free pass for omitting citations.
    """
    if generated.answer is None:
        return GuardrailOutcome(allowed=False, reason="no_answer_to_verify"), 0.0

    cited_texts = [c.text for c in context if c.chunk_id in generated.cited_chunk_ids]
    premise_chunks = cited_texts or [c.text for c in context]
    if not premise_chunks:
        return GuardrailOutcome(allowed=False, reason="no_context_to_verify_against"), 0.0

    premise = " ".join(premise_chunks)
    probability = verifier.entailment_probability(premise, generated.answer)
    if probability < threshold:
        return GuardrailOutcome(allowed=False, reason="ungrounded"), probability
    return GuardrailOutcome(allowed=True, reason="ok"), probability
