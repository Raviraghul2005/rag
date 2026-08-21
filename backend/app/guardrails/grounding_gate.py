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
    question: str = "",
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

    The hypothesis is the question and answer joined, not the answer alone. An NLI
    model scores whether the premise entails a *claim*, and spec §10's 60-token answer
    cap pushes the generator toward terse fragments that aren't claims at all — a bare
    "176 मील" has no subject to entail. Measured on a real case where the passage
    states the distance explicitly: answer alone scored 0.007 (a false rejection of a
    correct answer), question+answer scored 0.994. Crucially this does not soften the
    guardrail — on the same passage a wrong "500 मील" scores 0.0002 and an irrelevant
    answer 0.0006, both still far below threshold.
    """
    if generated.answer is None:
        return GuardrailOutcome(allowed=False, reason="no_answer_to_verify"), 0.0

    cited_texts = [c.text for c in context if c.chunk_id in generated.cited_chunk_ids]
    premise_chunks = cited_texts or [c.text for c in context]
    if not premise_chunks:
        return GuardrailOutcome(allowed=False, reason="no_context_to_verify_against"), 0.0

    premise = " ".join(premise_chunks)
    hypothesis = f"{question} {generated.answer}".strip() if question else generated.answer
    probability = verifier.entailment_probability(premise, hypothesis)
    if probability < threshold:
        return GuardrailOutcome(allowed=False, reason="ungrounded"), probability
    return GuardrailOutcome(allowed=True, reason="ok"), probability
