from __future__ import annotations

from app.models.guardrails import GuardrailOutcome
from app.models.retrieval import ScoredChunk


def retrieval_confidence_gate(
    results: list[ScoredChunk], tau_abs: float, tau_margin: float
) -> GuardrailOutcome:
    """Abstain when top-1 fused score is low in absolute terms, OR when the margin
    between top-1 and the rest is thin (spec §11.2). The margin test is the one that
    catches off-topic queries specifically: several mediocre, similarly-scored matches
    is the signature of "nothing in the corpus is actually relevant," which a flat
    absolute threshold alone would miss whenever every score in the corpus happens to
    sit a bit low or high. Costs nothing extra — reuses scores retrieval already
    computed.
    """
    if not results:
        return GuardrailOutcome(allowed=False, reason="no_results")

    top1 = results[0].fused_score
    if top1 < tau_abs:
        return GuardrailOutcome(allowed=False, reason="low_absolute_confidence")

    rest = [r.fused_score for r in results[1:5]]
    if rest:
        margin = top1 - (sum(rest) / len(rest))
        if margin < tau_margin:
            return GuardrailOutcome(allowed=False, reason="low_margin_confidence")

    return GuardrailOutcome(allowed=True, reason="ok")
