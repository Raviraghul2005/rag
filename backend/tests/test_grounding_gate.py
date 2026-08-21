from __future__ import annotations

from app.guardrails.grounding_gate import grounding_gate
from app.models.generation import GeneratedAnswer
from app.models.retrieval import ScoredChunk


class FakeVerifier:
    """Stands in for GroundingVerifier — no model load in the fast unit-test suite.
    Returns a fixed probability regardless of input, unless `by_pair` overrides it for
    specific (premise, hypothesis) pairs."""

    def __init__(self, probability: float = 0.9, by_pair: dict | None = None):
        self.probability = probability
        self.by_pair = by_pair or {}
        self.calls: list[tuple[str, str]] = []

    def entailment_probability(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        return self.by_pair.get((premise, hypothesis), self.probability)


def _chunk(chunk_id: str, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id, text=text, dense_score=0.9, sparse_score=0.5,
        fused_score=0.03, language="hi", doc_id="d1",
    )


def test_passes_when_entailment_above_threshold():
    verifier = FakeVerifier(probability=0.95)
    generated = GeneratedAnswer(answer="Delhi", cited_chunk_ids=["c1"], sufficient=True, provider="groq")
    context = [_chunk("c1", "Delhi is the capital of India.")]

    outcome, prob = grounding_gate(verifier, generated, context, threshold=0.5)
    assert outcome.allowed is True
    assert prob == 0.95


def test_fails_when_entailment_below_threshold():
    verifier = FakeVerifier(probability=0.1)
    generated = GeneratedAnswer(answer="Mumbai", cited_chunk_ids=["c1"], sufficient=True, provider="groq")
    context = [_chunk("c1", "Delhi is the capital of India.")]

    outcome, prob = grounding_gate(verifier, generated, context, threshold=0.5)
    assert outcome.allowed is False
    assert outcome.reason == "ungrounded"
    assert prob == 0.1


def test_no_answer_short_circuits_without_calling_verifier():
    verifier = FakeVerifier()
    generated = GeneratedAnswer(answer=None, cited_chunk_ids=[], sufficient=False, provider="groq")

    outcome, prob = grounding_gate(verifier, generated, [_chunk("c1", "text")], threshold=0.5)
    assert outcome.allowed is False
    assert outcome.reason == "no_answer_to_verify"
    assert verifier.calls == []


def test_premise_uses_only_cited_chunks_when_citations_present():
    verifier = FakeVerifier()
    generated = GeneratedAnswer(answer="Delhi", cited_chunk_ids=["c1"], sufficient=True, provider="groq")
    context = [
        _chunk("c1", "Delhi is the capital."),
        _chunk("c2", "Unrelated distractor passage."),
    ]

    grounding_gate(verifier, generated, context, threshold=0.5)
    premise_used = verifier.calls[0][0]
    assert "Delhi is the capital." in premise_used
    assert "Unrelated distractor" not in premise_used


def test_falls_back_to_full_context_when_no_citations_given():
    verifier = FakeVerifier()
    generated = GeneratedAnswer(answer="Delhi", cited_chunk_ids=[], sufficient=True, provider="groq")
    context = [_chunk("c1", "Delhi is the capital.")]

    grounding_gate(verifier, generated, context, threshold=0.5)
    assert "Delhi is the capital." in verifier.calls[0][0]


def test_no_context_at_all_fails_without_calling_verifier():
    verifier = FakeVerifier()
    generated = GeneratedAnswer(answer="Delhi", cited_chunk_ids=[], sufficient=True, provider="groq")

    outcome, prob = grounding_gate(verifier, generated, [], threshold=0.5)
    assert outcome.allowed is False
    assert outcome.reason == "no_context_to_verify_against"
    assert verifier.calls == []


def test_hypothesis_combines_question_and_answer():
    """Regression: a terse answer alone isn't a claim an NLI model can verify. Measured
    on a real case (passage stating a distance explicitly), the bare answer "176 मील"
    scored 0.007 and was wrongly refused, while question+answer scored 0.994. The gate
    must send the combined form."""
    verifier = FakeVerifier(probability=0.9)
    generated = GeneratedAnswer(answer="176 मील", cited_chunk_ids=["c1"], sufficient=True, provider="groq")
    context = [_chunk("c1", "स्कॉट्सडेल से ग्रैंड कैन्यन तक की दूरी 176 मील है।")]

    grounding_gate(verifier, generated, context, threshold=0.5, question="दूरी स्कॉट्सडेल से ग्रैंड कैन्यन तक")

    hypothesis_used = verifier.calls[0][1]
    assert "दूरी स्कॉट्सडेल से ग्रैंड कैन्यन तक" in hypothesis_used
    assert "176 मील" in hypothesis_used


def test_falls_back_to_answer_alone_when_no_question_given():
    verifier = FakeVerifier(probability=0.9)
    generated = GeneratedAnswer(answer="Delhi", cited_chunk_ids=["c1"], sufficient=True, provider="groq")

    grounding_gate(verifier, generated, [_chunk("c1", "Delhi is the capital.")], threshold=0.5)

    assert verifier.calls[0][1] == "Delhi"
