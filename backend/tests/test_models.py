from app.models.chunk import Chunk
from app.models.guardrails import GuardrailOutcome
from app.models.retrieval import ScoredChunk


def test_chunk_construction():
    chunk = Chunk(
        chunk_id="c1",
        text="sample",
        doc_id="d1",
        language="hi",
        query_type=None,
        char_start=0,
        char_end=6,
        strategy="recursive_512",
    )
    assert chunk.extra == {}


def test_guardrail_outcome_shape():
    outcome = GuardrailOutcome(allowed=False, reason="off_topic")
    assert outcome.allowed is False


def test_scored_chunk_colbert_score_optional():
    chunk = ScoredChunk(
        chunk_id="c1",
        text="sample",
        dense_score=0.9,
        sparse_score=0.5,
        fused_score=0.8,
        language="hi",
        doc_id="d1",
    )
    assert chunk.colbert_score is None
