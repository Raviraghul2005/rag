from __future__ import annotations

from app.generation.prompt import build_messages, trim_context
from app.models.retrieval import ScoredChunk


def _chunk(chunk_id: str, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id, text=text, dense_score=0.9, sparse_score=0.5,
        fused_score=0.03, language="hi", doc_id="d1",
    )


def test_trim_context_keeps_top_ranked_chunk_even_if_it_alone_exceeds_budget():
    huge = _chunk("c1", " ".join(["word"] * 500))
    result = trim_context([huge], token_budget=50)
    assert result == [huge]


def test_trim_context_stops_before_exceeding_budget():
    chunks = [_chunk(f"c{i}", " ".join(["word"] * 40)) for i in range(5)]
    result = trim_context(chunks, token_budget=100)
    assert len(result) == 2  # 40 + 40 = 80 fits, +40 more would exceed 100


def test_trim_context_empty_input():
    assert trim_context([], token_budget=100) == []


def test_build_messages_includes_chunk_ids_and_question():
    chunks = [_chunk("c1", "दिल्ली भारत की राजधानी है।")]
    messages = build_messages("राजधानी क्या है?", chunks)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "c1" in messages[1]["content"]
    assert "राजधानी क्या है?" in messages[1]["content"]


def test_build_messages_handles_empty_context():
    messages = build_messages("some question", [])
    assert "no context passages retrieved" in messages[1]["content"]
