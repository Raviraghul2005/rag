from __future__ import annotations

from app.guardrails.retrieval_gate import retrieval_confidence_gate
from app.models.retrieval import ScoredChunk


def _chunk(fused: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id="c", text="t", dense_score=0.5, sparse_score=0.5,
        fused_score=fused, language="hi", doc_id="d",
    )


def test_blocks_when_no_results():
    result = retrieval_confidence_gate([], tau_abs=0.01, tau_margin=0.005)
    assert result.allowed is False
    assert result.reason == "no_results"


def test_blocks_low_absolute_confidence():
    results = [_chunk(0.001), _chunk(0.0005)]
    result = retrieval_confidence_gate(results, tau_abs=0.01, tau_margin=0.0001)
    assert result.allowed is False
    assert result.reason == "low_absolute_confidence"


def test_blocks_low_margin_many_mediocre_matches():
    # High absolute score but everything else is nearly as high — the "off-topic
    # query with several similar mediocre matches" signature spec §11.2 targets.
    results = [_chunk(0.02), _chunk(0.019), _chunk(0.0195), _chunk(0.0198)]
    result = retrieval_confidence_gate(results, tau_abs=0.01, tau_margin=0.01)
    assert result.allowed is False
    assert result.reason == "low_margin_confidence"


def test_allows_strong_clear_top_result():
    results = [_chunk(0.03), _chunk(0.005), _chunk(0.004)]
    result = retrieval_confidence_gate(results, tau_abs=0.01, tau_margin=0.01)
    assert result.allowed is True


def test_single_result_only_checks_absolute_threshold():
    results = [_chunk(0.02)]
    result = retrieval_confidence_gate(results, tau_abs=0.01, tau_margin=0.5)
    assert result.allowed is True  # no "rest" to compute a margin against
