from __future__ import annotations

from app.models.retrieval import RetrievalResult, ScoredChunk
from app.stt.speculative import SpeculativeRetrievalCache


def _result(query: str) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        results=[ScoredChunk(chunk_id="c1", text="t", dense_score=0.9, sparse_score=0.5,
                              fused_score=0.03, language="hi", doc_id="d1")],
    )


def test_should_speculate_respects_min_tokens():
    cache = SpeculativeRetrievalCache(min_tokens=3)
    assert cache.should_speculate("one two") is False
    assert cache.should_speculate("one two three") is True


def test_exact_match_hit():
    cache = SpeculativeRetrievalCache()
    cache.store("भारत की राजधानी", _result("भारत की राजधानी"))

    resolved = cache.resolve("भारत की राजधानी")
    assert resolved is not None
    assert cache.stats.hits == 1
    assert cache.stats.misses == 0


def test_no_match_is_a_miss():
    cache = SpeculativeRetrievalCache()
    cache.store("भारत की राजधानी क्या", _result("भारत की राजधानी क्या"))

    resolved = cache.resolve("पूरी तरह अलग प्रश्न")
    assert resolved is None
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0


def test_normalizes_whitespace_before_hashing():
    cache = SpeculativeRetrievalCache()
    cache.store("भारत   की  राजधानी", _result("भारत की राजधानी"))

    resolved = cache.resolve("भारत की राजधानी")
    assert resolved is not None
    assert cache.stats.hits == 1


def test_stored_exception_counts_as_error_not_hit():
    cache = SpeculativeRetrievalCache()
    cache.store("भारत की राजधानी", RuntimeError("speculative retrieval crashed"))

    resolved = cache.resolve("भारत की राजधानी")
    assert resolved is None
    assert cache.stats.errors == 1
    assert cache.stats.hits == 0


def test_hit_rate_computation():
    cache = SpeculativeRetrievalCache()
    cache.store("a", _result("a"))
    cache.resolve("a")  # hit
    cache.resolve("b")  # miss
    cache.resolve("c")  # miss
    assert cache.stats.hit_rate == 1 / 3


def test_clear_empties_cache():
    cache = SpeculativeRetrievalCache()
    cache.store("भारत", _result("भारत"))
    cache.clear()
    assert cache.resolve("भारत") is None
