from __future__ import annotations

from app.retrieval.fusion import reciprocal_rank_fusion


def test_chunk_ranked_first_in_both_lists_scores_highest():
    dense = ["a", "b", "c"]
    sparse = ["a", "c", "b"]
    scores = reciprocal_rank_fusion([dense, sparse], k=60)
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]


def test_chunk_present_in_only_one_list_still_scores():
    dense = ["a", "b"]
    sparse = ["c"]
    scores = reciprocal_rank_fusion([dense, sparse], k=60)
    assert set(scores) == {"a", "b", "c"}
    assert scores["c"] == 1.0 / 61  # rank 1 in its only list


def test_empty_lists_produce_empty_scores():
    assert reciprocal_rank_fusion([[], []], k=60) == {}


def test_smaller_k_amplifies_rank_differences():
    ranked = ["a", "b", "c"]
    scores_small_k = reciprocal_rank_fusion([ranked], k=1)
    scores_large_k = reciprocal_rank_fusion([ranked], k=1000)
    gap_small = scores_small_k["a"] - scores_small_k["c"]
    gap_large = scores_large_k["a"] - scores_large_k["c"]
    assert gap_small > gap_large
