from __future__ import annotations

import math

from app.evaluation.metrics import mean_ignoring_none, mrr, ndcg_at_k, recall_at_k


def test_recall_at_k_full_hit():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_at_k_partial_hit():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_at_k_respects_k_cutoff():
    # "b" is relevant but ranked outside the top 1.
    assert recall_at_k(["x", "b"], {"b"}, k=1) == 0.0


def test_recall_at_k_empty_relevant_set_is_none():
    assert recall_at_k(["a", "b"], set(), k=5) is None


def test_mrr_first_result_relevant():
    assert mrr(["a", "b"], {"a"}) == 1.0


def test_mrr_second_result_relevant():
    assert mrr(["x", "a"], {"a"}) == 0.5


def test_mrr_no_relevant_result_found():
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_mrr_empty_relevant_set_is_none():
    assert mrr(["a"], set()) is None


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_ndcg_worse_ranking_scores_lower_than_perfect():
    perfect = ndcg_at_k(["a", "b"], {"a", "b"}, k=2)
    worse = ndcg_at_k(["x", "a"], {"a", "b"}, k=2)
    assert worse < perfect


def test_ndcg_no_hits_is_zero():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_ndcg_matches_hand_computed_value():
    # relevant = {a, c}; retrieved = [a, b, c] -> DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5
    # IDCG (ideal: a, c both in top 2) = 1/log2(2) + 1/log2(3) = 1 + 0.6309...
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c"}
    expected_dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(retrieved, relevant, k=3) == expected_dcg / expected_idcg


def test_mean_ignoring_none_skips_nones():
    assert mean_ignoring_none([1.0, None, 3.0]) == 2.0


def test_mean_ignoring_none_all_none_returns_none():
    assert mean_ignoring_none([None, None]) is None
