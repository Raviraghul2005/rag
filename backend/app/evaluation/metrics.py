from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float | None:
    """Fraction of the query's known-relevant passages that appear in the top k
    retrieved results. Returns None (not 0.0) when relevant_ids is empty — that's an
    undefined ratio, not a failure, and averaging code must skip it rather than let it
    silently drag the mean down (spec §14.1's table is per-strategy, per-language; a
    query with no ground-truth relevant passages shouldn't exist in a well-built eval
    set, but this stays honest about the edge case rather than assuming it can't happen).
    """
    if not relevant_ids:
        return None
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float | None:
    """Reciprocal rank of the first relevant result, 0.0 if none of the retrieved
    results are relevant. None when relevant_ids is empty, same reasoning as recall_at_k.
    """
    if not relevant_ids:
        return None
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float | None:
    """Binary-relevance nDCG@k (each relevant passage contributes gain 1, not a graded
    relevance score — the eval set only has is_selected boolean labels, spec §5).
    None when relevant_ids is empty, same reasoning as recall_at_k.
    """
    if not relevant_ids:
        return None
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mean_ignoring_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None
