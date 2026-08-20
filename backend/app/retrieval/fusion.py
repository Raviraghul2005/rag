from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> dict[str, float]:
    """Combines multiple ranked lists of chunk_ids into one fused score per chunk_id.

    RRF score for a chunk_id = sum over lists it appears in of 1/(k + rank), rank
    1-indexed. A chunk_id absent from a list simply contributes nothing from that list
    — RRF needs no score normalization across dense/sparse, which is the point of using
    it here (spec §8): dense inner-product and BM25 scores live on incomparable scales.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores
