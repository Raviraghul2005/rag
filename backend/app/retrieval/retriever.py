from __future__ import annotations

import logging

from app.indexing.dense_index import DenseIndex
from app.indexing.embeddings import E5Encoder
from app.indexing.sparse_index import BM25SparseIndex
from app.indexing.store import MetadataStore
from app.models.retrieval import RetrievalResult, ScoredChunk
from app.retrieval.fusion import reciprocal_rank_fusion
from config.loader import RetrievalConfig

logger = logging.getLogger(__name__)


class Retriever:
    """Dense + sparse hybrid retrieval with RRF fusion (spec §8).

    ColBERT late-interaction rerank is dropped: it needs BGE-M3's multi-vector output,
    and this build's CPU pivot to e5-small (see app/indexing/embeddings.py) only
    produces single dense vectors. `rerank_candidates` is kept as a config knob for
    the degradation ladder's shape, but on this build it's a no-op pass-through
    straight from the fused ranking to `final_top_k` — every result's `colbert_score`
    is `None`, which is the graceful-degradation state spec §12 defines for
    "ColBERT rerank fails -> serve fused results", here permanent rather than a
    failure branch.
    """

    def __init__(
        self,
        encoder: E5Encoder,
        dense: DenseIndex,
        sparse: BM25SparseIndex,
        store: MetadataStore,
        config: RetrievalConfig,
    ):
        self.encoder = encoder
        self.dense = dense
        self.sparse = sparse
        self.store = store
        self.config = config

    def retrieve(self, query: str) -> RetrievalResult:
        # Single encode call for the whole request (spec §8: "never encode the query
        # more than once per request") — its vector feeds only the dense search; BM25
        # tokenizes the raw query text separately, no second model call involved.
        query_vector = self.encoder.encode_queries([query])[0]

        dense_ids, dense_scores = self.dense.search(query_vector, top_k=self.config.dense_top_k)
        try:
            sparse_hits = self.sparse.search(query, top_k=self.config.sparse_top_k)
        except Exception:
            # Degradation ladder (spec §12.6): sparse index fails -> dense-only. Logged,
            # not swallowed silently — an operator needs to know the sparse side is down
            # even though the request itself still succeeds.
            logger.warning("sparse index search failed, degrading to dense-only", exc_info=True)
            sparse_hits = []

        candidates: dict[str, dict] = {}

        dense_chunk_ids: list[str] = []
        for row, score in zip(dense_ids, dense_scores):
            if row < 0:
                continue
            meta = self.store.get_by_row(int(row))
            if meta is None:
                continue
            dense_chunk_ids.append(meta["chunk_id"])
            candidates.setdefault(meta["chunk_id"], {"meta": meta})["dense_score"] = float(score)

        sparse_chunk_ids: list[str] = []
        for row, score in sparse_hits:
            meta = self.store.get_by_row(row)
            if meta is None:
                continue
            sparse_chunk_ids.append(meta["chunk_id"])
            candidates.setdefault(meta["chunk_id"], {"meta": meta})["sparse_score"] = float(score)

        fused = reciprocal_rank_fusion([dense_chunk_ids, sparse_chunk_ids], k=self.config.rrf_k)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        top = ordered[: self.config.final_top_k]

        results = [
            ScoredChunk(
                chunk_id=chunk_id,
                text=candidates[chunk_id]["meta"]["text"],
                dense_score=candidates[chunk_id].get("dense_score", 0.0),
                sparse_score=candidates[chunk_id].get("sparse_score", 0.0),
                fused_score=fused_score,
                colbert_score=None,
                language=candidates[chunk_id]["meta"]["language"],
                doc_id=candidates[chunk_id]["meta"]["doc_id"],
            )
            for chunk_id, fused_score in top
        ]
        return RetrievalResult(query=query, results=results)
