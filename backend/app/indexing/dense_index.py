from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class DenseIndex:
    """FAISS HNSW over L2-normalized vectors, inner-product metric (spec §7).

    Row order is the index's implicit id space (HNSWFlat assigns sequential ids on add
    and offers no external-id remap) — callers keep a parallel chunk_id_order list to
    translate row ids back to chunk_ids. Same convention as the sparse index and
    metadata store, all three built from one shared chunk ordering.
    """

    def __init__(self, dim: int, ef_construction: int = 200, ef_search: int = 64):
        self.dim = dim
        self.ef_search = ef_search
        self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (row_indices, inner_product_scores), both shape (top_k,). Missing
        results (fewer candidates than top_k) come back as row index -1 from FAISS."""
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, ids = self.index.search(query, top_k)
        return ids[0], scores[0]

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path, ef_search: int = 64) -> "DenseIndex":
        index = faiss.read_index(str(path))
        wrapper = cls.__new__(cls)
        wrapper.dim = index.d
        wrapper.ef_search = ef_search
        wrapper.index = index
        wrapper.index.hnsw.efSearch = ef_search
        return wrapper
