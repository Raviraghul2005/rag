from __future__ import annotations

import numpy as np

from app.indexing.dense_index import DenseIndex


def _normalized(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_search_returns_nearest_by_inner_product():
    rng = np.random.default_rng(0)
    vectors = _normalized(rng.standard_normal((50, 16)).astype(np.float32))
    index = DenseIndex(dim=16)
    index.add(vectors)

    ids, scores = index.search(vectors[7], top_k=1)
    assert ids[0] == 7
    assert scores[0] > 0.99  # self-match, near-1.0 inner product


def test_save_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    vectors = _normalized(rng.standard_normal((20, 8)).astype(np.float32))
    index = DenseIndex(dim=8)
    index.add(vectors)

    path = tmp_path / "dense.faiss"
    index.save(path)
    reloaded = DenseIndex.load(path)

    assert reloaded.ntotal == 20
    ids, _ = reloaded.search(vectors[3], top_k=1)
    assert ids[0] == 3


def test_ntotal_tracks_additions():
    index = DenseIndex(dim=4)
    assert index.ntotal == 0
    index.add(_normalized(np.random.default_rng(2).standard_normal((5, 4)).astype(np.float32)))
    assert index.ntotal == 5
