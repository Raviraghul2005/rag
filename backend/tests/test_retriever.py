from __future__ import annotations

import json

import pytest

from app.indexing.embeddings import E5Encoder
from app.retrieval.retriever import Retriever
from config.loader import RetrievalConfig
from scripts.build_index import build_strategy, load_documents


@pytest.fixture(scope="module")
def encoder():
    return E5Encoder()


@pytest.fixture(scope="module")
def built_retriever(tmp_path_factory, encoder):
    tmp_path = tmp_path_factory.mktemp("retriever_index")
    corpus_path = tmp_path / "corpus.jsonl"
    rows = [
        {"passage_id": "hi_1", "language": "hi", "query_type": "description",
         "is_selected": True, "source_query_id": "q1",
         "text": "दिल्ली भारत की राजधानी है। यह उत्तर भारत में स्थित एक बड़ा शहर है।"},
        {"passage_id": "hi_2", "language": "hi", "query_type": "description",
         "is_selected": True, "source_query_id": "q2",
         "text": "मुंबई भारत का सबसे बड़ा शहर है और वित्तीय राजधानी मानी जाती है।"},
        {"passage_id": "ta_1", "language": "ta", "query_type": "numeric",
         "is_selected": False, "source_query_id": "q3",
         "text": "இந்தியாவின் மக்கள் தொகை நூற்று நாற்பது கோடிக்கும் அதிகம்."},
    ]
    with open(corpus_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    docs = load_documents(corpus_path)
    build_strategy(
        "recursive_512", docs, encoder=encoder, out_dir=tmp_path / "index",
        ef_construction=32, ef_search=16, force=False,
    )

    from pathlib import Path

    from app.indexing.dense_index import DenseIndex
    from app.indexing.sparse_index import BM25SparseIndex
    from app.indexing.store import MetadataStore

    strategy_dir: Path = tmp_path / "index" / "recursive_512"
    dense = DenseIndex.load(strategy_dir / "dense.faiss")
    sparse = BM25SparseIndex.load(strategy_dir / "sparse")
    store = MetadataStore(strategy_dir / "meta.sqlite3")
    config = RetrievalConfig(
        dense_top_k=10, sparse_top_k=10, rrf_k=60, rerank_candidates=5, final_top_k=2, ef_search=16
    )
    return Retriever(encoder, dense, sparse, store, config)


def test_retrieve_returns_at_most_final_top_k(built_retriever):
    result = built_retriever.retrieve("भारत की राजधानी कौन सा शहर है?")
    assert len(result.results) <= 2
    assert result.query == "भारत की राजधानी कौन सा शहर है?"


def test_retrieve_ranks_relevant_chunk_first(built_retriever):
    result = built_retriever.retrieve("दिल्ली राजधानी")
    assert result.results
    assert "दिल्ली" in result.results[0].text


def test_every_result_carries_full_score_breakdown(built_retriever):
    result = built_retriever.retrieve("मुंबई वित्तीय शहर")
    for r in result.results:
        assert isinstance(r.dense_score, float)
        assert isinstance(r.sparse_score, float)
        assert isinstance(r.fused_score, float)
        assert r.colbert_score is None  # documented CPU-build degradation
        assert r.language
        assert r.doc_id


def test_single_encode_call_per_request(built_retriever, monkeypatch):
    calls = []
    original = built_retriever.encoder.encode_queries

    def counting(texts):
        calls.append(texts)
        return original(texts)

    monkeypatch.setattr(built_retriever.encoder, "encode_queries", counting)
    built_retriever.retrieve("दिल्ली भारत")
    assert len(calls) == 1


def test_degrades_to_dense_only_when_sparse_search_raises(built_retriever, monkeypatch):
    def broken_search(*args, **kwargs):
        raise RuntimeError("sparse index corrupted")

    monkeypatch.setattr(built_retriever.sparse, "search", broken_search)
    result = built_retriever.retrieve("दिल्ली राजधानी")
    assert result.results  # still returns results, just dense-only
    assert all(r.sparse_score == 0.0 for r in result.results)
