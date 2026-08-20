from __future__ import annotations

import json

import pytest

from app.indexing.embeddings import E5Encoder
from scripts.build_index import build_strategy, load_documents


@pytest.fixture(scope="module")
def encoder():
    # Real encoder, not a stub: dense_vectors_for() needs it for every strategy except
    # late_chunking (which carries its own context vectors). Module-scoped so the
    # ONNX load/quantize cost is paid once for this file's tests, not once per test.
    return E5Encoder()


def _write_corpus(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_documents_maps_corpus_fields(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus_path,
        [
            {
                "passage_id": "hi_q1_0",
                "language": "hi",
                "query_type": "description",
                "is_selected": True,
                "source_query_id": "q1",
                "text": "भारत एक देश है।",
            }
        ],
    )
    docs = load_documents(corpus_path)
    assert len(docs) == 1
    assert docs[0].doc_id == "hi_q1_0"
    assert docs[0].language == "hi"
    assert docs[0].query_type == "description"


def test_build_strategy_encoder_free_round_trip(tmp_path, encoder):
    docs = load_documents(_seed_corpus(tmp_path))
    stats = build_strategy(
        "recursive_512", docs, encoder=encoder, out_dir=tmp_path / "index",
        ef_construction=32, ef_search=16, force=False,
    )
    assert stats["n_documents"] == 3
    assert stats["n_chunks"] >= 3
    assert (tmp_path / "index" / "recursive_512" / "dense.faiss").exists()
    assert (tmp_path / "index" / "recursive_512" / "sparse" / "meta.json").exists()
    assert (tmp_path / "index" / "recursive_512" / "meta.sqlite3").exists()
    assert (tmp_path / "index" / "recursive_512" / "BUILD_COMPLETE").exists()


def test_build_strategy_skips_when_already_done(tmp_path, encoder):
    docs = load_documents(_seed_corpus(tmp_path))
    out_dir = tmp_path / "index"
    first = build_strategy(
        "fixed_256_overlap_64", docs, encoder=encoder, out_dir=out_dir,
        ef_construction=32, ef_search=16, force=False,
    )
    # Second call must not re-chunk/re-encode — it should just replay the saved stats.
    second = build_strategy(
        "fixed_256_overlap_64", docs, encoder=encoder, out_dir=out_dir,
        ef_construction=32, ef_search=16, force=False,
    )
    assert first == second


def _seed_corpus(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_corpus(
        corpus_path,
        [
            {"passage_id": "hi_1", "language": "hi", "query_type": "description",
             "is_selected": True, "source_query_id": "q1",
             "text": "भारत एक विशाल देश है। यहाँ अनेक भाषाएँ बोली जाती हैं।"},
            {"passage_id": "bn_1", "language": "bn", "query_type": "description",
             "is_selected": True, "source_query_id": "q2",
             "text": "ভারত একটি বিশাল দেশ। এখানে অনেক ভাষায় কথা বলা হয়।"},
            {"passage_id": "ta_1", "language": "ta", "query_type": "numeric",
             "is_selected": False, "source_query_id": "q3",
             "text": "இந்தியா ஒரு பரந்த நாடு. இங்கு பல மொழிகள் பேசப்படுகின்றன."},
        ],
    )
    return corpus_path
