from __future__ import annotations

from app.indexing.store import MetadataStore
from app.models.chunk import Chunk


def _chunk(i: int, **extra) -> Chunk:
    return Chunk(
        chunk_id=f"doc1::strat::{i}",
        text=f"chunk text {i}",
        doc_id="doc1",
        language="hi",
        query_type="description",
        char_start=i * 10,
        char_end=i * 10 + 9,
        strategy="strat",
        extra=extra,
    )


def test_add_all_and_get_by_row(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    store.add_all(chunks)

    row = store.get_by_row(1)
    assert row["chunk_id"] == "doc1::strat::1"
    assert row["text"] == "chunk text 1"
    assert store.count() == 3


def test_get_by_chunk_id_round_trip(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.add_all([_chunk(0), _chunk(1)])

    row = store.get_by_chunk_id("doc1::strat::1")
    assert row["row_index"] == 1
    assert row["language"] == "hi"


def test_missing_row_returns_none(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.add_all([_chunk(0)])
    assert store.get_by_row(99) is None
    assert store.get_by_chunk_id("nope") is None


def test_context_vector_stripped_from_stored_extra(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.add_all([_chunk(0, context_vector=[0.1, 0.2], filter_language="hi")])

    row = store.get_by_row(0)
    assert "context_vector" not in row["extra"]
    assert row["extra"]["filter_language"] == "hi"
