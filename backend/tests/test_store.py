from __future__ import annotations

import asyncio

import pytest

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


@pytest.mark.asyncio
async def test_readable_from_a_worker_thread(tmp_path):
    """Regression test: app/pipeline.py reads the store via
    asyncio.to_thread(retriever.retrieve, ...), which runs on a different thread than
    the one that constructed this connection (main thread, at app startup). Without
    check_same_thread=False this raises "SQLite objects created in a thread can only
    be used in that same thread" on every real request - caught by
    scripts/replay_benchmark.py against the live index, not by any prior unit test,
    since those all exercise the pipeline against fake retrievers that never touch a
    real sqlite connection."""
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.add_all([_chunk(0)])

    row = await asyncio.to_thread(store.get_by_row, 0)

    assert row["chunk_id"] == "doc1::strat::0"
