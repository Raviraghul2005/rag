from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.models.chunk import Chunk


class MetadataStore:
    """SQLite store keyed by chunk_id, holding chunk text + payload for filtered search
    (spec §7). One file per chunking strategy, sitting alongside that strategy's dense
    index and sparse index under data/index/<strategy>/.
    """

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                row_index INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE NOT NULL,
                text TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                language TEXT NOT NULL,
                query_type TEXT,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                extra TEXT NOT NULL
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunks(chunk_id)")
        self.conn.commit()

    def add_all(self, chunks: list[Chunk]) -> None:
        """Row index == position in `chunks`, matching the row order used to build the
        dense and sparse indexes for the same strategy."""
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
                (row_index, chunk_id, text, doc_id, language, query_type,
                 char_start, char_end, strategy, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    i,
                    c.chunk_id,
                    c.text,
                    c.doc_id,
                    c.language,
                    c.query_type,
                    c.char_start,
                    c.char_end,
                    c.strategy,
                    json.dumps({k: v for k, v in c.extra.items() if k != "context_vector"}),
                )
                for i, c in enumerate(chunks)
            ],
        )
        self.conn.commit()

    def get_by_row(self, row_index: int) -> dict | None:
        row = self.conn.execute(
            "SELECT chunk_id, text, doc_id, language, query_type, extra "
            "FROM chunks WHERE row_index = ?",
            (row_index,),
        ).fetchone()
        if row is None:
            return None
        chunk_id, text, doc_id, language, query_type, extra = row
        return {
            "chunk_id": chunk_id,
            "text": text,
            "doc_id": doc_id,
            "language": language,
            "query_type": query_type,
            "extra": json.loads(extra),
        }

    def get_by_chunk_id(self, chunk_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT row_index, text, doc_id, language, query_type, extra "
            "FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        row_index, text, doc_id, language, query_type, extra = row
        return {
            "row_index": row_index,
            "text": text,
            "doc_id": doc_id,
            "language": language,
            "query_type": query_type,
            "extra": json.loads(extra),
        }

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
