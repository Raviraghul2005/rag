from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Document:
    doc_id: str
    text: str
    language: str
    query_type: str | None = None


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    language: str
    query_type: str | None
    char_start: int
    char_end: int
    strategy: str
    extra: dict[str, Any] = field(default_factory=dict)


class ChunkStrategy(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...
