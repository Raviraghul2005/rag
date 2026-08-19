from __future__ import annotations

from pydantic import BaseModel


class ScoredChunk(BaseModel):
    chunk_id: str
    text: str
    dense_score: float
    sparse_score: float
    fused_score: float
    colbert_score: float | None = None
    language: str
    doc_id: str


class RetrievalResult(BaseModel):
    query: str
    results: list[ScoredChunk]
