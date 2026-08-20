from __future__ import annotations

from pydantic import BaseModel, Field


class GeneratedAnswer(BaseModel):
    answer: str | None
    cited_chunk_ids: list[str] = Field(default_factory=list)
    sufficient: bool
    provider: str  # "groq" | "cerebras" | "none"
    reasoning_tokens: int | None = None
    completion_tokens: int | None = None
    parse_ok: bool = True
    truncated: bool = False  # answer exceeded generation.max_tokens and was cut
