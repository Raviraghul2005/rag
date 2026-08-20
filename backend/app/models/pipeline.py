from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from app.models.retrieval import ScoredChunk


class StageTimings(BaseModel):
    # ms per stage; callers must fill these via time.perf_counter(), never wall-clock (spec §13.3)
    stages: dict[str, float] = Field(default_factory=dict)

    def record(self, name: str, duration_ms: float) -> None:
        self.stages[name] = duration_ms

    @property
    def total_ms(self) -> float:
        return sum(self.stages.values())


class PipelineRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    transcript: str
    language: str | None = None
    # Which chunking strategy's index to retrieve against — None means "use the
    # server's configured default" (config.chunking.active_strategy). Lets the
    # frontend's strategy selector (spec §15 item 5) switch retrieval live and prove
    # all five strategies are real, not just benchmarked offline.
    strategy: str | None = None


class PipelineResponse(BaseModel):
    request_id: str
    answer: str | None
    cited_chunk_ids: list[str] = Field(default_factory=list)
    # Full retrieval records for the cited chunks — spec §15 item 3: "each citation
    # expandable to the source passage and its score." Populated from the same
    # retrieval_result the answer was generated from, not a second lookup.
    cited_chunks: list[ScoredChunk] = Field(default_factory=list)
    sufficient: bool
    timings: StageTimings
    outcome: str  # "answered" | "refused" | "unavailable"
    refusal_reason: str | None = None  # guardrail reason code; spec §15.4's "visible refusal state"
    retrieval_top_score: float | None = None
    retrieval_margin: float | None = None
    grounding_score: float | None = None
    provider: str | None = None  # which LLM served this ("groq" | "cerebras" | "none")
    reasoning_tokens: int | None = None
    strategy_used: str | None = None
