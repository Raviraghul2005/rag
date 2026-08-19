from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


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


class PipelineResponse(BaseModel):
    request_id: str
    answer: str | None
    cited_chunk_ids: list[str] = Field(default_factory=list)
    sufficient: bool
    timings: StageTimings
    outcome: str
