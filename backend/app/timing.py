from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from app.models.pipeline import StageTimings


@contextmanager
def stage_timer(timings: StageTimings, stage_name: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        timings.record(stage_name, (perf_counter() - start) * 1000)
