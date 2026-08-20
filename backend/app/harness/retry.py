from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class RetryExhausted(Exception):
    """All attempts failed. Wraps the last error so callers can inspect the real cause
    without walking a chain of retry-loop frames."""

    def __init__(self, attempts: int, last_error: BaseException):
        super().__init__(f"exhausted {attempts} attempt(s): {last_error!r}")
        self.attempts = attempts
        self.last_error = last_error


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int,
    base_delay_s: float = 0.2,
    max_delay_s: float = 5.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Exponential backoff with full jitter (spec §12.3: "retries with exponential
    backoff + jitter on all external calls"). Bounded attempts (max_retries + 1 total
    tries) and bounded per-attempt delay (max_delay_s) give the caller a predictable
    worst-case time, which is what lets a total request deadline sit on top of this.
    """
    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retry_on as exc:
            last_error = exc
            if attempt == max_retries:
                break
            # Full jitter (AWS's term): uniform(0, backoff) rather than backoff +/- jitter,
            # so retries from concurrent callers spread out instead of clustering.
            backoff = min(max_delay_s, base_delay_s * (2**attempt))
            await asyncio.sleep(random.random() * backoff)
    assert last_error is not None
    raise RetryExhausted(max_retries + 1, last_error)
