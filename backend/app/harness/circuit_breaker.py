from __future__ import annotations

import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"        # normal operation
    OPEN = "open"             # tripped, calls routed to failover
    HALF_OPEN = "half_open"   # cooldown elapsed, probing with the next call


class CircuitBreaker:
    """Per-provider breaker (spec §12.4): N consecutive failures opens the circuit and
    routes to failover; after a cooldown, one probe call is allowed through (half-open).
    A probe success closes the circuit; a probe failure reopens it and restarts the
    cooldown clock.

    Not thread-safe by design — one breaker instance per provider per process, driven
    from the single async pipeline task, matches how it's used in app/generation.
    """

    def __init__(self, failure_threshold: int, cooldown_s: float):
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._state is CircuitState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.cooldown_s:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def allow_request(self) -> bool:
        return self.state is not CircuitState.OPEN

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        was_probing = self._state is CircuitState.HALF_OPEN
        if was_probing or self._consecutive_failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
