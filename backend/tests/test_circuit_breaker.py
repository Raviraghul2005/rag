from __future__ import annotations

from app.harness.circuit_breaker import CircuitBreaker, CircuitState


def test_starts_closed_and_allows_requests():
    breaker = CircuitBreaker(failure_threshold=3, cooldown_s=30)
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow_request()


def test_opens_after_threshold_consecutive_failures():
    breaker = CircuitBreaker(failure_threshold=3, cooldown_s=30)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.allow_request()


def test_success_resets_failure_count():
    breaker = CircuitBreaker(failure_threshold=3, cooldown_s=30)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED  # only 2 consecutive since the reset


def test_transitions_to_half_open_after_cooldown(monkeypatch):
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=10)
    clock = {"t": 0.0}
    monkeypatch.setattr("app.harness.circuit_breaker.time.monotonic", lambda: clock["t"])

    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN

    clock["t"] = 5.0
    assert breaker.state is CircuitState.OPEN  # cooldown not elapsed yet

    clock["t"] = 10.0
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow_request()


def test_half_open_success_closes_circuit(monkeypatch):
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=10)
    clock = {"t": 0.0}
    monkeypatch.setattr("app.harness.circuit_breaker.time.monotonic", lambda: clock["t"])

    breaker.record_failure()
    clock["t"] = 10.0
    assert breaker.state is CircuitState.HALF_OPEN
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_half_open_failure_reopens_and_restarts_cooldown(monkeypatch):
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=10)
    clock = {"t": 0.0}
    monkeypatch.setattr("app.harness.circuit_breaker.time.monotonic", lambda: clock["t"])

    breaker.record_failure()
    clock["t"] = 10.0
    assert breaker.state is CircuitState.HALF_OPEN
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN

    clock["t"] = 15.0  # 5s since the reopen — cooldown restarted, not elapsed
    assert breaker.state is CircuitState.OPEN

    clock["t"] = 20.0  # full 10s since the reopen
    assert breaker.state is CircuitState.HALF_OPEN
