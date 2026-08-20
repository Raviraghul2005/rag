from __future__ import annotations

import pytest

from app.harness.retry import RetryExhausted, retry_with_backoff


@pytest.mark.asyncio
async def test_succeeds_on_first_try_without_delay():
    calls = []

    async def func():
        calls.append(1)
        return "ok"

    result = await retry_with_backoff(func, max_retries=3, base_delay_s=0.001)
    assert result == "ok"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retries_then_succeeds():
    attempts = {"n": 0}

    async def func():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("transient")
        return "recovered"

    result = await retry_with_backoff(func, max_retries=5, base_delay_s=0.001, max_delay_s=0.01)
    assert result == "recovered"
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_exhausts_and_raises_with_last_error():
    async def func():
        raise ValueError("always fails")

    with pytest.raises(RetryExhausted) as exc_info:
        await retry_with_backoff(func, max_retries=2, base_delay_s=0.001, max_delay_s=0.01)
    assert exc_info.value.attempts == 3  # initial try + 2 retries
    assert isinstance(exc_info.value.last_error, ValueError)


@pytest.mark.asyncio
async def test_only_retries_specified_exception_types():
    async def func():
        raise KeyError("not retryable here")

    with pytest.raises(KeyError):
        await retry_with_backoff(
            func, max_retries=3, base_delay_s=0.001, retry_on=(ValueError,)
        )
