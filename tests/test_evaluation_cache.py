"""Focused concurrency tests for the async single-flight evaluation cache."""

from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar

import pytest

from agent_evolve.application.evaluation_cache import (
    AsyncEvaluationCache,
    EvaluationCacheEventType,
    EvaluationCacheTraceEvent,
)


T = TypeVar("T")


def run(awaitable: Awaitable[T]) -> T:
    return asyncio.run(awaitable)


async def eventually(predicate, *, turns: int = 100) -> None:
    for _ in range(turns):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


def test_concurrent_identical_requests_coalesce_and_later_hit() -> None:
    async def scenario() -> None:
        cache: AsyncEvaluationCache[tuple[bool, str]] = AsyncEvaluationCache()
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def evaluate() -> tuple[bool, str]:
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            # An ordinary invalid domain result is data, not a cache failure.
            return False, "constraint violation"

        first = asyncio.create_task(cache.get_or_evaluate("config-a", evaluate))
        await started.wait()
        second = asyncio.create_task(cache.get_or_evaluate("config-a", evaluate))
        await eventually(lambda: not second.done())
        release.set()

        assert await first == (False, "constraint violation")
        assert await second == (False, "constraint violation")
        assert await cache.get_or_evaluate("config-a", evaluate) == (
            False,
            "constraint violation",
        )
        assert calls == 1
        assert await cache.snapshot() == cache_snapshot(
            cached_entries=1,
            hits=1,
            misses=1,
            coalesced=1,
        )

    run(scenario())


def test_distinct_hashes_evaluate_independently() -> None:
    async def scenario() -> None:
        cache: AsyncEvaluationCache[str] = AsyncEvaluationCache()
        started: set[str] = set()
        both_started = asyncio.Event()
        release = asyncio.Event()

        async def evaluate(key: str) -> str:
            started.add(key)
            if len(started) == 2:
                both_started.set()
            await release.wait()
            return f"value:{key}"

        first = asyncio.create_task(
            cache.get_or_evaluate("config-a", lambda: evaluate("config-a"))
        )
        second = asyncio.create_task(
            cache.get_or_evaluate("config-b", lambda: evaluate("config-b"))
        )
        await both_started.wait()
        release.set()

        assert await asyncio.gather(first, second) == ["value:config-a", "value:config-b"]
        assert await cache.snapshot() == cache_snapshot(cached_entries=2, misses=2)

    run(scenario())


def test_exception_is_not_cached_and_later_request_retries() -> None:
    async def scenario() -> None:
        cache: AsyncEvaluationCache[str] = AsyncEvaluationCache()
        calls = 0

        async def evaluate() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("evaluation failed")
            return "recovered"

        with pytest.raises(RuntimeError, match="evaluation failed"):
            await cache.get_or_evaluate("config-a", evaluate)
        assert await cache.snapshot() == cache_snapshot(misses=1)

        assert await cache.get_or_evaluate("config-a", evaluate) == "recovered"
        assert calls == 2
        assert await cache.snapshot() == cache_snapshot(cached_entries=1, misses=2)

    run(scenario())


def test_cancelling_one_awaiter_does_not_cancel_shared_evaluation() -> None:
    async def scenario() -> None:
        cache: AsyncEvaluationCache[str] = AsyncEvaluationCache()
        started = asyncio.Event()
        release = asyncio.Event()
        evaluator_cancelled = False
        calls = 0

        async def evaluate() -> str:
            nonlocal calls, evaluator_cancelled
            calls += 1
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                evaluator_cancelled = True
                raise
            return "shared"

        cancelled_waiter = asyncio.create_task(
            cache.get_or_evaluate("config-a", evaluate)
        )
        await started.wait()
        surviving_waiter = asyncio.create_task(
            cache.get_or_evaluate("config-a", evaluate)
        )
        await asyncio.sleep(0)

        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter
        assert not evaluator_cancelled

        release.set()
        assert await surviving_waiter == "shared"
        assert calls == 1
        assert await cache.get_or_evaluate("config-a", evaluate) == "shared"
        assert await cache.snapshot() == cache_snapshot(
            cached_entries=1,
            hits=1,
            misses=1,
            coalesced=1,
        )

    run(scenario())


def test_cancelled_evaluation_is_not_cached() -> None:
    async def scenario() -> None:
        cache: AsyncEvaluationCache[str] = AsyncEvaluationCache()
        calls = 0

        async def cancelled() -> str:
            nonlocal calls
            calls += 1
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await cache.get_or_evaluate("config-a", cancelled)
        assert await cache.snapshot() == cache_snapshot(misses=1)

        async def recovered() -> str:
            nonlocal calls
            calls += 1
            return "recovered"

        assert await cache.get_or_evaluate("config-a", recovered) == "recovered"
        assert calls == 2
        assert await cache.snapshot() == cache_snapshot(cached_entries=1, misses=2)

    run(scenario())


def test_bounded_cache_uses_lru_and_traces_ordered_lifecycle_events() -> None:
    async def scenario() -> None:
        events: list[EvaluationCacheTraceEvent] = []
        cache: AsyncEvaluationCache[str] = AsyncEvaluationCache(
            capacity=2,
            trace_callback=events.append,
        )

        async def value(key: str) -> str:
            return key.upper()

        assert await cache.get_or_evaluate("a", lambda: value("a")) == "A"
        assert await cache.get_or_evaluate("b", lambda: value("b")) == "B"
        assert await cache.get_or_evaluate("a", lambda: value("a")) == "A"
        assert await cache.get_or_evaluate("c", lambda: value("c")) == "C"

        # Accessing A made B the least-recently-used successful value.
        assert [event.event_type for event in events] == [
            EvaluationCacheEventType.MISS,
            EvaluationCacheEventType.MISS,
            EvaluationCacheEventType.HIT,
            EvaluationCacheEventType.MISS,
            EvaluationCacheEventType.EVICTED,
        ]
        assert [event.sequence for event in events] == [1, 2, 3, 4, 5]
        assert events[-1].config_hash == "b"
        assert events[-1].snapshot == cache_snapshot(
            capacity=2,
            cached_entries=2,
            hits=1,
            misses=3,
            evictions=1,
        )

        # B was evicted, whereas A remains a hit.
        assert await cache.get_or_evaluate("a", lambda: value("wrong")) == "A"
        assert await cache.get_or_evaluate("b", lambda: value("b")) == "B"
        assert await cache.snapshot() == cache_snapshot(
            capacity=2,
            cached_entries=2,
            hits=2,
            misses=4,
            evictions=2,
        )

    run(scenario())


def test_arguments_are_fail_closed_and_trace_callback_is_observational() -> None:
    with pytest.raises(ValueError, match="capacity"):
        AsyncEvaluationCache[str](capacity=0)
    with pytest.raises(ValueError, match="capacity"):
        AsyncEvaluationCache[str](capacity=True)

    async def scenario() -> None:
        def broken_trace(_event: EvaluationCacheTraceEvent) -> None:
            raise RuntimeError("telemetry outage")

        cache: AsyncEvaluationCache[str] = AsyncEvaluationCache(
            trace_callback=broken_trace
        )

        async def evaluate() -> str:
            return "ok"

        with pytest.raises(ValueError, match="nonempty"):
            await cache.get_or_evaluate("", evaluate)
        with pytest.raises(ValueError, match="nonempty"):
            await cache.get_or_evaluate(1, evaluate)  # type: ignore[arg-type]
        assert await cache.get_or_evaluate("config-a", evaluate) == "ok"

    run(scenario())


def cache_snapshot(
    *,
    capacity: int | None = None,
    cached_entries: int = 0,
    in_flight: int = 0,
    hits: int = 0,
    misses: int = 0,
    coalesced: int = 0,
    evictions: int = 0,
):
    from agent_evolve.application.evaluation_cache import EvaluationCacheSnapshot

    return EvaluationCacheSnapshot(
        capacity=capacity,
        cached_entries=cached_entries,
        in_flight=in_flight,
        hits=hits,
        misses=misses,
        coalesced=coalesced,
        evictions=evictions,
    )
