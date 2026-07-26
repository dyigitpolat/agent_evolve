"""Cancellation-safe single-flight caching for expensive async evaluations.

The caller owns canonical configuration hashing.  This module never serializes or
normalizes configurations, so cache identity cannot silently lose type or value
information.  Every normally returned value is cacheable, including a domain's
ordinary invalid-result value; raised exceptions and cancellations are not.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Generic, Optional, TypeVar


ValueT = TypeVar("ValueT")


class EvaluationCacheEventType(str, Enum):
    """The deliberately small observable lifecycle of the cache."""

    HIT = "hit"
    MISS = "miss"
    COALESCED = "coalesced"
    EVICTED = "evicted"


@dataclass(frozen=True, slots=True)
class EvaluationCacheSnapshot:
    """Immutable cache occupancy and cumulative request counters."""

    capacity: Optional[int]
    cached_entries: int
    in_flight: int
    hits: int
    misses: int
    coalesced: int
    evictions: int


@dataclass(frozen=True, slots=True)
class EvaluationCacheTraceEvent:
    """One ordered cache lifecycle event with its post-event snapshot."""

    sequence: int
    event_type: EvaluationCacheEventType
    config_hash: str
    snapshot: EvaluationCacheSnapshot


EvaluationFactory = Callable[[], Awaitable[ValueT]]
EvaluationCacheTraceCallback = Callable[[EvaluationCacheTraceEvent], None]


class AsyncEvaluationCache(Generic[ValueT]):
    """Coalesce identical evaluations and optionally retain results in an LRU.

    ``get_or_evaluate`` shields shared work from each individual awaiter.  Thus,
    cancelling one awaiter does not cancel an evaluation still needed by another
    awaiter (or prevent its result from being cached).  If the evaluation itself
    raises or is cancelled, its in-flight entry is removed and a later request is
    a fresh miss.

    Trace callbacks are synchronous observation hooks.  They run in serialized
    event order, and callback failures are isolated from cache semantics.
    """

    def __init__(
        self,
        *,
        capacity: Optional[int] = None,
        trace_callback: Optional[EvaluationCacheTraceCallback] = None,
    ) -> None:
        if capacity is not None and (type(capacity) is not int or capacity < 1):
            raise ValueError("capacity must be None or a positive integer")
        if trace_callback is not None and not callable(trace_callback):
            raise TypeError("trace_callback must be callable or None")

        self._capacity = capacity
        self._trace_callback = trace_callback
        self._values: "OrderedDict[str, ValueT]" = OrderedDict()
        self._in_flight: dict[str, "asyncio.Task[ValueT]"] = {}
        self._lock = asyncio.Lock()

        self._hits = 0
        self._misses = 0
        self._coalesced = 0
        self._evictions = 0
        self._event_sequence = 0

    async def get_or_evaluate(
        self,
        config_hash: str,
        evaluate: EvaluationFactory[ValueT],
    ) -> ValueT:
        """Return a cached value or await one shared evaluation for ``config_hash``."""

        self._validate_config_hash(config_hash)
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")

        async with self._lock:
            if config_hash in self._values:
                value = self._values.pop(config_hash)
                self._values[config_hash] = value
                self._hits += 1
                self._emit_locked(EvaluationCacheEventType.HIT, config_hash)
                return value

            task = self._in_flight.get(config_hash)
            if task is None:
                task = asyncio.create_task(
                    self._evaluate_and_store(config_hash, evaluate)
                )
                self._in_flight[config_hash] = task
                self._misses += 1
                self._emit_locked(EvaluationCacheEventType.MISS, config_hash)
            else:
                self._coalesced += 1
                self._emit_locked(EvaluationCacheEventType.COALESCED, config_hash)

        # An awaiter's cancellation must not propagate into shared evaluation work.
        return await asyncio.shield(task)

    async def snapshot(self) -> EvaluationCacheSnapshot:
        """Return an atomic occupancy and counter snapshot."""

        async with self._lock:
            return self._snapshot_locked()

    async def _evaluate_and_store(
        self,
        config_hash: str,
        evaluate: EvaluationFactory[ValueT],
    ) -> ValueT:
        current_task = asyncio.current_task()
        if current_task is None:  # pragma: no cover - guaranteed by create_task
            raise RuntimeError("evaluation did not run inside an asyncio task")

        try:
            value = await evaluate()
        except BaseException:
            async with self._lock:
                if self._in_flight.get(config_hash) is current_task:
                    del self._in_flight[config_hash]
            raise

        async with self._lock:
            if self._in_flight.get(config_hash) is not current_task:
                raise RuntimeError("single-flight evaluation lost its ownership entry")
            del self._in_flight[config_hash]
            self._values[config_hash] = value
            self._values.move_to_end(config_hash)

            if self._capacity is not None and len(self._values) > self._capacity:
                evicted_hash, _ = self._values.popitem(last=False)
                self._evictions += 1
                self._emit_locked(EvaluationCacheEventType.EVICTED, evicted_hash)

        return value

    @staticmethod
    def _validate_config_hash(config_hash: str) -> None:
        if type(config_hash) is not str or not config_hash:
            raise ValueError("config_hash must be a nonempty string")

    def _snapshot_locked(self) -> EvaluationCacheSnapshot:
        return EvaluationCacheSnapshot(
            capacity=self._capacity,
            cached_entries=len(self._values),
            in_flight=len(self._in_flight),
            hits=self._hits,
            misses=self._misses,
            coalesced=self._coalesced,
            evictions=self._evictions,
        )

    def _emit_locked(
        self,
        event_type: EvaluationCacheEventType,
        config_hash: str,
    ) -> None:
        callback = self._trace_callback
        if callback is None:
            return
        self._event_sequence += 1
        event = EvaluationCacheTraceEvent(
            sequence=self._event_sequence,
            event_type=event_type,
            config_hash=config_hash,
            snapshot=self._snapshot_locked(),
        )
        try:
            callback(event)
        except Exception:
            # Observation must not turn a valid evaluation into a failed one.
            return
