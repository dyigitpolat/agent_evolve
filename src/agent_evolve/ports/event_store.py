"""Append-only event-store port and shared contract errors."""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from agent_evolve.domain.event import EventEnvelope
from agent_evolve.domain.ids import RunId


class EventStoreError(RuntimeError):
    pass


class EventSequenceError(EventStoreError):
    pass


class OptimisticConcurrencyError(EventSequenceError):
    pass


class EventRunMismatchError(EventStoreError):
    pass


class DuplicateEventIdError(EventStoreError):
    pass


class EventCausationError(EventSequenceError):
    """A causation link is dangling, forward-pointing, or cross-run."""


class CorruptEventLogError(EventStoreError):
    pass


@runtime_checkable
class EventStore(Protocol):
    """Store immutable events using optimistic per-run sequencing."""

    def append(
        self,
        event: EventEnvelope,
        *,
        expected_previous_sequence: int,
    ) -> None: ...

    def read(
        self,
        run_id: RunId,
        *,
        after_sequence: int = 0,
    ) -> Iterable[EventEnvelope]: ...
