"""Single append-before-publish authority for one run's event sequence."""

from __future__ import annotations

import threading
from typing import Callable, Iterable, Optional, Tuple

from agent_evolve.domain.event import (
    CURRENT_EVENT_SCHEMA_VERSION,
    EventEnvelope,
    EventPayload,
)
from agent_evolve.domain.ids import CorrelationId, EventId, RunId
from agent_evolve.ports.clock import Clock
from agent_evolve.ports.event_store import EventStore, EventStoreError
from agent_evolve.ports.id_factory import IdFactory

EventSink = Callable[[EventEnvelope], None]


class EventAppendObservedError(EventStoreError):
    """Append raised, but the exact event is readable as the stream tail.

    The recorder reconciles its sequence state but does not publish to sinks.
    The caller still receives an error because the adapter could not affirm its
    complete durability contract (for example, a directory fsync may have failed).
    """


class EventAppendReconciliationError(EventStoreError):
    """A failed append could not be reconciled to an unambiguous stream state."""


class EventRecorder:
    """Create, append, then publish immutable events for exactly one run.

    Sink exceptions intentionally propagate, but only after the event is durable.
    This preserves scientific history while making observer failures visible.
    """

    def __init__(
        self,
        *,
        run_id: RunId,
        event_store: EventStore,
        id_factory: IdFactory,
        clock: Clock,
        sinks: Iterable[EventSink] = (),
    ) -> None:
        self.run_id = run_id
        self._store = event_store
        self._ids = id_factory
        self._clock = clock
        self._sinks: Tuple[EventSink, ...] = tuple(sinks)
        self._lock = threading.RLock()

        existing = tuple(event_store.read(run_id))
        self._last_sequence = existing[-1].sequence_number if existing else 0
        self._last_offset = existing[-1].monotonic_offset_ns if existing else -1
        self._clock_origin_ns = clock.monotonic_ns()
        # A reopened recorder cannot reconstruct a previous process's monotonic
        # origin. Continue strictly after the durable offset; M2 resume will also
        # project elapsed wall time across process boundaries.
        self._offset_origin = self._last_offset + 1 if existing else 0

    @property
    def last_sequence(self) -> int:
        with self._lock:
            return self._last_sequence

    def events(self, *, after_sequence: int = 0) -> Tuple[EventEnvelope, ...]:
        return tuple(self._store.read(self.run_id, after_sequence=after_sequence))

    def _reconcile_failed_append(self, event: EventEnvelope) -> bool:
        """Return whether *event* is the exact readable tail after append failed."""

        try:
            observed = tuple(self._store.read(self.run_id))
        except Exception:
            raise EventAppendReconciliationError(
                "failed event append has an unreadable or ambiguous outcome"
            ) from None

        at_sequence = tuple(
            item
            for item in observed
            if item.sequence_number == event.sequence_number
        )
        if not at_sequence:
            if observed and observed[-1].sequence_number >= event.sequence_number:
                raise EventAppendReconciliationError(
                    "failed event append conflicts with the readable stream"
                )
            return False
        if len(at_sequence) != 1 or at_sequence[0] != event or observed[-1] != event:
            raise EventAppendReconciliationError(
                "failed event append conflicts with the readable stream"
            )
        return True

    def record(
        self,
        payload: EventPayload,
        *,
        correlation_id: Optional[CorrelationId] = None,
        causation_event_id: Optional[EventId] = None,
    ) -> EventEnvelope:
        if not isinstance(payload, EventPayload):
            raise TypeError("payload must be an EventPayload")
        with self._lock:
            now_monotonic = self._clock.monotonic_ns()
            delta = now_monotonic - self._clock_origin_ns
            if delta < 0:
                raise RuntimeError("injected monotonic clock moved backwards")
            offset = self._offset_origin + delta
            sequence = self._last_sequence + 1
            event = EventEnvelope(
                schema_version=CURRENT_EVENT_SCHEMA_VERSION,
                event_id=self._ids.new_event_id(),
                run_id=self.run_id,
                sequence_number=sequence,
                event_type=payload.EVENT_TYPE,
                wall_timestamp_utc=self._clock.utc_now(),
                monotonic_offset_ns=offset,
                correlation_id=correlation_id,
                causation_event_id=causation_event_id,
                payload=payload,
            )
            try:
                self._store.append(
                    event,
                    expected_previous_sequence=self._last_sequence,
                )
            except Exception:
                if not self._reconcile_failed_append(event):
                    raise
                # Make a subsequent operation continue at the observed stream
                # tail instead of retrying/reusing an already-recorded sequence.
                self._last_sequence = sequence
                self._last_offset = offset
                raise EventAppendObservedError(
                    "event append reported failure after the exact event became readable; "
                    "observer sinks were not invoked"
                ) from None
            # Advance before observers run: a failing observer cannot make a later
            # append reuse an already-durable sequence number.
            self._last_sequence = sequence
            self._last_offset = offset
            for sink in self._sinks:
                sink(event)
            return event
