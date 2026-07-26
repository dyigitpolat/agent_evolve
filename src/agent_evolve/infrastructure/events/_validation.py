"""Shared validation for EventStore implementations."""

from __future__ import annotations

from typing import AbstractSet, Sequence

from agent_evolve.domain.event import EventEnvelope, validated_event_snapshot
from agent_evolve.domain.ids import EventId, RunId
from agent_evolve.ports.event_store import (
    DuplicateEventIdError,
    EventCausationError,
    EventRunMismatchError,
    EventSequenceError,
    OptimisticConcurrencyError,
)


def prepare_event_for_append(event: EventEnvelope) -> EventEnvelope:
    """Detach and fully revalidate a caller-owned event before persistence."""

    return validated_event_snapshot(event)


def validate_append(
    existing: Sequence[EventEnvelope],
    event: EventEnvelope,
    *,
    expected_previous_sequence: int,
    known_event_ids: AbstractSet[EventId],
) -> None:
    if (
        type(expected_previous_sequence) is not int
        or expected_previous_sequence < 0
    ):
        raise EventSequenceError("expected_previous_sequence must be non-negative")

    current_sequence = existing[-1].sequence_number if existing else 0
    if expected_previous_sequence != current_sequence:
        raise OptimisticConcurrencyError(
            f"Expected previous sequence {expected_previous_sequence}, "
            f"but run is at {current_sequence}"
        )
    if event.sequence_number != current_sequence + 1:
        raise EventSequenceError(
            f"Event sequence must be {current_sequence + 1}, got {event.sequence_number}"
        )
    if existing and event.run_id != existing[0].run_id:
        raise EventRunMismatchError(
            f"Event run {event.run_id} does not match stream {existing[0].run_id}"
        )
    if event.event_id in known_event_ids:
        raise DuplicateEventIdError(f"Duplicate event ID {event.event_id}")
    if event.causation_event_id is not None and all(
        prior.event_id != event.causation_event_id for prior in existing
    ):
        raise EventCausationError(
            f"Causation event {event.causation_event_id} must precede the event "
            f"in the same run {event.run_id}"
        )
    if existing and event.monotonic_offset_ns < existing[-1].monotonic_offset_ns:
        raise EventSequenceError("monotonic_offset_ns cannot decrease within a run")


def validate_loaded_stream(run_id: RunId, events: Sequence[EventEnvelope]) -> None:
    previous_sequence = 0
    previous_offset = 0
    ids = set()
    for event in events:
        if event.run_id != run_id:
            raise EventRunMismatchError(
                f"Event run {event.run_id} does not match stream {run_id}"
            )
        if event.sequence_number != previous_sequence + 1:
            raise EventSequenceError(
                f"Non-contiguous event sequence for {run_id}: "
                f"expected {previous_sequence + 1}, got {event.sequence_number}"
            )
        if event.event_id in ids:
            raise DuplicateEventIdError(f"Duplicate event ID {event.event_id}")
        if event.causation_event_id is not None and event.causation_event_id not in ids:
            raise EventCausationError(
                f"Causation event {event.causation_event_id} must precede the event "
                f"in the same run {run_id}"
            )
        if previous_sequence and event.monotonic_offset_ns < previous_offset:
            raise EventSequenceError("monotonic_offset_ns cannot decrease within a run")
        ids.add(event.event_id)
        previous_sequence = event.sequence_number
        previous_offset = event.monotonic_offset_ns
