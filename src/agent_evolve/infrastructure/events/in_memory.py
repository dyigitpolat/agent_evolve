"""Thread-safe in-memory EventStore, primarily for tests and ephemeral runs."""

from __future__ import annotations

import threading
from typing import Dict, List, Tuple

from agent_evolve.domain.event import EventEnvelope, validated_event_snapshot
from agent_evolve.domain.ids import EventId, RunId
from agent_evolve.infrastructure.events._validation import (
    prepare_event_for_append,
    validate_append,
)


class InMemoryEventStore:
    def __init__(self) -> None:
        self._events: Dict[RunId, List[EventEnvelope]] = {}
        self._event_ids: set[EventId] = set()
        self._lock = threading.RLock()

    def append(
        self,
        event: EventEnvelope,
        *,
        expected_previous_sequence: int,
    ) -> None:
        snapshot = prepare_event_for_append(event)
        with self._lock:
            stream = self._events.setdefault(snapshot.run_id, [])
            validate_append(
                stream,
                snapshot,
                expected_previous_sequence=expected_previous_sequence,
                known_event_ids=self._event_ids,
            )
            stream.append(snapshot)
            self._event_ids.add(snapshot.event_id)

    def read(
        self,
        run_id: RunId,
        *,
        after_sequence: int = 0,
    ) -> Tuple[EventEnvelope, ...]:
        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError("after_sequence must be a non-negative integer")
        if type(run_id) is not RunId:
            raise TypeError("run_id must be an exact RunId")
        canonical_run_id = RunId(run_id.value)
        with self._lock:
            return tuple(
                validated_event_snapshot(event)
                for event in self._events.get(canonical_run_id, ())
                if event.sequence_number > after_sequence
            )
