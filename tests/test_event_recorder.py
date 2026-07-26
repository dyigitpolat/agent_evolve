"""EventRecorder ordering, clocks, reopen behavior, and sink isolation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_evolve.application.event_recorder import (
    EventAppendObservedError,
    EventAppendReconciliationError,
    EventRecorder,
)
from agent_evolve.domain.event import RunFinished, RunStarted
from agent_evolve.infrastructure.clock import FakeClock
from agent_evolve.infrastructure.events import InMemoryEventStore
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.event_store import EventStoreError

HASH = "b" * 64


def test_recorder_assigns_contiguous_ids_and_deterministic_times():
    store = InMemoryEventStore()
    ids = DeterministicIdFactory("recorder")
    clock = FakeClock(datetime(2026, 7, 13, tzinfo=timezone.utc), start_monotonic_ns=50)
    run_id = ids.new_run_id()
    recorder = EventRecorder(run_id=run_id, event_store=store, id_factory=ids, clock=clock)

    first = recorder.record(RunStarted(HASH))
    clock.advance_ns(12)
    second = recorder.record(RunFinished("done"), causation_event_id=first.event_id)

    assert [event.sequence_number for event in store.read(run_id)] == [1, 2]
    assert first.monotonic_offset_ns == 0
    assert second.monotonic_offset_ns == 12
    assert second.causation_event_id == first.event_id


def test_recorder_appends_before_publishing_to_sink():
    store = InMemoryEventStore()
    ids = DeterministicIdFactory("publish")
    clock = FakeClock()
    run_id = ids.new_run_id()
    observed = []

    def sink(event):
        assert store.read(run_id)[-1] == event
        observed.append(event)

    recorder = EventRecorder(
        run_id=run_id,
        event_store=store,
        id_factory=ids,
        clock=clock,
        sinks=(sink,),
    )
    event = recorder.record(RunStarted(HASH))
    assert observed == [event]


def test_sink_failure_does_not_erase_event_or_reuse_sequence():
    store = InMemoryEventStore()
    ids = DeterministicIdFactory("sink_failure")
    clock = FakeClock()
    run_id = ids.new_run_id()

    def broken_sink(event):
        raise RuntimeError("observer failed")

    recorder = EventRecorder(
        run_id=run_id,
        event_store=store,
        id_factory=ids,
        clock=clock,
        sinks=(broken_sink,),
    )
    with pytest.raises(RuntimeError, match="observer failed"):
        recorder.record(RunStarted(HASH))
    assert recorder.last_sequence == 1
    assert len(store.read(run_id)) == 1

    with pytest.raises(RuntimeError, match="observer failed"):
        recorder.record(RunFinished("still persisted"))
    assert [event.sequence_number for event in store.read(run_id)] == [1, 2]


def test_store_failure_prevents_publication_and_sequence_advance():
    class BrokenStore(InMemoryEventStore):
        def append(self, event, *, expected_previous_sequence):
            raise EventStoreError("disk unavailable")

    store = BrokenStore()
    ids = DeterministicIdFactory("store_failure")
    run_id = ids.new_run_id()
    observed = []
    recorder = EventRecorder(
        run_id=run_id,
        event_store=store,
        id_factory=ids,
        clock=FakeClock(),
        sinks=(observed.append,),
    )
    with pytest.raises(EventStoreError, match="disk unavailable"):
        recorder.record(RunStarted(HASH))
    assert recorder.last_sequence == 0
    assert observed == []


def test_commit_then_raise_is_reconciled_without_reusing_sequence_or_publishing():
    class CommitThenRaiseStore(InMemoryEventStore):
        def __init__(self):
            super().__init__()
            self.fail_once = True

        def append(self, event, *, expected_previous_sequence):
            super().append(event, expected_previous_sequence=expected_previous_sequence)
            if self.fail_once:
                self.fail_once = False
                raise EventStoreError("directory fsync failed after append")

    store = CommitThenRaiseStore()
    ids = DeterministicIdFactory("commit_then_raise")
    run_id = ids.new_run_id()
    observed = []
    recorder = EventRecorder(
        run_id=run_id,
        event_store=store,
        id_factory=ids,
        clock=FakeClock(),
        sinks=(observed.append,),
    )

    with pytest.raises(EventAppendObservedError, match="became readable"):
        recorder.record(RunStarted(HASH))
    assert recorder.last_sequence == 1
    assert len(store.read(run_id)) == 1
    assert observed == []

    second = recorder.record(RunFinished("continued safely"))
    assert second.sequence_number == 2
    assert [event.sequence_number for event in store.read(run_id)] == [1, 2]
    assert observed == [second]


def test_failed_append_with_unreadable_reconciliation_is_explicitly_ambiguous():
    class UnreadableBrokenStore(InMemoryEventStore):
        def __init__(self):
            super().__init__()
            self.initialized = False

        def read(self, run_id, *, after_sequence=0):
            if not self.initialized:
                self.initialized = True
                return ()
            raise EventStoreError("event stream unreadable")

        def append(self, event, *, expected_previous_sequence):
            raise EventStoreError("append outcome unknown")

    store = UnreadableBrokenStore()
    ids = DeterministicIdFactory("ambiguous_append")
    run_id = ids.new_run_id()
    recorder = EventRecorder(
        run_id=run_id,
        event_store=store,
        id_factory=ids,
        clock=FakeClock(),
    )
    with pytest.raises(EventAppendReconciliationError, match="ambiguous outcome"):
        recorder.record(RunStarted(HASH))
    assert recorder.last_sequence == 0


def test_reopened_recorder_continues_sequence_and_monotonic_offset():
    store = InMemoryEventStore()
    ids = DeterministicIdFactory("resume")
    run_id = ids.new_run_id()
    first_clock = FakeClock(start_monotonic_ns=100)
    first = EventRecorder(
        run_id=run_id, event_store=store, id_factory=ids, clock=first_clock
    ).record(RunStarted(HASH))

    second_clock = FakeClock(start_monotonic_ns=0)
    reopened = EventRecorder(
        run_id=run_id, event_store=store, id_factory=ids, clock=second_clock
    )
    second = reopened.record(RunFinished("reopened"))
    assert second.sequence_number == 2
    assert second.monotonic_offset_ns > first.monotonic_offset_ns
