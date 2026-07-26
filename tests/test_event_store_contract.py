"""Shared EventStore contract and JSONL durability/corruption checks."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from agent_evolve.domain.event import (
    EventCodecError,
    EventEnvelope,
    RunFinished,
    RunStarted,
    event_from_json,
    event_to_json,
)
from agent_evolve.domain.ids import EventId, RunId
from agent_evolve.infrastructure.events import InMemoryEventStore, JsonlEventStore
from agent_evolve.ports.event_store import (
    CorruptEventLogError,
    DuplicateEventIdError,
    EventCausationError,
    EventRunMismatchError,
    EventSequenceError,
    OptimisticConcurrencyError,
)

HASH = "a" * 64


def _event(run: str, event: str, sequence: int, payload, *, offset: int | None = None):
    return EventEnvelope(
        schema_version=1,
        event_id=EventId(event),
        run_id=RunId(run),
        sequence_number=sequence,
        event_type=payload.EVENT_TYPE,
        wall_timestamp_utc=datetime(2026, 7, 13, tzinfo=timezone.utc),
        monotonic_offset_ns=sequence if offset is None else offset,
        correlation_id=None,
        causation_event_id=None,
        payload=payload,
    )


@pytest.fixture(params=["memory", "jsonl"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryEventStore()
    return JsonlEventStore(tmp_path / "events")


def test_store_appends_and_reads_contiguous_stream(store):
    run_id = RunId("run_contract")
    first = _event("run_contract", "event_contract_1", 1, RunStarted(HASH))
    second = _event("run_contract", "event_contract_2", 2, RunFinished("done"))
    store.append(first, expected_previous_sequence=0)
    store.append(second, expected_previous_sequence=1)
    assert tuple(store.read(run_id)) == (first, second)
    assert tuple(store.read(run_id, after_sequence=1)) == (second,)


def test_store_rejects_sequence_gap_and_stale_writer(store):
    first = _event("run_sequence", "event_sequence_1", 1, RunStarted(HASH))
    store.append(first, expected_previous_sequence=0)
    gap = _event("run_sequence", "event_sequence_3", 3, RunFinished("done"))
    with pytest.raises(EventSequenceError, match="must be 2"):
        store.append(gap, expected_previous_sequence=1)

    second = _event("run_sequence", "event_sequence_2", 2, RunFinished("done"))
    with pytest.raises(OptimisticConcurrencyError, match="run is at 1"):
        store.append(second, expected_previous_sequence=0)


def test_store_rejects_duplicate_event_id_even_across_runs(store):
    first = _event("run_duplicate_a", "event_globally_same", 1, RunStarted(HASH))
    second = _event("run_duplicate_b", "event_globally_same", 1, RunStarted(HASH))
    store.append(first, expected_previous_sequence=0)
    with pytest.raises(DuplicateEventIdError, match="Duplicate event ID"):
        store.append(second, expected_previous_sequence=0)


def test_store_requires_causation_to_reference_an_earlier_same_run_event(store):
    first = _event("run_causal", "event_causal_1", 1, RunStarted(HASH))
    dangling = _event("run_causal", "event_causal_2", 2, RunFinished("done"))
    dangling = replace(
        dangling,
        causation_event_id=EventId("event_missing"),
    )
    store.append(first, expected_previous_sequence=0)
    with pytest.raises(EventCausationError, match="must precede"):
        store.append(dangling, expected_previous_sequence=1)

    valid = replace(
        dangling,
        causation_event_id=first.event_id,
    )
    store.append(valid, expected_previous_sequence=1)
    assert tuple(store.read(RunId("run_causal")))[-1] == valid


def test_jsonl_reopens_to_identical_typed_events(tmp_path):
    root = tmp_path / "events"
    store = JsonlEventStore(root)
    event = _event("run_reopen", "event_reopen_1", 1, RunStarted(HASH))
    store.append(event, expected_previous_sequence=0)

    reopened = JsonlEventStore(root)
    assert reopened.read(RunId("run_reopen")) == (event,)
    assert reopened.path_for_run(RunId("run_reopen")).read_bytes().endswith(b"\n")


def test_jsonl_reopen_detects_truncated_final_line(tmp_path):
    root = tmp_path / "events"
    store = JsonlEventStore(root)
    event = _event("run_truncated", "event_truncated_1", 1, RunStarted(HASH))
    store.append(event, expected_previous_sequence=0)
    path = store.path_for_run(RunId("run_truncated"))
    path.write_bytes(path.read_bytes().rstrip(b"\n"))

    with pytest.raises(CorruptEventLogError, match="truncated final line"):
        JsonlEventStore(root)


def test_jsonl_reopen_detects_run_file_mismatch(tmp_path):
    root = tmp_path / "events"
    store = JsonlEventStore(root)
    event = _event("run_actual", "event_actual_1", 1, RunStarted(HASH))
    store.append(event, expected_previous_sequence=0)
    store.path_for_run(RunId("run_actual")).rename(root / "run_wrong.jsonl")

    with pytest.raises(EventRunMismatchError, match="contains run"):
        JsonlEventStore(root)


def test_jsonl_reopen_revalidates_contiguous_sequences(tmp_path):
    root = tmp_path / "events"
    store = JsonlEventStore(root)
    first = _event("run_tampered", "event_tampered_1", 1, RunStarted(HASH))
    second = _event("run_tampered", "event_tampered_2", 2, RunFinished("done"))
    store.append(first, expected_previous_sequence=0)
    store.append(second, expected_previous_sequence=1)
    path = store.path_for_run(RunId("run_tampered"))
    data = path.read_text(encoding="utf-8").replace(
        '"sequence_number":2', '"sequence_number":3'
    )
    path.write_text(data, encoding="utf-8")

    with pytest.raises(CorruptEventLogError, match="Non-contiguous event sequence"):
        JsonlEventStore(root)


def test_jsonl_reopen_rejects_tampered_dangling_causation(tmp_path):
    root = tmp_path / "events"
    store = JsonlEventStore(root)
    first = _event("run_causation_tamper", "event_causation_1", 1, RunStarted(HASH))
    second = replace(
        _event("run_causation_tamper", "event_causation_2", 2, RunFinished("done")),
        causation_event_id=first.event_id,
    )
    store.append(first, expected_previous_sequence=0)
    store.append(second, expected_previous_sequence=1)
    path = store.path_for_run(RunId("run_causation_tamper"))
    data = path.read_text(encoding="utf-8").replace(
        '"causation_event_id":"event_causation_1"',
        '"causation_event_id":"event_missing"',
    )
    path.write_text(data, encoding="utf-8")

    with pytest.raises(CorruptEventLogError, match="Causation event"):
        JsonlEventStore(root)


@pytest.mark.parametrize("malformation", ["codec", "utf8"])
def test_jsonl_malformed_event_errors_do_not_echo_or_retain_raw_content(
    tmp_path,
    malformation,
):
    root = tmp_path / "events"
    store = JsonlEventStore(root)
    event = _event("run_safe_log", "event_safe_log_1", 1, RunStarted(HASH))
    store.append(event, expected_previous_sequence=0)
    path = store.path_for_run(RunId("run_safe_log"))
    raw_secret = "sk-abcdefghijklmnopqrstuv"
    if malformation == "codec":
        data = path.read_text(encoding="utf-8").replace(
            '"event_id":"event_safe_log_1"',
            f'"event_id":"event_{raw_secret}"',
        )
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(f'{{"value":"{raw_secret}"}}'.encode() + b"\xff\n")

    with pytest.raises(CorruptEventLogError) as caught:
        JsonlEventStore(root)

    assert raw_secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "mutation",
    ["event_type", "event_id", "payload", "noncanonical_timezone"],
)
def test_store_revalidates_low_level_mutation_before_persistence(store, mutation):
    run_id = RunId("run_mutation_reject")
    event = _event(
        run_id.value,
        "event_mutation_reject_1",
        1,
        RunStarted(HASH),
    )
    raw_secret = "sk-abcdefghijklmnopqrstuv"
    if mutation == "event_type":
        object.__setattr__(event, "event_type", raw_secret)
    elif mutation == "event_id":
        forged_id = object.__new__(EventId)
        object.__setattr__(forged_id, "value", f"event_{raw_secret}")
        object.__setattr__(event, "event_id", forged_id)
    elif mutation == "payload":
        object.__setattr__(event.payload, "experiment_spec_hash", raw_secret)
    else:
        object.__setattr__(
            event,
            "wall_timestamp_utc",
            datetime(2026, 7, 13, 12, tzinfo=timezone(timedelta(hours=5))),
        )

    with pytest.raises(EventCodecError) as caught:
        store.append(event, expected_previous_sequence=0)

    assert raw_secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert store.read(run_id) == ()


def test_store_snapshot_is_independent_of_caller_mutation_after_append_and_read(store):
    run_id = RunId("run_snapshot_copy")
    event = _event(
        run_id.value,
        "event_snapshot_copy_1",
        1,
        RunStarted(HASH),
    )
    expected = event_from_json(event_to_json(event))
    store.append(event, expected_previous_sequence=0)

    object.__setattr__(event, "event_type", "mutated_after_append")
    object.__setattr__(event.payload, "experiment_spec_hash", "b" * 64)
    first_read = store.read(run_id)
    assert first_read == (expected,)

    object.__setattr__(first_read[0], "event_type", "mutated_after_read")
    object.__setattr__(first_read[0].payload, "experiment_spec_hash", "c" * 64)
    assert store.read(run_id) == (expected,)
