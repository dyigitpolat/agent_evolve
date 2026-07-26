"""Redaction journal ordering, crash consistency, and leak-resistance tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agent_evolve.application.artifact_journal import (
    ArtifactJournal,
    ArtifactPostWriteVerificationError,
    ArtifactPreparationError,
    ArtifactSizeLimitError,
)
from agent_evolve.application.artifact_replay import verify_artifact_journal
from agent_evolve.application.event_recorder import EventAppendObservedError, EventRecorder
from agent_evolve.domain.artifact import ArtifactRef, ArtifactRole, artifact_ref_for_bytes
from agent_evolve.domain.event import ArtifactRegistered, event_to_json
from agent_evolve.infrastructure.artifacts import InMemoryArtifactStore
from agent_evolve.infrastructure.clock import FakeClock
from agent_evolve.infrastructure.events import InMemoryEventStore
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.infrastructure.sanitization import (
    StrictJsonSanitizer,
    TopLevelAllowlistMinimizer,
)
from agent_evolve.ports.artifact_store import (
    JSON_MEDIA_TYPE,
    ArtifactStoreError,
    canonical_json_bytes,
)
from agent_evolve.ports.event_store import EventStoreError

RAW_SECRET = "openrouter-runtime-secret-987654"


def _recorder(event_store=None, *, sinks=()):
    event_store = event_store or InMemoryEventStore()
    ids = DeterministicIdFactory("journal")
    run_id = ids.new_run_id()
    return (
        EventRecorder(
            run_id=run_id,
            event_store=event_store,
            id_factory=ids,
            clock=FakeClock(),
            sinks=sinks,
        ),
        event_store,
        run_id,
    )


def _journal(
    store,
    recorder,
    *,
    max_size_bytes=4096,
    hook=None,
    sanitizer=None,
    minimizer=None,
):
    return ArtifactJournal(
        artifact_store=store,
        event_recorder=recorder,
        minimizer=minimizer
        or TopLevelAllowlistMinimizer(
            {ArtifactRole.DIAGNOSTICS: {"message", "api_key", "extra"}}
        ),
        sanitizer=sanitizer
        or StrictJsonSanitizer(exact_secret_values=(RAW_SECRET,)),
        max_size_bytes=max_size_bytes,
        after_artifact_put=hook,
    )


def test_journal_orders_redaction_put_full_verification_then_event_append():
    calls = []

    class TrackingMinimizer:
        policy_id = "tracking-minimizer"
        policy_version = "1"
        policy_config_sha256 = "1" * 64

        def minimize_json(self, value, *, role):
            calls.append("minimize")
            return {"message": value["message"], "api_key": value["api_key"]}

    class TrackingSanitizer(StrictJsonSanitizer):
        def sanitize_json(self, value, *, role):
            calls.append("sanitize")
            return super().sanitize_json(value, role=role)

    class TrackingArtifactStore(InMemoryArtifactStore):
        def put_bytes(self, content, *, media_type):
            calls.append("put")
            assert RAW_SECRET.encode() not in content
            return super().put_bytes(content, media_type=media_type)

        def stat(self, artifact_id):
            calls.append("stat")
            return super().stat(artifact_id)

        def read_bytes(self, artifact_id, *, expected_media_type=None):
            calls.append("read")
            return super().read_bytes(
                artifact_id,
                expected_media_type=expected_media_type,
            )

    class TrackingEventStore(InMemoryEventStore):
        def append(self, event, *, expected_previous_sequence):
            calls.append("append_event")
            return super().append(
                event,
                expected_previous_sequence=expected_previous_sequence,
            )

    event_store = TrackingEventStore()
    recorder, _, run_id = _recorder(event_store)
    store = TrackingArtifactStore()
    journal = _journal(
        store,
        recorder,
        minimizer=TrackingMinimizer(),
        sanitizer=TrackingSanitizer(exact_secret_values=(RAW_SECRET,)),
    )

    result = journal.register_json(
        {"message": f"token={RAW_SECRET}", "api_key": RAW_SECRET, "drop": RAW_SECRET},
        role=ArtifactRole.DIAGNOSTICS,
    )

    assert calls == ["minimize", "sanitize", "put", "stat", "read", "append_event"]
    content = store.read_bytes(result.artifact_ref.artifact_id)
    assert RAW_SECRET.encode() not in content
    assert content == canonical_json_bytes(
        {"api_key": "[REDACTED]", "message": "[REDACTED]"}
    )
    event = event_store.read(run_id)[0]
    assert event == result.event
    assert isinstance(event.payload, ArtifactRegistered)
    assert event.payload.artifact_ref == result.artifact_ref
    assert event.payload.role is ArtifactRole.DIAGNOSTICS
    assert event.payload.minimization_policy_id == "tracking-minimizer"
    assert event.payload.minimization_policy_version == "1"
    assert event.payload.minimization_policy_config_sha256 == "1" * 64
    assert event.payload.sanitization_policy_id == "strict-json-redaction"
    assert event.payload.sanitization_policy_version == "1"
    assert RAW_SECRET not in event_to_json(event)


def test_store_failure_emits_no_event_and_contains_no_raw_secret():
    class BrokenStore(InMemoryArtifactStore):
        def put_bytes(self, content, *, media_type):
            assert RAW_SECRET.encode() not in content
            raise ArtifactStoreError("durable store unavailable")

    recorder, event_store, run_id = _recorder()
    with pytest.raises(ArtifactStoreError, match="unavailable") as caught:
        _journal(BrokenStore(), recorder).register_json(
            {"message": RAW_SECRET, "api_key": RAW_SECRET},
            role=ArtifactRole.DIAGNOSTICS,
        )
    assert event_store.read(run_id) == ()
    assert RAW_SECRET not in str(caught.value)


def test_event_append_failure_leaves_verified_store_only_orphan():
    class BrokenEventStore(InMemoryEventStore):
        def append(self, event, *, expected_previous_sequence):
            raise EventStoreError("event disk unavailable")

    store = InMemoryArtifactStore()
    recorder, event_store, run_id = _recorder(BrokenEventStore())
    expected_bytes = canonical_json_bytes({"api_key": "[REDACTED]", "message": "safe"})
    expected_ref = artifact_ref_for_bytes(expected_bytes, media_type=JSON_MEDIA_TYPE)

    with pytest.raises(EventStoreError, match="event disk unavailable"):
        _journal(store, recorder).register_json(
            {"message": "safe", "api_key": RAW_SECRET},
            role=ArtifactRole.DIAGNOSTICS,
        )
    assert store.stat(expected_ref.artifact_id) == expected_ref
    assert event_store.read(run_id) == ()


def test_commit_then_raise_append_leaves_a_replayable_registration_orphan():
    class CommitThenRaiseEventStore(InMemoryEventStore):
        def append(self, event, *, expected_previous_sequence):
            super().append(event, expected_previous_sequence=expected_previous_sequence)
            raise EventStoreError("post-commit durability step failed")

    artifact_store = InMemoryArtifactStore()
    event_store = CommitThenRaiseEventStore()
    recorder, _, run_id = _recorder(event_store)

    with pytest.raises(EventAppendObservedError, match="became readable"):
        _journal(artifact_store, recorder).register_json(
            {"message": "safe", "api_key": RAW_SECRET},
            role=ArtifactRole.DIAGNOSTICS,
        )

    assert recorder.last_sequence == 1
    events = event_store.read(run_id)
    assert len(events) == 1
    report = verify_artifact_journal(events, artifact_store=artifact_store)
    assert report.registration_event_count == 1
    assert len(report.orphan_registrations) == 1


def test_post_put_crash_hook_leaves_orphan_before_stat_or_event():
    calls = []

    class TrackingStore(InMemoryArtifactStore):
        def put_bytes(self, content, *, media_type):
            calls.append("put")
            return super().put_bytes(content, media_type=media_type)

        def stat(self, artifact_id):
            calls.append("stat")
            return super().stat(artifact_id)

    store = TrackingStore()
    recorder, event_store, run_id = _recorder()
    hook_refs = []

    def crash(ref):
        calls.append("crash")
        hook_refs.append(ref)
        raise RuntimeError("simulated process crash")

    with pytest.raises(RuntimeError, match="simulated process crash"):
        _journal(store, recorder, hook=crash).register_json(
            {"message": "safe", "api_key": RAW_SECRET},
            role=ArtifactRole.DIAGNOSTICS,
        )
    assert calls == ["put", "crash"]
    assert event_store.read(run_id) == ()
    # Use the base implementation so the assertion itself does not alter calls.
    assert InMemoryArtifactStore.stat(store, hook_refs[0].artifact_id) == hook_refs[0]


def test_sink_failure_propagates_after_registration_is_durable():
    def broken_sink(event):
        raise RuntimeError("observer failed")

    store = InMemoryArtifactStore()
    recorder, event_store, run_id = _recorder(sinks=(broken_sink,))
    with pytest.raises(RuntimeError, match="observer failed"):
        _journal(store, recorder).register_json(
            {"message": "safe", "api_key": RAW_SECRET},
            role=ArtifactRole.DIAGNOSTICS,
        )

    events = event_store.read(run_id)
    assert len(events) == 1
    assert isinstance(events[0].payload, ArtifactRegistered)
    assert store.stat(events[0].payload.artifact_id) == events[0].payload.artifact_ref


def test_size_rejection_happens_before_any_store_call():
    class ForbiddenStore(InMemoryArtifactStore):
        def put_bytes(self, content, *, media_type):
            raise AssertionError("put must not be called")

    recorder, event_store, run_id = _recorder()
    with pytest.raises(ArtifactSizeLimitError, match="size limit"):
        _journal(ForbiddenStore(), recorder, max_size_bytes=2).register_json(
            {"message": "safe", "api_key": RAW_SECRET},
            role=ArtifactRole.DIAGNOSTICS,
        )
    assert event_store.read(run_id) == ()


def test_policy_failures_are_wrapped_without_raw_values_or_chained_exceptions():
    class LeakyMinimizer:
        def minimize_json(self, value, *, role):
            raise ValueError(RAW_SECRET)

    recorder, event_store, run_id = _recorder()
    with pytest.raises(ArtifactPreparationError) as caught:
        _journal(
            InMemoryArtifactStore(),
            recorder,
            minimizer=LeakyMinimizer(),
        ).register_json(
            {"message": RAW_SECRET},
            role=ArtifactRole.DIAGNOSTICS,
        )
    assert str(caught.value) == (
        "artifact preparation failed under the configured safety policies"
    )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert RAW_SECRET not in str(caught.value)
    assert event_store.read(run_id) == ()


@pytest.mark.parametrize("tamper", ["return", "stat", "read"])
def test_post_write_verification_rejects_full_ref_or_payload_tampering(tamper):
    class LyingStore(InMemoryArtifactStore):
        def put_bytes(self, content, *, media_type):
            ref = super().put_bytes(content, media_type=media_type)
            if tamper == "return":
                return replace(ref, media_type="application/x-tampered")
            return ref

        def stat(self, artifact_id):
            ref = super().stat(artifact_id)
            if tamper == "stat":
                return replace(ref, sha256_hex="f" * 64)
            return ref

        def read_bytes(self, artifact_id, *, expected_media_type=None):
            content = super().read_bytes(
                artifact_id,
                expected_media_type=expected_media_type,
            )
            return b"{}" if tamper == "read" else content

    recorder, event_store, run_id = _recorder()
    with pytest.raises(ArtifactPostWriteVerificationError):
        _journal(LyingStore(), recorder).register_json(
            {"message": "safe", "api_key": RAW_SECRET},
            role=ArtifactRole.DIAGNOSTICS,
        )
    assert event_store.read(run_id) == ()
