"""Artifact registration event codec and fail-closed replay contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from agent_evolve.application.artifact_journal import ArtifactJournal
from agent_evolve.application.artifact_replay import (
    ArtifactRegistrationIntegrityError,
    ArtifactStreamOrderError,
    DanglingArtifactReferenceError,
    IncompatibleArtifactRoleError,
    validate_artifact_reference_schema,
    verify_artifact_journal,
)
import agent_evolve.application.artifact_replay as replay_module
from agent_evolve.application.event_recorder import EventRecorder
from agent_evolve.domain.artifact import ArtifactRole
from agent_evolve.domain.event import (
    ArtifactRegistered,
    CandidateProposed,
    LLMCallCompleted,
    RunStarted,
    event_from_json,
    event_to_json,
)
from agent_evolve.domain.ids import CandidateId, GenerationId, LLMCallId
from agent_evolve.infrastructure.artifacts import InMemoryArtifactStore
from agent_evolve.infrastructure.clock import FakeClock
from agent_evolve.infrastructure.events import InMemoryEventStore
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.infrastructure.sanitization import (
    StrictJsonSanitizer,
    TopLevelAllowlistMinimizer,
)
from agent_evolve.ports.artifact_store import JSON_MEDIA_TYPE, canonical_json_bytes


def _setup(namespace="replay"):
    event_store = InMemoryEventStore()
    artifact_store = InMemoryArtifactStore()
    ids = DeterministicIdFactory(namespace)
    run_id = ids.new_run_id()
    recorder = EventRecorder(
        run_id=run_id,
        event_store=event_store,
        id_factory=ids,
        clock=FakeClock(),
    )
    minimizer = TopLevelAllowlistMinimizer(
        {
            ArtifactRole.RUN_MANIFEST: {"manifest"},
            ArtifactRole.CANDIDATE_CONFIGURATION: {"configuration"},
            ArtifactRole.LLM_RESPONSE: {"response"},
            ArtifactRole.DIAGNOSTICS: {"diagnostics"},
        }
    )
    journal = ArtifactJournal(
        artifact_store=artifact_store,
        event_recorder=recorder,
        minimizer=minimizer,
        sanitizer=StrictJsonSanitizer(),
        max_size_bytes=4096,
    )
    return artifact_store, event_store, recorder, journal, run_id


def _registration_for(ref, role, *, media_type=None, sha256_hex=None):
    return ArtifactRegistered(
        artifact_id=ref.artifact_id,
        sha256_hex=sha256_hex or ref.sha256_hex,
        size_bytes=ref.size_bytes,
        media_type=media_type or ref.media_type,
        role=role,
        minimization_policy_id="top-level-allowlist",
        minimization_policy_version="1",
        minimization_policy_config_sha256="a" * 64,
        sanitization_policy_id="strict-json-redaction",
        sanitization_policy_version="1",
    )


def _candidate(ref):
    return CandidateProposed(
        candidate_id=CandidateId("candidate_replay"),
        generation_id=GenerationId("generation_replay"),
        content_hash="c" * 64,
        proposal_index=0,
        configuration_artifact_id=ref.artifact_id,
    )


def test_registration_event_is_frozen_typed_and_canonical_json_round_trips():
    artifact_store, event_store, recorder, journal, run_id = _setup()
    result = journal.register_json(
        {"diagnostics": {"message": "safe"}},
        role=ArtifactRole.DIAGNOSTICS,
    )
    payload = result.event.payload
    encoded_once = event_to_json(result.event)
    encoded_twice = event_to_json(result.event)

    assert encoded_once == encoded_twice
    assert event_from_json(encoded_once) == result.event
    assert event_from_json(encoded_once).payload.role is ArtifactRole.DIAGNOSTICS
    assert payload.artifact_ref == result.artifact_ref
    with pytest.raises(FrozenInstanceError):
        payload.media_type = "application/x-mutated"
    with pytest.raises(ValueError, match="storage-safe"):
        ArtifactRegistered(
            artifact_id=result.artifact_ref.artifact_id,
            sha256_hex=result.artifact_ref.sha256_hex,
            size_bytes=result.artifact_ref.size_bytes,
            media_type=result.artifact_ref.media_type,
            role=ArtifactRole.DIAGNOSTICS,
            minimization_policy_id="top-level-allowlist",
            minimization_policy_version="1",
            minimization_policy_config_sha256="a" * 64,
            sanitization_policy_id="contains spaces",
            sanitization_policy_version="1",
        )
    with pytest.raises(ValueError, match="non-secret"):
        ArtifactRegistered(
            artifact_id=result.artifact_ref.artifact_id,
            sha256_hex=result.artifact_ref.sha256_hex,
            size_bytes=result.artifact_ref.size_bytes,
            media_type=result.artifact_ref.media_type,
            role=ArtifactRole.DIAGNOSTICS,
            minimization_policy_id="prefix_sk-abcdefghijklmnopqrstuv",
            minimization_policy_version="1",
            minimization_policy_config_sha256="a" * 64,
            sanitization_policy_id="strict-json-redaction",
            sanitization_policy_version="1",
        )


def test_complete_event_vocabulary_role_map_is_validated_fail_closed(monkeypatch):
    validate_artifact_reference_schema()
    monkeypatch.delitem(
        replay_module._REFERENCE_ROLES[CandidateProposed],
        "configuration_artifact_id",
    )
    with pytest.raises(
        replay_module.ArtifactReferenceSchemaError,
        match="do not match",
    ):
        validate_artifact_reference_schema()


def test_replay_accepts_preceding_compatible_references_and_registration_orphans():
    artifact_store, event_store, recorder, journal, run_id = _setup()
    config = journal.register_json(
        {"configuration": {"x": 1}},
        role=ArtifactRole.CANDIDATE_CONFIGURATION,
    )
    recorder.record(_candidate(config.artifact_ref))
    orphan = journal.register_json(
        {"diagnostics": {"message": "never referenced"}},
        role=ArtifactRole.DIAGNOSTICS,
    )
    events = event_store.read(run_id)

    first = verify_artifact_journal(events, artifact_store=artifact_store)
    second = verify_artifact_journal(tuple(events), artifact_store=artifact_store)

    assert first == second
    assert first.run_id == run_id
    assert first.event_count == 3
    assert first.registration_event_count == 2
    assert first.semantic_reference_count == 1
    assert first.unique_registration_count == 2
    assert first.unique_referenced_registration_count == 1
    assert len(first.orphan_registrations) == 1
    assert first.orphan_registrations[0].artifact_id == orphan.artifact_ref.artifact_id
    assert first.orphan_registrations[0].role is ArtifactRole.DIAGNOSTICS


def test_replay_allows_store_only_orphans_because_store_enumeration_is_out_of_scope():
    artifact_store, event_store, recorder, journal, run_id = _setup()
    artifact_store.put_bytes(canonical_json_bytes({"orphan": True}), media_type=JSON_MEDIA_TYPE)

    report = verify_artifact_journal((), artifact_store=artifact_store)

    assert report.event_count == 0
    assert report.run_id is None
    assert report.orphan_registrations == ()


def test_replay_rejects_dangling_reference():
    artifact_store, event_store, recorder, journal, run_id = _setup()
    ref = artifact_store.put_bytes(
        canonical_json_bytes({"configuration": {"x": 1}}),
        media_type=JSON_MEDIA_TYPE,
    )
    recorder.record(_candidate(ref))

    with pytest.raises(DanglingArtifactReferenceError, match="no preceding"):
        verify_artifact_journal(
            event_store.read(run_id),
            artifact_store=artifact_store,
        )


def test_replay_rejects_late_registration_at_the_reference_boundary():
    artifact_store, event_store, recorder, journal, run_id = _setup()
    ref = artifact_store.put_bytes(
        canonical_json_bytes({"configuration": {"x": 1}}),
        media_type=JSON_MEDIA_TYPE,
    )
    recorder.record(_candidate(ref))
    recorder.record(_registration_for(ref, ArtifactRole.CANDIDATE_CONFIGURATION))

    with pytest.raises(DanglingArtifactReferenceError, match="no preceding"):
        verify_artifact_journal(
            event_store.read(run_id),
            artifact_store=artifact_store,
        )


def test_replay_rejects_role_substitution_for_same_artifact_id():
    artifact_store, event_store, recorder, journal, run_id = _setup()
    ref = artifact_store.put_bytes(
        canonical_json_bytes({"configuration": {"x": 1}}),
        media_type=JSON_MEDIA_TYPE,
    )
    recorder.record(_registration_for(ref, ArtifactRole.DIAGNOSTICS))
    recorder.record(_candidate(ref))

    with pytest.raises(IncompatibleArtifactRoleError, match="compatible role"):
        verify_artifact_journal(
            event_store.read(run_id),
            artifact_store=artifact_store,
        )


@pytest.mark.parametrize("tamper", ["sha256", "media_type"])
def test_replay_rejects_registration_full_ref_and_media_tampering(tamper):
    artifact_store, event_store, recorder, journal, run_id = _setup()
    ref = artifact_store.put_bytes(
        canonical_json_bytes({"diagnostics": "safe"}),
        media_type=JSON_MEDIA_TYPE,
    )
    kwargs = {
        "sha256_hex": "f" * 64 if tamper == "sha256" else None,
        "media_type": "application/x-tampered" if tamper == "media_type" else None,
    }
    recorder.record(_registration_for(ref, ArtifactRole.DIAGNOSTICS, **kwargs))

    with pytest.raises(ArtifactRegistrationIntegrityError):
        verify_artifact_journal(
            event_store.read(run_id),
            artifact_store=artifact_store,
        )


@pytest.mark.parametrize("tamper", ["stat", "read"])
def test_replay_independently_recomputes_full_ref_from_store_bytes(tamper):
    artifact_store, event_store, recorder, journal, run_id = _setup()
    registration = journal.register_json(
        {"diagnostics": "safe"},
        role=ArtifactRole.DIAGNOSTICS,
    )

    class LyingReadStore:
        def put_bytes(self, content, *, media_type):
            raise AssertionError("replay is read-only")

        def stat(self, artifact_id):
            ref = artifact_store.stat(artifact_id)
            return replace(ref, size_bytes=ref.size_bytes + 1) if tamper == "stat" else ref

        def read_bytes(self, artifact_id, *, expected_media_type=None):
            content = artifact_store.read_bytes(
                artifact_id,
                expected_media_type=expected_media_type,
            )
            return b"{}" if tamper == "read" else content

    with pytest.raises(ArtifactRegistrationIntegrityError):
        verify_artifact_journal(
            event_store.read(run_id),
            artifact_store=LyingReadStore(),
        )


@pytest.mark.parametrize("failure_boundary", ["stat", "read", "decode"])
def test_initial_registration_verification_drops_private_exception_context(
    failure_boundary,
    monkeypatch,
):
    artifact_store, event_store, recorder, journal, run_id = _setup()
    journal.register_json(
        {"diagnostics": "safe"},
        role=ArtifactRole.DIAGNOSTICS,
    )
    raw_secret = "registration-adapter-private-secret"

    class ExplodingStore:
        def stat(self, artifact_id):
            if failure_boundary == "stat":
                raise ValueError(raw_secret)
            return artifact_store.stat(artifact_id)

        def read_bytes(self, artifact_id, *, expected_media_type=None):
            if failure_boundary == "read":
                raise ValueError(raw_secret)
            return artifact_store.read_bytes(
                artifact_id,
                expected_media_type=expected_media_type,
            )

    if failure_boundary == "decode":
        def explode_during_decode(content):
            raise ValueError(raw_secret)

        monkeypatch.setattr(replay_module, "decode_json_bytes", explode_during_decode)

    with pytest.raises(ArtifactRegistrationIntegrityError) as caught:
        verify_artifact_journal(
            event_store.read(run_id),
            artifact_store=ExplodingStore(),
        )

    assert raw_secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "content",
    [
        b"not-json-\xff",
        b'{"b":2, "a":1}',
        b'{"duplicate":1,"duplicate":2}',
    ],
)
def test_replay_rejects_non_utf8_ambiguous_or_noncanonical_json(content):
    artifact_store, event_store, recorder, journal, run_id = _setup()
    ref = artifact_store.put_bytes(content, media_type=JSON_MEDIA_TYPE)
    recorder.record(_registration_for(ref, ArtifactRole.DIAGNOSTICS))

    with pytest.raises(ArtifactRegistrationIntegrityError):
        verify_artifact_journal(
            event_store.read(run_id),
            artifact_store=artifact_store,
        )


def test_replay_role_rules_cover_manifest_response_and_configuration_fields():
    artifact_store, event_store, recorder, journal, run_id = _setup()
    manifest = journal.register_json(
        {"manifest": {"benchmark": "fixture"}},
        role=ArtifactRole.RUN_MANIFEST,
    )
    recorder.record(RunStarted("a" * 64, manifest.artifact_ref.artifact_id))
    response = journal.register_json(
        {"response": {"text": "safe"}},
        role=ArtifactRole.LLM_RESPONSE,
    )
    recorder.record(
        LLMCallCompleted(
            call_id=LLMCallId("call_replay"),
            response_artifact_id=response.artifact_ref.artifact_id,
        )
    )
    configuration = journal.register_json(
        {"configuration": {"x": 1}},
        role=ArtifactRole.CANDIDATE_CONFIGURATION,
    )
    recorder.record(_candidate(configuration.artifact_ref))

    report = verify_artifact_journal(
        event_store.read(run_id),
        artifact_store=artifact_store,
    )
    assert report.semantic_reference_count == 3
    assert report.unique_referenced_registration_count == 3
    assert report.orphan_registrations == ()


def test_replay_rejects_noncontiguous_or_multiple_run_input():
    artifact_store, event_store, recorder, journal, run_id = _setup()
    registration = journal.register_json(
        {"diagnostics": "safe"},
        role=ArtifactRole.DIAGNOSTICS,
    ).event
    with pytest.raises(ArtifactStreamOrderError, match="contiguous"):
        verify_artifact_journal(
            (replace(registration, sequence_number=2),),
            artifact_store=artifact_store,
        )

    second_artifact_store, second_event_store, second_recorder, second_journal, second_run = (
        _setup("replay_other")
    )
    second = second_journal.register_json(
        {"diagnostics": "other"},
        role=ArtifactRole.DIAGNOSTICS,
    ).event
    # Sequence two avoids reaching the sequence check before the cross-run check.
    with pytest.raises(ArtifactStreamOrderError, match="exactly one run"):
        verify_artifact_journal(
            (registration, replace(second, sequence_number=2)),
            artifact_store=artifact_store,
        )
