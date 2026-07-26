"""Read-only replay and integrity verification for artifact registrations."""

from __future__ import annotations

import types
from dataclasses import dataclass, fields
from typing import Any, Iterable, Optional, Tuple, Union, get_args, get_origin, get_type_hints

from agent_evolve.domain.artifact import ArtifactRef, ArtifactRole, artifact_ref_for_bytes
from agent_evolve.domain.event import (
    EVENT_PAYLOAD_TYPES,
    ArtifactRegistered,
    CandidateProposed,
    EvaluationFailed,
    EventEnvelope,
    EventPayload,
    LLMCallCompleted,
    LLMCallFailed,
    LLMRequestArtifactLinked,
    ProviderAttemptFailed,
    RunAborted,
    RunStarted,
    validated_event_snapshot,
)
from agent_evolve.domain.ids import ArtifactId, RunId
from agent_evolve.ports.artifact_store import (
    JSON_MEDIA_TYPE,
    ArtifactStore,
    canonical_json_bytes,
    decode_json_bytes,
)


class ArtifactReplayError(RuntimeError):
    """The event stream and artifact store do not form a valid journal."""


class ArtifactStreamOrderError(ArtifactReplayError):
    """Replay input is not one contiguous run stream in sequence order."""


class ArtifactRegistrationIntegrityError(ArtifactReplayError):
    """A registration claim is inconsistent with the durable store."""


class DanglingArtifactReferenceError(ArtifactReplayError):
    """A semantic reference has no preceding registration."""


class IncompatibleArtifactRoleError(ArtifactReplayError):
    """An artifact was registered earlier, but not under the required role."""


class ArtifactReferenceSchemaError(ArtifactReplayError):
    """An ArtifactId-bearing payload field has no explicit role rule."""


_SNAPSHOT_FAILED = object()
_REGISTRATION_VERIFICATION_FAILED = object()
_REGISTRATION_MISMATCH = object()


@dataclass(frozen=True, slots=True)
class ArtifactRegistrationKey:
    artifact_id: ArtifactId
    role: ArtifactRole


@dataclass(frozen=True, slots=True)
class ArtifactReplayReport:
    run_id: Optional[RunId]
    event_count: int
    registration_event_count: int
    semantic_reference_count: int
    unique_registration_count: int
    unique_referenced_registration_count: int
    orphan_registrations: Tuple[ArtifactRegistrationKey, ...]


@dataclass(frozen=True, slots=True)
class _RegistrationClaim:
    artifact_ref: ArtifactRef
    minimization_policy_id: str
    minimization_policy_version: str
    minimization_policy_config_sha256: str
    policy_id: str
    policy_version: str


# The map is intentionally explicit and fail-closed.  Adding an ArtifactId field
# to the event vocabulary requires declaring which semantic role may satisfy it.
_REFERENCE_ROLES: dict[type[EventPayload], dict[str, ArtifactRole]] = {
    RunStarted: {"manifest_artifact_id": ArtifactRole.RUN_MANIFEST},
    RunAborted: {"diagnostics_artifact_id": ArtifactRole.DIAGNOSTICS},
    LLMCallCompleted: {"response_artifact_id": ArtifactRole.LLM_RESPONSE},
    LLMRequestArtifactLinked: {"request_artifact_id": ArtifactRole.LLM_REQUEST},
    LLMCallFailed: {"diagnostics_artifact_id": ArtifactRole.DIAGNOSTICS},
    ProviderAttemptFailed: {"diagnostics_artifact_id": ArtifactRole.DIAGNOSTICS},
    CandidateProposed: {
        "configuration_artifact_id": ArtifactRole.CANDIDATE_CONFIGURATION
    },
    EvaluationFailed: {"diagnostics_artifact_id": ArtifactRole.DIAGNOSTICS},
}


def _annotation_contains_artifact_id(annotation: Any) -> bool:
    if annotation is ArtifactId:
        return True
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType) or origin in (tuple, list, dict):
        return any(_annotation_contains_artifact_id(arg) for arg in get_args(annotation))
    return False


def _annotation_is_scalar_artifact_id(annotation: Any) -> bool:
    if annotation is ArtifactId:
        return True
    origin = get_origin(annotation)
    args = get_args(annotation)
    return origin in (Union, types.UnionType) and set(args) == {
        ArtifactId,
        type(None),
    }


def _artifact_fields(payload: EventPayload) -> tuple[str, ...]:
    hints = get_type_hints(type(payload))
    return tuple(
        field.name
        for field in fields(payload)
        if _annotation_contains_artifact_id(hints[field.name])
    )


def validate_artifact_reference_schema() -> None:
    """Fail if the complete event vocabulary and semantic role map drift apart."""

    payload_types = set(EVENT_PAYLOAD_TYPES)
    extra_rule_types = set(_REFERENCE_ROLES) - payload_types
    if extra_rule_types:
        raise ArtifactReferenceSchemaError(
            "artifact reference role rules include an unregistered event payload"
        )

    for payload_type in EVENT_PAYLOAD_TYPES:
        if payload_type is ArtifactRegistered:
            hints = get_type_hints(payload_type)
            registration_artifact_fields = {
                field.name
                for field in fields(payload_type)
                if _annotation_contains_artifact_id(hints[field.name])
            }
            if registration_artifact_fields != {"artifact_id"}:
                raise ArtifactReferenceSchemaError(
                    "ArtifactRegistered identity fields require an explicit schema review"
                )
            continue
        hints = get_type_hints(payload_type)
        artifact_fields = {
            field.name
            for field in fields(payload_type)
            if _annotation_contains_artifact_id(hints[field.name])
        }
        declared_roles = _REFERENCE_ROLES.get(payload_type, {})
        if artifact_fields != set(declared_roles):
            raise ArtifactReferenceSchemaError(
                "artifact reference role rules do not match the event schema"
            )
        for field_name in artifact_fields:
            if not _annotation_is_scalar_artifact_id(hints[field_name]):
                raise ArtifactReferenceSchemaError(
                    "non-scalar ArtifactId references require an explicit replay design"
                )
            if not isinstance(declared_roles[field_name], ArtifactRole):
                raise ArtifactReferenceSchemaError(
                    "artifact reference rules must declare a constrained role"
                )


# Validate the whole vocabulary at import, not only payload types encountered by
# a particular run.  A schema change therefore cannot remain latent in quiet runs.
validate_artifact_reference_schema()


def _try_verify_registration_store(
    registration: ArtifactRegistered,
    store: ArtifactStore,
) -> ArtifactRef | object:
    expected = registration.artifact_ref
    try:
        stat_ref = store.stat(expected.artifact_id)
        content = store.read_bytes(
            expected.artifact_id,
            expected_media_type=expected.media_type,
        )
        content_ref = artifact_ref_for_bytes(content, media_type=expected.media_type)
        decoded = decode_json_bytes(content)
        canonical_content = canonical_json_bytes(decoded)
        if (
            stat_ref != expected
            or content_ref != expected
            or canonical_content != content
        ):
            return _REGISTRATION_MISMATCH
    except Exception:
        # Return from the handler so secret-bearing adapter/codec exceptions do
        # not become context on the public integrity error raised by the caller.
        return _REGISTRATION_VERIFICATION_FAILED
    return expected


def _verify_registration_store(
    registration: ArtifactRegistered,
    store: ArtifactStore,
) -> ArtifactRef:
    expected = registration.artifact_ref
    if expected.media_type != JSON_MEDIA_TYPE:
        raise ArtifactRegistrationIntegrityError(
            "M1c registrations must use the canonical JSON media type"
        )
    verification = _try_verify_registration_store(registration, store)
    if verification is _REGISTRATION_VERIFICATION_FAILED:
        raise ArtifactRegistrationIntegrityError(
            "a registered artifact could not be verified in the durable store"
        )
    if verification is _REGISTRATION_MISMATCH:
        raise ArtifactRegistrationIntegrityError(
            "a registration does not match a canonical durable artifact reference"
        )
    assert type(verification) is ArtifactRef
    return verification


def verify_artifact_journal(
    events: Iterable[EventEnvelope],
    *,
    artifact_store: ArtifactStore,
) -> ArtifactReplayReport:
    """Verify one run deterministically without mutating events or artifacts.

    Registrations that are never referenced are valid: they can result from a
    crash after registration but before the semantic event.  Conversely, a
    store-only orphan has no journal record and is deliberately invisible here.
    """

    try:
        stream = tuple(validated_event_snapshot(event) for event in events)
    except Exception:
        stream = _SNAPSHOT_FAILED
    if stream is _SNAPSHOT_FAILED:
        raise ArtifactStreamOrderError(
            "artifact replay requires canonically validated event snapshots"
        )
    assert isinstance(stream, tuple)

    run_id: Optional[RunId] = None
    expected_sequence = 1
    event_count = 0
    registration_event_count = 0
    semantic_reference_count = 0
    registrations: dict[ArtifactRegistrationKey, _RegistrationClaim] = {}
    refs_by_id: dict[ArtifactId, ArtifactRef] = {}
    roles_by_id: dict[ArtifactId, set[ArtifactRole]] = {}
    referenced: set[ArtifactRegistrationKey] = set()

    for event in stream:
        event_count += 1
        if run_id is None:
            run_id = event.run_id
        elif event.run_id != run_id:
            raise ArtifactStreamOrderError(
                "artifact replay requires events from exactly one run"
            )
        if event.sequence_number != expected_sequence:
            raise ArtifactStreamOrderError(
                "artifact replay requires a contiguous sequence beginning at one"
            )
        expected_sequence += 1

        payload = event.payload
        if isinstance(payload, ArtifactRegistered):
            registration_event_count += 1
            ref = _verify_registration_store(payload, artifact_store)
            previous_ref = refs_by_id.get(ref.artifact_id)
            if previous_ref is not None and previous_ref != ref:
                raise ArtifactRegistrationIntegrityError(
                    "registrations disagree on a complete artifact reference"
                )
            refs_by_id[ref.artifact_id] = ref
            key = ArtifactRegistrationKey(ref.artifact_id, payload.role)
            claim = _RegistrationClaim(
                artifact_ref=ref,
                minimization_policy_id=payload.minimization_policy_id,
                minimization_policy_version=payload.minimization_policy_version,
                minimization_policy_config_sha256=(
                    payload.minimization_policy_config_sha256
                ),
                policy_id=payload.sanitization_policy_id,
                policy_version=payload.sanitization_policy_version,
            )
            previous_claim = registrations.get(key)
            if previous_claim is not None and previous_claim != claim:
                raise ArtifactRegistrationIntegrityError(
                    "duplicate registrations disagree on provenance"
                )
            registrations[key] = claim
            roles_by_id.setdefault(ref.artifact_id, set()).add(payload.role)
            continue

        artifact_fields = _artifact_fields(payload)
        declared_roles = _REFERENCE_ROLES.get(type(payload), {})
        if set(artifact_fields) != set(declared_roles):
            raise ArtifactReferenceSchemaError(
                "artifact reference role rules do not match the event schema"
            )
        for field_name in artifact_fields:
            artifact_id = getattr(payload, field_name)
            if artifact_id is None:
                continue
            semantic_reference_count += 1
            role = declared_roles[field_name]
            key = ArtifactRegistrationKey(artifact_id, role)
            if key not in registrations:
                if artifact_id in roles_by_id:
                    raise IncompatibleArtifactRoleError(
                        "an artifact reference has no preceding compatible role"
                    )
                raise DanglingArtifactReferenceError(
                    "an artifact reference has no preceding registration"
                )
            referenced.add(key)

    orphan_keys = tuple(
        sorted(
            set(registrations) - referenced,
            key=lambda key: (key.artifact_id.value, key.role.value),
        )
    )
    return ArtifactReplayReport(
        run_id=run_id,
        event_count=event_count,
        registration_event_count=registration_event_count,
        semantic_reference_count=semantic_reference_count,
        unique_registration_count=len(registrations),
        unique_referenced_registration_count=len(referenced),
        orphan_registrations=orphan_keys,
    )
