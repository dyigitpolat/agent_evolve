"""Redaction-before-persistence application service for JSON artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent_evolve.application.event_recorder import EventRecorder
from agent_evolve.domain.artifact import ArtifactRef, ArtifactRole, artifact_ref_for_bytes
from agent_evolve.domain.event import ArtifactRegistered, EventEnvelope
from agent_evolve.domain.ids import CorrelationId, EventId
from agent_evolve.ports.artifact_sanitizer import ArtifactMinimizer, ArtifactSanitizer
from agent_evolve.ports.artifact_store import (
    JSON_MEDIA_TYPE,
    ArtifactStore,
    canonical_json_bytes,
)


class ArtifactJournalError(RuntimeError):
    """Base error for safe preparation or post-write verification failures."""


class ArtifactPreparationError(ArtifactJournalError):
    """An input could not be minimized, sanitized, and encoded safely."""


class ArtifactSizeLimitError(ArtifactPreparationError):
    """The sanitized canonical representation exceeds the configured limit."""


class ArtifactPostWriteVerificationError(ArtifactJournalError):
    """The durable store did not reproduce the complete expected reference."""


AfterArtifactPutHook = Callable[[ArtifactRef], None]
_PREPARATION_FAILED = object()


@dataclass(frozen=True, slots=True)
class ArtifactRegistrationResult:
    """Successful durable registration returned by :class:`ArtifactJournal`."""

    artifact_ref: ArtifactRef
    event: EventEnvelope


class ArtifactJournal:
    """Prepare, durably store, verify, and then journal a canonical JSON value.

    Raw caller input is passed only to the in-memory minimizer and sanitizer.
    The sole artifact-store call that writes bytes receives the already
    sanitized canonical encoding.
    """

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        event_recorder: EventRecorder,
        minimizer: ArtifactMinimizer,
        sanitizer: ArtifactSanitizer,
        max_size_bytes: int,
        after_artifact_put: AfterArtifactPutHook | None = None,
    ) -> None:
        if (
            isinstance(max_size_bytes, bool)
            or not isinstance(max_size_bytes, int)
            or max_size_bytes < 1
        ):
            raise ValueError("max_size_bytes must be a positive integer")
        self._artifact_store = artifact_store
        self._event_recorder = event_recorder
        self._minimizer = minimizer
        self._sanitizer = sanitizer
        self._max_size_bytes = max_size_bytes
        self._after_artifact_put = after_artifact_put

    @staticmethod
    def _preparation_failure() -> ArtifactPreparationError:
        # This message intentionally contains no value, field path, match count,
        # or chained exception from a caller-supplied policy implementation.
        return ArtifactPreparationError(
            "artifact preparation failed under the configured safety policies"
        )

    def _minimize_without_leaky_context(self, value: Any, role: ArtifactRole) -> Any:
        try:
            return (
                self._minimizer.minimize_json(value, role=role),
                self._minimizer.policy_id,
                self._minimizer.policy_version,
                self._minimizer.policy_config_sha256,
            )
        except Exception:
            # Returning from the handler clears Python's implicit exception
            # context before the caller constructs the public safe error.
            return _PREPARATION_FAILED

    def _sanitize_without_leaky_context(
        self,
        value: Any,
        role: ArtifactRole,
    ) -> Any:
        try:
            sanitized = self._sanitizer.sanitize_json(value, role=role)
            return (
                sanitized,
                self._sanitizer.policy_id,
                self._sanitizer.policy_version,
            )
        except Exception:
            return _PREPARATION_FAILED

    @staticmethod
    def _encode_without_leaky_context(value: Any) -> bytes | object:
        try:
            return canonical_json_bytes(value)
        except Exception:
            return _PREPARATION_FAILED

    @staticmethod
    def _registration_without_leaky_context(
        expected_ref: ArtifactRef,
        role: ArtifactRole,
        minimization_policy_id: Any,
        minimization_policy_version: Any,
        minimization_policy_config_sha256: Any,
        policy_id: Any,
        policy_version: Any,
    ) -> ArtifactRegistered | object:
        try:
            return ArtifactRegistered(
                artifact_id=expected_ref.artifact_id,
                sha256_hex=expected_ref.sha256_hex,
                size_bytes=expected_ref.size_bytes,
                media_type=expected_ref.media_type,
                role=role,
                minimization_policy_id=minimization_policy_id,
                minimization_policy_version=minimization_policy_version,
                minimization_policy_config_sha256=(
                    minimization_policy_config_sha256
                ),
                sanitization_policy_id=policy_id,
                sanitization_policy_version=policy_version,
            )
        except Exception:
            return _PREPARATION_FAILED

    def register_json(
        self,
        value: Any,
        *,
        role: ArtifactRole,
        correlation_id: Optional[CorrelationId] = None,
        causation_event_id: Optional[EventId] = None,
    ) -> ArtifactRegistrationResult:
        """Register one minimized and sanitized JSON artifact.

        Ordering is a safety and crash-consistency contract, not an
        implementation detail: minimize -> sanitize -> canonical encode -> size
        gate -> durable put -> full verification -> durable event append.
        """

        if not isinstance(role, ArtifactRole):
            raise TypeError("role must be an ArtifactRole")

        minimization_result = self._minimize_without_leaky_context(value, role)
        if minimization_result is _PREPARATION_FAILED:
            raise self._preparation_failure()
        (
            minimized,
            minimization_policy_id,
            minimization_policy_version,
            minimization_policy_config_sha256,
        ) = minimization_result
        sanitized_result = self._sanitize_without_leaky_context(minimized, role)
        if sanitized_result is _PREPARATION_FAILED:
            raise self._preparation_failure()
        sanitized, policy_id, policy_version = sanitized_result
        encoded = self._encode_without_leaky_context(sanitized)
        if encoded is _PREPARATION_FAILED:
            raise self._preparation_failure()
        assert isinstance(encoded, bytes)  # narrowed from the private sentinel union

        if len(encoded) > self._max_size_bytes:
            raise ArtifactSizeLimitError(
                "sanitized artifact exceeds the configured size limit"
            )

        expected_ref = artifact_ref_for_bytes(encoded, media_type=JSON_MEDIA_TYPE)
        # Constructing the payload before the write validates all provenance
        # metadata.  It is appended only after post-write verification succeeds.
        registration = self._registration_without_leaky_context(
            expected_ref,
            role,
            minimization_policy_id,
            minimization_policy_version,
            minimization_policy_config_sha256,
            policy_id,
            policy_version,
        )
        if registration is _PREPARATION_FAILED:
            raise self._preparation_failure()
        assert isinstance(registration, ArtifactRegistered)

        persisted_ref = self._artifact_store.put_bytes(
            encoded,
            media_type=JSON_MEDIA_TYPE,
        )
        if self._after_artifact_put is not None:
            # A test/deployment crash hook receives only content metadata, never
            # the raw or sanitized payload.  Raising leaves a store-only orphan.
            self._after_artifact_put(expected_ref)

        try:
            stat_ref = self._artifact_store.stat(expected_ref.artifact_id)
            reread = self._artifact_store.read_bytes(
                expected_ref.artifact_id,
                expected_media_type=JSON_MEDIA_TYPE,
            )
            reread_ref = artifact_ref_for_bytes(
                reread,
                media_type=JSON_MEDIA_TYPE,
            )
        except Exception:
            raise ArtifactPostWriteVerificationError(
                "artifact could not be fully verified after persistence"
            ) from None

        if (
            persisted_ref != expected_ref
            or stat_ref != expected_ref
            or reread_ref != expected_ref
            or reread != encoded
        ):
            raise ArtifactPostWriteVerificationError(
                "artifact metadata or bytes changed during persistence"
            )

        event = self._event_recorder.record(
            registration,
            correlation_id=correlation_id,
            causation_event_id=causation_event_id,
        )
        return ArtifactRegistrationResult(artifact_ref=expected_ref, event=event)
