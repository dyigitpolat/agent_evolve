"""Shared verification for artifact-store adapters."""

from __future__ import annotations

import agent_evolve.domain.artifact as artifact_domain
from agent_evolve.domain.artifact import ArtifactRef, validate_media_type
from agent_evolve.domain.ids import ArtifactId
from agent_evolve.ports.artifact_store import ArtifactTypeError, CorruptArtifactError


def require_artifact_id(value: ArtifactId) -> None:
    if not isinstance(value, ArtifactId):
        raise ArtifactTypeError("artifact_id must be an ArtifactId")


def require_content_and_media_type(content: bytes, media_type: str) -> None:
    if not isinstance(content, bytes):
        raise ArtifactTypeError("artifact content must be immutable bytes")
    try:
        validate_media_type(media_type)
    except (TypeError, ValueError) as exc:
        raise ArtifactTypeError(f"Invalid artifact media type: {exc}") from exc


def require_expected_media_type(expected_media_type: str | None) -> None:
    if expected_media_type is None:
        return
    try:
        validate_media_type(expected_media_type)
    except (TypeError, ValueError) as exc:
        raise ArtifactTypeError(f"Invalid expected media type: {exc}") from exc


def verify_content(ref: ArtifactRef, content: bytes) -> None:
    if not isinstance(ref, ArtifactRef):
        raise CorruptArtifactError("Stored artifact metadata is not an ArtifactRef")
    if not isinstance(content, bytes):
        raise CorruptArtifactError("Stored artifact content is not immutable bytes")
    if len(content) != ref.size_bytes:
        raise CorruptArtifactError(
            f"Artifact {ref.artifact_id} size does not match its metadata"
        )
    payload_digest = artifact_domain.content_sha256(content)
    if payload_digest != ref.sha256_hex:
        raise CorruptArtifactError(
            f"Artifact {ref.artifact_id} payload does not match its SHA-256 digest"
        )
    identity_digest = artifact_domain.artifact_identity_sha256(
        content,
        media_type=ref.media_type,
    )
    if ref.artifact_id != ArtifactId(f"artifact_{identity_digest}"):
        raise CorruptArtifactError(
            f"Artifact {ref.artifact_id} ID does not match its typed payload"
        )


def verify_expected_media_type(
    ref: ArtifactRef,
    expected_media_type: str | None,
) -> None:
    require_expected_media_type(expected_media_type)
    if expected_media_type is not None and ref.media_type != expected_media_type:
        raise ArtifactTypeError(
            f"Artifact {ref.artifact_id} has media type {ref.media_type!r}, "
            f"not {expected_media_type!r}"
        )
