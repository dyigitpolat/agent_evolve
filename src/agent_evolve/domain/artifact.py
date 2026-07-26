"""Immutable metadata and identity rules for content-addressed artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from agent_evolve.domain.durable_text import (
    contains_credential_shape,
    contains_identifier_content_marker,
    contains_inline_content_marker,
)
from agent_evolve.domain.ids import ArtifactId

SHA256_HEX_LENGTH = 64
MAX_MEDIA_TYPE_LENGTH = 256
_ARTIFACT_ID_DOMAIN = b"agent-evolve:artifact-id:v2\x00"
_MEDIA_TYPE_TOKEN = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%&'*+-.^_`|~"
)
_ALLOWED_PARAMETERIZED_MEDIA_TYPES = frozenset(
    {
        "text/plain; charset=utf-8",
    }
)


class ArtifactRole(str, Enum):
    """Constrained semantic purposes for durable experiment artifacts.

    A role is intentionally narrower than a media type.  Replay uses it to
    ensure, for example, that a diagnostics blob cannot be substituted for a
    candidate configuration merely because both happen to contain JSON.
    """

    RUN_MANIFEST = "run_manifest"
    CANDIDATE_CONFIGURATION = "candidate_configuration"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    DIAGNOSTICS = "diagnostics"


def _validate_sha256_hex(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("sha256_hex must be a string")
    if len(value) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("sha256_hex must be a lowercase SHA-256 hex digest")


def validate_media_type(value: str) -> None:
    """Validate an exact media-type label with a closed parameter grammar.

    Exact equality remains part of artifact identity. Parameterized forms are
    accepted only when this schema names the complete canonical label; arbitrary
    parameter tails are content-bearing metadata channels and fail closed.
    """

    if type(value) is not str:
        raise TypeError("media_type must be a string")
    if not value or value != value.strip():
        raise ValueError("media_type must be a non-empty, trimmed string")
    if len(value) > MAX_MEDIA_TYPE_LENGTH:
        raise ValueError("media_type exceeds the durable metadata limit")
    if any(ord(character) < 32 or ord(character) > 126 for character in value):
        raise ValueError("media_type must contain printable ASCII only")
    if (
        contains_credential_shape(value)
        or contains_inline_content_marker(value)
        or contains_identifier_content_marker(value)
    ):
        raise ValueError("media_type violates the durable metadata policy")
    parts = value.split(";")
    essence = parts[0]
    if len(parts) > 1 and value not in _ALLOWED_PARAMETERIZED_MEDIA_TYPES:
        raise ValueError("media_type violates the durable metadata policy")
    if essence.count("/") != 1:
        raise ValueError("media_type must contain a type and subtype")
    type_name, subtype_name = essence.split("/", 1)
    if (
        not type_name
        or not subtype_name
        or any(character not in _MEDIA_TYPE_TOKEN for character in type_name)
        or any(character not in _MEDIA_TYPE_TOKEN for character in subtype_name)
    ):
        raise ValueError("media_type type and subtype must use valid token characters")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Verified immutable metadata for an artifact payload."""

    artifact_id: ArtifactId
    sha256_hex: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, ArtifactId):
            raise TypeError("artifact_id must be an ArtifactId")
        _validate_sha256_hex(self.sha256_hex)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        validate_media_type(self.media_type)


def content_sha256(content: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact immutable payload bytes."""

    if not isinstance(content, bytes):
        raise TypeError("artifact content must be bytes")
    return hashlib.sha256(content).hexdigest()


def artifact_identity_sha256(content: bytes, *, media_type: str) -> str:
    """Return the portable identity digest for an exact typed payload.

    The versioned preimage is unambiguous across implementations::

        domain || uint64_be(media_type_length) || media_type_ascii
               || uint64_be(payload_length) || payload

    Lengths are byte lengths. The domain tag and length framing ensure that the
    digest identifies the exact ``(media_type, payload bytes)`` tuple rather than
    an ambiguous concatenation. The payload checksum exposed on :class:`ArtifactRef`
    deliberately remains the plain SHA-256 of only the payload bytes.
    """

    if not isinstance(content, bytes):
        raise TypeError("artifact content must be bytes")
    validate_media_type(media_type)
    media_type_bytes = media_type.encode("ascii", errors="strict")
    try:
        media_type_length = len(media_type_bytes).to_bytes(8, "big", signed=False)
        payload_length = len(content).to_bytes(8, "big", signed=False)
    except OverflowError as exc:  # pragma: no cover - impossible on supported hosts.
        raise ValueError("artifact identity fields exceed uint64 framing") from exc

    identity = hashlib.sha256()
    identity.update(_ARTIFACT_ID_DOMAIN)
    identity.update(media_type_length)
    identity.update(media_type_bytes)
    identity.update(payload_length)
    identity.update(content)
    return identity.hexdigest()


def artifact_ref_for_bytes(content: bytes, *, media_type: str) -> ArtifactRef:
    """Build the canonical reference for exact payload bytes and a media type."""

    payload_digest = content_sha256(content)
    identity_digest = artifact_identity_sha256(content, media_type=media_type)
    return ArtifactRef(
        artifact_id=ArtifactId(f"artifact_{identity_digest}"),
        sha256_hex=payload_digest,
        size_bytes=len(content),
        media_type=media_type,
    )
