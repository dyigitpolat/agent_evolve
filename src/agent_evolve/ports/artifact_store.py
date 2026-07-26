"""Content-addressed artifact-store port, errors, and strict codecs."""

from __future__ import annotations

import json
import math
from typing import Any, Protocol, runtime_checkable

from agent_evolve.domain.artifact import ArtifactRef
from agent_evolve.domain.ids import ArtifactId

JSON_MEDIA_TYPE = "application/json"
UTF8_TEXT_MEDIA_TYPE = "text/plain; charset=utf-8"


class ArtifactStoreError(RuntimeError):
    """Base error for artifact persistence and retrieval failures."""


class ArtifactNotFoundError(ArtifactStoreError):
    """The requested artifact ID is not present."""


class CorruptArtifactError(ArtifactStoreError):
    """Stored bytes or metadata do not verify against their artifact ID."""


class ArtifactCollisionError(ArtifactStoreError):
    """Different typed payloads resolved to the same artifact identity."""


class ArtifactTypeError(TypeError, ArtifactStoreError):
    """A value or stored media type does not match the requested representation."""


class ArtifactSerializationError(ArtifactStoreError):
    """A text or JSON value cannot be encoded or decoded strictly."""


@runtime_checkable
class ArtifactStore(Protocol):
    """Persist and verify immutable typed byte payloads by content address."""

    def put_bytes(self, content: bytes, *, media_type: str) -> ArtifactRef: ...

    def stat(self, artifact_id: ArtifactId) -> ArtifactRef: ...

    def read_bytes(
        self,
        artifact_id: ArtifactId,
        *,
        expected_media_type: str | None = None,
    ) -> bytes: ...


def _validate_json_value(
    value: Any,
    *,
    path: str = "$",
    ancestors: set[int] | None = None,
) -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        _validate_utf8_string(value, path=path)
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactSerializationError(f"Non-finite JSON number at {path}")
        return
    if isinstance(value, (list, tuple)):
        ancestors = set() if ancestors is None else ancestors
        identity = id(value)
        if identity in ancestors:
            raise ArtifactSerializationError(f"Cyclic JSON container at {path}")
        ancestors.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_json_value(
                    item,
                    path=f"{path}[{index}]",
                    ancestors=ancestors,
                )
        finally:
            ancestors.remove(identity)
        return
    if isinstance(value, dict):
        ancestors = set() if ancestors is None else ancestors
        identity = id(value)
        if identity in ancestors:
            raise ArtifactSerializationError(f"Cyclic JSON container at {path}")
        ancestors.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ArtifactSerializationError(
                        f"JSON object key at {path} must be a string"
                    )
                _validate_utf8_string(key, path=f"{path} object key")
                _validate_json_value(
                    item,
                    path=f"{path}[{key!r}]",
                    ancestors=ancestors,
                )
        finally:
            ancestors.remove(identity)
        return
    raise ArtifactSerializationError(
        f"Value at {path} has unsupported JSON type {type(value).__name__}"
    )


def _validate_utf8_string(value: str, *, path: str) -> None:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ArtifactSerializationError(
            f"JSON string at {path} is not strict UTF-8 encodable"
        ) from exc


def canonical_json_bytes(value: Any) -> bytes:
    """Encode the supported JSON value domain deterministically as UTF-8."""

    try:
        _validate_json_value(value)
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return encoded.encode("utf-8", errors="strict")
    except ArtifactSerializationError:
        raise
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise ArtifactSerializationError(f"JSON encoding failed: {exc}") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactSerializationError(f"Duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str) -> None:
    raise ArtifactSerializationError(f"Non-standard JSON number {value!r}")


def decode_json_bytes(content: bytes) -> Any:
    """Decode strict UTF-8 JSON, rejecting duplicates and non-standard numbers."""

    if not isinstance(content, bytes):
        raise ArtifactTypeError("JSON artifact content must be bytes")
    try:
        text = content.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_number,
        )
    except ArtifactSerializationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ArtifactSerializationError(f"JSON decoding failed: {exc}") from exc
    try:
        _validate_json_value(value)
    except RecursionError as exc:
        raise ArtifactSerializationError("JSON value exceeds the nesting limit") from exc
    return value


def put_json(
    store: ArtifactStore,
    value: Any,
    *,
    media_type: str = JSON_MEDIA_TYPE,
) -> ArtifactRef:
    return store.put_bytes(canonical_json_bytes(value), media_type=media_type)


def read_json(
    store: ArtifactStore,
    artifact_id: ArtifactId,
    *,
    expected_media_type: str = JSON_MEDIA_TYPE,
) -> Any:
    return decode_json_bytes(
        store.read_bytes(artifact_id, expected_media_type=expected_media_type)
    )


def put_text(
    store: ArtifactStore,
    value: str,
    *,
    media_type: str = UTF8_TEXT_MEDIA_TYPE,
) -> ArtifactRef:
    if not isinstance(value, str):
        raise ArtifactTypeError("text artifact value must be a string")
    try:
        content = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ArtifactSerializationError(f"UTF-8 encoding failed: {exc}") from exc
    return store.put_bytes(content, media_type=media_type)


def read_text(
    store: ArtifactStore,
    artifact_id: ArtifactId,
    *,
    expected_media_type: str = UTF8_TEXT_MEDIA_TYPE,
) -> str:
    content = store.read_bytes(
        artifact_id,
        expected_media_type=expected_media_type,
    )
    try:
        return content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ArtifactSerializationError(f"UTF-8 decoding failed: {exc}") from exc
