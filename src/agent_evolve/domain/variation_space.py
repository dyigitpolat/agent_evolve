"""Immutable option identities for finite atomic variation spaces.

An :class:`AtomicEditOption` is deliberately more than a provider-facing
replacement value.  It binds that value to one typed path, one variation
family, and exact source metadata so observations cannot silently migrate
between coordinates or catalog revisions.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from agent_evolve.domain.patch import (
    JsonPath,
    canonical_path_bytes,
    validate_json_path,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonValue,
    canonical_typed_json_bytes,
    freeze_json,
    is_json_scalar,
)


_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_FAMILY = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_METADATA_KEY = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_MAX_METADATA_ITEMS = 64
_MAX_METADATA_VALUE_BYTES = 16_384
_ATOMIC_EDIT_OPTION_DOMAIN = b"agent-evolve:atomic-edit-option:v1\x00"


def _utf8(value: str, *, name: str, max_bytes: int) -> bytes:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be strict UTF-8") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{name} exceeds its byte limit")
    return encoded


def _frame(payload: bytes) -> bytes:
    if type(payload) is not bytes:
        raise TypeError("framed payloads must be exact bytes")
    return len(payload).to_bytes(8, "big", signed=False) + payload


@dataclass(frozen=True, slots=True, eq=False)
class AtomicEditOption:
    """One closed, source-bound scalar replacement at one exact JSON path.

    ``metadata`` is canonical: keys are unique and lexicographically ordered.
    This makes the complete option identity independent of mapping iteration
    order while retaining a compact tuple representation at the domain edge.
    """

    option_id: str
    path: JsonPath
    replacement: FrozenJsonValue
    family: str
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or _OPTION_ID.fullmatch(
            self.option_id
        ) is None:
            raise ValueError(
                "option_id must use the closed lowercase identifier grammar"
            )
        if type(self.path) is not JsonPath:
            raise TypeError("path must be an exact JsonPath")
        validate_json_path(self.path)
        if not self.path.segments:
            raise ValueError("an atomic edit option cannot replace the root")
        if freeze_json(self.replacement) is not self.replacement:
            raise TypeError("replacement must already be frozen typed JSON")
        if not is_json_scalar(self.replacement):
            raise TypeError("replacement must be a typed-JSON scalar")
        if type(self.family) is not str or _FAMILY.fullmatch(self.family) is None:
            raise ValueError("family must use the closed lowercase token grammar")
        if type(self.metadata) is not tuple:
            raise TypeError("metadata must be an exact tuple")
        if len(self.metadata) > _MAX_METADATA_ITEMS:
            raise ValueError("metadata exceeds the item limit")

        keys: list[str] = []
        for item in self.metadata:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("metadata entries must be exact key/value tuples")
            key, value = item
            if type(key) is not str or _METADATA_KEY.fullmatch(key) is None:
                raise ValueError(
                    "metadata keys must use the closed lowercase token grammar"
                )
            _utf8(
                value,
                name=f"metadata value for {key!r}",
                max_bytes=_MAX_METADATA_VALUE_BYTES,
            )
            keys.append(key)
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError("metadata keys must be unique and canonically sorted")

    @property
    def identity_sha256(self) -> str:
        """Return a type-sensitive digest binding every semantic field."""

        validate_atomic_edit_option(self)
        digest = hashlib.sha256()
        digest.update(_ATOMIC_EDIT_OPTION_DOMAIN)
        digest.update(_frame(self.option_id.encode("ascii")))
        digest.update(_frame(canonical_path_bytes(self.path)))
        digest.update(_frame(canonical_typed_json_bytes(self.replacement)))
        digest.update(_frame(self.family.encode("ascii")))
        digest.update(len(self.metadata).to_bytes(8, "big", signed=False))
        for key, value in self.metadata:
            digest.update(_frame(key.encode("ascii")))
            digest.update(_frame(value.encode("utf-8", errors="strict")))
        return digest.hexdigest()

    def _validated_values(
        self,
    ) -> tuple[str, bytes, bytes, str, tuple[tuple[str, str], ...]]:
        if type(self) is not AtomicEditOption:
            raise TypeError("option must be an exact AtomicEditOption")
        AtomicEditOption.__post_init__(self)
        return (
            self.option_id,
            canonical_path_bytes(self.path),
            canonical_typed_json_bytes(self.replacement),
            self.family,
            self.metadata,
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not AtomicEditOption or type(other) is not AtomicEditOption:
            return False
        return self._validated_values() == other._validated_values()

    def __hash__(self) -> int:
        return hash((AtomicEditOption, self._validated_values()))


def validate_atomic_edit_option(option: AtomicEditOption) -> None:
    """Revalidate an exact option at a public trust boundary."""

    if type(option) is not AtomicEditOption:
        raise TypeError("option must be an exact AtomicEditOption")
    AtomicEditOption.__post_init__(option)


__all__ = ["AtomicEditOption", "validate_atomic_edit_option"]
