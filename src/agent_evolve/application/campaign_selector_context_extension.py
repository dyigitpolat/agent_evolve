"""Authenticated additive context for one campaign selector request.

Campaign runtimes own the complete trusted selector context.  A workload
adapter may nevertheless need to publish a small, decision-local annotation
that can only be computed after the runtime has selected a parent and exposed
the current evidence cohort.  Memory transfer authority is one example.

This module provides the sole generic escape hatch.  An extension is attached
under one reserved top-level key, is source-sealed and content-addressed, and
may only be added to an otherwise byte-identical trusted context.  It cannot
replace or delete workload or core-authored fields.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY = "campaign_selector_context_extension"
CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_MAX_BYTES = 65_536

_EXTENSION_DOMAIN = b"agent-evolve:campaign-selector-context-extension:v1\x00"
_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_LOWER_SHA256 = frozenset("0123456789abcdef")


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("selector context extension did not freeze to an object")
    return frozen


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_SHA256 for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class CampaignSelectorContextExtension:
    """One source-sealed, decision-local workload annotation."""

    extension_id: str
    extension_version: int
    definition_sha256: str
    payload: FrozenJsonObject
    payload_sha256: str = field(init=False)
    extension_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.extension_id) is not str or _TOKEN.fullmatch(
            self.extension_id
        ) is None:
            raise ValueError("extension_id must use the closed lowercase token grammar")
        if type(self.extension_version) is not int or self.extension_version <= 0:
            raise ValueError("extension_version must be a positive exact integer")
        _require_sha256(self.definition_sha256, name="definition_sha256")
        if (
            type(self.payload) is not FrozenJsonObject
            or freeze_json(self.payload) is not self.payload
            or not self.payload.items
        ):
            raise TypeError("payload must be a non-empty exact frozen object")
        payload_bytes = _canonical_bytes(thaw_json(self.payload))
        if len(payload_bytes) > CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_MAX_BYTES:
            raise ValueError("selector context extension exceeds its byte bound")
        payload_sha256 = typed_json_sha256(self.payload)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(
            self,
            "extension_sha256",
            hashlib.sha256(
                _EXTENSION_DOMAIN + _canonical_bytes(self._unsigned_record())
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "extension_id": self.extension_id,
            "extension_version": self.extension_version,
            "definition_sha256": self.definition_sha256,
            "payload_sha256": typed_json_sha256(self.payload),
        }

    def to_binding_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "extension_sha256": self.extension_sha256,
        }

    def to_prompt_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self.to_binding_record(),
            "payload": thaw_json(self.payload),
        }

    @classmethod
    def from_prompt_record(
        cls,
        record: FrozenJsonObject,
    ) -> "CampaignSelectorContextExtension":
        """Replay and authenticate a model-visible extension record."""

        if type(record) is not FrozenJsonObject:
            raise TypeError("extension record must be an exact frozen object")
        values = dict(record.items)
        if set(values) != {
            "schema_version",
            "extension_id",
            "extension_version",
            "definition_sha256",
            "payload_sha256",
            "extension_sha256",
            "payload",
        }:
            raise ValueError("extension record has an invalid field set")
        if values["schema_version"] != 1:
            raise ValueError("extension record uses an unknown schema version")
        payload = values["payload"]
        if type(payload) is not FrozenJsonObject:
            raise TypeError("extension payload must remain a frozen object")
        extension = cls(
            extension_id=values["extension_id"],
            extension_version=values["extension_version"],
            definition_sha256=values["definition_sha256"],
            payload=payload,
        )
        if (
            values["payload_sha256"] != extension.payload_sha256
            or values["extension_sha256"] != extension.extension_sha256
        ):
            raise ValueError("selector context extension authentication failed")
        return extension


def attach_campaign_selector_context_extension(
    trusted_context: FrozenJsonObject,
    extension: CampaignSelectorContextExtension,
) -> FrozenJsonObject:
    """Append one extension without changing any trusted context field."""

    if type(trusted_context) is not FrozenJsonObject:
        raise TypeError("trusted_context must be an exact frozen object")
    if type(extension) is not CampaignSelectorContextExtension:
        raise TypeError("extension must be exact")
    extension.__post_init__()
    values = thaw_json(trusted_context)
    if type(values) is not dict:  # pragma: no cover - closed root.
        raise AssertionError("trusted selector context did not thaw to an object")
    if CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY in values:
        raise ValueError("trusted context already uses the reserved extension key")
    values[CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY] = extension.to_prompt_record()
    return _object(values)


def resolve_campaign_selector_context_extension(
    *,
    trusted_context: FrozenJsonObject,
    selector_context: FrozenJsonObject,
) -> CampaignSelectorContextExtension | None:
    """Validate exact identity or one authenticated additive extension."""

    if type(trusted_context) is not FrozenJsonObject:
        raise TypeError("trusted_context must be an exact frozen object")
    if type(selector_context) is not FrozenJsonObject:
        raise TypeError("selector_context must be an exact frozen object")
    trusted = dict(trusted_context.items)
    observed = dict(selector_context.items)
    if CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY in trusted:
        raise ValueError("trusted context uses the reserved extension key")
    if observed == trusted:
        return None
    if set(observed) != {
        *trusted,
        CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY,
    }:
        raise ValueError("selector context is not an additive extension")
    if any(observed[key] != value for key, value in trusted.items()):
        raise ValueError("selector context changed a trusted base field")
    record = observed[CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY]
    if type(record) is not FrozenJsonObject:
        raise TypeError("selector context extension record must be frozen")
    return CampaignSelectorContextExtension.from_prompt_record(record)


__all__ = [
    "CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY",
    "CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_MAX_BYTES",
    "CampaignSelectorContextExtension",
    "attach_campaign_selector_context_extension",
    "resolve_campaign_selector_context_extension",
]
