"""Authenticated, workload-neutral projection of finite-option prompt facts.

Finite variation options may carry replay and catalog hashes alongside the
semantic facts a model needs to compare actions.  Sending every metadata field
can dominate the prompt without expanding model authority: trusted code still
owns the full child configurations and exact option identities.  This policy
lets a benchmark composition root explicitly allowlist prompt-visible metadata
while retaining an authenticated join to every sealed source option.

The legacy ``ALL`` mode reproduces ``FiniteVariationContract.prompt_records``
exactly.  No key is removed by heuristic and no outcome value is consulted.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.patch import require_sha256


POLICY_ID = "finite_option_prompt_projection"
POLICY_VERSION = 1
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:finite-option-prompt-projection:v1\x00"
    b"always=option-id,family,description;metadata=all-or-explicit-allowlist;"
    b"unknown-allowlist-key=fail;source-option-identities=authenticated;"
    b"outcome-access=false;heuristic-key-removal=false"
).hexdigest()

_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_CONFIGURATION_DOMAIN = b"agent-evolve:finite-option-prompt-config:v1\x00"
_PROMPT_RECORD_DOMAIN = b"agent-evolve:finite-option-prompt-record:v1\x00"
_PROJECTION_DOMAIN = b"agent-evolve:finite-option-prompt-projection:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


class PromptMetadataProjectionMode(str, Enum):
    """Closed choice between legacy-complete and explicit semantic metadata."""

    ALL = "all"
    ALLOWLIST = "allowlist"


def _validate_metadata_keys(values: tuple[str, ...], *, name: str) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or _TOKEN.fullmatch(value) is None for value in values
    ):
        raise TypeError(f"{name} must be an exact tuple of closed metadata keys")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")


@dataclass(frozen=True, slots=True)
class FiniteOptionPromptRecord:
    """One projected record plus its hidden exact source-option join."""

    option_id: str
    family: str
    description: str
    metadata: tuple[tuple[str, str], ...]
    source_option_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be a non-empty exact string")
        if type(self.family) is not str or not self.family:
            raise ValueError("family must be a non-empty exact string")
        if type(self.description) is not str or not self.description:
            raise ValueError("description must be a non-empty exact string")
        if type(self.metadata) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            for item in self.metadata
        ):
            raise TypeError("metadata must contain exact key/value string pairs")
        keys = tuple(key for key, _ in self.metadata)
        _validate_metadata_keys(keys, name="metadata keys")
        require_sha256(
            self.source_option_identity_sha256,
            "source_option_identity_sha256",
        )

    def to_prompt_record(self) -> dict[str, object]:
        """Return only model-visible semantic facts, never hidden identities."""

        self.__post_init__()
        return {
            "option_id": self.option_id,
            "family": self.family,
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @property
    def prompt_record_sha256(self) -> str:
        return hashlib.sha256(
            _PROMPT_RECORD_DOMAIN + _canonical_json(self.to_prompt_record())
        ).hexdigest()

    def to_binding_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "option_id": self.option_id,
            "source_option_identity_sha256": self.source_option_identity_sha256,
            "prompt_record_sha256": self.prompt_record_sha256,
        }


@dataclass(frozen=True, slots=True)
class FiniteOptionPromptProjection:
    """Immutable projection receipt and ordered model-visible records."""

    source_contract_sha256: str
    mode: PromptMetadataProjectionMode
    included_metadata_keys: tuple[str, ...]
    available_metadata_keys: tuple[str, ...]
    omitted_metadata_keys: tuple[str, ...]
    records: tuple[FiniteOptionPromptRecord, ...]
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    policy_configuration_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.source_contract_sha256, "source_contract_sha256")
        if type(self.mode) is not PromptMetadataProjectionMode:
            raise TypeError("mode must be exact PromptMetadataProjectionMode")
        for name in (
            "included_metadata_keys",
            "available_metadata_keys",
            "omitted_metadata_keys",
        ):
            _validate_metadata_keys(getattr(self, name), name=name)
        available = set(self.available_metadata_keys)
        included = set(self.included_metadata_keys)
        omitted = set(self.omitted_metadata_keys)
        if not included.issubset(available):
            raise ValueError("included metadata keys escape the available key set")
        if omitted != available - included or omitted & included:
            raise ValueError("omitted metadata keys do not complement included keys")
        if self.mode is PromptMetadataProjectionMode.ALL and omitted:
            raise ValueError("all-metadata projection cannot omit a key")
        if (
            type(self.records) is not tuple
            or not self.records
            or any(
                type(value) is not FiniteOptionPromptRecord for value in self.records
            )
        ):
            raise TypeError("records must contain exact prompt records")
        for value in self.records:
            value.__post_init__()
            record_keys = {key for key, _ in value.metadata}
            if not record_keys.issubset(included):
                raise ValueError("a prompt record escaped the included metadata set")
        option_ids = tuple(value.option_id for value in self.records)
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("prompt records cannot repeat an option ID")
        if self.policy_id != POLICY_ID:
            raise ValueError("policy_id drifted")
        if self.policy_version != POLICY_VERSION:
            raise ValueError("policy_version drifted")
        if self.policy_definition_sha256 != POLICY_DEFINITION_SHA256:
            raise ValueError("policy_definition_sha256 drifted")
        require_sha256(
            self.policy_configuration_sha256,
            "policy_configuration_sha256",
        )
        expected_configuration = FiniteOptionPromptProjectionPolicy(
            metadata_keys=(
                None
                if self.mode is PromptMetadataProjectionMode.ALL
                else self.included_metadata_keys
            )
        ).configuration_sha256
        if self.policy_configuration_sha256 != expected_configuration:
            raise ValueError("policy configuration does not match projection mode")

    def prompt_records(self) -> tuple[dict[str, object], ...]:
        self.__post_init__()
        return tuple(value.to_prompt_record() for value in self.records)

    def _unsigned_binding_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "source_contract_sha256": self.source_contract_sha256,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
                "configuration_sha256": self.policy_configuration_sha256,
                "mode": self.mode.value,
            },
            "included_metadata_keys": list(self.included_metadata_keys),
            "available_metadata_keys": list(self.available_metadata_keys),
            "omitted_metadata_keys": list(self.omitted_metadata_keys),
            "ordered_records": [value.to_binding_record() for value in self.records],
            "outcome_values_consulted": False,
        }

    @property
    def projection_sha256(self) -> str:
        return hashlib.sha256(
            _PROJECTION_DOMAIN + _canonical_json(self._unsigned_binding_record())
        ).hexdigest()

    def to_binding_record(self) -> dict[str, object]:
        return {
            **self._unsigned_binding_record(),
            "projection_sha256": self.projection_sha256,
        }

    def to_prompt_contract_record(self) -> dict[str, object]:
        """Return the compact receipt rendered beside projected options.

        The complete per-option source joins remain inside the input binding.
        The prompt needs only their aggregate projection digest plus the closed
        policy/configuration facts: ``input_binding_sha256`` authenticates the
        complete receipt without paying to render hundreds of hidden hashes.
        """

        self.__post_init__()
        return {
            "schema_version": 1,
            "projection_sha256": self.projection_sha256,
            "source_contract_sha256": self.source_contract_sha256,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
                "configuration_sha256": self.policy_configuration_sha256,
                "mode": self.mode.value,
            },
            "included_metadata_keys": list(self.included_metadata_keys),
            "available_metadata_keys": list(self.available_metadata_keys),
            "omitted_metadata_keys": list(self.omitted_metadata_keys),
            "ordered_record_count": len(self.records),
            "outcome_values_consulted": False,
        }

    def require_contract(self, contract: FiniteVariationContract) -> None:
        """Fail closed unless replay from ``contract`` is byte-identical."""

        if type(contract) is not FiniteVariationContract:
            raise TypeError("contract must be exact FiniteVariationContract")
        validate_finite_variation_contract(contract)
        policy = FiniteOptionPromptProjectionPolicy(
            metadata_keys=(
                None
                if self.mode is PromptMetadataProjectionMode.ALL
                else self.included_metadata_keys
            )
        )
        expected = policy.project(contract)
        if expected.projection_sha256 != self.projection_sha256:
            raise ValueError("prompt projection differs from the sealed contract")


@dataclass(frozen=True, slots=True)
class FiniteOptionPromptProjectionPolicy:
    """Project all metadata or an explicit composition-root allowlist."""

    metadata_keys: tuple[str, ...] | None = None
    policy_id: str = POLICY_ID
    policy_version: int = POLICY_VERSION
    definition_sha256: str = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if self.metadata_keys is not None:
            _validate_metadata_keys(self.metadata_keys, name="metadata_keys")
        if self.policy_id != POLICY_ID:
            raise ValueError("policy_id drifted")
        if self.policy_version != POLICY_VERSION:
            raise ValueError("policy_version drifted")
        if self.definition_sha256 != POLICY_DEFINITION_SHA256:
            raise ValueError("definition_sha256 drifted")

    @property
    def mode(self) -> PromptMetadataProjectionMode:
        return (
            PromptMetadataProjectionMode.ALL
            if self.metadata_keys is None
            else PromptMetadataProjectionMode.ALLOWLIST
        )

    @property
    def configuration_sha256(self) -> str:
        self.__post_init__()
        return hashlib.sha256(
            _CONFIGURATION_DOMAIN
            + _canonical_json(
                {
                    "schema_version": 1,
                    "mode": self.mode.value,
                    "metadata_keys": (
                        None if self.metadata_keys is None else list(self.metadata_keys)
                    ),
                }
            )
        ).hexdigest()

    def project(
        self,
        contract: FiniteVariationContract,
    ) -> FiniteOptionPromptProjection:
        self.__post_init__()
        if type(contract) is not FiniteVariationContract:
            raise TypeError("contract must be exact FiniteVariationContract")
        validate_finite_variation_contract(contract)
        available = tuple(
            sorted({key for option in contract.options for key, _ in option.metadata})
        )
        included = available if self.metadata_keys is None else self.metadata_keys
        unknown = set(included) - set(available)
        if unknown:
            raise ValueError(
                "metadata allowlist names keys absent from the finite contract: "
                + ",".join(sorted(unknown))
            )
        included_set = set(included)
        records = tuple(
            FiniteOptionPromptRecord(
                option_id=option.option_id,
                family=option.family,
                description=option.description,
                metadata=tuple(
                    (key, value)
                    for key, value in option.metadata
                    if key in included_set
                ),
                source_option_identity_sha256=option.identity_sha256,
            )
            for option in contract.options
        )
        return FiniteOptionPromptProjection(
            source_contract_sha256=contract.identity_sha256,
            mode=self.mode,
            included_metadata_keys=included,
            available_metadata_keys=available,
            omitted_metadata_keys=tuple(sorted(set(available) - included_set)),
            records=records,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            policy_configuration_sha256=self.configuration_sha256,
        )


__all__ = [
    "FiniteOptionPromptProjection",
    "FiniteOptionPromptProjectionPolicy",
    "FiniteOptionPromptRecord",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
    "PromptMetadataProjectionMode",
]
