"""Sealed finite, parent-relative variation choices.

The language model selects only an opaque option identifier.  Trusted benchmark
code owns each complete child configuration, so one option may change any
number of coordinates without expanding the model's proposal authority.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    _typed_json_sha256_canonical_bytes,
    canonical_typed_json_bytes,
    freeze_json,
    typed_json_sha256,
)


_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_METADATA_KEY = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_MAX_DESCRIPTION_BYTES = 2_048
_MAX_METADATA_ITEMS = 64
_MAX_METADATA_VALUE_BYTES = 4_096
MAX_FINITE_VARIATION_OPTIONS = 1_024
_OPTION_HASH_DOMAIN = b"agent-evolve:finite-variation-option:v1\x00"
_CONTRACT_HASH_DOMAIN = b"agent-evolve:finite-variation-contract:v1\x00"
_ACTION_EVIDENCE_BINDING_HASH_DOMAIN = (
    b"agent-evolve:finite-action-evidence-binding:v1\x00"
)


def _frame(payload: bytes) -> bytes:
    if type(payload) is not bytes:
        raise TypeError("framed payloads must be exact bytes")
    return len(payload).to_bytes(8, "big", signed=False) + payload


def _prompt_text(value: str, *, name: str, max_bytes: int) -> bytes:
    """Admit bounded, canonical, single-line printable prompt data."""

    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be canonical non-empty text")
    try:
        encoded = value.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain printable ASCII only") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{name} exceeds its byte limit")
    if any(byte < 0x20 or byte > 0x7E for byte in encoded):
        raise ValueError(f"{name} must be single-line printable ASCII")
    return encoded


@dataclass(frozen=True, slots=True, eq=False)
class FiniteVariationOption:
    """One immutable full-child choice bound to one exact parent phenotype.

    ``description`` and ``metadata`` are the only option data intended for an
    LLM prompt.  The child configuration remains engine-owned and is never a
    model-authored payload.
    """

    option_id: str
    parent_configuration_sha256: str
    child_configuration: FrozenJsonObject
    family: str
    description: str
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or _OPTION_ID.fullmatch(
            self.option_id
        ) is None:
            raise ValueError(
                "option_id must use the closed lowercase identifier grammar"
            )
        require_sha256(
            self.parent_configuration_sha256,
            "parent_configuration_sha256",
        )
        if type(self.child_configuration) is not FrozenJsonObject:
            raise TypeError("child_configuration must be an exact FrozenJsonObject")
        if freeze_json(self.child_configuration) is not self.child_configuration:
            raise TypeError("child_configuration must already be frozen typed JSON")
        if (
            typed_json_sha256(self.child_configuration)
            == self.parent_configuration_sha256
        ):
            raise ValueError("a finite variation option must change its parent")
        if type(self.family) is not str or _TOKEN.fullmatch(self.family) is None:
            raise ValueError("family must use the closed lowercase token grammar")
        _prompt_text(
            self.description,
            name="description",
            max_bytes=_MAX_DESCRIPTION_BYTES,
        )
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
            _prompt_text(
                value,
                name=f"metadata value for {key!r}",
                max_bytes=_MAX_METADATA_VALUE_BYTES,
            )
            keys.append(key)
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError("metadata keys must be unique and canonically sorted")

    @property
    def child_configuration_sha256(self) -> str:
        validate_finite_variation_option(self)
        return typed_json_sha256(self.child_configuration)

    @property
    def identity_sha256(self) -> str:
        """Bind parent, full child, family, and all prompt-visible semantics."""

        validate_finite_variation_option(self)
        digest = hashlib.sha256()
        digest.update(_OPTION_HASH_DOMAIN)
        digest.update(_frame(self.option_id.encode("ascii")))
        digest.update(bytes.fromhex(self.parent_configuration_sha256))
        digest.update(_frame(canonical_typed_json_bytes(self.child_configuration)))
        digest.update(_frame(self.family.encode("ascii")))
        digest.update(_frame(self.description.encode("ascii")))
        digest.update(len(self.metadata).to_bytes(8, "big", signed=False))
        for key, value in self.metadata:
            digest.update(_frame(key.encode("ascii")))
            digest.update(_frame(value.encode("ascii")))
        return digest.hexdigest()

    def prompt_record(self) -> dict[str, object]:
        """Return bounded option facts without exposing the sealed child."""

        validate_finite_variation_option(self)
        return {
            "option_id": self.option_id,
            "family": self.family,
            "description": self.description,
            "metadata": {key: value for key, value in self.metadata},
        }

    def evidence_record(self) -> dict[str, object]:
        """Return replay identity without copying the full child tree."""

        return {
            **self.prompt_record(),
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "option_identity_sha256": self.identity_sha256,
        }

    def _validated_values(self) -> tuple[object, ...]:
        validate_finite_variation_option(self)
        return (
            self.option_id,
            self.parent_configuration_sha256,
            canonical_typed_json_bytes(self.child_configuration),
            self.family,
            self.description,
            self.metadata,
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not FiniteVariationOption or type(
            other
        ) is not FiniteVariationOption:
            return False
        return self._validated_values() == other._validated_values()

    def __hash__(self) -> int:
        return hash((FiniteVariationOption, self._validated_values()))


@dataclass(frozen=True, slots=True)
class FiniteActionEvidenceBinding:
    """Replay-safe link from one reflection contrast to its executed action.

    The binding deliberately carries only opaque finite-option identity and
    prompt-safe taxonomy.  It neither copies a benchmark configuration nor
    assumes that a later portfolio uses the same parent-bound contract.
    ``option_identity_sha256`` and ``contract_identity_sha256`` retain the
    exact originating action and palette for trusted replay.
    """

    contrast_id: str
    option_id: str
    family: str
    option_identity_sha256: str
    contract_identity_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.contrast_id, "contrast_id")
        if type(self.option_id) is not str or _OPTION_ID.fullmatch(
            self.option_id
        ) is None:
            raise ValueError(
                "option_id must use the closed lowercase identifier grammar"
            )
        if type(self.family) is not str or _TOKEN.fullmatch(self.family) is None:
            raise ValueError("family must use the closed lowercase token grammar")
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        require_sha256(self.contract_identity_sha256, "contract_identity_sha256")

    def _identity_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "contrast_id": self.contrast_id,
            "option_id": self.option_id,
            "family": self.family,
            "option_identity_sha256": self.option_identity_sha256,
            "contract_identity_sha256": self.contract_identity_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        """Bind the source contrast and exact originating finite action."""

        return hashlib.sha256(
            _ACTION_EVIDENCE_BINDING_HASH_DOMAIN
            + canonical_typed_json_bytes(freeze_json(self._identity_record()))
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        """Return a framework-neutral portfolio/evidence projection."""

        return {
            **self._identity_record(),
            "binding_identity_sha256": self.identity_sha256,
        }

    def finite_option_record(self) -> dict[str, object]:
        """Reproduce the prompt-safe attribution nested in a contrast row."""

        self.__post_init__()
        return {
            "option_id": self.option_id,
            "family": self.family,
            "option_identity_sha256": self.option_identity_sha256,
            "contract_identity_sha256": self.contract_identity_sha256,
        }


def validate_finite_variation_option(option: FiniteVariationOption) -> None:
    """Revalidate an exact option at a public trust boundary."""

    if type(option) is not FiniteVariationOption:
        raise TypeError("option must be an exact FiniteVariationOption")
    FiniteVariationOption.__post_init__(option)


def _validated_option_identity_material(
    option: FiniteVariationOption,
) -> tuple[bytes, str, str]:
    """Validate once and derive child bytes, child digest, and option digest."""

    validate_finite_variation_option(option)
    child_bytes = canonical_typed_json_bytes(option.child_configuration)
    child_sha256 = _typed_json_sha256_canonical_bytes(child_bytes)
    digest = hashlib.sha256()
    digest.update(_OPTION_HASH_DOMAIN)
    digest.update(_frame(option.option_id.encode("ascii")))
    digest.update(bytes.fromhex(option.parent_configuration_sha256))
    digest.update(_frame(child_bytes))
    digest.update(_frame(option.family.encode("ascii")))
    digest.update(_frame(option.description.encode("ascii")))
    digest.update(len(option.metadata).to_bytes(8, "big", signed=False))
    for key, value in option.metadata:
        digest.update(_frame(key.encode("ascii")))
        digest.update(_frame(value.encode("ascii")))
    return child_bytes, child_sha256, digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ValidatedFiniteVariationIdentityIndex:
    """Immutable identities derived by one complete contract validation pass."""

    parent_configuration_sha256: str
    contract_identity_sha256: str
    option_ids: tuple[str, ...]
    option_identity_sha256s: tuple[str, ...]
    child_configuration_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha256(
            self.parent_configuration_sha256,
            "parent_configuration_sha256",
        )
        require_sha256(
            self.contract_identity_sha256,
            "contract_identity_sha256",
        )
        count = len(self.option_ids)
        if count == 0 or any(
            type(value) is not str or _OPTION_ID.fullmatch(value) is None
            for value in self.option_ids
        ):
            raise ValueError("option_ids must be a non-empty exact identifier tuple")
        for name in (
            "option_identity_sha256s",
            "child_configuration_sha256s",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != count:
                raise ValueError(f"{name} must cover every indexed option")
            for index, value in enumerate(values):
                require_sha256(value, f"{name}[{index}]")
        if len(set(self.option_ids)) != count:
            raise ValueError("indexed option IDs must be unique")
        if len(set(self.option_identity_sha256s)) != count:
            raise ValueError("indexed option identities must be unique")
        if len(set(self.child_configuration_sha256s)) != count:
            raise ValueError("indexed child configurations must be unique")


@dataclass(frozen=True, slots=True, eq=False)
class FiniteVariationContract:
    """One catalog snapshot sealed against one immutable parent configuration."""

    catalog_id: str
    catalog_version: int
    catalog_definition_sha256: str
    parent_configuration: FrozenJsonObject
    options: tuple[FiniteVariationOption, ...]

    def __post_init__(self) -> None:
        _validated_finite_variation_identity_index(self)

    @property
    def parent_configuration_sha256(self) -> str:
        return _validated_finite_variation_identity_index(
            self
        ).parent_configuration_sha256

    @property
    def identity_sha256(self) -> str:
        """Bind catalog semantics, parent identity, option order, and all children."""

        return _validated_finite_variation_identity_index(
            self
        ).contract_identity_sha256

    def resolve(self, option_id: str) -> FiniteVariationOption:
        """Resolve exactly one selected ID without accepting aliases."""

        validate_finite_variation_contract(self)
        if type(option_id) is not str:
            raise TypeError("option_id must be an exact string")
        matches = tuple(
            option for option in self.options if option.option_id == option_id
        )
        if len(matches) != 1:
            raise ValueError("selected option_id is outside the sealed contract")
        return matches[0]

    def prompt_records(self) -> tuple[dict[str, object], ...]:
        """Return the exact ordered, prompt-safe action palette."""

        validate_finite_variation_contract(self)
        return tuple(option.prompt_record() for option in self.options)

    def evidence_record(self) -> dict[str, object]:
        validate_finite_variation_contract(self)
        return {
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "catalog_definition_sha256": self.catalog_definition_sha256,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "contract_identity_sha256": self.identity_sha256,
            "options": [option.evidence_record() for option in self.options],
        }

    def _validated_values(self) -> tuple[object, ...]:
        validate_finite_variation_contract(self)
        return (
            self.catalog_id,
            self.catalog_version,
            self.catalog_definition_sha256,
            canonical_typed_json_bytes(self.parent_configuration),
            tuple(option._validated_values() for option in self.options),
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not FiniteVariationContract or type(
            other
        ) is not FiniteVariationContract:
            return False
        return self._validated_values() == other._validated_values()

    def __hash__(self) -> int:
        return hash((FiniteVariationContract, self._validated_values()))


def _validated_finite_variation_identity_index(
    contract: FiniteVariationContract,
) -> ValidatedFiniteVariationIdentityIndex:
    """Validate one contract graph and derive every repeated identity once."""

    if type(contract) is not FiniteVariationContract:
        raise TypeError("contract must be an exact FiniteVariationContract")
    if type(contract.catalog_id) is not str or _TOKEN.fullmatch(
        contract.catalog_id
    ) is None:
        raise ValueError("catalog_id must use the closed lowercase token grammar")
    if type(contract.catalog_version) is not int or contract.catalog_version <= 0:
        raise ValueError("catalog_version must be a positive exact integer")
    require_sha256(
        contract.catalog_definition_sha256,
        "catalog_definition_sha256",
    )
    if type(contract.parent_configuration) is not FrozenJsonObject:
        raise TypeError("parent_configuration must be an exact FrozenJsonObject")
    parent_bytes = canonical_typed_json_bytes(contract.parent_configuration)
    parent_sha256 = _typed_json_sha256_canonical_bytes(parent_bytes)
    if type(contract.options) is not tuple:
        raise TypeError("options must be an exact tuple")
    if not contract.options:
        raise ValueError("a finite variation contract requires at least one option")
    if len(contract.options) > MAX_FINITE_VARIATION_OPTIONS:
        raise ValueError("finite variation contract exceeds its option limit")

    option_ids: list[str] = []
    option_hashes: list[str] = []
    child_hashes: list[str] = []
    for option in contract.options:
        _child_bytes, child_sha256, option_sha256 = (
            _validated_option_identity_material(option)
        )
        if option.parent_configuration_sha256 != parent_sha256:
            raise ValueError("option is bound to a different parent configuration")
        # The option validator already rejects a child whose typed digest is
        # its declared parent digest.  Reusing that validated digest avoids a
        # second full structural comparison of large frozen configurations.
        option_ids.append(option.option_id)
        option_hashes.append(option_sha256)
        child_hashes.append(child_sha256)
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("finite variation option IDs must be unique")
    if len(set(option_hashes)) != len(option_hashes):
        raise ValueError("finite variation option identities must be unique")
    if len(set(child_hashes)) != len(child_hashes):
        raise ValueError("finite variation options must materialize distinct children")

    digest = hashlib.sha256()
    digest.update(_CONTRACT_HASH_DOMAIN)
    digest.update(_frame(contract.catalog_id.encode("ascii")))
    digest.update(contract.catalog_version.to_bytes(8, "big", signed=False))
    digest.update(bytes.fromhex(contract.catalog_definition_sha256))
    digest.update(_frame(parent_bytes))
    digest.update(len(contract.options).to_bytes(8, "big", signed=False))
    for option_sha256 in option_hashes:
        digest.update(bytes.fromhex(option_sha256))
    return ValidatedFiniteVariationIdentityIndex(
        parent_configuration_sha256=parent_sha256,
        contract_identity_sha256=digest.hexdigest(),
        option_ids=tuple(option_ids),
        option_identity_sha256s=tuple(option_hashes),
        child_configuration_sha256s=tuple(child_hashes),
    )


def validated_finite_variation_identity_index(
    contract: FiniteVariationContract,
) -> ValidatedFiniteVariationIdentityIndex:
    """Return an immutable index from one complete public validation pass."""

    return _validated_finite_variation_identity_index(contract)


def validate_finite_variation_contract(contract: FiniteVariationContract) -> None:
    """Revalidate an exact contract and its complete option graph."""

    if type(contract) is not FiniteVariationContract:
        raise TypeError("contract must be an exact FiniteVariationContract")
    FiniteVariationContract.__post_init__(contract)


def bind_finite_action_evidence(
    *,
    contrast_id: str,
    contract: FiniteVariationContract,
    option_id: str,
) -> FiniteActionEvidenceBinding:
    """Issue one action-evidence binding only from a sealed contract option."""

    validate_finite_variation_contract(contract)
    option = contract.resolve(option_id)
    return FiniteActionEvidenceBinding(
        contrast_id=contrast_id,
        option_id=option.option_id,
        family=option.family,
        option_identity_sha256=option.identity_sha256,
        contract_identity_sha256=contract.identity_sha256,
    )


__all__ = [
    "FiniteActionEvidenceBinding",
    "FiniteVariationContract",
    "ValidatedFiniteVariationIdentityIndex",
    "FiniteVariationOption",
    "MAX_FINITE_VARIATION_OPTIONS",
    "bind_finite_action_evidence",
    "validate_finite_variation_contract",
    "validate_finite_variation_option",
    "validated_finite_variation_identity_index",
]
