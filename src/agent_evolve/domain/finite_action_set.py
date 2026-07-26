"""Immutable, outcome-blind finite action sets for matched selector tests.

The exact recommendation attached to an insight remains an anchor.  A finite
action set is a separate experimental authority that exposes a small local
neighbourhood around that anchor to both a model and an engine comparator.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
    validate_finite_variation_contract,
    validate_finite_variation_option,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256


MIN_MATCHED_FINITE_ACTIONS = 4
MAX_MATCHED_FINITE_ACTIONS = 8

_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_PATH = re.compile(
    r"^\$\.[^.\[\]\s]+(?:\.[^.\[\]\s]+|\[(?:0|[1-9][0-9]*)\])*$"
)
_OPTION_PROMPT_DOMAIN = b"agent-evolve:finite-action-option-prompt:v1\x00"
_PRESENTATION_DOMAIN = b"agent-evolve:finite-action-presentation:v1\x00"
_SUPPORT_DOMAIN = b"agent-evolve:finite-action-support:v1\x00"
_CARD_DOMAIN = b"agent-evolve:finite-action-card:v1\x00"
_AUTHORITY_DOMAIN = b"agent-evolve:finite-action-set-authority:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _policy_identity(
    policy_id: str,
    policy_version: int,
    definition_sha256: str,
    *,
    name: str,
) -> None:
    if type(policy_id) is not str or _TOKEN.fullmatch(policy_id) is None:
        raise ValueError(f"{name} policy_id must use the token grammar")
    if type(policy_version) is not int or policy_version <= 0:
        raise ValueError(f"{name} policy_version must be positive")
    require_sha256(definition_sha256, f"{name} definition_sha256")


def _canonical_paths(values: tuple[str, ...]) -> None:
    if type(values) is not tuple or not values:
        raise ValueError("changed_paths must be a non-empty exact tuple")
    if any(type(value) is not str or _PATH.fullmatch(value) is None for value in values):
        raise TypeError("changed_paths must contain canonical rooted JSON paths")
    if values != tuple(sorted(set(values))):
        raise ValueError("changed_paths must be unique and canonical")


class FiniteActionSourceMode(str, Enum):
    COMPILED_ACTIVE_CARD = "compiled_active_card"
    COMPILED_SHUFFLED_CARD = "compiled_shuffled_card"
    EVIDENCE_FREE_CARD = "evidence_free_card"


@dataclass(frozen=True, slots=True)
class FiniteActionOptionAuthority:
    """One full child plus its path, phenotype, and prompt identities."""

    option: FiniteVariationOption
    changed_paths: tuple[str, ...]
    phenotype_policy_id: str
    phenotype_policy_version: int
    phenotype_identity_sha256: str
    prompt_record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.option) is not FiniteVariationOption:
            raise TypeError("option must be an exact FiniteVariationOption")
        validate_finite_variation_option(self.option)
        _canonical_paths(self.changed_paths)
        if type(self.phenotype_policy_id) is not str or _TOKEN.fullmatch(
            self.phenotype_policy_id
        ) is None:
            raise ValueError("phenotype_policy_id must use the token grammar")
        if type(self.phenotype_policy_version) is not int or self.phenotype_policy_version <= 0:
            raise ValueError("phenotype_policy_version must be positive")
        require_sha256(
            self.phenotype_identity_sha256,
            "phenotype_identity_sha256",
        )
        object.__setattr__(
            self,
            "prompt_record_sha256",
            _hash(_OPTION_PROMPT_DOMAIN, self.option.prompt_record()),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "option": self.option.evidence_record(),
            "changed_paths": list(self.changed_paths),
            "phenotype_policy_id": self.phenotype_policy_id,
            "phenotype_policy_version": self.phenotype_policy_version,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "prompt_record_sha256": self.prompt_record_sha256,
        }


@dataclass(frozen=True, slots=True)
class FiniteActionPresentationAuthority:
    """Exact provider-visible ordering and structural prompt commitment."""

    policy_id: str
    policy_version: int
    definition_sha256: str
    ordered_option_ids: tuple[str, ...]
    ordered_prompt_record_sha256s: tuple[str, ...]
    prompt_shape_sha256: str
    presentation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _policy_identity(
            self.policy_id,
            self.policy_version,
            self.definition_sha256,
            name="presentation",
        )
        count = len(self.ordered_option_ids)
        if type(self.ordered_option_ids) is not tuple or not self.ordered_option_ids:
            raise ValueError("ordered_option_ids must be a non-empty exact tuple")
        if any(type(value) is not str or not value for value in self.ordered_option_ids):
            raise TypeError("ordered_option_ids must contain non-empty strings")
        if len(set(self.ordered_option_ids)) != count:
            raise ValueError("ordered_option_ids must be unique")
        if (
            type(self.ordered_prompt_record_sha256s) is not tuple
            or len(self.ordered_prompt_record_sha256s) != count
        ):
            raise ValueError("prompt-record identities must cover every option")
        for value in self.ordered_prompt_record_sha256s:
            require_sha256(value, "ordered_prompt_record_sha256")
        require_sha256(self.prompt_shape_sha256, "prompt_shape_sha256")
        object.__setattr__(
            self,
            "presentation_sha256",
            _hash(_PRESENTATION_DOMAIN, self.to_record()),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "ordered_option_ids": list(self.ordered_option_ids),
            "ordered_prompt_record_sha256s": list(
                self.ordered_prompt_record_sha256s
            ),
            "prompt_shape_sha256": self.prompt_shape_sha256,
        }


@dataclass(frozen=True, slots=True)
class FiniteActionSupportAuthority:
    """The exact K children shared by every matched selector on one parent."""

    parent_candidate_id: CandidateId
    source_contract_sha256: str
    support_contract: FiniteVariationContract
    endpoint_definition_sha256: str
    context_projection_sha256: str
    options: tuple[FiniteActionOptionAuthority, ...]
    anchor_option_id: str
    presentation: FiniteActionPresentationAuthority
    compatible_option_count: int
    support_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.parent_candidate_id) is not CandidateId:
            raise TypeError("parent_candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(self.parent_candidate_id)
        require_sha256(self.source_contract_sha256, "source_contract_sha256")
        validate_finite_variation_contract(self.support_contract)
        for name in ("endpoint_definition_sha256", "context_projection_sha256"):
            require_sha256(getattr(self, name), name)
        count = len(self.options)
        if type(self.options) is not tuple or not (
            MIN_MATCHED_FINITE_ACTIONS <= count <= MAX_MATCHED_FINITE_ACTIONS
        ):
            raise ValueError(
                "finite action support cardinality must lie in "
                f"[{MIN_MATCHED_FINITE_ACTIONS},{MAX_MATCHED_FINITE_ACTIONS}]"
            )
        if any(type(value) is not FiniteActionOptionAuthority for value in self.options):
            raise TypeError("options must contain exact option authorities")
        for value in self.options:
            FiniteActionOptionAuthority.__post_init__(value)
        if tuple(value.option for value in self.options) != self.support_contract.options:
            raise ValueError("option authorities differ from the support contract")
        if len({value.option.option_id for value in self.options}) != count:
            raise ValueError("support option IDs must be unique")
        if len({value.option.identity_sha256 for value in self.options}) != count:
            raise ValueError("support option identities must be unique")
        if len({value.option.child_configuration_sha256 for value in self.options}) != count:
            raise ValueError("support children must be unique")
        if len({value.phenotype_identity_sha256 for value in self.options}) != count:
            raise ValueError("support phenotypes must be unique")
        if type(self.anchor_option_id) is not str or self.anchor_option_id not in {
            value.option.option_id for value in self.options
        }:
            raise ValueError("anchor_option_id must identify exactly one support option")
        if type(self.presentation) is not FiniteActionPresentationAuthority:
            raise TypeError("presentation must be exact")
        FiniteActionPresentationAuthority.__post_init__(self.presentation)
        if self.presentation.ordered_option_ids != tuple(
            value.option.option_id for value in self.options
        ):
            raise ValueError("presentation order differs from the support")
        if self.presentation.ordered_prompt_record_sha256s != tuple(
            value.prompt_record_sha256 for value in self.options
        ):
            raise ValueError("presentation prompt records differ from the support")
        if type(self.compatible_option_count) is not int or self.compatible_option_count != count:
            raise ValueError("every support option must pass prospective compatibility")
        object.__setattr__(
            self,
            "support_sha256",
            _hash(_SUPPORT_DOMAIN, self.to_record()),
        )

    @property
    def cardinality(self) -> int:
        return len(self.options)

    @property
    def parent_configuration_sha256(self) -> str:
        return self.support_contract.parent_configuration_sha256

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "parent_candidate_id": self.parent_candidate_id.value,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "source_contract_sha256": self.source_contract_sha256,
            "support_contract": self.support_contract.evidence_record(),
            "endpoint_definition_sha256": self.endpoint_definition_sha256,
            "context_projection_sha256": self.context_projection_sha256,
            "options": [value.to_record() for value in self.options],
            "anchor_option_id": self.anchor_option_id,
            "presentation": {
                **self.presentation.to_record(),
                "presentation_sha256": self.presentation.presentation_sha256,
            },
            "compatible_option_count": self.compatible_option_count,
        }


@dataclass(frozen=True, slots=True)
class FiniteActionCardAuthority:
    """Immutable card provenance; K neighbours never become recommendations."""

    source_mode: FiniteActionSourceMode
    reference: InsightRef
    card_content_sha256: str
    registered_source_evidence_sha256: str | None
    exact_anchor_requirement_sha256: str
    compilation_request_sha256: str | None
    compilation_receipt_sha256: str | None
    executable_spec_sha256: str | None
    prompt_card_record_sha256: str
    card_authority_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.source_mode) is not FiniteActionSourceMode:
            raise TypeError("source_mode must be an exact FiniteActionSourceMode")
        if type(self.reference) is not InsightRef:
            raise TypeError("reference must be an exact InsightRef")
        InsightRef.__post_init__(self.reference)
        for name in (
            "card_content_sha256",
            "exact_anchor_requirement_sha256",
            "prompt_card_record_sha256",
        ):
            require_sha256(getattr(self, name), name)
        optional = (
            self.registered_source_evidence_sha256,
            self.compilation_request_sha256,
            self.compilation_receipt_sha256,
            self.executable_spec_sha256,
        )
        if self.source_mode is FiniteActionSourceMode.EVIDENCE_FREE_CARD:
            if any(value is not None for value in optional[1:]):
                raise ValueError("evidence-free card cannot claim compiler evidence")
        else:
            if any(value is None for value in optional):
                raise ValueError("compiled card authority requires complete evidence")
        for value in optional:
            if value is not None:
                require_sha256(value, "optional card evidence SHA-256")
        object.__setattr__(
            self,
            "card_authority_sha256",
            _hash(_CARD_DOMAIN, self.to_record()),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_mode": self.source_mode.value,
            "reference": {
                "insight_id": self.reference.insight_id.value,
                "version": self.reference.version,
            },
            "card_content_sha256": self.card_content_sha256,
            "registered_source_evidence_sha256": (
                self.registered_source_evidence_sha256
            ),
            "exact_anchor_requirement_sha256": (
                self.exact_anchor_requirement_sha256
            ),
            "compilation_request_sha256": self.compilation_request_sha256,
            "compilation_receipt_sha256": self.compilation_receipt_sha256,
            "executable_spec_sha256": self.executable_spec_sha256,
            "prompt_card_record_sha256": self.prompt_card_record_sha256,
        }


@dataclass(frozen=True, slots=True)
class FiniteActionSetAuthority:
    """Card plus matched support, sealed before any current candidate outcome."""

    support: FiniteActionSupportAuthority
    card: FiniteActionCardAuthority
    support_compilation_request_sha256: str
    support_compilation_draft_sha256: str
    support_compiler_policy_id: str
    support_compiler_policy_version: int
    support_compiler_definition_sha256: str
    current_outcome_access: bool = False
    authority_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.support) is not FiniteActionSupportAuthority:
            raise TypeError("support must be exact")
        FiniteActionSupportAuthority.__post_init__(self.support)
        if type(self.card) is not FiniteActionCardAuthority:
            raise TypeError("card must be exact")
        FiniteActionCardAuthority.__post_init__(self.card)
        require_sha256(
            self.support_compilation_request_sha256,
            "support_compilation_request_sha256",
        )
        require_sha256(
            self.support_compilation_draft_sha256,
            "support_compilation_draft_sha256",
        )
        _policy_identity(
            self.support_compiler_policy_id,
            self.support_compiler_policy_version,
            self.support_compiler_definition_sha256,
            name="support compiler",
        )
        if type(self.current_outcome_access) is not bool:
            raise TypeError("current_outcome_access must be bool")
        if self.current_outcome_access:
            raise ValueError("finite action set authority must be outcome-blind")
        object.__setattr__(
            self,
            "authority_sha256",
            _hash(_AUTHORITY_DOMAIN, self.to_record()),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "support": {
                **self.support.to_record(),
                "support_sha256": self.support.support_sha256,
            },
            "card": {
                **self.card.to_record(),
                "card_authority_sha256": self.card.card_authority_sha256,
            },
            "support_compilation_request_sha256": (
                self.support_compilation_request_sha256
            ),
            "support_compilation_draft_sha256": (
                self.support_compilation_draft_sha256
            ),
            "support_compiler_policy_id": self.support_compiler_policy_id,
            "support_compiler_policy_version": self.support_compiler_policy_version,
            "support_compiler_definition_sha256": (
                self.support_compiler_definition_sha256
            ),
            "current_outcome_access": self.current_outcome_access,
        }


__all__ = [
    "FiniteActionCardAuthority",
    "FiniteActionOptionAuthority",
    "FiniteActionPresentationAuthority",
    "FiniteActionSetAuthority",
    "FiniteActionSourceMode",
    "FiniteActionSupportAuthority",
    "MAX_MATCHED_FINITE_ACTIONS",
    "MIN_MATCHED_FINITE_ACTIONS",
]
