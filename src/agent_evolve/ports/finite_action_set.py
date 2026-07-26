"""Inverted port for compiling a small local finite-action support.

Benchmark code selects opaque option IDs from an already frozen parent-bound
catalog.  Trusted application code resolves the IDs, derives patches and
phenotypes, and seals the resulting authority.  Neither side receives current
candidate outcomes through this interface.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agent_evolve.domain.finite_action_set import (
    MAX_MATCHED_FINITE_ACTIONS,
    MIN_MATCHED_FINITE_ACTIONS,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_REQUEST_DOMAIN = b"agent-evolve:finite-action-set-compilation-request:v1\x00"
_DRAFT_DOMAIN = b"agent-evolve:finite-action-set-compilation-draft:v1\x00"


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


@dataclass(frozen=True, slots=True)
class FiniteActionSetCompilationRequest:
    """Outcome-free request to choose K IDs around one exact anchor."""

    parent_candidate_id: CandidateId
    finite_contract: FiniteVariationContract
    anchor_option_id: str
    anchor_option_identity_sha256: str
    exact_anchor_requirement_sha256: str
    card_reference: InsightRef
    card_content_sha256: str
    context_projection_sha256: str
    endpoint_definition_sha256: str
    required_cardinality: int
    current_outcome_access: bool = False
    request_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.parent_candidate_id) is not CandidateId:
            raise TypeError("parent_candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(self.parent_candidate_id)
        validate_finite_variation_contract(self.finite_contract)
        if type(self.anchor_option_id) is not str or not self.anchor_option_id:
            raise ValueError("anchor_option_id must be non-empty")
        anchor = self.finite_contract.resolve(self.anchor_option_id)
        require_sha256(
            self.anchor_option_identity_sha256,
            "anchor_option_identity_sha256",
        )
        if anchor.identity_sha256 != self.anchor_option_identity_sha256:
            raise ValueError("anchor option identity differs from the frozen contract")
        require_sha256(
            self.exact_anchor_requirement_sha256,
            "exact_anchor_requirement_sha256",
        )
        if type(self.card_reference) is not InsightRef:
            raise TypeError("card_reference must be an exact InsightRef")
        InsightRef.__post_init__(self.card_reference)
        for name in (
            "card_content_sha256",
            "context_projection_sha256",
            "endpoint_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.required_cardinality) is not int or not (
            MIN_MATCHED_FINITE_ACTIONS
            <= self.required_cardinality
            <= MAX_MATCHED_FINITE_ACTIONS
        ):
            raise ValueError(
                "required_cardinality must lie in "
                f"[{MIN_MATCHED_FINITE_ACTIONS},{MAX_MATCHED_FINITE_ACTIONS}]"
            )
        if self.required_cardinality > len(self.finite_contract.options):
            raise ValueError("source contract is smaller than the requested support")
        if type(self.current_outcome_access) is not bool:
            raise TypeError("current_outcome_access must be bool")
        if self.current_outcome_access:
            raise ValueError("finite action support compilation must be outcome-blind")
        object.__setattr__(
            self,
            "request_sha256",
            _hash(_REQUEST_DOMAIN, self.to_record()),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "parent_candidate_id": self.parent_candidate_id.value,
            "parent_configuration_sha256": (
                self.finite_contract.parent_configuration_sha256
            ),
            "finite_contract_sha256": self.finite_contract.identity_sha256,
            "anchor_option_id": self.anchor_option_id,
            "anchor_option_identity_sha256": self.anchor_option_identity_sha256,
            "exact_anchor_requirement_sha256": (
                self.exact_anchor_requirement_sha256
            ),
            "card_reference": {
                "insight_id": self.card_reference.insight_id.value,
                "version": self.card_reference.version,
            },
            "card_content_sha256": self.card_content_sha256,
            "context_projection_sha256": self.context_projection_sha256,
            "endpoint_definition_sha256": self.endpoint_definition_sha256,
            "required_cardinality": self.required_cardinality,
            "current_outcome_access": self.current_outcome_access,
        }


@dataclass(frozen=True, slots=True)
class FiniteActionSetDraft:
    """Untrusted compiler output: IDs and presentation policy, never children."""

    request_sha256: str
    ordered_option_ids: tuple[str, ...]
    anchor_option_id: str
    presentation_policy_id: str
    presentation_policy_version: int
    presentation_definition_sha256: str
    prompt_shape_sha256: str
    draft_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        if type(self.ordered_option_ids) is not tuple or not self.ordered_option_ids:
            raise ValueError("ordered_option_ids must be a non-empty exact tuple")
        if any(type(value) is not str or not value for value in self.ordered_option_ids):
            raise TypeError("ordered_option_ids must contain non-empty strings")
        if len(set(self.ordered_option_ids)) != len(self.ordered_option_ids):
            raise ValueError("ordered_option_ids must be unique")
        if type(self.anchor_option_id) is not str or self.anchor_option_id not in set(
            self.ordered_option_ids
        ):
            raise ValueError("anchor_option_id must occur in ordered_option_ids")
        _policy_identity(
            self.presentation_policy_id,
            self.presentation_policy_version,
            self.presentation_definition_sha256,
            name="presentation",
        )
        require_sha256(self.prompt_shape_sha256, "prompt_shape_sha256")
        object.__setattr__(
            self,
            "draft_sha256",
            _hash(_DRAFT_DOMAIN, self.to_record()),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "ordered_option_ids": list(self.ordered_option_ids),
            "anchor_option_id": self.anchor_option_id,
            "presentation_policy_id": self.presentation_policy_id,
            "presentation_policy_version": self.presentation_policy_version,
            "presentation_definition_sha256": self.presentation_definition_sha256,
            "prompt_shape_sha256": self.prompt_shape_sha256,
        }


@runtime_checkable
class FiniteActionSetCompiler(Protocol):
    policy_id: str
    policy_version: int
    definition_sha256: str

    def compile(
        self,
        request: FiniteActionSetCompilationRequest,
    ) -> FiniteActionSetDraft: ...


def validate_finite_action_set_compiler_identity(
    compiler: FiniteActionSetCompiler,
) -> tuple[str, int, str]:
    if not isinstance(compiler, FiniteActionSetCompiler):
        raise TypeError("compiler must implement FiniteActionSetCompiler")
    identity = (
        compiler.policy_id,
        compiler.policy_version,
        compiler.definition_sha256,
    )
    _policy_identity(*identity, name="support compiler")
    return identity


def validate_finite_action_set_draft(
    request: FiniteActionSetCompilationRequest,
    draft: FiniteActionSetDraft,
) -> None:
    if type(request) is not FiniteActionSetCompilationRequest:
        raise TypeError("request must be an exact FiniteActionSetCompilationRequest")
    FiniteActionSetCompilationRequest.__post_init__(request)
    if type(draft) is not FiniteActionSetDraft:
        raise TypeError("draft must be an exact FiniteActionSetDraft")
    FiniteActionSetDraft.__post_init__(draft)
    if draft.request_sha256 != request.request_sha256:
        raise ValueError("finite action draft is bound to a different request")
    if len(draft.ordered_option_ids) != request.required_cardinality:
        raise ValueError("finite action draft has the wrong cardinality")
    if draft.anchor_option_id != request.anchor_option_id:
        raise ValueError("finite action draft changed the exact anchor")
    for option_id in draft.ordered_option_ids:
        request.finite_contract.resolve(option_id)


__all__ = [
    "FiniteActionSetCompilationRequest",
    "FiniteActionSetCompiler",
    "FiniteActionSetDraft",
    "validate_finite_action_set_compiler_identity",
    "validate_finite_action_set_draft",
]
