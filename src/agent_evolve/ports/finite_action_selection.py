"""Matched decision contracts for model and prospective engine selectors."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from agent_evolve.domain.finite_action_set import FiniteActionSetAuthority
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_RANK_TOKEN_DOMAIN = b"agent-evolve:prospective-uniform-rank-token:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:finite-action-decision:v1\x00"


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
) -> None:
    if type(policy_id) is not str or _TOKEN.fullmatch(policy_id) is None:
        raise ValueError("selector policy_id must use the token grammar")
    if type(policy_version) is not int or policy_version <= 0:
        raise ValueError("selector policy_version must be positive")
    require_sha256(definition_sha256, "selector definition_sha256")


class FiniteActionSelectorKind(str, Enum):
    MODEL = "model"
    ENGINE = "engine"


@dataclass(frozen=True, slots=True)
class ProspectiveUniformRankToken:
    """One engine ordinal committed before current-run candidate outcomes."""

    task_sha256: str
    authority_sha256: str
    support_sha256: str
    card_reference: InsightRef
    card_content_sha256: str
    cardinality: int
    selected_ordinal: int
    schedule_policy_id: str
    schedule_policy_version: int
    schedule_definition_sha256: str
    schedule_state_sha256: str
    pre_outcome_phase_commit_sha256: str
    current_outcome_access: bool = False
    token_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "task_sha256",
            "authority_sha256",
            "support_sha256",
            "card_content_sha256",
            "schedule_definition_sha256",
            "schedule_state_sha256",
            "pre_outcome_phase_commit_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.card_reference) is not InsightRef:
            raise TypeError("card_reference must be an exact InsightRef")
        InsightRef.__post_init__(self.card_reference)
        if type(self.cardinality) is not int or self.cardinality <= 0:
            raise ValueError("cardinality must be positive")
        if type(self.selected_ordinal) is not int or not (
            0 <= self.selected_ordinal < self.cardinality
        ):
            raise ValueError("selected_ordinal is outside the support")
        _policy_identity(
            self.schedule_policy_id,
            self.schedule_policy_version,
            self.schedule_definition_sha256,
        )
        if type(self.current_outcome_access) is not bool:
            raise TypeError("current_outcome_access must be bool")
        if self.current_outcome_access:
            raise ValueError("prospective engine rank cannot access current outcomes")
        object.__setattr__(
            self,
            "token_sha256",
            _hash(_RANK_TOKEN_DOMAIN, self.to_record()),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "task_sha256": self.task_sha256,
            "authority_sha256": self.authority_sha256,
            "support_sha256": self.support_sha256,
            "card_reference": {
                "insight_id": self.card_reference.insight_id.value,
                "version": self.card_reference.version,
            },
            "card_content_sha256": self.card_content_sha256,
            "cardinality": self.cardinality,
            "selected_ordinal": self.selected_ordinal,
            "schedule_policy_id": self.schedule_policy_id,
            "schedule_policy_version": self.schedule_policy_version,
            "schedule_definition_sha256": self.schedule_definition_sha256,
            "schedule_state_sha256": self.schedule_state_sha256,
            "pre_outcome_phase_commit_sha256": (
                self.pre_outcome_phase_commit_sha256
            ),
            "current_outcome_access": self.current_outcome_access,
        }


@dataclass(frozen=True, slots=True)
class EngineFiniteActionRequest:
    authority: FiniteActionSetAuthority
    prospective_rank: ProspectiveUniformRankToken

    def __post_init__(self) -> None:
        if type(self.authority) is not FiniteActionSetAuthority:
            raise TypeError("authority must be an exact FiniteActionSetAuthority")
        FiniteActionSetAuthority.__post_init__(self.authority)
        if type(self.prospective_rank) is not ProspectiveUniformRankToken:
            raise TypeError("prospective_rank must be an exact token")
        ProspectiveUniformRankToken.__post_init__(self.prospective_rank)
        token = self.prospective_rank
        expected = (
            self.authority.authority_sha256,
            self.authority.support.support_sha256,
            self.authority.card.reference,
            self.authority.card.card_content_sha256,
            self.authority.support.cardinality,
        )
        observed = (
            token.authority_sha256,
            token.support_sha256,
            token.card_reference,
            token.card_content_sha256,
            token.cardinality,
        )
        if observed != expected:
            raise ValueError("prospective rank token differs from its authority")


@dataclass(frozen=True, slots=True)
class FiniteActionDecision:
    """Outcome-free exact selection from one authenticated K-option support."""

    authority_sha256: str
    support_sha256: str
    selector_kind: FiniteActionSelectorKind
    selected_ordinal: int
    option_id: str
    option_identity_sha256: str
    child_configuration_sha256: str
    phenotype_identity_sha256: str
    selector_policy_id: str
    selector_policy_version: int
    selector_definition_sha256: str
    prospective_token_sha256: str | None
    model_call_id: LLMCallId | None
    model_prompt_sha256: str | None
    model_telemetry_sha256: str | None
    propensity_numerator: int | None
    propensity_denominator: int | None
    current_outcome_access: bool = False
    decision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "authority_sha256",
            "support_sha256",
            "option_identity_sha256",
            "child_configuration_sha256",
            "phenotype_identity_sha256",
            "selector_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.selector_kind) is not FiniteActionSelectorKind:
            raise TypeError("selector_kind must be an exact FiniteActionSelectorKind")
        if type(self.selected_ordinal) is not int or self.selected_ordinal < 0:
            raise ValueError("selected_ordinal must be non-negative")
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        _policy_identity(
            self.selector_policy_id,
            self.selector_policy_version,
            self.selector_definition_sha256,
        )
        if type(self.current_outcome_access) is not bool:
            raise TypeError("current_outcome_access must be bool")
        if self.current_outcome_access:
            raise ValueError("finite action decision cannot access current outcomes")
        if self.selector_kind is FiniteActionSelectorKind.ENGINE:
            if self.prospective_token_sha256 is None:
                raise ValueError("engine decision requires a prospective rank token")
            require_sha256(
                self.prospective_token_sha256,
                "prospective_token_sha256",
            )
            if (
                self.model_call_id is not None
                or self.model_prompt_sha256 is not None
                or self.model_telemetry_sha256 is not None
            ):
                raise ValueError("engine decision cannot claim model evidence")
            if (
                type(self.propensity_numerator) is not int
                or type(self.propensity_denominator) is not int
                or self.propensity_numerator != 1
                or self.propensity_denominator <= 0
            ):
                raise ValueError("prospective uniform decision requires propensity 1/K")
        else:
            if self.prospective_token_sha256 is not None:
                raise ValueError("model decision cannot claim an engine rank token")
            if type(self.model_call_id) is not LLMCallId:
                raise ValueError("model decision requires an exact model_call_id")
            LLMCallId.__post_init__(self.model_call_id)
            if self.model_prompt_sha256 is None:
                raise ValueError("model decision requires its exact prompt identity")
            require_sha256(self.model_prompt_sha256, "model_prompt_sha256")
            if self.model_telemetry_sha256 is None:
                raise ValueError("model decision requires authenticated telemetry")
            require_sha256(self.model_telemetry_sha256, "model_telemetry_sha256")
            if self.propensity_numerator is not None or self.propensity_denominator is not None:
                raise ValueError("non-randomized model decision has unknown propensity")
        object.__setattr__(
            self,
            "decision_sha256",
            _hash(_DECISION_DOMAIN, self.to_record()),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "authority_sha256": self.authority_sha256,
            "support_sha256": self.support_sha256,
            "selector_kind": self.selector_kind.value,
            "selected_ordinal": self.selected_ordinal,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "phenotype_identity_sha256": self.phenotype_identity_sha256,
            "selector_policy_id": self.selector_policy_id,
            "selector_policy_version": self.selector_policy_version,
            "selector_definition_sha256": self.selector_definition_sha256,
            "prospective_token_sha256": self.prospective_token_sha256,
            "model_call_id": (
                None if self.model_call_id is None else self.model_call_id.value
            ),
            "model_prompt_sha256": self.model_prompt_sha256,
            "model_telemetry_sha256": self.model_telemetry_sha256,
            "propensity": (
                None
                if self.propensity_numerator is None
                else {
                    "numerator": self.propensity_numerator,
                    "denominator": self.propensity_denominator,
                }
            ),
            "current_outcome_access": self.current_outcome_access,
        }


def validate_finite_action_decision(
    authority: FiniteActionSetAuthority,
    decision: FiniteActionDecision,
) -> None:
    if type(authority) is not FiniteActionSetAuthority:
        raise TypeError("authority must be an exact FiniteActionSetAuthority")
    FiniteActionSetAuthority.__post_init__(authority)
    if type(decision) is not FiniteActionDecision:
        raise TypeError("decision must be an exact FiniteActionDecision")
    FiniteActionDecision.__post_init__(decision)
    if (
        decision.authority_sha256 != authority.authority_sha256
        or decision.support_sha256 != authority.support.support_sha256
    ):
        raise ValueError("finite action decision is bound to a different authority")
    if decision.selected_ordinal >= authority.support.cardinality:
        raise ValueError("finite action decision ordinal is outside the support")
    row = authority.support.options[decision.selected_ordinal]
    expected = (
        row.option.option_id,
        row.option.identity_sha256,
        row.option.child_configuration_sha256,
        row.phenotype_identity_sha256,
    )
    observed = (
        decision.option_id,
        decision.option_identity_sha256,
        decision.child_configuration_sha256,
        decision.phenotype_identity_sha256,
    )
    if observed != expected:
        raise ValueError("finite action decision changed the selected support row")
    if (
        decision.selector_kind is FiniteActionSelectorKind.ENGINE
        and decision.propensity_denominator != authority.support.cardinality
    ):
        raise ValueError("engine propensity denominator differs from K")


@runtime_checkable
class EngineFiniteActionPolicy(Protocol):
    policy_id: str
    policy_version: int
    definition_sha256: str
    schedule_state_sha256: str

    def freeze_rank(
        self,
        authority: FiniteActionSetAuthority,
        *,
        task_sha256: str,
        pre_outcome_phase_commit_sha256: str,
    ) -> ProspectiveUniformRankToken: ...

    def choose(self, request: EngineFiniteActionRequest) -> FiniteActionDecision: ...


__all__ = [
    "EngineFiniteActionPolicy",
    "EngineFiniteActionRequest",
    "FiniteActionDecision",
    "FiniteActionSelectorKind",
    "ProspectiveUniformRankToken",
    "validate_finite_action_decision",
]
