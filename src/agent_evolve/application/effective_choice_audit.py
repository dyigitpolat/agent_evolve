"""Fail-closed proof that a model call has a real finite choice set.

Prompt text can advertise many actions while an exact treatment contract leaves
only one executable choice.  This module audits the application-layer
``InvocationPlan`` instead of trusting prose or an adapter schema.  A passing
receipt proves that one outcome-blind ``FiniteActionSetAuthority`` supplies at
least the configured number of distinct children and phenotypes, that the plan
uses that exact support contract, and that the authority's card is the plan's
only explicit memory assignment.

The generic generator request currently carries the finite variation contract
but not the provider adapter's rendered output enum.  Consequently this module
does not claim to validate that enum and deliberately has no PydanticAI import.
That final adapter/trace equality check belongs at a boundary which exposes the
rendered schema as a typed value.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum

from agent_evolve.application.agentic_evolution import (
    InvocationPlan,
    MutationResponseMode,
)
from agent_evolve.domain.finite_action_set import FiniteActionSetAuthority
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.memory.staged_causal import ResolvedInsightAssignment


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_PLAN_BINDING_DOMAIN = b"agent-evolve:effective-choice-plan-binding:v1\x00"
_QUARANTINE_BINDING_DOMAIN = b"agent-evolve:effective-choice-quarantine-binding:v1\x00"
_RECEIPT_DOMAIN = b"agent-evolve:effective-choice-audit-receipt:v1\x00"

EFFECTIVE_CHOICE_AUDIT_POLICY_ID = "effective_finite_choice_audit"
EFFECTIVE_CHOICE_AUDIT_POLICY_VERSION = 1
EFFECTIVE_CHOICE_AUDIT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:effective-finite-choice-audit:v1;authority-required;"
    b"configured-minimum-cardinality;exact-plan-support-contract;"
    b"no-exact-treatment-or-compilation;outcome-blind;single-card-binding;"
    b"unique-option-option-identity-child-and-phenotype-identities"
).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, record: object) -> str:
    return hashlib.sha256(domain + _canonical_json(record)).hexdigest()


def _ref_record(reference: InsightRef) -> dict[str, object]:
    return {
        "insight_id": reference.insight_id.value,
        "version": reference.version,
    }


class EffectiveChoiceAuditError(ValueError):
    """The admitted plan does not expose the claimed effective choice set."""


class SelectedCardBindingMode(str, Enum):
    """Which explicit, replayable plan field selected the authority's card."""

    RESOLVED_ASSIGNMENT = "resolved_assignment"
    EXPLICIT_QUARANTINE = "explicit_quarantine"


@dataclass(frozen=True, slots=True)
class EffectiveChoiceAuditReceipt:
    """Immutable proof of one plan's effective finite-choice cardinality."""

    configured_minimum_cardinality: int
    generation: int
    invocation_label: str
    parent_candidate_id: CandidateId
    parent_configuration_sha256: str
    selected_card_reference: InsightRef
    selected_card_binding_mode: SelectedCardBindingMode
    selected_card_binding_sha256: str
    authority_sha256: str
    support_sha256: str
    presentation_sha256: str
    plan_contract_sha256: str
    authority_contract_sha256: str
    outcome_blind: bool
    exact_treatment_fields_absent: bool
    effective_cardinality: int
    option_ids: tuple[str, ...]
    option_identity_sha256s: tuple[str, ...]
    child_configuration_sha256s: tuple[str, ...]
    phenotype_identity_sha256s: tuple[str, ...]
    audited_plan_sha256: str
    policy_id: str = EFFECTIVE_CHOICE_AUDIT_POLICY_ID
    policy_version: int = EFFECTIVE_CHOICE_AUDIT_POLICY_VERSION
    policy_definition_sha256: str = EFFECTIVE_CHOICE_AUDIT_DEFINITION_SHA256
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            type(self.configured_minimum_cardinality) is not int
            or self.configured_minimum_cardinality < 2
        ):
            raise ValueError("configured minimum cardinality must be at least two")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("audited generation must be positive")
        if (
            type(self.invocation_label) is not str
            or not self.invocation_label
            or self.invocation_label != self.invocation_label.strip()
        ):
            raise ValueError("audited invocation_label must be canonical")
        if type(self.parent_candidate_id) is not CandidateId:
            raise TypeError("audit parent_candidate_id must be exact")
        CandidateId.__post_init__(self.parent_candidate_id)
        if type(self.selected_card_reference) is not InsightRef:
            raise TypeError("audit selected_card_reference must be exact")
        InsightRef.__post_init__(self.selected_card_reference)
        if type(self.selected_card_binding_mode) is not SelectedCardBindingMode:
            raise TypeError("audit selected_card_binding_mode must be exact")
        for name in (
            "parent_configuration_sha256",
            "selected_card_binding_sha256",
            "authority_sha256",
            "support_sha256",
            "presentation_sha256",
            "plan_contract_sha256",
            "authority_contract_sha256",
            "audited_plan_sha256",
            "policy_definition_sha256",
        ):
            require_sha256(getattr(self, name), f"effective choice audit {name}")
        if self.plan_contract_sha256 != self.authority_contract_sha256:
            raise ValueError("audit plan and authority contract identities differ")
        if type(self.outcome_blind) is not bool or not self.outcome_blind:
            raise ValueError("effective choice audit must prove outcome blindness")
        if (
            type(self.exact_treatment_fields_absent) is not bool
            or not self.exact_treatment_fields_absent
        ):
            raise ValueError("effective choice audit requires absent exact treatments")
        if (
            type(self.effective_cardinality) is not int
            or self.effective_cardinality < self.configured_minimum_cardinality
        ):
            raise ValueError(
                "effective choice cardinality is below its configured minimum"
            )
        count = self.effective_cardinality
        if (
            type(self.option_ids) is not tuple
            or len(self.option_ids) != count
            or any(type(value) is not str or not value for value in self.option_ids)
        ):
            raise ValueError("audit option_ids must cover the effective support")
        if len(set(self.option_ids)) != count:
            raise ValueError("effective option IDs must be unique")
        for name in (
            "option_identity_sha256s",
            "child_configuration_sha256s",
            "phenotype_identity_sha256s",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != count:
                raise ValueError(f"audit {name} must cover the effective support")
            for value in values:
                require_sha256(value, f"effective choice audit {name}")
            if len(set(values)) != count:
                raise ValueError(f"effective {name} must be unique")
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("effective choice audit policy_id is invalid")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("effective choice audit policy_version must be positive")
        expected = _hash(_RECEIPT_DOMAIN, self.to_record())
        if self.receipt_sha256:
            require_sha256(self.receipt_sha256, "effective choice audit receipt_sha256")
            if self.receipt_sha256 != expected:
                raise ValueError("effective choice audit receipt is not authentic")
        else:
            object.__setattr__(self, "receipt_sha256", expected)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "configured_minimum_cardinality": self.configured_minimum_cardinality,
            "generation": self.generation,
            "invocation_label": self.invocation_label,
            "parent_candidate_id": self.parent_candidate_id.value,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "selected_card_reference": _ref_record(self.selected_card_reference),
            "selected_card_binding_mode": self.selected_card_binding_mode.value,
            "selected_card_binding_sha256": self.selected_card_binding_sha256,
            "authority_sha256": self.authority_sha256,
            "support_sha256": self.support_sha256,
            "presentation_sha256": self.presentation_sha256,
            "plan_contract_sha256": self.plan_contract_sha256,
            "authority_contract_sha256": self.authority_contract_sha256,
            "outcome_blind": self.outcome_blind,
            "exact_treatment_fields_absent": self.exact_treatment_fields_absent,
            "effective_cardinality": self.effective_cardinality,
            "option_ids": list(self.option_ids),
            "option_identity_sha256s": list(self.option_identity_sha256s),
            "child_configuration_sha256s": list(self.child_configuration_sha256s),
            "phenotype_identity_sha256s": list(self.phenotype_identity_sha256s),
            "audited_plan_sha256": self.audited_plan_sha256,
        }


def _configured_minimum(value: int) -> int:
    if type(value) is not int or value < 2:
        raise ValueError("minimum_cardinality must be an exact integer of at least two")
    return value


def _card_binding(
    plan: InvocationPlan,
    card_reference: InsightRef,
) -> tuple[SelectedCardBindingMode, str]:
    resolved = plan.resolved_insight_assignment
    quarantine = plan.quarantine_test_insights
    if resolved is not None:
        if type(resolved) is not ResolvedInsightAssignment:
            raise EffectiveChoiceAuditError("resolved card assignment is not exact")
        ResolvedInsightAssignment.__post_init__(resolved)
        if quarantine or resolved.selection_decision.selected != (card_reference,):
            raise EffectiveChoiceAuditError(
                "resolved assignment does not select exactly the authority card"
            )
        return SelectedCardBindingMode.RESOLVED_ASSIGNMENT, resolved.assignment_sha256
    if quarantine != (card_reference,):
        raise EffectiveChoiceAuditError(
            "explicit quarantine does not select exactly the authority card"
        )
    return (
        SelectedCardBindingMode.EXPLICIT_QUARANTINE,
        _hash(
            _QUARANTINE_BINDING_DOMAIN,
            {
                "mode": SelectedCardBindingMode.EXPLICIT_QUARANTINE.value,
                "reference": _ref_record(card_reference),
            },
        ),
    )


def _plan_binding_record(
    *,
    plan: InvocationPlan,
    authority: FiniteActionSetAuthority,
    binding_mode: SelectedCardBindingMode,
    binding_sha256: str,
    option_ids: tuple[str, ...],
    option_identity_sha256s: tuple[str, ...],
    child_configuration_sha256s: tuple[str, ...],
    phenotype_identity_sha256s: tuple[str, ...],
) -> dict[str, object]:
    parent = plan.parents[0]
    contract = plan.finite_variation_contract
    assert contract is not None
    return {
        "schema_version": 1,
        "generation": plan.generation,
        "invocation_label": plan.label,
        "operator_kind": plan.operator_kind.value,
        "mutation_response_mode": plan.mutation_response_mode.value,
        "parent_candidate_id": parent.candidate_id.value,
        "parent_configuration_sha256": parent.occurrence.configuration_hash,
        "plan_contract_sha256": contract.identity_sha256,
        "authority_sha256": authority.authority_sha256,
        "support_sha256": authority.support.support_sha256,
        "presentation_sha256": authority.support.presentation.presentation_sha256,
        "selected_card_reference": _ref_record(authority.card.reference),
        "selected_card_binding_mode": binding_mode.value,
        "selected_card_binding_sha256": binding_sha256,
        "use_memory": plan.use_memory,
        "exact_treatment_fields_present": {
            "insight_treatment_requirement": (
                plan.insight_treatment_requirement is not None
            ),
            "compiled_hypothesis_treatment": (
                plan.compiled_hypothesis_treatment is not None
            ),
            "compiled_hypothesis_eligibility": bool(
                plan.compiled_hypothesis_eligibility
            ),
        },
        "current_outcome_access": authority.current_outcome_access,
        "option_ids": list(option_ids),
        "option_identity_sha256s": list(option_identity_sha256s),
        "child_configuration_sha256s": list(child_configuration_sha256s),
        "phenotype_identity_sha256s": list(phenotype_identity_sha256s),
    }


def audit_effective_choice_plan(
    plan: InvocationPlan,
    *,
    minimum_cardinality: int,
) -> EffectiveChoiceAuditReceipt:
    """Validate and seal one real, outcome-blind K-choice plan.

    The configured minimum is deliberately explicit: an experiment requesting
    K=8 must not silently accept the domain type's lower bound of four.
    """

    minimum = _configured_minimum(minimum_cardinality)
    if type(plan) is not InvocationPlan:
        raise TypeError("effective choice audit requires an exact InvocationPlan")
    try:
        InvocationPlan.__post_init__(plan)
    except (TypeError, ValueError) as exc:
        raise EffectiveChoiceAuditError(
            "invocation plan failed its application-layer contract"
        ) from exc
    authority = plan.finite_action_set_authority
    if authority is None:
        raise EffectiveChoiceAuditError("finite action authority is absent")
    if type(authority) is not FiniteActionSetAuthority:
        raise EffectiveChoiceAuditError("finite action authority is not exact")
    try:
        FiniteActionSetAuthority.__post_init__(authority)
    except (TypeError, ValueError) as exc:
        raise EffectiveChoiceAuditError("finite action authority is invalid") from exc
    if (
        plan.mutation_response_mode
        is not MutationResponseMode.FINITE_OPTION_SELECTION_V1
    ):
        raise EffectiveChoiceAuditError("plan is not an opaque finite-option selection")
    contract = plan.finite_variation_contract
    if type(contract) is not FiniteVariationContract:
        raise EffectiveChoiceAuditError("plan finite variation contract is absent")
    try:
        validate_finite_variation_contract(contract)
    except (TypeError, ValueError) as exc:
        raise EffectiveChoiceAuditError(
            "plan finite variation contract is invalid"
        ) from exc
    support = authority.support
    authority_contract = support.support_contract
    if (
        contract != authority_contract
        or contract.identity_sha256 != authority_contract.identity_sha256
    ):
        raise EffectiveChoiceAuditError(
            "plan contract differs from the authority support contract"
        )
    if (
        plan.insight_treatment_requirement is not None
        or plan.compiled_hypothesis_treatment is not None
        or plan.compiled_hypothesis_eligibility
    ):
        raise EffectiveChoiceAuditError(
            "finite-choice plan retains an exact treatment or compilation"
        )
    if plan.use_memory:
        raise EffectiveChoiceAuditError(
            "finite-choice plan must bind one explicit card before admission"
        )
    if authority.current_outcome_access is not False:
        raise EffectiveChoiceAuditError("finite action authority is not outcome-blind")
    parent = plan.parents[0]
    if (
        support.parent_candidate_id != parent.candidate_id
        or support.parent_configuration_sha256 != parent.occurrence.configuration_hash
        or support.support_contract.parent_configuration != parent.configuration
    ):
        raise EffectiveChoiceAuditError("finite support is bound to a foreign parent")
    binding_mode, binding_sha256 = _card_binding(plan, authority.card.reference)
    options = support.options
    cardinality = len(options)
    if cardinality < minimum or support.compatible_option_count != cardinality:
        raise EffectiveChoiceAuditError(
            "effective finite-choice cardinality is below its configured minimum"
        )
    option_ids = tuple(value.option.option_id for value in options)
    option_identity_sha256s = tuple(value.option.identity_sha256 for value in options)
    child_configuration_sha256s = tuple(
        value.option.child_configuration_sha256 for value in options
    )
    phenotype_identity_sha256s = tuple(
        value.phenotype_identity_sha256 for value in options
    )
    for name, values in (
        ("option IDs", option_ids),
        ("option identities", option_identity_sha256s),
        ("child configurations", child_configuration_sha256s),
        ("phenotype identities", phenotype_identity_sha256s),
    ):
        if len(values) != cardinality or len(set(values)) != cardinality:
            raise EffectiveChoiceAuditError(
                f"effective finite support does not have K unique {name}"
            )
    if tuple(value.option for value in options) != contract.options:
        raise EffectiveChoiceAuditError(
            "authority option records differ from the plan contract"
        )
    if support.presentation.ordered_option_ids != option_ids:
        raise EffectiveChoiceAuditError(
            "provider presentation does not expose the full authority support"
        )
    plan_record = _plan_binding_record(
        plan=plan,
        authority=authority,
        binding_mode=binding_mode,
        binding_sha256=binding_sha256,
        option_ids=option_ids,
        option_identity_sha256s=option_identity_sha256s,
        child_configuration_sha256s=child_configuration_sha256s,
        phenotype_identity_sha256s=phenotype_identity_sha256s,
    )
    return EffectiveChoiceAuditReceipt(
        configured_minimum_cardinality=minimum,
        generation=plan.generation,
        invocation_label=plan.label,
        parent_candidate_id=parent.candidate_id,
        parent_configuration_sha256=parent.occurrence.configuration_hash,
        selected_card_reference=authority.card.reference,
        selected_card_binding_mode=binding_mode,
        selected_card_binding_sha256=binding_sha256,
        authority_sha256=authority.authority_sha256,
        support_sha256=support.support_sha256,
        presentation_sha256=support.presentation.presentation_sha256,
        plan_contract_sha256=contract.identity_sha256,
        authority_contract_sha256=authority_contract.identity_sha256,
        outcome_blind=True,
        exact_treatment_fields_absent=True,
        effective_cardinality=cardinality,
        option_ids=option_ids,
        option_identity_sha256s=option_identity_sha256s,
        child_configuration_sha256s=child_configuration_sha256s,
        phenotype_identity_sha256s=phenotype_identity_sha256s,
        audited_plan_sha256=_hash(_PLAN_BINDING_DOMAIN, plan_record),
    )


def validate_effective_choice_audit_receipt(
    receipt: EffectiveChoiceAuditReceipt,
    plan: InvocationPlan,
) -> None:
    """Re-derive a receipt from its plan and reject any stale or forged proof."""

    if type(receipt) is not EffectiveChoiceAuditReceipt:
        raise TypeError("receipt must be an exact EffectiveChoiceAuditReceipt")
    EffectiveChoiceAuditReceipt.__post_init__(receipt)
    expected = audit_effective_choice_plan(
        plan,
        minimum_cardinality=receipt.configured_minimum_cardinality,
    )
    if receipt != expected:
        raise ValueError("effective choice receipt differs from the audited plan")


__all__ = [
    "EFFECTIVE_CHOICE_AUDIT_DEFINITION_SHA256",
    "EFFECTIVE_CHOICE_AUDIT_POLICY_ID",
    "EFFECTIVE_CHOICE_AUDIT_POLICY_VERSION",
    "EffectiveChoiceAuditError",
    "EffectiveChoiceAuditReceipt",
    "SelectedCardBindingMode",
    "audit_effective_choice_plan",
    "validate_effective_choice_audit_receipt",
]
