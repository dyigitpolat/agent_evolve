"""Focused tests for the generic effective finite-choice guard."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from fractions import Fraction

import pytest

from agent_evolve.application.agentic_evolution import (
    EvolutionCandidate,
    InvocationPlan,
    MutationResponseMode,
    OperatorKind,
)
from agent_evolve.application.effective_choice_audit import (
    EffectiveChoiceAuditError,
    SelectedCardBindingMode,
    audit_effective_choice_plan,
    validate_effective_choice_audit_receipt,
)
from agent_evolve.application.matched_finite_action_block import (
    finite_action_mutation_boundary,
)
from agent_evolve.domain.finite_action_set import (
    FiniteActionCardAuthority,
    FiniteActionOptionAuthority,
    FiniteActionPresentationAuthority,
    FiniteActionSetAuthority,
    FiniteActionSourceMode,
    FiniteActionSupportAuthority,
)
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import (
    CandidateId,
    InsightId,
    OperatorInvocationId,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from agent_evolve.policies.memory.randomized_subset import (
    InsightSelectionDecision,
    InsightSelectionMode,
)
from agent_evolve.policies.memory.staged_causal import (
    DelayedCreditMode,
    MemoryAssignmentArm,
    ResolvedInsightAssignment,
    insight_selection_decision_sha256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _parent() -> EvolutionCandidate:
    configuration = freeze_json({"value": 0})
    digest = typed_json_sha256(configuration)
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId("candidate_effective_choice_parent"),
            configuration_hash=digest,
            configuration_artifact_hash=digest,
            proposal_sequence=0,
        ),
        configuration=configuration,
        objectives=(("score", 0.0),),
        valid=True,
        generation=0,
        label="effective-choice-parent",
    )


def _authority(
    parent: EvolutionCandidate,
    card_reference: InsightRef,
    *,
    cardinality: int = 4,
) -> FiniteActionSetAuthority:
    options = tuple(
        FiniteVariationOption(
            option_id=f"fixture.value_{value}",
            parent_configuration_sha256=parent.occurrence.configuration_hash,
            child_configuration=freeze_json({"value": value}),
            family="fixture_value",
            description=f"Set value to {value}.",
        )
        for value in range(1, cardinality + 1)
    )
    contract = FiniteVariationContract(
        catalog_id="fixture_effective_choices",
        catalog_version=1,
        catalog_definition_sha256=_sha("fixture effective choices v1"),
        parent_configuration=parent.configuration,
        options=options,
    )
    option_authorities = tuple(
        FiniteActionOptionAuthority(
            option=option,
            changed_paths=("$.value",),
            phenotype_policy_id="fixture_exact_phenotype",
            phenotype_policy_version=1,
            phenotype_identity_sha256=_sha(f"phenotype:{option.option_id}"),
        )
        for option in options
    )
    presentation = FiniteActionPresentationAuthority(
        policy_id="fixture_choice_presentation",
        policy_version=1,
        definition_sha256=_sha("fixture choice presentation v1"),
        ordered_option_ids=tuple(option.option_id for option in options),
        ordered_prompt_record_sha256s=tuple(
            option.prompt_record_sha256 for option in option_authorities
        ),
        prompt_shape_sha256=_sha("fixture choice prompt shape"),
    )
    support = FiniteActionSupportAuthority(
        parent_candidate_id=parent.candidate_id,
        source_contract_sha256=contract.identity_sha256,
        support_contract=contract,
        endpoint_definition_sha256=_sha("fixture endpoint"),
        context_projection_sha256=_sha("fixture context"),
        options=option_authorities,
        anchor_option_id=options[0].option_id,
        presentation=presentation,
        compatible_option_count=cardinality,
    )
    card = FiniteActionCardAuthority(
        source_mode=FiniteActionSourceMode.EVIDENCE_FREE_CARD,
        reference=card_reference,
        card_content_sha256=_sha("fixture card content"),
        registered_source_evidence_sha256=None,
        exact_anchor_requirement_sha256=_sha("fixture anchor requirement"),
        compilation_request_sha256=None,
        compilation_receipt_sha256=None,
        executable_spec_sha256=None,
        prompt_card_record_sha256=_sha("fixture prompt card"),
    )
    return FiniteActionSetAuthority(
        support=support,
        card=card,
        support_compilation_request_sha256=_sha("fixture support request"),
        support_compilation_draft_sha256=_sha("fixture support draft"),
        support_compiler_policy_id="fixture_support_compiler",
        support_compiler_policy_version=1,
        support_compiler_definition_sha256=_sha("fixture support compiler v1"),
        current_outcome_access=False,
    )


def _plan() -> InvocationPlan:
    parent = _parent()
    reference = InsightRef(InsightId("insight_effective_choice_card"), 1)
    authority = _authority(parent, reference)
    allowed, mutation = finite_action_mutation_boundary(
        contract=authority.support.support_contract,
        parent_candidate_id=parent.candidate_id,
    )
    return InvocationPlan(
        operator_kind=OperatorKind.TYPED_MUTATION,
        parents=(parent,),
        generation=1,
        label="effective_choice_model_selection",
        allowed_top_level=allowed,
        mutation_contract=mutation,
        mutation_response_mode=MutationResponseMode.FINITE_OPTION_SELECTION_V1,
        finite_variation_contract=authority.support.support_contract,
        quarantine_test_insights=(reference,),
        finite_action_set_authority=authority,
    )


def _resolved_assignment(reference: InsightRef) -> ResolvedInsightAssignment:
    context_hash = _sha("resolved assignment context")
    decision = InsightSelectionDecision(
        context_hash=context_hash,
        eligible=(reference,),
        selected=(reference,),
        exploitation_subset=(reference,),
        score_snapshot=((reference, 1.0),),
        subset_size=1,
        exploration_probability=Fraction(0),
        mode=InsightSelectionMode.EXPLOIT,
        selected_subset_probability=Fraction(1),
    )
    return ResolvedInsightAssignment(
        credit_unit_id=OperatorInvocationId("operator_effective_choice_assignment"),
        exact_context_hash=context_hash,
        estimand_stratum_hash=_sha("resolved assignment stratum"),
        block_id="effective_choice_fixture",
        arm=MemoryAssignmentArm.ADAPTIVE,
        selection_decision=decision,
        selection_decision_sha256=insight_selection_decision_sha256(decision),
        score_snapshot_sha256=_sha("resolved assignment snapshot"),
        prompt_shape_sha256=_sha("resolved assignment prompt shape"),
        credit_mode=DelayedCreditMode.WAVE_SEALED_ITT,
    )


def test_audit_seals_exact_k_choice_quarantine_plan() -> None:
    plan = _plan()
    authority = plan.finite_action_set_authority
    assert authority is not None

    receipt = audit_effective_choice_plan(plan, minimum_cardinality=4)

    assert receipt.effective_cardinality == 4
    assert receipt.configured_minimum_cardinality == 4
    assert receipt.selected_card_reference == authority.card.reference
    assert (
        receipt.selected_card_binding_mode
        is SelectedCardBindingMode.EXPLICIT_QUARANTINE
    )
    assert receipt.outcome_blind is True
    assert receipt.exact_treatment_fields_absent is True
    assert receipt.plan_contract_sha256 == receipt.authority_contract_sha256
    assert receipt.authority_sha256 == authority.authority_sha256
    assert receipt.support_sha256 == authority.support.support_sha256
    assert receipt.option_ids == tuple(
        value.option.option_id for value in authority.support.options
    )
    assert len(set(receipt.option_identity_sha256s)) == 4
    assert len(set(receipt.child_configuration_sha256s)) == 4
    assert len(set(receipt.phenotype_identity_sha256s)) == 4
    validate_effective_choice_audit_receipt(receipt, plan)


def test_audit_accepts_exact_resolved_assignment_card_binding() -> None:
    quarantine_plan = _plan()
    authority = quarantine_plan.finite_action_set_authority
    assert authority is not None
    assignment = _resolved_assignment(authority.card.reference)
    plan = replace(
        quarantine_plan,
        quarantine_test_insights=(),
        resolved_insight_assignment=assignment,
    )

    receipt = audit_effective_choice_plan(plan, minimum_cardinality=4)

    assert (
        receipt.selected_card_binding_mode
        is SelectedCardBindingMode.RESOLVED_ASSIGNMENT
    )
    assert receipt.selected_card_binding_sha256 == assignment.assignment_sha256
    validate_effective_choice_audit_receipt(receipt, plan)


def test_guard_rejects_missing_authority_and_effective_cardinality_below_config() -> (
    None
):
    plan = _plan()

    with pytest.raises(EffectiveChoiceAuditError, match="authority is absent"):
        audit_effective_choice_plan(
            replace(plan, finite_action_set_authority=None),
            minimum_cardinality=4,
        )
    with pytest.raises(EffectiveChoiceAuditError, match="below its configured minimum"):
        audit_effective_choice_plan(plan, minimum_cardinality=5)


@pytest.mark.parametrize(
    "corrupt",
    (
        "foreign_card",
        "outcome_access",
        "compiled_fields",
        "duplicate_phenotype",
        "singleton_support",
    ),
)
def test_guard_rejects_corrupted_effective_choice_proofs(corrupt: str) -> None:
    plan = _plan()
    authority = plan.finite_action_set_authority
    assert authority is not None
    if corrupt == "foreign_card":
        object.__setattr__(
            plan,
            "quarantine_test_insights",
            (InsightRef(InsightId("insight_foreign_card"), 1),),
        )
    elif corrupt == "outcome_access":
        object.__setattr__(authority, "current_outcome_access", True)
    elif corrupt == "compiled_fields":
        object.__setattr__(plan, "compiled_hypothesis_eligibility", (object(),))
    elif corrupt == "duplicate_phenotype":
        first, second, *_ = authority.support.options
        object.__setattr__(
            second,
            "phenotype_identity_sha256",
            first.phenotype_identity_sha256,
        )
    else:
        object.__setattr__(
            authority.support,
            "options",
            authority.support.options[:1],
        )
        object.__setattr__(authority.support, "compatible_option_count", 1)

    with pytest.raises(
        EffectiveChoiceAuditError,
        match="application-layer contract",
    ):
        audit_effective_choice_plan(plan, minimum_cardinality=4)


def test_receipt_validator_detects_stale_plan_and_forged_support_ids() -> None:
    plan = _plan()
    receipt = audit_effective_choice_plan(plan, minimum_cardinality=4)
    stale_plan = replace(plan, label="different_effective_choice_call")
    forged_ids = tuple(f"forged.option_{index}" for index in range(4))
    forged_receipt = replace(
        receipt,
        option_ids=forged_ids,
        receipt_sha256="",
    )

    with pytest.raises(ValueError, match="differs from the audited plan"):
        validate_effective_choice_audit_receipt(receipt, stale_plan)
    with pytest.raises(ValueError, match="differs from the audited plan"):
        validate_effective_choice_audit_receipt(forged_receipt, plan)
