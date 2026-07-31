from __future__ import annotations

import hashlib

import pytest

from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionDescriptor,
    AdaptiveActionFactorCell,
    AdaptiveActionOutcome,
    AdaptiveActionSetOutcome,
    OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION,
    OutcomeAdaptiveActionRacingPolicy,
)
from agent_evolve.application.same_prefix_paired_audit import (
    FactorStratifiedSamePrefixPairedAuditDesigner,
    SamePrefixPairedAuditAdjudicator,
    SamePrefixPairedAuditArm,
    SamePrefixPairedAuditWinner,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _market() -> tuple[AdaptiveActionDescriptor, ...]:
    actions = []
    for branch in ("branch_a", "branch_b"):
        for role in ("coverage", "interaction", "local"):
            lane_id = f"{branch}.{role}"
            for rank in (1, 2):
                actions.append(
                    AdaptiveActionDescriptor(
                        action_sha256=_sha(
                            f"paired-action:{branch}:{role}:{rank}"
                        ),
                        phenotype_sha256=_sha(
                            f"paired-phenotype:{branch}:{role}:{rank}"
                        ),
                        lane_id=lane_id,
                        operator_id=f"{role}.r{rank}",
                        native_rank=rank,
                        lane_size=2,
                        prior_score=float(3 - rank) / 2.0,
                        parent_generated_in_current_run=False,
                        semantic_cell_ids=(f"role:{role}",),
                        factor_cells=tuple(
                            sorted(
                                (
                                    AdaptiveActionFactorCell(
                                        family_id="evolutionary_role",
                                        level_id=role,
                                    ),
                                    AdaptiveActionFactorCell(
                                        family_id=(
                                            "materialized_rank_layer"
                                        ),
                                        level_id=f"layer{rank - 1}",
                                    ),
                                    AdaptiveActionFactorCell(
                                        family_id="source_branch",
                                        level_id=branch,
                                    ),
                                    AdaptiveActionFactorCell(
                                        family_id="source_branch_role",
                                        level_id=f"{branch}:{role}",
                                    ),
                                )
                            )
                        ),
                    )
                )
    return tuple(actions)


def _policy(*, random_seed: int) -> OutcomeAdaptiveActionRacingPolicy:
    return OutcomeAdaptiveActionRacingPolicy(
        diagnostic_slots=4,
        randomized_audit_slots=1,
        reference_gain_scale=0.001,
        reference_gain_evidence_sha256=_sha("paired-prior-evidence"),
        trace_alternative_count=12,
        exploration_pool_size=4,
        audit_exploration_probability=0.1,
        stratified_audit_coverage_family_ids=("source_branch_role",),
        stratified_audit_stratum_family_ids=(
            "evolutionary_role",
            "materialized_rank_layer",
            "source_branch",
        ),
        random_seed=random_seed,
        policy_version=(
            OUTCOME_ADAPTIVE_ACTION_RACING_STRATIFIED_AUDIT_POLICY_VERSION
        ),
    )


def _outcome(action_sha256: str, gain: float) -> AdaptiveActionOutcome:
    return AdaptiveActionOutcome(
        action_sha256=action_sha256,
        evaluation_sha256=_sha(f"paired-evaluation:{action_sha256}"),
        feasible=True,
        marginal_archive_gain=float(gain),
    )


def _fixture():
    market = _market()
    by_lane_rank = {
        (value.lane_id, value.native_rank): value for value in market
    }
    selected = tuple(
        sorted(
            (
                by_lane_rank[("branch_a.coverage", 1)].action_sha256,
                by_lane_rank[("branch_a.coverage", 2)].action_sha256,
                by_lane_rank[("branch_a.interaction", 1)].action_sha256,
                by_lane_rank[("branch_a.interaction", 2)].action_sha256,
                by_lane_rank[("branch_a.local", 1)].action_sha256,
                by_lane_rank[("branch_b.coverage", 1)].action_sha256,
                by_lane_rank[("branch_b.interaction", 1)].action_sha256,
            )
        )
    )
    diagnostic = tuple(sorted(selected[:4]))
    outcomes = tuple(_outcome(value, 0.0) for value in selected)
    request_sha256 = _sha("paired-v6-request")
    return market, selected, diagnostic, outcomes, request_sha256


def _find_decisions():
    market, selected, diagnostic, outcomes, request_sha256 = _fixture()
    legacy = None
    exploration = None
    for seed in range(10_000):
        decision = _policy(random_seed=seed).select_next(
            residual_request_sha256=request_sha256,
            actions=market,
            evaluation_slots=8,
            diagnostic_action_sha256s=diagnostic,
            diagnostic_joint_gain=0.0,
            selected_action_sha256s=selected,
            outcomes=outcomes,
        )
        evidence = decision.to_record(include_evidence=True)["evidence"]
        if evidence["audit_exploration_branch"] is False:
            legacy = decision
        if (
            evidence["audit_exploration_branch"] is True
            and decision.selected_action_sha256s[0]
            != evidence["legacy_audit_anchor_action_sha256"]
        ):
            exploration = decision
        if legacy is not None and exploration is not None:
            break
    assert legacy is not None
    assert exploration is not None
    return market, legacy, exploration


def test_designer_freezes_a_distinct_counterfactual_for_legacy_arm() -> None:
    market, legacy_decision, _ = _find_decisions()
    designer = FactorStratifiedSamePrefixPairedAuditDesigner(random_seed=19)

    plan = designer.design(decision=legacy_decision, actions=market)
    repeated = designer.design(decision=legacy_decision, actions=market)

    assert plan == repeated
    assert plan.plan_sha256 == repeated.plan_sha256
    assert plan.authoritative_arm is SamePrefixPairedAuditArm.LEGACY
    assert (
        plan.authoritative_action_sha256 == plan.legacy_action_sha256
    )
    assert plan.exploration_action_sha256 != plan.legacy_action_sha256
    assert plan.exploration_action_sha256 in (
        plan.distinct_exploration_support_action_sha256s
    )
    assert not (
        {
            plan.legacy_action_sha256,
            plan.exploration_action_sha256,
        }
        & set(plan.common_prefix_action_sha256s)
    )
    record = plan.to_record(include_evidence=True)
    assert record["current_arm_outcomes_observed"] is False
    assert record["assay_union_may_enter_authoritative_archive"] is False
    assert record["evidence"][
        "plan_must_be_committed_before_arm_evaluation"
    ] is True


def test_designer_reuses_an_authoritative_distinct_exploration_arm() -> None:
    market, _, exploration_decision = _find_decisions()
    designer = FactorStratifiedSamePrefixPairedAuditDesigner(random_seed=23)

    plan = designer.design(
        decision=exploration_decision,
        actions=market,
    )

    assert plan.authoritative_arm is SamePrefixPairedAuditArm.EXPLORATION
    assert plan.exploration_action_sha256 == (
        exploration_decision.selected_action_sha256s[0]
    )
    evidence = plan.to_record(include_evidence=True)["evidence"]
    assert evidence["authoritative_action_reused_when_exploration"] is True
    assert evidence["stratum_draw_hex"] is None
    assert evidence["action_draw_hex"] is None


def _set_outcome(
    *,
    prefix: tuple[str, ...],
    outcome: AdaptiveActionOutcome,
    prior_gain: float,
    fixed_gain: float,
    conditional_gain: float,
) -> AdaptiveActionSetOutcome:
    return AdaptiveActionSetOutcome(
        prior_action_evaluation_bindings=tuple(
            sorted(
                (
                    action_sha256,
                    _sha(f"paired-prefix-evaluation:{action_sha256}"),
                )
                for action_sha256 in prefix
            )
        ),
        current_action_evaluation_bindings=(
            (
                outcome.action_sha256,
                outcome.evaluation_sha256,
            ),
        ),
        prior_selected_set_gain=float(prior_gain),
        current_wave_fixed_set_gain=float(fixed_gain),
        augmented_selected_set_gain=float(
            prior_gain + conditional_gain
        ),
        conditional_set_gain=float(conditional_gain),
    )


def test_adjudicator_compares_both_arms_at_one_prefix_without_union() -> None:
    market, legacy_decision, _ = _find_decisions()
    plan = FactorStratifiedSamePrefixPairedAuditDesigner(
        random_seed=29
    ).design(decision=legacy_decision, actions=market)
    legacy_outcome = _outcome(plan.legacy_action_sha256, 0.01)
    exploration_outcome = _outcome(plan.exploration_action_sha256, 0.03)
    legacy_set = _set_outcome(
        prefix=plan.common_prefix_action_sha256s,
        outcome=legacy_outcome,
        prior_gain=0.4,
        fixed_gain=0.02,
        conditional_gain=0.01,
    )
    exploration_set = _set_outcome(
        prefix=plan.common_prefix_action_sha256s,
        outcome=exploration_outcome,
        prior_gain=0.4,
        fixed_gain=0.03,
        conditional_gain=0.025,
    )

    observation = SamePrefixPairedAuditAdjudicator().adjudicate(
        plan=plan,
        legacy_outcome=legacy_outcome,
        exploration_outcome=exploration_outcome,
        legacy_set_outcome=legacy_set,
        exploration_set_outcome=exploration_set,
    )

    assert observation.winner is SamePrefixPairedAuditWinner.EXPLORATION
    assert observation.conditional_gain_delta == pytest.approx(0.015)
    assert observation.authoritative_set_outcome is legacy_set
    record = observation.to_record(include_evidence=True)
    assert record["assay_union_admitted_to_authoritative_archive"] is False
    assert record["counterfactual_endpoints_share_one_prefix"] is True


def test_adjudicator_rejects_different_prefixes() -> None:
    market, legacy_decision, _ = _find_decisions()
    plan = FactorStratifiedSamePrefixPairedAuditDesigner(
        random_seed=31
    ).design(decision=legacy_decision, actions=market)
    legacy_outcome = _outcome(plan.legacy_action_sha256, 0.01)
    exploration_outcome = _outcome(plan.exploration_action_sha256, 0.03)
    legacy_set = _set_outcome(
        prefix=plan.common_prefix_action_sha256s,
        outcome=legacy_outcome,
        prior_gain=0.4,
        fixed_gain=0.02,
        conditional_gain=0.01,
    )
    altered_prefix = tuple(
        sorted(
            (
                *plan.common_prefix_action_sha256s[:-1],
                _sha("foreign-prefix-action"),
            )
        )
    )
    exploration_set = _set_outcome(
        prefix=altered_prefix,
        outcome=exploration_outcome,
        prior_gain=0.4,
        fixed_gain=0.03,
        conditional_gain=0.025,
    )

    with pytest.raises(ValueError, match="exact common prefix"):
        SamePrefixPairedAuditAdjudicator().adjudicate(
            plan=plan,
            legacy_outcome=legacy_outcome,
            exploration_outcome=exploration_outcome,
            legacy_set_outcome=legacy_set,
            exploration_set_outcome=exploration_set,
        )
