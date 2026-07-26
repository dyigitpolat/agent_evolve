from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib

import pytest

from agent_evolve.application.campaign_learning import (
    CampaignInsightAuditBinding,
    CampaignInsightLifecycleDecision,
    CampaignInsightPromotionPolicy,
)
from agent_evolve.application.insight_memory import InsightLifecycleState
from agent_evolve.domain.insight import InsightRef
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.memory.global_falsification import (
    GlobalHypothesisVerdict,
)
from agent_evolve.policies.memory.randomized_subset import (
    InsightSelectionDecision,
    InsightSelectionMode,
    InsightTrial,
    estimate_marginal_effect,
)
from tests.test_campaign_closed_loop_learning import (
    _audit_binding,
    _setup,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _trial_panel(
    *,
    block_count: int,
    treated_reward: float = 1.0,
    control_reward: float = 0.0,
    missing_generation: bool = False,
) -> tuple[InsightRef, str, tuple[InsightTrial, ...]]:
    ids = DeterministicIdFactory("conservative_promotion")
    target = InsightRef(ids.new_insight_id(), 1)
    other = InsightRef(ids.new_insight_id(), 1)
    eligible = tuple(sorted((target, other)))
    context = _sha("conservative-promotion-context")
    trials: list[InsightTrial] = []
    for generation in range(1, block_count + 1):
        for selected, reward in (
            (target, treated_reward),
            (other, control_reward),
        ):
            decision = InsightSelectionDecision(
                context_hash=context,
                eligible=eligible,
                selected=(selected,),
                exploitation_subset=(eligible[0],),
                score_snapshot=tuple((reference, 0.0) for reference in eligible),
                subset_size=1,
                exploration_probability=Fraction(1, 1),
                mode=InsightSelectionMode.EXPLORE_UNIFORM,
                selected_subset_probability=Fraction(1, 2),
            )
            trials.append(
                InsightTrial(
                    credit_unit_id=ids.new_operator_invocation_id(),
                    candidate_ids=(ids.new_candidate_id(),),
                    reward_definition_hash=_sha("normalized-frontier-gain"),
                    decision=decision,
                    reward=reward,
                    treatment_binding_sha256=_sha(
                        f"treatment:{generation}:{selected.insight_id.value}"
                    ),
                    generation=(
                        None
                        if missing_generation and generation == block_count
                        else generation
                    ),
                )
            )
    return target, context, tuple(trials)


def _evidence(
    policy: CampaignInsightPromotionPolicy,
    *,
    block_count: int,
    family_hypothesis_count: int,
    treated_reward: float = 1.0,
    control_reward: float = 0.0,
    missing_generation: bool = False,
):
    target, context, trials = _trial_panel(
        block_count=block_count,
        treated_reward=treated_reward,
        control_reward=control_reward,
        missing_generation=missing_generation,
    )
    estimate = estimate_marginal_effect(trials, target, context_hash=context)
    evidence = policy.evaluate(
        estimate=estimate,
        source_trials=trials,
        source_trial_set_sha256=_sha("sealed-trial-panel"),
        family_hypothesis_count=family_hypothesis_count,
    )
    return estimate, evidence


def test_default_policy_needs_exact_block_evidence_after_multiplicity() -> None:
    policy = CampaignInsightPromotionPolicy()
    _, five_blocks = _evidence(
        policy,
        block_count=5,
        family_hypothesis_count=3,
    )
    assert five_blocks.exact_one_sided_p_value == Fraction(1, 32)
    assert five_blocks.adjusted_alpha == 0.05 / 3
    assert policy.admits(five_blocks) is False
    assert five_blocks.failure_reasons == (
        "exact_block_sign_evidence_exceeds_adjusted_alpha",
    )

    _, six_blocks = _evidence(
        policy,
        block_count=6,
        family_hypothesis_count=3,
    )
    assert six_blocks.exact_one_sided_p_value == Fraction(1, 64)
    assert six_blocks.failure_reasons == ()
    assert policy.admits(six_blocks) is True
    assert six_blocks.to_record()["practical_effect_margin_hex"] == (0.01).hex()
    assert six_blocks.to_record()["evidence_sha256"] == six_blocks.evidence_sha256
    assert (
        policy.definition_sha256 == CampaignInsightPromotionPolicy().definition_sha256
    )


def test_large_point_effect_cannot_replace_independent_blocks() -> None:
    policy = CampaignInsightPromotionPolicy(
        minimum_treated_trials=1,
        minimum_control_trials=1,
        minimum_effective_support=1.0,
    )
    estimate, evidence = _evidence(
        policy,
        block_count=1,
        family_hypothesis_count=1,
        treated_reward=1000.0,
    )
    assert estimate.effect == 1000.0
    assert policy.admits(evidence) is False
    assert "insufficient_independent_blocks" in evidence.failure_reasons
    assert "exact_block_sign_evidence_exceeds_adjusted_alpha" in (
        evidence.failure_reasons
    )


def test_margin_generation_identity_and_policy_identity_fail_closed() -> None:
    policy = CampaignInsightPromotionPolicy(minimum_effect=0.1)
    _, below_margin = _evidence(
        policy,
        block_count=6,
        family_hypothesis_count=1,
        treated_reward=0.1,
    )
    assert "pooled_effect_does_not_clear_practical_margin" in (
        below_margin.failure_reasons
    )
    assert below_margin.successful_block_count == 0

    _, missing_block = _evidence(
        policy,
        block_count=6,
        family_hypothesis_count=1,
        missing_generation=True,
    )
    assert missing_block.missing_generation_trial_count == 2
    assert "missing_authenticated_generation_block" in missing_block.failure_reasons

    foreign = CampaignInsightPromotionPolicy(minimum_effect=0.2)
    with pytest.raises(ValueError, match="another policy"):
        foreign.admits(missing_block)


def test_counterexample_deprecates_even_when_utility_cannot_promote() -> None:
    memory, learning, entries, _, _, context, barrier = _setup()
    contradicted = entries[1]
    result = learning.close_generation(
        memory_credit_batch=barrier,
        audits=(
            _audit_binding(
                contradicted,
                GlobalHypothesisVerdict.COUNTEREXAMPLE,
                context,
            ),
        ),
    )
    decision = result.decisions[0]
    assert decision.decision is CampaignInsightLifecycleDecision.DEPRECATE
    assert decision.causal_usefulness.randomized_promotion_evidence.passes_policy is (
        False
    )
    assert memory.entries_for((contradicted.reference,))[0].lifecycle_state is (
        InsightLifecycleState.DEPRECATED
    )


def test_semantic_support_thresholds_must_be_post_origin() -> None:
    memory, learning, entries, _, _, context, barrier = _setup()
    supported = entries[0]
    valid = _audit_binding(supported, GlobalHypothesisVerdict.SUPPORT, context)
    origin_available_decisions = tuple(
        replace(value, post_origin_revision_evidence=False)
        for value in valid.receipt.decisions
    )
    origin_only_receipt = replace(
        valid.receipt,
        decisions=origin_available_decisions,
    )
    origin_only = CampaignInsightAuditBinding(
        request=valid.request,
        receipt=origin_only_receipt,
        exact_context_sha256=context,
    )

    with pytest.raises(ValueError, match="post-origin evidence thresholds"):
        learning.close_generation(
            memory_credit_batch=barrier,
            audits=(origin_only,),
        )
    assert memory.transitions == ()


def test_audit_plan_projection_is_registered_canonical_and_read_only() -> None:
    _, learning, entries, _, _, _, _ = _setup()
    references = tuple(sorted(value.reference for value in entries))

    plans = learning.audit_plans_for(references)
    assert tuple(value.reference for value in plans) == references
    assert learning.audit_plans_for(references) == plans

    with pytest.raises(ValueError, match="canonically ordered"):
        learning.audit_plans_for(tuple(reversed(references)))
    missing = InsightRef(
        DeterministicIdFactory("missing_audit_plan").new_insight_id(),
        1,
    )
    with pytest.raises(ValueError, match="no registered semantic audit plan"):
        learning.audit_plans_for((missing,))
