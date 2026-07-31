from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionDescriptor,
    AdaptiveActionFactorCell,
    AdaptiveActionOutcome,
    AdaptiveActionRacingDecision,
    AdaptiveActionSetOutcome,
    AdaptiveActionWave,
)
from agent_evolve.application.residual_headroom_ledger import (
    ConservedResidualHeadroomLedger,
    ConservedResidualHeadroomProjector,
    ResidualHeadroomAdaptiveMarketProjector,
    ResidualHeadroomLedgerConfig,
    ResidualHeadroomStageClosure,
)
from agent_evolve.application.residual_headroom_campaign_runtime import (
    InMemoryTransactionalResidualHeadroomStore,
)
from agent_evolve.application.residual_portfolio_evolution import (
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.infrastructure.residual_headroom_journal import (
    DurableJsonlResidualHeadroomStore,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _action(
    name: str,
    *,
    lane: str,
    operator: str,
    role: str,
    rank: int,
    lane_size: int,
) -> AdaptiveActionDescriptor:
    return AdaptiveActionDescriptor(
        action_sha256=_sha(f"action:{name}"),
        phenotype_sha256=_sha(f"phenotype:{name}"),
        lane_id=lane,
        operator_id=operator,
        native_rank=rank,
        lane_size=lane_size,
        prior_score=0.5,
        parent_generated_in_current_run=False,
        factor_cells=(
            AdaptiveActionFactorCell(
                family_id="evolutionary_role",
                level_id=role,
            ),
        ),
    )


def _stage(
    *,
    context: str,
    generation: int,
    diagnostic_conditional: float,
    continuation_conditional: float,
    high_isolated: float,
    low_isolated: float,
    continuation_isolated: float,
):
    high = _action(
        "high",
        lane="lane_high",
        operator="mutate_high",
        role="coverage",
        rank=1,
        lane_size=2,
    )
    continuation = _action(
        "continuation",
        lane="lane_high",
        operator="mutate_high",
        role="coverage",
        rank=2,
        lane_size=2,
    )
    low = _action(
        "low",
        lane="lane_low",
        operator="mutate_low",
        role="interaction",
        rank=1,
        lane_size=1,
    )
    actions = (high, continuation, low)
    outcome_by_action = {
        action.action_sha256: AdaptiveActionOutcome(
            action_sha256=action.action_sha256,
            evaluation_sha256=_sha(
                f"evaluation:{action.action_sha256}"
            ),
            feasible=True,
            marginal_archive_gain=float(gain),
        )
        for action, gain in (
            (high, high_isolated),
            (low, low_isolated),
            (continuation, continuation_isolated),
        )
    }
    high_outcome = outcome_by_action[high.action_sha256]
    low_outcome = outcome_by_action[low.action_sha256]
    continuation_outcome = outcome_by_action[
        continuation.action_sha256
    ]
    diagnostic_bindings = tuple(
        sorted(
            (
                (high.action_sha256, high_outcome.evaluation_sha256),
                (low.action_sha256, low_outcome.evaluation_sha256),
            )
        )
    )
    diagnostic_set = AdaptiveActionSetOutcome(
        prior_action_evaluation_bindings=(),
        current_action_evaluation_bindings=diagnostic_bindings,
        prior_selected_set_gain=0.0,
        current_wave_fixed_set_gain=float(
            high_isolated + low_isolated
        ),
        augmented_selected_set_gain=diagnostic_conditional,
        conditional_set_gain=diagnostic_conditional,
    )
    diagnostic = AdaptiveActionRacingDecision(
        policy_id="test_headroom_policy",
        policy_version=1,
        policy_definition_sha256=_sha("headroom-policy"),
        residual_request_sha256=_sha(
            f"request:{context}:{generation}"
        ),
        wave=AdaptiveActionWave.DIAGNOSTIC,
        selected_action_sha256s=tuple(
            sorted((high.action_sha256, low.action_sha256))
        ),
        prior_selected_action_sha256s=(),
        observed_outcome_sha256s=(),
        observed_set_outcome_sha256s=(),
        selection_propensity=1.0,
        evidence=freeze_json({"outcome_blind": True}),
    )
    continuation_set = AdaptiveActionSetOutcome(
        prior_action_evaluation_bindings=diagnostic_bindings,
        current_action_evaluation_bindings=(
            (
                continuation.action_sha256,
                continuation_outcome.evaluation_sha256,
            ),
        ),
        prior_selected_set_gain=diagnostic_conditional,
        current_wave_fixed_set_gain=continuation_isolated,
        augmented_selected_set_gain=(
            diagnostic_conditional + continuation_conditional
        ),
        conditional_set_gain=continuation_conditional,
    )
    continuation_decision = AdaptiveActionRacingDecision(
        policy_id="test_headroom_policy",
        policy_version=1,
        policy_definition_sha256=_sha("headroom-policy"),
        residual_request_sha256=diagnostic.residual_request_sha256,
        wave=AdaptiveActionWave.ADAPTIVE,
        selected_action_sha256s=(continuation.action_sha256,),
        prior_selected_action_sha256s=tuple(
            sorted((high.action_sha256, low.action_sha256))
        ),
        observed_outcome_sha256s=tuple(
            sorted(
                (
                    high_outcome.outcome_sha256,
                    low_outcome.outcome_sha256,
                )
            )
        ),
        observed_set_outcome_sha256s=(
            diagnostic_set.set_outcome_sha256,
        ),
        selection_propensity=1.0,
        evidence=freeze_json({"current_outcomes_observed": True}),
    )
    closure = ConservedResidualHeadroomProjector().project(
        context_sha256=_sha(context),
        generation_index=generation,
        reference_gain_scale=0.1,
        reference_gain_evidence_sha256=_sha("gain-scale"),
        actions=actions,
        diagnostic_decision=diagnostic,
        continuation_decisions=(continuation_decision,),
        outcomes=tuple(
            sorted(
                outcome_by_action.values(),
                key=lambda value: value.action_sha256,
            )
        ),
        set_outcomes=(diagnostic_set, continuation_set),
    )
    return (
        closure,
        actions,
        (diagnostic, continuation_decision),
        tuple(outcome_by_action.values()),
        (diagnostic_set, continuation_set),
    )


def test_stage_projection_conserves_real_conditional_set_gain():
    closure, actions, _decisions, _outcomes, _set_outcomes = _stage(
        context="context-a",
        generation=1,
        diagnostic_conditional=1.0,
        continuation_conditional=0.5,
        high_isolated=0.8,
        low_isolated=0.6,
        continuation_isolated=0.8,
    )
    credit_by_action = {
        value.action_sha256: value.conditional_credit
        for value in closure.observations
    }

    assert closure.total_conditional_gain == 1.5
    assert sum(credit_by_action.values()) == pytest.approx(1.5)
    assert credit_by_action[actions[0].action_sha256] == pytest.approx(
        0.8 / 1.4
    )
    assert credit_by_action[actions[2].action_sha256] == pytest.approx(
        0.6 / 1.4
    )
    assert credit_by_action[actions[1].action_sha256] == 0.5
    assert closure.to_record()[
        "predicted_values_admitted_to_archive"
    ] is False
    assert ResidualHeadroomStageClosure.from_record(
        closure.to_record()
    ) == closure


def test_stage_projection_rejects_a_missing_decision_boundary():
    closure, actions, decisions, outcomes, set_outcomes = _stage(
        context="context-a",
        generation=1,
        diagnostic_conditional=1.0,
        continuation_conditional=0.5,
        high_isolated=0.8,
        low_isolated=0.6,
        continuation_isolated=0.8,
    )

    with pytest.raises(
        ValueError,
        match="each decision requires one ordered set outcome",
    ):
        ConservedResidualHeadroomProjector().project(
            context_sha256=closure.context_sha256,
            generation_index=1,
            reference_gain_scale=0.1,
            reference_gain_evidence_sha256=_sha("gain-scale"),
            actions=actions,
            diagnostic_decision=decisions[0],
            continuation_decisions=(),
            outcomes=outcomes,
            # Deliberately retain the continuation observation without its
            # preceding adaptive decision.
            set_outcomes=set_outcomes,
        )


def test_ledger_learns_late_bloom_and_context_isolation():
    (
        late_closure,
        actions,
        _decisions,
        _outcomes,
        _set_outcomes,
    ) = _stage(
        context="context-late",
        generation=1,
        diagnostic_conditional=0.1,
        continuation_conditional=0.8,
        high_isolated=0.1,
        low_isolated=0.0,
        continuation_isolated=0.8,
    )
    ledger = ConservedResidualHeadroomLedger(
        ResidualHeadroomLedgerConfig(
            cross_context_weight=0.0,
        )
    )
    empty = ledger.empty_state()
    state = ledger.append(empty, late_closure)
    continuation = actions[1]
    learned = ledger.estimate(
        state=state,
        context_sha256=late_closure.context_sha256,
        generation_index=2,
        action=continuation,
    )
    isolated = ledger.estimate(
        state=state,
        context_sha256=_sha("unseen-context"),
        generation_index=2,
        action=continuation,
    )
    empty_isolated = ledger.estimate(
        state=empty,
        context_sha256=_sha("unseen-context"),
        generation_index=2,
        action=continuation,
    )

    assert learned.late_bloom_headroom > 0.0
    assert learned.acquisition_score > isolated.acquisition_score
    assert isolated.expected_normalized_gain == (
        empty_isolated.expected_normalized_gain
    )
    assert isolated.acquisition_score == empty_isolated.acquisition_score


def test_ledger_distinguishes_late_bloom_from_saturation():
    late, late_actions, _decisions, _outcomes, _set_outcomes = _stage(
        context="context-late",
        generation=1,
        diagnostic_conditional=0.1,
        continuation_conditional=0.8,
        high_isolated=0.1,
        low_isolated=0.0,
        continuation_isolated=0.8,
    )
    (
        saturated,
        saturated_actions,
        _decisions,
        _outcomes,
        _set_outcomes,
    ) = _stage(
        context="context-saturated",
        generation=1,
        diagnostic_conditional=0.8,
        continuation_conditional=0.1,
        high_isolated=0.8,
        low_isolated=0.0,
        continuation_isolated=0.1,
    )
    ledger = ConservedResidualHeadroomLedger()
    state = ledger.append(ledger.empty_state(), late)
    state = ledger.append(state, saturated)
    late_estimate = ledger.estimate(
        state=state,
        context_sha256=late.context_sha256,
        generation_index=2,
        action=late_actions[1],
    )
    saturated_estimate = ledger.estimate(
        state=state,
        context_sha256=saturated.context_sha256,
        generation_index=2,
        action=saturated_actions[1],
    )

    assert late_estimate.late_bloom_headroom > 0.0
    assert late_estimate.saturation_risk == 0.0
    assert saturated_estimate.saturation_risk > 0.0
    assert saturated_estimate.late_bloom_headroom == 0.0


@dataclass(frozen=True, slots=True)
class _Delegate:
    projected: tuple[AdaptiveActionDescriptor, ...]
    projector_id: str = "test_headroom_delegate"
    projector_version: int = 1
    definition_sha256: str = _sha("headroom-delegate-definition")
    state_sha256: str = _sha("headroom-delegate-state")

    def __post_init__(self) -> None:
        assert self.projected

    async def project(
        self,
        request,
        proposals,
        actions,
        scores,
        required_action_sha256s,
    ):
        del request, proposals, actions, scores, required_action_sha256s
        return self.projected


def test_market_wrapper_uses_prior_only_headroom_without_archive_authority():
    closure, actions, _decisions, _outcomes, _set_outcomes = _stage(
        context="context-market",
        generation=1,
        diagnostic_conditional=0.8,
        continuation_conditional=0.6,
        high_isolated=0.8,
        low_isolated=0.0,
        continuation_isolated=0.6,
    )
    ledger = ConservedResidualHeadroomLedger()
    state = ledger.append(ledger.empty_state(), closure)
    wrapper = ResidualHeadroomAdaptiveMarketProjector(
        delegate=_Delegate((actions[0], actions[2])),
        ledger=ledger,
        ledger_state=state,
        context_sha256=closure.context_sha256,
        base_prior_weight=0.0,
        headroom_weight=1.0,
    )
    request = ResidualPortfolioDecisionRequest(
        campaign_scope_sha256=_sha("campaign"),
        prior_state_sha256=_sha("prior"),
        decision_index=2,
        phase=SearchPhase.COMPOSITION,
        remaining_decisions=2,
        remaining_evaluations=4,
        evaluation_slots=2,
        expert_proposal_slots=(("test_expert", 2),),
        proposal_context=freeze_json({"opaque_context": True}),
        reference_escrow_slots=0,
    )
    projected = asyncio.run(
        wrapper.project(request, (), (), (), ())
    )
    score_by_action = {
        value.action_sha256: value.prior_score for value in projected
    }

    assert score_by_action[actions[0].action_sha256] == 1.0
    assert score_by_action[actions[2].action_sha256] == 0.0
    assert wrapper.state_sha256 != wrapper.delegate.state_sha256
    assert state.to_record()["predicted_values_admitted_to_archive"] is False

    cold_wrapper = ResidualHeadroomAdaptiveMarketProjector(
        delegate=_Delegate((actions[0], actions[2])),
        ledger=ledger,
        ledger_state=ledger.empty_state(),
        context_sha256=closure.context_sha256,
        base_prior_weight=0.0,
        headroom_weight=1.0,
    )
    cold = asyncio.run(
        cold_wrapper.project(request, (), (), (), ())
    )
    assert cold == (actions[0], actions[2])


def test_headroom_store_commits_with_compare_and_swap():
    closure, _actions, _decisions, _outcomes, _set_outcomes = _stage(
        context="context-store",
        generation=1,
        diagnostic_conditional=0.5,
        continuation_conditional=0.2,
        high_isolated=0.5,
        low_isolated=0.0,
        continuation_isolated=0.2,
    )
    ledger = ConservedResidualHeadroomLedger()
    initial = ledger.empty_state()
    store = InMemoryTransactionalResidualHeadroomStore(
        ledger=ledger,
        state=initial,
    )
    ack = asyncio.run(
        store.commit(
            expected_prior_state_sha256=initial.state_sha256,
            closure=closure,
        )
    )

    assert ack.prior_state_sha256 == initial.state_sha256
    assert ack.new_state_sha256 == store.state.state_sha256
    assert ack.closure_sha256 == closure.closure_sha256
    assert ack.durable is False
    assert len(store.state.closures) == 1

    with pytest.raises(
        ValueError,
        match="compare-and-swap failed",
    ):
        asyncio.run(
            store.commit(
                expected_prior_state_sha256=initial.state_sha256,
                closure=closure,
            )
        )


def test_durable_headroom_store_recovers_authenticated_state(
    tmp_path: Path,
):
    closure, _actions, _decisions, _outcomes, _set_outcomes = _stage(
        context="context-durable",
        generation=1,
        diagnostic_conditional=0.4,
        continuation_conditional=0.3,
        high_isolated=0.4,
        low_isolated=0.0,
        continuation_isolated=0.3,
    )
    ledger = ConservedResidualHeadroomLedger()
    path = tmp_path / "headroom.jsonl"
    store = DurableJsonlResidualHeadroomStore(
        path=path,
        ledger=ledger,
    )
    initial_sha256 = store.state.state_sha256
    ack = asyncio.run(
        store.commit(
            expected_prior_state_sha256=initial_sha256,
            closure=closure,
        )
    )
    recovered = DurableJsonlResidualHeadroomStore(
        path=path,
        ledger=ledger,
    )

    assert ack.durable is True
    assert recovered.state == store.state
    assert recovered.commit_acks == (ack,)
    assert recovered.state.closures == (closure,)
