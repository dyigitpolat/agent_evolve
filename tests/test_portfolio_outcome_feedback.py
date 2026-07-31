from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace

import pytest

from agent_evolve.application.outcome_relation import OutcomeRelation
from agent_evolve.application.portfolio_evolution import PortfolioEvolution
from agent_evolve.application.portfolio_outcome_feedback import (
    PortfolioActionOutcomeFeedback,
    PortfolioOutcomeFeedbackLedger,
    PortfolioOutcomeFeedbackReceipt,
    observe_selected_portfolio_forecasts,
)
from agent_evolve.application.portfolio_evolution import (
    PortfolioCandidateFailureEvidence,
    PortfolioMemberDisposition,
)
from agent_evolve.domain.outcome import FailureCode
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationScope,
    ForecastConfidenceBin,
    ForecastPredictionReceipt,
    MeaningfulDirectionRequest,
    observe_forecast,
)
from agent_evolve.policies.selection.meaningful_direction import (
    AbsoluteToleranceDirectionAdjudicator,
    MetricDirectionResolution,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection
from tests.test_portfolio_evolution import (
    _CandidateInfeasibilityEvaluator,
    _build_wave,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def test_policy_frame_preserves_experiment_stratum_and_separates_identity() -> None:
    engine_scope = ForecastCalibrationScope(
        model_profile_sha256=_sha("model"),
        prompt_definition_sha256=_sha("engine-prompt"),
        selector_policy_definition_sha256=_sha("engine-selector"),
        benchmark_sha256=_sha("benchmark"),
        session_sha256=_sha("session"),
    )

    runtime_scope = engine_scope.for_policy_frame(
        prompt_definition_sha256=_sha("runtime-prompt"),
        selector_policy_definition_sha256=_sha("runtime-selector"),
    )

    assert runtime_scope.model_profile_sha256 == engine_scope.model_profile_sha256
    assert runtime_scope.benchmark_sha256 == engine_scope.benchmark_sha256
    assert runtime_scope.session_sha256 == engine_scope.session_sha256
    assert runtime_scope.prompt_definition_sha256 == _sha("runtime-prompt")
    assert runtime_scope.selector_policy_definition_sha256 == _sha(
        "runtime-selector"
    )
    assert runtime_scope.scope_sha256 != engine_scope.scope_sha256


def _action(*, wave_index: int, option_id: str) -> PortfolioActionOutcomeFeedback:
    scope = ForecastCalibrationScope(
        model_profile_sha256=_sha("model"),
        prompt_definition_sha256=_sha("prompt"),
        selector_policy_definition_sha256=_sha("selector"),
        benchmark_sha256=_sha("benchmark"),
        session_sha256=_sha("session"),
    )
    option_identity = _sha(option_id)
    proposal = _sha(f"proposal:{wave_index}")
    parent_identity = _sha(f"parent:{wave_index}")
    parent_outcome = _sha(f"parent-outcome:{wave_index}")
    child_outcome = _sha(f"child-outcome:{wave_index}:{option_id}")
    prediction = ForecastPredictionReceipt(
        scope=scope,
        wave_index=wave_index,
        selector_decision_sha256=proposal,
        parent_candidate_identity_sha256=parent_identity,
        option_id=option_id,
        option_identity_sha256=option_identity,
        family="geometry",
        metric_id="cost",
        asserted_direction=MetricEffectDirection.DECREASE,
        confidence=ForecastConfidenceBin.MEDIUM,
    )
    request = MeaningfulDirectionRequest(
        benchmark_sha256=scope.benchmark_sha256,
        session_sha256=scope.session_sha256,
        wave_index=wave_index,
        parent_candidate_identity_sha256=parent_identity,
        option_id=option_id,
        option_identity_sha256=option_identity,
        metric_id="cost",
        parent_outcome_sha256=parent_outcome,
        child_outcome_sha256=child_outcome,
        parent_metric_value=10.0,
        child_metric_value=9.0,
    )
    observation = observe_forecast(
        prediction,
        request,
        AbsoluteToleranceDirectionAdjudicator(
            benchmark_sha256=scope.benchmark_sha256,
            session_sha256=scope.session_sha256,
            resolutions=(MetricDirectionResolution("cost", 0.0),),
        ),
    )
    return PortfolioActionOutcomeFeedback(
        wave_index=wave_index,
        request_sha256=_sha(f"request:{wave_index}"),
        ranked_decision_sha256=_sha(f"ranked:{wave_index}"),
        proposal_sha256=proposal,
        parent_candidate_id=f"candidate_parent_{wave_index}",
        parent_candidate_identity_sha256=parent_identity,
        parent_outcome_sha256=parent_outcome,
        candidate_id=f"candidate_child_{wave_index}",
        candidate_outcome_sha256=child_outcome,
        option_id=option_id,
        option_identity_sha256=option_identity,
        family="geometry",
        changed_paths=("$.shape.radius",),
        observations=(observation,),
        parent_relation=OutcomeRelation.BETTER,
        reward=0.25,
        dominates_parent=True,
        better_than_parent=True,
    )


def _receipt(*, wave_index: int, option_id: str) -> PortfolioOutcomeFeedbackReceipt:
    action = _action(wave_index=wave_index, option_id=option_id)
    return PortfolioOutcomeFeedbackReceipt(
        wave_index=wave_index,
        request_sha256=action.request_sha256,
        ranked_decision_sha256=action.ranked_decision_sha256,
        scope=action.observations[0].prediction.scope,
        actions=(action,),
    )


def test_prompt_history_is_prior_only_and_card_blind() -> None:
    ledger = PortfolioOutcomeFeedbackLedger()
    first = _receipt(wave_index=1, option_id="option.early")
    current = _receipt(wave_index=3, option_id="option.current")
    ledger.append(first)
    ledger.append(current)

    record = thaw_json(ledger.prompt_history(cutoff_wave_index_exclusive=3))
    encoded = str(record)
    assert [value["option_id"] for value in record["actions"]] == ["option.early"]
    assert "option.current" not in encoded
    assert "card_key" not in encoded
    assert "insight_id" not in encoded
    assert "rationale" not in encoded
    assert record["actions"][0]["metric_feedback"][0] == {
        "metric_id": "cost",
        "predicted_direction": "decrease",
        "confidence": "medium",
        "observed_direction": "decrease",
        "correctness": True,
    }


def test_calibration_snapshot_uses_only_prior_wave_observations() -> None:
    ledger = PortfolioOutcomeFeedbackLedger()
    first = _receipt(wave_index=1, option_id="option.early")
    current = _receipt(wave_index=3, option_id="option.current")
    ledger.append(first)
    ledger.append(current)
    scope = first.scope
    snapshot = ledger.calibration_snapshot(
        scope=scope,
        cutoff_wave_index_exclusive=3,
    )
    assert snapshot.observation_count == 1
    assert snapshot.correct_count == 1
    assert snapshot.observations[0].prediction.option_id == "option.early"


def test_candidate_infeasible_action_is_retained_but_never_calibrated() -> None:
    scored = _action(wave_index=1, option_id="option.scored")
    failure = PortfolioCandidateFailureEvidence(
        detailed_evaluation_sha256=_sha("detailed-infeasible"),
        failure_code=FailureCode.EVALUATOR_DECLARED_INFEASIBLE,
        failure_message_sha256=_sha("private evaluator diagnostic"),
        retryable=False,
        exception_type=None,
        diagnostics_artifact_id=None,
    )
    infeasible = PortfolioActionOutcomeFeedback(
        wave_index=1,
        request_sha256=scored.request_sha256,
        ranked_decision_sha256=scored.ranked_decision_sha256,
        proposal_sha256=scored.proposal_sha256,
        parent_candidate_id=scored.parent_candidate_id,
        parent_candidate_identity_sha256=scored.parent_candidate_identity_sha256,
        parent_outcome_sha256=scored.parent_outcome_sha256,
        candidate_id="candidate_child_infeasible",
        candidate_outcome_sha256=_sha("infeasible-outcome"),
        option_id="option.infeasible",
        option_identity_sha256=_sha("option.infeasible"),
        family="geometry",
        changed_paths=("$.shape.radius",),
        observations=(),
        parent_relation=None,
        reward=-1.0,
        dominates_parent=False,
        better_than_parent=False,
        disposition=PortfolioMemberDisposition.CANDIDATE_INFEASIBLE,
        candidate_failure=failure,
    )
    receipt = PortfolioOutcomeFeedbackReceipt(
        wave_index=1,
        request_sha256=scored.request_sha256,
        ranked_decision_sha256=scored.ranked_decision_sha256,
        scope=scored.observations[0].prediction.scope,
        actions=(scored, infeasible),
    )
    ledger = PortfolioOutcomeFeedbackLedger()
    ledger.append(receipt)

    assert ledger.observations == scored.observations
    assert ledger.calibration_snapshot(
        scope=receipt.scope,
        cutoff_wave_index_exclusive=2,
    ).observation_count == 1
    no_yield = infeasible.to_prompt_record()
    assert no_yield["outcome_status"] == "candidate_infeasible_no_yield"
    assert no_yield["metric_feedback"] == []
    assert no_yield["parent_relation"] is None
    record = receipt.to_record()
    assert record["ranked_itt_action_count"] == 2
    assert record["scored_action_count"] == 1
    assert record["candidate_infeasible_action_count"] == 1

    with pytest.raises(ValueError, match="cannot publish a parent relation"):
        replace(infeasible, parent_relation=OutcomeRelation.WORSE)
    with pytest.raises(ValueError):
        replace(infeasible, observations=scored.observations)


def test_real_forecast_join_excludes_infeasible_rank_from_calibration() -> None:
    async def scenario():
        evaluator = _CandidateInfeasibilityEvaluator()
        ids, _, _, memory, engine, selector, wave = await _build_wave(
            "outcome_feedback_infeasible",
            detailed_evaluator=evaluator,
        )
        evaluator.reset_evidence()
        result = await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
            memory=memory,
        ).run(wave, defer_memory_credit=True)
        decision = result.selection_decision
        assert decision is not None
        scope = ForecastCalibrationScope(
            model_profile_sha256=_sha("model"),
            prompt_definition_sha256=_sha("prompt"),
            selector_policy_definition_sha256=_sha("selector"),
            benchmark_sha256=_sha("benchmark"),
            session_sha256=_sha("session"),
        )
        proposal = _sha("infeasible-join-proposal")
        predictions = tuple(
            ForecastPredictionReceipt(
                scope=scope,
                wave_index=1,
                selector_decision_sha256=proposal,
                parent_candidate_identity_sha256=(
                    wave.parent.occurrence.configuration_hash
                ),
                option_id=selected.option_id,
                option_identity_sha256=selected.option_identity_sha256,
                family=selected.family,
                metric_id="loss",
                asserted_direction=MetricEffectDirection.DECREASE,
                confidence=ForecastConfidenceBin.MEDIUM,
            )
            for selected in decision.members
        )
        feedback = observe_selected_portfolio_forecasts(
            wave_index=1,
            parent=wave.parent,
            result=result,
            selected_predictions=predictions,
            adjudicator=AbsoluteToleranceDirectionAdjudicator(
                benchmark_sha256=scope.benchmark_sha256,
                session_sha256=scope.session_sha256,
                resolutions=(MetricDirectionResolution("loss", 0.0),),
            ),
        )
        return result, feedback

    result, feedback = asyncio.run(scenario())
    assert len(result.receipt.members) == len(feedback.actions) == 3
    assert sum(len(value.observations) for value in feedback.actions) == 2
    no_yield = feedback.actions[2]
    assert no_yield.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE
    assert no_yield.observations == no_yield.metric_transitions == ()
    assert no_yield.parent_relation is None
    assert no_yield.candidate_failure == result.receipt.members[2].candidate_failure
    assert feedback.to_record()["candidate_infeasible_action_count"] == 1
