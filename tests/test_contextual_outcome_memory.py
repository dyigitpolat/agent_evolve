from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent_evolve.application.outcome_relation import OutcomeRelation
from agent_evolve.application.campaign_contextual_outcomes import (
    ContextualOutcomeCampaignEnricher,
)
from agent_evolve.application.portfolio_outcome_feedback import (
    ContextualOutcomeHistoryReceipt,
    ContextualOutcomeQuery,
    DecisionMetricTransition,
    OutcomeTransferScope,
    PortfolioActionOutcomeFeedback,
    PortfolioOutcomeFeedbackLedger,
    PortfolioOutcomeFeedbackReceipt,
)
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


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _action(
    *,
    wave_index: int,
    option_id: str,
    parent_candidate_id: str,
    parent_identity: str,
    parent_value: float = 10.0,
    child_value: float = 9.0,
    child_candidate_id: str | None = None,
) -> PortfolioActionOutcomeFeedback:
    scope = ForecastCalibrationScope(
        model_profile_sha256=_sha("model"),
        prompt_definition_sha256=_sha("prompt"),
        selector_policy_definition_sha256=_sha("selector"),
        benchmark_sha256=_sha("benchmark"),
        session_sha256=_sha("session"),
    )
    option_identity = _sha(option_id)
    proposal = _sha(f"proposal:{wave_index}:{option_id}")
    parent_outcome = _sha(f"parent-outcome:{parent_candidate_id}")
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
        parent_metric_value=parent_value,
        child_metric_value=child_value,
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
        request_sha256=_sha(f"request:{wave_index}:{option_id}"),
        ranked_decision_sha256=_sha(f"ranked:{wave_index}:{option_id}"),
        proposal_sha256=proposal,
        parent_candidate_id=parent_candidate_id,
        parent_candidate_identity_sha256=parent_identity,
        parent_outcome_sha256=parent_outcome,
        candidate_id=(
            child_candidate_id
            if child_candidate_id is not None
            else f"candidate_child_{wave_index}_{option_id.replace('.', '_')}"
        ),
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
        metric_transitions=(
            DecisionMetricTransition(
                metric_id="cost",
                parent_value=parent_value,
                child_value=child_value,
                actual_direction=observation.adjudication.actual_direction,
                adjudication_receipt_sha256=(
                    observation.adjudication.receipt_sha256
                ),
            ),
        ),
    )


def _receipt(action: PortfolioActionOutcomeFeedback) -> PortfolioOutcomeFeedbackReceipt:
    return PortfolioOutcomeFeedbackReceipt(
        wave_index=action.wave_index,
        request_sha256=action.request_sha256,
        ranked_decision_sha256=action.ranked_decision_sha256,
        scope=action.observations[0].prediction.scope,
        actions=(action,),
    )


def _query(*, include_cross: bool = True) -> ContextualOutcomeQuery:
    return ContextualOutcomeQuery(
        current_parent_candidate_id="candidate_parent_current",
        current_parent_configuration_sha256=_sha("parent-current"),
        cutoff_wave_index_exclusive=4,
        lineage_candidate_ids=("candidate_parent_ancestor",),
        families=("geometry",),
        changed_paths=("$.shape",),
        max_actions=8,
        include_cross_lineage_analogies=include_cross,
    )


def _ledger() -> PortfolioOutcomeFeedbackLedger:
    ledger = PortfolioOutcomeFeedbackLedger()
    same_parent = _action(
        wave_index=1,
        option_id="option.same",
        parent_candidate_id="candidate_parent_current",
        parent_identity=_sha("parent-current"),
    )
    same_lineage = _action(
        wave_index=2,
        option_id="option.lineage",
        parent_candidate_id="candidate_parent_ancestor",
        parent_identity=_sha("parent-ancestor"),
    )
    cross_lineage = _action(
        wave_index=3,
        option_id="option.cross",
        parent_candidate_id="candidate_parent_other",
        parent_identity=_sha("parent-other"),
    )
    same_generation = _action(
        wave_index=4,
        option_id="option.future",
        parent_candidate_id="candidate_parent_current",
        parent_identity=_sha("parent-current"),
    )
    for action in (same_parent, same_lineage, cross_lineage, same_generation):
        ledger.append(_receipt(action))
    return ledger


def test_contextual_history_labels_transfer_and_never_leaks_current_wave() -> None:
    receipt = _ledger().contextual_history(_query())
    assert tuple(value.option_id for value in receipt.actions) == (
        "option.same",
        "option.lineage",
        "option.cross",
    )
    assert receipt.transfer_scopes == (
        OutcomeTransferScope.SAME_PARENT,
        OutcomeTransferScope.SAME_LINEAGE,
        OutcomeTransferScope.CROSS_LINEAGE_ANALOGY,
    )
    assert "option.future" not in str(thaw_json(receipt.to_prompt_record()))


def test_action_whose_child_is_current_parent_is_same_lineage() -> None:
    ledger = PortfolioOutcomeFeedbackLedger()
    predecessor = _action(
        wave_index=1,
        option_id="option.predecessor",
        parent_candidate_id="candidate_parent_previous",
        parent_identity=_sha("parent-previous"),
        child_candidate_id="candidate_parent_current",
    )
    ledger.append(_receipt(predecessor))
    receipt = ledger.contextual_history(_query(include_cross=False))
    assert receipt.actions == (predecessor,)
    assert receipt.transfer_scopes == (OutcomeTransferScope.SAME_LINEAGE,)


def test_contextual_history_keeps_exact_numeric_baseline_and_epistemic_scope() -> None:
    receipt = _ledger().contextual_history(_query())
    record = thaw_json(receipt.to_prompt_record())
    first = record["actions"][0]
    assert first["transfer_scope"] == "same_parent"
    assert first["applicability"] == "direct_same_parent_evidence"
    assert first["source_parent_configuration_sha256"] == _sha("parent-current")
    assert first["metric_feedback"][0] == {
        "metric_id": "cost",
        "predicted_direction": "decrease",
        "confidence": "medium",
        "observed_direction": "decrease",
        "correctness": True,
        "parent_value_hex": (10.0).hex(),
        "child_value_hex": (9.0).hex(),
        "delta_hex": (-1.0).hex(),
    }
    cross = record["actions"][2]
    assert cross["applicability"] == "cross_lineage_analogy_requires_revalidation"
    encoded = str(record)
    assert "card_key" not in encoded
    assert "insight_id" not in encoded
    assert "design_rationale" not in encoded
    assert record["epistemic_status"] == (
        "observational_predictive_history_not_causal_credit"
    )
    assert record["receipt_sha256"] == receipt.receipt_sha256


def test_cross_lineage_evidence_can_be_excluded_without_changing_prompt_shape() -> None:
    receipt = _ledger().contextual_history(_query(include_cross=False))
    assert tuple(value.option_id for value in receipt.actions) == (
        "option.same",
        "option.lineage",
    )
    assert OutcomeTransferScope.CROSS_LINEAGE_ANALOGY not in receipt.transfer_scopes


def test_cross_lineage_transfer_is_opt_in_by_default() -> None:
    query = ContextualOutcomeQuery(
        current_parent_candidate_id="candidate_parent_current",
        current_parent_configuration_sha256=_sha("parent-current"),
        cutoff_wave_index_exclusive=4,
        lineage_candidate_ids=("candidate_parent_ancestor",),
    )
    assert query.include_cross_lineage_analogies is False
    default_history = _ledger().contextual_history(query)
    assert tuple(action.option_id for action in default_history.actions) == (
        "option.same",
        "option.lineage",
    )
    assert (
        ContextualOutcomeCampaignEnricher(
            ledger=PortfolioOutcomeFeedbackLedger()
        ).include_cross_lineage_analogies
        is False
    )


def test_contextual_history_is_deterministic_under_ledger_append_order() -> None:
    first = _ledger()
    second = PortfolioOutcomeFeedbackLedger()
    for receipt in reversed(first.receipts):
        second.append(receipt)
    first_result = first.contextual_history(_query())
    second_result = second.contextual_history(_query())
    assert first_result.receipt_sha256 == second_result.receipt_sha256
    assert first_result.to_prompt_record() == second_result.to_prompt_record()


def test_metric_transition_must_match_adjudication() -> None:
    action = _action(
        wave_index=1,
        option_id="option.invalid",
        parent_candidate_id="candidate_parent_current",
        parent_identity=_sha("parent-current"),
    )
    transition = action.metric_transitions[0]
    try:
        replace(
            action,
            metric_transitions=(
                replace(
                    transition,
                    actual_direction=MetricEffectDirection.INCREASE,
                ),
            ),
        )
    except ValueError as error:
        assert "differs from its adjudication" in str(error)
    else:  # pragma: no cover - explicit guard for the contract.
        raise AssertionError("foreign metric transition was accepted")


def test_metric_transition_values_must_match_numeric_adjudication_request() -> None:
    action = _action(
        wave_index=1,
        option_id="option.invalid_values",
        parent_candidate_id="candidate_parent_current",
        parent_identity=_sha("parent-current"),
    )
    with pytest.raises(ValueError, match="numeric values differ"):
        replace(
            action,
            metric_transitions=(
                replace(action.metric_transitions[0], parent_value=11.0),
            ),
        )


def test_contextual_history_rejects_forged_transfer_scope_label() -> None:
    receipt = _ledger().contextual_history(_query())
    with pytest.raises(ValueError, match="differ from query lineage"):
        ContextualOutcomeHistoryReceipt(
            query=receipt.query,
            actions=receipt.actions,
            transfer_scopes=(
                OutcomeTransferScope.SAME_LINEAGE,
                *receipt.transfer_scopes[1:],
            ),
        )


def test_contextual_query_rejects_noncanonical_candidate_paths() -> None:
    with pytest.raises(TypeError, match="candidate-path tuple"):
        replace(_query(), changed_paths=("$.shape..radius",))
