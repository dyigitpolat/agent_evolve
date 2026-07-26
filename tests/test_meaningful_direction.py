from __future__ import annotations

import hashlib

import pytest

from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
)
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


def _request(*, child: float, metric_id: str = "cost") -> MeaningfulDirectionRequest:
    return MeaningfulDirectionRequest(
        benchmark_sha256=_sha("benchmark"),
        session_sha256=_sha("session"),
        wave_index=2,
        parent_candidate_identity_sha256=_sha("parent"),
        option_id="option.1",
        option_identity_sha256=_sha("option.1"),
        metric_id=metric_id,
        parent_outcome_sha256=_sha("parent-outcome"),
        child_outcome_sha256=_sha(f"child-outcome:{child}"),
        parent_metric_value=1.0,
        child_metric_value=child,
    )


def _adjudicator() -> AbsoluteToleranceDirectionAdjudicator:
    return AbsoluteToleranceDirectionAdjudicator(
        benchmark_sha256=_sha("benchmark"),
        session_sha256=_sha("session"),
        resolutions=(MetricDirectionResolution("cost", 1.0e-6),),
    )


@pytest.mark.parametrize(
    ("child", "expected"),
    [
        (1.0 + 0.5e-6, MetricEffectDirection.UNCHANGED),
        (1.0 - 0.5e-6, MetricEffectDirection.UNCHANGED),
        (1.0 + 2.0e-6, MetricEffectDirection.INCREASE),
        (1.0 - 2.0e-6, MetricEffectDirection.DECREASE),
    ],
)
def test_absolute_resolution_classifies_meaningful_direction(
    child: float,
    expected: MetricEffectDirection,
) -> None:
    request = _request(child=child)
    receipt = _adjudicator().adjudicate(request)
    receipt.require_request(request)
    assert receipt.actual_direction is expected


def test_forecast_observation_closes_the_prediction_outcome_join() -> None:
    request = _request(child=0.9)
    scope = ForecastCalibrationScope(
        model_profile_sha256=_sha("model"),
        prompt_definition_sha256=_sha("prompt"),
        selector_policy_definition_sha256=_sha("selector"),
        benchmark_sha256=request.benchmark_sha256,
        session_sha256=request.session_sha256,
    )
    prediction = ForecastPredictionReceipt(
        scope=scope,
        wave_index=request.wave_index,
        selector_decision_sha256=_sha("decision"),
        parent_candidate_identity_sha256=request.parent_candidate_identity_sha256,
        option_id=request.option_id,
        option_identity_sha256=request.option_identity_sha256,
        family="atomic",
        metric_id=request.metric_id,
        asserted_direction=MetricEffectDirection.DECREASE,
        confidence=ForecastConfidenceBin.HIGH,
    )
    observation = observe_forecast(prediction, request, _adjudicator())
    assert observation.correctness is True
    assert observation.adjudication.actual_direction is MetricEffectDirection.DECREASE


def test_semantics_factory_requires_explicit_objective_tolerance() -> None:
    ordering = OutcomeOrderingSemantics(
        kind=OutcomeOrderingKind.PARETO,
        metric_priority=("objective:cost",),
        description="Lower cost is better.",
        equivalence="Exact relation for this isolated factory test.",
        policy_id="objective_pareto",
        policy_version=1,
        definition_sha256=_sha("relation"),
    )
    missing = OptimizationSemantics(
        semantics_id="direction_test",
        semantics_version=1,
        metrics=(
            MetricSemantics(
                metric_id="objective:cost",
                name="cost",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MINIMIZE,
                definition="Synthetic cost.",
                aggregation="One scalar.",
                witness_interpretation="Lower is better.",
                tolerance=None,
            ),
        ),
        outcome_ordering=ordering,
    )
    with pytest.raises(ValueError, match="explicit direction tolerances"):
        AbsoluteToleranceDirectionAdjudicator.from_optimization_semantics(
            benchmark_sha256=_sha("benchmark"),
            session_sha256=_sha("session"),
            semantics=missing,
        )
