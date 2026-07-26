from __future__ import annotations

import hashlib
from dataclasses import replace

from agent_evolve.application.action_archive_value import (
    NormalizedResidualFrontierCell,
    ResidualCellExpectedHypervolumeUtility,
)
from agent_evolve.application.action_target_realization import (
    ResidualTargetClosurePortfolioUtility,
    TargetMetricAlias,
    assess_action_target_realization,
)
from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
)
from agent_evolve.ports.action_allocation import (
    ForecastPortfolioUtilityInput,
    ForecastQuantile,
)
from agent_evolve.ports.action_forecast import (
    MetricForecastScale,
    ParentMetricValue,
    ResolvedActionForecast,
    ResolvedActionForecastBatch,
    ResolvedActionMetricForecast,
)
from agent_evolve.ports.frontier_target import (
    ObjectiveSpaceTarget,
    ObjectiveSpaceTargetAxis,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _target() -> ObjectiveSpaceTarget:
    return ObjectiveSpaceTarget(
        campaign_target_sha256=_sha("target"),
        purpose="test_action_to_frontier_realization",
        axes=(
            ObjectiveSpaceTargetAxis(
                metric_id="cost",
                goal="min",
                ideal=0.0,
                reference=200.0,
                parent_value=100.0,
                aspiration_value=90.0,
                signed_parent_to_aspiration_delta=-10.0,
                improving_raw_delta_sign="negative",
            ),
            ObjectiveSpaceTargetAxis(
                metric_id="quality",
                goal="max",
                ideal=30.0,
                reference=0.0,
                parent_value=10.0,
                aspiration_value=15.0,
                signed_parent_to_aspiration_delta=5.0,
                improving_raw_delta_sign="positive",
            ),
        ),
    )


def _member(
    option_id: str,
    *,
    cost_delta: float,
    quality_delta: float,
    probability_valid: float = 1.0,
) -> ResolvedActionForecast:
    def metric(metric_id: str, delta: float) -> ResolvedActionMetricForecast:
        return ResolvedActionMetricForecast(
            metric_id=metric_id,
            p10_delta=delta,
            p50_delta=delta,
            p90_delta=delta,
            confidence=0.75,
            citations=(),
        )

    return ResolvedActionForecast(
        option_id=option_id,
        option_identity_sha256=_sha(f"identity:{option_id}"),
        child_configuration_sha256=_sha(f"child:{option_id}"),
        family="fixture",
        probability_valid=probability_valid,
        metric_forecasts=(
            metric("objective:cost", cost_delta),
            metric("objective:quality", quality_delta),
        ),
    )


def _batch() -> ResolvedActionForecastBatch:
    return ResolvedActionForecastBatch(
        request_sha256=_sha("request"),
        context_sha256=_sha("context"),
        optimization_semantics_definition_sha256=_sha("semantics"),
        action_semantics_definition_sha256=_sha("actions"),
        finite_contract_identity_sha256=_sha("contract"),
        card_snapshot_sha256=_sha("cards"),
        forecasts=(
            _member("action.exact", cost_delta=-10.0, quality_delta=5.0),
            _member("action.partial", cost_delta=-5.0, quality_delta=2.5),
            _member("action.regress", cost_delta=5.0, quality_delta=-2.0),
        ),
        policy_id="fixture_forecaster",
        policy_version=1,
        policy_definition_sha256=_sha("forecaster"),
    )


def _aliases() -> tuple[TargetMetricAlias, ...]:
    return (
        TargetMetricAlias("cost", "objective:cost"),
        TargetMetricAlias("quality", "objective:quality"),
    )


def _semantics() -> OptimizationSemantics:
    return OptimizationSemantics(
        semantics_id="target_realization_fixture",
        semantics_version=1,
        metrics=(
            MetricSemantics(
                metric_id="objective:cost",
                name="cost",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MINIMIZE,
                definition="Fixture cost.",
                aggregation="One scalar.",
                witness_interpretation="Lower is better.",
            ),
            MetricSemantics(
                metric_id="objective:quality",
                name="quality",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MAXIMIZE,
                definition="Fixture quality.",
                aggregation="One scalar.",
                witness_interpretation="Higher is better.",
            ),
        ),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.PARETO,
            metric_priority=("objective:cost", "objective:quality"),
            description="Pareto order.",
            equivalence="Equal objective vectors are equivalent.",
            policy_id="fixture_pareto",
            policy_version=1,
            definition_sha256=_sha("pareto"),
        ),
    )


def test_target_realization_audit_distinguishes_attainment_partial_and_regression() -> None:
    assessed = assess_action_target_realization(
        target=_target(),
        forecasts=_batch(),
        aliases=_aliases(),
    )
    by_id = {value.option_id: value for value in assessed}

    exact = by_id["action.exact"].scenario(ForecastQuantile.P50)
    partial = by_id["action.partial"].scenario(ForecastQuantile.P50)
    regress = by_id["action.regress"].scenario(ForecastQuantile.P50)

    assert exact.attains_or_dominates_aspiration
    assert exact.normalized_shortfall_l1 == 0.0
    assert partial.shortfall_reduction_l1 > 0.0
    assert not partial.attains_or_dominates_aspiration
    assert regress.shortfall_reduction_l1 < 0.0


def test_target_closure_utility_rewards_a_realizable_bridge_and_is_monotone() -> None:
    batch = _batch()
    by_id = {value.option_id: value for value in batch.forecasts}
    utility = ResidualTargetClosurePortfolioUtility(
        target=_target(),
        aliases=_aliases(),
    )

    def score(option_ids: tuple[str, ...]) -> float:
        return utility(
            ForecastPortfolioUtilityInput(
                optimization_semantics=_semantics(),
                parent_metric_values=(
                    ParentMetricValue("objective:cost", 100.0),
                    ParentMetricValue("objective:quality", 10.0),
                ),
                metric_scales=(
                    MetricForecastScale("objective:cost", 10.0, _sha("cost-scale")),
                    MetricForecastScale(
                        "objective:quality", 5.0, _sha("quality-scale")
                    ),
                ),
                members=tuple(by_id[value] for value in option_ids),
                quantile=ForecastQuantile.P50,
            )
        )

    exact = score(("action.exact",))
    partial = score(("action.partial",))
    regression = score(("action.regress",))

    assert exact > partial > regression
    assert score(("action.exact", "action.regress")) == exact


def test_residual_cell_expected_hypervolume_is_archive_aligned_and_probabilistic() -> None:
    batch = _batch()
    by_id = {value.option_id: value for value in batch.forecasts}
    target = _target()
    utility = ResidualCellExpectedHypervolumeUtility(
        target=target,
        cell=NormalizedResidualFrontierCell(
            campaign_target_sha256=target.campaign_target_sha256,
            cell_sha256=_sha("cell"),
            geometry_sha256=_sha("geometry"),
            metric_ids=target.metric_ids,
            anchor_points=((0.4, 0.8), (0.7, 0.4)),
        ),
        aliases=_aliases(),
    )

    def score(members: tuple[ResolvedActionForecast, ...]) -> float:
        return utility(
            ForecastPortfolioUtilityInput(
                optimization_semantics=_semantics(),
                parent_metric_values=(
                    ParentMetricValue("objective:cost", 100.0),
                    ParentMetricValue("objective:quality", 10.0),
                ),
                metric_scales=(
                    MetricForecastScale("objective:cost", 10.0, _sha("cost-scale")),
                    MetricForecastScale(
                        "objective:quality", 5.0, _sha("quality-scale")
                    ),
                ),
                members=members,
                quantile=ForecastQuantile.P50,
            )
        )

    exact = by_id["action.exact"]
    partial = by_id["action.partial"]
    regress = by_id["action.regress"]
    assert score((exact,)) > score((partial,)) > score((regress,))
    assert score((exact, partial)) >= score((exact,))
    assert score((replace(exact, probability_valid=0.0),)) == 0.0
    assert utility.binding().policy_id == "residual_cell_expected_hypervolume"
