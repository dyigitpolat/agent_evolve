"""Workload-neutral projection from action forecasts to frontier closure.

The residual-frontier planner operates in objective space while the finite
action forecaster estimates parent-relative metric deltas.  This module is the
missing bridge: it converts those authenticated, provider-neutral values into
one-sided aspiration shortfall and exposes an identified portfolio utility for
the existing forecast allocator.  It neither evaluates a candidate nor knows
workload, model, provider, or option-family names.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from agent_evolve.ports.action_allocation import (
    ForecastPortfolioUtilityBinding,
    ForecastPortfolioUtilityInput,
    ForecastQuantile,
)
from agent_evolve.ports.action_forecast import (
    ResolvedActionForecast,
    ResolvedActionForecastBatch,
    ResolvedActionMetricForecast,
)
from agent_evolve.ports.frontier_target import (
    ObjectiveSpaceTarget,
)


TARGET_CLOSURE_PORTFOLIO_UTILITY_ID = "residual_target_closure"
TARGET_CLOSURE_PORTFOLIO_UTILITY_VERSION = 1
TARGET_CLOSURE_PORTFOLIO_UTILITY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:residual-target-closure-portfolio-utility:v1;"
    b"inputs=authenticated-objective-space-target-and-resolved-action-forecasts;"
    b"child=target-parent-plus-forecast-delta;"
    b"frame=affine-lower-is-better;shortfall=positive-part-child-minus-aspiration;"
    b"member-value=validity-times-parent-minus-child-l1-shortfall-plus-attainment;"
    b"set-value=best-plus-harmonic-redundancy;"
    b"workload-model-provider-identifiers=false;real-outcomes=false"
).hexdigest()
_TARGET_CLOSURE_BINDING_DOMAIN = (
    b"agent-evolve:residual-target-closure-utility-binding:v1\x00"
)


def _finite(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite exact float")
    return value


@dataclass(frozen=True, slots=True)
class TargetMetricAlias:
    """Explicit bridge when forecast and archive metric identifiers differ."""

    target_metric_id: str
    forecast_metric_id: str

    def __post_init__(self) -> None:
        for name in ("target_metric_id", "forecast_metric_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class TargetClosureScenario:
    """Aspiration-closure statistics under one forecast quantile scenario."""

    quantile: ForecastQuantile
    normalized_shortfalls: tuple[tuple[str, float], ...]
    normalized_shortfall_l1: float
    normalized_shortfall_linf: float
    shortfall_reduction_l1: float
    attains_or_dominates_aspiration: bool

    def __post_init__(self) -> None:
        if type(self.quantile) is not ForecastQuantile:
            raise TypeError("quantile must be an exact ForecastQuantile")
        if type(self.normalized_shortfalls) is not tuple or not (
            self.normalized_shortfalls
        ):
            raise ValueError("normalized_shortfalls must be a non-empty tuple")
        metric_ids = tuple(value[0] for value in self.normalized_shortfalls)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("normalized shortfalls must be unique and canonical")
        for metric_id, value in self.normalized_shortfalls:
            if type(metric_id) is not str or not metric_id:
                raise ValueError("shortfall metric_id must be non-empty")
            _finite(value, name="normalized shortfall")
            if value < 0.0:
                raise ValueError("normalized shortfall cannot be negative")
        for name in (
            "normalized_shortfall_l1",
            "normalized_shortfall_linf",
            "shortfall_reduction_l1",
        ):
            _finite(getattr(self, name), name=name)
        expected_l1 = sum(value for _, value in self.normalized_shortfalls)
        expected_linf = max(value for _, value in self.normalized_shortfalls)
        if self.normalized_shortfall_l1 != expected_l1:
            raise ValueError("L1 shortfall differs from its metric cells")
        if self.normalized_shortfall_linf != expected_linf:
            raise ValueError("Linf shortfall differs from its metric cells")
        if type(self.attains_or_dominates_aspiration) is not bool:
            raise TypeError("aspiration attainment must be an exact bool")
        if self.attains_or_dominates_aspiration != (expected_linf == 0.0):
            raise ValueError("aspiration attainment differs from shortfall")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "quantile": self.quantile.value,
            "normalized_shortfalls_hex": [
                {"metric_id": metric_id, "value_hex": value.hex()}
                for metric_id, value in self.normalized_shortfalls
            ],
            "normalized_shortfall_l1_hex": self.normalized_shortfall_l1.hex(),
            "normalized_shortfall_linf_hex": self.normalized_shortfall_linf.hex(),
            "shortfall_reduction_l1_hex": self.shortfall_reduction_l1.hex(),
            "attains_or_dominates_aspiration": (
                self.attains_or_dominates_aspiration
            ),
        }


@dataclass(frozen=True, slots=True)
class ActionTargetRealization:
    """Complete target-closure audit for one finite action forecast."""

    option_id: str
    option_identity_sha256: str
    probability_valid: float
    scenarios: tuple[TargetClosureScenario, ...]

    def __post_init__(self) -> None:
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be non-empty")
        if type(self.option_identity_sha256) is not str or len(
            self.option_identity_sha256
        ) != 64:
            raise ValueError("option_identity_sha256 must be a SHA-256 digest")
        _finite(self.probability_valid, name="probability_valid")
        if not 0.0 <= self.probability_valid <= 1.0:
            raise ValueError("probability_valid must lie in [0,1]")
        if type(self.scenarios) is not tuple or any(
            type(value) is not TargetClosureScenario for value in self.scenarios
        ):
            raise TypeError("scenarios must contain exact target-closure values")
        for value in self.scenarios:
            value.__post_init__()
        if tuple(value.quantile for value in self.scenarios) != (
            ForecastQuantile.P10,
            ForecastQuantile.P50,
            ForecastQuantile.P90,
        ):
            raise ValueError("scenarios must use canonical p10/p50/p90 order")

    def scenario(self, quantile: ForecastQuantile) -> TargetClosureScenario:
        self.__post_init__()
        return next(value for value in self.scenarios if value.quantile is quantile)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "probability_valid_hex": self.probability_valid.hex(),
            "scenarios": [value.to_record() for value in self.scenarios],
        }


def _metric_delta(
    forecast: ResolvedActionMetricForecast,
    quantile: ForecastQuantile,
) -> float:
    if quantile is ForecastQuantile.P10:
        return forecast.p10_delta
    if quantile is ForecastQuantile.P50:
        return forecast.p50_delta
    return forecast.p90_delta


def _alias_map(
    target: ObjectiveSpaceTarget,
    aliases: tuple[TargetMetricAlias, ...],
) -> dict[str, str]:
    target.__post_init__()
    if type(aliases) is not tuple or any(
        type(value) is not TargetMetricAlias for value in aliases
    ):
        raise TypeError("aliases must contain exact TargetMetricAlias values")
    for value in aliases:
        value.__post_init__()
    if aliases:
        target_ids = tuple(value.target_metric_id for value in aliases)
        forecast_ids = tuple(value.forecast_metric_id for value in aliases)
        if target_ids != tuple(sorted(set(target_ids))):
            raise ValueError("target aliases must be unique and canonical")
        if len(set(forecast_ids)) != len(forecast_ids):
            raise ValueError("forecast aliases must be one-to-one")
        if target_ids != target.metric_ids:
            raise ValueError("aliases must exactly cover the objective target")
        return {value.target_metric_id: value.forecast_metric_id for value in aliases}
    return {metric_id: metric_id for metric_id in target.metric_ids}


def _scenario(
    *,
    target: ObjectiveSpaceTarget,
    forecast: ResolvedActionForecast,
    aliases: dict[str, str],
    quantile: ForecastQuantile,
) -> TargetClosureScenario:
    forecast_by_id = {value.metric_id: value for value in forecast.metric_forecasts}
    required = set(aliases.values())
    if not required.issubset(forecast_by_id):
        raise ValueError("action forecast omits an objective target metric")
    shortfalls: list[tuple[str, float]] = []
    parent_l1 = 0.0
    for axis in target.axes:
        metric_forecast = forecast_by_id[aliases[axis.metric_id]]
        child_value = axis.parent_value + _metric_delta(metric_forecast, quantile)
        child_shortfall = max(
            0.0,
            axis.normalize(child_value) - axis.aspiration_normalized,
        )
        shortfalls.append((axis.metric_id, child_shortfall))
        parent_l1 += axis.parent_shortfall
    canonical = tuple(sorted(shortfalls))
    shortfall_l1 = sum(value for _, value in canonical)
    shortfall_linf = max(value for _, value in canonical)
    return TargetClosureScenario(
        quantile=quantile,
        normalized_shortfalls=canonical,
        normalized_shortfall_l1=shortfall_l1,
        normalized_shortfall_linf=shortfall_linf,
        shortfall_reduction_l1=parent_l1 - shortfall_l1,
        attains_or_dominates_aspiration=shortfall_linf == 0.0,
    )


def assess_action_target_realization(
    *,
    target: ObjectiveSpaceTarget,
    forecasts: ResolvedActionForecastBatch,
    aliases: tuple[TargetMetricAlias, ...] = (),
) -> tuple[ActionTargetRealization, ...]:
    """Audit every forecast against a shared target without evaluator access."""

    if type(target) is not ObjectiveSpaceTarget:
        raise TypeError("target must be an exact ObjectiveSpaceTarget")
    target.__post_init__()
    if type(forecasts) is not ResolvedActionForecastBatch:
        raise TypeError("forecasts must be an exact ResolvedActionForecastBatch")
    forecasts.__post_init__()
    mapping = _alias_map(target, aliases)
    return tuple(
        ActionTargetRealization(
            option_id=forecast.option_id,
            option_identity_sha256=forecast.option_identity_sha256,
            probability_valid=forecast.probability_valid,
            scenarios=tuple(
                _scenario(
                    target=target,
                    forecast=forecast,
                    aliases=mapping,
                    quantile=quantile,
                )
                for quantile in (
                    ForecastQuantile.P10,
                    ForecastQuantile.P50,
                    ForecastQuantile.P90,
                )
            ),
        )
        for forecast in forecasts.forecasts
    )


@dataclass(frozen=True, slots=True)
class ResidualTargetClosurePortfolioUtility:
    """Plug-in utility for the existing forecast-based greedy set allocator.

    The best predicted bridge carries most of the utility.  A bounded harmonic
    redundancy term rewards additional independent chances without allowing a
    large collection of mediocre actions to swamp the best bridge.
    """

    target: ObjectiveSpaceTarget
    aliases: tuple[TargetMetricAlias, ...] = ()
    attainment_bonus: float = 0.25
    redundancy_weight: float = 0.25

    def __post_init__(self) -> None:
        if type(self.target) is not ObjectiveSpaceTarget:
            raise TypeError("target must be an exact ObjectiveSpaceTarget")
        self.target.__post_init__()
        _alias_map(self.target, self.aliases)
        for name in ("attainment_bonus", "redundancy_weight"):
            value = _finite(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def binding_definition_sha256(self) -> str:
        """Bind the reusable policy law to this exact target configuration.

        ``ActionAllocationRequest`` records only the identified utility
        binding.  Including the target, aliases, and utility configuration in
        that identity prevents an allocation receipt from being replayed under
        a different frontier aspiration while retaining a stable policy ID.
        """

        self.__post_init__()
        payload = json.dumps(
            {
                "schema_version": 1,
                "base_policy_definition_sha256": (
                    TARGET_CLOSURE_PORTFOLIO_UTILITY_DEFINITION_SHA256
                ),
                "target": self.target.to_record(),
                "aliases": [
                    {
                        "target_metric_id": value.target_metric_id,
                        "forecast_metric_id": value.forecast_metric_id,
                    }
                    for value in self.aliases
                ],
                "attainment_bonus_hex": self.attainment_bonus.hex(),
                "redundancy_weight_hex": self.redundancy_weight.hex(),
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(_TARGET_CLOSURE_BINDING_DOMAIN + payload).hexdigest()

    def binding(self) -> ForecastPortfolioUtilityBinding:
        """Return the exact identified utility expected by the allocator port."""

        return ForecastPortfolioUtilityBinding(
            utility=self,
            policy_id=TARGET_CLOSURE_PORTFOLIO_UTILITY_ID,
            policy_version=TARGET_CLOSURE_PORTFOLIO_UTILITY_VERSION,
            definition_sha256=self.binding_definition_sha256,
        )

    def __call__(self, request: ForecastPortfolioUtilityInput) -> float:
        if type(request) is not ForecastPortfolioUtilityInput:
            raise TypeError("request must be exact ForecastPortfolioUtilityInput")
        request.__post_init__()
        self.__post_init__()
        mapping = _alias_map(self.target, self.aliases)
        parent_by_id = {value.metric_id: value.value for value in request.parent_metric_values}
        for axis in self.target.axes:
            forecast_metric_id = mapping[axis.metric_id]
            if parent_by_id.get(forecast_metric_id) != axis.parent_value:
                raise ValueError("forecast parent values differ from the target")
        values: list[float] = []
        for member in request.members:
            scenario = _scenario(
                target=self.target,
                forecast=member,
                aliases=mapping,
                quantile=request.quantile,
            )
            value = member.probability_valid * (
                scenario.shortfall_reduction_l1
                + (
                    self.attainment_bonus
                    if scenario.attains_or_dominates_aspiration
                    else 0.0
                )
            )
            values.append(value)
        ranked = sorted(values, reverse=True)
        total = ranked[0]
        if len(ranked) > 1 and self.redundancy_weight > 0.0:
            total += self.redundancy_weight * sum(
                max(0.0, value) / rank
                for rank, value in enumerate(ranked[1:], start=2)
            )
        if not math.isfinite(total):  # pragma: no cover - guarded arithmetic.
            raise ValueError("target-closure utility became non-finite")
        return float(total)


__all__ = [
    "ActionTargetRealization",
    "ResidualTargetClosurePortfolioUtility",
    "TARGET_CLOSURE_PORTFOLIO_UTILITY_DEFINITION_SHA256",
    "TARGET_CLOSURE_PORTFOLIO_UTILITY_ID",
    "TARGET_CLOSURE_PORTFOLIO_UTILITY_VERSION",
    "TargetClosureScenario",
    "TargetMetricAlias",
    "assess_action_target_realization",
]
