"""Archive-aligned utility for probabilistic finite-action consequences.

Residual frontier targets expose prior-only normalized anchor points.  This
module converts forecasted child outcomes into exact two- or three-dimensional
fixed-reference hypervolume and integrates independent action-validity
probabilities exactly for the small evolutionary batches used by AgentEvolve.
It contains no workload, model, provider, or action-family identifiers.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.policies.reward.affine_hypervolume_3d import hypervolume_3d
from agent_evolve.policies.reward.frozen_archive import hypervolume_2d
from agent_evolve.ports.action_allocation import (
    ForecastPortfolioUtilityBinding,
    ForecastPortfolioUtilityInput,
    ForecastQuantile,
)
from agent_evolve.ports.action_forecast import (
    ResolvedActionForecast,
    ResolvedActionMetricForecast,
)
from agent_evolve.ports.frontier_target import (
    CampaignPortfolioFrontierTarget,
    ObjectiveSpaceTarget,
)
from agent_evolve.application.action_target_realization import TargetMetricAlias


RESIDUAL_CELL_EXPECTED_HV_UTILITY_ID = "residual_cell_expected_hypervolume"
RESIDUAL_CELL_EXPECTED_HV_UTILITY_VERSION = 1
RESIDUAL_CELL_EXPECTED_HV_UTILITY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:residual-cell-expected-hypervolume:v1;"
    b"frame=prior-only-normalized-lower-is-better-objectives;"
    b"reference=unit-vector;archive=authenticated-residual-cell-anchors;"
    b"candidate=target-parent-plus-forecast-delta;set-value=exact-union-hv-gain;"
    b"validity=exact-independent-bernoulli-expectation;dimensions=two-or-three;"
    b"workload-model-provider-and-family-identifiers=false;outcomes=false"
).hexdigest()
_BINDING_DOMAIN = b"agent-evolve:residual-cell-expected-hv-binding:v1\x00"
RELIABILITY_ADJUSTED_RESIDUAL_CELL_EXPECTED_HV_UTILITY_ID = (
    "reliability_adjusted_residual_cell_expected_hypervolume"
)
RELIABILITY_ADJUSTED_RESIDUAL_CELL_EXPECTED_HV_UTILITY_VERSION = 1
RELIABILITY_ADJUSTED_RESIDUAL_CELL_EXPECTED_HV_UTILITY_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:reliability-adjusted-residual-cell-expected-"
        b"hypervolume:v1;source=base-residual-cell-expected-hv-v1;"
        b"joint-forecast-reliability=probability-valid-times-minimum-target-"
        b"metric-epistemic-confidence;unreliable-state=parent-no-gain;"
        b"exact-projected-metric-confidence-one;workload-model-provider-and-"
        b"family-identifiers=false;outcomes=false"
    ).hexdigest()
)
_RELIABILITY_BINDING_DOMAIN = (
    b"agent-evolve:reliability-adjusted-residual-cell-expected-hv-binding:v1\x00"
)


def _finite(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite exact float")
    return value


@dataclass(frozen=True, slots=True)
class NormalizedResidualFrontierCell:
    """Typed prior-only local archive geometry from a frontier target."""

    campaign_target_sha256: str
    cell_sha256: str
    geometry_sha256: str
    metric_ids: tuple[str, ...]
    anchor_points: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        require_sha256(self.campaign_target_sha256, "campaign_target_sha256")
        require_sha256(self.cell_sha256, "cell_sha256")
        require_sha256(self.geometry_sha256, "geometry_sha256")
        if (
            type(self.metric_ids) is not tuple
            or len(self.metric_ids) not in {2, 3}
            or self.metric_ids != tuple(sorted(set(self.metric_ids)))
        ):
            raise ValueError("metric_ids must be two or three canonical metrics")
        if type(self.anchor_points) is not tuple or not self.anchor_points:
            raise ValueError("anchor_points must be a non-empty exact tuple")
        for point in self.anchor_points:
            if type(point) is not tuple or len(point) != len(self.metric_ids):
                raise ValueError("anchor point dimensionality differs from metrics")
            for value in point:
                _finite(value, name="normalized anchor coordinate")
        if self.anchor_points != tuple(sorted(set(self.anchor_points))):
            raise ValueError("anchor_points must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "campaign_target_sha256": self.campaign_target_sha256,
            "cell_sha256": self.cell_sha256,
            "geometry_sha256": self.geometry_sha256,
            "metric_ids": list(self.metric_ids),
            "anchor_points_hex": [
                [value.hex() for value in point] for point in self.anchor_points
            ],
        }


def residual_frontier_cell_from_target(
    *,
    campaign_target: CampaignPortfolioFrontierTarget,
    objective_target: ObjectiveSpaceTarget,
) -> NormalizedResidualFrontierCell | None:
    """Read the optional generic residual-cell geometry fail-closed."""

    if type(campaign_target) is not CampaignPortfolioFrontierTarget:
        raise TypeError("campaign_target must be exact")
    campaign_target.__post_init__()
    if type(objective_target) is not ObjectiveSpaceTarget:
        raise TypeError("objective_target must be exact")
    objective_target.__post_init__()
    if objective_target.campaign_target_sha256 != campaign_target.target_sha256:
        raise ValueError("objective target belongs to a foreign campaign target")
    payload = thaw_json(campaign_target.payload)
    if type(payload) is not dict:  # pragma: no cover - frozen root is closed.
        raise AssertionError("campaign target payload is not an object")
    raw = payload.get("residual_frontier_cell")
    if raw is None:
        return None
    if type(raw) is not dict:
        raise ValueError("residual_frontier_cell must be an object")
    required = {
        "cell_sha256",
        "geometry_sha256",
        "normalized_anchor_points_decimal",
    }
    if not required.issubset(raw):
        raise ValueError("residual frontier cell omits archive geometry")
    rows = raw["normalized_anchor_points_decimal"]
    if type(rows) is not list or not rows:
        raise ValueError("residual frontier cell anchors must be non-empty")
    objective_record = payload.get("objective_space_target")
    if type(objective_record) is not dict or type(
        objective_record.get("axes")
    ) is not list:
        raise ValueError("residual frontier cell omits its objective axis order")
    raw_axis_ids = tuple(
        row.get("metric_id") if type(row) is dict else None
        for row in objective_record["axes"]
    )
    if (
        any(type(value) is not str for value in raw_axis_ids)
        or len(raw_axis_ids) != len(objective_target.axes)
        or set(raw_axis_ids) != set(objective_target.metric_ids)
    ):
        raise ValueError("residual frontier axis order differs from target metrics")
    anchors: list[tuple[float, ...]] = []
    for row in rows:
        if type(row) is not list or len(row) != len(objective_target.axes):
            raise ValueError("residual frontier anchor dimensionality differs")
        values: list[float] = []
        for value in row:
            if type(value) is not str:
                raise TypeError("residual frontier coordinates must be decimal text")
            try:
                decoded = float(value)
            except ValueError as error:
                raise ValueError("residual frontier coordinate is malformed") from error
            if not math.isfinite(decoded):
                raise ValueError("residual frontier coordinate must be finite")
            values.append(decoded)
        by_metric = dict(zip(raw_axis_ids, values, strict=True))
        anchors.append(
            tuple(by_metric[metric_id] for metric_id in objective_target.metric_ids)
        )
    return NormalizedResidualFrontierCell(
        campaign_target_sha256=campaign_target.target_sha256,
        cell_sha256=raw["cell_sha256"],
        geometry_sha256=raw["geometry_sha256"],
        metric_ids=objective_target.metric_ids,
        anchor_points=tuple(sorted(set(anchors))),
    )


def _metric_delta(
    forecast: ResolvedActionMetricForecast,
    quantile: ForecastQuantile,
) -> float:
    if quantile is ForecastQuantile.P10:
        return forecast.p10_delta
    if quantile is ForecastQuantile.P50:
        return forecast.p50_delta
    return forecast.p90_delta


def _hypervolume(points: tuple[tuple[float, ...], ...]) -> float:
    dimensions = len(points[0])
    if dimensions == 2:
        return hypervolume_2d(points, (1.0, 1.0))  # type: ignore[arg-type]
    if dimensions == 3:
        return hypervolume_3d(points, (1.0, 1.0, 1.0))  # type: ignore[arg-type]
    raise ValueError("residual-cell hypervolume supports two or three objectives")


@dataclass(frozen=True, slots=True)
class ResidualCellExpectedHypervolumeUtility:
    """Expected local archive HV gain under forecast quantile and validity."""

    target: ObjectiveSpaceTarget
    cell: NormalizedResidualFrontierCell
    aliases: tuple[TargetMetricAlias, ...] = ()
    max_exact_validity_members: int = 8

    def __post_init__(self) -> None:
        if type(self.target) is not ObjectiveSpaceTarget:
            raise TypeError("target must be exact")
        self.target.__post_init__()
        if type(self.cell) is not NormalizedResidualFrontierCell:
            raise TypeError("cell must be exact")
        self.cell.__post_init__()
        if self.cell.campaign_target_sha256 != self.target.campaign_target_sha256:
            raise ValueError("residual cell belongs to a foreign target")
        if self.cell.metric_ids != self.target.metric_ids:
            raise ValueError("residual cell metric frame differs from target")
        if type(self.aliases) is not tuple or any(
            type(value) is not TargetMetricAlias for value in self.aliases
        ):
            raise TypeError("aliases must contain exact target metric aliases")
        for value in self.aliases:
            value.__post_init__()
        target_ids = tuple(value.target_metric_id for value in self.aliases)
        forecast_ids = tuple(value.forecast_metric_id for value in self.aliases)
        if self.aliases and (
            target_ids != self.target.metric_ids
            or len(set(forecast_ids)) != len(forecast_ids)
        ):
            raise ValueError("aliases must be one-to-one and cover the target")
        if (
            type(self.max_exact_validity_members) is not int
            or self.max_exact_validity_members <= 0
            or self.max_exact_validity_members > 16
        ):
            raise ValueError("max_exact_validity_members must lie in [1,16]")

    @property
    def binding_definition_sha256(self) -> str:
        self.__post_init__()
        payload = json.dumps(
            {
                "schema_version": 1,
                "base_policy_definition_sha256": (
                    RESIDUAL_CELL_EXPECTED_HV_UTILITY_DEFINITION_SHA256
                ),
                "target": self.target.to_record(),
                "cell": self.cell.to_record(),
                "aliases": [
                    {
                        "target_metric_id": value.target_metric_id,
                        "forecast_metric_id": value.forecast_metric_id,
                    }
                    for value in self.aliases
                ],
                "max_exact_validity_members": self.max_exact_validity_members,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(_BINDING_DOMAIN + payload).hexdigest()

    def binding(self) -> ForecastPortfolioUtilityBinding:
        return ForecastPortfolioUtilityBinding(
            utility=self,
            policy_id=RESIDUAL_CELL_EXPECTED_HV_UTILITY_ID,
            policy_version=RESIDUAL_CELL_EXPECTED_HV_UTILITY_VERSION,
            definition_sha256=self.binding_definition_sha256,
        )

    def _point(
        self,
        member: ResolvedActionForecast,
        quantile: ForecastQuantile,
    ) -> tuple[float, ...]:
        forecast_by_id = {
            value.metric_id: value for value in member.metric_forecasts
        }
        alias = (
            {
                value.target_metric_id: value.forecast_metric_id
                for value in self.aliases
            }
            if self.aliases
            else {metric_id: metric_id for metric_id in self.target.metric_ids}
        )
        point: list[float] = []
        for axis in self.target.axes:
            metric = forecast_by_id.get(alias[axis.metric_id])
            if metric is None:
                raise ValueError("action forecast omits a target metric")
            point.append(
                axis.normalize(
                    axis.parent_value + _metric_delta(metric, quantile)
                )
            )
        return tuple(point)

    def __call__(self, request: ForecastPortfolioUtilityInput) -> float:
        if type(request) is not ForecastPortfolioUtilityInput:
            raise TypeError("request must be an exact utility input")
        request.__post_init__()
        self.__post_init__()
        if len(request.members) > self.max_exact_validity_members:
            raise ValueError("portfolio exceeds exact validity integration limit")
        alias = (
            {
                value.target_metric_id: value.forecast_metric_id
                for value in self.aliases
            }
            if self.aliases
            else {metric_id: metric_id for metric_id in self.target.metric_ids}
        )
        parent_by_id = {
            value.metric_id: value.value for value in request.parent_metric_values
        }
        for axis in self.target.axes:
            if parent_by_id.get(alias[axis.metric_id]) != axis.parent_value:
                raise ValueError("forecast parent values differ from target")
        points = tuple(
            self._point(member, request.quantile) for member in request.members
        )
        base = _hypervolume(self.cell.anchor_points)
        expected = 0.0
        count = len(request.members)
        for mask in range(1 << count):
            probability = 1.0
            admitted: list[tuple[float, ...]] = []
            for index, member in enumerate(request.members):
                valid = member.probability_valid
                if mask & (1 << index):
                    probability *= valid
                    admitted.append(points[index])
                else:
                    probability *= 1.0 - valid
            if probability == 0.0:
                continue
            gain = _hypervolume(
                tuple(sorted(set((*self.cell.anchor_points, *admitted))))
            ) - base
            expected += probability * max(0.0, gain)
        if not math.isfinite(expected) or expected < 0.0:
            raise RuntimeError("expected residual-cell hypervolume became invalid")
        return float(expected)


@dataclass(frozen=True, slots=True)
class ReliabilityAdjustedResidualCellExpectedHypervolumeUtility(
    ResidualCellExpectedHypervolumeUtility
):
    """Conservative exploit value using the weakest target-metric confidence.

    Model-derived confidence is not treated as calibrated outcome probability.
    It gates how much authority an optimistic consequence forecast receives in
    the exploitation head.  A separate exploration head can still value the
    same low-confidence action for information or reachability.
    """

    @property
    def binding_definition_sha256(self) -> str:
        self.__post_init__()
        payload = json.dumps(
            {
                "schema_version": 1,
                "base_policy_definition_sha256": (
                    RELIABILITY_ADJUSTED_RESIDUAL_CELL_EXPECTED_HV_UTILITY_DEFINITION_SHA256
                ),
                "target": self.target.to_record(),
                "cell": self.cell.to_record(),
                "aliases": [
                    {
                        "target_metric_id": value.target_metric_id,
                        "forecast_metric_id": value.forecast_metric_id,
                    }
                    for value in self.aliases
                ],
                "max_exact_validity_members": self.max_exact_validity_members,
                "reliability": (
                    "probability_valid_times_minimum_target_metric_confidence"
                ),
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(
            _RELIABILITY_BINDING_DOMAIN + payload
        ).hexdigest()

    def binding(self) -> ForecastPortfolioUtilityBinding:
        return ForecastPortfolioUtilityBinding(
            utility=self,
            policy_id=(
                RELIABILITY_ADJUSTED_RESIDUAL_CELL_EXPECTED_HV_UTILITY_ID
            ),
            policy_version=(
                RELIABILITY_ADJUSTED_RESIDUAL_CELL_EXPECTED_HV_UTILITY_VERSION
            ),
            definition_sha256=self.binding_definition_sha256,
        )

    def __call__(self, request: ForecastPortfolioUtilityInput) -> float:
        if type(request) is not ForecastPortfolioUtilityInput:
            raise TypeError("request must be an exact utility input")
        request.__post_init__()
        self.__post_init__()
        if len(request.members) > self.max_exact_validity_members:
            raise ValueError("portfolio exceeds exact validity integration limit")
        alias = (
            {
                value.target_metric_id: value.forecast_metric_id
                for value in self.aliases
            }
            if self.aliases
            else {metric_id: metric_id for metric_id in self.target.metric_ids}
        )
        parent_by_id = {
            value.metric_id: value.value for value in request.parent_metric_values
        }
        for axis in self.target.axes:
            if parent_by_id.get(alias[axis.metric_id]) != axis.parent_value:
                raise ValueError("forecast parent values differ from target")
        points = tuple(
            self._point(member, request.quantile) for member in request.members
        )
        target_forecast_ids = tuple(
            alias[metric_id] for metric_id in self.target.metric_ids
        )
        reliabilities: list[float] = []
        for member in request.members:
            metric_by_id = {
                value.metric_id: value for value in member.metric_forecasts
            }
            if not set(target_forecast_ids).issubset(metric_by_id):
                raise ValueError("action forecast omits a target metric")
            weakest_confidence = min(
                metric_by_id[metric_id].confidence
                for metric_id in target_forecast_ids
            )
            reliabilities.append(
                member.probability_valid * weakest_confidence
            )
        base = _hypervolume(self.cell.anchor_points)
        expected = 0.0
        count = len(request.members)
        for mask in range(1 << count):
            probability = 1.0
            admitted: list[tuple[float, ...]] = []
            for index, reliability in enumerate(reliabilities):
                if mask & (1 << index):
                    probability *= reliability
                    admitted.append(points[index])
                else:
                    probability *= 1.0 - reliability
            if probability == 0.0:
                continue
            gain = _hypervolume(
                tuple(sorted(set((*self.cell.anchor_points, *admitted))))
            ) - base
            expected += probability * max(0.0, gain)
        if not math.isfinite(expected) or expected < 0.0:
            raise RuntimeError(
                "reliability-adjusted residual-cell hypervolume became invalid"
            )
        return float(expected)


__all__ = [
    "NormalizedResidualFrontierCell",
    "RESIDUAL_CELL_EXPECTED_HV_UTILITY_DEFINITION_SHA256",
    "RESIDUAL_CELL_EXPECTED_HV_UTILITY_ID",
    "RESIDUAL_CELL_EXPECTED_HV_UTILITY_VERSION",
    "RELIABILITY_ADJUSTED_RESIDUAL_CELL_EXPECTED_HV_UTILITY_DEFINITION_SHA256",
    "RELIABILITY_ADJUSTED_RESIDUAL_CELL_EXPECTED_HV_UTILITY_ID",
    "RELIABILITY_ADJUSTED_RESIDUAL_CELL_EXPECTED_HV_UTILITY_VERSION",
    "ReliabilityAdjustedResidualCellExpectedHypervolumeUtility",
    "ResidualCellExpectedHypervolumeUtility",
    "residual_frontier_cell_from_target",
]
