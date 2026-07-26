"""Trusted overlay of exact metric projections onto probabilistic forecasts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.action_forecast import (
    ActionForecastRequest,
    ResolvedActionForecast,
    ResolvedActionForecastBatch,
    ResolvedActionMetricForecast,
    validate_resolved_action_forecasts,
)
from agent_evolve.ports.action_metric_projection import (
    ExactActionMetricProjectionBatch,
)


EXACT_METRIC_OVERLAY_POLICY_ID = "exact_metric_projection_overlay"
EXACT_METRIC_OVERLAY_POLICY_VERSION = 1
EXACT_METRIC_OVERLAY_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:exact-metric-projection-overlay:v1;"
    b"validate-source-forecast-against-request;validate-projection-request-"
    b"contract-option-and-metric-identities;replace-projected-quantiles-with-"
    b"one-exact-delta;confidence=one;preserve-evidence-citations;preserve-"
    b"unprojected-cells-and-validity;allocation=false;evaluation=false"
).hexdigest()
_RESULT_DOMAIN = b"agent-evolve:exact-metric-projection-overlay-result:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


@dataclass(frozen=True, slots=True, eq=False)
class ActionMetricProjectionOverlayResult:
    """Authenticated join of model forecasts and exact metric projections."""

    source_forecast_receipt_sha256: str
    projection_receipt_sha256: str
    forecasts: ResolvedActionForecastBatch

    def __post_init__(self) -> None:
        require_sha256(
            self.source_forecast_receipt_sha256,
            "source_forecast_receipt_sha256",
        )
        require_sha256(
            self.projection_receipt_sha256,
            "projection_receipt_sha256",
        )
        if type(self.forecasts) is not ResolvedActionForecastBatch:
            raise TypeError("forecasts must be an exact resolved batch")
        self.forecasts.__post_init__()
        if self.forecasts.policy_id != EXACT_METRIC_OVERLAY_POLICY_ID:
            raise ValueError("forecasts do not use the exact overlay policy")
        if (
            self.forecasts.policy_version
            != EXACT_METRIC_OVERLAY_POLICY_VERSION
            or self.forecasts.policy_definition_sha256
            != EXACT_METRIC_OVERLAY_POLICY_DEFINITION_SHA256
        ):
            raise ValueError("forecast overlay policy identity differs")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "source_forecast_receipt_sha256": (
                self.source_forecast_receipt_sha256
            ),
            "projection_receipt_sha256": self.projection_receipt_sha256,
            "overlaid_forecast_receipt_sha256": self.forecasts.receipt_sha256,
            "overlay_policy": {
                "policy_id": EXACT_METRIC_OVERLAY_POLICY_ID,
                "policy_version": EXACT_METRIC_OVERLAY_POLICY_VERSION,
                "definition_sha256": (
                    EXACT_METRIC_OVERLAY_POLICY_DEFINITION_SHA256
                ),
            },
        }

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(
            _RESULT_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ActionMetricProjectionOverlayResult
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


def apply_exact_action_metric_projections(
    *,
    request: ActionForecastRequest,
    forecasts: ResolvedActionForecastBatch,
    projections: ExactActionMetricProjectionBatch,
) -> ActionMetricProjectionOverlayResult:
    """Replace only projection-authorized metric cells with exact deltas."""

    if type(request) is not ActionForecastRequest:
        raise TypeError("request must be an exact ActionForecastRequest")
    request.__post_init__()
    if type(forecasts) is not ResolvedActionForecastBatch:
        raise TypeError("forecasts must be an exact resolved batch")
    validate_resolved_action_forecasts(request, forecasts)
    if type(projections) is not ExactActionMetricProjectionBatch:
        raise TypeError("projections must be an exact projection batch")
    projections.__post_init__()
    if projections.forecast_request_sha256 != request.request_sha256:
        raise ValueError("metric projections belong to a foreign forecast request")
    contract = request.finite_variation_contract
    if projections.finite_contract_identity_sha256 != contract.identity_sha256:
        raise ValueError("metric projections belong to a foreign finite contract")
    options_by_id = {value.option_id: value for value in contract.options}
    required_metric_ids = {
        value.metric_id for value in request.parent_metric_values
    }
    exact_by_key: dict[tuple[str, str], float] = {}
    for projection in projections.projections:
        option = options_by_id.get(projection.option_id)
        if option is None:
            raise ValueError("metric projection names a foreign finite option")
        if (
            projection.option_identity_sha256 != option.identity_sha256
            or projection.child_configuration_sha256
            != option.child_configuration_sha256
        ):
            raise ValueError("metric projection differs from its sealed option")
        if projection.metric_id not in required_metric_ids:
            raise ValueError("metric projection names a non-requested metric")
        exact_by_key[(projection.option_id, projection.metric_id)] = (
            projection.delta
        )

    overlaid: list[ResolvedActionForecast] = []
    for forecast in forecasts.forecasts:
        metrics: list[ResolvedActionMetricForecast] = []
        for metric in forecast.metric_forecasts:
            delta = exact_by_key.get((forecast.option_id, metric.metric_id))
            if delta is None:
                metrics.append(metric)
                continue
            metrics.append(
                ResolvedActionMetricForecast(
                    metric_id=metric.metric_id,
                    p10_delta=delta,
                    p50_delta=delta,
                    p90_delta=delta,
                    confidence=1.0,
                    citations=metric.citations,
                )
            )
        overlaid.append(
            ResolvedActionForecast(
                option_id=forecast.option_id,
                option_identity_sha256=forecast.option_identity_sha256,
                child_configuration_sha256=forecast.child_configuration_sha256,
                family=forecast.family,
                probability_valid=forecast.probability_valid,
                metric_forecasts=tuple(metrics),
            )
        )
    resolved = ResolvedActionForecastBatch(
        request_sha256=forecasts.request_sha256,
        context_sha256=forecasts.context_sha256,
        optimization_semantics_definition_sha256=(
            forecasts.optimization_semantics_definition_sha256
        ),
        action_semantics_definition_sha256=(
            forecasts.action_semantics_definition_sha256
        ),
        finite_contract_identity_sha256=forecasts.finite_contract_identity_sha256,
        card_snapshot_sha256=forecasts.card_snapshot_sha256,
        forecasts=tuple(overlaid),
        policy_id=EXACT_METRIC_OVERLAY_POLICY_ID,
        policy_version=EXACT_METRIC_OVERLAY_POLICY_VERSION,
        policy_definition_sha256=EXACT_METRIC_OVERLAY_POLICY_DEFINITION_SHA256,
    )
    validate_resolved_action_forecasts(request, resolved)
    return ActionMetricProjectionOverlayResult(
        source_forecast_receipt_sha256=forecasts.receipt_sha256,
        projection_receipt_sha256=projections.receipt_sha256,
        forecasts=resolved,
    )


__all__ = [
    "ActionMetricProjectionOverlayResult",
    "EXACT_METRIC_OVERLAY_POLICY_DEFINITION_SHA256",
    "EXACT_METRIC_OVERLAY_POLICY_ID",
    "EXACT_METRIC_OVERLAY_POLICY_VERSION",
    "apply_exact_action_metric_projections",
]
