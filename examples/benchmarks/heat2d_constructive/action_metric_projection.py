"""Exact cheap metric projection for constructive Heat2D finite actions.

The qualified decoder treats ``material_fraction`` as an explicit requested
control and reprojects every geometry to that value.  Its objective delta is
therefore available from the sealed parent/child configurations without a PDE
solve.  Thermal behavior remains outside this adapter's authority.
"""

from __future__ import annotations

import hashlib

from agent_evolve.agentic import (
    ActionForecastRequest,
    ExactActionMetricProjection,
    ExactActionMetricProjectionBatch,
    MetricRole,
    thaw_json,
)


HEAT2D_EXACT_METRIC_PROJECTOR_ID = "heat2d_exact_material_projection"
HEAT2D_EXACT_METRIC_PROJECTOR_VERSION = 1
HEAT2D_EXACT_METRIC_PROJECTOR_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:heat2d-exact-material-projection:v1;"
    b"public-qualified-decoder-law=requested-material-fraction-is-exact-"
    b"projected-objective;delta=sealed-child-control-minus-sealed-parent-"
    b"control;thermal-authority=false;evaluator-outcomes=false"
).hexdigest()


class Heat2DExactMaterialProjector:
    """Project exact material deltas while leaving thermal probabilistic."""

    def project(
        self,
        request: ActionForecastRequest,
    ) -> ExactActionMetricProjectionBatch:
        if type(request) is not ActionForecastRequest:
            raise TypeError("request must be an exact ActionForecastRequest")
        request.__post_init__()
        matches = tuple(
            metric.metric_id
            for metric in request.optimization_semantics.metrics
            if metric.role is MetricRole.OBJECTIVE
            and metric.name == "material_fraction"
        )
        if len(matches) != 1:
            raise ValueError(
                "Heat exact projector requires one material_fraction objective"
            )
        metric_id = matches[0]
        if metric_id not in request.required_metric_ids:
            raise ValueError("forecast request does not include material_fraction")
        contract = request.finite_variation_contract
        parent = thaw_json(contract.parent_configuration)
        if type(parent) is not dict:
            raise TypeError("Heat parent configuration must be an object")
        parent_value = parent.get("material_fraction")
        if type(parent_value) not in {int, float} or type(parent_value) is bool:
            raise TypeError("Heat parent material_fraction must be numeric")
        projections: list[ExactActionMetricProjection] = []
        for option in contract.options:
            child = thaw_json(option.child_configuration)
            if type(child) is not dict:
                raise TypeError("Heat child configuration must be an object")
            child_value = child.get("material_fraction")
            if type(child_value) not in {int, float} or type(child_value) is bool:
                raise TypeError("Heat child material_fraction must be numeric")
            projections.append(
                ExactActionMetricProjection(
                    option_id=option.option_id,
                    option_identity_sha256=option.identity_sha256,
                    child_configuration_sha256=(
                        option.child_configuration_sha256
                    ),
                    metric_id=metric_id,
                    delta=float(child_value) - float(parent_value),
                )
            )
        return ExactActionMetricProjectionBatch(
            forecast_request_sha256=request.request_sha256,
            finite_contract_identity_sha256=contract.identity_sha256,
            projections=tuple(
                sorted(
                    projections,
                    key=lambda value: (value.option_id, value.metric_id),
                )
            ),
            projector_id=HEAT2D_EXACT_METRIC_PROJECTOR_ID,
            projector_version=HEAT2D_EXACT_METRIC_PROJECTOR_VERSION,
            projector_definition_sha256=(
                HEAT2D_EXACT_METRIC_PROJECTOR_DEFINITION_SHA256
            ),
        )


__all__ = [
    "HEAT2D_EXACT_METRIC_PROJECTOR_DEFINITION_SHA256",
    "HEAT2D_EXACT_METRIC_PROJECTOR_ID",
    "HEAT2D_EXACT_METRIC_PROJECTOR_VERSION",
    "Heat2DExactMaterialProjector",
]
