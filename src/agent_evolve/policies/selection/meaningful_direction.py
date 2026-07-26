"""Workload-neutral adjudication of meaningful metric directions.

Benchmarks publish one explicit absolute resolution per metric. The policy
turns an exact parent/child contrast into the categorical vocabulary used by
forecast calibration. This is an identified scientific policy, not a
universal floating-point epsilon.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.core.optimization_semantics import OptimizationSemantics
from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.selection.forecast_calibration import (
    MeaningfulDirectionAdjudicationReceipt,
    MeaningfulDirectionRequest,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection


_METRIC = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_DEFINITION_DOMAIN = b"agent-evolve:absolute-metric-direction:def:v1\x00"
POLICY_ID = "absolute_tolerance_metric_direction"
POLICY_VERSION = 1


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


@dataclass(frozen=True, slots=True)
class MetricDirectionResolution:
    """One metric's prospectively declared indistinguishable change band."""

    metric_id: str
    absolute_tolerance: float

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or _METRIC.fullmatch(self.metric_id) is None:
            raise ValueError("metric_id must use the closed metric grammar")
        if (
            type(self.absolute_tolerance) is not float
            or not math.isfinite(self.absolute_tolerance)
            or self.absolute_tolerance < 0.0
        ):
            raise ValueError("absolute_tolerance must be a finite non-negative float")

    def to_record(self) -> dict[str, str]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "absolute_tolerance_hex": self.absolute_tolerance.hex(),
        }


@dataclass(frozen=True, slots=True)
class AbsoluteToleranceDirectionAdjudicator:
    """Classify exact parent/child deltas under declared metric resolutions."""

    benchmark_sha256: str
    session_sha256: str
    resolutions: tuple[MetricDirectionResolution, ...]
    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    definition_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.benchmark_sha256, "benchmark_sha256")
        require_sha256(self.session_sha256, "session_sha256")
        if (
            type(self.resolutions) is not tuple
            or not self.resolutions
            or any(
                type(value) is not MetricDirectionResolution
                for value in self.resolutions
            )
        ):
            raise ValueError("resolutions must contain exact metric resolutions")
        for value in self.resolutions:
            value.__post_init__()
        if self.resolutions != tuple(
            sorted(self.resolutions, key=lambda value: value.metric_id)
        ):
            raise ValueError("resolutions must use canonical metric order")
        if len({value.metric_id for value in self.resolutions}) != len(
            self.resolutions
        ):
            raise ValueError("resolutions cannot repeat a metric")
        definition = {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "comparison": "absolute_child_minus_parent",
            "unchanged_rule": "absolute_delta_lte_declared_tolerance",
            "resolutions": [value.to_record() for value in self.resolutions],
        }
        computed = hashlib.sha256(
            _DEFINITION_DOMAIN + _canonical_json(definition)
        ).hexdigest()
        if self.definition_sha256 not in ("", computed):
            raise ValueError("definition_sha256 differs from the resolution policy")
        object.__setattr__(self, "definition_sha256", computed)

    @classmethod
    def from_optimization_semantics(
        cls,
        *,
        benchmark_sha256: str,
        session_sha256: str,
        semantics: OptimizationSemantics,
    ) -> "AbsoluteToleranceDirectionAdjudicator":
        """Bind every projected decision metric with an explicit tolerance.

        Objective-only semantics retain the historical unprefixed objective
        names. Mixed objective/violation/constraint semantics use their exact
        role-prefixed IDs through :class:`DecisionMetricProjection`.
        """

        if type(semantics) is not OptimizationSemantics:
            raise TypeError("semantics must be exact OptimizationSemantics")
        semantics.__post_init__()
        return cls.from_decision_metric_projection(
            benchmark_sha256=benchmark_sha256,
            session_sha256=session_sha256,
            projection=DecisionMetricProjection.from_optimization_semantics(semantics),
        )

    @classmethod
    def from_decision_metric_projection(
        cls,
        *,
        benchmark_sha256: str,
        session_sha256: str,
        projection: DecisionMetricProjection,
    ) -> "AbsoluteToleranceDirectionAdjudicator":
        """Construct the adjudicator from the same projection used at feedback."""

        if type(projection) is not DecisionMetricProjection:
            raise TypeError("projection must be exact DecisionMetricProjection")
        projection.__post_init__()
        missing = tuple(
            metric.metric_id
            for metric in projection.metrics
            if metric.absolute_tolerance is None
        )
        if missing:
            raise ValueError(
                "decision metrics require explicit direction tolerances: "
                + ", ".join(sorted(missing))
            )
        return cls(
            benchmark_sha256=benchmark_sha256,
            session_sha256=session_sha256,
            resolutions=tuple(
                sorted(
                    (
                        MetricDirectionResolution(
                            metric_id=metric.metric_id,
                            absolute_tolerance=float(metric.absolute_tolerance),
                        )
                        for metric in projection.metrics
                    ),
                    key=lambda value: value.metric_id,
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "definition_sha256": self.definition_sha256,
            "benchmark_sha256": self.benchmark_sha256,
            "session_sha256": self.session_sha256,
            "resolutions": [value.to_record() for value in self.resolutions],
        }

    def adjudicate(
        self,
        request: MeaningfulDirectionRequest,
    ) -> MeaningfulDirectionAdjudicationReceipt:
        self.__post_init__()
        if type(request) is not MeaningfulDirectionRequest:
            raise TypeError("request must be exact MeaningfulDirectionRequest")
        request.revalidate()
        if (
            request.benchmark_sha256 != self.benchmark_sha256
            or request.session_sha256 != self.session_sha256
        ):
            raise ValueError("direction request belongs to a foreign benchmark session")
        tolerance_by_metric = {
            value.metric_id: value.absolute_tolerance for value in self.resolutions
        }
        if request.metric_id not in tolerance_by_metric:
            raise ValueError("direction request names an undeclared metric")
        delta = request.child_metric_value - request.parent_metric_value
        tolerance = tolerance_by_metric[request.metric_id]
        if abs(delta) <= tolerance:
            direction = MetricEffectDirection.UNCHANGED
        elif delta < 0.0:
            direction = MetricEffectDirection.DECREASE
        else:
            direction = MetricEffectDirection.INCREASE
        return MeaningfulDirectionAdjudicationReceipt(
            request_sha256=request.request_sha256,
            benchmark_sha256=request.benchmark_sha256,
            session_sha256=request.session_sha256,
            wave_index=request.wave_index,
            parent_candidate_identity_sha256=(request.parent_candidate_identity_sha256),
            option_id=request.option_id,
            option_identity_sha256=request.option_identity_sha256,
            metric_id=request.metric_id,
            parent_outcome_sha256=request.parent_outcome_sha256,
            child_outcome_sha256=request.child_outcome_sha256,
            actual_direction=direction,
            adjudicator_policy_id=self.policy_id,
            adjudicator_policy_version=self.policy_version,
            adjudicator_definition_sha256=self.definition_sha256,
        )


__all__ = [
    "AbsoluteToleranceDirectionAdjudicator",
    "MetricDirectionResolution",
    "POLICY_ID",
    "POLICY_VERSION",
]
