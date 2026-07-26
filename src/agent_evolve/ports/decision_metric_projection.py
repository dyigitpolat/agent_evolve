"""Benchmark-owned projection of sealed outcomes into forecast metric space.

Optimization semantics describe more than the Pareto objective vector.  A
calibrated selector may also need to forecast violations or constraint
projections which live only in the sealed detailed evaluation.  This module
defines the immutable, provider-neutral binding between those two spaces.

Objective-only benchmarks retain their historical unprefixed metric IDs.  As
soon as a violation or constraint participates, every decision metric uses its
role-prefixed semantic ID.  That compatibility rule is deterministic and part
of the projection definition rather than a workload-specific convention.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum

from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSense,
    OptimizationSemantics,
)
from agent_evolve.domain.patch import require_sha256


_METRIC = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_DEFINITION_DOMAIN = b"agent-evolve:decision-metric-projection:def:v1\x00"


class DecisionMetricValueSource(str, Enum):
    """Closed sealed-outcome location from which a value is read."""

    CANDIDATE_OBJECTIVE = "candidate_objective"
    DETAILED_VIOLATION_OR_CONSTRAINT = "detailed_violation_or_constraint"


@dataclass(frozen=True, slots=True)
class DecisionMetricBinding:
    """One semantic metric and its exact forecast/evaluation value binding."""

    metric_id: str
    semantic_metric_id: str
    value_name: str
    role: MetricRole
    sense: MetricSense
    source: DecisionMetricValueSource
    absolute_tolerance: float | None

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or _METRIC.fullmatch(self.metric_id) is None:
            raise ValueError("metric_id must use the closed metric grammar")
        if (
            type(self.semantic_metric_id) is not str
            or _METRIC.fullmatch(self.semantic_metric_id) is None
        ):
            raise ValueError("semantic_metric_id must use the closed metric grammar")
        if type(self.value_name) is not str or _NAME.fullmatch(self.value_name) is None:
            raise ValueError("value_name must use the closed metric-name grammar")
        if type(self.role) is not MetricRole or self.role not in {
            MetricRole.OBJECTIVE,
            MetricRole.VIOLATION,
            MetricRole.CONSTRAINT,
        }:
            raise ValueError(
                "decision metrics require objective/violation/constraint role"
            )
        if type(self.sense) is not MetricSense:
            raise TypeError("sense must be an exact MetricSense")
        if type(self.source) is not DecisionMetricValueSource:
            raise TypeError("source must be an exact DecisionMetricValueSource")
        expected_source = (
            DecisionMetricValueSource.CANDIDATE_OBJECTIVE
            if self.role is MetricRole.OBJECTIVE
            else DecisionMetricValueSource.DETAILED_VIOLATION_OR_CONSTRAINT
        )
        if self.source is not expected_source:
            raise ValueError("decision metric role and sealed value source differ")
        tolerance = self.absolute_tolerance
        if tolerance is not None and (
            type(tolerance) is not float
            or not math.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError(
                "absolute_tolerance must be a finite non-negative float or None"
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "semantic_metric_id": self.semantic_metric_id,
            "value_name": self.value_name,
            "role": self.role.value,
            "sense": self.sense.value,
            "source": self.source.value,
            "absolute_tolerance_hex": (
                None
                if self.absolute_tolerance is None
                else self.absolute_tolerance.hex()
            ),
        }


@dataclass(frozen=True, slots=True)
class DecisionMetricProjection:
    """Immutable projection contract derived from benchmark semantics.

    The projection owns no evaluator and contains no model-authored data.  The
    application layer applies it to an ``EvolutionCandidate`` after evaluation.
    """

    optimization_semantics_definition_sha256: str
    metrics: tuple[DecisionMetricBinding, ...]
    objective_only_legacy_metric_ids: bool
    definition_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(
            self.optimization_semantics_definition_sha256,
            "optimization_semantics_definition_sha256",
        )
        if (
            type(self.metrics) is not tuple
            or not self.metrics
            or any(type(value) is not DecisionMetricBinding for value in self.metrics)
        ):
            raise ValueError("metrics must contain exact decision metric bindings")
        for value in self.metrics:
            value.__post_init__()
        metric_ids = tuple(value.metric_id for value in self.metrics)
        semantic_ids = tuple(value.semantic_metric_id for value in self.metrics)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("decision metric IDs must be unique and canonical")
        if len(set(semantic_ids)) != len(semantic_ids):
            raise ValueError("semantic decision metric IDs must be unique")
        if type(self.objective_only_legacy_metric_ids) is not bool:
            raise TypeError("objective_only_legacy_metric_ids must be exact bool")
        all_objectives = all(
            value.role is MetricRole.OBJECTIVE for value in self.metrics
        )
        if self.objective_only_legacy_metric_ids != all_objectives:
            raise ValueError(
                "legacy metric IDs apply exactly to objective-only projections"
            )
        if self.objective_only_legacy_metric_ids and any(
            value.metric_id != value.value_name for value in self.metrics
        ):
            raise ValueError("objective-only compatibility IDs must equal value names")
        if not self.objective_only_legacy_metric_ids and any(
            value.metric_id != value.semantic_metric_id for value in self.metrics
        ):
            raise ValueError("mixed-role projections must use semantic metric IDs")
        definition = {
            "schema_version": 1,
            "optimization_semantics_definition_sha256": (
                self.optimization_semantics_definition_sha256
            ),
            "objective_only_legacy_metric_ids": (self.objective_only_legacy_metric_ids),
            "metrics": [value.to_record() for value in self.metrics],
        }
        encoded = json.dumps(
            definition,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii", errors="strict")
        computed = hashlib.sha256(_DEFINITION_DOMAIN + encoded).hexdigest()
        if self.definition_sha256 not in ("", computed):
            raise ValueError("definition_sha256 differs from decision projection")
        object.__setattr__(self, "definition_sha256", computed)

    @classmethod
    def from_optimization_semantics(
        cls,
        semantics: OptimizationSemantics,
    ) -> "DecisionMetricProjection":
        """Project all objective/violation/constraint metrics, never diagnostics."""

        if type(semantics) is not OptimizationSemantics:
            raise TypeError("semantics must be exact OptimizationSemantics")
        semantics.__post_init__()
        decision_metrics = tuple(
            metric
            for metric in semantics.metrics
            if metric.role
            in {
                MetricRole.OBJECTIVE,
                MetricRole.VIOLATION,
                MetricRole.CONSTRAINT,
            }
        )
        if not decision_metrics:
            raise ValueError("optimization semantics publish no decision metrics")
        objective_only = all(
            metric.role is MetricRole.OBJECTIVE for metric in decision_metrics
        )
        bindings = tuple(
            sorted(
                (
                    DecisionMetricBinding(
                        metric_id=(metric.name if objective_only else metric.metric_id),
                        semantic_metric_id=metric.metric_id,
                        value_name=metric.name,
                        role=metric.role,
                        sense=metric.sense,
                        source=(
                            DecisionMetricValueSource.CANDIDATE_OBJECTIVE
                            if metric.role is MetricRole.OBJECTIVE
                            else DecisionMetricValueSource.DETAILED_VIOLATION_OR_CONSTRAINT
                        ),
                        absolute_tolerance=(
                            None
                            if metric.tolerance is None
                            else float(metric.tolerance)
                        ),
                    )
                    for metric in decision_metrics
                ),
                key=lambda value: value.metric_id,
            )
        )
        return cls(
            optimization_semantics_definition_sha256=semantics.definition_sha256,
            metrics=bindings,
            objective_only_legacy_metric_ids=objective_only,
        )

    @property
    def metric_ids(self) -> tuple[str, ...]:
        self.__post_init__()
        return tuple(value.metric_id for value in self.metrics)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "optimization_semantics_definition_sha256": (
                self.optimization_semantics_definition_sha256
            ),
            "objective_only_legacy_metric_ids": (self.objective_only_legacy_metric_ids),
            "metrics": [value.to_record() for value in self.metrics],
            "definition_sha256": self.definition_sha256,
        }


__all__ = [
    "DecisionMetricBinding",
    "DecisionMetricProjection",
    "DecisionMetricValueSource",
]
