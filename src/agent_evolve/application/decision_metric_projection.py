"""Apply a benchmark decision-metric projection to sealed candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.core.optimization_semantics import MetricRole
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection


@dataclass(frozen=True, slots=True)
class ProjectedDecisionMetrics:
    """Canonical finite values extracted under one immutable projection."""

    projection_definition_sha256: str
    values: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        require_sha256(
            self.projection_definition_sha256,
            "projection_definition_sha256",
        )
        if type(self.values) is not tuple or not self.values:
            raise ValueError("values must be a non-empty exact tuple")
        metric_ids: list[str] = []
        for value in self.values:
            if type(value) is not tuple or len(value) != 2:
                raise TypeError("values must contain exact metric/value pairs")
            metric_id, number = value
            if type(metric_id) is not str or not metric_id:
                raise TypeError("projected metric IDs must be non-empty strings")
            if type(number) is not float or not math.isfinite(number):
                raise TypeError("projected metric values must be finite floats")
            metric_ids.append(metric_id)
        if tuple(metric_ids) != tuple(sorted(set(metric_ids))):
            raise ValueError("projected metric values must be unique and canonical")

    @property
    def metric_map(self) -> dict[str, float]:
        self.__post_init__()
        return dict(self.values)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "projection_definition_sha256": self.projection_definition_sha256,
            "values": [
                {"metric_id": metric_id, "value_hex": value.hex()}
                for metric_id, value in self.values
            ],
        }


def project_candidate_decision_metrics(
    candidate: EvolutionCandidate,
    projection: DecisionMetricProjection,
) -> ProjectedDecisionMetrics:
    """Extract canonical objectives and sealed violation/constraint values."""

    if type(candidate) is not EvolutionCandidate:
        raise TypeError("candidate must be exact EvolutionCandidate")
    EvolutionCandidate.__post_init__(candidate)
    if type(projection) is not DecisionMetricProjection:
        raise TypeError("projection must be exact DecisionMetricProjection")
    projection.__post_init__()
    if not candidate.valid:
        raise ValueError("decision metrics require a valid evaluated candidate")

    objectives = candidate.objective_map
    detailed_values: dict[str, float] | None = None
    values: list[tuple[str, float]] = []
    for binding in projection.metrics:
        if binding.role is MetricRole.OBJECTIVE:
            if binding.value_name not in objectives:
                raise ValueError(
                    "decision projection objective is absent from the candidate: "
                    f"{binding.value_name}"
                )
            value = objectives[binding.value_name]
        else:
            detailed = candidate.detailed_evaluation
            if detailed is None or not detailed.success:
                raise ValueError(
                    "violation/constraint projection requires successful detailed "
                    "evaluation evidence"
                )
            if detailed_values is None:
                detailed_values = dict(detailed.violations)
            if binding.value_name not in detailed_values:
                raise ValueError(
                    "decision projection violation/constraint is absent from "
                    f"detailed evidence: {binding.value_name}"
                )
            value = detailed_values[binding.value_name]
        if type(value) is not float or not math.isfinite(value):
            raise TypeError(
                "candidate decision metrics must be finite canonical floats"
            )
        values.append((binding.metric_id, value))
    return ProjectedDecisionMetrics(
        projection_definition_sha256=projection.definition_sha256,
        values=tuple(values),
    )


__all__ = [
    "ProjectedDecisionMetrics",
    "project_candidate_decision_metrics",
]
