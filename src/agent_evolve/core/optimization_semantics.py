"""Immutable, prompt-safe semantics for optimization metrics and ordering.

The application core must not guess what a benchmark metric means.  This
module gives benchmark adapters a small inverted API for publishing that
meaning once, binding it to the objective and outcome-relation identities,
and rendering the same canonical record into every agentic prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from agent_evolve.core.problem import ObjectiveSpec, validate_objective_specs
from agent_evolve.domain.patch import require_sha256


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_SEMANTICS_HASH_DOMAIN = b"agent-evolve:optimization-semantics:v1\x00"


class MetricRole(str, Enum):
    """Closed role vocabulary for values exposed to an optimizer."""

    OBJECTIVE = "objective"
    VIOLATION = "violation"
    CONSTRAINT = "constraint"
    DIAGNOSTIC = "diagnostic"


class MetricSense(str, Enum):
    """How better values of a metric are interpreted."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    TARGET = "target"
    SATISFY_BOUNDS = "satisfy_bounds"
    INFORMATIONAL = "informational"


class OutcomeOrderingKind(str, Enum):
    """High-level structure of the benchmark's outcome comparison."""

    LEXICOGRAPHIC = "lexicographic"
    PARETO = "pareto"
    SCALAR = "scalar"
    CUSTOM = "custom"


def _nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty exact string")
    return value


def _token(value: object, name: str) -> str:
    text = _nonempty(value, name)
    if _TOKEN.fullmatch(text) is None:
        raise ValueError(f"{name} must use the closed token grammar")
    return text


def _finite_optional(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number or None")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class MetricSemantics:
    """Exact human-facing meaning of one objective/evidence metric."""

    metric_id: str
    name: str
    role: MetricRole
    sense: MetricSense
    definition: str
    aggregation: str
    witness_interpretation: str
    reference_target: float | None = None
    bounds: tuple[float | None, float | None] | None = None
    tolerance: float | None = None

    def __post_init__(self) -> None:
        _token(self.name, "metric name")
        if type(self.role) is not MetricRole:
            raise TypeError("role must be an exact MetricRole")
        if type(self.sense) is not MetricSense:
            raise TypeError("sense must be an exact MetricSense")
        expected_prefix = f"{self.role.value}:"
        if (
            type(self.metric_id) is not str
            or not self.metric_id.startswith(expected_prefix)
            or _TOKEN.fullmatch(self.metric_id[len(expected_prefix) :]) is None
        ):
            raise ValueError(
                "metric_id must be '<role>:<closed-token-name>' and match role"
            )
        _nonempty(self.definition, "metric definition")
        _nonempty(self.aggregation, "metric aggregation")
        _nonempty(self.witness_interpretation, "witness_interpretation")
        target = _finite_optional(self.reference_target, "reference_target")
        tolerance = _finite_optional(self.tolerance, "tolerance")
        if tolerance is not None and tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if self.sense is MetricSense.TARGET and target is None:
            raise ValueError("target-sense metrics require reference_target")
        if self.bounds is not None:
            if type(self.bounds) is not tuple or len(self.bounds) != 2:
                raise TypeError("bounds must be an exact (lower, upper) tuple")
            lower = _finite_optional(self.bounds[0], "bounds lower")
            upper = _finite_optional(self.bounds[1], "bounds upper")
            if lower is None and upper is None:
                raise ValueError("bounds must publish at least one endpoint")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("bounds lower endpoint exceeds upper endpoint")
        elif self.sense is MetricSense.SATISFY_BOUNDS:
            raise ValueError("satisfy-bounds metrics require bounds")

    def to_record(self) -> dict[str, object]:
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "role": self.role.value,
            "sense": self.sense.value,
            "definition": self.definition,
            "aggregation": self.aggregation,
            "reference_target": self.reference_target,
            "bounds": None if self.bounds is None else list(self.bounds),
            "tolerance": self.tolerance,
            "witness_interpretation": self.witness_interpretation,
        }


@dataclass(frozen=True, slots=True)
class OutcomeOrderingSemantics:
    """Human-readable ordering bound to the executable relation policy."""

    kind: OutcomeOrderingKind
    metric_priority: tuple[str, ...]
    description: str
    equivalence: str
    policy_id: str
    policy_version: int
    definition_sha256: str

    def __post_init__(self) -> None:
        if type(self.kind) is not OutcomeOrderingKind:
            raise TypeError("kind must be an exact OutcomeOrderingKind")
        if type(self.metric_priority) is not tuple or not self.metric_priority:
            raise ValueError("metric_priority must be a non-empty exact tuple")
        if any(type(value) is not str or not value for value in self.metric_priority):
            raise TypeError("metric_priority entries must be non-empty strings")
        if len(set(self.metric_priority)) != len(self.metric_priority):
            raise ValueError("metric_priority must not contain duplicates")
        _nonempty(self.description, "outcome ordering description")
        _nonempty(self.equivalence, "outcome equivalence description")
        _token(self.policy_id, "outcome policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("outcome policy_version must be a positive exact integer")
        require_sha256(self.definition_sha256, "outcome definition_sha256")

    @property
    def relation_identity(self) -> tuple[str, int, str]:
        return self.policy_id, self.policy_version, self.definition_sha256

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "metric_priority": list(self.metric_priority),
            "description": self.description,
            "equivalence": self.equivalence,
            "relation_policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.definition_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class OptimizationSemantics:
    """Versioned semantic contract supplied by a benchmark adapter."""

    semantics_id: str
    semantics_version: int
    metrics: tuple[MetricSemantics, ...]
    outcome_ordering: OutcomeOrderingSemantics
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _token(self.semantics_id, "semantics_id")
        if type(self.semantics_version) is not int or self.semantics_version <= 0:
            raise ValueError("semantics_version must be a positive exact integer")
        if type(self.metrics) is not tuple or not self.metrics:
            raise ValueError("metrics must be a non-empty exact tuple")
        for metric in self.metrics:
            if type(metric) is not MetricSemantics:
                raise TypeError("metrics must contain exact MetricSemantics values")
            MetricSemantics.__post_init__(metric)
        metric_ids = tuple(metric.metric_id for metric in self.metrics)
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("metric IDs must be unique")
        if type(self.outcome_ordering) is not OutcomeOrderingSemantics:
            raise TypeError(
                "outcome_ordering must be an exact OutcomeOrderingSemantics"
            )
        OutcomeOrderingSemantics.__post_init__(self.outcome_ordering)
        missing = set(self.outcome_ordering.metric_priority) - set(metric_ids)
        if missing:
            raise ValueError(
                "outcome metric_priority references unknown metrics: "
                + ", ".join(sorted(missing))
            )
        encoded = json.dumps(
            self._definition_record(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        object.__setattr__(
            self,
            "definition_sha256",
            hashlib.sha256(_SEMANTICS_HASH_DOMAIN + encoded).hexdigest(),
        )

    def _definition_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "semantics_id": self.semantics_id,
            "semantics_version": self.semantics_version,
            "metrics": [metric.to_record() for metric in self.metrics],
            "outcome_ordering": self.outcome_ordering.to_record(),
        }

    @property
    def identity(self) -> tuple[str, int, str]:
        return self.semantics_id, self.semantics_version, self.definition_sha256

    def to_record(self) -> dict[str, object]:
        return {
            **self._definition_record(),
            "definition_sha256": self.definition_sha256,
        }

    def validate_binding(
        self,
        objectives: Sequence[ObjectiveSpec],
        outcome_relation_identity: tuple[str, int, str],
    ) -> None:
        """Bind published prose to executable objective and relation semantics."""

        validate_objective_specs(objectives)
        if self.outcome_ordering.relation_identity != outcome_relation_identity:
            raise ValueError(
                "optimization semantics outcome ordering differs from the "
                "executable outcome relation"
            )
        objective_metrics = {
            metric.name: metric
            for metric in self.metrics
            if metric.role is MetricRole.OBJECTIVE
        }
        expected_names = {objective.name for objective in objectives}
        if set(objective_metrics) != expected_names:
            raise ValueError(
                "optimization semantics objective metrics differ from declared "
                "problem objectives"
            )
        for objective in objectives:
            expected_sense = (
                MetricSense.MINIMIZE
                if objective.goal == "min"
                else MetricSense.MAXIMIZE
            )
            if objective_metrics[objective.name].sense is not expected_sense:
                raise ValueError(
                    f"optimization semantics sense differs for {objective.name!r}"
                )


def render_optimization_semantics(semantics: OptimizationSemantics) -> str:
    """Return the canonical prompt block shared by proposal and reflection."""

    if type(semantics) is not OptimizationSemantics:
        raise TypeError("semantics must be an exact OptimizationSemantics")
    OptimizationSemantics.__post_init__(semantics)
    payload = json.dumps(
        semantics.to_record(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "\n".join(
        (
            "OPTIMIZATION SEMANTICS (VERSIONED, AUTHORITATIVE)",
            "Use these exact metric definitions, witness signs, and outcome "
            "ordering; do not infer semantics from metric names.",
            payload,
        )
    )


__all__ = [
    "MetricRole",
    "MetricSemantics",
    "MetricSense",
    "OptimizationSemantics",
    "OutcomeOrderingKind",
    "OutcomeOrderingSemantics",
    "render_optimization_semantics",
]
