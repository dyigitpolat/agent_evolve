"""Provider-neutral contracts for prompt-visible parent measurements.

The selector may reason about an already evaluated parent, but it must not
infer whether a displayed number is a raw physical observation or a value
resolved for optimization.  These immutable values bind both views to one
candidate occurrence, evaluator, objective-resolution law, and decision-metric
schema.  They contain no provider or workload vocabulary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.core.optimization_semantics import MetricRole
from agent_evolve.domain.patch import require_sha256
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_METRIC_ID = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_CANDIDATE_DOMAIN = b"agent-evolve:parent-measurement-candidate:v1\x00"
_PROJECTION_DOMAIN = b"agent-evolve:parent-measurement-projection:v1\x00"
_BINDING_DOMAIN = b"agent-evolve:parent-measurement-binding:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _token(value: object, name: str) -> str:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")
    return value


def _metric_id(value: object, name: str) -> str:
    if type(value) is not str or _METRIC_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed metric grammar")
    return value


def _finite(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")
    return value


@dataclass(frozen=True, slots=True)
class ParentMeasurementProjection:
    """Authority and metric schema under which parent values are projected."""

    benchmark_sha256: str
    session_sha256: str
    decision_metrics: DecisionMetricProjection
    evaluator_id: str
    evaluator_version: int
    evaluator_context_sha256: str
    objective_resolution_policy_id: str
    objective_resolution_policy_version: int
    objective_resolution_definition_sha256: str
    definition_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        require_sha256(self.benchmark_sha256, "benchmark_sha256")
        require_sha256(self.session_sha256, "session_sha256")
        if type(self.decision_metrics) is not DecisionMetricProjection:
            raise TypeError("decision_metrics must be exact DecisionMetricProjection")
        self.decision_metrics.__post_init__()
        _token(self.evaluator_id, "evaluator_id")
        if type(self.evaluator_version) is not int or self.evaluator_version <= 0:
            raise ValueError("evaluator_version must be a positive exact integer")
        require_sha256(self.evaluator_context_sha256, "evaluator_context_sha256")
        _token(
            self.objective_resolution_policy_id,
            "objective_resolution_policy_id",
        )
        if (
            type(self.objective_resolution_policy_version) is not int
            or self.objective_resolution_policy_version <= 0
        ):
            raise ValueError("objective_resolution_policy_version must be positive")
        require_sha256(
            self.objective_resolution_definition_sha256,
            "objective_resolution_definition_sha256",
        )
        computed = _hash(_PROJECTION_DOMAIN, self._unsigned_record())
        if self.definition_sha256 not in ("", computed):
            raise ValueError("definition_sha256 differs from parent projection")
        object.__setattr__(self, "definition_sha256", computed)

    @property
    def evaluator_identity(self) -> tuple[str, int, str]:
        self.__post_init__()
        return (
            self.evaluator_id,
            self.evaluator_version,
            self.evaluator_context_sha256,
        )

    @property
    def objective_resolution_identity(self) -> tuple[str, int, str]:
        self.__post_init__()
        return (
            self.objective_resolution_policy_id,
            self.objective_resolution_policy_version,
            self.objective_resolution_definition_sha256,
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "benchmark_sha256": self.benchmark_sha256,
            "session_sha256": self.session_sha256,
            "decision_metric_projection": self.decision_metrics.to_record(),
            "evaluator": {
                "evaluator_id": self.evaluator_id,
                "evaluator_version": self.evaluator_version,
                "evaluator_context_sha256": self.evaluator_context_sha256,
            },
            "objective_resolution": {
                "policy_id": self.objective_resolution_policy_id,
                "policy_version": self.objective_resolution_policy_version,
                "definition_sha256": (self.objective_resolution_definition_sha256),
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "definition_sha256": self.definition_sha256}


@dataclass(frozen=True, slots=True)
class ParentCandidateMeasurementIdentity:
    """Exact selected occurrence whose completed measurement is displayed."""

    candidate_id: str
    configuration_sha256: str
    configuration_artifact_sha256: str
    proposal_sequence: int
    operator_invocation_id: str | None
    identity_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if (
            type(self.candidate_id) is not str
            or _CANDIDATE_ID.fullmatch(self.candidate_id) is None
        ):
            raise ValueError("candidate_id must use the closed candidate grammar")
        require_sha256(self.configuration_sha256, "configuration_sha256")
        require_sha256(
            self.configuration_artifact_sha256,
            "configuration_artifact_sha256",
        )
        if type(self.proposal_sequence) is not int or self.proposal_sequence < 0:
            raise ValueError("proposal_sequence must be a non-negative exact integer")
        if self.operator_invocation_id is not None and (
            type(self.operator_invocation_id) is not str
            or _CANDIDATE_ID.fullmatch(self.operator_invocation_id) is None
        ):
            raise ValueError("operator_invocation_id must be canonical or None")
        computed = _hash(_CANDIDATE_DOMAIN, self._unsigned_record())
        if self.identity_sha256 not in ("", computed):
            raise ValueError("identity_sha256 differs from parent occurrence")
        object.__setattr__(self, "identity_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "configuration_sha256": self.configuration_sha256,
            "configuration_artifact_sha256": self.configuration_artifact_sha256,
            "proposal_sequence": self.proposal_sequence,
            "operator_invocation_id": self.operator_invocation_id,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "identity_sha256": self.identity_sha256}


@dataclass(frozen=True, slots=True)
class ParentRawScientificMetricValue:
    """One evaluator-scale observation before objective resolution."""

    metric_id: str
    semantic_metric_id: str
    value_name: str
    role: MetricRole
    value: float

    def __post_init__(self) -> None:
        _metric_id(self.metric_id, "metric_id")
        _metric_id(self.semantic_metric_id, "semantic_metric_id")
        _token(self.value_name, "value_name")
        if type(self.role) is not MetricRole or self.role is MetricRole.DIAGNOSTIC:
            raise TypeError("role must be a decision-relevant exact MetricRole")
        _finite(self.value, "raw scientific value")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "semantic_metric_id": self.semantic_metric_id,
            "value_name": self.value_name,
            "role": self.role.value,
            "value_hex": self.value.hex(),
        }


@dataclass(frozen=True, slots=True)
class ParentDecisionMetricValue:
    """One value consumed by optimization after declared resolution."""

    metric_id: str
    value: float

    def __post_init__(self) -> None:
        _metric_id(self.metric_id, "metric_id")
        _finite(self.value, "decision metric value")

    def to_record(self) -> dict[str, str]:
        self.__post_init__()
        return {"metric_id": self.metric_id, "value_hex": self.value.hex()}


@dataclass(frozen=True, slots=True)
class ParentMeasurementBinding:
    """Self-authenticating completed-parent measurement shown to a selector."""

    projection: ParentMeasurementProjection
    candidate: ParentCandidateMeasurementIdentity
    detailed_evaluation_sha256: str
    objective_resolution_receipt_sha256: str | None
    raw_scientific_metrics: tuple[ParentRawScientificMetricValue, ...]
    decision_metrics: tuple[ParentDecisionMetricValue, ...]
    binding_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if type(self.projection) is not ParentMeasurementProjection:
            raise TypeError("projection must be exact ParentMeasurementProjection")
        self.projection.__post_init__()
        if type(self.candidate) is not ParentCandidateMeasurementIdentity:
            raise TypeError("candidate must be exact parent candidate identity")
        self.candidate.__post_init__()
        require_sha256(
            self.detailed_evaluation_sha256,
            "detailed_evaluation_sha256",
        )
        if self.objective_resolution_receipt_sha256 is not None:
            require_sha256(
                self.objective_resolution_receipt_sha256,
                "objective_resolution_receipt_sha256",
            )
        if type(self.raw_scientific_metrics) is not tuple or any(
            type(value) is not ParentRawScientificMetricValue
            for value in self.raw_scientific_metrics
        ):
            raise TypeError("raw_scientific_metrics must contain exact values")
        if type(self.decision_metrics) is not tuple or any(
            type(value) is not ParentDecisionMetricValue
            for value in self.decision_metrics
        ):
            raise TypeError("decision_metrics must contain exact values")
        for value in self.raw_scientific_metrics:
            value.__post_init__()
        for value in self.decision_metrics:
            value.__post_init__()
        expected = self.projection.decision_metrics.metrics
        expected_ids = tuple(value.metric_id for value in expected)
        raw_ids = tuple(value.metric_id for value in self.raw_scientific_metrics)
        decision_ids = tuple(value.metric_id for value in self.decision_metrics)
        if raw_ids != expected_ids or decision_ids != expected_ids:
            raise ValueError("parent measurement metric schema differs from projection")
        for raw, metric in zip(self.raw_scientific_metrics, expected, strict=True):
            if (
                raw.semantic_metric_id != metric.semantic_metric_id
                or raw.value_name != metric.value_name
                or raw.role is not metric.role
            ):
                raise ValueError("raw metric metadata differs from decision projection")
        computed = _hash(_BINDING_DOMAIN, self._unsigned_record())
        if self.binding_sha256 not in ("", computed):
            raise ValueError("binding_sha256 differs from parent measurement")
        object.__setattr__(self, "binding_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "visibility": "completed_selected_parent_before_current_wave",
            "projection": self.projection.to_record(),
            "candidate": self.candidate.to_record(),
            "detailed_evaluation_sha256": self.detailed_evaluation_sha256,
            "objective_resolution_receipt_sha256": (
                self.objective_resolution_receipt_sha256
            ),
            "raw_scientific_metrics": [
                value.to_record() for value in self.raw_scientific_metrics
            ],
            "decision_metrics": [value.to_record() for value in self.decision_metrics],
            "interpretation": {
                "raw_scientific_metrics": (
                    "evaluator-scale observations retained without replacement"
                ),
                "decision_metrics": (
                    "values consumed by optimization under the bound resolution law"
                ),
                "current_wave_outcomes_included": False,
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}


__all__ = [
    "ParentCandidateMeasurementIdentity",
    "ParentDecisionMetricValue",
    "ParentMeasurementBinding",
    "ParentMeasurementProjection",
    "ParentRawScientificMetricValue",
]
