"""Leakage-safe Airfoil-v7 trim response-model baseline.

This module is deliberately benchmark-local.  AgentEvolve's generic core does
not know that a trim option changes an angle of attack, that the evaluator has
three operating points, or how lift violation and normalized drag aggregate.

The fitted model uses one fully observed adaptation parent (nonce zero) and the
64 trim children around it.  At each operating point, the other two trim
coordinates are irrelevant to the corresponding CFD solve.  The full
Cartesian table therefore supplies 16 replicated observations for each of the
12 ``(point, delta-alpha)`` response cells.  Fitting is fail-closed: all 16
replicates must agree bit-for-bit.  A target decision then adds those learned
lift-residual and normalized-drag deltas to *parent-only* detailed evidence.
No target-child outcome is accepted by the API.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any

from agent_evolve.agentic import (
    DetailedEvaluation,
    DetailedEvaluationPayload,
    EvaluationCheck,
    EvaluationCheckStatus,
    FiniteVariationContract,
    FiniteVariationOption,
    thaw_json,
    typed_json_sha256,
)
from examples.benchmarks.engibench_airfoil.v7_contract import (
    LIFT_TARGET,
    NEUTRAL_POINT_DRAGS,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    EVALUATOR_IDENTITY,
    OBJECTIVE_NAME,
    VIOLATION_NAME,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    AirfoilV7TrimVariationCatalog,
    TRIM_DELTAS_DEG,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_OPTION_PART_TO_DELTA = {
    "n050": -0.50,
    "n025": -0.25,
    "p025": 0.25,
    "p050": 0.50,
}
_DELTA_TO_OPTION_PART = {value: key for key, value in _OPTION_PART_TO_DELTA.items()}
_EXPECTED_DELTA_VECTORS = tuple(itertools.product(TRIM_DELTAS_DEG, repeat=3))
_EXPECTED_OPTION_IDS = tuple(
    "trim." + ".".join(_DELTA_TO_OPTION_PART[value] for value in deltas)
    for deltas in _EXPECTED_DELTA_VECTORS
)

MODEL_SCHEMA_VERSION = 1
MODEL_ID = "airfoil_v7_additive_pointwise_trim_response"
MODEL_VERSION = 1
MODEL_DEFINITION = {
    "schema_version": MODEL_SCHEMA_VERSION,
    "model_id": MODEL_ID,
    "model_version": MODEL_VERSION,
    "training": {
        "parent": "nonce_0_parent_detailed_evidence",
        "children": "all_64_nonce_0_trim_outcomes",
        "response": (
            "child_point_signed_lift_residual_minus_parent_point_"
            "signed_lift_residual_and_child_point_normalized_drag_ratio_"
            "minus_parent_point_normalized_drag_ratio"
        ),
        "replicates_per_cell": 16,
        "separability_gate": "bit_identical_response_within_each_point_delta_cell",
    },
    "transfer": {
        "allowed_target_observation": "target_parent_detailed_evidence_only",
        "target_child_outcomes": "forbidden",
        "prediction": "target_parent_point_value_plus_nonce_0_response_delta",
    },
    "aggregation": {
        "violation": "sum(abs(predicted_signed_lift_residual)/abs(lift_target))",
        "objective": "mean(predicted_normalized_drag_ratio)",
    },
    "ranking": {
        "primary": f"{VIOLATION_NAME}:ascending_exact",
        "secondary": f"{OBJECTIVE_NAME}:ascending_exact",
        "display_tie_breaker": "option_id_ascii",
    },
}

_MODEL_DEFINITION_DOMAIN = b"agent-evolve:airfoil-v7-trim-response-definition:v1\x00"
_TRAINING_INPUT_DOMAIN = b"agent-evolve:airfoil-v7-trim-response-training:v1\x00"
_MODEL_IDENTITY_DOMAIN = b"agent-evolve:airfoil-v7-trim-response-model:v1\x00"
_PARENT_EVIDENCE_DOMAIN = b"agent-evolve:airfoil-v7-parent-evidence:v1\x00"
_REQUEST_DOMAIN = b"agent-evolve:airfoil-v7-trim-response-request:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:airfoil-v7-trim-response-decision:v1\x00"
_ORACLE_MANIFEST_FRAMING = (
    b"agent-evolve:airfoil-v7-finite-oracle-manifest:v1\x00"
)
_ORACLE_RECORD_FRAMING = b"agent-evolve:airfoil-v7-finite-oracle-record:v1\x00"
_ORACLE_RESULT_FRAMING = b"agent-evolve:airfoil-v7-finite-oracle-result:v1\x00"
_ORACLE_FINALIZATION_FRAMING = (
    b"agent-evolve:airfoil-v7-finite-oracle-finalization:v1\x00"
)


class AirfoilV7PhysicsBaselineError(RuntimeError):
    """A source seal, response assumption, or target request is invalid."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


MODEL_DEFINITION_SHA256 = _domain_hash(_MODEL_DEFINITION_DOMAIN, MODEL_DEFINITION)


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float_identity(value: float) -> str:
    return value.hex()


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _option_id_for_deltas(deltas: tuple[float, float, float]) -> str:
    try:
        parts = tuple(_DELTA_TO_OPTION_PART[value] for value in deltas)
    except KeyError as exc:
        raise ValueError("trim deltas are outside the frozen four-value vocabulary") from exc
    return "trim." + ".".join(parts)


def _deltas_from_option_id(option_id: str) -> tuple[float, float, float]:
    parts = option_id.split(".")
    if len(parts) != 4 or parts[0] != "trim":
        raise ValueError("trim option_id must have exactly three delta components")
    try:
        values = tuple(_OPTION_PART_TO_DELTA[part] for part in parts[1:])
    except KeyError as exc:
        raise ValueError("trim option_id uses an unknown delta component") from exc
    if len(values) != 3:  # pragma: no cover - guarded by the split length
        raise AssertionError("unreachable trim arity")
    return values  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class AirfoilV7PointEvidence:
    """The only pointwise aerodynamic values used by the response model."""

    cl: float
    signed_lift_residual: float
    normalized_drag_ratio: float

    def __post_init__(self) -> None:
        for name in ("cl", "signed_lift_residual", "normalized_drag_ratio"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be one canonical finite float")
        if self.normalized_drag_ratio < 0.0:
            raise ValueError("normalized_drag_ratio must be non-negative")
        if not _close(self.signed_lift_residual, self.cl - LIFT_TARGET):
            raise ValueError("signed lift residual is inconsistent with cl and target")

    def to_record(self) -> dict[str, float]:
        return {
            "cl": self.cl,
            "signed_lift_residual": self.signed_lift_residual,
            "normalized_drag_ratio": self.normalized_drag_ratio,
        }

    def identity_record(self) -> dict[str, str]:
        return {name: _float_identity(value) for name, value in self.to_record().items()}


@dataclass(frozen=True, slots=True)
class AirfoilV7ParentEvidence:
    """A parent-bound, receipt-addressed three-point evidence projection."""

    parent_configuration_sha256: str
    evaluator_id: str
    evaluator_version: int
    evaluator_context_sha256: str
    receipt_sha256: str
    source_evidence_sha256: str
    objective: float
    violation: float
    points: tuple[AirfoilV7PointEvidence, AirfoilV7PointEvidence, AirfoilV7PointEvidence]

    def __post_init__(self) -> None:
        _require_sha256(self.parent_configuration_sha256, "parent_configuration_sha256")
        _require_sha256(self.evaluator_context_sha256, "evaluator_context_sha256")
        _require_sha256(self.receipt_sha256, "receipt_sha256")
        _require_sha256(self.source_evidence_sha256, "source_evidence_sha256")
        if type(self.evaluator_id) is not str or not self.evaluator_id:
            raise ValueError("evaluator_id must be non-empty")
        if type(self.evaluator_version) is not int or self.evaluator_version <= 0:
            raise ValueError("evaluator_version must be a positive exact integer")
        if type(self.objective) is not float or not math.isfinite(self.objective):
            raise TypeError("objective must be one canonical finite float")
        if type(self.violation) is not float or not math.isfinite(self.violation):
            raise TypeError("violation must be one canonical finite float")
        if self.violation < 0.0:
            raise ValueError("violation must be non-negative")
        if type(self.points) is not tuple or len(self.points) != 3 or any(
            type(point) is not AirfoilV7PointEvidence for point in self.points
        ):
            raise TypeError("points must be exactly three AirfoilV7PointEvidence values")
        expected_objective = sum(
            point.normalized_drag_ratio for point in self.points
        ) / 3.0
        expected_violation = sum(
            abs(point.signed_lift_residual) / abs(LIFT_TARGET)
            for point in self.points
        )
        if not _close(self.objective, expected_objective):
            raise ValueError("parent objective is inconsistent with point evidence")
        if not _close(self.violation, expected_violation):
            raise ValueError("parent violation is inconsistent with point evidence")

    def identity_record(self) -> dict[str, object]:
        return {
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "evaluator": {
                "evaluator_id": self.evaluator_id,
                "evaluator_version": self.evaluator_version,
                "evaluator_context_sha256": self.evaluator_context_sha256,
            },
            "receipt_sha256": self.receipt_sha256,
            "source_evidence_sha256": self.source_evidence_sha256,
            "objective_hex": _float_identity(self.objective),
            "violation_hex": _float_identity(self.violation),
            "points": [point.identity_record() for point in self.points],
        }

    @property
    def projection_sha256(self) -> str:
        return _domain_hash(_PARENT_EVIDENCE_DOMAIN, self.identity_record())

    def to_trace_record(self) -> dict[str, object]:
        return {
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "evaluator": {
                "evaluator_id": self.evaluator_id,
                "evaluator_version": self.evaluator_version,
                "evaluator_context_sha256": self.evaluator_context_sha256,
            },
            "receipt_sha256": self.receipt_sha256,
            "source_evidence_sha256": self.source_evidence_sha256,
            "projection_sha256": self.projection_sha256,
            "objective": self.objective,
            "violation": self.violation,
            "points": [point.to_record() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class AirfoilV7TrimOutcome:
    """One nonce-zero trim child outcome used during adaptation only."""

    ordinal: int
    option_id: str
    option_identity_sha256: str
    parent_configuration_sha256: str
    child_configuration_sha256: str
    delta_alpha_deg: tuple[float, float, float]
    points: tuple[AirfoilV7PointEvidence, AirfoilV7PointEvidence, AirfoilV7PointEvidence]
    raw_receipt_sha256: str
    terminal_record_sha256: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal <= 0:
            raise ValueError("ordinal must be a positive exact integer")
        if type(self.option_id) is not str or self.option_id != _option_id_for_deltas(
            self.delta_alpha_deg
        ):
            raise ValueError("option_id and delta_alpha_deg disagree")
        for name in (
            "option_identity_sha256",
            "parent_configuration_sha256",
            "child_configuration_sha256",
            "raw_receipt_sha256",
            "terminal_record_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.delta_alpha_deg) is not tuple or len(self.delta_alpha_deg) != 3:
            raise TypeError("delta_alpha_deg must be an exact three-value tuple")
        if any(type(value) is not float for value in self.delta_alpha_deg):
            raise TypeError("delta_alpha_deg values must be canonical floats")
        if type(self.points) is not tuple or len(self.points) != 3 or any(
            type(point) is not AirfoilV7PointEvidence for point in self.points
        ):
            raise TypeError("points must be exactly three AirfoilV7PointEvidence values")

    def identity_record(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "parent_configuration_sha256": self.parent_configuration_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "delta_alpha_deg_hex": [
                _float_identity(value) for value in self.delta_alpha_deg
            ],
            "points": [point.identity_record() for point in self.points],
            "raw_receipt_sha256": self.raw_receipt_sha256,
            "terminal_record_sha256": self.terminal_record_sha256,
        }


@dataclass(frozen=True, slots=True)
class AirfoilV7TrainingSourceSeal:
    """Identity of the recursively sealed oracle inputs used by the fit."""

    oracle_run_id: str
    nonce: int
    manifest_sha256: str
    oracle_result_sha256: str
    oracle_recursive_content_sha256: str
    oracle_finalization_record_sha256: str
    parent_result_file_sha256: str

    def __post_init__(self) -> None:
        if type(self.oracle_run_id) is not str or _RUN_ID.fullmatch(
            self.oracle_run_id
        ) is None:
            raise ValueError("oracle_run_id has invalid syntax")
        if self.nonce != 0:
            raise ValueError("the response baseline is frozen to nonce zero adaptation")
        for name in (
            "manifest_sha256",
            "oracle_result_sha256",
            "oracle_recursive_content_sha256",
            "oracle_finalization_record_sha256",
            "parent_result_file_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    def to_record(self) -> dict[str, object]:
        return {
            "oracle_run_id": self.oracle_run_id,
            "nonce": self.nonce,
            "manifest_sha256": self.manifest_sha256,
            "oracle_result_sha256": self.oracle_result_sha256,
            "oracle_recursive_content_sha256": self.oracle_recursive_content_sha256,
            "oracle_finalization_record_sha256": (
                self.oracle_finalization_record_sha256
            ),
            "parent_result_file_sha256": self.parent_result_file_sha256,
        }


@dataclass(frozen=True, slots=True)
class AirfoilV7TrimTrainingSet:
    """Exactly one parent plus the complete 64-action trim table."""

    source: AirfoilV7TrainingSourceSeal
    parent: AirfoilV7ParentEvidence
    outcomes: tuple[AirfoilV7TrimOutcome, ...]

    def __post_init__(self) -> None:
        if type(self.source) is not AirfoilV7TrainingSourceSeal:
            raise TypeError("source must be an exact AirfoilV7TrainingSourceSeal")
        if type(self.parent) is not AirfoilV7ParentEvidence:
            raise TypeError("parent must be an exact AirfoilV7ParentEvidence")
        if type(self.outcomes) is not tuple or len(self.outcomes) != 64 or any(
            type(outcome) is not AirfoilV7TrimOutcome for outcome in self.outcomes
        ):
            raise TypeError("outcomes must contain exactly 64 trim outcomes")
        ids = tuple(outcome.option_id for outcome in self.outcomes)
        if set(ids) != set(_EXPECTED_OPTION_IDS) or len(set(ids)) != 64:
            raise ValueError("training outcomes do not cover the exact 64-option grid")
        if len({outcome.ordinal for outcome in self.outcomes}) != 64:
            raise ValueError("training outcome ordinals must be unique")
        if any(
            outcome.parent_configuration_sha256
            != self.parent.parent_configuration_sha256
            for outcome in self.outcomes
        ):
            raise ValueError("a training outcome is bound to a different parent")

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "source": self.source.to_record(),
            "parent": self.parent.identity_record(),
            "outcomes": [outcome.identity_record() for outcome in self.outcomes],
        }

    @property
    def input_sha256(self) -> str:
        return _domain_hash(_TRAINING_INPUT_DOMAIN, self.identity_record())


@dataclass(frozen=True, slots=True)
class AirfoilV7TrimResponseCell:
    point_index: int
    delta_alpha_deg: float
    delta_signed_lift_residual: float
    delta_normalized_drag_ratio: float
    replicate_count: int
    source_option_ids: tuple[str, ...]
    source_raw_receipt_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.point_index not in (0, 1, 2):
            raise ValueError("point_index must be 0, 1, or 2")
        if type(self.delta_alpha_deg) is not float or self.delta_alpha_deg not in (
            TRIM_DELTAS_DEG
        ):
            raise ValueError("delta_alpha_deg is outside the frozen vocabulary")
        for name in (
            "delta_signed_lift_residual",
            "delta_normalized_drag_ratio",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be one canonical finite float")
        if self.replicate_count != 16:
            raise ValueError("each response cell must have exactly 16 replicates")
        if (
            type(self.source_option_ids) is not tuple
            or len(self.source_option_ids) != 16
            or tuple(sorted(self.source_option_ids)) != self.source_option_ids
            or len(set(self.source_option_ids)) != 16
        ):
            raise ValueError("source_option_ids must be 16 unique sorted IDs")
        if (
            type(self.source_raw_receipt_sha256s) is not tuple
            or len(self.source_raw_receipt_sha256s) != 16
        ):
            raise ValueError("source receipt identities must retain all 16 replicates")
        for value in self.source_raw_receipt_sha256s:
            _require_sha256(value, "source_raw_receipt_sha256")

    def identity_record(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "delta_alpha_deg_hex": _float_identity(self.delta_alpha_deg),
            "delta_signed_lift_residual_hex": _float_identity(
                self.delta_signed_lift_residual
            ),
            "delta_normalized_drag_ratio_hex": _float_identity(
                self.delta_normalized_drag_ratio
            ),
            "replicate_count": self.replicate_count,
            "source_option_ids": list(self.source_option_ids),
            "source_raw_receipt_sha256s": list(
                self.source_raw_receipt_sha256s
            ),
        }

    def to_trace_record(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "delta_alpha_deg": self.delta_alpha_deg,
            "delta_signed_lift_residual": self.delta_signed_lift_residual,
            "delta_normalized_drag_ratio": self.delta_normalized_drag_ratio,
            "replicate_count": self.replicate_count,
            "source_option_ids": list(self.source_option_ids),
            "source_raw_receipt_sha256s": list(
                self.source_raw_receipt_sha256s
            ),
        }


@dataclass(frozen=True, slots=True)
class AirfoilV7TrimResponseModel:
    definition_sha256: str
    training_input_sha256: str
    training_source: AirfoilV7TrainingSourceSeal
    evaluator_context_sha256: str
    cells: tuple[AirfoilV7TrimResponseCell, ...]

    def __post_init__(self) -> None:
        if self.definition_sha256 != MODEL_DEFINITION_SHA256:
            raise ValueError("model definition identity changed")
        _require_sha256(self.training_input_sha256, "training_input_sha256")
        _require_sha256(self.evaluator_context_sha256, "evaluator_context_sha256")
        if type(self.training_source) is not AirfoilV7TrainingSourceSeal:
            raise TypeError("training_source must be exact")
        if type(self.cells) is not tuple or len(self.cells) != 12 or any(
            type(cell) is not AirfoilV7TrimResponseCell for cell in self.cells
        ):
            raise TypeError("cells must contain exactly 12 response cells")
        observed = tuple((cell.point_index, cell.delta_alpha_deg) for cell in self.cells)
        expected = tuple(
            (point_index, delta)
            for point_index in range(3)
            for delta in TRIM_DELTAS_DEG
        )
        if observed != expected:
            raise ValueError("response cells must use canonical point/delta order")

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "definition_sha256": self.definition_sha256,
            "training_input_sha256": self.training_input_sha256,
            "training_source": self.training_source.to_record(),
            "evaluator_context_sha256": self.evaluator_context_sha256,
            "cells": [cell.identity_record() for cell in self.cells],
        }

    @property
    def model_sha256(self) -> str:
        return _domain_hash(_MODEL_IDENTITY_DOMAIN, self.identity_record())

    def cell(self, point_index: int, delta_alpha_deg: float) -> AirfoilV7TrimResponseCell:
        matches = tuple(
            cell
            for cell in self.cells
            if cell.point_index == point_index
            and cell.delta_alpha_deg == delta_alpha_deg
        )
        if len(matches) != 1:  # pragma: no cover - constructor guarantees this
            raise AirfoilV7PhysicsBaselineError("response cell lookup is ambiguous")
        return matches[0]

    def to_trace_record(self) -> dict[str, object]:
        return {
            "kind": "airfoil_v7_trim_response_model",
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "definition_sha256": self.definition_sha256,
            "training_input_sha256": self.training_input_sha256,
            "training_source": self.training_source.to_record(),
            "evaluator_context_sha256": self.evaluator_context_sha256,
            "cells": [cell.to_trace_record() for cell in self.cells],
            "model_sha256": self.model_sha256,
        }


def fit_airfoil_v7_trim_response_model(
    training: AirfoilV7TrimTrainingSet,
) -> AirfoilV7TrimResponseModel:
    """Fit the exact 12-cell additive response lookup, failing on interactions."""

    if type(training) is not AirfoilV7TrimTrainingSet:
        raise TypeError("training must be an exact AirfoilV7TrimTrainingSet")
    AirfoilV7TrimTrainingSet.__post_init__(training)
    grouped: dict[tuple[int, float], list[AirfoilV7TrimOutcome]] = defaultdict(list)
    for outcome in training.outcomes:
        for point_index, delta in enumerate(outcome.delta_alpha_deg):
            grouped[(point_index, delta)].append(outcome)

    cells: list[AirfoilV7TrimResponseCell] = []
    for point_index in range(3):
        parent_point = training.parent.points[point_index]
        for delta in TRIM_DELTAS_DEG:
            observations = sorted(
                grouped[(point_index, delta)], key=lambda outcome: outcome.option_id
            )
            if len(observations) != 16:
                raise AirfoilV7PhysicsBaselineError(
                    "each point/delta response cell requires 16 observations"
                )
            response_pairs = tuple(
                (
                    outcome.points[point_index].signed_lift_residual
                    - parent_point.signed_lift_residual,
                    outcome.points[point_index].normalized_drag_ratio
                    - parent_point.normalized_drag_ratio,
                )
                for outcome in observations
            )
            bit_patterns = {
                (_float_identity(lift_delta), _float_identity(drag_delta))
                for lift_delta, drag_delta in response_pairs
            }
            if len(bit_patterns) != 1:
                raise AirfoilV7PhysicsBaselineError(
                    "pointwise trim separability failed: replicated response "
                    f"disagrees for point={point_index}, delta={delta:+.2f}"
                )
            lift_delta, drag_delta = response_pairs[0]
            cells.append(
                AirfoilV7TrimResponseCell(
                    point_index=point_index,
                    delta_alpha_deg=delta,
                    delta_signed_lift_residual=lift_delta,
                    delta_normalized_drag_ratio=drag_delta,
                    replicate_count=16,
                    source_option_ids=tuple(
                        outcome.option_id for outcome in observations
                    ),
                    source_raw_receipt_sha256s=tuple(
                        outcome.raw_receipt_sha256 for outcome in observations
                    ),
                )
            )
    return AirfoilV7TrimResponseModel(
        definition_sha256=MODEL_DEFINITION_SHA256,
        training_input_sha256=training.input_sha256,
        training_source=training.source,
        evaluator_context_sha256=training.parent.evaluator_context_sha256,
        cells=tuple(cells),
    )


@dataclass(frozen=True, slots=True)
class AirfoilV7PredictedPoint:
    point_index: int
    delta_alpha_deg: float
    parent_signed_lift_residual: float
    parent_normalized_drag_ratio: float
    predicted_signed_lift_residual: float
    predicted_normalized_drag_ratio: float

    def __post_init__(self) -> None:
        if self.point_index not in (0, 1, 2):
            raise ValueError("predicted point_index must be 0, 1, or 2")
        if type(self.delta_alpha_deg) is not float or self.delta_alpha_deg not in (
            TRIM_DELTAS_DEG
        ):
            raise ValueError("predicted delta_alpha_deg is outside the vocabulary")
        for name in (
            "parent_signed_lift_residual",
            "parent_normalized_drag_ratio",
            "predicted_signed_lift_residual",
            "predicted_normalized_drag_ratio",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be one canonical finite float")
        if (
            self.parent_normalized_drag_ratio < 0.0
            or self.predicted_normalized_drag_ratio < 0.0
        ):
            raise ValueError("parent and predicted drag ratios must be non-negative")

    def to_record(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "delta_alpha_deg": self.delta_alpha_deg,
            "parent_signed_lift_residual": self.parent_signed_lift_residual,
            "parent_normalized_drag_ratio": self.parent_normalized_drag_ratio,
            "predicted_signed_lift_residual": self.predicted_signed_lift_residual,
            "predicted_normalized_drag_ratio": self.predicted_normalized_drag_ratio,
        }


@dataclass(frozen=True, slots=True)
class AirfoilV7TrimPrediction:
    option_id: str
    option_identity_sha256: str
    child_configuration_sha256: str
    predicted_violation: float
    predicted_objective: float
    points: tuple[AirfoilV7PredictedPoint, ...]
    rank: int

    def __post_init__(self) -> None:
        deltas = _deltas_from_option_id(self.option_id)
        _require_sha256(self.option_identity_sha256, "option_identity_sha256")
        _require_sha256(
            self.child_configuration_sha256, "child_configuration_sha256"
        )
        if type(self.predicted_violation) is not float or not math.isfinite(
            self.predicted_violation
        ):
            raise TypeError("predicted_violation must be one canonical finite float")
        if type(self.predicted_objective) is not float or not math.isfinite(
            self.predicted_objective
        ):
            raise TypeError("predicted_objective must be one canonical finite float")
        if self.predicted_violation < 0.0:
            raise ValueError("predicted_violation must be non-negative")
        if type(self.rank) is not int or not 1 <= self.rank <= 64:
            raise ValueError("rank must be an exact integer between 1 and 64")
        if type(self.points) is not tuple or len(self.points) != 3 or any(
            type(point) is not AirfoilV7PredictedPoint for point in self.points
        ):
            raise TypeError("prediction points must contain exactly three values")
        if tuple(point.point_index for point in self.points) != (0, 1, 2):
            raise ValueError("prediction points must use canonical point order")
        if tuple(point.delta_alpha_deg for point in self.points) != deltas:
            raise ValueError("prediction point deltas disagree with option_id")
        expected_violation = sum(
            abs(point.predicted_signed_lift_residual) / abs(LIFT_TARGET)
            for point in self.points
        )
        expected_objective = sum(
            point.predicted_normalized_drag_ratio for point in self.points
        ) / 3.0
        if self.predicted_violation != expected_violation:
            raise ValueError("predicted violation is not the exact point aggregation")
        if self.predicted_objective != expected_objective:
            raise ValueError("predicted objective is not the exact point aggregation")

    def to_record(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "predicted_violation": self.predicted_violation,
            "predicted_objective": self.predicted_objective,
            "rank": self.rank,
            "points": [point.to_record() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class AirfoilV7PhysicsPortfolioDecision:
    model_sha256: str
    target_contract_sha256: str
    target_parent_evidence_sha256: str
    request_sha256: str
    top_k: int
    predictions: tuple[AirfoilV7TrimPrediction, ...]
    selected_option_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "model_sha256",
            "target_contract_sha256",
            "target_parent_evidence_sha256",
            "request_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.top_k) is not int or not 1 <= self.top_k <= 64:
            raise ValueError("top_k must be between 1 and 64")
        if type(self.predictions) is not tuple or len(self.predictions) != 64:
            raise ValueError("the response baseline must predict all 64 trim options")
        if any(
            type(prediction) is not AirfoilV7TrimPrediction
            for prediction in self.predictions
        ):
            raise TypeError("predictions must contain exact trim predictions")
        expected_ranks = tuple(range(1, 65))
        if tuple(prediction.rank for prediction in self.predictions) != expected_ranks:
            raise ValueError("predictions must be in complete canonical rank order")
        if len({prediction.option_id for prediction in self.predictions}) != 64:
            raise ValueError("predictions must contain 64 distinct options")
        expected_order = tuple(
            sorted(
                self.predictions,
                key=lambda prediction: (
                    prediction.predicted_violation,
                    prediction.predicted_objective,
                    prediction.option_id,
                ),
            )
        )
        if self.predictions != expected_order:
            raise ValueError("predictions violate exact violation-then-drag ordering")
        if type(self.selected_option_ids) is not tuple or self.selected_option_ids != tuple(
            prediction.option_id for prediction in self.predictions[: self.top_k]
        ):
            raise ValueError("selected options must be the exact top-k rank prefix")

    def identity_record(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_sha256": self.model_sha256,
            "target_contract_sha256": self.target_contract_sha256,
            "target_parent_evidence_sha256": self.target_parent_evidence_sha256,
            "request_sha256": self.request_sha256,
            "top_k": self.top_k,
            "selected_option_ids": list(self.selected_option_ids),
            "predictions": [
                {
                    "option_id": prediction.option_id,
                    "option_identity_sha256": prediction.option_identity_sha256,
                    "child_configuration_sha256": (
                        prediction.child_configuration_sha256
                    ),
                    "predicted_violation_hex": _float_identity(
                        prediction.predicted_violation
                    ),
                    "predicted_objective_hex": _float_identity(
                        prediction.predicted_objective
                    ),
                    "rank": prediction.rank,
                }
                for prediction in self.predictions
            ],
        }

    @property
    def decision_sha256(self) -> str:
        return _domain_hash(_DECISION_DOMAIN, self.identity_record())

    def to_trace_record(self) -> dict[str, object]:
        return {
            "kind": "airfoil_v7_physics_response_portfolio_decision",
            "schema_version": MODEL_SCHEMA_VERSION,
            "information_boundary": {
                "adaptation_observations": "nonce_0_parent_plus_64_trim_children",
                "target_observations": "target_parent_detailed_evidence_only",
                "target_child_outcomes_observed": 0,
            },
            "ranking": MODEL_DEFINITION["ranking"],
            "model_sha256": self.model_sha256,
            "target_contract_sha256": self.target_contract_sha256,
            "target_parent_evidence_sha256": self.target_parent_evidence_sha256,
            "request_sha256": self.request_sha256,
            "top_k": self.top_k,
            "selected_option_ids": list(self.selected_option_ids),
            "predictions": [prediction.to_record() for prediction in self.predictions],
            "decision_sha256": self.decision_sha256,
        }


def _point_from_observed(value: object, *, label: str) -> AirfoilV7PointEvidence:
    if not isinstance(value, Mapping):
        raise AirfoilV7PhysicsBaselineError(f"{label} observed evidence is not an object")
    return AirfoilV7PointEvidence(
        cl=_require_finite(value.get("cl"), f"{label}.cl"),
        signed_lift_residual=_require_finite(
            value.get("signed_lift_residual"), f"{label}.signed_lift_residual"
        ),
        normalized_drag_ratio=_require_finite(
            value.get("normalized_drag_ratio"), f"{label}.normalized_drag_ratio"
        ),
    )


def parent_evidence_from_detailed_evaluation(
    *,
    parent_configuration_sha256: str,
    detailed_evaluation: DetailedEvaluation | DetailedEvaluationPayload,
) -> AirfoilV7ParentEvidence:
    """Project one successful parent evaluation; no child data can enter here."""

    _require_sha256(parent_configuration_sha256, "parent_configuration_sha256")
    if type(detailed_evaluation) is DetailedEvaluation:
        if not detailed_evaluation.success:
            raise AirfoilV7PhysicsBaselineError("target parent evaluation failed")
        checks = detailed_evaluation.checks
        objectives = dict(detailed_evaluation.objectives)
        violations = dict(detailed_evaluation.violations)
        receipt = detailed_evaluation.receipt
        evaluator = detailed_evaluation.payload.evaluator
        source_evidence_sha256 = detailed_evaluation.evidence_sha256
    elif type(detailed_evaluation) is DetailedEvaluationPayload:
        if detailed_evaluation.failure is not None:
            raise AirfoilV7PhysicsBaselineError("target parent evaluation failed")
        checks = detailed_evaluation.checks
        objectives = dict(detailed_evaluation.objectives)
        violations = dict(detailed_evaluation.violations)
        receipt = detailed_evaluation.receipt
        evaluator = detailed_evaluation.evaluator
        payload_identity = {
            "checks": [check.to_record() for check in checks],
            "objectives": {
                name: _float_identity(value) for name, value in objectives.items()
            },
            "violations": {
                name: _float_identity(value) for name, value in violations.items()
            },
            "receipt_sha256": None if receipt is None else receipt.sha256_hex,
            "evaluator": evaluator.to_record(),
        }
        source_evidence_sha256 = _domain_hash(
            _PARENT_EVIDENCE_DOMAIN, payload_identity
        )
    else:
        raise TypeError(
            "detailed_evaluation must be an exact DetailedEvaluation or "
            "DetailedEvaluationPayload"
        )
    if receipt is None:
        raise AirfoilV7PhysicsBaselineError("parent evidence requires a durable receipt")
    by_name: dict[str, EvaluationCheck] = {check.name: check for check in checks}
    points: list[AirfoilV7PointEvidence] = []
    for point_index in range(3):
        name = f"point_{point_index}_convergence"
        check = by_name.get(name)
        if check is None or check.status is not EvaluationCheckStatus.PASS:
            raise AirfoilV7PhysicsBaselineError(
                f"parent evidence requires passing {name}"
            )
        points.append(
            _point_from_observed(thaw_json(check.observed_value), label=name)
        )
    try:
        objective = objectives[OBJECTIVE_NAME]
        violation = violations[VIOLATION_NAME]
    except KeyError as exc:
        raise AirfoilV7PhysicsBaselineError(
            "parent evidence is missing the Airfoil-v7 aggregate metrics"
        ) from exc
    return AirfoilV7ParentEvidence(
        parent_configuration_sha256=parent_configuration_sha256,
        evaluator_id=evaluator.evaluator_id,
        evaluator_version=evaluator.evaluator_version,
        evaluator_context_sha256=evaluator.evaluator_context_sha256,
        receipt_sha256=receipt.sha256_hex,
        source_evidence_sha256=source_evidence_sha256,
        objective=float(objective),
        violation=float(violation),
        points=tuple(points),  # type: ignore[arg-type]
    )


def _validated_trim_contract_deltas(
    contract: FiniteVariationContract,
) -> dict[str, tuple[float, float, float]]:
    if type(contract) is not FiniteVariationContract:
        raise TypeError("target_contract must be an exact FiniteVariationContract")
    FiniteVariationContract.__post_init__(contract)
    if (
        contract.catalog_id != AirfoilV7TrimVariationCatalog.catalog_id
        or contract.catalog_version != AirfoilV7TrimVariationCatalog.catalog_version
        or contract.catalog_definition_sha256
        != AirfoilV7TrimVariationCatalog.definition_sha256
    ):
        raise ValueError("target contract is not the frozen Airfoil-v7 trim catalog")
    if len(contract.options) != 64 or {
        option.option_id for option in contract.options
    } != set(_EXPECTED_OPTION_IDS):
        raise ValueError("target trim contract must contain the exact 64-option set")
    parent = thaw_json(contract.parent_configuration)
    if not isinstance(parent, Mapping) or not isinstance(parent.get("alpha_deg"), list):
        raise ValueError("target parent has no three-point alpha vector")
    parent_alpha = parent["alpha_deg"]
    if len(parent_alpha) != 3:
        raise ValueError("target parent alpha vector must have three points")
    deltas_by_id: dict[str, tuple[float, float, float]] = {}
    for option in contract.options:
        if type(option) is not FiniteVariationOption or option.family != "trim_only":
            raise ValueError("target contract contains a non-trim option")
        deltas = _deltas_from_option_id(option.option_id)
        child = thaw_json(option.child_configuration)
        if not isinstance(child, Mapping) or not isinstance(child.get("alpha_deg"), list):
            raise ValueError("trim child has no three-point alpha vector")
        if child.get("upper_coefficients") != parent.get("upper_coefficients") or child.get(
            "lower_coefficients"
        ) != parent.get("lower_coefficients"):
            raise ValueError("a trim option changed target-parent geometry")
        child_alpha = child["alpha_deg"]
        if len(child_alpha) != 3:
            raise ValueError("trim child alpha vector must have three points")
        for point_index, expected_delta in enumerate(deltas):
            observed_delta = _require_finite(
                child_alpha[point_index], "child alpha"
            ) - _require_finite(parent_alpha[point_index], "parent alpha")
            if not _close(observed_delta, expected_delta):
                raise ValueError("trim option ID and materialized child disagree")
        deltas_by_id[option.option_id] = deltas
    return deltas_by_id


class AirfoilV7TrimPhysicsResponseSelector:
    """Engine-owned Airfoil-local H arm; it never calls an LLM or target child."""

    def __init__(self, model: AirfoilV7TrimResponseModel) -> None:
        if type(model) is not AirfoilV7TrimResponseModel:
            raise TypeError("model must be an exact AirfoilV7TrimResponseModel")
        AirfoilV7TrimResponseModel.__post_init__(model)
        self._model = model

    @property
    def model(self) -> AirfoilV7TrimResponseModel:
        return self._model

    def select(
        self,
        *,
        target_contract: FiniteVariationContract,
        target_parent_evaluation: DetailedEvaluation | DetailedEvaluationPayload,
        top_k: int = 3,
    ) -> AirfoilV7PhysicsPortfolioDecision:
        if type(top_k) is not int or not 1 <= top_k <= 64:
            raise ValueError("top_k must be a positive exact integer at most 64")
        deltas_by_id = _validated_trim_contract_deltas(target_contract)
        parent_sha256 = typed_json_sha256(target_contract.parent_configuration)
        parent = parent_evidence_from_detailed_evaluation(
            parent_configuration_sha256=parent_sha256,
            detailed_evaluation=target_parent_evaluation,
        )
        if parent.evaluator_context_sha256 != self._model.evaluator_context_sha256:
            raise AirfoilV7PhysicsBaselineError(
                "target evaluator context differs from the adaptation model"
            )
        request_record = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_sha256": self._model.model_sha256,
            "target_contract_sha256": target_contract.identity_sha256,
            "target_parent_evidence_sha256": parent.projection_sha256,
            "top_k": top_k,
            "target_child_outcomes_observed": 0,
        }
        request_sha256 = _domain_hash(_REQUEST_DOMAIN, request_record)
        options_by_id = {option.option_id: option for option in target_contract.options}
        unordered: list[tuple[FiniteVariationOption, float, float, tuple[AirfoilV7PredictedPoint, ...]]] = []
        for option_id in _EXPECTED_OPTION_IDS:
            option = options_by_id[option_id]
            predicted_points: list[AirfoilV7PredictedPoint] = []
            for point_index, delta in enumerate(deltas_by_id[option_id]):
                parent_point = parent.points[point_index]
                cell = self._model.cell(point_index, delta)
                predicted_points.append(
                    AirfoilV7PredictedPoint(
                        point_index=point_index,
                        delta_alpha_deg=delta,
                        parent_signed_lift_residual=(
                            parent_point.signed_lift_residual
                        ),
                        parent_normalized_drag_ratio=(
                            parent_point.normalized_drag_ratio
                        ),
                        predicted_signed_lift_residual=(
                            parent_point.signed_lift_residual
                            + cell.delta_signed_lift_residual
                        ),
                        predicted_normalized_drag_ratio=(
                            parent_point.normalized_drag_ratio
                            + cell.delta_normalized_drag_ratio
                        ),
                    )
                )
            predicted_violation = sum(
                abs(point.predicted_signed_lift_residual) / abs(LIFT_TARGET)
                for point in predicted_points
            )
            predicted_objective = sum(
                point.predicted_normalized_drag_ratio for point in predicted_points
            ) / 3.0
            unordered.append(
                (
                    option,
                    predicted_violation,
                    predicted_objective,
                    tuple(predicted_points),
                )
            )
        ordered = sorted(
            unordered,
            key=lambda row: (row[1], row[2], row[0].option_id),
        )
        predictions = tuple(
            AirfoilV7TrimPrediction(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                child_configuration_sha256=option.child_configuration_sha256,
                predicted_violation=violation,
                predicted_objective=objective,
                points=points,
                rank=rank,
            )
            for rank, (option, violation, objective, points) in enumerate(
                ordered, start=1
            )
        )
        return AirfoilV7PhysicsPortfolioDecision(
            model_sha256=self._model.model_sha256,
            target_contract_sha256=target_contract.identity_sha256,
            target_parent_evidence_sha256=parent.projection_sha256,
            request_sha256=request_sha256,
            top_k=top_k,
            predictions=predictions,
            selected_option_ids=tuple(
                prediction.option_id for prediction in predictions[:top_k]
            ),
        )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AirfoilV7PhysicsBaselineError(f"{label} is unreadable") from exc
    if type(value) is not dict:
        raise AirfoilV7PhysicsBaselineError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> tuple[str, int]:
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest(), len(content)


def _verify_self_hash(
    record: Mapping[str, Any],
    *,
    field: str,
    framing: bytes,
    label: str,
) -> str:
    claimed = _require_sha256(record.get(field), f"{label}.{field}")
    unsigned = dict(record)
    unsigned.pop(field, None)
    expected = hashlib.sha256(framing + _canonical_bytes(unsigned)).hexdigest()
    if claimed != expected:
        raise AirfoilV7PhysicsBaselineError(f"{label} self-hash failed")
    return claimed


def _verify_recursive_oracle_seal(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, object]]]:
    finalized_path = run_dir / "finalized.json"
    finalized = _read_json_object(finalized_path, "oracle finalization")
    if finalized.get("kind") != "oracle_run_finalized":
        raise AirfoilV7PhysicsBaselineError("oracle finalization kind changed")
    _verify_self_hash(
        finalized,
        field="record_sha256",
        framing=_ORACLE_RECORD_FRAMING,
        label="oracle finalization",
    )
    expected_files = finalized.get("files")
    if type(expected_files) is not dict:
        raise AirfoilV7PhysicsBaselineError("oracle finalization file map is malformed")
    observed_files: dict[str, dict[str, object]] = {}
    aggregate = hashlib.sha256(_ORACLE_FINALIZATION_FRAMING)
    paths = sorted(
        (
            path
            for path in run_dir.rglob("*")
            if path.is_file() and path != finalized_path
        ),
        key=lambda path: path.relative_to(run_dir).as_posix(),
    )
    for path in paths:
        if path.is_symlink():
            raise AirfoilV7PhysicsBaselineError("oracle seal contains a symbolic link")
        relative = path.relative_to(run_dir).as_posix()
        content = path.read_bytes()
        observed_files[relative] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
        encoded = relative.encode("utf-8")
        aggregate.update(len(encoded).to_bytes(8, "big"))
        aggregate.update(encoded)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
    if expected_files != observed_files:
        raise AirfoilV7PhysicsBaselineError("oracle recursive file map changed")
    if finalized.get("recursive_file_count") != len(observed_files):
        raise AirfoilV7PhysicsBaselineError("oracle recursive file count changed")
    if finalized.get("recursive_content_sha256") != aggregate.hexdigest():
        raise AirfoilV7PhysicsBaselineError("oracle recursive content seal changed")
    return finalized, observed_files


def _binding_path(value: object, *, label: str) -> tuple[Path, str]:
    if type(value) is not dict:
        raise AirfoilV7PhysicsBaselineError(f"{label} binding is malformed")
    try:
        path = Path(value["path"]).expanduser().resolve(strict=True)
    except (KeyError, TypeError, OSError) as exc:
        raise AirfoilV7PhysicsBaselineError(f"{label} path is invalid") from exc
    sha256, size = _sha256_file(path)
    if value.get("sha256") != sha256 or value.get("bytes") != size:
        raise AirfoilV7PhysicsBaselineError(f"{label} bytes changed")
    return path, sha256


def _point_from_raw(raw_point: object, point_index: int) -> AirfoilV7PointEvidence:
    if type(raw_point) is not dict or raw_point.get("index") != point_index:
        raise AirfoilV7PhysicsBaselineError("raw receipt point order changed")
    cl = _require_finite(raw_point.get("cl"), "raw point cl")
    cd = _require_finite(raw_point.get("cd"), "raw point cd")
    return AirfoilV7PointEvidence(
        cl=cl,
        signed_lift_residual=cl - LIFT_TARGET,
        normalized_drag_ratio=cd / NEUTRAL_POINT_DRAGS[point_index],
    )


def _parent_from_prior_result(
    *,
    prior_result: Mapping[str, Any],
    parent_configuration_sha256: str,
) -> AirfoilV7ParentEvidence:
    seeds = prior_result.get("seeds")
    if type(seeds) is not list:
        raise AirfoilV7PhysicsBaselineError("prior result has no seed evidence")
    matches: list[dict[str, Any]] = []
    for seed in seeds:
        if type(seed) is not dict or type(seed.get("configuration")) is not dict:
            continue
        from agent_evolve.agentic import freeze_json

        if typed_json_sha256(freeze_json(seed["configuration"])) == parent_configuration_sha256:
            matches.append(seed)
    if len(matches) != 1:
        raise AirfoilV7PhysicsBaselineError(
            "prior result does not contain exactly one nonce-zero parent"
        )
    seed = matches[0]
    detailed = seed.get("detailed_evaluation")
    if type(detailed) is not dict or detailed.get("failure") is not None:
        raise AirfoilV7PhysicsBaselineError("nonce-zero parent evidence failed")
    checks = detailed.get("checks")
    if type(checks) is not list:
        raise AirfoilV7PhysicsBaselineError("nonce-zero parent checks are missing")
    by_name = {
        check.get("name"): check
        for check in checks
        if type(check) is dict and type(check.get("name")) is str
    }
    if any(
        type(by_name.get(f"point_{point_index}_convergence")) is not dict
        or by_name[f"point_{point_index}_convergence"].get("status") != "pass"
        for point_index in range(3)
    ):
        raise AirfoilV7PhysicsBaselineError(
            "nonce-zero parent point convergence evidence is incomplete"
        )
    points = tuple(
        _point_from_observed(
            by_name[f"point_{point_index}_convergence"].get("observed_value"),
            label=f"nonce_zero.point_{point_index}",
        )
        for point_index in range(3)
    )
    evaluator = detailed.get("evaluator")
    receipt = detailed.get("receipt")
    objectives = detailed.get("objectives")
    violations = detailed.get("violations")
    if not all(type(value) is dict for value in (evaluator, receipt, objectives, violations)):
        raise AirfoilV7PhysicsBaselineError("nonce-zero parent evidence is malformed")
    return AirfoilV7ParentEvidence(
        parent_configuration_sha256=parent_configuration_sha256,
        evaluator_id=str(evaluator.get("evaluator_id")),
        evaluator_version=int(evaluator.get("evaluator_version")),
        evaluator_context_sha256=str(evaluator.get("evaluator_context_sha256")),
        receipt_sha256=str(receipt.get("sha256_hex")),
        source_evidence_sha256=str(detailed.get("evidence_sha256")),
        objective=_require_finite(objectives.get(OBJECTIVE_NAME), "parent objective"),
        violation=_require_finite(violations.get(VIOLATION_NAME), "parent violation"),
        points=points,  # type: ignore[arg-type]
    )


def load_sealed_nonce_zero_trim_training_set(
    oracle_run_dir: Path,
) -> AirfoilV7TrimTrainingSet:
    """Authenticate the sealed oracle and extract only the allowed H-arm data."""

    run_dir = oracle_run_dir.expanduser().resolve(strict=True)
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    finalized, _ = _verify_recursive_oracle_seal(run_dir)
    manifest = _read_json_object(run_dir / "oracle_manifest.json", "oracle manifest")
    manifest_sha256 = _verify_self_hash(
        manifest,
        field="manifest_sha256",
        framing=_ORACLE_MANIFEST_FRAMING,
        label="oracle manifest",
    )
    result = _read_json_object(run_dir / "oracle_result.json", "oracle result")
    result_sha256 = _verify_self_hash(
        result,
        field="result_sha256",
        framing=_ORACLE_RESULT_FRAMING,
        label="oracle result",
    )
    oracle = manifest.get("oracle")
    if type(oracle) is not dict:
        raise AirfoilV7PhysicsBaselineError("oracle manifest payload is malformed")
    run_id = oracle.get("run_id")
    if (
        type(run_id) is not str
        or result.get("run_id") != run_id
        or result.get("manifest_sha256") != manifest_sha256
        or result.get("status") != "completed_80_action_oracle"
        or result.get("successful_candidates") != 80
    ):
        raise AirfoilV7PhysicsBaselineError("oracle result identity/status changed")
    catalog = _read_json_object(run_dir / "catalog_contract.json", "oracle catalog")
    if oracle.get("catalog") != catalog:
        raise AirfoilV7PhysicsBaselineError(
            "sealed manifest and materialized oracle catalog disagree"
        )
    parent_record = catalog.get("parent")
    contract_record = catalog.get("contract")
    evaluation_order = catalog.get("evaluation_order")
    if (
        type(parent_record) is not dict
        or parent_record.get("nonce") != 0
        or type(contract_record) is not dict
        or type(evaluation_order) is not list
    ):
        raise AirfoilV7PhysicsBaselineError("oracle catalog/parent binding changed")
    parent_configuration_sha256 = _require_sha256(
        contract_record.get("parent_configuration_sha256"),
        "catalog parent_configuration_sha256",
    )
    if parent_record.get("typed_configuration_sha256") != parent_configuration_sha256:
        raise AirfoilV7PhysicsBaselineError("catalog nonce-zero parent identity changed")

    prior_runs = oracle.get("prior_runs")
    if type(prior_runs) is not list or not prior_runs or type(prior_runs[0]) is not dict:
        raise AirfoilV7PhysicsBaselineError("oracle prior-run binding is missing")
    prior_files = prior_runs[0].get("files")
    if type(prior_files) is not dict:
        raise AirfoilV7PhysicsBaselineError("oracle prior-run files are malformed")
    prior_result_path, prior_result_file_sha256 = _binding_path(
        prior_files.get("result.json"), label="nonce-zero parent result"
    )
    prior_finalized_path, _ = _binding_path(
        prior_files.get("finalized.json"), label="nonce-zero parent finalization"
    )
    prior_finalized = _read_json_object(
        prior_finalized_path, "nonce-zero parent finalization"
    )
    prior_result_relative = prior_result_path.relative_to(prior_finalized_path.parent).as_posix()
    if (
        type(prior_finalized.get("files")) is not dict
        or prior_finalized["files"].get(prior_result_relative, {}).get("sha256")
        != prior_result_file_sha256
    ):
        raise AirfoilV7PhysicsBaselineError(
            "nonce-zero parent result is not covered by its finalization"
        )
    prior_result = _read_json_object(prior_result_path, "nonce-zero parent result")
    parent = _parent_from_prior_result(
        prior_result=prior_result,
        parent_configuration_sha256=parent_configuration_sha256,
    )
    if parent.evaluator_context_sha256 != EVALUATOR_IDENTITY.evaluator_context_sha256:
        raise AirfoilV7PhysicsBaselineError("adaptation evaluator context changed")

    result_rows = result.get("results")
    if type(result_rows) is not list:
        raise AirfoilV7PhysicsBaselineError("oracle result rows are missing")
    result_by_id = {
        row.get("option_id"): row for row in result_rows if type(row) is dict
    }
    outcomes: list[AirfoilV7TrimOutcome] = []
    parent_candidate = parent_record.get("candidate")
    if type(parent_candidate) is not dict or type(parent_candidate.get("alpha_deg")) is not list:
        raise AirfoilV7PhysicsBaselineError("nonce-zero parent candidate is malformed")
    parent_alpha = parent_candidate["alpha_deg"]
    for catalog_row in evaluation_order:
        if type(catalog_row) is not dict or catalog_row.get("family") != "trim_only":
            continue
        option_id = str(catalog_row.get("option_id"))
        ordinal = catalog_row.get("ordinal")
        if type(ordinal) is not int:
            raise AirfoilV7PhysicsBaselineError("trim ordinal is malformed")
        option_dir = run_dir / "options" / f"{ordinal:03d}-{option_id}"
        terminal_path = option_dir / "terminal.json"
        terminal = _read_json_object(terminal_path, f"terminal {option_id}")
        terminal_record_sha256 = _verify_self_hash(
            terminal,
            field="record_sha256",
            framing=_ORACLE_RECORD_FRAMING,
            label=f"terminal {option_id}",
        )
        if (
            terminal.get("disposition") != "success"
            or terminal.get("option_id") != option_id
            or terminal.get("ordinal") != ordinal
            or terminal.get("option_identity_sha256")
            != catalog_row.get("option_identity_sha256")
        ):
            raise AirfoilV7PhysicsBaselineError(
                f"terminal identity/status changed for {option_id}"
            )
        raw_binding = terminal.get("raw_receipt")
        if type(raw_binding) is not dict or type(raw_binding.get("relative_path")) is not str:
            raise AirfoilV7PhysicsBaselineError(f"raw receipt link missing for {option_id}")
        raw_path = (run_dir / raw_binding["relative_path"]).resolve(strict=True)
        if not raw_path.is_relative_to(run_dir):
            raise AirfoilV7PhysicsBaselineError("raw receipt path escaped oracle run")
        raw_sha256, raw_size = _sha256_file(raw_path)
        if raw_binding.get("sha256") != raw_sha256 or raw_binding.get("bytes") != raw_size:
            raise AirfoilV7PhysicsBaselineError(f"raw receipt changed for {option_id}")
        raw = _read_json_object(raw_path, f"raw receipt {option_id}")
        child = catalog_row.get("child_configuration")
        if type(child) is not dict or raw.get("candidate") != child:
            raise AirfoilV7PhysicsBaselineError(
                f"catalog and raw child disagree for {option_id}"
            )
        if raw.get("candidate_sha256") != catalog_row.get("raw_candidate_sha256"):
            raise AirfoilV7PhysicsBaselineError(
                f"raw candidate identity changed for {option_id}"
            )
        raw_points = raw.get("points")
        if type(raw_points) is not list or len(raw_points) != 3:
            raise AirfoilV7PhysicsBaselineError(
                f"raw receipt does not have three points for {option_id}"
            )
        points = tuple(
            _point_from_raw(raw_points[index], index) for index in range(3)
        )
        deltas = _deltas_from_option_id(option_id)
        child_alpha = child.get("alpha_deg")
        if type(child_alpha) is not list or len(child_alpha) != 3:
            raise AirfoilV7PhysicsBaselineError("trim child alpha vector is malformed")
        for point_index, delta in enumerate(deltas):
            if not _close(float(child_alpha[point_index]) - float(parent_alpha[point_index]), delta):
                raise AirfoilV7PhysicsBaselineError(
                    f"trim materialization disagrees with {option_id}"
                )
        result_row = result_by_id.get(option_id)
        if type(result_row) is not dict:
            raise AirfoilV7PhysicsBaselineError(f"result row missing for {option_id}")
        objective = sum(point.normalized_drag_ratio for point in points) / 3.0
        violation = sum(
            abs(point.signed_lift_residual) / abs(LIFT_TARGET) for point in points
        )
        row_objectives = result_row.get("objectives")
        row_violations = result_row.get("violations")
        if (
            type(row_objectives) is not dict
            or type(row_violations) is not dict
            or not _close(
                objective,
                _require_finite(row_objectives.get(OBJECTIVE_NAME), "result objective"),
            )
            or not _close(
                violation,
                _require_finite(row_violations.get(VIOLATION_NAME), "result violation"),
            )
            or result_row.get("terminal_record_sha256") != terminal_record_sha256
        ):
            raise AirfoilV7PhysicsBaselineError(
                f"raw receipt and aggregate result disagree for {option_id}"
            )
        outcomes.append(
            AirfoilV7TrimOutcome(
                ordinal=ordinal,
                option_id=option_id,
                option_identity_sha256=str(catalog_row.get("option_identity_sha256")),
                parent_configuration_sha256=parent_configuration_sha256,
                child_configuration_sha256=str(
                    catalog_row.get("typed_child_configuration_sha256")
                ),
                delta_alpha_deg=deltas,
                points=points,  # type: ignore[arg-type]
                raw_receipt_sha256=raw_sha256,
                terminal_record_sha256=terminal_record_sha256,
            )
        )
    source = AirfoilV7TrainingSourceSeal(
        oracle_run_id=run_id,
        nonce=0,
        manifest_sha256=manifest_sha256,
        oracle_result_sha256=result_sha256,
        oracle_recursive_content_sha256=str(
            finalized.get("recursive_content_sha256")
        ),
        oracle_finalization_record_sha256=str(finalized.get("record_sha256")),
        parent_result_file_sha256=prior_result_file_sha256,
    )
    return AirfoilV7TrimTrainingSet(
        source=source,
        parent=parent,
        outcomes=tuple(sorted(outcomes, key=lambda outcome: outcome.ordinal)),
    )


def load_sealed_nonce_zero_trim_response_model(
    oracle_run_dir: Path,
) -> AirfoilV7TrimResponseModel:
    """Authenticate, extract, and fit the frozen nonce-zero response model."""

    return fit_airfoil_v7_trim_response_model(
        load_sealed_nonce_zero_trim_training_set(oracle_run_dir)
    )


__all__ = [
    "AirfoilV7ParentEvidence",
    "AirfoilV7PhysicsBaselineError",
    "AirfoilV7PhysicsPortfolioDecision",
    "AirfoilV7PointEvidence",
    "AirfoilV7PredictedPoint",
    "AirfoilV7TrainingSourceSeal",
    "AirfoilV7TrimOutcome",
    "AirfoilV7TrimPhysicsResponseSelector",
    "AirfoilV7TrimPrediction",
    "AirfoilV7TrimResponseCell",
    "AirfoilV7TrimResponseModel",
    "AirfoilV7TrimTrainingSet",
    "MODEL_DEFINITION",
    "MODEL_DEFINITION_SHA256",
    "MODEL_ID",
    "MODEL_VERSION",
    "fit_airfoil_v7_trim_response_model",
    "load_sealed_nonce_zero_trim_response_model",
    "load_sealed_nonce_zero_trim_training_set",
    "parent_evidence_from_detailed_evaluation",
]
