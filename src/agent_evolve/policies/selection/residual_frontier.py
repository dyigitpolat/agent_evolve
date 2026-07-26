"""Residual-hypervolume frontier cells for workload-neutral search planning.

The policy treats the already evaluated Pareto archive as a staircase with
potentially missing cells.  It constructs outcome-blind aspiration points at
pairwise midpoints, then measures the exact fixed-reference hypervolume that
would be added if an aspiration were attained.  A positive value identifies a
frontier gap; a zero value identifies a midpoint already dominated by the
archive.

Only an authenticated affine archive snapshot and already evaluated candidate
objectives enter this module.  Candidate fields, workload names, model/provider
metadata, prompts, and unevaluated outcomes are deliberately outside the API.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.policies.reward.affine_hypervolume_3d import hypervolume_3d
from agent_evolve.policies.reward.frozen_archive import hypervolume_2d

if TYPE_CHECKING:
    from agent_evolve.application.agentic_evolution import EvolutionCandidate
    from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot


RESIDUAL_FRONTIER_POLICY_ID = "residual_hypervolume_frontier_cells"
RESIDUAL_FRONTIER_POLICY_VERSION = 1
RESIDUAL_FRONTIER_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:residual-hypervolume-frontier-cells:v1;"
    b"input=authenticated-affine-prior-archive;dimensions=2-or-3;"
    b"aspirations=pairwise-normalized-midpoints;"
    b"score=exact-fixed-reference-hypervolume-marginal;"
    b"zero-or-roundoff-residuals=excluded;"
    b"ordering=descending-residual-then-canonical-cell-identity;"
    b"candidate-binding=objective-vector-only;"
    b"current-future-outcomes=false;workload-model-provider-fields=false"
).hexdigest()
_CELL_DOMAIN = b"agent-evolve:residual-hypervolume-frontier-cell:v1\x00"
_GEOMETRY_DOMAIN = b"agent-evolve:residual-hypervolume-frontier-geometry:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _decimal(value: float) -> str:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError("residual-frontier values must be finite exact floats")
    return format(0.0 if value == 0.0 else value, ".17g")


def _finite_hex(value: object, *, name: str) -> float:
    if type(value) is not str:
        raise TypeError(f"{name} must be canonical binary64 hex text")
    try:
        result = float.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be canonical binary64 hex text") from error
    if not math.isfinite(result) or result.hex() != value:
        raise ValueError(f"{name} must be a canonical finite binary64 value")
    return result


def _hypervolume(points: tuple[tuple[float, ...], ...]) -> float:
    if not points:
        return 0.0
    dimension = len(points[0])
    if dimension == 2:
        return hypervolume_2d(points, (1.0, 1.0))  # type: ignore[arg-type]
    if dimension == 3:
        return hypervolume_3d(points, (1.0, 1.0, 1.0))  # type: ignore[arg-type]
    raise ValueError("residual-frontier hypervolume supports only 2-D or 3-D")


@dataclass(frozen=True, slots=True)
class ResidualFrontierAxis:
    metric_id: str
    goal: str
    ideal: float
    reference: float

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or not self.metric_id:
            raise ValueError("residual-frontier metric_id must be non-empty")
        if self.goal not in {"min", "max"}:
            raise ValueError("residual-frontier goal must be min or max")
        for name in ("ideal", "reference"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite exact float")
        if self.goal == "min" and not self.reference > self.ideal:
            raise ValueError("minimization reference must exceed ideal")
        if self.goal == "max" and not self.ideal > self.reference:
            raise ValueError("maximization ideal must exceed reference")

    def normalize(self, value: float) -> float:
        self.__post_init__()
        if type(value) is not float or not math.isfinite(value):
            raise TypeError("objective value must be a finite exact float")
        if self.goal == "min":
            return (value - self.ideal) / (self.reference - self.ideal)
        return (self.ideal - value) / (self.ideal - self.reference)

    def denormalize(self, value: float) -> float:
        """Map lower-is-better affine coordinates back to metric space.

        This is the exact inverse of :meth:`normalize` for both minimization
        and maximization axes.  Keeping the inverse on the authenticated axis
        prevents target allocators and prompt adapters from reimplementing
        orientation-sensitive arithmetic.
        """

        self.__post_init__()
        if type(value) is not float or not math.isfinite(value):
            raise TypeError("normalized objective must be a finite exact float")
        return self.ideal + value * (self.reference - self.ideal)

    def to_record(self) -> dict[str, str]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "goal": self.goal,
            "ideal_hex": self.ideal.hex(),
            "reference_hex": self.reference.hex(),
        }


@dataclass(frozen=True, slots=True)
class ResidualFrontierCell:
    anchor_points: tuple[tuple[float, ...], tuple[float, ...]]
    aspiration_point: tuple[float, ...]
    potential_hypervolume_gain: float
    cell_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.anchor_points) is not tuple or len(self.anchor_points) != 2:
            raise TypeError("residual cell requires exactly two anchor points")
        dimension = len(self.anchor_points[0])
        if dimension not in (2, 3):
            raise ValueError("residual cell supports only 2-D or 3-D")
        if self.anchor_points != tuple(sorted(set(self.anchor_points))):
            raise ValueError("residual cell anchors must be unique and canonical")
        for point in (*self.anchor_points, self.aspiration_point):
            if type(point) is not tuple or len(point) != dimension:
                raise TypeError("residual cell points must share one dimension")
            if any(type(value) is not float or not math.isfinite(value) for value in point):
                raise TypeError("residual cell points must contain finite exact floats")
        expected = tuple(
            (left + right) / 2.0
            for left, right in zip(*self.anchor_points, strict=True)
        )
        if self.aspiration_point != expected:
            raise ValueError("residual aspiration must be the exact anchor midpoint")
        if (
            type(self.potential_hypervolume_gain) is not float
            or not math.isfinite(self.potential_hypervolume_gain)
            or self.potential_hypervolume_gain <= 0.0
        ):
            raise ValueError("residual cell must carry positive finite opportunity")
        object.__setattr__(
            self,
            "cell_sha256",
            hashlib.sha256(_CELL_DOMAIN + _canonical_json(self._unsigned_record())).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "anchor_points_hex": [
                [value.hex() for value in point] for point in self.anchor_points
            ],
            "aspiration_point_hex": [value.hex() for value in self.aspiration_point],
            "potential_hypervolume_gain_hex": self.potential_hypervolume_gain.hex(),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "cell_sha256": self.cell_sha256}


@dataclass(frozen=True, slots=True)
class ResidualFrontierGeometry:
    archive_utility_snapshot_sha256: str
    axes: tuple[ResidualFrontierAxis, ...]
    normalized_archive_points: tuple[tuple[float, ...], ...]
    base_hypervolume: float
    cells: tuple[ResidualFrontierCell, ...]
    geometry_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(
            self.archive_utility_snapshot_sha256,
            "archive_utility_snapshot_sha256",
        )
        if type(self.axes) is not tuple or len(self.axes) not in (2, 3):
            raise TypeError("residual geometry requires two or three axes")
        for axis in self.axes:
            if type(axis) is not ResidualFrontierAxis:
                raise TypeError("residual geometry axes must be exact")
            axis.__post_init__()
        if len({axis.metric_id for axis in self.axes}) != len(self.axes):
            raise ValueError("residual geometry axes must be unique")
        dimension = len(self.axes)
        if (
            type(self.normalized_archive_points) is not tuple
            or not self.normalized_archive_points
            or self.normalized_archive_points
            != tuple(sorted(set(self.normalized_archive_points)))
        ):
            raise ValueError("normalized archive points must be non-empty and canonical")
        for point in self.normalized_archive_points:
            if type(point) is not tuple or len(point) != dimension:
                raise TypeError("normalized archive point dimension differs")
            if any(type(value) is not float or not math.isfinite(value) for value in point):
                raise TypeError("normalized archive points must be finite")
        if (
            type(self.base_hypervolume) is not float
            or not math.isfinite(self.base_hypervolume)
            or self.base_hypervolume < 0.0
        ):
            raise ValueError("base_hypervolume must be finite and non-negative")
        if not math.isclose(
            self.base_hypervolume,
            _hypervolume(self.normalized_archive_points),
            rel_tol=0.0,
            abs_tol=64.0 * math.ulp(max(1.0, self.base_hypervolume)),
        ):
            raise ValueError("residual geometry base hypervolume is inconsistent")
        if type(self.cells) is not tuple:
            raise TypeError("residual cells must be an exact tuple")
        for cell in self.cells:
            if type(cell) is not ResidualFrontierCell:
                raise TypeError("residual cells must contain exact values")
            cell.__post_init__()
        expected_cells = tuple(
            sorted(
                self.cells,
                key=lambda value: (
                    -value.potential_hypervolume_gain,
                    value.cell_sha256,
                ),
            )
        )
        if self.cells != expected_cells or len({cell.cell_sha256 for cell in self.cells}) != len(self.cells):
            raise ValueError("residual cells must be unique and opportunity ordered")
        object.__setattr__(
            self,
            "geometry_sha256",
            hashlib.sha256(
                _GEOMETRY_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": RESIDUAL_FRONTIER_POLICY_ID,
                "policy_version": RESIDUAL_FRONTIER_POLICY_VERSION,
                "definition_sha256": RESIDUAL_FRONTIER_POLICY_DEFINITION_SHA256,
            },
            "archive_utility_snapshot_sha256": self.archive_utility_snapshot_sha256,
            "axes": [axis.to_record() for axis in self.axes],
            "normalized_archive_points_hex": [
                [value.hex() for value in point]
                for point in self.normalized_archive_points
            ],
            "base_hypervolume_hex": self.base_hypervolume.hex(),
            "cells": [cell.to_record() for cell in self.cells],
            "current_or_future_candidate_outcomes_consulted": False,
            "workload_model_provider_fields_consulted": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "geometry_sha256": self.geometry_sha256}


def residual_frontier_geometry(
    archive_utility: ArchiveUtilitySnapshot,
) -> ResidualFrontierGeometry:
    """Project exact positive midpoint residuals from one prior archive."""

    from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot

    if type(archive_utility) is not ArchiveUtilitySnapshot:
        raise TypeError("archive_utility must be an exact snapshot")
    archive_utility.__post_init__()
    snapshot = thaw_json(archive_utility.snapshot_receipt)
    if type(snapshot) is not dict:
        raise TypeError("affine archive snapshot must thaw to an object")
    if snapshot.get("definition_sha256") != archive_utility.definition_sha256:
        raise ValueError("affine snapshot definition differs from outer utility")
    spec = snapshot.get("spec")
    raw_axes = spec.get("axes") if type(spec) is dict else None
    raw_points = snapshot.get("normalized_archive_points")
    if type(raw_axes) is not list or len(raw_axes) not in (2, 3):
        raise ValueError("residual-frontier utility requires affine 2-D or 3-D axes")
    if type(raw_points) is not list or not raw_points:
        raise ValueError("residual-frontier utility requires archive points")
    axes: list[ResidualFrontierAxis] = []
    for index, raw in enumerate(raw_axes):
        if type(raw) is not dict:
            raise TypeError(f"axis[{index}] must be an object")
        metric_id = raw.get("metric_id")
        goal = raw.get("goal")
        if type(metric_id) is not str or goal not in {"min", "max"}:
            raise ValueError(f"axis[{index}] is malformed")
        axes.append(
            ResidualFrontierAxis(
                metric_id=metric_id,
                goal=goal,
                ideal=_finite_hex(raw.get("ideal_hex"), name=f"axis[{index}].ideal"),
                reference=_finite_hex(
                    raw.get("reference_hex"),
                    name=f"axis[{index}].reference",
                ),
            )
        )
    dimension = len(axes)
    points = tuple(
        sorted(
            {
                tuple(
                    _finite_hex(value, name=f"point[{row_index}][{axis_index}]")
                    for axis_index, value in enumerate(row)
                )
                for row_index, row in enumerate(raw_points)
                if type(row) is list and len(row) == dimension
            }
        )
    )
    if len(points) != len(raw_points):
        raise ValueError("normalized archive points are malformed or duplicated")
    base = _hypervolume(points)
    recorded_base = _finite_hex(
        snapshot.get("base_hypervolume_hex"),
        name="base_hypervolume",
    )
    tolerance = 64.0 * math.ulp(max(1.0, base, recorded_base))
    if not math.isclose(base, recorded_base, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError("recorded affine hypervolume differs from exact projection")
    cells: list[ResidualFrontierCell] = []
    for left, right in combinations(points, 2):
        aspiration = tuple(
            (first + second) / 2.0
            for first, second in zip(left, right, strict=True)
        )
        gain = max(0.0, _hypervolume((*points, aspiration)) - base)
        if gain <= tolerance:
            continue
        cells.append(
            ResidualFrontierCell(
                anchor_points=(left, right),
                aspiration_point=aspiration,
                potential_hypervolume_gain=float(gain),
            )
        )
    return ResidualFrontierGeometry(
        archive_utility_snapshot_sha256=archive_utility.snapshot_sha256,
        axes=tuple(axes),
        normalized_archive_points=points,
        base_hypervolume=float(base),
        cells=tuple(
            sorted(
                cells,
                key=lambda value: (
                    -value.potential_hypervolume_gain,
                    value.cell_sha256,
                ),
            )
        ),
    )


def normalized_candidate_point(
    geometry: ResidualFrontierGeometry,
    candidate: EvolutionCandidate,
) -> tuple[float, ...]:
    """Bind an evaluated candidate to the same normalized affine frame."""

    from agent_evolve.application.agentic_evolution import EvolutionCandidate

    if type(geometry) is not ResidualFrontierGeometry:
        raise TypeError("geometry must be exact ResidualFrontierGeometry")
    geometry.__post_init__()
    if type(candidate) is not EvolutionCandidate:
        raise TypeError("candidate must be an exact EvolutionCandidate")
    candidate.__post_init__()
    objectives = candidate.objective_map
    metric_ids = tuple(axis.metric_id for axis in geometry.axes)
    if set(objectives) != set(metric_ids):
        raise ValueError("candidate objectives differ from residual-frontier axes")
    return tuple(
        axis.normalize(float(objectives[axis.metric_id])) for axis in geometry.axes
    )


def residual_anchor_parents(
    *,
    geometry: ResidualFrontierGeometry,
    candidates: tuple[EvolutionCandidate, ...],
) -> tuple[EvolutionCandidate, EvolutionCandidate] | None:
    """Return distinct evaluated parents nearest the best residual anchors."""

    from agent_evolve.application.agentic_evolution import EvolutionCandidate

    if type(geometry) is not ResidualFrontierGeometry:
        raise TypeError("geometry must be exact ResidualFrontierGeometry")
    geometry.__post_init__()
    if type(candidates) is not tuple or any(
        type(value) is not EvolutionCandidate for value in candidates
    ):
        raise TypeError("candidates must contain exact EvolutionCandidate values")
    if len(candidates) < 2:
        return None
    if not geometry.cells:
        return None
    rows = tuple((candidate, normalized_candidate_point(geometry, candidate)) for candidate in candidates)
    anchors = geometry.cells[0].anchor_points
    best: tuple[float, tuple[str, str], tuple[EvolutionCandidate, EvolutionCandidate]] | None = None
    for left, right in combinations(rows, 2):
        for ordered in ((left, right), (right, left)):
            distance = sum(
                abs(value - target)
                for (_, point), anchor in zip(ordered, anchors, strict=True)
                for value, target in zip(point, anchor, strict=True)
            )
            parents = (ordered[0][0], ordered[1][0])
            identities = tuple(parent.occurrence.configuration_hash for parent in parents)
            proposal = (distance, identities, parents)
            if best is None or proposal[:2] < best[:2]:
                best = proposal
    if best is None:  # pragma: no cover - combinations guarantee a value.
        return None
    return best[2]


__all__ = [
    "RESIDUAL_FRONTIER_POLICY_DEFINITION_SHA256",
    "RESIDUAL_FRONTIER_POLICY_ID",
    "RESIDUAL_FRONTIER_POLICY_VERSION",
    "ResidualFrontierAxis",
    "ResidualFrontierCell",
    "ResidualFrontierGeometry",
    "normalized_candidate_point",
    "residual_anchor_parents",
    "residual_frontier_geometry",
]
