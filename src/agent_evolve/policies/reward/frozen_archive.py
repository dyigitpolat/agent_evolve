"""Order-invariant marginal hypervolume credit against a frozen wave archive.

The policy deliberately freezes the archive before a concurrent proposal wave.
Every invocation is therefore scored against the same history cutoff, rather
than against an archive that changes with task completion or publication order.
It is an exact two-objective policy; higher-dimensional indicators belong behind
a separate policy with a distinct reward-definition hash.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Protocol

from agent_evolve.core.problem import ObjectiveSpec, validate_objective_specs


ObjectivePoint = tuple[tuple[str, float], ...]
_DEFINITION_DOMAIN = b"agent-evolve:frozen-archive-marginal-hv-reward:v1\x00"
_SNAPSHOT_DOMAIN = b"agent-evolve:frozen-archive-snapshot-2d:v1\x00"


class _RewardCandidate(Protocol):
    valid: bool
    operator_compliant: bool

    @property
    def objective_map(self) -> dict[str, float]: ...


def _finite(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _canonical_point(
    value: Mapping[str, Real],
    objectives: tuple[ObjectiveSpec, ObjectiveSpec],
    *,
    name: str,
) -> ObjectivePoint:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    expected = {objective.name for objective in objectives}
    supplied = set(value)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(
            f"{name} objective keys differ: missing={missing}, extra={extra}"
        )
    return tuple(
        (
            objective.name,
            _finite(value[objective.name], name=f"{name}.{objective.name}"),
        )
        for objective in objectives
    )


def _point_record(point: ObjectivePoint) -> list[list[str]]:
    return [[name, value.hex()] for name, value in point]


def _transform(
    point: ObjectivePoint,
    objectives: tuple[ObjectiveSpec, ObjectiveSpec],
) -> tuple[float, float]:
    values = dict(point)
    return tuple(
        values[objective.name] if objective.goal == "min" else -values[objective.name]
        for objective in objectives
    )  # type: ignore[return-value]


def hypervolume_2d(
    points: Sequence[tuple[Real, Real]],
    reference_point: tuple[Real, Real],
) -> float:
    """Exact sweep for a two-objective minimization front.

    Points that do not strictly improve both coordinates of the reference have
    zero measure inside its rectangle. Dominated and duplicate points are
    removed deterministically before the sweep.
    """

    if type(reference_point) is not tuple or len(reference_point) != 2:
        raise TypeError("reference_point must be an exact two-item tuple")
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise TypeError("points must be a sequence of exact two-item tuples")
    canonical_points: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if type(point) is not tuple or len(point) != 2:
            raise TypeError(f"points[{index}] must be an exact two-item tuple")
        canonical_points.append(
            (
                _finite(point[0], name=f"points[{index}][0]"),
                _finite(point[1], name=f"points[{index}][1]"),
            )
        )
    reference = (
        _finite(reference_point[0], name="reference_point[0]"),
        _finite(reference_point[1], name="reference_point[1]"),
    )
    eligible = sorted(set(canonical_points))
    eligible = [
        point
        for point in eligible
        if point[0] < reference[0] and point[1] < reference[1]
    ]
    frontier: list[tuple[float, float]] = []
    best_second = math.inf
    for point in eligible:
        if point[1] < best_second:
            frontier.append(point)
            best_second = point[1]
    area = 0.0
    for index, point in enumerate(frontier):
        next_first = (
            frontier[index + 1][0] if index + 1 < len(frontier) else reference[0]
        )
        area += (next_first - point[0]) * (reference[1] - point[1])
    if area < 0 or not math.isfinite(area):  # pragma: no cover - sweep guard.
        raise RuntimeError("hypervolume sweep produced an invalid area")
    return area


@dataclass(frozen=True, slots=True)
class FrozenArchiveSnapshot2D:
    """Canonical pre-wave archive and fixed reference for one reward stratum."""

    objectives: tuple[ObjectiveSpec, ObjectiveSpec]
    reference_point: ObjectivePoint
    archive_points: tuple[ObjectivePoint, ...]
    base_hypervolume: float = field(init=False)
    normalization: float = field(init=False)
    definition_hash: str = field(init=False)
    snapshot_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.objectives) is not tuple or len(self.objectives) != 2:
            raise ValueError(
                "frozen hypervolume reward requires exactly two objectives"
            )
        validate_objective_specs(self.objectives)
        if type(self.reference_point) is not tuple:
            raise TypeError("reference_point must be an exact tuple")
        if tuple(name for name, _ in self.reference_point) != tuple(
            objective.name for objective in self.objectives
        ):
            raise ValueError("reference_point must follow objective order")
        for _, value in self.reference_point:
            _finite(value, name="reference point value")
        if type(self.archive_points) is not tuple:
            raise TypeError("archive_points must be an exact tuple")
        expected_names = tuple(objective.name for objective in self.objectives)
        for point in self.archive_points:
            if (
                type(point) is not tuple
                or tuple(name for name, _ in point) != expected_names
            ):
                raise ValueError("archive points must follow objective order")
            for _, value in point:
                _finite(value, name="archive point value")
        canonical = tuple(
            sorted(
                set(self.archive_points),
                key=lambda point: tuple(value for _, value in point),
            )
        )
        if canonical != self.archive_points:
            raise ValueError("archive_points must be unique and canonically sorted")

        reference = _transform(self.reference_point, self.objectives)
        transformed = tuple(
            _transform(point, self.objectives) for point in self.archive_points
        )
        base = hypervolume_2d(transformed, reference)
        normalization = max(abs(base), 1.0)
        definition_record = {
            "schema_version": 1,
            "indicator": "two_objective_fixed_reference_hypervolume",
            "credit": "individual_candidate_marginal_gain_against_frozen_archive",
            "normalization": "divide_by_max_abs_base_hypervolume_or_one",
            "invalid_or_operator_noncompliant_reward": -1.0,
            "zero_gain_reward": 0.0,
            "objectives": [
                {"name": objective.name, "goal": objective.goal}
                for objective in self.objectives
            ],
            "reference_point": _point_record(self.reference_point),
        }
        snapshot_record = {
            "definition": definition_record,
            "archive_points": [_point_record(point) for point in self.archive_points],
            "base_hypervolume": base.hex(),
            "normalization": normalization.hex(),
        }
        definition_payload = json.dumps(
            definition_record,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        snapshot_payload = json.dumps(
            snapshot_record,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        object.__setattr__(self, "base_hypervolume", base)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(
            self,
            "definition_hash",
            hashlib.sha256(_DEFINITION_DOMAIN + definition_payload).hexdigest(),
        )
        object.__setattr__(
            self,
            "snapshot_hash",
            hashlib.sha256(_SNAPSHOT_DOMAIN + snapshot_payload).hexdigest(),
        )

    @classmethod
    def create(
        cls,
        *,
        objectives: Sequence[ObjectiveSpec],
        reference_point: Mapping[str, Real],
        archive_points: Sequence[Mapping[str, Real]],
    ) -> "FrozenArchiveSnapshot2D":
        objective_tuple = tuple(objectives)
        if len(objective_tuple) != 2:
            raise ValueError(
                "frozen hypervolume reward requires exactly two objectives"
            )
        validate_objective_specs(objective_tuple)
        typed_objectives = (objective_tuple[0], objective_tuple[1])
        reference = _canonical_point(
            reference_point,
            typed_objectives,
            name="reference_point",
        )
        points = tuple(
            sorted(
                {
                    _canonical_point(
                        point,
                        typed_objectives,
                        name=f"archive_points[{index}]",
                    )
                    for index, point in enumerate(archive_points)
                },
                key=lambda point: tuple(value for _, value in point),
            )
        )
        return cls(typed_objectives, reference, points)

    def augmented_hypervolume(self, point: Mapping[str, Real]) -> float:
        candidate = _canonical_point(point, self.objectives, name="candidate")
        reference = _transform(self.reference_point, self.objectives)
        transformed = tuple(
            _transform(archive_point, self.objectives)
            for archive_point in (*self.archive_points, candidate)
        )
        return hypervolume_2d(transformed, reference)


@dataclass(frozen=True, slots=True)
class FrozenArchiveRewardRecord:
    status: str
    reward: float
    marginal_hypervolume_gain: float
    base_hypervolume: float
    augmented_hypervolume: float
    normalization: float
    reward_definition_hash: str
    archive_snapshot_hash: str
    candidate_point: ObjectivePoint | None


@dataclass(frozen=True, slots=True)
class FrozenArchiveMarginalHypervolumeReward:
    """Callable engine policy plus an inspectable per-candidate reward record."""

    snapshot: FrozenArchiveSnapshot2D

    @property
    def definition_hash(self) -> str:
        return self.snapshot.definition_hash

    def record(self, child: _RewardCandidate) -> FrozenArchiveRewardRecord:
        if not child.valid:
            return self._failure_record("invalid_candidate")
        if not child.operator_compliant:
            return self._failure_record("operator_noncompliant")
        candidate = _canonical_point(
            child.objective_map,
            self.snapshot.objectives,
            name="child.objectives",
        )
        augmented = self.snapshot.augmented_hypervolume(dict(candidate))
        gain = max(0.0, augmented - self.snapshot.base_hypervolume)
        reward = gain / self.snapshot.normalization
        return FrozenArchiveRewardRecord(
            status="credited",
            reward=float(reward),
            marginal_hypervolume_gain=float(gain),
            base_hypervolume=self.snapshot.base_hypervolume,
            augmented_hypervolume=augmented,
            normalization=self.snapshot.normalization,
            reward_definition_hash=self.definition_hash,
            archive_snapshot_hash=self.snapshot.snapshot_hash,
            candidate_point=candidate,
        )

    def _failure_record(self, status: str) -> FrozenArchiveRewardRecord:
        return FrozenArchiveRewardRecord(
            status=status,
            reward=-1.0,
            marginal_hypervolume_gain=0.0,
            base_hypervolume=self.snapshot.base_hypervolume,
            augmented_hypervolume=self.snapshot.base_hypervolume,
            normalization=self.snapshot.normalization,
            reward_definition_hash=self.definition_hash,
            archive_snapshot_hash=self.snapshot.snapshot_hash,
            candidate_point=None,
        )

    def __call__(
        self,
        child: _RewardCandidate,
        parents: tuple[_RewardCandidate, ...],
        objectives: Sequence[ObjectiveSpec],
    ) -> float:
        del parents
        if tuple(objectives) != self.snapshot.objectives:
            raise ValueError(
                "reward objectives differ from the frozen archive snapshot"
            )
        return self.record(child).reward


__all__ = [
    "FrozenArchiveMarginalHypervolumeReward",
    "FrozenArchiveRewardRecord",
    "FrozenArchiveSnapshot2D",
    "hypervolume_2d",
]
