"""Order-invariant joint portfolio credit against one frozen archive.

The legacy portfolio workflow aggregated parent-relative child rewards.  That
can score a new Pareto extreme negatively even when the complete portfolio
substantially expands the archive.  This module instead measures the joint
hypervolume gain of a whole candidate wave against a pre-wave archive shared by
all concurrent lanes.

The implementation is intentionally limited to exact two-objective indicators.
Other dimensionalities belong behind separately identified policies rather
than silently changing the indicator or its approximation error.
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
from agent_evolve.policies.reward.frozen_archive import hypervolume_2d


ObjectivePoint = tuple[tuple[str, float], ...]
_DEFINITION_DOMAIN = b"agent-evolve:frozen-archive-wave-hv-reward:v1\x00"
_SNAPSHOT_DOMAIN = b"agent-evolve:frozen-archive-wave-hv-snapshot:v1\x00"


class WaveRewardCandidate(Protocol):
    """Minimal candidate surface required by joint archive credit."""

    valid: bool
    operator_compliant: bool
    evidence_compliant: bool

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


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


@dataclass(frozen=True, slots=True)
class FrozenArchiveWaveSnapshot2D:
    """Exact archive cutoff and fixed reference for joint wave credit."""

    objectives: tuple[ObjectiveSpec, ObjectiveSpec]
    reference_point: ObjectivePoint
    archive_points: tuple[ObjectivePoint, ...]
    base_hypervolume: float = field(init=False)
    normalization: float = field(init=False)
    definition_hash: str = field(init=False)
    snapshot_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.objectives) is not tuple or len(self.objectives) != 2:
            raise ValueError("joint wave hypervolume requires exactly two objectives")
        validate_objective_specs(self.objectives)
        names = tuple(objective.name for objective in self.objectives)
        if (
            type(self.reference_point) is not tuple
            or tuple(name for name, _ in self.reference_point) != names
        ):
            raise ValueError("reference_point must follow exact objective order")
        for _, value in self.reference_point:
            _finite(value, name="reference point value")
        if type(self.archive_points) is not tuple:
            raise TypeError("archive_points must be an exact tuple")
        for point in self.archive_points:
            if type(point) is not tuple or tuple(name for name, _ in point) != names:
                raise ValueError("archive points must follow exact objective order")
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
        definition = {
            "schema_version": 1,
            "indicator": "exact_two_objective_fixed_reference_hypervolume",
            "credit": "joint_candidate_wave_gain_against_frozen_archive",
            "concurrent_lane_cutoff": "same_pre_generation_archive",
            "normalization": "divide_by_max_abs_base_hypervolume_or_one",
            "inadmissible_members": (
                "excluded_from_archive gain; fixed wave slots retain opportunity cost"
            ),
            "zero_gain_reward": 0.0,
            "objectives": [
                {"name": objective.name, "goal": objective.goal}
                for objective in self.objectives
            ],
            "reference_point": _point_record(self.reference_point),
        }
        snapshot = {
            "definition": definition,
            "archive_points": [_point_record(point) for point in self.archive_points],
            "base_hypervolume": base.hex(),
            "normalization": normalization.hex(),
        }
        object.__setattr__(self, "base_hypervolume", base)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(
            self,
            "definition_hash",
            hashlib.sha256(
                _DEFINITION_DOMAIN + _canonical_json(definition)
            ).hexdigest(),
        )
        object.__setattr__(
            self,
            "snapshot_hash",
            hashlib.sha256(_SNAPSHOT_DOMAIN + _canonical_json(snapshot)).hexdigest(),
        )

    @classmethod
    def create(
        cls,
        *,
        objectives: Sequence[ObjectiveSpec],
        reference_point: Mapping[str, Real],
        archive_points: Sequence[Mapping[str, Real]],
    ) -> "FrozenArchiveWaveSnapshot2D":
        objective_tuple = tuple(objectives)
        if len(objective_tuple) != 2:
            raise ValueError("joint wave hypervolume requires exactly two objectives")
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

    def augmented_hypervolume(self, points: Sequence[Mapping[str, Real]]) -> float:
        """Return exact HV after jointly adding all supplied admissible points."""

        if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
            raise TypeError("points must be a sequence of objective mappings")
        candidates = tuple(
            _canonical_point(point, self.objectives, name=f"points[{index}]")
            for index, point in enumerate(points)
        )
        reference = _transform(self.reference_point, self.objectives)
        transformed = tuple(
            _transform(point, self.objectives)
            for point in (*self.archive_points, *candidates)
        )
        return hypervolume_2d(transformed, reference)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "definition_hash": self.definition_hash,
            "snapshot_hash": self.snapshot_hash,
            "objectives": [
                {"name": objective.name, "goal": objective.goal}
                for objective in self.objectives
            ],
            "reference_point": _point_record(self.reference_point),
            "archive_points": [_point_record(point) for point in self.archive_points],
            "base_hypervolume_hex": self.base_hypervolume.hex(),
            "normalization_hex": self.normalization.hex(),
        }


@dataclass(frozen=True, slots=True)
class FrozenArchiveWaveRewardRecord:
    """Inspectable joint archive-gain result for one fixed-cardinality wave."""

    status: str
    reward: float
    joint_hypervolume_gain: float
    base_hypervolume: float
    augmented_hypervolume: float
    normalization: float
    member_count: int
    admitted_count: int
    invalid_count: int
    operator_noncompliant_count: int
    evidence_noncompliant_count: int
    missing_count: int
    candidate_points: tuple[ObjectivePoint, ...]
    reward_definition_hash: str
    archive_snapshot_hash: str

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reward_hex": self.reward.hex(),
            "joint_hypervolume_gain_hex": self.joint_hypervolume_gain.hex(),
            "base_hypervolume_hex": self.base_hypervolume.hex(),
            "augmented_hypervolume_hex": self.augmented_hypervolume.hex(),
            "normalization_hex": self.normalization.hex(),
            "member_count": self.member_count,
            "admitted_count": self.admitted_count,
            "invalid_count": self.invalid_count,
            "operator_noncompliant_count": self.operator_noncompliant_count,
            "evidence_noncompliant_count": self.evidence_noncompliant_count,
            "missing_count": self.missing_count,
            "candidate_points": [
                _point_record(point) for point in self.candidate_points
            ],
            "reward_definition_hash": self.reward_definition_hash,
            "archive_snapshot_hash": self.archive_snapshot_hash,
        }


@dataclass(frozen=True, slots=True)
class FrozenArchiveJointWaveHypervolumeReward:
    """Score a complete wave by normalized counterfactual archive gain."""

    snapshot: FrozenArchiveWaveSnapshot2D

    @property
    def definition_hash(self) -> str:
        return self.snapshot.definition_hash

    def record(
        self,
        children: Sequence[WaveRewardCandidate | None],
    ) -> FrozenArchiveWaveRewardRecord:
        if isinstance(children, (str, bytes)) or not isinstance(children, Sequence):
            raise TypeError("children must be a sequence of candidates or None")
        if not children:
            raise ValueError("joint wave reward requires at least one planned member")

        admitted: list[ObjectivePoint] = []
        invalid_count = 0
        operator_noncompliant_count = 0
        evidence_noncompliant_count = 0
        missing_count = 0
        for index, child in enumerate(children):
            if child is None:
                missing_count += 1
            elif not child.valid:
                invalid_count += 1
            elif not child.operator_compliant:
                operator_noncompliant_count += 1
            elif not child.evidence_compliant:
                evidence_noncompliant_count += 1
            else:
                admitted.append(
                    _canonical_point(
                        child.objective_map,
                        self.snapshot.objectives,
                        name=f"children[{index}].objectives",
                    )
                )

        augmented = self.snapshot.augmented_hypervolume(
            tuple(dict(point) for point in admitted)
        )
        gain = max(0.0, augmented - self.snapshot.base_hypervolume)
        reward = gain / self.snapshot.normalization
        return FrozenArchiveWaveRewardRecord(
            status="credited" if admitted else "no_admissible_candidates",
            reward=float(reward),
            joint_hypervolume_gain=float(gain),
            base_hypervolume=self.snapshot.base_hypervolume,
            augmented_hypervolume=augmented,
            normalization=self.snapshot.normalization,
            member_count=len(children),
            admitted_count=len(admitted),
            invalid_count=invalid_count,
            operator_noncompliant_count=operator_noncompliant_count,
            evidence_noncompliant_count=evidence_noncompliant_count,
            missing_count=missing_count,
            candidate_points=tuple(admitted),
            reward_definition_hash=self.definition_hash,
            archive_snapshot_hash=self.snapshot.snapshot_hash,
        )

    def __call__(self, children: Sequence[WaveRewardCandidate | None]) -> float:
        return self.record(children).reward


__all__ = [
    "FrozenArchiveJointWaveHypervolumeReward",
    "FrozenArchiveWaveRewardRecord",
    "FrozenArchiveWaveSnapshot2D",
    "WaveRewardCandidate",
]
