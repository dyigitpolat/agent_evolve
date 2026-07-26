"""Scale-invariant fixed-reference hypervolume for two-objective campaigns.

The policy freezes an affine, dimensionless objective transform before a
generation.  Workloads inject only metric identifiers, optimization goals,
and fixed ideal/reference values; archive credit, portfolio credit, and pair
utility all reuse the resulting authenticated snapshot.

For a minimization axis ``u=(x-ideal)/(reference-ideal)``.  For a
maximization axis ``u=(ideal-x)/(ideal-reference)``.  In both cases the ideal
maps to zero, the fixed reference maps to one, and ordinary two-dimensional
minimization hypervolume is measured against ``(1, 1)``.  This makes the
utility invariant to positive changes of units when the injected bounds are
transformed with the observations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Protocol

from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.reward.frozen_archive import hypervolume_2d


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_DEFINITION_DOMAIN = b"agent-evolve:affine-hypervolume-2d-definition:v1\x00"
_SNAPSHOT_DOMAIN = b"agent-evolve:affine-hypervolume-2d-snapshot:v1\x00"
_UTILITY_ID = "affine_fixed_reference_hypervolume_2d"
_UTILITY_VERSION = 1


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _finite(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    if type(result) is not FrozenJsonObject:  # pragma: no cover - closed input.
        raise AssertionError("affine hypervolume record did not freeze as an object")
    return result


@dataclass(frozen=True, slots=True)
class AffineObjectiveAxis:
    """One objective's fixed affine orientation and scale."""

    metric_id: str
    goal: str
    ideal: float
    reference: float

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or _TOKEN.fullmatch(self.metric_id) is None:
            raise ValueError("metric_id must use the closed token grammar")
        if self.goal not in {"min", "max"}:
            raise ValueError("goal must be 'min' or 'max'")
        ideal = _finite(self.ideal, name="ideal")
        reference = _finite(self.reference, name="reference")
        if self.goal == "min" and not reference > ideal:
            raise ValueError("a minimization reference must exceed its ideal")
        if self.goal == "max" and not ideal > reference:
            raise ValueError("a maximization ideal must exceed its reference")

    @property
    def span(self) -> float:
        self.__post_init__()
        return float(abs(self.reference - self.ideal))

    def normalize(self, value: Real) -> float:
        number = _finite(value, name=f"objective {self.metric_id}")
        if self.goal == "min":
            return (number - self.ideal) / (self.reference - self.ideal)
        return (self.ideal - number) / (self.ideal - self.reference)

    def to_record(self) -> dict[str, str]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "goal": self.goal,
            "ideal_hex": float(self.ideal).hex(),
            "reference_hex": float(self.reference).hex(),
            "span_hex": self.span.hex(),
        }


@dataclass(frozen=True, slots=True)
class AffineHypervolume2DSpec:
    """Complete immutable definition of a dimensionless 2-D indicator."""

    axes: tuple[AffineObjectiveAxis, AffineObjectiveAxis]
    reference_provenance: str

    def __post_init__(self) -> None:
        if type(self.axes) is not tuple or len(self.axes) != 2 or any(
            type(axis) is not AffineObjectiveAxis for axis in self.axes
        ):
            raise TypeError("axes must contain exactly two AffineObjectiveAxis values")
        for axis in self.axes:
            AffineObjectiveAxis.__post_init__(axis)
        if self.axes[0].metric_id == self.axes[1].metric_id:
            raise ValueError("affine hypervolume axes must name distinct metrics")
        if (
            type(self.reference_provenance) is not str
            or not self.reference_provenance.strip()
            or self.reference_provenance != self.reference_provenance.strip()
        ):
            raise ValueError("reference_provenance must be canonical non-empty text")

    @property
    def metric_ids(self) -> tuple[str, str]:
        return (self.axes[0].metric_id, self.axes[1].metric_id)

    @property
    def raw_area_scale(self) -> float:
        return self.axes[0].span * self.axes[1].span

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "indicator": "exact_affine_normalized_two_objective_hypervolume",
            "axes": [axis.to_record() for axis in self.axes],
            "normalized_reference": ["0x1.0000000000000p+0"] * 2,
            "raw_area_scale_hex": self.raw_area_scale.hex(),
            "reference_provenance": self.reference_provenance,
            "out_of_reference_points": "zero_measure_clipped_by_hypervolume_sweep",
        }

    @property
    def definition_sha256(self) -> str:
        return hashlib.sha256(
            _DEFINITION_DOMAIN + _canonical_json(self.to_record())
        ).hexdigest()

    def normalize(self, point: Mapping[str, Real]) -> tuple[float, float]:
        if not isinstance(point, Mapping):
            raise TypeError("objective point must be a mapping")
        if set(point) != set(self.metric_ids):
            raise ValueError("objective point differs from the affine metric set")
        return (
            self.axes[0].normalize(point[self.axes[0].metric_id]),
            self.axes[1].normalize(point[self.axes[1].metric_id]),
        )


RawObjectivePoint = tuple[tuple[str, float], tuple[str, float]]


def _raw_point(
    spec: AffineHypervolume2DSpec,
    value: Mapping[str, Real],
) -> RawObjectivePoint:
    spec.normalize(value)
    return (
        (spec.axes[0].metric_id, float(value[spec.axes[0].metric_id])),
        (spec.axes[1].metric_id, float(value[spec.axes[1].metric_id])),
    )


def _point_record(point: RawObjectivePoint) -> list[list[str]]:
    return [[metric_id, value.hex()] for metric_id, value in point]


@dataclass(frozen=True, slots=True)
class AffineHypervolumeSnapshot2D:
    """One frozen archive cutoff in raw and normalized coordinates."""

    spec: AffineHypervolume2DSpec
    raw_archive_points: tuple[RawObjectivePoint, ...]
    normalized_archive_points: tuple[tuple[float, float], ...] = field(init=False)
    base_hypervolume: float = field(init=False)
    raw_oriented_base_hypervolume: float = field(init=False)
    snapshot_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.spec) is not AffineHypervolume2DSpec:
            raise TypeError("spec must be an exact AffineHypervolume2DSpec")
        self.spec.__post_init__()
        if type(self.raw_archive_points) is not tuple:
            raise TypeError("raw_archive_points must be an exact tuple")
        expected_names = self.spec.metric_ids
        for point in self.raw_archive_points:
            if (
                type(point) is not tuple
                or len(point) != 2
                or tuple(name for name, _ in point) != expected_names
            ):
                raise ValueError("raw archive point differs from the affine axes")
            for name, value in point:
                _finite(value, name=f"archive point {name}")
        canonical = tuple(
            sorted(
                set(self.raw_archive_points),
                key=lambda point: tuple(value for _, value in point),
            )
        )
        if self.raw_archive_points != canonical:
            raise ValueError("raw archive points must be unique and canonical")
        normalized = tuple(
            self.spec.normalize(dict(point)) for point in self.raw_archive_points
        )
        base = hypervolume_2d(normalized, (1.0, 1.0))
        object.__setattr__(self, "normalized_archive_points", normalized)
        object.__setattr__(self, "base_hypervolume", base)
        object.__setattr__(
            self,
            "raw_oriented_base_hypervolume",
            base * self.spec.raw_area_scale,
        )
        object.__setattr__(
            self,
            "snapshot_sha256",
            hashlib.sha256(
                _SNAPSHOT_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    @classmethod
    def create(
        cls,
        *,
        spec: AffineHypervolume2DSpec,
        archive_points: Sequence[Mapping[str, Real]],
    ) -> "AffineHypervolumeSnapshot2D":
        if isinstance(archive_points, (str, bytes)) or not isinstance(
            archive_points, Sequence
        ):
            raise TypeError("archive_points must be a sequence")
        points = tuple(
            sorted(
                {
                    _raw_point(spec, value)
                    for value in archive_points
                },
                key=lambda point: tuple(value for _, value in point),
            )
        )
        return cls(spec=spec, raw_archive_points=points)

    def augmented_hypervolume(
        self,
        points: Sequence[Mapping[str, Real]],
    ) -> float:
        if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
            raise TypeError("points must be a sequence")
        normalized = tuple(self.spec.normalize(value) for value in points)
        return hypervolume_2d(
            (*self.normalized_archive_points, *normalized),
            (1.0, 1.0),
        )

    def joint_gain(self, points: Sequence[Mapping[str, Real]]) -> float:
        return max(0.0, self.augmented_hypervolume(points) - self.base_hypervolume)

    def marginal_gain(self, point: Mapping[str, Real]) -> float:
        return self.joint_gain((point,))

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "definition_sha256": self.spec.definition_sha256,
            "spec": self.spec.to_record(),
            "raw_archive_points": [
                _point_record(point) for point in self.raw_archive_points
            ],
            "normalized_archive_points": [
                [first.hex(), second.hex()]
                for first, second in self.normalized_archive_points
            ],
            "base_hypervolume_hex": self.base_hypervolume.hex(),
            "raw_oriented_base_hypervolume_hex": (
                self.raw_oriented_base_hypervolume.hex()
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "snapshot_sha256": self.snapshot_sha256}


class AffineWaveRewardCandidate(Protocol):
    valid: bool
    operator_compliant: bool
    evidence_compliant: bool

    @property
    def objective_map(self) -> dict[str, float]: ...


@dataclass(frozen=True, slots=True)
class AffineHypervolumeWaveRewardRecord:
    reward: float
    raw_oriented_gain: float
    base_hypervolume: float
    augmented_hypervolume: float
    member_count: int
    admitted_count: int
    snapshot_sha256: str
    definition_sha256: str

    def to_record(self) -> dict[str, object]:
        return {
            "reward_hex": self.reward.hex(),
            "raw_oriented_gain_hex": self.raw_oriented_gain.hex(),
            "base_hypervolume_hex": self.base_hypervolume.hex(),
            "augmented_hypervolume_hex": self.augmented_hypervolume.hex(),
            "member_count": self.member_count,
            "admitted_count": self.admitted_count,
            "snapshot_sha256": self.snapshot_sha256,
            "definition_sha256": self.definition_sha256,
        }


@dataclass(frozen=True, slots=True)
class AffineFrozenArchiveJointWaveReward:
    """Dimensionless joint gain for one fixed-cardinality candidate wave."""

    snapshot: AffineHypervolumeSnapshot2D

    @property
    def definition_hash(self) -> str:
        return self.snapshot.spec.definition_sha256

    def record(
        self,
        children: Sequence[AffineWaveRewardCandidate | None],
    ) -> AffineHypervolumeWaveRewardRecord:
        if isinstance(children, (str, bytes)) or not isinstance(children, Sequence):
            raise TypeError("children must be a sequence")
        if not children:
            raise ValueError("wave reward requires at least one planned member")
        admitted = tuple(
            child.objective_map
            for child in children
            if child is not None
            and child.valid
            and child.operator_compliant
            and child.evidence_compliant
        )
        augmented = self.snapshot.augmented_hypervolume(admitted)
        gain = max(0.0, augmented - self.snapshot.base_hypervolume)
        return AffineHypervolumeWaveRewardRecord(
            reward=gain,
            raw_oriented_gain=gain * self.snapshot.spec.raw_area_scale,
            base_hypervolume=self.snapshot.base_hypervolume,
            augmented_hypervolume=augmented,
            member_count=len(children),
            admitted_count=len(admitted),
            snapshot_sha256=self.snapshot.snapshot_sha256,
            definition_sha256=self.definition_hash,
        )

    def __call__(self, children: Sequence[AffineWaveRewardCandidate | None]) -> float:
        return self.record(children).reward


@dataclass(frozen=True, slots=True)
class AffineHypervolumeArchiveUtility:
    """Campaign archive-utility port for the affine indicator."""

    spec: AffineHypervolume2DSpec
    utility_id: str = field(init=False, default=_UTILITY_ID)
    utility_version: int = field(init=False, default=_UTILITY_VERSION)

    @property
    def definition_sha256(self) -> str:
        return self.spec.definition_sha256

    @staticmethod
    def _archive_points(archive: FrozenJsonObject) -> tuple[dict[str, float], ...]:
        value = thaw_json(archive)
        front = value.get("front_candidates")
        if type(front) is not list:
            raise ValueError("campaign archive omitted front_candidates")
        points: list[dict[str, float]] = []
        for candidate in front:
            if type(candidate) is not dict or type(candidate.get("objectives")) is not list:
                raise ValueError("campaign archive candidate omitted objectives")
            point: dict[str, float] = {}
            for row in candidate["objectives"]:
                if (
                    type(row) is not dict
                    or type(row.get("metric_id")) is not str
                    or type(row.get("value_hex")) is not str
                ):
                    raise ValueError("campaign archive objective row is invalid")
                point[row["metric_id"]] = float.fromhex(row["value_hex"])
            points.append(point)
        return tuple(points)

    def freeze(
        self,
        *,
        benchmark: FrozenJsonObject,
        generation: int,
        archive: FrozenJsonObject,
    ) -> ArchiveUtilitySnapshot:
        self.spec.__post_init__()
        snapshot = AffineHypervolumeSnapshot2D.create(
            spec=self.spec,
            archive_points=self._archive_points(archive),
        )
        return ArchiveUtilitySnapshot(
            utility_id=self.utility_id,
            utility_version=self.utility_version,
            definition_sha256=self.definition_sha256,
            generation=generation,
            benchmark_sha256=typed_json_sha256(benchmark),
            archive_sha256=typed_json_sha256(archive),
            snapshot_receipt=_object(snapshot.to_record()),
            scalar_utility_hex=snapshot.base_hypervolume.hex(),
        )

    def require_snapshot(
        self,
        value: ArchiveUtilitySnapshot,
    ) -> AffineHypervolumeSnapshot2D:
        if type(value) is not ArchiveUtilitySnapshot:
            raise TypeError("value must be an exact ArchiveUtilitySnapshot")
        value.__post_init__()
        if (
            value.utility_id != self.utility_id
            or value.utility_version != self.utility_version
            or value.definition_sha256 != self.definition_sha256
        ):
            raise ValueError("archive utility snapshot uses a foreign affine spec")
        record = thaw_json(value.snapshot_receipt)
        raw = record.get("raw_archive_points")
        if type(raw) is not list:
            raise ValueError("affine snapshot omitted raw archive points")
        points: list[dict[str, float]] = []
        for item in raw:
            if type(item) is not list:
                raise ValueError("affine raw point is not a list")
            point: dict[str, float] = {}
            for pair in item:
                if (
                    type(pair) is not list
                    or len(pair) != 2
                    or type(pair[0]) is not str
                    or type(pair[1]) is not str
                ):
                    raise ValueError("affine raw point item is invalid")
                point[pair[0]] = float.fromhex(pair[1])
            points.append(point)
        replayed = AffineHypervolumeSnapshot2D.create(
            spec=self.spec,
            archive_points=tuple(points),
        )
        if replayed.to_record() != record:
            raise ValueError("affine snapshot receipt does not replay exactly")
        return replayed


__all__ = [
    "AffineFrozenArchiveJointWaveReward",
    "AffineHypervolume2DSpec",
    "AffineHypervolumeArchiveUtility",
    "AffineHypervolumeSnapshot2D",
    "AffineHypervolumeWaveRewardRecord",
    "AffineObjectiveAxis",
    "AffineWaveRewardCandidate",
]
