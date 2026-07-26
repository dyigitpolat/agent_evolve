"""Exact scale-invariant fixed-reference hypervolume for three objectives.

This module is the three-dimensional counterpart of
``affine_hypervolume``.  It is deliberately workload-neutral: a composition
root supplies three metric identifiers, their optimization senses, and a
prospectively fixed ideal/reference box.  The implementation then exposes one
authenticated utility definition and replayable archive snapshots.

All axes are oriented to minimization and normalized so that the ideal is
zero and the reference is one.  Hypervolume is computed exactly (up to IEEE
754 arithmetic) by sweeping the first axis and reusing the exact 2-D union
area on each slab.  Duplicate, dominated, and out-of-reference points do not
change the measure.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real

from agent_evolve.application.evolution_campaign import ArchiveUtilitySnapshot
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.policies.reward.affine_hypervolume import (
    AffineHypervolumeWaveRewardRecord,
    AffineObjectiveAxis,
    AffineWaveRewardCandidate,
)
from agent_evolve.policies.reward.frozen_archive import hypervolume_2d


_DEFINITION_DOMAIN = b"agent-evolve:affine-hypervolume-3d-definition:v1\x00"
_SNAPSHOT_DOMAIN = b"agent-evolve:affine-hypervolume-3d-snapshot:v1\x00"
_REFERENCE_ENVELOPE_DOMAIN = (
    b"agent-evolve:affine-hypervolume-3d-reference-envelope:v1\x00"
)
_UTILITY_ID = "affine_fixed_reference_hypervolume_3d"
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
    if type(result) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("affine 3-D hypervolume record is not an object")
    return result


def hypervolume_3d(
    points: Sequence[tuple[Real, Real, Real]],
    reference_point: tuple[Real, Real, Real],
) -> float:
    """Return the union volume dominated by a minimization point set.

    A point contributes only when it strictly improves every reference
    coordinate.  The first-axis sweep partitions the union into disjoint
    slabs; within each slab :func:`hypervolume_2d` computes the active union
    area.  This is deterministic and exact for the small campaign archives
    AgentEvolve uses.
    """

    if type(reference_point) is not tuple or len(reference_point) != 3:
        raise TypeError("reference_point must be an exact three-item tuple")
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise TypeError("points must be a sequence of exact three-item tuples")
    reference = tuple(
        _finite(value, name=f"reference_point[{index}]")
        for index, value in enumerate(reference_point)
    )
    canonical: list[tuple[float, float, float]] = []
    for index, point in enumerate(points):
        if type(point) is not tuple or len(point) != 3:
            raise TypeError(f"points[{index}] must be an exact three-item tuple")
        canonical.append(
            tuple(
                _finite(value, name=f"points[{index}][{axis}]")
                for axis, value in enumerate(point)
            )
        )
    eligible = tuple(
        point
        for point in sorted(set(canonical))
        if all(point[index] < reference[index] for index in range(3))
    )
    if not eligible:
        return 0.0
    boundaries = (*sorted({point[0] for point in eligible}), reference[0])
    volume = 0.0
    for lower, upper in zip(boundaries, boundaries[1:]):
        active = tuple(
            (point[1], point[2]) for point in eligible if point[0] <= lower
        )
        volume += (upper - lower) * hypervolume_2d(
            active,
            (reference[1], reference[2]),
        )
    if volume < 0 or not math.isfinite(volume):  # pragma: no cover - sweep guard.
        raise RuntimeError("3-D hypervolume sweep produced an invalid volume")
    return volume


@dataclass(frozen=True, slots=True)
class AffineHypervolume3DSpec:
    """Complete immutable definition of a dimensionless 3-D indicator."""

    axes: tuple[AffineObjectiveAxis, AffineObjectiveAxis, AffineObjectiveAxis]
    reference_provenance: str

    def __post_init__(self) -> None:
        if type(self.axes) is not tuple or len(self.axes) != 3 or any(
            type(axis) is not AffineObjectiveAxis for axis in self.axes
        ):
            raise TypeError("axes must contain exactly three AffineObjectiveAxis values")
        for axis in self.axes:
            AffineObjectiveAxis.__post_init__(axis)
        if len({axis.metric_id for axis in self.axes}) != 3:
            raise ValueError("affine hypervolume axes must name distinct metrics")
        if (
            type(self.reference_provenance) is not str
            or not self.reference_provenance.strip()
            or self.reference_provenance != self.reference_provenance.strip()
        ):
            raise ValueError("reference_provenance must be canonical non-empty text")

    @property
    def metric_ids(self) -> tuple[str, str, str]:
        return tuple(axis.metric_id for axis in self.axes)  # type: ignore[return-value]

    @property
    def raw_volume_scale(self) -> float:
        return math.prod(axis.span for axis in self.axes)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "indicator": "exact_affine_normalized_three_objective_hypervolume",
            "axes": [axis.to_record() for axis in self.axes],
            "normalized_reference": ["0x1.0000000000000p+0"] * 3,
            "raw_volume_scale_hex": self.raw_volume_scale.hex(),
            "reference_provenance": self.reference_provenance,
            "out_of_reference_points": "zero_measure_clipped_by_hypervolume_sweep",
            "algorithm": "first_axis_exact_slab_sweep_with_exact_2d_union",
        }

    @property
    def definition_sha256(self) -> str:
        return hashlib.sha256(
            _DEFINITION_DOMAIN + _canonical_json(self.to_record())
        ).hexdigest()

    def normalize(self, point: Mapping[str, Real]) -> tuple[float, float, float]:
        if not isinstance(point, Mapping):
            raise TypeError("objective point must be a mapping")
        if set(point) != set(self.metric_ids):
            raise ValueError("objective point differs from the affine metric set")
        return tuple(
            axis.normalize(point[axis.metric_id]) for axis in self.axes
        )  # type: ignore[return-value]


def audit_affine_reference_envelope_3d(
    *,
    spec: AffineHypervolume3DSpec,
    points: Sequence[Mapping[str, Real]],
    evidence_identity_sha256: str,
) -> dict[str, object]:
    """Authenticate whether a fixed reference strictly contains evidence.

    Filesystem provenance remains a workload-adapter responsibility.  This
    workload-neutral boundary binds the adapter's evidence identity to the
    exact objective values, reports per-axis headroom, and makes a failed
    containment gate explicit before an optimization campaign starts.
    """

    if type(spec) is not AffineHypervolume3DSpec:
        raise TypeError("spec must be an exact AffineHypervolume3DSpec")
    spec.__post_init__()
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise TypeError("points must be a sequence of objective mappings")
    if not points:
        raise ValueError("reference-envelope evidence must not be empty")
    if (
        type(evidence_identity_sha256) is not str
        or len(evidence_identity_sha256) != 64
        or any(value not in "0123456789abcdef" for value in evidence_identity_sha256)
    ):
        raise ValueError("evidence_identity_sha256 must be lowercase SHA-256")

    normalized_points: list[tuple[float, float, float]] = []
    raw_records: list[list[list[str]]] = []
    violations: list[dict[str, object]] = []
    for point_index, point in enumerate(points):
        normalized = spec.normalize(point)
        normalized_points.append(normalized)
        raw_records.append(
            [
                [axis.metric_id, float(point[axis.metric_id]).hex()]
                for axis in spec.axes
            ]
        )
        outside = [
            {
                "metric_id": axis.metric_id,
                "raw_value_hex": float(point[axis.metric_id]).hex(),
                "normalized_value_hex": normalized[axis_index].hex(),
            }
            for axis_index, axis in enumerate(spec.axes)
            if normalized[axis_index] >= 1.0
        ]
        if outside:
            violations.append(
                {
                    "point_index": point_index,
                    "outside_axes": outside,
                }
            )

    componentwise = []
    for axis_index, axis in enumerate(spec.axes):
        worst_index = max(
            range(len(normalized_points)),
            key=lambda point_index: normalized_points[point_index][axis_index],
        )
        worst = normalized_points[worst_index][axis_index]
        componentwise.append(
            {
                "metric_id": axis.metric_id,
                "evidence_point_index": worst_index,
                "raw_value_hex": float(points[worst_index][axis.metric_id]).hex(),
                "normalized_value_hex": worst.hex(),
                "normalized_margin_to_reference_hex": (1.0 - worst).hex(),
            }
        )

    record: dict[str, object] = {
        "schema_version": 1,
        "evidence_identity_sha256": evidence_identity_sha256,
        "utility_definition_sha256": spec.definition_sha256,
        "point_count": len(points),
        "metric_ids": list(spec.metric_ids),
        "points": raw_records,
        "componentwise_worst": componentwise,
        "strictly_contains_all": not violations,
        "violations": violations,
    }
    record["audit_sha256"] = hashlib.sha256(
        _REFERENCE_ENVELOPE_DOMAIN + _canonical_json(record)
    ).hexdigest()
    return record


RawObjectivePoint3D = tuple[
    tuple[str, float], tuple[str, float], tuple[str, float]
]


def _raw_point(
    spec: AffineHypervolume3DSpec,
    value: Mapping[str, Real],
) -> RawObjectivePoint3D:
    spec.normalize(value)
    return tuple(
        (axis.metric_id, float(value[axis.metric_id])) for axis in spec.axes
    )  # type: ignore[return-value]


def _point_record(point: RawObjectivePoint3D) -> list[list[str]]:
    return [[metric_id, value.hex()] for metric_id, value in point]


@dataclass(frozen=True, slots=True)
class AffineHypervolumeSnapshot3D:
    """One replayable archive cutoff in raw and normalized coordinates."""

    spec: AffineHypervolume3DSpec
    raw_archive_points: tuple[RawObjectivePoint3D, ...]
    normalized_archive_points: tuple[tuple[float, float, float], ...] = field(
        init=False
    )
    base_hypervolume: float = field(init=False)
    raw_oriented_base_hypervolume: float = field(init=False)
    snapshot_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.spec) is not AffineHypervolume3DSpec:
            raise TypeError("spec must be an exact AffineHypervolume3DSpec")
        self.spec.__post_init__()
        if type(self.raw_archive_points) is not tuple:
            raise TypeError("raw_archive_points must be an exact tuple")
        expected_names = self.spec.metric_ids
        for point in self.raw_archive_points:
            if (
                type(point) is not tuple
                or len(point) != 3
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
        base = hypervolume_3d(normalized, (1.0, 1.0, 1.0))
        object.__setattr__(self, "normalized_archive_points", normalized)
        object.__setattr__(self, "base_hypervolume", base)
        object.__setattr__(
            self,
            "raw_oriented_base_hypervolume",
            base * self.spec.raw_volume_scale,
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
        spec: AffineHypervolume3DSpec,
        archive_points: Sequence[Mapping[str, Real]],
    ) -> "AffineHypervolumeSnapshot3D":
        if isinstance(archive_points, (str, bytes)) or not isinstance(
            archive_points, Sequence
        ):
            raise TypeError("archive_points must be a sequence")
        points = tuple(
            sorted(
                {_raw_point(spec, value) for value in archive_points},
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
        return hypervolume_3d(
            (*self.normalized_archive_points, *normalized),
            (1.0, 1.0, 1.0),
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
                [value.hex() for value in point]
                for point in self.normalized_archive_points
            ],
            "base_hypervolume_hex": self.base_hypervolume.hex(),
            "raw_oriented_base_hypervolume_hex": (
                self.raw_oriented_base_hypervolume.hex()
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "snapshot_sha256": self.snapshot_sha256}


@dataclass(frozen=True, slots=True)
class AffineFrozenArchiveJointWaveReward3D:
    """Dimensionless joint gain for one fixed-cardinality 3-D wave."""

    snapshot: AffineHypervolumeSnapshot3D

    def __post_init__(self) -> None:
        if type(self.snapshot) is not AffineHypervolumeSnapshot3D:
            raise TypeError("snapshot must be exact AffineHypervolumeSnapshot3D")
        self.snapshot.__post_init__()

    @property
    def definition_hash(self) -> str:
        return self.snapshot.spec.definition_sha256

    def record(
        self,
        children: Sequence[AffineWaveRewardCandidate | None],
    ) -> AffineHypervolumeWaveRewardRecord:
        self.__post_init__()
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
            raw_oriented_gain=gain * self.snapshot.spec.raw_volume_scale,
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
class AffineHypervolumeArchiveUtility3D:
    """Campaign archive-utility port for one affine 3-D indicator."""

    spec: AffineHypervolume3DSpec
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
        snapshot = AffineHypervolumeSnapshot3D.create(
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
    ) -> AffineHypervolumeSnapshot3D:
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
        replayed = AffineHypervolumeSnapshot3D.create(
            spec=self.spec,
            archive_points=tuple(points),
        )
        if replayed.to_record() != record:
            raise ValueError("affine snapshot receipt does not replay exactly")
        return replayed


__all__ = [
    "AffineFrozenArchiveJointWaveReward3D",
    "AffineHypervolume3DSpec",
    "AffineHypervolumeArchiveUtility3D",
    "AffineHypervolumeSnapshot3D",
    "audit_affine_reference_envelope_3d",
    "hypervolume_3d",
]
