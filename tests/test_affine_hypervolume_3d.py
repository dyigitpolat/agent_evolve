from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_evolve.domain.typed_json import freeze_json, thaw_json
from agent_evolve.policies.reward.affine_hypervolume import AffineObjectiveAxis
from agent_evolve.policies.reward.affine_hypervolume_3d import (
    AffineFrozenArchiveJointWaveReward3D,
    AffineHypervolume3DSpec,
    AffineHypervolumeArchiveUtility3D,
    AffineHypervolumeSnapshot3D,
    audit_affine_reference_envelope_3d,
    hypervolume_3d,
)


@dataclass(frozen=True)
class _Candidate:
    objective_map: dict[str, float]
    valid: bool = True
    operator_compliant: bool = True
    evidence_compliant: bool = True


def _spec(scale: float = 1.0) -> AffineHypervolume3DSpec:
    return AffineHypervolume3DSpec(
        axes=(
            AffineObjectiveAxis("energy", "min", 0.0, 0.3 * scale),
            AffineObjectiveAxis("latency", "min", 0.0, 0.6),
            AffineObjectiveAxis("area", "min", 0.0, 1.2e-7),
        ),
        reference_provenance="prospectively fixed engineering envelope",
    )


def test_single_box_and_union_are_exact() -> None:
    assert hypervolume_3d(((0.5, 0.5, 0.5),), (1.0, 1.0, 1.0)) == 0.125
    assert hypervolume_3d(
        ((0.5, 0.5, 0.5), (0.25, 0.75, 0.75)),
        (1.0, 1.0, 1.0),
    ) == 0.140625


def test_duplicates_dominance_and_out_of_reference_are_zero_measure() -> None:
    points = (
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        (0.7, 0.7, 0.7),
        (0.1, 1.0, 0.1),
    )
    assert hypervolume_3d(points, (1.0, 1.0, 1.0)) == 0.125


def test_snapshot_is_unit_invariant_and_replays_from_campaign_receipt() -> None:
    seeds = (
        {"energy": 0.03, "latency": 0.45, "area": 9.8e-8},
        {"energy": 0.20, "latency": 0.23, "area": 6.8e-8},
    )
    original = AffineHypervolumeSnapshot3D.create(
        spec=_spec(), archive_points=seeds
    )
    scaled = AffineHypervolumeSnapshot3D.create(
        spec=_spec(1_000.0),
        archive_points=tuple(
            {**point, "energy": point["energy"] * 1_000.0} for point in seeds
        ),
    )
    assert original.base_hypervolume == pytest.approx(scaled.base_hypervolume)

    utility = AffineHypervolumeArchiveUtility3D(_spec())
    archive = freeze_json(
        {
            "front_candidates": [
                {
                    "objectives": [
                        {"metric_id": key, "value_hex": value.hex()}
                        for key, value in point.items()
                    ]
                }
                for point in seeds
            ]
        }
    )
    receipt = utility.freeze(
        benchmark=freeze_json({"benchmark": "timeloop-test"}),
        generation=1,
        archive=archive,
    )
    replayed = utility.require_snapshot(receipt)
    assert replayed.to_record() == thaw_json(receipt.snapshot_receipt)


def test_joint_gain_can_capture_complementary_tradeoffs() -> None:
    snapshot = AffineHypervolumeSnapshot3D.create(
        spec=_spec(),
        archive_points=(
            {"energy": 0.10, "latency": 0.30, "area": 8.0e-8},
        ),
    )
    first = {"energy": 0.06, "latency": 0.40, "area": 7.0e-8}
    second = {"energy": 0.14, "latency": 0.20, "area": 6.0e-8}
    assert snapshot.joint_gain((first, second)) >= snapshot.marginal_gain(first)
    assert snapshot.joint_gain((first, second)) >= snapshot.marginal_gain(second)


def test_joint_wave_reward_3d_is_frozen_and_excludes_invalid_members() -> None:
    snapshot = AffineHypervolumeSnapshot3D.create(
        spec=_spec(),
        archive_points=(
            {"energy": 0.10, "latency": 0.30, "area": 8.0e-8},
        ),
    )
    child = _Candidate({"energy": 0.06, "latency": 0.40, "area": 7.0e-8})
    invalid = _Candidate(
        {"energy": 0.01, "latency": 0.10, "area": 1.0e-8},
        valid=False,
    )

    record = AffineFrozenArchiveJointWaveReward3D(snapshot).record((child, invalid))

    assert record.member_count == 2
    assert record.admitted_count == 1
    assert record.reward == snapshot.joint_gain((child.objective_map,))
    assert record.raw_oriented_gain == pytest.approx(
        record.reward * snapshot.spec.raw_volume_scale
    )
    assert record.snapshot_sha256 == snapshot.snapshot_sha256


def test_reference_envelope_audit_binds_headroom_and_fails_closed() -> None:
    inside = audit_affine_reference_envelope_3d(
        spec=_spec(),
        points=(
            {"energy": 0.20, "latency": 0.30, "area": 8.0e-8},
            {"energy": 0.10, "latency": 0.50, "area": 1.0e-7},
        ),
        evidence_identity_sha256="a" * 64,
    )
    assert inside["strictly_contains_all"] is True
    assert inside["violations"] == []
    assert inside["point_count"] == 2
    assert len(str(inside["audit_sha256"])) == 64

    outside = audit_affine_reference_envelope_3d(
        spec=_spec(),
        points=({"energy": 0.20, "latency": 0.60, "area": 8.0e-8},),
        evidence_identity_sha256="b" * 64,
    )
    assert outside["strictly_contains_all"] is False
    assert outside["violations"] == [
        {
            "point_index": 0,
            "outside_axes": [
                {
                    "metric_id": "latency",
                    "raw_value_hex": float(0.60).hex(),
                    "normalized_value_hex": float(1.0).hex(),
                }
            ],
        }
    ]
