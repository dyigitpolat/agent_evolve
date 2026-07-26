from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_evolve.domain.typed_json import freeze_json, thaw_json
from agent_evolve.policies.reward.affine_hypervolume import (
    AffineFrozenArchiveJointWaveReward,
    AffineHypervolume2DSpec,
    AffineHypervolumeArchiveUtility,
    AffineHypervolumeSnapshot2D,
    AffineObjectiveAxis,
)


def _heat_spec(scale: float = 1.0) -> AffineHypervolume2DSpec:
    return AffineHypervolume2DSpec(
        axes=(
            AffineObjectiveAxis(
                "thermal_term",
                "min",
                0.0,
                0.00075 * scale,
            ),
            AffineObjectiveAxis(
                "material_fraction",
                "min",
                0.30,
                0.61,
            ),
        ),
        reference_provenance="developmental fixed pre-outcome reference",
    )


def _seeds(scale: float = 1.0) -> tuple[dict[str, float], ...]:
    return (
        {
            "thermal_term": 0.0002982181533116508 * scale,
            "material_fraction": 0.44999999999999996,
        },
        {
            "thermal_term": 0.0003736918189897142 * scale,
            "material_fraction": 0.37999999999999995,
        },
    )


def test_heat_qualified_seed_snapshot_matches_preregistered_value() -> None:
    snapshot = AffineHypervolumeSnapshot2D.create(
        spec=_heat_spec(),
        archive_points=_seeds(),
    )
    assert snapshot.base_hypervolume.hex() == "0x1.b261aca41a142p-2"
    assert snapshot.raw_oriented_base_hypervolume == pytest.approx(
        snapshot.base_hypervolume * 0.00075 * 0.31
    )


def test_integer_bounds_have_canonical_float_serialization() -> None:
    axis = AffineObjectiveAxis("score", "max", 10, 0)
    assert axis.to_record()["ideal_hex"] == float(10).hex()
    assert axis.to_record()["reference_hex"] == float(0).hex()
    assert axis.normalize(5) == 0.5


def test_affine_indicator_is_invariant_to_positive_unit_rescaling() -> None:
    original = AffineHypervolumeSnapshot2D.create(
        spec=_heat_spec(),
        archive_points=_seeds(),
    )
    rescaled = AffineHypervolumeSnapshot2D.create(
        spec=_heat_spec(scale=1_000_000.0),
        archive_points=_seeds(scale=1_000_000.0),
    )
    child = {"thermal_term": 0.00025, "material_fraction": 0.41}
    scaled_child = {
        "thermal_term": child["thermal_term"] * 1_000_000.0,
        "material_fraction": child["material_fraction"],
    }
    assert original.base_hypervolume == pytest.approx(rescaled.base_hypervolume)
    assert original.joint_gain((child,)) == pytest.approx(
        rescaled.joint_gain((scaled_child,))
    )


@dataclass(frozen=True)
class _Candidate:
    objective_map: dict[str, float]
    valid: bool = True
    operator_compliant: bool = True
    evidence_compliant: bool = True


def test_joint_wave_reward_uses_dimensionless_gain_and_excludes_invalid() -> None:
    snapshot = AffineHypervolumeSnapshot2D.create(
        spec=_heat_spec(),
        archive_points=_seeds(),
    )
    reward = AffineFrozenArchiveJointWaveReward(snapshot)
    child = _Candidate({"thermal_term": 0.00025, "material_fraction": 0.41})
    invalid = _Candidate(
        {"thermal_term": 0.0001, "material_fraction": 0.31},
        valid=False,
    )
    record = reward.record((child, invalid))
    assert record.admitted_count == 1
    assert record.member_count == 2
    assert record.reward == snapshot.joint_gain((child.objective_map,))
    assert record.raw_oriented_gain == pytest.approx(
        record.reward * snapshot.spec.raw_area_scale
    )


def test_campaign_archive_snapshot_replays_exactly() -> None:
    spec = _heat_spec()
    utility = AffineHypervolumeArchiveUtility(spec)
    archive = freeze_json(
        {
            "front_candidates": [
                {
                    "objectives": [
                        {
                            "metric_id": metric,
                            "value_hex": value.hex(),
                        }
                        for metric, value in point.items()
                    ]
                }
                for point in _seeds()
            ]
        }
    )
    benchmark = freeze_json({"benchmark": "heat-test"})
    frozen = utility.freeze(
        benchmark=benchmark,
        generation=1,
        archive=archive,
    )
    replayed = utility.require_snapshot(frozen)
    assert replayed.base_hypervolume.hex() == "0x1.b261aca41a142p-2"
    assert replayed.to_record() == thaw_json(frozen.snapshot_receipt)
