from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.reward.frozen_archive import (
    FrozenArchiveMarginalHypervolumeReward,
    FrozenArchiveSnapshot2D,
    hypervolume_2d,
)


MIN_OBJECTIVES = (
    ObjectiveSpec("area", "min"),
    ObjectiveSpec("depth", "min"),
)


@dataclass(frozen=True)
class _Candidate:
    objective_map: dict[str, float]
    valid: bool = True
    operator_compliant: bool = True


def _snapshot(*, reverse: bool = False) -> FrozenArchiveSnapshot2D:
    points = [{"area": 5, "depth": 8}, {"area": 8, "depth": 5}]
    if reverse:
        points.reverse()
    return FrozenArchiveSnapshot2D.create(
        objectives=MIN_OBJECTIVES,
        reference_point={"area": 10, "depth": 10},
        archive_points=points,
    )


def test_exact_two_dimensional_sweep_removes_dominated_duplicates_and_outside_points() -> None:
    assert hypervolume_2d(
        [(5, 8), (8, 5), (6, 9), (5, 8), (11, 1)],
        (10, 10),
    ) == pytest.approx(16.0)
    assert hypervolume_2d([], (10, 10)) == 0.0


@pytest.mark.parametrize(
    ("points", "reference", "message"),
    [
        (((1.0, 2.0, 3.0),), (8.0, 8.0), r"points\[0\]"),
        (((1.0, 2.0),), (8.0,), "reference_point"),
        (([1.0, 2.0],), (8.0, 8.0), r"points\[0\]"),
    ],
)
def test_hypervolume_2d_rejects_malformed_points_instead_of_skipping(
    points, reference, message
) -> None:
    with pytest.raises(TypeError, match=message):
        hypervolume_2d(points, reference)


def test_frozen_snapshot_and_reward_are_order_invariant_and_archive_relative() -> None:
    first = _snapshot()
    second = _snapshot(reverse=True)
    assert first.snapshot_hash == second.snapshot_hash
    assert first.definition_hash == second.definition_hash
    assert first.base_hypervolume == pytest.approx(16.0)

    policy = FrozenArchiveMarginalHypervolumeReward(first)
    record = policy.record(_Candidate({"area": 4.0, "depth": 7.0}))
    assert record.status == "credited"
    assert record.augmented_hypervolume == pytest.approx(22.0)
    assert record.marginal_hypervolume_gain == pytest.approx(6.0)
    assert record.reward == pytest.approx(0.375)
    assert record.reward_definition_hash == first.definition_hash
    assert record.archive_snapshot_hash == first.snapshot_hash
    assert policy(
        _Candidate({"area": 4.0, "depth": 7.0}),
        (),
        MIN_OBJECTIVES,
    ) == pytest.approx(0.375)

    dominated = policy.record(_Candidate({"area": 9.0, "depth": 9.0}))
    assert dominated.status == "credited"
    assert dominated.marginal_hypervolume_gain == 0.0
    assert dominated.reward == 0.0


def test_definition_hash_excludes_archive_cutoff_but_snapshot_hash_binds_it() -> None:
    first = _snapshot()
    later = FrozenArchiveSnapshot2D.create(
        objectives=MIN_OBJECTIVES,
        reference_point={"area": 10, "depth": 10},
        archive_points=[
            {"area": 5, "depth": 8},
            {"area": 8, "depth": 5},
            {"area": 4, "depth": 7},
        ],
    )
    assert first.definition_hash == later.definition_hash
    assert first.snapshot_hash != later.snapshot_hash


def test_maximization_direction_and_failure_gates_are_explicit() -> None:
    objectives = (
        ObjectiveSpec("quality", "max"),
        ObjectiveSpec("cost", "min"),
    )
    snapshot = FrozenArchiveSnapshot2D.create(
        objectives=objectives,
        reference_point={"quality": 0, "cost": 10},
        archive_points=[{"quality": 5, "cost": 8}],
    )
    policy = FrozenArchiveMarginalHypervolumeReward(snapshot)
    assert snapshot.base_hypervolume == pytest.approx(10.0)
    improved = policy.record(_Candidate({"quality": 8.0, "cost": 9.0}))
    assert improved.marginal_hypervolume_gain == pytest.approx(3.0)
    assert improved.reward == pytest.approx(0.3)

    invalid = policy.record(
        _Candidate({}, valid=False, operator_compliant=True)
    )
    noncompliant = policy.record(
        _Candidate(
            {"quality": 9.0, "cost": 1.0},
            valid=True,
            operator_compliant=False,
        )
    )
    assert invalid.status == "invalid_candidate" and invalid.reward == -1.0
    assert noncompliant.status == "operator_noncompliant"
    assert noncompliant.reward == -1.0
    with pytest.raises(ValueError, match="differ"):
        policy(
            _Candidate({"quality": 8.0, "cost": 9.0}),
            (),
            tuple(reversed(objectives)),
        )


def test_snapshot_rejects_wrong_dimensions_keys_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        FrozenArchiveSnapshot2D.create(
            objectives=(ObjectiveSpec("only", "min"),),
            reference_point={"only": 1},
            archive_points=[],
        )
    with pytest.raises(ValueError, match="objective keys differ"):
        FrozenArchiveSnapshot2D.create(
            objectives=MIN_OBJECTIVES,
            reference_point={"area": 10, "wrong": 10},
            archive_points=[],
        )
    with pytest.raises(ValueError, match="finite"):
        FrozenArchiveSnapshot2D.create(
            objectives=MIN_OBJECTIVES,
            reference_point={"area": 10, "depth": float("nan")},
            archive_points=[],
        )
