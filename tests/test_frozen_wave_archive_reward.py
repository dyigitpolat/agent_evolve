from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.reward import (
    FrozenArchiveJointWaveHypervolumeReward,
    FrozenArchiveWaveSnapshot2D,
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
    evidence_compliant: bool = True


def _snapshot(*, reverse: bool = False) -> FrozenArchiveWaveSnapshot2D:
    points = [{"area": 5, "depth": 8}, {"area": 8, "depth": 5}]
    if reverse:
        points.reverse()
    return FrozenArchiveWaveSnapshot2D.create(
        objectives=MIN_OBJECTIVES,
        reference_point={"area": 10, "depth": 10},
        archive_points=points,
    )


def test_joint_wave_credit_is_order_invariant_and_not_sum_of_individual_gains() -> None:
    first = _snapshot()
    second = _snapshot(reverse=True)
    assert first.snapshot_hash == second.snapshot_hash
    assert first.definition_hash == second.definition_hash
    assert first.base_hypervolume == pytest.approx(16.0)

    policy = FrozenArchiveJointWaveHypervolumeReward(first)
    children = (
        _Candidate({"area": 4.0, "depth": 7.0}),
        _Candidate({"area": 7.0, "depth": 4.0}),
    )
    record = policy.record(children)
    reversed_record = policy.record(tuple(reversed(children)))

    assert record.status == "credited"
    assert record.augmented_hypervolume == pytest.approx(27.0)
    assert record.joint_hypervolume_gain == pytest.approx(11.0)
    assert record.reward == pytest.approx(11.0 / 16.0)
    assert reversed_record.reward == record.reward
    assert reversed_record.joint_hypervolume_gain == record.joint_hypervolume_gain
    # Each child alone adds six units, but their rectangles overlap by one.
    assert record.joint_hypervolume_gain != pytest.approx(12.0)
    assert policy(children) == record.reward
    assert record.reward_definition_hash == first.definition_hash
    assert record.archive_snapshot_hash == first.snapshot_hash


def test_inadmissible_members_keep_slot_accounting_but_do_not_enter_archive() -> None:
    policy = FrozenArchiveJointWaveHypervolumeReward(_snapshot())
    record = policy.record(
        (
            _Candidate({"area": 4.0, "depth": 7.0}),
            _Candidate({}, valid=False),
            _Candidate(
                {"area": 1.0, "depth": 1.0},
                operator_compliant=False,
            ),
            _Candidate(
                {"area": 1.0, "depth": 1.0},
                evidence_compliant=False,
            ),
            None,
        )
    )

    assert record.member_count == 5
    assert record.admitted_count == 1
    assert record.invalid_count == 1
    assert record.operator_noncompliant_count == 1
    assert record.evidence_noncompliant_count == 1
    assert record.missing_count == 1
    assert record.joint_hypervolume_gain == pytest.approx(6.0)
    assert record.reward == pytest.approx(0.375)
    assert record.to_record()["member_count"] == 5

    empty = policy.record((None, _Candidate({}, valid=False)))
    assert empty.status == "no_admissible_candidates"
    assert empty.reward == 0.0
    assert empty.augmented_hypervolume == empty.base_hypervolume


def test_joint_wave_handles_mixed_objective_directions_and_zero_gain() -> None:
    objectives = (
        ObjectiveSpec("quality", "max"),
        ObjectiveSpec("cost", "min"),
    )
    snapshot = FrozenArchiveWaveSnapshot2D.create(
        objectives=objectives,
        reference_point={"quality": 0, "cost": 10},
        archive_points=[{"quality": 5, "cost": 8}],
    )
    policy = FrozenArchiveJointWaveHypervolumeReward(snapshot)
    record = policy.record(
        (
            _Candidate({"quality": 8.0, "cost": 9.0}),
            _Candidate({"quality": 4.0, "cost": 9.0}),
        )
    )
    assert snapshot.base_hypervolume == pytest.approx(10.0)
    assert record.joint_hypervolume_gain == pytest.approx(3.0)
    assert record.reward == pytest.approx(0.3)

    dominated = policy.record((_Candidate({"quality": 1.0, "cost": 9.0}),))
    assert dominated.status == "credited"
    assert dominated.reward == 0.0


def test_snapshot_and_wave_fail_closed_on_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        FrozenArchiveWaveSnapshot2D.create(
            objectives=(ObjectiveSpec("only", "min"),),
            reference_point={"only": 1},
            archive_points=[],
        )
    with pytest.raises(ValueError, match="objective keys differ"):
        FrozenArchiveWaveSnapshot2D.create(
            objectives=MIN_OBJECTIVES,
            reference_point={"area": 10, "wrong": 10},
            archive_points=[],
        )
    with pytest.raises(ValueError, match="finite"):
        FrozenArchiveWaveSnapshot2D.create(
            objectives=MIN_OBJECTIVES,
            reference_point={"area": 10, "depth": float("nan")},
            archive_points=[],
        )
    policy = FrozenArchiveJointWaveHypervolumeReward(_snapshot())
    with pytest.raises(ValueError, match="at least one"):
        policy.record(())
    with pytest.raises(ValueError, match="objective keys differ"):
        policy.record((_Candidate({"area": 1.0, "wrong": 2.0}),))
