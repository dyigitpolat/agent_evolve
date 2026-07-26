"""Provider-free checks for balanced complete-subset memory blocks."""

from __future__ import annotations

import itertools
import math
from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from agent_evolve.domain.ids import InsightId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.policies.memory import (
    BALANCED_SUBSET_BLOCK_POLICY_DEFINITION_SHA256,
    BalancedSubsetBlockPlanner,
    CausalSearchScorePolicy,
    StableMemoryAssignmentUnit,
    canonical_memory_subset_catalog,
)
from agent_evolve.policies.memory.randomized_subset import InsightSelectionMode


CONTEXT = "a" * 64
STRATUM = "b" * 64
A = InsightRef(InsightId("insight_a"), 1)
B = InsightRef(InsightId("insight_b"), 1)
C = InsightRef(InsightId("insight_c"), 1)
D = InsightRef(InsightId("insight_d"), 1)


def _snapshot(references=(A, B, C), *, offset: float = 0.0):
    return CausalSearchScorePolicy().genesis(
        exact_context_hash=CONTEXT,
        estimand_stratum_hash=STRATUM,
        priors={
            reference: float(index) + offset
            for index, reference in enumerate(references)
        },
    )


def _units(count: int) -> tuple[StableMemoryAssignmentUnit, ...]:
    return tuple(
        StableMemoryAssignmentUnit(
            unit_key=f"unit_slot_{index}",
            generation=index + 1,
            lane_id="lane_primary",
        )
        for index in range(count)
    )


def _two_lane_units() -> tuple[StableMemoryAssignmentUnit, ...]:
    # Lane-major order gives each lane one independently permuted full block.
    return tuple(
        StableMemoryAssignmentUnit(
            unit_key=f"unit_{lane}_generation_{generation}",
            generation=generation,
            lane_id=f"lane_{lane}",
        )
        for lane in ("anchor", "partner")
        for generation in (1, 2, 3)
    )


def test_two_complete_three_choose_two_blocks_have_exact_balanced_support() -> None:
    snapshot = _snapshot(references=(C, A, B))
    units = _two_lane_units()
    planner = BalancedSubsetBlockPlanner()
    plan = planner.plan(
        snapshot=snapshot,
        ordered_units=units,
        subset_size=2,
        full_block_permutation_ranks=(0, math.factorial(3) - 1),
    )

    assert plan.catalog == tuple(itertools.combinations((A, B, C), 2))
    assert canonical_memory_subset_catalog(snapshot, 2) == plan.catalog
    assert [value.subset_rank for value in plan.assignments] == [0, 1, 2, 2, 1, 0]
    assert plan.full_block_count == 2
    assert plan.remainder_size == 0
    assert [(value.treated_count, value.control_count) for value in plan.support] == [
        (4, 2),
        (4, 2),
        (4, 2),
    ]
    assert all(value.has_overlap for value in plan.support)
    assert all(
        value.decision.mode is InsightSelectionMode.EXPLORE_UNIFORM
        and value.decision.selected_subset_probability == Fraction(1, 3)
        and value.decision.selected == plan.catalog[value.subset_rank]
        for value in plan.assignments
    )

    anchor_second = plan.assignment_for(2, "lane_anchor")
    assert anchor_second.unit.unit_key == "unit_anchor_generation_2"
    assert plan.assignment_for("unit_anchor_generation_2") == anchor_second
    assert plan.assignment_for(unit_key="unit_anchor_generation_2") == anchor_second
    assert plan.plan_sha256 == plan.receipt_sha256
    assert plan.to_record()["receipt_sha256"] == plan.receipt_sha256
    assert (
        plan.to_record()["policy"]["policy_definition_sha256"]
        == BALANCED_SUBSET_BLOCK_POLICY_DEFINITION_SHA256
    )
    replayed = planner.replay(plan, snapshot=snapshot, ordered_units=units)
    assert replayed.to_record() == plan.to_record()

    with pytest.raises(FrozenInstanceError):
        plan.subset_size = 1


def test_full_block_ranks_are_exact_external_permutation_coordinates() -> None:
    snapshot = _snapshot()
    planner = BalancedSubsetBlockPlanner()
    forward = planner.plan(
        snapshot=snapshot,
        ordered_units=_units(3),
        subset_size=2,
        full_block_permutation_ranks=(0,),
    )
    rank_one = planner.plan(
        snapshot=snapshot,
        ordered_units=_units(3),
        subset_size=2,
        full_block_permutation_ranks=(1,),
    )

    assert tuple(value.subset_rank for value in forward.assignments) == (0, 1, 2)
    assert tuple(value.subset_rank for value in rank_one.assignments) == (0, 2, 1)
    assert rank_one.receipt_sha256 != forward.receipt_sha256
    assert sorted(value.subset_rank for value in rank_one.assignments) == [0, 1, 2]

    for invalid_rank in (-1, math.factorial(3), True):
        with pytest.raises(ValueError, match=r"full_block_permutation_ranks\[0\]"):
            planner.plan(
                snapshot=snapshot,
                ordered_units=_units(3),
                subset_size=2,
                full_block_permutation_ranks=(invalid_rank,),  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="exactly one rank"):
        planner.plan(
            snapshot=snapshot,
            ordered_units=_units(3),
            subset_size=2,
            full_block_permutation_ranks=(),
        )


def test_remainder_uses_exact_no_replacement_selection_then_permutation() -> None:
    snapshot = _snapshot(references=(A, B, C, D))
    catalog_size = math.comb(4, 2)
    planner = BalancedSubsetBlockPlanner()
    plan = planner.plan(
        snapshot=snapshot,
        ordered_units=_units(catalog_size + 2),
        subset_size=2,
        full_block_permutation_ranks=(0,),
        # The final combination rank chooses catalog ranks (4, 5), then rank
        # one reverses their presentation order.
        remainder_selection_rank=math.comb(catalog_size, 2) - 1,
        remainder_permutation_rank=1,
    )

    assert plan.full_block_count == 1
    assert plan.remainder_size == 2
    assert tuple(value.subset_rank for value in plan.assignments[:catalog_size]) == (
        0,
        1,
        2,
        3,
        4,
        5,
    )
    assert tuple(value.subset_rank for value in plan.assignments[-2:]) == (5, 4)
    assert len({value.subset_rank for value in plan.assignments[-2:]}) == 2
    assert all(value.is_remainder for value in plan.assignments[-2:])
    assert all(value.has_overlap for value in plan.support)

    with pytest.raises(ValueError, match="requires exact remainder"):
        planner.plan(
            snapshot=snapshot,
            ordered_units=_units(catalog_size + 2),
            subset_size=2,
            full_block_permutation_ranks=(0,),
        )
    with pytest.raises(ValueError, match="remainder_selection_rank"):
        planner.plan(
            snapshot=snapshot,
            ordered_units=_units(catalog_size + 2),
            subset_size=2,
            full_block_permutation_ranks=(0,),
            remainder_selection_rank=math.comb(catalog_size, 2),
            remainder_permutation_rank=0,
        )
    with pytest.raises(ValueError, match="remainder_permutation_rank"):
        planner.plan(
            snapshot=snapshot,
            ordered_units=_units(catalog_size + 2),
            subset_size=2,
            full_block_permutation_ranks=(0,),
            remainder_selection_rank=0,
            remainder_permutation_rank=math.factorial(2),
        )


def test_preflight_rejects_any_plan_without_both_arms_for_every_insight() -> None:
    snapshot = _snapshot()
    planner = BalancedSubsetBlockPlanner()

    # Catalog ranks 0 and 1 are (A,B) and (A,C), leaving A with no control.
    with pytest.raises(
        ValueError,
        match=r"missing overlap: insight_a@1\(treated=2,control=0\)",
    ):
        planner.plan(
            snapshot=snapshot,
            ordered_units=_units(2),
            subset_size=2,
            full_block_permutation_ranks=(),
            remainder_selection_rank=0,
            remainder_permutation_rank=0,
        )

    for invalid_size in (0, len(snapshot.entries), True):
        with pytest.raises((TypeError, ValueError), match="subset_size"):
            planner.plan(
                snapshot=snapshot,
                ordered_units=_units(3),
                subset_size=invalid_size,  # type: ignore[arg-type]
                full_block_permutation_ranks=(0,),
            )


def test_stable_units_and_receipt_replay_fail_closed_on_basis_changes() -> None:
    snapshot = _snapshot()
    units = _units(3)
    planner = BalancedSubsetBlockPlanner()
    plan = planner.plan(
        snapshot=snapshot,
        ordered_units=units,
        subset_size=2,
        full_block_permutation_ranks=(2,),
    )
    duplicate_key = (
        units[0],
        StableMemoryAssignmentUnit(
            unit_key=units[0].unit_key,
            generation=10,
            lane_id="lane_other",
        ),
        units[2],
    )
    with pytest.raises(ValueError, match="stable unit_key"):
        planner.plan(
            snapshot=snapshot,
            ordered_units=duplicate_key,
            subset_size=2,
            full_block_permutation_ranks=(2,),
        )

    with pytest.raises(ValueError, match="does not match the receipt"):
        planner.replay(plan, snapshot=snapshot, ordered_units=tuple(reversed(units)))
    changed_snapshot = _snapshot(offset=0.5)
    with pytest.raises(ValueError, match="does not match the receipt"):
        planner.replay(plan, snapshot=changed_snapshot, ordered_units=units)
    with pytest.raises(KeyError, match="no balanced memory assignment"):
        plan.assignment_for(unit_key="unit_missing")
    with pytest.raises(ValueError, match="cannot be combined"):
        plan.assignment_for(1, "lane_primary", unit_key="unit_slot_0")
