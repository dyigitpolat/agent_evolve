"""Pareto dominance and minimax-rank selection."""

from __future__ import annotations

import math

import pytest

from agent_evolve.core.problem import ObjectiveSpec, ProblemContractError
from agent_evolve.core.results import (
    Candidate,
    compute_pareto_front,
    dominates,
    select_minimax_rank,
    sort_by_minimax_rank,
)

OBJS = [ObjectiveSpec("value", "max"), ObjectiveSpec("cost", "min")]


def _c(value, cost):
    return Candidate(configuration={"v": value}, objectives={"value": value, "cost": cost})


def test_dominates_basic():
    assert dominates({"value": 10, "cost": 1}, {"value": 5, "cost": 2}, OBJS)
    assert not dominates({"value": 5, "cost": 2}, {"value": 10, "cost": 1}, OBJS)
    # equal vectors do not dominate
    assert not dominates({"value": 5, "cost": 2}, {"value": 5, "cost": 2}, OBJS)


def test_pareto_front_filters_dominated():
    cands = [_c(10, 5), _c(8, 2), _c(5, 5), _c(9, 9)]
    front = compute_pareto_front(cands, OBJS)
    fronts = {(c.objectives["value"], c.objectives["cost"]) for c in front}
    assert (10, 5) in fronts and (8, 2) in fronts
    assert (5, 5) not in fronts  # dominated by (8,2) and (10,5)
    assert (9, 9) not in fronts  # dominated by (10,5)


def test_minimax_rank_prefers_balanced():
    # (8,2) is rank1 on cost and rank2 on value -> bottleneck 2
    # (10,5) is rank1 on value and rank2 on cost -> bottleneck 2 (tie) ; sum equal
    cands = [_c(10, 5), _c(8, 2)]
    best = select_minimax_rank(cands, OBJS)
    assert best is not None
    ordered = sort_by_minimax_rank(cands, OBJS)
    assert ordered[0] is best


def test_minimax_rank_empty():
    assert select_minimax_rank([], OBJS) is None


@pytest.mark.parametrize(
    "bad",
    [
        {"value": 1.0},
        {"value": 1.0, "cost": math.nan},
        {"value": 1.0, "cost": math.inf},
        {"value": True, "cost": 1.0},
    ],
)
def test_pareto_rejects_incomplete_or_nonfinite_vectors(bad):
    with pytest.raises(ProblemContractError):
        dominates(bad, {"value": 0.0, "cost": 2.0}, OBJS)


def test_minimax_rejects_missing_objective():
    broken = Candidate(configuration={}, objectives={"value": 1.0})
    with pytest.raises(ProblemContractError):
        select_minimax_rank([broken, _c(2, 1)], OBJS)


def test_pareto_front_collapses_exact_duplicates():
    # A population re-visiting the same configuration across generations must
    # not put it on the front once per visit (the quickstart's 59-rows /
    # 4-distinct-configs bug). First occurrence survives, order preserved.
    a, b = _c(10, 5), _c(8, 2)
    front = compute_pareto_front([a, _c(10, 5), b, _c(10, 5), _c(8, 2)], OBJS)
    assert len(front) == 2
    assert front[0] is a and front[1] is b


def test_pareto_front_keeps_noisy_reevaluations_distinct():
    # Same configuration, different measured objectives: two genuine
    # measurements. The library must not silently pick one.
    noisy_a = Candidate(configuration={"v": 1}, objectives={"value": 10.0, "cost": 5.0})
    noisy_b = Candidate(configuration={"v": 1}, objectives={"value": 10.5, "cost": 5.5})
    front = compute_pareto_front([noisy_a, noisy_b], OBJS)
    assert len(front) == 2


def test_pareto_front_duplicate_of_dominated_still_filtered():
    dup_dominated = [_c(10, 5), _c(5, 5), _c(5, 5)]
    front = compute_pareto_front(dup_dominated, OBJS)
    assert [(c.objectives["value"], c.objectives["cost"]) for c in front] == [(10, 5)]
