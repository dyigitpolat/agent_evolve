"""Tests for Pareto dominance, front computation, and best-candidate selection."""

import pytest

from agent_evolve.problem import ObjectiveSpec
from agent_evolve.results import (
    Candidate,
    SearchResult,
    compute_pareto_front,
    dominates,
    select_best_candidate,
    select_minimax_rank,
)

MAX_MIN = [ObjectiveSpec("accuracy", "max"), ObjectiveSpec("cost", "min")]
ONLY_MIN = [ObjectiveSpec("latency", "min"), ObjectiveSpec("cost", "min")]


# ------------------------------------------------------------------
# dominates
# ------------------------------------------------------------------

class TestDominates:
    def test_strictly_better(self):
        a = {"accuracy": 0.9, "cost": 1.0}
        b = {"accuracy": 0.8, "cost": 2.0}
        assert dominates(a, b, MAX_MIN)
        assert not dominates(b, a, MAX_MIN)

    def test_equal_not_dominating(self):
        a = {"accuracy": 0.9, "cost": 1.0}
        assert not dominates(a, a, MAX_MIN)

    def test_trade_off_no_dominance(self):
        a = {"accuracy": 0.9, "cost": 5.0}
        b = {"accuracy": 0.7, "cost": 1.0}
        assert not dominates(a, b, MAX_MIN)
        assert not dominates(b, a, MAX_MIN)

    def test_all_min(self):
        a = {"latency": 1.0, "cost": 2.0}
        b = {"latency": 3.0, "cost": 4.0}
        assert dominates(a, b, ONLY_MIN)


# ------------------------------------------------------------------
# compute_pareto_front
# ------------------------------------------------------------------

class TestParetoFront:
    def test_single_candidate(self):
        c = Candidate({"x": 1}, {"accuracy": 0.9, "cost": 1.0})
        front = compute_pareto_front([c], MAX_MIN)
        assert front == [c]

    def test_dominated_filtered(self):
        c1 = Candidate({"x": 1}, {"accuracy": 0.9, "cost": 1.0})
        c2 = Candidate({"x": 2}, {"accuracy": 0.8, "cost": 2.0})
        front = compute_pareto_front([c1, c2], MAX_MIN)
        assert front == [c1]

    def test_trade_off_both_kept(self):
        c1 = Candidate({"x": 1}, {"accuracy": 0.95, "cost": 5.0})
        c2 = Candidate({"x": 2}, {"accuracy": 0.80, "cost": 1.0})
        front = compute_pareto_front([c1, c2], MAX_MIN)
        assert len(front) == 2

    def test_empty(self):
        assert compute_pareto_front([], MAX_MIN) == []

    def test_three_candidates(self):
        c1 = Candidate({}, {"accuracy": 0.9, "cost": 2.0})
        c2 = Candidate({}, {"accuracy": 0.8, "cost": 1.0})
        c3 = Candidate({}, {"accuracy": 0.7, "cost": 3.0})  # dominated by both
        front = compute_pareto_front([c1, c2, c3], MAX_MIN)
        assert c1 in front
        assert c2 in front
        assert c3 not in front


# ------------------------------------------------------------------
# select_best_candidate
# ------------------------------------------------------------------

class TestSelectBest:
    def test_single(self):
        c = Candidate({}, {"accuracy": 0.9, "cost": 1.0})
        assert select_best_candidate([c], MAX_MIN) is c

    def test_prefers_max_first(self):
        c1 = Candidate({}, {"accuracy": 0.95, "cost": 5.0})
        c2 = Candidate({}, {"accuracy": 0.80, "cost": 1.0})
        best = select_best_candidate([c1, c2], MAX_MIN)
        assert best is c1

    def test_custom_priority(self):
        c1 = Candidate({}, {"accuracy": 0.95, "cost": 5.0})
        c2 = Candidate({}, {"accuracy": 0.80, "cost": 1.0})
        best = select_best_candidate([c1, c2], MAX_MIN, priority_order=["cost", "accuracy"])
        assert best is c2

    def test_empty(self):
        assert select_best_candidate([], MAX_MIN) is None


# ------------------------------------------------------------------
# select_minimax_rank
# ------------------------------------------------------------------

class TestMinimaxRank:
    def test_single(self):
        c = Candidate({}, {"accuracy": 0.9, "cost": 1.0})
        assert select_minimax_rank([c], MAX_MIN) is c

    def test_balanced_preferred(self):
        c_balanced = Candidate({}, {"accuracy": 0.85, "cost": 2.0})
        c_extreme1 = Candidate({}, {"accuracy": 0.99, "cost": 9.0})
        c_extreme2 = Candidate({}, {"accuracy": 0.50, "cost": 0.5})
        best = select_minimax_rank([c_balanced, c_extreme1, c_extreme2], MAX_MIN)
        assert best is c_balanced

    def test_empty(self):
        assert select_minimax_rank([], MAX_MIN) is None


# ------------------------------------------------------------------
# SearchResult
# ------------------------------------------------------------------

class TestSearchResult:
    def test_construction(self):
        c = Candidate({}, {"accuracy": 0.9, "cost": 1.0})
        sr = SearchResult(objectives=MAX_MIN, best=c, pareto_front=[c])
        assert sr.best is c
        assert len(sr.pareto_front) == 1
