"""Tests for the evolution loop with mock LLM procedures.

No LLM calls — the procedure callables are plain Python stubs.
"""

import pytest

from agent_evolve.problem import ObjectiveSpec
from agent_evolve.results import SearchResult
from agent_evolve._support import format_search_space_description
from agent_evolve.loop import run_evolution_loop


# ------------------------------------------------------------------
# Test problems
# ------------------------------------------------------------------

class _SimpleProblem:
    @property
    def objectives(self):
        return [ObjectiveSpec("score", "max")]

    def evaluate(self, config):
        return {"score": float(config.get("x", 0))}


class _SelectiveProblem:
    """Rejects configs where x < 5."""

    @property
    def objectives(self):
        return [ObjectiveSpec("score", "max")]

    def evaluate(self, config):
        x = config.get("x", 0)
        if x < 5:
            raise ValueError(f"x={x} too small")
        return {"score": float(x)}


class _BiObjectiveProblem:
    @property
    def objectives(self):
        return [
            ObjectiveSpec("value", "max"),
            ObjectiveSpec("cost", "min"),
        ]

    def evaluate(self, config):
        x = config.get("x", 0)
        return {"value": float(x), "cost": float(x * 0.5)}


# ------------------------------------------------------------------
# Mock procedures
# ------------------------------------------------------------------

def _make_mock_procs():
    """Return a dict of mock LLM procedure callables."""

    def generate_initial_candidates(search_space_desc, n_candidates):
        return [{"x": i * 10} for i in range(1, n_candidates + 1)]

    def regenerate_candidates(failed_str, n, desc, ci, pi):
        return [{"x": 50 + i} for i in range(n)]

    def generate_offspring(pareto_str, n, desc, ci, pi):
        return [{"x": 80 + i} for i in range(n)]

    def regenerate_offspring(failed, pareto, n, desc, ci, pi):
        return [{"x": 90 + i} for i in range(n)]

    def generate_failure_insights(failed_str, desc, n_failed):
        return [f"insight {i}" for i in range(n_failed)]

    def generate_constraint_instruction(failed_str, desc):
        return "Mock constraint instruction."

    def update_constraint_instruction(prev, failed_str, desc):
        return prev + " Updated."

    def generate_performance_insights(stats_str, desc):
        return "Mock performance insights."

    def update_performance_insights(prev, pareto_str, total, pareto_size):
        return prev + " Updated."

    return dict(
        generate_initial_candidates=generate_initial_candidates,
        regenerate_candidates=regenerate_candidates,
        generate_offspring=generate_offspring,
        regenerate_offspring=regenerate_offspring,
        generate_failure_insights=generate_failure_insights,
        generate_constraint_instruction=generate_constraint_instruction,
        update_constraint_instruction=update_constraint_instruction,
        generate_performance_insights=generate_performance_insights,
        update_performance_insights=update_performance_insights,
    )


def _run(problem, *, pop_size=4, generations=2, candidates_per_batch=4,
         max_regen_rounds=5, max_failed_examples=3, procs=None):
    if procs is None:
        procs = _make_mock_procs()
    objectives = list(problem.objectives)
    desc = format_search_space_description(objectives)
    return run_evolution_loop(
        problem=problem,
        objectives=objectives,
        search_space_desc=desc,
        pop_size=pop_size,
        generations=generations,
        candidates_per_batch=candidates_per_batch,
        max_regen_rounds=max_regen_rounds,
        max_failed_examples=max_failed_examples,
        **procs,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestBasicLoop:
    def test_returns_search_result(self):
        result = _run(_SimpleProblem())
        assert isinstance(result, SearchResult)

    def test_best_has_positive_score(self):
        result = _run(_SimpleProblem())
        assert result.best.objectives["score"] > 0

    def test_pareto_front_nonempty(self):
        result = _run(_SimpleProblem())
        assert len(result.pareto_front) > 0

    def test_history_matches_generations(self):
        result = _run(_SimpleProblem(), generations=3)
        assert len(result.history) == 3
        assert [h["gen"] for h in result.history] == [1, 2, 3]

    def test_all_candidates_tracked(self):
        result = _run(_SimpleProblem(), pop_size=3, generations=2)
        assert len(result.all_candidates) >= 6


class TestRegenerationLoop:
    def test_failures_trigger_regen(self):
        result = _run(_SelectiveProblem(), pop_size=3, max_regen_rounds=3)
        assert len(result.pareto_front) > 0

    def test_all_pareto_valid(self):
        result = _run(_SelectiveProblem(), pop_size=3, max_regen_rounds=3)
        for c in result.pareto_front:
            assert c.objectives["score"] >= 5


class TestBiObjective:
    def test_pareto_front_has_tradeoffs(self):
        result = _run(_BiObjectiveProblem(), pop_size=4, generations=2)
        assert len(result.pareto_front) >= 1

    def test_best_candidate_exists(self):
        result = _run(_BiObjectiveProblem(), pop_size=4, generations=2)
        assert "value" in result.best.objectives
        assert "cost" in result.best.objectives
