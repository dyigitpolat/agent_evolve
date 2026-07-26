"""End-to-end loop with a fake harness + fake problem (no LLM)."""

from __future__ import annotations

from agent_evolve.session.loop import LoopConfig, run_evolution_loop
from fakes import FakeHarness, SimpleProblem


def _bind(harness, problem):
    from agent_evolve.harness.base import HarnessContext, LLMConfig

    harness.bind(
        HarnessContext(objectives=list(problem.objectives), search_space_desc="desc"),
        LLMConfig(model="mock"),
    )


def test_loop_produces_pareto_and_counts_evaluations():
    problem = SimpleProblem()
    harness = FakeHarness()
    _bind(harness, problem)
    events = []
    result = run_evolution_loop(
        problem=problem,
        harness=harness,
        config=LoopConfig(pop_size=4, generations=3, candidates_per_batch=4, seed=0),
        on_event=events.append,
    )
    assert result.pareto_front, "expected a non-empty Pareto front"
    assert result.best.objectives  # a real best was chosen
    assert result.evaluations == sum(
        bool(c.metadata["evaluation_attempted"]) for c in result.all_candidates
    )
    assert result.evaluations < len(result.all_candidates)  # one cheap pre-check reject
    assert result.evaluations > 0
    # history has one entry per generation
    assert len(result.history) == 3
    assert len(result.best_per_generation) == 3
    # events include generation_complete and search_complete
    kinds = {e["kind"] for e in events}
    assert "generation_complete" in kinds and "search_complete" in kinds


def test_loop_is_deterministic_with_seed():
    def run():
        problem = SimpleProblem()
        harness = FakeHarness()
        _bind(harness, problem)
        return run_evolution_loop(
            problem=problem,
            harness=harness,
            config=LoopConfig(pop_size=4, generations=2, candidates_per_batch=4, seed=7),
        )

    r1, r2 = run(), run()
    assert r1.best.objectives == r2.best.objectives
    assert r1.evaluations == r2.evaluations


def test_surplus_valid_candidates_are_retained_and_counted():
    """A provider that over-returns must not make paid evaluations disappear."""

    class OverproducingHarness(FakeHarness):
        def _batch(self, op, n):
            self.calls.append(op)
            return [{"a": i, "b": 0} for i in range(5)]  # ignores requested n=3

    problem = SimpleProblem()
    harness = OverproducingHarness()
    _bind(harness, problem)
    result = run_evolution_loop(
        problem=problem,
        harness=harness,
        config=LoopConfig(
            pop_size=3,
            generations=1,
            candidates_per_batch=5,
            max_regen_rounds=0,
            seed=0,
        ),
    )

    assert result.evaluations == 5
    assert len(result.all_candidates) == 5
    assert result.history[0]["valid_count"] == 5
    assert result.best.objectives["score"] == 4.0


def test_evaluations_count_objective_calls_not_validation_rejections():
    problem = SimpleProblem()
    harness = FakeHarness(
        lambda i: {"a": 1, "b": 1} if i == 0 else {"a": 99, "b": 99}
    )
    _bind(harness, problem)
    result = run_evolution_loop(
        problem=problem,
        harness=harness,
        config=LoopConfig(
            pop_size=2,
            generations=1,
            candidates_per_batch=2,
            max_regen_rounds=1,
            seed=0,
        ),
    )

    assert result.evaluations == 1
    assert len(result.all_candidates) == 2
    attempted = sorted(c.metadata["evaluation_attempted"] for c in result.all_candidates)
    assert attempted == [False, True]
    failure = next(c for c in result.all_candidates if not c.metadata["valid"])
    assert failure.metadata["failure_phase"] == "constraint"
    assert "error_message" in failure.metadata
