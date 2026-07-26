"""Candidate dedup + the render_candidate formatting hook."""

from __future__ import annotations

from agent_evolve.core.formatting import CandidateResult, prettify_results
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.harness.base import HarnessContext, LLMConfig
from agent_evolve.session.loop import LoopConfig, run_evolution_loop
from fakes import FakeHarness, SimpleProblem


def test_prettify_results_uses_render_hook():
    objs = [ObjectiveSpec("score", "max")]
    r = CandidateResult(configuration={"a": 3, "b": 2}, objectives={"score": 3.0}, is_valid=True)
    out = prettify_results([r], objs, render=lambda c: f"COMPACT(a={c['a']})")
    assert "COMPACT(a=3)" in out
    assert '"a": 3' not in out  # rendered, not JSON


def test_prettify_results_defaults_to_json():
    objs = [ObjectiveSpec("score", "max")]
    r = CandidateResult(configuration={"a": 3}, objectives={"score": 3.0}, is_valid=True)
    out = prettify_results([r], objs)
    assert '"a": 3' in out


def test_loop_dedups_identical_candidates():
    problem = SimpleProblem()
    harness = FakeHarness(lambda i: {"a": 1, "b": 1})  # every proposal identical
    harness.bind(
        HarnessContext(objectives=list(problem.objectives), search_space_desc="d",
                       candidate_model=problem.candidate_model),
        LLMConfig(model="mock"),
    )
    result = run_evolution_loop(
        problem=problem,
        harness=harness,
        config=LoopConfig(pop_size=4, generations=2, candidates_per_batch=4, seed=0),
    )
    # Only one unique config exists, so it is evaluated exactly once despite many batches.
    assert result.evaluations == 1
