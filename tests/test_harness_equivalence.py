"""The pydantic-ai and fake harnesses drive the loop to the same result.

Both consume the identical candidate factory, so a divergence here means the
harness boundary leaked behavior. The pydantic-ai path exercises its real typed
``Proposal`` and candidate normalization with an offline model-runner override.
"""

from __future__ import annotations

from agent_evolve import optimize
from agent_evolve.harness.base import HarnessContext, LLMConfig
from agent_evolve.harness.registry import harness_registry
from agent_evolve.session.loop import LoopConfig, run_evolution_loop
from fakes import CannedPydanticAIHarness, FakeHarness, SimpleProblem, default_factory


def _run(harness):
    problem = SimpleProblem()
    harness.bind(
        HarnessContext(
            objectives=list(problem.objectives),
            search_space_desc="desc",
            candidate_model=problem.candidate_model,
        ),
        LLMConfig(model="mock"),
    )
    return run_evolution_loop(
        problem=problem,
        harness=harness,
        config=LoopConfig(pop_size=4, generations=3, candidates_per_batch=4, seed=0),
    )


def _canned_pydantic_ai():
    return CannedPydanticAIHarness(default_factory)


def test_pydantic_ai_path_runs_end_to_end_without_provider_call():
    result = _run(_canned_pydantic_ai())
    assert result.pareto_front
    assert result.evaluations > 0


def test_pydantic_ai_and_fake_harness_agree():
    fake = _run(FakeHarness(default_factory))
    pydantic_ai = _run(_canned_pydantic_ai())
    assert fake.best.objectives == pydantic_ai.best.objectives
    assert fake.evaluations == pydantic_ai.evaluations
    assert {tuple(sorted(c.objectives.items())) for c in fake.pareto_front} == {
        tuple(sorted(c.objectives.items())) for c in pydantic_ai.pareto_front
    }


def test_optimize_accepts_an_externally_registered_harness():
    """An out-of-tree adapter is selected by name through ``proposer``.

    There is no separate ``harness`` parameter: a registered harness *is* a
    proposer, and two parameters meaning almost the same thing is the kind of
    surface this release exists to remove.
    """
    harness_registry.register("offline_external", lambda: FakeHarness(default_factory))

    result = optimize(
        SimpleProblem(),
        budget=8,
        proposer="offline_external",
        model="offline",
        seed=0,
    )

    assert result.pareto_front
    assert result.evaluations > 0
