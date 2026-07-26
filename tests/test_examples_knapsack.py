"""The knapsack example runs through fake and typed pydantic-ai paths offline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from agent_evolve.harness.base import HarnessContext, LLMConfig
from agent_evolve.session.loop import LoopConfig, run_evolution_loop
from fakes import CannedPydanticAIHarness, FakeHarness

_KNAPSACK = Path(__file__).resolve().parents[1] / "examples" / "knapsack" / "problem_def.py"


def _load_knapsack():
    spec = importlib.util.spec_from_file_location("knapsack_problem_def", _KNAPSACK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KnapsackProblem()


def _factory(i: int):
    return {"selection": [i % 10]}


def _run(harness, problem):
    harness.bind(
        HarnessContext(
            objectives=list(problem.objectives),
            search_space_desc=problem.search_space_description(),
            candidate_model=getattr(problem, "candidate_model", None),
        ),
        LLMConfig(model="mock"),
    )
    return run_evolution_loop(
        problem=problem,
        harness=harness,
        config=LoopConfig(pop_size=4, generations=2, candidates_per_batch=4, seed=0),
    )


def test_knapsack_runs_on_fake_and_pydantic_ai_harnesses():
    problem = _load_knapsack()
    fake = _run(FakeHarness(_factory), problem)
    pydantic_ai = _run(CannedPydanticAIHarness(_factory), problem)
    assert fake.pareto_front and pydantic_ai.pareto_front
    assert fake.best.objectives == pydantic_ai.best.objectives
