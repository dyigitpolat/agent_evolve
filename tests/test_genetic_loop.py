"""The genetic loop must evolve a population and respect its budget exactly."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.genetic import Locus
from agent_evolve.session.genetic_loop import (
    GeneticConfig,
    OperatorChoice,
    domination_rank,
    run_genetic_loop,
)


class _Candidate(BaseModel):
    genome: list[Literal[0, 1]]


class _CountOnes:
    """Maximize ones, minimize a decoy. Deterministic, free, no workload."""

    candidate_model = _Candidate
    objectives = (
        ObjectiveSpec(name="ones", goal="max"),
        ObjectiveSpec(name="tail", goal="min"),
    )

    def __init__(self, length: int = 8) -> None:
        self.length = length
        self.calls = 0

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ({"genome": [0] * self.length},)

    def validate(self, config):  # pragma: no cover - always feasible here
        from agent_evolve.core.problem import ValidationOutcome

        return ValidationOutcome(ok=True)

    def materialize(self, config):
        return tuple(config["genome"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        self.calls += 1
        return {"ones": float(sum(artifact)),
                "tail": float(sum(artifact[len(artifact) // 2:]))}


def test_loop_improves_on_its_seed() -> None:
    problem = _CountOnes(length=8)
    result = run_genetic_loop(
        problem=problem,
        config=GeneticConfig(population_size=6, offspring_per_generation=4,
                             generations=6, seed=11, evaluation_budget=40),
    )
    best_ones = max(c.objectives["ones"] for c in result.all_candidates)
    assert best_ones > 0.0, "an all-zero seed was never improved on"


def test_budget_is_never_exceeded() -> None:
    # The budget is the one sizing number the caller gives, and it counts the
    # expensive thing. Overrunning it silently would invalidate every
    # matched-budget comparison the project makes.
    for budget in (5, 12, 31):
        problem = _CountOnes(length=6)
        result = run_genetic_loop(
            problem=problem,
            config=GeneticConfig(population_size=4, offspring_per_generation=3,
                                 generations=20, seed=3,
                                 evaluation_budget=budget),
        )
        assert problem.calls <= budget, (
            f"asked for {budget} evaluations, the problem was called "
            f"{problem.calls} times"
        )
        assert result.evaluations <= budget


def test_a_chooser_cannot_author_a_candidate() -> None:
    # The claim this project defends is guidance over OPERATOR CHOICE rather
    # than artifact authoring. That distinction is enforced by the type: there
    # is no field on OperatorChoice that can carry a configuration.
    assert not any(
        f in OperatorChoice.__dataclass_fields__
        for f in ("config", "configuration", "candidate", "genome", "child")
    )
    assert set(OperatorChoice.__dataclass_fields__) == {
        "parent_a", "parent_b", "mask", "mutate_loci"
    }


def test_chooser_decisions_are_actually_applied() -> None:
    # A guided arm that silently falls through to random choice would produce a
    # null indistinguishable from an honest one.
    seen: List[int] = []

    def chooser(population, count, state=None):
        seen.append(len(population))
        return [
            OperatorChoice(parent_a=0, parent_b=0,
                           mask=tuple([False] * 6),
                           mutate_loci=(Locus("genome", 0),))
            for _ in range(count)
        ]

    problem = _CountOnes(length=6)
    run_genetic_loop(
        problem=problem,
        config=GeneticConfig(population_size=4, offspring_per_generation=2,
                             generations=3, seed=5, evaluation_budget=30),
        chooser=chooser,
    )
    assert seen, "the chooser was never consulted"
    assert all(n > 0 for n in seen)


def test_a_malformed_mask_does_not_crash_the_run() -> None:
    # A model-backed chooser will sometimes return the wrong length. Losing a
    # whole run to that is worse than repairing it and carrying on.
    def bad(population, count, state=None):
        return [OperatorChoice(parent_a=0, parent_b=1, mask=(True,))
                for _ in range(count)]

    problem = _CountOnes(length=8)
    result = run_genetic_loop(
        problem=problem,
        config=GeneticConfig(population_size=4, offspring_per_generation=2,
                             generations=3, seed=7, evaluation_budget=25),
        chooser=bad,
    )
    assert result.evaluations > 0


def test_no_seed_is_a_clear_error_not_a_crash() -> None:
    class _Seedless(_CountOnes):
        def seeds(self):
            return ()

    with pytest.raises(ValueError, match="needs at least one seed"):
        run_genetic_loop(problem=_Seedless(),
                         config=GeneticConfig(evaluation_budget=10))


def test_domination_rank_is_weight_free_and_pareto_correct() -> None:
    specs = (ObjectiveSpec(name="a", goal="min"),
             ObjectiveSpec(name="b", goal="min"))
    points = [
        {"a": 1.0, "b": 1.0},   # dominates both others
        {"a": 2.0, "b": 2.0},   # dominated by the first
        {"a": 1.0, "b": 3.0},   # dominated by the first only
    ]
    assert domination_rank(points, specs) == [0.0, 1.0, 1.0]


def test_domination_rank_respects_max_direction() -> None:
    specs = (ObjectiveSpec(name="a", goal="max"),)
    assert domination_rank([{"a": 5.0}, {"a": 1.0}], specs) == [0.0, 1.0]
