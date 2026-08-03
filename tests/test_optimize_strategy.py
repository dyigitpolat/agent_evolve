"""`optimize()` must route to the loop that measures better, and say so."""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.api import _resolve_strategy
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome


class _Candidate(BaseModel):
    genome: list[Literal[0, 1]]


class _Seeded:
    candidate_model = _Candidate
    objectives = (ObjectiveSpec(name="ones", goal="max"),)

    def __init__(self, seeds=None) -> None:
        self._seeds = ({"genome": [0, 0, 0, 0, 0, 0]},) if seeds is None else seeds
        self.calls = 0

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return self._seeds

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return tuple(config["genome"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        self.calls += 1
        return {"ones": float(sum(artifact))}


def test_auto_prefers_genetics_when_seeds_exist() -> None:
    # The authoring loop loses to uniform random on every genome length
    # measured; recombination wins on every one. So wherever genetics can run,
    # they should.
    assert _resolve_strategy("auto", True, lambda _m: None) == "genetic"


def test_auto_falls_back_to_authoring_without_seeds_and_says_why() -> None:
    said: list[str] = []
    assert _resolve_strategy("auto", False, said.append) == "authoring"
    assert said, "the fallback was silent"
    assert "seeds" in said[0].lower(), "the message does not name the fix"


def test_an_unknown_strategy_is_rejected_by_name() -> None:
    with pytest.raises(ValueError, match="strategy must be"):
        _resolve_strategy("evolutionary", True, lambda _m: None)


def test_optimize_runs_the_genetic_loop_with_no_credential() -> None:
    # The whole point of a drop-in optimizer: it works out of the box, offline,
    # and the provider is an upgrade rather than a requirement.
    problem = _Seeded()
    result = optimize(problem, budget=24, seed=5)
    assert result.evaluations <= 24
    assert problem.calls <= 24
    best = max(c.objectives["ones"] for c in result.all_candidates)
    assert best > 0.0, "the all-zero seed was never improved on"


def test_explicit_authoring_still_reaches_the_old_loop() -> None:
    # Opting back in must keep working; changing a default is not licence to
    # remove the thing it defaulted away from.
    problem = _Seeded()
    result = optimize(problem, budget=6, seed=5, strategy="authoring",
                      proposer="random")
    assert result.evaluations <= 6


def test_budget_is_respected_through_the_public_api() -> None:
    for budget in (4, 9, 20):
        problem = _Seeded()
        optimize(problem, budget=budget, seed=2)
        assert problem.calls <= budget, (
            f"budget {budget} but evaluate() was called {problem.calls} times"
        )


def test_the_budget_is_actually_spent_not_merely_respected() -> None:
    # Under-spending is as damaging as overrunning: an arm that quietly stops
    # early is not budget-matched to the control it is compared against. The
    # first sizing heuristic derived the population from the SEED COUNT, giving
    # a population of two on a single-seed problem -- which cannot recombine
    # into anything its parents do not already contain -- and ended runs with
    # half the budget unspent.
    for budget in (16, 24):
        problem = _Seeded()
        result = optimize(problem, budget=budget, seed=7)
        assert result.evaluations >= budget - 2, (
            f"budget {budget} but only {result.evaluations} evaluations were "
            "spent; the loop stopped on generations rather than on budget"
        )


def test_a_single_seed_still_yields_a_recombinable_population() -> None:
    # Population size must come from the budget, not from how many seeds the
    # caller happened to supply.
    from agent_evolve.session.genetic_loop import GeneticConfig  # noqa: F401

    problem = _Seeded()
    result = optimize(problem, budget=40, seed=1)
    distinct = {tuple(c.configuration["genome"]) for c in result.all_candidates}
    assert len(distinct) > 2, (
        "a single seed produced a population that never diversified"
    )


def test_example_config_seeds_a_problem_that_never_declared_seeds() -> None:
    # A problem that names a representative configuration but never implements
    # seeds() has not chosen to start from proposals -- it simply predates the
    # obligation. Using its example means the genetic loop can run at all;
    # without it such a problem silently falls back to authoring.
    from agent_evolve.contract import as_problem

    class _Legacy:
        candidate_model = _Candidate
        objectives = (ObjectiveSpec(name="ones", goal="max"),)
        example_config = {"genome": [1, 0, 1, 0, 1, 0]}

        def validate(self, config) -> bool:
            return True

        def evaluate(self, config) -> Mapping[str, float]:
            return {"ones": float(sum(config["genome"]))}

    bound = as_problem(_Legacy())
    assert [dict(s) for s in bound.seeds()] == [{"genome": [1, 0, 1, 0, 1, 0]}]


def test_an_explicit_empty_seeds_is_respected_not_overridden() -> None:
    # The contract says returning () means "start from proposals alone". A
    # problem that implements seeds() and returns nothing has made that choice,
    # and an example_config must not silently reverse it.
    from agent_evolve.contract import as_problem

    class _ChoseNone(_Seeded):
        example_config = {"genome": [1, 1, 1, 1, 1, 1]}

        def seeds(self):
            return ()

    assert tuple(as_problem(_ChoseNone()).seeds()) == ()
