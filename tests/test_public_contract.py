"""The shipped public surface: five obligations, one entry point, no credentials.

Every test here runs offline, needs no research corpus and no provider. This is
the suite a stranger who cloned the repository can run.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, Field

import agent_evolve
from agent_evolve import ObjectiveSpec, ValidationOutcome, artifact_key, as_problem, optimize
from agent_evolve.session.evaluate import EvaluationCache, evaluate_batch


class Candidate(BaseModel):
    x: int = Field(..., ge=0, le=9)
    y: int = Field(..., ge=0, le=9)


class Complete:
    """A problem stating all five obligations."""

    candidate_model = Candidate

    def __init__(self) -> None:
        self.evaluations = 0

    @property
    def objectives(self):
        return [ObjectiveSpec("score", "max"), ObjectiveSpec("cost", "min")]

    def seeds(self):
        return [{"x": 1, "y": 1}]

    def validate(self, config) -> ValidationOutcome:
        if config.get("x", 0) + config.get("y", 0) > 15:
            return ValidationOutcome(False, "constraint", "x + y must be at most 15")
        return ValidationOutcome(True)

    def materialize(self, config):
        # Order-insensitive: (1, 2) and (2, 1) are the same artifact.
        return tuple(sorted((config["x"], config["y"])))

    def evaluate(self, artifact):
        self.evaluations += 1
        return {"score": float(sum(artifact)), "cost": float(max(artifact))}


class Legacy:
    """A problem predating the contract: objectives + evaluate only."""

    candidate_model = Candidate

    @property
    def objectives(self):
        return [ObjectiveSpec("score", "max")]

    def evaluate(self, config):
        return {"score": float(config["x"])}


# -- the public surface -----------------------------------------------------


def test_public_surface_is_small_and_documented():
    assert len(agent_evolve.__all__) <= 12
    for name in agent_evolve.__all__:
        assert hasattr(agent_evolve, name), name


def test_legacy_names_still_resolve_so_the_measured_tree_keeps_working():
    # B1's whole point: the shipped tree is the measured one. Cutting the
    # advertised surface must not break what the research imports.
    assert agent_evolve.WorkloadKit is not None
    with pytest.raises(AttributeError):
        agent_evolve.definitely_not_a_real_export


def test_importing_the_package_does_not_pull_in_the_whole_research_stack():
    import subprocess
    import sys

    loaded = subprocess.run(
        [sys.executable, "-c",
         "import agent_evolve, sys; "
         "print(len([m for m in sys.modules if m.startswith('agent_evolve')]))"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert int(loaded) < 60, f"import loaded {loaded} modules"


# -- the five obligations ---------------------------------------------------


def test_as_problem_accepts_a_complete_problem_unchanged():
    p = Complete()
    assert as_problem(p) is p


def test_as_problem_adapts_a_legacy_problem():
    adapted = as_problem(Legacy())
    assert adapted is not None
    assert tuple(adapted.seeds()) == ()
    assert adapted.materialize({"x": 3, "y": 4}) == {"x": 3, "y": 4}
    assert adapted.validate({"x": 1}).ok


def test_as_problem_rejects_a_problem_missing_the_irreducible_members():
    class NoObjectives:
        def evaluate(self, artifact):
            return {}

    with pytest.raises(TypeError, match="objectives"):
        as_problem(NoObjectives())


def test_seeds_are_evaluated_before_anything_is_proposed():
    p = Complete()
    result = optimize(p, budget=12, proposer="random", seed=0)
    seeded = [c for c in result.all_candidates if c.configuration == {"x": 1, "y": 1}]
    assert seeded, "the declared seed never reached the evaluator"


# -- materialize earns its keep --------------------------------------------


def test_distinct_configurations_with_one_artifact_are_evaluated_once():
    """The measured justification for materialize being an obligation."""
    p = Complete()
    cache = EvaluationCache()
    objectives = list(p.objectives)

    evaluate_batch(p, [{"x": 1, "y": 2}], objectives, cache=cache)
    assert p.evaluations == 1

    # Same artifact, different configuration. Keying on the configuration --
    # which is what this did before -- would pay for it a second time.
    evaluate_batch(p, [{"x": 2, "y": 1}], objectives, cache=cache)
    assert p.evaluations == 1
    assert cache.hits == 1


def test_artifact_key_is_stable_across_key_order():
    assert artifact_key({"a": 1, "b": 2}) == artifact_key({"b": 2, "a": 1})
    assert artifact_key((1, 2)) != artifact_key((2, 1))


def test_artifact_key_never_raises_on_exotic_artifacts():
    """materialize() may return anything; keying it must not be the thing that fails."""

    class Unhashable:
        __hash__ = None  # type: ignore[assignment]

        def __repr__(self) -> str:
            return "<unhashable artifact>"

    for artifact in (object(), Unhashable(), {1, 2, 3}, b"bytes", 3.5, None):
        key = artifact_key(artifact)
        assert isinstance(key, str) and len(key) == 64

    # Equal-reprs collapse; that is the documented fallback, not an accident.
    assert artifact_key(Unhashable()) == artifact_key(Unhashable())


# -- the budget is a promise ------------------------------------------------


@pytest.mark.parametrize("budget", [4, 9, 17])
def test_budget_is_a_hard_cap_not_a_suggestion(budget):
    p = Complete()
    result = optimize(p, budget=budget, proposer="random", seed=1)
    assert result.evaluations <= budget
    assert p.evaluations <= budget


def test_budget_must_be_a_positive_integer():
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            optimize(Complete(), budget=bad, proposer="random")


# -- credential-free operation ---------------------------------------------


def test_random_proposer_needs_no_credentials_and_no_network(monkeypatch):
    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_TOKEN", "_SECRET")):
            monkeypatch.delenv(name, raising=False)
    result = optimize(Complete(), budget=12, proposer="random", seed=0)
    assert result.evaluations > 0
    assert result.pareto_front


def test_auto_falls_back_to_random_and_says_so(monkeypatch):
    monkeypatch.setattr("agent_evolve.api.credentials_present", lambda: False)
    said = []
    optimize(Complete(), budget=8, proposer="auto", seed=0, on_progress=said.append)
    assert any("random" in m for m in said), "the fallback must be announced, not silent"


def test_unknown_proposer_is_rejected_by_name():
    with pytest.raises(ValueError, match="proposer"):
        optimize(Complete(), budget=8, proposer="magic")


# -- validation feedback ----------------------------------------------------


def test_validation_message_reaches_the_failure_record():
    p = Complete()
    _, failed, _ = evaluate_batch(
        p, [{"x": 9, "y": 9}], list(p.objectives), cache=EvaluationCache()
    )
    assert failed and "at most 15" in (failed[0].error_message or "")
    assert p.evaluations == 0, "an invalid candidate must not reach the evaluator"
