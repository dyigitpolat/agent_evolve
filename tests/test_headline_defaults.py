"""The default purchase: the seams that measured, and not the one that did not.

`optimize()` used to build the completion seam inside the per-offspring
chooser's own branch, so a caller who wanted the two mechanisms that won --
model-proposed initialization and the weighted prior -- could only have them
bundled with the mechanism that lost: ten sealed null verdicts, one model call
per offspring, 61% of the six-arm ablation's ledger for 0.94x the speed of
doing nothing.

These tests pin the split. The seam is built whenever the run makes model
calls; the chooser is opt-in and refuses rather than no-ops; the sentinel
guidance defaults resolve to the measured stack on the model path and to the
literal pre-sentinel defaults on the credential-free one, where the fossil
stream must not move.

Everything here runs offline against a fake completion callable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.api import _resolve_guidance
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome

TEMPLATE = {"genome": [0, 0, 0, 0, 0, 0]}


class _Candidate(BaseModel):
    genome: list[Literal[0, 1]]


class _Problem:
    """A toy one-max: cheap enough to run a 96-evaluation budget in a test."""

    candidate_model = _Candidate
    objectives = (ObjectiveSpec(name="ones", goal="max"),)

    def __init__(self) -> None:
        self.stream: list[list[int]] = []

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return (dict(TEMPLATE),)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return tuple(config["genome"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        self.stream.append(list(artifact))
        return {"ones": float(sum(artifact))}


def _recording_seam(prompts: list, reply: str = "{}"):
    """A `completion_for` stand-in that records every prompt it is handed."""

    def _factory(model, settings=None, **kwargs):
        def _complete(prompt: str) -> str:
            prompts.append(prompt)
            return reply

        return _complete

    return _factory


def _install_seam(monkeypatch, prompts: list, reply: str = "{}") -> None:
    monkeypatch.setattr(
        "agent_evolve.integrations.completion.completion_for",
        _recording_seam(prompts, reply))


# --- t1/t2: the chooser is opt-in, the seam is not --------------------------

def test_the_chooser_is_off_by_default_and_the_init_seam_still_fires(monkeypatch):
    """The whole point of the split: the winning seam without the losing one."""

    prompts: list[str] = []
    _install_seam(monkeypatch, prompts,
                  reply=json.dumps([{"genome": [1, 1, 1, 1, 1, 0]},
                                    {"genome": [0, 1, 1, 0, 1, 1]},
                                    {"genome": [1, 0, 0, 1, 1, 1]}]))

    def _forbidden(*_args, **_kwargs):
        raise AssertionError(
            "the per-offspring chooser was constructed on a default run; it "
            "has ten sealed null verdicts and costs 107-171x the run it advises"
        )

    monkeypatch.setattr("agent_evolve.policies.llm_chooser.llm_chooser",
                        _forbidden)

    problem = _Problem()
    result = optimize(problem, budget=16, seed=6, proposer="llm")

    assert result.evaluations <= 16
    assert any("INITIAL POPULATION" in p for p in prompts), (
        "the completion seam was not reachable without the chooser: no "
        "initialization prompt was ever sent"
    )


def test_chooser_llm_constructs_the_chooser(monkeypatch):
    prompts: list[str] = []
    _install_seam(monkeypatch, prompts)
    built: list[dict] = []

    def _spy(complete, **kwargs):
        built.append(kwargs)
        return None  # the loop falls back to its random control

    monkeypatch.setattr("agent_evolve.policies.llm_chooser.llm_chooser", _spy)

    optimize(_Problem(), budget=16, seed=6, proposer="llm", chooser="llm")
    assert built, "chooser='llm' was asked for and no chooser was built"
    assert built[0]["budget"] == 16


def test_chooser_llm_on_a_run_with_no_model_call_is_refused_by_name():
    # No-silent-no-op: a chooser that cannot call a model never chooses, and
    # the run would be indistinguishable from one that never asked.
    with pytest.raises(ValueError, match="proposer"):
        optimize(_Problem(), budget=8, seed=1, proposer="random",
                 chooser="llm")
    with pytest.raises(ValueError, match="credential"):
        optimize(_Problem(), budget=8, seed=1, proposer="random",
                 chooser="llm")


def test_a_bogus_chooser_is_rejected_by_name():
    with pytest.raises(ValueError, match="chooser"):
        optimize(_Problem(), budget=8, chooser="sometimes")


def test_chooser_llm_refuses_the_authoring_strategy_by_name(monkeypatch):
    prompts: list[str] = []
    _install_seam(monkeypatch, prompts)
    with pytest.raises(ValueError, match="genetic"):
        optimize(_Problem(), budget=8, proposer="llm", strategy="authoring",
                 chooser="llm")


# --- t4: the credential-free path is byte-identical -------------------------

def test_the_sentinels_resolve_to_the_literal_old_defaults_offline():
    said: list[str] = []
    assert _resolve_guidance("auto", "auto", budget=96, model_calls=False,
                             announce=said.append) == ("rule", 0)
    assert not said, "the offline resolution announced a choice it did not make"


def test_the_credential_free_run_is_unchanged_by_the_sentinels():
    """`auto` offline must be the same run as the pre-sentinel literals.

    The fossil test holds the stream itself; this holds the equivalence the
    sentinels were required to preserve.
    """

    literal_problem = _Problem()
    literal = optimize(literal_problem, budget=24, seed=7, proposer="random",
                       prior="rule", structure_budget=0)
    auto_problem = _Problem()
    auto = optimize(auto_problem, budget=24, seed=7, proposer="random")

    assert auto_problem.stream == literal_problem.stream
    assert auto.history == literal.history
    assert [(c.configuration, c.objectives) for c in auto.pareto_front] == [
        (c.configuration, c.objectives) for c in literal.pareto_front]


# --- t5: the screen is sized from the budget --------------------------------

@pytest.mark.parametrize("budget, expected", [
    (40, 0), (47, 0), (48, 8), (96, 16), (200, 16),
])
def test_auto_structure_budget_is_sized_from_the_budget(budget, expected):
    said: list[str] = []
    prior, structure_budget = _resolve_guidance(
        "auto", "auto", budget=budget, model_calls=True, announce=said.append)
    # 2026-08-19: the model prior is bought exactly when a screen will run.
    # Announcing prior='llm-weighted' beside structure_budget=0 was a promise
    # the run never cashed (the prior seat only acts on screen evidence).
    assert prior == ("llm-weighted" if expected else "rule")
    assert structure_budget == expected
    assert (any("prior" in m for m in said) == bool(expected)), (
        "the prior default must announce exactly when the screen is on"
    )
    assert (any("structure_budget" in m for m in said) == bool(expected)), (
        "a non-trivial screen must announce itself, and a skipped one must not"
    )


def test_the_auto_screen_reaches_the_loop_and_is_charged(monkeypatch):
    prompts: list[str] = []
    _install_seam(monkeypatch, prompts)
    said: list[str] = []
    problem = _Problem()
    result = optimize(problem, budget=96, seed=3, proposer="llm",
                      on_progress=said.append)
    record = next(h["structure"] for h in result.history if "structure" in h)
    assert record["evaluated"] == 16, (
        "the auto-sized screen did not reach the loop at budget 96"
    )
    assert result.evaluations <= 96
    assert any("six-arm ablation" in m for m in said), (
        "the auto-sizing did not cite the row it comes from"
    )


def test_an_explicit_value_still_wins_over_the_sentinel(monkeypatch):
    prompts: list[str] = []
    _install_seam(monkeypatch, prompts)
    result = optimize(_Problem(), budget=96, seed=3, proposer="llm",
                      structure_budget=4, prior="rule")
    record = next(h["structure"] for h in result.history if "structure" in h)
    assert record["evaluated"] == 4


def test_the_sentinel_does_not_disarm_the_nonsense_checks():
    with pytest.raises(ValueError, match="structure_budget"):
        optimize(_Problem(), budget=8, structure_budget=-1)
    with pytest.raises(ValueError, match="structure_budget"):
        optimize(_Problem(), budget=8, structure_budget=8)
    with pytest.raises(ValueError, match="prior"):
        optimize(_Problem(), budget=8, prior="bogus")


# --- t6/t7: what `authorship="auto"` now buys -------------------------------

def test_authorship_auto_with_a_model_buys_surrogate_and_initialization(monkeypatch):
    prompts: list[str] = []
    _install_seam(monkeypatch, prompts,
                  reply=json.dumps([{"genome": [1, 1, 1, 1, 1, 0]},
                                    {"genome": [0, 1, 1, 0, 1, 1]},
                                    {"genome": [1, 0, 0, 1, 1, 1]}]))
    said: list[str] = []
    result = optimize(_Problem(), budget=16, seed=6, proposer="llm",
                      on_progress=said.append)
    announced = " ".join(said)
    assert "surrogate" in announced and "initialization" in announced, (
        "the resolved authorship did not name both seams it bought"
    )
    assert "authorship='off'" in announced, "the way out was not named"
    mechanisms = {m.mechanism for m in result.telemetry.mechanisms}
    assert "init_author" in mechanisms, (
        "authorship='auto' with a model did not turn on model-proposed "
        "initialization"
    )


def test_authorship_auto_without_a_model_stays_off():
    said: list[str] = []
    result = optimize(_Problem(), budget=12, seed=6, proposer="random",
                      on_progress=said.append)
    assert not [m for m in result.telemetry.mechanisms
                if m.authored_by == "llm"], (
        "an offline run reported an llm-authored mechanism"
    )


def test_the_guided_preset_names_the_default_purchase():
    from agent_evolve.session.authorship import AuthorshipConfig

    config = AuthorshipConfig.preset("guided")
    assert config.surrogate == "llm"
    assert config.initialization == "llm"
    assert config.operators == "off" and config.generation == "off"


def test_an_explicit_llm_request_with_no_credential_is_refused_by_name(monkeypatch):
    """2026-08-20: found by the release CI's stranger job. proposer='llm' with
    no credential ran the classical path to completion and said nothing -- a
    run launched to measure a model measured the control, and the only trace
    was calls: 0. An explicit request that cannot be honoured refuses loudly
    and names both ways out."""

    import agent_evolve.integrations.completion as seam

    monkeypatch.setattr(seam, "completion_for",
                        lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="proposer='random'"):
        optimize(_Problem(), budget=16, seed=6, proposer="llm")
