"""Banked mechanisms must be reachable from the public surface.

The structure phase, the prior proposers, the reasoning-effort pin and the
completion journal all existed before this file did -- reachable from
``GeneticConfig`` and ``completion_for``, and from nothing a user can call.
A mechanism the public surface cannot reach is a mechanism the release does
not have. These tests pin the wiring: ``optimize()`` knobs, their validation,
their refusals on the wrong strategy, and the CLI's ``--json`` output.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome


class _Candidate(BaseModel):
    width: Literal[8, 16]
    regs: bool
    depth: Literal[256, 512, 1024]


class CliProblem:
    """An AND-gate landscape, importable by the CLI as module:attribute."""

    candidate_model = _Candidate
    objectives = (
        ObjectiveSpec(name="energy", goal="min"),
        ObjectiveSpec(name="speed", goal="max"),
    )

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ({"width": 16, "regs": False, "depth": 512},)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return dict(config)

    def evaluate(self, artifact) -> Mapping[str, float]:
        self.seen.append(dict(artifact))
        good = artifact["width"] == 8 and artifact["regs"]
        return {"energy": (1.0 if good else 4.0) + artifact["depth"] / 4096.0,
                "speed": 2.0}


# --- optimize() knobs --------------------------------------------------------

def test_structure_budget_runs_the_phase_through_optimize():
    problem = CliProblem()
    result = optimize(problem, budget=16, seed=2, structure_budget=4)
    record = next(h["structure"] for h in result.history if "structure" in h)
    assert record["evaluated"] == 4
    assert result.evaluations <= 16


def test_rule_weighted_prior_reaches_the_loop():
    problem = CliProblem()
    result = optimize(problem, budget=16, seed=3, structure_budget=4,
                      prior="rule-weighted")
    record = next(h["structure"] for h in result.history if "structure" in h)
    assert "allowed" in record


def test_llm_priors_without_a_credential_fall_back_out_loud():
    said: list[str] = []
    optimize(CliProblem(), budget=12, seed=4, structure_budget=4,
             prior="llm-weighted", on_progress=said.append)
    assert any("rule" in m for m in said), (
        "the fallback from a model prior to the rule was silent"
    )


def test_bad_knob_values_are_rejected_by_name():
    with pytest.raises(ValueError, match="prior"):
        optimize(CliProblem(), budget=8, prior="nonsense")
    with pytest.raises(ValueError, match="structure_budget"):
        optimize(CliProblem(), budget=8, structure_budget=-1)
    with pytest.raises(ValueError, match="structure_budget"):
        optimize(CliProblem(), budget=8, structure_budget=8)
    with pytest.raises(ValueError, match="effort"):
        optimize(CliProblem(), budget=8, effort=17)


def test_genetic_only_knobs_refuse_the_authoring_strategy_by_name():
    # A knob the run would silently ignore is a silent no-op; refuse and name
    # the ways out, exactly as seal= does in the other direction.
    with pytest.raises(ValueError, match="genetic"):
        optimize(CliProblem(), budget=8, strategy="authoring",
                 proposer="random", structure_budget=4)
    with pytest.raises(ValueError, match="genetic"):
        optimize(CliProblem(), budget=8, strategy="authoring",
                 proposer="random", effort="low")


def test_effort_with_no_model_call_is_announced():
    said: list[str] = []
    optimize(CliProblem(), budget=8, seed=5, effort="low",
             on_progress=said.append)
    assert any("effort" in m.lower() for m in said), (
        "effort was requested, no model call will honour it, and nobody said so"
    )


def test_a_journal_path_is_created_even_when_no_call_happens(tmp_path):
    # An empty journal file is a measured zero; an absent file cannot be told
    # apart from a run that never looked.
    path = tmp_path / "calls.jsonl"
    optimize(CliProblem(), budget=8, seed=6, journal=str(path))
    assert path.exists()
    assert path.read_text() == ""


# --- CLI ---------------------------------------------------------------------

def test_run_json_emits_one_parseable_document(capsys):
    from agent_evolve.cli import main

    code = main(["run", "test_public_knobs:CliProblem",
                 "--budget", "8", "--seed", "1", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert set(payload) >= {"best", "pareto_front", "evaluations",
                            "telemetry", "provider_usage", "history"}
    assert payload["telemetry"]["real_evaluations"] == payload["evaluations"]


def test_run_without_json_still_prints_the_prose(capsys):
    from agent_evolve.cli import main

    code = main(["run", "test_public_knobs:CliProblem",
                 "--budget", "8", "--seed", "1", "--proposer", "random"])
    out = capsys.readouterr().out
    assert code == 0
    assert "best" in out and "evaluations" in out
