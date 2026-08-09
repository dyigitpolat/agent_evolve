"""Model-authored initialization: validated value-by-value, seeds first,
rejects cost a slot not the run."""

from __future__ import annotations

import json
from typing import Any, Dict, Literal, Mapping, Sequence

from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.policies.llm_init import InitTelemetry, author_initial_population


class _Candidate(BaseModel):
    genome: list[Literal[0, 1]]


TEMPLATE = {"genome": [0, 0, 0, 0, 0, 0]}


def test_out_of_domain_and_wrong_shape_members_are_rejected_individually():
    reply = json.dumps([
        {"genome": [1, 1, 1, 0, 0, 1]},          # good
        {"genome": [1, 2, 0, 0, 0, 0]},          # 2 not in domain
        {"dna": [1, 1, 1, 1, 1, 1]},             # wrong field
        {"genome": [1, 1, 1]},                   # wrong length
        {"genome": [0, 1, 0, 1, 0, 1]},          # good
    ])
    tel = InitTelemetry()
    members = author_initial_population(
        lambda _p: reply, candidate_model=_Candidate, template=TEMPLATE,
        k=5, domain_context="card", telemetry=tel)
    assert [m["genome"] for m in members] == [[1, 1, 1, 0, 0, 1],
                                              [0, 1, 0, 1, 0, 1]]
    assert tel.accepted == 2 and tel.rejected_out_of_domain == 1
    assert tel.rejected_shape == 2


def test_unparseable_reply_yields_nothing_and_is_counted():
    tel = InitTelemetry()
    assert author_initial_population(
        lambda _p: "just start anywhere", candidate_model=_Candidate,
        template=TEMPLATE, k=4, domain_context="c", telemetry=tel) == []
    assert tel.unparseable == 1


class _Problem:
    candidate_model = _Candidate
    objectives = (ObjectiveSpec("ones", "max"),)

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


def test_init_llm_end_to_end_seeds_first_then_proposals(monkeypatch):
    proposed = [[1, 1, 1, 1, 1, 0], [0, 1, 1, 0, 1, 1], [1, 0, 0, 1, 1, 1]]

    def _canned(model, settings=None, **kwargs):
        return lambda prompt: json.dumps([{"genome": g} for g in proposed])

    monkeypatch.setattr(
        "agent_evolve.integrations.completion.completion_for", _canned)
    problem = _Problem()
    result = optimize(problem, budget=12, seed=6, proposer="llm",
                      authorship="init-llm")
    # evaluation order: the caller's seed first, then the model's members
    assert problem.stream[0] == TEMPLATE["genome"]
    assert problem.stream[1:4] == proposed, (
        "proposed members must form the initial population after the seed"
    )
    mechanisms = {m.mechanism: m for m in result.telemetry.mechanisms}
    assert mechanisms["init_author"].counters["accepted"] == 3
