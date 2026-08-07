"""Operator arms earn slots by survival; authored children pass the material
check or their slot falls back to the classical arm. Credit is assigned when
children are measured, retirement is preregistered, and the classical
comparator is structurally always in the run.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.core.authored import authored_artifact
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.infrastructure.authored_runtime import AuthoredRuntime
from agent_evolve.policies.llm_operator import author_operators
from agent_evolve.policies.llm_surrogate import AuthorTelemetry
from agent_evolve.policies.operator_portfolio import (
    OperatorPortfolio,
    VariationArm,
    child_within_material,
    classical_arm,
    segment_arm,
)

SPECS = [ObjectiveSpec("ones", "max")]


class _Candidate(BaseModel):
    genome: list[Literal[0, 1]]


A = {"genome": [0, 0, 0, 1, 1, 1]}
B = {"genome": [1, 1, 1, 0, 0, 0]}
DOMAINS = {f"genome[{i}]": (0, 1) for i in range(6)}


# --- the material check ------------------------------------------------------

def test_children_within_parental_or_declared_material_pass():
    assert child_within_material({"genome": [0, 1, 0, 1, 0, 1]}, A, B, DOMAINS)


def test_out_of_domain_values_and_wrong_shapes_are_rejected():
    assert not child_within_material({"genome": [0, 0, 0, 1, 1, 7]}, A, B, DOMAINS)
    assert not child_within_material({"genome": [0, 0, 0, 1, 1]}, A, B, DOMAINS)
    assert not child_within_material({"dna": [0, 0, 0, 1, 1, 1]}, A, B, DOMAINS)
    assert not child_within_material([0, 0, 0, 1, 1, 1], A, B, DOMAINS)


# --- credit and retirement ---------------------------------------------------

def test_the_portfolio_requires_the_classical_comparator():
    with pytest.raises(ValueError, match="classical"):
        OperatorPortfolio([segment_arm()])


def test_survival_credit_lands_on_the_arm_that_made_the_child():
    portfolio = OperatorPortfolio([classical_arm(), segment_arm()])
    portfolio.record_measured("segment", survived=True)
    portfolio.record_measured("segment", survived=False)
    portfolio.record_measured("classical(fallback)", survived=True)
    assert portfolio.credit.arm("segment").measured == 2
    assert portfolio.credit.arm("segment").survived == 1
    assert portfolio.credit.arm("classical").survived == 1, (
        "fallback children must credit the classical arm, not the failed one"
    )


def test_retirement_fires_exactly_on_the_preregistered_rule():
    portfolio = OperatorPortfolio([classical_arm(), segment_arm()])
    for _ in range(3):
        portfolio.record_measured("segment", survived=False)
    portfolio.record_measured("classical", survived=True)
    assert portfolio.review() == (), "3 measured must not retire"
    portfolio.record_measured("segment", survived=False)
    assert portfolio.review() == ("segment",), "4 measured, 0 survived must"
    assert "segment" not in {arm.name for arm in portfolio.active_arms()}
    assert portfolio.review() == (), "retirement is one-way and once"


def test_assignment_holds_the_classical_floor_and_caps_authored_slots():
    artifact = authored_artifact(
        "operator",
        "def vary(a, b, loci, domains, seed):\n    return dict(a)\n",
        name="op", authored_by="llm")
    portfolio = OperatorPortfolio(
        [classical_arm(), VariationArm(name="op", kind="authored",
                                       artifact=artifact)],
        runtime=AuthoredRuntime(), max_authored_fraction=0.5)
    assigned = portfolio.assign(4, random.Random(1))
    assert assigned[0].kind == "classical", "slot 0 is the floor"
    assert sum(1 for arm in assigned if arm.kind == "authored") <= 2


# --- authored construction through the runtime -------------------------------

def test_an_authored_operator_constructs_in_domain_children():
    source = (
        "def vary(a, b, loci, domains, seed):\n"
        "    import random\n"
        "    r = random.Random(seed)\n"
        "    child = {'genome': list(a['genome'])}\n"
        "    for i in range(len(child['genome'])):\n"
        "        if r.random() < 0.5:\n"
        "            child['genome'][i] = b['genome'][i]\n"
        "    return child\n"
    )
    artifact = authored_artifact("operator", source, name="mix",
                                 authored_by="llm")
    portfolio = OperatorPortfolio(
        [classical_arm(),
         VariationArm(name="mix", kind="authored", artifact=artifact)],
        runtime=AuthoredRuntime())
    kids, origins = portfolio.construct_generation(
        [(A, B)] * 4, _Candidate, None, random.Random(2), generation=1)
    assert len(kids) == 4
    assert any(origin == "mix" for origin in origins), origins
    assert portfolio.counters.authored_children >= 1


def test_a_rule_breaking_authored_child_falls_back_to_classical_and_counts():
    source = (
        "def vary(a, b, loci, domains, seed):\n"
        "    return {'genome': [7, 7, 7, 7, 7, 7]}\n"
    )
    artifact = authored_artifact("operator", source, name="rogue",
                                 authored_by="llm")
    portfolio = OperatorPortfolio(
        [classical_arm(),
         VariationArm(name="rogue", kind="authored", artifact=artifact)],
        runtime=AuthoredRuntime())
    kids, origins = portfolio.construct_generation(
        [(A, B)] * 3, _Candidate, None, random.Random(3), generation=1)
    assert len(kids) == 3
    assert portfolio.counters.rejected_out_of_domain >= 1
    assert not any(origin == "rogue" for origin in origins)
    assert all(set(kid["genome"]) <= {0, 1} for kid in kids), (
        "an out-of-material child leaked into the generation"
    )


# --- authoring seam ----------------------------------------------------------

def test_author_operators_accepts_multiple_fenced_blocks():
    reply = (
        "```python\ndef vary(a, b, loci, domains, seed):\n    return dict(a)\n```\n"
        "```python\ndef vary(a, b, loci, domains, seed):\n    return dict(b)\n```\n"
    )
    tel = AuthorTelemetry()
    artifacts = author_operators(lambda _p: reply, objectives=SPECS,
                                 schema_text="s", telemetry=tel)
    assert len(artifacts) == 2 and tel.accepted == 2
    assert {a.name for a in artifacts} == {"llm_op1", "llm_op2"}


def test_author_operators_gates_each_block_independently():
    reply = (
        "```python\nimport os\ndef vary(a, b, loci, domains, seed):\n"
        "    return dict(a)\n```\n"
        "```python\ndef vary(a, b, loci, domains, seed):\n    return dict(b)\n```\n"
    )
    tel = AuthorTelemetry()
    artifacts = author_operators(lambda _p: reply, objectives=SPECS,
                                 schema_text="s", telemetry=tel)
    assert len(artifacts) == 1 and tel.forbidden_import == 1


# --- end to end --------------------------------------------------------------

class _Problem:
    candidate_model = _Candidate
    objectives = tuple(SPECS)

    def __init__(self) -> None:
        self.calls = 0

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ({"genome": [0, 0, 0, 0, 0, 0]},)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return tuple(config["genome"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        self.calls += 1
        return {"ones": float(sum(artifact))}


def test_rule_portfolio_end_to_end_respects_the_budget():
    problem = _Problem()
    result = optimize(problem, budget=20, seed=13, authorship="operators")
    assert problem.calls <= 20 and result.evaluations <= 20
    summaries = [h["portfolio"] for h in result.history if "portfolio" in h]
    assert summaries, "no portfolio summary reached the history"
    arms = summaries[-1]["arms"]
    assert "classical" in arms and "segment" in arms
    assert sum(measured for _p, measured, _s in arms.values()) > 0
    rows = [m for m in result.telemetry.mechanisms
            if m.mechanism == "operator_portfolio"]
    assert rows and rows[0].counters["classical__measured"] > 0


def test_llm_operators_end_to_end_with_a_canned_author(monkeypatch):
    reply = (
        "```python\n"
        "def vary(a, b, loci, domains, seed):\n"
        "    import random\n"
        "    r = random.Random(seed)\n"
        "    child = {'genome': [max(x, y) for x, y in\n"
        "             zip(a['genome'], b['genome'])]}\n"
        "    i = r.randrange(len(child['genome']))\n"
        "    child['genome'][i] = 1\n"
        "    return child\n"
        "```\n"
    )

    def _canned_completion_for(model, settings=None, **kwargs):
        return lambda prompt: reply

    monkeypatch.setattr(
        "agent_evolve.integrations.completion.completion_for",
        _canned_completion_for,
    )
    problem = _Problem()
    result = optimize(problem, budget=20, seed=14, proposer="llm",
                      authorship="operators-llm")
    assert problem.calls <= 20
    mechanisms = {m.mechanism: m for m in result.telemetry.mechanisms}
    assert mechanisms["operator_author"].counters["accepted"] == 1
    portfolio = mechanisms["operator_portfolio"].counters
    authored_measured = [v for k, v in portfolio.items()
                         if k.startswith("llm_op") and k.endswith("__measured")]
    assert authored_measured and sum(authored_measured) > 0, (
        "the authored arm never got a measured child"
    )
