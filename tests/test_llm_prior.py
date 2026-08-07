"""The model-driven prior: what it accepts, and what it refuses to repair."""

from __future__ import annotations

import json
import random
from typing import Literal

from pydantic import BaseModel

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.llm_prior import PriorTelemetry, llm_prior_proposer
from agent_evolve.policies.structure import attribute, crossed_screen

SPECS = [ObjectiveSpec("energy", "min"), ObjectiveSpec("speed", "max")]


class Arch(BaseModel):
    width: Literal[8, 16]
    regs: bool
    depth: Literal[256, 512, 1024]


TEMPLATE = {"width": 16, "regs": False, "depth": 512}


def _attr():
    screen = crossed_screen(TEMPLATE, Arch, size=4, rng=random.Random(1))
    return attribute(
        [(c, {"energy": 1.0 if (c["width"] == 8 and c["regs"]) else 4.0,
              "speed": 2.0}) for c in screen], SPECS, Arch)


def _proposer(reply, tel=None):
    return llm_prior_proposer(lambda _p: reply, objectives=SPECS,
                              telemetry=tel or PriorTelemetry())


def test_a_well_formed_restriction_is_applied():
    tel = PriorTelemetry()
    reply = json.dumps({"restrictions": {"width": [8], "regs": [True]},
                        "free": ["depth"]})
    prior = _proposer(reply, tel)(_attr(), Arch)
    assert prior.allowed == {"width": [8], "regs": [True]}
    assert tel.calls == 1 and tel.restricted_loci == 2


def test_values_outside_the_declared_domain_reject_the_whole_reply():
    # Repairing would substitute the harness's prior for the model's and the
    # measurement would attribute the wrong decision.
    tel = PriorTelemetry()
    reply = json.dumps({"restrictions": {"width": [8, 32]}, "free": []})
    prior = _proposer(reply, tel)(_attr(), Arch)
    assert prior.allowed == {}
    assert tel.out_of_domain == 1


def test_a_reply_that_authors_a_candidate_is_refused_and_counted():
    tel = PriorTelemetry()
    reply = json.dumps({"restrictions": {"width": 8, "regs": True, "depth": 256}})
    prior = _proposer(reply, tel)(_attr(), Arch)
    assert prior.allowed == {}
    assert tel.wrote_candidate == 1


def test_unparseable_replies_fall_back_to_the_unguided_distribution():
    tel = PriorTelemetry()
    prior = _proposer("I would keep everything, honestly.", tel)(_attr(), Arch)
    assert prior.allowed == {}
    assert tel.unparseable == 1


def test_a_provider_error_is_counted_and_never_raised():
    tel = PriorTelemetry()

    def boom(_prompt):
        raise RuntimeError("provider down")

    prior = llm_prior_proposer(boom, objectives=SPECS, telemetry=tel)(_attr(), Arch)
    assert prior.allowed == {} and tel.errors == 1


def test_restricting_to_the_whole_domain_is_recorded_as_free():
    tel = PriorTelemetry()
    reply = json.dumps({"restrictions": {"width": [8, 16]}, "free": ["regs", "depth"]})
    prior = _proposer(reply, tel)(_attr(), Arch)
    assert prior.allowed == {} and tel.empty == 1


def test_the_prompt_carries_the_screen_and_never_asks_for_a_candidate():
    seen = {}

    def capture(prompt):
        seen["p"] = prompt
        return json.dumps({"restrictions": {}, "free": ["width", "regs", "depth"]})

    llm_prior_proposer(capture, objectives=SPECS)(_attr(), Arch)
    assert "non-dominated" in seen["p"] and "must not propose one" in seen["p"]
    assert "energy (min)" in seen["p"] and "speed (max)" in seen["p"]
