"""The semantic channel: meanings the author wrote must reach every prompt.

The structural channel (names, domains, directions) was always faithful;
these tests pin the SEMANTIC one -- objective descriptions, pydantic Field
descriptions, problem and evaluate docstrings -- and its graceful absence:
a problem that documents nothing gets the structural card, never a crash.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.policies.semantics import (
    domain_card,
    objective_lines,
    parameter_lines,
)

SPECS = [
    ObjectiveSpec("energy", "min", description="joules per inference"),
    ObjectiveSpec("speed", "max"),
]


class Arch(BaseModel):
    """A toy accelerator: width times depth against an energy budget."""

    width: Literal[8, 16] = Field(description="datapath width in bits")
    regs: bool
    depth: Literal[256, 512, 1024]


class _Problem:
    candidate_model = Arch
    objectives = tuple(SPECS)

    def seeds(self):
        return ({"width": 16, "regs": False, "depth": 512},)

    def validate(self, config):
        return ValidationOutcome(ok=True)

    def materialize(self, config):
        return dict(config)

    def evaluate(self, artifact):
        """Runs the cycle-accurate simulator on the mapped design."""

        return {"energy": 1.0, "speed": 2.0}


def test_objective_lines_carry_descriptions_and_survive_their_absence():
    lines = objective_lines(SPECS)
    assert lines[0] == "energy (min) -- joules per inference"
    assert lines[1] == "speed (max)", "an absent description adds nothing"


def test_parameter_lines_harvest_field_descriptions():
    lines = parameter_lines(Arch)
    by_field = {line.split(":")[0]: line for line in lines}
    assert "datapath width in bits" in by_field["width"]
    assert by_field["regs"] == "regs: one of [False, True]"


def test_domain_card_assembles_every_available_source():
    card = domain_card(_Problem())
    assert "A toy accelerator" in card, "the problem docstring is the prose"
    assert "cycle-accurate simulator" in card, "evaluate.__doc__ is EVALUATION"
    assert "joules per inference" in card
    assert "datapath width in bits" in card
    assert "one of [8, 16]" in card


def test_domain_card_degrades_to_the_structural_rendering():
    class Bare:
        candidate_model = Arch
        objectives = (ObjectiveSpec("ones", "max"),)

    Bare.__doc__ = None
    card = domain_card(Bare())
    assert "ones (max)" in card and "width" in card


def test_prompts_carry_the_card_and_the_descriptions():
    from agent_evolve.policies.llm_prior import llm_prior_proposer
    from agent_evolve.policies.structure import attribute, crossed_screen
    import random

    screen = crossed_screen({"width": 16, "regs": False, "depth": 512},
                            Arch, size=4, rng=random.Random(1))
    attr = attribute([(c, {"energy": 1.0, "speed": 2.0}) for c in screen],
                     SPECS, Arch)
    seen = {}

    def capture(prompt):
        seen["p"] = prompt
        return json.dumps({"restrictions": {}, "free": []})

    llm_prior_proposer(capture, objectives=SPECS,
                       domain_context=domain_card(_Problem()))(attr, Arch)
    assert "A toy accelerator" in seen["p"], "the card must lead the prompt"
    assert "joules per inference" in seen["p"]
    assert "datapath width in bits" in seen["p"]


def test_an_empty_card_leaves_the_prompt_untouched():
    from agent_evolve.policies.llm_chooser import llm_chooser

    seen = {}

    def capture(prompt):
        seen["p"] = prompt
        return "[]"

    chooser = llm_chooser(capture, objectives=[ObjectiveSpec("ones", "max")],
                          budget=8)
    chooser([({"genome": [0, 1]}, 0.0)], 1, None)
    assert "\n\n\nOBJECTIVES" not in seen["p"], (
        "an absent card must not leave a hole in the prompt"
    )
    assert "ones (max)" in seen["p"]