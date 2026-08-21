"""The graded prior: hard restriction is its 0/1 special case, provably."""

from __future__ import annotations

import json
import random
from typing import Literal

import pytest
from pydantic import BaseModel

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.genetic import (
    DomainRestriction,
    Locus,
    mutate,
    uniform_candidate,
)
from agent_evolve.policies.structure import attribute, crossed_screen
from agent_evolve.policies.weighted_prior import (
    COMMITTED_PROMPT,
    PROMPT,
    WeightedPriorTelemetry,
    WeightedRestriction,
    llm_weighted_prior_proposer,
    statistical_weighted_prior,
)

SPECS = [ObjectiveSpec("energy", "min"), ObjectiveSpec("speed", "max")]


class Arch(BaseModel):
    width: Literal[8, 16]
    regs: bool
    depth: Literal[256, 512, 1024]


TEMPLATE = {"width": 16, "regs": False, "depth": 512}


# --- the special-case theorem ------------------------------------------------

def test_hard_weights_reproduce_the_hard_restriction_draw_for_draw():
    # WeightedRestriction.hard() must not merely agree in distribution with
    # DomainRestriction -- it must consume the SAME rng stream, or swapping the
    # generic form in for the special case would silently change every run.
    allowed = {"width": [8], "depth": [256, 512]}
    hard = DomainRestriction(allowed)
    graded = WeightedRestriction.hard(allowed)

    a, b = random.Random(9), random.Random(9)
    for _ in range(50):
        assert (
            uniform_candidate(TEMPLATE, Arch, rng=a, restriction=hard)
            == uniform_candidate(TEMPLATE, Arch, rng=b, restriction=graded)
        )
    a, b = random.Random(11), random.Random(11)
    for _ in range(50):
        assert (
            mutate(TEMPLATE, Arch, rng=a, restriction=hard)
            == mutate(TEMPLATE, Arch, rng=b, restriction=graded)
        )


def test_uniform_positive_weights_report_none():
    # The mechanism behind the theorem: equal weights mean "uniform", and
    # uniform means the sampler takes the untouched r.choice path.
    assert WeightedRestriction(
        {"width": ((8, 16), (2.5, 2.5))}
    ).weights_for(Locus("width"), (8, 16)) is None

    graded = WeightedRestriction({"width": ((8, 16), (3.0, 1.0))})
    assert graded.weights_for(Locus("width"), (8, 16)) == (3.0, 1.0)
    assert graded.weights_for(Locus("depth"), (256,)) is None


# --- sampling behaviour ------------------------------------------------------

def test_weighted_draws_stay_inside_the_positive_support():
    prior = WeightedRestriction({
        "width": ((8, 16), (3.0, 1.0)),
        "depth": ((256, 512, 1024), (1.0, 0.0, 2.0)),
    })
    r = random.Random(3)
    for _ in range(100):
        c = uniform_candidate(TEMPLATE, Arch, rng=r, restriction=prior)
        assert c["width"] in (8, 16)
        assert c["depth"] in (256, 1024), "a zero weight must exclude the value"


def test_weights_bias_the_draw():
    prior = WeightedRestriction({"width": ((8, 16), (9.0, 1.0))})
    r = random.Random(4)
    drawn = [
        uniform_candidate(TEMPLATE, Arch, rng=r, restriction=prior)["width"]
        for _ in range(200)
    ]
    assert drawn.count(8) > 140, (
        f"9:1 weights drew width=8 only {drawn.count(8)}/200 times; uniform "
        "would sit near 100"
    )


def test_narrows_only_never_empties_and_counts_the_miss():
    prior = WeightedRestriction({"width": ((32, 64), (1.0, 1.0))})
    assert prior.narrow(Locus("width"), (8, 16)) == (8, 16)
    assert prior.misses == ["width"]
    # and a missed prior does not get to bias the domain it failed to narrow
    assert prior.weights_for(Locus("width"), (8, 16)) is None


def test_allowed_reports_the_positive_support():
    prior = WeightedRestriction({"depth": ((256, 512, 1024), (1.0, 0.0, 2.0))})
    assert prior.allowed == {"depth": (256, 1024)}


def test_all_zero_weights_are_rejected_at_construction():
    with pytest.raises(ValueError, match="weight"):
        WeightedRestriction({"width": ((8, 16), (0.0, 0.0))})
    with pytest.raises(ValueError, match="non-negative"):
        WeightedRestriction({"width": ((8, 16), (1.0, -2.0))})
    with pytest.raises(ValueError, match="values"):
        WeightedRestriction({"width": ((8, 16), (1.0,))})


# --- the credential-free comparator -----------------------------------------

def _and_attr():
    screen = crossed_screen(TEMPLATE, Arch, size=4, rng=random.Random(1))
    return attribute(
        [(c, {"energy": 1.0 if (c["width"] == 8 and c["regs"]) else 4.0,
              "speed": 2.0}) for c in screen], SPECS, Arch)


def test_statistical_weighted_prior_grades_what_the_screen_separates():
    prior = statistical_weighted_prior(_and_attr(), Arch)
    values, weights = prior.weighted["width"]
    by = dict(zip(values, weights))
    # graded, not gambled: the losing level keeps mass instead of being erased
    assert by[8] > by[16] > 0.0
    assert "regs" in prior.weighted
    assert "depth" not in prior.weighted, "one screened level cannot separate"


def test_statistical_weighted_prior_abstains_on_a_flat_screen():
    flat = [dict(TEMPLATE) for _ in range(4)]
    attr = attribute([(c, {"energy": 1.0, "speed": 1.0}) for c in flat], SPECS, Arch)
    assert statistical_weighted_prior(attr, Arch).weighted == {}


# --- the model-driven proposer: what it accepts, what it refuses to repair ---

def _proposer(reply, tel=None):
    return llm_weighted_prior_proposer(
        lambda _p: reply, objectives=SPECS,
        telemetry=tel or WeightedPriorTelemetry())


def test_a_well_formed_weighted_prior_is_applied():
    tel = WeightedPriorTelemetry()
    reply = json.dumps({
        "weights": {"width": {"values": [8, 16], "weights": [3, 1]}},
        "free": ["regs", "depth"],
    })
    prior = _proposer(reply, tel)(_and_attr(), Arch)
    assert prior.weighted["width"] == ((8, 16), (3.0, 1.0))
    assert tel.calls == 1 and tel.restricted_loci == 1


def test_values_outside_the_declared_domain_reject_the_whole_reply():
    tel = WeightedPriorTelemetry()
    reply = json.dumps(
        {"weights": {"width": {"values": [8, 32], "weights": [1, 1]}}})
    prior = _proposer(reply, tel)(_and_attr(), Arch)
    assert prior.weighted == {} and tel.out_of_domain == 1


def test_negative_or_non_numeric_weights_reject_the_whole_reply():
    for weights in ([1, -2], [1, "high"], [1, None]):
        tel = WeightedPriorTelemetry()
        reply = json.dumps(
            {"weights": {"width": {"values": [8, 16], "weights": weights}}})
        prior = _proposer(reply, tel)(_and_attr(), Arch)
        assert prior.weighted == {} and tel.out_of_domain == 1, weights


def test_all_zero_weights_reject_the_whole_reply():
    tel = WeightedPriorTelemetry()
    reply = json.dumps(
        {"weights": {"width": {"values": [8, 16], "weights": [0, 0]}}})
    prior = _proposer(reply, tel)(_and_attr(), Arch)
    assert prior.weighted == {} and tel.out_of_domain == 1


def test_mismatched_lists_reject_the_whole_reply():
    tel = WeightedPriorTelemetry()
    reply = json.dumps(
        {"weights": {"width": {"values": [8, 16], "weights": [1]}}})
    prior = _proposer(reply, tel)(_and_attr(), Arch)
    assert prior.weighted == {} and tel.unparseable == 1


def test_a_reply_that_authors_a_candidate_is_refused_and_counted():
    tel = WeightedPriorTelemetry()
    reply = json.dumps({"weights": {"width": 8, "regs": True, "depth": 256}})
    prior = _proposer(reply, tel)(_and_attr(), Arch)
    assert prior.weighted == {} and tel.wrote_candidate == 1


def test_even_full_domain_weights_are_recorded_as_free():
    tel = WeightedPriorTelemetry()
    reply = json.dumps(
        {"weights": {"width": {"values": [8, 16], "weights": [2, 2]}}})
    prior = _proposer(reply, tel)(_and_attr(), Arch)
    assert prior.weighted == {} and tel.empty == 1


def test_uneven_full_domain_weights_are_a_real_prior():
    # Grading the whole domain is exactly the new expressive power: nothing is
    # excluded, but the draws lean where the proposer says.
    tel = WeightedPriorTelemetry()
    reply = json.dumps(
        {"weights": {"width": {"values": [8, 16], "weights": [3, 1]}}})
    prior = _proposer(reply, tel)(_and_attr(), Arch)
    assert prior.weighted["width"] == ((8, 16), (3.0, 1.0))
    assert prior.allowed["width"] == (8, 16), "support is untouched"


def test_unparseable_replies_fall_back_to_the_unguided_distribution():
    tel = WeightedPriorTelemetry()
    prior = _proposer("I would weight things sensibly.", tel)(_and_attr(), Arch)
    assert prior.weighted == {} and tel.unparseable == 1


def test_a_provider_error_is_counted_and_never_raised():
    tel = WeightedPriorTelemetry()

    def boom(_prompt):
        raise RuntimeError("provider down")

    prior = llm_weighted_prior_proposer(
        boom, objectives=SPECS, telemetry=tel)(_and_attr(), Arch)
    assert prior.weighted == {} and tel.errors == 1


# --- the two closing clauses: the shipped one, and the committed variant ----

#: The shipped prompt, copied out literally. Every sealed prior row was
#: measured under these bytes; a variant that edits them instead of adding
#: itself beside them would silently redefine the arm it is compared against.
CAUTIOUS_PROMPT_TODAY = """The search has spent {n} evaluations on a screening design.

OBJECTIVES: {goals}

SEARCH SPACE (declared domains):
{schema}

SCREEN EVIDENCE (per locus value, within-screen):
{screen}

Read the table as evidence about THIS evaluator. It may or may not match how
such problems usually behave; where they disagree, the table wins.

Propose a WEIGHTED sampling prior. For any locus the evidence (or the
parameter's meaning) separates, list the values worth sampling and a
non-negative weight for each: higher weight, more draws. A value you omit gets
weight zero and is not sampled; a locus you omit stays free. You are not
choosing a candidate and must not propose one. Over-restricting a
frontier-spreading parameter destroys diversity: leave a locus free unless you
have a reason.

Reply with ONLY this JSON shape and no other text:
{{"weights": {{"<param>": {{"values": [...], "weights": [...]}}}}, "free": ["<param>", ...]}}"""


def test_the_cautious_prompt_has_not_moved_by_one_character():
    assert PROMPT == CAUTIOUS_PROMPT_TODAY


def test_cautious_is_the_default_and_renders_the_same_bytes():
    seen: list[str] = []

    def capture(prompt):
        seen.append(prompt)
        return json.dumps({"weights": {}, "free": ["width"]})

    attr = _and_attr()
    llm_weighted_prior_proposer(capture, objectives=SPECS)(attr, Arch)
    llm_weighted_prior_proposer(
        capture, objectives=SPECS, style="cautious")(attr, Arch)
    assert seen[0] == seen[1], "naming the default must not change the ask"


def test_the_committed_variant_swaps_the_closing_clause_and_nothing_else():
    head = "choosing a candidate and must not propose one. "
    tail = "\n\nReply with ONLY this JSON shape"
    assert COMMITTED_PROMPT.split(head)[0] == PROMPT.split(head)[0]
    assert (COMMITTED_PROMPT[COMMITTED_PROMPT.index(tail):]
            == PROMPT[PROMPT.index(tail):])
    # the caution is gone, and what replaced it asks for commitment
    assert "leave a locus free unless you" not in COMMITTED_PROMPT
    assert "Over-restricting" not in COMMITTED_PROMPT
    assert "give weights proportional to that evidence" in COMMITTED_PROMPT
    assert "hedging every parameter as free discards the screen" in (
        COMMITTED_PROMPT)


def test_the_committed_style_is_what_the_run_actually_sends():
    seen: list[str] = []

    def capture(prompt):
        seen.append(prompt)
        return json.dumps(
            {"weights": {"width": {"values": [8, 16], "weights": [3, 1]}}})

    propose = llm_weighted_prior_proposer(
        capture, objectives=SPECS, style="committed")
    prior = propose(_and_attr(), Arch)
    assert "hedging every parameter as free" in seen[0]
    assert "leave a locus free unless you" not in seen[0]
    # the screen still reaches the model, and the parse is the shared one
    assert "non-dominated" in seen[0] and "must not propose one" in seen[0]
    assert prior.weighted["width"] == ((8, 16), (3.0, 1.0))
    assert propose.style == "committed"


def test_the_committed_style_validates_a_reply_exactly_as_the_cautious_one():
    reply = json.dumps(
        {"weights": {"width": {"values": [8, 32], "weights": [1, 1]}}})
    for style in ("cautious", "committed"):
        tel = WeightedPriorTelemetry()
        prior = llm_weighted_prior_proposer(
            lambda _p: reply, objectives=SPECS, telemetry=tel,
            style=style)(_and_attr(), Arch)
        assert prior.weighted == {} and tel.out_of_domain == 1, style


def test_an_unknown_prior_style_is_refused_by_name():
    with pytest.raises(ValueError, match="style"):
        llm_weighted_prior_proposer(
            lambda _p: "{}", objectives=SPECS, style="hedged")


def test_the_prompt_carries_the_screen_and_never_asks_for_a_candidate():
    seen = {}

    def capture(prompt):
        seen["p"] = prompt
        return json.dumps({"weights": {}, "free": ["width", "regs", "depth"]})

    llm_weighted_prior_proposer(capture, objectives=SPECS)(_and_attr(), Arch)
    assert "non-dominated" in seen["p"] and "must not propose one" in seen["p"]
    assert "energy (min)" in seen["p"] and "speed (max)" in seen["p"]
    assert "weight" in seen["p"]
