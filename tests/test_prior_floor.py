"""The prior floor: an installed prior may bias a locus, never close it.

The defect this file pins is not that a prior can be wrong -- every prior can
be wrong, and the loop's unwind test exists for that. It is that a prior built
from a screen could zero the exact region the screen never measured, after
which the run had no way to sample there and therefore no way to find out.
Measured on the analog venue: 20 of 20 cells never reopened an excluded value,
and the excluded region held the venue's best configurations.

So every installed prior now keeps ``PRIOR_FLOOR`` of each locus's mass on
every declared value. Four things have to be true for that to be a fix rather
than a second mechanism, and each has a test below:

  * the floor actually binds, including on the exclusion-by-omission that a
    model reply writes when it simply leaves a value out;
  * ``PRIOR_FLOOR = 0`` restores the previous arithmetic BYTE for byte, so the
    standing comparators are still comparators;
  * flooring only ever flattens, so a prior admitted under a concentration cap
    is still inside it afterwards;
  * the proposal's own ranking survives -- the floor insures the values it was
    least sure about, it does not erase its opinion of them.
"""

from __future__ import annotations

import json
from typing import Literal

import pytest
from pydantic import BaseModel

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.structure import attribute, crossed_screen
from agent_evolve.policies.weighted_prior import (
    PRIOR_FLOOR,
    WeightedPriorTelemetry,
    floor_weights,
    llm_weighted_prior_proposer,
    statistical_weighted_prior,
)

SPECS = [ObjectiveSpec("energy", "min"), ObjectiveSpec("speed", "max")]


class Arch(BaseModel):
    width: Literal[8, 16]
    regs: bool
    depth: Literal[256, 512, 1024]


TEMPLATE = {"width": 16, "regs": False, "depth": 512}


def _attr(size: int = 8):
    screen = crossed_screen(TEMPLATE, Arch, size=size)
    return attribute(
        [(c, {"energy": 1.0 if (c["width"] == 8 and c["regs"]) else 4.0,
              "speed": 2.0}) for c in screen], SPECS, Arch)


def _proposer(reply: str, tel=None, **kwargs):
    return llm_weighted_prior_proposer(
        lambda _p: reply, objectives=SPECS,
        telemetry=tel or WeightedPriorTelemetry(), **kwargs)


# --- the helper's four properties -------------------------------------------

#: The value the mechanism tests exercise. The DEFAULT is 0.0 (off): the
#: one-factor reading came back a wash (3W/3L, median -0.021 -- the floor
#: rescues wrong boxes and taxes right ones in equal measure), so the floor
#: is an opt-in study knob, not substrate. These tests pin the mechanism a
#: study gets when it opts in, and the byte-identity it gets when it does not.
FLOOR = 0.02


def test_the_default_is_off_and_returns_the_callers_own_tuples_unchanged():
    assert PRIOR_FLOOR == 0.0
    values, weights = (8, 16), (3.0, 0.0)
    assert floor_weights(values, weights, domain=(8, 16)) == (values, weights)


def test_a_floor_of_zero_returns_the_callers_own_tuples_unchanged():
    values, weights = (8, 16), (3.0, 0.0)
    got = floor_weights(values, weights, domain=(8, 16), floor=0.0)
    assert got == (values, weights)


def test_an_entry_already_above_the_floor_is_untouched():
    # Not "equal to": the pre-floor arithmetic has to survive as the same
    # numbers, because every standing cell was measured with them.
    values, weights = (8, 16), (3.0, 1.0)
    assert floor_weights(values, weights, domain=(8, 16),
                         floor=FLOOR) == (values, weights)


def test_the_floor_binds_on_an_excluded_value_and_lands_exactly_on_it():
    values, weights = floor_weights((8, 16), (1.0, 0.0), domain=(8, 16),
                                    floor=FLOOR)
    assert values == (8, 16)
    assert weights[1] == pytest.approx(FLOOR)
    assert sum(weights) == pytest.approx(1.0)


def test_a_value_the_proposal_omitted_is_folded_back_in_at_the_floor():
    # Exclusion by omission is the cheapest exclusion to write and was the one
    # way out of the floor. The declared domain is what the floor is over.
    values, weights = floor_weights((256,), (1.0,),
                                    domain=(256, 512, 1024), floor=FLOOR)
    assert values == (256, 512, 1024)
    assert min(weights) == pytest.approx(FLOOR)
    assert weights[0] > weights[1]


def test_flooring_only_ever_flattens_so_a_concentration_cap_survives():
    # `reguidance` admits a prior only under a max/min ratio cap. Flooring
    # after admission must not be able to push one back out of it.
    def ratio(weights):
        # A zero weight is an infinite ratio, and reading it as anything else
        # is how "the cap held" gets said about a prior that excluded.
        low = min(weights)
        return float("inf") if low <= 0.0 else max(weights) / low

    for weights in [(9.0, 1.0), (1.0, 0.0), (5.0, 4.0, 0.0), (1.0,) * 4]:
        floored = floor_weights(range(len(weights)), weights, floor=FLOOR)[1]
        assert ratio(floored) <= ratio(weights) + 1e-12, weights
        assert ratio(floored) < float("inf"), weights


def test_the_proposals_own_ranking_survives_the_floor():
    # A clip would tie every below-floor value at the floor and throw away the
    # ranking of exactly the values the proposal was least sure about.
    values, weights = floor_weights(
        (0, 1, 2, 3), (60.0, 30.0, 0.6, 0.3), domain=(0, 1, 2, 3),
        floor=FLOOR)
    assert weights[0] > weights[1] > weights[2] > weights[3]
    assert min(weights) >= FLOOR - 1e-12


def test_a_floor_wider_than_an_equal_share_flattens_instead_of_failing():
    # k values cannot each hold more than 1/k. Asking for more is answered
    # with uniform, not with an exception.
    _values, weights = floor_weights((0, 1, 2), (5.0, 1.0, 0.0), floor=0.9)
    assert weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))


# --- the two install paths ---------------------------------------------------

def test_the_admitted_prior_keeps_the_floor_on_every_declared_value():
    reply = json.dumps({"weights": {
        "width": {"values": [8], "weights": [1]},
        "depth": {"values": [256, 512], "weights": [4, 1]}}})
    prior = _proposer(reply, prior_floor=FLOOR)(_attr(), Arch)

    width_values, width_weights = prior.weighted["width"]
    assert width_values == (8, 16), "the excluded value is back in the entry"
    assert min(width_weights) == pytest.approx(FLOOR)

    depth_values, depth_weights = prior.weighted["depth"]
    assert depth_values == (256, 512, 1024)
    assert min(depth_weights) == pytest.approx(FLOOR)
    assert depth_weights[0] > depth_weights[1] > depth_weights[2]

    # The consequence, pinned rather than discovered: a floored locus makes no
    # exclusion claim, so the loop's unwind test has nothing to unwind on it.
    assert prior.allowed["width"] == (8, 16)
    assert prior.allowed["depth"] == (256, 512, 1024)


def test_the_default_admitted_prior_is_the_floorless_one_byte_for_byte():
    reply = json.dumps({"weights": {
        "width": {"values": [8], "weights": [1]},
        "depth": {"values": [256, 512], "weights": [4, 1]}}})
    prior = _proposer(reply)(_attr(), Arch)
    assert prior.weighted["width"] == ((8,), (1.0,))
    assert prior.weighted["depth"] == ((256, 512), (4.0, 1.0))
    assert prior.allowed["width"] == (8,), "the exclusion is back"


def test_the_floor_does_not_admit_a_reply_the_validation_rejected():
    # The floor is applied at INSTALL, never as a repair: an all-zero field is
    # still a whole-reply rejection, not a field flattened to uniform.
    tel = WeightedPriorTelemetry()
    reply = json.dumps(
        {"weights": {"width": {"values": [8, 16], "weights": [0, 0]}}})
    prior = _proposer(reply, tel, prior_floor=FLOOR)(_attr(), Arch)
    assert prior.weighted == {} and tel.out_of_domain == 1


def test_the_statistical_prior_floors_what_a_wide_screen_would_erase():
    # Laplace smoothing keeps a screened level positive, but "positive" is not
    # "reachable": on a wide screen 0.5/(n+1) can fall under the floor, and
    # then the rule prior is making the same unfalsifiable bet the model one
    # was making.
    prior = statistical_weighted_prior(_attr(), Arch, prior_floor=FLOOR)
    for _name, (_values, weights) in prior.weighted.items():
        shares = [w / sum(weights) for w in weights]
        assert min(shares) >= min(FLOOR, 1.0 / len(shares)) - 1e-12


def test_the_default_statistical_prior_is_the_pre_floor_arithmetic():
    attr = _attr()
    floored = statistical_weighted_prior(attr, Arch, prior_floor=FLOOR)
    raw = statistical_weighted_prior(attr, Arch)
    assert set(raw.weighted) == set(floored.weighted)
    for name, (values, weights) in raw.weighted.items():
        summaries = {s.value: (s.nondominated + 0.5) / (s.n + 1.0)
                     for s in attr.for_locus(name)}
        assert values == tuple(summaries)
        assert weights == tuple(summaries.values())


def test_the_floor_is_read_at_call_time_so_a_study_can_turn_it_off():
    # The knob is the module constant, and it is read per call rather than
    # bound at import: a row that runs the pre-floor substrate as its control
    # sets it once and every prior in that process obeys.
    import agent_evolve.policies.weighted_prior as wp

    reply = json.dumps(
        {"weights": {"width": {"values": [8], "weights": [1]}}})
    before = wp.PRIOR_FLOOR
    try:
        wp.PRIOR_FLOOR = 0.0
        assert _proposer(reply)(_attr(), Arch).weighted["width"] == ((8,), (1.0,))
        wp.PRIOR_FLOOR = 0.25
        _values, weights = _proposer(reply)(_attr(), Arch).weighted["width"]
        assert min(weights) == pytest.approx(0.25)
    finally:
        wp.PRIOR_FLOOR = before
