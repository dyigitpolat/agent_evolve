"""The revision channel: what it fires on, what it refuses, what it installs.

Everything here runs offline against fake completion callables. The channel is
a pure consumer of one -- it never sees a problem, an evaluator or a cache --
so a fake reply is the whole of its input, and the arithmetic it does to that
reply (normalize, mix, cap) is checked exactly rather than by tolerance.

The two properties damping is FOR are tested as properties: below 1 no
revision can introduce an exclusion, and a convex mix of two vectors each
within the concentration cap stays within it (the mediant inequality), so the
cap survives however many revisions land.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.policies.measurement_evidence import (
    evidence_digest, render_elite_table, render_measurement_evidence)
from agent_evolve.policies.reguidance import (
    ELITE_TABLE_TITLE, EVIDENCE_VERSION, MECHANISM_VERSION,
    MECHANISM_VERSION_IMMIGRANTS, TILT_CAP, Reguidance, ReguidanceTelemetry)
from agent_evolve.policies.weighted_prior import WeightedRestriction
from agent_evolve.session.authorship import (AuthorshipConfig, build_authorship)
from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop

GENOME = 8
TEMPLATE: Dict[str, Any] = {"genome": [0] * GENOME, "mode": "a"}
OBJECTIVES = (ObjectiveSpec(name="ones", goal="max"),
              ObjectiveSpec(name="edges", goal="min"))


class _Candidate(BaseModel):
    genome: list[Literal[0, 1]]
    mode: Literal["a", "b", "c"]


class _Problem:
    """A cheap two-objective toy: one-max against a switching penalty."""

    candidate_model = _Candidate
    objectives = OBJECTIVES

    def __init__(self) -> None:
        self.stream: List[Dict[str, Any]] = []

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return (dict(TEMPLATE, genome=list(TEMPLATE["genome"])),)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return (tuple(config["genome"]), config["mode"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        genome, mode = artifact
        self.stream.append({"genome": list(genome), "mode": mode})
        bonus = {"a": 0.0, "b": 0.5, "c": 1.0}[mode]
        return {
            "ones": float(sum(genome)) + bonus,
            "edges": float(sum(1 for i in range(len(genome) - 1)
                               if genome[i] != genome[i + 1])),
        }


def _rows(*specs) -> List[tuple]:
    """``(config, objectives)`` rows in measurement order, written by hand."""

    out = []
    for genome, mode, ones, edges in specs:
        out.append(({"genome": list(genome), "mode": mode},
                    {"ones": float(ones), "edges": float(edges)}))
    return out


ROWS = _rows(
    ([1] * GENOME, "a", 8, 0),                       # rank-0: holds mode "a"
    ([0] * GENOME, "b", 0, 0),                       # rank-0 on edges alone
    ([1, 0] * (GENOME // 2), "c", 4, 7),
)


def _policy(reply, **kwargs) -> Reguidance:
    """A policy over the toy schema whose model always says *reply*."""

    prompts: List[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return reply(prompt) if callable(reply) else reply

    policy = Reguidance(
        complete,
        objectives=OBJECTIVES,
        candidate_model=_Candidate,
        template=dict(TEMPLATE),
        every=kwargs.pop("every", 1),
        min_rows=kwargs.pop("min_rows", 1),
        **kwargs,
    )
    policy.prompts = prompts                    # type: ignore[attr-defined]
    return policy


def _revise(policy: Reguidance, rows=ROWS, *, restriction=None, charges=None,
            gen: int = 1, population=()):
    return policy.maybe_revise(
        rows=rows, population=population, specs=OBJECTIVES,
        restriction=restriction,
        charges=len(rows) if charges is None else charges, gen=gen)


def _reply(weights=None, free=(), immigrants=None) -> str:
    body: Dict[str, Any] = {
        "weights": {name: {"values": list(values), "weights": list(ws)}
                    for name, (values, ws) in dict(weights or {}).items()},
        "free": list(free),
    }
    if immigrants is not None:
        body["immigrants"] = list(immigrants)
    return json.dumps(body)


# --- u1: off is off ---------------------------------------------------------

def test_the_default_loop_and_the_default_authorship_carry_no_revision_channel():
    assert GeneticConfig().reguidance is None, (
        "the revision channel is on by default; the pre-seam loop is no "
        "longer the default path"
    )
    policies = build_authorship(AuthorshipConfig(), complete=lambda _p: "{}",
                                objectives=list(OBJECTIVES),
                                init_template=dict(TEMPLATE), budget=64,
                                population_size=8)
    assert policies.reguidance is None and policies.reguidance_author is None


def test_the_credential_free_byte_fossil_still_holds():
    """Run the fossil file itself: 'off' means byte-identical, not similar."""

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_fossil_stream.py",
         "-p", "no:cacheprovider", "-q"],
        cwd=root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]


# --- u2: the declared cadence, and only it ----------------------------------

def test_events_fire_on_the_declared_cadence_and_stop_at_the_cap():
    policy = _policy(_reply({"mode": (["a", "b", "c"], [3, 1, 1])}),
                     every=16, min_rows=1, max_events=3)
    problem = _Problem()
    run_genetic_loop(
        problem=problem,
        config=GeneticConfig(population_size=8, offspring_per_generation=6,
                             generations=40, seed=11, evaluation_budget=64,
                             reguidance=policy),
    )
    fired = [event["at_charges"] for event in policy.telemetry.events]
    assert 0 < len(fired) <= 3, f"the event cap was not honoured: {fired}"
    assert len(set(fired)) == len(fired), f"two events at one charge: {fired}"
    assert fired[0] >= 16, "an event fired before the declared cadence"
    assert all(b - a >= 16 for a, b in zip(fired, fired[1:])), (
        f"events fired closer together than the declared cadence: {fired}"
    )
    assert policy.telemetry.calls == len(fired)


def test_nothing_fires_before_the_cadence_and_no_call_is_made():
    policy = _policy(_reply({"mode": (["a", "b", "c"], [3, 1, 1])}), every=64)
    outcome = _revise(policy, charges=8)
    assert outcome.restriction is None and outcome.note is None
    assert policy.telemetry.calls == 0, (
        "a checkpoint that is not due spent a model call anyway"
    )


def test_a_checkpoint_with_too_few_new_rows_does_not_fire():
    policy = _policy(_reply({"mode": (["a", "b", "c"], [3, 1, 1])}),
                     every=1, min_rows=8)
    assert _revise(policy, charges=32).note is None
    assert policy.telemetry.calls == 0


# --- u3: the damping arithmetic, exactly ------------------------------------

def test_an_absent_field_mixes_the_reply_into_a_uniform_base():
    policy = _policy(_reply({"mode": (["a", "b", "c"], [4, 1, 1])}),
                     damping=0.5)
    outcome = _revise(policy)
    assert outcome.note["admitted"] is True
    values, weights = policy.installed["mode"]
    assert values == ("a", "b", "c")
    # base uniform (1/3, 1/3, 1/3); proposal normalized (4/6, 1/6, 1/6);
    # mixed at 0.5 = (1/2, 1/4, 1/4).
    assert weights == pytest.approx((0.5, 0.25, 0.25), abs=1e-12)
    assert isinstance(outcome.restriction, WeightedRestriction)


def test_a_freed_field_moves_halfway_back_to_uniform():
    policy = _policy(lambda prompt: _reply({"mode": (["a", "b", "c"], [4, 1, 1])})
                     if policy.telemetry.calls == 1 else _reply(free=["mode"]),
                     damping=0.5)
    _revise(policy, charges=8)
    assert policy.installed["mode"][1] == pytest.approx((0.5, 0.25, 0.25),
                                                        abs=1e-12)
    later = ROWS + _rows(([1] * GENOME, "b", 9, 0))
    _revise(policy, rows=later, charges=16, gen=2)
    # base (1/2, 1/4, 1/4) mixed with uniform at 0.5.
    assert policy.installed["mode"][1] == pytest.approx(
        (0.5 * 0.5 + 0.5 / 3.0, 0.25 * 0.5 + 0.5 / 3.0, 0.25 * 0.5 + 0.5 / 3.0),
        abs=1e-12)


def test_a_field_the_reply_does_not_mention_is_left_exactly_as_it_was():
    policy = _policy(lambda prompt: _reply({"mode": (["a", "b", "c"], [4, 1, 1])})
                     if policy.telemetry.calls == 1
                     else _reply({"genome": ([0, 1], [1, 3])}),
                     damping=0.5)
    _revise(policy, charges=8)
    before = policy.installed["mode"]
    later = ROWS + _rows(([1] * GENOME, "b", 9, 0))
    _revise(policy, rows=later, charges=16, gen=2)
    assert policy.installed["mode"] == before, (
        "silence about a parameter was read as evidence about it"
    )
    assert "genome" in policy.installed


# --- u4/u5/u6: the refusal taxonomy, whole-reply -----------------------------

def test_a_concentration_above_the_cap_refuses_the_whole_reply():
    policy = _policy(_reply({"mode": (["a", "b", "c"], [9, 1, 1])}),
                     max_weight_ratio=8.0)
    outcome = _revise(policy)
    assert "over_concentrated" in outcome.note["refused"]
    assert policy.installed == {} and outcome.restriction is None
    assert policy.telemetry.revisions_refused == 1
    assert policy.telemetry.revisions_admitted == 0


def test_zeroing_a_value_a_front_member_holds_refuses_the_whole_reply():
    # ROWS[0] is rank-0 and holds mode "a".
    policy = _policy(_reply({"mode": (["a", "b", "c"], [0, 1, 1])}))
    outcome = _revise(policy)
    assert outcome.note["refused"].startswith("excludes_front")
    assert policy.installed == {}
    assert policy.telemetry.revisions_refused == 1


def test_omitting_a_front_value_is_admitted_and_damps():
    """Silence keeps mass at the gate, and what reading it as a zero cost.

    ROWS[0] is rank-0 and holds ``mode="a"``; this reply names ``mode`` and
    lists only "b" and "c". Read as a zero it was refused WHOLE, and that
    reading -- not the model -- was the binding constraint on the channel:
    10 of 11 live refusals were ``excludes_front`` on subset replies; on the
    losing taped pair the late revisions were refused exactly where the
    oracle's hindsight alignment peaked (delta loglik 2.55 at k = 2); and the
    oracle's OWN replies, authored with the winning run's front in hand, were
    refused at the late checkpoints of both studies (s101 at k = 3, s105 at
    k = 2 and k = 3). Damping already makes the exclusion impossible, so the
    reply is admitted and the omitted front value is DAMPED, not dropped.
    (2026-08-21, oracle studies s101_lost and s105_selfwin.)
    """

    policy = _policy(_reply({"mode": (["b", "c"], [1, 1])}), damping=0.5)
    note = _revise(policy).note
    assert note.get("admitted") is True, f"a subset reply was refused: {note}"
    values, weights = policy.installed["mode"]
    assert values == ("a", "b", "c")
    # base uniform (1/3, 1/3, 1/3); proposal normalized (0, 1/2, 1/2); mixed
    # at 0.5 = (1/6, 5/12, 5/12) -- the omitted front value keeps HALF.
    assert weights == pytest.approx((1 / 6, 5 / 12, 5 / 12), abs=1e-12)
    assert weights[0] == pytest.approx(0.5 * (1 / 3), abs=1e-12), (
        "the omitted front value lost more mass than the mixture takes, so "
        "silence is still being read as an exclusion somewhere"
    )
    assert policy.telemetry.revisions_refused == 0


def test_an_explicit_zero_on_a_front_value_still_refuses_the_whole_reply():
    """The half v3 did not relax: a zero the reply WROTE is still a zero."""

    policy = _policy(_reply({"mode": (["a", "b", "c"], [0.0, 1, 1])}))
    outcome = _revise(policy)
    assert outcome.note["refused"].startswith("excludes_front")
    assert policy.installed == {} and outcome.restriction is None
    assert policy.telemetry.revisions_refused == 1


def test_at_damping_one_there_is_no_mixture_and_silence_is_a_zero_again():
    """The premise the relaxed gate rests on, written as its condition.

    "Silence keeps mass" is a fact about the MIXTURE: an unlisted value keeps
    ``(1 - a)`` of the share it held. At ``a = 1`` the reply is installed as
    written, an omission really does zero a front value, and the gate is the
    only guarantee left -- so it refuses, exactly as it did before. Every
    measured cell runs at 0.5; this is the boundary, not the campaign.
    """

    policy = _policy(_reply({"mode": (["b", "c"], [1, 1])}), damping=1.0)
    assert _revise(policy).note["refused"].startswith("excludes_front")
    assert policy.installed == {}


def test_an_undeclared_field_and_an_undeclared_value_are_each_refused_whole():
    field = _policy(_reply({"speed": (["fast"], [2])}))
    assert "undeclared parameter" in _revise(field).note["refused"]
    assert field.telemetry.revisions_refused == 1
    assert field.installed == {}

    value = _policy(_reply({"mode": (["a", "b", "z"], [1, 1, 2])}))
    assert "undeclared value" in _revise(value).note["refused"]
    assert value.telemetry.revisions_refused == 1
    assert value.installed == {}


def test_unparseable_mismatched_and_negative_replies_are_refused_by_name():
    assert _revise(_policy("no json here")).note["refused"] == "unparsed"
    mismatched = _policy(json.dumps(
        {"weights": {"mode": {"values": ["a", "b"], "weights": [1]}}}))
    assert "malformed entry" in _revise(mismatched).note["refused"]
    negative = _policy(_reply({"mode": (["a", "b", "c"], [1, -1, 1])}))
    assert "invalid_weight" in _revise(negative).note["refused"]
    zeroed = _policy(_reply({"mode": (["a", "b", "c"], [0, 0, 0])}))
    assert "all_zero" in _revise(zeroed).note["refused"]
    wrote_candidate = _policy(json.dumps({"weights": {"mode": "c"}}))
    assert _revise(wrote_candidate).note["refused"] == "wrote_candidate"


# --- u7: the bet, and the revert --------------------------------------------

def test_a_revision_that_produced_no_rank_zero_row_is_reverted_first():
    replies = iter([
        _reply({"mode": (["a", "b", "c"], [4, 1, 1])}),   # installs
        "not json at all",                                # refused, after revert
        _reply({"mode": (["a", "b", "c"], [1, 1, 4])}),   # revises anew
    ])
    policy = _policy(lambda _prompt: next(replies), every=1, min_rows=1)
    seeded = WeightedRestriction({"mode": (("a", "b", "c"), (1.0, 1.0, 2.0))})

    _revise(policy, charges=4, restriction=seeded)
    assert policy.installed["mode"][1] != (1.0, 1.0, 2.0), (
        "the first event installed nothing, so there is no bet to revert"
    )

    # Everything measured after the event is dominated: the bet lost.
    dominated = ROWS + _rows(([0] * GENOME, "c", 1, 4),
                             ([0] * GENOME, "a", 0, 3))
    outcome = _revise(policy, rows=dominated, charges=8, gen=2)
    assert policy.telemetry.revisions_reverted == 1
    assert outcome.note["reverted"]["rows_at_event"] == len(ROWS)
    assert policy.installed["mode"] == (("a", "b", "c"), (1.0, 1.0, 2.0)), (
        "the revert did not restore the weights the event replaced, verbatim"
    )
    assert outcome.restriction is not None, (
        "a revert changed the installed weights and the loop was not told"
    )

    revived = dominated + _rows(([1] * GENOME, "c", 9, 0))
    _revise(policy, rows=revived, charges=12, gen=3)
    assert policy.telemetry.revisions_admitted == 2
    assert policy.telemetry.revisions_reverted == 1, (
        "a revision whose rows ARE rank-0 was reverted anyway"
    )


def test_a_rank_zero_row_that_moves_no_front_member_is_not_enough():
    """The 20370102 shape: revisions landing beside the front, never past it.

    On three objectives nearly every fresh point is non-dominated, so the
    first reading of the bet (some post-event row is rank-0) admitted 23
    revisions and reverted 0 across the first live venue's runs while one of
    them sat flat for 120 charges. The bet is strict domination of a
    pre-event front member: the tilted prior must move the front, not merely
    land beside it. (2026-08-19, W1 pilot.)
    """

    replies = iter([
        _reply({"mode": (["a", "b", "c"], [4, 1, 1])}),   # installs
        _reply({"mode": (["a", "b", "c"], [1, 4, 1])}),   # after revert, anew
        _reply({"mode": (["a", "b", "c"], [1, 1, 4])}),   # kept: front moved
    ])
    policy = _policy(lambda _prompt: next(replies), every=1, min_rows=1)

    _revise(policy, charges=4)
    beside = ROWS + _rows(([1] * GENOME, "b", 9, 1))   # rank-0; dominates nothing
    _revise(policy, rows=beside, charges=8, gen=2)
    assert policy.telemetry.revisions_reverted == 1, (
        "a revision whose window merely landed BESIDE the front survived"
    )

    past = beside + _rows(([1] * GENOME, "c", 9, 0))   # dominates the (8, 0) member
    _revise(policy, rows=past, charges=12, gen=3)
    assert policy.telemetry.revisions_admitted == 3
    assert policy.telemetry.revisions_reverted == 1, (
        "a revision whose window dominated a pre-event front member was "
        "reverted anyway"
    )


# --- u8: immigrants ---------------------------------------------------------

def test_immigrants_are_validated_capped_and_reach_the_next_generation():
    good = {"genome": [1] * GENOME, "mode": "c"}
    reply = _reply(
        {"mode": (["a", "b", "c"], [1, 1, 2])},
        immigrants=[
            {"genome": [1] * (GENOME - 1) + [2], "mode": "a"},   # out of domain
            dict(TEMPLATE, genome=list(TEMPLATE["genome"])),     # already measured
            good,
        ])
    policy = _policy(reply, every=8, min_rows=1, immigrants=2, max_events=1)
    problem = _Problem()
    result = run_genetic_loop(
        problem=problem,
        config=GeneticConfig(population_size=8, offspring_per_generation=6,
                             generations=12, seed=5, evaluation_budget=48,
                             reguidance=policy),
    )
    telemetry = policy.telemetry
    assert (telemetry.immigrants_proposed, telemetry.immigrants_accepted,
            telemetry.immigrants_rejected) == (3, 1, 2)
    reasons = {entry["reason"]
               for entry in telemetry.events[0]["immigrants"]["rejected"]}
    assert reasons == {"out_of_domain", "already_measured"}
    injected = [h["reguide_immigrants"] for h in result.history
                if "reguide_immigrants" in h]
    assert injected == [1], f"the accepted immigrant never reached a generation: {injected}"
    assert good in problem.stream, "the injected immigrant was never measured"


def test_the_event_splits_its_rejections_by_reason():
    """One aggregate count names no failure; three counts name three.

    ``already_measured`` is the one the live measurement was drowning in --
    proposals arrived on schedule and were accepted 0 of 69, every drop a
    configuration the run had already charged -- and it is the one the novelty
    clause is aimed at, so a campaign has to be able to read it apart from a
    model that misread a declared vocabulary (``out_of_domain``) or wrote
    something that is not a configuration of this schema at all (``shape``).

    The ``shape`` member here names a field the schema does not declare, which
    is what ``shape`` means under v3i: a member that merely names FEWER fields
    than the schema is completed from the best measured row and is not a
    failure at all (see u9f). The reason's meaning narrowed; the split did not.
    """

    duplicate = dict(ROWS[0][0], genome=list(ROWS[0][0]["genome"]))
    fresh = {"genome": [1] * (GENOME - 1) + [0], "mode": "c"}
    policy = _policy(_reply({"mode": (["a", "b", "c"], [1, 1, 2])},
                            immigrants=[
                                {"mode": "c", "speed": "fast"},      # shape
                                {"genome": [1] * (GENOME - 1) + [2],
                                 "mode": "a"},                       # domain
                                duplicate,                           # measured
                                fresh,
                            ]),
                     immigrants=4)
    outcome = _revise(policy)

    assert outcome.immigrants == (fresh,)
    assert outcome.note["immigrants"]["rejected_by_reason"] == {
        "shape": 1, "out_of_domain": 1, "already_measured": 1}, (
        "the event reports a rejection total the campaign cannot act on"
    )
    assert outcome.note["immigrants"]["rejected"] == [
        {"reason": "shape"}, {"reason": "out_of_domain"},
        {"reason": "already_measured"}], (
        "the per-member rejection list changed shape; the split was meant to "
        "be added beside it, not to replace it"
    )
    assert (policy.telemetry.immigrants_proposed,
            policy.telemetry.immigrants_accepted,
            policy.telemetry.immigrants_rejected) == (4, 1, 3)


def test_immigrants_are_not_read_when_none_were_bought():
    policy = _policy(_reply({"mode": (["a", "b", "c"], [1, 1, 2])},
                            immigrants=[{"genome": [1] * GENOME, "mode": "c"}]))
    outcome = _revise(policy)
    assert outcome.immigrants == ()
    assert policy.telemetry.immigrants_proposed == 0
    assert "immigrants" not in outcome.note, (
        "a run that bought no proposals recorded a requirement anyway"
    )
    assert policy.telemetry.immigrants_shortfall == 0


def test_a_short_immigrants_list_admits_the_weights_and_counts_the_shortfall():
    """Required-k is a compliance METER, not a second refusal mode.

    The two halves of a reply are judged separately: an under-answered joint
    proposal cannot cost the run the prior half that was admissible on its
    own. What the shortfall buys is the number a campaign needs -- how often
    the required-k ask was answered at all -- which the optional clause it
    replaced could never report, having been answered zero times.
    (2026-08-21, oracle studies s101_lost and s105_selfwin.)
    """

    good = {"genome": [1] * GENOME, "mode": "c"}
    also = {"genome": [1] * (GENOME - 1) + [0], "mode": "b"}
    policy = _policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])},
                            immigrants=[good, also]),
                     immigrants=3)
    outcome = _revise(policy)

    assert outcome.note["admitted"] is True, "the weights half was refused too"
    assert outcome.immigrants == (good, also), (
        "the members that WERE written were not accepted"
    )
    assert outcome.note["immigrants"] == {
        "accepted": 2, "rejected": [], "rejected_by_reason": {},
        "proposed": 2, "required": 3, "shortfall": 1}
    assert policy.telemetry.immigrants_shortfall == 1
    assert (policy.telemetry.immigrants_proposed,
            policy.telemetry.immigrants_accepted,
            policy.telemetry.immigrants_rejected) == (2, 2, 0)


def test_a_missing_immigrants_array_counts_the_whole_requirement_as_short():
    policy = _policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])}),
                     immigrants=3)
    outcome = _revise(policy)

    assert outcome.note["admitted"] is True and outcome.immigrants == ()
    assert outcome.note["immigrants"] == {
        "accepted": 0, "rejected": [], "rejected_by_reason": {},
        "proposed": 0, "required": 3, "shortfall": 3}
    assert policy.telemetry.immigrants_shortfall == 3
    assert policy.telemetry.immigrants_proposed == 0, (
        "an array that was never written counted as members proposed"
    )


# --- u9: the control seam ---------------------------------------------------

def test_the_evidence_view_changes_the_digest_and_nothing_else():
    reply = _reply({"mode": (["a", "b", "c"], [4, 1, 1])})
    identity = _policy(reply)
    donor = _policy(reply, evidence_view=lambda rows: list(rows)[:2])

    first = _revise(identity).note
    second = _revise(donor).note
    assert first["evidence_sha256"] != second["evidence_sha256"], (
        "swapping the rows the model is shown did not change the journalled "
        "digest, so a shuffled-evidence control would be unfalsifiable"
    )
    assert first["rows_shown"] == 3 and second["rows_shown"] == 2
    assert identity.installed == donor.installed
    assert (identity.telemetry.as_dict() == donor.telemetry.as_dict())


def test_the_gate_reads_the_run_by_default_and_the_view_only_when_told():
    """The confound the W1 pilot's shuffled arm was built with, and its fix.

    ROWS[0] is rank-0 in the RUN and holds ``mode="a"``; the donor view drops
    it, so the VIEWED front holds only ``"b"`` and ``"c"``. One reply zeroing
    ``"a"`` therefore reads as ``excludes_front`` against reality and as an
    ordinary concentration against the view. The product refuses it -- the
    gate that protects a live run reads what the run measured. The control
    admits it, which is the whole point: refusals it accrues for evidence it
    was never shown are not refusals the arm it controls could ever meet.
    """

    reply = _reply({"mode": (["a", "b", "c"], [0, 1, 1])})
    view = lambda rows: list(rows)[1:]                    # noqa: E731

    product = _policy(reply, evidence_view=view)
    assert _revise(product).note["refused"].startswith("excludes_front")
    assert product.installed == {}
    assert product.gate_reads_view is False, "the default is not the safe one"

    control = _policy(reply, evidence_view=view, gate_reads_view=True)
    note = _revise(control).note
    assert note.get("admitted") is True, (
        f"the control's gate still read the run's own front: {note}")
    assert note["gate_reads_view"] is True, (
        "a run whose gate read a view did not journal that it had"
    )
    assert control.installed["mode"][0] == ("a", "b", "c")
    assert control.installed["mode"][1][0] > 0.0, (
        "damping below 1 must keep the zeroed value sampled even here"
    )


def test_the_gate_view_flag_changes_nothing_when_no_view_is_declared():
    """True is a no-op on the product's own path: identity view, same front."""

    reply = _reply({"mode": (["a", "b", "c"], [0, 1, 1])})
    for policy in (_policy(reply), _policy(reply, gate_reads_view=True)):
        assert _revise(policy).note["refused"].startswith("excludes_front")
        assert policy.installed == {}


# --- u9b: the v2 evidence bundle --------------------------------------------
#
# The bundle is two renderings and one digest. The measured trace answers
# "which knob moves which cost" with rank correlations that need rows the
# mid-run reader does not have; the occupancy table answers "what is the front
# made of" with counts that survive tens of rows. Both read the SAME viewed
# rows, so the control parameter transforms all of what the model sees.

#: A trace whose front has two members, both holding ``mode="a"``, so the
#: occupancy table has something to say and the front is not one row.
FRONT_ROWS = _rows(
    ([1] * GENOME, "a", 8, 3),                       # rank-0
    ([0, 1] * (GENOME // 2), "a", 5, 1),             # rank-0
    ([1, 1, 0, 0] * (GENOME // 4), "b", 4, 4),       # dominated
    ([0] * GENOME, "c", 0, 5),                       # dominated
)


def _halves(note) -> tuple:
    """The bundle split at its declared section title."""

    head, sep, tail = note["evidence_text"].partition(
        f"\n\n  {ELITE_TABLE_TITLE}:\n")
    assert sep, f"the bundle carries no elite section:\n{note['evidence_text']}"
    return head, tail


def test_the_bundle_is_the_measured_trace_then_the_occupancy_table():
    policy = _policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])}))
    note = _revise(policy, rows=FRONT_ROWS).note
    head, tail = _halves(note)

    view = [(config, objectives, False) for config, objectives in FRONT_ROWS]
    assert head == render_measurement_evidence(
        view, list(OBJECTIVES), policy._locus_domains,
        front_shown=policy.front_shown, effects_shown=policy.effects_shown,
        charged=len(FRONT_ROWS))
    assert tail == render_elite_table(view, list(OBJECTIVES),
                                      policy._locus_domains)
    assert "the current non-dominated front" in head
    assert "counts of configurations, never scores" in tail
    # Both halves reach the model, in that order, in one prompt.
    prompt = policy.prompts[0]
    assert prompt.count(note["evidence_text"]) == 1
    assert prompt.index(head) < prompt.index(ELITE_TABLE_TITLE) < prompt.index(tail)


def test_the_event_always_journals_the_bundle_verbatim_and_its_version():
    admitted = _policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])}))
    refused = _policy("no json here")

    def _boom(_prompt: str) -> str:
        raise RuntimeError("the provider hung up")

    errored = Reguidance(_boom, objectives=OBJECTIVES,
                         candidate_model=_Candidate, template=dict(TEMPLATE),
                         every=1, min_rows=1)
    for policy in (admitted, refused, errored):
        note = policy.maybe_revise(
            rows=FRONT_ROWS, population=(), specs=OBJECTIVES, restriction=None,
            charges=len(FRONT_ROWS), gen=1).note
        assert note["evidence"] == EVIDENCE_VERSION == "v2"
        assert evidence_digest(note["evidence_text"]) == note["evidence_sha256"], (
            "the journalled text is not the text that was digested, so an "
            "oracle study reading it would be reading a different prompt"
        )
        assert note["evidence_text"] in policy.telemetry.events[-1]["evidence_text"]


def test_the_digest_covers_the_whole_bundle_and_not_the_measured_half():
    policy = _policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])}))
    note = _revise(policy, rows=FRONT_ROWS).note
    head, _tail = _halves(note)
    assert evidence_digest(head) != note["evidence_sha256"], (
        "the digest was taken over the measured half alone, so the occupancy "
        "table could move without the journal noticing"
    )


def test_moving_which_rows_are_rank_zero_moves_the_occupancy_table_and_the_digest():
    reply = _reply({"mode": (["a", "b", "c"], [2, 1, 1])})
    wide = _revise(_policy(reply), rows=FRONT_ROWS).note
    # One extra row dominates the whole trace: the front collapses to it, and
    # its mode is "c" where the old front was "a" twice.
    narrowed = FRONT_ROWS + _rows(([1] * GENOME, "c", 9, 0))
    tight = _revise(_policy(reply), rows=narrowed).note

    assert _halves(wide)[1] != _halves(tight)[1], (
        "the front's membership changed and the occupancy table did not"
    )
    assert wide["evidence_sha256"] != tight["evidence_sha256"]


def test_the_evidence_view_transforms_both_halves_of_the_bundle():
    reply = _reply({"mode": (["a", "b", "c"], [2, 1, 1])})
    identity = _policy(reply)
    donor = _policy(reply, evidence_view=lambda rows: list(rows)[:2])

    first = _revise(identity, rows=FRONT_ROWS).note
    second = _revise(donor, rows=FRONT_ROWS).note
    for a, b in zip(_halves(first), _halves(second)):
        assert a != b, (
            "a control arm swapped the rows the model reads and one half of "
            "the bundle carried on rendering the run's own measurements"
        )
    assert second["rows_shown"] == 2
    assert "all 2 measured configurations" in _halves(second)[1], (
        "the occupancy table counted rows the control never showed"
    )


# --- u9c: what the immigrants clause asks for -------------------------------

def test_the_immigrants_clause_grounds_proposals_in_the_occupancy_table():
    grounding = ("recombinations or refinements of what\nthe occupancy table "
                 "says the front rewards, never repeats of configurations\n"
                 "the run has already measured")
    bought = _policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])}),
                     immigrants=3)
    _revise(bought, rows=FRONT_ROWS)
    prompt = bought.prompts[0]
    assert 'Your reply MUST also carry an "immigrants" key holding EXACTLY 3 ' \
           'COMPLETE' in prompt, (
        "the clause still offers the channel it was measured to have to "
        "require: an optional immigrants key went unanswered in ~30 calls"
    )
    assert grounding in prompt
    assert '{"immigrants": [{"<parameter>": <value>, ...}]}' in prompt, (
        "the reworded clause changed the schema it asks for"
    )

    none = _policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])}), immigrants=0)
    _revise(none, rows=FRONT_ROWS)
    assert "immigrants" not in none.prompts[0], (
        "a run that bought no immigrants was asked for them anyway"
    )


def test_the_immigrants_clause_grounds_novelty_in_the_measured_count():
    """The 0-of-69 signature's fix, stated as the model can act on it.

    Required, the clause was ANSWERED -- twelve proposals a cell, on schedule
    -- and accepted zero times: at roughly 320 measured rows a recombination
    of the elites on display is usually a configuration the run has already
    charged, and the dedup drops it before it costs anything. "Never repeats"
    was already in the prose; the ground was not, because the model cannot
    count rows it was never handed. So the clause states the count and states
    the price of a repeat.
    """

    policy = _policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])}),
                     immigrants=3)
    _revise(policy, rows=FRONT_ROWS)
    asked = " ".join(policy.prompts[0].split())
    assert f"this run has ALREADY MEASURED {len(FRONT_ROWS)} configurations" \
        in asked, (
        "the novelty ask names no count, which is the state that was measured "
        "to be answered 12 times a cell and accepted 0 of 69"
    )
    assert ("A proposal that repeats one of those "
            f"{len(FRONT_ROWS)} is REJECTED without being measured and WASTES "
            "the slot it took.") in asked
    assert ("Every one of the 3 must differ from every configuration this run "
            "has measured, in at least one parameter") in asked

    # The count is the RUN's, not the rendering's: a control arm that shows
    # donor rows is asked for novelty against the same dedup its proposals
    # will actually meet.
    donor = _policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])}),
                    immigrants=3, evidence_view=lambda rows: list(rows)[:2])
    _revise(donor, rows=FRONT_ROWS)
    assert f"ALREADY MEASURED {len(FRONT_ROWS)} configurations" in \
        " ".join(donor.prompts[0].split())


def test_the_novelty_ground_moves_with_the_run_and_the_reply_shape_does_not():
    """Two events, two counts, one schema line."""

    seen: List[int] = []

    def _reply_for(prompt: str) -> str:
        seen.append(len(prompt))
        return _reply({"mode": (["a", "b", "c"], [2, 1, 1])})

    policy = _policy(_reply_for, immigrants=2, max_events=2, every=1,
                     min_rows=1)
    _revise(policy, rows=FRONT_ROWS, charges=4)
    _revise(policy, rows=FRONT_ROWS + ROWS, charges=16)

    counts = [line for prompt in policy.prompts
              for line in prompt.split("\n") if "ALREADY MEASURED" in line]
    assert len(counts) == 2 and counts[0] != counts[1], (
        f"the measured count did not track the run's own rows: {counts}"
    )
    assert "ALREADY MEASURED 4" in counts[0]
    assert "ALREADY MEASURED 7" in counts[1]
    for prompt in policy.prompts:
        assert '{"immigrants": [{"<parameter>": <value>, ...}]}' in prompt, (
            "the novelty wording changed the reply shape the clause asks for"
        )


# --- u9f: the v3i key repair -------------------------------------------------
#
# Required-k made the channel answer; the KEYS then killed every answer. On the
# analog venue -- 24 flat fields spelled ``bias_nmos__w`` -- every member of
# every cell was refused as ``shape``, while the same clause over the six-field
# NAS schema passed, so what the model failed was the spelling of the schema
# and not the search. Two bounded repairs, and a hard floor under both: one
# flatten, one fill from the run's best measured row, and an invented key is
# still ``shape``.

#: A schema whose field names ENCODE a nesting -- the analog shape, at three
#: fields instead of twenty-four. A model handed these writes the groups back
#: as objects, which is the spelling the flatten inverts.
FLAT_TEMPLATE: Dict[str, Any] = {"bias_nmos__w": 1, "bias_nmos__l": 1,
                                 "mode": "a"}


class _FlatCandidate(BaseModel):
    bias_nmos__w: Literal[1, 2, 4]
    bias_nmos__l: Literal[1, 2, 4]
    mode: Literal["a", "b", "c"]


FLAT_ROWS = [
    ({"bias_nmos__w": 1, "bias_nmos__l": 1, "mode": "a"},
     {"ones": 1.0, "edges": 1.0}),
    ({"bias_nmos__w": 2, "bias_nmos__l": 1, "mode": "b"},
     {"ones": 2.0, "edges": 3.0}),
]


def _flat_policy(reply, **kwargs) -> Reguidance:
    prompts: List[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return reply(prompt) if callable(reply) else reply

    policy = Reguidance(complete, objectives=OBJECTIVES,
                        candidate_model=_FlatCandidate,
                        template=dict(FLAT_TEMPLATE), every=1, min_rows=1,
                        **kwargs)
    policy.prompts = prompts                    # type: ignore[attr-defined]
    return policy


def _flat_revise(policy: Reguidance, rows=FLAT_ROWS):
    return policy.maybe_revise(rows=rows, population=(), specs=OBJECTIVES,
                               restriction=None, charges=len(rows), gen=1)


#: A trace whose BEST row on the first objective is not its first row, so a
#: fill that silently took row 0 would be visible.
FILL_ROWS = _rows(
    ([0, 1] * (GENOME // 2), "b", 4, 1),             # rank-0, ones = 4
    ([1] * GENOME, "a", 8, 3),                       # rank-0, ones = 8: best
    ([0] * GENOME, "c", 0, 5),                       # dominated
)


def test_a_nested_member_is_flattened_once_and_then_accepted():
    """The spelling the flat field names encode, inverted exactly once."""

    nested = {"bias_nmos": {"w": 2, "l": 4}, "mode": "b"}
    policy = _flat_policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])},
                                 immigrants=[nested]),
                          immigrants=1)
    outcome = _flat_revise(policy)

    assert outcome.immigrants == (
        {"bias_nmos__w": 2, "bias_nmos__l": 4, "mode": "b"},), (
        "a member that nested the groups its field names encode was refused, "
        "which is how every analog cell lost every member it was sent"
    )
    record = outcome.note["immigrants"]
    assert record["accepted"] == 1 and record["normalized"] == 1
    assert "template_filled" not in record, (
        "a member that carried every field after the flatten was also filled"
    )
    assert policy.telemetry.immigrants_rejected == 0


def test_a_nested_member_that_is_also_partial_is_flattened_and_then_filled():
    """The two repairs compose, and each is counted where it happened."""

    policy = _flat_policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])},
                                 immigrants=[{"bias_nmos": {"w": 4}}]),
                          immigrants=1)
    outcome = _flat_revise(policy)

    # Best on "ones" (max) among the rank-0 rows is FLAT_ROWS[1]: w=2, l=1,
    # mode "b". The member names w, so only l and mode are filled.
    assert outcome.immigrants == (
        {"bias_nmos__w": 4, "bias_nmos__l": 1, "mode": "b"},)
    record = outcome.note["immigrants"]
    assert (record["normalized"], record["template_filled"]) == (1, 1)
    assert record["fill_source"] == "best_measured"


def test_a_member_spelled_with_the_double_underscore_passes_untouched():
    written = {"bias_nmos__w": 4, "bias_nmos__l": 2, "mode": "c"}
    policy = _flat_policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])},
                                 immigrants=[dict(written)]),
                          immigrants=1)
    outcome = _flat_revise(policy)

    assert outcome.immigrants == (written,)
    record = outcome.note["immigrants"]
    assert "normalized" not in record and "template_filled" not in record, (
        "a member the model spelled correctly was recorded as repaired"
    )


def test_a_partial_member_is_completed_from_the_best_measured_row():
    """Not the seed, not row 0: the rank-0 row best on the FIRST objective.

    A member naming three of twenty-four fields is the model saying which
    values belong together and staying silent about the rest -- the same
    silence the weights half is already read that way. Completed from the run's
    current best it becomes a recombination against the best; refused, it is
    the whole channel dying of its own prompt (0 accepted, every analog cell).
    """

    policy = _policy(_reply({"mode": (["a", "b", "c"], [1, 1, 2])},
                            immigrants=[{"mode": "c"}]),
                     immigrants=1)
    outcome = _revise(policy, rows=FILL_ROWS)

    assert outcome.immigrants == ({"genome": [1] * GENOME, "mode": "c"},), (
        "the fill did not read the rank-0 row with the best first objective"
    )
    record = outcome.note["immigrants"]
    assert record["accepted"] == 1 and record["template_filled"] == 1
    assert record["fill_source"] == "best_measured"
    assert record["rejected"] == [] and record["shortfall"] == 0
    assert (policy.telemetry.immigrants_proposed,
            policy.telemetry.immigrants_accepted) == (1, 1)
    assert outcome.immigrants[0]["genome"] is not FILL_ROWS[1][0]["genome"], (
        "the filled member shares its list with the measured row it was "
        "completed from, so mutating one mutates a row of the trace"
    )


def test_a_partial_member_holding_a_bad_value_is_still_out_of_domain():
    """The fill completes what is MISSING; it launders nothing that is there."""

    policy = _policy(_reply({"mode": (["a", "b", "c"], [1, 1, 2])},
                            immigrants=[{"mode": "z"}]),
                     immigrants=1)
    outcome = _revise(policy, rows=FILL_ROWS)

    assert outcome.immigrants == ()
    record = outcome.note["immigrants"]
    assert record["rejected_by_reason"] == {"out_of_domain": 1}, (
        "a partial member's own undeclared value survived the completion"
    )
    assert record["template_filled"] == 1, (
        "the fill fired and went uncounted because the member then failed"
    )
    assert policy.telemetry.immigrants_accepted == 0


def test_a_member_with_an_undeclared_key_is_still_shape():
    """The floor under both repairs: spelling is repaired, content is not."""

    invented = {"genome": [1] * GENOME, "mode": "c", "speed": "fast"}
    nested_invention = {"genome": {"x": 1}, "mode": "c"}
    policy = _policy(_reply({"mode": (["a", "b", "c"], [1, 1, 2])},
                            immigrants=[invented, nested_invention]),
                     immigrants=2)
    outcome = _revise(policy, rows=FILL_ROWS)

    assert outcome.immigrants == ()
    record = outcome.note["immigrants"]
    assert record["rejected_by_reason"] == {"shape": 2}
    assert "template_filled" not in record and "normalized" not in record, (
        "a member the harness could not repair was counted as repaired"
    )


def test_a_partial_member_falls_back_to_the_template_when_nothing_is_measured():
    """The boundary, named on the event rather than assumed away.

    A checkpoint cannot fire on an empty trace, so this is reached only by the
    channel's own entry point -- but the fill has to be defined there anyway,
    and an event that filled from the template rather than from the best row
    has to SAY so, because the two are different proposals.
    """

    policy = _policy(_reply({"mode": (["a", "b", "c"], [1, 1, 2])}),
                     immigrants=1)
    note: Dict[str, Any] = {}
    accepted = policy._immigrants([{"mode": "c"}], (), OBJECTIVES, note)

    assert accepted == ({"genome": list(TEMPLATE["genome"]), "mode": "c"},)
    assert note["immigrants"]["fill_source"] == "template"
    assert note["immigrants"]["template_filled"] == 1


def test_the_clause_shows_the_schema_and_says_what_a_partial_member_means():
    policy = _policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])}),
                     immigrants=3)
    _revise(policy, rows=FRONT_ROWS)
    prompt = policy.prompts[0]
    asked = " ".join(prompt.split())

    assert "Every member must carry these exact keys:" in prompt
    assert json.dumps(dict(TEMPLATE)) in prompt, (
        "the clause describes the schema without ever showing one member of "
        "it, which is the state 24-flat-field replies were refused in"
    )
    assert ("A member that names only SOME of these keys is COMPLETED from "
            "the run's current best configuration, so a partial member is a "
            "deliberate recombination against the best rather than an error")\
        in asked
    assert 'The keys are FLAT: write "group__field", never {"group": ' \
           '{"field": ...}}, and never a key this schema does not declare' \
        in asked

    none = _policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])}), immigrants=0)
    _revise(none, rows=FRONT_ROWS)
    assert "Every member must carry these exact keys" not in none.prompts[0]


def test_the_next_clause_names_what_the_last_event_lost_its_members_to():
    """The loop the channel never had: the model learns it was rejected.

    Twelve ``shape`` rejections a cell, every cell, is a model repeating a
    misreading it was never told about. The line is read back off the policy's
    own journalled events -- no new state -- and says nothing after an event
    that rejected nothing.
    """

    replies = iter([
        _reply({"mode": (["a", "b", "c"], [2, 1, 1])},
               immigrants=[{"mode": "c", "speed": "fast"},          # shape
                           {"genome": [1] * GENOME, "x": 2},        # shape
                           {"genome": [1] * GENOME, "mode": "z"}]),  # domain
        _reply({"mode": (["a", "b", "c"], [2, 1, 1])},
               immigrants=[{"genome": [1] * (GENOME - 1) + [0], "mode": "c"},
                           {"genome": [0, 1] * (GENOME // 2), "mode": "c"},
                           {"genome": [1, 1, 0, 0] * (GENOME // 4),
                            "mode": "b"}]),
        _reply({"mode": (["a", "b", "c"], [2, 1, 1])}),
    ])
    policy = _policy(lambda _prompt: next(replies), immigrants=3,
                     max_events=3, every=1, min_rows=1)

    _revise(policy, rows=FILL_ROWS, charges=4)
    assert "Last time" not in policy.prompts[0], (
        "the first event of a run reported a previous event"
    )
    assert policy.telemetry.events[0]["immigrants"]["rejected_by_reason"] == {
        "shape": 2, "out_of_domain": 1}

    _revise(policy, rows=FILL_ROWS + ROWS, charges=8, gen=2)
    assert ("Last time: 2 rejected as shape, 1 rejected as out_of_domain."
            in " ".join(policy.prompts[1].split())), (
        f"the clause carried no feedback from the last event:\n"
        f"{policy.prompts[1]}"
    )
    assert policy.telemetry.events[1]["immigrants"]["rejected_by_reason"] == {}

    _revise(policy, rows=FILL_ROWS + ROWS + FRONT_ROWS, charges=12, gen=3)
    assert "Last time" not in policy.prompts[2], (
        "an event that rejected nothing still reported rejections to the next"
    )


# --- u9d: the focused tilt, asked for and metered ---------------------------
#
# The ask is an ask. The live pilot's replies tilted or freed all 24 fields of
# the analog venue at once where the oracle tilted 2 to 8, so the prompt names
# a ceiling and the harness COUNTS what came back -- on every reply, whatever
# the verdict. A second refusal mode is exactly what v3 exists to remove.

def test_the_prompt_asks_for_a_focused_tilt():
    policy = _policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])}))
    _revise(policy, rows=FRONT_ROWS)
    asked = " ".join(policy.prompts[0].split())
    assert (f'Name AT MOST {TILT_CAP} parameters in "weights" -- the table\'s '
            "strongest cases -- and leave the rest unlisted.") in asked
    assert TILT_CAP == 4


def test_tilt_breadth_is_recorded_on_admitted_and_refused_replies_alike():
    both = {"mode": (["a", "b", "c"], [2, 1, 1]), "genome": ([0, 1], [1, 2])}
    admitted = _policy(_reply(both))
    note = _revise(admitted).note
    assert note["admitted"] is True and note["tilt_breadth"] == 2
    assert admitted.telemetry.breadth_total == 2

    over = dict(both, mode=(["a", "b", "c"], [9, 1, 1]))   # refused on ratio
    refused = _policy(_reply(over))
    note = _revise(refused).note
    assert note["refused"].startswith("over_concentrated")
    assert note["tilt_breadth"] == 2, (
        "a refused reply's breadth went uncounted, so a campaign's median "
        "would be taken over the admitted replies alone"
    )
    assert refused.telemetry.breadth_total == 2

    unreadable = _policy("no json here")
    assert _revise(unreadable).note["tilt_breadth"] == 0
    assert unreadable.telemetry.breadth_total == 0


# --- u9e: the bundle marker -------------------------------------------------

def test_every_event_says_which_mechanism_wrote_it_and_which_evidence_it_read():
    """Two markers, moving independently, so a cell self-identifies.

    v3 changed what the harness ASKS FOR and what it ADMITS, not what it
    shows, so the evidence version stays where it was: a study may pool
    bundles across the mechanism change while refusing to pool mechanisms.
    """

    admitted = _policy(_reply({"mode": (["a", "b", "c"], [2, 1, 1])}))
    refused = _policy("no json here")

    def _boom(_prompt: str) -> str:
        raise RuntimeError("the provider hung up")

    errored = Reguidance(_boom, objectives=OBJECTIVES,
                         candidate_model=_Candidate, template=dict(TEMPLATE),
                         every=1, min_rows=1)
    for policy in (admitted, refused, errored):
        note = policy.maybe_revise(
            rows=FRONT_ROWS, population=(), specs=OBJECTIVES, restriction=None,
            charges=len(FRONT_ROWS), gen=1).note
        assert note["mechanism"] == MECHANISM_VERSION == "v3", (
            f"an event does not name its mechanism: {note}"
        )
        assert note["evidence"] == EVIDENCE_VERSION == "v2", (
            "the mechanism version dragged the evidence version with it, so "
            "a study pooling prompts would be pooling two bundles"
        )
        assert policy.telemetry.events[-1]["mechanism"] == "v3"


def test_an_event_that_bought_the_proposal_channel_marks_itself_v3i():
    """The marker names the CHANNEL that moved, not the release it shipped in.

    v3i repairs the joint-proposal channel and nothing else, so a run with
    ``immigrants = 0`` renders the v3 prompt and admits by the v3 rule -- and
    says v3. A cell can therefore be pooled by what it actually ran rather
    than by which build produced it.
    """

    reply = _reply({"mode": (["a", "b", "c"], [2, 1, 1])})
    bought = _policy(reply, immigrants=2)
    note = _revise(bought, rows=FRONT_ROWS).note
    assert note["mechanism"] == MECHANISM_VERSION_IMMIGRANTS == "v3i"
    assert note["evidence"] == EVIDENCE_VERSION == "v2", (
        "the mechanism marker dragged the evidence version with it again"
    )

    unbought = _policy(reply)
    assert _revise(unbought, rows=FRONT_ROWS).note["mechanism"] == "v3"

    # The channel was BOUGHT whatever the reply then did with it.
    refused = _policy("no json here", immigrants=2)
    assert _revise(refused, rows=FRONT_ROWS).note["mechanism"] == "v3i"


# --- u10: a policy failure never kills a run --------------------------------

def test_a_completion_that_raises_is_counted_and_the_run_finishes():
    def _boom(_prompt: str) -> str:
        raise RuntimeError("the provider hung up")

    telemetry = ReguidanceTelemetry()
    policy = Reguidance(_boom, objectives=OBJECTIVES,
                        candidate_model=_Candidate, template=dict(TEMPLATE),
                        every=8, min_rows=1, telemetry=telemetry)
    problem = _Problem()
    result = run_genetic_loop(
        problem=problem,
        config=GeneticConfig(population_size=8, offspring_per_generation=6,
                             generations=10, seed=3, evaluation_budget=32,
                             reguidance=policy),
    )
    assert telemetry.errors >= 1
    assert telemetry.revisions_admitted == 0
    assert result.evaluations <= 32 and result.pareto_front
    assert all("error" in event for event in telemetry.events)


# --- u11: the two properties damping exists for -----------------------------

def test_mixing_two_capped_vectors_keeps_the_cap():
    """The mediant inequality, on the code that does the mixing."""

    rng = random.Random(20370819)
    domain = ["a", "b", "c"]
    for _trial in range(200):
        base = [rng.uniform(1.0, 8.0) for _ in domain]
        base = [w * 8.0 / max(base) if max(base) / min(base) > 8.0 else w
                for w in base]
        proposal = [rng.uniform(1.0, 8.0) for _ in domain]
        assert max(base) / min(base) <= 8.0 + 1e-12
        assert max(proposal) / min(proposal) <= 8.0 + 1e-12
        policy = _policy("{}", damping=rng.random())
        policy._installed = {"mode": (tuple(domain), tuple(base))}
        mixed = policy._damp({"mode": list(zip(domain, proposal))}, [])
        weights = mixed["mode"][1]
        assert max(weights) / min(weights) <= 8.0 + 1e-12, (
            f"mixing {base} with {proposal} broke the cap: {weights}"
        )


def test_damping_below_one_can_never_introduce_an_exclusion():
    policy = _policy("{}", damping=0.999)
    policy._installed = {"mode": (("a", "b", "c"), (1.0, 1.0, 1.0))}
    # A proposal that would exclude two of the three values outright.
    mixed = policy._damp({"mode": [("a", 1.0), ("b", 0.0), ("c", 0.0)]}, [])
    assert all(w > 0.0 for w in mixed["mode"][1]), (
        "a damped revision zeroed a value, which is the one failure class "
        "the mixture exists to make structurally impossible"
    )
    assert WeightedRestriction(mixed).allowed["mode"] == ("a", "b", "c")


# --- u12: the wiring --------------------------------------------------------

def _seam(monkeypatch, reply_for) -> List[str]:
    prompts: List[str] = []

    def _factory(model, settings=None, **kwargs):
        def _complete(prompt: str) -> str:
            prompts.append(prompt)
            return reply_for(prompt)

        return _complete

    monkeypatch.setattr(
        "agent_evolve.integrations.completion.completion_for", _factory)
    return prompts


def test_the_adaptive_preset_reaches_the_loop_through_optimize(monkeypatch):
    revision = _reply({"mode": (["a", "b", "c"], [2, 1, 1])})

    def _reply_for(prompt: str) -> str:
        if "You are REVISING the weighted sampling prior mid-run." in prompt:
            return revision
        return ""

    _seam(monkeypatch, _reply_for)
    said: List[str] = []
    problem = _Problem()
    result = optimize(problem, budget=64, seed=4, proposer="llm",
                      authorship=AuthorshipConfig.preset("adaptive"),
                      on_progress=said.append)

    events = [h["reguide"] for h in result.history if "reguide" in h]
    assert events, "the adaptive preset made no revision event reach the history"
    assert all("evidence_sha256" in event for event in events)
    mechanisms = {m.mechanism for m in result.telemetry.mechanisms}
    assert "reguidance" in mechanisms, (
        "the revision channel's counters never reached the result"
    )
    counters = next(m.counters for m in result.telemetry.mechanisms
                    if m.mechanism == "reguidance")
    assert counters["calls"] == len(events)
    assert any("revised every" in message for message in said), (
        "the resolved cadence was never announced"
    )
    assert result.evaluations <= 64


def test_adaptive_knobs_without_adaptation_are_refused_by_name():
    with pytest.raises(ValueError, match="adapt_every"):
        AuthorshipConfig(adapt_every=8)
    with pytest.raises(ValueError, match="adapt_immigrants"):
        AuthorshipConfig(adapt_immigrants=2)
    with pytest.raises(ValueError, match="adaptation"):
        AuthorshipConfig(adaptation="sometimes")
    with pytest.raises(ValueError, match="adapt_damping"):
        AuthorshipConfig(adaptation="llm", adapt_damping=1.5)
    with pytest.raises(ValueError, match="adapt_gate_reads_view"):
        AuthorshipConfig(adapt_gate_reads_view=True)


def test_the_gate_view_flag_is_off_by_default_and_reaches_the_policy():
    said: List[str] = []
    kwargs = dict(complete=lambda _p: "{}", objectives=list(OBJECTIVES),
                  init_template=dict(TEMPLATE), candidate_model=_Candidate,
                  budget=64, population_size=8)
    default = build_authorship(AuthorshipConfig(adaptation="llm"),
                               announce=said.append, **kwargs)
    assert default.reguidance.gate_reads_view is False
    assert not any("gate_reads_view" in message for message in said), (
        "the product's own default announced itself as a control"
    )
    control = build_authorship(
        AuthorshipConfig(adaptation="llm", adapt_gate_reads_view=True),
        announce=said.append, **kwargs)
    assert control.reguidance.gate_reads_view is True
    assert any("adapt_gate_reads_view=True" in message for message in said), (
        "a control arm's gate was installed without saying so"
    )


def test_adaptation_without_a_model_call_leaves_the_counters_a_home():
    said: List[str] = []
    policies = build_authorship(
        AuthorshipConfig(adaptation="llm"), complete=None,
        objectives=list(OBJECTIVES), init_template=dict(TEMPLATE),
        announce=said.append, candidate_model=_Candidate, budget=64,
        population_size=8)
    assert policies.reguidance is None
    assert policies.reguidance_author is not None
    assert policies.reguidance_author.mechanism == "reguidance"
    assert any("needs a model call" in message for message in said)


def test_the_resolved_cadence_comes_from_the_budget_and_the_population():
    said: List[str] = []
    policies = build_authorship(
        AuthorshipConfig(adaptation="llm"), complete=lambda _p: "{}",
        objectives=list(OBJECTIVES), init_template=dict(TEMPLATE),
        announce=said.append, candidate_model=_Candidate, budget=384,
        population_size=12)
    assert policies.reguidance.every == 64        # max(2 * 12, 384 // 6)
    assert any("every 64 charged evaluations" in message for message in said)
