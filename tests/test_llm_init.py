"""Model-authored initialization: validated value-by-value, seeds first,
rejects cost a slot not the run -- and, under the split ask, labelled."""

from __future__ import annotations

import json
from typing import Any, Dict, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.policies.llm_init import InitTelemetry, author_initial_population
from agent_evolve.session.authorship import AuthorshipConfig, build_authorship


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


# --- the ask: "joint" is the sealed one, byte for byte ----------------------

#: Today's prompt, copied out literally. If a refinement moves the default
#: ask by so much as a character, every sealed initialization row was
#: measured under a different prompt from the one that ships, and this test is
#: the only thing that would say so.
JOINT_PROMPT_TODAY = """card

You are proposing the INITIAL POPULATION for an evolutionary search over the
problem above. These 5 configurations are the raw material every later
recombination works with; they should be individually strong bets AND
collectively diverse (covering distinct promising regions/trade-offs), based
on what the parameters and objectives MEAN.

Reply with ONLY a JSON list of exactly 5 configuration objects, each with
every parameter field, every value taken from that parameter's declared
domain. No commentary."""


def _capture(reply):
    """(complete, seen) -- a fake provider that records the prompt it got."""

    seen: list[str] = []

    def complete(prompt):
        seen.append(prompt)
        return reply

    return complete, seen


def test_the_joint_ask_is_the_default_and_its_prompt_has_not_moved():
    complete, seen = _capture(json.dumps([{"genome": [1, 1, 1, 1, 1, 1]}]))
    tel = InitTelemetry()
    members = author_initial_population(
        complete, candidate_model=_Candidate, template=TEMPLATE, k=5,
        domain_context="card", telemetry=tel)
    assert seen == [JOINT_PROMPT_TODAY]
    # and the default parse is the sealed one: one bare list, one label
    assert [m["genome"] for m in members] == [[1, 1, 1, 1, 1, 1]]
    assert tel.labels == ["joint"]
    assert tel.exploit_proposed == tel.explore_proposed == 0


def test_an_unknown_style_is_refused_by_name():
    with pytest.raises(ValueError, match="init style"):
        author_initial_population(
            lambda _p: "[]", candidate_model=_Candidate, template=TEMPLATE,
            k=4, domain_context="card", style="clever")


# --- the split ask ----------------------------------------------------------

def test_the_split_ask_is_one_call_naming_both_sub_asks_and_their_sizes():
    complete, seen = _capture(
        json.dumps({"exploit": [{"genome": [1, 1, 1, 1, 1, 1]}],
                    "explore": [{"genome": [0, 0, 0, 0, 0, 1]}]}))
    author_initial_population(
        complete, candidate_model=_Candidate, template=TEMPLATE, k=5,
        domain_context="card", style="split")
    assert len(seen) == 1, "the split ask must not cost a second call"
    prompt = seen[0]
    assert "EXPLOIT: exactly 3" in prompt and "EXPLORE: exactly 2" in prompt
    assert '{"exploit": [...], "explore": [...]}' in prompt
    assert "STRONGEST INDIVIDUAL" in prompt and "DISTINCT promising" in prompt


def test_split_members_come_back_exploit_first_labelled_and_sub_capped():
    reply = json.dumps({
        "exploit": [{"genome": [1, 1, 1, 1, 1, 1]},
                    {"genome": [1, 1, 1, 1, 1, 0]},
                    {"genome": [1, 1, 1, 1, 0, 0]}],   # over the k_exploit cap
        "explore": [{"genome": [0, 0, 0, 0, 0, 1]},
                    {"genome": [0, 1, 0, 1, 0, 1]},
                    {"genome": [1, 0, 1, 0, 1, 0]}],   # over the k_explore cap
    })
    tel = InitTelemetry()
    members = author_initial_population(
        lambda _p: reply, candidate_model=_Candidate, template=TEMPLATE, k=4,
        domain_context="card", telemetry=tel, style="split")
    # k = 4 -> 2 strongest bets, 2 coverage; the surplus is never read
    assert [m["genome"] for m in members] == [[1, 1, 1, 1, 1, 1],
                                              [1, 1, 1, 1, 1, 0],
                                              [0, 0, 0, 0, 0, 1],
                                              [0, 1, 0, 1, 0, 1]]
    assert tel.labels == ["exploit", "exploit", "explore", "explore"]
    assert tel.proposed == 4 and tel.accepted == 4
    assert tel.exploit_proposed == 2 and tel.exploit_accepted == 2
    assert tel.explore_proposed == 2 and tel.explore_accepted == 2
    assert tel.as_dict()["exploit_accepted"] == 2


def test_each_half_is_validated_exactly_as_a_joint_member_is():
    reply = json.dumps({
        "exploit": [{"genome": [1, 2, 1, 1, 1, 1]},     # 2 not in domain
                    {"genome": [1, 1, 1, 1, 1, 0]}],
        "explore": [{"dna": [0, 0, 0, 0, 0, 1]},        # wrong field
                    {"genome": [0, 1, 0, 1, 0, 1]}],
    })
    tel = InitTelemetry()
    members = author_initial_population(
        lambda _p: reply, candidate_model=_Candidate, template=TEMPLATE, k=4,
        domain_context="card", telemetry=tel, style="split")
    assert [m["genome"] for m in members] == [[1, 1, 1, 1, 1, 0],
                                              [0, 1, 0, 1, 0, 1]]
    assert tel.labels == ["exploit", "explore"]
    assert tel.rejected_out_of_domain == 1 and tel.rejected_shape == 1
    assert tel.exploit_accepted == 1 and tel.explore_accepted == 1


def test_a_reply_missing_a_half_is_read_as_a_bare_list_and_left_unlabelled():
    # Never repaired: what came back is read as the list it is, and it counts
    # towards NEITHER half -- a member nothing labelled is not evidence about
    # exploitation or about coverage.
    reply = json.dumps({"exploit": [{"genome": [1, 1, 1, 1, 1, 1]},
                                    {"genome": [0, 1, 0, 1, 0, 1]}]})
    tel = InitTelemetry()
    members = author_initial_population(
        lambda _p: reply, candidate_model=_Candidate, template=TEMPLATE, k=4,
        domain_context="card", telemetry=tel, style="split")
    assert [m["genome"] for m in members] == [[1, 1, 1, 1, 1, 1],
                                              [0, 1, 0, 1, 0, 1]]
    assert tel.labels == ["unlabeled", "unlabeled"]
    assert tel.accepted == 2
    assert tel.exploit_proposed == 0 and tel.explore_proposed == 0
    assert tel.unparseable == 0


def test_a_split_reply_with_neither_shape_is_unparseable_not_repaired():
    tel = InitTelemetry()
    assert author_initial_population(
        lambda _p: "I would start with a balanced spread.",
        candidate_model=_Candidate, template=TEMPLATE, k=4,
        domain_context="card", telemetry=tel, style="split") == []
    assert tel.unparseable == 1 and tel.accepted == 0


# --- the knob ---------------------------------------------------------------

def test_init_style_is_validated_and_refuses_to_be_a_silent_no_op():
    with pytest.raises(ValueError, match="init_style"):
        AuthorshipConfig(initialization="llm", init_style="halves")
    with pytest.raises(ValueError, match="initialization='off'"):
        AuthorshipConfig(init_style="split")
    assert AuthorshipConfig().init_style == "joint"


def test_build_authorship_carries_init_style_into_the_ask():
    complete, seen = _capture(
        json.dumps({"exploit": [{"genome": [1, 1, 1, 1, 1, 1]}],
                    "explore": [{"genome": [0, 0, 0, 0, 0, 1]}]}))
    said: list[str] = []
    policies = build_authorship(
        AuthorshipConfig(initialization="llm", init_style="split"),
        complete=complete, candidate_model=_Candidate,
        init_template=TEMPLATE, init_k=4, schema_text="card",
        announce=said.append)
    assert "EXPLOIT: exactly 2" in seen[0]
    assert [m["genome"] for m in policies.initial_proposals] == [
        [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1]]
    assert policies.init_author.telemetry.labels == ["exploit", "explore"]
    assert any("init_style='split'" in m for m in said), (
        "a run that changes the ask must say so"
    )


def test_build_authorship_defaults_to_the_joint_ask():
    complete, seen = _capture(json.dumps([{"genome": [1, 1, 1, 1, 1, 1]}]))
    build_authorship(
        AuthorshipConfig(initialization="llm"), complete=complete,
        candidate_model=_Candidate, init_template=TEMPLATE, init_k=5,
        schema_text="card")
    assert seen == [JOINT_PROMPT_TODAY]
