"""The authored generator: the model writes the sampler, the harness polices it.

Four properties carry the mechanism, and each is an enforcement test here.
Every emitted candidate is validated value-by-value and a reject costs a pool
slot, not the run. Mass generation charges ZERO evaluations -- structurally,
because this seam never sees the problem or the cache -- and the budget cap
still binds while the pool runs to thousands. The diversity guard measures
what a generator emitted as a SET, so a collapsed sampler is visible in the
telemetry instead of hiding behind a flat curve. And off is inert: the
byte-identity of the credential-free default is pinned by
``tests/test_fossil_stream.py``, which this seam must leave untouched.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.core.authored import CONTRACTS, authored_artifact
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.infrastructure.authored_runtime import AuthoredRuntime
from agent_evolve.policies.llm_generator import (
    AuthoredGenerator,
    GeneratorTelemetry,
    author_generator,
    candidate_key,
    render_generation_feedback,
    revise_generator,
    validate_pool,
)
from agent_evolve.policies.llm_surrogate import AuthorTelemetry
from agent_evolve.session.authorship import AuthorshipConfig, build_authorship
from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop

SPECS = [ObjectiveSpec("ones", "max")]
N = 6
TEMPLATE = {"genome": [0] * N}
DOMAINS = {f"genome[{i}]": [0, 1] for i in range(N)}


class _Candidate(BaseModel):
    genome: list[Literal[0, 1]]


class _Problem:
    candidate_model = _Candidate
    objectives = tuple(SPECS)

    def __init__(self) -> None:
        self.calls = 0
        self.stream: list[list[int]] = []

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return (dict(TEMPLATE),)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return tuple(config["genome"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        self.calls += 1
        self.stream.append(list(artifact))
        return {"ones": float(sum(artifact))}


def _generator(source: str, **kwargs) -> AuthoredGenerator:
    artifact = authored_artifact("generator", source, name="gen",
                                 authored_by="llm")
    return AuthoredGenerator(artifact, AuthoredRuntime(), **kwargs)


def _propose(generator: AuthoredGenerator, want: int, *, seed: int = 1,
             archive=(TEMPLATE,)):
    return generator.propose(
        template=dict(TEMPLATE), candidate_model=_Candidate, restriction=None,
        archive=[dict(c) for c in archive], want=want,
        rng=random.Random(seed), seed=seed)


UNIFORM_SOURCE = (
    "def propose(archive, n, domains, seed):\n"
    "    import random\n"
    "    r = random.Random(seed)\n"
    "    width = len(archive[0]['genome'])\n"
    "    return [{'genome': [r.choice(domains['genome[%d]' % j])\n"
    "                        for j in range(width)]} for _ in range(n)]\n"
)

FENCED_UNIFORM = f"```python\n{UNIFORM_SOURCE}```\n"

#: Distinct by construction: a sampler that cannot collide, which is what
#: separates "this generator collapsed" from "this space is small".
ENUMERATING_SOURCE = (
    "def propose(archive, n, domains, seed):\n"
    "    width = len(archive[0]['genome'])\n"
    "    out = []\n"
    "    for i in range(n):\n"
    "        bits = [(i + seed) >> j & 1 for j in range(width)]\n"
    "        out.append({'genome': [domains['genome[%d]' % j][bits[j]]\n"
    "                               for j in range(width)]})\n"
    "    return out\n"
)


# --- the contract ------------------------------------------------------------

def test_the_generator_contract_is_the_declared_entry_point():
    contract = CONTRACTS["generator"]
    assert (contract.entry_point, contract.positional_arity) == ("propose", 4)
    assert "propose(archive: list[dict], n: int" in contract.description


# --- (a) value-by-value validation, counted rejects, uniform shortfall --------

def test_validate_pool_rejects_each_bad_candidate_on_its_own_ground():
    report = validate_pool(
        [
            {"genome": [1, 1, 1, 0, 0, 1]},          # good
            {"genome": [1, 2, 0, 0, 0, 0]},          # 2 is not in the domain
            {"dna": [1, 1, 1, 1, 1, 1]},             # wrong field
            {"genome": [1, 1, 1]},                   # wrong length
            "not even a dict",                       # wrong type
            {"genome": [0, 1, 0, 1, 0, 1]},          # good
        ],
        template=dict(TEMPLATE), domains=DOMAINS,
    )
    assert [c["genome"] for c in report.accepted] == [[1, 1, 1, 0, 0, 1],
                                                      [0, 1, 0, 1, 0, 1]]
    assert report.rejected_out_of_domain == 1
    assert report.rejected_shape == 3
    assert report.emitted == 6


def test_an_undeclared_locus_must_keep_the_template_value():
    template = {"genome": [0] * N, "note": "fixed"}
    report = validate_pool(
        [{"genome": [1] * N, "note": "fixed"},
         {"genome": [1] * N, "note": "invented"}],
        template=template, domains=DOMAINS,
    )
    assert len(report.accepted) == 1 and report.rejected_out_of_domain == 1


def test_an_out_of_domain_generator_has_its_slots_filled_uniformly():
    generator = _generator(
        "def propose(archive, n, domains, seed):\n"
        "    good = {'genome': [1, 1, 1, 1, 1, 1]}\n"
        "    bad = {'genome': [7, 7, 7, 7, 7, 7]}\n"
        "    wrong_shape = {'genome': [1, 1]}\n"
        "    out = [good]\n"
        "    for i in range(n - 1):\n"
        "        out.append(bad if i % 2 else wrong_shape)\n"
        "    return out\n"
    )
    generator.repair = False                    # the per-CANDIDATE fallback
    pool = _propose(generator, want=2)          # pool_factor 4 -> 8 candidates
    assert len(pool) == 8, "the pool must be topped up to the size asked for"
    assert all(set(c["genome"]) <= {0, 1} and len(c["genome"]) == N
               for c in pool), "an invalid candidate reached the pool"
    counters = generator.telemetry.as_dict()
    assert counters["emitted"] == 8 and counters["accepted"] == 1
    assert counters["rejected_out_of_domain"] == 3
    assert counters["rejected_shape"] == 4
    assert counters["filled_uniform"] == 7


def test_the_per_locus_fallback_keeps_what_the_model_got_right():
    """The same emission under the emit harness's repair, which is the default.

    Nothing invalid reaches the pool either way -- that invariant is the
    gate's, not the fallback's. What changes is WHOSE draw fills the slot: a
    candidate whose every locus is out of domain still cannot be repaired
    into anything of the model's, but one that got some loci right keeps
    them, and the harness's share is counted rather than absorbed.
    """

    generator = _generator(
        "def propose(archive, n, domains, seed):\n"
        "    good = {'genome': [1, 1, 1, 1, 1, 1]}\n"
        "    mixed = {'genome': [1, 7, 0, 7, 1, 0]}\n"
        "    short = {'genome': [1, 0, 1]}\n"
        "    return [good, mixed, short] + [{'genome': [0, 1] * 3}\n"
        "                                   for _ in range(n - 3)]\n",
        max_revisions=0,
    )
    pool = _propose(generator, want=2)
    assert all(set(c["genome"]) <= {0, 1} and len(c["genome"]) == N
               for c in pool), "an invalid candidate reached the pool"
    counters = generator.telemetry.as_dict()
    assert counters["rejected_shape"] == 0 and counters["rejected_out_of_domain"] == 0
    assert counters["repaired"] == 2, "the mixed and the short one were assembled"
    # Four loci decided by the harness: two out-of-domain in `mixed`, three
    # positions missing from `short` -- minus nothing, because `short`'s three
    # supplied positions were all admissible.
    assert counters["repaired_loci"] == 5
    census = generator.census
    assert census.out_of_domain_by_locus == {"genome[1]": 1, "genome[3]": 1}
    assert census.samples["domain:genome[1]"] == 7
    assert set(census.repaired_by_locus) == {
        "genome[1]", "genome[3]", "genome[3]", "genome[4]", "genome[5]"}


def test_a_crashing_generator_costs_a_pool_not_a_run():
    generator = _generator(
        "def propose(archive, n, domains, seed):\n"
        "    raise ValueError('no')\n"
    )
    pool = _propose(generator, want=3)
    assert len(pool) == 12 and generator.telemetry.runtime_failures == 1
    assert generator.telemetry.filled_uniform == 12


# --- (b) mass generation charges zero evaluations ----------------------------

def test_this_seam_has_no_route_to_an_evaluation():
    import inspect

    import agent_evolve.policies.llm_generator as module

    source = inspect.getsource(module)
    for forbidden in ("session.evaluate", "EvaluationCache", "evaluate_batch",
                      "agent_evolve.contract"):
        assert forbidden not in source, (
            f"llm_generator imports {forbidden}: mass generation could then "
            "reach the budget by something other than the loop measuring it"
        )
    taken = set(inspect.signature(AuthoredGenerator.propose).parameters)
    assert not taken & {"problem", "cache", "evaluate", "budget"}, taken


def test_a_pool_of_five_thousand_charges_nothing_and_the_budget_still_binds(
        monkeypatch):
    monkeypatch.setattr(
        "agent_evolve.integrations.completion.completion_for",
        lambda model, settings=None, **kwargs: (lambda prompt: FENCED_UNIFORM))
    problem = _Problem()
    result = optimize(
        problem, budget=16, seed=5, proposer="llm",
        authorship=AuthorshipConfig(generation="llm",
                                    generation_pool_size=5000))
    assert problem.calls <= 16, "mass generation spent evaluations"
    assert result.evaluations <= 16
    counters = {m.mechanism: m.counters for m in result.telemetry.mechanisms}
    emitted = counters["authored_generator"]["emitted"]
    assert emitted >= 5000, f"the pool never reached mass scale ({emitted})"
    assert result.telemetry.real_evaluations <= 16
    assert emitted > 100 * result.telemetry.real_evaluations, (
        "generation is supposed to be free; this ratio is the evidence"
    )


def test_the_pool_never_shrinks_below_what_the_generation_can_afford():
    generator = _generator(UNIFORM_SOURCE, pool_size=2)
    assert len(_propose(generator, want=9)) == 9
    assert generator.pool_for(3) == 3


# --- (c) off is inert --------------------------------------------------------

def test_the_generator_is_off_by_default_everywhere():
    assert GeneticConfig().generator is None
    assert AuthorshipConfig().generation == "off"
    policies = build_authorship(AuthorshipConfig())
    assert policies.generator is None and policies.generator_author is None
    assert build_authorship(AuthorshipConfig(generation="llm"),
                            complete=None).generator is None, (
        "with no model call the generator must not exist at all"
    )


def test_failed_authoring_still_reports_its_counters(monkeypatch):
    monkeypatch.setattr(
        "agent_evolve.integrations.completion.completion_for",
        lambda model, settings=None, **kwargs: (lambda prompt: "no code here"))
    result = optimize(_Problem(), budget=10, seed=2, proposer="llm",
                      authorship="generation-llm")
    mechanisms = {m.mechanism: m.counters for m in result.telemetry.mechanisms}
    assert mechanisms["generator_author"]["no_code_block"] == 2, (
        "an authoring failure must leave counters, not just an announcement"
    )
    assert "authored_generator" not in mechanisms
    assert result.evaluations <= 10


def test_asking_for_generation_without_a_model_leaves_the_stream_unchanged():
    # The fossil test pins the default stream byte-for-byte; this pins the
    # weaker but sharper claim that the new knob, unusable, changes nothing.
    off = _Problem()
    optimize(off, budget=16, seed=11, authorship=AuthorshipConfig())
    asked = _Problem()
    optimize(asked, budget=16, seed=11,
             authorship=AuthorshipConfig(generation="llm"))
    assert asked.stream == off.stream


def test_generation_and_operators_are_refused_rather_than_silently_merged():
    with pytest.raises(ValueError, match="construct the generation"):
        AuthorshipConfig(generation="llm", operators="llm")
    with pytest.raises(ValueError, match="construct the generation"):
        run_genetic_loop(
            problem=_Problem(),
            config=GeneticConfig(generator=object(), portfolio=object()))


# --- (d) the diversity / novelty guard ---------------------------------------

def test_duplicates_and_archive_overlap_are_measured_as_rates():
    repeat = {"genome": [1, 0, 1, 0, 1, 0]}
    report = validate_pool(
        [dict(repeat), dict(repeat), {"genome": [0] * N},
         {"genome": [1] * N}],
        template=dict(TEMPLATE), domains=DOMAINS,
        seen={candidate_key({"genome": [0] * N})},
    )
    assert report.duplicates == 1 and report.archive_overlap == 1
    assert report.duplicate_rate == pytest.approx(0.25)
    assert report.archive_overlap_rate == pytest.approx(0.25)
    assert report.novelty_rate == pytest.approx(0.5)
    assert report.acceptance_rate == pytest.approx(1.0), (
        "everything emitted was in-domain; only the SET was degenerate"
    )
    assert [c["genome"] for c in report.accepted] == [[1, 0, 1, 0, 1, 0],
                                                      [1] * N]


def test_a_degenerate_generator_is_visible_in_the_telemetry():
    generator = _generator(
        "def propose(archive, n, domains, seed):\n"
        "    return [{'genome': [1, 0, 1, 0, 1, 0]} for _ in range(n)]\n",
        max_revisions=0,
    )
    pool = _propose(generator, want=4)
    counters = generator.telemetry.as_dict()
    assert counters["emitted"] == 16 and counters["accepted"] == 1
    assert counters["duplicates"] == 15, "the collapse was not counted"
    assert counters["filled_uniform"] == 15
    assert len({candidate_key(c) for c in pool}) > 1, (
        "the guard must not hand a collapsed pool back to the loop"
    )
    note = generator.note()
    assert note["duplicate_rate"] == pytest.approx(15 / 16)

    # Second batch: the one distinct configuration is now in the archive, so
    # the same reply is overlap rather than novelty -- once, with the fifteen
    # repeats still attributed to the collapse that produced them.
    generator.note_archive([{"genome": [1, 0, 1, 0, 1, 0]}])
    _propose(generator, want=4, seed=2)
    assert generator.telemetry.archive_overlap == 1
    assert generator.telemetry.duplicates == 30
    assert generator.note()["archive_overlap_rate"] == pytest.approx(1 / 16)


def test_the_guard_reaches_the_run_history(monkeypatch):
    collapsed = (
        "```python\n"
        "def propose(archive, n, domains, seed):\n"
        "    return [{'genome': [1, 1, 0, 0, 1, 1]} for _ in range(n)]\n"
        "```\n"
    )
    monkeypatch.setattr(
        "agent_evolve.integrations.completion.completion_for",
        lambda model, settings=None, **kwargs: (lambda prompt: collapsed))
    result = optimize(_Problem(), budget=12, seed=3, proposer="llm",
                      authorship="generation-llm")
    notes = [h["generate"] for h in result.history if "generate" in h]
    assert notes, "no mass-generation record reached the history"
    assert any(note["duplicate_rate"] > 0.5 for note in notes)
    rows = [m for m in result.telemetry.mechanisms
            if m.mechanism == "authored_generator"]
    assert rows and rows[0].counters["duplicates"] > 0


# --- (e) authoring and revision face the same gate ---------------------------

@pytest.mark.parametrize("reply,counter", [
    ("no fenced block here, just prose", "no_code_block"),
    ("```python\nimport os\ndef propose(archive, n, domains, seed):\n"
     "    return []\n```", "forbidden_import"),
    ("```python\ndef generate(archive, n, domains, seed):\n"
     "    return []\n```", "wrong_entry_point"),
    ("```python\ndef propose(archive, n, domains, seed)\n    return []\n```",
     "unparseable"),
])
def test_authoring_rejects_whole_replies_and_counts_why(reply, counter):
    telemetry = AuthorTelemetry()
    assert author_generator(lambda _p: reply, objectives=SPECS,
                            schema_text="card", attempts=1,
                            telemetry=telemetry) is None
    assert telemetry.as_dict()[counter] == 1
    assert telemetry.accepted == 0


@pytest.mark.parametrize("reply,counter", [
    ("still no block", "no_code_block"),
    ("```python\nimport socket\ndef propose(archive, n, domains, seed):\n"
     "    return []\n```", "forbidden_import"),
    ("```python\ndef sample(archive, n, domains, seed):\n    return []\n```",
     "wrong_entry_point"),
])
def test_revision_faces_the_identical_gate(reply, counter):
    original = authored_artifact("generator", UNIFORM_SOURCE, name="gen",
                                 authored_by="llm")
    telemetry = AuthorTelemetry()
    assert revise_generator(lambda _p: reply, artifact=original,
                            feedback="it collapsed", attempts=1,
                            telemetry=telemetry) is None
    assert telemetry.as_dict()[counter] == 1


def test_an_accepted_revision_keeps_the_authoring_lineage():
    original = authored_artifact("generator", UNIFORM_SOURCE, name="gen",
                                 authored_by="llm")
    telemetry = AuthorTelemetry()
    revised = revise_generator(
        lambda prompt: FENCED_UNIFORM if "measured" in prompt else "",
        artifact=original, feedback="fix it", telemetry=telemetry)
    assert revised is not None and revised.name == "gen_rev"
    assert revised.entry_point == "propose" and telemetry.accepted == 1


def test_revision_fires_on_a_measured_defect_and_never_on_a_clean_run():
    calls: list[str] = []

    def revise(artifact, feedback):
        calls.append(feedback)
        return authored_artifact("generator", UNIFORM_SOURCE, name="fixed",
                                 authored_by="llm")

    clean = _generator(ENUMERATING_SOURCE, revise=revise)
    _propose(clean, want=3)
    _propose(clean, want=3, seed=2)
    assert not clean.deficient() and calls == []

    broken = _generator(
        "def propose(archive, n, domains, seed):\n"
        "    return [{'genome': [9] * 6} for _ in range(n)]\n",
        revise=revise)
    _propose(broken, want=3)
    assert broken.deficient()
    _propose(broken, want=3, seed=2)
    assert broken.telemetry.revisions == 1
    assert broken.telemetry.revisions_accepted == 1
    assert broken.artifact.name == "fixed"
    assert "value outside its declared domain" in calls[0]
    _propose(broken, want=3, seed=3)
    assert broken.telemetry.revisions == 1, "max_revisions is a cap, not a rate"


def test_the_feedback_carries_acceptance_novelty_and_measured_quality():
    telemetry = GeneratorTelemetry(
        batches=2, emitted=100, accepted=60, rejected_shape=10,
        rejected_out_of_domain=10, duplicates=15, archive_overlap=5,
        filled_uniform=40, measured=8, survived=2)
    text = render_generation_feedback(
        telemetry, None,
        [({"genome": [1] * N}, {"ones": 6.0})])
    for expected in ("emitted: 100", "accepted by the harness: 60",
                     "duplicate within the batch: 15",
                     "already measured in this run: 5",
                     "uniform random draws: 40",
                     "survived selection into the next population: 2",
                     "ones=6"):
        assert expected in text, text


# --- end to end --------------------------------------------------------------

def test_an_authored_generator_runs_the_generation_end_to_end(monkeypatch):
    monkeypatch.setattr(
        "agent_evolve.integrations.completion.completion_for",
        lambda model, settings=None, **kwargs: (lambda prompt: FENCED_UNIFORM))
    problem = _Problem()
    # revisions ablated to 0: this cell measures ONE-SHOT authoring, which is
    # the arm the ladder rows are defined on.
    result = optimize(problem, budget=20, seed=8, proposer="llm",
                      authorship=AuthorshipConfig(generation="llm",
                                                  generation_revisions=0))
    assert problem.calls <= 20
    mechanisms = {m.mechanism: m.counters for m in result.telemetry.mechanisms}
    assert mechanisms["generator_author"]["accepted"] == 1, (
        "one-shot authoring must make exactly one accepted artifact"
    )
    counters = mechanisms["authored_generator"]
    assert counters["batches"] > 0 and counters["accepted"] > 0
    assert counters["measured"] > 0
    assert counters["revisions"] == 0
    assert counters["rejected_out_of_domain"] == 0
    assert counters["rejected_shape"] == 0


def test_the_generative_preset_screens_the_authored_pool(monkeypatch):
    surrogate = (
        "```python\n"
        "def fit_predict(train_x, train_y, test_x):\n"
        "    return [{'ones': float(sum(row['genome']))} for row in test_x]\n"
        "```\n"
    )

    def _canned(model, settings=None, **kwargs):
        def complete(prompt: str) -> str:
            return FENCED_UNIFORM if "GENERATOR" in prompt else surrogate
        return complete

    monkeypatch.setattr(
        "agent_evolve.integrations.completion.completion_for", _canned)
    problem = _Problem()
    result = optimize(problem, budget=24, seed=4, proposer="llm",
                      authorship="generative")
    assert problem.calls <= 24
    screens = [h["screen"] for h in result.history if "screen" in h]
    generated = [h["generate"] for h in result.history if "generate" in h]
    assert screens and generated
    assert any(note["active"] and note["pool"] > note["held_out"]
               for note in screens), (
        "the screen never ordered the generator's pool"
    )
    assert result.telemetry.virtual_evaluations > 0
    assert result.telemetry.real_evaluations <= 24
