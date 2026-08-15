"""The measurement-conditioned channel: what it shows, what it emits, what refuses it.

Every mechanism in this package until now conditioned the model on MEANING.
This one conditions it on the run's own measurements, and the tests here
defend the three things that makes true rather than claimed:

* the channel is OFF by default and off is the seam every sealed row ran;
* what the model is SHOWN is a pure function of the measured trace, swappable
  through one declared parameter, so a shuffled-evidence control can be built
  without editing the product;
* a model-authored RESTRICTION is refused by a gate rather than trusted, and
  the one unrecoverable failure -- excluding a measured optimum -- is checked
  against the run's own front.
"""

from __future__ import annotations

import random
from typing import Literal

import pytest
from pydantic import BaseModel

from agent_evolve.core.authored import authored_artifact
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.infrastructure.authored_runtime import AuthoredRuntime
from agent_evolve.policies.llm_generator import AuthoredGenerator
from agent_evolve.policies.measurement_evidence import (
    LocusPrior,
    admit_locus_prior,
    apply_locus_prior,
    front_of,
    locus_effects,
    parse_locus_prior,
    render_measurement_evidence,
    spearman,
)
from agent_evolve.session.authorship import AuthorshipConfig

SOURCE = """
def propose(archive, n, domains, seed):
    import random
    r = random.Random(seed)
    keys = sorted(domains)
    return [{k: r.choice(domains[k]) for k in keys} for _ in range(n)]
"""

SOURCE_V2 = SOURCE.replace("keys = sorted(domains)",
                           "keys = sorted(domains)  # revised")


class Candidate(BaseModel):
    knob: Literal["a", "b", "c", "d"]
    other: Literal["p", "q"]


TEMPLATE = {"knob": "a", "other": "p"}
SPECS = (ObjectiveSpec("cost", "min", "the measured cost"),)
DOMAINS = {"knob": ["a", "b", "c", "d"], "other": ["p", "q"]}


def artifact(source=SOURCE, name="gen"):
    return authored_artifact("generator", source, name=name, authored_by="llm")


def generator(**kwargs):
    kwargs.setdefault("objectives", SPECS)
    return AuthoredGenerator(artifact(), AuthoredRuntime(), pool_factor=2,
                             **kwargs)


def drive(gen, generations=6, want=4, cost=None):
    """Run *generations* propose/measure rounds. Returns every pool drawn."""

    rng = random.Random(0)
    pools = []
    for g in range(generations):
        pool = gen.propose(template=TEMPLATE, candidate_model=Candidate,
                           restriction=None, archive=[], want=want, rng=rng,
                           seed=g)
        pools.append(list(pool))
        for config in pool[:want]:
            value = (cost or (lambda c: 1.0))(config)
            gen.record_measured(config, survived=value < 5.0,
                                objectives={"cost": value})
    return pools


def seed_front(gen, k=4):
    """Record a deterministic trace whose front is exactly ``knob == "a"``.

    The gate refuses any prior that excludes a measured front member, so a
    test about what an ADMITTED prior does must not leave the front to a
    random draw.
    """

    for i in range(k):
        knob = DOMAINS["knob"][i % 4]
        gen.record_measured({"knob": knob, "other": "p"},
                            survived=(knob == "a"),
                            objectives={"cost": float(DOMAINS["knob"].index(knob))})


def rows(n=8, cost=None):
    out = []
    for i in range(n):
        config = {"knob": DOMAINS["knob"][i % 4], "other": DOMAINS["other"][i % 2]}
        value = (cost or (lambda c: float(DOMAINS["knob"].index(c["knob"]))))(config)
        out.append((config, {"cost": value}, value < 2.0))
    return out


# ------------------------------------------------------------------ OFF is off

def test_off_by_default_makes_no_evidence_call_and_no_record():
    calls = []
    gen = generator(reauthor=lambda a, e: calls.append(e))
    drive(gen)
    assert gen.reauthor_every == 0
    assert calls == []
    assert gen.evidence_log == []
    assert gen.telemetry.reauthorings == 0
    assert "evidence" not in gen.note()


def test_off_draws_the_same_pools_as_a_generator_built_without_the_knobs():
    plain = AuthoredGenerator(artifact(), AuthoredRuntime(), pool_factor=2)
    conditioned_off = generator(reauthor=lambda a, e: None, reauthor_every=0,
                                max_reauthorings=3)
    assert drive(plain) == drive(conditioned_off)


# ---------------------------------------------------------------- the cadence

def test_the_cadence_is_charged_evaluations_and_it_is_what_it_says():
    seen = []
    gen = generator(reauthor=lambda a, e: seen.append(gen.telemetry.measured),
                    reauthor_every=8, max_reauthorings=4)
    drive(gen, generations=6, want=4)          # 24 charged evaluations
    # A pool is drawn BEFORE the generation it feeds is measured, so the
    # ticks land at the propose calls that follow the 8th and 16th charge.
    assert seen == [8, 16]


def test_a_cadence_never_fires_before_the_archive_has_grown_by_it():
    seen = []
    gen = generator(reauthor=lambda a, e: seen.append(1),
                    reauthor_every=100, max_reauthorings=4)
    drive(gen, generations=6, want=4)
    assert seen == []


def test_the_reauthoring_budget_binds():
    seen = []
    gen = generator(reauthor=lambda a, e: seen.append(1),
                    reauthor_every=4, max_reauthorings=2)
    drive(gen, generations=8, want=4)
    assert len(seen) == 2


def test_a_negative_cadence_is_refused():
    with pytest.raises(ValueError):
        generator(reauthor_every=-1)


# ------------------------------------------------- what the model actually sees

def test_the_evidence_is_this_run_s_own_measurements():
    shown = []
    gen = generator(reauthor=lambda a, e: shown.append(e) or None,
                    reauthor_every=4, max_reauthorings=1)
    drive(gen, generations=3, want=4,
          cost=lambda c: 0.5 if c["knob"] == "a" else 7.0)
    assert shown, "the cadence must have fired"
    text = shown[0]
    assert "the current non-dominated front" in text
    assert "cost=0.5" in text                 # a measured value, not a summary
    assert "which parameter moves which cost" in text


def test_the_evidence_view_is_the_control_seam_and_changes_only_what_is_shown():
    other_run = rows(6, cost=lambda c: 42.0)
    shown = []
    gen = generator(reauthor=lambda a, e: shown.append(e) or None,
                    reauthor_every=4, max_reauthorings=1,
                    evidence_view=lambda _rows: other_run)
    drive(gen, generations=3, want=4)
    assert "cost=42" in shown[0]
    assert gen.evidence_log[0]["rows_shown"] == len(other_run)

    honest = []
    plain = generator(reauthor=lambda a, e: honest.append(e) or None,
                      reauthor_every=4, max_reauthorings=1)
    drive(plain, generations=3, want=4)
    assert (gen.evidence_log[0]["evidence_sha256"]
            != plain.evidence_log[0]["evidence_sha256"])


def test_an_evidence_view_that_throws_cannot_kill_the_run():
    def boom(_rows):
        raise RuntimeError("control arms have bugs too")

    gen = generator(reauthor=lambda a, e: None, reauthor_every=4,
                    max_reauthorings=2, evidence_view=boom)
    drive(gen, generations=4, want=4)
    assert gen.telemetry.reauthorings == 0     # nothing to reason over
    assert gen.telemetry.batches == 4


# ----------------------------------------------- telemetry as correctness

def test_every_evidence_call_records_what_was_seen_and_whether_it_landed():
    replies = [artifact(SOURCE_V2, "v2"), None]
    gen = generator(reauthor=lambda a, e: replies.pop(0),
                    reauthor_every=4, max_reauthorings=2)
    drive(gen, generations=4, want=4)
    assert len(gen.evidence_log) == 2
    first, second = gen.evidence_log
    assert first["accepted"] is True and first["emitted"]
    assert first["replaced"] != first["emitted"]
    assert second["accepted"] is False and second["emitted"] is None
    for record in gen.evidence_log:
        assert len(record["evidence_sha256"]) == 64
        assert record["rows_shown"] > 0
    assert gen.telemetry.reauthorings == 2
    assert gen.telemetry.reauthorings_accepted == 1


def test_an_unusable_reauthoring_keeps_the_artifact_already_in_hand():
    gen = generator(reauthor=lambda a, e: None, reauthor_every=4,
                    max_reauthorings=2)
    before = gen.artifact.source_sha256
    drive(gen, generations=4, want=4)
    assert gen.artifact.source_sha256 == before


def test_a_reauthoring_that_raises_does_not_kill_the_run():
    def boom(_a, _e):
        raise RuntimeError("the provider fell over")

    gen = generator(reauthor=boom, reauthor_every=4, max_reauthorings=2)
    drive(gen, generations=4, want=4)
    assert gen.telemetry.reauthorings == 2
    assert gen.telemetry.reauthorings_accepted == 0
    assert all(r["accepted"] is False for r in gen.evidence_log)


def test_the_note_carries_each_evidence_record_exactly_once():
    gen = generator(reauthor=lambda a, e: None, reauthor_every=4,
                    max_reauthorings=3)
    rng = random.Random(0)
    seen = []
    for g in range(4):
        pool = gen.propose(template=TEMPLATE, candidate_model=Candidate,
                           restriction=None, archive=[], want=4, rng=rng, seed=g)
        for config in pool[:4]:
            gen.record_measured(config, survived=True, objectives={"cost": 1.0})
        seen.extend(gen.note()["evidence"])
    assert len(seen) == len(gen.evidence_log) == 3


# ------------------------------------------------------- the evidence itself

def test_spearman_reports_undefined_rather_than_zero():
    assert spearman([1, 2], [1, 2]) is None            # too few pairs
    assert spearman([1, 1, 1], [1, 2, 3]) is None      # constant predictor
    assert spearman([1, 2, 3], [5, 5, 5]) is None      # constant response
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_locus_effects_puts_the_dominant_knob_first():
    effects = locus_effects(rows(12), SPECS, DOMAINS)
    assert effects[0].locus == "knob"
    assert effects[0].correlation["cost"] == pytest.approx(1.0)
    assert effects[0].strength > effects[1].strength


def test_locus_effects_reports_the_values_never_measured():
    partial = [({"knob": "a", "other": "p"}, {"cost": 1.0}, True),
               ({"knob": "b", "other": "p"}, {"cost": 2.0}, False),
               ({"knob": "a", "other": "q"}, {"cost": 3.0}, False)]
    effect = {e.locus: e for e in locus_effects(partial, SPECS, DOMAINS)}["knob"]
    assert set(effect.unmeasured) == {"c", "d"}


def test_rendering_the_same_rows_twice_is_the_same_bytes():
    body = rows(9)
    assert (render_measurement_evidence(body, SPECS, DOMAINS)
            == render_measurement_evidence(body, SPECS, DOMAINS))


def test_an_empty_trace_renders_as_an_empty_trace():
    assert "no candidate has been measured" in render_measurement_evidence(
        [], SPECS, DOMAINS)


# ---------------------------------------------------------------- the gate

def _front():
    body = rows(8)
    return front_of(body, SPECS)


@pytest.mark.parametrize("reply, reason", [
    ("not json at all", "unparsed"),
    ('{"restrict": {}}', "empty"),
    ('{"restrict": {"nosuch": ["a"]}}', "undeclared parameter"),
    ('{"restrict": {"knob": ["zzz"]}}', "undeclared value"),
    ('{"restrict": {"knob": []}}', "undeclared value"),
])
def test_the_gate_refuses_and_says_why(reply, reason):
    verdict = admit_locus_prior(parse_locus_prior(reply), domains=DOMAINS)
    assert not verdict.admitted
    assert verdict.reason.startswith(reason)


def test_the_gate_refuses_a_prior_that_excludes_a_measured_front_member():
    front = _front()
    best = front[0][0]["knob"]
    other = [v for v in DOMAINS["knob"] if v != best][:2]
    verdict = admit_locus_prior(
        LocusPrior(restrict={"knob": tuple(other)}), domains=DOMAINS, front=front)
    assert not verdict.admitted
    assert "excludes a measured front member" in verdict.reason


def test_the_gate_refuses_an_over_narrow_prior():
    wide = {f"k{i}": [f"v{j}" for j in range(10)] for i in range(3)}
    prior = LocusPrior(restrict={name: ("v0",) for name in wide})
    assert not admit_locus_prior(prior, domains=wide,
                                 max_narrowing_log10=2.0).admitted
    assert admit_locus_prior(prior, domains=wide,
                             max_narrowing_log10=4.0).admitted


def test_the_gate_admits_a_narrowing_that_keeps_the_front():
    front = _front()
    best = front[0][0]["knob"]
    verdict = admit_locus_prior(
        LocusPrior(restrict={"knob": (best, "b")}), domains=DOMAINS, front=front)
    assert verdict.admitted
    assert verdict.narrowing > 0


def test_one_bad_entry_refuses_the_whole_reply_rather_than_being_dropped():
    prior = parse_locus_prior(
        '{"restrict": {"knob": ["a", "b"], "other": ["nope"]}}')
    verdict = admit_locus_prior(prior, domains=DOMAINS)
    assert not verdict.admitted
    assert verdict.prior is not None            # judged, not silently repaired


def test_applying_a_prior_narrows_and_never_empties():
    narrowed = apply_locus_prior(DOMAINS, LocusPrior(restrict={"knob": ("a", "b")}))
    assert narrowed["knob"] == ["a", "b"]
    assert narrowed["other"] == ["p", "q"]
    untouched = apply_locus_prior(DOMAINS, LocusPrior(restrict={"knob": ("zz",)}))
    assert untouched["knob"] == DOMAINS["knob"]


# ------------------------------------------------- the prior inside the loop

def test_an_admitted_prior_moves_where_the_generator_SAMPLES():
    gen = generator(reauthor_every=4, max_priors=1,
                    prior_author=lambda p: '{"restrict": {"knob": ["a", "b"]}}')
    seed_front(gen)
    gen.propose(template=TEMPLATE, candidate_model=Candidate, restriction=None,
                archive=[], want=4, rng=random.Random(0), seed=1)
    assert gen.telemetry.priors_admitted == 1
    # What the GENERATOR emitted, not the pool it was topped up into: a short
    # pool is filled with schema-uniform draws over the DECLARED domains,
    # which is the credential-free fallback and not the generator's output.
    assert gen.last_report is not None and gen.last_report.accepted
    assert all(c["knob"] in ("a", "b") for c in gen.last_report.accepted)


def test_a_prior_narrows_sampling_and_not_what_is_LEGAL():
    # The generator ignores the narrowed domains entirely and keeps drawing
    # from the whole space. Those candidates are still schema-legal, so they
    # must be admitted -- a prior must never manufacture a rejection and fire
    # the defect-repair channel on a generator that did nothing wrong.
    ignore_domains = """
def propose(archive, n, domains, seed):
    import random
    r = random.Random(seed)
    return [{"knob": r.choice(["a", "b", "c", "d"]),
             "other": r.choice(["p", "q"])} for _ in range(n)]
"""
    gen = AuthoredGenerator(
        artifact(ignore_domains), AuthoredRuntime(), pool_factor=2,
        objectives=SPECS, reauthor_every=4, max_priors=1,
        prior_author=lambda p: '{"restrict": {"knob": ["a"]}}')
    seed_front(gen)
    drive(gen, generations=4, want=4,
          cost=lambda c: float(DOMAINS["knob"].index(c["knob"])))
    assert gen.telemetry.priors_admitted == 1
    assert gen.telemetry.rejected_out_of_domain == 0


def test_a_prior_that_stops_paying_is_unwound():
    gen = generator(reauthor_every=4, max_priors=1, prior_unwind_batches=1,
                    prior_author=lambda p: '{"restrict": {"knob": ["a", "b"]}}')
    seed_front(gen)
    # Nothing measured after the prior lands ever survives, so the bet stops
    # paying and the generator must finish on the declared domains.
    drive(gen, generations=4, want=4, cost=lambda c: 99.0)
    assert gen.telemetry.priors_admitted == 1
    assert gen.telemetry.priors_unwound == 1


def test_a_refused_prior_is_counted_and_never_installed():
    gen = generator(reauthor_every=4, max_priors=1,
                    prior_author=lambda p: '{"restrict": {"knob": ["nope"]}}')
    drive(gen, generations=6, want=4)
    assert gen.telemetry.priors_proposed == 1
    assert gen.telemetry.priors_refused == 1
    assert gen.telemetry.priors_admitted == 0
    assert gen.evidence_log[-1]["verdict"]["admitted"] is False


def test_a_prior_author_that_raises_is_a_refusal_and_not_a_crash():
    def boom(_prompt):
        raise RuntimeError("provider down")

    gen = generator(reauthor_every=4, max_priors=1, prior_author=boom)
    drive(gen, generations=6, want=4)
    assert gen.telemetry.priors_refused == 1


def test_a_prior_needs_a_declared_cadence():
    with pytest.raises(ValueError):
        generator(prior_author=lambda p: "{}", reauthor_every=0)


# ------------------------------------------------------------- the config knob

def test_the_config_refuses_a_prior_without_a_cadence():
    with pytest.raises(ValueError):
        AuthorshipConfig(generation="llm", generation_locus_prior=True)


def test_the_config_refuses_the_channel_without_an_authored_generator():
    with pytest.raises(ValueError):
        AuthorshipConfig(generation_reauthor_every=8)


def test_the_config_defaults_the_channel_off():
    config = AuthorshipConfig(generation="llm")
    assert config.generation_reauthor_every == 0
    assert config.generation_reauthorings == 0
    assert config.generation_locus_prior is False


def test_build_authorship_wires_the_channel_and_leaves_it_off_by_default():
    from agent_evolve.session.authorship import build_authorship

    def complete(prompt):
        return f"```python\n{SOURCE}\n```"

    off = build_authorship(AuthorshipConfig(generation="llm"),
                           complete=complete, objectives=SPECS,
                           schema_text="x").generator
    assert off.reauthor is None and off.prior_author is None
    assert off.reauthor_every == 0

    on = build_authorship(
        AuthorshipConfig(generation="llm", generation_reauthor_every=8,
                         generation_reauthorings=2,
                         generation_locus_prior=True),
        complete=complete, objectives=SPECS, schema_text="x").generator
    assert on.reauthor is not None and on.prior_author is not None
    assert on.reauthor_every == 8 and on.max_reauthorings == 2
    assert tuple(on.objectives) == tuple(SPECS)
