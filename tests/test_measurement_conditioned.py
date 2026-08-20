"""The measurement-conditioned channel: what it shows, what it emits, what refuses it.

Every mechanism in this package until now conditioned the model on MEANING.
This one conditions it on the run's own measurements, and the tests here
defend the three things that makes true rather than claimed:

* the channel is OFF by default and off is the seam every sealed row ran;
* what the model is SHOWN is a pure function of the measured trace, swappable
  through one declared parameter, so a shuffled-evidence control can be built
  without editing the product;
* a model-authored RESTRICTION is refused by a gate rather than trusted, is
  GRADED -- per-locus value weights that bias sampling and exclude nothing,
  so the one unrecoverable failure of the hard form (excluding a measured
  optimum) is structurally impossible -- capped in concentration, and
  unwound when it stops producing survivors.
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
    ELITE_ENRICHMENT_FLOOR,
    ELITE_MIN_COUNT,
    MIN_EVIDENCE_ROWS,
    WeightedProposal,
    admit_weighted_restriction,
    apply_weighted_restriction,
    front_of,
    locus_effects,
    parse_weighted_restriction,
    render_elite_table,
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
    # `evidence_min_rows == reauthor_every` is the PURE-CADENCE rule: the
    # first call waits for the same count every later one does. It is the
    # pre-W11 behaviour, kept reachable as a declared configuration.
    seen = []
    gen = generator(reauthor=lambda a, e: seen.append(gen.telemetry.measured),
                    reauthor_every=8, evidence_min_rows=8, max_reauthorings=4)
    drive(gen, generations=6, want=4)          # 24 charged evaluations
    # A pool is drawn BEFORE the generation it feeds is measured, so the
    # ticks land at the propose calls that follow the 8th and 16th charge.
    assert seen == [8, 16]


def test_a_cadence_never_fires_before_the_archive_has_grown_by_it():
    seen = []
    gen = generator(reauthor=lambda a, e: seen.append(1), evidence_min_rows=100,
                    reauthor_every=100, max_reauthorings=4)
    drive(gen, generations=6, want=4)
    assert seen == []


# -------------------------------------- W11: the channel speaks when it CAN

def test_the_first_call_waits_on_EVIDENCE_and_the_rest_on_the_cadence():
    """The defect, stated as a rule: a cadence cannot also be a start gate.

    `reauthor_every` says how often an evidence call recurs. Read as "wait
    for that many of the generator's own children" it also delayed the FIRST
    call past evidence the run already held -- W11, measured as a locus prior
    authored at a median charge of 40 against a 43.5-charge target. The first
    call now fires at `evidence_min_rows` rows (default: the fewest a
    determinable effect can be computed from); every later one at the cadence.
    """

    seen = []
    gen = generator(reauthor=lambda a, e: seen.append(len(gen._rows)),
                    reauthor_every=8, max_reauthorings=4)
    assert gen.evidence_min_rows == MIN_EVIDENCE_ROWS
    drive(gen, generations=6, want=4)          # 24 charged evaluations
    # First tick at the earliest propose holding >= 3 rows (there are 4 by
    # then); the cadence governs from there: 4 -> 12 -> 20.
    assert seen == [4, 12, 20]


def test_a_cadence_larger_than_the_run_still_speaks_once_it_has_evidence():
    seen = []
    gen = generator(reauthor=lambda a, e: seen.append(1),
                    reauthor_every=100, max_reauthorings=4)
    drive(gen, generations=6, want=4)
    assert seen == [1]          # once, on evidence -- and not again, on cadence


def test_the_evidence_floor_is_honoured_and_nothing_fires_beneath_it():
    seen = []
    gen = generator(reauthor=lambda a, e: seen.append(len(gen._rows)),
                    reauthor_every=1, evidence_min_rows=9, max_reauthorings=4)
    drive(gen, generations=6, want=4)
    assert seen and seen[0] >= 9        # never authored from fewer rows
    assert seen[0] == 12                # the first propose that holds nine


def test_a_negative_evidence_floor_is_refused():
    with pytest.raises(ValueError):
        generator(evidence_min_rows=-1)


def test_off_stays_off_however_much_evidence_arrives():
    seen = []
    gen = generator(reauthor=lambda a, e: seen.append(1), reauthor_every=0,
                    max_reauthorings=4)
    drive(gen, generations=6, want=4)
    assert seen == [] and gen.evidence_log == []


def test_evidence_is_what_was_measured_and_attribution_is_what_it_produced():
    """The seam the fix turns on: two clocks, and neither forges the other."""

    gen = generator(reauthor=lambda a, e: None, reauthor_every=4,
                    max_reauthorings=1)
    for i in range(6):                       # measurements the run holds, that
        gen.note_measured({"knob": DOMAINS["knob"][i % 4], "other": "p"},
                          objectives={"cost": float(i)}, survived=(i == 0))
    assert len(gen._rows) == 6               # ... this generator did not make
    assert gen.telemetry.measured == 0       # so it is credited with none
    assert gen.telemetry.survived == 0
    gen.propose(template=TEMPLATE, candidate_model=Candidate, restriction=None,
                archive=[], want=4, rng=random.Random(0), seed=1)
    # The channel spoke off evidence the generator did not produce, and BOTH
    # clocks are in the record: nothing here can be read as attribution.
    assert gen.telemetry.reauthorings == 1
    assert gen.evidence_log[-1]["at_rows"] == 6
    assert gen.evidence_log[-1]["at_measured"] == 0
    gen.record_measured({"knob": "a", "other": "q"}, survived=True,
                        objectives={"cost": 0.0})
    assert len(gen._rows) == 7 and gen.telemetry.measured == 1
    assert gen.telemetry.survived == 1       # its own child, credited once


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


# ------------------------------------------------- occupancy among the elite
#
# Every share below is computed by hand in the comment beside it. The table
# exists because rank correlations over tens of rows are noise; a test that
# checked it by recomputing it in the test would be checking nothing.

def _row(knob, other, cost, survived=True):
    return ({"knob": knob, "other": other}, {"cost": float(cost)}, survived)


def test_elite_occupancy_prints_the_shares_it_computed_by_hand():
    # Front = the three cost-1 rows; the two cost-5 rows are dominated.
    body = [_row("a", "p", 1), _row("a", "q", 1), _row("b", "p", 1),
            _row("c", "p", 5), _row("d", "q", 5), _row("d", "q", 5)]
    text = render_elite_table(body, SPECS, DOMAINS)

    # knob among 3 elite: a 2/3, b 1/3, c 0, d 0.  knob among 6 rows:
    # a 2/6, b 1/6, c 1/6, d 2/6.  "a" and "b" are enriched; "c" and "d" are
    # held by nothing on the front, so neither clause admits them.
    assert ("    knob: a 2/3=0.67 vs 0.33 overall (+0.33); "
            "b 1/3=0.33 vs 0.17 overall (+0.17)") in text
    assert " c " not in text and " d " not in text
    # other among 3 elite: p 2/3, q 1/3.  Among 6 rows: p 3/6, q 3/6.
    # "q" is under-represented and held once, so neither clause admits it.
    assert "    other: p 2/3=0.67 vs 0.50 overall (+0.17)" in text
    assert "q " not in text
    assert ("    (0 of 2 parameters showed no value the front concentrates "
            "on)") in text


def test_a_value_the_front_holds_twice_prints_even_when_it_is_diluted():
    """The OR in the inclusion rule, exercised on the arm that is not enrichment."""

    body = [_row("a", "p", 1), _row("a", "q", 1),          # elite
            _row("b", "p", 1), _row("b", "q", 1),          # elite
            _row("a", "p", 5), _row("a", "q", 5),
            _row("a", "p", 5), _row("a", "q", 5)]
    text = render_elite_table(body, SPECS, DOMAINS)
    # "a" is 2/4 = 0.50 among the elite against 6/8 = 0.75 overall: enrichment
    # of -0.25, which the floor refuses, and a count of 2, which admits it.
    assert ELITE_MIN_COUNT == 2
    assert "a 2/4=0.50 vs 0.75 overall (-0.25)" in text
    assert "b 2/4=0.50 vs 0.25 overall (+0.25)" in text


def test_a_thin_enrichment_held_once_is_below_the_floor_and_is_dropped():
    # "b" sits on the front once: 1/4 = 0.25 elite against 5/25 = 0.20
    # overall. The gap is 0.05 and the rule admits only MORE than that.
    body = ([_row("a", "p", 1), _row("b", "q", 1),
             _row("c", "p", 1), _row("d", "q", 1)]
            + [_row("abcd"[i % 4], "pq"[i % 2], 5) for i in range(16)]
            + [_row("b", "p", 5) for _ in range(5)])
    text = render_elite_table(body, SPECS, DOMAINS)
    assert ELITE_ENRICHMENT_FLOOR == 0.05
    assert "b 1/4" not in text, (
        "a value one front member holds, enriched by less than the floor, "
        f"was printed anyway:\n{text}"
    )


def test_a_field_with_nothing_to_say_is_omitted_and_counted_in_one_line():
    body = ([_row("a", "p", 1), _row("b", "q", 1),
             _row("c", "p", 1), _row("d", "q", 1)]
            + [_row("abcd"[i % 4], "pq"[i % 2], 5) for i in range(16)])
    text = render_elite_table(body, SPECS, DOMAINS)
    # knob: every value is 1/4 elite against 5/20 overall -- zero enrichment,
    # one occurrence each, so the whole field says nothing and is dropped.
    assert "knob" not in text
    # other: p and q are each held twice, so the count clause keeps them.
    assert "    other: p 2/4=0.50 vs 0.50 overall (+0.00); " \
           "q 2/4=0.50 vs 0.50 overall (+0.00)" in text
    assert ("    (1 of 2 parameters showed no value the front concentrates "
            "on)") in text


def test_a_front_larger_than_the_cap_is_truncated_in_measurement_order():
    body = [_row("a", "p", 1), _row("a", "q", 1), _row("b", "p", 1),
            _row("c", "q", 1)]
    text = render_elite_table(body, SPECS, DOMAINS, max_elite=2)
    assert "first 2 of 4 non-dominated configurations, in measurement order" \
        in text
    # Only the first two rows count, and both hold "a": 2/2 against 2/4.
    assert "a 2/2=1.00 vs 0.50 overall (+0.50)" in text
    assert "b " not in text and "c " not in text


def test_a_sequence_field_pools_its_positions_into_one_entry():
    body = [({"genome": [1, 1]}, {"cost": 1.0}, True),
            ({"genome": [1, 1]}, {"cost": 1.0}, True),
            ({"genome": [0, 0]}, {"cost": 5.0}, False),
            ({"genome": [0, 0]}, {"cost": 5.0}, False)]
    domains = {"genome[0]": [0, 1], "genome[1]": [0, 1]}
    text = render_elite_table(body, SPECS, domains)
    assert "genome[0]" not in text and "genome[1]" not in text, (
        "the occupancy table keyed a sequence per POSITION, which is not the "
        f"key the weight table it feeds is written in:\n{text}"
    )
    # Two elite rows x two positions = four in-domain slots, all holding 1;
    # four rows x two positions = eight slots, four of them holding 1.
    assert "    genome: 1 4/4=1.00 vs 0.50 overall (+0.50)" in text
    assert "    (0 of 1 parameters showed no value the front concentrates on)" \
        in text


def test_no_objective_value_reaches_the_occupancy_table():
    body = [_row("a", "p", 1337.5), _row("b", "q", 1337.5),
            _row("c", "p", 4242.25), _row("d", "q", 4242.25)]
    text = render_elite_table(body, SPECS, DOMAINS)
    assert "1337" not in text and "4242" not in text, (
        f"occupancy rendered a score, which is the other section's job:\n{text}"
    )
    assert "cost" not in text


def test_an_empty_trace_has_no_elite_to_render():
    assert "no candidate has been measured" in render_elite_table(
        [], SPECS, DOMAINS)


def test_rendering_the_same_rows_twice_is_the_same_occupancy_bytes():
    body = rows(9)
    assert (render_elite_table(body, SPECS, DOMAINS)
            == render_elite_table(body, SPECS, DOMAINS))


# ---------------------------------------------------------------- the gate

def _front():
    body = rows(8)
    return front_of(body, SPECS)


@pytest.mark.parametrize("reply, reason", [
    ("not json at all", "unparsed"),
    ('{"weight": {"knob": ["a"]}}', "unparsed"),         # list where a map goes
    ('{"weight": {"knob": {"a": "big"}}}', "unparsed"),  # weight not a number
    ('{"weight": {}}', "empty"),
    ('{"weight": {"knob": {"a": 1, "b": 1}}}', "empty"),  # equal = biases nothing
    ('{"weight": {"nosuch": {"a": 2}}}', "undeclared parameter"),
    ('{"weight": {"knob": {"zzz": 2}}}', "undeclared value"),
    ('{"weight": {"knob": {"a": 0}}}', "invalid_weight"),
    ('{"weight": {"knob": {"a": -3}}}', "invalid_weight"),
])
def test_the_gate_refuses_and_says_why(reply, reason):
    verdict = admit_weighted_restriction(parse_weighted_restriction(reply),
                                         domains=DOMAINS)
    assert not verdict.admitted
    assert verdict.reason.startswith(reason)


def test_the_gate_refuses_an_over_concentrated_prior():
    prior = WeightedProposal(weight={"knob": {"a": 16.0}})
    refused = admit_weighted_restriction(prior, domains=DOMAINS,
                                         max_weight_ratio=8.0)
    assert not refused.admitted
    assert refused.reason.startswith("over_concentrated")
    assert refused.concentration == 16.0
    admitted = admit_weighted_restriction(prior, domains=DOMAINS,
                                          max_weight_ratio=16.0)
    assert admitted.admitted


def test_an_admissible_prior_structurally_excludes_nothing():
    # The graded form has no excludes_front refusal to test, because there is
    # nothing it COULD exclude: whatever the weights, every declared value
    # keeps positive mass. This is the structural fix for W1's vacuous-or-veto
    # pathology -- assert it on the applied domains, where it is load-bearing.
    front = _front()
    best = front[0][0]["knob"]
    others = {v: 8.0 for v in DOMAINS["knob"] if v != best}
    verdict = admit_weighted_restriction(
        WeightedProposal(weight={"knob": others}), domains=DOMAINS)
    assert verdict.admitted                     # the hard form had to veto this
    # The admitted prior IS the package's generic graded form, and its
    # support is the full declared domain: the generic `allowed` view -- the
    # one the unwind test and structure record read -- excludes nothing.
    from agent_evolve.policies.weighted_prior import WeightedRestriction
    assert isinstance(verdict.prior, WeightedRestriction)
    assert set(verdict.prior.allowed["knob"]) == set(DOMAINS["knob"])
    biased = apply_weighted_restriction(DOMAINS, verdict.prior)
    for name, declared in DOMAINS.items():
        assert set(biased[name]) == set(declared)   # nothing excluded, ever


def test_the_gate_admits_a_measured_bias_and_reports_its_concentration():
    verdict = admit_weighted_restriction(
        WeightedProposal(weight={"knob": {"a": 4.0, "b": 2.0}}),
        domains=DOMAINS)
    assert verdict.admitted
    assert verdict.concentration == 4.0          # implicit 1.0 on c and d


def test_one_bad_entry_refuses_the_whole_reply_rather_than_being_dropped():
    prior = parse_weighted_restriction(
        '{"weight": {"knob": {"a": 2, "b": 2}, "other": {"nope": 2}}}')
    verdict = admit_weighted_restriction(prior, domains=DOMAINS)
    assert not verdict.admitted
    assert verdict.proposal is not None         # judged, not silently repaired


def test_applying_a_prior_biases_mass_and_keeps_every_value():
    verdict = admit_weighted_restriction(
        WeightedProposal(weight={"knob": {"a": 3.0, "b": 2.0}}),
        domains=DOMAINS)
    assert verdict.admitted
    biased = apply_weighted_restriction(DOMAINS, verdict.prior)
    assert biased["knob"].count("a") == 3
    assert biased["knob"].count("b") == 2
    assert biased["knob"].count("c") == 1 and biased["knob"].count("d") == 1
    assert biased["other"] == ["p", "q"]        # unweighted loci untouched


# ------------------------------------------------- the prior inside the loop

def test_an_admitted_prior_moves_where_the_generator_SAMPLES():
    gen = generator(reauthor_every=4, max_priors=1,
                    prior_author=lambda p:
                    '{"weight": {"knob": {"a": 8, "b": 8}}}')
    seed_front(gen)
    gen.propose(template=TEMPLATE, candidate_model=Candidate, restriction=None,
                archive=[], want=16, rng=random.Random(0), seed=1)
    assert gen.telemetry.priors_admitted == 1
    # What the GENERATOR emitted, not the pool it was topped up into: a short
    # pool is filled with schema-uniform draws over the DECLARED domains,
    # which is the credential-free fallback and not the generator's output.
    # The bias is GRADED: mass concentrates on the weighted values but the
    # unweighted ones stay reachable, so assert the shift, not an exclusion.
    assert gen.last_report is not None and gen.last_report.accepted
    knobs = [c["knob"] for c in gen.last_report.accepted]
    weighted = sum(1 for k in knobs if k in ("a", "b"))
    assert weighted > len(knobs) - weighted     # mass moved where the weights say


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
        prior_author=lambda p: '{"weight": {"knob": {"a": 4}}}')
    seed_front(gen)
    drive(gen, generations=4, want=4,
          cost=lambda c: float(DOMAINS["knob"].index(c["knob"])))
    assert gen.telemetry.priors_admitted == 1
    assert gen.telemetry.rejected_out_of_domain == 0


def test_a_prior_that_stops_paying_is_unwound():
    gen = generator(reauthor_every=4, max_priors=1, prior_unwind_batches=1,
                    prior_author=lambda p:
                    '{"weight": {"knob": {"a": 4, "b": 4}}}')
    seed_front(gen)
    # Nothing measured after the prior lands ever survives, so the bet stops
    # paying and the generator must finish on the declared domains.
    drive(gen, generations=4, want=4, cost=lambda c: 99.0)
    assert gen.telemetry.priors_admitted == 1
    assert gen.telemetry.priors_unwound == 1


def test_a_refused_prior_is_counted_and_never_installed():
    gen = generator(reauthor_every=4, max_priors=1,
                    prior_author=lambda p: '{"weight": {"knob": {"nope": 3}}}')
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


# ------------------------------ W11: the loop's own measurements are evidence

class _CountingProblem:
    """A two-locus problem whose evaluator is a pure, cheap lookup."""

    candidate_model = Candidate
    objectives = tuple(SPECS)

    def __init__(self) -> None:
        self.calls = 0

    def seeds(self):
        return (dict(TEMPLATE),)

    def validate(self, config):
        from agent_evolve.core.problem import ValidationOutcome
        for name, values in DOMAINS.items():
            if config.get(name) not in values:
                return ValidationOutcome(False, "structural", f"bad {name}")
        return ValidationOutcome(True)

    def materialize(self, config):
        return (config["knob"], config["other"])

    def evaluate(self, artifact):
        self.calls += 1
        # knob drives the cost monotonically in its declared order, so the
        # evidence carries a determinable effect for the model to name.
        return {"cost": float(DOMAINS["knob"].index(artifact[0]))}


def _run_loop(gen, *, population_size=8, offspring=4, budget=20, seed=3):
    from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop

    problem = _CountingProblem()
    run_genetic_loop(problem=problem, config=GeneticConfig(
        seed=seed, population_size=population_size,
        offspring_per_generation=offspring, generations=6,
        evaluation_budget=budget, generator=gen))
    return problem


def test_the_initial_population_is_visible_to_the_first_prior():
    """THE W11 PIN. The run's first charges are evidence, and arrive as such.

    Before the fix the generator's evidence view held only what the generator
    itself produced, so the initial population -- every charge the run had
    made when the first pool is drawn -- was invisible, and no prior could be
    authored until two generations of offspring had been measured. On the EDA
    venue that put the prior at a median charge of 40.0 against a 43.5-charge
    target, with the target already hit before the prior existed on 19 of 48
    seeds. The prior must now be authorable off generation 0's measurements.
    """

    prompts = []
    gen = generator(reauthor_every=10, max_priors=1,
                    prior_author=lambda p: prompts.append(p) or
                    '{"weight": {"knob": {"a": 4, "b": 2}}}')
    _run_loop(gen, population_size=8, offspring=4, budget=20)

    priors = [r for r in gen.evidence_log if r["kind"] == "locus_prior"]
    assert len(priors) == 1 and priors[0]["accepted"] is True
    first = priors[0]
    # Authored off the initial population and nothing else: eight rows, none
    # of them this generator's own child.
    assert first["at_rows"] == 8 == first["rows_shown"]
    assert first["at_measured"] == 0
    assert gen.telemetry.priors_admitted == 1
    # ... and the evidence really carried the initial population's numbers.
    assert "measured so far: 8 configurations" in prompts[0]
    assert "which parameter moves which cost" in prompts[0]


def test_a_cadence_that_counts_only_children_is_what_made_the_prior_late():
    """The same loop under the pre-W11 rule, so the fix's size is measured.

    `evidence_min_rows == reauthor_every` restores the old start gate exactly.
    The prior then waits for the cadence to be met in rows the generator
    itself produced, which on this loop takes until offspring have been
    measured -- strictly later, off strictly more charges.
    """

    def build(**kwargs):
        seen = []
        gen = generator(reauthor_every=10, max_priors=1,
                        prior_author=lambda p: seen.append(p) or
                        '{"weight": {"knob": {"a": 4, "b": 2}}}', **kwargs)
        _run_loop(gen, population_size=8, offspring=4, budget=20)
        return [r for r in gen.evidence_log if r["kind"] == "locus_prior"]

    early = build()
    late = build(evidence_min_rows=10)
    assert early and late
    assert early[0]["at_rows"] < late[0]["at_rows"]
    assert early[0]["at_rows"] == 8 and late[0]["at_rows"] == 12


def test_the_generator_is_still_credited_only_with_its_own_children():
    """Evidence widened; attribution did not. The unwind rule still works."""

    gen = generator(reauthor_every=10, max_priors=1, prior_unwind_batches=1,
                    prior_author=lambda p:
                    '{"weight": {"knob": {"c": 4, "d": 4}}}')
    _run_loop(gen, population_size=8, offspring=4, budget=20)
    # The eight initial-population rows are evidence, never survival credit.
    assert gen.telemetry.measured == len(gen._rows) - 8
    assert gen.telemetry.measured > 0
    assert gen.telemetry.priors_admitted == 1


def test_a_prior_authored_early_still_excludes_nothing():
    """Every safety property survives the earlier tick, checked on the loop."""

    gen = generator(reauthor_every=10, max_priors=1,
                    prior_author=lambda p:
                    '{"weight": {"knob": {"a": 8, "b": 8}}}')
    problem = _run_loop(gen, population_size=8, offspring=4, budget=20)
    assert gen.telemetry.priors_admitted == 1
    assert gen.telemetry.priors_refused == 0
    # Support is the DECLARED domain: nothing the schema allows was dropped.
    prior = gen._prior
    if prior is not None:
        values, weights = dict(prior.weighted)["knob"]
        assert list(values) == DOMAINS["knob"]
        assert all(w > 0.0 for w in weights)
    assert problem.calls <= 20


@pytest.mark.parametrize("reply,counter", [
    ("not json at all", "priors_refused"),
    ('{"weight": {"nope": {"a": 3}}}', "priors_refused"),
    ('{"weight": {"knob": {"a": 400}}}', "priors_refused"),
    ('{"weight": {"knob": {"a": -1}}}', "priors_refused"),
    ('{"weight": {}}', "priors_refused"),
])
def test_the_gate_still_refuses_garbage_at_the_earlier_tick(reply, counter):
    gen = generator(reauthor_every=10, max_priors=1,
                    prior_author=lambda p: reply)
    _run_loop(gen, population_size=8, offspring=4, budget=20)
    assert getattr(gen.telemetry, counter) == 1
    assert gen.telemetry.priors_admitted == 0
