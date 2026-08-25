"""The screening phase: designs that can identify structure, and priors from it."""

from __future__ import annotations

import random
from typing import Literal

import pytest
from pydantic import BaseModel

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.structure import (
    attribute,
    crossed_screen,
    render_attribution,
    restriction_is_paying,
    statistical_prior,
)

SPECS = [ObjectiveSpec("energy", "min"), ObjectiveSpec("speed", "max")]


class Arch(BaseModel):
    width: Literal[8, 16]
    regs: bool
    depth: Literal[256, 512, 1024]
    mesh: Literal[4, 8, 16, 32]


TEMPLATE = {"width": 16, "regs": False, "depth": 512, "mesh": 8}


def _truth(cfg):
    """An AND-gate landscape: only width=8 AND regs=True is any good, and mesh
    alone drives speed -- the shape that defeated a marginally-balanced screen."""
    good = cfg["width"] == 8 and cfg["regs"]
    return {"energy": (1.0 if good else 4.0) + cfg["depth"] / 4096.0,
            "speed": float(cfg["mesh"])}


def test_crossed_screen_fully_crosses_the_blocking_loci():
    screen = crossed_screen(TEMPLATE, Arch, size=4, rng=random.Random(1))
    assert len(screen) == 4
    # the two binary loci have the smallest domains, so they are the block
    cells = {(c["width"], c["regs"]) for c in screen}
    assert cells == {(8, True), (8, False), (16, True), (16, False)}


def test_crossed_screen_matches_free_loci_across_blocks():
    # every block sees the SAME setting of the non-blocking loci: that is what
    # keeps mesh from leaking into the width/regs contrast
    screen = crossed_screen(TEMPLATE, Arch, size=4, rng=random.Random(2))
    assert len({(c["depth"], c["mesh"]) for c in screen}) == 1


# --- the free ladder is space-filling, not the main diagonal ----------------
#
# Until 2026-08-25 every free locus read the same ladder index, so a screen with
# nothing crossable WAS the space's main diagonal. These tests pin the repair
# and keep the defect's own statistic in the file as the control.

_LEVELS = tuple(range(20))


def _analog_shaped_model(n_loci: int = 24, levels: tuple = _LEVELS):
    """24 loci x 20 declared levels, screened at 16 rows: the analog venue's shape.

    Nothing is crossable here -- the smallest domain already outruns the budget
    -- so the whole design is the free ladder, which is exactly the case that
    produced the diagonal.
    """
    from pydantic import create_model
    model = create_model(
        "Analog",
        **{f"f{i}": (Literal[levels], ...) for i in range(n_loci)})
    return model, {f"f{i}": levels[0] for i in range(n_loci)}


def _level_indices(screen, model, template):
    """Each row as a vector of LEVEL INDICES, one column per locus."""
    from agent_evolve.policies.genetic import loci_of, locus_domain, read_locus
    loci = list(loci_of(template))
    domains = {lc: locus_domain(model, lc) for lc in loci}
    return [[domains[lc].index(read_locus(row, lc)) for lc in loci] for row in screen]


def _spearman(a, b):
    def ranked(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        rank = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            share = (i + j) / 2.0
            for k in range(i, j + 1):
                rank[order[k]] = share
            i = j + 1
        return rank
    ra, rb = ranked(a), ranked(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return 0.0 if da == 0 or db == 0 else num / (da * db)


def _mean_abs_pairwise_rho(rows):
    columns = list(zip(*rows))
    pairs = [abs(_spearman(columns[i], columns[j]))
             for i in range(len(columns)) for j in range(i + 1, len(columns))]
    return sum(pairs) / len(pairs)


def test_the_free_ladder_is_decorrelated_and_the_diagonal_is_the_control():
    model, template = _analog_shaped_model()
    rows = _level_indices(
        crossed_screen(template, model, size=16, rng=random.Random(11)),
        model, template)
    assert len(rows) == 16 and len(rows[0]) == 24

    # The control: what this function built before the fix -- every locus on one
    # shared index. Its statistic is exactly 1.0, and its row 0 is the space's
    # minimum corner.
    diagonal = [[t % len(_LEVELS)] * 24 for t in range(16)]
    assert _mean_abs_pairwise_rho(diagonal) == pytest.approx(1.0)
    assert diagonal[0] == [0] * 24

    assert _mean_abs_pairwise_rho(rows) < 0.3
    # and no row is a single level held across every locus
    assert not any(len(set(row)) == 1 for row in rows)


def test_the_screen_reaches_levels_the_diagonal_could_never_show():
    # The measured consequence of the diagonal: 16 rows over a 20-level ladder
    # never showed a locus above level 15, which is the box every analog prior
    # was fit inside. A space-filling design reaches the top of the ladder.
    model, template = _analog_shaped_model()
    rows = _level_indices(
        crossed_screen(template, model, size=16, rng=random.Random(12)),
        model, template)
    assert max(max(row) for row in rows) > 15
    assert max(t % len(_LEVELS) for t in range(16)) == 15, "the diagonal's ceiling"


def _flat_model(n_loci: int = 6, levels: tuple = (0, 1, 2, 3)):
    from pydantic import create_model
    model = create_model(
        "Flat", **{f"g{i}": (Literal[levels], ...) for i in range(n_loci)})
    return model, {f"g{i}": levels[0] for i in range(n_loci)}


def test_every_locus_still_sweeps_its_whole_ladder():
    # Decorrelation must not cost coverage: each locus's column is whole
    # permutations, so every level appears floor(rows / levels) times at least.
    model, template = _flat_model()
    screen = crossed_screen(template, model, size=16, blocking=(),
                            rng=random.Random(13))
    assert len(screen) == 16
    for name in template:
        counts = {level: 0 for level in (0, 1, 2, 3)}
        for row in screen:
            counts[row[name]] += 1
        assert set(counts) == {0, 1, 2, 3}
        assert min(counts.values()) >= 16 // 4


def test_attribution_keeps_its_per_level_n_after_the_fix():
    model, template = _flat_model()
    screen = crossed_screen(template, model, size=16, blocking=(),
                            rng=random.Random(14))
    attr = attribute([(c, {"energy": float(sum(c.values())), "speed": 1.0})
                      for c in screen], SPECS, model)
    for name in template:
        summaries = attr.for_locus(name)
        assert len(summaries) == 4
        assert min(s.n for s in summaries) >= 16 // 4


def test_the_design_is_exact_given_the_seed():
    model, template = _analog_shaped_model()
    a = crossed_screen(template, model, size=16, rng=random.Random(15))
    b = crossed_screen(template, model, size=16, rng=random.Random(15))
    c = crossed_screen(template, model, size=16, rng=random.Random(16))
    assert a == b
    assert a != c, "a different seed must give a different design"


def test_crossed_screen_respects_the_budget_exactly():
    for size in (1, 3, 4, 7, 12):
        got = crossed_screen(TEMPLATE, Arch, size=size, rng=random.Random(size))
        assert len(got) == size


def test_screen_plus_rule_finds_the_and_gate():
    screen = crossed_screen(TEMPLATE, Arch, size=4, rng=random.Random(3))
    attr = attribute([(c, _truth(c)) for c in screen], SPECS, Arch)
    prior = statistical_prior(attr, Arch)
    assert prior.allowed["width"] == [8]
    assert prior.allowed["regs"] == [True]


def test_the_rule_abstains_when_the_screen_separates_nothing():
    flat = [dict(TEMPLATE) for _ in range(4)]
    attr = attribute([(c, {"energy": 1.0, "speed": 1.0}) for c in flat], SPECS, Arch)
    assert statistical_prior(attr, Arch).allowed == {}


def test_attribution_normalises_maximised_objectives():
    # speed is maximised; a screen that only differs in speed must credit the
    # HIGHER value as non-dominated, not the lower one
    cfgs = [dict(TEMPLATE, mesh=m) for m in (4, 32)]
    attr = attribute([(c, {"energy": 1.0, "speed": float(c["mesh"])}) for c in cfgs],
                     SPECS, Arch)
    by_value = {s.value: s.nondominated for s in attr.for_locus("mesh")}
    assert by_value[32] == 1 and by_value[4] == 0


def test_render_is_readable_and_carries_counts():
    screen = crossed_screen(TEMPLATE, Arch, size=4, rng=random.Random(4))
    text = render_attribution(attribute([(c, _truth(c)) for c in screen], SPECS, Arch))
    assert "width" in text and "non-dominated" in text and "n=" in text


def test_unwind_test_holds_a_paying_bet_and_drops_a_losing_one():
    from agent_evolve.policies.genetic import DomainRestriction
    r = DomainRestriction({"width": [8]})
    assert restriction_is_paying(r, screen_best=2.0, current_best=1.5)
    assert not restriction_is_paying(r, screen_best=2.0, current_best=3.0)
    # an empty restriction is never unwound: there is no bet to lose
    assert restriction_is_paying(DomainRestriction({}), 2.0, 99.0)


def test_screen_ignores_loci_the_schema_leaves_undeclared():
    class Loose(BaseModel):
        width: Literal[8, 16]
        note: str
    screen = crossed_screen({"width": 16, "note": "x"}, Loose, size=2,
                            rng=random.Random(5))
    assert {c["width"] for c in screen} == {8, 16}
    assert all(c["note"] == "x" for c in screen)


# --- the phase inside the loop ---------------------------------------------

def _and_gate_problem():
    """A problem whose good region is a conjunction of two loci, so a prior
    that finds it is worth a lot and one that misses it is worth nothing."""
    from agent_evolve.core.problem import ValidationOutcome

    class P:
        candidate_model = Arch
        objectives = (ObjectiveSpec("energy", "min"), ObjectiveSpec("speed", "max"))

        def __init__(self):
            self.seen = []

        def seeds(self):
            return [dict(TEMPLATE)]

        def validate(self, config):
            return ValidationOutcome(ok=True)

        def materialize(self, config):
            return dict(config)

        def evaluate(self, artifact):
            self.seen.append(dict(artifact))
            return _truth(artifact)
    return P()


def test_structure_phase_off_by_default_and_charges_nothing():
    from agent_evolve.contract import as_problem
    from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop
    p = _and_gate_problem()
    res = run_genetic_loop(problem=as_problem(p),
                           config=GeneticConfig(population_size=4,
                                                offspring_per_generation=2,
                                                generations=20,
                                                evaluation_budget=12, seed=1))
    assert res.evaluations <= 12
    assert not any("structure" in h for h in res.history)


def test_the_screen_fix_leaves_the_unscreened_fossil_byte_identical():
    """A run that buys no screen must not move one byte when the screen changes.

    The digest below was taken from the pre-fix build (2026-08-25) over the
    full sequence of configurations the problem was asked to evaluate. The
    screen fix is default-ON, so this is the guard that says where its blast
    radius stops: ``structure_budget=0`` never calls :func:`crossed_screen`,
    and every fossil row recorded that way stays exactly quotable.
    """
    import hashlib
    import json

    from agent_evolve.contract import as_problem
    from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop
    p = _and_gate_problem()
    run_genetic_loop(
        problem=as_problem(p),
        config=GeneticConfig(population_size=4, offspring_per_generation=3,
                             generations=20, evaluation_budget=32,
                             structure_budget=0, seed=20260825))
    blob = json.dumps(p.seen, sort_keys=True, default=str).encode()
    assert len(p.seen) == 30
    assert hashlib.sha256(blob).hexdigest() == (
        "ba38d56c8502ce82b649db2de18333813efadf4121165022351e9a067042fbac")


def test_structure_phase_spends_its_budget_and_installs_a_prior():
    from agent_evolve.contract import as_problem
    from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop
    p = _and_gate_problem()
    res = run_genetic_loop(
        problem=as_problem(p),
        config=GeneticConfig(population_size=4, offspring_per_generation=2,
                             generations=20, evaluation_budget=16,
                             structure_budget=4, seed=2))
    record = next(h["structure"] for h in res.history if "structure" in h)
    assert record["evaluated"] == 4
    # the screen sees the AND-gate and the shipped rule narrows both loci
    assert record["allowed"].get("width") == [8]
    assert record["allowed"].get("regs") == [True]
    # the screen is charged, not free: its 4 evaluations are inside the budget
    assert 4 <= res.evaluations <= 16


def test_restricted_search_stays_inside_the_prior():
    from agent_evolve.contract import as_problem
    from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop
    p = _and_gate_problem()
    run_genetic_loop(
        problem=as_problem(p),
        config=GeneticConfig(population_size=4, offspring_per_generation=3,
                             generations=20, evaluation_budget=20,
                             structure_budget=4, seed=3))
    after_screen = p.seen[4:]
    assert after_screen, "the run must continue past the screen"
    # The prior governs SAMPLING, not the caller's own input: the seed survives
    # untouched, and everything the loop draws respects the restriction.
    drawn = [c for c in after_screen if c != TEMPLATE]
    assert drawn, "the loop must draw something of its own"
    assert all(c["width"] == 8 and c["regs"] is True for c in drawn)
    assert TEMPLATE in after_screen, "a prior must not delete the caller's seed"


def test_a_wrong_prior_is_unwound_rather_than_held():
    # A proposer that excludes the good region is the failure mode that cannot
    # recover on its own; the loop must notice and drop the bet.
    from agent_evolve.contract import as_problem
    from agent_evolve.policies.genetic import DomainRestriction
    from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop
    p = _and_gate_problem()
    res = run_genetic_loop(
        problem=as_problem(p),
        config=GeneticConfig(population_size=4, offspring_per_generation=3,
                             generations=20, evaluation_budget=24,
                             structure_budget=4, seed=4,
                             prior_proposer=lambda attr, model: DomainRestriction(
                                 {"width": [16], "regs": [False]})))
    record = next(h["structure"] for h in res.history if "structure" in h)
    assert "unwound" in record
    assert any(c["width"] == 8 and c["regs"] is True for c in p.seen[5:])


def test_a_raising_proposer_does_not_kill_the_run():
    from agent_evolve.contract import as_problem
    from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop
    def boom(attr, model):
        raise RuntimeError("proposer exploded")
    p = _and_gate_problem()
    res = run_genetic_loop(
        problem=as_problem(p),
        config=GeneticConfig(population_size=4, offspring_per_generation=2,
                             generations=20, evaluation_budget=16,
                             structure_budget=4, seed=5, prior_proposer=boom))
    record = next(h["structure"] for h in res.history if "structure" in h)
    assert "proposer_error" in record and res.evaluations >= 4


def test_a_weighted_prior_runs_the_phase_and_the_seed_survives():
    # The graded rule keeps every level with unequal mass, so its support is
    # the whole domain: nothing to unwind, only a bias -- and the loop must
    # carry it end to end without touching the caller's seed.
    from agent_evolve.contract import as_problem
    from agent_evolve.policies.weighted_prior import statistical_weighted_prior
    from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop
    p = _and_gate_problem()
    res = run_genetic_loop(
        problem=as_problem(p),
        config=GeneticConfig(population_size=4, offspring_per_generation=3,
                             generations=20, evaluation_budget=20,
                             structure_budget=4, seed=6,
                             prior_proposer=statistical_weighted_prior))
    record = next(h["structure"] for h in res.history if "structure" in h)
    assert set(record["allowed"].get("width", ())) == {8, 16}
    assert dict(TEMPLATE) in p.seen, "a prior must not delete the caller's seed"
    assert res.evaluations <= 20


class _Seq(BaseModel):
    genome: list[Literal["a", "b", "c"]]


SEQ_TEMPLATE = {"genome": ["a", "a", "a", "a", "a", "a"]}


def test_pooled_screen_makes_pure_then_spiked_candidates():
    screen = crossed_screen(SEQ_TEMPLATE, _Seq, size=5, pool_by_field=True)
    assert len(screen) == 5
    # one pure candidate per vocabulary value first
    assert screen[0]["genome"] == ["a"] * 6
    assert screen[1]["genome"] == ["b"] * 6
    assert screen[2]["genome"] == ["c"] * 6
    # then spiked: the value holds every other position
    assert screen[3]["genome"][0::2] == ["a", "a", "a"]
    assert len(set(screen[3]["genome"])) > 1, "a spike is not a pure repeat"


def test_the_pooled_design_is_untouched_by_the_ladder_fix():
    # The pooled variant returns before the free ladder is ever built: its
    # pure-then-spiked sequence is a property of the vocabulary, not of a seed.
    a = crossed_screen(SEQ_TEMPLATE, _Seq, size=7, pool_by_field=True,
                       rng=random.Random(21))
    b = crossed_screen(SEQ_TEMPLATE, _Seq, size=7, pool_by_field=True,
                       rng=random.Random(22))
    c = crossed_screen(SEQ_TEMPLATE, _Seq, size=7, pool_by_field=True)
    assert a == b == c
    assert a[0]["genome"] == ["a"] * 6


def test_pooled_attribution_counts_every_position_as_an_observation():
    screen = crossed_screen(SEQ_TEMPLATE, _Seq, size=3, pool_by_field=True)
    # value 'a' is good, everything else is bad
    rows = [(c, {"energy": 1.0 if set(c["genome"]) == {"a"} else 4.0,
                 "speed": 1.0}) for c in screen]
    attr = attribute(rows, SPECS, _Seq, pool_by_field=True)
    by_value = {s.value: s for s in attr.for_locus("genome")}
    assert by_value["a"].n == 6, "6 positions of the pure-a candidate"
    assert by_value["a"].nondominated == 6
    assert by_value["b"].nondominated == 0
    prior = statistical_prior(attr, _Seq)
    assert prior.allowed == {"genome": ["a"]}


def test_bare_sequence_locus_resolves_the_shared_vocabulary():
    from agent_evolve.policies.genetic import Locus, locus_domain
    assert locus_domain(_Seq, Locus("genome")) == ("a", "b", "c")


def test_pooled_phase_runs_inside_the_loop_and_narrows_the_field():
    from agent_evolve.contract import as_problem
    from agent_evolve.core.problem import ValidationOutcome
    from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop

    class P:
        candidate_model = _Seq
        objectives = (ObjectiveSpec("energy", "min"), ObjectiveSpec("speed", "max"))

        def __init__(self):
            self.seen = []

        def seeds(self):
            return [dict(SEQ_TEMPLATE)]

        def validate(self, config):
            return ValidationOutcome(ok=True)

        def materialize(self, config):
            return dict(config)

        def evaluate(self, artifact):
            self.seen.append(dict(artifact))
            good = artifact["genome"].count("a")
            return {"energy": 10.0 - good, "speed": 1.0}

    p = P()
    res = run_genetic_loop(
        problem=as_problem(p),
        config=GeneticConfig(population_size=4, offspring_per_generation=2,
                             generations=20, evaluation_budget=16,
                             structure_budget=3, structure_pooled=True,
                             seed=8))
    record = next(h["structure"] for h in res.history if "structure" in h)
    assert record["evaluated"] == 3
    assert record["allowed"].get("genome") == ["a"]
    assert res.evaluations <= 16


def test_a_wrong_weighted_prior_is_unwound():
    # Zero weights are exclusions, so a graded prior that zeroes the good
    # region makes exactly the refutable claim the unwind test checks.
    from agent_evolve.contract import as_problem
    from agent_evolve.policies.weighted_prior import WeightedRestriction
    from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop
    p = _and_gate_problem()
    res = run_genetic_loop(
        problem=as_problem(p),
        config=GeneticConfig(population_size=4, offspring_per_generation=3,
                             generations=20, evaluation_budget=24,
                             structure_budget=4, seed=7,
                             prior_proposer=lambda attr, model: WeightedRestriction({
                                 "width": ((8, 16), (0.0, 1.0)),
                                 "regs": ((False, True), (1.0, 0.0))})))
    record = next(h["structure"] for h in res.history if "structure" in h)
    assert "unwound" in record
    assert any(c["width"] == 8 and c["regs"] is True for c in p.seen[5:])
