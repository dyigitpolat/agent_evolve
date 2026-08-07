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
