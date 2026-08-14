"""Ordering on the objectives a surrogate CAN predict, and nothing else.

The defect pinned here was measured on a live co-design venue: an authored
surrogate whose cross-validated rank fidelity read 0.855 on area and 0.606 on
latency was refused outright because it read 0.329 on energy, and the gate
required rho >= 0.5 on EVERY objective. Two objectives it ordered well went
unused because of a third it could not.

What this file pins, in the order it matters:

1. **The verdict has a SCOPE.** ``passing_objectives`` names what the gate
   certified; a partial pass certifies those and nothing else.
2. **The screen orders on that scope and treats the rest as UNKNOWN** -- it
   does not read the excluded objectives at all, so an artifact that emits
   nonsense on one it was not certified for cannot influence the order.
3. **The cost of that is real and is pinned, not assumed.** Domination over a
   subset is a STRICTER relation than domination over the whole problem, so a
   candidate that is excellent only on an excluded objective SINKS. The test
   that demonstrates it is here on purpose: if someone later decides the
   exploration floor can be flat again, this is the reason it cannot.
4. **Degenerate cases are refused**, and the conjunction is untouched.
5. **Telemetry is a correctness requirement**: a run that screened on one
   objective must not be readable as one that screened on three.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.core.telemetry import harvest_telemetry
from agent_evolve.policies.surrogate import (
    ORDERING_GATE,
    PARTIAL_ORDERING_GATE,
    PREDICTION_GATE,
    GatePolicy,
    additive_surrogate,
    validate_surrogate,
)
from agent_evolve.session.screening import (
    Screening,
    ScreenReport,
    screen_offspring,
)

#: Three objectives: two the shape of the problem determines exactly, and one
#: that is a deterministic function of the genome no first-order model can
#: reach. That is the measured situation in miniature.
SPECS = [
    ObjectiveSpec("ones", "max"),
    ObjectiveSpec("zeros", "min"),
    ObjectiveSpec("scramble", "min"),
]
NAMES = tuple(spec.name for spec in SPECS)


def _scramble(genome: Sequence[int]) -> float:
    """Deterministic, and uncorrelated with anything additive."""

    bits = "".join(str(int(b)) for b in genome)
    return float((hash(("k7", bits)) % 9973) / 9973.0)


def _truth(genome: Sequence[int]) -> Dict[str, float]:
    ones = float(sum(genome))
    return {"ones": ones, "zeros": float(len(genome)) - ones,
            "scramble": _scramble(genome)}


def _rows(n: int, width: int = 6, seed: int = 0):
    import random
    rng = random.Random(seed)
    out = []
    seen = set()
    while len(out) < n:
        genome = tuple(rng.randint(0, 1) for _ in range(width))
        if genome in seen:
            continue
        seen.add(genome)
        out.append(({"genome": list(genome)}, _truth(genome)))
    return out


DATA = _rows(24, width=6, seed=1)


def _builder(exact: Sequence[str]):
    """A builder that predicts *exact* truthfully and everything else flat.

    A flat prediction carries no ordering at all -- ``_spearman`` reports 0.0
    for a fully tied side -- so it is the cleanest possible stand-in for "the
    surrogate cannot order this objective", with no accidental correlation.
    """

    def build(evaluated, specs):
        def predict(pool):
            out = []
            for candidate in pool:
                truth = _truth(candidate["genome"])
                out.append({spec.name: (truth[spec.name]
                                        if spec.name in exact else 1.0)
                            for spec in specs})
            return out
        return predict
    return build


# --- the policy is typed, and clamps rather than surprises -------------------

def test_the_partial_policy_is_the_ordering_policy_plus_one_field():
    """One factor apart, so any measurement of it isolates one cause."""

    assert PARTIAL_ORDERING_GATE.min_passing_objectives == 2
    assert ORDERING_GATE.min_passing_objectives == 0
    for field in ("purpose", "scheme", "min_rank_correlation",
                  "min_effective_holdout", "min_rows", "folds",
                  "holdout_fraction"):
        assert (getattr(PARTIAL_ORDERING_GATE, field)
                == getattr(ORDERING_GATE, field)), field
    assert PARTIAL_ORDERING_GATE.admits_partial
    assert not ORDERING_GATE.admits_partial


def test_how_many_objectives_are_required_is_clamped_at_both_ends():
    assert ORDERING_GATE.objectives_required(3) == 3, "0 means ALL"
    assert PARTIAL_ORDERING_GATE.objectives_required(3) == 2
    assert PARTIAL_ORDERING_GATE.objectives_required(2) == 2, (
        "on a two-objective problem 'at least two' IS the conjunction, and "
        "the change must not touch such a problem at all"
    )
    assert PARTIAL_ORDERING_GATE.objectives_required(1) == 1, (
        "a policy written for three objectives must not deadlock a "
        "single-objective problem"
    )
    greedy = ORDERING_GATE.replace(min_passing_objectives=99)
    assert greedy.objectives_required(3) == 3
    assert GatePolicy(min_passing_objectives=1).objectives_required(5) == 1
    with pytest.raises(ValueError):
        GatePolicy(min_passing_objectives=-1)


# --- the defect itself -------------------------------------------------------

def test_one_unpredictable_objective_no_longer_vetoes_the_two_it_can_order():
    """The measured tl_v2 situation, reproduced in a unit test."""

    builder = _builder(["ones", "zeros"])
    conjunction = validate_surrogate(builder, DATA, SPECS, seed=4,
                                     policy=ORDERING_GATE)
    assert not conjunction.passed and conjunction.reason == "rank", (
        "the shipped gate must still reject: this is the before-picture"
    )
    assert conjunction.passing_objectives == (), (
        "a failed verdict certifies nothing"
    )

    partial = validate_surrogate(builder, DATA, SPECS, seed=4,
                                 policy=PARTIAL_ORDERING_GATE)
    assert partial.passed
    assert partial.passing_objectives == ("ones", "zeros")
    assert partial.partial is True
    assert partial.declared_objectives == NAMES
    assert partial.per_objective_spearman["scramble"] < 0.5, (
        "the excluded objective must still be MEASURED and reported"
    )


def test_a_surrogate_that_orders_only_one_of_three_is_still_refused():
    """Ordering by domination over a single objective is a different
    mechanism -- a total order on that objective -- not a weaker version of
    this one. The shipped partial policy refuses it."""

    builder = _builder(["ones"])
    verdict = validate_surrogate(builder, DATA, SPECS, seed=4,
                                 policy=PARTIAL_ORDERING_GATE)
    assert not verdict.passed and verdict.reason == "rank"
    assert "1 of 3 objectives reach it, 2 needed" in verdict.detail, (
        "the shortfall must be legible without re-deriving it"
    )

    lenient = ORDERING_GATE.replace(min_passing_objectives=1)
    admitted = validate_surrogate(builder, DATA, SPECS, seed=4, policy=lenient)
    assert admitted.passed and admitted.passing_objectives == ("ones",), (
        "a caller that declares one is enough gets one -- and the verdict "
        "says loudly that one is all it got"
    )


def test_a_verdict_that_certifies_nothing_can_never_pass():
    """The floor under every partial policy: at least one objective."""

    blind = _builder([])
    for policy in (ORDERING_GATE.replace(min_passing_objectives=1),
                   PARTIAL_ORDERING_GATE, ORDERING_GATE):
        verdict = validate_surrogate(blind, DATA, SPECS, seed=4, policy=policy)
        assert not verdict.passed, policy
        assert verdict.passing_objectives == ()


def test_an_anti_ranking_surrogate_is_rejected_under_the_partial_policy_too():
    """The one term no policy can switch off, re-pinned on the new one."""

    def inverted(evaluated, specs):
        def predict(pool):
            return [{spec.name: -_truth(c["genome"])[spec.name]
                     for spec in specs} for c in pool]
        return predict

    for policy in (ORDERING_GATE, PARTIAL_ORDERING_GATE,
                   ORDERING_GATE.replace(min_passing_objectives=1),
                   PREDICTION_GATE):
        verdict = validate_surrogate(inverted, DATA, SPECS, seed=2,
                                     policy=policy)
        assert not verdict.passed, policy
        assert verdict.passing_objectives == ()


def test_the_conjunction_is_untouched_and_says_what_it_certified():
    """The default policy's decisions must not move, and a full pass must
    report the full scope so no consumer needs to branch on the policy."""

    honest = _builder(list(NAMES))
    for policy in (ORDERING_GATE, PREDICTION_GATE, PARTIAL_ORDERING_GATE):
        verdict = validate_surrogate(honest, DATA, SPECS, seed=7, policy=policy)
        assert verdict.passed, policy
        assert verdict.passing_objectives == NAMES
        assert verdict.partial is False

    # And the pass/fail decision under the conjunction is exactly the two
    # `all(...)` conditions it always was, recomputed here from the verdict's
    # own reported numbers rather than trusted.
    for builder in (_builder(["ones", "zeros"]), _builder(["ones"]),
                    _builder([]), additive_surrogate, _builder(list(NAMES))):
        for policy in (ORDERING_GATE, PREDICTION_GATE):
            verdict = validate_surrogate(builder, DATA, SPECS, seed=11,
                                         policy=policy)
            if not verdict.per_objective_spearman:
                continue
            rank_ok = all(v >= policy.min_rank_correlation
                          for v in verdict.per_objective_spearman.values())
            mse_ok = all(verdict.per_objective_mse[n]
                         < verdict.baseline_mse[n] for n in NAMES)
            assert verdict.passed == (
                rank_ok and (mse_ok or not policy.error_rejects))


# --- the screen orders on the scope, and on nothing else ---------------------

def _cfg(i: int) -> Dict[str, Any]:
    return {"genome": [i]}


def test_the_screen_does_not_read_the_objectives_it_was_not_certified_for():
    """Unknown, not assumed. Two predictors that agree on the certified
    objectives and disagree wildly on the excluded one must produce the SAME
    order -- otherwise the excluded objective is still steering the loop."""

    pool = [_cfg(i) for i in range(6)]
    base = [{"ones": float(i), "zeros": float(6 - i)} for i in range(6)]

    def with_scramble(values):
        def predict(candidates):
            return [dict(row, scramble=v) for row, v in zip(base, values)]
        return predict

    a = screen_offspring(pool, [], SPECS, with_scramble([9.0] * 6),
                         objectives=["ones", "zeros"])
    b = screen_offspring(pool, [], SPECS,
                         with_scramble([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]),
                         objectives=["ones", "zeros"])
    assert a is not None and b is not None
    assert a.order == b.order, (
        "the excluded objective changed the order: it is being consumed"
    )
    assert a.screened_objectives == ("ones", "zeros")
    assert a.declared_objectives == NAMES
    assert a.partial is True
    assert all(set(row) == {"ones", "zeros"} for row in a.predicted), (
        "an uncertified prediction must not even be reported as one"
    )

    full = screen_offspring(pool, [], SPECS,
                            with_scramble([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]))
    assert full.screened_objectives == NAMES and full.partial is False


def test_ordering_on_a_subset_sinks_a_candidate_strong_on_the_rest():
    """The COST of partial screening, pinned rather than argued.

    Domination over a subset is a stricter relation than domination over the
    whole problem: pairs that are incomparable on three objectives compare on
    two. So a candidate that is excellent only on the objective the screen
    cannot see is dominated, and sinks to the bottom of the order. This is
    exactly why `Screening.unscreened_objective_floor` exists, and why a
    partial screen must never be given the flat floor a full one gets.
    """

    specs = [ObjectiveSpec(n, "min") for n in ("m1", "m2", "m3")]
    pool = [_cfg(0), _cfg(1)]
    predictions = [
        {"m1": 10.0, "m2": 10.0, "m3": 0.0},   # only m3 is good
        {"m1": 1.0, "m2": 1.0, "m3": 50.0},    # good on m1, m2
    ]

    def predict(candidates):
        return predictions

    whole = screen_offspring(pool, [], specs, predict)
    assert whole.order == (0, 1), (
        "on all three objectives neither dominates, so the tie breaks on "
        "index and the strong-on-m3 candidate keeps its place"
    )
    subset = screen_offspring(pool, [], specs, predict,
                              objectives=["m1", "m2"])
    assert subset.order == (1, 0), (
        "on the subset the m3 specialist is dominated and sinks -- the known "
        "bias of this mechanism, and the reason for the raised floor"
    )


def test_the_screen_refuses_an_empty_or_undeclared_scope():
    pool = [_cfg(0)]

    def predict(candidates):
        return [{"ones": 1.0, "zeros": 1.0, "scramble": 1.0}]

    assert screen_offspring(pool, [], SPECS, predict, objectives=[]) is None, (
        "ordering on nothing is not ordering, and must not look like it"
    )
    with pytest.raises(ValueError, match="declared objectives"):
        screen_offspring(pool, [], SPECS, predict, objectives=["latency"])


# --- the floor rises with what the screen cannot see --------------------------

def _report(screened: Sequence[str]) -> ScreenReport:
    return ScreenReport(order=(0,), predicted=({},), virtual_evaluations=1,
                        surrogate_name="x",
                        screened_objectives=tuple(screened),
                        declared_objectives=NAMES)


def test_the_exploration_floor_rises_with_the_unscreened_share():
    screening = Screening(builders=(("additive", "rule", additive_surrogate),),
                          exploration_floor=0.25)
    assert screening.unscreened_objective_floor == 1.0
    assert screening.exploration_floor_for(_report(NAMES)) == 0.25, (
        "a screen that saw the whole problem gets the ordinary floor"
    )
    assert math.isclose(
        screening.exploration_floor_for(_report(["ones", "zeros"])), 1 / 3), (
        "blind to one of three objectives: reserve a third of the generation"
    )
    assert math.isclose(
        screening.exploration_floor_for(_report(["ones"])), 2 / 3)

    flat = Screening(builders=(("additive", "rule", additive_surrogate),),
                     exploration_floor=0.25, unscreened_objective_floor=0.0)
    assert flat.exploration_floor_for(_report(["ones"])) == 0.25, (
        "the knob is a knob: at 0.0 a partial screen is treated like a full "
        "one, which is the arm the default was measured against"
    )
    high = Screening(builders=(("additive", "rule", additive_surrogate),),
                     exploration_floor=0.75)
    assert high.exploration_floor_for(_report(["ones", "zeros"])) == 0.75, (
        "the ordinary floor is a lower bound, never lowered by this rule"
    )
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="unscreened_objective_floor"):
            Screening(builders=(("a", "rule", additive_surrogate),),
                      unscreened_objective_floor=bad)


# --- the variance guard, per objective ---------------------------------------

def test_a_builder_whose_certified_objectives_move_between_splits_is_refused():
    """Every split passing is not enough when they disagree about WHAT passed.

    An artifact certified on {ones, zeros} by one re-partition and on
    {ones, scramble} by the next has shown it can order `ones` and nothing
    else stably. The certified set is the intersection and it must still meet
    the policy's requirement.
    """

    # The builder is refitted once per FOLD, so the alternation has to be
    # driven off the split rather than off the fit: fold k of split s is call
    # s * folds + k. Splits 0 and 2 certify {ones, zeros}, split 1 certifies
    # {ones, scramble}, every split passes the policy, and the intersection is
    # {ones} alone.
    folds = PARTIAL_ORDERING_GATE.folds_for(len(DATA))
    calls = {"n": 0}

    def unstable(evaluated, specs):
        split = calls["n"] // folds
        calls["n"] += 1
        exact = (["ones", "scramble"] if split % 2 else ["ones", "zeros"])

        def predict(pool):
            out = []
            for candidate in pool:
                truth = _truth(candidate["genome"])
                out.append({spec.name: (truth[spec.name]
                                        if spec.name in exact else 1.0)
                            for spec in specs})
            return out
        return predict

    screening = Screening(builders=(("unstable", "llm", unstable),),
                          gate=PARTIAL_ORDERING_GATE, validation_splits=3)
    assert not screening.refresh(DATA, SPECS, seed=6)
    assert calls["n"] == 3 * folds, (
        "every split ran, so none of them FAILED -- what failed is that they "
        "certified different objectives"
    )
    assert screening.telemetry.rejected_unstable_subset == 1, (
        "this failure must be countable and must not read as 'no data'"
    )
    assert screening.telemetry.rejected_validation == 0

    stable = Screening(builders=(("stable", "rule", _builder(["ones", "zeros"])),),
                       gate=PARTIAL_ORDERING_GATE, validation_splits=3)
    assert stable.refresh(DATA, SPECS, seed=6)
    assert stable.telemetry.rejected_unstable_subset == 0


def test_more_certified_objectives_beats_a_better_error_ratio():
    """Arbitration must not prefer a narrow scope for a flattering ratio: an
    artifact that orders the whole problem is a different instrument from one
    that orders two thirds of it."""

    screening = Screening(
        builders=(("narrow", "llm", _builder(["ones", "zeros"])),
                  ("wide", "rule", _builder(list(NAMES)))),
        gate=PARTIAL_ORDERING_GATE, validation_splits=1)
    assert screening.refresh(DATA, SPECS, seed=5)
    assert screening.authored_by == "rule", (
        "the builder certified on all three objectives must win"
    )
    report = screening.screen([_cfg(0), _cfg(1)], [], SPECS)
    assert report is not None and report.screened_objectives == NAMES

    only_narrow = Screening(
        builders=(("narrow", "llm", _builder(["ones", "zeros"])),),
        gate=PARTIAL_ORDERING_GATE, validation_splits=1)
    assert only_narrow.refresh(DATA, SPECS, seed=5)
    narrow_report = only_narrow.screen([_cfg(0), _cfg(1)], [], SPECS)
    assert narrow_report.screened_objectives == ("ones", "zeros")


# --- telemetry is a correctness requirement -----------------------------------

def test_a_run_can_never_imply_it_screened_on_more_than_it_did():
    screening = Screening(
        builders=(("narrow", "llm", _builder(["ones", "zeros"])),),
        gate=PARTIAL_ORDERING_GATE, validation_splits=1)
    assert screening.refresh(DATA, SPECS, seed=5)
    for _ in range(3):
        screening.screen([_cfg(0), _cfg(1)], [], SPECS)

    counters = screening.telemetry.as_dict()
    assert counters["screens"] == 3
    assert counters["screens_partial"] == 3
    assert counters["screens_full"] == 0
    assert counters["screened_on:ones"] == 3
    assert counters["screened_on:zeros"] == 3
    assert "screened_on:scramble" not in counters, (
        "an objective the screen never ordered on must not appear at all"
    )
    assert all(isinstance(v, int) for v in counters.values()), (
        "harvest_telemetry casts every counter to int; a nested structure "
        "here would crash a run at the end rather than at the edit"
    )
    harvested = harvest_telemetry((screening,), real_evaluations=1,
                                  virtual_evaluations=1)
    assert harvested.mechanisms[0].counters["screened_on:ones"] == 3


# --- end to end, through the shipped configuration ---------------------------

class _Problem:
    candidate_model = type(
        "G", (BaseModel,), {"__annotations__": {"genome": list[Literal[0, 1]]}}
    )
    objectives = tuple(SPECS)

    def __init__(self) -> None:
        self.calls = 0

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ({"genome": [0, 0, 0, 0, 0, 0]},)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return tuple(config["genome"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        self.calls += 1
        return _truth(artifact)


def _screens(result) -> List[Mapping[str, Any]]:
    return [h["screen"] for h in result.history if "screen" in h]


def test_the_default_run_is_the_conjunction_and_says_which_objectives():
    from agent_evolve.session.authorship import AuthorshipConfig

    assert AuthorshipConfig().screen_min_passing_objectives == 0, (
        "partial screening is opt-in; the shipped default is the conjunction"
    )
    problem = _Problem()
    result = optimize(problem, budget=24, seed=9, authorship="surrogate")
    assert problem.calls <= 24
    active = [s for s in _screens(result) if s.get("active")]
    for note in active:
        assert "objectives" in note and "partial" in note, (
            "every screened generation records its scope"
        )
        if note.get("advanced", 0) or "objectives_declared" in note:
            assert note["objectives_declared"] == list(NAMES)


def test_a_partial_run_holds_a_bigger_floor_and_records_the_subset():
    from agent_evolve.session.authorship import AuthorshipConfig

    problem = _Problem()
    result = optimize(
        problem, budget=24, seed=9,
        authorship=AuthorshipConfig(surrogate="rule",
                                    screen_min_passing_objectives=2))
    assert problem.calls <= 24
    assert result.telemetry.real_evaluations == result.evaluations
    partial_notes = [s for s in _screens(result)
                     if s.get("active") and s.get("partial")]
    for note in partial_notes:
        assert len(note["objectives"]) < len(note["objectives_declared"])
        assert note["held_out"] >= 1, (
            "the exploration floor must hold every screened generation"
        )
    rows = [m for m in result.telemetry.mechanisms
            if m.mechanism == "surrogate_screen"]
    assert rows, "the screen must report its counters"
    counters = rows[0].counters
    screened_keys = [k for k in counters if k.startswith("screened_on:")]
    assert set(screened_keys) <= {f"screened_on:{n}" for n in NAMES}
    assert (counters.get("screens_full", 0)
            + counters.get("screens_partial", 0) == counters.get("screens", 0))


def test_a_two_objective_problem_is_untouched_by_the_partial_policy():
    """`min_passing_objectives=2` on two objectives IS the conjunction, and
    the run must be bit-identical -- the change must not leak into problems
    it has no business touching."""

    from agent_evolve.session.authorship import AuthorshipConfig

    class _TwoObjective(_Problem):
        objectives = (ObjectiveSpec("ones", "max"), ObjectiveSpec("zeros", "min"))

        def evaluate(self, artifact) -> Mapping[str, float]:
            self.calls += 1
            row = _truth(artifact)
            return {"ones": row["ones"], "zeros": row["zeros"]}

    baseline = optimize(_TwoObjective(), budget=24, seed=9,
                        authorship="surrogate")
    partial = optimize(
        _TwoObjective(), budget=24, seed=9,
        authorship=AuthorshipConfig(surrogate="rule",
                                    screen_min_passing_objectives=2))
    assert [dict(c.objectives) for c in baseline.all_candidates] == [
        dict(c.objectives) for c in partial.all_candidates], (
        "the same seed on a two-objective problem must produce the same run"
    )
    assert [dict(h) for h in baseline.history] == [
        dict(h) for h in partial.history]
    assert (baseline.telemetry.virtual_evaluations
            == partial.telemetry.virtual_evaluations)
    assert not any(s.get("partial") for s in _screens(partial))
