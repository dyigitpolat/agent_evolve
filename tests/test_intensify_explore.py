"""Declared-domain exploration and incumbent intensification: two knobs, one fix.

Both mechanisms answer one measurement, and neither is expected to pay alone.

The analog venue's screen fitted every prior inside a box: the highest level any
installed prior allowed was 13-15 of a 23-32 level ladder, in 20 of 20 cells.
Pooling 12,899 real evaluations, the best point inside that box reads -0.4002
reward9 and the best outside reads -0.0566, and 100 of the pooled top-100 carry
at least one coordinate the box excludes. Our finals sit at -0.353 to -0.555 --
at the ceiling of the box, not failing to exploit what is inside it. So
``explore="coverage"`` draws against the DECLARED domain rather than the prior's
narrowed support, which is the whole point and the one detail that makes the
mechanism different from the coverage channel that could have been written
inside ``allowed``.

But the box is soft, and our arms already leave it: two cells spend 82% and 96%
of their post-screen charges outside and come back with -0.457 and -0.452, no
better than the cells that stay in. Leaving without intensifying pays nothing.
So ``intensify="incumbent"`` is the other half -- and it is the classical
reading of what a third-party optimizer's winning populations turned out to be
when they were measured: one incumbent-anchored cluster in 6 of 6 cells, zero of
156 entropy-controlled coordinate pairs above z = 2, modal purity 0.695-0.842
over 6-12 fields. Not a couplings table. One member with most of its fields
pinned and the rest redrawn.

Every test here is offline and classical: no provider, no network, no
credential, and neither mechanism ever makes a model call. The rule these tests
are strictest about is the same one the polish file is strictest about: both
knobs are OFF by default, and "off" means byte-identical rather than similar.
The absolute pre-substrate digests live in ``test_fossil_stream.py``; what is
pinned here is that stating the two knobs off is the same run as not stating
them, at two budgets.
"""

from __future__ import annotations

import inspect
import json
import random
from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple

import pytest
from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.policies.genetic import (
    DomainRestriction,
    Locus,
    coverage_candidate,
    coverage_counts,
    incumbent_candidate,
)
from agent_evolve.session.genetic_loop import (
    GeneticConfig,
    best_member,
    explore_probability,
    run_genetic_loop,
)

# --------------------------------------------------------------------------
# a venue with enough loci to pin 6-12 of them
# --------------------------------------------------------------------------

_LEVEL = Literal[0, 1, 2, 3, 4, 5, 6, 7]
_WIDTH = 12
_DOMAIN = tuple(range(8))


class _Vector(BaseModel):
    genome: list[_LEVEL]


class _Ladder:
    """Twelve loci of eight declared levels each, minimised on their sum.

    One objective, so the rank-0 set is the argmin and "the incumbent" is a
    statement a test can check rather than a convention it has to trust. The
    evaluate stream is recorded because every claim below is about WHICH
    configurations were put in front of the evaluator, in what order.
    """

    candidate_model = _Vector
    objectives = (ObjectiveSpec(name="cost", goal="min"),)

    def __init__(self, seed_genome: Sequence[int] = (0,) * _WIDTH) -> None:
        self.stream: List[List[int]] = []
        self._seed = list(seed_genome)

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ({"genome": list(self._seed)},)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return tuple(config["genome"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        self.stream.append(list(artifact))
        return {"cost": float(sum(artifact))}


class _Seedless(_Ladder):
    """The same venue with nothing to evolve from: resolves to authoring."""

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ()


def _loci(width: int = _WIDTH) -> Tuple[Locus, ...]:
    return tuple(Locus("genome", index) for index in range(width))


def _run(problem: _Ladder, **overrides) -> Any:
    settings: Dict[str, Any] = dict(population_size=4,
                                    offspring_per_generation=4,
                                    generations=100, seed=7,
                                    evaluation_budget=40)
    settings.update(overrides)
    return run_genetic_loop(problem=problem, config=GeneticConfig(**settings))


def _counts(result, key: str) -> List[int]:
    return [entry[key] for entry in result.history if key in entry]


# --------------------------------------------------------------------------
# the schedule
# --------------------------------------------------------------------------


def test_the_exploration_schedule_is_linear_in_the_charged_budget() -> None:
    """e0 at the first charge, e_min at the last, and the midpoint at half."""

    schedule = (0.5, 0.1)
    assert explore_probability(schedule, spent=0, budget=320) == 0.5
    assert explore_probability(schedule, spent=320, budget=320) == 0.1
    assert explore_probability(schedule, spent=160, budget=320) == \
        pytest.approx(0.3)
    assert explore_probability(schedule, spent=80, budget=320) == \
        pytest.approx(0.4)
    assert explore_probability(schedule, spent=240, budget=320) == \
        pytest.approx(0.2)
    # A run that somehow charges past its own budget is pinned at the floor
    # rather than extrapolated below it.
    assert explore_probability(schedule, spent=400, budget=320) == 0.1
    # And one with no budget has no horizon to decline over.
    assert explore_probability(schedule, spent=17, budget=None) == 0.5
    assert explore_probability(schedule, spent=17, budget=0) == 0.5
    # Monotone over the whole budget, and never outside the declared ends.
    values = [explore_probability(schedule, spent=n, budget=320)
              for n in range(321)]
    assert values == sorted(values, reverse=True)
    assert min(values) == 0.1 and max(values) == 0.5


def test_a_flat_schedule_is_a_constant_and_a_zero_schedule_is_off() -> None:
    assert explore_probability((0.25, 0.25), spent=99, budget=320) == 0.25
    assert explore_probability((0.0, 0.0), spent=0, budget=320) == 0.0


# --------------------------------------------------------------------------
# coverage: the DECLARED domain, counted off the run's own trace
# --------------------------------------------------------------------------


def test_coverage_targets_the_least_measured_declared_values() -> None:
    """A hand-built trace with exactly one unmeasured value per locus.

    Row *k* sets locus *i* to ``(i + k) % 8``, so seven rows cover seven of the
    eight declared levels at every locus and leave exactly one uncovered -- a
    different one per locus. The least-measured set is then a singleton
    everywhere and the draw is fully determined: whatever RNG it is handed, the
    candidate is the vector of the eight-modulo levels the trace never saw.
    """

    template = {"genome": [0] * _WIDTH}
    rows = [{"genome": [(index + k) % 8 for index in range(_WIDTH)]}
            for k in range(7)]
    counts = coverage_counts(rows, template, _Vector)

    # The tally is aligned to the declared domain, one slot per declared value.
    for index in range(_WIDTH):
        tally = counts[Locus("genome", index)]
        assert len(tally) == len(_DOMAIN)
        assert sum(tally) == len(rows)
        assert tally[(index + 7) % 8] == 0
        assert all(count == 1 for value, count in enumerate(tally)
                   if value != (index + 7) % 8)

    for rng_seed in (0, 1, 2):
        candidate = coverage_candidate(template, _Vector, counts=counts,
                                       rng=random.Random(rng_seed))
        assert candidate["genome"] == [(index + 7) % 8
                                       for index in range(_WIDTH)]


def test_coverage_counts_the_declared_domain_and_not_the_allowed_one() -> None:
    """The load-bearing detail, stated as a test.

    A restriction that admits two of eight levels does not shrink what coverage
    counts or what it draws: the tally still has eight slots, and a trace made
    entirely of allowed rows leaves the six EXCLUDED levels as the
    least-measured ones. A channel that counted inside `allowed` would report
    this venue fully covered while the whole region holding its optimum went
    unmeasured -- which is the analog venue's measured history.
    """

    template = {"genome": [0] * _WIDTH}
    rows = [{"genome": [k % 2] * _WIDTH} for k in range(6)]
    counts = coverage_counts(rows, template, _Vector)
    assert len(counts[Locus("genome", 0)]) == len(_DOMAIN)

    candidate = coverage_candidate(template, _Vector, counts=counts,
                                   rng=random.Random(11))
    assert all(value >= 2 for value in candidate["genome"])
    # And the helper takes no restriction at all: there is no argument through
    # which a prior could narrow this draw.
    assert "restriction" not in inspect.signature(
        coverage_candidate).parameters


def test_an_empty_trace_covers_nothing_and_draws_over_everything() -> None:
    template = {"genome": [0] * _WIDTH}
    counts = coverage_counts([], template, _Vector)
    assert all(tally == [0] * len(_DOMAIN) for tally in counts.values())
    drawn = {value
             for rng_seed in range(40)
             for value in coverage_candidate(
                 template, _Vector, counts=counts,
                 rng=random.Random(rng_seed))["genome"]}
    assert drawn == set(_DOMAIN)


def test_a_trace_extends_a_tally_in_place_rather_than_being_re_walked() -> None:
    template = {"genome": [0] * _WIDTH}
    counts = coverage_counts([{"genome": [3] * _WIDTH}], template, _Vector)
    same = coverage_counts([{"genome": [3] * _WIDTH}], template, _Vector,
                           counts=counts)
    assert same is counts
    assert counts[Locus("genome", 0)][3] == 2
    assert coverage_counts([{"genome": [3] * _WIDTH}] * 2, template,
                           _Vector) == counts


def test_a_row_that_does_not_carry_a_locus_contributes_nothing_to_it() -> None:
    """A ragged row counts where it has values and is silent where it does not."""

    template = {"genome": [0] * _WIDTH}
    counts = coverage_counts([{"genome": [5, 5]}], template, _Vector)
    assert counts[Locus("genome", 0)][5] == 1
    assert counts[Locus("genome", 11)] == [0] * len(_DOMAIN)


def test_a_locus_the_schema_does_not_constrain_gets_no_tally(
) -> None:
    template = {"genome": [0] * _WIDTH, "free": "anything"}
    counts = coverage_counts([template], template, _Vector)
    assert Locus("free") not in counts
    candidate = coverage_candidate(template, _Vector, counts=counts,
                                   rng=random.Random(3))
    assert candidate["free"] == "anything"


# --------------------------------------------------------------------------
# intensification: pin q, resample the rest through the prior
# --------------------------------------------------------------------------


def test_intensification_pins_exactly_the_named_loci(
) -> None:
    """Pinned loci keep the incumbent; every other one is redrawn.

    The restriction admits one value, so the resampled half is deterministic
    and "exactly q pinned" is an exact count rather than a probable one.
    """

    incumbent = {"genome": [0] * _WIDTH}
    restriction = DomainRestriction({"genome": (7,)})
    for pinned in (1, 6, 9, _WIDTH):
        pin = _loci()[:pinned]
        child = incumbent_candidate(incumbent, _Vector, pin=pin,
                                    rng=random.Random(2),
                                    restriction=restriction)
        assert child["genome"][:pinned] == [0] * pinned
        assert child["genome"][pinned:] == [7] * (_WIDTH - pinned)
        assert sum(1 for value in child["genome"] if value == 0) == pinned


def test_intensification_resamples_through_the_prior_in_force() -> None:
    """The unpinned half is drawn through the restriction, not around it."""

    incumbent = {"genome": [3] * _WIDTH}
    restriction = DomainRestriction({"genome": (4, 5)})
    pin = _loci()[:6]
    for rng_seed in range(8):
        child = incumbent_candidate(incumbent, _Vector, pin=pin,
                                    rng=random.Random(rng_seed),
                                    restriction=restriction)
        assert child["genome"][:6] == [3] * 6
        assert all(value in (4, 5) for value in child["genome"][6:])
    # With no restriction the same call draws from the whole declared domain --
    # a fresh draw like `uniform_candidate`'s, not a mutation, so the value the
    # incumbent already carries is among the ones it can land on again.
    seen = {value
            for rng_seed in range(40)
            for value in incumbent_candidate(
                incumbent, _Vector, pin=pin,
                rng=random.Random(rng_seed))["genome"][6:]}
    assert seen == set(_DOMAIN)


def test_pinning_every_locus_reproduces_the_incumbent() -> None:
    """Which is exactly why the loop dedups: q = n is a copy, not a child."""

    incumbent = {"genome": list(range(_WIDTH))}
    child = incumbent_candidate(incumbent, _Vector, pin=_loci(),
                                rng=random.Random(0))
    assert child == incumbent


def test_a_locus_with_no_declared_domain_is_kept_however_it_was_pinned() -> None:
    incumbent = {"genome": [0] * _WIDTH, "free": "x"}
    child = incumbent_candidate(incumbent, _Vector, pin=(),
                                rng=random.Random(1))
    assert child["free"] == "x"


# --------------------------------------------------------------------------
# the incumbent
# --------------------------------------------------------------------------


def test_the_incumbent_is_the_rank_zero_best_on_the_first_objective() -> None:
    specs = (ObjectiveSpec(name="a", goal="min"),
             ObjectiveSpec(name="b", goal="max"))
    population = [({"tag": 0}, {"a": 3.0, "b": 9.0}),
                  ({"tag": 1}, {"a": 1.0, "b": 1.0}),
                  ({"tag": 2}, {"a": 2.0, "b": 5.0}),
                  ({"tag": 3}, {"a": 0.0, "b": 0.0})]
    # Ranks are supplied, so the choice among rank-0 members is what is tested.
    assert best_member(population, [0, 0, 1, 0], specs)["tag"] == 3
    # A maximised first objective is read through its own goal.
    flipped = (ObjectiveSpec(name="a", goal="max"),)
    assert best_member(population, [0, 0, 1, 0], flipped)["tag"] == 0
    # Ties go to the member measured earliest.
    tied = [({"tag": 0}, {"a": 1.0}), ({"tag": 1}, {"a": 1.0})]
    assert best_member(tied, [0, 0], (ObjectiveSpec(name="a", goal="min"),)
                       )["tag"] == 0
    assert best_member([], [], specs) is None
    # A population in which nothing is rank 0 has no incumbent to name.
    assert best_member(population, [1, 1, 1, 1], specs) is None


# --------------------------------------------------------------------------
# the loop: slots, order, and the counters
# --------------------------------------------------------------------------


def test_exploration_reaches_the_values_the_prior_excludes() -> None:
    """The mechanism's whole reason for existing, measured end to end.

    The same run, with the same restriction, differs in exactly one knob. With
    exploration off nothing outside the two allowed levels is ever measured
    after the seed; with it on, the run measures values the prior cannot draw.
    """

    restriction = DomainRestriction({"genome": (0, 1)})
    boxed = _Ladder()
    _run(boxed, restriction=DomainRestriction({"genome": (0, 1)}))
    assert all(value in (0, 1) for row in boxed.stream for value in row)

    opened = _Ladder()
    result = _run(opened, restriction=restriction, explore="coverage")
    outside = [row for row in opened.stream
               if any(value not in (0, 1) for value in row)]
    assert outside, "exploration drew nothing the prior excludes"
    # And it is the exploration slots that did it: the counter moved.
    assert sum(_counts(result, "explore")) >= len(outside)


def test_a_flat_full_schedule_makes_every_slot_an_exploration_slot() -> None:
    """The schedule's two ends, read at the loop rather than at the formula."""

    always = _Ladder()
    result = _run(always, explore="coverage", explore_schedule=(1.0, 1.0))
    counts = _counts(result, "explore")
    assert counts and all(count == 4 for count in counts[:-1])

    never = _Ladder()
    off_result = _run(never, explore="coverage", explore_schedule=(0.0, 0.0))
    # Recorded, and recorded as zero: a measured zero is not an absence.
    assert _counts(off_result, "explore") == [0] * len(
        _counts(off_result, "explore"))
    assert _counts(off_result, "explore")


def test_the_declining_schedule_spends_early_and_not_late() -> None:
    problem = _Ladder()
    result = _run(problem, explore="coverage", evaluation_budget=200,
                  explore_schedule=(1.0, 0.0))
    counts = _counts(result, "explore")
    assert len(counts) >= 8
    half = len(counts) // 2
    assert sum(counts[:half]) > sum(counts[half:])


def test_intensification_pins_the_incumbent_and_draws_the_rest_from_the_prior(
) -> None:
    """A generation made entirely of intensified children, checked one by one.

    ``intensify_fraction=1.0`` takes every offspring slot, so every charge
    after the initial population is an intensified child and none of them is a
    bred one. The seed is the venue's optimum and the restriction admits only
    level 7, so each child reads as exactly q incumbent levels and 12 - q
    prior-drawn ones -- and q must lie in the declared band.
    """

    # The cage here is DELIBERATELY maximally wrong (the all-zero seed is the
    # venue optimum and sits outside the {7} support) so prior-drawn values
    # are identifiable. X2b now falsifies exactly such a cage once its warmup
    # and streak elapse (generation 6), so the budget ends the run before the
    # falsification window opens and the invariant is measured while the
    # restriction is in force.
    problem = _Ladder()
    result = _run(problem, intensify="incumbent", intensify_fraction=1.0,
                  restriction=DomainRestriction({"genome": (7,)}),
                  intensify_pin_range=(6, 12), evaluation_budget=18)
    # The initial population is the seed and the one distinct uniform draw.
    assert problem.stream[0] == [0] * _WIDTH
    # X2c spends any stranded budget on deliberately UNRESTRICTED fill draws
    # after the generations end; the intensify invariant governs the loop's
    # own offspring, so the disclosed fill count is sliced off the tail.
    filled = sum(h.get("fill", 0)
                 for h in result.history if isinstance(h, dict))
    offspring = problem.stream[2:len(problem.stream) - filled or None]
    assert offspring
    for row in offspring:
        assert set(row) <= {0, 7}
        pinned = sum(1 for value in row if value == 0)
        assert 6 <= pinned <= 11        # 12 would be the incumbent, deduped
    assert sum(_counts(result, "intensify")) == len(offspring)


def test_the_fraction_is_a_floored_share_of_the_generation() -> None:
    """A quarter of four offspring is one slot; a half is two.

    Floored, and floored on the declared arithmetic rather than on binary
    noise -- 0.3 of ten slots is three, not the two that flooring
    2.9999999999999996 would give.
    """

    for fraction, slots in ((0.25, 1), (0.5, 2), (0.75, 3), (0.1, 0)):
        problem = _Ladder()
        result = _run(problem, intensify="incumbent",
                      intensify_fraction=fraction,
                      intensify_pin_range=(6, 11))
        counts = _counts(result, "intensify")
        assert counts, "the run has to reach a generation to count one"
        assert all(count == slots for count in counts[:-1]), (
            f"intensify_fraction={fraction} of four offspring should take "
            f"{slots} slot(s), and read {counts}")


def test_the_pin_band_is_clamped_to_a_genome_shorter_than_it() -> None:
    """A band read off a 24-locus venue must not ask a 4-locus one for 12."""

    class _Short(BaseModel):
        genome: list[_LEVEL]

    problem = _Ladder([0, 0, 0, 0])
    problem.candidate_model = _Short
    result = _run(problem, intensify="incumbent", intensify_fraction=1.0,
                  intensify_pin_range=(6, 12))
    # Every draw pins the whole genome, which is the incumbent, which is
    # already measured -- so the mechanism honestly reports nothing.
    assert _counts(result, "intensify") == [0] * len(
        _counts(result, "intensify"))
    assert result.evaluations > 0


def test_a_slot_that_can_only_repeat_the_trace_is_given_back_to_breeding(
) -> None:
    """Dedup, and what dedup costs: nothing.

    Pinning all twelve loci reproduces the incumbent, which the initial
    population already measured. The counter reads zero in every generation,
    and the generation still spends its whole offspring count -- an arm that
    quietly shrank its generations would be spending less budget than the
    control it is compared against.
    """

    deduped = _Ladder()
    result = _run(deduped, intensify="incumbent", intensify_fraction=1.0,
                  intensify_pin_range=(12, 12))
    assert _counts(result, "intensify") == [0] * len(
        _counts(result, "intensify"))

    plain = _Ladder()
    baseline = _run(plain)
    assert deduped.stream == plain.stream
    assert result.evaluations == baseline.evaluations


def test_exploration_takes_its_slots_before_intensification() -> None:
    """The declared order, and the arithmetic that follows from it."""

    problem = _Ladder()
    result = _run(problem, explore="coverage", explore_schedule=(1.0, 1.0),
                  intensify="incumbent", intensify_fraction=1.0)
    explored = _counts(result, "explore")
    intensified = _counts(result, "intensify")
    assert explored and intensified
    # Exploration took every slot, so intensification -- which is offered what
    # exploration left -- got none of them, however large its fraction is.
    assert all(count == 4 for count in explored[:-1])
    assert intensified == [0] * len(intensified)


def test_a_swept_generation_supersedes_both_mechanisms() -> None:
    """Polish already IS the generation's answer, so no slot is taken from it."""

    problem = _Ladder([0] * _WIDTH)
    result = _run(problem, polish="sweep", polish_after=1,
                  explore="coverage", explore_schedule=(1.0, 1.0),
                  intensify="incumbent", intensify_fraction=1.0)
    swept = {entry["gen"] for entry in result.history
             if entry.get("polish", {}).get("engaged")}
    assert swept, "the run has to sweep at least once to check what it skipped"
    for entry in result.history:
        if entry.get("gen") in swept:
            assert entry["explore"] == 0
            assert entry["intensify"] == 0


def test_the_generation_spends_the_slots_it_would_have_spent_anyway() -> None:
    """Both mechanisms take SLOTS, never charges: the budget buys the same run."""

    for knobs in ({"explore": "coverage"},
                  {"intensify": "incumbent"},
                  {"explore": "coverage", "intensify": "incumbent"}):
        problem = _Ladder()
        result = _run(problem, evaluation_budget=48, **knobs)
        assert result.evaluations <= 48
        assert len(problem.stream) == result.evaluations


# --------------------------------------------------------------------------
# off is off
# --------------------------------------------------------------------------


def test_both_knobs_off_is_byte_identical_to_the_absent_knobs() -> None:
    """Two runs: the default, and the default said out loud. Same run."""

    for budget in (24, 48):
        absent = _Ladder([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
        stated = _Ladder([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
        first = _run(absent, evaluation_budget=budget)
        second = _run(stated, evaluation_budget=budget, explore="off",
                      intensify="off", explore_schedule=(0.5, 0.1),
                      intensify_fraction=0.25, intensify_pin_range=(6, 12))
        assert absent.stream == stated.stream
        assert json.dumps(first.history, sort_keys=True, default=str) == \
            json.dumps(second.history, sort_keys=True, default=str)
        assert first.evaluations == second.evaluations
        # Nothing mechanism-shaped is recorded when nobody asked for either.
        assert not any("explore" in entry or "intensify" in entry
                       for entry in first.history)


def test_off_draws_nothing_so_the_streams_cannot_diverge_later() -> None:
    """The strict form: the RNG stream itself, not just the first generation.

    A mechanism that drew from the main stream and then discarded the draw
    would pass a one-generation comparison and fail here, because every later
    generation would be shifted. Two budgets, so a divergence that only shows
    up once the schedule has declined is still caught.
    """

    for budget in (24, 48):
        absent = _Ladder()
        stated = _Ladder()
        optimize(absent, budget=budget, seed=13, proposer="random")
        optimize(stated, budget=budget, seed=13, proposer="random",
                 explore="off", intensify="off")
        assert absent.stream == stated.stream
        assert absent.stream, "a run that measured nothing compares nothing"


def test_the_fossil_shape_is_unmoved_by_stating_the_knobs_off() -> None:
    """The public entry point, on the fossil's own seeds and budget.

    The absolute digests are pinned in ``test_fossil_stream.py``; what is
    pinned here is that naming the two new knobs at their defaults does not
    change the credential-free default path.
    """

    for seed in (11, 23):
        absent = _Ladder()
        stated = _Ladder()
        first = optimize(absent, budget=16, seed=seed)
        second = optimize(stated, budget=16, seed=seed, explore="off",
                          intensify="off")
        assert absent.stream == stated.stream
        assert first.evaluations == second.evaluations


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_the_loop_refuses_an_unknown_mode_by_name() -> None:
    with pytest.raises(ValueError, match="explore must be 'off' or 'coverage'"):
        _run(_Ladder(), explore="everywhere")
    with pytest.raises(ValueError,
                       match="intensify must be 'off' or 'incumbent'"):
        _run(_Ladder(), intensify="hard")


def test_the_loop_refuses_a_schedule_that_is_not_two_probabilities() -> None:
    with pytest.raises(ValueError, match="explore_schedule must be a pair"):
        _run(_Ladder(), explore_schedule=(0.5,))
    with pytest.raises(ValueError, match="explore_schedule must be a pair"):
        _run(_Ladder(), explore_schedule=0.5)
    with pytest.raises(ValueError, match="two probabilities"):
        _run(_Ladder(), explore_schedule=(1.5, 0.1))
    with pytest.raises(ValueError, match="two probabilities"):
        _run(_Ladder(), explore_schedule=(0.5, "0.1"))
    # Validated whether or not the mechanism is on: a nonsense schedule beside
    # explore="off" is a bug waiting for the day someone turns it on.
    with pytest.raises(ValueError, match="two probabilities"):
        _run(_Ladder(), explore="off", explore_schedule=(-1.0, 0.1))


def test_the_loop_refuses_an_impossible_pin_band_or_fraction() -> None:
    with pytest.raises(ValueError, match="intensify_pin_range must be"):
        _run(_Ladder(), intensify_pin_range=(0, 4))
    with pytest.raises(ValueError, match="intensify_pin_range must be"):
        _run(_Ladder(), intensify_pin_range=(9, 3))
    with pytest.raises(ValueError, match="intensify_pin_range must be a pair"):
        _run(_Ladder(), intensify_pin_range=(1, 2, 3))
    with pytest.raises(ValueError, match="intensify_fraction must be"):
        _run(_Ladder(), intensify_fraction=1.5)
    with pytest.raises(ValueError, match="intensify_fraction must be"):
        _run(_Ladder(), intensify_fraction="a quarter")


def test_optimize_validates_both_knobs_by_name() -> None:
    with pytest.raises(ValueError, match="explore must be 'off' or 'coverage'"):
        optimize(_Ladder(), budget=8, explore="yes")
    with pytest.raises(ValueError,
                       match="intensify must be 'off' or 'incumbent'"):
        optimize(_Ladder(), budget=8, intensify="a lot")
    with pytest.raises(ValueError, match="two probabilities"):
        optimize(_Ladder(), budget=8, explore_schedule=(2.0, 0.1))
    with pytest.raises(ValueError, match="intensify_pin_range must be"):
        optimize(_Ladder(), budget=8, intensify_pin_range=(0, 2))


def test_optimize_refuses_the_knobs_on_the_authoring_strategy() -> None:
    """A knob the run would ignore is a silent no-op, so it is refused."""

    with pytest.raises(ValueError, match="explore"):
        optimize(_Seedless(), budget=8, explore="coverage")
    with pytest.raises(ValueError, match="intensify"):
        optimize(_Seedless(), budget=8, intensify="incumbent")
    with pytest.raises(ValueError, match="explore_schedule"):
        optimize(_Seedless(), budget=8, explore_schedule=(0.9, 0.2))
    with pytest.raises(ValueError, match="intensify_fraction"):
        optimize(_Seedless(), budget=8, intensify_fraction=0.5)
    with pytest.raises(ValueError, match="intensify_pin_range"):
        optimize(_Seedless(), budget=8, intensify_pin_range=(2, 3))
    # ... and a knob nobody moved is this package's own default, so a seedless
    # run that never asked for anything still runs.
    optimize(_Seedless(), budget=4, proposer="random")


def test_optimize_carries_both_knobs_into_the_genetic_loop() -> None:
    problem = _Ladder()
    result = optimize(problem, budget=40, seed=5, proposer="random",
                      explore="coverage", intensify="incumbent")
    assert result.evaluations <= 40
    assert any("explore" in entry for entry in result.history)
    assert any("intensify" in entry for entry in result.history)
