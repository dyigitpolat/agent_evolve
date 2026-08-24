"""Endgame polish and crowding-distance survival: the two knobs, and their off.

Both mechanisms here are answers to a MEASURED loss.

Polish answers the exact-optimum race. On ten NAS seeds NSGA-II reaches the
exact optimum 6W/4L against this loop while losing 1W/9L to it at 10% of
optimum: the region is found and the last grid step is not taken, because a
converged population keeps recombining a front it has already solved. "sweep"
enumerates the 1-mutation neighbourhood of that front instead, deterministically
and with no model call.

Crowding answers the many-objective unit. Where almost nothing dominates
anything -- the five-objective fleet unit -- survival by domination count alone
is near-random, so nothing measured there can host a claim.

Every test is offline and classical: no provider, no network, no credential.
The rule these tests are strictest about is that both knobs are OFF by default
and that "off" means byte-identical, not merely similar.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple

import pytest
from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.policies.genetic import (
    crowding_distances,
    one_mutation_neighbourhood,
    truncation_survival,
)
from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop


# --------------------------------------------------------------------------
# problems
# --------------------------------------------------------------------------


class _Bits(BaseModel):
    genome: list[Literal[0, 1]]


class _OneMax:
    """One-max over eight bits, recording its own evaluate-call stream.

    The stream and the loop's log go into ONE list, in order, so a test can say
    exactly how many charges fell inside each generation -- which is what the
    stall rule counts, and the only way to check it without asking the loop to
    report its own arithmetic.
    """

    candidate_model = _Bits
    objectives = (ObjectiveSpec(name="ones", goal="max"),)

    def __init__(self, seed_genome: Sequence[int]) -> None:
        self.events: List[Tuple[str, Any]] = []
        self._seed = list(seed_genome)

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ({"genome": list(self._seed)},)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return tuple(config["genome"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        self.events.append(("eval", list(artifact)))
        return {"ones": float(sum(artifact))}

    def log(self, message: str) -> None:
        self.events.append(("log", message))


_SIXTEEN = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


class _Flat(BaseModel):
    x: _SIXTEEN
    z: _SIXTEEN


class _AllNonDominated:
    """Two objectives that read the same axis in opposite directions.

    Nothing dominates anything, so every member of every population carries
    domination count zero. This is the fleet unit's geometry in miniature, and
    it is also the only shape in which a rotation over "the rank-0 members" has
    more than one member to rotate over.
    """

    candidate_model = _Flat
    objectives = (ObjectiveSpec(name="lo", goal="min"),
                  ObjectiveSpec(name="hi", goal="max"))

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ({"x": 8, "z": 8},)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return (config["x"], config["z"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        return {"lo": float(artifact[0]), "hi": float(artifact[0])}


class _Seedless(_OneMax):
    """The same problem with nothing to evolve from: resolves to authoring."""

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ()


# --------------------------------------------------------------------------
# reading a run back out of its own event stream
# --------------------------------------------------------------------------


def _marks(events: Sequence[Tuple[str, Any]]) -> List[Tuple[int, int]]:
    """``(charges, best)`` at the end of each generation, oldest first.

    ``charges`` is the evaluate-call count, which for a problem that never
    refuses is exactly what the budget counts. ``best`` stands in for the
    front: with one maximised objective the rank-0 set is the argmax and the
    front's objective-vector set is ``{best}``, so "the front moved" and "the
    best improved" are the same statement.
    """

    out: List[Tuple[int, int]] = []
    charges = 0
    best = 0
    for kind, payload in events:
        if kind == "eval":
            charges += 1
            best = max(best, sum(payload))
        elif payload.startswith("generation ") and (
                " budget used" in payload or " evaluated, population " in payload):
            out.append((charges, best))
    return out


def _predict(
    marks: Sequence[Tuple[int, int]],
    *,
    after: int,
    reserve: float,
    budget: int,
    resets: bool = True,
) -> Dict[int, bool]:
    """Re-derive, generation by generation, when a sweep is allowed to engage.

    An independent implementation of the declared rule -- stall counted in
    CHARGES, reset by any front change, gated by the reserve -- read off the
    charge stream rather than off the loop's own counters. *resets* False is
    the counterfactual: the same rule with the reset removed, which is how the
    reset itself is shown to be doing work rather than merely being harmless.
    """

    allowed: Dict[int, bool] = {}
    front: int | None = None
    stall = 0
    mark = 0
    for generation in range(1, len(marks) + 1):
        charges, best = marks[generation - 1]
        if front is None or (best != front and resets):
            front, stall = best, 0
        else:
            front = best
            stall += charges - mark
        mark = charges
        allowed[generation] = (stall >= after
                               and budget - charges >= reserve * budget)
    return allowed


def _notes(result) -> Dict[int, Dict[str, Any]]:
    return {entry["gen"]: entry["polish"]
            for entry in result.history if "polish" in entry}


def _run(problem: _OneMax, **overrides) -> Any:
    settings = dict(population_size=4, offspring_per_generation=3,
                    generations=100, seed=5, evaluation_budget=48)
    settings.update(overrides)
    return run_genetic_loop(problem=problem, config=GeneticConfig(**settings),
                            log=problem.log)


def _assert_rule_holds(problem: _OneMax, result, *, after: int,
                       reserve: float, budget: int) -> Dict[int, bool]:
    """Every engagement is one the declared rule allows, and no other."""

    allowed = _predict(_marks(problem.events), after=after, reserve=reserve,
                       budget=budget)
    for generation, note in _notes(result).items():
        if note["engaged"]:
            assert allowed[generation], (
                f"generation {generation} swept, but the stall/reserve rule "
                "does not allow it there")
            assert note["proposed"] > 0
            assert note["member_index"] >= 0
        elif allowed.get(generation):
            # Allowed but not engaged has exactly one honest reading: there was
            # nothing left in the neighbourhood to propose.
            assert note["proposed"] == 0 and note["member_index"] == -1
    return allowed


# --------------------------------------------------------------------------
# B1 -- stall detection
# --------------------------------------------------------------------------


def test_the_stall_rule_decides_every_engagement_on_a_solved_front() -> None:
    """A seed at the optimum: the front never moves, so only charges gate."""

    problem = _OneMax([1] * 8)
    result = _run(problem, polish="sweep", polish_after=1)
    allowed = _assert_rule_holds(problem, result, after=1, reserve=0.15,
                                 budget=48)
    notes = _notes(result)
    engaged = [g for g, note in notes.items() if note["engaged"]]
    # Nothing is exhausted before the first sweep, so the first engagement is
    # exactly the first generation the rule permits -- not merely one of them.
    assert engaged and min(engaged) == min(g for g, ok in allowed.items() if ok)
    # The first generation cannot sweep: nothing has stalled yet, because the
    # front it would have to be stalled against did not exist a generation ago.
    assert notes[1]["engaged"] is False
    assert allowed[1] is False
    # Eight bits means eight neighbours. Once they are all measured the sweep
    # has nothing to add and says so rather than re-proposing what it has seen.
    assert sum(note["proposed"] for note in notes.values()) <= 8
    assert any(allowed[g] and not notes[g]["engaged"] for g in notes)


def test_a_front_change_resets_the_stall_and_the_generation_breeds() -> None:
    """The discriminating case: the sweep finds the optimum and stands down.

    Seeded one bit short, the sweep reaches the all-ones genome, the front
    moves, and the next generation must breed. The counterfactual rule -- the
    same arithmetic with the reset removed -- would have swept there, so this
    pins the reset itself and not just the threshold.
    """

    problem = _OneMax([1, 1, 1, 1, 1, 1, 1, 0])
    result = _run(problem, polish="sweep", polish_after=1)
    marks = _marks(problem.events)
    notes = _notes(result)
    _assert_rule_holds(problem, result, after=1, reserve=0.15, budget=48)

    with_reset = _predict(marks, after=1, reserve=0.15, budget=48)
    without = _predict(marks, after=1, reserve=0.15, budget=48, resets=False)
    disagree = [g for g in notes
                if without.get(g) and not with_reset.get(g)]
    assert disagree, (
        "this run never moved its front after a sweep, so it cannot "
        "distinguish a tracker that resets from one that does not")
    for generation in disagree:
        assert notes[generation]["engaged"] is False
        # And not because the neighbourhood ran out: the very next generation
        # proposes again once the charges have accumulated afresh.
        assert any(notes[later]["engaged"]
                   for later in notes if later > generation)


def test_the_stall_counts_charges_and_not_generations() -> None:
    """A run that measures nothing new never stalls, however long it runs.

    One locus and no mutation: every offspring is a copy of a parent, so every
    generation hits the evaluation cache and charges nothing. Generations pile
    up; charges do not; the sweep correctly never engages. A tracker counting
    generations would have swept on the second one.
    """

    class _OneLocus(BaseModel):
        x: Literal[0, 1, 2, 3, 4, 5, 6, 7]

    class _Frozen:
        candidate_model = _OneLocus
        objectives = (ObjectiveSpec(name="score", goal="min"),)

        def __init__(self) -> None:
            self.charges = 0

        def seeds(self):
            return ({"x": 5},)

        def validate(self, config):
            return ValidationOutcome(ok=True)

        def materialize(self, config):
            return config["x"]

        def evaluate(self, artifact):
            self.charges += 1
            return {"score": float(artifact)}

    problem = _Frozen()
    result = run_genetic_loop(
        problem=problem,
        config=GeneticConfig(population_size=2, offspring_per_generation=2,
                             generations=25, seed=4, evaluation_budget=64,
                             mutation_rate=0.0, polish="sweep",
                             polish_after=1))
    notes = _notes(result)
    assert len(notes) == 25                 # the generations really did run
    assert not any(note["engaged"] for note in notes.values())
    assert problem.charges <= 2             # ... and charged only the seeds


# --------------------------------------------------------------------------
# B1 -- what a sweep proposes
# --------------------------------------------------------------------------


def test_a_sweep_proposes_the_neighbourhood_minus_seen_in_order() -> None:
    """Exactly the 1-mutation neighbourhood, deduped, capped, in locus order."""

    problem = _OneMax([1] * 8)
    result = _run(problem, polish="sweep", polish_after=1)
    notes = _notes(result)
    swept = sorted(g for g, note in notes.items() if note["engaged"])
    assert swept, "the run has to sweep at least once to check what it swept"

    # Replay the stream: what was measured before each generation, and what was
    # measured during it.
    boundaries: List[int] = []
    measured: List[Tuple[int, ...]] = []
    for kind, payload in problem.events:
        if kind == "eval":
            measured.append(tuple(payload))
        elif payload.startswith("generation ") and (
                " budget used" in payload or " evaluated, population " in payload):
            boundaries.append(len(measured))

    member = {"genome": [1] * 8}
    for generation in swept:
        before = set(measured[:boundaries[generation - 1]])
        during = measured[boundaries[generation - 1]:boundaries[generation]]
        expected = [tuple(candidate["genome"]) for candidate
                    in one_mutation_neighbourhood(member, _Bits)
                    if tuple(candidate["genome"]) not in before]
        assert during == expected[:len(during)]
        assert len(during) == notes[generation]["proposed"]
        assert len(during) <= 3             # capped at the generation's want


def test_the_neighbourhood_is_every_locus_every_value_nearest_first() -> None:
    """The operator itself: complete, current excluded, nearest value first."""

    class _Pair(BaseModel):
        a: Literal[0, 1, 2, 3]
        b: Literal[0, 1]

    neighbours = list(one_mutation_neighbourhood({"a": 2, "b": 0}, _Pair))
    assert neighbours == [
        {"a": 1, "b": 0},                   # locus a, distance 1
        {"a": 3, "b": 0},                   # locus a, distance 1, declared later
        {"a": 0, "b": 0},                   # locus a, distance 2
        {"a": 2, "b": 1},                   # locus b, the only other value
    ]
    # A locus the schema does not constrain has no neighbour to name.
    assert list(one_mutation_neighbourhood({"a": 2, "b": 0, "free": "x"},
                                           _Pair)) == [
        {"a": 1, "b": 0, "free": "x"},
        {"a": 3, "b": 0, "free": "x"},
        {"a": 0, "b": 0, "free": "x"},
        {"a": 2, "b": 1, "free": "x"},
    ]


def test_the_sweep_rotates_through_the_rank_zero_members() -> None:
    """One member per polish generation, first-measured first, wrapping."""

    result = run_genetic_loop(
        problem=_AllNonDominated(),
        config=GeneticConfig(population_size=4, offspring_per_generation=3,
                             generations=100, seed=3, evaluation_budget=60,
                             polish="sweep", polish_after=1))
    order = [note["member_index"] for note in _notes(result).values()
             if note["engaged"]]
    assert len(order) >= 8
    assert order == [index % 4 for index in range(len(order))]


# --------------------------------------------------------------------------
# B1 -- the reserve guard
# --------------------------------------------------------------------------


def test_the_reserve_refuses_a_sweep_once_the_budget_is_nearly_gone() -> None:
    problem = _OneMax([1] * 8)
    result = _run(problem, polish="sweep", polish_after=1, polish_reserve=0.9,
                  evaluation_budget=48)
    allowed = _assert_rule_holds(problem, result, after=1, reserve=0.9,
                                 budget=48)
    # The reserve has to actually bite somewhere, or this measures nothing.
    assert any(not permitted for permitted in allowed.values())


def test_a_reserve_of_one_never_lets_a_sweep_engage() -> None:
    """Nothing may have been spent, and evaluating the seeds already spent."""

    problem = _OneMax([1] * 8)
    result = _run(problem, polish="sweep", polish_after=1, polish_reserve=1.0)
    assert not any(note["engaged"] for note in _notes(result).values())


def test_polish_after_zero_reads_as_twice_the_population() -> None:
    """The auto threshold is a real number, not "immediately"."""

    problem = _OneMax([1] * 8)
    result = _run(problem, polish="sweep", polish_after=0, population_size=6)
    allowed = _assert_rule_holds(problem, result, after=12, reserve=0.15,
                                 budget=48)
    engaged = [g for g, note in _notes(result).items() if note["engaged"]]
    # The seed is the optimum, so nothing is exhausted before the first sweep:
    # the first engagement is exactly the first generation the rule permits.
    assert engaged and min(engaged) == min(g for g, ok in allowed.items() if ok)
    # Twelve charges, not two generations: the run bred for six before sweeping.
    assert min(engaged) > 2


# --------------------------------------------------------------------------
# B1 -- off is off
# --------------------------------------------------------------------------


def test_polish_off_is_byte_identical_to_the_absent_knob() -> None:
    """Two runs: the default, and the default said out loud. Same run."""

    absent = _OneMax([1, 0, 1, 0, 1, 1, 0, 0])
    stated = _OneMax([1, 0, 1, 0, 1, 1, 0, 0])
    first = _run(absent)
    second = _run(stated, polish="off", survival="count")

    assert absent.events == stated.events
    assert json.dumps(first.history, sort_keys=True, default=str) == \
        json.dumps(second.history, sort_keys=True, default=str)
    assert first.evaluations == second.evaluations
    # And nothing polish-shaped is recorded when nobody asked for polish.
    assert not any("polish" in entry for entry in first.history)


def test_polish_off_leaves_the_public_entry_point_untouched() -> None:
    problem_a = _OneMax([0] * 8)
    problem_b = _OneMax([0] * 8)
    first = optimize(problem_a, budget=24, seed=9)
    second = optimize(problem_b, budget=24, seed=9, polish="off",
                      survival="count")
    assert [event for event in problem_a.events if event[0] == "eval"] == \
        [event for event in problem_b.events if event[0] == "eval"]
    assert first.evaluations == second.evaluations


# --------------------------------------------------------------------------
# B2 -- crowding-distance survival
# --------------------------------------------------------------------------


def _tied(vectors: Sequence[Mapping[str, float]]):
    """A population in which the domination count separates nobody."""

    return [(dict(vector, tag=index), 0.0)
            for index, vector in enumerate(vectors)]


_DIAGONAL = [
    {"a": 0.0, "b": 0.0, "c": 0.0},
    {"a": 1.0, "b": 1.0, "c": 1.0},
    {"a": 0.1, "b": 0.1, "c": 0.1},
    {"a": 0.5, "b": 0.5, "c": 0.5},
    {"a": 0.9, "b": 0.9, "c": 0.9},
]


def test_crowding_distance_on_a_hand_computed_three_objective_set() -> None:
    """Five points on the diagonal; every objective reads the same spread.

    Per objective the sorted order is 0, 0.1, 0.5, 0.9, 1 over a span of 1, so
    the two ends are boundaries and the interior gaps are (0.5-0)=0.5,
    (0.9-0.1)=0.8 and (1-0.5)=0.5. Three identical objectives triple each.
    """

    distances = crowding_distances(_DIAGONAL)
    assert distances[0] == float("inf")
    assert distances[1] == float("inf")
    assert distances[2] == pytest.approx(1.5)
    assert distances[3] == pytest.approx(2.4)
    assert distances[4] == pytest.approx(1.5)


def test_crowding_keeps_the_boundaries_and_the_widest_gap() -> None:
    population = _tied(_DIAGONAL)
    kept = truncation_survival(
        population, keep=3, key_of=lambda config: json.dumps(config,
                                                             sort_keys=True),
        method="crowding",
        objectives_of=lambda config: {k: v for k, v in config.items()
                                      if k != "tag"})
    assert [config["tag"] for config, _value in kept] == [0, 1, 3]
    # Count cannot see any of that: with one domination count for everybody it
    # keeps whoever was measured first, which is the near-random rule the
    # five-objective unit was losing to.
    counted = truncation_survival(
        population, keep=3,
        key_of=lambda config: json.dumps(config, sort_keys=True))
    assert [config["tag"] for config, _value in counted] == [0, 1, 2]


def test_crowding_equals_count_when_the_ranking_separates_everyone() -> None:
    population = [({"tag": index, "f": float(index)}, float(index))
                  for index in range(6)]
    key_of = (lambda config: str(config["tag"]))
    objectives_of = (lambda config: {"f": config["f"]})
    assert truncation_survival(population, keep=4, key_of=key_of) == \
        truncation_survival(population, keep=4, key_of=key_of,
                            method="crowding", objectives_of=objectives_of)


def test_crowding_is_stable_where_it_also_cannot_separate() -> None:
    """Identical vectors get identical distances and keep measurement order."""

    population = _tied([{"a": 1.0, "b": 2.0}] * 4)
    kept = truncation_survival(
        population, keep=2,
        key_of=lambda config: json.dumps(config, sort_keys=True),
        method="crowding",
        objectives_of=lambda config: {"a": config["a"], "b": config["b"]})
    assert [config["tag"] for config, _value in kept] == [0, 1]


def test_crowding_without_objectives_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="objectives_of"):
        truncation_survival(_tied(_DIAGONAL), keep=2,
                            key_of=lambda config: str(config["tag"]),
                            method="crowding")


def test_an_unknown_survival_method_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="'count' or 'crowding'"):
        truncation_survival([], keep=1, key_of=str, method="elitist")


def test_the_loop_passes_survival_through_and_count_stays_identical() -> None:
    def run(**overrides):
        return run_genetic_loop(
            problem=_AllNonDominated(),
            config=GeneticConfig(population_size=5, offspring_per_generation=3,
                                 generations=12, seed=6,
                                 evaluation_budget=40, **overrides))

    default = run()
    stated = run(survival="count")
    crowded = run(survival="crowding")
    assert json.dumps(default.history, sort_keys=True, default=str) == \
        json.dumps(stated.history, sort_keys=True, default=str)
    # Where nothing dominates anything, the two rules keep different members.
    assert {json.dumps(point.configuration, sort_keys=True)
            for point in crowded.pareto_front} != \
        {json.dumps(point.configuration, sort_keys=True)
         for point in default.pareto_front}


def test_an_unknown_polish_mode_is_refused_by_the_loop() -> None:
    with pytest.raises(ValueError, match="'off' or 'sweep'"):
        run_genetic_loop(problem=_OneMax([1] * 8),
                         config=GeneticConfig(polish="buff"))


# --------------------------------------------------------------------------
# B3 -- the public entry point
# --------------------------------------------------------------------------


def test_optimize_validates_both_knobs_by_name() -> None:
    with pytest.raises(ValueError, match="polish must be 'off' or 'sweep'"):
        optimize(_OneMax([0] * 8), budget=8, polish="yes")
    with pytest.raises(ValueError, match="survival must be 'count' or 'crowding'"):
        optimize(_OneMax([0] * 8), budget=8, survival="nsga2")


def test_optimize_refuses_both_knobs_on_the_authoring_strategy() -> None:
    """A knob the run would ignore is a silent no-op, so it is refused."""

    with pytest.raises(ValueError, match="polish"):
        optimize(_Seedless([0] * 8), budget=8, polish="sweep")
    with pytest.raises(ValueError, match="survival"):
        optimize(_Seedless([0] * 8), budget=8, survival="crowding")


def test_optimize_carries_both_knobs_into_the_genetic_loop() -> None:
    problem = _OneMax([1] * 8)
    result = optimize(problem, budget=32, seed=5, polish="sweep",
                      survival="crowding")
    assert result.evaluations <= 32
    assert any("polish" in entry for entry in result.history)
