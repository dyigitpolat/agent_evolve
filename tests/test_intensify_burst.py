"""The consolidation burst: find something, then look next door.

The measured defect this sub-mode answers: both runs that ever reached the
analog venue's fully-feasible plateau then spent 3.4% and 12.4% of their
remaining charges on it -- while every one of the discovery's 44 one-step
neighbours was measured (post hoc) to be on the plateau too. The loop had no
move that holds a discovery: an intensified child resamples its unpinned loci
through the prior, elite offspring mutate through the same prior, and the
prior repels from a region it never predicted (8 of that discovery's 24
fields sat above the installed prior's lid). ``intensify_burst=N`` arms N
one-locus, prior-free probes of whichever member last advanced an objective
best, spent from intensification's OWN slots -- never extra charges.

Everything here is offline and classical. The strictest rule is inherited
from the polish and intensify files: the knob is OFF by default, and "off"
means byte-identical rather than similar.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.policies.genetic import Locus, local_probe_candidate
from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop

import random

# --------------------------------------------------------------------------
# the venue: twelve loci, eight levels, minimised on their sum
# --------------------------------------------------------------------------

_LEVEL = Literal[0, 1, 2, 3, 4, 5, 6, 7]
_WIDTH = 12


class _Vector(BaseModel):
    genome: list[_LEVEL]


class _Ladder:
    candidate_model = _Vector
    objectives = (ObjectiveSpec(name="cost", goal="min"),)

    def __init__(self, seed_genome: Sequence[int] = (4,) * _WIDTH) -> None:
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


def _run(problem: _Ladder, **overrides) -> Any:
    settings: Dict[str, Any] = dict(population_size=4,
                                    offspring_per_generation=4,
                                    generations=100, seed=7,
                                    evaluation_budget=40)
    settings.update(overrides)
    return run_genetic_loop(problem=problem, config=GeneticConfig(**settings))


# --------------------------------------------------------------------------
# off is off, and nonsense is refused
# --------------------------------------------------------------------------


def test_stating_the_knob_at_zero_is_the_same_run_as_not_stating_it() -> None:
    unstated = _Ladder()
    _run(unstated, intensify="incumbent")
    stated = _Ladder()
    _run(stated, intensify="incumbent", intensify_burst=0)
    assert stated.stream == unstated.stream


def test_a_burst_without_intensification_is_refused_loudly() -> None:
    with pytest.raises(ValueError, match="intensify_burst requires"):
        _run(_Ladder(), intensify_burst=3)


@pytest.mark.parametrize("bad", [-1, True, 1.5, "4"])
def test_the_knob_must_be_a_nonnegative_integer(bad) -> None:
    with pytest.raises(ValueError, match="intensify_burst"):
        _run(_Ladder(), intensify="incumbent", intensify_burst=bad)


# --------------------------------------------------------------------------
# the probe itself
# --------------------------------------------------------------------------


def test_a_probe_moves_exactly_one_locus_to_a_nearest_declared_value() -> None:
    anchor = {"genome": [4] * _WIDTH}
    for attempt in range(20):
        probe = local_probe_candidate(anchor, _Vector,
                                      rng=random.Random(attempt))
        moved = [index for index in range(_WIDTH)
                 if probe["genome"][index] != 4]
        assert len(moved) == 1
        # span=2: the two nearest other declared levels of 4 are 3 and 5.
        assert probe["genome"][moved[0]] in (3, 5)


def test_a_probe_at_a_domain_ceiling_steps_inward() -> None:
    # The discovery this mechanism exists for sat with fields AT the domain
    # ceiling; a probe there has only inward neighbours to offer.
    anchor = {"genome": [7] * _WIDTH}
    probe = local_probe_candidate(anchor, _Vector, rng=random.Random(0))
    moved = [index for index in range(_WIDTH)
             if probe["genome"][index] != 7]
    assert len(moved) == 1
    assert probe["genome"][moved[0]] in (6, 5)


def test_a_probe_on_a_short_genome_is_not_a_copy() -> None:
    # The pin band clamps to short genomes and degenerates to copies there
    # (disclosed for the 6-locus NAS venue); the probe does not.
    class _Short(BaseModel):
        genome: list[_LEVEL]

    anchor = {"genome": [2] * 6}
    probe = local_probe_candidate(anchor, _Short, rng=random.Random(1))
    assert sum(1 for index in range(6)
               if probe["genome"][index] != 2) == 1


def test_an_anchor_with_no_movable_locus_comes_back_unchanged() -> None:
    class _Frozen(BaseModel):
        genome: list[Literal[3]]

    anchor = {"genome": [3, 3]}
    assert local_probe_candidate(anchor, _Frozen,
                                 rng=random.Random(0)) == anchor


# --------------------------------------------------------------------------
# the loop: advances arm it, probes are one step from the advancer
# --------------------------------------------------------------------------


def _advancer_walk(stream: List[List[int]], upto: int) -> List[int]:
    """The member a burst would anchor on after the first *upto* charges."""

    best = float("inf")
    anchor: List[int] = []
    for row in stream[:upto]:
        if float(sum(row)) < best:
            best = float(sum(row))
            anchor = row
    return anchor


def test_every_probe_is_one_locus_from_the_member_that_last_advanced() -> None:
    # Exploration off and the whole generation given to intensification, with
    # the counter refilled far beyond the budget: after the first advance,
    # every offspring slot is a burst probe of the current advancer. The
    # anchor is recomputed here exactly as the loop computes it -- the last
    # strict improver, generation by generation -- so the claim is about
    # WHICH configurations were charged, not about a counter.
    problem = _Ladder()
    result = _run(problem, intensify="incumbent", intensify_fraction=1.0,
                  intensify_burst=1000)
    bursts = [entry.get("burst", 0) for entry in result.history[1:]]
    assert sum(bursts) > 0, "the burst never engaged on a descending venue"
    charged = 0
    population = 4
    charged += population                      # generation 0: the init pool
    for entry, burst_count in zip(result.history[1:], bursts):
        generation = problem.stream[charged:charged + entry["valid_count"]]
        if burst_count:
            anchor = _advancer_walk(problem.stream, charged)
            probes = sum(
                1 for row in generation
                if sum(1 for a, b in zip(row, anchor) if a != b) == 1)
            assert probes >= burst_count, (
                f"gen {entry['gen']}: {burst_count} bursts recorded, only "
                f"{probes} charged rows are one locus from the advancer")
        charged += entry["valid_count"]


def test_the_burst_is_recorded_every_generation_including_its_zeros() -> None:
    result = _run(_Ladder(), intensify="incumbent", intensify_burst=2)
    assert all("burst" in entry for entry in result.history[1:])


def test_the_burst_spends_slots_and_never_extra_charges() -> None:
    plain = _Ladder()
    _run(plain, intensify="incumbent")
    bursting = _Ladder()
    _run(bursting, intensify="incumbent", intensify_burst=4)
    assert len(bursting.stream) <= 40 and len(plain.stream) <= 40
