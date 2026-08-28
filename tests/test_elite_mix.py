"""The elite mixture: discoveries circulate; the prior keeps its exclusions.

The measured story is on :class:`EliteMixture`. The rules these tests are
strictest about, in the package's standing order: 0.0 constructs nothing and
is byte-identical; the mixture only ever rides a prior (no restriction, no
change); ``narrow`` is delegated so an exclusion is never reopened; and a
locus the front holds no opinion on returns the base's own answer INCLUDING
``None``, so an unweighted draw stays on the unweighted RNG stream.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.policies.genetic import (
    DomainRestriction,
    EliteMixture,
    Locus,
)
from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop

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


class _Weighted:
    """A graded prior that concentrates every locus's mass on level 6."""

    def narrow(self, locus: Locus, domain: tuple) -> tuple:
        return domain

    def weights_for(self, locus: Locus, values: Sequence[Any]):
        return tuple(100.0 if v == 6 else 1.0 for v in values)


def _run(problem: _Ladder, **overrides) -> Any:
    settings: Dict[str, Any] = dict(population_size=4,
                                    offspring_per_generation=4,
                                    generations=100, seed=7,
                                    evaluation_budget=40)
    settings.update(overrides)
    return run_genetic_loop(problem=problem, config=GeneticConfig(**settings))


# --------------------------------------------------------------------------
# off is off; nonsense is refused; no prior means no mixture
# --------------------------------------------------------------------------


def test_stating_zero_is_the_same_run_as_not_stating_it() -> None:
    unstated = _Ladder()
    _run(unstated, restriction=_Weighted())
    stated = _Ladder()
    _run(stated, restriction=_Weighted(), elite_mix=0.0)
    assert stated.stream == unstated.stream


@pytest.mark.parametrize("bad", [-0.1, 1.0, 2, "0.1"])
def test_the_weight_must_be_a_mixing_weight(bad) -> None:
    with pytest.raises(ValueError, match="elite_mix"):
        _run(_Ladder(), elite_mix=bad)


def test_with_no_prior_in_force_the_mixture_rides_nothing() -> None:
    plain = _Ladder()
    _run(plain)
    mixed = _Ladder()
    result = _run(mixed, elite_mix=0.3)
    assert mixed.stream == plain.stream
    # And the telemetry says so rather than staying silent: the key is
    # present with the measured zero.
    assert all(entry.get("elite_mix_opined", 0) == 0
               for entry in result.history)


# --------------------------------------------------------------------------
# the mixture arithmetic, one call at a time
# --------------------------------------------------------------------------


def _front(*genomes: Sequence[int]):
    return tuple({"genome": list(g)} for g in genomes)


def test_no_front_opinion_returns_the_bases_answer_including_none() -> None:
    mix = EliteMixture(0.5)
    mix.over(DomainRestriction())          # weights_for -> None
    mix.front_rows = _front()
    assert mix.weights_for(Locus("genome", 0), (0, 1, 2)) is None
    assert mix.opined == 0


def test_the_front_marginal_mixes_against_a_uniform_base() -> None:
    mix = EliteMixture(0.5)
    mix.over(DomainRestriction())
    mix.front_rows = _front([2] * _WIDTH, [2] * _WIDTH, [3] * _WIDTH)
    weights = mix.weights_for(Locus("genome", 0), (2, 3, 4))
    # base uniform (1/3 each); front marginal (2/3, 1/3, 0); eps = 0.5
    assert weights == pytest.approx((0.5 / 3 + 0.5 * 2 / 3,
                                     0.5 / 3 + 0.5 / 3,
                                     0.5 / 3))
    assert mix.opined == 1


def test_the_front_marginal_mixes_against_graded_base_weights() -> None:
    mix = EliteMixture(0.2)
    mix.over(_Weighted())                  # 100 on level 6, 1 elsewhere
    mix.front_rows = _front([0] * _WIDTH)
    weights = mix.weights_for(Locus("genome", 3), (0, 6))
    base = (1.0 / 101.0, 100.0 / 101.0)
    assert weights == pytest.approx((0.8 * base[0] + 0.2,
                                     0.8 * base[1]))


def test_a_row_that_does_not_carry_the_locus_contributes_nothing() -> None:
    mix = EliteMixture(0.5)
    mix.over(DomainRestriction())
    mix.front_rows = ({"other": 1}, {"genome": [5] * _WIDTH})
    weights = mix.weights_for(Locus("genome", 0), (4, 5))
    assert weights == pytest.approx((0.25, 0.75))


def test_narrow_is_delegated_so_an_exclusion_is_never_reopened() -> None:
    base = DomainRestriction(allowed={"genome": [1, 2]})
    mix = EliteMixture(0.5)
    mix.over(base)
    mix.front_rows = _front([7] * _WIDTH)  # the front sits OUTSIDE the box
    assert mix.narrow(Locus("genome", 0), (1, 2, 7)) == (1, 2)


def test_riding_none_is_none() -> None:
    assert EliteMixture(0.5).over(None) is None


# --------------------------------------------------------------------------
# the loop: front values circulate through the two seams
# --------------------------------------------------------------------------


def test_the_mutation_seam_draws_toward_the_front_under_the_mixture() -> None:
    # The seam-level fact, measured exactly: mutating the SAME child through
    # the prior alone pulls toward the prior's mass (level 6); through the
    # mixture, the front's disagreeing values (level 2) take a share of the
    # draws close to epsilon. Deterministic given the RNG seeds; the venue's
    # own selection pressure never enters.
    from agent_evolve.policies.genetic import mutate

    kid = {"genome": [4] * _WIDTH}
    prior = _Weighted()
    mix = EliteMixture(0.4)
    mix.over(prior)
    mix.front_rows = _front([2] * _WIDTH)

    def moved_to(restriction, want, runs=400):
        hits = draws = 0
        for i in range(runs):
            out = mutate(kid, _Vector, rate=1.0 / _WIDTH,
                         restriction=restriction, rng=random.Random(i))
            for j in range(_WIDTH):
                if out["genome"][j] != 4:
                    draws += 1
                    hits += out["genome"][j] == want
        return hits / draws

    assert moved_to(prior, want=2) < 0.05          # repelled
    share = moved_to(mix, want=2)
    assert 0.25 < share < 0.55                     # ~epsilon, front circulates


def test_the_loop_engages_the_mixture_and_says_so() -> None:
    plain = _Ladder(seed_genome=(1,) * _WIDTH)
    _run(plain, restriction=_Weighted())
    mixed = _Ladder(seed_genome=(1,) * _WIDTH)
    result = _run(mixed, restriction=_Weighted(), elite_mix=0.4)
    assert mixed.stream != plain.stream
    assert result.history[-1].get("elite_mix_opined", 0) > 0


def test_the_final_result_is_not_worse_on_the_venue_the_prior_misleads() -> None:
    def best(problem: _Ladder) -> float:
        return min(sum(row) for row in problem.stream)

    plain = _Ladder(seed_genome=(1,) * _WIDTH)
    _run(plain, restriction=_Weighted())
    mixed = _Ladder(seed_genome=(1,) * _WIDTH)
    _run(mixed, restriction=_Weighted(), elite_mix=0.4)
    assert best(mixed) <= best(plain)
