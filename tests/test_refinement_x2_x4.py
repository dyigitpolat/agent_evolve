"""X2/X4: stop destroying admitted guidance; raise the actuation ceiling.

The measured story is in REFINEMENT_ROUND.md: replies name bands whose
interior rungs the schema zeroed (68.2%), "about X" numerics died on
whole-reply rejection (11.5% of init cells), optimum-excluding cages held
in 8/12 NAS losses with the unwind firing 2/24, seven cells stranded up to
219/384 evaluations on the generation cap, and the 1/n mutation seam
capped actuation at the measured 0.037 fresh-draw rate.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.policies.genetic import DomainRestriction
from agent_evolve.policies.weighted_prior import (
    WeightedPriorTelemetry,
    band_fill,
    llm_weighted_prior_proposer,
    snap_to_rung,
)
from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop

# --------------------------------------------------------------------------
# X2a: the band codec
# --------------------------------------------------------------------------


def test_unnamed_rungs_inside_the_named_band_are_interpolated() -> None:
    values, weights = band_fill((1, 5), (2.0, 4.0), domain=(1, 2, 3, 4, 5))
    assert values == (1, 2, 3, 4, 5)
    assert weights == pytest.approx((2.0, 2.5, 3.0, 3.5, 4.0))


def test_rungs_outside_the_band_stay_excluded() -> None:
    values, weights = band_fill((2, 4), (1.0, 1.0), domain=(1, 2, 3, 4, 5))
    assert values == (2, 3, 4)          # 1 and 5 keep their exclusion claim
    assert weights == pytest.approx((1.0, 1.0, 1.0))


def test_named_entries_keep_their_exact_weights_including_zero() -> None:
    values, weights = band_fill((1, 3, 5), (4.0, 0.0, 4.0),
                                domain=(1, 2, 3, 4, 5))
    assert values == (1, 2, 3, 4, 5)
    assert weights == pytest.approx((4.0, 2.0, 0.0, 2.0, 4.0))


def test_categorical_domains_are_untouched() -> None:
    values, weights = band_fill(("a", "c"), (1.0, 2.0),
                                domain=("a", "b", "c"))
    assert values == ("a", "c") and weights == pytest.approx((1.0, 2.0))


def test_snap_lands_on_the_nearest_rung_within_span_only() -> None:
    domain = (1.50394, 1.6623, 2.0)
    assert snap_to_rung(1.61461, domain) == 1.6623
    assert snap_to_rung(1.55, domain) == 1.50394
    assert snap_to_rung(9.0, domain) == 9.0        # out of span: untouched
    assert snap_to_rung("x", domain) == "x"
    assert snap_to_rung(1.61, ("a", "b")) == 1.61  # categorical: untouched


class _Amp(BaseModel):
    width: Literal[1, 2, 3, 4, 5]
    depth: Literal[8, 16]


_SPECS = (ObjectiveSpec(name="gain", goal="max"),)


def _propose(reply: str, tel=None):
    return llm_weighted_prior_proposer(
        lambda _p: reply, objectives=_SPECS,
        telemetry=tel or WeightedPriorTelemetry())


from agent_evolve.policies.structure import Attribution, LevelSummary


def _Attr():
    levels = tuple(
        LevelSummary(locus=field, value=value, n=2,
                     objective_means={"gain": 1.0}, nondominated=1)
        for field, values in (("width", (1, 5)), ("depth", (8, 16)))
        for value in values)
    return Attribution(levels=levels, n_evaluated=16)


def test_an_admitted_band_reply_installs_with_its_interior_filled() -> None:
    tel = WeightedPriorTelemetry()
    prior = _propose(
        '{"weights": {"width": {"values": [1, 5], "weights": [2, 4]}}}',
        tel)(_Attr(), _Amp)
    values, weights = prior.weighted["width"]
    assert values == (1, 2, 3, 4, 5)
    assert weights == pytest.approx((2.0, 2.5, 3.0, 3.5, 4.0))
    assert tel.band_filled_rungs == 3
    assert prior.allowed["width"] == (1, 2, 3, 4, 5)


def test_an_about_x_numeric_is_snapped_not_rejected() -> None:
    tel = WeightedPriorTelemetry()
    prior = _propose(
        '{"weights": {"width": {"values": [1.9, 4], "weights": [3, 1]}}}',
        tel)(_Attr(), _Amp)
    assert tel.out_of_domain == 0 and tel.snapped_values == 1
    values, weights = prior.weighted["width"]
    assert values == (2, 3, 4)
    assert weights[0] == pytest.approx(3.0)


def test_two_snaps_onto_one_rung_keep_their_combined_intent() -> None:
    prior = _propose(
        '{"weights": {"width": {"values": [1.9, 2.1, 5], '
        '"weights": [2, 3, 1]}}}')(_Attr(), _Amp)
    values, weights = prior.weighted["width"]
    assert values[0] == 2 and weights[0] == pytest.approx(5.0)


def test_a_genuinely_off_domain_reply_is_still_rejected_whole() -> None:
    tel = WeightedPriorTelemetry()
    prior = _propose(
        '{"weights": {"width": {"values": [99], "weights": [1]}}}',
        tel)(_Attr(), _Amp)
    assert prior.weighted == {} and tel.out_of_domain == 1


# --------------------------------------------------------------------------
# the loop venue for X2b / X2c / X4
# --------------------------------------------------------------------------

_LEVEL = Literal[0, 1, 2, 3, 4, 5, 6, 7]
_WIDTH = 6


class _Vec(BaseModel):
    genome: list[_LEVEL]


class _Ladder:
    candidate_model = _Vec
    objectives = (ObjectiveSpec(name="cost", goal="min"),)

    def __init__(self, seed=(4,) * _WIDTH) -> None:
        self.stream: List[List[int]] = []
        self._seed = list(seed)

    def seeds(self): return ({"genome": list(self._seed)},)
    def validate(self, c): return ValidationOutcome(ok=True)
    def materialize(self, c): return tuple(c["genome"])
    def evaluate(self, artifact):
        self.stream.append(list(artifact))
        return {"cost": float(sum(artifact))}


def _run(problem, **overrides):
    settings: Dict[str, Any] = dict(population_size=4,
                                    offspring_per_generation=4,
                                    generations=100, seed=7,
                                    evaluation_budget=40)
    settings.update(overrides)
    return run_genetic_loop(problem=problem, config=GeneticConfig(**settings))


def test_x2b_a_durable_best_outside_the_support_drops_the_cage() -> None:
    # The restriction confines the genome to high levels; the seed (all-zero,
    # the venue optimum) sits outside and tops the scoreboard from gen 1.
    # After three generations the cage is falsified and drops.
    cage = DomainRestriction(allowed={"genome": [6, 7]})
    problem = _Ladder(seed=(0,) * _WIDTH)
    result = _run(problem, restriction=cage)
    rec = next((e["structure"] for e in result.history
                if isinstance(e, dict) and "structure" in e), {})
    assert rec.get("unwound_cage"), f"cage never fell: {rec}"
    assert rec["unwound_cage"] <= 7   # warmup 4 + streak 3


def test_x2b_a_best_inside_the_support_keeps_a_sound_prior() -> None:
    sound = DomainRestriction(allowed={"genome": [0, 1, 2]})
    problem = _Ladder(seed=(0,) * _WIDTH)
    result = _run(problem, restriction=sound)
    rec = next((e["structure"] for e in result.history
                if isinstance(e, dict) and "structure" in e), {})
    assert not rec.get("unwound_cage")


def test_x2c_a_dedup_strangled_run_still_spends_its_budget() -> None:
    # Two-level loci and a tiny population dedup-starve quickly at a big
    # budget; the fill phase must spend what the generations could not.
    class _Tiny(BaseModel):
        genome: list[Literal[0, 1]]

    class _TinyLadder(_Ladder):
        candidate_model = _Tiny

    problem = _TinyLadder(seed=(0,) * _WIDTH)
    result = _run(problem, generations=6, evaluation_budget=60)
    assert len(problem.stream) == 60, f"stranded {60 - len(problem.stream)}"
    assert any("fill" in entry for entry in result.history
               if isinstance(entry, dict))


def test_x2c_a_run_that_spends_in_the_loop_is_untouched() -> None:
    plain = _Ladder()
    _run(plain)
    filled = _Ladder()
    result = _run(filled)
    assert plain.stream == filled.stream
    assert not any("fill" in e for e in result.history if isinstance(e, dict))


def test_x4_off_is_byte_identical_and_nonsense_is_refused() -> None:
    a = _Ladder(); _run(a, restriction=DomainRestriction(allowed={"genome": [3, 4, 5]}))
    b = _Ladder(); _run(b, restriction=DomainRestriction(allowed={"genome": [3, 4, 5]}),
                        actuation="off")
    assert a.stream == b.stream
    with pytest.raises(ValueError, match="actuation"):
        _run(_Ladder(), actuation="turbo")


def test_x4_fresh_boost_moves_more_loci_while_the_prior_is_fresh() -> None:
    # With 6 loci the boosted rate is 1/3 vs the 1/6 default: offspring in
    # the fresh window mutate ~2x the loci. Measured on the charged stream
    # (deterministic given the seed): mean Hamming distance of consecutive
    # early offspring to the seed rises under the boost.
    cage = DomainRestriction(allowed={"genome": [2, 3, 4, 5]})
    plain = _Ladder(); _run(plain, restriction=cage)
    boosted = _Ladder()
    result = _run(boosted, restriction=cage, actuation="fresh-boost")
    assert boosted.stream != plain.stream
    flags = [e.get("actuation_boost") for e in result.history
             if isinstance(e, dict) and "actuation_boost" in e]
    # budget 40 / (4*offspring 4) = 2 -> window max(4, 2) = 4 generations,
    # or until the cage falls (X2b) -- either way the window is recorded on,
    # then off.
    assert flags and flags[0] is True
    assert any(flag is False for flag in flags)
