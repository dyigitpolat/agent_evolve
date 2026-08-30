"""X3 the incumbent-ball walk; X1 the joint-candidate channel.

Both answer trace-measured gaps (REFINEMENT_ROUND.md): nothing in the loop
ever localized (0.9% of evaluations near our own champion vs the
specialist's 45%), and live replies were already naming 4-8 joint configs
that the product-of-marginals schema shredded (reproduction probability
~1e-17). Standing rules: off is byte-identical; nonsense is refused;
telemetry records measured zeros.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.policies.genetic import ball_candidate
from agent_evolve.policies.weighted_prior import (
    WeightedPriorTelemetry,
    WeightedRestriction,
    llm_weighted_prior_proposer,
)
from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop

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


# --------------------------------------------------------------------------
# X3: the ball
# --------------------------------------------------------------------------


def test_a_ball_draw_steps_each_ordered_locus_within_the_radius() -> None:
    anchor = {"genome": [4] * _WIDTH}
    for trial in range(30):
        out = ball_candidate(anchor, _Vec, rng=random.Random(trial), radius=2)
        assert all(abs(v - 4) <= 2 for v in out["genome"])


def test_a_ball_at_the_domain_edge_clamps_instead_of_wrapping() -> None:
    anchor = {"genome": [0] * _WIDTH}
    for trial in range(30):
        out = ball_candidate(anchor, _Vec, rng=random.Random(trial), radius=3)
        assert all(0 <= v <= 3 for v in out["genome"])


def test_walk_off_is_byte_identical_and_nonsense_is_refused() -> None:
    a = _Ladder(); _run(a)
    b = _Ladder(); _run(b, walk="off")
    assert a.stream == b.stream
    with pytest.raises(ValueError, match="walk"):
        _run(_Ladder(), walk="stroll")
    with pytest.raises(ValueError, match="walk_fraction"):
        _run(_Ladder(), walk="ball", walk_fraction=1.5)


def test_the_walk_takes_slots_and_localizes_the_stream() -> None:
    plain = _Ladder()
    _run(plain)
    walked = _Ladder()
    result = _run(walked, walk="ball")
    walks = [e["walk"] for e in result.history
             if isinstance(e, dict) and "walk" in e]
    assert walks and sum(walks) > 0

    def near_champion_share(problem):
        best = None; near = 0; n = 0
        for row in problem.stream:
            if best is not None:
                n += 1
                near += max(abs(a - b) for a, b in zip(row, best)) <= 1
            if best is None or sum(row) < sum(best):
                best = row
        return near / max(1, n)

    assert near_champion_share(walked) > near_champion_share(plain)


# --------------------------------------------------------------------------
# X1: the joint pool
# --------------------------------------------------------------------------

class _Amp(BaseModel):
    width: Literal[1, 2, 3, 4, 5]
    depth: Literal[8, 16]


_SPECS = (ObjectiveSpec(name="gain", goal="max"),)

from agent_evolve.policies.structure import Attribution, LevelSummary


def _attr():
    levels = tuple(
        LevelSummary(locus=field, value=value, n=2,
                     objective_means={"gain": 1.0}, nondominated=1)
        for field, values in (("width", (1, 5)), ("depth", (8, 16)))
        for value in values)
    return Attribution(levels=levels, n_evaluated=16)


def _propose(reply: str, tel=None, **kwargs):
    return llm_weighted_prior_proposer(
        lambda _p: reply, objectives=_SPECS,
        telemetry=tel or WeightedPriorTelemetry(), **kwargs)


def test_joints_zero_keeps_the_prompt_and_ignores_a_candidates_block() -> None:
    tel = WeightedPriorTelemetry()
    prior = _propose(
        '{"weights": {"width": {"values": [1, 2], "weights": [1, 2]}}, '
        '"candidates": [{"config": {"width": 3}, "weight": 1}]}',
        tel)(_attr(), _Amp)
    assert prior.candidates == ()
    assert tel.joint_accepted == 0


def test_joints_invites_and_admits_weighted_fragments() -> None:
    tel = WeightedPriorTelemetry()
    prompts: List[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return ('{"weights": {"width": {"values": [1, 2], "weights": [1, 2]}},'
                ' "candidates": ['
                '{"config": {"width": 3, "depth": 16}, "weight": 2},'
                '{"config": {"width": 4.9}, "weight": 1},'
                '{"config": {"bogus": 1}, "weight": 1},'
                '{"config": {"width": 99}, "weight": 1}]}')

    proposer = llm_weighted_prior_proposer(
        complete, objectives=_SPECS, telemetry=tel, joints=4)
    prior = proposer(_attr(), _Amp)
    assert "candidates" in prompts[0]          # the invitation is in the prompt
    assert tel.joint_accepted == 2 and tel.joint_rejected == 2
    assert ({"width": 3, "depth": 16}, 2.0) in prior.candidates
    assert ({"width": 5}, 1.0) in prior.candidates   # 4.9 snapped to 5


def test_joint_share_draws_the_pool_and_records_it() -> None:
    pool = WeightedRestriction(
        {"genome": (tuple(range(8)), tuple(1.0 for _ in range(8)))},
        candidates=(({"genome": [0] * _WIDTH}, 5.0),))
    problem = _Ladder(seed=(7,) * _WIDTH)
    result = _run(problem, restriction=pool, joint_share=0.5)
    joints = [e["joint"] for e in result.history
              if isinstance(e, dict) and "joint" in e]
    assert sum(joints) >= 1
    assert [0] * _WIDTH in problem.stream      # the named joint was charged


def test_joint_share_zero_never_touches_a_populated_pool() -> None:
    pool = WeightedRestriction(
        {"genome": (tuple(range(8)), tuple(1.0 for _ in range(8)))},
        candidates=(({"genome": [0] * _WIDTH}, 5.0),))
    plain = _Ladder(seed=(7,) * _WIDTH)
    _run(plain, restriction=WeightedRestriction(
        {"genome": (tuple(range(8)), tuple(1.0 for _ in range(8)))}))
    pooled = _Ladder(seed=(7,) * _WIDTH)
    _run(pooled, restriction=pool)
    assert plain.stream == pooled.stream


def test_joint_share_without_an_invitation_is_refused_at_the_api() -> None:
    from agent_evolve.api import optimize
    with pytest.raises(ValueError, match="joint_share"):
        optimize(_Ladder(), budget=8, joint_share=0.5)
