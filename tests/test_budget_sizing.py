"""How many members the genetic default runs with, at every budget.

Two claims, and the file exists because they pull against each other.

Below the sealed ceiling the sizing is a CONTROL ARM. Every row this package
has sealed was measured at a budget of 384 or less, so those numbers may not
move -- not by a member, not by an off-by-one at a boundary. The table below
pins them as literals rather than as a formula, because a formula that is
"obviously" equivalent is exactly how a control arm moves.

Above it the old cap was measured to be a THROTTLE: twelve members converge
long before a four-figure budget is gone, the offspring that follow are
recombinations the population already holds, they are served from the
evaluation cache, and the run ends with the budget unspent -- 969 to 1212 of
2000 charges, six of six cheap-tier cells, against 1696 to 1842 for the
uniform comparator it was being compared to. The comparison was decided by
what each arm could SPEND. So above 384 the population grows with the budget,
and the second half of the table pins that curve.

The two halves meet without a step: 384 and 385 size identically, so the guard
is a change of rule and not a discontinuity a campaign could straddle.

Everything here is arithmetic on a number. Nothing runs a model.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.api import (
    _SEALED_BUDGET_CEILING,
    _generation_cap,
    _genetic_sizing,
)
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome

#: budget -> (population, offspring per generation).
#:
#: The first five rows are the pre-change expression's own output, at the
#: budgets the sealed rows were measured at. The last three are the new curve:
#: one member per 32 charges, floored at the old cap of twelve and ceilinged at
#: 64.
SIZING = {
    16: (4, 2),
    40: (10, 8),
    96: (12, 10),
    192: (12, 10),
    384: (12, 10),
    500: (15, 13),
    2000: (62, 60),
    10000: (64, 62),
}


@pytest.mark.parametrize("budget", sorted(SIZING))
def test_the_sizing_table_is_what_the_budget_buys(budget: int) -> None:
    assert _genetic_sizing(budget) == SIZING[budget], (
        f"budget {budget} no longer sizes to {SIZING[budget]}. At or below "
        f"{_SEALED_BUDGET_CEILING} that is a moved control arm and every "
        "sealed row measured against it; above it, re-pin the curve here in "
        "the same commit that changes it."
    )


def test_every_sealed_budget_runs_the_expression_the_seals_were_measured_with():
    """Not a table lookup: the old expression, evaluated, at every budget <= 384.

    The table above pins eight points. This pins the whole guarded range
    against the literal arithmetic the sealed rows were measured with, so a
    rewrite that happens to agree at the eight sampled budgets and disagrees at
    a ninth cannot pass.
    """

    for budget in range(0, _SEALED_BUDGET_CEILING + 1):
        pop = max(4, min(budget // 4, 12))
        assert _genetic_sizing(budget) == (pop, max(2, pop - 2)), (
            f"budget {budget} is at or below the sealed ceiling and no longer "
            "runs the sealed expression"
        )


def test_the_guard_changes_the_rule_without_stepping_the_size():
    """384 and 385 size the same, so no campaign straddles a jump."""

    assert (_genetic_sizing(_SEALED_BUDGET_CEILING)
            == _genetic_sizing(_SEALED_BUDGET_CEILING + 1) == (12, 10))


def test_the_population_grows_with_the_budget_and_stops_at_the_ceiling():
    """The property the table samples: monotone, capped, offspring in step."""

    sizes = [_genetic_sizing(budget)
             for budget in range(_SEALED_BUDGET_CEILING, 4001, 16)]
    assert all(b[0] >= a[0] for a, b in zip(sizes, sizes[1:])), (
        "the population is not monotone in the budget"
    )
    assert all(offspring == pop - 2 for pop, offspring in sizes[1:]), (
        "offspring stopped following the population above the guard"
    )
    assert max(pop for pop, _o in sizes) == 64 == _genetic_sizing(10 ** 6)[0], (
        "the population cap moved; per-generation selection cost is what that "
        "number is paying for"
    )


# --- the number the loop is actually built with -----------------------------

TEMPLATE = {"genome": [0] * 12}


class _Candidate(BaseModel):
    genome: list[Literal[0, 1]]


class _Problem:
    candidate_model = _Candidate
    objectives = (ObjectiveSpec(name="ones", goal="max"),)

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return (dict(TEMPLATE, genome=list(TEMPLATE["genome"])),)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return tuple(config["genome"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        return {"ones": float(sum(artifact))}


class _Captured(Exception):
    """The loop's config, delivered by the shortest path that cannot run it."""

    def __init__(self, config: Any) -> None:
        super().__init__("captured")
        self.config = config


#: budget -> the generation cap the loop is built with. Re-pinned 2026-08-25
#: with the defect fix in `_generation_cap`: at 384 (and everywhere at or
#: below it, and on up to 415 where the population is still twelve) this is
#: the literal old expression; at 2000 the old expression gave 133, which was
#: a THIRD of what the budget needs and stranded a fifth of it. The full
#: argument, the measurement and the guard live in `_generation_cap`'s
#: docstring and in `test_generation_cap.py`.
GENERATION_CAP = {384: 153, 2000: 800}


@pytest.mark.parametrize("budget,expected", [(384, (12, 10)), (2000, (62, 60))])
def test_optimize_builds_the_loop_with_the_sizing_the_table_states(
    monkeypatch, budget: int, expected) -> None:
    """The table is only worth pinning if `optimize` is what consults it."""

    from agent_evolve.session import genetic_loop

    def _capture(*, problem, config, **kwargs):
        raise _Captured(config)

    monkeypatch.setattr(genetic_loop, "run_genetic_loop", _capture)
    with pytest.raises(_Captured) as caught:
        optimize(_Problem(), budget=budget, seed=7)
    config = caught.value.config
    assert (config.population_size, config.offspring_per_generation) == expected
    assert config.evaluation_budget == budget
    # The generation cap is a cap, and a cap that binds before the budget does
    # is a defect. Pinned as a literal here for the same reason the sizing is:
    # a formula that is "obviously" equivalent is how a control arm moves.
    assert config.generations == GENERATION_CAP[budget]
    assert config.generations == _generation_cap(budget, expected[1])
    if budget <= _SEALED_BUDGET_CEILING:
        assert config.generations == max(1, 4 * budget // expected[1]), (
            "a sealed budget's generation cap moved")
