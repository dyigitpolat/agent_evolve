"""The generation cap must never be what stops a run before its budget does.

`generations` is documented as a CAP -- duplicate offspring are served from
the evaluation cache without spending budget, so the loop's real stop
condition is the budget and the cap only bounds a run that has stopped making
progress. That sentence stopped being true above the sealed budget ceiling:
`_genetic_sizing` grows the population with the budget, the old cap DIVIDED by
the offspring count, and so the cap shrank as the budget rose. At budget 2000
it bought 133 generations of 60 offspring, and a run whose offspring
frequently repeat what the population already holds ran out of generations
with a fifth of its budget unspent.

Measured on the cheap tier before the fix: the operator-portfolio arm spent a
mean 1615.1 evaluations of 2000 while its comparator spent 2000.0 on 12 of 12
cells -- so the matched-budget comparison was settled by what each arm could
SPEND. The reproduction below is offline and provider-free, and it reads 1233
and 1240 of 2000 against the old expression.

Two claims here, and as in `test_budget_sizing` they pull against each other:
the sealed range may not move by one generation, and above it the budget has
to be spendable.
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

# --- the arithmetic ----------------------------------------------------------


def test_every_sealed_budget_keeps_the_cap_the_seals_were_measured_with():
    """The literal old expression, at every budget the sealed rows used."""

    for budget in range(0, _SEALED_BUDGET_CEILING + 1):
        _pop, offspring = _genetic_sizing(budget)
        assert _generation_cap(budget, offspring) == max(
            1, 4 * budget // max(1, offspring)), (
            f"budget {budget} is at or below the sealed ceiling and no longer "
            "runs the generation cap the seals were measured with")


def test_the_guard_is_inert_until_the_population_actually_grows():
    """The two branches agree wherever the offspring count is ten or fewer.

    The guard is on the budget, as `_genetic_sizing`'s is, but what it guards
    is the DIVISOR -- so the fix is inert past its own ceiling, all the way to
    the first budget that buys a thirteenth member. Pinning that boundary here
    is what lets a campaign at B in (384, 415] be pooled with a sealed one.
    """

    for budget in range(0, 416):
        _pop, offspring = _genetic_sizing(budget)
        assert _generation_cap(budget, offspring) == max(
            1, 4 * budget // max(1, offspring)), budget
    _pop, offspring = _genetic_sizing(416)
    assert _generation_cap(416, offspring) == 166 != 4 * 416 // offspring


@pytest.mark.parametrize("budget,expected", [(2000, 800), (10000, 4000)])
def test_the_cap_stops_shrinking_as_the_population_grows(budget, expected):
    _pop, offspring = _genetic_sizing(budget)
    assert _generation_cap(budget, offspring) == expected


def test_the_cap_can_always_pay_for_the_whole_budget():
    """The property the numbers above sample: cap * offspring >= budget.

    A cap below `budget / offspring` cannot spend the budget even with zero
    duplicates, which is the shape of the defect. Four times that is the
    headroom the sealed rows were measured with, and it is kept.
    """

    for budget in list(range(1, 400)) + [416, 500, 1000, 2000, 5000, 10000]:
        _pop, offspring = _genetic_sizing(budget)
        assert _generation_cap(budget, offspring) * offspring >= budget, budget


# --- the run --------------------------------------------------------------

N_ITEMS = 14
VALUES = tuple(range(1, N_ITEMS + 1))
WEIGHTS = tuple((7 * i + 11) % 23 + 5 for i in range(N_ITEMS))
CAPACITY = sum(WEIGHTS) // 2


class _Candidate(BaseModel):
    take: list[Literal[0, 1]]


class _Knapsack:
    """A cheap two-objective knapsack: the shape the defect was measured on.

    Narrow enough that a converged population re-proposes what it already
    holds -- which is the condition under which the cap binds, and the reason
    a wide random space would not reproduce anything.
    """

    candidate_model = _Candidate
    objectives = (ObjectiveSpec(name="value", goal="max"),
                  ObjectiveSpec(name="weight", goal="min"))

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ({"take": [0] * N_ITEMS},)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return tuple(config["take"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        value = sum(a * b for a, b in zip(artifact, VALUES))
        weight = sum(a * b for a, b in zip(artifact, WEIGHTS))
        return {"value": float(value if weight <= CAPACITY else 0),
                "weight": float(weight)}


@pytest.mark.parametrize("authorship", ["off", "operators"])
def test_a_two_thousand_evaluation_run_spends_its_budget(authorship: str):
    """Both cheap arms reach their budget, portfolio included.

    `authorship="operators"` is the credential-free portfolio -- the classical
    arm plus the rule `segment` arm under UCB credit -- which is the arm the
    defect was measured on and which makes no model call, so this is the
    treatment arm's mechanism without the treatment arm's price.
    """

    result = optimize(_Knapsack(), budget=2000, seed=20375042,
                      authorship=authorship)
    assert result.evaluations >= 1950, (
        f"{authorship!r} spent {result.evaluations} of 2000; the generation "
        "cap is deciding this run instead of the budget")


def test_the_old_expression_is_what_stranded_the_budget(monkeypatch):
    """The defect, reproduced: the same run under the pre-fix arithmetic.

    Moving the sealed ceiling out of reach restores the old expression at
    every budget without editing it back in, so this reads the defect rather
    than a re-implementation of it.
    """

    import agent_evolve.api as api

    monkeypatch.setattr(api, "_SEALED_BUDGET_CEILING", 10 ** 9)
    stranded = optimize(_Knapsack(), budget=2000, seed=20375042,
                        authorship="operators")
    assert stranded.evaluations < 1500, (
        "the pre-fix expression no longer strands the budget on this venue, "
        "so this file is no longer measuring the defect it names")
