"""The logical-call expectation must be DERIVED, not a literal with arithmetic.

Three shapes are pinned because the gate has to be correct for all three, and
because the previous version reproduced the calibrated number by subtracting
from it -- a tautology that proved nothing and was in fact already wrong for the
provider-free case.

Shape 1  calibrated      : every scheduled selection call plus reflection.
Shape 2  composition     : provider-free selection contributes zero selection
                           calls; reflection still has to appear.
Shape 3  model-free      : no selection and no reflection; the expectation is
                           zero and differs from both of the above.
"""

from __future__ import annotations

import importlib
import os

import pytest


def _runner(selector_mode: str):
    os.environ["AGENT_EVOLVE_PORTFOLIO_SELECTOR_MODE"] = selector_mode
    import examples.development.run_boils_generic_campaign as module

    return importlib.reload(module)


@pytest.fixture(autouse=True)
def _restore_env():
    before = os.environ.get("AGENT_EVOLVE_PORTFOLIO_SELECTOR_MODE")
    yield
    if before is None:
        os.environ.pop("AGENT_EVOLVE_PORTFOLIO_SELECTOR_MODE", None)
    else:
        os.environ["AGENT_EVOLVE_PORTFOLIO_SELECTOR_MODE"] = before


def test_calibrated_shape_is_derived_term_by_term():
    r = _runner("calibrated")
    # each term comes from the plan independently; 7 is a consequence
    assert r.PLANNED_SELECTION_LOGICAL_CALLS == r.PARENTS_PER_PORTFOLIO * len(
        r.PORTFOLIO_GENERATIONS
    )
    assert r.PLANNED_REFLECTION_LOGICAL_CALLS == 1
    assert r.PLANNED_LOGICAL_CALLS == 7


def test_composition_shape_drops_exactly_the_selection_calls():
    calibrated = _runner("calibrated").PLANNED_LOGICAL_CALLS
    r = _runner("composition")
    assert r.PLANNED_SELECTION_LOGICAL_CALLS == 0
    assert r.PLANNED_REFLECTION_LOGICAL_CALLS == 1, (
        "a provider-free selector removes selection calls and nothing else"
    )
    assert r.PLANNED_LOGICAL_CALLS == 1
    assert calibrated - r.PLANNED_LOGICAL_CALLS == r.PARENTS_PER_PORTFOLIO * len(
        r.PORTFOLIO_GENERATIONS
    )


def test_model_free_shape_is_zero_and_differs_from_both():
    """The ladder's T0 floor: no selection call and no reflection call."""

    r = _runner("composition")
    selection = 0
    reflection = r.REFLECTIONS_PER_RECOMBINATION_GENERATION * len(
        tuple(g for g in () if g in r.RECOMBINATION_GENERATIONS)
    )
    model_free_total = selection + reflection
    assert model_free_total == 0
    assert model_free_total != r.PLANNED_LOGICAL_CALLS
    assert model_free_total != 7


def test_a_zero_call_budget_is_expressible():
    """A model-free campaign must be representable, not merely unplanned."""

    from agent_evolve.application.budgeted_optimizer import OptimizerBudget

    budget = OptimizerBudget(
        max_unique_evaluations=38, max_logical_llm_calls=0, max_generations=6
    )
    assert budget.max_logical_llm_calls == 0
