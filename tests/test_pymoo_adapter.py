"""``from_pymoo``: a pymoo problem satisfies the five-obligation contract.

What is worth testing here is the *seam*: the quantization rule actually
declares grids inside the pymoo bounds, those grids are visible to the genetic
operators through the ordinary schema path (``loci_of``/``locus_domain``, no
side channel), the objectives carry pymoo's always-minimize convention, and
``optimize`` runs the adapted problem credential-free within its budget.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymoo")

import numpy as np
from pymoo.problems import get_problem

from agent_evolve.api import optimize
from agent_evolve.contract import Problem, as_problem
from agent_evolve.integrations.pymoo_adapter import from_pymoo
from agent_evolve.policies.genetic import loci_of, locus_domain


def _domains(problem):
    """Per-variable domains exactly as the genetic operators would read them."""
    seed = problem.seeds()[0]
    return {
        str(locus): locus_domain(problem.candidate_model, locus)
        for locus in loci_of(seed)
    }


# -- the quantization rule ---------------------------------------------------

def test_grid_respects_bounds_and_hits_endpoints():
    # zdt1 with asymmetric variable count keeps the test honest about ordering.
    pymoo_problem = get_problem("zdt1", n_var=3)
    problem = from_pymoo(pymoo_problem, grid_points=5)

    domains = _domains(problem)
    assert set(domains) == {"x0", "x1", "x2"}
    for i, grid in enumerate(domains.values()):
        lo, hi = float(pymoo_problem.xl[i]), float(pymoo_problem.xu[i])
        assert len(grid) == 5
        assert grid[0] == lo and grid[-1] == hi          # endpoints included
        assert all(lo <= v <= hi for v in grid)          # never outside bounds
        assert grid == tuple(sorted(grid))               # evenly ordered
        steps = np.diff(np.asarray(grid))
        assert np.allclose(steps, steps[0])              # evenly spaced


def test_grid_respects_nonunit_bounds():
    # bnh has genuinely different per-variable bounds: x0 in [0,5], x1 in [0,3].
    pymoo_problem = get_problem("bnh")
    domains = _domains(from_pymoo(pymoo_problem, grid_points=6))
    for i, grid in enumerate(domains.values()):
        assert grid[0] == float(pymoo_problem.xl[i])
        assert grid[-1] == float(pymoo_problem.xu[i])
        assert len(grid) == 6


def test_grid_points_below_two_is_refused():
    with pytest.raises(ValueError, match="grid_points"):
        from_pymoo(get_problem("zdt1", n_var=2), grid_points=1)


# -- objectives: named f0..f{n-1}, all minimized -----------------------------

def test_objectives_are_named_and_minimized():
    problem = from_pymoo(get_problem("zdt1", n_var=4))
    assert [spec.name for spec in problem.objectives] == ["f0", "f1"]
    assert all(spec.goal == "min" for spec in problem.objectives)
    assert isinstance(problem, Problem)      # structural: all five obligations
    as_problem(problem)                      # and the contract's own gate


# -- loci: visible to the genetic operators through the ordinary path --------

def test_loci_and_domains_visible_to_genetic_operators():
    problem = from_pymoo(get_problem("zdt1", n_var=4), grid_points=8)
    seed = problem.seeds()[0]
    loci = loci_of(seed)
    assert len(loci) == 4                    # one heritable locus per variable
    for locus in loci:
        domain = locus_domain(problem.candidate_model, locus)
        assert len(domain) == 8              # every locus has a declared domain
        assert seed[locus.field] in domain   # the seed is drawn from it


# -- seeds: the midpoint configuration, snapped to the grid ------------------

def test_seed_is_the_grid_midpoint():
    # Odd grid: the exact midpoint is on the grid.
    problem = from_pymoo(get_problem("zdt1", n_var=3), grid_points=5)
    (seed,) = problem.seeds()
    assert seed == {"x0": 0.5, "x1": 0.5, "x2": 0.5}
    assert problem.validate(seed).ok
    # Even grid: the midpoint is off-grid, so the seed snaps to the nearest
    # grid value (in float arithmetic; the notional 1/3-vs-2/3 tie is not an
    # exact float tie) and must still validate.
    problem_even = from_pymoo(get_problem("zdt1", n_var=2), grid_points=4)
    (seed_even,) = problem_even.seeds()
    assert problem_even.validate(seed_even).ok
    grid = _domains(problem_even)["x0"]
    assert abs(seed_even["x0"] - 0.5) == min(abs(g - 0.5) for g in grid)


# -- validate: off-grid rejected with a usable message -----------------------

def test_validate_rejects_off_grid_and_names_the_nearest_value():
    problem = from_pymoo(get_problem("zdt1", n_var=2), grid_points=5)
    outcome = problem.validate({"x0": 0.6, "x1": 0.5})
    assert not outcome.ok
    assert "0.5" in outcome.message          # the nearest allowed value, named
    missing = problem.validate({"x0": 0.5})
    assert not missing.ok and "x1" in missing.message


# -- evaluate: pymoo's own numbers, constraint-honest ------------------------

def test_evaluate_matches_pymoo_directly():
    pymoo_problem = get_problem("zdt1", n_var=4)
    problem = from_pymoo(pymoo_problem, grid_points=5)
    artifact = problem.materialize(problem.seeds()[0])
    assert artifact == (0.5, 0.5, 0.5, 0.5)
    values = problem.evaluate(artifact)
    expected = pymoo_problem.evaluate(np.asarray(artifact), return_values_of=["F"])
    assert values["f0"] == pytest.approx(float(expected[0]))
    assert values["f1"] == pytest.approx(float(expected[1]))


def test_constrained_problem_raises_on_violation_only():
    problem = from_pymoo(get_problem("bnh"), grid_points=7)
    feasible = problem.evaluate(problem.materialize(problem.seeds()[0]))
    assert set(feasible) == {"f0", "f1"}
    with pytest.raises(ValueError, match="violation"):
        problem.evaluate((0.0, 3.0))         # (x0-5)^2 + x1^2 = 34 > 25


# -- the whole point: optimize() runs it, credential-free, within budget -----

def test_optimize_runs_the_adapted_problem_within_budget():
    result = optimize(
        from_pymoo(get_problem("zdt1", n_var=4), grid_points=8),
        budget=20,
        proposer="random",
        seed=0,
    )
    assert 0 < result.evaluations <= 20
    assert set(result.best.objectives) == {"f0", "f1"}
    domains = _domains(from_pymoo(get_problem("zdt1", n_var=4), grid_points=8))
    for name, value in result.best.configuration.items():
        assert value in domains[name]        # the search never left the grid
