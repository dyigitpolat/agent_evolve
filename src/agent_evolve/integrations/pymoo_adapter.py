"""Use a pymoo problem as an ``agent_evolve`` problem: ``from_pymoo(p)``.

pymoo and this package disagree about what a candidate is. pymoo evaluates
real-valued vectors inside box bounds; this package searches typed
configurations whose every axis declares a *finite* domain, because the genetic
operators (:mod:`agent_evolve.policies.genetic`) read their allowed values from
the candidate schema and refuse to invent any. The adapter reconciles the two
by quantizing each continuous axis onto a declared grid.

The quantization rule
---------------------
Decision variable ``i`` with box bounds ``[xl_i, xu_i]`` becomes a required
field ``x{i}`` on a generated pydantic model, typed::

    Literal[g_0, ..., g_{k-1}]      k = grid_points (default 16)
    g_j = xl_i + j * (xu_i - xl_i) / (k - 1)

that is, ``k`` evenly spaced values *including both endpoints*, deduplicated
(a variable with ``xl_i == xu_i`` collapses to its single legal value). The
``Literal`` grid serializes to a JSON-schema ``enum``, which is exactly what
``locus_domain`` reads -- so every variable is a mutable locus with a visible
domain, and the uninformed random proposer draws only legal vectors. Grid
values are exact ``linspace`` floats; Python's ``repr`` round-trips them
through JSON without drift, so schema, validation and evaluation all see the
same numbers. The resolution cost is honest and stated: the adapter searches
the ``k**n_var`` lattice, not the continuum -- raise ``grid_points`` to refine.

The rest of the mapping
-----------------------
* ``objectives``  -- pymoo always minimizes, so the ``n_obj`` outputs of
  ``_evaluate`` become ``ObjectiveSpec("f0", "min") .. ObjectiveSpec(f"f{n-1}",
  "min")``.
* ``seeds()``     -- the single midpoint configuration: for each variable, the
  grid value nearest ``(xl_i + xu_i) / 2`` (exact ties resolve to the lower
  grid value, deterministically).
* ``materialize`` -- config -> the ordered tuple ``(x0, ..., x{n-1})``, which
  is the vector pymoo measures and a stable cache key.
* ``evaluate``    -- one call to the pymoo problem's own ``evaluate`` (which
  dispatches to its ``_evaluate``). Constrained problems are honored: a
  candidate violating the pymoo convention (``G <= 0``, ``H == 0``) raises
  ``ValueError`` naming the total violation, which is the contract's word for
  "infeasible in a way only measurement could reveal".

The adapter never imports pymoo. It reads the four attributes every pymoo
``Problem`` carries (``n_var``, ``n_obj``, ``xl``, ``xu``) and calls
``evaluate``, so any object with that surface works and the module imports
cleanly where pymoo is absent.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import Field, create_model

from agent_evolve.contract import Config, ObjectiveSpec, ValidationOutcome

__all__ = ["from_pymoo", "PymooProblemAdapter"]


def _box_bounds(pymoo_problem: Any) -> tuple[np.ndarray, np.ndarray, int]:
    """Read and normalize ``(xl, xu, n_var)``, explaining every refusal."""
    missing = [name for name in ("n_var", "n_obj", "xl", "xu")
               if getattr(pymoo_problem, name, None) is None]
    if missing:
        raise TypeError(
            f"{type(pymoo_problem).__name__} lacks {', '.join(missing)}. "
            "from_pymoo needs a box-bounded pymoo Problem: n_var, n_obj and "
            "finite xl/xu bounds (pass xl=/xu= to the pymoo Problem constructor)."
        )
    if getattr(pymoo_problem, "vars", None) is not None:
        raise TypeError(
            "this pymoo problem declares mixed/typed variables via 'vars'; "
            "from_pymoo supports only box-bounded real vectors (xl/xu). "
            "Wrap it as a plain Problem, or express it directly as an "
            "agent_evolve Problem, whose candidates are typed already."
        )
    n_var = int(pymoo_problem.n_var)
    xl = np.broadcast_to(np.asarray(pymoo_problem.xl, dtype=float), (n_var,))
    xu = np.broadcast_to(np.asarray(pymoo_problem.xu, dtype=float), (n_var,))
    bad = [i for i in range(n_var)
           if not (np.isfinite(xl[i]) and np.isfinite(xu[i]) and xl[i] <= xu[i])]
    if bad:
        raise ValueError(
            f"variables {bad} have no finite ordered bounds; a Literal grid "
            "needs finite xl <= xu for every variable."
        )
    return xl, xu, n_var


class PymooProblemAdapter:
    """A pymoo problem presented through the five-obligation contract.

    Built by :func:`from_pymoo`; see the module docstring for the mapping and
    the quantization rule.
    """

    def __init__(self, pymoo_problem: Any, *, grid_points: int = 16) -> None:
        if not isinstance(grid_points, int) or isinstance(grid_points, bool) \
                or grid_points < 2:
            raise ValueError(
                f"grid_points must be an integer >= 2, got {grid_points!r}: "
                "one point per axis is not a search space, it is a constant."
            )
        self._pymoo = pymoo_problem
        self.grid_points = grid_points
        xl, xu, n_var = _box_bounds(pymoo_problem)
        self._n_var = n_var
        self._n_obj = int(pymoo_problem.n_obj)

        # The documented rule: k evenly spaced values, endpoints included,
        # exact linspace floats, deduplicated in order.
        self._grids: tuple[tuple[float, ...], ...] = tuple(
            tuple(dict.fromkeys(float(v)
                                for v in np.linspace(xl[i], xu[i], grid_points)))
            for i in range(n_var)
        )

        self.candidate_model = create_model(
            f"Pymoo{type(pymoo_problem).__name__}Candidate",
            **{
                self._field(i): (
                    Literal[grid],  # tuple subscript == one Literal per value
                    Field(...,
                          description=f"decision variable {i}, quantized to "
                                      f"{len(grid)} grid values in "
                                      f"[{xl[i]:g}, {xu[i]:g}]"),
                )
                for i, grid in enumerate(self._grids)
            },
        )
        self.objectives: Sequence[ObjectiveSpec] = tuple(
            ObjectiveSpec(f"f{j}", "min") for j in range(self._n_obj)
        )

    @staticmethod
    def _field(i: int) -> str:
        return f"x{i}"

    # -- where to start ----------------------------------------------------
    def seeds(self) -> Sequence[Config]:
        """The midpoint configuration, snapped to the grid.

        For each variable the grid value nearest ``(xl + xu) / 2`` under
        float arithmetic; an exact float tie resolves to the lower value via
        ``argmin``'s first-hit rule. Either way the seed is deterministic.
        """
        seed: dict[str, float] = {}
        for i, grid in enumerate(self._grids):
            arr = np.asarray(grid)
            midpoint = (arr[0] + arr[-1]) / 2.0
            seed[self._field(i)] = float(arr[int(np.argmin(np.abs(arr - midpoint)))])
        return (seed,)

    # -- cheap rejection that explains itself ------------------------------
    def validate(self, config: Config) -> ValidationOutcome:
        for i, grid in enumerate(self._grids):
            name = self._field(i)
            if name not in config:
                return ValidationOutcome(
                    False, "structural",
                    f"missing field '{name}'; supply all of "
                    f"x0..x{self._n_var - 1}.")
            value = config[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return ValidationOutcome(
                    False, "structural",
                    f"'{name}' must be a number from its declared grid, got "
                    f"{value!r}.")
            if float(value) not in grid:
                nearest = min(grid, key=lambda g: abs(g - float(value)))
                return ValidationOutcome(
                    False, "structural",
                    f"'{name}'={value!r} is not on the declared grid; the "
                    f"nearest allowed value is {nearest!r} (grid spans "
                    f"[{grid[0]!r}, {grid[-1]!r}] in {len(grid)} steps).")
        return ValidationOutcome(True)

    # -- candidate -> the artifact that gets measured ----------------------
    def materialize(self, config: Config) -> tuple[float, ...]:
        return tuple(float(config[self._field(i)]) for i in range(self._n_var))

    # -- measure it --------------------------------------------------------
    def evaluate(self, artifact: Any) -> Mapping[str, float]:
        x = np.asarray(artifact, dtype=float)
        wants = ["F"]
        has_constraints = getattr(self._pymoo, "has_constraints", None)
        constrained = bool(has_constraints()) if callable(has_constraints) else False
        if constrained:
            wants += ["G", "H"]
        out = self._pymoo.evaluate(x, return_values_of=wants)
        values = dict(zip(wants, out if isinstance(out, (list, tuple)) else [out]))
        if constrained:
            g = np.atleast_1d(np.asarray(values.get("G"), dtype=float)).ravel() \
                if values.get("G") is not None else np.zeros(0)
            h = np.atleast_1d(np.asarray(values.get("H"), dtype=float)).ravel() \
                if values.get("H") is not None else np.zeros(0)
            violation = float(np.clip(g, 0.0, None).sum() + np.abs(h).sum())
            if violation > 0.0:
                raise ValueError(
                    f"infeasible under the pymoo constraints: total violation "
                    f"{violation:.6g} (G must be <= 0, H must be == 0). Move "
                    "toward the feasible region.")
        f = np.asarray(values["F"], dtype=float).ravel()
        if f.size != self._n_obj:
            raise ValueError(
                f"pymoo evaluate returned {f.size} objective values, but the "
                f"problem declares n_obj={self._n_obj}.")
        return {f"f{j}": float(f[j]) for j in range(self._n_obj)}

    # -- optional: one honest sentence for a model-driven proposer ---------
    def search_space_description(self) -> str:
        return (
            f"{type(self._pymoo).__name__} (pymoo): minimize "
            f"{self._n_obj} objectives f0..f{self._n_obj - 1} over "
            f"{self._n_var} decision variables x0..x{self._n_var - 1}, each "
            f"restricted to its declared grid of at most {self.grid_points} "
            "evenly spaced values between its bounds."
        )


def from_pymoo(pymoo_problem: Any, *, grid_points: int = 16) -> PymooProblemAdapter:
    """Present *pymoo_problem* as a five-obligation ``agent_evolve`` problem.

    ``optimize(from_pymoo(p), budget=..., proposer="random")`` is the whole
    swap; see the module docstring for the quantization rule and the mapping.
    *grid_points* sets the per-variable resolution (evenly spaced, endpoints
    included, default 16).
    """
    return PymooProblemAdapter(pymoo_problem, grid_points=grid_points)
