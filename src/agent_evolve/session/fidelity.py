"""A problem's optional CHEAPER evaluation fidelity, and its separate ledger.

Some evaluators expose a knob that buys a correlated answer for a fraction of
the cost -- fewer mapper iterations, a coarser mesh, a shorter simulation, a
subsample of the workload. A campaign whose budget is denominated in REAL
evaluations can spend that cheaper mode on evidence the surrogate needs and
still be honest about what it paid, but only if the two are never added up.

This module is the seam, and its whole design is the separation:

* :class:`ProxySource` holds the problem's ``evaluate_proxy`` bound method and
  a key function, **and nothing else**. It has no reference to the problem,
  the evaluation cache, or the budget, so "a proxy evaluation cannot become a
  charged evaluation" is a property of the object graph rather than a promise
  in a docstring -- the same construction that makes the screen unable to
  spend budget (see :mod:`agent_evolve.session.screening`).
* Every proxy call is counted in :class:`ProxyLedger`, reported beside the
  charged count and never folded into it.
* A hard ``ceiling`` bounds what a run may spend at the cheap fidelity, so a
  consumer that asks for a proxy value in a loop degrades to "no proxy" rather
  than to an unbounded bill.

A problem opts in by defining::

    def evaluate_proxy(self, config) -> Mapping[str, float]: ...

returning EXACTLY the declared objectives, at a cheaper and correlated
fidelity. Optionally it may describe that fidelity for the record::

    proxy_fidelity_name: str            # e.g. "timeloop@50-mappings"
    proxy_cost_ratio: float             # measured seconds(proxy)/seconds(real)

Nothing here decides whether a proxy is GOOD. That is the gate's job: a proxy
enters the screen as evidence or as a candidate ordering and is
cross-validated against the run's own REAL measurements like anything else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from agent_evolve.core.problem import ObjectiveSpec, normalize_objective_values

__all__ = ["ProxyLedger", "ProxySource", "proxy_fidelity_builder"]

Config = Dict[str, Any]


def _canonical_key(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, default=str)


@dataclass
class ProxyLedger:
    """What the cheap fidelity was asked for, and what it cost. Never charged.

    ``evaluations`` counts calls that actually ran the cheap evaluator;
    ``cache_hits`` counts repeats served from memory, because a consumer that
    re-asks for the same candidate must not be reported as having spent
    anything twice. ``refused_ceiling`` is the count of requests declined
    because the run had spent its proxy allowance.
    """

    evaluations: int = 0
    cache_hits: int = 0
    failures: int = 0
    refused_ceiling: int = 0
    rows_offered: int = 0
    rows_used: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "proxy_evaluations": self.evaluations,
            "proxy_cache_hits": self.cache_hits,
            "proxy_failures": self.failures,
            "proxy_refused_ceiling": self.refused_ceiling,
            "proxy_rows_offered": self.rows_offered,
            "proxy_rows_used": self.rows_used,
        }


class ProxySource:
    """The cheap fidelity, memoized, counted, and bounded.

    Construct with :meth:`for_problem`, which returns ``None`` when the
    problem exposes no cheaper fidelity -- so every consumer's "off" path is
    the absence of an object rather than a flag it might forget to read.
    """

    def __init__(
        self,
        evaluate: Callable[[Config], Mapping[str, float]],
        objectives: Sequence[ObjectiveSpec],
        *,
        key: Callable[[Config], str] = _canonical_key,
        name: str = "proxy",
        cost_ratio: Optional[float] = None,
        ceiling: Optional[int] = None,
    ) -> None:
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")
        if ceiling is not None and ceiling < 0:
            raise ValueError(f"ceiling must not be negative, got {ceiling}")
        # DELIBERATELY only these: a bound method, a key function, the
        # objective specs, and counters. No problem, no cache, no budget.
        self._evaluate = evaluate
        self._key = key
        self._specs = list(objectives)
        self.name = str(name)
        self.cost_ratio = None if cost_ratio is None else float(cost_ratio)
        self.ceiling = ceiling
        self.ledger = ProxyLedger()
        self._memo: Dict[str, Optional[Dict[str, float]]] = {}
        #: the harvest contract (`core.telemetry`), so the cheap fidelity's
        #: ledger reaches the SearchResult like any other mechanism's counters
        self.telemetry = self.ledger
        self.mechanism = "proxy_fidelity"
        self.authored_by = "none"

    # -- construction ------------------------------------------------------
    @classmethod
    def for_problem(cls, problem: Any, *, ceiling: Optional[int] = None
                    ) -> Optional["ProxySource"]:
        """The problem's cheap fidelity, or ``None`` if it does not have one."""

        evaluate = getattr(problem, "evaluate_proxy", None)
        if not callable(evaluate):
            return None
        key = getattr(problem, "candidate_key", None)
        return cls(
            evaluate,
            list(problem.objectives),
            key=key if callable(key) else _canonical_key,
            name=str(getattr(problem, "proxy_fidelity_name", "proxy")),
            cost_ratio=getattr(problem, "proxy_cost_ratio", None),
            ceiling=ceiling,
        )

    # -- use ---------------------------------------------------------------
    def key(self, config: Config) -> str:
        return self._key(config)

    def evaluate(self, config: Config) -> Optional[Dict[str, float]]:
        """The cheap objectives for *config*, or ``None``.

        ``None`` means "this run has no cheap answer for this candidate" --
        the evaluator refused, returned something the objective contract does
        not accept, or the proxy allowance is spent. Every consumer treats
        that as "no proxy", never as a value.
        """

        token = self._key(config)
        if token in self._memo:
            self.ledger.cache_hits += 1
            return self._memo[token]
        if self.ceiling is not None and self.ledger.evaluations >= self.ceiling:
            self.ledger.refused_ceiling += 1
            return None
        self.ledger.evaluations += 1
        try:
            values = normalize_objective_values(self._evaluate(config), self._specs)
        except Exception:
            self.ledger.failures += 1
            self._memo[token] = None
            return None
        self._memo[token] = values
        return values

    def rows(
        self, candidates: Sequence[Config], *, exclude: Sequence[str] = ()
    ) -> list[tuple[Config, Dict[str, float]]]:
        """``(config, cheap objectives)`` for each candidate that yields one.

        ``exclude`` names candidate keys that already have a REAL measurement.
        A real row always supersedes a cheap one: the point of the seam is to
        add evidence where there is none, never to dilute evidence there is.
        """

        skip = set(exclude)
        out = []
        for candidate in candidates:
            if self._key(candidate) in skip:
                continue
            self.ledger.rows_offered += 1
            values = self.evaluate(candidate)
            if values is not None:
                out.append((dict(candidate), values))
        return out


def proxy_fidelity_builder(source: ProxySource) -> Callable[..., Any]:
    """The cheap fidelity itself, as a :data:`SurrogateBuilder`.

    A surrogate is anything that ORDERS candidates without charging the
    budget, and a correlated cheap evaluation is exactly that. Presenting it
    as a builder means it earns its place the same way an authored artifact
    does: the screen's gate cross-validates it against the run's own REAL
    measurements, per objective, and admits it only if it ranks them. Nothing
    about this asserts that a cheap fidelity ranks well -- it makes the claim
    checkable in the loop that would rely on it.

    The builder ignores the training rows because there is nothing to fit.
    """

    def build(rows: Any, specs: Any) -> Callable[[Sequence[Config]], Any]:
        def predict(configs: Sequence[Config]) -> Any:
            out = []
            for config in configs:
                values = source.evaluate(config)
                if values is None:
                    return None            # no cheap answer -> screen nothing
                out.append(values)
            return out

        return predict

    return build
