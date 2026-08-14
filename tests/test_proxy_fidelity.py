"""The cheap fidelity may inform the search. It may never pay for it.

Every test here defends one sentence: **a proxy evaluation is not an
evaluation.** The budget a claim is denominated in counts full-fidelity
evaluator calls, and a mechanism that could quietly move work into a cheaper
column would make every evaluation-efficiency number in the program
unreadable. The separation is therefore pinned structurally (the object graph
cannot reach the cache), numerically (identical charged counts, run against
run) and in the report (a field of its own on the telemetry).
"""

from __future__ import annotations

import json

import pytest

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.surrogate import ORDERING_GATE
from agent_evolve.session.evaluate import EvaluationCache
from agent_evolve.session.fidelity import (
    ProxyLedger,
    ProxySource,
    proxy_fidelity_builder,
)
from agent_evolve.session.genetic_loop import GeneticConfig, run_genetic_loop
from agent_evolve.session.screening import Screening
from agent_evolve.policies.surrogate import additive_surrogate, knn_surrogate


# ------------------------------------------------------------------ a problem
class _Additive:
    """A cheap, exactly-solvable problem with a CHEAP FIDELITY of its own.

    ``evaluate`` is the truth. ``evaluate_proxy`` is a rounded, deliberately
    lossy version of it -- correlated, never equal -- which is what a real
    cheap fidelity looks like (fewer mapper iterations, a coarser mesh).
    """

    def __init__(self, n_loci: int = 6, proxy_noise: float = 0.0) -> None:
        from pydantic import create_model
        from typing import Literal

        fields = {f"x{i}": (Literal[0, 1, 2, 3], ...) for i in range(n_loci)}
        self.candidate_model = create_model("Cand", **fields)
        self.n_loci = n_loci
        self.proxy_noise = proxy_noise
        self.objectives = [ObjectiveSpec("cost", "min"),
                           ObjectiveSpec("mass", "min")]
        self.real_calls = 0
        self.proxy_calls = 0

    def seeds(self):
        return ({f"x{i}": 0 for i in range(self.n_loci)},)

    def _cost(self, config):
        return float(sum((i + 1) * config[f"x{i}"] for i in range(self.n_loci)))

    def _mass(self, config):
        return float(sum((self.n_loci - i) * config[f"x{i}"]
                         for i in range(self.n_loci)))

    def evaluate(self, config):
        self.real_calls += 1
        return {"cost": self._cost(config), "mass": self._mass(config)}

    def evaluate_proxy(self, config):
        self.proxy_calls += 1
        blur = self.proxy_noise
        return {"cost": round(self._cost(config) * (1.0 + blur), 3),
                "mass": round(self._mass(config) * (1.0 - blur), 3)}


class _Rugged(_Additive):
    """Non-additive truth, so no additive/knn rule is exact and the CHEAP
    fidelity has something to win on. Its proxy is the same landscape with a
    coarse rounding -- correlated, never equal."""

    def _cost(self, config):
        values = [config[f"x{i}"] for i in range(self.n_loci)]
        return float(sum((i + 1) * v * v for i, v in enumerate(values))
                     + 3.0 * values[0] * values[-1]
                     + 5.0 * values[1] * values[2])

    def _mass(self, config):
        values = [config[f"x{i}"] for i in range(self.n_loci)]
        return float(sum((self.n_loci - i) * (v ** 3) % 17 for i, v in
                         enumerate(values)) + 4.0 * values[2] * values[3])

    def evaluate_proxy(self, config):
        self.proxy_calls += 1
        return {"cost": round(self._cost(config) * 1.03, 1),
                "mass": round(self._mass(config) * 0.97, 1)}


class _NoProxy(_Additive):
    evaluate_proxy = None                      # the attribute is not callable


def _distinct(problem, count, seed=0):
    """`count` DISTINCT configurations -- a test whose candidates collide is
    testing de-duplication, not the thing it says it is."""

    import random as _random

    rng = _random.Random(seed)
    seen, out = set(), []
    while len(out) < count:
        config = {f"x{i}": rng.randrange(4) for i in range(problem.n_loci)}
        token = json.dumps(config, sort_keys=True)
        if token in seen:
            continue
        seen.add(token)
        out.append(config)
    return out


def _screening():
    return Screening(builders=(("additive", "rule", additive_surrogate),
                               ("knn", "rule", knn_surrogate)),
                     gate=ORDERING_GATE)


def _run(problem, *, budget, mode, seed=11, ceiling=None):
    config = GeneticConfig(
        population_size=6, offspring_per_generation=6, generations=6,
        seed=seed, evaluation_budget=budget,
        evaluation_cache=EvaluationCache(),
        screening=_screening(), proxy_fidelity=mode, proxy_ceiling=ceiling)
    return run_genetic_loop(problem=problem, config=config)


# ============================================================ the invariant
@pytest.mark.parametrize("mode", ["rows", "screen", "both"])
def test_a_proxy_evaluation_is_never_charged_to_the_evaluation_budget(mode):
    """THE invariant. Same seed, same budget, one factor: the cheap fidelity.

    The charged count must be identical to the run that had no proxy at all,
    and the proxy count must be reported separately and non-zero -- i.e. the
    cheap fidelity really did run, and really did not land in the budget.
    """

    budget = 24
    without = _Additive()
    baseline = _run(without, budget=budget, mode="off")
    withproxy = _Additive()
    measured = _run(withproxy, budget=budget, mode=mode)

    # the charged ledger is exactly the full-fidelity evaluator calls, and
    # never one more, in both arms
    assert baseline.telemetry.real_evaluations == without.real_calls <= budget
    assert measured.telemetry.real_evaluations == withproxy.real_calls <= budget
    # the cheap fidelity ran, and landed in its OWN column
    assert baseline.telemetry.proxy_evaluations == 0
    assert without.proxy_calls == 0
    assert measured.telemetry.proxy_evaluations > 0
    assert measured.telemetry.proxy_evaluations == withproxy.proxy_calls
    assert measured.evaluations == measured.telemetry.real_evaluations


def test_the_cheap_fidelity_never_reaches_the_archive_or_the_reported_front():
    """A proxy that LIES must not change one reported number.

    Its objective values are 100x the truth, so if a single cheap row reached
    the archive, the front and every objective vector would show it.
    """

    class _Liar(_Additive):
        def evaluate_proxy(self, config):
            self.proxy_calls += 1
            return {"cost": 100.0 * self._cost(config) + 7.0,
                    "mass": 100.0 * self._mass(config) + 7.0}

    problem = _Liar()
    result = _run(problem, budget=24, mode="both")
    assert problem.proxy_calls > 0
    for record in result.evaluations_detail if hasattr(
            result, "evaluations_detail") else []:      # pragma: no cover
        pass
    truth = {(json.dumps(c.configuration, sort_keys=True)): dict(c.objectives)
             for c in result.pareto_front}
    for config_json, objectives in truth.items():
        config = json.loads(config_json)
        assert objectives["cost"] == problem._cost(config)
        assert objectives["mass"] == problem._mass(config)


def test_the_proxy_source_cannot_reach_the_problem_or_the_cache():
    """Structural, not conventional: the object graph has no route to a charge."""

    problem = _Additive()
    source = ProxySource.for_problem(problem)
    reachable = list(vars(source).values())
    assert problem not in reachable
    assert not any(isinstance(value, EvaluationCache) for value in reachable)
    # the one callable it holds is the problem's cheap fidelity, nothing else
    assert source._evaluate == problem.evaluate_proxy
    assert not any(getattr(value, "__name__", "") == "evaluate"
                   for value in reachable)


def test_a_problem_without_a_cheap_fidelity_gets_no_proxy_and_no_error():
    problem = _NoProxy()
    assert ProxySource.for_problem(problem) is None
    result = _run(problem, budget=18, mode="both")
    assert result.telemetry.proxy_evaluations == 0


def test_proxy_off_is_byte_identical_to_a_run_with_no_cheap_fidelity():
    """"off" is the default and must be a no-op, or every ablation is dirty."""

    a = _run(_Additive(), budget=24, mode="off", seed=5)
    b = _run(_NoProxy(), budget=24, mode="off", seed=5)
    assert [dict(c.objectives) for c in a.pareto_front] == \
           [dict(c.objectives) for c in b.pareto_front]
    assert a.telemetry.real_evaluations == b.telemetry.real_evaluations


# ================================================================ the ceiling
def test_the_ceiling_bounds_the_run_and_degrades_to_no_proxy():
    problem = _Additive()
    result = _run(problem, budget=24, mode="both", ceiling=5)
    assert problem.proxy_calls <= 5
    assert result.telemetry.proxy_evaluations <= 5
    assert result.telemetry.real_evaluations <= 24


def test_a_failing_cheap_fidelity_is_counted_and_never_raises():
    class _Broken(_Additive):
        def evaluate_proxy(self, config):
            self.proxy_calls += 1
            raise RuntimeError("the cheap simulator fell over")

    problem = _Broken()
    result = _run(problem, budget=18, mode="both")
    assert result.telemetry.real_evaluations == 18
    counters = {m.mechanism: m.counters for m in result.telemetry.mechanisms}
    assert counters["proxy_fidelity"]["proxy_failures"] > 0


def test_a_cheap_fidelity_that_returns_a_malformed_vector_is_refused():
    class _Malformed(_Additive):
        def evaluate_proxy(self, config):
            self.proxy_calls += 1
            return {"cost": float("nan"), "mass": 1.0}

    source = ProxySource.for_problem(_Malformed())
    assert source.evaluate({"x0": 1}) is None
    assert source.ledger.failures == 1


# ================================================== rows: real supersedes cheap
def test_a_real_measurement_supersedes_the_cheap_row_for_the_same_candidate():
    problem = _Additive(proxy_noise=0.5)
    source = ProxySource.for_problem(problem)
    screen = _screening()
    screen.attach_proxy(source, mode="rows")
    candidates = _distinct(problem, 10, seed=1)
    screen.prime(candidates, [])
    assert source.ledger.evaluations == 10

    # every candidate now measured for real -> no cheap row may survive
    evaluated = [(c, problem.evaluate(c)) for c in candidates]
    screen.refresh(evaluated, problem.objectives, seed=3)
    assert screen.telemetry.proxy_rows_used == 0

    # measure only half: the other half's cheap rows remain the only evidence
    screen2 = _screening()
    screen2.attach_proxy(ProxySource.for_problem(problem), mode="rows")
    screen2.prime(candidates, [])
    screen2.refresh(evaluated[:5], problem.objectives, seed=3)
    assert screen2.telemetry.proxy_rows_used == 5


def test_priming_skips_candidates_the_campaign_already_measured():
    problem = _Additive()
    source = ProxySource.for_problem(problem)
    screen = _screening()
    screen.attach_proxy(source, mode="rows")
    candidates = _distinct(problem, 8, seed=2)
    already = [source.key(c) for c in candidates[:6]]
    added = screen.prime(candidates, already)
    assert added == 2
    assert source.ledger.evaluations == 2


def test_cheap_rows_let_the_gate_reach_a_verdict_where_the_budget_cannot():
    """The measured blockage this seam exists for: n < min_rows.

    With three real rows the ordering gate cannot look at all (its floor is
    eight pooled points). With cheap rows for the same generation's
    candidates it can -- and what it then decides is still the gate's
    decision, taken on cross-validated evidence.
    """

    problem = _Additive()
    specs = problem.objectives
    candidates = _distinct(problem, 16, seed=3)
    evaluated = [(c, problem.evaluate(c)) for c in candidates[:3]]

    bare = _screening()
    assert bare.refresh(evaluated, specs, seed=1) is False
    assert bare.telemetry.rejected_insufficient_rows > 0

    primed = _screening()
    primed.attach_proxy(ProxySource.for_problem(problem), mode="rows")
    primed.prime(candidates[3:], [])
    assert primed.refresh(evaluated, specs, seed=1) is True
    assert primed.telemetry.rejected_insufficient_rows == 0
    assert primed.telemetry.proxy_rows_used == 13


# ================================================ screen: the proxy as builder
def test_the_cheap_fidelity_competes_as_a_surrogate_and_must_pass_the_gate():
    problem = _Rugged()                        # correlated, not identical
    specs = problem.objectives
    candidates = _distinct(problem, 20, seed=4)
    evaluated = [(c, problem.evaluate(c)) for c in candidates]

    screen = _screening()
    screen.attach_proxy(ProxySource.for_problem(problem), mode="screen")
    assert screen.builders[0][1] == "proxy"
    assert screen.refresh(evaluated, specs, seed=2) is True
    assert screen.telemetry.chosen_proxy == 1


def test_an_ANTI_CORRELATED_cheap_fidelity_is_rejected_by_the_gate():
    """The safety property: cheapness is not a licence to order the search."""

    class _Backwards(_Additive):
        def evaluate_proxy(self, config):
            self.proxy_calls += 1
            return {"cost": -self._cost(config), "mass": -self._mass(config)}

    problem = _Backwards()
    specs = problem.objectives
    candidates = _distinct(problem, 20, seed=4)
    evaluated = [(c, problem.evaluate(c)) for c in candidates]
    screen = Screening(builders=(), gate=ORDERING_GATE)
    screen.attach_proxy(ProxySource.for_problem(problem), mode="screen")
    assert screen.refresh(evaluated, specs, seed=2) is False
    assert screen.telemetry.rejected_rank > 0


def test_the_proxy_builder_screens_nothing_when_the_cheap_fidelity_refuses():
    class _Refuses(_Additive):
        def evaluate_proxy(self, config):
            raise RuntimeError("no")

    source = ProxySource.for_problem(_Refuses())
    predict = proxy_fidelity_builder(source)([], [])
    assert predict([{"x0": 1}]) is None


# ==================================================================== ledger
def test_the_ledger_separates_calls_from_repeats():
    problem = _Additive()
    source = ProxySource.for_problem(problem)
    config = {f"x{i}": 1 for i in range(problem.n_loci)}
    source.evaluate(config)
    source.evaluate(config)
    source.evaluate(config)
    assert source.ledger.evaluations == 1
    assert source.ledger.cache_hits == 2
    assert problem.proxy_calls == 1
    assert source.ledger.as_dict()["proxy_evaluations"] == 1


def test_the_ledger_reaches_the_result_as_its_own_mechanism():
    result = _run(_Additive(), budget=24, mode="both")
    names = [m.mechanism for m in result.telemetry.mechanisms]
    assert "proxy_fidelity" in names
    counters = {m.mechanism: m.counters for m in result.telemetry.mechanisms}
    assert counters["proxy_fidelity"]["proxy_evaluations"] == \
        result.telemetry.proxy_evaluations


def test_an_unknown_proxy_mode_is_refused_rather_than_ignored():
    screen = _screening()
    with pytest.raises(ValueError):
        screen.attach_proxy(ProxySource.for_problem(_Additive()), mode="maybe")


def test_a_negative_ceiling_is_refused():
    with pytest.raises(ValueError):
        ProxySource(lambda c: {}, [ObjectiveSpec("a", "min")], ceiling=-1)


def test_ledger_dict_is_all_integers():
    assert all(isinstance(v, int) for v in ProxyLedger().as_dict().values())


def test_one_refused_candidate_does_not_blind_the_screen_to_the_rest():
    """A pool member the cheap fidelity refuses is ranked LAST, not fatal."""

    class _Picky(_Additive):
        def evaluate_proxy(self, config):
            self.proxy_calls += 1
            if config["x0"] == 3:
                raise RuntimeError("the cheap simulator refuses this one")
            return {"cost": self._cost(config), "mass": self._mass(config)}

    problem = _Picky()
    source = ProxySource.for_problem(problem)
    predict = proxy_fidelity_builder(source)([], problem.objectives)
    pool = [{f"x{i}": 0 for i in range(problem.n_loci)},
            {f"x{i}": 1 for i in range(problem.n_loci)},
            dict({f"x{i}": 0 for i in range(problem.n_loci)}, x0=3)]
    predicted = predict(pool)
    assert predicted is not None and len(predicted) == 3
    # the refused one is worse than every candidate the proxy could measure
    assert predicted[2]["cost"] > max(predicted[0]["cost"], predicted[1]["cost"])
    assert predicted[2]["mass"] > max(predicted[0]["mass"], predicted[1]["mass"])


def test_a_pool_the_cheap_fidelity_refuses_entirely_screens_nothing():
    class _Refuses(_Additive):
        def evaluate_proxy(self, config):
            raise RuntimeError("no")

    source = ProxySource.for_problem(_Refuses())
    predict = proxy_fidelity_builder(source)([], _Refuses().objectives)
    assert predict([{"x0": 1}, {"x0": 2}]) is None
