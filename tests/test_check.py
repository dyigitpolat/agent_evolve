"""``check()`` must say, before any spend, whether this budget can show anything.

The toy problems here are the suite's existing ones: ``_CountOnes`` (declared
finite domains, deterministic, counts its own evaluations) and ``Complete``
(integer fields whose schema declares no finite set -- the undeclared-domain
case). The one problem defined locally is the degenerate case the verdict
exists for: an objective that never moves.
"""

from __future__ import annotations

import dataclasses

import pytest

from test_genetic_loop import _CountOnes
from test_public_contract import Complete

from agent_evolve.policies.check import NO_HEADROOM_VERDICT, CheckReport, check


class _Flat(_CountOnes):
    """Constant objectives: nothing any optimizer could ever demonstrate."""

    def evaluate(self, artifact):
        self.calls += 1
        return {"ones": 1.0, "tail": 0.0}


# -- the report is populated, field by field --------------------------------


def test_report_fields_are_populated() -> None:
    problem = _CountOnes(length=6)
    report = check(problem, 24, probe=40, seed=0)

    assert dataclasses.is_dataclass(report) and isinstance(report, CheckReport)
    assert report.problem == "_CountOnes"
    assert report.budget == 24 and report.probe == 40

    # loci count and per-locus domain sizes, from the problem's own schema
    assert report.locus_count == 6
    assert [d.locus for d in report.loci] == [f"genome[{i}]" for i in range(6)]
    assert all(d.domain_size == 2 for d in report.loci)
    assert report.undeclared_loci == ()

    # the probe went through validate/materialize/evaluate without failures
    assert report.draws == 40 and report.evaluated == 40
    assert report.failure_rate == 0.0
    assert (report.validate_failures, report.materialize_failures,
            report.evaluate_failures) == (0, 0, 0)
    assert report.eval_seconds_median >= 0.0

    # per-objective spread, in declaration order and direction
    assert [(s.objective, s.goal) for s in report.spread] == [
        ("ones", "max"), ("tail", "min")]
    for s in report.spread:
        assert s.count == 40
        assert s.minimum <= s.mean <= s.maximum
        assert s.stdev >= 0.0

    # headroom estimate per objective, always in the improving direction
    assert [h.objective for h in report.headroom] == ["ones", "tail"]
    for h in report.headroom:
        assert h.headroom >= 0.0 and h.noise >= 0.0
        best = max if h.goal == "max" else min
        assert h.best_of_probe == best([h.best_of_probe, h.best_of_budget])

    assert isinstance(report.verdict, str) and report.verdict
    rendered = report.render()
    assert "_CountOnes" in rendered and "verdict" in rendered


def test_the_report_names_undeclared_domain_loci() -> None:
    # Complete's candidate model bounds two integers without enumerating them,
    # so neither locus declares a finite domain; the probe cannot vary them and
    # must say so rather than invent values the problem never declared.
    report = check(Complete(), 12, probe=12, seed=0)
    assert report.undeclared_loci == ("x", "y")
    assert all(not d.declared for d in report.loci)
    assert "declare no finite domain" in report.verdict


# -- the no-headroom verdict ------------------------------------------------


def test_no_headroom_verdict_triggers_on_a_constant_objective() -> None:
    report = check(_Flat(length=6), 20, probe=30, seed=1)
    assert all(s.stdev == 0.0 for s in report.spread)
    assert all(h.headroom == 0.0 and h.below_noise for h in report.headroom)
    assert NO_HEADROOM_VERDICT in report.verdict


def test_headroom_is_reported_when_the_objective_actually_varies() -> None:
    # At budget 1 the expected best of "budget random draws" is just the mean
    # draw, so a varying objective must show headroom above that noise floor.
    report = check(_CountOnes(length=8), 1, probe=60, seed=2)
    by_name = {h.objective: h for h in report.headroom}
    assert by_name["ones"].headroom > 0.0
    assert not by_name["ones"].below_noise
    assert NO_HEADROOM_VERDICT not in report.verdict


# -- the budget is respected ------------------------------------------------


def test_budget_must_be_a_positive_integer() -> None:
    for bad in (0, -3, 2.5, True):
        with pytest.raises(ValueError, match="budget"):
            check(_CountOnes(length=4), bad, probe=8, seed=0)
    with pytest.raises(ValueError, match="probe"):
        check(_CountOnes(length=4), 8, probe=0, seed=0)


def test_the_probe_bounds_the_spend_and_the_budget_is_what_is_assessed() -> None:
    # `budget` is the optimizer budget under assessment, not the probe's spend:
    # the probe costs at most `probe` evaluations no matter how large a budget
    # is being assessed, the same way `agent_evolve check` spends repeats*budget
    # to assess one budget.
    problem = _CountOnes(length=6)
    report = check(problem, 200, probe=25, seed=0)
    assert problem.calls <= 25
    assert report.draws == 25 and report.evaluated == problem.calls
    assert report.budget == 200


def test_the_stated_budget_actually_drives_the_headroom_estimate() -> None:
    # Identical probes (same seed), different budgets. A huge budget means
    # random search will reach everything the probe found, so headroom must
    # collapse below noise; a budget of 1 leaves the most on the table.
    small = check(_CountOnes(length=6), 1, probe=40, seed=3)
    large = check(_CountOnes(length=6), 4000, probe=40, seed=3)

    ones_small = {h.objective: h for h in small.headroom}["ones"]
    ones_large = {h.objective: h for h in large.headroom}["ones"]
    assert ones_small.best_of_probe == ones_large.best_of_probe, (
        "same seed and probe must observe the same draws")
    assert ones_small.headroom > ones_large.headroom
    assert all(h.below_noise for h in large.headroom)
    assert NO_HEADROOM_VERDICT in large.verdict


# -- the probe records pipeline failures instead of crashing ----------------


def test_failures_are_counted_per_stage_and_exampled() -> None:
    class _Fragile(_CountOnes):
        def validate(self, config):
            from agent_evolve.core.problem import ValidationOutcome

            if sum(config["genome"]) >= 5:
                return ValidationOutcome(
                    ok=False, failure_phase="constraint",
                    message="too many ones; at most 4 are feasible")
            return ValidationOutcome(ok=True)

        def evaluate(self, artifact):
            self.calls += 1
            if sum(artifact) == 4:
                raise ValueError("a four-one artifact cannot be measured")
            return super().evaluate(artifact)

    report = check(_Fragile(length=8), 20, probe=80, seed=4)
    assert report.validate_failures > 0
    assert report.evaluate_failures > 0
    assert report.draws == 80
    assert report.evaluated == 80 - report.validate_failures - report.evaluate_failures
    assert report.failure_rate == pytest.approx(
        (report.validate_failures + report.evaluate_failures) / 80)
    joined = " ".join(report.failure_examples)
    assert "at most 4 are feasible" in joined
    assert "cannot be measured" in joined


# -- the CLI subcommand is wired --------------------------------------------


def test_diagnose_subcommand_renders_the_report(capsys) -> None:
    from agent_evolve.cli import main

    code = main([
        "diagnose", "test_genetic_loop:_CountOnes",
        "--budget", "16", "--probe", "20", "--seed", "0",
    ])
    out = capsys.readouterr().out
    assert code == 0
    assert "problem check: _CountOnes" in out
    assert "headroom at budget 16" in out
    assert "verdict" in out
