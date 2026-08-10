"""Virtual pre-screening: free predictions, honestly gated, budget untouchable.

Three properties carry the mechanism. The surrogate must EARN the right to
order candidates, every generation, on held-out data (the gate). The screen
must have no route to the budget (structural: it never sees the problem).
And the ledger must report virtual and real evaluations separately, so a
budget claim can never quietly count the free ones.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Sequence

import pytest
from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.policies.surrogate import (
    additive_surrogate,
    knn_surrogate,
    validate_surrogate,
)
from agent_evolve.session.screening import Screening, screen_offspring

SPECS = [ObjectiveSpec("ones", "max"), ObjectiveSpec("zeros", "min")]


def _truth(genome) -> Dict[str, float]:
    ones = float(sum(genome))
    return {"ones": ones, "zeros": float(len(genome)) - ones}


def _data(rows: Sequence[Sequence[int]]):
    return [({"genome": list(row)}, _truth(row)) for row in rows]


LEARNABLE = _data([
    [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1], [1, 1, 1, 0, 0, 1], [1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1], [1, 0, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1],
])

NOISE = [({"genome": [i % 2] * 6}, {"ones": float((i * 7919) % 13),
                                    "zeros": float((i * 104729) % 11)})
         for i in range(12)]


# --- the rule surrogates -----------------------------------------------------

def test_the_additive_surrogate_learns_an_additive_landscape():
    predict = additive_surrogate(LEARNABLE, SPECS)
    [estimate] = predict([{"genome": [1, 1, 0, 0, 1, 1]}])
    assert abs(estimate["ones"] - 4.0) < 1.0
    assert abs(estimate["zeros"] - 2.0) < 1.0


def test_the_knn_surrogate_averages_its_neighbours():
    predict = knn_surrogate(LEARNABLE, SPECS, k=1)
    [estimate] = predict([{"genome": [0, 0, 0, 0, 0, 0]}])
    assert estimate == {"ones": 0.0, "zeros": 6.0}


# --- the gate ----------------------------------------------------------------

def test_validation_passes_a_learnable_landscape_and_fails_noise():
    good = validate_surrogate(additive_surrogate, LEARNABLE, SPECS, seed=1)
    assert good.passed, good
    bad = validate_surrogate(additive_surrogate, NOISE, SPECS, seed=1)
    assert not bad.passed


def test_validation_refuses_to_judge_on_too_little_data():
    verdict = validate_surrogate(additive_surrogate, LEARNABLE[:5], SPECS)
    assert not verdict.passed and "8" in verdict.detail


def test_validation_fails_a_surrogate_that_is_wrong_on_one_objective():
    # Right about 'ones', wrong about 'zeros': ordering the pool by half the
    # problem must not pass as ordering it by all of it.
    def half_right(evaluated, specs):
        full = additive_surrogate(evaluated, specs)

        def predict(pool):
            out = full(pool)
            if out is None:
                return None
            return [{"ones": row["ones"], "zeros": 6.0 - row["zeros"]}
                    for row in out]

        return predict

    verdict = validate_surrogate(half_right, LEARNABLE, SPECS, seed=2)
    assert not verdict.passed
    assert verdict.per_objective_mse["ones"] < verdict.baseline_mse["ones"], (
        "the test problem drifted: 'ones' should still be predicted well"
    )


def test_a_monotone_but_mse_poor_surrogate_still_passes_the_rank_term():
    # Rank fidelity and MSE fidelity are different axes. A surrogate that
    # predicts 3*truth - 5 orders every pool perfectly (Spearman 1.0) while
    # its MSE is far worse than the train-mean baseline: the rank term must
    # not be the reason it fails.
    def monotone_but_scaled(evaluated, specs):
        def predict(pool):
            return [{"ones": 3.0 * sum(cfg["genome"]) - 5.0,
                     "zeros": 3.0 * (6.0 - sum(cfg["genome"])) - 5.0}
                    for cfg in pool]

        return predict

    verdict = validate_surrogate(monotone_but_scaled, LEARNABLE, SPECS, seed=1)
    assert verdict.per_objective_spearman["ones"] == pytest.approx(1.0)
    assert verdict.per_objective_spearman["zeros"] == pytest.approx(1.0)
    assert not verdict.passed, "MSE-poor by construction: the error gate holds"
    assert any(verdict.per_objective_mse[k] >= verdict.baseline_mse[k]
               for k in ("ones", "zeros")), (
        "the test problem drifted: 3x-5 should lose to the baseline on MSE"
    )
    assert "rank" not in verdict.detail, (
        "a perfectly-ranking surrogate must never be blamed on rank agreement"
    )


def test_an_anti_ranking_surrogate_with_decent_mse_is_rejected():
    # The terra failure mode (study-2 trace read): on 8/14 losing seeds the
    # authored artifact PASSED the MSE-split gate and still hurt the
    # endpoint, because low error is not faithful ordering. Here the train
    # rows cluster high (mean 5 ones) while the holdout rows sit low (0-2
    # ones), so a range-compressed predictor centered on the holdout beats
    # the train-mean baseline on every objective -- while ordering the
    # candidates exactly backwards. The old gate admitted this; the rank
    # term must reject it.
    def ones_row(count):
        return [1] * count + [0] * (6 - count)

    # seed=0 puts indices {1, 8, 9} in the holdout split of 12 rows.
    rows = [ones_row(c) for c in (4, 0, 5, 6, 4, 5, 6, 4, 1, 2, 5, 6)]
    skewed = _data(rows)

    def anti_ranker(evaluated, specs):
        def predict(pool):
            return [{"ones": 2.0 - 0.2 * sum(cfg["genome"]),
                     "zeros": 4.0 + 0.2 * sum(cfg["genome"])}
                    for cfg in pool]

        return predict

    verdict = validate_surrogate(anti_ranker, skewed, SPECS, seed=0)
    assert all(verdict.per_objective_mse[k] < verdict.baseline_mse[k]
               for k in ("ones", "zeros")), (
        "the test problem drifted: the MSE gate alone must admit this artifact"
    )
    assert verdict.per_objective_spearman["ones"] == pytest.approx(-1.0)
    assert verdict.per_objective_spearman["zeros"] == pytest.approx(-1.0)
    assert not verdict.passed, (
        "an artifact that inverts the candidate ordering must not screen, "
        "however small its MSE"
    )
    assert "rank" in verdict.detail
    disabled = validate_surrogate(anti_ranker, skewed, SPECS, seed=0,
                                  min_rank_correlation=-1.0)
    assert disabled.passed, (
        "with the rank term disabled the MSE gate admits it: the rejection "
        "above is the rank term's doing, nothing else's"
    )


# --- the screen --------------------------------------------------------------

def test_screen_orders_by_predicted_domination():
    predict = additive_surrogate(LEARNABLE, SPECS)
    pool = [{"genome": [0, 0, 0, 0, 0, 1]}, {"genome": [1, 1, 1, 1, 1, 1]}]
    report = screen_offspring(pool, [], SPECS, predict)
    assert report is not None
    assert report.order[0] == 1, "the dominant prediction must lead"
    assert report.virtual_evaluations == 2


def test_screen_returns_none_when_the_predictor_fails():
    assert screen_offspring(
        [{"genome": [0] * 6}], [], SPECS, lambda pool: None) is None


def test_screening_refresh_arbitrates_per_generation():
    screening = Screening(
        builders=(("additive", "rule", additive_surrogate),))
    assert screening.refresh(LEARNABLE, SPECS, seed=3)
    assert screening.telemetry.validated == 1
    assert not screening.refresh(NOISE, SPECS, seed=3)
    assert screening.telemetry.rejected_validation >= 1
    assert screening.screen([{"genome": [0] * 6}], [], SPECS) is None, (
        "a screen whose surrogate failed today's gate must screen nothing"
    )


def test_the_variance_guard_rejects_a_sometimes_lucky_builder():
    # The ladder1 E2 failure mode: an artifact that predicts well on one
    # split and garbage on another passed the old single-split gate half the
    # time and then mis-screened mid-run. Under the all-splits gate it must
    # be rejected -- surviving EVERY split of today's data is the bar.
    calls = {"n": 0}

    def lucky_sometimes(evaluated, specs):
        calls["n"] += 1
        good = additive_surrogate(evaluated, specs)
        if calls["n"] % 2 == 1:
            return good

        def garbage(pool):
            rows = good(pool)
            if rows is None:
                return None
            return [{k: v + 50.0 for k, v in row.items()} for row in rows]

        return garbage

    screening = Screening(
        builders=(("lucky", "llm", lucky_sometimes),), validation_splits=3)
    assert not screening.refresh(LEARNABLE, SPECS, seed=6), (
        "a builder that fails any split must not screen"
    )
    assert screening.telemetry.rejected_validation == 1
    consistent = Screening(
        builders=(("additive", "rule", additive_surrogate),),
        validation_splits=3)
    assert consistent.refresh(LEARNABLE, SPECS, seed=6), (
        "a consistently predictive builder must still pass all splits"
    )


# --- end to end through optimize() ------------------------------------------

class _Problem:
    candidate_model = type(
        "G", (BaseModel,), {"__annotations__": {"genome": list[Literal[0, 1]]}}
    )
    objectives = tuple(SPECS)

    def __init__(self) -> None:
        self.calls = 0

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ({"genome": [0, 0, 0, 0, 0, 0]},)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return tuple(config["genome"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        self.calls += 1
        return _truth(artifact)


def test_screening_cannot_touch_the_budget():
    problem = _Problem()
    result = optimize(problem, budget=24, seed=9, authorship="surrogate")
    assert problem.calls <= 24
    assert result.evaluations <= 24
    assert result.telemetry is not None
    assert result.telemetry.real_evaluations == result.evaluations
    assert result.telemetry.virtual_evaluations > 0, (
        "screening never engaged: no virtual evaluations were spent"
    )
    screens = [h["screen"] for h in result.history if "screen" in h]
    assert any(s.get("advanced", 0) > 0 for s in screens), (
        "no generation ever advanced a screened candidate"
    )
    assert all(s["held_out"] >= 1 for s in screens), (
        "the exploration floor must hold every screened generation"
    )
    rows = [m for m in result.telemetry.mechanisms
            if m.mechanism == "surrogate_screen"]
    assert rows and rows[0].counters["virtual_evaluations"] > 0


def test_authorship_off_reports_no_virtual_spend():
    result = optimize(_Problem(), budget=12, seed=9)
    assert result.telemetry.virtual_evaluations == 0
    assert not any("screen" in h for h in result.history)


def test_bogus_authorship_is_rejected_by_name():
    with pytest.raises(ValueError, match="authorship"):
        optimize(_Problem(), budget=8, authorship="bogus")
    with pytest.raises(ValueError, match="operator"):
        from agent_evolve.session.authorship import AuthorshipConfig
        AuthorshipConfig(operators="bogus")


def test_authorship_refuses_the_authoring_strategy_by_name():
    with pytest.raises(ValueError, match="genetic"):
        optimize(_Problem(), budget=8, strategy="authoring",
                 proposer="random", authorship="surrogate")
