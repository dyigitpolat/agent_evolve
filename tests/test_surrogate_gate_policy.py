"""The gate proves what its consumer relies on, on evidence that is enough.

Two defects are pinned here, both measured on an expensive venue where the
surrogate screen went dark:

1. **The evidence was a 2-to-7 point holdout.** At the budgets an expensive
   evaluator admits (B <= 24), a single 30% split leaves so few held-out
   points that no statistic computed on it means anything -- 27-28% of gate
   calls never reached scoring at all. Cross-validation holds every row out
   exactly once and pools, so 16 measured rows are scored on 16 points.

2. **The gate rejected on a term its consumer does not consume.** The screen
   sorts candidates by predicted domination and never reads a magnitude, yet
   an MSE-vs-train-mean test rejected 59-63% of the calls that reached
   scoring while rank fidelity -- the term the screen's output can actually
   be wrong about -- rejected 1.4-2.4%.

What must NOT move: an artifact that inverts the ordering is rejected under
every policy, a failed gate still means the loop measures its own picks, and
the variance guard still refuses a builder that is only sometimes right.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Dict, Sequence

import pytest

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.surrogate import (
    ORDERING_GATE,
    PREDICTION_GATE,
    GatePolicy,
    SurrogateValidation,
    additive_surrogate,
    validate_surrogate,
)
from agent_evolve.session.screening import Screening

SPECS = [ObjectiveSpec("ones", "max"), ObjectiveSpec("zeros", "min")]


def _truth(genome) -> Dict[str, float]:
    ones = float(sum(genome))
    return {"ones": ones, "zeros": float(len(genome)) - ones}


def _data(rows: Sequence[Sequence[int]]):
    return [({"genome": list(row)}, _truth(row)) for row in rows]


def _additive_rows(n: int, width: int = 8, seed: int = 0):
    """*n* distinct-ish genomes over a perfectly additive landscape."""

    rng = random.Random(seed)
    return _data([[rng.randint(0, 1) for _ in range(width)] for _ in range(n)])


#: Enough rows that the additive rule is genuinely predictive out of sample.
#: Below ~16 rows over a space this size it is NOT, and the gate says so -- see
#: ``test_the_gate_tracks_whether_the_artifact_is_actually_predictive``.
LEARNABLE = _additive_rows(24, width=6, seed=1)


# --- the policy is typed, and its fields are checked -------------------------

def test_a_policy_declares_a_purpose_and_refuses_nonsense():
    assert ORDERING_GATE.purpose == "ordering"
    assert PREDICTION_GATE.purpose == "prediction"
    assert ORDERING_GATE.error_rejects is False, (
        "the ordering purpose must not reject on a magnitude its consumer "
        "never reads"
    )
    assert PREDICTION_GATE.error_rejects is True
    for bad in (
        {"purpose": "vibes"},
        {"scheme": "bootstrap"},
        {"min_rank_correlation": 1.5},
        {"min_effective_holdout": 1},
        {"folds": 1},
        {"holdout_fraction": 0.0},
        {"min_rows": 1},
    ):
        with pytest.raises(ValueError):
            GatePolicy(**bad)


def test_the_ordering_policy_still_measures_the_error_it_stopped_gating_on():
    """"Does not reject" is not "is not measured": the ratio arbitrates."""

    def biased(evaluated, specs):
        inner = additive_surrogate(evaluated, specs)

        def predict(pool):
            rows = inner(pool)
            return None if rows is None else [
                {k: 3.0 * v - 5.0 for k, v in row.items()} for row in rows]

        return predict

    verdict = validate_surrogate(biased, LEARNABLE, SPECS, seed=1,
                                 policy=ORDERING_GATE)
    assert verdict.passed, "a monotone predictor orders perfectly: it may screen"
    assert verdict.mse_ratio and all(r > 1.0 for r in verdict.mse_ratio.values()), (
        "the magnitudes are worse than the train mean and the gate must SAY so"
    )
    strict = validate_surrogate(biased, LEARNABLE, SPECS, seed=1,
                                policy=PREDICTION_GATE)
    assert not strict.passed and strict.reason == "error", (
        "a consumer that reads magnitudes must still be protected from these"
    )


# --- cross-validation: all the rows, pooled ----------------------------------

def test_cross_validation_scores_every_row_and_the_holdout_scheme_does_not():
    cv = validate_surrogate(additive_surrogate, LEARNABLE, SPECS, seed=3,
                            policy=PREDICTION_GATE)
    split = validate_surrogate(
        additive_surrogate, LEARNABLE, SPECS, seed=3,
        policy=PREDICTION_GATE.replace(scheme="holdout"))
    assert cv.holdout == len(LEARNABLE), (
        "cross-validation must hold every row out exactly once"
    )
    assert cv.scheme == "cross_validated" and cv.folds >= 2
    assert split.holdout == max(2, int(len(LEARNABLE) * 0.3))
    assert split.scheme == "holdout" and split.folds == 1


@pytest.mark.parametrize("n", [8, 12, 16, 24])
def test_the_effective_holdout_is_the_whole_run_at_the_budgets_that_matter(n):
    """The expensive-venue regime: B <= 24 measured rows, 3 objectives.

    Under the old single split those runs were judged on 2-7 points. Under
    cross-validation they are judged on all of them, which is the whole
    reason a rank threshold above chance is affordable at all.
    """

    rows = _additive_rows(n, seed=n)
    cv = validate_surrogate(additive_surrogate, rows, SPECS, seed=5,
                            policy=ORDERING_GATE)
    old = validate_surrogate(
        additive_surrogate, rows, SPECS, seed=5,
        policy=ORDERING_GATE.replace(scheme="holdout", min_effective_holdout=2))
    assert cv.holdout == n
    assert old.holdout <= max(2, int(n * 0.3)) < n
    assert cv.holdout >= ORDERING_GATE.min_effective_holdout


def test_cross_validation_and_the_single_split_agree_where_data_is_plentiful():
    """The two schemes are the same question asked with different care.

    At large n a 30% holdout is already enough, so the schemes must agree --
    that is what makes cross-validation a data-efficiency fix rather than a
    change of bar. The disagreements must live at small n, where they are
    the point.
    """

    plentiful = mismatched = 0
    for seed in range(12):
        rows = _additive_rows(400, seed=seed)
        cv = validate_surrogate(additive_surrogate, rows, SPECS, seed=seed,
                                policy=PREDICTION_GATE)
        split = validate_surrogate(
            additive_surrogate, rows, SPECS, seed=seed,
            policy=PREDICTION_GATE.replace(scheme="holdout"))
        plentiful += 1
        if cv.passed != split.passed:
            mismatched += 1
        assert cv.passed, "an additive model of an additive landscape predicts"
        for name in ("ones", "zeros"):
            assert cv.mse_ratio[name] == pytest.approx(
                split.mse_ratio[name], abs=0.15), (
                "at n=400 the pooled and the single-split error ratios must "
                "be the same number to within sampling noise"
            )
    assert plentiful == 12 and mismatched == 0


# --- the minimum effective holdout -------------------------------------------

def test_a_rank_statistic_on_two_points_cannot_pass_the_ordering_gate():
    """Spearman on 2 points is +-1 by construction. It is not evidence.

    Same artifact, same 24 rows, two ways of spending them: two held-out
    points is refused outright, twenty-four is admitted. The refusal is the
    guard that makes a rank-gated screen safe at small n.
    """

    rows = _additive_rows(24, seed=4)
    tiny = ORDERING_GATE.replace(scheme="holdout", holdout_fraction=0.08)
    verdict = validate_surrogate(additive_surrogate, rows, SPECS, seed=4,
                                 policy=tiny)
    assert verdict.holdout == 2
    assert not verdict.passed and verdict.reason == "insufficient_holdout"
    assert "not evidence" in verdict.detail
    pooled = validate_surrogate(additive_surrogate, rows, SPECS, seed=4,
                                policy=ORDERING_GATE)
    assert pooled.passed and pooled.holdout == 24


def test_the_gate_tracks_whether_the_artifact_is_actually_predictive():
    """Rank-gating is not a rubber stamp: it follows the evidence.

    The same rule surrogate on the same landscape is out-of-sample useless
    at 8 measured rows over an 8-locus space and reliable at 24. A gate worth
    having refuses the first and admits the second, and the difference must
    show up as a REJECTION RATE, not as an opinion.
    """

    def passes(n: int) -> int:
        return sum(
            validate_surrogate(additive_surrogate, _additive_rows(n, seed=s),
                               SPECS, seed=s, policy=ORDERING_GATE).passed
            for s in range(8)
        )

    starved, fed = passes(8), passes(24)
    assert starved <= 5, "8 rows over 2^8 configurations do not license a screen"
    assert fed == 8, "24 rows of an additive landscape plainly do"


def test_too_few_measured_rows_is_still_refused_outright():
    verdict = validate_surrogate(additive_surrogate, LEARNABLE[:5], SPECS,
                                 policy=ORDERING_GATE)
    assert not verdict.passed and verdict.reason == "insufficient_rows"
    assert "8" in verdict.detail


# --- the rank threshold is above chance, measured ----------------------------

def _exact_spearman_null(n: int, threshold: float) -> float:
    """P(Spearman >= threshold) for a predictor with no information at all.

    Exact over all n! rank permutations -- no table, no approximation.
    """

    hits = total = 0
    for permutation in itertools.permutations(range(n)):
        sum_d2 = sum((i - permutation[i]) ** 2 for i in range(n))
        rho = 1.0 - 6.0 * sum_d2 / (n * (n * n - 1))
        hits += rho >= threshold
        total += 1
    return hits / total


def test_the_ordering_gates_rank_threshold_is_meaningfully_above_chance():
    """The old default was 0.0 -- "no worse than a coin". This is not that.

    Measured against the exact permutation null at the smallest holdout the
    ordering policy will accept. The screen additionally requires the
    threshold on EVERY objective and on EVERY validation split, so the
    per-objective rate below is an upper bound on the gate's.
    """

    n = ORDERING_GATE.min_effective_holdout
    chance = _exact_spearman_null(n, 0.0)
    gated = _exact_spearman_null(n, ORDERING_GATE.min_rank_correlation)
    assert chance > 0.45, "sanity: a coin agrees with a coin about half the time"
    assert gated <= 0.12, (
        f"a chance predictor reaches {ORDERING_GATE.min_rank_correlation} on "
        f"{gated:.1%} of draws at n={n}: that is not a gate"
    )
    assert gated < chance / 4.0


# --- legacy equivalence: the prediction/holdout path is bit-identical --------

def _legacy_validate(builder, evaluated, specs, *, holdout_fraction=0.3,
                     seed=0, min_rank_correlation=0.0):
    """The pre-fix implementation, verbatim, as the reference to match.

    Kept here rather than described, so "the single-holdout path is
    preserved" is a comparison and not a claim.
    """

    from agent_evolve.policies.surrogate import _average_ranks

    def spearman(predicted, actual):
        pr, ar = _average_ranks(predicted), _average_ranks(actual)
        mean = (len(pr) + 1) / 2.0
        dp = [r - mean for r in pr]
        da = [r - mean for r in ar]
        vp, va = sum(d * d for d in dp), sum(d * d for d in da)
        if vp <= 0.0 or va <= 0.0:
            return 0.0
        return sum(p * a for p, a in zip(dp, da)) / math.sqrt(vp * va)

    names = [s.name for s in specs]
    n = len(evaluated)
    if n < 8:
        return SurrogateValidation(False, {}, {}, 0, detail="too little")
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    cut = max(2, int(n * holdout_fraction))
    holdout_idx, train_idx = indices[:cut], indices[cut:]
    train = [evaluated[i] for i in train_idx]
    holdout = [evaluated[i] for i in holdout_idx]
    try:
        predictions = builder(train, specs)([cfg for cfg, _o in holdout])
    except Exception:
        return SurrogateValidation(False, {}, {}, len(holdout), detail="builder")
    if predictions is None or len(predictions) != len(holdout):
        return SurrogateValidation(False, {}, {}, len(holdout), detail="none")
    train_mean = {name: sum(float(o[name]) for _c, o in train) / len(train)
                  for name in names}
    mse = {name: 0.0 for name in names}
    baseline = {name: 0.0 for name in names}
    pv = {name: [] for name in names}
    av = {name: [] for name in names}
    for (_cfg, actual), predicted in zip(holdout, predictions):
        for name in names:
            value = float(predicted[name])
            mse[name] += (value - float(actual[name])) ** 2
            baseline[name] += (train_mean[name] - float(actual[name])) ** 2
            pv[name].append(value)
            av[name].append(float(actual[name]))
    count = len(holdout)
    mse = {k: v / count for k, v in mse.items()}
    baseline = {k: v / count for k, v in baseline.items()}
    rank = {name: spearman(pv[name], av[name]) for name in names}
    passed = (all(mse[k] < baseline[k] for k in names)
              and all(rank[k] >= min_rank_correlation for k in names))
    return SurrogateValidation(passed, mse, baseline, count,
                               per_objective_spearman=rank)


def test_the_single_holdout_path_reproduces_the_pre_fix_gate_exactly():
    """Bit-identical, over every builder shape and seed this suite has.

    This is what licenses a before/after comparison run from one binary:
    ``purpose="prediction", scheme="holdout"`` IS the old gate.
    """

    def knn3(evaluated, specs):
        from agent_evolve.policies.surrogate import knn_surrogate
        return knn_surrogate(evaluated, specs)

    def inverted(evaluated, specs):
        inner = additive_surrogate(evaluated, specs)

        def predict(pool):
            rows = inner(pool)
            return None if rows is None else [
                {k: -v for k, v in row.items()} for row in rows]

        return predict

    legacy_policy = PREDICTION_GATE.replace(scheme="holdout")
    compared = 0
    for builder in (additive_surrogate, knn3, inverted):
        for n in (8, 9, 13, 20, 41):
            rows = _additive_rows(n, seed=n)
            for seed in range(5):
                new = validate_surrogate(builder, rows, SPECS, seed=seed,
                                         policy=legacy_policy)
                old = _legacy_validate(builder, rows, SPECS, seed=seed)
                assert new.passed == old.passed, (builder, n, seed)
                assert new.holdout == old.holdout
                assert dict(new.per_objective_mse) == dict(old.per_objective_mse)
                assert dict(new.baseline_mse) == dict(old.baseline_mse)
                assert (dict(new.per_objective_spearman)
                        == dict(old.per_objective_spearman))
                compared += 1
    assert compared == 75


# --- the screen declares its purpose, and the safety properties hold ---------

def test_the_screen_validates_under_the_ordering_policy():
    screening = Screening(builders=(("additive", "rule", additive_surrogate),))
    assert screening.gate is ORDERING_GATE, (
        "the screen consumes an order; it must say so to the gate"
    )
    with pytest.raises(TypeError):
        Screening(builders=(("additive", "rule", additive_surrogate),),
                  gate="ordering")


def test_the_variance_guard_still_refuses_a_builder_that_sometimes_misorders():
    """The ladder1 E2 failure mode, expressed in the currency that matters.

    A builder that orders correctly on one fit and BACKWARDS on the next is
    exactly what a screen must never install, and the multi-split guard is
    what catches it: one partition may be lucky, all of them are not.
    """

    calls = {"n": 0}

    def sometimes_backwards(evaluated, specs):
        calls["n"] += 1
        inner = additive_surrogate(evaluated, specs)
        if calls["n"] % 2 == 1:
            return inner

        def backwards(pool):
            rows = inner(pool)
            return None if rows is None else [
                {k: -v for k, v in row.items()} for row in rows]

        return backwards

    rows = _additive_rows(24, seed=11)
    screening = Screening(builders=(("unstable", "llm", sometimes_backwards),),
                          validation_splits=3)
    assert not screening.refresh(rows, SPECS, seed=6)
    assert screening.telemetry.rejected_validation == 1
    assert screening.screen([{"genome": [0] * 8}], [], SPECS) is None, (
        "a screen whose surrogate failed today's gate must screen nothing"
    )
    steady = Screening(builders=(("additive", "rule", additive_surrogate),),
                       validation_splits=3)
    assert steady.refresh(rows, SPECS, seed=6), (
        "a consistently faithful builder must still pass every split"
    )


def test_rejections_are_counted_by_reason():
    """Telemetry must separate "not predictive" from "never had enough data".

    The census that diagnosed this defect had to monkeypatch the gate to
    learn which term closed it. That is a product gap, and this is it shut.
    """

    screening = Screening(builders=(("additive", "rule", additive_surrogate),))
    assert not screening.refresh(LEARNABLE[:4], SPECS, seed=1)
    counters = screening.telemetry.as_dict()
    assert counters["gate_calls"] == 1
    assert counters["rejected_insufficient_rows"] == 1
    assert counters["rejected_rank"] == 0
    assert screening.telemetry.rejected_validation == 1

    def anti(evaluated, specs):
        def predict(pool):
            return [{"ones": -float(sum(cfg["genome"])),
                     "zeros": float(sum(cfg["genome"]))} for cfg in pool]

        return predict

    misordering = Screening(builders=(("anti", "llm", anti),))
    assert not misordering.refresh(_additive_rows(24, seed=2), SPECS, seed=1)
    assert misordering.telemetry.as_dict()["rejected_rank"] == 1
