"""Cheap predictive models of the evaluator, and the gate they must pass.

A surrogate exists to spend VIRTUAL evaluations so real ones go further: the
loop builds more offspring than it can afford to measure, asks the surrogate
to order them, and pays the evaluator only for the promising ones. That is
only honest if the surrogate is actually predictive, so every surrogate --
rule or model-authored alike -- passes :func:`validate_surrogate` before it
may order anything, and is re-validated as data accumulates. The gate is
sharp and preregistered in code: strictly beat the trivial train-mean
predictor on EVERY declared objective, on a held-out split, or sit out this
generation.

The two shipped surrogates are dependency-free rules. They are the
comparators any model-authored surrogate has to beat: an authored form that
cannot out-predict a shrunk per-locus mean has no business ordering
candidates.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.genetic import loci_of, read_locus

__all__ = [
    "Predict",
    "SurrogateBuilder",
    "SurrogateValidation",
    "knn_surrogate",
    "additive_surrogate",
    "validate_surrogate",
]

Config = Dict[str, Any]

#: Batch predictor: configurations in, one objective dict per configuration
#: out, or ``None`` when prediction is unavailable (the caller then screens
#: nothing rather than screening on garbage).
Predict = Callable[[Sequence[Config]], Optional[Sequence[Mapping[str, float]]]]

#: Fits a predictor to evaluated data. Rule builders run in-process; authored
#: builders wrap the out-of-process runtime behind the same signature.
SurrogateBuilder = Callable[
    [Sequence[Tuple[Config, Mapping[str, float]]], Sequence[ObjectiveSpec]],
    Predict,
]


def knn_surrogate(
    evaluated: Sequence[Tuple[Config, Mapping[str, float]]],
    specs: Sequence[ObjectiveSpec],
    *,
    k: int = 3,
) -> Predict:
    """Nearest neighbours by Hamming distance over loci; mean of their outcomes."""

    data = [(cfg, dict(obj)) for cfg, obj in evaluated]
    names = [s.name for s in specs]

    def predict(pool: Sequence[Config]) -> Optional[Sequence[Mapping[str, float]]]:
        if not data:
            return None
        out = []
        for candidate in pool:
            loci = loci_of(candidate)

            def distance(cfg: Config) -> int:
                return sum(
                    1 for lc in loci
                    if read_locus(cfg, lc) != read_locus(candidate, lc)
                )

            nearest = sorted(data, key=lambda pair: distance(pair[0]))[:k]
            out.append({
                name: sum(obj[name] for _c, obj in nearest) / len(nearest)
                for name in names
            })
        return out

    return predict


def additive_surrogate(
    evaluated: Sequence[Tuple[Config, Mapping[str, float]]],
    specs: Sequence[ObjectiveSpec],
) -> Predict:
    """Global mean plus shrunk per-locus-value deviations.

    The program's own landscape studies measured 88-99.99% additive structure
    on every venue censused, which is why a first-order model is the honest
    default rather than a strawman. Deviations shrink by ``n/(n+1)`` so a
    value seen once moves a prediction half as far as its raw mean would.
    """

    names = [s.name for s in specs]
    if not evaluated:
        return lambda pool: None
    grand = {
        name: sum(float(obj[name]) for _c, obj in evaluated) / len(evaluated)
        for name in names
    }
    per_value: Dict[Tuple[str, Any], Dict[str, float]] = {}
    counts: Dict[Tuple[str, Any], int] = {}
    for cfg, obj in evaluated:
        for lc in loci_of(cfg):
            key = (str(lc), read_locus(cfg, lc))
            counts[key] = counts.get(key, 0) + 1
            bucket = per_value.setdefault(key, {name: 0.0 for name in names})
            for name in names:
                bucket[name] += float(obj[name])

    def predict(pool: Sequence[Config]) -> Optional[Sequence[Mapping[str, float]]]:
        out = []
        for candidate in pool:
            estimate = dict(grand)
            for lc in loci_of(candidate):
                key = (str(lc), read_locus(candidate, lc))
                n = counts.get(key, 0)
                if not n:
                    continue
                shrink = n / (n + 1.0)
                for name in names:
                    deviation = per_value[key][name] / n - grand[name]
                    estimate[name] += shrink * deviation
            out.append(estimate)
        return out

    return predict


@dataclass(frozen=True)
class SurrogateValidation:
    """The gate's verdict, with the numbers that produced it."""

    passed: bool
    per_objective_mse: Mapping[str, float]
    baseline_mse: Mapping[str, float]
    holdout: int
    detail: str = ""


def validate_surrogate(
    builder: SurrogateBuilder,
    evaluated: Sequence[Tuple[Config, Mapping[str, float]]],
    specs: Sequence[ObjectiveSpec],
    *,
    holdout_fraction: float = 0.3,
    seed: int = 0,
) -> SurrogateValidation:
    """May this surrogate order candidates, on today's evidence?

    Strictly beat the train-mean predictor on EVERY declared objective, on a
    held-out split it never fitted. Failing one objective fails the gate: a
    surrogate that predicts latency and guesses energy would order the pool
    by half the problem while claiming to order it by all of it.
    """

    names = [s.name for s in specs]
    n = len(evaluated)
    if n < 8:
        return SurrogateValidation(
            False, {}, {}, 0,
            detail=f"needs at least 8 evaluated points, has {n}",
        )
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    cut = max(2, int(n * holdout_fraction))
    holdout_idx, train_idx = indices[:cut], indices[cut:]
    train = [evaluated[i] for i in train_idx]
    holdout = [evaluated[i] for i in holdout_idx]

    try:
        predict = builder(train, specs)
        predictions = predict([cfg for cfg, _obj in holdout])
    except Exception as error:
        return SurrogateValidation(
            False, {}, {}, len(holdout),
            detail=f"builder failed: {type(error).__name__}: {error}"[:200],
        )
    if predictions is None or len(predictions) != len(holdout):
        return SurrogateValidation(
            False, {}, {}, len(holdout), detail="no usable predictions",
        )

    train_mean = {
        name: sum(float(obj[name]) for _c, obj in train) / len(train)
        for name in names
    }
    return _score(holdout, predictions, names, train_mean)


def _score(holdout, predictions, names, train_mean) -> SurrogateValidation:
    mse = {name: 0.0 for name in names}
    baseline = {name: 0.0 for name in names}
    for (_cfg, actual), predicted in zip(holdout, predictions):
        for name in names:
            value = predicted.get(name) if isinstance(predicted, Mapping) else None
            if (value is None or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                return SurrogateValidation(
                    False, {}, {}, len(holdout),
                    detail=f"non-finite or missing prediction for {name!r}",
                )
            mse[name] += (float(value) - float(actual[name])) ** 2
            baseline[name] += (train_mean[name] - float(actual[name])) ** 2
    count = len(holdout)
    mse = {k: v / count for k, v in mse.items()}
    baseline = {k: v / count for k, v in baseline.items()}
    passed = all(mse[name] < baseline[name] for name in names)
    return SurrogateValidation(passed, mse, baseline, count)
