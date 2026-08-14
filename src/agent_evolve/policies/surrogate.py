"""Cheap predictive models of the evaluator, and the gate they must pass.

A surrogate exists to spend VIRTUAL evaluations so real ones go further: the
loop builds more offspring than it can afford to measure, asks the surrogate
to order them, and pays the evaluator only for the promising ones. That is
only honest if the surrogate is actually predictive, so every surrogate --
rule or model-authored alike -- passes :func:`validate_surrogate` before it
may order anything, and is re-validated as data accumulates.

**The gate proves what its consumer relies on, and nothing else.** That is a
:class:`GatePolicy`, declared by the caller rather than assumed here. A
consumer that only ORDERS candidates (:mod:`agent_evolve.session.screening`
never reads a predicted magnitude -- it ranks by predicted domination) is
gated on RANK FIDELITY; the error ratio against the train-mean predictor is
still computed and returned, because it ARBITRATES among gate-passers, but a
magnitude test cannot reject an artifact whose magnitudes nobody consumes. A
consumer that reads PREDICTED VALUES is gated on both. Gating ordering on
magnitude was measured to switch the mechanism off wholesale: on an expensive
venue at B <= 24, 27-28% of gate calls never reached scoring and the
MSE-vs-train-mean term rejected 59-63% of the rest, while the rank term --
the one the consumer actually depends on -- rejected 1.4-2.4%.

**The evidence is cross-validated, not a single 30% holdout.** Every row is
held out exactly once and the statistics pool across the folds, so a run with
16 measured rows scores its surrogate on 16 held-out points instead of 4. At
the budgets an expensive venue admits, that is the difference between a rank
statistic and a coin flip -- and it is why the ordering policy can afford a
rank threshold meaningfully above chance plus a minimum effective holdout,
rather than the "no worse than chance" floor a 2-point holdout forced. The
single-holdout scheme remains available and agrees with cross-validation
where there is enough data for the question not to matter.

The two shipped surrogates are dependency-free rules. They are the
comparators any model-authored surrogate has to beat: an authored form that
cannot out-predict a shrunk per-locus mean has no business ordering
candidates.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.genetic import loci_of, read_locus

__all__ = [
    "GatePolicy",
    "ORDERING_GATE",
    "PREDICTION_GATE",
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


#: Purposes a consumer may declare. ``"ordering"`` means "I rank candidates
#: with this and never read a predicted magnitude"; ``"prediction"`` means "I
#: read the values themselves".
PURPOSES: Tuple[str, ...] = ("ordering", "prediction")

#: Validation schemes. ``"cross_validated"`` holds every row out exactly once
#: and pools the statistics; ``"holdout"`` is the single random split.
SCHEMES: Tuple[str, ...] = ("cross_validated", "holdout")


@dataclass(frozen=True)
class GatePolicy:
    """What the consumer relies on, and therefore what the gate must prove.

    The gate has two terms and they answer different questions. RANK
    fidelity asks "does this artifact order candidates the way the evaluator
    would" -- the only question a screen's output can be wrong about, since
    a screen consumes an ORDER. The ERROR ratio (holdout MSE against the
    train-mean predictor's) asks "are the magnitudes usable" -- which
    matters to a consumer that reads predicted values and not to one that
    sorts by them.

    A policy declares which terms REJECT. Both terms are always COMPUTED and
    returned: the error ratio arbitrates among gate-passers even under the
    ordering purpose (:mod:`agent_evolve.session.screening` picks the lowest
    median ratio among the artifacts that passed), so "does not gate" is not
    "is not measured".

    Fields:

    ``purpose``
        ``"ordering"`` -- rank fidelity gates, the error ratio arbitrates.
        ``"prediction"`` -- both gate.
    ``min_rank_correlation``
        The Spearman every objective must reach. The ordering purpose sets
        this meaningfully above chance rather than at the 0.0 ("no worse
        than a coin") floor a 2-point holdout used to force.
    ``min_effective_holdout``
        How many held-out points the statistics must be pooled over before
        the verdict is trusted at all. A Spearman on two points is +-1 by
        construction and is not evidence of anything.
    ``scheme`` / ``folds``
        ``"cross_validated"`` (default) with ``folds`` (0 = chosen from n),
        or the single ``"holdout"`` split of ``holdout_fraction``.
    ``min_rows``
        The absolute floor on measured rows below which no split is honest.
    """

    purpose: str = "prediction"
    min_rank_correlation: float = 0.0
    min_effective_holdout: int = 2
    scheme: str = "cross_validated"
    folds: int = 0
    holdout_fraction: float = 0.3
    min_rows: int = 8

    def __post_init__(self) -> None:
        if self.purpose not in PURPOSES:
            raise ValueError(
                f"purpose must be one of {PURPOSES}, got {self.purpose!r}")
        if self.scheme not in SCHEMES:
            raise ValueError(
                f"scheme must be one of {SCHEMES}, got {self.scheme!r}")
        if not -1.0 <= self.min_rank_correlation <= 1.0:
            raise ValueError(
                "min_rank_correlation must be in [-1, 1], got "
                f"{self.min_rank_correlation}")
        if self.min_effective_holdout < 2:
            raise ValueError(
                "min_effective_holdout must be at least 2, got "
                f"{self.min_effective_holdout}")
        if self.folds < 0 or self.folds == 1:
            raise ValueError(
                f"folds must be 0 (choose from n) or at least 2, got {self.folds}")
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError(
                f"holdout_fraction must be in (0, 1), got {self.holdout_fraction}")
        if self.min_rows < 2:
            raise ValueError(f"min_rows must be at least 2, got {self.min_rows}")

    @property
    def error_rejects(self) -> bool:
        """Does a worse-than-train-mean MSE reject, or only arbitrate?"""

        return self.purpose == "prediction"

    def folds_for(self, n: int) -> int:
        """How many folds *n* rows get: the declared count, or one from n.

        Bounded at 5 rather than leave-one-out because a fold costs one
        builder fit, and an authored builder's fit is an out-of-process
        call -- the statistic is pooled over every row either way.
        """

        if self.scheme != "cross_validated":
            return 1
        if self.folds:
            return max(2, min(self.folds, n))
        return max(2, min(5, n // 2))

    def replace(self, **overrides: Any) -> "GatePolicy":
        """This policy with fields overridden. Validated like any other."""

        return replace(self, **overrides)

    @classmethod
    def for_purpose(cls, purpose: str, **overrides: Any) -> "GatePolicy":
        """The standing policy for *purpose*, with optional overrides."""

        if purpose not in PURPOSES:
            raise ValueError(
                f"purpose must be one of {PURPOSES}, got {purpose!r}")
        if purpose == "ordering":
            base: Dict[str, Any] = {
                "min_rank_correlation": 0.5, "min_effective_holdout": 8}
        else:
            base = {"min_rank_correlation": 0.0, "min_effective_holdout": 2}
        base.update(overrides)
        base.pop("purpose", None)
        return cls(purpose=purpose, **base)


#: The gate for a consumer that ORDERS candidates and never reads a predicted
#: magnitude -- the surrogate screen. Rank fidelity 0.5 on every objective,
#: pooled over at least 8 held-out points. Under cross-validation the second
#: condition is met by any run that clears ``min_rows``, which is the point:
#: the threshold is affordable because the evidence is no longer a 2-point
#: split. 0.5 is above chance by a stated margin -- against the exact
#: permutation null a predictor with no information at all reaches it on 10.8%
#: of draws at n = 8, 4.9% at n = 12 and 1.3% at n = 20 -- and the screen
#: additionally requires it on EVERY objective and on EVERY validation split
#: (session.screening's variance guard).
ORDERING_GATE = GatePolicy.for_purpose("ordering")

#: The gate for a consumer that reads PREDICTED VALUES. Both terms reject,
#: at the historical thresholds.
PREDICTION_GATE = GatePolicy.for_purpose("prediction")


@dataclass(frozen=True)
class SurrogateValidation:
    """The gate's verdict, with the numbers that produced it.

    ``per_objective_spearman`` records the rank agreement between predicted
    and actual held-out values (average-rank ties; 0.0 when either side is
    entirely tied, i.e. no measurable ordering). It is empty on verdicts
    that never reached scoring (too little data, builder failure, unusable
    predictions).

    ``holdout`` is the EFFECTIVE holdout: how many held-out points the
    statistics were pooled over. Under cross-validation that is every row.

    ``reason`` is the machine-readable term that rejected -- ``""`` when the
    verdict passed -- so a caller can count WHY a gate closed without
    parsing ``detail``.
    """

    passed: bool
    per_objective_mse: Mapping[str, float]
    baseline_mse: Mapping[str, float]
    holdout: int
    detail: str = ""
    per_objective_spearman: Mapping[str, float] = field(default_factory=dict)
    reason: str = ""
    purpose: str = "prediction"
    scheme: str = "holdout"
    folds: int = 1

    @property
    def mse_ratio(self) -> Dict[str, float]:
        """Holdout MSE over the train-mean baseline's, where the baseline is
        non-degenerate. Below 1.0 is predictive. This is the ARBITRATION
        number: it separates gate-passers under every purpose, including the
        one where it does not reject."""

        return {
            name: self.per_objective_mse[name] / self.baseline_mse[name]
            for name in self.per_objective_mse
            if self.baseline_mse.get(name, 0.0) > 0.0
        }


def _average_ranks(values: Sequence[float]) -> List[float]:
    """Ranks 1..n with tied values sharing the average of their positions."""

    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _spearman(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Spearman rank correlation: Pearson correlation of average ranks.

    Hand-rolled and dependency-free. When either side is entirely tied
    there is no ordering to agree with (or none offered), so the
    correlation is reported as 0.0 -- no measurable agreement.
    """

    predicted_ranks = _average_ranks(predicted)
    actual_ranks = _average_ranks(actual)
    mean = (len(predicted_ranks) + 1) / 2.0  # both sides rank 1..n
    dp = [rank - mean for rank in predicted_ranks]
    da = [rank - mean for rank in actual_ranks]
    vp = sum(d * d for d in dp)
    va = sum(d * d for d in da)
    if vp <= 0.0 or va <= 0.0:
        return 0.0
    return sum(p * a for p, a in zip(dp, da)) / math.sqrt(vp * va)


def _splits(
    n: int, policy: GatePolicy, seed: int
) -> List[Tuple[List[int], List[int]]]:
    """``(train_idx, holdout_idx)`` pairs: k folds, or the one random split.

    Cross-validation shuffles once and cuts contiguous blocks, so every row
    is held out exactly once and the union of the holdouts is the data.
    """

    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    k = policy.folds_for(n)
    if k <= 1:
        cut = max(2, int(n * policy.holdout_fraction))
        return [(indices[cut:], indices[:cut])]
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out: List[Tuple[List[int], List[int]]] = []
    start = 0
    for size in sizes:
        held = indices[start:start + size]
        held_set = set(held)
        out.append(([i for i in indices if i not in held_set], held))
        start += size
    return out


class _Pooled:
    """Squared errors and (predicted, actual) pairs, accumulated over folds."""

    __slots__ = ("names", "mse", "baseline", "predicted", "actual", "count")

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)
        self.mse = {name: 0.0 for name in names}
        self.baseline = {name: 0.0 for name in names}
        self.predicted: Dict[str, List[float]] = {name: [] for name in names}
        self.actual: Dict[str, List[float]] = {name: [] for name in names}
        self.count = 0

    def add(self, holdout, predictions, train_mean) -> Optional[str]:
        """Fold in one split's results; the objective that was unusable, if any."""

        for (_cfg, actual), predicted in zip(holdout, predictions):
            for name in self.names:
                value = (predicted.get(name)
                         if isinstance(predicted, Mapping) else None)
                if (value is None or isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))):
                    return name
                self.mse[name] += (float(value) - float(actual[name])) ** 2
                self.baseline[name] += (
                    train_mean[name] - float(actual[name])) ** 2
                self.predicted[name].append(float(value))
                self.actual[name].append(float(actual[name]))
        self.count += len(holdout)
        return None


def validate_surrogate(
    builder: SurrogateBuilder,
    evaluated: Sequence[Tuple[Config, Mapping[str, float]]],
    specs: Sequence[ObjectiveSpec],
    *,
    policy: Optional[GatePolicy] = None,
    seed: int = 0,
    holdout_fraction: Optional[float] = None,
    min_rank_correlation: Optional[float] = None,
) -> SurrogateValidation:
    """May this surrogate be used for *policy*'s purpose, on today's evidence?

    Two terms, computed on every declared objective over data the surrogate
    never fitted, and *policy* decides which of them REJECT:

    - **Rank agreement**: Spearman correlation between predicted and actual
      held-out values must reach ``policy.min_rank_correlation``. MSE
      fidelity is not rank fidelity -- the study-2 analog trace read
      measured authored surrogates that passed the MSE split on half their
      losing seeds while still hurting the endpoint, i.e. they misordered
      the candidate pool, and ordering is the only thing a screen does with
      a surrogate. This term rejects under EVERY purpose.
    - **Error**: strictly beat the train-mean predictor's MSE, on every
      objective -- a surrogate that predicts latency and guesses energy
      would order the pool by half the problem while claiming to order it
      by all of it. This term rejects only under ``purpose="prediction"``.
      Under ``purpose="ordering"`` it is computed and returned
      (``mse_ratio``) and ARBITRATES among passers, because a consumer that
      sorts by predictions never reads their magnitudes -- gating ordering
      on magnitude was measured switching the screen off wholesale on the
      venues where evaluations are expensive enough for it to matter.

    Evidence comes from ``policy.scheme``: cross-validation (every row held
    out once, statistics pooled -- the default, and the only honest way to
    ask this question of 16 measured rows) or one random holdout split. A
    verdict is refused outright below ``policy.min_rows`` measured rows or
    below ``policy.min_effective_holdout`` pooled held-out points: a
    Spearman on two points is +-1 by construction and gates nothing.

    ``holdout_fraction`` and ``min_rank_correlation`` remain accepted as
    direct overrides of the corresponding policy fields.
    """

    policy = policy or PREDICTION_GATE
    overrides: Dict[str, Any] = {}
    if holdout_fraction is not None:
        overrides["holdout_fraction"] = holdout_fraction
    if min_rank_correlation is not None:
        overrides["min_rank_correlation"] = min_rank_correlation
    if overrides:
        policy = policy.replace(**overrides)

    names = [s.name for s in specs]
    n = len(evaluated)
    stamp: Dict[str, Any] = {
        "purpose": policy.purpose,
        "scheme": policy.scheme,
        "folds": policy.folds_for(n) if n else 0,
    }
    if n < policy.min_rows:
        return SurrogateValidation(
            False, {}, {}, 0,
            detail=(f"needs at least {policy.min_rows} evaluated points, "
                    f"has {n}"),
            reason="insufficient_rows", **stamp,
        )

    pooled = _Pooled(names)
    for train_idx, holdout_idx in _splits(n, policy, seed):
        train = [evaluated[i] for i in train_idx]
        holdout = [evaluated[i] for i in holdout_idx]
        if not train or not holdout:
            continue
        try:
            predict = builder(train, specs)
            predictions = predict([cfg for cfg, _obj in holdout])
        except Exception as error:
            return SurrogateValidation(
                False, {}, {}, pooled.count,
                detail=f"builder failed: {type(error).__name__}: {error}"[:200],
                reason="builder_failed", **stamp,
            )
        if predictions is None or len(predictions) != len(holdout):
            return SurrogateValidation(
                False, {}, {}, pooled.count, detail="no usable predictions",
                reason="no_predictions", **stamp,
            )
        train_mean = {
            name: sum(float(obj[name]) for _c, obj in train) / len(train)
            for name in names
        }
        unusable = pooled.add(holdout, predictions, train_mean)
        if unusable is not None:
            return SurrogateValidation(
                False, {}, {}, pooled.count,
                detail=f"non-finite or missing prediction for {unusable!r}",
                reason="bad_prediction", **stamp,
            )
    return _score(pooled, names, policy, stamp)


def _score(pooled: "_Pooled", names, policy: GatePolicy,
           stamp: Mapping[str, Any]) -> SurrogateValidation:
    count = pooled.count
    if count < policy.min_effective_holdout:
        return SurrogateValidation(
            False, {}, {}, count,
            detail=(f"effective holdout {count} is below "
                    f"{policy.min_effective_holdout}: a rank statistic on "
                    "that many points is not evidence"),
            reason="insufficient_holdout", **stamp,
        )
    mse = {k: v / count for k, v in pooled.mse.items()}
    baseline = {k: v / count for k, v in pooled.baseline.items()}
    spearman = {
        name: _spearman(pooled.predicted[name], pooled.actual[name])
        for name in names
    }
    mse_ok = all(mse[name] < baseline[name] for name in names)
    rank_ok = all(spearman[name] >= policy.min_rank_correlation
                  for name in names)
    detail = ""
    reason = ""
    if not rank_ok:
        worst = min(names, key=lambda name: spearman[name])
        reason = "rank"
        detail = (
            f"rank agreement {spearman[worst]:.3f} on {worst!r} is below "
            f"{policy.min_rank_correlation:.3f}: MSE fidelity is not rank "
            "fidelity"
        )
    elif not mse_ok and policy.error_rejects:
        worst = max(names,
                    key=lambda name: (mse[name] / baseline[name]
                                      if baseline[name] > 0.0 else math.inf))
        reason = "error"
        ratio = (f"{mse[worst] / baseline[worst]:.2f}x"
                 if baseline[worst] > 0.0 else "at a degenerate")
        detail = (f"holdout error is {ratio} the train-mean baseline on "
                  f"{worst!r}: the magnitudes this purpose consumes are not "
                  "predictive")
    passed = rank_ok and (mse_ok or not policy.error_rejects)
    return SurrogateValidation(
        passed, mse, baseline, count,
        detail=detail, per_objective_spearman=spearman, reason=reason,
        **stamp,
    )
