"""Virtual pre-screening: order a pool of offspring before paying for any.

This module is deliberately starved: :func:`screen_offspring` receives
configurations and objective vectors and a predictor -- never the problem,
never the evaluation cache -- so "the surrogate cannot spend budget" is a
property of the import graph, not a convention. The only route from here to
a real evaluation is that the loop measures the candidates this module
merely ordered.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.surrogate import (
    Predict,
    SurrogateBuilder,
    validate_surrogate,
)

__all__ = ["ScreenReport", "screen_offspring", "Screening", "ScreeningTelemetry"]

Config = Dict[str, Any]


@dataclass(frozen=True)
class ScreenReport:
    """The pool, ordered by predicted worth. Indices address the caller's pool."""

    order: Tuple[int, ...]
    predicted: Tuple[Mapping[str, float], ...]
    virtual_evaluations: int
    surrogate_name: str


def _oriented(
    rows: Sequence[Mapping[str, float]], specs: Sequence[ObjectiveSpec]
) -> Optional[list[tuple]]:
    """Objective vectors as "smaller is better" float tuples, or ``None``.

    ``None`` means a row was missing an objective or carried something that
    is not a finite number -- the same "screen nothing" answer this module
    already gives for malformed predictions, rather than an exception from
    the middle of a ranking loop.
    """

    signs = [1.0 if spec.goal == "min" else -1.0 for spec in specs]
    names = [spec.name for spec in specs]
    out: list[tuple] = []
    for row in rows:
        vector = []
        for sign, name in zip(signs, names):
            value = row.get(name)
            if (value is None or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                return None
            vector.append(sign * float(value))
        out.append(tuple(vector))
    return out


def _dominator_counts(
    rows: Sequence[Mapping[str, float]],
    against: Sequence[Mapping[str, float]],
    specs: Sequence[ObjectiveSpec],
) -> Optional[list[int]]:
    """How many of ``rows + against`` dominate each of *rows*. Lower is better.

    Exactly what ``sum(1 for other in field if dominates(other, row))`` says,
    computed over DISTINCT objective vectors weighted by how many rows carry
    each. Dominance is a property of the vector alone, so this returns the
    same integers -- but a pool of n candidates whose predictions take k
    distinct values costs O(k^2) instead of O(n^2), and a surrogate over a
    discrete space collapses thousands of candidates onto tens of vectors.
    The comparison itself works on pre-oriented float tuples: the general
    ``core.results.dominates`` re-validates every objective on every call
    (a ``numbers.Real`` ABC check per number), which is right for a contract
    boundary and ruinous inside a quadratic loop -- measured at 21.7s of a
    25.6s run before this, on one screened pool of 2,000.
    """

    keys = _oriented(rows, specs)
    other_keys = _oriented(against, specs)
    if keys is None or other_keys is None:
        return None

    multiplicity: Dict[tuple, int] = {}
    for key in keys:
        multiplicity[key] = multiplicity.get(key, 0) + 1
    for key in other_keys:
        multiplicity[key] = multiplicity.get(key, 0) + 1
    distinct = list(multiplicity.items())

    def _beats(a: tuple, b: tuple) -> bool:
        better = False
        for x, y in zip(a, b):
            if x > y:
                return False
            if x < y:
                better = True
        return better

    counted = {
        key: sum(weight for other, weight in distinct if _beats(other, key))
        for key, _weight in distinct
    }
    return [counted[key] for key in keys]


def screen_offspring(
    pool: Sequence[Config],
    population_objectives: Sequence[Mapping[str, float]],
    specs: Sequence[ObjectiveSpec],
    predict: Predict,
    *,
    surrogate_name: str = "surrogate",
) -> Optional[ScreenReport]:
    """Order *pool* by predicted domination against the measured population.

    A candidate's rank counts how many points dominate its PREDICTION -- other
    predictions and the population's real measurements together -- so a pool
    member that merely reshuffles known-dominated territory sinks, and one
    predicted past the current front rises. Never scalarized. ``None`` (from
    the predictor, or on malformed predictions) means "screen nothing": the
    caller falls back to measuring its original picks.
    """

    if not pool:
        return None
    predictions = predict(list(pool))
    if predictions is None or len(predictions) != len(pool):
        return None
    names = [s.name for s in specs]
    clean: list[dict[str, float]] = []
    for predicted in predictions:
        row = {}
        for name in names:
            value = predicted.get(name) if isinstance(predicted, Mapping) else None
            if (value is None or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                return None
            row[name] = float(value)
        clean.append(row)

    ranks = _dominator_counts(
        clean, [dict(objectives) for objectives in population_objectives], specs)
    if ranks is None:
        return None
    order = tuple(sorted(range(len(pool)), key=lambda i: (ranks[i], i)))
    return ScreenReport(
        order=order,
        predicted=tuple(clean),
        virtual_evaluations=len(pool),
        surrogate_name=surrogate_name,
    )


class ScreeningTelemetry:
    """What the screen did, counted. Reaches the result via harvest."""

    __slots__ = ("refreshes", "validated", "rejected_validation", "screens",
                 "screen_failures", "virtual_evaluations", "chosen_llm",
                 "chosen_rule", "revisions", "revisions_accepted")

    def __init__(self) -> None:
        self.refreshes = 0
        self.validated = 0
        self.rejected_validation = 0
        self.screens = 0
        self.screen_failures = 0
        self.virtual_evaluations = 0
        self.chosen_llm = 0
        self.chosen_rule = 0
        self.revisions = 0
        self.revisions_accepted = 0

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__slots__}


class Screening:
    """The screening policy: builders, the gate, and the current predictor.

    ``builders`` is an ordered sequence of ``(name, authored_by, builder)``;
    each generation, :meth:`refresh` re-validates them in order on the data
    measured so far and installs the FIRST that passes the gate -- so an
    authored surrogate listed ahead of the rules is used exactly when it
    earns it, and the rules are the standing fallback.
    """

    def __init__(
        self,
        builders: Sequence[Tuple[str, str, SurrogateBuilder]],
        *,
        pool_factor: int = 4,
        exploration_floor: float = 0.25,
        validation_splits: int = 3,
        revise: Any = None,
        max_revisions: int = 2,
        max_training_rows: int = 1024,
    ) -> None:
        if pool_factor < 2:
            raise ValueError(f"pool_factor must be at least 2, got {pool_factor}")
        if max_training_rows < 1:
            raise ValueError(
                f"max_training_rows must be at least 1, got {max_training_rows}"
            )
        if not 0.0 <= exploration_floor < 1.0:
            raise ValueError(
                f"exploration_floor must be in [0, 1), got {exploration_floor}"
            )
        if validation_splits < 1:
            raise ValueError(
                f"validation_splits must be at least 1, got {validation_splits}"
            )
        self.builders = tuple(builders)
        self.pool_factor = int(pool_factor)
        self.exploration_floor = float(exploration_floor)
        self.validation_splits = int(validation_splits)
        #: The evolving-surrogate hook: called with (evaluated, specs) when
        #: the llm builder exists and did not win this refresh, at most
        #: max_revisions times per run. Returns a replacement
        #: (name, authored_by, builder) entry -- authored from the current
        #: artifact plus its measured validation residuals -- or None. The
        #: revision competes from the NEXT refresh under the same gate; a
        #: model that cannot fix its artifact keeps losing to the rules.
        self.revise = revise
        self.max_revisions = int(max_revisions)
        #: How many of the most recent measurements a refresh fits and
        #: validates on. Refitting every builder on EVERYTHING measured so
        #: far makes one refresh O(n) and a run O(n^2): at B=10,000 the screen
        #: alone runs for over a quarter of an hour and never finishes a run,
        #: which is precisely the regime an authored generator exists for.
        #: The recent window is also the better statistics for a distribution
        #: the search keeps moving. The default is far above any campaign run
        #: to date (all at B <= 150), so every measured run is unaffected.
        self.max_training_rows = int(max_training_rows)
        self.telemetry = ScreeningTelemetry()
        self.mechanism = "surrogate_screen"
        self.authored_by = "none"
        self._predict: Optional[Predict] = None
        self._name = ""

    def refresh(
        self,
        evaluated: Sequence[Tuple[Config, Mapping[str, float]]],
        specs: Sequence[ObjectiveSpec],
        *,
        seed: int = 0,
    ) -> bool:
        """Re-arbitrate: today's data decides WHO may screen, if anyone.

        Every builder is validated on ``validation_splits`` INDEPENDENT
        held-out splits and must pass the gate on EVERY one; among the
        survivors, the lowest median mse/baseline ratio wins the generation.
        The all-splits requirement is the variance guard the ladder1 E2 row
        demanded: a high-variance authored artifact can pass one small
        holdout by luck and then mis-screen mid-run -- measured at the
        cheapest scale, where authored screening HURT the endpoint under the
        single-split gate. Surviving every split of today's data is the
        in-loop generalization of "pass on both frozen datasets
        independently". Best-passing across splits stays the arbitration:
        listing the authored builder first would be trust, this is
        measurement. The gate's rank-agreement term applies on every split
        too, but it only GATES -- the ratio arbitrating among passers stays
        pure mse/baseline (rank-unfaithful passers were the measured
        failure, not mis-ranking among passers).

        Only the most recent ``max_training_rows`` measurements take part:
        see that field for why a refresh must not grow with the run.
        """

        self.telemetry.refreshes += 1
        self._predict = None
        data = list(evaluated)
        if len(data) > self.max_training_rows:
            data = data[-self.max_training_rows:]
        best: Optional[Tuple[float, str, str, SurrogateBuilder]] = None
        for name, authored_by, builder in self.builders:
            ratios = []
            failed = False
            for split in range(self.validation_splits):
                verdict = validate_surrogate(
                    builder, data, specs, seed=seed + split * 7919)
                if not verdict.passed:
                    failed = True
                    break
                per = [
                    verdict.per_objective_mse[objective]
                    / verdict.baseline_mse[objective]
                    for objective in verdict.per_objective_mse
                    if verdict.baseline_mse[objective] > 0.0
                ]
                ratios.append(sum(per) / len(per) if per else 1.0)
            if failed:
                self.telemetry.rejected_validation += 1
                continue
            ratio = statistics.median(ratios) if ratios else 1.0
            if best is None or ratio < best[0]:
                best = (ratio, name, authored_by, builder)
        # Revise only when the rules measurably beat the artifact -- a refresh
        # where nothing passes the gate carries no feedback a revision could
        # use, and the revision budget is small.
        llm_won = best is not None and best[2] == "llm"
        if (best is not None and not llm_won and self.revise is not None
                and self.telemetry.revisions < self.max_revisions
                and any(authored_by == "llm"
                        for _n, authored_by, _b in self.builders)):
            self.telemetry.revisions += 1
            try:
                replacement = self.revise(data, specs)
            except Exception:
                replacement = None
            if replacement is not None:
                self.telemetry.revisions_accepted += 1
                rebuilt = []
                swapped = False
                for entry in self.builders:
                    if not swapped and entry[1] == "llm":
                        rebuilt.append(tuple(replacement))
                        swapped = True
                    else:
                        rebuilt.append(entry)
                self.builders = tuple(rebuilt)

        if best is None:
            return False
        _ratio, name, authored_by, builder = best
        try:
            self._predict = builder(data, specs)
        except Exception:
            self.telemetry.screen_failures += 1
            return False
        self._name = name
        self.authored_by = authored_by
        self.telemetry.validated += 1
        if authored_by == "llm":
            self.telemetry.chosen_llm += 1
        else:
            self.telemetry.chosen_rule += 1
        return True

    def screen(
        self,
        pool: Sequence[Config],
        population_objectives: Sequence[Mapping[str, float]],
        specs: Sequence[ObjectiveSpec],
    ) -> Optional[ScreenReport]:
        if self._predict is None:
            return None
        self.telemetry.screens += 1
        try:
            report = screen_offspring(
                pool, population_objectives, specs, self._predict,
                surrogate_name=self._name,
            )
        except Exception:
            self.telemetry.screen_failures += 1
            return None
        if report is None:
            self.telemetry.screen_failures += 1
            return None
        self.telemetry.virtual_evaluations += report.virtual_evaluations
        return report
