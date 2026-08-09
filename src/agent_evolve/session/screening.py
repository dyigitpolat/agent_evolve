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
from agent_evolve.core.results import dominates
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

    field = clean + [dict(objectives) for objectives in population_objectives]
    ranks = [
        sum(1 for other in field if dominates(other, row, specs))
        for row in clean
    ]
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
    ) -> None:
        if pool_factor < 2:
            raise ValueError(f"pool_factor must be at least 2, got {pool_factor}")
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
        measurement.
        """

        self.telemetry.refreshes += 1
        self._predict = None
        best: Optional[Tuple[float, str, str, SurrogateBuilder]] = None
        for name, authored_by, builder in self.builders:
            ratios = []
            failed = False
            for split in range(self.validation_splits):
                verdict = validate_surrogate(
                    builder, evaluated, specs, seed=seed + split * 7919)
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
                replacement = self.revise(list(evaluated), specs)
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
            self._predict = builder(list(evaluated), specs)
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
