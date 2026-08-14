"""Virtual pre-screening: order a pool of offspring before paying for any.

This module is deliberately starved: :func:`screen_offspring` receives
configurations and objective vectors and a predictor -- never the problem,
never the evaluation cache -- so "the surrogate cannot spend budget" is a
property of the import graph, not a convention. The only route from here to
a real evaluation is that the loop measures the candidates this module
merely ordered.

Because ORDERING is the whole of what this module consumes, it validates its
surrogates under :data:`~agent_evolve.policies.surrogate.ORDERING_GATE`:
rank fidelity rejects, and the error ratio against the train-mean predictor
is computed for arbitration among passers. The screen is the reason that
distinction exists -- under a gate that also rejected on magnitude it went
dark on the venues where saving an evaluation is worth anything, ordering 7
of 186 generations on an expensive venue against 54% on a cheap one.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.surrogate import (
    ORDERING_GATE,
    GatePolicy,
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

    This function consumes an ORDER and nothing else: the returned
    ``predicted`` rows are telemetry, and the loop reads only ``order``. That
    is why the gate this screen validates under is
    :data:`~agent_evolve.policies.surrogate.ORDERING_GATE` -- rank fidelity
    is what the output can be wrong about, and a calibration test on
    magnitudes nobody reads can only reject artifacts that would have
    ordered correctly.
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

    #: Why a builder was rejected, counted per (builder, split) verdict.
    #: Without this a campaign cannot tell "the model is not predictive"
    #: from "the gate never had enough data to look" -- which is exactly the
    #: distinction that turned out to decide whether the mechanism runs at
    #: all -- and every study that needed it had to monkeypatch the gate.
    REJECTIONS = ("rejected_insufficient_rows", "rejected_insufficient_holdout",
                  "rejected_rank", "rejected_error", "rejected_builder_failed",
                  "rejected_no_predictions", "rejected_bad_prediction")

    #: The cheap fidelity's own counters, kept BESIDE the charged ones and
    #: never added to them. `proxy_rows_used` is how many gate rows came from
    #: the cheap evaluator on the last refresh; `chosen_proxy` counts the
    #: generations the cheap fidelity itself won the gate and did the
    #: ordering.
    PROXY = ("proxy_rows_used", "chosen_proxy")

    __slots__ = ("refreshes", "validated", "rejected_validation", "screens",
                 "screen_failures", "virtual_evaluations", "chosen_llm",
                 "chosen_rule", "revisions", "revisions_accepted",
                 "gate_calls") + REJECTIONS + PROXY

    def __init__(self) -> None:
        for name in self.__slots__:
            setattr(self, name, 0)

    def record(self, verdict: Any) -> None:
        """Count one gate verdict, passed or rejected and why."""

        self.gate_calls += 1
        if verdict.passed:
            return
        name = f"rejected_{verdict.reason or 'unknown'}"
        if name in self.REJECTIONS:
            setattr(self, name, getattr(self, name) + 1)

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
        gate: GatePolicy = ORDERING_GATE,
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
        if not isinstance(gate, GatePolicy):
            raise TypeError(f"gate must be a GatePolicy, got {type(gate).__name__}")
        self.builders = tuple(builders)
        self.pool_factor = int(pool_factor)
        self.exploration_floor = float(exploration_floor)
        self.validation_splits = int(validation_splits)
        #: What this consumer relies on, declared to the gate rather than
        #: assumed by it. The screen consumes an ORDER (`screen_offspring`
        #: reads `report.order` and nothing else), so rank fidelity is the
        #: hard term and the error ratio arbitrates among passers. Overriding
        #: this with a prediction-purpose policy restores the historical
        #: behaviour, at the historical cost: a magnitude test on a small
        #: holdout rejects most of the artifacts that would have ordered
        #: correctly.
        self.gate = gate
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
        #: The cheap fidelity, if the problem has one and the loop attached
        #: it. `_proxy_rows` is gate EVIDENCE bought at that fidelity: it is
        #: keyed so a real measurement always supersedes it, it never reaches
        #: the archive, the population or the budget, and it is counted in
        #: the source's own ledger.
        self._proxy: Any = None
        self._proxy_mode = "off"
        self._proxy_rows: Dict[str, Tuple[Config, Mapping[str, float]]] = {}

    # ------------------------------------------------------------------ proxy
    def attach_proxy(self, source: Any, *, mode: str = "rows") -> None:
        """Let this screen spend a CHEAPER evaluation fidelity.

        ``mode="rows"`` -- the cheap evaluator buys gate EVIDENCE: rows the
        campaign could not afford at full price, so the gate can reach a
        verdict at budgets where the run has not yet measured its minimum
        number of rows. This is the term that closes "too few rows", which is
        a property of the budget and which no gate policy can fix.

        ``mode="screen"`` -- the cheap evaluator competes AS a surrogate,
        first in the builder order, and is cross-validated against the run's
        own real measurements exactly like an authored artifact. This is the
        term that can close a rank veto, and only if the cheap fidelity
        really does rank the expensive one.

        ``mode="both"`` -- both. ``mode="off"`` -- neither, and the screen is
        then byte-identical to a screen with no proxy at all.
        """

        if mode not in ("off", "rows", "screen", "both"):
            raise ValueError(
                "proxy mode must be 'off', 'rows', 'screen' or 'both', "
                f"got {mode!r}")
        if mode == "off" or source is None:
            self._proxy, self._proxy_mode = None, "off"
            return
        self._proxy = source
        self._proxy_mode = mode
        if mode in ("screen", "both"):
            from agent_evolve.session.fidelity import proxy_fidelity_builder
            name = f"proxy:{getattr(source, 'name', 'proxy')}"
            if not any(entry[0] == name for entry in self.builders):
                self.builders = ((name, "proxy", proxy_fidelity_builder(source)),
                                 ) + self.builders

    def prime(self, candidates: Sequence[Config],
              measured_keys: Sequence[str] = ()) -> int:
        """Buy gate evidence at the cheap fidelity. Returns rows added.

        Called by the loop with the candidates it is about to consider. Rows
        already measured for real are skipped, and any cheap row whose
        candidate later gets measured is dropped by :meth:`refresh` -- cheap
        evidence exists to fill a hole, never to outvote the real thing.
        """

        if self._proxy is None or self._proxy_mode not in ("rows", "both"):
            return 0
        known = set(measured_keys)
        added = 0
        for config, values in self._proxy.rows(candidates, exclude=known):
            token = self._proxy.key(config)
            if token in self._proxy_rows:
                continue
            self._proxy_rows[token] = (config, values)
            added += 1
        self._proxy.ledger.rows_used = len(self._proxy_rows)
        return added

    def refresh(
        self,
        evaluated: Sequence[Tuple[Config, Mapping[str, float]]],
        specs: Sequence[ObjectiveSpec],
        *,
        seed: int = 0,
    ) -> bool:
        """Re-arbitrate: today's data decides WHO may screen, if anyone.

        Every builder is validated on ``validation_splits`` INDEPENDENT
        re-partitions of today's data and must pass ``self.gate`` on EVERY
        one; among the survivors, the lowest median mse/baseline ratio wins
        the generation. The all-splits requirement is the variance guard the
        ladder1 E2 row demanded: a high-variance authored artifact can pass
        one partition by luck and then mis-screen mid-run -- measured at the
        cheapest scale, where authored screening HURT the endpoint under the
        single-split gate. Surviving every re-partition of today's data is
        the in-loop generalization of "pass on both frozen datasets
        independently"; under cross-validation each split already scores the
        artifact on every row, so what the splits vary is which rows it was
        FITTED on, which is the instability the guard exists to catch.
        Best-passing across splits stays the arbitration: listing the
        authored builder first would be trust, this is measurement. The
        gate's rank-agreement term applies on every split too, but it only
        GATES -- the ratio arbitrating among passers stays pure mse/baseline
        (rank-unfaithful passers were the measured failure, not mis-ranking
        among passers), and under the ordering purpose that ratio is the ONLY
        thing the error term does.

        Only the most recent ``max_training_rows`` measurements take part:
        see that field for why a refresh must not grow with the run.
        """

        self.telemetry.refreshes += 1
        self._predict = None
        data = list(evaluated)
        # Cheap-fidelity evidence, where the campaign has none of its own.
        # REAL SUPERSEDES CHEAP, always and by key; the cheap rows are
        # appended after the real ones so the recent-window trim below drops
        # them first when the run has measured more than the window holds.
        if len(data) > self.max_training_rows:
            data = data[-self.max_training_rows:]
        proxy_used = 0
        if self._proxy is not None and self._proxy_rows:
            measured = {self._proxy.key(config) for config, _values in data}
            extra = [row for token, row in self._proxy_rows.items()
                     if token not in measured]
            room = self.max_training_rows - len(data)
            extra = extra[:room] if room > 0 else []
            proxy_used = len(extra)
            data = data + extra
        self.telemetry.proxy_rows_used = proxy_used
        best: Optional[Tuple[float, str, str, SurrogateBuilder]] = None
        for name, authored_by, builder in self.builders:
            ratios = []
            failed = False
            for split in range(self.validation_splits):
                verdict = validate_surrogate(
                    builder, data, specs, policy=self.gate,
                    seed=seed + split * 7919)
                self.telemetry.record(verdict)
                if not verdict.passed:
                    failed = True
                    break
                per = list(verdict.mse_ratio.values())
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
        elif authored_by == "proxy":
            self.telemetry.chosen_proxy += 1
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
