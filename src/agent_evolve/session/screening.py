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
    """The pool, ordered by predicted worth. Indices address the caller's pool.

    ``screened_objectives`` names the objectives the order was actually
    computed over, and ``declared_objectives`` names the problem's. They
    differ when the gate certified the surrogate on only some of them. A
    consumer that reads ``order`` without reading these two is free to
    believe the pool was ranked on the whole problem when it was ranked on
    part of it, so both travel with the order rather than beside it.
    """

    order: Tuple[int, ...]
    predicted: Tuple[Mapping[str, float], ...]
    virtual_evaluations: int
    surrogate_name: str
    screened_objectives: Tuple[str, ...] = ()
    declared_objectives: Tuple[str, ...] = ()

    @property
    def partial(self) -> bool:
        """Was this order computed over a STRICT SUBSET of the objectives?"""

        return len(self.screened_objectives) < len(self.declared_objectives)


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
    objectives: Optional[Sequence[str]] = None,
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

    ``objectives`` restricts the domination test to the objectives the gate
    certified this surrogate for; ``None`` means all of them. **The excluded
    objectives are treated as UNKNOWN, not as satisfied**: they are neither
    read from the prediction nor compared, so a surrogate that emits nonsense
    on an objective it was not certified for cannot influence the order
    through it. That is a deliberate asymmetry with a cost, stated here
    because a caller must weigh it: domination over a subset is a STRICTER
    relation than domination over the whole (more pairs compare, fewer are
    incomparable), so a candidate that is excellent only on an excluded
    objective is dominated on the subset and sinks. The screen is therefore
    biased against exactly the trade-off it cannot see, and the caller's
    exploration floor -- not this function -- is what keeps unscreened picks
    in the generation (see :meth:`Screening.exploration_floor_for`).
    """

    if not pool:
        return None
    names = [s.name for s in specs]
    if objectives is None:
        screened = list(names)
    else:
        wanted = set(objectives)
        unknown = wanted - set(names)
        if unknown:
            raise ValueError(
                "objectives to screen on must be declared objectives; "
                f"{sorted(unknown)} are not among {names}")
        screened = [name for name in names if name in wanted]
    # Ordering on nothing is not ordering. A caller that reaches here with an
    # empty subset has a gate bug, and screening the pool by index would hide
    # it behind a plausible-looking order.
    if not screened:
        return None
    specs = [spec for spec in specs if spec.name in set(screened)]
    predictions = predict(list(pool))
    if predictions is None or len(predictions) != len(pool):
        return None
    clean: list[dict[str, float]] = []
    for predicted in predictions:
        row = {}
        for name in screened:
            value = predicted.get(name) if isinstance(predicted, Mapping) else None
            if (value is None or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                return None
            row[name] = float(value)
        clean.append(row)

    ranks = _dominator_counts(
        clean, [dict(measured) for measured in population_objectives], specs)
    if ranks is None:
        return None
    order = tuple(sorted(range(len(pool)), key=lambda i: (ranks[i], i)))
    return ScreenReport(
        order=order,
        predicted=tuple(clean),
        virtual_evaluations=len(pool),
        surrogate_name=surrogate_name,
        screened_objectives=tuple(screened),
        declared_objectives=tuple(names),
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
    #: ``rejected_unstable_subset`` is counted separately and is NOT a gate
    #: reason: every split passed, but they certified different objectives,
    #: so the artifact is not stably predictive on enough of them. It exists
    #: because a partial verdict makes that failure possible for the first
    #: time, and a campaign must be able to see it rather than read it as
    #: "the gate never had data".

    #: The prefix under which ``as_dict`` reports, per objective, how many
    #: screens ordered on it. A run that screened on two of three objectives
    #: says so here in a form no reader can mistake for "screened on all
    #: three", and it says it WITHOUT this module knowing any objective name.
    SCREENED_PREFIX = "screened_on:"

    __slots__ = ("refreshes", "validated", "rejected_validation", "screens",
                 "screen_failures", "virtual_evaluations", "chosen_llm",
                 "chosen_rule", "revisions", "revisions_accepted",
                 "gate_calls", "screens_full", "screens_partial",
                 "rejected_unstable_subset",
                 "_screened") + REJECTIONS + PROXY

    def __init__(self) -> None:
        for name in self.__slots__:
            setattr(self, name, 0)
        #: objective name -> screens whose order was computed over it.
        self._screened: Dict[str, int] = {}

    def record(self, verdict: Any) -> None:
        """Count one gate verdict, passed or rejected and why."""

        self.gate_calls += 1
        if verdict.passed:
            return
        name = f"rejected_{verdict.reason or 'unknown'}"
        if name in self.REJECTIONS:
            setattr(self, name, getattr(self, name) + 1)

    def record_screen(self, report: Any) -> None:
        """Count one screen, and WHICH objectives its order was computed over.

        This is a correctness requirement, not a nicety: an order over a
        subset is a different object from an order over the whole problem,
        and a run that cannot distinguish them can report an endpoint it
        cannot attribute.
        """

        if report.partial:
            self.screens_partial += 1
        else:
            self.screens_full += 1
        for name in report.screened_objectives:
            self._screened[name] = self._screened.get(name, 0) + 1

    def as_dict(self) -> dict[str, int]:
        counters = {name: getattr(self, name) for name in self.__slots__
                    if not name.startswith("_")}
        for name, count in sorted(self._screened.items()):
            counters[f"{self.SCREENED_PREFIX}{name}"] = count
        return counters


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
        unscreened_objective_floor: float = 1.0,
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
        if not 0.0 <= unscreened_objective_floor <= 1.0:
            raise ValueError(
                "unscreened_objective_floor must be in [0, 1], got "
                f"{unscreened_objective_floor}"
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
        #: How much of a generation is reserved from the screen when the gate
        #: certified the surrogate on only SOME objectives, expressed as a
        #: multiple of the share of objectives the screen is blind to.
        #:
        #: The screen orders by domination over the certified subset, and
        #: domination over a subset is a stricter relation than domination
        #: over the whole problem: a candidate that is excellent only on an
        #: excluded objective is dominated on the subset and sinks. So a
        #: partial screen is not merely less informed than a full one, it is
        #: SYSTEMATICALLY biased against the objectives it cannot see, and
        #: the flat 0.25 floor -- sized for a screen that might be wrong,
        #: not for one that is wrong in a known direction -- is not the right
        #: protection. At 1.0 (the default) the reserved share is the
        #: unscreened share of the objectives: 1/3 of the generation stays
        #: unscreened when 2 of 3 objectives are certified, 2/3 when 1 of 3
        #: is, and the ordinary ``exploration_floor`` still applies as a
        #: lower bound. At 0.0 a partial screen is treated exactly like a
        #: full one, which is the arm this default was measured against.
        self.unscreened_objective_floor = float(unscreened_objective_floor)
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
        #: The objectives the gate certified the installed surrogate for.
        #: ``None`` when nothing is installed. The screen orders on exactly
        #: these and treats the rest as unknown.
        self._objectives: Optional[Tuple[str, ...]] = None

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
        self._objectives = None
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
        names = [spec.name for spec in specs]
        required = self.gate.objectives_required(len(names))
        best: Optional[Tuple[Tuple[int, float], str, str,
                             SurrogateBuilder, Tuple[str, ...]]] = None
        for name, authored_by, builder in self.builders:
            verdicts = []
            failed = False
            for split in range(self.validation_splits):
                verdict = validate_surrogate(
                    builder, data, specs, policy=self.gate,
                    seed=seed + split * 7919)
                self.telemetry.record(verdict)
                if not verdict.passed:
                    failed = True
                    break
                verdicts.append(verdict)
            if failed:
                self.telemetry.rejected_validation += 1
                continue
            # The variance guard applies PER OBJECTIVE, because that is the
            # granularity the verdict now has. An artifact certified on
            # {area, latency} by one re-partition and on {area, energy} by
            # the next is stable on {area} alone -- it has not shown it can
            # order latency or energy across fits -- so the certified set is
            # the INTERSECTION and it must still meet the policy's
            # requirement. Under the conjunction every passing split
            # certifies every objective, so the intersection is the whole set
            # and this is a no-op.
            certified = set(names)
            for verdict in verdicts:
                certified &= set(verdict.passing_objectives)
            scope = tuple(n for n in names if n in certified)
            if len(scope) < required or not scope:
                self.telemetry.rejected_unstable_subset += 1
                continue
            ratios = []
            for verdict in verdicts:
                per = [verdict.mse_ratio[n] for n in scope
                       if n in verdict.mse_ratio]
                ratios.append(sum(per) / len(per) if per else 1.0)
            ratio = statistics.median(ratios) if ratios else 1.0
            # More certified objectives beats a better error ratio: an
            # artifact that can order the whole problem is a different
            # instrument from one that can order a third of it, and the
            # ratio -- an average over whichever objectives each artifact
            # got certified on -- is not comparable across different scopes.
            # Under the conjunction every survivor has the same scope, so
            # this reduces to the historical "lowest median ratio wins".
            key = (-len(scope), ratio)
            if best is None or key < best[0]:
                best = (key, name, authored_by, builder, scope)
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
        _key, name, authored_by, builder, scope = best
        try:
            self._predict = builder(data, specs)
        except Exception:
            self.telemetry.screen_failures += 1
            return False
        self._objectives = scope
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
                objectives=self._objectives,
            )
        except Exception:
            self.telemetry.screen_failures += 1
            return None
        if report is None:
            self.telemetry.screen_failures += 1
            return None
        self.telemetry.virtual_evaluations += report.virtual_evaluations
        self.telemetry.record_screen(report)
        return report

    def exploration_floor_for(self, report: ScreenReport) -> float:
        """The share of the generation to keep away from THIS screen.

        ``exploration_floor`` when the screen ordered on the whole problem.
        When it ordered on a subset, the floor rises with the share of
        objectives it could not see, scaled by
        ``unscreened_objective_floor`` -- see that field for why a partial
        screen needs more protection than a full one rather than the same.
        The caller applies it; this class does not touch the budget.
        """

        declared = len(report.declared_objectives)
        screened = len(report.screened_objectives)
        if declared <= 0 or screened >= declared:
            return self.exploration_floor
        unscreened_share = (declared - screened) / declared
        return max(self.exploration_floor,
                   self.unscreened_objective_floor * unscreened_share)
