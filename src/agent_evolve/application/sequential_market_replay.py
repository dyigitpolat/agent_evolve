"""Provider-free sequential replay over sealed markets with real outcomes.

A market record freezes one already-materialized candidate market together
with the archive it faced, an exact hypervolume reference, and the real
evaluated outcomes.  The replay harness then re-runs any allocation policy
under the honest information boundary: at every step the policy sees ONLY
the outcomes of candidates it already selected, selects one next candidate,
and only then has that candidate's outcome revealed.

Outputs are causal receipts: per-step selections with propensities, exact
conditional marginal hypervolume gains, cumulative realized gain, and
regret against the exact oracle-K subset enumerated over evaluated
candidates only.  Oracle ceilings are retrospective upper bounds over
already-generated candidates, never prospective performance claims.

Reference baseline policies (uniform random, native-rank round-robin,
frozen-score top-K, and the V70-style frozen lane-heads walk) are included
for comparison, plus an adapter that drives the composed V8-lite policy.
No provider, workload, objective-name, or model branch exists anywhere.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    ObjectivePoint,
    ObservedConversionOutcome,
    PositiveGainForecast,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionDescriptor,
    AdaptiveActionOutcome,
)
from agent_evolve.application.v8lite_allocation_policy import (
    V8LiteAllocationPolicy,
)
from agent_evolve.domain.patch import require_sha256

SEQUENTIAL_MARKET_REPLAY_ID = "sequential_market_replay"
SEQUENTIAL_MARKET_REPLAY_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_MARKET_DOMAIN = b"agent-evolve:sequential-replay-market:v1\x00"
_RESULT_DOMAIN = b"agent-evolve:sequential-replay-result:v1\x00"

#: Exhaustive oracle enumeration guard: beyond this many subsets the
#: harness refuses rather than silently approximating.
DEFAULT_ORACLE_SUBSET_LIMIT = 250_000
#: Union hypervolume is computed by float slicing, so a point lying
#: exactly on the archive frontier can yield a difference of a few
#: ulps instead of exact zero.  Gains at or below this epsilon (in
#: normalized hypervolume units; real recorded gains are >= 1e-5) are
#: treated as exactly zero everywhere: replay marginals, oracle
#: positivity, and the exact gain port.
HYPERVOLUME_GAIN_EPSILON = 1.0e-12


def _clamped_gain(value: float) -> float:
    return value if value > HYPERVOLUME_GAIN_EPSILON else 0.0
#: Neutral prior score used when a candidate carries no frozen score but
#: an adapter needs a probability-typed prior.
NEUTRAL_PRIOR_SCORE = 0.5


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _stable_unit_interval(*parts: object) -> float:
    payload = _canonical_json(list(parts))
    numerator = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return numerator / float(2**64)


def exact_hypervolume(
    points: tuple[tuple[float, ...], ...],
    reference: tuple[float, ...],
) -> float:
    """Exact minimization hypervolume by recursive objective slicing."""

    if type(reference) is not tuple or not reference:
        raise ValueError("reference must be a non-empty exact tuple")
    kept = sorted(
        {
            point
            for point in points
            if len(point) == len(reference)
            and all(
                value < bound
                for value, bound in zip(point, reference, strict=True)
            )
        }
    )
    if not kept:
        return 0.0
    if len(reference) == 1:
        return reference[0] - min(point[0] for point in kept)
    kept.sort(key=lambda point: point[-1])
    total = 0.0
    for index, point in enumerate(kept):
        z_low = point[-1]
        z_high = (
            kept[index + 1][-1]
            if index + 1 < len(kept)
            else reference[-1]
        )
        if z_high > z_low:
            slab = exact_hypervolume(
                tuple(value[:-1] for value in kept[: index + 1]),
                reference[:-1],
            )
            total += slab * (z_high - z_low)
    return total


def _point_values(
    point: ObjectivePoint,
    metric_ids: tuple[str, ...],
) -> tuple[float, ...]:
    mapping = dict(point)
    if tuple(sorted(mapping)) != metric_ids:
        raise ValueError("objective point uses a foreign metric frame")
    return tuple(mapping[metric_id] for metric_id in metric_ids)


@dataclass(frozen=True, slots=True)
class MarketCandidateRecord:
    """One frozen market candidate with its real outcome, if any."""

    action_sha256: str
    engine_id: str
    native_rank: int
    frozen_score: float | None
    forecast: PositiveGainForecast | None
    evaluated: bool
    feasible: bool
    objectives: ObjectivePoint | None
    #: The candidate's parent objective point, known at proposal time.
    anchor_point: ObjectivePoint | None = None

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        _require_token(self.engine_id, name="engine_id")
        if type(self.native_rank) is not int or self.native_rank <= 0:
            raise ValueError("native_rank must be positive")
        if self.frozen_score is not None and (
            type(self.frozen_score) is not float
            or not math.isfinite(self.frozen_score)
        ):
            raise ValueError("frozen_score must be finite or None")
        if self.forecast is not None:
            if type(self.forecast) is not PositiveGainForecast:
                raise TypeError("forecast must be exact or None")
            self.forecast.__post_init__()
        if type(self.evaluated) is not bool or type(self.feasible) is not bool:
            raise TypeError("evaluated and feasible must be exact")
        if self.feasible and not self.evaluated:
            raise ValueError("a feasible candidate must be evaluated")
        if (self.objectives is not None) != self.feasible:
            raise ValueError(
                "objectives must be present exactly for feasible candidates"
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        record = {
            "action_sha256": self.action_sha256,
            "engine_id": self.engine_id,
            "native_rank": self.native_rank,
            "frozen_score_hex": (
                None
                if self.frozen_score is None
                else self.frozen_score.hex()
            ),
            "forecast": (
                None
                if self.forecast is None
                else self.forecast.to_record()
            ),
            "evaluated": self.evaluated,
            "feasible": self.feasible,
            "objectives": (
                None
                if self.objectives is None
                else [
                    {"metric_id": metric_id, "value_hex": value.hex()}
                    for metric_id, value in self.objectives
                ]
            ),
        }
        # Emitted only when present so a market recorded without parent
        # anchors keeps its pre-existing bytes and market_sha256.
        if self.anchor_point is not None:
            record["anchor_point"] = [
                {"metric_id": metric_id, "value_hex": value.hex()}
                for metric_id, value in self.anchor_point
            ]
        return record


@dataclass(frozen=True, slots=True)
class MarketRecord:
    """Minimal sealed-market view sufficient for allocator replay."""

    market_id: str
    archive_points: tuple[ObjectivePoint, ...]
    hv_reference_point: ObjectivePoint
    candidates: tuple[MarketCandidateRecord, ...]
    market_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.market_id, name="market_id")
        if (
            type(self.hv_reference_point) is not tuple
            or not self.hv_reference_point
            or self.hv_reference_point
            != tuple(sorted(self.hv_reference_point))
        ):
            raise ValueError(
                "hv_reference_point must be non-empty and canonical"
            )
        metric_ids = self.metric_ids
        if type(self.archive_points) is not tuple or not self.archive_points:
            raise ValueError("archive_points must be non-empty")
        for point in self.archive_points:
            _point_values(point, metric_ids)
        if type(self.candidates) is not tuple or not self.candidates:
            raise ValueError("candidates must be non-empty")
        seen: set[str] = set()
        lanes: dict[str, list[int]] = {}
        for value in self.candidates:
            if type(value) is not MarketCandidateRecord:
                raise TypeError("candidates must be exact market records")
            value.__post_init__()
            if value.action_sha256 in seen:
                raise ValueError("candidate identities repeat")
            seen.add(value.action_sha256)
            lanes.setdefault(value.engine_id, []).append(
                value.native_rank
            )
            if value.objectives is not None:
                _point_values(value.objectives, metric_ids)
        for engine_id, ranks in lanes.items():
            if sorted(ranks) != list(range(1, len(ranks) + 1)):
                raise ValueError(
                    f"engine {engine_id} ranks must be contiguous"
                )
        object.__setattr__(
            self,
            "market_sha256",
            _hash(
                _MARKET_DOMAIN,
                {
                    "schema_version": 1,
                    "market_id": self.market_id,
                    "hv_reference_point": [
                        {
                            "metric_id": metric_id,
                            "value_hex": value.hex(),
                        }
                        for metric_id, value in self.hv_reference_point
                    ],
                    "archive_points": [
                        [
                            {
                                "metric_id": metric_id,
                                "value_hex": value.hex(),
                            }
                            for metric_id, value in point
                        ]
                        for point in self.archive_points
                    ],
                    "candidates": [
                        value.to_record() for value in self.candidates
                    ],
                },
            ),
        )

    @property
    def metric_ids(self) -> tuple[str, ...]:
        return tuple(
            metric_id for metric_id, _value in self.hv_reference_point
        )

    def candidate(self, action_sha256: str) -> MarketCandidateRecord:
        for value in self.candidates:
            if value.action_sha256 == action_sha256:
                return value
        raise ValueError("candidate is outside the market")

    def hypervolume(
        self,
        extra_points: tuple[ObjectivePoint, ...] = (),
    ) -> float:
        metric_ids = self.metric_ids
        reference = _point_values(self.hv_reference_point, metric_ids)
        return exact_hypervolume(
            tuple(
                _point_values(point, metric_ids)
                for point in (*self.archive_points, *extra_points)
            ),
            reference,
        )


class ExactHypervolumeGainPort:
    """Module-level gain port valuing points against arbitrary archives."""

    utility_id = "exact_hypervolume_gain"
    utility_version = 1

    def __init__(self, hv_reference_point: ObjectivePoint) -> None:
        self._reference = hv_reference_point
        self._metric_ids = tuple(
            metric_id for metric_id, _value in hv_reference_point
        )
        self.definition_sha256 = _hash(
            _MARKET_DOMAIN,
            {
                "utility": self.utility_id,
                "reference": [
                    {"metric_id": metric_id, "value_hex": value.hex()}
                    for metric_id, value in hv_reference_point
                ],
            },
        )

    def marginal_archive_gain(
        self,
        archive_points: tuple[ObjectivePoint, ...],
        objective_point: ObjectivePoint,
    ) -> float:
        reference = _point_values(self._reference, self._metric_ids)
        base = tuple(
            _point_values(point, self._metric_ids)
            for point in archive_points
        )
        with_point = (
            *base,
            _point_values(objective_point, self._metric_ids),
        )
        return _clamped_gain(
            float(
                exact_hypervolume(with_point, reference)
                - exact_hypervolume(base, reference)
            )
        )


@dataclass(frozen=True, slots=True)
class ReplaySelection:
    """One policy answer: which candidate, at what propensity."""

    action_sha256: str
    selection_propensity: float
    evidence: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        if (
            type(self.selection_propensity) is not float
            or not math.isfinite(self.selection_propensity)
            or not 0.0 < self.selection_propensity <= 1.0
        ):
            raise ValueError(
                "selection_propensity must be a positive probability"
            )


@dataclass(frozen=True, slots=True)
class ReplayStepReceipt:
    step_index: int
    action_sha256: str
    selection_propensity: float
    evaluated: bool
    feasible: bool
    imputed_zero_outcome: bool
    marginal_gain: float
    cumulative_gain: float

    def to_record(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "action_sha256": self.action_sha256,
            "selection_propensity_hex": (
                self.selection_propensity.hex()
            ),
            "evaluated": self.evaluated,
            "feasible": self.feasible,
            "imputed_zero_outcome": self.imputed_zero_outcome,
            "marginal_gain_hex": self.marginal_gain.hex(),
            "cumulative_gain_hex": self.cumulative_gain.hex(),
        }


@dataclass(frozen=True, slots=True)
class ReplayResult:
    market_sha256: str
    policy_id: str
    budget: int
    restricted_to_evaluated: bool
    receipts: tuple[ReplayStepReceipt, ...]
    realized_gain: float
    oracle_gain: float
    oracle_subset: tuple[str, ...]
    regret: float
    result_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result_sha256",
            _hash(_RESULT_DOMAIN, self.to_record()),
        )

    @property
    def selected_action_sha256s(self) -> tuple[str, ...]:
        return tuple(
            value.action_sha256 for value in self.receipts
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "market_sha256": self.market_sha256,
            "policy_id": self.policy_id,
            "budget": self.budget,
            "restricted_to_evaluated": self.restricted_to_evaluated,
            "receipts": [value.to_record() for value in self.receipts],
            "realized_gain_hex": self.realized_gain.hex(),
            "oracle_gain_hex": self.oracle_gain.hex(),
            "oracle_subset": list(self.oracle_subset),
            "regret_hex": self.regret.hex(),
            "oracle_is_retrospective_upper_bound": True,
        }


class UniformRandomReplayPolicy:
    """Seeded uniform selection over the selectable support."""

    def __init__(self, seed: int) -> None:
        self.policy_id = "replay_baseline.uniform_random"
        self._seed = seed

    def select(
        self,
        *,
        record: MarketRecord,
        revealed: tuple[ReplayStepReceipt, ...],
        selectable_action_sha256s: tuple[str, ...],
        step_index: int,
        budget: int,
    ) -> ReplaySelection:
        # The draw must not condition on unrevealed outcomes, so it
        # hashes only the outcome-blind market identity and support.
        draw = _stable_unit_interval(
            self._seed,
            record.market_id,
            "uniform_random",
            step_index,
            list(selectable_action_sha256s),
        )
        support = tuple(sorted(selectable_action_sha256s))
        chosen = support[
            min(int(draw * len(support)), len(support) - 1)
        ]
        return ReplaySelection(
            action_sha256=chosen,
            selection_propensity=1.0 / len(support),
        )


class NativeRankRoundRobinReplayPolicy:
    """Round-robin engines in canonical order; best native rank first."""

    def __init__(self) -> None:
        self.policy_id = "replay_baseline.native_rank_round_robin"

    def select(
        self,
        *,
        record: MarketRecord,
        revealed: tuple[ReplayStepReceipt, ...],
        selectable_action_sha256s: tuple[str, ...],
        step_index: int,
        budget: int,
    ) -> ReplaySelection:
        selectable = set(selectable_action_sha256s)
        engines = sorted(
            {
                value.engine_id
                for value in record.candidates
                if value.action_sha256 in selectable
            }
        )
        engine_id = engines[step_index % len(engines)]
        chosen = min(
            (
                value
                for value in record.candidates
                if value.action_sha256 in selectable
                and value.engine_id == engine_id
            ),
            key=lambda value: (value.native_rank, value.action_sha256),
        )
        return ReplaySelection(
            action_sha256=chosen.action_sha256,
            selection_propensity=1.0,
        )


class FrozenScoreTopKReplayPolicy:
    """Descending global frozen score; native rank breaks ties."""

    def __init__(self) -> None:
        self.policy_id = "replay_baseline.frozen_score_topk"

    def select(
        self,
        *,
        record: MarketRecord,
        revealed: tuple[ReplayStepReceipt, ...],
        selectable_action_sha256s: tuple[str, ...],
        step_index: int,
        budget: int,
    ) -> ReplaySelection:
        selectable = set(selectable_action_sha256s)
        chosen = min(
            (
                value
                for value in record.candidates
                if value.action_sha256 in selectable
            ),
            key=lambda value: (
                -(
                    value.frozen_score
                    if value.frozen_score is not None
                    else NEUTRAL_PRIOR_SCORE
                ),
                value.native_rank,
                value.action_sha256,
            ),
        )
        return ReplaySelection(
            action_sha256=chosen.action_sha256,
            selection_propensity=1.0,
        )


class LaneHeadsReplayPolicy:
    """V70 D4 behavior: one frozen-score lane head per engine, cycled."""

    def __init__(self) -> None:
        self.policy_id = "replay_baseline.lane_heads"

    def select(
        self,
        *,
        record: MarketRecord,
        revealed: tuple[ReplayStepReceipt, ...],
        selectable_action_sha256s: tuple[str, ...],
        step_index: int,
        budget: int,
    ) -> ReplaySelection:
        def frozen(value: MarketCandidateRecord) -> float:
            return (
                value.frozen_score
                if value.frozen_score is not None
                else NEUTRAL_PRIOR_SCORE
            )

        universe = tuple(
            sorted(
                (
                    value
                    for value in record.candidates
                    if value.action_sha256
                    in set(selectable_action_sha256s)
                    or any(
                        value.action_sha256 == item.action_sha256
                        for item in revealed
                    )
                ),
                key=lambda value: value.action_sha256,
            )
        )
        by_engine: dict[str, list[MarketCandidateRecord]] = {}
        for value in universe:
            by_engine.setdefault(value.engine_id, []).append(value)
        for values in by_engine.values():
            values.sort(
                key=lambda value: (
                    -frozen(value),
                    value.native_rank,
                    value.action_sha256,
                )
            )
        engine_order = sorted(
            by_engine,
            key=lambda engine_id: (
                -frozen(by_engine[engine_id][0]),
                engine_id,
            ),
        )
        walk = []
        depth = 0
        while len(walk) < len(universe):
            for engine_id in engine_order:
                if depth < len(by_engine[engine_id]):
                    walk.append(by_engine[engine_id][depth])
            depth += 1
        selectable = set(selectable_action_sha256s)
        chosen = next(
            value
            for value in walk
            if value.action_sha256 in selectable
        )
        return ReplaySelection(
            action_sha256=chosen.action_sha256,
            selection_propensity=1.0,
        )


class V8LiteReplayPolicy:
    """Drive the composed V8-lite policy inside the replay boundary.

    The eligible universe is the harness's fixed selectable support plus
    already-selected candidates; native ranks are re-ranked densely
    within that universe so descriptor lane invariants hold.  Candidates
    without a frozen score receive the neutral prior score.
    """

    def __init__(
        self,
        policy: V8LiteAllocationPolicy,
        *,
        frozen_fit_training_run_count: int = 0,
        prior_conversion_outcomes: tuple[
            ObservedConversionOutcome,
            ...,
        ] = (),
    ) -> None:
        if type(policy) is not V8LiteAllocationPolicy:
            raise TypeError("policy must be an exact V8-lite policy")
        policy.__post_init__()
        self.policy_id = "replay_adapter.v8lite"
        self._policy = policy
        self._frozen_fit_training_run_count = (
            frozen_fit_training_run_count
        )
        self._prior_conversion_outcomes = prior_conversion_outcomes

    @staticmethod
    def _phenotype(action_sha256: str) -> str:
        return hashlib.sha256(
            f"phenotype:{action_sha256}".encode("ascii")
        ).hexdigest()

    @staticmethod
    def _outcome_blind_request_sha256(
        record: MarketRecord,
        universe_ids: tuple[str, ...],
    ) -> str:
        """Cutoff identity that never conditions on unrevealed outcomes."""

        return hashlib.sha256(
            _canonical_json(
                [
                    "replay-request",
                    record.market_id,
                    list(universe_ids),
                ]
            )
        ).hexdigest()

    def _descriptors(
        self,
        record: MarketRecord,
        universe_ids: tuple[str, ...],
    ) -> tuple[AdaptiveActionDescriptor, ...]:
        universe = tuple(
            record.candidate(value) for value in universe_ids
        )
        by_engine: dict[str, list[MarketCandidateRecord]] = {}
        for value in universe:
            by_engine.setdefault(value.engine_id, []).append(value)
        descriptors: list[AdaptiveActionDescriptor] = []
        for engine_id in sorted(by_engine):
            lane = sorted(
                by_engine[engine_id],
                key=lambda value: (
                    value.native_rank,
                    value.action_sha256,
                ),
            )
            for dense_rank, value in enumerate(lane, start=1):
                prior = (
                    value.frozen_score
                    if value.frozen_score is not None
                    else NEUTRAL_PRIOR_SCORE
                )
                descriptors.append(
                    AdaptiveActionDescriptor(
                        action_sha256=value.action_sha256,
                        phenotype_sha256=self._phenotype(
                            value.action_sha256
                        ),
                        lane_id=engine_id,
                        operator_id=engine_id,
                        native_rank=dense_rank,
                        lane_size=len(lane),
                        prior_score=float(
                            min(max(prior, 0.0), 1.0)
                        ),
                        parent_generated_in_current_run=False,
                    )
                )
        return tuple(descriptors)

    def select(
        self,
        *,
        record: MarketRecord,
        revealed: tuple[ReplayStepReceipt, ...],
        selectable_action_sha256s: tuple[str, ...],
        step_index: int,
        budget: int,
    ) -> ReplaySelection:
        universe_ids = tuple(
            sorted(
                {
                    *selectable_action_sha256s,
                    *(value.action_sha256 for value in revealed),
                }
            )
        )
        descriptors = self._descriptors(record, universe_ids)
        request_sha256 = self._outcome_blind_request_sha256(
            record,
            universe_ids,
        )
        pilot_width = self._policy.pilot_width_for(
            evaluation_slots=budget,
            engine_count=len(
                {value.lane_id for value in descriptors}
            ),
        )
        if pilot_width <= 0:
            raise ValueError("replay budget leaves no pilot")
        selected_ids = tuple(
            sorted(value.action_sha256 for value in revealed)
        )
        outcome_by_action = {
            value.action_sha256: value for value in revealed
        }
        outcomes = tuple(
            AdaptiveActionOutcome(
                action_sha256=action_sha256,
                evaluation_sha256=hashlib.sha256(
                    f"replay-evaluation:{action_sha256}".encode(
                        "ascii"
                    )
                ).hexdigest(),
                feasible=outcome_by_action[action_sha256].feasible,
                marginal_archive_gain=(
                    outcome_by_action[action_sha256].marginal_gain
                ),
            )
            for action_sha256 in selected_ids
        )
        if step_index < pilot_width:
            if self._policy.revision_2:
                decision = self._policy.design_pilot_seat(
                    residual_request_sha256=request_sha256,
                    actions=descriptors,
                    evaluation_slots=budget,
                    selected_action_sha256s=selected_ids,
                    outcomes=outcomes,
                )
                return ReplaySelection(
                    action_sha256=(
                        decision.selected_action_sha256s[0]
                    ),
                    selection_propensity=(
                        decision.selection_propensity
                    ),
                    evidence={"phase": decision.phase},
                )
            decision = self._policy.design_pilot(
                residual_request_sha256=request_sha256,
                actions=descriptors,
                evaluation_slots=budget,
            )
            seat = decision.pilot_design.seats[step_index]
            return ReplaySelection(
                action_sha256=seat.selected_action_sha256,
                selection_propensity=seat.selection_propensity,
                evidence={"phase": decision.phase},
            )
        pilot_ids = tuple(
            sorted(
                value.action_sha256
                for value in revealed[:pilot_width]
            )
        )
        pilot_points = tuple(
            record.candidate(value).objectives
            for value in pilot_ids
            if record.candidate(value).objectives is not None
        )
        diagnostic_joint_gain = _clamped_gain(
            float(
                record.hypervolume(pilot_points)
                - record.hypervolume()
            )
        )
        forecasts = tuple(
            (value.action_sha256, value.forecast)
            for value in (
                record.candidate(item) for item in universe_ids
            )
            if value.forecast is not None
        )
        anchors = tuple(
            (value.action_sha256, value.anchor_point)
            for value in (
                record.candidate(item) for item in universe_ids
            )
            if value.anchor_point is not None
        )
        # The archive-geometry channel is defined against the CURRENT
        # archive, so a revision that declares it is handed the sealed
        # archive plus every point already revealed at this cutoff.  Every
        # earlier revision keeps the sealed archive exactly, so its replay
        # bytes are unchanged.
        archive_points = record.archive_points
        if getattr(self._policy, "revision_3", False):
            revealed_points = tuple(
                point
                for point in (
                    record.candidate(value.action_sha256).objectives
                    for value in revealed
                )
                if point is not None
            )
            archive_points = (*archive_points, *revealed_points)
        decision = self._policy.select_next(
            residual_request_sha256=request_sha256,
            actions=descriptors,
            evaluation_slots=budget,
            diagnostic_action_sha256s=pilot_ids,
            diagnostic_joint_gain=diagnostic_joint_gain,
            selected_action_sha256s=selected_ids,
            outcomes=outcomes,
            archive_points=archive_points,
            forecasts=forecasts,
            anchor_points=anchors,
            frozen_fit_training_run_count=(
                self._frozen_fit_training_run_count
            ),
            prior_conversion_outcomes=(
                self._prior_conversion_outcomes
            ),
        )
        return ReplaySelection(
            action_sha256=decision.selected_action_sha256s[0],
            selection_propensity=decision.selection_propensity,
            evidence={"phase": decision.phase},
        )


@dataclass(frozen=True, slots=True)
class SequentialMarketReplay:
    """Run one policy over one sealed market under the honest boundary."""

    oracle_subset_limit: int = DEFAULT_ORACLE_SUBSET_LIMIT

    def __post_init__(self) -> None:
        if (
            type(self.oracle_subset_limit) is not int
            or self.oracle_subset_limit <= 0
        ):
            raise ValueError("oracle_subset_limit must be positive")

    def oracle(
        self,
        record: MarketRecord,
        budget: int,
    ) -> tuple[float, tuple[str, ...]]:
        """Exact best size-<=budget subset over evaluated candidates.

        Enumeration runs over the deduplicated non-dominated positive
        subset, which is exact: a candidate with zero marginal
        hypervolume adds nothing to any union, and any member of an
        optimal subset can be replaced by a candidate that weakly
        dominates it without reducing the union.  The subset limit
        therefore applies to the reduced frontier, not the raw market.
        """

        record.__post_init__()
        if type(budget) is not int or budget <= 0:
            raise ValueError("budget must be positive")
        metric_ids = record.metric_ids
        base = record.hypervolume()
        unique_by_point: dict[
            tuple[float, ...],
            MarketCandidateRecord,
        ] = {}
        for value in record.candidates:
            if not value.feasible or value.objectives is None:
                continue
            if (
                _clamped_gain(
                    record.hypervolume((value.objectives,)) - base
                )
                <= 0.0
            ):
                continue
            point = _point_values(value.objectives, metric_ids)
            incumbent = unique_by_point.get(point)
            if (
                incumbent is None
                or value.action_sha256 < incumbent.action_sha256
            ):
                unique_by_point[point] = value
        frontier = tuple(
            candidate
            for point, candidate in unique_by_point.items()
            if not any(
                other != point
                and all(
                    o <= p
                    for o, p in zip(other, point, strict=True)
                )
                for other in unique_by_point
            )
        )
        subset_size = min(budget, len(frontier))
        if subset_size == 0:
            return 0.0, ()
        if (
            math.comb(len(frontier), subset_size)
            > self.oracle_subset_limit
        ):
            raise ValueError(
                "oracle enumeration exceeds the configured subset limit"
            )
        best_gain = 0.0
        best_subset: tuple[str, ...] = ()
        for subset in itertools.combinations(frontier, subset_size):
            gain = (
                record.hypervolume(
                    tuple(value.objectives for value in subset)
                )
                - base
            )
            key = tuple(
                sorted(value.action_sha256 for value in subset)
            )
            if gain > best_gain or (
                gain == best_gain and best_subset and key < best_subset
            ):
                best_gain = gain
                best_subset = key
        best_gain = _clamped_gain(float(best_gain))
        if best_gain == 0.0:
            return 0.0, ()
        return best_gain, best_subset

    def run(
        self,
        *,
        record: MarketRecord,
        policy: object,
        budget: int,
        restrict_to_evaluated: bool = True,
    ) -> ReplayResult:
        record.__post_init__()
        policy_id = getattr(policy, "policy_id", None)
        _require_token(str(policy_id), name="policy_id")
        if type(budget) is not int or budget <= 0:
            raise ValueError("budget must be positive")
        selectable = {
            value.action_sha256
            for value in record.candidates
            if not restrict_to_evaluated or value.evaluated
        }
        if budget > len(selectable):
            raise ValueError("budget exceeds the selectable market")
        receipts: list[ReplayStepReceipt] = []
        revealed_points: list[ObjectivePoint] = []
        cumulative = 0.0
        base_hv = record.hypervolume()
        for step_index in range(budget):
            selection = policy.select(
                record=record,
                revealed=tuple(receipts),
                selectable_action_sha256s=tuple(sorted(selectable)),
                step_index=step_index,
                budget=budget,
            )
            if type(selection) is not ReplaySelection:
                raise TypeError("policy must return an exact selection")
            selection.__post_init__()
            if selection.action_sha256 not in selectable:
                raise ValueError(
                    "policy selected outside the selectable support"
                )
            candidate = record.candidate(selection.action_sha256)
            imputed = not candidate.evaluated
            if candidate.feasible and candidate.objectives is not None:
                with_candidate = record.hypervolume(
                    (*revealed_points, candidate.objectives)
                )
                without_candidate = record.hypervolume(
                    tuple(revealed_points)
                )
                # Union hypervolume is monotone; sub-epsilon float
                # differences from a zero-add candidate are exact zero.
                marginal = _clamped_gain(
                    float(with_candidate - without_candidate)
                )
                revealed_points.append(candidate.objectives)
            else:
                marginal = 0.0
            cumulative = _clamped_gain(
                float(
                    record.hypervolume(tuple(revealed_points))
                    - base_hv
                )
            )
            selectable.discard(selection.action_sha256)
            receipts.append(
                ReplayStepReceipt(
                    step_index=step_index,
                    action_sha256=selection.action_sha256,
                    selection_propensity=(
                        selection.selection_propensity
                    ),
                    evaluated=candidate.evaluated,
                    feasible=candidate.feasible,
                    imputed_zero_outcome=imputed,
                    marginal_gain=marginal,
                    cumulative_gain=cumulative,
                )
            )
        oracle_gain, oracle_subset = self.oracle(record, budget)
        return ReplayResult(
            market_sha256=record.market_sha256,
            policy_id=str(policy_id),
            budget=budget,
            restricted_to_evaluated=restrict_to_evaluated,
            receipts=tuple(receipts),
            realized_gain=float(cumulative),
            oracle_gain=float(oracle_gain),
            oracle_subset=oracle_subset,
            regret=float(oracle_gain - cumulative),
        )


def _normalized_point(
    raw: dict[str, float],
    axes: tuple[dict[str, object], ...],
) -> ObjectivePoint:
    values: list[tuple[str, float]] = []
    for axis in axes:
        metric_id = str(axis["metric_id"])
        reference = float(axis["reference"])
        ideal = float(axis["ideal"])
        if reference == ideal:
            raise ValueError("axis reference must differ from ideal")
        values.append(
            (
                metric_id,
                float(
                    (float(raw[metric_id]) - ideal)
                    / (reference - ideal)
                ),
            )
        )
    return tuple(sorted(values))


#: Frozen-score keys consumed from corpus records, in priority order.
#: Values are min-max normalized per market (higher is better) so they
#: are usable both as an ordering and as a probability-typed prior.
CORPUS_FROZEN_SCORE_KEYS = ("frozen_prior_score", "raw")


def _corpus_engine_id(raw: dict[str, object]) -> str:
    lane = raw.get("lane")
    if isinstance(lane, dict) and lane.get("lane_id"):
        return str(lane["lane_id"])
    for key in ("expert_id", "family"):
        if raw.get(key):
            return str(raw[key])
    return "engine.unattributed"


def _corpus_action_sha256(
    market_id: str,
    raw: dict[str, object],
) -> str:
    value = raw.get("action_sha256")
    if isinstance(value, str) and len(value) == 64:
        return value
    identity = [
        raw.get(key)
        for key in (
            "candidate_id",
            "proposal_id",
            "locus_key",
            "option_id",
            "native_rank",
        )
        if raw.get(key) is not None
    ]
    if not identity:
        raise ValueError("corpus candidate carries no identity")
    return hashlib.sha256(
        _canonical_json(
            [
                "replay-candidate-id",
                market_id,
                [str(part) for part in identity],
            ]
        )
    ).hexdigest()


def _corpus_frozen_raw(raw: dict[str, object]) -> float | None:
    frozen_scores = raw.get("frozen_scores")
    if not isinstance(frozen_scores, dict):
        return None
    for key in CORPUS_FROZEN_SCORE_KEYS:
        value = frozen_scores.get(key)
        if isinstance(value, (int, float)) and math.isfinite(
            float(value)
        ):
            return float(value)
    return None


def _corpus_forecast(
    raw: dict[str, object],
    axes: tuple[dict[str, object], ...],
    metric_ids: tuple[str, ...],
) -> PositiveGainForecast | None:
    """Absolute p10/p50/p90 points from parent objectives plus deltas.

    Corpus forecasts are per-metric DELTAS relative to the candidate's
    parent, so they are usable only when parent objectives cover every
    metric.  Direction-only forecasts degrade to no forecast.
    """

    forecast_raw = raw.get("forecast")
    parent = raw.get("parent")
    if not isinstance(forecast_raw, list) or not isinstance(
        parent,
        dict,
    ):
        return None
    parent_objectives = parent.get("objectives")
    if not isinstance(parent_objectives, dict) or not set(
        metric_ids
    ) <= set(parent_objectives):
        return None
    deltas: dict[str, dict[str, float]] = {}
    confidences: list[float] = []
    for row in forecast_raw:
        if not isinstance(row, dict) or "metric_id" not in row:
            return None
        metric_id = str(row["metric_id"])
        try:
            deltas[metric_id] = {
                quantile: float(row[f"{quantile}_delta"])
                for quantile in ("p10", "p50", "p90")
            }
        except (KeyError, TypeError, ValueError):
            return None
        confidence = row.get("confidence")
        if isinstance(confidence, (int, float)) and math.isfinite(
            float(confidence)
        ):
            confidences.append(min(max(float(confidence), 0.0), 1.0))
    if set(deltas) != set(metric_ids):
        return None
    quantile_points = tuple(
        (
            quantile,
            _normalized_point(
                {
                    metric_id: float(parent_objectives[metric_id])
                    + deltas[metric_id][quantile]
                    for metric_id in metric_ids
                },
                axes,
            ),
        )
        for quantile in ("p10", "p50", "p90")
    )
    reliability = (
        sum(confidences) / len(confidences) if confidences else 1.0
    )
    return PositiveGainForecast(
        quantile_points=quantile_points,
        reliability=float(reliability),
    )


def _corpus_anchor(
    raw: dict[str, object],
    axes: tuple[dict[str, object], ...],
    metric_ids: tuple[str, ...],
) -> ObjectivePoint | None:
    """The candidate's parent objective point, when the corpus records it."""

    parent = raw.get("parent")
    if not isinstance(parent, dict):
        return None
    objectives = parent.get("objectives")
    if not isinstance(objectives, dict) or not set(metric_ids) <= set(
        objectives
    ):
        return None
    return _normalized_point(
        {
            metric_id: float(objectives[metric_id])
            for metric_id in metric_ids
        },
        axes,
    )


def market_record_from_corpus(payload: dict[str, object]) -> MarketRecord:
    """Build a MarketRecord from one jul27 replay-corpus JSON payload.

    Objective values are normalized onto each axis's ideal-to-reference
    interval, so the unit point is the hypervolume reference and the
    stored exact hypervolumes match the corpus's normalized values.
    Engine identity falls back lane_id -> expert_id -> family; frozen
    scores are min-max normalized per market; forecasts are rebuilt as
    absolute quantile points from parent objectives plus recorded
    deltas, degrading to no forecast when either half is missing.
    """

    market_id = str(payload["market_id"])
    axes = tuple(payload["hv_reference_point"]["axes"])
    metric_ids = tuple(
        sorted(str(axis["metric_id"]) for axis in axes)
    )
    reference = tuple(
        sorted((metric_id, 1.0) for metric_id in metric_ids)
    )
    archive_points = tuple(
        _normalized_point(point, axes)
        for point in payload["archive_at_market"]["points"]
    )
    raw_candidates = tuple(payload["candidates"])
    frozen_values = [
        value
        for value in (
            _corpus_frozen_raw(raw) for raw in raw_candidates
        )
        if value is not None
    ]
    frozen_low = min(frozen_values) if frozen_values else 0.0
    frozen_high = max(frozen_values) if frozen_values else 0.0
    candidates: list[MarketCandidateRecord] = []
    for raw in raw_candidates:
        objectives_raw = raw.get("objectives")
        evaluated = bool(raw.get("evaluated", False))
        feasible = (
            evaluated
            and raw.get("valid") is not False
            and objectives_raw is not None
        )
        frozen_raw = _corpus_frozen_raw(raw)
        if frozen_raw is None:
            frozen_score = None
        elif frozen_high == frozen_low:
            frozen_score = NEUTRAL_PRIOR_SCORE
        else:
            frozen_score = float(
                (frozen_raw - frozen_low)
                / (frozen_high - frozen_low)
            )
        candidates.append(
            (
                _corpus_engine_id(raw),
                int(raw["native_rank"]),
                MarketCandidateRecord(
                    action_sha256=_corpus_action_sha256(
                        market_id,
                        raw,
                    ),
                    engine_id=_corpus_engine_id(raw),
                    native_rank=1,
                    frozen_score=frozen_score,
                    forecast=_corpus_forecast(
                        raw,
                        axes,
                        metric_ids,
                    ),
                    evaluated=evaluated,
                    feasible=feasible,
                    objectives=(
                        None
                        if not feasible
                        else _normalized_point(objectives_raw, axes)
                    ),
                    anchor_point=_corpus_anchor(raw, axes, metric_ids),
                ),
            )
        )
    # A failed materialization can leave native-rank gaps inside an
    # engine, so ranks are densified per engine preserving order.
    by_engine: dict[str, list[tuple[int, MarketCandidateRecord]]] = {}
    for engine_id, native_rank, candidate in candidates:
        by_engine.setdefault(engine_id, []).append(
            (native_rank, candidate)
        )
    dense: list[MarketCandidateRecord] = []
    for engine_id in sorted(by_engine):
        lane = sorted(
            by_engine[engine_id],
            key=lambda item: (item[0], item[1].action_sha256),
        )
        for rank, (_original_rank, candidate) in enumerate(
            lane,
            start=1,
        ):
            dense.append(
                MarketCandidateRecord(
                    action_sha256=candidate.action_sha256,
                    engine_id=candidate.engine_id,
                    native_rank=rank,
                    frozen_score=candidate.frozen_score,
                    forecast=candidate.forecast,
                    evaluated=candidate.evaluated,
                    feasible=candidate.feasible,
                    objectives=candidate.objectives,
                    anchor_point=candidate.anchor_point,
                )
            )
    return MarketRecord(
        market_id=str(payload["market_id"]),
        archive_points=archive_points,
        hv_reference_point=reference,
        candidates=tuple(dense),
    )


__all__ = [
    "DEFAULT_ORACLE_SUBSET_LIMIT",
    "HYPERVOLUME_GAIN_EPSILON",
    "NEUTRAL_PRIOR_SCORE",
    "SEQUENTIAL_MARKET_REPLAY_ID",
    "SEQUENTIAL_MARKET_REPLAY_VERSION",
    "ExactHypervolumeGainPort",
    "FrozenScoreTopKReplayPolicy",
    "LaneHeadsReplayPolicy",
    "MarketCandidateRecord",
    "MarketRecord",
    "NativeRankRoundRobinReplayPolicy",
    "ReplayResult",
    "ReplaySelection",
    "ReplayStepReceipt",
    "SequentialMarketReplay",
    "UniformRandomReplayPolicy",
    "V8LiteReplayPolicy",
    "exact_hypervolume",
    "market_record_from_corpus",
]
