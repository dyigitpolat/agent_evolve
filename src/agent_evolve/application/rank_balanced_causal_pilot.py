"""Rank-balanced randomized pilot selection with exact seat propensities.

This policy repairs the deterministic one-lane-head-per-engine pilot.  A
stale frozen prior score can promote a deep native rank over an engine's own
top ranks; a head-only pilot then never observes where the engine's native
ordering actually converts.  The repaired design keeps engine coverage but:

* spreads seats across native rank BANDS via a low-discrepancy schedule, so
  interior ranks are purchased, not only heads;
* block-randomizes the within-band head between the native-rank order and the
  frozen-score order, so rank-source disagreement receives exposure; and
* mixes an exploration floor toward uniform-within-engine.

Every seat is drawn from an explicitly defined mixture distribution over the
engine's remaining candidates.  The exact propensity of every support member
(selected or not) is emitted as an exact rational, so later causal analysis
can inverse-propensity-weight any realized pilot.

The policy knows no workload, objective, model, provider, or prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

from agent_evolve.domain.patch import require_sha256


RANK_BALANCED_CAUSAL_PILOT_POLICY_ID = "rank_balanced_causal_pilot"
RANK_BALANCED_CAUSAL_PILOT_POLICY_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_DEFINITION_DOMAIN = (
    b"agent-evolve:rank-balanced-causal-pilot-definition:v1\x00"
)
_MARKET_DOMAIN = b"agent-evolve:rank-balanced-causal-pilot-market:v1\x00"
_DESIGN_DOMAIN = b"agent-evolve:rank-balanced-causal-pilot-design:v1\x00"


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


def rank_band_index(
    native_rank: int,
    lane_size: int,
    band_count: int,
) -> int:
    """Map one native rank into its lane band (band zero is the top)."""

    if (
        type(native_rank) is not int
        or type(lane_size) is not int
        or type(band_count) is not int
        or native_rank <= 0
        or lane_size <= 0
        or band_count <= 0
        or native_rank > lane_size
    ):
        raise ValueError("rank band inputs must be consistent positives")
    return min(
        ((native_rank - 1) * band_count) // lane_size,
        band_count - 1,
    )


def _validated_band_weights(
    band_count: int,
    band_weights: tuple[float, ...] | None,
) -> tuple[Fraction, ...]:
    if band_weights is None:
        return tuple(
            Fraction(1, band_count) for _ in range(band_count)
        )
    if (
        type(band_weights) is not tuple
        or len(band_weights) != band_count
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or value <= 0.0
            for value in band_weights
        )
    ):
        raise ValueError(
            "band_weights must be one positive float per band"
        )
    weights = tuple(Fraction(value) for value in band_weights)
    if sum(weights, Fraction(0)) != Fraction(1):
        raise ValueError("band_weights must sum to exactly one")
    return weights


def rank_band_schedule(
    lane_size: int,
    band_count: int,
    band_weights: tuple[float, ...] | None = None,
) -> tuple[int, ...]:
    """Low-discrepancy native-rank visitation order for one lane.

    Ranks are grouped into contiguous bands and visited by an exact
    weighted quota walk (D'Hondt over per-band visit counts): each step
    visits the open band maximizing ``weight / (visits + 1)``, taking that
    band's next-best native rank.  With equal weights (``band_weights``
    omitted) this round-robins band heads: a six-item lane with two bands
    yields the validated order ``(1, 4, 2, 5, 3, 6)`` and three bands
    yield ``(1, 3, 5, 2, 4, 6)``.  Unequal weights concentrate early
    seats on heavier bands while every positively weighted band retains a
    nonzero visitation floor.
    """

    if (
        type(lane_size) is not int
        or type(band_count) is not int
        or lane_size <= 0
        or band_count <= 0
    ):
        raise ValueError("schedule inputs must be positive integers")
    weights = _validated_band_weights(band_count, band_weights)
    bands: list[list[int]] = [[] for _ in range(band_count)]
    for rank in range(1, lane_size + 1):
        bands[rank_band_index(rank, lane_size, band_count)].append(rank)
    visits = [0] * band_count
    schedule: list[int] = []
    while len(schedule) < lane_size:
        open_bands = [
            band
            for band in range(band_count)
            if visits[band] < len(bands[band])
        ]
        chosen = min(
            open_bands,
            key=lambda band: (
                -(weights[band] / (visits[band] + 1)),
                band,
            ),
        )
        schedule.append(bands[chosen][visits[chosen]])
        visits[chosen] += 1
    return tuple(schedule)


@dataclass(frozen=True, slots=True)
class RankBalancedPilotCandidate:
    """Portable, outcome-blind view of one pilotable candidate."""

    action_sha256: str
    engine_id: str
    native_rank: int
    frozen_score: float | None = None
    forecast_summary: float | None = None

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        _require_token(self.engine_id, name="engine_id")
        if type(self.native_rank) is not int or self.native_rank <= 0:
            raise ValueError("native_rank must be a positive integer")
        if self.frozen_score is not None and (
            type(self.frozen_score) is not float
            or not math.isfinite(self.frozen_score)
        ):
            raise ValueError("frozen_score must be a finite float or None")
        if self.forecast_summary is not None and (
            type(self.forecast_summary) is not float
            or not math.isfinite(self.forecast_summary)
        ):
            raise ValueError(
                "forecast_summary must be a finite float or None"
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "action_sha256": self.action_sha256,
            "engine_id": self.engine_id,
            "native_rank": self.native_rank,
            "frozen_score_hex": (
                None
                if self.frozen_score is None
                else self.frozen_score.hex()
            ),
            "forecast_summary_hex": (
                None
                if self.forecast_summary is None
                else self.forecast_summary.hex()
            ),
        }


@dataclass(frozen=True, slots=True)
class RankBalancedPilotPropensity:
    """Exact mixture propensity of one support member at one seat."""

    action_sha256: str
    propensity_numerator: int
    propensity_denominator: int

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        if (
            type(self.propensity_numerator) is not int
            or type(self.propensity_denominator) is not int
            or self.propensity_numerator < 0
            or self.propensity_denominator <= 0
        ):
            raise ValueError("propensity must be an exact rational")

    @property
    def exact(self) -> Fraction:
        return Fraction(
            self.propensity_numerator,
            self.propensity_denominator,
        )

    @property
    def propensity(self) -> float:
        return float(self.exact)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "action_sha256": self.action_sha256,
            "propensity_numerator": self.propensity_numerator,
            "propensity_denominator": self.propensity_denominator,
            "propensity_hex": self.propensity.hex(),
        }


@dataclass(frozen=True, slots=True)
class RankBalancedPilotSeat:
    """One realized pilot seat plus its complete support distribution."""

    seat_ordinal: int
    engine_id: str
    engine_seat_index: int
    target_band_index: int
    effective_band_index: int
    selected_action_sha256: str
    branch: str
    directed_order: str
    support_propensities: tuple[RankBalancedPilotPropensity, ...]

    def __post_init__(self) -> None:
        if type(self.seat_ordinal) is not int or self.seat_ordinal <= 0:
            raise ValueError("seat_ordinal must be positive")
        _require_token(self.engine_id, name="engine_id")
        if (
            type(self.engine_seat_index) is not int
            or self.engine_seat_index < 0
        ):
            raise ValueError("engine_seat_index must be non-negative")
        for name in ("target_band_index", "effective_band_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be non-negative")
        require_sha256(
            self.selected_action_sha256,
            "selected_action_sha256",
        )
        for value, name in (
            (self.branch, "branch"),
            (self.directed_order, "directed_order"),
        ):
            _require_token(value, name=name)
        if (
            type(self.support_propensities) is not tuple
            or not self.support_propensities
            or any(
                type(value) is not RankBalancedPilotPropensity
                for value in self.support_propensities
            )
        ):
            raise TypeError(
                "support_propensities must be exact and non-empty"
            )
        support_ids = tuple(
            value.action_sha256 for value in self.support_propensities
        )
        if support_ids != tuple(sorted(set(support_ids))):
            raise ValueError("support propensities must be canonical")
        if self.selected_action_sha256 not in set(support_ids):
            raise ValueError("selected action must be in the seat support")
        for value in self.support_propensities:
            value.__post_init__()
        if sum(
            (value.exact for value in self.support_propensities),
            Fraction(0),
        ) != Fraction(1):
            raise ValueError("seat propensities must sum to exactly one")
        if self.selected_propensity_exact <= 0:
            raise ValueError("selected propensity must be positive")

    @property
    def selected_propensity_exact(self) -> Fraction:
        return next(
            value.exact
            for value in self.support_propensities
            if value.action_sha256 == self.selected_action_sha256
        )

    @property
    def selection_propensity(self) -> float:
        return float(self.selected_propensity_exact)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "seat_ordinal": self.seat_ordinal,
            "engine_id": self.engine_id,
            "engine_seat_index": self.engine_seat_index,
            "target_band_index": self.target_band_index,
            "effective_band_index": self.effective_band_index,
            "selected_action_sha256": self.selected_action_sha256,
            "branch": self.branch,
            "directed_order": self.directed_order,
            "selection_propensity_hex": self.selection_propensity.hex(),
            "support_propensities": [
                value.to_record()
                for value in self.support_propensities
            ],
        }


@dataclass(frozen=True, slots=True)
class RankBalancedPilotDesign:
    """One deterministic-given-seed realized pilot with causal receipts."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    residual_request_sha256: str
    market_sha256: str
    pilot_width: int
    seats: tuple[RankBalancedPilotSeat, ...]
    design_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.policy_id, name="policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be positive")
        for value, name in (
            (self.policy_definition_sha256, "policy_definition_sha256"),
            (self.residual_request_sha256, "residual_request_sha256"),
            (self.market_sha256, "market_sha256"),
        ):
            require_sha256(value, name)
        if type(self.pilot_width) is not int or self.pilot_width <= 0:
            raise ValueError("pilot_width must be positive")
        if (
            type(self.seats) is not tuple
            or len(self.seats) != self.pilot_width
            or any(
                type(value) is not RankBalancedPilotSeat
                for value in self.seats
            )
        ):
            raise TypeError("seats must exactly fill the pilot width")
        for ordinal, value in enumerate(self.seats, start=1):
            value.__post_init__()
            if value.seat_ordinal != ordinal:
                raise ValueError("seat ordinals must be sequential")
        selected = tuple(
            value.selected_action_sha256 for value in self.seats
        )
        if len(selected) != len(set(selected)):
            raise ValueError("a pilot cannot repeat an action")
        object.__setattr__(
            self,
            "design_sha256",
            _hash(_DESIGN_DOMAIN, self._unsigned_record()),
        )

    @property
    def selected_action_sha256s(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                value.selected_action_sha256 for value in self.seats
            )
        )

    @property
    def design_propensity(self) -> float:
        product = Fraction(1)
        for value in self.seats:
            product *= value.selected_propensity_exact
        return float(product)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "residual_request_sha256": self.residual_request_sha256,
            "market_sha256": self.market_sha256,
            "pilot_width": self.pilot_width,
            "seats": [value.to_record() for value in self.seats],
            "candidate_outcomes_observed": False,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "selected_action_sha256s": list(
                self.selected_action_sha256s
            ),
            "design_propensity_hex": self.design_propensity.hex(),
            "design_sha256": self.design_sha256,
        }


#: Default per-band seat mass.  The V70 full-market census measured
#: positive rates concentrated in the top native-rank bands (ranks 1-2
#: converted at roughly triple the deep-rank rate), so the default mass
#: leans onto the top and middle thirds while every band keeps a nonzero
#: visitation floor.  All values are exact dyadic floats.
DEFAULT_PILOT_BAND_WEIGHTS = (0.5, 0.3125, 0.1875)


@dataclass(frozen=True, slots=True)
class RankBalancedCausalPilotPolicy:
    """Engine-covering, band-scheduled, block-randomized pilot design."""

    band_count: int = 3
    band_weights: tuple[float, ...] = DEFAULT_PILOT_BAND_WEIGHTS
    exploration_epsilon: float = 0.125
    random_seed: int = 0
    policy_id: str = RANK_BALANCED_CAUSAL_PILOT_POLICY_ID
    policy_version: int = RANK_BALANCED_CAUSAL_PILOT_POLICY_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.band_count) is not int or self.band_count <= 0:
            raise ValueError("band_count must be positive")
        _validated_band_weights(self.band_count, self.band_weights)
        if (
            type(self.exploration_epsilon) is not float
            or not math.isfinite(self.exploration_epsilon)
            or not 0.0 <= self.exploration_epsilon < 1.0
        ):
            raise ValueError("exploration_epsilon must lie in [0, 1)")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        _require_token(self.policy_id, name="policy_id")
        if self.policy_version != RANK_BALANCED_CAUSAL_PILOT_POLICY_VERSION:
            raise ValueError("policy_version is unsupported")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "band_count": self.band_count,
                    "band_weights_hex": [
                        value.hex() for value in self.band_weights
                    ],
                    "exploration_epsilon_hex": (
                        self.exploration_epsilon.hex()
                    ),
                    "random_seed": self.random_seed,
                    "seat_allocation": (
                        "engine_coverage_floor_then_dhondt_by_"
                        "engine_requested_width"
                    ),
                    "within_engine_schedule": (
                        "weighted_quota_rank_band_heads_low_discrepancy"
                    ),
                    "within_band_order": (
                        "blocked_randomization_native_rank_vs_"
                        "frozen_score"
                    ),
                    "exploration_floor": (
                        "epsilon_uniform_within_engine_remaining"
                    ),
                    "propensities": (
                        "exact_rational_mixture_conditional_on_prefix"
                    ),
                    "candidate_outcomes_observed": False,
                    "workload_objective_model_provider_prompt_branches": (
                        False
                    ),
                },
            ),
        )

    @staticmethod
    def _validated_engines(
        candidates: tuple[RankBalancedPilotCandidate, ...],
    ) -> dict[str, tuple[RankBalancedPilotCandidate, ...]]:
        if type(candidates) is not tuple or not candidates:
            raise ValueError("candidates must be a non-empty exact tuple")
        seen: set[str] = set()
        engines: dict[str, list[RankBalancedPilotCandidate]] = {}
        for value in candidates:
            if type(value) is not RankBalancedPilotCandidate:
                raise TypeError(
                    "candidates must contain exact pilot candidates"
                )
            value.__post_init__()
            if value.action_sha256 in seen:
                raise ValueError("candidate identities repeat")
            seen.add(value.action_sha256)
            engines.setdefault(value.engine_id, []).append(value)
        result: dict[str, tuple[RankBalancedPilotCandidate, ...]] = {}
        for engine_id in sorted(engines):
            lane = tuple(
                sorted(
                    engines[engine_id],
                    key=lambda value: value.native_rank,
                )
            )
            if tuple(value.native_rank for value in lane) != tuple(
                range(1, len(lane) + 1)
            ):
                raise ValueError(
                    "each engine must be a contiguous ranked list"
                )
            frozen_flags = {
                value.frozen_score is not None for value in lane
            }
            if len(frozen_flags) != 1:
                raise ValueError(
                    "an engine must carry frozen scores for all "
                    "candidates or none"
                )
            result[engine_id] = lane
        return result

    def _market_sha256(
        self,
        engines: dict[str, tuple[RankBalancedPilotCandidate, ...]],
    ) -> str:
        return _hash(
            _MARKET_DOMAIN,
            {
                engine_id: [value.to_record() for value in lane]
                for engine_id, lane in engines.items()
            },
        )

    @staticmethod
    def _seat_engine_sequence(
        engines: dict[str, tuple[RankBalancedPilotCandidate, ...]],
        pilot_width: int,
    ) -> tuple[str, ...]:
        """Coverage-floor seats first, then D'Hondt extra seats."""

        widths = {
            engine_id: len(lane) for engine_id, lane in engines.items()
        }
        floor_order = sorted(
            widths,
            key=lambda engine_id: (-widths[engine_id], engine_id),
        )
        sequence: list[str] = []
        awarded = {engine_id: 0 for engine_id in widths}
        for engine_id in floor_order:
            if len(sequence) >= pilot_width:
                break
            sequence.append(engine_id)
            awarded[engine_id] += 1
        while len(sequence) < pilot_width:
            open_engines = [
                engine_id
                for engine_id in widths
                if awarded[engine_id] < widths[engine_id]
            ]
            if not open_engines:
                raise ValueError(
                    "pilot width exceeds the candidate market"
                )
            chosen = min(
                open_engines,
                key=lambda engine_id: (
                    -Fraction(widths[engine_id], awarded[engine_id] + 1),
                    -widths[engine_id],
                    # Ascending engine id wins ties deterministically.
                    engine_id,
                ),
            )
            sequence.append(chosen)
            awarded[chosen] += 1
        return tuple(sequence)

    def design_pilot(
        self,
        *,
        residual_request_sha256: str,
        candidates: tuple[RankBalancedPilotCandidate, ...],
        pilot_width: int,
    ) -> RankBalancedPilotDesign:
        """Realize one seeded pilot and its exact causal propensities."""

        self.__post_init__()
        require_sha256(
            residual_request_sha256,
            "residual_request_sha256",
        )
        engines = self._validated_engines(candidates)
        if (
            type(pilot_width) is not int
            or not 1 <= pilot_width <= len(candidates)
        ):
            raise ValueError("pilot_width must fit the candidate market")
        market_sha256 = self._market_sha256(engines)
        engine_sequence = self._seat_engine_sequence(
            engines,
            pilot_width,
        )
        epsilon = Fraction(self.exploration_epsilon)
        selected: set[str] = set()
        block_native_first: dict[tuple[str, int], bool] = {}
        seats: list[RankBalancedPilotSeat] = []
        engine_seat_counts: dict[str, int] = {}
        for seat_ordinal, engine_id in enumerate(
            engine_sequence,
            start=1,
        ):
            lane = engines[engine_id]
            engine_seat_index = engine_seat_counts.get(engine_id, 0)
            engine_seat_counts[engine_id] = engine_seat_index + 1
            remaining = tuple(
                value
                for value in lane
                if value.action_sha256 not in selected
            )
            if not remaining:  # pragma: no cover - engine seats are capped
                raise AssertionError("engine seat exceeded its lane")
            band_sequence = tuple(
                rank_band_index(
                    rank,
                    len(lane),
                    self.band_count,
                )
                for rank in rank_band_schedule(
                    len(lane),
                    self.band_count,
                    self.band_weights,
                )
            )
            target_band = band_sequence[engine_seat_index % len(lane)]
            effective_band = target_band
            band_members: tuple[RankBalancedPilotCandidate, ...] = ()
            for offset in range(len(lane)):
                effective_band = band_sequence[
                    (engine_seat_index + offset) % len(lane)
                ]
                band_members = tuple(
                    value
                    for value in remaining
                    if rank_band_index(
                        value.native_rank,
                        len(lane),
                        self.band_count,
                    )
                    == effective_band
                )
                if band_members:
                    break
            if not band_members:  # pragma: no cover - remaining is non-empty
                raise AssertionError("no band covers a remaining candidate")
            has_frozen = lane[0].frozen_score is not None
            native_head = min(
                band_members,
                key=lambda value: (
                    value.native_rank,
                    value.action_sha256,
                ),
            )
            if has_frozen:
                frozen_head = min(
                    band_members,
                    key=lambda value: (
                        -value.frozen_score,
                        value.native_rank,
                        value.action_sha256,
                    ),
                )
            else:
                frozen_head = native_head
            block_key = (engine_id, engine_seat_index // 2)
            block_first = engine_seat_index % 2 == 0
            if has_frozen and block_first:
                block_native_first[block_key] = (
                    _stable_unit_interval(
                        self.random_seed,
                        market_sha256,
                        residual_request_sha256,
                        "pilot_block_order",
                        engine_id,
                        engine_seat_index // 2,
                    )
                    < 0.5
                )
            native_first = block_native_first.get(block_key, True)
            if not has_frozen:
                directed_order = "native_rank"
                directed: dict[str, Fraction] = {
                    native_head.action_sha256: Fraction(1)
                }
                directed_head = native_head
            elif block_first:
                # The block order is drawn at this seat, so both orders
                # are equally likely before the draw.
                directed_order = (
                    "block_first_native_rank"
                    if native_first
                    else "block_first_frozen_score"
                )
                directed = {native_head.action_sha256: Fraction(0)}
                directed[native_head.action_sha256] += Fraction(1, 2)
                directed[frozen_head.action_sha256] = directed.get(
                    frozen_head.action_sha256,
                    Fraction(0),
                ) + Fraction(1, 2)
                directed_head = (
                    native_head if native_first else frozen_head
                )
            else:
                # The block order was drawn and logged one seat earlier,
                # so this seat's directed head is deterministic.
                forced_native = not native_first
                directed_order = (
                    "block_second_native_rank"
                    if forced_native
                    else "block_second_frozen_score"
                )
                directed_head = (
                    native_head if forced_native else frozen_head
                )
                directed = {directed_head.action_sha256: Fraction(1)}
            support = tuple(
                sorted(
                    remaining,
                    key=lambda value: value.action_sha256,
                )
            )
            uniform_share = epsilon / len(support)
            propensity_values: list[RankBalancedPilotPropensity] = []
            for value in support:
                mixture = (Fraction(1) - epsilon) * directed.get(
                    value.action_sha256,
                    Fraction(0),
                ) + uniform_share
                propensity_values.append(
                    RankBalancedPilotPropensity(
                        action_sha256=value.action_sha256,
                        propensity_numerator=mixture.numerator,
                        propensity_denominator=mixture.denominator,
                    )
                )
            propensities = tuple(propensity_values)
            exploration_draw = _stable_unit_interval(
                self.random_seed,
                market_sha256,
                residual_request_sha256,
                "pilot_exploration_branch",
                seat_ordinal,
            )
            if exploration_draw < self.exploration_epsilon:
                choice_draw = _stable_unit_interval(
                    self.random_seed,
                    market_sha256,
                    residual_request_sha256,
                    "pilot_exploration_choice",
                    seat_ordinal,
                )
                chosen = support[
                    min(
                        int(choice_draw * len(support)),
                        len(support) - 1,
                    )
                ]
                branch = "exploration"
            else:
                chosen = directed_head
                branch = "directed"
            selected.add(chosen.action_sha256)
            seats.append(
                RankBalancedPilotSeat(
                    seat_ordinal=seat_ordinal,
                    engine_id=engine_id,
                    engine_seat_index=engine_seat_index,
                    target_band_index=target_band,
                    effective_band_index=effective_band,
                    selected_action_sha256=chosen.action_sha256,
                    branch=branch,
                    directed_order=directed_order,
                    support_propensities=propensities,
                )
            )
        return RankBalancedPilotDesign(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=residual_request_sha256,
            market_sha256=market_sha256,
            pilot_width=pilot_width,
            seats=tuple(seats),
        )


SEQUENTIAL_ADAPTIVE_BAND_PILOT_POLICY_ID = (
    "sequential_adaptive_band_pilot"
)
SEQUENTIAL_ADAPTIVE_BAND_PILOT_POLICY_VERSION = 1
_SEQUENTIAL_DEFINITION_DOMAIN = (
    b"agent-evolve:sequential-adaptive-band-pilot-definition:v1\x00"
)


@dataclass(frozen=True, slots=True)
class PilotSeatObservation:
    """One revealed outcome of an earlier pilot seat."""

    action_sha256: str
    feasible: bool
    marginal_archive_gain: float

    def __post_init__(self) -> None:
        require_sha256(self.action_sha256, "action_sha256")
        if type(self.feasible) is not bool:
            raise TypeError("feasible must be exact")
        if (
            type(self.marginal_archive_gain) is not float
            or not math.isfinite(self.marginal_archive_gain)
            or self.marginal_archive_gain < 0.0
        ):
            raise ValueError(
                "marginal_archive_gain must be finite and non-negative"
            )

    @property
    def positive(self) -> bool:
        return self.marginal_archive_gain > 0.0


@dataclass(frozen=True, slots=True)
class SequentialAdaptiveBandPilotPolicy:
    """Sequential pilot: engine and band mass adapt to revealed outcomes.

    Repairs measured on the deterministic one-shot pilot:

    * engine floor seats are ordered by the engines' posterior positive
      rate (Beta shrinkage toward the within-market global posterior),
      tie-broken by ascending engine id — never by lane width, which was
      measured anti-correlated with conversion on many-engine markets;
    * the target band is SAMPLED per seat from the base band weights
      multiplied by the band-level posterior positive rate (pooled
      across engines, temperature-controlled), so revealed top-band
      successes concentrate later mass toward heads while top-band
      zeros preserve interior exploration; and
    * blocked randomization and the epsilon-uniform floor are retained,
      so every propensity remains an exact rational of the mixture.

    Adaptivity uses only outcomes revealed BEFORE the seat; the policy
    remains workload-, model-, and provider-blind.
    """

    band_count: int = 3
    band_weights: tuple[float, ...] = DEFAULT_PILOT_BAND_WEIGHTS
    exploration_epsilon: float = 0.125
    adaptation_temperature: float = 1.0
    prior_strength: float = 2.0
    root_prior_probability: float = 0.5
    random_seed: int = 0
    policy_id: str = SEQUENTIAL_ADAPTIVE_BAND_PILOT_POLICY_ID
    policy_version: int = SEQUENTIAL_ADAPTIVE_BAND_PILOT_POLICY_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.band_count) is not int or self.band_count <= 0:
            raise ValueError("band_count must be positive")
        _validated_band_weights(self.band_count, self.band_weights)
        if (
            type(self.exploration_epsilon) is not float
            or not math.isfinite(self.exploration_epsilon)
            or not 0.0 <= self.exploration_epsilon < 1.0
        ):
            raise ValueError("exploration_epsilon must lie in [0, 1)")
        if (
            type(self.adaptation_temperature) is not float
            or not math.isfinite(self.adaptation_temperature)
            or self.adaptation_temperature < 0.0
        ):
            raise ValueError(
                "adaptation_temperature must be finite and non-negative"
            )
        if (
            type(self.prior_strength) is not float
            or not math.isfinite(self.prior_strength)
            or self.prior_strength <= 0.0
        ):
            raise ValueError("prior_strength must be positive")
        if (
            type(self.root_prior_probability) is not float
            or not math.isfinite(self.root_prior_probability)
            or not 0.0 < self.root_prior_probability < 1.0
        ):
            raise ValueError(
                "root_prior_probability must lie in (0, 1)"
            )
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        _require_token(self.policy_id, name="policy_id")
        if (
            self.policy_version
            != SEQUENTIAL_ADAPTIVE_BAND_PILOT_POLICY_VERSION
        ):
            raise ValueError("policy_version is unsupported")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _SEQUENTIAL_DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "band_count": self.band_count,
                    "band_weights_hex": [
                        value.hex() for value in self.band_weights
                    ],
                    "exploration_epsilon_hex": (
                        self.exploration_epsilon.hex()
                    ),
                    "adaptation_temperature_hex": (
                        self.adaptation_temperature.hex()
                    ),
                    "prior_strength_hex": self.prior_strength.hex(),
                    "root_prior_probability_hex": (
                        self.root_prior_probability.hex()
                    ),
                    "random_seed": self.random_seed,
                    "engine_floor_order": (
                        "posterior_positive_rate_then_engine_id_"
                        "never_lane_width"
                    ),
                    "band_mass": (
                        "base_weights_times_pooled_band_posterior_"
                        "temperature_controlled_sampled_per_seat"
                    ),
                    "within_band_order": (
                        "blocked_randomization_native_rank_vs_"
                        "frozen_score"
                    ),
                    "exploration_floor": (
                        "epsilon_uniform_within_engine_remaining"
                    ),
                    "propensities": (
                        "exact_rational_mixture_conditional_on_"
                        "revealed_history"
                    ),
                    "future_outcomes_visible": False,
                    "workload_objective_model_provider_prompt_branches": (
                        False
                    ),
                },
            ),
        )

    @staticmethod
    def _posterior(
        *,
        prior_mean: float,
        prior_strength: float,
        successes: float,
        failures: float,
    ) -> float:
        return (prior_strength * prior_mean + successes) / (
            prior_strength + successes + failures
        )

    def design_seat(
        self,
        *,
        residual_request_sha256: str,
        candidates: tuple[RankBalancedPilotCandidate, ...],
        selected_action_sha256s: tuple[str, ...],
        observations: tuple[PilotSeatObservation, ...],
        seat_ordinal: int,
    ) -> RankBalancedPilotSeat:
        """Choose one pilot seat given the revealed pilot history."""

        self.__post_init__()
        require_sha256(
            residual_request_sha256,
            "residual_request_sha256",
        )
        engines = RankBalancedCausalPilotPolicy._validated_engines(
            candidates
        )
        by_action = {
            value.action_sha256: value for value in candidates
        }
        if (
            type(selected_action_sha256s) is not tuple
            or selected_action_sha256s
            != tuple(sorted(set(selected_action_sha256s)))
            or not set(selected_action_sha256s) <= set(by_action)
        ):
            raise ValueError(
                "selected_action_sha256s must be a canonical subset"
            )
        if type(observations) is not tuple or any(
            type(value) is not PilotSeatObservation
            for value in observations
        ):
            raise TypeError(
                "observations must contain exact seat observations"
            )
        observed_ids = set()
        for value in observations:
            value.__post_init__()
            if value.action_sha256 in observed_ids:
                raise ValueError("observations repeat an action")
            observed_ids.add(value.action_sha256)
        if not observed_ids <= set(selected_action_sha256s):
            raise ValueError(
                "observations must cover only selected actions"
            )
        if type(seat_ordinal) is not int or seat_ordinal <= 0:
            raise ValueError("seat_ordinal must be positive")
        market_sha256 = _hash(
            _MARKET_DOMAIN,
            {
                engine_id: [value.to_record() for value in lane]
                for engine_id, lane in engines.items()
            },
        )

        # Posterior evidence from revealed pilot outcomes only.
        global_positive = sum(
            value.positive for value in observations
        )
        global_count = len(observations)
        global_posterior = self._posterior(
            prior_mean=self.root_prior_probability,
            prior_strength=self.prior_strength,
            successes=float(global_positive),
            failures=float(global_count - global_positive),
        )
        engine_counts: dict[str, list[float]] = {}
        band_counts: dict[int, list[float]] = {}
        for value in observations:
            candidate = by_action[value.action_sha256]
            lane = engines[candidate.engine_id]
            band = rank_band_index(
                candidate.native_rank,
                len(lane),
                self.band_count,
            )
            row = engine_counts.setdefault(
                candidate.engine_id,
                [0.0, 0.0],
            )
            row[0] += 1.0
            row[1] += float(value.positive)
            band_row = band_counts.setdefault(band, [0.0, 0.0])
            band_row[0] += 1.0
            band_row[1] += float(value.positive)

        def engine_posterior(engine_id: str) -> float:
            count, positive = engine_counts.get(
                engine_id,
                [0.0, 0.0],
            )
            return self._posterior(
                prior_mean=global_posterior,
                prior_strength=self.prior_strength,
                successes=positive,
                failures=count - positive,
            )

        def band_posterior(band: int) -> float:
            count, positive = band_counts.get(band, [0.0, 0.0])
            return self._posterior(
                prior_mean=global_posterior,
                prior_strength=self.prior_strength,
                successes=positive,
                failures=count - positive,
            )

        selected = set(selected_action_sha256s)
        open_engines = {
            engine_id: tuple(
                value
                for value in lane
                if value.action_sha256 not in selected
            )
            for engine_id, lane in engines.items()
        }
        open_engines = {
            engine_id: remaining
            for engine_id, remaining in open_engines.items()
            if remaining
        }
        if not open_engines:
            raise ValueError("no engine has a remaining candidate")
        seated_engines = {
            by_action[value].engine_id
            for value in selected_action_sha256s
        }
        unseated = [
            engine_id
            for engine_id in open_engines
            if engine_id not in seated_engines
        ]
        # Coverage floor: while any engine is unseated, seats go to
        # unseated engines, ordered by posterior conversion evidence.
        pool = unseated if unseated else sorted(open_engines)
        engine_id = min(
            pool,
            key=lambda value: (-engine_posterior(value), value),
        )
        lane = engines[engine_id]
        remaining = open_engines[engine_id]
        engine_seat_index = sum(
            by_action[value].engine_id == engine_id
            for value in selected_action_sha256s
        )

        # Adapted band mass over non-empty bands: exact rationals.
        non_empty: dict[int, tuple[RankBalancedPilotCandidate, ...]] = {}
        for value in remaining:
            band = rank_band_index(
                value.native_rank,
                len(lane),
                self.band_count,
            )
            non_empty.setdefault(band, ())
            non_empty[band] = (*non_empty[band], value)
        weights: dict[int, Fraction] = {}
        for band in sorted(non_empty):
            multiplier = band_posterior(band) ** (
                self.adaptation_temperature
            )
            weights[band] = Fraction(
                float(self.band_weights[band])
            ) * Fraction(float(multiplier))
        total_weight = sum(weights.values(), Fraction(0))
        if total_weight <= 0:  # pragma: no cover - weights are positive
            raise AssertionError("band mass vanished")
        band_mass = {
            band: value / total_weight
            for band, value in weights.items()
        }

        has_frozen = lane[0].frozen_score is not None
        block_key_index = engine_seat_index // 2
        block_first = engine_seat_index % 2 == 0
        if has_frozen and block_first:
            native_first = (
                _stable_unit_interval(
                    self.random_seed,
                    market_sha256,
                    residual_request_sha256,
                    "adaptive_pilot_block_order",
                    engine_id,
                    block_key_index,
                )
                < 0.5
            )
        elif has_frozen:
            native_first = (
                _stable_unit_interval(
                    self.random_seed,
                    market_sha256,
                    residual_request_sha256,
                    "adaptive_pilot_block_order",
                    engine_id,
                    block_key_index,
                )
                < 0.5
            )
        else:
            native_first = True

        directed: dict[str, Fraction] = {}
        directed_heads: dict[int, RankBalancedPilotCandidate] = {}
        for band, members in non_empty.items():
            native_head = min(
                members,
                key=lambda value: (
                    value.native_rank,
                    value.action_sha256,
                ),
            )
            if has_frozen:
                frozen_head = min(
                    members,
                    key=lambda value: (
                        -value.frozen_score,
                        value.native_rank,
                        value.action_sha256,
                    ),
                )
            else:
                frozen_head = native_head
            if has_frozen and block_first:
                for head in (native_head, frozen_head):
                    directed[head.action_sha256] = directed.get(
                        head.action_sha256,
                        Fraction(0),
                    )
                directed[native_head.action_sha256] += (
                    band_mass[band] / 2
                )
                directed[frozen_head.action_sha256] += (
                    band_mass[band] / 2
                )
                directed_heads[band] = (
                    native_head if native_first else frozen_head
                )
            else:
                head = (
                    native_head
                    if (not has_frozen) or native_first
                    else frozen_head
                )
                # Block-second seats reuse the block order drawn at
                # the first seat of the block, deterministically.
                if has_frozen and not block_first:
                    head = (
                        frozen_head if native_first else native_head
                    )
                directed[head.action_sha256] = directed.get(
                    head.action_sha256,
                    Fraction(0),
                ) + band_mass[band]
                directed_heads[band] = head

        epsilon = Fraction(self.exploration_epsilon)
        support = tuple(
            sorted(
                remaining,
                key=lambda value: value.action_sha256,
            )
        )
        uniform_share = epsilon / len(support)
        propensity_values: list[RankBalancedPilotPropensity] = []
        for value in support:
            mixture = (Fraction(1) - epsilon) * directed.get(
                value.action_sha256,
                Fraction(0),
            ) + uniform_share
            propensity_values.append(
                RankBalancedPilotPropensity(
                    action_sha256=value.action_sha256,
                    propensity_numerator=mixture.numerator,
                    propensity_denominator=mixture.denominator,
                )
            )

        exploration_draw = _stable_unit_interval(
            self.random_seed,
            market_sha256,
            residual_request_sha256,
            "adaptive_pilot_exploration_branch",
            seat_ordinal,
        )
        if exploration_draw < self.exploration_epsilon:
            choice_draw = _stable_unit_interval(
                self.random_seed,
                market_sha256,
                residual_request_sha256,
                "adaptive_pilot_exploration_choice",
                seat_ordinal,
            )
            chosen = support[
                min(
                    int(choice_draw * len(support)),
                    len(support) - 1,
                )
            ]
            branch = "exploration"
            effective_band = rank_band_index(
                chosen.native_rank,
                len(lane),
                self.band_count,
            )
        else:
            band_draw = _stable_unit_interval(
                self.random_seed,
                market_sha256,
                residual_request_sha256,
                "adaptive_pilot_band",
                seat_ordinal,
            )
            cumulative = Fraction(0)
            effective_band = sorted(non_empty)[-1]
            for band in sorted(non_empty):
                cumulative += band_mass[band]
                if band_draw < float(cumulative):
                    effective_band = band
                    break
            chosen = directed_heads[effective_band]
            branch = "directed"
        if not has_frozen:
            directed_order = "native_rank"
        elif block_first:
            directed_order = (
                "block_first_native_rank"
                if native_first
                else "block_first_frozen_score"
            )
        else:
            directed_order = (
                "block_second_frozen_score"
                if native_first
                else "block_second_native_rank"
            )
        return RankBalancedPilotSeat(
            seat_ordinal=seat_ordinal,
            engine_id=engine_id,
            engine_seat_index=engine_seat_index,
            target_band_index=effective_band,
            effective_band_index=effective_band,
            selected_action_sha256=chosen.action_sha256,
            branch=branch,
            directed_order=directed_order,
            support_propensities=tuple(propensity_values),
        )


__all__ = [
    "DEFAULT_PILOT_BAND_WEIGHTS",
    "PilotSeatObservation",
    "RANK_BALANCED_CAUSAL_PILOT_POLICY_ID",
    "RANK_BALANCED_CAUSAL_PILOT_POLICY_VERSION",
    "SEQUENTIAL_ADAPTIVE_BAND_PILOT_POLICY_ID",
    "SEQUENTIAL_ADAPTIVE_BAND_PILOT_POLICY_VERSION",
    "SequentialAdaptiveBandPilotPolicy",
    "RankBalancedCausalPilotPolicy",
    "RankBalancedPilotCandidate",
    "RankBalancedPilotDesign",
    "RankBalancedPilotPropensity",
    "RankBalancedPilotSeat",
    "rank_band_index",
    "rank_band_schedule",
]
