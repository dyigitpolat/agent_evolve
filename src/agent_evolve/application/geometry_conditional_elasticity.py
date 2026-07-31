"""Geometry-conditional seat elasticity for pilot lane allocation (R3).

Defects D2/D6 (jul28 pareto defect theory): the pilot's fixed per-lane
coverage floor spends seats on lanes whose archive state makes them
structurally dead (numerical proposals re-deriving a saturated archive:
0/16 positives; coverage lanes whose parents sit >=0.25 normalized off the
front: 0/13), while the lane holding the positives is under-seated (V70:
83% of positives, 1/6 seats).  The conditioning variable is archive state,
not lane identity, so this module replaces the fixed floor with elastic
per-lane seat bids computed from three state features the receipts already
expose:

* parent distance-to-front of the lane's proposals (outcome-blind,
  additive-epsilon in the normalized objective frame);
* archive saturation, the lane's predicted self-overlap: the fraction of
  its proposals whose central forecast lands inside the currently
  dominated region (outcome-blind; lanes without forecasts stay at the
  neutral prior); and
* the lane's conversion posterior over outcomes revealed BEFORE the seat
  (Beta shrinkage toward the global revealed posterior).

Seats follow a D'Hondt walk over the bids (bid divided by seats already
awarded plus one), so far-front, saturated lanes bid low and concede seats
while a global exploration floor keeps every lane's bid strictly positive;
the within-engine seat design (band schedule, blocked randomization,
epsilon-uniform floor, exact rational propensities) is delegated unchanged
to the existing sequential adaptive pilot.  Composition is config-gated in
``v9_candidate_policy``; no existing module is modified.  The policy knows
no workload, objective name, model, provider, or prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    _beta_posterior_mean,
)

GEOMETRY_CONDITIONAL_ELASTICITY_POLICY_ID = (
    "geometry_conditional_elasticity"
)
GEOMETRY_CONDITIONAL_ELASTICITY_POLICY_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_DEFINITION_DOMAIN = (
    b"agent-evolve:geometry-conditional-elasticity-definition:v1\x00"
)


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


@dataclass(frozen=True, slots=True)
class ElasticSeatConfig:
    """All R3 constants; every float is an exact dyadic value."""

    exploration_floor_bid: float = 0.0625
    distance_weight: float = 4.0
    saturation_weight: float = 1.0

    def __post_init__(self) -> None:
        if (
            type(self.exploration_floor_bid) is not float
            or not math.isfinite(self.exploration_floor_bid)
            or not 0.0 < self.exploration_floor_bid <= 1.0
        ):
            raise ValueError(
                "exploration_floor_bid must lie in (0, 1]"
            )
        if (
            type(self.distance_weight) is not float
            or not math.isfinite(self.distance_weight)
            or self.distance_weight < 0.0
        ):
            raise ValueError(
                "distance_weight must be finite and non-negative"
            )
        if (
            type(self.saturation_weight) is not float
            or not math.isfinite(self.saturation_weight)
            or not 0.0 <= self.saturation_weight <= 1.0
        ):
            raise ValueError(
                "saturation_weight must lie in [0, 1]"
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "exploration_floor_bid_hex": (
                self.exploration_floor_bid.hex()
            ),
            "distance_weight_hex": self.distance_weight.hex(),
            "saturation_weight_hex": self.saturation_weight.hex(),
        }


@dataclass(frozen=True, slots=True)
class LaneGeometryEvidence:
    """Outcome-blind and prequential state features of one lane.

    ``parent_front_distances`` are additive-epsilon distances of the
    lane's proposals' parents behind the current front (empty when the
    lane records no parents).  ``predicted_dominated`` flags, one per
    proposal carrying a forecast, are True when the central forecast
    point adds zero archive gain (the lane re-derives the dominated
    region).  ``revealed_positive`` holds the positivity of the lane's
    outcomes revealed strictly before the current seat.
    """

    engine_id: str
    parent_front_distances: tuple[float, ...] = ()
    predicted_dominated: tuple[bool, ...] = ()
    revealed_positive: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        _require_token(self.engine_id, name="engine_id")
        if type(self.parent_front_distances) is not tuple or any(
            type(value) is not float
            or not math.isfinite(value)
            or value < 0.0
            for value in self.parent_front_distances
        ):
            raise ValueError(
                "parent_front_distances must be finite non-negative "
                "exact floats"
            )
        for name in ("predicted_dominated", "revealed_positive"):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not bool for value in values
            ):
                raise TypeError(f"{name} must contain exact booleans")


@dataclass(frozen=True, slots=True)
class LaneSeatBid:
    """One lane's elastic seat bid with its feature decomposition."""

    engine_id: str
    conversion_posterior: float
    saturation_posterior: float
    mean_parent_front_distance: float | None
    distance_factor: float
    bid: float
    floored: bool

    def __post_init__(self) -> None:
        _require_token(self.engine_id, name="engine_id")
        for name in (
            "conversion_posterior",
            "saturation_posterior",
        ):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.mean_parent_front_distance is not None and (
            type(self.mean_parent_front_distance) is not float
            or not math.isfinite(self.mean_parent_front_distance)
            or self.mean_parent_front_distance < 0.0
        ):
            raise ValueError(
                "mean_parent_front_distance must be finite and "
                "non-negative or None"
            )
        for name in ("distance_factor", "bid"):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be strictly positive")
        if type(self.floored) is not bool:
            raise TypeError("floored must be exact")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "engine_id": self.engine_id,
            "conversion_posterior_hex": (
                self.conversion_posterior.hex()
            ),
            "saturation_posterior_hex": (
                self.saturation_posterior.hex()
            ),
            "mean_parent_front_distance_hex": (
                None
                if self.mean_parent_front_distance is None
                else self.mean_parent_front_distance.hex()
            ),
            "distance_factor_hex": self.distance_factor.hex(),
            "bid_hex": self.bid.hex(),
            "floored": self.floored,
        }


@dataclass(frozen=True, slots=True)
class ElasticSeatBidder:
    """Compute elastic lane bids and walk seats by D'Hondt over them."""

    config: ElasticSeatConfig = ElasticSeatConfig()
    prior_strength: float = 2.0
    root_prior_probability: float = 0.5
    policy_id: str = GEOMETRY_CONDITIONAL_ELASTICITY_POLICY_ID
    policy_version: int = (
        GEOMETRY_CONDITIONAL_ELASTICITY_POLICY_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.config) is not ElasticSeatConfig:
            raise TypeError("config must be exact")
        self.config.__post_init__()
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
        _require_token(self.policy_id, name="policy_id")
        if (
            self.policy_id != GEOMETRY_CONDITIONAL_ELASTICITY_POLICY_ID
            or self.policy_version
            != GEOMETRY_CONDITIONAL_ELASTICITY_POLICY_VERSION
        ):
            raise ValueError("policy identity is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "config": self.config.to_record(),
                    "prior_strength_hex": self.prior_strength.hex(),
                    "root_prior_probability_hex": (
                        self.root_prior_probability.hex()
                    ),
                    "bid": (
                        "conversion_posterior_times_exp_neg_weighted_"
                        "mean_parent_front_distance_times_one_minus_"
                        "weighted_saturation_floored"
                    ),
                    "conversion_evidence": (
                        "revealed_lane_positivity_beta_shrunk_toward_"
                        "global_revealed_posterior"
                    ),
                    "saturation_evidence": (
                        "outcome_blind_central_forecast_self_overlap_"
                        "per_lane_beta_from_root_prior_no_cross_lane_"
                        "pooling"
                    ),
                    "seat_walk": (
                        "dhondt_bid_over_awarded_plus_one_tie_break_"
                        "ascending_engine_id"
                    ),
                    "fixed_per_lane_coverage_floor": False,
                    "global_exploration_floor": (
                        "strictly_positive_bid_floor_plus_unchanged_"
                        "within_engine_epsilon_uniform_propensities"
                    ),
                    "campaign_level_unexposed_lane_diagnostic_seat": (
                        "out_of_scope_for_market_level_composition"
                    ),
                    "future_outcomes_visible": False,
                    "workload_objective_model_provider_prompt_branches": (
                        False
                    ),
                },
            ),
        )

    def _posterior(
        self,
        *,
        prior_mean: float,
        successes: float,
        failures: float,
    ) -> float:
        return _beta_posterior_mean(
            prior_mean=prior_mean,
            prior_strength=self.prior_strength,
            successes=successes,
            failures=failures,
        )

    def lane_bids(
        self,
        lanes: tuple[LaneGeometryEvidence, ...],
    ) -> tuple[LaneSeatBid, ...]:
        """Elastic bids for every lane, in ascending engine order."""

        self.__post_init__()
        if type(lanes) is not tuple or not lanes:
            raise ValueError("lanes must be a non-empty exact tuple")
        engine_ids = []
        for value in lanes:
            if type(value) is not LaneGeometryEvidence:
                raise TypeError(
                    "lanes must contain exact lane evidence"
                )
            value.__post_init__()
            engine_ids.append(value.engine_id)
        if len(engine_ids) != len(set(engine_ids)):
            raise ValueError("lane engines repeat")
        global_revealed = [
            positive
            for value in lanes
            for positive in value.revealed_positive
        ]
        global_conversion = self._posterior(
            prior_mean=self.root_prior_probability,
            successes=float(sum(global_revealed)),
            failures=float(
                len(global_revealed) - sum(global_revealed)
            ),
        )
        bids: list[LaneSeatBid] = []
        for lane in sorted(lanes, key=lambda value: value.engine_id):
            conversion = self._posterior(
                prior_mean=global_conversion,
                successes=float(sum(lane.revealed_positive)),
                failures=float(
                    len(lane.revealed_positive)
                    - sum(lane.revealed_positive)
                ),
            )
            # Saturation is a lane SELF-overlap feature: a lane without
            # forecast evidence stays at the neutral root prior instead
            # of inheriting another lane's saturation.
            saturation = self._posterior(
                prior_mean=self.root_prior_probability,
                successes=float(sum(lane.predicted_dominated)),
                failures=float(
                    len(lane.predicted_dominated)
                    - sum(lane.predicted_dominated)
                ),
            )
            if lane.parent_front_distances:
                mean_distance: float | None = float(
                    math.fsum(lane.parent_front_distances)
                    / len(lane.parent_front_distances)
                )
                distance_factor = math.exp(
                    -self.config.distance_weight * mean_distance
                )
            else:
                mean_distance = None
                distance_factor = 1.0
            raw = (
                conversion
                * distance_factor
                * (
                    1.0
                    - self.config.saturation_weight * saturation
                )
            )
            floored = raw < self.config.exploration_floor_bid
            bids.append(
                LaneSeatBid(
                    engine_id=lane.engine_id,
                    conversion_posterior=float(conversion),
                    saturation_posterior=float(saturation),
                    mean_parent_front_distance=mean_distance,
                    distance_factor=float(distance_factor),
                    bid=float(
                        max(raw, self.config.exploration_floor_bid)
                    ),
                    floored=floored,
                )
            )
        return tuple(bids)

    @staticmethod
    def choose_engine(
        *,
        bids: tuple[LaneSeatBid, ...],
        seats_awarded: dict[str, int],
        open_engine_ids: frozenset[str],
    ) -> str:
        """One D'Hondt step over the elastic bids.

        The winning lane maximizes ``bid / (seats_awarded + 1)`` among
        lanes that still hold a remaining candidate; exact ties break by
        ascending engine id.  This REPLACES the fixed coverage floor: a
        lane earns its early seats through its bid, never through mere
        membership.
        """

        if type(bids) is not tuple or not bids:
            raise ValueError("bids must be a non-empty exact tuple")
        open_bids = [
            value
            for value in bids
            if value.engine_id in open_engine_ids
        ]
        if not open_bids:
            raise ValueError("no open engine holds a remaining candidate")
        return min(
            open_bids,
            key=lambda value: (
                -(
                    value.bid
                    / (seats_awarded.get(value.engine_id, 0) + 1)
                ),
                value.engine_id,
            ),
        ).engine_id


__all__ = [
    "GEOMETRY_CONDITIONAL_ELASTICITY_POLICY_ID",
    "GEOMETRY_CONDITIONAL_ELASTICITY_POLICY_VERSION",
    "ElasticSeatBidder",
    "ElasticSeatConfig",
    "LaneGeometryEvidence",
    "LaneSeatBid",
]
