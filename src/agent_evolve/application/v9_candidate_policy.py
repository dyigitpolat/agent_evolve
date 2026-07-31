"""V9 candidate composition: v8lite_r2 plus config-gated R1/R2/R3 arms.

The frozen ``v8lite_r2`` composition stays the reference; this module
never modifies it.  Three independently config-gated refinements from the
jul28 pareto defect theory compose OVER it, each ablatable on its own:

* R1 (``region_conditional_credit``) — the continuation challenger's
  conversion credit moves from (engine x rank band) cells to
  (engine x parent-front-region x radius class) cells with the same
  Beta-shrinkage hierarchy, plus a learned demote-only forecast-trust
  channel;
* R2 (``head_mass_conditional_seat``) — when the calibrated model's
  predicted positive mass concentrates on one candidate strictly above a
  threshold, the FIRST seat becomes the deterministic argmax (an exact
  point-mass, propensity one) instead of a sampled pilot seat; and
* R3 (``geometry_conditional_elasticity``) — pilot lane selection walks
  D'Hondt over elastic per-lane bids (parent distance-to-front, forecast
  self-overlap saturation, revealed conversion) instead of the fixed
  coverage floor; the within-engine seat design (bands, blocked
  randomization, epsilon floor, exact rational propensities) is delegated
  unchanged to the sequential adaptive pilot.

With every flag off, every decision is delegated verbatim to the inner
``v8lite_r2`` policy, so the base arm is bit-identical to the reference.
Terminal seats are ALWAYS delegated through the inner policy, which itself
delegates to the frozen V7 terminal hierarchical-exploitation rule: no arm
alters the V7 terminal rule.  These arms carry NO live authority; they
exist for provider-free replay evaluation (gate M-lite v3).

The policy knows no workload, objective name, model, provider, or prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    ObjectivePoint,
    ObservedConversionOutcome,
    PositiveGainCandidate,
    PositiveGainForecast,
    _require_objective_point,
)
from agent_evolve.application.geometry_conditional_elasticity import (
    ElasticSeatBidder,
    ElasticSeatConfig,
    LaneGeometryEvidence,
)
from agent_evolve.application.head_mass_conditional_seat import (
    HeadMassSeatAssessor,
    HeadMassSeatConfig,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionDescriptor,
    AdaptiveActionOutcome,
    AdaptiveActionSetOutcome,
)
from agent_evolve.application.rank_balanced_causal_pilot import (
    PilotSeatObservation,
    RankBalancedPilotCandidate,
)
from agent_evolve.application.region_conditional_credit import (
    RegionConditionalChallengerPolicy,
    RegionConditionalOutcome,
    RegionCreditConfig,
    RegionFeatures,
    RegionScoredCandidate,
    parent_front_distance,
)
from agent_evolve.application.sequential_market_replay import (
    MarketRecord,
    ReplaySelection,
    ReplayStepReceipt,
    V8LiteReplayPolicy,
    _clamped_gain,
    _corpus_action_sha256,
    _normalized_point,
)
from agent_evolve.application.v8lite_allocation_policy import (
    V8LITE_ALLOCATION_POLICY_VERSION_ID_R2,
    V8LITE_PHASE_ADAPTIVE,
    V8LITE_PHASE_PILOT,
    V8LITE_PHASE_PROTECTED_FALLBACK,
    V8LiteAllocationConfig,
    V8LiteAllocationPolicy,
    V8LiteDecision,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import freeze_json

V9_CANDIDATE_POLICY_ID = "v9_candidate_allocation"
V9_CANDIDATE_POLICY_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_DEFINITION_DOMAIN = b"agent-evolve:v9-candidate-definition:v1\x00"


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


def v9_arm_version_id(*, r1: bool, r2: bool, r3: bool) -> str:
    """Deterministic arm token for one flag combination."""

    suffix = "".join(
        token
        for token, enabled in (("r1", r1), ("r2", r2), ("r3", r3))
        if enabled
    )
    return f"v9_{suffix}" if suffix else "v9_base"


@dataclass(frozen=True, slots=True)
class V9CandidateConfig:
    """Flags and component configs; the inner v8lite config is shared."""

    r1_region_conditional_credit: bool = False
    r2_head_mass_conditional_seat: bool = False
    r3_geometry_conditional_elasticity: bool = False
    base: V8LiteAllocationConfig = V8LiteAllocationConfig()
    credit: RegionCreditConfig = RegionCreditConfig()
    head: HeadMassSeatConfig = HeadMassSeatConfig()
    elastic: ElasticSeatConfig = ElasticSeatConfig()

    def __post_init__(self) -> None:
        for name in (
            "r1_region_conditional_credit",
            "r2_head_mass_conditional_seat",
            "r3_geometry_conditional_elasticity",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be exact")
        if type(self.base) is not V8LiteAllocationConfig:
            raise TypeError("base must be an exact v8lite config")
        self.base.__post_init__()
        if type(self.credit) is not RegionCreditConfig:
            raise TypeError("credit must be exact")
        self.credit.__post_init__()
        if type(self.head) is not HeadMassSeatConfig:
            raise TypeError("head must be exact")
        self.head.__post_init__()
        if type(self.elastic) is not ElasticSeatConfig:
            raise TypeError("elastic must be exact")
        self.elastic.__post_init__()

    @property
    def arm_version_id(self) -> str:
        return v9_arm_version_id(
            r1=self.r1_region_conditional_credit,
            r2=self.r2_head_mass_conditional_seat,
            r3=self.r3_geometry_conditional_elasticity,
        )

    def flags_record(self) -> dict[str, bool]:
        return {
            "r1": self.r1_region_conditional_credit,
            "r2": self.r2_head_mass_conditional_seat,
            "r3": self.r3_geometry_conditional_elasticity,
        }


def _validated_feature_map(
    region_features: tuple[tuple[str, RegionFeatures], ...],
) -> dict[str, RegionFeatures]:
    if type(region_features) is not tuple:
        raise TypeError("region_features must be an exact tuple")
    result: dict[str, RegionFeatures] = {}
    for value in region_features:
        if type(value) is not tuple or len(value) != 2:
            raise TypeError(
                "region_features must pair action and features"
            )
        action_sha256, features = value
        require_sha256(action_sha256, "feature action_sha256")
        if type(features) is not RegionFeatures:
            raise TypeError("features must be exact")
        features.__post_init__()
        if action_sha256 in result:
            raise ValueError("region_features repeat an action")
        result[action_sha256] = features
    return result


def _validated_forecast_map(
    forecasts: tuple[tuple[str, PositiveGainForecast], ...],
) -> dict[str, PositiveGainForecast]:
    if type(forecasts) is not tuple:
        raise TypeError("forecasts must be an exact tuple")
    result: dict[str, PositiveGainForecast] = {}
    for value in forecasts:
        if type(value) is not tuple or len(value) != 2:
            raise TypeError("forecasts must pair action and forecast")
        action_sha256, forecast = value
        require_sha256(action_sha256, "forecast action_sha256")
        if type(forecast) is not PositiveGainForecast:
            raise TypeError("forecast must be exact")
        forecast.__post_init__()
        if action_sha256 in result:
            raise ValueError("forecasts repeat an action")
        result[action_sha256] = forecast
    return result


def _validated_point_map(
    revealed_objective_points: tuple[tuple[str, ObjectivePoint], ...],
) -> dict[str, ObjectivePoint]:
    if type(revealed_objective_points) is not tuple:
        raise TypeError(
            "revealed_objective_points must be an exact tuple"
        )
    result: dict[str, ObjectivePoint] = {}
    for value in revealed_objective_points:
        if type(value) is not tuple or len(value) != 2:
            raise TypeError(
                "revealed_objective_points must pair action and point"
            )
        action_sha256, point = value
        require_sha256(action_sha256, "revealed action_sha256")
        _require_objective_point(point, name="revealed point")
        if action_sha256 in result:
            raise ValueError(
                "revealed_objective_points repeat an action"
            )
        result[action_sha256] = point
    return result


@dataclass(frozen=True, slots=True)
class V9CandidatePolicy:
    """Compose config-gated R1/R2/R3 over the frozen v8lite_r2 inner."""

    archive_gain_utility: object = field(repr=False, compare=False)
    config: V9CandidateConfig = V9CandidateConfig()
    policy_id: str = V9_CANDIDATE_POLICY_ID
    policy_version_id: str = field(init=False)
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.config) is not V9CandidateConfig:
            raise TypeError("config must be an exact v9 config")
        self.config.__post_init__()
        if (
            type(self.policy_id) is not str
            or _TOKEN.fullmatch(self.policy_id) is None
            or self.policy_id != V9_CANDIDATE_POLICY_ID
        ):
            raise ValueError("policy identity is immutable")
        object.__setattr__(
            self,
            "policy_version_id",
            self.config.arm_version_id,
        )
        inner = self.inner_policy()
        challenger = self.region_challenger()
        assessor = self.head_assessor()
        bidder = self.elastic_bidder()
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": V9_CANDIDATE_POLICY_VERSION,
                    "policy_version_id": self.policy_version_id,
                    "flags": self.config.flags_record(),
                    "inner": {
                        "policy_id": inner.policy_id,
                        "policy_version_id": inner.policy_version_id,
                        "definition_sha256": inner.definition_sha256,
                    },
                    "components": {
                        "r1_region_conditional_credit": {
                            "policy_id": challenger.policy_id,
                            "policy_version": (
                                challenger.policy_version
                            ),
                            "definition_sha256": (
                                challenger.definition_sha256
                            ),
                        },
                        "r2_head_mass_conditional_seat": {
                            "policy_id": assessor.policy_id,
                            "policy_version": assessor.policy_version,
                            "definition_sha256": (
                                assessor.definition_sha256
                            ),
                        },
                        "r3_geometry_conditional_elasticity": {
                            "policy_id": bidder.policy_id,
                            "policy_version": bidder.policy_version,
                            "definition_sha256": (
                                bidder.definition_sha256
                            ),
                        },
                    },
                    "base_arm_bit_identical_to_inner": True,
                    "v7_terminal_rule_altered": False,
                    "live_authority": False,
                    "workload_objective_model_provider_prompt_branches": (
                        False
                    ),
                },
            ),
        )

    @property
    def r1(self) -> bool:
        return self.config.r1_region_conditional_credit

    @property
    def r2(self) -> bool:
        return self.config.r2_head_mass_conditional_seat

    @property
    def r3(self) -> bool:
        return self.config.r3_geometry_conditional_elasticity

    def inner_policy(self) -> V8LiteAllocationPolicy:
        return V8LiteAllocationPolicy(
            archive_gain_utility=self.archive_gain_utility,
            policy_version_id=(
                V8LITE_ALLOCATION_POLICY_VERSION_ID_R2
            ),
            config=self.config.base,
        )

    def region_challenger(self) -> RegionConditionalChallengerPolicy:
        return RegionConditionalChallengerPolicy(
            base=self.inner_policy().challenger_policy(),
            credit=self.config.credit,
        )

    def head_assessor(self) -> HeadMassSeatAssessor:
        return HeadMassSeatAssessor(config=self.config.head)

    def elastic_bidder(self) -> ElasticSeatBidder:
        return ElasticSeatBidder(
            config=self.config.elastic,
            prior_strength=self.config.base.prior_strength,
        )

    def identity_record(self) -> dict[str, object]:
        """Public identity dict for one arm combination."""

        self.__post_init__()
        inner = self.inner_policy()
        challenger = self.region_challenger()
        assessor = self.head_assessor()
        bidder = self.elastic_bidder()
        return {
            "policy_id": self.policy_id,
            "policy_version": V9_CANDIDATE_POLICY_VERSION,
            "policy_version_id": self.policy_version_id,
            "flags": self.config.flags_record(),
            "definition_sha256": self.definition_sha256,
            "inner": inner.identity_record(),
            "components": {
                "r1_region_conditional_credit": {
                    "enabled": self.r1,
                    "policy_id": challenger.policy_id,
                    "definition_sha256": (
                        challenger.definition_sha256
                    ),
                },
                "r2_head_mass_conditional_seat": {
                    "enabled": self.r2,
                    "policy_id": assessor.policy_id,
                    "definition_sha256": assessor.definition_sha256,
                },
                "r3_geometry_conditional_elasticity": {
                    "enabled": self.r3,
                    "policy_id": bidder.policy_id,
                    "definition_sha256": bidder.definition_sha256,
                },
            },
            "v7_terminal_rule_altered": False,
            "live_authority": False,
        }

    # ------------------------------------------------------------------
    # Evidence construction shared by R1 scoring and R2 head assessment.
    # ------------------------------------------------------------------

    def _region_outcomes(
        self,
        *,
        by_action: dict[str, AdaptiveActionDescriptor],
        selected_action_sha256s: tuple[str, ...],
        outcomes: tuple[AdaptiveActionOutcome, ...],
        archive_points: tuple[ObjectivePoint, ...],
        reference_point: ObjectivePoint,
        feature_map: dict[str, RegionFeatures],
        forecast_map: dict[str, PositiveGainForecast],
        point_map: dict[str, ObjectivePoint],
        prior_conversion_outcomes: tuple[
            ObservedConversionOutcome,
            ...,
        ],
    ) -> tuple[RegionConditionalOutcome, ...]:
        """Prior evidence first, then within-market revealed outcomes.

        Prior (pre-market) evidence carries no comparable parent
        geometry, so it enters the hierarchy at the global and engine
        levels only (``region_id=None``).  Within-market outcomes are
        classified against the CURRENT archive; when the outcome's
        candidate carried a forecast, the (predicted, actual) direction
        pair versus the same base archive feeds the trust channel.
        """

        challenger = self.region_challenger()
        gain_port = self.archive_gain_utility
        result: list[RegionConditionalOutcome] = []
        for ordinal, value in enumerate(
            prior_conversion_outcomes,
            start=1,
        ):
            if type(value) is not ObservedConversionOutcome:
                raise TypeError(
                    "prior_conversion_outcomes must contain exact "
                    "conversion outcomes"
                )
            result.append(
                RegionConditionalOutcome(
                    observation_ordinal=ordinal,
                    engine_id=value.engine_id,
                    feasible=value.feasible,
                    marginal_archive_gain=(
                        value.marginal_archive_gain
                    ),
                )
            )
        outcome_by_action = {
            value.action_sha256: value for value in outcomes
        }
        for ordinal, action_sha256 in enumerate(
            sorted(selected_action_sha256s),
            start=len(result) + 1,
        ):
            outcome = outcome_by_action[action_sha256]
            descriptor = by_action[action_sha256]
            region_id, radius_class_id = challenger.region_for(
                archive_points=archive_points,
                reference_point=reference_point,
                features=feature_map.get(
                    action_sha256,
                    RegionFeatures(),
                ),
            )
            forecast = forecast_map.get(action_sha256)
            predicted: bool | None = None
            actual: bool | None = None
            if forecast is not None:
                predicted = (
                    gain_port.marginal_archive_gain(
                        archive_points,
                        forecast.point("p50"),
                    )
                    > 0.0
                )
                point = point_map.get(action_sha256)
                actual = (
                    point is not None
                    and gain_port.marginal_archive_gain(
                        archive_points,
                        point,
                    )
                    > 0.0
                )
            result.append(
                RegionConditionalOutcome(
                    observation_ordinal=ordinal,
                    engine_id=descriptor.lane_id,
                    feasible=outcome.feasible,
                    marginal_archive_gain=(
                        outcome.marginal_archive_gain
                    ),
                    region_id=region_id,
                    radius_class_id=radius_class_id,
                    forecast_predicted_positive=predicted,
                    forecast_actual_positive=actual,
                )
            )
        return tuple(result)

    def _score_candidates(
        self,
        *,
        candidates: tuple[AdaptiveActionDescriptor, ...],
        archive_points: tuple[ObjectivePoint, ...],
        reference_point: ObjectivePoint,
        feature_map: dict[str, RegionFeatures],
        forecast_map: dict[str, PositiveGainForecast],
        observed_outcomes: tuple[RegionConditionalOutcome, ...],
        future_seats_remaining: int,
        horizon_total: int,
        frozen_fit_training_run_count: int,
    ):
        """Rank candidates with the arm's active calibrated model.

        With R1 on, the region-conditional challenger scores with full
        region evidence and learned trust.  With R1 off, the SAME
        scorer runs with features and leaf evidence stripped, so every
        estimate collapses to the engine/global levels of the shrinkage
        hierarchy and the trust multiplier stays exactly one.
        """

        challenger = self.region_challenger()
        scored = tuple(
            RegionScoredCandidate(
                candidate=PositiveGainCandidate(
                    action_sha256=value.action_sha256,
                    engine_id=value.lane_id,
                    native_rank=value.native_rank,
                    lane_size=value.lane_size,
                    forecast=forecast_map.get(value.action_sha256),
                    frozen_score=value.prior_score,
                ),
                features=(
                    feature_map.get(
                        value.action_sha256,
                        RegionFeatures(),
                    )
                    if self.r1
                    else RegionFeatures()
                ),
            )
            for value in candidates
        )
        outcomes = (
            observed_outcomes
            if self.r1
            else tuple(
                RegionConditionalOutcome(
                    observation_ordinal=value.observation_ordinal,
                    engine_id=value.engine_id,
                    feasible=value.feasible,
                    marginal_archive_gain=(
                        value.marginal_archive_gain
                    ),
                )
                for value in observed_outcomes
            )
        )
        return challenger.score_market(
            candidates=scored,
            archive_points=archive_points,
            reference_point=reference_point,
            observed_outcomes=outcomes,
            future_seats_remaining=future_seats_remaining,
            horizon_total=horizon_total,
            frozen_fit_training_run_count=(
                frozen_fit_training_run_count
            ),
        )

    # ------------------------------------------------------------------
    # Pilot seats.
    # ------------------------------------------------------------------

    def design_pilot_seat(
        self,
        *,
        residual_request_sha256: str,
        actions: tuple[AdaptiveActionDescriptor, ...],
        evaluation_slots: int,
        selected_action_sha256s: tuple[str, ...],
        outcomes: tuple[AdaptiveActionOutcome, ...],
        archive_points: tuple[ObjectivePoint, ...] = (),
        reference_point: ObjectivePoint | None = None,
        region_features: tuple[
            tuple[str, RegionFeatures],
            ...,
        ] = (),
        forecasts: tuple[
            tuple[str, PositiveGainForecast],
            ...,
        ] = (),
        prior_conversion_outcomes: tuple[
            ObservedConversionOutcome,
            ...,
        ] = (),
        frozen_fit_training_run_count: int = 0,
    ) -> V8LiteDecision:
        """One pilot seat under the arm's gated pilot refinements."""

        self.__post_init__()
        inner = self.inner_policy()
        if not (self.r2 or self.r3):
            return inner.design_pilot_seat(
                residual_request_sha256=residual_request_sha256,
                actions=actions,
                evaluation_slots=evaluation_slots,
                selected_action_sha256s=selected_action_sha256s,
                outcomes=outcomes,
            )
        feature_map = _validated_feature_map(region_features)
        forecast_map = _validated_forecast_map(forecasts)
        by_action = {value.action_sha256: value for value in actions}
        seat_ordinal = len(selected_action_sha256s) + 1
        pilot_width = inner.pilot_width_for(
            evaluation_slots=evaluation_slots,
            engine_count=len(
                {value.lane_id for value in actions}
            ),
        )
        if len(selected_action_sha256s) >= pilot_width:
            raise ValueError("the pilot is already complete")

        if (
            self.r2
            and seat_ordinal == 1
            and archive_points
            and reference_point is not None
        ):
            ranking = self._score_candidates(
                candidates=actions,
                archive_points=archive_points,
                reference_point=reference_point,
                feature_map=feature_map,
                forecast_map=forecast_map,
                observed_outcomes=self._region_outcomes(
                    by_action=by_action,
                    selected_action_sha256s=(),
                    outcomes=(),
                    archive_points=archive_points,
                    reference_point=reference_point,
                    feature_map=feature_map,
                    forecast_map=forecast_map,
                    point_map={},
                    prior_conversion_outcomes=(
                        prior_conversion_outcomes
                    ),
                ),
                future_seats_remaining=evaluation_slots - 1,
                horizon_total=evaluation_slots,
                frozen_fit_training_run_count=(
                    frozen_fit_training_run_count
                ),
            )
            assessment = self.head_assessor().assess(ranking)
            if assessment.fired:
                return V8LiteDecision(
                    policy_id=self.policy_id,
                    policy_version_id=self.policy_version_id,
                    policy_definition_sha256=self.definition_sha256,
                    residual_request_sha256=(
                        residual_request_sha256
                    ),
                    phase=V8LITE_PHASE_PILOT,
                    authority_policy_id=(
                        self.head_assessor().policy_id
                    ),
                    selected_action_sha256s=(
                        assessment.argmax_action_sha256,
                    ),
                    selection_propensity=1.0,
                    evidence=freeze_json(
                        {
                            "head_mass_seat": (
                                assessment.to_record()
                            ),
                            "seat_ordinal": 1,
                            "deterministic_argmax_seat": True,
                            "support_propensities": [
                                {
                                    "action_sha256": (
                                        assessment.argmax_action_sha256
                                    ),
                                    "propensity_hex": (1.0).hex(),
                                }
                            ],
                            "remaining_seats_stochastic": True,
                            "candidate_outcomes_observed": False,
                        }
                    ),
                )

        if not self.r3:
            return inner.design_pilot_seat(
                residual_request_sha256=residual_request_sha256,
                actions=actions,
                evaluation_slots=evaluation_slots,
                selected_action_sha256s=selected_action_sha256s,
                outcomes=outcomes,
            )

        # R3: elastic lane bids choose the engine; the within-engine
        # seat is delegated to the inner sequential adaptive pilot over
        # the chosen lane only (band adaptation therefore pools within
        # the lane, which the definition sha records).
        selected = set(selected_action_sha256s)
        outcome_by_action = {
            value.action_sha256: value for value in outcomes
        }
        lanes: dict[str, list[AdaptiveActionDescriptor]] = {}
        for value in actions:
            lanes.setdefault(value.lane_id, []).append(value)
        gain_port = self.archive_gain_utility
        lane_evidence: list[LaneGeometryEvidence] = []
        for engine_id in sorted(lanes):
            members = lanes[engine_id]
            distances: list[float] = []
            predicted: list[bool] = []
            revealed: list[bool] = []
            for value in members:
                features = feature_map.get(value.action_sha256)
                if (
                    features is not None
                    and features.parent_point is not None
                ):
                    distances.append(
                        parent_front_distance(
                            archive_points,
                            features.parent_point,
                        )
                    )
                forecast = forecast_map.get(value.action_sha256)
                if forecast is not None and archive_points:
                    predicted.append(
                        gain_port.marginal_archive_gain(
                            archive_points,
                            forecast.point("p50"),
                        )
                        <= 0.0
                    )
                outcome = outcome_by_action.get(value.action_sha256)
                if outcome is not None:
                    revealed.append(
                        outcome.marginal_archive_gain > 0.0
                    )
            lane_evidence.append(
                LaneGeometryEvidence(
                    engine_id=engine_id,
                    parent_front_distances=tuple(distances),
                    predicted_dominated=tuple(predicted),
                    revealed_positive=tuple(revealed),
                )
            )
        bidder = self.elastic_bidder()
        bids = bidder.lane_bids(tuple(lane_evidence))
        seats_awarded = {
            engine_id: sum(
                by_action[value].lane_id == engine_id
                for value in selected_action_sha256s
            )
            for engine_id in lanes
        }
        open_engine_ids = frozenset(
            engine_id
            for engine_id, members in lanes.items()
            if any(
                value.action_sha256 not in selected
                for value in members
            )
        )
        engine_id = bidder.choose_engine(
            bids=bids,
            seats_awarded=seats_awarded,
            open_engine_ids=open_engine_ids,
        )
        lane_members = sorted(
            lanes[engine_id],
            key=lambda value: (
                value.native_rank,
                value.action_sha256,
            ),
        )
        lane_candidates = tuple(
            RankBalancedPilotCandidate(
                action_sha256=value.action_sha256,
                engine_id=value.lane_id,
                native_rank=value.native_rank,
                frozen_score=value.prior_score,
            )
            for value in lane_members
        )
        lane_selected = tuple(
            sorted(
                value
                for value in selected_action_sha256s
                if by_action[value].lane_id == engine_id
            )
        )
        lane_observations = tuple(
            PilotSeatObservation(
                action_sha256=value,
                feasible=outcome_by_action[value].feasible,
                marginal_archive_gain=(
                    outcome_by_action[value].marginal_archive_gain
                ),
            )
            for value in lane_selected
            if value in outcome_by_action
        )
        seat = inner.pilot_policy().design_seat(
            residual_request_sha256=residual_request_sha256,
            candidates=lane_candidates,
            selected_action_sha256s=lane_selected,
            observations=lane_observations,
            seat_ordinal=seat_ordinal,
        )
        return V8LiteDecision(
            policy_id=self.policy_id,
            policy_version_id=self.policy_version_id,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=residual_request_sha256,
            phase=V8LITE_PHASE_PILOT,
            authority_policy_id=self.elastic_bidder().policy_id,
            selected_action_sha256s=(
                seat.selected_action_sha256,
            ),
            selection_propensity=seat.selection_propensity,
            evidence=freeze_json(
                {
                    "elastic_lane_bids": [
                        value.to_record() for value in bids
                    ],
                    "chosen_engine_id": engine_id,
                    "seats_awarded_before": {
                        key: value
                        for key, value in sorted(
                            seats_awarded.items()
                        )
                    },
                    "pilot_seat": seat.to_record(),
                    "seat_ordinal": seat_ordinal,
                    "fixed_coverage_floor_used": False,
                    "candidate_outcomes_observed_before_seat": len(
                        outcomes
                    ),
                }
            ),
        )

    # ------------------------------------------------------------------
    # Continuation seats.
    # ------------------------------------------------------------------

    def select_next(
        self,
        *,
        residual_request_sha256: str,
        actions: tuple[AdaptiveActionDescriptor, ...],
        evaluation_slots: int,
        diagnostic_action_sha256s: tuple[str, ...],
        diagnostic_joint_gain: float,
        selected_action_sha256s: tuple[str, ...],
        outcomes: tuple[AdaptiveActionOutcome, ...],
        archive_points: tuple[ObjectivePoint, ...],
        reference_point: ObjectivePoint | None = None,
        region_features: tuple[
            tuple[str, RegionFeatures],
            ...,
        ] = (),
        forecasts: tuple[
            tuple[str, PositiveGainForecast],
            ...,
        ] = (),
        revealed_objective_points: tuple[
            tuple[str, ObjectivePoint],
            ...,
        ] = (),
        frozen_fit_training_run_count: int = 0,
        prior_conversion_outcomes: tuple[
            ObservedConversionOutcome,
            ...,
        ] = (),
        set_outcomes: tuple[AdaptiveActionSetOutcome, ...] = (),
    ) -> V8LiteDecision:
        """Select one continuation action at the current cutoff."""

        self.__post_init__()
        inner = self.inner_policy()
        seats_left = evaluation_slots - len(selected_action_sha256s)
        terminal = (
            seats_left <= self.config.base.terminal_hierarchical_slots
        )
        if terminal or not self.r1 or reference_point is None:
            # Terminal seats: EXACT V7 delegation through the inner
            # policy; non-R1 arms: the inner challenger unchanged.
            return inner.select_next(
                residual_request_sha256=residual_request_sha256,
                actions=actions,
                evaluation_slots=evaluation_slots,
                diagnostic_action_sha256s=diagnostic_action_sha256s,
                diagnostic_joint_gain=diagnostic_joint_gain,
                selected_action_sha256s=selected_action_sha256s,
                outcomes=outcomes,
                archive_points=archive_points,
                forecasts=forecasts,
                frozen_fit_training_run_count=(
                    frozen_fit_training_run_count
                ),
                prior_conversion_outcomes=(
                    prior_conversion_outcomes
                ),
                set_outcomes=set_outcomes,
            )
        feature_map = _validated_feature_map(region_features)
        forecast_map = _validated_forecast_map(forecasts)
        point_map = _validated_point_map(revealed_objective_points)
        by_action = {value.action_sha256: value for value in actions}
        outcome_by_action = {
            value.action_sha256: value for value in outcomes
        }
        if set(outcome_by_action) != set(selected_action_sha256s):
            raise ValueError(
                "observations must exactly cover all previously "
                "selected actions"
            )
        if not set(selected_action_sha256s) <= set(by_action):
            raise ValueError(
                "selected action is outside the sealed market"
            )
        selected_phenotypes = {
            by_action[value].phenotype_sha256
            for value in selected_action_sha256s
        }
        remaining = tuple(
            value
            for value in actions
            if value.action_sha256 not in outcome_by_action
            and value.phenotype_sha256 not in selected_phenotypes
        )
        if not remaining:
            raise ValueError(
                "no unevaluated action can fill the slate"
            )
        observed = self._region_outcomes(
            by_action=by_action,
            selected_action_sha256s=selected_action_sha256s,
            outcomes=outcomes,
            archive_points=archive_points,
            reference_point=reference_point,
            feature_map=feature_map,
            forecast_map=forecast_map,
            point_map=point_map,
            prior_conversion_outcomes=prior_conversion_outcomes,
        )
        ranking = self._score_candidates(
            candidates=remaining,
            archive_points=archive_points,
            reference_point=reference_point,
            feature_map=feature_map,
            forecast_map=forecast_map,
            observed_outcomes=observed,
            future_seats_remaining=seats_left - 1,
            horizon_total=evaluation_slots,
            frozen_fit_training_run_count=(
                frozen_fit_training_run_count
            ),
        )
        top_action_sha256 = ranking.ranked_action_sha256s[0]
        top_score = ranking.score_for(top_action_sha256)
        if top_score.score > 0.0:
            return V8LiteDecision(
                policy_id=self.policy_id,
                policy_version_id=self.policy_version_id,
                policy_definition_sha256=self.definition_sha256,
                residual_request_sha256=residual_request_sha256,
                phase=V8LITE_PHASE_ADAPTIVE,
                authority_policy_id=ranking.policy_id,
                selected_action_sha256s=(top_action_sha256,),
                selection_propensity=1.0,
                evidence=freeze_json(
                    {
                        "challenger_ranking": ranking.to_record(
                            include_scores=True
                        ),
                        "selected_score_sha256": (
                            top_score.score_sha256
                        ),
                        "protected_fallback_used": False,
                        "region_conditional_credit": True,
                        "seats_left_before_decision": seats_left,
                        "prior_conversion_evidence_count": len(
                            prior_conversion_outcomes
                        ),
                        "unobserved_candidate_outcomes_available": (
                            False
                        ),
                    }
                ),
                challenger_ranking=ranking,
            )
        # Protected fallback: the frozen V7 incumbent decides, exactly
        # as the inner v8lite composition falls back.
        delegated = inner.terminal_policy().select_next(
            residual_request_sha256=residual_request_sha256,
            actions=actions,
            evaluation_slots=evaluation_slots,
            diagnostic_action_sha256s=diagnostic_action_sha256s,
            diagnostic_joint_gain=diagnostic_joint_gain,
            selected_action_sha256s=selected_action_sha256s,
            outcomes=outcomes,
            set_outcomes=set_outcomes,
        )
        return V8LiteDecision(
            policy_id=self.policy_id,
            policy_version_id=self.policy_version_id,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=residual_request_sha256,
            phase=V8LITE_PHASE_PROTECTED_FALLBACK,
            authority_policy_id=delegated.policy_id,
            selected_action_sha256s=(
                delegated.selected_action_sha256s
            ),
            selection_propensity=delegated.selection_propensity,
            evidence=freeze_json(
                {
                    "protected_fallback_used": True,
                    "fallback_reason": (
                        "challenger_top_score_non_positive"
                    ),
                    "region_conditional_credit": True,
                    "challenger_ranking": ranking.to_record(
                        include_scores=True
                    ),
                    "seats_left_before_decision": seats_left,
                    "delegated_decision": delegated.to_record(
                        include_evidence=True
                    ),
                }
            ),
            delegated_decision=delegated,
            challenger_ranking=ranking,
        )


class V9ReplayPolicy:
    """Drive one V9 arm inside the sealed-market replay boundary.

    The universe, descriptors, and outcome-blind request identity are
    built by the SAME code the v8lite adapter uses (delegated to an
    internal ``V8LiteReplayPolicy``), so the base arm is bit-identical
    to the reference.  Region features, forecasts, and revealed
    objective points are keyed by action and passed through outcome-
    blind: revealed points cover only already-revealed candidates.
    """

    def __init__(
        self,
        policy: V9CandidatePolicy,
        *,
        frozen_fit_training_run_count: int = 0,
        prior_conversion_outcomes: tuple[
            ObservedConversionOutcome,
            ...,
        ] = (),
        region_features: tuple[
            tuple[str, RegionFeatures],
            ...,
        ] = (),
    ) -> None:
        if type(policy) is not V9CandidatePolicy:
            raise TypeError("policy must be an exact V9 policy")
        policy.__post_init__()
        self.policy_id = (
            f"replay_adapter.{policy.policy_version_id}"
        )
        self._policy = policy
        self._frozen_fit_training_run_count = (
            frozen_fit_training_run_count
        )
        self._prior_conversion_outcomes = prior_conversion_outcomes
        self._region_features = _validated_feature_map(
            region_features
        )
        self._helper = V8LiteReplayPolicy(
            policy.inner_policy(),
            frozen_fit_training_run_count=(
                frozen_fit_training_run_count
            ),
            prior_conversion_outcomes=prior_conversion_outcomes,
        )

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
        descriptors = self._helper._descriptors(record, universe_ids)
        request_sha256 = self._helper._outcome_blind_request_sha256(
            record,
            universe_ids,
        )
        pilot_width = self._policy.inner_policy().pilot_width_for(
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
        forecasts = tuple(
            (value.action_sha256, value.forecast)
            for value in (
                record.candidate(item) for item in universe_ids
            )
            if value.forecast is not None
        )
        region_features = tuple(
            sorted(
                (action_sha256, features)
                for action_sha256, features in (
                    self._region_features.items()
                )
                if action_sha256 in set(universe_ids)
            )
        )
        if step_index < pilot_width:
            decision = self._policy.design_pilot_seat(
                residual_request_sha256=request_sha256,
                actions=descriptors,
                evaluation_slots=budget,
                selected_action_sha256s=selected_ids,
                outcomes=outcomes,
                archive_points=record.archive_points,
                reference_point=record.hv_reference_point,
                region_features=region_features,
                forecasts=forecasts,
                prior_conversion_outcomes=(
                    self._prior_conversion_outcomes
                ),
                frozen_fit_training_run_count=(
                    self._frozen_fit_training_run_count
                ),
            )
            return ReplaySelection(
                action_sha256=decision.selected_action_sha256s[0],
                selection_propensity=decision.selection_propensity,
                evidence={
                    "phase": decision.phase,
                    "authority_policy_id": (
                        decision.authority_policy_id
                    ),
                },
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
        revealed_objective_points = tuple(
            (value, record.candidate(value).objectives)
            for value in selected_ids
            if record.candidate(value).objectives is not None
        )
        decision = self._policy.select_next(
            residual_request_sha256=request_sha256,
            actions=descriptors,
            evaluation_slots=budget,
            diagnostic_action_sha256s=pilot_ids,
            diagnostic_joint_gain=diagnostic_joint_gain,
            selected_action_sha256s=selected_ids,
            outcomes=outcomes,
            archive_points=record.archive_points,
            reference_point=record.hv_reference_point,
            region_features=region_features,
            forecasts=forecasts,
            revealed_objective_points=revealed_objective_points,
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
            evidence={
                "phase": decision.phase,
                "authority_policy_id": (
                    decision.authority_policy_id
                ),
            },
        )


def region_features_from_corpus(
    payload: dict[str, object],
) -> tuple[tuple[str, RegionFeatures], ...]:
    """Outcome-blind provenance features from one corpus market payload.

    Parent objective points are normalized onto the SAME axes frame the
    replay loader uses; candidates without a recorded parent objective
    vector or radius degrade to absent features (region ``no_parent``,
    radius class ``none``), so markets without provenance stay at the
    hierarchy's engine level by construction.
    """

    market_id = str(payload["market_id"])
    axes = tuple(payload["hv_reference_point"]["axes"])
    metric_ids = tuple(
        sorted(str(axis["metric_id"]) for axis in axes)
    )
    result: list[tuple[str, RegionFeatures]] = []
    for raw in payload["candidates"]:
        action_sha256 = _corpus_action_sha256(market_id, raw)
        parent = raw.get("parent")
        parent_point: ObjectivePoint | None = None
        if isinstance(parent, dict):
            parent_objectives = parent.get("objectives")
            if isinstance(parent_objectives, dict) and set(
                metric_ids
            ) <= set(parent_objectives):
                parent_point = _normalized_point(
                    {
                        metric_id: float(
                            parent_objectives[metric_id]
                        )
                        for metric_id in metric_ids
                    },
                    axes,
                )
        radius = raw.get("radius")
        result.append(
            (
                action_sha256,
                RegionFeatures(
                    parent_point=parent_point,
                    radius=(
                        int(radius)
                        if isinstance(radius, int)
                        and not isinstance(radius, bool)
                        and radius >= 0
                        else None
                    ),
                ),
            )
        )
    return tuple(sorted(result))


__all__ = [
    "V9_CANDIDATE_POLICY_ID",
    "V9_CANDIDATE_POLICY_VERSION",
    "V9CandidateConfig",
    "V9CandidatePolicy",
    "V9ReplayPolicy",
    "region_features_from_corpus",
    "v9_arm_version_id",
]
