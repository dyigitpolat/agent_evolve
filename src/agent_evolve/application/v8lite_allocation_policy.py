"""V8-lite composition: repaired pilot and challenger, preserved terminal.

One sealed materialized-action market is allocated in three phases:

* PILOT — the rank-balanced causal pilot replaces the deterministic
  one-lane-head-per-engine design (V70 failure a): engine coverage floor,
  configurable top-heavy band mass, blocked randomization between the
  native-rank and frozen-score orders, and exact seat propensities.
* ADAPTIVE / CONTINUATION — the calibrated positive-gain challenger ranks
  every remaining candidate with no hard abstention (V70 failure b) and
  values only archive-conditioned gain (V70 failure c).  The existing
  outcome-adaptive controller is retained as the protected fallback: when
  the challenger's best score is non-positive, the incumbent decision is
  used unchanged.
* TERMINAL — EXACTLY the V7 terminal hierarchical-exploitation rule, by
  delegation to the frozen ``OutcomeAdaptiveActionRacingPolicy`` version 7
  instance.  The V7 rule worked prospectively and is not altered.

The policy knows no workload, objective, model, provider, or prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    ArchiveConditionedGainPort,
    CalibratedPositiveGainOpportunityPolicy,
    CalibratedPositiveGainRanking,
    ObjectivePoint,
    ObservedConversionOutcome,
    PositiveGainCandidate,
    PositiveGainForecast,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION,
    AdaptiveActionDescriptor,
    AdaptiveActionOutcome,
    AdaptiveActionRacingDecision,
    AdaptiveActionSetOutcome,
    OutcomeAdaptiveActionRacingPolicy,
)
from agent_evolve.application.rank_balanced_causal_pilot import (
    DEFAULT_PILOT_BAND_WEIGHTS,
    SEQUENTIAL_ADAPTIVE_BAND_PILOT_POLICY_ID,
    PilotSeatObservation,
    RankBalancedCausalPilotPolicy,
    RankBalancedPilotCandidate,
    RankBalancedPilotDesign,
    SequentialAdaptiveBandPilotPolicy,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json

V8LITE_ALLOCATION_POLICY_ID = "v8lite_allocation"
V8LITE_ALLOCATION_POLICY_VERSION_ID = "v8lite_r1"
V8LITE_ALLOCATION_POLICY_VERSION_ID_R2 = "v8lite_r2"
V8LITE_ALLOCATION_POLICY_VERSION_ID_R3 = "v8lite_r3"
V8LITE_ALLOCATION_POLICY_VERSION_ID_R4 = "v8lite_r4"
V8LITE_ALLOCATION_POLICY_VERSION = 1
_V8LITE_VERSION_IDS = frozenset(
    {
        V8LITE_ALLOCATION_POLICY_VERSION_ID,
        V8LITE_ALLOCATION_POLICY_VERSION_ID_R2,
        V8LITE_ALLOCATION_POLICY_VERSION_ID_R3,
        V8LITE_ALLOCATION_POLICY_VERSION_ID_R4,
    }
)
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_DEFINITION_DOMAIN = b"agent-evolve:v8lite-allocation-definition:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:v8lite-allocation-decision:v1\x00"

V8LITE_PHASE_PILOT = "pilot"
V8LITE_PHASE_ADAPTIVE = "adaptive"
V8LITE_PHASE_PROTECTED_FALLBACK = "protected_fallback"
V8LITE_PHASE_TERMINAL = "terminal"


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
class V8LiteAllocationConfig:
    """All V8-lite constants; no magic numbers live in policy code."""

    # Pilot phase.
    diagnostic_slots: int = 4
    band_count: int = 3
    band_weights: tuple[float, ...] = DEFAULT_PILOT_BAND_WEIGHTS
    exploration_epsilon: float = 0.125
    # Calibrated challenger.
    lambda_: float = 1.0
    beta: float = 1.0
    mixture_weight: float = 0.5
    prior_strength: float = 2.0
    probability_floor: float = 1.0e-6
    scenario_weights: tuple[float, float, float] = (0.25, 0.5, 0.25)
    frozen_score_weight: float = 0.125
    frozen_score_minimum_training_runs: int = 10
    # Shared scale and preserved V7 terminal/fallback controller.
    reference_gain_scale: float = 1.0e-4
    reference_gain_evidence_sha256: str = "0" * 64
    ucb_strength: float = 1.0
    counterfactual_strength: float = 0.5
    diversity_strength: float = 0.25
    randomized_audit_slots: int = 1
    randomized_audit_after_directed_steps: int = 1
    minimum_post_audit_optimization_slots: int = 1
    terminal_hierarchical_slots: int = 1
    native_rank_strength: float = 0.25
    exploration_pool_size: int = 4
    trace_alternative_count: int = 9
    random_seed: int = 0
    # r2-only constants; excluded from to_record so the r1 identity
    # bytes are preserved (the r2 identity adds them explicitly).
    adaptation_temperature: float = 1.0
    # r3 carries no new constant: the archive-geometry tie-break is a
    # strict ordering, not a weighted term.

    def __post_init__(self) -> None:
        # Structural bounds are owned by the composed policies; this
        # config only guards fields it consumes directly.
        if type(self.diagnostic_slots) is not int or self.diagnostic_slots <= 0:
            raise ValueError("diagnostic_slots must be positive")
        if (
            type(self.randomized_audit_slots) is not int
            or self.randomized_audit_slots <= 0
        ):
            raise ValueError("randomized_audit_slots must be positive")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "diagnostic_slots": self.diagnostic_slots,
            "band_count": self.band_count,
            "band_weights_hex": [
                value.hex() for value in self.band_weights
            ],
            "exploration_epsilon_hex": self.exploration_epsilon.hex(),
            "lambda_hex": self.lambda_.hex(),
            "beta_hex": self.beta.hex(),
            "mixture_weight_hex": self.mixture_weight.hex(),
            "prior_strength_hex": self.prior_strength.hex(),
            "probability_floor_hex": self.probability_floor.hex(),
            "scenario_weights_hex": [
                value.hex() for value in self.scenario_weights
            ],
            "frozen_score_weight_hex": self.frozen_score_weight.hex(),
            "frozen_score_minimum_training_runs": (
                self.frozen_score_minimum_training_runs
            ),
            "reference_gain_scale_hex": (
                self.reference_gain_scale.hex()
            ),
            "reference_gain_evidence_sha256": (
                self.reference_gain_evidence_sha256
            ),
            "ucb_strength_hex": self.ucb_strength.hex(),
            "counterfactual_strength_hex": (
                self.counterfactual_strength.hex()
            ),
            "diversity_strength_hex": self.diversity_strength.hex(),
            "randomized_audit_slots": self.randomized_audit_slots,
            "randomized_audit_after_directed_steps": (
                self.randomized_audit_after_directed_steps
            ),
            "minimum_post_audit_optimization_slots": (
                self.minimum_post_audit_optimization_slots
            ),
            "terminal_hierarchical_slots": (
                self.terminal_hierarchical_slots
            ),
            "native_rank_strength_hex": (
                self.native_rank_strength.hex()
            ),
            "exploration_pool_size": self.exploration_pool_size,
            "trace_alternative_count": self.trace_alternative_count,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True, slots=True)
class V8LiteDecision:
    """One phase-attributed V8-lite allocation step."""

    policy_id: str
    policy_version_id: str
    policy_definition_sha256: str
    residual_request_sha256: str
    phase: str
    authority_policy_id: str
    selected_action_sha256s: tuple[str, ...]
    selection_propensity: float
    evidence: FrozenJsonObject
    delegated_decision: AdaptiveActionRacingDecision | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    pilot_design: RankBalancedPilotDesign | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    challenger_ranking: CalibratedPositiveGainRanking | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    decision_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.policy_id, name="policy_id")
        _require_token(
            self.policy_version_id,
            name="policy_version_id",
        )
        require_sha256(
            self.policy_definition_sha256,
            "policy_definition_sha256",
        )
        require_sha256(
            self.residual_request_sha256,
            "residual_request_sha256",
        )
        if self.phase not in {
            V8LITE_PHASE_PILOT,
            V8LITE_PHASE_ADAPTIVE,
            V8LITE_PHASE_PROTECTED_FALLBACK,
            V8LITE_PHASE_TERMINAL,
        }:
            raise ValueError("phase is not a V8-lite phase")
        _require_token(
            self.authority_policy_id,
            name="authority_policy_id",
        )
        if (
            type(self.selected_action_sha256s) is not tuple
            or not self.selected_action_sha256s
            or self.selected_action_sha256s
            != tuple(sorted(set(self.selected_action_sha256s)))
        ):
            raise ValueError(
                "selected_action_sha256s must be a canonical exact tuple"
            )
        for value in self.selected_action_sha256s:
            require_sha256(value, "selected_action_sha256")
        if (
            type(self.selection_propensity) is not float
            or not math.isfinite(self.selection_propensity)
            or not 0.0 < self.selection_propensity <= 1.0
        ):
            raise ValueError(
                "selection_propensity must be a positive probability"
            )
        if (
            type(self.evidence) is not FrozenJsonObject
            or freeze_json(self.evidence) is not self.evidence
        ):
            raise TypeError("evidence must be an exact frozen object")
        object.__setattr__(
            self,
            "decision_sha256",
            _hash(_DECISION_DOMAIN, self._unsigned_record()),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version_id": self.policy_version_id,
                "definition_sha256": self.policy_definition_sha256,
            },
            "residual_request_sha256": self.residual_request_sha256,
            "phase": self.phase,
            "authority_policy_id": self.authority_policy_id,
            "selected_action_sha256s": list(
                self.selected_action_sha256s
            ),
            "selection_propensity_hex": (
                self.selection_propensity.hex()
            ),
            "delegated_decision_sha256": (
                None
                if self.delegated_decision is None
                else self.delegated_decision.decision_sha256
            ),
            "pilot_design_sha256": (
                None
                if self.pilot_design is None
                else self.pilot_design.design_sha256
            ),
            "challenger_ranking_sha256": (
                None
                if self.challenger_ranking is None
                else self.challenger_ranking.ranking_sha256
            ),
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._unsigned_record(),
            "decision_sha256": self.decision_sha256,
        }


@dataclass(frozen=True, slots=True)
class V8LiteAllocationPolicy:
    """Compose the repaired pilot and challenger over the preserved V7."""

    archive_gain_utility: ArchiveConditionedGainPort = field(
        repr=False,
        compare=False,
    )
    config: V8LiteAllocationConfig = V8LiteAllocationConfig()
    policy_id: str = V8LITE_ALLOCATION_POLICY_ID
    policy_version_id: str = V8LITE_ALLOCATION_POLICY_VERSION_ID
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.config) is not V8LiteAllocationConfig:
            raise TypeError("config must be an exact V8-lite config")
        self.config.__post_init__()
        _require_token(self.policy_id, name="policy_id")
        if (
            self.policy_id != V8LITE_ALLOCATION_POLICY_ID
            or self.policy_version_id not in _V8LITE_VERSION_IDS
        ):
            raise ValueError("policy identity is immutable")
        # Constructing every component validates the whole configuration
        # and binds component identities into this policy's definition.
        pilot = self.pilot_policy()
        challenger = self.challenger_policy()
        terminal = self.terminal_policy()
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(_DEFINITION_DOMAIN, self._identity(
                pilot=pilot,
                challenger=challenger,
                terminal=terminal,
            )),
        )

    @property
    def revision_2(self) -> bool:
        """r3 is r2 plus the challenger's archive-geometry channel."""

        return self.policy_version_id in {
            V8LITE_ALLOCATION_POLICY_VERSION_ID_R2,
            V8LITE_ALLOCATION_POLICY_VERSION_ID_R3,
            V8LITE_ALLOCATION_POLICY_VERSION_ID_R4,
        }

    @property
    def revision_3(self) -> bool:
        return self.policy_version_id in {
            V8LITE_ALLOCATION_POLICY_VERSION_ID_R3,
            V8LITE_ALLOCATION_POLICY_VERSION_ID_R4,
        }

    @property
    def revision_4(self) -> bool:
        """r4 is r3 plus the challenger's repaired tail-risk algebra."""

        return (
            self.policy_version_id
            == V8LITE_ALLOCATION_POLICY_VERSION_ID_R4
        )

    def _identity(
        self,
        *,
        pilot: object,
        challenger: CalibratedPositiveGainOpportunityPolicy,
        terminal: OutcomeAdaptiveActionRacingPolicy,
    ) -> dict[str, object]:
        identity = {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version_id": self.policy_version_id,
            "policy_version": V8LITE_ALLOCATION_POLICY_VERSION,
            "config": self.config.to_record(),
            "components": {
                "pilot": {
                    "policy_id": pilot.policy_id,
                    "policy_version": pilot.policy_version,
                    "definition_sha256": pilot.definition_sha256,
                },
                "challenger": {
                    "policy_id": challenger.policy_id,
                    "policy_version": challenger.policy_version,
                    "definition_sha256": challenger.definition_sha256,
                },
                "terminal_and_fallback": {
                    "policy_id": terminal.policy_id,
                    "policy_version": terminal.policy_version,
                    "definition_sha256": terminal.definition_sha256,
                },
            },
            "phases": {
                "pilot": "rank_balanced_causal_pilot",
                "continuation": (
                    "calibrated_positive_gain_challenger_with_"
                    "protected_v7_fallback"
                ),
                "terminal": (
                    "exact_v7_terminal_hierarchical_exploitation_"
                    "by_delegation"
                ),
            },
            "challenger_fallback_condition": (
                "top_calibrated_score_non_positive"
            ),
            "v7_terminal_rule_altered": False,
            "hard_abstention": False,
            "unobserved_outcomes_accepted": False,
            "workload_objective_model_provider_prompt_branches": False,
        }
        if self.revision_2:
            identity["phases"]["pilot"] = (
                "sequential_adaptive_band_pilot"
            )
            identity["r2"] = {
                "adaptation_temperature_hex": (
                    self.config.adaptation_temperature.hex()
                ),
                "pilot_width_cap": (
                    "max_engine_count_or_ceil_budget_over_three"
                ),
                "challenger_within_cell_tie_break": (
                    "native_rank_quality"
                ),
                "pilot_seats_sequential_with_revealed_outcomes": (
                    True
                ),
            }
        if self.revision_3:
            identity["r3"] = {
                "challenger_within_cell_tie_break": (
                    "max_min_chebyshev_dispersion_from_bought_anchors_"
                    "then_chebyshev_excess_over_current_archive_front"
                ),
                "tie_break_carries_no_weight_or_constant": True,
                "current_archive_required": True,
            }
        if self.revision_4:
            identity["r4"] = {
                "challenger_conversion_tail_risk": (
                    "expected_shortfall_below_the_conversion_branch_"
                    "own_expected_gain"
                ),
                "magnitude_channel_sign_preserved_below_half": True,
                "carries_no_new_constant": True,
            }
        return identity

    def identity_record(self) -> dict[str, object]:
        """Public identity dict binding every composed component."""

        self.__post_init__()
        return {
            **self._identity(
                pilot=self.pilot_policy(),
                challenger=self.challenger_policy(),
                terminal=self.terminal_policy(),
            ),
            "definition_sha256": self.definition_sha256,
        }

    def pilot_policy(
        self,
    ) -> RankBalancedCausalPilotPolicy | SequentialAdaptiveBandPilotPolicy:
        if self.revision_2:
            return SequentialAdaptiveBandPilotPolicy(
                band_count=self.config.band_count,
                band_weights=self.config.band_weights,
                exploration_epsilon=self.config.exploration_epsilon,
                adaptation_temperature=(
                    self.config.adaptation_temperature
                ),
                prior_strength=self.config.prior_strength,
                random_seed=self.config.random_seed,
            )
        return RankBalancedCausalPilotPolicy(
            band_count=self.config.band_count,
            band_weights=self.config.band_weights,
            exploration_epsilon=self.config.exploration_epsilon,
            random_seed=self.config.random_seed,
        )

    def pilot_width_for(
        self,
        *,
        evaluation_slots: int,
        engine_count: int,
    ) -> int:
        """Seats the pilot claims before the challenger takes over."""

        width = min(
            self.config.diagnostic_slots,
            evaluation_slots - self.config.randomized_audit_slots,
        )
        if self.revision_2:
            # r2 cap: coverage never exceeds the engine count or one
            # third of the budget, whichever is larger, so evidence-
            # driven challenger seats start earlier at wide budgets.
            width = min(
                width,
                max(
                    engine_count,
                    math.ceil(evaluation_slots / 3),
                ),
            )
        return width

    def challenger_policy(self) -> CalibratedPositiveGainOpportunityPolicy:
        return CalibratedPositiveGainOpportunityPolicy(
            archive_gain_utility=self.archive_gain_utility,
            lambda_=self.config.lambda_,
            beta=self.config.beta,
            mixture_weight=self.config.mixture_weight,
            prior_strength=self.config.prior_strength,
            band_count=self.config.band_count,
            reference_gain_scale=self.config.reference_gain_scale,
            probability_floor=self.config.probability_floor,
            scenario_weights=self.config.scenario_weights,
            frozen_score_weight=self.config.frozen_score_weight,
            frozen_score_minimum_training_runs=(
                self.config.frozen_score_minimum_training_runs
            ),
            within_cell_rank_tie_break=self.revision_2,
            anchor_geometry_tie_break=self.revision_3,
            downside_shortfall_tail_risk=self.revision_4,
        )

    def terminal_policy(self) -> OutcomeAdaptiveActionRacingPolicy:
        """The frozen V7 incumbent used for terminal seats and fallback."""

        return OutcomeAdaptiveActionRacingPolicy(
            diagnostic_slots=self.config.diagnostic_slots,
            randomized_audit_slots=self.config.randomized_audit_slots,
            reference_gain_scale=self.config.reference_gain_scale,
            reference_gain_evidence_sha256=(
                self.config.reference_gain_evidence_sha256
            ),
            prior_strength=self.config.prior_strength,
            ucb_strength=self.config.ucb_strength,
            counterfactual_strength=(
                self.config.counterfactual_strength
            ),
            diversity_strength=self.config.diversity_strength,
            randomized_audit_after_directed_steps=(
                self.config.randomized_audit_after_directed_steps
            ),
            exploration_pool_size=self.config.exploration_pool_size,
            trace_alternative_count=(
                self.config.trace_alternative_count
            ),
            minimum_post_audit_optimization_slots=(
                self.config.minimum_post_audit_optimization_slots
            ),
            terminal_hierarchical_slots=(
                self.config.terminal_hierarchical_slots
            ),
            native_rank_strength=self.config.native_rank_strength,
            random_seed=self.config.random_seed,
            policy_version=(
                OUTCOME_ADAPTIVE_ACTION_RACING_HORIZON_HIERARCHICAL_POLICY_VERSION
            ),
        )

    @staticmethod
    def _validated_forecasts(
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

    @staticmethod
    def _validated_anchors(
        anchor_points: tuple[tuple[str, ObjectivePoint], ...],
    ) -> dict[str, ObjectivePoint]:
        """Outcome-blind parent anchors, one per action at most."""

        if type(anchor_points) is not tuple:
            raise TypeError("anchor_points must be an exact tuple")
        result: dict[str, ObjectivePoint] = {}
        for value in anchor_points:
            if type(value) is not tuple or len(value) != 2:
                raise TypeError("anchor_points must pair action and point")
            action_sha256, point = value
            require_sha256(action_sha256, "anchor action_sha256")
            if type(point) is not tuple or not point:
                raise TypeError("anchor point must be an exact tuple")
            if action_sha256 in result:
                raise ValueError("anchor_points repeat an action")
            result[action_sha256] = point
        return result

    def design_pilot(
        self,
        *,
        residual_request_sha256: str,
        actions: tuple[AdaptiveActionDescriptor, ...],
        evaluation_slots: int,
    ) -> V8LiteDecision:
        """Phase one: rank-balanced randomized pilot over the market."""

        self.__post_init__()
        if self.revision_2:
            raise ValueError(
                "revision 2 pilots are sequential; use "
                "design_pilot_seat"
            )
        if type(actions) is not tuple or not actions:
            raise ValueError("actions must be a non-empty exact tuple")
        for value in actions:
            if type(value) is not AdaptiveActionDescriptor:
                raise TypeError("actions must contain exact descriptors")
            value.__post_init__()
        if (
            type(evaluation_slots) is not int
            or not 2 <= evaluation_slots <= len(actions)
        ):
            raise ValueError("evaluation_slots must fit the action market")
        pilot_width = self.pilot_width_for(
            evaluation_slots=evaluation_slots,
            engine_count=len(
                {value.lane_id for value in actions}
            ),
        )
        if pilot_width <= 0:
            raise ValueError("pilot design leaves no pilot")
        design = self.pilot_policy().design_pilot(
            residual_request_sha256=residual_request_sha256,
            candidates=tuple(
                RankBalancedPilotCandidate(
                    action_sha256=value.action_sha256,
                    engine_id=value.lane_id,
                    native_rank=value.native_rank,
                    frozen_score=value.prior_score,
                )
                for value in actions
            ),
            pilot_width=pilot_width,
        )
        return V8LiteDecision(
            policy_id=self.policy_id,
            policy_version_id=self.policy_version_id,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=residual_request_sha256,
            phase=V8LITE_PHASE_PILOT,
            authority_policy_id=design.policy_id,
            selected_action_sha256s=design.selected_action_sha256s,
            selection_propensity=design.design_propensity,
            evidence=freeze_json(
                {
                    "pilot_design": design.to_record(),
                    "candidate_outcomes_observed": False,
                }
            ),
            pilot_design=design,
        )

    def design_pilot_seat(
        self,
        *,
        residual_request_sha256: str,
        actions: tuple[AdaptiveActionDescriptor, ...],
        evaluation_slots: int,
        selected_action_sha256s: tuple[str, ...],
        outcomes: tuple[AdaptiveActionOutcome, ...],
    ) -> V8LiteDecision:
        """Revision-2 sequential pilot: one seat given revealed history."""

        self.__post_init__()
        if not self.revision_2:
            raise ValueError(
                "sequential pilot seats require revision 2"
            )
        if type(actions) is not tuple or not actions:
            raise ValueError("actions must be a non-empty exact tuple")
        for value in actions:
            if type(value) is not AdaptiveActionDescriptor:
                raise TypeError("actions must contain exact descriptors")
            value.__post_init__()
        if (
            type(evaluation_slots) is not int
            or not 2 <= evaluation_slots <= len(actions)
        ):
            raise ValueError("evaluation_slots must fit the action market")
        by_action = {value.action_sha256: value for value in actions}
        pilot_width = self.pilot_width_for(
            evaluation_slots=evaluation_slots,
            engine_count=len(
                {value.lane_id for value in actions}
            ),
        )
        if pilot_width <= 0:
            raise ValueError("pilot design leaves no pilot")
        if len(selected_action_sha256s) >= pilot_width:
            raise ValueError("the pilot is already complete")
        outcome_by_action = {
            value.action_sha256: value for value in outcomes
        }
        seat = self.pilot_policy().design_seat(
            residual_request_sha256=residual_request_sha256,
            candidates=tuple(
                RankBalancedPilotCandidate(
                    action_sha256=value.action_sha256,
                    engine_id=value.lane_id,
                    native_rank=value.native_rank,
                    frozen_score=value.prior_score,
                )
                for value in actions
            ),
            selected_action_sha256s=selected_action_sha256s,
            observations=tuple(
                PilotSeatObservation(
                    action_sha256=value.action_sha256,
                    feasible=value.feasible,
                    marginal_archive_gain=(
                        value.marginal_archive_gain
                    ),
                )
                for value in sorted(
                    outcomes,
                    key=lambda item: item.action_sha256,
                )
            ),
            seat_ordinal=len(selected_action_sha256s) + 1,
        )
        if seat.selected_action_sha256 not in by_action:
            raise AssertionError(  # pragma: no cover
                "pilot seat escaped the market"
            )
        return V8LiteDecision(
            policy_id=self.policy_id,
            policy_version_id=self.policy_version_id,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=residual_request_sha256,
            phase=V8LITE_PHASE_PILOT,
            authority_policy_id=(
                SEQUENTIAL_ADAPTIVE_BAND_PILOT_POLICY_ID
            ),
            selected_action_sha256s=(
                seat.selected_action_sha256,
            ),
            selection_propensity=seat.selection_propensity,
            evidence=freeze_json(
                {
                    "pilot_seat": seat.to_record(),
                    "pilot_width": pilot_width,
                    "sequential_adaptive_pilot": True,
                    "revealed_outcome_count": len(outcomes),
                }
            ),
        )

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
        forecasts: tuple[tuple[str, PositiveGainForecast], ...] = (),
        anchor_points: tuple[tuple[str, ObjectivePoint], ...] = (),
        frozen_fit_training_run_count: int = 0,
        prior_conversion_outcomes: tuple[
            ObservedConversionOutcome,
            ...,
        ] = (),
        set_outcomes: tuple[AdaptiveActionSetOutcome, ...] = (),
    ) -> V8LiteDecision:
        """Select one continuation action at the current cutoff.

        ``prior_conversion_outcomes`` carries conversion evidence that
        was fully observed BEFORE this market opened (for example a
        warm-start from earlier completed markets).  It is re-ordinaled
        ahead of the within-market outcomes, so the prequential
        no-future-leak invariant is preserved: nothing later than the
        current cutoff can enter the evidence stream.
        """

        self.__post_init__()
        require_sha256(
            residual_request_sha256,
            "residual_request_sha256",
        )
        if type(actions) is not tuple or not actions:
            raise ValueError("actions must be a non-empty exact tuple")
        for value in actions:
            if type(value) is not AdaptiveActionDescriptor:
                raise TypeError("actions must contain exact descriptors")
            value.__post_init__()
        if (
            type(evaluation_slots) is not int
            or not 2 <= evaluation_slots <= len(actions)
        ):
            raise ValueError("evaluation_slots must fit the action market")
        if (
            type(selected_action_sha256s) is not tuple
            or selected_action_sha256s
            != tuple(sorted(set(selected_action_sha256s)))
        ):
            raise ValueError(
                "selected_action_sha256s must be an exact canonical tuple"
            )
        seats_left = evaluation_slots - len(selected_action_sha256s)
        if seats_left <= 0:
            raise ValueError("the evaluation slate is already complete")
        terminal = seats_left <= self.config.terminal_hierarchical_slots
        incumbent = self.terminal_policy()
        if terminal:
            # EXACT V7 terminal hierarchical exploitation by delegation.
            delegated = incumbent.select_next(
                residual_request_sha256=residual_request_sha256,
                actions=actions,
                evaluation_slots=evaluation_slots,
                diagnostic_action_sha256s=diagnostic_action_sha256s,
                diagnostic_joint_gain=diagnostic_joint_gain,
                selected_action_sha256s=selected_action_sha256s,
                outcomes=outcomes,
                set_outcomes=set_outcomes,
            )
            return self._wrap_delegated(
                residual_request_sha256=residual_request_sha256,
                phase=V8LITE_PHASE_TERMINAL,
                delegated=delegated,
                extra_evidence={
                    "terminal_delegation": True,
                    "seats_left_before_decision": seats_left,
                },
            )

        forecast_by_action = self._validated_forecasts(forecasts)
        anchor_by_action = self._validated_anchors(anchor_points)
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
            raise ValueError("selected action is outside the sealed market")
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
            raise ValueError("no unevaluated action can fill the slate")
        if type(prior_conversion_outcomes) is not tuple or any(
            type(value) is not ObservedConversionOutcome
            for value in prior_conversion_outcomes
        ):
            raise TypeError(
                "prior_conversion_outcomes must contain exact "
                "conversion outcomes"
            )
        prior_evidence = tuple(
            ObservedConversionOutcome(
                observation_ordinal=ordinal,
                engine_id=value.engine_id,
                native_rank=value.native_rank,
                lane_size=value.lane_size,
                feasible=value.feasible,
                marginal_archive_gain=value.marginal_archive_gain,
            )
            for ordinal, value in enumerate(
                prior_conversion_outcomes,
                start=1,
            )
        )
        conversion_outcomes = prior_evidence + tuple(
            ObservedConversionOutcome(
                observation_ordinal=ordinal,
                engine_id=by_action[action_sha256].lane_id,
                native_rank=by_action[action_sha256].native_rank,
                lane_size=by_action[action_sha256].lane_size,
                feasible=outcome_by_action[action_sha256].feasible,
                marginal_archive_gain=(
                    outcome_by_action[
                        action_sha256
                    ].marginal_archive_gain
                ),
            )
            for ordinal, action_sha256 in enumerate(
                sorted(selected_action_sha256s),
                start=len(prior_evidence) + 1,
            )
        )
        ranking = self.challenger_policy().score_market(
            candidates=tuple(
                PositiveGainCandidate(
                    action_sha256=value.action_sha256,
                    engine_id=value.lane_id,
                    native_rank=value.native_rank,
                    lane_size=value.lane_size,
                    forecast=forecast_by_action.get(
                        value.action_sha256
                    ),
                    frozen_score=value.prior_score,
                    anchor_point=anchor_by_action.get(
                        value.action_sha256
                    ),
                )
                for value in remaining
            ),
            archive_points=archive_points,
            observed_outcomes=conversion_outcomes,
            future_seats_remaining=seats_left - 1,
            horizon_total=evaluation_slots,
            frozen_fit_training_run_count=(
                frozen_fit_training_run_count
            ),
            # The anchors this market has already paid to sample.  They
            # join the archive-geometry channel's reference front, which is
            # what stops a second seat landing in a region the first seat
            # already bought.
            covered_anchors=tuple(
                anchor_by_action[action_sha256]
                for action_sha256 in sorted(selected_action_sha256s)
                if action_sha256 in anchor_by_action
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
                        "seats_left_before_decision": seats_left,
                        "prior_conversion_evidence_count": len(
                            prior_evidence
                        ),
                        "unobserved_candidate_outcomes_available": (
                            False
                        ),
                    }
                ),
                challenger_ranking=ranking,
            )
        # Protected fallback: the existing controller decides unchanged.
        delegated = incumbent.select_next(
            residual_request_sha256=residual_request_sha256,
            actions=actions,
            evaluation_slots=evaluation_slots,
            diagnostic_action_sha256s=diagnostic_action_sha256s,
            diagnostic_joint_gain=diagnostic_joint_gain,
            selected_action_sha256s=selected_action_sha256s,
            outcomes=outcomes,
            set_outcomes=set_outcomes,
        )
        return self._wrap_delegated(
            residual_request_sha256=residual_request_sha256,
            phase=V8LITE_PHASE_PROTECTED_FALLBACK,
            delegated=delegated,
            extra_evidence={
                "protected_fallback_used": True,
                "fallback_reason": (
                    "challenger_top_score_non_positive"
                ),
                "challenger_ranking": ranking.to_record(
                    include_scores=True
                ),
                "seats_left_before_decision": seats_left,
            },
            challenger_ranking=ranking,
        )

    def _wrap_delegated(
        self,
        *,
        residual_request_sha256: str,
        phase: str,
        delegated: AdaptiveActionRacingDecision,
        extra_evidence: dict[str, object],
        challenger_ranking: CalibratedPositiveGainRanking | None = None,
    ) -> V8LiteDecision:
        return V8LiteDecision(
            policy_id=self.policy_id,
            policy_version_id=self.policy_version_id,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=residual_request_sha256,
            phase=phase,
            authority_policy_id=delegated.policy_id,
            selected_action_sha256s=(
                delegated.selected_action_sha256s
            ),
            selection_propensity=delegated.selection_propensity,
            evidence=freeze_json(
                {
                    **extra_evidence,
                    "delegated_decision": delegated.to_record(
                        include_evidence=True
                    ),
                }
            ),
            delegated_decision=delegated,
            challenger_ranking=challenger_ranking,
        )


__all__ = [
    "V8LITE_ALLOCATION_POLICY_ID",
    "V8LITE_ALLOCATION_POLICY_VERSION",
    "V8LITE_ALLOCATION_POLICY_VERSION_ID",
    "V8LITE_ALLOCATION_POLICY_VERSION_ID_R2",
    "V8LITE_ALLOCATION_POLICY_VERSION_ID_R3",
    "V8LITE_ALLOCATION_POLICY_VERSION_ID_R4",
    "V8LITE_PHASE_ADAPTIVE",
    "V8LITE_PHASE_PILOT",
    "V8LITE_PHASE_PROTECTED_FALLBACK",
    "V8LITE_PHASE_TERMINAL",
    "V8LiteAllocationConfig",
    "V8LiteAllocationPolicy",
    "V8LiteDecision",
]
