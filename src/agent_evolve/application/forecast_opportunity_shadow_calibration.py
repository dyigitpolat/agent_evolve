"""Project a same-prefix forecast shadow into calibration evidence.

The projector joins only authenticated, already-observed records:

* the protected forecast-opportunity decision and its sealed raw score;
* the portable action descriptor used before evaluation; and
* the final same-prefix challenger-versus-fallback observation.

It does not inspect workload schemas, objective names, prompts, models, or
providers.  The resulting observation can therefore enter the generic
prequential calibration memory without adding a workload integration method.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionDescriptor,
    AdaptiveActionRacingDecision,
)
from agent_evolve.application.prequential_archive_opportunity_calibration import (
    ArchiveOpportunityActionContext,
    ArchiveOpportunityCalibrationEvidenceRole,
    ArchiveOpportunityCalibrationObservation,
    ArchiveOpportunityCalibrationRequest,
)
from agent_evolve.application.same_prefix_paired_audit import (
    FORECAST_OPPORTUNITY_SAME_PREFIX_AUDIT_DESIGNER_IDS,
    FORECAST_OPPORTUNITY_SAME_PREFIX_SHADOW_DESIGNER_ID,
    FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID,
    SamePrefixPairedAuditArm,
    SamePrefixPairedAuditObservation,
)
from agent_evolve.domain.typed_json import thaw_json


FORECAST_OPPORTUNITY_SHADOW_CALIBRATION_PROJECTOR_ID = (
    "forecast_opportunity_shadow_calibration"
)
FORECAST_OPPORTUNITY_SHADOW_CALIBRATION_PROJECTOR_VERSION = 2
_DEFINITION_DOMAIN = (
    b"agent-evolve:forecast-opportunity-shadow-calibration-projector:v2\x00"
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


def _read_nonnegative_hex(record: dict[str, object], key: str) -> float:
    raw = record.get(key)
    if type(raw) is not str:
        raise TypeError(f"{key} must be an exact hexadecimal float")
    try:
        value = float.fromhex(raw)
    except ValueError as error:
        raise ValueError(f"{key} is not a hexadecimal float") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be finite and non-negative")
    return float(value)


@dataclass(frozen=True, slots=True)
class ForecastOpportunityShadowCalibrationProjector:
    """Create one calibration target from a sealed same-prefix forecast arm."""

    projector_id: str = (
        FORECAST_OPPORTUNITY_SHADOW_CALIBRATION_PROJECTOR_ID
    )
    projector_version: int = (
        FORECAST_OPPORTUNITY_SHADOW_CALIBRATION_PROJECTOR_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.projector_id
            != FORECAST_OPPORTUNITY_SHADOW_CALIBRATION_PROJECTOR_ID
            or self.projector_version
            != FORECAST_OPPORTUNITY_SHADOW_CALIBRATION_PROJECTOR_VERSION
        ):
            raise ValueError("shadow calibration projector identity is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "projector_id": self.projector_id,
                    "projector_version": self.projector_version,
                    "forecast_source": (
                        "authenticated-protected-decision-raw-score"
                    ),
                    "target": (
                        "exact-forecast-arm-conditional-gain-at-identical-"
                        "prefix"
                    ),
                    "accepted_designers": sorted(
                        FORECAST_OPPORTUNITY_SAME_PREFIX_AUDIT_DESIGNER_IDS
                    ),
                    "fallback_target_retained_in_paired_observation": True,
                    "paired_evidence_role_and_sampling_propensity_recorded": (
                        True
                    ),
                    "current_or_future_candidate_outcomes_used_for_forecast": (
                        False
                    ),
                    "workload_objective_model_provider_prompt_config_"
                    "branches": False,
                },
            ),
        )

    @staticmethod
    def _score_record(
        decision: AdaptiveActionRacingDecision,
        action_sha256: str,
        *,
        require_recommended: bool,
    ) -> dict[str, object]:
        evidence = thaw_json(decision.evidence)
        ranking = evidence.get("opportunity_ranking")
        if type(ranking) is not dict:
            raise TypeError("decision lacks an exact opportunity ranking")
        recommended = ranking.get("recommended_action_sha256s")
        scores = ranking.get("scores")
        if (
            type(recommended) is not list
            or type(scores) is not list
        ):
            raise TypeError("decision opportunity ranking is incomplete")
        if require_recommended and action_sha256 not in recommended:
            raise ValueError(
                "legacy shadow challenger is not a recommendation"
            )
        matches = [
            value
            for value in scores
            if type(value) is dict
            and value.get("action_sha256") == action_sha256
        ]
        if len(matches) != 1:
            raise ValueError(
                "opportunity ranking must expose one challenger score"
            )
        return matches[0]

    def project(
        self,
        *,
        decision_index: int,
        evidence_cutoff_ordinal: int,
        decision: AdaptiveActionRacingDecision,
        action: AdaptiveActionDescriptor,
        observation: SamePrefixPairedAuditObservation,
    ) -> ArchiveOpportunityCalibrationObservation:
        """Join the sealed raw score to the later real challenger outcome."""

        self.__post_init__()
        if type(decision_index) is not int or decision_index <= 0:
            raise ValueError("decision_index must be positive")
        if (
            type(evidence_cutoff_ordinal) is not int
            or evidence_cutoff_ordinal <= 0
        ):
            raise ValueError("evidence_cutoff_ordinal must be positive")
        if type(decision) is not AdaptiveActionRacingDecision:
            raise TypeError("decision must be exact")
        if type(action) is not AdaptiveActionDescriptor:
            raise TypeError("action must be exact")
        if type(observation) is not SamePrefixPairedAuditObservation:
            raise TypeError("observation must be exact")
        decision.__post_init__()
        action.__post_init__()
        observation.__post_init__()
        plan = observation.plan
        if plan.designer_id not in (
            FORECAST_OPPORTUNITY_SAME_PREFIX_AUDIT_DESIGNER_IDS
        ):
            raise ValueError("observation is not a forecast audit")
        is_legacy_final_shadow = (
            plan.designer_id
            == FORECAST_OPPORTUNITY_SAME_PREFIX_SHADOW_DESIGNER_ID
        )
        if (
            plan.racing_decision_sha256 != decision.decision_sha256
            or plan.exploration_action_sha256 != action.action_sha256
            or decision.selected_action_sha256s
            != (plan.authoritative_action_sha256,)
            or decision.prior_selected_action_sha256s
            != plan.common_prefix_action_sha256s
            or (
                is_legacy_final_shadow
                and plan.authoritative_arm
                is not SamePrefixPairedAuditArm.EXPLORATION
            )
            or (
                plan.designer_id
                == FORECAST_STRATIFIED_SAME_PREFIX_AUDIT_DESIGNER_ID
                and plan.authoritative_arm
                not in {
                    SamePrefixPairedAuditArm.LEGACY,
                    SamePrefixPairedAuditArm.EXPLORATION,
                }
            )
        ):
            raise ValueError(
                "decision, descriptor, and forecast shadow do not join"
            )
        score = self._score_record(
            decision,
            action.action_sha256,
            require_recommended=is_legacy_final_shadow,
        )
        raw_acquisition_key = (
            "raw_acquisition_value_hex"
            if "raw_acquisition_value_hex" in score
            else "acquisition_value_hex"
        )
        paired_role = (
            ArchiveOpportunityCalibrationEvidenceRole
            .SAME_PREFIX_PAIRED_AUTHORITATIVE
            if plan.authoritative_arm
            is SamePrefixPairedAuditArm.EXPLORATION
            else (
                ArchiveOpportunityCalibrationEvidenceRole
                .SAME_PREFIX_PAIRED_COUNTERFACTUAL
            )
        )
        return ArchiveOpportunityCalibrationObservation(
            request=ArchiveOpportunityCalibrationRequest(
                context=ArchiveOpportunityActionContext(
                    action_sha256=action.action_sha256,
                    decision_index=decision_index,
                    lane_id=action.lane_id,
                    operator_id=action.operator_id,
                    native_rank=action.native_rank,
                    lane_size=action.lane_size,
                    prior_score=action.prior_score,
                    parent_generated_in_current_run=(
                        action.parent_generated_in_current_run
                    ),
                ),
                forecast_reliability=_read_nonnegative_hex(
                    score,
                    "reliability_hex",
                ),
                raw_adverse_gain=_read_nonnegative_hex(
                    score,
                    "adverse_gain_hex",
                ),
                raw_central_gain=_read_nonnegative_hex(
                    score,
                    "central_gain_hex",
                ),
                raw_favorable_gain=_read_nonnegative_hex(
                    score,
                    "favorable_gain_hex",
                ),
                raw_acquisition_value=_read_nonnegative_hex(
                    score,
                    raw_acquisition_key,
                ),
                prefix_gain=float(
                    observation.exploration_set_outcome
                    .prior_selected_set_gain
                ),
                prefix_action_count=len(
                    plan.common_prefix_action_sha256s
                ),
            ),
            realized_conditional_gain=float(
                observation.exploration_set_outcome.conditional_set_gain
            ),
            decision_sha256=decision.decision_sha256,
            outcome_sha256=observation.exploration_outcome.outcome_sha256,
            evidence_cutoff_ordinal=evidence_cutoff_ordinal,
            evidence_role=(
                ArchiveOpportunityCalibrationEvidenceRole
                .AUTHORITATIVE_SELECTED
                if is_legacy_final_shadow
                else paired_role
            ),
            sampling_propensity=(
                1.0
                if is_legacy_final_shadow
                else plan.exploration_selection_propensity
            ),
            paired_observation_sha256=(
                None
                if is_legacy_final_shadow
                else observation.observation_sha256
            ),
        )


__all__ = [
    "FORECAST_OPPORTUNITY_SHADOW_CALIBRATION_PROJECTOR_ID",
    "FORECAST_OPPORTUNITY_SHADOW_CALIBRATION_PROJECTOR_VERSION",
    "ForecastOpportunityShadowCalibrationProjector",
]
