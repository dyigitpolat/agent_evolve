"""Calibrated wrapper for current-prefix forecast opportunity.

The raw CPO policy continues to project sealed objective scenarios through the
workload's archive-utility port.  This wrapper adds a strictly prior
calibration authority over the resulting scalar opportunity.  It preserves
the raw policy as an ablation and returns the same ranking contract, so the
protected challenger and workload adapters do not depend on a calibration
implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.current_prefix_forecast_opportunity import (
    CurrentPrefixForecastOpportunityPolicy,
    CurrentPrefixForecastOpportunityRanking,
    CurrentPrefixForecastOpportunityScore,
)
from agent_evolve.application.forecast_geometry_portfolio import (
    MaterializedForecastGeometryBatch,
)
from agent_evolve.application.prequential_archive_opportunity_calibration import (
    ArchiveOpportunityActionContext,
    ArchiveOpportunityCalibrationPort,
    ArchiveOpportunityCalibrationRequest,
    validate_archive_opportunity_calibration_port,
)


CALIBRATED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_ID = (
    "calibrated_current_prefix_forecast_opportunity"
)
CALIBRATED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_POLICY_DOMAIN = (
    b"agent-evolve:calibrated-current-prefix-opportunity-policy:v1\x00"
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
class CalibratedCurrentPrefixForecastOpportunityPolicy:
    """Compose raw archive opportunity with prior-only calibration."""

    base_policy: CurrentPrefixForecastOpportunityPolicy = field(
        repr=False,
        compare=False,
    )
    calibration: ArchiveOpportunityCalibrationPort = field(
        repr=False,
        compare=False,
    )
    policy_id: str = (
        CALIBRATED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_ID
    )
    policy_version: int = (
        CALIBRATED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.base_policy) is not CurrentPrefixForecastOpportunityPolicy:
            raise TypeError("base_policy must be the exact raw CPO policy")
        self.base_policy.__post_init__()
        calibration_identity = (
            validate_archive_opportunity_calibration_port(
                self.calibration
            )
        )
        _require_token(self.policy_id, name="policy_id")
        if (
            self.policy_id
            != CALIBRATED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_ID
            or self.policy_version
            != CALIBRATED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_VERSION
        ):
            raise ValueError("calibrated policy identity is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _POLICY_DOMAIN,
                {
                    "schema_version": 1,
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "base_policy": {
                        "policy_id": self.base_policy.policy_id,
                        "policy_version": (
                            self.base_policy.policy_version
                        ),
                        "definition_sha256": (
                            self.base_policy.definition_sha256
                        ),
                    },
                    "calibration": {
                        "calibration_id": calibration_identity[0],
                        "calibration_version": calibration_identity[1],
                        "definition_sha256": calibration_identity[2],
                    },
                    "prefix_scale": (
                        "real_current_prefix_marginal_archive_utility"
                    ),
                    "selection": (
                        "descending_calibrated_expected_opportunity_"
                        "subject_to_lower_bound_abstention"
                    ),
                    "raw_policy_preserved_as_ablation": True,
                    "eligible_candidate_outcomes_observed": False,
                    "workload_objective_model_provider_prompt_branches": False,
                },
            ),
        )

    def _prefix_gain(
        self,
        *,
        prior_candidates: tuple[EvolutionCandidate, ...],
        current_prefix_candidates: tuple[EvolutionCandidate, ...],
    ) -> float:
        result = self.base_policy.archive_utility.portfolio_marginal_utility(
            prior_candidates,
            tuple(
                dict(value.objectives)
                for value in sorted(
                    current_prefix_candidates,
                    key=lambda item: item.candidate_id.value,
                )
            ),
        )
        if (
            type(result) is not float
            or not math.isfinite(result)
            or result < 0.0
        ):
            raise ValueError(
                "archive utility returned an invalid real-prefix gain"
            )
        return result

    def rank(
        self,
        *,
        prior_candidates: tuple[EvolutionCandidate, ...],
        current_prefix_candidates: tuple[EvolutionCandidate, ...],
        geometry: MaterializedForecastGeometryBatch,
        consumed_action_sha256s: tuple[str, ...] = (),
        eligible_action_sha256s: tuple[str, ...] | None = None,
        recommendation_count: int = 1,
        action_contexts: tuple[
            ArchiveOpportunityActionContext,
            ...,
        ] = (),
    ) -> CurrentPrefixForecastOpportunityRanking:
        """Calibrate every raw score against one sealed prior snapshot."""

        self.__post_init__()
        raw = self.base_policy.rank(
            prior_candidates=prior_candidates,
            current_prefix_candidates=current_prefix_candidates,
            geometry=geometry,
            consumed_action_sha256s=consumed_action_sha256s,
            eligible_action_sha256s=eligible_action_sha256s,
            recommendation_count=0,
        )
        if (
            type(action_contexts) is not tuple
            or any(
                type(value) is not ArchiveOpportunityActionContext
                for value in action_contexts
            )
        ):
            raise TypeError(
                "action_contexts must contain exact generic contexts"
            )
        for value in action_contexts:
            value.__post_init__()
        context_by_action = {
            value.action_sha256: value for value in action_contexts
        }
        if (
            len(context_by_action) != len(action_contexts)
            or tuple(sorted(context_by_action))
            != raw.eligible_action_sha256s
        ):
            raise ValueError(
                "action_contexts must cover the exact eligible market"
            )
        if not current_prefix_candidates:
            raise ValueError(
                "calibration requires a non-empty observed real prefix"
            )
        prefix_gain = self._prefix_gain(
            prior_candidates=prior_candidates,
            current_prefix_candidates=current_prefix_candidates,
        )
        calibrated_scores: list[
            CurrentPrefixForecastOpportunityScore
        ] = []
        calibration_by_action = {}
        for score in raw.scores:
            calibration = self.calibration.calibrate(
                ArchiveOpportunityCalibrationRequest(
                    context=context_by_action[score.action_sha256],
                    forecast_reliability=score.reliability,
                    raw_adverse_gain=score.adverse_gain,
                    raw_central_gain=score.central_gain,
                    raw_favorable_gain=score.favorable_gain,
                    raw_acquisition_value=score.acquisition_value,
                    prefix_gain=float(prefix_gain),
                    prefix_action_count=len(
                        current_prefix_candidates
                    ),
                )
            )
            calibration.__post_init__()
            if score.abstained:
                abstention_reason = score.abstention_reason
            elif calibration.abstained:
                abstention_reason = (
                    f"calibration_{calibration.abstention_reason}"
                )
            else:
                abstention_reason = None
            calibrated_scores.append(
                CurrentPrefixForecastOpportunityScore(
                    action_sha256=score.action_sha256,
                    phenotype_identity_sha256=(
                        score.phenotype_identity_sha256
                    ),
                    reliability=score.reliability,
                    adverse_gain=score.adverse_gain,
                    central_gain=score.central_gain,
                    favorable_gain=score.favorable_gain,
                    robust_gain=score.robust_gain,
                    exploration_gain=score.exploration_gain,
                    acquisition_value=float(
                        calibration.calibrated_acquisition_value
                    ),
                    abstained=abstention_reason is not None,
                    abstention_reason=abstention_reason,
                    source_member_sha256=score.source_member_sha256,
                    raw_acquisition_value=score.acquisition_value,
                    calibration_result=calibration,
                )
            )
            calibration_by_action[score.action_sha256] = calibration
        calibrated_scores.sort(
            key=lambda value: value.action_sha256
        )
        ranked = sorted(
            (
                value
                for value in calibrated_scores
                if not value.abstained
            ),
            key=lambda value: (
                -value.acquisition_value,
                -calibration_by_action[
                    value.action_sha256
                ].lower_expected_gain,
                -value.reliability,
                value.action_sha256,
            ),
        )
        recommended: list[str] = []
        phenotypes: set[str] = set()
        if type(recommendation_count) is not int or recommendation_count < 0:
            raise ValueError("recommendation_count must be non-negative")
        for value in ranked:
            if value.phenotype_identity_sha256 in phenotypes:
                continue
            recommended.append(value.action_sha256)
            phenotypes.add(value.phenotype_identity_sha256)
            if len(recommended) == recommendation_count:
                break
        return CurrentPrefixForecastOpportunityRanking(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            geometry_batch_sha256=raw.geometry_batch_sha256,
            archive_sha256=raw.archive_sha256,
            consumed_action_sha256s=raw.consumed_action_sha256s,
            eligible_action_sha256s=raw.eligible_action_sha256s,
            scores=tuple(calibrated_scores),
            recommended_action_sha256s=tuple(sorted(recommended)),
            current_prefix_outcomes_observed=(
                raw.current_prefix_outcomes_observed
            ),
            eligible_candidate_outcomes_observed=False,
        )


__all__ = [
    "CALIBRATED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_ID",
    "CALIBRATED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_POLICY_VERSION",
    "CalibratedCurrentPrefixForecastOpportunityPolicy",
]
