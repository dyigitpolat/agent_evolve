"""Protected current-prefix forecast opportunity for causal continuations.

The challenger composes two existing policies without weakening either
boundary:

* an incumbent outcome-adaptive policy authors a complete fallback decision;
* a sealed forecast-geometry policy may replace that one action when it
  identifies positive conditional archive opportunity against the real
  evaluated prefix.

No eligible candidate outcome is visible to the challenger.  If forecast
geometry is absent, exhausted, or has no positive opportunity, the incumbent
decision is returned under an authenticated composite receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.current_prefix_forecast_opportunity import (
    CurrentPrefixForecastOpportunityPolicyPort,
    validate_current_prefix_forecast_opportunity_policy,
)
from agent_evolve.application.forecast_geometry_portfolio import (
    MaterializedForecastGeometryBatch,
    MaterializedForecastGeometryProjectionPort,
)
from agent_evolve.application.outcome_adaptive_action_racing import (
    AdaptiveActionDescriptor,
    AdaptiveActionRacingDecision,
    AdaptiveActionWave,
)
from agent_evolve.application.prequential_archive_opportunity_calibration import (
    ArchiveOpportunityActionContext,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionEvaluation,
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import freeze_json


PROTECTED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_ID = (
    "protected_current_prefix_forecast_opportunity"
)
PROTECTED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_VERSION = 2
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_DEFINITION_DOMAIN = (
    b"agent-evolve:protected-current-prefix-forecast-opportunity:v2\x00"
)
_PRIOR_DOMAIN = (
    b"agent-evolve:protected-current-prefix-forecast-prior:v1\x00"
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


def _prior_sha256(
    candidates: tuple[EvolutionCandidate, ...],
) -> str:
    return _hash(
        _PRIOR_DOMAIN,
        [
            {
                "candidate_id": value.candidate_id.value,
                "configuration_sha256": value.occurrence.configuration_hash,
                "generation": value.generation,
                "valid": value.valid,
                "operator_compliant": value.operator_compliant,
                "evidence_compliant": value.evidence_compliant,
                "objectives": [
                    {
                        "metric_id": metric_id,
                        "value_hex": number.hex(),
                    }
                    for metric_id, number in value.objectives
                ],
            }
            for value in candidates
        ],
    )


@dataclass(frozen=True, slots=True)
class ProtectedCurrentPrefixForecastOpportunityChallenger:
    """Challenge one incumbent continuation while preserving abstention."""

    prior_candidates: tuple[EvolutionCandidate, ...]
    opportunity_policy: CurrentPrefixForecastOpportunityPolicyPort = field(
        repr=False,
        compare=False,
    )
    geometry_projection: MaterializedForecastGeometryProjectionPort = field(
        repr=False,
        compare=False,
    )
    fallback_policy_id: str
    fallback_policy_version: int
    fallback_policy_definition_sha256: str
    challenger_id: str = PROTECTED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_ID
    challenger_version: int = (
        PROTECTED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.prior_candidates) is not tuple
            or not self.prior_candidates
            or any(
                type(value) is not EvolutionCandidate
                for value in self.prior_candidates
            )
        ):
            raise TypeError(
                "prior_candidates must contain exact non-empty candidates"
            )
        for value in self.prior_candidates:
            value.__post_init__()
        candidate_ids = tuple(
            value.candidate_id for value in self.prior_candidates
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("prior candidate IDs must be unique")
        opportunity_identity = (
            validate_current_prefix_forecast_opportunity_policy(
                self.opportunity_policy
            )
        )
        if not isinstance(
            self.geometry_projection,
            MaterializedForecastGeometryProjectionPort,
        ):
            raise TypeError(
                "geometry_projection must implement its application port"
            )
        _require_token(
            self.geometry_projection.projection_id,
            name="geometry projection_id",
        )
        if (
            type(self.geometry_projection.projection_version) is not int
            or self.geometry_projection.projection_version <= 0
        ):
            raise ValueError("geometry projection_version must be positive")
        require_sha256(
            self.geometry_projection.definition_sha256,
            "geometry projection definition_sha256",
        )
        _require_token(
            self.fallback_policy_id,
            name="fallback_policy_id",
        )
        if (
            type(self.fallback_policy_version) is not int
            or self.fallback_policy_version <= 0
        ):
            raise ValueError("fallback_policy_version must be positive")
        require_sha256(
            self.fallback_policy_definition_sha256,
            "fallback_policy_definition_sha256",
        )
        if (
            self.challenger_id
            != PROTECTED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_ID
            or self.challenger_version
            != PROTECTED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_VERSION
        ):
            raise ValueError("challenger identity is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "challenger_id": self.challenger_id,
                    "challenger_version": self.challenger_version,
                    "prior_sha256": _prior_sha256(
                        self.prior_candidates
                    ),
                    "opportunity_policy": {
                        "policy_id": opportunity_identity[0],
                        "policy_version": opportunity_identity[1],
                        "definition_sha256": opportunity_identity[2],
                    },
                    "geometry_projection": {
                        "projection_id": (
                            self.geometry_projection.projection_id
                        ),
                        "projection_version": (
                            self.geometry_projection.projection_version
                        ),
                        "definition_sha256": (
                            self.geometry_projection.definition_sha256
                        ),
                    },
                    "fallback_policy": {
                        "policy_id": self.fallback_policy_id,
                        "policy_version": self.fallback_policy_version,
                        "definition_sha256": (
                            self.fallback_policy_definition_sha256
                        ),
                    },
                    "selection": (
                        "positive_current_prefix_opportunity_else_fallback"
                    ),
                    "forecast_geometry_market_relation": (
                        "sealed_authored_geometry_intersected_with_"
                        "phenotype_projected_adaptive_market"
                    ),
                    "out_of_adaptive_market_forecasts": "ignored_and_logged",
                    "eligible_candidate_outcomes_observed": False,
                    "workload_objective_model_provider_prompt_config_"
                    "branches": False,
                },
            ),
        )

    async def freeze_geometry(
        self,
        *,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedForecastGeometryBatch:
        """Freeze one complete forecast market before current outcomes."""

        self.__post_init__()
        geometry = await self.geometry_projection.project(
            request,
            proposals,
        )
        if type(geometry) is not MaterializedForecastGeometryBatch:
            raise TypeError("geometry projection returned a foreign batch")
        geometry.__post_init__()
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        if (
            geometry.residual_request_sha256 != request.request_sha256
            or geometry.proposal_sha256s != proposal_sha256s
        ):
            raise ValueError("geometry changed the sealed proposal cutoff")
        return geometry

    def challenge(
        self,
        *,
        fallback: AdaptiveActionRacingDecision,
        geometry: MaterializedForecastGeometryBatch,
        adaptive_actions: tuple[AdaptiveActionDescriptor, ...],
        selected_evaluations: tuple[MaterializedActionEvaluation, ...],
        excluded_action_sha256s: tuple[str, ...] = (),
    ) -> AdaptiveActionRacingDecision:
        """Select a positive forecast opportunity or preserve the fallback."""

        self.__post_init__()
        if type(fallback) is not AdaptiveActionRacingDecision:
            raise TypeError("fallback must be an exact racing decision")
        fallback.__post_init__()
        if (
            fallback.policy_id != self.fallback_policy_id
            or fallback.policy_version != self.fallback_policy_version
            or fallback.policy_definition_sha256
            != self.fallback_policy_definition_sha256
        ):
            raise ValueError("fallback decision changed policy identity")
        if fallback.wave is AdaptiveActionWave.DIAGNOSTIC:
            raise ValueError("diagnostic decisions cannot be challenged")
        if len(fallback.selected_action_sha256s) != 1:
            raise ValueError("continuation fallback must select one action")
        if type(geometry) is not MaterializedForecastGeometryBatch:
            raise TypeError("geometry must be an exact batch")
        geometry.__post_init__()
        if (
            geometry.residual_request_sha256
            != fallback.residual_request_sha256
        ):
            raise ValueError("geometry and fallback cutoffs differ")
        if (
            type(adaptive_actions) is not tuple
            or not adaptive_actions
            or any(
                type(value) is not AdaptiveActionDescriptor
                for value in adaptive_actions
            )
        ):
            raise TypeError("adaptive_actions must contain exact descriptors")
        action_by_sha256: dict[str, AdaptiveActionDescriptor] = {}
        for value in adaptive_actions:
            value.__post_init__()
            if value.action_sha256 in action_by_sha256:
                raise ValueError("adaptive market repeats an action")
            action_by_sha256[value.action_sha256] = value
        if (
            type(excluded_action_sha256s) is not tuple
            or excluded_action_sha256s
            != tuple(sorted(set(excluded_action_sha256s)))
        ):
            raise ValueError(
                "excluded action hashes must be unique and canonical"
            )
        for value in excluded_action_sha256s:
            require_sha256(value, "excluded_action_sha256")
        if (
            not set(excluded_action_sha256s) <= set(action_by_sha256)
            or set(excluded_action_sha256s)
            & (
                set(fallback.prior_selected_action_sha256s)
                | set(fallback.selected_action_sha256s)
            )
        ):
            raise ValueError(
                "excluded actions must be open non-fallback market members"
            )
        if (
            type(selected_evaluations) is not tuple
            or not selected_evaluations
            or any(
                type(value) is not MaterializedActionEvaluation
                for value in selected_evaluations
            )
        ):
            raise TypeError(
                "selected_evaluations must contain the real non-empty prefix"
            )
        evaluation_by_action: dict[str, MaterializedActionEvaluation] = {}
        for value in selected_evaluations:
            value.__post_init__()
            action_sha256 = value.action.action_sha256
            if action_sha256 in evaluation_by_action:
                raise ValueError("selected evaluations repeat an action")
            evaluation_by_action[action_sha256] = value
        selected_ids = tuple(sorted(evaluation_by_action))
        if selected_ids != fallback.prior_selected_action_sha256s:
            raise ValueError("fallback and real selected prefixes differ")
        expected_outcomes = tuple(
            sorted(
                value.evaluation_sha256
                for value in selected_evaluations
            )
        )
        # A fallback binds projected outcome hashes, while this challenger sees
        # evaluation receipts.  The caller validates the projected cutoff; here
        # we only require the same action prefix and non-empty observed hashes.
        if not fallback.observed_outcome_sha256s or not expected_outcomes:
            raise ValueError("continuation challenge requires observed outcomes")

        geometry_by_action = {
            value.action_sha256: value for value in geometry.members
        }
        intersecting_action_ids = tuple(
            sorted(set(geometry_by_action) & set(action_by_sha256))
        )
        for action_sha256 in intersecting_action_ids:
            member = geometry_by_action[action_sha256]
            if (
                member.phenotype_identity_sha256
                != action_by_sha256[action_sha256].phenotype_sha256
            ):
                raise ValueError(
                    "geometry and adaptive phenotype identities differ"
                )
        selected_phenotypes = {
            action_by_sha256[value].phenotype_sha256
            for value in selected_ids
        }
        eligible = tuple(
            sorted(
                action_sha256
                for action_sha256 in intersecting_action_ids
                if (
                    action_sha256 not in evaluation_by_action
                    and action_sha256
                    not in set(excluded_action_sha256s)
                    and geometry_by_action[
                        action_sha256
                    ].phenotype_identity_sha256
                    not in selected_phenotypes
                )
            )
        )
        consumed_geometry = tuple(
            sorted(set(selected_ids) & set(geometry_by_action))
        )
        current_generations = {
            value.candidate.generation
            for value in selected_evaluations
        }
        if (
            len(current_generations) != 1
            or next(iter(current_generations)) <= 0
        ):
            raise ValueError(
                "selected prefix must identify one positive decision index"
            )
        decision_index = next(iter(current_generations))
        ranking = self.opportunity_policy.rank(
            prior_candidates=self.prior_candidates,
            current_prefix_candidates=tuple(
                sorted(
                    (
                        value.candidate
                        for value in selected_evaluations
                    ),
                    key=lambda value: value.candidate_id.value,
                )
            ),
            geometry=geometry,
            consumed_action_sha256s=consumed_geometry,
            eligible_action_sha256s=eligible,
            recommendation_count=1,
            action_contexts=tuple(
                ArchiveOpportunityActionContext(
                    action_sha256=action_sha256,
                    decision_index=decision_index,
                    lane_id=action_by_sha256[action_sha256].lane_id,
                    operator_id=(
                        action_by_sha256[action_sha256].operator_id
                    ),
                    native_rank=(
                        action_by_sha256[action_sha256].native_rank
                    ),
                    lane_size=(
                        action_by_sha256[action_sha256].lane_size
                    ),
                    prior_score=(
                        action_by_sha256[action_sha256].prior_score
                    ),
                    parent_generated_in_current_run=(
                        action_by_sha256[
                            action_sha256
                        ].parent_generated_in_current_run
                    ),
                )
                for action_sha256 in eligible
            ),
        )
        if ranking.recommended_action_sha256s:
            selected_action_sha256s = (
                ranking.recommended_action_sha256s
            )
            selection_source = (
                "current_prefix_forecast_opportunity"
            )
            propensity = 1.0
        else:
            selected_action_sha256s = (
                fallback.selected_action_sha256s
            )
            selection_source = "protected_fallback"
            propensity = fallback.selection_propensity
        if not set(selected_action_sha256s) <= (
            set(action_by_sha256) - set(selected_ids)
        ):
            raise ValueError("challenger selected outside the open market")
        if (
            type(propensity) is not float
            or not math.isfinite(propensity)
            or not 0.0 < propensity <= 1.0
        ):
            raise ValueError("composite selection propensity is invalid")
        return AdaptiveActionRacingDecision(
            policy_id=self.challenger_id,
            policy_version=self.challenger_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=(
                fallback.residual_request_sha256
            ),
            wave=fallback.wave,
            selected_action_sha256s=selected_action_sha256s,
            prior_selected_action_sha256s=(
                fallback.prior_selected_action_sha256s
            ),
            observed_outcome_sha256s=(
                fallback.observed_outcome_sha256s
            ),
            observed_set_outcome_sha256s=(
                fallback.observed_set_outcome_sha256s
            ),
            selection_propensity=float(propensity),
            evidence=freeze_json(
                {
                    "selection_source": selection_source,
                    "fallback_decision": fallback.to_record(
                        include_evidence=True
                    ),
                    "opportunity_ranking": ranking.to_record(
                        include_scores=True
                    ),
                    "geometry_batch_sha256": geometry.batch_sha256,
                    "geometry_member_count": len(geometry_by_action),
                    "adaptive_market_action_count": len(
                        action_by_sha256
                    ),
                    "intersecting_forecast_action_sha256s": list(
                        intersecting_action_ids
                    ),
                    "ignored_out_of_market_forecast_action_sha256s": list(
                        sorted(
                            set(geometry_by_action)
                            - set(action_by_sha256)
                        )
                    ),
                    "selected_real_prefix_action_sha256s": list(
                        selected_ids
                    ),
                    "selected_evaluation_sha256s": list(
                        expected_outcomes
                    ),
                    "excluded_action_sha256s": list(
                        excluded_action_sha256s
                    ),
                    "excluded_action_outcomes_used_for_selection": False,
                    "eligible_candidate_outcomes_observed": False,
                    "fallback_preserved_on_abstention": True,
                    "workload_objective_model_provider_prompt_config_"
                    "branches": False,
                }
            ),
        )


__all__ = [
    "PROTECTED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_ID",
    "PROTECTED_CURRENT_PREFIX_FORECAST_OPPORTUNITY_VERSION",
    "ProtectedCurrentPrefixForecastOpportunityChallenger",
]
