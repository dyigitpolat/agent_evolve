"""Portable pre-evaluation consequence features for residual actions.

The feature ABI is expressed only in generic campaign quantities: prior
broker evidence, native rank, mutation radius/arity, calibrated LLM forecasts,
parent archive opportunity, and remaining-horizon phase.  Workload objective
geometry is delegated to an authenticated scalar archive-utility port.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
import statistics

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.candidate_archive_consequence import (
    CandidateArchiveConsequenceUtilityPort,
    candidate_archive_utility,
    validate_candidate_archive_consequence_utility,
)
from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.frozen_hurdle_score import (
    MaterializedActionFeatureBatch,
    MaterializedActionFeatureVector,
)
from agent_evolve.application.materialized_action_broker import (
    BrokerEvidenceChannel,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.typed_json import freeze_json

from .materialized_hierarchical_residual_expert import (
    MaterializedHierarchicalResidualActionEvidence,
    MaterializedHierarchicalResidualActionEvidencePort,
)


PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_NAMES = (
    "broker_mean",
    "broker_standard_deviation",
    "broker_upper_bound",
    "opportunity_scaled_index",
    "gain_global_mean",
    "gain_global_count_log1p",
    "gain_local_mean",
    "gain_local_count_log1p",
    "positive_global_mean",
    "positive_global_count_log1p",
    "native_rank_reciprocal",
    "native_rank_percentile",
    "reference_action",
    "radius_normalized",
    "component_count_normalized",
    "probability_valid",
    "forecast_available",
    "forecast_minimum_confidence",
    "forecast_mean_confidence",
    "predicted_p10_complement",
    "predicted_p50_complement",
    "predicted_p90_complement",
    "predicted_complement_spread",
    "parent_available",
    "parent_leave_one_out_contribution",
    "parent_leave_one_out_fraction_of_opportunity",
    "parent_generated_in_live_run",
    "phase_composition",
    "phase_terminal_conversion",
)
PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_PROJECTION_ID = (
    "portable_residual_consequence_features"
)
PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_PROJECTION_VERSION = 1
PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_GROUPS = (
    (
        "empirical_history",
        (
            "broker_mean",
            "broker_standard_deviation",
            "broker_upper_bound",
            "opportunity_scaled_index",
            "gain_global_mean",
            "gain_global_count_log1p",
            "gain_local_mean",
            "gain_local_count_log1p",
            "positive_global_mean",
            "positive_global_count_log1p",
        ),
    ),
    (
        "forecast",
        (
            "forecast_available",
            "forecast_minimum_confidence",
            "forecast_mean_confidence",
            "predicted_p10_complement",
            "predicted_p50_complement",
            "predicted_p90_complement",
            "predicted_complement_spread",
        ),
    ),
    (
        "parent",
        (
            "parent_available",
            "parent_leave_one_out_contribution",
            "parent_leave_one_out_fraction_of_opportunity",
            "parent_generated_in_live_run",
        ),
    ),
    (
        "phase",
        (
            "phase_composition",
            "phase_terminal_conversion",
        ),
    ),
    (
        "proposal_structure",
        (
            "native_rank_reciprocal",
            "native_rank_percentile",
            "reference_action",
            "radius_normalized",
            "component_count_normalized",
            "probability_valid",
        ),
    ),
)
_DEFINITION_DOMAIN = (
    b"agent-evolve:portable-residual-consequence-features:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _feature_abi_sha256() -> str:
    return hashlib.sha256(
        b"agent-evolve:portable-residual-consequence-feature-abi:v1\x00"
        + _canonical_json(PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_NAMES)
    ).hexdigest()


@dataclass(slots=True)
class PortableResidualConsequenceFeatureProjection:
    """Project a sealed action population using one authenticated prior cutoff."""

    prior_candidates: tuple[EvolutionCandidate, ...]
    initial_candidate_ids: tuple[CandidateId, ...]
    evidence_sources: tuple[
        MaterializedHierarchicalResidualActionEvidencePort,
        ...,
    ] = field(repr=False, compare=False)
    broker: RegretBrokeredMaterializedActionPolicy = field(
        repr=False,
        compare=False,
    )
    archive_utility: CandidateArchiveConsequenceUtilityPort = field(
        repr=False,
        compare=False,
    )
    projection_id: str = (
        PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_PROJECTION_ID
    )
    projection_version: int = (
        PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_PROJECTION_VERSION
    )
    feature_names: tuple[str, ...] = (
        PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_NAMES
    )
    definition_sha256: str = field(init=False)
    _cache: dict[str, MaterializedActionFeatureBatch] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )
    _lock: asyncio.Lock = field(
        init=False,
        default_factory=asyncio.Lock,
        repr=False,
    )

    def __post_init__(self) -> None:
        if (
            type(self.prior_candidates) is not tuple
            or not self.prior_candidates
            or any(
                type(candidate) is not EvolutionCandidate
                for candidate in self.prior_candidates
            )
        ):
            raise TypeError(
                "prior_candidates must contain exact non-empty candidates"
            )
        for candidate in self.prior_candidates:
            candidate.__post_init__()
        candidate_ids = tuple(
            candidate.candidate_id for candidate in self.prior_candidates
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("prior candidate IDs must be unique")
        if (
            type(self.initial_candidate_ids) is not tuple
            or self.initial_candidate_ids
            != tuple(
                sorted(
                    set(self.initial_candidate_ids),
                    key=lambda value: value.value,
                )
            )
        ):
            raise ValueError(
                "initial_candidate_ids must be unique and canonical"
            )
        if any(
            type(value) is not CandidateId
            for value in self.initial_candidate_ids
        ):
            raise TypeError("initial candidate IDs must be exact")
        if not set(self.initial_candidate_ids).issubset(set(candidate_ids)):
            raise ValueError("initial candidates are absent from the prior")
        if (
            type(self.evidence_sources) is not tuple
            or not self.evidence_sources
            or any(
                not isinstance(
                    value,
                    MaterializedHierarchicalResidualActionEvidencePort,
                )
                for value in self.evidence_sources
            )
        ):
            raise TypeError(
                "evidence_sources must implement their runtime port"
            )
        expert_ids = tuple(
            value.expert_id for value in self.evidence_sources
        )
        if expert_ids != tuple(sorted(set(expert_ids))):
            raise ValueError("evidence sources must be unique and canonical")
        if type(self.broker) is not RegretBrokeredMaterializedActionPolicy:
            raise TypeError("broker must be exact")
        self.broker.__post_init__()
        utility_identity = (
            validate_candidate_archive_consequence_utility(
                self.archive_utility
            )
        )
        if (
            self.projection_id
            != PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_PROJECTION_ID
            or self.projection_version
            != PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_PROJECTION_VERSION
            or self.feature_names
            != PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_NAMES
        ):
            raise ValueError("portable feature projection identity is immutable")
        self.definition_sha256 = hashlib.sha256(
            _DEFINITION_DOMAIN
            + _canonical_json(
                {
                    "schema_version": 1,
                    "feature_abi_sha256": _feature_abi_sha256(),
                    "archive_utility": {
                        "utility_id": utility_identity[0],
                        "utility_version": utility_identity[1],
                        "definition_sha256": utility_identity[2],
                    },
                    "prior_only": True,
                    "current_candidate_outcomes": False,
                    "objective_names_in_core": False,
                    "workload_model_provider_branches": False,
                }
            )
        ).hexdigest()

    def _evidence(
        self,
        action_sha256: str,
    ) -> MaterializedHierarchicalResidualActionEvidence:
        matches = tuple(
            value
            for source in self.evidence_sources
            if (value := source.evidence_for(action_sha256)) is not None
        )
        if len(matches) != 1:
            raise ValueError(
                "one sealed action must have exactly one semantic evidence row"
            )
        return matches[0]

    def _utility(self, candidates: tuple[EvolutionCandidate, ...]) -> float:
        return candidate_archive_utility(self.archive_utility, candidates)

    def _marginal(
        self,
        objective_point: dict[str, float],
    ) -> float:
        result = self.archive_utility.marginal_utility(
            self.prior_candidates,
            objective_point,
        )
        if type(result) is not float or not math.isfinite(result) or result < 0.0:
            raise ValueError(
                "archive marginal utility must be finite and non-negative"
            )
        return result

    def _predicted_complements(
        self,
        evidence: MaterializedHierarchicalResidualActionEvidence,
        parent: EvolutionCandidate | None,
    ) -> tuple[float, float, float]:
        if parent is None:
            return (0.0, 0.0, 0.0)
        parent_objectives = parent.objective_map
        forecasts = {
            value.metric_id: value
            for value in evidence.effect_predictions
        }
        if set(forecasts) != set(parent_objectives):
            raise ValueError(
                "forecast metrics differ from the parent objective vector"
            )
        result: list[float] = []
        for field_name in ("p10_delta", "p50_delta", "p90_delta"):
            point = {
                metric_id: parent_objectives[metric_id]
                + float(getattr(forecasts[metric_id], field_name))
                for metric_id in parent_objectives
            }
            result.append(self._marginal(point))
        return tuple(result)  # type: ignore[return-value]

    async def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionFeatureBatch:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if any(
            candidate.generation >= request.decision_index
            for candidate in self.prior_candidates
        ):
            raise ValueError("prior candidates cross the current decision cutoff")
        if type(proposals) is not tuple or not proposals:
            raise ValueError("proposals must be a non-empty exact tuple")
        async with self._lock:
            cached = self._cache.get(request.request_sha256)
            if cached is not None:
                expected = tuple(
                    sorted(value.proposal_sha256 for value in proposals)
                )
                if cached.proposal_sha256s != expected:
                    raise ValueError(
                        "one request identity cannot name two proposal universes"
                    )
                return cached

            actions = tuple(
                action
                for proposal in proposals
                for action in proposal.actions
            )
            action_by_sha256 = {
                value.action_sha256: value for value in actions
            }
            if len(action_by_sha256) != len(actions):
                raise ValueError("proposal union repeats an action identity")
            evidence_by_action = {
                action_sha256: self._evidence(action_sha256)
                for action_sha256 in action_by_sha256
            }
            expert_counts = Counter(
                value.action.expert_id
                for value in evidence_by_action.values()
            )
            maximum_radius = max(
                value.plan.radius for value in evidence_by_action.values()
            )
            maximum_components = max(
                len(value.plan.component_option_ids)
                for value in evidence_by_action.values()
            )
            prior_by_id = {
                value.candidate_id: value
                for value in self.prior_candidates
            }
            full_utility = self._utility(self.prior_candidates)
            parent_contributions: dict[CandidateId, float] = {}
            for index, candidate in enumerate(self.prior_candidates):
                reduced = (
                    self.prior_candidates[:index]
                    + self.prior_candidates[index + 1 :]
                )
                parent_contributions[candidate.candidate_id] = max(
                    0.0,
                    full_utility - self._utility(reduced),
                )
            opportunity = max(parent_contributions.values(), default=0.0)
            initial_ids = set(self.initial_candidate_ids)
            rows: list[MaterializedActionFeatureVector] = []
            row_evidence: list[dict[str, object]] = []
            for action_sha256 in sorted(action_by_sha256):
                action = action_by_sha256[action_sha256]
                evidence = evidence_by_action[action_sha256]
                score = self.broker.score(action)
                estimates = {
                    value.channel: value for value in score.estimates
                }
                gain = estimates[BrokerEvidenceChannel.GAIN]
                positive = estimates[BrokerEvidenceChannel.POSITIVE]
                parent_id = evidence.plan.parent_candidate_id
                parent = prior_by_id.get(parent_id)
                parent_contribution = parent_contributions.get(
                    parent_id,
                    0.0,
                )
                predicted = self._predicted_complements(
                    evidence,
                    parent,
                )
                confidences = tuple(
                    value.confidence for value in evidence.effect_predictions
                )
                expert_count = expert_counts[action.expert_id]
                native_rank = evidence.provider_rank
                broker_mean = score.return_estimate.mean
                broker_sd = score.return_estimate.standard_deviation
                broker_upper = score.upper_confidence_bound
                opportunity_scaled_index = (
                    broker_mean
                    if request.phase is SearchPhase.TERMINAL_CONVERSION
                    else broker_mean
                    + min(
                        max(0.0, broker_upper - broker_mean),
                        max(opportunity, broker_mean),
                    )
                )
                values = (
                    broker_mean,
                    broker_sd,
                    broker_upper,
                    opportunity_scaled_index,
                    gain.global_mean,
                    math.log1p(gain.global_count),
                    gain.local_mean,
                    math.log1p(gain.local_count),
                    positive.global_mean,
                    math.log1p(positive.global_count),
                    1.0 / native_rank,
                    (
                        1.0
                        if expert_count <= 1
                        else 1.0
                        - ((native_rank - 1.0) / (expert_count - 1.0))
                    ),
                    1.0 if action.reference_action else 0.0,
                    evidence.plan.radius / max(1, maximum_radius),
                    len(evidence.plan.component_option_ids)
                    / max(1, maximum_components),
                    evidence.probability_valid,
                    1.0 if evidence.effect_predictions else 0.0,
                    min(confidences, default=0.0),
                    (
                        statistics.fmean(confidences)
                        if confidences
                        else 0.0
                    ),
                    predicted[0],
                    predicted[1],
                    predicted[2],
                    max(predicted) - min(predicted),
                    1.0 if parent is not None else 0.0,
                    parent_contribution,
                    (
                        parent_contribution / opportunity
                        if opportunity > 0.0
                        else 0.0
                    ),
                    (
                        1.0
                        if parent is not None and parent_id not in initial_ids
                        else 0.0
                    ),
                    (
                        1.0
                        if request.phase is SearchPhase.COMPOSITION
                        else 0.0
                    ),
                    (
                        1.0
                        if request.phase is SearchPhase.TERMINAL_CONVERSION
                        else 0.0
                    ),
                )
                if len(values) != len(self.feature_names):
                    raise AssertionError("portable feature ABI drifted")
                rows.append(
                    MaterializedActionFeatureVector(
                        action_sha256=action_sha256,
                        values=tuple(float(value) for value in values),
                    )
                )
                row_evidence.append(
                    {
                        "action_sha256": action_sha256,
                        "semantic_evidence": evidence.to_record(),
                        "broker_score": score.to_record(),
                        "features_hex": [
                            float(value).hex() for value in values
                        ],
                    }
                )
            frozen_evidence = freeze_json(
                {
                    "schema_version": 1,
                    "feature_abi_sha256": _feature_abi_sha256(),
                    "prior_candidate_ids": [
                        value.candidate_id.value
                        for value in self.prior_candidates
                    ],
                    "prior_configuration_sha256s": [
                        value.occurrence.configuration_hash
                        for value in self.prior_candidates
                    ],
                    "prior_utility_hex": full_utility.hex(),
                    "opportunity_hex": opportunity.hex(),
                    "rows": row_evidence,
                    "candidate_outcomes_observed": False,
                }
            )
            batch = MaterializedActionFeatureBatch(
                projection_id=self.projection_id,
                projection_version=self.projection_version,
                projection_definition_sha256=self.definition_sha256,
                residual_request_sha256=request.request_sha256,
                proposal_sha256s=tuple(
                    sorted(value.proposal_sha256 for value in proposals)
                ),
                feature_names=self.feature_names,
                vectors=tuple(rows),
                candidate_outcomes_observed=False,
                evidence=frozen_evidence,
            )
            self._cache[request.request_sha256] = batch
            return batch


__all__ = [
    "PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_GROUPS",
    "PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_NAMES",
    "PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_PROJECTION_ID",
    "PORTABLE_RESIDUAL_CONSEQUENCE_FEATURE_PROJECTION_VERSION",
    "PortableResidualConsequenceFeatureProjection",
]
