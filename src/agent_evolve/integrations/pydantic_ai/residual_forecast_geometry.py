"""Project Pydantic-AI residual forecasts into generic scenario geometry."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.forecast_geometry_portfolio import (
    ForecastGeometryScenario,
    MaterializedForecastGeometryBatch,
    MaterializedForecastGeometryMember,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.core.optimization_semantics import MetricSense
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256

from .materialized_hierarchical_residual_expert import (
    MaterializedHierarchicalResidualActionEvidence,
    MaterializedHierarchicalResidualActionEvidencePort,
)


PYDANTIC_AI_RESIDUAL_FORECAST_GEOMETRY_PROJECTION_ID = (
    "pydantic_ai_residual_forecast_geometry"
)
PYDANTIC_AI_RESIDUAL_FORECAST_GEOMETRY_PROJECTION_VERSION = 3
_DEFINITION_DOMAIN = (
    b"agent-evolve:pydantic-ai-residual-forecast-geometry:v3\x00"
)


class ResidualForecastEvidenceCoverageMode(str, Enum):
    """How forecast geometry treats heterogeneous proposal engines."""

    COMPLETE = "complete_action_coverage"
    AVAILABLE_ONLY = "available_forecast_evidence"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


@dataclass(slots=True)
class PydanticAIResidualForecastGeometryProjection:
    """Translate sealed metric forecasts without interpreting a workload.

    Objective senses are ordinary evaluator semantics, not privileged search
    hints.  They permit portable favorable/adverse scenarios for mixed-sense
    problems while lower/median/upper numeric scenarios preserve the raw
    forecast geometry for ablations.
    """

    prior_candidates: tuple[EvolutionCandidate, ...]
    evidence_sources: tuple[
        MaterializedHierarchicalResidualActionEvidencePort,
        ...,
    ] = field(repr=False, compare=False)
    objective_senses: tuple[tuple[str, MetricSense], ...]
    coverage_mode: ResidualForecastEvidenceCoverageMode = (
        ResidualForecastEvidenceCoverageMode.COMPLETE
    )
    projection_id: str = (
        PYDANTIC_AI_RESIDUAL_FORECAST_GEOMETRY_PROJECTION_ID
    )
    projection_version: int = (
        PYDANTIC_AI_RESIDUAL_FORECAST_GEOMETRY_PROJECTION_VERSION
    )
    definition_sha256: str = field(init=False)
    _cache: dict[str, MaterializedForecastGeometryBatch] = field(
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
            raise TypeError("evidence_sources must implement their port")
        expert_ids = tuple(
            value.expert_id for value in self.evidence_sources
        )
        if expert_ids != tuple(sorted(set(expert_ids))):
            raise ValueError("evidence sources must be unique and canonical")
        if (
            type(self.objective_senses) is not tuple
            or not self.objective_senses
            or self.objective_senses
            != tuple(sorted(self.objective_senses, key=lambda value: value[0]))
        ):
            raise ValueError("objective_senses must be non-empty and canonical")
        metric_ids: list[str] = []
        for metric_id, sense in self.objective_senses:
            if type(metric_id) is not str or not metric_id:
                raise ValueError("objective metric IDs must be non-empty")
            if type(sense) is not MetricSense:
                raise TypeError("objective senses must be exact MetricSense values")
            metric_ids.append(metric_id)
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("objective_senses repeats a metric")
        if type(self.coverage_mode) is not ResidualForecastEvidenceCoverageMode:
            raise TypeError(
                "coverage_mode must be an exact "
                "ResidualForecastEvidenceCoverageMode"
            )
        eligible = tuple(
            value
            for value in self.prior_candidates
            if (
                value.valid
                and value.operator_compliant
                and value.evidence_compliant
            )
        )
        if not eligible:
            raise ValueError(
                "forecast geometry requires an eligible prior candidate"
            )
        if any(
            set(value.objective_map) != set(metric_ids)
            for value in eligible
        ):
            raise ValueError(
                "eligible prior candidate objectives differ from senses"
            )
        if (
            self.projection_id
            != PYDANTIC_AI_RESIDUAL_FORECAST_GEOMETRY_PROJECTION_ID
            or self.projection_version
            != PYDANTIC_AI_RESIDUAL_FORECAST_GEOMETRY_PROJECTION_VERSION
        ):
            raise ValueError("projection identity is immutable")
        self.definition_sha256 = hashlib.sha256(
            _DEFINITION_DOMAIN
            + _canonical_json(
                {
                    "schema_version": 1,
                    "projection_id": self.projection_id,
                    "projection_version": self.projection_version,
                    "objective_senses": [
                        {
                            "metric_id": metric_id,
                            "sense": sense.value,
                        }
                        for metric_id, sense in self.objective_senses
                    ],
                    "prior_candidate_eligibility": (
                        "valid_and_operator_and_evidence_compliant"
                    ),
                    "ineligible_ledger_rows_may_omit_objectives": True,
                    "scenarios": {
                        "lower_numeric": "parent_plus_p10",
                        "median": "parent_plus_p50",
                        "upper_numeric": "parent_plus_p90",
                        "favorable": "sense_conditioned_favorable_quantile",
                        "adverse": "sense_conditioned_adverse_quantile",
                    },
                    "reliability": (
                        "probability_valid_times_minimum_metric_confidence"
                    ),
                    "coverage_mode": self.coverage_mode.value,
                    "missing_forecast_evidence": (
                        "fail_closed"
                        if self.coverage_mode
                        is ResidualForecastEvidenceCoverageMode.COMPLETE
                        else "exclude_action_and_log_identity"
                    ),
                    "current_candidate_outcomes_observed": False,
                    "workload_model_provider_branches": False,
                }
            )
        ).hexdigest()

    def _evidence(
        self,
        action_sha256: str,
    ) -> MaterializedHierarchicalResidualActionEvidence | None:
        matches = tuple(
            value
            for source in self.evidence_sources
            if (value := source.evidence_for(action_sha256)) is not None
        )
        if len(matches) > 1:
            raise ValueError(
                "one action must have exactly one forecast evidence row"
            )
        if not matches:
            if (
                self.coverage_mode
                is ResidualForecastEvidenceCoverageMode.COMPLETE
            ):
                raise ValueError(
                    "one action must have exactly one forecast evidence row"
                )
            return None
        return matches[0]

    def _member(
        self,
        evidence: MaterializedHierarchicalResidualActionEvidence,
        prior_by_id: dict[object, EvolutionCandidate],
    ) -> MaterializedForecastGeometryMember:
        parent = prior_by_id.get(evidence.plan.parent_candidate_id)
        if parent is None:
            raise ValueError("forecast action names a parent outside the cutoff")
        forecast_by_metric = {
            value.metric_id: value for value in evidence.effect_predictions
        }
        sense_by_metric = dict(self.objective_senses)
        if (
            set(forecast_by_metric) != set(parent.objective_map)
            or set(forecast_by_metric) != set(sense_by_metric)
        ):
            raise ValueError("forecast, parent, and objective frames differ")

        def point(kind: str) -> tuple[tuple[str, float], ...]:
            values = []
            for metric_id, parent_value in parent.objectives:
                forecast = forecast_by_metric[metric_id]
                sense = sense_by_metric[metric_id]
                if kind == "lower_numeric":
                    delta = forecast.p10_delta
                elif kind == "median":
                    delta = forecast.p50_delta
                elif kind == "upper_numeric":
                    delta = forecast.p90_delta
                elif kind == "favorable":
                    delta = (
                        forecast.p10_delta
                        if sense is MetricSense.MINIMIZE
                        else forecast.p90_delta
                    )
                elif kind == "adverse":
                    delta = (
                        forecast.p90_delta
                        if sense is MetricSense.MINIMIZE
                        else forecast.p10_delta
                    )
                else:  # pragma: no cover - closed call sites.
                    raise AssertionError("unknown forecast scenario")
                values.append((metric_id, parent_value + delta))
            return tuple(sorted(values))

        scenarios = tuple(
            sorted(
                (
                    ForecastGeometryScenario(
                        scenario_id=scenario_id,
                        objective_point=point(scenario_id),
                    )
                    for scenario_id in (
                        "adverse",
                        "favorable",
                        "lower_numeric",
                        "median",
                        "upper_numeric",
                    )
                ),
                key=lambda value: value.scenario_id,
            )
        )
        source_record = freeze_json(evidence.to_record())
        return MaterializedForecastGeometryMember(
            action_sha256=evidence.action.action_sha256,
            phenotype_identity_sha256=(
                evidence.action.phenotype_identity_sha256
            ),
            reliability=float(
                evidence.probability_valid
                * min(
                    value.confidence
                    for value in evidence.effect_predictions
                )
            ),
            scenarios=scenarios,
            source_evidence_sha256=typed_json_sha256(source_record),
        )

    async def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedForecastGeometryBatch:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        if any(
            value.generation >= request.decision_index
            for value in self.prior_candidates
        ):
            raise ValueError("prior candidates cross the current cutoff")
        if type(proposals) is not tuple or not proposals:
            raise ValueError("proposals must be a non-empty exact tuple")
        for proposal in proposals:
            if type(proposal) is not MaterializedActionProposalBatch:
                raise TypeError("proposals must contain exact batches")
            proposal.__post_init__()
            proposal.require_request(request)
        proposal_sha256s = tuple(
            sorted(value.proposal_sha256 for value in proposals)
        )
        async with self._lock:
            cached = self._cache.get(request.request_sha256)
            if cached is not None:
                if cached.proposal_sha256s != proposal_sha256s:
                    raise ValueError(
                        "one request identity names two proposal markets"
                    )
                return cached
            actions = tuple(
                action
                for proposal in proposals
                for action in proposal.actions
            )
            if len({value.action_sha256 for value in actions}) != len(actions):
                raise ValueError("proposal union repeats an action")
            prior_by_id = {
                value.candidate_id: value
                for value in self.prior_candidates
                if (
                    value.valid
                    and value.operator_compliant
                    and value.evidence_compliant
                )
            }
            member_values: list[MaterializedForecastGeometryMember] = []
            unprojected_action_sha256s: list[str] = []
            for action in actions:
                evidence = self._evidence(action.action_sha256)
                if evidence is None:
                    unprojected_action_sha256s.append(
                        action.action_sha256
                    )
                    continue
                member_values.append(
                    self._member(evidence, prior_by_id)
                )
            if not member_values:
                raise ValueError(
                    "forecast projection requires at least one evidence row"
                )
            members = tuple(
                sorted(
                    member_values,
                    key=lambda value: value.action_sha256,
                )
            )
            covered_action_sha256s = tuple(
                value.action_sha256 for value in members
            )
            unprojected = tuple(sorted(unprojected_action_sha256s))
            batch = MaterializedForecastGeometryBatch(
                projection_id=self.projection_id,
                projection_version=self.projection_version,
                projection_definition_sha256=self.definition_sha256,
                residual_request_sha256=request.request_sha256,
                proposal_sha256s=proposal_sha256s,
                members=members,
                candidate_outcomes_observed=False,
                evidence=freeze_json(
                    {
                        "member_source_evidence_sha256s": [
                            {
                                "action_sha256": value.action_sha256,
                                "source_evidence_sha256": (
                                    value.source_evidence_sha256
                                ),
                            }
                            for value in members
                        ],
                        "objective_senses": [
                            {
                                "metric_id": metric_id,
                                "sense": sense.value,
                            }
                            for metric_id, sense in self.objective_senses
                        ],
                        "coverage_mode": self.coverage_mode.value,
                        "proposal_action_count": len(actions),
                        "forecast_covered_action_sha256s": list(
                            covered_action_sha256s
                        ),
                        "unprojected_action_sha256s": list(unprojected),
                        "complete_action_coverage": not unprojected,
                        "unprojected_actions_remain_fallback_eligible": True,
                        "candidate_outcomes_observed": False,
                        "workload_model_provider_branches": False,
                    }
                ),
            )
            self._cache[request.request_sha256] = batch
            return batch


__all__ = [
    "PYDANTIC_AI_RESIDUAL_FORECAST_GEOMETRY_PROJECTION_ID",
    "PYDANTIC_AI_RESIDUAL_FORECAST_GEOMETRY_PROJECTION_VERSION",
    "PydanticAIResidualForecastGeometryProjection",
    "ResidualForecastEvidenceCoverageMode",
]
