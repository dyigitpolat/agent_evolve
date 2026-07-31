from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agent_evolve.application.agentic_evolution import AgenticEvolutionEngine
from agent_evolve.application.portfolio_evolution import (
    PortfolioEvolution,
    PortfolioVariationWaveRequest,
    PortfolioVariationWaveResult,
)
from agent_evolve.application.portfolio_campaign_runtime import (
    AgenticPortfolioCampaignRuntime,
)
from agent_evolve.application.campaign_variation_trace import (
    CampaignVariationTraceSummary,
    project_finite_contract_proposal_topology,
    summarize_campaign_variation_trace,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import InsightId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.ports.artifact_store import canonical_json_bytes
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_selection import (
    ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    CALIBRATED_PORTFOLIO_BASE_INSTRUCTION,
    CALIBRATED_PORTFOLIO_COMMON_POOL_PROMPT_DEFINITION_SHA256,
    CALIBRATED_PORTFOLIO_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256,
    CALIBRATED_PORTFOLIO_PROMPT_DEFINITION_SHA256,
    CALIBRATED_PORTFOLIO_PROJECTED_PROMPT_DEFINITION_SHA256,
    CALIBRATED_PORTFOLIO_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256,
    CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID,
    CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID,
    FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION,
    HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID,
    MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION,
    OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID,
    STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION,
    CalibratedPortfolioFeasibilityWitnessMode,
    PydanticAICalibratedPortfolioSelectionPolicy,
    PydanticAIAcquisitionCertifiedResidualPortfolioSelectionPolicy,
    PydanticAIRegretBoundedInformationPortfolioSelectionPolicy,
    PydanticAIConstraintDecoupledHorizonPortfolioSelectionPolicy,
    PydanticAIConstraintDecoupledTargetConditionedPortfolioSelectionPolicy,
    PydanticAIContextualSearchAllocationPortfolioSelectionPolicy,
    PydanticAIEvidenceCalibratedSourceMixPortfolioSelectionPolicy,
    PydanticAIMinimumInterventionHorizonPortfolioSelectionPolicy,
    PydanticAIFrontierProbeCalibratedPortfolioSelectionPolicy,
    PydanticAIHorizonBoundedCalibratedPortfolioSelectionPolicy,
    PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy,
    PydanticAIOperatorStratifiedCalibratedPortfolioSelectionPolicy,
    PydanticAIStructuralPosteriorCalibratedPortfolioSelectionPolicy,
    PydanticAITargetConditionedCalibratedPortfolioSelectionPolicy,
    TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    allocate_calibrated_portfolio_proposal,
    calibrated_portfolio_prompt_definition_sha256,
    decode_calibrated_portfolio_audit,
    decode_calibrated_portfolio_proposal,
    render_calibrated_portfolio_selection_prompt,
    render_calibrated_portfolio_selection_prompt_for_allocator,
)
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_campaign import (
    CalibratedPortfolioCampaignCoordinator,
)
from agent_evolve.integrations.pydantic_ai.portfolio_selection import (
    PORTFOLIO_SELECTION_DISJOINT_POLICY_DEFINITION_SHA256,
    PORTFOLIO_SELECTION_DISJOINT_POLICY_VERSION,
    PORTFOLIO_SELECTION_POLICY_VERSION,
    PydanticAIPortfolioSelectionPolicy,
    render_portfolio_selection_prompt,
)
from agent_evolve.policies.selection.calibrated_slate import (
    MetricOptimizationGoal,
    SlateMetricObjective,
    SlateStructuralEvidence,
)
from agent_evolve.policies.selection.acquisition_certified_slate import (
    AcquisitionCertifiedSlateContext,
    AcquisitionCertifiedSlateContextRegistry,
    AcquisitionCertifiedSlatePolicy,
)
from agent_evolve.policies.selection.regret_bounded_slate import (
    RegretBoundedSlatePolicy,
    ResidualInformationAssayValuePolicy,
)
from agent_evolve.policies.selection.calibrated_portfolio_binding import (
    CalibratedPortfolioAllocationContext,
    CalibratedPortfolioInputBinding,
    CalibratedPortfolioOptionEvidence,
    common_pool_required_option_ids,
)
from agent_evolve.policies.selection.common_candidate_pool import (
    TaskKeyedCommonCandidatePoolPolicy,
)
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationScope,
    ForecastCalibrationSnapshot,
)
from agent_evolve.policies.selection.model_anchored_slate import (
    ModelAnchoredCalibratedSlatePolicy,
    ModelAnchoredSlateDecision,
)
from agent_evolve.policies.selection.structural_posterior_slate import (
    HorizonBoundedStructuralPosteriorSlateDecision,
    HorizonBoundedStructuralPosteriorSlatePolicy,
    OperatorStratifiedStructuralPosteriorSlateDecision,
    OperatorStratifiedStructuralPosteriorSlatePolicy,
    StructuralPosteriorSlateDecision,
    StructuralPosteriorSlatePolicy,
    build_controller_owned_family_exposure_phases,
    build_terminal_tapered_family_exposure_phases,
)
from agent_evolve.policies.selection.frontier_probe_slate import (
    FrontierProbeSlateDecision,
    FrontierProbeSlatePolicy,
)
from agent_evolve.policies.selection.target_conditioned_allocator import (
    RegisteredTargetConditionedAllocationContextProvider,
    TargetConditionedAllocationContext,
    TargetConditionedSlateAllocatorAdapter,
)
from agent_evolve.policies.selection.target_conditioned_features import (
    project_portable_transition,
)
from agent_evolve.policies.selection.target_conditioned_prequential import (
    TargetConditionedAcquisitionProfile,
    TargetConditionedMetaPrior,
)
from agent_evolve.policies.selection.finite_option_prompt_projection import (
    FiniteOptionPromptProjectionPolicy,
)
from agent_evolve.policies.variation.compositional_finite_catalog import (
    BoundedCompositionalFiniteVariationCatalog,
    CompositionSelectionExposure,
)
from agent_evolve.policies.variation.source_union_finite_catalog import (
    SourceUnionFiniteVariationCatalog,
    required_source_evaluation_option_ids,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioCard,
    PortfolioSelectionRequest,
    PortfolioSelectionSupplementalAudit,
    finite_option_ids_have_pairwise_disjoint_parent_patch_subset,
    pairwise_disjoint_parent_patch_witness,
    project_family_exposure_bounds_to_pairwise_disjoint_feasibility,
    validate_pairwise_disjoint_parent_patch_selection,
)
from agent_evolve.ports.finite_acquisition import (
    FiniteAcquisitionCandidate,
    FiniteAcquisitionObjective,
    FiniteAcquisitionObservation,
)
from agent_evolve.ports.finite_acquisition_batch import (
    FiniteAcquisitionBatchScoreDecision,
    FiniteAcquisitionBatchScoreRequest,
    FiniteAcquisitionSlateScore,
)
from agent_evolve.ports.contextual_search_allocation import (
    ContextualPortfolioAllocationContract,
)
from agent_evolve.ports.archive_context import (
    CampaignPortfolioArchiveContextProjection,
)
from agent_evolve.ports.frontier_target import CampaignPortfolioFrontierTarget
from agent_evolve.ports.portfolio_memory_dose import (
    BoundedPortfolioMemoryDoseContract,
    PortfolioMemoryDoseCardSupport,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)
from agent_evolve.ports.variation_catalog import bind_finite_variation_catalog
from agent_evolve.ports.variation_source import (
    PRIMARY_VARIATION_SOURCE_ID,
    finite_variation_source_by_option,
)
from examples.benchmarks.boils_abc.actions import (
    ACTION_IDS,
    DEFAULT_ACTION_SEQUENCE,
    CandidateConfig,
)
from examples.benchmarks.boils_abc.finite_variation_catalog import (
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.global_restart_catalog import (
    GLOBAL_RESTART_SOURCE_ID,
    BoilsGlobalRestartVariationCatalog,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8", errors="strict")).hexdigest()


def _frozen(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _cards() -> tuple[PortfolioCard, ...]:
    return (
        PortfolioCard(
            card_key="card.a",
            reference=InsightRef(InsightId("insight_calibrated_card_a"), 1),
            content_sha256=_sha("card a content"),
            evidence_sha256=_sha("card a evidence"),
            prompt_payload=_frozen({"claim": "Test the first assigned hypothesis."}),
        ),
        PortfolioCard(
            card_key="card.b",
            reference=InsightRef(InsightId("insight_calibrated_card_b"), 1),
            content_sha256=_sha("card b content"),
            evidence_sha256=_sha("card b evidence"),
            prompt_payload=_frozen({"claim": "Test the second assigned hypothesis."}),
        ),
    )


def _boils_contract() -> FiniteVariationContract:
    return bind_finite_variation_catalog(
        BoilsFiniteVariationCatalog(),
        _frozen({"sequence": list(DEFAULT_ACTION_SEQUENCE)}),
    )


def _boils_source_union_contract() -> FiniteVariationContract:
    return bind_finite_variation_catalog(
        SourceUnionFiniteVariationCatalog(
            primary_catalog=BoilsFiniteVariationCatalog(),
            source_catalogs=(BoilsGlobalRestartVariationCatalog(),),
        ),
        _frozen({"sequence": list(DEFAULT_ACTION_SEQUENCE)}),
    )


def _boils_request(
    ids: DeterministicIdFactory,
    *,
    require_disjoint: bool,
) -> PortfolioSelectionRequest:
    return PortfolioSelectionRequest(
        call_id=ids.new_llm_call_id(),
        operation="select_calibrated_portfolio",
        instruction="Propose a generic sealed slate for engine allocation.",
        context=_frozen({"benchmark": "boils-finite-catalog-provider-free"}),
        finite_variation_contract=_boils_contract(),
        cards=_cards(),
        portfolio_size=4,
        required_metric_ids=("total_levels", "total_lut_count"),
        require_pairwise_disjoint_parent_patches=require_disjoint,
        max_output_tokens=32_768,
        temperature=0.0,
    )


def _boils_source_union_request(
    ids: DeterministicIdFactory,
) -> PortfolioSelectionRequest:
    request = _boils_request(ids, require_disjoint=False)
    return replace(
        request,
        finite_variation_contract=_boils_source_union_contract(),
    )


def _scope() -> ForecastCalibrationScope:
    return ForecastCalibrationScope(
        model_profile_sha256=_sha("provider-free-model-profile"),
        prompt_definition_sha256=CALIBRATED_PORTFOLIO_PROMPT_DEFINITION_SHA256,
        selector_policy_definition_sha256=(
            CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
        benchmark_sha256=_sha("boils-finite-catalog"),
        session_sha256=_sha("calibrated-provider-free-session"),
    )


def _context(
    request: PortfolioSelectionRequest,
) -> CalibratedPortfolioAllocationContext:
    scope = _scope()
    return CalibratedPortfolioAllocationContext(
        scope=scope,
        wave_index=2,
        parent_candidate_identity_sha256=(
            request.finite_variation_contract.parent_configuration_sha256
        ),
        objectives=(
            SlateMetricObjective(
                metric_id="total_levels",
                goal=MetricOptimizationGoal.MINIMIZE,
                weight=1.0,
                definition_sha256=_sha("minimize total levels"),
            ),
            SlateMetricObjective(
                metric_id="total_lut_count",
                goal=MetricOptimizationGoal.MINIMIZE,
                weight=1.0,
                definition_sha256=_sha("minimize total lut count"),
            ),
        ),
        assigned_card_keys=("card.a", "card.b"),
        calibration_snapshot=ForecastCalibrationSnapshot(
            scope=scope,
            cutoff_wave_index_exclusive=2,
            observations=(),
        ),
    )


def _position(option: FiniteVariationOption) -> str:
    return dict(option.metadata)["position"]


def _evidence(
    request: PortfolioSelectionRequest,
    option: FiniteVariationOption,
) -> CalibratedPortfolioOptionEvidence:
    del request
    metadata = dict(option.metadata)
    position = metadata.get("position", f"source.{option.family}")
    return CalibratedPortfolioOptionEvidence(
        option_id=option.option_id,
        option_identity_sha256=option.identity_sha256,
        locus_key=f"locus.p{position}",
        phenotype_identity_sha256=option.child_configuration_sha256,
        structural_evidence=SlateStructuralEvidence(
            frozen_archive_snapshot_sha256=_sha("frozen archive before wave 2"),
            evidence_receipt_sha256=_sha(f"structural evidence {option.option_id}"),
            archive_novelty_score=0.5,
            structural_coverage_score=0.5,
        ),
    )


def _binding(request: PortfolioSelectionRequest) -> CalibratedPortfolioInputBinding:
    return CalibratedPortfolioInputBinding(
        request_sha256=request.request_sha256,
        context=_context(request),
        option_evidence=tuple(
            sorted(
                (
                    _evidence(request, option)
                    for option in request.finite_variation_contract.options
                ),
                key=lambda value: value.option_id,
            )
        ),
    )


def _model_anchored_binding(
    request: PortfolioSelectionRequest,
) -> CalibratedPortfolioInputBinding:
    legacy = _binding(request)
    anchored_scope = replace(
        legacy.context.scope,
        selector_policy_definition_sha256=(
            MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    anchored_snapshot = replace(
        legacy.context.calibration_snapshot,
        scope=anchored_scope,
    )
    return replace(
        legacy,
        context=replace(
            legacy.context,
            scope=anchored_scope,
            calibration_snapshot=anchored_snapshot,
        ),
    )


def _structural_posterior_binding(
    request: PortfolioSelectionRequest,
) -> CalibratedPortfolioInputBinding:
    legacy = _binding(request)
    scope = replace(
        legacy.context.scope,
        selector_policy_definition_sha256=(
            STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    return replace(
        legacy,
        context=replace(
            legacy.context,
            scope=scope,
            calibration_snapshot=replace(
                legacy.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )


def _operator_stratified_binding(
    request: PortfolioSelectionRequest,
) -> CalibratedPortfolioInputBinding:
    legacy = _binding(request)
    scope = replace(
        legacy.context.scope,
        selector_policy_definition_sha256=(
            OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    return replace(
        legacy,
        context=replace(
            legacy.context,
            scope=scope,
            calibration_snapshot=replace(
                legacy.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )


def _horizon_bounded_binding(
    request: PortfolioSelectionRequest,
) -> CalibratedPortfolioInputBinding:
    legacy = _binding(request)
    scope = replace(
        legacy.context.scope,
        selector_policy_definition_sha256=(
            HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    return replace(
        legacy,
        context=replace(
            legacy.context,
            scope=scope,
            calibration_snapshot=replace(
                legacy.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )


def _frontier_probe_binding(
    request: PortfolioSelectionRequest,
) -> CalibratedPortfolioInputBinding:
    legacy = _binding(request)
    scope = replace(
        legacy.context.scope,
        selector_policy_definition_sha256=(
            FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    return replace(
        legacy,
        context=replace(
            legacy.context,
            scope=scope,
            calibration_snapshot=replace(
                legacy.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )


def _target_conditioned_binding(
    request: PortfolioSelectionRequest,
) -> CalibratedPortfolioInputBinding:
    legacy = _binding(request)
    scope = replace(
        legacy.context.scope,
        selector_policy_definition_sha256=(
            TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    return replace(
        legacy,
        context=replace(
            legacy.context,
            scope=scope,
            calibration_snapshot=replace(
                legacy.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )


def _target_conditioned_context(
    request: PortfolioSelectionRequest,
) -> tuple[TargetConditionedAcquisitionProfile, TargetConditionedAllocationContext]:
    artifact_path = (
        Path(__file__).resolve().parents[2]
        / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
        "trap_portable_profile_v1.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    profile = TargetConditionedAcquisitionProfile.from_record(artifact["profile"])
    prior = TargetConditionedMetaPrior.from_record(artifact["meta_prior"])
    parent_sha256 = request.finite_variation_contract.parent_configuration_sha256
    archive_sha256 = _sha("target-conditioned BOiLS archive utility")
    archive_context = CampaignPortfolioArchiveContextProjection(
        projector_id="fixture.affine",
        projector_version=1,
        definition_sha256=_sha("target-conditioned affine fixture"),
        archive_utility_snapshot_sha256=archive_sha256,
        parent_configuration_sha256=parent_sha256,
        payload=_frozen(
            {
                "optimization_frame": {
                    "axes": [
                        {"metric_id": "total_levels"},
                        {"metric_id": "total_lut_count"},
                    ],
                    "reference_directions": [
                        {
                            "direction_id": "balanced_tradeoff",
                            "normalized_importance_decimal": ["0.5", "0.5"],
                        }
                    ],
                    "base_hypervolume_decimal": "0.2",
                },
                "archive": {
                    "normalized_points_decimal": [["0.2", "0.7"], ["0.6", "0.2"]]
                },
                "parent": {"normalized_point_decimal": ["0.4", "0.4"]},
            }
        ),
    )
    frontier_target = CampaignPortfolioFrontierTarget(
        allocator_id="fixture",
        allocator_version=1,
        definition_sha256=_sha("target-conditioned target fixture"),
        archive_utility_snapshot_sha256=archive_sha256,
        lane_id="lane.0",
        parent_configuration_sha256=parent_sha256,
        direction_id="balanced_tradeoff",
        opportunity_rank=1,
        payload=_frozen(
            {
                "target_direction": {
                    "normalized_weights_decimal": ["1", "1"],
                    "opportunity_from_ideal_decimal": "0.4",
                },
                "assigned_parent": {
                    "normalized_point_decimal": ["0.4", "0.4"],
                    "achievement_decimal": "0.3",
                    "regret_above_archive_best_decimal": "0.1",
                },
            }
        ),
    )
    transitions = tuple(
        project_portable_transition(
            option_id=option.option_id,
            option_identity_sha256=option.identity_sha256,
            parent_configuration=request.finite_variation_contract.parent_configuration,
            child_configuration=option.child_configuration,
        )
        for option in sorted(
            request.finite_variation_contract.options,
            key=lambda value: value.option_id,
        )
    )
    return profile, TargetConditionedAllocationContext(
        finite_contract_sha256=request.finite_variation_contract.identity_sha256,
        cutoff_receipt_sha256=_sha("target-conditioned pre-call cutoff"),
        archive_context=archive_context,
        frontier_target=frontier_target,
        state=prior.initial_state(campaign_scope_sha256=_sha("BOiLS T-RAP campaign")),
        transition_receipts=transitions,
        campaign_generation=1,
        lane_slot=0,
        remaining_proposal_horizon=2,
    )


def _proposal_option_ids(contract: FiniteVariationContract) -> tuple[str, ...]:
    by_position: dict[str, list[str]] = {}
    for option in contract.options:
        by_position.setdefault(_position(option), []).append(option.option_id)
    # The first two proposals deliberately conflict at position 00.  A valid
    # K4 exists across the later distinct positions.
    return (
        by_position["00"][0],
        by_position["00"][1],
        by_position["01"][0],
        by_position["02"][0],
        by_position["03"][0],
        by_position["04"][0],
        by_position["05"][0],
        by_position["06"][0],
    )


class _ProviderFreeSlateRunner:
    def __init__(
        self,
        option_ids: tuple[str, ...],
        *,
        omit_second_card: bool = False,
    ) -> None:
        self.option_ids = option_ids
        self.omit_second_card = omit_second_card
        self.calls = 0
        self.requests: list[StructuredGenerationRequest[Any]] = []

    async def __call__(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        self.calls += 1
        self.requests.append(request)
        members = [
            {
                "option_id": option_id,
                "supporting_card_keys": [
                    "card.a" if self.omit_second_card or index % 2 else "card.b"
                ],
                "effect_predictions": [
                    {
                        "metric_id": "total_levels",
                        "direction": "decrease",
                        "confidence": "high",
                    },
                    {
                        "metric_id": "total_lut_count",
                        "direction": "decrease",
                        "confidence": "medium",
                    },
                ],
                "role_proposal": (
                    "exploit" if index < 4 else "falsify" if index < 7 else "coverage"
                ),
                "design_rationale": (
                    f"Provider-free rationale {index} for sealed option {option_id}."
                ),
            }
            for index, option_id in enumerate(self.option_ids, start=1)
        ]
        value = request.output_type.model_validate({"members": members}, strict=True)
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider-free/calibrated-v2",
            resolved_model="provider-free/calibrated-v2",
            resolved_provider="provider-free",
            provider_response_id="provider-free-calibrated-response",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=100,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1_000,
        )


@dataclass(frozen=True, slots=True)
class _FixtureAcquisitionBatchScorer:
    weights: tuple[tuple[str, float], ...]
    policy_id: str = "fixture_acquisition_batch_scorer"
    policy_version: int = 1
    definition_sha256: str = _sha("fixture acquisition batch scorer")

    def score(
        self,
        request: FiniteAcquisitionBatchScoreRequest,
    ) -> FiniteAcquisitionBatchScoreDecision:
        weight_by_id = dict(self.weights)
        return FiniteAcquisitionBatchScoreDecision(
            request_sha256=request.request_sha256,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            scores=tuple(
                FiniteAcquisitionSlateScore(
                    slate=slate,
                    log_acquisition_value=float(
                        sum(weight_by_id[value] for value in slate.candidate_ids)
                    ),
                )
                for slate in request.slates
            ),
        )


class _NoCandidateGenerator:
    async def propose(self, request):
        del request
        raise AssertionError("sealed portfolio members are engine materialized")

    async def reflect(self, request):
        del request
        raise AssertionError("this provider-free wave does not reflect")


class _CountingBoilsProblem:
    candidate_model = CandidateConfig
    objectives = (
        ObjectiveSpec("total_lut_count", "min"),
        ObjectiveSpec("total_levels", "min"),
    )

    def __init__(self) -> None:
        self.evaluations = 0

    @staticmethod
    def validate(configuration: object) -> bool:
        CandidateConfig.model_validate(configuration, strict=True)
        return True

    def evaluate(self, configuration: dict[str, object]) -> dict[str, float]:
        candidate = CandidateConfig.model_validate(configuration, strict=True)
        self.evaluations += 1
        action_index = {action_id: index for index, action_id in enumerate(ACTION_IDS)}
        ordinal_sum = sum(action_index[value] for value in candidate.sequence)
        return {
            "total_lut_count": float(12_000 - ordinal_sum),
            "total_levels": float(80 + (ordinal_sum % 7)),
        }


def test_real_boils_catalog_k8_proposal_allocates_only_four_disjoint_evaluations() -> (
    None
):
    async def scenario():
        ids = DeterministicIdFactory("calibrated_boils_e2e")
        problem = _CountingBoilsProblem()
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=_NoCandidateGenerator(),
            id_factory=ids,
            memory=None,
            seed=17,
            evaluator_concurrency=4,
        )
        parent = await engine.register_seed(
            {"sequence": list(DEFAULT_ACTION_SEQUENCE)},
            label="parent",
        )
        problem.evaluations = 0
        request = _boils_request(ids, require_disjoint=True)
        assert request.finite_variation_contract.parent_configuration == (
            parent.configuration
        )
        proposal_ids = _proposal_option_ids(request.finite_variation_contract)
        runner = _ProviderFreeSlateRunner(proposal_ids)
        selector = PydanticAICalibratedPortfolioSelectionPolicy(
            generate_once=runner,
            binding_for=_binding,
        )
        result = await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
        ).run(
            PortfolioVariationWaveRequest(
                selection_request=request,
                parent=parent,
                generation=2,
                label_prefix="calibrated_boils",
            )
        )
        return problem, runner, proposal_ids, request, result

    problem, runner, proposal_ids, request, result = asyncio.run(scenario())

    assert runner.calls == 1
    assert len(proposal_ids) == 8
    assert problem.evaluations == 4
    assert len(result.receipt.members) == 4
    selected_ids = tuple(
        value.materialization.option_id for value in result.receipt.members
    )
    assert len(set(selected_ids)) == 4
    validate_pairwise_disjoint_parent_patch_selection(
        request.finite_variation_contract,
        selected_ids,
    )
    selected_positions = tuple(
        _position(request.finite_variation_contract.resolve(option_id))
        for option_id in selected_ids
    )
    assert len(set(selected_positions)) == 4

    audit = result.selection_decision_audit_record
    assert audit is not None
    supplemental = audit["supplemental_selector_audit"]
    payload = supplemental["payload"]
    assert payload["proposal_size"] == 8
    assert payload["evaluation_size"] == 4
    assert len(payload["original_k8_response"]["members"]) == 8
    assert len(payload["allocation"]["selected"]) == 4
    assert payload["invariants"]["only_selected_k4_reaches_evaluator"] is True
    assert payload["invariants"]["prior_only"] is True
    assert payload["invariants"]["caller_instruction_rendered"] is False
    assert payload["invariants"]["input_binding_frozen_before_call"] is True
    assert payload["invariants"]["administered_card_keys"] == ["card.a", "card.b"]
    original_rationales = tuple(
        value["design_rationale"]
        for value in payload["original_k8_response"]["members"]
    )
    assert original_rationales == tuple(
        f"Provider-free rationale {index} for sealed option {option_id}."
        for index, option_id in enumerate(proposal_ids, start=1)
    )
    assert result.supplemental_selection_audit is not None
    assert supplemental["audit_sha256"] == (
        result.supplemental_selection_audit.audit_sha256
    )
    decoded = decode_calibrated_portfolio_audit(
        result.supplemental_selection_audit,
        request=request,
        binding=_binding(request),
    )
    assert decoded.slate.to_record() == payload["calibrated_slate"]
    assert decoded.allocation.to_record() == payload["allocation"]
    assert len(decoded.selected_prediction_receipts) == 8
    assert {value.option_id for value in decoded.selected_prediction_receipts} == set(
        selected_ids
    )
    assert tuple(
        value.receipt_sha256 for value in decoded.selected_prediction_receipts
    ) == tuple(payload["selected_prediction_receipt_sha256s"])
    assert not hasattr(decoded, "design_rationale")


async def _registered_calibrated_wave() -> tuple[
    PortfolioSelectionRequest,
    _ProviderFreeSlateRunner,
    CalibratedPortfolioCampaignCoordinator,
    PortfolioVariationWaveRequest,
    PortfolioVariationWaveResult,
]:
    ids = DeterministicIdFactory("calibrated_campaign_coordinator")
    problem = _CountingBoilsProblem()
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=_NoCandidateGenerator(),
        id_factory=ids,
        memory=None,
        seed=17,
        evaluator_concurrency=4,
    )
    parent = await engine.register_seed(
        {"sequence": list(DEFAULT_ACTION_SEQUENCE)},
        label="parent",
    )
    request = _boils_request(ids, require_disjoint=True)
    runner = _ProviderFreeSlateRunner(
        _proposal_option_ids(request.finite_variation_contract)
    )
    coordinator = CalibratedPortfolioCampaignCoordinator()
    coordinator.register(request, _binding(request))
    selector = coordinator.build_selector(runner)
    assert type(selector) is PydanticAICalibratedPortfolioSelectionPolicy
    wave = PortfolioVariationWaveRequest(
        selection_request=request,
        parent=parent,
        generation=2,
        label_prefix="calibrated_campaign_bridge",
    )
    result = await PortfolioEvolution(
        engine=engine,
        selector=selector,
        ids=ids,
    ).run(wave)
    return request, runner, coordinator, wave, result


def test_campaign_coordinator_seals_prompt_and_strictly_decodes_k4() -> None:
    request, runner, coordinator, wave, result = asyncio.run(
        _registered_calibrated_wave()
    )
    assert coordinator.registered_request_count == 1
    assert coordinator.binding_for(request) == _binding(request)
    with pytest.raises(ValueError, match="already registered"):
        coordinator.register(request, _binding(request))

    tampered_request = replace(
        request,
        instruction="A changed instruction after prospective registration.",
    )
    with pytest.raises(ValueError, match="foreign or unregistered"):
        coordinator.binding_for(tampered_request)

    selector_audit = AgenticPortfolioCampaignRuntime._selector_audit(
        generation=2,
        parent_slot=0,
        wave=wave,
        result=result,
        prior_audit_set_sha256=_sha("prior selector audit set"),
        prompt_renderer=coordinator,
    )
    plaintext = thaw_json(selector_audit.plaintext_audit)
    assert plaintext["request_text"] == runner.requests[0].prompt
    assert plaintext["request_text"] == coordinator.render(request)
    assert plaintext["request_text"].startswith(CALIBRATED_PORTFOLIO_BASE_INSTRUCTION)

    predictions = coordinator.decode_selected_predictions(result)
    assert len(predictions) == 8
    selected_option_ids = tuple(
        value.materialization.option_id for value in result.receipt.members
    )
    assert tuple(dict.fromkeys(value.option_id for value in predictions)) == (
        selected_option_ids
    )
    assert (
        tuple(value.metric_id for value in predictions)
        == (
            "total_levels",
            "total_lut_count",
        )
        * 4
    )
    assert coordinator.decode_selected_source_ids(result) == (
        "primary",
        "primary",
        "primary",
        "primary",
    )

    audit = result.supplemental_selection_audit
    assert audit is not None
    payload = thaw_json(audit.payload)
    payload["invariants"]["only_selected_k4_reaches_evaluator"] = False
    frozen = freeze_json(payload)
    assert type(frozen) is FrozenJsonObject
    tampered_audit = PortfolioSelectionSupplementalAudit(
        audit_kind=audit.audit_kind,
        request_sha256=audit.request_sha256,
        decision_sha256=audit.decision_sha256,
        payload=frozen,
    )
    tampered_result = replace(
        result,
        supplemental_selection_audit=tampered_audit,
    )
    with pytest.raises(ValueError, match="exact typed replay"):
        coordinator.decode_selected_predictions(tampered_result)

    foreign_coordinator = CalibratedPortfolioCampaignCoordinator()
    with pytest.raises(ValueError, match="foreign campaign request"):
        foreign_coordinator.decode_selected_predictions(result)
    with pytest.raises(ValueError, match="foreign campaign request"):
        foreign_coordinator.decode_selected_source_ids(result)


def test_target_conditioned_campaign_registers_precall_context_and_replays() -> None:
    ids = DeterministicIdFactory("target_conditioned_campaign_bridge")
    request = _boils_request(ids, require_disjoint=True)
    binding = _target_conditioned_binding(request)
    profile, context = _target_conditioned_context(request)
    context_provider = RegisteredTargetConditionedAllocationContextProvider()
    allocator = TargetConditionedSlateAllocatorAdapter(
        context_provider=context_provider,
        profile=profile,
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(allocator=allocator)
    runner = _ProviderFreeSlateRunner(
        _proposal_option_ids(request.finite_variation_contract)
    )

    with pytest.raises(TypeError, match="pre-call context"):
        coordinator.register(request, binding)
    assert coordinator.registered_request_count == 0
    assert context_provider.registered_context_count == 0

    coordinator.register(
        request,
        binding,
        target_conditioned_context=context,
    )
    selector = coordinator.build_selector(runner)
    assert type(selector) is (
        PydanticAITargetConditionedCalibratedPortfolioSelectionPolicy
    )

    result = asyncio.run(selector.select(request))

    assert runner.calls == 1
    assert coordinator.registered_request_count == 1
    assert context_provider.registered_context_count == 1
    assert result.supplemental_audit is not None
    assert result.supplemental_audit.audit_kind == (
        "target_conditioned_calibrated_portfolio_k8_to_k4"
    )
    decoded = decode_calibrated_portfolio_audit(
        result.supplemental_audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert len(decoded.allocation.selected) == 4
    assert decoded.allocation.prior_only is True
    assert tuple(value.option_id for value in decoded.allocation.selected) == tuple(
        value.option_id for value in result.decision.members
    )
    assert all(
        value.role.value == "target_conditioned_acquisition"
        for value in decoded.allocation.selected
    )


def test_constraint_decoupled_target_conditioned_campaign_reconciles_and_replays() -> (
    None
):
    ids = DeterministicIdFactory("constraint_decoupled_target_campaign_bridge")
    request = _boils_request(ids, require_disjoint=True)
    strict_binding = _target_conditioned_binding(request)
    scope = replace(
        strict_binding.context.scope,
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            constraint_decoupled=True,
        ),
        selector_policy_definition_sha256=(
            CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = replace(
        strict_binding,
        context=replace(
            strict_binding.context,
            scope=scope,
            calibration_snapshot=replace(
                strict_binding.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )
    profile, context = _target_conditioned_context(request)
    context_provider = RegisteredTargetConditionedAllocationContextProvider()
    allocator = TargetConditionedSlateAllocatorAdapter(
        context_provider=context_provider,
        profile=profile,
    )
    duplicate_option_id = request.finite_variation_contract.options[0].option_id
    runner = _ProviderFreeSlateRunner((duplicate_option_id,) * 8)
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=allocator,
        constraint_decoupled=True,
    )
    coordinator.register(
        request,
        binding,
        target_conditioned_context=context,
    )

    selector = coordinator.build_selector(runner)
    assert type(selector) is (
        PydanticAIConstraintDecoupledTargetConditionedPortfolioSelectionPolicy
    )
    result = asyncio.run(selector.select(request))

    audit = result.supplemental_audit
    assert audit is not None
    assert audit.audit_kind == (
        "constraint_decoupled_target_conditioned_portfolio_k8_to_k4"
    )
    payload = thaw_json(audit.payload)
    assert type(payload) is dict
    assert tuple(
        value["option_id"]
        for value in payload["original_model_response"]["members"]
    ) == (duplicate_option_id,) * 8
    reconciled_ids = tuple(
        value["option_id"] for value in payload["original_k8_response"]["members"]
    )
    assert len(set(reconciled_ids)) == 8
    assert payload["semantic_reconciliation"]["duplicate_model_member_count"] == 7
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert len(decoded.allocation.selected) == 4
    assert decoded.allocation.prior_only is True


def test_constraint_decoupled_target_guarantees_global_source_evaluation() -> None:
    ids = DeterministicIdFactory("constraint_target_global_source_bridge")
    request = _boils_source_union_request(ids)
    required_source_ids = required_source_evaluation_option_ids(
        request.finite_variation_contract
    )
    assert len(required_source_ids) == 1
    required_source_id = required_source_ids[0]
    assert (
        dict(request.finite_variation_contract.resolve(required_source_id).metadata)[
            "evaluation_source"
        ]
        == GLOBAL_RESTART_SOURCE_ID
    )

    strict_binding = _target_conditioned_binding(request)
    scope = replace(
        strict_binding.context.scope,
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            constraint_decoupled=True,
        ),
        selector_policy_definition_sha256=(
            CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = replace(
        strict_binding,
        context=replace(
            strict_binding.context,
            scope=scope,
            calibration_snapshot=replace(
                strict_binding.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )
    profile, context = _target_conditioned_context(request)
    context_provider = RegisteredTargetConditionedAllocationContextProvider()
    allocator = TargetConditionedSlateAllocatorAdapter(
        context_provider=context_provider,
        profile=profile,
    )
    local_option_id = request.finite_variation_contract.options[0].option_id
    runner = _ProviderFreeSlateRunner((local_option_id,) * 8)
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=allocator,
        constraint_decoupled=True,
    )
    coordinator.register(
        request,
        binding,
        target_conditioned_context=context,
    )

    result = asyncio.run(coordinator.build_selector(runner).select(request))

    audit = result.supplemental_audit
    assert audit is not None
    payload = thaw_json(audit.payload)
    assert type(payload) is dict
    original_ids = tuple(
        value["option_id"] for value in payload["original_model_response"]["members"]
    )
    reconciled_ids = tuple(
        value["option_id"] for value in payload["original_k8_response"]["members"]
    )
    selected_ids = tuple(value.option_id for value in result.decision.members)
    assert required_source_id not in original_ids
    assert required_source_id in reconciled_ids
    assert required_source_id in selected_ids
    source_receipt = next(
        value
        for value in payload["semantic_reconciliation"]["members"]
        if value["option_id"] == required_source_id
    )
    assert "required_evaluation_source" in source_receipt["reasons"]

    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert required_source_id in tuple(
        value.option_id for value in decoded.allocation.selected
    )
    assert required_source_id in (
        decoded.allocation.request.allocation_request.required_option_ids
    )


def test_default_calibrated_prompt_and_audit_bytes_remain_golden() -> None:
    ids = DeterministicIdFactory("calibrated_default_bytes_golden")
    request = _boils_request(ids, require_disjoint=True)
    binding = _binding(request)
    prompt = render_calibrated_portfolio_selection_prompt(request, binding)
    runner = _ProviderFreeSlateRunner(
        _proposal_option_ids(request.finite_variation_contract)
    )
    result = asyncio.run(
        PydanticAICalibratedPortfolioSelectionPolicy(
            runner,
            lambda _: binding,
        ).select(request)
    )
    audit = result.supplemental_audit
    assert audit is not None
    payload_bytes = json.dumps(
        thaw_json(audit.payload),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")

    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == (
        "4c7b16fd7d9a42233536b000d7914e5889bf2fb27746e81367756f5bc3a8545a"
    )
    assert binding.binding_sha256 == (
        "7c5bace7b811f44452d9e45456eef0a5293731a9554d97a72852653c42ce7420"
    )
    assert result.decision.decision_sha256 == (
        "8589dc6121fdfc7db5028f75d8f02005d69293d77922a4cc7fd409409077fe09"
    )
    assert hashlib.sha256(payload_bytes).hexdigest() == (
        "d73681c7b6ba7ac0f2ca935929d21b9d6e4d554e2620be2de87cef49656c2058"
    )
    assert audit.audit_sha256 == (
        "9225cb9a372d900546d945b2345558d53364fc60e86b750595d823dd241a55e7"
    )
    proposal = decode_calibrated_portfolio_proposal(
        audit,
        request=request,
        binding=binding,
    )
    assert len(proposal.model_members) == 8
    assert proposal.selector_policy_definition_sha256 == (
        CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )
    counterfactual = allocate_calibrated_portfolio_proposal(
        proposal,
        request=request,
        binding=binding,
        allocator=ModelAnchoredCalibratedSlatePolicy(model_anchor_count=3),
    )
    assert type(counterfactual) is ModelAnchoredSlateDecision
    assert len(counterfactual.selected) == 4

    anchored_binding = _model_anchored_binding(request)
    anchored_prompt = render_calibrated_portfolio_selection_prompt_for_allocator(
        request,
        anchored_binding,
        ModelAnchoredCalibratedSlatePolicy(),
    )
    assert hashlib.sha256(anchored_prompt.encode("utf-8")).hexdigest() == (
        "411f23dbb5289bebe8f2b0461bd3cafdf4e582a29a420dc6bce2f4caca75510f"
    )
    assert anchored_binding.binding_sha256 == (
        "12e85d0c0794c87e7c49aaad23922ece353a1714805cb5bafca23dfbbb7e3bc1"
    )


def test_projected_prompt_and_audit_use_only_the_binding_owned_projection() -> None:
    ids = DeterministicIdFactory("calibrated_projection_replay")
    request = _boils_request(ids, require_disjoint=True)
    legacy_binding = _binding(request)
    policy = FiniteOptionPromptProjectionPolicy(
        metadata_keys=("abc_commands_json", "position", "replacement_action")
    )
    with pytest.raises(ValueError, match="foreign prompt definition"):
        render_calibrated_portfolio_selection_prompt(
            request,
            replace(
                legacy_binding,
                option_prompt_projection=policy.project(
                    request.finite_variation_contract
                ),
            ),
        )
    projected_scope = replace(
        legacy_binding.context.scope,
        prompt_definition_sha256=(
            calibrated_portfolio_prompt_definition_sha256(policy)
        ),
    )
    projected_context = replace(
        legacy_binding.context,
        scope=projected_scope,
        calibration_snapshot=replace(
            legacy_binding.context.calibration_snapshot,
            scope=projected_scope,
        ),
    )
    projected_binding = replace(
        legacy_binding,
        context=projected_context,
        option_prompt_projection=policy.project(request.finite_variation_contract),
    )
    prompt = render_calibrated_portfolio_selection_prompt(
        request,
        projected_binding,
    )
    machine_contract = json.loads(prompt.splitlines()[3])
    projection = projected_binding.option_prompt_projection
    assert projection is not None
    assert machine_contract["option_prompt_projection"] == (
        projection.to_prompt_contract_record()
    )
    assert machine_contract["schema_version"] == 4
    assert (
        machine_contract["proposal_constraints"][
            "require_pairwise_disjoint_parent_patches"
        ]
        is True
    )
    # Preserve the historical prompt bytes when this optional constraint is
    # absent; a null field would change every workload prompt without adding a
    # constraint.  Non-null values remain rendered and hash-bound.
    assert "min_distinct_families" not in machine_contract["proposal_constraints"]
    assert "pairwise disjoint" in prompt
    assert machine_contract["prompt_definition_sha256"] == (
        CALIBRATED_PORTFOLIO_PROJECTED_PROMPT_DEFINITION_SHA256
    )
    assert machine_contract["ordered_options"] == list(projection.prompt_records())
    removed_value = dict(request.finite_variation_contract.options[0].metadata)[
        "action_definition_sha256"
    ]
    assert removed_value not in prompt

    runner = _ProviderFreeSlateRunner(
        _proposal_option_ids(request.finite_variation_contract)
    )
    result = asyncio.run(
        PydanticAICalibratedPortfolioSelectionPolicy(
            runner,
            lambda _: projected_binding,
        ).select(request)
    )
    audit = result.supplemental_audit
    assert audit is not None
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=projected_binding,
    )
    assert len(decoded.allocation.selected) == 4
    audit_payload = thaw_json(audit.payload)
    assert audit_payload["schema_version"] == 3
    assert audit_payload["prompt_definition_sha256"] == (
        CALIBRATED_PORTFOLIO_PROJECTED_PROMPT_DEFINITION_SHA256
    )
    assert runner.requests[0].prompt == prompt

    with pytest.raises(ValueError, match="header is not authenticated"):
        decode_calibrated_portfolio_audit(
            audit,
            request=request,
            binding=legacy_binding,
        )

    tampered_projection = replace(
        projection,
        records=tuple(reversed(projection.records)),
    )
    with pytest.raises(ValueError, match="differs from the sealed contract"):
        render_calibrated_portfolio_selection_prompt(
            request,
            replace(
                projected_binding,
                option_prompt_projection=tampered_projection,
            ),
        )

    anchored_allocator = ModelAnchoredCalibratedSlatePolicy()
    anchored_legacy_binding = _model_anchored_binding(request)
    anchored_projected_scope = replace(
        anchored_legacy_binding.context.scope,
        prompt_definition_sha256=(
            CALIBRATED_PORTFOLIO_PROJECTED_PROMPT_DEFINITION_SHA256
        ),
    )
    anchored_binding = replace(
        anchored_legacy_binding,
        context=replace(
            anchored_legacy_binding.context,
            scope=anchored_projected_scope,
            calibration_snapshot=replace(
                anchored_legacy_binding.context.calibration_snapshot,
                scope=anchored_projected_scope,
            ),
        ),
        option_prompt_projection=projection,
    )
    anchored_runner = _ProviderFreeSlateRunner(
        _proposal_option_ids(request.finite_variation_contract)
    )
    anchored_result = asyncio.run(
        PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy(
            anchored_runner,
            lambda _: anchored_binding,
            allocator=anchored_allocator,
        ).select(request)
    )
    anchored_audit = anchored_result.supplemental_audit
    assert anchored_audit is not None
    anchored_decoded = decode_calibrated_portfolio_audit(
        anchored_audit,
        request=request,
        binding=anchored_binding,
        allocator=anchored_allocator,
    )
    assert type(anchored_decoded.allocation) is ModelAnchoredSlateDecision
    anchored_payload = thaw_json(anchored_audit.payload)
    assert anchored_payload["schema_version"] == 4
    assert (
        json.loads(anchored_runner.requests[0].prompt.splitlines()[3])[
            "option_prompt_projection"
        ]
        == projection.to_prompt_contract_record()
    )


def test_prompt_renders_the_complete_hard_engine_allocation_constraint() -> None:
    request = replace(
        _boils_request(
            DeterministicIdFactory("calibrated_hard_feasibility"),
            require_disjoint=True,
        ),
        min_distinct_families=3,
    )
    request.__post_init__()
    prompt = render_calibrated_portfolio_selection_prompt(request, _binding(request))
    machine_contract = json.loads(prompt.splitlines()[3])

    assert machine_contract["proposal_constraints"] == {
        "proposal_size": 8,
        "engine_evaluation_size": 4,
        "distinct_option_ids": True,
        "required_metric_ids": ["total_levels", "total_lut_count"],
        "confidence_bins": ["low", "medium", "high", "unknown"],
        "role_proposals": ["exploit", "falsify", "coverage"],
        "assigned_card_keys": ["card.a", "card.b"],
        "require_supporting_cards": True,
        "require_pairwise_disjoint_parent_patches": True,
        "engine_verified_feasible_option_id_witness": [
            "boils_abc.p00.blut",
            "boils_abc.p01.balance",
            "boils_abc.p02.balance",
            "boils_abc.p03.dsdb",
        ],
        "witness_objective_values_consulted": False,
        "witness_is_quality_recommendation": False,
        "min_distinct_families": 3,
    }
    witness = pairwise_disjoint_parent_patch_witness(
        request.finite_variation_contract,
        tuple(option.option_id for option in request.finite_variation_contract.options),
        portfolio_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
    )
    assert witness == tuple(
        machine_contract["proposal_constraints"][
            "engine_verified_feasible_option_id_witness"
        ]
    )
    assert "subset of exactly 4 options" in prompt
    assert "pairwise disjoint and spans at least 3 distinct option families" in prompt


def test_request_keyed_witness_is_authenticated_diverse_and_outcome_free() -> None:
    request = _boils_request(
        DeterministicIdFactory("calibrated_request_keyed_witness"),
        require_disjoint=True,
    )
    canonical_binding = _binding(request)
    keyed_prompt_definition = calibrated_portfolio_prompt_definition_sha256(
        feasibility_witness_mode=(
            CalibratedPortfolioFeasibilityWitnessMode.REQUEST_KEYED
        )
    )
    assert keyed_prompt_definition == (
        CALIBRATED_PORTFOLIO_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256
    )
    keyed_scope = replace(
        canonical_binding.context.scope,
        prompt_definition_sha256=keyed_prompt_definition,
    )
    keyed_context = replace(
        canonical_binding.context,
        scope=keyed_scope,
        calibration_snapshot=replace(
            canonical_binding.context.calibration_snapshot,
            scope=keyed_scope,
        ),
    )
    keyed_binding = replace(canonical_binding, context=keyed_context)

    canonical_contract = json.loads(
        render_calibrated_portfolio_selection_prompt(
            request,
            canonical_binding,
        ).splitlines()[3]
    )
    keyed_prompt = render_calibrated_portfolio_selection_prompt(
        request,
        keyed_binding,
    )
    keyed_contract = json.loads(keyed_prompt.splitlines()[3])
    canonical_witness = tuple(
        canonical_contract["proposal_constraints"][
            "engine_verified_feasible_option_id_witness"
        ]
    )
    keyed_constraints = keyed_contract["proposal_constraints"]
    keyed_witness = tuple(
        keyed_constraints["engine_verified_feasible_option_id_witness"]
    )

    assert keyed_contract["schema_version"] == 7
    assert keyed_witness != canonical_witness
    assert keyed_constraints["witness_ordering_policy"] == (
        "request_keyed_domain_separated_sha256_v1"
    )
    assert keyed_constraints["witness_ordering_key_sha256"] == (request.request_sha256)
    assert keyed_constraints["witness_objective_values_consulted"] is False
    assert keyed_witness == pairwise_disjoint_parent_patch_witness(
        request.finite_variation_contract,
        tuple(option.option_id for option in request.finite_variation_contract.options),
        portfolio_size=request.portfolio_size,
        ordering_key_sha256=request.request_sha256,
    )
    assert (
        validate_pairwise_disjoint_parent_patch_selection(
            request.finite_variation_contract,
            keyed_witness,
        )
        is None
    )
    assert (
        render_calibrated_portfolio_selection_prompt(
            request,
            keyed_binding,
        )
        == keyed_prompt
    )


def test_pairwise_witness_preserves_a_feasible_semantic_preference() -> None:
    request = _boils_request(
        DeterministicIdFactory("minimum_intervention_witness"),
        require_disjoint=True,
    )
    option_ids = tuple(
        option.option_id for option in request.finite_variation_contract.options
    )
    canonical = pairwise_disjoint_parent_patch_witness(
        request.finite_variation_contract,
        option_ids,
        portfolio_size=request.portfolio_size,
    )
    preferred = pairwise_disjoint_parent_patch_witness(
        request.finite_variation_contract,
        option_ids,
        portfolio_size=request.portfolio_size,
        ordering_key_sha256=_sha("minimum intervention alternative witness"),
    )
    assert canonical is not None and preferred is not None
    assert preferred != canonical

    model_first_pool = tuple(dict.fromkeys((*preferred, *option_ids)))
    assert pairwise_disjoint_parent_patch_witness(
        request.finite_variation_contract,
        model_first_pool,
        portfolio_size=request.portfolio_size,
    ) == tuple(sorted(canonical))
    projected = pairwise_disjoint_parent_patch_witness(
        request.finite_variation_contract,
        model_first_pool,
        portfolio_size=request.portfolio_size,
        preferred_option_ids=preferred,
    )
    assert projected == preferred


def test_hidden_feasibility_certificate_withholds_anchor_but_preserves_gate() -> None:
    request = _boils_request(
        DeterministicIdFactory("calibrated_hidden_witness"),
        require_disjoint=True,
    )
    canonical_binding = _binding(request)
    hidden_prompt_definition = calibrated_portfolio_prompt_definition_sha256(
        feasibility_witness_mode=(
            CalibratedPortfolioFeasibilityWitnessMode.HIDDEN_CERTIFICATE
        )
    )
    assert hidden_prompt_definition == (
        CALIBRATED_PORTFOLIO_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256
    )
    hidden_scope = replace(
        canonical_binding.context.scope,
        prompt_definition_sha256=hidden_prompt_definition,
    )
    hidden_context = replace(
        canonical_binding.context,
        scope=hidden_scope,
        calibration_snapshot=replace(
            canonical_binding.context.calibration_snapshot,
            scope=hidden_scope,
        ),
    )
    hidden_binding = replace(canonical_binding, context=hidden_context)

    prompt = render_calibrated_portfolio_selection_prompt(
        request,
        hidden_binding,
    )
    machine_contract = json.loads(prompt.splitlines()[3])
    constraints = machine_contract["proposal_constraints"]
    certificate = constraints["engine_verified_feasibility_certificate"]

    assert machine_contract["schema_version"] == 15
    assert "engine_verified_feasible_option_id_witness" not in constraints
    assert certificate["feasible_subset_exists"] is True
    assert certificate["member_option_ids_rendered"] is False
    assert certificate["objective_values_consulted"] is False
    assert certificate["is_quality_recommendation"] is False
    assert len(certificate["certificate_sha256"]) == 64
    assert "withholds its member IDs to prevent answer anchoring" in prompt
    assert (
        render_calibrated_portfolio_selection_prompt(request, hidden_binding) == prompt
    )


def test_current_200_option_universe_uses_complete_wire_enum() -> None:
    request = _boils_request(
        DeterministicIdFactory("calibrated_exact_option_wire"),
        require_disjoint=True,
    )
    binding = _binding(request)

    async def schema_runner(low_level_request: StructuredGenerationRequest[Any]):
        schema = low_level_request.output_type.model_json_schema()
        option_schema = schema["$defs"]["CalibratedPortfolioSlateMember"]["properties"][
            "option_id"
        ]
        assert option_schema["enum"] == [
            option.option_id for option in request.finite_variation_contract.options
        ]
        assert len(low_level_request.repair_literal_sets) == 1
        repair_set = low_level_request.repair_literal_sets[0]
        assert repair_set.field_path == ("members", "*", "option_id")
        assert repair_set.allowed_literals == tuple(
            option.option_id for option in request.finite_variation_contract.options
        )
        raise AssertionError("exact enum schema inspected")

    with pytest.raises(AssertionError, match="schema inspected"):
        asyncio.run(
            PydanticAICalibratedPortfolioSelectionPolicy(
                generate_once=schema_runner,
                binding_for=lambda _: binding,
            ).select(request)
        )


def test_over_bound_finite_universe_uses_compact_wire_and_exact_local_gate() -> None:
    base_request = _boils_request(
        DeterministicIdFactory("calibrated_over_bound_option_wire"),
        require_disjoint=True,
    )
    over_bound_contract = replace(
        base_request.finite_variation_contract,
        options=tuple(
            replace(
                option,
                option_id=f"{option.option_id}.{'x' * 120}",
            )
            for option in base_request.finite_variation_contract.options
        ),
    )
    request = replace(
        base_request,
        finite_variation_contract=over_bound_contract,
    )
    binding = _binding(request)
    valid_option_ids = _proposal_option_ids(request.finite_variation_contract)

    async def invalid_runner(low_level_request: StructuredGenerationRequest[Any]):
        schema = low_level_request.output_type.model_json_schema()
        option_schema = schema["$defs"]["CalibratedPortfolioSlateMember"]["properties"][
            "option_id"
        ]
        assert "enum" not in option_schema
        assert option_schema["type"] == "string"
        assert option_schema["maxLength"] >= max(map(len, valid_option_ids))
        assert low_level_request.repair_literal_sets == ()
        members = [
            {
                "option_id": ("option.foreign" if index == 0 else option_id),
                "supporting_card_keys": ["card.a" if index % 2 else "card.b"],
                "effect_predictions": [
                    {
                        "metric_id": metric_id,
                        "direction": "unknown",
                        "confidence": "unknown",
                    }
                    for metric_id in request.required_metric_ids
                ],
                "role_proposal": "coverage",
                "design_rationale": "Exercise exact local finite-option membership.",
            }
            for index, option_id in enumerate(valid_option_ids)
        ]
        with pytest.raises(ValidationError) as caught:
            low_level_request.output_type.model_validate(
                {"members": members}, strict=True
            )
        assert caught.value.errors(include_url=False)[0]["type"] == (
            "finite_option_out_of_contract"
        )
        raise AssertionError("foreign compact option must not reach a response")

    with pytest.raises(AssertionError, match="must not reach"):
        asyncio.run(
            PydanticAICalibratedPortfolioSelectionPolicy(
                generate_once=invalid_runner,
                binding_for=lambda _: binding,
            ).select(request)
        )


def test_request_keyed_witness_rejects_an_unsealed_ordering_key() -> None:
    request = _boils_request(
        DeterministicIdFactory("calibrated_bad_witness_key"),
        require_disjoint=True,
    )
    with pytest.raises(ValueError, match="ordering_key_sha256"):
        pairwise_disjoint_parent_patch_witness(
            request.finite_variation_contract,
            tuple(
                option.option_id for option in request.finite_variation_contract.options
            ),
            portfolio_size=request.portfolio_size,
            ordering_key_sha256="not-a-sha",
        )


def test_model_anchored_selector_coordinator_and_strict_replay_share_allocator() -> (
    None
):
    async def scenario():
        ids = DeterministicIdFactory("model_anchored_calibrated_e2e")
        problem = _CountingBoilsProblem()
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=_NoCandidateGenerator(),
            id_factory=ids,
            memory=None,
            seed=19,
            evaluator_concurrency=4,
        )
        parent = await engine.register_seed(
            {"sequence": list(DEFAULT_ACTION_SEQUENCE)},
            label="parent",
        )
        request = _boils_request(ids, require_disjoint=True)
        binding = _model_anchored_binding(request)
        runner = _ProviderFreeSlateRunner(
            _proposal_option_ids(request.finite_variation_contract)
        )
        allocator = ModelAnchoredCalibratedSlatePolicy(model_anchor_count=3)
        coordinator = CalibratedPortfolioCampaignCoordinator(allocator=allocator)
        coordinator.register(request, binding)
        selector = coordinator.build_selector(runner)
        assert type(selector) is (
            PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy
        )
        wave = PortfolioVariationWaveRequest(
            selection_request=request,
            parent=parent,
            generation=2,
            label_prefix="model_anchored_calibrated",
        )
        result = await PortfolioEvolution(
            engine=engine,
            selector=selector,
            ids=ids,
        ).run(wave)
        return request, binding, runner, allocator, coordinator, result

    request, binding, runner, allocator, coordinator, result = asyncio.run(scenario())
    assert runner.calls == 1
    assert runner.requests[0].prompt == coordinator.render(request)
    assert result.selection_decision is not None
    assert result.selection_decision.policy_id == (
        MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID
    )
    assert result.selection_decision.policy_version == (
        MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    assert result.selection_decision.policy_definition_sha256 == (
        MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )
    assert (
        PydanticAICalibratedPortfolioSelectionPolicy.policy_definition_sha256
        != PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy.policy_definition_sha256
    )
    audit = result.supplemental_selection_audit
    assert audit is not None
    assert audit.audit_kind == "model_anchored_calibrated_portfolio_k8_to_k4"
    payload = thaw_json(audit.payload)
    assert payload["allocator_policy"] == allocator.to_record()
    assert payload["policy_id"] != CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID
    assert len(payload["composition_identity_sha256"]) == 64

    proposal = decode_calibrated_portfolio_proposal(
        audit,
        request=request,
        binding=binding,
    )
    assert len(proposal.model_members) == 8
    assert tuple(value.model_rank for value in proposal.model_members) == tuple(
        range(1, 9)
    )
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert type(decoded.allocation) is ModelAnchoredSlateDecision
    assert decoded.slate == proposal.slate
    assert len(decoded.allocation.selected) == 4
    assert len(decoded.selected_prediction_receipts) == 8
    assert coordinator.decode_selected_predictions(result) == (
        decoded.selected_prediction_receipts
    )

    with pytest.raises(ValueError, match="exact typed replay"):
        decode_calibrated_portfolio_audit(
            audit,
            request=request,
            binding=binding,
            allocator=ModelAnchoredCalibratedSlatePolicy(model_anchor_count=2),
        )
    wrong_profile = CalibratedPortfolioCampaignCoordinator()
    wrong_profile.register(request, binding)
    with pytest.raises(ValueError, match="foreign allocator profile"):
        wrong_profile.decode_selected_predictions(result)


def test_structural_posterior_selector_has_authenticated_generic_round_trip() -> None:
    ids = DeterministicIdFactory("structural_posterior_calibrated_e2e")
    request = _boils_request(ids, require_disjoint=True)
    binding = _structural_posterior_binding(request)
    runner = _ProviderFreeSlateRunner(
        _proposal_option_ids(request.finite_variation_contract)
    )
    allocator = StructuralPosteriorSlatePolicy()
    coordinator = CalibratedPortfolioCampaignCoordinator(allocator=allocator)
    coordinator.register(request, binding)
    selector = coordinator.build_selector(runner)

    assert type(selector) is (
        PydanticAIStructuralPosteriorCalibratedPortfolioSelectionPolicy
    )
    result = asyncio.run(selector.select(request))

    assert runner.calls == 1
    assert result.decision.policy_id == (
        STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID
    )
    assert result.decision.policy_version == (
        STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    audit = result.supplemental_audit
    assert audit is not None
    assert audit.audit_kind == ("structural_posterior_calibrated_portfolio_k8_to_k4")
    payload = thaw_json(audit.payload)
    assert payload["allocator_policy"] == allocator.to_record()
    assert len(payload["composition_identity_sha256"]) == 64
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert type(decoded.allocation) is StructuralPosteriorSlateDecision
    assert len(decoded.allocation.selected) == 4
    assert len(decoded.selected_prediction_receipts) == 8


def test_frontier_probe_selector_has_authenticated_generic_round_trip() -> None:
    ids = DeterministicIdFactory("frontier_probe_calibrated_e2e")
    request = _boils_request(ids, require_disjoint=False)
    binding = _frontier_probe_binding(request)
    runner = _ProviderFreeSlateRunner(
        _proposal_option_ids(request.finite_variation_contract)
    )
    allocator = FrontierProbeSlatePolicy()
    coordinator = CalibratedPortfolioCampaignCoordinator(allocator=allocator)
    coordinator.register(request, binding)
    selector = coordinator.build_selector(runner)

    assert type(selector) is PydanticAIFrontierProbeCalibratedPortfolioSelectionPolicy
    result = asyncio.run(selector.select(request))

    assert runner.calls == 1
    assert result.decision.policy_id == (
        FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID
    )
    assert result.decision.policy_version == (
        FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    assert [value.option_id for value in result.decision.members] == list(
        _proposal_option_ids(request.finite_variation_contract)[:4]
    )
    audit = result.supplemental_audit
    assert audit is not None
    assert audit.audit_kind == "frontier_probe_calibrated_portfolio_k8_to_k4"
    payload = thaw_json(audit.payload)
    assert payload["allocator_policy"] == allocator.to_record()
    assert (
        payload["allocation"]["constraint_projection"]["source_request_sha256"]
        == payload["allocation"]["constraint_projection"]["projected_request_sha256"]
    )
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert type(decoded.allocation) is FrontierProbeSlateDecision
    assert decoded.allocation.ideal_target_feasible is True
    assert decoded.allocation.selected_probe_option_id is None
    assert len(decoded.selected_prediction_receipts) == 8


def test_frontier_probe_live_adapter_rejects_mating_constraints_on_evaluation() -> None:
    ids = DeterministicIdFactory("frontier_probe_mating_constraint_rejection")
    request = _boils_request(ids, require_disjoint=True)
    binding = _frontier_probe_binding(request)
    runner = _ProviderFreeSlateRunner(
        _proposal_option_ids(request.finite_variation_contract)
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=FrontierProbeSlatePolicy()
    )
    coordinator.register(request, binding)

    with pytest.raises(
        ValueError,
        match="leave family and pairwise patch constraints",
    ):
        asyncio.run(coordinator.build_selector(runner).select(request))


def test_campaign_coordinator_revalidates_prior_only_binding_on_registration() -> None:
    ids = DeterministicIdFactory("calibrated_campaign_current_wave_leak")
    request = _boils_request(ids, require_disjoint=True)
    binding = _binding(request)
    leaked_snapshot = ForecastCalibrationSnapshot(
        scope=binding.context.scope,
        cutoff_wave_index_exclusive=binding.context.wave_index + 1,
        observations=(),
    )
    # Simulate a corrupted/deserialized object crossing the application seam;
    # the coordinator must not trust the fact that it was once constructed.
    object.__setattr__(
        binding.context,
        "calibration_snapshot",
        leaked_snapshot,
    )
    coordinator = CalibratedPortfolioCampaignCoordinator()
    with pytest.raises(ValueError, match="current/future-wave"):
        coordinator.register(request, binding)
    assert coordinator.registered_request_count == 0


def test_current_wave_snapshot_is_rejected_before_low_level_call() -> None:
    ids = DeterministicIdFactory("calibrated_current_wave_leak")
    request = _boils_request(ids, require_disjoint=True)
    runner = _ProviderFreeSlateRunner(
        _proposal_option_ids(request.finite_variation_contract)
    )

    def leaked_binding(
        current: PortfolioSelectionRequest,
    ) -> CalibratedPortfolioInputBinding:
        prior = _context(current)
        return replace(
            _binding(current),
            context=replace(
                prior,
                calibration_snapshot=ForecastCalibrationSnapshot(
                    scope=prior.scope,
                    cutoff_wave_index_exclusive=3,
                    observations=(),
                ),
            ),
        )

    selector = PydanticAICalibratedPortfolioSelectionPolicy(
        generate_once=runner,
        binding_for=leaked_binding,
    )
    with pytest.raises(ValueError, match="current/future-wave"):
        asyncio.run(selector.select(request))
    assert runner.calls == 0


def test_proposal_must_administer_every_prospectively_assigned_card() -> None:
    ids = DeterministicIdFactory("calibrated_card_administration")
    request = _boils_request(ids, require_disjoint=True)
    runner = _ProviderFreeSlateRunner(
        _proposal_option_ids(request.finite_variation_contract),
        omit_second_card=True,
    )
    selector = PydanticAICalibratedPortfolioSelectionPolicy(
        generate_once=runner,
        binding_for=_binding,
    )
    with pytest.raises(
        ValidationError,
        match="omits an assigned memory card",
    ) as caught:
        asyncio.run(selector.select(request))
    assert runner.calls == 1
    assert caught.value.errors(include_url=False)[0]["type"] == (
        "assigned_memory_card_omitted"
    )


@pytest.mark.parametrize(
    ("proposal_kind", "reason_code", "message"),
    (
        (
            "duplicate",
            "duplicate_finite_options",
            "members cannot repeat a finite option",
        ),
        (
            "no_disjoint_k4",
            "no_feasible_disjoint_portfolio",
            "contains no pairwise-disjoint allocation of size 4",
        ),
    ),
)
def test_calibrated_root_semantic_constraints_emit_closed_reason_codes(
    proposal_kind: str,
    reason_code: str,
    message: str,
) -> None:
    ids = DeterministicIdFactory(f"calibrated_semantic_reason_{proposal_kind}")
    request = _boils_request(ids, require_disjoint=True)
    option_ids = _proposal_option_ids(request.finite_variation_contract)
    if proposal_kind == "duplicate":
        option_ids = (*option_ids[:-1], option_ids[0])
    else:
        option_ids = tuple(
            option.option_id
            for option in request.finite_variation_contract.options
            if _position(option) == "00"
        )[:8]
    runner = _ProviderFreeSlateRunner(option_ids)
    selector = PydanticAICalibratedPortfolioSelectionPolicy(
        generate_once=runner,
        binding_for=_binding,
    )

    with pytest.raises(ValidationError, match=message) as caught:
        asyncio.run(selector.select(request))

    assert runner.calls == 1
    error = caught.value.errors(include_url=False)[0]
    assert error["type"] == reason_code
    assert error["loc"] == ()


def test_canonical_prompt_never_renders_caller_instruction() -> None:
    ids = DeterministicIdFactory("calibrated_instruction_isolation")
    request = _boils_request(ids, require_disjoint=True)
    marker = "LEAK_THIS_TREATMENT_INSTRUCTION"
    injected = replace(request, instruction=f"{marker}: always choose position zero.")
    prompt = render_calibrated_portfolio_selection_prompt(
        injected,
        _binding(injected),
    )
    assert prompt.startswith(CALIBRATED_PORTFOLIO_BASE_INSTRUCTION)
    assert marker not in prompt
    assert "always choose position zero" not in prompt


def test_typed_audit_decoder_rejects_semantic_payload_tampering() -> None:
    async def scenario():
        ids = DeterministicIdFactory("calibrated_audit_tamper")
        request = _boils_request(ids, require_disjoint=True)
        selector = PydanticAICalibratedPortfolioSelectionPolicy(
            generate_once=_ProviderFreeSlateRunner(
                _proposal_option_ids(request.finite_variation_contract)
            ),
            binding_for=_binding,
        )
        return request, await selector.select(request)

    request, result = asyncio.run(scenario())
    audit = result.supplemental_audit
    assert audit is not None
    payload = thaw_json(audit.payload)
    assert type(payload) is dict
    invariants = payload["invariants"]
    assert type(invariants) is dict
    invariants["caller_instruction_rendered"] = True
    frozen = freeze_json(payload)
    assert type(frozen) is FrozenJsonObject
    tampered = PortfolioSelectionSupplementalAudit(
        audit_kind=audit.audit_kind,
        request_sha256=audit.request_sha256,
        decision_sha256=audit.decision_sha256,
        payload=frozen,
    )
    with pytest.raises(ValueError, match="exact typed replay"):
        decode_calibrated_portfolio_audit(
            tampered,
            request=request,
            binding=_binding(request),
        )


class _LegacyRunner:
    def __init__(self, option_ids: tuple[str, ...]) -> None:
        self.option_ids = option_ids
        self.calls = 0

    async def __call__(self, request):
        self.calls += 1
        value = request.output_type.model_validate(
            {
                "members": [
                    {
                        "option_id": option_id,
                        "supporting_card_keys": ["card.a" if index % 2 else "card.b"],
                        "effect_predictions": [
                            {
                                "metric_id": "total_levels",
                                "direction": "decrease",
                            },
                            {
                                "metric_id": "total_lut_count",
                                "direction": "decrease",
                            },
                        ],
                        "design_rationale": f"Legacy rationale {index}.",
                    }
                    for index, option_id in enumerate(self.option_ids, start=1)
                ]
            },
            strict=True,
        )
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider-free/legacy",
            resolved_model="provider-free/legacy",
            resolved_provider="provider-free",
            provider_response_id="provider-free-legacy-response",
            finish_reason="stop",
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1,
        )


class _ProviderFreeHierarchicalSlateRunner:
    def __init__(self, contract: FiniteVariationContract) -> None:
        self.contract = contract
        self.calls = 0
        composites = tuple(
            value for value in contract.options if value.family == "composite_r2"
        )
        if len(composites) < 2:
            raise ValueError("hierarchical fixture requires two composites")
        by_position: dict[str, FiniteVariationOption] = {}
        for option in contract.options:
            position = dict(option.metadata).get("position")
            if position is not None and position not in by_position:
                by_position[position] = option
        self.ordered = (
            composites[0],
            by_position["00"],
            by_position["01"],
            composites[1],
            by_position["02"],
            by_position["03"],
            by_position["04"],
            by_position["05"],
        )

    async def __call__(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        self.calls += 1
        members = []
        for rank, option in enumerate(self.ordered, start=1):
            common = {
                "supporting_card_keys": ["card.a" if rank % 2 else "card.b"],
                "effect_predictions": [
                    {
                        "metric_id": "total_levels",
                        "direction": "decrease",
                        "confidence": "medium",
                    },
                    {
                        "metric_id": "total_lut_count",
                        "direction": "decrease",
                        "confidence": "medium",
                    },
                ],
                "role_proposal": "exploit" if rank <= 4 else "coverage",
                "design_rationale": f"Hierarchical provider-free rank {rank}.",
            }
            if option.family == "composite_r2":
                metadata = dict(option.metadata)
                members.append(
                    {
                        "action_kind": "compose_r2",
                        "composite_option_id": option.option_id,
                        "component_option_ids": sorted(
                            [
                                metadata["left_option_id"],
                                metadata["right_option_id"],
                            ]
                        ),
                        **common,
                    }
                )
            else:
                members.append(
                    {
                        "action_kind": "atomic",
                        "option_id": option.option_id,
                        **common,
                    }
                )
        value = request.output_type.model_validate({"members": members}, strict=True)
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider-free/hierarchical",
            resolved_model="provider-free/hierarchical",
            resolved_provider="provider-free",
            provider_response_id="provider-free-hierarchical-response",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=100,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1,
        )


class _ProviderFreeHierarchicalAtomicSlateRunner:
    """Emit eight authenticated atomic actions for decoupled reconciliation."""

    def __init__(
        self,
        contract: FiniteVariationContract,
        option_ids: tuple[str, ...],
    ) -> None:
        self.contract = contract
        self.option_ids = option_ids
        self.calls = 0
        if len(option_ids) != 8:
            raise ValueError("hierarchical atomic fixture requires eight options")
        for option_id in option_ids:
            if contract.resolve(option_id).family == "composite_r2":
                raise ValueError("hierarchical atomic fixture received a composite")

    async def __call__(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        self.calls += 1
        members = [
            {
                "action_kind": "atomic",
                "option_id": option_id,
                "supporting_card_keys": ["card.a" if rank % 2 else "card.b"],
                "effect_predictions": [
                    {
                        "metric_id": "total_levels",
                        "direction": "decrease",
                        "confidence": "medium",
                    },
                    {
                        "metric_id": "total_lut_count",
                        "direction": "decrease",
                        "confidence": "medium",
                    },
                ],
                "role_proposal": "exploit" if rank <= 4 else "coverage",
                "design_rationale": (f"Hierarchical atomic provider-free rank {rank}."),
            }
            for rank, option_id in enumerate(self.option_ids, start=1)
        ]
        value = request.output_type.model_validate({"members": members}, strict=True)
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider-free/hierarchical-atomic",
            resolved_model="provider-free/hierarchical-atomic",
            resolved_provider="provider-free",
            provider_response_id="provider-free-hierarchical-atomic-response",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=100,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1,
        )


def _hierarchical_boils_request_and_binding():
    ids = DeterministicIdFactory("hierarchical_calibrated_provider_free")
    catalog = BoundedCompositionalFiniteVariationCatalog(
        BoilsFiniteVariationCatalog(),
        max_composite_options=8,
        selection_exposure=(CompositionSelectionExposure.HIERARCHICAL_RANKED_UNION),
        required_composite_proposals=2,
    )
    contract = bind_finite_variation_catalog(
        catalog,
        _frozen({"sequence": list(DEFAULT_ACTION_SEQUENCE)}),
    )
    request = replace(
        _boils_request(ids, require_disjoint=True),
        finite_variation_contract=contract,
    )
    legacy_context = _context(request)
    prompt_definition = calibrated_portfolio_prompt_definition_sha256(
        finite_variation_contract=contract,
    )
    scope = replace(
        legacy_context.scope,
        prompt_definition_sha256=prompt_definition,
    )
    context = replace(
        legacy_context,
        scope=scope,
        calibration_snapshot=replace(
            legacy_context.calibration_snapshot,
            scope=scope,
        ),
    )
    evidence = tuple(
        CalibratedPortfolioOptionEvidence(
            option_id=option.option_id,
            option_identity_sha256=option.identity_sha256,
            locus_key=f"locus.option_{index:04d}",
            phenotype_identity_sha256=option.child_configuration_sha256,
            structural_evidence=SlateStructuralEvidence(
                frozen_archive_snapshot_sha256=_sha("hierarchical frozen archive"),
                evidence_receipt_sha256=_sha(
                    f"hierarchical evidence {option.option_id}"
                ),
                archive_novelty_score=0.5,
                structural_coverage_score=0.5,
            ),
        )
        for index, option in enumerate(contract.options, start=1)
    )
    binding = CalibratedPortfolioInputBinding(
        request_sha256=request.request_sha256,
        context=context,
        option_evidence=tuple(sorted(evidence, key=lambda value: value.option_id)),
    )
    return request, binding


def test_hierarchical_ranked_union_resolves_exactly_two_engine_composites() -> None:
    request, binding = _hierarchical_boils_request_and_binding()
    runner = _ProviderFreeHierarchicalSlateRunner(request.finite_variation_contract)

    result = asyncio.run(
        PydanticAICalibratedPortfolioSelectionPolicy(
            runner,
            lambda _: binding,
        ).select(request)
    )

    assert runner.calls == 1
    audit = result.supplemental_audit
    assert audit is not None
    payload = thaw_json(audit.payload)
    assert type(payload) is dict
    proposals = payload["original_k8_response"]["members"]
    assert (
        sum(
            value.get("hierarchical_action", {}).get("action_kind") == "compose_r2"
            for value in proposals
        )
        == 2
    )
    prompt = render_calibrated_portfolio_selection_prompt(request, binding)
    assert '"required_composite_proposals":2' in prompt
    assert "action_kind=compose_r2" in prompt
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
    )
    assert len(decoded.slate.members) == 8
    variation_trace = summarize_campaign_variation_trace(
        (result,),
        required_composite_proposals=2,
    ).to_record()
    assert variation_trace["proposal_action_kind_counts"] == {
        "atomic": 6,
        "compose_r2": 2,
    }
    assert variation_trace["evaluated_member_count"] == 4
    assert variation_trace["exact_required_composite_call_rate"] == 1.0
    assert variation_trace["effective_composite_proposal_count_histogram"] == {"2": 1}
    assert canonical_json_bytes(variation_trace)

    option_by_id = {
        option.option_id: option for option in request.finite_variation_contract.options
    }
    proposal_contract = replace(
        request.finite_variation_contract,
        options=tuple(option_by_id[value["option_id"]] for value in proposals),
    )
    engine_topology = project_finite_contract_proposal_topology(
        source_contract=request.finite_variation_contract,
        proposal_contract=proposal_contract,
    )
    engine_audit = replace(
        audit,
        payload=freeze_json({"proposal_topology": engine_topology}),
    )
    engine_result = replace(result, supplemental_audit=engine_audit)
    engine_trace = summarize_campaign_variation_trace(
        (engine_result,),
        required_composite_proposals=2,
    ).to_record()
    assert engine_trace["proposal_action_kind_counts"] == {
        "atomic": 6,
        "compose_r2": 2,
    }
    assert engine_trace["calls"][0]["proposal_provenance"] == (
        "engine_authenticated_finite_contract"
    )
    assert engine_trace["model_original_proposal_call_count"] == 0
    assert engine_trace["engine_authenticated_proposal_call_count"] == 1
    assert engine_trace["required_composite_capacity_available_rate"] == 1.0


def test_variation_trace_accepts_arbitrary_width_engine_screen() -> None:
    trace = CampaignVariationTraceSummary(
        selector_call_count=1,
        proposal_member_count=64,
        evaluated_member_count=4,
        proposal_action_kind_counts=(("atomic", 48), ("compose_r2", 16)),
        evaluated_action_kind_counts=(("atomic", 4),),
        hierarchical_call_count=1,
        model_original_proposal_call_count=0,
        engine_authenticated_proposal_call_count=1,
        required_composite_proposals=2,
        exact_required_composite_call_count=0,
        required_composite_capacity_available_call_count=1,
        capacity_projected_call_count=0,
        effective_composite_proposal_count_histogram=((16, 1),),
        calls=({"capacity_projected": False},),
    ).to_record()

    assert trace["schema_version"] == 4
    assert trace["effective_composite_proposal_count_histogram"] == {"16": 1}
    assert trace["composite_proposal_count"] == 16
    assert trace["exact_required_composite_call_rate"] is None
    assert trace["required_composite_capacity_available_rate"] == 1.0


def test_source_mix_replay_honors_authenticated_deferred_proposal_support() -> None:
    """Regression for the shared-N10 post-G1 replay failure.

    A protected global evaluation witness can already consume the exact
    hierarchical-composition capacity.  V11 then records one structurally
    reserved composite as explicitly deferred.  Typed replay must authenticate
    that deterministic receipt instead of reapplying the provider-facing hard
    reservation to the reconciled K8.
    """

    from agent_evolve.policies.selection.calibrated_portfolio_binding import (
        proposal_support_candidates,
    )
    from agent_evolve.policies.selection.proposal_support import (
        StructuralProposalSupportPolicy,
    )

    request, legacy_binding = _hierarchical_boils_request_and_binding()
    contract = request.finite_variation_contract
    model_option_ids = tuple(
        value.option_id for value in contract.options if value.family != "composite_r2"
    )[:8]
    runner = _ProviderFreeHierarchicalAtomicSlateRunner(
        contract,
        model_option_ids,
    )
    target_composite = next(
        value.option_id
        for value in reversed(contract.options)
        if value.family == "composite_r2"
    )
    pool = TaskKeyedCommonCandidatePoolPolicy(
        replicate_seed=20260722,
        candidate_pool_size=None,
    ).select(
        benchmark_sha256=legacy_binding.context.scope.benchmark_sha256,
        wave_index=1,
        parent_configuration_sha256=(
            request.finite_variation_contract.parent_configuration_sha256
        ),
        contract=request.finite_variation_contract,
        evaluation_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=True,
        required_option_ids=common_pool_required_option_ids(request),
    )
    evidence = tuple(
        replace(
            value,
            structural_evidence=replace(
                value.structural_evidence,
                archive_novelty_score=(
                    1.0 if value.option_id == target_composite else 0.5
                ),
                structural_coverage_score=(
                    1.0 if value.option_id == target_composite else 0.5
                ),
            ),
        )
        for value in legacy_binding.option_evidence
    )
    support = StructuralProposalSupportPolicy().select(
        request_sha256=request.request_sha256,
        common_candidate_pool_decision_sha256=pool.decision_sha256,
        model_selection_size=pool.model_selection_size,
        candidates=proposal_support_candidates(request, evidence, pool),
    )
    assert target_composite in support.required_option_ids

    scope = replace(
        legacy_binding.context.scope,
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            proposal_support=True,
            finite_variation_contract=request.finite_variation_contract,
            hierarchical_composition_required_proposals=2,
            feasibility_witness_mode=(
                CalibratedPortfolioFeasibilityWitnessMode.TASK_KEYED_COMMON_POOL
            ),
            constraint_decoupled=True,
        ),
        selector_policy_definition_sha256=(
            EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = CalibratedPortfolioInputBinding(
        request_sha256=request.request_sha256,
        context=replace(
            legacy_binding.context,
            wave_index=1,
            scope=scope,
            calibration_snapshot=replace(
                legacy_binding.context.calibration_snapshot,
                scope=scope,
                cutoff_wave_index_exclusive=1,
            ),
        ),
        option_evidence=evidence,
        common_candidate_pool=pool,
        proposal_support=support,
    )
    allocator = HorizonBoundedStructuralPosteriorSlatePolicy(
        build_terminal_tapered_family_exposure_phases(
            family="composite_r2",
            terminal_wave_index=5,
        )
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=allocator,
        constraint_decoupled=True,
        minimum_intervention_projection=True,
        evidence_calibrated_source_mix=True,
    )
    coordinator.register(request, binding)
    result = asyncio.run(coordinator.build_selector(runner).select(request))

    audit = result.supplemental_audit
    assert audit is not None
    payload = thaw_json(audit.payload)
    assert type(payload) is dict
    assert payload["semantic_reconciliation"][
        "deferred_proposal_support_option_ids"
    ] == [target_composite]
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert decoded.slate.to_record() == payload["calibrated_slate"]


def test_contextual_search_allocation_enforces_exact_source_operator_slice() -> None:
    request, legacy_binding = _hierarchical_boils_request_and_binding()
    contract = request.finite_variation_contract
    model_atom_by_position: dict[str, str] = {}
    for option in contract.options:
        position = dict(option.metadata).get("position")
        if position is not None:
            model_atom_by_position.setdefault(position, option.option_id)
    model_option_ids = tuple(
        model_atom_by_position[position]
        for position in sorted(model_atom_by_position)[:8]
    )
    runner = _ProviderFreeHierarchicalAtomicSlateRunner(
        contract,
        model_option_ids,
    )
    pool = TaskKeyedCommonCandidatePoolPolicy(
        replicate_seed=20260722,
        candidate_pool_size=None,
    ).select(
        benchmark_sha256=legacy_binding.context.scope.benchmark_sha256,
        wave_index=1,
        parent_configuration_sha256=contract.parent_configuration_sha256,
        contract=contract,
        evaluation_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=True,
        required_option_ids=common_pool_required_option_ids(request),
    )
    allocation = ContextualPortfolioAllocationContract(
        campaign_scope_sha256=_sha("contextual campaign scope"),
        query_sha256=_sha("contextual allocation query"),
        decision_sha256=_sha("contextual allocation decision"),
        campaign_generation=1,
        controller_wave_index=1,
        phase_id="basin_acquisition",
        slice_id="elite_01",
        evaluation_slots=request.portfolio_size,
        source_target_counts=(("engine", 2), ("model", 2)),
        operator_target_counts=(("atomic", 2), ("composite", 2)),
        minimum_single_path_interventions=1,
        minimum_disjoint_parent_patch_pairs=2,
    )
    scope = replace(
        legacy_binding.context.scope,
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            finite_variation_contract=contract,
            feasibility_witness_mode=(
                CalibratedPortfolioFeasibilityWitnessMode.TASK_KEYED_COMMON_POOL
            ),
            constraint_decoupled=True,
        ),
        selector_policy_definition_sha256=(
            CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = CalibratedPortfolioInputBinding(
        request_sha256=request.request_sha256,
        context=replace(
            legacy_binding.context,
            wave_index=1,
            scope=scope,
            calibration_snapshot=replace(
                legacy_binding.context.calibration_snapshot,
                scope=scope,
                cutoff_wave_index_exclusive=1,
            ),
        ),
        option_evidence=legacy_binding.option_evidence,
        common_candidate_pool=pool,
        contextual_allocation=allocation,
    )
    allocator = HorizonBoundedStructuralPosteriorSlatePolicy(
        build_terminal_tapered_family_exposure_phases(
            family="composite_r2",
            terminal_wave_index=5,
        )
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=allocator,
        constraint_decoupled=True,
        minimum_intervention_projection=True,
        evidence_calibrated_source_mix=True,
        contextual_search_allocation=True,
    )
    coordinator.register(request, binding)
    selector = coordinator.build_selector(runner)
    assert type(selector) is (
        PydanticAIContextualSearchAllocationPortfolioSelectionPolicy
    )
    result = asyncio.run(selector.select(request))

    audit = result.supplemental_audit
    assert audit is not None
    payload = thaw_json(audit.payload)
    assert type(payload) is dict
    reconciliation = payload["semantic_reconciliation"]
    assert reconciliation["policy_id"] == (
        "semantic_slate_contextual_search_allocation_reconciliation"
    )
    assert reconciliation["contextual_allocation_contract_sha256"] == (
        allocation.contract_sha256
    )
    projection = reconciliation["contextual_allocation_projection"]
    assert projection["exact"] is True
    assert projection["source_l1_deviation"] == 0
    assert projection["operator_l1_deviation"] == 0
    assert projection["minimum_single_path_interventions"] == 1
    assert projection["realized_single_path_interventions"] >= 1
    assert projection["minimum_disjoint_parent_patch_pairs"] == 2
    assert projection["realized_disjoint_parent_patch_pairs"] >= 2
    assert projection["requested_source_target_counts"] == [
        list(value) for value in allocation.source_target_counts
    ]
    assert projection["realized_source_target_counts"] == [
        list(value) for value in allocation.source_target_counts
    ]
    composition_projection = reconciliation["composition_capacity_projection"]
    assert composition_projection["policy_id"] == (
        "nearest_exact_k_binary_composition_capacity_projection"
    )
    assert composition_projection["proposal_size"] == pool.model_selection_size
    assert composition_projection["preferred_composite_count"] == 2
    assert composition_projection["effective_composite_count"] == 2
    assert composition_projection["capacity_projected"] is False
    assert composition_projection["objective_values_consulted"] is False
    assert (
        composition_projection[
            "workload_model_provider_identifiers_consulted"
        ]
        is False
    )
    selected_ids = tuple(value.option_id for value in result.decision.members)
    reconciled_by_id = {
        value["option_id"]: value for value in reconciliation["members"]
    }
    observed_sources = {
        "model": sum(
            reconciled_by_id[option_id]["origin"] == "model"
            for option_id in selected_ids
        ),
        "engine": sum(
            reconciled_by_id[option_id]["origin"] != "model"
            for option_id in selected_ids
        ),
    }
    observed_operators = {
        "composite": sum(
            contract.resolve(option_id).family == "composite_r2"
            for option_id in selected_ids
        ),
        "atomic": sum(
            contract.resolve(option_id).family != "composite_r2"
            for option_id in selected_ids
        ),
    }
    assert observed_sources == dict(allocation.source_target_counts)
    assert observed_operators == dict(allocation.operator_target_counts)
    assert set(reconciliation["contextual_allocation_option_ids"]) == set(selected_ids)
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert decoded.slate.to_record() == payload["calibrated_slate"]
    exact_realization = coordinator.decode_contextual_allocation_realization(result)
    assert exact_realization is not None
    assert exact_realization.exact is True
    assert exact_realization.contract_sha256 == allocation.contract_sha256
    assert exact_realization.requested_source_target_counts == (
        allocation.source_target_counts
    )
    assert exact_realization.realized_operator_target_counts == (
        allocation.operator_target_counts
    )
    assert exact_realization.realized_single_path_interventions >= 1
    assert exact_realization.requested_minimum_disjoint_parent_patch_pairs == 2
    assert exact_realization.realized_disjoint_parent_patch_pairs >= 2

    impossible = ContextualPortfolioAllocationContract(
        campaign_scope_sha256=allocation.campaign_scope_sha256,
        query_sha256=_sha("contextual recourse query"),
        decision_sha256=_sha("contextual recourse decision"),
        campaign_generation=allocation.campaign_generation,
        controller_wave_index=allocation.controller_wave_index,
        phase_id=allocation.phase_id,
        slice_id=allocation.slice_id,
        evaluation_slots=allocation.evaluation_slots,
        source_target_counts=(("engine", 0), ("model", 4)),
        operator_target_counts=(("atomic", 0), ("composite", 4)),
    )
    recourse_binding = replace(binding, contextual_allocation=impossible)
    recourse_coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=allocator,
        constraint_decoupled=True,
        minimum_intervention_projection=True,
        evidence_calibrated_source_mix=True,
        contextual_search_allocation=True,
    )
    recourse_coordinator.register(request, recourse_binding)
    recourse_result = asyncio.run(
        recourse_coordinator.build_selector(
            _ProviderFreeHierarchicalAtomicSlateRunner(
                contract,
                model_option_ids,
            )
        ).select(request)
    )
    recourse_audit = recourse_result.supplemental_audit
    assert recourse_audit is not None
    recourse_payload = thaw_json(recourse_audit.payload)
    assert type(recourse_payload) is dict
    recourse = recourse_payload["semantic_reconciliation"]
    recourse_projection = recourse["contextual_allocation_projection"]
    assert recourse_projection["exact"] is False
    assert (
        recourse_projection["source_l1_deviation"]
        + recourse_projection["operator_l1_deviation"]
        > 0
    )
    recourse_selected = tuple(
        value.option_id for value in recourse_result.decision.members
    )
    recourse_by_id = {value["option_id"]: value for value in recourse["members"]}
    assert dict(recourse_projection["realized_source_target_counts"]) == {
        "model": sum(
            recourse_by_id[option_id]["origin"] == "model"
            for option_id in recourse_selected
        ),
        "engine": sum(
            recourse_by_id[option_id]["origin"] != "model"
            for option_id in recourse_selected
        ),
    }
    assert dict(recourse_projection["realized_operator_target_counts"]) == {
        "composite": sum(
            contract.resolve(option_id).family == "composite_r2"
            for option_id in recourse_selected
        ),
        "atomic": sum(
            contract.resolve(option_id).family != "composite_r2"
            for option_id in recourse_selected
        ),
    }
    recourse_realization = (
        recourse_coordinator.decode_contextual_allocation_realization(recourse_result)
    )
    assert recourse_realization is not None
    assert recourse_realization.exact is False
    assert recourse_realization.contract_sha256 == impossible.contract_sha256
    assert (
        recourse_realization.source_l1_deviation
        == (recourse_projection["source_l1_deviation"])
    )
    assert (
        recourse_realization.operator_l1_deviation
        == (recourse_projection["operator_l1_deviation"])
    )


def test_contextual_allocation_uses_variation_source_not_engine_origin() -> None:
    ids = DeterministicIdFactory("contextual_variation_source_provider_free")
    request = _boils_source_union_request(ids)
    legacy_binding = _binding(request)
    contract = request.finite_variation_contract
    source_by_option = finite_variation_source_by_option(contract)
    pool = TaskKeyedCommonCandidatePoolPolicy(
        replicate_seed=20260722,
        candidate_pool_size=16,
    ).select(
        benchmark_sha256=legacy_binding.context.scope.benchmark_sha256,
        wave_index=1,
        parent_configuration_sha256=contract.parent_configuration_sha256,
        contract=contract,
        evaluation_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=False,
        required_option_ids=common_pool_required_option_ids(request),
    )
    model_option_ids = tuple(
        option_id
        for option_id in pool.option_ids
        if source_by_option[option_id] == PRIMARY_VARIATION_SOURCE_ID
    )[:8]
    assert len(model_option_ids) == 8
    runner = _ProviderFreeSlateRunner(model_option_ids)
    allocation = ContextualPortfolioAllocationContract(
        campaign_scope_sha256=_sha("variation-source campaign"),
        query_sha256=_sha("variation-source query"),
        decision_sha256=_sha("variation-source decision"),
        campaign_generation=1,
        controller_wave_index=1,
        phase_id="basin_acquisition",
        slice_id="elite",
        evaluation_slots=request.portfolio_size,
        source_target_counts=(("global_restart", 2), ("primary", 2)),
        operator_target_counts=(
            ("atomic", 2),
            ("composite", 0),
            ("global", 2),
        ),
    )
    scope = replace(
        legacy_binding.context.scope,
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            finite_variation_contract=contract,
            feasibility_witness_mode=(
                CalibratedPortfolioFeasibilityWitnessMode.TASK_KEYED_COMMON_POOL
            ),
            constraint_decoupled=True,
        ),
        selector_policy_definition_sha256=(
            CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = replace(
        legacy_binding,
        context=replace(
            legacy_binding.context,
            wave_index=1,
            scope=scope,
            calibration_snapshot=replace(
                legacy_binding.context.calibration_snapshot,
                scope=scope,
                cutoff_wave_index_exclusive=1,
            ),
        ),
        common_candidate_pool=pool,
        contextual_allocation=allocation,
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=HorizonBoundedStructuralPosteriorSlatePolicy(
            build_controller_owned_family_exposure_phases(
                family="composite_r2",
            )
        ),
        constraint_decoupled=True,
        minimum_intervention_projection=True,
        evidence_calibrated_source_mix=True,
        contextual_search_allocation=True,
    )
    coordinator.register(request, binding)
    result = asyncio.run(coordinator.build_selector(runner).select(request))

    audit = result.supplemental_audit
    assert audit is not None
    payload = thaw_json(audit.payload)
    assert type(payload) is dict
    reconciliation = payload["semantic_reconciliation"]
    projection = reconciliation["contextual_allocation_projection"]
    assert projection["exact"] is True
    assert dict(projection["realized_source_target_counts"]) == {
        "global_restart": 2,
        "primary": 2,
    }
    assert dict(projection["realized_operator_target_counts"]) == {
        "atomic": 2,
        "composite": 0,
        "global": 2,
    }
    selected_ids = tuple(value.option_id for value in result.decision.members)
    assert {
        source_id: sum(source_by_option[option_id] == source_id for option_id in selected_ids)
        for source_id in ("global_restart", "primary")
    } == {"global_restart": 2, "primary": 2}
    # The model proposed only primary options, so both global-source members
    # are engine insertions.  Source and reconciliation provenance are not the
    # same posterior axis.
    origin_by_option = {
        value["option_id"]: value["origin"] for value in reconciliation["members"]
    }
    assert all(
        origin_by_option[option_id] != "model"
        for option_id in selected_ids
        if source_by_option[option_id] == "global_restart"
    )
def test_contextual_projection_jointly_selects_memory_dose_feasible_k4() -> None:
    """Regression for the paid V15 Timeloop G5 staged-recourse failure.

    The card-compatible composite is deliberately last in the engine stratum.
    A structure-only contextual witness excludes it, after which adding it to
    K8 cannot repair the frozen K4.  The joint projection must instead put the
    compatible action in the exact contextual K4 before K8 composition.
    """

    base_request, legacy_binding = _hierarchical_boils_request_and_binding()
    contract = base_request.finite_variation_contract
    model_atom_by_position: dict[str, str] = {}
    for option in contract.options:
        position = dict(option.metadata).get("position")
        if position is not None:
            model_atom_by_position.setdefault(position, option.option_id)
    model_option_ids = tuple(
        model_atom_by_position[position]
        for position in sorted(model_atom_by_position)[:8]
    )
    supported_option = next(
        value
        for value in reversed(contract.options)
        if value.family == "composite_r2"
    )
    card = base_request.cards[0]
    dose = BoundedPortfolioMemoryDoseContract(
        card_supports=(
            PortfolioMemoryDoseCardSupport(
                card_key=card.card_key,
                card_content_sha256=card.content_sha256,
                finite_contract_identity_sha256=contract.identity_sha256,
                compatible_options=(
                    (supported_option.option_id, supported_option.identity_sha256),
                ),
                support_policy_id="test_exact_support",
                support_policy_version=1,
                support_policy_definition_sha256=_sha("test exact support"),
            ),
        ),
        proposed_supported_member_bounds=(1, 1),
        evaluated_supported_member_bounds=(1, 1),
        minimum_unattributed_proposed_members=7,
        minimum_unattributed_evaluated_members=3,
    )
    request = replace(
        base_request,
        require_supporting_cards=False,
        memory_dose_contract=dose,
    )
    pool = TaskKeyedCommonCandidatePoolPolicy(
        replicate_seed=20260722,
        candidate_pool_size=None,
    ).select(
        benchmark_sha256=legacy_binding.context.scope.benchmark_sha256,
        wave_index=5,
        parent_configuration_sha256=contract.parent_configuration_sha256,
        contract=contract,
        evaluation_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=True,
        required_option_ids=common_pool_required_option_ids(request),
    )
    assert supported_option.option_id in pool.option_ids
    allocation = ContextualPortfolioAllocationContract(
        campaign_scope_sha256=_sha("joint-dose campaign scope"),
        query_sha256=_sha("joint-dose allocation query"),
        decision_sha256=_sha("joint-dose allocation decision"),
        campaign_generation=5,
        controller_wave_index=5,
        phase_id="terminal_conversion",
        slice_id="active_memory",
        evaluation_slots=request.portfolio_size,
        source_target_counts=(("engine", 2), ("model", 2)),
        operator_target_counts=(("atomic", 2), ("composite", 2)),
    )
    scope = replace(
        legacy_binding.context.scope,
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            bounded_memory_dose=True,
            finite_variation_contract=contract,
            hierarchical_composition_required_proposals=2,
            feasibility_witness_mode=(
                CalibratedPortfolioFeasibilityWitnessMode.TASK_KEYED_COMMON_POOL
            ),
            constraint_decoupled=True,
        ),
        selector_policy_definition_sha256=(
            CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = CalibratedPortfolioInputBinding(
        request_sha256=request.request_sha256,
        context=replace(
            legacy_binding.context,
            wave_index=5,
            scope=scope,
            assigned_card_keys=dose.assigned_card_keys,
            calibration_snapshot=replace(
                legacy_binding.context.calibration_snapshot,
                scope=scope,
                cutoff_wave_index_exclusive=5,
            ),
        ),
        option_evidence=legacy_binding.option_evidence,
        common_candidate_pool=pool,
        contextual_allocation=allocation,
    )
    allocator = HorizonBoundedStructuralPosteriorSlatePolicy(
        build_terminal_tapered_family_exposure_phases(
            family="composite_r2",
            terminal_wave_index=5,
            terminal_exposure=2,
        )
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=allocator,
        constraint_decoupled=True,
        minimum_intervention_projection=True,
        evidence_calibrated_source_mix=True,
        contextual_search_allocation=True,
    )
    coordinator.register(request, binding)

    result = asyncio.run(
        coordinator.build_selector(
            _ProviderFreeHierarchicalAtomicSlateRunner(contract, model_option_ids)
        ).select(request)
    )

    assert supported_option.option_id in {
        value.option_id for value in result.decision.members
    }
    assert result.supplemental_audit is not None
    payload = thaw_json(result.supplemental_audit.payload)
    assert type(payload) is dict
    projection = payload["semantic_reconciliation"][
        "contextual_allocation_projection"
    ]
    assert projection["policy_version"] == 3
    assert projection["exact"] is True
    dose_witness = projection["memory_dose_feasibility_witness"]
    assert dose_witness["supported_member_count"] == 1
    assert dose_witness["covered_card_keys"] == [card.card_key]
    assert supported_option.option_id in {
        value["option_id"] for value in dose_witness["members"]
    }
    assert projection["objective_values_consulted"] is False
    assert projection["workload_model_provider_identifiers_consulted"] is False
    decoded = decode_calibrated_portfolio_audit(
        result.supplemental_audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert decoded.slate.to_record() == payload["calibrated_slate"]


def test_hierarchical_prompt_retains_authenticated_filtered_component_refs() -> None:
    request, _ = _hierarchical_boils_request_and_binding()
    contract = request.finite_variation_contract
    composite = next(
        option for option in contract.options if option.family == "composite_r2"
    )
    filtered_source_id = dict(composite.metadata)["left_option_id"]
    filtered = FiniteVariationContract(
        catalog_id="eligible_hierarchical_fixture",
        catalog_version=1,
        catalog_definition_sha256=_sha("eligible hierarchical fixture"),
        parent_configuration=contract.parent_configuration,
        options=tuple(
            option
            for option in contract.options
            if option.option_id != filtered_source_id
        ),
    )

    prompt_definition = calibrated_portfolio_prompt_definition_sha256(
        finite_variation_contract=filtered,
    )

    assert prompt_definition == calibrated_portfolio_prompt_definition_sha256(
        hierarchical_composition_required_proposals=2,
    )


def test_operator_stratified_allocator_evaluates_hierarchical_assay_member() -> None:
    request, legacy_binding = _hierarchical_boils_request_and_binding()
    scope = replace(
        legacy_binding.context.scope,
        selector_policy_definition_sha256=(
            OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = replace(
        legacy_binding,
        context=replace(
            legacy_binding.context,
            scope=scope,
            calibration_snapshot=replace(
                legacy_binding.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )
    runner = _ProviderFreeHierarchicalSlateRunner(request.finite_variation_contract)
    allocator = OperatorStratifiedStructuralPosteriorSlatePolicy()
    coordinator = CalibratedPortfolioCampaignCoordinator(allocator=allocator)
    coordinator.register(request, binding)
    selector = coordinator.build_selector(runner)

    assert type(selector) is (
        PydanticAIOperatorStratifiedCalibratedPortfolioSelectionPolicy
    )
    result = asyncio.run(selector.select(request))

    # The public ranked decision stays allocator-agnostic; the supplemental
    # audit retains the exact operator-stratified allocation for replay.
    assert len(result.decision.members) == request.portfolio_size
    audit = result.supplemental_audit
    assert audit is not None
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert type(decoded.allocation) is (
        OperatorStratifiedStructuralPosteriorSlateDecision
    )
    selected_families = {
        request.finite_variation_contract.resolve(member.option_id).family
        for member in result.decision.members
    }
    assert "composite_r2" in selected_families
    variation_trace = summarize_campaign_variation_trace(
        (result,),
        required_composite_proposals=2,
    ).to_record()
    assert variation_trace["composite_evaluated_count"] >= 1


def test_horizon_bounded_allocator_tapers_exposure_without_changing_adapter() -> None:
    request, legacy_binding = _hierarchical_boils_request_and_binding()
    scope = replace(
        legacy_binding.context.scope,
        selector_policy_definition_sha256=(
            HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    discovery_binding = replace(
        legacy_binding,
        context=replace(
            legacy_binding.context,
            scope=scope,
            calibration_snapshot=replace(
                legacy_binding.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )
    phases = build_terminal_tapered_family_exposure_phases(
        family="composite_r2",
        terminal_wave_index=5,
    )
    allocator = HorizonBoundedStructuralPosteriorSlatePolicy(phases)

    discovery_coordinator = CalibratedPortfolioCampaignCoordinator(allocator=allocator)
    discovery_coordinator.register(request, discovery_binding)
    discovery_selector = discovery_coordinator.build_selector(
        _ProviderFreeHierarchicalSlateRunner(request.finite_variation_contract)
    )
    assert type(discovery_selector) is (
        PydanticAIHorizonBoundedCalibratedPortfolioSelectionPolicy
    )
    discovery = asyncio.run(discovery_selector.select(request))
    discovery_audit = discovery.supplemental_audit
    assert discovery_audit is not None
    discovery_decoded = decode_calibrated_portfolio_audit(
        discovery_audit,
        request=request,
        binding=discovery_binding,
        allocator=allocator,
    )
    assert type(discovery_decoded.allocation) is (
        HorizonBoundedStructuralPosteriorSlateDecision
    )

    terminal_binding = replace(
        discovery_binding,
        context=replace(discovery_binding.context, wave_index=5),
    )
    terminal_coordinator = CalibratedPortfolioCampaignCoordinator(allocator=allocator)
    terminal_coordinator.register(request, terminal_binding)
    terminal = asyncio.run(
        terminal_coordinator.build_selector(
            _ProviderFreeHierarchicalSlateRunner(request.finite_variation_contract)
        ).select(request)
    )
    terminal_audit = terminal.supplemental_audit
    assert terminal_audit is not None
    terminal_decoded = decode_calibrated_portfolio_audit(
        terminal_audit,
        request=request,
        binding=terminal_binding,
        allocator=allocator,
    )

    def composite_count(result) -> int:
        return sum(
            request.finite_variation_contract.resolve(member.option_id).family
            == "composite_r2"
            for member in result.decision.members
        )

    assert composite_count(discovery) == 2
    assert composite_count(terminal) == 0
    assert (
        discovery_decoded.allocation.policy_configuration_sha256
        == terminal_decoded.allocation.policy_configuration_sha256
        == allocator.configuration_sha256
    )
    assert discovery_decoded.allocation.active_exposure_phase.start_wave_index == 0
    assert terminal_decoded.allocation.active_exposure_phase.start_wave_index == 5


def test_acquisition_certified_selector_reserves_anchor_and_replays_replacement() -> (
    None
):
    ids = DeterministicIdFactory("acquisition_certified_selector")
    request = _boils_request(ids, require_disjoint=False)
    options = request.finite_variation_contract.options
    assert len(options) >= 12
    anchor_ids = tuple(sorted(value.option_id for value in options[:4]))
    model_ids = tuple(value.option_id for value in options[4:12])
    promoted_residual_id = model_ids[0]

    legacy_binding = _binding(request)
    scope = replace(
        legacy_binding.context.scope,
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            constraint_decoupled=True,
        ),
        selector_policy_definition_sha256=(
            ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = replace(
        legacy_binding,
        context=replace(
            legacy_binding.context,
            scope=scope,
            calibration_snapshot=replace(
                legacy_binding.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )
    registry = AcquisitionCertifiedSlateContextRegistry()
    registry.register(
        AcquisitionCertifiedSlateContext(
            campaign_scope_sha256=_sha("acquisition certified campaign"),
            finite_contract_sha256=(
                request.finite_variation_contract.identity_sha256
            ),
            cutoff_index=8,
            seed=29,
            objectives=(
                FiniteAcquisitionObjective(
                    "total_levels", "min", 0.0, 100.0
                ),
                FiniteAcquisitionObjective(
                    "total_lut_count", "min", 0.0, 20_000.0
                ),
            ),
            observations=(
                FiniteAcquisitionObservation(
                    candidate_id="observed.seed",
                    configuration_sha256=_sha("acquisition observed seed"),
                    features=(0.0,),
                    objectives=(
                        ("total_levels", 75.0),
                        ("total_lut_count", 10_000.0),
                    ),
                ),
            ),
            candidates=tuple(
                FiniteAcquisitionCandidate(
                    candidate_id=option.option_id,
                    configuration_sha256=option.child_configuration_sha256,
                    features=(rank / (len(options) + 1.0),),
                )
                for rank, option in enumerate(options, start=1)
            ),
            reference_option_ids=anchor_ids,
        )
    )
    anchor_weight = {
        option_id: float(10 - rank)
        for rank, option_id in enumerate(anchor_ids)
    }
    weights = tuple(
        sorted(
            (
                option.option_id,
                30.0
                if option.option_id == promoted_residual_id
                else anchor_weight.get(option.option_id, 0.0),
            )
            for option in options
        )
    )
    allocator = AcquisitionCertifiedSlatePolicy(
        context_provider=registry,
        scorer=_FixtureAcquisitionBatchScorer(weights),
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=allocator,
        constraint_decoupled=True,
    )
    coordinator.register(request, binding)
    selector = coordinator.build_selector(_ProviderFreeSlateRunner(model_ids))
    assert type(selector) is (
        PydanticAIAcquisitionCertifiedResidualPortfolioSelectionPolicy
    )

    result = asyncio.run(selector.select(request))

    audit = result.supplemental_audit
    assert audit is not None
    payload = thaw_json(audit.payload)
    assert type(payload) is dict
    reconciled_ids = {
        value["option_id"] for value in payload["original_k8_response"]["members"]
    }
    assert set(anchor_ids) <= reconciled_ids
    allocation = payload["allocation"]
    assert allocation["reference_option_ids"] == list(anchor_ids)
    selected_ids = {value.option_id for value in result.decision.members}
    weakest_anchor = min(anchor_ids, key=anchor_weight.__getitem__)
    assert selected_ids == {
        promoted_residual_id,
        *set(anchor_ids).difference({weakest_anchor}),
    }
    assert float.fromhex(allocation["certificate_margin_hex"]) > 0.0
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert decoded.allocation.to_record() == allocation


def test_regret_bounded_selector_admits_authenticated_residual_inside_envelope() -> (
    None
):
    ids = DeterministicIdFactory("regret_bounded_selector")
    request = _boils_request(ids, require_disjoint=False)
    options = request.finite_variation_contract.options
    assert len(options) >= 12
    anchor_ids = tuple(sorted(value.option_id for value in options[:4]))
    model_ids = tuple(value.option_id for value in options[4:12])
    promoted_residual_id = model_ids[0]

    legacy_binding = _binding(request)
    scope = replace(
        legacy_binding.context.scope,
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            constraint_decoupled=True,
        ),
        selector_policy_definition_sha256=(
            REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = replace(
        legacy_binding,
        context=replace(
            legacy_binding.context,
            scope=scope,
            calibration_snapshot=replace(
                legacy_binding.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )
    registry = AcquisitionCertifiedSlateContextRegistry()
    registry.register(
        AcquisitionCertifiedSlateContext(
            campaign_scope_sha256=_sha("regret bounded campaign"),
            finite_contract_sha256=request.finite_variation_contract.identity_sha256,
            cutoff_index=8,
            seed=31,
            objectives=(
                FiniteAcquisitionObjective("total_levels", "min", 0.0, 100.0),
                FiniteAcquisitionObjective(
                    "total_lut_count", "min", 0.0, 20_000.0
                ),
            ),
            observations=(
                FiniteAcquisitionObservation(
                    candidate_id="observed.seed",
                    configuration_sha256=_sha("regret observed seed"),
                    features=(0.0,),
                    objectives=(
                        ("total_levels", 75.0),
                        ("total_lut_count", 10_000.0),
                    ),
                ),
            ),
            candidates=tuple(
                FiniteAcquisitionCandidate(
                    candidate_id=option.option_id,
                    configuration_sha256=option.child_configuration_sha256,
                    features=(rank / (len(options) + 1.0),),
                )
                for rank, option in enumerate(options, start=1)
            ),
            reference_option_ids=anchor_ids,
        )
    )
    weights = tuple(
        sorted(
            (
                option.option_id,
                0.95
                if option.option_id == promoted_residual_id
                else 1.0
                if option.option_id in anchor_ids
                else 0.0,
            )
            for option in options
        )
    )
    allocator = RegretBoundedSlatePolicy(
        context_provider=registry,
        scorer=_FixtureAcquisitionBatchScorer(weights),
        future_value_policy=ResidualInformationAssayValuePolicy(0.06),
        minimum_acquisition_retention_ratio=0.95,
        allow_development_assay=True,
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=allocator,
        constraint_decoupled=True,
    )
    coordinator.register(request, binding)
    selector = coordinator.build_selector(_ProviderFreeSlateRunner(model_ids))
    assert type(selector) is PydanticAIRegretBoundedInformationPortfolioSelectionPolicy

    result = asyncio.run(selector.select(request))

    audit = result.supplemental_audit
    assert audit is not None
    payload = thaw_json(audit.payload)
    assert type(payload) is dict
    allocation = payload["allocation"]
    assert allocation["reference_option_ids"] == list(anchor_ids)
    assert allocation["selected_future_value"]["authority"] == "development_assay"
    assert float.fromhex(allocation["acquisition_regret_hex"]) == pytest.approx(0.05)
    selected_ids = {value.option_id for value in result.decision.members}
    assert promoted_residual_id in selected_ids
    assert len(selected_ids.intersection(anchor_ids)) == 3
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert decoded.allocation.to_record() == allocation


def test_constraint_decoupled_selector_reconciles_duplicate_infeasible_semantics() -> (
    None
):
    ids = DeterministicIdFactory("constraint_decoupled_semantic_reconciliation")
    request = _boils_request(ids, require_disjoint=True)
    legacy_binding = _binding(request)
    scope = replace(
        legacy_binding.context.scope,
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            constraint_decoupled=True,
        ),
        selector_policy_definition_sha256=(
            CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = replace(
        legacy_binding,
        context=replace(
            legacy_binding.context,
            scope=scope,
            calibration_snapshot=replace(
                legacy_binding.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )
    duplicate_option_id = request.finite_variation_contract.options[0].option_id
    calls: list[StructuredGenerationRequest[Any]] = []

    async def runner(
        low_level_request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        calls.append(low_level_request)
        members = [
            {
                "option_id": duplicate_option_id,
                "supporting_card_keys": ["card.a"],
                "effect_predictions": [
                    {
                        "metric_id": "total_levels",
                        "direction": "decrease",
                        "confidence": "high",
                    },
                    {
                        "metric_id": "total_lut_count",
                        "direction": "decrease",
                        "confidence": "medium",
                    },
                ],
                "role_proposal": "exploit",
                "design_rationale": (
                    f"Repeated semantic preference at model rank {rank}."
                ),
            }
            for rank in range(1, 9)
        ]
        value = low_level_request.output_type.model_validate(
            {"members": members},
            strict=True,
        )
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider-free/constraint-decoupled",
            resolved_model="provider-free/constraint-decoupled",
            resolved_provider="provider-free",
            provider_response_id="constraint-decoupled-response",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=100,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1,
        )

    allocator = HorizonBoundedStructuralPosteriorSlatePolicy(
        build_terminal_tapered_family_exposure_phases(
            family="composite_r2",
            terminal_wave_index=5,
        )
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=allocator,
        constraint_decoupled=True,
    )
    coordinator.register(request, binding)
    selector = coordinator.build_selector(runner)
    assert type(selector) is (
        PydanticAIConstraintDecoupledHorizonPortfolioSelectionPolicy
    )
    result = asyncio.run(selector.select(request))

    assert len(calls) == 1
    assert coordinator.render(request) == calls[0].prompt
    assert "Trusted code owns deduplication" in calls[0].prompt
    audit = result.supplemental_audit
    assert audit is not None
    payload = thaw_json(audit.payload)
    assert type(payload) is dict
    original_ids = tuple(
        value["option_id"] for value in payload["original_model_response"]["members"]
    )
    reconciled_ids = tuple(
        value["option_id"] for value in payload["original_k8_response"]["members"]
    )
    assert original_ids == (duplicate_option_id,) * 8
    assert len(set(reconciled_ids)) == 8
    reconciliation = payload["semantic_reconciliation"]
    assert reconciliation["duplicate_model_member_count"] == 7
    assert sum(value["origin"] != "model" for value in reconciliation["members"]) == 7
    selected_ids = tuple(value.option_id for value in result.decision.members)
    validate_pairwise_disjoint_parent_patch_selection(
        request.finite_variation_contract,
        selected_ids,
    )
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert decoded.slate.to_record() == payload["calibrated_slate"]


def test_minimum_intervention_selector_retains_feasible_model_semantics() -> None:
    ids = DeterministicIdFactory("minimum_intervention_semantic_projection")
    request = _boils_request(ids, require_disjoint=True)
    legacy_binding = _binding(request)
    scope = replace(
        legacy_binding.context.scope,
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            constraint_decoupled=True,
        ),
        selector_policy_definition_sha256=(
            MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = replace(
        legacy_binding,
        context=replace(
            legacy_binding.context,
            scope=scope,
            calibration_snapshot=replace(
                legacy_binding.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )
    model_option_ids = _proposal_option_ids(request.finite_variation_contract)
    runner = _ProviderFreeSlateRunner(model_option_ids)
    allocator = HorizonBoundedStructuralPosteriorSlatePolicy(
        build_terminal_tapered_family_exposure_phases(
            family="composite_r2",
            terminal_wave_index=5,
        )
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=allocator,
        constraint_decoupled=True,
        minimum_intervention_projection=True,
    )
    coordinator.register(request, binding)
    selector = coordinator.build_selector(runner)
    assert type(selector) is (
        PydanticAIMinimumInterventionHorizonPortfolioSelectionPolicy
    )
    result = asyncio.run(selector.select(request))

    audit = result.supplemental_audit
    assert audit is not None
    payload = thaw_json(audit.payload)
    assert type(payload) is dict
    assert payload["policy_id"] == (
        "pydantic_ai_minimum_intervention_horizon_portfolio"
    )
    reconciliation = payload["semantic_reconciliation"]
    assert reconciliation["policy_id"] == (
        "semantic_slate_minimum_intervention_reconciliation"
    )
    assert reconciliation["projection_objective"] == [
        "maximize_retained_model_member_count",
        "prefer_original_model_rank_lexicographically",
        "canonical_nonmodel_tie_break",
    ]
    assert reconciliation["retained_model_member_count"] == 8
    assert reconciliation["engine_inserted_member_count"] == 0
    assert reconciliation["semantic_intervention_count"] == 0
    assert (
        reconciliation["evaluation_witness_retained_model_member_count"]
        == request.portfolio_size
    )
    assert (
        tuple(
            value["option_id"] for value in payload["original_k8_response"]["members"]
        )
        == model_option_ids
    )
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert decoded.slate.to_record() == payload["calibrated_slate"]


def test_source_mix_protects_one_task_keyed_global_member_in_wave_one() -> None:
    ids = DeterministicIdFactory("evidence_calibrated_source_mix")
    request = _boils_request(ids, require_disjoint=True)
    legacy = _binding(request)
    pool = TaskKeyedCommonCandidatePoolPolicy(
        replicate_seed=20260722,
        candidate_pool_size=24,
    ).select(
        benchmark_sha256=legacy.context.scope.benchmark_sha256,
        wave_index=1,
        parent_configuration_sha256=(
            request.finite_variation_contract.parent_configuration_sha256
        ),
        contract=request.finite_variation_contract,
        evaluation_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=True,
    )
    scope = replace(
        legacy.context.scope,
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            feasibility_witness_mode=(
                CalibratedPortfolioFeasibilityWitnessMode.TASK_KEYED_COMMON_POOL
            ),
            constraint_decoupled=True,
        ),
        selector_policy_definition_sha256=(
            EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = replace(
        legacy,
        context=replace(
            legacy.context,
            wave_index=1,
            scope=scope,
            calibration_snapshot=replace(
                legacy.context.calibration_snapshot,
                scope=scope,
                cutoff_wave_index_exclusive=1,
            ),
        ),
        common_candidate_pool=pool,
    )
    model_option_ids = pool.option_ids[:8]
    allocator = HorizonBoundedStructuralPosteriorSlatePolicy(
        build_terminal_tapered_family_exposure_phases(
            family="composite_r2",
            terminal_wave_index=5,
        )
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=allocator,
        constraint_decoupled=True,
        minimum_intervention_projection=True,
        evidence_calibrated_source_mix=True,
    )
    coordinator.register(request, binding)
    selector = coordinator.build_selector(_ProviderFreeSlateRunner(model_option_ids))
    assert type(selector) is (
        PydanticAIEvidenceCalibratedSourceMixPortfolioSelectionPolicy
    )
    result = asyncio.run(selector.select(request))

    audit = result.supplemental_audit
    assert audit is not None
    payload = thaw_json(audit.payload)
    assert type(payload) is dict
    reconciliation = payload["semantic_reconciliation"]
    assert reconciliation["policy_id"] == (
        "semantic_slate_evidence_calibrated_source_mix_reconciliation"
    )
    protected = tuple(reconciliation["protected_allocation_option_ids"])
    assert len(protected) == 1
    assert protected[0] not in model_option_ids
    assert protected[0] in {value.option_id for value in result.decision.members}
    protected_member = next(
        value
        for value in reconciliation["members"]
        if value["option_id"] == protected[0]
    )
    assert protected_member["origin"] == "engine_global_coverage"
    assert reconciliation["retained_model_member_count"] >= 1
    assert reconciliation["source_mix_outcomes_consulted"] is False
    decoded = decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )
    assert decoded.slate.to_record() == payload["calibrated_slate"]


def test_constraint_decoupled_engine_repairs_provider_memory_dose_globally() -> None:
    ids = DeterministicIdFactory("constraint_decoupled_memory_dose")
    legacy_request = _boils_request(ids, require_disjoint=True)
    proposal_ids = _proposal_option_ids(legacy_request.finite_variation_contract)
    cards = {value.card_key: value for value in legacy_request.cards}
    supported_rows = (("card.a", proposal_ids[0]), ("card.b", proposal_ids[2]))
    dose = BoundedPortfolioMemoryDoseContract(
        card_supports=tuple(
            PortfolioMemoryDoseCardSupport(
                card_key=card_key,
                card_content_sha256=cards[card_key].content_sha256,
                finite_contract_identity_sha256=(
                    legacy_request.finite_variation_contract.identity_sha256
                ),
                compatible_options=(
                    (
                        option_id,
                        legacy_request.finite_variation_contract.resolve(
                            option_id
                        ).identity_sha256,
                    ),
                ),
                support_policy_id="test_exact_support",
                support_policy_version=1,
                support_policy_definition_sha256=_sha("test exact support"),
            )
            for card_key, option_id in supported_rows
        ),
        proposed_supported_member_bounds=(2, 2),
        evaluated_supported_member_bounds=(2, 2),
        minimum_unattributed_proposed_members=6,
        minimum_unattributed_evaluated_members=2,
    )
    request = replace(
        legacy_request,
        require_supporting_cards=False,
        memory_dose_contract=dose,
    )
    legacy_binding = _binding(request)
    scope = replace(
        legacy_binding.context.scope,
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            bounded_memory_dose=True,
            constraint_decoupled=True,
        ),
        selector_policy_definition_sha256=(
            CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = replace(
        legacy_binding,
        context=replace(
            legacy_binding.context,
            scope=scope,
            calibration_snapshot=replace(
                legacy_binding.context.calibration_snapshot,
                scope=scope,
            ),
        ),
    )

    async def runner(
        low_level_request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        value = low_level_request.output_type.model_validate(
            {
                "members": [
                    {
                        "option_id": option_id,
                        "supporting_card_keys": ["card.a", "card.b"],
                        "effect_predictions": [
                            {
                                "metric_id": "total_levels",
                                "direction": "decrease",
                                "confidence": "high",
                            },
                            {
                                "metric_id": "total_lut_count",
                                "direction": "decrease",
                                "confidence": "medium",
                            },
                        ],
                        "role_proposal": "exploit",
                        "design_rationale": (
                            f"Semantically preferred action at rank {rank}."
                        ),
                    }
                    for rank, option_id in enumerate(proposal_ids, start=1)
                ]
            },
            strict=True,
        )
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider-free/constraint-decoupled-dose",
            resolved_model="provider-free/constraint-decoupled-dose",
            resolved_provider="provider-free",
            provider_response_id="constraint-decoupled-dose-response",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=100,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1,
        )

    allocator = HorizonBoundedStructuralPosteriorSlatePolicy(
        build_terminal_tapered_family_exposure_phases(
            family="composite_r2",
            terminal_wave_index=5,
        )
    )
    result = asyncio.run(
        PydanticAIConstraintDecoupledHorizonPortfolioSelectionPolicy(
            generate_once=runner,
            binding_for=lambda _: binding,
            allocator=allocator,
        ).select(request)
    )
    audit = result.supplemental_audit
    assert audit is not None
    payload = thaw_json(audit.payload)
    assert type(payload) is dict
    assert all(
        len(value["supporting_card_keys"]) == 2
        for value in payload["original_model_response"]["members"]
    )
    reconciled = payload["original_k8_response"]["members"]
    assert sum(bool(value["supporting_card_keys"]) for value in reconciled) == 2
    assert {card for value in reconciled for card in value["supporting_card_keys"]} == {
        "card.a",
        "card.b",
    }
    assert payload["allocation"]["memory_dose_assessment"]["passed"] is True
    decode_calibrated_portfolio_audit(
        audit,
        request=request,
        binding=binding,
        allocator=allocator,
    )


def test_legacy_v1_default_accepts_overlap_but_opt_in_schema_rejects_it() -> None:
    ids = DeterministicIdFactory("legacy_disjoint_default")
    request = _boils_request(ids, require_disjoint=False)
    same_position = tuple(
        option.option_id
        for option in request.finite_variation_contract.options
        if _position(option) == "00"
    )[:4]
    runner = _LegacyRunner(same_position)
    legacy = asyncio.run(PydanticAIPortfolioSelectionPolicy(runner).select(request))
    assert runner.calls == 1
    assert legacy.decision.policy_version == PORTFOLIO_SELECTION_POLICY_VERSION == 1
    assert legacy.supplemental_audit is None
    assert "require_pairwise_disjoint_parent_patches" not in request.to_record()
    explicit_false = replace(request, require_pairwise_disjoint_parent_patches=False)
    assert explicit_false.to_record() == request.to_record()
    assert explicit_false.request_sha256 == request.request_sha256
    assert render_portfolio_selection_prompt(explicit_false) == (
        render_portfolio_selection_prompt(request)
    )

    constrained = replace(
        request,
        call_id=ids.new_llm_call_id(),
        require_pairwise_disjoint_parent_patches=True,
    )
    assert constrained.request_sha256 != request.request_sha256
    rejecting_runner = _LegacyRunner(same_position)
    with pytest.raises(ValidationError, match="overlapping parent-relative patches"):
        asyncio.run(
            PydanticAIPortfolioSelectionPolicy(rejecting_runner).select(constrained)
        )

    by_position: dict[str, str] = {}
    for option in request.finite_variation_contract.options:
        by_position.setdefault(_position(option), option.option_id)
    valid_runner = _LegacyRunner(
        tuple(by_position[value] for value in ("00", "01", "02", "03"))
    )
    constrained_result = asyncio.run(
        PydanticAIPortfolioSelectionPolicy(valid_runner).select(constrained)
    )
    assert PORTFOLIO_SELECTION_DISJOINT_POLICY_VERSION == 2
    assert constrained_result.decision.policy_version == (
        PORTFOLIO_SELECTION_DISJOINT_POLICY_VERSION
    )
    assert constrained_result.decision.policy_definition_sha256 == (
        PORTFOLIO_SELECTION_DISJOINT_POLICY_DEFINITION_SHA256
    )


def _joint_feasibility_contract() -> FiniteVariationContract:
    parent = _frozen({"a": 0, "b": 0})
    parent_sha256 = typed_json_sha256(parent)
    options = tuple(
        FiniteVariationOption(
            option_id=option_id,
            parent_configuration_sha256=parent_sha256,
            child_configuration=_frozen(child),
            family=family,
            description=description,
        )
        for option_id, child, family, description in (
            ("alpha.a", {"a": 1, "b": 0}, "alpha", "Change only a."),
            ("alpha.b", {"a": 0, "b": 1}, "alpha", "Change only b."),
            (
                "beta.ab",
                {"a": 2, "b": 2},
                "beta",
                "Change both a and b.",
            ),
        )
    )
    return FiniteVariationContract(
        catalog_id="joint_family_disjoint_test",
        catalog_version=1,
        catalog_definition_sha256=_sha("joint family disjoint catalog"),
        parent_configuration=parent,
        options=options,
    )


def test_disjoint_preflight_jointly_enforces_family_minimum() -> None:
    contract = _joint_feasibility_contract()
    assert finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
        contract,
        tuple(option.option_id for option in contract.options),
        portfolio_size=2,
    )
    assert not finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
        contract,
        tuple(option.option_id for option in contract.options),
        portfolio_size=2,
        min_distinct_families=2,
    )
    ids = DeterministicIdFactory("joint_family_disjoint_request")
    common = {
        "call_id": ids.new_llm_call_id(),
        "operation": "select_portfolio",
        "instruction": "Select two sealed options.",
        "context": _frozen({"benchmark": "joint-feasibility"}),
        "finite_variation_contract": contract,
        "cards": _cards(),
        "portfolio_size": 2,
        "required_metric_ids": ("loss",),
        "min_distinct_families": 2,
    }
    unconstrained = PortfolioSelectionRequest(**common)
    assert unconstrained.require_pairwise_disjoint_parent_patches is False
    with pytest.raises(ValueError, match="no feasible pairwise-disjoint"):
        PortfolioSelectionRequest(
            **common,
            require_pairwise_disjoint_parent_patches=True,
        )


def test_family_exposure_projection_preserves_feasible_bounds_and_recourses_blindly() -> (
    None
):
    contract = _joint_feasibility_contract()
    option_ids = tuple(option.option_id for option in contract.options)

    feasible = (("alpha", 2, 2),)
    assert (
        project_family_exposure_bounds_to_pairwise_disjoint_feasibility(
            contract,
            option_ids,
            portfolio_size=2,
            requested_bounds=feasible,
        )
        == feasible
    )

    # The only disjoint K=2 portfolio is {alpha.a, alpha.b}; beta.ab overlaps
    # both.  Projection therefore changes beta exact-one to exact-zero using
    # only sealed patch topology and family labels.
    assert project_family_exposure_bounds_to_pairwise_disjoint_feasibility(
        contract,
        option_ids,
        portfolio_size=2,
        requested_bounds=(("beta", 1, 1),),
    ) == (("beta", 0, 0),)
    assert pairwise_disjoint_parent_patch_witness(
        contract,
        option_ids,
        portfolio_size=2,
        family_exposure_bounds=(("beta", 0, 0),),
    ) == ("alpha.a", "alpha.b")


def test_family_exposure_projection_minimizes_total_interval_violation() -> None:
    contract = _joint_feasibility_contract()
    option_ids = tuple(option.option_id for option in contract.options)

    # Exact one from each family is structurally impossible.  The deterministic
    # closest feasible vector is alpha=2, beta=0 (L1 violation two).
    assert project_family_exposure_bounds_to_pairwise_disjoint_feasibility(
        contract,
        option_ids,
        portfolio_size=2,
        requested_bounds=(("alpha", 1, 1), ("beta", 1, 1)),
    ) == (("alpha", 2, 2), ("beta", 0, 0))


def test_family_exposure_preflight_prunes_mandatory_duplicate_slots() -> None:
    parent = _frozen({key: 0 for key in ("a", "b", "c", "d", "e")})
    parent_sha256 = typed_json_sha256(parent)
    rows = (
        ("bulk.a", "a", "bulk"),
        ("bulk.b", "b", "bulk"),
        ("gamma.c", "c", "gamma"),
        ("delta.d", "d", "delta"),
        ("epsilon.e", "e", "epsilon"),
    )
    options = tuple(
        FiniteVariationOption(
            option_id=option_id,
            parent_configuration_sha256=parent_sha256,
            child_configuration=_frozen(
                {
                    candidate_key: int(candidate_key == changed_key)
                    for candidate_key in ("a", "b", "c", "d", "e")
                }
            ),
            family=family,
            description=f"Change only {changed_key}.",
        )
        for option_id, changed_key, family in rows
    )
    contract = FiniteVariationContract(
        catalog_id="mandatory_duplicate_family_pruning",
        catalog_version=1,
        catalog_definition_sha256=_sha("mandatory duplicate family pruning"),
        parent_configuration=parent,
        options=options,
    )
    option_ids = tuple(option.option_id for option in options)

    # Two mandatory members from one family consume one duplicate slot, so a
    # K=4 portfolio can span at most three families.  This necessary condition
    # must be rejected before enumerating the exact K-subsets.
    assert not finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
        contract,
        option_ids,
        portfolio_size=4,
        min_distinct_families=4,
        family_exposure_bounds=(("bulk", 2, 2),),
    )
    assert project_family_exposure_bounds_to_pairwise_disjoint_feasibility(
        contract,
        option_ids,
        portfolio_size=4,
        min_distinct_families=4,
        requested_bounds=(("bulk", 2, 2),),
    ) == (("bulk", 1, 1),)


def test_parent_patch_certificate_revalidates_content_before_cache_reuse() -> None:
    contract = _joint_feasibility_contract()
    option_ids = tuple(option.option_id for option in contract.options)
    assert finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
        contract,
        option_ids,
        portfolio_size=2,
    )

    # Corrupt the otherwise frozen graph after the certificate is cached. A
    # trust-boundary lookup must validate current content, not trust object ID.
    object.__setattr__(
        contract.options[0],
        "child_configuration",
        contract.parent_configuration,
    )
    with pytest.raises(ValueError):
        finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
            contract,
            option_ids,
            portfolio_size=2,
        )


def test_bounded_memory_dose_runs_through_schema_allocator_and_final_decision() -> None:
    ids = DeterministicIdFactory("calibrated_bounded_memory_dose")
    legacy_request = _boils_request(ids, require_disjoint=True)
    proposal_ids = _proposal_option_ids(legacy_request.finite_variation_contract)
    cards = {value.card_key: value for value in legacy_request.cards}
    supported_rows = (("card.a", proposal_ids[0]), ("card.b", proposal_ids[2]))
    supports = tuple(
        PortfolioMemoryDoseCardSupport(
            card_key=card_key,
            card_content_sha256=cards[card_key].content_sha256,
            finite_contract_identity_sha256=(
                legacy_request.finite_variation_contract.identity_sha256
            ),
            compatible_options=(
                (
                    option_id,
                    legacy_request.finite_variation_contract.resolve(
                        option_id
                    ).identity_sha256,
                ),
            ),
            support_policy_id="test_exact_support",
            support_policy_version=1,
            support_policy_definition_sha256=_sha("test exact support"),
        )
        for card_key, option_id in supported_rows
    )
    dose = BoundedPortfolioMemoryDoseContract(
        card_supports=supports,
        proposed_supported_member_bounds=(2, 2),
        evaluated_supported_member_bounds=(2, 2),
        minimum_unattributed_proposed_members=6,
        minimum_unattributed_evaluated_members=2,
    )
    request = replace(
        legacy_request,
        require_supporting_cards=False,
        memory_dose_contract=dose,
    )
    legacy_binding = _binding(request)
    bounded_scope = replace(
        legacy_binding.context.scope,
        prompt_definition_sha256=(
            calibrated_portfolio_prompt_definition_sha256(bounded_memory_dose=True)
        ),
    )
    binding = replace(
        legacy_binding,
        context=replace(
            legacy_binding.context,
            scope=bounded_scope,
            calibration_snapshot=replace(
                legacy_binding.context.calibration_snapshot,
                scope=bounded_scope,
            ),
        ),
    )
    calls: list[StructuredGenerationRequest[Any]] = []

    async def runner(
        low_level_request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        calls.append(low_level_request)
        citations = {
            proposal_ids[0]: ["card.a"],
            proposal_ids[2]: ["card.b"],
        }
        members = [
            {
                "option_id": option_id,
                "supporting_card_keys": citations.get(option_id, []),
                "effect_predictions": [
                    {
                        "metric_id": "total_levels",
                        "direction": "decrease",
                        "confidence": "high",
                    },
                    {
                        "metric_id": "total_lut_count",
                        "direction": "decrease",
                        "confidence": "medium",
                    },
                ],
                "role_proposal": (
                    "exploit" if index < 4 else "falsify" if index < 7 else "coverage"
                ),
                "design_rationale": f"Bounded dose proposal {index}.",
            }
            for index, option_id in enumerate(proposal_ids, start=1)
        ]
        value = low_level_request.output_type.model_validate(
            {"members": members},
            strict=True,
        )
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider-free/bounded-dose",
            resolved_model="provider-free/bounded-dose",
            resolved_provider="provider-free",
            provider_response_id="provider-free-bounded-dose-response",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=100,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1_000,
        )

    result = asyncio.run(
        PydanticAICalibratedPortfolioSelectionPolicy(
            generate_once=runner,
            binding_for=lambda _: binding,
        ).select(request)
    )
    assert len(calls) == 1
    assert "prompt_exposed_exploration_not_blinded_control" in calls[0].prompt
    assessment = result.decision.memory_dose_assessment
    assert assessment is not None and assessment.passed
    assert len(assessment.supported_member_ranks) == 2
    assert len(assessment.unattributed_member_ranks) == 2
    selected_ids = {value.option_id for value in result.decision.members}
    assert {proposal_ids[0], proposal_ids[2]}.issubset(selected_ids)
    assert result.supplemental_audit is not None
    decoded = decode_calibrated_portfolio_audit(
        result.supplemental_audit,
        request=request,
        binding=binding,
    )
    assert decoded.allocation.memory_dose_assessment == assessment


def test_task_keyed_common_pool_is_model_independent_and_exactly_reranked() -> None:
    ids = DeterministicIdFactory("common_pool_rerank")
    request = replace(
        _boils_request(ids, require_disjoint=True),
        require_supporting_cards=False,
    )
    policy = TaskKeyedCommonCandidatePoolPolicy(replicate_seed=20260719)
    pool = policy.select(
        benchmark_sha256=_scope().benchmark_sha256,
        wave_index=2,
        parent_configuration_sha256=(
            request.finite_variation_contract.parent_configuration_sha256
        ),
        contract=request.finite_variation_contract,
        evaluation_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=True,
    )
    # Provider execution fields and logical call identity cannot perturb the
    # common pool. Only task-level inputs above enter its receipt.
    second_request = replace(
        request,
        call_id=ids.new_llm_call_id(),
        max_output_tokens=65_536,
        temperature=0.7,
    )
    second_pool = policy.select(
        benchmark_sha256=_scope().benchmark_sha256,
        wave_index=2,
        parent_configuration_sha256=(
            second_request.finite_variation_contract.parent_configuration_sha256
        ),
        contract=second_request.finite_variation_contract,
        evaluation_size=second_request.portfolio_size,
        min_distinct_families=second_request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=True,
    )
    assert second_pool.to_record() == pool.to_record()
    assert len(pool.option_ids) == 8
    assert set(pool.feasibility_witness_option_ids).issubset(pool.option_ids)
    assert set(pool.option_ids[: request.portfolio_size]) != set(
        pool.feasibility_witness_option_ids
    )

    legacy = _binding(request)
    scope = replace(
        legacy.context.scope,
        prompt_definition_sha256=(
            CALIBRATED_PORTFOLIO_COMMON_POOL_PROMPT_DEFINITION_SHA256
        ),
        selector_policy_definition_sha256=(
            MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = replace(
        legacy,
        context=replace(
            legacy.context,
            scope=scope,
            assigned_card_keys=(),
            calibration_snapshot=replace(
                legacy.context.calibration_snapshot,
                scope=scope,
            ),
        ),
        common_candidate_pool=pool,
    )
    calls: list[StructuredGenerationRequest[Any]] = []

    async def runner(
        low_level_request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        calls.append(low_level_request)
        members = [
            {
                "option_id": option_id,
                "supporting_card_keys": [],
                "effect_predictions": [
                    {
                        "metric_id": "total_levels",
                        "direction": "decrease",
                        "confidence": "high",
                    },
                    {
                        "metric_id": "total_lut_count",
                        "direction": "decrease",
                        "confidence": "medium",
                    },
                ],
                "role_proposal": "exploit" if index <= 4 else "coverage",
                "design_rationale": f"Common-pool rerank {index}.",
            }
            for index, option_id in enumerate(reversed(pool.option_ids), start=1)
        ]
        value = low_level_request.output_type.model_validate(
            {"members": members}, strict=True
        )
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider-free/common-pool-model-a",
            resolved_model="provider-free/common-pool-model-a",
            resolved_provider="provider-free",
            provider_response_id="provider-free-common-pool-response",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=100,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1_000,
        )

    result = asyncio.run(
        PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy(
            generate_once=runner,
            binding_for=lambda _: binding,
            allocator=ModelAnchoredCalibratedSlatePolicy(model_anchor_count=3),
        ).select(request)
    )
    assert len(calls) == 1
    assert calls[0].output_type.ordered_common_pool_option_ids == pool.option_ids
    assert "task_keyed_common_candidate_pool" in calls[0].prompt
    assert "engine_verified_feasible_option_id_witness" not in calls[0].prompt
    assert {member.option_id for member in result.decision.members}.issubset(
        pool.option_ids
    )
    assert result.supplemental_audit is not None
    decoded = decode_calibrated_portfolio_audit(
        result.supplemental_audit,
        request=request,
        binding=binding,
        allocator=ModelAnchoredCalibratedSlatePolicy(model_anchor_count=3),
    )
    assert decoded.slate.members[0].option_id == pool.option_ids[-1]


def test_hard_dose_exact_action_is_guaranteed_in_common_pool_and_replayed() -> None:
    ids = DeterministicIdFactory("common_pool_exact_memory_support")
    base_request = replace(
        _boils_request(ids, require_disjoint=True),
        require_supporting_cards=False,
    )
    policy = TaskKeyedCommonCandidatePoolPolicy(replicate_seed=20260720)
    baseline = policy.select(
        benchmark_sha256=_scope().benchmark_sha256,
        wave_index=2,
        parent_configuration_sha256=(
            base_request.finite_variation_contract.parent_configuration_sha256
        ),
        contract=base_request.finite_variation_contract,
        evaluation_size=base_request.portfolio_size,
        min_distinct_families=base_request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=True,
    )
    required_option_id = next(
        option.option_id
        for option in base_request.finite_variation_contract.options
        if option.option_id not in baseline.option_ids
    )
    card = base_request.cards[0]
    option = base_request.finite_variation_contract.resolve(required_option_id)
    dose = BoundedPortfolioMemoryDoseContract(
        card_supports=(
            PortfolioMemoryDoseCardSupport(
                card_key=card.card_key,
                card_content_sha256=card.content_sha256,
                finite_contract_identity_sha256=(
                    base_request.finite_variation_contract.identity_sha256
                ),
                compatible_options=((required_option_id, option.identity_sha256),),
                support_policy_id="test_exact_support",
                support_policy_version=1,
                support_policy_definition_sha256=_sha("test exact support"),
            ),
        ),
        proposed_supported_member_bounds=(1, 1),
        evaluated_supported_member_bounds=(1, 1),
        minimum_unattributed_proposed_members=7,
        minimum_unattributed_evaluated_members=3,
        maximum_cards_per_member=1,
        require_every_assigned_card=True,
    )
    request = replace(base_request, memory_dose_contract=dose)
    required = common_pool_required_option_ids(request)
    assert required == (required_option_id,)

    explicit = replace(
        base_request,
        candidate_pool_required_option_ids=(required_option_id,),
    )
    assert common_pool_required_option_ids(explicit) == (required_option_id,)
    combined = replace(
        request,
        candidate_pool_required_option_ids=(required_option_id,),
    )
    assert common_pool_required_option_ids(combined) == (required_option_id,)

    treated = policy.select(
        benchmark_sha256=_scope().benchmark_sha256,
        wave_index=2,
        parent_configuration_sha256=(
            request.finite_variation_contract.parent_configuration_sha256
        ),
        contract=request.finite_variation_contract,
        evaluation_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=True,
        required_option_ids=required,
    )
    assert required_option_id in treated.option_ids
    assert treated.required_option_ids == required
    assert treated.option_ids[0] != required_option_id
    policy.require_decision(
        treated,
        benchmark_sha256=_scope().benchmark_sha256,
        wave_index=2,
        parent_configuration_sha256=(
            request.finite_variation_contract.parent_configuration_sha256
        ),
        contract=request.finite_variation_contract,
        evaluation_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=True,
        required_option_ids=required,
    )
    with pytest.raises(ValueError, match="differs from exact policy replay"):
        policy.require_decision(
            treated,
            benchmark_sha256=_scope().benchmark_sha256,
            wave_index=2,
            parent_configuration_sha256=(
                request.finite_variation_contract.parent_configuration_sha256
            ),
            contract=request.finite_variation_contract,
            evaluation_size=request.portfolio_size,
            min_distinct_families=request.min_distinct_families,
            require_pairwise_disjoint_parent_patches=True,
            required_option_ids=(),
        )


def test_task_keyed_common_universe_allows_exact_k8_subset_from_k24() -> None:
    ids = DeterministicIdFactory("common_universe_k24_select_k8")
    request = replace(
        _boils_request(ids, require_disjoint=True),
        require_supporting_cards=False,
    )
    legacy = _binding(request)
    policy = TaskKeyedCommonCandidatePoolPolicy(
        replicate_seed=20260720,
        candidate_pool_size=24,
        model_selection_size=8,
    )
    pool = policy.select(
        benchmark_sha256=legacy.context.scope.benchmark_sha256,
        wave_index=legacy.context.wave_index,
        parent_configuration_sha256=(
            request.finite_variation_contract.parent_configuration_sha256
        ),
        contract=request.finite_variation_contract,
        evaluation_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=True,
    )
    assert len(pool.option_ids) == 24
    assert pool.model_selection_size == 8
    selected_ids = (
        tuple(pool.feasibility_witness_option_ids)
        + tuple(
            option_id
            for option_id in pool.option_ids
            if option_id not in pool.feasibility_witness_option_ids
        )[:4]
    )
    assert len(selected_ids) == 8

    scope = replace(
        legacy.context.scope,
        prompt_definition_sha256=(
            CALIBRATED_PORTFOLIO_COMMON_POOL_PROMPT_DEFINITION_SHA256
        ),
        selector_policy_definition_sha256=(
            MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        ),
    )
    binding = replace(
        legacy,
        context=replace(
            legacy.context,
            scope=scope,
            assigned_card_keys=(),
            calibration_snapshot=replace(
                legacy.context.calibration_snapshot,
                scope=scope,
            ),
        ),
        common_candidate_pool=pool,
    )
    calls: list[StructuredGenerationRequest[Any]] = []

    async def runner(
        low_level_request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        calls.append(low_level_request)
        members = [
            {
                "option_id": option_id,
                "supporting_card_keys": [],
                "effect_predictions": [
                    {
                        "metric_id": metric_id,
                        "direction": "unknown",
                        "confidence": "unknown",
                    }
                    for metric_id in request.required_metric_ids
                ],
                "role_proposal": "coverage",
                "design_rationale": "Select from the fixed K24 universe.",
            }
            for option_id in selected_ids
        ]
        value = low_level_request.output_type.model_validate(
            {"members": members}, strict=True
        )
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider-free/common-universe-model",
            resolved_model="provider-free/common-universe-model",
            resolved_provider="provider-free",
            provider_response_id="provider-free-common-universe-response",
            finish_reason="stop",
            input_tokens=100,
            output_tokens=100,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1_000,
        )

    result = asyncio.run(
        PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy(
            generate_once=runner,
            binding_for=lambda _: binding,
            allocator=ModelAnchoredCalibratedSlatePolicy(model_anchor_count=4),
        ).select(request)
    )
    assert len(calls) == 1
    assert calls[0].output_type.ordered_common_pool_option_ids == pool.option_ids
    assert "candidate universe of 24" in calls[0].prompt
    assert len(result.decision.members) == 4
    assert {member.option_id for member in result.decision.members}.issubset(
        selected_ids
    )


def test_common_universe_scale_assays_are_nested_under_one_sampling_state() -> None:
    ids = DeterministicIdFactory("nested_common_universe_scale")
    request = replace(
        _boils_request(ids, require_disjoint=True),
        require_supporting_cards=False,
    )
    binding = _binding(request)

    def select(candidate_pool_size: int, model_selection_size: int):
        return TaskKeyedCommonCandidatePoolPolicy(
            replicate_seed=20260720,
            candidate_pool_size=candidate_pool_size,
            model_selection_size=model_selection_size,
        ).select(
            benchmark_sha256=binding.context.scope.benchmark_sha256,
            wave_index=binding.context.wave_index,
            parent_configuration_sha256=(
                request.finite_variation_contract.parent_configuration_sha256
            ),
            contract=request.finite_variation_contract,
            evaluation_size=request.portfolio_size,
            min_distinct_families=request.min_distinct_families,
            require_pairwise_disjoint_parent_patches=True,
        )

    k8 = select(8, 8)
    k24 = select(24, 8)
    k48 = select(48, 8)
    k24_select12 = select(24, 12)

    assert k8.state_identity_sha256 == k24.state_identity_sha256
    assert k24.state_identity_sha256 == k48.state_identity_sha256
    assert k24_select12.state_identity_sha256 == k24.state_identity_sha256
    assert (
        len(
            {
                k8.task_identity_sha256,
                k24.task_identity_sha256,
                k48.task_identity_sha256,
            }
        )
        == 3
    )
    assert k24_select12.task_identity_sha256 != k24.task_identity_sha256
    assert k8.feasibility_witness_option_ids == k24.feasibility_witness_option_ids
    assert k24.feasibility_witness_option_ids == k48.feasibility_witness_option_ids
    assert set(k8.option_ids).issubset(k24.option_ids)
    assert set(k24.option_ids).issubset(k48.option_ids)
    assert k8.option_ids == tuple(
        option_id for option_id in k48.option_ids if option_id in k8.option_ids
    )
    assert k24.option_ids == tuple(
        option_id for option_id in k48.option_ids if option_id in k24.option_ids
    )
    assert k24_select12.option_ids == k24.option_ids
    assert k24.to_prompt_record()["state_identity_sha256"] == (
        k24.state_identity_sha256
    )
    assert set(k8.option_ids[: request.portfolio_size]) != set(
        k8.feasibility_witness_option_ids
    )


def test_complete_common_universe_exposes_every_finite_option_with_exact_replay() -> (
    None
):
    ids = DeterministicIdFactory("complete_common_universe")
    request = replace(
        _boils_request(ids, require_disjoint=True),
        require_supporting_cards=False,
    )
    binding = _binding(request)
    policy = TaskKeyedCommonCandidatePoolPolicy(
        replicate_seed=20260721,
        candidate_pool_size=None,
        model_selection_size=8,
    )
    decision = policy.select(
        benchmark_sha256=binding.context.scope.benchmark_sha256,
        wave_index=binding.context.wave_index,
        parent_configuration_sha256=(
            request.finite_variation_contract.parent_configuration_sha256
        ),
        contract=request.finite_variation_contract,
        evaluation_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=True,
    )

    assert decision.candidate_pool_size == len(
        request.finite_variation_contract.options
    )
    assert set(decision.option_ids) == {
        value.option_id for value in request.finite_variation_contract.options
    }
    assert decision.model_selection_size == 8
    assert policy.to_record()["candidate_pool_mode"] == ("complete_finite_contract")
    policy.require_decision(
        decision,
        benchmark_sha256=binding.context.scope.benchmark_sha256,
        wave_index=binding.context.wave_index,
        parent_configuration_sha256=(
            request.finite_variation_contract.parent_configuration_sha256
        ),
        contract=request.finite_variation_contract,
        evaluation_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=True,
    )


def test_task_keyed_common_pool_rejects_membership_substitution() -> None:
    ids = DeterministicIdFactory("common_pool_membership_reject")
    request = replace(
        _boils_request(ids, require_disjoint=True),
        require_supporting_cards=False,
    )
    legacy = _binding(request)
    policy = TaskKeyedCommonCandidatePoolPolicy(replicate_seed=7)
    pool = policy.select(
        benchmark_sha256=legacy.context.scope.benchmark_sha256,
        wave_index=legacy.context.wave_index,
        parent_configuration_sha256=(
            request.finite_variation_contract.parent_configuration_sha256
        ),
        contract=request.finite_variation_contract,
        evaluation_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=True,
    )
    scope = replace(
        legacy.context.scope,
        prompt_definition_sha256=(
            CALIBRATED_PORTFOLIO_COMMON_POOL_PROMPT_DEFINITION_SHA256
        ),
    )
    binding = replace(
        legacy,
        context=replace(
            legacy.context,
            scope=scope,
            assigned_card_keys=(),
            calibration_snapshot=replace(
                legacy.context.calibration_snapshot,
                scope=scope,
            ),
        ),
        common_candidate_pool=pool,
    )
    foreign = next(
        option.option_id
        for option in request.finite_variation_contract.options
        if option.option_id not in pool.option_ids
    )

    async def invalid_runner(low_level_request: StructuredGenerationRequest[Any]):
        option_ids = (foreign, *pool.option_ids[1:])
        members = [
            {
                "option_id": option_id,
                "supporting_card_keys": [],
                "effect_predictions": [
                    {
                        "metric_id": metric_id,
                        "direction": "unknown",
                        "confidence": "unknown",
                    }
                    for metric_id in request.required_metric_ids
                ],
                "role_proposal": "coverage",
                "design_rationale": "Attempted membership substitution.",
            }
            for option_id in option_ids
        ]
        with pytest.raises(ValidationError):
            low_level_request.output_type.model_validate(
                {"members": members}, strict=True
            )
        raise AssertionError("invalid common-pool output must not reach a response")

    with pytest.raises(AssertionError, match="must not reach"):
        asyncio.run(
            PydanticAICalibratedPortfolioSelectionPolicy(
                generate_once=invalid_runner,
                binding_for=lambda _: binding,
            ).select(request)
        )
