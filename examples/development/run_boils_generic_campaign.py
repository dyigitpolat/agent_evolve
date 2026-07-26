#!/usr/bin/env python3
"""Prepare, run, or control a BOiLS delayed-identifiable campaign.

``prepare`` performs source sealing, evaluator-provenance verification, campaign
preparation, and exact provider-bound request construction without reading a
credential, calling a provider, or executing ABC.  ``live`` is the only mode
that reads ``OPENROUTER_API_KEY``.  ``control`` uses the identical campaign
runtime and evaluation schedule with an outcome-blind local selector and local
canonical reflection, so it never reads a provider credential.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from dotenv import load_dotenv  # noqa: E402

from agent_evolve.agentic import (  # noqa: E402
    BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_DEFINITION_SHA256,
    BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_ID,
    BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_VERSION,
    IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_DEFINITION_SHA256,
    IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_ID,
    IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_VERSION,
    PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256,
    PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID,
    PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION,
    EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256,
    EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID,
    EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION,
    PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_DEFINITION_SHA256,
    PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_ID,
    PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_VERSION,
    BoundedPortfolioMemoryDoseContract,
    DeterministicIdFactory,
    IdentifiableMutationReflectionContrast,
    IdentifiableReflectionEvidenceSnapshot,
    InsightDraft,
    InsightMemoryBank,
    MetricComparisonAnchor,
    MetricComparisonAnchorKind,
    MetricEffectDirection,
    MetricEffectPrediction,
    PortfolioCard,
    PortfolioExperimentalArm,
    PortfolioMemoryContextProjectionBinding,
    PortfolioMemoryContextTransferAssessment,
    PortfolioMemoryDoseCardSemantics,
    PortfolioMemoryDoseSupportScope,
    PortfolioMemoryMatchedControlWavePlan,
    PortfolioRewardAggregationBinding,
    PortfolioSelectionRequest,
    PortfolioVariationWaveRequest,
    ReflectionConsumerScope,
    ReflectionInsightKind,
    admit_portfolio_card_sources,
    compose_portfolio_evolution,
    freeze_json,
    portfolio_card_from_insight_entry,
    project_action_neutral_insight_prompt_payload,
    assess_portfolio_memory_context_transfer,
    derive_portfolio_memory_dose_card_support,
    assess_portfolio_optimization_memory,
    typed_json_sha256,
)
from agent_evolve.application.agentic_evolution import (  # noqa: E402
    EvolutionCandidate,
)
from agent_evolve.application.budgeted_optimizer import OptimizerBudget  # noqa: E402
from agent_evolve.application.calibrated_campaign import (  # noqa: E402
    CalibratedCampaignBindingFactory,
    equal_weight_slate_objectives,
)
from agent_evolve.application.campaign_evidence_registry import (  # noqa: E402
    CampaignEvidenceRegistry,
)
from agent_evolve.application.action_forecast_partitioning import (  # noqa: E402
    ConcurrentActionForecastWave,
)
from agent_evolve.application.outcome_conditioned_portfolio_selection import (  # noqa: E402
    OUTCOME_CONDITIONED_PORTFOLIO_POLICY_DEFINITION_SHA256,
    OutcomeConditionedPortfolioSelectionPolicy,
    outcome_conditioned_selected_predictions,
)
from agent_evolve.application.campaign_diagnostic_blocks import (  # noqa: E402
    CampaignDiagnosticSupportCardInput,
    CampaignDiagnosticSupportLaneInput,
)
from agent_evolve.application.portfolio_memory_matched_control import (  # noqa: E402
    PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_DEFINITION_SHA256,
    PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_ID,
    PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_VERSION,
    PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256,
    PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID,
    PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION,
    PortfolioMemoryLaneSupportResolver,
    PortfolioMemoryMatchedSupportResolution,
    PortfolioMemoryMatchedControlPlanner,
    PortfolioMemoryMatchedSupportResolver,
    materialize_portfolio_memory_matched_arm,
)
from agent_evolve.application.campaign_execution import (  # noqa: E402
    CampaignArchiveCutoffReceipt,
    CampaignExecutionEvent,
    CampaignJournalAck,
    CampaignStageRequest,
    EvolutionCampaignScheduler,
)
from agent_evolve.application.campaign_generation_audit import (  # noqa: E402
    TransactionalPortfolioGenerationAuditor,
)
from agent_evolve.application.campaign_contextual_outcomes import (  # noqa: E402
    ContextualOutcomeCampaignEnricher,
)
from agent_evolve.application.campaign_learning import (  # noqa: E402
    ClosedLoopCampaignLearning,
)
from agent_evolve.application.campaign_learning_runtime import (  # noqa: E402
    CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY,
    CampaignDiagnosticExposureReceipt,
    CampaignReflectionLearningRecordCodec,
    ClosedLoopCampaignLearningRuntime,
    StructuredCampaignReflectionLearningProjector,
)
from agent_evolve.application.identifiable_reflection_request import (  # noqa: E402
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256,
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID,
    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION,
    IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_DEFINITION_SHA256,
    IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_ID,
    IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_VERSION,
    bind_reflection_contract_to_evidence_actions,
    build_identifiable_reflection_generation_request,
    identifiable_reflection_request_construction_record,
)
from agent_evolve.application.detailed_evaluation import (  # noqa: E402
    DetailedEvaluation,
    DetailedEvaluationPayload,
    EvaluationTimings,
)
from agent_evolve.application.evolution_campaign import (  # noqa: E402
    CampaignAgentRuntimeReceipt,
    CampaignConcurrency,
    CampaignPolicies,
    CampaignPolicyBinding,
    CampaignProtocol,
    CampaignReflectionSupervisionPolicy,
    CampaignSeed,
    PreparedEvolutionCampaign,
    SealedCutoffDelayedAdmissionCadence,
    TerminalReflectionPolicy,
    EvolutionCampaign,
    ReflectionFailureMode,
)
from agent_evolve.application.evaluation_accounting import (  # noqa: E402
    CampaignEvaluationAccounting,
    CampaignPortfolioEvidenceAccounting,
)
from agent_evolve.application.finite_action_hypothesis_semantics import (  # noqa: E402
    PortableFiniteActionHypothesisMatcher,
    PortableFiniteActionInsightSemanticCompiler,
)
from agent_evolve.application.parent_measurement import (  # noqa: E402
    attach_parent_measurement_to_context,
    bind_parent_measurement,
    create_parent_measurement_projection,
)
from agent_evolve.application.portfolio_campaign_runtime import (  # noqa: E402
    AgenticPortfolioCampaignRuntime,
    ArchiveDiverseEliteCampaignParentSelector,
    ArchiveReservoirCampaignParentSelector,
    ResidualHypervolumeCampaignParentSelector,
    StagnationAwareDiverseCampaignParentSelector,
    CampaignIdentifiableReflectionInput,
    CampaignPortfolioMemoryEstimandProjection,
    CampaignPortfolioWaveContext,
    CampaignPortfolioWavePreparationReceipt,
    CommittedRegistryIdentifiableReflectionEvidenceSource,
    CAMPAIGN_ARCHIVE_CONTEXT_KEY,
    CAMPAIGN_FRONTIER_TARGET_KEY,
)
from agent_evolve.application.campaign_selector_context_extension import (  # noqa: E402
    CampaignSelectorContextExtension,
    attach_campaign_selector_context_extension,
)
from agent_evolve.application.portfolio_hypothesis_observations import (  # noqa: E402
    FinitePortfolioActionSemanticsCompiler,
    ObjectiveDeltaMetricEffectProjector,
)
from agent_evolve.application.portfolio_outcome_feedback import (  # noqa: E402
    CalibratedCampaignOutcomeUpdater,
    PortfolioOutcomeFeedbackLedger,
)
from agent_evolve.application.contextual_campaign_planning import (  # noqa: E402
    CampaignContextualSearchPlanner,
    FiniteContractContextualJointCapabilityProjector,
)
from agent_evolve.application.contextual_search_controller import (  # noqa: E402
    ContextualSearchLedger,
    audit_completed_contextual_search_ledger,
)
from agent_evolve.application.composite_outcome_updater import (  # noqa: E402
    CompositeCampaignPortfolioOutcomeUpdater,
)
from agent_evolve.application.target_conditioned_campaign import (  # noqa: E402
    TargetConditionedCampaignOutcomeUpdater,
    TargetConditionedCampaignSpecification,
)
from agent_evolve.application.portfolio_evolution import (  # noqa: E402
    PortfolioMemberDisposition,
)
from agent_evolve.application.portfolio_recombination import (  # noqa: E402
    bind_portfolio_recombination_source_utilities,
)
from agent_evolve.domain.ids import (  # noqa: E402
    CandidateId,
    LLMCallId,
    OperatorInvocationId,
)
from agent_evolve.domain.insight import InsightRef  # noqa: E402
from agent_evolve.domain.lineage import CandidateOccurrence  # noqa: E402
from agent_evolve.domain.llm_task_queue import PartitionedRetryBudget  # noqa: E402
from agent_evolve.domain.outcome import FailureCategory  # noqa: E402
from agent_evolve.domain.typed_json import (  # noqa: E402
    FrozenJsonObject,
    thaw_json,
)
from agent_evolve.infrastructure.artifacts import (  # noqa: E402
    FileSystemArtifactStore,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.action_forecast import (  # noqa: E402
    ACTION_FORECAST_POLICY_DEFINITION_SHA256,
    PydanticAIActionForecastBlockPolicy,
)
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_campaign import (  # noqa: E402
    CalibratedPortfolioCampaignCoordinator,
)
from agent_evolve.integrations.pydantic_ai.campaign_acquisition import (  # noqa: E402
    CampaignAcquisitionMode,
    build_campaign_acquisition_allocator,
    build_campaign_proposal_support_policy,
    campaign_constraint_decoupled_acquisition_from_environment,
    campaign_contextual_search_allocation_from_environment,
    campaign_evidence_calibrated_source_mix_from_environment,
    campaign_minimum_intervention_projection_from_environment,
    campaign_operator_assay_minimum_from_environment,
    campaign_residual_frontier_planning_from_environment,
    campaign_selector_policy_definition_sha256,
)
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_selection import (  # noqa: E402
    CalibratedPortfolioAllocator,
    CalibratedPortfolioFeasibilityWitnessMode,
    PydanticAIConstraintDecoupledHorizonPortfolioSelectionPolicy,
    PydanticAIConstraintDecoupledTargetConditionedPortfolioSelectionPolicy,
    PydanticAIContextualSearchAllocationPortfolioSelectionPolicy,
    PydanticAIEvidenceCalibratedSourceMixPortfolioSelectionPolicy,
    PydanticAIMinimumInterventionHorizonPortfolioSelectionPolicy,
    PydanticAIFullSupportCalibratedPortfolioSelectionPolicy,
    PydanticAIHorizonBoundedCalibratedPortfolioSelectionPolicy,
    PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy,
    PydanticAIOperatorStratifiedCalibratedPortfolioSelectionPolicy,
    PydanticAIStructuralPosteriorCalibratedPortfolioSelectionPolicy,
    PydanticAITargetConditionedCalibratedPortfolioSelectionPolicy,
    calibrated_portfolio_prompt_definition_sha256,
)
from agent_evolve.integrations.pydantic_ai.model_execution_profile import (  # noqa: E402
    OpenRouterModelExecutionProfile,
    openrouter_model_execution_profile,
)
from agent_evolve.campaign_profiles import CampaignExperimentProfile  # noqa: E402
from agent_evolve.campaign_presets import PortfolioScaleShape  # noqa: E402
from agent_evolve.campaign_variation_topology import (  # noqa: E402
    CampaignVariationTopology,
    CampaignVariationTopologyMode,
)
from agent_evolve.reference_method import (  # noqa: E402
    ReferenceCampaignImplementations,
    rebind_reference_campaign_implementations,
    reference_atomic_variation_topology_binding,
    reference_campaign_experiment_profile,
    reference_contextual_outcomes_binding,
    reference_hierarchical_r2_variation_topology_binding,
)
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (  # noqa: E402
    validate_openrouter_outbound_request_manifest_record,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (  # noqa: E402
    ProgressAwareOpenRouterConfig,
    ProgressAwareRetryMode,
    create_progress_aware_openrouter_runner,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    StructuredEvidencePublicationPolicy,
    structured_generation_outcome_record,
)
from agent_evolve.policies.memory.compatibility_matching import (  # noqa: E402
    LaneCardMatchingCard,
    LaneCardMatchingLane,
)
from agent_evolve.policies.memory import (  # noqa: E402
    StableMemoryAssignmentUnit,
)
from agent_evolve.policies.memory.global_falsification import (  # noqa: E402
    HypothesisAuditScope,
    ObservedMetricEffect,
)
from agent_evolve.policies.reward.affine_hypervolume import (  # noqa: E402
    AffineFrozenArchiveJointWaveReward,
    AffineHypervolume2DSpec,
    AffineHypervolumeArchiveUtility,
    AffineObjectiveAxis,
)
from agent_evolve.policies.reward.contextual_marginal_utility import (  # noqa: E402
    FixedReferenceContextualMarginalUtilityProjector,
)
from agent_evolve.policies.selection.finite_option_prompt_projection import (  # noqa: E402
    FiniteOptionPromptProjectionPolicy,
)
from agent_evolve.policies.selection.forecast_calibration import (  # noqa: E402
    ForecastCalibrationScope,
)
from agent_evolve.policies.selection.meaningful_direction import (  # noqa: E402
    AbsoluteToleranceDirectionAdjudicator,
    MetricDirectionResolution,
)
from agent_evolve.policies.selection.full_support_slate import (  # noqa: E402
    FullSupportSlatePolicy,
)
from agent_evolve.policies.selection.model_anchored_slate import (  # noqa: E402
    ModelAnchoredCalibratedSlatePolicy,
)
from agent_evolve.policies.selection.structural_posterior_slate import (  # noqa: E402
    HorizonBoundedStructuralPosteriorSlatePolicy,
    OperatorStratifiedStructuralPosteriorSlatePolicy,
    StructuralPosteriorSlatePolicy,
    build_controller_owned_family_exposure_phases,
    build_terminal_tapered_family_exposure_phases,
)
from agent_evolve.policies.selection.common_candidate_pool import (  # noqa: E402
    TaskKeyedCommonCandidatePoolPolicy,
)
from agent_evolve.policies.selection.affine_frontier_context import (  # noqa: E402
    AffineFrontierContextMode,
    AuthenticatedAffineFrontierContextProjector,
    affine_frontier_context_projector,
)
from agent_evolve.policies.selection.affine_frontier_target import (  # noqa: E402
    AuthenticatedAffineFrontierTargetAllocator,
    DirectionCoveredAffineFrontierTargetAllocator,
)
from agent_evolve.policies.selection.residual_frontier_target import (  # noqa: E402
    ResidualHypervolumeFrontierTargetAllocator,
)
from agent_evolve.policies.selection.target_conditioned_allocator import (  # noqa: E402
    TargetConditionedSlateAllocatorAdapter,
)
from agent_evolve.policies.selection.random_portfolio import (  # noqa: E402
    DeterministicRandomFeasiblePortfolioPolicy,
)
from agent_evolve.ports.agentic_generator import (  # noqa: E402
    AgenticCallTelemetry,
    ReflectionGenerationResult,
)
from agent_evolve.ports.action_forecast import (  # noqa: E402
    ActionForecastPartitionPolicyBinding,
)
from agent_evolve.ports.decision_metric_projection import (  # noqa: E402
    DecisionMetricProjection,
)
from agent_evolve.ports.variation_source import (  # noqa: E402
    finite_variation_source_ids,
)
from agent_evolve.ports.parent_measurement import (  # noqa: E402
    ParentMeasurementProjection,
)
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredStreamCleanupPolicy,
    StructuredStreamLivenessPolicy,
    StructuredStreamProgress,
)
from examples.benchmarks.boils_abc.campaign_reflection import (  # noqa: E402
    OBJECTIVE_IDS,
    REFLECTION_DECISION_PATHS,
    REFLECTION_OPTION_FAMILIES,
    boils_reflection_contract,
    build_boils_identifiable_reflection_learning_envelope,
    build_boils_identifiable_reflection_request,
)
from examples.benchmarks.boils_abc.campaign_workload import (  # noqa: E402
    WORKLOAD_DEFINITION_SHA256,
    WORKLOAD_ID,
    compose_boils_campaign_workload,
    shared_initial_design_workload_definition_sha256,
)
from examples.benchmarks.boils_abc.detailed_evaluation import (  # noqa: E402
    compose_boils_scientific_workload,
)
from examples.benchmarks.boils_abc.evaluator import (  # noqa: E402
    AbcEvaluatorSettings,
    BoilsAbcEvaluator,
    BoilsEvaluation,
    BoilsEvaluationFailure,
)
from examples.benchmarks.boils_abc.finite_variation_catalog import (  # noqa: E402
    FINITE_CATALOG_ID,
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.global_restart_catalog import (  # noqa: E402
    BoilsGlobalRestartVariationCatalog,
)
from agent_evolve.policies.variation.source_union_finite_catalog import (  # noqa: E402
    SourceUnionFiniteVariationCatalog,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    BatchedDurableJsonlJournal,
    DurableJsonlJournal,
    finalize_run_directory,
    read_jsonl,
    source_identity,
    write_json_atomic,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "boils_abc/generic_campaign"
)
MODEL_PROFILE_NAME = os.environ.get("AGENT_EVOLVE_MODEL_PROFILE", "deepseek")
MODEL_EXECUTION_PROFILE: OpenRouterModelExecutionProfile = (
    openrouter_model_execution_profile(MODEL_PROFILE_NAME)
)
MODEL = MODEL_EXECUTION_PROFILE.requested_model
PROVIDER_ONLY = MODEL_EXECUTION_PROFILE.provider_only
RESOLVED_PROVIDER = MODEL_EXECUTION_PROFILE.accepted_resolved_providers[0]
MAX_OUTPUT_TOKENS = MODEL_EXECUTION_PROFILE.max_output_tokens
TEMPERATURE = MODEL_EXECUTION_PROFILE.temperature
PORTFOLIO_SELECTOR_MODE = os.environ.get(
    "AGENT_EVOLVE_PORTFOLIO_SELECTOR_MODE",
    "calibrated",
)
if PORTFOLIO_SELECTOR_MODE not in {"calibrated", "outcome_conditioned"}:
    raise ValueError(
        "AGENT_EVOLVE_PORTFOLIO_SELECTOR_MODE must be calibrated or "
        "outcome_conditioned"
    )
_ACTION_FORECAST_BLOCK_ROWS_RAW = os.environ.get(
    "AGENT_EVOLVE_ACTION_FORECAST_BLOCK_ROWS",
    "32",
)
if (
    not _ACTION_FORECAST_BLOCK_ROWS_RAW.isascii()
    or not _ACTION_FORECAST_BLOCK_ROWS_RAW.isdigit()
    or not 8 <= int(_ACTION_FORECAST_BLOCK_ROWS_RAW) <= 64
):
    raise ValueError("AGENT_EVOLVE_ACTION_FORECAST_BLOCK_ROWS must lie in [8,64]")
ACTION_FORECAST_BLOCK_ROWS = int(_ACTION_FORECAST_BLOCK_ROWS_RAW)
ACTION_FORECAST_PARTITION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:generic-campaign-action-forecast-partition:v1;"
    + f"rows={ACTION_FORECAST_BLOCK_ROWS};cells-per-row=objective-count".encode(
        "ascii"
    )
).hexdigest()
FEASIBILITY_WITNESS_MODE = CalibratedPortfolioFeasibilityWitnessMode(
    os.environ.get("AGENT_EVOLVE_FEASIBILITY_WITNESS_MODE", "canonical")
)
COMMON_POOL_ACQUISITION = FEASIBILITY_WITNESS_MODE is (
    CalibratedPortfolioFeasibilityWitnessMode.TASK_KEYED_COMMON_POOL
)
ACQUISITION_MODE = CampaignAcquisitionMode(
    os.environ.get(
        "AGENT_EVOLVE_ACQUISITION_MODE",
        "model_top_k" if COMMON_POOL_ACQUISITION else "full_support",
    )
)
OPERATOR_ASSAY_MINIMUM = campaign_operator_assay_minimum_from_environment(os.environ)
CONSTRAINT_DECOUPLED_ACQUISITION = (
    campaign_constraint_decoupled_acquisition_from_environment(os.environ)
)
MINIMUM_INTERVENTION_PROJECTION = (
    campaign_minimum_intervention_projection_from_environment(os.environ)
)
EVIDENCE_CALIBRATED_SOURCE_MIX = (
    campaign_evidence_calibrated_source_mix_from_environment(os.environ)
)
CONTEXTUAL_SEARCH_ALLOCATION = campaign_contextual_search_allocation_from_environment(
    os.environ
)
RESIDUAL_FRONTIER_PLANNING = campaign_residual_frontier_planning_from_environment(
    os.environ
)
if RESIDUAL_FRONTIER_PLANNING and not CONTEXTUAL_SEARCH_ALLOCATION:
    raise ValueError(
        "residual frontier planning requires contextual search allocation"
    )
if CONTEXTUAL_SEARCH_ALLOCATION and not EVIDENCE_CALIBRATED_SOURCE_MIX:
    raise ValueError(
        "contextual search allocation requires evidence-calibrated source mix"
    )
if EVIDENCE_CALIBRATED_SOURCE_MIX and not MINIMUM_INTERVENTION_PROJECTION:
    raise ValueError("evidence-calibrated source mix requires minimum intervention")
if MINIMUM_INTERVENTION_PROJECTION and not CONSTRAINT_DECOUPLED_ACQUISITION:
    raise ValueError("minimum-intervention projection requires constraint decoupling")
if (
    CONSTRAINT_DECOUPLED_ACQUISITION
    and ACQUISITION_MODE
    not in {
        CampaignAcquisitionMode.HORIZON_BOUNDED,
        CampaignAcquisitionMode.TARGET_CONDITIONED,
    }
):
    raise ValueError(
        "constraint-decoupled acquisition requires horizon_bounded or "
        "target_conditioned mode"
    )
ARCHIVE_CONTEXT_MODE = AffineFrontierContextMode(
    os.environ.get("AGENT_EVOLVE_ARCHIVE_CONTEXT_MODE", "off")
)
_TARGET_CONDITIONED_FREEZE_PATH = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "trap_portable_profile_v1.json"
)


def _target_conditioned_specification() -> TargetConditionedCampaignSpecification:
    return TargetConditionedCampaignSpecification.from_freeze_record(
        json.loads(_TARGET_CONDITIONED_FREEZE_PATH.read_text(encoding="utf-8"))
    )


def _frontier_target_allocator():
    """Bind the target geometry to the explicit T-RAP or T-RAP+ identity."""

    if RESIDUAL_FRONTIER_PLANNING:
        return ResidualHypervolumeFrontierTargetAllocator()
    if (
        CONSTRAINT_DECOUPLED_ACQUISITION
        and ACQUISITION_MODE is CampaignAcquisitionMode.TARGET_CONDITIONED
    ):
        return DirectionCoveredAffineFrontierTargetAllocator()
    return AuthenticatedAffineFrontierTargetAllocator()


def _replicate_seed_from_environment(default: int) -> int:
    raw = os.environ.get("AGENT_EVOLVE_REPLICATE_SEED")
    if raw is None:
        return default
    if not raw.isascii() or not raw.isdigit():
        raise ValueError("AGENT_EVOLVE_REPLICATE_SEED must be decimal digits")
    value = int(raw)
    if not 0 <= value < 2**63:
        raise ValueError("AGENT_EVOLVE_REPLICATE_SEED is outside [0,2^63)")
    return value


def _candidate_pool_size_from_environment(default: int) -> int | None:
    raw = os.environ.get("AGENT_EVOLVE_COMMON_CANDIDATE_POOL_SIZE")
    if raw is None:
        return default
    if raw == "all":
        return None
    if not raw.isascii() or not raw.isdigit():
        raise ValueError(
            "AGENT_EVOLVE_COMMON_CANDIDATE_POOL_SIZE must be decimal digits or all"
        )
    value = int(raw)
    if not 8 <= value <= 4096:
        raise ValueError(
            "AGENT_EVOLVE_COMMON_CANDIDATE_POOL_SIZE must lie in [8, 4096]"
        )
    return value


def _shared_initial_design_path_from_environment() -> Path | None:
    raw = os.environ.get("AGENT_EVOLVE_SHARED_INITIAL_DESIGN_PATH")
    if raw is None:
        return None
    if not raw.strip():
        raise ValueError("AGENT_EVOLVE_SHARED_INITIAL_DESIGN_PATH cannot be empty")
    resolved = Path(raw).expanduser().resolve(strict=True)
    if resolved != WORKSPACE_ROOT and WORKSPACE_ROOT not in resolved.parents:
        raise ValueError("shared initial design must live inside the workspace")
    if not resolved.is_file():
        raise ValueError("shared initial design must be a regular file")
    return resolved


OUTER_SEED = _replicate_seed_from_environment(20_260_717)
SHARED_INITIAL_DESIGN_PATH = _shared_initial_design_path_from_environment()
GLOBAL_LOCAL_INITIAL_DESIGN = SHARED_INITIAL_DESIGN_PATH is not None
if GLOBAL_LOCAL_INITIAL_DESIGN and not COMMON_POOL_ACQUISITION:
    raise ValueError(
        "the shared-initial-design treatment requires task_keyed_common_pool"
    )
REQUIRED_SEED_COUNT = 10 if GLOBAL_LOCAL_INITIAL_DESIGN else 2
GENERATION_COUNT = 5 if GLOBAL_LOCAL_INITIAL_DESIGN else 6
PORTFOLIO_GENERATIONS = (1, 3, 5)
RECOMBINATION_GENERATIONS = (2, 4) if GLOBAL_LOCAL_INITIAL_DESIGN else (2, 4, 6)
PARENTS_PER_PORTFOLIO = 2
PORTFOLIO_WIDTH = 4 if COMMON_POOL_ACQUISITION else 8
CALIBRATED_PROPOSAL_WIDTH = 8
COMMON_CANDIDATE_POOL_SIZE = _candidate_pool_size_from_environment(8)
VARIATION_TOPOLOGY = CampaignVariationTopology.from_environment(os.environ)
if (
    ACQUISITION_MODE
    in {
        CampaignAcquisitionMode.OPERATOR_STRATIFIED,
        CampaignAcquisitionMode.HORIZON_BOUNDED,
        CampaignAcquisitionMode.TARGET_CONDITIONED,
    }
    and VARIATION_TOPOLOGY.mode is not CampaignVariationTopologyMode.HIERARCHICAL_R2
):
    raise ValueError("operator-assay acquisition requires hierarchical_r2 topology")
if (
    ACQUISITION_MODE is CampaignAcquisitionMode.OPERATOR_STRATIFIED
    and OPERATOR_ASSAY_MINIMUM > VARIATION_TOPOLOGY.required_composite_proposals
):
    raise ValueError("operator assay minimum exceeds hierarchical proposal minimum")
# Stable aliases remain in the artifact schema for compatibility with the
# already-sealed flat-composition campaigns.
COMPOSITE_OPTION_COUNT = VARIATION_TOPOLOGY.max_composite_options
COMPOSITION_SELECTION_EXPOSURE = VARIATION_TOPOLOGY.selection_exposure
REQUIRED_COMPOSITE_PROPOSALS = VARIATION_TOPOLOGY.required_composite_proposals
RECOMBINATIONS_PER_PARENT = 1 if GLOBAL_LOCAL_INITIAL_DESIGN else 2
CAMPAIGN_SCALE_SHAPE = PortfolioScaleShape(
    "g5_k4_r1" if GLOBAL_LOCAL_INITIAL_DESIGN else "g6_k4_r2",
    GENERATION_COUNT,
    PARENTS_PER_PORTFOLIO,
    PORTFOLIO_WIDTH,
    RECOMBINATIONS_PER_PARENT,
)
PLANNED_UNIQUE_EVALUATIONS = REQUIRED_SEED_COUNT + PARENTS_PER_PORTFOLIO * (
    len(PORTFOLIO_GENERATIONS) * PORTFOLIO_WIDTH
    + len(RECOMBINATION_GENERATIONS) * RECOMBINATIONS_PER_PARENT
)
MAX_CACHE_REUSE_OCCURRENCES = 6
PLANNED_LOGICAL_CALLS = 7
EVALUATOR_CONCURRENCY = 8
AGENT_CONCURRENCY = MODEL_EXECUTION_PROFILE.effective_max_connections(default=3)
AGENT_QUEUE_CAPACITY = 8
PARTITIONED_RETRY_BUDGET = (
    PartitionedRetryBudget(output_invalid_retries=2, transport_retries=2)
    if ACQUISITION_MODE
    in {
        CampaignAcquisitionMode.CALIBRATED_FRONTIER,
        CampaignAcquisitionMode.HIERARCHICAL_SUPPORT,
        CampaignAcquisitionMode.OPERATOR_STRATIFIED,
        CampaignAcquisitionMode.HORIZON_BOUNDED,
        CampaignAcquisitionMode.TARGET_CONDITIONED,
    }
    else None
)
MAX_ATTEMPTS = 5 if PARTITIONED_RETRY_BUDGET is not None else 3
FIRST_EVENT_TIMEOUT_NS = 300_000_000_000
IDLE_TIMEOUT_NS = 300_000_000_000
CLEANUP_TIMEOUT_NS = 5_000_000_000
CONNECT_TIMEOUT_SECONDS = 90.0
BASE_BACKOFF_NS = 1_000_000_000
MAX_BACKOFF_NS = 30_000_000_000


def _finite_variation_catalog():
    local = VARIATION_TOPOLOGY.decorate(BoilsFiniteVariationCatalog())
    if (
        PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
        or (
            CONSTRAINT_DECOUPLED_ACQUISITION
            and (
                ACQUISITION_MODE is CampaignAcquisitionMode.TARGET_CONDITIONED
                or CONTEXTUAL_SEARCH_ALLOCATION
            )
        )
    ):
        return SourceUnionFiniteVariationCatalog(
            primary_catalog=local,
            source_catalogs=(BoilsGlobalRestartVariationCatalog(),),
        )
    return local


def _require_pairwise_disjoint_evaluation_patches() -> bool:
    """Keep the legacy diversity constraint outside independent T-RAP+ probes."""

    return not CONSTRAINT_DECOUPLED_ACQUISITION


def _contextual_joint_capability_projector(
) -> FiniteContractContextualJointCapabilityProjector:
    """Bind generic finite-structure constraints used by this workload adapter."""

    return FiniteContractContextualJointCapabilityProjector(
        min_distinct_families=3,
        require_pairwise_disjoint_parent_patches=(
            _require_pairwise_disjoint_evaluation_patches()
        ),
        require_declared_source_floor_options=True,
    )


AGENTIC_ID_NAMESPACE = (
    "boils_shn10_gl_g5_v1"
    if GLOBAL_LOCAL_INITIAL_DESIGN
    else "boils_g6_identifiable_v1"
)
CAMPAIGN_PHASE = (
    "boils_shared_n10_global_local_g5"
    if GLOBAL_LOCAL_INITIAL_DESIGN
    else "boils_generic_g6_delayed_identifiable"
)
PROTOCOL_ID = (
    (
        "boils_generic_shared_n10_operator_stratified_g5_k4_r1_v2"
        if ACQUISITION_MODE is CampaignAcquisitionMode.OPERATOR_STRATIFIED
        else "boils_generic_shared_n10_global_local_g5_k4_r1_v1"
    )
    if GLOBAL_LOCAL_INITIAL_DESIGN
    else (
        {
            CampaignAcquisitionMode.MODEL_TOP_K: (
                "boils_generic_common_pool_model_anchored_g6_v2"
            ),
            CampaignAcquisitionMode.CALIBRATED_FRONTIER: (
                "boils_generic_common_pool_calibrated_frontier_g6_v3"
            ),
            CampaignAcquisitionMode.HIERARCHICAL_SUPPORT: (
                "boils_generic_hierarchical_support_g6_v4"
            ),
            CampaignAcquisitionMode.OPERATOR_STRATIFIED: (
                "boils_generic_operator_stratified_g6_v5"
            ),
            CampaignAcquisitionMode.HORIZON_BOUNDED: (
                "boils_generic_horizon_bounded_g6_v1"
            ),
            CampaignAcquisitionMode.TARGET_CONDITIONED: (
                "boils_generic_target_conditioned_g6_v1"
            ),
        }[ACQUISITION_MODE]
        if COMMON_POOL_ACQUISITION
        else "boils_generic_calibrated_g6_delayed_identifiable_v2"
    )
)
TASK_SHA256 = hashlib.sha256(
    (
        b"agent-evolve:boils-shared-n10-global-local-g5-k4-r1-task:v1"
        if GLOBAL_LOCAL_INITIAL_DESIGN
        else b"agent-evolve:boils-generic-production-learning-g6-randomized-credit-task:v2"
    )
).hexdigest()
METRIC_ADJUDICATOR_SHA256 = hashlib.sha256(
    b"agent-evolve:boils-objective-delta-adjudicator:v1"
).hexdigest()
CONSTRUCTION_PROBE_ID = (
    "boils_shared_n10_global_local_g5_construction_probe"
    if GLOBAL_LOCAL_INITIAL_DESIGN
    else "boils_g6_delayed_identifiable_construction_probe"
)
CONSTRUCTION_PROBE_VERSION = 4
CONSTRUCTION_PROBE_DEFINITION_SHA256 = hashlib.sha256(
    (
        b"agent-evolve:boils-shared-n10-global-local-g5-construction-probe:v2;"
        if GLOBAL_LOCAL_INITIAL_DESIGN
        else b"agent-evolve:boils-g6-delayed-identifiable-construction-probe:v4;"
    )
    + b"six-bootstrap-waves;two-synthetic-g5-bounded-dose-waves;"
    + b"sealed-g1-direct-mutation-reflection;exact-action-complete-support;"
    + b"randomized-singleton-credit;provider-evaluator-credential-free;"
    + b"contextual-finite-variation-source-completeness;"
    + b"prospective-joint-capability-evidence"
).hexdigest()
LEVEL_REFERENCE = 80.0
LUT_REFERENCE = 12_000.0
MEMORY_AGGREGATION_ID = "affine_archive_joint_wave_gain"
MEMORY_AGGREGATION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:affine-archive-joint-wave-gain:v1"
).hexdigest()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    if type(result) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("BOiLS campaign record did not freeze to an object")
    return result


_SHARED_INITIAL_DESIGN_DOMAIN = b"agent-evolve:shared-initial-design:v1\x00"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _load_shared_initial_design() -> tuple[
    tuple[CampaignSeed, ...] | None,
    FrozenJsonObject | None,
]:
    """Load a configuration-only, source-sealed common initial design."""

    path = SHARED_INITIAL_DESIGN_PATH
    if path is None:
        return None, None
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("shared initial design must be strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise TypeError("shared initial design root must be an object")
    required_keys = {
        "candidate_count",
        "candidates",
        "design_id",
        "design_sha256",
        "design_version",
        "excluded_source_fields",
        "outcome_blind_projection",
        "schema_version",
        "source",
        "source_kind",
        "source_seed",
        "workload_id",
    }
    if set(value) != required_keys:
        raise ValueError("shared initial design uses an unexpected root schema")
    if value["schema_version"] != 1 or value["design_version"] != 1:
        raise ValueError("unsupported shared initial-design schema/version")
    if value["workload_id"] != WORKLOAD_ID:
        raise ValueError("shared initial design targets a different workload")
    if value["outcome_blind_projection"] is not True:
        raise ValueError("shared initial design is not an outcome-blind projection")
    if value["source_seed"] != OUTER_SEED:
        raise ValueError("shared initial-design source seed differs from outer seed")
    candidates = value["candidates"]
    if type(candidates) is not list or len(candidates) != REQUIRED_SEED_COUNT:
        raise ValueError(
            "shared initial design must contain the preregistered seed count"
        )
    if value["candidate_count"] != len(candidates):
        raise ValueError("shared initial-design candidate count is inconsistent")
    claimed_design_sha256 = value["design_sha256"]
    unsigned = dict(value)
    del unsigned["design_sha256"]
    observed_design_sha256 = hashlib.sha256(
        _SHARED_INITIAL_DESIGN_DOMAIN + _canonical_json_bytes(unsigned)
    ).hexdigest()
    if claimed_design_sha256 != observed_design_sha256:
        raise ValueError("shared initial-design identity does not verify")

    seeds: list[CampaignSeed] = []
    identities: set[str] = set()
    for ordinal, candidate in enumerate(candidates, start=1):
        if type(candidate) is not dict or set(candidate) != {
            "canonical_json_sha256",
            "configuration",
            "seed_id",
            "source_suggestion_id",
            "source_suggestion_ordinal",
        }:
            raise ValueError(
                f"shared initial-design candidate {ordinal} has an invalid schema"
            )
        configuration = candidate["configuration"]
        if type(configuration) is not dict:
            raise TypeError("shared initial-design configuration must be an object")
        identity = hashlib.sha256(_canonical_json_bytes(configuration)).hexdigest()
        if candidate["canonical_json_sha256"] != identity:
            raise ValueError("shared initial-design candidate identity mismatch")
        if identity in identities:
            raise ValueError("shared initial design contains duplicate candidates")
        identities.add(identity)
        seeds.append(
            CampaignSeed(
                seed_id=candidate["seed_id"],
                configuration=_object(configuration),
            )
        )
    artifact_sha256 = hashlib.sha256(payload).hexdigest()
    metadata = _object(
        {
            "schema_version": 1,
            "mode": "shared_outcome_blind_initial_design",
            "design_id": value["design_id"],
            "design_version": value["design_version"],
            "design_sha256": claimed_design_sha256,
            "artifact_path": path.relative_to(WORKSPACE_ROOT).as_posix(),
            "artifact_sha256": artifact_sha256,
            "candidate_count": len(seeds),
            "source_kind": value["source_kind"],
            "source_seed": value["source_seed"],
            "outcome_blind_projection": True,
        }
    )
    return tuple(seeds), metadata


def _canonical_json_size(value: object) -> int:
    return len(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )


def _telemetry_record(value: AgenticCallTelemetry) -> dict[str, object]:
    return {
        "requested_model": value.requested_model,
        "resolved_model": value.resolved_model,
        "resolved_provider": value.resolved_provider,
        "provider_response_id": value.provider_response_id,
        "finish_reason": value.finish_reason,
        "input_tokens": value.input_tokens,
        "output_tokens": value.output_tokens,
        "reasoning_tokens": value.reasoning_tokens,
        "cache_read_tokens": value.cache_read_tokens,
        "cache_write_tokens": value.cache_write_tokens,
        "cost_usd": None if value.cost_usd is None else str(value.cost_usd),
        "latency_ns": value.latency_ns,
        "attempt_count": value.attempt_count,
    }


def _progress_record(value: StructuredStreamProgress) -> dict[str, object]:
    if type(value) is not StructuredStreamProgress:
        raise TypeError("value must be an exact StructuredStreamProgress")
    value.__post_init__()
    return {
        "call_id": value.call_id,
        "provider_attempt_id": value.provider_attempt_id,
        "sequence": value.sequence,
        "kind": value.kind.value,
        "channel": value.channel.value,
        "elapsed_ns": value.elapsed_ns,
        "event_content_utf8_bytes": value.event_content_utf8_bytes,
        "cumulative_content_utf8_bytes": value.cumulative_content_utf8_bytes,
        "rolling_content_sha256": value.rolling_content_sha256,
    }


def _compatibility_audit_sha256(rows: tuple[dict[str, str], ...]) -> str:
    """Hash a typed-JSON audit without leaking tuple containers downstream."""

    if type(rows) is not tuple:
        raise TypeError("rows must be an exact tuple")
    if any(
        type(row) is not dict
        or any(
            type(key) is not str or type(value) is not str for key, value in row.items()
        )
        for row in rows
    ):
        raise TypeError("audit rows must be exact string dictionaries")
    return typed_json_sha256(freeze_json({"schema_version": 1, "rows": list(rows)}))


def _expected_forecast_feedback_counts(arm: str) -> tuple[int, int]:
    """Return receipt and per-metric observation counts for one G6 run."""

    if arm == "control":
        return 0, 0
    if arm != "live":
        raise ValueError("arm must be live or control")
    selector_receipts = len(PORTFOLIO_GENERATIONS) * PARENTS_PER_PORTFOLIO
    observations = selector_receipts * PORTFOLIO_WIDTH * len(OBJECTIVE_IDS)
    return selector_receipts, observations


def _provider_response_telemetry_gate(
    *, arm: str, outcome_rows: tuple[dict[str, object], ...]
) -> bool:
    """Verify terminal route and profile-specific reasoning for every live call."""

    if arm == "control":
        return not outcome_rows
    if arm != "live" or (
        len(outcome_rows) < PLANNED_LOGICAL_CALLS
        if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
        else len(outcome_rows) != PLANNED_LOGICAL_CALLS
    ):
        return False
    for row in outcome_rows:
        authenticated = row.get("authenticated_record")
        if (
            type(authenticated) is not dict
            or authenticated.get("status") != "succeeded"
        ):
            return False
        response = authenticated.get("response")
        if type(response) is not dict:
            return False
        if (
            response.get("requested_model") != MODEL
            or response.get("resolved_model")
            not in MODEL_EXECUTION_PROFILE.accepted_resolved_models
            or response.get("resolved_provider")
            not in MODEL_EXECUTION_PROFILE.accepted_resolved_providers
            or response.get("finish_reason")
            not in MODEL_EXECUTION_PROFILE.accepted_finish_reasons
            or not MODEL_EXECUTION_PROFILE.accepts_reasoning_tokens(
                response.get("reasoning_tokens")
            )
            or type(response.get("provider_response_id")) is not str
            or not response["provider_response_id"]
        ):
            return False
    return True


def _affinity_sets(count: int) -> tuple[tuple[int, ...], ...]:
    if type(count) is not int or count <= 0:
        raise ValueError("evaluator concurrency must be a positive integer")
    allowed = tuple(sorted(os.sched_getaffinity(0)))
    if len(allowed) < count:
        raise RuntimeError("insufficient CPU affinity for BOiLS evaluator leases")
    return tuple((cpu,) for cpu in allowed[-count:])


def _source_paths() -> tuple[Path, ...]:
    core = tuple(sorted((AGENT_EVOLVE_ROOT / "src/agent_evolve").rglob("*.py")))
    workload = tuple(
        sorted((AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc").glob("*.py"))
    )
    locks = tuple(
        path
        for path in (
            AGENT_EVOLVE_ROOT / "pyproject.toml",
            AGENT_EVOLVE_ROOT / "uv.lock",
        )
        if path.is_file()
    )
    initial_design = (
        () if SHARED_INITIAL_DESIGN_PATH is None else (SHARED_INITIAL_DESIGN_PATH,)
    )
    return (
        Path(__file__),
        AGENT_EVOLVE_ROOT / "examples/development/durable_run_artifacts.py",
        *locks,
        *core,
        *workload,
        *initial_design,
    )


def _snapshot_sources(run_dir: Path, paths: tuple[Path, ...]) -> dict[str, object]:
    root = run_dir / "source_snapshot"
    root.mkdir(exist_ok=False)
    aggregate = hashlib.sha256(b"agent-evolve:source-set:v1\x00")
    rows: list[dict[str, object]] = []
    labels: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve(strict=True)
        label = resolved.relative_to(WORKSPACE_ROOT).as_posix()
        if label in labels:
            raise ValueError("source snapshot paths must be unique")
        labels.add(label)
        content = resolved.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        destination = root / label
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        label_bytes = label.encode("utf-8", errors="strict")
        aggregate.update(len(label_bytes).to_bytes(8, "big"))
        aggregate.update(label_bytes)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
        rows.append({"path": label, "size_bytes": len(content), "sha256": digest})
    return {
        "schema_version": 1,
        "snapshot_directory": "source_snapshot",
        "file_count": len(rows),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": rows,
    }


def _require_source_closure(expected_sha256: str) -> dict[str, object]:
    current = source_identity(_source_paths(), relative_to=WORKSPACE_ROOT)
    if current["aggregate_sha256"] != expected_sha256:
        raise RuntimeError("sealed campaign source changed")
    return current


def _option_prompt_projection() -> FiniteOptionPromptProjectionPolicy:
    return FiniteOptionPromptProjectionPolicy(
        metadata_keys=VARIATION_TOPOLOGY.prompt_metadata_keys(
            ("abc_commands_json", "position", "replacement_action")
        )
    )


def _hierarchical_prompt_composition_count() -> int | None:
    return VARIATION_TOPOLOGY.hierarchical_composition_required_proposals


def _reference_variation_topology_binding(implementation: object):
    if VARIATION_TOPOLOGY.mode is CampaignVariationTopologyMode.ATOMIC:
        return reference_atomic_variation_topology_binding(implementation)
    if VARIATION_TOPOLOGY.mode is CampaignVariationTopologyMode.HIERARCHICAL_R2:
        return reference_hierarchical_r2_variation_topology_binding(
            implementation,
            max_composite_options=VARIATION_TOPOLOGY.max_composite_options,
            required_composite_proposals=(
                VARIATION_TOPOLOGY.required_composite_proposals
            ),
        )
    return None


def _allocator() -> CalibratedPortfolioAllocator:
    return build_campaign_acquisition_allocator(
        ACQUISITION_MODE,
        common_pool_enabled=COMMON_POOL_ACQUISITION,
        operator_assay_minimum=OPERATOR_ASSAY_MINIMUM,
        family_exposure_phases=(
            build_controller_owned_family_exposure_phases(
                family="composite_r2",
            )
            if CONTEXTUAL_SEARCH_ALLOCATION
            else build_terminal_tapered_family_exposure_phases(
                family="composite_r2",
                terminal_wave_index=PORTFOLIO_GENERATIONS[-1],
            )
            if ACQUISITION_MODE is CampaignAcquisitionMode.HORIZON_BOUNDED
            else None
        ),
        target_conditioned_profile=(
            _target_conditioned_specification().profile
            if ACQUISITION_MODE is CampaignAcquisitionMode.TARGET_CONDITIONED
            else None
        ),
    )


def _proposal_support_policy():
    return build_campaign_proposal_support_policy(
        ACQUISITION_MODE,
        common_pool_enabled=COMMON_POOL_ACQUISITION,
    )


def _selector_policy_definition_sha256() -> str:
    return campaign_selector_policy_definition_sha256(
        _allocator(),
        constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
        minimum_intervention_projection=MINIMUM_INTERVENTION_PROJECTION,
        evidence_calibrated_source_mix=EVIDENCE_CALIBRATED_SOURCE_MIX,
        contextual_search_allocation=CONTEXTUAL_SEARCH_ALLOCATION,
    )


def _selection_contract_label() -> str:
    if not COMMON_POOL_ACQUISITION:
        return "calibrated_full_support_k8_to_k8"
    base = {
        CampaignAcquisitionMode.MODEL_TOP_K: (
            "task_keyed_common_pool_model_anchored_k8_to_k4"
        ),
        CampaignAcquisitionMode.CALIBRATED_FRONTIER: (
            "task_keyed_common_pool_structural_posterior_k8_to_k4"
        ),
        CampaignAcquisitionMode.HIERARCHICAL_SUPPORT: (
            "hierarchical_support_structural_posterior_k8_to_k4"
        ),
        CampaignAcquisitionMode.OPERATOR_STRATIFIED: (
            "operator_stratified_hierarchical_k8_to_k4"
        ),
        CampaignAcquisitionMode.HORIZON_BOUNDED: (
            "horizon_bounded_hierarchical_k8_to_k4"
        ),
        CampaignAcquisitionMode.TARGET_CONDITIONED: (
            "constraint_decoupled_target_conditioned_realizable_k8_to_k4"
            if CONSTRAINT_DECOUPLED_ACQUISITION
            else "target_conditioned_realizable_k8_to_k4"
        ),
    }[ACQUISITION_MODE]
    return f"contextual_source_operator_{base}" if CONTEXTUAL_SEARCH_ALLOCATION else base


def _common_candidate_pool_policy() -> TaskKeyedCommonCandidatePoolPolicy | None:
    if FEASIBILITY_WITNESS_MODE is not (
        CalibratedPortfolioFeasibilityWitnessMode.TASK_KEYED_COMMON_POOL
    ):
        return None
    return TaskKeyedCommonCandidatePoolPolicy(
        replicate_seed=OUTER_SEED,
        candidate_pool_size=COMMON_CANDIDATE_POOL_SIZE,
        model_selection_size=CALIBRATED_PROPOSAL_WIDTH,
    )


def _model_profile_sha256() -> str:
    return MODEL_EXECUTION_PROFILE.profile_sha256


def _provider_config() -> ProgressAwareOpenRouterConfig:
    return ProgressAwareOpenRouterConfig(
        model_name=MODEL,
        provider_only=PROVIDER_ONLY,
        connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS,
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=FIRST_EVENT_TIMEOUT_NS,
            idle_timeout_ns=IDLE_TIMEOUT_NS,
            absolute_timeout_ns=None,
            cleanup_policy=StructuredStreamCleanupPolicy(
                cancel_drain_timeout_ns=CLEANUP_TIMEOUT_NS,
                transport_retire_timeout_ns=CLEANUP_TIMEOUT_NS,
            ),
        ),
        max_connections=AGENT_CONCURRENCY,
        max_pending=AGENT_QUEUE_CAPACITY,
        max_attempts=MAX_ATTEMPTS,
        retry_budget=PARTITIONED_RETRY_BUDGET,
        base_backoff_ns=BASE_BACKOFF_NS,
        max_backoff_ns=MAX_BACKOFF_NS,
        rate_limit_backoff_floor_ns=(
            MODEL_EXECUTION_PROFILE.rate_limit_backoff_floor_ns
        ),
        jitter_seed=OUTER_SEED,
        jitter_domain=f"{CAMPAIGN_PHASE}-openrouter-queue-v1",
        app_title=("AgentEvolve AAAI 2027 BOiLS delayed-identifiable campaign"),
        reasoning_config=MODEL_EXECUTION_PROFILE.reasoning_config,
        structured_output_mode=(MODEL_EXECUTION_PROFILE.structured_output_mode),
        structured_output_strict=(MODEL_EXECUTION_PROFILE.structured_output_strict),
        json_schema_dialect=MODEL_EXECUTION_PROFILE.json_schema_dialect,
        provider_require_parameters=(
            MODEL_EXECUTION_PROFILE.provider_require_parameters
        ),
        supports_forced_tool_choice=(
            MODEL_EXECUTION_PROFILE.supports_forced_tool_choice
        ),
        retry_mode=(ProgressAwareRetryMode.OPAQUE_HTTP_400_AND_BOUNDED_SCHEMA_REPAIR),
    )


def _affine_spec() -> AffineHypervolume2DSpec:
    """Freeze the legacy BOiLS/log2 reference prospectively for both arms."""

    return AffineHypervolume2DSpec(
        axes=(
            AffineObjectiveAxis("total_levels", "min", 0.0, LEVEL_REFERENCE),
            AffineObjectiveAxis("total_lut_count", "min", 0.0, LUT_REFERENCE),
        ),
        reference_provenance=(
            "prospective BOiLS/log2 development reference reused from sealed "
            "portfolio-Q control protocol: levels=80, LUTs=12000; any candidate "
            "outside the reference receives zero hypervolume measure and is also "
            "reported as a campaign health failure"
        ),
    )


@dataclass(frozen=True, slots=True)
class _RecombinationUtilityBinder:
    utility: AffineHypervolumeArchiveUtility

    def bind(self, *, source_archive_utility, source_wave, source_result):
        snapshot = self.utility.require_snapshot(source_archive_utility)
        candidates = {
            value.candidate_id: value for value in source_result.scored_candidates
        }
        marginal = {
            candidate_id: snapshot.marginal_gain(candidate.objective_map)
            for candidate_id, candidate in candidates.items()
        }
        pair = {
            pair_ids: snapshot.joint_gain(
                tuple(candidates[value].objective_map for value in pair_ids)
            )
            for pair_ids in combinations(sorted(candidates), 2)
        }
        return bind_portfolio_recombination_source_utilities(
            snapshot=source_archive_utility,
            source_wave=source_wave,
            source_result=source_result,
            marginal_utilities=marginal,
            exact_pair_utilities=pair,
        )


class _PreparationRuntime:
    def __init__(self, *, source_closure_sha256: str, arm: str) -> None:
        self.source_closure_sha256 = source_closure_sha256
        self.arm = arm

    def prepare(self, request):
        return CampaignAgentRuntimeReceipt(
            request_sha256=request.request_sha256,
            runtime_id="boils_progress_aware_campaign",
            runtime_version=1,
            definition_sha256=_sha("boils-progress-aware-campaign-v1"),
            accepted=True,
            evidence=_object(
                {
                    "provider_calls": 0,
                    "credential_read": False,
                    "abc_executions": 0,
                    "arm": self.arm,
                    "provider_config": _provider_config().to_manifest_record(),
                    "source_closure_sha256": self.source_closure_sha256,
                }
            ),
        )


class _PreparationJournal:
    def __init__(self, journal: DurableJsonlJournal) -> None:
        self.journal = journal

    def append(self, record) -> None:
        self.journal.append(thaw_json(record))


class _ExecutionJournal:
    def __init__(self, journal: DurableJsonlJournal, started_ns: int) -> None:
        self.journal = journal
        self.started_ns = started_ns

    async def append(self, event: CampaignExecutionEvent):
        self.journal.append(
            {
                "observation": {
                    "monotonic_ns_since_execution_start": (
                        time.perf_counter_ns() - self.started_ns
                    ),
                    "observed_at_utc": _utc_now(),
                },
                "authenticated_campaign_event": event.to_record(),
            }
        )
        return CampaignJournalAck(event.event_sha256, True)


class _WavePreparationJournal:
    def __init__(self, journal: DurableJsonlJournal, started_ns: int) -> None:
        self.journal = journal
        self.started_ns = started_ns

    def record_prepared_wave(
        self, receipt: CampaignPortfolioWavePreparationReceipt
    ) -> None:
        self.journal.append(
            {
                "observation": {
                    "monotonic_ns_since_execution_start": (
                        time.perf_counter_ns() - self.started_ns
                    ),
                    "observed_at_utc": _utc_now(),
                },
                "authenticated_wave_preparation": receipt.to_record(),
            }
        )


@dataclass(slots=True)
class _OwnedRunner:
    runner: Any

    async def close(self):
        await self.runner.aclose()
        snapshot = await self.runner.snapshot()
        return _object(
            {
                "ownership": "campaign_runtime",
                "runner_closed": bool(snapshot.closed),
                "pending": snapshot.pending,
                "in_flight": snapshot.in_flight,
            }
        )


class _EvaluatorObserver:
    def __init__(self, journal: DurableJsonlJournal) -> None:
        self.journal = journal
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, result: BoilsEvaluation | BoilsEvaluationFailure) -> None:
        record = result.as_dict() if type(result) is BoilsEvaluation else asdict(result)
        with self._lock:
            self.calls += 1
            self.journal.append(
                {
                    "schema_version": 1,
                    "observation_ordinal": self.calls,
                    "observation_kind": type(result).__name__,
                    "observed_at_utc": _utc_now(),
                    "evaluation": record,
                }
            )


def _binding(name: str, implementation: object) -> CampaignPolicyBinding:
    return CampaignPolicyBinding(
        implementation=implementation,
        policy_id=name,
        policy_version=1,
        definition_sha256=_sha(f"boils-generic-campaign-policy:{name}:v1"),
    )


@dataclass(frozen=True, slots=True)
class _Bundle:
    arm: str
    settings: AbcEvaluatorSettings
    evaluator: Any
    evaluator_observer: _EvaluatorObserver
    benchmark: Any
    config: Any
    shared_initial_design: FrozenJsonObject | None
    workload_ports: Any
    prepared: PreparedEvolutionCampaign
    policies: CampaignPolicies
    experiment_profile: CampaignExperimentProfile | None
    ids: DeterministicIdFactory
    memory: InsightMemoryBank
    seed_card: PortfolioCard
    parent_selector: (
        ArchiveDiverseEliteCampaignParentSelector
        | StagnationAwareDiverseCampaignParentSelector
        | ResidualHypervolumeCampaignParentSelector
        | ArchiveReservoirCampaignParentSelector
    )
    utility: AffineHypervolumeArchiveUtility
    binding_factory: CalibratedCampaignBindingFactory
    coordinator: CalibratedPortfolioCampaignCoordinator
    feedback_ledger: PortfolioOutcomeFeedbackLedger
    direction_adjudicator: AbsoluteToleranceDirectionAdjudicator
    parent_measurement_projection: ParentMeasurementProjection


def _target_conditioned_controller(
    bundle: _Bundle,
    coordinator: CalibratedPortfolioCampaignCoordinator,
) -> TargetConditionedCampaignOutcomeUpdater | None:
    if (
        bundle.arm != "live"
        or ACQUISITION_MODE is not CampaignAcquisitionMode.TARGET_CONDITIONED
    ):
        return None
    specification = _target_conditioned_specification()
    if type(coordinator.allocator) is not TargetConditionedSlateAllocatorAdapter:
        raise TypeError("BOiLS target-conditioned coordinator has a foreign allocator")
    if coordinator.allocator.profile != specification.profile:
        raise ValueError("BOiLS target-conditioned profile drifted")
    return TargetConditionedCampaignOutcomeUpdater(
        state=specification.initial_state(
            campaign_scope_sha256=bundle.prepared.preparation_sha256
        ),
        selected_decision=(
            lambda wave, result: coordinator.decode_target_conditioned_allocation(
                result
            )
        ),
        selected_context=(
            lambda wave, result: coordinator.decode_target_conditioned_context(result)
        ),
        marginal_utility=FixedReferenceContextualMarginalUtilityProjector(
            bundle.utility
        ),
    )


def _seed_memory(
    memory: InsightMemoryBank,
) -> PortfolioCard:
    entry = memory.extend(
        (
            InsightDraft(
                claim="Explore diverse sealed BOiLS single-position actions.",
                trigger="A parent-local finite BOiLS catalog is available.",
                mechanism=(
                    "Diverse action families and positions test distinct logic "
                    "restructuring mechanisms while the engine owns materialization."
                ),
                affected_paths=("$.sequence[0]",),
                evidence_summary="Prospective non-empirical bootstrap prior.",
                confidence=0.5,
            ),
        ),
        initial_score=0.0,
        applicable_operator_kinds=("typed_mutation",),
    )[0]
    return PortfolioCard(
        card_key="card.boils.bootstrap",
        reference=entry.reference,
        content_sha256=entry.draft.content_sha256,
        evidence_sha256=_sha("boils-generic-bootstrap-evidence-v1"),
        prompt_payload=_object(
            {
                "epistemic_status": "prior_hypothesis",
                "claim": entry.draft.claim,
                "adaptive_score_consumption": False,
            }
        ),
        assigned_score=0.0,
    )


def _calibration_scope(prepared: PreparedEvolutionCampaign) -> ForecastCalibrationScope:
    projection = _option_prompt_projection()
    return ForecastCalibrationScope(
        model_profile_sha256=_model_profile_sha256(),
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            projection,
            proposal_support=_proposal_support_policy() is not None,
            hierarchical_composition_required_proposals=(
                _hierarchical_prompt_composition_count()
            ),
            feasibility_witness_mode=FEASIBILITY_WITNESS_MODE,
            constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
        ),
        selector_policy_definition_sha256=_selector_policy_definition_sha256(),
        benchmark_sha256=typed_json_sha256(prepared.benchmark_session.benchmark),
        session_sha256=prepared.benchmark_session.session_sha256,
    )


def _outcome_conditioned_selector(
    *,
    runner: Any,
    bundle: _Bundle,
) -> OutcomeConditionedPortfolioSelectionPolicy:
    """Compose the workload-neutral all-action policy for BOiLS.

    BOiLS deliberately supplies neither a hand-authored action glossary nor an
    exact action-effect projector here.  The generic layer derives structural
    semantics from the sealed finite contract and asks the model to forecast
    both objectives.  This keeps the assay a direct portability test rather
    than a domain-knowledge shortcut.
    """

    semantics = bundle.benchmark.optimization_semantics
    if semantics is None:
        raise TypeError("BOiLS outcome-conditioned selector requires semantics")
    return OutcomeConditionedPortfolioSelectionPolicy(
        forecaster=ConcurrentActionForecastWave(
            block_policy=PydanticAIActionForecastBlockPolicy(runner),
            max_concurrency=AGENT_CONCURRENCY,
        ),
        optimization_semantics=semantics,
        partition_policy=ActionForecastPartitionPolicyBinding(
            policy_id="generic_campaign_action_forecast_blocks",
            policy_version=1,
            policy_definition_sha256=(
                ACTION_FORECAST_PARTITION_DEFINITION_SHA256
            ),
            max_rows_per_block=ACTION_FORECAST_BLOCK_ROWS,
            max_metric_cells_per_block=(
                ACTION_FORECAST_BLOCK_ROWS * len(OBJECTIVE_IDS)
            ),
        ),
        action_semantics_factory=None,
        metric_projector=None,
        risk_aversion=0.5,
        diversity_weight=0.05,
        beam_width=256,
    )


def _outcome_conditioned_calibration_scope(
    bundle: _Bundle,
) -> ForecastCalibrationScope:
    """Isolate feedback by consequence prompt and selector identity."""

    source = bundle.binding_factory.scope
    source.revalidate()
    return ForecastCalibrationScope(
        model_profile_sha256=source.model_profile_sha256,
        prompt_definition_sha256=ACTION_FORECAST_POLICY_DEFINITION_SHA256,
        selector_policy_definition_sha256=(
            OUTCOME_CONDITIONED_PORTFOLIO_POLICY_DEFINITION_SHA256
        ),
        benchmark_sha256=source.benchmark_sha256,
        session_sha256=source.session_sha256,
    )


def _prepare_bundle(
    *,
    run_dir: Path,
    preparation_journal: DurableJsonlJournal,
    evaluator_journal: DurableJsonlJournal,
    source_closure_sha256: str,
    arm: str,
    evaluator_factory: Any = BoilsAbcEvaluator,
) -> _Bundle:
    if arm not in {"live", "control"}:
        raise ValueError("arm must be live or control")
    affinity_sets = _affinity_sets(EVALUATOR_CONCURRENCY)
    settings = AbcEvaluatorSettings.current_circuit_panel(
        circuit_names=("log2",),
        affinity_sets=affinity_sets,
        per_circuit_timeout_s=60.0,
    )
    evaluator_observer = _EvaluatorObserver(evaluator_journal)
    evaluator = evaluator_factory(settings, observer=evaluator_observer)
    if not hasattr(evaluator, "evaluate") or not hasattr(evaluator, "provenance"):
        raise TypeError("evaluator_factory must produce the BOiLS evaluator port")
    finite_variation_catalog = _finite_variation_catalog()
    scientific = compose_boils_scientific_workload(
        settings,
        artifact_store=FileSystemArtifactStore(run_dir / "evaluator_receipts"),
        evaluator=evaluator,
        finite_variation_catalog=finite_variation_catalog,
    )
    benchmark = scientific.benchmark
    shared_seeds, shared_initial_design = _load_shared_initial_design()
    shared_design_record = (
        None if shared_initial_design is None else thaw_json(shared_initial_design)
    )
    config = compose_boils_campaign_workload(
        benchmark=benchmark,
        evaluator_preflight_receipt=_object(
            {
                "qualified": True,
                "mode": "cryptographic_startup_preflight_no_candidate_execution",
                "provider_calls": 0,
                "abc_executions": 0,
                "provenance": evaluator.provenance(),
            }
        ),
        resource_lease_receipt=_object(
            {
                "resource": "exclusive_cpu_affinity_slots",
                "active": True,
                "affinity_sets": [list(value) for value in affinity_sets],
            }
        ),
        evaluator_concurrency_cap=EVALUATOR_CONCURRENCY,
        seeds=shared_seeds,
        seed_design_sha256=(
            None
            if shared_design_record is None
            else str(shared_design_record["design_sha256"])
        ),
    )
    workload_ports = config.build_ports()
    ids = DeterministicIdFactory(f"{AGENTIC_ID_NAMESPACE}_{arm}")
    memory = InsightMemoryBank(id_factory=ids)
    seed_card = _seed_memory(memory)
    parent_selector = (
        ResidualHypervolumeCampaignParentSelector()
        if arm == "live" and RESIDUAL_FRONTIER_PLANNING
        else StagnationAwareDiverseCampaignParentSelector()
        if (
            arm == "live"
            and COMMON_POOL_ACQUISITION
            and ACQUISITION_MODE is CampaignAcquisitionMode.HORIZON_BOUNDED
        )
        else ArchiveDiverseEliteCampaignParentSelector()
        if arm == "live" and COMMON_POOL_ACQUISITION
        else ArchiveReservoirCampaignParentSelector(reservoir_limit=8)
    )
    utility = AffineHypervolumeArchiveUtility(_affine_spec())
    archive_context_projector = (
        AuthenticatedAffineFrontierContextProjector()
        if arm == "live"
        and ACQUISITION_MODE is CampaignAcquisitionMode.TARGET_CONDITIONED
        else affine_frontier_context_projector(ARCHIVE_CONTEXT_MODE)
    )
    selector_policy = (
        _allocator()
        if arm == "live"
        else DeterministicRandomFeasiblePortfolioPolicy(seed=OUTER_SEED)
    )
    experiment_profile: CampaignExperimentProfile | None = None
    if arm == "live" and COMMON_POOL_ACQUISITION and COMMON_CANDIDATE_POOL_SIZE is None:
        context_local_successor = (
            _proposal_support_policy() is not None
            and archive_context_projector is not None
            and VARIATION_TOPOLOGY.mode is not CampaignVariationTopologyMode.FLAT_R2
        )
        experiment_profile = reference_campaign_experiment_profile(
            profile_id=f"reference_boils_{MODEL_EXECUTION_PROFILE.profile_id}",
            model_execution=MODEL_EXECUTION_PROFILE,
            implementations=ReferenceCampaignImplementations(
                parent_selection=parent_selector,
                memory_assignment=memory,
                portfolio_selection=selector_policy,
                recombination=object(),
                reflection=object(),
                archive_context=archive_context_projector,
                variation_topology=(
                    _reference_variation_topology_binding(finite_variation_catalog)
                    if context_local_successor
                    else None
                ),
                contextual_outcomes=(
                    reference_contextual_outcomes_binding(object())
                    if context_local_successor
                    else None
                ),
            ),
            candidate_pool_size=COMMON_CANDIDATE_POOL_SIZE,
            evaluator_concurrency=EVALUATOR_CONCURRENCY,
            agent_concurrency=AGENT_CONCURRENCY,
            agent_queue_capacity=AGENT_QUEUE_CAPACITY,
            hierarchical_proposal_support=(_proposal_support_policy() is not None),
            operator_stratified_acquisition=(
                ACQUISITION_MODE is CampaignAcquisitionMode.OPERATOR_STRATIFIED
            ),
            horizon_bounded_acquisition=(
                ACQUISITION_MODE is CampaignAcquisitionMode.HORIZON_BOUNDED
            ),
            target_conditioned_acquisition=(
                ACQUISITION_MODE is CampaignAcquisitionMode.TARGET_CONDITIONED
            ),
            constraint_decoupled_acquisition=(CONSTRAINT_DECOUPLED_ACQUISITION),
            minimum_intervention_projection=MINIMUM_INTERVENTION_PROJECTION,
            evidence_calibrated_source_mix=EVIDENCE_CALIBRATED_SOURCE_MIX,
            contextual_search_allocation=CONTEXTUAL_SEARCH_ALLOCATION,
            scale_shape=CAMPAIGN_SCALE_SHAPE,
            model_selection_size=CALIBRATED_PROPOSAL_WIDTH,
        )
        policies = experiment_profile.behavior(archive_utility=utility).bind()
    else:
        policies = CampaignPolicies(
            cadence=SealedCutoffDelayedAdmissionCadence(),
            parent_selection=_binding("archive_reservoir", parent_selector),
            memory_assignment=_binding("closed_loop_memory", object()),
            portfolio_selection=_binding(
                (
                    "full_support_calibrated_k8_to_k8"
                    if arm == "live"
                    else f"outcome_blind_random_k{PORTFOLIO_WIDTH}"
                ),
                selector_policy,
            ),
            recombination=_binding("disjoint_patch_union", object()),
            reflection=_binding(
                (
                    "queued_identifiable_mutation_reflection"
                    if arm == "live"
                    else "local_identifiable_mutation_reflection"
                ),
                object(),
            ),
            reflection_supervision=CampaignReflectionSupervisionPolicy(
                ReflectionFailureMode.FAIL_AT_NEXT_STAGE_BOUNDARY
            ),
            archive_utility=utility,
        )
    protocol = CampaignProtocol(
        protocol_id=PROTOCOL_ID,
        protocol_version=1,
        definition_sha256=_sha(PROTOCOL_ID),
        outer_seed=OUTER_SEED,
        generation_count=GENERATION_COUNT,
        required_seed_count=REQUIRED_SEED_COUNT,
        parents_per_portfolio_generation=PARENTS_PER_PORTFOLIO,
        portfolio_width=PORTFOLIO_WIDTH,
        recombinations_per_parent=RECOMBINATIONS_PER_PARENT,
        reflections_per_recombination_generation=1,
        reflection_promotion_block_pairs=1,
        terminal_reflection_policy=(
            TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER
        ),
    )
    prepared = EvolutionCampaign(
        protocol=protocol,
        workload=workload_ports,
        policies=policies,
        runtime=_PreparationRuntime(
            source_closure_sha256=source_closure_sha256,
            arm=arm,
        ),
        budget=OptimizerBudget(
            max_unique_evaluations=PLANNED_UNIQUE_EVALUATIONS,
            max_logical_llm_calls=PLANNED_LOGICAL_CALLS,
            max_generations=GENERATION_COUNT,
        ),
        concurrency=CampaignConcurrency(
            evaluator_concurrency=EVALUATOR_CONCURRENCY,
            agent_concurrency=AGENT_CONCURRENCY,
            agent_queue_capacity=AGENT_QUEUE_CAPACITY,
        ),
        journals=(_PreparationJournal(preparation_journal),),
    ).prepare()
    if experiment_profile is not None:
        experiment_profile.prepared_conformance_record(
            prepared=prepared,
            archive_utility=utility,
            outer_seed=OUTER_SEED,
        )
    if (
        prepared.schedule.portfolio_generations != PORTFOLIO_GENERATIONS
        or prepared.schedule.paired_recombination_generations
        != RECOMBINATION_GENERATIONS
        or tuple(
            (
                wave.source_generation,
                wave.promotion_barrier_generation,
                wave.call_count,
            )
            for wave in prepared.schedule.reflection_waves
        )
        != ((2, 4, 1),)
        or prepared.schedule.planned_candidate_evaluations
        + protocol.required_seed_count
        != PLANNED_UNIQUE_EVALUATIONS
        or prepared.schedule.planned_agent_calls != PLANNED_LOGICAL_CALLS
    ):
        raise RuntimeError(
            "prepared schedule differs from the configured evaluation/call contract"
        )
    ledger = PortfolioOutcomeFeedbackLedger()
    binding_factory = CalibratedCampaignBindingFactory(
        scope=_calibration_scope(prepared),
        objectives=equal_weight_slate_objectives(benchmark.objectives),
        ledger=ledger,
        option_prompt_projection=_option_prompt_projection(),
        common_candidate_pool_policy=_common_candidate_pool_policy(),
        proposal_support_policy=_proposal_support_policy(),
        assign_all_cards_by_default=_common_candidate_pool_policy() is None,
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=_allocator(),
        constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
        minimum_intervention_projection=MINIMUM_INTERVENTION_PROJECTION,
        evidence_calibrated_source_mix=EVIDENCE_CALIBRATED_SOURCE_MIX,
        contextual_search_allocation=CONTEXTUAL_SEARCH_ALLOCATION,
    )
    direction = AbsoluteToleranceDirectionAdjudicator(
        benchmark_sha256=binding_factory.scope.benchmark_sha256,
        session_sha256=binding_factory.scope.session_sha256,
        resolutions=tuple(
            MetricDirectionResolution(metric_id=metric_id, absolute_tolerance=0.0)
            for metric_id in OBJECTIVE_IDS
        ),
    )
    detailed = benchmark.detailed_evaluator
    semantics = benchmark.optimization_semantics
    if detailed is None or semantics is None:
        raise RuntimeError("BOiLS scientific authorities were not bound")
    parent_measurement = create_parent_measurement_projection(
        benchmark_sha256=binding_factory.scope.benchmark_sha256,
        session_sha256=binding_factory.scope.session_sha256,
        decision_metrics=DecisionMetricProjection.from_optimization_semantics(
            semantics
        ),
        evaluator=detailed.evaluator_identity,
        objective_resolution_identity=None,
    )
    return _Bundle(
        arm=arm,
        settings=settings,
        evaluator=evaluator,
        evaluator_observer=evaluator_observer,
        benchmark=benchmark,
        config=config,
        shared_initial_design=shared_initial_design,
        workload_ports=workload_ports,
        prepared=prepared,
        policies=policies,
        experiment_profile=experiment_profile,
        ids=ids,
        memory=memory,
        seed_card=seed_card,
        parent_selector=parent_selector,
        utility=utility,
        binding_factory=binding_factory,
        coordinator=coordinator,
        feedback_ledger=ledger,
        direction_adjudicator=direction,
        parent_measurement_projection=parent_measurement,
    )


def _production_learning_runtime(
    bundle: _Bundle,
) -> tuple[
    ClosedLoopCampaignLearningRuntime,
    CampaignEvidenceRegistry,
    CommittedRegistryIdentifiableReflectionEvidenceSource,
]:
    detailed = bundle.benchmark.detailed_evaluator
    if detailed is None:
        raise RuntimeError("production learning requires detailed evaluator")
    evaluator_contract_sha256 = typed_json_sha256(
        _object(detailed.evaluator_identity.to_record())
    )
    scope = HypothesisAuditScope(
        workload_instance_sha256s=(bundle.config.configuration_sha256,),
        evaluator_contract_sha256=evaluator_contract_sha256,
        metric_adjudicator_definition_sha256=(
            bundle.direction_adjudicator.definition_sha256
        ),
        campaign_sha256s=(bundle.prepared.preparation_sha256,),
    )
    evidence = CampaignEvidenceRegistry()
    learning = ClosedLoopCampaignLearning(memory=bundle.memory)
    runtime = ClosedLoopCampaignLearningRuntime(
        learning=learning,
        reflection_projection=StructuredCampaignReflectionLearningProjector(
            semantic_compiler=PortableFiniteActionInsightSemanticCompiler(),
            scope=scope,
            applicable_operator_kinds=("typed_mutation",),
            diagnostic_operator_kind="typed_mutation",
            diagnostic_editable_paths=REFLECTION_DECISION_PATHS,
            initial_score=0.0,
            minimum_support_clusters=2,
            minimum_support_instances=1,
        ),
        generation_auditor=TransactionalPortfolioGenerationAuditor(
            evidence_registry=evidence,
            campaign_sha256=bundle.prepared.preparation_sha256,
            workload_instance_sha256=bundle.config.configuration_sha256,
            evaluator_contract_sha256=evaluator_contract_sha256,
            metric_projector=ObjectiveDeltaMetricEffectProjector(
                METRIC_ADJUDICATOR_SHA256
            ),
            action_semantics_compiler=FinitePortfolioActionSemanticsCompiler(),
            hypothesis_matcher=PortableFiniteActionHypothesisMatcher(),
        ),
    )
    return (
        runtime,
        evidence,
        CommittedRegistryIdentifiableReflectionEvidenceSource(
            registry=evidence,
            campaign_sha256=bundle.prepared.preparation_sha256,
            workload_instance_sha256=bundle.config.configuration_sha256,
            evaluator_contract_sha256=evaluator_contract_sha256,
        ),
    )


@dataclass(frozen=True, slots=True)
class _BoilsDiagnosticMemoryBlock:
    """One admitted reflection cohort with a pre-provider M/N diagnostic."""

    exposure: CampaignDiagnosticExposureReceipt
    eligible_references: tuple[InsightRef, ...]
    estimand_context: FrozenJsonObject
    active_unit_rank: int
    cohort_selection_key_sha256: str

    @property
    def exact_context_sha256(self) -> str:
        return typed_json_sha256(self.estimand_context)


@dataclass(slots=True)
class _BoilsDiagnosticBlockCoordinator:
    """Seal BOiLS' workload description around the generic diagnostic law.

    The only workload-specific content here is the named objective/utility
    estimand. Cohort selection, complete-support resolution, randomization,
    assignment, and delayed credit remain core AgentEvolve mechanisms.
    """

    memory: InsightMemoryBank
    learning_runtime: ClosedLoopCampaignLearningRuntime
    utility: AffineHypervolumeArchiveUtility
    _blocks: dict[int, _BoilsDiagnosticMemoryBlock] = field(
        init=False,
        default_factory=dict,
    )
    _eligible_receipts: dict[int, tuple[str, ...]] = field(
        init=False,
        default_factory=dict,
    )

    def resolve(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> _BoilsDiagnosticMemoryBlock | None:
        generation = context.stage_request.step.generation
        receipt_sha256s = context.stage_request.test_eligible_reflection_receipt_sha256s
        cached = self._blocks.get(generation)
        if cached is not None:
            if self._eligible_receipts[generation] != receipt_sha256s:
                raise ValueError(
                    "one BOiLS diagnostic generation received inconsistent "
                    "reflection cohorts"
                )
            return cached
        if not receipt_sha256s:
            return None
        exposures = self.learning_runtime.diagnostic_exposures(receipt_sha256s)
        if len(exposures) != 1:
            raise ValueError("BOiLS requires one sealed diagnostic exposure")
        exposure = exposures[0]
        entries = self.memory.entries_for(exposure.references)
        references = tuple(sorted(value.reference for value in entries))
        if not references:
            raise ValueError("BOiLS diagnostic cohort cannot be empty")
        if PARENTS_PER_PORTFOLIO != 2:
            raise ValueError("the randomized M/N diagnostic requires two lanes")
        lane_ids = tuple(
            f"reservoir_{index + 1:04d}" for index in range(PARENTS_PER_PORTFOLIO)
        )
        rank = (
            int.from_bytes(
                hashlib.sha256(
                    (
                        f"{TASK_SHA256}:{OUTER_SEED}:{exposure.receipt_sha256}:"
                        f"generation:{generation}:active-neutral-lane-pair"
                    ).encode("ascii", errors="strict")
                ).digest(),
                "big",
            )
            % 2
        )
        cohort_selection_key_sha256 = _sha(
            f"{TASK_SHA256}:{OUTER_SEED}:{exposure.receipt_sha256}:"
            f"generation:{generation}:complete-support-cohort"
        )
        estimand_context = _object(
            {
                "schema_version": 1,
                "workload_family": "abc_logic_synthesis_recipe_codesign",
                "treatment_unit": "one_complete_candidate_portfolio_wave",
                "intervention": (
                    "memory_guided_action_dose_package_vs_canonical_prompt_"
                    "redacted_neutral_with_shared_required_actions"
                ),
                "outcome": "fixed_affine_archive_joint_wave_gain",
                "objective_directions": {
                    metric_id: "minimize" for metric_id in OBJECTIVE_IDS
                },
                "archive_utility_id": self.utility.utility_id,
                "archive_utility_definition_sha256": (self.utility.definition_sha256),
                "reward_aggregation_id": MEMORY_AGGREGATION_ID,
                "reward_aggregation_definition_sha256": (
                    MEMORY_AGGREGATION_DEFINITION_SHA256
                ),
                "assignment_design": {
                    "policy_id": PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID,
                    "policy_version": (PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION),
                    "policy_definition_sha256": (
                        PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256
                    ),
                    "reflection_exposure_receipt_sha256": (exposure.receipt_sha256),
                    "eligible_references": [
                        {
                            "insight_id": value.insight_id.value,
                            "version": value.version,
                        }
                        for value in references
                    ],
                    "lane_ids": list(lane_ids),
                    "subset_size": 1,
                    "active_neutral_pair": True,
                    "one_card_supported_in_both_lanes_required_for_matched_assay": (
                        True
                    ),
                    "card_selection": {
                        "policy_id": PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID,
                        "policy_version": (
                            PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION
                        ),
                        "policy_definition_sha256": (
                            PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256
                        ),
                        "selection_key_sha256": cohort_selection_key_sha256,
                        "select_one_card_after_prospective_two_lane_support_audit": True,
                    },
                    "active_unit_rank": rank,
                    "provider_and_outcome_blind": True,
                    "no_shared_support_recourse": {
                        "policy_id": PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_ID,
                        "policy_version": (
                            PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_VERSION
                        ),
                        "definition_sha256": (
                            PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_DEFINITION_SHA256
                        ),
                        "administer_each_exactly_supported_lane": True,
                        "optimization_exposure_only": True,
                        "card_vs_neutral_effect_identified": False,
                        "online_causal_credit_allowed": False,
                    },
                },
                "adaptive_score_consumption": False,
                "causal_claim_boundary": (
                    "lane_randomized_single_block_diagnostic_not_same_parent_"
                    "or_full_candidate_pool_matched"
                ),
                "card_vs_neutral_effect_identified": False,
                "required_successor_design": (
                    "replicated_same_parent_same_full_pool_active_neutral_slots"
                ),
            }
        )
        block = _BoilsDiagnosticMemoryBlock(
            exposure=exposure,
            eligible_references=references,
            estimand_context=estimand_context,
            active_unit_rank=rank,
            cohort_selection_key_sha256=cohort_selection_key_sha256,
        )
        self._blocks[generation] = block
        self._eligible_receipts[generation] = receipt_sha256s
        return block

    def project(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> CampaignPortfolioMemoryEstimandProjection | None:
        block = self.resolve(context)
        if block is None:
            return None
        return CampaignPortfolioMemoryEstimandProjection(
            estimand_context=block.estimand_context,
            estimand_stratum_sha256=block.exact_context_sha256,
        )

    def require_projected_context(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> _BoilsDiagnosticMemoryBlock | None:
        block = self.resolve(context)
        if block is None:
            return None
        values = dict(context.evidence_context.items)
        if (
            values.get("memory_estimand_stratum_sha256") != block.exact_context_sha256
            or values.get("memory_estimand_context") != block.estimand_context
        ):
            raise ValueError("BOiLS selector context differs from its sealed estimand")
        return block


class _WaveFactory:
    def __init__(
        self,
        *,
        bundle: _Bundle,
        learning_runtime: ClosedLoopCampaignLearningRuntime | None,
        records: list[dict[str, object]],
        ids: DeterministicIdFactory | None = None,
        binding_factory: CalibratedCampaignBindingFactory | None = None,
        coordinator: CalibratedPortfolioCampaignCoordinator | None = None,
        diagnostic_coordinator: _BoilsDiagnosticBlockCoordinator | None = None,
        target_conditioned_controller: (
            TargetConditionedCampaignOutcomeUpdater | None
        ) = None,
    ) -> None:
        self.bundle = bundle
        self.learning_runtime = learning_runtime
        self.records = records
        self.ids = bundle.ids if ids is None else ids
        self.binding_factory = (
            bundle.binding_factory if binding_factory is None else binding_factory
        )
        self.coordinator = bundle.coordinator if coordinator is None else coordinator
        self.diagnostic_coordinator = diagnostic_coordinator
        self.target_conditioned_controller = target_conditioned_controller
        if self.learning_runtime is not None and self.diagnostic_coordinator is None:
            self.diagnostic_coordinator = _BoilsDiagnosticBlockCoordinator(
                memory=bundle.memory,
                learning_runtime=self.learning_runtime,
                utility=bundle.utility,
            )

    def _request(
        self,
        context,
        cards,
        source_registry=None,
        memory_dose_contract=None,
        experimental_view_receipt=None,
        candidate_pool_required_option_ids=(),
        context_extension_payload=None,
    ):
        request_context = context.evidence_context
        if context_extension_payload is not None:
            if (
                type(context_extension_payload) is not dict
                or not context_extension_payload
            ):
                raise TypeError(
                    "context_extension_payload must be a non-empty exact dict"
                )
            request_context = attach_campaign_selector_context_extension(
                request_context,
                CampaignSelectorContextExtension(
                    extension_id="portfolio_memory_context_transfer",
                    extension_version=1,
                    definition_sha256=(
                        PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_DEFINITION_SHA256
                    ),
                    payload=_object(context_extension_payload),
                ),
            )
        return PortfolioSelectionRequest(
            call_id=self.ids.new_llm_call_id(),
            operation="select_portfolio",
            instruction=(
                "Propose sealed BOiLS options under the authenticated calibrated "
                "portfolio contract. The engine alone materializes candidates."
            ),
            context=request_context,
            finite_variation_contract=context.variation.contract,
            cards=cards,
            portfolio_size=PORTFOLIO_WIDTH,
            required_metric_ids=OBJECTIVE_IDS,
            min_distinct_families=3,
            # The calibrated slate must administer every assigned card across
            # the evaluated K8, but unrelated exploratory members may remain
            # card-unclaimed. Requiring a claim on every member would make a
            # one-path diagnostic card incompatible with four disjoint paths.
            require_supporting_cards=False,
            require_pairwise_disjoint_parent_patches=(
                _require_pairwise_disjoint_evaluation_patches()
            ),
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            source_registry=source_registry,
            experimental_view_receipt=experimental_view_receipt,
            memory_dose_contract=memory_dose_contract,
            candidate_pool_required_option_ids=(candidate_pool_required_option_ids),
        )

    def _register(self, context, selection, *, diagnostic) -> None:
        if self.bundle.arm == "control":
            self.records.append(
                {
                    "generation": context.stage_request.step.generation,
                    "parent_slot": context.parent_slot,
                    "parent_candidate_id": context.parent.candidate_id.value,
                    "selection_request_sha256": selection.request_sha256,
                    "selection_policy": "outcome_blind_random_k8",
                    "eligible_option_count": len(
                        selection.finite_variation_contract.options
                    ),
                    "evaluation_width": PORTFOLIO_WIDTH,
                    "diagnostic": diagnostic,
                }
            )
            return
        bounded_memory_dose = selection.memory_dose_contract is not None
        expected_prompt_definition = calibrated_portfolio_prompt_definition_sha256(
            self.binding_factory.option_prompt_projection,
            bounded_memory_dose=bounded_memory_dose,
            proposal_support=_proposal_support_policy() is not None,
            feasibility_witness_mode=FEASIBILITY_WITNESS_MODE,
            finite_variation_contract=selection.finite_variation_contract,
            hierarchical_composition_required_proposals=(
                _hierarchical_prompt_composition_count()
            ),
            constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
        )
        factory = self.binding_factory
        if factory.scope.prompt_definition_sha256 != expected_prompt_definition:
            factory = replace(
                factory,
                scope=replace(
                    factory.scope,
                    prompt_definition_sha256=expected_prompt_definition,
                ),
            )
        binding = factory.build(
            request=selection,
            variation=context.variation,
            wave_index=context.stage_request.step.generation,
            frozen_archive_snapshot_sha256=(
                context.stage_request.archive_utility.snapshot_sha256
            ),
            contextual_allocation=context.contextual_allocation,
        )
        target_context = (
            None
            if self.target_conditioned_controller is None
            else self.target_conditioned_controller.context_for_wave(
                build=context,
                selection=selection,
            )
        )
        self.coordinator.register(
            selection,
            binding,
            target_conditioned_context=target_context,
        )
        prompt = self.coordinator.render(selection)
        self.records.append(
            {
                "generation": context.stage_request.step.generation,
                "parent_slot": context.parent_slot,
                "parent_candidate_id": context.parent.candidate_id.value,
                "selection_request_sha256": selection.request_sha256,
                "calibrated_binding_sha256": binding.binding_sha256,
                "calibrated_prompt_sha256": _sha(prompt),
                "calibrated_prompt_utf8_bytes": len(prompt.encode("utf-8")),
                "prompt_definition_sha256": (
                    binding.context.scope.prompt_definition_sha256
                ),
                "selector_policy_definition_sha256": (
                    binding.context.scope.selector_policy_definition_sha256
                ),
                "option_prompt_projection_sha256": (
                    None
                    if binding.option_prompt_projection is None
                    else binding.option_prompt_projection.projection_sha256
                ),
                "proposal_width": CALIBRATED_PROPOSAL_WIDTH,
                "evaluation_width": PORTFOLIO_WIDTH,
                "eligible_option_count": len(
                    selection.finite_variation_contract.options
                ),
                "selector_context_utf8_bytes": _canonical_json_size(
                    thaw_json(selection.context)
                ),
                "diagnostic": diagnostic,
                "memory_dose_contract": (
                    None
                    if selection.memory_dose_contract is None
                    else selection.memory_dose_contract.to_record()
                ),
            }
        )

    def _bootstrap_wave(self, context, *, status: str, evidence=None):
        selection = self._request(context, (self.bundle.seed_card,))
        wave = PortfolioVariationWaveRequest(
            selection_request=selection,
            parent=context.parent,
            generation=context.stage_request.step.generation,
            label_prefix=(
                f"boils_g{context.stage_request.step.generation:02d}_"
                f"p{context.parent_slot + 1:02d}"
            ),
            phase=CAMPAIGN_PHASE,
        )
        self._register(
            context,
            selection,
            diagnostic={
                "status": status,
                "memory_credit_issued": False,
                "evidence": evidence,
            },
        )
        return wave

    def _matched_dose_wave(
        self,
        context,
        *,
        diagnostic_block: _BoilsDiagnosticMemoryBlock,
        matched_plan,
        assignment,
        arm_view,
        entry,
        support,
        support_resolution_receipt_sha256: str,
    ):
        projection = PortfolioMemoryContextProjectionBinding.from_selector_context(
            context.evidence_context
        )
        dose = None
        if arm_view.memory_dose_allowed:
            dose = BoundedPortfolioMemoryDoseContract(
                card_supports=(support,),
                proposed_supported_member_bounds=(1, 1),
                evaluated_supported_member_bounds=(1, 1),
                minimum_unattributed_proposed_members=7,
                minimum_unattributed_evaluated_members=PORTFOLIO_WIDTH - 1,
                maximum_cards_per_member=1,
                require_every_assigned_card=True,
            )
        selection = self._request(
            context,
            arm_view.cards,
            arm_view.source_registry,
            memory_dose_contract=dose,
            experimental_view_receipt=arm_view.experimental_view_receipt,
            candidate_pool_required_option_ids=(
                arm_view.required_common_pool_option_ids
            ),
        )
        if matched_plan.reference != entry.reference:
            raise ValueError(
                "BOiLS matched assignment differs from its selected source card"
            )
        snapshot = self.bundle.utility.require_snapshot(
            context.stage_request.archive_utility
        )
        reward = AffineFrozenArchiveJointWaveReward(snapshot)
        aggregation = PortfolioRewardAggregationBinding(
            aggregate=lambda outcomes, reward=reward: float(
                reward(tuple(value.candidate for value in outcomes))
            ),
            aggregation_id=MEMORY_AGGREGATION_ID,
            aggregation_version=1,
            definition_sha256=MEMORY_AGGREGATION_DEFINITION_SHA256,
        )
        wave = PortfolioVariationWaveRequest(
            selection_request=selection,
            parent=context.parent,
            generation=context.stage_request.step.generation,
            label_prefix=(
                f"boils_g{context.stage_request.step.generation:02d}_"
                f"p{context.parent_slot + 1:02d}"
            ),
            phase=CAMPAIGN_PHASE,
            matched_memory_control=PortfolioMemoryMatchedControlWavePlan(
                plan=matched_plan,
                assignment=assignment,
                arm_view=arm_view,
                aggregation=aggregation,
                context_projection=projection,
            ),
        )
        self._register(
            context,
            selection,
            diagnostic={
                "status": "applied_randomized_active_neutral_arm",
                "experimental_arm": assignment.arm.value,
                "exposure_receipt_sha256": (diagnostic_block.exposure.receipt_sha256),
                "selected_insight_id": entry.reference.insight_id.value,
                "selected_insight_version": entry.reference.version,
                "estimand_context_sha256": projection.estimand_context_sha256,
                "memory_dose_contract_sha256": (
                    None if dose is None else dose.contract_sha256
                ),
                "memory_dose_support": support.to_record(),
                "complete_support_resolution_receipt_sha256": (
                    support_resolution_receipt_sha256
                ),
                "matched_control_plan_sha256": matched_plan.plan_sha256,
                "matched_arm_view_sha256": arm_view.view_sha256,
                "memory_credit_issued": False,
                "matched_control_outcome_pending": True,
                "causal_credit_status": (
                    "append_only_single_block_diagnostic_no_online_credit"
                ),
            },
        )
        return wave

    def _supported_optimization_memory_wave(
        self,
        context,
        *,
        diagnostic_block: _BoilsDiagnosticMemoryBlock,
        entry,
        card,
        support,
        matched_resolution,
        lane_resolution,
        context_transfer_assessment,
    ):
        """Administer an exact lane-supported dose without causal overclaiming."""

        if (
            type(context_transfer_assessment)
            is not PortfolioMemoryContextTransferAssessment
            or not context_transfer_assessment.exact_action_replay_authorized
        ):
            raise ValueError(
                "forced memory dose requires exact source-parent authority"
            )

        source_registry = admit_portfolio_card_sources((entry,), (card,))
        dose = BoundedPortfolioMemoryDoseContract(
            card_supports=(support,),
            proposed_supported_member_bounds=(1, 1),
            evaluated_supported_member_bounds=(1, 1),
            minimum_unattributed_proposed_members=7,
            minimum_unattributed_evaluated_members=PORTFOLIO_WIDTH - 1,
            maximum_cards_per_member=1,
            require_every_assigned_card=True,
        )
        selection = self._request(
            context,
            (card,),
            source_registry,
            memory_dose_contract=dose,
            context_extension_payload={
                "optimization_memory_context_transfer": {
                    **context_transfer_assessment.to_record(),
                    "contextual_forced_action_dose_allowed": True,
                }
            },
        )
        wave = PortfolioVariationWaveRequest(
            selection_request=selection,
            parent=context.parent,
            generation=context.stage_request.step.generation,
            label_prefix=(
                f"boils_g{context.stage_request.step.generation:02d}_"
                f"p{context.parent_slot + 1:02d}"
            ),
            phase=CAMPAIGN_PHASE,
        )
        self._register(
            context,
            selection,
            diagnostic={
                "status": "applied_exact_lane_supported_optimization_memory",
                "experimental_arm": "optimization_exposure",
                "exposure_receipt_sha256": diagnostic_block.exposure.receipt_sha256,
                "selected_insight_id": entry.reference.insight_id.value,
                "selected_insight_version": entry.reference.version,
                "memory_dose_contract_sha256": dose.contract_sha256,
                "memory_dose_support": support.to_record(),
                "memory_context_transfer_assessment": (
                    context_transfer_assessment.to_record()
                ),
                "matched_support_resolution_receipt_sha256": (
                    matched_resolution.receipt_sha256
                ),
                "complete_support_resolution": matched_resolution.to_record(),
                "lane_support_resolution": lane_resolution.to_record(),
                "memory_credit_issued": False,
                "matched_control_outcome_pending": False,
                "causal_credit_status": (
                    "optimization_exposure_only_no_card_vs_neutral_effect_claim"
                ),
            },
        )
        return wave

    def _advisory_optimization_memory_wave(
        self,
        context,
        *,
        diagnostic_block: _BoilsDiagnosticMemoryBlock,
        entry,
        card,
        support,
        matched_resolution,
        lane_resolution,
        assessment,
        context_transfer_assessment,
    ):
        """Expose signed evidence without forcing its observed exact action."""

        if (
            type(context_transfer_assessment)
            is not PortfolioMemoryContextTransferAssessment
        ):
            raise TypeError("context transfer assessment must be exact")
        contextual_forced_action_dose_allowed = (
            assessment.forced_action_dose_allowed
            and context_transfer_assessment.exact_action_replay_authorized
        )
        if contextual_forced_action_dose_allowed:
            raise ValueError(
                "context-authorized favorable memory must use the bounded dose path"
            )
        source_registry = admit_portfolio_card_sources((entry,), (card,))
        selection = self._request(
            context,
            (card,),
            source_registry,
            context_extension_payload={
                "optimization_memory_context_transfer": {
                    **context_transfer_assessment.to_record(),
                    "base_signed_forced_action_dose_allowed": (
                        assessment.forced_action_dose_allowed
                    ),
                    "contextual_forced_action_dose_allowed": False,
                    "instruction": (
                        "Treat the source action as advisory in this parent context; "
                        "do not assume its observed benefit transfers."
                    ),
                }
            },
        )
        wave = PortfolioVariationWaveRequest(
            selection_request=selection,
            parent=context.parent,
            generation=context.stage_request.step.generation,
            label_prefix=(
                f"boils_g{context.stage_request.step.generation:02d}_"
                f"p{context.parent_slot + 1:02d}"
            ),
            phase=CAMPAIGN_PHASE,
        )
        self._register(
            context,
            selection,
            diagnostic={
                "status": "applied_signed_advisory_optimization_memory",
                "experimental_arm": "optimization_advisory_exposure",
                "exposure_receipt_sha256": diagnostic_block.exposure.receipt_sha256,
                "selected_insight_id": entry.reference.insight_id.value,
                "selected_insight_version": entry.reference.version,
                "optimization_memory_assessment": assessment.to_record(),
                "memory_context_transfer_assessment": (
                    context_transfer_assessment.to_record()
                ),
                "contextual_forced_action_dose_allowed": False,
                "memory_dose_contract_sha256": None,
                "memory_dose_support": support.to_record(),
                "matched_support_resolution_receipt_sha256": (
                    matched_resolution.receipt_sha256
                ),
                "complete_support_resolution": matched_resolution.to_record(),
                "lane_support_resolution": lane_resolution.to_record(),
                "memory_credit_issued": False,
                "matched_control_outcome_pending": False,
                "causal_credit_status": (
                    "signed_advisory_only_no_forced_action_no_effect_claim"
                ),
            },
        )
        return wave

    def build_batch(
        self,
        contexts: tuple[CampaignPortfolioWaveContext, ...],
    ) -> tuple[PortfolioVariationWaveRequest, ...]:
        if type(contexts) is not tuple or not contexts:
            raise ValueError("wave batch requires a non-empty exact context tuple")
        if any(type(value) is not CampaignPortfolioWaveContext for value in contexts):
            raise TypeError("wave batch contains a foreign context")
        generations = {value.stage_request.step.generation for value in contexts}
        if len(generations) != 1:
            raise ValueError("wave batch cannot mix generations")
        generation = next(iter(generations))
        if (
            generation != 5
            or self.learning_runtime is None
            or self.bundle.arm == "control"
        ):
            return tuple(self.build(value) for value in contexts)
        if len(contexts) != PARENTS_PER_PORTFOLIO:
            raise RuntimeError("G5 compatibility matching requires both parent lanes")

        coordinator = self.diagnostic_coordinator
        if coordinator is None:
            raise RuntimeError("G5 requires its sealed diagnostic coordinator")
        diagnostic_by_lane = {
            context.parent_lane.lane_id: coordinator.require_projected_context(context)
            for context in contexts
        }
        if any(value is None for value in diagnostic_by_lane.values()):
            raise RuntimeError("G5 requires exactly one admitted reflection")
        diagnostic_receipts = {
            value.exposure.receipt_sha256
            for value in diagnostic_by_lane.values()
            if value is not None
        }
        diagnostic_contexts = {
            value.exact_context_sha256
            for value in diagnostic_by_lane.values()
            if value is not None
        }
        if len(diagnostic_receipts) != 1 or len(diagnostic_contexts) != 1:
            raise RuntimeError("G5 lanes received different diagnostic blocks")
        diagnostic_block = next(
            value for value in diagnostic_by_lane.values() if value is not None
        )
        exposure = diagnostic_block.exposure
        if not diagnostic_block.eligible_references:
            raise RuntimeError("reflection cohort cannot be empty")
        entries = tuple(
            sorted(
                self.bundle.memory.entries_for(diagnostic_block.eligible_references),
                key=lambda value: value.reference,
            )
        )
        if not entries:
            raise RuntimeError("G5 memory bank lost its admitted cohort")

        contexts_by_lane = {value.parent_lane.lane_id: value for value in contexts}
        lane_contracts = tuple(
            context.variation.contract
            for _, context in sorted(contexts_by_lane.items())
        )
        cards_by_key = {}
        entries_by_card = {}
        assessments_by_card = {}
        semantics = self.bundle.benchmark.optimization_semantics
        if semantics is None:
            raise RuntimeError("BOiLS signed memory requires optimization semantics")
        for ordinal, entry in enumerate(entries, start=1):
            assessment = assess_portfolio_optimization_memory(
                entry.evidence_lineage,
                semantics,
            )
            prompt_payload = project_action_neutral_insight_prompt_payload(
                entry,
                prompt_payload=_object(
                    {
                        "claim": entry.draft.claim,
                        "trigger": entry.draft.trigger,
                        "mechanism": entry.draft.mechanism,
                        "quarantined": True,
                        "optimization_memory_assessment": assessment.to_record(),
                    }
                ),
                finite_variation_contracts=lane_contracts,
            )
            card = portfolio_card_from_insight_entry(
                entry,
                card_key=f"card.boils.g05.r{ordinal:02d}",
                prompt_payload=prompt_payload,
                evidence_sha256=entry.evidence_lineage.identity_sha256,
                source_receipt_sha256=exposure.receipt_sha256,
                assigned_score=0.0,
            )
            if card.source_binding is None:  # pragma: no cover - constructor closes.
                raise AssertionError("source-bound card lost its binding")
            cards_by_key[card.card_key] = card
            entries_by_card[card.card_key] = entry
            assessments_by_card[card.card_key] = assessment

        support_lanes = tuple(
            CampaignDiagnosticSupportLaneInput(
                lane=LaneCardMatchingLane(
                    lane_id=lane_id,
                    lane_identity_sha256=typed_json_sha256(
                        _object(
                            {
                                "schema_version": 1,
                                "generation": generation,
                                "lane_id": lane_id,
                                "parent_candidate_id": (
                                    context.parent.candidate_id.value
                                ),
                                "finite_contract_identity_sha256": (
                                    context.variation.contract.identity_sha256
                                ),
                            }
                        )
                    ),
                ),
                finite_variation_contract=context.variation.contract,
            )
            for lane_id, context in sorted(contexts_by_lane.items())
        )
        support_cards = tuple(
            CampaignDiagnosticSupportCardInput(
                card=LaneCardMatchingCard(
                    card_key=card_key,
                    card_identity_sha256=card.source_binding.binding_sha256,
                ),
                semantics=PortfolioMemoryDoseCardSemantics.from_insight(
                    card_key=card.card_key,
                    card_content_sha256=card.content_sha256,
                    draft=entries_by_card[card_key].draft,
                    evidence_lineage=(entries_by_card[card_key].evidence_lineage),
                ),
            )
            for card_key, card in sorted(cards_by_key.items())
        )
        exact_support_cards = tuple(
            CampaignDiagnosticSupportCardInput(
                card=value.card,
                semantics=PortfolioMemoryDoseCardSemantics.from_insight(
                    card_key=value.card.card_key,
                    card_content_sha256=(
                        cards_by_key[value.card.card_key].content_sha256
                    ),
                    draft=entries_by_card[value.card.card_key].draft,
                    evidence_lineage=(
                        entries_by_card[value.card.card_key].evidence_lineage
                    ),
                    support_scope=(PortfolioMemoryDoseSupportScope.EXACT_SOURCE_PARENT),
                ),
            )
            for value in support_cards
        )
        exact_semantics_by_card = {
            value.card.card_key: value.semantics for value in exact_support_cards
        }
        dose_support_cards = tuple(
            value
            for value in exact_support_cards
            if assessments_by_card[value.card.card_key].forced_action_dose_allowed
        )
        if dose_support_cards:
            resolution = PortfolioMemoryMatchedSupportResolver().resolve(
                lanes=support_lanes,
                cards=dose_support_cards,
                selection_key_sha256=(diagnostic_block.cohort_selection_key_sha256),
            )
        else:
            resolution = PortfolioMemoryMatchedSupportResolution(
                lane_ids=tuple(sorted(value.lane.lane_id for value in support_lanes)),
                eligible_card_keys=(),
                selected_card_key=None,
                selected_lane_supports=(),
                selection_key_sha256=(diagnostic_block.cohort_selection_key_sha256),
            )
        if not resolution.eligible:
            lanes_by_id = {lane.lane.lane_id: lane for lane in support_lanes}
            lane_keys = {
                lane_id: typed_json_sha256(
                    _object(
                        {
                            "schema_version": 1,
                            "cohort_selection_key_sha256": (
                                diagnostic_block.cohort_selection_key_sha256
                            ),
                            "lane_id": lane_id,
                            "lane_identity_sha256": lane.lane.lane_identity_sha256,
                        }
                    )
                )
                for lane_id, lane in sorted(lanes_by_id.items())
            }
            dose_lane_resolutions = (
                {
                    lane_id: PortfolioMemoryLaneSupportResolver().resolve(
                        lane=lane,
                        cards=dose_support_cards,
                        selection_key_sha256=lane_keys[lane_id],
                    )
                    for lane_id, lane in sorted(lanes_by_id.items())
                }
                if dose_support_cards
                else {}
            )
            advisory_lane_resolutions = {
                lane_id: PortfolioMemoryLaneSupportResolver().resolve(
                    lane=lane,
                    cards=support_cards,
                    selection_key_sha256=lane_keys[lane_id],
                )
                for lane_id, lane in sorted(lanes_by_id.items())
            }
            waves = []
            for context in contexts:
                lane_id = context.parent_lane.lane_id
                dose_lane_resolution = dose_lane_resolutions.get(lane_id)
                if dose_lane_resolution is not None and dose_lane_resolution.eligible:
                    selected_lane_card_key = dose_lane_resolution.selected_card_key
                    selected_lane_support = dose_lane_resolution.selected_support
                    assert selected_lane_card_key is not None
                    assert selected_lane_support is not None
                    context_transfer_assessment = (
                        assess_portfolio_memory_context_transfer(
                            exact_semantics_by_card[selected_lane_card_key],
                            context.variation.contract,
                        )
                    )
                    waves.append(
                        self._supported_optimization_memory_wave(
                            context,
                            diagnostic_block=diagnostic_block,
                            entry=entries_by_card[selected_lane_card_key],
                            card=cards_by_key[selected_lane_card_key],
                            support=selected_lane_support,
                            matched_resolution=resolution,
                            lane_resolution=dose_lane_resolution,
                            context_transfer_assessment=(context_transfer_assessment),
                        )
                    )
                    continue
                lane_resolution = advisory_lane_resolutions[lane_id]
                if not lane_resolution.eligible:
                    waves.append(
                        self._bootstrap_wave(
                            context,
                            status="no_lane_support_for_reflected_memory",
                            evidence={
                                "complete_support_resolution": resolution.to_record(),
                                "lane_support_resolution": (
                                    lane_resolution.to_record()
                                ),
                                "memory_credit_issued": False,
                            },
                        )
                    )
                    continue
                selected_lane_card_key = lane_resolution.selected_card_key
                selected_lane_support = lane_resolution.selected_support
                assert selected_lane_card_key is not None
                assert selected_lane_support is not None
                assessment = assessments_by_card[selected_lane_card_key]
                context_transfer_assessment = assess_portfolio_memory_context_transfer(
                    exact_semantics_by_card[selected_lane_card_key],
                    context.variation.contract,
                )
                waves.append(
                    self._advisory_optimization_memory_wave(
                        context,
                        diagnostic_block=diagnostic_block,
                        entry=entries_by_card[selected_lane_card_key],
                        card=cards_by_key[selected_lane_card_key],
                        support=selected_lane_support,
                        matched_resolution=resolution,
                        lane_resolution=lane_resolution,
                        assessment=assessment,
                        context_transfer_assessment=context_transfer_assessment,
                    )
                )
            return tuple(waves)

        selected_card_key = resolution.selected_card_key
        assert selected_card_key is not None
        selected_entry = entries_by_card[selected_card_key]
        selected_card = cards_by_key[selected_card_key]
        source_registry = admit_portfolio_card_sources(
            (selected_entry,),
            (selected_card,),
        )
        ordered_units = tuple(
            StableMemoryAssignmentUnit(
                unit_key=(
                    f"reflection.{exposure.receipt_sha256[:12]}."
                    f"g{generation:02d}.{lane_id}"
                ),
                generation=generation,
                lane_id=lane_id,
            )
            for lane_id in sorted(contexts_by_lane)
        )
        matched_plan = PortfolioMemoryMatchedControlPlanner().plan(
            reference=selected_entry.reference,
            exact_context_sha256=diagnostic_block.exact_context_sha256,
            ordered_units=ordered_units,
            active_unit_rank=diagnostic_block.active_unit_rank,
        )
        waves = []
        for context in contexts:
            lane_id = context.parent_lane.lane_id
            assignment = matched_plan.assignment_for(
                generation=generation,
                lane_id=lane_id,
            )
            arm_view = materialize_portfolio_memory_matched_arm(
                plan=matched_plan,
                assignment=assignment,
                source_card=selected_card,
                source_registry=source_registry,
                finite_variation_contract=context.variation.contract,
            )
            waves.append(
                self._matched_dose_wave(
                    context,
                    diagnostic_block=diagnostic_block,
                    matched_plan=matched_plan,
                    assignment=assignment,
                    arm_view=arm_view,
                    entry=selected_entry,
                    support=resolution.support_for(lane_id),
                    support_resolution_receipt_sha256=resolution.receipt_sha256,
                )
            )
        return tuple(waves)

    def build(self, context: CampaignPortfolioWaveContext):
        generation = context.stage_request.step.generation
        if generation not in PORTFOLIO_GENERATIONS:
            raise ValueError("wave factory received a non-portfolio generation")
        if generation in (1, 3) or self.learning_runtime is None:
            if generation == 3 and (
                context.stage_request.test_eligible_reflection_receipt_sha256s
            ):
                raise RuntimeError("G3 cannot consume a delayed G2 reflection")
            return self._bootstrap_wave(
                context,
                status=(
                    "bootstrap_prior"
                    if generation == 1
                    else "delayed_reflection_bootstrap_not_yet_admitted"
                    if generation == 3
                    else "preparation_probe_without_admission"
                ),
            )
        exposures = self.learning_runtime.diagnostic_exposures(
            context.stage_request.test_eligible_reflection_receipt_sha256s
        )
        if self.bundle.arm == "control":
            return self._bootstrap_wave(
                context,
                status="control_ignores_admitted_reflected_memory",
                evidence={
                    "admitted_exposure_count": len(exposures),
                    "selection_policy": "outcome_blind_random_k8",
                },
            )
        if not exposures:
            raise RuntimeError("G5 did not receive the G4-admitted reflection")
        raise RuntimeError(
            "live G5 reflected memory requires atomic compatibility-aware "
            "batch construction"
        )


def _prediction(
    metric_id: str, direction: MetricEffectDirection
) -> MetricEffectPrediction:
    return MetricEffectPrediction(
        metric_id=metric_id,
        direction=direction,
        comparison_anchor=MetricComparisonAnchor(
            MetricComparisonAnchorKind.CURRENT_PARENT
        ),
    )


def _local_control_insights(
    contrasts: tuple[IdentifiableMutationReflectionContrast, ...],
) -> tuple[InsightDraft, ...]:
    if len(contrasts) < 2:
        raise ValueError("control reflection requires two contrasts")
    return tuple(
        InsightDraft(
            claim=(
                f"Prospectively retest {contrast.option_family} at "
                f"{contrast.affected_path} under the current parent."
            ),
            trigger=(
                f"The sealed {contrast.option_family} family is available at "
                f"{contrast.affected_path}."
            ),
            mechanism=(
                "The prior direct mutation suggests a predictive association; "
                "this rationale is not a causal or mechanistic conclusion."
            ),
            affected_paths=(contrast.affected_path,),
            evidence_summary="Authenticated direct single-mutation observation.",
            confidence=0.5,
            evidence_contrast_ids=(contrast.contrast_id,),
            effect_predictions=tuple(
                _prediction(value.metric_id, value.direction)
                for value in contrast.metrics
            ),
            recommended_option_families=(contrast.option_family,),
            recommended_option_ids=(contrast.option_id,),
            action_template="Test one sealed finite option in the named family.",
            falsification_condition=(
                "Any observed metric direction that differs from this direct "
                "parent-relative observation falsifies the predictive rule."
            ),
            insight_kind=ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
            consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
            factor_capabilities=(contrast.option_family,),
        )
        for contrast in contrasts[:2]
    )


def _reflection_exclusion_rows(
    evidence: IdentifiableReflectionEvidenceSnapshot,
) -> list[dict[str, object]]:
    """Preserve why authenticated source outcomes were not reflected upon."""

    if type(evidence) is not IdentifiableReflectionEvidenceSnapshot:
        raise TypeError("evidence must be exact identifiable reflection evidence")
    evidence.__post_init__()
    return [
        {"reason": reason.value, "count": count}
        for reason, count in evidence.exclusions
    ]


@dataclass(slots=True)
class _LiveReflectionExecutor:
    generator: PydanticAIAgenticGenerator
    ids: DeterministicIdFactory
    records: list[dict[str, object]]
    optimization_semantics: Any

    async def reflect(self, reflection_input: CampaignIdentifiableReflectionInput):
        reflection_request = build_boils_identifiable_reflection_request(
            call_id=self.ids.new_llm_call_id(),
            reflection_input=reflection_input,
            optimization_semantics=self.optimization_semantics,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            min_insights=2,
            max_insights=min(8, len(reflection_input.evidence.contrasts)),
        )
        result = await self.generator.reflect(reflection_request)
        telemetry = result.telemetry
        try:
            MODEL_EXECUTION_PROFILE.validate_telemetry(telemetry)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "reflection violated model/provider/reasoning gates"
            ) from error
        construction = identifiable_reflection_request_construction_record(
            reflection_request,
            reflection_input.evidence,
        )
        encoded = build_boils_identifiable_reflection_learning_envelope(
            reflection_input=reflection_input,
            request=reflection_request,
            result=result,
            optimization_semantics=self.optimization_semantics,
        )
        learning = CampaignReflectionLearningRecordCodec.decode(encoded)
        envelope = thaw_json(encoded)
        record = {
            "call_id": reflection_request.call_id.value,
            "source_generation": reflection_input.query.wave.source_generation,
            "origin_cutoff_event_index": (
                reflection_input.query.sealed_cutoff_event_index_inclusive
            ),
            "identifiable_reflection_input_sha256": reflection_input.input_sha256,
            "registry_snapshot_sha256": (
                reflection_input.source.registry_snapshot_sha256
            ),
            "evidence_snapshot_sha256": reflection_input.evidence.snapshot_sha256,
            "identifiable_contrast_count": len(reflection_input.evidence.contrasts),
            "typed_exclusion_count": sum(
                count for _reason, count in reflection_input.evidence.exclusions
            ),
            "typed_exclusions": _reflection_exclusion_rows(reflection_input.evidence),
            "insight_count": len(result.insights),
            "recombination_results_exposed": False,
            "request_construction": construction,
            "insights": [value.content_record() for value in result.insights],
            CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY: envelope[
                CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY
            ],
            "campaign_reflection_learning_record_sha256": learning.record_sha256,
            "telemetry": _telemetry_record(telemetry),
            "quarantined": True,
        }
        self.records.append(record)
        return _object(record)


@dataclass(slots=True)
class _ControlReflectionExecutor:
    ids: DeterministicIdFactory
    records: list[dict[str, object]]
    optimization_semantics: Any

    async def reflect(self, reflection_input: CampaignIdentifiableReflectionInput):
        reflection_request = build_boils_identifiable_reflection_request(
            call_id=self.ids.new_llm_call_id(),
            reflection_input=reflection_input,
            optimization_semantics=self.optimization_semantics,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            min_insights=2,
            max_insights=min(8, len(reflection_input.evidence.contrasts)),
        )
        insights = _local_control_insights(reflection_input.evidence.contrasts)
        result = ReflectionGenerationResult(
            insights=insights,
            telemetry=AgenticCallTelemetry(
                requested_model="provider-free-control",
                resolved_model="provider-free-control",
                resolved_provider="local-deterministic-control",
                provider_response_id=None,
                finish_reason="policy_completed",
                input_tokens=0,
                output_tokens=0,
                reasoning_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_usd=Decimal("0"),
                latency_ns=0,
            ),
            evidence_catalog_identity_sha256=(
                reflection_request.evidence_catalog.catalog_identity_sha256
            ),
        )
        construction = identifiable_reflection_request_construction_record(
            reflection_request,
            reflection_input.evidence,
        )
        encoded = build_boils_identifiable_reflection_learning_envelope(
            reflection_input=reflection_input,
            request=reflection_request,
            result=result,
            optimization_semantics=self.optimization_semantics,
        )
        learning = CampaignReflectionLearningRecordCodec.decode(encoded)
        self.records.append(
            {
                "call_id": reflection_request.call_id.value,
                "source_generation": reflection_input.query.wave.source_generation,
                "origin_cutoff_event_index": (
                    reflection_input.query.sealed_cutoff_event_index_inclusive
                ),
                "identifiable_reflection_input_sha256": (reflection_input.input_sha256),
                "registry_snapshot_sha256": (
                    reflection_input.source.registry_snapshot_sha256
                ),
                "evidence_snapshot_sha256": (reflection_input.evidence.snapshot_sha256),
                "identifiable_contrast_count": len(reflection_input.evidence.contrasts),
                "typed_exclusion_count": sum(
                    count for _reason, count in reflection_input.evidence.exclusions
                ),
                "typed_exclusions": _reflection_exclusion_rows(
                    reflection_input.evidence
                ),
                "insight_count": len(result.insights),
                "recombination_results_exposed": False,
                "request_construction": construction,
                "campaign_reflection_learning_record_sha256": (learning.record_sha256),
                "provider_calls": 0,
                "quarantined": True,
            }
        )
        return encoded


def _construction_parent(
    *, ordinal: int, configuration: FrozenJsonObject, bundle: _Bundle
) -> EvolutionCandidate:
    configuration_sha256 = typed_json_sha256(configuration)
    objectives = (
        ("total_levels", float(72 - ordinal)),
        ("total_lut_count", float(8_100 - ordinal * 100)),
    )
    detailed = bundle.benchmark.detailed_evaluator
    if detailed is None:
        raise RuntimeError("construction parent requires detailed evaluator")
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId(f"candidate_boils_prepare_{ordinal}"),
            configuration_hash=configuration_sha256,
            configuration_artifact_hash=configuration_sha256,
            proposal_sequence=ordinal,
        ),
        configuration=configuration,
        objectives=objectives,
        valid=True,
        generation=0,
        label=f"boils_prepare_parent_{ordinal}",
        design_rationale="provider_and_abc_free_contract_construction_only",
        detailed_evaluation=DetailedEvaluation(
            phenotype=bundle.benchmark.phenotype_identity.identify(
                thaw_json(configuration)
            ),
            payload=DetailedEvaluationPayload(
                failure=None,
                objectives=objectives,
                violations=(),
                checks=(),
                receipt=None,
                evaluator=detailed.evaluator_identity,
            ),
            timings=EvaluationTimings(total_wall_seconds=0.0),
        ),
    )


def _reflection_probe(
    bundle: _Bundle,
    parents: tuple[EvolutionCandidate, ...],
) -> dict[str, object]:
    semantics = bundle.benchmark.optimization_semantics
    detailed = bundle.benchmark.detailed_evaluator
    if semantics is None or detailed is None:
        raise RuntimeError("reflection probe requires scientific authorities")
    evaluator_contract_sha256 = typed_json_sha256(
        _object(detailed.evaluator_identity.to_record())
    )
    known = tuple(
        sorted(
            bundle.benchmark.phenotype_identity.identify(
                thaw_json(parent.configuration)
            ).value_sha256
            for parent in parents
        )
    )
    compiler = FinitePortfolioActionSemanticsCompiler()
    contrasts: list[IdentifiableMutationReflectionContrast] = []
    ordinal = 0
    for parent in parents:
        variation = bundle.workload_ports.catalog.bind(
            bundle.prepared.benchmark_session.benchmark,
            parent.configuration,
            known,
        )
        for option in variation.contract.options[:2]:
            ordinal += 1
            parent_config = thaw_json(parent.configuration)
            child_config = thaw_json(option.child_configuration)
            parent_sequence = parent_config.get("sequence")
            child_sequence = child_config.get("sequence")
            if type(parent_sequence) is not list or type(child_sequence) is not list:
                raise RuntimeError("BOiLS preparation contrast lost its sequence")
            changed = tuple(
                index
                for index, (before, after) in enumerate(
                    zip(parent_sequence, child_sequence, strict=True)
                )
                if before != after
            )
            if len(changed) != 1:
                raise RuntimeError("BOiLS finite option is not a single mutation")
            index = changed[0]
            source_observation = _sha(
                f"boils-g6-prepare-direct-observation:{ordinal}:"
                f"{option.identity_sha256}"
            )
            contrasts.append(
                IdentifiableMutationReflectionContrast(
                    contrast_id=source_observation,
                    source_observation_sha256=source_observation,
                    source_evidence_id=_sha(
                        f"boils-g6-prepare-source-evidence:{ordinal}"
                    ),
                    event_index=1,
                    workload_instance_sha256=bundle.config.configuration_sha256,
                    evaluator_contract_sha256=evaluator_contract_sha256,
                    campaign_sha256=bundle.prepared.preparation_sha256,
                    parent_candidate_id=parent.candidate_id,
                    child_candidate_id=CandidateId(
                        f"candidate_boils_prepare_mutation_{ordinal}"
                    ),
                    operator_invocation_id=OperatorInvocationId(
                        f"operator_boils_prepare_mutation_{ordinal}"
                    ),
                    finite_contract_identity_sha256=(
                        variation.contract.identity_sha256
                    ),
                    action_semantics_compiler_id=compiler.compiler_id,
                    action_semantics_compiler_version=compiler.compiler_version,
                    action_semantics_definition_sha256=compiler.definition_sha256,
                    option_id=option.option_id,
                    option_identity_sha256=option.identity_sha256,
                    option_family=option.family,
                    affected_path=f"$.sequence[{index}]",
                    parent_local_value=freeze_json(parent_sequence[index]),
                    child_local_value=freeze_json(child_sequence[index]),
                    parent_configuration_sha256=typed_json_sha256(parent.configuration),
                    child_configuration_sha256=typed_json_sha256(
                        option.child_configuration
                    ),
                    parent_outcome_sha256=_sha(
                        f"boils-g6-prepare-parent-outcome:{ordinal}"
                    ),
                    child_outcome_sha256=_sha(
                        f"boils-g6-prepare-child-outcome:{ordinal}"
                    ),
                    metrics=tuple(
                        ObservedMetricEffect(
                            metric_id=metric_id,
                            direction=MetricEffectDirection.DECREASE,
                            delta=-float(ordinal),
                            adjudicator_definition_sha256=(METRIC_ADJUDICATOR_SHA256),
                        )
                        for metric_id in OBJECTIVE_IDS
                    ),
                    mechanism_identifying_design=False,
                    permitted_insight_kinds=(
                        ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
                    ),
                )
            )
    evidence = IdentifiableReflectionEvidenceSnapshot(
        campaign_sha256=bundle.prepared.preparation_sha256,
        workload_instance_sha256=bundle.config.configuration_sha256,
        evaluator_contract_sha256=evaluator_contract_sha256,
        prior_cutoff_event_index_exclusive=0,
        sealed_cutoff_event_index_inclusive=1,
        contrasts=tuple(sorted(contrasts, key=lambda value: value.contrast_id)),
        exclusions=(),
    )
    request = build_identifiable_reflection_generation_request(
        call_id=LLMCallId("call_boils_prepare_reflection_1"),
        evidence=evidence,
        insight_contract=bind_reflection_contract_to_evidence_actions(
            boils_reflection_contract(),
            evidence,
        ),
        optimization_semantics=semantics,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        min_insights=2,
        max_insights=min(8, len(evidence.contrasts)),
    )
    record = identifiable_reflection_request_construction_record(
        request,
        evidence,
    )
    gates = {
        "exact_request_builder_identity": record["request_builder"]
        == {
            "builder_id": IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_ID,
            "builder_version": IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_VERSION,
            "definition_sha256": (
                IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_DEFINITION_SHA256
            ),
        },
        "semantic_v3": request.insight_contract.is_semantic_v3,
        "empirical_only": request.insight_contract.allowed_insight_kinds
        == (ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,),
        "exact_evidence_citation_mapping": len(record["evidence_citation_mapping"])
        == len(evidence.contrasts),
        "sealed_g1_cutoff": (
            evidence.prior_cutoff_event_index_exclusive,
            evidence.sealed_cutoff_event_index_inclusive,
        )
        == (0, 1),
        "no_recombination_facts": "recombination" not in request.prompt.lower(),
        "no_full_contrast_ids_in_prompt": all(
            value.contrast_id not in request.prompt for value in evidence.contrasts
        ),
        "max_output_tokens_exact": request.max_output_tokens == MAX_OUTPUT_TOKENS,
        "bounded_multi_insight_cohort": (
            request.min_insights,
            request.max_insights,
        )
        == (2, min(8, len(evidence.contrasts))),
    }
    return {
        "status": "provider_credential_abc_free_reflection_construction",
        "provider_calls": 0,
        "credential_read": False,
        "abc_executions": 0,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "request": record,
        "evidence": evidence.to_record(),
    }


def _all_wave_probe(
    bundle: _Bundle,
    *,
    wave_sink: Any | None = None,
    wave_context_sink: Any | None = None,
) -> dict[str, object]:
    if wave_sink is not None and not callable(wave_sink):
        raise TypeError("wave_sink must be callable or None")
    if wave_context_sink is not None and not callable(wave_context_sink):
        raise TypeError("wave_context_sink must be callable or None")
    session = bundle.prepared.benchmark_session
    parents = tuple(
        _construction_parent(
            ordinal=ordinal,
            configuration=seed.configuration,
            bundle=bundle,
        )
        for ordinal, seed in enumerate(
            bundle.prepared.seeds.seeds[:PARENTS_PER_PORTFOLIO],
            start=1,
        )
    )
    reflection = _reflection_probe(bundle, parents)
    known = tuple(
        sorted(
            bundle.benchmark.phenotype_identity.identify(
                thaw_json(parent.configuration)
            ).value_sha256
            for parent in parents
        )
    )
    memory_projection = bundle.workload_ports.evidence.initialize_memory(
        session, bundle.prepared.seeds
    )
    archive = _object(
        {
            "preparation_only": True,
            "front_size": len(parents),
            "front_candidates": [
                {
                    "objectives": [
                        {"metric_id": name, "value_hex": value.hex()}
                        for name, value in parent.objectives
                    ]
                }
                for parent in parents
            ],
        }
    )
    ledger = PortfolioOutcomeFeedbackLedger()
    binding_factory = CalibratedCampaignBindingFactory(
        scope=bundle.binding_factory.scope,
        objectives=bundle.binding_factory.objectives,
        ledger=ledger,
        option_prompt_projection=bundle.binding_factory.option_prompt_projection,
        common_candidate_pool_policy=(
            bundle.binding_factory.common_candidate_pool_policy
        ),
        proposal_support_policy=bundle.binding_factory.proposal_support_policy,
        assign_all_cards_by_default=(
            bundle.binding_factory.assign_all_cards_by_default
        ),
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=_allocator(),
        constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
        minimum_intervention_projection=MINIMUM_INTERVENTION_PROJECTION,
        evidence_calibrated_source_mix=EVIDENCE_CALIBRATED_SOURCE_MIX,
        contextual_search_allocation=CONTEXTUAL_SEARCH_ALLOCATION,
    )
    target_controller = _target_conditioned_controller(bundle, coordinator)
    records: list[dict[str, object]] = []
    probe_contextual_planner = (
        CampaignContextualSearchPlanner(
            ledger=ContextualSearchLedger(),
            campaign_scope_sha256=_sha(
                "agent-evolve:contextual-search-campaign:"
                + bundle.prepared.preparation_sha256
            ),
            joint_capability_projector=_contextual_joint_capability_projector(),
            frontier_target_allocator=_frontier_target_allocator(),
        )
        if bundle.arm == "live" and CONTEXTUAL_SEARCH_ALLOCATION
        else None
    )
    factory = _WaveFactory(
        bundle=bundle,
        learning_runtime=None,
        records=records,
        ids=DeterministicIdFactory(f"{AGENTIC_ID_NAMESPACE}_prepare_probe"),
        binding_factory=binding_factory,
        coordinator=coordinator,
        target_conditioned_controller=target_controller,
    )
    rows = []
    steps = {step.generation: step for step in bundle.prepared.schedule.steps}
    for generation in PORTFOLIO_GENERATIONS:
        utility = bundle.utility.freeze(
            benchmark=session.benchmark,
            generation=generation,
            archive=archive,
        )
        stage = CampaignStageRequest(
            preparation_sha256=bundle.prepared.preparation_sha256,
            runtime_start_receipt_sha256=_sha("boils-prepare-runtime-start"),
            step=steps[generation],
            archive_cutoff=CampaignArchiveCutoffReceipt(
                request_sha256=_sha(f"boils-prepare-cutoff-{generation}"),
                preparation_sha256=bundle.prepared.preparation_sha256,
                generation=generation,
                archive=archive,
                evidence=_object({"preparation_only": True}),
            ),
            archive_utility=utility,
            source_portfolio=None,
            test_eligible_reflection_receipt_sha256s=(),
            prior_selector_audit_set_sha256=_sha(
                f"boils-prepare-prior-audit-{generation}"
            ),
        )
        generation_contexts: list[CampaignPortfolioWaveContext] = []
        for slot, parent in enumerate(parents):
            variation = bundle.workload_ports.catalog.bind(
                session.benchmark, parent.configuration, known
            )
            context = bundle.workload_ports.evidence.context(
                session,
                parent.configuration,
                variation,
                memory_projection,
            )
            parent_measurement = bind_parent_measurement(
                candidate=parent,
                variation=variation,
                projection=bundle.parent_measurement_projection,
            )
            context = attach_parent_measurement_to_context(context, parent_measurement)
            cards = bundle.workload_ports.evidence.cards(
                session,
                parent.configuration,
                variation,
                memory_projection,
            )
            generation_contexts.append(
                CampaignPortfolioWaveContext(
                    prepared=bundle.prepared,
                    stage_request=stage,
                    parent_slot=slot,
                    parent=parent,
                    variation=variation,
                    evidence_context=context,
                    evidence_cards=cards,
                    memory=memory_projection,
                    parent_measurement=parent_measurement,
                )
            )
        if target_controller is not None:
            targets = _frontier_target_allocator().allocate(
                archive_utility=utility,
                lanes=tuple(
                    (value.parent_lane.lane_id, value.parent)
                    for value in generation_contexts
                ),
            )
            by_target = {value.lane_id: value for value in targets}
            projector = AuthenticatedAffineFrontierContextProjector()
            rebound = []
            for value in generation_contexts:
                projection = projector.project(
                    archive_utility=utility,
                    parent=value.parent,
                )
                target = by_target[value.parent_lane.lane_id]
                evidence = thaw_json(value.evidence_context)
                if type(evidence) is not dict:
                    raise TypeError("preparation context must be an object")
                evidence[CAMPAIGN_ARCHIVE_CONTEXT_KEY] = projection.to_record()
                evidence[CAMPAIGN_FRONTIER_TARGET_KEY] = target.to_record()
                rebound.append(
                    replace(
                        value,
                        evidence_context=_object(evidence),
                        archive_context=projection,
                        frontier_target=target,
                    )
                )
            generation_contexts = rebound
        if probe_contextual_planner is not None:
            plan = probe_contextual_planner.plan(tuple(generation_contexts))
            by_slice = {value.slice_id: value for value in plan.contracts}
            by_target = {value.lane_id: value for value in plan.frontier_targets}
            rebound = []
            for value in generation_contexts:
                lane_id = value.parent_lane.lane_id
                evidence = thaw_json(value.evidence_context)
                if type(evidence) is not dict:
                    raise TypeError("preparation context must be an object")
                evidence[CAMPAIGN_FRONTIER_TARGET_KEY] = by_target[lane_id].to_record()
                rebound.append(
                    replace(
                        value,
                        evidence_context=_object(evidence),
                        contextual_allocation=by_slice[lane_id],
                        frontier_target=by_target[lane_id],
                    )
                )
            generation_contexts = rebound
        for campaign_context in generation_contexts:
            slot = campaign_context.parent_slot
            parent = campaign_context.parent
            variation = campaign_context.variation
            context = campaign_context.evidence_context
            cards = campaign_context.evidence_cards
            parent_measurement = campaign_context.parent_measurement
            assert parent_measurement is not None
            wave = factory.build(campaign_context)
            if wave_sink is not None:
                wave_sink(wave)
            if wave_context_sink is not None:
                wave_context_sink(wave, campaign_context)
            row: dict[str, object] = {
                "probe_kind": "bootstrap_wave",
                "generation": generation,
                "parent_slot": slot,
                "request_sha256": wave.selection_request.request_sha256,
                "eligible_option_count": len(variation.contract.options),
                "parent_measurement_binding_sha256": (
                    parent_measurement.binding_sha256
                ),
                "evaluation_width": PORTFOLIO_WIDTH,
                "variation_source_ids": list(
                    finite_variation_source_ids(variation.contract)
                ),
                "contextual_allocation": (
                    None
                    if campaign_context.contextual_allocation is None
                    else campaign_context.contextual_allocation.to_record()
                ),
            }
            if bundle.arm == "live":
                binding = coordinator.binding_for(wave.selection_request)
                prompt = coordinator.render(wave.selection_request)
                row.update(
                    {
                        "selection_contract": _selection_contract_label(),
                        "binding_sha256": binding.binding_sha256,
                        "prompt_sha256": _sha(prompt),
                        "proposal_width": CALIBRATED_PROPOSAL_WIDTH,
                    }
                )
            else:
                registered = records[-1]
                if (
                    registered["selection_request_sha256"]
                    != wave.selection_request.request_sha256
                ):
                    raise RuntimeError("control wave registration drifted")
                row.update(
                    {
                        "selection_contract": "outcome_blind_random_k8",
                        "policy_output_width": PORTFOLIO_WIDTH,
                    }
                )
            rows.append(row)
            if generation == 5:
                synthetic_card = PortfolioCard(
                    card_key=f"card.boils.synthetic.g05.p{slot + 1:02d}",
                    reference=bundle.seed_card.reference,
                    content_sha256=_sha(f"boils-g6-synthetic-dose-card:{slot + 1}"),
                    evidence_sha256=_sha(
                        f"boils-g6-synthetic-dose-evidence:{slot + 1}"
                    ),
                    prompt_payload=_object(
                        {
                            "epistemic_status": "synthetic_preparation_probe",
                            "provider_calls": 0,
                            "abc_executions": 0,
                        }
                    ),
                    assigned_score=0.0,
                )
                support = None
                for family in tuple(
                    sorted({value.family for value in variation.contract.options})
                ):
                    for path in REFLECTION_DECISION_PATHS:
                        try:
                            support = derive_portfolio_memory_dose_card_support(
                                PortfolioMemoryDoseCardSemantics(
                                    card_key=synthetic_card.card_key,
                                    card_content_sha256=(synthetic_card.content_sha256),
                                    affected_paths=(path,),
                                    recommended_option_families=(family,),
                                ),
                                variation.contract,
                            )
                        except ValueError:
                            continue
                        break
                    if support is not None:
                        break
                if support is None:
                    raise RuntimeError(
                        "synthetic G5 dose probe found no compatible finite action"
                    )
                dose = BoundedPortfolioMemoryDoseContract(
                    card_supports=(support,),
                    proposed_supported_member_bounds=(1, 1),
                    evaluated_supported_member_bounds=(1, 1),
                    minimum_unattributed_proposed_members=7,
                    minimum_unattributed_evaluated_members=PORTFOLIO_WIDTH - 1,
                    maximum_cards_per_member=1,
                    require_every_assigned_card=True,
                )
                synthetic_selection = factory._request(
                    campaign_context,
                    (synthetic_card,),
                    memory_dose_contract=dose,
                )
                factory._register(
                    campaign_context,
                    synthetic_selection,
                    diagnostic={
                        "status": "synthetic_g5_bounded_dose_probe",
                        "memory_credit_issued": False,
                    },
                )
                dose_row: dict[str, object] = {
                    "probe_kind": "synthetic_g5_bounded_dose",
                    "generation": generation,
                    "parent_slot": slot,
                    "request_sha256": synthetic_selection.request_sha256,
                    "eligible_option_count": len(variation.contract.options),
                    "parent_measurement_binding_sha256": (
                        parent_measurement.binding_sha256
                    ),
                    "evaluation_width": PORTFOLIO_WIDTH,
                    "variation_source_ids": list(
                        finite_variation_source_ids(variation.contract)
                    ),
                    "contextual_allocation": (
                        None
                        if campaign_context.contextual_allocation is None
                        else campaign_context.contextual_allocation.to_record()
                    ),
                    "memory_dose_contract": dose.to_record(),
                    "memory_dose_support": support.to_record(),
                }
                if bundle.arm == "live":
                    binding = coordinator.binding_for(synthetic_selection)
                    prompt = coordinator.render(synthetic_selection)
                    dose_row.update(
                        {
                            "selection_contract": (
                                f"synthetic_bounded_dose_{_selection_contract_label()}"
                            ),
                            "binding_sha256": binding.binding_sha256,
                            "prompt_sha256": _sha(prompt),
                            "proposal_width": CALIBRATED_PROPOSAL_WIDTH,
                        }
                    )
                else:
                    registered = records[-1]
                    if registered["selection_request_sha256"] != (
                        synthetic_selection.request_sha256
                    ):
                        raise RuntimeError("control dose probe registration drifted")
                    dose_row.update(
                        {
                            "selection_contract": (
                                "synthetic_bounded_dose_construction_only"
                            ),
                            "policy_output_width": PORTFOLIO_WIDTH,
                        }
                    )
                rows.append(dose_row)
    request_hashes = tuple(value["request_sha256"] for value in rows)
    bootstrap_rows = tuple(
        value for value in rows if value["probe_kind"] == "bootstrap_wave"
    )
    dose_rows = tuple(
        value for value in rows if value["probe_kind"] == "synthetic_g5_bounded_dose"
    )
    contextual_plans = (
        ()
        if probe_contextual_planner is None
        else tuple(value.to_record() for value in probe_contextual_planner.plans)
    )
    gates = {
        "exact_bootstrap_wave_count": len(bootstrap_rows) == 6,
        "exact_synthetic_g5_dose_count": len(dose_rows) == 2,
        "exact_portfolio_generations": tuple(
            sorted({value["generation"] for value in bootstrap_rows})
        )
        == PORTFOLIO_GENERATIONS,
        "g3_bootstrap_has_no_admission": all(
            record["diagnostic"]["status"]
            == "delayed_reflection_bootstrap_not_yet_admitted"
            for row, record in zip(rows, records, strict=True)
            if row["generation"] == 3
        ),
        "exact_g5_dose_bounds": all(
            value["memory_dose_contract"]["proposed_supported_member_bounds"] == [1, 1]
            and value["memory_dose_contract"]["evaluated_supported_member_bounds"]
            == [1, 1]
            and value["memory_dose_contract"]["minimum_unattributed_proposed_members"]
            == 7
            and value["memory_dose_contract"]["minimum_unattributed_evaluated_members"]
            == PORTFOLIO_WIDTH - 1
            for value in dose_rows
        ),
        "unique_request_hashes": len(set(request_hashes)) == len(request_hashes),
        "arm_registration_contract": coordinator.registered_request_count
        == (len(rows) if bundle.arm == "live" else 0),
        "arm_record_contract": (
            all(
                row["prompt_sha256"] == record["calibrated_prompt_sha256"]
                for row, record in zip(rows, records, strict=True)
            )
            if bundle.arm == "live"
            else all(
                row["request_sha256"] == record["selection_request_sha256"]
                and record["selection_policy"] == "outcome_blind_random_k8"
                for row, record in zip(rows, records, strict=True)
            )
        ),
        "arm_width_contract": all(
            row["evaluation_width"] == PORTFOLIO_WIDTH
            and (
                row.get("proposal_width") == CALIBRATED_PROPOSAL_WIDTH
                if bundle.arm == "live"
                else row.get("policy_output_width") == PORTFOLIO_WIDTH
            )
            for row in rows
        ),
        "all_parent_measurements_bound": all(
            row["parent_measurement_binding_sha256"] is not None for row in rows
        ),
        "contextual_variation_source_contract": (
            all(
                row["contextual_allocation"] is not None
                and len(row["variation_source_ids"]) >= 2
                and {
                    value[0]
                    for value in row["contextual_allocation"][
                        "source_target_counts"
                    ]
                }
                == set(row["variation_source_ids"])
                for row in rows
            )
            if CONTEXTUAL_SEARCH_ALLOCATION
            else all(row["contextual_allocation"] is None for row in rows)
        ),
        "prospective_joint_capability_evidence": (
            len(contextual_plans) == len(PORTFOLIO_GENERATIONS)
            and all(
                len(
                    plan["stage_allocation"]["decision"]["query"][
                        "joint_count_capabilities"
                    ]
                )
                == PARENTS_PER_PORTFOLIO
                and len(
                    plan["stage_allocation"]["decision"][
                        "joint_capability_selection"
                    ]
                )
                == PARENTS_PER_PORTFOLIO
                for plan in contextual_plans
            )
            if CONTEXTUAL_SEARCH_ALLOCATION
            else not contextual_plans
        ),
        "reflection_construction": reflection["all_gates_pass"],
    }
    return {
        "status": "provider_credential_abc_free_all_wave_construction",
        "probe_contract": {
            "probe_id": CONSTRUCTION_PROBE_ID,
            "probe_version": CONSTRUCTION_PROBE_VERSION,
            "definition_sha256": CONSTRUCTION_PROBE_DEFINITION_SHA256,
        },
        "provider_calls": 0,
        "credential_read": False,
        "abc_executions": 0,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "rows": rows,
        "contextual_plans": list(contextual_plans),
        "reflection_probe": reflection,
    }


def _construction_probe_commitment(
    probe: dict[str, object],
) -> dict[str, object]:
    """Bind preregistration to the exact successful construction probe."""

    if type(probe) is not dict or probe.get("all_gates_pass") is not True:
        raise ValueError("construction probe must be an exact successful record")
    expected_contract = {
        "probe_id": CONSTRUCTION_PROBE_ID,
        "probe_version": CONSTRUCTION_PROBE_VERSION,
        "definition_sha256": CONSTRUCTION_PROBE_DEFINITION_SHA256,
    }
    if probe.get("probe_contract") != expected_contract:
        raise ValueError("construction probe contract identity drifted")
    reflection_probe = probe.get("reflection_probe")
    if type(reflection_probe) is not dict:
        raise TypeError("construction probe lost reflection evidence")
    request = reflection_probe.get("request")
    evidence = reflection_probe.get("evidence")
    if type(request) is not dict or type(evidence) is not dict:
        raise TypeError("construction probe lost request/evidence identities")
    request_identity = request.get("request_identity_sha256")
    evidence_identity = evidence.get("snapshot_sha256")
    for name, value in (
        ("reflection_request_identity_sha256", request_identity),
        ("reflection_evidence_snapshot_sha256", evidence_identity),
    ):
        if type(value) is not str or len(value) != 64:
            raise ValueError(f"construction probe {name} is unavailable")
    return {
        **expected_contract,
        "probe_sha256": typed_json_sha256(freeze_json(probe)),
        "reflection_request_identity_sha256": request_identity,
        "reflection_evidence_snapshot_sha256": evidence_identity,
    }


def _manifest(
    *,
    run_id: str,
    mode: str,
    source: dict[str, object],
    source_snapshot: dict[str, object],
) -> dict[str, object]:
    projection = _option_prompt_projection()
    provider = _provider_config()
    archive_context = (
        AuthenticatedAffineFrontierContextProjector()
        if mode == "live"
        and (
            ACQUISITION_MODE is CampaignAcquisitionMode.TARGET_CONDITIONED
            or PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
        )
        else affine_frontier_context_projector(ARCHIVE_CONTEXT_MODE)
    )
    finite_variation_catalog = _finite_variation_catalog()
    _, shared_initial_design = _load_shared_initial_design()
    shared_design_record = (
        None if shared_initial_design is None else thaw_json(shared_initial_design)
    )
    workload_definition_sha256 = (
        WORKLOAD_DEFINITION_SHA256
        if shared_design_record is None
        else shared_initial_design_workload_definition_sha256(
            seed_design_sha256=str(shared_design_record["design_sha256"]),
            seed_count=int(shared_design_record["candidate_count"]),
        )
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "created_at_utc": _utc_now(),
        "claim_boundary": {
            "workflow_development_only": True,
            "paper_ready_result": False,
            "memory_efficacy_claim": False,
            "matched_control_is_separate_run": True,
            "single_seed_strong_baseline_repair": GLOBAL_LOCAL_INITIAL_DESIGN,
            "sota_claim": False,
        },
        "workload": {
            "workload_id": WORKLOAD_ID,
            "workload_definition_sha256": workload_definition_sha256,
            "initial_design": shared_design_record,
            "circuit_names": ["log2"],
            "finite_catalog_id": FINITE_CATALOG_ID,
            "finite_catalog_version": finite_variation_catalog.catalog_version,
            "finite_catalog_definition_sha256": (
                finite_variation_catalog.definition_sha256
            ),
            "composite_option_count": COMPOSITE_OPTION_COUNT,
            "composition_selection_exposure": (COMPOSITION_SELECTION_EXPOSURE.value),
            "required_composite_proposals": (
                VARIATION_TOPOLOGY.hierarchical_composition_required_proposals
            ),
            "variation_topology": VARIATION_TOPOLOGY.to_record(),
            "objectives": list(OBJECTIVE_IDS),
            "real_evaluator_receipts": "evaluator_receipts/",
            "real_evaluator_observations": "real_evaluator_observations.jsonl",
        },
        "protocol": {
            "protocol_id": PROTOCOL_ID,
            "generations": GENERATION_COUNT,
            "required_seed_count": REQUIRED_SEED_COUNT,
            "portfolio_generations": list(PORTFOLIO_GENERATIONS),
            "recombination_generations": list(RECOMBINATION_GENERATIONS),
            "parents_per_portfolio": PARENTS_PER_PORTFOLIO,
            "model_proposal_width": CALIBRATED_PROPOSAL_WIDTH,
            "common_candidate_pool_size": (
                COMMON_CANDIDATE_POOL_SIZE if COMMON_POOL_ACQUISITION else None
            ),
            "common_candidate_pool_mode": (
                "disabled"
                if not COMMON_POOL_ACQUISITION
                else (
                    "complete_finite_contract"
                    if COMMON_CANDIDATE_POOL_SIZE is None
                    else "fixed_size"
                )
            ),
            "engine_evaluation_width": PORTFOLIO_WIDTH,
            "recombinations_per_parent": RECOMBINATIONS_PER_PARENT,
            "planned_unique_evaluations": PLANNED_UNIQUE_EVALUATIONS,
            "planned_candidate_occurrences": PLANNED_UNIQUE_EVALUATIONS,
            "maximum_cache_reuse_occurrences": MAX_CACHE_REUSE_OCCURRENCES,
            "planned_logical_calls": PLANNED_LOGICAL_CALLS,
            "cadence": {
                "policy_id": SealedCutoffDelayedAdmissionCadence().policy_id,
                "policy_version": (
                    SealedCutoffDelayedAdmissionCadence().policy_version
                ),
                "definition_sha256": (
                    SealedCutoffDelayedAdmissionCadence().definition_sha256
                ),
            },
            "terminal_reflection_policy": (
                TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER.value
            ),
            "reflection_chronology": {
                "source_generation": 2,
                "sealed_evidence_portfolio_generation": 1,
                "promotion_barrier_generation": 4,
                "first_consumer_generation": 5,
                "terminal_reflection": False,
            },
        },
        "model": {
            **provider.to_manifest_record(),
            "execution_profile": MODEL_EXECUTION_PROFILE.to_record(),
            "resolved_provider_required": RESOLVED_PROVIDER,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature_hex": (None if TEMPERATURE is None else TEMPERATURE.hex()),
            "reasoning_mode": None,
        },
        "queue": {
            "max_connections": AGENT_CONCURRENCY,
            "max_pending": AGENT_QUEUE_CAPACITY,
            "max_attempts": MAX_ATTEMPTS,
            "base_backoff_ns": BASE_BACKOFF_NS,
            "max_backoff_ns": MAX_BACKOFF_NS,
            "retry_mode": provider.retry_mode.value,
            "exponential_backoff": True,
        },
        "calibrated_selection": {
            "implementation_mode": PORTFOLIO_SELECTOR_MODE,
            "outcome_conditioned_all_action": (
                None
                if PORTFOLIO_SELECTOR_MODE != "outcome_conditioned"
                else {
                    "base_policy_definition_sha256": (
                        OUTCOME_CONDITIONED_PORTFOLIO_POLICY_DEFINITION_SHA256
                    ),
                    "forecast_policy_definition_sha256": (
                        ACTION_FORECAST_POLICY_DEFINITION_SHA256
                    ),
                    "partition_policy_definition_sha256": (
                        ACTION_FORECAST_PARTITION_DEFINITION_SHA256
                    ),
                    "max_rows_per_block": ACTION_FORECAST_BLOCK_ROWS,
                    "all_finite_actions_forecast": True,
                    "terminal_probe_quota": 0,
                    "optional_workload_action_semantics": False,
                    "optional_exact_metric_projection": False,
                }
            ),
            "archive_context_projection": (
                {"mode": ARCHIVE_CONTEXT_MODE.value, "enabled": False}
                if archive_context is None
                else {
                    "mode": ARCHIVE_CONTEXT_MODE.value,
                    "enabled": True,
                    "projector_id": archive_context.projector_id,
                    "projector_version": archive_context.projector_version,
                    "definition_sha256": archive_context.definition_sha256,
                }
            ),
            "selector_policy_definition_sha256": (_selector_policy_definition_sha256()),
            "prompt_definition_sha256": (
                calibrated_portfolio_prompt_definition_sha256(
                    projection,
                    proposal_support=_proposal_support_policy() is not None,
                    hierarchical_composition_required_proposals=(
                        _hierarchical_prompt_composition_count()
                    ),
                    feasibility_witness_mode=FEASIBILITY_WITNESS_MODE,
                    constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
                )
            ),
            "bounded_dose_prompt_definition_sha256": (
                calibrated_portfolio_prompt_definition_sha256(
                    projection,
                    bounded_memory_dose=True,
                    proposal_support=_proposal_support_policy() is not None,
                    hierarchical_composition_required_proposals=(
                        _hierarchical_prompt_composition_count()
                    ),
                    feasibility_witness_mode=FEASIBILITY_WITNESS_MODE,
                    constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
                )
            ),
            "feasibility_witness_mode": FEASIBILITY_WITNESS_MODE.value,
            "allocator": _allocator().to_record(),
            "option_prompt_projection": {
                "policy_id": projection.policy_id,
                "policy_version": projection.policy_version,
                "definition_sha256": projection.definition_sha256,
                "configuration_sha256": projection.configuration_sha256,
                "metadata_keys": list(projection.metadata_keys or ()),
            },
            "hard_allocation_contract": {
                "exact_subset_size": PORTFOLIO_WIDTH,
                "pairwise_disjoint_parent_patches": True,
                "minimum_distinct_families": 3,
                "bootstrap_requires_card_attribution": False,
                "per_member_card_claim_required": False,
            },
            "g5_memory_treatment": {
                "mode": (
                    "matched_active_neutral_if_shared_support_else_"
                    "independent_exact_lane_supported_optimization_exposure"
                ),
                "active_arm_hard_dose_enabled": True,
                "neutral_arm_hard_dose_enabled": False,
                "same_required_common_pool_actions": True,
                "proposed_supported_member_bounds": [1, 1],
                "evaluated_supported_member_bounds": [1, 1],
                "minimum_unattributed_proposed_members": 7,
                "minimum_unattributed_evaluated_members": PORTFOLIO_WIDTH - 1,
                "maximum_cards_per_member": 1,
                "require_every_assigned_card": True,
                "dose_policy": {
                    "policy_id": BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_ID,
                    "policy_version": BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_VERSION,
                    "definition_sha256": (
                        BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_DEFINITION_SHA256
                    ),
                },
                "support_policies": {
                    "advisory_local_intervention": {
                        "policy_id": PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID,
                        "policy_version": (
                            PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION
                        ),
                        "definition_sha256": (
                            PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256
                        ),
                    },
                    "forced_exact_source_parent": {
                        "policy_id": (
                            EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID
                        ),
                        "policy_version": (
                            EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION
                        ),
                        "definition_sha256": (
                            EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256
                        ),
                    },
                    "context_transfer_assessment": {
                        "policy_id": PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_ID,
                        "policy_version": (
                            PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_VERSION
                        ),
                        "definition_sha256": (
                            PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_DEFINITION_SHA256
                        ),
                    },
                },
                "randomized_active_neutral_pair": {
                    "policy_id": PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID,
                    "policy_version": (PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION),
                    "definition_sha256": (
                        PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256
                    ),
                    "one_source_bound_card": True,
                    "two_stable_lane_units": True,
                    "same_parent_matched": False,
                    "full_candidate_pool_matched": False,
                    "single_block_card_effect_identified": False,
                    "online_score_update_allowed": False,
                    "reward_aggregation_id": MEMORY_AGGREGATION_ID,
                    "reward_aggregation_definition_sha256": (
                        MEMORY_AGGREGATION_DEFINITION_SHA256
                    ),
                },
                "no_shared_support_recourse": {
                    "policy_id": PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_ID,
                    "policy_version": PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_VERSION,
                    "definition_sha256": (
                        PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_DEFINITION_SHA256
                    ),
                    "administer_each_exactly_supported_lane": True,
                    "optimization_exposure_only": True,
                    "card_vs_neutral_effect_identified": False,
                    "online_causal_credit_allowed": False,
                },
            },
        },
        "reflection": {
            "semantic_contract_version": 3,
            "insight_cohort_bounds": [2, 8],
            "decision_paths": list(REFLECTION_DECISION_PATHS),
            "option_families": list(REFLECTION_OPTION_FAMILIES),
            "contract_identity_sha256": boils_reflection_contract().identity_sha256,
            "fact_schema": {
                "schema_id": IDENTIFIABLE_REFLECTION_FACT_SCHEMA_ID,
                "schema_version": IDENTIFIABLE_REFLECTION_FACT_SCHEMA_VERSION,
                "definition_sha256": (
                    IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256
                ),
            },
            "request_builder": {
                "builder_id": IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_ID,
                "builder_version": (IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_VERSION),
                "definition_sha256": (
                    IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_DEFINITION_SHA256
                ),
            },
            "evidence_policy": {
                "policy_id": IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_ID,
                "policy_version": IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_VERSION,
                "definition_sha256": (
                    IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_DEFINITION_SHA256
                ),
            },
            "source": "sealed_direct_single_mutation_observations",
            "non_single_intervention_handling": (
                "typed_exclusion_from_single_mutation_reflection_with_exact_"
                "source_partition_accounting"
            ),
            "recombination_results_exposed": False,
            "citation_mode": "authenticated_request_scoped_short_keys_v1",
            "visibility": "g2_quarantined_g4_admitted_g5_first_consumer",
        },
        "construction_probe": {
            "probe_id": CONSTRUCTION_PROBE_ID,
            "probe_version": CONSTRUCTION_PROBE_VERSION,
            "definition_sha256": CONSTRUCTION_PROBE_DEFINITION_SHA256,
            "content_identity_deferred_until_preparation": True,
        },
        "utility": _affine_spec().to_record(),
        "utility_definition_sha256": _affine_spec().definition_sha256,
        "durable_journals": {
            "preparation": "preparation.jsonl",
            "request": "request_evidence.jsonl",
            "output": "output_evidence.jsonl",
            "outcome": "queue_outcomes.jsonl",
            "outbound": "outbound_requests.jsonl",
            "progress": "stream_progress.jsonl",
            "wave_preparation": "wave_preparations.jsonl",
            "campaign": "campaign_events.jsonl",
            "engine": "engine_events.jsonl",
        },
        "prepare_contract": {
            "credential_read": False,
            "provider_calls": 0,
            "abc_executions": 0,
            "cryptographic_evaluator_preflight": True,
        },
        "source_identity": source,
        "source_snapshot": source_snapshot,
    }


def _execution_exit_code(status: object) -> int:
    """Make terminal health visible to shell schedulers after finalization."""

    if status == "completed_healthy":
        return 0
    if status == "completed_unhealthy":
        return 2
    raise RuntimeError("execution summary has an unknown terminal status")


def _is_exact_active_neutral_arm_pair(values: tuple[object, ...]) -> bool:
    """Check canonical serialized arm values rather than display labels."""

    return (
        len(values) == 2
        and values.count(PortfolioExperimentalArm.MEMORY.value) == 1
        and values.count(PortfolioExperimentalArm.NEUTRAL.value) == 1
    )


def _read_live_api_key() -> str:
    """The sole credential-read boundary; prepare and control never call it."""

    load_dotenv(WORKSPACE_ROOT / ".env", override=False)
    load_dotenv(AGENT_EVOLVE_ROOT / ".env", override=False)
    value = os.environ.get("OPENROUTER_API_KEY")
    if type(value) is not str or not value:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")
    return value


def _preregistration_contract(
    *,
    bundle: _Bundle,
    source_aggregate_sha256: str,
    construction_probe: dict[str, object],
) -> dict[str, object]:
    """Return the exact machine-checkable contract required by live mode."""

    return {
        "schema_version": 1,
        "experiment_id": (PROTOCOL_ID + "_" + MODEL_EXECUTION_PROFILE.profile_id),
        "arm": "live",
        "experiment_profile": (
            None
            if bundle.experiment_profile is None
            else bundle.experiment_profile.to_record()
        ),
        "experiment_profile_conformance": (
            None
            if bundle.experiment_profile is None
            else bundle.experiment_profile.prepared_conformance_record(
                prepared=bundle.prepared,
                archive_utility=bundle.utility,
                outer_seed=OUTER_SEED,
            )
        ),
        "source_aggregate_sha256": source_aggregate_sha256,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": bundle.prepared.protocol.protocol_sha256,
        "preparation_sha256": bundle.prepared.preparation_sha256,
        "model": MODEL,
        "provider_only": list(PROVIDER_ONLY),
        "reasoning_effort": MODEL_EXECUTION_PROFILE.reasoning_effort,
        "reasoning_mode": None,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "outer_seed": OUTER_SEED,
        "initial_design": (
            None
            if getattr(bundle, "shared_initial_design", None) is None
            else thaw_json(bundle.shared_initial_design)
        ),
        "required_seed_count": REQUIRED_SEED_COUNT,
        "planned_unique_evaluations": PLANNED_UNIQUE_EVALUATIONS,
        "planned_candidate_occurrences": PLANNED_UNIQUE_EVALUATIONS,
        "maximum_cache_reuse_occurrences": MAX_CACHE_REUSE_OCCURRENCES,
        "planned_logical_calls": PLANNED_LOGICAL_CALLS,
        "model_proposal_width": CALIBRATED_PROPOSAL_WIDTH,
        "common_candidate_pool_size": (
            COMMON_CANDIDATE_POOL_SIZE if COMMON_POOL_ACQUISITION else None
        ),
        "common_candidate_pool_mode": (
            "disabled"
            if not COMMON_POOL_ACQUISITION
            else (
                "complete_finite_contract"
                if COMMON_CANDIDATE_POOL_SIZE is None
                else "fixed_size"
            )
        ),
        "engine_evaluation_width": PORTFOLIO_WIDTH,
        "selection_policy": {
            "selector_policy_definition_sha256": (_selector_policy_definition_sha256()),
            "prompt_definition_sha256": (
                calibrated_portfolio_prompt_definition_sha256(
                    _option_prompt_projection(),
                    proposal_support=_proposal_support_policy() is not None,
                    hierarchical_composition_required_proposals=(
                        _hierarchical_prompt_composition_count()
                    ),
                    feasibility_witness_mode=FEASIBILITY_WITNESS_MODE,
                    constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
                )
            ),
            "allocator": _allocator().to_record(),
            "common_candidate_pool": (
                None
                if _common_candidate_pool_policy() is None
                else _common_candidate_pool_policy().to_record()
            ),
        },
        "portfolio_generations": list(PORTFOLIO_GENERATIONS),
        "recombination_generations": list(RECOMBINATION_GENERATIONS),
        "cadence_definition_sha256": (
            SealedCutoffDelayedAdmissionCadence().definition_sha256
        ),
        "terminal_reflection_policy": (
            TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER.value
        ),
        "identifiable_evidence_policy_definition_sha256": (
            IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_DEFINITION_SHA256
        ),
        "identifiable_fact_schema_definition_sha256": (
            IDENTIFIABLE_REFLECTION_FACT_SCHEMA_DEFINITION_SHA256
        ),
        "identifiable_request_builder": {
            "builder_id": IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_ID,
            "builder_version": IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_VERSION,
            "definition_sha256": (
                IDENTIFIABLE_REFLECTION_REQUEST_BUILDER_DEFINITION_SHA256
            ),
        },
        "reflection_insight_cohort_bounds": [2, 8],
        "construction_probe": _construction_probe_commitment(construction_probe),
        "bounded_memory_dose_policy_definition_sha256": (
            BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_DEFINITION_SHA256
        ),
        "memory_dose_support_policy_definition_sha256": (
            PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256
        ),
        "diagnostic_memory_assignment_policy": {
            "policy_id": PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID,
            "policy_version": PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION,
            "definition_sha256": (PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256),
            "active_neutral_pair": True,
            "one_card_supported_in_both_lanes_required_for_matched_assay": True,
            "no_shared_support_recourse": {
                "policy_id": PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_ID,
                "policy_version": PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_VERSION,
                "definition_sha256": (
                    PORTFOLIO_MEMORY_LANE_SUPPORT_POLICY_DEFINITION_SHA256
                ),
                "administer_each_exactly_supported_lane": True,
                "optimization_exposure_only": True,
                "card_vs_neutral_effect_identified": False,
                "online_causal_credit_allowed": False,
            },
            "single_block_card_effect_identified": False,
            "online_score_update_allowed": False,
        },
        "g5_memory_treatment": {
            "mode": (
                "matched_active_neutral_if_shared_support_else_"
                "independent_exact_lane_supported_optimization_exposure"
            ),
            "active_arm_hard_dose_enabled": True,
            "neutral_arm_hard_dose_enabled": False,
            "same_required_common_pool_actions": True,
            "dose_bounds_if_enabled": {
                "proposed_supported_members": [1, 1],
                "evaluated_supported_members": [1, 1],
                "minimum_unattributed_proposed_members": 7,
                "minimum_unattributed_evaluated_members": PORTFOLIO_WIDTH - 1,
            },
        },
        "claim_boundary": (
            "single_seed_shared_initial_design_strong_baseline_repair_not_sota"
            if GLOBAL_LOCAL_INITIAL_DESIGN
            else "developmental_trace_validation_not_efficacy"
        ),
    }


def _validate_preregistration(
    *,
    path: Path,
    bundle: _Bundle,
    source_aggregate_sha256: str,
    construction_probe: dict[str, object],
) -> dict[str, object]:
    """Fail before credential access when the external preregistration drifts."""

    resolved = path.expanduser().resolve(strict=True)
    if WORKSPACE_ROOT not in resolved.parents:
        raise RuntimeError("preregistration must live inside the workspace")
    try:
        payload = resolved.read_bytes()
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("preregistration must be strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise RuntimeError("preregistration root must be an object")
    expected = _preregistration_contract(
        bundle=bundle,
        source_aggregate_sha256=source_aggregate_sha256,
        construction_probe=construction_probe,
    )
    if value != expected:
        raise RuntimeError("preregistration differs from the exact prepared contract")
    return {
        "path": resolved.relative_to(WORKSPACE_ROOT).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "validated_contract": expected,
    }


def _open_execution_journals(run_dir: Path) -> dict[str, Any]:
    return {
        "engine": DurableJsonlJournal(run_dir / "engine_events.jsonl"),
        "campaign": DurableJsonlJournal(run_dir / "campaign_events.jsonl"),
        "requests": DurableJsonlJournal(run_dir / "request_evidence.jsonl"),
        "outputs": DurableJsonlJournal(run_dir / "output_evidence.jsonl"),
        "outcomes": DurableJsonlJournal(run_dir / "queue_outcomes.jsonl"),
        "outbound": DurableJsonlJournal(run_dir / "outbound_requests.jsonl"),
        "progress": BatchedDurableJsonlJournal(
            run_dir / "stream_progress.jsonl", max_unfsynced_rows=32
        ),
        "wave_preparations": DurableJsonlJournal(run_dir / "wave_preparations.jsonl"),
    }


def _candidate_outcome_accounting(
    history: tuple[EvolutionCandidate, ...] | list[EvolutionCandidate],
) -> dict[str, object]:
    """Separate scientific infeasibility from orchestration failure evidence."""

    scored = 0
    candidate_infeasible = 0
    runtime_failures: list[dict[str, object]] = []
    for candidate in history:
        if type(candidate) is not EvolutionCandidate:
            raise TypeError("candidate history contains a foreign value")
        detailed = candidate.detailed_evaluation
        failure = None if detailed is None else detailed.payload.failure
        if candidate.valid and detailed is not None and failure is None:
            scored += 1
            continue
        if _is_typed_candidate_infeasible(candidate):
            candidate_infeasible += 1
            continue
        runtime_failures.append(
            {
                "candidate_id": candidate.candidate_id.value,
                "candidate_valid": candidate.valid,
                "detailed_evaluation_present": detailed is not None,
                "failure_category": (
                    None if failure is None else failure.category.value
                ),
                "failure_code": None if failure is None else failure.code.value,
            }
        )
    return {
        "evaluated_count": len(history),
        "scored_count": scored,
        "typed_candidate_infeasible_count": candidate_infeasible,
        "runtime_failure_count": len(runtime_failures),
        "runtime_failures": runtime_failures,
    }


def _is_typed_candidate_infeasible(candidate: EvolutionCandidate) -> bool:
    """Return whether one terminal outcome is candidate-attributable failure."""

    if type(candidate) is not EvolutionCandidate:
        raise TypeError("candidate must be an exact EvolutionCandidate")
    detailed = candidate.detailed_evaluation
    failure = None if detailed is None else detailed.payload.failure
    return (
        not candidate.valid
        and detailed is not None
        and failure is not None
        and failure.category is FailureCategory.CANDIDATE
    )


async def _execute(
    *,
    bundle: _Bundle,
    run_dir: Path,
    journals: dict[str, Any],
    expected_source_sha256: str,
) -> dict[str, object]:
    started_ns = time.perf_counter_ns()

    def observed(value: dict[str, object]) -> dict[str, object]:
        return {
            "observation": {
                "monotonic_ns_since_execution_start": (
                    time.perf_counter_ns() - started_ns
                ),
                "observed_at_utc": _utc_now(),
            },
            "authenticated_record": value,
        }

    runner = None
    owned = None
    reflection_records: list[dict[str, object]] = []
    wave_records: list[dict[str, object]] = []
    _require_source_closure(expected_source_sha256)
    optimization_semantics = bundle.benchmark.optimization_semantics
    if optimization_semantics is None:
        raise RuntimeError("BOiLS reflection requires optimization semantics")
    if bundle.arm == "live":
        api_key = _read_live_api_key()

        def progress_sink(value: StructuredStreamProgress) -> None:
            journals["progress"].append(observed(_progress_record(value)))

        def outcome_sink(value: object) -> None:
            journals["progress"].flush()
            journals["outcomes"].append(
                observed(structured_generation_outcome_record(value))
            )

        runner = create_progress_aware_openrouter_runner(
            api_key=api_key,
            config=_provider_config(),
            progress_sink=progress_sink,
            outcome_sink=outcome_sink,
            request_evidence_sink=lambda value: journals["requests"].append(
                observed(value)
            ),
            output_evidence_sink=lambda value: journals["outputs"].append(
                observed(value)
            ),
            outbound_request_manifest_sink=lambda value: journals["outbound"].append(
                observed(value)
            ),
            evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
        )
        generator = PydanticAIAgenticGenerator(runner)
        selector = (
            _outcome_conditioned_selector(runner=runner, bundle=bundle)
            if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
            else bundle.coordinator.build_selector(runner)
        )
        expected_selector_type = (
            PydanticAIContextualSearchAllocationPortfolioSelectionPolicy
            if CONTEXTUAL_SEARCH_ALLOCATION
            else PydanticAIConstraintDecoupledTargetConditionedPortfolioSelectionPolicy
            if (
                CONSTRAINT_DECOUPLED_ACQUISITION
                and ACQUISITION_MODE is CampaignAcquisitionMode.TARGET_CONDITIONED
            )
            else PydanticAITargetConditionedCalibratedPortfolioSelectionPolicy
            if ACQUISITION_MODE is CampaignAcquisitionMode.TARGET_CONDITIONED
            else PydanticAIEvidenceCalibratedSourceMixPortfolioSelectionPolicy
            if EVIDENCE_CALIBRATED_SOURCE_MIX
            else PydanticAIMinimumInterventionHorizonPortfolioSelectionPolicy
            if MINIMUM_INTERVENTION_PROJECTION
            else PydanticAIConstraintDecoupledHorizonPortfolioSelectionPolicy
            if CONSTRAINT_DECOUPLED_ACQUISITION
            else {
                FullSupportSlatePolicy: (
                    PydanticAIFullSupportCalibratedPortfolioSelectionPolicy
                ),
                ModelAnchoredCalibratedSlatePolicy: (
                    PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy
                ),
                StructuralPosteriorSlatePolicy: (
                    PydanticAIStructuralPosteriorCalibratedPortfolioSelectionPolicy
                ),
                OperatorStratifiedStructuralPosteriorSlatePolicy: (
                    PydanticAIOperatorStratifiedCalibratedPortfolioSelectionPolicy
                ),
                HorizonBoundedStructuralPosteriorSlatePolicy: (
                    PydanticAIHorizonBoundedCalibratedPortfolioSelectionPolicy
                ),
            }[type(_allocator())]
        )
        if (
            PORTFOLIO_SELECTOR_MODE == "calibrated"
            and type(selector) is not expected_selector_type
        ):
            raise TypeError("BOiLS coordinator built a foreign selector")
        reflection_executor = _LiveReflectionExecutor(
            generator,
            bundle.ids,
            reflection_records,
            optimization_semantics,
        )
        owned = _OwnedRunner(runner)
    else:
        generator = None
        selector = DeterministicRandomFeasiblePortfolioPolicy(seed=OUTER_SEED)
        reflection_executor = _ControlReflectionExecutor(
            bundle.ids,
            reflection_records,
            optimization_semantics,
        )

    class _NeverGenerator:
        async def propose(self, request):  # pragma: no cover - finite portfolio path.
            raise AssertionError(f"unexpected propose request: {request}")

        async def reflect(self, request):  # pragma: no cover - external executor.
            raise AssertionError(f"unexpected engine reflection: {request}")

    composition = compose_portfolio_evolution(
        bundle.benchmark,
        generator=_NeverGenerator() if generator is None else generator,
        selector=selector,
        seed=OUTER_SEED,
        id_factory=bundle.ids,
        memory=bundle.memory,
        evaluator_concurrency=EVALUATOR_CONCURRENCY,
        engine_trace_sink=lambda value: journals["engine"].append(
            observed(dict(value))
        ),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
    )
    learning, evidence, identifiable_evidence_source = _production_learning_runtime(
        bundle
    )
    target_controller = (
        None
        if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
        else _target_conditioned_controller(bundle, bundle.coordinator)
    )
    contextual_ledger = (
        ContextualSearchLedger()
        if bundle.arm == "live"
        and CONTEXTUAL_SEARCH_ALLOCATION
        and PORTFOLIO_SELECTOR_MODE == "calibrated"
        else None
    )
    contextual_scope_sha256 = _sha(
        "agent-evolve:contextual-search-campaign:" + bundle.prepared.preparation_sha256
    )
    contextual_planner = (
        None
        if contextual_ledger is None
        else CampaignContextualSearchPlanner(
            ledger=contextual_ledger,
            campaign_scope_sha256=contextual_scope_sha256,
            joint_capability_projector=_contextual_joint_capability_projector(),
            frontier_target_allocator=_frontier_target_allocator(),
        )
    )
    wave_factory = _WaveFactory(
        bundle=bundle,
        learning_runtime=learning,
        records=wave_records,
        target_conditioned_controller=target_controller,
    )
    runtime_archive_context_projector = (
        AuthenticatedAffineFrontierContextProjector()
        if target_controller is not None
        or PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
        else affine_frontier_context_projector(ARCHIVE_CONTEXT_MODE)
        if bundle.experiment_profile is None
        else bundle.experiment_profile.archive_context_projector
    )
    if bundle.experiment_profile is None:
        policies = CampaignPolicies(
            cadence=bundle.policies.cadence,
            parent_selection=bundle.policies.parent_selection,
            memory_assignment=_binding("closed_loop_memory", learning),
            portfolio_selection=bundle.policies.portfolio_selection,
            recombination=bundle.policies.recombination,
            reflection=bundle.policies.reflection,
            reflection_supervision=bundle.policies.reflection_supervision,
            archive_utility=bundle.policies.archive_utility,
        )
    else:
        profile = bundle.experiment_profile
        runtime_profile = rebind_reference_campaign_implementations(
            profile,
            ReferenceCampaignImplementations(
                parent_selection=bundle.parent_selector,
                memory_assignment=learning,
                portfolio_selection=profile.portfolio_selection.implementation,
                recombination=profile.recombination.implementation,
                reflection=reflection_executor,
                archive_context=runtime_archive_context_projector,
                variation_topology=profile.variation_topology,
                contextual_outcomes=profile.contextual_outcomes,
            ),
        )
        runtime_archive_context_projector = runtime_profile.archive_context_projector
        policies = runtime_profile.behavior(archive_utility=bundle.utility).bind()
        runtime_profile.prepared_conformance_record(
            prepared=bundle.prepared,
            archive_utility=bundle.utility,
            outer_seed=OUTER_SEED,
        )
    if policies.policies_sha256 != bundle.prepared.policies_sha256:
        raise RuntimeError("runtime policies differ from preparation")
    outcome_conditioned_scope = (
        _outcome_conditioned_calibration_scope(bundle)
        if bundle.arm == "live"
        and PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
        else None
    )
    calibrated_outcome_updater = (
        None
        if bundle.arm == "control"
        else CalibratedCampaignOutcomeUpdater(
            ledger=bundle.feedback_ledger,
            selected_forecasts=(
                (
                    lambda wave, result, scope=outcome_conditioned_scope: (
                        outcome_conditioned_selected_predictions(
                            scope=scope,
                            wave=wave,
                            result=result,
                        )
                    )
                )
                if outcome_conditioned_scope is not None
                else lambda wave, result: (
                    bundle.coordinator.decode_selected_predictions(result)
                )
            ),
            adjudicator_for=lambda wave, result: bundle.direction_adjudicator,
            **(
                {}
                if contextual_ledger is None
                else {
                    "contextual_ledger": contextual_ledger,
                    "selected_search_sources": (
                        lambda wave, result: (
                            bundle.coordinator.decode_selected_source_ids(result)
                        )
                    ),
                    "selected_allocation_realization": (
                        lambda wave, result: (
                            bundle.coordinator.decode_contextual_allocation_realization(
                                result
                            )
                        )
                    ),
                    "contextual_marginal_utility": (
                        FixedReferenceContextualMarginalUtilityProjector(bundle.utility)
                    ),
                    "contextual_campaign_scope_sha256": (contextual_scope_sha256),
                }
            ),
        )
    )
    outcome_updater = (
        CompositeCampaignPortfolioOutcomeUpdater(
            (calibrated_outcome_updater, target_controller)
        )
        if target_controller is not None
        else calibrated_outcome_updater
    )
    runtime = AgenticPortfolioCampaignRuntime(
        prepared=bundle.prepared,
        workload_config=bundle.config,
        workload_ports=bundle.workload_ports,
        composition=composition,
        parent_selector=bundle.parent_selector,
        wave_factory=wave_factory,
        task_sha256=TASK_SHA256,
        parent_measurement_projection=bundle.parent_measurement_projection,
        archive_context_projector=runtime_archive_context_projector,
        memory_estimand_projector=wave_factory.diagnostic_coordinator,
        identifiable_reflection_executor=reflection_executor,
        identifiable_reflection_evidence_source=identifiable_evidence_source,
        context_enricher=(
            None
            if bundle.arm == "control"
            else ContextualOutcomeCampaignEnricher(
                ledger=bundle.feedback_ledger,
                max_actions=24,
                include_cross_lineage_analogies=False,
            )
        ),
        contextual_search_planner=contextual_planner,
        frontier_target_allocator=(
            ResidualHypervolumeFrontierTargetAllocator()
            if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
            else _frontier_target_allocator()
            if target_controller is not None
            else None
        ),
        outcome_updater=outcome_updater,
        recombination_utility_binder=_RecombinationUtilityBinder(bundle.utility),
        owned_resources=owned,
        selector_request_prompt_renderer=(
            None
            if bundle.arm == "control"
            else selector
            if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
            else bundle.coordinator
        ),
        learning_lifecycle=learning,
        wave_preparation_observer=_WavePreparationJournal(
            journals["wave_preparations"], started_ns
        ),
    )
    started = time.perf_counter()
    result = await EvolutionCampaignScheduler(
        prepared=bundle.prepared,
        policies=policies,
        stages=runtime,
        reflections=runtime,
        lifecycle=runtime,
        journal=_ExecutionJournal(journals["campaign"], started_ns),
    ).run()
    wall_s = time.perf_counter() - started
    if runner is not None:
        await runner.aclose()
    source_after = _require_source_closure(expected_source_sha256)
    stage_counts = [value.candidate_occurrence_count for value in result.stage_receipts]
    stage_unique = [value.unique_evaluation_count for value in result.stage_receipts]
    evaluation_accounting = CampaignEvaluationAccounting(
        planned_candidate_occurrences=PLANNED_UNIQUE_EVALUATIONS,
        seed_occurrences=(result.counters.candidate_occurrences - sum(stage_counts)),
        seed_unique_evaluations=(
            result.counters.unique_evaluations - sum(stage_unique)
        ),
        stage_occurrences=tuple(stage_counts),
        stage_unique_evaluations=tuple(stage_unique),
        candidate_occurrences=result.counters.candidate_occurrences,
        unique_evaluations=result.counters.unique_evaluations,
    )
    detailed_receipts = [
        candidate.detailed_evaluation.payload.receipt
        for candidate in runtime.history
        if candidate.detailed_evaluation is not None
        and candidate.detailed_evaluation.payload.receipt is not None
    ]
    candidate_outcomes = _candidate_outcome_accounting(runtime.history)
    all_candidate_outcomes_terminal_and_typed = (
        candidate_outcomes["evaluated_count"] == PLANNED_UNIQUE_EVALUATIONS
        and candidate_outcomes["runtime_failure_count"] == 0
        and candidate_outcomes["scored_count"]
        + candidate_outcomes["typed_candidate_infeasible_count"]
        == PLANNED_UNIQUE_EVALUATIONS
    )
    outbound = tuple(
        validate_openrouter_outbound_request_manifest_record(
            row["authenticated_record"]
        )
        for row in read_jsonl(run_dir / "outbound_requests.jsonl")
    )
    outbound_gate = (
        bundle.arm == "control"
        and not outbound
        or bundle.arm == "live"
        and bool(outbound)
        and all(
            record["settings"]["model"] == MODEL
            and record["settings"]["provider"]
            == {"only": list(PROVIDER_ONLY), "allow_fallbacks": False}
            and record["settings"]["reasoning"]
            == MODEL_EXECUTION_PROFILE.outbound_reasoning_setting
            and record["settings"]["max_completion_tokens"] == MAX_OUTPUT_TOKENS
            and all(record["forbidden_fields_absent"].values())
            for record in outbound
        )
    )
    outcome_rows = tuple(read_jsonl(run_dir / "queue_outcomes.jsonl"))
    provider_response_gate = _provider_response_telemetry_gate(
        arm=bundle.arm,
        outcome_rows=outcome_rows,
    )
    provider_response_telemetry = [
        row["authenticated_record"]["response"]
        for row in outcome_rows
        if type(row.get("authenticated_record")) is dict
        and type(row["authenticated_record"].get("response")) is dict
    ]
    reference_violations = [
        {
            "candidate_id": candidate.candidate_id.value,
            "objectives": dict(candidate.objective_map),
            "normalized": [
                value.hex()
                for value in _affine_spec().normalize(candidate.objective_map)
            ],
        }
        for candidate in runtime.history
        if candidate.valid
        if any(
            value >= 1.0 for value in _affine_spec().normalize(candidate.objective_map)
        )
    ]
    wave_preparations = runtime.wave_preparation_receipts
    eligible_receipt_counts = {
        generation: tuple(
            len(value.test_eligible_reflection_receipts)
            for value in wave_preparations
            if value.generation == generation
        )
        for generation in PORTFOLIO_GENERATIONS
    }
    forecast_receipt_count = len(bundle.feedback_ledger.receipts)
    forecast_observation_count = len(bundle.feedback_ledger.observations)
    forecast_actions = tuple(
        action
        for receipt in bundle.feedback_ledger.receipts
        for action in receipt.actions
    )
    g5_records = tuple(value for value in wave_records if value["generation"] == 5)
    g5_memory_statuses = tuple(value["diagnostic"]["status"] for value in g5_records)
    allowed_g5_statuses = (
        {"control_ignores_admitted_reflected_memory"}
        if bundle.arm == "control"
        else {
            "applied_randomized_active_neutral_arm",
            "applied_exact_lane_supported_optimization_memory",
            "applied_signed_advisory_optimization_memory",
            "no_lane_support_for_reflected_memory",
        }
    )
    memory_trial_count = len(bundle.memory.trials)
    matched_g5_memory_lane_count = sum(
        value == "applied_randomized_active_neutral_arm" for value in g5_memory_statuses
    )
    optimization_g5_memory_lane_count = sum(
        value == "applied_exact_lane_supported_optimization_memory"
        for value in g5_memory_statuses
    )
    advisory_g5_memory_lane_count = sum(
        value == "applied_signed_advisory_optimization_memory"
        for value in g5_memory_statuses
    )
    administered_g5_memory_lane_count = (
        matched_g5_memory_lane_count
        + optimization_g5_memory_lane_count
        + advisory_g5_memory_lane_count
    )
    typed_infeasible_matching_lane_count = sum(
        value == "no_lane_support_for_reflected_memory" for value in g5_memory_statuses
    )
    matched_plan_sha256s = tuple(
        value["diagnostic"].get("matched_control_plan_sha256")
        for value in g5_records
        if value["diagnostic"]["status"] == "applied_randomized_active_neutral_arm"
    )
    matched_view_sha256s = tuple(
        value["diagnostic"].get("matched_arm_view_sha256")
        for value in g5_records
        if value["diagnostic"]["status"] == "applied_randomized_active_neutral_arm"
    )
    matched_arms = tuple(
        value["diagnostic"].get("experimental_arm")
        for value in g5_records
        if value["diagnostic"]["status"] == "applied_randomized_active_neutral_arm"
    )
    diagnostic_estimand_sha256s = tuple(
        value["diagnostic"].get("estimand_context_sha256")
        for value in g5_records
        if value["diagnostic"]["status"] == "applied_randomized_active_neutral_arm"
    )
    g5_prepared_memory_credits = tuple(
        value.memory_credit_identity
        for value in wave_preparations
        if value.generation == 5 and value.memory_credit_identity is not None
    )
    fallback_matching_records = tuple(
        (
            value["diagnostic"].get("complete_support_resolution")
            if value["diagnostic"]["status"]
            in {
                "applied_exact_lane_supported_optimization_memory",
                "applied_signed_advisory_optimization_memory",
            }
            else value["diagnostic"]
            .get("evidence", {})
            .get("complete_support_resolution")
        )
        for value in g5_records
        if value["diagnostic"]["status"]
        in {
            "applied_exact_lane_supported_optimization_memory",
            "applied_signed_advisory_optimization_memory",
            "no_lane_support_for_reflected_memory",
        }
    )
    lane_support_records = tuple(
        (
            value["diagnostic"].get("lane_support_resolution")
            if value["diagnostic"]["status"]
            in {
                "applied_exact_lane_supported_optimization_memory",
                "applied_signed_advisory_optimization_memory",
            }
            else value["diagnostic"].get("evidence", {}).get("lane_support_resolution")
        )
        for value in g5_records
        if value["diagnostic"]["status"]
        in {
            "applied_exact_lane_supported_optimization_memory",
            "applied_signed_advisory_optimization_memory",
            "no_lane_support_for_reflected_memory",
        }
    )
    g5_stage_result = thaw_json(
        next(value for value in result.stage_receipts if value.generation == 5).result
    )
    g5_closed_loop = g5_stage_result.get("closed_loop_learning")
    g5_generation_audit = (
        {}
        if type(g5_closed_loop) is not dict
        else g5_closed_loop.get("evidence", {}).get(
            "generation_audit_preparation",
            {},
        )
    )
    matched_control_outcomes = tuple(
        g5_generation_audit.get("matched_memory_control_outcomes", ())
    )
    evidence_observations = evidence.observations
    portfolio_candidates = tuple(
        candidate
        for candidate in runtime.history
        if candidate.generation in PORTFOLIO_GENERATIONS
    )
    portfolio_scored_occurrences = sum(
        candidate.valid for candidate in portfolio_candidates
    )
    portfolio_candidate_infeasible_occurrences = sum(
        _is_typed_candidate_infeasible(candidate) for candidate in portfolio_candidates
    )
    reflection_source_generation = bundle.prepared.schedule.reflection_waves[
        0
    ].source_generation
    reflection_source_scored_occurrences = sum(
        candidate.valid and candidate.generation < reflection_source_generation
        for candidate in portfolio_candidates
    )
    expected_mutation_event_indices = {
        candidate.generation for candidate in portfolio_candidates if candidate.valid
    }
    portfolio_evidence_accounting = CampaignPortfolioEvidenceAccounting(
        planned_portfolio_occurrences=(
            len(PORTFOLIO_GENERATIONS) * PARENTS_PER_PORTFOLIO * PORTFOLIO_WIDTH
        ),
        portfolio_scored_occurrences=portfolio_scored_occurrences,
        portfolio_candidate_infeasible_occurrences=(
            portfolio_candidate_infeasible_occurrences
        ),
        authenticated_mutation_observations=len(evidence_observations),
        reflection_source_scored_occurrences=(reflection_source_scored_occurrences),
        reflection_identifiable_contrasts=sum(
            int(value["identifiable_contrast_count"]) for value in reflection_records
        ),
        reflection_typed_exclusions=sum(
            int(value["typed_exclusion_count"]) for value in reflection_records
        ),
        forecast_enabled=bundle.arm == "live",
        planned_selector_receipts=(
            0
            if bundle.arm == "control"
            else len(PORTFOLIO_GENERATIONS) * PARENTS_PER_PORTFOLIO
        ),
        forecast_receipts=forecast_receipt_count,
        forecast_actions=len(forecast_actions),
        forecast_scored_actions=sum(
            action.disposition is PortfolioMemberDisposition.SCORED
            for action in forecast_actions
        ),
        forecast_candidate_infeasible_actions=sum(
            action.disposition is PortfolioMemberDisposition.CANDIDATE_INFEASIBLE
            for action in forecast_actions
        ),
        objective_metric_count=len(OBJECTIVE_IDS),
        forecast_observations=forecast_observation_count,
    )
    expected_stage_counts = [
        (
            PARENTS_PER_PORTFOLIO * PORTFOLIO_WIDTH
            if generation in PORTFOLIO_GENERATIONS
            else PARENTS_PER_PORTFOLIO * RECOMBINATIONS_PER_PARENT
        )
        for generation in range(1, GENERATION_COUNT + 1)
    ]
    contextual_completion_audit = (
        None
        if contextual_planner is None or contextual_ledger is None
        else audit_completed_contextual_search_ledger(
            contextual_ledger,
            campaign_scope_sha256=contextual_planner.campaign_scope_sha256,
            expected_wave_count=len(PORTFOLIO_GENERATIONS),
            expected_post_recombination_wave_indices=tuple(
                generation // 2 for generation in RECOMBINATION_GENERATIONS
            ),
            expected_observation_count=(
                len(PORTFOLIO_GENERATIONS) * PARENTS_PER_PORTFOLIO * PORTFOLIO_WIDTH
            ),
            expected_allocation_realization_count=(
                len(PORTFOLIO_GENERATIONS) * PARENTS_PER_PORTFOLIO
            ),
        )
    )
    health = {
        "exact_generations": result.counters.generations_completed == GENERATION_COUNT,
        "exact_evaluation_accounting": True,
        "bounded_cache_reuse": evaluation_accounting.within_cache_reuse_limit(
            MAX_CACHE_REUSE_OCCURRENCES
        ),
        "exact_stage_counts": stage_counts == expected_stage_counts,
        "exact_logical_calls": result.counters.logical_agent_calls
        == PLANNED_LOGICAL_CALLS,
        "real_evaluator_observations": bundle.evaluator_observer.calls
        == evaluation_accounting.unique_evaluations,
        "real_detailed_receipts": len(detailed_receipts) == PLANNED_UNIQUE_EVALUATIONS,
        "all_candidate_outcomes_terminal_and_typed": (
            all_candidate_outcomes_terminal_and_typed
        ),
        "exact_portfolio_evidence_terminal_partition": (
            portfolio_evidence_accounting.exact_portfolio_outcome_partition
        ),
        "six_wave_preparations": len(wave_preparations)
        == 6
        == len(read_jsonl(run_dir / "wave_preparations.jsonl")),
        "g1_g3_no_reflection_eligibility": (
            eligible_receipt_counts[1] == (0, 0)
            and eligible_receipt_counts[3] == (0, 0)
        ),
        "g5_first_reflection_consumer": eligible_receipt_counts[5] == (1, 1),
        "one_g2_typed_delayed_reflection": (
            len(reflection_records) == 1
            and reflection_records[0]["source_generation"] == 2
            and reflection_records[0]["origin_cutoff_event_index"] == 1
            and portfolio_evidence_accounting.exact_reflection_contrast_accounting
            and 2 <= reflection_records[0]["insight_count"] <= 8
            and not reflection_records[0]["recombination_results_exposed"]
            and len(result.reflection_receipts) == 1
            and result.reflection_receipts[0].source_generation == 2
        ),
        "one_g4_test_admission": (
            len(result.test_admission_receipts) == 1
            and result.test_admission_receipts[0].barrier_generation == 4
        ),
        "no_terminal_reflection": tuple(
            value.source_generation
            for value in bundle.prepared.schedule.reflection_waves
        )
        == (2,),
        "g5_memory_behavior_typed": (
            len(g5_records) == 2
            and set(g5_memory_statuses).issubset(allowed_g5_statuses)
            and (
                bundle.arm == "control"
                and memory_trial_count == 0
                and all(
                    value["diagnostic"]["memory_credit_issued"] is False
                    for value in g5_records
                )
                or administered_g5_memory_lane_count == 2
                and matched_g5_memory_lane_count == 2
                and memory_trial_count == 0
                and all(
                    value["diagnostic"]["memory_credit_issued"] is False
                    for value in g5_records
                )
                or matched_g5_memory_lane_count == 0
                and optimization_g5_memory_lane_count
                + advisory_g5_memory_lane_count
                + typed_infeasible_matching_lane_count
                == 2
                and memory_trial_count == 0
                and all(
                    value["diagnostic"]["memory_credit_issued"] is False
                    for value in g5_records
                )
            )
        ),
        "g5_memory_treatment_conforms": (
            bundle.arm == "control"
            or (
                matched_g5_memory_lane_count == 2
                and _is_exact_active_neutral_arm_pair(matched_arms)
                and len(matched_plan_sha256s) == 2
                and len(set(matched_plan_sha256s)) == 1
                and len(matched_view_sha256s) == 2
                and len(set(matched_view_sha256s)) == 2
                and all(
                    type(value) is str and len(value) == 64
                    for value in matched_view_sha256s
                )
                and sum(
                    value.get("memory_dose_contract") is not None
                    for value in g5_records
                )
                == 1
            )
            or (
                matched_g5_memory_lane_count == 0
                and optimization_g5_memory_lane_count
                + advisory_g5_memory_lane_count
                + typed_infeasible_matching_lane_count
                == 2
                and all(
                    (
                        value.get("memory_dose_contract") is not None
                        if value["diagnostic"]["status"]
                        == "applied_exact_lane_supported_optimization_memory"
                        else value.get("memory_dose_contract") is None
                    )
                    for value in g5_records
                )
            )
        ),
        "g5_matched_control_observation_realized": (
            bundle.arm == "control"
            or matched_g5_memory_lane_count == 0
            or (
                matched_g5_memory_lane_count == 2
                and memory_trial_count == 0
                and len(g5_prepared_memory_credits) == 2
                and all(
                    thaw_json(value).get("evidence_kind")
                    == "randomized_active_neutral_arm"
                    for value in g5_prepared_memory_credits
                )
                and len(diagnostic_estimand_sha256s) == 2
                and len(set(diagnostic_estimand_sha256s)) == 1
                and len(matched_control_outcomes) == 1
                and matched_control_outcomes[0].get("plan_sha256")
                in set(matched_plan_sha256s)
                and matched_control_outcomes[0].get(
                    "single_block_card_effect_identified"
                )
                is False
                and matched_control_outcomes[0].get("online_score_update_allowed")
                is False
            )
        ),
        "typed_no_full_matching_evidence_if_infeasible": (
            bundle.arm == "control"
            or matched_g5_memory_lane_count == 2
            or (
                matched_g5_memory_lane_count == 0
                and len(fallback_matching_records) == 2
                and all(
                    type(value) is dict
                    and value.get("eligible") is False
                    and type(value.get("receipt_sha256")) is str
                    for value in fallback_matching_records
                )
            )
        ),
        "typed_lane_support_evidence_if_matched_recourse": (
            bundle.arm == "control"
            or matched_g5_memory_lane_count == 2
            or (
                matched_g5_memory_lane_count == 0
                and len(lane_support_records) == 2
                and all(
                    type(value) is dict
                    and value.get("eligible")
                    == (
                        g5_records[index]["diagnostic"]["status"]
                        in {
                            "applied_exact_lane_supported_optimization_memory",
                            "applied_signed_advisory_optimization_memory",
                        }
                    )
                    and type(value.get("receipt_sha256")) is str
                    for index, value in enumerate(lane_support_records)
                )
            )
        ),
        "exact_authenticated_mutation_observations": (
            portfolio_evidence_accounting.exact_authenticated_mutation_evidence
            and {value.event_index for value in evidence_observations}
            == expected_mutation_event_indices
        ),
        "exact_forecast_feedback": (
            portfolio_evidence_accounting.exact_forecast_feedback
        ),
        "initial_design_contract_exact": (
            (
                not GLOBAL_LOCAL_INITIAL_DESIGN
                and bundle.shared_initial_design is None
                and len(bundle.prepared.seeds.seeds) == 2
            )
            or (
                GLOBAL_LOCAL_INITIAL_DESIGN
                and bundle.shared_initial_design is not None
                and len(bundle.prepared.seeds.seeds) == REQUIRED_SEED_COUNT
                and thaw_json(bundle.shared_initial_design).get(
                    "outcome_blind_projection"
                )
                is True
            )
        ),
        "source_closure_unchanged": source_after["aggregate_sha256"]
        == expected_source_sha256,
        "outbound_transport_gate": outbound_gate,
        "provider_response_telemetry_gate": provider_response_gate,
        "affine_reference_contains_every_candidate": not reference_violations,
        "contextual_search_closed_loop": (
            not CONTEXTUAL_SEARCH_ALLOCATION
            or bundle.arm == "control"
            or (
                contextual_planner is not None
                and contextual_ledger is not None
                and len(contextual_planner.plans) == len(PORTFOLIO_GENERATIONS)
                and len(contextual_ledger.observations)
                == len(PORTFOLIO_GENERATIONS) * PARENTS_PER_PORTFOLIO * PORTFOLIO_WIDTH
                and contextual_completion_audit is not None
                and contextual_completion_audit.healthy
            )
        ),
        "target_conditioned_closed_loop": (
            target_controller is None
            or (
                target_controller.state.cutoff_generation
                == PORTFOLIO_GENERATIONS[-1]
                and target_controller.state.selected_observation_count
                == len(PORTFOLIO_GENERATIONS)
                * PARENTS_PER_PORTFOLIO
                * PORTFOLIO_WIDTH
            )
        ),
        "cleanup_released": result.cleanup_receipt.released,
    }
    status = "completed_healthy" if all(health.values()) else "completed_unhealthy"
    return {
        "schema_version": 1,
        "status": status,
        "health": health,
        "campaign_result": result.to_record(),
        "initial_design": (
            None
            if bundle.shared_initial_design is None
            else thaw_json(bundle.shared_initial_design)
        ),
        "experiment_profile": (
            None
            if bundle.experiment_profile is None
            else bundle.experiment_profile.to_record()
        ),
        "experiment_profile_conformance": (
            None
            if bundle.experiment_profile is None
            else bundle.experiment_profile.prepared_conformance_record(
                prepared=bundle.prepared,
                archive_utility=bundle.utility,
                outer_seed=OUTER_SEED,
            )
        ),
        "wall_s": wall_s,
        "stage_candidate_counts": stage_counts,
        "stage_unique_evaluation_counts": stage_unique,
        "evaluation_accounting": evaluation_accounting.to_record(),
        "provider_calls": 0 if bundle.arm == "control" else len(outbound),
        "target_conditioned_state": (
            None
            if target_controller is None
            else target_controller.state.to_record()
        ),
        "provider_response_telemetry": provider_response_telemetry,
        "evaluator_observation_count": bundle.evaluator_observer.calls,
        "evaluator_receipt_count": len(detailed_receipts),
        "evaluator_artifact_ids": [
            value.artifact_id.value for value in detailed_receipts
        ],
        "candidate_outcome_accounting": candidate_outcomes,
        "portfolio_evidence_accounting": (portfolio_evidence_accounting.to_record()),
        "wave_records": wave_records,
        "contextual_search_plans": (
            []
            if contextual_planner is None
            else [value.to_record() for value in contextual_planner.plans]
        ),
        "contextual_search_observations": (
            []
            if contextual_ledger is None
            else [value.to_record() for value in contextual_ledger.observations]
        ),
        "contextual_search_delayed_credits": (
            []
            if contextual_ledger is None
            else [value.to_record() for value in contextual_ledger.delayed_credits]
        ),
        "contextual_search_allocation_realizations": (
            []
            if contextual_ledger is None
            else [
                value.to_record() for value in contextual_ledger.allocation_realizations
            ]
        ),
        "contextual_search_completion_audit": (
            None
            if contextual_completion_audit is None
            else contextual_completion_audit.to_record()
        ),
        "reflection_records": reflection_records,
        "memory_trial_count": memory_trial_count,
        "memory_trials": [
            {
                "credit_unit_id": value.credit_unit_id.value,
                "candidate_ids": [item.value for item in value.candidate_ids],
                "reward_definition_hash": value.reward_definition_hash,
                "reward": value.reward,
                "treatment_binding_sha256": value.treatment_binding_sha256,
                "generation": value.generation,
                "eligible_references": [
                    {
                        "insight_id": item.insight_id.value,
                        "version": item.version,
                    }
                    for item in value.decision.eligible
                ],
                "selected_references": [
                    {
                        "insight_id": item.insight_id.value,
                        "version": item.version,
                    }
                    for item in value.decision.selected
                ],
                "selected_subset_probability": str(
                    value.decision.selected_subset_probability
                ),
            }
            for value in bundle.memory.trials
        ],
        "intended_g5_memory_lane_count": 2,
        "administered_g5_memory_lane_count": (administered_g5_memory_lane_count),
        "matched_g5_memory_lane_count": matched_g5_memory_lane_count,
        "optimization_g5_memory_lane_count": optimization_g5_memory_lane_count,
        "advisory_g5_memory_lane_count": advisory_g5_memory_lane_count,
        "typed_infeasible_matching_lane_count": (typed_infeasible_matching_lane_count),
        "g5_causal_memory_credit_claim": False,
        "g5_randomized_active_card_assignment_audited": (
            bundle.arm == "live"
            and matched_g5_memory_lane_count == 2
            and len(matched_control_outcomes) == 1
        ),
        "g5_lane_support_resolutions": list(lane_support_records),
        "g5_matched_control_outcomes": list(matched_control_outcomes),
        "g5_card_vs_neutral_effect_identified": False,
        "g5_required_successor_design": (
            "replicated_same_parent_same_full_pool_active_neutral_slots"
        ),
        "g5_memory_statuses": list(g5_memory_statuses),
        "reflection_eligibility_counts": {
            str(key): list(value) for key, value in eligible_receipt_counts.items()
        },
        "affine_reference_violations": reference_violations,
        "evidence_registry": evidence.snapshot().to_record(),
        "forecast_feedback": {
            "receipt_count": forecast_receipt_count,
            "observation_count": forecast_observation_count,
        },
        "final_front": [
            {
                "candidate_id": value.candidate_id.value,
                "generation": value.generation,
                "objectives": dict(value.objective_map),
            }
            for value in runtime.final_front
        ],
        "source_closure": {
            "launch_sha256": expected_source_sha256,
            "postrun_sha256": source_after["aggregate_sha256"],
        },
    }


async def _main_async(args: argparse.Namespace) -> int:
    run_dir = (ARTIFACT_ROOT / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    preparation: DurableJsonlJournal | None = None
    evaluator_journal: DurableJsonlJournal | None = None
    journals: dict[str, Any] = {}
    try:
        paths = _source_paths()
        source = source_identity(paths, relative_to=WORKSPACE_ROOT)
        snapshot = _snapshot_sources(run_dir, paths)
        if (
            snapshot["aggregate_sha256"] != source["aggregate_sha256"]
            or snapshot["file_count"] != source["file_count"]
        ):
            raise RuntimeError("source changed while creating the launch snapshot")
        write_json_atomic(
            run_dir / "manifest.json",
            _manifest(
                run_id=args.run_id,
                mode=args.mode,
                source=source,
                source_snapshot=snapshot,
            ),
        )
        preparation = DurableJsonlJournal(run_dir / "preparation.jsonl")
        evaluator_journal = DurableJsonlJournal(
            run_dir / "real_evaluator_observations.jsonl"
        )
        arm = "control" if args.mode == "control" else "live"
        bundle = _prepare_bundle(
            run_dir=run_dir,
            preparation_journal=preparation,
            evaluator_journal=evaluator_journal,
            source_closure_sha256=str(source["aggregate_sha256"]),
            arm=arm,
        )
        probe = _all_wave_probe(bundle)
        write_json_atomic(run_dir / "provider_free_construction_probe.json", probe)
        if not probe["all_gates_pass"]:
            raise RuntimeError("BOiLS provider-free construction gate failed")
        _require_source_closure(str(source["aggregate_sha256"]))
        if args.mode == "prepare":
            preregistration = _preregistration_contract(
                bundle=bundle,
                source_aggregate_sha256=str(source["aggregate_sha256"]),
                construction_probe=probe,
            )
            write_json_atomic(
                run_dir / "preregistration_template.json", preregistration
            )
            summary = {
                "schema_version": 1,
                "status": "prepared_provider_credential_and_abc_execution_free",
                "provider_calls": 0,
                "credential_read": False,
                "abc_executions": bundle.evaluator_observer.calls,
                "evaluator_cryptographic_preflight": bundle.evaluator.provenance(),
                "preparation": bundle.prepared.to_record(),
                "experiment_profile": (
                    None
                    if bundle.experiment_profile is None
                    else bundle.experiment_profile.to_record()
                ),
                "experiment_profile_conformance": (
                    None
                    if bundle.experiment_profile is None
                    else bundle.experiment_profile.prepared_conformance_record(
                        prepared=bundle.prepared,
                        archive_utility=bundle.utility,
                        outer_seed=OUTER_SEED,
                    )
                ),
                "construction_probe": probe,
                "source_snapshot": snapshot,
                "preregistration_template": preregistration,
                "live_command_shape": (
                    "python examples/development/run_boils_generic_campaign.py "
                    "live --run-id <id> --prereg <workspace-path>"
                ),
                "control_command_shape": (
                    "python examples/development/run_boils_generic_campaign.py "
                    "control --run-id <id>"
                ),
            }
            write_json_atomic(run_dir / "summary.json", summary)
            preparation.close()
            evaluator_journal.close()
            finalize_run_directory(run_dir, status=str(summary["status"]))
            print(json.dumps(summary, sort_keys=True))
            return 0

        if args.mode == "live":
            if args.prereg is None:
                raise RuntimeError("live mode requires --prereg")
            write_json_atomic(
                run_dir / "preregistration_identity.json",
                _validate_preregistration(
                    path=Path(args.prereg),
                    bundle=bundle,
                    source_aggregate_sha256=str(source["aggregate_sha256"]),
                    construction_probe=probe,
                ),
            )
        elif args.prereg is not None:
            raise RuntimeError("control mode does not accept --prereg")

        journals = _open_execution_journals(run_dir)
        summary = await _execute(
            bundle=bundle,
            run_dir=run_dir,
            journals=journals,
            expected_source_sha256=str(source["aggregate_sha256"]),
        )
        write_json_atomic(run_dir / "summary.json", summary)
        for journal in journals.values():
            journal.close()
        journals = {}
        preparation.close()
        evaluator_journal.close()
        finalize_run_directory(run_dir, status=str(summary["status"]))
        print(json.dumps(summary, sort_keys=True))
        return _execution_exit_code(summary["status"])
    except BaseException as error:
        for journal in journals.values():
            journal.close()
        if preparation is not None:
            preparation.close()
        if evaluator_journal is not None:
            evaluator_journal.close()
        if not (run_dir / "summary.json").exists():
            write_json_atomic(
                run_dir / "summary.json",
                {
                    "schema_version": 1,
                    "status": "failed_before_completion",
                    "failure_type": type(error).__name__,
                    "failure_sha256": _sha(
                        f"{type(error).__qualname__}\x00{str(error)}"
                    ),
                },
            )
        if not (run_dir / "finalized.json").exists():
            finalize_run_directory(run_dir, status="failed")
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "live", "control"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prereg")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_main_async(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
