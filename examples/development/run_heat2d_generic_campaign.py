#!/usr/bin/env python3
"""Prepare or run the calibrated G6 AgentEvolve Heat2D Pareto campaign.

``--prepare`` is credential-, provider-, and PDE-free.  ``--live`` is the only
mode that reads ``OPENROUTER_API_KEY`` and requires a preregistration file.
The evolutionary mechanism itself is the generic ``EvolutionCampaignScheduler``
plus ``AgenticPortfolioCampaignRuntime``.  In the reference treatment, each
model call ranks K8 actions from the complete finite contract and the engine
evaluates k4.  This file injects only Heat2D seeds, evidence, scientific
objective resolution, an affine archive
utility, and provider/runtime composition.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from examples.development.launch_record import (  # noqa: E402
    install_launch_recorder,
    record_campaign_launch,
)

# Observe the launch environment before any module body below reads it.  This
# runner resolves most of its configuration at import time, so the observer
# has to be installed here to see those reads.  Instrumentation only, and
# only for the provider-free ``prepare`` mode; ``live`` is untouched.
install_launch_recorder()

from agent_evolve.settings import load_credentials  # noqa: E402

from agent_evolve.agentic import (  # noqa: E402
    BoundedPortfolioMemoryDoseContract,
    DeterministicIdFactory,
    InsightDraft,
    InsightMemoryBank,
    MetricComparisonAnchorKind,
    MetricEffectDirection,
    PortfolioCard,
    PortfolioMemoryContextProjectionBinding,
    PortfolioMemoryCreditPlan,
    PortfolioMemoryDoseCardSemantics,
    PortfolioMemoryDoseSupportScope,
    PortfolioOptimizationMemoryAssessment,
    PortfolioMemoryMatchedControlWavePlan,
    PortfolioRewardAggregationBinding,
    PortfolioSelectionRequest,
    PortfolioVariationWaveRequest,
    ReflectionEvidenceCatalog,
    ReflectionConsumerScope,
    ReflectionGenerationRequest,
    ReflectionInsightContract,
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
from agent_evolve.application.calibrated_campaign import (  # noqa: E402
    CalibratedCampaignBindingFactory,
    equal_weight_slate_objectives,
)
from agent_evolve.application.campaign_execution import (  # noqa: E402
    CampaignArchiveCutoffReceipt,
    CampaignExecutionEvent,
    CampaignJournalAck,
    CampaignStageRequest,
    EvolutionCampaignScheduler,
)
from agent_evolve.application.campaign_learning_runtime import (  # noqa: E402
    CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY,
    CampaignDiagnosticExposureReceipt,
    CampaignReflectionLearningRecordCodec,
    ClosedLoopCampaignLearningRuntime,
    StructuredCampaignReflectionLearningProjector,
)
from agent_evolve.application.campaign_evidence_registry import (  # noqa: E402
    CampaignEvidenceRegistry,
)
from agent_evolve.application.campaign_diagnostic_blocks import (  # noqa: E402
    CampaignDiagnosticSingletonBlock,
    CampaignDiagnosticSupportCardInput,
    CampaignDiagnosticSupportLaneInput,
)
from agent_evolve.application.campaign_generation_audit import (  # noqa: E402
    TransactionalPortfolioGenerationAuditor,
)
from agent_evolve.application.action_forecast_partitioning import (  # noqa: E402
    ConcurrentActionForecastWave,
)
from agent_evolve.application.outcome_conditioned_portfolio_selection import (  # noqa: E402
    OUTCOME_CONDITIONED_PORTFOLIO_POLICY_DEFINITION_SHA256,
    OutcomeConditionedPortfolioSelectionPolicy,
    outcome_conditioned_selected_predictions,
)
from agent_evolve.application.campaign_contextual_outcomes import (  # noqa: E402
    ContextualOutcomeCampaignEnricher,
)
from agent_evolve.application.campaign_learning import (  # noqa: E402
    ClosedLoopCampaignLearning,
)
from agent_evolve.application.finite_action_hypothesis_semantics import (  # noqa: E402
    PortableFiniteActionHypothesisMatcher,
    PortableFiniteActionInsightSemanticCompiler,
)
from agent_evolve.application.agentic_evolution import (  # noqa: E402
    EvolutionCandidate,
)
from agent_evolve.application.insight_memory import (  # noqa: E402
    InsightLifecycleState,
    compose_epistemic_prompt_payload,
)
from agent_evolve.application.detailed_evaluation import (  # noqa: E402
    DetailedEvaluation,
    DetailedEvaluationPayload,
    EvaluationTimings,
    EvaluatorIdentity,
)
from agent_evolve.application.evolution_campaign import (  # noqa: E402
    CampaignAgentRuntimeReceipt,
    CampaignPolicies,
    CampaignPolicyBinding,
    CampaignReflectionWave,
    CampaignReflectionSupervisionPolicy,
    CampaignWorkloadPorts,
    PreparedEvolutionCampaign,
    ReflectionFailureMode,
    ReflectionLaunchMode,
    ReflectionVisibility,
)
from agent_evolve.application.evaluation_accounting import (  # noqa: E402
    CampaignEvaluationAccounting,
)
from agent_evolve.application.identifiable_reflection_request import (  # noqa: E402
    identifiable_reflection_request_construction_record,
)
from agent_evolve.application.identifiable_reflection_evidence import (  # noqa: E402
    IdentifiableMutationReflectionContrast,
    IdentifiableReflectionEvidenceSnapshot,
)
from agent_evolve.application.portfolio_campaign_runtime import (  # noqa: E402
    AgenticPortfolioCampaignRuntime,
    ArchiveDiverseEliteCampaignParentSelector,
    ResidualHypervolumeCampaignParentSelector,
    StagnationAwareDiverseCampaignParentSelector,
    CampaignIdentifiableReflectionInput,
    CampaignIdentifiableReflectionEvidenceProjection,
    CampaignIdentifiableReflectionEvidenceQuery,
    CampaignPortfolioMemoryEstimandProjection,
    CampaignPortfolioWaveContext,
    CampaignPortfolioWavePreparationReceipt,
    CommittedRegistryIdentifiableReflectionEvidenceSource,
    CAMPAIGN_ARCHIVE_CONTEXT_KEY,
    CAMPAIGN_FRONTIER_TARGET_KEY,
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
from agent_evolve.application.portfolio_hypothesis_observations import (  # noqa: E402
    FinitePortfolioActionSemanticsCompiler,
    ObjectiveDeltaMetricEffectProjector,
)
from agent_evolve.application.portfolio_memory_matched_control import (  # noqa: E402
    PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256,
    PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID,
    PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION,
    PortfolioMemoryMatchedControlPlanner,
    PortfolioMemoryMatchedSupportResolution,
    PortfolioMemoryMatchedSupportResolver,
    materialize_portfolio_memory_matched_arm,
)
from agent_evolve.application.parent_measurement import (  # noqa: E402
    attach_parent_measurement_to_context,
    bind_parent_measurement,
    create_parent_measurement_projection,
)
from agent_evolve.application.outcome_relation import (  # noqa: E402
    objective_pareto_outcome_binding,
)
from agent_evolve.application.portfolio_recombination import (  # noqa: E402
    bind_portfolio_recombination_source_utilities,
)
from agent_evolve.campaign_workload import (  # noqa: E402
    AgenticCampaignEvidenceProjections,
    AgenticCampaignWorkloadConfig,
)
from agent_evolve.campaign_presets import (  # noqa: E402
    DelayedPortfolioCampaignPreset,
    PortfolioCampaignBehavior,
)
from agent_evolve.domain.typed_json import (  # noqa: E402
    FrozenJsonObject,
    thaw_json,
)
from agent_evolve.domain.ids import (  # noqa: E402
    CandidateId,
    LLMCallId,
    OperatorInvocationId,
)
from agent_evolve.domain.insight import InsightRef  # noqa: E402
from agent_evolve.domain.lineage import CandidateOccurrence  # noqa: E402
from agent_evolve.domain.llm_task_queue import (  # noqa: E402
    PartitionedRetryBudget,
    ValidationIssueReasonCode,
)
from agent_evolve.core.optimization_semantics import (  # noqa: E402
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
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
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (  # noqa: E402
    validate_openrouter_outbound_request_manifest_record,
)
from agent_evolve.integrations.pydantic_ai.model_execution_profile import (  # noqa: E402
    OpenRouterModelExecutionProfile,
    openrouter_model_execution_profile,
)
from agent_evolve.campaign_profiles import CampaignExperimentProfile  # noqa: E402
from agent_evolve.campaign_variation_topology import (  # noqa: E402
    CampaignVariationTopology,
    CampaignVariationTopologyMode,
)
from agent_evolve.reference_method import (  # noqa: E402
    ReferenceCampaignImplementations,
    reference_atomic_variation_topology_binding,
    reference_campaign_experiment_profile,
    reference_contextual_outcomes_binding,
    reference_hierarchical_r2_variation_topology_binding,
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
from agent_evolve.integrations.pydantic_ai.sealed_output_replay import (  # noqa: E402
    SealedAcceptedOutputReplaySource,
    SealedReplayJsonlFile,
    SealedReplayThenLiveStructuredRunner,
    load_sealed_accepted_output_replay_jsonl,
)
from agent_evolve.policies.memory import (  # noqa: E402
    BalancedSubsetBlockPlan,
    BalancedSubsetBlockPlanner,
    CausalSearchScorePolicy,
    MemoryAssignmentArm,
    ResolvedInsightAssignment,
    StableMemoryAssignmentUnit,
)
from agent_evolve.policies.memory.compatibility_matching import (  # noqa: E402
    LaneCardMatchingCard,
    LaneCardMatchingLane,
)
from agent_evolve.policies.memory.global_falsification import (  # noqa: E402
    HypothesisAuditScope,
    ObservedMetricEffect,
)
from agent_evolve.policies.objective_resolution.fixed_grid import (  # noqa: E402
    FixedGridMetricSpec,
    FixedGridObjectiveResolution,
    FixedGridRoundingLaw,
)
from agent_evolve.policies.reward.affine_hypervolume import (  # noqa: E402
    AffineFrozenArchiveJointWaveReward,
    AffineHypervolume2DSpec,
    AffineHypervolumeArchiveUtility,
    AffineHypervolumeSnapshot2D,
    AffineObjectiveAxis,
)
from agent_evolve.policies.reward.contextual_marginal_utility import (  # noqa: E402
    FixedReferenceContextualMarginalUtilityProjector,
)
from agent_evolve.policies.selection.forecast_calibration import (  # noqa: E402
    ForecastCalibrationScope,
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
)
from agent_evolve.policies.selection.residual_frontier_target import (  # noqa: E402
    ResidualHypervolumeFrontierTargetAllocator,
)
from agent_evolve.policies.selection.target_conditioned_allocator import (  # noqa: E402
    TargetConditionedSlateAllocatorAdapter,
)
from agent_evolve.policies.selection.finite_option_prompt_projection import (  # noqa: E402
    FiniteOptionPromptProjectionPolicy,
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
from agent_evolve.ports.action_forecast import (  # noqa: E402
    ActionForecastPartitionPolicyBinding,
)
from agent_evolve.policies.selection.meaningful_direction import (  # noqa: E402
    AbsoluteToleranceDirectionAdjudicator,
    MetricDirectionResolution,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry  # noqa: E402
from agent_evolve.ports.decision_metric_projection import (  # noqa: E402
    DecisionMetricProjection,
)
from agent_evolve.ports.objective_resolution import (  # noqa: E402
    ObjectiveResolutionRequest,
    objective_resolution_policy_metadata,
    resolve_objectives,
)
from agent_evolve.ports.parent_measurement import (  # noqa: E402
    ParentMeasurementProjection,
)
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredStreamCleanupPolicy,
    StructuredStreamLivenessPolicy,
    StructuredStreamProgress,
)
from examples.benchmarks.heat2d_constructive.artifact_boundary import (  # noqa: E402
    artifact_scripts_dir,
)
from examples.benchmarks.heat2d_constructive.campaign_workload import (  # noqa: E402
    compose_heat2d_pareto_campaign_workload,
)
from examples.development.heat2d_campaign_reflection import (  # noqa: E402
    build_heat2d_identifiable_reflection_learning_envelope,
    build_heat2d_identifiable_reflection_request,
)
from examples.benchmarks.heat2d_constructive.finite_variation_catalog import (  # noqa: E402
    CATALOG_ID,
    LOCUS_GRIDS,
)
from examples.benchmarks.heat2d_constructive.multiobjective_v1 import (  # noqa: E402
    FORMULATION_DEFINITION_SHA256,
    MATERIAL_OBJECTIVE_NAME,
    THERMAL_OBJECTIVE_NAME,
    WORKLOAD_ID,
    Heat2DMultiObjectiveV1Problem,
    create_multiobjective_benchmark,
)
from examples.benchmarks.heat2d_constructive.problem_def import (  # noqa: E402
    DirectV3Evaluator,
    Heat2DDirectV3Settings,
)
from examples.benchmarks.heat2d_constructive.action_metric_projection import (  # noqa: E402
    Heat2DExactMaterialProjector,
)
from examples.benchmarks.heat2d_constructive.action_semantics import (  # noqa: E402
    heat2d_action_space_semantics,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    BatchedDurableJsonlJournal,
    DurableJsonlJournal,
    finalize_run_directory,
    read_jsonl,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "benchmark_q1/engibench_heat2d/generic_campaign"
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


def _optional_replay_source_path() -> Path | None:
    raw = os.environ.get("AGENT_EVOLVE_SEALED_REPLAY_SOURCE")
    if raw is None:
        return None
    if not raw.strip():
        raise ValueError("AGENT_EVOLVE_SEALED_REPLAY_SOURCE cannot be empty")
    path = Path(raw).expanduser().resolve(strict=True)
    if WORKSPACE_ROOT not in path.parents or not path.is_dir():
        raise ValueError("sealed replay source must be a workspace directory")
    return path


SEALED_REPLAY_SOURCE_PATH = _optional_replay_source_path()
SEMANTIC_READINESS_PROCESS_CPU_LIMIT_S = 300.0
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
    and ACQUISITION_MODE is not CampaignAcquisitionMode.HORIZON_BOUNDED
):
    raise ValueError("constraint-decoupled acquisition requires horizon_bounded mode")
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


OUTER_SEED = _replicate_seed_from_environment(20_260_716)
AGENTIC_ID_NAMESPACE = "heat_g6_ident_v5"
TASK_SHA256 = hashlib.sha256(
    b"agent-evolve:heat2d-pareto-v1-delayed-identifiable-campaign-task:v5"
).hexdigest()
PROTOCOL_ID = (
    {
        CampaignAcquisitionMode.MODEL_TOP_K: (
            "heat2d_generic_common_pool_model_anchored_g6_v1"
        ),
        CampaignAcquisitionMode.CALIBRATED_FRONTIER: (
            "heat2d_generic_common_pool_calibrated_frontier_g6_v2"
        ),
        CampaignAcquisitionMode.HIERARCHICAL_SUPPORT: (
            "heat2d_generic_hierarchical_support_g6_v3"
        ),
        CampaignAcquisitionMode.OPERATOR_STRATIFIED: (
            "heat2d_generic_operator_stratified_g6_v4"
        ),
        CampaignAcquisitionMode.HORIZON_BOUNDED: (
            "heat2d_generic_horizon_bounded_g6_v1"
        ),
        CampaignAcquisitionMode.TARGET_CONDITIONED: (
            "heat2d_generic_target_conditioned_g6_v1"
        ),
    }[ACQUISITION_MODE]
    if COMMON_POOL_ACQUISITION
    else "heat2d_generic_calibrated_g6_delayed_identifiable_v5"
)
GENERATION_COUNT = 6
PORTFOLIO_GENERATIONS = (1, 3, 5)
RECOMBINATION_GENERATIONS = (2, 4, 6)
REFLECTION_SOURCE_GENERATIONS = (2,)
REFLECTION_ADMISSION_GENERATIONS = (4,)
FIRST_REFLECTION_CONSUMER_GENERATION = 5
MAX_CACHE_REUSE_OCCURRENCES = 6
PLANNED_LOGICAL_LLM_CALLS = 7
CALIBRATED_PROPOSAL_WIDTH = 8
PORTFOLIO_WIDTH = 4 if COMMON_POOL_ACQUISITION else 8
COMMON_CANDIDATE_POOL_SIZE = _candidate_pool_size_from_environment(8)
PARENTS_PER_PORTFOLIO = 2
RECOMBINATIONS_PER_PARENT = 2
PLANNED_UNIQUE_EVALUATIONS = 2 + PARENTS_PER_PORTFOLIO * (
    len(PORTFOLIO_GENERATIONS) * PORTFOLIO_WIDTH
    + len(RECOMBINATION_GENERATIONS) * RECOMBINATIONS_PER_PARENT
)
EVALUATOR_CONCURRENCY = 1
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


def _contextual_joint_capability_projector(
) -> FiniteContractContextualJointCapabilityProjector:
    """Expose Heat2D selection structure through the generic planner port."""

    return FiniteContractContextualJointCapabilityProjector(
        min_distinct_families=3,
        require_pairwise_disjoint_parent_patches=True,
        require_declared_source_floor_options=True,
    )


QUALIFIED_ALL_VOID_THERMAL_TERM = 0.004492585018256053
QUALIFIED_ALL_VOID_MANIFEST_SHA256 = (
    "c14dfa3e45892d52ace28ab859307b708e4fdb3a1f5752f4a8912b81c00ec55d"
)
THERMAL_AFFINE_REFERENCE = 0.005
MATERIAL_AFFINE_REFERENCE = 0.61
MEMORY_AGGREGATION_ID = "affine_archive_joint_wave_gain"
MEMORY_AGGREGATION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:affine-archive-joint-wave-gain:v1"
).hexdigest()
OBJECTIVE_IDS = tuple(sorted((THERMAL_OBJECTIVE_NAME, MATERIAL_OBJECTIVE_NAME)))
OPTION_FAMILIES = (
    "additive_geometry",
    "material_fraction",
    "subtractive_geometry",
)
REFLECTION_DECISION_PATHS = tuple(
    sorted(f"$.{locus}" for locus, _values, _family in LOCUS_GRIDS)
)
REFLECTION_CONSUMER_SCOPES = (ReflectionConsumerScope.MUTATION_SELECTION,)
REFLECTION_INSIGHT_KINDS = (
    ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
    ReflectionInsightKind.MECHANISTIC_CONJECTURE,
)
REFLECTION_COMPARISON_ANCHORS = (MetricComparisonAnchorKind.CURRENT_PARENT,)


def _reflection_contract(
    allowed_option_families: tuple[str, ...],
) -> ReflectionInsightContract:
    """Bind model reflection to workload-owned semantics, not prose alone."""

    families = tuple(sorted(set(allowed_option_families)))
    return ReflectionInsightContract(
        required_metric_ids=OBJECTIVE_IDS,
        allowed_option_families=families,
        allowed_decision_paths=REFLECTION_DECISION_PATHS,
        allowed_insight_kinds=REFLECTION_INSIGHT_KINDS,
        allowed_consumer_scopes=REFLECTION_CONSUMER_SCOPES,
        allowed_comparison_anchor_kinds=REFLECTION_COMPARISON_ANCHORS,
        allowed_factor_capabilities=families,
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _canonical_json_size(value: object) -> int:
    return len(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii", errors="strict")
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    if type(result) is not FrozenJsonObject:
        raise TypeError("expected a frozen typed-JSON object")
    return result


def _telemetry_record(value: AgenticCallTelemetry) -> dict[str, object]:
    AgenticCallTelemetry.__post_init__(value)
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
    StructuredStreamProgress.__post_init__(value)
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


def _candidate_history_record(candidate: Any) -> dict[str, object]:
    detailed = candidate.detailed_evaluation
    return {
        "candidate_id": candidate.candidate_id.value,
        "configuration_sha256": candidate.occurrence.configuration_hash,
        "generation": candidate.generation,
        "label": candidate.label,
        "operator_kind": (
            None if candidate.operator_kind is None else candidate.operator_kind.value
        ),
        "parent_ids": [value.value for value in candidate.parent_ids],
        "common_ancestor_id": (
            None
            if candidate.common_ancestor_id is None
            else candidate.common_ancestor_id.value
        ),
        "objectives": {
            name: {"value": value, "value_hex": value.hex()}
            for name, value in candidate.objectives
        },
        "valid": candidate.valid,
        "operator_compliant": candidate.operator_compliant,
        "evidence_compliant": candidate.evidence_compliant,
        "preservation_verified": candidate.preservation_verified,
        "detailed_evaluation_sha256": (
            None if detailed is None else detailed.evidence_sha256
        ),
    }


def _archive_front_points(
    archive: dict[str, object],
) -> tuple[dict[str, float], ...]:
    front = archive.get("front_candidates")
    if type(front) is not list:
        raise RuntimeError("stage archive omitted its front candidates")
    points = []
    for candidate in front:
        if type(candidate) is not dict or type(candidate.get("objectives")) is not list:
            raise RuntimeError("stage archive candidate omitted objectives")
        point = {}
        for value in candidate["objectives"]:
            if type(value) is not dict:
                raise RuntimeError("stage archive objective is not an object")
            point[str(value["metric_id"])] = float.fromhex(str(value["value_hex"]))
        points.append(point)
    return tuple(points)


def _archive_trajectory_record(
    *,
    label: str,
    generation: int,
    front_candidates: list[dict[str, object]],
    spec: AffineHypervolume2DSpec,
) -> dict[str, object]:
    points = []
    for candidate in front_candidates:
        objectives = candidate["objectives"]
        if type(objectives) is dict:
            point = {name: float(value["value"]) for name, value in objectives.items()}
        else:
            point = {
                str(value["metric_id"]): float.fromhex(str(value["value_hex"]))
                for value in objectives
            }
        points.append(point)
    snapshot = AffineHypervolumeSnapshot2D.create(
        spec=spec,
        archive_points=tuple(points),
    )
    return {
        "label": label,
        "generation": generation,
        "normalized_hypervolume_hex": snapshot.base_hypervolume.hex(),
        "raw_oriented_hypervolume_hex": (snapshot.raw_oriented_base_hypervolume.hex()),
        "snapshot_sha256": snapshot.snapshot_sha256,
        "front_candidates": front_candidates,
    }


def _pde_evidence_record(
    run_dir: Path,
    *,
    expected_physical_evaluations: int,
) -> dict[str, object]:
    if (
        type(expected_physical_evaluations) is not int
        or expected_physical_evaluations < 0
    ):
        raise ValueError("expected physical evaluations must be non-negative")
    manifests = sorted((run_dir / "pde/evaluations").glob("direct-v3-*/manifest.json"))
    rows = []
    for path in manifests:
        value = json.loads(path.read_text(encoding="utf-8"))
        container = value.get("container_result", {})
        measurement = container.get("resource_measurement", {})
        result = container.get("result", {})
        exact_volume = container.get("exact_volume_contract", {})
        candidate = value.get("candidate", {})
        checks = value.get("checks", {})
        volume = value.get("volume_agreement", {})
        elapsed = float(value.get("elapsed_s", math.inf))
        peak_rss = measurement.get("peak_rss_bytes_by_linux_kib_convention")
        scientific_pass = (
            value.get("schema_version") == 3
            and value.get("evaluator_id") == "engibench-heatconduction2d-direct-v3"
            and value.get("all_checks_pass") is True
            and value.get("full_pde_solve_count") == 1
            and checks.get("exact_cross_runtime_fe_volume_identity_matches") is True
            and volume.get("exact_identity_matches") is True
        )
        resource_pass = (
            math.isfinite(elapsed)
            and elapsed < 45.0
            and type(peak_rss) is int
            and peak_rss < 3 * 1024**3
        )
        exact_material_fraction = None
        numerator = exact_volume.get("exact_scaled_numerator_decimal")
        mesh_denominator = exact_volume.get("mesh_mass_denominator")
        exponent = exact_volume.get("binary64_common_denominator_exponent")
        if (
            type(numerator) is str
            and numerator.isdecimal()
            and type(mesh_denominator) is int
            and mesh_denominator > 0
            and type(exponent) is int
            and exponent >= 0
        ):
            exact_material_fraction = float(
                Fraction(int(numerator), mesh_denominator * (1 << exponent))
            )
        rows.append(
            {
                "relative_manifest": path.relative_to(run_dir).as_posix(),
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "raw_array_sha256": candidate.get("raw_array_sha256"),
                "exact_volume_contract_sha256": exact_volume.get("contract_sha256"),
                "elapsed_s": elapsed,
                "peak_rss_bytes": peak_rss,
                "objectives": {
                    MATERIAL_OBJECTIVE_NAME: exact_material_fraction,
                    THERMAL_OBJECTIVE_NAME: result.get(THERMAL_OBJECTIVE_NAME),
                },
                "scientific_contract_pass": scientific_pass,
                "resource_gate_pass": resource_pass,
            }
        )
    return {
        "manifest_count": len(rows),
        "manifest_count_matches_physical_evaluations": (
            len(rows) == expected_physical_evaluations
        ),
        "all_scientific_contracts_pass": all(
            value["scientific_contract_pass"] for value in rows
        ),
        "all_under_45_s_and_3_gib": all(value["resource_gate_pass"] for value in rows),
        "rows": rows,
    }


def _affine_spec() -> AffineHypervolume2DSpec:
    return AffineHypervolume2DSpec(
        axes=(
            AffineObjectiveAxis(
                THERMAL_OBJECTIVE_NAME,
                "min",
                0.0,
                THERMAL_AFFINE_REFERENCE,
            ),
            AffineObjectiveAxis(
                MATERIAL_OBJECTIVE_NAME,
                "min",
                0.30,
                MATERIAL_AFFINE_REFERENCE,
            ),
        ),
        reference_provenance=(
            "prospective support-bound reference: thermal=0.005 exceeds the "
            "qualified all-void maximum-compliance observation "
            f"{QUALIFIED_ALL_VOID_THERMAL_TERM.hex()} from manifest "
            f"{QUALIFIED_ALL_VOID_MANIFEST_SHA256}; material=0.61 exceeds the "
            "hard 0.60 material bound by 0.01"
        ),
    )


def _option_prompt_projection() -> FiniteOptionPromptProjectionPolicy:
    """Expose only the Heat catalog metadata needed to understand an action."""

    return FiniteOptionPromptProjectionPolicy(
        metadata_keys=VARIATION_TOPOLOGY.prompt_metadata_keys(("locus", "target_value"))
    )


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


def _reference_parent_selector(experiment_profile: object | None):
    """Resolve and verify the parent selector before any external work.

    Keep this version-sensitive conformance check on the provider/PDE-free
    preparation path as well as the live path.  Otherwise a newly promoted
    reference method can prepare successfully and fail only after credentials
    have been read, which censors an assay for an integration-only reason.
    """

    parent_selector = (
        ArchiveDiverseEliteCampaignParentSelector()
        if experiment_profile is None
        else experiment_profile.parent_selection.implementation
    )
    allowed_types = (
        (ArchiveDiverseEliteCampaignParentSelector,)
        if experiment_profile is None
        else (
            ArchiveDiverseEliteCampaignParentSelector,
            StagnationAwareDiverseCampaignParentSelector,
            ResidualHypervolumeCampaignParentSelector,
        )
    )
    if type(parent_selector) not in allowed_types:
        raise TypeError("reference profile parent-selection implementation has drifted")
    return parent_selector


def _default_allocator() -> CalibratedPortfolioAllocator:
    """Use the preregistered evaluator-allocation policy for the active arm."""

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
        _default_allocator(),
        constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
        minimum_intervention_projection=MINIMUM_INTERVENTION_PROJECTION,
        evidence_calibrated_source_mix=EVIDENCE_CALIBRATED_SOURCE_MIX,
        contextual_search_allocation=CONTEXTUAL_SEARCH_ALLOCATION,
    )


def _calibrated_wave_prompt_composition_exact(
    record: dict[str, object],
) -> bool:
    """Authenticate prompt identity from the request's actual dose contract.

    A reflected card can be visible without a hard memory-dose contract.  In
    particular, the task-keyed common-pool assay disables model-authored dose
    attribution while retaining advisory reflected cards.  Generation number
    therefore cannot determine the prompt definition.
    """

    return (
        record["allocator"] == _default_allocator().to_record()
        and record["prompt_definition_sha256"]
        == calibrated_portfolio_prompt_definition_sha256(
            _option_prompt_projection(),
            bounded_memory_dose=(record["bounded_reflection_memory_dose"] is not None),
            proposal_support=_proposal_support_policy() is not None,
            hierarchical_composition_required_proposals=(
                VARIATION_TOPOLOGY.hierarchical_composition_required_proposals
            ),
            feasibility_witness_mode=FEASIBILITY_WITNESS_MODE,
            constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
        )
        and record["selector_policy_definition_sha256"]
        == _selector_policy_definition_sha256()
        and record["option_prompt_projection_sha256"] is not None
    )


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


def _objective_resolution() -> FixedGridObjectiveResolution:
    """Bind Heat decisions to the repeat-qualified 1e-12 absolute grid.

    Direct-v3 retains every raw physical value in its detailed evidence.  The
    optimization layer consumes the coarser grid, removing sub-resolution
    Pareto and reward changes without altering the evaluator or its cache key.
    """

    return FixedGridObjectiveResolution(
        metric_specs=tuple(
            FixedGridMetricSpec(
                metric_id=metric_id,
                decimal_origin=Decimal("0"),
                decimal_quantum=Decimal("0.000000000001"),
                rounding_law=FixedGridRoundingLaw.NEAREST_TIES_TO_EVEN,
            )
            for metric_id in OBJECTIVE_IDS
        )
    )


def _heat_optimization_semantics(
    relation_identity: tuple[str, int, str],
) -> OptimizationSemantics:
    metrics = (
        MetricSemantics(
            metric_id=f"objective:{THERMAL_OBJECTIVE_NAME}",
            name=THERMAL_OBJECTIVE_NAME,
            role=MetricRole.OBJECTIVE,
            sense=MetricSense.MINIMIZE,
            definition="Direct-v3 physical thermal compliance term.",
            aggregation="One serialized 1001-resolution PDE solve.",
            witness_interpretation="Lower thermal compliance is better.",
            tolerance=0.0,
        ),
        MetricSemantics(
            metric_id=f"objective:{MATERIAL_OBJECTIVE_NAME}",
            name=MATERIAL_OBJECTIVE_NAME,
            role=MetricRole.OBJECTIVE,
            sense=MetricSense.MINIMIZE,
            definition="Exact projected CG1 material fraction.",
            aggregation="Exact volume numerator divided by its bound denominator.",
            witness_interpretation="Lower material use is better.",
            tolerance=0.0,
        ),
    )
    return OptimizationSemantics(
        semantics_id="heat2d_constructive_pareto",
        semantics_version=1,
        metrics=metrics,
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.PARETO,
            metric_priority=tuple(value.metric_id for value in metrics),
            description="Minimize thermal compliance and exact material use.",
            equivalence="Both fixed-grid decision objectives agree exactly.",
            policy_id=relation_identity[0],
            policy_version=relation_identity[1],
            definition_sha256=relation_identity[2],
        ),
    )


def _heat_evaluator_identity(
    problem: Heat2DMultiObjectiveV1Problem,
) -> EvaluatorIdentity:
    settings = problem.settings
    context = {
        "schema_version": 1,
        "workload_id": WORKLOAD_ID,
        "formulation_definition_sha256": FORMULATION_DEFINITION_SHA256,
        "underlying_evaluator": "engibench-heatconduction2d-direct-v3",
        "resolution": settings.resolution,
        "required_numpy_version": settings.required_numpy_version,
        "external_concurrency": settings.external_concurrency,
    }
    return EvaluatorIdentity(
        evaluator_id="heat2d_constructive_pareto_direct_v3",
        evaluator_version=1,
        evaluator_context_sha256=_sha(
            json.dumps(
                context,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class _HeatDetailedEvaluationAdapter:
    """Preserve direct-v3 evaluator identity and raw measurements in candidates."""

    problem: Heat2DMultiObjectiveV1Problem
    evaluator_identity: EvaluatorIdentity

    def __post_init__(self) -> None:
        if type(self.problem) is not Heat2DMultiObjectiveV1Problem:
            raise TypeError("problem must be the exact Heat2D Pareto problem")
        if type(self.evaluator_identity) is not EvaluatorIdentity:
            raise TypeError("evaluator_identity must be exact")
        if self.evaluator_identity != _heat_evaluator_identity(self.problem):
            raise ValueError("Heat evaluator identity differs from its problem")

    def evaluate_evidence(
        self,
        configuration: dict[str, object],
    ) -> DetailedEvaluationPayload:
        self.__post_init__()
        result = self.problem.evaluate_detailed(configuration)
        return DetailedEvaluationPayload(
            failure=None,
            objectives=tuple(
                (objective.name, float(result.objective_values[objective.name]))
                for objective in self.problem.objectives
            ),
            violations=(),
            checks=(),
            receipt=None,
            evaluator=self.evaluator_identity,
        )


def _scientific_benchmark(settings: Heat2DDirectV3Settings):
    base = create_multiobjective_benchmark(settings)
    problem = base.problem
    if type(problem) is not Heat2DMultiObjectiveV1Problem:
        raise TypeError("Heat benchmark published a foreign problem")
    relation = objective_pareto_outcome_binding(problem.objectives)
    evaluator = _HeatDetailedEvaluationAdapter(
        problem=problem,
        evaluator_identity=_heat_evaluator_identity(problem),
    )
    if len(base.finite_variation_catalogs) != 1:
        raise RuntimeError("Heat benchmark must expose one atomic catalog")
    selected_catalog = VARIATION_TOPOLOGY.decorate(base.finite_variation_catalogs[0])
    return replace(
        base,
        finite_variation_catalogs=(selected_catalog,),
        detailed_evaluator=evaluator,
        outcome_relation=relation,
        optimization_semantics=_heat_optimization_semantics(relation.identity),
        objective_resolution=_objective_resolution(),
    )


def _memory_estimand_context() -> FrozenJsonObject:
    value = freeze_json(
        {
            "schema_version": 1,
            "estimand_id": "heat2d_g6_balanced_memory_assignment_effect",
            "workload": {
                "workload_id": WORKLOAD_ID,
                "formulation_definition_sha256": FORMULATION_DEFINITION_SHA256,
                "representation": "fixed_four_primitive_csg_dense_cg1_projection",
                "evaluator_panel": {
                    "evaluator_id": "engibench-heatconduction2d-direct-v3",
                    "resolution": 1001,
                    "external_concurrency": 1,
                },
                "objectives": [
                    {"metric_id": metric_id, "direction": "minimize"}
                    for metric_id in OBJECTIVE_IDS
                ],
            },
            "treatment": {
                "operator_kind": "typed_mutation",
                "portfolio_width": PORTFOLIO_WIDTH,
                "assigned_insights_per_wave": 2,
            },
            "endpoint": {
                "archive_utility_id": AffineHypervolumeArchiveUtility(
                    _affine_spec()
                ).utility_id,
                "archive_utility_definition_sha256": (_affine_spec().definition_sha256),
                "reward_aggregation_id": MEMORY_AGGREGATION_ID,
                "reward_aggregation_definition_sha256": (
                    MEMORY_AGGREGATION_DEFINITION_SHA256
                ),
            },
            "assignment_design": {
                "policy": "balanced_uniform_k_subset_blocks",
                "units": [
                    {"generation": generation, "lane_id": lane}
                    for lane in ("elite", "explorer")
                    for generation in PORTFOLIO_GENERATIONS
                ],
            },
            "adaptive_score_consumption": False,
            "causal_claim_allowed": False,
        }
    )
    if type(value) is not FrozenJsonObject:
        raise TypeError("memory estimand context must be a frozen typed-JSON object")
    return value


MEMORY_ESTIMAND_CONTEXT = _memory_estimand_context()
MEMORY_CONTEXT_SHA256 = typed_json_sha256(MEMORY_ESTIMAND_CONTEXT)


def _source_paths() -> tuple[Path, ...]:
    core = tuple(sorted((AGENT_EVOLVE_ROOT / "src/agent_evolve").rglob("*.py")))
    heat = tuple(
        sorted(
            (AGENT_EVOLVE_ROOT / "examples/benchmarks/heat2d_constructive").glob("*.py")
        )
    )
    scripts = artifact_scripts_dir()
    numeric = tuple(
        scripts / name
        for name in (
            "calibration_harness.py",
            "heat2d_constructive_layout.py",
            "heat2d_integrity_adapter.py",
            "heat2d_exact_volume_contract.py",
            "run_heat2d_direct_v1.py",
            "heat2d_direct_v1_container.py",
            "run_heat2d_direct_v3.py",
            "heat2d_direct_v3_container.py",
        )
    )
    dependency_locks = tuple(
        path
        for path in (
            AGENT_EVOLVE_ROOT / "pyproject.toml",
            AGENT_EVOLVE_ROOT / "uv.lock",
        )
        if path.is_file()
    )
    return (
        Path(__file__),
        AGENT_EVOLVE_ROOT / "examples/development/heat2d_campaign_reflection.py",
        AGENT_EVOLVE_ROOT / "examples/development/durable_run_artifacts.py",
        AGENT_EVOLVE_ROOT / "examples/development/launch_record.py",
        *dependency_locks,
        *core,
        *heat,
        *numeric,
    )


def _load_sealed_replay_source(
) -> tuple[SealedAcceptedOutputReplaySource, dict[str, object]] | None:
    """Verify an optional accepted-output prefix from a finalized prior run."""

    root = SEALED_REPLAY_SOURCE_PATH
    if root is None:
        return None
    finalization = verify_finalized_run_directory(root)
    files = finalization.get("files")
    if type(files) is not dict:
        raise RuntimeError("sealed replay finalization has no file map")

    def sealed_file(name: str) -> SealedReplayJsonlFile:
        record = files.get(name)
        if type(record) is not dict or type(record.get("sha256")) is not str:
            raise RuntimeError(f"sealed replay source omits authenticated {name}")
        return SealedReplayJsonlFile(root / name, str(record["sha256"]))

    source = load_sealed_accepted_output_replay_jsonl(
        source_id=root.name,
        request_evidence=sealed_file("request_evidence.jsonl"),
        output_evidence=sealed_file("output_evidence.jsonl"),
        terminal_outcomes=sealed_file("queue_outcomes.jsonl"),
    )
    finalization_sha256 = finalization.get("finalization_sha256")
    if type(finalization_sha256) is not str:
        raise RuntimeError("sealed replay source lacks finalization identity")
    return source, {
        "schema_version": 1,
        "relative_path": root.relative_to(WORKSPACE_ROOT).as_posix(),
        "finalization_sha256": finalization_sha256,
        "source_identity_sha256": source.source_identity_sha256,
        "accepted_output_count": source.accepted_output_count,
        "requested_models": list(source.requested_models),
        "fail_closed_before_prefix_exhaustion": True,
        "live_provider_after_prefix_exhaustion": True,
    }


def _snapshot_sources(
    run_dir: Path,
    paths: tuple[Path, ...],
) -> dict[str, object]:
    """Copy the exact mutable source closure into the finalized run."""

    snapshot_root = run_dir / "source_snapshot"
    snapshot_root.mkdir(exist_ok=False)
    records: list[dict[str, object]] = []
    labels: set[str] = set()
    aggregate = hashlib.sha256(b"agent-evolve:source-set:v1\x00")
    for path in paths:
        resolved = path.expanduser().resolve(strict=True)
        label = resolved.relative_to(WORKSPACE_ROOT).as_posix()
        if label in labels:
            raise ValueError("source snapshot paths must be unique")
        labels.add(label)
        content = resolved.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        destination = snapshot_root / label
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"source snapshot copy failed verification: {label}")
        records.append({"path": label, "size_bytes": len(content), "sha256": digest})
        label_bytes = label.encode("utf-8", errors="strict")
        aggregate.update(len(label_bytes).to_bytes(8, "big"))
        aggregate.update(label_bytes)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
    return {
        "schema_version": 1,
        "snapshot_directory": "source_snapshot",
        "file_count": len(records),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": records,
    }


def _seed_memory(memory: InsightMemoryBank) -> None:
    memory.extend(
        (
            InsightDraft(
                claim=(
                    "A useful Pareto portfolio should cover both objective axes and "
                    "include candidates aimed at distinct frontier extremes."
                ),
                trigger="A finite multiobjective action palette is available.",
                mechanism=(
                    "Axis-diverse interventions preserve information about the local "
                    "thermal/material trade-off rather than collapsing to one scalar."
                ),
                affected_paths=("$.material_fraction",),
                evidence_summary="Generic multiobjective search prior to be tested.",
                confidence=0.55,
            ),
            InsightDraft(
                claim=(
                    "Prefer semantically novel changes on distinct loci and families "
                    "so successful branches remain eligible for exact recombination."
                ),
                trigger="Several one-locus finite changes are sealed for one parent.",
                mechanism=(
                    "Disjoint ancestor-relative patches can be unioned without repair, "
                    "while semantic deduplication avoids spending evaluations on aliases."
                ),
                affected_paths=("$.left_lobe", "$.right_bar"),
                evidence_summary="Generic variation/recombination diversity prior.",
                confidence=0.65,
            ),
            InsightDraft(
                claim=(
                    "Reserve one portfolio member to falsify the dominant local story "
                    "instead of allocating every slot to similar high-ranked actions."
                ),
                trigger="Predictions are uncertain and only one generation is observed.",
                mechanism=(
                    "A counter-hypothesis member reveals ranking or mechanism error and "
                    "reduces premature convergence."
                ),
                affected_paths=("$.central_hole",),
                evidence_summary="Generic uncertainty-calibration prior.",
                confidence=0.50,
            ),
        ),
        initial_score=0.0,
        applicable_operator_kinds=("typed_mutation",),
    )


def _memory_plan(memory: InsightMemoryBank) -> BalancedSubsetBlockPlan:
    snapshot = CausalSearchScorePolicy().genesis(
        exact_context_hash=MEMORY_CONTEXT_SHA256,
        estimand_stratum_hash=MEMORY_CONTEXT_SHA256,
        priors={entry.reference: entry.initial_score for entry in memory.entries},
    )
    units = tuple(
        StableMemoryAssignmentUnit(
            unit_key=f"{lane}.g{generation:02d}",
            generation=generation,
            lane_id=lane,
        )
        for lane in ("elite", "explorer")
        for generation in PORTFOLIO_GENERATIONS
    )
    catalog_size = math.comb(len(snapshot.entries), 2)
    full_block_count, remainder_size = divmod(len(units), catalog_size)
    full_ranks = tuple(
        int.from_bytes(
            hashlib.sha256(
                f"{PROTOCOL_ID}:balanced-block:{index}".encode("ascii")
            ).digest(),
            "big",
        )
        % math.factorial(catalog_size)
        for index in range(full_block_count)
    )
    if remainder_size:
        raise RuntimeError("G6 Heat memory units must form complete subset blocks")
    return BalancedSubsetBlockPlanner().plan(
        snapshot=snapshot,
        ordered_units=units,
        subset_size=2,
        full_block_permutation_ranks=full_ranks,
        remainder_selection_rank=None,
        remainder_permutation_rank=None,
    )


@dataclass(slots=True)
class _Evidence:
    memory: InsightMemoryBank
    plan: BalancedSubsetBlockPlan

    def initialize_memory(self, benchmark, session, seeds):
        del benchmark, session
        return _object(
            {
                "memory_stratum_sha256": MEMORY_CONTEXT_SHA256,
                "assignment_plan_sha256": self.plan.receipt_sha256,
                "seed_ids": [value.seed_id for value in seeds.seeds],
                "trial_count": 0,
                "adaptive_score_consumption": False,
                "g6_causal_claim_allowed": False,
                "portfolio_outcome_feedback": {
                    "schema_version": 1,
                    "observation_count": 0,
                    "prompt_history": {
                        "schema_version": 1,
                        "actions": [],
                        "treatment_visibility": "action_outcomes_only",
                    },
                    "provider_calls": 0,
                },
            }
        )

    def context(self, benchmark, session, parent, variation, memory):
        del benchmark, session
        receipt = variation.eligibility_receipt
        assert receipt is not None
        frozen_parent = freeze_json(parent)
        if type(frozen_parent) is not FrozenJsonObject:
            raise TypeError("Heat parent context must be an object")
        memory_record = thaw_json(memory)
        feedback = memory_record.get("portfolio_outcome_feedback", {})
        if type(feedback) is not dict:
            raise TypeError("portfolio outcome feedback memory must be an object")
        prompt_history = feedback.get("prompt_history", {})
        if type(prompt_history) is not dict:
            raise TypeError("portfolio outcome prompt history must be an object")
        actions = prompt_history.get("actions", [])
        if type(actions) is not list:
            raise TypeError("portfolio outcome prompt actions must be a list")
        return _object(
            {
                "schema_version": 2,
                "workload_family": "constructive_heat_conduction_pareto",
                "objective_directions": {
                    MATERIAL_OBJECTIVE_NAME: "minimize",
                    THERMAL_OBJECTIVE_NAME: "minimize",
                },
                "parent_configuration_sha256": typed_json_sha256(frozen_parent),
                "parent_configuration": thaw_json(frozen_parent),
                "prior_action_outcome_history": {
                    "schema_version": 1,
                    "actions": actions,
                    "treatment_visibility": "action_outcomes_only",
                },
                "history_visibility": (
                    "card_blind_evaluated_actions_only_from_completed_prior_stages"
                ),
                "finite_contract_sha256": variation.contract.identity_sha256,
                "eligibility_receipt_sha256": receipt.receipt_sha256,
                "raw_option_count": len(receipt.option_phenotypes),
                "eligible_option_count": len(receipt.eligible_option_ids),
                "known_excluded_count": len(receipt.known_excluded_option_ids),
                "semantic_alias_count": len(receipt.alias_excluded_option_ids),
                "memory_projection_sha256": typed_json_sha256(memory),
                "memory_estimand_stratum_sha256": MEMORY_CONTEXT_SHA256,
                "memory_estimand_context": thaw_json(MEMORY_ESTIMAND_CONTEXT),
                "fixed_affine_utility_definition_sha256": (
                    _affine_spec().definition_sha256
                ),
            }
        )

    def cards(self, benchmark, session, parent, variation, memory):
        del benchmark, session, parent, variation, memory
        return tuple(
            _object(
                {
                    "insight_id": entry.reference.insight_id.value,
                    "version": entry.reference.version,
                    "content_sha256": entry.draft.content_sha256,
                    "claim": entry.draft.claim,
                    "trigger": entry.draft.trigger,
                    "mechanism": entry.draft.mechanism,
                    "evidence_summary": entry.draft.evidence_summary,
                    "status": "preregistered_hypothesis_to_test",
                }
            )
            for entry in self.memory.entries
        )


@dataclass(frozen=True, slots=True)
class _DiagnosticMemoryBlock:
    """One pre-provider eligible cohort awaiting current-lane support audit."""

    exposure: CampaignDiagnosticExposureReceipt
    eligible_references: tuple[InsightRef, ...]
    estimand_context: FrozenJsonObject
    full_block_permutation_rank: int
    cohort_selection_key_sha256: str

    @property
    def active_references(self) -> tuple[InsightRef, ...]:
        """Compatibility alias: these are eligible, not yet selected, cards."""

        return self.eligible_references

    @property
    def exact_context_sha256(self) -> str:
        return typed_json_sha256(self.estimand_context)

    def to_record(self) -> dict[str, object]:
        """Project the sealed generic block without reviving its old plan API."""

        return {
            "schema_version": 1,
            "reflection_exposure_receipt_sha256": self.exposure.receipt_sha256,
            "eligible_references": [
                {
                    "insight_id": value.insight_id.value,
                    "version": value.version,
                }
                for value in self.eligible_references
            ],
            "estimand_context": thaw_json(self.estimand_context),
            "estimand_context_sha256": self.exact_context_sha256,
            "full_block_permutation_rank": self.full_block_permutation_rank,
            "cohort_selection_key_sha256": self.cohort_selection_key_sha256,
        }


@dataclass(slots=True)
class _HeatDiagnosticBlockCoordinator:
    """Seal one deterministic diagnostic cohort for every portfolio stage.

    The coordinator deliberately owns no campaign-context authority.  It can
    derive the exact context projection that an upstream trusted runtime port
    must apply, and the downstream wave factory can verify that projection,
    but neither operation creates a second cohort or assignment plan.  This
    keeps both parent lanes in one complete randomized block even when context
    enrichment and wave construction are separate runtime phases.
    """

    memory: InsightMemoryBank
    seed_plan: BalancedSubsetBlockPlan
    learning_runtime: ClosedLoopCampaignLearningRuntime | None = None
    _diagnostic_blocks: dict[int, _DiagnosticMemoryBlock] = field(
        init=False, default_factory=dict
    )
    _exposure_block_counts: dict[str, int] = field(init=False, default_factory=dict)
    _eligible_receipts_by_generation: dict[int, tuple[str, ...]] = field(
        init=False,
        default_factory=dict,
    )

    @staticmethod
    def _diagnostic_estimand_context(
        exposure: CampaignDiagnosticExposureReceipt,
        *,
        eligible_references: tuple[InsightRef, ...],
        cohort_selection_key_sha256: str,
        full_block_permutation_rank: int,
    ) -> FrozenJsonObject:
        return _object(
            {
                "schema_version": 1,
                "workload_family": "constructive_heat_conduction_pareto",
                "treatment_unit": "one_complete_candidate_portfolio_wave",
                "intervention": (
                    "pre_outcome_inclusion_of_one_memory_card_from_the_sealed_cohort"
                ),
                "outcome": "fixed_affine_archive_joint_wave_gain",
                "objective_directions": {
                    MATERIAL_OBJECTIVE_NAME: "minimize",
                    THERMAL_OBJECTIVE_NAME: "minimize",
                },
                "archive_utility_id": AffineHypervolumeArchiveUtility(
                    _affine_spec()
                ).utility_id,
                "archive_utility_definition_sha256": _affine_spec().definition_sha256,
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
                    "reflection_exposure_receipt_sha256": exposure.receipt_sha256,
                    "eligible_references": [
                        {
                            "insight_id": value.insight_id.value,
                            "version": value.version,
                        }
                        for value in eligible_references
                    ],
                    "lane_ids": ["elite", "explorer"],
                    "active_arm_count": 1,
                    "neutral_arm_count": 1,
                    "shared_card_support_required_across_lanes": True,
                    "selection_key_sha256": cohort_selection_key_sha256,
                    "active_unit_rank": full_block_permutation_rank,
                    "externally_ranked_before_provider_call": True,
                },
                "adaptive_score_consumption": False,
                "causal_claim_boundary": (
                    "lane_randomized_single_block_active_vs_neutral_"
                    "not_same_parent_or_full_candidate_pool_matched"
                ),
                "card_vs_neutral_effect_identified": False,
                "online_score_update_allowed": False,
                "required_successor_design": (
                    "replicated_same_parent_same_full_pool_active_neutral_slots"
                ),
            }
        )

    def resolve(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> _DiagnosticMemoryBlock | None:
        generation = context.stage_request.step.generation
        if self.learning_runtime is None or generation == 1:
            return None
        receipt_sha256s = context.stage_request.test_eligible_reflection_receipt_sha256s
        cached = self._diagnostic_blocks.get(generation)
        if cached is not None:
            if self._eligible_receipts_by_generation[generation] != receipt_sha256s:
                raise ValueError(
                    "one diagnostic generation received inconsistent authenticated "
                    "reflection cohorts"
                )
            return cached
        if not receipt_sha256s:
            return None
        exposures = self.learning_runtime.diagnostic_exposures(receipt_sha256s)
        active: list[
            tuple[CampaignDiagnosticExposureReceipt, tuple[InsightRef, ...]]
        ] = []
        for exposure in exposures:
            references = tuple(
                entry.reference
                for entry in self.memory.entries_for(exposure.references)
                if entry.lifecycle_state is InsightLifecycleState.QUARANTINED
            )
            if references:
                active.append((exposure, references))
        if not active:
            return None
        exposure, active_references = min(
            active,
            key=lambda value: (
                self._exposure_block_counts.get(value[0].receipt_sha256, 0),
                value[0].barrier_generation,
                value[0].receipt_sha256,
            ),
        )
        permutation_rank = int.from_bytes(
            hashlib.sha256(
                (
                    f"{PROTOCOL_ID}:{OUTER_SEED}:{exposure.receipt_sha256}:"
                    f"generation:{generation}:two-lane-diagnostic"
                ).encode("ascii")
            ).digest(),
            "big",
        ) % math.factorial(2)
        cohort_selection_key_sha256 = _sha(
            f"{PROTOCOL_ID}:{OUTER_SEED}:{exposure.receipt_sha256}:"
            f"generation:{generation}:complete-support-cohort"
        )
        estimand_context = self._diagnostic_estimand_context(
            exposure,
            eligible_references=active_references,
            cohort_selection_key_sha256=cohort_selection_key_sha256,
            full_block_permutation_rank=permutation_rank,
        )
        block = _DiagnosticMemoryBlock(
            exposure=exposure,
            eligible_references=active_references,
            estimand_context=estimand_context,
            full_block_permutation_rank=permutation_rank,
            cohort_selection_key_sha256=cohort_selection_key_sha256,
        )
        self._diagnostic_blocks[generation] = block
        self._eligible_receipts_by_generation[generation] = receipt_sha256s
        self._exposure_block_counts[exposure.receipt_sha256] = (
            self._exposure_block_counts.get(exposure.receipt_sha256, 0) + 1
        )
        return block

    def project(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> CampaignPortfolioMemoryEstimandProjection | None:
        """Return the typed estimand projection the trusted runtime must seal."""

        diagnostic = self.resolve(context)
        if diagnostic is None:
            return None
        return CampaignPortfolioMemoryEstimandProjection(
            estimand_context=diagnostic.estimand_context,
            estimand_stratum_sha256=diagnostic.exact_context_sha256,
        )

    def require_projected_context(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> _DiagnosticMemoryBlock | None:
        """Verify that trusted upstream enrichment sealed this exact block."""

        diagnostic = self.resolve(context)
        if diagnostic is None:
            return None
        values = dict(context.evidence_context.items)
        if (
            values.get("memory_estimand_stratum_sha256")
            != diagnostic.exact_context_sha256
            or values.get("memory_estimand_context") != diagnostic.estimand_context
        ):
            raise ValueError(
                "selector context differs from its sealed diagnostic cohort"
            )
        return diagnostic


@dataclass(slots=True)
class _WaveFactory:
    ids: DeterministicIdFactory
    memory: InsightMemoryBank
    plan: BalancedSubsetBlockPlan
    utility: AffineHypervolumeArchiveUtility
    binding_factory: CalibratedCampaignBindingFactory
    coordinator: CalibratedPortfolioCampaignCoordinator
    records: list[dict[str, object]]
    target_conditioned_controller: (
        TargetConditionedCampaignOutcomeUpdater | None
    ) = None
    optimization_semantics: OptimizationSemantics | None = None
    bounded_dose_binding_factory: CalibratedCampaignBindingFactory | None = None
    learning_runtime: ClosedLoopCampaignLearningRuntime | None = None
    diagnostic_coordinator: _HeatDiagnosticBlockCoordinator | None = None
    matched_support_resolutions: list[Any] = field(
        init=False,
        default_factory=list,
    )
    matched_control_plans: list[Any] = field(
        init=False,
        default_factory=list,
    )
    matched_control_recourses: list[FrozenJsonObject] = field(
        init=False,
        default_factory=list,
    )

    def __post_init__(self) -> None:
        if self.optimization_semantics is not None:
            if type(self.optimization_semantics) is not OptimizationSemantics:
                raise TypeError("optimization_semantics must be exact or None")
            self.optimization_semantics.__post_init__()
        if self.diagnostic_coordinator is None:
            self.diagnostic_coordinator = _HeatDiagnosticBlockCoordinator(
                memory=self.memory,
                seed_plan=self.plan,
                learning_runtime=self.learning_runtime,
            )
            return
        if (
            self.diagnostic_coordinator.memory is not self.memory
            or self.diagnostic_coordinator.seed_plan is not self.plan
            or self.diagnostic_coordinator.learning_runtime is not self.learning_runtime
        ):
            raise ValueError(
                "diagnostic coordinator differs from wave-factory dependencies"
            )

    @property
    def _diagnostic_blocks(self) -> dict[int, _DiagnosticMemoryBlock]:
        coordinator = self.diagnostic_coordinator
        if coordinator is None:  # pragma: no cover - sealed by __post_init__.
            raise AssertionError("wave factory lost its diagnostic coordinator")
        return coordinator._diagnostic_blocks

    @staticmethod
    def _reflection_card(
        entry,
        exposure,
        *,
        card_key: str,
        assigned_score: float,
        finite_variation_contracts: tuple[Any, ...],
        optimization_memory_assessment: (
            PortfolioOptimizationMemoryAssessment | None
        ) = None,
    ):
        lineage = entry.evidence_lineage
        if lineage is None:
            raise ValueError("a reflected diagnostic card lost its evidence lineage")
        hypothesis = freeze_json(entry.draft.hypothesis_record())
        if type(hypothesis) is not FrozenJsonObject:
            raise AssertionError("reflection hypothesis did not freeze to an object")
        payload = compose_epistemic_prompt_payload(
            empirical_evidence=lineage.empirical_evidence,
            hypothesis=hypothesis,
        )
        if optimization_memory_assessment is not None:
            payload_record = thaw_json(payload)
            payload_record["optimization_memory_assessment"] = (
                optimization_memory_assessment.to_record()
            )
            payload = freeze_json(payload_record)
            if type(payload) is not FrozenJsonObject:
                raise AssertionError("signed reflection payload lost its object root")
        payload = project_action_neutral_insight_prompt_payload(
            entry,
            prompt_payload=payload,
            finite_variation_contracts=finite_variation_contracts,
        )
        return portfolio_card_from_insight_entry(
            entry,
            card_key=card_key,
            prompt_payload=payload,
            evidence_sha256=lineage.identity_sha256,
            source_receipt_sha256=exposure.receipt_sha256,
            assigned_score=assigned_score,
        )

    @staticmethod
    def _bounded_reflection_memory_dose(
        *,
        cards: tuple[PortfolioCard, ...],
        selected_entries: tuple[Any, ...],
        finite_contract: Any,
        optimization_semantics: OptimizationSemantics | None = None,
    ) -> BoundedPortfolioMemoryDoseContract | None:
        """Bind reflected-card claims to compatible current-parent actions.

        Seed/control cards intentionally remain outside this dose.  A reflected
        card is identifiable through its admitted source binding and receives
        exactly one supported proposal/evaluation member.  The generic dose
        validator owns all provider-schema and allocator enforcement.
        """

        if len(cards) != len(selected_entries):
            raise ValueError("cards and selected memory entries lost their join")
        supports = []
        for card, entry in zip(cards, selected_entries, strict=True):
            if card.source_binding is None:
                continue
            if entry.evidence_lineage is None:
                raise ValueError("a source-bound card lost reflected evidence")
            if optimization_semantics is not None:
                assessment = assess_portfolio_optimization_memory(
                    entry.evidence_lineage,
                    optimization_semantics,
                )
                if not assessment.forced_action_dose_allowed:
                    continue
            draft = entry.draft
            exact_semantics = PortfolioMemoryDoseCardSemantics.from_insight(
                card_key=card.card_key,
                card_content_sha256=card.content_sha256,
                draft=draft,
                evidence_lineage=entry.evidence_lineage,
                support_scope=PortfolioMemoryDoseSupportScope.EXACT_SOURCE_PARENT,
            )
            transfer = assess_portfolio_memory_context_transfer(
                exact_semantics,
                finite_contract,
            )
            if not transfer.exact_action_replay_authorized:
                continue
            supports.append(
                derive_portfolio_memory_dose_card_support(
                    exact_semantics,
                    finite_contract,
                )
            )
        if not supports:
            return None
        card_supports = tuple(sorted(supports, key=lambda value: value.card_key))
        administered = len(card_supports)
        if administered > PORTFOLIO_WIDTH:
            raise ValueError("reflected-card dose exceeds the evaluated portfolio")
        return BoundedPortfolioMemoryDoseContract(
            card_supports=card_supports,
            proposed_supported_member_bounds=(administered, administered),
            evaluated_supported_member_bounds=(administered, administered),
            minimum_unattributed_proposed_members=(
                CALIBRATED_PROPOSAL_WIDTH - administered
            ),
            minimum_unattributed_evaluated_members=(PORTFOLIO_WIDTH - administered),
            maximum_cards_per_member=1,
            require_every_assigned_card=True,
        )

    def _build_resolved(
        self,
        context: CampaignPortfolioWaveContext,
        *,
        diagnostic: _DiagnosticMemoryBlock | None,
        assignment_plan: BalancedSubsetBlockPlan,
        assignment: Any,
        issue_memory_credit: bool = True,
        diagnostic_design: CampaignDiagnosticSingletonBlock | None = None,
        subset_authorization_sha256: str | None = None,
        support_resolution_record: dict[str, object] | None = None,
        diagnostic_recourse_record: dict[str, object] | None = None,
    ):
        generation = context.stage_request.step.generation
        lane_id = ("elite", "explorer")[context.parent_slot]
        if diagnostic is None and diagnostic_design is not None:
            raise ValueError("diagnostic design requires its sealed cohort")
        if diagnostic is not None and diagnostic_design is None:
            raise ValueError("sealed diagnostic cohort requires its realized design")
        if type(issue_memory_credit) is not bool:
            raise TypeError("issue_memory_credit must be an exact bool")
        if diagnostic_recourse_record is not None and issue_memory_credit:
            raise ValueError("diagnostic recourse cannot issue memory credit")
        if diagnostic_recourse_record is None and not issue_memory_credit:
            raise ValueError("credit-free resolved waves require a recourse receipt")
        decision = assignment.decision
        selection_context = context.evidence_context
        card_by_ref = {}
        for payload in context.evidence_cards:
            value = thaw_json(payload)
            card_by_ref[(value["insight_id"], value["version"])] = payload
        scores = dict(decision.score_snapshot)
        selected_entries = self.memory.entries_for(decision.selected)
        semantics = self.optimization_semantics
        cards: list[PortfolioCard] = []
        for ordinal, entry in enumerate(selected_entries, start=1):
            card_key = f"card.{ordinal:02d}"
            if (
                diagnostic is not None
                and entry.reference in diagnostic.active_references
            ):
                cards.append(
                    self._reflection_card(
                        entry,
                        diagnostic.exposure,
                        card_key=card_key,
                        assigned_score=float(scores[entry.reference]),
                        finite_variation_contracts=(context.variation.contract,),
                        optimization_memory_assessment=(
                            None
                            if semantics is None
                            else assess_portfolio_optimization_memory(
                                entry.evidence_lineage, semantics
                            )
                        ),
                    )
                )
                continue
            payload = card_by_ref[
                (entry.reference.insight_id.value, entry.reference.version)
            ]
            cards.append(
                PortfolioCard(
                    card_key=card_key,
                    reference=entry.reference,
                    content_sha256=entry.draft.content_sha256,
                    evidence_sha256=typed_json_sha256(payload),
                    prompt_payload=payload,
                    assigned_score=float(scores[entry.reference]),
                )
            )
        source_registry = None
        if any(card.source_binding is not None for card in cards):
            source_registry = admit_portfolio_card_sources(
                selected_entries,
                tuple(cards),
            )
        memory_dose_contract = self._bounded_reflection_memory_dose(
            cards=tuple(cards),
            selected_entries=selected_entries,
            finite_contract=context.variation.contract,
            optimization_semantics=semantics,
        )
        selection = PortfolioSelectionRequest(
            call_id=self.ids.new_llm_call_id(),
            operation="select_portfolio",
            instruction=(
                "Propose sealed candidate options under the authenticated portfolio "
                "contract. The calibrated adapter, not this caller text, owns the "
                "rendered K8 proposal and full-support evaluation behavior."
            ),
            context=selection_context,
            finite_variation_contract=context.variation.contract,
            cards=tuple(cards),
            portfolio_size=PORTFOLIO_WIDTH,
            required_metric_ids=OBJECTIVE_IDS,
            min_distinct_families=3,
            require_supporting_cards=False,
            require_pairwise_disjoint_parent_patches=True,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            source_registry=source_registry,
            memory_dose_contract=memory_dose_contract,
        )
        selected_binding_factory = self.binding_factory
        if memory_dose_contract is not None:
            if self.bounded_dose_binding_factory is None:
                raise RuntimeError(
                    "a reflected memory dose requires its calibrated prompt scope"
                )
            selected_binding_factory = self.bounded_dose_binding_factory
        calibrated = selected_binding_factory.build(
            request=selection,
            variation=context.variation,
            wave_index=generation,
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
            calibrated,
            target_conditioned_context=target_context,
        )
        prompt = self.coordinator.render(selection)
        snapshot = self.utility.require_snapshot(context.stage_request.archive_utility)
        resolved_assignment = None
        memory_credit = None
        if issue_memory_credit:
            reward = AffineFrozenArchiveJointWaveReward(snapshot)
            credit_unit_id = self.ids.new_operator_invocation_id()
            resolved_assignment = ResolvedInsightAssignment.resolve(
                credit_unit_id=credit_unit_id,
                snapshot=assignment_plan.snapshot,
                expected_snapshot_sha256=assignment_plan.snapshot.snapshot_sha256,
                block_id=assignment.unit.unit_key,
                arm=MemoryAssignmentArm.DIAGNOSTIC,
                selection_decision=decision,
                prompt_shape_sha256=selection.card_snapshot_sha256,
            )
            memory_credit = PortfolioMemoryCreditPlan(
                decision=decision,
                credit_unit_id=credit_unit_id,
                aggregation=PortfolioRewardAggregationBinding(
                    aggregate=lambda outcomes, reward=reward: float(
                        reward(tuple(value.candidate for value in outcomes))
                    ),
                    aggregation_id=MEMORY_AGGREGATION_ID,
                    aggregation_version=1,
                    definition_sha256=MEMORY_AGGREGATION_DEFINITION_SHA256,
                ),
                card_snapshot_sha256=selection.card_snapshot_sha256,
                score_snapshot=assignment_plan.snapshot,
                assignment=resolved_assignment,
                card_source_registry_sha256=(
                    None if source_registry is None else source_registry.registry_sha256
                ),
                quarantine_admission=(
                    None if diagnostic is None else diagnostic.exposure.memory_admission
                ),
                quarantine_admission_subset_authorization_sha256=(
                    subset_authorization_sha256
                ),
                context_projection=(
                    PortfolioMemoryContextProjectionBinding.from_selector_context(
                        selection.context
                    )
                ),
            )
        wave = PortfolioVariationWaveRequest(
            selection_request=selection,
            parent=context.parent,
            generation=generation,
            label_prefix=f"heatg{generation:02d}p{context.parent_slot + 1:02d}",
            phase="generic_heat2d_portfolio",
            memory_credit=memory_credit,
        )
        self.records.append(
            {
                "generation": generation,
                "parent_slot": context.parent_slot,
                "parent_candidate_id": context.parent.candidate_id.value,
                "lane_id": lane_id,
                "resolved_memory_assignment": (
                    None
                    if resolved_assignment is None
                    else resolved_assignment.to_record()
                ),
                "resolved_memory_assignment_sha256": (
                    None
                    if resolved_assignment is None
                    else resolved_assignment.assignment_sha256
                ),
                "selection_request": selection.to_record(),
                "bounded_reflection_memory_dose": (
                    None
                    if memory_dose_contract is None
                    else memory_dose_contract.to_record()
                ),
                "calibrated_input_binding": calibrated.to_record(),
                "calibrated_prompt_sha256": _sha(prompt),
                "calibrated_prompt_utf8_bytes": len(prompt.encode("utf-8")),
                "prompt_definition_sha256": (
                    calibrated.context.scope.prompt_definition_sha256
                ),
                "selector_policy_definition_sha256": (
                    calibrated.context.scope.selector_policy_definition_sha256
                ),
                "allocator": self.coordinator.allocator.to_record(),
                "option_prompt_projection_sha256": (
                    None
                    if calibrated.option_prompt_projection is None
                    else calibrated.option_prompt_projection.projection_sha256
                ),
                "parent_measurement_binding_sha256": (
                    None
                    if getattr(context, "parent_measurement", None) is None
                    else context.parent_measurement.binding_sha256
                ),
                "selector_context_utf8_bytes": _canonical_json_size(
                    thaw_json(selection.context)
                ),
                "parent_measurement_utf8_bytes": (
                    0
                    if getattr(context, "parent_measurement", None) is None
                    else _canonical_json_size(context.parent_measurement.to_record())
                ),
                "proposal_width": CALIBRATED_PROPOSAL_WIDTH,
                "evaluation_width": PORTFOLIO_WIDTH,
                "assignment": assignment.to_record(),
                "assignment_plan_sha256": assignment_plan.receipt_sha256,
                "diagnostic_reflection_exposure": (
                    None
                    if diagnostic is None
                    else {
                        "exposure_receipt_sha256": (diagnostic.exposure.receipt_sha256),
                        "barrier_generation": (diagnostic.exposure.barrier_generation),
                        "eligible_references": [
                            {
                                "insight_id": value.insight_id.value,
                                "version": value.version,
                            }
                            for value in diagnostic.eligible_references
                        ],
                        "diagnostic_block_design": diagnostic_design.to_record(),
                        "complete_support_resolution": support_resolution_record,
                        "estimand_context": thaw_json(diagnostic.estimand_context),
                        "estimand_context_sha256": (diagnostic.exact_context_sha256),
                    }
                ),
                "diagnostic_recourse": diagnostic_recourse_record,
                "affine_snapshot": snapshot.to_record(),
            }
        )
        return wave

    def _build_matched_control_wave(
        self,
        context: CampaignPortfolioWaveContext,
        *,
        diagnostic: _DiagnosticMemoryBlock,
        matched_plan: Any,
        assignment: Any,
        arm_view: Any,
        support: Any,
        support_resolution: Any,
    ) -> PortfolioVariationWaveRequest:
        """Materialize one generic active/neutral lane without online credit."""

        generation = context.stage_request.step.generation
        lane_id = ("elite", "explorer")[context.parent_slot]
        memory_dose_contract = None
        if arm_view.memory_dose_allowed:
            memory_dose_contract = BoundedPortfolioMemoryDoseContract(
                card_supports=(support,),
                proposed_supported_member_bounds=(1, 1),
                evaluated_supported_member_bounds=(1, 1),
                minimum_unattributed_proposed_members=(CALIBRATED_PROPOSAL_WIDTH - 1),
                minimum_unattributed_evaluated_members=(PORTFOLIO_WIDTH - 1),
                maximum_cards_per_member=1,
                require_every_assigned_card=True,
            )
        selection = PortfolioSelectionRequest(
            call_id=self.ids.new_llm_call_id(),
            operation="select_portfolio",
            instruction=(
                "Propose sealed candidate options under the authenticated portfolio "
                "contract. The calibrated adapter owns K8 proposal and full-support "
                "evaluation behavior."
            ),
            context=context.evidence_context,
            finite_variation_contract=context.variation.contract,
            cards=arm_view.cards,
            portfolio_size=PORTFOLIO_WIDTH,
            required_metric_ids=OBJECTIVE_IDS,
            min_distinct_families=3,
            require_supporting_cards=False,
            require_pairwise_disjoint_parent_patches=True,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            source_registry=arm_view.source_registry,
            experimental_view_receipt=arm_view.experimental_view_receipt,
            memory_dose_contract=memory_dose_contract,
            candidate_pool_required_option_ids=(
                arm_view.required_common_pool_option_ids
            ),
        )
        selected_binding_factory = self.binding_factory
        if memory_dose_contract is not None:
            if self.bounded_dose_binding_factory is None:
                raise RuntimeError(
                    "an active matched arm requires its calibrated prompt scope"
                )
            selected_binding_factory = self.bounded_dose_binding_factory
        calibrated = selected_binding_factory.build(
            request=selection,
            variation=context.variation,
            wave_index=generation,
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
            calibrated,
            target_conditioned_context=target_context,
        )
        prompt = self.coordinator.render(selection)
        snapshot = self.utility.require_snapshot(context.stage_request.archive_utility)
        reward = AffineFrozenArchiveJointWaveReward(snapshot)
        aggregation = PortfolioRewardAggregationBinding(
            aggregate=lambda outcomes, reward=reward: float(
                reward(tuple(value.candidate for value in outcomes))
            ),
            aggregation_id=MEMORY_AGGREGATION_ID,
            aggregation_version=1,
            definition_sha256=MEMORY_AGGREGATION_DEFINITION_SHA256,
        )
        projection = PortfolioMemoryContextProjectionBinding.from_selector_context(
            selection.context
        )
        wave = PortfolioVariationWaveRequest(
            selection_request=selection,
            parent=context.parent,
            generation=generation,
            label_prefix=f"heatg{generation:02d}p{context.parent_slot + 1:02d}",
            phase="generic_heat2d_portfolio",
            matched_memory_control=PortfolioMemoryMatchedControlWavePlan(
                plan=matched_plan,
                assignment=assignment,
                arm_view=arm_view,
                aggregation=aggregation,
                context_projection=projection,
            ),
        )
        self.records.append(
            {
                "generation": generation,
                "parent_slot": context.parent_slot,
                "parent_candidate_id": context.parent.candidate_id.value,
                "lane_id": lane_id,
                "resolved_memory_assignment": None,
                "resolved_memory_assignment_sha256": None,
                "selection_request": selection.to_record(),
                "bounded_reflection_memory_dose": (
                    None
                    if memory_dose_contract is None
                    else memory_dose_contract.to_record()
                ),
                "calibrated_input_binding": calibrated.to_record(),
                "calibrated_prompt_sha256": _sha(prompt),
                "calibrated_prompt_utf8_bytes": len(prompt.encode("utf-8")),
                "prompt_definition_sha256": (
                    calibrated.context.scope.prompt_definition_sha256
                ),
                "selector_policy_definition_sha256": (
                    calibrated.context.scope.selector_policy_definition_sha256
                ),
                "allocator": self.coordinator.allocator.to_record(),
                "option_prompt_projection_sha256": (
                    None
                    if calibrated.option_prompt_projection is None
                    else calibrated.option_prompt_projection.projection_sha256
                ),
                "parent_measurement_binding_sha256": (
                    None
                    if getattr(context, "parent_measurement", None) is None
                    else context.parent_measurement.binding_sha256
                ),
                "selector_context_utf8_bytes": _canonical_json_size(
                    thaw_json(selection.context)
                ),
                "parent_measurement_utf8_bytes": (
                    0
                    if getattr(context, "parent_measurement", None) is None
                    else _canonical_json_size(context.parent_measurement.to_record())
                ),
                "proposal_width": CALIBRATED_PROPOSAL_WIDTH,
                "evaluation_width": PORTFOLIO_WIDTH,
                "assignment": assignment.to_record(),
                "assignment_plan_sha256": matched_plan.plan_sha256,
                "matched_memory_control": {
                    "plan_sha256": matched_plan.plan_sha256,
                    "arm": assignment.arm.value,
                    "arm_view_sha256": arm_view.view_sha256,
                    "support_resolution_receipt_sha256": (
                        support_resolution.receipt_sha256
                    ),
                    "outcome_pending": True,
                    "single_block_card_effect_identified": False,
                    "online_score_update_allowed": False,
                },
                "diagnostic_reflection_exposure": {
                    "exposure_receipt_sha256": diagnostic.exposure.receipt_sha256,
                    "barrier_generation": diagnostic.exposure.barrier_generation,
                    "eligible_references": [
                        {
                            "insight_id": value.insight_id.value,
                            "version": value.version,
                        }
                        for value in diagnostic.eligible_references
                    ],
                    "diagnostic_block_design": matched_plan.to_record(),
                    "complete_support_resolution": support_resolution.to_record(),
                    "estimand_context": thaw_json(diagnostic.estimand_context),
                    "estimand_context_sha256": diagnostic.exact_context_sha256,
                },
                "diagnostic_recourse": None,
                "affine_snapshot": snapshot.to_record(),
            }
        )
        return wave

    def build_batch(
        self,
        contexts: tuple[CampaignPortfolioWaveContext, ...],
    ) -> tuple[PortfolioVariationWaveRequest, ...]:
        if type(contexts) is not tuple or not contexts:
            raise ValueError("Heat2D wave batch requires a non-empty exact tuple")
        generations = {value.stage_request.step.generation for value in contexts}
        if len(generations) != 1:
            raise ValueError("Heat2D wave batch cannot mix generations")
        generation = next(iter(generations))
        if generation != FIRST_REFLECTION_CONSUMER_GENERATION:
            return tuple(self.build(value) for value in contexts)
        if len(contexts) != 2:
            raise RuntimeError("Heat2D diagnostic G5 requires both stable lanes")
        coordinator = self.diagnostic_coordinator
        if coordinator is None:  # pragma: no cover - sealed by __post_init__.
            raise AssertionError("wave factory lost its diagnostic coordinator")

        def lane_id_for(context: Any) -> str:
            return ("elite", "explorer")[context.parent_slot]

        cohort_by_lane = {
            lane_id_for(context): coordinator.require_projected_context(context)
            for context in contexts
        }
        cohorts = tuple(value for value in cohort_by_lane.values() if value is not None)
        if not cohorts:
            return tuple(self.build(value) for value in contexts)
        if len(cohorts) != len(contexts):
            raise RuntimeError("Heat2D G5 lanes received inconsistent cohorts")
        if len({value.exposure.receipt_sha256 for value in cohorts}) != 1:
            raise RuntimeError("Heat2D G5 lanes received different exposures")
        diagnostic = cohorts[0]
        entries = tuple(
            sorted(
                self.memory.entries_for(diagnostic.eligible_references),
                key=lambda value: value.reference,
            )
        )
        if not entries:
            raise RuntimeError("Heat2D G5 lost its eligible reflected cohort")

        contexts_by_lane = {lane_id_for(value): value for value in contexts}
        lane_contracts = tuple(
            context.variation.contract
            for _, context in sorted(contexts_by_lane.items())
        )
        cards_by_key: dict[str, PortfolioCard] = {}
        entries_by_key: dict[str, Any] = {}
        assessments_by_key: dict[str, PortfolioOptimizationMemoryAssessment] = {}
        semantics = self.optimization_semantics
        for ordinal, entry in enumerate(entries, start=1):
            assessment = (
                None
                if semantics is None
                else assess_portfolio_optimization_memory(
                    entry.evidence_lineage,
                    semantics,
                )
            )
            card = self._reflection_card(
                entry,
                diagnostic.exposure,
                card_key=f"card.heat.g05.r{ordinal:02d}",
                assigned_score=0.0,
                finite_variation_contracts=lane_contracts,
                optimization_memory_assessment=assessment,
            )
            if card.source_binding is None:  # pragma: no cover - constructor closes.
                raise AssertionError("source-bound Heat2D card lost its binding")
            cards_by_key[card.card_key] = card
            entries_by_key[card.card_key] = entry
            if assessment is not None:
                assessments_by_key[card.card_key] = assessment

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
                    card_key=card_key,
                    card_content_sha256=card.content_sha256,
                    draft=entries_by_key[card_key].draft,
                    evidence_lineage=(entries_by_key[card_key].evidence_lineage),
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
                    draft=entries_by_key[value.card.card_key].draft,
                    evidence_lineage=(
                        entries_by_key[value.card.card_key].evidence_lineage
                    ),
                    support_scope=(PortfolioMemoryDoseSupportScope.EXACT_SOURCE_PARENT),
                ),
            )
            for value in support_cards
        )
        dose_support_cards = tuple(
            value
            for value in exact_support_cards
            if semantics is None
            or assessments_by_key[value.card.card_key].forced_action_dose_allowed
        )
        if dose_support_cards:
            resolution = PortfolioMemoryMatchedSupportResolver().resolve(
                lanes=support_lanes,
                cards=dose_support_cards,
                selection_key_sha256=diagnostic.cohort_selection_key_sha256,
            )
        else:
            resolution = PortfolioMemoryMatchedSupportResolution(
                lane_ids=tuple(sorted(value.lane.lane_id for value in support_lanes)),
                eligible_card_keys=(),
                selected_card_key=None,
                selected_lane_supports=(),
                selection_key_sha256=diagnostic.cohort_selection_key_sha256,
            )
        self.matched_support_resolutions.append(resolution)
        if not resolution.eligible:
            recourse = _object(
                {
                    "schema_version": 1,
                    "generation": generation,
                    "status": "no_shared_support_active_neutral_card",
                    "complete_support_resolution": resolution.to_record(),
                    "memory_dose_administered": False,
                    "memory_credit_issued": False,
                }
            )
            self.matched_control_recourses.append(recourse)
            return tuple(
                self._build_resolved(
                    context,
                    diagnostic=None,
                    assignment_plan=self.plan,
                    assignment=self.plan.assignment_for(
                        generation,
                        lane_id_for(context),
                    ),
                    issue_memory_credit=False,
                    diagnostic_recourse_record=thaw_json(recourse),
                )
                for context in contexts
            )

        selected_card_key = resolution.selected_card_key
        assert selected_card_key is not None
        selected_entry = entries_by_key[selected_card_key]
        selected_card = cards_by_key[selected_card_key]
        source_registry = admit_portfolio_card_sources(
            (selected_entry,),
            (selected_card,),
        )
        ordered_units = tuple(
            StableMemoryAssignmentUnit(
                unit_key=(
                    f"reflection.{diagnostic.exposure.receipt_sha256[:12]}."
                    f"g{generation:02d}.{lane_id}"
                ),
                generation=generation,
                lane_id=lane_id,
            )
            for lane_id in sorted(contexts_by_lane)
        )
        matched_plan = PortfolioMemoryMatchedControlPlanner().plan(
            reference=selected_entry.reference,
            exact_context_sha256=diagnostic.exact_context_sha256,
            ordered_units=ordered_units,
            active_unit_rank=diagnostic.full_block_permutation_rank,
        )
        self.matched_control_plans.append(matched_plan)
        waves = []
        for context in contexts:
            lane_id = lane_id_for(context)
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
                self._build_matched_control_wave(
                    context,
                    diagnostic=diagnostic,
                    matched_plan=matched_plan,
                    assignment=assignment,
                    arm_view=arm_view,
                    support=resolution.support_for(lane_id),
                    support_resolution=resolution,
                )
            )
        return tuple(waves)

    def build(self, context: CampaignPortfolioWaveContext):
        generation = context.stage_request.step.generation
        coordinator = self.diagnostic_coordinator
        if coordinator is None:  # pragma: no cover - sealed by __post_init__.
            raise AssertionError("wave factory lost its diagnostic coordinator")
        diagnostic = coordinator.require_projected_context(context)
        if diagnostic is not None:
            raise RuntimeError(
                "Heat2D reflected memory requires atomic complete-support batch "
                "construction"
            )
        lane_id = ("elite", "explorer")[context.parent_slot]
        return self._build_resolved(
            context,
            diagnostic=None,
            assignment_plan=self.plan,
            assignment=self.plan.assignment_for(generation, lane_id),
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


@dataclass(frozen=True, slots=True)
class _HeatReflectionContrast:
    """Normalized Heat evidence consumed by pure reflection construction."""

    contrast_id: str
    wave_ordinal: int
    selection_role: str
    source_option_ids: tuple[str, ...]
    source_families: tuple[str, ...]
    source_parent_objectives: tuple[FrozenJsonObject, ...]
    target_objectives: FrozenJsonObject
    reward_hex: str
    dominates_any_parent: bool
    better_than_any_parent: bool

    def __post_init__(self) -> None:
        if type(self.contrast_id) is not str or len(self.contrast_id) != 64:
            raise ValueError("contrast_id must be a lowercase SHA-256 identity")
        try:
            bytes.fromhex(self.contrast_id)
        except ValueError as error:
            raise ValueError(
                "contrast_id must be a lowercase SHA-256 identity"
            ) from error
        if self.contrast_id != self.contrast_id.lower():
            raise ValueError("contrast_id must be a lowercase SHA-256 identity")
        if type(self.wave_ordinal) is not int or self.wave_ordinal <= 0:
            raise ValueError("wave_ordinal must be a positive exact integer")
        if type(self.selection_role) is not str or not self.selection_role:
            raise ValueError("selection_role must be a non-empty string")
        for name, values in (
            ("source_option_ids", self.source_option_ids),
            ("source_families", self.source_families),
        ):
            if type(values) is not tuple or any(
                type(value) is not str or not value for value in values
            ):
                raise TypeError(f"{name} must be an exact tuple of non-empty strings")
        if (
            type(self.source_parent_objectives) is not tuple
            or not self.source_parent_objectives
            or any(
                type(value) is not FrozenJsonObject
                for value in self.source_parent_objectives
            )
        ):
            raise TypeError(
                "source_parent_objectives must contain frozen objective objects"
            )
        if type(self.target_objectives) is not FrozenJsonObject:
            raise TypeError("target_objectives must be a frozen objective object")
        if type(self.reward_hex) is not str:
            raise TypeError("reward_hex must be a string")
        try:
            reward = float.fromhex(self.reward_hex)
        except ValueError as error:
            raise ValueError(
                "reward_hex must encode a finite binary64 value"
            ) from error
        if not math.isfinite(reward):
            raise ValueError("reward_hex must encode a finite binary64 value")
        if type(self.dominates_any_parent) is not bool:
            raise TypeError("dominates_any_parent must be an exact bool")
        if type(self.better_than_any_parent) is not bool:
            raise TypeError("better_than_any_parent must be an exact bool")

    def to_prompt_record(self, *, evidence_citation_key: str) -> dict[str, object]:
        self.__post_init__()
        if type(evidence_citation_key) is not str or not evidence_citation_key:
            raise ValueError("evidence_citation_key must be a non-empty string")
        return {
            "contrast_id": self.contrast_id,
            "evidence_citation_key": evidence_citation_key,
            "wave_ordinal": self.wave_ordinal,
            "selection_role": self.selection_role,
            "source_option_ids": list(self.source_option_ids),
            "source_families": list(self.source_families),
            "source_parent_objectives": [
                thaw_json(value) for value in self.source_parent_objectives
            ],
            "target_objectives": thaw_json(self.target_objectives),
            "reward_hex": self.reward_hex,
            "dominates_any_parent": self.dominates_any_parent,
            "better_than_any_parent": self.better_than_any_parent,
        }


def _normalize_reflection_contrasts(
    source_results: tuple[Any, ...],
) -> tuple[_HeatReflectionContrast, ...]:
    """Project runtime recombination results into immutable prompt evidence."""

    contrasts: list[_HeatReflectionContrast] = []
    for wave_ordinal, result in enumerate(source_results, start=1):
        receipt = result.receipt
        for member, outcome in zip(receipt.members, result.outcomes, strict=True):
            candidate = outcome.candidate
            if candidate is None:
                raise ValueError(
                    "reflection source outcome lacks an evaluated candidate"
                )
            contrasts.append(
                _HeatReflectionContrast(
                    contrast_id=member.outcome_sha256,
                    wave_ordinal=wave_ordinal,
                    selection_role=member.selection_role,
                    source_option_ids=tuple(member.source_option_ids),
                    source_families=tuple(member.source_families),
                    source_parent_objectives=tuple(
                        _object(dict(parent.objective_map))
                        for parent in outcome.prepared.plan.parents
                    ),
                    target_objectives=_object(dict(candidate.objective_map)),
                    reward_hex=outcome.reward.hex(),
                    dominates_any_parent=outcome.dominates_any_parent,
                    better_than_any_parent=outcome.better_than_any_parent,
                )
            )
    if not contrasts:
        raise ValueError("reflection requires at least one evaluated contrast")
    return tuple(contrasts)


def _build_reflection_generation_request(
    *,
    call_id: LLMCallId,
    contrasts: tuple[_HeatReflectionContrast, ...],
    allowed_option_families: tuple[str, ...],
) -> ReflectionGenerationRequest:
    """Build the exact provider-bound request without performing any I/O."""

    if type(call_id) is not LLMCallId:
        raise TypeError("call_id must be an exact LLMCallId")
    if (
        type(contrasts) is not tuple
        or not contrasts
        or any(type(value) is not _HeatReflectionContrast for value in contrasts)
    ):
        raise TypeError("contrasts must contain exact _HeatReflectionContrast values")
    for contrast in contrasts:
        _HeatReflectionContrast.__post_init__(contrast)
    if (
        type(allowed_option_families) is not tuple
        or not allowed_option_families
        or allowed_option_families != tuple(sorted(set(allowed_option_families)))
    ):
        raise ValueError("allowed_option_families must be non-empty and canonical")

    available = tuple(sorted(contrast.contrast_id for contrast in contrasts))
    evidence_catalog = ReflectionEvidenceCatalog.from_contrast_ids(available)
    prompt_contrasts = [
        contrast.to_prompt_record(
            evidence_citation_key=evidence_catalog.citation_key_for_contrast_id(
                contrast.contrast_id
            )
        )
        for contrast in contrasts
    ]
    contract = _reflection_contract(allowed_option_families)
    prompt = json.dumps(
        {
            "task": (
                "Derive one or two falsifiable, workload-transferable insights from "
                "the exact recombination contrasts for prospective use by the next "
                "finite typed-mutation selector. Each insight must name exactly one "
                "atomic allowed decision path and make a non-unknown directional "
                "prediction for every required metric relative to the current parent. "
                "Explain when the mechanism should apply, recommend only allowed "
                "option families, and cite exact evidence citation keys from the "
                "request-scoped catalog. Full contrast IDs are authenticated context "
                "and must not be copied into the output. These are unverified "
                "hypotheses under quarantine, never established facts."
            ),
            "objectives": list(OBJECTIVE_IDS),
            "contrasts": prompt_contrasts,
            "quarantine": "until a later preregistered testing block closes",
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return ReflectionGenerationRequest(
        call_id=call_id,
        operation="extract_insights",
        prompt=prompt,
        max_insights=2,
        min_insights=1,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        available_contrast_ids=available,
        insight_contract=contract,
        evidence_catalog=evidence_catalog,
    )


def _reflection_request_construction_record(
    request: ReflectionGenerationRequest,
) -> dict[str, object]:
    """Seal request, prompt, catalog, contract, and their exact citation join."""

    ReflectionGenerationRequest.__post_init__(request)
    catalog = request.evidence_catalog
    contract = request.insight_contract
    if catalog is None or contract is None:
        raise ValueError("Heat reflection construction requires catalog and contract")
    prompt = json.loads(request.prompt)
    prompt_contrasts = prompt.get("contrasts") if type(prompt) is dict else None
    if type(prompt_contrasts) is not list or any(
        type(value) is not dict for value in prompt_contrasts
    ):
        raise ValueError("reflection prompt contrasts must be an exact object list")
    expected_mapping = tuple(
        sorted((entry.contrast_id, entry.citation_key) for entry in catalog.entries)
    )
    observed_mapping = tuple(
        sorted(
            (
                str(value.get("contrast_id")),
                str(value.get("evidence_citation_key")),
            )
            for value in prompt_contrasts
        )
    )
    no_legacy_evidence_key = all(
        "evidence_key" not in value for value in prompt_contrasts
    )
    mapping_record = {
        "schema_version": 1,
        "entries": [
            {
                "contrast_id": contrast_id,
                "evidence_citation_key": citation_key,
            }
            for contrast_id, citation_key in observed_mapping
        ],
    }
    mapping_sha256 = typed_json_sha256(_object(mapping_record))
    identity_record = {
        "schema_version": 1,
        "call_id": request.call_id.value,
        "operation": request.operation,
        "prompt_sha256": _sha(request.prompt),
        "max_insights": request.max_insights,
        "min_insights": request.min_insights,
        "max_output_tokens": request.max_output_tokens,
        "temperature_hex": (
            None if request.temperature is None else request.temperature.hex()
        ),
        "available_contrast_ids": list(request.available_contrast_ids),
        "evidence_catalog_identity_sha256": catalog.catalog_identity_sha256,
        "insight_contract_identity_sha256": contract.identity_sha256,
        "evidence_citation_mapping_sha256": mapping_sha256,
    }
    return {
        **identity_record,
        "request_identity_sha256": typed_json_sha256(_object(identity_record)),
        "prompt_utf8_bytes": len(request.prompt.encode("utf-8", errors="strict")),
        "evidence_citation_mapping": mapping_record["entries"],
        "exact_evidence_citation_mapping": observed_mapping == expected_mapping,
        "no_legacy_evidence_key": no_legacy_evidence_key,
    }


@dataclass(slots=True)
class _ReflectionExecutor:
    generator: PydanticAIAgenticGenerator
    ids: DeterministicIdFactory
    records: list[dict[str, object]]
    optimization_semantics: OptimizationSemantics

    async def reflect(
        self,
        reflection_input: CampaignIdentifiableReflectionInput,
    ) -> FrozenJsonObject:
        if type(reflection_input) is not CampaignIdentifiableReflectionInput:
            raise TypeError("reflection_input must be exact")
        reflection_request = build_heat2d_identifiable_reflection_request(
            call_id=self.ids.new_llm_call_id(),
            reflection_input=reflection_input,
            optimization_semantics=self.optimization_semantics,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            min_insights=(2 if len(reflection_input.evidence.contrasts) >= 2 else 1),
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
        learning_envelope = build_heat2d_identifiable_reflection_learning_envelope(
            reflection_input=reflection_input,
            request=reflection_request,
            result=result,
            optimization_semantics=self.optimization_semantics,
        )
        learning_record = CampaignReflectionLearningRecordCodec.decode(
            learning_envelope
        )
        construction = identifiable_reflection_request_construction_record(
            reflection_request,
            reflection_input.evidence,
        )
        contract = reflection_request.insight_contract
        evidence_catalog = reflection_request.evidence_catalog
        if contract is None or evidence_catalog is None:  # pragma: no cover
            raise AssertionError("generic Heat reflection lost its contracts")
        record = {
            "call_id": reflection_request.call_id.value,
            "source_generation": reflection_input.query.wave.source_generation,
            "source_portfolio_generation": (
                reflection_input.query.source_portfolio_generation
            ),
            "sealed_cutoff_event_index_inclusive": (
                reflection_input.query.sealed_cutoff_event_index_inclusive
            ),
            "identifiable_reflection_input_sha256": reflection_input.input_sha256,
            "request_construction": construction,
            "available_contrast_ids": list(reflection_request.available_contrast_ids),
            "evidence_catalog": evidence_catalog.to_record(),
            "evidence_catalog_identity_sha256": (
                result.evidence_catalog_identity_sha256
            ),
            "insight_contract": contract.to_record(),
            "insights": [value.content_record() for value in result.insights],
            CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY: thaw_json(learning_envelope)[
                CAMPAIGN_REFLECTION_LEARNING_RECORD_KEY
            ],
            "campaign_reflection_learning_record_sha256": (
                learning_record.record_sha256
            ),
            "telemetry": _telemetry_record(telemetry),
            "quarantined": True,
            "lifecycle_promoted": False,
            "source_stage_payload_exposed": False,
            "recombination_results_exposed": False,
        }
        self.records.append(record)
        return _object(record)


@dataclass(frozen=True, slots=True)
class _PreparationRuntime:
    provider: ProgressAwareOpenRouterConfig
    source_closure_sha256: str

    def prepare(self, request):
        return CampaignAgentRuntimeReceipt(
            request_sha256=request.request_sha256,
            runtime_id="progress_aware_openrouter_campaign",
            runtime_version=1,
            definition_sha256=_sha("progress-aware-openrouter-campaign-v1"),
            accepted=True,
            evidence=_object(
                {
                    "provider_calls": 0,
                    "credential_read": False,
                    "provider_config": self.provider.to_manifest_record(),
                    "source_closure_sha256": self.source_closure_sha256,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class _PolicyTag:
    name: str


def _binding(name: str, implementation: object) -> CampaignPolicyBinding:
    return CampaignPolicyBinding(
        implementation=implementation,
        policy_id=name,
        policy_version=1,
        definition_sha256=_sha(f"heat2d-generic-campaign-policy:{name}:v1"),
    )


class _PreparationJournal:
    def __init__(self, journal: DurableJsonlJournal) -> None:
        self.journal = journal

    def append(self, record) -> None:
        self.journal.append(thaw_json(record))


class _ExecutionJournal:
    def __init__(self, journal: DurableJsonlJournal, execution_started_ns: int) -> None:
        self.journal = journal
        self.execution_started_ns = execution_started_ns

    async def append(self, event: CampaignExecutionEvent):
        self.journal.append(
            {
                "observation": {
                    "monotonic_ns_since_execution_start": (
                        time.perf_counter_ns() - self.execution_started_ns
                    ),
                    "observed_at_utc": _utc_now(),
                },
                "authenticated_campaign_event": event.to_record(),
            }
        )
        return CampaignJournalAck(event.event_sha256, True)


class _WavePreparationJournal:
    """Synchronously publish prospective, validated waves before dispatch."""

    def __init__(self, journal: DurableJsonlJournal, execution_started_ns: int) -> None:
        self.journal = journal
        self.execution_started_ns = execution_started_ns

    def record_prepared_wave(
        self,
        receipt: CampaignPortfolioWavePreparationReceipt,
    ) -> None:
        self.journal.append(
            {
                "observation": {
                    "monotonic_ns_since_execution_start": (
                        time.perf_counter_ns() - self.execution_started_ns
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


@dataclass(frozen=True, slots=True)
class _Bundle:
    benchmark: Any
    config: AgenticCampaignWorkloadConfig
    workload_ports: CampaignWorkloadPorts
    policies: CampaignPolicies
    experiment_profile: CampaignExperimentProfile | None
    prepared: PreparedEvolutionCampaign
    memory: InsightMemoryBank
    memory_plan: BalancedSubsetBlockPlan
    utility: AffineHypervolumeArchiveUtility
    ids: DeterministicIdFactory
    feedback_ledger: PortfolioOutcomeFeedbackLedger
    binding_factory: CalibratedCampaignBindingFactory
    bounded_dose_binding_factory: CalibratedCampaignBindingFactory
    coordinator: CalibratedPortfolioCampaignCoordinator
    direction_adjudicator: AbsoluteToleranceDirectionAdjudicator
    parent_measurement_projection: ParentMeasurementProjection


def _target_conditioned_controller(
    bundle: _Bundle,
    coordinator: CalibratedPortfolioCampaignCoordinator,
) -> TargetConditionedCampaignOutcomeUpdater | None:
    if ACQUISITION_MODE is not CampaignAcquisitionMode.TARGET_CONDITIONED:
        return None
    specification = _target_conditioned_specification()
    if type(coordinator.allocator) is not TargetConditionedSlateAllocatorAdapter:
        raise TypeError("Heat2D target-conditioned coordinator has a foreign allocator")
    if coordinator.allocator.profile != specification.profile:
        raise ValueError("Heat2D target-conditioned profile drifted")
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


@dataclass(frozen=True, slots=True)
class _ProductionLearningBundle:
    runtime: ClosedLoopCampaignLearningRuntime
    coordinator: ClosedLoopCampaignLearning
    evidence_registry: CampaignEvidenceRegistry
    identifiable_reflection_evidence_source: (
        CommittedRegistryIdentifiableReflectionEvidenceSource
    )


def _production_learning_bundle(bundle: _Bundle) -> _ProductionLearningBundle:
    """Compose the workload-neutral lifecycle around Heat's injected identities."""

    evaluator = bundle.benchmark.detailed_evaluator
    if evaluator is None:
        raise RuntimeError("production learning requires a detailed evaluator")
    evaluator_contract_sha256 = typed_json_sha256(
        _object(evaluator.evaluator_identity.to_record())
    )
    campaign_sha256 = bundle.prepared.preparation_sha256
    workload_instance_sha256 = bundle.config.configuration_sha256
    adjudicator_sha256 = bundle.direction_adjudicator.definition_sha256
    scope = HypothesisAuditScope(
        workload_instance_sha256s=(workload_instance_sha256,),
        evaluator_contract_sha256=evaluator_contract_sha256,
        metric_adjudicator_definition_sha256=adjudicator_sha256,
        campaign_sha256s=(campaign_sha256,),
    )
    evidence_registry = CampaignEvidenceRegistry()
    coordinator = ClosedLoopCampaignLearning(memory=bundle.memory)
    generation_auditor = TransactionalPortfolioGenerationAuditor(
        evidence_registry=evidence_registry,
        campaign_sha256=campaign_sha256,
        workload_instance_sha256=workload_instance_sha256,
        evaluator_contract_sha256=evaluator_contract_sha256,
        metric_projector=ObjectiveDeltaMetricEffectProjector(adjudicator_sha256),
        action_semantics_compiler=FinitePortfolioActionSemanticsCompiler(),
        hypothesis_matcher=PortableFiniteActionHypothesisMatcher(),
    )
    runtime = ClosedLoopCampaignLearningRuntime(
        learning=coordinator,
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
        generation_auditor=generation_auditor,
    )
    return _ProductionLearningBundle(
        runtime=runtime,
        coordinator=coordinator,
        evidence_registry=evidence_registry,
        identifiable_reflection_evidence_source=(
            CommittedRegistryIdentifiableReflectionEvidenceSource(
                registry=evidence_registry,
                campaign_sha256=campaign_sha256,
                workload_instance_sha256=workload_instance_sha256,
                evaluator_contract_sha256=evaluator_contract_sha256,
            )
        ),
    )


def _model_profile_sha256() -> str:
    return MODEL_EXECUTION_PROFILE.profile_sha256


def _calibration_scope(
    prepared: PreparedEvolutionCampaign,
    *,
    allocator: (
        FullSupportSlatePolicy
        | ModelAnchoredCalibratedSlatePolicy
        | StructuralPosteriorSlatePolicy
        | OperatorStratifiedStructuralPosteriorSlatePolicy
        | HorizonBoundedStructuralPosteriorSlatePolicy
    ),
    option_prompt_projection: FiniteOptionPromptProjectionPolicy,
    bounded_memory_dose: bool = False,
) -> ForecastCalibrationScope:
    allocator.__post_init__()
    option_prompt_projection.__post_init__()
    return ForecastCalibrationScope(
        model_profile_sha256=_model_profile_sha256(),
        prompt_definition_sha256=calibrated_portfolio_prompt_definition_sha256(
            option_prompt_projection,
            bounded_memory_dose=bounded_memory_dose,
            proposal_support=_proposal_support_policy() is not None,
            hierarchical_composition_required_proposals=(
                VARIATION_TOPOLOGY.hierarchical_composition_required_proposals
            ),
            feasibility_witness_mode=FEASIBILITY_WITNESS_MODE,
            constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
        ),
        selector_policy_definition_sha256=_selector_policy_definition_sha256(),
        benchmark_sha256=typed_json_sha256(prepared.benchmark_session.benchmark),
        session_sha256=prepared.benchmark_session.session_sha256,
    )


def _parent_measurement_projection(
    prepared: PreparedEvolutionCampaign,
    benchmark: Any,
) -> ParentMeasurementProjection:
    """Build the one treatment/control-shared Heat measurement authority."""

    if type(prepared) is not PreparedEvolutionCampaign:
        raise TypeError("prepared must be exact PreparedEvolutionCampaign")
    prepared.__post_init__()
    evaluator = benchmark.detailed_evaluator
    semantics = benchmark.optimization_semantics
    resolution = benchmark.objective_resolution
    if evaluator is None or semantics is None or resolution is None:
        raise RuntimeError("Heat parent measurement authorities were not bound")
    return create_parent_measurement_projection(
        benchmark_sha256=typed_json_sha256(prepared.benchmark_session.benchmark),
        session_sha256=prepared.benchmark_session.session_sha256,
        decision_metrics=DecisionMetricProjection.from_optimization_semantics(
            semantics
        ),
        evaluator=evaluator.evaluator_identity,
        objective_resolution_identity=objective_resolution_policy_metadata(resolution),
    )


def _calibrated_selector(
    *,
    runner: Any,
    coordinator: CalibratedPortfolioCampaignCoordinator,
    allocator: CalibratedPortfolioAllocator | None = None,
) -> (
    PydanticAIFullSupportCalibratedPortfolioSelectionPolicy
    | PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy
    | PydanticAIStructuralPosteriorCalibratedPortfolioSelectionPolicy
    | PydanticAIOperatorStratifiedCalibratedPortfolioSelectionPolicy
    | PydanticAIHorizonBoundedCalibratedPortfolioSelectionPolicy
    | PydanticAIConstraintDecoupledHorizonPortfolioSelectionPolicy
    | PydanticAIEvidenceCalibratedSourceMixPortfolioSelectionPolicy
    | PydanticAIMinimumInterventionHorizonPortfolioSelectionPolicy
    | PydanticAIContextualSearchAllocationPortfolioSelectionPolicy
    | PydanticAITargetConditionedCalibratedPortfolioSelectionPolicy
):
    """One obvious injection seam for allocator experiments."""

    selected = coordinator.allocator if allocator is None else allocator
    expected_allocator_type = type(_default_allocator())
    if type(selected) is not expected_allocator_type:
        raise TypeError("Heat live selector uses a foreign allocation policy")
    selected.__post_init__()
    if selected != coordinator.allocator:
        raise ValueError("selector allocator differs from campaign coordinator")
    built = coordinator.build_selector(runner)
    expected_selector_type = (
        PydanticAIContextualSearchAllocationPortfolioSelectionPolicy
        if CONTEXTUAL_SEARCH_ALLOCATION
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
        }[type(selected)]
    )
    if type(built) is not expected_selector_type:
        raise TypeError("Heat coordinator built a foreign selector")
    return built


def _outcome_conditioned_selector(
    *,
    runner: Any,
    bundle: _Bundle,
) -> OutcomeConditionedPortfolioSelectionPolicy:
    """Compose the generic all-action selector with optional Heat authorities."""

    semantics = bundle.benchmark.optimization_semantics
    if type(semantics) is not OptimizationSemantics:
        raise TypeError("Heat outcome-conditioned selector requires exact semantics")
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
        action_semantics_factory=heat2d_action_space_semantics,
        metric_projector=Heat2DExactMaterialProjector(),
        risk_aversion=0.5,
        diversity_weight=0.05,
        beam_width=256,
    )


def _outcome_conditioned_calibration_scope(bundle: _Bundle) -> ForecastCalibrationScope:
    """Keep forecast learning in a selector/prompt-specific empirical stratum."""

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
        jitter_domain="heat2d-generic-g6-delayed-identifiable-v5",
        app_title=("AgentEvolve AAAI 2027 delayed-identifiable Heat2D G6 campaign"),
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


def _evaluator_settings(output_root: Path) -> Heat2DDirectV3Settings:
    return Heat2DDirectV3Settings(
        output_root=output_root,
        resolution=1001,
        cpu_set="8",
        timeout_s=180.0,
        required_numpy_version="2.3.5",
        external_concurrency=1,
    )


def _prepare_bundle(
    *,
    run_dir: Path,
    run_id: str,
    preparation_journal: DurableJsonlJournal,
    source_closure_sha256: str,
    allocator: FullSupportSlatePolicy
    | ModelAnchoredCalibratedSlatePolicy
    | None = None,
) -> _Bundle:
    settings = _evaluator_settings(run_dir / "pde")
    benchmark = _scientific_benchmark(settings)
    preflight = benchmark.problem.preflight()
    ids = DeterministicIdFactory(AGENTIC_ID_NAMESPACE)
    memory = InsightMemoryBank(
        id_factory=ids,
        exploration_probability=Fraction(1, 1),
    )
    _seed_memory(memory)
    plan = _memory_plan(memory)
    evidence = _Evidence(memory, plan)
    workload_kit = compose_heat2d_pareto_campaign_workload(
        benchmark=benchmark,
        evaluator_concurrency_cap=1,
        evaluator_preflight_receipt=_object(preflight),
        resource_lease_receipt=_object(
            {
                "cpu_set": "8",
                "external_concurrency": 1,
                "lease_scope": "serialized_direct_v3_campaign",
            }
        ),
        evidence=AgenticCampaignEvidenceProjections(
            projection_id="heat2d_generic_campaign_evidence",
            projection_version=1,
            definition_sha256=_sha("heat2d-generic-campaign-evidence-v1"),
            initialize_memory=evidence.initialize_memory,
            context=evidence.context,
            cards=evidence.cards,
        ),
    )
    config = workload_kit.to_campaign_workload()
    utility = AffineHypervolumeArchiveUtility(_affine_spec())
    archive_context_projector = (
        AuthenticatedAffineFrontierContextProjector()
        if ACQUISITION_MODE is CampaignAcquisitionMode.TARGET_CONDITIONED
        else affine_frontier_context_projector(ARCHIVE_CONTEXT_MODE)
    )
    parent_selector = (
        ResidualHypervolumeCampaignParentSelector()
        if RESIDUAL_FRONTIER_PLANNING
        else StagnationAwareDiverseCampaignParentSelector()
        if ACQUISITION_MODE is CampaignAcquisitionMode.HORIZON_BOUNDED
        else ArchiveDiverseEliteCampaignParentSelector()
    )
    selected_allocator = _default_allocator() if allocator is None else allocator
    expected_allocator_type = type(_default_allocator())
    if type(selected_allocator) is not expected_allocator_type:
        raise TypeError("allocator differs from the active Heat acquisition arm")
    selected_allocator.__post_init__()
    option_prompt_projection = _option_prompt_projection()
    portfolio_policy_definition = _sha(
        json.dumps(
            {
                "schema_version": 1,
                "selector_policy_definition_sha256": (
                    _selector_policy_definition_sha256()
                ),
                "allocator": selected_allocator.to_record(),
                "option_prompt_projection": {
                    "policy_id": option_prompt_projection.policy_id,
                    "policy_version": option_prompt_projection.policy_version,
                    "definition_sha256": option_prompt_projection.definition_sha256,
                    "configuration_sha256": (
                        option_prompt_projection.configuration_sha256
                    ),
                    "metadata_keys": list(option_prompt_projection.metadata_keys or ()),
                },
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    experiment_profile: CampaignExperimentProfile | None
    if COMMON_POOL_ACQUISITION and COMMON_CANDIDATE_POOL_SIZE is None:
        context_local_successor = (
            _proposal_support_policy() is not None
            and archive_context_projector is not None
            and VARIATION_TOPOLOGY.mode is not CampaignVariationTopologyMode.FLAT_R2
        )
        experiment_profile = reference_campaign_experiment_profile(
            profile_id=f"reference_heat_{MODEL_EXECUTION_PROFILE.profile_id}",
            model_execution=MODEL_EXECUTION_PROFILE,
            implementations=ReferenceCampaignImplementations(
                parent_selection=parent_selector,
                memory_assignment=plan,
                portfolio_selection=selected_allocator,
                recombination=_PolicyTag("recombination"),
                reflection=_PolicyTag("reflection_catalog_v1"),
                archive_context=archive_context_projector,
                variation_topology=(
                    _reference_variation_topology_binding(workload_kit)
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
            constraint_decoupled_acquisition=(CONSTRAINT_DECOUPLED_ACQUISITION),
            minimum_intervention_projection=MINIMUM_INTERVENTION_PROJECTION,
            evidence_calibrated_source_mix=EVIDENCE_CALIBRATED_SOURCE_MIX,
            contextual_search_allocation=CONTEXTUAL_SEARCH_ALLOCATION,
        )
        behavior = experiment_profile.behavior(archive_utility=utility)
    else:
        experiment_profile = None
        behavior = PortfolioCampaignBehavior(
            parent_selection=_binding("archive_elite_explorer", parent_selector),
            memory_assignment=_binding("balanced_complete_g6", plan),
            portfolio_selection=CampaignPolicyBinding(
                implementation=selected_allocator,
                policy_id="full_support_calibrated_k8_to_k8",
                policy_version=1,
                definition_sha256=portfolio_policy_definition,
            ),
            recombination=_binding(
                "archive_aware_disjoint_union", _PolicyTag("recombination")
            ),
            reflection=_binding(
                "async_quarantined_reflection_catalog_v1",
                _PolicyTag("reflection_catalog_v1"),
            ),
            reflection_supervision=CampaignReflectionSupervisionPolicy(
                ReflectionFailureMode.FAIL_AT_NEXT_STAGE_BOUNDARY
            ),
            archive_utility=utility,
        )
    policies = behavior.bind()
    preset = (
        experiment_profile.preset(outer_seed=OUTER_SEED)
        if experiment_profile is not None
        else DelayedPortfolioCampaignPreset.generations(
            GENERATION_COUNT,
            outer_seed=OUTER_SEED,
            parents_per_portfolio_generation=PARENTS_PER_PORTFOLIO,
            portfolio_width=PORTFOLIO_WIDTH,
            recombinations_per_parent=RECOMBINATIONS_PER_PARENT,
            evaluator_concurrency=EVALUATOR_CONCURRENCY,
            agent_concurrency=AGENT_CONCURRENCY,
            agent_queue_capacity=AGENT_QUEUE_CAPACITY,
        )
    )
    campaign = preset.compose(
        workload=config,
        behavior=behavior,
        runtime=_PreparationRuntime(_provider_config(), source_closure_sha256),
        journals=(_PreparationJournal(preparation_journal),),
    )
    workload_ports = campaign.workload
    prepared = campaign.prepare()
    if experiment_profile is not None:
        experiment_profile.prepared_conformance_record(
            prepared=prepared,
            archive_utility=utility,
            outer_seed=OUTER_SEED,
        )
    if (
        tuple(step.planned_agent_calls for step in prepared.schedule.steps)
        != (2, 1, 2, 0, 2, 0)
        or tuple(
            (wave.source_generation, wave.promotion_barrier_generation)
            for wave in prepared.schedule.reflection_waves
        )
        != ((2, 4),)
        or prepared.schedule.planned_candidate_evaluations
        + prepared.protocol.required_seed_count
        != PLANNED_UNIQUE_EVALUATIONS
        or prepared.schedule.planned_agent_calls != PLANNED_LOGICAL_LLM_CALLS
    ):
        raise RuntimeError(
            "prepared Heat G6 schedule differs from its configured evaluation/call contract"
        )
    feedback_ledger = PortfolioOutcomeFeedbackLedger()
    scope = _calibration_scope(
        prepared,
        allocator=selected_allocator,
        option_prompt_projection=option_prompt_projection,
    )
    binding_factory = CalibratedCampaignBindingFactory(
        scope=scope,
        objectives=equal_weight_slate_objectives(benchmark.objectives),
        ledger=feedback_ledger,
        option_prompt_projection=option_prompt_projection,
        common_candidate_pool_policy=_common_candidate_pool_policy(),
        proposal_support_policy=_proposal_support_policy(),
        assign_all_cards_by_default=_common_candidate_pool_policy() is None,
    )
    bounded_dose_binding_factory = CalibratedCampaignBindingFactory(
        scope=_calibration_scope(
            prepared,
            allocator=selected_allocator,
            option_prompt_projection=option_prompt_projection,
            bounded_memory_dose=True,
        ),
        objectives=equal_weight_slate_objectives(benchmark.objectives),
        ledger=feedback_ledger,
        option_prompt_projection=option_prompt_projection,
        common_candidate_pool_policy=_common_candidate_pool_policy(),
        proposal_support_policy=_proposal_support_policy(),
        assign_all_cards_by_default=_common_candidate_pool_policy() is None,
    )
    coordinator = CalibratedPortfolioCampaignCoordinator(
        allocator=selected_allocator,
        constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
        minimum_intervention_projection=MINIMUM_INTERVENTION_PROJECTION,
        evidence_calibrated_source_mix=EVIDENCE_CALIBRATED_SOURCE_MIX,
        contextual_search_allocation=CONTEXTUAL_SEARCH_ALLOCATION,
    )
    direction_adjudicator = AbsoluteToleranceDirectionAdjudicator(
        benchmark_sha256=scope.benchmark_sha256,
        session_sha256=scope.session_sha256,
        resolutions=tuple(
            MetricDirectionResolution(metric_id=metric_id, absolute_tolerance=0.0)
            for metric_id in OBJECTIVE_IDS
        ),
    )
    parent_measurement_projection = _parent_measurement_projection(
        prepared,
        benchmark,
    )
    return _Bundle(
        benchmark,
        config,
        workload_ports,
        policies,
        experiment_profile,
        prepared,
        memory,
        plan,
        utility,
        ids,
        feedback_ledger,
        binding_factory,
        bounded_dose_binding_factory,
        coordinator,
        direction_adjudicator,
        parent_measurement_projection,
    )


def _manifest(
    run_id: str,
    mode: str,
    source: dict[str, object],
    source_snapshot: dict[str, object] | None = None,
) -> dict[str, object]:
    spec = _affine_spec()
    allocator = _default_allocator()
    option_projection = _option_prompt_projection()
    archive_context = (
        AuthenticatedAffineFrontierContextProjector()
        if ACQUISITION_MODE is CampaignAcquisitionMode.TARGET_CONDITIONED
        else affine_frontier_context_projector(ARCHIVE_CONTEXT_MODE)
    )
    campaign_preset = DelayedPortfolioCampaignPreset.generations(
        GENERATION_COUNT,
        outer_seed=OUTER_SEED,
        parents_per_portfolio_generation=PARENTS_PER_PORTFOLIO,
        portfolio_width=PORTFOLIO_WIDTH,
        recombinations_per_parent=RECOMBINATIONS_PER_PARENT,
        evaluator_concurrency=EVALUATOR_CONCURRENCY,
        agent_concurrency=AGENT_CONCURRENCY,
        agent_queue_capacity=AGENT_QUEUE_CAPACITY,
    )
    campaign_protocol = campaign_preset.protocol(required_seed_count=2)
    invocation_observation = DirectV3Evaluator(
        _evaluator_settings(ARTIFACT_ROOT / run_id / "pde")
    ).invocation_observation()
    replay = _load_sealed_replay_source()
    return {
        "schema_version": 1,
        "run_id": run_id,
        "mode": mode,
        "created_at_utc": _utc_now(),
        "sealed_accepted_output_replay": None if replay is None else replay[1],
        "claim_boundary": {
            "workflow_development_only": True,
            "paper_ready_result": False,
            "matched_baseline_in_this_run": False,
            "g6_memory_causal_claim": False,
            "reflection_efficacy_claim": False,
        },
        "workload": {
            "problem_workload_id": WORKLOAD_ID,
            "formulation_definition_sha256": FORMULATION_DEFINITION_SHA256,
            "evaluator": "direct_v3",
            "resolution": 1001,
            "cpu_set": "8",
            "external_concurrency": 1,
            "objectives": list(OBJECTIVE_IDS),
            "finite_catalog_id": CATALOG_ID,
        },
        "protocol": {
            "protocol_id": campaign_protocol.protocol_id,
            "experiment_protocol_label": PROTOCOL_ID,
            "preset_definition_sha256": campaign_preset.definition_sha256,
            "protocol_definition_sha256": (campaign_protocol.definition_sha256),
            "deterministic_id_namespace": AGENTIC_ID_NAMESPACE,
            "generations": GENERATION_COUNT,
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
            "reflections": len(REFLECTION_SOURCE_GENERATIONS),
            "reflection_source_generations": list(REFLECTION_SOURCE_GENERATIONS),
            "reflection_admission_generations": list(REFLECTION_ADMISSION_GENERATIONS),
            "first_reflection_consumer_generation": (
                FIRST_REFLECTION_CONSUMER_GENERATION
            ),
            "terminal_reflection": False,
            "reflection_promotion_block_pairs": 1,
            "planned_unique_evaluations": PLANNED_UNIQUE_EVALUATIONS,
            "planned_candidate_occurrences": PLANNED_UNIQUE_EVALUATIONS,
            "maximum_cache_reuse_occurrences": MAX_CACHE_REUSE_OCCURRENCES,
            "planned_logical_llm_calls": PLANNED_LOGICAL_LLM_CALLS,
            "recombination_shortfall_health_failure": True,
            "cache_reuse_requires_exact_accounting": True,
        },
        "model": {
            **_provider_config().to_manifest_record(),
            "execution_profile": MODEL_EXECUTION_PROFILE.to_record(),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature_hex": (None if TEMPERATURE is None else TEMPERATURE.hex()),
            "reasoning_mode": None,
        },
        "portfolio_selection": {
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
                    "optional_workload_action_semantics": True,
                    "optional_exact_metric_projection": True,
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
                    option_projection,
                    proposal_support=_proposal_support_policy() is not None,
                    hierarchical_composition_required_proposals=(
                        VARIATION_TOPOLOGY.hierarchical_composition_required_proposals
                    ),
                    feasibility_witness_mode=FEASIBILITY_WITNESS_MODE,
                    constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
                )
            ),
            "bounded_memory_dose_prompt_definition_sha256": (
                calibrated_portfolio_prompt_definition_sha256(
                    option_projection,
                    bounded_memory_dose=True,
                    proposal_support=_proposal_support_policy() is not None,
                    hierarchical_composition_required_proposals=(
                        VARIATION_TOPOLOGY.hierarchical_composition_required_proposals
                    ),
                    feasibility_witness_mode=FEASIBILITY_WITNESS_MODE,
                    constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
                )
            ),
            "feasibility_witness_mode": FEASIBILITY_WITNESS_MODE.value,
            "allocator": allocator.to_record(),
            "option_prompt_projection": {
                "policy_id": option_projection.policy_id,
                "policy_version": option_projection.policy_version,
                "definition_sha256": option_projection.definition_sha256,
                "configuration_sha256": option_projection.configuration_sha256,
                "metadata_keys": list(option_projection.metadata_keys or ()),
            },
            "hard_allocation_contract_rendered_preprovider": {
                "exact_subset_size": PORTFOLIO_WIDTH,
                "pairwise_disjoint_parent_patches": True,
                "minimum_distinct_families": 3,
            },
            "closed_validation_reason_codes": sorted(
                value.value for value in ValidationIssueReasonCode
            ),
        },
        "failure_observability": {
            "validated_wave_preparation_journal": "wave_preparations.jsonl",
            "wave_preparation_published_before_selector_dispatch": True,
            "cancelled_sibling_terminal_outcomes_required": True,
            "raw_invalid_model_output_persisted": False,
        },
        "parent_measurement": {
            "enabled_for_live_parent_selection": True,
            "raw_scientific_and_decision_values_separate": True,
            "current_wave_outcomes_included": False,
        },
        "reflection": {
            "evidence_mode": "sealed_direct_single_mutation_g1_only",
            "request_input_port": "CampaignIdentifiableReflectionInput",
            "citation_mode": "authenticated_request_scoped_short_keys_v1",
            "citation_keys": "contiguous_e0001_through_eNNNN",
            "resolution": "exact_key_to_full_contrast_id_no_fuzzy_matching",
            "supervision": CampaignReflectionSupervisionPolicy(
                ReflectionFailureMode.FAIL_AT_NEXT_STAGE_BOUNDARY
            ).to_record(),
            "visibility": "quarantined_until_block_close",
            "g4_admission_g5_first_consumer": True,
            "terminal_reflection_disabled": True,
            "bounded_g5_memory_dose": {
                "supported_proposed_members_per_reflected_card": 1,
                "supported_evaluated_members_per_reflected_card": 1,
                "maximum_cards_per_member": 1,
                "unattributed_members_are_prompt_exposed_not_blinded": True,
            },
        },
        "utility": spec.to_record(),
        "utility_definition_sha256": spec.definition_sha256,
        "utility_reference_qualification": {
            "all_void_thermal_term": QUALIFIED_ALL_VOID_THERMAL_TERM,
            "all_void_thermal_term_hex": QUALIFIED_ALL_VOID_THERMAL_TERM.hex(),
            "source_manifest_sha256": QUALIFIED_ALL_VOID_MANIFEST_SHA256,
            "thermal_reference": THERMAL_AFFINE_REFERENCE,
            "material_reference": MATERIAL_AFFINE_REFERENCE,
        },
        "objective_resolution": _objective_resolution().to_record(),
        "memory": {
            "seed_insights": 3,
            "assigned_per_wave": 2,
            "portfolio_trials": 6,
            "complete_subset_blocks": 2,
            "lane_ids": ["elite", "explorer"],
            "assignments_frozen_preprovider": True,
            "adaptive_score_consumption": False,
            "causal_effect_claim_allowed": False,
        },
        "source_identity": source,
        "source_snapshot": source_snapshot,
        "evaluator_process_invocation_observation": invocation_observation,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "python_executable_invoked": os.path.abspath(sys.executable),
            "python_executable_resolved": str(
                Path(sys.executable).resolve(strict=True)
            ),
            "python_prefix": os.path.abspath(sys.prefix),
            "python_base_prefix": os.path.abspath(sys.base_prefix),
            "numpy_module_origin": (
                None
                if importlib.util.find_spec("numpy") is None
                else importlib.util.find_spec("numpy").origin
            ),
            "resolved_package_versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pydantic", "pydantic-ai", "openai")
            },
        },
    }


def _eligibility_probe(bundle: _Bundle) -> dict[str, object]:
    # This is deliberately exercised before provider credentials or PDE work.
    # It closes a prior gap where method V11 passed preparation but failed in
    # the live-only composition block.
    _reference_parent_selector(bundle.experiment_profile)
    session = bundle.prepared.benchmark_session
    known_wall_started = time.perf_counter()
    known_cpu_started = time.process_time()
    known = tuple(
        sorted(
            bundle.benchmark.phenotype_identity.identify(
                thaw_json(seed.configuration)
            ).value_sha256
            for seed in bundle.prepared.seeds.seeds
        )
    )
    known_cpu = time.process_time() - known_cpu_started
    known_wall = time.perf_counter() - known_wall_started
    rows = []
    memory = bundle.workload_ports.evidence.initialize_memory(
        session, bundle.prepared.seeds
    )
    for seed in bundle.prepared.seeds.seeds:
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        variation = bundle.workload_ports.catalog.bind(
            session.benchmark, seed.configuration, known
        )
        bind_cpu = time.process_time() - cpu_started
        bind_wall = time.perf_counter() - wall_started
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        context = bundle.workload_ports.evidence.context(
            session, seed.configuration, variation, memory
        )
        cards = bundle.workload_ports.evidence.cards(
            session, seed.configuration, variation, memory
        )
        evidence_cpu = time.process_time() - cpu_started
        evidence_wall = time.perf_counter() - wall_started
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        replay = bundle.workload_ports.catalog.bind(
            session.benchmark, seed.configuration, known
        )
        bundle.workload_ports.evidence.context(
            session, seed.configuration, replay, memory
        )
        bundle.workload_ports.evidence.cards(
            session, seed.configuration, replay, memory
        )
        cache_cpu = time.process_time() - cpu_started
        cache_wall = time.perf_counter() - wall_started
        receipt = variation.eligibility_receipt
        assert receipt is not None
        rows.append(
            {
                "seed_id": seed.seed_id,
                "first_bind_wall_s": bind_wall,
                "first_bind_process_cpu_s": bind_cpu,
                "context_cards_wall_s": evidence_wall,
                "context_cards_process_cpu_s": evidence_cpu,
                "cached_bind_context_cards_wall_s": cache_wall,
                "cached_bind_context_cards_process_cpu_s": cache_cpu,
                "raw_option_count": len(receipt.option_phenotypes),
                "eligible_option_count": len(receipt.eligible_option_ids),
                "known_excluded_option_count": len(receipt.known_excluded_option_ids),
                "semantic_alias_count": len(receipt.alias_excluded_option_ids),
                "binding_object_reused": replay is variation,
                "context_sha256": typed_json_sha256(context),
                "card_sha256s": [typed_json_sha256(value) for value in cards],
            }
        )
    return {
        "status": "provider_and_pde_free_semantic_readiness",
        "provider_calls": 0,
        "pde_solves": 0,
        "resolution": 1001,
        "known_seed_identity_wall_s": known_wall,
        "known_seed_identity_process_cpu_s": known_cpu,
        "parents": rows,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gate_under_60_process_cpu_s_each": all(
            row["first_bind_process_cpu_s"] < 60 for row in rows
        ),
        "semantic_readiness_process_cpu_limit_s": (
            SEMANTIC_READINESS_PROCESS_CPU_LIMIT_S
        ),
        "gate_under_semantic_readiness_process_cpu_limit_each": all(
            row["first_bind_process_cpu_s"]
            < SEMANTIC_READINESS_PROCESS_CPU_LIMIT_S
            for row in rows
        ),
        "wall_under_60_s_each_observed": all(
            row["first_bind_wall_s"] < 60 for row in rows
        ),
    }


def _construction_parent(
    *,
    ordinal: int,
    configuration: FrozenJsonObject,
    benchmark: Any,
) -> EvolutionCandidate:
    """Create a clearly non-scientific parent for contract-only preparation."""

    configuration_sha256 = typed_json_sha256(configuration)
    payload = thaw_json(configuration)
    material = float(payload["material_fraction"])
    thermal = 0.00025 + ordinal * 0.00005
    raw_objectives = (
        (THERMAL_OBJECTIVE_NAME, thermal),
        (MATERIAL_OBJECTIVE_NAME, material),
    )
    resolution_policy = benchmark.objective_resolution
    evaluator = benchmark.detailed_evaluator
    if resolution_policy is None or evaluator is None:
        raise RuntimeError("construction parent requires bound Heat authorities")
    resolution = resolve_objectives(
        resolution_policy,
        ObjectiveResolutionRequest(
            configuration=configuration,
            objectives=benchmark.objectives,
            raw_objectives=raw_objectives,
        ),
    )
    detailed = DetailedEvaluation(
        phenotype=benchmark.phenotype_identity.identify(payload),
        payload=DetailedEvaluationPayload(
            failure=None,
            objectives=raw_objectives,
            violations=(),
            checks=(),
            receipt=None,
            evaluator=evaluator.evaluator_identity,
        ),
        timings=EvaluationTimings(total_wall_seconds=0.0),
    )
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId(f"candidate_heat_prepare_{ordinal}"),
            configuration_hash=configuration_sha256,
            configuration_artifact_hash=configuration_sha256,
            proposal_sequence=ordinal,
        ),
        configuration=configuration,
        objectives=resolution.decision_objectives,
        valid=True,
        generation=0,
        label=f"heat_prepare_parent_{ordinal}",
        design_rationale="provider_and_pde_free_contract_construction_only",
        detailed_evaluation=detailed,
        objective_resolution_receipt=resolution,
    )


def _synthetic_preparation_reflection_contrasts(
    *,
    source_generation: int,
    parents: tuple[EvolutionCandidate, ...],
) -> tuple[_HeatReflectionContrast, ...]:
    """Create non-scientific evidence shaped like one four-child Heat wave."""

    if source_generation not in RECOMBINATION_GENERATIONS:
        raise ValueError("source_generation is not a planned reflection source")
    if type(parents) is not tuple or len(parents) != PARENTS_PER_PORTFOLIO:
        raise ValueError("preparation reflection probe requires the exact parent count")
    parent_objectives = tuple(_object(dict(parent.objective_map)) for parent in parents)
    contrasts = []
    for child_ordinal in range(1, PORTFOLIO_WIDTH + 1):
        family = OPTION_FAMILIES[(child_ordinal - 1) % len(OPTION_FAMILIES)]
        contrasts.append(
            _HeatReflectionContrast(
                contrast_id=_sha(
                    "heat-prepare-reflection-contrast:"
                    f"g{source_generation}:child{child_ordinal}"
                ),
                wave_ordinal=((child_ordinal - 1) // RECOMBINATIONS_PER_PARENT) + 1,
                selection_role=(
                    "prepare_synthetic_elite"
                    if child_ordinal <= RECOMBINATIONS_PER_PARENT
                    else "prepare_synthetic_explorer"
                ),
                source_option_ids=(
                    f"heat2d.prepare.g{source_generation}.child{child_ordinal}",
                ),
                source_families=(family,),
                source_parent_objectives=parent_objectives,
                target_objectives=_object(
                    {
                        MATERIAL_OBJECTIVE_NAME: (
                            0.30 + source_generation / 1_000 + child_ordinal / 10_000
                        ),
                        THERMAL_OBJECTIVE_NAME: (
                            0.00020
                            + source_generation / 1_000_000
                            + child_ordinal / 10_000_000
                        ),
                    }
                ),
                reward_hex=(child_ordinal / 100).hex(),
                dominates_any_parent=child_ordinal % 2 == 0,
                better_than_any_parent=True,
            )
        )
    return tuple(contrasts)


def _provider_free_reflection_construction_probe(
    parents: tuple[EvolutionCandidate, ...],
) -> dict[str, object]:
    """Construct the sole G2 request from synthetic sealed G1 mutations."""

    if type(parents) is not tuple or len(parents) != PARENTS_PER_PORTFOLIO:
        raise ValueError("identifiable reflection probe requires both G1 parents")
    campaign_sha256 = _sha("heat-prepare-identifiable-campaign")
    workload_sha256 = _sha("heat-prepare-identifiable-workload")
    evaluator_sha256 = _sha("heat-prepare-identifiable-evaluator")
    compiler = FinitePortfolioActionSemanticsCompiler()
    adjudicator_sha256 = _sha("heat-prepare-identifiable-adjudicator")
    contrasts = []
    for ordinal, parent in enumerate(parents, start=1):
        parent_payload = thaw_json(parent.configuration)
        parent_material = float(parent_payload["material_fraction"])
        child_material = parent_material - 0.01
        contrast_id = _sha(f"heat-prepare-g1-direct-mutation:{ordinal}")
        contrasts.append(
            IdentifiableMutationReflectionContrast(
                contrast_id=contrast_id,
                source_observation_sha256=contrast_id,
                source_evidence_id=contrast_id,
                event_index=1,
                workload_instance_sha256=workload_sha256,
                evaluator_contract_sha256=evaluator_sha256,
                campaign_sha256=campaign_sha256,
                parent_candidate_id=parent.candidate_id,
                child_candidate_id=CandidateId(
                    f"candidate_heat_prepare_g1_child_{ordinal}"
                ),
                operator_invocation_id=OperatorInvocationId(
                    f"operator_heat_prepare_g1_mutation_{ordinal}"
                ),
                finite_contract_identity_sha256=_sha(
                    f"heat-prepare-g1-finite-contract:{ordinal}"
                ),
                action_semantics_compiler_id=compiler.compiler_id,
                action_semantics_compiler_version=compiler.compiler_version,
                action_semantics_definition_sha256=compiler.definition_sha256,
                option_id=f"heat2d.prepare.l00.v{ordinal:02d}",
                option_identity_sha256=_sha(f"heat-prepare-g1-option:{ordinal}"),
                option_family="material_fraction",
                affected_path="$.material_fraction",
                parent_local_value=freeze_json(parent_material),
                child_local_value=freeze_json(child_material),
                parent_configuration_sha256=typed_json_sha256(parent.configuration),
                child_configuration_sha256=_sha(
                    f"heat-prepare-g1-child-configuration:{ordinal}"
                ),
                parent_outcome_sha256=_sha(f"heat-prepare-g1-parent-outcome:{ordinal}"),
                child_outcome_sha256=_sha(f"heat-prepare-g1-child-outcome:{ordinal}"),
                metrics=tuple(
                    ObservedMetricEffect(
                        metric_id=metric_id,
                        direction=MetricEffectDirection.DECREASE,
                        delta=-0.01,
                        adjudicator_definition_sha256=adjudicator_sha256,
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
        campaign_sha256=campaign_sha256,
        workload_instance_sha256=workload_sha256,
        evaluator_contract_sha256=evaluator_sha256,
        prior_cutoff_event_index_exclusive=0,
        sealed_cutoff_event_index_inclusive=1,
        contrasts=tuple(sorted(contrasts, key=lambda value: value.contrast_id)),
        exclusions=(),
    )
    wave = CampaignReflectionWave(
        source_generation=2,
        call_count=1,
        launch_mode=ReflectionLaunchMode.ASYNC_AFTER_STAGE_SEAL,
        visibility=ReflectionVisibility.QUARANTINED_UNTIL_BLOCK_CLOSE,
        promotion_barrier_generation=4,
    )
    query = CampaignIdentifiableReflectionEvidenceQuery(
        reflection_request_sha256=_sha("heat-prepare-reflection-request"),
        preparation_sha256=_sha("heat-prepare-preparation"),
        runtime_start_receipt_sha256=_sha("heat-prepare-runtime-start"),
        campaign_sha256=campaign_sha256,
        workload_instance_sha256=workload_sha256,
        evaluator_contract_sha256=evaluator_sha256,
        wave=wave,
        source_stage_receipt_sha256=_sha("heat-prepare-g2-stage"),
        source_portfolio_generation=1,
        prior_cutoff_event_index_exclusive=0,
        sealed_cutoff_event_index_inclusive=1,
    )
    reflection_input = CampaignIdentifiableReflectionInput(
        query=query,
        source=CampaignIdentifiableReflectionEvidenceProjection(
            query_sha256=query.query_sha256,
            registry_snapshot_sha256=_sha("heat-prepare-g1-registry"),
            registry_captured_through_event_index=1,
            evidence=evidence,
        ),
    )
    semantics = _heat_optimization_semantics(
        (
            "heat_prepare_pareto",
            1,
            _sha("heat-prepare-pareto-relation"),
        )
    )
    request = build_heat2d_identifiable_reflection_request(
        call_id=DeterministicIdFactory(
            f"{AGENTIC_ID_NAMESPACE}_reflect_probe"
        ).new_llm_call_id(),
        reflection_input=reflection_input,
        optimization_semantics=semantics,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        min_insights=(2 if len(evidence.contrasts) >= 2 else 1),
        max_insights=min(8, len(evidence.contrasts)),
    )
    construction = identifiable_reflection_request_construction_record(
        request,
        evidence,
    )
    prompt = json.loads(request.prompt)
    rows = [
        {
            "source_generation": 2,
            "source_portfolio_generation": 1,
            "promotion_barrier_generation": 4,
            "first_consumer_generation": 5,
            **construction,
        }
    ]
    expected_mapping = [value.to_record() for value in request.evidence_catalog.entries]
    exact_generation_coverage = (2,) == REFLECTION_SOURCE_GENERATIONS
    exact_mapping_every_request = (
        construction["evidence_citation_mapping"] == expected_mapping
    )
    no_legacy_key_every_request = (
        "contrasts" not in prompt
        and "identifiable_mutation_contrasts" in prompt
        and "recombination" not in request.prompt.casefold()
    )
    exact_contract_every_request = (
        request.insight_contract is not None
        and construction["insight_contract_identity_sha256"]
        == request.insight_contract.identity_sha256
    )
    all_request_identities_unique = True
    all_prompt_identities_unique = True
    all_catalog_identities_unique = True
    all_acceptance_gates_pass = all(
        (
            len(rows) == 1,
            exact_generation_coverage,
            exact_mapping_every_request,
            no_legacy_key_every_request,
            exact_contract_every_request,
            all_request_identities_unique,
            all_prompt_identities_unique,
            all_catalog_identities_unique,
        )
    )
    return {
        "schema_version": 1,
        "status": "provider_pde_credential_free_reflection_construction",
        "provider_calls": 0,
        "credential_read": False,
        "pde_solves": 0,
        "scientific_values": "synthetic_contract_construction_only",
        "planned_source_generations": list(REFLECTION_SOURCE_GENERATIONS),
        "sealed_source_portfolio_generations": [1],
        "promotion_barrier_generations": list(REFLECTION_ADMISSION_GENERATIONS),
        "first_consumer_generation": FIRST_REFLECTION_CONSUMER_GENERATION,
        "terminal_reflection": False,
        "constructed_reflection_request_count": len(rows),
        "exact_generation_coverage": exact_generation_coverage,
        "exact_evidence_citation_mapping_every_request": (exact_mapping_every_request),
        "no_legacy_evidence_key_every_request": no_legacy_key_every_request,
        "exact_contract_identity_every_request": exact_contract_every_request,
        "all_request_identities_unique": all_request_identities_unique,
        "all_prompt_identities_unique": all_prompt_identities_unique,
        "all_catalog_identities_unique": all_catalog_identities_unique,
        "all_acceptance_gates_pass": all_acceptance_gates_pass,
        "rows": rows,
    }


def _calibrated_all_wave_probe(
    bundle: _Bundle,
    *,
    wave_sink: Any | None = None,
) -> dict[str, object]:
    """Construct and register every G6 portfolio wave without provider/PDE work."""

    if wave_sink is not None and not callable(wave_sink):
        raise TypeError("wave_sink must be callable or None")

    session = bundle.prepared.benchmark_session
    parents = tuple(
        _construction_parent(
            ordinal=ordinal,
            configuration=seed.configuration,
            benchmark=bundle.benchmark,
        )
        for ordinal, seed in enumerate(bundle.prepared.seeds.seeds, start=1)
    )
    reflection_construction = _provider_free_reflection_construction_probe(parents)
    known = tuple(
        sorted(
            bundle.benchmark.phenotype_identity.identify(
                thaw_json(parent.configuration)
            ).value_sha256
            for parent in parents
        )
    )
    memory_projection = bundle.workload_ports.evidence.initialize_memory(
        session,
        bundle.prepared.seeds,
    )
    archive = _object(
        {
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
            "preparation_only": True,
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
        allocator=bundle.coordinator.allocator,
        constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
        minimum_intervention_projection=MINIMUM_INTERVENTION_PROJECTION,
        evidence_calibrated_source_mix=EVIDENCE_CALIBRATED_SOURCE_MIX,
        contextual_search_allocation=CONTEXTUAL_SEARCH_ALLOCATION,
    )
    target_controller = _target_conditioned_controller(bundle, coordinator)
    probe_contextual_planner = (
        CampaignContextualSearchPlanner(
            ledger=ContextualSearchLedger(),
            campaign_scope_sha256=_sha(
                "agent-evolve:contextual-search-campaign:"
                + bundle.prepared.preparation_sha256
            ),
            joint_capability_projector=_contextual_joint_capability_projector(),
            frontier_target_allocator=(
                ResidualHypervolumeFrontierTargetAllocator()
                if RESIDUAL_FRONTIER_PLANNING
                else AuthenticatedAffineFrontierTargetAllocator()
            ),
        )
        if CONTEXTUAL_SEARCH_ALLOCATION
        else None
    )
    records: list[dict[str, object]] = []
    factory = _WaveFactory(
        ids=DeterministicIdFactory(f"{AGENTIC_ID_NAMESPACE}_prepare_probe"),
        memory=bundle.memory,
        plan=bundle.memory_plan,
        utility=bundle.utility,
        binding_factory=binding_factory,
        coordinator=coordinator,
        records=records,
        target_conditioned_controller=target_controller,
        optimization_semantics=bundle.benchmark.optimization_semantics,
    )
    rows: list[dict[str, object]] = []
    step_by_generation = {
        step.generation: step for step in bundle.prepared.schedule.steps
    }
    for generation in PORTFOLIO_GENERATIONS:
        step = step_by_generation[generation]
        utility = bundle.utility.freeze(
            benchmark=session.benchmark,
            generation=generation,
            archive=archive,
        )
        cutoff = CampaignArchiveCutoffReceipt(
            request_sha256=_sha(f"heat-prepare-cutoff-request:{generation}"),
            preparation_sha256=bundle.prepared.preparation_sha256,
            generation=generation,
            archive=archive,
            evidence=_object({"preparation_only": True, "generation": generation}),
        )
        stage_request = CampaignStageRequest(
            preparation_sha256=bundle.prepared.preparation_sha256,
            runtime_start_receipt_sha256=_sha("heat-prepare-runtime-start"),
            step=step,
            archive_cutoff=cutoff,
            archive_utility=utility,
            source_portfolio=None,
            test_eligible_reflection_receipt_sha256s=(),
            prior_selector_audit_set_sha256=_sha(
                f"heat-prepare-prior-audit:{generation}"
            ),
        )
        generation_contexts: list[CampaignPortfolioWaveContext] = []
        workload_context_sizes: dict[str, int] = {}
        for parent_slot, parent in enumerate(parents):
            variation = bundle.workload_ports.catalog.bind(
                session.benchmark,
                parent.configuration,
                known,
            )
            evidence_context = bundle.workload_ports.evidence.context(
                session,
                parent.configuration,
                variation,
                memory_projection,
            )
            workload_context_utf8_bytes = _canonical_json_size(
                thaw_json(evidence_context)
            )
            parent_measurement = bind_parent_measurement(
                candidate=parent,
                variation=variation,
                projection=bundle.parent_measurement_projection,
            )
            evidence_context = attach_parent_measurement_to_context(
                evidence_context,
                parent_measurement,
            )
            evidence_cards = bundle.workload_ports.evidence.cards(
                session,
                parent.configuration,
                variation,
                memory_projection,
            )
            context = CampaignPortfolioWaveContext(
                prepared=bundle.prepared,
                stage_request=stage_request,
                parent_slot=parent_slot,
                parent=parent,
                variation=variation,
                evidence_context=evidence_context,
                evidence_cards=evidence_cards,
                memory=memory_projection,
                parent_measurement=parent_measurement,
            )
            generation_contexts.append(context)
            workload_context_sizes[context.parent_lane.lane_id] = (
                workload_context_utf8_bytes
            )
        if target_controller is not None:
            assigned_targets = AuthenticatedAffineFrontierTargetAllocator().allocate(
                archive_utility=utility,
                lanes=tuple(
                    (value.parent_lane.lane_id, value.parent)
                    for value in generation_contexts
                ),
            )
            targets = {value.lane_id: value for value in assigned_targets}
            projector = AuthenticatedAffineFrontierContextProjector()
            rebound = []
            for value in generation_contexts:
                projection = projector.project(
                    archive_utility=utility,
                    parent=value.parent,
                )
                target = targets[value.parent_lane.lane_id]
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
            contracts = {value.slice_id: value for value in plan.contracts}
            targets = {value.lane_id: value for value in plan.frontier_targets}
            rebound = []
            for value in generation_contexts:
                lane_id = value.parent_lane.lane_id
                evidence = thaw_json(value.evidence_context)
                if type(evidence) is not dict:
                    raise TypeError("preparation context must be an object")
                evidence[CAMPAIGN_FRONTIER_TARGET_KEY] = targets[lane_id].to_record()
                rebound.append(
                    replace(
                        value,
                        evidence_context=_object(evidence),
                        contextual_allocation=contracts[lane_id],
                        frontier_target=targets[lane_id],
                    )
                )
            generation_contexts = rebound
        for context in generation_contexts:
            parent_slot = context.parent_slot
            parent_measurement = context.parent_measurement
            if parent_measurement is None:  # pragma: no cover - sealed above.
                raise AssertionError("Heat preparation context lost parent measurement")
            workload_context_utf8_bytes = workload_context_sizes[
                context.parent_lane.lane_id
            ]
            wave = factory.build(context)
            if wave_sink is not None:
                wave_sink(wave)
            memory_credit = wave.memory_credit
            if memory_credit is None:  # pragma: no cover - frozen protocol.
                raise AssertionError("Heat preparation wave omitted memory credit")
            prompt = coordinator.render(wave.selection_request)
            binding = coordinator.binding_for(wave.selection_request)
            rows.append(
                {
                    "generation": generation,
                    "lane_id": ("elite", "explorer")[parent_slot],
                    "request_sha256": wave.selection_request.request_sha256,
                    "binding_sha256": binding.binding_sha256,
                    "prompt_sha256": _sha(prompt),
                    "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                    "prompt_definition_sha256": (
                        binding.context.scope.prompt_definition_sha256
                    ),
                    "selector_policy_definition_sha256": (
                        binding.context.scope.selector_policy_definition_sha256
                    ),
                    "allocator_policy_id": bundle.coordinator.allocator.policy_id,
                    "allocator_definition_sha256": (
                        bundle.coordinator.allocator.definition_sha256
                    ),
                    "allocator_configuration_sha256": (
                        bundle.coordinator.allocator.configuration_sha256
                    ),
                    "option_prompt_projection_sha256": (
                        None
                        if binding.option_prompt_projection is None
                        else binding.option_prompt_projection.projection_sha256
                    ),
                    "option_prompt_projection_configuration_sha256": (
                        None
                        if binding.option_prompt_projection is None
                        else binding.option_prompt_projection.policy_configuration_sha256
                    ),
                    "parent_measurement_binding_sha256": (
                        parent_measurement.binding_sha256
                    ),
                    "workload_context_utf8_bytes": workload_context_utf8_bytes,
                    "selector_context_utf8_bytes": _canonical_json_size(
                        thaw_json(wave.selection_request.context)
                    ),
                    "parent_measurement_context_delta_utf8_bytes": (
                        _canonical_json_size(thaw_json(wave.selection_request.context))
                        - workload_context_utf8_bytes
                    ),
                    "eligible_option_count": len(
                        wave.selection_request.finite_variation_contract.options
                    ),
                    "proposal_width": CALIBRATED_PROPOSAL_WIDTH,
                    "evaluation_width": wave.selection_request.portfolio_size,
                    "resolved_memory_assignment_sha256": (
                        memory_credit.assignment.assignment_sha256
                    ),
                    "memory_score_snapshot_sha256": (
                        memory_credit.score_snapshot.snapshot_sha256
                    ),
                    "memory_treatment_binding_sha256": (
                        memory_credit.treatment_binding_sha256
                    ),
                }
            )
    expected_waves = len(PORTFOLIO_GENERATIONS) * PARENTS_PER_PORTFOLIO
    request_hashes = tuple(row["request_sha256"] for row in rows)
    return {
        "status": "provider_and_pde_free_all_wave_construction",
        "provider_calls": 0,
        "credential_read": False,
        "pde_solves": 0,
        "scientific_values": "synthetic_contract_construction_only",
        "portfolio_generations": list(PORTFOLIO_GENERATIONS),
        "planned_reflections": len(REFLECTION_SOURCE_GENERATIONS),
        "reflection_construction_probe": reflection_construction,
        "all_reflection_construction_gates_pass": reflection_construction[
            "all_acceptance_gates_pass"
        ],
        "constructed_wave_count": len(rows),
        "contextual_search_plans": (
            []
            if probe_contextual_planner is None
            else [value.to_record() for value in probe_contextual_planner.plans]
        ),
        "registered_request_count": coordinator.registered_request_count,
        "all_request_hashes_unique": len(set(request_hashes)) == len(request_hashes),
        "all_memory_assignments_authenticated": (
            len({row["resolved_memory_assignment_sha256"] for row in rows})
            == expected_waves
            and all(
                row["memory_score_snapshot_sha256"]
                == bundle.memory_plan.snapshot.snapshot_sha256
                for row in rows
            )
            and len({row["memory_treatment_binding_sha256"] for row in rows})
            == expected_waves
        ),
        "all_prompt_hashes_match_wave_records": all(
            row["prompt_sha256"] == record["calibrated_prompt_sha256"]
            for row, record in zip(rows, records, strict=True)
        ),
        "full_support_allocator_exact": all(
            row["allocator_policy_id"] == _default_allocator().policy_id
            and row["allocator_definition_sha256"]
            == _default_allocator().definition_sha256
            and row["allocator_configuration_sha256"]
            == _default_allocator().configuration_sha256
            for row in rows
        ),
        "active_allocator_exact": all(
            row["allocator_policy_id"] == _default_allocator().policy_id
            and row["allocator_definition_sha256"]
            == _default_allocator().definition_sha256
            and row["allocator_configuration_sha256"]
            == _default_allocator().configuration_sha256
            for row in rows
        ),
        "projected_prompt_identity_exact": all(
            row["prompt_definition_sha256"]
            == calibrated_portfolio_prompt_definition_sha256(
                _option_prompt_projection(),
                proposal_support=_proposal_support_policy() is not None,
                hierarchical_composition_required_proposals=(
                    VARIATION_TOPOLOGY.hierarchical_composition_required_proposals
                ),
                feasibility_witness_mode=FEASIBILITY_WITNESS_MODE,
                constraint_decoupled=CONSTRAINT_DECOUPLED_ACQUISITION,
            )
            and row["selector_policy_definition_sha256"]
            == _selector_policy_definition_sha256()
            and row["option_prompt_projection_sha256"] is not None
            and row["option_prompt_projection_configuration_sha256"]
            == _option_prompt_projection().configuration_sha256
            for row in rows
        ),
        "parent_measurement_bound_every_wave": all(
            row["parent_measurement_binding_sha256"] is not None for row in rows
        ),
        "exact_expected_wave_count": len(rows) == expected_waves,
        "rows": rows,
    }


async def _live(
    *,
    bundle: _Bundle,
    run_dir: Path,
    journals: dict[str, Any],
    expected_source_aggregate_sha256: str,
) -> dict[str, object]:
    load_credentials(WORKSPACE_ROOT / ".env", override=False, optional=True)
    load_credentials(AGENT_EVOLVE_ROOT / ".env", override=False, optional=True)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if type(api_key) is not str or not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")

    execution_started_ns = time.perf_counter_ns()

    def observed(value: dict[str, object]) -> dict[str, object]:
        return {
            "observation": {
                "monotonic_ns_since_execution_start": (
                    time.perf_counter_ns() - execution_started_ns
                ),
                "observed_at_utc": _utc_now(),
            },
            "authenticated_record": value,
        }

    def progress_sink(value: StructuredStreamProgress) -> None:
        journals["progress"].append(observed(_progress_record(value)))

    def outcome_sink(value: object) -> None:
        journals["progress"].flush()
        journals["outcomes"].append(
            observed(structured_generation_outcome_record(value))
        )

    def request_sink(value: dict[str, object]) -> None:
        journals["requests"].append(observed(value))

    def output_sink(value: dict[str, object]) -> None:
        journals["outputs"].append(observed(value))

    def outbound_sink(value: dict[str, object]) -> None:
        journals["outbound"].append(observed(value))

    def engine_sink(value: dict[str, object]) -> None:
        journals["engine"].append(observed(dict(value)))

    live_runner = create_progress_aware_openrouter_runner(
        api_key=api_key,
        config=_provider_config(),
        progress_sink=progress_sink,
        outcome_sink=outcome_sink,
        request_evidence_sink=request_sink,
        output_evidence_sink=output_sink,
        outbound_request_manifest_sink=outbound_sink,
        evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
    )
    replay = _load_sealed_replay_source()
    replay_contract = None if replay is None else replay[1]
    runner = (
        live_runner
        if replay is None
        else SealedReplayThenLiveStructuredRunner(
            source=replay[0],
            requested_model=MODEL,
            live_runner=live_runner,
            decision_receipt_sink=lambda value: journals["replay"].append(
                observed(value)
            ),
        )
    )
    reflection_records: list[dict[str, object]] = []
    wave_records: list[dict[str, object]] = []
    generator = PydanticAIAgenticGenerator(runner)
    selector = (
        _outcome_conditioned_selector(runner=runner, bundle=bundle)
        if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
        else _calibrated_selector(
            runner=runner,
            coordinator=bundle.coordinator,
        )
    )
    composition = compose_portfolio_evolution(
        bundle.benchmark,
        generator=generator,
        selector=selector,
        seed=OUTER_SEED,
        id_factory=bundle.ids,
        memory=bundle.memory,
        evaluator_concurrency=1,
        engine_trace_sink=engine_sink,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
    )
    parent_selector = _reference_parent_selector(bundle.experiment_profile)
    learning_bundle = _production_learning_bundle(bundle)
    target_controller = (
        None
        if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
        else _target_conditioned_controller(bundle, bundle.coordinator)
    )
    contextual_ledger = (
        ContextualSearchLedger()
        if CONTEXTUAL_SEARCH_ALLOCATION
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
            frontier_target_allocator=(
                ResidualHypervolumeFrontierTargetAllocator()
                if RESIDUAL_FRONTIER_PLANNING
                else AuthenticatedAffineFrontierTargetAllocator()
            ),
        )
    )
    diagnostic_coordinator = _HeatDiagnosticBlockCoordinator(
        memory=bundle.memory,
        seed_plan=bundle.memory_plan,
        learning_runtime=learning_bundle.runtime,
    )
    wave_factory = _WaveFactory(
        ids=bundle.ids,
        memory=bundle.memory,
        plan=bundle.memory_plan,
        utility=bundle.utility,
        binding_factory=bundle.binding_factory,
        coordinator=bundle.coordinator,
        records=wave_records,
        target_conditioned_controller=target_controller,
        optimization_semantics=bundle.benchmark.optimization_semantics,
        bounded_dose_binding_factory=bundle.bounded_dose_binding_factory,
        learning_runtime=learning_bundle.runtime,
        diagnostic_coordinator=diagnostic_coordinator,
    )
    runtime_archive_context_projector = (
        AuthenticatedAffineFrontierContextProjector()
        if target_controller is not None
        or PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
        else affine_frontier_context_projector(ARCHIVE_CONTEXT_MODE)
        if bundle.experiment_profile is None
        else bundle.experiment_profile.archive_context_projector
    )
    outcome_conditioned_scope = (
        _outcome_conditioned_calibration_scope(bundle)
        if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
        else None
    )
    calibrated_outcome_updater = CalibratedCampaignOutcomeUpdater(
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
        parent_selector=parent_selector,
        wave_factory=wave_factory,
        task_sha256=TASK_SHA256,
        parent_measurement_projection=bundle.parent_measurement_projection,
        archive_context_projector=runtime_archive_context_projector,
        memory_estimand_projector=diagnostic_coordinator,
        identifiable_reflection_executor=_ReflectionExecutor(
            generator,
            bundle.ids,
            reflection_records,
            bundle.benchmark.optimization_semantics,
        ),
        identifiable_reflection_evidence_source=(
            learning_bundle.identifiable_reflection_evidence_source
        ),
        context_enricher=ContextualOutcomeCampaignEnricher(
            ledger=bundle.feedback_ledger,
            max_actions=24,
            include_cross_lineage_analogies=False,
        ),
        contextual_search_planner=contextual_planner,
        frontier_target_allocator=(
            ResidualHypervolumeFrontierTargetAllocator()
            if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
            else AuthenticatedAffineFrontierTargetAllocator()
            if target_controller is not None
            else None
        ),
        outcome_updater=outcome_updater,
        recombination_utility_binder=_RecombinationUtilityBinder(bundle.utility),
        owned_resources=_OwnedRunner(runner),
        selector_request_prompt_renderer=(
            selector
            if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
            else bundle.coordinator
        ),
        learning_lifecycle=learning_bundle.runtime,
        wave_preparation_observer=_WavePreparationJournal(
            journals["wave_preparations"],
            execution_started_ns,
        ),
    )
    started = time.perf_counter()
    result = None
    try:
        result = await EvolutionCampaignScheduler(
            prepared=bundle.prepared,
            policies=bundle.policies,
            stages=runtime,
            reflections=runtime,
            lifecycle=runtime,
            journal=_ExecutionJournal(journals["campaign"], execution_started_ns),
        ).run()
    finally:
        await runner.aclose()
    wall = time.perf_counter() - started
    assert result is not None
    stage_counts = [value.candidate_occurrence_count for value in result.stage_receipts]
    stage_unique = [value.unique_evaluation_count for value in result.stage_receipts]
    selector_telemetry: list[dict[str, object]] = []
    selector_call_ids: list[str] = []
    stage_records = []
    for stage in result.stage_receipts:
        stage_record = thaw_json(stage.result)
        stage_records.append(stage_record)
        for receipt in stage_record.get("portfolio_wave_receipts", []):
            selector_call_ids.append(str(receipt["selection_call_id"]))
            selector_telemetry.append(dict(receipt["selection_telemetry"]))
    reflection_telemetry = [dict(value["telemetry"]) for value in reflection_records]
    reflection_call_ids = [str(value["call_id"]) for value in reflection_records]
    logical_call_ids = tuple(sorted((*selector_call_ids, *reflection_call_ids)))
    all_telemetry = [*selector_telemetry, *reflection_telemetry]
    telemetry_gate = all(
        value.get("requested_model") == MODEL
        and value.get("resolved_model")
        in MODEL_EXECUTION_PROFILE.accepted_resolved_models
        and value.get("resolved_provider")
        in MODEL_EXECUTION_PROFILE.accepted_resolved_providers
        and value.get("finish_reason")
        in MODEL_EXECUTION_PROFILE.accepted_finish_reasons
        and MODEL_EXECUTION_PROFILE.accepts_reasoning_tokens(
            value.get("reasoning_tokens")
        )
        and type(value.get("attempt_count")) is int
        and 1 <= value["attempt_count"] <= MAX_ATTEMPTS
        for value in all_telemetry
    )

    outbound_rows = read_jsonl(run_dir / "outbound_requests.jsonl")
    outbound_records = tuple(
        validate_openrouter_outbound_request_manifest_record(
            row["authenticated_record"]
        )
        for row in outbound_rows
    )
    forbidden_wire_fields_absent = all(
        all(value is True for value in record["forbidden_fields_absent"].values())
        for record in outbound_records
    )
    outbound_profile_gate = (
        bool(outbound_records)
        and forbidden_wire_fields_absent
        and all(
            record["settings"]["model"] == MODEL
            and record["settings"]["provider"]
            == {"only": list(PROVIDER_ONLY), "allow_fallbacks": False}
            and record["settings"]["reasoning"]
            == MODEL_EXECUTION_PROFILE.outbound_reasoning_setting
            and record["settings"]["max_completion_tokens"] == MAX_OUTPUT_TOKENS
            for record in outbound_records
        )
    )
    outbound_wire_gate = outbound_profile_gate and (
        PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
        or tuple(sorted({str(record["call_id"]) for record in outbound_records}))
        == logical_call_ids
    )
    durable_evidence_counts = {
        "request": len(read_jsonl(run_dir / "request_evidence.jsonl")),
        "output": len(read_jsonl(run_dir / "output_evidence.jsonl")),
        "outcome": len(read_jsonl(run_dir / "queue_outcomes.jsonl")),
        "outbound_physical_attempt": len(outbound_records),
        "wave_preparation": len(read_jsonl(run_dir / "wave_preparations.jsonl")),
    }

    spec = _affine_spec()
    reference_violations = []
    history = [_candidate_history_record(candidate) for candidate in runtime.history]
    for candidate in runtime.history:
        normalized = spec.normalize(candidate.objective_map)
        if any(value >= 1.0 for value in normalized):
            reference_violations.append(
                {
                    "candidate_id": candidate.candidate_id.value,
                    "objectives": candidate.objective_map,
                    "normalized": [value.hex() for value in normalized],
                }
            )
    initial_hypervolumes = {
        value["affine_snapshot"]["base_hypervolume_hex"]
        for value in wave_records
        if value["generation"] == 1
    }

    seed_front = [value for value in history if value["generation"] == 0]
    trajectory = [
        _archive_trajectory_record(
            label="g0_seed_cutoff",
            generation=0,
            front_candidates=seed_front,
            spec=spec,
        )
    ]
    expected_initial_hv_hex = str(trajectory[0]["normalized_hypervolume_hex"])
    for generation, stage_record in enumerate(stage_records, start=1):
        archive_after = stage_record.get("archive_after")
        if (
            type(archive_after) is not dict
            or type(archive_after.get("front_candidates")) is not list
        ):
            raise RuntimeError("completed stage omitted authenticated archive front")
        trajectory.append(
            _archive_trajectory_record(
                label=f"g{generation}_archive_after",
                generation=generation,
                front_candidates=list(archive_after["front_candidates"]),
                spec=spec,
            )
        )

    known_cost = sum(
        (
            Decimal("0")
            if value["cost_usd"] is None
            else Decimal(str(value["cost_usd"]))
            for value in all_telemetry
        ),
        Decimal("0"),
    )
    telemetry_summary = {
        "selector_call_count": len(selector_telemetry),
        "reflection_call_count": len(reflection_telemetry),
        "logical_call_ids": list(logical_call_ids),
        "input_tokens": sum(int(value["input_tokens"]) for value in all_telemetry),
        "output_tokens": sum(int(value["output_tokens"]) for value in all_telemetry),
        "reasoning_tokens": sum(
            int(value["reasoning_tokens"]) for value in all_telemetry
        ),
        "cache_read_tokens": sum(
            int(value["cache_read_tokens"]) for value in all_telemetry
        ),
        "cache_write_tokens": sum(
            int(value["cache_write_tokens"]) for value in all_telemetry
        ),
        "known_cost_usd": str(known_cost),
        "unknown_cost_call_count": sum(
            value["cost_usd"] is None for value in all_telemetry
        ),
        "latencies_ns": [int(value["latency_ns"]) for value in all_telemetry],
        "attempt_counts": [int(value["attempt_count"]) for value in all_telemetry],
        "finish_reasons": [str(value["finish_reason"]) for value in all_telemetry],
    }
    evidence_registry_snapshot = learning_bundle.evidence_registry.snapshot()
    diagnostic_wave_records = tuple(
        value
        for value in wave_records
        if value["diagnostic_reflection_exposure"] is not None
    )
    reflected_memory_entries = tuple(
        entry for entry in bundle.memory.entries if entry.evidence_lineage is not None
    )
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
    pde_evidence = _pde_evidence_record(
        run_dir,
        expected_physical_evaluations=evaluation_accounting.unique_evaluations,
    )
    postrun_source = source_identity(_source_paths(), relative_to=WORKSPACE_ROOT)
    source_closure_unchanged = (
        postrun_source["aggregate_sha256"] == expected_source_aggregate_sha256
    )
    controlled_reflection_diagnostic_available = (
        len(diagnostic_wave_records) == 2
        and {value["generation"] for value in diagnostic_wave_records} == {5}
        and len(
            {
                value["matched_memory_control"]["plan_sha256"]
                for value in diagnostic_wave_records
            }
        )
        == 1
        and {
            value["matched_memory_control"]["arm"] for value in diagnostic_wave_records
        }
        == {"m", "n"}
        and all(
            value["matched_memory_control"]["single_block_card_effect_identified"]
            is False
            and value["matched_memory_control"]["online_score_update_allowed"] is False
            for value in diagnostic_wave_records
        )
    )
    recourse_records = tuple(
        thaw_json(value) for value in wave_factory.matched_control_recourses
    )
    valid_credit_free_diagnostic_recourse = (
        len(recourse_records) == 1
        and recourse_records[0].get("generation") == 5
        and recourse_records[0].get("status") == "no_shared_support_active_neutral_card"
        and recourse_records[0].get("memory_dose_administered") is False
        and recourse_records[0].get("memory_credit_issued") is False
        and type(recourse_records[0].get("complete_support_resolution")) is dict
        and recourse_records[0]["complete_support_resolution"].get(
            "card_vs_neutral_effect_identified"
        )
        is False
        and recourse_records[0]["complete_support_resolution"].get(
            "outcome_values_consulted"
        )
        is False
        and recourse_records[0]["complete_support_resolution"].get(
            "provider_fields_consulted"
        )
        is False
    )
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
        "sealed_replay_prefix_fully_consumed": (
            replay is None or runner.remaining_entry_count == 0
        ),
        "exact_generations": (
            result.counters.generations_completed == GENERATION_COUNT
        ),
        "exact_occurrences": (
            result.counters.candidate_occurrences == PLANNED_UNIQUE_EVALUATIONS
        ),
        "exact_evaluation_accounting": True,
        "bounded_cache_reuse": evaluation_accounting.within_cache_reuse_limit(
            MAX_CACHE_REUSE_OCCURRENCES
        ),
        "exact_logical_calls": (
            result.counters.logical_agent_calls == PLANNED_LOGICAL_LLM_CALLS
        ),
        "exact_stage_occurrences": stage_counts
        == [
            PARENTS_PER_PORTFOLIO * PORTFOLIO_WIDTH,
            PARENTS_PER_PORTFOLIO * RECOMBINATIONS_PER_PARENT,
            PARENTS_PER_PORTFOLIO * PORTFOLIO_WIDTH,
            PARENTS_PER_PORTFOLIO * RECOMBINATIONS_PER_PARENT,
            PARENTS_PER_PORTFOLIO * PORTFOLIO_WIDTH,
            PARENTS_PER_PORTFOLIO * RECOMBINATIONS_PER_PARENT,
        ],
        "exact_associational_memory_trials": len(bundle.memory.trials) == 4,
        "one_delayed_identifiable_reflection": len(reflection_records) == 1,
        "production_reflection_registration_complete": len(reflected_memory_entries)
        == sum(len(value["insights"]) for value in reflection_records),
        "g5_reflection_diagnostic_protocol_resolved": (
            controlled_reflection_diagnostic_available
            or valid_credit_free_diagnostic_recourse
        ),
        "production_action_evidence_registry_complete": (
            len(evidence_registry_snapshot.observations)
            == len(PORTFOLIO_GENERATIONS) * PARENTS_PER_PORTFOLIO * PORTFOLIO_WIDTH
            and evidence_registry_snapshot.captured_through_event_index == 5
            and {value.event_index for value in evidence_registry_snapshot.observations}
            == {1, 3, 5}
        ),
        "nonempty_final_front": bool(runtime.final_front),
        "cleanup_released": result.cleanup_receipt.released,
        "seven_exact_model_provider_reasoning_receipts": (
            len(selector_telemetry) == 6
            and len(reflection_telemetry) == 1
            and telemetry_gate
        ),
        "outbound_wire_matches_model_profile": outbound_wire_gate,
        "durable_logical_request_output_outcome_counts": (
            (
                durable_evidence_counts["request"] >= PLANNED_LOGICAL_LLM_CALLS
                and durable_evidence_counts["output"] >= PLANNED_LOGICAL_LLM_CALLS
                and durable_evidence_counts["outcome"] >= PLANNED_LOGICAL_LLM_CALLS
                if PORTFOLIO_SELECTOR_MODE == "outcome_conditioned"
                else durable_evidence_counts["request"]
                == PLANNED_LOGICAL_LLM_CALLS
                and durable_evidence_counts["output"]
                == PLANNED_LOGICAL_LLM_CALLS
                and durable_evidence_counts["outcome"]
                == PLANNED_LOGICAL_LLM_CALLS
            )
            and durable_evidence_counts["outbound_physical_attempt"]
            >= PLANNED_LOGICAL_LLM_CALLS
            and durable_evidence_counts["wave_preparation"]
            == len(PORTFOLIO_GENERATIONS) * PARENTS_PER_PORTFOLIO
        ),
        "affine_reference_contains_every_candidate": not reference_violations,
        "qualified_initial_hypervolume_reproduced": initial_hypervolumes
        == {expected_initial_hv_hex},
        "archive_aware_recombination_enabled": all(
            stage_records[index].get("archive_aware_source_utility") is True
            for index in (1, 3, 5)
        ),
        "calibrated_k8_to_k8_registration_complete": (
            bundle.coordinator.registered_request_count == 6
            and all(
                record["proposal_width"] == CALIBRATED_PROPOSAL_WIDTH
                and record["evaluation_width"] == PORTFOLIO_WIDTH
                and record["calibrated_prompt_utf8_bytes"] > 0
                for record in wave_records
            )
        ),
        "projected_prompt_composition_exact": all(
            _calibrated_wave_prompt_composition_exact(record) for record in wave_records
        ),
        "parent_measurement_bound_every_selector_wave": all(
            record["parent_measurement_binding_sha256"] is not None
            for record in wave_records
        ),
        "card_blind_forecast_feedback_complete": (
            len(bundle.feedback_ledger.receipts) == 6
            and len(bundle.feedback_ledger.observations)
            == len(evidence_registry_snapshot.observations) * len(OBJECTIVE_IDS)
        ),
        "contextual_search_closed_loop": (
            not CONTEXTUAL_SEARCH_ALLOCATION
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
        "all_candidates_use_fixed_grid_objectives": all(
            candidate.objective_resolution_receipt is not None
            for candidate in runtime.history
        ),
        "all_candidates_preserve_detailed_evaluator_evidence": all(
            candidate.detailed_evaluation is not None for candidate in runtime.history
        ),
        "physical_direct_v3_manifests_match_accounting": pde_evidence[
            "manifest_count_matches_physical_evaluations"
        ],
        "all_direct_v3_scientific_contracts_pass": pde_evidence[
            "all_scientific_contracts_pass"
        ],
        "all_pde_evaluations_under_45_s_and_3_gib": pde_evidence[
            "all_under_45_s_and_3_gib"
        ],
        "source_closure_unchanged": source_closure_unchanged,
    }
    health_pass = all(health.values())
    status = "completed_healthy" if health_pass else "completed_unhealthy"
    return {
        "schema_version": 1,
        "status": status,
        "health_pass": health_pass,
        "campaign_result": result.to_record(),
        "counters": result.counters.to_record(),
        "stage_occurrence_counts": stage_counts,
        "stage_unique_evaluation_counts": stage_unique,
        "evaluation_accounting": evaluation_accounting.to_record(),
        "wall_s": wall,
        "target_conditioned_state": (
            None
            if target_controller is None
            else target_controller.state.to_record()
        ),
        "health": health,
        "scientific_diagnostics": {
            "g5_controlled_reflection_identification_available": (
                controlled_reflection_diagnostic_available
            ),
            "g5_credit_free_recourse_used": (valid_credit_free_diagnostic_recourse),
            "g5_card_effect_identified": False,
        },
        "provider_telemetry": telemetry_summary,
        "durable_evidence_counts": durable_evidence_counts,
        "sealed_accepted_output_replay": {
            "source": replay_contract,
            "decision_count": len(read_jsonl(run_dir / "replay_decisions.jsonl")),
            "remaining_entry_count": (
                None if replay is None else runner.remaining_entry_count
            ),
        },
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
        "reference_violations": reference_violations,
        "expected_initial_hypervolume_hex": expected_initial_hv_hex,
        "archive_hypervolume_trajectory": trajectory,
        "final_normalized_hypervolume_hex": trajectory[-1][
            "normalized_hypervolume_hex"
        ],
        "final_raw_oriented_hypervolume_hex": trajectory[-1][
            "raw_oriented_hypervolume_hex"
        ],
        "candidate_history": history,
        "final_front": [
            _candidate_history_record(value) for value in runtime.final_front
        ],
        "pde_evidence": pde_evidence,
        "source_closure": {
            "launch_aggregate_sha256": expected_source_aggregate_sha256,
            "postrun_aggregate_sha256": postrun_source["aggregate_sha256"],
            "unchanged": source_closure_unchanged,
            "postrun_file_count": postrun_source["file_count"],
        },
        "memory": {
            "trial_count": len(bundle.memory.trials),
            "plan": bundle.memory_plan.to_record(),
            "diagnostic_block_plans": [
                value.to_record()
                for _, value in sorted(wave_factory._diagnostic_blocks.items())
            ],
            "matched_support_resolutions": [
                value.to_record() for value in wave_factory.matched_support_resolutions
            ],
            "matched_control_plans": [
                value.to_record() for value in wave_factory.matched_control_plans
            ],
            "matched_control_recourses": [
                thaw_json(value) for value in wave_factory.matched_control_recourses
            ],
            "reflected_entries": [
                value.to_record() for value in reflected_memory_entries
            ],
            "action_evidence_registry": evidence_registry_snapshot.to_record(),
            "score_evidence_postrun_diagnostic_only": list(
                bundle.memory.score_evidence(MEMORY_CONTEXT_SHA256)
            ),
            "adaptive_score_consumption": False,
            "causal_claim_allowed": False,
        },
        "forecast_feedback": {
            "receipt_count": len(bundle.feedback_ledger.receipts),
            "observation_count": len(bundle.feedback_ledger.observations),
            "prompt_history": thaw_json(
                bundle.feedback_ledger.prompt_history(
                    cutoff_wave_index_exclusive=GENERATION_COUNT + 1
                )
            ),
        },
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
        "wave_preparation_records": wave_records,
    }


def _readiness_contract_projection(
    readiness: dict[str, object],
) -> dict[str, object]:
    """Remove machine-timing observations while retaining the bounded gate."""

    parents = readiness.get("parents")
    if type(parents) is not list or any(type(value) is not dict for value in parents):
        raise TypeError("Heat readiness parents must be an object list")
    return {
        "status": readiness.get("status"),
        "resolution": readiness.get("resolution"),
        "provider_calls": readiness.get("provider_calls"),
        "pde_solves": readiness.get("pde_solves"),
        "semantic_readiness_process_cpu_limit_s": readiness.get(
            "semantic_readiness_process_cpu_limit_s"
        ),
        "gate_under_semantic_readiness_process_cpu_limit_each": readiness.get(
            "gate_under_semantic_readiness_process_cpu_limit_each"
        ),
        "parents": [
            {
                key: value.get(key)
                for key in (
                    "seed_id",
                    "binding_object_reused",
                    "raw_option_count",
                    "eligible_option_count",
                    "known_excluded_option_count",
                    "semantic_alias_count",
                    "context_sha256",
                    "card_sha256s",
                )
            }
            for value in parents
        ],
    }


def _preregistration_contract(
    *,
    bundle: _Bundle,
    source_aggregate_sha256: str,
    readiness: dict[str, object],
    calibrated_probe: dict[str, object],
) -> dict[str, object]:
    """Freeze the exact provider-bound Heat treatment before credential use."""

    if type(source_aggregate_sha256) is not str or len(source_aggregate_sha256) != 64:
        raise ValueError("source_aggregate_sha256 must be a SHA-256 hex digest")
    profile = bundle.experiment_profile
    replay = _load_sealed_replay_source()
    return {
        "schema_version": 1,
        "experiment_id": (
            PROTOCOL_ID
            if profile is None
            else f"{profile.method_id}_{MODEL_EXECUTION_PROFILE.profile_id}"
        ),
        "source_aggregate_sha256": source_aggregate_sha256,
        "preparation_sha256": bundle.prepared.preparation_sha256,
        "protocol_sha256": bundle.prepared.protocol.protocol_sha256,
        "model_execution_profile_sha256": MODEL_EXECUTION_PROFILE.profile_sha256,
        "method_definition_sha256": (
            None if profile is None else profile.method_definition_sha256
        ),
        "experiment_definition_sha256": (
            None if profile is None else profile.experiment_definition_sha256
        ),
        "experiment_profile_conformance": (
            None
            if profile is None
            else profile.prepared_conformance_record(
                prepared=bundle.prepared,
                archive_utility=bundle.utility,
                outer_seed=OUTER_SEED,
            )
        ),
        "readiness_contract": _readiness_contract_projection(readiness),
        "construction_probe_sha256": typed_json_sha256(freeze_json(calibrated_probe)),
        "outer_seed": OUTER_SEED,
        "planned_unique_evaluations": PLANNED_UNIQUE_EVALUATIONS,
        "planned_logical_llm_calls": PLANNED_LOGICAL_LLM_CALLS,
        "candidate_pool_mode": (
            "complete_finite_contract"
            if COMMON_POOL_ACQUISITION and COMMON_CANDIDATE_POOL_SIZE is None
            else "legacy_or_fixed"
        ),
        "model_selection_size": CALIBRATED_PROPOSAL_WIDTH,
        "engine_evaluation_width": PORTFOLIO_WIDTH,
        "sealed_accepted_output_replay": (
            None if replay is None else replay[1]
        ),
    }


def _validate_preregistration(
    *,
    path: Path,
    bundle: _Bundle,
    source_aggregate_sha256: str,
    readiness: dict[str, object],
    calibrated_probe: dict[str, object],
) -> dict[str, object]:
    prereg = path.expanduser().resolve(strict=True)
    if WORKSPACE_ROOT not in prereg.parents:
        raise RuntimeError("preregistration must live inside the workspace")
    observed = json.loads(prereg.read_text(encoding="utf-8"))
    expected = _preregistration_contract(
        bundle=bundle,
        source_aggregate_sha256=source_aggregate_sha256,
        readiness=readiness,
        calibrated_probe=calibrated_probe,
    )
    if observed != expected:
        raise RuntimeError("preregistration differs from the prepared Heat contract")
    return {
        "path": prereg.relative_to(WORKSPACE_ROOT).as_posix(),
        "sha256": hashlib.sha256(prereg.read_bytes()).hexdigest(),
        "size_bytes": prereg.stat().st_size,
        "validated_exact_contract": True,
    }


def _record_launch(
    args: argparse.Namespace,
    run_dir: Path,
    source_paths: tuple[Path, ...],
    source: dict[str, object],
) -> None:
    """Publish the complete launch environment beside the source closure.

    Called twice: once as soon as the run directory exists, so that even a
    failed preparation records how it was launched, and once again just
    before finalization, when the observed ambient-input set is complete.
    Never raises; a failure is journaled as ``launch_record_error.json``.
    """

    record_campaign_launch(
        mode=args.mode,
        run_id=args.run_id,
        run_dir=run_dir,
        workspace_root=WORKSPACE_ROOT,
        agent_evolve_root=AGENT_EVOLVE_ROOT,
        source_paths=source_paths,
        source_closure=source,
        dotenv_paths=(WORKSPACE_ROOT / ".env", AGENT_EVOLVE_ROOT / ".env"),
    )


async def _main_async(args: argparse.Namespace) -> int:
    run_dir = (ARTIFACT_ROOT / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    preparation: DurableJsonlJournal | None = None
    try:
        source_paths = _source_paths()
        source = source_identity(source_paths, relative_to=WORKSPACE_ROOT)
        source_snapshot = _snapshot_sources(run_dir, source_paths)
        write_json_atomic(
            run_dir / "manifest.json",
            _manifest(args.run_id, args.mode, source, source_snapshot),
        )
        _record_launch(args, run_dir, source_paths, source)
        preparation = DurableJsonlJournal(run_dir / "preparation.jsonl")
        bundle = _prepare_bundle(
            run_dir=run_dir,
            run_id=args.run_id,
            preparation_journal=preparation,
            source_closure_sha256=str(source["aggregate_sha256"]),
        )
        readiness = _eligibility_probe(bundle)
        write_json_atomic(run_dir / "readiness.json", readiness)
        if not readiness[
            "gate_under_semantic_readiness_process_cpu_limit_each"
        ]:
            raise RuntimeError(
                "semantic binding readiness exceeded the preregistered "
                f"{SEMANTIC_READINESS_PROCESS_CPU_LIMIT_S:g} process-CPU-second "
                "ceiling"
            )
        calibrated_probe = _calibrated_all_wave_probe(bundle)
        write_json_atomic(run_dir / "calibrated_all_wave_probe.json", calibrated_probe)
        if not (
            calibrated_probe["exact_expected_wave_count"]
            and calibrated_probe["all_request_hashes_unique"]
            and calibrated_probe["all_memory_assignments_authenticated"]
            and calibrated_probe["all_prompt_hashes_match_wave_records"]
            and calibrated_probe["active_allocator_exact"]
            and calibrated_probe["projected_prompt_identity_exact"]
            and calibrated_probe["parent_measurement_bound_every_wave"]
            and calibrated_probe["all_reflection_construction_gates_pass"]
            and calibrated_probe["registered_request_count"] == 6
        ):
            raise RuntimeError("calibrated G6 all-wave preparation gate failed")
        if args.mode == "prepare":
            preregistration = _preregistration_contract(
                bundle=bundle,
                source_aggregate_sha256=str(source["aggregate_sha256"]),
                readiness=readiness,
                calibrated_probe=calibrated_probe,
            )
            write_json_atomic(
                run_dir / "preregistration_template.json", preregistration
            )
            summary = {
                "schema_version": 1,
                "status": "prepared_provider_and_pde_free",
                "provider_calls": 0,
                "credential_read": False,
                "pde_solves": 0,
                "preparation": bundle.prepared.to_record(),
                "memory_plan": bundle.memory_plan.to_record(),
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
                "readiness": readiness,
                "calibrated_all_wave_probe": calibrated_probe,
                "preregistration_template": preregistration,
                "source_snapshot": source_snapshot,
            }
            write_json_atomic(run_dir / "summary.json", summary)
            _record_launch(args, run_dir, source_paths, source)
            preparation.close()
            finalize_run_directory(run_dir, status=summary["status"])
            print(json.dumps(summary, sort_keys=True))
            return 0

        if args.prereg is None:
            raise RuntimeError("live mode requires --prereg")
        prereg_identity = _validate_preregistration(
            path=Path(args.prereg),
            bundle=bundle,
            source_aggregate_sha256=str(source["aggregate_sha256"]),
            readiness=readiness,
            calibrated_probe=calibrated_probe,
        )
        write_json_atomic(
            run_dir / "preregistration_identity.json",
            prereg_identity,
        )
        journals: dict[str, Any] = {
            "engine": DurableJsonlJournal(run_dir / "engine_events.jsonl"),
            "campaign": DurableJsonlJournal(run_dir / "campaign_events.jsonl"),
            "requests": DurableJsonlJournal(run_dir / "request_evidence.jsonl"),
            "outputs": DurableJsonlJournal(run_dir / "output_evidence.jsonl"),
            "outcomes": DurableJsonlJournal(run_dir / "queue_outcomes.jsonl"),
            "outbound": DurableJsonlJournal(run_dir / "outbound_requests.jsonl"),
            "replay": DurableJsonlJournal(run_dir / "replay_decisions.jsonl"),
            "progress": BatchedDurableJsonlJournal(
                run_dir / "stream_progress.jsonl", max_unfsynced_rows=32
            ),
            "wave_preparations": DurableJsonlJournal(
                run_dir / "wave_preparations.jsonl"
            ),
        }
        try:
            summary = await _live(
                bundle=bundle,
                run_dir=run_dir,
                journals=journals,
                expected_source_aggregate_sha256=str(source["aggregate_sha256"]),
            )
            write_json_atomic(run_dir / "summary.json", summary)
        except BaseException as exc:
            failure = {
                "schema_version": 1,
                "status": "failed",
                "failure_type": type(exc).__name__,
                "failure_digest_sha256": hashlib.sha256(
                    type(exc).__qualname__.encode("utf-8")
                    + b"\x00"
                    + str(exc).encode("utf-8", errors="replace")
                ).hexdigest(),
            }
            write_json_atomic(run_dir / "summary.json", failure)
            raise
        finally:
            for journal in journals.values():
                journal.close()
        preparation.close()
        finalize_run_directory(run_dir, status=str(summary["status"]))
        print(json.dumps(summary, sort_keys=True))
        return 0
    except BaseException:
        if preparation is not None:
            preparation.close()
        if not (run_dir / "summary.json").exists():
            write_json_atomic(
                run_dir / "summary.json",
                {"schema_version": 1, "status": "failed_before_live_execution"},
            )
        if not (run_dir / "finalized.json").exists():
            finalize_run_directory(run_dir, status="failed")
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "live"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prereg")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_main_async(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
