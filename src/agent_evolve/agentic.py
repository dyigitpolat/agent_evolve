"""Stable, benchmark-neutral composition surface for agentic evolution.

Benchmark packages should depend on this module, not on the internal
``application`` package.  A benchmark supplies domain semantics through the
frozen :class:`AgenticBenchmark` bundle; :func:`compose_agentic_optimizer`
wires those semantics into the evolution engine and archive exactly once.

The façade deliberately keeps provider construction outside its boundary.  A
caller injects any :class:`AgenticGenerator`, planner (or deferred planner
factory), ID factory, and memory bank.  Runtime-bound feedback can likewise be
constructed through a deferred factory.  This keeps benchmark adapters
independent of Pydantic AI, OpenRouter, or any future model runtime while still
guaranteeing that stateful policies receive the composition root's exact engine,
IDs, and memory objects.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    CrossoverResponseMode,
    EvolutionCandidate,
    InvocationOutcome,
    InvocationPlan,
    MutationContract,
    MutationResponseMode,
    OperatorKind,
    REWARD_DEFINITION_HASH,
    ReflectionCallExecutionError,
    ReflectionCallReceipt,
    ReflectionCallRequest,
    ReflectionCallStatus,
    ReflectionPublication,
    ReflectionPublicationResult,
    ReflectionRowProjectionBinding,
    RewardPolicyBinding,
    default_evidence_prompt,
    default_parent_relative_reward,
)
from agent_evolve.application.budgeted_optimizer import (
    BudgetedAgenticOptimizer,
    FrozenWaveReward,
    GenerationPlan,
    GenerationPlanner,
    OptimizerBudget,
    OptimizerBudgetExceeded,
    OptimizerContractError,
    OptimizerExecutionError,
    OptimizerPlanningError,
    OptimizerResult,
    OptimizerSlot,
    OptimizerState,
    SeedAdmissionPolicy,
)
from agent_evolve.application.detailed_evaluation import (
    DetailedEvaluation,
    DetailedEvaluationAdapter,
    DetailedEvaluationPayload,
    EvaluationCheck,
    EvaluationCheckStatus,
    EvaluationTimings,
    EvaluatorIdentity,
)
from agent_evolve.application.effective_choice_audit import (
    EFFECTIVE_CHOICE_AUDIT_DEFINITION_SHA256,
    EFFECTIVE_CHOICE_AUDIT_POLICY_ID,
    EFFECTIVE_CHOICE_AUDIT_POLICY_VERSION,
    EffectiveChoiceAuditError,
    EffectiveChoiceAuditReceipt,
    SelectedCardBindingMode,
    audit_effective_choice_plan,
    validate_effective_choice_audit_receipt,
)
from agent_evolve.application.generation_feedback import (
    GenerationFeedbackContext,
    GenerationFeedbackInterceptor,
    GenerationFeedbackReceipt,
    GenerationFeedbackReservation,
    GenerationFeedbackResult,
    generation_feedback_receipt_hash,
    seal_generation_feedback,
    validate_generation_feedback_receipt,
)
from agent_evolve.application.g3_causal_screen import (
    G1_DIAGNOSTIC_SLOT_IDS,
    G2_SLOT_IDS,
    G3_SLOT_IDS,
    G3BenchmarkBoundary,
    G3CausalScreenPlanner,
    G3ExpectedEndpoint,
    G3ExpectedUnion,
    G3TerminalValidationAuthority,
    G3_SCREEN_BUDGET,
    FrozenDiagnosticPermutation,
    ParentBoundActionChoice,
    PreparedHypothesisMatrix,
)
from agent_evolve.application.g3_causal_validation import (
    G3CausalScreenResultValidationReceipt,
    G3MechanismDecision,
    G3TerminalStateValidationReceipt,
    G3TerminalValidationError,
    validate_g3_causal_screen_result,
    validate_g3_terminal_state,
)
from agent_evolve.application.g3_postseal_curation import (
    G3CurationSourceScope,
    G3PostsealCurationAuthority,
    G3PostsealCurationFactory,
    G3PostsealCurationInterceptor,
    G3PostsealCurationReceipt,
    G3PostsealCurationSpec,
    G3_POSTSEAL_CURATION_DEFINITION_SHA256,
    build_g3_postseal_curation_reservation,
)
from agent_evolve.application.action_allocation import (
    GREEDY_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256,
    GREEDY_RISK_DIVERSITY_ALLOCATOR_ID,
    GREEDY_RISK_DIVERSITY_ALLOCATOR_VERSION,
    GreedyRiskAdjustedDiversityAllocator,
)
from agent_evolve.application.action_allocation_frame import (
    AllocationSurfaceGateRejected,
    AuditedGreedyForecastFrameAllocator,
)
from agent_evolve.application.action_allocation_frame_commit import (
    FrameActionAllocationCommitRejected,
    build_frame_action_allocation_phase_commit,
    validate_frame_action_allocation_phase_commit,
)
from agent_evolve.application.derived_action_semantics import (
    derive_action_space_semantics,
)
from agent_evolve.application.action_forecast_partitioning import (
    ActionForecastBlockHealthSubsetBinding,
    ActionForecastHealthFrameKind,
    ActionForecastHealthPolicyBinding,
    ActionForecastHealthSubsetPolicyBinding,
    ActionForecastWaveError,
    ConcurrentActionForecastWave,
    ResolvedActionForecastHealthAssessment,
    assess_resolved_action_forecast_block_health,
    assess_resolved_action_forecast_block_subset_health,
    assess_resolved_action_forecast_health,
    build_action_forecast_block_requests,
    build_action_forecast_partition_layout,
    lenient_action_forecast_health_policy,
)
from agent_evolve.application.two_stage_action_evolution import (
    ACTION_EVALUATION_REUSE_POLICY_DEFINITION_SHA256,
    ACTION_EVALUATION_REUSE_POLICY_ID,
    ACTION_EVALUATION_REUSE_POLICY_VERSION,
    DURABLE_PHASE_COMMIT_POLICY_DEFINITION_SHA256,
    DURABLE_PHASE_COMMIT_POLICY_ID,
    DURABLE_PHASE_COMMIT_POLICY_VERSION,
    ActionEvaluationReuseMode,
    ActionEvaluationReusePolicyBinding,
    ActionAllocationArmExecution,
    ActionForecastArmExecution,
    ActionForecastArmPlan,
    DurablePhaseCommitPolicyBinding,
    DurablePhaseCommitRequirement,
    FiniteActionEvaluationRequest,
    FiniteActionEvaluationResult,
    FiniteActionEvaluator,
    FiniteActionEvaluatorBinding,
    PreparedTwoStageActionEvolution,
    PreparedTwoStageActionEvolutionRequest,
    PreparedTwoStageActionEvolutionResult,
    SCIENTIFIC_ARM_ORDER,
    TwoStageActionPhase,
    TwoStageActionPhaseCommit,
    TwoStageActionPhaseCommitError,
    TwoStageActionPhaseCommitSink,
    TwoStageActionPhaseReceipt,
    optional_phase_commit_policy,
    per_arm_evaluation_reuse_policy,
    required_scientific_phase_commit_policy,
)
from agent_evolve.application.gated_agentic_generator import (
    AgenticTelemetryPolicy,
    TelemetryGatedAgenticGenerator,
)
from agent_evolve.application.insight_memory import (
    InsightLifecycleChangeRequest,
    InsightLifecycleState,
    InsightMemoryEntry,
    InsightEvidenceLineage,
    InsightMemoryBank,
    InsightOrigin,
    InsightRelation,
    InsightRelationKind,
    QuarantineTestAdmissionReceipt,
)
from agent_evolve.application.materialized_variation import (
    materialized_disjoint_invocation,
    materialized_finite_action_decision,
)
from agent_evolve.application.matched_finite_action_block import (
    MatchedFiniteActionBenchmark,
    MatchedFiniteActionBlockFactory,
    MatchedFiniteActionBlockPlanner,
    finite_action_mutation_boundary,
)
from agent_evolve.application.multi_option_evolution import (
    AdaptiveShuffledMateCrossoverPolicy,
    G3CrossoverPlanPolicy,
    MULTI_OPTION_EVOLUTION_BUDGET,
    MULTI_OPTION_G1_SLOT_IDS,
    MULTI_OPTION_G2_SLOT_IDS,
    MULTI_OPTION_G3_CORE_SLOT_IDS,
    MULTI_OPTION_G3_CROSSOVER_SLOT_IDS,
    MULTI_OPTION_G3_SLOT_IDS,
    MULTI_OPTION_G3_UNION_SOURCES,
    MultiOptionEvolutionBenchmark,
    MultiOptionEvolutionPlanner,
    MultiOptionEvolutionPlannerFactory,
    OrderedTwoSeedRolePolicy,
    ParentBoundFiniteChoice,
    SeedRolePolicy,
    SeedRoleSelection,
)
from agent_evolve.policies.variation.exact_parent_crossover import (
    DEFAULT_MAX_EXACT_PARENT_CROSSOVER_LOCI,
    MAX_EXACT_PARENT_CROSSOVER_LOCI,
    MIN_EXACT_PARENT_CROSSOVER_LOCI,
    ExactParentCrossoverContract,
    ExactParentCrossoverLocus,
    ExactParentCrossoverMaterialization,
    ExactParentCrossoverReceipt,
    ExactParentImportPlan,
    ExactParentLocusAttribution,
    ExactParentSource,
    build_exact_parent_import_plan,
    canonical_candidate_path_text,
    derive_exact_parent_crossover_contract,
    exact_parent_import_exclusions_sha256,
    materialize_exact_parent_crossover,
    replay_exact_parent_crossover,
    resolve_exact_parent_import_for_target,
    validate_exact_parent_import_exclusions,
)
from agent_evolve.policies.variation.compositional_finite_catalog import (
    COMPOSITION_LEFT_OPTION_METADATA_KEY,
    COMPOSITION_REQUIRED_PROPOSALS_METADATA_KEY,
    COMPOSITION_RIGHT_OPTION_METADATA_KEY,
    COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY,
    COMPOSITIONAL_FINITE_CATALOG_POLICY_ID,
    COMPOSITIONAL_FINITE_CATALOG_POLICY_VERSION,
    COMPOSITE_OPTION_FAMILY,
    HIERARCHICAL_COMPOSITIONAL_FINITE_CATALOG_POLICY_ID,
    HIERARCHICAL_COMPOSITIONAL_FINITE_CATALOG_POLICY_VERSION,
    BoundedCompositionalFiniteVariationCatalog,
    CompositionSelectionExposure,
)
from agent_evolve.policies.variation.source_union_finite_catalog import (
    EVALUATION_SOURCE_METADATA_KEY,
    EVALUATION_SOURCE_MINIMUM_METADATA_KEY,
    SOURCE_UNION_POLICY_ID,
    SOURCE_UNION_POLICY_VERSION,
    SourceUnionFiniteVariationCatalog,
    required_source_evaluation_option_ids,
)
from agent_evolve.application.executable_hypothesis import (
    CompiledHypothesisTreatment,
    compile_registered_hypothesis_treatment,
    registered_source_evidence_sha256,
)
from agent_evolve.application.finite_action_set import (
    compile_and_seal_finite_action_set,
)
from agent_evolve.application.finite_action_selection import (
    MODEL_FINITE_ACTION_SELECTOR_DEFINITION_SHA256,
    MODEL_FINITE_ACTION_SELECTOR_POLICY_ID,
    MODEL_FINITE_ACTION_SELECTOR_POLICY_VERSION,
    model_finite_action_telemetry_sha256,
    seal_model_finite_action_decision,
)
from agent_evolve.application.outcome_relation import (
    ObjectiveParetoOutcomePolicy,
    OutcomeComparator,
    OutcomeRelation,
    OutcomeRelationPolicyBinding,
    objective_pareto_outcome_binding,
)
from agent_evolve.application.finite_variation_eligibility import (
    FINITE_VARIATION_ELIGIBILITY_POLICY_DEFINITION_SHA256,
    FINITE_VARIATION_ELIGIBILITY_POLICY_ID,
    FINITE_VARIATION_ELIGIBILITY_POLICY_VERSION,
    EligibleFiniteVariationView,
    FiniteVariationEligibilityReceipt,
    OptionPhenotypeBinding,
    eligible_finite_variation_view,
    exact_configuration_phenotype_bindings,
)
from agent_evolve.application.portfolio_evolution import (
    ProviderTrafficWitness,
    EXACT_MEMORY_CONTEXT_PROJECTION_DEFINITION_SHA256,
    PORTFOLIO_MATERIALIZATION_POLICY_ID,
    PORTFOLIO_MATERIALIZATION_POLICY_VERSION,
    MaterializedPortfolioEngine,
    PortfolioCandidateFailureEvidence,
    PortfolioEvolution,
    PortfolioMemberMaterializationReceipt,
    PortfolioMemberDisposition,
    PortfolioMemoryCreditBatchPreparation,
    PortfolioMemoryCreditBatchReceipt,
    PortfolioMemoryCreditPlan,
    PortfolioMemoryCreditReceipt,
    PortfolioMemoryContextProjectionBinding,
    PortfolioMemoryMatchedControlWavePlan,
    PortfolioPendingMemoryCredit,
    PortfolioRewardAggregationBinding,
    PortfolioVariationMemberReceipt,
    PortfolioVariationWaveReceipt,
    PortfolioVariationWaveRequest,
    PortfolioVariationWaveResult,
    portfolio_selection_telemetry_sha256,
)
from agent_evolve.application.portfolio_recombination import (
    ArchiveAwareDisjointPairSelectionDecision,
    ArchiveAwareDisjointParentPairPolicy,
    FrozenArchiveBranchUtility,
    FrozenArchiveSourcePairUtility,
    FrozenArchiveSourceUtilityContext,
    FrozenArchiveSourceUtilityReceipt,
    ObservedSourceBranch,
    PortfolioPairAttemptReceipt,
    PortfolioRecombination,
    PortfolioRecombinationBranchBinding,
    PortfolioRecombinationMemberReceipt,
    PortfolioRecombinationNoPairReason,
    PortfolioRecombinationNoPairReceipt,
    PortfolioRecombinationSourceExclusionReason,
    PortfolioRecombinationSourceExclusionReceipt,
    PortfolioRecombinationWaveReceipt,
    PortfolioRecombinationWaveRequest,
    PortfolioRecombinationWaveResult,
    bind_portfolio_recombination_source_utilities,
    frozen_archive_source_utility_context,
    portfolio_recombination_observed_sources,
)
from agent_evolve.application.portfolio_projection import (
    admit_portfolio_card_sources,
    bind_portfolio_experimental_view,
    portfolio_card_from_insight_entry,
    project_action_neutral_insight_prompt_payload,
)
from agent_evolve.application.pareto_archive import (
    EvidenceAdmissionPolicy,
    ParetoArchive,
)
from agent_evolve.application.post_evolution_reflection import (
    POST_EVOLUTION_REFLECTION_DEFINITION_SHA256,
    PostEvolutionPredecessorResolver,
    PostEvolutionReflectionAuthority,
    PostEvolutionReflectionFactory,
    PostEvolutionReflectionInterceptor,
    PostEvolutionReflectionReceipt,
    PostEvolutionReflectionSource,
    PostEvolutionReflectionSourceScope,
    PostEvolutionReflectionSpec,
)
from agent_evolve.application.reflection_workflow import (
    ContrastShardedReflectionWorkflow,
    PlannedReflectionBatchCall,
    ReflectionPromptShard,
    ReflectionWorkflow,
    ReflectionWorkflowExecutionError,
    ReflectionWorkflowRequest,
    ReflectionWorkflowResult,
    StrictBatchedReflectionWorkflow,
)
from agent_evolve.application.campaign_selector_context_extension import (
    CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY,
    CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_MAX_BYTES,
    CampaignSelectorContextExtension,
    attach_campaign_selector_context_extension,
    resolve_campaign_selector_context_extension,
)
from agent_evolve.application.portfolio_memory_dose import (
    EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256,
    EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID,
    EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION,
    PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_DEFINITION_SHA256,
    PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_ID,
    PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_VERSION,
    PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256,
    PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID,
    PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION,
    PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_DEFINITION_SHA256,
    PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_ID,
    PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_VERSION,
    PortfolioMemoryContextTransferAssessment,
    PortfolioMemoryDoseCardSemantics,
    PortfolioMemoryDoseSupportScope,
    PortfolioMemoryTransferLadderAssessment,
    PortfolioMemoryTransferTier,
    assess_portfolio_memory_context_transfer,
    assess_portfolio_memory_transfer_ladder,
    derive_portfolio_memory_advisory_card_support,
    derive_portfolio_memory_dose_card_support,
)
from agent_evolve.application.portfolio_memory_transfer import (
    PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_DEFINITION_SHA256,
    PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_ID,
    PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_VERSION,
    PortfolioMemoryTransferCard,
    PortfolioMemoryTransferLane,
    PortfolioMemoryTransferLaneResolution,
    PortfolioMemoryTransferLaneResolver,
)
from agent_evolve.application.contextual_search_controller import (
    CONTEXTUAL_SEARCH_CONTROLLER_DEFINITION_SHA256,
    CONTEXTUAL_SEARCH_CONTROLLER_ID,
    CONTEXTUAL_SEARCH_CONTROLLER_VERSION,
    ContextualArmAllocation,
    ContextualArmPosterior,
    ContextualPortfolioAllocationSlice,
    ContextualSearchCompletionAudit,
    ContextualSearchDecision,
    ContextualSearchDelayedCredit,
    ContextualSearchLedger,
    ContextualSearchObservation,
    ContextualSearchQuery,
    ContextualSearchSnapshot,
    ContextualSearchStageAllocation,
    PhaseAwareContextualSearchController,
    SearchArmKind,
    SearchPhase,
    audit_completed_contextual_search_ledger,
    slice_contextual_search_decision,
)
from agent_evolve.application.contextual_campaign_planning import (
    CampaignContextualJointCapabilityProjector,
    CampaignContextualPlanningContext,
    CampaignContextualSearchPlan,
    CampaignContextualSearchPlanner,
    FiniteContractContextualJointCapabilityProjector,
)
from agent_evolve.application.contextual_campaign_outcomes import (
    CONTEXTUAL_PORTFOLIO_OUTCOME_DEFINITION_SHA256,
    CONTEXTUAL_PORTFOLIO_OUTCOME_POLICY_ID,
    CONTEXTUAL_PORTFOLIO_OUTCOME_POLICY_VERSION,
    ContextualPortfolioOutcomeBatch,
    observe_contextual_portfolio_outcomes,
)
from agent_evolve.application.contextual_delayed_credit import (
    CONTEXTUAL_DELAYED_CREDIT_DEFINITION_SHA256,
    CONTEXTUAL_DELAYED_CREDIT_POLICY_ID,
    CONTEXTUAL_DELAYED_CREDIT_POLICY_VERSION,
    ContextualPostRecombinationCreditBatch,
    ContextualTerminalPersistenceCreditBatch,
    observe_contextual_post_recombination_credit,
    observe_contextual_terminal_persistence,
)
from agent_evolve.application.portfolio_optimization_memory import (
    PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_DEFINITION_SHA256,
    PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_ID,
    PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_VERSION,
    PortfolioOptimizationMemoryAssessment,
    PortfolioOptimizationMemoryDirective,
    PortfolioOptimizationMemoryDisposition,
    PortfolioOptimizationMetricSign,
    assess_portfolio_optimization_memory,
)
from agent_evolve.application.identifiable_reflection_evidence import (
    IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_DEFINITION_SHA256,
    IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_ID,
    IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_VERSION,
    MAX_REFLECTION_LOCAL_INTERVENTION_VALUE_BYTES,
    IdentifiableMutationReflectionContrast,
    IdentifiableReflectionEvidenceSnapshot,
    ReflectionEvidenceExclusionReason,
    ReflectionFalsificationFeedback,
    project_identifiable_reflection_evidence,
)
from agent_evolve.application.evolution_campaign import (
    SealedCutoffDelayedAdmissionCadence,
)
from agent_evolve.core.problem import (
    ObjectiveSpec,
    Problem,
    ProblemContractError,
    ValidationOutcome,
    validate_objective_specs,
)
from agent_evolve.core.action_semantics import (
    ActionAxisCoordinateSemantics,
    ActionAxisSemantics,
    ActionSpaceSemantics,
    render_action_space_semantics,
)
from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
    render_optimization_semantics,
)
from agent_evolve.domain.artifact import ArtifactRef, artifact_ref_for_bytes
from agent_evolve.domain.finite_variation import (
    FiniteActionEvidenceBinding,
    FiniteVariationContract,
    FiniteVariationOption,
    bind_finite_action_evidence,
)
from agent_evolve.domain.finite_action_set import (
    FiniteActionCardAuthority,
    FiniteActionOptionAuthority,
    FiniteActionPresentationAuthority,
    FiniteActionSetAuthority,
    FiniteActionSourceMode,
    FiniteActionSupportAuthority,
    MAX_MATCHED_FINITE_ACTIONS,
    MIN_MATCHED_FINITE_ACTIONS,
)
from agent_evolve.domain.ids import (
    CandidateId,
    InsightId,
    LLMCallId,
    OperatorInvocationId,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.outcome import FailureCategory, FailureCode, FailureRecord
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey, require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    FrozenJsonValue,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory, UuidIdFactory
from agent_evolve.infrastructure.resource_lease import (
    FileExclusiveResourceLease,
    ResourceConflictDetected,
    ResourceLeaseUnavailable,
)
from agent_evolve.infrastructure.subprocess_boundary import (
    ExplicitEnvironmentSubprocessBoundary,
)
from agent_evolve.policies.structured_output_budget import (
    FixedStructuredOutputBudgetPolicy,
)
from agent_evolve.policies.memory.prompt_shape import PromptShapeCommitmentPolicy
from agent_evolve.policies.memory.treatment_compliance import (
    FiniteTreatmentAction,
    InsightTreatmentRequirement,
    StrictTreatmentCompliancePolicy,
    TREATMENT_COMPLIANCE_DEFINITION_SHA256,
    TREATMENT_COMPLIANCE_POLICY_ID,
    TREATMENT_COMPLIANCE_POLICY_VERSION,
    TreatmentAdmissionReceipt,
    TreatmentAdmissionRequest,
    TreatmentActionBinding,
    TreatmentAssignmentRole,
    TreatmentClaimMode,
    TreatmentCompliancePolicy,
    TreatmentComplianceRejected,
    TreatmentComplianceViolation,
    TreatmentInsightEvidence,
    TreatmentInsightBinding,
    TreatmentPreflightReceipt,
    TreatmentPreflightRequest,
    validate_treatment_admission_receipt,
    validate_treatment_preflight_receipt,
)
from agent_evolve.policies.feedback.held_out_asn import (
    G1ReflectionFeedbackInterceptor,
    HELD_OUT_SELECTOR_POLICY_ID,
    HELD_OUT_SELECTOR_POLICY_VERSION,
    HeldOutASNAssignmentCommitment,
    HeldOutASNAssignments,
    HeldOutASNPlanSet,
    HeldOutASNPlannerAdapter,
    HeldOutArm,
    HeldOutArmAssignment,
    HeldOutAssignmentUnavailable,
    HeldOutAssignmentUnavailableReason,
    HeldOutScoreMapEntry,
    REFLECTIVE_FEEDBACK_POLICY_ID,
    REFLECTIVE_FEEDBACK_POLICY_VERSION,
    ReflectedCard,
    ReflectedCardBatch,
    ReflectedCardMailbox,
    ReflectiveFeedbackContractError,
    build_reflected_card_batch,
    reflection_contrast_id,
    register_neutral_sham_card,
)
from agent_evolve.policies.selection.phenotype_recourse import (
    PhenotypeIdentity,
    PhenotypeIdentityPolicy,
    SemanticProjectionPhenotypeIdentityPolicy,
    TypedConfigurationPhenotypeIdentityPolicy,
)
from agent_evolve.policies.selection.finite_action import (
    POLICY_DEFINITION_SHA256 as UNIFORM_FINITE_ACTION_POLICY_DEFINITION_SHA256,
    POLICY_ID as UNIFORM_FINITE_ACTION_POLICY_ID,
    POLICY_VERSION as UNIFORM_FINITE_ACTION_POLICY_VERSION,
    TaskKeyedUniformFiniteActionPolicy,
)
from agent_evolve.policies.selection.archive_elite import (
    ArchiveEliteParentSelection,
    ArchiveEliteParentSelectionReceipt,
    ArchiveEliteParentSelector,
    ArchiveReservoirCrowdingKind,
    ArchiveReservoirParentSelection,
    ArchiveReservoirParentSelectionReceipt,
    ArchiveReservoirParentSelector,
    ArchiveReservoirRankedCandidate,
    POLICY_DEFINITION_SHA256 as ARCHIVE_ELITE_PARENT_POLICY_DEFINITION_SHA256,
    POLICY_ID as ARCHIVE_ELITE_PARENT_POLICY_ID,
    POLICY_VERSION as ARCHIVE_ELITE_PARENT_POLICY_VERSION,
    RESERVOIR_POLICY_DEFINITION_SHA256 as ARCHIVE_RESERVOIR_PARENT_POLICY_DEFINITION_SHA256,
    RESERVOIR_POLICY_ID as ARCHIVE_RESERVOIR_PARENT_POLICY_ID,
    RESERVOIR_POLICY_VERSION as ARCHIVE_RESERVOIR_PARENT_POLICY_VERSION,
    TaskKeyedArchiveEliteParentPolicy,
    TaskKeyedArchiveReservoirParentPolicy,
    validate_archive_elite_parent_selection,
    validate_archive_reservoir_parent_selection,
)
from agent_evolve.policies.selection.elite_explorer import (
    ArchiveEliteExplorerParentSelection,
    ArchiveEliteExplorerParentSelectionReceipt,
    ArchiveEliteExplorerParentSelector,
    EliteExplorerFallbackReason,
    EliteExplorerLaneId,
    EliteExplorerLaneReceipt,
    EliteExplorerLaneSource,
    POLICY_DEFINITION_SHA256 as ARCHIVE_ELITE_EXPLORER_PARENT_POLICY_DEFINITION_SHA256,
    POLICY_ID as ARCHIVE_ELITE_EXPLORER_PARENT_POLICY_ID,
    POLICY_VERSION as ARCHIVE_ELITE_EXPLORER_PARENT_POLICY_VERSION,
    ROTATION_LAW_ID as ARCHIVE_ELITE_EXPLORER_PARENT_ROTATION_LAW_ID,
    TaskKeyedArchiveEliteExplorerParentPolicy,
    validate_archive_elite_explorer_parent_selection,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    AgenticGenerator,
    CandidateDraft,
    FiniteVariationSelectionDraft,
    InsightDraft,
    MetricComparisonAnchor,
    MetricComparisonAnchorKind,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionConsumerScope,
    ReflectionEvidenceCatalog,
    ReflectionEvidenceCatalogEntry,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    ReflectionInsightContract,
    ReflectionInsightKind,
    VariationGenerationRequest,
    VariationGenerationResult,
    resolve_finite_variation_selection,
    validate_reflection_evidence_catalog_result,
    validate_reflection_insight_draft,
)
from agent_evolve.ports.action_forecast import (
    ActionEvidenceCitation,
    ActionForecastBlockPolicy,
    ActionForecastBlockRequest,
    ActionForecastBlockResult,
    ActionForecastBlockSpec,
    ActionForecastDraft,
    ActionForecastEvidenceMode,
    ActionForecastPartitionLayout,
    ActionForecastPartitionPolicyBinding,
    ActionForecastPolicy,
    ActionForecastRequest,
    ActionForecastResult,
    ActionMetricForecast,
    MetricForecastScale,
    ParentMetricValue,
    ResolvedActionEvidenceCitation,
    ResolvedActionForecast,
    ResolvedActionForecastBatch,
    ResolvedActionForecastBlock,
    ResolvedActionMetricForecast,
    resolve_action_forecast_block,
    resolve_action_forecasts,
    validate_resolved_action_forecasts,
)
from agent_evolve.ports.action_metric_projection import (
    ExactActionMetricProjection,
    ExactActionMetricProjectionBatch,
)
from agent_evolve.ports.action_allocation import (
    ActionAllocationRequest,
    ActionAllocationResult,
    ActionPortfolioDecision,
    AllocatedActionMember,
    DeterministicActionAllocator,
    ForecastPortfolioUtility,
    ForecastPortfolioUtilityBinding,
    ForecastPortfolioUtilityInput,
    ForecastQuantile,
    PortfolioAllocationScore,
    validate_action_portfolio_decision,
)
from agent_evolve.ports.action_allocation_frame import (
    ActionAllocationFrameSubsetPolicyBinding,
    ActionAllocationSurfaceAudit,
    ActionForecastAllocationFrameKind,
    AllocationCandidateScoreDiagnostic,
    AllocationCandidateScoreDiagnosticInput,
    AllocationScoreDiagnostic,
    AllocationScoreDiagnosticBinding,
    AllocationSurfaceGatePolicyBinding,
    AllocationSurfaceStepAudit,
    AuditedFrameActionAllocationResult,
    FrameActionAllocationRequest,
    FrameActionPortfolioDecision,
    ResolvedActionForecastAllocationFrame,
    allocation_score_multiset_sha256,
    allocation_surface_failure_codes,
    bind_action_forecast_block_allocation_frame,
    bind_action_forecast_block_subset_allocation_frame,
    bind_complete_action_forecast_allocation_frame,
    validate_frame_action_portfolio_decision,
)
from agent_evolve.ports.action_allocation_frame_commit import (
    FrameActionAllocationCommitInput,
    FrameActionAllocationTreatmentExecution,
    frame_source_call_and_request_identity,
    validate_treatment_occurrence_frame_request,
)
from agent_evolve.ports.portfolio_selection import (
    CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD,
    CANONICAL_REDACTED_PORTFOLIO_EVIDENCE_SHA256,
    CardScoreComponent,
    CardTransferAdjudicationRequest,
    CardTransferAdjudicator,
    CardTransferScoreReceipt,
    PortfolioCard,
    PortfolioCardPromptPayload,
    PortfolioCardSourceBinding,
    PortfolioCardSourceRegistry,
    PortfolioCardViewReceipt,
    PortfolioCardViewTransform,
    PortfolioExperimentalArm,
    PortfolioExperimentalViewReceipt,
    PortfolioMemberDraft,
    PortfolioSelectionPolicy,
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
    RankedPortfolioDecision,
    RankedPortfolioMember,
    derive_portfolio_card_view,
    portfolio_card_snapshot_sha256,
    portfolio_card_score_state_sha256,
    resolve_ranked_portfolio_decision,
    validate_card_transfer_score_receipt,
    validate_portfolio_experimental_view,
    validate_ranked_portfolio_decision,
)
from agent_evolve.ports.portfolio_memory_dose import (
    BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_DEFINITION_SHA256,
    BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_ID,
    BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_VERSION,
    BoundedPortfolioMemoryDoseContract,
    PortfolioMemoryDoseAssessment,
    PortfolioMemoryDoseCardSupport,
    PortfolioMemoryDoseMember,
    PortfolioMemoryDoseRejected,
    PortfolioMemoryDoseStage,
    PortfolioMemoryDoseViolation,
    PortfolioMemoryExposureScope,
    assess_evaluated_portfolio_memory_dose,
    assess_proposed_portfolio_memory_dose,
    require_passing_portfolio_memory_dose,
)
from agent_evolve.ports.structured_output_budget import (
    StructuredOutputBudgetPolicy,
    StructuredOutputRequestKind,
    resolve_structured_output_budget,
)
from agent_evolve.ports.objective_resolution import (
    EXACT_OBJECTIVE_RESOLUTION_DEFINITION_SHA256,
    EXACT_OBJECTIVE_RESOLUTION_POLICY_ID,
    EXACT_OBJECTIVE_RESOLUTION_POLICY_VERSION,
    ExactObjectiveResolution,
    ObjectiveResolutionPort,
    ObjectiveResolutionReceipt,
    ObjectiveResolutionRequest,
    ObjectiveResolutionResult,
    objective_resolution_policy_metadata,
    resolve_objectives,
)
from agent_evolve.ports.id_factory import IdFactory
from agent_evolve.ports.contextual_search_allocation import (
    ContextualArmCountCapability,
    ContextualArmCountCapabilityWitness,
    ContextualJointCountVector,
    ContextualLaneJointCountCapability,
    ContextualPortfolioAllocationContract,
    ContextualPortfolioAllocationRealization,
)
from agent_evolve.ports.frontier_target import (
    CampaignPortfolioFrontierTarget,
    CampaignPortfolioFrontierTargetAllocator,
)
from agent_evolve.ports.executable_hypothesis import (
    ExecutableHypothesisTestSpec,
    HypothesisApplicabilityPort,
    HypothesisApplicabilityStatus,
    HypothesisCompilationReceipt,
    HypothesisCompilationRequest,
    validate_hypothesis_compilation,
    validate_hypothesis_compiler_identity,
)
from agent_evolve.ports.finite_action_set import (
    FiniteActionSetCompilationRequest,
    FiniteActionSetCompiler,
    FiniteActionSetDraft,
    validate_finite_action_set_compiler_identity,
)
from agent_evolve.ports.finite_action_selection import (
    EngineFiniteActionPolicy,
    EngineFiniteActionRequest,
    FiniteActionDecision,
    FiniteActionSelectorKind,
    ProspectiveUniformRankToken,
    validate_finite_action_decision,
)
from agent_evolve.ports.resource_lease import (
    ExclusiveResourceLease,
    ResourceConflictObservation,
    ResourceConflictProbe,
    ResourceLeaseReceipt,
)
from agent_evolve.ports.subprocess_boundary import (
    BoundedSubprocessBoundary,
    ChildProcessPolicy,
    ChildProcessResult,
    EXPLICIT_ENVIRONMENT_BOUNDARY_DEFINITION_SHA256,
)
from agent_evolve.ports.variation_catalog import (
    FiniteVariationCatalog,
    bind_finite_variation_catalog,
)
from agent_evolve.ports.variation_source import (
    PRIMARY_VARIATION_SOURCE_ID,
    VARIATION_DIVERSITY_SIGNATURE_METADATA_KEY,
    VARIATION_OPERATOR_METADATA_KEY,
    VARIATION_SOURCE_METADATA_KEY,
    VARIATION_SOURCE_MINIMUM_METADATA_KEY,
    finite_variation_diversity_signature,
    finite_variation_operator_id,
    finite_variation_source_by_option,
    finite_variation_source_id,
    finite_variation_source_ids,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
TraceSink = Callable[[Mapping[str, object]], None]
PromptBuilder = Callable[[str, Any, tuple[dict[str, object], ...]], str]


def _catalog_identity(catalog: FiniteVariationCatalog) -> tuple[str, int, str]:
    catalog_id = getattr(catalog, "catalog_id", None)
    catalog_version = getattr(catalog, "catalog_version", None)
    definition_sha256 = getattr(catalog, "definition_sha256", None)
    if type(catalog_id) is not str or _TOKEN.fullmatch(catalog_id) is None:
        raise ValueError("finite variation catalog_id has invalid syntax")
    if type(catalog_version) is not int or catalog_version <= 0:
        raise ValueError("finite variation catalog_version must be positive")
    require_sha256(definition_sha256, "finite variation definition_sha256")
    return catalog_id, catalog_version, definition_sha256


def _catalog_option_families(
    catalog: FiniteVariationCatalog,
) -> tuple[str, ...]:
    """Read the explicit family vocabulary required by action semantics.

    Existing finite catalogs remain valid without this optional declaration.
    A benchmark that publishes action semantics must make each catalog's
    executable family vocabulary inspectable without materializing options for
    an arbitrary parent configuration.
    """

    families = getattr(catalog, "option_families", None)
    if (
        type(families) is not tuple
        or not families
        or any(
            type(family) is not str or _TOKEN.fullmatch(family) is None
            for family in families
        )
    ):
        raise TypeError(
            "action-semantics catalogs must publish a non-empty exact "
            "option_families tuple"
        )
    if families != tuple(sorted(set(families))):
        raise ValueError(
            "catalog option_families must be unique and canonically sorted"
        )
    return families


@dataclass(frozen=True, slots=True)
class AgenticBenchmark:
    """Frozen inverted-API bundle for one optimization benchmark.

    Detailed evaluation is intentionally all-or-nothing: an evidence adapter
    requires an explicit outcome relation, while an outcome relation cannot be
    supplied without detailed evidence.  The conservative exact-configuration
    phenotype policy remains a safe default; domains may inject a semantic
    projection when they can prove evaluator equivalence.  Objective resolution
    is likewise opt-in: ``None`` preserves the historical exact evaluator
    projection, while an injected resolver separates raw measurement evidence
    from the stable decision values consumed by selection and rewards.
    """

    problem: Problem[dict[str, object]]
    reward: RewardPolicyBinding = field(
        default_factory=lambda: RewardPolicyBinding(
            default_parent_relative_reward,
            REWARD_DEFINITION_HASH,
        )
    )
    detailed_evaluator: DetailedEvaluationAdapter | None = None
    outcome_relation: OutcomeRelationPolicyBinding | None = None
    optimization_semantics: OptimizationSemantics | None = None
    action_semantics: ActionSpaceSemantics | None = None
    phenotype_identity: PhenotypeIdentityPolicy = field(
        default_factory=TypedConfigurationPhenotypeIdentityPolicy
    )
    finite_variation_catalogs: tuple[FiniteVariationCatalog, ...] = ()
    hypothesis_compiler: HypothesisApplicabilityPort | None = None
    finite_action_set_compiler: FiniteActionSetCompiler | None = None
    objective_resolution: ObjectiveResolutionPort | None = None
    _objectives: tuple[ObjectiveSpec, ...] = field(init=False, repr=False)
    _catalog_identities: tuple[tuple[str, int, str], ...] = field(
        init=False,
        repr=False,
    )
    _evaluator_identity: EvaluatorIdentity | None = field(init=False, repr=False)
    _phenotype_policy_identity: tuple[str, int] = field(init=False, repr=False)
    _optimization_semantics_identity: tuple[str, int, str] | None = field(
        init=False,
        repr=False,
    )
    _objective_resolution_identity: tuple[str, int, str] | None = field(
        init=False,
        repr=False,
    )
    _action_semantics_identity: tuple[str, int, str] | None = field(
        init=False,
        repr=False,
    )
    _catalog_option_family_bindings: tuple[tuple[str, ...], ...] = field(
        init=False,
        repr=False,
    )
    _hypothesis_compiler_identity: tuple[str, int, str] | None = field(
        init=False,
        repr=False,
    )
    _finite_action_set_compiler_identity: tuple[str, int, str] | None = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.problem, Problem):
            raise TypeError("problem must implement the public Problem protocol")
        objectives = tuple(self.problem.objectives)
        validate_objective_specs(objectives)
        candidate_model = getattr(self.problem, "candidate_model", None)
        if not isinstance(candidate_model, type) or not issubclass(
            candidate_model,
            BaseModel,
        ):
            raise TypeError("agentic problems must publish a Pydantic candidate_model")
        schema = candidate_model.model_json_schema(by_alias=False)
        if type(schema) is not dict or schema.get("type") != "object":
            raise TypeError("candidate_model must publish an object JSON schema")

        if type(self.reward) is not RewardPolicyBinding:
            raise TypeError("reward must be an exact RewardPolicyBinding")
        RewardPolicyBinding.__post_init__(self.reward)

        identify = getattr(self.phenotype_identity, "identify", None)
        if not callable(identify):
            raise TypeError("phenotype_identity must implement identify")
        phenotype_policy_identity = (
            getattr(self.phenotype_identity, "policy_id", None),
            getattr(self.phenotype_identity, "policy_version", None),
        )
        PhenotypeIdentity(
            policy_id=phenotype_policy_identity[0],
            policy_version=phenotype_policy_identity[1],
            value_sha256="0" * 64,
        )

        evaluator = self.detailed_evaluator
        relation = self.outcome_relation
        evaluator_identity: EvaluatorIdentity | None = None
        if evaluator is None:
            if relation is not None:
                raise ValueError(
                    "outcome_relation requires a detailed_evaluator; "
                    "objective-only mode derives its Pareto relation"
                )
        else:
            if not isinstance(evaluator, DetailedEvaluationAdapter):
                raise TypeError(
                    "detailed_evaluator must implement DetailedEvaluationAdapter"
                )
            evaluator_identity = getattr(evaluator, "evaluator_identity", None)
            if type(evaluator_identity) is not EvaluatorIdentity:
                raise TypeError(
                    "detailed_evaluator must publish an exact evaluator_identity"
                )
            EvaluatorIdentity.__post_init__(evaluator_identity)
            if type(relation) is not OutcomeRelationPolicyBinding:
                raise ValueError(
                    "detailed evaluation requires an explicit outcome_relation"
                )
            OutcomeRelationPolicyBinding.__post_init__(relation)

        active_relation = (
            objective_pareto_outcome_binding(objectives)
            if relation is None
            else relation
        )
        semantics = self.optimization_semantics
        if semantics is None:
            semantics = getattr(self.problem, "optimization_semantics", None)
        if semantics is not None:
            if type(semantics) is not OptimizationSemantics:
                raise TypeError(
                    "optimization_semantics must be an exact OptimizationSemantics"
                )
            OptimizationSemantics.__post_init__(semantics)
            semantics.validate_binding(objectives, active_relation.identity)

        objective_resolution_identity = (
            None
            if self.objective_resolution is None
            else objective_resolution_policy_metadata(self.objective_resolution)
        )

        catalogs = self.finite_variation_catalogs
        if type(catalogs) is not tuple:
            raise TypeError("finite_variation_catalogs must be an exact tuple")
        catalog_identities: list[tuple[str, int, str]] = []
        for catalog in catalogs:
            if not isinstance(catalog, FiniteVariationCatalog):
                raise TypeError(
                    "finite_variation_catalogs must implement FiniteVariationCatalog"
                )
            catalog_identities.append(_catalog_identity(catalog))
        catalog_ids = tuple(identity[0] for identity in catalog_identities)
        if len(set(catalog_ids)) != len(catalog_ids):
            raise ValueError("finite variation catalog IDs must be unique")

        hypothesis_compiler_identity: tuple[str, int, str] | None = None
        compiler = self.hypothesis_compiler
        if compiler is not None:
            if not isinstance(compiler, HypothesisApplicabilityPort):
                raise TypeError(
                    "hypothesis_compiler must implement HypothesisApplicabilityPort"
                )
            compiler_id = getattr(compiler, "policy_id", None)
            compiler_version = getattr(compiler, "policy_version", None)
            compiler_sha256 = getattr(compiler, "definition_sha256", None)
            if type(compiler_id) is not str or _TOKEN.fullmatch(compiler_id) is None:
                raise ValueError("hypothesis compiler policy_id has invalid syntax")
            if type(compiler_version) is not int or compiler_version <= 0:
                raise ValueError("hypothesis compiler policy_version must be positive")
            require_sha256(
                compiler_sha256,
                "hypothesis compiler definition_sha256",
            )
            hypothesis_compiler_identity = (
                compiler_id,
                compiler_version,
                compiler_sha256,
            )

        finite_action_set_compiler_identity: tuple[str, int, str] | None = None
        action_set_compiler = self.finite_action_set_compiler
        if action_set_compiler is not None:
            finite_action_set_compiler_identity = (
                validate_finite_action_set_compiler_identity(action_set_compiler)
            )

        action_semantics = self.action_semantics
        if action_semantics is None:
            action_semantics = getattr(self.problem, "action_semantics", None)
        catalog_option_family_bindings: tuple[tuple[str, ...], ...] = ()
        if action_semantics is not None:
            if type(action_semantics) is not ActionSpaceSemantics:
                raise TypeError(
                    "action_semantics must be an exact ActionSpaceSemantics"
                )
            ActionSpaceSemantics.__post_init__(action_semantics)
            catalog_option_family_bindings = tuple(
                _catalog_option_families(catalog) for catalog in catalogs
            )
            action_semantics.validate_catalog_binding(
                tuple(catalog_identities),
                tuple(
                    family
                    for families in catalog_option_family_bindings
                    for family in families
                ),
            )

        object.__setattr__(self, "_objectives", objectives)
        object.__setattr__(self, "_catalog_identities", tuple(catalog_identities))
        object.__setattr__(self, "_evaluator_identity", evaluator_identity)
        object.__setattr__(self, "optimization_semantics", semantics)
        object.__setattr__(self, "action_semantics", action_semantics)
        object.__setattr__(
            self,
            "_optimization_semantics_identity",
            None if semantics is None else semantics.identity,
        )
        object.__setattr__(
            self,
            "_objective_resolution_identity",
            objective_resolution_identity,
        )
        object.__setattr__(
            self,
            "_action_semantics_identity",
            None if action_semantics is None else action_semantics.identity,
        )
        object.__setattr__(
            self,
            "_catalog_option_family_bindings",
            catalog_option_family_bindings,
        )
        object.__setattr__(
            self,
            "_phenotype_policy_identity",
            phenotype_policy_identity,
        )
        object.__setattr__(
            self,
            "_hypothesis_compiler_identity",
            hypothesis_compiler_identity,
        )
        object.__setattr__(
            self,
            "_finite_action_set_compiler_identity",
            finite_action_set_compiler_identity,
        )

    @property
    def objectives(self) -> tuple[ObjectiveSpec, ...]:
        """Return the objective declaration frozen at adapter construction."""

        return self._objectives

    @property
    def finite_variation_catalog_identities(
        self,
    ) -> tuple[tuple[str, int, str], ...]:
        """Return identities in their frozen catalog declaration order."""

        return self._catalog_identities

    def validate_binding(self) -> None:
        """Fail if mutable adapter objects changed after bundle construction."""

        if tuple(self.problem.objectives) != self._objectives:
            raise ValueError("problem objectives changed after benchmark binding")
        current_phenotype_identity = (
            getattr(self.phenotype_identity, "policy_id", None),
            getattr(self.phenotype_identity, "policy_version", None),
        )
        if current_phenotype_identity != self._phenotype_policy_identity:
            raise ValueError("phenotype policy identity changed after binding")
        current_evaluator_identity = (
            None
            if self.detailed_evaluator is None
            else getattr(self.detailed_evaluator, "evaluator_identity", None)
        )
        if current_evaluator_identity != self._evaluator_identity:
            raise ValueError("detailed evaluator identity changed after binding")
        current_semantics_identity = (
            None
            if self.optimization_semantics is None
            else self.optimization_semantics.identity
        )
        if current_semantics_identity != self._optimization_semantics_identity:
            raise ValueError("optimization semantics identity changed after binding")
        current_objective_resolution_identity = (
            None
            if self.objective_resolution is None
            else objective_resolution_policy_metadata(self.objective_resolution)
        )
        if current_objective_resolution_identity != self._objective_resolution_identity:
            raise ValueError(
                "objective-resolution policy identity changed after binding"
            )
        current_action_semantics_identity = (
            None if self.action_semantics is None else self.action_semantics.identity
        )
        if current_action_semantics_identity != self._action_semantics_identity:
            raise ValueError("action semantics identity changed after binding")
        identities = tuple(
            _catalog_identity(catalog) for catalog in self.finite_variation_catalogs
        )
        if identities != self._catalog_identities:
            raise ValueError("finite variation catalog identity changed after binding")
        current_compiler_identity = (
            None
            if self.hypothesis_compiler is None
            else (
                getattr(self.hypothesis_compiler, "policy_id", None),
                getattr(self.hypothesis_compiler, "policy_version", None),
                getattr(self.hypothesis_compiler, "definition_sha256", None),
            )
        )
        if current_compiler_identity != self._hypothesis_compiler_identity:
            raise ValueError("hypothesis compiler identity changed after binding")
        current_action_set_compiler_identity = (
            None
            if self.finite_action_set_compiler is None
            else validate_finite_action_set_compiler_identity(
                self.finite_action_set_compiler
            )
        )
        if (
            current_action_set_compiler_identity
            != self._finite_action_set_compiler_identity
        ):
            raise ValueError(
                "finite action set compiler identity changed after binding"
            )
        if self.action_semantics is not None:
            current_family_bindings = tuple(
                _catalog_option_families(catalog)
                for catalog in self.finite_variation_catalogs
            )
            if current_family_bindings != self._catalog_option_family_bindings:
                raise ValueError(
                    "finite variation catalog family vocabulary changed after binding"
                )
            self.action_semantics.validate_catalog_binding(
                identities,
                tuple(
                    family
                    for families in current_family_bindings
                    for family in families
                ),
            )

    def bind_finite_variation(
        self,
        catalog_id: str,
        parent_configuration: object,
    ) -> FiniteVariationContract:
        """Seal one named benchmark catalog against an exact parent."""

        self.validate_binding()
        if type(catalog_id) is not str or _TOKEN.fullmatch(catalog_id) is None:
            raise ValueError("catalog_id has invalid syntax")
        matches = tuple(
            catalog
            for catalog in self.finite_variation_catalogs
            if catalog.catalog_id == catalog_id
        )
        if len(matches) != 1:
            raise KeyError(f"unknown finite variation catalog {catalog_id!r}")
        catalog = matches[0]
        frozen = freeze_json(parent_configuration)
        if type(frozen) is not FrozenJsonObject:
            raise TypeError("finite variation parents must be typed-JSON objects")
        contract = bind_finite_variation_catalog(catalog, frozen)
        expected_identity = next(
            identity
            for identity in self._catalog_identities
            if identity[0] == catalog_id
        )
        if _catalog_identity(catalog) != expected_identity:
            raise ValueError("finite variation catalog changed while binding options")
        return contract

    def compile_registered_hypothesis_treatment(
        self,
        *,
        catalog_id: str,
        parent_candidate_id: CandidateId,
        parent_configuration: object,
        entry: InsightMemoryEntry,
        requested_operator_kind: str,
        context_projection_sha256: str,
        endpoint_definition_sha256: str,
    ) -> CompiledHypothesisTreatment:
        """Compile one exact registered card into a strict executable treatment."""

        if type(entry) is not InsightMemoryEntry:
            raise TypeError("entry must be an exact InsightMemoryEntry")
        entry.__post_init__()
        self.validate_binding()
        compiler = self.hypothesis_compiler
        if compiler is None:
            raise RuntimeError("benchmark has no hypothesis compiler")
        contract = self.bind_finite_variation(catalog_id, parent_configuration)
        request = HypothesisCompilationRequest(
            parent_candidate_id=parent_candidate_id,
            reference=entry.reference,
            insight=entry.draft,
            source_evidence_sha256=registered_source_evidence_sha256(entry),
            requested_operator_kind=requested_operator_kind,
            source_operator_kinds=entry.applicable_operator_kinds,
            parent_configuration_sha256=contract.parent_configuration_sha256,
            finite_contract=contract,
            context_projection_sha256=context_projection_sha256,
            endpoint_definition_sha256=endpoint_definition_sha256,
        )
        return compile_registered_hypothesis_treatment(
            entry=entry,
            request=request,
            compiler=compiler,
        )

    def compile_finite_action_set(
        self,
        *,
        compiled_anchor: CompiledHypothesisTreatment,
        required_cardinality: int,
        source_mode: FiniteActionSourceMode = (
            FiniteActionSourceMode.COMPILED_ACTIVE_CARD
        ),
    ) -> tuple[FiniteActionSetAuthority, FiniteActionSetDraft]:
        """Seal a matched K-option neighbourhood around an exact card anchor.

        The benchmark-owned compiler returns only option IDs.  This composition
        root resolves and authenticates the complete children and phenotypes;
        no outcome or evaluator object crosses the compiler boundary.
        """

        self.validate_binding()
        compiler = self.finite_action_set_compiler
        if compiler is None:
            raise RuntimeError("benchmark has no finite action set compiler")
        if type(compiled_anchor) is not CompiledHypothesisTreatment:
            raise TypeError(
                "compiled_anchor must be an exact CompiledHypothesisTreatment"
            )
        CompiledHypothesisTreatment.__post_init__(compiled_anchor)
        if len(compiled_anchor.requirement.allowed_actions) != 1:
            raise ValueError("finite action compilation requires one exact anchor")
        source_contract = compiled_anchor.request.finite_contract
        catalog_identity = (
            source_contract.catalog_id,
            source_contract.catalog_version,
            source_contract.catalog_definition_sha256,
        )
        if catalog_identity not in self._catalog_identities:
            raise ValueError("compiled anchor uses a foreign benchmark catalog")
        anchor = compiled_anchor.requirement.allowed_actions[0]
        request = FiniteActionSetCompilationRequest(
            parent_candidate_id=compiled_anchor.request.parent_candidate_id,
            finite_contract=source_contract,
            anchor_option_id=anchor.option_id,
            anchor_option_identity_sha256=anchor.option_identity_sha256,
            exact_anchor_requirement_sha256=(
                compiled_anchor.requirement.requirement_sha256
            ),
            card_reference=compiled_anchor.request.reference,
            card_content_sha256=compiled_anchor.request.insight.content_sha256,
            context_projection_sha256=(
                compiled_anchor.request.context_projection_sha256
            ),
            endpoint_definition_sha256=(
                compiled_anchor.request.endpoint_definition_sha256
            ),
            required_cardinality=required_cardinality,
            current_outcome_access=False,
        )
        return compile_and_seal_finite_action_set(
            compiled_anchor=compiled_anchor,
            request=request,
            compiler=compiler,
            phenotype_identity=self.phenotype_identity,
            source_mode=source_mode,
        )


@dataclass(frozen=True, slots=True)
class _BoundGenerationPlanner:
    """Authenticate planner waves against benchmark-owned policy bindings."""

    benchmark: AgenticBenchmark
    delegate: GenerationPlanner

    def plan(
        self,
        state: OptimizerState,
        budget: OptimizerBudget,
    ) -> GenerationPlan:
        plan = self.delegate.plan(state, budget)
        if type(plan) is not GenerationPlan:
            raise TypeError("planner must return an exact GenerationPlan")
        GenerationPlan.__post_init__(plan)
        if plan.reward.binding.definition_hash != self.benchmark.reward.definition_hash:
            raise ValueError(
                "planner wave reward differs from the benchmark reward binding"
            )
        if plan.reward.binding.binding_sha256 != self.benchmark.reward.binding_sha256:
            raise ValueError(
                "planner wave total reward binding differs from the benchmark"
            )

        expected_by_catalog_parent: dict[
            tuple[str, str],
            FiniteVariationContract,
        ] = {}
        for slot in plan.slots:
            compiled_bindings = tuple(
                {
                    value.binding_sha256: value
                    for value in (
                        *slot.plan.compiled_hypothesis_eligibility,
                        *(
                            ()
                            if slot.plan.compiled_hypothesis_treatment is None
                            else (slot.plan.compiled_hypothesis_treatment,)
                        ),
                    )
                }.values()
            )
            for compiled in compiled_bindings:
                compiled.__post_init__()
                if (
                    compiled.request.endpoint_definition_sha256
                    != plan.reward.binding.definition_hash
                ):
                    raise ValueError(
                        "compiled treatment endpoint differs from generation reward/Q"
                    )
                compiler_identity = self.benchmark._hypothesis_compiler_identity
                receipt_identity = (
                    compiled.receipt.compiler_policy_id,
                    compiled.receipt.compiler_policy_version,
                    compiled.receipt.compiler_definition_sha256,
                )
                if compiler_identity is None or receipt_identity != compiler_identity:
                    raise ValueError(
                        "compiled treatment was not issued by the benchmark compiler"
                    )
            contract = slot.plan.finite_variation_contract
            if contract is None:
                continue
            parent = slot.plan.parents[0]
            key = (
                contract.catalog_id,
                typed_json_sha256(parent.configuration),
            )
            expected = expected_by_catalog_parent.get(key)
            if expected is None:
                expected = self.benchmark.bind_finite_variation(
                    contract.catalog_id,
                    parent.configuration,
                )
                expected_by_catalog_parent[key] = expected
            finite_authority = slot.plan.finite_action_set_authority
            if finite_authority is None:
                if contract.identity_sha256 != expected.identity_sha256:
                    raise ValueError(
                        "finite variation contract was not produced by the "
                        "benchmark-bound catalog"
                    )
            else:
                compiler_identity = self.benchmark._finite_action_set_compiler_identity
                authority_identity = (
                    finite_authority.support_compiler_policy_id,
                    finite_authority.support_compiler_policy_version,
                    finite_authority.support_compiler_definition_sha256,
                )
                if compiler_identity is None or authority_identity != compiler_identity:
                    raise ValueError(
                        "finite action authority was not issued by the "
                        "benchmark-bound support compiler"
                    )
                if (
                    finite_authority.support.support_contract.identity_sha256
                    != contract.identity_sha256
                    or finite_authority.support.source_contract_sha256
                    != expected.identity_sha256
                    or finite_authority.support.parent_candidate_id
                    != parent.candidate_id
                ):
                    raise ValueError(
                        "finite action authority differs from its benchmark plan"
                    )
        return plan


@runtime_checkable
class GenerationPlannerFactory(Protocol):
    """Build a planner against the exact runtime objects owned by composition.

    Stateful planners must never be constructed against look-alike engines, ID
    factories, or memory banks.  This deferred seam preserves the inverted API:
    benchmark packages provide a factory, while the public composition root owns
    construction order and passes the authoritative runtime identities exactly
    once.
    """

    def build(
        self,
        *,
        benchmark: AgenticBenchmark,
        engine: AgenticEvolutionEngine,
        id_factory: IdFactory,
        memory: InsightMemoryBank,
    ) -> GenerationPlanner: ...


@runtime_checkable
class GenerationFeedbackInterceptorFactory(Protocol):
    """Build post-wave feedback against the exact composed runtime objects."""

    def build(
        self,
        *,
        benchmark: AgenticBenchmark,
        engine: AgenticEvolutionEngine,
        id_factory: IdFactory,
        memory: InsightMemoryBank,
        planner: GenerationPlanner,
    ) -> GenerationFeedbackInterceptor: ...


@dataclass(frozen=True, slots=True)
class AgenticOptimizerComposition:
    """Fully wired agentic engine, archive, and budgeted optimizer."""

    benchmark: AgenticBenchmark
    id_factory: IdFactory
    memory: InsightMemoryBank
    engine: AgenticEvolutionEngine
    archive: ParetoArchive
    planner: GenerationPlanner
    feedback_interceptor: GenerationFeedbackInterceptor | None
    optimizer: BudgetedAgenticOptimizer

    def __post_init__(self) -> None:
        if self.engine.problem is not self.benchmark.problem:
            raise ValueError("composition engine is bound to a different problem")
        if self.engine.ids is not self.id_factory:
            raise ValueError("composition engine is bound to a different ID factory")
        if self.engine.memory is not self.memory:
            raise ValueError("composition engine is bound to a different memory bank")
        if self.engine.outcome_relation_binding is not (
            self.archive.outcome_relation_binding
        ):
            raise ValueError("engine and archive must share one relation binding")
        if self.optimizer.engine is not self.engine:
            raise ValueError("optimizer is bound to a different engine")
        if self.optimizer.archive is not self.archive:
            raise ValueError("optimizer is bound to a different archive")
        bound_planner = self.optimizer.planner
        if (
            type(bound_planner) is not _BoundGenerationPlanner
            or bound_planner.delegate is not self.planner
        ):
            raise ValueError("optimizer is bound to a different runtime planner")
        if self.optimizer.feedback_interceptor is not self.feedback_interceptor:
            raise ValueError("optimizer is bound to a different feedback interceptor")
        if self.engine.reward_binding.binding_sha256 != (
            self.benchmark.reward.binding_sha256
        ):
            raise ValueError(
                "engine total reward binding differs from the benchmark binding"
            )
        if self.engine.optimization_semantics is not (
            self.benchmark.optimization_semantics
        ):
            raise ValueError(
                "engine and benchmark must share one optimization semantics value"
            )
        if self.engine.objective_resolution is not (
            self.benchmark.objective_resolution
        ):
            raise ValueError(
                "engine and benchmark must share one objective-resolution port"
            )

    @property
    def outcome_relation(self) -> OutcomeRelationPolicyBinding:
        """The exact shared engine/archive relation object."""

        return self.engine.outcome_relation_binding

    def bind_finite_variation(
        self,
        catalog_id: str,
        parent_configuration: object,
    ) -> FiniteVariationContract:
        return self.benchmark.bind_finite_variation(
            catalog_id,
            parent_configuration,
        )


@dataclass(frozen=True, slots=True)
class PortfolioEvolutionComposition:
    """Benchmark-bound engine and ranked-portfolio application service.

    Unlike :class:`AgenticOptimizerComposition`, this composition has no
    generation planner, archive scheduler, budget, or optimizer.  Workloads
    inject only their benchmark semantics plus the two model-facing ports;
    the composition root owns the exact engine, IDs, and memory identities.
    """

    benchmark: AgenticBenchmark
    id_factory: IdFactory
    memory: InsightMemoryBank
    engine: AgenticEvolutionEngine
    portfolio: PortfolioEvolution

    def __post_init__(self) -> None:
        if type(self.benchmark) is not AgenticBenchmark:
            raise TypeError("benchmark must be an exact AgenticBenchmark")
        if not isinstance(self.id_factory, IdFactory):
            raise TypeError("id_factory must implement IdFactory")
        if type(self.memory) is not InsightMemoryBank:
            raise TypeError("memory must be an exact InsightMemoryBank")
        if type(self.engine) is not AgenticEvolutionEngine:
            raise TypeError("engine must be an exact AgenticEvolutionEngine")
        if type(self.portfolio) is not PortfolioEvolution:
            raise TypeError("portfolio must be an exact PortfolioEvolution")
        if self.engine.problem is not self.benchmark.problem:
            raise ValueError("composition engine is bound to a different problem")
        if self.engine.ids is not self.id_factory:
            raise ValueError("composition engine is bound to a different ID factory")
        if self.engine.memory is not self.memory:
            raise ValueError("composition engine is bound to a different memory bank")
        if self.portfolio.engine is not self.engine:
            raise ValueError("portfolio service is bound to a different engine")
        if self.portfolio.ids is not self.id_factory:
            raise ValueError("portfolio service is bound to a different ID factory")
        if self.portfolio.memory is not self.memory:
            raise ValueError("portfolio service is bound to a different memory bank")
        if self.engine.reward_binding.binding_sha256 != (
            self.benchmark.reward.binding_sha256
        ):
            raise ValueError(
                "engine total reward binding differs from the benchmark binding"
            )
        if self.engine.optimization_semantics is not (
            self.benchmark.optimization_semantics
        ):
            raise ValueError(
                "engine and benchmark must share one optimization semantics value"
            )
        if self.engine.objective_resolution is not (
            self.benchmark.objective_resolution
        ):
            raise ValueError(
                "engine and benchmark must share one objective-resolution port"
            )

    @property
    def outcome_relation(self) -> OutcomeRelationPolicyBinding:
        """Return the engine's authoritative benchmark outcome relation."""

        return self.engine.outcome_relation_binding

    def bind_finite_variation(
        self,
        catalog_id: str,
        parent_configuration: object,
    ) -> FiniteVariationContract:
        """Seal one benchmark-owned finite catalog against an exact parent."""

        return self.benchmark.bind_finite_variation(
            catalog_id,
            parent_configuration,
        )


def compose_portfolio_evolution(
    benchmark: AgenticBenchmark,
    *,
    generator: AgenticGenerator,
    selector: PortfolioSelectionPolicy,
    seed: int,
    id_factory: IdFactory | None = None,
    memory: InsightMemoryBank | None = None,
    initial_proposal_sequence: int = 0,
    evaluator_concurrency: int = 4,
    engine_trace_sink: TraceSink | None = None,
    prompt_builder: PromptBuilder = default_evidence_prompt,
    prompt_shape_commitment_policy: PromptShapeCommitmentPolicy | None = None,
    reflection_row_projection: ReflectionRowProjectionBinding | None = None,
    reflection_workflow: ReflectionWorkflow | None = None,
    max_output_tokens: int = 2_048,
    structured_output_budget_policy: StructuredOutputBudgetPolicy | None = None,
    temperature: float | None = 0.2,
    treatment_compliance_policy: TreatmentCompliancePolicy | None = None,
    provider_traffic_witness: "ProviderTrafficWitness | None" = None,
) -> PortfolioEvolutionComposition:
    """Compose ranked finite-option evolution without an optimizer surrogate.

    The benchmark owns evaluation, reward, outcome ordering, phenotype
    identity, and semantic declarations.  The selector owns only ranked opaque
    option choice.  Exact child materialization and concurrent evaluation stay
    under the benchmark-bound engine, so the same public composition works for
    continuous, discrete, and constructive workloads.
    """

    if type(benchmark) is not AgenticBenchmark:
        raise TypeError("benchmark must be an exact AgenticBenchmark")
    benchmark.validate_binding()
    if not isinstance(generator, AgenticGenerator):
        raise TypeError("generator must implement AgenticGenerator")
    if not isinstance(selector, PortfolioSelectionPolicy):
        raise TypeError("selector must implement PortfolioSelectionPolicy")
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    if type(initial_proposal_sequence) is not int or initial_proposal_sequence < 0:
        raise ValueError("initial_proposal_sequence must be non-negative")
    if type(evaluator_concurrency) is not int or evaluator_concurrency <= 0:
        raise ValueError("evaluator_concurrency must be positive")
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if temperature is not None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) < 0
        ):
            raise ValueError("temperature must be finite and non-negative or None")
        temperature = float(temperature)
    if not callable(prompt_builder):
        raise TypeError("prompt_builder must be callable")
    if engine_trace_sink is not None and not callable(engine_trace_sink):
        raise TypeError("engine_trace_sink must be callable")

    ids = UuidIdFactory() if id_factory is None else id_factory
    if not isinstance(ids, IdFactory):
        raise TypeError("id_factory must implement IdFactory")
    active_memory = InsightMemoryBank(id_factory=ids) if memory is None else memory
    if type(active_memory) is not InsightMemoryBank:
        raise TypeError("memory must be an exact InsightMemoryBank")

    engine = AgenticEvolutionEngine(
        problem=benchmark.problem,
        generator=generator,
        id_factory=ids,
        memory=active_memory,
        seed=seed,
        initial_proposal_sequence=initial_proposal_sequence,
        evaluator_concurrency=evaluator_concurrency,
        trace_sink=engine_trace_sink,
        reward_policy=benchmark.reward.score,
        reward_definition_hash=benchmark.reward.definition_hash,
        failure_score=benchmark.reward.failure_score,
        prompt_builder=prompt_builder,
        prompt_shape_commitment_policy=prompt_shape_commitment_policy,
        reflection_row_projection=reflection_row_projection,
        reflection_workflow=reflection_workflow,
        max_output_tokens=max_output_tokens,
        structured_output_budget_policy=structured_output_budget_policy,
        temperature=temperature,
        phenotype_identity_policy=benchmark.phenotype_identity,
        detailed_evaluator=benchmark.detailed_evaluator,
        outcome_relation_binding=benchmark.outcome_relation,
        optimization_semantics=benchmark.optimization_semantics,
        objective_resolution=benchmark.objective_resolution,
        treatment_compliance_policy=treatment_compliance_policy,
    )
    # Recheck after constructing provider-facing collaborators.  This mirrors
    # the optimizer composition's frozen benchmark drift protection.
    benchmark.validate_binding()
    portfolio = PortfolioEvolution(
        engine=engine,
        selector=selector,
        ids=ids,
        memory=active_memory,
        provider_traffic_witness=provider_traffic_witness,
    )
    return PortfolioEvolutionComposition(
        benchmark=benchmark,
        id_factory=ids,
        memory=active_memory,
        engine=engine,
        portfolio=portfolio,
    )


def compose_agentic_optimizer(
    benchmark: AgenticBenchmark,
    *,
    generator: AgenticGenerator,
    planner: GenerationPlanner | None = None,
    planner_factory: GenerationPlannerFactory | None = None,
    budget: OptimizerBudget,
    seed: int,
    id_factory: IdFactory | None = None,
    memory: InsightMemoryBank | None = None,
    initial_proposal_sequence: int = 0,
    evaluator_concurrency: int = 4,
    engine_trace_sink: TraceSink | None = None,
    optimizer_trace_sink: TraceSink | None = None,
    prompt_builder: PromptBuilder = default_evidence_prompt,
    prompt_shape_commitment_policy: PromptShapeCommitmentPolicy | None = None,
    reflection_row_projection: ReflectionRowProjectionBinding | None = None,
    reflection_workflow: ReflectionWorkflow | None = None,
    max_output_tokens: int = 2_048,
    structured_output_budget_policy: StructuredOutputBudgetPolicy | None = None,
    temperature: float | None = 0.2,
    evidence_admission_policy: EvidenceAdmissionPolicy = (
        EvidenceAdmissionPolicy.REQUIRE_COMPLIANT
    ),
    seed_admission_policy: SeedAdmissionPolicy | None = None,
    feedback_interceptor: GenerationFeedbackInterceptor | None = None,
    feedback_interceptor_factory: GenerationFeedbackInterceptorFactory | None = None,
    treatment_compliance_policy: TreatmentCompliancePolicy | None = None,
) -> AgenticOptimizerComposition:
    """Compose one benchmark without leaking domain policy into the core.

    The optimizer receives a narrow planner guard.  It verifies that every
    wave uses the benchmark reward identity and that every finite-selection
    contract is a deterministic snapshot of the benchmark-owned catalog.
    """

    if type(benchmark) is not AgenticBenchmark:
        raise TypeError("benchmark must be an exact AgenticBenchmark")
    benchmark.validate_binding()
    if not isinstance(generator, AgenticGenerator):
        raise TypeError("generator must implement AgenticGenerator")
    if (planner is None) == (planner_factory is None):
        raise ValueError("supply exactly one of planner or planner_factory")
    if planner is not None and not callable(getattr(planner, "plan", None)):
        raise TypeError("planner must implement plan(state, budget)")
    if planner_factory is not None and not isinstance(
        planner_factory,
        GenerationPlannerFactory,
    ):
        raise TypeError("planner_factory must implement build")
    if feedback_interceptor is not None and feedback_interceptor_factory is not None:
        raise ValueError(
            "supply at most one of feedback_interceptor or feedback_interceptor_factory"
        )
    if feedback_interceptor is not None and not isinstance(
        feedback_interceptor,
        GenerationFeedbackInterceptor,
    ):
        raise TypeError(
            "feedback_interceptor must implement reserve and after_generation"
        )
    if feedback_interceptor_factory is not None and not isinstance(
        feedback_interceptor_factory,
        GenerationFeedbackInterceptorFactory,
    ):
        raise TypeError("feedback_interceptor_factory must implement build")
    if type(budget) is not OptimizerBudget:
        raise TypeError("budget must be an exact OptimizerBudget")
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    if type(initial_proposal_sequence) is not int or initial_proposal_sequence < 0:
        raise ValueError("initial_proposal_sequence must be non-negative")
    if type(evaluator_concurrency) is not int or evaluator_concurrency <= 0:
        raise ValueError("evaluator_concurrency must be positive")
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if temperature is not None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) < 0
        ):
            raise ValueError("temperature must be finite and non-negative or None")
        temperature = float(temperature)
    if not callable(prompt_builder):
        raise TypeError("prompt_builder must be callable")
    if engine_trace_sink is not None and not callable(engine_trace_sink):
        raise TypeError("engine_trace_sink must be callable")
    if optimizer_trace_sink is not None and not callable(optimizer_trace_sink):
        raise TypeError("optimizer_trace_sink must be callable")
    if type(evidence_admission_policy) is not EvidenceAdmissionPolicy:
        raise TypeError("evidence_admission_policy must be an EvidenceAdmissionPolicy")

    ids = UuidIdFactory() if id_factory is None else id_factory
    if not isinstance(ids, IdFactory):
        raise TypeError("id_factory must implement IdFactory")
    active_memory = InsightMemoryBank(id_factory=ids) if memory is None else memory
    if not isinstance(active_memory, InsightMemoryBank):
        raise TypeError("memory must be an InsightMemoryBank")

    engine = AgenticEvolutionEngine(
        problem=benchmark.problem,
        generator=generator,
        id_factory=ids,
        memory=active_memory,
        seed=seed,
        initial_proposal_sequence=initial_proposal_sequence,
        evaluator_concurrency=evaluator_concurrency,
        trace_sink=engine_trace_sink,
        reward_policy=benchmark.reward.score,
        reward_definition_hash=benchmark.reward.definition_hash,
        failure_score=benchmark.reward.failure_score,
        prompt_builder=prompt_builder,
        prompt_shape_commitment_policy=prompt_shape_commitment_policy,
        reflection_row_projection=reflection_row_projection,
        reflection_workflow=reflection_workflow,
        max_output_tokens=max_output_tokens,
        structured_output_budget_policy=structured_output_budget_policy,
        temperature=temperature,
        phenotype_identity_policy=benchmark.phenotype_identity,
        detailed_evaluator=benchmark.detailed_evaluator,
        outcome_relation_binding=benchmark.outcome_relation,
        optimization_semantics=benchmark.optimization_semantics,
        objective_resolution=benchmark.objective_resolution,
        treatment_compliance_policy=treatment_compliance_policy,
    )
    runtime_planner = planner
    if planner_factory is not None:
        runtime_planner = planner_factory.build(
            benchmark=benchmark,
            engine=engine,
            id_factory=ids,
            memory=active_memory,
        )
    if not callable(getattr(runtime_planner, "plan", None)):
        raise TypeError(
            "planner_factory must return a planner implementing plan(state, budget)"
        )

    runtime_feedback_interceptor = feedback_interceptor
    if feedback_interceptor_factory is not None:
        runtime_feedback_interceptor = feedback_interceptor_factory.build(
            benchmark=benchmark,
            engine=engine,
            id_factory=ids,
            memory=active_memory,
            planner=runtime_planner,
        )
    if runtime_feedback_interceptor is not None and not isinstance(
        runtime_feedback_interceptor,
        GenerationFeedbackInterceptor,
    ):
        raise TypeError(
            "feedback_interceptor_factory must return an interceptor implementing "
            "reserve and after_generation"
        )
    # Factories receive collaborators that may themselves be mutable.  Recheck
    # every frozen benchmark identity after construction so a defective or
    # adversarial factory cannot drift domain policy between admission and use.
    benchmark.validate_binding()
    # Pass the engine's exact binding object.  In objective-only mode the engine
    # creates the default Pareto binding, so constructing another equivalent
    # value here would lose the stronger object-identity invariant.
    archive = ParetoArchive(
        benchmark.objectives,
        evidence_admission_policy=evidence_admission_policy,
        outcome_relation_binding=engine.outcome_relation_binding,
    )
    bound_planner = _BoundGenerationPlanner(benchmark, runtime_planner)
    optimizer = BudgetedAgenticOptimizer(
        engine=engine,
        archive=archive,
        planner=bound_planner,
        budget=budget,
        seed_admission_policy=seed_admission_policy,
        feedback_interceptor=runtime_feedback_interceptor,
        trace_sink=optimizer_trace_sink,
    )
    return AgenticOptimizerComposition(
        benchmark=benchmark,
        id_factory=ids,
        memory=active_memory,
        engine=engine,
        archive=archive,
        planner=runtime_planner,
        feedback_interceptor=runtime_feedback_interceptor,
        optimizer=optimizer,
    )


_LAZY_WORKLOAD_FACADE_EXPORTS = {
    "AgenticCampaignEvidenceProjections": (
        "agent_evolve.campaign_workload",
        "AgenticCampaignEvidenceProjections",
    ),
    "AgenticCampaignWorkloadConfig": (
        "agent_evolve.campaign_workload",
        "AgenticCampaignWorkloadConfig",
    ),
    "WorkloadKit": ("agent_evolve.workload_kit", "WorkloadKit"),
    "campaign_seed": ("agent_evolve.workload_kit", "campaign_seed"),
    "generic_schema_evidence_projections": (
        "agent_evolve.workload_kit",
        "generic_schema_evidence_projections",
    ),
    "WorkloadPromptExtensionView": (
        "agent_evolve.workload_prompt",
        "WorkloadPromptExtensionView",
    ),
}


def __getattr__(name: str) -> object:
    """Lazily expose workload composition types on the public façade.

    ``campaign_workload`` and ``workload_kit`` themselves depend on the core
    benchmark façade.  Lazy resolution keeps that dependency one-way during
    module initialization while letting benchmark adapters import exclusively
    from ``agent_evolve.agentic``.
    """

    target = _LAZY_WORKLOAD_FACADE_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    from importlib import import_module

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "AgenticCampaignEvidenceProjections",
    "AgenticCampaignWorkloadConfig",
    "BoundedSubprocessBoundary",
    "ChildProcessPolicy",
    "ChildProcessResult",
    "EXPLICIT_ENVIRONMENT_BOUNDARY_DEFINITION_SHA256",
    "ExplicitEnvironmentSubprocessBoundary",
    "ARCHIVE_ELITE_EXPLORER_PARENT_POLICY_DEFINITION_SHA256",
    "ARCHIVE_ELITE_EXPLORER_PARENT_POLICY_ID",
    "ARCHIVE_ELITE_EXPLORER_PARENT_POLICY_VERSION",
    "ARCHIVE_ELITE_EXPLORER_PARENT_ROTATION_LAW_ID",
    "ARCHIVE_ELITE_PARENT_POLICY_DEFINITION_SHA256",
    "ARCHIVE_ELITE_PARENT_POLICY_ID",
    "ARCHIVE_ELITE_PARENT_POLICY_VERSION",
    "FINITE_VARIATION_ELIGIBILITY_POLICY_DEFINITION_SHA256",
    "FINITE_VARIATION_ELIGIBILITY_POLICY_ID",
    "FINITE_VARIATION_ELIGIBILITY_POLICY_VERSION",
    "PORTFOLIO_MATERIALIZATION_POLICY_ID",
    "PORTFOLIO_MATERIALIZATION_POLICY_VERSION",
    "ARCHIVE_RESERVOIR_PARENT_POLICY_DEFINITION_SHA256",
    "ARCHIVE_RESERVOIR_PARENT_POLICY_ID",
    "ARCHIVE_RESERVOIR_PARENT_POLICY_VERSION",
    "ACTION_EVALUATION_REUSE_POLICY_DEFINITION_SHA256",
    "ACTION_EVALUATION_REUSE_POLICY_ID",
    "ACTION_EVALUATION_REUSE_POLICY_VERSION",
    "DURABLE_PHASE_COMMIT_POLICY_DEFINITION_SHA256",
    "DURABLE_PHASE_COMMIT_POLICY_ID",
    "DURABLE_PHASE_COMMIT_POLICY_VERSION",
    "EFFECTIVE_CHOICE_AUDIT_DEFINITION_SHA256",
    "EFFECTIVE_CHOICE_AUDIT_POLICY_ID",
    "EFFECTIVE_CHOICE_AUDIT_POLICY_VERSION",
    "DEFAULT_MAX_EXACT_PARENT_CROSSOVER_LOCI",
    "ActionEvaluationReuseMode",
    "ActionEvaluationReusePolicyBinding",
    "CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD",
    "CANONICAL_REDACTED_PORTFOLIO_EVIDENCE_SHA256",
    "ActionAllocationFrameSubsetPolicyBinding",
    "ActionAllocationRequest",
    "ActionAllocationResult",
    "ActionAllocationArmExecution",
    "ActionAllocationSurfaceAudit",
    "ActionAxisCoordinateSemantics",
    "ActionAxisSemantics",
    "ActionEvidenceCitation",
    "ActionForecastBlockPolicy",
    "ActionForecastBlockHealthSubsetBinding",
    "ActionForecastBlockRequest",
    "ActionForecastBlockResult",
    "ActionForecastBlockSpec",
    "ActionForecastAllocationFrameKind",
    "ActionForecastDraft",
    "ActionForecastEvidenceMode",
    "ActionForecastHealthPolicyBinding",
    "ActionForecastHealthSubsetPolicyBinding",
    "ActionForecastHealthFrameKind",
    "ActionForecastPartitionLayout",
    "ActionForecastPartitionPolicyBinding",
    "ActionForecastPolicy",
    "ActionForecastRequest",
    "ActionForecastResult",
    "ActionForecastArmExecution",
    "ActionForecastArmPlan",
    "ActionForecastWaveError",
    "ConcurrentActionForecastWave",
    "DurablePhaseCommitPolicyBinding",
    "DurablePhaseCommitRequirement",
    "derive_action_space_semantics",
    "ActionMetricForecast",
    "ActionPortfolioDecision",
    "ActionSpaceSemantics",
    "AgenticBenchmark",
    "AgenticCallTelemetry",
    "AgenticEvolutionEngine",
    "AgenticGenerator",
    "AgenticOptimizerComposition",
    "WorkloadKit",
    "WorkloadPromptExtensionView",
    "campaign_seed",
    "generic_schema_evidence_projections",
    "ArchiveEliteExplorerParentSelection",
    "ArchiveEliteExplorerParentSelectionReceipt",
    "ArchiveEliteExplorerParentSelector",
    "ArchiveEliteParentSelection",
    "ArchiveEliteParentSelectionReceipt",
    "ArchiveEliteParentSelector",
    "ArchiveReservoirCrowdingKind",
    "ArchiveReservoirParentSelection",
    "ArchiveReservoirParentSelectionReceipt",
    "ArchiveReservoirParentSelector",
    "ArchiveReservoirRankedCandidate",
    "AgenticTelemetryPolicy",
    "AdaptiveShuffledMateCrossoverPolicy",
    "GenerationPlannerFactory",
    "GenerationFeedbackInterceptorFactory",
    "G3CrossoverPlanPolicy",
    "AllocatedActionMember",
    "AllocationCandidateScoreDiagnostic",
    "AllocationCandidateScoreDiagnosticInput",
    "AllocationScoreDiagnostic",
    "AllocationScoreDiagnosticBinding",
    "AllocationSurfaceGatePolicyBinding",
    "AllocationSurfaceGateRejected",
    "AllocationSurfaceStepAudit",
    "ArrayIndex",
    "ArtifactRef",
    "AuditedFrameActionAllocationResult",
    "AuditedGreedyForecastFrameAllocator",
    "DeterministicIdFactory",
    "BudgetedAgenticOptimizer",
    "CandidateId",
    "CandidateDraft",
    "EliteExplorerFallbackReason",
    "EliteExplorerLaneId",
    "EliteExplorerLaneReceipt",
    "EliteExplorerLaneSource",
    "CompiledHypothesisTreatment",
    "CardScoreComponent",
    "CardTransferAdjudicationRequest",
    "CardTransferAdjudicator",
    "CardTransferScoreReceipt",
    "ContrastShardedReflectionWorkflow",
    "DetailedEvaluation",
    "DetailedEvaluationAdapter",
    "DetailedEvaluationPayload",
    "DeterministicActionAllocator",
    "EngineFiniteActionPolicy",
    "EngineFiniteActionRequest",
    "EvaluationCheck",
    "EvaluationCheckStatus",
    "EvaluationTimings",
    "EvaluatorIdentity",
    "EffectiveChoiceAuditError",
    "EffectiveChoiceAuditReceipt",
    "EligibleFiniteVariationView",
    "ExactParentCrossoverContract",
    "ExactParentCrossoverLocus",
    "ExactParentCrossoverMaterialization",
    "ExactParentCrossoverReceipt",
    "ExactParentImportPlan",
    "ExactParentLocusAttribution",
    "ExactParentSource",
    "EXACT_OBJECTIVE_RESOLUTION_DEFINITION_SHA256",
    "EXACT_OBJECTIVE_RESOLUTION_POLICY_ID",
    "EXACT_OBJECTIVE_RESOLUTION_POLICY_VERSION",
    "ExactObjectiveResolution",
    "ExecutableHypothesisTestSpec",
    "EvidenceAdmissionPolicy",
    "ExclusiveResourceLease",
    "EvolutionCandidate",
    "ExactActionMetricProjection",
    "ExactActionMetricProjectionBatch",
    "FailureCategory",
    "FailureCode",
    "FailureRecord",
    "FileExclusiveResourceLease",
    "FiniteActionEvidenceBinding",
    "FiniteActionEvaluationRequest",
    "FiniteActionEvaluationResult",
    "FiniteActionEvaluator",
    "FiniteActionEvaluatorBinding",
    "FiniteActionCardAuthority",
    "FiniteActionDecision",
    "FiniteActionOptionAuthority",
    "FiniteActionPresentationAuthority",
    "FiniteActionSetAuthority",
    "FiniteActionSetCompilationRequest",
    "FiniteActionSetCompiler",
    "FiniteActionSetDraft",
    "FiniteActionSelectorKind",
    "FiniteActionSourceMode",
    "FiniteActionSupportAuthority",
    "FiniteVariationCatalog",
    "FiniteVariationContract",
    "FiniteVariationEligibilityReceipt",
    "FiniteVariationOption",
    "FiniteVariationSelectionDraft",
    "BoundedCompositionalFiniteVariationCatalog",
    "CompositionSelectionExposure",
    "SourceUnionFiniteVariationCatalog",
    "PRIMARY_VARIATION_SOURCE_ID",
    "VARIATION_DIVERSITY_SIGNATURE_METADATA_KEY",
    "VARIATION_OPERATOR_METADATA_KEY",
    "VARIATION_SOURCE_METADATA_KEY",
    "VARIATION_SOURCE_MINIMUM_METADATA_KEY",
    "EVALUATION_SOURCE_METADATA_KEY",
    "EVALUATION_SOURCE_MINIMUM_METADATA_KEY",
    "SOURCE_UNION_POLICY_ID",
    "SOURCE_UNION_POLICY_VERSION",
    "required_source_evaluation_option_ids",
    "finite_variation_source_by_option",
    "finite_variation_source_id",
    "finite_variation_source_ids",
    "finite_variation_diversity_signature",
    "finite_variation_operator_id",
    "COMPOSITION_LEFT_OPTION_METADATA_KEY",
    "COMPOSITION_REQUIRED_PROPOSALS_METADATA_KEY",
    "COMPOSITION_RIGHT_OPTION_METADATA_KEY",
    "COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY",
    "COMPOSITIONAL_FINITE_CATALOG_POLICY_ID",
    "COMPOSITIONAL_FINITE_CATALOG_POLICY_VERSION",
    "COMPOSITE_OPTION_FAMILY",
    "HIERARCHICAL_COMPOSITIONAL_FINITE_CATALOG_POLICY_ID",
    "HIERARCHICAL_COMPOSITIONAL_FINITE_CATALOG_POLICY_VERSION",
    "MAX_MATCHED_FINITE_ACTIONS",
    "MODEL_FINITE_ACTION_SELECTOR_DEFINITION_SHA256",
    "MODEL_FINITE_ACTION_SELECTOR_POLICY_ID",
    "MODEL_FINITE_ACTION_SELECTOR_POLICY_VERSION",
    "MIN_MATCHED_FINITE_ACTIONS",
    "ForecastPortfolioUtility",
    "ForecastPortfolioUtilityBinding",
    "ForecastPortfolioUtilityInput",
    "ForecastQuantile",
    "FrameActionAllocationRequest",
    "FrameActionAllocationCommitInput",
    "FrameActionAllocationCommitRejected",
    "FrameActionAllocationTreatmentExecution",
    "FrameActionPortfolioDecision",
    "FixedStructuredOutputBudgetPolicy",
    "FrozenJsonObject",
    "FrozenJsonValue",
    "FrozenWaveReward",
    "GenerationPlan",
    "GenerationPlanner",
    "GenerationFeedbackContext",
    "GenerationFeedbackInterceptor",
    "GenerationFeedbackReceipt",
    "GenerationFeedbackReservation",
    "GenerationFeedbackResult",
    "GREEDY_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256",
    "GREEDY_RISK_DIVERSITY_ALLOCATOR_ID",
    "GREEDY_RISK_DIVERSITY_ALLOCATOR_VERSION",
    "GreedyRiskAdjustedDiversityAllocator",
    "G1ReflectionFeedbackInterceptor",
    "G1_DIAGNOSTIC_SLOT_IDS",
    "G2_SLOT_IDS",
    "G3_SLOT_IDS",
    "G3BenchmarkBoundary",
    "G3CausalScreenPlanner",
    "G3CausalScreenResultValidationReceipt",
    "G3CurationSourceScope",
    "G3ExpectedEndpoint",
    "G3ExpectedUnion",
    "G3MechanismDecision",
    "G3PostsealCurationAuthority",
    "G3PostsealCurationFactory",
    "G3PostsealCurationInterceptor",
    "G3PostsealCurationReceipt",
    "G3PostsealCurationSpec",
    "G3_POSTSEAL_CURATION_DEFINITION_SHA256",
    "HypothesisApplicabilityPort",
    "HypothesisApplicabilityStatus",
    "HypothesisCompilationReceipt",
    "HypothesisCompilationRequest",
    "G3TerminalStateValidationReceipt",
    "G3TerminalValidationError",
    "G3TerminalValidationAuthority",
    "G3_SCREEN_BUDGET",
    "FrozenDiagnosticPermutation",
    "HELD_OUT_SELECTOR_POLICY_ID",
    "HELD_OUT_SELECTOR_POLICY_VERSION",
    "HeldOutASNAssignmentCommitment",
    "HeldOutASNAssignments",
    "HeldOutASNPlanSet",
    "HeldOutASNPlannerAdapter",
    "HeldOutArm",
    "HeldOutArmAssignment",
    "HeldOutAssignmentUnavailable",
    "HeldOutAssignmentUnavailableReason",
    "HeldOutScoreMapEntry",
    "FiniteTreatmentAction",
    "IdFactory",
    "InsightEvidenceLineage",
    "InsightLifecycleChangeRequest",
    "InsightLifecycleState",
    "InsightMemoryBank",
    "InsightMemoryEntry",
    "InsightOrigin",
    "InsightRelation",
    "InsightRelationKind",
    "QuarantineTestAdmissionReceipt",
    "InsightTreatmentRequirement",
    "InsightDraft",
    "InsightId",
    "InsightRef",
    "InvocationOutcome",
    "InvocationPlan",
    "LLMCallId",
    "JsonPath",
    "MatchedFiniteActionBenchmark",
    "MatchedFiniteActionBlockFactory",
    "MatchedFiniteActionBlockPlanner",
    "MaterializedPortfolioEngine",
    "MULTI_OPTION_EVOLUTION_BUDGET",
    "MULTI_OPTION_G1_SLOT_IDS",
    "MULTI_OPTION_G2_SLOT_IDS",
    "MULTI_OPTION_G3_CORE_SLOT_IDS",
    "MULTI_OPTION_G3_CROSSOVER_SLOT_IDS",
    "MULTI_OPTION_G3_SLOT_IDS",
    "MULTI_OPTION_G3_UNION_SOURCES",
    "MultiOptionEvolutionBenchmark",
    "MultiOptionEvolutionPlanner",
    "MultiOptionEvolutionPlannerFactory",
    "OrderedTwoSeedRolePolicy",
    "ParentBoundFiniteChoice",
    "SeedRolePolicy",
    "SeedRoleSelection",
    "MutationContract",
    "MutationResponseMode",
    "CrossoverResponseMode",
    "MAX_EXACT_PARENT_CROSSOVER_LOCI",
    "MIN_EXACT_PARENT_CROSSOVER_LOCI",
    "MetricRole",
    "MetricSemantics",
    "MetricSense",
    "MetricEffectDirection",
    "MetricEffectPrediction",
    "MetricComparisonAnchor",
    "MetricComparisonAnchorKind",
    "MetricForecastScale",
    "ObjectiveParetoOutcomePolicy",
    "ObjectiveResolutionPort",
    "ObjectiveResolutionReceipt",
    "ObjectiveResolutionRequest",
    "ObjectiveResolutionResult",
    "ObjectiveSpec",
    "OptimizationSemantics",
    "ObjectKey",
    "OperatorKind",
    "OperatorInvocationId",
    "OptionPhenotypeBinding",
    "OptimizerBudget",
    "OptimizerBudgetExceeded",
    "OptimizerContractError",
    "OptimizerExecutionError",
    "OptimizerPlanningError",
    "OptimizerResult",
    "OptimizerSlot",
    "OptimizerState",
    "OutcomeComparator",
    "OutcomeRelation",
    "OutcomeRelationPolicyBinding",
    "OutcomeOrderingKind",
    "OutcomeOrderingSemantics",
    "ParetoArchive",
    "ParentMetricValue",
    "ParentBoundActionChoice",
    "POST_EVOLUTION_REFLECTION_DEFINITION_SHA256",
    "PlannedReflectionBatchCall",
    "PhenotypeIdentity",
    "PhenotypeIdentityPolicy",
    "Problem",
    "ProblemContractError",
    "PortfolioCard",
    "PortfolioCardPromptPayload",
    "PortfolioCardSourceBinding",
    "PortfolioCardSourceRegistry",
    "PortfolioCardViewReceipt",
    "PortfolioCardViewTransform",
    "PortfolioCandidateFailureEvidence",
    "PortfolioExperimentalArm",
    "PortfolioExperimentalViewReceipt",
    "PortfolioEvolution",
    "PortfolioEvolutionComposition",
    "PortfolioMemberDraft",
    "PortfolioMemberMaterializationReceipt",
    "PortfolioMemberDisposition",
    "PortfolioMemoryCreditBatchPreparation",
    "PortfolioMemoryCreditBatchReceipt",
    "PortfolioMemoryCreditPlan",
    "PortfolioMemoryCreditReceipt",
    "PortfolioMemoryContextProjectionBinding",
    "PortfolioMemoryMatchedControlWavePlan",
    "PortfolioPendingMemoryCredit",
    "EXACT_MEMORY_CONTEXT_PROJECTION_DEFINITION_SHA256",
    "PortfolioPairAttemptReceipt",
    "ArchiveAwareDisjointPairSelectionDecision",
    "ArchiveAwareDisjointParentPairPolicy",
    "FrozenArchiveBranchUtility",
    "FrozenArchiveSourcePairUtility",
    "FrozenArchiveSourceUtilityContext",
    "FrozenArchiveSourceUtilityReceipt",
    "ObservedSourceBranch",
    "PortfolioRecombination",
    "PortfolioRecombinationBranchBinding",
    "PortfolioRecombinationMemberReceipt",
    "PortfolioRecombinationNoPairReason",
    "PortfolioRecombinationNoPairReceipt",
    "PortfolioRecombinationSourceExclusionReason",
    "PortfolioRecombinationSourceExclusionReceipt",
    "PortfolioRecombinationWaveReceipt",
    "PortfolioRecombinationWaveRequest",
    "PortfolioRecombinationWaveResult",
    "bind_portfolio_recombination_source_utilities",
    "frozen_archive_source_utility_context",
    "portfolio_recombination_observed_sources",
    "PortfolioRewardAggregationBinding",
    "PortfolioSelectionPolicy",
    "PortfolioSelectionRequest",
    "PortfolioSelectionResult",
    "PortfolioVariationMemberReceipt",
    "PortfolioVariationWaveReceipt",
    "PortfolioVariationWaveRequest",
    "PortfolioVariationWaveResult",
    "PortfolioAllocationScore",
    "PostEvolutionPredecessorResolver",
    "PostEvolutionReflectionAuthority",
    "PostEvolutionReflectionFactory",
    "PostEvolutionReflectionInterceptor",
    "PostEvolutionReflectionReceipt",
    "PostEvolutionReflectionSource",
    "PostEvolutionReflectionSourceScope",
    "PostEvolutionReflectionSpec",
    "PreparedTwoStageActionEvolution",
    "PreparedHypothesisMatrix",
    "ReflectionCallExecutionError",
    "ReflectionCallReceipt",
    "ReflectionCallRequest",
    "ReflectionCallStatus",
    "ReflectionEvidenceCatalog",
    "ReflectionEvidenceCatalogEntry",
    "ReflectionPublication",
    "ReflectionPublicationResult",
    "PreparedTwoStageActionEvolutionRequest",
    "PreparedTwoStageActionEvolutionResult",
    "ProspectiveUniformRankToken",
    "REWARD_DEFINITION_HASH",
    "ReflectionGenerationRequest",
    "ReflectionGenerationResult",
    "ReflectionInsightContract",
    "ReflectionInsightKind",
    "ReflectionConsumerScope",
    "ReflectionRowProjectionBinding",
    "ReflectionPromptShard",
    "ReflectionWorkflow",
    "ReflectionWorkflowExecutionError",
    "ReflectionWorkflowRequest",
    "ReflectionWorkflowResult",
    "RankedPortfolioDecision",
    "RankedPortfolioMember",
    "REFLECTIVE_FEEDBACK_POLICY_ID",
    "REFLECTIVE_FEEDBACK_POLICY_VERSION",
    "ReflectedCard",
    "ReflectedCardBatch",
    "ReflectedCardMailbox",
    "ReflectiveFeedbackContractError",
    "ResourceConflictDetected",
    "ResourceConflictObservation",
    "ResourceConflictProbe",
    "ResourceLeaseReceipt",
    "ResourceLeaseUnavailable",
    "ResolvedActionEvidenceCitation",
    "ResolvedActionForecast",
    "ResolvedActionForecastBatch",
    "ResolvedActionForecastBlock",
    "ResolvedActionForecastHealthAssessment",
    "ResolvedActionForecastAllocationFrame",
    "ResolvedActionMetricForecast",
    "RewardPolicyBinding",
    "SelectedCardBindingMode",
    "SemanticProjectionPhenotypeIdentityPolicy",
    "StructuredOutputBudgetPolicy",
    "StructuredOutputRequestKind",
    "StrictBatchedReflectionWorkflow",
    "SCIENTIFIC_ARM_ORDER",
    "StrictTreatmentCompliancePolicy",
    "TREATMENT_COMPLIANCE_DEFINITION_SHA256",
    "TREATMENT_COMPLIANCE_POLICY_ID",
    "TREATMENT_COMPLIANCE_POLICY_VERSION",
    "TelemetryGatedAgenticGenerator",
    "TaskKeyedUniformFiniteActionPolicy",
    "TaskKeyedArchiveEliteExplorerParentPolicy",
    "TaskKeyedArchiveEliteParentPolicy",
    "TaskKeyedArchiveReservoirParentPolicy",
    "TreatmentAdmissionReceipt",
    "TreatmentAdmissionRequest",
    "TreatmentActionBinding",
    "TreatmentAssignmentRole",
    "TreatmentClaimMode",
    "TreatmentCompliancePolicy",
    "TreatmentComplianceRejected",
    "TreatmentComplianceViolation",
    "TreatmentInsightEvidence",
    "TreatmentInsightBinding",
    "TreatmentPreflightReceipt",
    "TreatmentPreflightRequest",
    "TwoStageActionPhase",
    "TwoStageActionPhaseCommit",
    "TwoStageActionPhaseCommitError",
    "TwoStageActionPhaseCommitSink",
    "TwoStageActionPhaseReceipt",
    "validate_treatment_admission_receipt",
    "validate_treatment_preflight_receipt",
    "validate_effective_choice_audit_receipt",
    "TypedConfigurationPhenotypeIdentityPolicy",
    "UNIFORM_FINITE_ACTION_POLICY_DEFINITION_SHA256",
    "UNIFORM_FINITE_ACTION_POLICY_ID",
    "UNIFORM_FINITE_ACTION_POLICY_VERSION",
    "UuidIdFactory",
    "ValidationOutcome",
    "VariationGenerationRequest",
    "VariationGenerationResult",
    "artifact_ref_for_bytes",
    "audit_effective_choice_plan",
    "allocation_score_multiset_sha256",
    "allocation_surface_failure_codes",
    "admit_portfolio_card_sources",
    "bind_portfolio_experimental_view",
    "bind_finite_action_evidence",
    "bind_finite_variation_catalog",
    "bind_action_forecast_block_allocation_frame",
    "bind_action_forecast_block_subset_allocation_frame",
    "bind_complete_action_forecast_allocation_frame",
    "build_exact_parent_import_plan",
    "assess_resolved_action_forecast_health",
    "assess_resolved_action_forecast_block_health",
    "assess_resolved_action_forecast_block_subset_health",
    "build_action_forecast_block_requests",
    "build_action_forecast_partition_layout",
    "build_frame_action_allocation_phase_commit",
    "build_reflected_card_batch",
    "build_g3_postseal_curation_reservation",
    "compile_and_seal_finite_action_set",
    "compile_registered_hypothesis_treatment",
    "compose_agentic_optimizer",
    "compose_portfolio_evolution",
    "canonical_candidate_path_text",
    "default_parent_relative_reward",
    "default_evidence_prompt",
    "derive_portfolio_card_view",
    "derive_exact_parent_crossover_contract",
    "eligible_finite_variation_view",
    "exact_configuration_phenotype_bindings",
    "exact_parent_import_exclusions_sha256",
    "freeze_json",
    "frame_source_call_and_request_identity",
    "generation_feedback_receipt_hash",
    "objective_pareto_outcome_binding",
    "objective_resolution_policy_metadata",
    "portfolio_card_snapshot_sha256",
    "portfolio_card_score_state_sha256",
    "portfolio_card_from_insight_entry",
    "project_action_neutral_insight_prompt_payload",
    "portfolio_selection_telemetry_sha256",
    "optional_phase_commit_policy",
    "per_arm_evaluation_reuse_policy",
    "required_scientific_phase_commit_policy",
    "reflection_contrast_id",
    "register_neutral_sham_card",
    "render_optimization_semantics",
    "resolve_structured_output_budget",
    "resolve_objectives",
    "render_action_space_semantics",
    "resolve_action_forecast_block",
    "resolve_action_forecasts",
    "resolve_finite_variation_selection",
    "resolve_ranked_portfolio_decision",
    "seal_generation_feedback",
    "thaw_json",
    "typed_json_sha256",
    "validate_generation_feedback_receipt",
    "validate_g3_causal_screen_result",
    "validate_g3_terminal_state",
    "validate_action_portfolio_decision",
    "validate_frame_action_portfolio_decision",
    "validate_frame_action_allocation_phase_commit",
    "validate_finite_action_set_compiler_identity",
    "validate_hypothesis_compilation",
    "validate_hypothesis_compiler_identity",
    "validate_treatment_occurrence_frame_request",
    "validate_card_transfer_score_receipt",
    "validate_portfolio_experimental_view",
    "validate_reflection_insight_draft",
    "validate_reflection_evidence_catalog_result",
    "validate_archive_elite_explorer_parent_selection",
    "validate_archive_elite_parent_selection",
    "validate_archive_reservoir_parent_selection",
    "validate_ranked_portfolio_decision",
    "validate_resolved_action_forecasts",
    "lenient_action_forecast_health_policy",
    "materialized_disjoint_invocation",
    "materialize_exact_parent_crossover",
    "materialized_finite_action_decision",
    "finite_action_mutation_boundary",
    "model_finite_action_telemetry_sha256",
    "seal_model_finite_action_decision",
    "replay_exact_parent_crossover",
    "resolve_exact_parent_import_for_target",
    "validate_finite_action_decision",
    "validate_exact_parent_import_exclusions",
    "BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_DEFINITION_SHA256",
    "BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_ID",
    "BOUNDED_PORTFOLIO_MEMORY_DOSE_POLICY_VERSION",
    "BoundedPortfolioMemoryDoseContract",
    "IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_DEFINITION_SHA256",
    "IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_ID",
    "IDENTIFIABLE_REFLECTION_EVIDENCE_POLICY_VERSION",
    "IdentifiableMutationReflectionContrast",
    "IdentifiableReflectionEvidenceSnapshot",
    "MAX_REFLECTION_LOCAL_INTERVENTION_VALUE_BYTES",
    "EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256",
    "EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID",
    "EXACT_PARENT_PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION",
    "PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_DEFINITION_SHA256",
    "PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_ID",
    "PORTFOLIO_MEMORY_CONTEXT_TRANSFER_POLICY_VERSION",
    "PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_DEFINITION_SHA256",
    "PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_ID",
    "PORTFOLIO_MEMORY_DOSE_SUPPORT_POLICY_VERSION",
    "PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_DEFINITION_SHA256",
    "PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_ID",
    "PORTFOLIO_MEMORY_TRANSFER_LADDER_POLICY_VERSION",
    "PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_DEFINITION_SHA256",
    "PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_ID",
    "PORTFOLIO_MEMORY_TRANSFER_LANE_RESOLVER_VERSION",
    "CONTEXTUAL_SEARCH_CONTROLLER_DEFINITION_SHA256",
    "CONTEXTUAL_SEARCH_CONTROLLER_ID",
    "CONTEXTUAL_SEARCH_CONTROLLER_VERSION",
    "PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_DEFINITION_SHA256",
    "PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_ID",
    "PORTFOLIO_OPTIMIZATION_MEMORY_POLICY_VERSION",
    "PortfolioMemoryDoseAssessment",
    "CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_KEY",
    "CAMPAIGN_SELECTOR_CONTEXT_EXTENSION_MAX_BYTES",
    "CampaignSelectorContextExtension",
    "attach_campaign_selector_context_extension",
    "resolve_campaign_selector_context_extension",
    "PortfolioMemoryContextTransferAssessment",
    "PortfolioMemoryDoseCardSemantics",
    "PortfolioMemoryDoseSupportScope",
    "PortfolioMemoryTransferLadderAssessment",
    "PortfolioMemoryTransferTier",
    "PortfolioMemoryTransferCard",
    "PortfolioMemoryTransferLane",
    "PortfolioMemoryTransferLaneResolution",
    "PortfolioMemoryTransferLaneResolver",
    "ContextualArmAllocation",
    "ContextualArmPosterior",
    "CampaignContextualJointCapabilityProjector",
    "CampaignContextualPlanningContext",
    "CampaignContextualSearchPlan",
    "CampaignContextualSearchPlanner",
    "FiniteContractContextualJointCapabilityProjector",
    "CampaignPortfolioFrontierTarget",
    "CampaignPortfolioFrontierTargetAllocator",
    "ContextualPortfolioAllocationSlice",
    "ContextualSearchCompletionAudit",
    "ContextualSearchDecision",
    "ContextualSearchDelayedCredit",
    "ContextualSearchLedger",
    "ContextualSearchObservation",
    "ContextualSearchQuery",
    "ContextualSearchSnapshot",
    "ContextualSearchStageAllocation",
    "PhaseAwareContextualSearchController",
    "SearchArmKind",
    "SearchPhase",
    "audit_completed_contextual_search_ledger",
    "slice_contextual_search_decision",
    "CONTEXTUAL_PORTFOLIO_OUTCOME_DEFINITION_SHA256",
    "CONTEXTUAL_PORTFOLIO_OUTCOME_POLICY_ID",
    "CONTEXTUAL_PORTFOLIO_OUTCOME_POLICY_VERSION",
    "ContextualPortfolioOutcomeBatch",
    "observe_contextual_portfolio_outcomes",
    "CONTEXTUAL_DELAYED_CREDIT_DEFINITION_SHA256",
    "CONTEXTUAL_DELAYED_CREDIT_POLICY_ID",
    "CONTEXTUAL_DELAYED_CREDIT_POLICY_VERSION",
    "ContextualPostRecombinationCreditBatch",
    "ContextualTerminalPersistenceCreditBatch",
    "observe_contextual_post_recombination_credit",
    "observe_contextual_terminal_persistence",
    "ContextualPortfolioAllocationContract",
    "ContextualPortfolioAllocationRealization",
    "ContextualArmCountCapability",
    "ContextualArmCountCapabilityWitness",
    "ContextualJointCountVector",
    "ContextualLaneJointCountCapability",
    "PortfolioMemoryDoseCardSupport",
    "PortfolioMemoryDoseMember",
    "PortfolioMemoryDoseRejected",
    "PortfolioMemoryDoseStage",
    "PortfolioMemoryDoseViolation",
    "PortfolioOptimizationMemoryAssessment",
    "PortfolioOptimizationMemoryDirective",
    "PortfolioOptimizationMemoryDisposition",
    "PortfolioOptimizationMetricSign",
    "PortfolioMemoryExposureScope",
    "ReflectionEvidenceExclusionReason",
    "ReflectionFalsificationFeedback",
    "SealedCutoffDelayedAdmissionCadence",
    "assess_evaluated_portfolio_memory_dose",
    "assess_portfolio_memory_context_transfer",
    "assess_portfolio_memory_transfer_ladder",
    "assess_portfolio_optimization_memory",
    "assess_proposed_portfolio_memory_dose",
    "derive_portfolio_memory_dose_card_support",
    "derive_portfolio_memory_advisory_card_support",
    "project_identifiable_reflection_evidence",
    "require_passing_portfolio_memory_dose",
]
