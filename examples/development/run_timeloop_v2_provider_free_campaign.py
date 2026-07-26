#!/usr/bin/env python3
"""Provider-free G6 proof of Timeloop's production AgentEvolve bridge.

This is a structural and scientific-contract conformance run, not an efficacy
claim.  A deterministic typed-output double replaces provider transport and a
cheap objective function replaces Docker, while the production-generic path
still executes:

* six alternating portfolio/recombination generations;
* calibrated one-call K8 proposal followed by engine-owned k4 evaluation in
  the complete-contract reference treatment;
* G1 direct single-mutation evidence and a single G2 reflection;
* sealed-cutoff delayed admission at G4 and first memory consumption at G5;
* compatibility-aware, bounded one-card dose per G5 parent lane;
* no terminal reflection and typed no-yield recourse when no full matching
  exists.

The same runner boundary accepts the detailed Timeloop benchmark and pinned
Docker evaluator.  No provider credential is read anywhere in this module.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.agentic import (  # noqa: E402
    AgenticBenchmark,
    BoundedPortfolioMemoryDoseContract,
    DeterministicIdFactory,
    InsightDraft,
    InsightMemoryBank,
    MetricComparisonAnchor,
    MetricComparisonAnchorKind,
    MetricEffectPrediction,
    PortfolioCard,
    PortfolioMemoryContextProjectionBinding,
    PortfolioMemoryDoseCardSemantics,
    PortfolioMemoryMatchedControlWavePlan,
    PortfolioRewardAggregationBinding,
    PortfolioSelectionRequest,
    ReflectionConsumerScope,
    ReflectionInsightKind,
    TypedConfigurationPhenotypeIdentityPolicy,
    admit_portfolio_card_sources,
    compose_portfolio_evolution,
    freeze_json,
    portfolio_card_from_insight_entry,
    project_action_neutral_insight_prompt_payload,
    typed_json_sha256,
)
from agent_evolve.application.budgeted_optimizer import OptimizerBudget  # noqa: E402
from agent_evolve.application.calibrated_campaign import (  # noqa: E402
    CalibratedCampaignBindingFactory,
    equal_weight_slate_objectives,
)
from agent_evolve.application.campaign_evidence_registry import (  # noqa: E402
    CampaignEvidenceRegistry,
)
from agent_evolve.application.campaign_diagnostic_blocks import (  # noqa: E402
    CampaignDiagnosticCompatibilityAudit,
    CampaignDiagnosticCompleteSupportCohortSelection,
    CampaignDiagnosticSupportCardInput,
    CampaignDiagnosticSupportLaneInput,
    CampaignDiagnosticSingletonBlock,
)
from agent_evolve.application.campaign_execution import (  # noqa: E402
    CampaignExecutionEvent,
    CampaignExecutionResult,
    CampaignJournalAck,
    EvolutionCampaignScheduler,
)
from agent_evolve.application.campaign_generation_audit import (  # noqa: E402
    TransactionalPortfolioGenerationAuditor,
)
from agent_evolve.application.campaign_learning import (  # noqa: E402
    ClosedLoopCampaignLearning,
)
from agent_evolve.application.campaign_learning_runtime import (  # noqa: E402
    CampaignDiagnosticExposureReceipt,
    CampaignReflectionLearningRecordCodec,
    ClosedLoopCampaignLearningRuntime,
    StructuredCampaignReflectionLearningProjector,
)
from agent_evolve.application.campaign_contextual_outcomes import (  # noqa: E402
    ContextualOutcomeCampaignEnricher,
)
from agent_evolve.application.campaign_variation_trace import (  # noqa: E402
    summarize_campaign_variation_trace,
)
from agent_evolve.application.evolution_campaign import (  # noqa: E402
    ArchiveUtilitySnapshot,
    CampaignAgentRuntimeReceipt,
    CampaignConcurrency,
    CampaignPolicies,
    CampaignPolicyBinding,
    CampaignProtocol,
    CampaignReflectionSupervisionPolicy,
    EvolutionCampaign,
    ReflectionFailureMode,
    SealedCutoffDelayedAdmissionCadence,
    TerminalReflectionPolicy,
)
from agent_evolve.application.finite_action_hypothesis_semantics import (  # noqa: E402
    PortableFiniteActionHypothesisMatcher,
    PortableFiniteActionInsightSemanticCompiler,
)
from agent_evolve.application.identifiable_reflection_evidence import (  # noqa: E402
    IdentifiableMutationReflectionHypothesisCluster,
    cluster_identifiable_mutation_reflection_hypotheses,
)
from agent_evolve.application.portfolio_campaign_runtime import (  # noqa: E402
    AgenticPortfolioCampaignRuntime,
    ArchiveEliteExplorerCampaignParentSelector,
    ArchiveReservoirCampaignParentSelector,
    ResidualHypervolumeCampaignParentSelector,
    StagnationAwareDiverseCampaignParentSelector,
    CampaignIdentifiableReflectionInput,
    CampaignPortfolioMemoryEstimandProjection,
    CampaignPortfolioWaveContext,
    CommittedRegistryIdentifiableReflectionEvidenceSource,
)
from agent_evolve.application.portfolio_evolution import (  # noqa: E402
    PortfolioVariationWaveRequest,
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
from agent_evolve.application.portfolio_memory_matched_control import (  # noqa: E402
    PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256,
    PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID,
    PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION,
    PortfolioMemoryMatchedControlPlanner,
    PortfolioMemoryMatchedSupportResolver,
    materialize_portfolio_memory_matched_arm,
)
from agent_evolve.domain.typed_json import (  # noqa: E402
    FrozenJsonObject,
    thaw_json,
)
from agent_evolve.domain.insight import InsightRef  # noqa: E402
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_campaign import (  # noqa: E402
    CalibratedPortfolioCampaignCoordinator,
)
from agent_evolve.integrations.pydantic_ai.model_execution_profile import (  # noqa: E402
    DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
    OpenRouterModelExecutionProfile,
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
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_selection import (  # noqa: E402
    CalibratedPortfolioAllocator,
    CalibratedPortfolioFeasibilityWitnessMode,
    PydanticAIContextualSearchAllocationPortfolioSelectionPolicy,
    PydanticAITargetConditionedCalibratedPortfolioSelectionPolicy,
    calibrated_portfolio_prompt_definition_sha256,
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
)
from agent_evolve.policies.memory.compatibility_matching import (  # noqa: E402
    LaneCardMatchingCard,
    LaneCardMatchingLane,
)
from agent_evolve.policies.memory.balanced_subset_blocks import (  # noqa: E402
    StableMemoryAssignmentUnit,
)
from agent_evolve.policies.memory.staged_causal import (  # noqa: E402
    insight_selection_decision_sha256,
)
from agent_evolve.policies.reward.affine_hypervolume_3d import (  # noqa: E402
    AffineFrozenArchiveJointWaveReward3D,
    AffineHypervolumeArchiveUtility3D,
)
from agent_evolve.policies.reward.contextual_marginal_utility import (  # noqa: E402
    FixedReferenceContextualMarginalUtilityProjector,
)
from agent_evolve.policies.memory.global_falsification import (  # noqa: E402
    HypothesisAuditScope,
)
from agent_evolve.policies.selection.forecast_calibration import (  # noqa: E402
    ForecastCalibrationScope,
)
from agent_evolve.policies.selection.meaningful_direction import (  # noqa: E402
    AbsoluteToleranceDirectionAdjudicator,
    MetricDirectionResolution,
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
from agent_evolve.policies.selection.frontier_probe_slate import (  # noqa: E402
    FrontierProbeSlatePolicy,
)
from agent_evolve.policies.selection.target_conditioned_allocator import (  # noqa: E402
    TargetConditionedSlateAllocatorAdapter,
)
from agent_evolve.policies.selection.structural_posterior_slate import (  # noqa: E402
    build_controller_owned_family_exposure_phases,
    build_terminal_tapered_family_exposure_phases,
)
from agent_evolve.ports.agentic_generator import (  # noqa: E402
    AgenticCallTelemetry,
    ReflectionGenerationResult,
)
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)
from agent_evolve.ports.portfolio_selection import (  # noqa: E402
    pairwise_disjoint_parent_patch_witness,
)
from examples.benchmarks.timeloop_codesign.v2.campaign_reflection import (  # noqa: E402
    OBJECTIVE_IDS,
    REFLECTION_DECISION_PATHS,
    build_timeloop_v2_identifiable_learning_envelope,
    build_timeloop_v2_identifiable_reflection_request,
)
from examples.benchmarks.timeloop_codesign.v2.affine_utility import (  # noqa: E402
    timeloop_v2_affine_hypervolume_spec,
)
from examples.benchmarks.timeloop_codesign.v2.campaign_workload import (  # noqa: E402
    WORKLOAD_DEFINITION_SHA256,
    compose_timeloop_v2_campaign_workload,
)
from examples.benchmarks.timeloop_codesign.v2.candidate import (  # noqa: E402
    normalize_candidate,
)
from examples.benchmarks.timeloop_codesign.v2.detailed_evaluation import (  # noqa: E402
    timeloop_v2_optimization_semantics,
)
from examples.benchmarks.timeloop_codesign.v2.evaluator import (  # noqa: E402
    TimeloopV2Settings,
)
from examples.benchmarks.timeloop_codesign.v2.finite_variation_catalog import (  # noqa: E402
    TimeloopV2FiniteVariationCatalog,
)
from examples.benchmarks.timeloop_codesign.v2.frozen_panels import (  # noqa: E402
    frozen_network_panel,
)
from examples.benchmarks.timeloop_codesign.v2.problem_def import (  # noqa: E402
    TimeloopV2CoDesignProblem,
)


OUTER_SEED = 20260717
GENERATION_COUNT = 6
PARENTS_PER_PORTFOLIO = 2
CALIBRATED_PROPOSAL_WIDTH = 8
RECOMBINATIONS_PER_PARENT = 2
PLANNED_LOGICAL_CALLS = 7
MAX_OUTPUT_TOKENS = 384_000
TEMPERATURE = 0.0
FEASIBILITY_WITNESS_MODE = CalibratedPortfolioFeasibilityWitnessMode(
    os.environ.get("AGENT_EVOLVE_FEASIBILITY_WITNESS_MODE", "canonical")
)
ARCHIVE_CONTEXT_MODE = AffineFrontierContextMode(
    os.environ.get("AGENT_EVOLVE_ARCHIVE_CONTEXT_MODE", "off")
)
MEMORY_AGGREGATION_ID = "affine_archive_joint_wave_gain"
MEMORY_AGGREGATION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:affine-archive-joint-wave-gain:v1"
).hexdigest()
_TARGET_CONDITIONED_FREEZE_PATH = (
    AGENT_EVOLVE_ROOT.parent
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "trap_portable_profile_v1.json"
)


def _target_conditioned_specification() -> TargetConditionedCampaignSpecification:
    return TargetConditionedCampaignSpecification.from_freeze_record(
        json.loads(_TARGET_CONDITIONED_FREEZE_PATH.read_text(encoding="utf-8"))
    )


def _common_pool_enabled() -> bool:
    return FEASIBILITY_WITNESS_MODE is (
        CalibratedPortfolioFeasibilityWitnessMode.TASK_KEYED_COMMON_POOL
    )


ACQUISITION_MODE = CampaignAcquisitionMode(
    os.environ.get(
        "AGENT_EVOLVE_ACQUISITION_MODE",
        "model_top_k" if _common_pool_enabled() else "full_support",
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


def _proposal_support_policy():
    return build_campaign_proposal_support_policy(
        ACQUISITION_MODE,
        common_pool_enabled=_common_pool_enabled(),
    )


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


PORTFOLIO_WIDTH = 4 if _common_pool_enabled() else 8
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
PLANNED_CANDIDATE_OCCURRENCES = 2 + PARENTS_PER_PORTFOLIO * (
    3 * PORTFOLIO_WIDTH + 3 * RECOMBINATIONS_PER_PARENT
)
_CAMPAIGN_SHA256 = hashlib.sha256(
    b"agent-evolve:timeloop-v2-provider-free-g6-delayed-identifiable:v1"
).hexdigest()
_EVALUATOR_CONTRACT_SHA256 = hashlib.sha256(
    b"agent-evolve:timeloop-v2-provider-free-evaluator:v2"
).hexdigest()
_METRIC_ADJUDICATOR_SHA256 = hashlib.sha256(
    b"agent-evolve:timeloop-v2-objective-delta-adjudicator:v1"
).hexdigest()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("Timeloop provider-free record is not an object")
    return frozen


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


def _memory_trial_record(value: object) -> dict[str, object]:
    return {
        "credit_unit_id": value.credit_unit_id.value,
        "candidate_ids": [item.value for item in value.candidate_ids],
        "reward_definition_sha256": value.reward_definition_hash,
        "selection_decision_sha256": insight_selection_decision_sha256(value.decision),
        "reward_hex": value.reward.hex(),
        "treatment_binding_sha256": value.treatment_binding_sha256,
        "generation": value.generation,
    }


def _memory_transition_record(value: object) -> dict[str, object]:
    return {
        "sequence": value.sequence,
        "reference": {
            "insight_id": value.reference.insight_id.value,
            "version": value.reference.version,
        },
        "prior_state": value.prior_state.value,
        "new_state": value.new_state.value,
        "reason": value.reason,
        "supporting_evidence": list(value.supporting_evidence),
    }


@dataclass(frozen=True, slots=True)
class _Evaluation:
    objective_values: dict[str, float]


class _DeterministicEvaluator:
    """Cheap outcome-varying double behind the real Timeloop problem port."""

    _DATAFLOW = {
        "weight_stationary": 0.92,
        "output_stationary": 0.96,
        "row_stationary": 0.90,
        "no_local_reuse": 1.18,
    }
    _UTILIZATION = {"low": 1.22, "medium": 1.08, "high": 0.98, "full": 0.94}
    _RESIDENCY = {
        "balanced": 1.00,
        "input_reuse": 0.96,
        "weight_reuse": 0.93,
        "output_reuse": 0.95,
    }

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, config: object) -> _Evaluation:
        self.calls += 1
        candidate = normalize_candidate(config)
        policies = tuple(
            getattr(candidate, f"policy_cluster_{index}") for index in range(3)
        )
        dataflow = sum(self._DATAFLOW[value.dataflow_family] for value in policies) / 3
        utilization = (
            sum(self._UTILIZATION[value.spatial_utilization] for value in policies) / 3
        )
        residency = (
            sum(self._RESIDENCY[value.buffer_residency_bias] for value in policies) / 3
        )
        axis_diversity = len({value.primary_spatial_axis for value in policies})
        loop_diversity = len({value.outer_loop_order for value in policies})
        buffer_bits = candidate.global_buffer_depth * candidate.global_buffer_width
        precision = candidate.datawidth_bits / 8.0
        register_factor = 0.96 if candidate.register_enabled else 1.05
        energy = (
            0.0025
            * dataflow
            * residency
            * precision
            * register_factor
            * (1.0 + candidate.pe_mesh_x / 96.0)
            * (1.0 + 65_536.0 / buffer_bits)
        )
        latency = (
            0.14
            * utilization
            * dataflow
            * (8.0 / candidate.pe_mesh_x)
            * (1.0 - 0.015 * (axis_diversity - 1))
            * (1.0 - 0.01 * (loop_diversity - 1))
        )
        area = 1.0e-6 * (
            0.6 * candidate.pe_mesh_x * candidate.datawidth_bits
            + buffer_bits / 128.0
            + (24.0 if candidate.register_enabled else 0.0)
        )
        return _Evaluation(
            objective_values={
                "energy_joules": float(energy),
                "latency_seconds": float(latency),
                "area_square_meters": float(area),
            }
        )


class _NeverGenerator:
    async def propose(self, request):  # pragma: no cover - materialized only.
        raise AssertionError(f"unexpected model-authored candidate: {request}")

    async def reflect(self, request):  # pragma: no cover - external executor.
        raise AssertionError(f"unexpected engine reflection: {request}")


def _option_locus(option: object) -> str:
    metadata = dict(option.metadata)
    locus = metadata.get("locus")
    if type(locus) is not str or not locus:
        raise RuntimeError("Timeloop option omitted its exact locus")
    return locus


class _ProviderFreeCalibratedRunner:
    """Produce a schema-valid K8 without provider transport or hidden outcomes."""

    def __init__(self) -> None:
        self.calls = 0
        self.records: list[dict[str, object]] = []

    @staticmethod
    def _proposal_options(output_type: type) -> tuple[object, ...]:
        contract = output_type.finite_variation_contract
        common_pool = output_type.ordered_common_pool_option_ids
        eligible_options = (
            tuple(contract.resolve(option_id) for option_id in common_pool)
            if common_pool is not None
            else contract.options
        )
        dose = output_type.memory_dose_contract
        required_composites = getattr(
            output_type,
            "required_composite_proposals",
            0,
        )
        if type(required_composites) is not int or not 0 <= required_composites < 8:
            raise RuntimeError("Timeloop output schema published an invalid hierarchy")
        if (
            required_composites == 0
            and common_pool is not None
            and len(eligible_options) == (CALIBRATED_PROPOSAL_WIDTH)
        ):
            if dose is None:
                return eligible_options
            support = dose.card_supports[0]
            supported = next(
                (
                    option
                    for option in eligible_options
                    if support.supports(
                        option.option_id,
                        option.identity_sha256,
                    )
                ),
                None,
            )
            if supported is None:
                raise RuntimeError(
                    "Timeloop common pool omitted its required memory-dose action"
                )
            return (supported,) + tuple(
                option for option in eligible_options if option != supported
            )
        composite_options = tuple(
            option
            for option in eligible_options
            if dict(option.metadata).get("composition_selection_exposure")
            == "hierarchical_ranked_union"
        )
        atomic_options = tuple(
            option for option in eligible_options if option not in composite_options
        )
        if required_composites:
            if len(composite_options) < required_composites:
                raise RuntimeError(
                    "Timeloop hierarchy lacks its required composite proposal stratum"
                )
            required_ids = set(output_type.required_proposal_support_option_ids)
            required_options = tuple(
                option
                for option in eligible_options
                if option.option_id in required_ids
            )
            if {option.option_id for option in required_options} != required_ids:
                raise RuntimeError(
                    "Timeloop proposal support escaped its candidate pool"
                )
            supported = None
            if dose is not None:
                support = dose.card_supports[0]
                supported = next(
                    (
                        option
                        for option in eligible_options
                        if support.supports(option.option_id, option.identity_sha256)
                    ),
                    None,
                )
                if supported is None:
                    raise RuntimeError(
                        "Timeloop hierarchy omitted its required memory-dose action"
                    )

            atomic_capacity = CALIBRATED_PROPOSAL_WIDTH - required_composites
            mandatory_composites = tuple(
                option for option in required_options if option in composite_options
            )
            mandatory_atomics = tuple(
                option for option in required_options if option in atomic_options
            )
            if supported is not None:
                if (
                    supported in composite_options
                    and supported not in mandatory_composites
                ):
                    mandatory_composites = (*mandatory_composites, supported)
                elif supported in atomic_options and supported not in mandatory_atomics:
                    mandatory_atomics = (*mandatory_atomics, supported)
            if len(mandatory_composites) > required_composites:
                raise RuntimeError("Timeloop hierarchy over-subscribed composite slots")
            if len(mandatory_atomics) > atomic_capacity:
                raise RuntimeError("Timeloop hierarchy over-subscribed atomic slots")

            family_bounds = output_type.required_evaluation_family_bounds
            if family_bounds:
                witness_ids = pairwise_disjoint_parent_patch_witness(
                    contract,
                    tuple(option.option_id for option in eligible_options),
                    portfolio_size=output_type.evaluation_portfolio_size,
                    min_distinct_families=output_type.min_distinct_families,
                    family_exposure_bounds=family_bounds,
                )
                if witness_ids is None:
                    raise RuntimeError(
                        "Timeloop hierarchy has no family-bounded evaluation witness"
                    )
                by_id = {option.option_id: option for option in eligible_options}
                chosen: list[object] = []
                for option in (
                    *mandatory_composites,
                    *mandatory_atomics,
                    *(by_id[option_id] for option_id in witness_ids),
                ):
                    if option not in chosen:
                        chosen.append(option)
                chosen_composites = sum(
                    option in composite_options for option in chosen
                )
                chosen_atomics = len(chosen) - chosen_composites
                if (
                    chosen_composites > required_composites
                    or chosen_atomics > atomic_capacity
                ):
                    raise RuntimeError(
                        "Timeloop mandatory proposal support conflicts with the "
                        "family-bounded evaluation witness"
                    )
                for option in composite_options:
                    if chosen_composites == required_composites:
                        break
                    if option not in chosen:
                        chosen.append(option)
                        chosen_composites += 1
                for option in atomic_options:
                    if chosen_atomics == atomic_capacity:
                        break
                    if option not in chosen:
                        chosen.append(option)
                        chosen_atomics += 1
                if (
                    chosen_composites != required_composites
                    or chosen_atomics != atomic_capacity
                    or len(chosen) != CALIBRATED_PROPOSAL_WIDTH
                ):
                    raise RuntimeError(
                        "Timeloop family-bounded witness could not fill its exact K8: "
                        f"required_composites={required_composites}, "
                        f"atomic_capacity={atomic_capacity}, "
                        f"eligible_composites={len(composite_options)}, "
                        f"eligible_atomics={len(atomic_options)}, "
                        f"chosen_composites={chosen_composites}, "
                        f"chosen_atomics={chosen_atomics}, "
                        f"chosen_total={len(chosen)}, "
                        f"mandatory_composites={len(mandatory_composites)}, "
                        f"mandatory_atomics={len(mandatory_atomics)}, "
                        f"witness_size={len(witness_ids)}"
                    )
                selected = tuple(chosen)
                if supported is not None:
                    selected = (supported,) + tuple(
                        option for option in selected if option != supported
                    )
                return selected

            selected_atomics = list(mandatory_atomics)
            used_loci = {_option_locus(option) for option in selected_atomics}
            used_families = {option.family for option in selected_atomics}
            # Keep an engine-checkable K4 atomic witness in every deterministic
            # conformance slate.  Composite members are proposal assays, not a
            # hidden relaxation of evaluator feasibility.
            for option in atomic_options:
                locus = _option_locus(option)
                if (
                    option in selected_atomics
                    or locus in used_loci
                    or option.family in used_families
                ):
                    continue
                selected_atomics.append(option)
                used_loci.add(locus)
                used_families.add(option.family)
                if len(selected_atomics) >= max(4, len(mandatory_atomics)):
                    break
            for option in atomic_options:
                if len(selected_atomics) == atomic_capacity:
                    break
                if option not in selected_atomics:
                    selected_atomics.append(option)
            selected_composites = list(mandatory_composites)
            for option in composite_options:
                if len(selected_composites) == required_composites:
                    break
                if option not in selected_composites:
                    selected_composites.append(option)
            if (
                len(selected_atomics) != atomic_capacity
                or len(selected_composites) != required_composites
            ):
                raise RuntimeError("Timeloop hierarchy could not fill its exact K8")
            selected = (*selected_atomics, *selected_composites)
            if supported is not None:
                selected = (supported,) + tuple(
                    option for option in selected if option != supported
                )
            return tuple(selected)
        chosen: list[object] = []
        used_loci: set[str] = set()
        used_families: set[str] = set()
        if dose is not None:
            support = dose.card_supports[0]
            eligible_by_id = {option.option_id: option for option in eligible_options}
            supported = next(
                (
                    eligible_by_id[option_id]
                    for option_id, _ in support.compatible_options
                    if option_id in eligible_by_id
                ),
                None,
            )
            if supported is None:
                raise RuntimeError(
                    "Timeloop common pool omitted its required memory-dose action"
                )
            chosen.append(supported)
            used_loci.add(_option_locus(supported))
            used_families.add(supported.family)
        for option in eligible_options:
            locus = _option_locus(option)
            if option in chosen or locus in used_loci or option.family in used_families:
                continue
            chosen.append(option)
            used_loci.add(locus)
            used_families.add(option.family)
            if len(chosen) == CALIBRATED_PROPOSAL_WIDTH:
                break
        # Family diversity is a preference of this deterministic conformance
        # double, not a production finite-contract invariant.  A memory-bound
        # supported member can consume the only eligible locus in one family;
        # finish from distinct loci rather than manufacturing a false
        # no-eight-options failure.
        for option in eligible_options:
            if len(chosen) == CALIBRATED_PROPOSAL_WIDTH:
                break
            locus = _option_locus(option)
            if option in chosen or locus in used_loci:
                continue
            chosen.append(option)
            used_loci.add(locus)
        if len(chosen) != CALIBRATED_PROPOSAL_WIDTH:
            raise RuntimeError("Timeloop palette did not yield eight disjoint loci")
        return tuple(chosen)

    @staticmethod
    def _with_required_proposal_support(
        output_type: type,
        options: tuple[object, ...],
    ) -> tuple[object, ...]:
        """Honor engine-reserved K8 support without disturbing its K4 witness.

        This deterministic transport double keeps its first four options as a
        feasible, pairwise-disjoint evaluator witness (and keeps a memory-dose
        action at index zero when assigned).  Any missing proposal-only support
        reservations replace tail exploration positions.  Production model
        ranking remains unconstrained; this is solely conformance-fixture
        behavior required by the production output schema.
        """

        required = tuple(sorted(output_type.required_proposal_support_option_ids))
        if not required:
            return options
        option_ids = tuple(option.option_id for option in options)
        missing = tuple(value for value in required if value not in option_ids)
        if not missing:
            return options
        allowed = output_type.allowed_common_pool_option_ids
        if allowed is not None and not set(missing).issubset(allowed):
            raise RuntimeError(
                "Timeloop proposal support escaped its sealed common pool"
            )
        replacement_slots = tuple(
            index
            for index in range(len(options) - 1, 3, -1)
            if options[index].option_id not in required
        )
        if len(replacement_slots) < len(missing):
            raise RuntimeError(
                "Timeloop proposal tail cannot fit required support options"
            )
        updated = list(options)
        for index, option_id in zip(
            replacement_slots[: len(missing)],
            missing,
            strict=True,
        ):
            updated[index] = output_type.finite_variation_contract.resolve(option_id)
        updated_ids = tuple(option.option_id for option in updated)
        if len(set(updated_ids)) != len(updated_ids):
            raise RuntimeError(
                "Timeloop proposal-support insertion duplicated an option"
            )
        return tuple(updated)

    async def __call__(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        self.calls += 1
        output_type = request.output_type
        options = self._with_required_proposal_support(
            output_type,
            self._proposal_options(output_type),
        )
        assigned_cards = tuple(sorted(output_type.assigned_card_keys))
        members = []
        for index, option in enumerate(options):
            common = {
                "supporting_card_keys": (list(assigned_cards) if index == 0 else []),
                "effect_predictions": [
                    {
                        "metric_id": metric_id,
                        "direction": "decrease",
                        "confidence": "medium",
                    }
                    for metric_id in OBJECTIVE_IDS
                ],
                "role_proposal": (
                    "exploit" if index < 4 else "falsify" if index < 7 else "coverage"
                ),
                "design_rationale": (
                    "Provider-free typed conformance proposal over one sealed "
                    "Timeloop action."
                ),
            }
            if output_type.required_composite_proposals == 0:
                members.append({"option_id": option.option_id, **common})
                continue
            metadata = dict(option.metadata)
            if metadata.get("composition_selection_exposure") == (
                "hierarchical_ranked_union"
            ):
                members.append(
                    {
                        "action_kind": "compose_r2",
                        "composite_option_id": option.option_id,
                        "component_option_ids": sorted(
                            [metadata["left_option_id"], metadata["right_option_id"]]
                        ),
                        **common,
                    }
                )
            else:
                members.append(
                    {"action_kind": "atomic", "option_id": option.option_id, **common}
                )
        value = output_type.model_validate({"members": members}, strict=True)
        self.records.append(
            {
                "call_id": request.call_id.value,
                "proposal_option_ids": [value.option_id for value in options],
                "proposal_supporting_card_keys": [
                    list(member["supporting_card_keys"]) for member in members
                ],
                "proposal_width": len(options),
                "assigned_card_keys": list(assigned_cards),
                "required_proposal_support_option_ids": sorted(
                    output_type.required_proposal_support_option_ids
                ),
                "bounded_memory_dose": output_type.memory_dose_contract is not None,
                "provider_calls": 0,
            }
        )
        return StructuredGenerationResponse(
            value=value,
            requested_model="provider-free/timeloop-calibrated-double",
            resolved_model="provider-free/timeloop-calibrated-double",
            resolved_provider="local-deterministic-conformance",
            provider_response_id=None,
            finish_reason="policy_completed",
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=0,
        )


class _RecordingSelector:
    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.results: list[tuple[PortfolioSelectionRequest, object]] = []

    async def select(self, request: PortfolioSelectionRequest):
        result = await self.delegate.select(request)
        self.results.append((request, result))
        return result


def _local_reflection_draft(
    cluster: IdentifiableMutationReflectionHypothesisCluster,
    *,
    incompatible: bool,
) -> InsightDraft:
    contrast = cluster.representative
    direction_signature = ", ".join(
        f"{value.metric_id}={value.direction.value}" for value in contrast.metrics
    )
    intervention_signature = (
        f"{thaw_json(contrast.parent_local_value)!r}->"
        f"{thaw_json(contrast.child_local_value)!r}"
    )
    family = contrast.option_family
    if incompatible:
        family = (
            "compute_parallelism"
            if contrast.option_family != "compute_parallelism"
            else "memory_capacity"
        )
    return InsightDraft(
        claim=(
            f"Prospectively retest sealed {family} option {contrast.option_id} "
            f"at {contrast.affected_path} under the current parent; predicted "
            f"directions: {direction_signature}; observed local intervention: "
            f"{intervention_signature}."
        ),
        trigger=(f"A sealed {family} action is available at {contrast.affected_path}."),
        mechanism=(
            "The direct mutation suggests a parent-relative predictive "
            "association; it does not establish a causal mechanism."
        ),
        affected_paths=(contrast.affected_path,),
        evidence_summary=(
            f"{len(cluster.contrasts)} authenticated direct single-mutation "
            "observation(s) share this local intervention and metric-direction "
            "signature."
        ),
        confidence=0.5,
        evidence_contrast_ids=cluster.contrast_ids,
        effect_predictions=tuple(
            MetricEffectPrediction(
                metric_id=value.metric_id,
                direction=value.direction,
                comparison_anchor=MetricComparisonAnchor(
                    MetricComparisonAnchorKind.CURRENT_PARENT
                ),
            )
            for value in contrast.metrics
        ),
        recommended_option_families=(family,),
        recommended_option_ids=(contrast.option_id,),
        action_template="Test one compatible sealed finite action.",
        falsification_condition=(
            "Any observed parent-relative metric direction that differs from "
            "this direct observation falsifies the predictive rule."
        ),
        insight_kind=ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
        consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
        factor_capabilities=(family,),
    )


class _ReflectionExecutor:
    """Local result double over the exact production identifiable request."""

    def __init__(
        self,
        *,
        ids: DeterministicIdFactory,
        optimization_semantics: object,
        max_output_tokens: int,
        temperature: float | None,
        incompatible_card: bool = False,
    ) -> None:
        self.ids = ids
        self.optimization_semantics = optimization_semantics
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.incompatible_card = incompatible_card
        self.generations: list[int] = []
        self.records: list[FrozenJsonObject] = []
        self.inputs: list[CampaignIdentifiableReflectionInput] = []
        self.prompts: list[str] = []

    async def reflect(self, reflection_input: CampaignIdentifiableReflectionInput):
        contrasts = reflection_input.evidence.contrasts
        clusters = cluster_identifiable_mutation_reflection_hypotheses(contrasts)
        insight_count = min(8, len(clusters))
        if insight_count < PARENTS_PER_PORTFOLIO:
            raise RuntimeError(
                "Timeloop reflection requires at least two scored G1 mutations"
            )
        request = build_timeloop_v2_identifiable_reflection_request(
            call_id=self.ids.new_llm_call_id(),
            reflection_input=reflection_input,
            optimization_semantics=self.optimization_semantics,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            min_insights=insight_count,
            max_insights=insight_count,
        )
        insights = tuple(
            _local_reflection_draft(
                cluster,
                incompatible=self.incompatible_card,
            )
            for cluster in clusters[:insight_count]
        )
        result = ReflectionGenerationResult(
            insights=insights,
            telemetry=AgenticCallTelemetry(
                requested_model="provider-free/timeloop-reflection-double",
                resolved_model="provider-free/timeloop-reflection-double",
                resolved_provider="local-deterministic-conformance",
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
                request.evidence_catalog.catalog_identity_sha256
            ),
        )
        envelope = build_timeloop_v2_identifiable_learning_envelope(
            reflection_input=reflection_input,
            request=request,
            result=result,
            optimization_semantics=self.optimization_semantics,
        )
        self.generations.append(reflection_input.query.wave.source_generation)
        self.records.append(envelope)
        self.inputs.append(reflection_input)
        self.prompts.append(request.prompt)
        return envelope


@dataclass(frozen=True, slots=True)
class _TimeloopDiagnosticCohort:
    """One sealed reflection cohort and its pre-provider estimand."""

    exposure: CampaignDiagnosticExposureReceipt
    eligible_references: tuple[InsightRef, ...]
    estimand_context: FrozenJsonObject
    full_block_permutation_rank: int
    cohort_selection_key_sha256: str

    def __post_init__(self) -> None:
        if type(self.exposure) is not CampaignDiagnosticExposureReceipt:
            raise TypeError("exposure must be exact CampaignDiagnosticExposureReceipt")
        self.exposure.__post_init__()
        if self.eligible_references != self.exposure.references:
            raise ValueError("diagnostic cohort differs from its admitted references")
        if len(self.eligible_references) < PARENTS_PER_PORTFOLIO:
            raise ValueError("Timeloop diagnostic cohort cannot underfill its lanes")
        if type(self.estimand_context) is not FrozenJsonObject:
            raise TypeError("estimand_context must be an exact FrozenJsonObject")
        if type(self.full_block_permutation_rank) is not int or not (
            0
            <= self.full_block_permutation_rank
            < math.factorial(PARENTS_PER_PORTFOLIO)
        ):
            raise ValueError("diagnostic permutation rank is outside the exact law")
        if (
            type(self.cohort_selection_key_sha256) is not str
            or len(self.cohort_selection_key_sha256) != 64
        ):
            raise ValueError("cohort selection key must be lowercase SHA-256")

    @property
    def exact_context_sha256(self) -> str:
        return typed_json_sha256(self.estimand_context)


@dataclass(slots=True)
class _TimeloopDiagnosticBlockCoordinator:
    """Seal one workload-neutral randomized memory estimand for G5.

    The coordinator binds cohort, utility, and randomization identity before
    provider dispatch.  The wave factory separately derives the prospective
    lane/card graph and may realize this law only after the generic complete-
    support audit passes.
    """

    memory: InsightMemoryBank
    learning_runtime: ClosedLoopCampaignLearningRuntime
    utility: AffineHypervolumeArchiveUtility3D
    outer_seed: int
    task_sha256: str
    _cohorts: dict[int, _TimeloopDiagnosticCohort] = field(
        init=False,
        default_factory=dict,
    )
    _eligible_receipts: dict[int, tuple[str, ...]] = field(
        init=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if type(self.learning_runtime) is not ClosedLoopCampaignLearningRuntime:
            raise TypeError("learning_runtime must be exact closed-loop runtime")
        if type(self.utility) is not AffineHypervolumeArchiveUtility3D:
            raise TypeError("Timeloop diagnostic credit requires affine 3-D utility")
        if type(self.outer_seed) is not int:
            raise TypeError("outer_seed must be an exact integer")
        if type(self.task_sha256) is not str or len(self.task_sha256) != 64:
            raise ValueError("task_sha256 must be lowercase SHA-256")

    def resolve(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> _TimeloopDiagnosticCohort | None:
        generation = context.stage_request.step.generation
        receipt_sha256s = context.stage_request.test_eligible_reflection_receipt_sha256s
        cached = self._cohorts.get(generation)
        if cached is not None:
            if self._eligible_receipts[generation] != receipt_sha256s:
                raise ValueError(
                    "one diagnostic generation received inconsistent reflection cohorts"
                )
            return cached
        if not receipt_sha256s:
            return None
        exposures = self.learning_runtime.diagnostic_exposures(receipt_sha256s)
        if len(exposures) != 1:
            raise ValueError("Timeloop requires one sealed diagnostic exposure")
        exposure = exposures[0]
        entries = self.memory.entries_for(exposure.references)
        if len(entries) < PARENTS_PER_PORTFOLIO:
            raise ValueError("Timeloop diagnostic cohort cannot underfill two lanes")
        rank = int.from_bytes(
            hashlib.sha256(
                (
                    f"{self.task_sha256}:{self.outer_seed}:"
                    f"{exposure.receipt_sha256}:generation:{generation}:"
                    "complete-compatible-singleton"
                ).encode("ascii", errors="strict")
            ).digest(),
            "big",
        ) % math.factorial(PARENTS_PER_PORTFOLIO)
        cohort_selection_key_sha256 = hashlib.sha256(
            (
                f"{self.task_sha256}:{self.outer_seed}:"
                f"{exposure.receipt_sha256}:generation:{generation}:"
                "complete-support-cohort"
            ).encode("ascii", errors="strict")
        ).hexdigest()
        estimand_context = _object(
            {
                "schema_version": 1,
                "workload_family": "timeloop_architecture_mapspace_codesign",
                "treatment_unit": "one_complete_candidate_portfolio_wave",
                "intervention": (
                    "pre_outcome_active_memory_card_and_supported_action_dose_"
                    "versus_canonical_redacted_neutral_view"
                ),
                "outcome": "fixed_affine_3d_archive_joint_wave_gain",
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
                    "policy_version": PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION,
                    "policy_definition_sha256": (
                        PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256
                    ),
                    "reflection_exposure_receipt_sha256": (exposure.receipt_sha256),
                    "eligible_references": [
                        {
                            "insight_id": value.insight_id.value,
                            "version": value.version,
                        }
                        for value in exposure.references
                    ],
                    "lane_count": PARENTS_PER_PORTFOLIO,
                    "subset_size": 1,
                    "active_neutral_pair": True,
                    "one_card_supported_in_both_lanes_required": True,
                    "card_selection": {
                        "policy_id": PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_ID,
                        "policy_version": (
                            PORTFOLIO_MEMORY_MATCHED_CONTROL_POLICY_VERSION
                        ),
                        "policy_definition_sha256": (
                            PORTFOLIO_MEMORY_MATCHED_CONTROL_DEFINITION_SHA256
                        ),
                        "selection_key_sha256": cohort_selection_key_sha256,
                        "select_one_card_after_two_lane_support_audit": True,
                    },
                    "active_unit_rank": rank,
                    "rank_identity": {
                        "task_sha256": self.task_sha256,
                        "outer_seed": self.outer_seed,
                        "generation": generation,
                    },
                    "provider_and_outcome_blind": True,
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
        cohort = _TimeloopDiagnosticCohort(
            exposure=exposure,
            eligible_references=exposure.references,
            estimand_context=estimand_context,
            full_block_permutation_rank=rank,
            cohort_selection_key_sha256=cohort_selection_key_sha256,
        )
        self._cohorts[generation] = cohort
        self._eligible_receipts[generation] = receipt_sha256s
        return cohort

    def project(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> CampaignPortfolioMemoryEstimandProjection | None:
        cohort = self.resolve(context)
        if cohort is None:
            return None
        return CampaignPortfolioMemoryEstimandProjection(
            estimand_context=cohort.estimand_context,
            estimand_stratum_sha256=cohort.exact_context_sha256,
        )

    def require_projected_context(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> _TimeloopDiagnosticCohort | None:
        cohort = self.resolve(context)
        if cohort is None:
            return None
        values = dict(context.evidence_context.items)
        if (
            values.get("memory_estimand_stratum_sha256") != cohort.exact_context_sha256
            or values.get("memory_estimand_context") != cohort.estimand_context
        ):
            raise ValueError("selector context differs from its sealed estimand")
        return cohort


class _WaveFactory:
    def __init__(
        self,
        *,
        composition: object,
        learning_runtime: ClosedLoopCampaignLearningRuntime,
        diagnostic_coordinator: _TimeloopDiagnosticBlockCoordinator,
        utility: AffineHypervolumeArchiveUtility3D,
        seed_card: PortfolioCard,
        binding_factory: CalibratedCampaignBindingFactory,
        coordinator: CalibratedPortfolioCampaignCoordinator,
        target_conditioned_controller: (
            TargetConditionedCampaignOutcomeUpdater | None
        ),
        evaluation_mating_constraints: bool,
        max_output_tokens: int,
        temperature: float | None,
    ) -> None:
        self.composition = composition
        self.learning_runtime = learning_runtime
        self.diagnostic_coordinator = diagnostic_coordinator
        self.utility = utility
        self.seed_card = seed_card
        self.binding_factory = binding_factory
        self.coordinator = coordinator
        self.target_conditioned_controller = target_conditioned_controller
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        if type(evaluation_mating_constraints) is not bool:
            raise TypeError("evaluation_mating_constraints must be exact bool")
        self.evaluation_mating_constraints = evaluation_mating_constraints
        self.wave_records: list[dict[str, object]] = []
        self.dose_contracts: list[BoundedPortfolioMemoryDoseContract] = []
        self.matching_receipts: list[object] = []
        self.cohort_selection_receipts: list[
            CampaignDiagnosticCompleteSupportCohortSelection
        ] = []
        self.compatibility_audits: list[CampaignDiagnosticCompatibilityAudit] = []
        self.diagnostic_blocks: list[CampaignDiagnosticSingletonBlock] = []
        self.matched_support_resolutions: list[object] = []
        self.matched_control_plans: list[object] = []
        self.recourse_receipts: list[FrozenJsonObject] = []

    def _request(
        self,
        context: CampaignPortfolioWaveContext,
        cards: tuple[PortfolioCard, ...],
        *,
        source_registry=None,
        memory_dose_contract=None,
        experimental_view_receipt=None,
        candidate_pool_required_option_ids=(),
    ) -> PortfolioSelectionRequest:
        return PortfolioSelectionRequest(
            call_id=self.composition.id_factory.new_llm_call_id(),
            operation="select_portfolio",
            instruction=(
                "Propose sealed Timeloop options under the authenticated "
                "calibrated portfolio contract."
            ),
            context=context.evidence_context,
            finite_variation_contract=context.variation.contract,
            cards=cards,
            portfolio_size=PORTFOLIO_WIDTH,
            required_metric_ids=OBJECTIVE_IDS,
            min_distinct_families=(4 if self.evaluation_mating_constraints else None),
            require_supporting_cards=False,
            require_pairwise_disjoint_parent_patches=(
                self.evaluation_mating_constraints
            ),
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            source_registry=source_registry,
            experimental_view_receipt=experimental_view_receipt,
            memory_dose_contract=memory_dose_contract,
            candidate_pool_required_option_ids=(candidate_pool_required_option_ids),
        )

    def _register(
        self,
        context: CampaignPortfolioWaveContext,
        request: PortfolioSelectionRequest,
        *,
        status: str,
        evidence: object = None,
    ) -> None:
        bounded = request.memory_dose_contract is not None
        expected_prompt = calibrated_portfolio_prompt_definition_sha256(
            bounded_memory_dose=bounded,
            proposal_support=_proposal_support_policy() is not None,
            hierarchical_composition_required_proposals=(
                VARIATION_TOPOLOGY.hierarchical_composition_required_proposals
            ),
            feasibility_witness_mode=FEASIBILITY_WITNESS_MODE,
            constraint_decoupled=self.coordinator.constraint_decoupled,
        )
        factory = self.binding_factory
        if factory.scope.prompt_definition_sha256 != expected_prompt:
            factory = replace(
                factory,
                scope=replace(factory.scope, prompt_definition_sha256=expected_prompt),
            )
        binding = factory.build(
            request=request,
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
                selection=request,
            )
        )
        self.coordinator.register(
            request,
            binding,
            target_conditioned_context=target_context,
        )
        prompt = self.coordinator.render(request)
        self.wave_records.append(
            {
                "generation": context.stage_request.step.generation,
                "parent_slot": context.parent_slot,
                "parent_candidate_id": context.parent.candidate_id.value,
                "request_sha256": request.request_sha256,
                "status": status,
                "evidence": evidence,
                "proposal_width": CALIBRATED_PROPOSAL_WIDTH,
                "evaluation_width": PORTFOLIO_WIDTH,
                "bounded_memory_dose": bounded,
                "memory_dose_contract": (
                    None
                    if request.memory_dose_contract is None
                    else request.memory_dose_contract.to_record()
                ),
                "prompt_sha256": _sha(prompt),
                "prompt_utf8_bytes": len(prompt.encode("utf-8")),
            }
        )

    def _bootstrap_wave(
        self,
        context: CampaignPortfolioWaveContext,
        *,
        status: str,
        evidence: object = None,
    ) -> PortfolioVariationWaveRequest:
        request = self._request(context, (self.seed_card,))
        self._register(context, request, status=status, evidence=evidence)
        return PortfolioVariationWaveRequest(
            selection_request=request,
            parent=context.parent,
            generation=context.stage_request.step.generation,
            label_prefix=(
                f"timeloop_g{context.stage_request.step.generation:02d}_"
                f"p{context.parent_slot + 1:02d}"
            ),
            phase="timeloop_g6_delayed_identifiable",
        )

    def _matched_dose_wave(
        self,
        context: CampaignPortfolioWaveContext,
        *,
        cohort: _TimeloopDiagnosticCohort,
        matched_plan: object,
        assignment: object,
        arm_view: object,
        entry: object,
        support: object,
        support_resolution_receipt_sha256: str,
    ) -> PortfolioVariationWaveRequest:
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
        request = self._request(
            context,
            arm_view.cards,
            source_registry=arm_view.source_registry,
            memory_dose_contract=dose,
            experimental_view_receipt=arm_view.experimental_view_receipt,
            candidate_pool_required_option_ids=(
                arm_view.required_common_pool_option_ids
            ),
        )
        if dose is not None:
            self.dose_contracts.append(dose)
        snapshot = self.utility.require_snapshot(context.stage_request.archive_utility)
        reward = AffineFrozenArchiveJointWaveReward3D(snapshot)
        aggregation = PortfolioRewardAggregationBinding(
            aggregate=lambda outcomes, reward=reward: float(
                reward(tuple(value.candidate for value in outcomes))
            ),
            aggregation_id=MEMORY_AGGREGATION_ID,
            aggregation_version=1,
            definition_sha256=MEMORY_AGGREGATION_DEFINITION_SHA256,
        )
        self._register(
            context,
            request,
            status="applied_randomized_active_neutral_arm",
            evidence={
                "experimental_arm": assignment.arm.value,
                "exposure_receipt_sha256": cohort.exposure.receipt_sha256,
                "insight_id": entry.reference.insight_id.value,
                "insight_version": entry.reference.version,
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
                "credit_status": (
                    "append_only_single_block_diagnostic_no_online_credit"
                ),
            },
        )
        return PortfolioVariationWaveRequest(
            selection_request=request,
            parent=context.parent,
            generation=context.stage_request.step.generation,
            label_prefix=f"timeloop_g05_p{context.parent_slot + 1:02d}",
            phase="timeloop_g6_delayed_identifiable",
            matched_memory_control=PortfolioMemoryMatchedControlWavePlan(
                plan=matched_plan,
                assignment=assignment,
                arm_view=arm_view,
                aggregation=aggregation,
                context_projection=projection,
            ),
        )

    def build_batch(
        self,
        contexts: tuple[CampaignPortfolioWaveContext, ...],
    ) -> tuple[PortfolioVariationWaveRequest, ...]:
        if not contexts:
            raise ValueError("Timeloop wave batch cannot be empty")
        generation = contexts[0].stage_request.step.generation
        if any(value.stage_request.step.generation != generation for value in contexts):
            raise ValueError("Timeloop wave batch cannot mix generations")
        if generation != 5:
            return tuple(self.build(value) for value in contexts)
        if len(contexts) != PARENTS_PER_PORTFOLIO:
            raise RuntimeError("Timeloop G5 requires both parent lanes")

        cohort_by_lane = {}
        for context in contexts:
            cohort = self.diagnostic_coordinator.require_projected_context(context)
            if cohort is None:
                raise RuntimeError("Timeloop G5 requires one projected cohort")
            cohort_by_lane[context.parent_lane.lane_id] = cohort
        if (
            len({value.exposure.receipt_sha256 for value in cohort_by_lane.values()})
            != 1
        ):
            raise RuntimeError("Timeloop G5 lanes received different exposures")
        cohort = next(iter(cohort_by_lane.values()))
        exposure = cohort.exposure
        entries = tuple(
            sorted(
                self.composition.memory.entries_for(cohort.eligible_references),
                key=lambda value: value.reference,
            )
        )
        if len(entries) < PARENTS_PER_PORTFOLIO:
            raise RuntimeError("Timeloop G5 lost its eligible insight cohort")

        contexts_by_lane = {value.parent_lane.lane_id: value for value in contexts}
        lane_contracts = tuple(
            context.variation.contract
            for _, context in sorted(contexts_by_lane.items())
        )
        cards_by_key: dict[str, PortfolioCard] = {}
        entries_by_key = {}
        for ordinal, entry in enumerate(entries, start=1):
            prompt_payload = project_action_neutral_insight_prompt_payload(
                entry,
                prompt_payload=_object(
                    {
                        "claim": entry.draft.claim,
                        "trigger": entry.draft.trigger,
                        "mechanism": entry.draft.mechanism,
                        "quarantined": True,
                    }
                ),
                finite_variation_contracts=lane_contracts,
            )
            card = portfolio_card_from_insight_entry(
                entry,
                card_key=f"card.timeloop.g05.r{ordinal:02d}",
                prompt_payload=prompt_payload,
                evidence_sha256=entry.evidence_lineage.identity_sha256,
                source_receipt_sha256=exposure.receipt_sha256,
                assigned_score=0.0,
            )
            if card.source_binding is None:  # pragma: no cover - constructor closes.
                raise AssertionError("source-bound Timeloop card lost its binding")
            cards_by_key[card.card_key] = card
            entries_by_key[card.card_key] = entry

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
        resolution = PortfolioMemoryMatchedSupportResolver().resolve(
            lanes=support_lanes,
            cards=support_cards,
            selection_key_sha256=cohort.cohort_selection_key_sha256,
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
            self.recourse_receipts.append(recourse)
            return tuple(
                self._bootstrap_wave(
                    context,
                    status="no_shared_support_active_neutral_card",
                    evidence=thaw_json(recourse),
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
                unit_key=(f"reflection.{exposure.receipt_sha256[:12]}.g05.{lane_id}"),
                generation=5,
                lane_id=lane_id,
            )
            for lane_id in sorted(contexts_by_lane)
        )
        matched_plan = PortfolioMemoryMatchedControlPlanner().plan(
            reference=selected_entry.reference,
            exact_context_sha256=cohort.exact_context_sha256,
            ordered_units=ordered_units,
            active_unit_rank=cohort.full_block_permutation_rank,
        )
        self.matched_control_plans.append(matched_plan)
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
                    cohort=cohort_by_lane[lane_id],
                    matched_plan=matched_plan,
                    assignment=assignment,
                    arm_view=arm_view,
                    entry=selected_entry,
                    support=resolution.support_for(lane_id),
                    support_resolution_receipt_sha256=resolution.receipt_sha256,
                )
            )
        return tuple(waves)

    def build(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> PortfolioVariationWaveRequest:
        generation = context.stage_request.step.generation
        if generation == 1:
            return self._bootstrap_wave(context, status="bootstrap_prior")
        if generation == 3:
            if context.stage_request.test_eligible_reflection_receipt_sha256s:
                raise RuntimeError("G3 cannot consume the delayed G2 reflection")
            return self._bootstrap_wave(
                context,
                status="delayed_reflection_not_yet_admitted",
            )
        if generation == 5:
            raise RuntimeError("G5 must be constructed atomically across lanes")
        raise ValueError("Timeloop wave factory received a non-portfolio generation")


class _DirectPortfolioWaveFactory:
    """Build shared campaign waves for a direct provider-free selector.

    The direct policy receives the complete eligible finite contract and
    returns the evaluated K8 itself.  No learned-memory dose is administered:
    this is the uninformed control seam, while cadence, parents, evaluation,
    recombination, and candidate budgets remain shared with the treatment.
    """

    def __init__(self, *, composition: object, seed_card: PortfolioCard) -> None:
        self.composition = composition
        self.seed_card = seed_card
        self.wave_records: list[dict[str, object]] = []
        self.dose_contracts: list[BoundedPortfolioMemoryDoseContract] = []
        self.matching_receipts: list[object] = []
        self.cohort_selection_receipts: list[
            CampaignDiagnosticCompleteSupportCohortSelection
        ] = []
        self.compatibility_audits: list[CampaignDiagnosticCompatibilityAudit] = []
        self.diagnostic_blocks: list[CampaignDiagnosticSingletonBlock] = []
        self.matched_support_resolutions: list[object] = []
        self.matched_control_plans: list[object] = []
        self.recourse_receipts: list[FrozenJsonObject] = []

    def build_batch(
        self,
        contexts: tuple[CampaignPortfolioWaveContext, ...],
    ) -> tuple[PortfolioVariationWaveRequest, ...]:
        return tuple(self.build(context) for context in contexts)

    def build(
        self,
        context: CampaignPortfolioWaveContext,
    ) -> PortfolioVariationWaveRequest:
        generation = context.stage_request.step.generation
        if generation not in {1, 3, 5}:
            raise ValueError(
                "Timeloop direct control received a non-portfolio generation"
            )
        request = PortfolioSelectionRequest(
            call_id=self.composition.id_factory.new_llm_call_id(),
            operation="select_portfolio",
            instruction=(
                "Select an outcome-blind control portfolio from the complete "
                "eligible finite Timeloop contract."
            ),
            context=context.evidence_context,
            finite_variation_contract=context.variation.contract,
            cards=(self.seed_card,),
            portfolio_size=PORTFOLIO_WIDTH,
            required_metric_ids=OBJECTIVE_IDS,
            min_distinct_families=None,
            require_supporting_cards=False,
            require_pairwise_disjoint_parent_patches=False,
            max_output_tokens=1,
            temperature=None,
        )
        self.wave_records.append(
            {
                "generation": generation,
                "parent_slot": context.parent_slot,
                "parent_candidate_id": context.parent.candidate_id.value,
                "request_sha256": request.request_sha256,
                "status": "provider_free_direct_control",
                "proposal_width": PORTFOLIO_WIDTH,
                "evaluation_width": PORTFOLIO_WIDTH,
                "bounded_memory_dose": False,
                "memory_dose_contract": None,
            }
        )
        return PortfolioVariationWaveRequest(
            selection_request=request,
            parent=context.parent,
            generation=generation,
            label_prefix=(
                f"timeloop_control_g{generation:02d}_p{context.parent_slot + 1:02d}"
            ),
            phase="timeloop_g6_provider_free_direct_control",
        )


class _ArchiveUtility:
    utility_id = "timeloop_provider_free_archive_trace"
    utility_version = 2
    definition_sha256 = _sha("timeloop-provider-free-archive-trace-v2")

    def freeze(self, *, benchmark, generation, archive):
        return ArchiveUtilitySnapshot(
            utility_id=self.utility_id,
            utility_version=self.utility_version,
            definition_sha256=self.definition_sha256,
            generation=generation,
            benchmark_sha256=typed_json_sha256(benchmark),
            archive_sha256=typed_json_sha256(archive),
            snapshot_receipt=_object(
                {"generation": generation, "role": "provider_free_trace_only"}
            ),
        )


class _PreparationRuntime:
    def prepare(self, request):
        return CampaignAgentRuntimeReceipt(
            request_sha256=request.request_sha256,
            runtime_id="timeloop_provider_free_g6_runtime",
            runtime_version=1,
            definition_sha256=_sha("timeloop-provider-free-g6-runtime-v1"),
            accepted=True,
            evidence=_object({"provider_calls": 0, "docker_calls": 0}),
        )


class _PreparationJournal:
    def append(self, record):
        if type(record) is not FrozenJsonObject:
            raise TypeError("campaign preparation journal requires a frozen record")


class _ExecutionJournal:
    def __init__(self) -> None:
        self.events: list[CampaignExecutionEvent] = []

    async def append(self, event):
        self.events.append(event)
        return CampaignJournalAck(event.event_sha256, True)


def _binding(name: str, implementation: object) -> CampaignPolicyBinding:
    return CampaignPolicyBinding(
        implementation=implementation,
        policy_id=name,
        policy_version=1,
        definition_sha256=_sha(f"timeloop-campaign-policy:{name}:v1"),
    )


def _seed_memory(
    memory: InsightMemoryBank,
) -> PortfolioCard:
    entry = memory.extend(
        (
            InsightDraft(
                claim="Explore diverse sealed Timeloop single-locus actions.",
                trigger="A parent-local Timeloop finite catalog is available.",
                mechanism=(
                    "The exact catalog exposes architecture and mapspace-policy "
                    "variation without model-authored candidate values."
                ),
                affected_paths=("$.pe_mesh_x",),
                evidence_summary="Prospective non-empirical bootstrap prior.",
                confidence=0.5,
            ),
        ),
        initial_score=0.0,
        applicable_operator_kinds=("typed_mutation",),
    )[0]
    return PortfolioCard(
        card_key="card.timeloop.bootstrap",
        reference=entry.reference,
        content_sha256=entry.draft.content_sha256,
        evidence_sha256=_sha("timeloop-provider-free-bootstrap-evidence-v2"),
        prompt_payload=_object(
            {
                "epistemic_status": "prior_hypothesis",
                "adaptive_score_consumption": False,
            }
        ),
        assigned_score=0.0,
    )


@dataclass(slots=True)
class ProviderFreeTimeloopCampaignRun:
    execution: CampaignExecutionResult
    evaluator: object
    memory: InsightMemoryBank
    evidence_registry: CampaignEvidenceRegistry
    reflection_executor: object
    wave_factory: object
    calibrated_runner: object
    selector: _RecordingSelector
    feedback_ledger: PortfolioOutcomeFeedbackLedger
    contextual_search_ledger: ContextualSearchLedger | None
    contextual_search_planner: CampaignContextualSearchPlanner | None
    target_conditioned_controller: TargetConditionedCampaignOutcomeUpdater | None
    journal: object
    engine_traces: list[dict[str, object]]
    execution_mode: str
    docker_enabled: bool
    provider_enabled: bool
    scientific_claim: str
    experiment_profile: CampaignExperimentProfile | None
    experiment_profile_conformance: dict[str, object] | None

    def summary(self) -> dict[str, object]:
        decoded = tuple(
            CampaignReflectionLearningRecordCodec.decode(value)
            for value in self.reflection_executor.records
        )
        evaluator_calls = getattr(self.evaluator, "calls", None)
        if type(evaluator_calls) is not int or evaluator_calls < 0:
            raise RuntimeError("campaign evaluator did not publish an exact call count")
        provider_calls = (
            getattr(self.calibrated_runner, "calls", None)
            if self.provider_enabled
            else 0
        )
        if type(provider_calls) is not int or provider_calls < 0:
            raise RuntimeError("campaign runner did not publish provider call count")
        dose_results = tuple(
            result
            for request, result in self.selector.results
            if request.memory_dose_contract is not None
        )
        proposal_widths = {
            value["proposal_width"] for value in self.wave_factory.wave_records
        }
        if len(proposal_widths) != 1:
            raise RuntimeError("campaign wave records mix proposal widths")
        proposal_width = int(next(iter(proposal_widths)))
        portfolio_learning_records = []
        for receipt in self.execution.stage_receipts:
            if receipt.kind.value != "portfolio":
                continue
            result = thaw_json(receipt.result)
            if type(result) is not dict:
                raise TypeError("portfolio stage result must be an object")
            portfolio_learning_records.append(result.get("closed_loop_learning"))
        variation_trace = summarize_campaign_variation_trace(
            tuple(result for _, result in self.selector.results),
            required_composite_proposals=(
                VARIATION_TOPOLOGY.hierarchical_composition_required_proposals
            ),
        )
        contextual_allocation_projections: list[dict[str, object]] = []
        for _, result in self.selector.results:
            audit = result.supplemental_audit
            if audit is None:
                continue
            payload = thaw_json(audit.payload)
            if type(payload) is not dict:
                raise TypeError("selector audit payload must be an object")
            reconciliation = payload.get("semantic_reconciliation")
            if type(reconciliation) is not dict:
                continue
            projection = reconciliation.get("contextual_allocation_projection")
            if projection is None:
                continue
            if type(projection) is not dict:
                raise TypeError("contextual allocation projection must be an object")
            contextual_allocation_projections.append(projection)
        contextual_completion_audit = (
            None
            if self.contextual_search_ledger is None
            or self.contextual_search_planner is None
            else audit_completed_contextual_search_ledger(
                self.contextual_search_ledger,
                campaign_scope_sha256=(
                    self.contextual_search_planner.campaign_scope_sha256
                ),
                expected_wave_count=sum(
                    value.kind.value == "portfolio"
                    for value in self.execution.stage_receipts
                ),
                expected_post_recombination_wave_indices=tuple(
                    value.generation // 2
                    for value in self.execution.stage_receipts
                    if value.kind.value == "recombination"
                ),
                expected_observation_count=(
                    len(self.selector.results) * PORTFOLIO_WIDTH
                ),
                expected_allocation_realization_count=len(self.selector.results),
            )
        )
        return {
            "status": self.execution.finalization_receipt.status.value,
            "generations_completed": self.execution.counters.generations_completed,
            "planned_candidate_occurrences": PLANNED_CANDIDATE_OCCURRENCES,
            "candidate_occurrences": self.execution.counters.candidate_occurrences,
            "unique_evaluations": self.execution.counters.unique_evaluations,
            "physical_evaluator_calls": evaluator_calls,
            "evaluator_calls": evaluator_calls,
            "logical_agent_calls": self.execution.counters.logical_agent_calls,
            "selector_calls": len(self.selector.results),
            "proposal_width": proposal_width,
            "k8_typed_proposals": (
                len(self.selector.results)
                if proposal_width == CALIBRATED_PROPOSAL_WIDTH
                else 0
            ),
            "variation_trace": variation_trace.to_record(),
            "direct_portfolio_selections": (
                len(self.selector.results)
                if type(self.wave_factory) is _DirectPortfolioWaveFactory
                else 0
            ),
            "outcome_feedback_receipts": len(self.feedback_ledger.receipts),
            "forecast_calibration_observations": len(self.feedback_ledger.observations),
            "contextual_search_plans": (
                []
                if self.contextual_search_planner is None
                else [
                    value.to_record() for value in self.contextual_search_planner.plans
                ]
            ),
            "contextual_search_observations": (
                []
                if self.contextual_search_ledger is None
                else [
                    value.to_record()
                    for value in self.contextual_search_ledger.observations
                ]
            ),
            "contextual_search_delayed_credits": (
                []
                if self.contextual_search_ledger is None
                else [
                    value.to_record()
                    for value in self.contextual_search_ledger.delayed_credits
                ]
            ),
            "contextual_search_allocation_realizations": (
                []
                if self.contextual_search_ledger is None
                else [
                    value.to_record()
                    for value in (self.contextual_search_ledger.allocation_realizations)
                ]
            ),
            "contextual_search_completion_audit": (
                None
                if contextual_completion_audit is None
                else contextual_completion_audit.to_record()
            ),
            "contextual_allocation_projections": (contextual_allocation_projections),
            "target_conditioned_state": (
                None
                if self.target_conditioned_controller is None
                else self.target_conditioned_controller.state.to_record()
            ),
            "reflection_generations": list(self.reflection_executor.generations),
            "canonical_reflection_records": len(decoded),
            "reflection_input_cutoffs": [
                [
                    value.query.prior_cutoff_event_index_exclusive,
                    value.query.sealed_cutoff_event_index_inclusive,
                ]
                for value in self.reflection_executor.inputs
            ],
            "memory_entries": len(self.memory.entries),
            "memory_trials": len(self.memory.trials),
            "memory_entry_records": [
                value.to_record() for value in self.memory.entries
            ],
            "memory_trial_records": [
                _memory_trial_record(value) for value in self.memory.trials
            ],
            "memory_lifecycle_transitions": [
                _memory_transition_record(value) for value in self.memory.transitions
            ],
            "portfolio_stage_learning_records": portfolio_learning_records,
            "authenticated_action_observations": len(
                self.evidence_registry.observations
            ),
            "bounded_g5_dose_request_count": len(self.wave_factory.dose_contracts),
            "bounded_g5_dose_result_count": len(dose_results),
            "bounded_g5_dose_assessments_pass": all(
                result.decision.memory_dose_assessment is not None
                and result.decision.memory_dose_assessment.passed
                for result in dose_results
            ),
            "compatibility_matching_receipts": len(self.wave_factory.matching_receipts),
            "diagnostic_cohort_selections": [
                value.to_record()
                for value in self.wave_factory.cohort_selection_receipts
            ],
            "diagnostic_compatibility_audits": [
                value.to_record() for value in self.wave_factory.compatibility_audits
            ],
            "diagnostic_memory_blocks": [
                value.to_record() for value in self.wave_factory.diagnostic_blocks
            ],
            "matched_memory_support_resolutions": [
                value.to_record()
                for value in self.wave_factory.matched_support_resolutions
            ],
            "matched_memory_control_plans": [
                value.to_record() for value in self.wave_factory.matched_control_plans
            ],
            "typed_recourse_receipts": len(self.wave_factory.recourse_receipts),
            "provider_calls": provider_calls,
            "docker_calls": evaluator_calls if self.docker_enabled else 0,
            "execution_mode": self.execution_mode,
            "scientific_claim": self.scientific_claim,
            "experiment_profile": (
                None
                if self.experiment_profile is None
                else self.experiment_profile.to_record()
            ),
            "experiment_profile_conformance": self.experiment_profile_conformance,
        }


def run_timeloop_campaign(
    *,
    benchmark: AgenticBenchmark,
    evaluator: object,
    execution_mode: str,
    id_namespace: str,
    campaign_sha256: str,
    evaluator_contract_sha256: str,
    protocol_id: str,
    protocol_definition_sha256: str,
    task_sha256: str,
    evaluator_preflight_receipt: FrozenJsonObject,
    resource_lease_receipt: FrozenJsonObject,
    docker_enabled: bool,
    scientific_claim: str,
    outer_seed: int = OUTER_SEED,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float | None = TEMPERATURE,
    incompatible_reflection_card: bool = False,
    calibrated_runner: object | None = None,
    calibrated_allocator: CalibratedPortfolioAllocator | None = None,
    target_conditioned_specification: (
        TargetConditionedCampaignSpecification | None
    ) = None,
    direct_portfolio_selector: object | None = None,
    portfolio_selector_override: object | None = None,
    reflection_executor_factory: (
        Callable[[DeterministicIdFactory, object], object] | None
    ) = None,
    model_profile_sha256: str | None = None,
    selector_policy_binding_id: str | None = None,
    reflection_policy_binding_id: str | None = None,
    provider_enabled: bool = False,
    owned_resources: object | None = None,
    execution_journal: object | None = None,
    engine_trace_sink: Callable[[dict[str, object]], None] | None = None,
    archive_utility: object | None = None,
    recombination_utility_binder: object | None = None,
    model_execution_profile: OpenRouterModelExecutionProfile | None = None,
    constraint_decoupled_acquisition: bool = CONSTRAINT_DECOUPLED_ACQUISITION,
    minimum_intervention_projection: bool = MINIMUM_INTERVENTION_PROJECTION,
    evidence_calibrated_source_mix: bool = EVIDENCE_CALIBRATED_SOURCE_MIX,
    contextual_search_allocation: bool = CONTEXTUAL_SEARCH_ALLOCATION,
    residual_frontier_planning: bool = RESIDUAL_FRONTIER_PLANNING,
    evaluator_concurrency: int = 1,
    agent_concurrency: int = 3,
    agent_queue_capacity: int = 8,
) -> ProviderFreeTimeloopCampaignRun:
    """Run the fixed G6 composition (38 reference or 62 legacy occurrences).

    Physical evaluator calls can be lower because the generic phenotype cache
    joins repeated recombination phenotypes instead of re-evaluating them.
    """

    if type(benchmark) is not AgenticBenchmark:
        raise TypeError("benchmark must be an exact AgenticBenchmark")
    if len(benchmark.finite_variation_catalogs) != 1:
        raise ValueError("Timeloop campaign requires one atomic finite catalog")
    benchmark = replace(
        benchmark,
        finite_variation_catalogs=(
            VARIATION_TOPOLOGY.decorate(benchmark.finite_variation_catalogs[0]),
        ),
    )
    benchmark.validate_binding()
    if benchmark.optimization_semantics is None:
        raise ValueError("Timeloop G6 requires benchmark optimization semantics")
    if type(getattr(evaluator, "calls", None)) is not int:
        raise TypeError("evaluator must publish an exact calls counter")
    if type(execution_mode) is not str or not execution_mode:
        raise ValueError("execution_mode must be non-empty")
    if type(docker_enabled) is not bool:
        raise TypeError("docker_enabled must be bool")
    if type(outer_seed) is not int:
        raise TypeError("outer_seed must be an exact integer")
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be a positive exact integer")
    if temperature is not None and (
        type(temperature) is not float or not 0.0 <= temperature <= 2.0
    ):
        raise ValueError("temperature must be None or a finite float in [0,2]")
    if type(incompatible_reflection_card) is not bool:
        raise TypeError("incompatible_reflection_card must be bool")
    if type(provider_enabled) is not bool:
        raise TypeError("provider_enabled must be bool")
    if model_execution_profile is not None and type(model_execution_profile) is not (
        OpenRouterModelExecutionProfile
    ):
        raise TypeError("model_execution_profile must be exact or None")
    if type(constraint_decoupled_acquisition) is not bool:
        raise TypeError("constraint_decoupled_acquisition must be an exact bool")
    if type(minimum_intervention_projection) is not bool:
        raise TypeError("minimum_intervention_projection must be an exact bool")
    if type(evidence_calibrated_source_mix) is not bool:
        raise TypeError("evidence_calibrated_source_mix must be an exact bool")
    if type(contextual_search_allocation) is not bool:
        raise TypeError("contextual_search_allocation must be an exact bool")
    if type(residual_frontier_planning) is not bool:
        raise TypeError("residual_frontier_planning must be an exact bool")
    if residual_frontier_planning and not contextual_search_allocation:
        raise ValueError(
            "residual frontier planning requires contextual search allocation"
        )
    if contextual_search_allocation and not evidence_calibrated_source_mix:
        raise ValueError(
            "contextual search allocation requires evidence-calibrated source mix"
        )
    if evidence_calibrated_source_mix and not minimum_intervention_projection:
        raise ValueError("evidence-calibrated source mix requires minimum intervention")
    if minimum_intervention_projection and not constraint_decoupled_acquisition:
        raise ValueError(
            "minimum intervention requires constraint-decoupled acquisition"
        )
    if (
        constraint_decoupled_acquisition
        and ACQUISITION_MODE is not CampaignAcquisitionMode.HORIZON_BOUNDED
    ):
        raise ValueError(
            "constraint-decoupled acquisition requires horizon_bounded mode"
        )
    for name, value in (
        ("evaluator_concurrency", evaluator_concurrency),
        ("agent_concurrency", agent_concurrency),
        ("agent_queue_capacity", agent_queue_capacity),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive exact integer")
    if agent_queue_capacity < agent_concurrency:
        raise ValueError("agent_queue_capacity must cover agent_concurrency")
    if calibrated_runner is not None and not callable(calibrated_runner):
        raise TypeError("calibrated_runner must be callable or None")
    if target_conditioned_specification is not None and type(
        target_conditioned_specification
    ) is not TargetConditionedCampaignSpecification:
        raise TypeError("target_conditioned_specification must be exact or None")
    if direct_portfolio_selector is not None:
        if contextual_search_allocation:
            raise ValueError(
                "direct portfolio selection cannot consume contextual allocations"
            )
        if not callable(getattr(direct_portfolio_selector, "select", None)):
            raise TypeError("direct_portfolio_selector must expose select")
        if calibrated_runner is not None or calibrated_allocator is not None:
            raise ValueError(
                "direct portfolio selection cannot be combined with calibrated "
                "runner or allocator injection"
            )
        if target_conditioned_specification is not None:
            raise ValueError(
                "direct portfolio selection cannot use target-conditioned state"
            )
    if portfolio_selector_override is not None:
        if direct_portfolio_selector is not None:
            raise ValueError(
                "portfolio selector override and direct selector are mutually exclusive"
            )
        if contextual_search_allocation:
            raise ValueError(
                "portfolio selector override cannot consume contextual allocations"
            )
        for attribute in (
            "select",
            "render",
            "decode_selected_predictions",
            "policy_definition_sha256",
            "prompt_definition_sha256",
        ):
            value = getattr(portfolio_selector_override, attribute, None)
            if attribute in {"policy_definition_sha256", "prompt_definition_sha256"}:
                if type(value) is not str or len(value) != 64:
                    raise TypeError(
                        f"portfolio selector override lacks exact {attribute}"
                    )
            elif not callable(value):
                raise TypeError(
                    f"portfolio selector override must expose callable {attribute}"
                )
    if reflection_executor_factory is not None and not callable(
        reflection_executor_factory
    ):
        raise TypeError("reflection_executor_factory must be callable or None")
    if engine_trace_sink is not None and not callable(engine_trace_sink):
        raise TypeError("engine_trace_sink must be callable or None")
    for name, value in (
        ("model_profile_sha256", model_profile_sha256),
        ("selector_policy_binding_id", selector_policy_binding_id),
        ("reflection_policy_binding_id", reflection_policy_binding_id),
    ):
        if value is not None and (type(value) is not str or not value):
            raise ValueError(f"{name} must be a non-empty string or None")

    ids = DeterministicIdFactory(id_namespace)
    memory = InsightMemoryBank(id_factory=ids)
    seed_card = _seed_memory(memory)
    config = compose_timeloop_v2_campaign_workload(
        evaluator_preflight_receipt=evaluator_preflight_receipt,
        resource_lease_receipt=resource_lease_receipt,
        benchmark=benchmark,
    )
    workload_ports = config.build_ports()
    evidence_registry = CampaignEvidenceRegistry()
    learning_runtime = ClosedLoopCampaignLearningRuntime(
        learning=ClosedLoopCampaignLearning(memory=memory),
        reflection_projection=StructuredCampaignReflectionLearningProjector(
            semantic_compiler=PortableFiniteActionInsightSemanticCompiler(),
            scope=HypothesisAuditScope(
                workload_instance_sha256s=(WORKLOAD_DEFINITION_SHA256,),
                evaluator_contract_sha256=evaluator_contract_sha256,
                metric_adjudicator_definition_sha256=(_METRIC_ADJUDICATOR_SHA256),
                campaign_sha256s=(campaign_sha256,),
            ),
            applicable_operator_kinds=("typed_mutation",),
            diagnostic_operator_kind="typed_mutation",
            diagnostic_editable_paths=REFLECTION_DECISION_PATHS,
            initial_score=0.0,
            minimum_support_clusters=2,
            minimum_support_instances=1,
        ),
        generation_auditor=TransactionalPortfolioGenerationAuditor(
            evidence_registry=evidence_registry,
            campaign_sha256=campaign_sha256,
            workload_instance_sha256=WORKLOAD_DEFINITION_SHA256,
            evaluator_contract_sha256=evaluator_contract_sha256,
            metric_projector=ObjectiveDeltaMetricEffectProjector(
                _METRIC_ADJUDICATOR_SHA256
            ),
            action_semantics_compiler=FinitePortfolioActionSemanticsCompiler(),
            hypothesis_matcher=PortableFiniteActionHypothesisMatcher(),
        ),
    )
    evidence_source = CommittedRegistryIdentifiableReflectionEvidenceSource(
        registry=evidence_registry,
        campaign_sha256=campaign_sha256,
        workload_instance_sha256=WORKLOAD_DEFINITION_SHA256,
        evaluator_contract_sha256=evaluator_contract_sha256,
    )
    reflection_executor = (
        _ReflectionExecutor(
            ids=ids,
            optimization_semantics=benchmark.optimization_semantics,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            incompatible_card=incompatible_reflection_card,
        )
        if reflection_executor_factory is None
        else reflection_executor_factory(ids, benchmark.optimization_semantics)
    )
    if any(
        not hasattr(reflection_executor, attribute)
        for attribute in ("reflect", "generations", "records", "inputs", "prompts")
    ):
        raise TypeError("reflection executor does not satisfy the campaign boundary")
    direct_selection = direct_portfolio_selector is not None
    active_target_specification = target_conditioned_specification
    if ACQUISITION_MODE is CampaignAcquisitionMode.TARGET_CONDITIONED:
        if active_target_specification is None:
            active_target_specification = _target_conditioned_specification()
    elif active_target_specification is not None:
        raise ValueError(
            "target-conditioned specification requires its exact acquisition mode"
        )
    if direct_selection:
        selected_runner = direct_portfolio_selector
        selected_allocator = None
        coordinator = None
        selector_delegate = direct_portfolio_selector
    else:
        selected_runner = (
            _ProviderFreeCalibratedRunner()
            if calibrated_runner is None
            else calibrated_runner
        )
        selected_allocator = (
            build_campaign_acquisition_allocator(
                ACQUISITION_MODE,
                common_pool_enabled=_common_pool_enabled(),
                operator_assay_minimum=OPERATOR_ASSAY_MINIMUM,
                family_exposure_phases=(
                    build_controller_owned_family_exposure_phases(
                        family="composite_r2",
                    )
                    if contextual_search_allocation
                    else build_terminal_tapered_family_exposure_phases(
                        family="composite_r2",
                        terminal_wave_index=GENERATION_COUNT - 1,
                    )
                    if ACQUISITION_MODE is CampaignAcquisitionMode.HORIZON_BOUNDED
                    else None
                ),
                target_conditioned_profile=(
                    None
                    if active_target_specification is None
                    else active_target_specification.profile
                ),
            )
            if calibrated_allocator is None
            else calibrated_allocator
        )
        coordinator = CalibratedPortfolioCampaignCoordinator(
            allocator=selected_allocator,
            constraint_decoupled=constraint_decoupled_acquisition,
            minimum_intervention_projection=minimum_intervention_projection,
            evidence_calibrated_source_mix=evidence_calibrated_source_mix,
            contextual_search_allocation=contextual_search_allocation,
        )
        selector_delegate = (
            portfolio_selector_override
            if portfolio_selector_override is not None
            else coordinator.build_selector(selected_runner)
        )
        if (
            portfolio_selector_override is None
            and contextual_search_allocation
            and type(selector_delegate) is not (
                PydanticAIContextualSearchAllocationPortfolioSelectionPolicy
            )
        ):
            raise TypeError("Timeloop coordinator built a foreign V12 selector")
        if (
            portfolio_selector_override is None
            and active_target_specification is not None
        ):
            if type(selected_allocator) is not TargetConditionedSlateAllocatorAdapter:
                raise TypeError("target-conditioned run has a foreign allocator")
            if selected_allocator.profile != active_target_specification.profile:
                raise ValueError("target-conditioned allocator profile drifted")
            if type(selector_delegate) is not (
                PydanticAITargetConditionedCalibratedPortfolioSelectionPolicy
            ):
                raise TypeError("Timeloop coordinator built a foreign T-RAP selector")
    selector = _RecordingSelector(selector_delegate)
    reference_profile_enabled = (
        not direct_selection
        and _common_pool_enabled()
        and COMMON_CANDIDATE_POOL_SIZE is None
    )
    if reference_profile_enabled and model_execution_profile is None:
        raise ValueError(
            "complete-contract reference treatment requires a model execution profile"
        )
    parent_selector = (
        ResidualHypervolumeCampaignParentSelector()
        if reference_profile_enabled and residual_frontier_planning
        else StagnationAwareDiverseCampaignParentSelector()
        if (
            reference_profile_enabled
            and ACQUISITION_MODE is CampaignAcquisitionMode.HORIZON_BOUNDED
        )
        else ArchiveEliteExplorerCampaignParentSelector()
        if reference_profile_enabled
        else ArchiveReservoirCampaignParentSelector(reservoir_limit=8)
    )
    utility = (
        AffineHypervolumeArchiveUtility3D(timeloop_v2_affine_hypervolume_spec())
        if archive_utility is None
        else archive_utility
    )
    archive_context_projector = (
        AuthenticatedAffineFrontierContextProjector()
        if active_target_specification is not None
        or portfolio_selector_override is not None
        else affine_frontier_context_projector(ARCHIVE_CONTEXT_MODE)
    )
    if not direct_selection and type(utility) is not AffineHypervolumeArchiveUtility3D:
        raise TypeError(
            "Timeloop treatment requires exact affine 3-D utility for memory credit"
        )
    experiment_profile: CampaignExperimentProfile | None = None
    if reference_profile_enabled:
        assert model_execution_profile is not None
        context_local_successor = (
            _proposal_support_policy() is not None
            and archive_context_projector is not None
            and VARIATION_TOPOLOGY.mode is not CampaignVariationTopologyMode.FLAT_R2
        )
        experiment_profile = reference_campaign_experiment_profile(
            profile_id=f"reference_timeloop_{model_execution_profile.profile_id}",
            model_execution=model_execution_profile,
            implementations=ReferenceCampaignImplementations(
                parent_selection=parent_selector,
                memory_assignment=learning_runtime,
                # The profile authenticates the engine-owned allocator, while
                # ``selector`` remains the runtime adapter that records and
                # executes its decisions.  Binding the adapter here would hide
                # the exact operator-assay configuration from method identity.
                portfolio_selection=(
                    portfolio_selector_override
                    if portfolio_selector_override is not None
                    else selected_allocator
                ),
                recombination=object(),
                reflection=reflection_executor,
                archive_context=archive_context_projector,
                variation_topology=(
                    _reference_variation_topology_binding(workload_ports.catalog)
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
            evaluator_concurrency=evaluator_concurrency,
            agent_concurrency=agent_concurrency,
            agent_queue_capacity=agent_queue_capacity,
            hierarchical_proposal_support=(_proposal_support_policy() is not None),
            operator_stratified_acquisition=(
                ACQUISITION_MODE is CampaignAcquisitionMode.OPERATOR_STRATIFIED
            ),
            horizon_bounded_acquisition=(
                ACQUISITION_MODE is CampaignAcquisitionMode.HORIZON_BOUNDED
            ),
            constraint_decoupled_acquisition=(constraint_decoupled_acquisition),
            minimum_intervention_projection=minimum_intervention_projection,
            evidence_calibrated_source_mix=evidence_calibrated_source_mix,
            contextual_search_allocation=contextual_search_allocation,
        )
        policies = experiment_profile.behavior(archive_utility=utility).bind()
    else:
        policies = CampaignPolicies(
            cadence=SealedCutoffDelayedAdmissionCadence(),
            parent_selection=_binding("archive_reservoir", parent_selector),
            memory_assignment=_binding("closed_loop_memory", learning_runtime),
            portfolio_selection=_binding(
                (
                    selector_policy_binding_id
                    or (
                        "provider_free_common_pool_model_anchored_m_to_k8_to_k4"
                        if _common_pool_enabled()
                        else "provider_free_full_support_k8_to_k8"
                    )
                ),
                selector,
            ),
            recombination=_binding("disjoint_patch_union", object()),
            reflection=_binding(
                (
                    reflection_policy_binding_id
                    or "provider_free_identifiable_mutation_reflection"
                ),
                reflection_executor,
            ),
            reflection_supervision=CampaignReflectionSupervisionPolicy(
                ReflectionFailureMode.FAIL_AT_NEXT_STAGE_BOUNDARY
            ),
            archive_utility=utility,
        )
    protocol = CampaignProtocol(
        protocol_id=protocol_id,
        protocol_version=1,
        definition_sha256=protocol_definition_sha256,
        outer_seed=outer_seed,
        generation_count=GENERATION_COUNT,
        required_seed_count=2,
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
        runtime=_PreparationRuntime(),
        budget=OptimizerBudget(
            max_unique_evaluations=PLANNED_CANDIDATE_OCCURRENCES,
            max_logical_llm_calls=PLANNED_LOGICAL_CALLS,
            max_generations=GENERATION_COUNT,
        ),
        concurrency=CampaignConcurrency(
            evaluator_concurrency=evaluator_concurrency,
            agent_concurrency=agent_concurrency,
            agent_queue_capacity=agent_queue_capacity,
        ),
        journals=(_PreparationJournal(),),
    ).prepare()
    experiment_profile_conformance = (
        None
        if experiment_profile is None
        else experiment_profile.prepared_conformance_record(
            prepared=prepared,
            archive_utility=utility,
            outer_seed=outer_seed,
        )
    )
    schedule = prepared.schedule
    if (
        schedule.portfolio_generations != (1, 3, 5)
        or schedule.paired_recombination_generations != (2, 4, 6)
        or tuple(step.planned_candidate_evaluations for step in schedule.steps)
        != (
            PARENTS_PER_PORTFOLIO * PORTFOLIO_WIDTH,
            PARENTS_PER_PORTFOLIO * RECOMBINATIONS_PER_PARENT,
            PARENTS_PER_PORTFOLIO * PORTFOLIO_WIDTH,
            PARENTS_PER_PORTFOLIO * RECOMBINATIONS_PER_PARENT,
            PARENTS_PER_PORTFOLIO * PORTFOLIO_WIDTH,
            PARENTS_PER_PORTFOLIO * RECOMBINATIONS_PER_PARENT,
        )
        or tuple(step.planned_agent_calls for step in schedule.steps)
        != (2, 1, 2, 0, 2, 0)
        or schedule.planned_candidate_evaluations + protocol.required_seed_count
        != PLANNED_CANDIDATE_OCCURRENCES
        or schedule.planned_agent_calls != PLANNED_LOGICAL_CALLS
        or tuple(
            (value.source_generation, value.promotion_barrier_generation)
            for value in schedule.reflection_waves
        )
        != ((2, 4),)
    ):
        raise RuntimeError(
            "prepared Timeloop schedule differs from its configured evaluation/call contract"
        )

    benchmark_sha256 = typed_json_sha256(prepared.benchmark_session.benchmark)
    feedback_ledger = PortfolioOutcomeFeedbackLedger()
    binding_factory = (
        None
        if direct_selection
        else CalibratedCampaignBindingFactory(
            scope=ForecastCalibrationScope(
                model_profile_sha256=(
                    model_profile_sha256
                    or _sha("timeloop-provider-free-calibrated-double")
                ),
                prompt_definition_sha256=(
                    portfolio_selector_override.prompt_definition_sha256
                    if portfolio_selector_override is not None
                    else calibrated_portfolio_prompt_definition_sha256(
                        proposal_support=_proposal_support_policy() is not None,
                        hierarchical_composition_required_proposals=(
                            VARIATION_TOPOLOGY.hierarchical_composition_required_proposals
                        ),
                        feasibility_witness_mode=FEASIBILITY_WITNESS_MODE,
                        constraint_decoupled=constraint_decoupled_acquisition,
                    )
                ),
                selector_policy_definition_sha256=(
                    portfolio_selector_override.policy_definition_sha256
                    if portfolio_selector_override is not None
                    else selector_delegate.policy_definition_sha256
                ),
                benchmark_sha256=benchmark_sha256,
                session_sha256=prepared.benchmark_session.session_sha256,
            ),
            objectives=equal_weight_slate_objectives(benchmark.objectives),
            ledger=feedback_ledger,
            common_candidate_pool_policy=(
                TaskKeyedCommonCandidatePoolPolicy(
                    replicate_seed=outer_seed,
                    candidate_pool_size=COMMON_CANDIDATE_POOL_SIZE,
                    model_selection_size=CALIBRATED_PROPOSAL_WIDTH,
                )
                if _common_pool_enabled()
                else None
            ),
            proposal_support_policy=_proposal_support_policy(),
            assign_all_cards_by_default=not _common_pool_enabled(),
        )
    )
    direction_adjudicator = (
        None
        if binding_factory is None
        else AbsoluteToleranceDirectionAdjudicator(
            benchmark_sha256=binding_factory.scope.benchmark_sha256,
            session_sha256=binding_factory.scope.session_sha256,
            resolutions=tuple(
                MetricDirectionResolution(
                    metric_id=metric_id,
                    absolute_tolerance=0.0,
                )
                for metric_id in OBJECTIVE_IDS
            ),
        )
    )
    engine_traces: list[dict[str, object]] = []

    def record_engine_trace(value: dict[str, object]) -> None:
        record = dict(value)
        engine_traces.append(record)
        if engine_trace_sink is not None:
            engine_trace_sink(record)

    composition = compose_portfolio_evolution(
        benchmark,
        generator=_NeverGenerator(),
        selector=selector,
        seed=outer_seed,
        id_factory=ids,
        memory=memory,
        evaluator_concurrency=evaluator_concurrency,
        engine_trace_sink=record_engine_trace,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    target_conditioned_controller = (
        None
        if active_target_specification is None
        else TargetConditionedCampaignOutcomeUpdater(
            state=active_target_specification.initial_state(
                campaign_scope_sha256=prepared.preparation_sha256
            ),
            selected_decision=(
                lambda wave, result: coordinator.decode_target_conditioned_allocation(
                    result
                )
            ),
            selected_context=(
                lambda wave, result: coordinator.decode_target_conditioned_context(
                    result
                )
            ),
            marginal_utility=FixedReferenceContextualMarginalUtilityProjector(
                utility
            ),
        )
    )
    contextual_search_ledger = (
        ContextualSearchLedger()
        if not direct_selection and contextual_search_allocation
        else None
    )
    evaluation_mating_constraints = (
        type(selected_allocator) is not FrontierProbeSlatePolicy
    )
    contextual_search_scope_sha256 = _sha(
        "agent-evolve:contextual-search-campaign:" + prepared.preparation_sha256
    )
    contextual_search_planner = (
        None
        if contextual_search_ledger is None
        else CampaignContextualSearchPlanner(
            ledger=contextual_search_ledger,
            campaign_scope_sha256=contextual_search_scope_sha256,
            joint_capability_projector=(
                FiniteContractContextualJointCapabilityProjector(
                    min_distinct_families=(
                        4 if evaluation_mating_constraints else None
                    ),
                    require_pairwise_disjoint_parent_patches=(
                        evaluation_mating_constraints
                    ),
                    require_declared_source_floor_options=True,
                )
            ),
            frontier_target_allocator=(
                ResidualHypervolumeFrontierTargetAllocator()
                if residual_frontier_planning
                else AuthenticatedAffineFrontierTargetAllocator()
            ),
        )
    )
    diagnostic_coordinator = (
        None
        if direct_selection
        else _TimeloopDiagnosticBlockCoordinator(
            memory=memory,
            learning_runtime=learning_runtime,
            utility=utility,
            outer_seed=outer_seed,
            task_sha256=task_sha256,
        )
    )
    wave_factory = (
        _DirectPortfolioWaveFactory(
            composition=composition,
            seed_card=seed_card,
        )
        if direct_selection
        else _WaveFactory(
            composition=composition,
            learning_runtime=learning_runtime,
            diagnostic_coordinator=diagnostic_coordinator,
            utility=utility,
            seed_card=seed_card,
            binding_factory=binding_factory,
            coordinator=coordinator,
            target_conditioned_controller=target_conditioned_controller,
            evaluation_mating_constraints=evaluation_mating_constraints,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
    )
    calibrated_outcome_updater = (
        None
        if direct_selection
        else CalibratedCampaignOutcomeUpdater(
            ledger=feedback_ledger,
            selected_forecasts=(
                (
                    lambda wave, result: (
                        portfolio_selector_override.decode_selected_predictions(
                            binding_factory.scope,
                            wave,
                            result,
                        )
                    )
                )
                if portfolio_selector_override is not None
                else lambda wave, result: (
                    coordinator.decode_selected_predictions(result)
                )
            ),
            adjudicator_for=lambda wave, result: direction_adjudicator,
            **(
                {}
                if contextual_search_ledger is None
                else {
                    "contextual_ledger": contextual_search_ledger,
                    "selected_search_sources": (
                        lambda wave, result: coordinator.decode_selected_source_ids(
                            result
                        )
                    ),
                    "selected_allocation_realization": (
                        lambda wave, result: (
                            coordinator.decode_contextual_allocation_realization(result)
                        )
                    ),
                    "contextual_marginal_utility": (
                        FixedReferenceContextualMarginalUtilityProjector(utility)
                    ),
                    "contextual_campaign_scope_sha256": (
                        contextual_search_scope_sha256
                    ),
                }
            ),
        )
    )
    outcome_updater = (
        CompositeCampaignPortfolioOutcomeUpdater(
            (calibrated_outcome_updater, target_conditioned_controller)
        )
        if target_conditioned_controller is not None
        else calibrated_outcome_updater
    )
    runtime = AgenticPortfolioCampaignRuntime(
        prepared=prepared,
        workload_config=config,
        workload_ports=workload_ports,
        composition=composition,
        parent_selector=parent_selector,
        wave_factory=wave_factory,
        task_sha256=task_sha256,
        archive_context_projector=(
            archive_context_projector
            if experiment_profile is None
            else experiment_profile.archive_context_projector
        ),
        memory_estimand_projector=diagnostic_coordinator,
        learning_lifecycle=learning_runtime,
        identifiable_reflection_executor=reflection_executor,
        identifiable_reflection_evidence_source=evidence_source,
        context_enricher=(
            None
            if direct_selection
            else ContextualOutcomeCampaignEnricher(
                ledger=feedback_ledger,
                max_actions=24,
                include_cross_lineage_analogies=False,
            )
        ),
        contextual_search_planner=contextual_search_planner,
        frontier_target_allocator=(
            ResidualHypervolumeFrontierTargetAllocator()
            if portfolio_selector_override is not None
            else AuthenticatedAffineFrontierTargetAllocator()
            if target_conditioned_controller is not None
            else None
        ),
        outcome_updater=outcome_updater,
        selector_request_prompt_renderer=(
            None
            if direct_selection
            else portfolio_selector_override
            if portfolio_selector_override is not None
            else coordinator
        ),
        recombination_utility_binder=recombination_utility_binder,
        owned_resources=owned_resources,
    )
    journal = _ExecutionJournal() if execution_journal is None else execution_journal
    execution = asyncio.run(
        EvolutionCampaignScheduler(
            prepared=prepared,
            policies=policies,
            stages=runtime,
            reflections=runtime,
            lifecycle=runtime,
            journal=journal,
        ).run()
    )
    return ProviderFreeTimeloopCampaignRun(
        execution=execution,
        evaluator=evaluator,
        memory=memory,
        evidence_registry=evidence_registry,
        reflection_executor=reflection_executor,
        wave_factory=wave_factory,
        calibrated_runner=selected_runner,
        selector=selector,
        feedback_ledger=feedback_ledger,
        contextual_search_ledger=contextual_search_ledger,
        contextual_search_planner=contextual_search_planner,
        target_conditioned_controller=target_conditioned_controller,
        journal=journal,
        engine_traces=engine_traces,
        execution_mode=execution_mode,
        docker_enabled=docker_enabled,
        provider_enabled=provider_enabled,
        scientific_claim=scientific_claim,
        experiment_profile=experiment_profile,
        experiment_profile_conformance=experiment_profile_conformance,
    )


def run_provider_free_timeloop_campaign(
    *,
    outer_seed: int = OUTER_SEED,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float | None = TEMPERATURE,
    id_namespace: str | None = None,
    incompatible_reflection_card: bool = False,
    calibrated_allocator: CalibratedPortfolioAllocator | None = None,
    target_conditioned_specification: (
        TargetConditionedCampaignSpecification | None
    ) = None,
    direct_portfolio_selector: object | None = None,
    archive_utility: object | None = None,
    recombination_utility_binder: object | None = None,
    execution_journal: object | None = None,
) -> ProviderFreeTimeloopCampaignRun:
    """Run six actual generations without a provider, credential, or Docker."""

    evaluator = _DeterministicEvaluator()
    panel = frozen_network_panel("resnet50")
    problem = TimeloopV2CoDesignProblem(
        TimeloopV2Settings(
            output_root=Path(__file__).resolve().parent / "provider_free"
        ),
        panel,
        evaluator=evaluator,
    )
    benchmark = AgenticBenchmark(
        problem=problem,
        optimization_semantics=timeloop_v2_optimization_semantics(problem),
        phenotype_identity=TypedConfigurationPhenotypeIdentityPolicy(),
        finite_variation_catalogs=(TimeloopV2FiniteVariationCatalog(panel),),
    )
    return run_timeloop_campaign(
        benchmark=benchmark,
        evaluator=evaluator,
        execution_mode="deterministic_provider_free_calibrated_double",
        id_namespace=(
            id_namespace
            if id_namespace is not None
            else (
                f"timeloop_v2_provider_free_g6_direct_{outer_seed}"
                if direct_portfolio_selector is not None
                else (
                    "timeloop_v2_provider_free_g6_incompatible"
                    if incompatible_reflection_card
                    else "timeloop_v2_provider_free_g6"
                )
            )
        ),
        campaign_sha256=_CAMPAIGN_SHA256,
        evaluator_contract_sha256=_EVALUATOR_CONTRACT_SHA256,
        protocol_id="timeloop_v2_provider_free_g6_delayed_identifiable",
        protocol_definition_sha256=_sha(
            "timeloop-v2-provider-free-g6-delayed-identifiable-v1"
        ),
        task_sha256=_sha("timeloop-v2-provider-free-g6-task-v1"),
        evaluator_preflight_receipt=_object(
            {
                "qualified": True,
                "mode": "deterministic_provider_free_double",
                "docker_calls": 0,
            }
        ),
        resource_lease_receipt=_object(
            {"resource": "provider_free_serial_timeloop_slot", "active": True}
        ),
        docker_enabled=False,
        scientific_claim="structural_conformance_only",
        outer_seed=outer_seed,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        incompatible_reflection_card=incompatible_reflection_card,
        calibrated_allocator=calibrated_allocator,
        target_conditioned_specification=target_conditioned_specification,
        direct_portfolio_selector=direct_portfolio_selector,
        archive_utility=archive_utility,
        recombination_utility_binder=recombination_utility_binder,
        execution_journal=execution_journal,
        model_execution_profile=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
        selector_policy_binding_id=(
            "provider_free_direct_control"
            if direct_portfolio_selector is not None
            else (
                None
                if calibrated_allocator is None
                else "provider_free_injected_calibrated_allocator"
            )
        ),
    )


def main() -> None:
    run = run_provider_free_timeloop_campaign()
    print(json.dumps(run.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
