"""One-call K=8 proposal followed by prior-calibrated K=4 allocation.

The adapter is deliberately an opt-in v2 selector.  It translates the model's
structured slate into the workload-neutral calibrated-slate policy, then
returns the same four-member ``PortfolioSelectionResult`` consumed by
``PortfolioEvolution``.  Evaluators therefore never know that eight actions
were proposed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations
from typing import Annotated, Any, ClassVar, Literal, TypeAlias, Union, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    create_model,
    model_validator,
)
from pydantic_core import PydanticCustomError

from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.finite_variation import (
    validated_finite_variation_identity_index,
)
from agent_evolve.domain.llm_task_queue import ValidationIssueReasonCode
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.application.action_structural_signature import (
    parent_relative_changed_paths_by_option,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
    LowLevelRunner,
)
from agent_evolve.policies.selection.calibrated_slate import (
    CalibratedSlate,
    CalibratedSlateMember,
    SlateAllocationDecision,
    SlateAllocationMode,
    SlateAllocationRequest,
    SlateRoleProposal,
    TraceCalibratedSlatePolicy,
)
from agent_evolve.policies.selection.acquisition_certified_slate import (
    AcquisitionCertifiedSlateContext,
    AcquisitionCertifiedSlateDecision,
    AcquisitionCertifiedSlatePolicy,
)
from agent_evolve.policies.selection.regret_bounded_slate import (
    RegretBoundedSlateDecision,
    RegretBoundedSlatePolicy,
)
from agent_evolve.policies.selection.model_anchored_slate import (
    ModelAnchoredCalibratedSlatePolicy,
    ModelAnchoredSlateDecision,
)
from agent_evolve.policies.selection.structural_posterior_slate import (
    FamilyExposureBound,
    FamilyExposurePhase,
    HorizonBoundedStructuralPosteriorSlateDecision,
    HorizonBoundedStructuralPosteriorSlatePolicy,
    OperatorStratifiedStructuralPosteriorSlateDecision,
    OperatorStratifiedStructuralPosteriorSlatePolicy,
    StructuralPosteriorSlateDecision,
    StructuralPosteriorSlatePolicy,
)
from agent_evolve.policies.selection.frontier_probe_slate import (
    FrontierProbeSlateDecision,
    FrontierProbeSlatePolicy,
)
from agent_evolve.policies.selection.full_support_slate import (
    FullSupportSlatePolicy,
)
from agent_evolve.policies.selection.target_conditioned_allocator import (
    ADAPTER_DEFINITION_SHA256 as TARGET_CONDITIONED_ALLOCATOR_DEFINITION_SHA256,
    ADAPTER_ID as TARGET_CONDITIONED_ALLOCATOR_ID,
    ADAPTER_VERSION as TARGET_CONDITIONED_ALLOCATOR_VERSION,
    TargetConditionedSlateAllocatorAdapter,
)
from agent_evolve.policies.selection.target_conditioned_prequential import (
    POLICY_DEFINITION_SHA256 as TARGET_CONDITIONED_CORE_DEFINITION_SHA256,
    POLICY_ID as TARGET_CONDITIONED_CORE_ID,
    POLICY_VERSION as TARGET_CONDITIONED_CORE_VERSION,
    TargetConditionedAcquisitionProfile,
    TargetConditionedSlateDecision,
)
from agent_evolve.policies.selection.target_conditioned_features import (
    TargetConditionedPortableFeatureProjector,
)
from agent_evolve.policies.variation.compositional_finite_catalog import (
    COMPOSITION_LEFT_OPTION_METADATA_KEY,
    COMPOSITION_REQUIRED_PROPOSALS_METADATA_KEY,
    COMPOSITION_RIGHT_OPTION_METADATA_KEY,
    COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY,
    CompositionSelectionExposure,
)
from agent_evolve.policies.variation.source_union_finite_catalog import (
    required_ranked_source_evaluation_option_ids,
    required_source_evaluation_option_ids,
)
from agent_evolve.policies.variation.exact_composition_capacity import (
    ExactKCompositionCapacityProjection,
    project_exact_k_binary_composition,
)
from agent_evolve.policies.selection.calibrated_portfolio_binding import (
    CalibratedPortfolioAllocationContext,
    CalibratedPortfolioBindingProvider,
    CalibratedPortfolioInputBinding,
)
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastConfidenceBin,
    ForecastPredictionReceipt,
)
from agent_evolve.policies.selection.finite_option_prompt_projection import (
    FiniteOptionPromptProjectionPolicy,
)
from agent_evolve.policies.selection.memory_dose_feasibility import (
    MemoryDoseAttributionFeasibilityWitness,
    find_memory_dose_attribution_feasibility_witness,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.portfolio_memory_dose import (
    BoundedPortfolioMemoryDoseContract,
    PortfolioMemoryDoseMember,
    PortfolioMemoryDoseStage,
    assess_evaluated_portfolio_memory_dose,
    assess_proposed_portfolio_memory_dose,
    require_passing_portfolio_memory_dose,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioMemberDraft,
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
    PortfolioSelectionSupplementalAudit,
    RankedPortfolioDecision,
    finite_option_ids_have_pairwise_disjoint_parent_patch_subset,
    pairwise_disjoint_parent_patch_witness,
    pairwise_disjoint_parent_patch_pairs,
    project_family_exposure_bounds_to_pairwise_disjoint_feasibility,
    resolve_ranked_portfolio_decision,
)
from agent_evolve.ports.structured_generator import (
    MAX_STRUCTURED_REPAIR_CONTEXT_UTF8_BYTES,
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    StructuredOutputRepairLiteralSet,
)
from agent_evolve.ports.variation_source import (
    finite_variation_operator_by_option,
    finite_variation_operator_id,
    finite_variation_source_by_option,
    finite_variation_source_minimum_counts,
)


CALIBRATED_PORTFOLIO_PROPOSAL_SIZE = 8
CALIBRATED_PORTFOLIO_EVALUATION_SIZE = 4
# Keep bounded finite contracts in the provider-visible schema whenever the
# complete enum remains modest.  All current reference workloads fit below
# this ceiling (the 200-option BOiLS contract is about 4.7 KiB of literals),
# while genuinely large contracts retain the exact fail-closed local gate.
MAX_INLINE_OPTION_ENUM_UTF8_BYTES = 8_192
CALIBRATED_PORTFOLIO_SELECTION_TOOL_NAME = "propose_calibrated_portfolio_slate"
CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID = "pydantic_ai_calibrated_portfolio"
CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION = 2
CALIBRATED_PORTFOLIO_BASE_INSTRUCTION = (
    "Propose a workload-grounded portfolio slate from the sealed finite options. "
    "Use only the supplied context, option records, and prospectively assigned "
    "memory cards. Follow the output contract exactly; do not invent actions, "
    "facts, card citations, or evidence."
)


class CalibratedPortfolioFeasibilityWitnessMode(str, Enum):
    """Closed prompt treatment for engine-generated feasibility witnesses."""

    CANONICAL = "canonical"
    REQUEST_KEYED = "request_keyed"
    HIDDEN_CERTIFICATE = "hidden_certificate"
    TASK_KEYED_COMMON_POOL = "task_keyed_common_pool"


CALIBRATED_PORTFOLIO_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v7\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00caller-instruction-rendered=false;sealed-context-options-cards=true;"
    b"k8-direction-confidence-role-rationale=true;"
    b"hard-engine-allocation-feasibility-rendered=true;"
    b"engine-verified-structural-witness-rendered=true;"
    b"witness-objective-values-consulted=false"
).hexdigest()
CALIBRATED_PORTFOLIO_PROJECTED_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v8\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00caller-instruction-rendered=false;sealed-context-options-cards=true;"
    b"k8-direction-confidence-role-rationale=true;"
    b"binding-owned-option-projection=true;projection-receipt-rendered=true;"
    b"projected-binding-schema=2;projected-machine-contract-schema=3;"
    b"hard-engine-allocation-feasibility-rendered=true;"
    b"engine-verified-structural-witness-rendered=true;"
    b"witness-objective-values-consulted=false"
).hexdigest()
CALIBRATED_PORTFOLIO_BOUNDED_DOSE_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v9\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00caller-instruction-rendered=false;sealed-context-options-cards=true;"
    b"k8-direction-confidence-role-rationale=true;"
    b"bounded-relevance-aware-memory-dose=true;"
    b"prompt-wide-exploration-not-blinded-control=true;"
    b"hard-engine-allocation-feasibility-rendered=true;"
    b"engine-verified-structural-witness-rendered=true;"
    b"witness-objective-values-consulted=false"
).hexdigest()
CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v10\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode(
        "utf-8",
        errors="strict",
    )
    + b"\x00caller-instruction-rendered=false;"
    b"sealed-context-options-cards=true;"
    b"k8-direction-confidence-role-rationale=true;"
    b"binding-owned-option-projection=true;projection-receipt-rendered=true;"
    b"bounded-relevance-aware-memory-dose=true;"
    b"prompt-wide-exploration-not-blinded-control=true;"
    b"hard-engine-allocation-feasibility-rendered=true;"
    b"engine-verified-structural-witness-rendered=true;"
    b"witness-objective-values-consulted=false"
).hexdigest()
CALIBRATED_PORTFOLIO_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v11\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00caller-instruction-rendered=false;"
    b"sealed-context-options-cards=true;"
    b"k8-direction-confidence-role-rationale=true;"
    b"hard-engine-allocation-feasibility-rendered=true;"
    b"engine-verified-structural-witness-rendered=true;"
    b"witness-order=request-keyed-domain-separated-sha256;"
    b"witness-objective-values-consulted=false"
).hexdigest()
CALIBRATED_PORTFOLIO_PROJECTED_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v12\x00"
        + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
        + b"\x00caller-instruction-rendered=false;"
        b"sealed-context-options-cards=true;"
        b"k8-direction-confidence-role-rationale=true;"
        b"binding-owned-option-projection=true;projection-receipt-rendered=true;"
        b"projected-binding-schema=2;projected-machine-contract-schema=8;"
        b"hard-engine-allocation-feasibility-rendered=true;"
        b"engine-verified-structural-witness-rendered=true;"
        b"witness-order=request-keyed-domain-separated-sha256;"
        b"witness-objective-values-consulted=false"
    ).hexdigest()
)
CALIBRATED_PORTFOLIO_BOUNDED_DOSE_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v13\x00"
        + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
        + b"\x00caller-instruction-rendered=false;"
        b"sealed-context-options-cards=true;"
        b"k8-direction-confidence-role-rationale=true;"
        b"bounded-relevance-aware-memory-dose=true;"
        b"prompt-wide-exploration-not-blinded-control=true;"
        b"hard-engine-allocation-feasibility-rendered=true;"
        b"engine-verified-structural-witness-rendered=true;"
        b"witness-order=request-keyed-domain-separated-sha256;"
        b"witness-objective-values-consulted=false"
    ).hexdigest()
)
CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v14\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00caller-instruction-rendered=false;"
    b"sealed-context-options-cards=true;"
    b"k8-direction-confidence-role-rationale=true;"
    b"binding-owned-option-projection=true;projection-receipt-rendered=true;"
    b"bounded-relevance-aware-memory-dose=true;"
    b"prompt-wide-exploration-not-blinded-control=true;"
    b"hard-engine-allocation-feasibility-rendered=true;"
    b"engine-verified-structural-witness-rendered=true;"
    b"witness-order=request-keyed-domain-separated-sha256;"
    b"witness-objective-values-consulted=false"
).hexdigest()
CALIBRATED_PORTFOLIO_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v27\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00sealed-context-options-cards=true;"
    b"hidden-feasibility-certificate=true;certificate-members-rendered=false;"
    b"hard-engine-allocation-feasibility-rendered=true;"
    b"witness-objective-values-consulted=false"
).hexdigest()
CALIBRATED_PORTFOLIO_PROJECTED_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v28\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00sealed-context-options-cards=true;binding-owned-option-projection=true;"
    b"hidden-feasibility-certificate=true;certificate-members-rendered=false;"
    b"hard-engine-allocation-feasibility-rendered=true;"
    b"witness-objective-values-consulted=false"
).hexdigest()
CALIBRATED_PORTFOLIO_BOUNDED_DOSE_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v29\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00sealed-context-options-cards=true;bounded-relevance-aware-memory-dose=true;"
    b"hidden-feasibility-certificate=true;certificate-members-rendered=false;"
    b"hard-engine-allocation-feasibility-rendered=true;"
    b"witness-objective-values-consulted=false"
).hexdigest()
CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v30\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00sealed-context-options-cards=true;binding-owned-option-projection=true;"
    b"bounded-relevance-aware-memory-dose=true;hidden-feasibility-certificate=true;"
    b"certificate-members-rendered=false;hard-engine-allocation-feasibility-rendered=true;"
    b"witness-objective-values-consulted=false"
).hexdigest()
CALIBRATED_PORTFOLIO_COMMON_POOL_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v23\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00task-keyed-candidate-universe=true;select-exactly-eight=true;"
    b"model-and-provider-fields-in-entropy=false;outcomes-in-entropy=false;"
    b"universe-may-exceed-eight=true;hidden-feasibility-certificate=true;"
    b"nested-universe-state-identity-rendered=true"
).hexdigest()
CALIBRATED_PORTFOLIO_PROJECTED_COMMON_POOL_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v24\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00task-keyed-candidate-universe=true;select-exactly-eight=true;"
    b"binding-owned-option-projection=true;universe-may-exceed-eight=true;"
    b"model-and-provider-fields-in-entropy=false;outcomes-in-entropy=false;"
    b"hidden-feasibility-certificate=true;"
    b"nested-universe-state-identity-rendered=true"
).hexdigest()
CALIBRATED_PORTFOLIO_BOUNDED_DOSE_COMMON_POOL_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v25\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00task-keyed-candidate-universe=true;select-exactly-eight=true;"
    b"bounded-relevance-aware-memory-dose=true;universe-may-exceed-eight=true;"
    b"model-and-provider-fields-in-entropy=false;outcomes-in-entropy=false;"
    b"hidden-feasibility-certificate=true;"
    b"nested-universe-state-identity-rendered=true"
).hexdigest()
CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_COMMON_POOL_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v26\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00task-keyed-candidate-universe=true;select-exactly-eight=true;"
    b"binding-owned-option-projection=true;bounded-relevance-aware-memory-dose=true;"
    b"universe-may-exceed-eight=true;model-and-provider-fields-in-entropy=false;"
    b"outcomes-in-entropy=false;hidden-feasibility-certificate=true;"
    b"nested-universe-state-identity-rendered=true"
).hexdigest()
CALIBRATED_PORTFOLIO_PROPOSAL_SUPPORT_COMMON_POOL_PROMPT_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v31\x00"
        + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
        + b"\x00task-keyed-candidate-universe=true;select-exactly-eight=true;"
        b"proposal-support-reservations=archive-novelty,structural-coverage;"
        b"reservations-force-evaluator-slots=false;universe-may-exceed-eight=true;"
        b"model-and-provider-fields-in-entropy=false;outcomes-in-entropy=false;"
        b"hidden-feasibility-certificate=true;"
        b"nested-universe-state-identity-rendered=true"
    ).hexdigest()
)
CALIBRATED_PORTFOLIO_PROJECTED_PROPOSAL_SUPPORT_COMMON_POOL_PROMPT_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v32\x00"
        + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
        + b"\x00task-keyed-candidate-universe=true;select-exactly-eight=true;"
        b"binding-owned-option-projection=true;"
        b"proposal-support-reservations=archive-novelty,structural-coverage;"
        b"reservations-force-evaluator-slots=false;universe-may-exceed-eight=true;"
        b"model-and-provider-fields-in-entropy=false;outcomes-in-entropy=false;"
        b"hidden-feasibility-certificate=true;"
        b"nested-universe-state-identity-rendered=true"
    ).hexdigest()
)
CALIBRATED_PORTFOLIO_BOUNDED_DOSE_PROPOSAL_SUPPORT_COMMON_POOL_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v33\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00task-keyed-candidate-universe=true;select-exactly-eight=true;"
    b"bounded-relevance-aware-memory-dose=true;"
    b"proposal-support-reservations=archive-novelty,structural-coverage;"
    b"reservations-force-evaluator-slots=false;universe-may-exceed-eight=true;"
    b"model-and-provider-fields-in-entropy=false;outcomes-in-entropy=false;"
    b"hidden-feasibility-certificate=true;"
    b"nested-universe-state-identity-rendered=true"
).hexdigest()
CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_PROPOSAL_SUPPORT_COMMON_POOL_PROMPT_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio-prompt:v34\x00"
    + CALIBRATED_PORTFOLIO_BASE_INSTRUCTION.encode("utf-8", errors="strict")
    + b"\x00task-keyed-candidate-universe=true;select-exactly-eight=true;"
    b"binding-owned-option-projection=true;bounded-relevance-aware-memory-dose=true;"
    b"proposal-support-reservations=archive-novelty,structural-coverage;"
    b"reservations-force-evaluator-slots=false;universe-may-exceed-eight=true;"
    b"model-and-provider-fields-in-entropy=false;outcomes-in-entropy=false;"
    b"hidden-feasibility-certificate=true;"
    b"nested-universe-state-identity-rendered=true"
).hexdigest()
CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-calibrated-portfolio:v2;"
    b"one-call-k8-sealed-proposal=true;engine-k4-allocation=true;"
    b"confidence-and-role-proposal=true;prior-only-calibration=true;"
    b"structural-evidence-injected=true;material-card-administration=true;"
    b"precall-request-evidence-binding=true;caller-instruction-rendered=false;"
    b"legacy-v1-selector-unchanged=true"
).hexdigest()

MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_model_anchored_calibrated_portfolio"
)
MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION = 1
MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-model-anchored-calibrated-portfolio:v1;"
    b"one-call-k8-sealed-proposal=true;engine-k4-allocation=true;"
    b"allocator=model-anchored-prior-calibrated;"
    b"allocator-configuration-authenticated=true;"
    b"confidence-and-role-proposal=true;prior-only-calibration=true;"
    b"structural-evidence-injected=true;material-card-administration=true;"
    b"precall-request-evidence-binding=true;caller-instruction-rendered=false;"
    b"legacy-v2-selector-unchanged=true"
).hexdigest()

STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_structural_posterior_calibrated_portfolio"
)
STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION = 2
STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-structural-posterior-calibrated-portfolio:v2;"
        b"one-call-k8-sealed-proposal=true;engine-k4-allocation=true;"
        b"allocator=calibrated-frontier-four-role-v2;"
        b"allocator-identity-authenticated=true;"
        b"confidence-abstention-and-role-proposal=true;"
        b"prior-only-calibration=true;structural-evidence-injected=true;"
        b"below-chance-forecast-inversion=true;"
        b"model-card-citations=diagnostic-only;"
        b"precall-request-evidence-binding=true;"
        b"caller-instruction-rendered=false;legacy-selectors-unchanged=true"
    ).hexdigest()
)

OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_operator_stratified_calibrated_portfolio"
)
OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION = 1
OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-operator-stratified-calibrated-portfolio:v1;"
        b"one-call-k8-sealed-proposal=true;engine-k4-allocation=true;"
        b"allocator=operator-stratified-calibrated-frontier-four-role-v1;"
        b"allocator-identity-and-assay-minimums-authenticated=true;"
        b"assay-minimums-are-not-quality-priors=true;"
        b"prior-only-calibration=true;structural-evidence-injected=true;"
        b"model-card-citations=diagnostic-only;"
        b"precall-request-evidence-binding=true;caller-instruction-rendered=false;"
        b"legacy-selectors-unchanged=true"
    ).hexdigest()
)

HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_horizon_bounded_calibrated_portfolio"
)
HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION = 1
HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-horizon-bounded-calibrated-portfolio:v1;"
        b"one-call-k8-sealed-proposal=true;engine-k4-allocation=true;"
        b"allocator=horizon-bounded-calibrated-frontier-four-role-v1;"
        b"allocator-identity-and-exposure-phases-authenticated=true;"
        b"phase-index=sealed-wave-index;lower-and-upper-family-bounds=true;"
        b"exposure-bounds-are-not-quality-priors=true;"
        b"infeasible-bounds=minimum-l1-structural-recourse;"
        b"prior-only-calibration=true;structural-evidence-injected=true;"
        b"model-card-citations=diagnostic-only;"
        b"precall-request-evidence-binding=true;caller-instruction-rendered=false;"
        b"legacy-selectors-unchanged=true"
    ).hexdigest()
)

CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_constraint_decoupled_horizon_portfolio"
)
CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_VERSION = 1
CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-constraint-decoupled-horizon-portfolio:v1;"
        b"model-authority=local-semantic-preference;"
        b"engine-authority=dedupe-support-composition-dose-feasibility-refill;"
        b"allocator=horizon-bounded-calibrated-frontier-four-role-v1;"
        b"original-and-reconciled-proposals-authenticated=true;"
        b"unknown-forecast-for-engine-insertions=true;"
        b"objective-values-consulted-by-reconciliation=false;"
        b"deterministic-lane-fallback=false;legacy-selectors-unchanged=true"
    ).hexdigest()
)
MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_minimum_intervention_horizon_portfolio"
)
MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_VERSION = 1
MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-minimum-intervention-horizon-portfolio:v1;"
        b"model-authority=local-semantic-hypothesis;"
        b"engine-authority=dedupe-support-composition-dose-feasibility-refill;"
        b"projection=max-retained-model-count-then-model-rank;"
        b"canonical-order=final-tie-break-only;"
        b"allocator=horizon-bounded-calibrated-frontier-four-role-v1;"
        b"original-reconciled-and-intervention-receipts-authenticated=true;"
        b"unknown-forecast-for-engine-insertions=true;"
        b"objective-values-consulted-by-reconciliation=false;"
        b"deterministic-lane-fallback=false;legacy-selectors-unchanged=true"
    ).hexdigest()
)
EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_evidence_calibrated_source_mix_portfolio"
)
EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_VERSION = 1
EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-evidence-calibrated-source-mix-portfolio:v1;"
        b"model-authority=local-semantic-hypothesis;"
        b"engine-authority=feasibility-soft-support-source-floor-allocation;"
        b"projection=protected-task-keyed-global-source-then-model-retention;"
        b"protected-source-phase=wave-one;protected-source-count=one;"
        b"soft-conflicts=proposal-support;hard=feasibility-composition-memory;"
        b"allocator=horizon-bounded-calibrated-frontier-four-role-v1;"
        b"required-allocation-membership-authenticated=true;"
        b"source-and-intervention-receipts-authenticated=true;"
        b"objective-values-consulted-by-source-mix=false;"
        b"workload-identifiers-consulted=false;legacy-selectors-unchanged=true"
    ).hexdigest()
)
CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_contextual_search_allocation_portfolio"
)
CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_VERSION = 3
CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-contextual-search-allocation-portfolio:v3;"
        b"model-authority=local-semantic-hypothesis;"
        b"engine-authority=feasibility-source-operator-allocation;"
        b"allocation=authenticated-prior-only-phase-aware-contract;"
        b"allocation-scope=stage-global-request-sliced-requested-k4;"
        b"structural-recourse=authenticated-minimum-l1-feasible-projection;"
        b"realized-allocation-membership=exact-k4;"
        b"channels=normalized-marginal-persistence-descendant-feasibility;"
        b"soft-conflicts=proposal-support;hard=feasibility-composition-memory;"
        b"post-recourse-composition=nearest-exact-k-capacity-projection;"
        b"allocator=horizon-bounded-calibrated-frontier-four-role-v1;"
        b"objective-values-consulted-by-reconciliation=false;"
        b"workload-model-provider-identifiers-consulted=false;"
        b"legacy-selectors-unchanged=true"
    ).hexdigest()
)
_CONSTRAINT_DECOUPLED_PROMPT_DOMAIN = (
    b"agent-evolve:constraint-decoupled-semantic-slate-prompt:v1\x00"
)

FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_frontier_probe_calibrated_portfolio"
)
FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION = 1
FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-frontier-probe-calibrated-portfolio:v1;"
    b"one-call-k8-sealed-proposal=true;engine-k4-allocation=true;"
    b"allocator=model-anchored-full-abstention-frontier-probe;"
    b"allocator-and-constraint-projection-authenticated=true;"
    b"full-vector-abstention-distinct-from-partial=true;"
    b"phenotype-uniqueness-and-bounded-memory-dose=hard;"
    b"evaluation-allocation-decoupled-from-mating-compatibility=true;"
    b"confidence-abstention-and-role-proposal=true;"
    b"structural-evidence-injected=true;"
    b"precall-request-evidence-binding=true;"
    b"caller-instruction-rendered=false;legacy-selectors-unchanged=true"
).hexdigest()
FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_full_support_calibrated_portfolio"
)
FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION = 1
FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-full-support-calibrated-portfolio:v1;"
    b"one-call-k8-sealed-proposal=true;engine-k8-evaluation=true;"
    b"allocator=full-authenticated-slate;model-order-preserved=true;"
    b"confidence-and-role-proposal=true;structural-evidence-injected=true;"
    b"bounded-memory-dose-supported=true;"
    b"precall-request-evidence-binding=true;caller-instruction-rendered=false;"
    b"legacy-k8-to-k4-selectors-unchanged=true"
).hexdigest()
TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_target_conditioned_calibrated_portfolio"
)
TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION = 1
TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-target-conditioned-calibrated-portfolio:v1;"
        b"one-call-k8-sealed-proposal=true;engine-k4-allocation=true;"
        b"allocator=target-conditioned-prequential-realizable-portfolio-v1;"
        b"context=append-only-authenticated-precall-branch;"
        b"features=portable-typed-configuration-transitions;"
        b"state=selected-only-generation-barrier;"
        b"realizability=injectable-complete-finite-set;"
        b"workload-model-provider-current-outcome-fields=false;"
        b"legacy-selectors-unchanged=true"
    ).hexdigest()
)
CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_constraint_decoupled_target_conditioned_portfolio"
)
CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_VERSION = 2
CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-constraint-decoupled-target-conditioned-"
        b"portfolio:v2;model-authority=local-semantic-hypothesis;"
        b"engine-authority=dedupe-support-composition-dose-feasibility-refill;"
        b"allocator=target-conditioned-prequential-realizable-portfolio-v1;"
        b"evaluation-source-floor=sealed-contract-task-keyed;"
        b"independent-evaluation-does-not-require-disjoint-parent-patches=true;"
        b"original-and-reconciled-proposals-authenticated=true;"
        b"unknown-forecast-for-engine-insertions=true;"
        b"objective-values-consulted-by-reconciliation=false;"
        b"workload-model-provider-identifiers-consulted=false;"
        b"legacy-target-conditioned-selector-unchanged=true"
    ).hexdigest()
)
ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_acquisition_certified_residual_portfolio"
)
ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_VERSION = 1
ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-acquisition-certified-residual-portfolio:v1;"
        b"model-authority=local-semantic-residual-hypothesis;"
        b"engine-authority=dedupe-feasibility-reference-reservation;"
        b"proposal=complete-numerical-reference-plus-model-residuals-k8;"
        b"allocation=common-realization-qlognehvi-exact-feasible-k4;"
        b"reference-retained-on-ties=true;strictly-prior-outcomes=true;"
        b"certificate-scope=acquisition-not-unseen-evaluator-outcome;"
        b"workload-model-provider-identifiers-consulted=false;"
        b"legacy-selectors-unchanged=true"
    ).hexdigest()
)
REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_ID = (
    "pydantic_ai_regret_bounded_information_portfolio"
)
REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_VERSION = 1
REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = (
    hashlib.sha256(
        b"agent-evolve:pydantic-ai-regret-bounded-information-portfolio:v1;"
        b"model-authority=local-semantic-residual-hypothesis;"
        b"engine-authority=dedupe-feasibility-reference-regret-envelope;"
        b"proposal=complete-numerical-reference-plus-model-residuals-k8;"
        b"allocation=frozen-acquisition-plus-authenticated-future-value;"
        b"one-step-retention-envelope=explicit;development-assay=typed;"
        b"workload-model-provider-identifiers-consulted=false"
    ).hexdigest()
)


CalibratedPortfolioAllocator: TypeAlias = Union[
    TraceCalibratedSlatePolicy,
    ModelAnchoredCalibratedSlatePolicy,
    StructuralPosteriorSlatePolicy,
    OperatorStratifiedStructuralPosteriorSlatePolicy,
    HorizonBoundedStructuralPosteriorSlatePolicy,
    FrontierProbeSlatePolicy,
    FullSupportSlatePolicy,
    TargetConditionedSlateAllocatorAdapter,
    AcquisitionCertifiedSlatePolicy,
    RegretBoundedSlatePolicy,
]
CalibratedPortfolioAllocationDecision: TypeAlias = Union[
    SlateAllocationDecision,
    ModelAnchoredSlateDecision,
    StructuralPosteriorSlateDecision,
    OperatorStratifiedStructuralPosteriorSlateDecision,
    HorizonBoundedStructuralPosteriorSlateDecision,
    FrontierProbeSlateDecision,
    TargetConditionedSlateDecision,
    AcquisitionCertifiedSlateDecision,
    RegretBoundedSlateDecision,
]


@dataclass(frozen=True, slots=True)
class _CalibratedPortfolioSelectionProfile:
    audit_kind: str
    payload_schema_version: int
    event_type: str
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    constraint_decoupled: bool = False
    minimum_intervention_projection: bool = False
    evidence_calibrated_source_mix: bool = False
    contextual_search_allocation: bool = False
    acquisition_certified_residual: bool = False
    regret_bounded_information: bool = False


_FOUR_ROLE_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="calibrated_portfolio_k8_to_k4",
    payload_schema_version=2,
    event_type="calibrated_portfolio_k8_to_k4",
    policy_id=CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION,
    policy_definition_sha256=(CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256),
)
_MODEL_ANCHORED_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="model_anchored_calibrated_portfolio_k8_to_k4",
    payload_schema_version=3,
    event_type="model_anchored_calibrated_portfolio_k8_to_k4",
    policy_id=MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION,
    policy_definition_sha256=(
        MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
)
_STRUCTURAL_POSTERIOR_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="structural_posterior_calibrated_portfolio_k8_to_k4",
    payload_schema_version=3,
    event_type="structural_posterior_calibrated_portfolio_k8_to_k4",
    policy_id=STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=(STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION),
    policy_definition_sha256=(
        STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
)
_OPERATOR_STRATIFIED_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="operator_stratified_calibrated_portfolio_k8_to_k4",
    payload_schema_version=1,
    event_type="operator_stratified_calibrated_portfolio_k8_to_k4",
    policy_id=OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=(OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION),
    policy_definition_sha256=(
        OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
)
_HORIZON_BOUNDED_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="horizon_bounded_calibrated_portfolio_k8_to_k4",
    payload_schema_version=1,
    event_type="horizon_bounded_calibrated_portfolio_k8_to_k4",
    policy_id=HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION,
    policy_definition_sha256=(
        HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
)
_CONSTRAINT_DECOUPLED_HORIZON_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="constraint_decoupled_horizon_portfolio_k8_to_k4",
    payload_schema_version=1,
    event_type="constraint_decoupled_horizon_portfolio_k8_to_k4",
    policy_id=CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=(CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_VERSION),
    policy_definition_sha256=(
        CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
    constraint_decoupled=True,
)
_MINIMUM_INTERVENTION_HORIZON_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="minimum_intervention_horizon_portfolio_k8_to_k4",
    payload_schema_version=1,
    event_type="minimum_intervention_horizon_portfolio_k8_to_k4",
    policy_id=MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_VERSION,
    policy_definition_sha256=(
        MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
    constraint_decoupled=True,
    minimum_intervention_projection=True,
)
_EVIDENCE_CALIBRATED_SOURCE_MIX_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="evidence_calibrated_source_mix_portfolio_k8_to_k4",
    payload_schema_version=1,
    event_type="evidence_calibrated_source_mix_portfolio_k8_to_k4",
    policy_id=EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=(EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_VERSION),
    policy_definition_sha256=(
        EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
    constraint_decoupled=True,
    minimum_intervention_projection=True,
    evidence_calibrated_source_mix=True,
)
_CONTEXTUAL_SEARCH_ALLOCATION_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="contextual_search_allocation_portfolio_k8_to_k4",
    payload_schema_version=1,
    event_type="contextual_search_allocation_portfolio_k8_to_k4",
    policy_id=CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=(CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_VERSION),
    policy_definition_sha256=(
        CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
    constraint_decoupled=True,
    minimum_intervention_projection=True,
    evidence_calibrated_source_mix=True,
    contextual_search_allocation=True,
)
_FRONTIER_PROBE_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="frontier_probe_calibrated_portfolio_k8_to_k4",
    payload_schema_version=4,
    event_type="frontier_probe_calibrated_portfolio_k8_to_k4",
    policy_id=FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION,
    policy_definition_sha256=(
        FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
)
_FULL_SUPPORT_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="full_support_calibrated_portfolio_k8_to_k8",
    payload_schema_version=1,
    event_type="full_support_calibrated_portfolio_k8_to_k8",
    policy_id=FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION,
    policy_definition_sha256=(
        FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
)
_TARGET_CONDITIONED_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="target_conditioned_calibrated_portfolio_k8_to_k4",
    payload_schema_version=1,
    event_type="target_conditioned_calibrated_portfolio_k8_to_k4",
    policy_id=TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION,
    policy_definition_sha256=(
        TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
)
_CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="constraint_decoupled_target_conditioned_portfolio_k8_to_k4",
    payload_schema_version=2,
    event_type="constraint_decoupled_target_conditioned_portfolio_k8_to_k4",
    policy_id=(CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_ID),
    policy_version=(
        CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_VERSION
    ),
    policy_definition_sha256=(
        CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
    constraint_decoupled=True,
)
_ACQUISITION_CERTIFIED_RESIDUAL_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="acquisition_certified_residual_portfolio_k8_to_k4",
    payload_schema_version=1,
    event_type="acquisition_certified_residual_portfolio_k8_to_k4",
    policy_id=ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=(
        ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_VERSION
    ),
    policy_definition_sha256=(
        ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
    constraint_decoupled=True,
    acquisition_certified_residual=True,
)
_REGRET_BOUNDED_INFORMATION_PROFILE = _CalibratedPortfolioSelectionProfile(
    audit_kind="regret_bounded_information_portfolio_k8_to_k4",
    payload_schema_version=1,
    event_type="regret_bounded_information_portfolio_k8_to_k4",
    policy_id=REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_ID,
    policy_version=REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_VERSION,
    policy_definition_sha256=(
        REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
    constraint_decoupled=True,
    acquisition_certified_residual=True,
    regret_bounded_information=True,
)


def calibrated_portfolio_prompt_definition_sha256(
    option_prompt_projection: FiniteOptionPromptProjectionPolicy | None = None,
    *,
    bounded_memory_dose: bool = False,
    proposal_support: bool = False,
    finite_variation_contract: Any | None = None,
    hierarchical_composition_required_proposals: int | None = None,
    feasibility_witness_mode: CalibratedPortfolioFeasibilityWitnessMode = (
        CalibratedPortfolioFeasibilityWitnessMode.CANONICAL
    ),
    constraint_decoupled: bool = False,
) -> str:
    """Choose the exact prompt definition before constructing a scope.

    Composition roots pass the same optional projection policy to this helper
    and :class:`CalibratedCampaignBindingFactory`.  This keeps the choice
    workload-neutral while preventing a projected prompt from claiming the
    byte-compatible legacy-v2 definition.
    """

    if type(bounded_memory_dose) is not bool:
        raise TypeError("bounded_memory_dose must be an exact bool")
    if type(proposal_support) is not bool:
        raise TypeError("proposal_support must be an exact bool")
    if type(constraint_decoupled) is not bool:
        raise TypeError("constraint_decoupled must be an exact bool")
    if proposal_support and feasibility_witness_mode is not (
        CalibratedPortfolioFeasibilityWitnessMode.TASK_KEYED_COMMON_POOL
    ):
        raise ValueError("proposal support requires task-keyed common-pool mode")
    if type(feasibility_witness_mode) is not CalibratedPortfolioFeasibilityWitnessMode:
        raise TypeError("feasibility_witness_mode must be an exact closed witness mode")
    if (
        option_prompt_projection is not None
        and type(option_prompt_projection) is not FiniteOptionPromptProjectionPolicy
    ):
        raise TypeError("option_prompt_projection must be exact projection policy")
    if option_prompt_projection is not None:
        option_prompt_projection.__post_init__()
    base_definition_sha256 = _prompt_definition_sha256_for_shape(
        projected=option_prompt_projection is not None,
        bounded_memory_dose=bounded_memory_dose,
        proposal_support=proposal_support,
        feasibility_witness_mode=feasibility_witness_mode,
    )
    if (
        finite_variation_contract is not None
        and hierarchical_composition_required_proposals is not None
    ):
        shape = _hierarchical_composition_shape(finite_variation_contract)
        observed = None if shape is None else shape.required_composite_proposals
        if observed != hierarchical_composition_required_proposals:
            raise ValueError(
                "explicit hierarchy count differs from the finite contract"
            )
    if finite_variation_contract is not None:
        resolved = _hierarchical_prompt_definition_sha256(
            base_definition_sha256,
            finite_variation_contract,
        )
    elif hierarchical_composition_required_proposals is None:
        resolved = base_definition_sha256
    else:
        resolved = _hierarchical_prompt_definition_for_count(
            base_definition_sha256,
            hierarchical_composition_required_proposals,
        )
    return (
        _constraint_decoupled_prompt_definition_sha256(resolved)
        if constraint_decoupled
        else resolved
    )


def _constraint_decoupled_prompt_definition_sha256(
    strict_prompt_definition_sha256: str,
) -> str:
    require_sha256(
        strict_prompt_definition_sha256,
        "strict_prompt_definition_sha256",
    )
    return hashlib.sha256(
        _CONSTRAINT_DECOUPLED_PROMPT_DOMAIN
        + bytes.fromhex(strict_prompt_definition_sha256)
        + _canonical_json(
            {
                "schema_version": 1,
                "model_authority": [
                    "finite_local_action",
                    "metric_forecast",
                    "semantic_rank",
                    "interaction_rationale",
                    "optional_card_reference",
                ],
                "engine_authority": [
                    "deduplication",
                    "proposal_support",
                    "composition_exposure",
                    "memory_dose",
                    "evaluation_feasibility",
                    "deterministic_refill",
                ],
            }
        )
    ).hexdigest()


def _prompt_definition_sha256_for_shape(
    *,
    projected: bool,
    bounded_memory_dose: bool,
    proposal_support: bool = False,
    feasibility_witness_mode: CalibratedPortfolioFeasibilityWitnessMode,
) -> str:
    """Resolve an already-validated prompt shape, including sealed projections."""

    if (
        feasibility_witness_mode
        is CalibratedPortfolioFeasibilityWitnessMode.TASK_KEYED_COMMON_POOL
    ):
        if proposal_support:
            if bounded_memory_dose:
                return (
                    CALIBRATED_PORTFOLIO_BOUNDED_DOSE_PROPOSAL_SUPPORT_COMMON_POOL_PROMPT_DEFINITION_SHA256
                    if not projected
                    else CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_PROPOSAL_SUPPORT_COMMON_POOL_PROMPT_DEFINITION_SHA256
                )
            return (
                CALIBRATED_PORTFOLIO_PROPOSAL_SUPPORT_COMMON_POOL_PROMPT_DEFINITION_SHA256
                if not projected
                else CALIBRATED_PORTFOLIO_PROJECTED_PROPOSAL_SUPPORT_COMMON_POOL_PROMPT_DEFINITION_SHA256
            )
        if bounded_memory_dose:
            return (
                CALIBRATED_PORTFOLIO_BOUNDED_DOSE_COMMON_POOL_PROMPT_DEFINITION_SHA256
                if not projected
                else CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_COMMON_POOL_PROMPT_DEFINITION_SHA256
            )
        return (
            CALIBRATED_PORTFOLIO_COMMON_POOL_PROMPT_DEFINITION_SHA256
            if not projected
            else CALIBRATED_PORTFOLIO_PROJECTED_COMMON_POOL_PROMPT_DEFINITION_SHA256
        )
    if (
        feasibility_witness_mode
        is CalibratedPortfolioFeasibilityWitnessMode.HIDDEN_CERTIFICATE
    ):
        if bounded_memory_dose:
            return (
                CALIBRATED_PORTFOLIO_BOUNDED_DOSE_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256
                if not projected
                else CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256
            )
        return (
            CALIBRATED_PORTFOLIO_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256
            if not projected
            else CALIBRATED_PORTFOLIO_PROJECTED_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256
        )
    if (
        feasibility_witness_mode
        is CalibratedPortfolioFeasibilityWitnessMode.REQUEST_KEYED
    ):
        if bounded_memory_dose:
            return (
                CALIBRATED_PORTFOLIO_BOUNDED_DOSE_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256
                if not projected
                else CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256
            )
        return (
            CALIBRATED_PORTFOLIO_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256
            if not projected
            else CALIBRATED_PORTFOLIO_PROJECTED_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256
        )
    if bounded_memory_dose:
        return (
            CALIBRATED_PORTFOLIO_BOUNDED_DOSE_PROMPT_DEFINITION_SHA256
            if not projected
            else (CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_PROMPT_DEFINITION_SHA256)
        )
    return (
        CALIBRATED_PORTFOLIO_PROMPT_DEFINITION_SHA256
        if not projected
        else CALIBRATED_PORTFOLIO_PROJECTED_PROMPT_DEFINITION_SHA256
    )


def _prompt_definition_sha256_for_binding(
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
    *,
    constraint_decoupled: bool = False,
) -> str:
    base_definition_sha256 = _prompt_definition_sha256_for_shape(
        projected=binding.option_prompt_projection is not None,
        bounded_memory_dose=request.memory_dose_contract is not None,
        proposal_support=binding.proposal_support is not None,
        feasibility_witness_mode=_feasibility_witness_mode_for_binding(
            request,
            binding,
            constraint_decoupled=constraint_decoupled,
        ),
    )
    resolved = _hierarchical_prompt_definition_sha256(
        base_definition_sha256,
        request.finite_variation_contract,
    )
    return (
        _constraint_decoupled_prompt_definition_sha256(resolved)
        if constraint_decoupled
        else resolved
    )


def _feasibility_witness_mode_for_binding(
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
    *,
    constraint_decoupled: bool = False,
) -> CalibratedPortfolioFeasibilityWitnessMode:
    observed = binding.context.scope.prompt_definition_sha256
    for mode in CalibratedPortfolioFeasibilityWitnessMode:
        base_expected = _prompt_definition_sha256_for_shape(
            projected=binding.option_prompt_projection is not None,
            bounded_memory_dose=request.memory_dose_contract is not None,
            proposal_support=binding.proposal_support is not None,
            feasibility_witness_mode=mode,
        )
        expected = _hierarchical_prompt_definition_sha256(
            base_expected,
            request.finite_variation_contract,
        )
        if constraint_decoupled:
            expected = _constraint_decoupled_prompt_definition_sha256(expected)
        if observed == expected:
            return mode
    raise ValueError("calibration scope names a foreign prompt definition")


def _payload_schema_version(
    profile: _CalibratedPortfolioSelectionProfile,
    binding: CalibratedPortfolioInputBinding,
    request: PortfolioSelectionRequest,
) -> int:
    return (
        profile.payload_schema_version
        + (0 if binding.option_prompt_projection is None else 1)
        + (0 if request.memory_dose_contract is None else 2)
        + (0 if binding.common_candidate_pool is None else 4)
        + (0 if binding.proposal_support is None else 8)
        + (0 if binding.contextual_allocation is None else 16)
        + (
            0
            if profile is _FULL_SUPPORT_PROFILE
            or request.portfolio_size == CALIBRATED_PORTFOLIO_EVALUATION_SIZE
            else 32
        )
    )


def _profile_for_allocator(
    allocator: CalibratedPortfolioAllocator,
) -> _CalibratedPortfolioSelectionProfile:
    if type(allocator) is TraceCalibratedSlatePolicy:
        allocator.__post_init__()
        if allocator.mode is not SlateAllocationMode.CALIBRATED_FOUR_ROLE:
            raise ValueError("v2 adapter requires calibrated four-role allocation")
        return _FOUR_ROLE_PROFILE
    if type(allocator) is ModelAnchoredCalibratedSlatePolicy:
        allocator.__post_init__()
        return _MODEL_ANCHORED_PROFILE
    if type(allocator) is StructuralPosteriorSlatePolicy:
        allocator.__post_init__()
        return _STRUCTURAL_POSTERIOR_PROFILE
    if type(allocator) is OperatorStratifiedStructuralPosteriorSlatePolicy:
        allocator.__post_init__()
        return _OPERATOR_STRATIFIED_PROFILE
    if type(allocator) is HorizonBoundedStructuralPosteriorSlatePolicy:
        allocator.__post_init__()
        return _HORIZON_BOUNDED_PROFILE
    if type(allocator) is FrontierProbeSlatePolicy:
        allocator.__post_init__()
        return _FRONTIER_PROBE_PROFILE
    if type(allocator) is FullSupportSlatePolicy:
        return _FULL_SUPPORT_PROFILE
    if type(allocator) is TargetConditionedSlateAllocatorAdapter:
        allocator.__post_init__()
        return _TARGET_CONDITIONED_PROFILE
    if type(allocator) is AcquisitionCertifiedSlatePolicy:
        raise ValueError(
            "acquisition-certified allocation requires constraint-decoupled authority"
        )
    if type(allocator) is RegretBoundedSlatePolicy:
        raise ValueError(
            "regret-bounded allocation requires constraint-decoupled authority"
        )
    raise TypeError("allocator must be an exact supported calibrated-slate policy")


def _profile_for_allocator_authority(
    allocator: CalibratedPortfolioAllocator,
    *,
    constraint_decoupled: bool,
    minimum_intervention_projection: bool = False,
    evidence_calibrated_source_mix: bool = False,
    contextual_search_allocation: bool = False,
) -> _CalibratedPortfolioSelectionProfile:
    if type(constraint_decoupled) is not bool:
        raise TypeError("constraint_decoupled must be an exact bool")
    if type(minimum_intervention_projection) is not bool:
        raise TypeError("minimum_intervention_projection must be an exact bool")
    if type(evidence_calibrated_source_mix) is not bool:
        raise TypeError("evidence_calibrated_source_mix must be an exact bool")
    if type(contextual_search_allocation) is not bool:
        raise TypeError("contextual_search_allocation must be an exact bool")
    if contextual_search_allocation and not evidence_calibrated_source_mix:
        raise ValueError(
            "contextual search allocation requires source-mix reconciliation"
        )
    if evidence_calibrated_source_mix and not minimum_intervention_projection:
        raise ValueError("evidence-calibrated source mix requires minimum intervention")
    if minimum_intervention_projection and not constraint_decoupled:
        raise ValueError("minimum intervention requires constraint-decoupled authority")
    if not constraint_decoupled:
        return _profile_for_allocator(allocator)
    if type(allocator) is AcquisitionCertifiedSlatePolicy:
        if (
            minimum_intervention_projection
            or evidence_calibrated_source_mix
            or contextual_search_allocation
        ):
            raise ValueError(
                "acquisition certification is one complete allocation authority"
            )
        allocator.__post_init__()
        return _ACQUISITION_CERTIFIED_RESIDUAL_PROFILE
    if type(allocator) is RegretBoundedSlatePolicy:
        if (
            minimum_intervention_projection
            or evidence_calibrated_source_mix
            or contextual_search_allocation
        ):
            raise ValueError(
                "regret-bounded information is one complete allocation authority"
            )
        allocator.__post_init__()
        return _REGRET_BOUNDED_INFORMATION_PROFILE
    if type(allocator) is TargetConditionedSlateAllocatorAdapter:
        if (
            minimum_intervention_projection
            or evidence_calibrated_source_mix
            or contextual_search_allocation
        ):
            raise ValueError(
                "target-conditioned constraint decoupling does not imply the "
                "horizon-only projection or source-mix treatments"
            )
        allocator.__post_init__()
        return _CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PROFILE
    if type(allocator) is not HorizonBoundedStructuralPosteriorSlatePolicy:
        raise TypeError(
            "constraint-decoupled acquisition requires the exact horizon-bounded "
            "or target-conditioned allocator"
        )
    allocator.__post_init__()
    if contextual_search_allocation:
        return _CONTEXTUAL_SEARCH_ALLOCATION_PROFILE
    if evidence_calibrated_source_mix:
        return _EVIDENCE_CALIBRATED_SOURCE_MIX_PROFILE
    if minimum_intervention_projection:
        return _MINIMUM_INTERVENTION_HORIZON_PROFILE
    return _CONSTRAINT_DECOUPLED_HORIZON_PROFILE


def _profile_for_audit_kind(
    audit_kind: str,
) -> _CalibratedPortfolioSelectionProfile:
    if audit_kind == _FOUR_ROLE_PROFILE.audit_kind:
        return _FOUR_ROLE_PROFILE
    if audit_kind == _MODEL_ANCHORED_PROFILE.audit_kind:
        return _MODEL_ANCHORED_PROFILE
    if audit_kind == _STRUCTURAL_POSTERIOR_PROFILE.audit_kind:
        return _STRUCTURAL_POSTERIOR_PROFILE
    if audit_kind == _OPERATOR_STRATIFIED_PROFILE.audit_kind:
        return _OPERATOR_STRATIFIED_PROFILE
    if audit_kind == _HORIZON_BOUNDED_PROFILE.audit_kind:
        return _HORIZON_BOUNDED_PROFILE
    if audit_kind == _CONSTRAINT_DECOUPLED_HORIZON_PROFILE.audit_kind:
        return _CONSTRAINT_DECOUPLED_HORIZON_PROFILE
    if audit_kind == _MINIMUM_INTERVENTION_HORIZON_PROFILE.audit_kind:
        return _MINIMUM_INTERVENTION_HORIZON_PROFILE
    if audit_kind == _EVIDENCE_CALIBRATED_SOURCE_MIX_PROFILE.audit_kind:
        return _EVIDENCE_CALIBRATED_SOURCE_MIX_PROFILE
    if audit_kind == _CONTEXTUAL_SEARCH_ALLOCATION_PROFILE.audit_kind:
        return _CONTEXTUAL_SEARCH_ALLOCATION_PROFILE
    if audit_kind == _FRONTIER_PROBE_PROFILE.audit_kind:
        return _FRONTIER_PROBE_PROFILE
    if audit_kind == _FULL_SUPPORT_PROFILE.audit_kind:
        return _FULL_SUPPORT_PROFILE
    if audit_kind == _TARGET_CONDITIONED_PROFILE.audit_kind:
        return _TARGET_CONDITIONED_PROFILE
    if audit_kind == _CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PROFILE.audit_kind:
        return _CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PROFILE
    if audit_kind == _ACQUISITION_CERTIFIED_RESIDUAL_PROFILE.audit_kind:
        return _ACQUISITION_CERTIFIED_RESIDUAL_PROFILE
    if audit_kind == _REGRET_BOUNDED_INFORMATION_PROFILE.audit_kind:
        return _REGRET_BOUNDED_INFORMATION_PROFILE
    raise ValueError("supplemental audit has a foreign audit kind")


def _profile_for_policy_definition_sha256(
    policy_definition_sha256: str,
) -> _CalibratedPortfolioSelectionProfile:
    profiles = (
        _FOUR_ROLE_PROFILE,
        _MODEL_ANCHORED_PROFILE,
        _STRUCTURAL_POSTERIOR_PROFILE,
        _OPERATOR_STRATIFIED_PROFILE,
        _HORIZON_BOUNDED_PROFILE,
        _CONSTRAINT_DECOUPLED_HORIZON_PROFILE,
        _MINIMUM_INTERVENTION_HORIZON_PROFILE,
        _EVIDENCE_CALIBRATED_SOURCE_MIX_PROFILE,
        _CONTEXTUAL_SEARCH_ALLOCATION_PROFILE,
        _FRONTIER_PROBE_PROFILE,
        _FULL_SUPPORT_PROFILE,
        _TARGET_CONDITIONED_PROFILE,
        _CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PROFILE,
        _ACQUISITION_CERTIFIED_RESIDUAL_PROFILE,
        _REGRET_BOUNDED_INFORMATION_PROFILE,
    )
    matches = tuple(
        value
        for value in profiles
        if value.policy_definition_sha256 == policy_definition_sha256
    )
    if len(matches) != 1:
        raise ValueError("selector definition does not name one exact profile")
    return matches[0]


_STRICT_CONFIG = ConfigDict(
    extra="forbid",
    strict=True,
    populate_by_name=True,
    validate_default=True,
)
_PROPOSAL_DOMAIN = b"agent-evolve:calibrated-portfolio-proposal:v2\x00"
_SEMANTIC_RECONCILIATION_DOMAIN = b"agent-evolve:semantic-slate-reconciliation:v1\x00"
_SEMANTIC_RECONCILIATION_POLICY_ID = "semantic_slate_engine_reconciliation"
_SEMANTIC_RECONCILIATION_POLICY_VERSION = 1
_SEMANTIC_RECONCILIATION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:semantic-slate-engine-reconciliation:v1;"
    b"local-model-semantics=true;cross-member-engine-authority=true;"
    b"deduplicate=true;required-support=true;exact-composition=true;"
    b"feasible-evaluation-witness=true;bounded-memory-dose=true;"
    b"deterministic-refill=true;model-engine-provenance=true;"
    b"objective-values-consulted=false;workload-identifiers-consulted=false"
).hexdigest()
_MINIMUM_INTERVENTION_RECONCILIATION_DOMAIN = (
    b"agent-evolve:semantic-slate-minimum-intervention-reconciliation:v1\x00"
)
_MINIMUM_INTERVENTION_RECONCILIATION_POLICY_ID = (
    "semantic_slate_minimum_intervention_reconciliation"
)
_MINIMUM_INTERVENTION_RECONCILIATION_POLICY_VERSION = 1
_MINIMUM_INTERVENTION_RECONCILIATION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:semantic-slate-minimum-intervention-reconciliation:v1;"
    b"local-model-semantics=true;cross-member-engine-authority=true;"
    b"feasibility-objective=max-retained-model-count-then-model-rank;"
    b"canonical-order=final-tie-break-only;deduplicate=true;"
    b"required-support=true;exact-composition=true;bounded-memory-dose=true;"
    b"deterministic-refill=true;intervention-telemetry=true;"
    b"model-engine-provenance=true;objective-values-consulted=false;"
    b"workload-identifiers-consulted=false"
).hexdigest()
_SOURCE_MIX_RECONCILIATION_DOMAIN = (
    b"agent-evolve:semantic-slate-source-mix-reconciliation:v1\x00"
)
_SOURCE_MIX_RECONCILIATION_POLICY_ID = (
    "semantic_slate_evidence_calibrated_source_mix_reconciliation"
)
_SOURCE_MIX_RECONCILIATION_POLICY_VERSION = 1
_SOURCE_MIX_RECONCILIATION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:semantic-slate-source-mix-reconciliation:v1;"
    b"local-model-semantics=true;cross-member-engine-authority=true;"
    b"protected-source=task-keyed-common-pool-feasibility-witness;"
    b"protected-source-count=one;protected-source-phase=wave-one;"
    b"required-evaluation-membership=true;"
    b"remaining-objective=max-retained-model-count-then-model-rank;"
    b"proposal-support-conflicts=deterministically-deferred;"
    b"hard-feasibility-composition-memory=true;"
    b"source-and-intervention-telemetry=true;"
    b"objective-values-consulted=false;workload-identifiers-consulted=false"
).hexdigest()
_CONTEXTUAL_SEARCH_RECONCILIATION_DOMAIN = (
    b"agent-evolve:semantic-slate-contextual-search-reconciliation:v4\x00"
)
_CONTEXTUAL_SEARCH_RECONCILIATION_POLICY_ID = (
    "semantic_slate_contextual_search_allocation_reconciliation"
)
_CONTEXTUAL_SEARCH_RECONCILIATION_POLICY_VERSION = 4
_CONTEXTUAL_SEARCH_RECONCILIATION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:semantic-slate-contextual-search-reconciliation:v4;"
    b"allocation=prospective-prior-only-request-slice;"
    b"source-marginals=sealed-finite-variation-source;"
    b"semantic-model-vs-engine-origin=separate-provenance;"
    b"operator-marginals=atomic-vs-composite;"
    b"requested-allocation=exact-when-structurally-feasible;"
    b"recourse=minimum-l1-feasible-marginal-projection;"
    b"recourse-tie-break=source-deviation,operator-deviation,model,atomic;"
    b"requested-and-realized-allocation-authenticated=true;"
    b"required-evaluation-membership=realized-exact;"
    b"remaining-objective=max-retained-model-count-then-model-rank;"
    b"proposal-support-conflicts=deterministically-deferred;"
    b"hard-feasibility-composition-memory=true;"
    b"post-recourse-composition=nearest-exact-k-capacity-projection;"
    b"objective-values-consulted=false;workload-model-provider-identifiers=false"
).hexdigest()
_CONTEXTUAL_ALLOCATION_PROJECTION_DOMAIN = (
    b"agent-evolve:contextual-allocation-feasibility-projection:v2\x00"
)
_CONTEXTUAL_ALLOCATION_PROJECTION_POLICY_ID = (
    "minimum_l1_contextual_allocation_feasibility_projection"
)
_CONTEXTUAL_ALLOCATION_PROJECTION_POLICY_VERSION = 2
_CONTEXTUAL_ALLOCATION_PROJECTION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:contextual-allocation-feasibility-projection:v2;"
    b"input=prospective-requested-source-operator-marginals;"
    b"source=sealed-finite-variation-source;"
    b"semantic-model-vs-engine-origin=separate-provenance;"
    b"constraints=finite-contract,model-slate,portfolio-structure,memory-dose;"
    b"objective=min-total-l1-then-source-l1-then-operator-l1;"
    b"tie-break=max-retained-model-then-max-atomic-then-canonical-witness;"
    b"objective-values-consulted=false;workload-model-provider-identifiers=false"
).hexdigest()
_CONTEXTUAL_ALLOCATION_JOINT_DOSE_PROJECTION_DOMAIN = (
    b"agent-evolve:contextual-allocation-feasibility-projection:v3\x00"
)
_CONTEXTUAL_ALLOCATION_JOINT_DOSE_PROJECTION_POLICY_VERSION = 3
_CONTEXTUAL_ALLOCATION_JOINT_DOSE_PROJECTION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:contextual-allocation-feasibility-projection:v3;"
    b"input=prospective-requested-source-operator-marginals;"
    b"source=sealed-finite-variation-source;"
    b"semantic-model-vs-engine-origin=separate-provenance;"
    b"constraints=finite-contract,model-slate,portfolio-structure,"
    b"bounded-memory-dose-attribution;"
    b"objective=min-total-l1-then-source-l1-then-operator-l1;"
    b"tie-break=max-retained-model-then-max-atomic-then-canonical-witness;"
    b"memory-dose-witness=exact-evaluated-subset-card-assignment;"
    b"objective-values-consulted=false;"
    b"workload-model-provider-identifiers=false"
).hexdigest()
_CONTEXTUAL_ALLOCATION_ASSAY_PROJECTION_DOMAIN = (
    b"agent-evolve:contextual-allocation-feasibility-projection:v4\x00"
)
_CONTEXTUAL_ALLOCATION_ASSAY_PROJECTION_POLICY_VERSION = 4
_CONTEXTUAL_ALLOCATION_ASSAY_PROJECTION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:contextual-allocation-feasibility-projection:v4;"
    b"input=prospective-requested-source-operator-marginals;"
    b"source=sealed-finite-variation-source;"
    b"proposal-operator-and-intervention-arity=independent-axes;"
    b"single-path-intervention=exact-parent-relative-json-patch-cardinality-one;"
    b"constraints=finite-contract,model-slate,portfolio-structure,"
    b"bounded-memory-dose-attribution,minimum-single-path-interventions;"
    b"objective=min-total-l1-then-source-l1-then-operator-l1;"
    b"objective-values-consulted=false;workload-model-provider-identifiers=false"
).hexdigest()
_CONTEXTUAL_ALLOCATION_OFFSPRING_OPPORTUNITY_PROJECTION_DOMAIN = (
    b"agent-evolve:contextual-allocation-feasibility-projection:v5\x00"
)
_CONTEXTUAL_ALLOCATION_OFFSPRING_OPPORTUNITY_PROJECTION_POLICY_VERSION = 5
_CONTEXTUAL_ALLOCATION_OFFSPRING_OPPORTUNITY_PROJECTION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:contextual-allocation-feasibility-projection:v5;"
    b"input=prospective-requested-source-operator-marginals;"
    b"source=sealed-finite-variation-source;"
    b"proposal-operator,intervention-arity,offspring-opportunity=independent-axes;"
    b"offspring-opportunity=pairwise-disjoint-parent-relative-patch-pair;"
    b"constraints=finite-contract,model-slate,portfolio-structure,"
    b"bounded-memory-dose-attribution,minimum-single-path-interventions,"
    b"minimum-disjoint-parent-patch-pairs;"
    b"objective=min-total-l1-then-source-l1-then-operator-l1;"
    b"objective-values-consulted=false;workload-model-provider-identifiers=false"
).hexdigest()
_HIERARCHICAL_PROMPT_DOMAIN = (
    b"agent-evolve:calibrated-hierarchical-composition-prompt:v1\x00"
)
_Rationale = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=16_384,
    ),
]
_Direction = Literal["decrease", "increase", "unchanged", "unknown"]
_Confidence = Literal["low", "medium", "high", "unknown"]
_RoleProposal = Literal["exploit", "falsify", "coverage"]


@dataclass(frozen=True, slots=True)
class _HierarchicalCompositionShape:
    """Authenticated ranked-union schema projected by a finite catalog."""

    required_composite_proposals: int
    atomic_option_ids: tuple[str, ...]
    component_option_ids: tuple[str, ...]
    composite_components: tuple[tuple[str, tuple[str, str]], ...]

    def __post_init__(self) -> None:
        if type(
            self.required_composite_proposals
        ) is not int or not 1 <= self.required_composite_proposals < (
            CALIBRATED_PORTFOLIO_PROPOSAL_SIZE
        ):
            raise ValueError("required composite proposal count is invalid")
        if (
            type(self.atomic_option_ids) is not tuple
            or not self.atomic_option_ids
            or self.atomic_option_ids != tuple(sorted(set(self.atomic_option_ids)))
        ):
            raise ValueError("atomic option IDs must be non-empty and canonical")
        if (
            type(self.component_option_ids) is not tuple
            or not self.component_option_ids
            or self.component_option_ids
            != tuple(sorted(set(self.component_option_ids)))
        ):
            raise ValueError("component option IDs must be non-empty and canonical")
        if (
            type(self.composite_components) is not tuple
            or len(self.composite_components) < self.required_composite_proposals
            or self.composite_components
            != tuple(sorted(set(self.composite_components)))
        ):
            raise ValueError("composite component bindings must be canonical")
        composite_ids: set[str] = set()
        for composite_id, components in self.composite_components:
            if type(composite_id) is not str or not composite_id:
                raise ValueError("composite option ID must be non-empty")
            if composite_id in composite_ids:
                raise ValueError("composite option IDs cannot repeat")
            composite_ids.add(composite_id)
            if (
                type(components) is not tuple
                or len(components) != 2
                or components != tuple(sorted(set(components)))
                or not set(components).issubset(self.component_option_ids)
            ):
                raise ValueError(
                    "composite components must name two authenticated source atoms"
                )

    @property
    def composite_option_ids(self) -> tuple[str, ...]:
        return tuple(value[0] for value in self.composite_components)

    @property
    def components_by_composite(self) -> dict[str, tuple[str, str]]:
        return dict(self.composite_components)

    def to_prompt_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "selection_exposure": (
                CompositionSelectionExposure.HIERARCHICAL_RANKED_UNION.value
            ),
            "ranked_union_member_kinds": ["atomic", "compose_r2"],
            "required_composite_proposals": self.required_composite_proposals,
            "model_may_only_select_engine_materialized_options": True,
            "model_may_not_edit_or_materialize_configurations": True,
            "composite_options": [
                {
                    "composite_option_id": composite_id,
                    "component_option_ids": list(components),
                }
                for composite_id, components in self.composite_components
            ],
        }


def _hierarchical_composition_shape(
    contract: Any,
    *,
    allowed_option_ids: tuple[str, ...] | None = None,
) -> _HierarchicalCompositionShape | None:
    """Decode a catalog-authenticated hierarchy without workload knowledge."""

    options = tuple(contract.options)
    option_ids = {value.option_id for value in options}
    allowed = option_ids if allowed_option_ids is None else set(allowed_option_ids)
    if not allowed.issubset(option_ids):
        raise ValueError("hierarchical allowed option IDs escape the finite contract")
    marked: list[tuple[str, tuple[str, str], int]] = []
    component_identities: dict[str, str] = {}
    identity_index = validated_finite_variation_identity_index(contract)
    identity_by_option_id = dict(
        zip(
            identity_index.option_ids,
            identity_index.option_identity_sha256s,
            strict=True,
        )
    )
    for option in options:
        metadata = dict(option.metadata)
        exposure = metadata.get(COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY)
        if exposure is None:
            continue
        if exposure != CompositionSelectionExposure.HIERARCHICAL_RANKED_UNION.value:
            raise ValueError("finite option declares an unknown composition exposure")
        try:
            required = int(metadata[COMPOSITION_REQUIRED_PROPOSALS_METADATA_KEY])
            left = metadata[COMPOSITION_LEFT_OPTION_METADATA_KEY]
            right = metadata[COMPOSITION_RIGHT_OPTION_METADATA_KEY]
            left_identity = metadata["left_option_identity_sha256"]
            right_identity = metadata["right_option_identity_sha256"]
        except (KeyError, ValueError) as error:
            raise ValueError("hierarchical composite metadata is incomplete") from error
        components = tuple(sorted((left, right)))
        if (
            any(type(value) is not str or not value for value in components)
            or len(set(components)) != 2
        ):
            raise ValueError("hierarchical composite names invalid source options")
        identities = {left: left_identity, right: right_identity}
        for component_id, component_identity in identities.items():
            require_sha256(component_identity, "component_option_identity_sha256")
            previous = component_identities.setdefault(
                component_id,
                component_identity,
            )
            if previous != component_identity:
                raise ValueError(
                    "hierarchical component identity differs across composites"
                )
            retained_identity = identity_by_option_id.get(component_id)
            if (
                retained_identity is not None
                and retained_identity != component_identity
            ):
                raise ValueError(
                    "hierarchical component identity differs from retained atom"
                )
        if option.option_id in allowed:
            marked.append((option.option_id, components, required))
    if not any(
        COMPOSITION_SELECTION_EXPOSURE_METADATA_KEY in dict(value.metadata)
        for value in options
    ):
        return None
    if not marked:
        raise ValueError("candidate universe contains no hierarchical composites")
    required_counts = {value[2] for value in marked}
    if len(required_counts) != 1:
        raise ValueError("hierarchical composites disagree on proposal count")
    required = next(iter(required_counts))
    composite_ids = {value[0] for value in marked}
    component_ids = {
        component_id for _, components, _ in marked for component_id in components
    }
    if component_ids & composite_ids:
        raise ValueError("hierarchical composite recursively names a composite")
    atomic_ids = tuple(sorted(allowed - composite_ids))
    return _HierarchicalCompositionShape(
        required_composite_proposals=required,
        atomic_option_ids=atomic_ids,
        component_option_ids=tuple(sorted(component_ids)),
        composite_components=tuple(
            sorted((composite_id, components) for composite_id, components, _ in marked)
        ),
    )


def _hierarchical_prompt_definition_sha256(
    base_definition_sha256: str,
    contract: Any,
) -> str:
    shape = _hierarchical_composition_shape(contract)
    if shape is None:
        return base_definition_sha256
    return _hierarchical_prompt_definition_for_count(
        base_definition_sha256,
        shape.required_composite_proposals,
    )


def _hierarchical_prompt_definition_for_count(
    base_definition_sha256: str,
    required_composite_proposals: int,
) -> str:
    if (
        type(required_composite_proposals) is not int
        or not 1 <= required_composite_proposals < CALIBRATED_PORTFOLIO_PROPOSAL_SIZE
    ):
        raise ValueError("hierarchical composition count must lie in [1, 8)")
    configuration = {
        "schema_version": 1,
        "base_prompt_definition_sha256": base_definition_sha256,
        "selection_exposure": (
            CompositionSelectionExposure.HIERARCHICAL_RANKED_UNION.value
        ),
        "required_composite_proposals": required_composite_proposals,
        "ranked_union_member_kinds": ["atomic", "compose_r2"],
        "exact_engine_materialization_only": True,
    }
    return hashlib.sha256(
        _HIERARCHICAL_PROMPT_DOMAIN + _canonical_json(configuration)
    ).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _proposal_sha256(record: dict[str, object]) -> str:
    return hashlib.sha256(_PROPOSAL_DOMAIN + _canonical_json(record)).hexdigest()


class _CalibratedPredictionOutputBase(BaseModel):
    model_config = _STRICT_CONFIG


def _validate_common_output_member(
    member: BaseModel,
    *,
    allowed_card_keys: frozenset[str],
    required_metric_ids: tuple[str, ...],
    require_supporting_cards: bool,
) -> None:
    card_keys = tuple(cast(Any, member).supporting_card_keys)
    if len(set(card_keys)) != len(card_keys):
        raise ValueError("supporting_card_keys cannot contain duplicates")
    if not set(card_keys).issubset(allowed_card_keys):
        raise ValueError("supporting_card_keys escape the request snapshot")
    if require_supporting_cards and not card_keys:
        raise ValueError("this request requires supporting-card attribution")
    predictions = tuple(cast(Any, member).effect_predictions)
    metric_ids = tuple(value.metric_id for value in predictions)
    if len(set(metric_ids)) != len(metric_ids):
        raise ValueError("effect_predictions cannot repeat a metric")
    if set(metric_ids) != set(required_metric_ids):
        raise ValueError("effect_predictions must cover the exact metrics")


class _CalibratedMemberOutputBase(BaseModel):
    model_config = _STRICT_CONFIG

    allowed_option_ids: ClassVar[frozenset[str]] = frozenset()
    allowed_card_keys: ClassVar[frozenset[str]] = frozenset()
    required_metric_ids: ClassVar[tuple[str, ...]] = ()
    require_supporting_cards: ClassVar[bool] = True

    @model_validator(mode="after")
    def _validate_member(self) -> "_CalibratedMemberOutputBase":
        option_id = cast(Any, self).option_id
        if option_id not in type(self).allowed_option_ids:
            raise PydanticCustomError(
                ValidationIssueReasonCode.FINITE_OPTION_OUT_OF_CONTRACT.value,
                "option_id escapes the request's sealed finite options",
            )
        _validate_common_output_member(
            self,
            allowed_card_keys=type(self).allowed_card_keys,
            required_metric_ids=type(self).required_metric_ids,
            require_supporting_cards=type(self).require_supporting_cards,
        )
        return self


class _CalibratedCompositeMemberOutputBase(BaseModel):
    model_config = _STRICT_CONFIG

    allowed_composite_option_ids: ClassVar[frozenset[str]] = frozenset()
    components_by_composite: ClassVar[dict[str, tuple[str, str]]] = {}
    allowed_card_keys: ClassVar[frozenset[str]] = frozenset()
    required_metric_ids: ClassVar[tuple[str, ...]] = ()
    require_supporting_cards: ClassVar[bool] = True

    @model_validator(mode="after")
    def _validate_member(self) -> "_CalibratedCompositeMemberOutputBase":
        composite_id = cast(Any, self).composite_option_id
        if composite_id not in type(self).allowed_composite_option_ids:
            raise PydanticCustomError(
                ValidationIssueReasonCode.FINITE_OPTION_OUT_OF_CONTRACT.value,
                "composite_option_id escapes the authenticated composition set",
            )
        components = tuple(cast(Any, self).component_option_ids)
        if components != type(self).components_by_composite[composite_id]:
            raise ValueError(
                "component_option_ids differ from the engine-materialized composite"
            )
        _validate_common_output_member(
            self,
            allowed_card_keys=type(self).allowed_card_keys,
            required_metric_ids=type(self).required_metric_ids,
            require_supporting_cards=type(self).require_supporting_cards,
        )
        return self


def _resolved_output_option_id(member: Any) -> str:
    kind = getattr(member, "action_kind", "atomic")
    if kind == "atomic":
        return cast(str, member.option_id)
    if kind == "compose_r2":
        return cast(str, member.composite_option_id)
    raise ValueError("output member declares an unknown action kind")


class _CalibratedSlateOutputBase(BaseModel):
    model_config = _STRICT_CONFIG

    assigned_card_keys: ClassVar[frozenset[str]] = frozenset()
    require_pairwise_disjoint_parent_patches: ClassVar[bool] = False
    finite_variation_contract: ClassVar[Any] = None
    evaluation_portfolio_size: ClassVar[int] = CALIBRATED_PORTFOLIO_EVALUATION_SIZE
    min_distinct_families: ClassVar[int | None] = None
    memory_dose_contract: ClassVar[BoundedPortfolioMemoryDoseContract | None] = None
    allowed_common_pool_option_ids: ClassVar[frozenset[str] | None] = None
    ordered_common_pool_option_ids: ClassVar[tuple[str, ...] | None] = None
    required_proposal_support_option_ids: ClassVar[frozenset[str]] = frozenset()
    required_composite_proposals: ClassVar[int] = 0
    required_evaluation_family_bounds: ClassVar[tuple[tuple[str, int, int], ...]] = ()
    enforce_cross_member_constraints: ClassVar[bool] = True

    @model_validator(mode="after")
    def _validate_slate(self) -> "_CalibratedSlateOutputBase":
        members = tuple(cast(Any, self).members)
        if len(members) != CALIBRATED_PORTFOLIO_PROPOSAL_SIZE:
            raise ValueError("members must contain exactly eight proposals")
        if not type(self).enforce_cross_member_constraints:
            return self
        option_ids = tuple(_resolved_output_option_id(value) for value in members)
        if len(set(option_ids)) != len(option_ids):
            raise PydanticCustomError(
                ValidationIssueReasonCode.DUPLICATE_FINITE_OPTIONS.value,
                "members cannot repeat a finite option",
            )
        composite_count = sum(
            getattr(value, "action_kind", "atomic") == "compose_r2" for value in members
        )
        if composite_count != type(self).required_composite_proposals:
            raise ValueError(
                "members differ from the required hierarchical composition count"
            )
        common_pool = type(self).allowed_common_pool_option_ids
        if common_pool is not None and not set(option_ids).issubset(common_pool):
            raise ValueError("members must come from the task-keyed candidate universe")
        if not type(self).required_proposal_support_option_ids.issubset(option_ids):
            raise PydanticCustomError(
                ValidationIssueReasonCode.PROPOSAL_SUPPORT_OPTION_OMITTED.value,
                "members omit an engine-reserved proposal-support option",
            )
        administered = {
            card for member in members for card in member.supporting_card_keys
        }
        if not type(self).assigned_card_keys.issubset(administered):
            raise PydanticCustomError(
                ValidationIssueReasonCode.ASSIGNED_MEMORY_CARD_OMITTED.value,
                "the proposed slate omits an assigned memory card",
            )
        dose = type(self).memory_dose_contract
        if dose is not None:
            dose_assessment = assess_proposed_portfolio_memory_dose(
                dose,
                tuple(
                    PortfolioMemoryDoseMember(
                        rank=rank,
                        option_id=_resolved_output_option_id(member),
                        option_identity_sha256=(
                            type(self)
                            .finite_variation_contract.resolve(
                                _resolved_output_option_id(member)
                            )
                            .identity_sha256
                        ),
                        supporting_card_keys=tuple(sorted(member.supporting_card_keys)),
                    )
                    for rank, member in enumerate(members, start=1)
                ),
            )
            if not dose_assessment.passed:
                raise PydanticCustomError(
                    ValidationIssueReasonCode.PORTFOLIO_MEMORY_DOSE_VIOLATION.value,
                    "the proposed slate violates its bounded memory-dose contract",
                )
        if type(self).require_pairwise_disjoint_parent_patches and not (
            finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
                type(self).finite_variation_contract,
                option_ids,
                portfolio_size=type(self).evaluation_portfolio_size,
                min_distinct_families=type(self).min_distinct_families,
                family_exposure_bounds=(type(self).required_evaluation_family_bounds),
            )
        ):
            raise PydanticCustomError(
                ValidationIssueReasonCode.NO_FEASIBLE_DISJOINT_PORTFOLIO.value,
                "proposed slate contains no pairwise-disjoint allocation of "
                f"size {type(self).evaluation_portfolio_size}",
            )
        return self


def _calibrated_output_type(
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
    *,
    required_evaluation_family_bounds: tuple[tuple[str, int, int], ...] = (),
    constraint_decoupled: bool = False,
) -> type[BaseModel]:
    if type(constraint_decoupled) is not bool:
        raise TypeError("constraint_decoupled must be an exact bool")
    context = binding.context
    option_ids = (
        tuple(option.option_id for option in request.finite_variation_contract.options)
        if binding.common_candidate_pool is None
        else binding.common_candidate_pool.option_ids
    )
    hierarchy = _hierarchical_composition_shape(
        request.finite_variation_contract,
        allowed_option_ids=option_ids,
    )
    card_keys = tuple(card.card_key for card in request.cards)
    metric_ids = request.required_metric_ids
    option_enum_utf8_bytes = len(_canonical_json(list(option_ids)))
    option_wire_type: object
    if option_enum_utf8_bytes <= MAX_INLINE_OPTION_ENUM_UTF8_BYTES:
        option_wire_type = Literal.__getitem__(option_ids)
    else:
        option_wire_type = Annotated[
            str,
            StringConstraints(
                strict=True,
                min_length=1,
                max_length=max(len(value) for value in option_ids),
            ),
        ]
    card_literal = Literal.__getitem__(card_keys)
    metric_literal = Literal.__getitem__(metric_ids)

    prediction_type = create_model(
        "CalibratedPortfolioMetricPrediction",
        __base__=_CalibratedPredictionOutputBase,
        __module__=__name__,
        metric_id=(metric_literal, ...),
        direction=(_Direction, ...),
        confidence=(_Confidence, ...),
    )
    card_field = (
        list[card_literal],
        Field(
            description=(
                "Exact supplied card keys that materially support this "
                "proposal. Across the complete slate, every prospectively "
                "assigned card must be cited while all supplied compatibility "
                "and bounded-dose constraints remain true."
                if request.memory_dose_contract is not None
                else "Cards that materially support this proposal."
            ),
            max_length=len(card_keys),
        ),
    )
    prediction_field = (
        list[prediction_type],
        Field(
            description="One categorical direction and confidence for every metric.",
            min_length=len(metric_ids),
            max_length=len(metric_ids),
        ),
    )
    rationale_field = (
        _Rationale,
        Field(description="Concise reason for proposing this sealed action."),
    )
    if hierarchy is None:
        member_type = create_model(
            "CalibratedPortfolioSlateMember",
            __base__=_CalibratedMemberOutputBase,
            __module__=__name__,
            option_id=(
                option_wire_type,
                Field(
                    description=(
                        "One exact option_id from the request's sealed "
                        "ordered_options list."
                    )
                ),
            ),
            supporting_card_keys=card_field,
            effect_predictions=prediction_field,
            role_proposal=(_RoleProposal, ...),
            design_rationale=rationale_field,
        )
        member_type.allowed_option_ids = frozenset(option_ids)
        member_type.allowed_card_keys = frozenset(card_keys)
        member_type.required_metric_ids = metric_ids
        member_type.require_supporting_cards = request.require_supporting_cards
        member_wire_type: object = member_type
    else:
        atomic_literal = Literal.__getitem__(hierarchy.atomic_option_ids)
        component_literal = Literal.__getitem__(hierarchy.component_option_ids)
        composite_literal = Literal.__getitem__(hierarchy.composite_option_ids)
        atomic_member_type = create_model(
            "HierarchicalCalibratedAtomicMember",
            __base__=_CalibratedMemberOutputBase,
            __module__=__name__,
            action_kind=(Literal["atomic"], ...),
            option_id=(
                atomic_literal,
                Field(description="One sealed atomic option ID."),
            ),
            supporting_card_keys=card_field,
            effect_predictions=prediction_field,
            role_proposal=(_RoleProposal, ...),
            design_rationale=rationale_field,
        )
        atomic_member_type.allowed_option_ids = frozenset(hierarchy.atomic_option_ids)
        atomic_member_type.allowed_card_keys = frozenset(card_keys)
        atomic_member_type.required_metric_ids = metric_ids
        atomic_member_type.require_supporting_cards = request.require_supporting_cards
        composite_member_type = create_model(
            "HierarchicalCalibratedCompositeMember",
            __base__=_CalibratedCompositeMemberOutputBase,
            __module__=__name__,
            action_kind=(Literal["compose_r2"], ...),
            composite_option_id=(
                composite_literal,
                Field(
                    description=(
                        "One engine-materialized radius-two composite option ID."
                    )
                ),
            ),
            component_option_ids=(
                list[component_literal],
                Field(
                    description=(
                        "The exact two atomic source IDs bound to the composite."
                    ),
                    min_length=2,
                    max_length=2,
                ),
            ),
            supporting_card_keys=card_field,
            effect_predictions=prediction_field,
            role_proposal=(_RoleProposal, ...),
            design_rationale=rationale_field,
        )
        composite_member_type.allowed_composite_option_ids = frozenset(
            hierarchy.composite_option_ids
        )
        composite_member_type.components_by_composite = (
            hierarchy.components_by_composite
        )
        composite_member_type.allowed_card_keys = frozenset(card_keys)
        composite_member_type.required_metric_ids = metric_ids
        composite_member_type.require_supporting_cards = (
            request.require_supporting_cards
        )
        member_wire_type = Annotated[
            Union[atomic_member_type, composite_member_type],
            Field(discriminator="action_kind"),
        ]

    members_description = (
        "Eight legal semantic suggestions in model-preferred order. Trusted "
        "code reconciles duplicates and all cross-member constraints."
        if constraint_decoupled
        else "Eight distinct proposals in model-preferred order."
    )
    if hierarchy is not None:
        members_description += (
            (
                " Use the discriminated ranked union. Composite component "
                "bindings remain exact; trusted code owns the final atomic/"
                "composite exposure count."
            )
            if constraint_decoupled
            else (
                " Use the discriminated ranked union and include exactly "
                f"{hierarchy.required_composite_proposals} compose_r2 members; "
                "each must repeat the authenticated two-component binding "
                "supplied by the engine."
            )
        )
    if request.require_pairwise_disjoint_parent_patches and not constraint_decoupled:
        family_requirement = (
            ""
            if request.min_distinct_families is None
            else (
                " spanning at least "
                f"{request.min_distinct_families} distinct option families"
            )
        )
        members_description += (
            " The eight proposals must contain at least one subset of exactly "
            f"{request.portfolio_size} options with pairwise-disjoint "
            f"parent-relative changed paths{family_requirement}."
        )
        if required_evaluation_family_bounds:
            rendered_bounds = ", ".join(
                f"{family}=[{minimum},{maximum}]"
                for family, minimum, maximum in required_evaluation_family_bounds
            )
            members_description += (
                " That feasible evaluation subset must also satisfy these "
                f"action-family exposure bounds: {rendered_bounds}."
            )
    if request.memory_dose_contract is not None and not constraint_decoupled:
        lower, upper = request.memory_dose_contract.proposed_supported_member_bounds
        members_description += (
            " Across the complete slate, cite every assigned memory card on "
            f"between {lower} and {upper} total members, using only each card's "
            "declared compatible option IDs."
        )
    output_type = create_model(
        "CalibratedPortfolioSlateProposal",
        __base__=_CalibratedSlateOutputBase,
        __module__=__name__,
        members=(
            list[member_wire_type],
            Field(
                description=members_description,
                min_length=CALIBRATED_PORTFOLIO_PROPOSAL_SIZE,
                max_length=CALIBRATED_PORTFOLIO_PROPOSAL_SIZE,
            ),
        ),
    )
    output_type.assigned_card_keys = frozenset(context.assigned_card_keys)
    output_type.require_pairwise_disjoint_parent_patches = (
        request.require_pairwise_disjoint_parent_patches
    )
    output_type.finite_variation_contract = request.finite_variation_contract
    output_type.evaluation_portfolio_size = request.portfolio_size
    output_type.min_distinct_families = request.min_distinct_families
    output_type.memory_dose_contract = request.memory_dose_contract
    output_type.allowed_common_pool_option_ids = (
        None
        if binding.common_candidate_pool is None
        else frozenset(binding.common_candidate_pool.option_ids)
    )
    output_type.ordered_common_pool_option_ids = (
        None
        if binding.common_candidate_pool is None
        else binding.common_candidate_pool.option_ids
    )
    output_type.required_proposal_support_option_ids = frozenset(
        ()
        if binding.proposal_support is None
        else binding.proposal_support.required_option_ids
    )
    output_type.required_composite_proposals = (
        0 if hierarchy is None else hierarchy.required_composite_proposals
    )
    output_type.required_evaluation_family_bounds = required_evaluation_family_bounds
    output_type.enforce_cross_member_constraints = not constraint_decoupled
    if constraint_decoupled:
        member_types = (
            (member_type,)
            if hierarchy is None
            else (atomic_member_type, composite_member_type)
        )
        for local_member_type in member_types:
            local_member_type.require_supporting_cards = False
    return output_type


def _bounded_option_repair_literal_sets(
    option_ids: tuple[str, ...],
    hierarchy: _HierarchicalCompositionShape | None = None,
) -> tuple[StructuredOutputRepairLiteralSet, ...]:
    """Restate a complete finite ID contract on repair when it fits the port bound."""

    try:
        literal_sets = (
            (
                StructuredOutputRepairLiteralSet(
                    field_path=("members", "*", "option_id"),
                    allowed_literals=option_ids,
                ),
            )
            if hierarchy is None
            else (
                StructuredOutputRepairLiteralSet(
                    field_path=("members", "*", "option_id"),
                    allowed_literals=hierarchy.atomic_option_ids,
                ),
                StructuredOutputRepairLiteralSet(
                    field_path=("members", "*", "composite_option_id"),
                    allowed_literals=hierarchy.composite_option_ids,
                ),
                StructuredOutputRepairLiteralSet(
                    field_path=("members", "*", "component_option_ids", "*"),
                    allowed_literals=hierarchy.component_option_ids,
                ),
            )
        )
    except ValueError:
        return ()
    # Validate the aggregate request boundary here so over-large universes
    # retain exact local validation without receiving a misleading partial list.
    if len(_canonical_json([value.to_record() for value in literal_sets])) > (
        MAX_STRUCTURED_REPAIR_CONTEXT_UTF8_BYTES
    ):
        return ()
    return literal_sets


def _validate_binding_for_request(
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
    *,
    selector_policy_definition_sha256: str = (
        CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    ),
    constraint_decoupled: bool = False,
    contextual_search_allocation: bool = False,
) -> None:
    if type(binding) is not CalibratedPortfolioInputBinding:
        raise TypeError("binding provider must return exact calibrated binding")
    binding.require_request(request)
    if (binding.contextual_allocation is not None) != contextual_search_allocation:
        raise ValueError("contextual allocation binding and selector profile disagree")
    context = binding.context
    full_support = (
        selector_policy_definition_sha256
        == FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )
    if full_support and (
        request.portfolio_size != CALIBRATED_PORTFOLIO_PROPOSAL_SIZE
    ):
        raise ValueError("full-support evaluation must cover all eight proposals")
    if not full_support and not (
        1 <= request.portfolio_size <= CALIBRATED_PORTFOLIO_PROPOSAL_SIZE
    ):
        raise ValueError("evaluation width must lie inside the proposed slate")
    if (
        len(request.finite_variation_contract.options)
        < CALIBRATED_PORTFOLIO_PROPOSAL_SIZE
    ):
        raise ValueError("calibrated v2 requires at least eight finite options")
    if (
        context.scope.selector_policy_definition_sha256
        != selector_policy_definition_sha256
    ):
        raise ValueError("calibration scope names a foreign selector policy")
    expected_prompt_definition = _prompt_definition_sha256_for_binding(
        request,
        binding,
        constraint_decoupled=constraint_decoupled,
    )
    if context.scope.prompt_definition_sha256 != expected_prompt_definition:
        raise ValueError("calibration scope names a foreign prompt definition")
    common_pool_mode = (
        _feasibility_witness_mode_for_binding(
            request,
            binding,
            constraint_decoupled=constraint_decoupled,
        )
        is CalibratedPortfolioFeasibilityWitnessMode.TASK_KEYED_COMMON_POOL
    )
    if common_pool_mode != (binding.common_candidate_pool is not None):
        raise ValueError(
            "task-keyed common-pool prompt mode and input binding disagree"
        )
    if binding.common_candidate_pool is not None and (
        binding.common_candidate_pool.model_selection_size
        != CALIBRATED_PORTFOLIO_PROPOSAL_SIZE
    ):
        raise ValueError(
            "task-keyed candidate universe must request exactly eight members"
        )
    if request.memory_dose_contract is not None and (
        request.memory_dose_contract.assigned_card_keys != context.assigned_card_keys
    ):
        raise ValueError(
            "bounded memory-dose cards differ from the calibrated assignment"
        )


def render_calibrated_portfolio_selection_prompt(
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
) -> str:
    """Render the canonical treatment-neutral sealed K=8 proposal contract.

    ``PortfolioSelectionRequest.instruction`` remains identity-bound in the
    request receipt but is deliberately never rendered.  This opt-in adapter
    therefore cannot inherit treatment behavior from a benchmark caller's
    free-form selector instruction.
    """

    if type(request) is not PortfolioSelectionRequest:
        raise TypeError("request must be exact PortfolioSelectionRequest")
    request.__post_init__()
    return _render_calibrated_portfolio_selection_prompt(
        request,
        binding,
        profile=_FOUR_ROLE_PROFILE,
    )


def render_calibrated_portfolio_selection_prompt_for_allocator(
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
    allocator: CalibratedPortfolioAllocator,
    *,
    constraint_decoupled: bool = False,
    minimum_intervention_projection: bool = False,
    evidence_calibrated_source_mix: bool = False,
    contextual_search_allocation: bool = False,
) -> str:
    """Render the sealed K8 prompt for one explicitly identified allocator."""

    if type(request) is not PortfolioSelectionRequest:
        raise TypeError("request must be exact PortfolioSelectionRequest")
    request.__post_init__()
    required_evaluation_family_bounds: tuple[tuple[str, int, int], ...] = ()
    if type(allocator) is HorizonBoundedStructuralPosteriorSlatePolicy:
        active_phase = allocator.exposure_phase_for_wave(binding.context.wave_index)
        required_evaluation_family_bounds = tuple(
            (
                value.family,
                value.minimum_evaluations,
                value.maximum_evaluations,
            )
            for value in active_phase.bounds
        )
        if request.require_pairwise_disjoint_parent_patches:
            option_ids = (
                tuple(
                    option.option_id
                    for option in request.finite_variation_contract.options
                )
                if binding.common_candidate_pool is None
                else binding.common_candidate_pool.option_ids
            )
            required_evaluation_family_bounds = (
                project_family_exposure_bounds_to_pairwise_disjoint_feasibility(
                    request.finite_variation_contract,
                    option_ids,
                    portfolio_size=request.portfolio_size,
                    min_distinct_families=request.min_distinct_families,
                    requested_bounds=required_evaluation_family_bounds,
                )
            )
    return _render_calibrated_portfolio_selection_prompt(
        request,
        binding,
        profile=_profile_for_allocator_authority(
            allocator,
            constraint_decoupled=constraint_decoupled,
            minimum_intervention_projection=minimum_intervention_projection,
            evidence_calibrated_source_mix=evidence_calibrated_source_mix,
            contextual_search_allocation=contextual_search_allocation,
        ),
        required_evaluation_family_bounds=required_evaluation_family_bounds,
    )


def _render_calibrated_portfolio_selection_prompt(
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
    *,
    profile: _CalibratedPortfolioSelectionProfile,
    required_evaluation_family_bounds: tuple[tuple[str, int, int], ...] = (),
) -> str:
    _validate_binding_for_request(
        request,
        binding,
        selector_policy_definition_sha256=profile.policy_definition_sha256,
        constraint_decoupled=profile.constraint_decoupled,
        contextual_search_allocation=profile.contextual_search_allocation,
    )
    context = binding.context
    contract = request.finite_variation_contract
    feasibility_witness_mode = _feasibility_witness_mode_for_binding(
        request,
        binding,
        constraint_decoupled=profile.constraint_decoupled,
    )
    common_pool_mode = (
        feasibility_witness_mode
        is CalibratedPortfolioFeasibilityWitnessMode.TASK_KEYED_COMMON_POOL
    )
    hidden_certificate_mode = (
        feasibility_witness_mode
        is CalibratedPortfolioFeasibilityWitnessMode.HIDDEN_CERTIFICATE
    )
    selectable_option_ids = (
        tuple(value.option_id for value in contract.options)
        if binding.common_candidate_pool is None
        else binding.common_candidate_pool.option_ids
    )
    hierarchy = _hierarchical_composition_shape(
        contract,
        allowed_option_ids=selectable_option_ids,
    )
    if common_pool_mode != (binding.common_candidate_pool is not None):
        raise ValueError(
            "task-keyed common-pool prompt mode and input binding disagree"
        )
    feasibility_witness = None
    if request.require_pairwise_disjoint_parent_patches and not common_pool_mode:
        feasibility_witness = pairwise_disjoint_parent_patch_witness(
            contract,
            tuple(option.option_id for option in contract.options),
            portfolio_size=request.portfolio_size,
            min_distinct_families=request.min_distinct_families,
            ordering_key_sha256=(
                request.request_sha256
                if feasibility_witness_mode
                is CalibratedPortfolioFeasibilityWitnessMode.REQUEST_KEYED
                else None
            ),
        )
        if feasibility_witness is None:
            raise ValueError(
                "validated finite contract lost its disjoint feasibility witness"
            )
    prompt_definition_sha256 = _prompt_definition_sha256_for_binding(
        request,
        binding,
        constraint_decoupled=profile.constraint_decoupled,
    )
    machine_contract: dict[str, object] = {
        "schema_version": (
            3
            + (0 if binding.option_prompt_projection is None else 1)
            + (0 if request.memory_dose_contract is None else 2)
            + (0 if binding.proposal_support is None else 16)
            + (0 if hierarchy is None else 32)
            + (64 if profile.constraint_decoupled else 0)
            + (
                0
                if feasibility_witness_mode
                is CalibratedPortfolioFeasibilityWitnessMode.CANONICAL
                else (8 if common_pool_mode else (12 if hidden_certificate_mode else 4))
            )
        ),
        "request_sha256": request.request_sha256,
        "input_binding_sha256": binding.binding_sha256,
        "context_sha256": request.context_sha256,
        "context": thaw_json(request.context),
        "finite_variation_contract": {
            "catalog_id": contract.catalog_id,
            "catalog_version": contract.catalog_version,
            "catalog_definition_sha256": contract.catalog_definition_sha256,
            "parent_configuration_sha256": contract.parent_configuration_sha256,
            "contract_identity_sha256": contract.identity_sha256,
        },
        "ordered_options": list(binding.prompt_records_for(request)),
        "cards": [card.prompt_record() for card in request.cards],
        "proposal_constraints": {
            "proposal_size": CALIBRATED_PORTFOLIO_PROPOSAL_SIZE,
            "engine_evaluation_size": request.portfolio_size,
            "distinct_option_ids": not profile.constraint_decoupled,
            "required_metric_ids": list(request.required_metric_ids),
            "confidence_bins": ["low", "medium", "high", "unknown"],
            "role_proposals": ["exploit", "falsify", "coverage"],
            "assigned_card_keys": list(context.assigned_card_keys),
            "require_supporting_cards": (
                request.require_supporting_cards and not profile.constraint_decoupled
            ),
            **(
                {}
                if not profile.constraint_decoupled
                else {"authority": "engine_reconciles_cross_member_constraints"}
            ),
            "require_pairwise_disjoint_parent_patches": (
                request.require_pairwise_disjoint_parent_patches
            ),
            **(
                {}
                if feasibility_witness is None
                else (
                    {
                        "engine_verified_feasibility_certificate": {
                            "schema_version": 1,
                            "feasible_subset_exists": True,
                            "certificate_sha256": hashlib.sha256(
                                b"agent-evolve:hidden-feasibility-certificate:v1\x00"
                                + bytes.fromhex(contract.identity_sha256)
                                + _canonical_json(list(feasibility_witness))
                            ).hexdigest(),
                            "member_option_ids_rendered": False,
                            "objective_values_consulted": False,
                            "is_quality_recommendation": False,
                        }
                    }
                    if hidden_certificate_mode
                    else {
                        "engine_verified_feasible_option_id_witness": list(
                            feasibility_witness
                        ),
                        "witness_objective_values_consulted": False,
                        "witness_is_quality_recommendation": False,
                        **(
                            {}
                            if feasibility_witness_mode
                            is CalibratedPortfolioFeasibilityWitnessMode.CANONICAL
                            else {
                                "witness_ordering_policy": (
                                    "request_keyed_domain_separated_sha256_v1"
                                ),
                                "witness_ordering_key_sha256": (request.request_sha256),
                            }
                        ),
                    }
                )
            ),
            **(
                {}
                if request.min_distinct_families is None
                else {"min_distinct_families": request.min_distinct_families}
            ),
            **(
                {}
                if hierarchy is None
                else {"hierarchical_composition": hierarchy.to_prompt_record()}
            ),
            **(
                {}
                if binding.common_candidate_pool is None
                else {
                    "task_keyed_common_candidate_pool": (
                        binding.common_candidate_pool.to_prompt_record()
                    ),
                    "candidate_universe_membership_fixed": True,
                    "model_selection_size": (
                        binding.common_candidate_pool.model_selection_size
                    ),
                    "exact_common_pool_membership_required": (
                        binding.common_candidate_pool.candidate_pool_size
                        == binding.common_candidate_pool.model_selection_size
                    ),
                    "common_pool_order_is_not_a_quality_ranking": True,
                }
            ),
        },
    }
    if request.memory_dose_contract is not None:
        machine_contract["proposal_constraints"]["memory_dose_contract"] = (
            request.memory_dose_contract.to_record()
        )
    if binding.proposal_support is not None:
        machine_contract["proposal_constraints"]["proposal_support"] = (
            binding.proposal_support.to_prompt_record()
        )
    if required_evaluation_family_bounds:
        machine_contract["proposal_constraints"][
            "required_evaluation_family_bounds"
        ] = [
            {
                "family": family,
                "minimum_evaluations": minimum,
                "maximum_evaluations": maximum,
            }
            for family, minimum, maximum in required_evaluation_family_bounds
        ]
    projection_contract = binding.prompt_projection_contract_for(request)
    if projection_contract is not None:
        machine_contract["prompt_definition_sha256"] = prompt_definition_sha256
        machine_contract["option_prompt_projection"] = projection_contract
    encoded = json.dumps(
        machine_contract,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    allocation_constraint = ""
    if profile.constraint_decoupled:
        pool_size = len(selectable_option_ids)
        allocation_constraint = (
            " Rank eight legal semantic suggestions from the fixed candidate "
            f"universe of {pool_size}. Aim for diverse high-value actions, but "
            "do not solve a global slate certificate. Trusted code owns "
            "deduplication, structural-support insertion, nearest-feasible "
            "post-recourse composition exposure, bounded card assignment, "
            "feasible K4 construction, and deterministic refill. Duplicate "
            "suggestions are accepted but may "
            "be replaced and therefore reduce retained model influence."
        )
    elif common_pool_mode:
        assert binding.common_candidate_pool is not None
        pool_size = binding.common_candidate_pool.candidate_pool_size
        if pool_size == CALIBRATED_PORTFOLIO_PROPOSAL_SIZE:
            allocation_constraint = (
                " The eight returned members MUST be exactly the eight option IDs "
                "in the task-keyed common candidate universe, each exactly once. "
                "Rerank them by expected Pareto value; do not add or replace "
                "members. Trusted code has certified that the fixed universe "
                "contains a feasible evaluation allocation. Presentation order "
                "is not a quality hint."
            )
        else:
            allocation_constraint = (
                " Select and rank exactly eight distinct option IDs from the "
                f"fixed task-keyed candidate universe of {pool_size}. Do not "
                "invent or use options outside that universe. The selected eight "
                "MUST contain at least one subset satisfying the engine's exact "
                "evaluation-size, changed-path, and family constraints. Rank the "
                "selected eight by expected Pareto value. Trusted code has "
                "certified feasibility of the complete universe, but its hidden "
                "certificate is not a quality hint. Presentation order is not a "
                "quality hint."
            )
    elif request.require_pairwise_disjoint_parent_patches:
        family_requirement = (
            ""
            if request.min_distinct_families is None
            else (
                " and spans at least "
                f"{request.min_distinct_families} distinct option families"
            )
        )
        allocation_constraint = (
            " The eight proposals MUST contain at least one subset of exactly "
            f"{request.portfolio_size} options whose parent-relative changed "
            f"paths are pairwise disjoint{family_requirement}. "
            + (
                "Trusted code certifies that such a subset exists but withholds "
                "its member IDs to prevent answer anchoring. Derive a feasible, "
                "quality-ranked slate from the supplied option records; the "
                "engine will validate the hard path and family constraints."
                if hidden_certificate_mode
                else (
                    "The machine contract supplies an engine-verified feasible "
                    "option-ID witness. It is a structural fallback, not a "
                    "quality ranking: either use that witness or replace members "
                    "only while preserving the same hard path and family constraints."
                )
            )
        )
    if required_evaluation_family_bounds and not profile.constraint_decoupled:
        rendered_bounds = ", ".join(
            f"{family}=[{minimum},{maximum}]"
            for family, minimum, maximum in required_evaluation_family_bounds
        )
        allocation_constraint += (
            " The feasible evaluation subset MUST satisfy these active, "
            f"pre-registered family exposure bounds: {rendered_bounds}."
        )
    if binding.proposal_support is not None and not profile.constraint_decoupled:
        required = ", ".join(binding.proposal_support.required_option_ids)
        allocation_constraint += (
            " Include the engine-reserved structural support option IDs "
            f"[{required}] somewhere in the eight-member proposal. These two "
            "reservations protect archive novelty and structural coverage; "
            "they are not quality rankings, may appear at any rank, and do not "
            "force an evaluator slot."
        )
    memory_dose_constraint = ""
    if request.memory_dose_contract is not None:
        lower, upper = request.memory_dose_contract.proposed_supported_member_bounds
        minimum_unattributed = (
            request.memory_dose_contract.minimum_unattributed_proposed_members
        )
        memory_dose_constraint = (
            (
                " Card citations are optional semantic attributions. Trusted "
                "code verifies compatibility and assigns the final bounded "
                f"dose of {lower}..{upper} supported members while preserving "
                f"at least {minimum_unattributed} unattributed members."
            )
            if profile.constraint_decoupled
            else (
                " Explicitly cite cards on between "
                f"{lower} and {upper} proposals, leave at least "
                f"{minimum_unattributed} proposals without card attribution, "
                "and cite a card only on one of its declared compatible options. "
                "Unattributed proposals remain prompt-exposed exploration, not "
                "blinded controls."
            )
        )
    composition_constraint = ""
    if hierarchy is not None:
        composition_constraint = (
            (
                " Submit exactly "
                f"{hierarchy.required_composite_proposals} compose_r2 suggestions "
                "in the raw eight-member response and rank legal atomic and "
                "compose_r2 suggestions together. For "
                "each composite suggestion, copy its exact engine-supplied ID "
                "and component binding. After evaluation, memory, and structural "
                "obligations are bound, trusted code may authenticate the nearest "
                "feasible composite count in the reconciled eight-member slate."
            )
            if profile.constraint_decoupled
            else (
                " Use the ranked action union: return exactly "
                f"{hierarchy.required_composite_proposals} members with "
                "action_kind=compose_r2 and the remaining members with "
                "action_kind=atomic. For each compose_r2 member, copy one exact "
                "engine-supplied composite_option_id and its bound two "
                "component_option_ids. Rank atomic and composed actions together "
                "by expected Pareto value. The engine, not the model, owns and "
                "has already verified composite materialization."
            )
        )
    return "\n".join(
        (
            CALIBRATED_PORTFOLIO_BASE_INSTRUCTION,
            "",
            "CALIBRATED PORTFOLIO SLATE PROPOSAL CONTRACT",
            encoded,
            (
                "Return exactly eight legal sealed semantic suggestions. For each, cite "
                if profile.constraint_decoupled
                else (
                    "Return exactly eight distinct sealed option IDs. For each, cite "
                    if hierarchy is None
                    else "Return exactly eight distinct sealed actions. For each, cite "
                )
            )
            + "only supplied card keys, predict every required metric with one "
            "closed confidence bin, propose one closed role, and give a concise "
            "rationale. "
            + (
                "The engine will allocate four evaluations from the slate."
                if request.portfolio_size == CALIBRATED_PORTFOLIO_EVALUATION_SIZE
                else "The engine will evaluate all eight proposals in the slate."
                if request.portfolio_size == CALIBRATED_PORTFOLIO_PROPOSAL_SIZE
                else (
                    "The engine will allocate "
                    f"{request.portfolio_size} evaluations from the slate."
                )
            )
            + allocation_constraint
            + memory_dose_constraint
            + composition_constraint,
        )
    )


def _validated_response(
    result: object,
    *,
    output_type: type[BaseModel],
) -> tuple[StructuredGenerationResponse[Any], int]:
    if type(result) is AttemptedStructuredGenerationResponse:
        AttemptedStructuredGenerationResponse.__post_init__(result)
        response = result.response
        attempt_count = result.attempt_count
    elif type(result) is StructuredGenerationResponse:
        response = result
        attempt_count = 1
    else:
        raise TypeError(
            "low-level runner must return StructuredGenerationResponse or "
            "AttemptedStructuredGenerationResponse"
        )
    StructuredGenerationResponse.__post_init__(response)
    if type(response.value) is not output_type:
        raise TypeError("low-level response value differs from calibrated schema")
    return response, attempt_count


def _telemetry(
    response: StructuredGenerationResponse[Any],
    *,
    attempt_count: int,
) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model=response.requested_model,
        resolved_model=response.resolved_model,
        resolved_provider=response.resolved_provider,
        provider_response_id=response.provider_response_id,
        finish_reason=response.finish_reason,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        reasoning_tokens=response.reasoning_tokens,
        cache_read_tokens=response.cache_read_tokens,
        cache_write_tokens=response.cache_write_tokens,
        cost_usd=response.cost_usd,
        latency_ns=response.latency_ns,
        attempt_count=attempt_count,
    )


def _raw_proposal_record(
    request: PortfolioSelectionRequest,
    context: CalibratedPortfolioAllocationContext,
    members: tuple[Any, ...],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "request_sha256": request.request_sha256,
        "scope_sha256": context.scope.scope_sha256,
        "wave_index": context.wave_index,
        "parent_candidate_identity_sha256": (context.parent_candidate_identity_sha256),
        "members": [
            {
                "model_rank": index,
                "option_id": _resolved_output_option_id(member),
                **(
                    {}
                    if not hasattr(member, "action_kind")
                    else {
                        "hierarchical_action": (
                            {
                                "action_kind": "compose_r2",
                                "component_option_ids": list(
                                    member.component_option_ids
                                ),
                            }
                            if member.action_kind == "compose_r2"
                            else {"action_kind": "atomic"}
                        )
                    }
                ),
                "supporting_card_keys": list(member.supporting_card_keys),
                "effect_predictions": [
                    {
                        "metric_id": prediction.metric_id,
                        "direction": prediction.direction,
                        "confidence": prediction.confidence,
                    }
                    for prediction in member.effect_predictions
                ],
                "role_proposal": member.role_proposal,
                "design_rationale": member.design_rationale,
            }
            for index, member in enumerate(members, start=1)
        ],
    }


@dataclass(frozen=True, slots=True)
class CalibratedPortfolioModelPrediction:
    """One canonical prediction exactly as proposed in the K8 response."""

    metric_id: str
    direction: MetricEffectDirection
    confidence: ForecastConfidenceBin

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or not self.metric_id:
            raise ValueError("metric_id must be a non-empty string")
        if type(self.direction) is not MetricEffectDirection:
            raise TypeError("direction must be exact MetricEffectDirection")
        if type(self.confidence) is not ForecastConfidenceBin:
            raise TypeError("confidence must be exact ForecastConfidenceBin")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "direction": self.direction.value,
            "confidence": self.confidence.value,
        }


@dataclass(frozen=True, slots=True)
class CalibratedPortfolioModelMember:
    """One authenticated model-ranked member of the original K8 proposal."""

    model_rank: int
    option_id: str
    supporting_card_keys: tuple[str, ...]
    effect_predictions: tuple[CalibratedPortfolioModelPrediction, ...]
    role_proposal: SlateRoleProposal
    design_rationale: str

    def __post_init__(self) -> None:
        if type(self.model_rank) is not int or self.model_rank <= 0:
            raise ValueError("model_rank must be a positive exact integer")
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be a non-empty string")
        if type(self.supporting_card_keys) is not tuple or any(
            type(value) is not str or not value for value in self.supporting_card_keys
        ):
            raise TypeError("supporting_card_keys must contain exact strings")
        if len(set(self.supporting_card_keys)) != len(self.supporting_card_keys):
            raise ValueError("supporting_card_keys cannot repeat")
        if type(self.effect_predictions) is not tuple or any(
            type(value) is not CalibratedPortfolioModelPrediction
            for value in self.effect_predictions
        ):
            raise TypeError("effect_predictions must contain exact predictions")
        for value in self.effect_predictions:
            value.__post_init__()
        if not self.effect_predictions:
            raise ValueError("effect_predictions cannot be empty")
        if type(self.role_proposal) is not SlateRoleProposal:
            raise TypeError("role_proposal must be exact SlateRoleProposal")
        if type(self.design_rationale) is not str or not self.design_rationale:
            raise ValueError("design_rationale must be a non-empty string")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "model_rank": self.model_rank,
            "option_id": self.option_id,
            "supporting_card_keys": list(self.supporting_card_keys),
            "effect_predictions": [
                value.to_record() for value in self.effect_predictions
            ],
            "role_proposal": self.role_proposal.value,
            "design_rationale": self.design_rationale,
        }


def _typed_model_members(
    proposal_record: dict[str, object],
) -> tuple[CalibratedPortfolioModelMember, ...]:
    rows = proposal_record["members"]
    if type(rows) is not list:  # Canonical schema validation has already run.
        raise AssertionError("proposal members did not remain a list")
    members: list[CalibratedPortfolioModelMember] = []
    for row in rows:
        if type(row) is not dict:
            raise AssertionError("proposal member did not remain an object")
        predictions = row["effect_predictions"]
        if type(predictions) is not list:
            raise AssertionError("proposal predictions did not remain a list")
        typed_predictions: list[CalibratedPortfolioModelPrediction] = []
        for prediction in predictions:
            if type(prediction) is not dict:
                raise AssertionError("proposal prediction did not remain an object")
            typed_predictions.append(
                CalibratedPortfolioModelPrediction(
                    metric_id=cast(str, prediction["metric_id"]),
                    direction=MetricEffectDirection(cast(str, prediction["direction"])),
                    confidence=ForecastConfidenceBin(
                        cast(str, prediction["confidence"])
                    ),
                )
            )
        members.append(
            CalibratedPortfolioModelMember(
                model_rank=cast(int, row["model_rank"]),
                option_id=cast(str, row["option_id"]),
                supporting_card_keys=tuple(
                    cast(list[str], row["supporting_card_keys"])
                ),
                effect_predictions=tuple(typed_predictions),
                role_proposal=SlateRoleProposal(cast(str, row["role_proposal"])),
                design_rationale=cast(str, row["design_rationale"]),
            )
        )
    return tuple(members)


class SemanticSlateMemberOrigin(str, Enum):
    """Truthful origin of one member after deterministic slate reconciliation."""

    MODEL = "model"
    ENGINE_REQUIRED_SUPPORT = "engine_required_support"
    ENGINE_FEASIBILITY = "engine_feasibility"
    ENGINE_MEMORY_DOSE = "engine_memory_dose"
    ENGINE_REFILL = "engine_refill"
    ENGINE_GLOBAL_COVERAGE = "engine_global_coverage"
    ENGINE_CONTEXTUAL_ALLOCATION = "engine_contextual_allocation"


@dataclass(frozen=True, slots=True)
class SemanticSlateReconciledMember:
    """One model-retained or engine-inserted member in the final K8."""

    reconciled_rank: int
    option_id: str
    origin: SemanticSlateMemberOrigin
    original_model_rank: int | None
    original_supporting_card_keys: tuple[str, ...]
    reconciled_supporting_card_keys: tuple[str, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.reconciled_rank) is not int or self.reconciled_rank <= 0:
            raise ValueError("reconciled_rank must be a positive exact integer")
        if type(self.option_id) is not str or not self.option_id:
            raise ValueError("option_id must be a non-empty exact string")
        if type(self.origin) is not SemanticSlateMemberOrigin:
            raise TypeError("origin must be exact SemanticSlateMemberOrigin")
        if self.original_model_rank is not None and (
            type(self.original_model_rank) is not int or self.original_model_rank <= 0
        ):
            raise ValueError("original_model_rank must be positive or None")
        for name in (
            "original_supporting_card_keys",
            "reconciled_supporting_card_keys",
            "reasons",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not str or not value for value in values
            ):
                raise TypeError(f"{name} must contain exact non-empty strings")
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "reconciled_rank": self.reconciled_rank,
            "option_id": self.option_id,
            "origin": self.origin.value,
            "original_model_rank": self.original_model_rank,
            "original_supporting_card_keys": list(self.original_supporting_card_keys),
            "reconciled_supporting_card_keys": list(
                self.reconciled_supporting_card_keys
            ),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class ContextualAllocationFeasibilityProjectionReceipt:
    """Authenticated requested-to-realized structural allocation projection."""

    contract_sha256: str
    requested_source_target_counts: tuple[tuple[str, int], ...]
    requested_operator_target_counts: tuple[tuple[str, int], ...]
    realized_source_target_counts: tuple[tuple[str, int], ...]
    realized_operator_target_counts: tuple[tuple[str, int], ...]
    evaluation_option_ids: tuple[str, ...]
    minimum_single_path_interventions: int = 0
    realized_single_path_interventions: int = 0
    minimum_disjoint_parent_patch_pairs: int = 0
    realized_disjoint_parent_patch_pairs: int = 0
    memory_dose_feasibility_witness: MemoryDoseAttributionFeasibilityWitness | None = (
        None
    )

    def __post_init__(self) -> None:
        require_sha256(self.contract_sha256, "contract_sha256")
        if (
            type(self.evaluation_option_ids) is not tuple
            or not self.evaluation_option_ids
            or len(self.evaluation_option_ids)
            > CALIBRATED_PORTFOLIO_PROPOSAL_SIZE
            or any(
                type(value) is not str or not value
                for value in self.evaluation_option_ids
            )
            or self.evaluation_option_ids
            != tuple(sorted(set(self.evaluation_option_ids)))
        ):
            raise ValueError(
                "evaluation_option_ids must be a canonical non-empty evaluation "
                "witness inside the proposed slate"
            )
        evaluation_width = len(self.evaluation_option_ids)
        for name in (
            "requested_source_target_counts",
            "realized_source_target_counts",
            "requested_operator_target_counts",
            "realized_operator_target_counts",
        ):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or not values
                or any(
                    type(value) is not tuple
                    or len(value) != 2
                    or type(value[0]) is not str
                    or not value[0]
                    or type(value[1]) is not int
                    or value[1] < 0
                    for value in values
                )
            ):
                raise TypeError(f"{name} must contain exact non-negative counts")
            if values != tuple(sorted(values)) or len(
                {value[0] for value in values}
            ) != len(values):
                raise ValueError(f"{name} must use canonical unique arms")
            if sum(value[1] for value in values) != evaluation_width:
                raise ValueError(f"{name} must cover the evaluation width")
        if tuple(value[0] for value in self.requested_source_target_counts) != tuple(
            value[0] for value in self.realized_source_target_counts
        ):
            raise ValueError("realized source arms differ from requested arms")
        if tuple(value[0] for value in self.requested_operator_target_counts) != tuple(
            value[0] for value in self.realized_operator_target_counts
        ):
            raise ValueError("realized operator arms differ from requested arms")
        if (
            type(self.minimum_single_path_interventions) is not int
            or not 0
            <= self.minimum_single_path_interventions
            <= evaluation_width
        ):
            raise ValueError("minimum single-path intervention floor is invalid")
        if (
            type(self.realized_single_path_interventions) is not int
            or not 0
            <= self.realized_single_path_interventions
            <= evaluation_width
        ):
            raise ValueError("realized single-path intervention count is invalid")
        if self.realized_single_path_interventions < (
            self.minimum_single_path_interventions
        ):
            raise ValueError("contextual witness violated its single-path floor")
        maximum_pairs = evaluation_width * (evaluation_width - 1) // 2
        if (
            type(self.minimum_disjoint_parent_patch_pairs) is not int
            or not 0 <= self.minimum_disjoint_parent_patch_pairs <= maximum_pairs
        ):
            raise ValueError("minimum disjoint parent-patch pair floor is invalid")
        if (
            type(self.realized_disjoint_parent_patch_pairs) is not int
            or not 0 <= self.realized_disjoint_parent_patch_pairs <= maximum_pairs
        ):
            raise ValueError("realized disjoint parent-patch pair count is invalid")
        if self.realized_disjoint_parent_patch_pairs < (
            self.minimum_disjoint_parent_patch_pairs
        ):
            raise ValueError(
                "contextual witness violated its disjoint parent-pair floor"
            )
        if self.memory_dose_feasibility_witness is not None:
            if type(self.memory_dose_feasibility_witness) is not (
                MemoryDoseAttributionFeasibilityWitness
            ):
                raise TypeError("memory_dose_feasibility_witness must be exact or None")
            self.memory_dose_feasibility_witness.__post_init__()
            if self.memory_dose_feasibility_witness.stage is not (
                PortfolioMemoryDoseStage.EVALUATED_PORTFOLIO
            ):
                raise ValueError(
                    "contextual projection requires an evaluated-dose witness"
                )
            if (
                tuple(
                    value[0]
                    for value in self.memory_dose_feasibility_witness.member_option_identities
                )
                != self.evaluation_option_ids
            ):
                raise ValueError(
                    "memory-dose witness differs from the evaluation witness"
                )

    @staticmethod
    def _l1(
        requested: tuple[tuple[str, int], ...],
        realized: tuple[tuple[str, int], ...],
    ) -> int:
        return sum(
            abs(left[1] - right[1])
            for left, right in zip(requested, realized, strict=True)
        )

    @property
    def source_l1_deviation(self) -> int:
        return self._l1(
            self.requested_source_target_counts,
            self.realized_source_target_counts,
        )

    @property
    def operator_l1_deviation(self) -> int:
        return self._l1(
            self.requested_operator_target_counts,
            self.realized_operator_target_counts,
        )

    @property
    def exact(self) -> bool:
        return self.source_l1_deviation == 0 and self.operator_l1_deviation == 0

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        joint_memory_dose = self.memory_dose_feasibility_witness is not None
        assay_bound = self.minimum_single_path_interventions > 0
        offspring_opportunity_bound = self.minimum_disjoint_parent_patch_pairs > 0
        record: dict[str, object] = {
            "schema_version": (
                5
                if offspring_opportunity_bound
                else 4
                if assay_bound
                else 3
                if joint_memory_dose
                else 2
            ),
            "policy_id": _CONTEXTUAL_ALLOCATION_PROJECTION_POLICY_ID,
            "policy_version": (
                _CONTEXTUAL_ALLOCATION_OFFSPRING_OPPORTUNITY_PROJECTION_POLICY_VERSION
                if offspring_opportunity_bound
                else _CONTEXTUAL_ALLOCATION_ASSAY_PROJECTION_POLICY_VERSION
                if assay_bound
                else _CONTEXTUAL_ALLOCATION_JOINT_DOSE_PROJECTION_POLICY_VERSION
                if joint_memory_dose
                else _CONTEXTUAL_ALLOCATION_PROJECTION_POLICY_VERSION
            ),
            "policy_definition_sha256": (
                _CONTEXTUAL_ALLOCATION_OFFSPRING_OPPORTUNITY_PROJECTION_POLICY_DEFINITION_SHA256
                if offspring_opportunity_bound
                else _CONTEXTUAL_ALLOCATION_ASSAY_PROJECTION_POLICY_DEFINITION_SHA256
                if assay_bound
                else _CONTEXTUAL_ALLOCATION_JOINT_DOSE_PROJECTION_POLICY_DEFINITION_SHA256
                if joint_memory_dose
                else _CONTEXTUAL_ALLOCATION_PROJECTION_POLICY_DEFINITION_SHA256
            ),
            "contract_sha256": self.contract_sha256,
            "requested_source_target_counts": [
                list(value) for value in self.requested_source_target_counts
            ],
            "requested_operator_target_counts": [
                list(value) for value in self.requested_operator_target_counts
            ],
            "realized_source_target_counts": [
                list(value) for value in self.realized_source_target_counts
            ],
            "realized_operator_target_counts": [
                list(value) for value in self.realized_operator_target_counts
            ],
            "source_l1_deviation": self.source_l1_deviation,
            "operator_l1_deviation": self.operator_l1_deviation,
            "exact": self.exact,
            "evaluation_option_ids": list(self.evaluation_option_ids),
            "objective_values_consulted": False,
            "workload_identifiers_consulted": False,
        }
        if assay_bound:
            record["minimum_single_path_interventions"] = (
                self.minimum_single_path_interventions
            )
            record["realized_single_path_interventions"] = (
                self.realized_single_path_interventions
            )
            record["intervention_axis"] = (
                "exact_parent_relative_changed_json_path_count"
            )
        if offspring_opportunity_bound:
            record["minimum_disjoint_parent_patch_pairs"] = (
                self.minimum_disjoint_parent_patch_pairs
            )
            record["realized_disjoint_parent_patch_pairs"] = (
                self.realized_disjoint_parent_patch_pairs
            )
            record["offspring_opportunity_axis"] = (
                "pairwise_disjoint_parent_relative_patch_pairs"
            )
        if joint_memory_dose:
            assert self.memory_dose_feasibility_witness is not None
            record["memory_dose_feasibility_witness"] = (
                self.memory_dose_feasibility_witness.to_record()
            )
            record["joint_constraint_families"] = [
                "bounded_memory_dose",
                "family_exposure",
                "pairwise_parent_patch",
                "source_operator_marginals",
            ]
            record["workload_model_provider_identifiers_consulted"] = False
        if assay_bound:
            record["joint_constraint_families"] = sorted(
                {
                    *record.get("joint_constraint_families", []),
                    "minimum_single_path_interventions",
                    "source_operator_marginals",
                }
            )
        if offspring_opportunity_bound:
            record["joint_constraint_families"] = sorted(
                {
                    *record.get("joint_constraint_families", []),
                    "minimum_disjoint_parent_patch_pairs",
                    "source_operator_marginals",
                }
            )
        return record

    @property
    def projection_sha256(self) -> str:
        domain = (
            _CONTEXTUAL_ALLOCATION_OFFSPRING_OPPORTUNITY_PROJECTION_DOMAIN
            if self.minimum_disjoint_parent_patch_pairs > 0
            else _CONTEXTUAL_ALLOCATION_ASSAY_PROJECTION_DOMAIN
            if self.minimum_single_path_interventions > 0
            else _CONTEXTUAL_ALLOCATION_JOINT_DOSE_PROJECTION_DOMAIN
            if self.memory_dose_feasibility_witness is not None
            else _CONTEXTUAL_ALLOCATION_PROJECTION_DOMAIN
        )
        return hashlib.sha256(
            domain + _canonical_json(self._unsigned_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "projection_sha256": self.projection_sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticSlateReconciliationReceipt:
    """Replayable proof of the boundary between model and engine authority."""

    original_model_proposal_sha256: str
    reconciled_proposal_sha256: str
    duplicate_model_member_count: int
    evaluation_feasibility_witness: tuple[str, ...]
    members: tuple[SemanticSlateReconciledMember, ...]
    minimum_intervention_projection: bool = False
    evidence_calibrated_source_mix: bool = False
    protected_allocation_option_ids: tuple[str, ...] = ()
    deferred_proposal_support_option_ids: tuple[str, ...] = ()
    contextual_search_allocation: bool = False
    contextual_allocation_contract_sha256: str | None = None
    contextual_allocation_option_ids: tuple[str, ...] = ()
    contextual_allocation_projection: (
        ContextualAllocationFeasibilityProjectionReceipt | None
    ) = None
    composition_capacity_projection: ExactKCompositionCapacityProjection | None = None

    def __post_init__(self) -> None:
        require_sha256(
            self.original_model_proposal_sha256,
            "original_model_proposal_sha256",
        )
        require_sha256(
            self.reconciled_proposal_sha256,
            "reconciled_proposal_sha256",
        )
        if (
            type(self.duplicate_model_member_count) is not int
            or self.duplicate_model_member_count < 0
        ):
            raise ValueError("duplicate_model_member_count must be non-negative")
        if type(self.minimum_intervention_projection) is not bool:
            raise TypeError("minimum_intervention_projection must be an exact bool")
        if type(self.evidence_calibrated_source_mix) is not bool:
            raise TypeError("evidence_calibrated_source_mix must be an exact bool")
        if (
            self.evidence_calibrated_source_mix
            and not self.minimum_intervention_projection
        ):
            raise ValueError(
                "evidence-calibrated source mix requires minimum intervention"
            )
        if type(self.contextual_search_allocation) is not bool:
            raise TypeError("contextual_search_allocation must be an exact bool")
        if self.contextual_search_allocation and not (
            self.minimum_intervention_projection and self.evidence_calibrated_source_mix
        ):
            raise ValueError(
                "contextual allocation requires minimum-intervention source mix"
            )
        for name in (
            "protected_allocation_option_ids",
            "deferred_proposal_support_option_ids",
            "contextual_allocation_option_ids",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not str or not value for value in values
            ):
                raise TypeError(f"{name} must contain exact option IDs")
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and canonical")
        if not self.evidence_calibrated_source_mix and (
            self.protected_allocation_option_ids
            or self.deferred_proposal_support_option_ids
        ):
            raise ValueError("legacy reconciliation cannot carry source-mix decisions")
        if self.contextual_search_allocation:
            if self.contextual_allocation_contract_sha256 is None:
                raise ValueError("contextual reconciliation omitted its contract")
            require_sha256(
                self.contextual_allocation_contract_sha256,
                "contextual_allocation_contract_sha256",
            )
            if not self.contextual_allocation_option_ids:
                raise ValueError("contextual reconciliation omitted allocated options")
            projection = self.contextual_allocation_projection
            if type(projection) is not (
                ContextualAllocationFeasibilityProjectionReceipt
            ):
                raise TypeError("contextual reconciliation omitted its projection")
            projection.__post_init__()
            if (
                projection.contract_sha256 != self.contextual_allocation_contract_sha256
                or set(projection.evaluation_option_ids)
                != set(self.contextual_allocation_option_ids)
            ):
                raise ValueError(
                    "contextual projection differs from its contract or witness"
                )
            if self.protected_allocation_option_ids:
                raise ValueError(
                    "contextual and fixed protected allocations cannot be combined"
                )
        elif (
            self.contextual_allocation_contract_sha256 is not None
            or self.contextual_allocation_option_ids
            or self.contextual_allocation_projection is not None
        ):
            raise ValueError(
                "non-contextual reconciliation cannot carry contextual allocation"
            )
        if self.composition_capacity_projection is not None:
            if not self.contextual_search_allocation:
                raise ValueError(
                    "post-recourse composition projection requires contextual search"
                )
            if type(self.composition_capacity_projection) is not (
                ExactKCompositionCapacityProjection
            ):
                raise TypeError("composition capacity projection has the wrong type")
            self.composition_capacity_projection.__post_init__()
        if (
            type(self.evaluation_feasibility_witness) is not tuple
            or not self.evaluation_feasibility_witness
            or any(
                type(value) is not str or not value
                for value in self.evaluation_feasibility_witness
            )
        ):
            raise ValueError("evaluation_feasibility_witness must be non-empty")
        if len(set(self.evaluation_feasibility_witness)) != len(
            self.evaluation_feasibility_witness
        ):
            raise ValueError("evaluation feasibility witness cannot repeat")
        if (
            type(self.members) is not tuple
            or len(self.members) != CALIBRATED_PORTFOLIO_PROPOSAL_SIZE
            or any(
                type(value) is not SemanticSlateReconciledMember
                for value in self.members
            )
        ):
            raise ValueError("members must contain exactly eight exact receipts")
        for value in self.members:
            value.__post_init__()
        if tuple(value.reconciled_rank for value in self.members) != tuple(
            range(1, CALIBRATED_PORTFOLIO_PROPOSAL_SIZE + 1)
        ):
            raise ValueError("reconciled member ranks must be contiguous")
        if len({value.option_id for value in self.members}) != len(self.members):
            raise ValueError("reconciled members cannot repeat options")
        if not set(self.evaluation_feasibility_witness).issubset(
            {value.option_id for value in self.members}
        ):
            raise ValueError("evaluation witness escapes the reconciled slate")
        if not set(self.protected_allocation_option_ids).issubset(
            self.evaluation_feasibility_witness
        ):
            raise ValueError(
                "protected allocation options must enter the feasibility witness"
            )
        if self.contextual_search_allocation and set(
            self.contextual_allocation_option_ids
        ) != set(self.evaluation_feasibility_witness):
            raise ValueError(
                "contextual allocation must equal the exact evaluation witness"
            )

    @property
    def required_allocation_option_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    *self.protected_allocation_option_ids,
                    *self.contextual_allocation_option_ids,
                }
            )
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        retained_model_member_count = sum(
            value.origin is SemanticSlateMemberOrigin.MODEL for value in self.members
        )
        engine_inserted_member_count = len(self.members) - retained_model_member_count
        witness_ids = set(self.evaluation_feasibility_witness)
        witness_retained_model_member_count = sum(
            value.origin is SemanticSlateMemberOrigin.MODEL
            and value.option_id in witness_ids
            for value in self.members
        )
        record: dict[str, object] = {
            "schema_version": (
                6
                if self.contextual_search_allocation
                and self.composition_capacity_projection is not None
                else 5
                if self.contextual_search_allocation
                else 3
                if self.evidence_calibrated_source_mix
                else 2
                if self.minimum_intervention_projection
                else 1
            ),
            "policy_id": (
                _CONTEXTUAL_SEARCH_RECONCILIATION_POLICY_ID
                if self.contextual_search_allocation
                else _SOURCE_MIX_RECONCILIATION_POLICY_ID
                if self.evidence_calibrated_source_mix
                else _MINIMUM_INTERVENTION_RECONCILIATION_POLICY_ID
                if self.minimum_intervention_projection
                else _SEMANTIC_RECONCILIATION_POLICY_ID
            ),
            "policy_version": (
                _CONTEXTUAL_SEARCH_RECONCILIATION_POLICY_VERSION
                if self.contextual_search_allocation
                else _SOURCE_MIX_RECONCILIATION_POLICY_VERSION
                if self.evidence_calibrated_source_mix
                else _MINIMUM_INTERVENTION_RECONCILIATION_POLICY_VERSION
                if self.minimum_intervention_projection
                else _SEMANTIC_RECONCILIATION_POLICY_VERSION
            ),
            "policy_definition_sha256": (
                _CONTEXTUAL_SEARCH_RECONCILIATION_POLICY_DEFINITION_SHA256
                if self.contextual_search_allocation
                else _SOURCE_MIX_RECONCILIATION_POLICY_DEFINITION_SHA256
                if self.evidence_calibrated_source_mix
                else _MINIMUM_INTERVENTION_RECONCILIATION_POLICY_DEFINITION_SHA256
                if self.minimum_intervention_projection
                else _SEMANTIC_RECONCILIATION_POLICY_DEFINITION_SHA256
            ),
            "original_model_proposal_sha256": (self.original_model_proposal_sha256),
            "reconciled_proposal_sha256": self.reconciled_proposal_sha256,
            "duplicate_model_member_count": self.duplicate_model_member_count,
            "evaluation_feasibility_witness": list(self.evaluation_feasibility_witness),
            "members": [value.to_record() for value in self.members],
            "objective_values_consulted": False,
            "workload_identifiers_consulted": False,
        }
        if self.contextual_search_allocation:
            record.update(
                {
                    "projection_objective": [
                        "satisfy_prospective_source_marginals",
                        "satisfy_prospective_operator_marginals",
                        "satisfy_structural_and_memory_feasibility",
                        "maximize_retained_model_member_count",
                        "prefer_original_model_rank_lexicographically",
                        "canonical_nonmodel_tie_break",
                    ],
                    "contextual_allocation_contract_sha256": (
                        self.contextual_allocation_contract_sha256
                    ),
                    "contextual_allocation_option_ids": list(
                        self.contextual_allocation_option_ids
                    ),
                    "contextual_allocation_projection": (
                        self.contextual_allocation_projection.to_record()
                    ),
                    "composition_capacity_projection": (
                        None
                        if self.composition_capacity_projection is None
                        else self.composition_capacity_projection.to_record()
                    ),
                    "deferred_proposal_support_option_ids": list(
                        self.deferred_proposal_support_option_ids
                    ),
                    "retained_model_member_count": retained_model_member_count,
                    "engine_inserted_member_count": engine_inserted_member_count,
                    "semantic_intervention_count": engine_inserted_member_count,
                    "evaluation_witness_retained_model_member_count": (
                        witness_retained_model_member_count
                    ),
                    "reconciliation_outcomes_consulted": False,
                    "controller_prior_outcomes_bound_by_contract": True,
                }
            )
        elif self.evidence_calibrated_source_mix:
            record.update(
                {
                    "projection_objective": [
                        "require_task_keyed_global_source_when_available_in_wave_one",
                        "maximize_retained_model_member_count",
                        "prefer_original_model_rank_lexicographically",
                        "canonical_nonmodel_tie_break",
                    ],
                    "protected_allocation_option_ids": list(
                        self.protected_allocation_option_ids
                    ),
                    "deferred_proposal_support_option_ids": list(
                        self.deferred_proposal_support_option_ids
                    ),
                    "retained_model_member_count": (retained_model_member_count),
                    "engine_inserted_member_count": (engine_inserted_member_count),
                    "semantic_intervention_count": (engine_inserted_member_count),
                    "evaluation_witness_retained_model_member_count": (
                        witness_retained_model_member_count
                    ),
                    "protected_source_count": len(self.protected_allocation_option_ids),
                    "source_mix_outcomes_consulted": False,
                }
            )
        elif self.minimum_intervention_projection:
            record.update(
                {
                    "projection_objective": [
                        "maximize_retained_model_member_count",
                        "prefer_original_model_rank_lexicographically",
                        "canonical_nonmodel_tie_break",
                    ],
                    "retained_model_member_count": (retained_model_member_count),
                    "engine_inserted_member_count": (engine_inserted_member_count),
                    "semantic_intervention_count": (engine_inserted_member_count),
                    "evaluation_witness_retained_model_member_count": (
                        witness_retained_model_member_count
                    ),
                }
            )
        return record

    @property
    def receipt_sha256(self) -> str:
        return hashlib.sha256(
            (
                _CONTEXTUAL_SEARCH_RECONCILIATION_DOMAIN
                if self.contextual_search_allocation
                else _SOURCE_MIX_RECONCILIATION_DOMAIN
                if self.evidence_calibrated_source_mix
                else _MINIMUM_INTERVENTION_RECONCILIATION_DOMAIN
                if self.minimum_intervention_projection
                else _SEMANTIC_RECONCILIATION_DOMAIN
            )
            + _canonical_json(self._unsigned_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "receipt_sha256": self.receipt_sha256,
        }


def _evaluation_subset_is_structurally_feasible(
    request: PortfolioSelectionRequest,
    option_ids: tuple[str, ...],
    *,
    required_evaluation_family_bounds: tuple[tuple[str, int, int], ...],
) -> bool:
    if len(option_ids) != request.portfolio_size:
        return False
    contract = request.finite_variation_contract
    families = tuple(contract.resolve(value).family for value in option_ids)
    if (
        request.min_distinct_families is not None
        and len(set(families)) < request.min_distinct_families
    ):
        return False
    if any(
        not minimum <= families.count(family) <= maximum
        for family, minimum, maximum in required_evaluation_family_bounds
    ):
        return False
    return not request.require_pairwise_disjoint_parent_patches or (
        finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
            contract,
            option_ids,
            portfolio_size=request.portfolio_size,
            min_distinct_families=request.min_distinct_families,
            family_exposure_bounds=required_evaluation_family_bounds,
        )
    )


def _first_evaluation_witness(
    request: PortfolioSelectionRequest,
    option_ids: tuple[str, ...],
    *,
    required_evaluation_family_bounds: tuple[tuple[str, int, int], ...],
    preferred_option_ids: tuple[str, ...] = (),
    required_option_ids: tuple[str, ...] = (),
) -> tuple[str, ...] | None:
    if request.require_pairwise_disjoint_parent_patches:
        return pairwise_disjoint_parent_patch_witness(
            request.finite_variation_contract,
            option_ids,
            portfolio_size=request.portfolio_size,
            min_distinct_families=request.min_distinct_families,
            family_exposure_bounds=required_evaluation_family_bounds,
            preferred_option_ids=preferred_option_ids,
            required_option_ids=required_option_ids,
        )
    if type(preferred_option_ids) is not tuple or any(
        type(value) is not str for value in preferred_option_ids
    ):
        raise TypeError("preferred evaluation options must be an exact string tuple")
    preferred_set = set(preferred_option_ids)
    if len(preferred_set) != len(preferred_option_ids):
        raise ValueError("preferred evaluation options cannot repeat")
    if not preferred_set.issubset(option_ids):
        raise ValueError("preferred evaluation options escape the pool")
    required_set = set(required_option_ids)
    if len(required_set) != len(required_option_ids):
        raise ValueError("required evaluation options cannot repeat")
    if required_option_ids != tuple(sorted(required_set)):
        raise ValueError("required evaluation options must be canonical")
    if not required_set.issubset(option_ids):
        raise ValueError("required evaluation options escape the pool")
    if len(required_option_ids) > request.portfolio_size:
        return None
    remaining = tuple(value for value in option_ids if value not in required_set)
    remaining_preferred = tuple(
        value for value in preferred_option_ids if value not in required_set
    )
    if remaining_preferred:
        preferred_set = set(remaining_preferred)
        nonpreferred = tuple(value for value in remaining if value not in preferred_set)
        for retained_count in range(
            min(
                request.portfolio_size - len(required_option_ids),
                len(remaining_preferred),
            ),
            -1,
            -1,
        ):
            for preferred_subset in combinations(
                remaining_preferred,
                retained_count,
            ):
                completion_size = (
                    request.portfolio_size - len(required_option_ids) - retained_count
                )
                for completion in combinations(nonpreferred, completion_size):
                    subset = (
                        *required_option_ids,
                        *preferred_subset,
                        *completion,
                    )
                    if _evaluation_subset_is_structurally_feasible(
                        request,
                        subset,
                        required_evaluation_family_bounds=(
                            required_evaluation_family_bounds
                        ),
                    ):
                        return subset
        return None
    for completion in combinations(
        remaining,
        request.portfolio_size - len(required_option_ids),
    ):
        subset = (*required_option_ids, *completion)
        if _evaluation_subset_is_structurally_feasible(
            request,
            subset,
            required_evaluation_family_bounds=required_evaluation_family_bounds,
        ):
            return subset
    return None


def _feasible_evaluation_subsets(
    request: PortfolioSelectionRequest,
    option_ids: tuple[str, ...],
    *,
    required_evaluation_family_bounds: tuple[tuple[str, int, int], ...],
    required_option_ids: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], ...]:
    """Enumerate feasible K-subsets after one finite-contract validation."""

    contract = request.finite_variation_contract
    family_by_option = {value.option_id: value.family for value in contract.options}
    allowed_pairs = (
        None
        if not request.require_pairwise_disjoint_parent_patches
        else {
            frozenset(value)
            for value in pairwise_disjoint_parent_patch_pairs(
                contract,
                option_ids,
            )
        }
    )
    feasible: list[tuple[str, ...]] = []
    required_set = set(required_option_ids)
    for subset in combinations(option_ids, request.portfolio_size):
        if not required_set.issubset(subset):
            continue
        families = tuple(family_by_option[value] for value in subset)
        if (
            request.min_distinct_families is not None
            and len(set(families)) < request.min_distinct_families
        ):
            continue
        if any(
            not minimum <= families.count(family) <= maximum
            for family, minimum, maximum in required_evaluation_family_bounds
        ):
            continue
        if allowed_pairs is not None and any(
            frozenset((left, right)) not in allowed_pairs
            for index, left in enumerate(subset)
            for right in subset[index + 1 :]
        ):
            continue
        feasible.append(subset)
    return tuple(feasible)


def _contextual_allocation_feasibility_projection(
    request: PortfolioSelectionRequest,
    option_ids: tuple[str, ...],
    *,
    original_by_option: dict[str, CalibratedPortfolioModelMember],
    composite_option_ids: set[str],
    binding: CalibratedPortfolioInputBinding,
    required_evaluation_family_bounds: tuple[tuple[str, int, int], ...],
) -> ContextualAllocationFeasibilityProjectionReceipt | None:
    """Jointly project marginals, structure, and memory dose to a feasible K4."""

    contract = binding.contextual_allocation
    if contract is None:
        raise ValueError("contextual witness requires an allocation contract")
    contract.__post_init__()
    if contract.evaluation_slots != request.portfolio_size:
        raise ValueError("contextual contract differs from the portfolio width")
    requested_source = contract.source_target_counts
    requested_operator = contract.operator_target_counts
    source_targets = dict(requested_source)
    operator_targets = dict(requested_operator)
    source_arm_ids = tuple(source_targets)
    operator_arm_ids = tuple(operator_targets)
    if source_arm_ids == ("engine", "model"):
        # Backward-compatible replay for already sealed v3 experiments.  New
        # plans use finite-variation source IDs and never enter this branch.
        source_by_option = {
            value: "model" if value in original_by_option else "engine"
            for value in option_ids
        }
        legacy_required_variation_source_option_ids = set(
            required_source_evaluation_option_ids(request.finite_variation_contract)
        )
        ranked_required_variation_source_option_ids: set[str] = set()
        required_variation_source_counts: dict[str, int] = {}
        # Preserve already-sealed v3 experiments whose operator axis predated
        # explicit finite-option operator metadata.
        operator_by_option = {
            value: "composite" if value in composite_option_ids else "atomic"
            for value in option_ids
        }
    else:
        contract_source_by_option = finite_variation_source_by_option(
            request.finite_variation_contract
        )
        if not set(option_ids).issubset(contract_source_by_option):
            raise ValueError("contextual universe escapes finite source attribution")
        source_by_option = {
            value: contract_source_by_option[value] for value in option_ids
        }
        if not set(source_by_option.values()).issubset(source_targets):
            raise ValueError("contextual contract omits a finite variation source")
        legacy_required_variation_source_option_ids = set()
        ranked_required_variation_source_option_ids = set(
            required_ranked_source_evaluation_option_ids(
                request.finite_variation_contract
            )
        )
        required_variation_source_counts = dict(
            finite_variation_source_minimum_counts(request.finite_variation_contract)
        )
        contract_operator_by_option = finite_variation_operator_by_option(
            request.finite_variation_contract
        )
        operator_by_option = {
            value: contract_operator_by_option[value] for value in option_ids
        }
    if not set(operator_by_option.values()).issubset(operator_targets):
        raise ValueError("contextual contract omits a finite variation operator")
    required_variation_source_option_ids = (
        legacy_required_variation_source_option_ids
        | ranked_required_variation_source_option_ids
    )
    ordered_groups = {
        (source_id, operator_id): tuple(
            value
            for value in option_ids
            if source_by_option[value] == source_id
            and operator_by_option[value] == operator_id
        )
        for source_id in source_arm_ids
        for operator_id in operator_arm_ids
    }
    family_by_option = {
        value.option_id: value.family
        for value in request.finite_variation_contract.options
    }
    contract_identity_index = validated_finite_variation_identity_index(
        request.finite_variation_contract
    )
    identity_by_option = dict(
        zip(
            contract_identity_index.option_ids,
            contract_identity_index.option_identity_sha256s,
            strict=True,
        )
    )
    changed_paths_by_option = parent_relative_changed_paths_by_option(
        request.finite_variation_contract
    )
    single_path_option_ids = {
        option_id
        for option_id, paths in changed_paths_by_option.items()
        if len(paths) == 1
    }
    if len(single_path_option_ids.intersection(option_ids)) < (
        contract.minimum_single_path_interventions
    ):
        raise ValueError(
            "contextual universe omitted its single-path intervention floor"
        )
    disjoint_parent_patch_pairs = (
        set()
        if not (
            request.require_pairwise_disjoint_parent_patches
            or contract.minimum_disjoint_parent_patch_pairs
        )
        else {
            frozenset(value)
            for value in pairwise_disjoint_parent_patch_pairs(
                request.finite_variation_contract,
                option_ids,
            )
        }
    )
    if len(disjoint_parent_patch_pairs) < (
        contract.minimum_disjoint_parent_patch_pairs
    ):
        raise ValueError(
            "contextual universe omitted its disjoint parent-pair floor"
        )
    disjoint_degree = {
        option_id: sum(
            option_id in pair for pair in disjoint_parent_patch_pairs
        )
        for option_id in option_ids
    }
    option_priority = {value: index for index, value in enumerate(option_ids)}
    ordered_groups = {
        key: tuple(
            sorted(
                values,
                key=lambda option_id: (
                    option_id not in required_variation_source_option_ids,
                    option_id not in single_path_option_ids,
                    -disjoint_degree[option_id],
                    option_priority[option_id],
                ),
            )
        )
        for key, values in ordered_groups.items()
    }
    if not required_variation_source_option_ids.issubset(option_ids):
        raise ValueError("contextual universe omitted a required variation source")
    if any(
        sum(source_by_option[value] == source_id for value in option_ids) < minimum
        for source_id, minimum in required_variation_source_counts.items()
    ):
        raise ValueError("contextual universe omitted a required variation source")
    if request.memory_dose_contract is not None:
        dose = request.memory_dose_contract

        def support_count(option_id: str) -> int:
            return sum(
                support.supports(option_id, identity_by_option[option_id])
                for support in dose.card_supports
            )

        # Card-compatible actions are feasibility pivots, not quality hints.
        # Put them first inside their already-fixed source/operator stratum so
        # the exact search does not enumerate irrelevant combinations before
        # discovering a required-card cover.
        ordered_groups = {
            key: tuple(
                sorted(
                    values,
                    key=lambda option_id: (
                        option_id not in required_variation_source_option_ids,
                        -support_count(option_id),
                        option_id not in single_path_option_ids,
                        -disjoint_degree[option_id],
                        option_priority[option_id],
                    ),
                )
            )
            for key, values in ordered_groups.items()
        }
    allowed_pairs = (
        None
        if not request.require_pairwise_disjoint_parent_patches
        else disjoint_parent_patch_pairs
    )
    dose_witness_cache: dict[
        tuple[str, ...], MemoryDoseAttributionFeasibilityWitness | None
    ] = {}

    def memory_dose_witness(
        subset: tuple[str, ...],
    ) -> MemoryDoseAttributionFeasibilityWitness | None:
        dose = request.memory_dose_contract
        if dose is None:
            return None
        key = tuple(sorted(subset))
        if key in dose_witness_cache:
            return dose_witness_cache[key]
        witness = find_memory_dose_attribution_feasibility_witness(
            dose,
            stage=PortfolioMemoryDoseStage.EVALUATED_PORTFOLIO,
            member_option_identities=tuple(
                (option_id, identity_by_option[option_id]) for option_id in key
            ),
        )
        if witness is None:
            dose_witness_cache[key] = None
            return None
        # Evaluated members retain their exact card attribution in the K8
        # proposal.  An evaluated witness that already exceeds the proposed
        # upper bound cannot be extended into a passing proposal slate.
        if witness.supported_member_count > dose.proposed_supported_member_bounds[1]:
            dose_witness_cache[key] = None
            return None
        dose_witness_cache[key] = witness
        return witness

    def structurally_feasible(subset: tuple[str, ...]) -> bool:
        """Evaluate one K4 against indexes validated once for this search."""

        if not required_variation_source_option_ids.issubset(subset):
            return False
        if any(
            sum(source_by_option[value] == source_id for value in subset) < minimum
            for source_id, minimum in required_variation_source_counts.items()
        ):
            return False
        families = tuple(family_by_option[value] for value in subset)
        if (
            request.min_distinct_families is not None
            and len(set(families)) < request.min_distinct_families
        ):
            return False
        if any(
            not minimum <= families.count(family) <= maximum
            for family, minimum, maximum in required_evaluation_family_bounds
        ):
            return False
        if sum(value in single_path_option_ids for value in subset) < (
            contract.minimum_single_path_interventions
        ):
            return False
        if sum(
            frozenset((left, right)) in disjoint_parent_patch_pairs
            for index, left in enumerate(subset)
            for right in subset[index + 1 :]
        ) < contract.minimum_disjoint_parent_patch_pairs:
            return False
        structure_passes = allowed_pairs is None or all(
            frozenset((left, right)) in allowed_pairs
            for index, left in enumerate(subset)
            for right in subset[index + 1 :]
        )
        if not structure_passes:
            return False
        return (
            request.memory_dose_contract is None
            or memory_dose_witness(subset) is not None
        )

    priority = {value: index for index, value in enumerate(option_ids)}
    family_maximum = {
        family: maximum for family, _, maximum in required_evaluation_family_bounds
    }

    def bounded_compositions(
        total: int,
        maxima: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        rows: list[tuple[int, ...]] = []

        def visit(index: int, remaining: int, prefix: tuple[int, ...]) -> None:
            if index == len(maxima) - 1:
                value = remaining
                if 0 <= value <= maxima[index]:
                    rows.append((*prefix, value))
                return
            minimum = max(0, remaining - sum(maxima[index + 1 :]))
            maximum = min(maxima[index], remaining)
            for value in range(minimum, maximum + 1):
                visit(index + 1, remaining - value, (*prefix, value))

        visit(0, total, ())
        return tuple(rows)

    def contingency_tables(
        row_counts: tuple[int, ...],
        column_counts: tuple[int, ...],
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        if sum(row_counts) != sum(column_counts):
            raise ValueError("contextual source/operator marginals differ in capacity")
        tables: list[tuple[tuple[int, ...], ...]] = []

        def visit(
            row_index: int,
            remaining_columns: tuple[int, ...],
            rows: tuple[tuple[int, ...], ...],
        ) -> None:
            if row_index == len(row_counts) - 1:
                if sum(remaining_columns) == row_counts[row_index]:
                    tables.append((*rows, remaining_columns))
                return
            for row in bounded_compositions(
                row_counts[row_index],
                remaining_columns,
            ):
                visit(
                    row_index + 1,
                    tuple(
                        remaining - used
                        for remaining, used in zip(
                            remaining_columns,
                            row,
                            strict=True,
                        )
                    ),
                    (*rows, row),
                )

        visit(0, column_counts, ())
        return tuple(tables)

    def first_witness(
        *,
        source_counts: tuple[int, ...],
        operator_counts: tuple[int, ...],
    ) -> tuple[tuple[str, ...], MemoryDoseAttributionFeasibilityWitness | None] | None:
        for table in contingency_tables(source_counts, operator_counts):
            group_rows = tuple(
                row
                for source_index, source_id in enumerate(source_arm_ids)
                for operator_index, operator_id in enumerate(operator_arm_ids)
                for row in (
                    (
                        ordered_groups[(source_id, operator_id)],
                        table[source_index][operator_index],
                    ),
                )
            )
            strata = tuple(
                (
                    index,
                    group,
                    count,
                    {family_by_option[value] for value in group},
                )
                for index, (group, count) in enumerate(group_rows)
                if count
            )
            if any(count > len(group) for _, group, count, _ in strata):
                continue
            # Constrained-family strata first makes impossible tables cheap to
            # reject while retaining an exact deterministic order.
            ordered_strata = tuple(
                sorted(strata, key=lambda value: (len(value[3]), value[0]))
            )

            def search(
                stratum_index: int,
                selected: tuple[str, ...],
            ) -> (
                tuple[tuple[str, ...], MemoryDoseAttributionFeasibilityWitness | None]
                | None
            ):
                if stratum_index == len(ordered_strata):
                    subset = tuple(sorted(selected, key=priority.__getitem__))
                    if not structurally_feasible(subset):
                        return None
                    return subset, memory_dose_witness(subset)
                _, group, count, _ = ordered_strata[stratum_index]
                remaining_strata = ordered_strata[stratum_index + 1 :]
                for choice in combinations(group, count):
                    combined = (*selected, *choice)
                    if allowed_pairs is not None and any(
                        frozenset((left, right)) not in allowed_pairs
                        for index, left in enumerate(combined)
                        for right in combined[index + 1 :]
                    ):
                        continue
                    families = tuple(family_by_option[value] for value in combined)
                    if any(
                        families.count(family) > maximum
                        for family, maximum in family_maximum.items()
                    ):
                        continue
                    if request.min_distinct_families == request.portfolio_size and len(
                        set(families)
                    ) != len(families):
                        continue
                    if request.min_distinct_families is not None:
                        possible_new = sum(
                            min(
                                remaining_count,
                                len(remaining_families.difference(families)),
                            )
                            for _, _, remaining_count, remaining_families in remaining_strata
                        )
                        if (
                            len(set(families)) + possible_new
                            < request.min_distinct_families
                        ):
                            continue
                    witness = search(stratum_index + 1, combined)
                    if witness is not None:
                        return witness
                return None

            witness = search(0, ())
            if witness is not None:
                return witness
        return None

    requested_source_counts = tuple(source_targets[value] for value in source_arm_ids)
    requested_operator_counts = tuple(
        operator_targets[value] for value in operator_arm_ids
    )

    source_count_vectors = bounded_compositions(
        request.portfolio_size,
        tuple(request.portfolio_size for _ in source_arm_ids),
    )
    operator_count_vectors = bounded_compositions(
        request.portfolio_size,
        tuple(request.portfolio_size for _ in operator_arm_ids),
    )

    candidate_marginals = sorted(
        (
            (source_counts, operator_counts)
            for source_counts in source_count_vectors
            for operator_counts in operator_count_vectors
        ),
        key=lambda value: (
            sum(
                abs(observed - requested)
                for observed, requested in zip(
                    value[0], requested_source_counts, strict=True
                )
            )
            + 2
            * sum(
                abs(observed - requested)
                for observed, requested in zip(
                    value[1], requested_operator_counts, strict=True
                )
            ),
            sum(
                abs(observed - requested)
                for observed, requested in zip(
                    value[0], requested_source_counts, strict=True
                )
            ),
            2
            * sum(
                abs(observed - requested)
                for observed, requested in zip(
                    value[1], requested_operator_counts, strict=True
                )
            ),
            value[0],
            value[1],
        ),
    )
    for realized_sources, realized_operators in candidate_marginals:
        resolved = first_witness(
            source_counts=realized_sources,
            operator_counts=realized_operators,
        )
        if resolved is None:
            continue
        witness, dose_witness = resolved
        return ContextualAllocationFeasibilityProjectionReceipt(
            contract_sha256=contract.contract_sha256,
            requested_source_target_counts=requested_source,
            requested_operator_target_counts=requested_operator,
            realized_source_target_counts=tuple(
                zip(source_arm_ids, realized_sources, strict=True)
            ),
            realized_operator_target_counts=tuple(
                zip(operator_arm_ids, realized_operators, strict=True)
            ),
            evaluation_option_ids=tuple(sorted(witness)),
            minimum_single_path_interventions=(
                contract.minimum_single_path_interventions
            ),
            realized_single_path_interventions=sum(
                value in single_path_option_ids for value in witness
            ),
            minimum_disjoint_parent_patch_pairs=(
                contract.minimum_disjoint_parent_patch_pairs
            ),
            realized_disjoint_parent_patch_pairs=sum(
                frozenset((left, right)) in disjoint_parent_patch_pairs
                for index, left in enumerate(witness)
                for right in witness[index + 1 :]
            ),
            memory_dose_feasibility_witness=dose_witness,
        )
    dose_diagnostic = None
    if request.memory_dose_contract is not None:
        dose = request.memory_dose_contract
        dose_diagnostic = {
            "assigned_card_count": len(dose.assigned_card_keys),
            "evaluated_supported_member_bounds": list(
                dose.evaluated_supported_member_bounds
            ),
            "minimum_unattributed_evaluated_members": (
                dose.minimum_unattributed_evaluated_members
            ),
            "supportable_option_count": sum(
                any(
                    support.supports(option_id, identity_by_option[option_id])
                    for support in dose.card_supports
                )
                for option_id in option_ids
            ),
        }
    raise ValueError(
        "no jointly feasible contextual allocation: "
        + json.dumps(
            {
                "option_count": len(option_ids),
                "requested_source_target_counts": list(requested_source),
                "requested_operator_target_counts": list(requested_operator),
                "required_ranked_source_option_ids": sorted(
                    ranked_required_variation_source_option_ids
                ),
                "required_legacy_source_option_ids": sorted(
                    legacy_required_variation_source_option_ids
                ),
                "minimum_single_path_interventions": (
                    contract.minimum_single_path_interventions
                ),
                "single_path_option_count": len(
                    single_path_option_ids.intersection(option_ids)
                ),
                "minimum_disjoint_parent_patch_pairs": (
                    contract.minimum_disjoint_parent_patch_pairs
                ),
                "available_disjoint_parent_patch_pairs": len(
                    disjoint_parent_patch_pairs
                ),
                "option_counts_by_source_operator": {
                    f"{source_id}|{operator_id}": len(
                        ordered_groups[(source_id, operator_id)]
                    )
                    for source_id in source_arm_ids
                    for operator_id in operator_arm_ids
                },
                "single_path_counts_by_source_operator": {
                    f"{source_id}|{operator_id}": sum(
                        value in single_path_option_ids
                        for value in ordered_groups[(source_id, operator_id)]
                    )
                    for source_id in source_arm_ids
                    for operator_id in operator_arm_ids
                },
                "minimum_distinct_families": request.min_distinct_families,
                "required_family_bounds": [
                    list(value) for value in required_evaluation_family_bounds
                ],
                "pairwise_disjoint_parent_patches": (
                    request.require_pairwise_disjoint_parent_patches
                ),
                "memory_dose": dose_diagnostic,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _dose_assignment_for_reconciled_slate(
    request: PortfolioSelectionRequest,
    option_ids: tuple[str, ...],
    original_by_option: dict[str, CalibratedPortfolioModelMember],
    *,
    required_evaluation_family_bounds: tuple[tuple[str, int, int], ...],
    required_option_ids: tuple[str, ...] = (),
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Solve exact card attribution and K-evaluation feasibility on one K8."""

    structural_witness = _first_evaluation_witness(
        request,
        option_ids,
        required_evaluation_family_bounds=required_evaluation_family_bounds,
        required_option_ids=required_option_ids,
    )
    if structural_witness is None:
        raise ValueError("reconciled slate has no structurally feasible evaluation")
    dose = request.memory_dose_contract
    if dose is None:
        request_card_keys = {value.card_key for value in request.cards}
        default_card_keys = tuple(sorted(request_card_keys))[:1]
        assignment: dict[str, tuple[str, ...]] = {}
        for option_id in option_ids:
            retained = tuple(
                sorted(
                    set(
                        ()
                        if option_id not in original_by_option
                        else original_by_option[option_id].supporting_card_keys
                    ).intersection(request_card_keys)
                )
            )
            assignment[option_id] = (
                retained
                if retained or not request.require_supporting_cards
                else default_card_keys
            )
        administered = {card for cards in assignment.values() for card in cards}
        for index, card_key in enumerate(
            sorted(
                set(binding_card for binding_card in request_card_keys).difference(
                    administered
                )
            )
        ):
            option_id = option_ids[index % len(option_ids)]
            assignment[option_id] = tuple(sorted({*assignment[option_id], card_key}))
        return assignment, structural_witness

    option_identity = {
        value.option_id: value.identity_sha256
        for value in request.finite_variation_contract.options
    }
    supports_by_option: dict[str, tuple[str, ...]] = {}
    for option_id in option_ids:
        supports_by_option[option_id] = tuple(
            support.card_key
            for support in dose.card_supports
            if support.supports(option_id, option_identity[option_id])
        )
    choices_by_option: list[tuple[tuple[str, ...], ...]] = []
    for option_id in option_ids:
        compatible = supports_by_option[option_id]
        preferred = tuple(
            sorted(
                set(
                    ()
                    if option_id not in original_by_option
                    else original_by_option[option_id].supporting_card_keys
                ).intersection(compatible)
            )
        )
        choices: list[tuple[str, ...]] = []
        if preferred and len(preferred) <= dose.maximum_cards_per_member:
            choices.append(preferred)
        choices.append(())
        for count in range(1, min(len(compatible), dose.maximum_cards_per_member) + 1):
            choices.extend(tuple(value) for value in combinations(compatible, count))
        choices_by_option.append(tuple(dict.fromkeys(choices)))

    lower, upper = dose.proposed_supported_member_bounds
    feasible_evaluation_subsets = _feasible_evaluation_subsets(
        request,
        option_ids,
        required_evaluation_family_bounds=required_evaluation_family_bounds,
        required_option_ids=required_option_ids,
    )
    if not feasible_evaluation_subsets:
        raise ValueError("reconciled slate lost its feasible evaluation subset")
    selected_assignment: dict[str, tuple[str, ...]] | None = None
    selected_witness: tuple[str, ...] | None = None

    def search(
        index: int,
        assignment: dict[str, tuple[str, ...]],
        supported_count: int,
    ) -> bool:
        nonlocal selected_assignment, selected_witness
        remaining = len(option_ids) - index
        if supported_count > upper or supported_count + remaining < lower:
            return False
        if index < len(option_ids):
            option_id = option_ids[index]
            for cards in choices_by_option[index]:
                assignment[option_id] = cards
                if search(
                    index + 1,
                    assignment,
                    supported_count + bool(cards),
                ):
                    return True
            assignment.pop(option_id, None)
            return False

        proposed_members = tuple(
            PortfolioMemoryDoseMember(
                rank=rank,
                option_id=option_id,
                option_identity_sha256=option_identity[option_id],
                supporting_card_keys=assignment[option_id],
            )
            for rank, option_id in enumerate(option_ids, start=1)
        )
        proposal_assessment = assess_proposed_portfolio_memory_dose(
            dose,
            proposed_members,
        )
        if not proposal_assessment.passed:
            return False
        for witness in feasible_evaluation_subsets:
            evaluated_members = tuple(
                PortfolioMemoryDoseMember(
                    rank=rank,
                    option_id=option_id,
                    option_identity_sha256=option_identity[option_id],
                    supporting_card_keys=assignment[option_id],
                )
                for rank, option_id in enumerate(witness, start=1)
            )
            evaluated = assess_evaluated_portfolio_memory_dose(
                dose,
                evaluated_members,
                proposal_assessment=proposal_assessment,
            )
            if evaluated.passed:
                selected_assignment = dict(assignment)
                selected_witness = witness
                return True
        return False

    if not search(0, {}, 0):
        raise ValueError(
            "reconciled slate has no jointly feasible memory-dose evaluation"
        )
    assert selected_assignment is not None and selected_witness is not None
    return selected_assignment, selected_witness


def _engine_inserted_model_member(
    request: PortfolioSelectionRequest,
    option_id: str,
    *,
    model_rank: int,
    supporting_card_keys: tuple[str, ...],
) -> CalibratedPortfolioModelMember:
    return CalibratedPortfolioModelMember(
        model_rank=model_rank,
        option_id=option_id,
        supporting_card_keys=supporting_card_keys,
        effect_predictions=tuple(
            CalibratedPortfolioModelPrediction(
                metric_id=metric_id,
                direction=MetricEffectDirection.UNKNOWN,
                confidence=ForecastConfidenceBin.UNKNOWN,
            )
            for metric_id in request.required_metric_ids
        ),
        role_proposal=SlateRoleProposal.COVERAGE,
        design_rationale=(
            "Trusted engine reconciliation inserted this legal finite action "
            "to satisfy cross-member feasibility without objective access."
        ),
    )


def _reconcile_semantic_members(
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
    model_members: tuple[CalibratedPortfolioModelMember, ...],
    *,
    required_evaluation_family_bounds: tuple[tuple[str, int, int], ...],
    original_model_proposal_sha256: str,
    minimum_intervention_projection: bool = False,
    evidence_calibrated_source_mix: bool = False,
    contextual_search_allocation: bool = False,
    acquisition_certification_reference_option_ids: tuple[str, ...] = (),
) -> tuple[
    tuple[CalibratedPortfolioModelMember, ...],
    SemanticSlateReconciliationReceipt,
]:
    """Reconcile model-local semantic suggestions into an exact feasible K8."""

    if type(minimum_intervention_projection) is not bool:
        raise TypeError("minimum_intervention_projection must be an exact bool")
    if type(evidence_calibrated_source_mix) is not bool:
        raise TypeError("evidence_calibrated_source_mix must be an exact bool")
    if type(contextual_search_allocation) is not bool:
        raise TypeError("contextual_search_allocation must be an exact bool")
    if (
        type(acquisition_certification_reference_option_ids) is not tuple
        or any(
            type(value) is not str or not value
            for value in acquisition_certification_reference_option_ids
        )
        or acquisition_certification_reference_option_ids
        != tuple(sorted(set(acquisition_certification_reference_option_ids)))
    ):
        raise ValueError(
            "acquisition certification reference IDs must be canonical and unique"
        )
    if acquisition_certification_reference_option_ids and len(
        acquisition_certification_reference_option_ids
    ) != request.portfolio_size:
        raise ValueError("acquisition reference must exactly fill the portfolio")
    if acquisition_certification_reference_option_ids and (
        minimum_intervention_projection
        or evidence_calibrated_source_mix
        or contextual_search_allocation
    ):
        raise ValueError("acquisition certification cannot mix allocation authorities")
    if contextual_search_allocation and not evidence_calibrated_source_mix:
        raise ValueError(
            "contextual allocation requires evidence-calibrated source mix"
        )
    if evidence_calibrated_source_mix and not minimum_intervention_projection:
        raise ValueError("evidence-calibrated source mix requires minimum intervention")

    selectable = (
        tuple(value.option_id for value in request.finite_variation_contract.options)
        if binding.common_candidate_pool is None
        else binding.common_candidate_pool.option_ids
    )
    selectable_set = set(selectable)
    if any(value.option_id not in selectable_set for value in model_members):
        raise ValueError("semantic proposal escapes the selectable finite universe")
    original_by_option: dict[str, CalibratedPortfolioModelMember] = {}
    for member in model_members:
        original_by_option.setdefault(member.option_id, member)
    priority = tuple(dict.fromkeys((*original_by_option.keys(), *selectable)))
    priority_index = {option_id: index for index, option_id in enumerate(priority)}
    required_support_order = (
        ()
        if binding.proposal_support is None
        else binding.proposal_support.required_option_ids
    )
    required_support = set(required_support_order)
    required_evaluation_source_order = required_source_evaluation_option_ids(
        request.finite_variation_contract
    )
    required_evaluation_sources = set(required_evaluation_source_order)
    if not required_evaluation_sources.issubset(selectable_set):
        raise ValueError("common candidate pool omitted a required evaluation source")
    if not set(acquisition_certification_reference_option_ids).issubset(
        selectable_set
    ):
        raise ValueError("selectable pool omitted an acquisition reference option")
    protected_allocation_option_ids: tuple[str, ...] = ()
    contextual_allocation_option_ids: tuple[str, ...] = ()
    contextual_allocation_projection = None
    hierarchy = _hierarchical_composition_shape(
        request.finite_variation_contract,
        allowed_option_ids=selectable,
    )
    composite_ids = set(() if hierarchy is None else hierarchy.composite_option_ids)
    target_composites = (
        None if hierarchy is None else hierarchy.required_composite_proposals
    )
    pool_witness = None
    if acquisition_certification_reference_option_ids:
        pool_witness = _first_evaluation_witness(
            request,
            priority,
            required_evaluation_family_bounds=required_evaluation_family_bounds,
            required_option_ids=acquisition_certification_reference_option_ids,
        )
        if pool_witness is None or set(pool_witness) != set(
            acquisition_certification_reference_option_ids
        ):
            raise ValueError(
                "complete numerical acquisition reference is not evaluator-feasible"
            )
    elif contextual_search_allocation:
        if binding.common_candidate_pool is None:
            raise ValueError(
                "contextual search allocation requires a common candidate pool"
            )
        if binding.contextual_allocation is None:
            raise ValueError(
                "contextual search profile omitted its prospective contract"
            )
        contextual_allocation_projection = (
            _contextual_allocation_feasibility_projection(
                request,
                priority,
                original_by_option=original_by_option,
                composite_option_ids=composite_ids,
                binding=binding,
                required_evaluation_family_bounds=(required_evaluation_family_bounds),
            )
        )
        if contextual_allocation_projection is None:
            raise ValueError(
                "common candidate pool has no feasible contextual allocation recourse"
            )
        pool_witness = contextual_allocation_projection.evaluation_option_ids
        contextual_allocation_option_ids = tuple(sorted(pool_witness))
        if target_composites is not None:
            target_composites = max(
                target_composites,
                dict(contextual_allocation_projection.realized_operator_target_counts)[
                    "composite"
                ],
            )
    elif evidence_calibrated_source_mix:
        if binding.common_candidate_pool is None:
            raise ValueError(
                "evidence-calibrated source mix requires a common candidate pool"
            )
        if binding.context.wave_index == 1:
            source_candidates = tuple(
                dict.fromkeys(
                    (
                        *binding.common_candidate_pool.feasibility_witness_option_ids,
                        *binding.common_candidate_pool.option_ids,
                    )
                )
            )
            for candidate in source_candidates:
                if candidate in original_by_option:
                    continue
                candidate_witness = _first_evaluation_witness(
                    request,
                    priority,
                    required_evaluation_family_bounds=(
                        required_evaluation_family_bounds
                    ),
                    preferred_option_ids=tuple(original_by_option),
                    required_option_ids=(candidate,),
                )
                if candidate_witness is not None:
                    protected_allocation_option_ids = (candidate,)
                    pool_witness = candidate_witness
                    break
            if not protected_allocation_option_ids:
                raise ValueError(
                    "common candidate pool has no protected global-source witness"
                )
    if pool_witness is None:
        pool_witness = _first_evaluation_witness(
            request,
            priority,
            required_evaluation_family_bounds=required_evaluation_family_bounds,
            preferred_option_ids=(
                tuple(original_by_option) if minimum_intervention_projection else ()
            ),
        )
    if pool_witness is None:
        raise ValueError("selectable universe lost its feasible evaluation witness")
    feasibility_ids = set(pool_witness)
    memory_ids: set[str] = set()
    # A contextual witness already satisfies semantic per-source exposure.
    # Do not crowd its K8 reconciliation slate with unrelated deterministic
    # representatives chosen only to keep bounded common pools source-aware.
    mandatory = set(feasibility_ids)
    if not contextual_search_allocation:
        mandatory.update(required_evaluation_sources)
    if not evidence_calibrated_source_mix:
        mandatory.update(required_support)

    dose = request.memory_dose_contract
    if dose is not None:
        option_identity = {
            value.option_id: value.identity_sha256
            for value in request.finite_variation_contract.options
        }

        def can_add(option_id: str) -> bool:
            return len(mandatory) < CALIBRATED_PORTFOLIO_PROPOSAL_SIZE and (
                target_composites is None
                or option_id not in composite_ids
                or sum(value in composite_ids for value in mandatory)
                < target_composites
            )

        for support in dose.card_supports:
            if any(
                value in mandatory and support.supports(value, option_identity[value])
                for value in selectable
            ):
                continue
            candidate = next(
                (
                    value
                    for value in priority
                    if can_add(value)
                    and support.supports(value, option_identity[value])
                ),
                None,
            )
            if candidate is None:
                raise ValueError("memory card has no selectable reconciliation action")
            mandatory.add(candidate)
            memory_ids.add(candidate)
        lower_supported, _ = dose.proposed_supported_member_bounds
        compatible_mandatory = {
            option_id
            for option_id in mandatory
            if any(
                support.supports(option_id, option_identity[option_id])
                for support in dose.card_supports
            )
        }
        for candidate in priority:
            if len(compatible_mandatory) >= lower_supported:
                break
            if candidate in mandatory or not can_add(candidate):
                continue
            if any(
                support.supports(candidate, option_identity[candidate])
                for support in dose.card_supports
            ):
                mandatory.add(candidate)
                compatible_mandatory.add(candidate)
                memory_ids.add(candidate)

    relaxed_proposal_support: set[str] = set()
    if evidence_calibrated_source_mix:
        for option_id in required_support_order:
            if option_id in mandatory:
                continue
            composition_capacity = (
                target_composites is None
                or option_id not in composite_ids
                or sum(value in composite_ids for value in mandatory)
                < target_composites
            )
            if (
                len(mandatory) < CALIBRATED_PORTFOLIO_PROPOSAL_SIZE
                and composition_capacity
            ):
                mandatory.add(option_id)
            else:
                relaxed_proposal_support.add(option_id)

    if len(mandatory) > CALIBRATED_PORTFOLIO_PROPOSAL_SIZE:
        raise ValueError("cross-member mandatory set exceeds the proposal size")
    mandatory_composites = sum(value in composite_ids for value in mandatory)
    composition_capacity_projection = None
    if target_composites is not None and contextual_search_allocation:
        composition_capacity_projection = project_exact_k_binary_composition(
            proposal_size=CALIBRATED_PORTFOLIO_PROPOSAL_SIZE,
            preferred_composite_count=target_composites,
            mandatory_atomic_count=len(mandatory) - mandatory_composites,
            mandatory_composite_count=mandatory_composites,
            selectable_atomic_count=(
                len(selectable) - sum(value in composite_ids for value in selectable)
            ),
            selectable_composite_count=sum(
                value in composite_ids for value in selectable
            ),
        )
        target_composites = composition_capacity_projection.effective_composite_count
    if target_composites is not None and mandatory_composites > target_composites:
        raise ValueError("mandatory set exceeds the composite exposure target")

    selected = set(mandatory)
    if target_composites is None:
        for option_id in priority:
            if len(selected) == CALIBRATED_PORTFOLIO_PROPOSAL_SIZE:
                break
            selected.add(option_id)
    else:
        composite_needed = target_composites - mandatory_composites
        atomic_needed = (
            CALIBRATED_PORTFOLIO_PROPOSAL_SIZE - len(selected) - composite_needed
        )
        for option_id in priority:
            if option_id in selected:
                continue
            if option_id in composite_ids and composite_needed:
                selected.add(option_id)
                composite_needed -= 1
            elif option_id not in composite_ids and atomic_needed:
                selected.add(option_id)
                atomic_needed -= 1
            if composite_needed == 0 and atomic_needed == 0:
                break
        if composite_needed or atomic_needed:
            raise ValueError("selectable universe cannot realize the K8 composition")
    if len(selected) != CALIBRATED_PORTFOLIO_PROPOSAL_SIZE:
        raise ValueError("selectable universe cannot refill exactly eight members")
    if (
        target_composites is not None
        and sum(value in composite_ids for value in selected) != target_composites
    ):
        raise ValueError("reconciled slate differs from its effective composition")
    selected_ids = tuple(sorted(selected, key=priority_index.__getitem__))
    card_assignment, evaluation_witness = _dose_assignment_for_reconciled_slate(
        request,
        selected_ids,
        original_by_option,
        required_evaluation_family_bounds=required_evaluation_family_bounds,
        required_option_ids=tuple(
            sorted(
                {
                    *protected_allocation_option_ids,
                    *contextual_allocation_option_ids,
                    *acquisition_certification_reference_option_ids,
                    *(
                        ()
                        if contextual_search_allocation
                        else required_evaluation_sources
                    ),
                }
            )
        ),
    )
    deferred_proposal_support_option_ids = tuple(
        sorted(relaxed_proposal_support.difference(selected))
    )

    reconciled: list[CalibratedPortfolioModelMember] = []
    receipts: list[SemanticSlateReconciledMember] = []
    for rank, option_id in enumerate(selected_ids, start=1):
        original = original_by_option.get(option_id)
        if original is not None:
            member = replace(
                original,
                model_rank=rank,
                supporting_card_keys=card_assignment[option_id],
            )
            origin = SemanticSlateMemberOrigin.MODEL
        else:
            member = _engine_inserted_model_member(
                request,
                option_id,
                model_rank=rank,
                supporting_card_keys=card_assignment[option_id],
            )
            origin = (
                SemanticSlateMemberOrigin.ENGINE_CONTEXTUAL_ALLOCATION
                if option_id in contextual_allocation_option_ids
                else SemanticSlateMemberOrigin.ENGINE_GLOBAL_COVERAGE
                if option_id in required_evaluation_sources
                else SemanticSlateMemberOrigin.ENGINE_GLOBAL_COVERAGE
                if option_id in protected_allocation_option_ids
                else SemanticSlateMemberOrigin.ENGINE_REQUIRED_SUPPORT
                if option_id in required_support
                else SemanticSlateMemberOrigin.ENGINE_MEMORY_DOSE
                if option_id in memory_ids
                else SemanticSlateMemberOrigin.ENGINE_FEASIBILITY
                if option_id in feasibility_ids
                else SemanticSlateMemberOrigin.ENGINE_REFILL
            )
        reason_values: set[str] = set()
        if original is not None:
            reason_values.add("model_semantic_preference")
        if option_id in required_support:
            reason_values.add("required_proposal_support")
        if option_id in required_evaluation_sources:
            reason_values.add("required_evaluation_source")
        if option_id in protected_allocation_option_ids:
            reason_values.add("protected_task_keyed_global_source")
        if option_id in contextual_allocation_option_ids:
            reason_values.add("contextual_search_allocation")
        if option_id in feasibility_ids:
            reason_values.add("evaluation_feasibility_witness")
        if option_id in memory_ids or card_assignment[option_id] != (
            () if original is None else original.supporting_card_keys
        ):
            reason_values.add("memory_dose_feasibility")
        if (
            original is None
            and option_id not in required_support
            and option_id not in required_evaluation_sources
            and option_id not in feasibility_ids
            and option_id not in memory_ids
        ):
            reason_values.add("deterministic_refill")
        reasons = tuple(sorted(reason_values))
        reconciled.append(member)
        receipts.append(
            SemanticSlateReconciledMember(
                reconciled_rank=rank,
                option_id=option_id,
                origin=origin,
                original_model_rank=(None if original is None else original.model_rank),
                original_supporting_card_keys=tuple(
                    sorted(() if original is None else original.supporting_card_keys)
                ),
                reconciled_supporting_card_keys=card_assignment[option_id],
                reasons=reasons,
            )
        )

    reconciled_record = _typed_proposal_record(
        request,
        binding.context,
        tuple(reconciled),
    )
    reconciled_sha256 = _proposal_sha256(reconciled_record)
    receipt = SemanticSlateReconciliationReceipt(
        original_model_proposal_sha256=original_model_proposal_sha256,
        reconciled_proposal_sha256=reconciled_sha256,
        duplicate_model_member_count=(len(model_members) - len(original_by_option)),
        evaluation_feasibility_witness=evaluation_witness,
        members=tuple(receipts),
        minimum_intervention_projection=minimum_intervention_projection,
        evidence_calibrated_source_mix=evidence_calibrated_source_mix,
        protected_allocation_option_ids=protected_allocation_option_ids,
        deferred_proposal_support_option_ids=(deferred_proposal_support_option_ids),
        contextual_search_allocation=contextual_search_allocation,
        contextual_allocation_contract_sha256=(
            None
            if binding.contextual_allocation is None
            else binding.contextual_allocation.contract_sha256
        ),
        contextual_allocation_option_ids=contextual_allocation_option_ids,
        contextual_allocation_projection=contextual_allocation_projection,
        composition_capacity_projection=composition_capacity_projection,
    )
    return tuple(reconciled), receipt


def _typed_proposal_record(
    request: PortfolioSelectionRequest,
    context: CalibratedPortfolioAllocationContext,
    members: tuple[CalibratedPortfolioModelMember, ...],
) -> dict[str, object]:
    hierarchy = _hierarchical_composition_shape(
        request.finite_variation_contract,
    )
    components = {} if hierarchy is None else hierarchy.components_by_composite
    return {
        "schema_version": 2,
        "request_sha256": request.request_sha256,
        "scope_sha256": context.scope.scope_sha256,
        "wave_index": context.wave_index,
        "parent_candidate_identity_sha256": (context.parent_candidate_identity_sha256),
        "members": [
            {
                "model_rank": rank,
                "option_id": member.option_id,
                **(
                    {}
                    if hierarchy is None
                    else {
                        "hierarchical_action": (
                            {
                                "action_kind": "compose_r2",
                                "component_option_ids": list(
                                    components[member.option_id]
                                ),
                            }
                            if member.option_id in components
                            else {"action_kind": "atomic"}
                        )
                    }
                ),
                "supporting_card_keys": list(member.supporting_card_keys),
                "effect_predictions": [
                    value.to_record() for value in member.effect_predictions
                ],
                "role_proposal": member.role_proposal.value,
                "design_rationale": member.design_rationale,
            }
            for rank, member in enumerate(members, start=1)
        ],
    }


def _build_calibrated_slate(
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
    output_members: tuple[CalibratedPortfolioModelMember, ...],
    *,
    proposal_sha256: str,
) -> tuple[CalibratedSlate, dict[str, CalibratedPortfolioModelMember]]:
    context = binding.context
    evidence_by_option = {value.option_id: value for value in binding.option_evidence}
    calibrated_members: list[CalibratedSlateMember] = []
    model_member_by_option: dict[str, CalibratedPortfolioModelMember] = {}
    for model_rank, member in enumerate(output_members, start=1):
        option = request.finite_variation_contract.resolve(member.option_id)
        evidence = evidence_by_option[option.option_id]
        if evidence.option_identity_sha256 != option.identity_sha256:
            raise ValueError("structural evidence belongs to a foreign option")
        predictions = tuple(
            ForecastPredictionReceipt(
                scope=context.scope,
                wave_index=context.wave_index,
                selector_decision_sha256=proposal_sha256,
                parent_candidate_identity_sha256=(
                    context.parent_candidate_identity_sha256
                ),
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                family=option.family,
                metric_id=prediction.metric_id,
                asserted_direction=MetricEffectDirection(prediction.direction),
                confidence=ForecastConfidenceBin(prediction.confidence),
            )
            for prediction in sorted(
                member.effect_predictions,
                key=lambda value: value.metric_id,
            )
        )
        calibrated_members.append(
            CalibratedSlateMember(
                model_rank=model_rank,
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                family=option.family,
                locus_key=evidence.locus_key,
                phenotype_identity_sha256=evidence.phenotype_identity_sha256,
                supporting_card_keys=tuple(sorted(member.supporting_card_keys)),
                role_proposal=SlateRoleProposal(member.role_proposal),
                rationale_sha256=hashlib.sha256(
                    member.design_rationale.encode("utf-8", errors="strict")
                ).hexdigest(),
                predictions=predictions,
                structural_evidence=evidence.structural_evidence,
            )
        )
        model_member_by_option[option.option_id] = member
    return (
        CalibratedSlate(
            scope=context.scope,
            wave_index=context.wave_index,
            selector_decision_sha256=proposal_sha256,
            parent_candidate_identity_sha256=(context.parent_candidate_identity_sha256),
            finite_contract_sha256=request.finite_variation_contract.identity_sha256,
            members=tuple(calibrated_members),
        ),
        model_member_by_option,
    )


def _allocate_slate(
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
    slate: CalibratedSlate,
    allocator: CalibratedPortfolioAllocator,
    *,
    required_option_ids: tuple[str, ...] = (),
) -> CalibratedPortfolioAllocationDecision:
    context = binding.context
    proposal_memory_dose_assessment = None
    if request.memory_dose_contract is not None:
        proposal_memory_dose_assessment = assess_proposed_portfolio_memory_dose(
            request.memory_dose_contract,
            tuple(
                PortfolioMemoryDoseMember(
                    rank=value.model_rank,
                    option_id=value.option_id,
                    option_identity_sha256=value.option_identity_sha256,
                    supporting_card_keys=value.supporting_card_keys,
                )
                for value in slate.members
            ),
        )
        require_passing_portfolio_memory_dose(proposal_memory_dose_assessment)
    allocation_request = SlateAllocationRequest(
        slate=slate,
        portfolio_size=request.portfolio_size,
        objectives=context.objectives,
        assigned_card_keys=context.assigned_card_keys,
        calibration_snapshot=context.calibration_snapshot,
        pairwise_disjoint_option_id_pairs=(
            pairwise_disjoint_parent_patch_pairs(
                request.finite_variation_contract,
                tuple(value.option_id for value in slate.members),
            )
            if request.require_pairwise_disjoint_parent_patches
            else None
        ),
        min_distinct_families=request.min_distinct_families,
        memory_dose_contract=request.memory_dose_contract,
        proposal_memory_dose_assessment=(proposal_memory_dose_assessment),
        required_option_ids=required_option_ids,
    )
    allocation = allocator.select(allocation_request)
    if type(allocator) is TraceCalibratedSlatePolicy:
        if type(allocation) is not SlateAllocationDecision:
            raise TypeError("four-role allocator returned a foreign decision")
    elif type(allocator) is ModelAnchoredCalibratedSlatePolicy:
        if type(allocation) is not ModelAnchoredSlateDecision:
            raise TypeError("model-anchored allocator returned a foreign decision")
    elif type(allocator) is StructuralPosteriorSlatePolicy:
        if type(allocation) is not StructuralPosteriorSlateDecision:
            raise TypeError(
                "structural-posterior allocator returned a foreign decision"
            )
    elif type(allocator) is OperatorStratifiedStructuralPosteriorSlatePolicy:
        if type(allocation) is not (OperatorStratifiedStructuralPosteriorSlateDecision):
            raise TypeError("operator-stratified allocator returned a foreign decision")
    elif type(allocator) is HorizonBoundedStructuralPosteriorSlatePolicy:
        if type(allocation) is not HorizonBoundedStructuralPosteriorSlateDecision:
            raise TypeError("horizon-bounded allocator returned a foreign decision")
    elif type(allocator) is FrontierProbeSlatePolicy:
        if type(allocation) is not FrontierProbeSlateDecision:
            raise TypeError("frontier-probe allocator returned a foreign decision")
    elif type(allocator) is FullSupportSlatePolicy:
        if type(allocation) is not SlateAllocationDecision:
            raise TypeError("full-support allocator returned a foreign decision")
    elif type(allocator) is TargetConditionedSlateAllocatorAdapter:
        if type(allocation) is not TargetConditionedSlateDecision:
            raise TypeError("target-conditioned allocator returned a foreign decision")
    elif type(allocator) is AcquisitionCertifiedSlatePolicy:
        if type(allocation) is not AcquisitionCertifiedSlateDecision:
            raise TypeError("acquisition-certified allocator returned a foreign decision")
    elif type(allocator) is RegretBoundedSlatePolicy:
        if type(allocation) is not RegretBoundedSlateDecision:
            raise TypeError("regret-bounded allocator returned a foreign decision")
    else:  # Defensive after profile/type validation.
        raise TypeError("allocator must be an exact supported policy")
    allocation.revalidate()
    return allocation


def _acquisition_certification_reference_option_ids(
    request: PortfolioSelectionRequest,
    allocator: CalibratedPortfolioAllocator,
) -> tuple[str, ...]:
    if type(allocator) not in {
        AcquisitionCertifiedSlatePolicy,
        RegretBoundedSlatePolicy,
    }:
        return ()
    context = allocator.context_provider.context_for(
        request.finite_variation_contract.identity_sha256
    )
    if type(context) is not AcquisitionCertifiedSlateContext:
        raise TypeError("acquisition context provider returned a foreign context")
    context.__post_init__()
    if context.finite_contract_sha256 != (
        request.finite_variation_contract.identity_sha256
    ):
        raise ValueError("acquisition context names a foreign finite contract")
    if len(context.reference_option_ids) != request.portfolio_size:
        raise ValueError("acquisition reference must exactly fill the portfolio")
    return context.reference_option_ids


def _audit_acquisition_certification_reference_option_ids(
    payload: dict[str, object],
    profile: _CalibratedPortfolioSelectionProfile,
) -> tuple[str, ...]:
    if not profile.acquisition_certified_residual:
        return ()
    allocation = payload.get("allocation")
    if type(allocation) is not dict:
        raise TypeError("acquisition-certified audit omitted its allocation")
    raw = allocation.get("reference_option_ids")
    if type(raw) is not list or any(type(value) is not str for value in raw):
        raise TypeError("acquisition reference IDs must remain an exact JSON array")
    reference = tuple(raw)
    if not reference or reference != tuple(sorted(set(reference))):
        raise ValueError("acquisition reference IDs are not canonical")
    return reference


def _required_allocation_option_ids(
    request: PortfolioSelectionRequest,
    profile: _CalibratedPortfolioSelectionProfile,
    reconciliation_receipt: SemanticSlateReconciliationReceipt | None,
) -> tuple[str, ...]:
    """Bind engine-required evaluator exposure to an exact selector profile."""

    reconciled = (
        ()
        if reconciliation_receipt is None
        else reconciliation_receipt.required_allocation_option_ids
    )
    source_floor = (
        required_source_evaluation_option_ids(request.finite_variation_contract)
        if profile.constraint_decoupled
        and (
            reconciliation_receipt is None
            or not (
                reconciliation_receipt.evidence_calibrated_source_mix
                or reconciliation_receipt.contextual_search_allocation
            )
        )
        else ()
    )
    required = tuple(sorted({*reconciled, *source_floor}))
    if len(required) > request.portfolio_size:
        raise ValueError("required evaluator exposure exceeds the portfolio width")
    return required


def _resolve_ranked_decision(
    request: PortfolioSelectionRequest,
    allocation: CalibratedPortfolioAllocationDecision,
    model_member_by_option: dict[str, CalibratedPortfolioModelMember],
    *,
    profile: _CalibratedPortfolioSelectionProfile,
) -> RankedPortfolioDecision:
    if profile is _FRONTIER_PROBE_PROFILE and (
        request.min_distinct_families is not None
        or request.require_pairwise_disjoint_parent_patches
    ):
        raise ValueError(
            "frontier-probe live evaluation requests must leave family and "
            "pairwise patch constraints to the downstream mating stage"
        )
    drafts: list[PortfolioMemberDraft] = []
    for selected in allocation.selected:
        original = model_member_by_option[selected.option_id]
        drafts.append(
            PortfolioMemberDraft(
                option_id=selected.option_id,
                supporting_card_keys=tuple(sorted(original.supporting_card_keys)),
                effect_predictions=tuple(
                    MetricEffectPrediction(
                        metric_id=prediction.metric_id,
                        direction=MetricEffectDirection(prediction.direction),
                    )
                    for prediction in sorted(
                        original.effect_predictions,
                        key=lambda value: value.metric_id,
                    )
                ),
                design_rationale=original.design_rationale,
            )
        )
    return resolve_ranked_portfolio_decision(
        request,
        tuple(drafts),
        policy_id=profile.policy_id,
        policy_version=profile.policy_version,
        policy_definition_sha256=profile.policy_definition_sha256,
        memory_dose_assessment=allocation.memory_dose_assessment,
    )


def _selected_predictions(
    slate: CalibratedSlate,
    allocation: CalibratedPortfolioAllocationDecision,
) -> tuple[ForecastPredictionReceipt, ...]:
    by_option = {value.option_id: value for value in slate.members}
    return tuple(
        prediction
        for selected in allocation.selected
        for prediction in by_option[selected.option_id].predictions
    )


def _resolved_decision_payload_key(
    profile: _CalibratedPortfolioSelectionProfile,
    request: PortfolioSelectionRequest,
) -> str:
    if profile is _FULL_SUPPORT_PROFILE:
        return "resolved_portfolio_decision"
    if request.portfolio_size == CALIBRATED_PORTFOLIO_EVALUATION_SIZE:
        # Preserve already published K4 evidence byte-for-byte.
        return "resolved_k4_decision"
    return "resolved_selected_decision"


def _evaluation_reachability_invariant_key(
    profile: _CalibratedPortfolioSelectionProfile,
    request: PortfolioSelectionRequest,
) -> str:
    if profile is _FULL_SUPPORT_PROFILE:
        return "entire_proposed_slate_reaches_evaluator"
    if request.portfolio_size == CALIBRATED_PORTFOLIO_EVALUATION_SIZE:
        # Preserve already published K4 evidence byte-for-byte.
        return "only_selected_k4_reaches_evaluator"
    return "only_selected_subset_reaches_evaluator"


def _audit_payload_record(
    *,
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
    proposal_record: dict[str, object],
    proposal_sha256: str,
    slate: CalibratedSlate,
    allocation: CalibratedPortfolioAllocationDecision,
    allocator: CalibratedPortfolioAllocator,
    profile: _CalibratedPortfolioSelectionProfile,
    decision: RankedPortfolioDecision,
    original_model_proposal_record: dict[str, object] | None = None,
    reconciliation_receipt: SemanticSlateReconciliationReceipt | None = None,
) -> dict[str, object]:
    if profile.constraint_decoupled != (
        original_model_proposal_record is not None
        and reconciliation_receipt is not None
    ):
        raise ValueError(
            "constraint-decoupled profile requires exact reconciliation evidence"
        )
    selected_predictions = _selected_predictions(slate, allocation)
    record: dict[str, object] = {
        "schema_version": _payload_schema_version(profile, binding, request),
        "event_type": profile.event_type,
        "policy_id": profile.policy_id,
        "policy_version": profile.policy_version,
        "policy_definition_sha256": profile.policy_definition_sha256,
        "prompt_definition_sha256": _prompt_definition_sha256_for_binding(
            request,
            binding,
            constraint_decoupled=profile.constraint_decoupled,
        ),
        "input_binding_sha256": binding.binding_sha256,
        "proposal_size": CALIBRATED_PORTFOLIO_PROPOSAL_SIZE,
        "evaluation_size": request.portfolio_size,
        "proposal_sha256": proposal_sha256,
        "original_k8_response": proposal_record,
        "calibrated_slate": slate.to_record(),
        "allocation": allocation.to_record(),
        "selected_prediction_receipt_sha256s": [
            value.receipt_sha256 for value in selected_predictions
        ],
        "invariants": {
            "one_low_level_call": True,
            "prior_only": allocation.prior_only,
            _evaluation_reachability_invariant_key(profile, request): True,
            "caller_instruction_rendered": False,
            "input_binding_frozen_before_call": True,
            "administered_card_keys": list(allocation.administered_card_keys),
            **(
                {}
                if not profile.constraint_decoupled
                else {
                    "model_owns_only_local_semantic_preferences": True,
                    "engine_owns_cross_member_constraints": True,
                }
            ),
        },
    }
    if reconciliation_receipt is not None:
        assert original_model_proposal_record is not None
        if reconciliation_receipt.original_model_proposal_sha256 != (
            _proposal_sha256(original_model_proposal_record)
        ):
            raise ValueError("reconciliation receipt differs from model proposal")
        if reconciliation_receipt.reconciled_proposal_sha256 != proposal_sha256:
            raise ValueError("reconciliation receipt differs from reconciled proposal")
        record["original_model_response"] = original_model_proposal_record
        record["semantic_reconciliation"] = reconciliation_receipt.to_record()
    resolved_key = _resolved_decision_payload_key(profile, request)
    join_key = (
        "selected_member_join"
        if profile is _FULL_SUPPORT_PROFILE
        else "selected_role_join"
    )
    record[resolved_key] = decision.to_audit_record()
    record[join_key] = [
        {
            "evaluation_rank": rank,
            "role": selected.role.value,
            "option_id": selected.option_id,
            "model_rank": selected.model_rank,
        }
        for rank, selected in enumerate(allocation.selected, start=1)
    ]
    if profile in {
        _MODEL_ANCHORED_PROFILE,
        _STRUCTURAL_POSTERIOR_PROFILE,
        _OPERATOR_STRATIFIED_PROFILE,
        _HORIZON_BOUNDED_PROFILE,
        _CONSTRAINT_DECOUPLED_HORIZON_PROFILE,
        _MINIMUM_INTERVENTION_HORIZON_PROFILE,
        _EVIDENCE_CALIBRATED_SOURCE_MIX_PROFILE,
        _CONTEXTUAL_SEARCH_ALLOCATION_PROFILE,
        _FRONTIER_PROBE_PROFILE,
        _FULL_SUPPORT_PROFILE,
        _TARGET_CONDITIONED_PROFILE,
        _CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PROFILE,
        _ACQUISITION_CERTIFIED_RESIDUAL_PROFILE,
        _REGRET_BOUNDED_INFORMATION_PROFILE,
    }:
        allocator_record = allocator.to_record()
        record["allocator_policy"] = allocator_record
        record["composition_identity_sha256"] = _composition_identity_sha256(
            profile,
            allocator_record,
        )
    if binding.common_candidate_pool is not None:
        record["common_candidate_pool"] = binding.common_candidate_pool.to_record()
    if binding.proposal_support is not None:
        record["proposal_support"] = binding.proposal_support.to_record()
    if binding.contextual_allocation is not None:
        record["contextual_allocation"] = binding.contextual_allocation.to_record()
    return record


def _output_members_from_proposal_record(
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
    record: object,
    *,
    constraint_decoupled: bool = False,
) -> tuple[Any, ...]:
    if type(record) is not dict:
        raise TypeError("original_k8_response must be an exact object")
    rows = record.get("members")
    if type(rows) is not list or len(rows) != CALIBRATED_PORTFOLIO_PROPOSAL_SIZE:
        raise ValueError("original_k8_response must contain exactly eight members")
    schema_rows: list[dict[str, object]] = []
    for expected_rank, row in enumerate(rows, start=1):
        if type(row) is not dict:
            raise TypeError("original_k8_response members must be exact objects")
        if type(row.get("model_rank")) is not int or row["model_rank"] != expected_rank:
            raise ValueError("original_k8_response has noncanonical model ranks")
        hierarchical_action = row.get("hierarchical_action")
        if hierarchical_action is None:
            schema_rows.append(
                {key: value for key, value in row.items() if key != "model_rank"}
            )
            continue
        if type(hierarchical_action) is not dict:
            raise TypeError("hierarchical_action must be an exact object")
        common = {
            key: value
            for key, value in row.items()
            if key not in {"model_rank", "option_id", "hierarchical_action"}
        }
        action_kind = hierarchical_action.get("action_kind")
        if action_kind == "atomic":
            schema_rows.append(
                {
                    "action_kind": "atomic",
                    "option_id": row.get("option_id"),
                    **common,
                }
            )
        elif action_kind == "compose_r2":
            schema_rows.append(
                {
                    "action_kind": "compose_r2",
                    "composite_option_id": row.get("option_id"),
                    "component_option_ids": hierarchical_action.get(
                        "component_option_ids"
                    ),
                    **common,
                }
            )
        else:
            raise ValueError("hierarchical_action declares an unknown kind")
    output_type = _calibrated_output_type(
        request,
        binding,
        constraint_decoupled=constraint_decoupled,
    )
    value = output_type.model_validate({"members": schema_rows}, strict=True)
    members = tuple(cast(Any, value).members)
    expected = _raw_proposal_record(request, binding.context, members)
    if record != expected:
        raise ValueError("original_k8_response is not the canonical proposal record")
    return members


@dataclass(frozen=True, slots=True)
class DecodedCalibratedPortfolioProposal:
    """Allocation-independent authenticated view of one original K8 proposal."""

    audit_kind: str
    selector_policy_id: str
    selector_policy_version: int
    selector_policy_definition_sha256: str
    input_binding_sha256: str
    proposal_sha256: str
    original_k8_response: FrozenJsonObject
    model_members: tuple[CalibratedPortfolioModelMember, ...]
    slate: CalibratedSlate

    def __post_init__(self) -> None:
        profile = _profile_for_audit_kind(self.audit_kind)
        observed_profile = (
            self.selector_policy_id,
            self.selector_policy_version,
            self.selector_policy_definition_sha256,
        )
        expected_profile = (
            profile.policy_id,
            profile.policy_version,
            profile.policy_definition_sha256,
        )
        if observed_profile != expected_profile:
            raise ValueError("decoded proposal names a foreign selector profile")
        for name, value in (
            ("input_binding_sha256", self.input_binding_sha256),
            ("proposal_sha256", self.proposal_sha256),
        ):
            require_sha256(value, name)
        if type(self.original_k8_response) is not FrozenJsonObject:
            raise TypeError("original_k8_response must be exact frozen JSON")
        if type(self.model_members) is not tuple or len(self.model_members) != 8:
            raise ValueError("model_members must contain exactly eight values")
        for expected_rank, value in enumerate(self.model_members, start=1):
            if type(value) is not CalibratedPortfolioModelMember:
                raise TypeError("model_members must be exact model members")
            value.__post_init__()
            if value.model_rank != expected_rank:
                raise ValueError("model_members must preserve canonical model ranks")
        if type(self.slate) is not CalibratedSlate:
            raise TypeError("slate must be exact CalibratedSlate")
        self.slate.revalidate()
        if self.slate.selector_decision_sha256 != self.proposal_sha256:
            raise ValueError("slate belongs to a foreign K8 proposal")


def _composition_identity_sha256(
    profile: _CalibratedPortfolioSelectionProfile,
    allocator_record: dict[str, object],
) -> str:
    return hashlib.sha256(
        b"agent-evolve:calibrated-portfolio-composition:v1\x00"
        + _canonical_json(
            {
                "selector_policy_id": profile.policy_id,
                "selector_policy_version": profile.policy_version,
                "selector_policy_definition_sha256": (profile.policy_definition_sha256),
                "allocator_policy": allocator_record,
            }
        )
    ).hexdigest()


def _require_declared_allocator_identity(
    payload: dict[str, object],
    *,
    profile: _CalibratedPortfolioSelectionProfile,
) -> None:
    allocation = payload.get("allocation")
    if type(allocation) is not dict:
        raise TypeError("allocation must remain an exact audit object")
    if profile is _FOUR_ROLE_PROFILE:
        expected = TraceCalibratedSlatePolicy(
            SlateAllocationMode.CALIBRATED_FOUR_ROLE
        ).to_record()
        observed = {
            "policy_id": allocation.get("policy_id"),
            "policy_version": allocation.get("policy_version"),
            "definition_sha256": allocation.get("policy_definition_sha256"),
            "mode": allocation.get("mode"),
        }
        if observed != expected:
            raise ValueError("audit allocation names a foreign four-role policy")
        return

    allocator_record = payload.get("allocator_policy")
    if type(allocator_record) is not dict:
        raise TypeError("configured allocator audit omitted allocator_policy")
    if profile is _ACQUISITION_CERTIFIED_RESIDUAL_PROFILE:
        if set(allocator_record) != {
            "policy_id",
            "policy_version",
            "definition_sha256",
            "context_provider",
            "scorer",
            "exact_combination_limit",
            "tie_tolerance_hex",
        }:
            raise ValueError("acquisition-certified allocator field set is invalid")
        if (
            allocator_record.get("policy_id"),
            allocator_record.get("policy_version"),
        ) != ("acquisition_certified_residual_slate", 1):
            raise ValueError("acquisition-certified allocator identity is invalid")
        require_sha256(
            allocator_record.get("definition_sha256"),
            "acquisition-certified allocator definition_sha256",
        )
        for field_name in ("context_provider", "scorer"):
            identity = allocator_record.get(field_name)
            if type(identity) is not dict or set(identity) != {
                "provider_id" if field_name == "context_provider" else "policy_id",
                "provider_version"
                if field_name == "context_provider"
                else "policy_version",
                "definition_sha256",
            }:
                raise TypeError(
                    f"acquisition-certified {field_name} identity is malformed"
                )
            require_sha256(
                identity.get("definition_sha256"),
                f"acquisition-certified {field_name}.definition_sha256",
            )
        if (
            allocation.get("policy_id"),
            allocation.get("policy_version"),
            allocation.get("policy_definition_sha256"),
        ) != (
            allocator_record["policy_id"],
            allocator_record["policy_version"],
            allocator_record["definition_sha256"],
        ):
            raise ValueError(
                "acquisition-certified decision and allocator identity disagree"
            )
        score_decision = allocation.get("score_decision")
        scorer = allocator_record["scorer"]
        if type(score_decision) is not dict or type(scorer) is not dict:
            raise TypeError("acquisition-certified score decision is malformed")
        score_policy = score_decision.get("policy")
        if type(score_policy) is not dict or score_policy != {
            "policy_id": scorer["policy_id"],
            "policy_version": scorer["policy_version"],
            "definition_sha256": scorer["definition_sha256"],
        }:
            raise ValueError("acquisition-certified scorer identity disagrees")
        expected_composition = _composition_identity_sha256(
            profile,
            allocator_record,
        )
        if payload.get("composition_identity_sha256") != expected_composition:
            raise ValueError("acquisition-certified composition identity is invalid")
        return
    if profile is _REGRET_BOUNDED_INFORMATION_PROFILE:
        if set(allocator_record) != {
            "policy_id",
            "policy_version",
            "definition_sha256",
            "context_provider",
            "scorer",
            "future_value_policy",
            "minimum_acquisition_retention_ratio_hex",
            "minimum_residual_audit_members",
            "calibration_error_bound_hex",
            "allow_development_assay",
            "exact_combination_limit",
            "tie_tolerance_hex",
        }:
            raise ValueError("regret-bounded allocator field set is invalid")
        if (
            allocator_record.get("policy_id"),
            allocator_record.get("policy_version"),
        ) != ("regret_bounded_information_slate", 2):
            raise ValueError("regret-bounded allocator identity is invalid")
        require_sha256(
            allocator_record.get("definition_sha256"),
            "regret-bounded allocator definition_sha256",
        )
        for field_name in ("context_provider", "scorer", "future_value_policy"):
            identity = allocator_record.get(field_name)
            expected_keys = {
                "provider_id",
                "provider_version",
                "definition_sha256",
            } if field_name == "context_provider" else {
                "policy_id",
                "policy_version",
                "definition_sha256",
            }
            if type(identity) is not dict or set(identity) != expected_keys:
                raise TypeError(f"regret-bounded {field_name} identity is malformed")
            require_sha256(
                identity.get("definition_sha256"),
                f"regret-bounded {field_name}.definition_sha256",
            )
        if (
            allocation.get("policy_id"),
            allocation.get("policy_version"),
            allocation.get("policy_definition_sha256"),
        ) != (
            allocator_record["policy_id"],
            allocator_record["policy_version"],
            allocator_record["definition_sha256"],
        ):
            raise ValueError("regret-bounded decision and allocator identity disagree")
        score_decision = allocation.get("score_decision")
        scorer = allocator_record["scorer"]
        if type(score_decision) is not dict or type(scorer) is not dict:
            raise TypeError("regret-bounded score decision is malformed")
        score_policy = score_decision.get("policy")
        if type(score_policy) is not dict or score_policy != {
            "policy_id": scorer["policy_id"],
            "policy_version": scorer["policy_version"],
            "definition_sha256": scorer["definition_sha256"],
        }:
            raise ValueError("regret-bounded scorer identity disagrees")
        expected_composition = _composition_identity_sha256(
            profile,
            allocator_record,
        )
        if payload.get("composition_identity_sha256") != expected_composition:
            raise ValueError("regret-bounded composition identity is invalid")
        return
    if profile in {
        _TARGET_CONDITIONED_PROFILE,
        _CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PROFILE,
    }:
        expected_allocator_fields = {
            "policy_id",
            "policy_version",
            "definition_sha256",
            "profile",
            "feature_projector",
            "structural_score_projector",
            "context_provider",
            "realizability_projector",
        }
        if set(allocator_record) != expected_allocator_fields:
            raise ValueError("target-conditioned allocator field set is invalid")
        if (
            allocator_record.get("policy_id"),
            allocator_record.get("policy_version"),
            allocator_record.get("definition_sha256"),
        ) != (
            TARGET_CONDITIONED_ALLOCATOR_ID,
            TARGET_CONDITIONED_ALLOCATOR_VERSION,
            TARGET_CONDITIONED_ALLOCATOR_DEFINITION_SHA256,
        ):
            raise ValueError("target-conditioned allocator identity is invalid")
        decoded_profile = TargetConditionedAcquisitionProfile.from_record(
            allocator_record.get("profile")
        )
        if allocator_record.get("feature_projector") != (
            TargetConditionedPortableFeatureProjector().to_record()
        ):
            raise ValueError("target-conditioned feature projector is invalid")
        for provider_field in (
            "structural_score_projector",
            "context_provider",
            "realizability_projector",
        ):
            provider = allocator_record.get(provider_field)
            if type(provider) is not dict or set(provider) != {
                "provider_id",
                "provider_version",
                "definition_sha256",
            }:
                raise TypeError(
                    f"target-conditioned {provider_field} identity is malformed"
                )
            if (
                type(provider.get("provider_id")) is not str
                or not provider["provider_id"]
            ):
                raise TypeError(f"target-conditioned {provider_field} id is invalid")
            if (
                type(provider.get("provider_version")) is not int
                or provider["provider_version"] <= 0
            ):
                raise TypeError(
                    f"target-conditioned {provider_field} version is invalid"
                )
            require_sha256(
                provider.get("definition_sha256"),
                f"target-conditioned {provider_field}.definition_sha256",
            )
        if (
            allocation.get("policy_id"),
            allocation.get("policy_version"),
            allocation.get("policy_definition_sha256"),
        ) != (
            TARGET_CONDITIONED_CORE_ID,
            TARGET_CONDITIONED_CORE_VERSION,
            TARGET_CONDITIONED_CORE_DEFINITION_SHA256,
        ):
            raise ValueError("target-conditioned allocation identity is invalid")
        if allocation.get("profile") != decoded_profile.to_record():
            raise ValueError("target-conditioned decision and profile disagree")
        expected_composition = _composition_identity_sha256(
            profile,
            allocator_record,
        )
        if payload.get("composition_identity_sha256") != expected_composition:
            raise ValueError("target-conditioned composition identity is invalid")
        return
    if profile is _FULL_SUPPORT_PROFILE:
        expected_allocator = FullSupportSlatePolicy().to_record()
        if allocator_record != expected_allocator:
            raise ValueError("full-support allocator identity is invalid")
        expected_allocation = TraceCalibratedSlatePolicy(
            SlateAllocationMode.DIRECT_MODEL_TOP_K
        ).to_record()
        observed_allocation = {
            "policy_id": allocation.get("policy_id"),
            "policy_version": allocation.get("policy_version"),
            "definition_sha256": allocation.get("policy_definition_sha256"),
            "mode": allocation.get("mode"),
        }
        if observed_allocation != expected_allocation:
            raise ValueError("full-support allocation decision identity is invalid")
        expected_composition = _composition_identity_sha256(
            profile,
            expected_allocator,
        )
        if payload.get("composition_identity_sha256") != expected_composition:
            raise ValueError("full-support composition identity is invalid")
        return
    if profile is _STRUCTURAL_POSTERIOR_PROFILE:
        expected_allocator = StructuralPosteriorSlatePolicy().to_record()
        if allocator_record != expected_allocator:
            raise ValueError("structural-posterior allocator identity is invalid")
        for key, expected_value in (
            ("policy_id", expected_allocator["policy_id"]),
            ("policy_version", expected_allocator["policy_version"]),
            ("policy_definition_sha256", expected_allocator["definition_sha256"]),
        ):
            if allocation.get(key) != expected_value:
                raise ValueError(
                    "allocation decision and structural-posterior identity disagree"
                )
        expected_composition = _composition_identity_sha256(
            profile,
            expected_allocator,
        )
        if payload.get("composition_identity_sha256") != expected_composition:
            raise ValueError("structural-posterior composition identity is invalid")
        return
    if profile is _OPERATOR_STRATIFIED_PROFILE:
        configuration = allocator_record.get("configuration")
        if type(configuration) is not dict:
            raise TypeError("operator-stratified allocator omitted its configuration")
        raw_minimums = configuration.get("required_family_minimums")
        if type(raw_minimums) is not list or not raw_minimums:
            raise TypeError("required_family_minimums must remain a non-empty list")
        parsed_minimums: list[tuple[str, int]] = []
        for value in raw_minimums:
            if type(value) is not dict or set(value) != {
                "family",
                "minimum_evaluations",
            }:
                raise TypeError("required family minimum must remain an exact object")
            family = value["family"]
            minimum = value["minimum_evaluations"]
            if type(family) is not str or type(minimum) is not int:
                raise TypeError("required family minimum fields have invalid types")
            parsed_minimums.append((family, minimum))
        expected_allocator = OperatorStratifiedStructuralPosteriorSlatePolicy(
            tuple(parsed_minimums)
        ).to_record()
        if allocator_record != expected_allocator:
            raise ValueError("operator-stratified allocator identity is invalid")
        for key, expected_value in (
            ("policy_id", expected_allocator["policy_id"]),
            ("policy_version", expected_allocator["policy_version"]),
            ("policy_definition_sha256", expected_allocator["definition_sha256"]),
        ):
            if allocation.get(key) != expected_value:
                raise ValueError(
                    "allocation decision and operator-stratified identity disagree"
                )
        expected_composition = _composition_identity_sha256(
            profile,
            expected_allocator,
        )
        if payload.get("composition_identity_sha256") != expected_composition:
            raise ValueError("operator-stratified composition identity is invalid")
        return
    if profile in {
        _HORIZON_BOUNDED_PROFILE,
        _CONSTRAINT_DECOUPLED_HORIZON_PROFILE,
        _MINIMUM_INTERVENTION_HORIZON_PROFILE,
        _EVIDENCE_CALIBRATED_SOURCE_MIX_PROFILE,
        _CONTEXTUAL_SEARCH_ALLOCATION_PROFILE,
    }:
        configuration = allocator_record.get("configuration")
        if type(configuration) is not dict:
            raise TypeError("horizon-bounded allocator omitted its configuration")
        raw_phases = configuration.get("family_exposure_phases")
        if type(raw_phases) is not list or not raw_phases:
            raise TypeError("family_exposure_phases must remain a non-empty list")
        parsed_phases: list[FamilyExposurePhase] = []
        for raw_phase in raw_phases:
            if type(raw_phase) is not dict or set(raw_phase) != {
                "start_wave_index",
                "bounds",
            }:
                raise TypeError("family exposure phase must remain an exact object")
            start = raw_phase["start_wave_index"]
            raw_bounds = raw_phase["bounds"]
            if type(start) is not int or type(raw_bounds) is not list:
                raise TypeError("family exposure phase fields have invalid types")
            parsed_bounds: list[FamilyExposureBound] = []
            for raw_bound in raw_bounds:
                if type(raw_bound) is not dict or set(raw_bound) != {
                    "family",
                    "minimum_evaluations",
                    "maximum_evaluations",
                }:
                    raise TypeError("family exposure bound must remain an exact object")
                family = raw_bound["family"]
                minimum = raw_bound["minimum_evaluations"]
                maximum = raw_bound["maximum_evaluations"]
                if (
                    type(family) is not str
                    or type(minimum) is not int
                    or type(maximum) is not int
                ):
                    raise TypeError("family exposure bound fields have invalid types")
                parsed_bounds.append(FamilyExposureBound(family, minimum, maximum))
            parsed_phases.append(FamilyExposurePhase(start, tuple(parsed_bounds)))
        expected_allocator = HorizonBoundedStructuralPosteriorSlatePolicy(
            tuple(parsed_phases)
        ).to_record()
        if allocator_record != expected_allocator:
            raise ValueError("horizon-bounded allocator identity is invalid")
        for key, expected_value in (
            ("policy_id", expected_allocator["policy_id"]),
            ("policy_version", expected_allocator["policy_version"]),
            ("policy_definition_sha256", expected_allocator["definition_sha256"]),
            (
                "policy_configuration_sha256",
                expected_allocator["configuration_sha256"],
            ),
        ):
            if allocation.get(key) != expected_value:
                raise ValueError(
                    "allocation decision and horizon-bounded identity disagree"
                )
        expected_composition = _composition_identity_sha256(
            profile,
            expected_allocator,
        )
        if payload.get("composition_identity_sha256") != expected_composition:
            raise ValueError("horizon-bounded composition identity is invalid")
        return
    if profile is _FRONTIER_PROBE_PROFILE:
        expected_allocator = FrontierProbeSlatePolicy().to_record()
        if allocator_record != expected_allocator:
            raise ValueError("frontier-probe allocator identity is invalid")
        for key, expected_value in (
            ("policy_id", expected_allocator["policy_id"]),
            ("policy_version", expected_allocator["policy_version"]),
            ("policy_definition_sha256", expected_allocator["definition_sha256"]),
            (
                "policy_configuration_sha256",
                expected_allocator["configuration_sha256"],
            ),
        ):
            if allocation.get(key) != expected_value:
                raise ValueError(
                    "allocation decision and frontier-probe identity disagree"
                )
        expected_composition = _composition_identity_sha256(
            profile,
            expected_allocator,
        )
        if payload.get("composition_identity_sha256") != expected_composition:
            raise ValueError("frontier-probe composition identity is invalid")
        return
    configuration = allocator_record.get("configuration")
    if type(configuration) is not dict:
        raise TypeError("model-anchored allocator omitted its configuration")
    model_anchor_count = configuration.get("model_anchor_count")
    if type(model_anchor_count) is not int:
        raise TypeError("model_anchor_count must remain an exact integer")
    expected_allocator = ModelAnchoredCalibratedSlatePolicy(
        model_anchor_count
    ).to_record()
    if allocator_record != expected_allocator:
        raise ValueError("model-anchored allocator identity/configuration is invalid")
    for key, expected_value in (
        ("policy_id", expected_allocator["policy_id"]),
        ("policy_version", expected_allocator["policy_version"]),
        ("policy_definition_sha256", expected_allocator["definition_sha256"]),
        (
            "policy_configuration_sha256",
            expected_allocator["configuration_sha256"],
        ),
    ):
        if allocation.get(key) != expected_value:
            raise ValueError("allocation decision and allocator identity disagree")
    expected_composition = _composition_identity_sha256(
        profile,
        expected_allocator,
    )
    if payload.get("composition_identity_sha256") != expected_composition:
        raise ValueError("model-anchored composition identity is invalid")


def _decoded_horizon_family_bounds(
    payload: dict[str, object],
    *,
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
) -> tuple[tuple[str, int, int], ...]:
    allocator_record = payload.get("allocator_policy")
    if type(allocator_record) is not dict:
        raise TypeError("constraint-decoupled audit omitted its allocator policy")
    configuration = allocator_record.get("configuration")
    if type(configuration) is not dict:
        raise TypeError("horizon allocator omitted its configuration")
    raw_phases = configuration.get("family_exposure_phases")
    if type(raw_phases) is not list or not raw_phases:
        raise TypeError("horizon allocator phases must remain a non-empty list")
    phases: list[FamilyExposurePhase] = []
    for raw_phase in raw_phases:
        if type(raw_phase) is not dict:
            raise TypeError("horizon phase must remain an exact object")
        start = raw_phase.get("start_wave_index")
        raw_bounds = raw_phase.get("bounds")
        if type(start) is not int or type(raw_bounds) is not list:
            raise TypeError("horizon phase fields have invalid types")
        bounds: list[FamilyExposureBound] = []
        for raw_bound in raw_bounds:
            if type(raw_bound) is not dict:
                raise TypeError("horizon bound must remain an exact object")
            family = raw_bound.get("family")
            minimum = raw_bound.get("minimum_evaluations")
            maximum = raw_bound.get("maximum_evaluations")
            if (
                type(family) is not str
                or type(minimum) is not int
                or type(maximum) is not int
            ):
                raise TypeError("horizon bound fields have invalid types")
            bounds.append(FamilyExposureBound(family, minimum, maximum))
        phases.append(FamilyExposurePhase(start, tuple(bounds)))
    allocator = HorizonBoundedStructuralPosteriorSlatePolicy(tuple(phases))
    if allocator.to_record() != allocator_record:
        raise ValueError("horizon allocator differs from its canonical replay")
    active = allocator.exposure_phase_for_wave(binding.context.wave_index)
    requested = tuple(
        (
            value.family,
            value.minimum_evaluations,
            value.maximum_evaluations,
        )
        for value in active.bounds
    )
    if not request.require_pairwise_disjoint_parent_patches:
        return requested
    option_ids = (
        tuple(value.option_id for value in request.finite_variation_contract.options)
        if binding.common_candidate_pool is None
        else binding.common_candidate_pool.option_ids
    )
    return project_family_exposure_bounds_to_pairwise_disjoint_feasibility(
        request.finite_variation_contract,
        option_ids,
        portfolio_size=request.portfolio_size,
        min_distinct_families=request.min_distinct_families,
        requested_bounds=requested,
    )


def decode_calibrated_portfolio_proposal(
    audit: PortfolioSelectionSupplementalAudit,
    *,
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
) -> DecodedCalibratedPortfolioProposal:
    """Authenticate the original K8 proposal without replaying its K4 allocator.

    This seam intentionally does not assert that the historical allocation is
    the expected allocation.  It validates the request, prospective binding,
    selector profile, declared allocator identity, canonical provider response,
    proposal digest, and reconstructed slate.  A caller can therefore apply a
    different trusted allocator to the returned slate in an offline analysis.
    """

    if type(audit) is not PortfolioSelectionSupplementalAudit:
        raise TypeError("audit must be exact PortfolioSelectionSupplementalAudit")
    audit.__post_init__()
    if type(request) is not PortfolioSelectionRequest:
        raise TypeError("request must be exact PortfolioSelectionRequest")
    request.__post_init__()
    profile = _profile_for_audit_kind(audit.audit_kind)
    _validate_binding_for_request(
        request,
        binding,
        selector_policy_definition_sha256=profile.policy_definition_sha256,
        constraint_decoupled=profile.constraint_decoupled,
        contextual_search_allocation=profile.contextual_search_allocation,
    )
    if audit.request_sha256 != request.request_sha256:
        raise ValueError("supplemental audit belongs to a foreign request")
    payload = thaw_json(audit.payload)
    if type(payload) is not dict:
        raise TypeError("calibrated supplemental payload must be an exact object")
    expected_keys = {
        "schema_version",
        "event_type",
        "policy_id",
        "policy_version",
        "policy_definition_sha256",
        "prompt_definition_sha256",
        "input_binding_sha256",
        "proposal_size",
        "evaluation_size",
        "proposal_sha256",
        "original_k8_response",
        "calibrated_slate",
        "allocation",
        "selected_prediction_receipt_sha256s",
        "invariants",
    }
    resolved_key = _resolved_decision_payload_key(profile, request)
    join_key = (
        "selected_member_join"
        if profile is _FULL_SUPPORT_PROFILE
        else "selected_role_join"
    )
    expected_keys.update({resolved_key, join_key})
    if profile.constraint_decoupled:
        expected_keys.update({"original_model_response", "semantic_reconciliation"})
    if profile in {
        _MODEL_ANCHORED_PROFILE,
        _STRUCTURAL_POSTERIOR_PROFILE,
        _OPERATOR_STRATIFIED_PROFILE,
        _HORIZON_BOUNDED_PROFILE,
        _CONSTRAINT_DECOUPLED_HORIZON_PROFILE,
        _MINIMUM_INTERVENTION_HORIZON_PROFILE,
        _EVIDENCE_CALIBRATED_SOURCE_MIX_PROFILE,
        _CONTEXTUAL_SEARCH_ALLOCATION_PROFILE,
        _FRONTIER_PROBE_PROFILE,
        _FULL_SUPPORT_PROFILE,
        _TARGET_CONDITIONED_PROFILE,
        _CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PROFILE,
        _ACQUISITION_CERTIFIED_RESIDUAL_PROFILE,
        _REGRET_BOUNDED_INFORMATION_PROFILE,
    }:
        expected_keys.update({"allocator_policy", "composition_identity_sha256"})
    if binding.common_candidate_pool is not None:
        expected_keys.add("common_candidate_pool")
    if binding.proposal_support is not None:
        expected_keys.add("proposal_support")
    if binding.contextual_allocation is not None:
        expected_keys.add("contextual_allocation")
    if set(payload) != expected_keys:
        raise ValueError("calibrated audit payload has a foreign field set")
    observed_header = (
        payload["schema_version"],
        payload["event_type"],
        payload["policy_id"],
        payload["policy_version"],
        payload["policy_definition_sha256"],
        payload["prompt_definition_sha256"],
        payload["input_binding_sha256"],
        payload["proposal_size"],
        payload["evaluation_size"],
    )
    expected_header = (
        _payload_schema_version(profile, binding, request),
        profile.event_type,
        profile.policy_id,
        profile.policy_version,
        profile.policy_definition_sha256,
        _prompt_definition_sha256_for_binding(
            request,
            binding,
            constraint_decoupled=profile.constraint_decoupled,
        ),
        binding.binding_sha256,
        CALIBRATED_PORTFOLIO_PROPOSAL_SIZE,
        request.portfolio_size,
    )
    if observed_header != expected_header:
        raise ValueError("calibrated proposal audit header is not authenticated")
    if (
        binding.common_candidate_pool is not None
        and payload["common_candidate_pool"]
        != binding.common_candidate_pool.to_record()
    ):
        raise ValueError("calibrated audit common candidate pool is inconsistent")
    if (
        binding.proposal_support is not None
        and payload["proposal_support"] != binding.proposal_support.to_record()
    ):
        raise ValueError("calibrated audit proposal support is inconsistent")
    if (
        binding.contextual_allocation is not None
        and payload["contextual_allocation"]
        != binding.contextual_allocation.to_record()
    ):
        raise ValueError("calibrated audit contextual allocation is inconsistent")
    _require_declared_allocator_identity(payload, profile=profile)
    resolved = payload[resolved_key]
    if type(resolved) is not dict:
        raise TypeError("resolved portfolio decision must remain an exact object")
    if resolved.get("decision_sha256") != audit.decision_sha256:
        raise ValueError("supplemental audit decision identity is inconsistent")
    proposal_record = payload["original_k8_response"]
    if profile.constraint_decoupled:
        original_model_record = payload["original_model_response"]
        _output_members_from_proposal_record(
            request,
            binding,
            original_model_record,
            constraint_decoupled=True,
        )
        if type(original_model_record) is not dict:
            raise AssertionError("model proposal record did not remain an object")
        original_model_members = _typed_model_members(original_model_record)
        required_family_bounds = (
            ()
            if profile
            in {
                _CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PROFILE,
                _ACQUISITION_CERTIFIED_RESIDUAL_PROFILE,
                _REGRET_BOUNDED_INFORMATION_PROFILE,
            }
            else _decoded_horizon_family_bounds(
                payload,
                request=request,
                binding=binding,
            )
        )
        expected_members, expected_reconciliation = _reconcile_semantic_members(
            request,
            binding,
            original_model_members,
            required_evaluation_family_bounds=required_family_bounds,
            original_model_proposal_sha256=_proposal_sha256(original_model_record),
            minimum_intervention_projection=(profile.minimum_intervention_projection),
            evidence_calibrated_source_mix=(profile.evidence_calibrated_source_mix),
            contextual_search_allocation=(profile.contextual_search_allocation),
            acquisition_certification_reference_option_ids=(
                _audit_acquisition_certification_reference_option_ids(
                    payload,
                    profile,
                )
            ),
        )
        expected_reconciled_record = _typed_proposal_record(
            request,
            binding.context,
            expected_members,
        )
        if proposal_record != expected_reconciled_record:
            raise ValueError("reconciled K8 differs from deterministic replay")
        if payload["semantic_reconciliation"] != (expected_reconciliation.to_record()):
            raise ValueError("semantic reconciliation receipt fails exact replay")
    # Constraint-decoupled profiles authenticate the reconciled K8 by exact
    # deterministic replay above.  Reapplying the provider-facing cross-member
    # validator here would incorrectly reject an explicitly recorded deferred
    # proposal-support reservation (for example when protected source and
    # hierarchical-composition constraints exhaust the K8 capacity).
    _output_members_from_proposal_record(
        request,
        binding,
        proposal_record,
        constraint_decoupled=profile.constraint_decoupled,
    )
    if type(proposal_record) is not dict:
        raise AssertionError("canonical proposal record did not remain an object")
    model_members = _typed_model_members(proposal_record)
    proposal_sha256 = _proposal_sha256(proposal_record)
    if payload["proposal_sha256"] != proposal_sha256:
        raise ValueError("calibrated proposal digest is inconsistent")
    slate, _ = _build_calibrated_slate(
        request,
        binding,
        model_members,
        proposal_sha256=proposal_sha256,
    )
    if payload["calibrated_slate"] != slate.to_record():
        raise ValueError("calibrated slate differs from the canonical K8 proposal")
    frozen_proposal = freeze_json(proposal_record)
    if type(frozen_proposal) is not FrozenJsonObject:
        raise AssertionError("canonical K8 proposal did not freeze as an object")
    return DecodedCalibratedPortfolioProposal(
        audit_kind=audit.audit_kind,
        selector_policy_id=profile.policy_id,
        selector_policy_version=profile.policy_version,
        selector_policy_definition_sha256=profile.policy_definition_sha256,
        input_binding_sha256=binding.binding_sha256,
        proposal_sha256=proposal_sha256,
        original_k8_response=frozen_proposal,
        model_members=model_members,
        slate=slate,
    )


def allocate_calibrated_portfolio_proposal(
    proposal: DecodedCalibratedPortfolioProposal,
    *,
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
    allocator: CalibratedPortfolioAllocator,
) -> CalibratedPortfolioAllocationDecision:
    """Counterfactually allocate an authenticated K8 proposal without a call.

    The proposal retains its historical selector scope.  The injected allocator
    is independently validated and receives only the reconstructed slate plus
    the same sealed request/binding facts.  This returns an allocation receipt;
    it does not rewrite the historical supplemental audit or claim that the new
    K4 was actually evaluated.
    """

    if type(proposal) is not DecodedCalibratedPortfolioProposal:
        raise TypeError("proposal must be exact DecodedCalibratedPortfolioProposal")
    proposal.__post_init__()
    if type(request) is not PortfolioSelectionRequest:
        raise TypeError("request must be exact PortfolioSelectionRequest")
    request.__post_init__()
    _profile_for_allocator(allocator)
    binding.require_request(request)
    proposal_record = thaw_json(proposal.original_k8_response)
    if type(proposal_record) is not dict:
        raise AssertionError("decoded K8 proposal did not thaw as an object")
    if proposal_record.get("request_sha256") != request.request_sha256:
        raise ValueError("decoded proposal belongs to a foreign request")
    if proposal.input_binding_sha256 != binding.binding_sha256:
        raise ValueError("decoded proposal belongs to a foreign input binding")
    expected_slate, _ = _build_calibrated_slate(
        request,
        binding,
        proposal.model_members,
        proposal_sha256=proposal.proposal_sha256,
    )
    if proposal.slate != expected_slate:
        raise ValueError("decoded proposal differs from its request/binding slate")
    profile = _profile_for_policy_definition_sha256(
        proposal.selector_policy_definition_sha256
    )
    return _allocate_slate(
        request,
        binding,
        proposal.slate,
        allocator,
        required_option_ids=_required_allocation_option_ids(
            request,
            profile,
            None,
        ),
    )


@dataclass(frozen=True, slots=True)
class DecodedCalibratedPortfolioAudit:
    """Strict typed replay view; rationale prose intentionally stays opaque."""

    slate: CalibratedSlate
    allocation: CalibratedPortfolioAllocationDecision
    selected_prediction_receipts: tuple[ForecastPredictionReceipt, ...]

    def __post_init__(self) -> None:
        if type(self.slate) is not CalibratedSlate:
            raise TypeError("slate must be exact CalibratedSlate")
        self.slate.revalidate()
        if type(self.allocation) not in {
            SlateAllocationDecision,
            ModelAnchoredSlateDecision,
            StructuralPosteriorSlateDecision,
            OperatorStratifiedStructuralPosteriorSlateDecision,
            HorizonBoundedStructuralPosteriorSlateDecision,
            FrontierProbeSlateDecision,
            TargetConditionedSlateDecision,
            AcquisitionCertifiedSlateDecision,
            RegretBoundedSlateDecision,
        }:
            raise TypeError("allocation must be an exact supported decision")
        self.allocation.revalidate()
        allocation_request = (
            self.allocation.request.allocation_request
            if type(self.allocation) is TargetConditionedSlateDecision
            else self.allocation.request
        )
        if allocation_request.slate != self.slate:
            raise ValueError("allocation belongs to a foreign calibrated slate")
        expected = _selected_predictions(self.slate, self.allocation)
        if self.selected_prediction_receipts != expected:
            raise ValueError("selected prediction receipts differ from the allocation")


def decode_calibrated_portfolio_audit(
    audit: PortfolioSelectionSupplementalAudit,
    *,
    request: PortfolioSelectionRequest,
    binding: CalibratedPortfolioInputBinding,
    allocator: CalibratedPortfolioAllocator = TraceCalibratedSlatePolicy(
        SlateAllocationMode.CALIBRATED_FOUR_ROLE
    ),
) -> DecodedCalibratedPortfolioAudit:
    """Rebuild and replay a typed slate/allocation from one supplemental audit.

    The untrusted JSON is not sufficient by itself.  The caller supplies the
    original sealed request and its independently authenticated pre-call input
    binding.  The decoder reconstructs every receipt, reruns the deterministic
    allocator, rebuilds the K4 ranked decision, and requires byte-semantic
    equality with the complete frozen audit payload.
    """

    profile = _profile_for_audit_kind(audit.audit_kind)
    expected_profile = _profile_for_allocator_authority(
        allocator,
        constraint_decoupled=profile.constraint_decoupled,
        minimum_intervention_projection=(profile.minimum_intervention_projection),
        evidence_calibrated_source_mix=(profile.evidence_calibrated_source_mix),
        contextual_search_allocation=profile.contextual_search_allocation,
    )
    if profile != expected_profile:
        raise ValueError("supplemental audit names a foreign allocator profile")
    proposal = decode_calibrated_portfolio_proposal(
        audit,
        request=request,
        binding=binding,
    )
    if proposal.selector_policy_definition_sha256 != profile.policy_definition_sha256:
        raise ValueError("supplemental audit names a foreign allocator profile")
    proposal_record = thaw_json(proposal.original_k8_response)
    if type(proposal_record) is not dict:
        raise AssertionError("decoded K8 proposal did not thaw as an object")
    original_model_record = None
    reconciliation_receipt = None
    if profile.constraint_decoupled:
        audit_payload = thaw_json(audit.payload)
        if type(audit_payload) is not dict:
            raise AssertionError("calibrated audit payload did not thaw as an object")
        original_model_record = audit_payload.get("original_model_response")
        if type(original_model_record) is not dict:
            raise TypeError("constraint-decoupled audit omitted model response")
        original_model_members = _typed_model_members(original_model_record)
        expected_members, reconciliation_receipt = _reconcile_semantic_members(
            request,
            binding,
            original_model_members,
            required_evaluation_family_bounds=(
                _decoded_horizon_family_bounds(
                    audit_payload,
                    request=request,
                    binding=binding,
                )
                if type(allocator) is HorizonBoundedStructuralPosteriorSlatePolicy
                else ()
            ),
            original_model_proposal_sha256=_proposal_sha256(original_model_record),
            minimum_intervention_projection=(profile.minimum_intervention_projection),
            evidence_calibrated_source_mix=(profile.evidence_calibrated_source_mix),
            contextual_search_allocation=(profile.contextual_search_allocation),
            acquisition_certification_reference_option_ids=(
                _acquisition_certification_reference_option_ids(
                    request,
                    allocator,
                )
            ),
        )
        if expected_members != proposal.model_members:
            raise ValueError("decoded reconciliation members fail exact replay")
    by_option = {value.option_id: value for value in proposal.model_members}
    allocation = _allocate_slate(
        request,
        binding,
        proposal.slate,
        allocator,
        required_option_ids=_required_allocation_option_ids(
            request,
            profile,
            reconciliation_receipt,
        ),
    )
    decision = _resolve_ranked_decision(
        request,
        allocation,
        by_option,
        profile=profile,
    )
    expected_payload = freeze_json(
        _audit_payload_record(
            request=request,
            binding=binding,
            proposal_record=proposal_record,
            proposal_sha256=proposal.proposal_sha256,
            slate=proposal.slate,
            allocation=allocation,
            allocator=allocator,
            profile=profile,
            decision=decision,
            original_model_proposal_record=original_model_record,
            reconciliation_receipt=reconciliation_receipt,
        )
    )
    if type(expected_payload) is not FrozenJsonObject:
        raise AssertionError("replayed calibrated audit did not freeze as an object")
    if expected_payload != audit.payload:
        raise ValueError("calibrated audit payload fails exact typed replay")
    if audit.decision_sha256 != decision.decision_sha256:
        raise ValueError("supplemental audit belongs to a foreign ranked decision")
    return DecodedCalibratedPortfolioAudit(
        slate=proposal.slate,
        allocation=allocation,
        selected_prediction_receipts=_selected_predictions(
            proposal.slate,
            allocation,
        ),
    )


@dataclass(slots=True)
class PydanticAICalibratedPortfolioSelectionPolicy:
    """Adapt one model-proposed slate into four engine-allocated members."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: TraceCalibratedSlatePolicy = TraceCalibratedSlatePolicy(
        SlateAllocationMode.CALIBRATED_FOUR_ROLE
    )

    policy_id: ClassVar[str] = CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID
    policy_version: ClassVar[int] = CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = (
        CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not TraceCalibratedSlatePolicy:
            raise TypeError("allocator must be exact TraceCalibratedSlatePolicy")
        self.allocator.__post_init__()
        if self.allocator.mode is not SlateAllocationMode.CALIBRATED_FOUR_ROLE:
            raise ValueError("v2 adapter requires calibrated four-role allocation")

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
        )


@dataclass(slots=True)
class PydanticAIFullSupportCalibratedPortfolioSelectionPolicy:
    """Quality-first adapter that evaluates the complete authenticated K8."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: FullSupportSlatePolicy = FullSupportSlatePolicy()

    policy_id: ClassVar[str] = FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID
    policy_version: ClassVar[int] = (
        FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not FullSupportSlatePolicy:
            raise TypeError("allocator must be exact FullSupportSlatePolicy")

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _FULL_SUPPORT_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
        )


@dataclass(slots=True)
class PydanticAITargetConditionedCalibratedPortfolioSelectionPolicy:
    """Portable T-RAP allocation over one authenticated model-proposed K8."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: TargetConditionedSlateAllocatorAdapter

    policy_id: ClassVar[str] = (
        TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID
    )
    policy_version: ClassVar[int] = (
        TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not TargetConditionedSlateAllocatorAdapter:
            raise TypeError("allocator must be exact target-conditioned adapter")
        self.allocator.__post_init__()

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _TARGET_CONDITIONED_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
        )


@dataclass(slots=True)
class PydanticAIConstraintDecoupledTargetConditionedPortfolioSelectionPolicy:
    """Semantic K8 hypotheses reconciled before portable T-RAP allocation."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: TargetConditionedSlateAllocatorAdapter

    policy_id: ClassVar[str] = (
        CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_ID
    )
    policy_version: ClassVar[int] = (
        CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not TargetConditionedSlateAllocatorAdapter:
            raise TypeError("allocator must be exact target-conditioned adapter")
        self.allocator.__post_init__()

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
            constraint_decoupled=True,
        )


@dataclass(slots=True)
class PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy:
    """Distinct adapter for model-anchor retention plus calibrated K4 fill."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: ModelAnchoredCalibratedSlatePolicy = ModelAnchoredCalibratedSlatePolicy()

    policy_id: ClassVar[str] = MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID
    policy_version: ClassVar[int] = (
        MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not ModelAnchoredCalibratedSlatePolicy:
            raise TypeError(
                "allocator must be exact ModelAnchoredCalibratedSlatePolicy"
            )
        self.allocator.__post_init__()

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _MODEL_ANCHORED_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
        )


@dataclass(slots=True)
class PydanticAIStructuralPosteriorCalibratedPortfolioSelectionPolicy:
    """Distinct adapter for prior-only structural-posterior K8-to-K4 allocation."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: StructuralPosteriorSlatePolicy = StructuralPosteriorSlatePolicy()

    policy_id: ClassVar[str] = (
        STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID
    )
    policy_version: ClassVar[int] = (
        STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not StructuralPosteriorSlatePolicy:
            raise TypeError("allocator must be exact StructuralPosteriorSlatePolicy")
        self.allocator.__post_init__()

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _STRUCTURAL_POSTERIOR_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
        )


@dataclass(slots=True)
class PydanticAIOperatorStratifiedCalibratedPortfolioSelectionPolicy:
    """K8-to-K4 adapter with authenticated operator-assay exposure."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: OperatorStratifiedStructuralPosteriorSlatePolicy = (
        OperatorStratifiedStructuralPosteriorSlatePolicy()
    )

    policy_id: ClassVar[str] = (
        OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID
    )
    policy_version: ClassVar[int] = (
        OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not (
            OperatorStratifiedStructuralPosteriorSlatePolicy
        ):
            raise TypeError("allocator must be exact operator-stratified policy")
        self.allocator.__post_init__()

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _OPERATOR_STRATIFIED_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
        )


@dataclass(slots=True)
class PydanticAIHorizonBoundedCalibratedPortfolioSelectionPolicy:
    """K8-to-K4 adapter with authenticated finite-horizon exposure bounds."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: HorizonBoundedStructuralPosteriorSlatePolicy

    policy_id: ClassVar[str] = HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID
    policy_version: ClassVar[int] = (
        HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise TypeError("allocator must be exact horizon-bounded policy")
        self.allocator.__post_init__()

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _HORIZON_BOUNDED_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
        )


@dataclass(slots=True)
class PydanticAIConstraintDecoupledHorizonPortfolioSelectionPolicy:
    """Semantic K8 suggestions reconciled by the engine before K4 allocation."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: HorizonBoundedStructuralPosteriorSlatePolicy

    policy_id: ClassVar[str] = (
        CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_ID
    )
    policy_version: ClassVar[int] = (
        CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise TypeError("allocator must be exact horizon-bounded policy")
        self.allocator.__post_init__()

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _CONSTRAINT_DECOUPLED_HORIZON_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
            constraint_decoupled=True,
        )


@dataclass(slots=True)
class PydanticAIMinimumInterventionHorizonPortfolioSelectionPolicy:
    """Model-semantic K8 projected to feasibility with minimum intervention."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: HorizonBoundedStructuralPosteriorSlatePolicy

    policy_id: ClassVar[str] = (
        MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_ID
    )
    policy_version: ClassVar[int] = (
        MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise TypeError("allocator must be exact horizon-bounded policy")
        self.allocator.__post_init__()

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _MINIMUM_INTERVENTION_HORIZON_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
            constraint_decoupled=True,
            minimum_intervention_projection=True,
        )


@dataclass(slots=True)
class PydanticAIEvidenceCalibratedSourceMixPortfolioSelectionPolicy:
    """Protect one outcome-blind global source before calibrated K4 allocation."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: HorizonBoundedStructuralPosteriorSlatePolicy

    policy_id: ClassVar[str] = (
        EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_ID
    )
    policy_version: ClassVar[int] = (
        EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise TypeError("allocator must be exact horizon-bounded policy")
        self.allocator.__post_init__()

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _EVIDENCE_CALIBRATED_SOURCE_MIX_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
            constraint_decoupled=True,
            minimum_intervention_projection=True,
            evidence_calibrated_source_mix=True,
        )


@dataclass(slots=True)
class PydanticAIContextualSearchAllocationPortfolioSelectionPolicy:
    """Enforce one authenticated prior-only source/operator allocation slice."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: HorizonBoundedStructuralPosteriorSlatePolicy

    policy_id: ClassVar[str] = (
        CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_ID
    )
    policy_version: ClassVar[int] = (
        CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise TypeError("allocator must be exact horizon-bounded policy")
        self.allocator.__post_init__()

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _CONTEXTUAL_SEARCH_ALLOCATION_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
            constraint_decoupled=True,
            minimum_intervention_projection=True,
            evidence_calibrated_source_mix=True,
            contextual_search_allocation=True,
        )


@dataclass(slots=True)
class PydanticAIAcquisitionCertifiedResidualPortfolioSelectionPolicy:
    """Reserve an optimizer reference K4 and certify every residual substitution."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: AcquisitionCertifiedSlatePolicy

    policy_id: ClassVar[str] = (
        ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_ID
    )
    policy_version: ClassVar[int] = (
        ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not AcquisitionCertifiedSlatePolicy:
            raise TypeError("allocator must be exact acquisition-certified policy")
        self.allocator.__post_init__()

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _ACQUISITION_CERTIFIED_RESIDUAL_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
            constraint_decoupled=True,
        )


@dataclass(slots=True)
class PydanticAIRegretBoundedInformationPortfolioSelectionPolicy:
    """Select residuals only inside an authenticated acquisition-regret envelope."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: RegretBoundedSlatePolicy

    policy_id: ClassVar[str] = REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_ID
    policy_version: ClassVar[int] = (
        REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not RegretBoundedSlatePolicy:
            raise TypeError("allocator must be exact regret-bounded policy")
        self.allocator.__post_init__()

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _REGRET_BOUNDED_INFORMATION_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
            constraint_decoupled=True,
        )


@dataclass(slots=True)
class PydanticAIFrontierProbeCalibratedPortfolioSelectionPolicy:
    """Live adapter for model anchors plus a full-abstention frontier probe."""

    generate_once: LowLevelRunner
    binding_for: CalibratedPortfolioBindingProvider
    allocator: FrontierProbeSlatePolicy = FrontierProbeSlatePolicy()

    policy_id: ClassVar[str] = FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID
    policy_version: ClassVar[int] = (
        FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION
    )
    policy_definition_sha256: ClassVar[str] = (
        FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not callable(self.binding_for):
            raise TypeError("binding_for must be callable")
        if type(self.allocator) is not FrontierProbeSlatePolicy:
            raise TypeError("allocator must be exact FrontierProbeSlatePolicy")
        self.allocator.__post_init__()

    @property
    def composition_identity_sha256(self) -> str:
        self.__post_init__()
        return _composition_identity_sha256(
            _FRONTIER_PROBE_PROFILE,
            self.allocator.to_record(),
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        return await _select_calibrated_portfolio(
            generate_once=self.generate_once,
            binding_for=self.binding_for,
            allocator=self.allocator,
            request=request,
        )


async def _select_calibrated_portfolio(
    *,
    generate_once: LowLevelRunner,
    binding_for: CalibratedPortfolioBindingProvider,
    allocator: CalibratedPortfolioAllocator,
    request: PortfolioSelectionRequest,
    constraint_decoupled: bool = False,
    minimum_intervention_projection: bool = False,
    evidence_calibrated_source_mix: bool = False,
    contextual_search_allocation: bool = False,
) -> PortfolioSelectionResult:
    profile = _profile_for_allocator_authority(
        allocator,
        constraint_decoupled=constraint_decoupled,
        minimum_intervention_projection=minimum_intervention_projection,
        evidence_calibrated_source_mix=evidence_calibrated_source_mix,
        contextual_search_allocation=contextual_search_allocation,
    )
    if type(request) is not PortfolioSelectionRequest:
        raise TypeError("request must be exact PortfolioSelectionRequest")
    request.__post_init__()
    # Resolve and authenticate the complete engine-owned input snapshot
    # before the provider sees either the prompt or its dynamic schema.
    binding = binding_for(request)
    _validate_binding_for_request(
        request,
        binding,
        selector_policy_definition_sha256=profile.policy_definition_sha256,
        constraint_decoupled=profile.constraint_decoupled,
        contextual_search_allocation=profile.contextual_search_allocation,
    )
    context = binding.context
    option_ids = (
        tuple(option.option_id for option in request.finite_variation_contract.options)
        if binding.common_candidate_pool is None
        else binding.common_candidate_pool.option_ids
    )
    required_evaluation_family_bounds: tuple[tuple[str, int, int], ...] = ()
    if type(allocator) is HorizonBoundedStructuralPosteriorSlatePolicy:
        active_phase = allocator.exposure_phase_for_wave(context.wave_index)
        required_evaluation_family_bounds = tuple(
            (
                value.family,
                value.minimum_evaluations,
                value.maximum_evaluations,
            )
            for value in active_phase.bounds
        )
        if request.require_pairwise_disjoint_parent_patches:
            required_evaluation_family_bounds = (
                project_family_exposure_bounds_to_pairwise_disjoint_feasibility(
                    request.finite_variation_contract,
                    option_ids,
                    portfolio_size=request.portfolio_size,
                    min_distinct_families=request.min_distinct_families,
                    requested_bounds=required_evaluation_family_bounds,
                )
            )
    output_type = _calibrated_output_type(
        request,
        binding,
        required_evaluation_family_bounds=required_evaluation_family_bounds,
        constraint_decoupled=profile.constraint_decoupled,
    )
    hierarchy = _hierarchical_composition_shape(
        request.finite_variation_contract,
        allowed_option_ids=option_ids,
    )
    low_level_request = StructuredGenerationRequest(
        call_id=request.call_id,
        operation=request.operation,
        prompt=render_calibrated_portfolio_selection_prompt_for_allocator(
            request,
            binding,
            allocator,
            constraint_decoupled=profile.constraint_decoupled,
            minimum_intervention_projection=(profile.minimum_intervention_projection),
            evidence_calibrated_source_mix=(profile.evidence_calibrated_source_mix),
            contextual_search_allocation=(profile.contextual_search_allocation),
        ),
        output_type=output_type,
        output_tool_name=CALIBRATED_PORTFOLIO_SELECTION_TOOL_NAME,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        repair_literal_sets=_bounded_option_repair_literal_sets(
            option_ids,
            hierarchy,
        ),
    )
    raw = await generate_once(low_level_request)
    response, attempt_count = _validated_response(raw, output_type=output_type)
    value = cast(Any, response.value)
    output_members = tuple(value.members)
    original_model_proposal_record = _raw_proposal_record(
        request,
        context,
        output_members,
    )
    model_members = _typed_model_members(original_model_proposal_record)
    acquisition_reference_option_ids = (
        _acquisition_certification_reference_option_ids(request, allocator)
    )
    reconciliation_receipt = None
    if profile.constraint_decoupled:
        model_members, reconciliation_receipt = _reconcile_semantic_members(
            request,
            binding,
            model_members,
            required_evaluation_family_bounds=(required_evaluation_family_bounds),
            original_model_proposal_sha256=_proposal_sha256(
                original_model_proposal_record
            ),
            minimum_intervention_projection=(profile.minimum_intervention_projection),
            evidence_calibrated_source_mix=(profile.evidence_calibrated_source_mix),
            contextual_search_allocation=(profile.contextual_search_allocation),
            acquisition_certification_reference_option_ids=(
                acquisition_reference_option_ids
            ),
        )
        proposal_record = _typed_proposal_record(
            request,
            context,
            model_members,
        )
    else:
        proposal_record = original_model_proposal_record
    proposal_sha256 = _proposal_sha256(proposal_record)
    slate, model_member_by_option = _build_calibrated_slate(
        request,
        binding,
        model_members,
        proposal_sha256=proposal_sha256,
    )
    allocation = _allocate_slate(
        request,
        binding,
        slate,
        allocator,
        required_option_ids=_required_allocation_option_ids(
            request,
            profile,
            reconciliation_receipt,
        ),
    )
    decision = _resolve_ranked_decision(
        request,
        allocation,
        model_member_by_option,
        profile=profile,
    )
    payload = freeze_json(
        _audit_payload_record(
            request=request,
            binding=binding,
            proposal_record=proposal_record,
            proposal_sha256=proposal_sha256,
            slate=slate,
            allocation=allocation,
            allocator=allocator,
            profile=profile,
            decision=decision,
            original_model_proposal_record=(
                original_model_proposal_record if profile.constraint_decoupled else None
            ),
            reconciliation_receipt=reconciliation_receipt,
        )
    )
    if type(payload) is not FrozenJsonObject:  # pragma: no cover - fixed object.
        raise AssertionError("calibrated audit did not freeze as an object")
    audit = PortfolioSelectionSupplementalAudit(
        audit_kind=profile.audit_kind,
        request_sha256=request.request_sha256,
        decision_sha256=decision.decision_sha256,
        payload=payload,
    )
    return PortfolioSelectionResult(
        decision=decision,
        telemetry=_telemetry(response, attempt_count=attempt_count),
        supplemental_audit=audit,
    )


__all__ = [
    "ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_ID",
    "ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_VERSION",
    "REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_ID",
    "REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_VERSION",
    "CALIBRATED_PORTFOLIO_BOUNDED_DOSE_COMMON_POOL_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_BOUNDED_DOSE_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_BOUNDED_DOSE_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_BOUNDED_DOSE_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_EVALUATION_SIZE",
    "CALIBRATED_PORTFOLIO_BASE_INSTRUCTION",
    "CALIBRATED_PORTFOLIO_COMMON_POOL_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_PROJECTED_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_PROJECTED_COMMON_POOL_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_PROJECTED_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_COMMON_POOL_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_HIDDEN_WITNESS_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_PROJECTED_BOUNDED_DOSE_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_PROJECTED_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_REQUEST_KEYED_WITNESS_PROMPT_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_PROPOSAL_SIZE",
    "CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID",
    "CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION",
    "CALIBRATED_PORTFOLIO_SELECTION_TOOL_NAME",
    "CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_ID",
    "CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_VERSION",
    "CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_ID",
    "CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_VERSION",
    "CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_ID",
    "CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_VERSION",
    "EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_ID",
    "EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_VERSION",
    "MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_ID",
    "MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_VERSION",
    "MAX_INLINE_OPTION_ENUM_UTF8_BYTES",
    "FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID",
    "FRONTIER_PROBE_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION",
    "FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID",
    "FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION",
    "HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID",
    "HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION",
    "MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID",
    "MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION",
    "OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID",
    "OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION",
    "STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID",
    "STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION",
    "TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_ID",
    "TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_VERSION",
    "CalibratedPortfolioAllocationDecision",
    "CalibratedPortfolioAllocator",
    "CalibratedPortfolioFeasibilityWitnessMode",
    "CalibratedPortfolioModelMember",
    "CalibratedPortfolioModelPrediction",
    "DecodedCalibratedPortfolioAudit",
    "DecodedCalibratedPortfolioProposal",
    "PydanticAICalibratedPortfolioSelectionPolicy",
    "PydanticAIAcquisitionCertifiedResidualPortfolioSelectionPolicy",
    "PydanticAIRegretBoundedInformationPortfolioSelectionPolicy",
    "PydanticAIConstraintDecoupledHorizonPortfolioSelectionPolicy",
    "PydanticAIConstraintDecoupledTargetConditionedPortfolioSelectionPolicy",
    "PydanticAIContextualSearchAllocationPortfolioSelectionPolicy",
    "PydanticAIEvidenceCalibratedSourceMixPortfolioSelectionPolicy",
    "PydanticAIMinimumInterventionHorizonPortfolioSelectionPolicy",
    "PydanticAIFullSupportCalibratedPortfolioSelectionPolicy",
    "PydanticAIFrontierProbeCalibratedPortfolioSelectionPolicy",
    "PydanticAIHorizonBoundedCalibratedPortfolioSelectionPolicy",
    "PydanticAIModelAnchoredCalibratedPortfolioSelectionPolicy",
    "PydanticAIOperatorStratifiedCalibratedPortfolioSelectionPolicy",
    "PydanticAIStructuralPosteriorCalibratedPortfolioSelectionPolicy",
    "PydanticAITargetConditionedCalibratedPortfolioSelectionPolicy",
    "SemanticSlateMemberOrigin",
    "SemanticSlateReconciledMember",
    "SemanticSlateReconciliationReceipt",
    "allocate_calibrated_portfolio_proposal",
    "calibrated_portfolio_prompt_definition_sha256",
    "decode_calibrated_portfolio_audit",
    "decode_calibrated_portfolio_proposal",
    "render_calibrated_portfolio_selection_prompt",
    "render_calibrated_portfolio_selection_prompt_for_allocator",
]
