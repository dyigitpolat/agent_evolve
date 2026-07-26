"""Workload-neutral composition for K-slate acquisition experiments.

Workload launchers choose an explicit mode; this module owns the matching
allocator and outer selector identity.  Keeping that decision out of workload
code prevents Airfoil, BOiLS, Heat, and Timeloop adapters from silently giving
the same method name to different K8-to-K4 behavior.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping

from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_selection import (
    CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    CalibratedPortfolioAllocator,
)
from agent_evolve.policies.selection.full_support_slate import FullSupportSlatePolicy
from agent_evolve.policies.selection.model_anchored_slate import (
    ModelAnchoredCalibratedSlatePolicy,
)
from agent_evolve.policies.selection.structural_posterior_slate import (
    FamilyExposurePhase,
    HorizonBoundedStructuralPosteriorSlatePolicy,
    OperatorStratifiedStructuralPosteriorSlatePolicy,
    StructuralPosteriorSlatePolicy,
)
from agent_evolve.policies.selection.proposal_support import (
    StructuralProposalSupportPolicy,
)
from agent_evolve.policies.selection.target_conditioned_allocator import (
    RegisteredTargetConditionedAllocationContextProvider,
    TargetConditionedAllocationContextProvider,
    TargetConditionedSlateAllocatorAdapter,
)
from agent_evolve.policies.selection.target_conditioned_prequential import (
    TargetConditionedAcquisitionProfile,
)
from agent_evolve.policies.variation.compositional_finite_catalog import (
    COMPOSITE_OPTION_FAMILY,
)


class CampaignAcquisitionMode(str, Enum):
    """Closed acquisition treatments shared by every workload launcher."""

    FULL_SUPPORT = "full_support"
    MODEL_TOP_K = "model_top_k"
    CALIBRATED_FRONTIER = "calibrated_frontier"
    HIERARCHICAL_SUPPORT = "hierarchical_support"
    OPERATOR_STRATIFIED = "operator_stratified"
    HORIZON_BOUNDED = "horizon_bounded"
    TARGET_CONDITIONED = "target_conditioned"


OPERATOR_ASSAY_MINIMUM_ENV = "AGENT_EVOLVE_OPERATOR_ASSAY_MINIMUM"
CONSTRAINT_DECOUPLED_ACQUISITION_ENV = (
    "AGENT_EVOLVE_CONSTRAINT_DECOUPLED_ACQUISITION"
)
MINIMUM_INTERVENTION_PROJECTION_ENV = (
    "AGENT_EVOLVE_MINIMUM_INTERVENTION_PROJECTION"
)
EVIDENCE_CALIBRATED_SOURCE_MIX_ENV = (
    "AGENT_EVOLVE_EVIDENCE_CALIBRATED_SOURCE_MIX"
)
CONTEXTUAL_SEARCH_ALLOCATION_ENV = "AGENT_EVOLVE_CONTEXTUAL_SEARCH_ALLOCATION"
RESIDUAL_FRONTIER_PLANNING_ENV = "AGENT_EVOLVE_RESIDUAL_FRONTIER_PLANNING"


def campaign_constraint_decoupled_acquisition_from_environment(
    environ: Mapping[str, str],
) -> bool:
    """Read the shared opt-in for engine-reconciled semantic proposals."""

    raw = environ.get(CONSTRAINT_DECOUPLED_ACQUISITION_ENV, "0")
    if raw not in {"0", "1"}:
        raise ValueError(
            f"{CONSTRAINT_DECOUPLED_ACQUISITION_ENV} must be exactly 0 or 1"
        )
    return raw == "1"


def campaign_minimum_intervention_projection_from_environment(
    environ: Mapping[str, str],
) -> bool:
    """Read the shared opt-in for minimum-intervention semantic projection."""

    raw = environ.get(MINIMUM_INTERVENTION_PROJECTION_ENV, "0")
    if raw not in {"0", "1"}:
        raise ValueError(
            f"{MINIMUM_INTERVENTION_PROJECTION_ENV} must be exactly 0 or 1"
        )
    return raw == "1"


def campaign_evidence_calibrated_source_mix_from_environment(
    environ: Mapping[str, str],
) -> bool:
    """Read the shared opt-in for a protected global proposal source."""

    raw = environ.get(EVIDENCE_CALIBRATED_SOURCE_MIX_ENV, "0")
    if raw not in {"0", "1"}:
        raise ValueError(
            f"{EVIDENCE_CALIBRATED_SOURCE_MIX_ENV} must be exactly 0 or 1"
        )
    return raw == "1"


def campaign_contextual_search_allocation_from_environment(
    environ: Mapping[str, str],
) -> bool:
    """Read the shared opt-in for prior-only source/operator allocation."""

    raw = environ.get(CONTEXTUAL_SEARCH_ALLOCATION_ENV, "0")
    if raw not in {"0", "1"}:
        raise ValueError(
            f"{CONTEXTUAL_SEARCH_ALLOCATION_ENV} must be exactly 0 or 1"
        )
    return raw == "1"


def campaign_residual_frontier_planning_from_environment(
    environ: Mapping[str, str],
) -> bool:
    """Read the shared opt-in for joint residual-cell search planning."""

    raw = environ.get(RESIDUAL_FRONTIER_PLANNING_ENV, "0")
    if raw not in {"0", "1"}:
        raise ValueError(
            f"{RESIDUAL_FRONTIER_PLANNING_ENV} must be exactly 0 or 1"
        )
    return raw == "1"


def campaign_operator_assay_minimum_from_environment(
    environ: Mapping[str, str],
) -> int:
    """Read the workload-neutral evaluator exposure floor."""

    raw = environ.get(OPERATOR_ASSAY_MINIMUM_ENV, "1")
    if not raw.isascii() or not raw.isdigit():
        raise ValueError(f"{OPERATOR_ASSAY_MINIMUM_ENV} must be decimal digits")
    value = int(raw)
    if not 1 <= value <= 4:
        raise ValueError(f"{OPERATOR_ASSAY_MINIMUM_ENV} must lie in [1, 4]")
    return value


def build_campaign_acquisition_allocator(
    mode: CampaignAcquisitionMode,
    *,
    common_pool_enabled: bool,
    operator_assay_minimum: int = 1,
    family_exposure_phases: tuple[FamilyExposurePhase, ...] | None = None,
    target_conditioned_profile: TargetConditionedAcquisitionProfile | None = None,
    target_conditioned_context_provider: (
        TargetConditionedAllocationContextProvider | None
    ) = None,
) -> CalibratedPortfolioAllocator:
    """Build one exact allocator without benchmark-specific parameters."""

    if type(mode) is not CampaignAcquisitionMode:
        raise TypeError("mode must be an exact CampaignAcquisitionMode")
    if type(common_pool_enabled) is not bool:
        raise TypeError("common_pool_enabled must be an exact bool")
    if (
        type(operator_assay_minimum) is not int
        or not 1 <= operator_assay_minimum <= 4
    ):
        raise ValueError("operator_assay_minimum must lie in [1, 4]")
    if (
        mode is not CampaignAcquisitionMode.OPERATOR_STRATIFIED
        and operator_assay_minimum != 1
    ):
        raise ValueError(
            "operator_assay_minimum is configurable only for operator_stratified"
        )
    if (
        mode is CampaignAcquisitionMode.HORIZON_BOUNDED
        and family_exposure_phases is None
    ):
        raise ValueError("horizon_bounded requires explicit family_exposure_phases")
    if (
        mode is not CampaignAcquisitionMode.HORIZON_BOUNDED
        and family_exposure_phases is not None
    ):
        raise ValueError(
            "family_exposure_phases are configurable only for horizon_bounded"
        )
    if mode is CampaignAcquisitionMode.TARGET_CONDITIONED:
        if not common_pool_enabled:
            raise ValueError(
                "target-conditioned K8-to-K4 acquisition requires a common pool"
            )
        if type(target_conditioned_profile) is not (
            TargetConditionedAcquisitionProfile
        ):
            raise TypeError("target-conditioned mode requires one exact profile")
        if target_conditioned_context_provider is None:
            target_conditioned_context_provider = (
                RegisteredTargetConditionedAllocationContextProvider()
            )
        return TargetConditionedSlateAllocatorAdapter(
            context_provider=target_conditioned_context_provider,
            profile=target_conditioned_profile,
        )
    if (
        target_conditioned_profile is not None
        or target_conditioned_context_provider is not None
    ):
        raise ValueError(
            "target-conditioned components are valid only for their exact mode"
        )
    if not common_pool_enabled:
        if mode is not CampaignAcquisitionMode.FULL_SUPPORT:
            raise ValueError(
                "K8-to-K4 acquisition requires the common candidate-pool contract"
            )
        return FullSupportSlatePolicy()
    if mode is CampaignAcquisitionMode.FULL_SUPPORT:
        raise ValueError("full-support K8 evaluation is incompatible with K4 width")
    if mode is CampaignAcquisitionMode.MODEL_TOP_K:
        return ModelAnchoredCalibratedSlatePolicy(model_anchor_count=4)
    if mode is CampaignAcquisitionMode.OPERATOR_STRATIFIED:
        return OperatorStratifiedStructuralPosteriorSlatePolicy(
            ((COMPOSITE_OPTION_FAMILY, operator_assay_minimum),)
        )
    if mode is CampaignAcquisitionMode.HORIZON_BOUNDED:
        assert family_exposure_phases is not None
        return HorizonBoundedStructuralPosteriorSlatePolicy(
            family_exposure_phases
        )
    return StructuralPosteriorSlatePolicy()


def build_campaign_proposal_support_policy(
    mode: CampaignAcquisitionMode,
    *,
    common_pool_enabled: bool,
) -> StructuralProposalSupportPolicy | None:
    """Build the proposal-support stage paired with one acquisition mode."""

    if type(mode) is not CampaignAcquisitionMode:
        raise TypeError("mode must be an exact CampaignAcquisitionMode")
    if type(common_pool_enabled) is not bool:
        raise TypeError("common_pool_enabled must be an exact bool")
    if mode in {
        CampaignAcquisitionMode.HIERARCHICAL_SUPPORT,
        CampaignAcquisitionMode.OPERATOR_STRATIFIED,
        CampaignAcquisitionMode.HORIZON_BOUNDED,
        CampaignAcquisitionMode.TARGET_CONDITIONED,
    }:
        if not common_pool_enabled:
            raise ValueError(
                "hierarchical proposal support requires the common candidate pool"
            )
        return StructuralProposalSupportPolicy()
    return None


def campaign_selector_policy_definition_sha256(
    allocator: CalibratedPortfolioAllocator,
    *,
    constraint_decoupled: bool = False,
    minimum_intervention_projection: bool = False,
    evidence_calibrated_source_mix: bool = False,
    contextual_search_allocation: bool = False,
) -> str:
    """Return the outer adapter identity paired with an exact allocator."""

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
            "contextual search allocation requires evidence-calibrated source mix"
        )
    if evidence_calibrated_source_mix and not minimum_intervention_projection:
        raise ValueError(
            "evidence-calibrated source mix requires minimum intervention"
        )
    if minimum_intervention_projection and not constraint_decoupled:
        raise ValueError(
            "minimum intervention requires constraint-decoupled acquisition"
        )
    if contextual_search_allocation:
        if type(allocator) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise ValueError(
                "contextual search allocation requires the horizon-bounded "
                "allocator"
            )
        return (
            CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        )
    if evidence_calibrated_source_mix:
        if type(allocator) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise ValueError(
                "evidence-calibrated source mix requires the horizon-bounded "
                "allocator"
            )
        return (
            EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        )
    if minimum_intervention_projection:
        if type(allocator) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise ValueError(
                "minimum intervention requires the horizon-bounded allocator"
            )
        return (
            MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        )
    if constraint_decoupled:
        if type(allocator) is TargetConditionedSlateAllocatorAdapter:
            return (
                CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
            )
        if type(allocator) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise ValueError(
                "constraint-decoupled acquisition requires the horizon-bounded "
                "or target-conditioned allocator"
            )
        return (
            CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        )

    if type(allocator) is FullSupportSlatePolicy:
        return FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    if type(allocator) is ModelAnchoredCalibratedSlatePolicy:
        return MODEL_ANCHORED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    if type(allocator) is StructuralPosteriorSlatePolicy:
        return (
            STRUCTURAL_POSTERIOR_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        )
    if type(allocator) is OperatorStratifiedStructuralPosteriorSlatePolicy:
        return (
            OPERATOR_STRATIFIED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        )
    if type(allocator) is HorizonBoundedStructuralPosteriorSlatePolicy:
        return (
            HORIZON_BOUNDED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        )
    if type(allocator) is TargetConditionedSlateAllocatorAdapter:
        return (
            TARGET_CONDITIONED_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
        )
    raise TypeError("allocator is not a supported campaign acquisition policy")


__all__ = [
    "CampaignAcquisitionMode",
    "CONSTRAINT_DECOUPLED_ACQUISITION_ENV",
    "CONTEXTUAL_SEARCH_ALLOCATION_ENV",
    "EVIDENCE_CALIBRATED_SOURCE_MIX_ENV",
    "MINIMUM_INTERVENTION_PROJECTION_ENV",
    "OPERATOR_ASSAY_MINIMUM_ENV",
    "RESIDUAL_FRONTIER_PLANNING_ENV",
    "build_campaign_acquisition_allocator",
    "build_campaign_proposal_support_policy",
    "campaign_constraint_decoupled_acquisition_from_environment",
    "campaign_contextual_search_allocation_from_environment",
    "campaign_evidence_calibrated_source_mix_from_environment",
    "campaign_minimum_intervention_projection_from_environment",
    "campaign_operator_assay_minimum_from_environment",
    "campaign_residual_frontier_planning_from_environment",
    "campaign_selector_policy_definition_sha256",
]
