"""Workload-neutral composition for K-slate acquisition experiments.

Workload launchers choose an explicit mode; this module owns the matching
allocator and outer selector identity.  Keeping that decision out of workload
code prevents Airfoil, BOiLS, Heat, and Timeloop adapters from silently giving
the same method name to different K8-to-K4 behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_selection import (
    ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
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
from agent_evolve.policies.selection.acquisition_certified_slate import (
    AcquisitionCertifiedSlateContextProvider,
    AcquisitionCertifiedSlatePolicy,
)
from agent_evolve.policies.selection.regret_bounded_slate import (
    RegretBoundedSlatePolicy,
    ResidualInformationAssayValuePolicy,
    SlateFutureValuePolicy,
    ZeroSlateFutureValuePolicy,
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
from agent_evolve.ports.finite_acquisition_batch import (
    FiniteAcquisitionBatchScorePolicy,
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
    ACQUISITION_CERTIFIED = "acquisition_certified"
    REGRET_BOUNDED_INFORMATION = "regret_bounded_information"


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
REGRET_MINIMUM_ACQUISITION_RETENTION_RATIO_ENV = (
    "AGENT_EVOLVE_REGRET_MINIMUM_ACQUISITION_RETENTION_RATIO"
)
REGRET_MINIMUM_RESIDUAL_AUDIT_MEMBERS_ENV = (
    "AGENT_EVOLVE_REGRET_MINIMUM_RESIDUAL_AUDIT_MEMBERS"
)
REGRET_RESIDUAL_ASSAY_VALUE_ENV = "AGENT_EVOLVE_REGRET_RESIDUAL_ASSAY_VALUE"
REGRET_ALLOW_DEVELOPMENT_ASSAY_ENV = (
    "AGENT_EVOLVE_REGRET_ALLOW_DEVELOPMENT_ASSAY"
)
REGRET_CALIBRATION_ERROR_BOUND_ENV = (
    "AGENT_EVOLVE_REGRET_CALIBRATION_ERROR_BOUND"
)


def _finite_environment_float(
    environ: Mapping[str, str],
    name: str,
    *,
    default: float | None,
) -> float | None:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a finite decimal number") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite decimal number")
    return value


@dataclass(frozen=True, slots=True)
class RegretBoundedInformationControls:
    """Workload-neutral, explicit authority for one RBIE treatment arm."""

    minimum_acquisition_retention_ratio: float
    minimum_residual_audit_members: int
    future_value_policy: SlateFutureValuePolicy
    calibration_error_bound: float | None
    allow_development_assay: bool

    def __post_init__(self) -> None:
        if (
            type(self.minimum_acquisition_retention_ratio) is not float
            or not math.isfinite(self.minimum_acquisition_retention_ratio)
            or not 0.0 < self.minimum_acquisition_retention_ratio <= 1.0
        ):
            raise ValueError("regret retention ratio must lie in (0, 1]")
        if (
            type(self.minimum_residual_audit_members) is not int
            or self.minimum_residual_audit_members < 0
        ):
            raise ValueError("minimum residual audit members must be non-negative")
        if not isinstance(self.future_value_policy, SlateFutureValuePolicy):
            raise TypeError("future_value_policy must implement its exact port")
        if self.calibration_error_bound is not None and (
            type(self.calibration_error_bound) is not float
            or not math.isfinite(self.calibration_error_bound)
            or self.calibration_error_bound < 0.0
        ):
            raise ValueError("regret calibration error must be non-negative")
        if type(self.allow_development_assay) is not bool:
            raise TypeError("allow_development_assay must be exact")
        is_assay = type(self.future_value_policy) is ResidualInformationAssayValuePolicy
        if is_assay != self.allow_development_assay:
            raise ValueError(
                "development assay authority and assay future-value policy must agree"
            )


def campaign_regret_bounded_information_controls_from_environment(
    environ: Mapping[str, str],
) -> RegretBoundedInformationControls:
    """Read a safe-by-default RBIE arm without workload-specific knowledge."""

    ratio = _finite_environment_float(
        environ,
        REGRET_MINIMUM_ACQUISITION_RETENTION_RATIO_ENV,
        default=1.0,
    )
    assert ratio is not None
    residual_audit_raw = environ.get(
        REGRET_MINIMUM_RESIDUAL_AUDIT_MEMBERS_ENV,
        "0",
    )
    if not residual_audit_raw.isascii() or not residual_audit_raw.isdigit():
        raise ValueError(
            f"{REGRET_MINIMUM_RESIDUAL_AUDIT_MEMBERS_ENV} must contain decimal digits"
        )
    residual_audit_members = int(residual_audit_raw)
    assay_value = _finite_environment_float(
        environ,
        REGRET_RESIDUAL_ASSAY_VALUE_ENV,
        default=None,
    )
    calibration_error = _finite_environment_float(
        environ,
        REGRET_CALIBRATION_ERROR_BOUND_ENV,
        default=None,
    )
    allow_raw = environ.get(REGRET_ALLOW_DEVELOPMENT_ASSAY_ENV, "0")
    if allow_raw not in {"0", "1"}:
        raise ValueError(
            f"{REGRET_ALLOW_DEVELOPMENT_ASSAY_ENV} must be exactly 0 or 1"
        )
    allow_assay = allow_raw == "1"
    if assay_value is not None and assay_value <= 0.0:
        raise ValueError(f"{REGRET_RESIDUAL_ASSAY_VALUE_ENV} must be positive")
    future_value_policy: SlateFutureValuePolicy = (
        ZeroSlateFutureValuePolicy()
        if assay_value is None
        else ResidualInformationAssayValuePolicy(float(assay_value))
    )
    return RegretBoundedInformationControls(
        minimum_acquisition_retention_ratio=float(ratio),
        minimum_residual_audit_members=residual_audit_members,
        future_value_policy=future_value_policy,
        calibration_error_bound=(
            None if calibration_error is None else float(calibration_error)
        ),
        allow_development_assay=allow_assay,
    )


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
    acquisition_certification_context_provider: (
        AcquisitionCertifiedSlateContextProvider | None
    ) = None,
    acquisition_batch_scorer: FiniteAcquisitionBatchScorePolicy | None = None,
    regret_minimum_acquisition_retention_ratio: float = 1.0,
    regret_minimum_residual_audit_members: int = 0,
    regret_future_value_policy: SlateFutureValuePolicy | None = None,
    regret_calibration_error_bound: float | None = None,
    regret_allow_development_assay: bool = False,
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
    if mode in {
        CampaignAcquisitionMode.ACQUISITION_CERTIFIED,
        CampaignAcquisitionMode.REGRET_BOUNDED_INFORMATION,
    }:
        if not common_pool_enabled:
            raise ValueError(
                "acquisition-certified K8-to-K4 allocation requires a common pool"
            )
        if not isinstance(
            acquisition_certification_context_provider,
            AcquisitionCertifiedSlateContextProvider,
        ):
            raise TypeError(
                "acquisition-certified mode requires its context provider"
            )
        if not isinstance(acquisition_batch_scorer, FiniteAcquisitionBatchScorePolicy):
            raise TypeError("acquisition-certified mode requires its batch scorer")
        if mode is CampaignAcquisitionMode.ACQUISITION_CERTIFIED:
            if (
                regret_minimum_acquisition_retention_ratio != 1.0
                or regret_minimum_residual_audit_members != 0
                or regret_future_value_policy is not None
                or regret_calibration_error_bound is not None
                or regret_allow_development_assay
            ):
                raise ValueError("regret controls require regret-bounded mode")
            return AcquisitionCertifiedSlatePolicy(
                context_provider=acquisition_certification_context_provider,
                scorer=acquisition_batch_scorer,
            )
        return RegretBoundedSlatePolicy(
            context_provider=acquisition_certification_context_provider,
            scorer=acquisition_batch_scorer,
            future_value_policy=(
                ZeroSlateFutureValuePolicy()
                if regret_future_value_policy is None
                else regret_future_value_policy
            ),
            minimum_acquisition_retention_ratio=(
                regret_minimum_acquisition_retention_ratio
            ),
            minimum_residual_audit_members=(
                regret_minimum_residual_audit_members
            ),
            calibration_error_bound=regret_calibration_error_bound,
            allow_development_assay=regret_allow_development_assay,
        )
    if (
        target_conditioned_profile is not None
        or target_conditioned_context_provider is not None
    ):
        raise ValueError(
            "target-conditioned components are valid only for their exact mode"
        )
    if (
        acquisition_certification_context_provider is not None
        or acquisition_batch_scorer is not None
    ):
        raise ValueError(
            "certified-acquisition components are valid only for their exact mode"
        )
    if (
        regret_minimum_acquisition_retention_ratio != 1.0
        or regret_minimum_residual_audit_members != 0
        or regret_future_value_policy is not None
        or regret_calibration_error_bound is not None
        or regret_allow_development_assay
    ):
        raise ValueError("regret controls require regret-bounded mode")
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
        if type(allocator) is AcquisitionCertifiedSlatePolicy:
            return (
                ACQUISITION_CERTIFIED_RESIDUAL_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
            )
        if type(allocator) is RegretBoundedSlatePolicy:
            return (
                REGRET_BOUNDED_INFORMATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
            )
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
    "REGRET_ALLOW_DEVELOPMENT_ASSAY_ENV",
    "REGRET_CALIBRATION_ERROR_BOUND_ENV",
    "REGRET_MINIMUM_ACQUISITION_RETENTION_RATIO_ENV",
    "REGRET_RESIDUAL_ASSAY_VALUE_ENV",
    "RegretBoundedInformationControls",
    "build_campaign_acquisition_allocator",
    "build_campaign_proposal_support_policy",
    "campaign_constraint_decoupled_acquisition_from_environment",
    "campaign_contextual_search_allocation_from_environment",
    "campaign_evidence_calibrated_source_mix_from_environment",
    "campaign_minimum_intervention_projection_from_environment",
    "campaign_operator_assay_minimum_from_environment",
    "campaign_residual_frontier_planning_from_environment",
    "campaign_regret_bounded_information_controls_from_environment",
    "campaign_selector_policy_definition_sha256",
]
