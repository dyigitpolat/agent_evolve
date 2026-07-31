"""One authenticated AgentEvolve successor method across workloads and models.

Workload adapters provide evaluators, finite catalogs, objective semantics, and
optional semantic prompt extensions.  They must not silently choose different
optimizer policies.  This module owns the public identities of the reference
parent-selection, memory, acquisition, recombination, and reflection policies,
then binds workload-local runtime objects behind those identities.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum

from agent_evolve.application.evolution_campaign import (
    CampaignPolicyBinding,
    CampaignReflectionSupervisionPolicy,
    ReflectionFailureMode,
)
from agent_evolve.application.portfolio_campaign_runtime import (
    ResidualHypervolumeCampaignParentSelector,
    StagnationAwareDiverseCampaignParentSelector,
)
from agent_evolve.application.contextual_search_controller import (
    CONTEXTUAL_SEARCH_CONTROLLER_DEFINITION_SHA256,
)
from agent_evolve.application.contextual_delayed_credit import (
    CONTEXTUAL_DELAYED_CREDIT_DEFINITION_SHA256,
)
from agent_evolve.campaign_presets import (
    REFERENCE_36_OFFSPRING_SCALE_SHAPE,
    PortfolioScaleShape,
)
from agent_evolve.campaign_profiles import CampaignExperimentProfile
from agent_evolve.integrations.pydantic_ai.model_execution_profile import (
    OpenRouterModelExecutionProfile,
)
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_selection import (
    CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
)
from agent_evolve.policies.selection.structural_posterior_slate import (
    HorizonBoundedStructuralPosteriorSlatePolicy,
    OperatorStratifiedStructuralPosteriorSlatePolicy,
)
from agent_evolve.policies.selection.target_conditioned_allocator import (
    TargetConditionedSlateAllocatorAdapter,
)
from agent_evolve.policies.selection.affine_frontier_target import (
    ALLOCATOR_DEFINITION_SHA256 as AFFINE_FRONTIER_TARGET_DEFINITION_SHA256,
    DIRECTION_COVERED_ALLOCATOR_DEFINITION_SHA256,
)
from agent_evolve.policies.selection.residual_frontier_target import (
    RESIDUAL_TARGET_ALLOCATOR_DEFINITION_SHA256,
)
from agent_evolve.ports.archive_context import (
    CampaignPortfolioArchiveContextProjector,
)
from agent_evolve.workload_prompt import WorkloadPromptArm


REFERENCE_AGENT_EVOLVE_METHOD_ID = "agent_evolve_reference_successor"
REFERENCE_AGENT_EVOLVE_METHOD_VERSION = 2
HIERARCHICAL_AGENT_EVOLVE_METHOD_ID = "agent_evolve_hierarchical_successor"
HIERARCHICAL_AGENT_EVOLVE_METHOD_VERSION = 3
FRONTIER_AGENT_EVOLVE_METHOD_ID = "agent_evolve_frontier_successor"
FRONTIER_AGENT_EVOLVE_METHOD_VERSION = 3
FRONTIER_HIERARCHICAL_AGENT_EVOLVE_METHOD_ID = (
    "agent_evolve_frontier_hierarchical_successor"
)
FRONTIER_HIERARCHICAL_AGENT_EVOLVE_METHOD_VERSION = 4
CONTEXT_LOCAL_AGENT_EVOLVE_METHOD_ID = "agent_evolve_context_local_successor"
CONTEXT_LOCAL_AGENT_EVOLVE_METHOD_VERSION = 5
OPERATOR_STRATIFIED_AGENT_EVOLVE_METHOD_ID = (
    "agent_evolve_operator_stratified_successor"
)
OPERATOR_STRATIFIED_AGENT_EVOLVE_METHOD_VERSION = 6
HORIZON_BOUNDED_AGENT_EVOLVE_METHOD_ID = "agent_evolve_horizon_bounded_successor"
HORIZON_BOUNDED_AGENT_EVOLVE_METHOD_VERSION = 7
STAGNATION_AWARE_AGENT_EVOLVE_METHOD_ID = "agent_evolve_stagnation_aware_successor"
STAGNATION_AWARE_AGENT_EVOLVE_METHOD_VERSION = 8
CONSTRAINT_DECOUPLED_AGENT_EVOLVE_METHOD_ID = (
    "agent_evolve_constraint_decoupled_successor"
)
CONSTRAINT_DECOUPLED_AGENT_EVOLVE_METHOD_VERSION = 9
MINIMUM_INTERVENTION_AGENT_EVOLVE_METHOD_ID = (
    "agent_evolve_minimum_intervention_successor"
)
MINIMUM_INTERVENTION_AGENT_EVOLVE_METHOD_VERSION = 10
EVIDENCE_CALIBRATED_SOURCE_MIX_AGENT_EVOLVE_METHOD_ID = (
    "agent_evolve_evidence_calibrated_source_mix_successor"
)
EVIDENCE_CALIBRATED_SOURCE_MIX_AGENT_EVOLVE_METHOD_VERSION = 11
CONTEXTUAL_SEARCH_AGENT_EVOLVE_METHOD_ID = "agent_evolve_contextual_search_successor"
CONTEXTUAL_SEARCH_AGENT_EVOLVE_METHOD_VERSION = 21
RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_ID = (
    "agent_evolve_residual_frontier_successor"
)
RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_VERSION = 23
TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_ID = (
    "agent_evolve_target_conditioned_successor"
)
TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_VERSION = 17
CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_ID = (
    "agent_evolve_constraint_decoupled_target_conditioned_successor"
)
CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_VERSION = 19


class ReferenceCampaignPolicyRole(str, Enum):
    """Closed optimizer-policy roles shared by every reference workload."""

    PARENT_SELECTION = "parent_selection"
    MEMORY_ASSIGNMENT = "memory_assignment"
    PORTFOLIO_SELECTION = "portfolio_selection"
    RECOMBINATION = "recombination"
    REFLECTION = "reflection"
    ARCHIVE_CONTEXT = "archive_context"
    VARIATION_TOPOLOGY = "variation_topology"
    CONTEXTUAL_OUTCOMES = "contextual_outcomes"


_POLICY_IDS = {
    ReferenceCampaignPolicyRole.PARENT_SELECTION: (
        "archive_elite_explorer_common",
        b"agent-evolve:reference-parent-selection:v1;archive-elite-explorer",
    ),
    ReferenceCampaignPolicyRole.MEMORY_ASSIGNMENT: (
        "parent_conditioned_predictive_memory",
        b"agent-evolve:reference-memory:v2;authenticated-local-transition;"
        b"quarantine;matched-active-neutral-diagnostics;no-unidentified-credit",
    ),
    ReferenceCampaignPolicyRole.PORTFOLIO_SELECTION: (
        "complete_contract_model_k8_engine_k4",
        b"agent-evolve:reference-acquisition:v2;task-keyed-complete-finite-contract;"
        b"model-ranked-k8;engine-allocated-k4;hidden-feasibility-certificate",
    ),
    ReferenceCampaignPolicyRole.RECOMBINATION: (
        "typed_disjoint_patch_union",
        b"agent-evolve:reference-recombination:v1;archive-aware;typed-disjoint-union;"
        b"infeasibility-recourse",
    ),
    ReferenceCampaignPolicyRole.REFLECTION: (
        "delayed_identifiable_mutation_reflection",
        b"agent-evolve:reference-reflection:v2;sealed-direct-single-mutation;"
        b"delayed-quarantine;parent-conditioned-executable-semantics",
    ),
}

_HIERARCHICAL_PORTFOLIO_POLICY = (
    "hierarchical_support_structural_k8_engine_k4",
    b"agent-evolve:reference-acquisition:v3;task-keyed-complete-finite-contract;"
    b"model-ranked-k8;proposal-reservations=archive-novelty,structural-coverage;"
    b"reservations-force-evaluator-slots=false;"
    b"engine-allocator=calibrated-frontier-four-role-v2;"
    b"hidden-feasibility-certificate=true",
)

_OPERATOR_STRATIFIED_PORTFOLIO_POLICY = (
    "operator_stratified_hierarchical_k8_engine_k4",
    b"agent-evolve:reference-acquisition:v4;task-keyed-complete-finite-contract;"
    b"model-ranked-k8;hierarchical-r2-proposal-minimum=authenticated;"
    b"engine-k4-composite-family-minimum=1;"
    b"minimum-is-assay-exposure-not-quality-prior;"
    b"remaining-slots=calibrated-frontier-four-role-v2;"
    b"joint-feasibility-and-role-optimization=true;outcome-blind=true;"
    b"hidden-feasibility-certificate=true;workload-specific-parameters=none",
)

_HORIZON_BOUNDED_PORTFOLIO_POLICY = (
    "horizon_bounded_hierarchical_k8_engine_k4",
    b"agent-evolve:reference-acquisition:v5;task-keyed-complete-finite-contract;"
    b"model-ranked-k8;hierarchical-r2-proposal-minimum=authenticated;"
    b"engine-k4-family-exposure-phases=authenticated;"
    b"lower-and-upper-bounds=true;phase-index=sealed-wave-index;"
    b"bounds-are-assay-exposure-not-quality-priors;"
    b"infeasible-bounds=minimum-l1-structural-recourse;"
    b"remaining-slots=calibrated-frontier-four-role-v2;"
    b"joint-feasibility-and-role-optimization=true;outcome-blind=true;"
    b"hidden-feasibility-certificate=true;workload-specific-parameters=none",
)

_CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_POLICY = (
    "constraint_decoupled_horizon_k8_engine_k4",
    b"agent-evolve:reference-acquisition:v6;"
    b"model-authority=local-finite-semantic-preferences;"
    b"engine-authority=dedupe-support-composition-dose-feasibility-refill;"
    b"original-and-reconciled-proposals-authenticated=true;"
    b"engine-inserted-forecast=unknown;"
    b"engine-k4-family-exposure-phases=authenticated;"
    b"objective-values-consulted-by-reconciliation=false;"
    b"workload-specific-parameters=none",
)

_TARGET_CONDITIONED_PORTFOLIO_POLICY = (
    "target_conditioned_realizable_k8_engine_k4",
    b"agent-evolve:reference-acquisition:v12;"
    b"model-authority=finite-semantic-hypotheses;"
    b"engine-authority=target-conditioned-prequential-realizable-allocation;"
    b"context=append-only-authenticated-precall-branch;"
    b"features=portable-typed-configuration-transitions;"
    b"state=selected-only-generation-barrier;"
    b"workload-model-provider-current-outcome-fields=false",
)

_CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_POLICY = (
    "constraint_decoupled_target_conditioned_k8_engine_k4",
    b"agent-evolve:reference-acquisition:v14;"
    b"model-authority=local-finite-semantic-hypotheses;"
    b"engine-authority=dedupe-support-composition-dose-feasibility-refill;"
    b"allocation=target-conditioned-prequential-realizable-portfolio-v1;"
    b"evaluation-source-floor=sealed-contract-task-keyed;"
    b"independent-evaluation-does-not-require-disjoint-parent-patches=true;"
    b"original-and-reconciled-proposals-authenticated=true;"
    b"engine-inserted-forecast=unknown;"
    b"objective-values-consulted-by-reconciliation=false;"
    b"workload-model-provider-specific-parameters=none;"
    b"frontier-target="
    + DIRECTION_COVERED_ALLOCATOR_DEFINITION_SHA256.encode("ascii"),
)

_MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_POLICY = (
    "minimum_intervention_horizon_k8_engine_k4",
    b"agent-evolve:reference-acquisition:v7;"
    b"model-authority=local-finite-semantic-hypotheses;"
    b"engine-authority=dedupe-support-composition-dose-feasibility-refill;"
    b"projection=max-retained-model-count-then-model-rank;"
    b"canonical-order=final-tie-break-only;"
    b"original-reconciled-and-intervention-receipts-authenticated=true;"
    b"engine-inserted-forecast=unknown;"
    b"engine-k4-family-exposure-phases=authenticated;"
    b"objective-values-consulted-by-reconciliation=false;"
    b"workload-specific-parameters=none",
)

_EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_POLICY = (
    "evidence_calibrated_source_mix_horizon_k8_engine_k4",
    b"agent-evolve:reference-acquisition:v8;"
    b"model-authority=local-finite-semantic-hypotheses;"
    b"engine-authority=feasibility-soft-support-source-floor-allocation;"
    b"protected-source=task-keyed-global-feasibility-member;"
    b"protected-source-count=one;protected-source-phase=wave-one;"
    b"remaining-projection=max-retained-model-count-then-model-rank;"
    b"proposal-support-conflicts=deterministically-deferred;"
    b"required-allocation-membership-authenticated=true;"
    b"engine-k4-family-exposure-phases=authenticated;"
    b"objective-values-consulted-by-source-mix=false;"
    b"workload-specific-parameters=none",
)

_CONTEXTUAL_SEARCH_PORTFOLIO_POLICY = (
    "contextual_source_operator_horizon_k8_engine_k4",
    b"agent-evolve:reference-acquisition:v12;"
    b"model-authority=local-finite-semantic-hypotheses;"
    b"engine-authority=feasibility-and-authenticated-contextual-allocation;"
    b"allocation-marginals=sealed-finite-variation-source,atomic-vs-composite;"
    b"semantic-model-vs-engine-origin=separate-provenance;"
    b"allocation-evidence=prior-only-normalized-marginal-utility;"
    b"allocation-phase=finite-horizon-generic;"
    b"operator-envelope=full-k4-structurally-feasible-range;"
    b"requested-allocation-exact-when-feasible=true;"
    b"structural-recourse=authenticated-minimum-l1-projection;"
    b"requested-and-realized-allocation-authenticated=true;"
    b"allocation-capability=prior-realized-workload-blind-count-witnesses;"
    b"frontier-target=coordinated-affine-opportunity-directions;"
    b"realized-allocation-membership-authenticated=true;"
    b"objective-values-consulted-by-reconciliation=false;"
    b"workload-model-provider-specific-parameters=none;"
    + CONTEXTUAL_SEARCH_CONTROLLER_DEFINITION_SHA256.encode("ascii")
    + b";frontier-target="
    + AFFINE_FRONTIER_TARGET_DEFINITION_SHA256.encode("ascii")
    + b";selector="
    + CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256.encode(
        "ascii"
    ),
)

_RESIDUAL_FRONTIER_CONTEXTUAL_SEARCH_PORTFOLIO_POLICY = (
    "residual_frontier_contextual_source_operator_horizon_k8_engine_k4",
    b"agent-evolve:reference-acquisition:v13;"
    b"base=contextual-source-operator-horizon-k8-engine-k4;"
    b"parent-and-target=joint-largest-positive-residual-hypervolume-cell;"
    b"lane-directions=canonical-needed-improvement-axes;"
    b"fallback=prior-stagnation-aware-plus-global-direction-coverage;"
    b"objective-values=authenticated-prior-affine-archive-only;"
    b"workload-model-provider-specific-parameters=none;"
    + CONTEXTUAL_SEARCH_CONTROLLER_DEFINITION_SHA256.encode("ascii")
    + b";frontier-target="
    + RESIDUAL_TARGET_ALLOCATOR_DEFINITION_SHA256.encode("ascii")
    + b";selector="
    + CONTEXTUAL_SEARCH_ALLOCATION_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256.encode(
        "ascii"
    ),
)

_CONTEXT_LOCAL_MEMORY_POLICY = (
    "exact_context_predictive_memory",
    b"agent-evolve:reference-memory:v3;authenticated-local-transition;"
    b"exact-source-parent-forced-replay-only;nonexact-transfer-advisory;"
    b"quarantine;no-unidentified-credit",
)
_ATOMIC_VARIATION_TOPOLOGY_POLICY = (
    "parent_bound_atomic_finite_variation",
    b"agent-evolve:reference-variation-topology:v1;complete-parent-bound-finite-"
    b"atomic-support;engine-materialized;provider-cannot-invent-actions",
)
_HIERARCHICAL_R2_VARIATION_TOPOLOGY_POLICY_ID = (
    "bounded_hierarchical_r2_finite_variation"
)
_HIERARCHICAL_R2_VARIATION_TOPOLOGY_DOMAIN = (
    b"agent-evolve:reference-variation-topology:hierarchical-r2:v2\x00"
)
_CONTEXTUAL_OUTCOMES_POLICY = (
    "parent_local_prior_outcome_history",
    b"agent-evolve:reference-contextual-outcomes:v1;max-actions=24;prior-only;"
    b"same-parent-first;direct-lineage-second;cross-lineage-excluded;"
    b"observational-predictive-history-not-causal-credit",
)
_MULTI_HORIZON_CONTEXTUAL_OUTCOMES_POLICY = (
    "multi_horizon_parent_local_outcomes",
    b"agent-evolve:reference-contextual-outcomes:v2;max-actions=24;prior-only;"
    b"same-parent-first;direct-lineage-second;cross-lineage-excluded;"
    b"immediate=observational-predictive-history-not-causal-credit;"
    b"post-recombination=stage-front-survival-and-selection-conditioned-"
    b"descendant-yield;terminal=successful-final-front-membership;"
    + CONTEXTUAL_DELAYED_CREDIT_DEFINITION_SHA256.encode("ascii"),
)


def _binding(
    role: ReferenceCampaignPolicyRole,
    implementation: object,
    *,
    hierarchical_proposal_support: bool = False,
    context_local_memory: bool = False,
    operator_stratified_acquisition: bool = False,
    horizon_bounded_acquisition: bool = False,
    target_conditioned_acquisition: bool = False,
    constraint_decoupled_acquisition: bool = False,
    minimum_intervention_projection: bool = False,
    evidence_calibrated_source_mix: bool = False,
    contextual_search_allocation: bool = False,
    residual_frontier_planning: bool = False,
) -> CampaignPolicyBinding:
    if type(role) is not ReferenceCampaignPolicyRole:
        raise TypeError("role must be an exact ReferenceCampaignPolicyRole")
    if implementation is None:
        raise ValueError("reference policy implementation cannot be None")
    if sum(
        (
            operator_stratified_acquisition,
            horizon_bounded_acquisition,
            target_conditioned_acquisition,
        )
    ) > 1:
        raise ValueError("acquisition treatments are mutually exclusive")
    if constraint_decoupled_acquisition and not (
        horizon_bounded_acquisition or target_conditioned_acquisition
    ):
        raise ValueError(
            "constraint-decoupled acquisition requires horizon-bounded or "
            "target-conditioned acquisition"
        )
    if minimum_intervention_projection and not constraint_decoupled_acquisition:
        raise ValueError(
            "minimum intervention requires constraint-decoupled acquisition"
        )
    if evidence_calibrated_source_mix and not minimum_intervention_projection:
        raise ValueError("evidence-calibrated source mix requires minimum intervention")
    if contextual_search_allocation and not evidence_calibrated_source_mix:
        raise ValueError(
            "contextual search allocation requires evidence-calibrated source mix"
        )
    if residual_frontier_planning and not contextual_search_allocation:
        raise ValueError(
            "residual frontier planning requires contextual search allocation"
        )
    if (
        role is ReferenceCampaignPolicyRole.PARENT_SELECTION
        and type(implementation)
        in {
            StagnationAwareDiverseCampaignParentSelector,
            ResidualHypervolumeCampaignParentSelector,
        }
    ):
        return CampaignPolicyBinding(
            implementation=implementation,
            policy_id=implementation.policy_id,
            policy_version=implementation.policy_version,
            definition_sha256=implementation.definition_sha256,
        )
    if role is ReferenceCampaignPolicyRole.MEMORY_ASSIGNMENT and context_local_memory:
        policy_id, definition = _CONTEXT_LOCAL_MEMORY_POLICY
    elif (
        role is ReferenceCampaignPolicyRole.PORTFOLIO_SELECTION
        and contextual_search_allocation
    ):
        policy_id, definition = (
            _RESIDUAL_FRONTIER_CONTEXTUAL_SEARCH_PORTFOLIO_POLICY
            if residual_frontier_planning
            else _CONTEXTUAL_SEARCH_PORTFOLIO_POLICY
        )
        if type(implementation) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise TypeError("contextual-search method requires its exact allocator")
        definition += b"\x00" + json.dumps(
            implementation.to_record(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    elif (
        role is ReferenceCampaignPolicyRole.PORTFOLIO_SELECTION
        and evidence_calibrated_source_mix
    ):
        policy_id, definition = _EVIDENCE_CALIBRATED_SOURCE_MIX_PORTFOLIO_POLICY
        if type(implementation) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise TypeError("source-mix method requires its exact allocator")
        definition += b"\x00" + json.dumps(
            implementation.to_record(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    elif (
        role is ReferenceCampaignPolicyRole.PORTFOLIO_SELECTION
        and minimum_intervention_projection
    ):
        policy_id, definition = _MINIMUM_INTERVENTION_HORIZON_PORTFOLIO_POLICY
        if type(implementation) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise TypeError("minimum-intervention method requires its exact allocator")
        definition += b"\x00" + json.dumps(
            implementation.to_record(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    elif (
        role is ReferenceCampaignPolicyRole.PORTFOLIO_SELECTION
        and constraint_decoupled_acquisition
    ):
        if target_conditioned_acquisition:
            policy_id, definition = (
                _CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_PORTFOLIO_POLICY
            )
            if type(implementation) is not TargetConditionedSlateAllocatorAdapter:
                raise TypeError(
                    "constraint-decoupled target method requires its exact allocator"
                )
        else:
            policy_id, definition = _CONSTRAINT_DECOUPLED_HORIZON_PORTFOLIO_POLICY
            if type(implementation) is not (
                HorizonBoundedStructuralPosteriorSlatePolicy
            ):
                raise TypeError(
                    "constraint-decoupled horizon method requires its exact allocator"
                )
        definition += b"\x00" + json.dumps(
            implementation.to_record(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    elif (
        role is ReferenceCampaignPolicyRole.PORTFOLIO_SELECTION
        and target_conditioned_acquisition
    ):
        policy_id, definition = _TARGET_CONDITIONED_PORTFOLIO_POLICY
        if type(implementation) is not TargetConditionedSlateAllocatorAdapter:
            raise TypeError("target-conditioned method requires its exact allocator")
        definition += b"\x00" + json.dumps(
            implementation.to_record(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    elif (
        role is ReferenceCampaignPolicyRole.PORTFOLIO_SELECTION
        and horizon_bounded_acquisition
    ):
        policy_id, definition = _HORIZON_BOUNDED_PORTFOLIO_POLICY
        if type(implementation) is not HorizonBoundedStructuralPosteriorSlatePolicy:
            raise TypeError("horizon-bounded method requires its exact allocator")
        definition += b"\x00" + json.dumps(
            implementation.to_record(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    elif (
        role is ReferenceCampaignPolicyRole.PORTFOLIO_SELECTION
        and operator_stratified_acquisition
    ):
        policy_id, definition = _OPERATOR_STRATIFIED_PORTFOLIO_POLICY
        if type(implementation) is not (
            OperatorStratifiedStructuralPosteriorSlatePolicy
        ):
            raise TypeError("operator-stratified method requires its exact allocator")
        definition += b"\x00" + json.dumps(
            implementation.to_record(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    elif (
        role is ReferenceCampaignPolicyRole.PORTFOLIO_SELECTION
        and hierarchical_proposal_support
    ):
        policy_id, definition = _HIERARCHICAL_PORTFOLIO_POLICY
    else:
        policy_id, definition = _POLICY_IDS[role]
    return CampaignPolicyBinding(
        implementation=implementation,
        policy_id=policy_id,
        policy_version=1,
        definition_sha256=hashlib.sha256(definition).hexdigest(),
    )


def _archive_context_binding(
    implementation: CampaignPortfolioArchiveContextProjector,
) -> CampaignPolicyBinding:
    if not isinstance(implementation, CampaignPortfolioArchiveContextProjector):
        raise TypeError(
            "archive_context must satisfy the generic archive-context projector port"
        )
    return CampaignPolicyBinding(
        implementation=implementation,
        policy_id=implementation.projector_id,
        policy_version=implementation.projector_version,
        definition_sha256=implementation.definition_sha256,
    )


def _method_identity(
    *,
    hierarchical_proposal_support: bool,
    frontier_context: bool,
    context_local_successor: bool = False,
    operator_stratified_acquisition: bool = False,
    horizon_bounded_acquisition: bool = False,
    target_conditioned_acquisition: bool = False,
    stagnation_aware_parent_selection: bool = False,
    constraint_decoupled_acquisition: bool = False,
    minimum_intervention_projection: bool = False,
    evidence_calibrated_source_mix: bool = False,
    contextual_search_allocation: bool = False,
    residual_frontier_planning: bool = False,
) -> tuple[str, int]:
    if sum(
        (
            operator_stratified_acquisition,
            horizon_bounded_acquisition,
            target_conditioned_acquisition,
        )
    ) > 1:
        raise ValueError("acquisition treatments are mutually exclusive")
    if residual_frontier_planning:
        if not contextual_search_allocation:
            raise ValueError(
                "residual frontier planning requires contextual search allocation"
            )
        if not (
            hierarchical_proposal_support
            and frontier_context
            and context_local_successor
            and horizon_bounded_acquisition
            and constraint_decoupled_acquisition
            and minimum_intervention_projection
            and evidence_calibrated_source_mix
        ) or stagnation_aware_parent_selection:
            raise ValueError(
                "residual-frontier successor requires the complete V21 stack "
                "and its joint residual parent policy"
            )
        return (
            RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_ID,
            RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_VERSION,
        )
    if contextual_search_allocation:
        if not evidence_calibrated_source_mix:
            raise ValueError(
                "contextual search allocation requires evidence-calibrated source mix"
            )
        if not (
            hierarchical_proposal_support
            and frontier_context
            and context_local_successor
            and horizon_bounded_acquisition
            and stagnation_aware_parent_selection
            and constraint_decoupled_acquisition
            and minimum_intervention_projection
        ):
            raise ValueError(
                "contextual-search successor requires the complete V11 stack"
            )
        return (
            CONTEXTUAL_SEARCH_AGENT_EVOLVE_METHOD_ID,
            CONTEXTUAL_SEARCH_AGENT_EVOLVE_METHOD_VERSION,
        )
    if evidence_calibrated_source_mix:
        if not minimum_intervention_projection:
            raise ValueError(
                "evidence-calibrated source mix requires minimum intervention"
            )
        if not (
            hierarchical_proposal_support
            and frontier_context
            and context_local_successor
            and horizon_bounded_acquisition
            and stagnation_aware_parent_selection
            and constraint_decoupled_acquisition
        ):
            raise ValueError(
                "evidence-calibrated source successor requires the complete V10 stack"
            )
        return (
            EVIDENCE_CALIBRATED_SOURCE_MIX_AGENT_EVOLVE_METHOD_ID,
            EVIDENCE_CALIBRATED_SOURCE_MIX_AGENT_EVOLVE_METHOD_VERSION,
        )
    if minimum_intervention_projection:
        if not constraint_decoupled_acquisition:
            raise ValueError(
                "minimum intervention requires constraint-decoupled acquisition"
            )
        if not (
            hierarchical_proposal_support
            and frontier_context
            and context_local_successor
            and horizon_bounded_acquisition
            and stagnation_aware_parent_selection
        ):
            raise ValueError(
                "minimum-intervention successor requires the complete V9 stack"
            )
        return (
            MINIMUM_INTERVENTION_AGENT_EVOLVE_METHOD_ID,
            MINIMUM_INTERVENTION_AGENT_EVOLVE_METHOD_VERSION,
        )
    if constraint_decoupled_acquisition:
        if target_conditioned_acquisition:
            if not (
                hierarchical_proposal_support
                and frontier_context
                and context_local_successor
            ) or stagnation_aware_parent_selection:
                raise ValueError(
                    "constraint-decoupled target successor requires hierarchical "
                    "support, authenticated frontier context, context-local memory, "
                    "and the target parent policy"
                )
            return (
                CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_ID,
                CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_VERSION,
            )
        if not (
            hierarchical_proposal_support
            and frontier_context
            and context_local_successor
            and horizon_bounded_acquisition
            and stagnation_aware_parent_selection
        ):
            raise ValueError(
                "constraint-decoupled successor requires the complete V8 stack"
            )
        return (
            CONSTRAINT_DECOUPLED_AGENT_EVOLVE_METHOD_ID,
            CONSTRAINT_DECOUPLED_AGENT_EVOLVE_METHOD_VERSION,
        )
    if target_conditioned_acquisition:
        if not (
            hierarchical_proposal_support
            and frontier_context
            and context_local_successor
        ) or stagnation_aware_parent_selection:
            raise ValueError(
                "target-conditioned successor requires hierarchical support, "
                "authenticated frontier context, context-local memory, and the "
                "target parent policy"
            )
        return (
            TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_ID,
            TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_VERSION,
        )
    if stagnation_aware_parent_selection:
        if not (
            hierarchical_proposal_support
            and frontier_context
            and context_local_successor
            and horizon_bounded_acquisition
        ):
            raise ValueError(
                "stagnation-aware successor requires hierarchical support, "
                "authenticated frontier context, context-local memory, and "
                "horizon-bounded acquisition"
            )
        return (
            STAGNATION_AWARE_AGENT_EVOLVE_METHOD_ID,
            STAGNATION_AWARE_AGENT_EVOLVE_METHOD_VERSION,
        )
    if horizon_bounded_acquisition:
        if not (
            hierarchical_proposal_support
            and frontier_context
            and context_local_successor
        ):
            raise ValueError(
                "horizon-bounded successor requires hierarchical support, "
                "authenticated frontier context, and context-local memory"
            )
        return (
            HORIZON_BOUNDED_AGENT_EVOLVE_METHOD_ID,
            HORIZON_BOUNDED_AGENT_EVOLVE_METHOD_VERSION,
        )
    if operator_stratified_acquisition:
        if not (
            hierarchical_proposal_support
            and frontier_context
            and context_local_successor
        ):
            raise ValueError(
                "operator-stratified successor requires hierarchical support, "
                "authenticated frontier context, and context-local memory"
            )
        return (
            OPERATOR_STRATIFIED_AGENT_EVOLVE_METHOD_ID,
            OPERATOR_STRATIFIED_AGENT_EVOLVE_METHOD_VERSION,
        )
    if context_local_successor:
        if not hierarchical_proposal_support or not frontier_context:
            raise ValueError(
                "context-local successor requires hierarchical support and "
                "authenticated frontier context"
            )
        return (
            CONTEXT_LOCAL_AGENT_EVOLVE_METHOD_ID,
            CONTEXT_LOCAL_AGENT_EVOLVE_METHOD_VERSION,
        )
    if hierarchical_proposal_support and frontier_context:
        return (
            FRONTIER_HIERARCHICAL_AGENT_EVOLVE_METHOD_ID,
            FRONTIER_HIERARCHICAL_AGENT_EVOLVE_METHOD_VERSION,
        )
    if hierarchical_proposal_support:
        return (
            HIERARCHICAL_AGENT_EVOLVE_METHOD_ID,
            HIERARCHICAL_AGENT_EVOLVE_METHOD_VERSION,
        )
    if frontier_context:
        return (
            FRONTIER_AGENT_EVOLVE_METHOD_ID,
            FRONTIER_AGENT_EVOLVE_METHOD_VERSION,
        )
    return (
        REFERENCE_AGENT_EVOLVE_METHOD_ID,
        REFERENCE_AGENT_EVOLVE_METHOD_VERSION,
    )


@dataclass(frozen=True, slots=True)
class ReferenceCampaignImplementations:
    """Runtime objects hidden behind the stable reference policy identities."""

    parent_selection: object
    memory_assignment: object
    portfolio_selection: object
    recombination: object
    reflection: object
    archive_context: CampaignPortfolioArchiveContextProjector | None = None
    variation_topology: CampaignPolicyBinding | None = None
    contextual_outcomes: CampaignPolicyBinding | None = None

    def __post_init__(self) -> None:
        for name in (
            "parent_selection",
            "memory_assignment",
            "portfolio_selection",
            "recombination",
            "reflection",
        ):
            if getattr(self, name) is None:
                raise ValueError(f"{name} implementation cannot be None")
        if self.archive_context is not None and not isinstance(
            self.archive_context,
            CampaignPortfolioArchiveContextProjector,
        ):
            raise TypeError(
                "archive_context must satisfy the generic archive-context "
                "projector port or be None"
            )
        for name in ("variation_topology", "contextual_outcomes"):
            value = getattr(self, name)
            if value is not None:
                if type(value) is not CampaignPolicyBinding:
                    raise TypeError(
                        f"{name} must be an exact CampaignPolicyBinding or None"
                    )
                CampaignPolicyBinding.__post_init__(value)
        if (self.variation_topology is None) is not (self.contextual_outcomes is None):
            raise ValueError(
                "variation_topology and contextual_outcomes must be bound together"
            )

    def bindings(
        self,
        *,
        hierarchical_proposal_support: bool = False,
        operator_stratified_acquisition: bool = False,
        horizon_bounded_acquisition: bool = False,
        target_conditioned_acquisition: bool = False,
        constraint_decoupled_acquisition: bool = False,
        minimum_intervention_projection: bool = False,
        evidence_calibrated_source_mix: bool = False,
        contextual_search_allocation: bool = False,
        residual_frontier_planning: bool = False,
    ) -> dict[ReferenceCampaignPolicyRole, CampaignPolicyBinding]:
        self.__post_init__()
        if type(hierarchical_proposal_support) is not bool:
            raise TypeError("hierarchical_proposal_support must be an exact bool")
        if type(operator_stratified_acquisition) is not bool:
            raise TypeError("operator_stratified_acquisition must be an exact bool")
        if type(horizon_bounded_acquisition) is not bool:
            raise TypeError("horizon_bounded_acquisition must be an exact bool")
        if type(target_conditioned_acquisition) is not bool:
            raise TypeError("target_conditioned_acquisition must be an exact bool")
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
        if sum(
            (
                operator_stratified_acquisition,
                horizon_bounded_acquisition,
                target_conditioned_acquisition,
            )
        ) > 1:
            raise ValueError("acquisition treatments are mutually exclusive")
        if operator_stratified_acquisition and not hierarchical_proposal_support:
            raise ValueError(
                "operator-stratified acquisition requires hierarchical support"
            )
        if horizon_bounded_acquisition and not hierarchical_proposal_support:
            raise ValueError(
                "horizon-bounded acquisition requires hierarchical support"
            )
        if target_conditioned_acquisition and not hierarchical_proposal_support:
            raise ValueError(
                "target-conditioned acquisition requires hierarchical support"
            )
        if constraint_decoupled_acquisition and not (
            horizon_bounded_acquisition or target_conditioned_acquisition
        ):
            raise ValueError(
                "constraint-decoupled acquisition requires horizon bounds or a "
                "target-conditioned allocator"
            )
        if minimum_intervention_projection and not constraint_decoupled_acquisition:
            raise ValueError(
                "minimum intervention requires constraint-decoupled acquisition"
            )
        if evidence_calibrated_source_mix and not minimum_intervention_projection:
            raise ValueError(
                "evidence-calibrated source mix requires minimum intervention"
            )
        if contextual_search_allocation and not evidence_calibrated_source_mix:
            raise ValueError(
                "contextual search allocation requires evidence-calibrated source mix"
            )
        if residual_frontier_planning and not contextual_search_allocation:
            raise ValueError(
                "residual frontier planning requires contextual search allocation"
            )
        context_local_memory = self.variation_topology is not None
        bindings = {
            role: _binding(
                role,
                getattr(self, role.value),
                hierarchical_proposal_support=hierarchical_proposal_support,
                context_local_memory=context_local_memory,
                operator_stratified_acquisition=(operator_stratified_acquisition),
                horizon_bounded_acquisition=horizon_bounded_acquisition,
                target_conditioned_acquisition=target_conditioned_acquisition,
                constraint_decoupled_acquisition=(constraint_decoupled_acquisition),
                minimum_intervention_projection=(minimum_intervention_projection),
                evidence_calibrated_source_mix=(evidence_calibrated_source_mix),
                contextual_search_allocation=contextual_search_allocation,
                residual_frontier_planning=residual_frontier_planning,
            )
            for role in ReferenceCampaignPolicyRole
            if role
            not in {
                ReferenceCampaignPolicyRole.ARCHIVE_CONTEXT,
                ReferenceCampaignPolicyRole.VARIATION_TOPOLOGY,
                ReferenceCampaignPolicyRole.CONTEXTUAL_OUTCOMES,
            }
        }
        if self.archive_context is not None:
            bindings[ReferenceCampaignPolicyRole.ARCHIVE_CONTEXT] = (
                _archive_context_binding(self.archive_context)
            )
        if self.variation_topology is not None:
            bindings[ReferenceCampaignPolicyRole.VARIATION_TOPOLOGY] = (
                self.variation_topology
            )
            assert self.contextual_outcomes is not None
            bindings[ReferenceCampaignPolicyRole.CONTEXTUAL_OUTCOMES] = (
                reference_multi_horizon_contextual_outcomes_binding(
                    self.contextual_outcomes.implementation
                )
                if contextual_search_allocation
                else self.contextual_outcomes
            )
        return bindings


def reference_atomic_variation_topology_binding(
    implementation: object,
) -> CampaignPolicyBinding:
    """Authenticate the complete parent-bound atomic finite-action topology."""

    if implementation is None:
        raise ValueError("variation-topology implementation cannot be None")
    policy_id, definition = _ATOMIC_VARIATION_TOPOLOGY_POLICY
    return CampaignPolicyBinding(
        implementation=implementation,
        policy_id=policy_id,
        policy_version=1,
        definition_sha256=hashlib.sha256(definition).hexdigest(),
    )


def reference_contextual_outcomes_binding(
    implementation: object,
) -> CampaignPolicyBinding:
    """Authenticate bounded parent-local prior-outcome retrieval."""

    if implementation is None:
        raise ValueError("contextual-outcome implementation cannot be None")
    policy_id, definition = _CONTEXTUAL_OUTCOMES_POLICY
    return CampaignPolicyBinding(
        implementation=implementation,
        policy_id=policy_id,
        policy_version=1,
        definition_sha256=hashlib.sha256(definition).hexdigest(),
    )


def reference_multi_horizon_contextual_outcomes_binding(
    implementation: object,
) -> CampaignPolicyBinding:
    """Authenticate actionable stage and terminal outcome credit."""

    if implementation is None:
        raise ValueError("contextual-outcome implementation cannot be None")
    policy_id, definition = _MULTI_HORIZON_CONTEXTUAL_OUTCOMES_POLICY
    return CampaignPolicyBinding(
        implementation=implementation,
        policy_id=policy_id,
        policy_version=2,
        definition_sha256=hashlib.sha256(definition).hexdigest(),
    )


def reference_hierarchical_r2_variation_topology_binding(
    implementation: object,
    *,
    max_composite_options: int,
    required_composite_proposals: int,
) -> CampaignPolicyBinding:
    """Authenticate bounded engine-materialized radius-two proposal strata."""

    if implementation is None:
        raise ValueError("variation-topology implementation cannot be None")
    if type(max_composite_options) is not int or max_composite_options <= 0:
        raise ValueError("max_composite_options must be a positive exact integer")
    if (
        type(required_composite_proposals) is not int
        or not 1 <= required_composite_proposals < 8
    ):
        raise ValueError("required_composite_proposals must lie in [1, 8)")
    record = {
        "schema_version": 2,
        "radius": 2,
        "max_composite_options": max_composite_options,
        "required_composite_proposals": required_composite_proposals,
        "required_composite_proposals_semantics": (
            "preferred_then_nearest_exact_k8_capacity_projection"
        ),
        "capacity_projection_inputs": (
            "current_parent_atomic_and_materialized_composite_counts"
        ),
        "retain_all_atomic_options": True,
        "pair_prefilter": "disjoint_parent_relative_typed_patch_paths",
        "pair_admission": "engine_replay_union_and_exact_rediff",
        "ranked_union_member_kinds": ["atomic", "compose_r2"],
        "provider_materialization_authority": False,
        "outcomes_consulted": False,
    }
    definition_sha256 = hashlib.sha256(
        _HIERARCHICAL_R2_VARIATION_TOPOLOGY_DOMAIN
        + json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return CampaignPolicyBinding(
        implementation=implementation,
        policy_id=_HIERARCHICAL_R2_VARIATION_TOPOLOGY_POLICY_ID,
        policy_version=2,
        definition_sha256=definition_sha256,
    )


def reference_campaign_experiment_profile(
    *,
    profile_id: str,
    model_execution: OpenRouterModelExecutionProfile,
    implementations: ReferenceCampaignImplementations,
    candidate_pool_size: int | None,
    evaluator_concurrency: int,
    agent_concurrency: int,
    agent_queue_capacity: int,
    hierarchical_proposal_support: bool = False,
    operator_stratified_acquisition: bool = False,
    horizon_bounded_acquisition: bool = False,
    target_conditioned_acquisition: bool = False,
    constraint_decoupled_acquisition: bool = False,
    minimum_intervention_projection: bool = False,
    evidence_calibrated_source_mix: bool = False,
    contextual_search_allocation: bool = False,
    prompt_arm: WorkloadPromptArm = WorkloadPromptArm.SEMANTIC,
    scale_shape: PortfolioScaleShape = REFERENCE_36_OFFSPRING_SCALE_SHAPE,
    model_selection_size: int = 8,
) -> CampaignExperimentProfile:
    """Bind one executable cell to the shared successor-method identity."""

    if type(implementations) is not ReferenceCampaignImplementations:
        raise TypeError("implementations must be exact")
    if candidate_pool_size is not None:
        raise ValueError(
            "the reference successor requires the complete finite contract"
        )
    if type(hierarchical_proposal_support) is not bool:
        raise TypeError("hierarchical_proposal_support must be an exact bool")
    if type(operator_stratified_acquisition) is not bool:
        raise TypeError("operator_stratified_acquisition must be an exact bool")
    if type(horizon_bounded_acquisition) is not bool:
        raise TypeError("horizon_bounded_acquisition must be an exact bool")
    if type(target_conditioned_acquisition) is not bool:
        raise TypeError("target_conditioned_acquisition must be an exact bool")
    if type(constraint_decoupled_acquisition) is not bool:
        raise TypeError("constraint_decoupled_acquisition must be an exact bool")
    if type(minimum_intervention_projection) is not bool:
        raise TypeError("minimum_intervention_projection must be an exact bool")
    if type(evidence_calibrated_source_mix) is not bool:
        raise TypeError("evidence_calibrated_source_mix must be an exact bool")
    if type(contextual_search_allocation) is not bool:
        raise TypeError("contextual_search_allocation must be an exact bool")
    if sum(
        (
            operator_stratified_acquisition,
            horizon_bounded_acquisition,
            target_conditioned_acquisition,
        )
    ) > 1:
        raise ValueError("acquisition treatments are mutually exclusive")
    if constraint_decoupled_acquisition and not (
        horizon_bounded_acquisition or target_conditioned_acquisition
    ):
        raise ValueError(
            "constraint-decoupled acquisition requires horizon-bounded or "
            "target-conditioned acquisition"
        )
    if target_conditioned_acquisition != (
        type(implementations.portfolio_selection)
        is TargetConditionedSlateAllocatorAdapter
    ):
        raise ValueError(
            "target-conditioned treatment flag differs from its allocator"
        )
    if minimum_intervention_projection and not constraint_decoupled_acquisition:
        raise ValueError(
            "minimum intervention requires constraint-decoupled acquisition"
        )
    if evidence_calibrated_source_mix and not minimum_intervention_projection:
        raise ValueError("evidence-calibrated source mix requires minimum intervention")
    if contextual_search_allocation and not evidence_calibrated_source_mix:
        raise ValueError(
            "contextual search allocation requires evidence-calibrated source mix"
        )
    if operator_stratified_acquisition and (
        not hierarchical_proposal_support
        or implementations.archive_context is None
        or implementations.variation_topology is None
        or implementations.variation_topology.policy_id
        != _HIERARCHICAL_R2_VARIATION_TOPOLOGY_POLICY_ID
    ):
        raise ValueError(
            "operator-stratified acquisition requires the complete "
            "context-local hierarchical successor stack"
        )
    if horizon_bounded_acquisition and (
        not hierarchical_proposal_support
        or implementations.archive_context is None
        or implementations.variation_topology is None
        or implementations.variation_topology.policy_id
        != _HIERARCHICAL_R2_VARIATION_TOPOLOGY_POLICY_ID
    ):
        raise ValueError(
            "horizon-bounded acquisition requires the complete "
            "context-local hierarchical successor stack"
        )
    if target_conditioned_acquisition and (
        not hierarchical_proposal_support
        or implementations.archive_context is None
        or implementations.variation_topology is None
        or implementations.variation_topology.policy_id
        != _HIERARCHICAL_R2_VARIATION_TOPOLOGY_POLICY_ID
    ):
        raise ValueError(
            "target-conditioned acquisition requires the complete context-local "
            "hierarchical successor stack"
        )
    if type(scale_shape) is not PortfolioScaleShape:
        raise TypeError("scale_shape must be an exact PortfolioScaleShape")
    scale_shape.__post_init__()
    if type(model_selection_size) is not int or model_selection_size <= 0:
        raise ValueError("model_selection_size must be a positive exact integer")
    residual_frontier_planning = (
        type(implementations.parent_selection)
        is ResidualHypervolumeCampaignParentSelector
    )
    bindings = implementations.bindings(
        hierarchical_proposal_support=hierarchical_proposal_support,
        operator_stratified_acquisition=operator_stratified_acquisition,
        horizon_bounded_acquisition=horizon_bounded_acquisition,
        target_conditioned_acquisition=target_conditioned_acquisition,
        constraint_decoupled_acquisition=constraint_decoupled_acquisition,
        minimum_intervention_projection=minimum_intervention_projection,
        evidence_calibrated_source_mix=evidence_calibrated_source_mix,
        contextual_search_allocation=contextual_search_allocation,
        residual_frontier_planning=residual_frontier_planning,
    )
    context_local_successor = implementations.variation_topology is not None
    stagnation_aware_parent_selection = (
        type(implementations.parent_selection)
        is StagnationAwareDiverseCampaignParentSelector
    )
    method_id, method_version = _method_identity(
        hierarchical_proposal_support=hierarchical_proposal_support,
        frontier_context=implementations.archive_context is not None,
        context_local_successor=context_local_successor,
        operator_stratified_acquisition=operator_stratified_acquisition,
        horizon_bounded_acquisition=horizon_bounded_acquisition,
        target_conditioned_acquisition=target_conditioned_acquisition,
        stagnation_aware_parent_selection=(stagnation_aware_parent_selection),
        constraint_decoupled_acquisition=constraint_decoupled_acquisition,
        minimum_intervention_projection=minimum_intervention_projection,
        evidence_calibrated_source_mix=evidence_calibrated_source_mix,
        contextual_search_allocation=contextual_search_allocation,
        residual_frontier_planning=residual_frontier_planning,
    )
    return CampaignExperimentProfile(
        profile_id=profile_id,
        profile_version=1,
        method_id=method_id,
        method_version=method_version,
        scale_shape=scale_shape,
        candidate_pool_size=candidate_pool_size,
        model_selection_size=model_selection_size,
        prompt_arm=prompt_arm,
        parent_selection=bindings[ReferenceCampaignPolicyRole.PARENT_SELECTION],
        memory_assignment=bindings[ReferenceCampaignPolicyRole.MEMORY_ASSIGNMENT],
        portfolio_selection=bindings[ReferenceCampaignPolicyRole.PORTFOLIO_SELECTION],
        recombination=bindings[ReferenceCampaignPolicyRole.RECOMBINATION],
        reflection=bindings[ReferenceCampaignPolicyRole.REFLECTION],
        model_execution=model_execution,
        archive_context=bindings.get(ReferenceCampaignPolicyRole.ARCHIVE_CONTEXT),
        variation_topology=bindings.get(ReferenceCampaignPolicyRole.VARIATION_TOPOLOGY),
        contextual_outcomes=bindings.get(
            ReferenceCampaignPolicyRole.CONTEXTUAL_OUTCOMES
        ),
        evaluator_concurrency=evaluator_concurrency,
        agent_concurrency=agent_concurrency,
        agent_queue_capacity=agent_queue_capacity,
        reflection_supervision=CampaignReflectionSupervisionPolicy(
            ReflectionFailureMode.BEST_EFFORT_DEGRADED
        ),
    )


def rebind_reference_campaign_implementations(
    profile: CampaignExperimentProfile,
    implementations: ReferenceCampaignImplementations,
) -> CampaignExperimentProfile:
    """Swap runtime objects without changing any scientific method identity."""

    if type(profile) is not CampaignExperimentProfile:
        raise TypeError("profile must be exact")
    profile.__post_init__()
    method_identity = (profile.method_id, profile.method_version)
    known_identities = {
        _method_identity(
            hierarchical_proposal_support=hierarchical,
            frontier_context=frontier,
        )
        for hierarchical in (False, True)
        for frontier in (False, True)
    }
    known_identities.add(
        _method_identity(
            hierarchical_proposal_support=True,
            frontier_context=True,
            context_local_successor=True,
        )
    )
    known_identities.add(
        _method_identity(
            hierarchical_proposal_support=True,
            frontier_context=True,
            context_local_successor=True,
            target_conditioned_acquisition=True,
        )
    )
    known_identities.add(
        _method_identity(
            hierarchical_proposal_support=True,
            frontier_context=True,
            context_local_successor=True,
            target_conditioned_acquisition=True,
            constraint_decoupled_acquisition=True,
        )
    )
    known_identities.add(
        _method_identity(
            hierarchical_proposal_support=True,
            frontier_context=True,
            context_local_successor=True,
            operator_stratified_acquisition=True,
        )
    )
    known_identities.add(
        _method_identity(
            hierarchical_proposal_support=True,
            frontier_context=True,
            context_local_successor=True,
            horizon_bounded_acquisition=True,
        )
    )
    known_identities.add(
        _method_identity(
            hierarchical_proposal_support=True,
            frontier_context=True,
            context_local_successor=True,
            horizon_bounded_acquisition=True,
            stagnation_aware_parent_selection=True,
        )
    )
    known_identities.add(
        _method_identity(
            hierarchical_proposal_support=True,
            frontier_context=True,
            context_local_successor=True,
            horizon_bounded_acquisition=True,
            stagnation_aware_parent_selection=True,
            constraint_decoupled_acquisition=True,
        )
    )
    known_identities.add(
        _method_identity(
            hierarchical_proposal_support=True,
            frontier_context=True,
            context_local_successor=True,
            horizon_bounded_acquisition=True,
            stagnation_aware_parent_selection=True,
            constraint_decoupled_acquisition=True,
            minimum_intervention_projection=True,
        )
    )
    known_identities.add(
        _method_identity(
            hierarchical_proposal_support=True,
            frontier_context=True,
            context_local_successor=True,
            horizon_bounded_acquisition=True,
            stagnation_aware_parent_selection=True,
            constraint_decoupled_acquisition=True,
            minimum_intervention_projection=True,
            evidence_calibrated_source_mix=True,
        )
    )
    known_identities.add(
        _method_identity(
            hierarchical_proposal_support=True,
            frontier_context=True,
            context_local_successor=True,
            horizon_bounded_acquisition=True,
            stagnation_aware_parent_selection=True,
            constraint_decoupled_acquisition=True,
            minimum_intervention_projection=True,
            evidence_calibrated_source_mix=True,
            contextual_search_allocation=True,
        )
    )
    known_identities.add(
        _method_identity(
            hierarchical_proposal_support=True,
            frontier_context=True,
            context_local_successor=True,
            horizon_bounded_acquisition=True,
            constraint_decoupled_acquisition=True,
            minimum_intervention_projection=True,
            evidence_calibrated_source_mix=True,
            contextual_search_allocation=True,
            residual_frontier_planning=True,
        )
    )
    if method_identity not in known_identities:
        raise ValueError("profile is not the reference AgentEvolve method")
    is_hierarchical = method_identity in {
        _method_identity(
            hierarchical_proposal_support=True,
            frontier_context=frontier,
        )
        for frontier in (False, True)
    }
    is_operator_stratified = method_identity == (
        OPERATOR_STRATIFIED_AGENT_EVOLVE_METHOD_ID,
        OPERATOR_STRATIFIED_AGENT_EVOLVE_METHOD_VERSION,
    )
    is_horizon_bounded = method_identity == (
        HORIZON_BOUNDED_AGENT_EVOLVE_METHOD_ID,
        HORIZON_BOUNDED_AGENT_EVOLVE_METHOD_VERSION,
    )
    is_stagnation_aware = method_identity == (
        STAGNATION_AWARE_AGENT_EVOLVE_METHOD_ID,
        STAGNATION_AWARE_AGENT_EVOLVE_METHOD_VERSION,
    )
    is_constraint_decoupled = method_identity == (
        CONSTRAINT_DECOUPLED_AGENT_EVOLVE_METHOD_ID,
        CONSTRAINT_DECOUPLED_AGENT_EVOLVE_METHOD_VERSION,
    )
    is_target_conditioned = method_identity in {
        (
            TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_ID,
            TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_VERSION,
        ),
        (
            CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_ID,
            CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_VERSION,
        ),
    }
    is_constraint_decoupled_target = method_identity == (
        CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_ID,
        CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_VERSION,
    )
    is_minimum_intervention = method_identity == (
        MINIMUM_INTERVENTION_AGENT_EVOLVE_METHOD_ID,
        MINIMUM_INTERVENTION_AGENT_EVOLVE_METHOD_VERSION,
    )
    is_evidence_calibrated_source_mix = method_identity == (
        EVIDENCE_CALIBRATED_SOURCE_MIX_AGENT_EVOLVE_METHOD_ID,
        EVIDENCE_CALIBRATED_SOURCE_MIX_AGENT_EVOLVE_METHOD_VERSION,
    )
    is_residual_frontier = method_identity == (
        RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_ID,
        RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_VERSION,
    )
    is_contextual_search = method_identity in {
        (
            CONTEXTUAL_SEARCH_AGENT_EVOLVE_METHOD_ID,
            CONTEXTUAL_SEARCH_AGENT_EVOLVE_METHOD_VERSION,
        ),
        (
            RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_ID,
            RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_VERSION,
        ),
    }
    is_evidence_calibrated_source_mix = (
        is_evidence_calibrated_source_mix or is_contextual_search
    )
    is_minimum_intervention = (
        is_minimum_intervention or is_evidence_calibrated_source_mix
    )
    is_constraint_decoupled = (
        is_constraint_decoupled
        or is_minimum_intervention
        or is_constraint_decoupled_target
    )
    is_stagnation_aware = is_stagnation_aware or (
        is_constraint_decoupled
        and not is_target_conditioned
        and not is_residual_frontier
    )
    is_horizon_bounded = (
        is_horizon_bounded or is_stagnation_aware or is_residual_frontier
    )
    is_context_local = method_identity in {
        (
            CONTEXT_LOCAL_AGENT_EVOLVE_METHOD_ID,
            CONTEXT_LOCAL_AGENT_EVOLVE_METHOD_VERSION,
        ),
        (
            OPERATOR_STRATIFIED_AGENT_EVOLVE_METHOD_ID,
            OPERATOR_STRATIFIED_AGENT_EVOLVE_METHOD_VERSION,
        ),
        (
            HORIZON_BOUNDED_AGENT_EVOLVE_METHOD_ID,
            HORIZON_BOUNDED_AGENT_EVOLVE_METHOD_VERSION,
        ),
        (
            STAGNATION_AWARE_AGENT_EVOLVE_METHOD_ID,
            STAGNATION_AWARE_AGENT_EVOLVE_METHOD_VERSION,
        ),
        (
            CONSTRAINT_DECOUPLED_AGENT_EVOLVE_METHOD_ID,
            CONSTRAINT_DECOUPLED_AGENT_EVOLVE_METHOD_VERSION,
        ),
        (
            MINIMUM_INTERVENTION_AGENT_EVOLVE_METHOD_ID,
            MINIMUM_INTERVENTION_AGENT_EVOLVE_METHOD_VERSION,
        ),
        (
            EVIDENCE_CALIBRATED_SOURCE_MIX_AGENT_EVOLVE_METHOD_ID,
            EVIDENCE_CALIBRATED_SOURCE_MIX_AGENT_EVOLVE_METHOD_VERSION,
        ),
        (
            CONTEXTUAL_SEARCH_AGENT_EVOLVE_METHOD_ID,
            CONTEXTUAL_SEARCH_AGENT_EVOLVE_METHOD_VERSION,
        ),
        (
            RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_ID,
            RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_VERSION,
        ),
        (
            TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_ID,
            TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_VERSION,
        ),
        (
            CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_ID,
            CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_VERSION,
        ),
    }
    is_hierarchical = is_hierarchical or is_context_local
    has_frontier_context = method_identity in {
        _method_identity(
            hierarchical_proposal_support=hierarchical,
            frontier_context=True,
        )
        for hierarchical in (False, True)
    }
    has_frontier_context = has_frontier_context or is_context_local
    if (implementations.archive_context is not None) is not has_frontier_context:
        raise ValueError(
            "runtime archive-context implementation differs from method identity"
        )
    if (implementations.variation_topology is not None) is not is_context_local:
        raise ValueError(
            "runtime contextual/variation implementations differ from method identity"
        )
    if (
        type(implementations.parent_selection)
        is StagnationAwareDiverseCampaignParentSelector
    ) is not is_stagnation_aware:
        raise ValueError(
            "runtime stagnation-aware parent implementation differs from "
            "method identity"
        )
    if (
        type(implementations.parent_selection)
        is ResidualHypervolumeCampaignParentSelector
    ) is not is_residual_frontier:
        raise ValueError(
            "runtime residual-frontier parent implementation differs from "
            "method identity"
        )
    if (
        (is_operator_stratified or is_horizon_bounded or is_target_conditioned)
        and implementations.variation_topology is not None
        and implementations.variation_topology.policy_id
        != _HIERARCHICAL_R2_VARIATION_TOPOLOGY_POLICY_ID
    ):
        raise ValueError(
            "operator-stratified runtime requires hierarchical-r2 topology"
        )
    bindings = implementations.bindings(
        hierarchical_proposal_support=is_hierarchical,
        operator_stratified_acquisition=is_operator_stratified,
        horizon_bounded_acquisition=is_horizon_bounded,
        target_conditioned_acquisition=is_target_conditioned,
        constraint_decoupled_acquisition=is_constraint_decoupled,
        minimum_intervention_projection=is_minimum_intervention,
        evidence_calibrated_source_mix=is_evidence_calibrated_source_mix,
        contextual_search_allocation=is_contextual_search,
        residual_frontier_planning=is_residual_frontier,
    )
    rebound = replace(
        profile,
        parent_selection=bindings[ReferenceCampaignPolicyRole.PARENT_SELECTION],
        memory_assignment=bindings[ReferenceCampaignPolicyRole.MEMORY_ASSIGNMENT],
        portfolio_selection=bindings[ReferenceCampaignPolicyRole.PORTFOLIO_SELECTION],
        recombination=bindings[ReferenceCampaignPolicyRole.RECOMBINATION],
        reflection=bindings[ReferenceCampaignPolicyRole.REFLECTION],
        archive_context=bindings.get(ReferenceCampaignPolicyRole.ARCHIVE_CONTEXT),
        variation_topology=bindings.get(ReferenceCampaignPolicyRole.VARIATION_TOPOLOGY),
        contextual_outcomes=bindings.get(
            ReferenceCampaignPolicyRole.CONTEXTUAL_OUTCOMES
        ),
    )
    if rebound.experiment_definition_sha256 != profile.experiment_definition_sha256:
        raise AssertionError("runtime rebinding changed experiment identity")
    return rebound


__all__ = [
    "CONSTRAINT_DECOUPLED_AGENT_EVOLVE_METHOD_ID",
    "CONSTRAINT_DECOUPLED_AGENT_EVOLVE_METHOD_VERSION",
    "CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_ID",
    "CONSTRAINT_DECOUPLED_TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_VERSION",
    "CONTEXTUAL_SEARCH_AGENT_EVOLVE_METHOD_ID",
    "CONTEXTUAL_SEARCH_AGENT_EVOLVE_METHOD_VERSION",
    "RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_ID",
    "RESIDUAL_FRONTIER_AGENT_EVOLVE_METHOD_VERSION",
    "EVIDENCE_CALIBRATED_SOURCE_MIX_AGENT_EVOLVE_METHOD_ID",
    "EVIDENCE_CALIBRATED_SOURCE_MIX_AGENT_EVOLVE_METHOD_VERSION",
    "MINIMUM_INTERVENTION_AGENT_EVOLVE_METHOD_ID",
    "MINIMUM_INTERVENTION_AGENT_EVOLVE_METHOD_VERSION",
    "FRONTIER_AGENT_EVOLVE_METHOD_ID",
    "FRONTIER_AGENT_EVOLVE_METHOD_VERSION",
    "FRONTIER_HIERARCHICAL_AGENT_EVOLVE_METHOD_ID",
    "FRONTIER_HIERARCHICAL_AGENT_EVOLVE_METHOD_VERSION",
    "CONTEXT_LOCAL_AGENT_EVOLVE_METHOD_ID",
    "CONTEXT_LOCAL_AGENT_EVOLVE_METHOD_VERSION",
    "HORIZON_BOUNDED_AGENT_EVOLVE_METHOD_ID",
    "HORIZON_BOUNDED_AGENT_EVOLVE_METHOD_VERSION",
    "STAGNATION_AWARE_AGENT_EVOLVE_METHOD_ID",
    "STAGNATION_AWARE_AGENT_EVOLVE_METHOD_VERSION",
    "TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_ID",
    "TARGET_CONDITIONED_AGENT_EVOLVE_METHOD_VERSION",
    "OPERATOR_STRATIFIED_AGENT_EVOLVE_METHOD_ID",
    "OPERATOR_STRATIFIED_AGENT_EVOLVE_METHOD_VERSION",
    "REFERENCE_AGENT_EVOLVE_METHOD_ID",
    "REFERENCE_AGENT_EVOLVE_METHOD_VERSION",
    "HIERARCHICAL_AGENT_EVOLVE_METHOD_ID",
    "HIERARCHICAL_AGENT_EVOLVE_METHOD_VERSION",
    "ReferenceCampaignImplementations",
    "ReferenceCampaignPolicyRole",
    "rebind_reference_campaign_implementations",
    "reference_atomic_variation_topology_binding",
    "reference_campaign_experiment_profile",
    "reference_contextual_outcomes_binding",
    "reference_multi_horizon_contextual_outcomes_binding",
    "reference_hierarchical_r2_variation_topology_binding",
]
