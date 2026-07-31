"""Generic planning and allocation for action-to-frontier realization.

This application service is the workload-neutral seam between an evolutionary
campaign's authenticated objective-space aspiration and the existing
all-option consequence forecaster.  It depends only on public optimization,
finite-action, and campaign-context contracts.  Benchmark adapters may inject
a richer action glossary; otherwise a structural glossary is derived from the
sealed parent-to-child patches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from agent_evolve.application.action_allocation import (
    FeasibleBeamRiskAdjustedDiversityAllocator,
)
from agent_evolve.application.action_archive_value import (
    NormalizedResidualFrontierCell,
    ReliabilityAdjustedResidualCellExpectedHypervolumeUtility,
    ResidualCellExpectedHypervolumeUtility,
    residual_frontier_cell_from_target,
)
from agent_evolve.application.action_role_value import (
    RoleAssignmentAudit,
    RoleFactorizedActionPortfolioUtility,
    audit_role_factorized_action_portfolio,
    build_role_factorized_action_utility,
)
from agent_evolve.application.action_target_realization import (
    ActionTargetRealization,
    ResidualTargetClosurePortfolioUtility,
    TargetMetricAlias,
    assess_action_target_realization,
)
from agent_evolve.application.derived_action_semantics import (
    derive_action_space_semantics,
)
from agent_evolve.application.portfolio_campaign_runtime import (
    CAMPAIGN_FRONTIER_TARGET_KEY,
)
from agent_evolve.core.action_semantics import ActionSpaceSemantics
from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSense,
    OptimizationSemantics,
)
from agent_evolve.domain.finite_variation import FiniteVariationContract
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.ports.action_allocation import (
    ActionAllocationRequest,
    ActionAllocationResult,
    ActionPortfolioDecision,
    ExactActionArmCountConstraint,
    ForecastPortfolioUtilityBinding,
    validate_action_portfolio_decision,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastEvidenceMode,
    ActionForecastRequest,
    MetricForecastScale,
    ParentMetricValue,
    ResolvedActionForecastBatch,
)
from agent_evolve.ports.frontier_target import (
    CampaignPortfolioFrontierTarget,
    ObjectiveSpaceTarget,
    campaign_frontier_target_from_record,
    objective_space_target_from_campaign_target,
)
from agent_evolve.ports.portfolio_selection import PortfolioSelectionRequest
from agent_evolve.ports.structured_generator import MAX_OUTPUT_TOKENS


TARGET_ACTION_FORECAST_SCALE_POLICY_ID = "target_delta_forecast_scale"
TARGET_ACTION_FORECAST_SCALE_POLICY_VERSION = 1
TARGET_ACTION_FORECAST_SCALE_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:target-delta-forecast-scale:v1;"
    b"nonzero=absolute-parent-to-aspiration-raw-delta;"
    b"zero=one-thirty-second-objective-affine-span;"
    b"metric-id=optimization-semantic-objective-id;"
    b"workload-model-provider-fields=false;outcomes=false"
).hexdigest()
_SCALE_BINDING_DOMAIN = b"agent-evolve:target-delta-scale-binding:v1\x00"


@dataclass(frozen=True, slots=True)
class TargetConditionedActionForecastPlan:
    """Typed forecast request plus the exact target/metric bridge it realizes."""

    campaign_target: CampaignPortfolioFrontierTarget
    objective_target: ObjectiveSpaceTarget
    aliases: tuple[TargetMetricAlias, ...]
    request: ActionForecastRequest
    residual_cell: NormalizedResidualFrontierCell | None
    min_distinct_families: int | None
    require_pairwise_disjoint_parent_patches: bool

    def __post_init__(self) -> None:
        if type(self.campaign_target) is not CampaignPortfolioFrontierTarget:
            raise TypeError("campaign_target must be exact")
        self.campaign_target.__post_init__()
        if type(self.objective_target) is not ObjectiveSpaceTarget:
            raise TypeError("objective_target must be exact")
        self.objective_target.__post_init__()
        if (
            self.objective_target.campaign_target_sha256
            != self.campaign_target.target_sha256
        ):
            raise ValueError("objective target belongs to a foreign campaign target")
        if type(self.aliases) is not tuple or any(
            type(value) is not TargetMetricAlias for value in self.aliases
        ):
            raise TypeError("aliases must contain exact target metric aliases")
        for value in self.aliases:
            value.__post_init__()
        if tuple(value.target_metric_id for value in self.aliases) != (
            self.objective_target.metric_ids
        ):
            raise ValueError("aliases must canonically cover the objective target")
        if type(self.request) is not ActionForecastRequest:
            raise TypeError("request must be an exact ActionForecastRequest")
        self.request.__post_init__()
        if (
            self.request.finite_variation_contract.parent_configuration_sha256
            != self.campaign_target.parent_configuration_sha256
        ):
            raise ValueError("forecast contract parent differs from frontier target")
        if set(value.forecast_metric_id for value in self.aliases) != set(
            self.request.required_metric_ids
        ):
            raise ValueError("forecast request and target aliases cover different metrics")
        if self.residual_cell is not None:
            if type(self.residual_cell) is not NormalizedResidualFrontierCell:
                raise TypeError("residual_cell must be exact or None")
            self.residual_cell.__post_init__()
            if (
                self.residual_cell.campaign_target_sha256
                != self.campaign_target.target_sha256
            ):
                raise ValueError("residual cell belongs to a foreign target")
        if self.min_distinct_families is not None and (
            type(self.min_distinct_families) is not int
            or self.min_distinct_families <= 0
        ):
            raise ValueError("min_distinct_families must be positive or None")
        if type(self.require_pairwise_disjoint_parent_patches) is not bool:
            raise TypeError(
                "require_pairwise_disjoint_parent_patches must be exact bool"
            )

    def assess(
        self,
        forecasts: ResolvedActionForecastBatch,
    ) -> tuple[ActionTargetRealization, ...]:
        self.__post_init__()
        return assess_action_target_realization(
            target=self.objective_target,
            forecasts=forecasts,
            aliases=self.aliases,
        )


def _target_from_selection_context(
    request: PortfolioSelectionRequest,
) -> tuple[CampaignPortfolioFrontierTarget, ObjectiveSpaceTarget]:
    context = thaw_json(request.context)
    if type(context) is not dict:  # pragma: no cover - frozen root is closed.
        raise AssertionError("portfolio selection context is not an object")
    record = context.get(CAMPAIGN_FRONTIER_TARGET_KEY)
    if record is None:
        raise ValueError("portfolio selection context omits a frontier target")
    campaign_target = campaign_frontier_target_from_record(record)
    objective_target = objective_space_target_from_campaign_target(campaign_target)
    if objective_target is None:
        raise ValueError("campaign frontier target omits raw objective-space axes")
    if campaign_target.parent_configuration_sha256 != (
        request.finite_variation_contract.parent_configuration_sha256
    ):
        raise ValueError("campaign frontier target names a foreign parent")
    return campaign_target, objective_target


def _objective_metric_aliases(
    target: ObjectiveSpaceTarget,
    semantics: OptimizationSemantics,
) -> tuple[TargetMetricAlias, ...]:
    objectives_by_name = {
        metric.name: metric
        for metric in semantics.metrics
        if metric.role is MetricRole.OBJECTIVE
    }
    aliases: list[TargetMetricAlias] = []
    for axis in target.axes:
        metric = objectives_by_name.get(axis.metric_id)
        if metric is None:
            raise ValueError(
                "frontier target metric is absent from objective semantics: "
                + axis.metric_id
            )
        expected_sense = (
            MetricSense.MINIMIZE if axis.goal == "min" else MetricSense.MAXIMIZE
        )
        if metric.sense is not expected_sense:
            raise ValueError("frontier target goal differs from objective semantics")
        aliases.append(TargetMetricAlias(axis.metric_id, metric.metric_id))
    return tuple(sorted(aliases, key=lambda value: value.target_metric_id))


def _scale_definition_sha256(
    *,
    target: ObjectiveSpaceTarget,
    target_metric_id: str,
    forecast_metric_id: str,
    scale: float,
) -> str:
    payload = json.dumps(
        {
            "schema_version": 1,
            "policy_id": TARGET_ACTION_FORECAST_SCALE_POLICY_ID,
            "policy_version": TARGET_ACTION_FORECAST_SCALE_POLICY_VERSION,
            "policy_definition_sha256": (
                TARGET_ACTION_FORECAST_SCALE_POLICY_DEFINITION_SHA256
            ),
            "campaign_target_sha256": target.campaign_target_sha256,
            "target_metric_id": target_metric_id,
            "forecast_metric_id": forecast_metric_id,
            "delta_scale_hex": scale.hex(),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(_SCALE_BINDING_DOMAIN + payload).hexdigest()


def build_target_conditioned_action_forecast_plan(
    *,
    selection_request: PortfolioSelectionRequest,
    optimization_semantics: OptimizationSemantics,
    call_id: LLMCallId,
    forecast_contract: FiniteVariationContract | None = None,
    action_semantics: ActionSpaceSemantics | None = None,
    evidence_mode: ActionForecastEvidenceMode = (
        ActionForecastEvidenceMode.CATALOG_ONLY
    ),
    operation: str = "forecast_target_realization",
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float | None = None,
) -> TargetConditionedActionForecastPlan:
    """Build one complete consequence-forecast request from generic contracts."""

    if type(selection_request) is not PortfolioSelectionRequest:
        raise TypeError("selection_request must be exact PortfolioSelectionRequest")
    selection_request.__post_init__()
    if type(optimization_semantics) is not OptimizationSemantics:
        raise TypeError("optimization_semantics must be exact")
    optimization_semantics.__post_init__()
    if type(evidence_mode) is not ActionForecastEvidenceMode:
        raise TypeError("evidence_mode must be exact ActionForecastEvidenceMode")
    resolved_contract = (
        selection_request.finite_variation_contract
        if forecast_contract is None
        else forecast_contract
    )
    if type(resolved_contract) is not FiniteVariationContract:
        raise TypeError("forecast_contract must be an exact contract or None")
    resolved_contract.__post_init__()
    source_contract = selection_request.finite_variation_contract
    if (
        resolved_contract.parent_configuration_sha256
        != source_contract.parent_configuration_sha256
    ):
        raise ValueError("forecast contract belongs to a foreign parent")
    source_options = {
        value.identity_sha256: value for value in source_contract.options
    }
    if any(
        source_options.get(value.identity_sha256) != value
        for value in resolved_contract.options
    ):
        raise ValueError("forecast contract is not an exact source-contract subset")
    campaign_target, objective_target = _target_from_selection_context(
        selection_request
    )
    aliases = _objective_metric_aliases(objective_target, optimization_semantics)
    alias_by_target = {
        value.target_metric_id: value.forecast_metric_id for value in aliases
    }
    parent_values: list[ParentMetricValue] = []
    scales: list[MetricForecastScale] = []
    for axis in objective_target.axes:
        forecast_metric_id = alias_by_target[axis.metric_id]
        scale = abs(axis.signed_parent_to_aspiration_delta)
        if scale == 0.0:
            scale = abs(axis.reference - axis.ideal) / 32.0
        if scale <= 0.0:  # pragma: no cover - target axis validation closes span.
            raise ValueError("target-conditioned forecast scale is not positive")
        parent_values.append(ParentMetricValue(forecast_metric_id, axis.parent_value))
        scales.append(
            MetricForecastScale(
                forecast_metric_id,
                float(scale),
                _scale_definition_sha256(
                    target=objective_target,
                    target_metric_id=axis.metric_id,
                    forecast_metric_id=forecast_metric_id,
                    scale=float(scale),
                ),
            )
        )
    parent_values.sort(key=lambda value: value.metric_id)
    scales.sort(key=lambda value: value.metric_id)
    resolved_action_semantics = (
        derive_action_space_semantics(resolved_contract)
        if action_semantics is None
        else action_semantics
    )
    resolved_action_semantics.validate_contract_binding(
        (
            resolved_contract.catalog_id,
            resolved_contract.catalog_version,
            resolved_contract.catalog_definition_sha256,
        ),
        tuple(value.family for value in resolved_contract.options),
    )
    grounded = evidence_mode is ActionForecastEvidenceMode.GROUNDED
    forecast_request = ActionForecastRequest(
        call_id=call_id,
        operation=operation,
        instruction=(
            "Forecast signed child-minus-parent objective changes and validity "
            "for every sealed finite action. Use the raw parent-to-aspiration "
            "target magnitudes in context to distinguish undershoot, plausible "
            "target closure, and destructive overshoot. Estimate consequences; "
            "do not select actions or claim evaluator observations."
        ),
        context=selection_request.context,
        optimization_semantics=optimization_semantics,
        action_semantics=resolved_action_semantics,
        finite_variation_contract=resolved_contract,
        cards=selection_request.cards if grounded else (),
        source_registry=selection_request.source_registry if grounded else None,
        evidence_mode=evidence_mode,
        experimental_view_receipt=(
            selection_request.experimental_view_receipt if grounded else None
        ),
        parent_metric_values=tuple(parent_values),
        metric_scales=tuple(scales),
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    return TargetConditionedActionForecastPlan(
        campaign_target=campaign_target,
        objective_target=objective_target,
        aliases=aliases,
        request=forecast_request,
        residual_cell=residual_frontier_cell_from_target(
            campaign_target=campaign_target,
            objective_target=objective_target,
        ),
        min_distinct_families=selection_request.min_distinct_families,
        require_pairwise_disjoint_parent_patches=(
            selection_request.require_pairwise_disjoint_parent_patches
        ),
    )


def build_target_conditioned_action_utility(
    *,
    plan: TargetConditionedActionForecastPlan,
    forecasts: ResolvedActionForecastBatch,
    portfolio_size: int,
    eligible_option_ids: tuple[str, ...] | None = None,
    utility_mode: str = "target_closure",
    role_slots: tuple[int, int, int] | None = None,
) -> ForecastPortfolioUtilityBinding:
    """Build one identified, workload-neutral acquisition utility."""

    if type(plan) is not TargetConditionedActionForecastPlan:
        raise TypeError("plan must be exact TargetConditionedActionForecastPlan")
    plan.__post_init__()
    plan.assess(forecasts)
    eligible = (
        tuple(sorted(value.option_id for value in forecasts.forecasts))
        if eligible_option_ids is None
        else eligible_option_ids
    )
    if utility_mode == "target_closure":
        return ResidualTargetClosurePortfolioUtility(
            target=plan.objective_target,
            aliases=plan.aliases,
        ).binding()
    if utility_mode == "expected_hypervolume":
        if plan.residual_cell is None:
            raise ValueError(
                "expected_hypervolume requires residual frontier cell anchors"
            )
        return ResidualCellExpectedHypervolumeUtility(
            target=plan.objective_target,
            cell=plan.residual_cell,
            aliases=plan.aliases,
        ).binding()
    if utility_mode == "reliability_adjusted_expected_hypervolume":
        if plan.residual_cell is None:
            raise ValueError(
                "reliability-adjusted expected_hypervolume requires residual "
                "frontier cell anchors"
            )
        return ReliabilityAdjustedResidualCellExpectedHypervolumeUtility(
            target=plan.objective_target,
            cell=plan.residual_cell,
            aliases=plan.aliases,
        ).binding()
    if utility_mode == "role_factorized":
        if plan.residual_cell is None:
            raise ValueError(
                "role_factorized requires residual frontier cell anchors"
            )
        resolved_role_slots = (
            (portfolio_size - 2, 1, 1) if role_slots is None else role_slots
        )
        if (
            type(resolved_role_slots) is not tuple
            or len(resolved_role_slots) != 3
            or any(type(value) is not int for value in resolved_role_slots)
        ):
            raise TypeError("role_slots must be an exact (exploit, bridge, probe) tuple")
        exploit_slots, bridge_slots, probe_slots = resolved_role_slots
        if exploit_slots + bridge_slots + probe_slots != portfolio_size:
            raise ValueError("role_slots must sum to portfolio_size")
        role_utility = build_role_factorized_action_utility(
            forecast_request=plan.request,
            forecasts=forecasts,
            eligible_option_ids=eligible,
            exploit_utility=(
                ReliabilityAdjustedResidualCellExpectedHypervolumeUtility(
                    target=plan.objective_target,
                    cell=plan.residual_cell,
                    aliases=plan.aliases,
                )
            ),
            bridge_utility=ResidualTargetClosurePortfolioUtility(
                target=plan.objective_target,
                aliases=plan.aliases,
            ),
            exploit_slots=exploit_slots,
            bridge_slots=bridge_slots,
            probe_slots=probe_slots,
        )
        return role_utility.binding()
    raise ValueError(
        "utility_mode must be target_closure, expected_hypervolume, "
        "reliability_adjusted_expected_hypervolume, or role_factorized"
    )


def allocate_target_conditioned_actions(
    *,
    plan: TargetConditionedActionForecastPlan,
    forecasts: ResolvedActionForecastBatch,
    portfolio_size: int,
    eligible_option_ids: tuple[str, ...] | None = None,
    risk_aversion: float = 0.5,
    diversity_weight: float = 0.05,
    beam_width: int = 256,
    utility_mode: str = "target_closure",
    required_option_ids: tuple[str, ...] = (),
    role_slots: tuple[int, int, int] | None = None,
    exact_arm_count_constraints: tuple[ExactActionArmCountConstraint, ...] = (),
    minimum_single_path_interventions: int = 0,
    minimum_disjoint_parent_patch_pairs: int = 0,
) -> ActionAllocationResult:
    """Allocate a robust target-closing set from one authenticated forecast."""

    if type(plan) is not TargetConditionedActionForecastPlan:
        raise TypeError("plan must be exact TargetConditionedActionForecastPlan")
    plan.__post_init__()
    plan.assess(forecasts)
    eligible = (
        tuple(sorted(value.option_id for value in forecasts.forecasts))
        if eligible_option_ids is None
        else eligible_option_ids
    )
    utility = build_target_conditioned_action_utility(
        plan=plan,
        forecasts=forecasts,
        portfolio_size=portfolio_size,
        eligible_option_ids=eligible,
        utility_mode=utility_mode,
        role_slots=role_slots,
    )
    request = ActionAllocationRequest(
        forecast_request=plan.request,
        forecasts=forecasts,
        eligible_option_ids=eligible,
        portfolio_size=portfolio_size,
        utility=utility,
        min_distinct_families=plan.min_distinct_families,
        require_pairwise_disjoint_parent_patches=(
            plan.require_pairwise_disjoint_parent_patches
        ),
        minimum_single_path_interventions=minimum_single_path_interventions,
        minimum_disjoint_parent_patch_pairs=(
            minimum_disjoint_parent_patch_pairs
        ),
        required_option_ids=required_option_ids,
        exact_arm_count_constraints=exact_arm_count_constraints,
    )
    return FeasibleBeamRiskAdjustedDiversityAllocator(
        risk_aversion=risk_aversion,
        diversity_weight=diversity_weight,
        beam_width=beam_width,
    ).allocate(request)


def audit_target_conditioned_role_allocation(
    *,
    plan: TargetConditionedActionForecastPlan,
    forecasts: ResolvedActionForecastBatch,
    decision: ActionPortfolioDecision,
    eligible_option_ids: tuple[str, ...] | None = None,
    role_slots: tuple[int, int, int] | None = None,
    exact_arm_count_constraints: tuple[ExactActionArmCountConstraint, ...] = (),
    required_option_ids: tuple[str, ...] = (),
    minimum_single_path_interventions: int = 0,
    minimum_disjoint_parent_patch_pairs: int = 0,
) -> tuple[RoleAssignmentAudit, ...]:
    """Rebuild, validate, and explain a final role-factorized allocation."""

    if type(plan) is not TargetConditionedActionForecastPlan:
        raise TypeError("plan must be exact TargetConditionedActionForecastPlan")
    plan.__post_init__()
    plan.assess(forecasts)
    if type(decision) is not ActionPortfolioDecision:
        raise TypeError("decision must be exact ActionPortfolioDecision")
    decision.__post_init__()
    eligible = (
        tuple(sorted(value.option_id for value in forecasts.forecasts))
        if eligible_option_ids is None
        else eligible_option_ids
    )
    utility = build_target_conditioned_action_utility(
        plan=plan,
        forecasts=forecasts,
        portfolio_size=len(decision.members),
        eligible_option_ids=eligible,
        utility_mode="role_factorized",
        role_slots=role_slots,
    )
    request = ActionAllocationRequest(
        forecast_request=plan.request,
        forecasts=forecasts,
        eligible_option_ids=eligible,
        portfolio_size=len(decision.members),
        utility=utility,
        min_distinct_families=plan.min_distinct_families,
        require_pairwise_disjoint_parent_patches=(
            plan.require_pairwise_disjoint_parent_patches
        ),
        minimum_single_path_interventions=minimum_single_path_interventions,
        minimum_disjoint_parent_patch_pairs=(
            minimum_disjoint_parent_patch_pairs
        ),
        required_option_ids=required_option_ids,
        exact_arm_count_constraints=exact_arm_count_constraints,
    )
    validate_action_portfolio_decision(request, decision)
    role_utility = utility.utility
    if type(role_utility) is not RoleFactorizedActionPortfolioUtility:
        raise AssertionError("role utility factory returned a foreign utility")
    return audit_role_factorized_action_portfolio(
        utility=role_utility,
        forecast_request=plan.request,
        forecasts=forecasts,
        selected_option_ids=tuple(
            sorted(value.option_id for value in decision.members)
        ),
    )


__all__ = [
    "TARGET_ACTION_FORECAST_SCALE_POLICY_DEFINITION_SHA256",
    "TARGET_ACTION_FORECAST_SCALE_POLICY_ID",
    "TARGET_ACTION_FORECAST_SCALE_POLICY_VERSION",
    "TargetConditionedActionForecastPlan",
    "allocate_target_conditioned_actions",
    "audit_target_conditioned_role_allocation",
    "build_target_conditioned_action_utility",
    "build_target_conditioned_action_forecast_plan",
]
