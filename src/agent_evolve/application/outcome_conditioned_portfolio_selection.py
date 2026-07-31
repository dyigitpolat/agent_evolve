"""All-action, outcome-conditioned portfolio selection for real campaigns.

The policy is an adapter between the campaign's stable
``PortfolioSelectionPolicy`` port and the generic action-forecast/allocation
ports.  It asks a model to forecast every action in a sealed, outcome-blind
candidate-pool projection (or the complete contract when no projection is
bound), optionally
overlays evaluator-independent exact metric projections, and lets trusted code
construct the final hard-feasible set.  Workload adapters may enrich action
semantics or exact cheap metric cells, but never rank or select actions.

The implementation also reconciles the existing bounded memory-dose contract:
memory-supported options become explicit hard requirements, while a sealed
proposal witness is extended only as far as necessary to preserve the original
proposal/evaluation join.  This keeps memory attribution causal and makes the
new selector usable in later generations rather than only in memory-free
assays.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from agent_evolve.application.action_forecast_partitioning import (
    PartitionedActionForecastPolicy,
    ResolvedActionForecastHealthAssessment,
    assess_resolved_action_forecast_health,
    build_action_forecast_partition_layout,
    lenient_action_forecast_health_policy,
)
from agent_evolve.application.action_metric_projection import (
    ActionMetricProjectionOverlayResult,
    apply_exact_action_metric_projections,
)
from agent_evolve.application.campaign_search_phase import (
    resolve_campaign_search_phase_context,
)
from agent_evolve.application.campaign_variation_trace import (
    project_finite_contract_proposal_topology,
)
from agent_evolve.application.contextual_action_allocation import (
    contextual_action_arm_count_constraints,
    contextual_allocation_realization,
    selected_variation_source_ids,
)
from agent_evolve.application.empirical_consequence_calibration import (
    ActionConsequenceCalibrationPolicy,
    ActionConsequenceCalibrationResult,
)
from agent_evolve.application.global_wave_action_allocation import (
    GlobalWaveActionAllocationCoordinator,
    GlobalWaveActionAllocationLane,
    GlobalWaveActionAllocationLaneResult,
    RoleSlots,
)
from agent_evolve.application.target_conditioned_action_forecast import (
    TargetConditionedActionForecastPlan,
    allocate_target_conditioned_actions,
    audit_target_conditioned_role_allocation,
    build_target_conditioned_action_forecast_plan,
)
from agent_evolve.core.action_semantics import ActionSpaceSemantics
from agent_evolve.core.optimization_semantics import OptimizationSemantics
from agent_evolve.domain.finite_variation import FiniteVariationContract
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.ports.action_allocation import ActionAllocationResult
from agent_evolve.ports.action_forecast import (
    ActionForecastEvidenceMode,
    ActionForecastPartitionPolicyBinding,
    ResolvedActionForecast,
    ResolvedActionForecastBatch,
)
from agent_evolve.ports.action_metric_projection import (
    ActionMetricProjector,
    ExactActionMetricProjectionBatch,
)
from agent_evolve.ports.contextual_search_allocation import (
    ContextualPortfolioAllocationContract,
    ContextualPortfolioAllocationRealization,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.portfolio_memory_dose import (
    PortfolioMemoryDoseAssessment,
    PortfolioMemoryDoseMember,
    assess_evaluated_portfolio_memory_dose,
    assess_proposed_portfolio_memory_dose,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioMemberDraft,
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
    PortfolioSelectionSupplementalAudit,
    resolve_ranked_portfolio_decision,
)
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationScope,
    ForecastConfidenceBin,
    ForecastPredictionReceipt,
)
from agent_evolve.policies.selection.common_candidate_pool import (
    CommonCandidatePoolDecision,
)


OUTCOME_CONDITIONED_PORTFOLIO_POLICY_ID = "outcome_conditioned_expert_portfolio"
OUTCOME_CONDITIONED_PORTFOLIO_POLICY_VERSION = 8
OUTCOME_CONDITIONED_PORTFOLIO_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:outcome-conditioned-expert-portfolio:v8;"
    b"proposal-universe=full-sealed-contract-with-outcome-blind-authenticated-"
    b"candidate-pool-forecast-projection;"
    b"consequence-source=partitioned-llm-plus-prior-only-calibration-port;"
    b"optional-authority=evaluator-independent-exact-metric-projection;"
    b"bootstrap-acquisition=target-closure-when-residual-cell-unidentified;"
    b"nonterminal-residual-acquisition=role-factorized-exploit-bridge-probe;"
    b"terminal-residual-acquisition=reliability-adjusted-expected-hypervolume;"
    b"hard-constraints=family-patch-required-option-memory-dose-contextual-"
    b"source-and-operator-marginals;"
    b"forecast-health=model-authoritative-unprojected-cells-only;"
    b"model-authors-consequences-not-selection;"
    b"optional-wave-authority=global-role-budget-and-cross-lane-diversity;"
    b"audit-proposal-topology=engine-authenticated-forecast-contract;"
    b"workload-branches=false;real-current-future-outcomes=false"
).hexdigest()
_FORECAST_CALL_DOMAIN = b"agent-evolve:outcome-conditioned-forecast-call:v1\x00"
_BOUND_POLICY_DOMAIN = b"agent-evolve:outcome-conditioned-policy-binding:v1\x00"


ActionSemanticsFactory = Callable[[FiniteVariationContract], ActionSpaceSemantics]
ContextualAllocationProvider = Callable[
    [PortfolioSelectionRequest], ContextualPortfolioAllocationContract | None
]
CandidatePoolProvider = Callable[
    [PortfolioSelectionRequest], CommonCandidatePoolDecision | None
]


def _forecast_contract_projection(
    request: PortfolioSelectionRequest,
    decision: CommonCandidatePoolDecision | None,
) -> tuple[FiniteVariationContract, dict[str, object]]:
    """Project the expensive forecast surface without hiding union support."""

    source = request.finite_variation_contract
    if decision is None:
        return source, {
            "mode": "complete_finite_contract",
            "source_contract_sha256": source.identity_sha256,
            "forecast_contract_sha256": source.identity_sha256,
            "source_option_count": len(source.options),
            "forecast_option_count": len(source.options),
            "common_candidate_pool_decision_sha256": None,
        }
    if type(decision) is not CommonCandidatePoolDecision:
        raise TypeError("candidate-pool provider returned a foreign decision")
    decision.__post_init__()
    if (
        decision.finite_contract_sha256 != source.identity_sha256
        or decision.parent_configuration_sha256
        != source.parent_configuration_sha256
        or decision.evaluation_size != request.portfolio_size
    ):
        raise ValueError("candidate-pool decision differs from the selection request")
    options = {value.option_id: value for value in source.options}
    try:
        projected_options = tuple(options[value] for value in decision.option_ids)
    except KeyError as error:  # pragma: no cover - decision validation closes this.
        raise ValueError("candidate-pool option escapes the source contract") from error
    projected = FiniteVariationContract(
        catalog_id=source.catalog_id,
        catalog_version=source.catalog_version,
        catalog_definition_sha256=source.catalog_definition_sha256,
        parent_configuration=source.parent_configuration,
        options=projected_options,
    )
    return projected, {
        "mode": "authenticated_outcome_blind_candidate_pool",
        "source_contract_sha256": source.identity_sha256,
        "forecast_contract_sha256": projected.identity_sha256,
        "source_option_count": len(source.options),
        "forecast_option_count": len(projected.options),
        "common_candidate_pool_decision_sha256": decision.decision_sha256,
        "common_candidate_pool_state_identity_sha256": (
            decision.state_identity_sha256
        ),
        "common_candidate_pool_task_identity_sha256": decision.task_identity_sha256,
        "forecast_option_ids": list(decision.option_ids),
        "outcomes_consulted": False,
        "model_or_provider_fields_consulted": False,
    }


def _identified_outcome_conditioned_acquisition(
    *,
    residual_cell_identified: bool,
    terminal: bool,
) -> tuple[str, str]:
    """Choose only an acquisition whose geometric estimand is identified."""

    if type(residual_cell_identified) is not bool or type(terminal) is not bool:
        raise TypeError("acquisition evidence flags must be exact bools")
    if not residual_cell_identified:
        return "target_closure", "directional_affine_bootstrap"
    if terminal:
        return (
            "reliability_adjusted_expected_hypervolume",
            "residual_frontier_cell",
        )
    return "role_factorized", "residual_frontier_cell"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    if type(frozen) is not FrozenJsonObject:  # pragma: no cover - closed root.
        raise AssertionError("selector audit did not freeze to an object")
    return frozen


def _plain_global_allocation_audit(
    value: GlobalWaveActionAllocationLaneResult | None,
) -> dict[str, object] | None:
    """Cross the immutable typed-JSON boundary exactly once for nesting."""

    if value is None:
        return None
    if type(value) is not GlobalWaveActionAllocationLaneResult:
        raise TypeError("global allocation must be an exact lane result or None")
    value.__post_init__()
    record = thaw_json(value.audit)
    if type(record) is not dict:  # pragma: no cover - frozen root is closed.
        raise AssertionError("global allocation audit did not thaw to an object")
    return record


def _forecast_call_id(request: PortfolioSelectionRequest) -> LLMCallId:
    digest = hashlib.sha256(
        _FORECAST_CALL_DOMAIN + bytes.fromhex(request.request_sha256)
    ).hexdigest()
    return LLMCallId(f"call_outcome_forecast_{digest[:40]}")


def _resolve_model_authority_health(
    *,
    request_forecasts: ResolvedActionForecastBatch,
    health: ResolvedActionForecastHealthAssessment,
    exact_projections: ExactActionMetricProjectionBatch | None,
) -> tuple[bool, dict[str, object]]:
    """Apply collapse gates only where the model retains metric authority.

    A complete exact projection removes the language model's decision
    authority for one metric. Collapsed raw values for that metric remain in
    the audit but cannot veto the wave. A partial projection grants no such
    exemption.
    """

    if type(request_forecasts) is not ResolvedActionForecastBatch:
        raise TypeError("request_forecasts must be an exact resolved batch")
    request_forecasts.__post_init__()
    if type(health) is not ResolvedActionForecastHealthAssessment:
        raise TypeError("health must be an exact resolved assessment")
    health.__post_init__()
    option_ids = {value.option_id for value in request_forecasts.forecasts}
    projected_by_metric: dict[str, set[str]] = {}
    if exact_projections is not None:
        if type(exact_projections) is not ExactActionMetricProjectionBatch:
            raise TypeError("exact_projections must be an exact batch or None")
        exact_projections.__post_init__()
        for value in exact_projections.projections:
            projected_by_metric.setdefault(value.metric_id, set()).add(value.option_id)
    fully_projected = tuple(
        sorted(
            metric_id
            for metric_id, projected_options in projected_by_metric.items()
            if projected_options == option_ids
        )
    )
    assessment_by_metric = {
        value.metric_id: value for value in health.metric_assessments
    }
    authoritative = tuple(sorted(set(assessment_by_metric).difference(fully_projected)))
    unresolved_failed = tuple(
        metric_id
        for metric_id in authoritative
        if not assessment_by_metric[metric_id].passes
    )
    signatures: set[tuple[object, ...]] = set()
    if authoritative:
        for forecast in request_forecasts.forecasts:
            metrics = {value.metric_id: value for value in forecast.metric_forecasts}
            signatures.add(
                (
                    forecast.probability_valid.hex(),
                    *(
                        (
                            metrics[metric_id].p10_delta.hex(),
                            metrics[metric_id].p50_delta.hex(),
                            metrics[metric_id].p90_delta.hex(),
                            metrics[metric_id].confidence.hex(),
                        )
                        for metric_id in authoritative
                    ),
                )
            )
    threshold_applied = (
        len(request_forecasts.forecasts) >= health.health_policy.minimum_rows
    )
    signature_passes = (
        not authoritative
        or not threshold_applied
        or len(signatures) >= health.health_policy.minimum_distinct_signatures
    )
    passes = signature_passes and not unresolved_failed
    return passes, {
        "schema_version": 1,
        "raw_health_passes": health.passes,
        "fully_projected_metric_ids": list(fully_projected),
        "model_authoritative_metric_ids": list(authoritative),
        "unresolved_failed_metric_ids": list(unresolved_failed),
        "model_authoritative_distinct_row_signature_count": len(signatures),
        "threshold_applied": threshold_applied,
        "signature_gate_passes": signature_passes,
        "passes": passes,
    }


def _evidence_mode(request: PortfolioSelectionRequest) -> ActionForecastEvidenceMode:
    if (
        request.source_registry is not None
        and request.experimental_view_receipt is not None
        and all(card.source_binding is not None for card in request.cards)
        and any(card.finite_action_evidence for card in request.cards)
    ):
        return ActionForecastEvidenceMode.GROUNDED
    return ActionForecastEvidenceMode.CATALOG_ONLY


def _telemetry_record(value: AgenticCallTelemetry) -> dict[str, object]:
    value.__post_init__()
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


def _aggregate_telemetry(
    values: tuple[AgenticCallTelemetry, ...],
    *,
    wall_latency_ns: int,
) -> AgenticCallTelemetry:
    if not values:
        raise ValueError("partitioned portfolio selection requires call telemetry")
    for value in values:
        value.__post_init__()
    requested_models = {value.requested_model for value in values}
    resolved_models = {value.resolved_model for value in values}
    resolved_providers = {value.resolved_provider for value in values}
    if (
        len(requested_models) != 1
        or len(resolved_models) != 1
        or len(resolved_providers) != 1
    ):
        raise ValueError("physical forecast blocks resolved to heterogeneous models")
    costs = tuple(value.cost_usd for value in values)
    cost: Decimal | None
    if any(value is None for value in costs):
        cost = None
    else:
        cost = sum((value for value in costs if value is not None), Decimal(0))
    finish_reasons = {value.finish_reason for value in values}
    return AgenticCallTelemetry(
        requested_model=next(iter(requested_models)),
        resolved_model=next(iter(resolved_models)),
        resolved_provider=next(iter(resolved_providers)),
        provider_response_id=None,
        finish_reason=(
            next(iter(finish_reasons)) if len(finish_reasons) == 1 else None
        ),
        input_tokens=sum(value.input_tokens for value in values),
        output_tokens=sum(value.output_tokens for value in values),
        reasoning_tokens=sum(value.reasoning_tokens for value in values),
        cache_read_tokens=sum(value.cache_read_tokens for value in values),
        cache_write_tokens=sum(value.cache_write_tokens for value in values),
        cost_usd=cost,
        latency_ns=wall_latency_ns,
        # Physical block count is recorded separately.  ``attempt_count`` keeps
        # its established retry meaning for the logical receipt.
        attempt_count=max(value.attempt_count for value in values),
    )


def _direction(metric: object) -> MetricEffectDirection:
    p10 = getattr(metric, "p10_delta")
    p50 = getattr(metric, "p50_delta")
    p90 = getattr(metric, "p90_delta")
    if any(
        type(value) is not float or not math.isfinite(value)
        for value in (p10, p50, p90)
    ):
        raise TypeError("resolved forecast deltas must be finite exact floats")
    if p10 == p50 == p90 == 0.0:
        return MetricEffectDirection.UNCHANGED
    if p90 < 0.0:
        return MetricEffectDirection.DECREASE
    if p10 > 0.0:
        return MetricEffectDirection.INCREASE
    return MetricEffectDirection.UNKNOWN


def _selected_card_keys(forecast: ResolvedActionForecast) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                citation.card_key
                for metric in forecast.metric_forecasts
                for citation in metric.citations
            }
        )
    )


def _preference_scores(
    plan: TargetConditionedActionForecastPlan,
    forecasts: ResolvedActionForecastBatch,
) -> dict[str, float]:
    """Cheap deterministic ordering for memory witnesses and proposal padding."""

    scores: dict[str, float] = {}
    for value in plan.assess(forecasts):
        p50 = value.scenarios[1]
        score = value.probability_valid * p50.shortfall_reduction_l1
        if not math.isfinite(score):  # pragma: no cover - typed inputs close this.
            raise RuntimeError("target preference score became non-finite")
        scores[value.option_id] = float(score)
    return scores


def _memory_assignment_candidates(
    request: PortfolioSelectionRequest,
    *,
    preference: dict[str, float],
    max_options_per_card: int,
    max_assignments: int,
) -> tuple[dict[str, str], ...]:
    """Enumerate bounded card-to-distinct-option matchings, best first."""

    dose = request.memory_dose_contract
    if dose is None:
        return ({},)
    dose.__post_init__()
    if dose.maximum_cards_per_member < 1:
        raise ValueError("memory dose cannot attribute any selected member")
    supports = dose.card_supports
    evaluated_lower, evaluated_upper = dose.evaluated_supported_member_bounds
    proposed_lower, proposed_upper = dose.proposed_supported_member_bounds
    if dose.require_every_assigned_card:
        subset_sizes = (len(supports),)
    else:
        low = max(evaluated_lower, proposed_lower)
        high = min(evaluated_upper, proposed_upper, len(supports))
        subset_sizes = tuple(range(low, high + 1))
    results: list[dict[str, str]] = []
    for size in subset_sizes:
        if (
            not evaluated_lower <= size <= evaluated_upper
            or not proposed_lower <= size <= proposed_upper
            or request.portfolio_size - size
            < dose.minimum_unattributed_evaluated_members
        ):
            continue
        for selected_supports in itertools.combinations(supports, size):
            option_lists = tuple(
                tuple(
                    option_id
                    for option_id, _identity in sorted(
                        support.compatible_options,
                        key=lambda item: (
                            -preference.get(item[0], float("-inf")),
                            item[1],
                            item[0],
                        ),
                    )[:max_options_per_card]
                )
                for support in selected_supports
            )
            for option_ids in itertools.product(*option_lists):
                if len(set(option_ids)) != len(option_ids):
                    continue
                results.append(
                    {
                        option_id: support.card_key
                        for support, option_id in zip(
                            selected_supports,
                            option_ids,
                            strict=True,
                        )
                    }
                )
                if len(results) >= max_assignments:
                    return tuple(results)
    if not results:
        raise ValueError("bounded memory dose has no candidate attribution matching")
    return tuple(results)


def _memory_assessments(
    request: PortfolioSelectionRequest,
    *,
    allocation: ActionAllocationResult,
    card_by_option: dict[str, str],
    preference: dict[str, float],
) -> tuple[
    dict[str, tuple[str, ...]],
    PortfolioMemoryDoseAssessment | None,
    PortfolioMemoryDoseAssessment | None,
]:
    selected = allocation.decision.members
    selected_support = {
        value.option_id: (
            ()
            if value.option_id not in card_by_option
            else (card_by_option[value.option_id],)
        )
        for value in selected
    }
    dose = request.memory_dose_contract
    if dose is None:
        return selected_support, None, None
    supported_count = sum(bool(value) for value in selected_support.values())
    proposed_size = max(
        request.portfolio_size,
        supported_count + dose.minimum_unattributed_proposed_members,
    )
    contract = request.finite_variation_contract
    if proposed_size > len(contract.options):
        raise ValueError("memory-dose proposal witness exceeds the finite contract")
    selected_ids = {value.option_id for value in selected}
    extras = sorted(
        (value for value in contract.options if value.option_id not in selected_ids),
        key=lambda value: (
            -preference.get(value.option_id, float("-inf")),
            value.identity_sha256,
            value.option_id,
        ),
    )[: proposed_size - len(selected)]
    identities = {value.option_id: value.identity_sha256 for value in contract.options}
    proposal_ids = tuple(value.option_id for value in selected) + tuple(
        value.option_id for value in extras
    )
    proposed_members = tuple(
        PortfolioMemoryDoseMember(
            rank=rank,
            option_id=option_id,
            option_identity_sha256=identities[option_id],
            supporting_card_keys=selected_support.get(option_id, ()),
        )
        for rank, option_id in enumerate(proposal_ids, start=1)
    )
    proposed = assess_proposed_portfolio_memory_dose(dose, proposed_members)
    if not proposed.passed:
        raise ValueError("constructed all-action proposal violates memory dose")
    evaluated_members = tuple(
        PortfolioMemoryDoseMember(
            rank=rank,
            option_id=value.option_id,
            option_identity_sha256=value.option_identity_sha256,
            supporting_card_keys=selected_support[value.option_id],
        )
        for rank, value in enumerate(selected, start=1)
    )
    evaluated = assess_evaluated_portfolio_memory_dose(
        dose,
        evaluated_members,
        proposal_assessment=proposed,
    )
    if not evaluated.passed:
        raise ValueError("allocated all-action portfolio violates memory dose")
    return selected_support, proposed, evaluated


def _bound_policy_definition(
    *,
    plan: TargetConditionedActionForecastPlan,
    partition_policy: ActionForecastPartitionPolicyBinding,
    consequence_calibration: ActionConsequenceCalibrationResult | None,
    metric_projection: ActionMetricProjectionOverlayResult | None,
    risk_aversion: float,
    diversity_weight: float,
    beam_width: int,
    global_allocation_receipt_sha256: str | None,
    contextual_allocation_contract_sha256: str | None,
    common_candidate_pool_decision_sha256: str | None,
) -> str:
    return hashlib.sha256(
        _BOUND_POLICY_DOMAIN
        + _canonical_json(
            {
                "base_definition_sha256": (
                    OUTCOME_CONDITIONED_PORTFOLIO_POLICY_DEFINITION_SHA256
                ),
                "optimization_semantics_definition_sha256": (
                    plan.request.optimization_semantics.definition_sha256
                ),
                "action_semantics_definition_sha256": (
                    plan.request.action_semantics.definition_sha256
                ),
                "partition_policy_sha256": partition_policy.binding_sha256,
                "consequence_calibration_receipt_sha256": (
                    None
                    if consequence_calibration is None
                    else consequence_calibration.receipt_sha256
                ),
                "metric_projection_receipt_sha256": (
                    None
                    if metric_projection is None
                    else metric_projection.projection_receipt_sha256
                ),
                "risk_aversion_hex": risk_aversion.hex(),
                "diversity_weight_hex": diversity_weight.hex(),
                "beam_width": beam_width,
                "global_allocation_receipt_sha256": (global_allocation_receipt_sha256),
                "contextual_allocation_contract_sha256": (
                    contextual_allocation_contract_sha256
                ),
                "common_candidate_pool_decision_sha256": (
                    common_candidate_pool_decision_sha256
                ),
            }
        )
    ).hexdigest()


def _confidence_bin(value: float) -> ForecastConfidenceBin:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError("forecast confidence must be a finite exact float")
    if value >= 0.75:
        return ForecastConfidenceBin.HIGH
    if value >= 0.4:
        return ForecastConfidenceBin.MEDIUM
    if value > 0.0:
        return ForecastConfidenceBin.LOW
    return ForecastConfidenceBin.UNKNOWN


def outcome_conditioned_selected_predictions(
    *,
    scope: ForecastCalibrationScope,
    wave: object,
    result: object,
) -> tuple[ForecastPredictionReceipt, ...]:
    """Decode selected categorical forecasts without integration-specific state."""

    from agent_evolve.application.portfolio_evolution import (
        PortfolioVariationWaveRequest,
        PortfolioVariationWaveResult,
    )

    if type(scope) is not ForecastCalibrationScope:
        raise TypeError("scope must be an exact ForecastCalibrationScope")
    scope.revalidate()
    if type(wave) is not PortfolioVariationWaveRequest:
        raise TypeError("wave must be an exact PortfolioVariationWaveRequest")
    wave.__post_init__()
    if type(result) is not PortfolioVariationWaveResult:
        raise TypeError("result must be an exact PortfolioVariationWaveResult")
    result.__post_init__()
    if (
        result.receipt.request_sha256 != wave.selection_request.request_sha256
        or result.selection_decision.decision_sha256 != result.receipt.decision_sha256
    ):
        raise ValueError("portfolio result differs from its selection wave")
    audit = result.supplemental_selection_audit
    if (
        audit is None
        or audit.audit_kind != "outcome_conditioned_expert_portfolio"
        or audit.request_sha256 != wave.selection_request.request_sha256
        or audit.decision_sha256 != result.selection_decision.decision_sha256
    ):
        raise ValueError("portfolio result omits its outcome-conditioned audit")
    payload = thaw_json(audit.payload)
    if type(payload) is not dict:  # pragma: no cover - frozen root is closed.
        raise AssertionError("outcome-conditioned audit did not thaw to an object")
    aliases_raw = payload.get("metric_aliases")
    selected_raw = payload.get("selected_forecasts")
    if type(aliases_raw) is not list or type(selected_raw) is not list:
        raise TypeError("outcome-conditioned audit omits forecast decoding data")
    alias_by_target: dict[str, str] = {}
    for row in aliases_raw:
        if type(row) is not dict:
            raise TypeError("metric alias audit rows must be objects")
        values = row
        target_id = values.get("target_metric_id")
        forecast_id = values.get("forecast_metric_id")
        if type(target_id) is not str or type(forecast_id) is not str:
            raise TypeError("metric alias audit row is malformed")
        alias_by_target[target_id] = forecast_id
    selected_confidence: dict[tuple[str, str], float] = {}
    for row in selected_raw:
        if type(row) is not dict:
            raise TypeError("selected forecast audit rows must be objects")
        values = row
        option_id = values.get("option_id")
        metrics = values.get("metric_forecasts")
        if type(option_id) is not str or type(metrics) is not list:
            raise TypeError("selected forecast audit row is malformed")
        for metric in metrics:
            if type(metric) is not dict:
                raise TypeError("selected metric audit row must be an object")
            metric_values = metric
            metric_id = metric_values.get("metric_id")
            confidence_hex = metric_values.get("confidence_hex")
            if type(metric_id) is not str or type(confidence_hex) is not str:
                raise TypeError("selected metric audit row is malformed")
            confidence = float.fromhex(confidence_hex)
            if not math.isfinite(confidence):
                raise ValueError("selected metric confidence is non-finite")
            selected_confidence[(option_id, metric_id)] = confidence
    predictions: list[ForecastPredictionReceipt] = []
    for member in result.selection_decision.members:
        effects = {value.metric_id: value for value in member.effect_predictions}
        for target_metric_id in wave.selection_request.required_metric_ids:
            effect = effects[target_metric_id]
            forecast_metric_id = alias_by_target[target_metric_id]
            confidence = selected_confidence[(member.option_id, forecast_metric_id)]
            predictions.append(
                ForecastPredictionReceipt(
                    scope=scope,
                    wave_index=wave.generation,
                    selector_decision_sha256=(
                        result.selection_decision.decision_sha256
                    ),
                    parent_candidate_identity_sha256=(
                        wave.parent.occurrence.configuration_hash
                    ),
                    option_id=member.option_id,
                    option_identity_sha256=member.option_identity_sha256,
                    family=member.family,
                    metric_id=target_metric_id,
                    asserted_direction=effect.direction,
                    confidence=_confidence_bin(confidence),
                )
            )
    return tuple(predictions)


def outcome_conditioned_selected_source_ids(
    *,
    wave: object,
    result: object,
) -> tuple[str, ...]:
    """Decode evaluated proposal-source arms from sealed finite metadata."""

    from agent_evolve.application.portfolio_evolution import (
        PortfolioVariationWaveRequest,
        PortfolioVariationWaveResult,
    )

    if type(wave) is not PortfolioVariationWaveRequest:
        raise TypeError("wave must be an exact PortfolioVariationWaveRequest")
    wave.__post_init__()
    if type(result) is not PortfolioVariationWaveResult:
        raise TypeError("result must be an exact PortfolioVariationWaveResult")
    result.__post_init__()
    if (
        result.receipt.request_sha256 != wave.selection_request.request_sha256
        or result.selection_decision.decision_sha256 != result.receipt.decision_sha256
    ):
        raise ValueError("portfolio result differs from its selection wave")
    return selected_variation_source_ids(
        finite_contract=wave.selection_request.finite_variation_contract,
        selected_option_ids=tuple(
            value.option_id for value in result.selection_decision.members
        ),
    )


def outcome_conditioned_contextual_allocation_realization(
    *,
    allocation: ContextualPortfolioAllocationContract | None,
    wave: object,
    result: object,
) -> ContextualPortfolioAllocationRealization | None:
    """Decode exact requested-to-realized source/operator exposure evidence."""

    from agent_evolve.application.portfolio_evolution import (
        PortfolioVariationWaveRequest,
        PortfolioVariationWaveResult,
    )

    if type(wave) is not PortfolioVariationWaveRequest:
        raise TypeError("wave must be an exact PortfolioVariationWaveRequest")
    wave.__post_init__()
    if type(result) is not PortfolioVariationWaveResult:
        raise TypeError("result must be an exact PortfolioVariationWaveResult")
    result.__post_init__()
    if (
        result.receipt.request_sha256 != wave.selection_request.request_sha256
        or result.selection_decision.decision_sha256 != result.receipt.decision_sha256
    ):
        raise ValueError("portfolio result differs from its selection wave")
    return contextual_allocation_realization(
        finite_contract=wave.selection_request.finite_variation_contract,
        allocation=allocation,
        selected_option_ids=tuple(
            value.option_id for value in result.selection_decision.members
        ),
    )


@dataclass(slots=True)
class OutcomeConditionedPortfolioSelectionPolicy:
    """Portable all-action selector implementing the campaign selection port."""

    forecaster: PartitionedActionForecastPolicy
    optimization_semantics: OptimizationSemantics
    partition_policy: ActionForecastPartitionPolicyBinding
    action_semantics_factory: ActionSemanticsFactory | None = None
    consequence_calibrator: ActionConsequenceCalibrationPolicy | None = None
    metric_projector: ActionMetricProjector | None = None
    wave_action_coordinator: GlobalWaveActionAllocationCoordinator | None = None
    contextual_allocation_provider: ContextualAllocationProvider | None = None
    candidate_pool_provider: CandidatePoolProvider | None = None
    risk_aversion: float = 0.5
    diversity_weight: float = 0.05
    beam_width: int = 256
    require_healthy_forecast: bool = True
    max_memory_options_per_card: int = 16
    max_memory_assignment_trials: int = 128

    def __post_init__(self) -> None:
        if not callable(getattr(self.forecaster, "forecast_partitioned", None)):
            raise TypeError("forecaster must satisfy PartitionedActionForecastPolicy")
        if type(self.optimization_semantics) is not OptimizationSemantics:
            raise TypeError("optimization_semantics must be exact")
        self.optimization_semantics.__post_init__()
        if type(self.partition_policy) is not ActionForecastPartitionPolicyBinding:
            raise TypeError("partition_policy must be exact")
        self.partition_policy.__post_init__()
        if self.action_semantics_factory is not None and not callable(
            self.action_semantics_factory
        ):
            raise TypeError("action_semantics_factory must be callable or None")
        if self.consequence_calibrator is not None and not isinstance(
            self.consequence_calibrator,
            ActionConsequenceCalibrationPolicy,
        ):
            raise TypeError(
                "consequence_calibrator must satisfy the calibration port or be None"
            )
        if self.metric_projector is not None and not callable(
            getattr(self.metric_projector, "project", None)
        ):
            raise TypeError("metric_projector must expose project or be None")
        if self.wave_action_coordinator is not None and not isinstance(
            self.wave_action_coordinator,
            GlobalWaveActionAllocationCoordinator,
        ):
            raise TypeError(
                "wave_action_coordinator must satisfy the global allocation port"
            )
        if self.contextual_allocation_provider is not None and not callable(
            self.contextual_allocation_provider
        ):
            raise TypeError("contextual_allocation_provider must be callable or None")
        if self.candidate_pool_provider is not None and not callable(
            self.candidate_pool_provider
        ):
            raise TypeError("candidate_pool_provider must be callable or None")
        for name in ("risk_aversion", "diversity_weight"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative float")
        for name in (
            "beam_width",
            "max_memory_options_per_card",
            "max_memory_assignment_trials",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.require_healthy_forecast) is not bool:
            raise TypeError("require_healthy_forecast must be exact bool")

    def render(self, request: PortfolioSelectionRequest) -> str:
        """Render the logical request audited above its physical block calls."""

        self.__post_init__()
        if type(request) is not PortfolioSelectionRequest:
            raise TypeError("request must be an exact PortfolioSelectionRequest")
        request.__post_init__()
        return (
            "OUTCOME-CONDITIONED ALL-ACTION PORTFOLIO REQUEST\n"
            "The model forecasts every sealed action in bounded physical blocks; "
            "trusted code performs hard-feasible set allocation.\n"
            + json.dumps(
                {
                    "selection_request": request.to_record(),
                    "partition_policy": self.partition_policy.to_record(),
                    "base_policy_definition_sha256": (
                        OUTCOME_CONDITIONED_PORTFOLIO_POLICY_DEFINITION_SHA256
                    ),
                },
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        self.__post_init__()
        if type(request) is not PortfolioSelectionRequest:
            raise TypeError("request must be an exact PortfolioSelectionRequest")
        request.__post_init__()
        phase = resolve_campaign_search_phase_context(request.context)
        contextual_allocation = (
            None
            if self.contextual_allocation_provider is None
            else self.contextual_allocation_provider(request)
        )
        if contextual_allocation is not None:
            if type(contextual_allocation) is not (
                ContextualPortfolioAllocationContract
            ):
                raise TypeError(
                    "contextual allocation provider returned a foreign contract"
                )
            contextual_allocation.__post_init__()
            if (
                contextual_allocation.campaign_generation != phase.campaign_generation
                or contextual_allocation.evaluation_slots != request.portfolio_size
            ):
                raise ValueError(
                    "contextual allocation differs from the selection wave"
                )
        candidate_pool = (
            None
            if self.candidate_pool_provider is None
            else self.candidate_pool_provider(request)
        )
        forecast_contract, forecast_universe_projection = (
            _forecast_contract_projection(request, candidate_pool)
        )
        exact_arm_count_constraints = contextual_action_arm_count_constraints(
            finite_contract=forecast_contract,
            allocation=contextual_allocation,
            portfolio_size=request.portfolio_size,
        )
        action_semantics = (
            None
            if self.action_semantics_factory is None
            else self.action_semantics_factory(forecast_contract)
        )
        plan = build_target_conditioned_action_forecast_plan(
            selection_request=request,
            optimization_semantics=self.optimization_semantics,
            call_id=_forecast_call_id(request),
            forecast_contract=forecast_contract,
            action_semantics=action_semantics,
            evidence_mode=_evidence_mode(request),
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
        )
        if (
            contextual_allocation is not None
            and contextual_allocation.slice_id != plan.campaign_target.lane_id
        ):
            raise ValueError("contextual allocation differs from the target lane")
        layout = build_action_forecast_partition_layout(
            plan.request,
            self.partition_policy,
        )
        started_ns = time.perf_counter_ns()
        partitioned = await self.forecaster.forecast_partitioned(
            plan.request,
            layout,
        )
        wall_latency_ns = time.perf_counter_ns() - started_ns
        physical_telemetry = tuple(
            value.telemetry
            for value in partitioned.block_results
            if value.telemetry is not None
        )
        telemetry = _aggregate_telemetry(
            physical_telemetry,
            wall_latency_ns=wall_latency_ns,
        )
        health = assess_resolved_action_forecast_health(
            plan.request,
            partitioned.forecasts,
            member_id="outcome_conditioned_selector",
            health_policy=lenient_action_forecast_health_policy(),
        )
        consequence_calibration = None
        metric_projection = None
        exact_projections = None
        decision_forecasts = partitioned.forecasts
        if self.consequence_calibrator is not None:
            consequence_calibration = self.consequence_calibrator.calibrate(
                request=plan.request,
                forecasts=partitioned.forecasts,
                cutoff_wave_index_exclusive=phase.campaign_generation,
                metric_aliases=plan.aliases,
            )
            if type(consequence_calibration) is not ActionConsequenceCalibrationResult:
                raise TypeError("consequence calibrator returned a foreign result type")
            decision_forecasts = consequence_calibration.forecasts
        if self.metric_projector is not None:
            exact_projections = self.metric_projector.project(plan.request)
            metric_projection = apply_exact_action_metric_projections(
                request=plan.request,
                forecasts=decision_forecasts,
                projections=exact_projections,
            )
            decision_forecasts = metric_projection.forecasts
        health_passes, health_resolution = _resolve_model_authority_health(
            request_forecasts=partitioned.forecasts,
            health=health,
            exact_projections=exact_projections,
        )
        if self.require_healthy_forecast and not health_passes:
            failed = health_resolution["unresolved_failed_metric_ids"]
            raise ValueError(
                "all-action forecast failed its model-authority health gate; "
                f"unresolved_metrics={failed}"
            )

        utility_mode, target_kind = _identified_outcome_conditioned_acquisition(
            residual_cell_identified=plan.residual_cell is not None,
            terminal=phase.terminal,
        )
        preference = _preference_scores(plan, decision_forecasts)
        memory_candidates = _memory_assignment_candidates(
            request,
            preference=preference,
            max_options_per_card=self.max_memory_options_per_card,
            max_assignments=self.max_memory_assignment_trials,
        )
        global_allocation: GlobalWaveActionAllocationLaneResult | None = None
        role_slots: RoleSlots | None = None
        if self.wave_action_coordinator is not None:
            global_allocation = await self.wave_action_coordinator.allocate(
                GlobalWaveActionAllocationLane(
                    generation=phase.campaign_generation,
                    request=request,
                    plan=plan,
                    forecasts=decision_forecasts,
                    utility_mode=utility_mode,
                    target_kind=target_kind,
                    memory_assignments=tuple(
                        tuple(sorted(value.items())) for value in memory_candidates
                    ),
                    risk_aversion=self.risk_aversion,
                    diversity_weight=self.diversity_weight,
                    beam_width=self.beam_width,
                    contextual_allocation=contextual_allocation,
                )
            )
            allocation = global_allocation.allocation
            card_by_option = dict(global_allocation.memory_assignment)
            role_slots = global_allocation.role_slots
        else:
            best: tuple[ActionAllocationResult, dict[str, str]] | None = None
            failures: list[str] = []
            for card_by_option in memory_candidates:
                required = tuple(
                    sorted(
                        {
                            *request.candidate_pool_required_option_ids,
                            *card_by_option,
                        }
                    )
                )
                try:
                    allocation = allocate_target_conditioned_actions(
                        plan=plan,
                        forecasts=decision_forecasts,
                        portfolio_size=request.portfolio_size,
                        risk_aversion=self.risk_aversion,
                        diversity_weight=self.diversity_weight,
                        beam_width=self.beam_width,
                        utility_mode=utility_mode,
                        required_option_ids=required,
                        exact_arm_count_constraints=(exact_arm_count_constraints),
                        minimum_single_path_interventions=(
                            0
                            if contextual_allocation is None
                            else contextual_allocation.minimum_single_path_interventions
                        ),
                        minimum_disjoint_parent_patch_pairs=(
                            0
                            if contextual_allocation is None
                            else contextual_allocation.minimum_disjoint_parent_patch_pairs
                        ),
                    )
                except (TypeError, ValueError, RuntimeError) as error:
                    failures.append(f"{type(error).__name__}:{error}")
                    continue
                if best is None or allocation.decision.final_score.total_utility > (
                    best[0].decision.final_score.total_utility
                ):
                    best = allocation, card_by_option
            if best is None:
                detail = "; ".join(failures[:4])
                raise ValueError(
                    "no hard-feasible outcome-conditioned memory assignment"
                    + ("; " + detail if detail else "")
                )
            allocation, card_by_option = best
        selected_support, proposed_dose, evaluated_dose = _memory_assessments(
            request,
            allocation=allocation,
            card_by_option=card_by_option,
            preference=preference,
        )
        selected_required_option_ids = tuple(
            sorted(
                {
                    *request.candidate_pool_required_option_ids,
                    *card_by_option,
                }
            )
        )

        forecast_by_id = {
            value.option_id: value for value in decision_forecasts.forecasts
        }
        alias_by_target = {
            value.target_metric_id: value.forecast_metric_id for value in plan.aliases
        }
        role_audit = (
            ()
            if utility_mode != "role_factorized"
            else audit_target_conditioned_role_allocation(
                plan=plan,
                forecasts=decision_forecasts,
                decision=allocation.decision,
                eligible_option_ids=(
                    None
                    if global_allocation is None
                    else global_allocation.eligible_option_ids
                ),
                role_slots=role_slots,
                exact_arm_count_constraints=exact_arm_count_constraints,
                required_option_ids=selected_required_option_ids,
                minimum_single_path_interventions=(
                    0
                    if contextual_allocation is None
                    else contextual_allocation.minimum_single_path_interventions
                ),
                minimum_disjoint_parent_patch_pairs=(
                    0
                    if contextual_allocation is None
                    else contextual_allocation.minimum_disjoint_parent_patch_pairs
                ),
            )
        )
        role_by_option: dict[str, str] = {}
        if role_audit:
            p50 = role_audit[1]
            role_by_option = {
                value.option_id: value.role.value for value in p50.assignments
            }

        drafts: list[PortfolioMemberDraft] = []
        for member in allocation.decision.members:
            forecast = forecast_by_id[member.option_id]
            supporting = selected_support[member.option_id]
            if request.memory_dose_contract is None:
                supporting = _selected_card_keys(forecast)
            if request.require_supporting_cards and not supporting:
                raise ValueError(
                    "grounded all-action selection produced an uncited member"
                )
            metrics = {value.metric_id: value for value in forecast.metric_forecasts}
            effects = tuple(
                MetricEffectPrediction(
                    metric_id=target_metric_id,
                    direction=_direction(metrics[alias_by_target[target_metric_id]]),
                )
                for target_metric_id in request.required_metric_ids
            )
            fallback_role = (
                "directional_bootstrap"
                if utility_mode == "target_closure"
                else "terminal_exploit"
            )
            role = role_by_option.get(member.option_id, fallback_role)
            drafts.append(
                PortfolioMemberDraft(
                    option_id=member.option_id,
                    supporting_card_keys=supporting,
                    effect_predictions=effects,
                    design_rationale=(
                        f"Trusted outcome-conditioned allocation role={role}; "
                        f"utility={utility_mode}; rank={member.rank}; "
                        "numeric consequences were forecast for the sealed action "
                        "and selection was performed by hard-feasible trusted code."
                    ),
                )
            )

        policy_definition_sha256 = _bound_policy_definition(
            plan=plan,
            partition_policy=self.partition_policy,
            consequence_calibration=consequence_calibration,
            metric_projection=metric_projection,
            risk_aversion=self.risk_aversion,
            diversity_weight=self.diversity_weight,
            beam_width=self.beam_width,
            global_allocation_receipt_sha256=(
                None
                if global_allocation is None
                else global_allocation.global_receipt_sha256
            ),
            contextual_allocation_contract_sha256=(
                None
                if contextual_allocation is None
                else contextual_allocation.contract_sha256
            ),
            common_candidate_pool_decision_sha256=(
                None if candidate_pool is None else candidate_pool.decision_sha256
            ),
        )
        decision = resolve_ranked_portfolio_decision(
            request,
            tuple(drafts),
            policy_id=OUTCOME_CONDITIONED_PORTFOLIO_POLICY_ID,
            policy_version=OUTCOME_CONDITIONED_PORTFOLIO_POLICY_VERSION,
            policy_definition_sha256=policy_definition_sha256,
            memory_dose_assessment=evaluated_dose,
        )
        selected_ids = {value.option_id for value in allocation.decision.members}
        contextual_realization = contextual_allocation_realization(
            finite_contract=request.finite_variation_contract,
            allocation=contextual_allocation,
            selected_option_ids=tuple(
                value.option_id for value in allocation.decision.members
            ),
        )
        selected_forecasts = tuple(
            value.to_record()
            for value in decision_forecasts.forecasts
            if value.option_id in selected_ids
        )
        audit = PortfolioSelectionSupplementalAudit(
            audit_kind="outcome_conditioned_expert_portfolio",
            request_sha256=request.request_sha256,
            decision_sha256=decision.decision_sha256,
            payload=_object(
                {
                    "schema_version": 2,
                    "phase": phase.to_record(),
                    "evidence_mode": plan.request.evidence_mode.value,
                    "forecast_request_sha256": plan.request.request_sha256,
                    "forecast_universe_projection": forecast_universe_projection,
                    "proposal_topology": project_finite_contract_proposal_topology(
                        source_contract=request.finite_variation_contract,
                        proposal_contract=forecast_contract,
                    ),
                    "partitioned_forecast": partitioned.to_record(),
                    "physical_call_count": len(partitioned.block_results),
                    "physical_call_telemetry": [
                        _telemetry_record(value) for value in physical_telemetry
                    ],
                    "forecast_health": health.to_record(),
                    "forecast_health_authority_resolution": health_resolution,
                    "raw_forecast_receipt_sha256": (
                        partitioned.forecasts.receipt_sha256
                    ),
                    "decision_forecast_receipt_sha256": (
                        decision_forecasts.receipt_sha256
                    ),
                    "consequence_calibration": (
                        None
                        if consequence_calibration is None
                        else consequence_calibration.to_record()
                    ),
                    "metric_projection": (
                        None
                        if metric_projection is None
                        else metric_projection.to_record()
                    ),
                    "utility_mode": utility_mode,
                    "target_kind": target_kind,
                    "allocation": allocation.decision.to_record(),
                    "global_wave_allocation": _plain_global_allocation_audit(
                        global_allocation
                    ),
                    "contextual_allocation": (
                        None
                        if contextual_allocation is None
                        else contextual_allocation.to_record()
                    ),
                    "contextual_allocation_realization": (
                        None
                        if contextual_realization is None
                        else contextual_realization.to_record()
                    ),
                    "role_assignment_audits": [
                        value.to_record() for value in role_audit
                    ],
                    "selected_forecasts": list(selected_forecasts),
                    "metric_aliases": [
                        {
                            "target_metric_id": value.target_metric_id,
                            "forecast_metric_id": value.forecast_metric_id,
                        }
                        for value in plan.aliases
                    ],
                    "required_option_ids": list(selected_required_option_ids),
                    "memory_card_by_option": dict(sorted(card_by_option.items())),
                    "proposed_memory_dose_assessment": (
                        None if proposed_dose is None else proposed_dose.to_record()
                    ),
                    "evaluated_memory_dose_assessment": (
                        None if evaluated_dose is None else evaluated_dose.to_record()
                    ),
                    "policy_definition_sha256": policy_definition_sha256,
                }
            ),
        )
        return PortfolioSelectionResult(
            decision=decision,
            telemetry=telemetry,
            supplemental_audit=audit,
        )


__all__ = [
    "OUTCOME_CONDITIONED_PORTFOLIO_POLICY_DEFINITION_SHA256",
    "OUTCOME_CONDITIONED_PORTFOLIO_POLICY_ID",
    "OUTCOME_CONDITIONED_PORTFOLIO_POLICY_VERSION",
    "ActionSemanticsFactory",
    "ContextualAllocationProvider",
    "OutcomeConditionedPortfolioSelectionPolicy",
    "outcome_conditioned_contextual_allocation_realization",
    "outcome_conditioned_selected_predictions",
    "outcome_conditioned_selected_source_ids",
]
