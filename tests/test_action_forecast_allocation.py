from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent_evolve.application.action_allocation import (
    GREEDY_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256,
    GREEDY_RISK_DIVERSITY_ALLOCATOR_ID,
    GREEDY_RISK_DIVERSITY_ALLOCATOR_VERSION,
    GreedyRiskAdjustedDiversityAllocator,
)
from agent_evolve.application.action_metric_projection import (
    EXACT_METRIC_OVERLAY_POLICY_ID,
    apply_exact_action_metric_projections,
)
from agent_evolve.application.action_role_value import (
    ActionAcquisitionRole,
)
from agent_evolve.application.archive_conditioned_action_target import (
    bind_archive_conditioned_affine_action_target,
)
from agent_evolve.application.insight_memory import (
    InsightEvidenceLineage,
    InsightLifecycleState,
    InsightMemoryEntry,
    InsightOrigin,
)
from agent_evolve.application.portfolio_projection import (
    admit_portfolio_card_sources,
    bind_portfolio_experimental_view,
    portfolio_card_from_insight_entry,
)
from agent_evolve.application.target_conditioned_action_forecast import (
    allocate_target_conditioned_actions,
    audit_target_conditioned_role_allocation,
    build_target_conditioned_action_forecast_plan,
)
from agent_evolve.core.action_semantics import (
    ActionAxisSemantics,
    ActionSpaceSemantics,
)
from agent_evolve.core.optimization_semantics import (
    MetricRole,
    MetricSemantics,
    MetricSense,
    OptimizationSemantics,
    OutcomeOrderingKind,
    OutcomeOrderingSemantics,
)
from agent_evolve.domain import typed_json as typed_json_domain
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
    bind_finite_action_evidence,
)
from agent_evolve.domain.ids import (
    CandidateId,
    InsightId,
    LLMCallId,
    OperatorInvocationId,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, typed_json_sha256
from agent_evolve.ports import action_forecast as action_forecast_port
from agent_evolve.ports.action_allocation import (
    ActionAllocationRequest,
    DeterministicActionAllocator,
    ForecastPortfolioUtilityBinding,
    ForecastPortfolioUtilityInput,
    ForecastQuantile,
    validate_action_portfolio_decision,
)
from agent_evolve.ports.action_forecast import (
    ActionEvidenceCitation,
    ActionForecastDraft,
    ActionForecastEvidenceMode,
    ActionForecastPolicy,
    ActionForecastRequest,
    ActionForecastResult,
    ActionMetricForecast,
    MetricForecastScale,
    ParentMetricValue,
    resolve_action_forecasts,
    validate_resolved_action_forecasts,
)
from agent_evolve.ports.action_metric_projection import (
    ActionMetricProjector,
    ExactActionMetricProjection,
    ExactActionMetricProjectionBatch,
)
from agent_evolve.ports.agentic_generator import (
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioExperimentalArm,
    PortfolioSelectionRequest,
)
from agent_evolve.policies.reward.affine_hypervolume import (
    AffineHypervolume2DSpec,
    AffineHypervolumeArchiveUtility,
    AffineObjectiveAxis,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _frozen(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    assert type(result) is FrozenJsonObject
    return result


def _contract(
    *,
    parent_x: int = 0,
    families: tuple[str, str, str, str] = (
        "shared",
        "shared",
        "beta",
        "gamma",
    ),
) -> FiniteVariationContract:
    parent = _frozen({"x": parent_x})
    parent_sha256 = typed_json_sha256(parent)
    return FiniteVariationContract(
        catalog_id="forecast_fixture",
        catalog_version=1,
        catalog_definition_sha256=_sha("forecast-fixture-catalog-v1"),
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=option_id,
                parent_configuration_sha256=parent_sha256,
                child_configuration=_frozen({"x": child_x}),
                family=family,
                description=f"Choose fixture coordinate {child_x}.",
            )
            for (option_id, child_x), family in zip(
                (
                    ("action.a", parent_x + 1),
                    ("action.b", parent_x + 2),
                    ("action.c", parent_x + 3),
                    ("action.d", parent_x + 4),
                ),
                families,
                strict=True,
            )
        ),
    )


def _semantics() -> OptimizationSemantics:
    relation_sha = _sha("forecast-fixture-pareto-v1")
    return OptimizationSemantics(
        semantics_id="forecast_fixture_semantics",
        semantics_version=1,
        metrics=(
            MetricSemantics(
                metric_id="objective:cost",
                name="cost",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MINIMIZE,
                definition="Fixture resource cost.",
                aggregation="One deterministic scalar.",
                witness_interpretation="Lower is better.",
            ),
            MetricSemantics(
                metric_id="objective:quality",
                name="quality",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MAXIMIZE,
                definition="Fixture design quality.",
                aggregation="One deterministic scalar.",
                witness_interpretation="Higher is better.",
            ),
        ),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.PARETO,
            metric_priority=("objective:cost", "objective:quality"),
            description="Use Pareto order over cost and quality.",
            equivalence="Equal objective vectors are equivalent.",
            policy_id="forecast_fixture_pareto",
            policy_version=1,
            definition_sha256=relation_sha,
        ),
    )


def _action_semantics(
    contract: FiniteVariationContract,
    *,
    definition: str = "One fixture coordinate selects a sealed child configuration.",
) -> ActionSpaceSemantics:
    return ActionSpaceSemantics(
        semantics_id="forecast_fixture_action_space",
        semantics_version=1,
        catalog_identities=(
            (
                contract.catalog_id,
                contract.catalog_version,
                contract.catalog_definition_sha256,
            ),
        ),
        axes=(
            ActionAxisSemantics(
                axis_id="fixture_coordinate",
                configuration_paths=("$.x",),
                option_families=tuple(
                    sorted({option.family for option in contract.options})
                ),
                definition=definition,
                independence=(
                    "Each option replaces the same scalar fixture coordinate; "
                    "options are mutually exclusive alternatives."
                ),
                excluded_interpretations=(
                    "Option identifiers do not encode metric outcomes.",
                ),
            ),
        ),
    )


def _entry(
    contract: FiniteVariationContract,
    *,
    index: int,
) -> InsightMemoryEntry:
    option = contract.options[index]
    contrast_id = _sha(f"contrast-{option.option_id}")
    binding = bind_finite_action_evidence(
        contrast_id=contrast_id,
        contract=contract,
        option_id=option.option_id,
    )
    metric_predictions = (
        MetricEffectPrediction(
            "objective:cost",
            MetricEffectDirection.UNCHANGED,
        ),
        MetricEffectPrediction(
            "objective:quality",
            MetricEffectDirection.INCREASE,
        ),
    )
    draft = InsightDraft(
        claim=f"Fixture claim for {option.option_id}.",
        trigger=f"The sealed option {option.option_id} is available.",
        mechanism="Use the cited finite action and test its metric deltas.",
        affected_paths=("$.x",),
        evidence_summary="One exact source contrast supports this action.",
        confidence=0.5,
        evidence_contrast_ids=(contrast_id,),
        effect_predictions=metric_predictions,
        recommended_option_families=(option.family,),
        recommended_option_ids=(option.option_id,),
        action_template=f"Choose {option.option_id}.",
        falsification_condition="Reject if the quality direction reverses.",
    )
    lineage = InsightEvidenceLineage(
        reflection_call_id=LLMCallId(f"call_forecast_fixture_{index}"),
        source_operator_invocation_ids=(
            OperatorInvocationId(f"operator_forecast_fixture_{index}"),
        ),
        source_candidate_ids=(
            CandidateId(f"candidate_forecast_fixture_{index}"),
        ),
        available_contrast_ids=(contrast_id,),
        cited_contrast_ids=(contrast_id,),
        finite_action_bindings=(binding,),
    )
    return InsightMemoryEntry(
        reference=InsightRef(InsightId(f"insight_forecast_fixture_{index}"), 1),
        draft=draft,
        initial_score=0.0,
        lifecycle_state=InsightLifecycleState.QUARANTINED,
        origin=InsightOrigin.REFLECTION,
        evidence_lineage=lineage,
    )


def _request(
    *,
    contract: FiniteVariationContract | None = None,
    evidence_contract: FiniteVariationContract | None = None,
) -> ActionForecastRequest:
    active_contract = contract or _contract()
    source_contract = evidence_contract or active_contract
    entries = tuple(_entry(source_contract, index=index) for index in range(4))
    cards = tuple(
        sorted(
            (
                portfolio_card_from_insight_entry(
                    entry,
                    card_key=f"card.{index}",
                    prompt_payload=_frozen(
                        {
                            "claim": (
                                "A bounded coordinate intervention changed the "
                                "observed metric vector."
                            ),
                            "arm": "fixture",
                        }
                    ),
                    evidence_sha256=_sha(f"evidence-{index}"),
                    source_receipt_sha256=_sha(f"receipt-{index}"),
                )
                for index, entry in enumerate(entries)
            ),
            key=lambda card: card.card_key,
        )
    )
    registry = admit_portfolio_card_sources(entries, cards)
    receipt = bind_portfolio_experimental_view(
        arm=PortfolioExperimentalArm.MEMORY,
        cards=cards,
        finite_variation_contract=active_contract,
        source_registry=registry,
        policy_id="fixture_memory_view",
        policy_version=1,
        policy_definition_sha256=_sha("fixture-memory-view-v1"),
    )
    return ActionForecastRequest(
        call_id=LLMCallId("call_all_option_forecast_fixture"),
        operation="forecast_all_actions",
        instruction="Forecast every sealed action and required metric.",
        context=_frozen({"benchmark": "generic_fixture", "parent": "held_out"}),
        optimization_semantics=_semantics(),
        action_semantics=_action_semantics(active_contract),
        finite_variation_contract=active_contract,
        cards=cards,
        source_registry=registry,
        evidence_mode=ActionForecastEvidenceMode.GROUNDED,
        experimental_view_receipt=receipt,
        parent_metric_values=(
            ParentMetricValue("objective:cost", 100.0),
            ParentMetricValue("objective:quality", 10.0),
        ),
        metric_scales=(
            MetricForecastScale(
                "objective:cost",
                10.0,
                _sha("fixture-cost-delta-scale"),
            ),
            MetricForecastScale(
                "objective:quality",
                5.0,
                _sha("fixture-quality-delta-scale"),
            ),
        ),
        temperature=0.0,
    )


_QUALITY_QUANTILES = {
    "action.a": (0.0, 10.0, 12.0),
    "action.b": (7.0, 8.0, 9.0),
    "action.c": (6.0, 6.0, 7.0),
    "action.d": (4.0, 5.0, 6.0),
}


def _drafts(request: ActionForecastRequest) -> tuple[ActionForecastDraft, ...]:
    cards_by_option = {
        card.finite_action_evidence[0].option_id: card for card in request.cards
    }
    drafts = []
    for option in request.finite_variation_contract.options:
        card = cards_by_option[option.option_id]
        binding = card.finite_action_evidence[0]
        citation = (
            ActionEvidenceCitation(card.card_key, binding.identity_sha256),
        )
        p10, p50, p90 = _QUALITY_QUANTILES[option.option_id]
        drafts.append(
            ActionForecastDraft(
                option_id=option.option_id,
                probability_valid=1.0,
                metric_forecasts=(
                    ActionMetricForecast(
                        "objective:cost",
                        0.0,
                        0.0,
                        0.0,
                        0.8,
                        citation,
                    ),
                    ActionMetricForecast(
                        "objective:quality",
                        p10,
                        p50,
                        p90,
                        0.7,
                        citation,
                    ),
                ),
            )
        )
    return tuple(drafts)


def _resolved(request: ActionForecastRequest):
    return resolve_action_forecasts(
        request,
        _drafts(request),
        policy_id="fixture_all_option_forecaster",
        policy_version=1,
        policy_definition_sha256=_sha("fixture-all-option-forecaster-v1"),
    )


def test_all_option_resolution_binds_semantics_cards_actions_and_metrics() -> None:
    request = _request()
    batch = _resolved(request)

    validate_resolved_action_forecasts(request, batch)
    assert batch.request_sha256 == request.request_sha256
    assert batch.finite_contract_identity_sha256 == (
        request.finite_variation_contract.identity_sha256
    )
    assert [value.option_id for value in batch.forecasts] == [
        value.option_id for value in request.finite_variation_contract.options
    ]
    assert all(value.probability_valid == 1.0 for value in batch.forecasts)
    assert len(batch.receipt_sha256) == 64
    first = batch.forecasts[0]
    assert first.option_identity_sha256 == request.finite_variation_contract.options[0].identity_sha256
    assert first.child_configuration_sha256 == request.finite_variation_contract.options[0].child_configuration_sha256
    assert first.metric_forecasts[1].p50_delta == 10.0
    assert first.metric_forecasts[1].citations[0].source_option_id == "action.a"
    assert request.to_record()["source_registry_sha256"] == (
        request.source_registry.registry_sha256
    )
    assert request.to_record()["schema_version"] == 2
    assert request.to_record()["action_semantics"] == {
        "semantics_id": request.action_semantics.semantics_id,
        "semantics_version": request.action_semantics.semantics_version,
        "definition_sha256": request.action_semantics.definition_sha256,
    }
    assert batch.to_record()["schema_version"] == 2
    assert batch.action_semantics_definition_sha256 == (
        request.action_semantics.definition_sha256
    )

    class _Policy:
        async def forecast(self, value: ActionForecastRequest) -> ActionForecastResult:
            return ActionForecastResult(_resolved(value), None)

    assert isinstance(_Policy(), ActionForecastPolicy)

    altered = replace(
        batch,
        forecasts=(
            replace(batch.forecasts[0], option_identity_sha256="0" * 64),
            *batch.forecasts[1:],
        ),
    )
    with pytest.raises(ValueError, match="sealed finite option"):
        validate_resolved_action_forecasts(request, altered)

    tampered_semantics = replace(
        batch,
        action_semantics_definition_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="different request snapshot"):
        validate_resolved_action_forecasts(request, tampered_semantics)

    stripped_citations = replace(
        batch,
        forecasts=tuple(
            replace(
                forecast,
                metric_forecasts=tuple(
                    replace(metric, citations=())
                    for metric in forecast.metric_forecasts
                ),
            )
            for forecast in batch.forecasts
        ),
    )
    with pytest.raises(ValueError, match="grounded resolved forecasts require"):
        validate_resolved_action_forecasts(request, stripped_citations)


def test_exact_metric_projection_overlays_only_authorized_cells_and_binds_receipts() -> None:
    request = _request()
    source = _resolved(request)
    projections = ExactActionMetricProjectionBatch(
        forecast_request_sha256=request.request_sha256,
        finite_contract_identity_sha256=(
            request.finite_variation_contract.identity_sha256
        ),
        projections=tuple(
            ExactActionMetricProjection(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                child_configuration_sha256=option.child_configuration_sha256,
                metric_id="objective:cost",
                delta=float(index + 1),
            )
            for index, option in enumerate(
                request.finite_variation_contract.options
            )
        ),
        projector_id="fixture_exact_cost",
        projector_version=1,
        projector_definition_sha256=_sha("fixture-exact-cost-v1"),
    )

    class _Projector:
        def project(
            self,
            value: ActionForecastRequest,
        ) -> ExactActionMetricProjectionBatch:
            assert value is request
            return projections

    assert isinstance(_Projector(), ActionMetricProjector)
    result = apply_exact_action_metric_projections(
        request=request,
        forecasts=source,
        projections=_Projector().project(request),
    )
    validate_resolved_action_forecasts(request, result.forecasts)
    assert result.forecasts.policy_id == EXACT_METRIC_OVERLAY_POLICY_ID
    assert result.source_forecast_receipt_sha256 == source.receipt_sha256
    assert result.projection_receipt_sha256 == projections.receipt_sha256
    assert len(result.receipt_sha256) == 64
    for index, (before, after) in enumerate(
        zip(source.forecasts, result.forecasts.forecasts, strict=True)
    ):
        before_by_id = {value.metric_id: value for value in before.metric_forecasts}
        after_by_id = {value.metric_id: value for value in after.metric_forecasts}
        exact = after_by_id["objective:cost"]
        assert (exact.p10_delta, exact.p50_delta, exact.p90_delta) == (
            float(index + 1),
        ) * 3
        assert exact.confidence == 1.0
        assert exact.citations == before_by_id["objective:cost"].citations
        assert after_by_id["objective:quality"] == (
            before_by_id["objective:quality"]
        )

    foreign = replace(projections, forecast_request_sha256="0" * 64)
    with pytest.raises(ValueError, match="foreign forecast request"):
        apply_exact_action_metric_projections(
            request=request,
            forecasts=source,
            projections=foreign,
        )

    first = projections.projections[0]
    wrong_option = replace(
        projections,
        projections=(
            replace(first, option_identity_sha256="0" * 64),
            *projections.projections[1:],
        ),
    )
    with pytest.raises(ValueError, match="sealed option"):
        apply_exact_action_metric_projections(
            request=request,
            forecasts=source,
            projections=wrong_option,
        )


def _archive_conditioned_plan():
    forecast_request = _request()
    request = PortfolioSelectionRequest(
        call_id=LLMCallId("call_archive_target_selector_fixture"),
        operation="select_portfolio",
        instruction="Select a bounded portfolio from the sealed fixture actions.",
        context=_frozen({"benchmark": "generic_fixture"}),
        finite_variation_contract=forecast_request.finite_variation_contract,
        cards=forecast_request.cards,
        portfolio_size=2,
        required_metric_ids=("objective:cost", "objective:quality"),
        require_supporting_cards=True,
        max_output_tokens=2_048,
        temperature=0.0,
        source_registry=forecast_request.source_registry,
        experimental_view_receipt=(
            forecast_request.experimental_view_receipt
        ),
    )
    utility = AffineHypervolumeArchiveUtility(
        AffineHypervolume2DSpec(
            axes=(
                AffineObjectiveAxis("quality", "max", 30.0, 0.0),
                AffineObjectiveAxis("cost", "min", 0.0, 200.0),
            ),
            reference_provenance="Fixture prospective reference.",
        )
    )
    archive = _frozen(
        {
            "front_candidates": [
                {
                    "objectives": [
                        {"metric_id": "cost", "value_hex": (100.0).hex()},
                        {"metric_id": "quality", "value_hex": (10.0).hex()},
                    ]
                }
            ]
        }
    )
    archive_utility = utility.freeze(
        benchmark=_frozen({"fixture": "benchmark"}),
        generation=1,
        archive=archive,
    )
    snapshot = utility.require_snapshot(archive_utility)
    rebound, target = bind_archive_conditioned_affine_action_target(
        selection_request=request,
        archive_utility=archive_utility,
        affine_snapshot=snapshot,
        parent_objectives={"cost": 100.0, "quality": 10.0},
        lane_id="fixture_lane",
    )
    plan = build_target_conditioned_action_forecast_plan(
        selection_request=rebound,
        optimization_semantics=_semantics(),
        call_id=LLMCallId("call_archive_conditioned_fixture"),
        evidence_mode=ActionForecastEvidenceMode.GROUNDED,
    )

    return plan, target


def test_archive_conditioned_target_reorders_anchor_axes_into_canonical_metrics() -> None:
    plan, target = _archive_conditioned_plan()

    assert plan.campaign_target == target
    assert plan.objective_target.metric_ids == ("cost", "quality")
    assert plan.residual_cell is not None
    assert plan.residual_cell.metric_ids == ("cost", "quality")
    assert plan.residual_cell.anchor_points == ((0.5, 2.0 / 3.0),)


def test_reliability_adjusted_hv_suppresses_low_confidence_optimism() -> None:
    plan, _ = _archive_conditioned_plan()
    batch = _resolved(plan.request)
    first = batch.forecasts[0]
    metrics = tuple(
        replace(value, confidence=0.01)
        if value.metric_id == "objective:quality"
        else value
        for value in first.metric_forecasts
    )
    low_confidence = replace(
        batch,
        forecasts=(replace(first, metric_forecasts=metrics), *batch.forecasts[1:]),
    )

    raw = allocate_target_conditioned_actions(
        plan=plan,
        forecasts=low_confidence,
        portfolio_size=1,
        risk_aversion=0.0,
        diversity_weight=0.0,
        utility_mode="expected_hypervolume",
    )
    adjusted = allocate_target_conditioned_actions(
        plan=plan,
        forecasts=low_confidence,
        portfolio_size=1,
        risk_aversion=0.0,
        diversity_weight=0.0,
        utility_mode="reliability_adjusted_expected_hypervolume",
    )

    assert raw.decision.members[0].option_id == "action.a"
    assert adjusted.decision.members[0].option_id == "action.b"


def test_role_factorized_utility_assigns_two_exploits_bridge_and_probe() -> None:
    plan, _ = _archive_conditioned_plan()
    batch = _resolved(plan.request)
    eligible = tuple(sorted(value.option_id for value in batch.forecasts))
    result = allocate_target_conditioned_actions(
        plan=plan,
        forecasts=batch,
        portfolio_size=4,
        diversity_weight=0.0,
        utility_mode="role_factorized",
    )
    assert result.decision.utility_policy_id == "role_factorized_action_portfolio"

    audits = audit_target_conditioned_role_allocation(
        plan=plan,
        forecasts=batch,
        decision=result.decision,
        eligible_option_ids=eligible,
    )
    assert [value.quantile for value in audits] == [
        ForecastQuantile.P10,
        ForecastQuantile.P50,
        ForecastQuantile.P90,
    ]
    assignments = next(
        value.assignments
        for value in audits
        if value.quantile is ForecastQuantile.P50
    )
    counts = {
        role: sum(value.role is role for value in assignments)
        for role in ActionAcquisitionRole
    }
    assert counts == {
        ActionAcquisitionRole.RELIABLE_ARCHIVE_EXPLOIT: 2,
        ActionAcquisitionRole.RESIDUAL_BRIDGE: 1,
        ActionAcquisitionRole.EPISTEMIC_PROBE: 1,
    }


def test_action_semantics_are_required_and_change_the_forecast_request_hash() -> None:
    request = _request()
    changed = replace(
        request,
        action_semantics=_action_semantics(
            request.finite_variation_contract,
            definition=(
                "One fixture coordinate selects a sealed alternative child "
                "configuration."
            ),
        ),
    )

    assert changed.request_sha256 != request.request_sha256
    with pytest.raises(TypeError, match="action_semantics"):
        replace(request, action_semantics=None)

    wrong_contract = replace(
        request.finite_variation_contract,
        catalog_definition_sha256=_sha("another-catalog-definition"),
    )
    with pytest.raises(ValueError, match="absent from action semantics"):
        replace(request, finite_variation_contract=wrong_contract)


@pytest.mark.parametrize(
    ("draft_builder", "message"),
    (
        (lambda values: values[:-1], "incomplete"),
        (lambda values: (*values, values[0]), "repeats"),
        (
            lambda values: (
                replace(values[0], option_id="foreign.action"),
                *values[1:],
            ),
            "foreign",
        ),
        (
            lambda values: (
                replace(
                    values[0],
                    metric_forecasts=values[0].metric_forecasts[:-1],
                ),
                *values[1:],
            ),
            "every required metric",
        ),
    ),
)
def test_resolution_rejects_partial_foreign_duplicate_or_metric_incomplete_output(
    draft_builder,
    message: str,
) -> None:
    request = _request()
    with pytest.raises(ValueError, match=message):
        resolve_action_forecasts(
            request,
            tuple(draft_builder(_drafts(request))),
            policy_id="fixture_all_option_forecaster",
            policy_version=1,
            policy_definition_sha256=_sha("fixture-all-option-forecaster-v1"),
        )


def test_quantiles_validity_and_citations_fail_closed() -> None:
    with pytest.raises(ValueError, match="p10 <= p50 <= p90"):
        ActionMetricForecast(
            "objective:quality",
            2.0,
            1.0,
            3.0,
            0.5,
            (ActionEvidenceCitation("card.0", "a" * 64),),
        )
    request = _request()
    with pytest.raises(ValueError, match="probability_valid"):
        replace(_drafts(request)[0], probability_valid=1.1)
    drafts = _drafts(request)
    foreign_citation = ActionEvidenceCitation("card.foreign", "a" * 64)
    altered_metric = replace(drafts[0].metric_forecasts[0], citations=(foreign_citation,))
    altered = replace(
        drafts[0],
        metric_forecasts=(altered_metric, drafts[0].metric_forecasts[1]),
    )
    with pytest.raises(ValueError, match="outside the request"):
        resolve_action_forecasts(
            request,
            (altered, *drafts[1:]),
            policy_id="fixture_all_option_forecaster",
            policy_version=1,
            policy_definition_sha256=_sha("fixture-all-option-forecaster-v1"),
        )

    admitted_card = request.cards[0]
    unknown_binding = ActionEvidenceCitation(admitted_card.card_key, "f" * 64)
    altered_metric = replace(
        drafts[0].metric_forecasts[0], citations=(unknown_binding,)
    )
    altered = replace(
        drafts[0],
        metric_forecasts=(altered_metric, drafts[0].metric_forecasts[1]),
    )
    with pytest.raises(ValueError, match="absent or ambiguous"):
        resolve_action_forecasts(
            request,
            (altered, *drafts[1:]),
            policy_id="fixture_all_option_forecaster",
            policy_version=1,
            policy_definition_sha256=_sha("fixture-all-option-forecaster-v1"),
        )


def test_resolution_indexes_prompt_citations_once_per_validation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    original = action_forecast_port._resolved_citation_index
    calls = 0

    def counted(value: ActionForecastRequest):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(action_forecast_port, "_resolved_citation_index", counted)
    batch = resolve_action_forecasts(
        request,
        _drafts(request),
        policy_id="fixture_all_option_forecaster",
        policy_version=1,
        policy_definition_sha256=_sha("fixture-all-option-forecaster-v1"),
    )

    # One index resolves the untrusted drafts and a second, independently built
    # index verifies the resulting immutable batch.  The count is independent
    # of option, metric, and citation cardinality.
    assert calls == 2
    validate_resolved_action_forecasts(request, batch)
    assert calls == 3


def test_typed_json_large_tree_uses_one_recursive_validation_per_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {
        "arms": [
            {
                "arm": arm,
                "forecasts": [
                    {
                        "option_id": f"action.{index:03d}",
                        "metrics": [
                            {"metric_id": "objective:quality", "p50": 0.0},
                            {"metric_id": "constraint:cost", "p50": 1.0},
                        ],
                    }
                    for index in range(80)
                ],
            }
            for arm in ("m", "p", "n")
        ]
    }
    original = typed_json_domain.validate_typed_json_limits
    calls = 0

    def counted(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(typed_json_domain, "validate_typed_json_limits", counted)
    frozen = typed_json_domain.freeze_json(raw)
    # One entry check plus one complete frozen-root check; recursive nodes share
    # the already-validated immutable limits value.
    assert calls == 2
    assert typed_json_domain.thaw_json(frozen) == raw
    assert calls == 3


def test_request_requires_exact_registry_but_allows_source_to_target_analogy() -> None:
    request = _request()
    with pytest.raises(ValueError, match="source registry differs"):
        replace(
            request,
            cards=request.cards[:-1],
        )

    source_contract = _contract()
    target_contract = _contract(parent_x=20)
    transfer_request = _request(
        contract=target_contract,
        evidence_contract=source_contract,
    )
    transfer_batch = _resolved(transfer_request)
    assert transfer_batch.forecasts[0].option_identity_sha256 == (
        target_contract.options[0].identity_sha256
    )
    assert transfer_batch.forecasts[0].metric_forecasts[0].citations[
        0
    ].source_contract_identity_sha256 == source_contract.identity_sha256


def test_evidence_mode_makes_catalog_only_no_memory_arm_representable() -> None:
    grounded = _request()
    catalog_only = replace(
        grounded,
        cards=(),
        source_registry=None,
        evidence_mode=ActionForecastEvidenceMode.CATALOG_ONLY,
        experimental_view_receipt=None,
    )
    drafts = tuple(
        replace(
            draft,
            metric_forecasts=tuple(
                replace(metric, citations=()) for metric in draft.metric_forecasts
            ),
        )
        for draft in _drafts(grounded)
    )
    batch = resolve_action_forecasts(
        catalog_only,
        drafts,
        policy_id="fixture_catalog_only_forecaster",
        policy_version=1,
        policy_definition_sha256=_sha("fixture-catalog-only-forecaster-v1"),
    )
    assert all(
        not metric.citations
        for forecast in batch.forecasts
        for metric in forecast.metric_forecasts
    )
    with pytest.raises(ValueError, match="forbid cards"):
        replace(grounded, evidence_mode=ActionForecastEvidenceMode.CATALOG_ONLY)
    with pytest.raises(ValueError, match="require admitted evidence cards"):
        replace(grounded, cards=(), source_registry=None)
    with pytest.raises(ValueError, match="forbid evidence citations"):
        resolve_action_forecasts(
            catalog_only,
            _drafts(grounded),
            policy_id="fixture_catalog_only_forecaster",
            policy_version=1,
            policy_definition_sha256=_sha("fixture-catalog-only-forecaster-v1"),
        )


def test_grounded_forecast_can_cite_a_different_source_action_as_analogy() -> None:
    request = _request()
    drafts = _drafts(request)
    donor = request.cards[1]
    donor_binding = donor.finite_action_evidence[0]
    citation = (
        ActionEvidenceCitation(donor.card_key, donor_binding.identity_sha256),
    )
    first = replace(
        drafts[0],
        metric_forecasts=tuple(
            replace(metric, citations=citation)
            for metric in drafts[0].metric_forecasts
        ),
    )
    batch = resolve_action_forecasts(
        request,
        (first, *drafts[1:]),
        policy_id="fixture_analogical_forecaster",
        policy_version=1,
        policy_definition_sha256=_sha("fixture-analogical-forecaster-v1"),
    )
    assert batch.forecasts[0].option_id == "action.a"
    assert batch.forecasts[0].metric_forecasts[0].citations[0].source_option_id == (
        "action.b"
    )


class _SetUtility:
    def __init__(self) -> None:
        self.member_counts: list[int] = []

    def __call__(self, request: ForecastPortfolioUtilityInput) -> float:
        self.member_counts.append(len(request.members))
        attribute = {
            ForecastQuantile.P10: "p10_delta",
            ForecastQuantile.P50: "p50_delta",
            ForecastQuantile.P90: "p90_delta",
        }[request.quantile]
        total = 0.0
        for member in request.members:
            quality = next(
                value
                for value in member.metric_forecasts
                if value.metric_id == "objective:quality"
            )
            total += getattr(quality, attribute) * member.probability_valid
        return float(total)


class _ZeroSetUtility(_SetUtility):
    """Set utility whose allocation score isolates the generic diversity term."""

    def __call__(self, request: ForecastPortfolioUtilityInput) -> float:
        self.member_counts.append(len(request.members))
        return 0.0


def _allocation_request(
    *,
    eligible: tuple[str, ...],
    portfolio_size: int,
    utility: _SetUtility,
    contract: FiniteVariationContract | None = None,
) -> ActionAllocationRequest:
    request = _request(contract=contract)
    return ActionAllocationRequest(
        forecast_request=request,
        forecasts=_resolved(request),
        eligible_option_ids=eligible,
        portfolio_size=portfolio_size,
        utility=ForecastPortfolioUtilityBinding(
            utility=utility,
            policy_id="fixture_set_utility",
            policy_version=1,
            definition_sha256=_sha("fixture-set-utility-v1"),
        ),
    )


def test_greedy_allocator_is_bounded_eligible_deterministic_and_receipt_bound() -> None:
    utility = _SetUtility()
    request = _allocation_request(
        eligible=("action.b", "action.c", "action.d"),
        portfolio_size=2,
        utility=utility,
    )
    allocator = GreedyRiskAdjustedDiversityAllocator(
        risk_aversion=0.5,
        diversity_weight=4.0,
    )
    result = allocator.allocate(request)
    replay = allocator.allocate(request)

    assert isinstance(allocator, DeterministicActionAllocator)
    assert [value.option_id for value in result.decision.members] == [
        "action.b",
        "action.c",
    ]
    assert "action.a" not in {value.option_id for value in result.decision.members}
    assert result.decision.candidate_evaluations == 3 + 2
    assert result.decision.receipt_sha256 == replay.decision.receipt_sha256
    assert result.decision.utility_policy_id == "fixture_set_utility"
    assert result.decision.allocator_policy_id == GREEDY_RISK_DIVERSITY_ALLOCATOR_ID
    assert (
        result.decision.allocator_policy_version
        == GREEDY_RISK_DIVERSITY_ALLOCATOR_VERSION
        == 2
    )
    assert result.decision.allocator_definition_sha256 == (
        GREEDY_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256
    )
    assert len(result.decision.allocator_configuration_sha256) == 64
    assert utility.member_counts.count(1) > 0
    assert utility.member_counts.count(2) > 0
    validate_action_portfolio_decision(request, result.decision)

    with pytest.raises(ValueError, match="portfolio_size exceeds"):
        replace(request, eligible_option_ids=("action.b",), portfolio_size=2)
    with pytest.raises(ValueError, match="ineligible"):
        validate_action_portfolio_decision(
            request,
            replace(
                result.decision,
                members=(
                    replace(
                        result.decision.members[0],
                        option_id="action.a",
                        option_identity_sha256=(
                            request.forecasts.forecasts[0].option_identity_sha256
                        ),
                        child_configuration_sha256=(
                            request.forecasts.forecasts[0].child_configuration_sha256
                        ),
                    ),
                    result.decision.members[1],
                ),
            ),
        )


@pytest.mark.parametrize(
    ("eligible", "portfolio_size", "expected_rewards", "expected_marginals"),
    (
        (
            ("action.a", "action.b"),
            2,
            (6.0, 6.0),
            (6.0, 0.0),
        ),
        (
            ("action.a", "action.b", "action.c"),
            3,
            (3.0, 6.0, 6.0),
            (3.0, 3.0, 0.0),
        ),
        (
            ("action.a", "action.b", "action.c", "action.d"),
            3,
            (2.0, 4.0, 6.0),
            (2.0, 2.0, 2.0),
        ),
    ),
    ids=("one-attainable-family", "two-attainable-families", "many-families"),
)
def test_diversity_reward_uses_constant_attainable_final_denominator(
    eligible: tuple[str, ...],
    portfolio_size: int,
    expected_rewards: tuple[float, ...],
    expected_marginals: tuple[float, ...],
) -> None:
    request = _allocation_request(
        eligible=eligible,
        portfolio_size=portfolio_size,
        utility=_ZeroSetUtility(),
    )
    result = GreedyRiskAdjustedDiversityAllocator(
        risk_aversion=0.0,
        diversity_weight=6.0,
    ).allocate(request)

    rewards = tuple(
        member.greedy_step_score.diversity_reward
        for member in result.decision.members
    )
    marginals = tuple(
        member.marginal_total_utility for member in result.decision.members
    )
    assert rewards == pytest.approx(expected_rewards)
    assert marginals == pytest.approx(expected_marginals)
    assert all(0.0 <= value <= 6.0 for value in rewards)
    assert all(value >= 0.0 for value in marginals)


def test_third_member_cannot_lose_diversity_reward_when_two_families_are_attainable(
) -> None:
    request = _allocation_request(
        eligible=("action.a", "action.b", "action.c"),
        portfolio_size=3,
        utility=_ZeroSetUtility(),
    )
    result = GreedyRiskAdjustedDiversityAllocator(
        risk_aversion=0.0,
        diversity_weight=9.0,
    ).allocate(request)

    first, second, third = result.decision.members
    assert len({first.family, second.family}) == 2
    assert third.family in {first.family, second.family}
    assert [
        member.greedy_step_score.diversity_reward
        for member in result.decision.members
    ] == pytest.approx([4.5, 9.0, 9.0])
    assert third.marginal_total_utility == pytest.approx(0.0)


def test_diversity_ties_use_canonical_option_identity_deterministically() -> None:
    request = _allocation_request(
        eligible=("action.a", "action.b", "action.c", "action.d"),
        portfolio_size=3,
        utility=_ZeroSetUtility(),
    )
    forecasts = {
        value.option_id: value for value in request.forecasts.forecasts
    }
    canonical = sorted(
        (forecasts[option_id] for option_id in request.eligible_option_ids),
        key=lambda value: (value.option_identity_sha256, value.option_id),
    )
    expected = [canonical[0]]
    while len(expected) < request.portfolio_size:
        selected_ids = {value.option_id for value in expected}
        selected_families = {value.family for value in expected}
        remaining = [
            value for value in canonical if value.option_id not in selected_ids
        ]
        novel = [
            value for value in remaining if value.family not in selected_families
        ]
        expected.append((novel or remaining)[0])

    allocator = GreedyRiskAdjustedDiversityAllocator(
        risk_aversion=0.0,
        diversity_weight=1.0,
    )
    first = allocator.allocate(request)
    second = allocator.allocate(request)
    assert [member.option_id for member in first.decision.members] == [
        value.option_id for value in expected
    ]
    assert first.decision.receipt_sha256 == second.decision.receipt_sha256


def test_diversity_policy_depends_on_family_partition_not_domain_labels() -> None:
    reward_sequences = []
    for families in (
        ("alpha", "alpha", "beta", "beta"),
        ("north", "north", "south", "south"),
    ):
        request = _allocation_request(
            eligible=("action.a", "action.b", "action.c", "action.d"),
            portfolio_size=3,
            utility=_ZeroSetUtility(),
            contract=_contract(families=families),
        )
        result = GreedyRiskAdjustedDiversityAllocator(
            risk_aversion=0.0,
            diversity_weight=8.0,
        ).allocate(request)
        reward_sequences.append(
            tuple(
                member.greedy_step_score.diversity_reward
                for member in result.decision.members
            )
        )

    assert reward_sequences[0] == pytest.approx((4.0, 8.0, 8.0))
    assert reward_sequences[1] == pytest.approx(reward_sequences[0])


def test_risk_adjustment_changes_greedy_first_action_without_domain_logic() -> None:
    neutral_utility = _SetUtility()
    request = _allocation_request(
        eligible=("action.a", "action.b", "action.c", "action.d"),
        portfolio_size=1,
        utility=neutral_utility,
    )
    risk_neutral = GreedyRiskAdjustedDiversityAllocator(
        risk_aversion=0.0,
        diversity_weight=0.0,
    ).allocate(request)
    risk_averse = GreedyRiskAdjustedDiversityAllocator(
        risk_aversion=1.0,
        diversity_weight=0.0,
    ).allocate(request)

    assert risk_neutral.decision.members[0].option_id == "action.a"
    assert risk_averse.decision.members[0].option_id == "action.b"


def test_allocator_rejects_unidentified_or_nonfinite_utility() -> None:
    with pytest.raises(TypeError, match="callable"):
        ForecastPortfolioUtilityBinding(
            utility=None,  # type: ignore[arg-type]
            policy_id="fixture_set_utility",
            policy_version=1,
            definition_sha256="a" * 64,
        )

    class _InvalidUtility:
        def __call__(self, request: ForecastPortfolioUtilityInput) -> float:
            del request
            return float("nan")

    request = _allocation_request(
        eligible=("action.a",),
        portfolio_size=1,
        utility=_InvalidUtility(),  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="finite float"):
        GreedyRiskAdjustedDiversityAllocator().allocate(request)


def test_public_facades_export_forecast_and_allocation_contracts() -> None:
    import agent_evolve.agentic as agentic
    import agent_evolve.application as application
    import agent_evolve.ports as ports

    assert agentic.ActionForecastRequest is ActionForecastRequest
    assert agentic.ActionAllocationRequest is ActionAllocationRequest
    assert ports.ActionForecastPolicy is ActionForecastPolicy
    assert ports.DeterministicActionAllocator is DeterministicActionAllocator
    assert application.GreedyRiskAdjustedDiversityAllocator is (
        GreedyRiskAdjustedDiversityAllocator
    )
