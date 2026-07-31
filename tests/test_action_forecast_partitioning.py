from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_evolve.application.action_forecast_partitioning import (
    ActionForecastBlockHealthSubsetBinding,
    ActionForecastHealthFrameKind,
    ActionForecastHealthSubsetPolicyBinding,
    ActionForecastWaveError,
    ConcurrentActionForecastWave,
    LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_DEFINITION_SHA256,
    LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_ID,
    LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_VERSION,
    PartitionedActionForecastPolicy,
    action_forecast_block_call_id,
    assess_resolved_action_forecast_block_health,
    assess_resolved_action_forecast_block_subset_health,
    assess_resolved_action_forecast_health,
    assemble_partitioned_action_forecasts,
    build_action_forecast_block_requests,
    build_action_forecast_partition_layout,
    lenient_action_forecast_health_policy,
    lenient_action_forecast_health_v2_policy,
)
from agent_evolve.application.outcome_conditioned_portfolio_selection import (
    _evidence_mode,
    _identified_outcome_conditioned_acquisition,
    _resolve_model_authority_health,
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
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastBlockPolicy,
    ActionForecastBlockRequest,
    ActionForecastBlockResult,
    ActionForecastBlockSpec,
    ActionForecastDraft,
    ActionForecastEvidenceMode,
    ActionForecastPartitionLayout,
    ActionForecastPartitionPolicyBinding,
    ActionForecastRequest,
    ActionMetricForecast,
    MetricForecastScale,
    ParentMetricValue,
    resolve_action_forecast_block,
    resolve_action_forecasts,
)
from agent_evolve.ports.action_metric_projection import (
    ExactActionMetricProjection,
    ExactActionMetricProjectionBatch,
)


_FORECAST_POLICY_ID = "fixture_partitioned_forecast"
_FORECAST_POLICY_VERSION = 1


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


_FORECAST_POLICY_DEFINITION_SHA256 = _sha("fixture-partitioned-forecast-v1")


def test_outcome_conditioned_acquisition_respects_estimand_availability() -> None:
    assert _identified_outcome_conditioned_acquisition(
        residual_cell_identified=False,
        terminal=False,
    ) == ("target_closure", "directional_affine_bootstrap")
    assert _identified_outcome_conditioned_acquisition(
        residual_cell_identified=False,
        terminal=True,
    ) == ("target_closure", "directional_affine_bootstrap")
    assert _identified_outcome_conditioned_acquisition(
        residual_cell_identified=True,
        terminal=False,
    ) == ("role_factorized", "residual_frontier_cell")
    assert _identified_outcome_conditioned_acquisition(
        residual_cell_identified=True,
        terminal=True,
    ) == (
        "reliability_adjusted_expected_hypervolume",
        "residual_frontier_cell",
    )


def test_grounded_forecast_mode_requires_action_specific_evidence_slots() -> None:
    generic_card = SimpleNamespace(
        source_binding=object(),
        finite_action_evidence=(),
    )
    grounded_card = SimpleNamespace(
        source_binding=object(),
        finite_action_evidence=(object(),),
    )

    assert _evidence_mode(
        SimpleNamespace(
            source_registry=object(),
            experimental_view_receipt=object(),
            cards=(generic_card,),
        )
    ) is ActionForecastEvidenceMode.CATALOG_ONLY
    assert _evidence_mode(
        SimpleNamespace(
            source_registry=object(),
            experimental_view_receipt=object(),
            cards=(generic_card, grounded_card),
        )
    ) is ActionForecastEvidenceMode.GROUNDED


def test_named_v2_health_factory_restores_frozen_historical_binding() -> None:
    historical = lenient_action_forecast_health_v2_policy()
    current = lenient_action_forecast_health_policy()

    assert LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_ID == (
        "lenient_normalized_forecast_health"
    )
    assert LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_VERSION == 2
    assert LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_DEFINITION_SHA256 == (
        "14fc199feff062d231e9b7721080816ebdb4b55ba7688195a7172c6b36dc57ae"
    )
    assert historical.policy_id == LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_ID
    assert historical.policy_version == LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_VERSION
    assert historical.policy_definition_sha256 == (
        LENIENT_ACTION_FORECAST_HEALTH_V2_POLICY_DEFINITION_SHA256
    )
    assert historical.binding_sha256 == (
        "d10616a9f3bc548846fd7722b9611ada65efc5ddbc61363b9940e3afd0f103fa"
    )
    assert current.policy_version == 3
    assert current.binding_sha256 != historical.binding_sha256


def _frozen(value: dict[str, object]) -> FrozenJsonObject:
    result = freeze_json(value)
    assert type(result) is FrozenJsonObject
    return result


def _request(*, option_count: int = 8) -> ActionForecastRequest:
    parent = _frozen({"coordinate": 0})
    parent_sha256 = typed_json_sha256(parent)
    contract = FiniteVariationContract(
        catalog_id="partition_fixture",
        catalog_version=1,
        catalog_definition_sha256=_sha("partition-fixture-catalog-v1"),
        parent_configuration=parent,
        options=tuple(
            FiniteVariationOption(
                option_id=f"action.{index:02d}",
                parent_configuration_sha256=parent_sha256,
                child_configuration=_frozen({"coordinate": index + 1}),
                family="coordinate",
                description=f"Choose sealed coordinate {index + 1}.",
            )
            for index in range(option_count)
        ),
    )
    semantics = OptimizationSemantics(
        semantics_id="partition_fixture_semantics",
        semantics_version=1,
        metrics=(
            MetricSemantics(
                metric_id="objective:cost",
                name="cost",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MINIMIZE,
                definition="Deterministic fixture cost.",
                aggregation="One scalar.",
                witness_interpretation="Lower is better.",
            ),
            MetricSemantics(
                metric_id="objective:quality",
                name="quality",
                role=MetricRole.OBJECTIVE,
                sense=MetricSense.MAXIMIZE,
                definition="Deterministic fixture quality.",
                aggregation="One scalar.",
                witness_interpretation="Higher is better.",
            ),
        ),
        outcome_ordering=OutcomeOrderingSemantics(
            kind=OutcomeOrderingKind.PARETO,
            metric_priority=("objective:cost", "objective:quality"),
            description="Use Pareto order over the fixture metrics.",
            equivalence="Equal metric vectors are equivalent.",
            policy_id="partition_fixture_pareto",
            policy_version=1,
            definition_sha256=_sha("partition-fixture-pareto-v1"),
        ),
    )
    action_semantics = ActionSpaceSemantics(
        semantics_id="partition_fixture_action_space",
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
                axis_id="coordinate",
                configuration_paths=("$.coordinate",),
                option_families=("coordinate",),
                definition="One sealed scalar coordinate intervention.",
                independence="Options are mutually exclusive alternatives.",
                excluded_interpretations=(
                    "Option identifiers do not reveal metric outcomes.",
                ),
            ),
        ),
    )
    return ActionForecastRequest(
        call_id=LLMCallId("call_partition_fixture"),
        operation="forecast_all_actions",
        instruction="Forecast every sealed action and required metric.",
        context=_frozen({"benchmark": "generic_partition_fixture"}),
        optimization_semantics=semantics,
        action_semantics=action_semantics,
        finite_variation_contract=contract,
        cards=(),
        source_registry=None,
        evidence_mode=ActionForecastEvidenceMode.CATALOG_ONLY,
        experimental_view_receipt=None,
        parent_metric_values=(
            ParentMetricValue("objective:cost", 100.0),
            ParentMetricValue("objective:quality", 10.0),
        ),
        metric_scales=(
            MetricForecastScale(
                "objective:cost",
                10.0,
                _sha("partition-fixture-cost-scale"),
            ),
            MetricForecastScale(
                "objective:quality",
                5.0,
                _sha("partition-fixture-quality-scale"),
            ),
        ),
        temperature=0.0,
    )


def _partition_policy(
    *,
    max_rows: int = 3,
    max_metric_cells: int = 6,
) -> ActionForecastPartitionPolicyBinding:
    return ActionForecastPartitionPolicyBinding(
        policy_id="bounded_contiguous_rows",
        policy_version=1,
        policy_definition_sha256=_sha("bounded-contiguous-rows-v1"),
        max_rows_per_block=max_rows,
        max_metric_cells_per_block=max_metric_cells,
    )


def _subset_policy() -> ActionForecastHealthSubsetPolicyBinding:
    return ActionForecastHealthSubsetPolicyBinding(
        policy_id="fixture_unseen_rows",
        policy_version=1,
        policy_definition_sha256=_sha("fixture-unseen-rows-v1"),
    )


def _draft_for_index(
    request: ActionForecastRequest,
    index: int,
    *,
    collapsed: bool = False,
) -> ActionForecastDraft:
    option = request.finite_variation_contract.options[index]
    if collapsed:
        cost_median = 320.0
        quality_median = 160.0
        cost_lower = cost_upper = 0.0
        quality_lower = quality_upper = 0.0
        probability_valid = 0.8
    else:
        normalized = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)[index % 8]
        cost_median = normalized * 10.0
        quality_median = -normalized * 5.0
        cost_lower = cost_upper = 5.0
        quality_lower = quality_upper = 2.5
        probability_valid = (0.4, 0.6, 0.8)[index % 3]
    return ActionForecastDraft(
        option_id=option.option_id,
        probability_valid=probability_valid,
        metric_forecasts=(
            ActionMetricForecast(
                "objective:cost",
                cost_median - cost_lower,
                cost_median,
                cost_median + cost_upper,
                0.7,
                (),
            ),
            ActionMetricForecast(
                "objective:quality",
                quality_median - quality_lower,
                quality_median,
                quality_median + quality_upper,
                0.6,
                (),
            ),
        ),
    )


def _draft_with_quantized_width_and_confidence(
    request: ActionForecastRequest,
    index: int,
    *,
    cost_confidence: float,
    quality_confidence: float,
) -> ActionForecastDraft:
    """Keep varied medians and nonzero code-floor widths while setting confidence."""

    base = _draft_for_index(request, index)
    confidences = (cost_confidence, quality_confidence)
    forecasts = tuple(
        replace(
            metric,
            p10_delta=metric.p50_delta - (0.25 * scale.delta_scale),
            p90_delta=metric.p50_delta + (0.25 * scale.delta_scale),
            confidence=confidence,
        )
        for metric, scale, confidence in zip(
            base.metric_forecasts,
            request.metric_scales,
            confidences,
            strict=True,
        )
    )
    return replace(base, metric_forecasts=forecasts)


def _block_result(
    block_request: ActionForecastBlockRequest,
) -> ActionForecastBlockResult:
    spec = block_request.block
    drafts = tuple(
        _draft_for_index(block_request.request, index)
        for index in range(spec.global_row_start, spec.global_row_stop)
    )
    return ActionForecastBlockResult(
        forecasts=resolve_action_forecast_block(
            block_request,
            drafts,
            policy_id=_FORECAST_POLICY_ID,
            policy_version=_FORECAST_POLICY_VERSION,
            policy_definition_sha256=_FORECAST_POLICY_DEFINITION_SHA256,
        ),
        telemetry=None,
    )


def _block_results(
    request: ActionForecastRequest,
    layout: ActionForecastPartitionLayout,
) -> tuple[ActionForecastBlockResult, ...]:
    return tuple(
        _block_result(block_request)
        for block_request in build_action_forecast_block_requests(request, layout)
    )


def test_exactly_projected_collapsed_metric_cannot_veto_model_authority_health() -> (
    None
):
    request = _request(option_count=8)
    drafts = []
    for index in range(8):
        draft = _draft_for_index(request, index)
        cost, quality = draft.metric_forecasts
        drafts.append(
            replace(
                draft,
                metric_forecasts=(
                    replace(
                        cost,
                        p10_delta=0.0,
                        p50_delta=0.0,
                        p90_delta=0.0,
                    ),
                    quality,
                ),
            )
        )
    forecasts = resolve_action_forecasts(
        request,
        tuple(drafts),
        policy_id=_FORECAST_POLICY_ID,
        policy_version=_FORECAST_POLICY_VERSION,
        policy_definition_sha256=_FORECAST_POLICY_DEFINITION_SHA256,
    )
    health = assess_resolved_action_forecast_health(
        request,
        forecasts,
        member_id="authority_fixture",
        health_policy=lenient_action_forecast_health_policy(),
    )
    assert health.passes is False
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
                delta=float(index),
            )
            for index, option in enumerate(request.finite_variation_contract.options)
        ),
        projector_id="fixture_exact_cost",
        projector_version=1,
        projector_definition_sha256=_sha("fixture-exact-cost-v1"),
    )

    passes, audit = _resolve_model_authority_health(
        request_forecasts=forecasts,
        health=health,
        exact_projections=projections,
    )

    assert passes is True
    assert audit["raw_health_passes"] is False
    assert audit["fully_projected_metric_ids"] == ["objective:cost"]
    assert audit["model_authoritative_metric_ids"] == ["objective:quality"]
    assert audit["unresolved_failed_metric_ids"] == []

    partial = replace(projections, projections=projections.projections[:-1])
    partial_passes, partial_audit = _resolve_model_authority_health(
        request_forecasts=forecasts,
        health=health,
        exact_projections=partial,
    )
    assert partial_passes is False
    assert partial_audit["fully_projected_metric_ids"] == []
    assert partial_audit["unresolved_failed_metric_ids"] == ["objective:cost"]


def test_layout_is_deterministic_contiguous_complete_and_cell_bounded() -> None:
    request = _request(option_count=8)
    policy = _partition_policy(max_rows=3, max_metric_cells=6)

    first = build_action_forecast_partition_layout(request, policy)
    second = build_action_forecast_partition_layout(request, policy)

    assert first == second
    assert first.layout_sha256 == second.layout_sha256
    assert [
        (block.global_row_start, block.global_row_stop) for block in first.blocks
    ] == [
        (0, 3),
        (3, 6),
        (6, 8),
    ]
    assert all(block.row_count <= 3 for block in first.blocks)
    assert all(block.row_count * len(first.metric_ids) <= 6 for block in first.blocks)
    assert (
        tuple(
            identity
            for block in first.blocks
            for identity in block.option_identity_sha256s
        )
        == first.option_identity_sha256s
    )
    block_requests = build_action_forecast_block_requests(request, first)
    assert len({value.block_call_id for value in block_requests}) == 3
    assert block_requests[0].block_call_id == action_forecast_block_call_id(
        request,
        first,
        first.blocks[0],
    )
    another_arm = replace(
        request,
        call_id=LLMCallId("call_partition_fixture_another_arm"),
    )
    another_layout = build_action_forecast_partition_layout(another_arm, policy)
    assert another_layout.layout_sha256 == first.layout_sha256
    assert (
        build_action_forecast_block_requests(another_arm, another_layout)[
            0
        ].block_call_id
        != block_requests[0].block_call_id
    )

    cell_limited = build_action_forecast_partition_layout(
        request,
        _partition_policy(max_rows=8, max_metric_cells=4),
    )
    assert [value.row_count for value in cell_limited.blocks] == [2, 2, 2, 2]
    assert cell_limited.layout_sha256 != first.layout_sha256

    with pytest.raises(ValueError, match="cannot hold one complete metric row"):
        build_action_forecast_partition_layout(
            request,
            _partition_policy(max_rows=8, max_metric_cells=1),
        )


@pytest.mark.parametrize("kind", ("gap", "overlap", "identity"))
def test_layout_rejects_gap_overlap_and_global_identity_drift(kind: str) -> None:
    request = _request(option_count=8)
    layout = build_action_forecast_partition_layout(request, _partition_policy())
    second = layout.blocks[1]
    if kind == "gap":
        altered = ActionForecastBlockSpec(
            block_index=1,
            global_row_start=4,
            global_row_stop=7,
            option_identity_sha256s=layout.option_identity_sha256s[4:7],
        )
    elif kind == "overlap":
        altered = ActionForecastBlockSpec(
            block_index=1,
            global_row_start=2,
            global_row_stop=5,
            option_identity_sha256s=layout.option_identity_sha256s[2:5],
        )
    else:
        altered = replace(
            second,
            option_identity_sha256s=(
                "0" * 64,
                *second.option_identity_sha256s[1:],
            ),
        )
    with pytest.raises(ValueError, match="gap-free|option identities"):
        replace(layout, blocks=(layout.blocks[0], altered, layout.blocks[2]))


def test_reassembly_is_completion_order_invariant_and_globally_revalidated() -> None:
    request = _request(option_count=8)
    layout = build_action_forecast_partition_layout(request, _partition_policy())
    results = _block_results(request, layout)

    canonical = assemble_partitioned_action_forecasts(request, layout, results)
    reversed_completion = assemble_partitioned_action_forecasts(
        request,
        layout,
        tuple(reversed(results)),
    )

    assert canonical == reversed_completion
    assert canonical.receipt_sha256 == reversed_completion.receipt_sha256
    assert canonical.forecasts.receipt_sha256 == (
        reversed_completion.forecasts.receipt_sha256
    )
    assert [value.option_id for value in canonical.forecasts.forecasts] == [
        value.option_id for value in request.finite_variation_contract.options
    ]
    assert [value.forecasts.block_index for value in canonical.block_results] == [
        0,
        1,
        2,
    ]

    with pytest.raises(ValueError, match="completely cover"):
        assemble_partitioned_action_forecasts(request, layout, results[:-1])
    with pytest.raises(ValueError, match="repeat"):
        assemble_partitioned_action_forecasts(
            request,
            layout,
            (results[0], results[0], results[2]),
        )


def test_reassembly_rejects_identity_metric_and_policy_drift() -> None:
    request = _request(option_count=8)
    layout = build_action_forecast_partition_layout(request, _partition_policy())
    results = _block_results(request, layout)

    first_block = results[0].forecasts
    identity_drift = replace(
        first_block,
        forecasts=(
            replace(first_block.forecasts[0], option_identity_sha256="0" * 64),
            *first_block.forecasts[1:],
        ),
    )
    with pytest.raises(ValueError, match="sealed finite option"):
        assemble_partitioned_action_forecasts(
            request,
            layout,
            (replace(results[0], forecasts=identity_drift), *results[1:]),
        )

    first_forecast = first_block.forecasts[0]
    metric_drift = replace(
        first_block,
        forecasts=(
            replace(
                first_forecast,
                metric_forecasts=(
                    replace(
                        first_forecast.metric_forecasts[0],
                        metric_id="objective:alien",
                    ),
                    first_forecast.metric_forecasts[1],
                ),
            ),
            *first_block.forecasts[1:],
        ),
    )
    with pytest.raises(ValueError, match="metric coverage"):
        assemble_partitioned_action_forecasts(
            request,
            layout,
            (replace(results[0], forecasts=metric_drift), *results[1:]),
        )

    policy_drift = replace(results[1].forecasts, policy_id="another_forecaster")
    with pytest.raises(ValueError, match="policy drift"):
        assemble_partitioned_action_forecasts(
            request,
            layout,
            (results[0], replace(results[1], forecasts=policy_drift), results[2]),
        )


class _ConcurrentFixtureBlockPolicy:
    def __init__(self, *, failing_index: int | None = None) -> None:
        self.failing_index = failing_index
        self.active = 0
        self.max_active = 0
        self.completed: list[int] = []

    async def forecast_block(
        self,
        request: ActionForecastBlockRequest,
    ) -> ActionForecastBlockResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.005 * (3 - request.block.block_index))
            if request.block.block_index == self.failing_index:
                raise RuntimeError("prospective fixture failure")
            return _block_result(request)
        finally:
            self.completed.append(request.block.block_index)
            self.active -= 1


def test_queue_agnostic_wave_is_bounded_and_failure_isolated() -> None:
    request = _request(option_count=8)
    layout = build_action_forecast_partition_layout(request, _partition_policy())
    successful_policy = _ConcurrentFixtureBlockPolicy()
    wave = ConcurrentActionForecastWave(successful_policy, max_concurrency=2)

    result = asyncio.run(wave.forecast_partitioned(request, layout))

    assert isinstance(successful_policy, ActionForecastBlockPolicy)
    assert isinstance(wave, PartitionedActionForecastPolicy)
    assert successful_policy.max_active == 2
    assert sorted(successful_policy.completed) == [0, 1, 2]
    assert len(result.forecasts.forecasts) == 8

    failing_policy = _ConcurrentFixtureBlockPolicy(failing_index=1)
    failing_wave = ConcurrentActionForecastWave(failing_policy, max_concurrency=2)
    with pytest.raises(ActionForecastWaveError) as captured:
        asyncio.run(failing_wave.forecast_partitioned(request, layout))
    assert sorted(failing_policy.completed) == [0, 1, 2]
    assert [
        value.forecasts.block_index for value in captured.value.successful_results
    ] == [
        0,
        2,
    ]
    assert [value.block_index for value in captured.value.failures] == [1]


def test_normalized_health_receipt_detects_obvious_wire_collapse() -> None:
    request = _request(option_count=8)
    healthy = resolve_action_forecasts(
        request,
        tuple(_draft_for_index(request, index) for index in range(8)),
        policy_id=_FORECAST_POLICY_ID,
        policy_version=_FORECAST_POLICY_VERSION,
        policy_definition_sha256=_FORECAST_POLICY_DEFINITION_SHA256,
    )
    policy = lenient_action_forecast_health_policy()
    assert policy.policy_id == "lenient_normalized_forecast_health"
    assert policy.policy_version == 3
    assert policy.extreme_abs_normalized_median == 32.0
    assert policy.policy_definition_sha256 == _sha(
        "agent-evolve:lenient-normalized-forecast-health:v3;"
        "minimum_rows=8;extreme_abs_normalized_median=32;"
        "collapse_share_threshold=0.95;minimum_distinct_signatures=2;"
        "unit_confidence_share_below_collapse_threshold=true"
    )

    first = assess_resolved_action_forecast_health(
        request,
        healthy,
        member_id="memory",
        health_policy=policy,
    )
    second = assess_resolved_action_forecast_health(
        request,
        healthy,
        member_id="memory",
        health_policy=policy,
    )

    assert first == second
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.frame_kind is ActionForecastHealthFrameKind.COMPLETE
    assert first.frame_receipt_sha256 == healthy.receipt_sha256
    assert first.layout_sha256 is None
    assert first.block_request_sha256 is None
    assert first.block_spec_sha256 is None
    assert first.block_index is None
    assert first.global_row_start == 0
    assert first.global_row_stop == 8
    assert first.to_record()["schema_version"] == 4
    assert "batch_receipt_sha256" not in first.to_record()
    assert first.passes is True
    assert first.distinct_row_signature_count == 8
    assert first.distinct_probability_valid_count == 3
    assert first.constant_confidence_metric_ids == (
        "objective:cost",
        "objective:quality",
    )
    assert all(value.extreme_median_share == 0.0 for value in first.metric_assessments)
    assert all(value.unit_confidence_share == 0.0 for value in first.metric_assessments)
    assert all(
        value.to_record()["schema_version"] == 2 for value in first.metric_assessments
    )

    collapsed = resolve_action_forecasts(
        request,
        tuple(_draft_for_index(request, index, collapsed=True) for index in range(8)),
        policy_id=_FORECAST_POLICY_ID,
        policy_version=_FORECAST_POLICY_VERSION,
        policy_definition_sha256=_FORECAST_POLICY_DEFINITION_SHA256,
    )
    assessment = assess_resolved_action_forecast_health(
        request,
        collapsed,
        member_id="memory",
        health_policy=policy,
    )

    assert assessment.passes is False
    assert assessment.distinct_row_signature_count == 1
    assert assessment.distinct_probability_valid_count == 1
    for metric in assessment.metric_assessments:
        assert metric.extreme_median_share == 1.0
        assert metric.largest_median_bucket_share == 1.0
        assert metric.zero_width_share == 1.0
        assert metric.unit_confidence_share == 0.0
        assert metric.distinct_cell_signature_count == 1
        assert metric.max_abs_normalized_median == 32.0
        assert metric.passes is False


def test_unit_confidence_concentration_fails_with_nonzero_quantized_widths() -> None:
    request = _request(option_count=20)
    concentrated_drafts = tuple(
        _draft_with_quantized_width_and_confidence(
            request,
            index,
            cost_confidence=1.0 if index < 19 else 0.5,
            quality_confidence=1.0 if index < 19 else 0.4,
        )
        for index in range(20)
    )
    policy = lenient_action_forecast_health_policy()
    batch = resolve_action_forecasts(
        request,
        concentrated_drafts,
        policy_id=_FORECAST_POLICY_ID,
        policy_version=_FORECAST_POLICY_VERSION,
        policy_definition_sha256=_FORECAST_POLICY_DEFINITION_SHA256,
    )
    layout = build_action_forecast_partition_layout(
        request,
        _partition_policy(max_rows=20, max_metric_cells=40),
    )
    block_request = build_action_forecast_block_requests(request, layout)[0]
    block = resolve_action_forecast_block(
        block_request,
        concentrated_drafts,
        policy_id=_FORECAST_POLICY_ID,
        policy_version=_FORECAST_POLICY_VERSION,
        policy_definition_sha256=_FORECAST_POLICY_DEFINITION_SHA256,
    )

    complete_health = assess_resolved_action_forecast_health(
        request,
        batch,
        member_id="memory",
        health_policy=policy,
    )
    block_health = assess_resolved_action_forecast_block_health(
        block_request,
        block,
        member_id="memory",
        health_policy=policy,
    )
    subset_health = assess_resolved_action_forecast_block_subset_health(
        block_request,
        block,
        member_id="memory",
        health_policy=policy,
        subset_policy=_subset_policy(),
        included_global_row_indices=tuple(range(20)),
    )

    assert complete_health.passes is False
    assert block_health.metric_assessments == complete_health.metric_assessments
    assert subset_health.metric_assessments == complete_health.metric_assessments
    assert block_health.passes is complete_health.passes
    assert subset_health.passes is complete_health.passes
    for metric in complete_health.metric_assessments:
        assert metric.unit_confidence_share == 0.95
        assert metric.zero_width_share == 0.0
        assert metric.extreme_median_share == 0.0
        assert metric.largest_median_bucket_share < 0.95
        assert metric.distinct_cell_signature_count >= 2
        assert metric.passes is False
        assert metric.to_record()["unit_confidence_share_hex"] == (0.95).hex()

    varied_drafts = tuple(
        _draft_with_quantized_width_and_confidence(
            request,
            index,
            cost_confidence=(0.2, 0.4, 0.6, 0.8)[index % 4],
            quality_confidence=(0.3, 0.5, 0.7, 0.9)[index % 4],
        )
        for index in range(20)
    )
    varied = resolve_action_forecasts(
        request,
        varied_drafts,
        policy_id=_FORECAST_POLICY_ID,
        policy_version=_FORECAST_POLICY_VERSION,
        policy_definition_sha256=_FORECAST_POLICY_DEFINITION_SHA256,
    )
    varied_health = assess_resolved_action_forecast_health(
        request,
        varied,
        member_id="memory",
        health_policy=policy,
    )

    assert varied_health.passes is True
    assert all(
        metric.unit_confidence_share == 0.0
        and metric.zero_width_share == 0.0
        and metric.passes is True
        for metric in varied_health.metric_assessments
    )


def test_block_health_uses_identical_gates_but_distinct_frame_binding() -> None:
    request = _request(option_count=8)
    partition_policy = _partition_policy(max_rows=8, max_metric_cells=16)
    layout = build_action_forecast_partition_layout(request, partition_policy)
    assert layout.block_count == 1
    block_request = build_action_forecast_block_requests(request, layout)[0]
    block = _block_result(block_request).forecasts
    batch = resolve_action_forecasts(
        request,
        tuple(_draft_for_index(request, index) for index in range(8)),
        policy_id=_FORECAST_POLICY_ID,
        policy_version=_FORECAST_POLICY_VERSION,
        policy_definition_sha256=_FORECAST_POLICY_DEFINITION_SHA256,
    )
    policy = lenient_action_forecast_health_policy()

    complete_health = assess_resolved_action_forecast_health(
        request,
        batch,
        member_id="memory",
        health_policy=policy,
    )
    block_health = assess_resolved_action_forecast_block_health(
        block_request,
        block,
        member_id="memory",
        health_policy=policy,
    )

    assert block_health.metric_assessments == complete_health.metric_assessments
    assert block_health.distinct_row_signature_count == (
        complete_health.distinct_row_signature_count
    )
    assert block_health.distinct_probability_valid_count == (
        complete_health.distinct_probability_valid_count
    )
    assert block_health.constant_confidence_metric_ids == (
        complete_health.constant_confidence_metric_ids
    )
    assert block_health.threshold_applied is complete_health.threshold_applied
    assert block_health.passes is complete_health.passes

    assert block_health.frame_kind is ActionForecastHealthFrameKind.PARTITION_BLOCK
    assert block_health.frame_receipt_sha256 == block.receipt_sha256
    assert block_health.request_sha256 == request.request_sha256
    assert block_health.layout_sha256 == layout.layout_sha256
    assert block_health.block_request_sha256 == block_request.block_request_sha256
    assert block_health.block_spec_sha256 == block_request.block.block_spec_sha256
    assert block_health.block_index == 0
    assert block_health.global_row_start == 0
    assert block_health.global_row_stop == 8
    assert block_health.receipt_sha256 != complete_health.receipt_sha256
    record = block_health.to_record()
    assert record["frame_kind"] == "partition_block"
    assert record["frame_receipt_sha256"] == block.receipt_sha256
    assert "batch_receipt_sha256" not in record

    subset_health = assess_resolved_action_forecast_block_subset_health(
        block_request,
        block,
        member_id="memory",
        health_policy=policy,
        subset_policy=_subset_policy(),
        included_global_row_indices=tuple(range(8)),
    )
    assert subset_health.metric_assessments == block_health.metric_assessments
    assert subset_health.distinct_row_signature_count == (
        block_health.distinct_row_signature_count
    )
    assert subset_health.distinct_probability_valid_count == (
        block_health.distinct_probability_valid_count
    )
    assert subset_health.constant_confidence_metric_ids == (
        block_health.constant_confidence_metric_ids
    )
    assert subset_health.threshold_applied is block_health.threshold_applied
    assert subset_health.passes is block_health.passes
    assert (
        subset_health.frame_kind is ActionForecastHealthFrameKind.PARTITION_BLOCK_SUBSET
    )
    assert subset_health.receipt_sha256 != block_health.receipt_sha256
    assert type(subset_health.subset_binding) is ActionForecastBlockHealthSubsetBinding
    assert subset_health.subset_binding.included_global_row_indices == tuple(range(8))
    assert subset_health.subset_binding.parent_block_receipt_sha256 == (
        block.receipt_sha256
    )


def test_block_health_rejects_tampering_and_mismatched_block_request() -> None:
    request = _request(option_count=8)
    layout = build_action_forecast_partition_layout(request, _partition_policy())
    block_requests = build_action_forecast_block_requests(request, layout)
    first_block = _block_result(block_requests[0]).forecasts
    second_block = _block_result(block_requests[1]).forecasts
    policy = lenient_action_forecast_health_policy()

    with pytest.raises(ValueError, match="different block request"):
        assess_resolved_action_forecast_block_health(
            block_requests[0],
            replace(first_block, layout_sha256="0" * 64),
            member_id="memory",
            health_policy=policy,
        )
    with pytest.raises(ValueError, match="different block request"):
        assess_resolved_action_forecast_block_health(
            block_requests[0],
            second_block,
            member_id="memory",
            health_policy=policy,
        )

    health = assess_resolved_action_forecast_block_health(
        block_requests[0],
        first_block,
        member_id="memory",
        health_policy=policy,
    )
    with pytest.raises(ValueError, match="row counts differ"):
        replace(health, global_row_stop=health.global_row_stop + 1)
    with pytest.raises(ValueError, match="forbid block identities"):
        replace(health, frame_kind=ActionForecastHealthFrameKind.COMPLETE)


def test_block_subset_health_binds_exact_ordered_rows_and_option_identities() -> None:
    request = _request(option_count=8)
    layout = build_action_forecast_partition_layout(request, _partition_policy())
    block_request = build_action_forecast_block_requests(request, layout)[0]
    block = _block_result(block_request).forecasts
    health_policy = lenient_action_forecast_health_policy()
    subset_policy = _subset_policy()

    health = assess_resolved_action_forecast_block_subset_health(
        block_request,
        block,
        member_id="memory",
        health_policy=health_policy,
        subset_policy=subset_policy,
        included_global_row_indices=(0, 2),
    )

    assert health.frame_kind is ActionForecastHealthFrameKind.PARTITION_BLOCK_SUBSET
    assert type(health.subset_binding) is ActionForecastBlockHealthSubsetBinding
    binding = health.subset_binding
    assert health.frame_receipt_sha256 == binding.binding_sha256
    assert binding.subset_policy == subset_policy
    assert binding.request_sha256 == request.request_sha256
    assert binding.layout_sha256 == layout.layout_sha256
    assert binding.block_request_sha256 == block_request.block_request_sha256
    assert binding.block_spec_sha256 == block_request.block.block_spec_sha256
    assert binding.parent_block_receipt_sha256 == block.receipt_sha256
    assert binding.block_index == 0
    assert binding.global_row_start == 0
    assert binding.global_row_stop == 3
    assert binding.included_global_row_indices == (0, 2)
    assert binding.included_option_identity_sha256s == (
        request.finite_variation_contract.options[0].identity_sha256,
        request.finite_variation_contract.options[2].identity_sha256,
    )
    assert health.metric_assessments[0].row_count == 2
    assert health.to_record()["subset_binding"] == binding.to_record()

    for invalid_rows, message in (
        ((), "non-empty"),
        ((2, 0), "block order"),
        ((0, 0), "block order"),
        ((0, 3), "outside the parent block"),
    ):
        with pytest.raises(ValueError, match=message):
            assess_resolved_action_forecast_block_subset_health(
                block_request,
                block,
                member_id="memory",
                health_policy=health_policy,
                subset_policy=subset_policy,
                included_global_row_indices=invalid_rows,
            )

    altered_binding = replace(
        binding,
        included_option_identity_sha256s=(
            "0" * 64,
            binding.included_option_identity_sha256s[1],
        ),
    )
    with pytest.raises(ValueError, match="authenticated binding"):
        replace(health, subset_binding=altered_binding)
    with pytest.raises(ValueError, match="forbids subset bindings"):
        replace(health, frame_kind=ActionForecastHealthFrameKind.PARTITION_BLOCK)
    with pytest.raises(ValueError, match="forbid block identities"):
        replace(health, frame_kind=ActionForecastHealthFrameKind.COMPLETE)


def test_unseen_subset_detects_collapse_hidden_by_three_distinct_anchors() -> None:
    request = _request(option_count=20)
    anchor_indices = frozenset((0, 9, 19))
    drafts = tuple(
        _draft_for_index(
            request,
            index,
            collapsed=index not in anchor_indices,
        )
        for index in range(20)
    )
    layout = build_action_forecast_partition_layout(
        request,
        _partition_policy(max_rows=20, max_metric_cells=40),
    )
    assert layout.block_count == 1
    block_request = build_action_forecast_block_requests(request, layout)[0]
    block = resolve_action_forecast_block(
        block_request,
        drafts,
        policy_id=_FORECAST_POLICY_ID,
        policy_version=_FORECAST_POLICY_VERSION,
        policy_definition_sha256=_FORECAST_POLICY_DEFINITION_SHA256,
    )
    health_policy = lenient_action_forecast_health_policy()
    full_health = assess_resolved_action_forecast_block_health(
        block_request,
        block,
        member_id="memory",
        health_policy=health_policy,
    )
    unseen_indices = tuple(index for index in range(20) if index not in anchor_indices)
    unseen_health = assess_resolved_action_forecast_block_subset_health(
        block_request,
        block,
        member_id="memory",
        health_policy=health_policy,
        subset_policy=_subset_policy(),
        included_global_row_indices=unseen_indices,
    )

    assert len(unseen_indices) == 17
    assert full_health.passes is True
    assert full_health.distinct_row_signature_count == 4
    for metric in full_health.metric_assessments:
        assert metric.extreme_median_share == pytest.approx(0.85)
        assert metric.largest_median_bucket_share == pytest.approx(0.85)
        assert metric.zero_width_share == pytest.approx(0.85)
        assert metric.passes is True

    assert unseen_health.passes is False
    assert unseen_health.distinct_row_signature_count == 1
    assert unseen_health.subset_binding is not None
    assert unseen_health.subset_binding.included_global_row_indices == unseen_indices
    assert unseen_health.subset_binding.parent_block_receipt_sha256 == (
        block.receipt_sha256
    )
    for metric in unseen_health.metric_assessments:
        assert metric.row_count == 17
        assert metric.extreme_median_share == 1.0
        assert metric.largest_median_bucket_share == 1.0
        assert metric.zero_width_share == 1.0
        assert metric.distinct_cell_signature_count == 1
        assert metric.max_abs_normalized_median == 32.0
        assert metric.passes is False
