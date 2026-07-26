from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent_evolve.application.action_evidence_consistency import (
    PRESENTED_ACTION_EVIDENCE_CONSISTENCY_POLICY_DEFINITION_SHA256,
    assess_presented_action_evidence_block_consistency,
    assess_presented_action_evidence_subset_consistency,
    bind_presented_action_evidence_subset,
    descriptive_presented_action_evidence_consistency_policy,
)
from agent_evolve.application.action_forecast_partitioning import (
    build_action_forecast_block_requests,
    build_action_forecast_partition_layout,
)
from agent_evolve.application.portfolio_projection import (
    bind_portfolio_experimental_view,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastBlockRequest,
    ActionForecastPartitionPolicyBinding,
    ActionForecastRequest,
    ResolvedActionForecastBlock,
    resolve_action_forecast_block,
)
from agent_evolve.ports.presented_action_evidence import (
    PRESENTED_ACTION_EVIDENCE_CONSISTENCY_SCOPE,
    PresentedActionEvidenceCell,
    PresentedActionEvidenceConsistencyFrameKind,
    PresentedActionEvidenceConsistencyPolicyBinding,
    PresentedActionEvidenceProvenanceKind,
    PresentedActionEvidenceSubsetPolicyBinding,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioCardViewTransform,
    PortfolioExperimentalArm,
    derive_portfolio_card_view,
)
from tests.test_action_forecast_allocation import _contract, _drafts, _request


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _resolved_frame(
    request: ActionForecastRequest,
    *,
    block_index: int = 0,
) -> tuple[ActionForecastBlockRequest, ResolvedActionForecastBlock]:
    partition_policy = ActionForecastPartitionPolicyBinding(
        policy_id="presented_evidence_fixture_partition",
        policy_version=1,
        policy_definition_sha256=_sha("presented-evidence-fixture-partition-v1"),
        max_rows_per_block=2,
        max_metric_cells_per_block=4,
    )
    layout = build_action_forecast_partition_layout(request, partition_policy)
    block_request = build_action_forecast_block_requests(request, layout)[block_index]
    spec = block_request.block
    block = resolve_action_forecast_block(
        block_request,
        _drafts(request)[spec.global_row_start : spec.global_row_stop],
        policy_id="presented_evidence_fixture_forecast",
        policy_version=1,
        policy_definition_sha256=_sha("presented-evidence-fixture-forecast-v1"),
    )
    return block_request, block


def _frame(
    *,
    parent_x: int = 0,
    block_index: int = 0,
) -> tuple[
    ActionForecastRequest,
    ActionForecastBlockRequest,
    ResolvedActionForecastBlock,
]:
    request = _request(contract=_contract(parent_x=parent_x))
    block_request, block = _resolved_frame(request, block_index=block_index)
    return request, block_request, block


def _placebo_request() -> ActionForecastRequest:
    source_request = _request()
    source_cards = source_request.cards
    donors = (*source_cards[1:], source_cards[0])
    transforms = tuple(
        sorted(
            (
                PortfolioCardViewTransform.EVIDENCE_PERMUTATION,
                PortfolioCardViewTransform.PROMPT_PERMUTATION,
                PortfolioCardViewTransform.SCORE_PERMUTATION,
            ),
            key=lambda value: value.value,
        )
    )
    placebo_cards = tuple(
        derive_portfolio_card_view(
            source,
            prompt_payload=donor.prompt_payload,
            evidence_sha256=donor.evidence_sha256,
            score_components=donor.score_components,
            assigned_score=donor.assigned_score,
            transforms=transforms,
            policy_id="fixture_placebo_card_view",
            policy_version=1,
            policy_definition_sha256=_sha("fixture-placebo-card-view-v1"),
            prompt_source_card=donor,
            evidence_source_card=donor,
            score_source_card=donor,
        )
        for source, donor in zip(source_cards, donors, strict=True)
    )
    assert source_request.source_registry is not None
    receipt = bind_portfolio_experimental_view(
        arm=PortfolioExperimentalArm.PERMUTED_PLACEBO,
        cards=placebo_cards,
        finite_variation_contract=source_request.finite_variation_contract,
        source_registry=source_request.source_registry,
        policy_id="fixture_placebo_population",
        policy_version=1,
        policy_definition_sha256=_sha("fixture-placebo-population-v1"),
    )
    return replace(
        source_request,
        cards=placebo_cards,
        experimental_view_receipt=receipt,
    )


def _cell(
    request: ActionForecastRequest,
    block: ResolvedActionForecastBlock,
    *,
    target_local_index: int,
    source_card_index: int,
    metric_id: str,
    presented_delta: float,
    provenance_kind: PresentedActionEvidenceProvenanceKind = (
        PresentedActionEvidenceProvenanceKind.CARD_SOURCE_RECEIPT
    ),
) -> PresentedActionEvidenceCell:
    card = request.cards[source_card_index]
    binding = card.finite_action_evidence[0]
    if provenance_kind is PresentedActionEvidenceProvenanceKind.CARD_SOURCE_RECEIPT:
        assert card.source_binding is not None
        provenance_sha256 = card.source_binding.source_receipt_sha256
    elif provenance_kind is PresentedActionEvidenceProvenanceKind.CARD_VIEW_RECEIPT:
        assert card.derived_view_receipt is not None
        provenance_sha256 = card.derived_view_receipt.receipt_sha256
    else:
        provenance_sha256 = request.card_snapshot_sha256
    return PresentedActionEvidenceCell(
        option_identity_sha256=(
            block.forecasts[target_local_index].option_identity_sha256
        ),
        metric_id=metric_id,
        presented_delta=presented_delta,
        card_key=card.card_key,
        action_evidence_binding_identity_sha256=binding.identity_sha256,
        provenance_kind=provenance_kind,
        provenance_sha256=provenance_sha256,
    )


def _canonical_cells(
    *values: PresentedActionEvidenceCell,
) -> tuple[PresentedActionEvidenceCell, ...]:
    return tuple(sorted(values, key=lambda value: value.sort_key))


def test_default_receipt_is_exact_descriptive_presented_evidence_not_truth() -> None:
    request, block_request, block = _frame()
    quality = _cell(
        request,
        block,
        target_local_index=0,
        source_card_index=0,
        metric_id="objective:quality",
        presented_delta=11.0,
    )
    cost = _cell(
        request,
        block,
        target_local_index=1,
        source_card_index=1,
        metric_id="objective:cost",
        presented_delta=5.0,
        provenance_kind=(
            PresentedActionEvidenceProvenanceKind.REQUEST_CARD_SNAPSHOT
        ),
    )
    cells = _canonical_cells(quality, cost)

    first = assess_presented_action_evidence_block_consistency(
        block_request,
        block,
        cells,
    )
    second = assess_presented_action_evidence_block_consistency(
        block_request,
        block,
        cells,
    )

    assert first == second
    assert first.frame_kind is PresentedActionEvidenceConsistencyFrameKind.BLOCK
    assert first.request_sha256 == request.request_sha256
    assert first.request_card_snapshot_sha256 == request.card_snapshot_sha256
    assert request.experimental_view_receipt is not None
    assert first.experimental_view_receipt_sha256 == (
        request.experimental_view_receipt.receipt_sha256
    )
    assert first.forecast_block_receipt_sha256 == block.receipt_sha256
    assert first.subset_binding is None
    assert first.decision_applied is False
    assert first.passes is None
    assert first.policy == descriptive_presented_action_evidence_consistency_policy()
    assert first.policy.maximum_normalized_absolute_error is None
    record = first.to_record()
    assert record["scientific_scope"] == PRESENTED_ACTION_EVIDENCE_CONSISTENCY_SCOPE
    assert record["decision_applied"] is False
    assert record["passes"] is None
    assert len(record["cell_set_sha256"]) == 64

    by_metric = {value.metric_id: value for value in first.cell_assessments}
    quality_assessment = by_metric["objective:quality"]
    assert quality_assessment.p10_delta == 0.0
    assert quality_assessment.p50_delta == 10.0
    assert quality_assessment.p90_delta == 12.0
    assert quality_assessment.presented_delta == 11.0
    assert quality_assessment.normalized_absolute_error == 0.2
    assert quality_assessment.direction_agreement is True
    assert quality_assessment.interval_coverage is True
    assert quality_assessment.forecast_cites_presented_binding is True
    assert quality_assessment.source_option_id == "action.a"
    assert quality_assessment.action_evidence_binding_identity_sha256 == (
        request.cards[0].finite_action_evidence[0].identity_sha256
    )

    cost_assessment = by_metric["objective:cost"]
    assert cost_assessment.p10_delta == 0.0
    assert cost_assessment.p50_delta == 0.0
    assert cost_assessment.p90_delta == 0.0
    assert cost_assessment.normalized_absolute_error == 0.5
    assert cost_assessment.direction_agreement is False
    assert cost_assessment.interval_coverage is False


def test_placebo_view_is_scored_against_its_exact_presentation_not_truth() -> None:
    request = _placebo_request()
    block_request, block = _resolved_frame(request)
    assert request.experimental_view_receipt is not None
    assert request.experimental_view_receipt.arm is (
        PortfolioExperimentalArm.PERMUTED_PLACEBO
    )
    cell = _cell(
        request,
        block,
        target_local_index=0,
        source_card_index=0,
        metric_id="objective:quality",
        presented_delta=11.0,
        provenance_kind=PresentedActionEvidenceProvenanceKind.CARD_VIEW_RECEIPT,
    )
    assessment = assess_presented_action_evidence_block_consistency(
        block_request,
        block,
        (cell,),
    )

    card = request.cards[0]
    assert card.derived_view_receipt is not None
    result = assessment.cell_assessments[0]
    assert assessment.experimental_view_receipt_sha256 == (
        request.experimental_view_receipt.receipt_sha256
    )
    assert result.card_view_receipt_sha256 == card.derived_view_receipt.receipt_sha256
    assert result.provenance_sha256 == card.derived_view_receipt.receipt_sha256
    assert assessment.to_record()["scientific_scope"] == (
        "presented_prompt_evidence_consistency_not_outcome_truth_or_calibration"
    )
    assert assessment.passes is None


def test_cell_tamper_changes_receipt_and_foreign_provenance_fails_closed() -> None:
    request, block_request, block = _frame()
    original = _cell(
        request,
        block,
        target_local_index=0,
        source_card_index=0,
        metric_id="objective:quality",
        presented_delta=11.0,
    )
    changed = replace(original, presented_delta=13.0)

    original_receipt = assess_presented_action_evidence_block_consistency(
        block_request,
        block,
        (original,),
    )
    changed_receipt = assess_presented_action_evidence_block_consistency(
        block_request,
        block,
        (changed,),
    )
    assert changed.cell_sha256 != original.cell_sha256
    assert changed_receipt.cell_set_sha256 != original_receipt.cell_set_sha256
    assert changed_receipt.receipt_sha256 != original_receipt.receipt_sha256
    assert changed_receipt.cell_assessments[0].interval_coverage is False

    foreign_provenance = replace(original, provenance_sha256=_sha("foreign-source"))
    with pytest.raises(ValueError, match="foreign prompt provenance"):
        assess_presented_action_evidence_block_consistency(
            block_request,
            block,
            (foreign_provenance,),
        )


def test_cell_order_and_duplicate_keys_fail_closed() -> None:
    request, block_request, block = _frame()
    cells = _canonical_cells(
        _cell(
            request,
            block,
            target_local_index=0,
            source_card_index=0,
            metric_id="objective:quality",
            presented_delta=11.0,
        ),
        _cell(
            request,
            block,
            target_local_index=1,
            source_card_index=1,
            metric_id="objective:cost",
            presented_delta=0.0,
        ),
    )
    assert len(cells) == 2
    with pytest.raises(ValueError, match="unique and canonical"):
        assess_presented_action_evidence_block_consistency(
            block_request,
            block,
            tuple(reversed(cells)),
        )
    with pytest.raises(ValueError, match="unique and canonical"):
        assess_presented_action_evidence_block_consistency(
            block_request,
            block,
            (cells[0], cells[0]),
        )


def test_foreign_binding_metric_and_block_row_are_rejected() -> None:
    request, block_request, block = _frame()
    valid = _cell(
        request,
        block,
        target_local_index=0,
        source_card_index=0,
        metric_id="objective:quality",
        presented_delta=11.0,
    )
    foreign_binding = replace(
        valid,
        action_evidence_binding_identity_sha256=_sha("foreign-binding"),
    )
    with pytest.raises(ValueError, match="foreign prompt-visible binding"):
        assess_presented_action_evidence_block_consistency(
            block_request,
            block,
            (foreign_binding,),
        )

    foreign_metric = replace(valid, metric_id="objective:foreign")
    with pytest.raises(ValueError, match="metric outside"):
        assess_presented_action_evidence_block_consistency(
            block_request,
            block,
            (foreign_metric,),
        )

    _request_two, _block_request_two, block_two = _frame(block_index=1)
    foreign_row = replace(
        valid,
        option_identity_sha256=block_two.forecasts[0].option_identity_sha256,
    )
    with pytest.raises(ValueError, match="row outside the block"):
        assess_presented_action_evidence_block_consistency(
            block_request,
            block,
            (foreign_row,),
        )


def test_subset_binds_exact_rows_policy_and_forecast_receipt() -> None:
    request, block_request, block = _frame()
    subset_policy = PresentedActionEvidenceSubsetPolicyBinding(
        policy_id="fixture_eligible_rows",
        policy_version=1,
        policy_definition_sha256=_sha("fixture-eligible-rows-v1"),
    )
    included_row = block_request.block.global_row_start
    subset = bind_presented_action_evidence_subset(
        block_request,
        block,
        subset_policy=subset_policy,
        included_global_row_indices=(included_row,),
    )
    included = _cell(
        request,
        block,
        target_local_index=0,
        source_card_index=0,
        metric_id="objective:quality",
        presented_delta=11.0,
    )
    assessment = assess_presented_action_evidence_subset_consistency(
        block_request,
        block,
        (included,),
        subset=subset,
    )
    assert assessment.frame_kind is PresentedActionEvidenceConsistencyFrameKind.SUBSET
    assert assessment.subset_binding == subset
    assert assessment.to_record()["subset_binding"]["binding_sha256"] == (
        subset.binding_sha256
    )
    assert subset.forecast_block_receipt_sha256 == block.receipt_sha256

    excluded = _cell(
        request,
        block,
        target_local_index=1,
        source_card_index=1,
        metric_id="objective:quality",
        presented_delta=8.0,
    )
    with pytest.raises(ValueError, match="row outside the subset"):
        assess_presented_action_evidence_subset_consistency(
            block_request,
            block,
            (excluded,),
            subset=subset,
        )

    tampered_subset = replace(
        subset,
        included_option_identity_sha256s=(_sha("foreign-subset-option"),),
    )
    with pytest.raises(ValueError, match="differs from its exact"):
        assess_presented_action_evidence_subset_consistency(
            block_request,
            block,
            (included,),
            subset=tampered_subset,
        )
    with pytest.raises(ValueError, match="unique and canonical"):
        bind_presented_action_evidence_subset(
            block_request,
            block,
            subset_policy=subset_policy,
            included_global_row_indices=(included_row + 1, included_row),
        )


def test_identified_fail_closed_policy_can_decide_without_changing_default() -> None:
    request, block_request, block = _frame()
    cell = _cell(
        request,
        block,
        target_local_index=0,
        source_card_index=0,
        metric_id="objective:quality",
        presented_delta=11.0,
    )
    strict = PresentedActionEvidenceConsistencyPolicyBinding(
        policy_id="strict_presented_evidence_fixture",
        policy_version=1,
        policy_definition_sha256=_sha("strict-presented-evidence-fixture-v1"),
        maximum_normalized_absolute_error=0.1,
        require_direction_agreement=True,
        require_interval_coverage=True,
    )
    result = assess_presented_action_evidence_block_consistency(
        block_request,
        block,
        (cell,),
        policy=strict,
    )
    assert result.decision_applied is True
    assert result.passes is False
    assert (
        descriptive_presented_action_evidence_consistency_policy().decision_applied
        is False
    )


def test_normalized_statistics_are_invariant_to_unrelated_domain_identity() -> None:
    request_a, block_request_a, block_a = _frame(parent_x=0)
    request_b, block_request_b, block_b = _frame(parent_x=100)
    cell_a = _cell(
        request_a,
        block_a,
        target_local_index=0,
        source_card_index=0,
        metric_id="objective:quality",
        presented_delta=11.0,
    )
    cell_b = _cell(
        request_b,
        block_b,
        target_local_index=0,
        source_card_index=0,
        metric_id="objective:quality",
        presented_delta=11.0,
    )
    assessment_a = assess_presented_action_evidence_block_consistency(
        block_request_a,
        block_a,
        (cell_a,),
    )
    assessment_b = assess_presented_action_evidence_block_consistency(
        block_request_b,
        block_b,
        (cell_b,),
    )
    result_a = assessment_a.cell_assessments[0]
    result_b = assessment_b.cell_assessments[0]
    assert result_a.p10_delta == result_b.p10_delta
    assert result_a.p50_delta == result_b.p50_delta
    assert result_a.p90_delta == result_b.p90_delta
    assert result_a.normalized_absolute_error == result_b.normalized_absolute_error
    assert result_a.direction_agreement == result_b.direction_agreement
    assert result_a.interval_coverage == result_b.interval_coverage
    assert assessment_a.request_sha256 != assessment_b.request_sha256
    assert assessment_a.receipt_sha256 != assessment_b.receipt_sha256


def test_default_policy_definition_and_binding_are_frozen() -> None:
    policy = descriptive_presented_action_evidence_consistency_policy()
    assert PRESENTED_ACTION_EVIDENCE_CONSISTENCY_POLICY_DEFINITION_SHA256 == _sha(
        "agent-evolve:descriptive-presented-action-evidence-consistency:v1;"
        "authenticated-prompt-provenance-required=true;"
        "maximum-normalized-absolute-error=none;"
        "require-direction-agreement=false;"
        "require-interval-coverage=false;"
        "scope=presented-evidence-not-truth-or-calibration"
    )
    assert policy.policy_version == 1
    assert policy.binding_sha256 == (
        "b3b943ededb34246b1f5c2a237e35f5531295b534fd5136ee4b030254f4eefaa"
    )
    assert policy.to_record()["authenticated_provenance_required"] is True
