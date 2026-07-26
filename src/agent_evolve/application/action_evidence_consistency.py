"""Join prompt-visible action evidence to exact resolved forecast cells.

The benchmark injects :class:`PresentedActionEvidenceCell` values.  This
application service validates their prompt provenance and performs only a pure
join and arithmetic projection.  It has no outcome-store, evaluator, provider,
or benchmark dependency and therefore cannot silently turn presented evidence
into a truth or calibration claim.
"""

from __future__ import annotations

import hashlib

from agent_evolve.domain.finite_variation import FiniteActionEvidenceBinding
from agent_evolve.ports.action_forecast import (
    ActionForecastBlockRequest,
    ResolvedActionForecast,
    ResolvedActionForecastBlock,
    ResolvedActionMetricForecast,
    validate_resolved_action_forecast_block,
)
from agent_evolve.ports.portfolio_selection import PortfolioCard
from agent_evolve.ports.presented_action_evidence import (
    PresentedActionEvidenceCell,
    PresentedActionEvidenceCellAssessment,
    PresentedActionEvidenceConsistencyAssessment,
    PresentedActionEvidenceConsistencyFrameKind,
    PresentedActionEvidenceConsistencyPolicyBinding,
    PresentedActionEvidenceProvenanceKind,
    PresentedActionEvidenceSubsetBinding,
    PresentedActionEvidenceSubsetPolicyBinding,
)


PRESENTED_ACTION_EVIDENCE_CONSISTENCY_POLICY_ID = (
    "descriptive_presented_action_evidence_consistency"
)
PRESENTED_ACTION_EVIDENCE_CONSISTENCY_POLICY_VERSION = 1
PRESENTED_ACTION_EVIDENCE_CONSISTENCY_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:descriptive-presented-action-evidence-consistency:v1;"
    b"authenticated-prompt-provenance-required=true;"
    b"maximum-normalized-absolute-error=none;"
    b"require-direction-agreement=false;"
    b"require-interval-coverage=false;"
    b"scope=presented-evidence-not-truth-or-calibration"
).hexdigest()


def descriptive_presented_action_evidence_consistency_policy(
) -> PresentedActionEvidenceConsistencyPolicyBinding:
    """Return the default authenticated, descriptive-only policy binding."""

    return PresentedActionEvidenceConsistencyPolicyBinding(
        policy_id=PRESENTED_ACTION_EVIDENCE_CONSISTENCY_POLICY_ID,
        policy_version=PRESENTED_ACTION_EVIDENCE_CONSISTENCY_POLICY_VERSION,
        policy_definition_sha256=(
            PRESENTED_ACTION_EVIDENCE_CONSISTENCY_POLICY_DEFINITION_SHA256
        ),
        maximum_normalized_absolute_error=None,
        require_direction_agreement=False,
        require_interval_coverage=False,
    )


def _validate_block(
    block_request: ActionForecastBlockRequest,
    block: ResolvedActionForecastBlock,
) -> None:
    if type(block_request) is not ActionForecastBlockRequest:
        raise TypeError("block_request must be an exact ActionForecastBlockRequest")
    if type(block) is not ResolvedActionForecastBlock:
        raise TypeError("block must be an exact ResolvedActionForecastBlock")
    validate_resolved_action_forecast_block(block_request, block)


def bind_presented_action_evidence_subset(
    block_request: ActionForecastBlockRequest,
    block: ResolvedActionForecastBlock,
    *,
    subset_policy: PresentedActionEvidenceSubsetPolicyBinding,
    included_global_row_indices: tuple[int, ...],
) -> PresentedActionEvidenceSubsetBinding:
    """Bind an identified exact row subset to one resolved block receipt."""

    _validate_block(block_request, block)
    if type(subset_policy) is not PresentedActionEvidenceSubsetPolicyBinding:
        raise TypeError("subset_policy must be an exact subset-policy binding")
    subset_policy.__post_init__()
    if type(included_global_row_indices) is not tuple or not (
        included_global_row_indices
    ) or any(type(value) is not int for value in included_global_row_indices):
        raise ValueError(
            "included_global_row_indices must be a non-empty exact tuple"
        )
    if included_global_row_indices != tuple(
        sorted(set(included_global_row_indices))
    ):
        raise ValueError("included global rows must be unique and canonical")
    spec = block_request.block
    if any(
        value < spec.global_row_start or value >= spec.global_row_stop
        for value in included_global_row_indices
    ):
        raise ValueError("an included global row is outside the exact block")
    option_identity_sha256s = tuple(
        block.forecasts[value - spec.global_row_start].option_identity_sha256
        for value in included_global_row_indices
    )
    return PresentedActionEvidenceSubsetBinding(
        subset_policy=subset_policy,
        request_sha256=block_request.request.request_sha256,
        layout_sha256=block_request.layout.layout_sha256,
        block_request_sha256=block_request.block_request_sha256,
        block_spec_sha256=spec.block_spec_sha256,
        forecast_block_receipt_sha256=block.receipt_sha256,
        block_index=spec.block_index,
        included_global_row_indices=included_global_row_indices,
        included_option_identity_sha256s=option_identity_sha256s,
    )


def _validate_subset(
    block_request: ActionForecastBlockRequest,
    block: ResolvedActionForecastBlock,
    subset: PresentedActionEvidenceSubsetBinding,
) -> None:
    if type(subset) is not PresentedActionEvidenceSubsetBinding:
        raise TypeError("subset must be an exact presented-evidence subset binding")
    subset.__post_init__()
    try:
        expected = bind_presented_action_evidence_subset(
            block_request,
            block,
            subset_policy=subset.subset_policy,
            included_global_row_indices=subset.included_global_row_indices,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("subset is invalid for the exact forecast block") from error
    if subset != expected:
        raise ValueError("subset differs from its exact forecast block binding")


def _prompt_binding_index(
    block_request: ActionForecastBlockRequest,
) -> dict[
    tuple[str, str],
    tuple[PortfolioCard, FiniteActionEvidenceBinding],
]:
    request = block_request.request
    index: dict[
        tuple[str, str],
        tuple[PortfolioCard, FiniteActionEvidenceBinding],
    ] = {}
    for card in request.cards:
        card.__post_init__()
        for binding in card.finite_action_evidence:
            binding.__post_init__()
            key = (card.card_key, binding.identity_sha256)
            if key in index:
                raise ValueError(
                    "request has an ambiguous prompt-visible action binding"
                )
            index[key] = (card, binding)
    return index


def _validate_cell_provenance(
    *,
    block_request: ActionForecastBlockRequest,
    cell: PresentedActionEvidenceCell,
    card: PortfolioCard,
) -> None:
    source_binding = card.source_binding
    if source_binding is None:
        raise ValueError("presented evidence requires a source-bound prompt card")
    if (
        cell.provenance_kind
        is PresentedActionEvidenceProvenanceKind.CARD_SOURCE_RECEIPT
    ):
        expected = source_binding.source_receipt_sha256
    elif (
        cell.provenance_kind
        is PresentedActionEvidenceProvenanceKind.CARD_VIEW_RECEIPT
    ):
        if card.derived_view_receipt is None:
            raise ValueError(
                "card-view provenance requires an exact derived-view receipt"
            )
        expected = card.derived_view_receipt.receipt_sha256
    else:
        expected = block_request.request.card_snapshot_sha256
    if cell.provenance_sha256 != expected:
        raise ValueError("presented evidence names foreign prompt provenance")


def _same_direction(left: float, right: float) -> bool:
    if left == 0.0 or right == 0.0:
        return left == right
    return (left > 0.0) == (right > 0.0)


def _cell_assessment(
    *,
    block_request: ActionForecastBlockRequest,
    cell: PresentedActionEvidenceCell,
    global_row_index: int,
    forecast: ResolvedActionForecast,
    metric: ResolvedActionMetricForecast,
    card: PortfolioCard,
    binding: FiniteActionEvidenceBinding,
) -> PresentedActionEvidenceCellAssessment:
    scale = next(
        value
        for value in block_request.request.metric_scales
        if value.metric_id == cell.metric_id
    )
    source_binding = card.source_binding
    if source_binding is None:  # Defensive after request and provenance validation.
        raise ValueError("presented evidence card lost its source binding")
    action_identity = cell.action_evidence_binding_identity_sha256
    cites = any(
        citation.card_key == cell.card_key
        and citation.action_binding_identity_sha256 == action_identity
        for citation in metric.citations
    )
    return PresentedActionEvidenceCellAssessment(
        presented_cell_sha256=cell.cell_sha256,
        global_row_index=global_row_index,
        option_id=forecast.option_id,
        option_identity_sha256=forecast.option_identity_sha256,
        metric_id=metric.metric_id,
        presented_delta=cell.presented_delta,
        p10_delta=metric.p10_delta,
        p50_delta=metric.p50_delta,
        p90_delta=metric.p90_delta,
        direction_agreement=_same_direction(
            metric.p50_delta,
            cell.presented_delta,
        ),
        interval_coverage=(
            metric.p10_delta <= cell.presented_delta <= metric.p90_delta
        ),
        normalized_absolute_error=(
            abs(metric.p50_delta - cell.presented_delta) / scale.delta_scale
        ),
        metric_delta_scale=scale.delta_scale,
        metric_scale_definition_sha256=scale.definition_sha256,
        card_key=cell.card_key,
        card_source_binding_sha256=source_binding.binding_sha256,
        card_source_receipt_sha256=source_binding.source_receipt_sha256,
        card_view_receipt_sha256=(
            None
            if card.derived_view_receipt is None
            else card.derived_view_receipt.receipt_sha256
        ),
        action_evidence_binding_identity_sha256=action_identity,
        source_contrast_id=binding.contrast_id,
        source_option_id=binding.option_id,
        source_family=binding.family,
        source_option_identity_sha256=binding.option_identity_sha256,
        source_contract_identity_sha256=binding.contract_identity_sha256,
        provenance_kind=cell.provenance_kind,
        provenance_sha256=cell.provenance_sha256,
        forecast_cites_presented_binding=cites,
    )


def _assess_presented_action_evidence_consistency(
    block_request: ActionForecastBlockRequest,
    block: ResolvedActionForecastBlock,
    cells: tuple[PresentedActionEvidenceCell, ...],
    *,
    subset: PresentedActionEvidenceSubsetBinding | None,
    policy: PresentedActionEvidenceConsistencyPolicyBinding | None,
) -> PresentedActionEvidenceConsistencyAssessment:
    _validate_block(block_request, block)
    if type(cells) is not tuple or not cells or any(
        type(value) is not PresentedActionEvidenceCell for value in cells
    ):
        raise ValueError("cells must be a non-empty exact presented-evidence tuple")
    for value in cells:
        value.__post_init__()
    keys = tuple(value.sort_key for value in cells)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("presented evidence cells must be unique and canonical")
    join_keys = tuple(value.join_key for value in cells)
    if len(set(join_keys)) != len(join_keys):
        raise ValueError("presented evidence cells cannot repeat an evidence join")
    resolved_policy = (
        descriptive_presented_action_evidence_consistency_policy()
        if policy is None
        else policy
    )
    if type(resolved_policy) is not PresentedActionEvidenceConsistencyPolicyBinding:
        raise TypeError("policy must be an exact consistency-policy binding")
    resolved_policy.__post_init__()
    if subset is not None:
        _validate_subset(block_request, block, subset)

    spec = block_request.block
    forecast_index = {
        forecast.option_identity_sha256: (
            spec.global_row_start + local_index,
            forecast,
        )
        for local_index, forecast in enumerate(block.forecasts)
    }
    allowed_option_identities = (
        frozenset(forecast_index)
        if subset is None
        else frozenset(subset.included_option_identity_sha256s)
    )
    prompt_bindings = _prompt_binding_index(block_request)
    assessed: list[PresentedActionEvidenceCellAssessment] = []
    for cell in cells:
        target = forecast_index.get(cell.option_identity_sha256)
        if target is None:
            raise ValueError("presented evidence targets a row outside the block")
        if cell.option_identity_sha256 not in allowed_option_identities:
            raise ValueError("presented evidence targets a row outside the subset")
        global_row_index, forecast = target
        metrics = {
            value.metric_id: value for value in forecast.metric_forecasts
        }
        metric = metrics.get(cell.metric_id)
        if metric is None:
            raise ValueError(
                "presented evidence targets a metric outside the resolved row"
            )
        prompt_binding = prompt_bindings.get(
            (
                cell.card_key,
                cell.action_evidence_binding_identity_sha256,
            )
        )
        if prompt_binding is None:
            raise ValueError(
                "presented evidence names a foreign prompt-visible binding"
            )
        card, binding = prompt_binding
        _validate_cell_provenance(
            block_request=block_request,
            cell=cell,
            card=card,
        )
        assessed.append(
            _cell_assessment(
                block_request=block_request,
                cell=cell,
                global_row_index=global_row_index,
                forecast=forecast,
                metric=metric,
                card=card,
                binding=binding,
            )
        )
    canonical_assessments = tuple(sorted(assessed, key=lambda value: value.sort_key))
    decision_applied = resolved_policy.decision_applied
    passes = (
        None
        if not decision_applied
        else all(
            (
                resolved_policy.maximum_normalized_absolute_error is None
                or value.normalized_absolute_error
                <= resolved_policy.maximum_normalized_absolute_error
            )
            and (
                not resolved_policy.require_direction_agreement
                or value.direction_agreement
            )
            and (
                not resolved_policy.require_interval_coverage
                or value.interval_coverage
            )
            for value in canonical_assessments
        )
    )
    experimental_view_receipt = block_request.request.experimental_view_receipt
    if experimental_view_receipt is None:
        raise ValueError(
            "presented evidence requires an authenticated experimental view"
        )
    return PresentedActionEvidenceConsistencyAssessment(
        frame_kind=(
            PresentedActionEvidenceConsistencyFrameKind.BLOCK
            if subset is None
            else PresentedActionEvidenceConsistencyFrameKind.SUBSET
        ),
        request_sha256=block_request.request.request_sha256,
        request_card_snapshot_sha256=(
            block_request.request.card_snapshot_sha256
        ),
        experimental_view_receipt_sha256=(
            experimental_view_receipt.receipt_sha256
        ),
        layout_sha256=block_request.layout.layout_sha256,
        block_request_sha256=block_request.block_request_sha256,
        block_spec_sha256=spec.block_spec_sha256,
        block_index=spec.block_index,
        forecast_block_receipt_sha256=block.receipt_sha256,
        subset_binding=subset,
        policy=resolved_policy,
        cell_assessments=canonical_assessments,
        decision_applied=decision_applied,
        passes=passes,
    )


def assess_presented_action_evidence_block_consistency(
    block_request: ActionForecastBlockRequest,
    block: ResolvedActionForecastBlock,
    cells: tuple[PresentedActionEvidenceCell, ...],
    *,
    policy: PresentedActionEvidenceConsistencyPolicyBinding | None = None,
) -> PresentedActionEvidenceConsistencyAssessment:
    """Assess injected prompt evidence against one complete resolved block."""

    return _assess_presented_action_evidence_consistency(
        block_request,
        block,
        cells,
        subset=None,
        policy=policy,
    )


def assess_presented_action_evidence_subset_consistency(
    block_request: ActionForecastBlockRequest,
    block: ResolvedActionForecastBlock,
    cells: tuple[PresentedActionEvidenceCell, ...],
    *,
    subset: PresentedActionEvidenceSubsetBinding,
    policy: PresentedActionEvidenceConsistencyPolicyBinding | None = None,
) -> PresentedActionEvidenceConsistencyAssessment:
    """Assess injected prompt evidence inside one authenticated row subset."""

    return _assess_presented_action_evidence_consistency(
        block_request,
        block,
        cells,
        subset=subset,
        policy=policy,
    )


__all__ = [
    "PRESENTED_ACTION_EVIDENCE_CONSISTENCY_POLICY_DEFINITION_SHA256",
    "PRESENTED_ACTION_EVIDENCE_CONSISTENCY_POLICY_ID",
    "PRESENTED_ACTION_EVIDENCE_CONSISTENCY_POLICY_VERSION",
    "assess_presented_action_evidence_block_consistency",
    "assess_presented_action_evidence_subset_consistency",
    "bind_presented_action_evidence_subset",
    "descriptive_presented_action_evidence_consistency_policy",
]
