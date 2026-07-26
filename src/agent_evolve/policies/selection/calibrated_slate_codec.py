"""Strict JSON-record decoder for authenticated calibrated-slate requests.

Allocation receipts persist the complete provider-free ``SlateAllocationRequest``
as canonical JSON.  This module reconstructs that domain object without a model
call and requires exact ``to_record()`` equality at every authenticated layer.
It is deliberately separate from the historical policy implementations so
adding replay support cannot change their selection behavior or hashes.
"""

from __future__ import annotations

from typing import Any

from agent_evolve.policies.selection.calibrated_slate import (
    CalibratedSlate,
    CalibratedSlateMember,
    MetricOptimizationGoal,
    SlateAllocationRequest,
    SlateMetricObjective,
    SlateRoleProposal,
    SlateStructuralEvidence,
)
from agent_evolve.policies.selection.forecast_calibration import (
    BetaCorrectnessPrior,
    ForecastCalibrationObservation,
    ForecastCalibrationScope,
    ForecastCalibrationSnapshot,
    ForecastConfidenceBin,
    ForecastPredictionReceipt,
    MeaningfulDirectionAdjudicationReceipt,
)
from agent_evolve.ports.agentic_generator import MetricEffectDirection
from agent_evolve.ports.portfolio_memory_dose import (
    BoundedPortfolioMemoryDoseContract,
    PortfolioMemoryDoseAssessment,
    PortfolioMemoryDoseCardSupport,
    PortfolioMemoryDoseStage,
    PortfolioMemoryDoseViolation,
    PortfolioMemoryExposureScope,
)


def _object(value: object, *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    return value


def _array(value: object, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an exact JSON array")
    return value


def _string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    return value


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")
    return value


def _hex_float(value: object, *, name: str) -> float:
    return float.fromhex(_string(value, name=name))


def _exact_record(
    decoded: object,
    record: dict[str, Any],
    *,
    name: str,
) -> None:
    to_record = getattr(decoded, "to_record", None)
    if not callable(to_record) or to_record() != record:
        raise ValueError(f"{name} fails exact authenticated record replay")


def _scope(record_value: object) -> ForecastCalibrationScope:
    record = _object(record_value, name="calibration scope")
    value = ForecastCalibrationScope(
        model_profile_sha256=_string(
            record.get("model_profile_sha256"), name="model_profile_sha256"
        ),
        prompt_definition_sha256=_string(
            record.get("prompt_definition_sha256"), name="prompt_definition_sha256"
        ),
        selector_policy_definition_sha256=_string(
            record.get("selector_policy_definition_sha256"),
            name="selector_policy_definition_sha256",
        ),
        benchmark_sha256=_string(
            record.get("benchmark_sha256"), name="benchmark_sha256"
        ),
        session_sha256=_string(
            record.get("session_sha256"), name="session_sha256"
        ),
    )
    _exact_record(value, record, name="calibration scope")
    return value


def _prediction(
    record_value: object,
    *,
    scope: ForecastCalibrationScope,
) -> ForecastPredictionReceipt:
    record = _object(record_value, name="forecast prediction")
    if record.get("scope_sha256") != scope.scope_sha256:
        raise ValueError("forecast prediction names a foreign calibration scope")
    value = ForecastPredictionReceipt(
        scope=scope,
        wave_index=_integer(record.get("wave_index"), name="wave_index"),
        selector_decision_sha256=_string(
            record.get("selector_decision_sha256"),
            name="selector_decision_sha256",
        ),
        parent_candidate_identity_sha256=_string(
            record.get("parent_candidate_identity_sha256"),
            name="parent_candidate_identity_sha256",
        ),
        option_id=_string(record.get("option_id"), name="option_id"),
        option_identity_sha256=_string(
            record.get("option_identity_sha256"), name="option_identity_sha256"
        ),
        family=_string(record.get("family"), name="family"),
        metric_id=_string(record.get("metric_id"), name="metric_id"),
        asserted_direction=MetricEffectDirection(
            _string(record.get("asserted_direction"), name="asserted_direction")
        ),
        confidence=ForecastConfidenceBin(
            _string(record.get("confidence"), name="confidence")
        ),
    )
    _exact_record(value, record, name="forecast prediction")
    return value


def _adjudication(record_value: object) -> MeaningfulDirectionAdjudicationReceipt:
    record = _object(record_value, name="direction adjudication")
    policy = _object(record.get("adjudicator"), name="adjudicator policy")
    value = MeaningfulDirectionAdjudicationReceipt(
        request_sha256=_string(
            record.get("request_sha256"), name="request_sha256"
        ),
        benchmark_sha256=_string(
            record.get("benchmark_sha256"), name="benchmark_sha256"
        ),
        session_sha256=_string(
            record.get("session_sha256"), name="session_sha256"
        ),
        wave_index=_integer(record.get("wave_index"), name="wave_index"),
        parent_candidate_identity_sha256=_string(
            record.get("parent_candidate_identity_sha256"),
            name="parent_candidate_identity_sha256",
        ),
        option_id=_string(record.get("option_id"), name="option_id"),
        option_identity_sha256=_string(
            record.get("option_identity_sha256"), name="option_identity_sha256"
        ),
        metric_id=_string(record.get("metric_id"), name="metric_id"),
        parent_outcome_sha256=_string(
            record.get("parent_outcome_sha256"), name="parent_outcome_sha256"
        ),
        child_outcome_sha256=_string(
            record.get("child_outcome_sha256"), name="child_outcome_sha256"
        ),
        actual_direction=MetricEffectDirection(
            _string(record.get("actual_direction"), name="actual_direction")
        ),
        adjudicator_policy_id=_string(
            policy.get("policy_id"), name="adjudicator.policy_id"
        ),
        adjudicator_policy_version=_integer(
            policy.get("policy_version"), name="adjudicator.policy_version"
        ),
        adjudicator_definition_sha256=_string(
            policy.get("definition_sha256"), name="adjudicator.definition_sha256"
        ),
    )
    _exact_record(value, record, name="direction adjudication")
    return value


def _observation(
    record_value: object,
    *,
    scope: ForecastCalibrationScope,
) -> ForecastCalibrationObservation:
    record = _object(record_value, name="calibration observation")
    value = ForecastCalibrationObservation(
        prediction=_prediction(record.get("prediction_receipt"), scope=scope),
        adjudication=_adjudication(record.get("adjudication_receipt")),
    )
    _exact_record(value, record, name="calibration observation")
    return value


def _snapshot(record_value: object) -> ForecastCalibrationSnapshot:
    record = _object(record_value, name="calibration snapshot")
    scope = _scope(record.get("scope"))
    prior_record = _object(record.get("prior"), name="calibration prior")
    prior = BetaCorrectnessPrior(
        alpha=_hex_float(prior_record.get("alpha_hex"), name="prior.alpha_hex"),
        beta=_hex_float(prior_record.get("beta_hex"), name="prior.beta_hex"),
    )
    if prior.to_record() != prior_record:
        raise ValueError("calibration prior fails exact authenticated replay")
    value = ForecastCalibrationSnapshot(
        scope=scope,
        cutoff_wave_index_exclusive=_integer(
            record.get("cutoff_wave_index_exclusive"),
            name="cutoff_wave_index_exclusive",
        ),
        observations=tuple(
            _observation(item, scope=scope)
            for item in _array(record.get("observations"), name="observations")
        ),
        prior=prior,
        family_min_support=_integer(
            record.get("family_min_support"), name="family_min_support"
        ),
    )
    _exact_record(value, record, name="calibration snapshot")
    return value


def _structural_evidence(record_value: object) -> SlateStructuralEvidence:
    record = _object(record_value, name="structural evidence")
    value = SlateStructuralEvidence(
        frozen_archive_snapshot_sha256=_string(
            record.get("frozen_archive_snapshot_sha256"),
            name="frozen_archive_snapshot_sha256",
        ),
        evidence_receipt_sha256=_string(
            record.get("evidence_receipt_sha256"), name="evidence_receipt_sha256"
        ),
        archive_novelty_score=_hex_float(
            record.get("archive_novelty_score_hex"),
            name="archive_novelty_score_hex",
        ),
        structural_coverage_score=_hex_float(
            record.get("structural_coverage_score_hex"),
            name="structural_coverage_score_hex",
        ),
    )
    _exact_record(value, record, name="structural evidence")
    return value


def _slate_member(
    record_value: object,
    *,
    scope: ForecastCalibrationScope,
) -> CalibratedSlateMember:
    record = _object(record_value, name="calibrated slate member")
    value = CalibratedSlateMember(
        model_rank=_integer(record.get("model_rank"), name="model_rank"),
        option_id=_string(record.get("option_id"), name="option_id"),
        option_identity_sha256=_string(
            record.get("option_identity_sha256"), name="option_identity_sha256"
        ),
        family=_string(record.get("family"), name="family"),
        locus_key=_string(record.get("locus_key"), name="locus_key"),
        phenotype_identity_sha256=_string(
            record.get("phenotype_identity_sha256"),
            name="phenotype_identity_sha256",
        ),
        supporting_card_keys=tuple(
            _string(value, name="supporting_card_key")
            for value in _array(
                record.get("supporting_card_keys"), name="supporting_card_keys"
            )
        ),
        role_proposal=SlateRoleProposal(
            _string(record.get("role_proposal"), name="role_proposal")
        ),
        rationale_sha256=_string(
            record.get("rationale_sha256"), name="rationale_sha256"
        ),
        predictions=tuple(
            _prediction(item, scope=scope)
            for item in _array(record.get("predictions"), name="predictions")
        ),
        structural_evidence=_structural_evidence(
            record.get("structural_evidence")
        ),
    )
    _exact_record(value, record, name="calibrated slate member")
    return value


def _slate(
    record_value: object,
    *,
    scope: ForecastCalibrationScope,
) -> CalibratedSlate:
    record = _object(record_value, name="calibrated slate")
    if record.get("scope_sha256") != scope.scope_sha256:
        raise ValueError("calibrated slate names a foreign scope")
    value = CalibratedSlate(
        scope=scope,
        wave_index=_integer(record.get("wave_index"), name="wave_index"),
        selector_decision_sha256=_string(
            record.get("selector_decision_sha256"),
            name="selector_decision_sha256",
        ),
        parent_candidate_identity_sha256=_string(
            record.get("parent_candidate_identity_sha256"),
            name="parent_candidate_identity_sha256",
        ),
        finite_contract_sha256=_string(
            record.get("finite_contract_sha256"), name="finite_contract_sha256"
        ),
        members=tuple(
            _slate_member(item, scope=scope)
            for item in _array(record.get("members"), name="slate.members")
        ),
    )
    _exact_record(value, record, name="calibrated slate")
    return value


def _objective(record_value: object) -> SlateMetricObjective:
    record = _object(record_value, name="slate objective")
    value = SlateMetricObjective(
        metric_id=_string(record.get("metric_id"), name="metric_id"),
        goal=MetricOptimizationGoal(
            _string(record.get("goal"), name="goal")
        ),
        weight=_hex_float(record.get("weight_hex"), name="weight_hex"),
        definition_sha256=_string(
            record.get("definition_sha256"), name="definition_sha256"
        ),
    )
    _exact_record(value, record, name="slate objective")
    return value


def _memory_contract(record_value: object) -> BoundedPortfolioMemoryDoseContract:
    record = _object(record_value, name="memory-dose contract")
    supports: list[PortfolioMemoryDoseCardSupport] = []
    for item in _array(record.get("card_supports"), name="card_supports"):
        support_record = _object(item, name="card support")
        policy = _object(support_record.get("support_policy"), name="support policy")
        compatible = tuple(
            (
                _string(
                    _object(value, name="compatible option").get("option_id"),
                    name="compatible option_id",
                ),
                _string(
                    _object(value, name="compatible option").get(
                        "option_identity_sha256"
                    ),
                    name="compatible option_identity_sha256",
                ),
            )
            for value in _array(
                support_record.get("compatible_options"),
                name="compatible_options",
            )
        )
        support = PortfolioMemoryDoseCardSupport(
            card_key=_string(support_record.get("card_key"), name="card_key"),
            card_content_sha256=_string(
                support_record.get("card_content_sha256"),
                name="card_content_sha256",
            ),
            finite_contract_identity_sha256=_string(
                support_record.get("finite_contract_identity_sha256"),
                name="finite_contract_identity_sha256",
            ),
            compatible_options=compatible,
            support_policy_id=_string(
                policy.get("policy_id"), name="support_policy.policy_id"
            ),
            support_policy_version=_integer(
                policy.get("policy_version"), name="support_policy.policy_version"
            ),
            support_policy_definition_sha256=_string(
                policy.get("definition_sha256"),
                name="support_policy.definition_sha256",
            ),
        )
        if support.to_record() != support_record:
            raise ValueError("card support fails exact authenticated replay")
        supports.append(support)
    policy = _object(record.get("policy"), name="memory-dose policy")

    def bounds(key: str) -> tuple[int, int]:
        cells = _array(record.get(key), name=key)
        if len(cells) != 2:
            raise ValueError(f"{key} must contain exactly two integers")
        return tuple(_integer(value, name=key) for value in cells)  # type: ignore[return-value]

    value = BoundedPortfolioMemoryDoseContract(
        card_supports=tuple(supports),
        proposed_supported_member_bounds=bounds("proposed_supported_member_bounds"),
        evaluated_supported_member_bounds=bounds(
            "evaluated_supported_member_bounds"
        ),
        minimum_unattributed_proposed_members=_integer(
            record.get("minimum_unattributed_proposed_members"),
            name="minimum_unattributed_proposed_members",
        ),
        minimum_unattributed_evaluated_members=_integer(
            record.get("minimum_unattributed_evaluated_members"),
            name="minimum_unattributed_evaluated_members",
        ),
        maximum_cards_per_member=_integer(
            record.get("maximum_cards_per_member"),
            name="maximum_cards_per_member",
        ),
        require_every_assigned_card=_boolean(
            record.get("require_every_assigned_card"),
            name="require_every_assigned_card",
        ),
        exposure_scope=PortfolioMemoryExposureScope(
            _string(record.get("exposure_scope"), name="exposure_scope")
        ),
        policy_id=_string(policy.get("policy_id"), name="policy.policy_id"),
        policy_version=_integer(
            policy.get("policy_version"), name="policy.policy_version"
        ),
        policy_definition_sha256=_string(
            policy.get("definition_sha256"), name="policy.definition_sha256"
        ),
    )
    _exact_record(value, record, name="memory-dose contract")
    return value


def _memory_assessment(record_value: object) -> PortfolioMemoryDoseAssessment:
    record = _object(record_value, name="memory-dose assessment")
    value = PortfolioMemoryDoseAssessment(
        contract_sha256=_string(
            record.get("contract_sha256"), name="contract_sha256"
        ),
        stage=PortfolioMemoryDoseStage(
            _string(record.get("stage"), name="stage")
        ),
        member_content_binding_sha256s=tuple(
            _string(item, name="member_content_binding_sha256")
            for item in _array(
                record.get("member_content_binding_sha256s"),
                name="member_content_binding_sha256s",
            )
        ),
        supported_member_ranks=tuple(
            _integer(item, name="supported_member_rank")
            for item in _array(
                record.get("supported_member_ranks"),
                name="supported_member_ranks",
            )
        ),
        unattributed_member_ranks=tuple(
            _integer(item, name="unattributed_member_rank")
            for item in _array(
                record.get("unattributed_member_ranks"),
                name="unattributed_member_ranks",
            )
        ),
        card_attribution_ranks=tuple(
            (
                _string(
                    _object(item, name="card attribution").get("card_key"),
                    name="card attribution key",
                ),
                tuple(
                    _integer(rank, name="card attribution rank")
                    for rank in _array(
                        _object(item, name="card attribution").get(
                            "member_ranks"
                        ),
                        name="card attribution ranks",
                    )
                ),
            )
            for item in _array(
                record.get("card_attribution_ranks"),
                name="card_attribution_ranks",
            )
        ),
        violations=tuple(
            PortfolioMemoryDoseViolation(
                _string(item, name="memory-dose violation")
            )
            for item in _array(record.get("violations"), name="violations")
        ),
        proposal_assessment_sha256=(
            None
            if record.get("proposal_assessment_sha256") is None
            else _string(
                record.get("proposal_assessment_sha256"),
                name="proposal_assessment_sha256",
            )
        ),
    )
    _exact_record(value, record, name="memory-dose assessment")
    return value


def decode_slate_allocation_request_record(
    record_value: object,
) -> SlateAllocationRequest:
    """Decode and authenticate one canonical ``SlateAllocationRequest`` record."""

    record = _object(record_value, name="slate allocation request")
    snapshot_value = record.get("calibration_snapshot")
    if snapshot_value is None:
        raise ValueError(
            "calibrated slate replay requires a recorded calibration snapshot"
        )
    snapshot = _snapshot(snapshot_value)
    contract_value = record.get("memory_dose_contract")
    assessment_value = record.get("proposal_memory_dose_assessment")
    if (contract_value is None) != (assessment_value is None):
        raise ValueError("memory-dose contract/assessment record is incomplete")
    contract = None if contract_value is None else _memory_contract(contract_value)
    assessment = (
        None
        if assessment_value is None
        else _memory_assessment(assessment_value)
    )
    pairs_value = record.get("pairwise_disjoint_option_id_pairs")
    pairs = (
        None
        if pairs_value is None
        else tuple(
            tuple(
                _string(cell, name="disjoint option_id")
                for cell in _array(item, name="disjoint option pair")
            )
            for item in _array(pairs_value, name="disjoint option pairs")
        )
    )
    value = SlateAllocationRequest(
        slate=_slate(record.get("slate"), scope=snapshot.scope),
        portfolio_size=_integer(
            record.get("portfolio_size"), name="portfolio_size"
        ),
        objectives=tuple(
            _objective(item)
            for item in _array(record.get("objectives"), name="objectives")
        ),
        assigned_card_keys=tuple(
            _string(item, name="assigned_card_key")
            for item in _array(
                record.get("assigned_card_keys"), name="assigned_card_keys"
            )
        ),
        calibration_snapshot=snapshot,
        pairwise_disjoint_option_id_pairs=pairs,  # type: ignore[arg-type]
        min_distinct_families=(
            None
            if record.get("min_distinct_families") is None
            else _integer(
                record.get("min_distinct_families"),
                name="min_distinct_families",
            )
        ),
        memory_dose_contract=contract,
        proposal_memory_dose_assessment=assessment,
    )
    _exact_record(value, record, name="slate allocation request")
    return value


__all__ = ["decode_slate_allocation_request_record"]
