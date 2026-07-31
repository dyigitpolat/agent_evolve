from __future__ import annotations

import json

import pytest

from agent_evolve.agentic import freeze_json, typed_json_sha256
from agent_evolve.application.anchor_residual_identification import (
    AnchorResidualIdentificationContract,
    AnchorResidualSelectionAudit,
    REGRET_BOUNDED_SELECTOR_AUDIT_KIND,
    STRICTLY_PRIOR_CERTIFICATE_SCOPE,
    project_anchor_residual_selection_audits,
)
from agent_evolve.application.campaign_execution import (
    CampaignSelectorAuditReceipt,
    CampaignStageReceipt,
    SelectorAuditExecutionMode,
    encode_selector_audit_text,
)
from agent_evolve.application.evolution_campaign import CampaignGenerationKind


def _allocation(*, retained: int = 3) -> dict[str, object]:
    reference = ["anchor.1", "anchor.2", "anchor.3", "anchor.4"]
    selected = ["anchor.1", "anchor.3", "anchor.4", "residual.1"]
    return {
        "policy_id": "regret_bounded_information_slate",
        "policy_version": 2,
        "policy_definition_sha256": "a" * 64,
        "certificate_scope": STRICTLY_PRIOR_CERTIFICATE_SCOPE,
        "reference_option_ids": reference,
        "selected_option_ids": selected,
        "reference_member_count": retained,
        "minimum_residual_audit_members": 1,
        "selected_residual_member_count": 1,
        "acquisition_retention_ratio_hex": float(0.94).hex(),
        "minimum_acquisition_retention_ratio_hex": float(0.50).hex(),
    }


def _selector_receipt() -> CampaignSelectorAuditReceipt:
    request_sha256 = "b" * 64
    decision_sha256 = "c" * 64
    response = {
        "ranked_decision": {"decision_sha256": decision_sha256},
        "supplemental_selector_audit": {
            "audit_kind": REGRET_BOUNDED_SELECTOR_AUDIT_KIND,
            "request_sha256": request_sha256,
            "decision_sha256": decision_sha256,
            "payload": {"allocation": _allocation()},
        },
    }
    plaintext = freeze_json(
        {
            "selector_call_id": "call.1",
            "request_sha256": request_sha256,
            "decision_sha256": decision_sha256,
            **encode_selector_audit_text("request_text", "sealed request"),
            **encode_selector_audit_text(
                "response_text",
                json.dumps(response, sort_keys=True, separators=(",", ":")),
            ),
        }
    )
    return CampaignSelectorAuditReceipt(
        generation=1,
        parent_slot=0,
        selector_call_id="call.1",
        request_sha256=request_sha256,
        decision_sha256=decision_sha256,
        trace_receipt_sha256=typed_json_sha256(plaintext),
        plaintext_audit=plaintext,
        prior_audit_set_sha256="d" * 64,
        execution_mode=SelectorAuditExecutionMode.FRESH,
    )


def test_projects_anchor_residual_audit_from_generic_campaign_receipt() -> None:
    stage = CampaignStageReceipt(
        request_sha256="e" * 64,
        preparation_sha256="f" * 64,
        generation=1,
        kind=CampaignGenerationKind.PORTFOLIO,
        candidate_occurrence_count=4,
        unique_evaluation_count=4,
        selector_audits=(_selector_receipt(),),
        result=freeze_json({}),
    )

    records = project_anchor_residual_selection_audits((stage,))
    assert len(records) == 1
    assert records[0].retained_anchor_option_ids == (
        "anchor.1",
        "anchor.3",
        "anchor.4",
    )
    assert records[0].residual_option_ids == ("residual.1",)
    assert records[0].displaced_anchor_option_ids == ("anchor.2",)

    assessment = AnchorResidualIdentificationContract(
        expected_selector_calls=1,
        portfolio_width=4,
        minimum_residual_members=1,
        exact_residual_members=1,
    ).assess(records)
    assert assessment.all_gates_pass is True
    assert assessment.to_record()["retained_anchor_member_count"] == 3
    assert assessment.to_record()["residual_member_count"] == 1


def test_rejects_reported_anchor_count_that_does_not_close() -> None:
    with pytest.raises(ValueError, match="retained-anchor count"):
        AnchorResidualSelectionAudit.from_allocation_record(
            generation=1,
            parent_slot=0,
            selector_call_id="call.1",
            request_sha256="b" * 64,
            decision_sha256="c" * 64,
            allocation=_allocation(retained=4),
        )
