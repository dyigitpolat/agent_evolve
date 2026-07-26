from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace

import pytest

from agent_evolve.application.campaign_generation_audit import (
    TransactionalPortfolioGenerationAuditor,
)
from agent_evolve.application.insight_memory import QuarantineTestAdmissionReceipt
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from tests.test_portfolio_evolution import _build_wave, _rebind_credit_plan


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _diagnostic_wave():
    ids, _, _, _, _, _, wave = asyncio.run(
        _build_wave("production_generation_audit_context")
    )
    credit = wave.memory_credit
    assert credit is not None
    reference = credit.decision.eligible[0]
    admission = QuarantineTestAdmissionReceipt(
        references=(reference,),
        operator_kind="typed_mutation",
        editable_paths=("$.x",),
        source_admission_request_sha256=_sha("context-test-admission"),
        memory_trial_count_cutoff=0,
    )
    credit = replace(credit, quarantine_admission=admission)
    return ids, replace(wave, memory_credit=credit), reference


def test_diagnostic_context_ignores_normal_eligible_control_cards() -> None:
    _, wave, reference = _diagnostic_wave()
    credit = wave.memory_credit
    assert credit is not None
    assert len(credit.decision.eligible) > 1

    bindings = TransactionalPortfolioGenerationAuditor._diagnostic_context_bindings(
        waves=(wave,),
        planned_references=(reference,),
    )

    assert tuple(value.reference for value in bindings) == (reference,)
    assert bindings[0].exact_context_sha256 == credit.assignment.exact_context_hash
    assert bindings[0].assignment_sha256s == (credit.assignment.assignment_sha256,)


def test_one_diagnostic_reference_cannot_mix_preoutcome_contexts() -> None:
    ids, first, reference = _diagnostic_wave()
    first_credit = first.memory_credit
    assert first_credit is not None
    second_context = freeze_json({"benchmark": "different-provider-free-context"})
    second_request = replace(
        first.selection_request,
        call_id=ids.new_llm_call_id(),
        context=second_context,
    )
    second_decision = replace(
        first_credit.decision,
        context_hash=typed_json_sha256(second_context),
    )
    second_credit = _rebind_credit_plan(
        first_credit,
        decision=second_decision,
        credit_unit_id=ids.new_operator_invocation_id(),
    )
    second = replace(
        first,
        selection_request=second_request,
        label_prefix="portfolio_wave_second_context",
        memory_credit=second_credit,
    )

    with pytest.raises(ValueError, match="cannot mix exact estimand contexts"):
        TransactionalPortfolioGenerationAuditor._diagnostic_context_bindings(
            waves=(first, second),
            planned_references=(reference,),
        )
