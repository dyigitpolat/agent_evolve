from __future__ import annotations

import asyncio
import importlib.util
from dataclasses import replace
from pathlib import Path
import sys

import pytest

from agent_evolve.application.budgeted_optimizer import (
    generation_receipt_hash,
    optimizer_result_hash,
)
from agent_evolve.application.g3_causal_validation import (
    G3TerminalValidationError,
    validate_g3_causal_screen_result,
    validate_g3_terminal_state,
)
from agent_evolve.application.generation_feedback import (
    generation_feedback_receipt_hash,
)


def _load_fixture_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "development"
        / "g3_provider_free_screen.py"
    )
    name = "_agent_evolve_test_g3_causal_validation_fixture"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_provider_free_g3 = _load_fixture_module().run_provider_free_g3


def _run_fixture():
    return asyncio.run(run_provider_free_g3())


def _pre_curation_state(run):
    state = run.result.final_state
    return replace(
        state,
        logical_llm_calls=5,
        feedback_receipts=state.feedback_receipts[:2],
    )


def _cache_snapshot(run):
    return asyncio.run(run.composition.engine.evaluation_cache_snapshot())


def _reseal_g3(state, slot_index, slot_result, *, candidate=None):
    original = state.generation_receipts[2]
    slot_results = list(original.slot_results)
    slot_results[slot_index] = slot_result
    provisional = replace(
        original,
        slot_results=tuple(slot_results),
        receipt_hash="0" * 64,
    )
    sealed = replace(
        provisional,
        receipt_hash=generation_receipt_hash(provisional),
    )
    receipts = (*state.generation_receipts[:2], sealed)
    candidates = list(state.candidates)
    if candidate is not None:
        candidates[8 + slot_index] = candidate
    return replace(
        state,
        candidates=tuple(candidates),
        generation_receipts=receipts,
    )


def test_terminal_gate_authenticates_exact_public_composed_g3_run() -> None:
    run = _run_fixture()
    receipt = validate_g3_terminal_state(
        state=_pre_curation_state(run),
        planner=run.planner,
        evaluation_cache_snapshot=_cache_snapshot(run),
    )

    assert receipt.receipt_sha256
    assert len(receipt.occurrence_ids) == 12
    assert len(set(receipt.phenotype_identity_sha256s)) == 11
    assert receipt.mechanism_decision.h1_pass is True
    assert receipt.mechanism_decision.h2_pass is True
    assert receipt.mechanism_decision.h3_pass is True
    assert receipt.mechanism_decision.advance_to_replication is True
    assert receipt.mechanism_decision.kill_reasons == ()


def test_terminal_gate_rejects_nonexact_cache_accounting() -> None:
    run = _run_fixture()
    snapshot = _cache_snapshot(run)
    snapshot["hits"] = 2

    with pytest.raises(G3TerminalValidationError, match="cache"):
        validate_g3_terminal_state(
            state=_pre_curation_state(run),
            planner=run.planner,
            evaluation_cache_snapshot=snapshot,
        )


def test_terminal_gate_rejects_resealed_reproduction_with_wrong_parent() -> None:
    run = _run_fixture()
    state = _pre_curation_state(run)
    original = state.generation_receipts[2].slot_results[0]
    wrong_plan = replace(original.slot.plan, parents=(state.candidates[0],))
    wrong_slot = replace(original.slot, plan=wrong_plan)
    wrong_prepared = replace(original.outcome.prepared, plan=wrong_plan)
    wrong_outcome = replace(original.outcome, prepared=wrong_prepared)
    tampered = _reseal_g3(
        state,
        0,
        replace(original, slot=wrong_slot, outcome=wrong_outcome),
    )

    with pytest.raises(G3TerminalValidationError, match="lineage|reproduction"):
        validate_g3_terminal_state(
            state=tampered,
            planner=run.planner,
            evaluation_cache_snapshot=_cache_snapshot(run),
        )


def test_terminal_gate_rejects_resealed_union_without_preservation() -> None:
    run = _run_fixture()
    state = _pre_curation_state(run)
    original = state.generation_receipts[2].slot_results[1]
    assert original.outcome.candidate is not None
    wrong_candidate = replace(
        original.outcome.candidate,
        preservation_verified=False,
    )
    wrong_outcome = replace(original.outcome, candidate=wrong_candidate)
    tampered = _reseal_g3(
        state,
        1,
        replace(original, outcome=wrong_outcome),
        candidate=wrong_candidate,
    )

    with pytest.raises(G3TerminalValidationError, match="union"):
        validate_g3_terminal_state(
            state=tampered,
            planner=run.planner,
            evaluation_cache_snapshot=_cache_snapshot(run),
        )


def test_terminal_gate_rejects_changed_diagnostic_seed_occurrence() -> None:
    run = _run_fixture()
    state = _pre_curation_state(run)
    diagnostic = state.candidates[0]
    wrong_occurrence = replace(
        diagnostic.occurrence,
        proposal_sequence=diagnostic.occurrence.proposal_sequence + 100,
    )
    wrong_diagnostic = replace(diagnostic, occurrence=wrong_occurrence)
    tampered = replace(
        state,
        candidates=(wrong_diagnostic, *state.candidates[1:]),
    )

    with pytest.raises(G3TerminalValidationError, match="seed occurrences"):
        validate_g3_terminal_state(
            state=tampered,
            planner=run.planner,
            evaluation_cache_snapshot=_cache_snapshot(run),
        )


def test_terminal_gate_rejects_resealed_candidate_side_lineage() -> None:
    run = _run_fixture()
    state = _pre_curation_state(run)
    original = state.generation_receipts[2].slot_results[0]
    assert original.outcome.candidate is not None
    wrong_candidate = replace(
        original.outcome.candidate,
        parent_ids=(state.candidates[0].candidate_id,),
    )
    wrong_outcome = replace(original.outcome, candidate=wrong_candidate)
    tampered = _reseal_g3(
        state,
        0,
        replace(original, outcome=wrong_outcome),
        candidate=wrong_candidate,
    )

    with pytest.raises(G3TerminalValidationError, match="lineage"):
        validate_g3_terminal_state(
            state=tampered,
            planner=run.planner,
            evaluation_cache_snapshot=_cache_snapshot(run),
        )


def _validate_final(run, *, result=None, curation_receipt=None):
    assert run.curation.curation_authority is not None
    assert run.curation.curation_receipt is not None
    return validate_g3_causal_screen_result(
        run.result if result is None else result,
        planner=run.planner,
        evaluation_cache_snapshot=_cache_snapshot(run),
        curation_spec=run.curation.spec,
        curation_authority=run.curation.curation_authority,
        curation_receipt=(
            run.curation.curation_receipt
            if curation_receipt is None
            else curation_receipt
        ),
    )


def test_final_gate_binds_engine_request_provider_and_revision_publication() -> None:
    run = _run_fixture()
    curation = run.curation
    assert curation.curation_receipt is not None
    call = curation.curation_receipt.call_receipt

    assert call.status.value == "completed"
    assert call.telemetry_sha256 is not None
    assert len(call.request.source_receipt_sha256s) == 3
    assert call.request.source_receipt_sha256s == tuple(
        receipt.receipt_hash for receipt in run.result.generation_receipts
    )
    assert len(call.request.source_operator_invocation_ids) == 10
    assert len(call.request.source_outcome_sha256s) == 10
    assert call.request.insight_contract_sha256 == (
        curation.spec.insight_contract.identity_sha256
    )
    assert call.request.revision_predecessors == (
        curation.curation_authority.revision_predecessor,
    )
    assert run.composition.engine.reflection_call_receipt(call.call_id) is call
    assert curation.curation_receipt.publication_outcome == "completed_revision"
    assert _validate_final(run).curation_publication_outcome == (
        "completed_revision"
    )


def test_final_gate_rejects_resealed_foreign_feedback_policy() -> None:
    run = _run_fixture()
    original = run.result.feedback_receipts[-1]
    provisional_feedback = replace(
        original,
        policy_id="foreign_self_attested_curation",
        policy_version=77,
        reservation_hash="a" * 64,
        receipt_hash="0" * 64,
    )
    foreign_feedback = replace(
        provisional_feedback,
        receipt_hash=generation_feedback_receipt_hash(provisional_feedback),
    )
    feedback_receipts = (
        *run.result.feedback_receipts[:2],
        foreign_feedback,
    )
    state = replace(
        run.result.final_state,
        feedback_receipts=feedback_receipts,
    )
    provisional_result = replace(
        run.result,
        final_state=state,
        feedback_receipts=feedback_receipts,
        result_hash="0" * 64,
    )
    foreign_result = replace(
        provisional_result,
        result_hash=optimizer_result_hash(provisional_result),
    )

    with pytest.raises(G3TerminalValidationError, match="policy|reservation"):
        _validate_final(run, result=foreign_result)


def test_final_gate_rejects_resealed_foreign_source_row_hash() -> None:
    run = _run_fixture()
    assert run.curation.curation_receipt is not None
    original = run.curation.curation_receipt
    request = original.call_receipt.request
    tampered_request = replace(
        request,
        source_outcome_sha256s=(
            "b" * 64,
            *request.source_outcome_sha256s[1:],
        ),
        request_sha256="",
    )
    tampered_call = replace(
        original.call_receipt,
        request=tampered_request,
        receipt_sha256="",
    )
    tampered_curation = replace(
        original,
        call_receipt=tampered_call,
        receipt_sha256="",
    )

    with pytest.raises(G3TerminalValidationError, match="engine-stored"):
        _validate_final(run, curation_receipt=tampered_curation)
