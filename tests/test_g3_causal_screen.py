"""Executable scientific-contract tests for the generic G0→G3 screen."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys

import pytest

from agent_evolve.application.agentic_evolution import (
    InsightAssignmentKind,
    OperatorKind,
    ProposalAuthority,
)
from agent_evolve.application.g3_causal_screen import (
    G1_DIAGNOSTIC_SLOT_IDS,
    G2_SLOT_IDS,
    G3_SLOT_IDS,
)
from agent_evolve.application.insight_memory import InsightRelationKind
from agent_evolve.policies.memory.prompt_shape import (
    seal_matched_prompt_structure,
)
from agent_evolve.policies.memory.staged_causal import (
    MemoryCheckpointClosureStatus,
)


def _load_fixture_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "development"
        / "g3_provider_free_screen.py"
    )
    name = "_agent_evolve_test_g3_provider_free_screen"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_FIXTURE = _load_fixture_module()
P_H = _FIXTURE.P_H
run_provider_free_g3 = _FIXTURE.run_provider_free_g3


def test_provider_free_g3_executes_the_exact_causal_protocol() -> None:
    run = asyncio.run(run_provider_free_g3())
    result = run.result
    state = result.final_state

    assert state.generation == 3
    assert len(state.candidates) == 12
    assert state.unique_evaluations == 11
    assert state.logical_llm_calls == 6
    assert len(result.seed_receipts) == 2
    assert len(result.generation_receipts) == 3
    assert len(result.feedback_receipts) == 3
    assert tuple(
        tuple(slot_result.slot.slot_id for slot_result in receipt.slot_results)
        for receipt in result.generation_receipts
    ) == (G1_DIAGNOSTIC_SLOT_IDS, G2_SLOT_IDS, G3_SLOT_IDS)

    g1, g2, g3 = result.generation_receipts
    assert (
        g1.logical_llm_calls_before,
        g1.logical_llm_calls_after,
        g1.unique_evaluations_before,
        g1.unique_evaluations_after,
    ) == (0, 2, 2, 4)
    assert (
        g2.logical_llm_calls_before,
        g2.logical_llm_calls_after,
        g2.unique_evaluations_before,
        g2.unique_evaluations_after,
    ) == (2, 5, 4, 8)
    assert (
        g3.logical_llm_calls_before,
        g3.logical_llm_calls_after,
        g3.unique_evaluations_before,
        g3.unique_evaluations_after,
    ) == (5, 5, 8, 11)
    assert tuple(
        slot.slot.proposal_authority for slot in g1.slot_results
    ) == (ProposalAuthority.MODEL,) * 2
    assert tuple(
        slot.slot.proposal_authority for slot in g2.slot_results
    ) == (
        ProposalAuthority.MODEL,
        ProposalAuthority.MODEL,
        ProposalAuthority.MODEL,
        ProposalAuthority.ENGINE,
    )
    assert tuple(
        slot.slot.proposal_authority for slot in g3.slot_results
    ) == (
        ProposalAuthority.REPRODUCTION,
        ProposalAuthority.ENGINE,
        ProposalAuthority.ENGINE,
        ProposalAuthority.ENGINE,
    )
    assert all(
        slot.outcome.failure_stage is None
        and slot.outcome.candidate is not None
        and slot.outcome.candidate.valid
        and slot.outcome.candidate.operator_compliant
        and slot.outcome.candidate.evidence_compliant
        for receipt in result.generation_receipts
        for slot in receipt.slot_results
    )
    assert g3.slot_results[0].outcome.candidate.configuration_dict == P_H
    assert g3.slot_results[0].outcome.candidate.operator_kind is OperatorKind.REPRODUCTION
    assert all(
        value.outcome.candidate.operator_kind
        is OperatorKind.THREE_WAY_RECOMBINATION
        and value.outcome.candidate.preservation_verified is True
        for value in g3.slot_results[1:]
    )

    cache_events = tuple(
        trace["cache_event_type"]
        for trace in run.traces
        if trace.get("event_type") == "evaluation_cache_event"
    )
    assert cache_events == ("miss",) * 8 + ("hit",) + ("miss",) * 3
    assert len(run.problem.evaluations) == 11


def test_provider_free_g3_binds_causal_memory_compilation_and_curation() -> None:
    run = asyncio.run(run_provider_free_g3())
    planner = run.planner
    result = run.result

    assert planner.diagnostic_permutation.permutation_rank == 1
    assert planner.diagnostic_permutation.subset_ranks_by_slot == (1, 0)
    assert planner.wave is not None
    assert planner.closure is not None
    assert planner.closure.status is MemoryCheckpointClosureStatus.SEALED
    assert planner.g1_rendered_prompt_receipt is not None
    assert planner.g2_rendered_prompt_receipt is not None
    assert (
        planner.g1_rendered_prompt_receipt.structure_sha256
        == planner.g2_rendered_prompt_receipt.structure_sha256
    )
    assert len(run.compiler.requests) == 8  # four prepared + four G1-bound
    assert len(run.generator.proposal_requests) == 5
    assert len(run.generator.reflection_requests) == 1
    assert run.curation.invoked_generations == [1, 2, 3]
    assert len(run.curation.curated_entries) == 1
    revision = run.curation.curated_entries[0]
    assert revision.initial_score == 0.0
    assert revision.relations[0].kind is InsightRelationKind.REVISES
    assert revision.relations[0].target == (
        planner.g2_assignments[0].selection_decision.selected[0]
    )
    expected_runtime_identities = (
        run.composition.benchmark,
        run.composition.engine,
        run.composition.id_factory,
        run.composition.memory,
    )
    assert run.planner_factory.runtime_identities == expected_runtime_identities
    assert run.curation_factory.runtime_identities == expected_runtime_identities
    assert run.composition.planner is planner
    assert run.composition.feedback_interceptor is run.curation
    assert tuple(
        value.used_logical_llm_calls for value in result.feedback_receipts
    ) == (0, 0, 1)
    assert run.validation_receipt.curation_status == "sealed_complete"
    assert run.validation_receipt.curation_publication_outcome == (
        "completed_revision"
    )
    assert run.curation.curation_authority is not None
    assert run.curation.curation_receipt is not None
    assert run.curation.curation_receipt.call_receipt.request.source_receipt_sha256s == (
        tuple(value.receipt_hash for value in result.generation_receipts)
    )
    assert run.curation.terminal_validation_receipt is not None
    assert (
        run.curation.terminal_validation_receipt.mechanism_decision.advance_to_replication
        is True
    )

    g1, g2, _ = result.generation_receipts
    assert all(
        value.outcome.candidate.insight_assignment_kind
        is InsightAssignmentKind.RESOLVED_CAUSAL
        for value in g1.slot_results
    )
    assert tuple(
        value.outcome.candidate.insight_assignment_kind
        for value in g2.slot_results[:3]
    ) == (
        InsightAssignmentKind.RESOLVED_CAUSAL,
        InsightAssignmentKind.RESOLVED_CAUSAL,
        InsightAssignmentKind.QUARANTINE_TEST,
    )
    assert g2.slot_results[2].outcome.treatment_admission_receipt is not None
    assert g2.slot_results[2].outcome.candidate.selected_insight_refs == (
        planner.neutral_reference,
    )
    assert all(
        assignment.block_id == "g2_matched_block"
        for assignment in planner.g2_assignments[:2]
    )
    authority = planner.terminal_validation_authority
    assert authority is not None
    assert tuple(
        value.reference for value in authority.g1_expected_endpoints
    ) == tuple(
        value.outcome.candidate.selected_insight_refs[0]
        for value in g1.slot_results
    )
    g3_plan_trace = next(
        value
        for value in run.traces
        if value.get("event_type") == "optimizer_generation_planned"
        and value.get("generation") == 3
    )
    assert dict(g3_plan_trace["metadata"])[
        "terminal_validation_authority_sha256"
    ] == authority.authority_sha256


def test_actual_prompt_structure_receipt_is_value_blinded_but_shape_strict() -> None:
    first = 'HEADER\n[{"claim":"A","score":1.0,"tags":["x"]}]\nEND'
    second = 'HEADER\n[{"claim":"B","score":-2.0,"tags":["y"]}]\nEND'

    receipt = seal_matched_prompt_structure((first, second))

    assert len(receipt.prompt_sha256s) == 2
    with pytest.raises(ValueError, match="do not share one structure"):
        seal_matched_prompt_structure(
            (first, 'HEADER\n[{"claim":"B","extra":true,"score":-2.0,"tags":["y"]}]\nEND')
        )
    with pytest.raises(ValueError, match="do not share one structure"):
        seal_matched_prompt_structure(
            (first, 'HEADER\n[{"claim":"B","score":-2.0,"tags":["y","z"]}]\nEND')
        )


def test_post_g3_provider_failure_preserves_the_sealed_terminal_endpoints() -> None:
    run = asyncio.run(run_provider_free_g3(fail_curation=True))
    result = run.result

    assert result.final_state.generation == 3
    assert len(result.final_state.candidates) == 12
    assert result.final_state.unique_evaluations == 11
    assert result.final_state.logical_llm_calls == 6
    assert run.curation.terminal_validation_receipt is not None
    assert run.curation.curation_failure_type == "RuntimeError"
    assert run.curation.curated_entries == ()
    assert len(run.generator.proposal_requests) == 5
    assert len(run.generator.reflection_requests) == 1
    metadata = dict(result.feedback_receipts[-1].result_metadata)
    assert metadata["curation_status"] == "incomplete"
    assert metadata["curation_failure_type"] == "RuntimeError"
    assert metadata["terminal_validation_receipt_sha256"] == (
        run.curation.terminal_validation_receipt.receipt_sha256
    )
    assert run.validation_receipt.curation_status == "incomplete"
    assert run.validation_receipt.curation_publication_outcome == "failed"
    assert run.curation.curation_receipt is not None
    assert run.curation.curation_receipt.call_receipt.status.value == "failed"
    assert run.curation.curation_receipt.call_receipt.telemetry is None
    assert tuple(
        value.slot.slot_id
        for value in result.generation_receipts[-1].slot_results
    ) == G3_SLOT_IDS


def test_overproduced_revision_batch_is_rejected_before_memory_publication() -> None:
    run = asyncio.run(run_provider_free_g3(overproduce_curation=True))

    assert run.validation_receipt.curation_status == "incomplete"
    assert run.validation_receipt.curation_publication_outcome == "failed"
    assert run.curation.curation_failure_type == "ReflectionCardContractError"
    assert run.curation.curated_entries == ()
    assert len(run.composition.memory.entries) == 3
    assert all(
        entry.reference.version == 1 for entry in run.composition.memory.entries
    )
    metadata = dict(run.result.feedback_receipts[-1].result_metadata)
    assert metadata["curation_status"] == "incomplete"
    assert metadata["curation_failure_type"] == "ReflectionCardContractError"
    assert run.result.final_state.logical_llm_calls == 6
    assert run.curation.curation_receipt is not None
    assert run.curation.curation_receipt.call_receipt.telemetry is not None
