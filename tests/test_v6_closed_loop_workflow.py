"""Provider-free proof of the complete v6 closed-loop mechanism."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

from agent_evolve.application.agentic_evolution import ProposalAuthority
from agent_evolve.policies.memory.staged_causal import (
    DeterministicMemoryControlPolicy,
    MemoryCheckpointClosureStatus,
)


def _load_support():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "development"
        / "v6_closed_loop_probe_support.py"
    )
    name = "_agent_evolve_v6_closed_loop_probe_support"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_SUPPORT = _load_support()
OfflineInsightConditionedGenerator = _SUPPORT.OfflineInsightConditionedGenerator
compose_probe = _SUPPORT.compose_probe
offline_generator_factory = _SUPPORT.offline_generator_factory


def test_v6_closed_loop_memory_recourse_and_disjoint_recombination() -> None:
    traces: list[dict[str, object]] = []
    composition = compose_probe(
        generator_factory=offline_generator_factory,
        trace_sink=traces.append,
    )
    result = asyncio.run(composition.optimizer.run(({"a": 4, "b": 4},)))
    planner = composition.planner
    generator = composition.generator
    assert isinstance(generator, OfflineInsightConditionedGenerator)

    # G1 creates no live mutable credit. Only its complete delayed receipt can
    # publish checkpoint one, which learns B's positive and A's negative effect.
    assert generator.selected_ids == [
        (composition.a_ref.insight_id.value,),
        (composition.b_ref.insight_id.value,),
        (composition.b_ref.insight_id.value,),
        (composition.a_ref.insight_id.value,),
    ]
    assert generator.mutable_trial_counts == [0, 0, 0, 0]
    assert composition.memory.trials == ()
    assert planner.closure is not None
    assert planner.closure.status is MemoryCheckpointClosureStatus.SEALED
    snapshot = planner.closure.snapshot
    assert snapshot is not None
    assert planner.wave is not None
    assert snapshot.checkpoint_index == 1
    assert snapshot.parent_snapshot_sha256 == planner.genesis.snapshot_sha256
    assert snapshot.source_wave_sha256 == planner.wave.wave_sha256
    by_ref = {entry.reference: entry for entry in snapshot.entries}
    assert by_ref[composition.a_ref].effect_estimate == -2.0
    assert by_ref[composition.b_ref].effect_estimate == 2.0
    assert by_ref[composition.a_ref].retrieval_score == -1.0
    assert by_ref[composition.b_ref].retrieval_score == 1.0

    # G1 and G2 commitments come from the engine's treatment-blinded prompt
    # projection. The exact permutation-rank-one control swaps labelled scores.
    adaptive = planner.adaptive_assignment
    control = planner.control_assignment
    assert adaptive is not None and control is not None
    assert planner.prompt_shape_sha256 is not None
    assert all(
        assignment.prompt_shape_sha256 == planner.prompt_shape_sha256
        for assignment in (*planner.diagnostic_assignments, adaptive, control)
    )
    controls = DeterministicMemoryControlPolicy()
    assert adaptive.selection_decision == controls.adaptive(
        snapshot=snapshot,
        subset_size=1,
    )
    assert control.selection_decision == controls.score_shuffled(
        snapshot=snapshot,
        subset_size=1,
        permutation_rank=1,
    )
    assert adaptive.selection_decision.selected == (composition.b_ref,)
    assert control.selection_decision.selected == (composition.a_ref,)
    assert controls.score_shuffled(
        snapshot=snapshot,
        subset_size=1,
        permutation_rank=0,
    ).selected == (composition.b_ref,)
    assert dict(control.selection_decision.score_snapshot) == {
        composition.a_ref: 1.0,
        composition.b_ref: -1.0,
    }

    g1, g2, g3, g4 = result.generation_receipts
    assert [item.outcome.candidate.configuration_dict for item in g1.slot_results] == [
        {"a": 3, "b": 4},
        {"a": 1, "b": 4},
    ]
    assert [
        item.outcome.candidate.configuration_dict for item in g2.slot_results[:2]
    ] == [{"a": 1, "b": 4}, {"a": 3, "b": 4}]

    # Four primary occurrences produce three phenotype clusters and one new
    # physical evaluation. The duplicate coverage pair is coalesced in flight.
    decision = planner.recourse_decision
    assert decision is not None
    ledger = decision.ledger
    assert len(ledger.primary_occurrences) == 4
    assert len({item.trial_id for item in ledger.primary_occurrences}) == 4
    assert len(ledger.clusters) == 3
    duplicate_cluster = next(
        cluster for cluster in ledger.clusters if len(cluster.occurrences) == 2
    )
    assert {item.candidate_id for item in duplicate_cluster.occurrences} == {
        g2.slot_results[2].outcome.candidate.candidate_id,
        g2.slot_results[3].outcome.candidate.candidate_id,
    }
    assert ledger.successful_primary_collision_credit == 1
    assert decision.selected_entry_ids == ("orthogonal_b",)
    assert (g2.unique_evaluations_before, g2.unique_evaluations_after) == (3, 4)
    assert g2.reserved_unique_evaluations == 4
    cache = asyncio.run(composition.engine.evaluation_cache_snapshot())
    assert cache["misses"] == 6
    assert cache["hits"] == 2
    assert cache["coalesced"] == 1
    assert len(composition.problem.evaluated) == 6

    # Recourse consumes only identity/status/budget facts and cannot chain.
    assert g3.slot_results[0].outcome.candidate.configuration_dict == {
        "a": 4,
        "b": 1,
    }
    assert "reward" not in repr(decision.to_trace_record()).lower()
    assert "objective" not in repr(decision.to_trace_record()).lower()
    assert planner.combined_ledger is not None
    recourse_trial = g3.slot_results[0].outcome.prepared.operator_invocation_id
    assert planner.combined_ledger.ignored_recourse_trial_ids == (recourse_trial,)
    assert planner.combined_ledger.successful_primary_collision_credit == 1
    assert all("recourse" not in item.slot.role for item in g4.slot_results)

    # Engine-authored replay combines the learned a branch and recourse b branch.
    final = g4.slot_results[0].outcome
    assert final.prepared.proposal_authority is ProposalAuthority.ENGINE
    assert final.prepared.call_id is None
    assert final.candidate is not None
    assert final.candidate.configuration_dict == {"a": 1, "b": 1}
    assert final.candidate.operator_compliant
    assert final.candidate.evidence_compliant
    assert final.candidate.preservation_verified
    assert result.final_state.unique_evaluations == 6
    assert result.final_state.logical_llm_calls == 4
    assert len(result.final_state.candidates) == 9
    assert tuple(
        candidate.configuration_dict
        for candidate in result.final_state.archive.front_candidates
    ) == ({"a": 1, "b": 1},)

    # Trace order is the causal contract: frozen, committed, terminated, sealed,
    # published, then matched G2 commitments. Engine verifies every commitment.
    event_types = [event["event_type"] for event in traces]
    assert event_types.count("memory_wave_frozen") == 1
    assert event_types.count("memory_wave_sealed") == 1
    assert event_types.count("memory_checkpoint_published") == 1
    committed = [
        event for event in traces if event["event_type"] == "assignment_committed"
    ]
    terminal = [event for event in traces if event["event_type"] == "trial_terminal"]
    assert len(committed) == len(terminal) == 4
    assert all(event["prompt_shape_commitment_verified"] is True for event in committed)
    assert all(event["terminal_status"] == "succeeded" for event in terminal)
    diagnostic_hashes = {
        assignment.assignment_sha256 for assignment in planner.diagnostic_assignments
    }
    matched_hashes = {adaptive.assignment_sha256, control.assignment_sha256}
    frozen_index = event_types.index("memory_wave_frozen")
    sealed_index = event_types.index("memory_wave_sealed")
    published_index = event_types.index("memory_checkpoint_published")
    diagnostic_commit_indices = [
        index
        for index, event in enumerate(traces)
        if event["event_type"] == "assignment_committed"
        and event["assignment_sha256"] in diagnostic_hashes
    ]
    diagnostic_terminal_indices = [
        index
        for index, event in enumerate(traces)
        if event["event_type"] == "trial_terminal"
        and event["assignment_sha256"] in diagnostic_hashes
    ]
    matched_commit_indices = [
        index
        for index, event in enumerate(traces)
        if event["event_type"] == "assignment_committed"
        and event["assignment_sha256"] in matched_hashes
    ]
    assert frozen_index < min(diagnostic_commit_indices)
    assert max(diagnostic_commit_indices) < min(diagnostic_terminal_indices)
    assert max(diagnostic_terminal_indices) < sealed_index < published_index
    assert published_index < min(matched_commit_indices)
    assert traces[sealed_index]["generation_receipt_hash"] == g1.receipt_hash
    assert traces[published_index]["snapshot_sha256"] == snapshot.snapshot_sha256
