from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from decimal import Decimal

import pytest

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    EvolutionCandidate,
    MutationResponseMode,
    OperatorKind,
    ProposalAuthority,
)
from agent_evolve.application.budgeted_optimizer import (
    BudgetedAgenticOptimizer,
    OptimizerBudget,
    OptimizerState,
    pareto_archive_snapshot_hash,
)
from agent_evolve.application.pareto_archive import (
    EvidenceAdmissionPolicy,
    ParetoArchive,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import (
    canonical_typed_json_bytes,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from examples.benchmarks.boils_abc import budgeted_v5_support as support
from examples.benchmarks.boils_abc.actions import ACTION_IDS, CandidateConfig
from examples.benchmarks.boils_abc.budgeted_v5_planner import (
    BoilsBudgetedV5Planner,
    BoilsV5FrozenFrontAlignedReward,
    BoilsV5PlanningError,
    PALETTE_SEED,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    AtomicMutationDraft,
    ReflectionGenerationResult,
    VariationGenerationResult,
)


OBJECTIVES = (
    ObjectiveSpec("total_lut_count", "min"),
    ObjectiveSpec("total_levels", "min"),
)
BUDGET = OptimizerBudget(
    max_unique_evaluations=9,
    max_logical_llm_calls=5,
    max_generations=2,
)


def _candidate(
    *,
    candidate_id: CandidateId,
    configuration: dict[str, object],
    objectives: tuple[tuple[str, float], ...],
    generation: int,
    label: str,
    proposal_sequence: int,
    parent: EvolutionCandidate | None = None,
    valid: bool = True,
    operator_compliant: bool = True,
) -> EvolutionCandidate:
    frozen = freeze_json(configuration)
    occurrence = CandidateOccurrence(
        candidate_id=candidate_id,
        configuration_hash=typed_json_sha256(frozen),
        configuration_artifact_hash=hashlib.sha256(
            canonical_typed_json_bytes(frozen)
        ).hexdigest(),
        proposal_sequence=proposal_sequence,
    )
    return EvolutionCandidate(
        occurrence=occurrence,
        configuration=frozen,
        objectives=objectives if valid else (),
        valid=valid,
        generation=generation,
        label=label,
        operator_kind=None if generation == 0 else OperatorKind.TYPED_MUTATION,
        parent_ids=() if parent is None else (parent.candidate_id,),
        operator_compliant=operator_compliant,
        operator_failure=None if operator_compliant else "offline contract failure",
    )


def _seed(ids: DeterministicIdFactory) -> EvolutionCandidate:
    return _candidate(
        candidate_id=ids.new_candidate_id(),
        configuration=support.parent_c_config(),
        objectives=support.PARENT_C_OBJECTIVES,
        generation=0,
        label="seed_0",
        proposal_sequence=1,
    )


def _state(
    *,
    generation: int,
    candidates: tuple[EvolutionCandidate, ...],
    calls: int,
    evaluations: int,
) -> OptimizerState:
    archive = ParetoArchive(
        OBJECTIVES,
        evidence_admission_policy=EvidenceAdmissionPolicy.RECORD_ONLY,
    )
    for candidate in candidates:
        archive.consider(candidate)
    snapshot = archive.snapshot()
    return OptimizerState(
        generation=generation,
        candidates=candidates,
        archive=snapshot,
        archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
        unique_evaluations=evaluations,
        logical_llm_calls=calls,
    )


def _replace(parent: EvolutionCandidate, index: int, value: str) -> dict[str, object]:
    configuration = parent.configuration_dict
    sequence = list(configuration["sequence"])
    sequence[index] = value
    return {"sequence": sequence}


def test_generation_one_freezes_exact_agentic_slots_palettes_and_ledger() -> None:
    ids = DeterministicIdFactory("boils_v5_g1")
    seed = _seed(ids)
    durable_decisions: list[dict[str, object]] = []
    planner = BoilsBudgetedV5Planner(ids, decision_sink=durable_decisions.append)

    plan = planner.plan(
        _state(generation=0, candidates=(seed,), calls=0, evaluations=1),
        BUDGET,
    )

    assert [slot.slot_id for slot in plan.slots] == [
        "G1-A1",
        "G1-A2",
        "G1-D1",
        "G1-D2",
        "G1-U",
        "G1-X",
    ]
    assert [slot.plan.label for slot in plan.slots] == [
        slot.slot_id for slot in plan.slots
    ]
    assert plan.logical_llm_call_reservation == 5
    assert plan.unique_evaluation_reservation == 6
    assert [slot.proposal_authority for slot in plan.slots] == [
        ProposalAuthority.MODEL,
        ProposalAuthority.MODEL,
        ProposalAuthority.MODEL,
        ProposalAuthority.MODEL,
        ProposalAuthority.MODEL,
        ProposalAuthority.ENGINE,
    ]
    assert [slot.plan.phase for slot in plan.slots[:5]] == [
        support.AREA_PHASE,
        support.AREA_PHASE,
        support.DEPTH_PHASE,
        support.DEPTH_PHASE,
        support.UNCERTAINTY_PHASE,
    ]
    for slot in plan.slots[:4]:
        assert slot.plan.use_memory is True
        assert slot.plan.memory_subset_size == 1
        assert slot.plan.memory_exploration_probability == 1
        assert slot.plan.memory_score_phase == slot.plan.phase
    assert plan.slots[4].plan.use_memory is False
    assert plan.slots[5].plan.use_memory is False
    assert all(
        slot.plan.mutation_response_mode
        is MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
        for slot in plan.slots
    )
    assert all(
        len(slot.plan.atomic_replacement_options) == 3 for slot in plan.slots[:5]
    )
    assert len(plan.slots[5].plan.atomic_replacement_options) == 1
    assert support.AREA_REQUIRED_ACTION in plan.slots[0].plan.atomic_replacement_options
    assert (
        support.DEPTH_REQUIRED_ACTION in plan.slots[2].plan.atomic_replacement_options
    )
    assert (
        plan.slots[0].plan.atomic_replacement_options
        == plan.slots[1].plan.atomic_replacement_options
    )
    assert (
        plan.slots[2].plan.atomic_replacement_options
        == plan.slots[3].plan.atomic_replacement_options
    )
    assert plan.slots[4].plan.mutation_contract.editable_paths == (
        support.UNCERTAINTY_PATH,
    )
    assert plan.slots[5].plan.mutation_contract.editable_paths == (
        support.COVERAGE_PATH,
    )
    assert plan.slots[5].materialized is not None
    assert (
        plan.slots[5].materialized.draft.replacement
        in plan.slots[5].plan.atomic_replacement_options
    )

    expected_palettes = {
        "G1-A1": ("dsdb", "resub", "blut"),
        "G1-A2": ("dsdb", "resub", "blut"),
        "G1-D1": ("balance", "sopb", "fraig"),
        "G1-D2": ("balance", "sopb", "fraig"),
        "G1-U": ("sopb", "dsdb", "refactor_z"),
        "G1-X": ("blut",),
    }
    expected_palette_hashes = {
        "G1-A1": "114b08f63889d57f45875e0f1ee58e2922bbf17a6a32be0cb7a9fdcbe0ad9be5",
        "G1-A2": "114b08f63889d57f45875e0f1ee58e2922bbf17a6a32be0cb7a9fdcbe0ad9be5",
        "G1-D1": "333660dca1d6afbd323939d729c382d36838968e821d322e0a4f0516aef3b9c8",
        "G1-D2": "333660dca1d6afbd323939d729c382d36838968e821d322e0a4f0516aef3b9c8",
        "G1-U": "cd3049b331322a566362a3d8a75295fbd48b42d464544c4c803aa99b382692d0",
        "G1-X": "581c61ee9e39bee2f90ec578d9508275499ea304c83df45fb9933fa5cd913f1d",
    }
    assert {
        slot.slot_id: slot.plan.atomic_replacement_options for slot in plan.slots
    } == expected_palettes

    decision = planner.generation1_decision
    assert decision is not None
    assert {
        slot.slot_id: slot.palette.decision_sha256 for slot in decision.slots
    } == expected_palette_hashes
    assert decision.slots[0].palette == decision.slots[1].palette
    assert decision.slots[2].palette == decision.slots[3].palette
    assert decision.slots[1].exposures_before != decision.slots[0].exposures_before
    assert decision.slots[3].exposures_before != decision.slots[2].exposures_before
    assert len(durable_decisions) == 1
    assert durable_decisions[0]["decision_sha256"] == decision.decision_sha256
    assert dict(plan.metadata)["decision_sha256"] == decision.decision_sha256
    assert durable_decisions[0]["slots"][0]["palette_options"]
    obligation = durable_decisions[0]["uncertainty_palette_obligation"]
    assert obligation == {
        "obligation_id": support.UNCERTAINTY_COVERAGE_OBLIGATION_ID,
        "obligation_version": 1,
        "path": support.UNCERTAINTY_PATH_TEXT,
        "required_action": "dsdb",
        "required_family": "gia_dsd_balance",
        "required_option_id": decision.slot("G1-U").palette.required_option_ids[0],
        "rationale": support.UNCERTAINTY_COVERAGE_OBLIGATION_RATIONALE,
    }
    assert PALETTE_SEED == 20_260_714


def _close_g1(
    *,
    ids: DeterministicIdFactory,
    seed: EvolutionCandidate,
    planner: BoilsBudgetedV5Planner,
    plan,
) -> tuple[EvolutionCandidate, ...]:
    objective_rows = (
        (("total_lut_count", 7_925.0), ("total_levels", 69.0)),
        (("total_lut_count", 7_950.0), ("total_levels", 68.0)),
        (("total_lut_count", 7_935.0), ("total_levels", 68.0)),
        (("total_lut_count", 7_940.0), ("total_levels", 69.0)),
        (("total_lut_count", 7_930.0), ("total_levels", 70.0)),
        (("total_lut_count", 7_900.0), ("total_levels", 70.0)),
    )
    choice_indices = (0, 1, 0, 1, 0, 0)
    values: list[EvolutionCandidate] = []
    for ordinal, (slot, objectives, choice_index) in enumerate(
        zip(plan.slots, objective_rows, choice_indices, strict=True),
        start=2,
    ):
        contract = slot.plan.mutation_contract
        assert contract is not None
        path = contract.editable_paths[0]
        index = path.segments[1].value
        replacement = slot.plan.atomic_replacement_options[choice_index]
        assert type(replacement) is str
        candidate_id = (
            slot.materialized.candidate_id
            if slot.materialized is not None
            else ids.new_candidate_id()
        )
        values.append(
            _candidate(
                candidate_id=candidate_id,
                configuration=_replace(seed, index, replacement),
                objectives=objectives,
                generation=1,
                label=slot.plan.label,
                proposal_sequence=ordinal,
                parent=seed,
            )
        )
    return tuple(values)


def test_frozen_reward_credits_front_extension_clipped_by_hv_reference() -> None:
    ids = DeterministicIdFactory("boils_v5_front_reward")
    seed = _seed(ids)
    planner = BoilsBudgetedV5Planner(ids)
    plan = planner.plan(
        _state(generation=0, candidates=(seed,), calls=0, evaluations=1),
        BUDGET,
    )
    reward = plan.reward.binding.score
    assert type(reward) is BoilsV5FrozenFrontAlignedReward

    clipped_extreme = _candidate(
        candidate_id=ids.new_candidate_id(),
        configuration=_replace(seed, support.AREA_PATH_INDEX, "dsdb"),
        objectives=(("total_lut_count", 8_063.0), ("total_levels", 67.0)),
        generation=1,
        label="clipped_extreme",
        proposal_sequence=2,
        parent=seed,
    )
    dominated = _candidate(
        candidate_id=ids.new_candidate_id(),
        configuration=_replace(seed, support.AREA_PATH_INDEX, "resub"),
        objectives=(("total_lut_count", 8_063.0), ("total_levels", 70.0)),
        generation=1,
        label="dominated",
        proposal_sequence=3,
        parent=seed,
    )

    extreme_record = reward.record(clipped_extreme)
    dominated_record = reward.record(dominated)
    assert extreme_record.base_hypervolume_record.reward == 0.0
    assert extreme_record.strictly_extends_frozen_front is True
    assert extreme_record.front_extension_raw_credit == 1.0
    assert extreme_record.reward == pytest.approx(1 / 168)
    assert dominated_record.base_hypervolume_record.reward == 0.0
    assert dominated_record.strictly_extends_frozen_front is False
    assert dominated_record.front_extension_raw_credit == 0.0
    assert dominated_record.reward == 0.0


def test_generation_two_enumerates_every_pair_and_materializes_selected_unions() -> (
    None
):
    ids = DeterministicIdFactory("boils_v5_g2")
    seed = _seed(ids)
    durable_decisions: list[dict[str, object]] = []
    planner = BoilsBudgetedV5Planner(ids, decision_sink=durable_decisions.append)
    g1_plan = planner.plan(
        _state(generation=0, candidates=(seed,), calls=0, evaluations=1),
        BUDGET,
    )
    g1 = _close_g1(ids=ids, seed=seed, planner=planner, plan=g1_plan)
    state = _state(
        generation=1,
        candidates=(seed, *g1),
        calls=5,
        evaluations=7,
    )

    g2_plan = planner.plan(state, BUDGET)

    decision = planner.generation2_decision
    assert decision is not None
    assert len(decision.enumeration) == 15
    assert sum(row.eligible for row in decision.enumeration) == 13
    assert len(decision.selection.eligible_rows) == 13
    assert [slot.slot_id for slot in g2_plan.slots] == ["G2-E", "G2-X"]
    assert g2_plan.logical_llm_call_reservation == 0
    assert g2_plan.unique_evaluation_reservation == 2
    assert all(
        slot.proposal_authority is ProposalAuthority.ENGINE for slot in g2_plan.slots
    )
    assert all(slot.materialized is not None for slot in g2_plan.slots)
    assert all(slot.plan.label == slot.slot_id for slot in g2_plan.slots)
    assert all(
        slot.materialized.materialization_policy_id == "disjoint_patch_union"
        for slot in g2_plan.slots
    )
    assert all(
        slot.materialized.candidate_id
        not in {
            seed.candidate_id,
            *(parent.candidate_id for parent in slot.plan.parents),
        }
        for slot in g2_plan.slots
    )
    assert decision.selection.exploit is not None
    assert decision.selection.coverage is not None
    assert (
        decision.selection.exploit.pair.target_configuration_sha256
        != decision.selection.coverage.pair.target_configuration_sha256
    )
    assert len(decision.individual_frozen_rewards) == 6
    assert all(row.status == "eligible" for row in decision.g1_checkpoint)
    exploit_paths = {
        dict(decision.selection.branch_paths)[candidate_id]
        for candidate_id in decision.selection.exploit.pair_ids
    }
    coverage_paths = {
        dict(decision.selection.branch_paths)[candidate_id]
        for candidate_id in decision.selection.coverage.pair_ids
    }
    assert coverage_paths - exploit_paths
    assert len(durable_decisions) == 2
    assert durable_decisions[1]["decision_sha256"] == decision.decision_sha256
    assert len(durable_decisions[1]["enumeration"]) == 15
    assert dict(g2_plan.metadata)["decision_sha256"] == decision.decision_sha256
    summary = planner.to_summary_record()
    assert summary["generation1"]["decision_sha256"] == (
        planner.generation1_decision.decision_sha256
    )
    assert summary["generation2"]["decision_sha256"] == decision.decision_sha256
    assert len(summary["summary_sha256"]) == 64


def test_g2_records_missing_slot_and_continues_without_substitution() -> None:
    ids = DeterministicIdFactory("boils_v5_incomplete")
    seed = _seed(ids)
    planner = BoilsBudgetedV5Planner(ids)
    g1_plan = planner.plan(
        _state(generation=0, candidates=(seed,), calls=0, evaluations=1),
        BUDGET,
    )
    g1 = _close_g1(ids=ids, seed=seed, planner=planner, plan=g1_plan)
    incomplete = _state(
        generation=1,
        candidates=(seed, *g1[:-1]),
        calls=5,
        evaluations=6,
    )

    plan = planner.plan(incomplete, BUDGET)
    decision = planner.generation2_decision

    assert decision is not None
    assert [row.status for row in decision.g1_checkpoint] == [
        "eligible",
        "eligible",
        "eligible",
        "eligible",
        "eligible",
        "missing_candidate",
    ]
    assert decision.g1_checkpoint[-1].candidate_id is None
    assert all(
        parent.label != "G1-X" for slot in plan.slots for parent in slot.plan.parents
    )
    trace = decision.to_trace_record()["failed_slot_continuation"]
    assert trace["substitution_allowed"] is False
    assert trace["g1_checkpoint"][-1] == {
        "slot_id": "G1-X",
        "status": "missing_candidate",
        "candidate_id": None,
    }


def test_g2_missing_uncertainty_slot_keeps_coverage_and_replays_exactly() -> None:
    def freeze_decision():
        ids = DeterministicIdFactory("boils_v5_missing_u")
        seed = _seed(ids)
        planner = BoilsBudgetedV5Planner(ids)
        g1_plan = planner.plan(
            _state(generation=0, candidates=(seed,), calls=0, evaluations=1),
            BUDGET,
        )
        g1 = _close_g1(ids=ids, seed=seed, planner=planner, plan=g1_plan)
        incomplete = _state(
            generation=1,
            candidates=(seed, *g1[:4], g1[5]),
            calls=5,
            evaluations=6,
        )
        plan = planner.plan(incomplete, BUDGET)
        decision = planner.generation2_decision
        assert decision is not None
        return plan, decision

    plan, decision = freeze_decision()
    replay_plan, replay_decision = freeze_decision()

    assert [row.status for row in decision.g1_checkpoint] == [
        "eligible",
        "eligible",
        "eligible",
        "eligible",
        "missing_candidate",
        "eligible",
    ]
    assert decision.g1_checkpoint[4].candidate_id is None
    assert decision.g1_checkpoint[5].candidate_id is not None
    assert all(
        parent.label != "G1-U" for slot in plan.slots for parent in slot.plan.parents
    )
    assert any(
        parent.label == "G1-X" for slot in plan.slots for parent in slot.plan.parents
    )
    assert len(decision.enumeration) == 10
    assert len(decision.selection.eligible_rows) == 8
    assert decision.to_trace_record() == replay_decision.to_trace_record()
    assert [slot.plan.parents for slot in plan.slots] == [
        slot.plan.parents for slot in replay_plan.slots
    ]


@pytest.mark.parametrize("valid_count", (0, 1))
def test_g2_records_typed_empty_wave_with_fewer_than_two_eligible_branches(
    valid_count: int,
) -> None:
    ids = DeterministicIdFactory(f"boils_v5_sparse_{valid_count}")
    seed = _seed(ids)
    records: list[dict[str, object]] = []
    planner = BoilsBudgetedV5Planner(ids, decision_sink=records.append)
    g1_plan = planner.plan(
        _state(generation=0, candidates=(seed,), calls=0, evaluations=1),
        BUDGET,
    )
    original = _close_g1(ids=ids, seed=seed, planner=planner, plan=g1_plan)
    g1 = tuple(
        candidate
        if index < valid_count
        else replace(
            candidate,
            objectives=(),
            valid=False,
            failure_message="offline candidate-local failure",
        )
        for index, candidate in enumerate(original)
    )

    plan = planner.plan(
        _state(
            generation=1,
            candidates=(seed, *g1),
            calls=5,
            evaluations=7,
        ),
        BUDGET,
    )

    decision = planner.generation2_decision
    assert decision is not None
    assert decision.enumeration == ()
    assert decision.selection.exploit is None
    assert decision.selection.coverage is None
    assert decision.selected_slot_ids == ()
    assert plan.slots == ()
    assert plan.logical_llm_call_reservation == 0
    assert plan.unique_evaluation_reservation == 0
    assert records[-1]["decision_sha256"] == decision.decision_sha256


def test_g2_records_typed_empty_wave_when_only_pair_is_not_replay_safe() -> None:
    ids = DeterministicIdFactory("boils_v5_no_safe_pair")
    seed = _seed(ids)
    planner = BoilsBudgetedV5Planner(ids)
    g1_plan = planner.plan(
        _state(generation=0, candidates=(seed,), calls=0, evaluations=1),
        BUDGET,
    )
    original = _close_g1(ids=ids, seed=seed, planner=planner, plan=g1_plan)
    # A1 and A2 are distinct edits of the same path; their only pair is a
    # replay-classified conflict and cannot be unioned by the mechanical policy.
    g1 = tuple(
        candidate
        if index < 2
        else replace(
            candidate,
            objectives=(),
            valid=False,
            failure_message="offline candidate-local failure",
        )
        for index, candidate in enumerate(original)
    )

    plan = planner.plan(
        _state(
            generation=1,
            candidates=(seed, *g1),
            calls=5,
            evaluations=7,
        ),
        BUDGET,
    )

    decision = planner.generation2_decision
    assert decision is not None
    assert len(decision.enumeration) == 1
    assert decision.enumeration[0].eligible is False
    assert decision.enumeration[0].rejection_type == "DisjointPatchRecombinationError"
    assert decision.selection.eligible_rows == ()
    assert decision.selected_slot_ids == ()
    assert plan.slots == ()


def test_decision_sink_failure_blocks_plan_before_engine_materialization() -> None:
    ids = DeterministicIdFactory("boils_v5_sink")
    seed = _seed(ids)
    calls = 0

    def fail(_record) -> None:
        nonlocal calls
        calls += 1
        raise OSError("fsync failed")

    planner = BoilsBudgetedV5Planner(ids, decision_sink=fail)
    with pytest.raises(OSError, match="fsync failed"):
        planner.plan(
            _state(generation=0, candidates=(seed,), calls=0, evaluations=1),
            BUDGET,
        )

    assert calls == 1
    assert planner.generation1_decision is not None
    with pytest.raises(BoilsV5PlanningError, match="already been frozen"):
        planner.plan(
            _state(generation=0, candidates=(seed,), calls=0, evaluations=1),
            BUDGET,
        )


class _OfflineBoilsProblem:
    candidate_model = CandidateConfig
    objectives = OBJECTIVES

    def __init__(self, events: list[tuple[str, str]]) -> None:
        self.events = events

    @staticmethod
    def search_space_description() -> str:
        return "Offline BOiLS contract fake."

    @staticmethod
    def validate(configuration: object) -> bool:
        CandidateConfig.model_validate(configuration, strict=True)
        return True

    def evaluate(self, configuration: dict[str, object]) -> dict[str, float]:
        parsed = CandidateConfig.model_validate(configuration, strict=True)
        sequence = tuple(parsed.sequence)
        self.events.append(("evaluation", typed_json_sha256(configuration)))
        if sequence == support.PARENT_C_SEQUENCE:
            return dict(support.PARENT_C_OBJECTIVES)
        changes = tuple(
            (index, action)
            for index, action in enumerate(sequence)
            if action != support.PARENT_C_SEQUENCE[index]
        )
        score = sum(
            (index + 1) * (ACTION_IDS.index(action) + 1) for index, action in changes
        )
        return {
            "total_lut_count": float(7_944 - score),
            "total_levels": float(
                69 - int(any(action == "fraig" for _, action in changes))
            ),
        }


def _offline_telemetry(ordinal: int) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="offline",
        provider_response_id=f"response-{ordinal}",
        finish_reason="stop",
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
    )


class _OfflineAtomicGenerator:
    def __init__(self, events: list[tuple[str, str]]) -> None:
        self.events = events
        self.calls = 0

    async def propose(self, request):
        self.calls += 1
        self.events.append(("provider", request.call_id.value))
        contract = request.atomic_mutation_contract
        assert contract is not None
        return VariationGenerationResult(
            draft=AtomicMutationDraft(
                path=contract.editable_path,
                replacement=contract.replacement_options[0],
                design_rationale="Offline deterministic first-option hypothesis.",
            ),
            telemetry=_offline_telemetry(self.calls),
        )

    async def reflect(self, request):
        return ReflectionGenerationResult((), _offline_telemetry(self.calls + 1))


def test_durable_decision_precedes_every_provider_and_child_evaluation() -> None:
    ids = DeterministicIdFactory("boils_v5_ordering")
    events: list[tuple[str, str]] = []
    memory, references = support.build_v5_insight_memory(ids)
    generator = _OfflineAtomicGenerator(events)
    problem = _OfflineBoilsProblem(events)
    planner = BoilsBudgetedV5Planner(
        ids,
        decision_sink=lambda record: events.append(
            ("decision", str(record["decision_sha256"]))
        ),
    )
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=support.ENGINE_RNG_SEED,
        prompt_builder=support.BoilsV5RolePromptRouter(),
    )
    archive = ParetoArchive(
        OBJECTIVES,
        evidence_admission_policy=EvidenceAdmissionPolicy.RECORD_ONLY,
    )
    optimizer = BudgetedAgenticOptimizer(
        engine=engine,
        archive=archive,
        planner=planner,
        budget=OptimizerBudget(7, 5, 1),
    )

    result = asyncio.run(optimizer.run((support.parent_c_config(),)))

    decision_index = next(
        index for index, event in enumerate(events) if event[0] == "decision"
    )
    provider_indices = [
        index for index, event in enumerate(events) if event[0] == "provider"
    ]
    evaluation_indices = [
        index for index, event in enumerate(events) if event[0] == "evaluation"
    ]
    assert len(provider_indices) == 5
    assert len(evaluation_indices) >= 2  # seed plus at least one unique child
    assert decision_index < min(provider_indices)
    assert decision_index < min(evaluation_indices[1:])
    expected = dict(references.expected_slot_references())
    by_label = {
        candidate.label: candidate for candidate in result.final_state.candidates
    }
    for label in ("G1-A1", "G1-A2", "G1-D1", "G1-D2"):
        assert by_label[label].selected_insight_refs == (expected[label],)
    assert by_label["G1-U"].selected_insight_refs == ()
    assert by_label["G1-X"].call_telemetry is None
