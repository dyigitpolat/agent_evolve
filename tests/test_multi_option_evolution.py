"""Provider-free integration tests for generic multi-option evolution."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from agent_evolve.agentic import AgenticBenchmark
from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    CrossoverResponseMode,
    OperatorKind,
    ProposalAuthority,
    RewardPolicyBinding,
)
from agent_evolve.application.budgeted_optimizer import (
    BudgetedAgenticOptimizer,
)
from agent_evolve.application.effective_choice_audit import (
    EffectiveChoiceAuditReceipt,
    validate_effective_choice_audit_receipt,
)
from agent_evolve.application.generation_feedback import (
    GenerationFeedbackReservation,
    GenerationFeedbackResult,
)
from agent_evolve.application.insight_memory import (
    InsightMemoryBank,
    InsightOrigin,
    context_stratum_hash,
)
from agent_evolve.application.multi_option_evolution import (
    MULTI_OPTION_EVOLUTION_BUDGET,
    MULTI_OPTION_G1_SLOT_IDS,
    MULTI_OPTION_G2_SLOT_IDS,
    MULTI_OPTION_G3_CORE_SLOT_IDS,
    MULTI_OPTION_G3_CROSSOVER_SLOT_IDS,
    MULTI_OPTION_G3_SLOT_IDS,
    MultiOptionEvolutionPlanner,
    MultiOptionEvolutionPlannerFactory,
)
from agent_evolve.application.pareto_archive import (
    EvidenceAdmissionPolicy,
    ParetoArchive,
)
from agent_evolve.application.post_evolution_reflection import (
    PostEvolutionReflectionFactory,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.memory.staged_causal import (
    MemoryAssignmentArm,
    MemoryCheckpointClosureStatus,
)
from agent_evolve.policies.memory.treatment_compliance import (
    TreatmentActionBinding,
)
from agent_evolve.policies.variation.exact_parent_crossover import (
    resolve_exact_parent_import_for_target,
)
from agent_evolve.policies.selection.finite_action import (
    TaskKeyedUniformFiniteActionPolicy,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    CandidateDraft,
    ExactParentCrossoverDraft,
    FiniteVariationSelectionDraft,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)
from agent_evolve.ports.executable_hypothesis import (
    ExecutableHypothesisTestSpec,
    HypothesisApplicabilityStatus,
    HypothesisCompilationReceipt,
    HypothesisCompilationRequest,
)
from agent_evolve.ports.finite_action_set import (
    FiniteActionSetCompilationRequest,
    FiniteActionSetDraft,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    value: int
    mate: int
    context: int


class _Problem:
    candidate_model = _Config
    objectives = (ObjectiveSpec("score", "min"),)

    def __init__(self) -> None:
        self.evaluations: list[tuple[int, int, int]] = []

    @staticmethod
    def search_space_description() -> str:
        return (
            "Choose one local value and optionally combine it with an "
            "orthogonal mate coordinate."
        )

    @staticmethod
    def validate(configuration: object) -> bool:
        candidate = _Config.model_validate(configuration, strict=True)
        return (
            0 <= candidate.value <= 4
            and candidate.mate in {0, 5}
            and candidate.context in {0, 1}
        )

    def evaluate(self, configuration: dict[str, object]) -> dict[str, float]:
        value = int(configuration["value"])
        mate = int(configuration["mate"])
        context = int(configuration["context"])
        self.evaluations.append((value, mate, context))
        return {"score": float(abs(value - 3) + 0.1 * abs(mate - 5) + 0.01 * context)}


class _K4ValueCatalog:
    catalog_id = "fixture_k4_value"
    catalog_version = 1
    definition_sha256 = _sha("fixture K4 value catalog v1")

    def options(
        self,
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent = thaw_json(parent_configuration)
        assert type(parent) is dict
        parent_sha256 = typed_json_sha256(parent_configuration)
        return tuple(
            FiniteVariationOption(
                option_id=f"local.value_p{value}",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json(
                    {
                        **parent,
                        "value": value,
                    }
                ),
                family="local_value",
                description=f"Set the local value coordinate to {value}.",
            )
            for value in (1, 2, 3, 4)
        )


class _MateCatalog:
    catalog_id = "fixture_disjoint_mate"
    catalog_version = 1
    definition_sha256 = _sha("fixture disjoint mate catalog v1")

    def options(
        self,
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent = thaw_json(parent_configuration)
        assert type(parent) is dict
        return (
            FiniteVariationOption(
                option_id="mate.p5",
                parent_configuration_sha256=typed_json_sha256(parent_configuration),
                child_configuration=freeze_json({**parent, "mate": 5, "context": 0}),
                family="orthogonal_mate",
                description=(
                    "Set the disjoint mate coordinate and its independent context."
                ),
            ),
        )


class _ExactAnchorCompiler:
    policy_id = "fixture_multi_option_anchor"
    policy_version = 1
    definition_sha256 = _sha("fixture multi-option exact anchor v1")

    def compile(
        self,
        request: HypothesisCompilationRequest,
    ) -> HypothesisCompilationReceipt:
        option = request.finite_contract.resolve(
            request.insight.recommended_option_ids[0]
        )
        spec = ExecutableHypothesisTestSpec(
            request_sha256=request.request_sha256,
            reference=request.reference,
            insight_content_sha256=request.insight.content_sha256,
            source_evidence_sha256=request.source_evidence_sha256,
            requested_operator_kind=request.requested_operator_kind,
            source_operator_kinds=request.source_operator_kinds,
            executable_operator_kinds=(request.requested_operator_kind,),
            parent_candidate_id=request.parent_candidate_id,
            parent_configuration_sha256=request.parent_configuration_sha256,
            finite_contract_sha256=request.finite_contract.identity_sha256,
            context_projection_sha256=request.context_projection_sha256,
            endpoint_definition_sha256=request.endpoint_definition_sha256,
            allowed_actions=(
                TreatmentActionBinding(option.option_id, option.identity_sha256),
            ),
            recommended_option_families=(option.family,),
            affected_paths=("$.value",),
            held_fixed_paths=("$.context", "$.mate"),
            effect_predictions=request.insight.effect_predictions,
            falsification_condition=str(request.insight.falsification_condition),
            compiler_policy_id=self.policy_id,
            compiler_policy_version=self.policy_version,
            compiler_definition_sha256=self.definition_sha256,
        )
        return HypothesisCompilationReceipt(
            request_sha256=request.request_sha256,
            status=HypothesisApplicabilityStatus.APPLICABLE,
            reason_codes=(),
            compiler_policy_id=self.policy_id,
            compiler_policy_version=self.policy_version,
            compiler_definition_sha256=self.definition_sha256,
            spec=spec,
        )


class _K4SupportCompiler:
    policy_id = "fixture_multi_option_k4_support"
    policy_version = 1
    definition_sha256 = _sha("fixture multi-option K4 support v1")

    def compile(
        self,
        request: FiniteActionSetCompilationRequest,
    ) -> FiniteActionSetDraft:
        option_ids = tuple(
            option.option_id for option in request.finite_contract.options
        )
        assert len(option_ids) == request.required_cardinality == 4
        return FiniteActionSetDraft(
            request_sha256=request.request_sha256,
            ordered_option_ids=option_ids,
            anchor_option_id=request.anchor_option_id,
            presentation_policy_id="fixture_multi_option_presentation",
            presentation_policy_version=1,
            presentation_definition_sha256=_sha("fixture multi-option presentation v1"),
            prompt_shape_sha256=_sha("fixture multi-option prompt shape v1"),
        )


@dataclass(frozen=True, slots=True)
class _MateChoice:
    catalog_id: str
    parent_configuration_sha256: str
    finite_contract_sha256: str
    option_id: str
    option_identity_sha256: str
    choice_sha256: str

    @classmethod
    def seal(
        cls,
        contract: FiniteVariationContract,
        option_id: str,
    ) -> "_MateChoice":
        option = contract.resolve(option_id)
        record = {
            "catalog_id": contract.catalog_id,
            "parent_configuration_sha256": (contract.parent_configuration_sha256),
            "finite_contract_sha256": contract.identity_sha256,
            "option_id": option.option_id,
            "option_identity_sha256": option.identity_sha256,
        }
        return cls(**record, choice_sha256=_sha(json.dumps(record, sort_keys=True)))

    def validate_contract(self, contract: FiniteVariationContract) -> None:
        option = contract.resolve(self.option_id)
        if (
            contract.catalog_id,
            contract.parent_configuration_sha256,
            contract.identity_sha256,
            option.identity_sha256,
        ) != (
            self.catalog_id,
            self.parent_configuration_sha256,
            self.finite_contract_sha256,
            self.option_identity_sha256,
        ):
            raise ValueError("mate choice differs from its parent-bound contract")


def _reward() -> RewardPolicyBinding:
    def score(child, parents, objectives):
        del objectives
        if not child.valid or not child.operator_compliant:
            return -1.0
        return parents[0].objective_map["score"] - child.objective_map["score"]

    return RewardPolicyBinding(score, _sha("fixture multi-option reward v1"))


def _telemetry(kind: str) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fixture",
        provider_response_id=f"fixture-multi-option-{kind}",
        finish_reason="stop",
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
        attempt_count=1,
    )


class _Generator:
    def __init__(self) -> None:
        self.planner: MultiOptionEvolutionPlanner | None = None
        self.anchor_by_reference: dict[str, str] = {}
        self.proposal_requests = []
        self.reflection_requests = []

    def _assigned_reference(self, prompt: str) -> str:
        matches = tuple(
            reference for reference in self.anchor_by_reference if reference in prompt
        )
        assert len(matches) == 1
        return matches[0]

    async def propose(self, request):
        self.proposal_requests.append(request)
        contract = request.finite_variation_contract
        if contract is not None:
            reference = self._assigned_reference(request.prompt)
            is_diagnostic = '"context":0' in request.prompt
            planner = self.planner
            assert planner is not None
            if is_diagnostic:
                option_id = self.anchor_by_reference[reference]
            elif reference == planner.adaptive_reference.insight_id.value:
                assert planner.uniform_decision is not None
                option_id = planner.uniform_decision.option_id
            else:
                option_id = "local.value_p2"
            option = contract.resolve(option_id)
            return VariationGenerationResult(
                draft=FiniteVariationSelectionDraft(
                    option_id=option.option_id,
                    option_identity_sha256=option.identity_sha256,
                    contract_identity_sha256=contract.identity_sha256,
                    design_rationale=(
                        "Choose one genuine option from the authenticated local support."
                    ),
                    claimed_insight_ids=(reference,),
                ),
                telemetry=_telemetry("finite"),
            )

        crossover_contract = request.exact_parent_crossover_contract
        if crossover_contract is not None:
            prompt_contract = json.loads(
                request.prompt.split(
                    "EXACT PARENT IMPORT CONTRACT\n",
                    1,
                )[1].split("\n", 1)[0]
            )
            assert prompt_contract["forbidden_import_locus_sets"] == [
                list(value) for value in crossover_contract.forbidden_import_locus_sets
            ]
            selected = next(
                (locus_id,)
                for locus_id in crossover_contract.locus_ids
                if (locus_id,) not in crossover_contract.forbidden_import_locus_sets
            )
            return VariationGenerationResult(
                draft=ExactParentCrossoverDraft(
                    contract_identity_sha256=(
                        crossover_contract.contract_identity_sha256
                    ),
                    import_locus_ids=selected,
                ),
                telemetry=_telemetry("crossover"),
            )

        # Legacy full-configuration crossover remains covered independently.
        parent_rows = json.loads(
            request.prompt.split("PARENTS\n", 1)[1].split("\n", 1)[0]
        )
        left = parent_rows[0]["configuration"]
        right = parent_rows[1]["configuration"]
        child = {
            "value": left["value"],
            "mate": right["mate"],
            "context": left["context"],
        }
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=child,
                design_rationale=(
                    "Cross the selected local action with the orthogonal mate."
                ),
                intended_changes=("$.mate", "$.value"),
                source_attribution=(
                    SourceAttribution("$.value", "left"),
                    SourceAttribution("$.mate", "right"),
                ),
            ),
            telemetry=_telemetry("crossover"),
        )

    async def reflect(self, request):
        self.reflection_requests.append(request)
        return ReflectionGenerationResult(
            insights=(),
            telemetry=_telemetry("reflection"),
        )


class _TerminalReflection:
    def __init__(self, engine: AgenticEvolutionEngine) -> None:
        self.engine = engine
        self.contexts = []

    def reserve(self, *, state, plan):
        del state
        return GenerationFeedbackReservation(
            policy_id="fixture_terminal_multi_option_reflection",
            policy_version=1,
            logical_llm_calls=int(plan.generation == 3),
            metadata=(("phase", f"generation_{plan.generation}"),),
        )

    async def after_generation(self, context):
        self.contexts.append(context)
        used = context.reservation.logical_llm_calls
        if used:
            await self.engine.reflect(
                tuple(
                    value.outcome for value in context.generation_receipt.slot_results
                ),
                label="multi_option_terminal_reflection",
                max_insights=1,
                min_insights=0,
                source_receipt_sha256s=(context.generation_receipt.receipt_hash,),
            )
        return GenerationFeedbackResult(
            logical_llm_calls_used=used,
            metadata=(("reflection", "terminal" if used else "deferred"),),
        )


@dataclass(slots=True)
class _Fixture:
    problem: _Problem
    generator: _Generator
    planner: MultiOptionEvolutionPlanner
    optimizer: BudgetedAgenticOptimizer
    feedback: _TerminalReflection
    active_references: tuple


def _fixture() -> _Fixture:
    ids = DeterministicIdFactory("multi_option_evolution")
    memory = InsightMemoryBank(id_factory=ids)
    entries = []
    for ordinal, option_id in ((1, "local.value_p1"), (4, "local.value_p4")):
        entry, added = memory.add(
            InsightDraft(
                claim=f"Local value intervention {ordinal} may improve the score.",
                trigger="The parent exposes the bounded local value coordinate.",
                mechanism="Move the value coordinate toward the objective target.",
                affected_paths=("$.value",),
                evidence_summary="Seeded provider-free diagnostic hypothesis.",
                confidence=0.5,
                effect_predictions=(
                    MetricEffectPrediction(
                        metric_id="objective:score",
                        direction=MetricEffectDirection.DECREASE,
                    ),
                ),
                recommended_option_families=("local_value",),
                recommended_option_ids=(option_id,),
                action_template="Choose a bounded local value action.",
                falsification_condition="The score does not improve.",
            ),
            initial_score=0.0,
            applicable_operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
            origin=InsightOrigin.SEED,
        )
        assert added
        entries.append(entry)
    active_references = tuple(sorted(value.reference for value in entries))
    problem = _Problem()
    reward = _reward()
    benchmark = AgenticBenchmark(
        problem=problem,
        reward=reward,
        finite_variation_catalogs=(_K4ValueCatalog(), _MateCatalog()),
        hypothesis_compiler=_ExactAnchorCompiler(),
        finite_action_set_compiler=_K4SupportCompiler(),
    )
    evolution_seed = {"value": 0, "mate": 0, "context": 1}
    mate_contract = benchmark.bind_finite_variation(
        _MateCatalog.catalog_id,
        evolution_seed,
    )
    mate_choice = _MateChoice.seal(mate_contract, "mate.p5")
    generator = _Generator()
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=17,
        evaluator_concurrency=8,
        reward_policy=reward.score,
        reward_definition_hash=reward.definition_hash,
    )
    phase = "multi_option_fixture"
    context_projection_sha256 = context_stratum_hash(
        problem_id=engine.problem_id,
        operator_kind=OperatorKind.TYPED_MUTATION.value,
        phase=phase,
    )
    factory = MultiOptionEvolutionPlannerFactory(
        reward_binding=reward,
        active_references=active_references,
        model_catalog_id=_K4ValueCatalog.catalog_id,
        mate_catalog_id=_MateCatalog.catalog_id,
        mate_choice=mate_choice,
        required_cardinality=4,
        uniform_policy=TaskKeyedUniformFiniteActionPolicy(
            schedule_seed_sha256=_sha("multi-option prospective schedule")
        ),
        task_sha256=_sha("multi-option task"),
        pre_outcome_phase_commit_sha256=_sha("multi-option phase commit"),
        endpoint_definition_sha256=reward.definition_hash,
        context_projection_sha256=context_projection_sha256,
        estimand_stratum_sha256=_sha("multi-option estimand"),
        phase=phase,
    )
    planner = factory.build(
        benchmark=benchmark,
        engine=engine,
        id_factory=ids,
        memory=memory,
    )
    generator.planner = planner
    generator.anchor_by_reference = {
        entry.reference.insight_id.value: entry.draft.recommended_option_ids[0]
        for entry in entries
    }
    feedback = _TerminalReflection(engine)
    optimizer = BudgetedAgenticOptimizer(
        engine=engine,
        archive=ParetoArchive(
            problem.objectives,
            evidence_admission_policy=EvidenceAdmissionPolicy.RECORD_ONLY,
        ),
        planner=planner,
        budget=MULTI_OPTION_EVOLUTION_BUDGET,
        feedback_interceptor=feedback,
    )
    return _Fixture(
        problem,
        generator,
        planner,
        optimizer,
        feedback,
        active_references,
    )


def test_full_multi_option_evolution_runs_through_reflection() -> None:
    fixture = _fixture()
    result = asyncio.run(
        fixture.optimizer.run(
            (
                {"value": 0, "mate": 0, "context": 0},
                {"value": 0, "mate": 0, "context": 1},
            )
        )
    )
    planner = fixture.planner

    assert result.final_state.generation == 3
    assert len(result.final_state.candidates) == 14
    assert result.final_state.unique_evaluations == 11
    assert result.final_state.logical_llm_calls == 7
    assert fixture.planner.crossover_policy.policy_version == 3
    assert len(fixture.generator.proposal_requests) == 6
    assert len(fixture.generator.reflection_requests) == 1
    assert tuple(
        tuple(value.slot.slot_id for value in receipt.slot_results)
        for receipt in result.generation_receipts
    ) == (
        MULTI_OPTION_G1_SLOT_IDS,
        MULTI_OPTION_G2_SLOT_IDS,
        MULTI_OPTION_G3_SLOT_IDS,
    )
    crossover_requests = tuple(
        request
        for request in fixture.generator.proposal_requests
        if request.operation == OperatorKind.TWO_PARENT_CROSSOVER.value
    )
    assert len(crossover_requests) == 2
    assert all(
        request.exact_parent_crossover_contract is not None
        and request.atomic_mutation_contract is None
        and request.finite_variation_contract is None
        for request in crossover_requests
    )
    assert all(
        slot_result.slot.plan is not None
        and slot_result.slot.plan.crossover_response_mode
        is CrossoverResponseMode.EXACT_PARENT_IMPORT_V1
        and slot_result.slot.plan.exact_parent_crossover_contract is not None
        for slot_result in result.generation_receipts[2].slot_results[4:]
    )
    known_targets = tuple(
        candidate.configuration
        for candidate in result.final_state.candidates
        if candidate.generation < 3
    ) + tuple(
        slot_result.outcome.candidate.configuration
        for slot_result in result.generation_receipts[2].slot_results[:4]
        if slot_result.outcome.candidate is not None
    )
    for slot_result in result.generation_receipts[2].slot_results[4:]:
        plan = slot_result.slot.plan
        assert plan is not None
        contract = plan.exact_parent_crossover_contract
        assert contract is not None
        expected_forbidden = tuple(
            sorted(
                {
                    resolved
                    for target in known_targets
                    if (
                        resolved := resolve_exact_parent_import_for_target(
                            base=plan.parents[0].configuration,
                            donor=plan.parents[1].configuration,
                            contract=contract,
                            target=target,
                        )
                    )
                    is not None
                }
            )
        )
        assert plan.forbidden_exact_parent_import_sets == expected_forbidden
        assert (
            slot_result.outcome.candidate is not None
            and slot_result.outcome.candidate.occurrence.configuration_hash
            not in {typed_json_sha256(target) for target in known_targets}
        )
    assert tuple(
        slot_result.slot.role
        for slot_result in result.generation_receipts[2].slot_results[4:]
    ) == (
        "model_selected_exact_parent_crossover",
        "model_selected_exact_parent_crossover",
    )
    assert tuple(
        value.outcome.prepared.proposal_authority
        for value in result.generation_receipts[2].slot_results
    ) == (
        ProposalAuthority.REPRODUCTION,
        ProposalAuthority.ENGINE,
        ProposalAuthority.ENGINE,
        ProposalAuthority.ENGINE,
        ProposalAuthority.MODEL,
        ProposalAuthority.MODEL,
    )
    assert all(
        value.outcome.failure_stage is None
        and value.outcome.candidate is not None
        and value.outcome.candidate.valid
        and value.outcome.candidate.operator_compliant
        for receipt in result.generation_receipts
        for value in receipt.slot_results
    )
    assert planner.wave is not None
    assert planner.closure is not None
    assert planner.closure.status is MemoryCheckpointClosureStatus.SEALED
    assert tuple(value.arm for value in planner.g1_assignments) == (
        MemoryAssignmentArm.DIAGNOSTIC,
        MemoryAssignmentArm.DIAGNOSTIC,
    )
    assert tuple(value.arm for value in planner.g2_assignments) == (
        MemoryAssignmentArm.ADAPTIVE,
        MemoryAssignmentArm.SCORE_SHUFFLED_CONTROL,
    )
    assert planner.adaptive_reference == fixture.active_references[1]
    assert planner.terminal_slot_ids == MULTI_OPTION_G3_SLOT_IDS
    assert len(planner.g3_union_materialization_receipt_sha256s) == 3
    audit_ledger = planner.effective_choice_audit_receipts
    assert tuple(audit_ledger) == (
        (1, MULTI_OPTION_G1_SLOT_IDS[0]),
        (1, MULTI_OPTION_G1_SLOT_IDS[1]),
        (2, MULTI_OPTION_G2_SLOT_IDS[0]),
        (2, MULTI_OPTION_G2_SLOT_IDS[1]),
    )
    assert all(
        type(receipt) is EffectiveChoiceAuditReceipt
        and receipt.effective_cardinality == 4
        and receipt.configured_minimum_cardinality == 4
        for receipt in audit_ledger.values()
    )
    plans_by_coordinate = {
        (receipt.generation, slot_result.slot.slot_id): slot_result.slot.plan
        for receipt in result.generation_receipts[:2]
        for slot_result in receipt.slot_results
        if slot_result.slot.plan is not None
        and slot_result.slot.plan.finite_action_set_authority is not None
    }
    assert tuple(plans_by_coordinate) == tuple(audit_ledger)
    for coordinate, receipt in audit_ledger.items():
        plan = plans_by_coordinate[coordinate]
        assert plan is not None
        validate_effective_choice_audit_receipt(receipt, plan)
    assert tuple(
        value.used_logical_llm_calls for value in result.feedback_receipts
    ) == (
        0,
        0,
        1,
    )
    assert tuple(
        (
            value.logical_llm_calls_before,
            value.logical_llm_calls_after,
            value.unique_evaluations_before,
            value.unique_evaluations_after,
            value.reserved_logical_llm_calls,
            value.reserved_unique_evaluations,
        )
        for value in result.generation_receipts
    ) == (
        (0, 2, 2, 4, 2, 2),
        (2, 4, 4, 7, 2, 4),
        (4, 6, 7, 11, 2, 5),
    )
    # Per GENERATION, exactly these evaluations -- and deliberately not their
    # interleave. The planner gathers the direct and engine slot groups
    # concurrently and the engine gathers its per-item coroutines, then BOTH
    # re-order outcomes by slot id before anything downstream reads them --
    # arrival order is scheduler behaviour, not a contract, and CPython 3.13
    # interleaves the two gathered groups differently than 3.11/3.12 did.
    # The exact-order form of this assertion was the "check on something
    # adjacent to the claim" pattern docs/scope.md names: it failed on a
    # green mechanism the first time the interpreter's scheduler moved.
    # (2026-08-24; the generation boundaries come from the receipts pinned
    # just above, and duplicates would break the multiset equality.)
    boundaries = (2, 4, 7, 11)              # seeds, then +2, +3, +4 per receipt
    expected_by_generation = (
        [(0, 0, 0), (0, 0, 1)],
        [(1, 0, 0), (4, 0, 0)],
        [(3, 0, 1), (2, 0, 1), (0, 5, 0)],
        [(3, 5, 1), (2, 5, 1), (3, 5, 0), (2, 5, 0)],
    )
    assert len(fixture.problem.evaluations) == boundaries[-1]
    start = 0
    for end, expected in zip(boundaries, expected_by_generation, strict=True):
        assert sorted(fixture.problem.evaluations[start:end]) == sorted(
            expected
        ), f"generation slice [{start}:{end}]"
        start = end


def test_same_support_alias_is_retained_while_crossover_excludes_known_union() -> None:
    fixture = _fixture()
    result = asyncio.run(
        fixture.optimizer.run(
            (
                {"value": 0, "mate": 0, "context": 0},
                {"value": 0, "mate": 0, "context": 1},
            )
        )
    )
    g2 = result.generation_receipts[1]
    adaptive = g2.slot_results[0].outcome.candidate
    uniform = g2.slot_results[2].outcome.candidate
    assert adaptive is not None and uniform is not None
    assert adaptive.candidate_id != uniform.candidate_id
    assert adaptive.occurrence.configuration_hash == (
        uniform.occurrence.configuration_hash
    )
    model_decision = g2.slot_results[0].outcome.finite_action_decision
    assert model_decision is not None
    assert fixture.planner.uniform_decision is not None
    assert model_decision.option_id == fixture.planner.uniform_decision.option_id
    assert model_decision.support_sha256 == (
        fixture.planner.uniform_decision.support_sha256
    )

    g3 = result.generation_receipts[2]
    adaptive_union = g3.slot_results[1].outcome.candidate
    uniform_union = g3.slot_results[3].outcome.candidate
    adaptive_cross = g3.slot_results[4].outcome.candidate
    assert (
        adaptive_union is not None
        and uniform_union is not None
        and adaptive_cross is not None
    )
    assert (
        len(
            {
                adaptive_union.candidate_id,
                uniform_union.candidate_id,
                adaptive_cross.candidate_id,
            }
        )
        == 3
    )
    assert adaptive_union.occurrence.configuration_hash == (
        uniform_union.occurrence.configuration_hash
    )
    assert adaptive_cross.occurrence.configuration_hash != (
        adaptive_union.occurrence.configuration_hash
    )
    assert adaptive_union.configuration_dict["mate"] == 5
    assert (
        tuple(value.slot.slot_id for value in g3.slot_results[:4])
        == MULTI_OPTION_G3_CORE_SLOT_IDS
    )
    assert (
        tuple(value.slot.slot_id for value in g3.slot_results[4:])
        == MULTI_OPTION_G3_CROSSOVER_SLOT_IDS
    )
    assert result.final_state.unique_evaluations < 13


def test_public_facade_exports_multi_option_planner() -> None:
    from agent_evolve import (
        EffectiveChoiceAuditReceipt as PackageAuditReceipt,
        MultiOptionEvolutionPlanner as PackagePlanner,
        MultiOptionEvolutionPlannerFactory as PackageFactory,
        PostEvolutionReflectionFactory as PackageReflectionFactory,
        audit_effective_choice_plan as package_audit_effective_choice_plan,
    )
    from agent_evolve.agentic import (
        EffectiveChoiceAuditReceipt as FacadeAuditReceipt,
        MultiOptionEvolutionPlanner as FacadePlanner,
        MultiOptionEvolutionPlannerFactory as FacadeFactory,
        PostEvolutionReflectionFactory as FacadeReflectionFactory,
        audit_effective_choice_plan as facade_audit_effective_choice_plan,
    )
    from agent_evolve.application.effective_choice_audit import (
        audit_effective_choice_plan,
    )

    assert PackagePlanner is MultiOptionEvolutionPlanner is FacadePlanner
    assert PackageFactory is MultiOptionEvolutionPlannerFactory is FacadeFactory
    assert PackageAuditReceipt is EffectiveChoiceAuditReceipt is FacadeAuditReceipt
    assert (
        PackageReflectionFactory
        is PostEvolutionReflectionFactory
        is FacadeReflectionFactory
    )
    assert (
        package_audit_effective_choice_plan
        is audit_effective_choice_plan
        is facade_audit_effective_choice_plan
    )
