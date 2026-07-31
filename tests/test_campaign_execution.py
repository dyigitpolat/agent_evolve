from __future__ import annotations

import asyncio
import hashlib
from functools import wraps

import pytest

from agent_evolve.application.budgeted_optimizer import OptimizerBudget
from agent_evolve.application.campaign_execution import (
    CampaignArchiveCutoffReceipt,
    CampaignCleanupReceipt,
    CampaignExecutionContractError,
    CampaignExecutionEventKind,
    CampaignExecutionStartReceipt,
    CampaignExecutionStatus,
    CampaignFinalizationReceipt,
    CampaignJournalAck,
    CampaignReflectionReceipt,
    CampaignReflectionStatus,
    CampaignReflectionTestAdmissionReceipt,
    CampaignSeedExecutionReceipt,
    CampaignSelectorAuditReceipt,
    CampaignStageReceipt,
    EvolutionCampaignScheduler,
    SelectorAuditExecutionMode,
    decode_selector_audit_text,
    encode_selector_audit_text,
)
from agent_evolve.application.evolution_campaign import (
    AlternatingPortfolioRecombinationCadence,
    ArchiveUtilitySnapshot,
    BenchmarkSessionRequest,
    CampaignAgentRuntimeReceipt,
    CampaignBenchmarkSession,
    CampaignConcurrency,
    CampaignGenerationKind,
    CampaignPolicies,
    CampaignPolicyBinding,
    CampaignProtocol,
    CampaignReflectionSupervisionPolicy,
    CampaignSeed,
    CampaignSeedBatch,
    CampaignWorkloadPorts,
    EvolutionCampaign,
    ReflectionFailureMode,
    SealedCutoffDelayedAdmissionCadence,
    TerminalReflectionPolicy,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


def test_large_selector_audit_text_is_losslessly_authenticated_in_chunks() -> None:
    response_text = '{"trace":"' + ("forecast-evidence," * 80_000) + '"}'
    payload = {
        "selector_call_id": "call_large_audit",
        "request_sha256": _sha("large request"),
        "decision_sha256": _sha("large decision"),
        "request_text": "bounded request",
        **encode_selector_audit_text("response_text", response_text),
        "request_text_kind": "exact_framework_prompt",
        "response_text_kind": "trusted_structured_decision_projection",
    }
    plaintext = _object(payload)

    receipt = CampaignSelectorAuditReceipt(
        generation=1,
        parent_slot=0,
        selector_call_id="call_large_audit",
        request_sha256=_sha("large request"),
        decision_sha256=_sha("large decision"),
        trace_receipt_sha256=typed_json_sha256(plaintext),
        plaintext_audit=plaintext,
        prior_audit_set_sha256=_sha("prior large audit set"),
        execution_mode=SelectorAuditExecutionMode.FRESH,
    )

    thawed = thaw_json(receipt.plaintext_audit)
    assert "response_text" not in thawed
    assert decode_selector_audit_text(thawed, "response_text") == response_text


class _ArchiveUtility:
    utility_id = "test_archive_utility"
    utility_version = 1
    definition_sha256 = _sha("test-archive-utility")

    def freeze(self, *, benchmark, generation, archive):
        return ArchiveUtilitySnapshot(
            utility_id=self.utility_id,
            utility_version=self.utility_version,
            definition_sha256=self.definition_sha256,
            generation=generation,
            benchmark_sha256=typed_json_sha256(benchmark),
            archive_sha256=typed_json_sha256(archive),
            snapshot_receipt=_object(
                {"reference_key": "fixed", "generation": generation}
            ),
        )


def _binding(name: str) -> CampaignPolicyBinding:
    return CampaignPolicyBinding(
        implementation=object(),
        policy_id=name,
        policy_version=1,
        definition_sha256=_sha(f"policy:{name}"),
    )


def _policies(
    failure_mode: ReflectionFailureMode = (
        ReflectionFailureMode.COLLECT_ALL_AT_BARRIER_THEN_FAIL
    ),
    *,
    cadence=None,
) -> CampaignPolicies:
    return CampaignPolicies(
        cadence=(
            AlternatingPortfolioRecombinationCadence()
            if cadence is None
            else cadence
        ),
        parent_selection=_binding("front_first"),
        memory_assignment=_binding("balanced_memory"),
        portfolio_selection=_binding("ranked_portfolio"),
        recombination=_binding("exact_recombination"),
        reflection=_binding("quarantined_reflection"),
        archive_utility=_ArchiveUtility(),
        reflection_supervision=CampaignReflectionSupervisionPolicy(failure_mode),
    )


class _BenchmarkPort:
    port_id = "test_benchmark"
    port_version = 1
    definition_sha256 = _sha("test-benchmark")

    def open(self, request: BenchmarkSessionRequest):
        return CampaignBenchmarkSession(
            request_sha256=request.request_sha256,
            benchmark=_object({"benchmark_key": "opaque"}),
            evaluator_concurrency_cap=8,
            preflight_receipt=_object({"qualified": True}),
            resource_lease=_object({"lease": "held"}),
        )


class _SeedPort:
    port_id = "test_seeds"
    port_version = 1
    definition_sha256 = _sha("test-seeds")

    def load(self, session):
        return CampaignSeedBatch(
            session_sha256=session.session_sha256,
            seeds=(
                CampaignSeed("seed_a", _object({"x": 0})),
                CampaignSeed("seed_b", _object({"x": 1})),
            ),
        )


class _CatalogPort:
    port_id = "test_catalog"
    port_version = 1
    definition_sha256 = _sha("test-catalog")

    def bind(self, benchmark, parent, known_phenotype_sha256s):
        del benchmark, parent, known_phenotype_sha256s
        raise AssertionError("campaign preparation must not bind a parent catalog")


class _EvidencePort:
    port_id = "test_evidence"
    port_version = 1
    definition_sha256 = _sha("test-evidence")

    def initialize_memory(self, session, seeds):
        del session, seeds
        return _object({})

    def context(self, session, parent, variation, memory):
        del session, parent, variation, memory
        return _object({})

    def cards(self, session, parent, variation, memory):
        del session, parent, variation, memory
        return (_object({"card": "opaque"}),)


class _PreparationRuntime:
    def prepare(self, request):
        return CampaignAgentRuntimeReceipt(
            request_sha256=request.request_sha256,
            runtime_id="test_preparation_runtime",
            runtime_version=1,
            definition_sha256=_sha("test-preparation-runtime"),
            accepted=True,
            evidence=_object({"accepted": True}),
        )


class _PreparationJournal:
    def append(self, record):
        assert type(record) is FrozenJsonObject


def _prepared(
    generations: int,
    *,
    failure_mode: ReflectionFailureMode = (
        ReflectionFailureMode.COLLECT_ALL_AT_BARRIER_THEN_FAIL
    ),
    delayed_admission: bool = False,
):
    cadence = (
        SealedCutoffDelayedAdmissionCadence()
        if delayed_admission
        else AlternatingPortfolioRecombinationCadence()
    )
    policies = _policies(failure_mode, cadence=cadence)
    protocol = CampaignProtocol(
        protocol_id="execution_test",
        protocol_version=1,
        definition_sha256=_sha("execution-test-protocol"),
        outer_seed=17,
        generation_count=generations,
        required_seed_count=2,
        parents_per_portfolio_generation=2,
        portfolio_width=4,
        recombinations_per_parent=2,
        reflections_per_recombination_generation=1,
        reflection_promotion_block_pairs=(1 if delayed_admission else 3),
        terminal_reflection_policy=(
            TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER
            if delayed_admission
            else TerminalReflectionPolicy.ALLOW_TERMINAL
        ),
    )
    schedule = policies.cadence.build(protocol)
    budget = OptimizerBudget(
        max_unique_evaluations=(
            protocol.required_seed_count + schedule.planned_candidate_evaluations
        ),
        max_logical_llm_calls=schedule.planned_agent_calls,
        max_generations=generations,
    )
    campaign = EvolutionCampaign(
        protocol=protocol,
        workload=CampaignWorkloadPorts(
            benchmark=_BenchmarkPort(),
            seeds=_SeedPort(),
            catalog=_CatalogPort(),
            evidence=_EvidencePort(),
        ),
        policies=policies,
        runtime=_PreparationRuntime(),
        budget=budget,
        concurrency=CampaignConcurrency(4, 2, 8),
        journals=(_PreparationJournal(),),
    )
    return campaign.prepare(), policies


class _DurableJournal:
    def __init__(self, *, fail_kind=None) -> None:
        self.fail_kind = fail_kind
        self.events = []

    async def append(self, event):
        if event.kind is self.fail_kind:
            raise OSError("injected durable journal failure")
        if self.events:
            assert event.sequence == len(self.events) + 1
            assert event.previous_event_sha256 == self.events[-1].event_sha256
        else:
            assert event.sequence == 1
            assert event.previous_event_sha256 is None
        self.events.append(event)
        return CampaignJournalAck(event.event_sha256, True)


class _Lifecycle:
    def __init__(
        self,
        *,
        seed_failure: bool = False,
        finalize_failure: bool = False,
        foreign_start: bool = False,
    ) -> None:
        self.seed_failure = seed_failure
        self.finalize_failure = finalize_failure
        self.foreign_start = foreign_start
        self.start_receipt = None
        self.finalization_requests = []
        self.cleanup_requests = []

    async def start(self, prepared):
        self.start_receipt = CampaignExecutionStartReceipt(
            preparation_sha256=(
                "f" * 64 if self.foreign_start else prepared.preparation_sha256
            ),
            runtime_preflight_receipt_sha256=(prepared.runtime_receipt.receipt_sha256),
            runtime_session_id="execution_session",
            seed_batch_sha256=prepared.seeds.batch_sha256,
            seed_receipts=tuple(
                CampaignSeedExecutionReceipt(
                    seed_id=seed.seed_id,
                    configuration_sha256=seed.configuration_sha256,
                    evaluated=True,
                    unique_evaluation=True,
                    valid=not (self.seed_failure and index == 0),
                    failure_type=(
                        "seed_invalid" if self.seed_failure and index == 0 else None
                    ),
                    evidence=_object(
                        {"admitted": not (self.seed_failure and index == 0)}
                    ),
                )
                for index, seed in enumerate(prepared.seeds.seeds)
            ),
            evidence=_object({"opened": True}),
        )
        return self.start_receipt

    async def finalize(self, request):
        self.finalization_requests.append(request)
        if self.finalize_failure:
            raise RuntimeError("injected finalization failure")
        return CampaignFinalizationReceipt(
            request_sha256=request.request_sha256,
            preparation_sha256=request.preparation_sha256,
            status=request.status,
            evidence=_object({"sealed": True}),
        )

    async def cleanup(self, request):
        self.cleanup_requests.append(request)
        return CampaignCleanupReceipt(
            request_sha256=request.request_sha256,
            preparation_sha256=request.preparation_sha256,
            released=True,
            evidence=_object({"released": True}),
        )


class _StageRuntime:
    def __init__(
        self,
        *,
        journal: _DurableJournal | None = None,
        fault_generation: int | None = None,
        receipt_fault: str | None = None,
        duplicate_audit: bool = False,
        overlap_started: asyncio.Event | None = None,
        overlap_release: asyncio.Event | None = None,
        overlap_observe_generations: tuple[int, ...] = (3,),
        overlap_release_generation: int = 3,
    ) -> None:
        self.journal = journal
        self.fault_generation = fault_generation
        self.receipt_fault = receipt_fault
        self.duplicate_audit = duplicate_audit
        self.overlap_started = overlap_started
        self.overlap_release = overlap_release
        self.overlap_observe_generations = overlap_observe_generations
        self.overlap_release_generation = overlap_release_generation
        self.archive_requests = []
        self.stage_requests = []
        self.overlap_observed = False
        self.overlap_observed_generations: list[int] = []

    async def snapshot_archive(self, request):
        self.archive_requests.append(request)
        return CampaignArchiveCutoffReceipt(
            request_sha256=request.request_sha256,
            preparation_sha256=request.preparation_sha256,
            generation=request.generation,
            archive=_object(
                {
                    "cutoff_generation": request.generation,
                    "prior_stage": request.prior_stage_receipt_sha256,
                }
            ),
            evidence=_object({"frozen_before_stage": True}),
        )

    async def execute_stage(self, request):
        self.stage_requests.append(request)
        generation = request.step.generation
        if self.fault_generation == generation:
            raise RuntimeError("injected stage failure")
        if (
            generation in self.overlap_observe_generations
            and self.overlap_started is not None
        ):
            await self.overlap_started.wait()
            self.overlap_observed = True
            self.overlap_observed_generations.append(generation)
            if generation == self.overlap_release_generation:
                assert self.overlap_release is not None
                self.overlap_release.set()
        if generation > 1 and self.journal is not None:
            assert any(
                event.kind is CampaignExecutionEventKind.STAGE_SEALED
                and thaw_json(event.payload)["stage_receipt"]["generation"]
                == generation - 1
                for event in self.journal.events
            )

        audits = ()
        if request.step.kind is CampaignGenerationKind.PORTFOLIO:
            values = []
            for slot in range(request.step.parent_count):
                effective_slot = 0 if self.duplicate_audit else slot
                selector_call_id = f"selector_{generation}_{effective_slot}"
                request_sha256 = _sha(f"selector-request:{generation}:{effective_slot}")
                decision_sha256 = _sha(
                    f"selector-decision:{generation}:{effective_slot}"
                )
                plaintext_audit = _object(
                    {
                        "selector_call_id": selector_call_id,
                        "request_sha256": request_sha256,
                        "decision_sha256": decision_sha256,
                        "request_text": f"request generation {generation} slot {slot}",
                        "response_text": f"response generation {generation} slot {slot}",
                    }
                )
                values.append(
                    CampaignSelectorAuditReceipt(
                        generation=generation,
                        parent_slot=slot,
                        selector_call_id=selector_call_id,
                        request_sha256=request_sha256,
                        decision_sha256=decision_sha256,
                        trace_receipt_sha256=typed_json_sha256(plaintext_audit),
                        plaintext_audit=plaintext_audit,
                        prior_audit_set_sha256=(
                            request.prior_selector_audit_set_sha256
                        ),
                        execution_mode=SelectorAuditExecutionMode.FRESH,
                    )
                )
            audits = tuple(values)

        receipt_generation = (
            generation + 1 if self.receipt_fault == "out_of_order" else generation
        )
        receipt_request_sha256 = (
            "f" * 64
            if self.receipt_fault == "foreign_request"
            else request.request_sha256
        )
        return CampaignStageReceipt(
            request_sha256=receipt_request_sha256,
            preparation_sha256=request.preparation_sha256,
            generation=receipt_generation,
            kind=request.step.kind,
            candidate_occurrence_count=(request.step.planned_candidate_evaluations),
            unique_evaluation_count=(request.step.planned_candidate_evaluations),
            selector_audits=audits,
            result=_object(
                {
                    "generation": generation,
                    "source_portfolio": (
                        None
                        if request.source_portfolio is None
                        else request.source_portfolio.receipt_sha256
                    ),
                    "test_eligible_reflections": list(
                        request.test_eligible_reflection_receipt_sha256s
                    ),
                }
            ),
        )


class _ReflectionRuntime:
    def __init__(
        self,
        *,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
        fail_source: int | None = None,
        abstain_source: int | None = None,
    ) -> None:
        self.started = started
        self.release = release
        self.fail_source = fail_source
        self.abstain_source = abstain_source
        self.requests = []
        self.test_admission_requests = []
        self.finished = False

    async def reflect(self, request):
        self.requests.append(request)
        if self.started is not None and request.wave.source_generation == 2:
            self.started.set()
            assert self.release is not None
            await self.release.wait()
        if request.wave.source_generation == self.fail_source:
            raise RuntimeError("injected asynchronous reflection failure")
        if request.wave.source_generation == self.abstain_source:
            return CampaignReflectionReceipt(
                request_sha256=request.request_sha256,
                preparation_sha256=request.preparation_sha256,
                source_generation=request.wave.source_generation,
                source_stage_receipt_sha256=request.source_stage.receipt_sha256,
                logical_agent_calls=request.wave.call_count,
                visibility=request.wave.visibility,
                status=CampaignReflectionStatus.ABSTAINED,
                failure_type=None,
                quarantined_result=_object(
                    {
                        "schema_version": 1,
                        "status": (
                            "abstained_no_identifiable_mutation_evidence"
                        ),
                        "evidence_tier": "e0",
                        "provider_calls": 0,
                        "publishable_reflection_content": False,
                    }
                ),
            )
        self.finished = True
        return CampaignReflectionReceipt(
            request_sha256=request.request_sha256,
            preparation_sha256=request.preparation_sha256,
            source_generation=request.wave.source_generation,
            source_stage_receipt_sha256=request.source_stage.receipt_sha256,
            logical_agent_calls=request.wave.call_count,
            visibility=request.wave.visibility,
            status=CampaignReflectionStatus.COMPLETED,
            failure_type=None,
            quarantined_result=_object(
                {"source_generation": request.wave.source_generation}
            ),
        )

    async def admit_for_testing(self, request):
        self.test_admission_requests.append(request)
        admitted = tuple(sorted(value.receipt_sha256 for value in request.reflections))
        eligible = tuple(
            sorted(
                (
                    *request.previously_test_eligible_reflection_receipt_sha256s,
                    *admitted,
                )
            )
        )
        return CampaignReflectionTestAdmissionReceipt(
            request_sha256=request.request_sha256,
            preparation_sha256=request.preparation_sha256,
            barrier_generation=request.barrier.generation,
            admitted_reflection_receipt_sha256s=admitted,
            test_eligible_reflection_receipt_sha256s=eligible,
            lifecycle_promoted=False,
            evidence=_object({"admitted_for_controlled_testing": True}),
        )


@_async_test
async def test_three_generation_execution_overlaps_tail_reflection_without_admission():
    prepared, policies = _prepared(3)
    journal = _DurableJournal()
    lifecycle = _Lifecycle()
    started = asyncio.Event()
    release = asyncio.Event()
    stages = _StageRuntime(
        journal=journal,
        overlap_started=started,
        overlap_release=release,
    )
    reflections = _ReflectionRuntime(started=started, release=release)

    result = await EvolutionCampaignScheduler(
        prepared=prepared,
        policies=policies,
        stages=stages,
        reflections=reflections,
        lifecycle=lifecycle,
        journal=journal,
    ).run()

    assert stages.overlap_observed
    assert reflections.finished
    assert result.tail_drain_receipt is not None
    assert result.tail_drain_receipt.admitted_for_testing is False
    assert result.tail_drain_receipt.lifecycle_promoted is False
    assert result.test_admission_receipts == ()
    assert reflections.test_admission_requests == []
    assert stages.stage_requests[2].test_eligible_reflection_receipt_sha256s == ()
    assert result.counters.generations_completed == 3
    assert result.counters.unique_evaluations == 22
    assert result.counters.logical_agent_calls == 5
    assert (
        lifecycle.finalization_requests[-1].status is CampaignExecutionStatus.COMPLETED
    )
    assert lifecycle.cleanup_requests[-1].status is CampaignExecutionStatus.COMPLETED


@_async_test
async def test_delayed_admission_overlaps_g3_g4_and_is_first_visible_at_g5():
    prepared, policies = _prepared(6, delayed_admission=True)
    journal = _DurableJournal()
    lifecycle = _Lifecycle()
    started = asyncio.Event()
    release = asyncio.Event()
    stages = _StageRuntime(
        journal=journal,
        overlap_started=started,
        overlap_release=release,
        overlap_observe_generations=(3, 4),
        overlap_release_generation=4,
    )
    reflections = _ReflectionRuntime(started=started, release=release)

    result = await EvolutionCampaignScheduler(
        prepared=prepared,
        policies=policies,
        stages=stages,
        reflections=reflections,
        lifecycle=lifecycle,
        journal=journal,
    ).run()

    assert stages.overlap_observed_generations == [3, 4]
    assert [request.wave.source_generation for request in reflections.requests] == [2]
    reflection_request = reflections.requests[0]
    assert reflection_request.source_stage.generation == 2
    reflection_record = reflection_request.to_record()
    assert reflection_record["sealed_evidence_cutoff"] == {
        "generation": 2,
        "stage_receipt_sha256": reflection_request.source_stage.receipt_sha256,
    }
    assert reflection_record["future_stage_evidence_permitted"] is False
    assert "admission_cutoff_stage_receipt_sha256" not in reflection_record

    assert len(reflections.test_admission_requests) == 1
    admission = reflections.test_admission_requests[0]
    assert admission.barrier.generation == 4
    assert admission.barrier.reflection_source_generations == (2,)
    assert admission.admission_cutoff_stage is not None
    assert admission.admission_cutoff_stage.generation == 4
    admission_record = admission.to_record()
    assert admission_record["admission_cutoff_stage_receipt_sha256"] == (
        admission.admission_cutoff_stage.receipt_sha256
    )
    assert admission_record["future_evidence_visible_to_reflection"] is False

    assert stages.stage_requests[2].step.generation == 3
    assert stages.stage_requests[3].step.generation == 4
    assert stages.stage_requests[4].step.generation == 5
    assert stages.stage_requests[2].test_eligible_reflection_receipt_sha256s == ()
    assert stages.stage_requests[3].test_eligible_reflection_receipt_sha256s == ()
    assert len(
        stages.stage_requests[4].test_eligible_reflection_receipt_sha256s
    ) == 1
    assert result.tail_drain_receipt is None
    assert result.counters.generations_completed == 6
    assert result.counters.unique_evaluations == 38
    assert result.counters.logical_agent_calls == 7
    assert result.counters.logical_agent_calls_dispatched_to_runtime == 7
    assert result.counters.logical_agent_calls_succeeded == 7


@_async_test
async def test_generation_six_barrier_admits_testing_before_generation_seven():
    prepared, policies = _prepared(7)
    journal = _DurableJournal()
    lifecycle = _Lifecycle()
    stages = _StageRuntime(journal=journal)
    reflections = _ReflectionRuntime()

    result = await EvolutionCampaignScheduler(
        prepared,
        policies,
        stages,
        reflections,
        lifecycle,
        journal,
    ).run()

    assert len(result.test_admission_receipts) == 1
    assert result.test_admission_receipts[0].barrier_generation == 6
    assert result.test_admission_receipts[0].lifecycle_promoted is False
    assert len(stages.stage_requests[6].test_eligible_reflection_receipt_sha256s) == 3
    assert result.tail_drain_receipt is None
    assert result.counters.unique_evaluations == 46
    assert result.counters.logical_agent_calls == 11
    assert result.counters.logical_agent_calls_dispatched_to_runtime == 11
    assert result.counters.logical_agent_calls_succeeded == 11
    assert result.counters.logical_agent_calls_failed == 0


@_async_test
async def test_reflection_failure_stops_at_next_completed_stage_boundary():
    prepared, policies = _prepared(
        6,
        failure_mode=ReflectionFailureMode.FAIL_AT_NEXT_STAGE_BOUNDARY,
    )
    journal = _DurableJournal()
    lifecycle = _Lifecycle()
    started = asyncio.Event()
    release = asyncio.Event()
    stages = _StageRuntime(
        journal=journal,
        overlap_started=started,
        overlap_release=release,
    )
    reflections = _ReflectionRuntime(
        started=started,
        release=release,
        fail_source=2,
    )

    with pytest.raises(
        CampaignExecutionContractError,
        match="next durable stage boundary",
    ):
        await EvolutionCampaignScheduler(
            prepared,
            policies,
            stages,
            reflections,
            lifecycle,
            journal,
        ).run()

    assert [value.step.generation for value in stages.stage_requests] == [1, 2, 3]
    assert lifecycle.finalization_requests[-1].status is CampaignExecutionStatus.FAILED
    counters = lifecycle.finalization_requests[-1].counters
    assert counters.generations_completed == 3
    assert counters.unique_evaluations == 22
    assert counters.logical_agent_calls == 5
    assert counters.logical_agent_calls_dispatched_to_runtime == 5
    assert counters.logical_agent_calls_succeeded == 4
    assert counters.logical_agent_calls_failed == 1
    kinds = [event.kind for event in journal.events]
    assert kinds.index(CampaignExecutionEventKind.REFLECTION_FAILED) < kinds.index(
        CampaignExecutionEventKind.EXECUTION_FAILED
    )


@_async_test
async def test_collect_at_barrier_settles_every_sibling_before_failure():
    prepared, policies = _prepared(
        6,
        failure_mode=ReflectionFailureMode.COLLECT_ALL_AT_BARRIER_THEN_FAIL,
    )
    journal = _DurableJournal()
    lifecycle = _Lifecycle()
    stages = _StageRuntime(journal=journal)
    reflections = _ReflectionRuntime(fail_source=2)

    with pytest.raises(
        CampaignExecutionContractError,
        match="block closed",
    ):
        await EvolutionCampaignScheduler(
            prepared,
            policies,
            stages,
            reflections,
            lifecycle,
            journal,
        ).run()

    assert [value.step.generation for value in stages.stage_requests] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert [value.wave.source_generation for value in reflections.requests] == [2, 4, 6]
    counters = lifecycle.finalization_requests[-1].counters
    assert counters.logical_agent_calls == 9
    assert counters.logical_agent_calls_dispatched_to_runtime == 9
    assert counters.logical_agent_calls_succeeded == 8
    assert counters.logical_agent_calls_failed == 1
    assert counters.logical_agent_calls_cancelled_before_dispatch == 0
    assert counters.logical_agent_calls_cancelled_after_dispatch == 0
    kinds = [event.kind for event in journal.events]
    assert kinds.count(CampaignExecutionEventKind.REFLECTION_COMPLETED) == 2
    assert kinds.count(CampaignExecutionEventKind.REFLECTION_FAILED) == 1
    assert CampaignExecutionEventKind.REFLECTION_CANCELLED not in kinds


@_async_test
async def test_best_effort_returns_degraded_without_partial_block_admission():
    prepared, policies = _prepared(
        6,
        failure_mode=ReflectionFailureMode.BEST_EFFORT_DEGRADED,
    )
    journal = _DurableJournal()
    lifecycle = _Lifecycle()
    reflections = _ReflectionRuntime(fail_source=2)

    result = await EvolutionCampaignScheduler(
        prepared,
        policies,
        _StageRuntime(journal=journal),
        reflections,
        lifecycle,
        journal,
    ).run()

    assert result.finalization_receipt.status is CampaignExecutionStatus.DEGRADED
    assert result.test_admission_receipts == ()
    assert tuple(value.status for value in result.reflection_receipts) == (
        CampaignReflectionStatus.FAILED,
        CampaignReflectionStatus.COMPLETED,
        CampaignReflectionStatus.COMPLETED,
    )
    assert result.counters.generations_completed == 6
    assert result.counters.unique_evaluations == 38
    assert result.counters.logical_agent_calls == 9
    assert result.counters.logical_agent_calls_dispatched_to_runtime == 9
    assert result.counters.logical_agent_calls_succeeded == 8
    assert result.counters.logical_agent_calls_failed == 1
    assert CampaignExecutionEventKind.EXECUTION_DEGRADED in {
        event.kind for event in journal.events
    }


@_async_test
async def test_e0_reflection_abstention_is_completed_without_test_admission():
    prepared, policies = _prepared(6, delayed_admission=True)
    journal = _DurableJournal()
    lifecycle = _Lifecycle()
    reflections = _ReflectionRuntime(abstain_source=2)

    result = await EvolutionCampaignScheduler(
        prepared,
        policies,
        _StageRuntime(journal=journal),
        reflections,
        lifecycle,
        journal,
    ).run()

    assert result.finalization_receipt.status is CampaignExecutionStatus.COMPLETED
    assert result.test_admission_receipts == ()
    assert tuple(value.status for value in result.reflection_receipts) == (
        CampaignReflectionStatus.ABSTAINED,
    )
    assert result.counters.logical_agent_calls == 7
    assert result.counters.logical_agent_calls_dispatched_to_runtime == 7
    assert result.counters.logical_agent_calls_succeeded == 6
    assert result.counters.logical_agent_calls_abstained == 1
    assert result.counters.logical_agent_calls_failed == 0
    kinds = {event.kind for event in journal.events}
    assert CampaignExecutionEventKind.REFLECTION_ABSTAINED in kinds
    assert CampaignExecutionEventKind.EXECUTION_DEGRADED not in kinds


@_async_test
async def test_abort_cancellation_is_typed_durable_and_accounted():
    prepared, policies = _prepared(
        6,
        failure_mode=ReflectionFailureMode.FAIL_AT_NEXT_STAGE_BOUNDARY,
    )
    journal = _DurableJournal()
    lifecycle = _Lifecycle()
    started = asyncio.Event()
    never_release = asyncio.Event()
    reflections = _ReflectionRuntime(started=started, release=never_release)

    with pytest.raises(RuntimeError, match="injected stage failure"):
        await EvolutionCampaignScheduler(
            prepared,
            policies,
            _StageRuntime(journal=journal, fault_generation=3),
            reflections,
            lifecycle,
            journal,
        ).run()

    finalization = lifecycle.finalization_requests[-1]
    assert len(finalization.reflection_cancellation_receipt_sha256s) == 1
    counters = finalization.counters
    assert counters.logical_agent_calls == 3
    assert counters.logical_agent_calls_dispatched_to_runtime == 3
    assert counters.logical_agent_calls_succeeded == 2
    assert counters.logical_agent_calls_cancelled_before_dispatch == 0
    assert counters.logical_agent_calls_cancelled_after_dispatch == 1
    assert CampaignExecutionEventKind.REFLECTION_CANCELLED in {
        event.kind for event in journal.events
    }


@_async_test
async def test_selector_audits_are_durably_persisted_before_later_stages():
    prepared, policies = _prepared(3)
    journal = _DurableJournal()
    stages = _StageRuntime(journal=journal)

    result = await EvolutionCampaignScheduler(
        prepared,
        policies,
        stages,
        _ReflectionRuntime(),
        _Lifecycle(),
        journal,
    ).run()

    stage_events = [
        event
        for event in journal.events
        if event.kind is CampaignExecutionEventKind.STAGE_SEALED
    ]
    assert len(stage_events) == 3
    first = thaw_json(stage_events[0].payload)["stage_receipt"]
    third = thaw_json(stage_events[2].payload)["stage_receipt"]
    assert len(first["selector_audits"]) == 2
    assert len(third["selector_audits"]) == 2
    assert all(
        audit["execution_mode"] == "fresh"
        for stage in (first, third)
        for audit in stage["selector_audits"]
    )
    assert all(
        audit["plaintext_audit"]["request_sha256"] == audit["request_sha256"]
        and audit["plaintext_audit"]["decision_sha256"] == audit["decision_sha256"]
        and audit["plaintext_audit"]["request_text"]
        and audit["plaintext_audit"]["response_text"]
        for stage in (first, third)
        for audit in stage["selector_audits"]
    )
    assert tuple(event.event_sha256 for event in journal.events) == (
        result.durable_event_sha256s
    )


@_async_test
async def test_stage_failure_still_finalizes_and_releases_runtime():
    prepared, policies = _prepared(3)
    journal = _DurableJournal()
    lifecycle = _Lifecycle()

    with pytest.raises(RuntimeError, match="injected stage failure"):
        await EvolutionCampaignScheduler(
            prepared,
            policies,
            _StageRuntime(journal=journal, fault_generation=2),
            _ReflectionRuntime(),
            lifecycle,
            journal,
        ).run()

    assert lifecycle.finalization_requests[-1].status is CampaignExecutionStatus.FAILED
    assert lifecycle.cleanup_requests[-1].status is CampaignExecutionStatus.FAILED
    kinds = [event.kind for event in journal.events]
    assert CampaignExecutionEventKind.EXECUTION_FAILED in kinds
    assert CampaignExecutionEventKind.EXECUTION_FINALIZED in kinds
    assert CampaignExecutionEventKind.RUNTIME_CLEANED in kinds
    failure_event = next(
        event
        for event in journal.events
        if event.kind is CampaignExecutionEventKind.EXECUTION_FAILED
    )
    failure_payload = thaw_json(failure_event.payload)
    assert "injected stage failure" not in str(failure_payload)
    assert failure_payload["failure_digest_sha256"]


@_async_test
async def test_failed_seed_accounting_prevents_stages_and_still_cleans_up():
    prepared, policies = _prepared(3)
    journal = _DurableJournal()
    lifecycle = _Lifecycle(seed_failure=True)
    stages = _StageRuntime(journal=journal)

    with pytest.raises(CampaignExecutionContractError, match="failed or unevaluated"):
        await EvolutionCampaignScheduler(
            prepared,
            policies,
            stages,
            _ReflectionRuntime(),
            lifecycle,
            journal,
        ).run()

    assert stages.stage_requests == []
    assert lifecycle.finalization_requests[-1].counters.unique_evaluations == 2
    assert lifecycle.cleanup_requests[-1].status is CampaignExecutionStatus.FAILED


@_async_test
async def test_durable_journal_failure_still_finalizes_and_cleans_up():
    prepared, policies = _prepared(3)
    journal = _DurableJournal(fail_kind=CampaignExecutionEventKind.STAGE_SEALED)
    lifecycle = _Lifecycle()

    with pytest.raises(OSError, match="durable journal failure"):
        await EvolutionCampaignScheduler(
            prepared,
            policies,
            _StageRuntime(),
            _ReflectionRuntime(),
            lifecycle,
            journal,
        ).run()

    assert lifecycle.finalization_requests[-1].status is CampaignExecutionStatus.FAILED
    assert lifecycle.cleanup_requests[-1].status is CampaignExecutionStatus.FAILED


@_async_test
async def test_finalization_failure_still_invokes_cleanup():
    prepared, policies = _prepared(3)
    lifecycle = _Lifecycle(finalize_failure=True)

    with pytest.raises(RuntimeError, match="finalization failure"):
        await EvolutionCampaignScheduler(
            prepared,
            policies,
            _StageRuntime(),
            _ReflectionRuntime(),
            lifecycle,
            _DurableJournal(),
        ).run()

    assert lifecycle.cleanup_requests[-1].status is CampaignExecutionStatus.FAILED
    assert lifecycle.cleanup_requests[-1].finalization_receipt_sha256 is None


@_async_test
async def test_foreign_valid_start_receipt_is_finalized_and_cleaned():
    prepared, policies = _prepared(3)
    lifecycle = _Lifecycle(foreign_start=True)

    with pytest.raises(CampaignExecutionContractError, match="start receipt"):
        await EvolutionCampaignScheduler(
            prepared,
            policies,
            _StageRuntime(),
            _ReflectionRuntime(),
            lifecycle,
            _DurableJournal(),
        ).run()

    assert lifecycle.finalization_requests[-1].status is CampaignExecutionStatus.FAILED
    assert lifecycle.cleanup_requests[-1].status is CampaignExecutionStatus.FAILED


@pytest.mark.parametrize("receipt_fault", ("out_of_order", "foreign_request"))
@_async_test
async def test_out_of_order_or_foreign_stage_receipt_is_rejected_and_cleaned(
    receipt_fault: str,
):
    prepared, policies = _prepared(3)
    journal = _DurableJournal()
    lifecycle = _Lifecycle()

    with pytest.raises(CampaignExecutionContractError, match="out-of-order or foreign"):
        await EvolutionCampaignScheduler(
            prepared,
            policies,
            _StageRuntime(journal=journal, receipt_fault=receipt_fault),
            _ReflectionRuntime(),
            lifecycle,
            journal,
        ).run()

    assert lifecycle.cleanup_requests[-1].status is CampaignExecutionStatus.FAILED


@_async_test
async def test_duplicate_selector_audit_is_not_accepted_as_fresh():
    prepared, policies = _prepared(3)
    journal = _DurableJournal()
    lifecycle = _Lifecycle()

    with pytest.raises(CampaignExecutionContractError, match="duplicate"):
        await EvolutionCampaignScheduler(
            prepared,
            policies,
            _StageRuntime(journal=journal, duplicate_audit=True),
            _ReflectionRuntime(),
            lifecycle,
            journal,
        ).run()

    assert lifecycle.cleanup_requests[-1].status is CampaignExecutionStatus.FAILED
