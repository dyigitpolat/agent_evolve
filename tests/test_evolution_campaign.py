from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from agent_evolve.application.budgeted_optimizer import OptimizerBudget
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
    ReflectionLaunchMode,
    SealedCutoffDelayedAdmissionCadence,
    TerminalReflectionPolicy,
    ReflectionVisibility,
    freeze_archive_utility,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _protocol(generation_count: int) -> CampaignProtocol:
    return CampaignProtocol(
        protocol_id="portfolio_q",
        protocol_version=1,
        definition_sha256=_sha("portfolio-q-protocol"),
        outer_seed=20260716,
        generation_count=generation_count,
        required_seed_count=2,
        parents_per_portfolio_generation=2,
        portfolio_width=4,
        recombinations_per_parent=2,
        reflections_per_recombination_generation=1,
    )


def _policy(name: str) -> CampaignPolicyBinding:
    return CampaignPolicyBinding(
        implementation=object(),
        policy_id=name,
        policy_version=1,
        definition_sha256=_sha(f"policy:{name}"),
    )


class _FakeArchiveUtility:
    utility_id = "fixed_archive_utility"
    utility_version = 1
    definition_sha256 = _sha("fixed-archive-utility")

    def freeze(self, *, benchmark, generation, archive):
        return ArchiveUtilitySnapshot(
            utility_id=self.utility_id,
            utility_version=self.utility_version,
            definition_sha256=self.definition_sha256,
            generation=generation,
            benchmark_sha256=typed_json_sha256(benchmark),
            archive_sha256=typed_json_sha256(archive),
            snapshot_receipt=_object({"reference": [10, 20]}),
        )


def _policies() -> CampaignPolicies:
    return CampaignPolicies(
        cadence=AlternatingPortfolioRecombinationCadence(),
        parent_selection=_policy("front_first"),
        memory_assignment=_policy("balanced_memory"),
        portfolio_selection=_policy("ranked_portfolio"),
        recombination=_policy("exact_recombination"),
        reflection=_policy("grounded_reflection"),
        archive_utility=_FakeArchiveUtility(),
    )


class _FakeBenchmark:
    port_id = "fake_benchmark"
    port_version = 1

    def __init__(self, workload_key: str, *, concurrency_cap: int = 8) -> None:
        self.workload_key = workload_key
        self.definition_sha256 = _sha(f"benchmark:{workload_key}")
        self.concurrency_cap = concurrency_cap
        self.requests: list[BenchmarkSessionRequest] = []

    def open(self, request: BenchmarkSessionRequest) -> CampaignBenchmarkSession:
        self.requests.append(request)
        return CampaignBenchmarkSession(
            request_sha256=request.request_sha256,
            benchmark=_object(
                {
                    "instance_key": self.workload_key,
                    "objective_arity": 2,
                }
            ),
            evaluator_concurrency_cap=self.concurrency_cap,
            preflight_receipt=_object(
                {"qualified": True, "adapter_key": self.workload_key}
            ),
            resource_lease=_object(
                {"lease_key": f"lease:{self.workload_key}", "active": True}
            ),
        )


class _FakeSeeds:
    port_id = "fake_seeds"
    port_version = 1

    def __init__(self, workload_key: str) -> None:
        self.workload_key = workload_key
        self.definition_sha256 = _sha(f"seeds:{workload_key}")

    def load(self, session: CampaignBenchmarkSession) -> CampaignSeedBatch:
        return CampaignSeedBatch(
            session_sha256=session.session_sha256,
            seeds=(
                CampaignSeed(
                    seed_id="seed_a",
                    configuration=_object(
                        {"workload_key": self.workload_key, "coordinate": 0}
                    ),
                ),
                CampaignSeed(
                    seed_id="seed_b",
                    configuration=_object(
                        {"workload_key": self.workload_key, "coordinate": 1}
                    ),
                ),
            ),
        )


class _FakeCatalog:
    port_id = "fake_catalog"
    port_version = 1

    def __init__(self, workload_key: str) -> None:
        self.definition_sha256 = _sha(f"catalog:{workload_key}")

    def bind(self, benchmark, parent, known_phenotype_sha256s):
        del benchmark, parent, known_phenotype_sha256s
        raise AssertionError("prepare must not enumerate a parent-bound catalog")


class _FakeEvidence:
    port_id = "fake_evidence"
    port_version = 1

    def __init__(self, workload_key: str) -> None:
        self.definition_sha256 = _sha(f"evidence:{workload_key}")

    def initialize_memory(self, session, seeds):
        del session, seeds
        return _object({})

    def context(self, session, parent, variation, memory):
        del session, parent, variation, memory
        return _object({})

    def cards(self, session, parent, variation, memory):
        del session, parent, variation, memory
        return (_object({"card": 1}),)


class _FakeRuntime:
    def __init__(self) -> None:
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        return CampaignAgentRuntimeReceipt(
            request_sha256=request.request_sha256,
            runtime_id="fake_agent_runtime",
            runtime_version=1,
            definition_sha256=_sha("fake-agent-runtime"),
            accepted=True,
            evidence=_object({"queue_preflight": "accepted"}),
        )


class _MemoryJournal:
    def __init__(self) -> None:
        self.records: list[FrozenJsonObject] = []

    def append(self, record: FrozenJsonObject) -> None:
        assert type(record) is FrozenJsonObject
        self.records.append(record)


def _workload(workload_key: str, *, concurrency_cap: int = 8):
    return CampaignWorkloadPorts(
        benchmark=_FakeBenchmark(
            workload_key,
            concurrency_cap=concurrency_cap,
        ),
        seeds=_FakeSeeds(workload_key),
        catalog=_FakeCatalog(workload_key),
        evidence=_FakeEvidence(workload_key),
    )


@pytest.mark.parametrize("generation_count", range(3, 25))
def test_alternating_cadence_counts_actual_generations_and_odd_terminal_portfolio(
    generation_count: int,
):
    schedule = AlternatingPortfolioRecombinationCadence().build(
        _protocol(generation_count)
    )

    assert len(schedule.steps) == generation_count
    assert schedule.portfolio_generations == tuple(range(1, generation_count + 1, 2))
    assert schedule.paired_recombination_generations == tuple(
        range(2, generation_count + 1, 2)
    )
    assert tuple(step.kind for step in schedule.steps) == tuple(
        CampaignGenerationKind.PORTFOLIO
        if generation % 2
        else CampaignGenerationKind.RECOMBINATION
        for generation in range(1, generation_count + 1)
    )
    assert schedule.unpaired_terminal_portfolio_generation == (
        generation_count if generation_count % 2 else None
    )


def test_six_generation_q_schedule_reproduces_exact_38_eval_9_call_envelope():
    protocol = _protocol(6)
    schedule = AlternatingPortfolioRecombinationCadence().build(protocol)

    assert protocol.required_seed_count + schedule.planned_candidate_evaluations == 38
    assert schedule.planned_agent_calls == 9
    assert tuple(
        (pair.portfolio_generation, pair.recombination_generation)
        for pair in schedule.recombination_pairs
    ) == ((1, 2), (3, 4), (5, 6))
    assert tuple(
        (
            wave.source_generation,
            wave.launch_mode,
            wave.visibility,
            wave.promotion_barrier_generation,
        )
        for wave in schedule.reflection_waves
    ) == (
        (
            2,
            ReflectionLaunchMode.ASYNC_AFTER_STAGE_SEAL,
            ReflectionVisibility.QUARANTINED_UNTIL_BLOCK_CLOSE,
            6,
        ),
        (
            4,
            ReflectionLaunchMode.ASYNC_AFTER_STAGE_SEAL,
            ReflectionVisibility.QUARANTINED_UNTIL_BLOCK_CLOSE,
            6,
        ),
        (
            6,
            ReflectionLaunchMode.ASYNC_AFTER_STAGE_SEAL,
            ReflectionVisibility.QUARANTINED_UNTIL_BLOCK_CLOSE,
            6,
        ),
    )
    assert tuple(
        (barrier.generation, barrier.reflection_source_generations)
        for barrier in schedule.promotion_barriers
    ) == ((6, (2, 4, 6)),)


def test_future_consumer_policy_drops_an_incomplete_unadmittable_cohort():
    protocol = replace(
        _protocol(6),
        terminal_reflection_policy=(
            TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER
        ),
    )
    schedule = AlternatingPortfolioRecombinationCadence().build(protocol)

    assert schedule.planned_agent_calls == 6
    assert tuple(step.planned_agent_calls for step in schedule.steps) == (
        2,
        0,
        2,
        0,
        2,
        0,
    )
    assert schedule.reflection_waves == ()
    assert schedule.promotion_barriers == ()


def test_future_consumer_policy_retains_complete_one_source_cohorts():
    protocol = replace(
        _protocol(6),
        terminal_reflection_policy=(
            TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER
        ),
        reflection_promotion_block_pairs=1,
    )
    schedule = AlternatingPortfolioRecombinationCadence().build(protocol)

    assert schedule.planned_agent_calls == 8
    assert tuple(wave.source_generation for wave in schedule.reflection_waves) == (
        2,
        4,
    )
    assert tuple(
        (barrier.generation, barrier.reflection_source_generations)
        for barrier in schedule.promotion_barriers
    ) == ((2, (2,)), (4, (4,)))


def test_sealed_cutoff_delayed_cadence_overlaps_one_pair_then_feeds_g5():
    protocol = replace(
        _protocol(6),
        terminal_reflection_policy=(
            TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER
        ),
        reflection_promotion_block_pairs=1,
    )
    schedule = SealedCutoffDelayedAdmissionCadence().build(protocol)

    assert tuple(step.planned_agent_calls for step in schedule.steps) == (
        2,
        1,
        2,
        0,
        2,
        0,
    )
    assert schedule.planned_agent_calls == 7
    assert tuple(
        (
            wave.source_generation,
            wave.promotion_barrier_generation,
        )
        for wave in schedule.reflection_waves
    ) == ((2, 4),)
    assert tuple(
        (barrier.generation, barrier.reflection_source_generations)
        for barrier in schedule.promotion_barriers
    ) == ((4, (2,)),)


@pytest.mark.parametrize("generation_count", range(3, 25))
def test_delayed_cadence_never_schedules_without_exact_future_portfolio_consumer(
    generation_count: int,
):
    protocol = replace(
        _protocol(generation_count),
        terminal_reflection_policy=(
            TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER
        ),
        reflection_promotion_block_pairs=1,
    )
    schedule = SealedCutoffDelayedAdmissionCadence().build(protocol)

    for wave in schedule.reflection_waves:
        assert wave.promotion_barrier_generation == wave.source_generation + 2
        consumer_generation = wave.promotion_barrier_generation + 1
        assert consumer_generation <= generation_count
        assert schedule.steps[consumer_generation - 1].kind is (
            CampaignGenerationKind.PORTFOLIO
        )


def test_delayed_cadence_rejects_ambiguous_terminal_or_cohort_configuration():
    cadence = SealedCutoffDelayedAdmissionCadence()

    with pytest.raises(ValueError, match="future-consumer terminal policy"):
        cadence.build(replace(_protocol(6), reflection_promotion_block_pairs=1))
    with pytest.raises(ValueError, match="one reflection per admission barrier"):
        cadence.build(
            replace(
                _protocol(6),
                terminal_reflection_policy=(
                    TerminalReflectionPolicy.REQUIRE_FUTURE_PORTFOLIO_CONSUMER
                ),
            )
        )


def test_fifteen_generation_learning_horizon_is_supported():
    schedule = AlternatingPortfolioRecombinationCadence().build(_protocol(15))

    assert len(schedule.steps) == 15
    assert schedule.portfolio_generations == (1, 3, 5, 7, 9, 11, 13, 15)
    assert schedule.unpaired_terminal_portfolio_generation == 15


def test_reflection_supervision_mode_is_authenticated_separately_from_visibility():
    values = tuple(
        CampaignReflectionSupervisionPolicy(mode) for mode in ReflectionFailureMode
    )

    assert len({value.configuration_sha256 for value in values}) == 3
    assert len({value.definition_sha256 for value in values}) == 1
    assert all(value.to_record()["visibility_independent"] for value in values)
    policies = _policies()
    fail_fast = CampaignPolicies(
        cadence=policies.cadence,
        parent_selection=policies.parent_selection,
        memory_assignment=policies.memory_assignment,
        portfolio_selection=policies.portfolio_selection,
        recombination=policies.recombination,
        reflection=policies.reflection,
        archive_utility=policies.archive_utility,
        reflection_supervision=values[0],
    )
    best_effort = CampaignPolicies(
        cadence=policies.cadence,
        parent_selection=policies.parent_selection,
        memory_assignment=policies.memory_assignment,
        portfolio_selection=policies.portfolio_selection,
        recombination=policies.recombination,
        reflection=policies.reflection,
        archive_utility=policies.archive_utility,
        reflection_supervision=values[-1],
    )
    assert fail_fast.policies_sha256 != best_effort.policies_sha256


def test_incomplete_reflection_block_stays_quarantined_without_a_false_barrier():
    schedule = AlternatingPortfolioRecombinationCadence().build(_protocol(8))

    assert tuple(
        wave.promotion_barrier_generation for wave in schedule.reflection_waves
    ) == (6, 6, 6, None)
    assert tuple(barrier.generation for barrier in schedule.promotion_barriers) == (6,)


def test_archive_utility_freeze_authenticates_reference_and_archive_cutoff():
    utility = _FakeArchiveUtility()
    benchmark = _object({"benchmark": "opaque"})
    archive = _object({"front": ["candidate_1", "candidate_2"]})

    first = freeze_archive_utility(
        utility,
        benchmark=benchmark,
        generation=3,
        archive=archive,
    )
    replay = freeze_archive_utility(
        utility,
        benchmark=benchmark,
        generation=3,
        archive=archive,
    )

    assert first.snapshot_sha256 == replay.snapshot_sha256
    assert first.definition_sha256 == utility.definition_sha256
    assert first.archive_sha256 == typed_json_sha256(archive)
    assert first.benchmark_sha256 == typed_json_sha256(benchmark)


def test_three_workloads_share_identical_protocol_and_schedule_records():
    protocol = _protocol(3)
    policies = _policies()
    runtime = _FakeRuntime()
    concurrency = CampaignConcurrency(
        evaluator_concurrency=4,
        agent_concurrency=2,
        agent_queue_capacity=8,
    )
    budget = OptimizerBudget(
        max_unique_evaluations=22,
        max_logical_llm_calls=5,
        max_generations=3,
    )
    prepared = []
    journals = []

    for workload_key in ("workload_1", "workload_2", "workload_3"):
        journal = _MemoryJournal()
        campaign = EvolutionCampaign(
            protocol=protocol,
            workload=_workload(workload_key),
            policies=policies,
            runtime=runtime,
            budget=budget,
            concurrency=concurrency,
            journals=(journal,),
        )
        prepared.append(campaign.prepare())
        journals.append(journal)

    first_schedule = prepared[0].schedule.to_record()
    first_protocol = prepared[0].protocol.to_record()
    assert all(value.schedule.to_record() == first_schedule for value in prepared)
    assert all(value.protocol.to_record() == first_protocol for value in prepared)
    assert all(
        value.schedule.unpaired_terminal_portfolio_generation == 3 for value in prepared
    )
    assert len({value.benchmark_session.session_sha256 for value in prepared}) == 3
    assert len({value.workload_ports_sha256 for value in prepared}) == 3
    assert len(runtime.requests) == 3
    assert all(len(journal.records) == 1 for journal in journals)

    for journal, value in zip(journals, prepared, strict=True):
        record = journal.records[0]
        assert record == freeze_json(value.to_record())
        thawed = value.to_record()
        assert thawed["protocol"] == first_protocol
        assert thawed["schedule"] == first_schedule

    portable_records = json.dumps(
        {"protocol": first_protocol, "schedule": first_schedule},
        sort_keys=True,
    ).lower()
    assert "provider" not in portable_records
    assert "model" not in portable_records


def test_prepare_rejects_budget_or_benchmark_concurrency_before_runtime():
    protocol = _protocol(3)
    runtime = _FakeRuntime()
    journal = _MemoryJournal()

    under_budget = EvolutionCampaign(
        protocol=protocol,
        workload=_workload("under_budget"),
        policies=_policies(),
        runtime=runtime,
        budget=OptimizerBudget(
            max_unique_evaluations=21,
            max_logical_llm_calls=5,
            max_generations=3,
        ),
        concurrency=CampaignConcurrency(4, 2, 8),
        journals=(journal,),
    )
    with pytest.raises(ValueError, match="maximum candidate evaluations"):
        under_budget.prepare()
    assert runtime.requests == []

    over_cap = EvolutionCampaign(
        protocol=protocol,
        workload=_workload("over_cap", concurrency_cap=2),
        policies=_policies(),
        runtime=runtime,
        budget=OptimizerBudget(22, 5, 3),
        concurrency=CampaignConcurrency(4, 2, 8),
        journals=(journal,),
    )
    with pytest.raises(ValueError, match="exceeds benchmark cap"):
        over_cap.prepare()
    assert runtime.requests == []
    assert journal.records == []


def test_campaign_boundary_is_exported_from_application_package():
    import agent_evolve.application as application

    assert application.EvolutionCampaign is EvolutionCampaign
    assert application.ArchiveUtilitySnapshot is ArchiveUtilitySnapshot
    assert (
        application.AlternatingPortfolioRecombinationCadence
        is AlternatingPortfolioRecombinationCadence
    )
