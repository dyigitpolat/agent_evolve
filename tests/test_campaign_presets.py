"""Conformance for the high-level delayed portfolio campaign preset."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent_evolve import (
    ANCHOR_HEAVY_36_OFFSPRING_SCALE_SHAPE,
    DelayedPortfolioCampaignPreset,
    CampaignExperimentProfile,
    PortfolioCampaignBehavior,
    REFERENCE_36_OFFSPRING_SCALE_SHAPE,
    REFERENCE_HEAVY_B32_SCALE_SHAPE,
    REFERENCE_HEAVY_B40_SCALE_SHAPE,
    SMALL_BUDGET_CAMPAIGN_SCALE_SHAPES,
    WorkloadKit,
    campaign_seed,
)
from agent_evolve.integrations.pydantic_ai.model_execution_profile import (
    DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
)
from agent_evolve.workload_prompt import WorkloadPromptArm
from agent_evolve.campaign_presets import (
    EQUAL_36_OFFSPRING_SCALE_SHAPES,
    EQUAL_60_OFFSPRING_SCALE_SHAPES,
)
from agent_evolve.agentic import AgenticBenchmark
from agent_evolve.application.evolution_campaign import (
    ArchiveUtilitySnapshot,
    CampaignAgentRuntimeReceipt,
    CampaignPolicyBinding,
    SealedCutoffDelayedAdmissionCadence,
)
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json
from examples.benchmarks.boils_abc.actions import DEFAULT_ACTION_SEQUENCE
from examples.benchmarks.boils_abc.budgeted_v5_support import PARENT_C_SEQUENCE
from examples.benchmarks.boils_abc.evaluator import AbcEvaluatorSettings
from examples.benchmarks.boils_abc.finite_variation_catalog import (
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


def _binding(name: str) -> CampaignPolicyBinding:
    return CampaignPolicyBinding(
        implementation=object(),
        policy_id=name,
        policy_version=1,
        definition_sha256=_sha(f"policy:{name}"),
    )


class _ArchiveUtility:
    utility_id = "preset_test_archive_utility"
    utility_version = 1
    definition_sha256 = _sha("preset-test-archive-utility")

    def freeze(self, *, benchmark, generation, archive):
        from agent_evolve.domain.typed_json import typed_json_sha256

        return ArchiveUtilitySnapshot(
            utility_id=self.utility_id,
            utility_version=self.utility_version,
            definition_sha256=self.definition_sha256,
            generation=generation,
            benchmark_sha256=typed_json_sha256(benchmark),
            archive_sha256=typed_json_sha256(archive),
            snapshot_receipt=_object({"test_only": True}),
        )


class _Runtime:
    def __init__(self) -> None:
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        return CampaignAgentRuntimeReceipt(
            request_sha256=request.request_sha256,
            runtime_id="preset_test_runtime",
            runtime_version=1,
            definition_sha256=_sha("preset-test-runtime"),
            accepted=True,
            evidence=_object({"provider_calls": 0, "accepted": True}),
        )


class _Journal:
    def __init__(self) -> None:
        self.records = []

    def append(self, record):
        self.records.append(record)


def _workload() -> WorkloadKit:
    settings = AbcEvaluatorSettings.current_circuit_panel(
        circuit_names=("log2",),
        affinity_sets=((0,),),
        per_circuit_timeout_s=60.0,
    )
    benchmark = AgenticBenchmark(
        problem=BoilsAbcProblem(settings),
        finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
    )
    return WorkloadKit(
        workload_id="preset_boils_conformance",
        workload_version=1,
        benchmark=benchmark,
        seeds=(
            campaign_seed(
                "seed_default",
                {"sequence": list(DEFAULT_ACTION_SEQUENCE)},
            ),
            campaign_seed(
                "seed_parent_c",
                {"sequence": list(PARENT_C_SEQUENCE)},
            ),
        ),
        evaluator_concurrency_cap=1,
        evaluator_preflight_receipt=_object(
            {"qualified": True, "abc_executions": 0, "provider_calls": 0}
        ),
        resource_lease_receipt=_object(
            {"resource": "one_pinned_cpu_affinity", "active": True}
        ),
    )


def _behavior() -> PortfolioCampaignBehavior:
    return PortfolioCampaignBehavior(
        parent_selection=_binding("archive_parent_selection"),
        memory_assignment=_binding("bounded_memory_assignment"),
        portfolio_selection=_binding("calibrated_portfolio_selection"),
        recombination=_binding("typed_patch_recombination"),
        reflection=_binding("delayed_identifiable_reflection"),
        archive_utility=_ArchiveUtility(),
    )


def test_g6_preset_derives_the_exact_38_evaluation_7_call_schedule() -> None:
    preset = DelayedPortfolioCampaignPreset.generations(6, outer_seed=33)
    protocol = preset.protocol(required_seed_count=2)
    budget = preset.budget(required_seed_count=2)

    assert protocol.generation_count == 6
    assert protocol.recombinations_per_parent == 2
    assert budget.max_unique_evaluations == 38
    assert budget.max_logical_llm_calls == 7
    assert budget.max_generations == 6


def test_three_named_scale_shapes_hold_offspring_and_total_opportunity_constant() -> None:
    assert tuple(EQUAL_60_OFFSPRING_SCALE_SHAPES) == (
        "g4_k8_r7",
        "g6_k8_r2",
        "g10_k4_r2",
    )
    expected_stages = {
        "g4_k8_r7": (16, 14, 16, 14),
        "g6_k8_r2": (16, 4, 16, 4, 16, 4),
        "g10_k4_r2": (8, 4, 8, 4, 8, 4, 8, 4, 8, 4),
    }
    for name, shape in EQUAL_60_OFFSPRING_SCALE_SHAPES.items():
        assert shape.planned_offspring_occurrences == 60
        preset = DelayedPortfolioCampaignPreset.scale_shape(
            shape,
            outer_seed=40,
        )
        protocol = preset.protocol(required_seed_count=2)
        budget = preset.budget(required_seed_count=2)
        steps = SealedCutoffDelayedAdmissionCadence().build(protocol).steps
        assert tuple(value.planned_candidate_evaluations for value in steps) == (
            expected_stages[name]
        )
        assert budget.max_unique_evaluations == 62


def test_equal_b38_shapes_trade_recombination_for_anchor_capacity_exactly() -> None:
    assert tuple(EQUAL_36_OFFSPRING_SCALE_SHAPES) == (
        "g6_k4_r2",
        "g6_k5_r1",
    )
    expected_stages = {
        "g6_k4_r2": (8, 4, 8, 4, 8, 4),
        "g6_k5_r1": (10, 2, 10, 2, 10, 2),
    }
    for name, shape in EQUAL_36_OFFSPRING_SCALE_SHAPES.items():
        assert shape.planned_offspring_occurrences == 36
        preset = DelayedPortfolioCampaignPreset.scale_shape(
            shape,
            outer_seed=41,
        )
        protocol = preset.protocol(required_seed_count=2)
        budget = preset.budget(required_seed_count=2)
        steps = SealedCutoffDelayedAdmissionCadence().build(protocol).steps
        assert tuple(value.planned_candidate_evaluations for value in steps) == (
            expected_stages[name]
        )
        assert budget.max_unique_evaluations == 38

    assert ANCHOR_HEAVY_36_OFFSPRING_SCALE_SHAPE == (
        EQUAL_36_OFFSPRING_SCALE_SHAPES["g6_k5_r1"]
    )


def test_reference_heavy_b40_shape_is_not_mislabeled_as_equal_b38() -> None:
    shape = SMALL_BUDGET_CAMPAIGN_SCALE_SHAPES["g7_k4_r1"]

    assert shape == REFERENCE_HEAVY_B40_SCALE_SHAPE
    assert shape.portfolio_generation_count == 4
    assert shape.recombination_generation_count == 3
    assert shape.planned_offspring_occurrences == 38
    assert shape.shape_id not in EQUAL_36_OFFSPRING_SCALE_SHAPES

    screen = SMALL_BUDGET_CAMPAIGN_SCALE_SHAPES["g6_k4_r1"]
    assert screen == REFERENCE_HEAVY_B32_SCALE_SHAPE
    assert screen.planned_offspring_occurrences == 30
    assert screen.shape_id not in EQUAL_36_OFFSPRING_SCALE_SHAPES


def test_preset_prepares_workload_kit_without_provider_or_evaluator_work() -> None:
    runtime = _Runtime()
    journal = _Journal()
    preset = DelayedPortfolioCampaignPreset.generations(6, outer_seed=34)

    prepared = preset.compose(
        workload=_workload(),
        behavior=_behavior(),
        runtime=runtime,
        journals=(journal,),
    ).prepare()

    assert tuple(
        step.planned_candidate_evaluations for step in prepared.schedule.steps
    ) == (8, 4, 8, 4, 8, 4)
    assert tuple(step.planned_agent_calls for step in prepared.schedule.steps) == (
        2,
        1,
        2,
        0,
        2,
        0,
    )
    assert tuple(
        (wave.source_generation, wave.promotion_barrier_generation)
        for wave in prepared.schedule.reflection_waves
    ) == ((2, 4),)
    assert prepared.budget.max_unique_evaluations == 38
    assert prepared.budget.max_logical_llm_calls == 7
    assert len(runtime.requests) == 1
    assert len(journal.records) == 1


def test_experiment_profile_authenticates_the_prepared_runtime_contract() -> None:
    runtime = _Runtime()
    journal = _Journal()
    behavior = _behavior()
    profile = CampaignExperimentProfile(
        profile_id="profile_preparation_conformance",
        profile_version=1,
        method_id="profile_preparation_method",
        method_version=1,
        scale_shape=REFERENCE_36_OFFSPRING_SCALE_SHAPE,
        candidate_pool_size=None,
        model_selection_size=8,
        prompt_arm=WorkloadPromptArm.SEMANTIC,
        parent_selection=behavior.parent_selection,
        memory_assignment=behavior.memory_assignment,
        portfolio_selection=behavior.portfolio_selection,
        recombination=behavior.recombination,
        reflection=behavior.reflection,
        model_execution=DEEPSEEK_V4_PRO_STREAMLAKE_XHIGH_NATIVE_JSON,
    )
    prepared = profile.compose(
        outer_seed=37,
        workload=_workload(),
        archive_utility=behavior.archive_utility,
        runtime=runtime,
        journals=(journal,),
    ).prepare()

    record = profile.prepared_conformance_record(
        prepared=prepared,
        archive_utility=behavior.archive_utility,
        outer_seed=37,
    )
    assert record["pass"] is True
    assert all(record["gates"].values())

    with pytest.raises(RuntimeError, match="concurrency_exact"):
        replace(profile, agent_concurrency=2).prepared_conformance_record(
            prepared=prepared,
            archive_utility=behavior.archive_utility,
            outer_seed=37,
        )
