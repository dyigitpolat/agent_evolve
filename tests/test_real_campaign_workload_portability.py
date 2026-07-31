"""Provider-free campaign portability over three real benchmark adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_evolve.agentic import AgenticBenchmark
from agent_evolve.application.budgeted_optimizer import OptimizerBudget
from agent_evolve.application.calibrated_campaign import (
    CalibratedCampaignBindingFactory,
    equal_weight_slate_objectives,
    equal_weight_slate_objectives_from_decision_metrics,
)
from agent_evolve.application.evolution_campaign import (
    AlternatingPortfolioRecombinationCadence,
    ArchiveUtilitySnapshot,
    BenchmarkSessionRequest,
    CampaignAgentRuntimeReceipt,
    CampaignConcurrency,
    CampaignPolicies,
    CampaignPolicyBinding,
    CampaignProtocol,
    CampaignSeed,
    EvolutionCampaign,
)
from agent_evolve.application.portfolio_outcome_feedback import (
    PortfolioOutcomeFeedbackLedger,
)
from agent_evolve.campaign_workload import (
    AgenticCampaignEvidenceProjections,
    AgenticCampaignWorkloadConfig,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.domain.ids import InsightId, LLMCallId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_campaign import (
    CalibratedPortfolioCampaignCoordinator,
)
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_selection import (
    CALIBRATED_PORTFOLIO_PROMPT_DEFINITION_SHA256,
    CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256,
    calibrated_portfolio_prompt_definition_sha256,
)
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationScope,
)
from agent_evolve.policies.selection.finite_option_prompt_projection import (
    FiniteOptionPromptProjectionPolicy,
)
from agent_evolve.ports.decision_metric_projection import DecisionMetricProjection
from agent_evolve.ports.portfolio_selection import (
    PortfolioCard,
    PortfolioSelectionRequest,
)
from examples.benchmarks.boils_abc.actions import DEFAULT_ACTION_SEQUENCE
from examples.benchmarks.boils_abc.evaluator import AbcEvaluatorSettings
from examples.benchmarks.boils_abc.finite_variation_catalog import (
    FINITE_CATALOG_ID,
    BoilsFiniteVariationCatalog,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem
from examples.benchmarks.engibench_airfoil.v7_contract import (
    AIRFOIL_V7_ARCHIVE_RELATION,
    AIRFOIL_V7_REWARD_BINDING,
    AirfoilV7PhenotypeIdentityPolicy,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    AIRFOIL_V7_ACTION_SEMANTICS,
    AirfoilV7Problem,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    AirfoilV7ShapeVariationCatalog,
    AirfoilV7TrimVariationCatalog,
    AirfoilV7UnionVariationCatalog,
)
from examples.benchmarks.heat2d_constructive.candidate import (
    SEED_LAYOUT_A,
    SEED_LAYOUT_B,
)
from examples.benchmarks.heat2d_constructive.multiobjective_v1 import (
    create_multiobjective_benchmark,
)
from examples.benchmarks.heat2d_constructive.problem_def import (
    Heat2DDirectV3Settings,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _object(value: dict[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(value)
    assert type(frozen) is FrozenJsonObject
    return frozen


class _ForbiddenBoilsEvaluator:
    def __init__(self) -> None:
        self.evaluate_calls = 0

    def evaluate(self, config):
        del config
        self.evaluate_calls += 1
        raise AssertionError("campaign portability must not execute ABC")


class _ForbiddenHeatEvaluator:
    def __init__(self) -> None:
        self.evaluate_calls = 0

    def evaluate(self, config):
        del config
        self.evaluate_calls += 1
        raise AssertionError("campaign portability must not execute a PDE solve")


class _ForbiddenAirfoilRawProblem:
    def __init__(self) -> None:
        self.evaluate_calls = 0

    def evaluate_raw(self, config):
        del config
        self.evaluate_calls += 1
        raise AssertionError("campaign portability must not execute Airfoil")


class _CountingPhenotypeIdentityPolicy:
    """Expose semantic-identity work without changing the identity law."""

    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.policy_id = delegate.policy_id
        self.policy_version = delegate.policy_version
        self.calls = 0

    def identify(self, configuration):
        self.calls += 1
        return self.delegate.identify(configuration)


class _RecordingEvidenceProjection:
    """Generic projection that consumes actual benchmark/catalog semantics."""

    def __init__(self) -> None:
        self.memory_calls = 0
        self.context_calls = 0
        self.card_calls = 0

    def initialize_memory(self, benchmark, session, seeds):
        self.memory_calls += 1
        descriptor = thaw_json(session.benchmark)
        return _object(
            {
                "schema_version": 1,
                "workload_id": descriptor["workload_id"],
                "objective_names": [item.name for item in benchmark.objectives],
                "seed_candidate_keys": [
                    benchmark.problem.candidate_key(thaw_json(seed.configuration))
                    for seed in seeds.seeds
                ],
                "insights": [],
            }
        )

    def context(self, benchmark, session, parent, variation, memory):
        self.context_calls += 1
        descriptor = thaw_json(session.benchmark)
        selected_catalog = descriptor["selected_finite_catalog"]
        return _object(
            {
                "schema_version": 1,
                "workload_id": descriptor["workload_id"],
                "search_space": benchmark.problem.search_space_description(),
                "parent_sha256": typed_json_sha256(parent),
                "catalog_id": selected_catalog["catalog_id"],
                "eligible_contract_catalog_id": variation.contract.catalog_id,
                "finite_option_count": len(variation.contract.options),
                "memory_sha256": typed_json_sha256(memory),
            }
        )

    def cards(self, benchmark, session, parent, variation, memory):
        del benchmark, parent, memory
        self.card_calls += 1
        descriptor = thaw_json(session.benchmark)
        selected_catalog = descriptor["selected_finite_catalog"]
        return tuple(
            _object(
                {
                    "schema_version": 1,
                    "workload_id": descriptor["workload_id"],
                    "catalog_id": selected_catalog["catalog_id"],
                    "eligible_contract_catalog_id": variation.contract.catalog_id,
                    "option": option.prompt_record(),
                }
            )
            for option in variation.contract.options[:2]
        )


class _Runtime:
    def __init__(self) -> None:
        self.requests = []

    def prepare(self, request):
        self.requests.append(request)
        return CampaignAgentRuntimeReceipt(
            request_sha256=request.request_sha256,
            runtime_id="provider_free_acceptance_runtime",
            runtime_version=1,
            definition_sha256=_sha("provider-free-acceptance-runtime"),
            accepted=True,
            evidence=_object({"provider_calls": 0, "accepted": True}),
        )


class _Journal:
    def __init__(self) -> None:
        self.records = []

    def append(self, record):
        self.records.append(record)


class _ArchiveUtility:
    utility_id = "portable_archive_utility"
    utility_version = 1
    definition_sha256 = _sha("portable-archive-utility")

    def freeze(self, *, benchmark, generation, archive):
        return ArchiveUtilitySnapshot(
            utility_id=self.utility_id,
            utility_version=self.utility_version,
            definition_sha256=self.definition_sha256,
            generation=generation,
            benchmark_sha256=typed_json_sha256(benchmark),
            archive_sha256=typed_json_sha256(archive),
            snapshot_receipt=_object({"utility": "test_only"}),
        )


def _policy(name: str) -> CampaignPolicyBinding:
    return CampaignPolicyBinding(
        implementation=object(),
        policy_id=name,
        policy_version=1,
        definition_sha256=_sha(f"policy:{name}"),
    )


def _shared_protocol() -> CampaignProtocol:
    return CampaignProtocol(
        protocol_id="portable_g3",
        protocol_version=1,
        definition_sha256=_sha("portable-g3-protocol"),
        outer_seed=20260716,
        generation_count=3,
        required_seed_count=2,
        parents_per_portfolio_generation=2,
        portfolio_width=4,
        recombinations_per_parent=2,
        reflections_per_recombination_generation=1,
    )


def _shared_policies() -> CampaignPolicies:
    return CampaignPolicies(
        cadence=AlternatingPortfolioRecombinationCadence(),
        parent_selection=_policy("portable_parent_selection"),
        memory_assignment=_policy("portable_memory_assignment"),
        portfolio_selection=_policy("portable_portfolio_selection"),
        recombination=_policy("portable_recombination"),
        reflection=_policy("portable_reflection"),
        archive_utility=_ArchiveUtility(),
    )


def _campaign_seed(seed_id: str, value: object) -> CampaignSeed:
    configuration = freeze_json(value)
    assert type(configuration) is FrozenJsonObject
    return CampaignSeed(seed_id=seed_id, configuration=configuration)


def _projection_binding(
    projection: _RecordingEvidenceProjection,
) -> AgenticCampaignEvidenceProjections:
    return AgenticCampaignEvidenceProjections(
        projection_id="real_benchmark_metadata_projection",
        projection_version=1,
        definition_sha256=_sha("real-benchmark-metadata-projection-v1"),
        initialize_memory=projection.initialize_memory,
        context=projection.context,
        cards=projection.cards,
    )


def _session_for_ports(ports):
    session = ports.benchmark.open(
        BenchmarkSessionRequest(
            protocol_sha256=_sha("binding-registry-protocol"),
            budget_sha256=_sha("binding-registry-budget"),
            outer_seed=20260716,
            requested_evaluator_concurrency=1,
        )
    )
    return session, ports.seeds.load(session)


def _with_counting_phenotype_policy(config):
    original = config.benchmark
    policy = _CountingPhenotypeIdentityPolicy(original.phenotype_identity)
    benchmark = AgenticBenchmark(
        problem=original.problem,
        reward=original.reward,
        detailed_evaluator=original.detailed_evaluator,
        outcome_relation=original.outcome_relation,
        optimization_semantics=original.optimization_semantics,
        action_semantics=original.action_semantics,
        phenotype_identity=policy,
        finite_variation_catalogs=original.finite_variation_catalogs,
        hypothesis_compiler=original.hypothesis_compiler,
        finite_action_set_compiler=original.finite_action_set_compiler,
    )
    return replace(config, benchmark=benchmark), policy


def _boils_workload():
    forbidden = _ForbiddenBoilsEvaluator()
    problem = BoilsAbcProblem(
        AbcEvaluatorSettings.current_circuit_panel(circuit_names=("log2",)),
        evaluator=forbidden,
    )
    benchmark = AgenticBenchmark(
        problem=problem,
        finite_variation_catalogs=(BoilsFiniteVariationCatalog(),),
    )
    first = list(DEFAULT_ACTION_SEQUENCE)
    second = list(DEFAULT_ACTION_SEQUENCE)
    second[0] = "rewrite"
    projection = _RecordingEvidenceProjection()
    config = AgenticCampaignWorkloadConfig(
        workload_id="boils-abc-log2",
        workload_version=1,
        definition_sha256=_sha("boils-abc-log2-campaign-workload"),
        benchmark=benchmark,
        seeds=(
            _campaign_seed("seed_default", {"sequence": first}),
            _campaign_seed("seed_rewrite0", {"sequence": second}),
        ),
        finite_catalog_id=FINITE_CATALOG_ID,
        evaluator_concurrency_cap=4,
        evaluator_preflight_receipt=_object(
            {"qualified": True, "evaluator": "boils_abc", "abc_calls": 0}
        ),
        resource_lease_receipt=_object(
            {"resource": "cpu_affinity_pool", "active": True}
        ),
        evidence=_projection_binding(projection),
    )
    return config, projection, forbidden


def _heat_workload(tmp_path: Path):
    forbidden = _ForbiddenHeatEvaluator()
    benchmark = create_multiobjective_benchmark(
        Heat2DDirectV3Settings(
            output_root=tmp_path / "heat2d",
            # The actual decoder has seven semantic aliases for seed A at this
            # coarse provider-free resolution, giving the portability test a
            # real coalescence witness without a PDE solve.
            resolution=11,
        ),
        evaluator=forbidden,
    )
    projection = _RecordingEvidenceProjection()
    config = AgenticCampaignWorkloadConfig(
        workload_id="engibench-heat2d-pareto-v1",
        workload_version=1,
        definition_sha256=_sha("engibench-heat2d-pareto-v1-campaign-workload"),
        benchmark=benchmark,
        seeds=(
            _campaign_seed("seed_layout_a", SEED_LAYOUT_A),
            _campaign_seed("seed_layout_b", SEED_LAYOUT_B),
        ),
        finite_catalog_id="heat2d_constructive_scalar_grid",
        evaluator_concurrency_cap=1,
        evaluator_preflight_receipt=_object(
            {"qualified": True, "evaluator": "heat2d_direct_v3", "pde_solves": 0}
        ),
        resource_lease_receipt=_object(
            {"resource": "serialized_pde_slot", "active": True}
        ),
        evidence=_projection_binding(projection),
    )
    return config, projection, forbidden


def _airfoil_workload():
    forbidden = _ForbiddenAirfoilRawProblem()
    problem = AirfoilV7Problem(raw_problem=forbidden)
    benchmark = AgenticBenchmark(
        problem=problem,
        reward=AIRFOIL_V7_REWARD_BINDING,
        detailed_evaluator=problem.detailed_evaluator,
        outcome_relation=AIRFOIL_V7_ARCHIVE_RELATION,
        action_semantics=AIRFOIL_V7_ACTION_SEMANTICS,
        phenotype_identity=AirfoilV7PhenotypeIdentityPolicy(),
        finite_variation_catalogs=(
            AirfoilV7ShapeVariationCatalog(),
            AirfoilV7TrimVariationCatalog(),
            AirfoilV7UnionVariationCatalog(),
        ),
    )
    neutral = {
        "representation_id": "external_bernstein_y_panel_v1",
        "upper_coefficients": [0.0] * 10,
        "lower_coefficients": [0.0] * 10,
        "alpha_deg": [2.5, 2.5, 2.5],
    }
    offset = {
        "representation_id": "external_bernstein_y_panel_v1",
        "upper_coefficients": [0.001] + [0.0] * 9,
        "lower_coefficients": [-0.001] + [0.0] * 9,
        "alpha_deg": [2.25, 2.5, 2.75],
    }
    projection = _RecordingEvidenceProjection()
    config = AgenticCampaignWorkloadConfig(
        workload_id="engibench-airfoil-v7",
        workload_version=7,
        definition_sha256=_sha("engibench-airfoil-v7-campaign-workload"),
        benchmark=benchmark,
        seeds=(
            _campaign_seed("seed_neutral", neutral),
            _campaign_seed("seed_offset", offset),
        ),
        finite_catalog_id="airfoil_v7_union",
        evaluator_concurrency_cap=1,
        evaluator_preflight_receipt=_object(
            {"qualified": True, "evaluator": "airfoil_v7", "solver_calls": 0}
        ),
        resource_lease_receipt=_object(
            {"resource": "serialized_airfoil_slot", "active": True}
        ),
        evidence=_projection_binding(projection),
    )
    return config, projection, forbidden


def _prepare_three_workloads(tmp_path: Path):
    workloads = (
        _boils_workload(),
        _heat_workload(tmp_path),
        _airfoil_workload(),
    )
    protocol = _shared_protocol()
    policies = _shared_policies()
    runtime = _Runtime()
    budget = OptimizerBudget(
        max_unique_evaluations=64,
        max_logical_llm_calls=16,
        max_generations=3,
    )
    concurrency = CampaignConcurrency(
        evaluator_concurrency=1,
        agent_concurrency=4,
        agent_queue_capacity=16,
    )

    prepared = []
    ports = []
    journals = []
    for config, projection, forbidden in workloads:
        workload_ports = config.build_ports()
        journal = _Journal()
        result = EvolutionCampaign(
            protocol=protocol,
            workload=workload_ports,
            policies=policies,
            runtime=runtime,
            budget=budget,
            concurrency=concurrency,
            journals=(journal,),
        ).prepare()
        prepared.append(result)
        ports.append(workload_ports)
        journals.append(journal)

    return workloads, protocol, runtime, tuple(prepared), tuple(ports), tuple(journals)


def test_same_campaign_prepares_three_real_workloads_without_optional_numpy(
    tmp_path: Path,
) -> None:
    workloads, protocol, runtime, prepared, _, journals = _prepare_three_workloads(
        tmp_path
    )

    # Preparation opens only the authenticated session and seed boundary.  In
    # particular, Heat2D preparation does not import NumPy or decode a field.
    for (_, projection, forbidden), journal in zip(workloads, journals, strict=True):
        assert projection.memory_calls == 0
        assert projection.context_calls == 0
        assert projection.card_calls == 0
        assert forbidden.evaluate_calls == 0
        assert len(journal.records) == 1

    assert len(runtime.requests) == 3
    assert len({item.schedule.schedule_sha256 for item in prepared}) == 1
    assert len({item.protocol.protocol_sha256 for item in prepared}) == 1
    assert all(item.protocol is protocol for item in prepared)
    assert len({item.benchmark_session.session_sha256 for item in prepared}) == 3
    assert len({item.workload_ports_sha256 for item in prepared}) == 3
    assert len({item.preparation_sha256 for item in prepared}) == 3


def test_real_semantic_catalog_cutoff_context_and_cards_across_three_workloads(
    tmp_path: Path,
) -> None:
    pytest.importorskip(
        "numpy",
        reason=(
            "the qualified Heat2D semantic phenotype decoder has an optional "
            "NumPy runtime dependency"
        ),
    )
    workloads, _, _, prepared, ports, _ = _prepare_three_workloads(tmp_path)
    observed_workloads = []
    observed_catalogs = []
    semantic_alias_counts = {}
    for (config, projection, forbidden), result, workload_ports in zip(
        workloads, prepared, ports, strict=True
    ):
        parent = result.seeds.seeds[0].configuration
        unfiltered = workload_ports.catalog.bind(
            result.benchmark_session.benchmark,
            parent,
            (),
        )
        assert unfiltered.eligibility_receipt is not None
        semantic_alias_counts[config.workload_id] = len(
            unfiltered.eligibility_receipt.alias_excluded_option_ids
        )
        known_child = unfiltered.contract.options[0]
        known_identity = config.benchmark.phenotype_identity.identify(
            thaw_json(known_child.child_configuration)
        )
        known = (known_identity.value_sha256,)
        variation = workload_ports.catalog.bind(
            result.benchmark_session.benchmark,
            parent,
            known,
        )
        assert variation.eligibility_receipt is not None
        assert variation.eligibility_receipt.known_phenotype_sha256s == known
        assert known_child.option_id in (
            variation.eligibility_receipt.known_excluded_option_ids
        )
        assert len(variation.contract.options) == len(unfiltered.contract.options) - 1
        assert known_child.option_id not in {
            option.option_id for option in variation.contract.options
        }
        memory = workload_ports.evidence.initialize_memory(
            result.benchmark_session,
            result.seeds,
        )
        context = workload_ports.evidence.context(
            result.benchmark_session,
            parent,
            variation,
            memory,
        )
        cards = workload_ports.evidence.cards(
            result.benchmark_session,
            parent,
            variation,
            memory,
        )

        context_record = thaw_json(context)
        card_records = tuple(thaw_json(card) for card in cards)
        observed_workloads.append(context_record["workload_id"])
        observed_catalogs.append(context_record["catalog_id"])
        assert context_record["finite_option_count"] == len(variation.contract.options)
        assert context_record["finite_option_count"] > 1
        assert context_record["eligible_contract_catalog_id"] == (
            "eligible_finite_variation"
        )
        assert len(cards) == 2
        assert all(card["workload_id"] == config.workload_id for card in card_records)
        assert all(
            card["catalog_id"] == config.finite_catalog_id for card in card_records
        )
        assert all(
            card["eligible_contract_catalog_id"] == "eligible_finite_variation"
            for card in card_records
        )
        assert projection.memory_calls == 1
        assert projection.context_calls == 1
        assert projection.card_calls == 1
        assert forbidden.evaluate_calls == 0

    assert tuple(observed_workloads) == (
        "boils-abc-log2",
        "engibench-heat2d-pareto-v1",
        "engibench-airfoil-v7",
    )
    assert tuple(observed_catalogs) == (
        "boils_abc_single_action",
        "heat2d_constructive_scalar_grid",
        "airfoil_v7_union",
    )
    assert semantic_alias_counts == {
        "boils-abc-log2": 0,
        "engibench-heat2d-pareto-v1": 7,
        "engibench-airfoil-v7": 0,
    }


def test_same_calibrated_k8_binding_and_prompt_crosses_three_real_workloads(
    tmp_path: Path,
) -> None:
    """Exercise the actual provider boundary inputs without a provider/evaluator."""

    pytest.importorskip(
        "numpy",
        reason="Heat2D semantic phenotype identity has an optional NumPy dependency",
    )
    workloads, _, _, prepared, ports, _ = _prepare_three_workloads(tmp_path)
    observed_metric_ids: dict[str, tuple[str, ...]] = {}
    legacy_prompt_sizes: dict[str, int] = {}
    projected_prompt_sizes: dict[str, int] = {}
    projected_prompt_sha256s: dict[str, str] = {}
    semantic_metadata_keys = {
        "boils-abc-log2": (
            "abc_commands_json",
            "position",
            "replacement_action",
        ),
        "engibench-heat2d-pareto-v1": ("locus", "target_value"),
        # Shape amplitude/mode and trim deltas are already carried verbatim by
        # mandatory family+description fields.  No optional metadata is needed.
        "engibench-airfoil-v7": (),
    }
    for index, ((config, _, forbidden), result, workload_ports) in enumerate(
        zip(workloads, prepared, ports, strict=True),
        start=1,
    ):
        parent = result.seeds.seeds[0].configuration
        variation = workload_ports.catalog.bind(
            result.benchmark_session.benchmark,
            parent,
            (),
        )
        semantics = config.benchmark.optimization_semantics
        if semantics is None:
            projection = None
            metric_ids = tuple(
                sorted(objective.name for objective in config.benchmark.objectives)
            )
            slate_objectives = equal_weight_slate_objectives(
                config.benchmark.objectives
            )
        else:
            projection = DecisionMetricProjection.from_optimization_semantics(semantics)
            metric_ids = projection.metric_ids
            slate_objectives = equal_weight_slate_objectives_from_decision_metrics(
                projection
            )
        card_key = f"card.portability.{index}"
        card = PortfolioCard(
            card_key=card_key,
            reference=InsightRef(InsightId(f"insight_portability_{index}"), 1),
            content_sha256=_sha(f"portable-card-content-{index}"),
            evidence_sha256=_sha(f"portable-card-evidence-{index}"),
            prompt_payload=_object(
                {"claim": ("Test one supplied mechanism without inventing an action.")}
            ),
        )
        caller_instruction = (
            f"CALLER-ONLY-INSTRUCTION-{config.workload_id}-MUST-NOT-RENDER"
        )
        request = PortfolioSelectionRequest(
            call_id=LLMCallId(f"call_portability_calibrated_{index}"),
            operation="select_calibrated_portfolio",
            instruction=caller_instruction,
            context=_object(
                {
                    "schema_version": 1,
                    "workload_id": config.workload_id,
                    "search_space": config.benchmark.problem.search_space_description(),
                    "parent_configuration_sha256": (
                        variation.parent_configuration_sha256
                    ),
                }
            ),
            finite_variation_contract=variation.contract,
            cards=(card,),
            portfolio_size=4,
            required_metric_ids=metric_ids,
            require_supporting_cards=True,
            require_pairwise_disjoint_parent_patches=False,
            max_output_tokens=384_000,
            temperature=0.0,
        )
        ledger = PortfolioOutcomeFeedbackLedger()
        scope = ForecastCalibrationScope(
            model_profile_sha256=_sha("shared-portability-model-profile"),
            prompt_definition_sha256=(CALIBRATED_PORTFOLIO_PROMPT_DEFINITION_SHA256),
            selector_policy_definition_sha256=(
                CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
            ),
            benchmark_sha256=variation.benchmark_sha256,
            session_sha256=result.benchmark_session.session_sha256,
        )
        factory = CalibratedCampaignBindingFactory(
            scope=scope,
            objectives=slate_objectives,
            ledger=ledger,
        )
        binding = factory.build(
            request=request,
            variation=variation,
            wave_index=1,
            frozen_archive_snapshot_sha256=_sha(
                f"portable-archive-{config.workload_id}"
            ),
        )
        coordinator = CalibratedPortfolioCampaignCoordinator()
        coordinator.register(request, binding)
        rendered = coordinator.render(request)

        projection_policy = FiniteOptionPromptProjectionPolicy(
            metadata_keys=semantic_metadata_keys[config.workload_id]
        )
        projected_factory = CalibratedCampaignBindingFactory(
            scope=replace(
                scope,
                prompt_definition_sha256=(
                    calibrated_portfolio_prompt_definition_sha256(projection_policy)
                ),
            ),
            objectives=slate_objectives,
            ledger=ledger,
            option_prompt_projection=projection_policy,
        )
        projected_binding = projected_factory.build(
            request=request,
            variation=variation,
            wave_index=1,
            frozen_archive_snapshot_sha256=_sha(
                f"portable-archive-{config.workload_id}"
            ),
        )
        projected_coordinator = CalibratedPortfolioCampaignCoordinator()
        projected_coordinator.register(request, projected_binding)
        projected_rendered = projected_coordinator.render(request)

        binding.require_request(request)
        assert coordinator.binding_for(request) is binding
        assert coordinator.registered_request_count == 1
        assert caller_instruction not in rendered
        assert binding.context.calibration_snapshot.observation_count == 0
        assert len(binding.option_evidence) == len(variation.contract.options)
        assert len(binding.option_evidence) >= 8
        assert request.required_metric_ids == metric_ids
        assert forbidden.evaluate_calls == 0
        assert binding.option_prompt_projection is None
        projection = projected_binding.option_prompt_projection
        assert projection is not None
        assert projection.policy_configuration_sha256 == (
            projection_policy.configuration_sha256
        )
        assert (
            projection.included_metadata_keys
            == (semantic_metadata_keys[config.workload_id])
        )
        assert projected_binding.binding_sha256 != binding.binding_sha256
        projected_binding.require_request(request)

        legacy_machine = json.loads(rendered.splitlines()[3])
        projected_machine = json.loads(projected_rendered.splitlines()[3])
        assert set(projected_machine) == set(legacy_machine) | {
            "option_prompt_projection",
            "prompt_definition_sha256",
        }
        assert legacy_machine["schema_version"] == 3
        assert projected_machine["schema_version"] == 4
        assert projected_machine["prompt_definition_sha256"] == (
            calibrated_portfolio_prompt_definition_sha256(projection_policy)
        )
        assert projected_machine["option_prompt_projection"] == (
            projection.to_prompt_contract_record()
        )
        assert projected_machine["input_binding_sha256"] == (
            projected_binding.binding_sha256
        )
        assert projected_machine["ordered_options"] == list(
            projected_binding.prompt_records_for(request)
        )
        assert legacy_machine["ordered_options"] == list(
            variation.contract.prompt_records()
        )
        assert projected_rendered.startswith(rendered.splitlines()[0])
        assert projected_rendered.endswith(rendered.splitlines()[-1])
        observed_metric_ids[config.workload_id] = metric_ids
        legacy_prompt_sizes[config.workload_id] = len(rendered.encode("utf-8"))
        projected_prompt_sizes[config.workload_id] = len(
            projected_rendered.encode("utf-8")
        )
        projected_prompt_sha256s[config.workload_id] = hashlib.sha256(
            projected_rendered.encode("utf-8")
        ).hexdigest()

    assert observed_metric_ids["boils-abc-log2"] == (
        "total_levels",
        "total_lut_count",
    )
    assert observed_metric_ids["engibench-heat2d-pareto-v1"] == (
        "material_fraction",
        "thermal_term",
    )
    airfoil_ids = observed_metric_ids["engibench-airfoil-v7"]
    assert any(value.startswith("objective:") for value in airfoil_ids)
    assert any(value.startswith("violation:") for value in airfoil_ids)
    assert legacy_prompt_sizes == {
        "boils-abc-log2": 104_023,
        "engibench-heat2d-pareto-v1": 46_481,
        "engibench-airfoil-v7": 19_259,
    }
    assert projected_prompt_sizes == {
        "boils-abc-log2": 49_019,
        "engibench-heat2d-pareto-v1": 34_207,
        "engibench-airfoil-v7": 16_673,
    }
    assert projected_prompt_sha256s == {
        "boils-abc-log2": (
            "4789134eee0a83020e9a37ee1e6c9a43cca6d12906da267ec65b2f243dfff9d8"
        ),
        "engibench-heat2d-pareto-v1": (
            "91b27962f6a647622473ff05901d4f774d48d2cfab7686f17a4c8a1990d9d3d9"
        ),
        "engibench-airfoil-v7": (
            "e5735d986ac780ae23416625acdfdaeb29d4d208b1db49c339997584e23d96ab"
        ),
    }
    assert {
        name: legacy_prompt_sizes[name] - projected_prompt_sizes[name]
        for name in semantic_metadata_keys
    } == {
        "boils-abc-log2": 55_004,
        "engibench-heat2d-pareto-v1": 12_274,
        "engibench-airfoil-v7": 2_586,
    }


def test_evidence_rejects_a_coherent_receipt_with_foreign_phenotype_hashes(
    tmp_path: Path,
) -> None:
    workloads, _, _, prepared, ports, _ = _prepare_three_workloads(tmp_path)
    config, projection, forbidden = workloads[0]
    result = prepared[0]
    workload_ports = ports[0]
    parent = result.seeds.seeds[0].configuration
    variation = workload_ports.catalog.bind(
        result.benchmark_session.benchmark,
        parent,
        (),
    )
    receipt = variation.eligibility_receipt
    assert receipt is not None
    first, second, *remainder = receipt.option_phenotypes
    tampered_receipt = replace(
        receipt,
        option_phenotypes=(
            replace(
                first,
                phenotype_identity_sha256=second.phenotype_identity_sha256,
            ),
            replace(
                second,
                phenotype_identity_sha256=first.phenotype_identity_sha256,
            ),
            *remainder,
        ),
    )
    tampered = replace(variation, eligibility_receipt=tampered_receipt)
    # The receipt is internally coherent and remains bound to the same eligible
    # contract/cutoff; only authority recomputation can detect the wrong law.
    tampered.__post_init__()
    memory = workload_ports.evidence.initialize_memory(
        result.benchmark_session,
        result.seeds,
    )

    with pytest.raises(ValueError, match="exact eligible view"):
        workload_ports.evidence.context(
            result.benchmark_session,
            parent,
            tampered,
            memory,
        )

    assert config.workload_id == "boils-abc-log2"
    assert projection.context_calls == 0
    assert forbidden.evaluate_calls == 0


def test_binding_registry_identifies_options_once_across_bind_context_and_cards(
    tmp_path: Path,
) -> None:
    del tmp_path
    base_config, projection, forbidden = _boils_workload()
    config, policy = _with_counting_phenotype_policy(base_config)
    ports = config.build_ports()
    session, seeds = _session_for_ports(ports)
    parent = seeds.seeds[0].configuration
    memory = ports.evidence.initialize_memory(session, seeds)

    variation = ports.catalog.bind(session.benchmark, parent, ())
    receipt = variation.eligibility_receipt
    assert receipt is not None
    option_count = len(receipt.option_phenotypes)
    assert option_count > 1
    assert policy.calls == option_count

    # An exact repeated bind returns the registry-issued object.  Applying a
    # different novelty cutoff reuses the same base contract/identity pass.
    assert ports.catalog.bind(session.benchmark, parent, ()) is variation
    known = (receipt.option_phenotypes[0].phenotype_identity_sha256,)
    filtered = ports.catalog.bind(session.benchmark, parent, known)
    assert filtered is ports.catalog.bind(session.benchmark, parent, known)
    assert policy.calls == option_count

    ports.evidence.context(session, parent, variation, memory)
    ports.evidence.cards(session, parent, variation, memory)
    ports.evidence.context(session, parent, filtered, memory)
    ports.evidence.cards(session, parent, filtered, memory)
    assert policy.calls == option_count
    assert projection.context_calls == 2
    assert projection.card_calls == 2
    assert forbidden.evaluate_calls == 0


def test_binding_registry_rejects_clones_foreign_ports_and_foreign_sessions() -> None:
    config, projection, forbidden = _boils_workload()
    ports = config.build_ports()
    session, seeds = _session_for_ports(ports)
    parent = seeds.seeds[0].configuration
    variation = ports.catalog.bind(session.benchmark, parent, ())
    memory = ports.evidence.initialize_memory(session, seeds)

    coherent_clone = replace(variation)
    coherent_clone.__post_init__()
    with pytest.raises(ValueError, match="exact eligible view"):
        ports.evidence.context(
            session,
            parent,
            coherent_clone,
            memory,
        )

    foreign_ports = config.build_ports()
    foreign_session, foreign_seeds = _session_for_ports(foreign_ports)
    foreign_variation = foreign_ports.catalog.bind(
        foreign_session.benchmark,
        parent,
        (),
    )
    assert foreign_variation == variation
    assert foreign_variation is not variation
    with pytest.raises(ValueError, match="exact eligible view"):
        ports.evidence.cards(
            session,
            parent,
            foreign_variation,
            memory,
        )
    with pytest.raises(ValueError, match="exact campaign port set"):
        ports.evidence.context(
            foreign_session,
            parent,
            variation,
            memory,
        )
    with pytest.raises(ValueError, match="exact campaign port set"):
        ports.evidence.initialize_memory(foreign_session, foreign_seeds)

    assert projection.context_calls == 0
    assert projection.card_calls == 0
    assert forbidden.evaluate_calls == 0


def test_public_bridge_has_no_model_provider_or_evaluator_dependency() -> None:
    import agent_evolve as public_api
    import agent_evolve.campaign_workload as module

    assert public_api.AgenticCampaignWorkloadConfig is AgenticCampaignWorkloadConfig
    assert (
        public_api.AgenticCampaignEvidenceProjections
        is AgenticCampaignEvidenceProjections
    )
    assert {
        "AgenticCampaignWorkloadConfig",
        "AgenticCampaignEvidenceProjections",
    }.issubset(public_api.__all__)

    source = Path(module.__file__).read_text(encoding="utf-8")
    forbidden_import_fragments = (
        "pydantic_ai",
        "openrouter",
        "BoilsAbcEvaluator",
        "DirectV3Evaluator",
        "AirfoilPanelEvaluator",
    )
    assert all(fragment not in source for fragment in forbidden_import_fragments)
