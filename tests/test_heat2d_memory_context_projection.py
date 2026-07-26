"""Provider- and PDE-free gate for all preregistered Heat portfolio waves."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_evolve.agentic import (
    DeterministicIdFactory,
    InsightMemoryBank,
    MetricComparisonAnchor,
    MetricComparisonAnchorKind,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionConsumerScope,
    ReflectionInsightKind,
    eligible_finite_variation_view,
    exact_configuration_phenotype_bindings,
)
from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.calibrated_campaign import (
    CalibratedCampaignBindingFactory,
    equal_weight_slate_objectives,
)
from agent_evolve.application.evolution_campaign import ParentVariationBinding
from agent_evolve.application.portfolio_outcome_feedback import (
    PortfolioOutcomeFeedbackLedger,
)
from agent_evolve.application.portfolio_memory_matched_control import (
    PortfolioMemoryMatchedSupportResolution,
)
from agent_evolve.application.insight_memory import (
    InsightEvidenceLineage,
    ReflectedInsightBatchItem,
)
from agent_evolve.domain.ids import CandidateId, LLMCallId, OperatorInvocationId
from agent_evolve.domain.finite_variation import FiniteActionEvidenceBinding
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import thaw_json, typed_json_sha256
from agent_evolve.integrations.pydantic_ai.calibrated_portfolio_campaign import (
    CalibratedPortfolioCampaignCoordinator,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    PydanticAIAgenticGenerator,
    REFLECTION_SEMANTIC_WIRE_CONTRACT_REVISION,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)
from agent_evolve.policies.reward.affine_hypervolume import (
    AffineHypervolumeArchiveUtility,
)
from agent_evolve.policies.selection.forecast_calibration import (
    ForecastCalibrationScope,
)
from examples.benchmarks.heat2d_constructive.candidate import (
    SEED_LAYOUT_A,
    SEED_LAYOUT_B,
)
from examples.benchmarks.heat2d_constructive.finite_variation_catalog import (
    CATALOG_ID,
)
from examples.benchmarks.heat2d_constructive.multiobjective_v1 import (
    MATERIAL_OBJECTIVE_NAME,
    THERMAL_OBJECTIVE_NAME,
    create_multiobjective_benchmark,
)
from examples.benchmarks.heat2d_constructive.problem_def import (
    Heat2DDirectV3Settings,
)
from examples.development import run_heat2d_generic_campaign as campaign


class _ForbiddenEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, config: object):
        del config
        self.calls += 1
        raise AssertionError("provider-free wave construction invoked the PDE")


class _ReflectionConstructionReached(Exception):
    """Sentinel proving construction reached the injected generator port."""


class _CaptureReflectionGenerator:
    def __init__(self) -> None:
        self.request = None

    async def reflect(self, request):
        self.request = request
        raise _ReflectionConstructionReached


class _OfflineStructuredReflectionRunner:
    """Exercise the real typed adapter while replacing only provider transport."""

    def __init__(self) -> None:
        self.requests: list[StructuredGenerationRequest[object]] = []

    async def __call__(self, request):
        self.requests.append(request)
        output = request.output_type.model_validate(
            {
                "insights": [
                    {
                        "claim": (
                            "Reducing material can preserve thermal quality for this "
                            "recombination context."
                        ),
                        "trigger": (
                            "A material-fraction recombination has comparable parent "
                            "thermal terms."
                        ),
                        "mechanism": (
                            "The lower material target retains the parent geometry."
                        ),
                        "affected_paths": ["$.material_fraction"],
                        "evidence_summary": (
                            "The authenticated contrast improved both measured metrics."
                        ),
                        "evidence_citation_keys": ["e0001"],
                        "confidence": 0.75,
                        "insight_kind": "empirical_predictive_rule",
                        "consumer_scopes": ["mutation_selection"],
                        "factor_capabilities": ["material_fraction"],
                        "effect_predictions": [
                            {
                                "metric_id": metric_id,
                                "direction": "decrease",
                                "comparison_anchor": {
                                    "kind": "current_parent",
                                },
                            }
                            for metric_id in campaign.OBJECTIVE_IDS
                        ],
                        "recommended_option_families": ["material_fraction"],
                        "action_template": (
                            "Apply the sealed material-fraction recombination."
                        ),
                        "falsification_condition": (
                            "A held-out matched recombination worsens either metric."
                        ),
                    }
                ]
            },
            strict=True,
        )


class _NoSharedSupportResolver:
    """Force the exact typed recourse observed in the first real Heat run."""

    def resolve(self, *, lanes, cards, selection_key_sha256):
        del cards
        return PortfolioMemoryMatchedSupportResolution(
            lane_ids=tuple(sorted(value.lane.lane_id for value in lanes)),
            eligible_card_keys=(),
            selected_card_key=None,
            selected_lane_supports=(),
            selection_key_sha256=selection_key_sha256,
        )
        return StructuredGenerationResponse(
            value=output,
            requested_model=campaign.MODEL,
            resolved_model=campaign.MODEL,
            resolved_provider=campaign.RESOLVED_PROVIDER,
            provider_response_id="provider-free-reflection",
            finish_reason="tool_call",
            input_tokens=100,
            output_tokens=50,
            reasoning_tokens=25,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1,
        )


def _reflection_source_result(contrast_id: str):
    parent = SimpleNamespace(
        objective_map={
            MATERIAL_OBJECTIVE_NAME: 0.45,
            THERMAL_OBJECTIVE_NAME: 0.00030,
        }
    )
    candidate = SimpleNamespace(
        objective_map={
            MATERIAL_OBJECTIVE_NAME: 0.38,
            THERMAL_OBJECTIVE_NAME: 0.00028,
        }
    )
    member = SimpleNamespace(
        outcome_sha256=contrast_id,
        selection_role="exploit",
        source_option_ids=("heat2d.l00.v00",),
        source_families=("material_fraction",),
        operator_invocation_id=OperatorInvocationId("operator_heat_reflection_source"),
        target_candidate_id=CandidateId("candidate_heat_reflection_source"),
    )
    outcome = SimpleNamespace(
        candidate=candidate,
        prepared=SimpleNamespace(plan=SimpleNamespace(parents=(parent,))),
        reward=0.25,
        dominates_any_parent=True,
        better_than_any_parent=True,
    )
    return SimpleNamespace(
        receipt=SimpleNamespace(
            branches=(SimpleNamespace(family="material_fraction"),),
            members=(member,),
        ),
        outcomes=(outcome,),
    )


def _parent(
    *,
    candidate_id: str,
    configuration: dict[str, object],
    proposal_sequence: int,
    thermal: float,
) -> EvolutionCandidate:
    frozen = campaign._object(configuration)
    configuration_sha256 = typed_json_sha256(frozen)
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId(candidate_id),
            configuration_hash=configuration_sha256,
            configuration_artifact_hash=configuration_sha256,
            proposal_sequence=proposal_sequence,
        ),
        configuration=frozen,
        objectives=tuple(
            sorted(
                (
                    (
                        MATERIAL_OBJECTIVE_NAME,
                        float(configuration["material_fraction"]),
                    ),
                    (THERMAL_OBJECTIVE_NAME, thermal),
                )
            )
        ),
        valid=True,
        generation=0,
        label=candidate_id,
    )


def test_planned_heat_reflection_is_g1_identifiable_and_provider_free() -> None:
    parents = (
        _parent(
            candidate_id="candidate_heat_prepare_probe_a",
            configuration=SEED_LAYOUT_A,
            proposal_sequence=0,
            thermal=0.00025,
        ),
        _parent(
            candidate_id="candidate_heat_prepare_probe_b",
            configuration=SEED_LAYOUT_B,
            proposal_sequence=1,
            thermal=0.00035,
        ),
    )

    probe = campaign._provider_free_reflection_construction_probe(parents)

    assert probe == campaign._provider_free_reflection_construction_probe(parents)
    assert probe["provider_calls"] == 0
    assert probe["credential_read"] is False
    assert probe["pde_solves"] == 0
    assert probe["planned_source_generations"] == [2]
    assert probe["sealed_source_portfolio_generations"] == [1]
    assert probe["promotion_barrier_generations"] == [4]
    assert probe["first_consumer_generation"] == 5
    assert probe["terminal_reflection"] is False
    assert probe["constructed_reflection_request_count"] == 1
    assert probe["exact_generation_coverage"] is True
    assert probe["exact_evidence_citation_mapping_every_request"] is True
    assert probe["no_legacy_evidence_key_every_request"] is True
    assert probe["exact_contract_identity_every_request"] is True
    assert probe["all_request_identities_unique"] is True
    assert probe["all_prompt_identities_unique"] is True
    assert probe["all_catalog_identities_unique"] is True
    assert probe["all_acceptance_gates_pass"] is True

    rows = probe["rows"]
    assert [row["source_generation"] for row in rows] == [2]
    assert len({row["request_identity_sha256"] for row in rows}) == 1
    assert len({row["prompt_sha256"] for row in rows}) == 1
    assert len({row["evidence_catalog_identity_sha256"] for row in rows}) == 1
    for row in rows:
        assert row["full_contrast_ids_exposed_to_model"] is False
        assert [
            entry["citation_key"] for entry in row["evidence_citation_mapping"]
        ] == ["e0001", "e0002"]


@pytest.mark.parametrize("force_no_shared_support", [False, True])
def test_all_six_g6_heat_waves_register_exact_calibrated_prompts_provider_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    force_no_shared_support: bool,
) -> None:
    if force_no_shared_support:
        monkeypatch.setattr(
            campaign,
            "PortfolioMemoryMatchedSupportResolver",
            _NoSharedSupportResolver,
        )
    forbidden = _ForbiddenEvaluator()
    benchmark = create_multiobjective_benchmark(
        Heat2DDirectV3Settings(output_root=tmp_path, resolution=41),
        evaluator=forbidden,
    )
    ids = DeterministicIdFactory("heat_projection_provider_free")
    memory = InsightMemoryBank(
        id_factory=ids,
        exploration_probability=Fraction(1, 1),
    )
    campaign._seed_memory(memory)
    plan = campaign._memory_plan(memory)
    contrast_id = "e" * 64
    reflected_draft = campaign.InsightDraft(
        claim="Test a compatible material-fraction action prospectively.",
        trigger="A sealed material-fraction option is available.",
        mechanism="The local material reduction may preserve thermal quality.",
        affected_paths=("$.material_fraction",),
        evidence_summary="Provider-free reflected-card factory fixture.",
        confidence=0.7,
        evidence_contrast_ids=(contrast_id,),
        effect_predictions=tuple(
            MetricEffectPrediction(
                metric_id=metric_id,
                direction=MetricEffectDirection.DECREASE,
                comparison_anchor=MetricComparisonAnchor(
                    MetricComparisonAnchorKind.CURRENT_PARENT
                ),
            )
            for metric_id in campaign.OBJECTIVE_IDS
        ),
        recommended_option_families=("material_fraction",),
        recommended_option_ids=("heat2d.l00.v00",),
        action_template="Select one compatible material-fraction action.",
        falsification_condition="A matched action worsens either metric.",
        insight_kind=ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
        consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
        factor_capabilities=("material_fraction",),
    )
    second_contrast_id = "d" * 64
    reflected_drafts = (
        reflected_draft,
        replace(
            reflected_draft,
            claim="Test a second compatible material-fraction action prospectively.",
            evidence_contrast_ids=(second_contrast_id,),
            recommended_option_ids=("heat2d.l00.v01",),
        ),
    )
    reflected_lineages = (
        InsightEvidenceLineage(
            reflection_call_id=LLMCallId("call_heat_projection_reflection"),
            source_operator_invocation_ids=(
                OperatorInvocationId("operator_heat_projection_reflection"),
            ),
            source_candidate_ids=(CandidateId("candidate_heat_projection_source"),),
            available_contrast_ids=(second_contrast_id, contrast_id),
            cited_contrast_ids=(contrast_id,),
            finite_action_bindings=(
                FiniteActionEvidenceBinding(
                    contrast_id=contrast_id,
                    option_id="heat2d.l00.v00",
                    family="material_fraction",
                    option_identity_sha256="1" * 64,
                    contract_identity_sha256="2" * 64,
                ),
            ),
        ),
        InsightEvidenceLineage(
            reflection_call_id=LLMCallId("call_heat_projection_reflection"),
            source_operator_invocation_ids=(
                OperatorInvocationId("operator_heat_projection_reflection"),
            ),
            source_candidate_ids=(CandidateId("candidate_heat_projection_source"),),
            available_contrast_ids=(second_contrast_id, contrast_id),
            cited_contrast_ids=(second_contrast_id,),
            finite_action_bindings=(
                FiniteActionEvidenceBinding(
                    contrast_id=second_contrast_id,
                    option_id="heat2d.l00.v01",
                    family="material_fraction",
                    option_identity_sha256="3" * 64,
                    contract_identity_sha256="4" * 64,
                ),
            ),
        ),
    )
    reflected_entries = memory.add_reflection_batch(
        tuple(
            ReflectedInsightBatchItem(draft, lineage)
            for draft, lineage in zip(
                reflected_drafts,
                reflected_lineages,
                strict=True,
            )
        ),
        applicable_operator_kinds=("typed_mutation",),
    )
    reflected_references = tuple(value.reference for value in reflected_entries)
    memory_admission = memory.admit_quarantine_test_assignment(
        reflected_references,
        operator_kind="typed_mutation",
        source_admission_request_sha256="f" * 64,
        editable_paths=campaign.REFLECTION_DECISION_PATHS,
    )
    exposure_sha256 = "a" * 64
    exposure = SimpleNamespace(
        receipt_sha256=exposure_sha256,
        barrier_generation=2,
        references=reflected_references,
        memory_admission=memory_admission,
    )
    learning_runtime = SimpleNamespace(
        diagnostic_exposures=lambda values: (
            (exposure,) if values == (exposure_sha256,) else ()
        )
    )
    evidence = campaign._Evidence(memory, plan)
    utility = AffineHypervolumeArchiveUtility(campaign._affine_spec())
    ledger = PortfolioOutcomeFeedbackLedger()
    allocator = campaign._default_allocator()
    option_prompt_projection = campaign._option_prompt_projection()
    coordinator = CalibratedPortfolioCampaignCoordinator(allocator=allocator)
    benchmark_record = campaign._object({"workload": "heat-projection-test"})
    binding_factory = CalibratedCampaignBindingFactory(
        scope=ForecastCalibrationScope(
            model_profile_sha256="1" * 64,
            prompt_definition_sha256=(
                campaign.calibrated_portfolio_prompt_definition_sha256(
                    option_prompt_projection
                )
            ),
            selector_policy_definition_sha256=(
                campaign.FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
            ),
            benchmark_sha256=typed_json_sha256(benchmark_record),
            session_sha256="2" * 64,
        ),
        objectives=equal_weight_slate_objectives(benchmark.objectives),
        ledger=ledger,
        option_prompt_projection=option_prompt_projection,
    )
    bounded_dose_binding_factory = CalibratedCampaignBindingFactory(
        scope=ForecastCalibrationScope(
            model_profile_sha256="1" * 64,
            prompt_definition_sha256=(
                campaign.calibrated_portfolio_prompt_definition_sha256(
                    option_prompt_projection,
                    bounded_memory_dose=True,
                )
            ),
            selector_policy_definition_sha256=(
                campaign.FULL_SUPPORT_CALIBRATED_PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
            ),
            benchmark_sha256=typed_json_sha256(benchmark_record),
            session_sha256="2" * 64,
        ),
        objectives=equal_weight_slate_objectives(benchmark.objectives),
        ledger=ledger,
        option_prompt_projection=option_prompt_projection,
    )
    records: list[dict[str, object]] = []
    factory = campaign._WaveFactory(
        ids=ids,
        memory=memory,
        plan=plan,
        utility=utility,
        binding_factory=binding_factory,
        coordinator=coordinator,
        records=records,
        bounded_dose_binding_factory=bounded_dose_binding_factory,
        learning_runtime=learning_runtime,
    )
    parents = (
        _parent(
            candidate_id="candidate_heat_seed_a",
            configuration=SEED_LAYOUT_A,
            proposal_sequence=0,
            thermal=0.00025,
        ),
        _parent(
            candidate_id="candidate_heat_seed_b",
            configuration=SEED_LAYOUT_B,
            proposal_sequence=1,
            thermal=0.00035,
        ),
    )
    initial_memory_projection = campaign._object(
        {
            "memory_stratum_sha256": campaign.MEMORY_CONTEXT_SHA256,
            "assignment_plan_sha256": plan.receipt_sha256,
            "trial_count": 0,
        }
    )
    evidence_cards = evidence.cards(
        None,
        None,
        None,
        None,
        initial_memory_projection,
    )
    archive = campaign._object(
        {
            "front_candidates": [
                {
                    "objectives": [
                        {"metric_id": name, "value_hex": value.hex()}
                        for name, value in parent.objectives
                    ]
                }
                for parent in parents
            ]
        }
    )
    variations = []
    for parent in parents:
        base_contract = benchmark.bind_finite_variation(
            CATALOG_ID,
            thaw_json(parent.configuration),
        )
        eligible = eligible_finite_variation_view(
            contract=base_contract,
            option_phenotypes=exact_configuration_phenotype_bindings(base_contract),
            known_phenotype_sha256s=(),
        )
        variations.append(
            ParentVariationBinding(
                benchmark_sha256=typed_json_sha256(benchmark_record),
                parent_configuration_sha256=typed_json_sha256(parent.configuration),
                known_phenotype_sha256s=(),
                contract=eligible.contract,
                eligibility_receipt=eligible.receipt,
            )
        )
    waves = []
    for generation in campaign.PORTFOLIO_GENERATIONS:
        memory_projection = campaign._object(
            {
                "memory_stratum_sha256": campaign.MEMORY_CONTEXT_SHA256,
                "assignment_plan_sha256": plan.receipt_sha256,
                "trial_count": 0 if generation == 1 else 2,
                "last_portfolio_generation": None if generation == 1 else 1,
            }
        )
        snapshot = utility.freeze(
            benchmark=benchmark_record,
            generation=generation,
            archive=archive,
        )
        generation_contexts = []
        for parent_slot, (parent, variation) in enumerate(
            zip(parents, variations, strict=True)
        ):
            evidence_context = evidence.context(
                None,
                None,
                thaw_json(parent.configuration),
                variation,
                memory_projection,
            )
            context = SimpleNamespace(
                stage_request=SimpleNamespace(
                    step=SimpleNamespace(generation=generation),
                    archive_utility=snapshot,
                    test_eligible_reflection_receipt_sha256s=(
                        (exposure_sha256,) if generation == 5 else ()
                    ),
                ),
                parent_slot=parent_slot,
                parent=parent,
                variation=variation,
                evidence_context=evidence_context,
                evidence_cards=evidence_cards,
            )
            diagnostic_coordinator = factory.diagnostic_coordinator
            assert diagnostic_coordinator is not None
            projected_estimand = diagnostic_coordinator.project(context)
            if projected_estimand is not None:
                if generation == 5 and parent_slot == 0:
                    with pytest.raises(
                        ValueError,
                        match="sealed diagnostic cohort",
                    ):
                        factory.build(context)
                projected_context = thaw_json(context.evidence_context)
                projected_context["memory_estimand_stratum_sha256"] = (
                    projected_estimand.estimand_stratum_sha256
                )
                projected_context["memory_estimand_context"] = thaw_json(
                    projected_estimand.estimand_context
                )
                context.evidence_context = campaign._object(projected_context)
            generation_contexts.append(context)
        waves.extend(factory.build_batch(tuple(generation_contexts)))

    assert forbidden.calls == 0
    assert len(waves) == len(records) == 6
    assert coordinator.registered_request_count == 6
    assert [(wave.generation, index % 2) for index, wave in enumerate(waves)] == [
        (1, 0),
        (1, 1),
        (3, 0),
        (3, 1),
        (5, 0),
        (5, 1),
    ]
    selector_hashes = set()
    binding_hashes = set()
    prompt_hashes = set()
    for wave_index, wave in enumerate(waves):
        credit = wave.memory_credit
        matched = wave.matched_memory_control
        projection = None
        if wave.generation == 5:
            assert credit is None
            if force_no_shared_support:
                assert matched is None
            else:
                assert matched is not None
                projection = matched.context_projection
        else:
            assert credit is not None
            assert matched is None
            projection = credit.resolve_context_projection(
                wave.selection_request.context
            )
        if projection is not None:
            projected = projection.replay(wave.selection_request.context)
            expected_context_sha256 = (
                campaign.MEMORY_CONTEXT_SHA256
                if wave.generation != 5
                else records[wave_index]["diagnostic_reflection_exposure"][
                    "estimand_context_sha256"
                ]
            )
            assert typed_json_sha256(projected) == expected_context_sha256
            if credit is not None:
                assert (
                    projection.estimand_context_sha256
                    == credit.decision.context_hash
                )
            else:
                assert matched is not None
                assert (
                    projection.estimand_context_sha256
                    == matched.plan.exact_context_sha256
                )
            assert projection.selector_context_sha256 == (
                wave.selection_request.context_sha256
            )
            if credit is not None:
                assert projection.binding_sha256 == (
                    credit.context_projection.binding_sha256
                )
            selector_hashes.add(projection.selector_context_sha256)
        else:
            selector_hashes.add(wave.selection_request.context_sha256)
        calibrated = coordinator.binding_for(wave.selection_request)
        calibrated.require_request(wave.selection_request)
        assert calibrated.option_prompt_projection is not None
        assert (
            calibrated.option_prompt_projection.policy_configuration_sha256
            == option_prompt_projection.configuration_sha256
        )
        assert coordinator.allocator.to_record() == allocator.to_record()
        binding_hashes.add(calibrated.binding_sha256)
        prompt = coordinator.render(wave.selection_request)
        prompt_hashes.add(campaign._sha(prompt))
        assert campaign._sha(prompt) == records[wave_index]["calibrated_prompt_sha256"]
        assert records[wave_index]["proposal_width"] == 8
        assert records[wave_index]["evaluation_width"] == 8
    assert len(selector_hashes) >= 2
    assert len({wave.selection_request.request_sha256 for wave in waves}) == 6
    assert len(binding_hashes) == 6
    assert len(prompt_hashes) == 6
    diagnostic_rows = [
        value
        for value in records
        if value["diagnostic_reflection_exposure"] is not None
    ]
    if force_no_shared_support:
        assert diagnostic_rows == []
        g5_waves = tuple(wave for wave in waves if wave.generation == 5)
        assert all(wave.memory_credit is None for wave in g5_waves)
        assert all(wave.matched_memory_control is None for wave in g5_waves)
        assert all(
            wave.selection_request.memory_dose_contract is None
            for wave in g5_waves
        )
        assert all(len(wave.selection_request.cards) == 2 for wave in g5_waves)
        g5_records = tuple(value for value in records if value["generation"] == 5)
        assert all(
            value["resolved_memory_assignment"] is None
            and value["resolved_memory_assignment_sha256"] is None
            and value["diagnostic_recourse"]["status"]
            == "no_shared_support_active_neutral_card"
            and value["diagnostic_recourse"]["memory_credit_issued"] is False
            for value in g5_records
        )
        assert len(factory.matched_support_resolutions) == 1
        assert not factory.matched_support_resolutions[0].eligible
        assert factory.matched_control_plans == []
        assert len(factory.matched_control_recourses) == 1
        return
    assert [value["generation"] for value in diagnostic_rows] == [5, 5]
    assert all(
        value["diagnostic_reflection_exposure"]["exposure_receipt_sha256"]
        == exposure_sha256
        for value in diagnostic_rows
    )
    diagnostic_record = factory._diagnostic_blocks[5].to_record()
    assert diagnostic_record["reflection_exposure_receipt_sha256"] == (
        exposure_sha256
    )
    assert diagnostic_record["estimand_context_sha256"] == diagnostic_rows[0][
        "diagnostic_reflection_exposure"
    ]["estimand_context_sha256"]
    assert len(diagnostic_record["eligible_references"]) >= 2
    assert diagnostic_record["full_block_permutation_rank"] in (0, 1)
    assert all(
        len(wave.selection_request.cards) == 1
        for wave in waves
        if wave.generation == 5
    )
    g5_matched = tuple(
        wave.matched_memory_control for wave in waves if wave.generation == 5
    )
    assert all(value is not None for value in g5_matched)
    assert {value.assignment.arm.value for value in g5_matched} == {"m", "n"}
    assert len({value.plan.plan_sha256 for value in g5_matched}) == 1
    assert all(wave.memory_credit is None for wave in waves if wave.generation == 5)
    assert all(
        wave.selection_request.memory_dose_contract is None
        for wave in waves
        if wave.generation in (1, 3)
    )
    g5_doses = tuple(
        wave.selection_request.memory_dose_contract
        for wave in waves
        if wave.generation == 5
    )
    assert sum(value is not None for value in g5_doses) == 1
    assert all(
        value.proposed_supported_member_bounds == (1, 1)
        and value.evaluated_supported_member_bounds == (1, 1)
        and value.minimum_unattributed_proposed_members == 7
        and value.minimum_unattributed_evaluated_members == 7
        for value in g5_doses
        if value is not None
    )
    assert len(factory.matched_support_resolutions) == 1
    assert factory.matched_support_resolutions[0].eligible
    assert len(factory.matched_control_plans) == 1
    assert factory.matched_control_recourses == []
