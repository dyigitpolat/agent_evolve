"""Provider/PDE-free conformance for Heat's delayed identifiable loop."""

from __future__ import annotations

import asyncio
from decimal import Decimal
import hashlib
import json
from types import SimpleNamespace

from agent_evolve.agentic import DeterministicIdFactory, InsightMemoryBank
from agent_evolve.application.campaign_learning_runtime import (
    CampaignReflectionLearningRecordCodec,
)
from agent_evolve.application.evolution_campaign import (
    CampaignReflectionWave,
    ReflectionLaunchMode,
    ReflectionVisibility,
)
from agent_evolve.application.identifiable_reflection_evidence import (
    project_identifiable_reflection_evidence,
)
from agent_evolve.application.insight_memory import ReflectedInsightBatchItem
from agent_evolve.application.portfolio_campaign_runtime import (
    CampaignIdentifiableReflectionEvidenceProjection,
    CampaignIdentifiableReflectionEvidenceQuery,
    CampaignIdentifiableReflectionInput,
)
from agent_evolve.domain.finite_variation import FiniteVariationContract
from agent_evolve.domain.ids import CandidateId, OperatorInvocationId
from agent_evolve.domain.typed_json import freeze_json, thaw_json
from agent_evolve.policies.memory.global_falsification import (
    AuthenticatedHypothesisObservation,
    CausalEstimandUnit,
    EvidenceCausalBoundary,
    EvidenceProvenance,
    InterventionIdentifiability,
    ObservedMetricEffect,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    InsightDraft,
    MetricComparisonAnchor,
    MetricComparisonAnchorKind,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionConsumerScope,
    ReflectionGenerationResult,
    ReflectionInsightKind,
)
from examples.benchmarks.heat2d_constructive.candidate import SEED_LAYOUT_A
from examples.benchmarks.heat2d_constructive.finite_variation_catalog import (
    CATALOG_DEFINITION_SHA256,
    CATALOG_ID,
    CATALOG_VERSION,
    Heat2DFiniteVariationCatalog,
)
from examples.development import run_heat2d_generic_campaign as campaign


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def test_prompt_identity_tracks_actual_memory_dose_not_generation() -> None:
    projection = campaign._option_prompt_projection()
    base = {
        "generation": campaign.FIRST_REFLECTION_CONSUMER_GENERATION,
        "allocator": campaign._default_allocator().to_record(),
        "prompt_definition_sha256": (
            campaign.calibrated_portfolio_prompt_definition_sha256(
                projection,
                bounded_memory_dose=False,
                feasibility_witness_mode=campaign.FEASIBILITY_WITNESS_MODE,
            )
        ),
        "selector_policy_definition_sha256": (
            campaign._selector_policy_definition_sha256()
        ),
        "option_prompt_projection_sha256": "1" * 64,
        "bounded_reflection_memory_dose": None,
    }
    assert campaign._calibrated_wave_prompt_composition_exact(base)

    bounded = {
        **base,
        "prompt_definition_sha256": (
            campaign.calibrated_portfolio_prompt_definition_sha256(
                projection,
                bounded_memory_dose=True,
                feasibility_witness_mode=campaign.FEASIBILITY_WITNESS_MODE,
            )
        ),
        "bounded_reflection_memory_dose": {"schema_version": 1},
    }
    assert campaign._calibrated_wave_prompt_composition_exact(bounded)
    assert not campaign._calibrated_wave_prompt_composition_exact(
        {**bounded, "prompt_definition_sha256": base["prompt_definition_sha256"]}
    )


def _observation(
    name: str,
    *,
    event_index: int,
    affected_path: str,
    option_family: str,
    parent: object,
    child: object,
) -> AuthenticatedHypothesisObservation:
    parent_configuration = freeze_json(parent)
    child_configuration = freeze_json(child)
    option_id = (
        "heat2d.l00.v03"
        if name == "g1_material"
        else f"heat2d.test.{name}"
    )
    occurrence_suffix = "g1" if event_index == 1 else "g3"
    finite_contract_sha256 = _sha(f"finite:{name}")
    compiler_sha256 = _sha("heat-test-action-semantics")
    return AuthenticatedHypothesisObservation(
        source_evidence_id=_sha(f"source:{name}"),
        event_index=event_index,
        workload_instance_sha256=_sha("heat-workload"),
        evaluator_contract_sha256=_sha("heat-evaluator"),
        campaign_sha256=_sha("heat-campaign"),
        parent_candidate_id=CandidateId(
            f"candidate_heat_parent_{occurrence_suffix}"
        ),
        child_candidate_id=CandidateId(
            f"candidate_heat_child_{occurrence_suffix}"
        ),
        operator_invocation_id=OperatorInvocationId(
            f"operator_heat_{occurrence_suffix}"
        ),
        finite_contract_identity_sha256=finite_contract_sha256,
        provenance=EvidenceProvenance.DIRECT_MUTATION,
        causal_boundary=EvidenceCausalBoundary(
            wave_sha256=_sha(f"wave:{name}"),
            estimand_unit=CausalEstimandUnit.WAVE,
        ),
        parent_configuration=parent_configuration,
        child_configuration=child_configuration,
        parent_configuration_sha256=(
            AuthenticatedHypothesisObservation.configuration_sha256(
                parent_configuration
            )
        ),
        child_configuration_sha256=(
            AuthenticatedHypothesisObservation.configuration_sha256(
                child_configuration
            )
        ),
        parent_outcome_sha256=_sha(f"parent-outcome:{name}"),
        child_outcome_sha256=_sha(f"child-outcome:{name}"),
        operator_family="typed_mutation",
        affected_paths=(affected_path,),
        observed_action=freeze_json(
            {
                "schema_version": 2,
                "option_id": option_id,
                "option_identity_sha256": _sha(f"option:{name}"),
                "finite_contract_identity_sha256": finite_contract_sha256,
                "option_family": option_family,
                "operator_family": "typed_mutation",
                "changed_paths": [affected_path],
                "compiler": {
                    "compiler_id": "heat_test_action_semantics",
                    "compiler_version": 1,
                    "definition_sha256": compiler_sha256,
                },
            }
        ),
        action_semantics_compiler_id="heat_test_action_semantics",
        action_semantics_compiler_version=1,
        action_semantics_definition_sha256=compiler_sha256,
        intervention_identifiability=InterventionIdentifiability.EXACT_SINGLE,
        metrics=tuple(
            ObservedMetricEffect(
                metric_id=metric_id,
                direction=MetricEffectDirection.DECREASE,
                delta=-0.01,
                adjudicator_definition_sha256=_sha("heat-adjudicator"),
            )
            for metric_id in campaign.OBJECTIVE_IDS
        ),
        lineage_cluster_sha256=_sha(f"cluster:{name}"),
        factorial_block_sha256=_sha(f"block:{name}"),
    )


def _input() -> CampaignIdentifiableReflectionInput:
    g1 = _observation(
        "g1_material",
        event_index=1,
        affected_path="$.material_fraction",
        option_family="material_fraction",
        parent={"material_fraction": 0.45, "ambient_marker": "sealed_g1"},
        child={"material_fraction": 0.41, "ambient_marker": "sealed_g1"},
    )
    ambient_g3 = _observation(
        "ambient_g3_secret",
        event_index=3,
        affected_path="$.central_hole.radius_x",
        option_family="subtractive_geometry",
        parent={"central_hole": {"radius_x": 0.10}, "ambient_marker": "leak_me"},
        child={"central_hole": {"radius_x": 0.08}, "ambient_marker": "leak_me"},
    )
    evidence = project_identifiable_reflection_evidence(
        (g1, ambient_g3),
        campaign_sha256=_sha("heat-campaign"),
        workload_instance_sha256=_sha("heat-workload"),
        evaluator_contract_sha256=_sha("heat-evaluator"),
        prior_cutoff_event_index_exclusive=0,
        sealed_cutoff_event_index_inclusive=1,
    )
    query = CampaignIdentifiableReflectionEvidenceQuery(
        reflection_request_sha256=_sha("heat-reflection-request"),
        preparation_sha256=_sha("heat-preparation"),
        runtime_start_receipt_sha256=_sha("heat-runtime-start"),
        campaign_sha256=_sha("heat-campaign"),
        workload_instance_sha256=_sha("heat-workload"),
        evaluator_contract_sha256=_sha("heat-evaluator"),
        wave=CampaignReflectionWave(
            source_generation=2,
            call_count=1,
            launch_mode=ReflectionLaunchMode.ASYNC_AFTER_STAGE_SEAL,
            visibility=ReflectionVisibility.QUARANTINED_UNTIL_BLOCK_CLOSE,
            promotion_barrier_generation=4,
        ),
        source_stage_receipt_sha256=_sha("heat-source-stage-g2"),
        source_portfolio_generation=1,
        prior_cutoff_event_index_exclusive=0,
        sealed_cutoff_event_index_inclusive=1,
    )
    return CampaignIdentifiableReflectionInput(
        query=query,
        source=CampaignIdentifiableReflectionEvidenceProjection(
            query_sha256=query.query_sha256,
            registry_snapshot_sha256=_sha("heat-registry-g1"),
            registry_captured_through_event_index=1,
            evidence=evidence,
        ),
    )


def _semantics():
    return campaign._heat_optimization_semantics(
        ("heat_test_pareto", 1, _sha("heat-test-pareto"))
    )


def _draft(
    contrast_ids: tuple[str, ...],
    *,
    recommended_option_id: str,
) -> InsightDraft:
    return InsightDraft(
        claim="Test lower material fraction as a local Pareto move.",
        trigger="A sealed material-fraction option is available.",
        mechanism="The local material reduction may preserve thermal quality.",
        affected_paths=("$.material_fraction",),
        evidence_summary="One authenticated G1 direct mutation improved both metrics.",
        confidence=0.7,
        evidence_contrast_ids=contrast_ids,
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
        recommended_option_ids=(recommended_option_id,),
        action_template="Select one compatible material-fraction action.",
        falsification_condition="A later matched action worsens either metric.",
        insight_kind=ReflectionInsightKind.EMPIRICAL_PREDICTIVE_RULE,
        consumer_scopes=(ReflectionConsumerScope.MUTATION_SELECTION,),
        factor_capabilities=("material_fraction",),
    )


class _Generator:
    def __init__(self) -> None:
        self.requests = []

    async def reflect(self, request):
        self.requests.append(request)
        assert request.evidence_catalog is not None
        return ReflectionGenerationResult(
            insights=(
                _draft(
                    request.available_contrast_ids,
                    recommended_option_id=request.insight_contract.allowed_option_ids[0],
                ),
            ),
            telemetry=AgenticCallTelemetry(
                requested_model=campaign.MODEL,
                resolved_model=campaign.MODEL,
                resolved_provider=campaign.RESOLVED_PROVIDER,
                provider_response_id="provider-free",
                finish_reason="tool_call",
                input_tokens=0,
                output_tokens=0,
                reasoning_tokens=1,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_usd=Decimal("0"),
                latency_ns=0,
            ),
            evidence_catalog_identity_sha256=(
                request.evidence_catalog.catalog_identity_sha256
            ),
        )


def _execute_provider_free():
    reflection_input = _input()
    generator = _Generator()
    records = []
    executor = campaign._ReflectionExecutor(
        generator=generator,
        ids=DeterministicIdFactory("heat_ident_test"),
        records=records,
        optimization_semantics=_semantics(),
    )
    envelope = asyncio.run(executor.reflect(reflection_input))
    return reflection_input, generator, records, envelope


def test_heat_executor_has_no_ambient_memory_or_recombination_input() -> None:
    reflection_input, generator, records, envelope = _execute_provider_free()

    assert len(generator.requests) == 1
    request = generator.requests[0]
    assert request.operation == "extract_identifiable_insights"
    assert len(request.available_contrast_ids) == 1
    assert request.insight_contract.allowed_option_ids == (
        reflection_input.evidence.contrasts[0].option_id,
    )
    assert "ambient_g3_secret" not in request.prompt
    assert "leak_me" not in request.prompt
    assert "recombination" not in request.prompt.casefold()
    prompt = json.loads(request.prompt)
    assert [
        row["affected_path"]
        for row in prompt["identifiable_mutation_contrasts"]
    ] == ["$.material_fraction"]
    assert reflection_input.evidence.contrasts[0].contrast_id not in request.prompt
    assert records[0]["source_portfolio_generation"] == 1
    assert records[0]["sealed_cutoff_event_index_inclusive"] == 1
    assert records[0]["source_stage_payload_exposed"] is False
    assert records[0]["recombination_results_exposed"] is False
    assert envelope == campaign._object(records[0])


def test_heat_learning_preserves_action_attribution_but_shifted_parent_is_advisory() -> None:
    reflection_input, _generator, _records, envelope = _execute_provider_free()
    learning = CampaignReflectionLearningRecordCodec.decode(envelope)
    contrast = reflection_input.evidence.contrasts[0]

    assert learning.origin_cutoff_event_index == 1
    assert learning.source_generation == 2
    assert learning.source_operator_invocation_ids == (
        contrast.operator_invocation_id,
    )
    assert learning.source_candidate_ids == tuple(
        sorted((contrast.parent_candidate_id, contrast.child_candidate_id))
    )
    assert len(learning.finite_action_bindings) == 1
    binding = learning.finite_action_bindings[0]
    assert binding.option_id == contrast.option_id
    assert binding.option_identity_sha256 == contrast.option_identity_sha256
    assert binding.contract_identity_sha256 == (
        contrast.finite_contract_identity_sha256
    )

    memory = InsightMemoryBank(
        id_factory=DeterministicIdFactory("heat_dose_test"),
    )
    entries = memory.add_reflection_batch(
        tuple(
            ReflectedInsightBatchItem(draft, learning.lineage_for(draft))
            for draft in learning.insights
        ),
        applicable_operator_kinds=("typed_mutation",),
    )
    entry = entries[0]
    parent = freeze_json(SEED_LAYOUT_A)
    finite = FiniteVariationContract(
        catalog_id=CATALOG_ID,
        catalog_version=CATALOG_VERSION,
        catalog_definition_sha256=CATALOG_DEFINITION_SHA256,
        parent_configuration=parent,
        options=Heat2DFiniteVariationCatalog.options(parent),
    )
    card = campaign._WaveFactory._reflection_card(
        entry,
        SimpleNamespace(receipt_sha256=_sha("heat-g4-admission")),
        card_key="card.01",
        assigned_score=0.0,
        finite_variation_contracts=(finite,),
    )
    dose = campaign._WaveFactory._bounded_reflection_memory_dose(
        cards=(card,),
        selected_entries=(entry,),
        finite_contract=finite,
    )
    # The reflected transition was observed on a synthetic G1 source parent,
    # not this complete Heat seed.  Matching the local material value is useful
    # advisory evidence, but cannot authorize forced replay on a different
    # full configuration.
    assert dose is None
    assert thaw_json(card.prompt_payload)["hypothesis"]["affected_paths"] == [
        "$.material_fraction"
    ]
