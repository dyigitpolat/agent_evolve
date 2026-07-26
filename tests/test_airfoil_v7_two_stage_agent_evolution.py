from __future__ import annotations

from decimal import Decimal
from dataclasses import replace
import hashlib
import json

import pytest

from agent_evolve.application.reflection_workflow import (
    ReflectionShardResult,
    ReflectionWorkflowResult,
)
from agent_evolve.application.action_allocation_frame import (
    AuditedGreedyForecastFrameAllocator,
)
from agent_evolve.application.action_allocation_frame_commit import (
    build_frame_action_allocation_phase_commit,
)
from agent_evolve.application.action_allocation_frame_commit_v3 import (
    build_operational_frame_action_allocation_phase_commit,
)
from agent_evolve.application.action_allocation_frame_v3 import (
    OperationalGreedyForecastFrameAllocator,
)
from agent_evolve.application.paired_allocation_comparison import (
    AllocationComparisonMethodWave,
    build_paired_allocation_comparison_commitment,
)
from agent_evolve.application.action_forecast_partitioning import (
    build_action_forecast_block_requests,
    build_action_forecast_partition_layout,
)
from agent_evolve.application.treatment_assignment import (
    assign_treatment_occurrences,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.application.two_stage_action_evolution import (
    TwoStageActionPhase,
    TwoStageActionPhaseCommit,
    TwoStageActionPhaseReceipt,
)
from agent_evolve.domain.typed_json import freeze_json, thaw_json, typed_json_sha256
from agent_evolve.ports.action_forecast import ActionForecastEvidenceMode
from agent_evolve.ports.action_forecast import (
    ActionEvidenceCitation,
    ActionForecastDraft,
    ActionForecastPartitionPolicyBinding,
    ActionMetricForecast,
    ResolvedActionForecast,
    ResolvedActionMetricForecast,
    resolve_action_forecast_block,
)
from agent_evolve.ports.action_allocation import (
    ForecastPortfolioUtilityInput,
    ForecastQuantile,
)
from agent_evolve.ports.action_allocation_frame import (
    ActionAllocationFrameSubsetPolicyBinding,
    AllocationCandidateScoreDiagnostic,
    AllocationCandidateScoreDiagnosticInput,
    AllocationScoreDiagnosticBinding,
    AllocationSurfaceGatePolicyBinding,
    FrameActionAllocationRequest,
    bind_action_forecast_block_subset_allocation_frame,
)
from agent_evolve.ports.action_allocation_frame_commit import (
    FrameActionAllocationTreatmentExecution,
    frame_source_call_and_request_identity,
)
from agent_evolve.ports.action_allocation_frame_commit_v3 import (
    OperationalFrameActionAllocationTreatmentExecution,
)
from agent_evolve.ports.action_allocation_frame_v3 import (
    AllocationScoreResolutionBinding,
    AllocationV3SeedSamplingLaw,
    AllocationV3SelectionBinding,
    AllocationV3TieMode,
    OperationalFrameActionAllocationRequest,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionGenerationResult,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioCardViewTransform,
    PortfolioExperimentalArm,
    portfolio_card_action_evidence_sha256,
)
from agent_evolve.ports.treatment_assignment import (
    OpaqueProviderSlotId,
    TreatmentAssignmentInput,
    TreatmentId,
    TreatmentOccurrence,
    TreatmentOccurrenceId,
)
from examples.development.airfoil_v7_two_stage_agent_evolution import (
    G1_SAMPLE_SIZE,
    G2_PORTFOLIO_SIZE,
    MAX_OUTPUT_TOKENS,
    OBJECTIVE_METRIC_ID,
    REQUIRED_METRIC_IDS,
    VIOLATION_METRIC_ID,
    AirfoilV7ForecastPortfolioUtility,
    bind_airfoil_mpn_allocation_commitment,
    bind_airfoil_mpn_frame_allocation_commitment,
    bind_airfoil_mpn_paired_allocation_commitment,
    build_airfoil_v7_forecast_arms,
    live_wiring_record,
    prepare_airfoil_v7_two_stage_generation,
)
import examples.development.airfoil_v7_two_stage_agent_evolution as target


EXPECTED_G1 = (
    "shape.camber_aft.p0015",
    "trim.p025.n025.p050",
    "shape.camber_aft.p0030",
    "trim.p050.n050.n050",
    "shape.camber_front.n0015",
    "trim.n050.p025.n050",
    "shape.camber_aft.n0030",
    "trim.n050.n025.p050",
)
EXPECTED_SAMPLE_RECEIPT = (
    "3a61d1436121eaae486b8dc195f501f10b8cad8c94d0f0f00cfe0a21d55d0b26"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


@pytest.fixture(scope="module")
def preparation():
    return prepare_airfoil_v7_two_stage_generation()


@pytest.fixture(scope="module")
def arms(preparation):
    return build_airfoil_v7_forecast_arms(preparation, _reflection(preparation))


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fake",
        provider_response_id="offline-reflection",
        finish_reason="tool_call",
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
    )


def _reflection(preparation) -> ReflectionWorkflowResult:
    call_id = LLMCallId("call_airfoil_twostage_reflection_fixture")
    drafts = tuple(
        InsightDraft(
            # Deliberately include the exact ID in prose: the benchmark card
            # projector must remove it while structured attribution retains it.
            claim=(
                f"Observed {observation.option_id} changed both measured metrics "
                f"in diagnostic case {observation.diagnostic_rank}."
            ),
            trigger=(
                f"Use {observation.option_id} only when its bounded intervention "
                "mechanism applies."
            ),
            mechanism=(
                "The coefficient vector acts like spanwise stations whose "
                "section incidence changes the flow response."
                if observation.diagnostic_rank == 1
                else "A bounded geometry or trim intervention changes the "
                "coupled flow response."
            ),
            affected_paths=(
                "$.alpha_deg"
                if observation.family == "trim_only"
                else "$.upper_coefficients"
            ,),
            evidence_summary=(
                "One exact requested child-minus-parent evaluation supports this "
                "conditional direction."
            ),
            confidence=0.5,
            evidence_contrast_ids=(observation.contrast_id,),
            effect_predictions=tuple(
                MetricEffectPrediction(
                    metric_id,
                    MetricEffectDirection.DECREASE,
                )
                for metric_id in REQUIRED_METRIC_IDS
            ),
            recommended_option_families=(observation.family,),
            recommended_option_ids=(observation.option_id,),
            action_template=f"Execute {observation.option_id} as a bounded trial.",
            falsification_condition=(
                f"Reject the {observation.option_id} mechanism if either metric "
                "direction reverses on another parent."
            ),
        )
        for observation in preparation.observations
    )
    generation = ReflectionGenerationResult(
        insights=drafts,
        telemetry=_telemetry(),
    )
    by_contrast = {
        observation.contrast_id: draft
        for observation, draft in zip(
            preparation.observations,
            drafts,
            strict=True,
        )
    }
    return ReflectionWorkflowResult(
        tuple(
            ReflectionShardResult(
                contrast_id=contrast_id,
                call_id=call_id,
                draft=by_contrast[contrast_id],
                generation_result=generation,
            )
            for contrast_id in sorted(by_contrast)
        )
    )


def test_preparation_replays_frozen_sample_and_excludes_g1_from_g2(
    preparation,
) -> None:
    replay = prepare_airfoil_v7_two_stage_generation()

    assert preparation.sample.receipt_sha256 == EXPECTED_SAMPLE_RECEIPT
    assert tuple(member.option_id for member in preparation.sample.members) == EXPECTED_G1
    assert replay.to_record() == preparation.to_record()
    assert len(preparation.contract.options) == 80
    assert len(preparation.observations) == G1_SAMPLE_SIZE
    assert len(preparation.eligible_g2_option_ids) == 72
    assert not set(EXPECTED_G1).intersection(preparation.eligible_g2_option_ids)
    assert set(EXPECTED_G1).union(preparation.eligible_g2_option_ids) == {
        option.option_id for option in preparation.contract.options
    }
    assert preparation.to_record()["provider_calls"] == 0
    assert preparation.evaluator.binding_record()["rank_exposed"] is False
    assert preparation.evaluator.binding_record()[
        "unselected_outcomes_exposed"
    ] is False


def test_predecision_decodes_only_manifest_finalization_and_exact_g1_terminals(
    monkeypatch,
) -> None:
    decoded: list[str] = []
    original = target._load_object

    def tracking(path, *, label):
        decoded.append(path.as_posix())
        return original(path, label=label)

    monkeypatch.setattr(target, "_load_object", tracking)
    prepared = prepare_airfoil_v7_two_stage_generation()

    assert not any(path.endswith("oracle_result.json") for path in decoded)
    assert not any(path.endswith("option_results.jsonl") for path in decoded)
    terminals = [path for path in decoded if path.endswith("/terminal.json")]
    assert len(terminals) == G1_SAMPLE_SIZE
    assert all(any(option_id in path for option_id in EXPECTED_G1) for path in terminals)
    assert thaw_json(prepared.oracle_seal)[
        "structural_seal_verification_decoded_outcome_file_count"
    ] == 0


def test_g1_observations_and_reflection_prompt_are_rank_free_and_subset_only(
    preparation,
) -> None:
    prompt = preparation.reflection_request.batch_prompt
    assert prompt is not None
    parsed = json.loads(prompt)

    assert len(parsed["observed_contrasts"]) == G1_SAMPLE_SIZE
    assert parsed["action_semantics"] == (
        target.AIRFOIL_V7_ACTION_SEMANTICS.to_record()
    )
    assert "unverified hypothesis" in parsed["instruction"]
    assert "observed metrics remain authoritative facts" in parsed["instruction"]
    assert preparation.reflection_request.max_output_tokens == MAX_OUTPUT_TOKENS
    assert preparation.reflection_request.temperature == 0.0
    assert preparation.reflection_request.insight_contract.required_metric_ids == (
        REQUIRED_METRIC_IDS
    )
    assert set(preparation.reflection_request.insight_contract.allowed_option_ids) == set(
        EXPECTED_G1
    )
    assert "rank" not in prompt.casefold()
    assert "rank_percentile" not in prompt.casefold()
    for option_id in EXPECTED_G1:
        assert option_id in prompt
    unselected = next(
        option.option_id
        for option in preparation.contract.options
        if option.option_id not in EXPECTED_G1
    )
    assert unselected not in prompt
    for observation in preparation.observations:
        record = observation.to_record()
        assert "rank" not in record["evaluation"]
        assert record["finite_action_evidence"]["option_id"] == observation.option_id
        assert len(record["evaluation"]["metrics"]) == 2


def test_post_reflection_builds_coherent_m_deranged_p_and_catalog_only_n(
    preparation,
    arms,
) -> None:
    assert len(arms.entries) == len(arms.source_cards) == len(arms.placebo_cards) == 8
    assert arms.memory_receipt.arm is PortfolioExperimentalArm.MEMORY
    assert arms.placebo_receipt.arm is PortfolioExperimentalArm.PERMUTED_PLACEBO
    assert all(card.source_binding is not None for card in arms.source_cards)
    assert all(card.derived_view_receipt is None for card in arms.source_cards)
    assert all(card.derived_view_receipt is not None for card in arms.placebo_cards)
    assert all(
        source != donor
        for source, donor in arms.placebo_receipt.source_donor_binding_pairs
    )
    assert {source for source, _ in arms.placebo_receipt.source_donor_binding_pairs} == {
        donor for _, donor in arms.placebo_receipt.source_donor_binding_pairs
    }

    observations_by_contrast = {
        observation.contrast_id: observation
        for observation in preparation.observations
    }
    observations_by_snapshot = {
        observation.empirical_evidence_snapshot().snapshot_sha256: observation
        for observation in preparation.observations
    }
    source_cards_by_binding = {
        card.source_binding.binding_sha256: card
        for card in arms.source_cards
        if card.source_binding is not None
    }
    donor_by_source = dict(arms.placebo_receipt.source_donor_binding_pairs)
    expected_placebo_transforms = tuple(
        sorted(
            (
                PortfolioCardViewTransform.EVIDENCE_PERMUTATION,
                PortfolioCardViewTransform.PROMPT_PERMUTATION,
                PortfolioCardViewTransform.SCORE_PERMUTATION,
            ),
            key=lambda value: value.value,
        )
    )
    entry_snapshot_sha256s = {
        entry.evidence_lineage.empirical_evidence[0].snapshot_sha256
        for entry in arms.entries
        if entry.evidence_lineage is not None
    }
    assert len(entry_snapshot_sha256s) == G1_SAMPLE_SIZE
    for entry in arms.entries:
        lineage = entry.evidence_lineage
        assert lineage is not None
        assert len(lineage.empirical_evidence) == 1
        assert len(lineage.finite_action_bindings) == 1
        snapshot = lineage.empirical_evidence[0]
        binding = lineage.finite_action_bindings[0]
        assert snapshot.contrast_id == binding.contrast_id
        assert lineage.cited_contrast_ids == (snapshot.contrast_id,)
        assert snapshot.fact_schema_id == target._EMPIRICAL_FACT_SCHEMA_ID
        assert snapshot.fact_schema_version == (
            target._EMPIRICAL_FACT_SCHEMA_VERSION
        )
        assert snapshot.fact_schema_definition_sha256 == (
            target._EMPIRICAL_FACT_SCHEMA_DEFINITION_SHA256
        )
    source_snapshot_sha256s = set()
    hallucinated_hypotheses = []
    for card in (*arms.source_cards, *arms.placebo_cards):
        payload = thaw_json(card.prompt_payload)
        neutral = json.dumps(payload, sort_keys=True).casefold()
        assert preparation.contract.identity_sha256 not in neutral
        for family in {option.family for option in preparation.contract.options}:
            assert family.casefold() not in neutral
        for option in preparation.contract.options:
            assert option.option_id.casefold() not in neutral
            assert option.identity_sha256 not in neutral
            assert option.child_configuration_sha256 not in neutral
        for binding in card.finite_action_evidence:
            assert binding.contrast_id not in neutral
            assert binding.identity_sha256 not in neutral
        assert set(payload) == {
            "schema_version",
            "empirical_facts",
            "hypothesis",
            "interpretation_policy",
        }
        assert payload["schema_version"] == 1
        assert payload["interpretation_policy"] == {
            "empirical_facts_are_observations": True,
            "hypothesis_is_observation": False,
            "mechanism_requires_independent_validation": True,
        }
        assert len(payload["empirical_facts"]) == 1
        empirical = payload["empirical_facts"][0]
        hypothesis = payload["hypothesis"]
        assert hypothesis["epistemic_status"] == "unverified_hypothesis"
        assert "recommended_option_ids" not in hypothesis
        assert "recommended_option_families" not in hypothesis
        assert "evidence_contrast_ids" not in hypothesis
        assert len(card.finite_action_evidence) == 1
        assert "contrast_id" not in empirical
        assert empirical["contrast_binding"] == (
            "structured_finite_action_evidence"
        )
        assert hypothesis["empirical_snapshot_sha256s"] == [
            empirical["snapshot_sha256"]
        ]
        action_contrast_id = card.finite_action_evidence[0].contrast_id
        action_observation = observations_by_contrast[action_contrast_id]
        empirical_observation = observations_by_snapshot[
            empirical["snapshot_sha256"]
        ]
        if card.derived_view_receipt is None:
            assert action_observation is empirical_observation
            source_snapshot_sha256s.add(empirical["snapshot_sha256"])
        else:
            view = card.derived_view_receipt
            source_binding = card.source_binding
            assert source_binding is not None
            source_sha256 = source_binding.binding_sha256
            donor_sha256 = donor_by_source[source_sha256]
            source_card = source_cards_by_binding[source_sha256]
            donor_card = source_cards_by_binding[donor_sha256]
            donor_payload = thaw_json(donor_card.prompt_payload)
            assert view.transforms == expected_placebo_transforms
            assert view.prompt_source_binding_sha256 == donor_sha256
            assert view.evidence_source_binding_sha256 == donor_sha256
            assert view.score_source_binding_sha256 == donor_sha256
            assert view.action_evidence_source_binding_sha256 is None
            assert card.reference == source_card.reference
            assert card.content_sha256 == source_card.content_sha256
            assert card.source_binding == source_card.source_binding
            assert card.evidence_sha256 == donor_card.evidence_sha256
            assert card.score_components == donor_card.score_components
            assert card.assigned_score == donor_card.assigned_score
            assert view.derived_evidence_sha256 == (
                donor_card.source_binding.source_evidence_sha256
            )
            assert view.derived_score_state_sha256 == (
                donor_card.source_binding.source_score_state_sha256
            )
            assert card.finite_action_evidence == (
                source_binding.finite_action_evidence
            )
            assert card.finite_action_evidence != (
                donor_card.finite_action_evidence
            )
            assert view.derived_action_evidence_sha256 == (
                portfolio_card_action_evidence_sha256(
                    source_binding.finite_action_evidence
                )
            )
            assert payload == donor_payload
            assert empirical["snapshot_sha256"] == donor_payload[
                "empirical_facts"
            ][0]["snapshot_sha256"]
            assert action_observation is not empirical_observation
            assert action_contrast_id != empirical_observation.contrast_id
        assert empirical["optimization_semantics_definition_sha256"] == (
            target.AIRFOIL_V7_OPTIMIZATION_SEMANTICS.definition_sha256
        )
        assert empirical["action_semantics_definition_sha256"] == (
            target.AIRFOIL_V7_ACTION_SEMANTICS.definition_sha256
        )
        assert empirical["facts"]["valid"] is True
        assert empirical["facts"]["evaluation_receipt_sha256"] == (
            empirical_observation.evaluation.terminal_record_sha256
        )
        metric_deltas = empirical["facts"]["observed_metric_deltas"]
        assert len(metric_deltas) == 2
        assert {
            value["metric_id"] for value in metric_deltas
        } == set(REQUIRED_METRIC_IDS)
        assert all(
            set(value)
            == {
                "metric_id",
                "parent_value",
                "child_value",
                "child_minus_parent_delta",
            }
            for value in metric_deltas
        )
        expected_deltas = {
            metric.metric_id: metric.delta
            for metric in empirical_observation.evaluation.metrics
        }
        assert {
            value["metric_id"]: value["child_minus_parent_delta"]
            for value in metric_deltas
        } == expected_deltas
        if "spanwise stations" in hypothesis["mechanism_hypothesis"]:
            hallucinated_hypotheses.append((empirical, hypothesis))

    assert source_snapshot_sha256s == entry_snapshot_sha256s
    assert len(hallucinated_hypotheses) == 2
    for empirical, hypothesis in hallucinated_hypotheses:
        assert "spanwise" not in json.dumps(empirical).casefold()
        assert hypothesis["epistemic_status"] == "unverified_hypothesis"
    trim_exclusions = next(
        axis
        for axis in target.AIRFOIL_V7_ACTION_SEMANTICS.to_record()["axes"]
        if axis["axis_id"] == "three_point_trim"
    )["excluded_interpretations"]
    exclusions = " ".join(trim_exclusions).casefold()
    assert "spanwise or chordwise stations" in exclusions
    assert "section-incidence controls" in exclusions

    assert arms.memory_request.evidence_mode is ActionForecastEvidenceMode.GROUNDED
    assert arms.memory_request.action_semantics is target.AIRFOIL_V7_ACTION_SEMANTICS
    assert arms.placebo_request.action_semantics is target.AIRFOIL_V7_ACTION_SEMANTICS
    assert (
        arms.catalog_only_request.action_semantics
        is target.AIRFOIL_V7_ACTION_SEMANTICS
    )
    assert arms.memory_request.experimental_view_receipt is arms.memory_receipt
    assert arms.placebo_request.evidence_mode is ActionForecastEvidenceMode.GROUNDED
    assert arms.placebo_request.experimental_view_receipt is arms.placebo_receipt
    assert arms.catalog_only_request.evidence_mode is (
        ActionForecastEvidenceMode.CATALOG_ONLY
    )
    assert arms.catalog_only_request.cards == ()
    assert arms.catalog_only_request.source_registry is None
    assert arms.catalog_only_request.experimental_view_receipt is None
    assert {
        arms.memory_request.context_sha256,
        arms.placebo_request.context_sha256,
        arms.catalog_only_request.context_sha256,
    } == {arms.memory_request.context_sha256}
    assert all(
        request.max_output_tokens == MAX_OUTPUT_TOKENS
        for request in (
            arms.memory_request,
            arms.placebo_request,
            arms.catalog_only_request,
        )
    )
    for request in (
        arms.memory_request,
        arms.placebo_request,
        arms.catalog_only_request,
    ):
        assert "exactly one primary prompt-visible evidence slot" in (
            request.instruction
        )
        assert "probability_valid_codes[i]" in request.instruction
        assert "every code matrix cell [i][j]" in request.instruction
        assert "closed ordinal" in request.instruction
        assert "action_rows[i]" not in request.instruction
        assert "one or more" not in request.instruction


def test_reflection_action_origin_mismatch_fails_before_forecast(preparation) -> None:
    result = _reflection(preparation)
    first = result.shards[0]
    wrong = next(
        observation
        for observation in preparation.observations
        if observation.option_id != first.draft.recommended_option_ids[0]
    )
    bad_draft = InsightDraft(
        claim=first.draft.claim,
        trigger=first.draft.trigger,
        mechanism=first.draft.mechanism,
        affected_paths=first.draft.affected_paths,
        evidence_summary=first.draft.evidence_summary,
        confidence=first.draft.confidence,
        evidence_contrast_ids=first.draft.evidence_contrast_ids,
        effect_predictions=first.draft.effect_predictions,
        recommended_option_families=(wrong.family,),
        recommended_option_ids=(wrong.option_id,),
        action_template=f"Execute {wrong.option_id}.",
        falsification_condition="Reject if the observed direction reverses.",
    )
    bad_generation = ReflectionGenerationResult(
        insights=(bad_draft, *result.shards[0].generation_result.insights[1:]),
        telemetry=_telemetry(),
    )
    bad_result = ReflectionWorkflowResult(
        (
            ReflectionShardResult(
                contrast_id=first.contrast_id,
                call_id=first.call_id,
                draft=bad_draft,
                generation_result=bad_generation,
            ),
            *result.shards[1:],
        )
    )

    with pytest.raises(ValueError, match="another exact action|another action family"):
        build_airfoil_v7_forecast_arms(preparation, bad_result)


def test_utility_and_evaluator_are_injected_benchmark_adapters(preparation) -> None:
    assert type(preparation.utility.utility) is AirfoilV7ForecastPortfolioUtility
    assert preparation.utility.policy_id == (
        "airfoil_v7_forecast_usefulness_probability"
    )
    evaluated = preparation.evaluator.evaluate_g1(EXPECTED_G1[:2])
    assert tuple(value.option_id for value in evaluated) == EXPECTED_G1[:2]
    assert all(value.active_wall_seconds > 0.0 for value in evaluated)
    assert all(value.outer_wall_seconds > 0.0 for value in evaluated)
    assert all("rank" not in value.to_record() for value in evaluated)
    with pytest.raises(ValueError, match="cannot repeat"):
        preparation.evaluator.evaluate_g1((EXPECTED_G1[0], EXPECTED_G1[0]))
    with pytest.raises(PermissionError, match="unavailable before"):
        preparation.evaluator.evaluate_g1((preparation.eligible_g2_option_ids[0],))

    wiring = live_wiring_record()
    assert wiring["provider_calls"]["count"] == 4
    assert wiring["provider_free_preparation_calls"] == 0
    assert wiring["provider_calls"]["stages"][1]["run_concurrently"] is True
    assert set(REQUIRED_METRIC_IDS) == {
        OBJECTIVE_METRIC_ID,
        VIOLATION_METRIC_ID,
    }


def _utility_member(
    option_id: str,
    *,
    delta_v: float,
    delta_f: float,
) -> ResolvedActionForecast:
    def metric(metric_id: str, delta: float) -> ResolvedActionMetricForecast:
        return ResolvedActionMetricForecast(
            metric_id=metric_id,
            p10_delta=delta,
            p50_delta=delta,
            p90_delta=delta,
            confidence=1.0,
            citations=(),
        )

    return ResolvedActionForecast(
        option_id=option_id,
        option_identity_sha256=("a" if option_id.endswith("a") else "b") * 64,
        child_configuration_sha256=("c" if option_id.endswith("a") else "d") * 64,
        family="fixture",
        probability_valid=1.0,
        metric_forecasts=tuple(
            sorted(
                (
                    metric(OBJECTIVE_METRIC_ID, delta_f),
                    metric(VIOLATION_METRIC_ID, delta_v),
                ),
                key=lambda value: value.metric_id,
            )
        ),
    )


def test_normalized_utility_bounds_drag_and_cannot_overturn_violation(
    preparation,
) -> None:
    utility = preparation.utility.utility
    good_violation_awful_drag = _utility_member(
        "fixture.a",
        delta_v=-0.005,
        delta_f=1.0e300,
    )
    bad_violation_great_drag = _utility_member(
        "fixture.b",
        delta_v=0.005,
        delta_f=-1.0e300,
    )

    def score(members) -> float:
        return utility(
            ForecastPortfolioUtilityInput(
                optimization_semantics=target.AIRFOIL_V7_OPTIMIZATION_SEMANTICS,
                parent_metric_values=preparation.parent_metric_values,
                metric_scales=preparation.metric_scales,
                members=tuple(members),
                quantile=ForecastQuantile.P50,
            )
        )

    good = score((good_violation_awful_drag,))
    bad = score((bad_violation_great_drag,))
    combined = score((good_violation_awful_drag, bad_violation_great_drag))
    assert 0.0 <= bad < good <= 1.0
    assert good <= combined <= 1.0


def _allocation_phase_commit(arms) -> TwoStageActionPhaseCommit:
    executions = []
    selected_options = tuple(
        arms.preparation.contract.resolve(option_id)
        for option_id in arms.preparation.eligible_g2_option_ids[:3]
    )
    for arm in ("m", "p", "n"):
        eligible_sha256 = target._hash(
            target._ELIGIBLE_ACTION_SET_DOMAIN,
            {
                "eligible_option_ids": list(
                    arms.preparation.eligible_g2_option_ids
                )
            },
        )
        request = {
            "schema_version": 1,
            "forecast_request_sha256": arms.request(arm).request_sha256,
            "forecast_receipt_sha256": "1" * 64,
            "eligible_option_ids": list(arms.preparation.eligible_g2_option_ids),
            "eligible_options_sha256": eligible_sha256,
            "portfolio_size": 3,
            "utility": arms.preparation.utility.to_record(),
        }
        request_sha256 = target._hash(
            target._ACTION_ALLOCATION_REQUEST_DOMAIN,
            request,
        )
        score = {
            name: 0.0.hex()
            for name in (
                "p10_utility",
                "p50_utility",
                "p90_utility",
                "downside_utility",
                "risk_penalty",
                "diversity_reward",
                "total_utility",
            )
        }
        unsigned_decision = {
            "schema_version": 1,
            "allocation_request_sha256": request_sha256,
            "forecast_receipt_sha256": "1" * 64,
            "finite_contract_identity_sha256": (
                arms.preparation.contract.identity_sha256
            ),
            "eligible_options_sha256": eligible_sha256,
            "members": [
                {
                    "rank": rank,
                    "option_id": option.option_id,
                    "option_identity_sha256": option.identity_sha256,
                    "child_configuration_sha256": (
                        option.child_configuration_sha256
                    ),
                    "family": option.family,
                    "greedy_step_score": score,
                    "marginal_total_utility_hex": 0.0.hex(),
                }
                for rank, option in enumerate(selected_options, start=1)
            ],
            "final_score": score,
            "candidate_evaluations": 1,
            "utility_policy": arms.preparation.utility.to_record(),
            "allocator_policy": {
                "policy_id": "fixture",
                "policy_version": 1,
                "definition_sha256": "3" * 64,
                "configuration_sha256": "4" * 64,
            },
        }
        decision = {
            **unsigned_decision,
            "receipt_sha256": target._hash(
                target._ACTION_PORTFOLIO_DECISION_DOMAIN,
                unsigned_decision,
            ),
        }
        executions.append(
            {
                "arm": arm,
                "allocation_request": request,
                "decision": decision,
            }
        )
    payload = freeze_json(
        {
            "schema_version": 1,
            "phase": "allocate",
            "run_request_sha256": "5" * 64,
            "arm_executions": executions,
        }
    )
    receipt = TwoStageActionPhaseReceipt(
        phase=TwoStageActionPhase.ALLOCATE,
        input_sha256="6" * 64,
        output_sha256=typed_json_sha256(payload),
    )
    return TwoStageActionPhaseCommit(receipt=receipt, payload=payload)


class _NoBoundaryOrExtreme:
    def __call__(
        self,
        request: AllocationCandidateScoreDiagnosticInput,
    ) -> AllocationCandidateScoreDiagnostic:
        request.__post_init__()
        return AllocationCandidateScoreDiagnostic(boundary_or_extreme=False)


def _airfoil_frame_allocation_bundle(arms):
    partition_policy = ActionForecastPartitionPolicyBinding(
        policy_id="airfoil_test_twenty_row_blocks",
        policy_version=1,
        policy_definition_sha256=_sha("airfoil-test-twenty-row-blocks-v1"),
        max_rows_per_block=20,
        max_metric_cells_per_block=40,
    )
    subset_policy = ActionAllocationFrameSubsetPolicyBinding(
        policy_id="airfoil_test_block_g2_intersection",
        policy_version=1,
        policy_definition_sha256=_sha(
            "airfoil-test-block-g2-intersection-v1"
        ),
    )
    global_g2 = set(arms.preparation.eligible_g2_option_ids)
    frame_requests: list[FrameActionAllocationRequest] = []
    for arm in ("m", "p", "n"):
        forecast_request = arms.request(arm)
        layout = build_action_forecast_partition_layout(
            forecast_request,
            partition_policy,
        )
        block_request = build_action_forecast_block_requests(
            forecast_request,
            layout,
        )[0]
        metric_ids = forecast_request.required_metric_ids
        citations = ()
        if forecast_request.evidence_mode is ActionForecastEvidenceMode.GROUNDED:
            card = forecast_request.cards[0]
            binding = card.finite_action_evidence[0]
            citations = (
                ActionEvidenceCitation(card.card_key, binding.identity_sha256),
            )
        drafts = tuple(
            ActionForecastDraft(
                option_id=(
                    forecast_request.finite_variation_contract.options[
                        global_index
                    ].option_id
                ),
                probability_valid=0.95,
                metric_forecasts=tuple(
                    ActionMetricForecast(
                        metric_id=metric_id,
                        p10_delta=(global_index + metric_index) / 100_000.0,
                        p50_delta=(global_index + metric_index + 1) / 100_000.0,
                        p90_delta=(global_index + metric_index + 2) / 100_000.0,
                        confidence=0.75,
                        citations=citations,
                    )
                    for metric_index, metric_id in enumerate(metric_ids)
                ),
            )
            for global_index in range(
                block_request.block.global_row_start,
                block_request.block.global_row_stop,
            )
        )
        resolved_block = resolve_action_forecast_block(
            block_request,
            drafts,
            policy_id="airfoil_test_resolved_block",
            policy_version=1,
            policy_definition_sha256=_sha("airfoil-test-resolved-block-v1"),
        )
        included_rows = tuple(
            index
            for index in range(
                block_request.block.global_row_start,
                block_request.block.global_row_stop,
            )
            if forecast_request.finite_variation_contract.options[index].option_id
            in global_g2
        )
        frame = bind_action_forecast_block_subset_allocation_frame(
            block_request,
            resolved_block,
            included_global_row_indices=included_rows,
            subset_policy=subset_policy,
            parent_receipt_sha256s=tuple(
                sorted(
                    (
                        _sha(f"airfoil-{arm}-block-health"),
                        _sha(f"airfoil-{arm}-subset-health"),
                    )
                )
            ),
        )
        frame_requests.append(
            FrameActionAllocationRequest(
                frame=frame,
                eligible_option_ids=tuple(
                    sorted(value.option_id for value in frame.forecasts)
                ),
                portfolio_size=3,
                utility=arms.preparation.utility,
            )
        )

    occurrences = tuple(
        TreatmentOccurrence(
            occurrence_id=TreatmentOccurrenceId(f"airfoil.trial.{arm}"),
            treatment_id=TreatmentId(arm),
            call_identity=frame_source_call_and_request_identity(request)[0],
            request_identity_sha256=frame_source_call_and_request_identity(request)[1],
        )
        for arm, request in zip(("m", "p", "n"), frame_requests, strict=True)
    )
    assignment = assign_treatment_occurrences(
        TreatmentAssignmentInput(
            experiment_commitment_sha256=_sha(
                "airfoil-frame-allocation-experiment"
            ),
            public_seed_material="public.seed.airfoil.frame.allocation",
            occurrences=occurrences,
            provider_slot_ids=tuple(
                OpaqueProviderSlotId(f"opaque.airfoil.{index:02d}")
                for index in range(3)
            ),
        )
    )
    allocator = AuditedGreedyForecastFrameAllocator(
        risk_aversion=0.5,
        diversity_weight=0.25,
        score_diagnostic=AllocationScoreDiagnosticBinding(
            diagnostic=_NoBoundaryOrExtreme(),
            policy_id="airfoil_test_no_extreme",
            policy_version=1,
            policy_definition_sha256=_sha("airfoil-test-no-extreme-v1"),
        ),
        gate_policy=AllocationSurfaceGatePolicyBinding(
            policy_id="airfoil_test_permissive_surface",
            policy_version=1,
            policy_definition_sha256=_sha(
                "airfoil-test-permissive-surface-v1"
            ),
            minimum_distinct_finite_scores=1,
            maximum_top_tie_share=1.0,
            maximum_boundary_or_extreme_share=1.0,
            minimum_winner_runner_gap=0.0,
        ),
    )
    executions = tuple(
        FrameActionAllocationTreatmentExecution(
            treatment_assignment=assignment,
            treatment_occurrence=occurrence,
            request=request,
            result=allocator.allocate(request),
        )
        for occurrence, request in zip(
            occurrences,
            frame_requests,
            strict=True,
        )
    )
    commit = build_frame_action_allocation_phase_commit(
        upstream_input_sha256=_sha("airfoil-frame-upstream-wave"),
        terminal_provider_ledger_commitment_sha256=_sha(
            "airfoil-fsynced-terminal-provider-ledger"
        ),
        executions=executions,
    )
    return executions, commit, allocator


@pytest.fixture(scope="module")
def frame_allocation_bundle(arms):
    return _airfoil_frame_allocation_bundle(arms)


def test_only_real_canonical_allocate_commit_opens_one_shot_selected_g2(
    preparation,
    arms,
) -> None:
    commit = _allocation_phase_commit(arms)
    commitment = bind_airfoil_mpn_allocation_commitment(arms, commit)

    assert tuple(value[0] for value in commitment.arm_allocation_pairs) == (
        "m",
        "p",
        "n",
    )
    assert commitment.phase_commit_receipt_sha256 == commit.receipt.receipt_sha256
    assert set(commitment.selected_option_ids).issubset(
        preparation.eligible_g2_option_ids
    )
    capability = preparation.evaluator.open_postdecision_evaluation(commitment)
    results = capability.evaluate_selected()
    assert tuple(value.option_id for value in results) == commitment.selected_option_ids
    with pytest.raises(RuntimeError, match="one-shot"):
        capability.evaluate_selected()


def test_non_allocate_or_noncanonical_phase_commit_fails_closed(arms) -> None:
    commit = _allocation_phase_commit(arms)
    wrong_phase = TwoStageActionPhaseCommit(
        receipt=TwoStageActionPhaseReceipt(
            phase=TwoStageActionPhase.FORECAST,
            input_sha256=commit.receipt.input_sha256,
            output_sha256=commit.receipt.output_sha256,
        ),
        payload=commit.payload,
    )
    with pytest.raises(ValueError, match="ALLOCATE"):
        bind_airfoil_mpn_allocation_commitment(arms, wrong_phase)

    payload = thaw_json(commit.payload)
    payload["arm_executions"][1], payload["arm_executions"][2] = (
        payload["arm_executions"][2],
        payload["arm_executions"][1],
    )
    frozen = freeze_json(payload)
    reordered = TwoStageActionPhaseCommit(
        receipt=TwoStageActionPhaseReceipt(
            phase=TwoStageActionPhase.ALLOCATE,
            input_sha256=commit.receipt.input_sha256,
            output_sha256=typed_json_sha256(frozen),
        ),
        payload=frozen,
    )
    with pytest.raises(ValueError, match="canonical M/P/N"):
        bind_airfoil_mpn_allocation_commitment(arms, reordered)


def test_passing_common_block_subset_commit_opens_airfoil_frame_firewall(
    arms,
    frame_allocation_bundle,
) -> None:
    executions, commit, _allocator = frame_allocation_bundle
    commitment = bind_airfoil_mpn_frame_allocation_commitment(
        arms,
        executions,
        commit,
    )

    assert tuple(value[0] for value in commitment.arm_allocation_pairs) == (
        "m",
        "p",
        "n",
    )
    eligible_count = len(executions[0].request.eligible_option_ids)
    expected_candidate_evaluations = sum(
        eligible_count - step for step in range(G2_PORTFOLIO_SIZE)
    )
    assert all(
        len(value.request.eligible_option_ids) == eligible_count
        for value in executions
    )
    assert all(
        value.result.decision.candidate_evaluations
        == expected_candidate_evaluations
        for value in executions
    )
    assert commitment.phase_commit_receipt_sha256 == (
        commit.receipt.receipt_sha256
    )
    assert set(commitment.selected_option_ids).issubset(
        arms.preparation.eligible_g2_option_ids
    )


def _airfoil_paired_comparison_bundle(arms, frame_allocation_bundle):
    v2_executions, v2_commit, _allocator = frame_allocation_bundle
    v3_requests = tuple(
        OperationalFrameActionAllocationRequest(
            allocation=value.request,
            risk_aversion=0.5,
            diversity_weight=0.25,
            score_resolution=AllocationScoreResolutionBinding(
                policy_id="airfoil_test_exact_resolution",
                policy_version=1,
                policy_definition_sha256=_sha(
                    "airfoil-test-exact-resolution-v1"
                ),
                maximum_indistinguishable_score_gap=0.0,
            ),
            tie_selection=AllocationV3SelectionBinding(
                policy_id="airfoil_test_public_rank",
                policy_version=1,
                policy_definition_sha256=_sha("airfoil-test-public-rank-v1"),
                mode=AllocationV3TieMode.PUBLIC_HASH_RANK,
                seed_sampling_law=AllocationV3SeedSamplingLaw.FIXED_PUBLIC,
                seed_provenance_sha256=_sha("airfoil-test-seed-release"),
                public_seed=20_260_715,
                allocation_unit_key="airfoil.v7.test.replicate.00.block.00",
            ),
        )
        for value in v2_executions
    )
    v3_allocator = OperationalGreedyForecastFrameAllocator()
    v3_executions = tuple(
        OperationalFrameActionAllocationTreatmentExecution(
            treatment_assignment=value.treatment_assignment,
            treatment_occurrence=value.treatment_occurrence,
            request=request,
            result=v3_allocator.allocate(request),
        )
        for value, request in zip(v2_executions, v3_requests, strict=True)
    )
    upstream = _sha("airfoil-frame-upstream-wave")
    ledger = _sha("airfoil-fsynced-terminal-provider-ledger")
    v3_commit = build_operational_frame_action_allocation_phase_commit(
        upstream_input_sha256=upstream,
        terminal_provider_ledger_commitment_sha256=ledger,
        executions=v3_executions,
    )
    schedule = _sha("airfoil-paired-schedule-release")
    methods = (
        AllocationComparisonMethodWave(
            comparison_method_id="allocator_v2",
            schedule_binding_sha256=schedule,
            executions=v2_executions,
            phase_commit=v2_commit,
        ),
        AllocationComparisonMethodWave(
            comparison_method_id="allocator_v3",
            schedule_binding_sha256=schedule,
            executions=v3_executions,
            phase_commit=v3_commit,
        ),
    )
    generic = build_paired_allocation_comparison_commitment(methods)
    airfoil = bind_airfoil_mpn_paired_allocation_commitment(
        arms,
        methods,
        generic,
        expected_schedule_binding_sha256=schedule,
    )
    return methods, generic, airfoil


def test_airfoil_paired_v2_v3_union_is_exact_one_shot_and_slot_bounded(
    arms,
    frame_allocation_bundle,
) -> None:
    methods, generic, commitment = _airfoil_paired_comparison_bundle(
        arms,
        frame_allocation_bundle,
    )

    assert generic.logical_slot_count == commitment.logical_slot_count == 18
    assert commitment.selected_option_ids == generic.selected_option_ids
    assert len(commitment.selected_option_ids) <= 18
    assert commitment.to_record()["raw_outcome_authority"] == (
        "selected_union_only"
    )
    with pytest.raises(ValueError, match="another Airfoil schedule"):
        bind_airfoil_mpn_paired_allocation_commitment(
            arms,
            methods,
            generic,
            expected_schedule_binding_sha256=_sha("foreign-airfoil-schedule"),
        )

    fresh = prepare_airfoil_v7_two_stage_generation()
    capability = fresh.evaluator.open_paired_postdecision_evaluation(commitment)
    results = capability.evaluate_selected_union()
    assert tuple(value.option_id for value in results) == (
        commitment.selected_option_ids
    )
    assert fresh.evaluator.binding_record()["provider_calls"] == 0
    with pytest.raises(RuntimeError, match="one-shot"):
        capability.evaluate_selected_union()


def test_airfoil_frame_firewall_rejects_incomplete_g2_subset_and_payload_tamper(
    arms,
    frame_allocation_bundle,
) -> None:
    executions, commit, allocator = frame_allocation_bundle
    reduced_request = replace(
        executions[1].request,
        eligible_option_ids=executions[1].request.eligible_option_ids[:-1],
    )
    reduced_execution = replace(
        executions[1],
        request=reduced_request,
        result=allocator.allocate(reduced_request),
    )
    reduced_executions = (
        executions[0],
        reduced_execution,
        executions[2],
    )
    reduced_commit = build_frame_action_allocation_phase_commit(
        upstream_input_sha256=_sha("airfoil-frame-upstream-wave"),
        terminal_provider_ledger_commitment_sha256=_sha(
            "airfoil-fsynced-terminal-provider-ledger"
        ),
        executions=reduced_executions,
    )
    with pytest.raises(ValueError, match="exact common G2 set"):
        bind_airfoil_mpn_frame_allocation_commitment(
            arms,
            reduced_executions,
            reduced_commit,
        )

    payload = thaw_json(commit.payload)
    payload["treatment_executions"][0]["audit"]["passes"] = False
    frozen = freeze_json(payload)
    tampered = TwoStageActionPhaseCommit(
        receipt=TwoStageActionPhaseReceipt(
            phase=TwoStageActionPhase.ALLOCATE,
            input_sha256=commit.receipt.input_sha256,
            output_sha256=typed_json_sha256(frozen),
        ),
        payload=frozen,
    )
    with pytest.raises(ValueError, match="exact execution payloads"):
        bind_airfoil_mpn_frame_allocation_commitment(
            arms,
            executions,
            tampered,
        )


def test_airfoil_frame_firewall_rejects_noncanonical_treatment_order(
    arms,
    frame_allocation_bundle,
) -> None:
    executions, _commit, _allocator = frame_allocation_bundle
    treatments = ("p", "m", "n")
    occurrences = tuple(
        replace(
            value.treatment_occurrence,
            treatment_id=TreatmentId(treatment),
        )
        for value, treatment in zip(executions, treatments, strict=True)
    )
    assignment = assign_treatment_occurrences(
        TreatmentAssignmentInput(
            experiment_commitment_sha256=_sha(
                "airfoil-frame-noncanonical-experiment"
            ),
            public_seed_material="public.seed.airfoil.noncanonical",
            occurrences=occurrences,
            provider_slot_ids=tuple(
                OpaqueProviderSlotId(f"opaque.noncanonical.{index:02d}")
                for index in range(3)
            ),
        )
    )
    noncanonical = tuple(
        FrameActionAllocationTreatmentExecution(
            treatment_assignment=assignment,
            treatment_occurrence=occurrence,
            request=value.request,
            result=value.result,
        )
        for occurrence, value in zip(occurrences, executions, strict=True)
    )
    commit = build_frame_action_allocation_phase_commit(
        upstream_input_sha256=_sha("airfoil-frame-noncanonical-input"),
        terminal_provider_ledger_commitment_sha256=_sha(
            "airfoil-frame-noncanonical-ledger"
        ),
        executions=noncanonical,
    )
    with pytest.raises(ValueError, match="canonical M/P/N"):
        bind_airfoil_mpn_frame_allocation_commitment(
            arms,
            noncanonical,
            commit,
        )
