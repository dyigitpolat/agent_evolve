"""Provider-free rehearsal of the exact Airfoil-wired G0→G3 live path."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal
from functools import cache
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from agent_evolve.application.g3_causal_validation import (
    validate_g3_causal_screen_result,
)
from agent_evolve.application.g3_postseal_curation import (
    G3PostsealCurationInterceptor,
)
from agent_evolve.domain.artifact import artifact_ref_for_bytes
from agent_evolve.domain.ids import ProviderAttemptId
from agent_evolve.domain.llm_task_queue import (
    AttemptRequestEvidence,
    AttemptRequestVariant,
    AttemptStatus,
    AttemptTelemetry,
    LLMTaskOutcome,
    TaskOutcomeStatus,
    TaskTelemetry,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    structured_generation_output_evidence_record,
    structured_generation_outcome_record,
    structured_generation_request_evidence_record,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    FiniteVariationSelectionDraft,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    VariationGenerationRequest,
    VariationGenerationResult,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)
from agent_evolve.application.detailed_evaluation import DetailedEvaluationPayload
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    EVALUATOR_IDENTITY,
    OBJECTIVE_NAME,
    VIOLATION_NAME,
    AirfoilV7Problem,
)
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    ADFLOW_EVALUATOR_ID,
    EVIDENCE_CONTRACT_ID,
    V2_EVALUATOR_ID,
)
from examples.benchmarks.engibench_airfoil import v7_g3_release as release
from examples.benchmarks.engibench_airfoil.problem_def import candidate_sha256
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    DEEPSEEK_G3_PROVIDER_PROFILE,
    GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE,
    AirfoilG3LaunchVerification,
    AirfoilG3LiveError,
    AirfoilG3ProviderProfile,
    OwnedAgenticGenerator,
    bind_provider_route,
    build_openrouter_config,
    build_telemetry_policy,
    compose_airfoil_g3_live,
)
from examples.benchmarks.engibench_airfoil.v7_g3_analysis import (
    analyze_airfoil_g3_live_result,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    compose_airfoil_g3_runtime_inputs,
)
from examples.development.durable_run_artifacts import (
    DurableJsonlJournal,
    read_jsonl,
)


class _NoRawCFD:
    def evaluate_raw(self, configuration):
        del configuration
        raise AssertionError("provider-free rehearsal must not invoke raw CFD")


@cache
def _prepared() -> release.AirfoilG3ReleasePreparation:
    return release.prepare_release()


class _FastDetailedEvaluator:
    evaluator_identity = EVALUATOR_IDENTITY

    def __init__(self, receipt_root: Path) -> None:
        self.calls: list[str] = []
        self.receipt_root = receipt_root
        self.receipt_root.mkdir(parents=True)
        self.receipt_paths: list[Path] = []

    def evaluate_evidence(self, configuration) -> DetailedEvaluationPayload:
        key = candidate_sha256(configuration)
        self.calls.append(key)
        ordinal = int(key[:12], 16)
        objective = 0.9 + (ordinal % 10_000) / 100_000.0
        violation = 0.2 + ((ordinal // 10_000) % 10_000) / 100_000.0
        record = {
            "schema_version": 2,
            "evaluator_id": V2_EVALUATOR_ID,
            "status": "evaluated",
            "candidate_sha256": key,
            "evaluator_calls": 3,
            "points": [
                {
                    "index": index,
                    "evaluator_evidence": {
                        "contract_id": EVIDENCE_CONTRACT_ID,
                        "evaluator_id": ADFLOW_EVALUATOR_ID,
                        "accepted": True,
                    },
                }
                for index in range(3)
            ],
        }
        content = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        path = self.receipt_root / f"{key}.json"
        path.write_bytes(content)
        self.receipt_paths.append(path)
        return DetailedEvaluationPayload(
            failure=None,
            objectives=((OBJECTIVE_NAME, objective),),
            violations=((VIOLATION_NAME, violation),),
            checks=(),
            receipt=artifact_ref_for_bytes(content, media_type="application/json"),
            evaluator=EVALUATOR_IDENTITY,
            active_wall_seconds=0.001,
            resource_queue_wall_seconds=None,
        )


def _telemetry(
    call_id: str,
    provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model=provider_profile.model_alias,
        resolved_model=provider_profile.canonical_model,
        resolved_provider=provider_profile.resolved_provider,
        provider_response_id=f"response-{call_id}",
        finish_reason="fixture",
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
        attempt_count=1,
    )


def _provider_success_outcome(
    request: StructuredGenerationRequest[Any],
    response: StructuredGenerationResponse[Any],
) -> LLMTaskOutcome[StructuredGenerationResponse[Any]]:
    prompt_sha256 = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
    attempt = AttemptTelemetry(
        attempt_number=1,
        status=AttemptStatus.SUCCEEDED,
        wait_time_ns=0,
        service_time_ns=1,
        will_retry=False,
        request_evidence=AttemptRequestEvidence(
            variant=AttemptRequestVariant.ORIGINAL,
            prompt_sha256=prompt_sha256,
            provider_attempt_id=ProviderAttemptId(
                "provider_attempt_" + hashlib.sha256(
                    request.call_id.value.encode("ascii")
                ).hexdigest()[:24]
            ),
        ),
    )
    return LLMTaskOutcome(
        status=TaskOutcomeStatus.SUCCEEDED,
        telemetry=TaskTelemetry(
            task_id=request.call_id.value,
            queue_time_ns=0,
            service_time_ns=1,
            total_time_ns=1,
            attempts=(attempt,),
        ),
        response=response,
    )


def _reflection_output_item(value: InsightDraft) -> dict[str, object]:
    return {
        "claim": value.claim,
        "trigger": value.trigger,
        "mechanism": value.mechanism,
        "affected_paths": list(value.affected_paths),
        "evidence_summary": value.evidence_summary,
        "confidence": value.confidence,
        "evidence_contrast_ids": list(value.evidence_contrast_ids),
        "effect_predictions": [
            {"metric_id": item.metric_id, "direction": item.direction.value}
            for item in value.effect_predictions
        ],
        "recommended_option_families": list(
            value.recommended_option_families
        ),
        "recommended_option_ids": list(value.recommended_option_ids),
        "action_template": value.action_template,
        "falsification_condition": value.falsification_condition,
    }


class _AirfoilGenerator:
    def __init__(
        self,
        options: dict[str, str],
        curation_mode: str,
        provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
    ) -> None:
        self.options = options
        self.curation_mode = curation_mode
        self.provider_profile = provider_profile
        self.proposal_requests: list[VariationGenerationRequest] = []
        self.reflection_requests: list[ReflectionGenerationRequest] = []
        self.proposal_results: dict[str, VariationGenerationResult] = {}
        self.reflection_results: dict[str, ReflectionGenerationResult] = {}

    async def propose(
        self,
        request: VariationGenerationRequest,
    ) -> VariationGenerationResult:
        self.proposal_requests.append(request)
        contract = request.finite_variation_contract
        assert contract is not None
        selected = next(
            insight_id
            for insight_id in self.options
            if f'"insight_id":"{insight_id}"' in request.prompt
        )
        option = contract.resolve(self.options[selected])
        result = VariationGenerationResult(
            draft=FiniteVariationSelectionDraft(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                contract_identity_sha256=contract.identity_sha256,
                design_rationale="Execute the exact sealed Airfoil treatment.",
                claimed_insight_ids=(selected,),
            ),
            telemetry=_telemetry(request.call_id.value, self.provider_profile),
        )
        self.proposal_results[request.call_id.value] = result
        return result

    async def reflect(
        self,
        request: ReflectionGenerationRequest,
    ) -> ReflectionGenerationResult:
        self.reflection_requests.append(request)
        if self.curation_mode == "transport_failure":
            raise RuntimeError("injected postseal provider failure")
        if self.curation_mode == "abstain":
            insights: tuple[InsightDraft, ...] = ()
        else:
            marker = "EVALUATED TRACE\n"
            evidence_text = request.prompt.split(marker, 1)[1]
            row, _ = json.JSONDecoder().raw_decode(evidence_text)
            contrast = row[0]["machine_derived_contrasts"][0]
            option_id = contrast["finite_variation_option"]["option_id"]
            insights = (
                InsightDraft(
                    claim="The direct adaptive trim association merits a held-out retest.",
                    trigger="A future parent admits the same three-point trim action.",
                    mechanism="The sealed outcome is associational and remains quarantined.",
                    affected_paths=("$.alpha_deg",),
                    evidence_summary="Only the exact direct adaptive G2 contrast is cited.",
                    confidence=0.1,
                    evidence_contrast_ids=(contrast["contrast_id"],),
                    effect_predictions=(
                        MetricEffectPrediction(
                            "objective:normalized_multipoint_drag",
                            MetricEffectDirection.DECREASE,
                        ),
                        MetricEffectPrediction(
                            "violation:normalized_lift_equality",
                            MetricEffectDirection.UNKNOWN,
                        ),
                    ),
                    recommended_option_families=("trim_only",),
                    recommended_option_ids=(option_id,),
                    action_template="Retest the same bounded three-coordinate trim.",
                    falsification_condition="The association fails on a new parent.",
                ),
            )
            if self.curation_mode == "contract_failure":
                invalid = insights[0]
                assert request.insight_contract is not None
                other_option_id = next(
                    value
                    for value in request.insight_contract.allowed_option_ids
                    if value != option_id
                )
                insights = (
                    InsightDraft(
                        claim=invalid.claim,
                        trigger=invalid.trigger,
                        mechanism=invalid.mechanism,
                        affected_paths=invalid.affected_paths,
                        evidence_summary=invalid.evidence_summary,
                        confidence=invalid.confidence,
                        evidence_contrast_ids=invalid.evidence_contrast_ids,
                        effect_predictions=invalid.effect_predictions,
                        recommended_option_families=("trim_only",),
                        recommended_option_ids=(other_option_id,),
                        action_template=invalid.action_template,
                        falsification_condition=invalid.falsification_condition,
                    ),
                )
        result = ReflectionGenerationResult(
            insights=insights,
            telemetry=_telemetry(request.call_id.value, self.provider_profile),
        )
        self.reflection_results[request.call_id.value] = result
        return result


class _Gate:
    def __init__(self, manifest_sha256: str, freeze_sha256: str) -> None:
        self.value = AirfoilG3LaunchVerification(manifest_sha256, freeze_sha256)
        self.calls = 0

    def verify(self) -> AirfoilG3LaunchVerification:
        self.calls += 1
        return self.value


@pytest.mark.parametrize(
    (
        "curation_mode",
        "expected_status",
        "expected_publications",
        "provider_profile",
    ),
    (
        ("revision", "sealed_complete", 1, DEEPSEEK_G3_PROVIDER_PROFILE),
        ("abstain", "sealed_complete", 0, DEEPSEEK_G3_PROVIDER_PROFILE),
        ("transport_failure", "incomplete", 0, DEEPSEEK_G3_PROVIDER_PROFILE),
        ("contract_failure", "incomplete", 0, DEEPSEEK_G3_PROVIDER_PROFILE),
        (
            "revision",
            "sealed_complete",
            1,
            GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE,
        ),
    ),
)
def test_airfoil_wired_live_composition_runs_exact_g0_g3_provider_free(
    tmp_path: Path,
    curation_mode: str,
    expected_status: str,
    expected_publications: int,
    provider_profile: AirfoilG3ProviderProfile,
) -> None:
    preparation = _prepared()
    permutation, _, _ = release.freeze_diagnostic_permutation(preparation)
    problem = AirfoilV7Problem(raw_problem=_NoRawCFD())
    evaluator = _FastDetailedEvaluator(tmp_path / "raw_receipts")
    problem.detailed_evaluator = evaluator
    trace_path = tmp_path / f"{curation_mode}.jsonl"
    trace = DurableJsonlJournal(trace_path)

    def trace_sink(source: str):
        return lambda row: trace.append(
            {"schema_version": 1, "source": source, **dict(row)}
        )

    inputs = compose_airfoil_g3_runtime_inputs(
        problem=problem,
        preparation=preparation,
        diagnostic_permutation=permutation,
        freeze_receipt_sha256="0" * 64,
        planner_trace_sink=trace_sink("planner"),
    )
    options = {
        entry.reference.insight_id.value: entry.draft.recommended_option_ids[0]
        for entry in (*inputs.active_entries, inputs.neutral_entry)
    }
    generator = _AirfoilGenerator(options, curation_mode, provider_profile)
    factory_calls = 0
    close_calls = 0

    def generator_factory(provider_profile, api_key, config, sinks):
        nonlocal factory_calls, close_calls
        assert provider_profile is generator.provider_profile
        del config
        sinks.__post_init__()
        assert api_key == "provider-free-key"
        factory_calls += 1

        def close() -> None:
            nonlocal close_calls
            close_calls += 1

        return OwnedAgenticGenerator(generator=generator, close=close)

    credential_calls = 0

    def credential_loader() -> str:
        nonlocal credential_calls
        # The optimizer has already evaluated both G0 seeds when it first asks
        # for a proposal; this is the chronology property under rehearsal.
        assert len(evaluator.calls) == 2
        credential_calls += 1
        return "provider-free-key"

    gate = _Gate("1" * 64, "0" * 64)
    live = compose_airfoil_g3_live(
        inputs,
        launch_gate=gate,
        expected_manifest_sha256="1" * 64,
        credential_loader=credential_loader,
        progress_sink=lambda value: None,
        outcome_sink=lambda value: None,
        request_evidence_sink=lambda value: None,
        output_evidence_sink=lambda value: None,
        provider_profile=provider_profile,
        generator_factory=generator_factory,
        engine_trace_sink=trace_sink("engine"),
        optimizer_trace_sink=trace_sink("optimizer"),
    )
    assert live.run_state == "not_started"
    result = asyncio.run(live.run())
    asyncio.run(live.aclose())
    trace.close()

    assert live.run_state == "completed"
    assert factory_calls == credential_calls == close_calls == 1
    assert gate.calls >= 3  # compose, immediately before G0, before credentials.
    assert len(generator.proposal_requests) == 5
    assert len(generator.reflection_requests) == 1
    assert len(evaluator.calls) == 11
    assert len(set(evaluator.calls)) == 11
    assert result.final_state.logical_llm_calls == 6
    assert result.final_state.unique_evaluations == 11
    assert len(result.final_state.candidates) == 12

    composition = live.analysis_composition
    curation = composition.feedback_interceptor
    assert type(curation) is G3PostsealCurationInterceptor
    assert curation.curation_authority is not None
    assert curation.curation_receipt is not None
    assert curation.curation_receipt.curation_status == expected_status
    assert len(curation.curation_receipt.call_receipt.publications) == (
        expected_publications
    )
    cache = asyncio.run(composition.engine.evaluation_cache_snapshot())
    validation = validate_g3_causal_screen_result(
        result,
        planner=composition.planner,
        evaluation_cache_snapshot=cache,
        curation_spec=curation.spec,
        curation_authority=curation.curation_authority,
        curation_receipt=curation.curation_receipt,
    )
    assert validation.curation_status == expected_status
    reflection = curation.curation_receipt.call_receipt
    provider_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []

    async def low_level_evidence_runner(
        request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        request_record = structured_generation_request_evidence_record(request)
        request_rows.append(request_record)
        call_id = request.call_id.value
        if call_id in generator.proposal_results:
            draft = generator.proposal_results[call_id].draft
            assert type(draft) is FiniteVariationSelectionDraft
            typed_value = request.output_type(
                option_id=draft.option_id,
                design_rationale=draft.design_rationale,
                claimed_insight_ids=list(draft.claimed_insight_ids),
            )
        elif curation_mode == "transport_failure":
            provider_rows.append(
                {
                    "schema_version": 5,
                    "task_id": call_id,
                    "status": "terminal_failure",
                    "cancellation_reason": None,
                    "queue_time_ns": 0,
                    "service_time_ns": 1,
                    "total_time_ns": 1,
                    "attempts": [
                        {
                            "attempt_number": 1,
                            "status": "terminal_failure",
                            "wait_time_ns": 0,
                            "service_time_ns": 1,
                            "will_retry": False,
                            "policy_backoff_ns": 0,
                            "retry_after_ns": 0,
                            "scheduled_delay_ns": 0,
                            "error_type": "RuntimeError",
                            "request_evidence": {
                                "variant": "original",
                                "prompt_sha256": request_record[
                                    "wire_prompt_sha256"
                                ],
                                "provider_attempt_id": (
                                    "provider_attempt_fixture_failure"
                                ),
                            },
                            "classification": {
                                "disposition": "fail",
                                "reason": "permanent",
                            },
                            "failure": {
                                "kind": "permanent",
                                "retryable": False,
                                "safe_message": "fixture provider failure",
                                "status_code": None,
                                "retry_after_seconds": None,
                                "stream_timeout_phase": None,
                                "output_failure_mode": None,
                                "validation_issues": [],
                            },
                        }
                    ],
                    "response": None,
                }
            )
            raise RuntimeError("fixture provider failure")
        else:
            reflection_result = generator.reflection_results[call_id]
            typed_value = request.output_type(
                insights=[
                    _reflection_output_item(value)
                    for value in reflection_result.insights
                ]
            )
        response = StructuredGenerationResponse(
            value=typed_value,
            requested_model=provider_profile.model_alias,
            resolved_model=provider_profile.canonical_model,
            resolved_provider=provider_profile.resolved_provider,
            provider_response_id=f"response-{call_id}",
            finish_reason="fixture",
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1,
        )
        outcome = _provider_success_outcome(request, response)
        provider_rows.append(structured_generation_outcome_record(outcome))
        output_rows.append(
            structured_generation_output_evidence_record(
                request,
                outcome,
                request_evidence=request_record,
            )
        )
        return response

    async def build_durable_provider_fixture() -> None:
        adapter = PydanticAIAgenticGenerator(low_level_evidence_runner)
        for request in generator.proposal_requests:
            await adapter.propose(request)
        try:
            await adapter.reflect(generator.reflection_requests[0])
        except RuntimeError:
            if curation_mode != "transport_failure":
                raise

    asyncio.run(build_durable_provider_fixture())
    assert len(request_rows) == len(provider_rows) == 6
    assert len(output_rows) == (5 if curation_mode == "transport_failure" else 6)
    outcome_path = tmp_path / "provider_outcomes.jsonl"
    request_path = tmp_path / "provider_requests.jsonl"
    output_path = tmp_path / "provider_outputs.jsonl"
    outcome_journal = DurableJsonlJournal(outcome_path)
    request_journal = DurableJsonlJournal(request_path)
    output_journal = DurableJsonlJournal(output_path)
    for row in provider_rows:
        outcome_journal.append(row)
    for row in request_rows:
        request_journal.append(row)
    for row in output_rows:
        output_journal.append(row)
    outcome_journal.close()
    request_journal.close()
    output_journal.close()
    durable_provider_rows = read_jsonl(outcome_path)
    durable_request_rows = read_jsonl(request_path)
    durable_output_rows = read_jsonl(output_path)
    analysis = asyncio.run(
        analyze_airfoil_g3_live_result(
            composition=composition,
            inputs=inputs,
            result=result,
            runtime_manifest_sha256="1" * 64,
            provider_outcomes=durable_provider_rows,
            provider_requests=durable_request_rows,
            provider_outputs=durable_output_rows,
            raw_evaluator_receipt_paths=evaluator.receipt_paths,
            run_started_at_utc="2026-07-15T00:00:00+00:00",
            run_finished_at_utc="2026-07-15T00:00:01+00:00",
            end_to_end_wall_seconds=1.0,
            provider_profile=provider_profile,
        )
    )
    assert analysis.curation_status == expected_status
    analysis_record = analysis.to_record()
    assert analysis.curation_classification == {
        "revision": "accepted_revision",
        "abstain": "true_abstention",
        "transport_failure": "provider_failure",
        "contract_failure": "rejected_nonempty",
    }[curation_mode]
    assert analysis_record["curation_classification"] == (
        analysis.curation_classification
    )
    structured_evidence = analysis_record["structured_evidence"]
    assert structured_evidence["request"]["request_records"] == 6
    assert structured_evidence["output"]["output_records"] == len(
        durable_output_rows
    )
    telemetry_record = analysis_record["telemetry"]
    if curation_mode == "transport_failure":
        assert telemetry_record["accounting_complete"] is False
        assert telemetry_record["reported_total_cost_usd"] is None
        assert telemetry_record["total_input_tokens"] is None
    else:
        assert telemetry_record["accounting_complete"] is True
        assert telemetry_record["reported_total_cost_usd"] == "0"
    if curation_mode == "contract_failure":
        assert telemetry_record["successful_response_telemetry_rows"] == 6
        assert durable_provider_rows[-1]["status"] == "succeeded"
        assert analysis.reflection_failure_type is not None
    # This specifically exercises the post-run invariant after a successful
    # revision appended one new version to the otherwise frozen memory bank.
    inputs.__post_init__()
    rows = read_jsonl(trace_path)
    assert rows
    assert {str(row["source"]) for row in rows} == {
        "engine",
        "optimizer",
        "planner",
    }
    with pytest.raises(AirfoilG3LiveError, match="single-use"):
        asyncio.run(live.run())


def test_live_route_snapshot_authenticates_temperature_and_large_output_budget() -> None:
    from examples.benchmarks.engibench_airfoil.v7_g3_live import (
        MAX_OUTPUT_TOKENS,
        MAX_REASONING_TOKENS,
        bind_streamlake_route,
        build_openrouter_config,
    )

    route = bind_streamlake_route()
    config = build_openrouter_config()
    assert route["max_completion_tokens"] == 384_000
    assert MAX_OUTPUT_TOKENS == MAX_REASONING_TOKENS == 384_000
    assert config.max_attempts == 2
    assert config.provider_only == ("streamlake",)
    assert config.stream_liveness_policy.absolute_timeout_ns == 600_000_000_000


def test_provider_profile_absolute_deadline_is_fail_closed() -> None:
    profile = DEEPSEEK_G3_PROVIDER_PROFILE

    with pytest.raises(ValueError, match="positive exact integer"):
        replace(profile, absolute_timeout_seconds=0)
    with pytest.raises(ValueError, match="must not exceed 1800"):
        replace(profile, absolute_timeout_seconds=1_801)
    with pytest.raises(ValueError, match="must cover first-event and idle"):
        replace(profile, absolute_timeout_seconds=179)


def test_gpt56_sol_profile_is_azure_only_xhigh_and_uses_provider_maximum() -> None:
    profile = GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE
    route = bind_provider_route(profile)
    config = build_openrouter_config(profile)
    telemetry = build_telemetry_policy(profile)

    assert route["requested_model"] == "openai/gpt-5.6-sol"
    assert route["canonical_model"] == "openai/gpt-5.6-sol-20260709"
    assert route["resolved_provider"] == "Azure"
    assert route["max_completion_tokens"] == 128_000
    assert config.provider_only == ("azure",)
    assert config.provider_options == {
        "only": ["azure"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert config.reasoning_config is not None
    assert config.reasoning_config.to_model_setting() == {"effort": "xhigh"}
    assert config.stream_liveness_policy.first_event_timeout_ns == 600_000_000_000
    assert config.stream_liveness_policy.idle_timeout_ns == 300_000_000_000
    assert profile.absolute_timeout_seconds == 600
    assert config.stream_liveness_policy.absolute_timeout_ns == 600_000_000_000
    assert profile.temperature is None
    assert telemetry.max_output_tokens == 128_000
    assert telemetry.max_reasoning_tokens == 128_000
    assert telemetry.max_cost_usd == Decimal("4.000000")
