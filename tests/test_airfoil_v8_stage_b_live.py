"""Provider/CFD-free end-to-end gate for Airfoil's real Stage-B A/U path."""

from __future__ import annotations

import asyncio
import hashlib
import json
from decimal import Decimal
from functools import cache
from pathlib import Path

from agent_evolve.agentic import (
    AgenticCallTelemetry,
    CandidateId,
    DetailedEvaluationPayload,
    FiniteVariationSelectionDraft,
    HypothesisCompilationRequest,
    OperatorKind,
    VariationGenerationResult,
    thaw_json,
)
from agent_evolve.domain.artifact import artifact_ref_for_bytes
from agent_evolve.ports.artifact_store import canonical_json_bytes
from examples.benchmarks.engibench_airfoil import v7_g3_release as release
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    ADFLOW_EVALUATOR_ID,
    EVIDENCE_CONTRACT_ID,
    V2_EVALUATOR_ID,
)
from examples.benchmarks.engibench_airfoil.problem_def import candidate_sha256
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    MAX_OUTPUT_TOKENS,
    OwnedAgenticGenerator,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    compose_airfoil_g3_runtime_inputs,
)
from examples.benchmarks.engibench_airfoil.v7_g3_release import (
    AirfoilV7TrimHypothesisCompiler,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    EVALUATOR_IDENTITY,
    OBJECTIVE_NAME,
    VIOLATION_NAME,
    AirfoilV7Problem,
)
from examples.benchmarks.engibench_airfoil.v8_stage_b_live import (
    AIRFOIL_V8_STAGE_B_BUDGET,
    AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256,
    AIRFOIL_V8_STAGE_B_CARD_LINEAGE_SHA256,
    AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
    airfoil_v8_stage_b_readiness_record,
    compose_airfoil_v8_stage_b_inputs,
    compose_airfoil_v8_stage_b_live,
    compose_airfoil_v8_stage_b_optimizer,
)
from examples.benchmarks.engibench_airfoil.v8_stage_b_runner import (
    _result_record,
)
from examples.benchmarks.engibench_airfoil.v9_stage_b_transfer import (
    airfoil_v9_stage_b_transfer_readiness_record,
    compose_airfoil_v9_stage_b_transfer_inputs,
    rank_airfoil_v9_transfer_parent_panel,
)


class _NoRawCFD:
    def evaluate_raw(self, configuration):
        del configuration
        raise AssertionError("provider-free Stage-B test must not invoke raw CFD")


@cache
def _prepared() -> release.AirfoilG3ReleasePreparation:
    return release.prepare_release()


class _FastDetailedEvaluator:
    evaluator_identity = EVALUATOR_IDENTITY

    def __init__(self, receipt_root: Path) -> None:
        self.calls: list[str] = []
        self.receipt_root = receipt_root
        self.receipt_root.mkdir(parents=True)

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
        content = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
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


def _telemetry(call_id: str) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="deepseek/deepseek-v4-pro",
        resolved_model="deepseek/deepseek-v4-pro-20260423",
        resolved_provider="StreamLake",
        provider_response_id=f"response-{call_id}",
        finish_reason="tool_call",
        input_tokens=100,
        output_tokens=20,
        reasoning_tokens=10,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0.0001"),
        latency_ns=1,
        attempt_count=1,
    )


class _DifferentFromUniformGenerator:
    def __init__(self, planner_getter) -> None:
        self.planner_getter = planner_getter
        self.requests = []

    async def propose(self, request):
        self.requests.append(request)
        planner = self.planner_getter()
        uniform = planner.uniform_decision
        assert uniform is not None
        contract = request.finite_variation_contract
        assert contract is not None
        selected = next(
            option for option in contract.options if option.option_id != uniform.option_id
        )
        return VariationGenerationResult(
            draft=FiniteVariationSelectionDraft(
                option_id=selected.option_id,
                option_identity_sha256=selected.identity_sha256,
                contract_identity_sha256=contract.identity_sha256,
                design_rationale=(
                    "Choose the strongest compatible local trim magnitude from "
                    "the exact learned-card K=8 support."
                ),
                claimed_insight_ids=(
                    "insight_airfoil_twostage_cards_000002",
                ),
            ),
            telemetry=_telemetry(request.call_id.value),
        )

    async def reflect(self, request):
        del request
        raise AssertionError("one-generation Stage-B block must not reflect")


def test_real_airfoil_stage_b_a_u_path_runs_provider_and_cfd_free(tmp_path) -> None:
    preparation = _prepared()
    permutation, _, _ = release.freeze_diagnostic_permutation(preparation)
    problem = AirfoilV7Problem(raw_problem=_NoRawCFD())
    evaluator = _FastDetailedEvaluator(tmp_path / "receipts")
    problem.detailed_evaluator = evaluator
    source = compose_airfoil_g3_runtime_inputs(
        problem=problem,
        preparation=preparation,
        diagnostic_permutation=permutation,
    )
    inputs = compose_airfoil_v8_stage_b_inputs(source)
    readiness = airfoil_v8_stage_b_readiness_record(inputs)

    finite_contract = inputs.benchmark.bind_finite_variation(
        "airfoil_v7_trim",
        inputs.seed_configuration,
    )
    compiler_request = HypothesisCompilationRequest(
        reference=inputs.learned_card.reference,
        insight=inputs.learned_card.draft,
        source_evidence_sha256=hashlib.sha256(
            b"canonical G3 v2 registered source evidence probe"
        ).hexdigest(),
        requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
        source_operator_kinds=inputs.learned_card.applicable_operator_kinds,
        parent_candidate_id=CandidateId("candidate_airfoil_v8_compiler_probe"),
        parent_configuration_sha256=finite_contract.parent_configuration_sha256,
        finite_contract=finite_contract,
        context_projection_sha256=(
            inputs.planner_factory.context_projection_sha256
        ),
        endpoint_definition_sha256=(
            inputs.planner_factory.endpoint_definition_sha256
        ),
    )
    legacy_receipt = AirfoilV7TrimHypothesisCompiler().compile(compiler_request)
    assert legacy_receipt.applicable is False
    assert legacy_receipt.reason_codes == ("foreign_source_operator_scope",)
    stage_b_receipt = inputs.benchmark.hypothesis_compiler.compile(compiler_request)
    assert stage_b_receipt.applicable is True
    assert stage_b_receipt.spec.source_operator_kinds == ("typed_mutation",)

    assert readiness["ready"] is True
    assert readiness["budget"] == AIRFOIL_V8_STAGE_B_BUDGET.to_trace_record()
    assert readiness["same_support"]["cardinality"] == (
        AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY
    )
    assert inputs.learned_card.draft.content_sha256 == (
        AIRFOIL_V8_STAGE_B_CARD_CONTENT_SHA256
    )
    assert inputs.learned_card.evidence_lineage is not None
    assert inputs.learned_card.evidence_lineage.identity_sha256 == (
        AIRFOIL_V8_STAGE_B_CARD_LINEAGE_SHA256
    )

    holder = {}
    generator = _DifferentFromUniformGenerator(
        lambda: holder["live"].composition.planner
    )
    credential_calls = 0
    factory_calls = 0
    close_calls = 0

    def credential_loader() -> str:
        nonlocal credential_calls
        assert len(evaluator.calls) == 1  # Seed is evaluated before provider init.
        credential_calls += 1
        return "provider-free-key"

    def generator_factory(api_key, config, sinks):
        nonlocal factory_calls, close_calls
        assert api_key == "provider-free-key"
        assert config.model_name == "deepseek/deepseek-v4-pro"
        assert config.reasoning_config.max_tokens == MAX_OUTPUT_TOKENS
        assert config.to_manifest_record()["queue"]["max_attempts"] == 2
        sinks.__post_init__()
        factory_calls += 1

        def close() -> None:
            nonlocal close_calls
            close_calls += 1

        return OwnedAgenticGenerator(generator=generator, close=close)

    traces: list[tuple[str, dict[str, object]]] = []

    def trace(source_name: str):
        return lambda row: traces.append((source_name, dict(row)))

    live = compose_airfoil_v8_stage_b_live(
        inputs,
        credential_loader=credential_loader,
        progress_sink=lambda row: None,
        outcome_sink=lambda row: None,
        request_evidence_sink=lambda row: None,
        output_evidence_sink=lambda row: None,
        engine_trace_sink=trace("engine"),
        optimizer_trace_sink=trace("optimizer"),
        generator_factory=generator_factory,
    )
    holder["live"] = live
    assert live.generator.initialized is False

    result = asyncio.run(live.run())
    asyncio.run(live.aclose())

    durable_result = _result_record(result, live)
    # This exact boundary caught a real post-optimizer finalization failure:
    # ArtifactId is a value object and must be projected to its string value.
    encoded_result = json.dumps(
        durable_result,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert encoded_result
    assert canonical_json_bytes(durable_result)
    assert durable_result["arms"]["A"]["candidate"][
        "evaluation_receipt"
    ].startswith("artifact_")
    assert durable_result["arms"]["U"]["candidate"][
        "evaluation_receipt"
    ].startswith("artifact_")

    assert credential_calls == factory_calls == close_calls == 1
    assert len(generator.requests) == 1
    assert generator.requests[0].max_output_tokens == MAX_OUTPUT_TOKENS
    assert len(evaluator.calls) == 3  # P_H seed plus distinct A and U children.
    assert result.final_state.unique_evaluations == 3
    assert result.final_state.logical_llm_calls == 1
    assert result.final_state.generation == 1
    planner = live.composition.planner
    assert planner.authority is not None
    assert planner.authority.support.cardinality == 8
    assert planner.uniform_decision is not None
    assert planner.uniform_decision.option_id != (
        result.generation_receipts[0]
        .slot_results[0]
        .outcome.finite_action_decision.option_id
    )
    sealed = [
        row
        for source_name, row in traces
        if source_name == "engine"
        and row.get("event_type") == "finite_action_decision_sealed"
    ]
    assert len(sealed) == 1
    assert sealed[0]["evaluator_entered"] is False
    assert sealed[0]["authority_sha256"] == planner.authority.authority_sha256


def test_fresh_transfer_parent_and_uniform_choice_are_outcome_blind(tmp_path) -> None:
    preparation = _prepared()
    permutation, _, _ = release.freeze_diagnostic_permutation(preparation)
    problem = AirfoilV7Problem(raw_problem=_NoRawCFD())
    evaluator = _FastDetailedEvaluator(tmp_path / "transfer_receipts")
    problem.detailed_evaluator = evaluator
    source = compose_airfoil_g3_runtime_inputs(
        problem=problem,
        preparation=preparation,
        diagnostic_permutation=permutation,
    )

    panel = rank_airfoil_v9_transfer_parent_panel(source)
    assert len(panel) == 252
    assert [row.nonce for row in panel[:4]] == [88, 229, 84, 66]
    assert panel[0].selection_sha256 == (
        "007e8cfc241e1f015ba664bb033608c282884e59484c7d279265746d9feca9b8"
    )
    source_parent_hashes = {
        preparation.diagnostic_parent.candidate.configuration_sha256,
        preparation.heldout_parent.candidate.configuration_sha256,
    }
    assert not source_parent_hashes.intersection(
        row.candidate.configuration_sha256 for row in panel
    )

    inputs = compose_airfoil_v9_stage_b_transfer_inputs(source, panel_index=0)
    readiness = airfoil_v9_stage_b_transfer_readiness_record(
        source,
        inputs,
        panel_index=0,
    )
    assert evaluator.calls == []
    assert readiness["transfer_parent"]["nonce"] == 88
    assert readiness["transfer_parent"]["outcomes_read_by_selection"] is False
    assert readiness["prospective_uniform"]["outcomes_read"] is False
    assert readiness["prospective_uniform"]["selected_ordinal"] == 1
    assert readiness["prospective_uniform"]["option_id"] == (
        "trim.p050.n025.p025"
    )
    assert inputs.seed_configuration != thaw_json(
        preparation.heldout_parent.candidate.configuration
    )

    holder = {}
    generator = _DifferentFromUniformGenerator(
        lambda: holder["composition"].planner
    )
    composition = compose_airfoil_v8_stage_b_optimizer(
        inputs,
        generator=generator,
    )
    holder["composition"] = composition
    result = asyncio.run(
        composition.optimizer.run((inputs.seed_configuration,))
    )
    assert result.final_state.unique_evaluations == 3
    assert len(evaluator.calls) == 3
    assert composition.planner.uniform_decision is not None
    assert composition.planner.uniform_decision.selected_ordinal == (
        readiness["prospective_uniform"]["selected_ordinal"]
    )
    assert composition.planner.uniform_decision.option_id == (
        readiness["prospective_uniform"]["option_id"]
    )
