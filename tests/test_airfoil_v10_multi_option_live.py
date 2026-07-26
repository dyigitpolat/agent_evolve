"""Provider-free gates for the Airfoil v10 generic live composition."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from functools import cache

from agent_evolve.agentic import (
    AgenticCallTelemetry,
    DetailedEvaluationPayload,
    FiniteVariationSelectionDraft,
)
from agent_evolve.application.multi_option_evolution import (
    MULTI_OPTION_G1_SLOT_IDS,
    MULTI_OPTION_G2_SLOT_IDS,
    MULTI_OPTION_G3_SLOT_IDS,
    MultiOptionEvolutionPlanner,
)
from agent_evolve.application.post_evolution_reflection import (
    PostEvolutionReflectionInterceptor,
)
from agent_evolve.domain.artifact import artifact_ref_for_bytes
from agent_evolve.ports.artifact_store import canonical_json_bytes
from agent_evolve.ports.agentic_generator import (
    CandidateDraft,
    ExactParentCrossoverDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationResult,
)
from examples.benchmarks.engibench_airfoil import v7_g3_release as release
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    ADFLOW_EVALUATOR_ID,
    EVIDENCE_CONTRACT_ID,
    V2_EVALUATOR_ID,
)
from examples.benchmarks.engibench_airfoil.problem_def import candidate_sha256
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    compose_airfoil_g3_runtime_inputs,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    EVALUATOR_IDENTITY,
    OBJECTIVE_NAME,
    VIOLATION_NAME,
    AirfoilV7Problem,
)
from examples.benchmarks.engibench_airfoil.v10_multi_option_inputs import (
    AIRFOIL_V10_CONTEXT_PROJECTION_SHA256,
    AIRFOIL_V10_MULTI_OPTION_PHASE,
    AirfoilV10MultiOptionInputs,
    compose_airfoil_v10_multi_option_inputs,
)
from examples.benchmarks.engibench_airfoil.v10_multi_option_live import (
    AirfoilV10MultiOptionLiveComposition,
    build_airfoil_v10_openrouter_config,
    compose_airfoil_v10_multi_option_live,
    compose_airfoil_v10_multi_option_optimizer,
)


class _NoRawCFD:
    def evaluate_raw(self, configuration):
        del configuration
        raise AssertionError("provider-free v10 test must not invoke raw CFD")


class _FastDetailedEvaluator:
    evaluator_identity = EVALUATOR_IDENTITY

    def __init__(self) -> None:
        self.calls: list[str] = []

    def evaluate_evidence(self, configuration) -> DetailedEvaluationPayload:
        key = candidate_sha256(configuration)
        self.calls.append(key)
        ordinal = int(key[:16], 16)
        objective = 0.8 + (ordinal % 1_000_003) / 10_000_000.0
        violation = 0.05 + ((ordinal // 1_000_003) % 1_000_033) / 10_000_000.0
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
        return DetailedEvaluationPayload(
            failure=None,
            objectives=((OBJECTIVE_NAME, objective),),
            violations=((VIOLATION_NAME, violation),),
            checks=(),
            receipt=artifact_ref_for_bytes(
                canonical_json_bytes(record),
                media_type="application/json",
            ),
            evaluator=EVALUATOR_IDENTITY,
            active_wall_seconds=0.001,
            resource_queue_wall_seconds=None,
        )


@cache
def _prepared() -> release.AirfoilG3ReleasePreparation:
    return release.prepare_release()


def _inputs() -> tuple[AirfoilV10MultiOptionInputs, _FastDetailedEvaluator]:
    preparation = _prepared()
    permutation, _, _ = release.freeze_diagnostic_permutation(preparation)
    problem = AirfoilV7Problem(raw_problem=_NoRawCFD())
    evaluator = _FastDetailedEvaluator()
    problem.detailed_evaluator = evaluator
    source = compose_airfoil_g3_runtime_inputs(
        problem=problem,
        preparation=preparation,
        diagnostic_permutation=permutation,
    )
    return compose_airfoil_v10_multi_option_inputs(source), evaluator


def _telemetry(kind: str) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/scripted",
        resolved_model="offline/scripted",
        resolved_provider="provider-free-test",
        provider_response_id=f"airfoil-v10-{kind}",
        finish_reason="tool_call",
        input_tokens=100,
        output_tokens=20,
        reasoning_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
        attempt_count=1,
    )


class _ScriptedEvolutionGenerator:
    def __init__(self, inputs: AirfoilV10MultiOptionInputs) -> None:
        self.reference_ids = tuple(
            card.reference.insight_id.value for card in inputs.active_cards
        )
        self.proposal_requests = []
        self.reflection_requests = []

    def _reference(self, prompt: str) -> str:
        matches = tuple(value for value in self.reference_ids if value in prompt)
        assert len(matches) == 1
        return matches[0]

    async def propose(self, request):
        self.proposal_requests.append(request)
        contract = request.finite_variation_contract
        if contract is not None:
            reference = self._reference(request.prompt)
            diagnostic = '"alpha_deg":[2.75,2.75,2.75]' in request.prompt
            learned = reference == self.reference_ids[0]
            if diagnostic:
                ordinal = 0 if learned else len(contract.options) - 1
            else:
                ordinal = 1 if learned else 2
            option = contract.options[ordinal]
            return VariationGenerationResult(
                draft=FiniteVariationSelectionDraft(
                    option_id=option.option_id,
                    option_identity_sha256=option.identity_sha256,
                    contract_identity_sha256=contract.identity_sha256,
                    design_rationale=(
                        "Select one genuine action after comparing the complete "
                        "authenticated K=8 neighbourhood."
                    ),
                    claimed_insight_ids=(reference,),
                ),
                telemetry=_telemetry("finite-choice"),
            )

        crossover_contract = request.exact_parent_crossover_contract
        if crossover_contract is not None:
            candidates = (
                crossover_contract.locus_ids[::2],
                crossover_contract.locus_ids[1::2],
                (crossover_contract.locus_ids[0],),
            )
            selected = next(
                value
                for value in candidates
                if value
                and len(value) < len(crossover_contract.locus_ids)
                and value not in crossover_contract.forbidden_import_locus_sets
            )
            assert selected
            assert len(selected) < len(crossover_contract.locus_ids)
            return VariationGenerationResult(
                draft=ExactParentCrossoverDraft(
                    contract_identity_sha256=(
                        crossover_contract.contract_identity_sha256
                    ),
                    import_locus_ids=selected,
                    claimed_insight_ids=(),
                ),
                telemetry=_telemetry("exact-parent-crossover"),
            )

        parent_rows = json.loads(
            request.prompt.split("PARENTS\n", 1)[1].split("\n", 1)[0]
        )
        left = parent_rows[0]["configuration"]
        right = parent_rows[1]["configuration"]
        child = {
            "alpha_deg": list(left["alpha_deg"]),
            "lower_coefficients": list(right["lower_coefficients"]),
            "representation_id": left["representation_id"],
            "upper_coefficients": list(right["upper_coefficients"]),
        }
        return VariationGenerationResult(
            draft=CandidateDraft(
                configuration=child,
                design_rationale=(
                    "Retain the left trim contribution and the right shape "
                    "contribution in one full two-parent child."
                ),
                intended_changes=(
                    "$.lower_coefficients",
                    "$.upper_coefficients",
                ),
                source_attribution=(
                    SourceAttribution("$.alpha_deg", "left"),
                    SourceAttribution("$.lower_coefficients", "right"),
                    SourceAttribution("$.upper_coefficients", "right"),
                ),
            ),
            telemetry=_telemetry("crossover"),
        )

    async def reflect(self, request):
        self.reflection_requests.append(request)
        return ReflectionGenerationResult(
            insights=(),
            telemetry=_telemetry("terminal-reflection"),
        )


def test_live_composition_is_lazy_and_gpt_profile_is_xhigh_only() -> None:
    inputs, evaluator = _inputs()
    credential_calls: list[str] = []
    factory_calls: list[object] = []

    def credential_loader() -> str:
        credential_calls.append("read")
        return "not-used"

    def generator_factory(*args):
        factory_calls.append(args)
        raise AssertionError("provider must remain lazy during composition")

    live = compose_airfoil_v10_multi_option_live(
        inputs,
        credential_loader=credential_loader,
        progress_sink=lambda value: None,
        outcome_sink=lambda value: None,
        request_evidence_sink=lambda value: None,
        output_evidence_sink=lambda value: None,
        outbound_request_manifest_sink=lambda value: None,
        provider_profile=GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE,
        generator_factory=generator_factory,
    )

    assert type(live) is AirfoilV10MultiOptionLiveComposition
    assert live.initialized_provider is False
    assert live.run_state == "not_started"
    assert credential_calls == []
    assert factory_calls == []
    assert evaluator.calls == []
    assert type(live.composition.planner) is MultiOptionEvolutionPlanner
    assert type(live.composition.feedback_interceptor) is (
        PostEvolutionReflectionInterceptor
    )
    assert live.composition.planner.phase == AIRFOIL_V10_MULTI_OPTION_PHASE
    assert live.composition.planner.context_projection_sha256 == (
        AIRFOIL_V10_CONTEXT_PROJECTION_SHA256
    )

    config = build_airfoil_v10_openrouter_config(GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE)
    assert config.reasoning_config is not None
    assert config.reasoning_config.to_model_setting() == {"effort": "xhigh"}
    manifest_text = json.dumps(config.to_manifest_record(), sort_keys=True).lower()
    assert '"mode"' not in manifest_text
    assert '"pro"' not in manifest_text
    asyncio.run(live.aclose())


def test_provider_free_composition_runs_full_evolution_and_reflection() -> None:
    inputs, evaluator = _inputs()
    generator = _ScriptedEvolutionGenerator(inputs)
    composition = compose_airfoil_v10_multi_option_optimizer(
        inputs,
        generator=generator,
    )
    result = asyncio.run(composition.optimizer.run(inputs.seed_configurations))

    assert result.final_state.generation == 3
    assert len(result.final_state.candidates) == 14
    assert result.final_state.logical_llm_calls == 7
    assert result.final_state.unique_evaluations <= 13
    assert len(generator.proposal_requests) == 6
    assert len(generator.reflection_requests) == 1
    assert tuple(
        tuple(slot.slot.slot_id for slot in receipt.slot_results)
        for receipt in result.generation_receipts
    ) == (
        MULTI_OPTION_G1_SLOT_IDS,
        MULTI_OPTION_G2_SLOT_IDS,
        MULTI_OPTION_G3_SLOT_IDS,
    )
    assert all(
        slot.outcome.failure_stage is None
        and slot.outcome.candidate is not None
        and slot.outcome.candidate.valid
        and slot.outcome.candidate.operator_compliant
        and slot.outcome.candidate.evidence_compliant
        for receipt in result.generation_receipts
        for slot in receipt.slot_results
    )
    assert len(evaluator.calls) == result.final_state.unique_evaluations

    reflection = composition.feedback_interceptor
    assert type(reflection) is PostEvolutionReflectionInterceptor
    assert reflection.reflection_receipt is not None
    assert reflection.reflection_receipt.reflection_status == "sealed_complete"
    assert reflection.reflection_receipt.publication_outcome == ("completed_abstention")
    assert tuple(
        value.used_logical_llm_calls for value in result.feedback_receipts
    ) == (0, 0, 1)
