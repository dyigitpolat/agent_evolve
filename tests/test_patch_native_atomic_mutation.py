from __future__ import annotations

import asyncio
import copy
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationPlan,
    MaterializedInvocation,
    MutationContract,
    MutationResponseMode,
    OperatorKind,
    ProposalAuthority,
)
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.integrations.pydantic_ai import agentic_generator as adapter_module
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    ATOMIC_MUTATION_TOOL_NAME,
    PydanticAIAgenticGenerator,
)
from agent_evolve.policies.variation.typed_patch import derive_patch
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    AtomicMutationDraft,
    AtomicMutationOutputContract,
    CandidateDraft,
    ReflectionGenerationResult,
    SourceAttribution,
    VariationGenerationRequest,
    VariationGenerationResult,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)
from examples.benchmarks.boils_abc.actions import (
    ACTION_IDS,
    DEFAULT_ACTION_SEQUENCE,
    CandidateConfig,
)


INDEX = 7
PATH = JsonPath((ObjectKey("sequence"), ArrayIndex(INDEX)))
PATH_TEXT = "$.sequence[7]"
PARENT = {"sequence": list(DEFAULT_ACTION_SEQUENCE)}


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fake",
        provider_response_id="response",
        finish_reason="stop",
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
    )


def _structured_response(value: Any) -> StructuredGenerationResponse[Any]:
    return StructuredGenerationResponse(
        value=value,
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="fake",
        provider_response_id="response",
        finish_reason="stop",
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
    )


class _BoilsFixtureProblem:
    candidate_model = CandidateConfig
    objectives = (ObjectiveSpec("score", "min"),)

    def __init__(self) -> None:
        self.evaluations = 0

    @staticmethod
    def search_space_description() -> str:
        return "A length-20 BOiLS action sequence."

    @staticmethod
    def validate(configuration: object) -> bool:
        CandidateConfig.model_validate(
            configuration,
            strict=True,
            by_alias=False,
            by_name=True,
        )
        return True

    def evaluate(self, configuration: dict[str, Any]) -> dict[str, float]:
        self.evaluations += 1
        return {"score": float(configuration["sequence"].count("refactor_z"))}


class _FixedGenerator:
    def __init__(self, draft: object) -> None:
        self.draft = draft
        self.requests: list[VariationGenerationRequest] = []

    async def propose(self, request: VariationGenerationRequest):
        self.requests.append(request)
        return VariationGenerationResult(  # type: ignore[arg-type]
            draft=self.draft,
            telemetry=_telemetry(),
        )

    async def reflect(self, request):
        del request
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


async def _engine_scenario(
    draft: object,
    *,
    atomic: bool = True,
    replacement_options: tuple[object, ...] = (),
):
    problem = _BoilsFixtureProblem()
    ids = DeterministicIdFactory("patch_native_atomic")
    generator = _FixedGenerator(draft)
    traces: list[dict[str, object]] = []
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=InsightMemoryBank(id_factory=ids),
        seed=1,
        trace_sink=traces.append,
    )
    parent = await engine.register_seed(copy.deepcopy(PARENT), label="parent")
    (outcome,) = await engine.run_invocations(
        (
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (parent,),
                generation=1,
                label="index_7",
                allowed_top_level=("sequence",),
                mutation_contract=MutationContract((PATH,)),
                mutation_response_mode=(
                    MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
                    if atomic
                    else MutationResponseMode.FULL_CONFIGURATION
                ),
                atomic_replacement_options=replacement_options,
            ),
        )
    )
    return problem, generator, parent, outcome, traces, engine


def test_exact_boils_index_7_regression_materializes_one_replayed_patch() -> None:
    problem, generator, parent, outcome, traces, engine = asyncio.run(
        _engine_scenario(
            AtomicMutationDraft(
                path=PATH,
                replacement="resub_z",
                design_rationale="Replace the failed refactor action only.",
            )
        )
    )

    candidate = outcome.candidate
    assert candidate is not None
    expected = copy.deepcopy(PARENT)
    assert expected["sequence"][INDEX] == "refactor_z"
    expected["sequence"][INDEX] = "resub_z"
    assert candidate.configuration_dict == expected
    assert all(
        candidate.configuration_dict["sequence"][index] == PARENT["sequence"][index]
        for index in range(len(DEFAULT_ACTION_SEQUENCE))
        if index != INDEX
    )
    assert candidate.operator_compliant and candidate.evidence_compliant
    assert candidate.source_attribution == (SourceAttribution(PATH_TEXT, "mutation"),)
    assert outcome.prepared.variation_case.operator_version == 2
    assert problem.evaluations == 2
    assert asyncio.run(engine.evaluation_cache_snapshot())["misses"] == 2

    request = generator.requests[0]
    assert request.atomic_mutation_contract is not None
    assert request.atomic_mutation_contract.editable_path == PATH
    assert "Return one atomic edit, not a complete candidate" in outcome.prepared.prompt
    assert "do not return a full configuration" in outcome.prepared.prompt
    event = next(item for item in traces if item["event_type"] == "candidate_evaluated")
    independent = derive_patch(
        parent.configuration,
        candidate.configuration,
        base_candidate_id=parent.candidate_id,
        target_candidate_id=candidate.candidate_id,
    )
    assert event["materialized_patch_hash"] == independent.patch_hash
    assert event["parent_patch_hashes"] == [independent.patch_hash]
    assert event["atomic_submitted_path"] == PATH_TEXT
    assert event["source_attribution_provenance"] == "system_derived"
    assert event["parent_configuration_hash"] == parent.occurrence.configuration_hash
    assert event["target_configuration_hash"] == candidate.occurrence.configuration_hash


def test_boils_provider_schema_has_const_path_and_other_ten_of_eleven_actions() -> None:
    contract = AtomicMutationOutputContract(freeze_json(PARENT), PATH)
    captured: list[StructuredGenerationRequest[Any]] = []

    async def runner(request: StructuredGenerationRequest[Any]):
        captured.append(request)
        schema = request.output_type.model_json_schema()
        assert schema["properties"]["path"] == {
            "type": "string",
            "const": PATH_TEXT,
        }
        replacement_schema = schema["properties"]["replacement"]
        assert tuple(ACTION_IDS) == (
            "rewrite",
            "rewrite_z",
            "refactor",
            "refactor_z",
            "resub",
            "resub_z",
            "balance",
            "fraig",
            "sopb",
            "blut",
            "dsdb",
        )
        assert replacement_schema["type"] == "string"
        assert set(replacement_schema["enum"]) == set(ACTION_IDS) - {"refactor_z"}
        assert len(replacement_schema["enum"]) == 10
        proposal = request.output_type.model_validate(
            {
                "path": PATH_TEXT,
                "replacement": "resub_z",
                "design_rationale": "One exact action edit.",
            },
            strict=True,
        )
        return _structured_response(proposal)

    result = asyncio.run(
        PydanticAIAgenticGenerator(runner).propose(
            VariationGenerationRequest(
                call_id=DeterministicIdFactory("atomic_schema").new_llm_call_id(),
                operation="typed_mutation",
                prompt="Return one edit.",
                candidate_model=CandidateConfig,
                atomic_mutation_contract=contract,
            )
        )
    )
    assert captured[0].output_tool_name == ATOMIC_MUTATION_TOOL_NAME
    assert result.draft == AtomicMutationDraft(
        path=PATH,
        replacement="resub_z",
        design_rationale="One exact action edit.",
    )


def test_task_keyed_atomic_option_catalog_is_ordered_and_enforced_end_to_end() -> None:
    options = ("dsdb", "resub_z", "blut")
    contract = AtomicMutationOutputContract(
        freeze_json(PARENT),
        PATH,
        replacement_options=options,
    )
    output_type = adapter_module._atomic_mutation_proposal_type(
        CandidateConfig,
        "typed_mutation",
        contract,
    )
    schema = output_type.model_json_schema()
    assert schema["properties"]["replacement"]["enum"] == list(options)
    with pytest.raises(ValidationError, match="option catalog"):
        output_type.model_validate(
            {
                "path": PATH_TEXT,
                "replacement": "rewrite",
                "design_rationale": "valid BOiLS action but foreign to this task",
            },
            strict=True,
        )

    _, generator, _, outcome, traces, _ = asyncio.run(
        _engine_scenario(
            AtomicMutationDraft(
                PATH,
                "resub_z",
                "Choose one option from the rotated catalog.",
            ),
            replacement_options=options,
        )
    )
    assert outcome.candidate is not None and outcome.candidate.operator_compliant
    request_contract = generator.requests[0].atomic_mutation_contract
    assert request_contract is not None
    assert request_contract.replacement_options == options
    assert "ORDERED LEGAL REPLACEMENT OPTIONS" in outcome.prepared.prompt
    assert "this list is the only legal replacement catalog" in outcome.prepared.prompt
    assert "Use one short sentence for design_rationale" in outcome.prepared.prompt
    assert "Set claimed_insight_ids to []" in outcome.prepared.prompt
    prepared = next(
        event for event in traces if event["event_type"] == "invocation_prepared"
    )
    assert prepared["atomic_replacement_options"] == list(options)

    _, _, _, outside, _, _ = asyncio.run(
        _engine_scenario(
            AtomicMutationDraft(
                PATH,
                "rewrite",
                "Bypass the structured adapter with a foreign option.",
            ),
            replacement_options=options,
        )
    )
    assert outside.candidate is not None
    assert outside.candidate.operator_compliant is False
    assert outside.candidate.operator_failure == (
        "mutation used a replacement outside its atomic option catalog"
    )


def test_engine_materialized_atomic_slot_reuses_lineage_and_evaluation_without_llm() -> (
    None
):
    async def scenario():
        problem = _BoilsFixtureProblem()
        ids = DeterministicIdFactory("engine_materialized_atomic")
        generator = _FixedGenerator(object())
        traces: list[dict[str, object]] = []
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=generator,
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=1,
            trace_sink=traces.append,
        )
        parent = await engine.register_seed(copy.deepcopy(PARENT), label="parent")
        receipt_hash = "a" * 64
        (outcome,) = await engine.run_materialized_invocations(
            (
                MaterializedInvocation(
                    plan=InvocationPlan(
                        OperatorKind.TYPED_MUTATION,
                        (parent,),
                        generation=1,
                        label="coverage_slot",
                        allowed_top_level=("sequence",),
                        mutation_contract=MutationContract((PATH,)),
                        mutation_response_mode=(
                            MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
                        ),
                        atomic_replacement_options=("resub_z", "dsdb"),
                    ),
                    draft=AtomicMutationDraft(
                        PATH,
                        "resub_z",
                        "Engine-owned task-keyed coverage choice.",
                    ),
                    candidate_id=ids.new_candidate_id(),
                    materialization_policy_id="task_keyed_atomic_coverage",
                    materialization_policy_version=1,
                    materialization_receipt_hash=receipt_hash,
                ),
            )
        )
        return problem, generator, traces, outcome, receipt_hash

    problem, generator, traces, outcome, receipt_hash = asyncio.run(scenario())
    assert problem.evaluations == 2
    assert generator.requests == []
    assert outcome.candidate is not None
    assert outcome.candidate.valid and outcome.candidate.operator_compliant
    assert outcome.candidate.call_telemetry is None
    assert outcome.prepared.call_id is None
    assert outcome.prepared.proposal_authority is ProposalAuthority.ENGINE
    assert not any(event["event_type"].startswith("llm_call") for event in traces)
    prepared = next(
        event for event in traces if event["event_type"] == "invocation_prepared"
    )
    completed = next(
        event for event in traces if event["event_type"] == "invocation_completed"
    )
    for event in (prepared, completed):
        assert event["proposal_authority"] == "engine"
        assert event["materialization_policy_id"] == "task_keyed_atomic_coverage"
        assert event["materialization_receipt_hash"] == receipt_hash


def test_atomic_option_catalog_rejects_parent_duplicates_and_wrong_mode() -> None:
    with pytest.raises(ValueError, match="exclude the parent"):
        AtomicMutationOutputContract(
            freeze_json(PARENT),
            PATH,
            replacement_options=("refactor_z",),
        )
    with pytest.raises(ValueError, match="duplicates"):
        AtomicMutationOutputContract(
            freeze_json(PARENT),
            PATH,
            replacement_options=("resub", "resub"),
        )


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            {
                "path": "$.sequence[8]",
                "replacement": "resub_z",
                "design_rationale": "wrong path",
            },
            "Input should be",
        ),
        (
            {
                "path": PATH_TEXT,
                "replacement": "raw_abc_command",
                "design_rationale": "wrong action",
            },
            "Input should be",
        ),
        (
            {
                "path": PATH_TEXT,
                "replacement": "refactor_z",
                "design_rationale": "same value",
            },
            "must differ",
        ),
        (
            {
                "path": PATH_TEXT,
                "replacement": "resub_z",
                "design_rationale": "full candidate smuggling",
                "configuration": PARENT,
            },
            "Extra inputs",
        ),
    ],
)
def test_atomic_wire_schema_rejects_wrong_path_action_same_value_and_full_candidate(
    payload: dict[str, object],
    match: str,
) -> None:
    output_type = adapter_module._atomic_mutation_proposal_type(
        CandidateConfig,
        "typed_mutation",
        AtomicMutationOutputContract(freeze_json(PARENT), PATH),
    )
    with pytest.raises(ValidationError, match=match):
        output_type.model_validate(payload, strict=True)


@pytest.mark.parametrize(
    "draft",
    [
        AtomicMutationDraft(
            JsonPath((ObjectKey("sequence"), ArrayIndex(8))),
            "resub_z",
            "malicious wrong path",
        ),
        AtomicMutationDraft(PATH, "refactor_z", "malicious same value"),
        CandidateDraft(
            configuration={
                "sequence": [
                    "resub_z" if index == INDEX else value
                    for index, value in enumerate(DEFAULT_ACTION_SEQUENCE)
                ]
            },
            design_rationale="malicious complete candidate",
        ),
    ],
)
def test_engine_rejects_bypassing_fake_drafts_before_evaluation(draft: object) -> None:
    problem, _, _, outcome, _, _ = asyncio.run(_engine_scenario(draft))
    assert outcome.candidate is None
    assert outcome.call_failure_type in {"TypeError", "ValueError"}
    assert outcome.failure_stage == "candidate"
    assert problem.evaluations == 1


class _CrossFieldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    lower: int
    upper: int

    @model_validator(mode="after")
    def _ordered(self):
        if self.lower >= self.upper:
            raise ValueError("lower must remain below upper")
        return self


class _StrictIntConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    value: int


class _AliasConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        populate_by_name=True,
    )

    public_name: str = Field(alias="wire_name")


def _atomic_type(
    model: type[BaseModel],
    parent: dict[str, object],
    path: JsonPath,
) -> type[BaseModel]:
    return adapter_module._atomic_mutation_proposal_type(
        model,
        "typed_mutation",
        AtomicMutationOutputContract(freeze_json(parent), path),
    )


def test_whole_candidate_cross_field_validation_and_scalar_identity_remain_authoritative() -> (
    None
):
    cross_field = _atomic_type(
        _CrossFieldConfig,
        {"lower": 1, "upper": 3},
        JsonPath((ObjectKey("lower"),)),
    )
    with pytest.raises(ValidationError, match="lower must remain below upper"):
        cross_field.model_validate(
            {
                "path": "$.lower",
                "replacement": 4,
                "design_rationale": "invalid coordinated state",
            },
            strict=True,
        )

    strict_int = _atomic_type(
        _StrictIntConfig,
        {"value": 1},
        JsonPath((ObjectKey("value"),)),
    )
    with pytest.raises(ValidationError):
        strict_int.model_validate(
            {
                "path": "$.value",
                "replacement": True,
                "design_rationale": "bool is not an integer",
            },
            strict=True,
        )


def test_alias_resolution_uses_field_name_representation() -> None:
    output_type = _atomic_type(
        _AliasConfig,
        {"public_name": "old"},
        JsonPath((ObjectKey("public_name"),)),
    )
    schema = output_type.model_json_schema()
    assert schema["properties"]["path"]["const"] == "$.public_name"
    assert schema["properties"]["replacement"]["type"] == "string"
    assert output_type.model_validate(
        {
            "path": "$.public_name",
            "replacement": "new",
            "design_rationale": "rename exactly one field",
        },
        strict=True,
    )


def test_default_full_configuration_mode_remains_version_one_and_unchanged() -> None:
    child = copy.deepcopy(PARENT)
    child["sequence"][INDEX] = "resub_z"
    draft = CandidateDraft(
        configuration=child,
        design_rationale="Legacy full-candidate response.",
        source_attribution=(SourceAttribution(PATH_TEXT, "mutation"),),
    )
    _, generator, _, outcome, traces, _ = asyncio.run(
        _engine_scenario(draft, atomic=False)
    )
    assert outcome.candidate is not None
    assert outcome.candidate.operator_compliant
    assert outcome.prepared.plan.mutation_response_mode is (
        MutationResponseMode.FULL_CONFIGURATION
    )
    assert outcome.prepared.variation_case.operator_version == 1
    assert generator.requests[0].atomic_mutation_contract is None
    event = next(item for item in traces if item["event_type"] == "candidate_evaluated")
    assert event["source_attribution_provenance"] == "model_authored"
    assert event["materialized_patch_hash"] is None


def test_atomic_mode_admission_excludes_nonmutation_and_nonatomic_contracts() -> None:
    async def scenario():
        problem = _BoilsFixtureProblem()
        ids = DeterministicIdFactory("atomic_admission")
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=_FixedGenerator(object()),
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=1,
        )
        left = await engine.register_seed(copy.deepcopy(PARENT), label="left")
        right_config = copy.deepcopy(PARENT)
        right_config["sequence"][0] = "rewrite_z"
        right = await engine.register_seed(right_config, label="right")
        with pytest.raises(ValueError, match="only typed mutation"):
            InvocationPlan(
                OperatorKind.TWO_PARENT_CROSSOVER,
                (left, right),
                generation=1,
                label="excluded",
                mutation_contract=MutationContract((PATH,)),
                mutation_response_mode=(
                    MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
                ),
            )
        with pytest.raises(ValueError, match="requires exactly one editable path"):
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (left,),
                generation=1,
                label="too_many",
                allowed_top_level=("sequence",),
                mutation_contract=MutationContract(
                    (
                        PATH,
                        JsonPath((ObjectKey("sequence"), ArrayIndex(8))),
                    ),
                    max_changed_paths=1,
                ),
                mutation_response_mode=(
                    MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
                ),
            )

    asyncio.run(scenario())
