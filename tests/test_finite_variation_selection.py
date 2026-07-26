"""Provider-free contract for sealed, coordinated finite variation selection."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    InvocationPlan,
    MutationContract,
    MutationResponseMode,
    OperatorKind,
    ProposalAuthority,
)
from agent_evolve.application.budgeted_optimizer import _invocation_plan_record
from agent_evolve.application.insight_memory import InsightMemoryBank
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    FINITE_VARIATION_SELECTION_TOOL_NAME,
    PydanticAIAgenticGenerator,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.policies.variation.typed_patch import derive_patch
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    FiniteVariationSelectionDraft,
    ReflectionGenerationResult,
    VariationGenerationRequest,
    VariationGenerationResult,
    resolve_finite_variation_selection,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)
from agent_evolve.ports.variation_catalog import (
    FiniteVariationCatalog,
    bind_finite_variation_catalog,
)


class _Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    coordinates: list[float]
    control: float


class _CoordinatedCatalog:
    catalog_id = "fixture_coordinated"
    catalog_version = 1
    definition_sha256 = "a" * 64

    def options(
        self,
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        parent_sha256 = typed_json_sha256(parent_configuration)
        return (
            FiniteVariationOption(
                option_id="shape.raise_and_trim",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json(
                    {"coordinates": [0.2, -0.1, 0.3], "control": 1.5}
                ),
                family="shape_and_control",
                description="Raise two coordinates and apply positive control.",
                metadata=(("amplitude", "medium"),),
            ),
            FiniteVariationOption(
                option_id="shape.lower_and_trim",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json(
                    {"coordinates": [-0.2, 0.1, -0.3], "control": -1.5}
                ),
                family="shape_and_control",
                description="Lower two coordinates and apply negative control.",
                metadata=(("amplitude", "medium"),),
            ),
        )


class _Problem:
    candidate_model = _Candidate
    objectives = (ObjectiveSpec("score", "min"),)

    def __init__(self) -> None:
        self.evaluations = 0

    @staticmethod
    def search_space_description() -> str:
        return "A generic vector and scalar control co-optimization problem."

    @staticmethod
    def validate(configuration: object) -> bool:
        _Candidate.model_validate(
            configuration,
            strict=True,
            by_alias=False,
            by_name=True,
        )
        return True

    def evaluate(self, configuration: dict[str, Any]) -> dict[str, float]:
        self.evaluations += 1
        return {
            "score": float(
                sum(abs(value) for value in configuration["coordinates"])
                + abs(configuration["control"])
            )
        }


def _telemetry() -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="fixture/model",
        resolved_model="fixture/model",
        resolved_provider="fixture",
        provider_response_id="fixture-response",
        finish_reason="stop",
        input_tokens=10,
        output_tokens=4,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
    )


class _SelectingGenerator:
    def __init__(self, contract: FiniteVariationContract) -> None:
        self.contract = contract
        self.requests: list[VariationGenerationRequest] = []

    async def propose(
        self,
        request: VariationGenerationRequest,
    ) -> VariationGenerationResult:
        self.requests.append(request)
        option = self.contract.options[0]
        return VariationGenerationResult(
            draft=FiniteVariationSelectionDraft(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                contract_identity_sha256=self.contract.identity_sha256,
                design_rationale="Select the coordinated sealed movement.",
            ),
            telemetry=_telemetry(),
        )

    async def reflect(self, request: object) -> ReflectionGenerationResult:
        del request
        return ReflectionGenerationResult(insights=(), telemetry=_telemetry())


def test_model_selects_one_literal_id_and_engine_retains_sealed_multi_edit() -> None:
    parent = freeze_json({"coordinates": [0.0, 0.0, 0.0], "control": 0.0})
    assert type(parent) is FrozenJsonObject
    catalog = _CoordinatedCatalog()
    assert isinstance(catalog, FiniteVariationCatalog)
    contract = bind_finite_variation_catalog(catalog, parent)
    captured: list[StructuredGenerationRequest[Any]] = []

    async def generate_once(
        request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        captured.append(request)
        schema = request.output_type.model_json_schema()
        assert schema["properties"]["option_id"]["enum"] == [
            "shape.raise_and_trim",
            "shape.lower_and_trim",
        ]
        assert "configuration" not in schema["properties"]
        with pytest.raises(ValidationError):
            request.output_type.model_validate(
                {
                    "option_id": "shape.unsealed",
                    "design_rationale": "Try an unsealed action.",
                },
                strict=True,
            )
        proposal = request.output_type.model_validate(
            {
                "option_id": "shape.raise_and_trim",
                "design_rationale": "Coupled movement matches the observed trend.",
                "claimed_insight_ids": ["insight_fixture"],
            },
            strict=True,
        )
        return StructuredGenerationResponse(
            value=proposal,
            requested_model="fixture/model",
            resolved_model="fixture/model",
            resolved_provider="fixture",
            provider_response_id="fixture-response",
            finish_reason="stop",
            input_tokens=10,
            output_tokens=4,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1,
        )

    result = asyncio.run(
        PydanticAIAgenticGenerator(generate_once).propose(
            VariationGenerationRequest(
                call_id=LLMCallId("call_finite_variation_fixture"),
                operation="typed_mutation",
                prompt="Select one sealed option from the supplied palette.",
                candidate_model=_Candidate,
                finite_variation_contract=contract,
            )
        )
    )

    assert captured[0].output_tool_name == FINITE_VARIATION_SELECTION_TOOL_NAME
    assert type(result.draft) is FiniteVariationSelectionDraft
    option = resolve_finite_variation_selection(contract, result.draft)
    assert result.draft.option_identity_sha256 == option.identity_sha256
    assert result.draft.contract_identity_sha256 == contract.identity_sha256
    assert thaw_json(option.child_configuration) == {
        "coordinates": [0.2, -0.1, 0.3],
        "control": 1.5,
    }
    assert option.prompt_record() == {
        "option_id": "shape.raise_and_trim",
        "family": "shape_and_control",
        "description": "Raise two coordinates and apply positive control.",
        "metadata": {"amplitude": "medium"},
    }


def test_engine_materializes_selected_coordinated_option_and_binds_receipts() -> None:
    async def scenario():
        problem = _Problem()
        ids = DeterministicIdFactory("finite_option_engine")
        traces: list[dict[str, object]] = []
        parent_configuration = freeze_json(
            {"coordinates": [0.0, 0.0, 0.0], "control": 0.0}
        )
        assert type(parent_configuration) is FrozenJsonObject
        contract = bind_finite_variation_catalog(
            _CoordinatedCatalog(),
            parent_configuration,
        )
        generator = _SelectingGenerator(contract)
        engine = AgenticEvolutionEngine(
            problem=problem,
            generator=generator,
            id_factory=ids,
            memory=InsightMemoryBank(id_factory=ids),
            seed=7,
            trace_sink=traces.append,
        )
        parent = await engine.register_seed(
            {"coordinates": [0.0, 0.0, 0.0], "control": 0.0},
            label="parent",
        )
        assert type(parent.configuration) is FrozenJsonObject
        mutation_contract = MutationContract(
            editable_paths=(
                JsonPath((ObjectKey("coordinates"), ArrayIndex(0))),
                JsonPath((ObjectKey("coordinates"), ArrayIndex(1))),
                JsonPath((ObjectKey("coordinates"), ArrayIndex(2))),
                JsonPath((ObjectKey("control"),)),
            ),
            max_changed_paths=4,
            max_operations=4,
        )
        plan = InvocationPlan(
            OperatorKind.TYPED_MUTATION,
            (parent,),
            generation=1,
            label="coordinated",
            allowed_top_level=("coordinates", "control"),
            mutation_contract=mutation_contract,
            mutation_response_mode=(
                MutationResponseMode.FINITE_OPTION_SELECTION_V1
            ),
            finite_variation_contract=contract,
        )
        (outcome,) = await engine.run_invocations((plan,))
        return problem, generator, traces, parent, contract, plan, outcome

    problem, generator, traces, parent, contract, plan, outcome = asyncio.run(
        scenario()
    )
    candidate = outcome.candidate
    assert candidate is not None
    assert candidate.configuration_dict == {
        "coordinates": [0.2, -0.1, 0.3],
        "control": 1.5,
    }
    assert candidate.operator_compliant and candidate.evidence_compliant
    assert len(candidate.source_attribution) == 4
    assert {item.source for item in candidate.source_attribution} == {"mutation"}
    assert outcome.prepared.proposal_authority is ProposalAuthority.MODEL
    assert outcome.prepared.call_id is not None
    assert outcome.prepared.variation_case.operator_version == 3
    assert problem.evaluations == 2
    assert generator.requests[0].finite_variation_contract == contract
    assert generator.requests[0].atomic_mutation_contract is None
    assert "ORDERED FINITE VARIATION OPTIONS" in outcome.prepared.prompt
    assert "do not author or return a candidate configuration" in (
        outcome.prepared.prompt
    )

    independent = derive_patch(
        parent.configuration,
        candidate.configuration,
        base_candidate_id=parent.candidate_id,
        target_candidate_id=candidate.candidate_id,
    )
    prepared = next(
        event for event in traces if event["event_type"] == "invocation_prepared"
    )
    evaluated = next(
        event for event in traces if event["event_type"] == "candidate_evaluated"
    )
    assert prepared["finite_variation_contract_sha256"] == contract.identity_sha256
    assert prepared["finite_variation_contract"] == contract.evidence_record()
    assert evaluated["finite_option_id"] == "shape.raise_and_trim"
    assert evaluated["finite_option_family"] == "shape_and_control"
    assert evaluated["finite_option_identity_sha256"] == (
        contract.options[0].identity_sha256
    )
    assert evaluated["finite_contract_identity_sha256"] == contract.identity_sha256
    assert evaluated["materialized_patch_hash"] == independent.patch_hash
    assert evaluated["source_attribution_provenance"] == "catalog_materialized"
    assert _invocation_plan_record(plan)["finite_variation_contract"] == (
        contract.evidence_record()
    )

    bad_option = FiniteVariationOption(
        option_id="shape.outside_contract",
        parent_configuration_sha256=contract.parent_configuration_sha256,
        child_configuration=freeze_json(
            {"coordinates": [0.0, 0.0, 0.0], "control": 2.0}
        ),
        family="shape_and_control",
        description="Change a coordinate outside the declared machine boundary.",
    )
    bad_contract = FiniteVariationContract(
        catalog_id="fixture_coordinated",
        catalog_version=1,
        catalog_definition_sha256="b" * 64,
        parent_configuration=contract.parent_configuration,
        options=(bad_option,),
    )
    with pytest.raises(ValueError, match="outside its machine contract"):
        InvocationPlan(
            OperatorKind.TYPED_MUTATION,
            (parent,),
            generation=1,
            label="bad_scope",
            allowed_top_level=("coordinates", "control"),
            mutation_contract=MutationContract(
                (
                    JsonPath((ObjectKey("coordinates"), ArrayIndex(0))),
                )
            ),
            mutation_response_mode=(
                MutationResponseMode.FINITE_OPTION_SELECTION_V1
            ),
            finite_variation_contract=bad_contract,
        )
