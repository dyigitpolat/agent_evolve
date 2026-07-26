"""Offline tests for the pinned Pydantic AI request/response codec."""

from __future__ import annotations

import hashlib
import json
import types
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Any, Generic, List, Literal, TypeVar, Union

import pytest
from annotated_types import Ge
from typing_extensions import TypedDict

pytest.importorskip("pydantic_ai")

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    create_model,
    field_validator,
)
from pydantic_core import core_schema
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition
from pydantic_ai.usage import RequestUsage

from agent_evolve.harness.base import HarnessContext, HarnessOutputError, LLMConfig
from agent_evolve.integrations.pydantic_ai import boundary_codec
from agent_evolve.integrations.pydantic_ai.boundary_codec import (
    BoundOutputContract,
    PydanticAIBoundaryCodecError,
    bind_recorded_output_type_contract,
    build_output_type_contract,
    decode_model_response,
    encode_model_request,
    encode_model_response,
    model_profile_sha256,
    project_model_request,
    validate_output_type_contract,
)
from agent_evolve.integrations.pydantic_ai.harness import PydanticAIHarness
from agent_evolve.ports.artifact_store import canonical_json_bytes


class _NeverClient:
    def __getattribute__(self, name):
        raise AssertionError("offline test touched the provider client")


def _production_model(*, settings=None, profile=None):
    # Bypass the provider constructor so no environment variable, HTTP client,
    # credential, or transport object is consulted. OpenAIChatModel only stores
    # this provider and its static profile callable during construction.
    provider = object.__new__(OpenRouterProvider)
    provider._client = _NeverClient()
    return OpenAIChatModel(
        "deepseek/deepseek-v4-pro",
        provider=provider,
        settings=settings,
        profile=profile,
    )


def _fixture_model(*, settings=None):
    async def forbidden_function(messages, info):
        raise AssertionError("offline fixture function was invoked")

    return FunctionModel(
        forbidden_function,
        model_name="fixture-model",
        settings=settings,
    )


class _NoCallCaptureModel(WrapperModel):
    def __init__(self, wrapped, output_contract):
        super().__init__(wrapped)
        self.output_contract = output_contract
        self.requests = []
        self.responses = []
        self.raw_requests = []
        self.invocations = 0

    async def request(self, messages, model_settings, model_request_parameters):
        self.invocations += 1
        self.raw_requests.append((messages, model_settings, model_request_parameters))
        self.requests.append(
            encode_model_request(
                self.wrapped,
                messages,
                model_settings,
                model_request_parameters,
                output_contract=self.output_contract,
            )
        )
        response = ModelResponse(
            parts=[TextPart(content="fixture response")],
            model_name="fixture-model",
            provider_name="fixture-provider",
        )
        self.responses.append(encode_model_response(response))
        return response


def _capture_run(output_type, prompt="same prompt", *, model=None, settings=None):
    contract = build_output_type_contract(output_type)
    capture = _NoCallCaptureModel(model or _production_model(), contract)
    agent = Agent(capture, output_type=output_type, retries=0)
    try:
        result = agent.run_sync(prompt, model_settings=settings)
    except UnexpectedModelBehavior:
        result = None
    return capture, contract, result


def _text_request_parts():
    return (
        [ModelRequest(parts=[UserPromptPart("prompt")])],
        ModelRequestParameters(),
        build_output_type_contract(str),
    )


def _graph_field(graph, name):
    return next(field for field in graph["fields"] if field["name"] == name)


def _assert_sanitized_boundary_error(caught, secret):
    error = caught.value
    assert type(error) is PydanticAIBoundaryCodecError
    assert secret not in str(error)
    assert secret not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_production_runs_omit_only_audited_openai_chat_volatile_fields():
    contract = build_output_type_contract(str)
    model = _NoCallCaptureModel(_production_model(), contract)
    agent = Agent(model, output_type=str, retries=0)
    assert agent.run_sync("same prompt", model_settings={"temperature": 0.2}).output
    assert agent.run_sync("same prompt", model_settings={"temperature": 0.2}).output
    assert model.invocations == 2
    assert model.requests[0] == model.requests[1]

    value = json.loads(model.requests[0])
    assert value["semantic_route"] == "openrouter-openai-chat-v1"
    assert value["messages"][0]["parts"] == [
        {"content": "same prompt", "part_kind": "user-prompt"}
    ]
    assert value["omitted_framework_fields"] == [
        "messages[*].timestamp",
        "messages[*].run_id",
        "messages[*].conversation_id",
        "messages[*].parts[*].timestamp",
    ]


def test_fixture_route_retains_ids_and_timestamps_and_never_calls_function():
    contract = build_output_type_contract(str)
    model = _NoCallCaptureModel(_fixture_model(), contract)
    agent = Agent(model, output_type=str, retries=0)
    assert agent.run_sync("same").output
    assert agent.run_sync("same").output
    assert model.requests[0] != model.requests[1]
    value = json.loads(model.requests[0])
    assert value["semantic_route"] == "offline-function-fixture-v1"
    assert value["omitted_framework_fields"] == []
    message = value["messages"][0]
    assert message["timestamp"].endswith("Z")
    assert message["run_id"]
    assert message["conversation_id"]
    assert message["parts"][0]["timestamp"].endswith("Z")


def test_exact_production_adapter_profile_provider_and_dependency_identity():
    model = _production_model(settings={"temperature": 0.1})
    messages, parameters, contract = _text_request_parts()
    value = project_model_request(
        model,
        messages,
        {"temperature": 0.2},
        parameters,
        output_contract=contract,
    )
    assert value["model_adapter_type"] == "pydantic_ai.models.openai.OpenAIChatModel"
    assert value["model_profile_type"] == (
        "pydantic_ai.providers.openrouter.OpenRouterModelProfile"
    )
    assert value["provider_type"] == "pydantic_ai.providers.openrouter.OpenRouterProvider"
    assert value["pydantic_ai_version"] == "1.107.1"
    assert value["pydantic_version"] == "2.13.4"
    assert value["pydantic_core_version"] == "2.46.4"
    assert len(value["model_profile"]) == 37
    assert value["model_profile"]["supports_inline_system_prompts"] is True
    assert value["model_profile"]["openai_chat_supports_multiple_system_messages"] is True
    assert value["model_profile"]["openrouter_supports_cache_control"] is False
    assert value["model_profile"]["openrouter_supports_cache_ttl"] is False
    assert value["model_profile"]["openrouter_supports_tool_cache"] is False
    assert (
        value["model_profile"]["openrouter_supports_dynamic_instruction_cache"]
        is False
    )
    assert value["model_profile"]["openrouter_max_cache_points"] is None
    assert value["model_profile"]["openai_chat_thinking_field"] == "reasoning"
    assert value["model_profile"]["openai_chat_send_back_thinking_parts"] == "field"
    assert value["model_profile"]["json_schema_transformer"] == (
        "pydantic_ai.profiles.openai.OpenAIJsonSchemaTransformer"
    )
    assert value["model_profile_sha256"] == model_profile_sha256(model)


def test_non_openrouter_adapter_and_base_profile_are_not_production_routes():
    messages, parameters, contract = _text_request_parts()
    with pytest.raises(PydanticAIBoundaryCodecError):
        encode_model_request(
            _NoCallCaptureModel(_production_model(), contract),
            messages,
            None,
            parameters,
            output_contract=contract,
        )

    from pydantic_ai.profiles import ModelProfile

    with pytest.raises(PydanticAIBoundaryCodecError):
        model_profile_sha256(_production_model(profile=ModelProfile()))


def test_prompt_setting_schema_model_defaults_and_profile_change_identity():
    first_capture, _, _ = _capture_run(
        str,
        "alpha",
        model=_production_model(settings={"temperature": 0.1}),
    )
    second_capture, _, _ = _capture_run(
        str,
        "alphb",
        model=_production_model(settings={"temperature": 0.1}),
    )
    assert first_capture.requests[-1] != second_capture.requests[-1]

    default_a, _, _ = _capture_run(
        str, model=_production_model(settings={"temperature": 0.1})
    )
    default_b, _, _ = _capture_run(
        str, model=_production_model(settings={"temperature": 0.2})
    )
    assert default_a.requests[-1] != default_b.requests[-1]

    class OutputA(BaseModel):
        value: int

    class OutputB(BaseModel):
        value: str

    typed_a, _, _ = _capture_run(OutputA, "typed")
    typed_b, _, _ = _capture_run(OutputB, "typed")
    assert typed_a.requests[-1] != typed_b.requests[-1]

    original = _production_model().profile
    changed = replace(
        original,
        openai_supports_tool_choice_required=(
            not original.openai_supports_tool_choice_required
        ),
    )
    assert model_profile_sha256(_production_model(profile=original)) != (
        model_profile_sha256(_production_model(profile=changed))
    )
    changed_openrouter = replace(
        original,
        openrouter_supports_cache_control=(
            not original.openrouter_supports_cache_control
        ),
    )
    assert model_profile_sha256(_production_model(profile=original)) != (
        model_profile_sha256(_production_model(profile=changed_openrouter))
    )


def test_structured_boundary_is_auto_and_binds_output_contract_before_prepare():
    class Output(BaseModel):
        value: int

    capture, contract, _ = _capture_run(Output, "typed")
    value = json.loads(capture.requests[-1])
    assert value["model_request_parameters"]["output_mode"] == "auto"
    assert value["model_profile"]["default_structured_output_mode"] == "tool"
    assert value["output_type_contract_sha256"] == contract.sha256_hex
    assert value["output_type_contract"]["json_schema"] == TypeAdapter(Output).json_schema()
    assert value["model_request_parameters"]["output_tools"][0]["kind"] == "output"


def test_output_contract_is_canonical_schema_and_type_sensitive():
    class OutputA(BaseModel):
        value: int

    class OutputB(BaseModel):
        value: str

    first = build_output_type_contract(OutputA)
    assert json.loads(first.canonical_bytes)["format_version"] == 3
    assert first.output_type is OutputA
    assert first == build_output_type_contract(OutputA)
    assert first != build_output_type_contract(OutputB)
    assert canonical_json_bytes(json.loads(first.canonical_bytes)) == first.canonical_bytes
    validate_output_type_contract(
        first,
        compiled_json_schema=TypeAdapter(OutputA).json_schema(),
    )
    with pytest.raises(PydanticAIBoundaryCodecError):
        validate_output_type_contract(
            first,
            compiled_json_schema=TypeAdapter(OutputB).json_schema(),
        )
    with pytest.raises(PydanticAIBoundaryCodecError):
        BoundOutputContract(
            output_type=OutputA,
            canonical_bytes=first.canonical_bytes,
            sha256_hex=first.sha256_hex,
            output_kind=first.output_kind,
        )


def test_recorded_contract_requires_expected_runtime_type_and_exact_rebuild():
    contract = build_output_type_contract(str)
    replayed = bind_recorded_output_type_contract(
        str,
        canonical_bytes=contract.canonical_bytes,
        sha256_hex=contract.sha256_hex,
    )
    assert replayed == contract
    assert replayed.output_type is str

    with pytest.raises(PydanticAIBoundaryCodecError):
        bind_recorded_output_type_contract(
            list[str],
            canonical_bytes=contract.canonical_bytes,
            sha256_hex=contract.sha256_hex,
        )

    model = _production_model()
    messages, parameters, _ = _text_request_parts()
    for changed_field in ("type_graph", "json_schema"):
        record = json.loads(contract.canonical_bytes)
        record[changed_field] = {"kind": "type", "name": "builtins.int"}
        forged_bytes = canonical_json_bytes(record)
        forged_sha256 = hashlib.sha256(forged_bytes).hexdigest()

        # Even the module-private construction capability cannot make altered
        # bytes pass: construction rebuilds from the exact runtime type.
        with pytest.raises(PydanticAIBoundaryCodecError):
            BoundOutputContract(
                output_type=str,
                canonical_bytes=forged_bytes,
                sha256_hex=forged_sha256,
                output_kind="text",
                _construction_token=boundary_codec._OUTPUT_BUILD_TOKEN,
            )
        with pytest.raises(PydanticAIBoundaryCodecError):
            bind_recorded_output_type_contract(
                str,
                canonical_bytes=forged_bytes,
                sha256_hex=forged_sha256,
            )

        # Low-level Python allocation can bypass any dataclass constructor, so
        # all public consumers independently rebuild before using the object.
        forged = object.__new__(BoundOutputContract)
        object.__setattr__(forged, "output_type", str)
        object.__setattr__(forged, "canonical_bytes", forged_bytes)
        object.__setattr__(forged, "sha256_hex", forged_sha256)
        object.__setattr__(forged, "output_kind", "text")
        with pytest.raises(PydanticAIBoundaryCodecError):
            validate_output_type_contract(forged)
        with pytest.raises(PydanticAIBoundaryCodecError):
            boundary_codec._project_parameters(
                parameters,
                output_contract=forged,
            )
        with pytest.raises(PydanticAIBoundaryCodecError):
            project_model_request(
                model,
                messages,
                None,
                parameters,
                output_contract=forged,
            )
        with pytest.raises(PydanticAIBoundaryCodecError):
            encode_model_request(
                model,
                messages,
                None,
                parameters,
                output_contract=forged,
            )


def test_dynamic_proposal_accepts_representative_constrained_candidate_graph():
    class Mode(str, Enum):
        fast = "fast"
        thorough = "thorough"

    class Nested(BaseModel):
        ratio: float = Field(gt=0.0, le=1.0)
        label: str = Field(
            min_length=2,
            max_length=20,
            pattern=r"^[a-z]+$",
        )

    class Candidate(BaseModel):
        count: int = Field(ge=0, le=10)
        price: Decimal
        mode: Mode
        strategy: Literal["a", "b"]
        nested: Nested | None
        weights: list[float]
        thresholds: list[Annotated[int, Ge(0)]]
        tags: list[str] = Field(default_factory=list)
        lookup: dict[str, int]
        options: dict[str, int] = Field(default_factory=dict)
        pair: tuple[int, str]
        choice: int | str

    proposal = create_model(
        "ProposalBoundaryRegression",
        thought_process=(str, ...),
        candidates=(list[Candidate], ...),
    )
    contract = build_output_type_contract(proposal)
    schema = TypeAdapter(proposal).json_schema()
    validate_output_type_contract(
        contract,
        compiled_json_schema=schema,
    )

    value = json.loads(contract.canonical_bytes)
    assert value["json_schema"] == schema
    proposal_graph = value["type_graph"]
    candidate_graph = _graph_field(proposal_graph, "candidates")["annotation"]["item"]
    count_metadata = _graph_field(candidate_graph, "count")["metadata"]
    assert count_metadata == [
        {"type": "annotated_types.Ge", "attributes": {"ge": 0}},
        {"type": "annotated_types.Le", "attributes": {"le": 10}},
    ]
    assert _graph_field(candidate_graph, "tags")["initialization"] == {
        "kind": "factory",
        "name": "builtins.list",
    }
    assert _graph_field(candidate_graph, "options")["initialization"] == {
        "kind": "factory",
        "name": "builtins.dict",
    }
    nested_union = _graph_field(candidate_graph, "nested")["annotation"]
    nested_graph = next(item for item in nested_union["args"] if item["kind"] == "pydantic-model")
    label_metadata = _graph_field(nested_graph, "label")["metadata"]
    assert any(item["type"] == "annotated_types.MinLen" for item in label_metadata)
    assert any(item["type"] == "annotated_types.MaxLen" for item in label_metadata)
    assert any(
        item["attributes"] == {"pattern": "^[a-z]+$"}
        for item in label_metadata
    )
    for string_list in (list[str], List[str]):
        string_list_contract = build_output_type_contract(string_list)
        assert string_list_contract.output_kind == "structured"
        validate_output_type_contract(
            string_list_contract,
            compiled_json_schema=TypeAdapter(string_list).json_schema(),
        )


def test_custom_json_schema_override_is_recorded_as_provider_schema():
    class CustomProviderSchema(BaseModel):
        value: int

        @classmethod
        def __get_pydantic_json_schema__(cls, schema, handler):
            result = handler(schema)
            result["properties"]["value"] = {
                "title": "Value",
                "type": "string",
            }
            return result

    contract = build_output_type_contract(CustomProviderSchema)
    compiled_schema = TypeAdapter(CustomProviderSchema).json_schema()
    validate_output_type_contract(
        contract,
        compiled_json_schema=compiled_schema,
    )
    record = json.loads(contract.canonical_bytes)
    graph_field = _graph_field(record["type_graph"], "value")
    assert graph_field["annotation"] == {
        "kind": "type",
        "name": "builtins.int",
    }
    assert record["json_schema"]["properties"]["value"]["type"] == "string"
    request_properties = record["request_shape"]["output_object"]["json_schema"][
        "properties"
    ]
    assert request_properties["value"]["type"] == "string"


def test_same_name_opposite_validators_need_external_code_provenance():
    def make_model(*, accepts_positive: bool):
        def gate(value: int) -> int:
            if (value > 0) is not accepts_positive:
                raise ValueError("runtime-only policy")
            return value

        return create_model(
            "SameNamedValidatorPolicy",
            __module__="recorded.runtime.policy",
            value=(int, ...),
            __validators__={"gate": field_validator("value")(gate)},
        )

    positive = make_model(accepts_positive=True)
    negative = make_model(accepts_positive=False)
    positive_contract = build_output_type_contract(positive)
    negative_contract = build_output_type_contract(negative)
    assert positive_contract.canonical_bytes == negative_contract.canonical_bytes
    assert positive_contract != negative_contract
    assert positive(value=1).value == 1
    with pytest.raises(ValidationError):
        positive(value=-1)
    assert negative(value=-1).value == -1
    with pytest.raises(ValidationError):
        negative(value=1)

    # Replay binds the independently selected runtime class. The serialized
    # provider contract intentionally cannot attest validator source code.
    for expected in (positive, negative):
        rebound = bind_recorded_output_type_contract(
            expected,
            canonical_bytes=positive_contract.canonical_bytes,
            sha256_hex=positive_contract.sha256_hex,
        )
        assert rebound.output_type is expected


def test_same_name_opposite_core_hooks_need_external_code_provenance():
    def make_model(*, accepts_positive: bool):
        def build_core_schema(cls, source_type, handler):
            schema = handler(source_type)

            def gate(value):
                if (value.value > 0) is not accepts_positive:
                    raise ValueError("runtime-only core policy")
                return value

            return core_schema.no_info_after_validator_function(gate, schema)

        return type(
            "SameNamedCoreHookPolicy",
            (BaseModel,),
            {
                "__module__": "recorded.runtime.policy",
                "__annotations__": {"value": int},
                "__get_pydantic_core_schema__": classmethod(build_core_schema),
            },
        )

    positive = make_model(accepts_positive=True)
    negative = make_model(accepts_positive=False)
    positive_contract = build_output_type_contract(positive)
    negative_contract = build_output_type_contract(negative)
    assert positive_contract.canonical_bytes == negative_contract.canonical_bytes
    assert positive_contract != negative_contract
    assert positive(value=1).value == 1
    with pytest.raises(ValidationError):
        positive(value=-1)
    assert negative(value=-1).value == -1
    with pytest.raises(ValidationError):
        negative(value=1)


def test_literal_graph_preserves_exact_bool_int_and_float_runtime_types():
    class ExactLiterals(BaseModel):
        boolean: Literal[True]
        integer: Literal[1]
        floating: Literal[1.0]

    graph = json.loads(build_output_type_contract(ExactLiterals).canonical_bytes)[
        "type_graph"
    ]
    values = {
        field["name"]: field["annotation"]["values"][0]
        for field in graph["fields"]
    }
    assert type(values["boolean"]) is bool
    assert type(values["integer"]) is int
    assert type(values["floating"]) is float


@pytest.mark.parametrize(
    "marker",
    [DeferredToolRequests, DeferredToolResults],
    ids=["requests", "results"],
)
def test_deferred_marker_classes_subclasses_and_instances_are_rejected(marker):
    class MarkerSubclass(marker):
        pass

    for output_spec in (
        marker,
        marker(),
        MarkerSubclass,
        MarkerSubclass(),
        [str, marker],
    ):
        with pytest.raises(PydanticAIBoundaryCodecError) as caught:
            build_output_type_contract(output_spec)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "marker",
    [DeferredToolRequests, DeferredToolResults],
    ids=["requests", "results"],
)
def test_deferred_markers_are_rejected_in_every_supported_annotation_position(marker):
    class MarkerSubclass(marker):
        pass

    annotations = (
        marker,
        MarkerSubclass,
        list[marker],
        dict[str, marker],
        tuple[int, marker],
        int | marker,
        Literal[marker],
        Literal[marker()],
        Annotated[marker, Ge(0)],
        Annotated[int, marker],
        Annotated[int, marker()],
        Annotated[int, MarkerSubclass],
        Annotated[int, MarkerSubclass()],
    )
    for index, annotation in enumerate(annotations):
        output = create_model(
            f"Nested{marker.__name__}{index}",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            value=(annotation, ...),
        )
        with pytest.raises(PydanticAIBoundaryCodecError):
            build_output_type_contract(output)

    # Trusted metadata is also traversed by value, not accepted solely because
    # its wrapper class is trusted.
    with pytest.raises(PydanticAIBoundaryCodecError):
        boundary_codec._metadata_record(Ge(marker))

    for index, default in enumerate((marker, marker(), [marker()])):
        output = create_model(
            f"DeferredDefault{marker.__name__}{index}",
            value=(int, default),
        )
        with pytest.raises(PydanticAIBoundaryCodecError):
            build_output_type_contract(output)

    factory_output = create_model(
        f"DeferredFactory{marker.__name__}",
        value=(int, Field(default_factory=marker)),
    )
    with pytest.raises(PydanticAIBoundaryCodecError):
        build_output_type_contract(factory_output)


def test_recursive_models_use_refs_without_weakening_marker_rejection():
    class RecursiveNode(BaseModel):
        value: int
        child: RecursiveNode | None = None

    RecursiveNode.model_rebuild()
    graph = json.loads(build_output_type_contract(RecursiveNode).canonical_bytes)[
        "type_graph"
    ]
    child_union = _graph_field(graph, "child")["annotation"]
    assert {
        item["kind"] for item in child_union["args"]
    } == {"model-ref", "type"}

    class RecursiveDeferred(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        child: RecursiveDeferred | None = None
        deferred: DeferredToolResults

    RecursiveDeferred.model_rebuild()
    with pytest.raises(PydanticAIBoundaryCodecError):
        build_output_type_contract(RecursiveDeferred)


def test_opaque_and_unbounded_annotation_shapes_fail_closed():
    @dataclass
    class DataclassValue:
        value: int

    class TypedDictValue(TypedDict):
        value: int

    class ArbitraryValue:
        pass

    class OpaqueMetadata:
        pass

    @dataclass
    class ForgedTrustedMetadata:
        ge: int

    ForgedTrustedMetadata.__module__ = "annotated_types"

    annotations = (
        Any,
        DataclassValue,
        TypedDictValue,
        ArbitraryValue,
        tuple[int, ...],
        Annotated[int, OpaqueMetadata()],
        Annotated[int, ForgedTrustedMetadata(0)],
    )
    for index, annotation in enumerate(annotations):
        output = create_model(
            f"ClosedGrammarRejection{index}",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            value=(annotation, ...),
        )
        with pytest.raises(PydanticAIBoundaryCodecError):
            build_output_type_contract(output)

    behavioral_factory = create_model(
        "BehavioralFactoryRejection",
        value=(list[int], Field(default_factory=lambda: [])),
    )
    with pytest.raises(PydanticAIBoundaryCodecError):
        build_output_type_contract(behavioral_factory)


def test_hostile_type_equality_cannot_forge_scalar_origin_or_metadata_identity():
    class ScalarSpoofMeta(type):
        def __eq__(cls, other):
            return other is int

        __hash__ = type.__hash__

    class ScalarSpoof(metaclass=ScalarSpoofMeta):
        @classmethod
        def __get_pydantic_core_schema__(cls, source_type, handler):
            return core_schema.str_schema()

    assert ScalarSpoof in (str, int, float, bool, Decimal)
    scalar_output = create_model("HostileScalarOutput", value=(ScalarSpoof, ...))
    assert scalar_output.model_json_schema()["properties"]["value"]["type"] == "string"
    with pytest.raises(PydanticAIBoundaryCodecError):
        build_output_type_contract(scalar_output)

    first = TypeVar("first")
    second = TypeVar("second")

    class OriginSpoofMeta(type):
        def __eq__(cls, other):
            return other is Union

        __hash__ = type.__hash__

    class OriginSpoof(Generic[first, second], metaclass=OriginSpoofMeta):
        @classmethod
        def __get_pydantic_core_schema__(cls, source_type, handler):
            return core_schema.str_schema()

    hostile_generic = OriginSpoof[int, str]
    assert getattr(hostile_generic, "__origin__") in (Union, types.UnionType)
    origin_output = create_model("HostileOriginOutput", value=(hostile_generic, ...))
    assert origin_output.model_json_schema()["properties"]["value"]["type"] == "string"
    with pytest.raises(PydanticAIBoundaryCodecError):
        build_output_type_contract(origin_output)

    class MetadataSpoofMeta(type):
        def __hash__(cls):
            return hash(Ge)

        def __eq__(cls, other):
            return cls is other or other is Ge

    @dataclass(frozen=True)
    class MetadataSpoof(metaclass=MetadataSpoofMeta):
        ge: int

    assert MetadataSpoof in frozenset((Ge,))
    metadata_output = create_model(
        "HostileMetadataOutput",
        value=(Annotated[int, MetadataSpoof(0)], ...),
    )
    with pytest.raises(PydanticAIBoundaryCodecError):
        build_output_type_contract(metadata_output)

    class ContainerSpoofMeta(type):
        def __eq__(cls, other):
            return other is list

        __hash__ = type.__hash__

    class ContainerSpoof(metaclass=ContainerSpoofMeta):
        def __iter__(self):
            raise AssertionError("an opaque container was traversed")

    container_metadata_output = create_model(
        "HostileContainerMetadataOutput",
        value=(Annotated[int, ContainerSpoof()], ...),
    )
    with pytest.raises(PydanticAIBoundaryCodecError) as caught:
        build_output_type_contract(container_metadata_output)
    assert caught.value.__context__ is None


def test_hostile_model_equality_cannot_hide_deferred_fields_as_recursion_refs():
    class EqualModelMeta(type(BaseModel)):
        enabled = False

        def __eq__(cls, other):
            return cls is other or (
                EqualModelMeta.enabled and isinstance(other, EqualModelMeta)
            )

        def __hash__(cls):
            return 1

    class HiddenDeferred(BaseModel, metaclass=EqualModelMeta):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        deferred: DeferredToolResults

    class RootOutput(BaseModel, metaclass=EqualModelMeta):
        hidden: HiddenDeferred

    EqualModelMeta.enabled = True
    assert HiddenDeferred in {RootOutput}
    with pytest.raises(PydanticAIBoundaryCodecError):
        build_output_type_contract(RootOutput)


def test_type_graph_depth_budget_fails_closed():
    annotation = int
    for _ in range(boundary_codec._TYPE_GRAPH_MAX_DEPTH + 1):
        annotation = list[annotation]
    output = create_model("ExcessivelyDeepOutput", value=(annotation, ...))
    with pytest.raises(PydanticAIBoundaryCodecError):
        build_output_type_contract(output)

    cycle = []
    cycle.append(cycle)
    with pytest.raises(PydanticAIBoundaryCodecError):
        boundary_codec._reject_nested_deferred(cycle)


def test_type_graph_node_budget_fails_closed():
    output = create_model(
        "ExcessivelyWideOutput",
        **{
            f"field_{index}": (int, ...)
            for index in range(boundary_codec._TYPE_GRAPH_MAX_NODES)
        },
    )
    with pytest.raises(PydanticAIBoundaryCodecError):
        build_output_type_contract(output)


def test_public_boundary_errors_discard_lower_exception_context_and_secrets(monkeypatch):
    secret = "Bearer sk-boundary-secret-should-never-escape"
    contract = build_output_type_contract(str)
    model = _production_model()
    messages, parameters, _ = _text_request_parts()

    def explode(*args, **kwargs):
        raise RuntimeError(secret)

    with monkeypatch.context() as scoped:
        scoped.setattr(boundary_codec, "_attempt_build_output_contract_payload", explode)
        with pytest.raises(PydanticAIBoundaryCodecError) as caught:
            BoundOutputContract(
                output_type=str,
                canonical_bytes=contract.canonical_bytes,
                sha256_hex=contract.sha256_hex,
                output_kind=contract.output_kind,
                _construction_token=boundary_codec._OUTPUT_BUILD_TOKEN,
            )
        _assert_sanitized_boundary_error(caught, secret)

    with monkeypatch.context() as scoped:
        scoped.setattr(boundary_codec, "_attempt_build_output_contract_payload", explode)
        with pytest.raises(PydanticAIBoundaryCodecError) as caught:
            validate_output_type_contract(contract)
        _assert_sanitized_boundary_error(caught, secret)

    with monkeypatch.context() as scoped:
        scoped.setattr(boundary_codec, "project_model_request", explode)
        with pytest.raises(PydanticAIBoundaryCodecError) as caught:
            encode_model_request(
                model,
                messages,
                None,
                parameters,
                output_contract=contract,
            )
        _assert_sanitized_boundary_error(caught, secret)


def test_history_media_invalid_literals_and_hostile_sequences_fail_closed():
    model = _production_model()
    messages, parameters, contract = _text_request_parts()
    with pytest.raises(PydanticAIBoundaryCodecError):
        encode_model_request(
            model,
            [*messages, ModelResponse(parts=[])],
            None,
            parameters,
            output_contract=contract,
        )

    media_message = ModelRequest(parts=[UserPromptPart([b"binary-like-content"])])
    with pytest.raises(PydanticAIBoundaryCodecError):
        encode_model_request(
            model,
            [media_message],
            None,
            parameters,
            output_contract=contract,
        )

    bad_message_kind = ModelRequest(parts=[UserPromptPart("x")], kind="bad")
    bad_part_kind = ModelRequest(parts=[UserPromptPart("x", part_kind="bad")])
    for message in (bad_message_kind, bad_part_kind):
        with pytest.raises(PydanticAIBoundaryCodecError):
            encode_model_request(
                model,
                [message],
                None,
                parameters,
                output_contract=contract,
            )

    class HostileParts(list):
        pass

    hostile = ModelRequest(parts=HostileParts([UserPromptPart("x")]))
    with pytest.raises(PydanticAIBoundaryCodecError):
        encode_model_request(
            model,
            [hostile],
            None,
            parameters,
            output_contract=contract,
        )


def test_normal_and_falsey_forbidden_tool_lists_fail_closed():
    model = _production_model()
    messages, parameters, contract = _text_request_parts()
    tool = ToolDefinition(
        name="arbitrary_tool",
        description="outside the M1e surface",
        parameters_json_schema={"type": "object", "properties": {}},
    )
    normal = replace(parameters, function_tools=[tool])

    class FalseyList(list):
        def __bool__(self):
            return False

    hostile = replace(parameters, function_tools=FalseyList([tool]))
    for forbidden in (normal, hostile):
        with pytest.raises(PydanticAIBoundaryCodecError):
            encode_model_request(
                model,
                messages,
                None,
                forbidden,
                output_contract=contract,
            )


def test_mutated_output_tool_and_parameter_field_types_fail_closed():
    class Output(BaseModel):
        value: int

    capture, contract, _ = _capture_run(Output, "typed")
    messages, settings, parameters = capture.raw_requests[-1]
    changed_tool = replace(parameters.output_tools[0], defer_loading=True)
    mismatched_schema_tool = replace(
        parameters.output_tools[0],
        parameters_json_schema={"type": "object", "properties": {}},
    )
    attacks = (
        replace(parameters, output_tools=[changed_tool]),
        replace(parameters, output_tools=[mismatched_schema_tool]),
        replace(parameters, output_tools=tuple(parameters.output_tools)),
        replace(parameters, allow_image_output=0),
    )
    for attack in attacks:
        with pytest.raises(PydanticAIBoundaryCodecError):
            encode_model_request(
                capture.wrapped,
                messages,
                settings,
                attack,
                output_contract=contract,
            )


def test_settings_metadata_and_projection_are_strict_and_detached():
    model = _production_model()
    metadata = {"tag": ["before"]}
    message = ModelRequest(parts=[UserPromptPart("prompt")], metadata=metadata)
    contract = build_output_type_contract(str)
    projected = project_model_request(
        model,
        [message],
        {"temperature": 0.2},
        ModelRequestParameters(),
        output_contract=contract,
    )
    metadata["tag"].append("after")
    assert projected["messages"][0]["metadata"] == {"tag": ["before"]}

    with pytest.raises(PydanticAIBoundaryCodecError):
        project_model_request(
            model,
            [message],
            {"extra_body": {"tuple_is_not_json": (1, 2)}},
            ModelRequestParameters(),
            output_contract=contract,
        )


def test_response_round_trip_preserves_typed_value_and_dependency_versions():
    response = ModelResponse(
        parts=[TextPart(content="fixture response")],
        model_name="fixture-model",
        provider_name="fixture-provider",
        usage=RequestUsage(input_tokens=1, output_tokens=2),
    )
    encoded = encode_model_response(response)
    decoded = decode_model_response(encoded)
    assert encode_model_response(decoded) == encoded
    assert decoded.parts == [TextPart(content="fixture response")]
    value = json.loads(encoded)
    assert value["pydantic_ai_version"] == "1.107.1"
    assert value["pydantic_version"] == "2.13.4"
    assert value["pydantic_core_version"] == "2.46.4"


@pytest.mark.parametrize("attack", [True, 1.0])
def test_response_decoder_rejects_bool_float_format_versions(attack):
    value = json.loads(encode_model_response(ModelResponse(parts=[TextPart(content="x")])))
    value["format_version"] = attack
    with pytest.raises(PydanticAIBoundaryCodecError):
        decode_model_response(canonical_json_bytes(value))


@pytest.mark.parametrize("attack", [True, 1.0])
def test_response_decoder_rejects_bool_float_usage_coercion(attack):
    response = ModelResponse(
        parts=[TextPart(content="safe")],
        usage=RequestUsage(input_tokens=1),
    )
    value = json.loads(encode_model_response(response))
    value["response"]["usage"]["input_tokens"] = attack
    with pytest.raises(PydanticAIBoundaryCodecError):
        decode_model_response(canonical_json_bytes(value))


def test_response_decoder_rejects_noncanonical_drift_extras_and_safe_errors():
    encoded = encode_model_response(ModelResponse(parts=[TextPart(content="safe")]))
    value = json.loads(encoded)
    with pytest.raises(PydanticAIBoundaryCodecError):
        decode_model_response(json.dumps(value, indent=2).encode())

    for key in ("pydantic_ai_version", "pydantic_version", "pydantic_core_version"):
        changed = json.loads(encoded)
        changed[key] = "0.0.0"
        with pytest.raises(PydanticAIBoundaryCodecError):
            decode_model_response(canonical_json_bytes(changed))

    value["unexpected"] = "secret-looking-value"
    with pytest.raises(PydanticAIBoundaryCodecError) as caught:
        decode_model_response(canonical_json_bytes(value))
    assert "secret-looking-value" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_dependency_version_drift_fails_closed(monkeypatch):
    monkeypatch.setattr(boundary_codec.pydantic, "__version__", "0.0.0")
    with pytest.raises(PydanticAIBoundaryCodecError):
        build_output_type_contract(str)


def test_harness_disables_retries_and_validates_contract_before_execution(monkeypatch):
    created = []

    class FakeAgent:
        def __init__(self, model, **kwargs):
            self.output_type = kwargs["output_type"]
            created.append((model, kwargs))

        def output_json_schema(self):
            return TypeAdapter(self.output_type).json_schema()

        def run_sync(self, instruction, **kwargs):
            return SimpleNamespace(output="offline", usage=lambda: SimpleNamespace())

    import pydantic_ai

    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)
    harness = PydanticAIHarness()
    harness.bind(
        HarnessContext(objectives=(), search_space_desc="offline"),
        LLMConfig(model="fixture", retries=9),
    )
    assert harness._output(str, "offline instruction") == "offline"
    assert harness.last_output_type_contract == build_output_type_contract(str)
    assert created == [
        (
            "fixture",
            {"output_type": str, "retries": 0},
        )
    ]


def test_harness_discards_agent_task_group_context_and_secret():
    secret = "Bearer sk-agent-task-group-secret"

    class ExplodingCaptureModel(WrapperModel):
        def __init__(self, wrapped):
            super().__init__(wrapped)
            self.invocations = 0

        async def request(self, messages, model_settings, model_request_parameters):
            self.invocations += 1
            raise RuntimeError(secret)

    model = ExplodingCaptureModel(_fixture_model())
    harness = PydanticAIHarness()
    harness.bind(
        HarnessContext(objectives=(), search_space_desc="offline"),
        LLMConfig(model=model),
    )
    with pytest.raises(HarnessOutputError) as caught:
        harness._output(str, "offline instruction")
    error = caught.value
    assert type(error) is HarnessOutputError
    assert str(error) == "pydantic_ai: agent execution failed"
    assert secret not in repr(error)
    assert "ExceptionGroup" not in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert model.invocations == 1


def test_harness_rejects_deferred_spec_before_agent_creation(monkeypatch):
    created = []

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))

    import pydantic_ai

    monkeypatch.setattr(pydantic_ai, "Agent", FakeAgent)
    harness = PydanticAIHarness()
    harness.bind(
        HarnessContext(objectives=(), search_space_desc="offline"),
        LLMConfig(model="fixture"),
    )
    with pytest.raises(PydanticAIBoundaryCodecError):
        harness._output([str, DeferredToolRequests], "offline")
    assert created == []
