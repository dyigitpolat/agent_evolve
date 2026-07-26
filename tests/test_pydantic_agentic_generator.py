from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from agent_evolve.domain.ids import LLMCallId
from agent_evolve.integrations.pydantic_ai import agentic_generator as adapter_module
from agent_evolve.integrations.pydantic_ai import (
    REFLECTION_OUTPUT_CONTRACT_NOTE,
    REFLECTION_OUTPUT_CONTRACT_NOTE_SHA256,
    REFLECTION_WIRE_CONTRACT_REVISION,
    render_reflection_prompt,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
    CANDIDATE_PROPOSAL_TOOL_NAME,
    EXACT_PARENT_CROSSOVER_TOOL_NAME,
    MAX_AFFECTED_PATHS,
    MAX_INTENDED_CHANGES,
    MAX_REFLECTION_TEXT_CHARS,
    PydanticAIAgenticGenerator,
    REFLECTION_TOOL_NAME,
)
from agent_evolve.ports.agentic_generator import (
    AgenticGenerator,
    CANDIDATE_COMPONENT_PATH_CONTRACT,
    ConflictResolutionDraft,
    ExactParentCrossoverDraft,
    ExactParentCrossoverOutputContract,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionEvidenceCatalog,
    ReflectionEvidenceCatalogEntry,
    ReflectionGenerationRequest,
    ReflectionInsightContract,
    SourceAttribution,
    TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT,
    VariationGenerationRequest,
    validate_reflection_evidence_catalog_result,
    validate_reflection_insight_draft,
)
from agent_evolve.ports.structured_generator import (
    GenerationFailureKind,
    MAX_PROMPT_UTF8_BYTES,
    StructuredGenerationError,
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)


_CONTRAST_A = "1" * 64
_CONTRAST_B = "2" * 64


class _Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)

    width: int
    public_name: str = Field(alias="wire_name")
    note: str | None = None

    @computed_field
    @property
    def doubled_width(self) -> int:
        return self.width * 2


class _FakeRunner:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.requests: list[StructuredGenerationRequest[Any]] = []

    async def __call__(self, request: StructuredGenerationRequest[Any]):
        self.requests.append(request)
        return self.handler(request)


def _response(
    value: Any,
    *,
    requested_model: str = "requested/model",
) -> StructuredGenerationResponse[Any]:
    return StructuredGenerationResponse(
        value=value,
        requested_model=requested_model,
        resolved_model="resolved/model",
        resolved_provider="downstream-provider",
        provider_response_id="response-123",
        finish_reason="stop",
        input_tokens=101,
        output_tokens=23,
        reasoning_tokens=7,
        cache_read_tokens=11,
        cache_write_tokens=13,
        cost_usd=Decimal("0.00125"),
        latency_ns=987_654,
    )


def _variation_request(**changes: Any) -> VariationGenerationRequest:
    values: dict[str, Any] = {
        "call_id": LLMCallId("call_agentic_propose_0001"),
        "operation": "mutate_candidate",
        "prompt": "Propose one complete candidate.",
        "candidate_model": _Candidate,
        "max_output_tokens": 777,
        "temperature": 0.25,
    }
    values.update(changes)
    return VariationGenerationRequest(**values)


def _reflection_request(**changes: Any) -> ReflectionGenerationRequest:
    values: dict[str, Any] = {
        "call_id": LLMCallId("call_agentic_reflect_0001"),
        "operation": "reflect_evaluation",
        "prompt": "Extract reusable insights.",
        "max_insights": 3,
        "max_output_tokens": 888,
        "temperature": 0.0,
        "available_contrast_ids": (_CONTRAST_A, _CONTRAST_B),
    }
    values.update(changes)
    return ReflectionGenerationRequest(**values)


def test_propose_builds_strict_dynamic_schema_and_maps_every_field() -> None:
    def handle(request: StructuredGenerationRequest[Any]):
        assert request.output_type.model_config["strict"] is True
        assert request.output_type.model_config["extra"] == "forbid"
        proposal = request.output_type.model_validate(
            {
                "configuration": {
                    "width": 9,
                    "wire_name": "candidate-nine",
                    "note": None,
                },
                "design_rationale": "  Wider search with guarded reuse.  ",
                "intended_changes": ["  increase width  ", "retain the baseline"],
                "source_attribution": [
                    {"path": " $.width ", "source": "mutation"},
                    {"path": "$.public_name", "source": "ancestor"},
                ],
                "claimed_insight_ids": [" insight_1 ", "insight_2"],
                "claimed_preservation_obligation_ids": [" obligation_4 "],
                "conflict_resolutions": [
                    {
                        "relation_id": " conflict_3 ",
                        "choice": "synthesize",
                        "explanation": "  combine safe portions  ",
                    }
                ],
            },
            strict=True,
        )
        return AttemptedStructuredGenerationResponse(
            response=_response(proposal),
            attempt_count=4,
        )

    runner = _FakeRunner(handle)
    generator = PydanticAIAgenticGenerator(runner)

    result = asyncio.run(generator.propose(_variation_request()))

    assert len(runner.requests) == 1
    low_request = runner.requests[0]
    assert type(low_request) is StructuredGenerationRequest
    assert low_request.call_id == LLMCallId("call_agentic_propose_0001")
    assert low_request.operation == "mutate_candidate"
    assert low_request.prompt == "Propose one complete candidate."
    assert low_request.output_tool_name == CANDIDATE_PROPOSAL_TOOL_NAME
    assert low_request.max_output_tokens == 777
    assert low_request.temperature == 0.25
    assert (
        low_request.output_type.model_fields["configuration"].annotation is _Candidate
    )
    wire_schema = low_request.output_type.model_json_schema()
    encoded_wire_schema = json.dumps(wire_schema, sort_keys=True)
    assert "$defs" not in wire_schema
    assert "_Candidate" not in encoded_wire_schema
    assert set(wire_schema["properties"]) == set(low_request.output_type.model_fields)
    assert wire_schema["properties"]["configuration"] == {
        "type": "object",
        "additionalProperties": True,
        "description": (
            "The complete proposed candidate configuration. This field's value, "
            "rather than the enclosing proposal object, is JSON-path root '$' "
            "for intended_changes and source_attribution."
        ),
    }
    assert wire_schema["properties"]["intended_changes"]["items"] == {
        "type": "string",
        "description": CANDIDATE_COMPONENT_PATH_CONTRACT,
    }
    assert (
        CANDIDATE_COMPONENT_PATH_CONTRACT
        in wire_schema["properties"]["intended_changes"]["description"]
    )
    assert (
        CANDIDATE_COMPONENT_PATH_CONTRACT
        in wire_schema["properties"]["source_attribution"]["description"]
    )
    assert wire_schema["properties"]["source_attribution"]["items"] == {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": CANDIDATE_COMPONENT_PATH_CONTRACT,
            },
            "source": {
                "type": "string",
                "enum": [
                    "ancestor",
                    "left",
                    "right",
                    "synthesized",
                    "mutation",
                ],
            },
        },
        "required": ["path", "source"],
        "additionalProperties": False,
    }
    assert not any(
        constraint in encoded_wire_schema
        for constraint in ("maxLength", "minLength", "maxItems", "pattern")
    )

    # Python-mode field names are retained; aliases and computed fields are not
    # allowed to leak into the framework-free candidate representation.
    assert type(result.draft.configuration) is dict
    assert result.draft.configuration == {
        "width": 9,
        "public_name": "candidate-nine",
        "note": None,
    }
    assert result.draft.design_rationale == "Wider search with guarded reuse."
    assert result.draft.intended_changes == (
        "increase width",
        "retain the baseline",
    )
    assert result.draft.source_attribution == (
        SourceAttribution(path="$.width", source="mutation"),
        SourceAttribution(path="$.public_name", source="ancestor"),
    )
    assert result.draft.claimed_insight_ids == ("insight_1", "insight_2")
    assert result.draft.claimed_preservation_obligation_ids == ("obligation_4",)
    assert result.draft.conflict_resolutions == (
        ConflictResolutionDraft(
            relation_id="conflict_3",
            choice="synthesize",
            explanation="combine safe portions",
        ),
    )

    telemetry = result.telemetry
    assert telemetry.requested_model == "requested/model"
    assert telemetry.resolved_model == "resolved/model"
    assert telemetry.resolved_provider == "downstream-provider"
    assert telemetry.provider_response_id == "response-123"
    assert telemetry.finish_reason == "stop"
    assert telemetry.input_tokens == 101
    assert telemetry.output_tokens == 23
    assert telemetry.reasoning_tokens == 7
    assert telemetry.cache_read_tokens == 11
    assert telemetry.cache_write_tokens == 13
    assert telemetry.cost_usd == Decimal("0.00125")
    assert telemetry.latency_ns == 987_654
    assert telemetry.attempt_count == 4


def test_direct_response_defaults_to_one_attempt_and_optional_lists_to_empty() -> None:
    def handle(request: StructuredGenerationRequest[Any]):
        proposal = request.output_type(
            configuration={
                "width": 2,
                "wire_name": "minimal",
                "note": None,
            },
            design_rationale="Minimal valid proposal",
        )
        return _response(proposal, requested_model="one/attempt")

    result = asyncio.run(
        PydanticAIAgenticGenerator(_FakeRunner(handle)).propose(_variation_request())
    )

    assert result.telemetry.requested_model == "one/attempt"
    assert result.telemetry.attempt_count == 1
    assert result.draft.intended_changes == ()
    assert result.draft.source_attribution == ()
    assert result.draft.claimed_insight_ids == ()
    assert result.draft.claimed_preservation_obligation_ids == ()
    assert result.draft.conflict_resolutions == ()


@pytest.mark.parametrize(
    ("operation", "operator_fields"),
    [
        ("typed_mutation", set()),
        ("two_parent_crossover", set()),
        ("three_way_recombination", {"conflict_resolutions"}),
        ("repair", set()),
    ],
)
def test_exact_operator_proposal_schemas_expose_only_relevant_evidence_fields(
    operation: str,
    operator_fields: set[str],
) -> None:
    common_fields = {
        "configuration",
        "design_rationale",
        "intended_changes",
        "source_attribution",
        "claimed_insight_ids",
    }
    output_type = adapter_module._candidate_proposal_type(_Candidate, operation)

    assert set(output_type.model_fields) == common_fields | operator_fields
    assert set(output_type.model_json_schema()["properties"]) == (
        common_fields | operator_fields
    )
    if operation == "three_way_recombination":
        conflict_item = output_type.model_json_schema()["properties"][
            "conflict_resolutions"
        ]["items"]
        assert conflict_item["required"] == [
            "relation_id",
            "choice",
            "explanation",
        ]
        assert conflict_item["properties"]["choice"]["enum"] == [
            "choose_left",
            "choose_right",
            "synthesize",
            "drop_both",
        ]
        assert conflict_item["additionalProperties"] is False

    valid = {
        "configuration": {"width": 3, "wire_name": operation, "note": None},
        "design_rationale": "Operator-specific minimal proposal",
    }
    proposal = output_type.model_validate(valid, strict=True)
    assert proposal.configuration.width == 3

    forbidden_fields = {
        "claimed_preservation_obligation_ids",
        "conflict_resolutions",
    } - operator_fields
    for forbidden_field in forbidden_fields:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            output_type.model_validate(
                {**valid, forbidden_field: []},
                strict=True,
            )


def test_two_parent_wire_schema_requires_discriminating_parent_evidence() -> None:
    crossover_schema = adapter_module._candidate_proposal_type(
        _Candidate,
        "two_parent_crossover",
    ).model_json_schema()
    mutation_schema = adapter_module._candidate_proposal_type(
        _Candidate,
        "typed_mutation",
    ).model_json_schema()

    intended = crossover_schema["properties"]["intended_changes"]
    attribution = crossover_schema["properties"]["source_attribution"]
    assert TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT in intended["description"]
    assert TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT in intended["items"]["description"]
    assert TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT in attribution["description"]
    assert (
        TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT
        in attribution["items"]["properties"]["path"]["description"]
    )
    assert (
        "differs from the other parent at the same path" in (attribution["description"])
    )
    assert (
        "omit the path from both source_attribution and intended_changes"
        in (intended["description"])
    )
    assert (
        "at least one discriminating left contribution" in (attribution["description"])
    )
    assert "executable inheritance instruction" in attribution["description"]
    assert "copies the exact immutable parent subtree" in attribution["description"]
    assert "one binary64 ULP" in attribution["description"]
    assert "Synthesized values remain model-authored" in attribution["description"]
    assert "smallest retained containing object or array" in attribution["description"]
    assert (
        "Any unclaimed value shared by both parents must remain present"
        in (attribution["description"])
    )
    assert attribution["items"]["properties"]["source"]["enum"] == [
        "left",
        "right",
        "synthesized",
    ]
    assert mutation_schema["properties"]["source_attribution"]["items"]["properties"][
        "source"
    ]["enum"] == [
        "ancestor",
        "left",
        "right",
        "synthesized",
        "mutation",
    ]
    assert (
        TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT
        not in mutation_schema["properties"]["source_attribution"]["description"]
    )

    crossover_type = adapter_module._candidate_proposal_type(
        _Candidate,
        "two_parent_crossover",
    )
    valid = {
        "configuration": {"width": 3, "wire_name": "child", "note": None},
        "design_rationale": "Use both parents.",
    }
    for rejected_source in ("ancestor", "mutation"):
        with pytest.raises(ValidationError, match="Input should be"):
            crossover_type.model_validate(
                {
                    **valid,
                    "source_attribution": [
                        {"path": "$.width", "source": rejected_source}
                    ],
                },
                strict=True,
            )


def test_exact_parent_crossover_wire_is_finite_minimal_and_bounded() -> None:
    contract = ExactParentCrossoverOutputContract(
        contract_identity_sha256="3" * 64,
        locus_ids=("locus_a", "locus_b", "locus_c", "locus_d"),
    )
    output_type = adapter_module._exact_parent_crossover_proposal_type(
        "two_parent_crossover",
        contract,
    )

    schema = output_type.model_json_schema()
    assert set(output_type.model_fields) == {
        "import_locus_ids",
        "claimed_insight_ids",
    }
    assert set(schema["properties"]) == {
        "import_locus_ids",
        "claimed_insight_ids",
    }
    imported = schema["properties"]["import_locus_ids"]
    assert imported["items"] == {
        "type": "string",
        "enum": ["locus_a", "locus_b", "locus_c", "locus_d"],
    }
    assert imported["uniqueItems"] is True
    assert imported["minItems"] == 1
    assert imported["maxItems"] == len(contract.locus_ids) - 1
    claimed = schema["properties"]["claimed_insight_ids"]
    assert claimed["uniqueItems"] is True
    assert claimed["maxItems"] == 0
    assert schema["required"] == ["import_locus_ids"]
    assert schema["additionalProperties"] is False
    assert not {
        "configuration",
        "design_rationale",
        "intended_changes",
        "source_attribution",
    }.intersection(schema["properties"])
    assert (
        len(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        < 2_000
    )
    wire_schema = OpenAIJsonSchemaTransformer(
        json.loads(json.dumps(schema)),
        strict=False,
    ).walk()
    assert wire_schema == schema


@pytest.mark.parametrize(
    "payload",
    [
        {"import_locus_ids": []},
        {"import_locus_ids": ["locus_a", "locus_b", "locus_c", "locus_d"]},
        {"import_locus_ids": ["locus_a", "locus_a"]},
        {"import_locus_ids": ["foreign_locus"]},
        {"import_locus_ids": ["locus_a"], "configuration": {}},
        {"import_locus_ids": ["locus_a"], "design_rationale": "verbose"},
        {"import_locus_ids": ["locus_a"], "intended_changes": []},
        {"import_locus_ids": ["locus_a"], "source_attribution": []},
        {"import_locus_ids": ["locus_a"], "claimed_insight_ids": ["foreign"]},
    ],
)
def test_exact_parent_crossover_wire_rejects_invalid_or_expansive_payloads(
    payload: dict[str, Any],
) -> None:
    output_type = adapter_module._exact_parent_crossover_proposal_type(
        "two_parent_crossover",
        ExactParentCrossoverOutputContract(
            contract_identity_sha256="4" * 64,
            locus_ids=("locus_a", "locus_b", "locus_c", "locus_d"),
        ),
    )

    with pytest.raises(ValidationError):
        output_type.model_validate(payload, strict=True)


def test_exact_parent_crossover_wire_excludes_known_child_set_without_enumeration() -> (
    None
):
    contract = ExactParentCrossoverOutputContract(
        contract_identity_sha256="6" * 64,
        locus_ids=("locus_a", "locus_b", "locus_c", "locus_d"),
        forbidden_import_locus_sets=(("locus_a", "locus_c"),),
    )
    output_type = adapter_module._exact_parent_crossover_proposal_type(
        "two_parent_crossover",
        contract,
    )

    schema = output_type.model_json_schema()
    imported = schema["properties"]["import_locus_ids"]
    assert imported["allOf"] == [
        {
            "not": {
                "allOf": [
                    {"minItems": 2},
                    {"maxItems": 2},
                    {"contains": {"const": "locus_a"}},
                    {"contains": {"const": "locus_c"}},
                ]
            }
        }
    ]
    assert (
        OpenAIJsonSchemaTransformer(
            json.loads(json.dumps(schema)),
            strict=False,
        ).walk()
        == schema
    )
    for ordering in (
        ["locus_a", "locus_c"],
        ["locus_c", "locus_a"],
    ):
        with pytest.raises(ValidationError, match="forbidden known-child"):
            output_type.model_validate(
                {"import_locus_ids": ordering},
                strict=True,
            )

    accepted = output_type.model_validate(
        {"import_locus_ids": ["locus_a", "locus_b"]},
        strict=True,
    )
    assert accepted.import_locus_ids == ["locus_a", "locus_b"]


def test_exact_parent_crossover_adapter_sorts_and_binds_the_compact_plan() -> None:
    contract = ExactParentCrossoverOutputContract(
        contract_identity_sha256="5" * 64,
        locus_ids=("locus_a", "locus_b", "locus_c", "locus_d"),
        claimable_insight_ids=("insight_alpha", "insight_beta"),
    )

    def handle(request: StructuredGenerationRequest[Any]):
        assert request.output_tool_name == EXACT_PARENT_CROSSOVER_TOOL_NAME
        assert request.operation == "two_parent_crossover"
        schema_bytes = json.dumps(
            request.output_type.model_json_schema(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert len(schema_bytes) < 2_000
        claimed = request.output_type.model_json_schema()["properties"][
            "claimed_insight_ids"
        ]
        assert claimed["items"]["enum"] == ["insight_alpha", "insight_beta"]
        assert claimed["maxItems"] == 2
        proposal = request.output_type.model_validate(
            {
                "import_locus_ids": ["locus_d", "locus_b"],
                "claimed_insight_ids": ["insight_beta", "insight_alpha"],
            },
            strict=True,
        )
        return _response(proposal)

    result = asyncio.run(
        PydanticAIAgenticGenerator(_FakeRunner(handle)).propose(
            _variation_request(
                operation="two_parent_crossover",
                exact_parent_crossover_contract=contract,
            )
        )
    )

    assert type(result.draft) is ExactParentCrossoverDraft
    assert result.draft == ExactParentCrossoverDraft(
        contract_identity_sha256=contract.contract_identity_sha256,
        import_locus_ids=("locus_b", "locus_d"),
        claimed_insight_ids=("insight_beta", "insight_alpha"),
    )


def test_three_way_schema_maps_conflicts_without_requesting_opaque_preservation_ids() -> (
    None
):
    def handle(request: StructuredGenerationRequest[Any]):
        assert request.operation == "three_way_recombination"
        assert (
            "claimed_preservation_obligation_ids"
            not in request.output_type.model_fields
        )
        proposal = request.output_type.model_validate(
            {
                "configuration": {
                    "width": 5,
                    "wire_name": "recombined",
                    "note": None,
                },
                "design_rationale": "Resolve the one genuine conflict.",
                "conflict_resolutions": [
                    {
                        "relation_id": "relation_1",
                        "choice": "choose_left",
                        "explanation": "The left value has stronger evidence.",
                    }
                ],
            },
            strict=True,
        )
        return _response(proposal)

    result = asyncio.run(
        PydanticAIAgenticGenerator(_FakeRunner(handle)).propose(
            _variation_request(operation="three_way_recombination")
        )
    )

    assert result.draft.claimed_preservation_obligation_ids == ()
    assert result.draft.conflict_resolutions == (
        ConflictResolutionDraft(
            relation_id="relation_1",
            choice="choose_left",
            explanation="The left value has stronger evidence.",
        ),
    )


def test_reflect_builds_bounded_schema_and_maps_insights_and_telemetry() -> None:
    def handle(request: StructuredGenerationRequest[Any]):
        reflection = request.output_type.model_validate(
            {
                "insights": [
                    {
                        "claim": "  Cache successful prefixes. ",
                        "trigger": " repeated prefix evaluation ",
                        "mechanism": " avoid duplicate work ",
                        "affected_paths": [" $.cache.enabled ", "$.cache.capacity"],
                        "evidence_summary": " three matching evaluations ",
                        "evidence_contrast_ids": [_CONTRAST_A],
                        "confidence": 0.875,
                    },
                    {
                        "claim": "Keep the fallback",
                        "trigger": "sparse evidence",
                        "mechanism": "limits downside",
                        "affected_paths": ["$.fallback"],
                        "evidence_summary": "one adverse case",
                        "evidence_contrast_ids": [_CONTRAST_A, _CONTRAST_B],
                        "confidence": 1,
                    },
                ]
            },
            strict=True,
        )
        return AttemptedStructuredGenerationResponse(_response(reflection), 6)

    runner = _FakeRunner(handle)
    result = asyncio.run(
        PydanticAIAgenticGenerator(runner).reflect(_reflection_request(max_insights=2))
    )

    low_request = runner.requests[0]
    assert low_request.call_id == LLMCallId("call_agentic_reflect_0001")
    assert low_request.operation == "reflect_evaluation"
    assert low_request.prompt == (
        "Extract reusable insights.\n\n"
        f"{adapter_module.REFLECTION_OUTPUT_CONTRACT_NOTE}"
    )
    assert "Every path must be rooted at '$'" in low_request.prompt
    assert adapter_module.REFLECTION_WIRE_CONTRACT_REVISION in low_request.prompt
    assert low_request.output_tool_name == REFLECTION_TOOL_NAME
    assert low_request.max_output_tokens == 888
    assert low_request.temperature == 0.0
    assert low_request.output_type.model_config["strict"] is True
    assert low_request.output_type.model_config["extra"] == "forbid"
    reflection_wire_schema = low_request.output_type.model_json_schema()
    assert "$defs" not in reflection_wire_schema
    assert reflection_wire_schema == {
        "type": "object",
        "properties": {
            "insights": {
                "type": "array",
                "minItems": 0,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_REFLECTION_TEXT_CHARS,
                        },
                        "trigger": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_REFLECTION_TEXT_CHARS,
                        },
                        "mechanism": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_REFLECTION_TEXT_CHARS,
                        },
                        "affected_paths": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": MAX_AFFECTED_PATHS,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": adapter_module.MAX_PATH_CHARS,
                                "pattern": adapter_module._JSON_PATH_PATTERN,
                                "description": (
                                    "A JSON-style path rooted at '$', such as "
                                    "'$.field' or '$[0]'."
                                ),
                            },
                        },
                        "evidence_summary": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_REFLECTION_TEXT_CHARS,
                        },
                        "evidence_contrast_ids": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "pattern": "^[0-9a-f]{64}$",
                                "enum": [_CONTRAST_A, _CONTRAST_B],
                            },
                            "uniqueItems": True,
                            "minItems": 1,
                            "maxItems": 2,
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": [
                        "claim",
                        "trigger",
                        "mechanism",
                        "affected_paths",
                        "evidence_summary",
                        "evidence_contrast_ids",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["insights"],
        "additionalProperties": False,
    }
    assert result.insights == (
        InsightDraft(
            claim="Cache successful prefixes.",
            trigger="repeated prefix evaluation",
            mechanism="avoid duplicate work",
            affected_paths=("$.cache.capacity", "$.cache.enabled"),
            evidence_summary="three matching evaluations",
            confidence=0.875,
            evidence_contrast_ids=(_CONTRAST_A,),
        ),
        InsightDraft(
            claim="Keep the fallback",
            trigger="sparse evidence",
            mechanism="limits downside",
            affected_paths=("$.fallback",),
            evidence_summary="one adverse case",
            confidence=1.0,
            evidence_contrast_ids=(_CONTRAST_A, _CONTRAST_B),
        ),
    )
    assert result.telemetry.attempt_count == 6
    assert result.telemetry.cost_usd == Decimal("0.00125")


def test_render_reflection_prompt_has_exact_utf8_boundaries_and_public_lineage() -> (
    None
):
    assert REFLECTION_WIRE_CONTRACT_REVISION == (
        "reflection_wire_jsonpath_contract_v3_provider_grammar"
    )
    assert adapter_module._JSON_PATH_PATTERN == r"^\$([.\[].*)?$"
    expected_note = (
        "REFLECTION OUTPUT CONTRACT\n"
        "Revision: reflection_wire_jsonpath_contract_v3_provider_grammar. "
        "For every insight, affected_paths must contain 1 to 256 JSON-style "
        "paths. Every path must be rooted at '$' (examples: '$', '$.field', "
        "or '$[0]')."
    )
    assert REFLECTION_OUTPUT_CONTRACT_NOTE == expected_note
    assert REFLECTION_OUTPUT_CONTRACT_NOTE_SHA256 == (
        "03b92db7cdb9b7c1f92a2508616047c450355f31eb2ff5c660063e1945b48138"
    )
    assert (
        REFLECTION_OUTPUT_CONTRACT_NOTE_SHA256
        == hashlib.sha256(
            REFLECTION_OUTPUT_CONTRACT_NOTE.encode("utf-8", errors="strict")
        ).hexdigest()
    )
    suffix = f"\n\n{REFLECTION_OUTPUT_CONTRACT_NOTE}"
    suffix_bytes = len(suffix.encode("utf-8", errors="strict"))
    available_bytes = MAX_PROMPT_UTF8_BYTES - suffix_bytes

    exact_ascii = "x" * available_bytes
    rendered_ascii = render_reflection_prompt(exact_ascii)
    assert rendered_ascii == f"{exact_ascii}{suffix}"
    assert len(rendered_ascii.encode("utf-8", errors="strict")) == (
        MAX_PROMPT_UTF8_BYTES
    )

    exact_utf8 = "é" * (available_bytes // 2)
    if available_bytes % 2:
        exact_utf8 += "x"
    rendered_utf8 = render_reflection_prompt(exact_utf8)
    assert rendered_utf8 == f"{exact_utf8}{suffix}"
    assert len(rendered_utf8.encode("utf-8", errors="strict")) == (
        MAX_PROMPT_UTF8_BYTES
    )

    note_would_overflow = f"{exact_ascii}x"
    assert len(note_would_overflow.encode("utf-8", errors="strict")) < (
        MAX_PROMPT_UTF8_BYTES
    )
    assert render_reflection_prompt(note_would_overflow) == note_would_overflow

    captured: list[StructuredGenerationRequest[Any]] = []

    def handle(request: StructuredGenerationRequest[Any]):
        captured.append(request)
        return _response(request.output_type())

    asyncio.run(
        PydanticAIAgenticGenerator(_FakeRunner(handle)).reflect(
            _reflection_request(prompt=note_would_overflow)
        )
    )
    assert captured[0].prompt == note_would_overflow
    path_schema = captured[0].output_type.model_json_schema()["properties"]["insights"][
        "items"
    ]["properties"]["affected_paths"]
    assert path_schema["minItems"] == 1
    assert path_schema["maxItems"] == MAX_AFFECTED_PATHS
    assert path_schema["items"]["pattern"] == adapter_module._JSON_PATH_PATTERN

    with pytest.raises(TypeError, match="exact string"):
        render_reflection_prompt(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        render_reflection_prompt(" \t ")
    with pytest.raises(ValueError, match="MAX_PROMPT_UTF8_BYTES"):
        render_reflection_prompt("x" * (MAX_PROMPT_UTF8_BYTES + 1))

    canonical_render = render_reflection_prompt("Extract reusable insights.")
    assert canonical_render == (f"Extract reusable insights.\n\n{expected_note}")
    assert hashlib.sha256(
        canonical_render.encode("utf-8", errors="strict")
    ).hexdigest() == (
        "7913fc2a87f6198347f63f57bc9520d79a6614900c29bb49bdc62436dc833071"
    )


def test_reflection_jsonpath_v3_is_provider_safe_and_runtime_equivalent() -> None:
    old_prefix_pattern = r"^\$(?:$|[.\[])"
    provider_safe_pattern = adapter_module._JSON_PATH_PATTERN
    assert provider_safe_pattern == r"^\$([.\[].*)?$"
    assert "|" not in provider_safe_pattern
    assert "(?:" not in provider_safe_pattern

    # The v3 whole-string form preserves the v2 admission decision for every
    # representative single-line value while avoiding the provider's rejected
    # alternation production.
    equivalent_values = (
        "$",
        "$.",
        "$[",
        "$.field",
        "$[0]",
        "$..nested",
        "$[token].field",
        "",
        "field",
        ".field",
        "[0]",
        "$field",
        "$/field",
        "$$.field",
        "prefix$.field",
        " $.field",
        "\t$.field",
    )
    for value in equivalent_values:
        assert bool(re.search(old_prefix_pattern, value)) is bool(
            re.search(provider_safe_pattern, value)
        )

    output_type = adapter_module._reflection_output_type(1, (_CONTRAST_A,))
    valid_insight = {
        "claim": "claim",
        "trigger": "trigger",
        "mechanism": "mechanism",
        "affected_paths": ["$.field"],
        "evidence_summary": "evidence",
        "evidence_contrast_ids": [_CONTRAST_A],
        "confidence": 0.5,
    }
    for rooted_path in ("$", "$.field", "$[0]", "$..nested"):
        validated = output_type.model_validate(
            {"insights": [{**valid_insight, "affected_paths": [rooted_path]}]},
            strict=True,
        )
        assert validated.insights[0].affected_paths == [rooted_path]
    # Whitespace normalization predates this wire repair and remains explicit:
    # StringConstraints strips only the outer whitespace before matching.
    for padded_path in (" $.field", "\t$.field\n"):
        validated = output_type.model_validate(
            {"insights": [{**valid_insight, "affected_paths": [padded_path]}]},
            strict=True,
        )
        assert validated.insights[0].affected_paths == ["$.field"]
    for hostile_path in (
        "",
        "field",
        ".field",
        "[0]",
        "$field",
        "$$.field",
        "prefix$.field",
        "x $.field",
        "\tX$.field",
        "$\n.field",
        "$.field\ntrailing",
    ):
        with pytest.raises(ValidationError):
            output_type.model_validate(
                {"insights": [{**valid_insight, "affected_paths": [hostile_path]}]},
                strict=True,
            )


def test_reflection_contrast_ids_are_closed_and_request_scoped() -> None:
    insight = {
        "claim": "claim",
        "trigger": "trigger",
        "mechanism": "mechanism",
        "affected_paths": ["$.field"],
        "evidence_summary": "human-readable evidence",
        "evidence_contrast_ids": [_CONTRAST_A],
        "confidence": 0.5,
    }
    output_type = adapter_module._reflection_output_type(1, (_CONTRAST_A,))
    assert output_type.model_validate({"insights": [insight]}, strict=True)
    assert (
        output_type.model_json_schema()["properties"]["insights"]["items"][
            "properties"
        ]["evidence_contrast_ids"]["minItems"]
        == 1
    )
    with pytest.raises(ValidationError):
        output_type.model_validate(
            {"insights": [{**insight, "evidence_contrast_ids": []}]},
            strict=True,
        )

    for bad_id in ("1" * 63, "A" * 64, _CONTRAST_B):
        with pytest.raises(ValidationError):
            output_type.model_validate(
                {"insights": [{**insight, "evidence_contrast_ids": [bad_id]}]},
                strict=True,
            )
    duplicate_type = adapter_module._reflection_output_type(
        1,
        (_CONTRAST_A, _CONTRAST_B),
    )
    with pytest.raises(ValidationError, match="cannot contain duplicates"):
        duplicate_type.model_validate(
            {
                "insights": [
                    {
                        **insight,
                        "evidence_contrast_ids": [_CONTRAST_A, _CONTRAST_A],
                    }
                ]
            },
            strict=True,
        )
    with pytest.raises(ValidationError):
        output_type.model_validate(
            {
                "insights": [
                    {
                        key: value
                        for key, value in insight.items()
                        if key != "evidence_contrast_ids"
                    }
                ]
            },
            strict=True,
        )

    no_evidence_type = adapter_module._reflection_output_type(1)
    assert no_evidence_type.model_validate(
        {"insights": [{**insight, "evidence_contrast_ids": []}]},
        strict=True,
    )
    with pytest.raises(ValidationError):
        no_evidence_type.model_validate({"insights": [insight]}, strict=True)


def test_reflection_evidence_catalog_is_canonical_authenticated_and_exact() -> None:
    catalog = ReflectionEvidenceCatalog.from_contrast_ids((_CONTRAST_A, _CONTRAST_B))

    assert catalog.citation_keys == ("e0001", "e0002")
    assert catalog.citation_key_for_contrast_id(_CONTRAST_A) == "e0001"
    assert catalog.citation_key_for_contrast_id(_CONTRAST_B) == "e0002"
    assert catalog.contrast_ids == (_CONTRAST_A, _CONTRAST_B)
    assert len(catalog.catalog_identity_sha256) == 64
    assert (
        ReflectionEvidenceCatalog.from_contrast_ids(
            (_CONTRAST_A, _CONTRAST_B)
        ).catalog_identity_sha256
        == catalog.catalog_identity_sha256
    )
    assert catalog.resolve_citation_keys(("e0002", "e0001")) == (
        _CONTRAST_A,
        _CONTRAST_B,
    )

    with pytest.raises(ValueError, match="unknown or foreign"):
        catalog.resolve_citation_keys(("e0003",))
    with pytest.raises(ValueError, match="duplicates"):
        catalog.resolve_citation_keys(("e0001", "e0001"))
    with pytest.raises(ValueError, match="not present"):
        catalog.citation_key_for_contrast_id("f" * 64)
    with pytest.raises(ValueError, match="contiguous"):
        ReflectionEvidenceCatalog(
            (
                ReflectionEvidenceCatalogEntry("e0002", _CONTRAST_A),
                ReflectionEvidenceCatalogEntry("e0003", _CONTRAST_B),
            )
        )
    with pytest.raises(ValueError, match="exact available_contrast_ids"):
        _reflection_request(
            available_contrast_ids=(_CONTRAST_A,),
            evidence_catalog=catalog,
        )


def test_catalog_reflection_resolves_short_keys_before_insight_construction() -> None:
    catalog = ReflectionEvidenceCatalog.from_contrast_ids((_CONTRAST_A, _CONTRAST_B))
    contract = ReflectionInsightContract(
        required_metric_ids=("objective:cost", "violation:capacity"),
        allowed_option_families=("control_only", "joint_edit"),
    )
    captured_output: dict[str, object] = {}

    def handle(request: StructuredGenerationRequest[Any]):
        properties = request.output_type.model_json_schema()["properties"]["insights"][
            "items"
        ]["properties"]
        assert "evidence_contrast_ids" not in properties
        assert properties["evidence_citation_keys"] == {
            "type": "array",
            "items": {
                "type": "string",
                "pattern": "^e[0-9]{4}$",
                "enum": ["e0001", "e0002"],
            },
            "uniqueItems": True,
            "minItems": 1,
            "maxItems": 2,
        }
        invalid = {
            "claim": "claim",
            "trigger": "trigger",
            "mechanism": "mechanism",
            "affected_paths": ["$.field"],
            "evidence_summary": "evidence",
            "evidence_citation_keys": ["e0003"],
            "confidence": 0.5,
            "effect_predictions": [
                {"metric_id": "objective:cost", "direction": "decrease"},
                {"metric_id": "violation:capacity", "direction": "increase"},
            ],
            "recommended_option_families": ["joint_edit"],
            "action_template": "Apply the sealed joint edit.",
            "falsification_condition": "The held-out effect reverses.",
        }
        with pytest.raises(ValidationError):
            request.output_type.model_validate(
                {"insights": [invalid]},
                strict=True,
            )
        with pytest.raises(ValidationError, match="cannot contain duplicates"):
            request.output_type.model_validate(
                {
                    "insights": [
                        {
                            **invalid,
                            "evidence_citation_keys": ["e0001", "e0001"],
                        }
                    ]
                },
                strict=True,
            )
        with pytest.raises(ValidationError):
            request.output_type.model_validate(
                {
                    "insights": [
                        {
                            **invalid,
                            "evidence_citation_keys": [_CONTRAST_A],
                        }
                    ]
                },
                strict=True,
            )
        reflection = request.output_type.model_validate(
            {
                "insights": [
                    {
                        **invalid,
                        "evidence_citation_keys": ["e0002", "e0001"],
                    }
                ]
            },
            strict=True,
        )
        captured_output.update(reflection.model_dump(mode="json", by_alias=False))
        return _response(reflection)

    runner = _FakeRunner(handle)
    result = asyncio.run(
        PydanticAIAgenticGenerator(runner).reflect(
            _reflection_request(
                max_insights=1,
                min_insights=1,
                insight_contract=contract,
                evidence_catalog=catalog,
            )
        )
    )

    assert captured_output["insights"][0]["evidence_citation_keys"] == [
        "e0002",
        "e0001",
    ]
    assert result.insights[0].evidence_contrast_ids == (
        _CONTRAST_A,
        _CONTRAST_B,
    )
    assert result.evidence_catalog_identity_sha256 == catalog.catalog_identity_sha256
    low_request = runner.requests[0]
    assert catalog.catalog_identity_sha256 in low_request.prompt
    assert '"citation_key":"e0001"' in low_request.prompt
    assert _CONTRAST_A in low_request.prompt
    assert (
        low_request.prompt_lineage.renderer_revision
        == adapter_module.REFLECTION_EVIDENCE_CATALOG_WIRE_CONTRACT_REVISION
    )
    assert low_request.prompt_lineage.renderer_definition_sha256 == (
        adapter_module.REFLECTION_EVIDENCE_CATALOG_PROMPT_RENDERER_DEFINITION_SHA256
    )
    validate_reflection_evidence_catalog_result(
        _reflection_request(
            max_insights=1,
            min_insights=1,
            insight_contract=contract,
            evidence_catalog=catalog,
        ),
        result,
    )
    with pytest.raises(ValueError, match="foreign evidence catalog identity"):
        validate_reflection_evidence_catalog_result(
            _reflection_request(
                max_insights=1,
                min_insights=1,
                insight_contract=contract,
                evidence_catalog=catalog,
            ),
            type(result)(
                insights=result.insights,
                telemetry=result.telemetry,
                evidence_catalog_identity_sha256="f" * 64,
            ),
        )


def test_v1_uncited_reflection_payloads_are_all_rejected_by_v2_wire_gate() -> None:
    """Replay the three historical failure shapes without a model or evaluator."""

    historical = (
        {
            "claim": "Index 1 rewrite to rewrite_z reduced LUTs in one context.",
            "trigger": "the frozen parent exposes $.sequence[1]",
            "mechanism": "a conjectured local rewrite effect",
            "affected_paths": ["$.sequence[1]"],
            "evidence_summary": "Contrast 0adf0881 was cited only by prefix.",
            "evidence_contrast_ids": [],
            "confidence": 0.7,
        },
        {
            "claim": "Index 12 refactor to refactor_z reduced LUTs in one context.",
            "trigger": "the frozen parent exposes $.sequence[12]",
            "mechanism": "a conjectured local refactor effect",
            "affected_paths": ["$.sequence[12]"],
            "evidence_summary": "Contrast 2d1a7fd was cited only by prefix.",
            "evidence_contrast_ids": [],
            "confidence": 0.7,
        },
        {
            "claim": "The historical index-18 prose misnamed its old action.",
            "trigger": "the frozen parent exposes $.sequence[18]",
            "mechanism": "model prose is not an intervention fact",
            "affected_paths": ["$.sequence[18]"],
            "evidence_summary": "Contrast bdfcf42 was cited only by prefix.",
            "evidence_contrast_ids": [],
            "confidence": 0.65,
        },
    )
    output_type = adapter_module._reflection_output_type(
        3,
        (_CONTRAST_A, _CONTRAST_B),
    )
    for payload in historical:
        with pytest.raises(ValidationError):
            output_type.model_validate(
                {"insights": [payload]},
                strict=True,
            )


def test_high_level_reflection_contract_keeps_legacy_defaults_but_validates_ids() -> (
    None
):
    legacy = InsightDraft(
        claim="legacy claim",
        trigger="legacy trigger",
        mechanism="legacy mechanism",
        affected_paths=("$.field",),
        evidence_summary="legacy prose",
        confidence=0.5,
    )
    assert legacy.evidence_contrast_ids == ()
    assert _reflection_request(available_contrast_ids=()).available_contrast_ids == ()

    with pytest.raises(TypeError, match="lowercase SHA-256"):
        InsightDraft(
            claim="claim",
            trigger="trigger",
            mechanism="mechanism",
            affected_paths=("$.field",),
            evidence_summary="evidence",
            confidence=0.5,
            evidence_contrast_ids=("A" * 64,),
        )
    with pytest.raises(ValueError, match="canonically sorted"):
        _reflection_request(
            available_contrast_ids=(_CONTRAST_B, _CONTRAST_A),
        )


def test_advanced_reflection_schema_is_request_scoped_exact_and_actionable() -> None:
    contract = ReflectionInsightContract(
        required_metric_ids=("objective:cost", "violation:capacity"),
        allowed_option_families=("control_only", "joint_edit"),
    )

    def handle(request: StructuredGenerationRequest[Any]):
        schema = request.output_type.model_json_schema()
        properties = schema["properties"]["insights"]["items"]["properties"]
        assert properties["effect_predictions"]["items"]["properties"]["metric_id"][
            "enum"
        ] == ["objective:cost", "violation:capacity"]
        assert properties["recommended_option_families"]["items"]["enum"] == [
            "control_only",
            "joint_edit",
        ]
        payload = {
            "claim": "The joint edit may lower cost without increasing violation.",
            "trigger": "The held-out parent exposes the same joint-edit family.",
            "mechanism": "Select one bounded joint edit from the finite palette.",
            "affected_paths": ["$.design", "$.control"],
            "evidence_summary": "The claim is limited to the cited contrast.",
            "evidence_contrast_ids": [_CONTRAST_A],
            "confidence": 0.5,
            "effect_predictions": [
                {"metric_id": "violation:capacity", "direction": "unchanged"},
                {"metric_id": "objective:cost", "direction": "decrease"},
            ],
            "recommended_option_families": ["joint_edit"],
            "action_template": "Choose a joint_edit option with bounded amplitude.",
            "falsification_condition": (
                "Falsify if held-out cost does not decrease or violation increases."
            ),
        }
        invalid_metric_payload = {
            **payload,
            "effect_predictions": [
                payload["effect_predictions"][0],
                payload["effect_predictions"][0],
            ],
        }
        for invalid in (
            {**payload, "effect_predictions": payload["effect_predictions"][:1]},
            invalid_metric_payload,
            {
                **payload,
                "effect_predictions": [
                    {
                        "metric_id": "objective:cost",
                        "direction": "unknown",
                    },
                    {
                        "metric_id": "violation:capacity",
                        "direction": "unknown",
                    },
                ],
            },
            {**payload, "recommended_option_families": ["foreign_family"]},
            {
                key: value
                for key, value in payload.items()
                if key != "falsification_condition"
            },
        ):
            with pytest.raises(ValidationError):
                request.output_type.model_validate({"insights": [invalid]}, strict=True)
        with pytest.raises(ValidationError) as exc_info:
            request.output_type.model_validate(
                {"insights": [invalid_metric_payload]},
                strict=True,
            )
        assert {
            error["type"] for error in exc_info.value.errors(include_url=False)
        } == {"reflection_metric_contract_violation"}
        reflection = request.output_type.model_validate(
            {"insights": [payload]}, strict=True
        )
        return _response(reflection)

    result = asyncio.run(
        PydanticAIAgenticGenerator(_FakeRunner(handle)).reflect(
            _reflection_request(max_insights=2, insight_contract=contract)
        )
    )
    assert result.insights[0].effect_predictions == (
        MetricEffectPrediction("objective:cost", MetricEffectDirection.DECREASE),
        MetricEffectPrediction("violation:capacity", MetricEffectDirection.UNCHANGED),
    )
    assert result.insights[0].recommended_option_families == ("joint_edit",)
    assert result.insights[0].falsification_condition is not None


def test_exact_action_reflection_schema_requires_one_allowed_option_id() -> None:
    contract = ReflectionInsightContract(
        required_metric_ids=("objective:cost",),
        allowed_option_families=("shape_only",),
        allowed_option_ids=("shape.lower.small", "shape.raise.small"),
    )

    def handle(request: StructuredGenerationRequest[Any]):
        properties = request.output_type.model_json_schema()["properties"]["insights"][
            "items"
        ]["properties"]
        assert properties["recommended_option_ids"]["items"]["enum"] == [
            "shape.lower.small",
            "shape.raise.small",
        ]
        payload = {
            "claim": "The cited bounded raise action may transfer.",
            "trigger": "The held-out parent exposes the same exact action ID.",
            "mechanism": "Apply the sealed small raise action.",
            "affected_paths": ["$.shape"],
            "evidence_summary": "Limited to the cited singleton contrast.",
            "evidence_contrast_ids": [_CONTRAST_A],
            "confidence": 0.5,
            "effect_predictions": [
                {"metric_id": "objective:cost", "direction": "decrease"}
            ],
            "recommended_option_families": ["shape_only"],
            "recommended_option_ids": ["shape.raise.small"],
            "action_template": "Select shape.raise.small exactly.",
            "falsification_condition": "Falsify if cost does not decrease.",
        }
        for invalid in (
            {
                key: value
                for key, value in payload.items()
                if key != "recommended_option_ids"
            },
            {**payload, "recommended_option_ids": ["shape.foreign.small"]},
        ):
            with pytest.raises(ValidationError):
                request.output_type.model_validate({"insights": [invalid]}, strict=True)
        return _response(
            request.output_type.model_validate({"insights": [payload]}, strict=True)
        )

    result = asyncio.run(
        PydanticAIAgenticGenerator(_FakeRunner(handle)).reflect(
            _reflection_request(insight_contract=contract)
        )
    )
    assert result.insights[0].recommended_option_ids == ("shape.raise.small",)


def test_advanced_reflection_contract_identity_and_evidence_gate_are_exact() -> None:
    contract = ReflectionInsightContract(
        required_metric_ids=("objective:cost", "violation:capacity"),
        allowed_option_families=("control_only", "joint_edit"),
    )
    record = contract.to_record()
    assert record["contract_identity_sha256"] == contract.identity_sha256
    assert (
        contract.identity_sha256
        == ReflectionInsightContract(
            required_metric_ids=("objective:cost", "violation:capacity"),
            allowed_option_families=("control_only", "joint_edit"),
        ).identity_sha256
    )
    assert (
        contract.identity_sha256
        != ReflectionInsightContract(
            required_metric_ids=("objective:cost",),
            allowed_option_families=("control_only", "joint_edit"),
        ).identity_sha256
    )
    assert (
        contract.identity_sha256
        != ReflectionInsightContract(
            required_metric_ids=("objective:cost", "violation:capacity"),
            allowed_option_families=("control_only",),
        ).identity_sha256
    )

    evidence_free = InsightDraft(
        claim="A bounded intervention may lower cost.",
        trigger="The same finite family is legal.",
        mechanism="Select one bounded catalog option.",
        affected_paths=("$.design",),
        evidence_summary="No machine contrast was cited.",
        evidence_contrast_ids=(),
        confidence=0.5,
        effect_predictions=(
            MetricEffectPrediction(
                "objective:cost",
                MetricEffectDirection.DECREASE,
            ),
            MetricEffectPrediction(
                "violation:capacity",
                MetricEffectDirection.UNCHANGED,
            ),
        ),
        recommended_option_families=("joint_edit",),
        action_template="Choose one bounded joint_edit option.",
        falsification_condition="Falsify if held-out cost does not decrease.",
    )
    with pytest.raises(ValueError, match="cite at least one evidence contrast"):
        validate_reflection_insight_draft(evidence_free, contract)
    validate_reflection_insight_draft(
        evidence_free,
        contract,
        allow_missing_evidence=True,
    )


def test_explicitly_inactive_advanced_contract_preserves_legacy_wire_schema() -> None:
    implicit = adapter_module._reflection_output_type(1, (_CONTRAST_A,))
    explicit = adapter_module._reflection_output_type(
        1,
        (_CONTRAST_A,),
        None,
    )
    assert json.dumps(
        implicit.model_json_schema(), sort_keys=True, separators=(",", ":")
    ) == json.dumps(explicit.model_json_schema(), sort_keys=True, separators=(",", ":"))


def test_proposal_schema_rejects_extra_coercion_blank_values_and_bounds() -> None:
    captured: list[type[BaseModel]] = []

    def handle(request: StructuredGenerationRequest[Any]):
        captured.append(request.output_type)
        return _response(
            request.output_type(
                configuration={"width": 1, "wire_name": "ok", "note": None},
                design_rationale="valid",
            )
        )

    asyncio.run(
        PydanticAIAgenticGenerator(_FakeRunner(handle)).propose(_variation_request())
    )
    output_type = captured[0]
    valid = {
        "configuration": {"width": 1, "wire_name": "ok", "note": None},
        "design_rationale": "valid",
    }

    with pytest.raises(ValidationError):
        output_type.model_validate({**valid, "unexpected": "field"})
    with pytest.raises(ValidationError):
        output_type.model_validate({**valid, "design_rationale": 7})
    with pytest.raises(ValidationError):
        output_type.model_validate({**valid, "design_rationale": " \t "})
    with pytest.raises(ValidationError):
        output_type.model_validate(
            {
                **valid,
                "intended_changes": ["change"] * (MAX_INTENDED_CHANGES + 1),
            }
        )
    with pytest.raises(ValidationError):
        output_type.model_validate(
            {
                **valid,
                "source_attribution": [{"path": "x", "source": "unknown"}],
            }
        )
    with pytest.raises(ValidationError):
        output_type.model_validate(
            {
                **valid,
                "conflict_resolutions": [
                    {
                        "relation_id": "r",
                        "choice": "choose_both",
                        "explanation": "not permitted",
                    }
                ],
            }
        )


def test_candidate_payload_allowed_by_compact_shape_still_fails_strict_core() -> None:
    output_type = adapter_module._candidate_proposal_type(
        _Candidate,
        "typed_mutation",
    )
    # This satisfies the advertised generic object/string/list envelope, but
    # violates the hidden detailed candidate, path, and enum contracts.
    compact_shape_only_payload = {
        "configuration": {
            "width": "9",
            "wire_name": "candidate-nine",
            "note": None,
        },
        "design_rationale": "valid",
        "source_attribution": [{"path": "width", "source": "mutation"}],
    }

    assert (
        output_type.model_json_schema()["properties"]["configuration"][
            "additionalProperties"
        ]
        is True
    )
    with pytest.raises(ValidationError) as caught:
        output_type.model_validate(compact_shape_only_payload, strict=True)
    assert {error["type"] for error in caught.value.errors()} >= {
        "int_type",
        "string_pattern_mismatch",
    }


@pytest.mark.parametrize("confidence", [True, -0.01, 1.01, float("nan")])
def test_reflection_schema_rejects_invalid_confidence(confidence: object) -> None:
    output_types: list[type[BaseModel]] = []

    def handle(request: StructuredGenerationRequest[Any]):
        output_types.append(request.output_type)
        return _response(request.output_type())

    asyncio.run(
        PydanticAIAgenticGenerator(_FakeRunner(handle)).reflect(_reflection_request())
    )
    insight = {
        "claim": "claim",
        "trigger": "trigger",
        "mechanism": "mechanism",
        "affected_paths": ["$.field"],
        "evidence_summary": "evidence",
        "evidence_contrast_ids": [_CONTRAST_A],
        "confidence": confidence,
    }
    with pytest.raises(ValidationError):
        output_types[0].model_validate({"insights": [insight]})


def test_reflection_schema_enforces_request_count_and_string_and_path_bounds() -> None:
    output_types: list[type[BaseModel]] = []

    def handle(request: StructuredGenerationRequest[Any]):
        output_types.append(request.output_type)
        return _response(request.output_type())

    asyncio.run(
        PydanticAIAgenticGenerator(_FakeRunner(handle)).reflect(
            _reflection_request(max_insights=1)
        )
    )
    output_type = output_types[0]
    valid_insight = {
        "claim": "claim",
        "trigger": "trigger",
        "mechanism": "mechanism",
        "affected_paths": ["$.field"],
        "evidence_summary": "evidence",
        "evidence_contrast_ids": [_CONTRAST_A],
        "confidence": 0.5,
    }

    wire_schema = output_type.model_json_schema()["properties"]["insights"]
    assert wire_schema["minItems"] == 0
    assert wire_schema["maxItems"] == 1
    insight_properties = wire_schema["items"]["properties"]
    assert insight_properties["affected_paths"] == {
        "type": "array",
        "minItems": 1,
        "maxItems": MAX_AFFECTED_PATHS,
        "items": {
            "type": "string",
            "minLength": 1,
            "maxLength": adapter_module.MAX_PATH_CHARS,
            "pattern": adapter_module._JSON_PATH_PATTERN,
            "description": (
                "A JSON-style path rooted at '$', such as '$.field' or '$[0]'."
            ),
        },
    }
    for text_field in (
        "claim",
        "trigger",
        "mechanism",
        "evidence_summary",
    ):
        assert insight_properties[text_field] == {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_REFLECTION_TEXT_CHARS,
        }
    assert insight_properties["confidence"] == {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
    }

    for rooted_path in ("$", "$.field", "$[0]"):
        assert output_type.model_validate(
            {"insights": [{**valid_insight, "affected_paths": [rooted_path]}]},
            strict=True,
        )
    for invalid_path in (
        "field",
        ".field",
        "[0]",
        "$" + "x" * adapter_module.MAX_PATH_CHARS,
    ):
        with pytest.raises(ValidationError):
            output_type.model_validate(
                {"insights": [{**valid_insight, "affected_paths": [invalid_path]}]},
                strict=True,
            )
    with pytest.raises(ValidationError):
        output_type.model_validate(
            {"insights": [{**valid_insight, "affected_paths": []}]},
            strict=True,
        )

    with pytest.raises(ValidationError):
        output_type.model_validate({"insights": [valid_insight, valid_insight]})
    with pytest.raises(ValidationError):
        output_type.model_validate(
            {
                "insights": [
                    {**valid_insight, "claim": "x" * (MAX_REFLECTION_TEXT_CHARS + 1)}
                ]
            }
        )
    with pytest.raises(ValidationError):
        output_type.model_validate(
            {
                "insights": [
                    {
                        **valid_insight,
                        "affected_paths": ["x"] * (MAX_AFFECTED_PATHS + 1),
                    }
                ]
            }
        )
    with pytest.raises(ValidationError):
        output_type.model_validate(
            {"insights": [{**valid_insight, "unexpected": "field"}]}
        )


def test_reflection_minimum_cardinality_is_request_scoped_and_on_wire() -> None:
    insight = {
        "claim": "claim",
        "trigger": "trigger",
        "mechanism": "mechanism",
        "affected_paths": ["$.field"],
        "evidence_summary": "evidence",
        "evidence_contrast_ids": [_CONTRAST_A],
        "confidence": 0.5,
    }
    output_type = adapter_module._reflection_output_type(
        2,
        (_CONTRAST_A,),
        min_insights=2,
    )
    wire_schema = output_type.model_json_schema()["properties"]["insights"]
    assert wire_schema["minItems"] == 2
    assert wire_schema["maxItems"] == 2
    with pytest.raises(ValidationError):
        output_type.model_validate({}, strict=True)
    with pytest.raises(ValidationError):
        output_type.model_validate({"insights": []}, strict=True)
    with pytest.raises(ValidationError):
        output_type.model_validate({"insights": [insight]}, strict=True)
    with pytest.raises(ValidationError, match="distinct normalized claims"):
        output_type.model_validate(
            {"insights": [insight, insight]},
            strict=True,
        )
    assert output_type.model_validate(
        {"insights": [insight, {**insight, "claim": "second claim"}]},
        strict=True,
    )

    with pytest.raises(ValueError, match="min_insights"):
        _reflection_request(max_insights=2, min_insights=3)


def test_reflection_wire_contract_advertises_every_replayed_runtime_failure() -> None:
    output_type = adapter_module._reflection_output_type(2, (_CONTRAST_A,))
    replayed_failure = {
        "insights": [
            {
                "claim": "x" * (MAX_REFLECTION_TEXT_CHARS + 1),
                "trigger": "trigger",
                "mechanism": "mechanism",
                "affected_paths": ["not-a-json-path"],
                "evidence_summary": "evidence",
                "evidence_contrast_ids": [_CONTRAST_A],
                "confidence": 0.5,
            }
        ]
    }

    schema = output_type.model_json_schema()
    assert "$defs" not in schema
    properties = schema["properties"]["insights"]["items"]["properties"]
    assert properties["claim"]["maxLength"] == MAX_REFLECTION_TEXT_CHARS
    assert properties["affected_paths"]["items"]["pattern"] == (
        adapter_module._JSON_PATH_PATTERN
    )
    with pytest.raises(ValidationError) as caught:
        output_type.model_validate(replayed_failure, strict=True)
    assert {error["type"] for error in caught.value.errors()} >= {
        "string_pattern_mismatch",
        "string_too_long",
    }


@pytest.mark.parametrize("method", ["propose", "reflect"])
def test_low_level_errors_propagate_with_identity_untouched(method: str) -> None:
    error = StructuredGenerationError(
        kind=GenerationFailureKind.RATE_LIMITED,
        retryable=True,
        safe_message="provider rate limit",
        status_code=429,
        retry_after_seconds=2,
    )

    async def raise_error(_request: StructuredGenerationRequest[Any]):
        raise error

    generator = PydanticAIAgenticGenerator(raise_error)
    awaitable = (
        generator.propose(_variation_request())
        if method == "propose"
        else generator.reflect(_reflection_request())
    )
    with pytest.raises(StructuredGenerationError) as caught:
        asyncio.run(awaitable)
    assert caught.value is error


def test_low_level_contract_rejects_wrong_response_and_wrong_value_type() -> None:
    wrong_response = _FakeRunner(lambda _request: object())
    with pytest.raises(TypeError, match="low-level runner must return"):
        asyncio.run(
            PydanticAIAgenticGenerator(wrong_response).propose(_variation_request())
        )

    wrong_value = _FakeRunner(
        lambda _request: _response(
            _Candidate(
                width=1,
                wire_name="wrong-level",
                note=None,
            )
        )
    )
    with pytest.raises(TypeError, match="requested output type"):
        asyncio.run(
            PydanticAIAgenticGenerator(wrong_value).propose(_variation_request())
        )


@pytest.mark.parametrize("attempt_count", [0, -1, True, 1.5])
def test_attempt_response_envelope_requires_positive_exact_count(
    attempt_count: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        AttemptedStructuredGenerationResponse(
            response=_response(object()),
            attempt_count=attempt_count,  # type: ignore[arg-type]
        )


def test_candidate_model_must_be_a_concrete_object_pydantic_model() -> None:
    from pydantic import RootModel

    class _RootCandidate(RootModel[int]):
        pass

    async def unused(_request: StructuredGenerationRequest[Any]):
        raise AssertionError("invalid model must fail before runner invocation")

    generator = PydanticAIAgenticGenerator(unused)
    for candidate_model in (dict, BaseModel, _RootCandidate):
        with pytest.raises(TypeError):
            asyncio.run(
                generator.propose(_variation_request(candidate_model=candidate_model))
            )


def test_each_call_uses_a_fresh_schema_bound_to_its_candidate_model() -> None:
    class _OtherCandidate(BaseModel):
        enabled: bool

    output_types: list[type[BaseModel]] = []

    def handle(request: StructuredGenerationRequest[Any]):
        output_types.append(request.output_type)
        candidate = request.output_type.model_fields["configuration"].annotation
        configuration = (
            {"width": 1, "wire_name": "first", "note": None}
            if candidate is _Candidate
            else {"enabled": True}
        )
        return _response(
            request.output_type(
                configuration=configuration,
                design_rationale="valid",
            )
        )

    generator = PydanticAIAgenticGenerator(_FakeRunner(handle))
    asyncio.run(generator.propose(_variation_request()))
    asyncio.run(generator.propose(_variation_request(candidate_model=_OtherCandidate)))

    assert output_types[0] is not output_types[1]
    assert output_types[0].model_fields["configuration"].annotation is _Candidate
    assert output_types[1].model_fields["configuration"].annotation is _OtherCandidate


def test_adapter_implements_protocol_and_has_no_provider_or_environment_boundary() -> (
    None
):
    async def unused(_request: StructuredGenerationRequest[Any]):
        raise AssertionError

    assert isinstance(PydanticAIAgenticGenerator(unused), AgenticGenerator)

    source = inspect.getsource(adapter_module)
    assert "async_generator" not in source
    assert "pydantic_ai import" not in source
    assert "httpx" not in source
    assert "openai" not in source
    assert "os.environ" not in source
    assert "dotenv" not in source
    assert Path(adapter_module.__file__).name == "agentic_generator.py"
