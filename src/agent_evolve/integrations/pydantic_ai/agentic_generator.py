"""High-level agentic generation over the provider-neutral structured port.

This module owns only schema construction and translation.  The injected
callable owns execution (and may itself be backed by a retrying queue), so this
adapter performs no provider setup, I/O, retrying, or exception translation.
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Generic, Literal, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    create_model,
    field_validator,
    model_validator,
)

from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey
from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonValue,
    freeze_json,
    is_json_scalar,
    thaw_json,
    typed_json_equal,
)
from agent_evolve.policies.variation.typed_patch import (
    replace_existing_path,
    value_at_path,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    AtomicMutationDraft,
    AtomicMutationOutputContract,
    CandidateDraft,
    ConflictResolutionDraft,
    FiniteVariationSelectionDraft,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    ReflectionInsightContract,
    SourceAttribution,
    VariationGenerationRequest,
    VariationGenerationResult,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)


CANDIDATE_PROPOSAL_TOOL_NAME = "return_candidate_proposal"
ATOMIC_MUTATION_TOOL_NAME = "return_atomic_mutation"
FINITE_VARIATION_SELECTION_TOOL_NAME = "select_finite_variation_option"
REFLECTION_TOOL_NAME = "return_reflection_insights"

# These are local admission limits, not experiment policy.  The provider sees a
# deliberately compact schema, while Pydantic retains these constraints in its
# core schema and validates them before a queue-owned attempt can succeed.
MAX_RATIONALE_CHARS = 16_384
MAX_CHANGE_CHARS = 4_096
MAX_INTENDED_CHANGES = 256
MAX_PATH_CHARS = 4_096
MAX_SOURCE_ATTRIBUTIONS = 4_096
MAX_ID_CHARS = 512
MAX_CLAIMED_INSIGHT_IDS = 1_024
MAX_PRESERVATION_IDS = 4_096
MAX_RELATION_ID_CHARS = 512
MAX_RESOLUTION_EXPLANATION_CHARS = 8_192
MAX_CONFLICT_RESOLUTIONS = 4_096
MAX_REFLECTION_TEXT_CHARS = 16_384
MAX_AFFECTED_PATHS = 256
MAX_EVIDENCE_CONTRAST_IDS = 256


_Rationale = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_RATIONALE_CHARS,
    ),
]
_Change = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_CHANGE_CHARS,
    ),
]
_Path = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_PATH_CHARS,
        pattern=r"^\$(?:$|[.\[])",
    ),
]
_Identifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_ID_CHARS,
    ),
]
_RelationIdentifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_RELATION_ID_CHARS,
    ),
]
_ResolutionExplanation = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_RESOLUTION_EXPLANATION_CHARS,
    ),
]
_ReflectionText = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_REFLECTION_TEXT_CHARS,
    ),
]
_ContrastIdentifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[0-9a-f]{64}$",
    ),
]

_STRICT_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    strict=True,
    frozen=True,
)


def _path_text(path: JsonPath) -> str:
    parts = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            parts.append(f".{segment.value}")
        elif type(segment) is ArrayIndex:
            parts.append(f"[{segment.value}]")
        else:  # pragma: no cover - JsonPath closes the segment union.
            raise AssertionError("unsupported path segment")
    return "".join(parts)


def _local_ref_value(root: dict[str, Any], reference: str) -> dict[str, Any]:
    """Resolve one local JSON pointer without accepting external schemas."""

    if type(reference) is not str or not reference.startswith("#/"):
        raise ValueError("atomic leaf schema contains a non-local $ref")
    current: object = root
    for raw_token in reference[2:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if type(current) is not dict or token not in current:
            raise ValueError("atomic leaf schema contains an unresolved $ref")
        current = current[token]
    if type(current) is not dict:
        raise ValueError("atomic leaf $ref must resolve to an object schema")
    return current


def _dereference_schema_top(
    schema: dict[str, Any],
    root: dict[str, Any],
    *,
    references: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Expand a top-level local ref while detecting recursive definitions."""

    if "$ref" not in schema:
        return copy.deepcopy(schema)
    reference = schema["$ref"]
    if type(reference) is not str:
        raise ValueError("atomic leaf schema $ref must be a string")
    if reference in references:
        raise ValueError("atomic leaf schema contains a recursive $ref")
    resolved = _dereference_schema_top(
        _local_ref_value(root, reference),
        root,
        references=references + (reference,),
    )
    siblings = {key: value for key, value in schema.items() if key != "$ref"}
    # Pydantic normally emits a bare local ref.  Siblings are legal JSON Schema,
    # but conflicting keys cannot be combined faithfully and therefore fail
    # closed instead of weakening either constraint.
    for key, value in siblings.items():
        if key in resolved and resolved[key] != value:
            raise ValueError("atomic leaf schema has conflicting $ref siblings")
        resolved[key] = copy.deepcopy(value)
    return resolved


def _descend_schema(
    schema: dict[str, Any],
    segment: ObjectKey | ArrayIndex,
    root: dict[str, Any],
) -> dict[str, Any]:
    current = _dereference_schema_top(schema, root)
    if type(segment) is ObjectKey:
        properties = current.get("properties")
        if type(properties) is dict and segment.value in properties:
            child = properties[segment.value]
            if type(child) is not dict:
                raise ValueError("atomic object property has no object schema")
            return child
    else:
        items = current.get("items")
        if type(items) is dict:
            return items
        prefix_items = current.get("prefixItems")
        if type(prefix_items) is list and segment.value < len(prefix_items):
            child = prefix_items[segment.value]
            if type(child) is not dict:
                raise ValueError("atomic tuple item has no object schema")
            return child

    for composition_key in ("anyOf", "oneOf", "allOf"):
        variants = current.get(composition_key)
        if type(variants) is not list:
            continue
        descended: list[dict[str, Any]] = []
        for variant in variants:
            if type(variant) is not dict:
                raise ValueError("atomic schema composition is malformed")
            try:
                descended.append(_descend_schema(variant, segment, root))
            except ValueError:
                continue
        if not descended:
            continue
        if len(descended) == 1:
            return descended[0]
        return {composition_key: descended}
    raise ValueError("atomic editable path cannot be resolved in candidate schema")


def _expand_local_refs(
    value: object,
    root: dict[str, Any],
    *,
    references: tuple[str, ...] = (),
) -> object:
    if type(value) is list:
        return [
            _expand_local_refs(item, root, references=references)
            for item in value
        ]
    if type(value) is not dict:
        return copy.deepcopy(value)
    current = value
    if "$ref" in current:
        reference = current["$ref"]
        if type(reference) is not str:
            raise ValueError("atomic leaf schema $ref must be a string")
        if reference in references:
            raise ValueError("atomic leaf schema contains a recursive $ref")
        current = _dereference_schema_top(
            current,
            root,
            references=references,
        )
        references = references + (reference,)
    return {
        key: _expand_local_refs(item, root, references=references)
        for key, item in current.items()
        if key != "$defs"
    }


def _schema_is_scalar(schema: dict[str, Any]) -> bool:
    scalar_types = {"null", "boolean", "integer", "number", "string"}
    if "const" in schema and not is_json_scalar(schema["const"]):
        return False
    enum = schema.get("enum")
    if type(enum) is list and (
        not enum or not all(is_json_scalar(item) for item in enum)
    ):
        return False
    declared_type = schema.get("type")
    if type(declared_type) is str and declared_type in scalar_types:
        return True
    if type(declared_type) is list and declared_type and all(
        type(item) is str and item in scalar_types for item in declared_type
    ):
        return True
    if "const" in schema:
        return True
    if type(enum) is list:
        return True
    for key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(key)
        if type(variants) is list and variants and all(
            type(item) is dict and _schema_is_scalar(item) for item in variants
        ):
            return True
    return False


def _atomic_leaf_schema(
    candidate_model: type[BaseModel],
    path: JsonPath,
) -> dict[str, Any]:
    """Resolve one exact candidate-model leaf into a closed scalar schema."""

    root = candidate_model.model_json_schema(by_alias=False)
    if type(root) is not dict:
        raise TypeError("candidate_model must return an object JSON schema")
    current = root
    for segment in path.segments:
        current = _descend_schema(current, segment, root)
    expanded = _expand_local_refs(current, root)
    if type(expanded) is not dict or not _schema_is_scalar(expanded):
        raise ValueError(
            "atomic editable path must resolve to a closed scalar JSON schema"
        )
    return expanded


def _exclude_current_scalar(
    schema: dict[str, Any],
    current: object,
) -> dict[str, Any]:
    """Make a provider-visible scalar schema exclude the observed parent value."""

    frozen_current = freeze_json(current)
    if not is_json_scalar(frozen_current):
        raise ValueError("atomic parent leaf must be a typed-JSON scalar")
    result = copy.deepcopy(schema)
    enum = result.get("enum")
    if type(enum) is list:
        remaining = [
            item
            for item in enum
            if not typed_json_equal(freeze_json(item), frozen_current)
        ]
        if len(remaining) != len(enum):
            if not remaining:
                raise ValueError("atomic leaf schema permits only the current value")
            result["enum"] = remaining
            return result
    if "const" in result and typed_json_equal(
        freeze_json(result["const"]), frozen_current
    ):
        raise ValueError("atomic leaf schema permits only the current value")
    exclusion = {"not": {"const": thaw_json(frozen_current)}}
    if "not" not in result:
        result.update(exclusion)
        return result
    return {"allOf": [result, exclusion]}


def _restrict_scalar_options(
    schema: dict[str, Any],
    options: tuple[FrozenJsonValue, ...],
) -> dict[str, Any]:
    """Intersect a leaf schema with one exact provider-visible option order."""

    if not options:
        return copy.deepcopy(schema)
    result = copy.deepcopy(schema)
    result["enum"] = [thaw_json(option) for option in options]
    return result


def _compact_candidate_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return the small provider contract while preserving the model's core schema."""

    properties: dict[str, object] = {}
    for name in model.model_fields:
        if name == "configuration":
            field_schema: dict[str, object] = {
                "type": "object",
                "additionalProperties": True,
            }
        elif name == "design_rationale":
            field_schema = {"type": "string"}
        elif name in {
            "intended_changes",
            "claimed_insight_ids",
            "claimed_preservation_obligation_ids",
        }:
            field_schema = {"type": "array", "items": {"type": "string"}}
        elif name == "source_attribution":
            field_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
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
                },
            }
        elif name == "conflict_resolutions":
            field_schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "relation_id": {"type": "string"},
                        "choice": {
                            "type": "string",
                            "enum": [
                                "choose_left",
                                "choose_right",
                                "synthesize",
                                "drop_both",
                            ],
                        },
                        "explanation": {"type": "string"},
                    },
                    "required": ["relation_id", "choice", "explanation"],
                    "additionalProperties": False,
                },
            }
        else:  # pragma: no cover - guarded by the closed dynamic field builder.
            raise RuntimeError(f"unsupported candidate wire field: {name}")
        properties[name] = field_schema
    required = [name for name, field in model.model_fields.items() if field.is_required()]
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


class _CompactCandidateSchemaBase(BaseModel):
    """Full local validation with a provider-compatible JSON-schema projection."""

    model_config = _STRICT_MODEL_CONFIG

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: Any,
        _handler: Any,
    ) -> dict[str, Any]:
        return _compact_candidate_json_schema(cls)


_AtomicScalar = bool | int | float | str | None


class _CompactAtomicMutationSchemaBase(BaseModel):
    """One scalar wire edit plus whole-candidate semantic validation."""

    model_config = _STRICT_MODEL_CONFIG
    expected_path_text: ClassVar[str]
    replacement_schema: ClassVar[dict[str, Any]]
    candidate_model: ClassVar[type[BaseModel]]
    output_contract: ClassVar[AtomicMutationOutputContract]

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: Any,
        _handler: Any,
    ) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "const": cls.expected_path_text,
                },
                "replacement": copy.deepcopy(cls.replacement_schema),
                "design_rationale": {"type": "string"},
                "claimed_insight_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["path", "replacement", "design_rationale"],
            "additionalProperties": False,
        }

    @model_validator(mode="after")
    def _validate_atomic_candidate(self) -> "_CompactAtomicMutationSchemaBase":
        contract = type(self).output_contract
        if self.path != type(self).expected_path_text:
            raise ValueError("atomic mutation returned the wrong path")
        replacement = freeze_json(self.replacement)
        if not is_json_scalar(replacement):  # pragma: no cover - closed field union.
            raise ValueError("atomic replacement must be a typed-JSON scalar")
        old_value = value_at_path(
            contract.parent_configuration,
            contract.editable_path,
        )
        if typed_json_equal(old_value, replacement):
            raise ValueError("atomic replacement must differ from the parent value")
        if contract.replacement_options and not any(
            typed_json_equal(replacement, option)
            for option in contract.replacement_options
        ):
            raise ValueError(
                "atomic replacement is outside the contracted option catalog"
            )
        target = replace_existing_path(
            contract.parent_configuration,
            contract.editable_path,
            replacement,
        )
        validated = type(self).candidate_model.model_validate(
            thaw_json(target),
            strict=True,
            by_alias=False,
            by_name=True,
        )
        validated_frozen = freeze_json(_configuration_dict(validated))
        if not typed_json_equal(validated_frozen, target):
            raise ValueError(
                "candidate validation changed the typed atomic target"
            )
        return self


class _CompactReflectionSchemaBase(BaseModel):
    """Bounded local reflection validation behind one generic wire envelope."""

    model_config = _STRICT_MODEL_CONFIG
    available_contrast_ids: ClassVar[tuple[str, ...]] = ()
    insight_contract: ClassVar[ReflectionInsightContract | None] = None
    min_insights: ClassVar[int] = 0
    max_insights: ClassVar[int] = 4

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        _core_schema: Any,
        _handler: Any,
    ) -> dict[str, Any]:
        contrast_item_schema: dict[str, Any] = {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
        }
        if cls.available_contrast_ids:
            contrast_item_schema["enum"] = list(cls.available_contrast_ids)
        insight_properties: dict[str, Any] = {
            "claim": {"type": "string"},
            "trigger": {"type": "string"},
            "mechanism": {"type": "string"},
            "affected_paths": {
                "type": "array",
                "items": {"type": "string"},
            },
            "evidence_summary": {"type": "string"},
            "evidence_contrast_ids": {
                "type": "array",
                "items": contrast_item_schema,
                "uniqueItems": True,
                "minItems": (
                    1 if cls.available_contrast_ids else 0
                ),
                "maxItems": (
                    min(
                        MAX_EVIDENCE_CONTRAST_IDS,
                        len(cls.available_contrast_ids),
                    )
                    if cls.available_contrast_ids
                    else 0
                ),
            },
            "confidence": {"type": "number"},
        }
        required = [
            "claim",
            "trigger",
            "mechanism",
            "affected_paths",
            "evidence_summary",
            "evidence_contrast_ids",
            "confidence",
        ]
        contract = cls.insight_contract
        if contract is not None:
            ReflectionInsightContract.__post_init__(contract)
            insight_properties.update(
                {
                    "effect_predictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "metric_id": {
                                    "type": "string",
                                    "enum": list(contract.required_metric_ids),
                                },
                                "direction": {
                                    "type": "string",
                                    "enum": [
                                        direction.value
                                        for direction in MetricEffectDirection
                                    ],
                                },
                            },
                            "required": ["metric_id", "direction"],
                            "additionalProperties": False,
                        },
                        "uniqueItems": True,
                        "minItems": len(contract.required_metric_ids),
                        "maxItems": len(contract.required_metric_ids),
                    },
                    "recommended_option_families": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(contract.allowed_option_families),
                        },
                        "uniqueItems": True,
                        "minItems": 1,
                        "maxItems": len(contract.allowed_option_families),
                    },
                    "action_template": {"type": "string"},
                    "falsification_condition": {"type": "string"},
                }
            )
            if contract.allowed_option_ids:
                insight_properties["recommended_option_ids"] = {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(contract.allowed_option_ids),
                    },
                    "uniqueItems": True,
                    "minItems": 1,
                    "maxItems": len(contract.allowed_option_ids),
                }
            required.extend(
                [
                    "effect_predictions",
                    "recommended_option_families",
                    "action_template",
                    "falsification_condition",
                ]
            )
            if contract.allowed_option_ids:
                required.append("recommended_option_ids")
        return {
            "type": "object",
            "properties": {
                "insights": {
                    "type": "array",
                    "minItems": cls.min_insights,
                    "maxItems": cls.max_insights,
                    "items": {
                        "type": "object",
                        "properties": insight_properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["insights"],
            "additionalProperties": False,
        }


class _SourceAttributionOutput(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    path: _Path
    source: Literal["ancestor", "left", "right", "synthesized", "mutation"]


class _ConflictResolutionOutput(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    relation_id: _RelationIdentifier
    choice: Literal["choose_left", "choose_right", "synthesize", "drop_both"]
    explanation: _ResolutionExplanation


class _ReflectionInsightOutput(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    claim: _ReflectionText
    trigger: _ReflectionText
    mechanism: _ReflectionText
    affected_paths: list[_Path] = Field(
        min_length=1,
        max_length=MAX_AFFECTED_PATHS,
    )
    evidence_summary: _ReflectionText
    evidence_contrast_ids: list[_ContrastIdentifier] = Field(
        max_length=MAX_EVIDENCE_CONTRAST_IDS,
    )
    confidence: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("evidence_contrast_ids")
    @classmethod
    def _unique_evidence_contrast_ids(cls, values: list[str]) -> list[str]:
        del cls
        if len(set(values)) != len(values):
            raise ValueError("evidence_contrast_ids cannot contain duplicates")
        return values


class _MetricEffectPredictionOutput(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    metric_id: _Identifier
    direction: Literal["decrease", "increase", "unchanged", "unknown"]


class _AdvancedReflectionInsightOutput(_ReflectionInsightOutput):
    """Strict runtime validator; the request supplies its closed vocabularies."""

    required_metric_ids: ClassVar[tuple[str, ...]] = ()
    allowed_option_families: ClassVar[tuple[str, ...]] = ()
    allowed_option_ids: ClassVar[tuple[str, ...]] = ()

    effect_predictions: list[_MetricEffectPredictionOutput]
    recommended_option_families: list[_Identifier] = Field(min_length=1)
    recommended_option_ids: list[_Identifier] = Field(default_factory=list)
    action_template: _ReflectionText
    falsification_condition: _ReflectionText

    @model_validator(mode="after")
    def _exact_advanced_contract(self) -> "_AdvancedReflectionInsightOutput":
        metric_ids = tuple(item.metric_id for item in self.effect_predictions)
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("effect_predictions cannot contain duplicate metrics")
        if set(metric_ids) != set(type(self).required_metric_ids):
            raise ValueError(
                "effect_predictions must cover the exact required metrics"
            )
        families = tuple(self.recommended_option_families)
        if len(set(families)) != len(families):
            raise ValueError(
                "recommended_option_families cannot contain duplicates"
            )
        if not set(families).issubset(type(self).allowed_option_families):
            raise ValueError(
                "recommended_option_families escape the request vocabulary"
            )
        option_ids = tuple(self.recommended_option_ids)
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("recommended_option_ids cannot contain duplicates")
        allowed_option_ids = type(self).allowed_option_ids
        if allowed_option_ids:
            if not option_ids:
                raise ValueError(
                    "exact-action reflections require a recommended option ID"
                )
            if not set(option_ids).issubset(allowed_option_ids):
                raise ValueError(
                    "recommended_option_ids escape the request vocabulary"
                )
        elif option_ids:
            raise ValueError(
                "recommended_option_ids require an exact-action vocabulary"
            )
        if all(item.direction == "unknown" for item in self.effect_predictions):
            raise ValueError(
                "an outcome-grounded insight must predict at least one direction"
            )
        return self


ResponseT = TypeVar("ResponseT")


@dataclass(frozen=True, slots=True)
class AttemptedStructuredGenerationResponse(Generic[ResponseT]):
    """A successful structured response plus its outer queue attempt count.

    A direct one-attempt runner can return ``StructuredGenerationResponse``
    unchanged.  A queue-composition function can return this envelope so retry
    evidence is not lost while the shared one-attempt port remains truthful.
    """

    response: StructuredGenerationResponse[ResponseT]
    attempt_count: int

    def __post_init__(self) -> None:
        if type(self.response) is not StructuredGenerationResponse:
            raise TypeError("response must be an exact StructuredGenerationResponse")
        StructuredGenerationResponse.__post_init__(self.response)
        if type(self.attempt_count) is not int or self.attempt_count <= 0:
            raise ValueError("attempt_count must be a positive exact integer")


LowLevelResult = (
    StructuredGenerationResponse[Any]
    | AttemptedStructuredGenerationResponse[Any]
)
LowLevelRunner = Callable[
    [StructuredGenerationRequest[Any]],
    Awaitable[LowLevelResult],
]


def _candidate_proposal_type(
    candidate_model: type[BaseModel],
    operation: str,
) -> type[BaseModel]:
    if not isinstance(candidate_model, type) or not issubclass(candidate_model, BaseModel):
        raise TypeError("candidate_model must be a Pydantic BaseModel subclass")
    if candidate_model is BaseModel or getattr(
        candidate_model, "__pydantic_root_model__", False
    ):
        raise TypeError("candidate_model must describe an object configuration")
    if getattr(candidate_model, "__parameters__", ()):
        raise TypeError("candidate_model must be a concrete Pydantic model")
    if type(operation) is not str or not operation.strip():
        raise ValueError("operation must be non-empty")

    fields: dict[str, tuple[Any, Any]] = {
        "configuration": (
            candidate_model,
            Field(description="The complete proposed candidate configuration."),
        ),
        "design_rationale": (
            _Rationale,
            Field(description="Why this design should improve the stated objectives."),
        ),
        "intended_changes": (
            list[_Change],
            Field(default_factory=list, max_length=MAX_INTENDED_CHANGES),
        ),
        "source_attribution": (
            list[_SourceAttributionOutput],
            Field(default_factory=list, max_length=MAX_SOURCE_ATTRIBUTIONS),
        ),
        "claimed_insight_ids": (
            list[_Identifier],
            Field(default_factory=list, max_length=MAX_CLAIMED_INSIGHT_IDS),
        ),
    }
    # Strict three-way recombination is the only current operator with opaque
    # preservation receipts or conflict-resolution annotations.  Omitting these
    # fields from mutation/crossover schemas prevents irrelevant, token-heavy
    # explanations from becoming an accidental second task.
    if operation not in {
        "typed_mutation",
        "two_parent_crossover",
        "three_way_recombination",
        "repair",
    }:
        fields["claimed_preservation_obligation_ids"] = (
            list[_Identifier],
            Field(default_factory=list, max_length=MAX_PRESERVATION_IDS),
        )
        fields["conflict_resolutions"] = (
            list[_ConflictResolutionOutput],
            Field(default_factory=list, max_length=MAX_CONFLICT_RESOLUTIONS),
        )
    elif operation == "three_way_recombination":
        fields["conflict_resolutions"] = (
            list[_ConflictResolutionOutput],
            Field(default_factory=list, max_length=MAX_CONFLICT_RESOLUTIONS),
        )
    return create_model(
        "CandidateProposal",
        __base__=_CompactCandidateSchemaBase,
        __module__=__name__,
        **fields,
    )


def _atomic_mutation_proposal_type(
    candidate_model: type[BaseModel],
    operation: str,
    contract: AtomicMutationOutputContract,
) -> type[BaseModel]:
    if not isinstance(candidate_model, type) or not issubclass(
        candidate_model, BaseModel
    ):
        raise TypeError("candidate_model must be a Pydantic BaseModel subclass")
    if candidate_model is BaseModel or getattr(
        candidate_model, "__pydantic_root_model__", False
    ):
        raise TypeError("candidate_model must describe an object configuration")
    if getattr(candidate_model, "__parameters__", ()):
        raise TypeError("candidate_model must be a concrete Pydantic model")
    if operation != "typed_mutation":
        raise ValueError("atomic mutation output is restricted to typed_mutation")
    if type(contract) is not AtomicMutationOutputContract:
        raise TypeError("contract must be an exact AtomicMutationOutputContract")
    AtomicMutationOutputContract.__post_init__(contract)

    path_text = _path_text(contract.editable_path)
    path_literal = Literal.__getitem__((path_text,))
    output_type = create_model(
        "AtomicMutationProposal",
        __base__=_CompactAtomicMutationSchemaBase,
        __module__=__name__,
        path=(path_literal, Field(description="The exact contracted JSON path.")),
        replacement=(
            _AtomicScalar,
            Field(description="The replacement scalar at the contracted path."),
        ),
        design_rationale=(
            _Rationale,
            Field(description="Why this one edit should improve the objectives."),
        ),
        claimed_insight_ids=(
            list[_Identifier],
            Field(default_factory=list, max_length=MAX_CLAIMED_INSIGHT_IDS),
        ),
    )
    output_type.expected_path_text = path_text
    leaf_schema = _atomic_leaf_schema(
        candidate_model,
        contract.editable_path,
    )
    current = value_at_path(
        contract.parent_configuration,
        contract.editable_path,
    )
    output_type.replacement_schema = _exclude_current_scalar(
        leaf_schema,
        current,
    )
    if contract.replacement_options:
        for option in contract.replacement_options:
            target = replace_existing_path(
                contract.parent_configuration,
                contract.editable_path,
                option,
            )
            validated = candidate_model.model_validate(
                thaw_json(target),
                strict=True,
                by_alias=False,
                by_name=True,
            )
            if not typed_json_equal(
                freeze_json(_configuration_dict(validated)),
                target,
            ):
                raise ValueError(
                    "candidate validation changed an atomic replacement option"
                )
        output_type.replacement_schema = _restrict_scalar_options(
            output_type.replacement_schema,
            contract.replacement_options,
        )
    output_type.candidate_model = candidate_model
    output_type.output_contract = contract
    return output_type


def _finite_variation_selection_type(
    candidate_model: type[BaseModel],
    operation: str,
    contract: FiniteVariationContract,
) -> type[BaseModel]:
    """Build one strict Literal-ID tool schema over prevalidated full children."""

    if not isinstance(candidate_model, type) or not issubclass(
        candidate_model, BaseModel
    ):
        raise TypeError("candidate_model must be a Pydantic BaseModel subclass")
    if candidate_model is BaseModel or getattr(
        candidate_model, "__pydantic_root_model__", False
    ):
        raise TypeError("candidate_model must describe an object configuration")
    if getattr(candidate_model, "__parameters__", ()):
        raise TypeError("candidate_model must be a concrete Pydantic model")
    if type(operation) is not str or not operation.strip():
        raise ValueError("operation must be non-empty")
    validate_finite_variation_contract(contract)

    for option in contract.options:
        validated = candidate_model.model_validate(
            thaw_json(option.child_configuration),
            strict=True,
            by_alias=False,
            by_name=True,
        )
        if not typed_json_equal(
            freeze_json(_configuration_dict(validated)),
            option.child_configuration,
        ):
            raise ValueError(
                "candidate validation changed a finite variation child"
            )

    option_ids = tuple(option.option_id for option in contract.options)
    option_literal = Literal.__getitem__(option_ids)
    output_type = create_model(
        "FiniteVariationSelectionProposal",
        __config__=_STRICT_MODEL_CONFIG,
        __module__=__name__,
        option_id=(
            option_literal,
            Field(description="The selected immutable variation option ID."),
        ),
        design_rationale=(
            _Rationale,
            Field(description="Why this sealed option should improve the objectives."),
        ),
        claimed_insight_ids=(
            list[_Identifier],
            Field(default_factory=list, max_length=MAX_CLAIMED_INSIGHT_IDS),
        ),
    )
    return output_type


def _reflection_output_type(
    max_insights: int,
    available_contrast_ids: tuple[str, ...] = (),
    insight_contract: ReflectionInsightContract | None = None,
    *,
    min_insights: int = 0,
) -> type[BaseModel]:
    if type(max_insights) is not int or not 1 <= max_insights <= 16:
        raise ValueError("max_insights must lie in [1,16]")
    if type(min_insights) is not int or not 0 <= min_insights <= max_insights:
        raise ValueError("min_insights must lie in [0,max_insights]")
    if type(available_contrast_ids) is not tuple:
        raise TypeError("available_contrast_ids must be an exact tuple")
    if available_contrast_ids:
        allowed_contrast_id = Literal.__getitem__(available_contrast_ids)
        contrast_field = (
            list[allowed_contrast_id],
            Field(
                min_length=1,
                max_length=min(
                    MAX_EVIDENCE_CONTRAST_IDS,
                    len(available_contrast_ids),
                ),
            ),
        )
    else:
        contrast_field = (
            list[_ContrastIdentifier],
            Field(max_length=0),
        )
    if insight_contract is None:
        insight_type = create_model(
            "ReflectionInsightOutput",
            __base__=_ReflectionInsightOutput,
            __module__=__name__,
            evidence_contrast_ids=contrast_field,
        )
    else:
        if type(insight_contract) is not ReflectionInsightContract:
            raise TypeError(
                "insight_contract must be an exact ReflectionInsightContract"
            )
        ReflectionInsightContract.__post_init__(insight_contract)
        metric_id_literal = Literal.__getitem__(
            insight_contract.required_metric_ids
        )
        option_family_literal = Literal.__getitem__(
            insight_contract.allowed_option_families
        )
        metric_prediction_type = create_model(
            "MetricEffectPredictionOutput",
            __base__=_MetricEffectPredictionOutput,
            __module__=__name__,
            metric_id=(metric_id_literal, ...),
        )
        insight_type = create_model(
            "InterventionInsightOutput",
            __base__=_AdvancedReflectionInsightOutput,
            __module__=__name__,
            evidence_contrast_ids=contrast_field,
            effect_predictions=(
                list[metric_prediction_type],
                Field(
                    min_length=len(insight_contract.required_metric_ids),
                    max_length=len(insight_contract.required_metric_ids),
                ),
            ),
            recommended_option_families=(
                list[option_family_literal],
                Field(
                    min_length=1,
                    max_length=len(insight_contract.allowed_option_families),
                ),
            ),
            **(
                {}
                if not insight_contract.allowed_option_ids
                else {
                    "recommended_option_ids": (
                        list[
                            Literal.__getitem__(
                                insight_contract.allowed_option_ids
                            )
                        ],
                        Field(
                            min_length=1,
                            max_length=len(
                                insight_contract.allowed_option_ids
                            ),
                        ),
                    )
                }
            ),
        )
        insight_type.required_metric_ids = insight_contract.required_metric_ids
        insight_type.allowed_option_families = (
            insight_contract.allowed_option_families
        )
        insight_type.allowed_option_ids = insight_contract.allowed_option_ids
    insights_field = (
        Field(default_factory=list, max_length=max_insights)
        if min_insights == 0
        else Field(
            min_length=min_insights,
            max_length=max_insights,
        )
    )
    output_type = create_model(
        "ReflectionOutput",
        __base__=_CompactReflectionSchemaBase,
        __module__=__name__,
        insights=(
            list[insight_type],
            insights_field,
        ),
    )
    output_type.available_contrast_ids = available_contrast_ids
    output_type.insight_contract = insight_contract
    output_type.min_insights = min_insights
    output_type.max_insights = max_insights
    return output_type


def _validated_response(
    result: object,
    *,
    output_type: type[BaseModel],
) -> tuple[StructuredGenerationResponse[Any], int]:
    if type(result) is AttemptedStructuredGenerationResponse:
        AttemptedStructuredGenerationResponse.__post_init__(result)
        response = result.response
        attempt_count = result.attempt_count
    elif type(result) is StructuredGenerationResponse:
        response = result
        attempt_count = 1
    else:
        raise TypeError(
            "low-level runner must return StructuredGenerationResponse or "
            "AttemptedStructuredGenerationResponse"
        )

    StructuredGenerationResponse.__post_init__(response)
    if type(response.value) is not output_type:
        raise TypeError("low-level response value does not match its requested output type")
    return response, attempt_count


def _telemetry(
    response: StructuredGenerationResponse[Any],
    *,
    attempt_count: int,
) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model=response.requested_model,
        resolved_model=response.resolved_model,
        resolved_provider=response.resolved_provider,
        provider_response_id=response.provider_response_id,
        finish_reason=response.finish_reason,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        reasoning_tokens=response.reasoning_tokens,
        cache_read_tokens=response.cache_read_tokens,
        cache_write_tokens=response.cache_write_tokens,
        cost_usd=response.cost_usd,
        latency_ns=response.latency_ns,
        attempt_count=attempt_count,
    )


def _configuration_dict(configuration: BaseModel) -> dict[str, Any]:
    # Invoke BaseModel's implementation directly so a candidate cannot replace
    # this trust-boundary operation with an overriding method.
    value = BaseModel.model_dump(
        configuration,
        mode="python",
        by_alias=False,
        exclude_unset=False,
        exclude_defaults=False,
        exclude_none=False,
        exclude_computed_fields=True,
        round_trip=True,
        warnings="error",
        fallback=None,
        serialize_as_any=False,
        polymorphic_serialization=False,
    )
    if type(value) is not dict:  # pragma: no cover - guarded by model admission.
        raise TypeError("candidate configuration must serialize to an exact dict")
    return value


class PydanticAIAgenticGenerator:
    """Map strict Pydantic outputs from an injected async runner into port values."""

    def __init__(self, generate_once: LowLevelRunner) -> None:
        if not callable(generate_once):
            raise TypeError("generate_once must be callable")
        self._generate_once = generate_once

    async def propose(
        self,
        request: VariationGenerationRequest,
    ) -> VariationGenerationResult:
        if type(request) is not VariationGenerationRequest:
            raise TypeError("request must be an exact VariationGenerationRequest")
        VariationGenerationRequest.__post_init__(request)

        atomic_contract = request.atomic_mutation_contract
        finite_contract = request.finite_variation_contract
        if atomic_contract is None and finite_contract is None:
            output_type = _candidate_proposal_type(
                request.candidate_model,
                request.operation,
            )
            output_tool_name = CANDIDATE_PROPOSAL_TOOL_NAME
        elif atomic_contract is not None:
            output_type = _atomic_mutation_proposal_type(
                request.candidate_model,
                request.operation,
                atomic_contract,
            )
            output_tool_name = ATOMIC_MUTATION_TOOL_NAME
        else:
            assert finite_contract is not None
            output_type = _finite_variation_selection_type(
                request.candidate_model,
                request.operation,
                finite_contract,
            )
            output_tool_name = FINITE_VARIATION_SELECTION_TOOL_NAME
        low_level_request = StructuredGenerationRequest(
            call_id=request.call_id,
            operation=request.operation,
            prompt=request.prompt,
            output_type=output_type,
            output_tool_name=output_tool_name,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
        )
        # Deliberately no exception handler: scheduler/provider errors retain
        # their original type, identity, and retry classification.
        low_level_result = await self._generate_once(low_level_request)
        response, attempt_count = _validated_response(
            low_level_result,
            output_type=output_type,
        )
        proposal = cast(Any, response.value)
        if atomic_contract is not None:
            replacement = freeze_json(proposal.replacement)
            draft: (
                CandidateDraft
                | AtomicMutationDraft
                | FiniteVariationSelectionDraft
            ) = AtomicMutationDraft(
                path=atomic_contract.editable_path,
                replacement=replacement,
                design_rationale=proposal.design_rationale,
                claimed_insight_ids=tuple(proposal.claimed_insight_ids),
            )
        elif finite_contract is not None:
            option = finite_contract.resolve(proposal.option_id)
            draft = FiniteVariationSelectionDraft(
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                contract_identity_sha256=finite_contract.identity_sha256,
                design_rationale=proposal.design_rationale,
                claimed_insight_ids=tuple(proposal.claimed_insight_ids),
            )
        else:
            draft = CandidateDraft(
                configuration=_configuration_dict(proposal.configuration),
                design_rationale=proposal.design_rationale,
                intended_changes=tuple(proposal.intended_changes),
                source_attribution=tuple(
                    SourceAttribution(path=item.path, source=item.source)
                    for item in proposal.source_attribution
                ),
                claimed_insight_ids=tuple(proposal.claimed_insight_ids),
                claimed_preservation_obligation_ids=tuple(
                    getattr(proposal, "claimed_preservation_obligation_ids", ())
                ),
                conflict_resolutions=tuple(
                    ConflictResolutionDraft(
                        relation_id=item.relation_id,
                        choice=item.choice,
                        explanation=item.explanation,
                    )
                    for item in getattr(proposal, "conflict_resolutions", ())
                ),
            )
        return VariationGenerationResult(
            draft=draft,
            telemetry=_telemetry(response, attempt_count=attempt_count),
        )

    async def reflect(
        self,
        request: ReflectionGenerationRequest,
    ) -> ReflectionGenerationResult:
        if type(request) is not ReflectionGenerationRequest:
            raise TypeError("request must be an exact ReflectionGenerationRequest")
        ReflectionGenerationRequest.__post_init__(request)

        output_type = _reflection_output_type(
            request.max_insights,
            request.available_contrast_ids,
            request.insight_contract,
            min_insights=request.min_insights,
        )
        low_level_request = StructuredGenerationRequest(
            call_id=request.call_id,
            operation=request.operation,
            prompt=request.prompt,
            output_type=output_type,
            output_tool_name=REFLECTION_TOOL_NAME,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
        )
        low_level_result = await self._generate_once(low_level_request)
        response, attempt_count = _validated_response(
            low_level_result,
            output_type=output_type,
        )
        reflection = cast(Any, response.value)
        insights = tuple(
            InsightDraft(
                claim=item.claim,
                trigger=item.trigger,
                mechanism=item.mechanism,
                affected_paths=tuple(item.affected_paths),
                evidence_summary=item.evidence_summary,
                confidence=float(item.confidence),
                evidence_contrast_ids=tuple(sorted(item.evidence_contrast_ids)),
                effect_predictions=tuple(
                    sorted(
                        (
                            MetricEffectPrediction(
                                metric_id=prediction.metric_id,
                                direction=MetricEffectDirection(
                                    prediction.direction
                                ),
                            )
                            for prediction in getattr(
                                item, "effect_predictions", ()
                            )
                        ),
                        key=lambda prediction: prediction.metric_id,
                    )
                ),
                recommended_option_families=tuple(
                    sorted(
                        getattr(item, "recommended_option_families", ())
                    )
                ),
                recommended_option_ids=tuple(
                    sorted(getattr(item, "recommended_option_ids", ()))
                ),
                action_template=getattr(item, "action_template", None),
                falsification_condition=getattr(
                    item, "falsification_condition", None
                ),
            )
            for item in reflection.insights
        )
        return ReflectionGenerationResult(
            insights=insights,
            telemetry=_telemetry(response, attempt_count=attempt_count),
        )


__all__ = [
    "ATOMIC_MUTATION_TOOL_NAME",
    "AttemptedStructuredGenerationResponse",
    "CANDIDATE_PROPOSAL_TOOL_NAME",
    "FINITE_VARIATION_SELECTION_TOOL_NAME",
    "LowLevelRunner",
    "PydanticAIAgenticGenerator",
    "REFLECTION_TOOL_NAME",
]
