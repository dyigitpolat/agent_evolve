"""Provider-free codec for the pinned Pydantic AI ``Model.request`` seam.

The production route is deliberately exact: Pydantic AI's OpenAI chat adapter,
an OpenRouter provider, and the complete OpenAI profile dataclass at the pinned
dependency versions.  A separate FunctionModel route exists only for offline
tests and retains framework timestamps and IDs rather than making a generic
claim that they are model-irrelevant.

This module constructs no provider/client and performs no I/O.
"""

from __future__ import annotations

import hashlib
import math
import types
from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Literal,
    Union,
    get_args,
    get_origin,
    is_typeddict,
)

import annotated_types
import pydantic
import pydantic_ai
import pydantic_core
import pydantic.types
from pydantic import BaseModel, TypeAdapter
from pydantic._internal._fields import _general_metadata_cls
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai._output import OutputSchema
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import (
    OpenAIJsonSchemaTransformer,
    OpenAIModelProfile,
)
from pydantic_ai.providers.openrouter import OpenRouterModelProfile, OpenRouterProvider
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition

from agent_evolve.ports.artifact_store import canonical_json_bytes, decode_json_bytes

REQUEST_FORMAT = "agent-evolve.pydantic-ai-request"
RESPONSE_FORMAT = "agent-evolve.pydantic-ai-response"
OUTPUT_CONTRACT_FORMAT = "agent-evolve.pydantic-ai-output-contract"
BOUNDARY_FORMAT_VERSION = 2
OUTPUT_CONTRACT_FORMAT_VERSION = 3
SUPPORTED_PYDANTIC_AI_VERSION = "1.107.1"
SUPPORTED_PYDANTIC_VERSION = "2.13.4"
SUPPORTED_PYDANTIC_CORE_VERSION = "2.46.4"
BOUNDARY_NAME = "model.request.pre-provider-prepare"

_PRODUCTION_ROUTE = "openrouter-openai-chat-v1"
_FIXTURE_ROUTE = "offline-function-fixture-v1"
_PRODUCTION_OMISSIONS = (
    "messages[*].timestamp",
    "messages[*].run_id",
    "messages[*].conversation_id",
    "messages[*].parts[*].timestamp",
)
_FAILED = object()
_OUTPUT_BUILD_TOKEN = object()
_SKIP_CONSTRUCTION_TOKEN_CHECK = object()

_MODEL_PROFILE_FIELDS = frozenset(
    {
        "supports_tools",
        "supports_tool_return_schema",
        "supports_json_schema_output",
        "supports_json_object_output",
        "supports_image_output",
        "supports_inline_system_prompts",
        "default_structured_output_mode",
        "prompted_output_template",
        "native_output_requires_schema_in_instructions",
        "json_schema_transformer",
        "supports_thinking",
        "thinking_always_enabled",
        "thinking_tags",
        "ignore_streamed_leading_whitespace",
        "supported_native_tools",
    }
)
_OPENAI_PROFILE_EXTRA_FIELDS = frozenset(
    {
        "openai_chat_thinking_field",
        "openai_chat_send_back_thinking_parts",
        "openai_supports_strict_tool_definition",
        "openai_supports_sampling_settings",
        "openai_unsupported_model_settings",
        "openai_supports_tool_choice_required",
        "openai_system_prompt_role",
        "openai_chat_supports_multiple_system_messages",
        "openai_chat_supports_web_search",
        "openai_chat_audio_input_encoding",
        "openai_chat_supports_file_urls",
        "openai_supports_encrypted_reasoning_content",
        "openai_supports_reasoning",
        "openai_supports_reasoning_effort_none",
        "openai_responses_requires_function_call_status_none",
        "openai_supports_phase",
        "openai_chat_supports_document_input",
    }
)
_OPENROUTER_PROFILE_EXTRA_FIELDS = frozenset(
    {
        "openrouter_supports_cache_control",
        "openrouter_supports_cache_ttl",
        "openrouter_supports_tool_cache",
        "openrouter_supports_dynamic_instruction_cache",
        "openrouter_max_cache_points",
    }
)
_MODEL_REQUEST_FIELDS = frozenset(
    {"parts", "timestamp", "instructions", "kind", "run_id", "conversation_id", "metadata"}
)
_USER_PROMPT_FIELDS = frozenset({"content", "timestamp", "part_kind"})
_REQUEST_PARAMETER_FIELDS = frozenset(
    {
        "function_tools",
        "native_tools",
        "output_mode",
        "output_object",
        "output_tools",
        "prompted_output_template",
        "allow_text_output",
        "allow_image_output",
        "instruction_parts",
        "thinking",
    }
)
_OUTPUT_OBJECT_FIELDS = frozenset({"json_schema", "name", "description", "strict"})
_TOOL_DEFINITION_FIELDS = frozenset(
    {
        "name",
        "parameters_json_schema",
        "description",
        "outer_typed_dict_key",
        "strict",
        "sequential",
        "kind",
        "metadata",
        "timeout",
        "defer_loading",
        "unless_native",
        "with_native",
        "tool_kind",
        "return_schema",
        "include_return_schema",
        "capability_id",
    }
)
_REQUEST_KEYS = frozenset(
    {
        "format",
        "format_version",
        "boundary",
        "semantic_route",
        "pydantic_ai_version",
        "pydantic_version",
        "pydantic_core_version",
        "model_name",
        "model_system",
        "model_adapter_type",
        "model_profile_type",
        "provider_type",
        "model_profile",
        "model_profile_sha256",
        "model_default_settings",
        "messages",
        "model_settings",
        "model_request_parameters",
        "output_type_contract",
        "output_type_contract_sha256",
        "omitted_framework_fields",
    }
)
_RESPONSE_KEYS = frozenset(
    {
        "format",
        "format_version",
        "pydantic_ai_version",
        "pydantic_version",
        "pydantic_core_version",
        "response",
    }
)
_OUTPUT_CONTRACT_KEYS = frozenset(
    {
        "format",
        "format_version",
        "pydantic_ai_version",
        "pydantic_version",
        "pydantic_core_version",
        "output_kind",
        "type_graph",
        "json_schema",
        "request_shape",
    }
)
_RESPONSE_ADAPTER = TypeAdapter(ModelResponse)


class PydanticAIBoundaryCodecError(RuntimeError):
    """The value is unsupported, unsafe to decode, or not replay-equivalent."""


def _error() -> PydanticAIBoundaryCodecError:
    # Never retain caller/provider values or lower-level exception context.
    return PydanticAIBoundaryCodecError(
        "Pydantic AI boundary value does not satisfy the pinned replay contract"
    )


def _qualified_type_name(value: type[Any]) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if type(module) is not str or not module or type(qualname) is not str or not qualname:
        raise _error()
    return f"{module}.{qualname}"


def _require_supported_versions() -> tuple[str, str, str]:
    versions = (
        pydantic_ai.__version__,
        pydantic.__version__,
        pydantic_core.__version__,
    )
    if versions != (
        SUPPORTED_PYDANTIC_AI_VERSION,
        SUPPORTED_PYDANTIC_VERSION,
        SUPPORTED_PYDANTIC_CORE_VERSION,
    ):
        raise _error()
    return versions


def _strict_json_clone(value: Any, *, ancestors: set[int] | None = None) -> Any:
    """Clone the exact JSON runtime domain without Pydantic coercion."""

    value_type = type(value)
    if value is None or value_type is bool or value_type is int:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise _error()
        return value
    if value_type is str:
        value.encode("utf-8", errors="strict")
        return value
    if value_type is list:
        ancestors = set() if ancestors is None else ancestors
        identity = id(value)
        if identity in ancestors:
            raise _error()
        ancestors.add(identity)
        try:
            return [_strict_json_clone(item, ancestors=ancestors) for item in value]
        finally:
            ancestors.remove(identity)
    if value_type is dict:
        ancestors = set() if ancestors is None else ancestors
        identity = id(value)
        if identity in ancestors:
            raise _error()
        ancestors.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise _error()
                key.encode("utf-8", errors="strict")
                result[key] = _strict_json_clone(item, ancestors=ancestors)
            return result
        finally:
            ancestors.remove(identity)
    raise _error()


def _json_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise _error()
    cloned = _strict_json_clone(value)
    if type(cloned) is not dict:
        raise _error()
    canonical_json_bytes(cloned)
    return cloned


def _exact_optional_string(value: Any, *, allow_empty: bool = True) -> str | None:
    if value is None:
        return None
    if type(value) is not str or (not allow_empty and not value):
        raise _error()
    value.encode("utf-8", errors="strict")
    return value


def _datetime_json(value: Any) -> str | None:
    if value is None:
        return None
    if type(value) is not datetime or value.tzinfo is not timezone.utc:
        raise _error()
    rendered = value.isoformat()
    if rendered.endswith("+00:00"):
        rendered = f"{rendered[:-6]}Z"
    return rendered


def _field_names(value: Any) -> frozenset[str]:
    return frozenset(field.name for field in fields(value))


def _common_profile_record(profile: ModelProfile) -> dict[str, Any]:
    boolean_fields = (
        "supports_tools",
        "supports_tool_return_schema",
        "supports_json_schema_output",
        "supports_json_object_output",
        "supports_image_output",
        "supports_inline_system_prompts",
        "native_output_requires_schema_in_instructions",
        "supports_thinking",
        "thinking_always_enabled",
        "ignore_streamed_leading_whitespace",
    )
    if any(type(getattr(profile, name)) is not bool for name in boolean_fields):
        raise _error()
    if (
        type(profile.default_structured_output_mode) is not str
        or profile.default_structured_output_mode not in ("tool", "native", "prompted")
        or type(profile.prompted_output_template) is not str
        or type(profile.thinking_tags) is not tuple
        or len(profile.thinking_tags) != 2
        or any(type(item) is not str for item in profile.thinking_tags)
        or type(profile.supported_native_tools) is not frozenset
        or any(not isinstance(item, type) for item in profile.supported_native_tools)
    ):
        raise _error()
    transformer = profile.json_schema_transformer
    if transformer is not None and not isinstance(transformer, type):
        raise _error()
    return {
        "supports_tools": profile.supports_tools,
        "supports_tool_return_schema": profile.supports_tool_return_schema,
        "supports_json_schema_output": profile.supports_json_schema_output,
        "supports_json_object_output": profile.supports_json_object_output,
        "supports_image_output": profile.supports_image_output,
        "supports_inline_system_prompts": profile.supports_inline_system_prompts,
        "default_structured_output_mode": profile.default_structured_output_mode,
        "prompted_output_template": profile.prompted_output_template,
        "native_output_requires_schema_in_instructions": (
            profile.native_output_requires_schema_in_instructions
        ),
        "json_schema_transformer": (
            None if transformer is None else _qualified_type_name(transformer)
        ),
        "supports_thinking": profile.supports_thinking,
        "thinking_always_enabled": profile.thinking_always_enabled,
        "thinking_tags": list(profile.thinking_tags),
        "ignore_streamed_leading_whitespace": profile.ignore_streamed_leading_whitespace,
        "supported_native_tools": sorted(
            _qualified_type_name(item) for item in profile.supported_native_tools
        ),
    }


def _openai_profile_record(profile: OpenRouterModelProfile) -> dict[str, Any]:
    if type(profile) is not OpenRouterModelProfile:
        raise _error()
    if _field_names(profile) != (
        _MODEL_PROFILE_FIELDS
        | _OPENAI_PROFILE_EXTRA_FIELDS
        | _OPENROUTER_PROFILE_EXTRA_FIELDS
    ):
        raise _error()
    if profile.json_schema_transformer is not OpenAIJsonSchemaTransformer:
        raise _error()
    boolean_fields = (
        "openai_supports_strict_tool_definition",
        "openai_supports_sampling_settings",
        "openai_supports_tool_choice_required",
        "openai_chat_supports_web_search",
        "openai_chat_supports_multiple_system_messages",
        "openai_chat_supports_file_urls",
        "openai_supports_encrypted_reasoning_content",
        "openai_supports_reasoning",
        "openai_supports_reasoning_effort_none",
        "openai_responses_requires_function_call_status_none",
        "openai_supports_phase",
        "openai_chat_supports_document_input",
    )
    if any(type(getattr(profile, name)) is not bool for name in boolean_fields):
        raise _error()
    send_back = profile.openai_chat_send_back_thinking_parts
    if not (
        (type(send_back) is str and send_back in ("auto", "tags", "field"))
        or (type(send_back) is bool and send_back is False)
    ):
        raise _error()
    if (
        profile.openai_chat_thinking_field is not None
        and type(profile.openai_chat_thinking_field) is not str
    ):
        raise _error()
    if (
        type(profile.openai_unsupported_model_settings) is not tuple
        or any(type(item) is not str for item in profile.openai_unsupported_model_settings)
        or (
            profile.openai_system_prompt_role is not None
            and (
                type(profile.openai_system_prompt_role) is not str
                or profile.openai_system_prompt_role not in ("system", "developer", "user")
            )
        )
        or type(profile.openai_chat_audio_input_encoding) is not str
        or profile.openai_chat_audio_input_encoding not in ("base64", "uri")
    ):
        raise _error()
    result = _common_profile_record(profile)
    result.update(
        {
            "openai_chat_thinking_field": profile.openai_chat_thinking_field,
            "openai_chat_send_back_thinking_parts": send_back,
            "openai_supports_strict_tool_definition": (
                profile.openai_supports_strict_tool_definition
            ),
            "openai_supports_sampling_settings": profile.openai_supports_sampling_settings,
            "openai_unsupported_model_settings": list(
                profile.openai_unsupported_model_settings
            ),
            "openai_supports_tool_choice_required": profile.openai_supports_tool_choice_required,
            "openai_system_prompt_role": profile.openai_system_prompt_role,
            "openai_chat_supports_multiple_system_messages": (
                profile.openai_chat_supports_multiple_system_messages
            ),
            "openai_chat_supports_web_search": profile.openai_chat_supports_web_search,
            "openai_chat_audio_input_encoding": profile.openai_chat_audio_input_encoding,
            "openai_chat_supports_file_urls": profile.openai_chat_supports_file_urls,
            "openai_supports_encrypted_reasoning_content": (
                profile.openai_supports_encrypted_reasoning_content
            ),
            "openai_supports_reasoning": profile.openai_supports_reasoning,
            "openai_supports_reasoning_effort_none": profile.openai_supports_reasoning_effort_none,
            "openai_responses_requires_function_call_status_none": (
                profile.openai_responses_requires_function_call_status_none
            ),
            "openai_supports_phase": profile.openai_supports_phase,
            "openai_chat_supports_document_input": profile.openai_chat_supports_document_input,
            "openrouter_supports_cache_control": (
                profile.openrouter_supports_cache_control
            ),
            "openrouter_supports_cache_ttl": profile.openrouter_supports_cache_ttl,
            "openrouter_supports_tool_cache": profile.openrouter_supports_tool_cache,
            "openrouter_supports_dynamic_instruction_cache": (
                profile.openrouter_supports_dynamic_instruction_cache
            ),
            "openrouter_max_cache_points": profile.openrouter_max_cache_points,
        }
    )
    openrouter_boolean_fields = (
        "openrouter_supports_cache_control",
        "openrouter_supports_cache_ttl",
        "openrouter_supports_tool_cache",
        "openrouter_supports_dynamic_instruction_cache",
    )
    if any(
        type(getattr(profile, name)) is not bool
        for name in openrouter_boolean_fields
    ):
        raise _error()
    if profile.openrouter_max_cache_points is not None and (
        type(profile.openrouter_max_cache_points) is not int
        or profile.openrouter_max_cache_points <= 0
    ):
        raise _error()
    return result


def _fixture_profile_record(profile: ModelProfile) -> dict[str, Any]:
    if type(profile) is not ModelProfile or _field_names(profile) != _MODEL_PROFILE_FIELDS:
        raise _error()
    # A custom transformer could inspect omitted/model-specific state.  The
    # fixture route is intentionally inert and closed to that behavior.
    if profile.json_schema_transformer is not None:
        raise _error()
    return _common_profile_record(profile)


@dataclass(frozen=True, slots=True)
class _ModelIdentity:
    route: str
    model_name: str
    model_system: str
    model_adapter_type: str
    model_profile_type: str
    provider_type: str | None
    profile_record: dict[str, Any]
    omissions: tuple[str, ...]


def _model_identity(model: Model) -> _ModelIdentity:
    if type(model) is OpenAIChatModel:
        provider = model.provider
        profile = model.profile
        if (
            type(provider) is not OpenRouterProvider
            or type(profile) is not OpenRouterModelProfile
            or type(model.model_name) is not str
            or not model.model_name
            or type(model.system) is not str
            or model.system != "openrouter"
        ):
            raise _error()
        return _ModelIdentity(
            route=_PRODUCTION_ROUTE,
            model_name=model.model_name,
            model_system=model.system,
            model_adapter_type=_qualified_type_name(type(model)),
            model_profile_type=_qualified_type_name(type(profile)),
            provider_type=_qualified_type_name(type(provider)),
            profile_record=_openai_profile_record(profile),
            omissions=_PRODUCTION_OMISSIONS,
        )
    if type(model) is FunctionModel:
        profile = model.profile
        if (
            model.provider is not None
            or type(profile) is not ModelProfile
            or type(model.model_name) is not str
            or not model.model_name
            or type(model.system) is not str
            or model.system != "function"
        ):
            raise _error()
        return _ModelIdentity(
            route=_FIXTURE_ROUTE,
            model_name=model.model_name,
            model_system=model.system,
            model_adapter_type=_qualified_type_name(type(model)),
            model_profile_type=_qualified_type_name(type(profile)),
            provider_type=None,
            profile_record=_fixture_profile_record(profile),
            omissions=(),
        )
    raise _error()


def model_profile_sha256(model: Model) -> str:
    """Hash the complete closed profile for one supported semantic model route."""

    attempted = _attempt_model_profile_sha256(model)
    if type(attempted) is not str:
        raise _error() from None
    return attempted


def _attempt_model_profile_sha256(model: Model) -> str | object:
    try:
        _require_supported_versions()
        identity = _model_identity(model)
        return hashlib.sha256(canonical_json_bytes(identity.profile_record)).hexdigest()
    except Exception:
        return _FAILED


_TYPE_GRAPH_MAX_DEPTH = 64
_TYPE_GRAPH_MAX_NODES = 2048


def _declared_dataclass_types(module: Any) -> tuple[type[Any], ...]:
    """Capture exact, package-owned metadata classes at module import time."""

    result: list[type[Any]] = []
    for name in dir(module):
        value = getattr(module, name)
        if (
            isinstance(value, type)
            and value.__module__ == module.__name__
            and is_dataclass(value)
            and not any(value is existing for existing in result)
        ):
            result.append(value)
    return tuple(result)


def _identity_unique_types(*groups: tuple[type[Any], ...]) -> tuple[type[Any], ...]:
    result: list[type[Any]] = []
    for group in groups:
        for value in group:
            if not any(value is existing for existing in result):
                result.append(value)
    return tuple(result)


_TRUSTED_METADATA_TYPES = _identity_unique_types(
    _declared_dataclass_types(annotated_types),
    _declared_dataclass_types(pydantic.types),
)
_PYDANTIC_GENERAL_METADATA_TYPE = _general_metadata_cls()


@dataclass(slots=True)
class _TypeGraphState:
    active_model_ids: set[int]
    active_annotations: set[int]
    nodes: int = 0


def _is_deferred_marker(value: Any) -> bool:
    marker_types = (DeferredToolRequests, DeferredToolResults)
    if isinstance(value, type):
        try:
            return issubclass(value, marker_types)
        except TypeError:
            return False
    return isinstance(value, marker_types)


def _reject_nested_deferred(
    value: Any,
    *,
    ancestors: set[int] | None = None,
    depth: int = 0,
) -> None:
    """Reject deferred classes/instances inside exact built-in containers."""

    if depth > _TYPE_GRAPH_MAX_DEPTH or _is_deferred_marker(value):
        raise _error()
    value_type = type(value)
    if not any(
        value_type is container_type
        for container_type in (list, tuple, dict, set, frozenset)
    ):
        return
    ancestors = set() if ancestors is None else ancestors
    identity = id(value)
    if identity in ancestors:
        raise _error()
    ancestors.add(identity)
    try:
        items = value.items() if value_type is dict else enumerate(value)
        for key, item in items:
            if value_type is dict:
                _reject_nested_deferred(key, ancestors=ancestors, depth=depth + 1)
            _reject_nested_deferred(item, ancestors=ancestors, depth=depth + 1)
    finally:
        ancestors.remove(identity)


def _metadata_record(value: Any) -> dict[str, Any]:
    """Serialize only reviewed constraint metadata; reject opaque semantics."""

    _reject_nested_deferred(value)
    value_type = type(value)
    if (
        any(value_type is trusted for trusted in _TRUSTED_METADATA_TYPES)
        and is_dataclass(value)
        and not isinstance(value, type)
    ):
        attributes: dict[str, Any] = {}
        for item in fields(value):
            attribute = getattr(value, item.name)
            _reject_nested_deferred(attribute)
            attributes[item.name] = _strict_json_clone(attribute)
        return {
            "type": _qualified_type_name(value_type),
            "attributes": attributes,
        }

    # Pydantic creates this pinned private metadata carrier for constraints such
    # as regex patterns. It is not a dataclass, but its exact JSON-valued state
    # is closed and the Pydantic version is part of the contract.
    if (
        value_type is _PYDANTIC_GENERAL_METADATA_TYPE
        and type(getattr(value, "__dict__", None)) is dict
    ):
        attributes = _strict_json_clone(value.__dict__)
        return {
            "type": _qualified_type_name(value_type),
            "attributes": attributes,
        }
    raise _error()


def _type_graph(
    annotation: Any,
    *,
    state: _TypeGraphState | None = None,
    depth: int = 0,
) -> dict[str, Any]:
    state = state or _TypeGraphState(active_model_ids=set(), active_annotations=set())
    state.nodes += 1
    if depth > _TYPE_GRAPH_MAX_DEPTH or state.nodes > _TYPE_GRAPH_MAX_NODES:
        raise _error()
    if _is_deferred_marker(annotation):
        raise _error()
    if annotation is Any or isinstance(annotation, ForwardRef):
        # Opaque values and unresolved forward references are outside M1e.
        raise _error()
    if annotation is None or annotation is type(None):
        return {"kind": "type", "name": "builtins.NoneType"}
    if any(annotation is scalar for scalar in (str, int, float, bool, Decimal)):
        return {"kind": "type", "name": _qualified_type_name(annotation)}

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        annotation_id = id(annotation)
        if annotation_id in state.active_model_ids:
            return {"kind": "model-ref", "name": _qualified_type_name(annotation)}
        state.active_model_ids.add(annotation_id)
        try:
            if type(annotation.model_fields) is not dict:
                raise _error()
            model_fields: list[dict[str, Any]] = []
            for name in sorted(annotation.model_fields):
                if type(name) is not str:
                    raise _error()
                field = annotation.model_fields[name]
                metadata = getattr(field, "metadata", None)
                if type(metadata) is not list:
                    raise _error()
                default_factory = getattr(field, "default_factory", None)
                if default_factory is list or default_factory is dict:
                    initialization = {
                        "kind": "factory",
                        "name": _qualified_type_name(default_factory),
                    }
                elif default_factory is not None:
                    # Arbitrary factory code is behavioral state that neither
                    # the JSON schema nor a qualified name can bind safely.
                    raise _error()
                elif field.is_required():
                    initialization = {"kind": "required"}
                else:
                    default = getattr(field, "default", None)
                    _reject_nested_deferred(default)
                    initialization = {
                        "kind": "value",
                        "value": _strict_json_clone(default),
                    }
                for auxiliary in (
                    getattr(field, "json_schema_extra", None),
                    getattr(field, "examples", None),
                ):
                    _reject_nested_deferred(auxiliary)
                    if auxiliary is not None:
                        _strict_json_clone(auxiliary)
                model_fields.append(
                    {
                        "name": name,
                        "annotation": _type_graph(
                            field.annotation,
                            state=state,
                            depth=depth + 1,
                        ),
                        "initialization": initialization,
                        "metadata": [_metadata_record(item) for item in metadata],
                    }
                )
            return {
                "kind": "pydantic-model",
                "name": _qualified_type_name(annotation),
                "fields": model_fields,
            }
        finally:
            state.active_model_ids.remove(annotation_id)

    if is_typeddict(annotation):
        raise _error()
    if isinstance(annotation, type) and is_dataclass(annotation):
        raise _error()
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        members: list[dict[str, Any]] = []
        for member in annotation:
            _reject_nested_deferred(member.value)
            members.append(
                {"name": member.name, "value": _strict_json_clone(member.value)}
            )
        return {
            "kind": "enum",
            "name": _qualified_type_name(annotation),
            "members": members,
        }

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        # Arbitrary classes and opaque typing constructs are not represented by
        # a qualified name alone.
        raise _error()
    annotation_id = id(annotation)
    if annotation_id in state.active_annotations:
        raise _error()
    state.active_annotations.add(annotation_id)
    try:
        if origin is Literal:
            if not args:
                raise _error()
            values: list[Any] = []
            for item in args:
                _reject_nested_deferred(item)
                values.append(_strict_json_clone(item))
            return {"kind": "literal", "values": values}
        if origin is Union or origin is types.UnionType:
            if len(args) < 2:
                raise _error()
            return {
                "kind": "union",
                "args": [
                    _type_graph(item, state=state, depth=depth + 1) for item in args
                ],
            }
        if origin is Annotated:
            if len(args) < 2:
                raise _error()
            base, *metadata = args
            return {
                "kind": "annotated",
                "base": _type_graph(base, state=state, depth=depth + 1),
                "metadata": [_metadata_record(item) for item in metadata],
            }
        if origin is list:
            if len(args) != 1:
                raise _error()
            return {
                "kind": "list",
                "item": _type_graph(args[0], state=state, depth=depth + 1),
            }
        if origin is dict:
            if len(args) != 2 or args[0] is not str:
                raise _error()
            return {
                "kind": "dict",
                "key": {"kind": "type", "name": "builtins.str"},
                "value": _type_graph(args[1], state=state, depth=depth + 1),
            }
        if origin is tuple:
            # Variable tuples carry Ellipsis as an argument. M1e deliberately
            # rejects them rather than reaching Ellipsis incidentally.
            if not args or any(item is Ellipsis for item in args):
                raise _error()
            return {
                "kind": "tuple",
                "items": [
                    _type_graph(item, state=state, depth=depth + 1) for item in args
                ],
            }
        raise _error()
    finally:
        state.active_annotations.remove(annotation_id)


@dataclass(frozen=True, slots=True)
class _OutputContractPayload:
    canonical_bytes: bytes
    sha256_hex: str
    output_kind: str


@dataclass(frozen=True, slots=True, eq=False)
class BoundOutputContract:
    """Canonical provider-request contract bound to one exact runtime output type.

    The serialized bytes deliberately do not claim to identify user validator or
    hook code across processes.  Every consumer rebuilds them from ``output_type``;
    replay additionally supplies its independently expected runtime type.
    """

    output_type: Any = field(repr=False)
    canonical_bytes: bytes
    sha256_hex: str
    output_kind: str
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        attempted = _attempt_validate_bound_output_contract(
            self,
            construction_token=_construction_token,
        )
        if attempted is not None:
            # Raise only after the validation frame has discarded every
            # lower-level exception, including its message and traceback.
            raise _error() from None

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is BoundOutputContract
            and self.output_type is other.output_type
            and self.canonical_bytes == other.canonical_bytes
            and self.sha256_hex == other.sha256_hex
            and self.output_kind == other.output_kind
        )


def _attempt_validate_bound_output_contract(
    contract: BoundOutputContract,
    *,
    construction_token: object = _SKIP_CONSTRUCTION_TOKEN_CHECK,
) -> None | object:
    try:
        if (
            type(contract) is not BoundOutputContract
            or (
                construction_token is not _SKIP_CONSTRUCTION_TOKEN_CHECK
                and construction_token is not _OUTPUT_BUILD_TOKEN
            )
            or type(contract.canonical_bytes) is not bytes
            or type(contract.sha256_hex) is not str
            or type(contract.output_kind) is not str
        ):
            raise _error()
        expected = _attempt_build_output_contract_payload(contract.output_type)
        if (
            type(expected) is not _OutputContractPayload
            or contract.canonical_bytes != expected.canonical_bytes
            or contract.sha256_hex != expected.sha256_hex
            or contract.output_kind != expected.output_kind
        ):
            raise _error()
        return None
    except Exception:
        return _FAILED


def build_output_type_contract(output_type: Any) -> BoundOutputContract:
    """Build and bind the provider-free contract for one exact runtime type."""

    attempted = _attempt_build_output_contract_payload(output_type)
    if type(attempted) is not _OutputContractPayload:
        raise _error() from None
    return BoundOutputContract(
        output_type=output_type,
        canonical_bytes=attempted.canonical_bytes,
        sha256_hex=attempted.sha256_hex,
        output_kind=attempted.output_kind,
        _construction_token=_OUTPUT_BUILD_TOKEN,
    )


def _attempt_build_output_contract_payload(
    output_type: Any,
) -> _OutputContractPayload | object:
    try:
        pydantic_ai_version, pydantic_version, core_version = _require_supported_versions()
        origin = get_origin(output_type)
        output_args = get_args(output_type)
        if output_type is str:
            output_kind = "text"
        elif origin is list and len(output_args) == 1 and output_args[0] is str:
            output_kind = "structured"
        elif isinstance(output_type, type) and issubclass(output_type, BaseModel):
            output_kind = "structured"
        else:
            # This rejects list/union output specs and every marker workflow,
            # including DeferredToolRequests, before Agent construction.
            raise _error()
        graph = _type_graph(output_type)
        schema = _strict_json_clone(TypeAdapter(output_type).json_schema())
        if type(schema) is not dict:
            raise _error()
        output_schema = OutputSchema.build(output_type)
        if output_kind == "text":
            if (
                output_schema.mode != "text"
                or output_schema.object_def is not None
                or output_schema.toolset is not None
                or output_schema.allows_text is not True
                or output_schema.allows_image is not False
                or output_schema.allows_deferred_tools is not False
            ):
                raise _error()
            request_shape = {
                "output_mode": "text",
                "output_object": None,
                "output_tools": [],
                "allow_text_output": True,
            }
        else:
            toolset = output_schema.toolset
            tool_defs = getattr(toolset, "_tool_defs", None)
            if (
                output_schema.mode != "auto"
                or type(output_schema.object_def) is not OutputObjectDefinition
                or toolset is None
                or type(tool_defs) is not list
                or len(tool_defs) != 1
                or output_schema.allows_text is not True
                or output_schema.allows_image is not False
                or output_schema.allows_deferred_tools is not False
            ):
                raise _error()
            request_shape = {
                "output_mode": "auto",
                "output_object": _project_output_object(output_schema.object_def),
                "output_tools": [_project_output_tool(tool_defs[0])],
                "allow_text_output": True,
            }
        value = {
            "format": OUTPUT_CONTRACT_FORMAT,
            "format_version": OUTPUT_CONTRACT_FORMAT_VERSION,
            "pydantic_ai_version": pydantic_ai_version,
            "pydantic_version": pydantic_version,
            "pydantic_core_version": core_version,
            "output_kind": output_kind,
            "type_graph": graph,
            "json_schema": schema,
            "request_shape": request_shape,
        }
        content = canonical_json_bytes(value)
        return _OutputContractPayload(
            canonical_bytes=content,
            sha256_hex=hashlib.sha256(content).hexdigest(),
            output_kind=output_kind,
        )
    except Exception:
        return _FAILED


def validate_output_type_contract(
    contract: BoundOutputContract,
    *,
    compiled_json_schema: dict[str, Any] | None = None,
) -> None:
    """Rebuild and compare a contract, optionally binding Agent's compiled schema."""

    attempted = _attempt_validate_output_type_contract(
        contract,
        compiled_json_schema=compiled_json_schema,
    )
    if attempted is not None:
        raise _error() from None


def _attempt_validate_output_type_contract(
    contract: BoundOutputContract,
    *,
    compiled_json_schema: dict[str, Any] | None = None,
) -> None | object:
    try:
        if _attempt_validate_bound_output_contract(contract) is not None:
            raise _error()
        if compiled_json_schema is not None:
            compiled = _strict_json_clone(compiled_json_schema)
            value = decode_json_bytes(contract.canonical_bytes)
            if canonical_json_bytes(compiled) != canonical_json_bytes(
                value["json_schema"]
            ):
                raise _error()
        return None
    except Exception:
        return _FAILED


def bind_recorded_output_type_contract(
    expected_output_type: Any,
    *,
    canonical_bytes: bytes,
    sha256_hex: str,
) -> BoundOutputContract:
    """Bind recorded bytes only after rebuilding them from the replay type."""

    attempted = _attempt_bind_recorded_output_type_contract(
        expected_output_type,
        canonical_bytes=canonical_bytes,
        sha256_hex=sha256_hex,
    )
    if type(attempted) is not BoundOutputContract:
        raise _error() from None
    return attempted


def _attempt_bind_recorded_output_type_contract(
    expected_output_type: Any,
    *,
    canonical_bytes: bytes,
    sha256_hex: str,
) -> BoundOutputContract | object:
    try:
        if type(canonical_bytes) is not bytes or type(sha256_hex) is not str:
            raise _error()
        expected = build_output_type_contract(expected_output_type)
        if (
            canonical_bytes != expected.canonical_bytes
            or sha256_hex != expected.sha256_hex
            or hashlib.sha256(canonical_bytes).hexdigest() != sha256_hex
        ):
            raise _error()
        return expected
    except Exception:
        return _FAILED


def _project_messages(
    messages: list[Any], *, route: str
) -> list[dict[str, Any]]:
    if type(messages) is not list or len(messages) != 1:
        raise _error()
    message = messages[0]
    if (
        type(message) is not ModelRequest
        or _field_names(message) != _MODEL_REQUEST_FIELDS
        or type(message.parts) is not list
        or len(message.parts) != 1
        or type(message.kind) is not str
        or message.kind != "request"
    ):
        raise _error()
    part = message.parts[0]
    if (
        type(part) is not UserPromptPart
        or _field_names(part) != _USER_PROMPT_FIELDS
        or type(part.part_kind) is not str
        or part.part_kind != "user-prompt"
        or type(part.content) is not str
    ):
        raise _error()
    part.content.encode("utf-8", errors="strict")
    instructions = _exact_optional_string(message.instructions)
    metadata = _json_mapping(message.metadata)
    timestamp = _datetime_json(message.timestamp)
    part_timestamp = _datetime_json(part.timestamp)
    run_id = _exact_optional_string(message.run_id, allow_empty=False)
    conversation_id = _exact_optional_string(message.conversation_id, allow_empty=False)
    projected_message: dict[str, Any] = {
        "kind": "request",
        "parts": [{"part_kind": "user-prompt", "content": part.content}],
        "instructions": instructions,
        "metadata": metadata,
    }
    if route == _FIXTURE_ROUTE:
        projected_message.update(
            {
                "timestamp": timestamp,
                "run_id": run_id,
                "conversation_id": conversation_id,
            }
        )
        projected_message["parts"][0]["timestamp"] = part_timestamp
    elif route != _PRODUCTION_ROUTE:
        raise _error()
    return [projected_message]


def _project_output_object(value: Any) -> dict[str, Any]:
    if type(value) is not OutputObjectDefinition or _field_names(value) != _OUTPUT_OBJECT_FIELDS:
        raise _error()
    name = _exact_optional_string(value.name, allow_empty=False)
    description = _exact_optional_string(value.description)
    if value.strict is not None and type(value.strict) is not bool:
        raise _error()
    schema = _json_mapping(value.json_schema)
    if schema is None:
        raise _error()
    return {
        "json_schema": schema,
        "name": name,
        "description": description,
        "strict": value.strict,
    }


def _project_output_tool(value: Any) -> dict[str, Any]:
    if type(value) is not ToolDefinition or _field_names(value) != _TOOL_DEFINITION_FIELDS:
        raise _error()
    if (
        type(value.name) is not str
        or value.name != "final_result"
        or type(value.description) is not str
        or value.description != "The final response which ends this conversation"
        or (value.outer_typed_dict_key is not None and type(value.outer_typed_dict_key) is not str)
        or (value.strict is not None and type(value.strict) is not bool)
        or type(value.sequential) is not bool
        or value.sequential is not False
        or type(value.kind) is not str
        or value.kind != "output"
        or value.metadata is not None
        or value.timeout is not None
        or type(value.defer_loading) is not bool
        or value.defer_loading is not False
        or value.unless_native is not None
        or value.with_native is not None
        or value.tool_kind is not None
        or value.return_schema is not None
        or value.include_return_schema is not None
        or value.capability_id is not None
    ):
        raise _error()
    schema = _json_mapping(value.parameters_json_schema)
    if schema is None:
        raise _error()
    return {
        "name": value.name,
        "parameters_json_schema": schema,
        "description": value.description,
        "outer_typed_dict_key": value.outer_typed_dict_key,
        "strict": value.strict,
        "sequential": value.sequential,
        "kind": value.kind,
        "metadata": None,
        "timeout": None,
        "defer_loading": value.defer_loading,
        "unless_native": None,
        "with_native": None,
        "tool_kind": None,
        "return_schema": None,
        "include_return_schema": None,
        "capability_id": None,
    }


def _project_parameters(
    parameters: ModelRequestParameters,
    *,
    output_contract: BoundOutputContract,
) -> dict[str, Any]:
    if _attempt_validate_bound_output_contract(output_contract) is _FAILED:
        raise _error()
    if (
        type(parameters) is not ModelRequestParameters
        or _field_names(parameters) != _REQUEST_PARAMETER_FIELDS
        or type(parameters.function_tools) is not list
        or len(parameters.function_tools) != 0
        or type(parameters.native_tools) is not list
        or len(parameters.native_tools) != 0
        or type(parameters.output_tools) is not list
        or type(parameters.allow_text_output) is not bool
        or type(parameters.allow_image_output) is not bool
        or parameters.allow_image_output is not False
        or parameters.prompted_output_template is not None
        or parameters.instruction_parts is not None
        or parameters.thinking is not None
        or type(parameters.output_mode) is not str
    ):
        raise _error()

    if output_contract.output_kind == "text":
        if (
            parameters.output_mode != "text"
            or parameters.allow_text_output is not True
            or parameters.output_object is not None
            or len(parameters.output_tools) != 0
        ):
            raise _error()
        output_object = None
        output_tools: list[dict[str, Any]] = []
    elif output_contract.output_kind == "structured":
        if (
            parameters.output_mode != "auto"
            or parameters.allow_text_output is not True
            or type(parameters.output_object) is not OutputObjectDefinition
            or len(parameters.output_tools) != 1
        ):
            raise _error()
        output_object = _project_output_object(parameters.output_object)
        output_tools = [_project_output_tool(parameters.output_tools[0])]
    else:  # pragma: no cover - BoundOutputContract closes this.
        raise _error()

    projected = {
        "function_tools": [],
        "native_tools": [],
        "output_mode": parameters.output_mode,
        "output_object": output_object,
        "output_tools": output_tools,
        "prompted_output_template": None,
        "allow_text_output": parameters.allow_text_output,
        "allow_image_output": False,
        "instruction_parts": None,
        "thinking": None,
    }
    contract_record = decode_json_bytes(output_contract.canonical_bytes)
    observed_shape = {
        "output_mode": projected["output_mode"],
        "output_object": projected["output_object"],
        "output_tools": projected["output_tools"],
        "allow_text_output": projected["allow_text_output"],
    }
    # Canonical comparison is type-sensitive, unlike Python equality where
    # True == 1 == 1.0 and hostile equality implementations can participate.
    if canonical_json_bytes(observed_shape) != canonical_json_bytes(
        contract_record["request_shape"]
    ):
        raise _error()
    return projected


def project_model_request(
    model: Model,
    messages: list[Any],
    model_settings: dict[str, Any] | None,
    model_request_parameters: ModelRequestParameters,
    *,
    output_contract: BoundOutputContract,
) -> dict[str, Any]:
    """Return the detached canonical projection for one supported request."""

    attempted = _attempt_project_model_request(
        model,
        messages,
        model_settings,
        model_request_parameters,
        output_contract=output_contract,
    )
    if type(attempted) is not dict:
        raise _error() from None
    return attempted


def _attempt_project_model_request(
    model: Model,
    messages: list[Any],
    model_settings: dict[str, Any] | None,
    model_request_parameters: ModelRequestParameters,
    *,
    output_contract: BoundOutputContract,
) -> dict[str, Any] | object:
    try:
        pydantic_ai_version, pydantic_version, core_version = _require_supported_versions()
        if (
            not isinstance(model, Model)
            or _attempt_validate_bound_output_contract(output_contract) is _FAILED
        ):
            raise _error()
        identity = _model_identity(model)
        profile_record = _strict_json_clone(identity.profile_record)
        contract_record = decode_json_bytes(output_contract.canonical_bytes)
        value = {
            "format": REQUEST_FORMAT,
            "format_version": BOUNDARY_FORMAT_VERSION,
            "boundary": BOUNDARY_NAME,
            "semantic_route": identity.route,
            "pydantic_ai_version": pydantic_ai_version,
            "pydantic_version": pydantic_version,
            "pydantic_core_version": core_version,
            "model_name": identity.model_name,
            "model_system": identity.model_system,
            "model_adapter_type": identity.model_adapter_type,
            "model_profile_type": identity.model_profile_type,
            "provider_type": identity.provider_type,
            "model_profile": profile_record,
            "model_profile_sha256": hashlib.sha256(
                canonical_json_bytes(profile_record)
            ).hexdigest(),
            "model_default_settings": _json_mapping(model.settings),
            "messages": _project_messages(messages, route=identity.route),
            "model_settings": _json_mapping(model_settings),
            "model_request_parameters": _project_parameters(
                model_request_parameters,
                output_contract=output_contract,
            ),
            "output_type_contract": contract_record,
            "output_type_contract_sha256": output_contract.sha256_hex,
            "omitted_framework_fields": list(identity.omissions),
        }
        if frozenset(value) != _REQUEST_KEYS:
            raise _error()
        detached = decode_json_bytes(canonical_json_bytes(value))
        if type(detached) is not dict or frozenset(detached) != _REQUEST_KEYS:
            raise _error()
        return detached
    except Exception:
        return _FAILED


def encode_model_request(
    model: Model,
    messages: list[Any],
    model_settings: dict[str, Any] | None,
    model_request_parameters: ModelRequestParameters,
    *,
    output_contract: BoundOutputContract,
) -> bytes:
    """Encode one supported request to strict canonical JSON bytes."""

    attempted = _attempt_encode_model_request(
        model,
        messages,
        model_settings,
        model_request_parameters,
        output_contract=output_contract,
    )
    if type(attempted) is not bytes:
        raise _error() from None
    return attempted


def _attempt_encode_model_request(
    model: Model,
    messages: list[Any],
    model_settings: dict[str, Any] | None,
    model_request_parameters: ModelRequestParameters,
    *,
    output_contract: BoundOutputContract,
) -> bytes | object:
    try:
        if _attempt_validate_bound_output_contract(output_contract) is _FAILED:
            raise _error()
        return canonical_json_bytes(
            project_model_request(
                model,
                messages,
                model_settings,
                model_request_parameters,
                output_contract=output_contract,
            )
        )
    except Exception:
        return _FAILED


def encode_model_response(response: ModelResponse) -> bytes:
    """Encode a typed response envelope for exact offline reconstruction."""

    attempted = _attempt_encode_model_response(response)
    if type(attempted) is not bytes:
        raise _error() from None
    return attempted


def _attempt_encode_model_response(response: ModelResponse) -> bytes | object:
    try:
        if type(response) is not ModelResponse:
            raise _error()
        pydantic_ai_version, pydantic_version, core_version = _require_supported_versions()
        value = {
            "format": RESPONSE_FORMAT,
            "format_version": BOUNDARY_FORMAT_VERSION,
            "pydantic_ai_version": pydantic_ai_version,
            "pydantic_version": pydantic_version,
            "pydantic_core_version": core_version,
            "response": _RESPONSE_ADAPTER.dump_python(
                response, mode="json", warnings="error"
            ),
        }
        content = canonical_json_bytes(value)
        if _attempt_decode_model_response(content) is _FAILED:
            raise _error()
        return content
    except Exception:
        return _FAILED


def decode_model_response(content: bytes) -> ModelResponse:
    """Decode canonical response bytes, rejecting all coercion/version drift."""

    attempted = _attempt_decode_model_response(content)
    if type(attempted) is not ModelResponse:
        raise _error() from None
    return attempted


def _attempt_decode_model_response(content: bytes) -> ModelResponse | object:
    try:
        if type(content) is not bytes:
            raise _error()
        value = decode_json_bytes(content)
        versions = _require_supported_versions()
        if (
            type(value) is not dict
            or frozenset(value) != _RESPONSE_KEYS
            or canonical_json_bytes(value) != content
            or type(value["format"]) is not str
            or value["format"] != RESPONSE_FORMAT
            or type(value["format_version"]) is not int
            or value["format_version"] != BOUNDARY_FORMAT_VERSION
            or any(
                type(value[key]) is not str
                for key in ("pydantic_ai_version", "pydantic_version", "pydantic_core_version")
            )
            or tuple(
                value[key]
                for key in ("pydantic_ai_version", "pydantic_version", "pydantic_core_version")
            )
            != versions
            or type(value["response"]) is not dict
        ):
            raise _error()
        response = _RESPONSE_ADAPTER.validate_python(value["response"])
        normalized = dict(value)
        normalized["response"] = _RESPONSE_ADAPTER.dump_python(
            response, mode="json", warnings="error"
        )
        # Canonical byte comparison is type-sensitive (unlike Python equality,
        # where True == 1 == 1.0) and rejects aliases/extras/coercion.
        if canonical_json_bytes(normalized) != content:
            raise _error()
        return response
    except Exception:
        return _FAILED


__all__ = [
    "BOUNDARY_FORMAT_VERSION",
    "BOUNDARY_NAME",
    "BoundOutputContract",
    "OUTPUT_CONTRACT_FORMAT",
    "OUTPUT_CONTRACT_FORMAT_VERSION",
    "PydanticAIBoundaryCodecError",
    "REQUEST_FORMAT",
    "RESPONSE_FORMAT",
    "SUPPORTED_PYDANTIC_AI_VERSION",
    "SUPPORTED_PYDANTIC_CORE_VERSION",
    "SUPPORTED_PYDANTIC_VERSION",
    "bind_recorded_output_type_contract",
    "build_output_type_contract",
    "decode_model_response",
    "encode_model_request",
    "encode_model_response",
    "model_profile_sha256",
    "project_model_request",
    "validate_output_type_contract",
]
