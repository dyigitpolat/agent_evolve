"""Pydantic-AI adapter for enum-coded full and partition-block forecasts.

The adapter owns only the prompt/schema boundary and translation into the
provider-neutral forecast ports.  Queueing, retry, transport liveness, model
routing, partition orchestration, and provider policy remain outside it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, ClassVar, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
)

from agent_evolve.domain.finite_variation import FiniteVariationOption
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
    LowLevelRunner,
)
from agent_evolve.ports.action_forecast import (
    ActionEvidenceCitation,
    ActionForecastBlockRequest,
    ActionForecastBlockResult,
    ActionForecastDraft,
    ActionForecastEvidenceMode,
    ActionForecastRequest,
    ActionForecastResult,
    ActionMetricForecast,
    resolve_action_forecast_block,
    resolve_action_forecasts,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)


ACTION_FORECAST_TOOL_NAME = "forecast_all_actions"
ACTION_FORECAST_BLOCK_TOOL_NAME = "forecast_action_block"
ACTION_FORECAST_POLICY_ID = "pydantic_ai_all_option_action_forecast"
ACTION_FORECAST_V4_POLICY_VERSION = 4
ACTION_FORECAST_V4_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-all-option-action-forecast:v4:"
    b"positional-code-matrices-ordinal-metric-scale-effects-asymmetric-"
    b"uncertainty-derived-confidence-discrete-validity-and-one-atomic-prompt-"
    b"visible-evidence-slot-per-grounded-cell-with-no-visible-numeric-bounds-"
    b"hash-bound-global-or-partition-block-positional-frames-block-local-only-"
    b"emission-and-logarithmic-effect-and-uncertainty-tails-through-32"
).hexdigest()
ACTION_FORECAST_POLICY_VERSION = 5
ACTION_FORECAST_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-all-option-action-forecast:v5:"
    b"positional-code-matrices-ordinal-metric-scale-effects-asymmetric-"
    b"adjacent-midpoint-quantization-floors-with-virtual-endpoints-minus-and-"
    b"plus-64-and-prompt-semantics-schema-6-and-excess-epistemic-uncertainty-"
    b"derived-confidence-excluding-quantization-floors-discrete-validity-and-"
    b"one-atomic-prompt-"
    b"visible-evidence-slot-per-grounded-cell-with-no-visible-numeric-bounds-"
    b"hash-bound-global-or-partition-block-positional-frames-block-local-only-"
    b"emission-and-logarithmic-effect-and-uncertainty-tails-through-32"
).hexdigest()

_STRICT_CONFIG = ConfigDict(
    extra="forbid",
    strict=True,
    frozen=True,
    validate_default=True,
)

# The v2 live trace demonstrated that a model may copy a provider-visible
# numeric endpoint into every prediction.  V4 removed numerical output fields
# and artificial floating-point boundaries; v5 preserves that exact enum shape
# while adding trusted quantization floors after typed admission.  Codes are
# mapped to benchmark-supplied practical metric scales only after admission.
_EFFECT_MULTIPLIERS: dict[str, float] = {
    "n32": -32.0,
    "n16": -16.0,
    "n8": -8.0,
    "n4": -4.0,
    "n2": -2.0,
    "n1": -1.0,
    "n0_5": -0.5,
    "n0_25": -0.25,
    "z": 0.0,
    "p0_25": 0.25,
    "p0_5": 0.5,
    "p1": 1.0,
    "p2": 2.0,
    "p4": 4.0,
    "p8": 8.0,
    "p16": 16.0,
    "p32": 32.0,
}
_UNCERTAINTY_MULTIPLIERS: dict[str, float] = {
    "u0": 0.0,
    "u0_25": 0.25,
    "u0_5": 0.5,
    "u1": 1.0,
    "u2": 2.0,
    "u4": 4.0,
    "u8": 8.0,
    "u16": 16.0,
    "u32": 32.0,
}
_VALIDITY_VALUES: dict[str, float] = {
    "p0_05": 0.05,
    "p0_2": 0.2,
    "p0_4": 0.4,
    "p0_6": 0.6,
    "p0_8": 0.8,
    "p0_95": 0.95,
}

_EFFECT_CODES = tuple(_EFFECT_MULTIPLIERS)
_UNCERTAINTY_CODES = tuple(_UNCERTAINTY_MULTIPLIERS)
_VALIDITY_CODES = tuple(_VALIDITY_VALUES)

_SUPPORTED_PROVIDER_WIRE_VERSIONS = (4, 5)
_CURRENT_PROVIDER_WIRE_VERSION = 5
_VIRTUAL_LOWER_EFFECT_CENTER = -64.0
_VIRTUAL_UPPER_EFFECT_CENTER = 64.0


def _build_effect_quantization_floors() -> dict[str, tuple[float, float]]:
    """Return asymmetric half-bin widths around every ordinal effect center."""

    items = tuple(_EFFECT_MULTIPLIERS.items())
    centers = tuple(value for _, value in items)
    if any(
        not math.isfinite(value)
        for value in (
            *centers,
            _VIRTUAL_LOWER_EFFECT_CENTER,
            _VIRTUAL_UPPER_EFFECT_CENTER,
        )
    ):
        raise RuntimeError("effect centers and virtual endpoints must be finite")
    extended = (
        _VIRTUAL_LOWER_EFFECT_CENTER,
        *centers,
        _VIRTUAL_UPPER_EFFECT_CENTER,
    )
    if any(
        lower >= upper
        for lower, upper in zip(extended[:-1], extended[1:], strict=True)
    ):
        raise RuntimeError("effect centers and virtual endpoints must be ordered")
    return {
        code: (
            (center - extended[index]) / 2.0,
            (extended[index + 2] - center) / 2.0,
        )
        for index, (code, center) in enumerate(items)
    }


_EFFECT_QUANTIZATION_FLOORS = _build_effect_quantization_floors()


class _ActionForecastWireBase(BaseModel):
    """Validator-free base whose complete contract is visible in JSON Schema."""

    model_config = _STRICT_CONFIG


def _prompt_visible_citation_pairs(
    request: ActionForecastRequest,
) -> tuple[tuple[str, str], ...]:
    pairs = tuple(
        sorted(
            (
                (card.card_key, binding.identity_sha256)
                for card in request.cards
                for binding in card.finite_action_evidence
            )
        )
    )
    if len(set(pairs)) != len(pairs):
        raise ValueError("prompt-visible card/action citation pairs must be unique")
    return pairs


def _prompt_visible_evidence_slots(
    request: ActionForecastRequest,
) -> tuple[tuple[str, str, str], ...]:
    """Assign compact atomic IDs to exact prompt-visible citation pairs."""

    return tuple(
        (f"e{index}", card_key, binding_sha256)
        for index, (card_key, binding_sha256) in enumerate(
            _prompt_visible_citation_pairs(request)
        )
    )


def _action_forecast_output_type(
    request: ActionForecastRequest,
    *,
    option_count: int | None = None,
    provider_wire_version: int = _CURRENT_PROVIDER_WIRE_VERSION,
) -> type[BaseModel]:
    request.__post_init__()
    grounded = request.evidence_mode is ActionForecastEvidenceMode.GROUNDED
    evidence_slot_ids = tuple(
        value[0] for value in _prompt_visible_evidence_slots(request)
    )
    return _cached_action_forecast_output_type(
        (
            len(request.finite_variation_contract.options)
            if option_count is None
            else option_count
        ),
        len(request.required_metric_ids),
        grounded,
        evidence_slot_ids,
        provider_wire_version,
    )


@lru_cache(maxsize=512)
def _cached_action_forecast_output_type(
    option_count: int,
    metric_count: int,
    grounded: bool,
    evidence_slot_ids: tuple[str, ...],
    provider_wire_version: int,
) -> type[BaseModel]:
    """Build a schema whose visible constraints are its complete wire contract."""

    if type(option_count) is not int or option_count <= 0:
        raise ValueError("option_count must be a positive exact integer")
    if type(metric_count) is not int or metric_count <= 0:
        raise ValueError("metric_count must be a positive exact integer")
    if type(grounded) is not bool:
        raise TypeError("grounded must be an exact bool")
    if grounded and not evidence_slot_ids:
        raise ValueError("grounded forecast schema requires evidence slots")
    if not grounded and evidence_slot_ids:
        raise ValueError("catalog-only forecast schema forbids evidence slots")
    if provider_wire_version not in _SUPPORTED_PROVIDER_WIRE_VERSIONS:
        raise ValueError("provider_wire_version must be 4 or 5")

    effect_literal = Literal.__getitem__(_EFFECT_CODES)
    uncertainty_literal = Literal.__getitem__(_UNCERTAINTY_CODES)
    validity_literal = Literal.__getitem__(_VALIDITY_CODES)

    effect_row = Annotated[
        list[effect_literal],
        Field(min_length=metric_count, max_length=metric_count),
    ]
    uncertainty_row = Annotated[
        list[uncertainty_literal],
        Field(min_length=metric_count, max_length=metric_count),
    ]
    matrix_fields: dict[str, Any] = {
        "probability_valid_codes": (
            list[validity_literal],
            Field(
                min_length=option_count,
                max_length=option_count,
                description=(
                    "One ordinal validity-probability code per ordered option."
                ),
            ),
        ),
        "median_effect_codes": (
            list[effect_row],
            Field(
                min_length=option_count,
                max_length=option_count,
                description=(
                    "Signed median child-minus-parent effects in metric-scale units; "
                    "matrix[i][j] maps to ordered option i and metric j."
                ),
            ),
        ),
        "lower_uncertainty_codes": (
            list[uncertainty_row],
            Field(
                min_length=option_count,
                max_length=option_count,
                description=(
                    "Nonnegative excess epistemic p50-to-p10 distance beyond "
                    "the effect code's lower quantization floor, in metric-scale "
                    "units."
                    if provider_wire_version == 5
                    else "Nonnegative p50-to-p10 distances in metric-scale units."
                ),
            ),
        ),
        "upper_uncertainty_codes": (
            list[uncertainty_row],
            Field(
                min_length=option_count,
                max_length=option_count,
                description=(
                    "Nonnegative excess epistemic p50-to-p90 distance beyond "
                    "the effect code's upper quantization floor, in metric-scale "
                    "units."
                    if provider_wire_version == 5
                    else "Nonnegative p50-to-p90 distances in metric-scale units."
                ),
            ),
        ),
    }
    if grounded:
        evidence_slot_literal = Literal.__getitem__(evidence_slot_ids)
        evidence_row = Annotated[
            list[evidence_slot_literal],
            Field(min_length=metric_count, max_length=metric_count),
        ]
        matrix_fields["evidence_slot_codes"] = (
            list[evidence_row],
            Field(
                min_length=option_count,
                max_length=option_count,
                description=(
                    "One atomic prompt-visible evidence slot per option-metric cell."
                ),
            ),
        )

    output_type = create_model(
        f"AllOptionActionForecastMatrixV{provider_wire_version}",
        __base__=_ActionForecastWireBase,
        __module__=__name__,
        **matrix_fields,
    )
    return output_type


def _render_action_forecast_prompt_frame(
    request: ActionForecastRequest,
    *,
    global_row_start: int,
    global_row_stop: int,
    frame_binding: dict[str, object],
    provider_wire_version: int,
) -> str:
    """Render one position-bound full or block forecast frame."""

    if type(request) is not ActionForecastRequest:
        raise TypeError("request must be an exact ActionForecastRequest")
    request.__post_init__()
    contract = request.finite_variation_contract
    if (
        type(global_row_start) is not int
        or type(global_row_stop) is not int
        or global_row_start < 0
        or global_row_stop <= global_row_start
        or global_row_stop > len(contract.options)
    ):
        raise ValueError("forecast frame must be a non-empty in-contract row slice")
    if type(frame_binding) is not dict:
        raise TypeError("frame_binding must be an exact dict")
    if provider_wire_version not in _SUPPORTED_PROVIDER_WIRE_VERSIONS:
        raise ValueError("provider_wire_version must be 4 or 5")
    options = contract.options[global_row_start:global_row_stop]
    evidence_slots = _prompt_visible_evidence_slots(request)
    effect_meanings = {
        "n32": "negative thirty-two times this metric's delta_scale",
        "n16": "negative sixteen times this metric's delta_scale",
        "n8": "negative eight times this metric's delta_scale",
        "n4": "negative four times this metric's delta_scale",
        "n2": "negative two times this metric's delta_scale",
        "n1": "negative one times this metric's delta_scale",
        "n0_5": "negative one half of this metric's delta_scale",
        "n0_25": "negative one quarter of this metric's delta_scale",
        "z": "zero child-minus-parent change",
        "p0_25": "positive one quarter of this metric's delta_scale",
        "p0_5": "positive one half of this metric's delta_scale",
        "p1": "positive one times this metric's delta_scale",
        "p2": "positive two times this metric's delta_scale",
        "p4": "positive four times this metric's delta_scale",
        "p8": "positive eight times this metric's delta_scale",
        "p16": "positive sixteen times this metric's delta_scale",
        "p32": "positive thirty-two times this metric's delta_scale",
    }
    if provider_wire_version == 5:
        uncertainty_meanings = {
            "u0": "zero excess epistemic scale units beyond the quantization floor",
            "u0_25": (
                "one quarter excess epistemic scale unit beyond the quantization floor"
            ),
            "u0_5": (
                "one half excess epistemic scale unit beyond the quantization floor"
            ),
            "u1": "one excess epistemic scale unit beyond the quantization floor",
            "u2": "two excess epistemic scale units beyond the quantization floor",
            "u4": "four excess epistemic scale units beyond the quantization floor",
            "u8": "eight excess epistemic scale units beyond the quantization floor",
            "u16": (
                "sixteen excess epistemic scale units beyond the quantization floor"
            ),
            "u32": (
                "thirty-two excess epistemic scale units beyond the quantization floor"
            ),
        }
    else:
        uncertainty_meanings = {
            "u0": "zero additional scale units",
            "u0_25": "one quarter scale unit",
            "u0_5": "one half scale unit",
            "u1": "one scale unit",
            "u2": "two scale units",
            "u4": "four scale units",
            "u8": "eight scale units",
            "u16": "sixteen scale units",
            "u32": "thirty-two scale units",
        }
    validity_meanings = {
        "p0_05": "about five percent probability valid",
        "p0_2": "about twenty percent probability valid",
        "p0_4": "about forty percent probability valid",
        "p0_6": "about sixty percent probability valid",
        "p0_8": "about eighty percent probability valid",
        "p0_95": "about ninety-five percent probability valid",
    }
    machine_contract = {
        "schema_version": 6 if provider_wire_version == 5 else 5,
        "request_sha256": request.request_sha256,
        "context_sha256": request.context_sha256,
        "context": thaw_json(request.context),
        "optimization_semantics": request.optimization_semantics.to_record(),
        "action_semantics": request.action_semantics.to_record(),
        "finite_variation_contract": {
            "catalog_id": contract.catalog_id,
            "catalog_version": contract.catalog_version,
            "catalog_definition_sha256": contract.catalog_definition_sha256,
            "parent_configuration_sha256": contract.parent_configuration_sha256,
            "contract_identity_sha256": contract.identity_sha256,
        },
        "forecast_frame": frame_binding,
        "ordered_options": [
            {
                "row_index": local_index,
                "global_row_index": global_row_start + local_index,
                **option.prompt_record(),
            }
            for local_index, option in enumerate(options)
        ],
        "forecast_metrics": [
            {
                "column_index": index,
                "metric_id": parent.metric_id,
                "parent_value": parent.value,
                "delta_scale": scale.delta_scale,
                "scale_definition_sha256": scale.definition_sha256,
            }
            for index, (parent, scale) in enumerate(
                zip(
                    request.parent_metric_values,
                    request.metric_scales,
                    strict=True,
                )
            )
        ],
        "evidence_mode": request.evidence_mode.value,
        "cards": [card.prompt_record() for card in request.cards],
        "evidence_slots": [
            {
                "evidence_slot_id": slot_id,
                "card_key": card_key,
                "action_binding_identity_sha256": binding_sha256,
            }
            for slot_id, card_key, binding_sha256 in evidence_slots
        ],
        "ordinal_codebook": {
            "median_effect_codes": effect_meanings,
            "uncertainty_codes": uncertainty_meanings,
            "probability_valid_codes": validity_meanings,
            "derived_confidence": (
                "trusted code derives confidence monotonically from only the total "
                "lower-plus-upper excess epistemic uncertainty, excluding fixed "
                "quantization floors; do not emit a confidence field"
                if provider_wire_version == 5
                else "trusted code derives confidence monotonically from the total "
                "lower-plus-upper uncertainty; do not emit a confidence field"
            ),
        },
        "output_contract": {
            "action_row_count": len(options),
            "action_row_binding": (
                "every top-level vector or matrix row i maps to ordered_options[i]"
            ),
            "metric_cell_count_per_row": len(request.required_metric_ids),
            "metric_cell_binding": (
                "every matrix cell [i][j] maps to forecast_metrics[j]"
            ),
            "delta_definition": "child_metric_minus_parent_metric",
            "quantile_derivation": {
                "p10_delta": (
                    "(median effect units - lower quantization-floor units - lower "
                    "excess epistemic uncertainty units) times delta_scale"
                    if provider_wire_version == 5
                    else "(median effect units - lower uncertainty units) times "
                    "delta_scale"
                ),
                "p50_delta": "median effect units times delta_scale",
                "p90_delta": (
                    "(median effect units + upper quantization-floor units + upper "
                    "excess epistemic uncertainty units) times delta_scale"
                    if provider_wire_version == 5
                    else "(median effect units + upper uncertainty units) times "
                    "delta_scale"
                ),
            },
            "provider_numeric_output_fields": [],
            "one_evidence_slot_required_per_metric": (
                request.evidence_mode is ActionForecastEvidenceMode.GROUNDED
            ),
        },
    }
    if provider_wire_version == 5:
        machine_contract["effect_quantization"] = {
            "rule": "adjacent_midpoints_with_virtual_endpoint_centers",
            "virtual_lower_effect_center_units": _VIRTUAL_LOWER_EFFECT_CENTER,
            "virtual_upper_effect_center_units": _VIRTUAL_UPPER_EFFECT_CENTER,
            "asymmetric_floor_units_by_effect_code": {
                code: {
                    "lower_floor_units": floors[0],
                    "upper_floor_units": floors[1],
                }
                for code, floors in _EFFECT_QUANTIZATION_FLOORS.items()
            },
        }
    encoded = json.dumps(
        machine_contract,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    citation_instruction = (
        "For every metric cell choose exactly one supplied evidence slot in "
        "evidence_slot_codes[i][j] as its primary attributable card/action source."
        if request.evidence_mode is ActionForecastEvidenceMode.GROUNDED
        else "Do not emit evidence_slot_codes; no evidence cards are supplied."
    )
    frame_instruction = (
        "This is a partition block. Forecast exactly and only the "
        "local_row_count ordered_options supplied in this frame; do not emit "
        "rows for absent global options."
        if frame_binding.get("frame_kind") == "partition_block"
        else "This is the complete forecast frame."
    )
    uncertainty_instructions: tuple[str, ...] = ()
    if provider_wire_version == 5:
        uncertainty_instructions = (
            "Lower and upper uncertainty codes encode only excess epistemic "
            "distance beyond the selected median effect code's trusted asymmetric "
            "midpoint quantization floors; trusted code adds those floors when "
            "deriving p10 and p90, while confidence uses only the emitted excess "
            "uncertainty.",
        )
    return "\n".join(
        (
            request.instruction,
            "",
            "ALL-OPTION ACTION FORECAST CONTRACT",
            encoded,
            "Return exactly one probability_valid_codes entry and one row in every "
            "required code matrix for each ordered_options entry, with exactly one "
            "code per forecast_metrics entry in each matrix row. "
            "Do not emit option IDs, metric IDs, or candidate configurations; "
            "trusted code reattaches those identities by position. Use only the "
            "closed ordinal codes. Do not emit any numeric forecast or confidence "
            "field. Negative/positive effect codes always mean signed raw "
            "child-minus-parent change, independent of whether a metric is "
            "minimized, maximized, or constrained.",
            *uncertainty_instructions,
            frame_instruction,
            citation_instruction,
        )
    )


def render_action_forecast_prompt(request: ActionForecastRequest) -> str:
    """Render the current v5 prompt-safe all-option forecast contract."""

    return _render_complete_action_forecast_prompt(
        request,
        provider_wire_version=_CURRENT_PROVIDER_WIRE_VERSION,
    )


def render_action_forecast_v4_prompt(request: ActionForecastRequest) -> str:
    """Render the sealed v4 contract for historical replay compatibility."""

    return _render_complete_action_forecast_prompt(request, provider_wire_version=4)


def _render_complete_action_forecast_prompt(
    request: ActionForecastRequest,
    *,
    provider_wire_version: int,
) -> str:
    """Render one explicitly versioned complete forecast contract."""

    if type(request) is not ActionForecastRequest:
        raise TypeError("request must be an exact ActionForecastRequest")
    request.__post_init__()
    option_count = len(request.finite_variation_contract.options)
    return _render_action_forecast_prompt_frame(
        request,
        global_row_start=0,
        global_row_stop=option_count,
        frame_binding={
            "frame_kind": "complete",
            "global_option_count": option_count,
            "global_row_start": 0,
            "global_row_stop": option_count,
            "local_row_count": option_count,
        },
        provider_wire_version=provider_wire_version,
    )


def render_action_forecast_block_prompt(
    block_request: ActionForecastBlockRequest,
) -> str:
    """Render one current v5 block without pretending it is a complete batch."""

    return _render_action_forecast_block_prompt(
        block_request,
        provider_wire_version=_CURRENT_PROVIDER_WIRE_VERSION,
    )


def render_action_forecast_v4_block_prompt(
    block_request: ActionForecastBlockRequest,
) -> str:
    """Render one sealed v4 block for historical replay compatibility."""

    return _render_action_forecast_block_prompt(block_request, provider_wire_version=4)


def _render_action_forecast_block_prompt(
    block_request: ActionForecastBlockRequest,
    *,
    provider_wire_version: int,
) -> str:
    """Render one explicitly versioned immutable block forecast contract."""

    if type(block_request) is not ActionForecastBlockRequest:
        raise TypeError("block_request must be exact ActionForecastBlockRequest")
    block_request.__post_init__()
    block = block_request.block
    return _render_action_forecast_prompt_frame(
        block_request.request,
        global_row_start=block.global_row_start,
        global_row_stop=block.global_row_stop,
        frame_binding={
            "frame_kind": "partition_block",
            "global_option_count": block_request.layout.row_count,
            "global_row_start": block.global_row_start,
            "global_row_stop": block.global_row_stop,
            "local_row_count": block.row_count,
            "layout_sha256": block_request.layout.layout_sha256,
            "block_index": block.block_index,
            "block_spec_sha256": block.block_spec_sha256,
            "block_request_sha256": block_request.block_request_sha256,
        },
        provider_wire_version=provider_wire_version,
    )


def plan_action_forecast_request(
    request: ActionForecastRequest,
) -> StructuredGenerationRequest[BaseModel]:
    """Purely plan the current v5 request without dispatching it."""

    return _plan_action_forecast_request(
        request,
        provider_wire_version=_CURRENT_PROVIDER_WIRE_VERSION,
    )


def plan_action_forecast_v4_request(
    request: ActionForecastRequest,
) -> StructuredGenerationRequest[BaseModel]:
    """Plan the sealed v4 request for historical replay compatibility."""

    return _plan_action_forecast_request(request, provider_wire_version=4)


def _plan_action_forecast_request(
    request: ActionForecastRequest,
    *,
    provider_wire_version: int,
) -> StructuredGenerationRequest[BaseModel]:
    """Purely plan one explicitly versioned request without dispatching it."""

    if type(request) is not ActionForecastRequest:
        raise TypeError("request must be an exact ActionForecastRequest")
    request.__post_init__()
    return StructuredGenerationRequest(
        call_id=request.call_id,
        operation=request.operation,
        prompt=(
            render_action_forecast_prompt(request)
            if provider_wire_version == _CURRENT_PROVIDER_WIRE_VERSION
            else render_action_forecast_v4_prompt(request)
        ),
        output_type=_action_forecast_output_type(
            request,
            provider_wire_version=provider_wire_version,
        ),
        output_tool_name=ACTION_FORECAST_TOOL_NAME,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
    )


def plan_action_forecast_block_request(
    block_request: ActionForecastBlockRequest,
) -> StructuredGenerationRequest[BaseModel]:
    """Plan one current v5 block retaining the global scientific binding."""

    return _plan_action_forecast_block_request(
        block_request,
        provider_wire_version=_CURRENT_PROVIDER_WIRE_VERSION,
    )


def plan_action_forecast_v4_block_request(
    block_request: ActionForecastBlockRequest,
) -> StructuredGenerationRequest[BaseModel]:
    """Plan one sealed v4 block for historical replay compatibility."""

    return _plan_action_forecast_block_request(block_request, provider_wire_version=4)


def _plan_action_forecast_block_request(
    block_request: ActionForecastBlockRequest,
    *,
    provider_wire_version: int,
) -> StructuredGenerationRequest[BaseModel]:
    """Plan one explicitly versioned physical block request."""

    if type(block_request) is not ActionForecastBlockRequest:
        raise TypeError("block_request must be exact ActionForecastBlockRequest")
    block_request.__post_init__()
    request = block_request.request
    return StructuredGenerationRequest(
        call_id=block_request.block_call_id,
        operation=request.operation,
        prompt=(
            render_action_forecast_block_prompt(block_request)
            if provider_wire_version == _CURRENT_PROVIDER_WIRE_VERSION
            else render_action_forecast_v4_block_prompt(block_request)
        ),
        output_type=_action_forecast_output_type(
            request,
            option_count=block_request.block.row_count,
            provider_wire_version=provider_wire_version,
        ),
        output_tool_name=ACTION_FORECAST_BLOCK_TOOL_NAME,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
    )


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
        raise TypeError(
            "low-level response value does not match the action forecast output type"
        )
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


def _drafts_from_wire(
    request: ActionForecastRequest,
    value: Any,
    options: tuple[FiniteVariationOption, ...],
    *,
    provider_wire_version: int,
) -> tuple[ActionForecastDraft, ...]:
    """Decode one admitted positional frame into provider-neutral drafts."""

    if type(options) is not tuple or not options or any(
        type(option) is not FiniteVariationOption for option in options
    ):
        raise ValueError("options must be a non-empty exact finite-option tuple")
    if provider_wire_version not in _SUPPORTED_PROVIDER_WIRE_VERSIONS:
        raise ValueError("provider_wire_version must be 4 or 5")
    citation_by_slot_id = {
        slot_id: ActionEvidenceCitation(
            card_key=card_key,
            action_binding_identity_sha256=binding_sha256,
        )
        for slot_id, card_key, binding_sha256 in (
            _prompt_visible_evidence_slots(request)
        )
    }
    drafts_list: list[ActionForecastDraft] = []
    for row_index, option in enumerate(options):
        metric_forecasts: list[ActionMetricForecast] = []
        for metric_index, (metric_id, scale) in enumerate(
            zip(
                request.required_metric_ids,
                request.metric_scales,
                strict=True,
            )
        ):
            effect_code = cast(
                str,
                value.median_effect_codes[row_index][metric_index],
            )
            median_units = _EFFECT_MULTIPLIERS[effect_code]
            lower_excess_units = _UNCERTAINTY_MULTIPLIERS[
                cast(
                    str,
                    value.lower_uncertainty_codes[row_index][metric_index],
                )
            ]
            upper_excess_units = _UNCERTAINTY_MULTIPLIERS[
                cast(
                    str,
                    value.upper_uncertainty_codes[row_index][metric_index],
                )
            ]
            lower_floor_units, upper_floor_units = (
                _EFFECT_QUANTIZATION_FLOORS[effect_code]
                if provider_wire_version == 5
                else (0.0, 0.0)
            )
            lower_units = lower_floor_units + lower_excess_units
            upper_units = upper_floor_units + upper_excess_units
            median = median_units * scale.delta_scale
            lower = lower_units * scale.delta_scale
            upper = upper_units * scale.delta_scale
            if not all(math.isfinite(item) for item in (median, lower, upper)):
                raise ValueError(
                    "metric-scale code denormalization produced a non-finite delta"
                )
            confidence = 1.0 / (
                1.0 + lower_excess_units + upper_excess_units
            )
            citations = ()
            if request.evidence_mode is ActionForecastEvidenceMode.GROUNDED:
                slot_id = cast(
                    str,
                    value.evidence_slot_codes[row_index][metric_index],
                )
                citations = (citation_by_slot_id[slot_id],)
            metric_forecasts.append(
                ActionMetricForecast(
                    metric_id=metric_id,
                    p10_delta=median - lower,
                    p50_delta=median,
                    p90_delta=median + upper,
                    confidence=confidence,
                    citations=citations,
                )
            )
        drafts_list.append(
            ActionForecastDraft(
                option_id=option.option_id,
                probability_valid=_VALIDITY_VALUES[
                    cast(str, value.probability_valid_codes[row_index])
                ],
                metric_forecasts=tuple(metric_forecasts),
            )
        )
    return tuple(drafts_list)


@dataclass(slots=True)
class PydanticAIActionForecastPolicy:
    """Translate one queued structured call into a trusted all-option batch."""

    generate_once: LowLevelRunner

    policy_id: ClassVar[str] = ACTION_FORECAST_POLICY_ID
    policy_version: ClassVar[int] = ACTION_FORECAST_POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = (
        ACTION_FORECAST_POLICY_DEFINITION_SHA256
    )
    provider_wire_version: ClassVar[int] = _CURRENT_PROVIDER_WIRE_VERSION

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")

    async def forecast(
        self,
        request: ActionForecastRequest,
    ) -> ActionForecastResult:
        if type(request) is not ActionForecastRequest:
            raise TypeError("request must be an exact ActionForecastRequest")
        request.__post_init__()
        low_level_request = _plan_action_forecast_request(
            request,
            provider_wire_version=self.provider_wire_version,
        )
        output_type = low_level_request.output_type
        raw = await self.generate_once(low_level_request)
        response, attempt_count = _validated_response(
            raw,
            output_type=output_type,
        )
        value = cast(Any, response.value)
        drafts = _drafts_from_wire(
            request,
            value,
            request.finite_variation_contract.options,
            provider_wire_version=self.provider_wire_version,
        )
        forecasts = resolve_action_forecasts(
            request,
            drafts,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.policy_definition_sha256,
        )
        return ActionForecastResult(
            forecasts=forecasts,
            telemetry=_telemetry(response, attempt_count=attempt_count),
        )


@dataclass(slots=True)
class PydanticAIActionForecastBlockPolicy:
    """Translate one queued physical block into a trusted partial receipt."""

    generate_once: LowLevelRunner

    policy_id: ClassVar[str] = ACTION_FORECAST_POLICY_ID
    policy_version: ClassVar[int] = ACTION_FORECAST_POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = (
        ACTION_FORECAST_POLICY_DEFINITION_SHA256
    )
    provider_wire_version: ClassVar[int] = _CURRENT_PROVIDER_WIRE_VERSION

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")

    async def forecast_block(
        self,
        block_request: ActionForecastBlockRequest,
    ) -> ActionForecastBlockResult:
        if type(block_request) is not ActionForecastBlockRequest:
            raise TypeError("block_request must be exact ActionForecastBlockRequest")
        block_request.__post_init__()
        low_level_request = _plan_action_forecast_block_request(
            block_request,
            provider_wire_version=self.provider_wire_version,
        )
        output_type = low_level_request.output_type
        raw = await self.generate_once(low_level_request)
        response, attempt_count = _validated_response(raw, output_type=output_type)
        value = cast(Any, response.value)
        spec = block_request.block
        options = block_request.request.finite_variation_contract.options[
            spec.global_row_start : spec.global_row_stop
        ]
        drafts = _drafts_from_wire(
            block_request.request,
            value,
            options,
            provider_wire_version=self.provider_wire_version,
        )
        forecasts = resolve_action_forecast_block(
            block_request,
            drafts,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.policy_definition_sha256,
        )
        return ActionForecastBlockResult(
            forecasts=forecasts,
            telemetry=_telemetry(response, attempt_count=attempt_count),
        )


class PydanticAIActionForecastV4Policy(PydanticAIActionForecastPolicy):
    """Decode and resolve the sealed v4 wire for historical replay only."""

    policy_version: ClassVar[int] = ACTION_FORECAST_V4_POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = (
        ACTION_FORECAST_V4_POLICY_DEFINITION_SHA256
    )
    provider_wire_version: ClassVar[int] = 4


class PydanticAIActionForecastV4BlockPolicy(PydanticAIActionForecastBlockPolicy):
    """Decode and resolve sealed v4 partition blocks for historical replay."""

    policy_version: ClassVar[int] = ACTION_FORECAST_V4_POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = (
        ACTION_FORECAST_V4_POLICY_DEFINITION_SHA256
    )
    provider_wire_version: ClassVar[int] = 4


__all__ = [
    "ACTION_FORECAST_BLOCK_TOOL_NAME",
    "ACTION_FORECAST_POLICY_DEFINITION_SHA256",
    "ACTION_FORECAST_POLICY_ID",
    "ACTION_FORECAST_POLICY_VERSION",
    "ACTION_FORECAST_TOOL_NAME",
    "ACTION_FORECAST_V4_POLICY_DEFINITION_SHA256",
    "ACTION_FORECAST_V4_POLICY_VERSION",
    "PydanticAIActionForecastBlockPolicy",
    "PydanticAIActionForecastPolicy",
    "PydanticAIActionForecastV4BlockPolicy",
    "PydanticAIActionForecastV4Policy",
    "plan_action_forecast_block_request",
    "plan_action_forecast_request",
    "plan_action_forecast_v4_block_request",
    "plan_action_forecast_v4_request",
    "render_action_forecast_block_prompt",
    "render_action_forecast_prompt",
    "render_action_forecast_v4_block_prompt",
    "render_action_forecast_v4_prompt",
]
