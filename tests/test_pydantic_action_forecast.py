from __future__ import annotations

import asyncio
import copy
import json
import math
from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from agent_evolve.application.action_forecast_partitioning import (
    build_action_forecast_block_requests,
    build_action_forecast_partition_layout,
)
from agent_evolve.integrations.pydantic_ai.action_forecast import (
    ACTION_FORECAST_BLOCK_TOOL_NAME,
    ACTION_FORECAST_POLICY_DEFINITION_SHA256,
    ACTION_FORECAST_POLICY_ID,
    ACTION_FORECAST_POLICY_VERSION,
    ACTION_FORECAST_TOOL_NAME,
    ACTION_FORECAST_V4_POLICY_DEFINITION_SHA256,
    ACTION_FORECAST_V4_POLICY_VERSION,
    PydanticAIActionForecastBlockPolicy,
    PydanticAIActionForecastPolicy,
    PydanticAIActionForecastV4BlockPolicy,
    PydanticAIActionForecastV4Policy,
    plan_action_forecast_block_request,
    plan_action_forecast_request,
    plan_action_forecast_v4_block_request,
    plan_action_forecast_v4_request,
    render_action_forecast_block_prompt,
    render_action_forecast_prompt,
    render_action_forecast_v4_prompt,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
)
from agent_evolve.ports.action_forecast import (
    ActionForecastEvidenceMode,
    ActionForecastPartitionPolicyBinding,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)
from tests.test_action_forecast_allocation import _request


_EFFECT_UNITS = {
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
_UNCERTAINTY_UNITS = {
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
_VALIDITY_VALUES = {
    "p0_05": 0.05,
    "p0_2": 0.2,
    "p0_4": 0.4,
    "p0_6": 0.6,
    "p0_8": 0.8,
    "p0_95": 0.95,
}


def _effect_quantization_floors(effect_code: str) -> tuple[float, float]:
    items = tuple(_EFFECT_UNITS.items())
    index = tuple(code for code, _ in items).index(effect_code)
    centers = (-64.0, *(value for _, value in items), 64.0)
    center = _EFFECT_UNITS[effect_code]
    return (
        (center - centers[index]) / 2.0,
        (centers[index + 2] - center) / 2.0,
    )


class _FakeRunner:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.requests: list[StructuredGenerationRequest[Any]] = []

    async def __call__(self, request: StructuredGenerationRequest[Any]):
        self.requests.append(request)
        return self.handler(request)


def _response(value: Any) -> StructuredGenerationResponse[Any]:
    return StructuredGenerationResponse(
        value=value,
        requested_model="deepseek/deepseek-v4-pro",
        resolved_model="deepseek/deepseek-v4-pro",
        resolved_provider="streamlake",
        provider_response_id="response-action-forecast",
        finish_reason="stop",
        input_tokens=2_000,
        output_tokens=700,
        reasoning_tokens=250,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0.02"),
        latency_ns=3_000_000_000,
    )


def _evidence_slot_by_pair(request) -> dict[tuple[str, str], str]:
    pairs = sorted(
        (card.card_key, binding.identity_sha256)
        for card in request.cards
        for binding in card.finite_action_evidence
    )
    return {pair: f"e{index}" for index, pair in enumerate(pairs)}


def _wire_payload(
    request,
    *,
    option_count: int | None = None,
    effect_code: str | None = None,
    lower_code: str = "u0_5",
    upper_code: str = "u1",
    validity_code: str = "p0_8",
) -> dict[str, object]:
    resolved_option_count = (
        len(request.finite_variation_contract.options)
        if option_count is None
        else option_count
    )
    metric_count = len(request.required_metric_ids)
    effect_pattern = tuple(_EFFECT_UNITS)
    payload: dict[str, object] = {
        "probability_valid_codes": [validity_code] * resolved_option_count,
        "median_effect_codes": [
            [
                effect_code
                or effect_pattern[(row_index * metric_count + metric_index) % len(effect_pattern)]
                for metric_index in range(metric_count)
            ]
            for row_index in range(resolved_option_count)
        ],
        "lower_uncertainty_codes": [
            [lower_code] * metric_count for _ in range(resolved_option_count)
        ],
        "upper_uncertainty_codes": [
            [upper_code] * metric_count for _ in range(resolved_option_count)
        ],
    }
    if request.evidence_mode is ActionForecastEvidenceMode.GROUNDED:
        slots = tuple(_evidence_slot_by_pair(request).values())
        payload["evidence_slot_codes"] = [
            [
                slots[(row_index + metric_index) % len(slots)]
                for metric_index in range(metric_count)
            ]
            for row_index in range(resolved_option_count)
        ]
    return payload


def _machine_contract(prompt: str) -> dict[str, object]:
    lines = prompt.splitlines()
    marker = lines.index("ALL-OPTION ACTION FORECAST CONTRACT")
    value = json.loads(lines[marker + 1])
    assert type(value) is dict
    return value


def _run_payload(request, payload: dict[str, object]):
    def handle(low_level: StructuredGenerationRequest[Any]):
        value = low_level.output_type.model_validate(payload, strict=True)
        return _response(value)

    runner = _FakeRunner(handle)
    result = asyncio.run(PydanticAIActionForecastPolicy(runner).forecast(request))
    return runner, result


def test_grounded_adapter_preserves_attempts_and_reattaches_identities() -> None:
    request = _request()
    payload = _wire_payload(request)

    def handle(low_level: StructuredGenerationRequest[Any]):
        value = low_level.output_type.model_validate(payload, strict=True)
        return AttemptedStructuredGenerationResponse(
            response=_response(value),
            attempt_count=4,
        )

    runner = _FakeRunner(handle)
    result = asyncio.run(PydanticAIActionForecastPolicy(runner).forecast(request))

    assert len(runner.requests) == 1
    low_level = runner.requests[0]
    assert low_level.call_id == request.call_id
    assert low_level.operation == request.operation
    assert low_level.output_tool_name == ACTION_FORECAST_TOOL_NAME
    assert low_level.max_output_tokens == request.max_output_tokens
    assert low_level.temperature == request.temperature
    assert [value.option_id for value in result.forecasts.forecasts] == [
        option.option_id for option in request.finite_variation_contract.options
    ]
    assert all(
        tuple(metric.metric_id for metric in forecast.metric_forecasts)
        == request.required_metric_ids
        for forecast in result.forecasts.forecasts
    )
    assert all(
        len(metric.citations) == 1
        for forecast in result.forecasts.forecasts
        for metric in forecast.metric_forecasts
    )
    assert result.forecasts.policy_id == ACTION_FORECAST_POLICY_ID
    assert result.forecasts.policy_version == ACTION_FORECAST_POLICY_VERSION == 5
    assert result.telemetry is not None
    assert result.telemetry.attempt_count == 4
    assert result.telemetry.output_tokens == 700

    effect_rows = payload["median_effect_codes"]
    for row_index, resolved in enumerate(result.forecasts.forecasts):
        for metric_index, metric in enumerate(resolved.metric_forecasts):
            scale = request.metric_scales[metric_index].delta_scale
            effect_code = effect_rows[row_index][metric_index]
            lower_floor, upper_floor = _effect_quantization_floors(effect_code)
            median = _EFFECT_UNITS[effect_code] * scale
            assert metric.p10_delta == median - (lower_floor + 0.5) * scale
            assert metric.p50_delta == median
            assert metric.p90_delta == median + (upper_floor + 1.0) * scale
            assert metric.confidence == 0.4


def test_public_planner_is_provider_free_and_equals_runtime_dispatch() -> None:
    request = _request()
    planned = plan_action_forecast_request(request)
    planned_again = plan_action_forecast_request(request)
    assert planned == planned_again
    assert planned.output_type is planned_again.output_type

    payload = _wire_payload(request)

    def handle(low_level: StructuredGenerationRequest[Any]):
        value = low_level.output_type.model_validate(payload, strict=True)
        return _response(value)

    runner = _FakeRunner(handle)
    asyncio.run(PydanticAIActionForecastPolicy(runner).forecast(request))

    assert runner.requests == [planned]
    assert runner.requests[0].output_type is planned.output_type
    assert runner.requests[0].prompt == render_action_forecast_prompt(request)


def test_block_adapter_emits_only_its_hash_bound_slice_and_partial_receipt() -> None:
    request = _request()
    partition_policy = ActionForecastPartitionPolicyBinding(
        policy_id="fixture_contiguous_partition",
        policy_version=1,
        policy_definition_sha256="a" * 64,
        max_rows_per_block=2,
        max_metric_cells_per_block=4,
    )
    layout = build_action_forecast_partition_layout(request, partition_policy)
    block_request = build_action_forecast_block_requests(request, layout)[1]
    planned = plan_action_forecast_block_request(block_request)
    payload = _wire_payload(request, option_count=block_request.block.row_count)

    def handle(low_level: StructuredGenerationRequest[Any]):
        value = low_level.output_type.model_validate(payload, strict=True)
        return AttemptedStructuredGenerationResponse(
            response=_response(value),
            attempt_count=2,
        )

    runner = _FakeRunner(handle)
    result = asyncio.run(
        PydanticAIActionForecastBlockPolicy(runner).forecast_block(block_request)
    )

    assert runner.requests == [planned]
    assert planned.call_id == block_request.block_call_id
    assert planned.output_tool_name == ACTION_FORECAST_BLOCK_TOOL_NAME
    assert planned.max_output_tokens == request.max_output_tokens
    schema = planned.output_type.model_json_schema()
    assert schema["properties"]["probability_valid_codes"]["minItems"] == 2
    assert schema["properties"]["median_effect_codes"]["maxItems"] == 2
    contract = _machine_contract(render_action_forecast_block_prompt(block_request))
    assert contract["forecast_frame"] == {
        "frame_kind": "partition_block",
        "global_option_count": 4,
        "global_row_start": 2,
        "global_row_stop": 4,
        "local_row_count": 2,
        "layout_sha256": layout.layout_sha256,
        "block_index": 1,
        "block_spec_sha256": block_request.block.block_spec_sha256,
        "block_request_sha256": block_request.block_request_sha256,
    }
    assert [value["row_index"] for value in contract["ordered_options"]] == [0, 1]
    assert [
        value["global_row_index"] for value in contract["ordered_options"]
    ] == [2, 3]
    prompt = render_action_forecast_block_prompt(block_request)
    assert "Forecast exactly and only the local_row_count ordered_options" in prompt
    assert "do not emit rows for absent global options" in prompt
    assert [value.option_id for value in result.forecasts.forecasts] == [
        "action.c",
        "action.d",
    ]
    assert result.forecasts.request_sha256 == request.request_sha256
    assert result.forecasts.layout_sha256 == layout.layout_sha256
    assert result.forecasts.block_index == 1
    assert result.telemetry is not None
    assert result.telemetry.attempt_count == 2


def test_v5_json_schema_is_exact_enum_only_positional_matrix_contract() -> None:
    request = _request()
    output_type = plan_action_forecast_request(request).output_type
    schema = output_type.model_json_schema()
    properties = schema["properties"]
    assert set(properties) == {
        "probability_valid_codes",
        "median_effect_codes",
        "lower_uncertainty_codes",
        "upper_uncertainty_codes",
        "evidence_slot_codes",
    }
    option_count = len(request.finite_variation_contract.options)
    metric_count = len(request.required_metric_ids)
    for name, field_schema in properties.items():
        assert field_schema["minItems"] == field_schema["maxItems"] == option_count
        if name != "probability_valid_codes":
            assert field_schema["items"]["minItems"] == metric_count
            assert field_schema["items"]["maxItems"] == metric_count

    assert properties["probability_valid_codes"]["items"]["enum"] == list(
        _VALIDITY_VALUES
    )
    assert properties["median_effect_codes"]["items"]["items"]["enum"] == list(
        _EFFECT_UNITS
    )
    for name in ("lower_uncertainty_codes", "upper_uncertainty_codes"):
        assert properties[name]["items"]["items"]["enum"] == list(
            _UNCERTAINTY_UNITS
        )
    assert properties["evidence_slot_codes"]["items"]["items"]["enum"] == [
        f"e{index}" for index in range(len(_evidence_slot_by_pair(request)))
    ]

    encoded_schema = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    assert '"type":"number"' not in encoded_schema
    assert '"type":"integer"' not in encoded_schema
    assert '"maximum"' not in encoded_schema
    assert '"minimum"' not in encoded_schema
    assert not any(
        option.option_id in encoded_schema
        for option in request.finite_variation_contract.options
    )
    assert not any(
        metric_id in encoded_schema for metric_id in request.required_metric_ids
    )
    assert not output_type.__pydantic_decorators__.model_validators


def test_v5_schema_rejects_shape_code_slot_and_legacy_field_violations() -> None:
    request = _request()
    output_type = plan_action_forecast_request(request).output_type
    valid = _wire_payload(request)

    cases: list[dict[str, object]] = []
    missing_validity = copy.deepcopy(valid)
    missing_validity["probability_valid_codes"] = missing_validity[
        "probability_valid_codes"
    ][:-1]
    cases.append(missing_validity)
    extra_validity = copy.deepcopy(valid)
    extra_validity["probability_valid_codes"].append("p0_8")
    cases.append(extra_validity)
    missing_row = copy.deepcopy(valid)
    missing_row["median_effect_codes"] = missing_row["median_effect_codes"][:-1]
    cases.append(missing_row)
    extra_row = copy.deepcopy(valid)
    extra_row["upper_uncertainty_codes"].append(
        extra_row["upper_uncertainty_codes"][-1]
    )
    cases.append(extra_row)
    missing_cell = copy.deepcopy(valid)
    missing_cell["lower_uncertainty_codes"][0] = missing_cell[
        "lower_uncertainty_codes"
    ][0][:-1]
    cases.append(missing_cell)
    extra_cell = copy.deepcopy(valid)
    extra_cell["evidence_slot_codes"][0].append("e0")
    cases.append(extra_cell)
    unknown_effect = copy.deepcopy(valid)
    unknown_effect["median_effect_codes"][0][0] = "positive_huge"
    cases.append(unknown_effect)
    numeric_effect = copy.deepcopy(valid)
    numeric_effect["median_effect_codes"][0][0] = 1
    cases.append(numeric_effect)
    boolean_validity = copy.deepcopy(valid)
    boolean_validity["probability_valid_codes"][0] = True
    cases.append(boolean_validity)
    missing_slots = copy.deepcopy(valid)
    del missing_slots["evidence_slot_codes"]
    cases.append(missing_slots)
    foreign_slot = copy.deepcopy(valid)
    foreign_slot["evidence_slot_codes"][0][0] = "e999"
    cases.append(foreign_slot)
    legacy_field = copy.deepcopy(valid)
    legacy_field["action_rows"] = []
    cases.append(legacy_field)

    for malformed in cases:
        with pytest.raises(ValidationError):
            output_type.model_validate(malformed, strict=True)


def test_every_enum_code_is_admitted_and_resolves_to_finite_ordered_quantiles() -> None:
    request = _request()
    effects = tuple(_EFFECT_UNITS)
    uncertainties = tuple(_UNCERTAINTY_UNITS)
    validities = tuple(_VALIDITY_VALUES)
    pair_by_slot = {
        slot: pair for pair, slot in _evidence_slot_by_pair(request).items()
    }

    for index in range(max(len(effects), len(uncertainties), len(validities))):
        effect_code = effects[index % len(effects)]
        lower_code = uncertainties[index % len(uncertainties)]
        upper_code = uncertainties[-1 - (index % len(uncertainties))]
        validity_code = validities[index % len(validities)]
        payload = _wire_payload(
            request,
            effect_code=effect_code,
            lower_code=lower_code,
            upper_code=upper_code,
            validity_code=validity_code,
        )
        _, result = _run_payload(request, payload)
        for row_index, forecast in enumerate(result.forecasts.forecasts):
            assert forecast.probability_valid == _VALIDITY_VALUES[validity_code]
            for metric_index, metric in enumerate(forecast.metric_forecasts):
                scale = request.metric_scales[metric_index].delta_scale
                lower_floor, upper_floor = _effect_quantization_floors(
                    effect_code
                )
                expected_p50 = _EFFECT_UNITS[effect_code] * scale
                expected_p10 = expected_p50 - (
                    lower_floor + _UNCERTAINTY_UNITS[lower_code]
                ) * scale
                expected_p90 = expected_p50 + (
                    upper_floor + _UNCERTAINTY_UNITS[upper_code]
                ) * scale
                assert all(
                    math.isfinite(value)
                    for value in (metric.p10_delta, metric.p50_delta, metric.p90_delta)
                )
                assert metric.p10_delta <= metric.p50_delta <= metric.p90_delta
                assert metric.p10_delta == expected_p10
                assert metric.p50_delta == expected_p50
                assert metric.p90_delta == expected_p90
                assert metric.confidence == 1.0 / (
                    1.0
                    + _UNCERTAINTY_UNITS[lower_code]
                    + _UNCERTAINTY_UNITS[upper_code]
                )
                expected_pair = pair_by_slot[
                    payload["evidence_slot_codes"][row_index][metric_index]
                ]
                citation = metric.citations[0]
                assert (
                    citation.card_key,
                    citation.action_binding_identity_sha256,
                ) == expected_pair


@pytest.mark.parametrize(
    ("effect_code", "lower_code", "upper_code", "expected_units", "confidence"),
    (
        ("p4", "u0_5", "u0_5", (2.5, 4.0, 6.5), 0.5),
        ("z", "u0", "u0", (-0.125, 0.0, 0.125), 1.0),
        ("n32", "u0", "u0", (-48.0, -32.0, -24.0), 1.0),
        ("p32", "u0", "u0", (24.0, 32.0, 48.0), 1.0),
    ),
)
def test_v5_adds_asymmetric_quantization_floors_beyond_excess_uncertainty(
    effect_code: str,
    lower_code: str,
    upper_code: str,
    expected_units: tuple[float, float, float],
    confidence: float,
) -> None:
    request = _request()
    payload = _wire_payload(
        request,
        effect_code=effect_code,
        lower_code=lower_code,
        upper_code=upper_code,
    )
    _, result = _run_payload(request, payload)

    metric = result.forecasts.forecasts[0].metric_forecasts[0]
    scale = request.metric_scales[0].delta_scale
    assert (
        metric.p10_delta / scale,
        metric.p50_delta / scale,
        metric.p90_delta / scale,
    ) == pytest.approx(expected_units)
    assert metric.confidence == confidence


def test_explicit_v4_compatibility_preserves_sealed_decoding_and_contract() -> None:
    request = _request()
    payload = _wire_payload(
        request,
        effect_code="p4",
        lower_code="u0_5",
        upper_code="u0_5",
    )

    def handle(low_level: StructuredGenerationRequest[Any]):
        value = low_level.output_type.model_validate(payload, strict=True)
        return _response(value)

    runner = _FakeRunner(handle)
    result = asyncio.run(PydanticAIActionForecastV4Policy(runner).forecast(request))
    planned_v4 = plan_action_forecast_v4_request(request)
    planned_v5 = plan_action_forecast_request(request)

    assert runner.requests == [planned_v4]
    assert planned_v4.output_type.__name__ == "AllOptionActionForecastMatrixV4"
    assert planned_v5.output_type.__name__ == "AllOptionActionForecastMatrixV5"
    assert _machine_contract(render_action_forecast_v4_prompt(request))[
        "schema_version"
    ] == 5
    assert "effect_quantization" not in _machine_contract(
        render_action_forecast_v4_prompt(request)
    )
    assert result.forecasts.policy_version == ACTION_FORECAST_V4_POLICY_VERSION == 4
    assert result.forecasts.policy_definition_sha256 == (
        ACTION_FORECAST_V4_POLICY_DEFINITION_SHA256
    )
    assert ACTION_FORECAST_V4_POLICY_DEFINITION_SHA256 == (
        "79cf864675cb9500062ecd86ce591c637adb3f0ec1e980576f212a47d3ad070a"
    )
    metric = result.forecasts.forecasts[0].metric_forecasts[0]
    scale = request.metric_scales[0].delta_scale
    assert (
        metric.p10_delta / scale,
        metric.p50_delta / scale,
        metric.p90_delta / scale,
    ) == pytest.approx((3.5, 4.0, 4.5))
    assert metric.confidence == 0.5

    partition_policy = ActionForecastPartitionPolicyBinding(
        policy_id="fixture_v4_compatibility_partition",
        policy_version=1,
        policy_definition_sha256="b" * 64,
        max_rows_per_block=2,
        max_metric_cells_per_block=4,
    )
    layout = build_action_forecast_partition_layout(request, partition_policy)
    block_request = build_action_forecast_block_requests(request, layout)[0]
    block_payload = _wire_payload(request, option_count=2, effect_code="p4")

    def handle_block(low_level: StructuredGenerationRequest[Any]):
        value = low_level.output_type.model_validate(block_payload, strict=True)
        return _response(value)

    block_runner = _FakeRunner(handle_block)
    block_result = asyncio.run(
        PydanticAIActionForecastV4BlockPolicy(block_runner).forecast_block(
            block_request
        )
    )
    assert block_runner.requests == [
        plan_action_forecast_v4_block_request(block_request)
    ]
    assert block_result.forecasts.policy_version == 4
    assert block_result.forecasts.policy_definition_sha256 == (
        ACTION_FORECAST_V4_POLICY_DEFINITION_SHA256
    )

    v4_properties = planned_v4.output_type.model_json_schema()["properties"]
    v5_properties = planned_v5.output_type.model_json_schema()["properties"]
    assert set(v4_properties) == set(v5_properties)
    for field_name in v4_properties:
        assert v4_properties[field_name]["type"] == v5_properties[field_name]["type"]
        assert v4_properties[field_name]["minItems"] == (
            v5_properties[field_name]["minItems"]
        )
        assert v4_properties[field_name]["maxItems"] == (
            v5_properties[field_name]["maxItems"]
        )
        if field_name == "probability_valid_codes":
            assert v4_properties[field_name]["items"]["enum"] == (
                v5_properties[field_name]["items"]["enum"]
            )
        else:
            assert v4_properties[field_name]["items"]["items"]["enum"] == (
                v5_properties[field_name]["items"]["items"]["enum"]
            )


def test_catalog_only_schema_omits_evidence_matrix_and_resolves_without_citations() -> None:
    grounded = _request()
    request = replace(
        grounded,
        cards=(),
        source_registry=None,
        evidence_mode=ActionForecastEvidenceMode.CATALOG_ONLY,
        experimental_view_receipt=None,
    )
    planned = plan_action_forecast_request(request)
    properties = planned.output_type.model_json_schema()["properties"]
    assert "evidence_slot_codes" not in properties

    payload = _wire_payload(request)
    with_slot = copy.deepcopy(payload)
    with_slot["evidence_slot_codes"] = [
        ["e0"] * len(request.required_metric_ids)
        for _ in request.finite_variation_contract.options
    ]
    with pytest.raises(ValidationError):
        planned.output_type.model_validate(with_slot, strict=True)

    runner, result = _run_payload(request, payload)
    contract = _machine_contract(runner.requests[0].prompt)
    assert contract["evidence_mode"] == "catalog_only"
    assert contract["cards"] == []
    assert contract["evidence_slots"] == []
    assert result.telemetry is not None
    assert result.telemetry.attempt_count == 1
    assert all(
        not metric.citations
        for forecast in result.forecasts.forecasts
        for metric in forecast.metric_forecasts
    )


def test_prompt_contains_codebook_index_and_slot_manifest_without_provenance() -> None:
    request = _request()
    prompt = render_action_forecast_prompt(request)
    contract = _machine_contract(prompt)

    assert contract["schema_version"] == 6
    assert contract["request_sha256"] == request.request_sha256
    assert contract["optimization_semantics"] == (
        request.optimization_semantics.to_record()
    )
    assert contract["action_semantics"] == request.action_semantics.to_record()
    assert contract["ordered_options"] == [
        {
            "row_index": index,
            "global_row_index": index,
            **option.prompt_record(),
        }
        for index, option in enumerate(request.finite_variation_contract.options)
    ]
    assert contract["forecast_frame"] == {
        "frame_kind": "complete",
        "global_option_count": 4,
        "global_row_start": 0,
        "global_row_stop": 4,
        "local_row_count": 4,
    }
    assert contract["cards"] == [card.prompt_record() for card in request.cards]
    assert contract["forecast_metrics"] == [
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
    ]
    assert contract["evidence_slots"] == [
        {
            "evidence_slot_id": slot_id,
            "card_key": card_key,
            "action_binding_identity_sha256": binding_sha256,
        }
        for (card_key, binding_sha256), slot_id in sorted(
            _evidence_slot_by_pair(request).items(),
            key=lambda value: value[1],
        )
    ]
    assert set(contract["ordinal_codebook"]["median_effect_codes"]) == set(
        _EFFECT_UNITS
    )
    assert set(contract["ordinal_codebook"]["uncertainty_codes"]) == set(
        _UNCERTAINTY_UNITS
    )
    assert set(contract["ordinal_codebook"]["probability_valid_codes"]) == set(
        _VALIDITY_VALUES
    )
    assert contract["effect_quantization"] == {
        "rule": "adjacent_midpoints_with_virtual_endpoint_centers",
        "virtual_lower_effect_center_units": -64.0,
        "virtual_upper_effect_center_units": 64.0,
        "asymmetric_floor_units_by_effect_code": {
            code: {
                "lower_floor_units": _effect_quantization_floors(code)[0],
                "upper_floor_units": _effect_quantization_floors(code)[1],
            }
            for code in _EFFECT_UNITS
        },
    }
    assert contract["effect_quantization"][
        "asymmetric_floor_units_by_effect_code"
    ]["n32"] == {"lower_floor_units": 16.0, "upper_floor_units": 8.0}
    assert contract["effect_quantization"][
        "asymmetric_floor_units_by_effect_code"
    ]["p32"] == {"lower_floor_units": 8.0, "upper_floor_units": 16.0}
    assert contract["effect_quantization"][
        "asymmetric_floor_units_by_effect_code"
    ]["z"] == {"lower_floor_units": 0.125, "upper_floor_units": 0.125}
    assert "excess epistemic" in contract["ordinal_codebook"][
        "uncertainty_codes"
    ]["u0_5"]
    assert "excluding fixed quantization floors" in contract[
        "ordinal_codebook"
    ]["derived_confidence"]
    assert "lower quantization-floor units" in contract["output_contract"][
        "quantile_derivation"
    ]["p10_delta"]
    assert contract["output_contract"]["provider_numeric_output_fields"] == []
    assert contract["output_contract"]["delta_definition"] == (
        "child_metric_minus_parent_metric"
    )
    assert "Do not emit any numeric forecast or confidence field." in prompt
    assert "4.4942328371557893e+307" not in prompt
    assert "Choose fixture coordinate 1." in prompt
    assert "Fixture resource cost." in prompt
    assert request.cards[0].finite_action_evidence[0].identity_sha256 in prompt
    assert ACTION_FORECAST_POLICY_VERSION == 5
    assert len(ACTION_FORECAST_POLICY_DEFINITION_SHA256) == 64
    assert ACTION_FORECAST_POLICY_DEFINITION_SHA256 != (
        ACTION_FORECAST_V4_POLICY_DEFINITION_SHA256
    )

    forbidden_keys = {
        "source_binding",
        "derived_view_receipt",
        "source_registry",
        "source_registry_sha256",
        "source_receipt_sha256",
        "evidence_lineage_identity_sha256",
        "card_source_binding_sha256",
        "child_configuration",
        "child_configuration_sha256",
    }

    def keys(value: object) -> set[str]:
        if type(value) is dict:
            return set(value) | {
                nested for item in value.values() for nested in keys(item)
            }
        if type(value) is list:
            return {nested for item in value for nested in keys(item)}
        return set()

    assert not (keys(contract) & forbidden_keys)
    assert request.source_registry is not None
    assert request.source_registry.registry_sha256 not in prompt
