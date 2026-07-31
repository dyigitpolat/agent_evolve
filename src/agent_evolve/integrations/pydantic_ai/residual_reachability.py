"""Pydantic-AI adapter for compact hierarchical residual proposals.

The model selects a parent and one or two opaque atomic option IDs.  The
application-layer reachability schema owns eligibility and trusted engine code
materializes the resulting configuration after this adapter returns.  Queueing,
retry, backoff, and model routing remain injected through ``LowLevelRunner``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Annotated, Any, ClassVar, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    create_model,
    model_validator,
)
from pydantic_core import PydanticCustomError

from agent_evolve.application.residual_reachability import (
    CrossParentFiniteActionSchema,
    HierarchicalResidualPlan,
    ResidualProposalRole,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.llm_task_queue import ValidationIssueReasonCode
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
    LowLevelRunner,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry
from agent_evolve.ports.portfolio_selection import (
    pairwise_disjoint_parent_patch_pairs,
)
from agent_evolve.ports.structured_generator import (
    MAX_OUTPUT_TOKENS,
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)


HIERARCHICAL_RESIDUAL_PROPOSAL_TOOL_NAME = "propose_residual_plans"
HIERARCHICAL_RESIDUAL_PROPOSAL_POLICY_ID = (
    "pydantic_ai_hierarchical_residual_proposal"
)
HIERARCHICAL_RESIDUAL_PROPOSAL_POLICY_VERSION = 2
HIERARCHICAL_RESIDUAL_PROPOSAL_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-hierarchical-residual-proposal:v2;"
    b"model-output=parent-id-plus-radius-one-or-two-atomic-id-tuple;"
    b"consequence=p10-p50-p90-parent-relative-metric-deltas-plus-confidence-"
    b"and-probability-valid;"
    b"engine-materialization-only=true;cross-parent-schema=true;"
    b"workload-model-provider-branches=false"
).hexdigest()

_STRICT = ConfigDict(
    extra="forbid",
    strict=True,
    populate_by_name=True,
    validate_default=True,
)
_OPERATION = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_Rationale = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=16_384,
    ),
]
_DECISION_DOMAIN = b"agent-evolve:hierarchical-residual-proposal-decision:v2\x00"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(value: object) -> str:
    return hashlib.sha256(_DECISION_DOMAIN + _canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class HierarchicalResidualProposalRequest:
    """Provider-neutral request for one proposal expert."""

    call_id: LLMCallId
    operation: str
    instruction: str
    context: FrozenJsonObject
    action_schema: CrossParentFiniteActionSchema
    proposal_count: int
    allowed_radii: tuple[int, ...]
    allowed_roles: tuple[ResidualProposalRole, ...]
    required_metric_ids: tuple[str, ...]
    minimum_distinct_parents: int
    expert_id: str
    expert_definition_sha256: str
    max_output_tokens: int
    temperature: float

    def __post_init__(self) -> None:
        if type(self.call_id) is not LLMCallId:
            raise TypeError("call_id must be an exact LLMCallId")
        LLMCallId.__post_init__(self.call_id)
        if type(self.operation) is not str or _OPERATION.fullmatch(self.operation) is None:
            raise ValueError("operation must use the closed token grammar")
        if type(self.instruction) is not str or not self.instruction.strip():
            raise ValueError("instruction must be non-empty")
        if len(self.instruction.encode("utf-8")) > 32_768:
            raise ValueError("instruction exceeds its byte limit")
        if type(self.context) is not FrozenJsonObject:
            raise TypeError("context must be an exact FrozenJsonObject")
        if freeze_json(self.context) is not self.context:
            raise TypeError("context must already be frozen typed JSON")
        if type(self.action_schema) is not CrossParentFiniteActionSchema:
            raise TypeError("action_schema must be exact")
        self.action_schema.__post_init__()
        if type(self.proposal_count) is not int or not 1 <= self.proposal_count <= 32:
            raise ValueError("proposal_count must lie in [1, 32]")
        if (
            type(self.allowed_radii) is not tuple
            or not self.allowed_radii
            or self.allowed_radii != tuple(sorted(set(self.allowed_radii)))
            or not set(self.allowed_radii).issubset({1, 2})
        ):
            raise ValueError("allowed_radii must be a canonical subset of {1, 2}")
        if (
            type(self.allowed_roles) is not tuple
            or not self.allowed_roles
            or any(type(value) is not ResidualProposalRole for value in self.allowed_roles)
        ):
            raise TypeError("allowed_roles must contain exact role values")
        if self.allowed_roles != tuple(
            sorted(set(self.allowed_roles), key=lambda value: value.value)
        ):
            raise ValueError("allowed_roles must be unique and canonical")
        if (
            type(self.required_metric_ids) is not tuple
            or not self.required_metric_ids
            or self.required_metric_ids != tuple(sorted(set(self.required_metric_ids)))
        ):
            raise ValueError("required_metric_ids must be non-empty and canonical")
        if (
            type(self.minimum_distinct_parents) is not int
            or not 1 <= self.minimum_distinct_parents <= self.proposal_count
            or self.minimum_distinct_parents > len(self.action_schema.bindings)
        ):
            raise ValueError("minimum_distinct_parents is infeasible")
        if type(self.expert_id) is not str or _OPERATION.fullmatch(self.expert_id) is None:
            raise ValueError("expert_id must use the closed token grammar")
        require_sha256(self.expert_definition_sha256, "expert_definition_sha256")
        if (
            type(self.max_output_tokens) is not int
            or not 1 <= self.max_output_tokens <= MAX_OUTPUT_TOKENS
        ):
            raise ValueError("max_output_tokens is outside the generic bound")
        if type(self.temperature) is not float or not math.isfinite(self.temperature):
            raise TypeError("temperature must be a finite exact float")

    @property
    def request_sha256(self) -> str:
        self.__post_init__()
        return _hash(
            {
                "schema_version": 1,
                "call_id": self.call_id.value,
                "operation": self.operation,
                "instruction": self.instruction,
                "context": thaw_json(self.context),
                "action_schema_sha256": self.action_schema.schema_sha256,
                "proposal_count": self.proposal_count,
                "allowed_radii": list(self.allowed_radii),
                "allowed_roles": [value.value for value in self.allowed_roles],
                "required_metric_ids": list(self.required_metric_ids),
                "minimum_distinct_parents": self.minimum_distinct_parents,
                "expert_id": self.expert_id,
                "expert_definition_sha256": self.expert_definition_sha256,
                "max_output_tokens": self.max_output_tokens,
                "temperature_hex": self.temperature.hex(),
            }
        )


class _ResidualMemberBase(BaseModel):
    model_config = _STRICT

    available_option_ids_by_parent: ClassVar[dict[str, frozenset[str]]] = {}
    compatible_radius_two_by_parent: ClassVar[
        dict[str, frozenset[tuple[str, str]]]
    ] = {}
    allowed_radii: ClassVar[frozenset[int]] = frozenset()
    required_metric_ids: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="after")
    def _validate_member(self) -> "_ResidualMemberBase":
        parent_id = cast(Any, self).parent_candidate_id
        option_ids = tuple(cast(Any, self).component_option_ids)
        if len(option_ids) not in type(self).allowed_radii:
            raise PydanticCustomError(
                ValidationIssueReasonCode.RESIDUAL_RADIUS_CONTRACT_VIOLATION.value,
                "component tuple has a forbidden radius",
            )
        if len(set(option_ids)) != len(option_ids):
            raise PydanticCustomError(
                ValidationIssueReasonCode.RESIDUAL_OPTION_CONTRACT_VIOLATION.value,
                "component option IDs cannot repeat",
            )
        if not set(option_ids).issubset(
            type(self).available_option_ids_by_parent[parent_id]
        ):
            raise PydanticCustomError(
                ValidationIssueReasonCode.RESIDUAL_OPTION_CONTRACT_VIOLATION.value,
                "a component is unavailable for the selected parent",
            )
        if len(option_ids) == 2 and tuple(sorted(option_ids)) not in type(
            self
        ).compatible_radius_two_by_parent[parent_id]:
            raise PydanticCustomError(
                ValidationIssueReasonCode.RESIDUAL_OPTION_CONTRACT_VIOLATION.value,
                "radius-two components are not a safe disjoint "
                "parent-relative pair",
            )
        metrics = tuple(item.metric_id for item in cast(Any, self).effect_predictions)
        if len(set(metrics)) != len(metrics) or set(metrics) != type(self).required_metric_ids:
            raise PydanticCustomError(
                ValidationIssueReasonCode.RESIDUAL_METRIC_CONTRACT_VIOLATION.value,
                "effect predictions must cover each requested metric once",
            )
        return self


class _ResidualMetricPredictionBase(BaseModel):
    model_config = _STRICT

    @model_validator(mode="after")
    def _validate_quantiles(self) -> "_ResidualMetricPredictionBase":
        p10 = cast(float, cast(Any, self).p10_delta)
        p50 = cast(float, cast(Any, self).p50_delta)
        p90 = cast(float, cast(Any, self).p90_delta)
        if not all(math.isfinite(value) for value in (p10, p50, p90)):
            raise PydanticCustomError(
                ValidationIssueReasonCode.RESIDUAL_QUANTILE_ORDER_VIOLATION.value,
                "effect quantiles must be finite",
            )
        if not p10 <= p50 <= p90:
            raise PydanticCustomError(
                ValidationIssueReasonCode.RESIDUAL_QUANTILE_ORDER_VIOLATION.value,
                "effect quantiles must satisfy p10 <= p50 <= p90",
            )
        return self


class _ResidualOutputBase(BaseModel):
    model_config = _STRICT

    proposal_count: ClassVar[int] = 1
    minimum_distinct_parents: ClassVar[int] = 1

    @model_validator(mode="after")
    def _validate_output(self) -> "_ResidualOutputBase":
        members = tuple(cast(Any, self).members)
        if len(members) != type(self).proposal_count:
            raise ValueError("members must have the exact requested size")
        identities = tuple(
            (value.parent_candidate_id, tuple(sorted(value.component_option_ids)))
            for value in members
        )
        if len(set(identities)) != len(identities):
            raise PydanticCustomError(
                ValidationIssueReasonCode.RESIDUAL_PLAN_DIVERSITY_VIOLATION.value,
                "members cannot repeat a parent-relative plan",
            )
        if len({value.parent_candidate_id for value in members}) < type(self).minimum_distinct_parents:
            raise PydanticCustomError(
                ValidationIssueReasonCode.RESIDUAL_PLAN_DIVERSITY_VIOLATION.value,
                "members do not cover enough distinct parents",
            )
        return self


def hierarchical_residual_output_type(
    request: HierarchicalResidualProposalRequest,
) -> type[BaseModel]:
    request.__post_init__()
    parent_ids = tuple(
        value.parent_candidate_id.value for value in request.action_schema.bindings
    )
    option_ids = request.action_schema.option_ids
    role_ids = tuple(value.value for value in request.allowed_roles)
    metric_ids = request.required_metric_ids
    parent_literal = Literal.__getitem__(parent_ids)
    option_literal = Literal.__getitem__(option_ids)
    role_literal = Literal.__getitem__(role_ids)
    metric_literal = Literal.__getitem__(metric_ids)
    prediction_type = create_model(
        "ResidualMetricPrediction",
        __base__=_ResidualMetricPredictionBase,
        __module__=__name__,
        metric_id=(metric_literal, ...),
        p10_delta=(float, ...),
        p50_delta=(float, ...),
        p90_delta=(float, ...),
        confidence=(float, Field(ge=0.0, le=1.0, strict=True)),
    )
    member_type = create_model(
        "HierarchicalResidualMember",
        __base__=_ResidualMemberBase,
        __module__=__name__,
        parent_candidate_id=(parent_literal, ...),
        component_option_ids=(
            list[option_literal],
            Field(min_length=min(request.allowed_radii), max_length=max(request.allowed_radii)),
        ),
        role=(role_literal, ...),
        probability_valid=(float, Field(ge=0.0, le=1.0, strict=True)),
        effect_predictions=(
            list[prediction_type],
            Field(min_length=len(metric_ids), max_length=len(metric_ids)),
        ),
        interaction_rationale=(_Rationale, ...),
    )
    member_type.available_option_ids_by_parent = {
        value.parent_candidate_id.value: frozenset(
            option.option_id for option in value.contract.options
        )
        for value in request.action_schema.bindings
    }
    member_type.compatible_radius_two_by_parent = {
        value.parent_candidate_id.value: (
            frozenset(
                pairwise_disjoint_parent_patch_pairs(
                    value.contract,
                    tuple(option.option_id for option in value.contract.options),
                )
            )
            if 2 in request.allowed_radii
            else frozenset()
        )
        for value in request.action_schema.bindings
    }
    member_type.allowed_radii = frozenset(request.allowed_radii)
    member_type.required_metric_ids = frozenset(metric_ids)
    output_type = create_model(
        "HierarchicalResidualProposalSlate",
        __base__=_ResidualOutputBase,
        __module__=__name__,
        members=(
            list[member_type],
            Field(min_length=request.proposal_count, max_length=request.proposal_count),
        ),
        slate_rationale=(_Rationale, ...),
    )
    output_type.proposal_count = request.proposal_count
    output_type.minimum_distinct_parents = request.minimum_distinct_parents
    return output_type


def render_hierarchical_residual_prompt(
    request: HierarchicalResidualProposalRequest,
) -> str:
    request.__post_init__()
    contract = {
        "schema_version": 1,
        "request_sha256": request.request_sha256,
        "expert_id": request.expert_id,
        "context": thaw_json(request.context),
        "cross_parent_action_schema": request.action_schema.prompt_record(),
        "constraints": {
            "proposal_count": request.proposal_count,
            "allowed_radii": list(request.allowed_radii),
            "allowed_roles": [value.value for value in request.allowed_roles],
            "required_metric_ids": list(request.required_metric_ids),
            "minimum_distinct_parents": request.minimum_distinct_parents,
            "distinct_parent_relative_plans": True,
            "select_only_available_option_ids_for_parent": True,
            "do_not_return_configurations": True,
            "effect_forecast_contract": {
                "deltas_are_parent_relative_in_raw_metric_units": True,
                "quantiles": ["p10_delta", "p50_delta", "p90_delta"],
                "ordered_quantiles_required": True,
                "confidence_and_probability_valid_range": [0.0, 1.0],
                "future_evaluator_outcomes_are_unknown": True,
            },
        },
    }
    return "\n".join(
        (
            request.instruction,
            "",
            "HIERARCHICAL RESIDUAL PROPOSAL CONTRACT",
            json.dumps(
                contract,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            "Return exactly the requested best-first proposal slate. Select only "
            "a supplied parent ID and atomic option IDs available for that parent. "
            "The trusted engine will materialize and validate each tuple. Forecast "
            "parent-relative p10/p50/p90 metric deltas in the raw metric units; "
            "do not collapse magnitude into direction labels.",
        )
    )


@dataclass(frozen=True, slots=True)
class HierarchicalResidualMetricForecast:
    """Provider-neutral magnitude forecast for one objective metric."""

    metric_id: str
    p10_delta: float
    p50_delta: float
    p90_delta: float
    confidence: float

    def __post_init__(self) -> None:
        if type(self.metric_id) is not str or not self.metric_id:
            raise ValueError("metric_id must be non-empty")
        for name in ("p10_delta", "p50_delta", "p90_delta", "confidence"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite exact float")
        if not self.p10_delta <= self.p50_delta <= self.p90_delta:
            raise ValueError("forecast quantiles must be ordered")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0,1]")

    @property
    def direction(self) -> str:
        self.__post_init__()
        if self.p10_delta == self.p50_delta == self.p90_delta == 0.0:
            return "unchanged"
        if self.p90_delta < 0.0:
            return "decrease"
        if self.p10_delta > 0.0:
            return "increase"
        return "unknown"

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "p10_delta_hex": self.p10_delta.hex(),
            "p50_delta_hex": self.p50_delta.hex(),
            "p90_delta_hex": self.p90_delta.hex(),
            "confidence_hex": self.confidence.hex(),
            "direction": self.direction,
        }


@dataclass(frozen=True, slots=True)
class HierarchicalResidualProposalSelection:
    request_sha256: str
    decision_sha256: str
    plans: tuple[HierarchicalResidualPlan, ...]
    rationales: tuple[str, ...]
    probability_valid: tuple[float, ...]
    effect_predictions: tuple[
        tuple[HierarchicalResidualMetricForecast, ...],
        ...,
    ]
    slate_rationale: str
    telemetry: AgenticCallTelemetry

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.decision_sha256, "decision_sha256")
        if type(self.plans) is not tuple or not self.plans:
            raise ValueError("plans must be a non-empty exact tuple")
        if any(type(value) is not HierarchicalResidualPlan for value in self.plans):
            raise TypeError("plans contain a foreign value")
        for value in self.plans:
            HierarchicalResidualPlan.__post_init__(value)
        if type(self.rationales) is not tuple or len(self.rationales) != len(self.plans):
            raise ValueError("rationales must align with plans")
        if (
            type(self.probability_valid) is not tuple
            or len(self.probability_valid) != len(self.plans)
        ):
            raise ValueError("probability_valid must align with plans")
        for value in self.probability_valid:
            if type(value) is not float or not 0.0 <= value <= 1.0:
                raise ValueError("probability_valid must contain probabilities")
        if type(self.effect_predictions) is not tuple or len(self.effect_predictions) != len(self.plans):
            raise ValueError("effect_predictions must align with plans")
        for values in self.effect_predictions:
            if type(values) is not tuple or not values:
                raise ValueError("each plan must forecast at least one metric")
            for value in values:
                if type(value) is not HierarchicalResidualMetricForecast:
                    raise TypeError("effect_predictions contain a foreign value")
                value.__post_init__()
            metric_ids = tuple(value.metric_id for value in values)
            if metric_ids != tuple(sorted(set(metric_ids))):
                raise ValueError("effect prediction metrics must be canonical")
        if type(self.slate_rationale) is not str or not self.slate_rationale:
            raise ValueError("slate_rationale must be non-empty")
        if type(self.telemetry) is not AgenticCallTelemetry:
            raise TypeError("telemetry must be exact")
        AgenticCallTelemetry.__post_init__(self.telemetry)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "decision_sha256": self.decision_sha256,
            "plans": [value.to_record() for value in self.plans],
            "rationales": list(self.rationales),
            "probability_valid_hex": [
                value.hex() for value in self.probability_valid
            ],
            "effect_predictions": [
                [value.to_record() for value in values]
                for values in self.effect_predictions
            ],
            "slate_rationale": self.slate_rationale,
        }


def _telemetry(
    response: StructuredGenerationResponse[Any], attempt_count: int
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


@dataclass(slots=True)
class PydanticAIHierarchicalResidualProposalPolicy:
    generate_once: LowLevelRunner

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")

    async def select(
        self,
        request: HierarchicalResidualProposalRequest,
    ) -> HierarchicalResidualProposalSelection:
        request.__post_init__()
        output_type = hierarchical_residual_output_type(request)
        prompt = render_hierarchical_residual_prompt(request)
        raw = await self.generate_once(
            StructuredGenerationRequest(
                call_id=request.call_id,
                operation=request.operation,
                prompt=prompt,
                output_type=output_type,
                output_tool_name=HIERARCHICAL_RESIDUAL_PROPOSAL_TOOL_NAME,
                max_output_tokens=request.max_output_tokens,
                temperature=request.temperature,
            )
        )
        if type(raw) is AttemptedStructuredGenerationResponse:
            response = raw.response
            attempt_count = raw.attempt_count
        elif type(raw) is StructuredGenerationResponse:
            response = raw
            attempt_count = 1
        else:
            raise TypeError("low-level runner returned a foreign response")
        StructuredGenerationResponse.__post_init__(response)
        if type(response.value) is not output_type:
            raise TypeError("response value differs from the requested output type")
        value = cast(Any, response.value)
        draft_record = {
            "schema_version": 1,
            "request_sha256": request.request_sha256,
            "members": [member.model_dump(mode="json") for member in value.members],
            "slate_rationale": value.slate_rationale,
        }
        decision_sha256 = _hash(draft_record)
        plans = tuple(
            HierarchicalResidualPlan(
                parent_candidate_id=next(
                    binding.parent_candidate_id
                    for binding in request.action_schema.bindings
                    if binding.parent_candidate_id.value == member.parent_candidate_id
                ),
                parent_contract_sha256=request.action_schema.contract_for(
                    next(
                        binding.parent_candidate_id
                        for binding in request.action_schema.bindings
                        if binding.parent_candidate_id.value
                        == member.parent_candidate_id
                    )
                ).identity_sha256,
                action_schema_sha256=request.action_schema.schema_sha256,
                component_option_ids=tuple(sorted(member.component_option_ids)),
                role=ResidualProposalRole(member.role),
                expert_id=request.expert_id,
                expert_definition_sha256=request.expert_definition_sha256,
                native_rank=rank,
                decision_receipt_sha256=decision_sha256,
            )
            for rank, member in enumerate(value.members, start=1)
        )
        result = HierarchicalResidualProposalSelection(
            request_sha256=request.request_sha256,
            decision_sha256=decision_sha256,
            plans=plans,
            rationales=tuple(member.interaction_rationale for member in value.members),
            probability_valid=tuple(
                float(member.probability_valid)
                for member in value.members
            ),
            effect_predictions=tuple(
                tuple(
                    sorted(
                        (
                            HierarchicalResidualMetricForecast(
                                metric_id=prediction.metric_id,
                                p10_delta=float(prediction.p10_delta),
                                p50_delta=float(prediction.p50_delta),
                                p90_delta=float(prediction.p90_delta),
                                confidence=float(prediction.confidence),
                            )
                            for prediction in member.effect_predictions
                        ),
                        key=lambda item: item.metric_id,
                    )
                )
                for member in value.members
            ),
            slate_rationale=value.slate_rationale,
            telemetry=_telemetry(response, attempt_count),
        )
        result.__post_init__()
        return result


__all__ = [
    "HIERARCHICAL_RESIDUAL_PROPOSAL_POLICY_DEFINITION_SHA256",
    "HIERARCHICAL_RESIDUAL_PROPOSAL_POLICY_ID",
    "HIERARCHICAL_RESIDUAL_PROPOSAL_POLICY_VERSION",
    "HIERARCHICAL_RESIDUAL_PROPOSAL_TOOL_NAME",
    "HierarchicalResidualProposalRequest",
    "HierarchicalResidualProposalSelection",
    "HierarchicalResidualMetricForecast",
    "PydanticAIHierarchicalResidualProposalPolicy",
    "hierarchical_residual_output_type",
    "render_hierarchical_residual_prompt",
]
