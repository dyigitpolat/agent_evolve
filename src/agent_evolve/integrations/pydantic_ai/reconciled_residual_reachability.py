"""Constraint-by-construction residual proposals with semantic conservation.

Models are useful for proposing semantic preferences, but they should not own
executable validity.  This adapter accepts structurally typed parent/option
preferences, then projects each member independently onto the exact finite
variation contract.  One invalid member therefore cannot discard an otherwise
useful slate.  When projection changes an action, the claims authored for the
rejected action are quarantined and a second identity-preserving call
re-grounds rationale and forecasts against the exact compiled action.  No
workload-, model-, or provider-specific repair branch is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Annotated, Any, Callable, ClassVar, Literal, cast

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
from agent_evolve.domain.finite_variation import FiniteVariationOption
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.llm_task_queue import ValidationIssueReasonCode
from agent_evolve.domain.typed_json import thaw_json
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
    LowLevelRunner,
)
from agent_evolve.integrations.pydantic_ai.residual_reachability import (
    HIERARCHICAL_RESIDUAL_PROPOSAL_TOOL_NAME,
    HierarchicalResidualMetricForecast,
    HierarchicalResidualProposalRequest,
    HierarchicalResidualProposalSelection,
    render_hierarchical_residual_prompt,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry
from agent_evolve.ports.portfolio_selection import (
    pairwise_disjoint_parent_patch_pairs,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)


RECONCILED_RESIDUAL_PROPOSAL_POLICY_ID = (
    "pydantic_ai_semantically_conserved_hierarchical_residual_proposal"
)
RECONCILED_RESIDUAL_PROPOSAL_POLICY_VERSION = 2
RECONCILED_RESIDUAL_PROPOSAL_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:semantically-conserved-hierarchical-residual-proposal:v2;"
    b"model-output=structurally-typed-parent-option-preferences;"
    b"member-local-projection=exact-finite-contract;"
    b"projection-distance=family-metadata-description;"
    b"projected-claims=quarantined;"
    b"postcompile-regrounding=identity-preserving-typed-call;"
    b"regrounding-failure=explicit-uninformative-claims;"
    b"quantile-canonicalization=stable-sort;"
    b"whole-slate-retry-for-local-semantic-invalidity=false;"
    b"workload-model-provider-branches=false"
).hexdigest()

_STRICT = ConfigDict(
    extra="forbid",
    strict=True,
    populate_by_name=True,
    validate_default=True,
)
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
_RECONCILIATION_DOMAIN = (
    b"agent-evolve:residual-proposal-reconciliation:v2\x00"
)
_POSTCOMPILE_DECISION_DOMAIN = (
    b"agent-evolve:postcompile-semantic-regrounding-decision:v1\x00"
)
_POSTCOMPILE_TOOL_NAME = "reground_compiled_residual_semantics"
_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


class _PreferenceMetricPredictionBase(BaseModel):
    model_config = _STRICT

    @model_validator(mode="after")
    def _finite(self) -> "_PreferenceMetricPredictionBase":
        values = tuple(
            cast(float, getattr(cast(Any, self), name))
            for name in (
                "p10_delta",
                "p50_delta",
                "p90_delta",
                "confidence",
            )
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("metric preference values must be finite")
        return self


class _PreferenceMemberBase(BaseModel):
    model_config = _STRICT

    allowed_radii: ClassVar[frozenset[int]] = frozenset()
    required_metric_ids: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="after")
    def _structural_contract(self) -> "_PreferenceMemberBase":
        option_ids = tuple(cast(Any, self).component_option_ids)
        if len(option_ids) not in type(self).allowed_radii:
            raise PydanticCustomError(
                ValidationIssueReasonCode.RESIDUAL_RADIUS_CONTRACT_VIOLATION.value,
                "component tuple has a forbidden radius",
            )
        metrics = tuple(
            item.metric_id for item in cast(Any, self).effect_predictions
        )
        if (
            len(set(metrics)) != len(metrics)
            or set(metrics) != type(self).required_metric_ids
        ):
            raise PydanticCustomError(
                ValidationIssueReasonCode.RESIDUAL_METRIC_CONTRACT_VIOLATION.value,
                "effect predictions must cover each requested metric once",
            )
        return self


class _PreferenceOutputBase(BaseModel):
    model_config = _STRICT

    proposal_count: ClassVar[int] = 1
    minimum_distinct_parents: ClassVar[int] = 1

    @model_validator(mode="after")
    def _slate_contract(self) -> "_PreferenceOutputBase":
        members = tuple(cast(Any, self).members)
        if len(members) != type(self).proposal_count:
            raise ValueError("members must have the exact requested size")
        if (
            len({value.parent_candidate_id for value in members})
            < type(self).minimum_distinct_parents
        ):
            raise PydanticCustomError(
                ValidationIssueReasonCode.RESIDUAL_PLAN_DIVERSITY_VIOLATION.value,
                "members do not cover enough distinct parents",
            )
        return self


class _PostCompilationMemberBase(BaseModel):
    model_config = _STRICT

    required_metric_ids: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="after")
    def _metric_contract(self) -> "_PostCompilationMemberBase":
        metrics = tuple(
            item.metric_id for item in cast(Any, self).effect_predictions
        )
        if (
            len(set(metrics)) != len(metrics)
            or set(metrics) != type(self).required_metric_ids
        ):
            raise PydanticCustomError(
                ValidationIssueReasonCode.RESIDUAL_METRIC_CONTRACT_VIOLATION.value,
                "post-compilation forecasts must cover every metric once",
            )
        return self


class _PostCompilationOutputBase(BaseModel):
    model_config = _STRICT

    required_member_indices: ClassVar[tuple[int, ...]] = ()

    @model_validator(mode="after")
    def _member_contract(self) -> "_PostCompilationOutputBase":
        indices = tuple(
            int(member.member_index)
            for member in cast(Any, self).members
        )
        if indices != type(self).required_member_indices:
            raise ValueError(
                "post-compilation members must preserve the exact requested "
                "member order"
            )
        return self


def reconciled_residual_preference_output_type(
    request: HierarchicalResidualProposalRequest,
) -> type[BaseModel]:
    """Build a structural DTO; trusted code owns cross-field feasibility."""

    request.__post_init__()
    parent_literal = Literal.__getitem__(
        tuple(
            value.parent_candidate_id.value
            for value in request.action_schema.bindings
        )
    )
    option_literal = Literal.__getitem__(request.action_schema.option_ids)
    role_literal = Literal.__getitem__(
        tuple(value.value for value in request.allowed_roles)
    )
    metric_literal = Literal.__getitem__(request.required_metric_ids)
    prediction_type = create_model(
        "ResidualMetricPreference",
        __base__=_PreferenceMetricPredictionBase,
        __module__=__name__,
        metric_id=(metric_literal, ...),
        p10_delta=(float, ...),
        p50_delta=(float, ...),
        p90_delta=(float, ...),
        confidence=(float, Field(ge=0.0, le=1.0, strict=True)),
    )
    member_type = create_model(
        "HierarchicalResidualPreferenceMember",
        __base__=_PreferenceMemberBase,
        __module__=__name__,
        parent_candidate_id=(parent_literal, ...),
        component_option_ids=(
            list[option_literal],
            Field(
                min_length=min(request.allowed_radii),
                max_length=max(request.allowed_radii),
            ),
        ),
        role=(role_literal, ...),
        probability_valid=(float, Field(ge=0.0, le=1.0, strict=True)),
        effect_predictions=(
            list[prediction_type],
            Field(
                min_length=len(request.required_metric_ids),
                max_length=len(request.required_metric_ids),
            ),
        ),
        interaction_rationale=(_Rationale, ...),
    )
    member_type.allowed_radii = frozenset(request.allowed_radii)
    member_type.required_metric_ids = frozenset(
        request.required_metric_ids
    )
    output_type = create_model(
        "HierarchicalResidualPreferenceSlate",
        __base__=_PreferenceOutputBase,
        __module__=__name__,
        members=(
            list[member_type],
            Field(
                min_length=request.proposal_count,
                max_length=request.proposal_count,
            ),
        ),
        slate_rationale=(_Rationale, ...),
    )
    output_type.proposal_count = request.proposal_count
    output_type.minimum_distinct_parents = (
        request.minimum_distinct_parents
    )
    return output_type


def postcompile_semantic_regrounding_output_type(
    *,
    projected_member_indices: tuple[int, ...],
    required_metric_ids: tuple[str, ...],
) -> type[BaseModel]:
    """Build a DTO that can annotate, but never alter, compiled identities."""

    if (
        type(projected_member_indices) is not tuple
        or not projected_member_indices
        or any(
            type(value) is not int or value <= 0
            for value in projected_member_indices
        )
        or projected_member_indices
        != tuple(sorted(set(projected_member_indices)))
    ):
        raise ValueError(
            "projected_member_indices must be a non-empty canonical tuple"
        )
    if (
        type(required_metric_ids) is not tuple
        or not required_metric_ids
        or any(type(value) is not str or not value for value in required_metric_ids)
        or required_metric_ids != tuple(sorted(set(required_metric_ids)))
    ):
        raise ValueError(
            "required_metric_ids must be a non-empty canonical tuple"
        )
    member_literal = Literal.__getitem__(projected_member_indices)
    metric_literal = Literal.__getitem__(required_metric_ids)
    prediction_type = create_model(
        "PostCompilationResidualMetricForecast",
        __base__=_PreferenceMetricPredictionBase,
        __module__=__name__,
        metric_id=(metric_literal, ...),
        p10_delta=(float, ...),
        p50_delta=(float, ...),
        p90_delta=(float, ...),
        confidence=(float, Field(ge=0.0, le=1.0, strict=True)),
    )
    member_type = create_model(
        "PostCompilationResidualSemanticClaims",
        __base__=_PostCompilationMemberBase,
        __module__=__name__,
        member_index=(member_literal, ...),
        semantic_fidelity_acknowledgement=(
            Literal["exact_compiled_action"],
            ...,
        ),
        probability_valid=(float, Field(ge=0.0, le=1.0, strict=True)),
        effect_predictions=(
            list[prediction_type],
            Field(
                min_length=len(required_metric_ids),
                max_length=len(required_metric_ids),
            ),
        ),
        interaction_rationale=(_Rationale, ...),
    )
    member_type.required_metric_ids = frozenset(required_metric_ids)
    output_type = create_model(
        "PostCompilationResidualSemanticSlate",
        __base__=_PostCompilationOutputBase,
        __module__=__name__,
        members=(
            list[member_type],
            Field(
                min_length=len(projected_member_indices),
                max_length=len(projected_member_indices),
            ),
        ),
        regrounding_rationale=(_Rationale, ...),
    )
    output_type.required_member_indices = projected_member_indices
    return output_type


def _semantic_record(option: FiniteVariationOption) -> dict[str, object]:
    prompt = option.prompt_record()
    return {
        "family": prompt["family"],
        "description": prompt["description"],
        "metadata": {
            key: value
            for key, value in cast(
                dict[str, object],
                prompt["metadata"],
            ).items()
            if not key.endswith("_sha256")
        },
    }


def _description_tokens(value: object) -> frozenset[str]:
    if type(value) is not str:
        return frozenset()
    return frozenset(token.lower() for token in _TOKEN.findall(value))


def _semantic_distance(
    source: dict[str, object],
    target: dict[str, object],
    target_option_id: str,
) -> tuple[int, int, int, str]:
    source_metadata = cast(dict[str, object], source["metadata"])
    target_metadata = cast(dict[str, object], target["metadata"])
    keys = set(source_metadata) | set(target_metadata)
    metadata_disagreement = sum(
        source_metadata.get(key) != target_metadata.get(key)
        for key in keys
    )
    token_disagreement = len(
        _description_tokens(source["description"])
        ^ _description_tokens(target["description"])
    )
    return (
        int(source["family"] != target["family"]),
        metadata_disagreement,
        token_disagreement,
        target_option_id,
    )


@dataclass(frozen=True, slots=True)
class ResidualMemberReconciliation:
    parent_candidate_id: str
    requested_component_option_ids: tuple[str, ...]
    reconciled_component_option_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    projection_score: tuple[int, ...] | None

    def to_record(self) -> dict[str, object]:
        unsigned: dict[str, object] = {
            "schema_version": 1,
            "parent_candidate_id": self.parent_candidate_id,
            "requested_component_option_ids": list(
                self.requested_component_option_ids
            ),
            "reconciled_component_option_ids": list(
                self.reconciled_component_option_ids
            ),
            "reason_codes": list(self.reason_codes),
            "projection_score": (
                None
                if self.projection_score is None
                else list(self.projection_score)
            ),
        }
        return {
            **unsigned,
            "reconciliation_sha256": _hash(
                _RECONCILIATION_DOMAIN,
                unsigned,
            ),
        }


def _nearest_available_option(
    *,
    requested_id: str,
    available_by_id: dict[str, FiniteVariationOption],
    union_semantics_by_id: dict[str, dict[str, object]],
) -> tuple[str, tuple[int, int, int, str]]:
    source = union_semantics_by_id[requested_id]
    ranked = tuple(
        (
            _semantic_distance(
                source,
                _semantic_record(option),
                option_id,
            ),
            option_id,
        )
        for option_id, option in available_by_id.items()
    )
    score, option_id = min(ranked)
    return option_id, score


def reconcile_component_preferences(
    *,
    schema: CrossParentFiniteActionSchema,
    parent_candidate_id: str,
    requested_option_ids: tuple[str, ...],
) -> ResidualMemberReconciliation:
    """Project one member onto its exact parent-bound feasible action set."""

    binding = next(
        value
        for value in schema.bindings
        if value.parent_candidate_id.value == parent_candidate_id
    )
    available_by_id = {
        value.option_id: value for value in binding.contract.options
    }
    union_semantics_by_id = {
        cast(str, value["option_id"]): {
            "family": value["family"],
            "description": value["description"],
            "metadata": {
                key: item
                for key, item in cast(
                    dict[str, object],
                    value["metadata"],
                ).items()
                if not key.endswith("_sha256")
            },
        }
        for value in schema.action_prompt_records
    }
    current: list[str] = []
    reason_codes: list[str] = []
    component_scores: list[tuple[int, int, int, str]] = []
    for requested_id in requested_option_ids:
        if requested_id in available_by_id:
            current.append(requested_id)
            continue
        replacement, score = _nearest_available_option(
            requested_id=requested_id,
            available_by_id=available_by_id,
            union_semantics_by_id=union_semantics_by_id,
        )
        current.append(replacement)
        component_scores.append(score)
        reason_codes.append("parent_unavailable_option_projected")

    if len(current) == 1:
        final = tuple(current)
    else:
        compatible = pairwise_disjoint_parent_patch_pairs(
            binding.contract,
            tuple(available_by_id),
        )
        compatible_set = frozenset(compatible)
        pair = tuple(sorted(current))
        if len(set(pair)) == 2 and pair in compatible_set:
            final = pair
        else:
            candidates: list[
                tuple[
                    tuple[int, int, int, str, int],
                    tuple[str, str],
                ]
            ] = []
            for keep_index, keep in enumerate(current):
                rejected = current[1 - keep_index]
                rejected_semantics = union_semantics_by_id[rejected]
                for compatible_pair in compatible:
                    if keep not in compatible_pair:
                        continue
                    replacement = (
                        compatible_pair[0]
                        if compatible_pair[1] == keep
                        else compatible_pair[1]
                    )
                    distance = _semantic_distance(
                        rejected_semantics,
                        _semantic_record(available_by_id[replacement]),
                        replacement,
                    )
                    candidates.append(
                        ((*distance, keep_index), compatible_pair)
                    )
            if not candidates:
                raise ValueError(
                    "finite contract exposes no compatible reconciliation"
                )
            projection_score, final = min(candidates)
            component_scores.append(projection_score[:4])
            reason_codes.append("overlapping_component_projected")

    integer_score = None
    if component_scores:
        integer_score = tuple(
            sum(value[index] for value in component_scores)
            for index in range(3)
        )
    return ResidualMemberReconciliation(
        parent_candidate_id=parent_candidate_id,
        requested_component_option_ids=requested_option_ids,
        reconciled_component_option_ids=tuple(sorted(final)),
        reason_codes=tuple(sorted(set(reason_codes))),
        projection_score=integer_score,
    )


def _telemetry(
    response: StructuredGenerationResponse[Any],
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


def _canonical_forecasts(
    predictions: object,
) -> tuple[HierarchicalResidualMetricForecast, ...]:
    return tuple(
        sorted(
            (
                HierarchicalResidualMetricForecast(
                    metric_id=prediction.metric_id,
                    p10_delta=sorted(
                        (
                            float(prediction.p10_delta),
                            float(prediction.p50_delta),
                            float(prediction.p90_delta),
                        )
                    )[0],
                    p50_delta=sorted(
                        (
                            float(prediction.p10_delta),
                            float(prediction.p50_delta),
                            float(prediction.p90_delta),
                        )
                    )[1],
                    p90_delta=sorted(
                        (
                            float(prediction.p10_delta),
                            float(prediction.p50_delta),
                            float(prediction.p90_delta),
                        )
                    )[2],
                    confidence=float(prediction.confidence),
                )
                for prediction in cast(Any, predictions)
            ),
            key=lambda item: item.metric_id,
        )
    )


def _neutral_forecasts(
    metric_ids: tuple[str, ...],
) -> tuple[HierarchicalResidualMetricForecast, ...]:
    return tuple(
        HierarchicalResidualMetricForecast(
            metric_id=metric_id,
            p10_delta=0.0,
            p50_delta=0.0,
            p90_delta=0.0,
            confidence=0.0,
        )
        for metric_id in metric_ids
    )


def _claims_record(
    *,
    probability_valid: float,
    forecasts: tuple[HierarchicalResidualMetricForecast, ...],
    rationale: str,
) -> dict[str, object]:
    return {
        "probability_valid_hex": probability_valid.hex(),
        "effect_predictions": [
            {
                "metric_id": prediction.metric_id,
                "p10_delta_hex": prediction.p10_delta.hex(),
                "p50_delta_hex": prediction.p50_delta.hex(),
                "p90_delta_hex": prediction.p90_delta.hex(),
                "confidence_hex": prediction.confidence.hex(),
            }
            for prediction in forecasts
        ],
        "interaction_rationale": rationale,
    }


def _response_telemetry_record(
    response: StructuredGenerationResponse[Any],
    attempt_count: int,
) -> dict[str, object]:
    return {
        "requested_model": response.requested_model,
        "resolved_model": response.resolved_model,
        "resolved_provider": response.resolved_provider,
        "provider_response_id": response.provider_response_id,
        "finish_reason": response.finish_reason,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "reasoning_tokens": response.reasoning_tokens,
        "cache_read_tokens": response.cache_read_tokens,
        "cache_write_tokens": response.cache_write_tokens,
        "cost_usd": (
            None if response.cost_usd is None else str(response.cost_usd)
        ),
        "latency_ns": response.latency_ns,
        "attempt_count": attempt_count,
    }


def _derived_regrounding_call_id(call_id: LLMCallId) -> LLMCallId:
    candidate = f"{call_id.value}_reground"
    if len(candidate) <= 128:
        return LLMCallId(candidate)
    digest = hashlib.sha256(
        call_id.value.encode("ascii", errors="strict")
    ).hexdigest()[:24]
    return LLMCallId(f"call_reground_{digest}")


def _option_record_for_parent(
    request: HierarchicalResidualProposalRequest,
    *,
    parent_candidate_id: str,
    option_id: str,
) -> dict[str, object]:
    binding = next(
        value
        for value in request.action_schema.bindings
        if value.parent_candidate_id.value == parent_candidate_id
    )
    option = next(
        value
        for value in binding.contract.options
        if value.option_id == option_id
    )
    return option.prompt_record()


def _postcompile_regrounding_prompt(
    *,
    request: HierarchicalResidualProposalRequest,
    members: object,
    reconciliations: tuple[ResidualMemberReconciliation, ...],
    canonical_predictions: tuple[
        tuple[HierarchicalResidualMetricForecast, ...],
        ...,
    ],
) -> tuple[str, tuple[int, ...]]:
    projected_indices = tuple(
        index
        for index, reconciliation in enumerate(reconciliations, start=1)
        if reconciliation.reason_codes
    )
    if not projected_indices:
        raise ValueError("post-compilation prompt requires projected members")
    source_members = tuple(cast(Any, members))
    payload_members: list[dict[str, object]] = []
    for member_index in projected_indices:
        member = source_members[member_index - 1]
        reconciliation = reconciliations[member_index - 1]
        original_claims = _claims_record(
            probability_valid=float(member.probability_valid),
            forecasts=canonical_predictions[member_index - 1],
            rationale=str(member.interaction_rationale),
        )
        payload_members.append(
            {
                "member_index": member_index,
                "parent_candidate_id": member.parent_candidate_id,
                "requested_component_option_ids": list(
                    reconciliation.requested_component_option_ids
                ),
                "exact_compiled_component_option_ids": list(
                    reconciliation.reconciled_component_option_ids
                ),
                "projection_reason_codes": list(
                    reconciliation.reason_codes
                ),
                "exact_compiled_option_records": [
                    _option_record_for_parent(
                        request,
                        parent_candidate_id=member.parent_candidate_id,
                        option_id=option_id,
                    )
                    for option_id in (
                        reconciliation.reconciled_component_option_ids
                    )
                ],
                "quarantined_original_claims": original_claims,
            }
        )
    contract = {
        "schema_version": 1,
        "source_request_sha256": request.request_sha256,
        "source_instruction": request.instruction,
        "strictly_prior_context": thaw_json(request.context),
        "required_metric_ids": list(request.required_metric_ids),
        "projected_members": payload_members,
        "authority": {
            "exact_compiled_parent_and_actions_are_immutable": True,
            "may_change_configuration_or_action_identity": False,
            "may_restore_quarantined_claims_without_new_justification": False,
            "current_candidate_outcomes_observed": False,
            "task": (
                "author fresh probability, raw-metric effect forecasts, and "
                "mechanistic rationale for each exact compiled action"
            ),
        },
        "forecast_contract": {
            "deltas_are_parent_relative_in_raw_metric_units": True,
            "ordered_quantiles_required": True,
            "confidence_and_probability_valid_range": [0.0, 1.0],
        },
    }
    prompt = "\n".join(
        (
            "You are a post-compilation semantic verifier inside a generic "
            "evolutionary optimizer. Trusted code changed one or more proposed "
            "actions to satisfy an exact finite contract. Claims written for "
            "the rejected actions have no authority. Re-ground every listed "
            "member against only its exact compiled parent/action records and "
            "the strictly-prior evidence. You cannot select, replace, reorder, "
            "or edit an executable action.",
            "",
            "POST-COMPILATION SEMANTIC REGROUNDING CONTRACT",
            json.dumps(
                contract,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    )
    return prompt, projected_indices


@dataclass(frozen=True, slots=True)
class PostCompilationSemanticRegrounding:
    status: str
    call_id: LLMCallId | None
    decision_sha256: str
    claims_by_member_index: tuple[
        tuple[
            int,
            float,
            tuple[HierarchicalResidualMetricForecast, ...],
            str,
        ],
        ...,
    ]
    telemetry: dict[str, object] | None
    regrounding_rationale: str | None
    failure_kind: str | None
    unordered_forecast_count: int

    def __post_init__(self) -> None:
        if (
            type(self.unordered_forecast_count) is not int
            or self.unordered_forecast_count < 0
        ):
            raise ValueError(
                "unordered_forecast_count must be a non-negative integer"
            )
        if self.status not in {
            "not_required",
            "regrounded",
            "quarantined_fallback",
        }:
            raise ValueError("unknown post-compilation regrounding status")
        if self.status == "not_required":
            if (
                self.call_id is not None
                or self.claims_by_member_index
                or self.telemetry is not None
                or self.regrounding_rationale is not None
                or self.failure_kind is not None
            ):
                raise ValueError("not-required regrounding cannot carry a call")
        else:
            if type(self.call_id) is not LLMCallId:
                raise TypeError("attempted regrounding requires a call ID")
            LLMCallId.__post_init__(cast(LLMCallId, self.call_id))
            if not self.claims_by_member_index:
                raise ValueError("attempted regrounding requires final claims")
        if self.status == "regrounded":
            if self.telemetry is None or self.regrounding_rationale is None:
                raise ValueError("successful regrounding requires evidence")
            if self.failure_kind is not None:
                raise ValueError("successful regrounding cannot have a failure")
        if self.status == "quarantined_fallback":
            if (
                self.telemetry is not None
                or self.regrounding_rationale is not None
                or self.failure_kind is None
            ):
                raise ValueError("fallback regrounding evidence is inconsistent")


@dataclass(slots=True)
class PydanticAIReconciledHierarchicalResidualProposalPolicy:
    generate_once: LowLevelRunner
    reconciliation_sink: Callable[[dict[str, object]], None] | None = None

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if self.reconciliation_sink is not None and not callable(
            self.reconciliation_sink
        ):
            raise TypeError("reconciliation_sink must be callable or None")

    async def _reground_projected_members(
        self,
        *,
        request: HierarchicalResidualProposalRequest,
        members: object,
        reconciliations: tuple[ResidualMemberReconciliation, ...],
        canonical_predictions: tuple[
            tuple[HierarchicalResidualMetricForecast, ...],
            ...,
        ],
    ) -> PostCompilationSemanticRegrounding:
        projected_indices = tuple(
            index
            for index, reconciliation in enumerate(
                reconciliations,
                start=1,
            )
            if reconciliation.reason_codes
        )
        if not projected_indices:
            result = PostCompilationSemanticRegrounding(
                status="not_required",
                call_id=None,
                decision_sha256=_hash(
                    _POSTCOMPILE_DECISION_DOMAIN,
                    {
                        "schema_version": 1,
                        "source_request_sha256": request.request_sha256,
                        "status": "not_required",
                    },
                ),
                claims_by_member_index=(),
                telemetry=None,
                regrounding_rationale=None,
                failure_kind=None,
                unordered_forecast_count=0,
            )
            result.__post_init__()
            return result

        prompt, rendered_indices = _postcompile_regrounding_prompt(
            request=request,
            members=members,
            reconciliations=reconciliations,
            canonical_predictions=canonical_predictions,
        )
        if rendered_indices != projected_indices:
            raise RuntimeError("post-compilation prompt lost projected members")
        output_type = postcompile_semantic_regrounding_output_type(
            projected_member_indices=projected_indices,
            required_metric_ids=request.required_metric_ids,
        )
        call_id = _derived_regrounding_call_id(request.call_id)
        try:
            raw = await self.generate_once(
                StructuredGenerationRequest(
                    call_id=call_id,
                    operation="postcompile_semantic_regrounding",
                    prompt=prompt,
                    output_type=output_type,
                    output_tool_name=_POSTCOMPILE_TOOL_NAME,
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
                raise TypeError(
                    "post-compilation response differs from requested type"
                )
            value = cast(Any, response.value)
            claims = tuple(
                (
                    int(member.member_index),
                    float(member.probability_valid),
                    _canonical_forecasts(member.effect_predictions),
                    (
                        "Post-compilation semantic re-grounding for the exact "
                        f"compiled action: {member.interaction_rationale}"
                    ),
                )
                for member in value.members
            )
            unordered_forecast_count = sum(
                (
                    float(prediction.p10_delta),
                    float(prediction.p50_delta),
                    float(prediction.p90_delta),
                )
                != tuple(
                    sorted(
                        (
                            float(prediction.p10_delta),
                            float(prediction.p50_delta),
                            float(prediction.p90_delta),
                        )
                    )
                )
                for member in value.members
                for prediction in member.effect_predictions
            )
            regrounding_rationale = (
                "Post-compilation semantic re-grounding for exact compiled "
                f"actions: {value.regrounding_rationale}"
            )
            decision_record = {
                "schema_version": 1,
                "source_request_sha256": request.request_sha256,
                "call_id": call_id.value,
                "status": "regrounded",
                "claims": [
                    {
                        "member_index": index,
                        **_claims_record(
                            probability_valid=probability_valid,
                            forecasts=forecasts,
                            rationale=rationale,
                        ),
                    }
                    for index, probability_valid, forecasts, rationale in claims
                ],
                "regrounding_rationale": regrounding_rationale,
            }
            result = PostCompilationSemanticRegrounding(
                status="regrounded",
                call_id=call_id,
                decision_sha256=_hash(
                    _POSTCOMPILE_DECISION_DOMAIN,
                    decision_record,
                ),
                claims_by_member_index=claims,
                telemetry=_response_telemetry_record(
                    response,
                    attempt_count,
                ),
                regrounding_rationale=regrounding_rationale,
                failure_kind=None,
                unordered_forecast_count=unordered_forecast_count,
            )
            result.__post_init__()
            return result
        except Exception as error:
            fallback_rationale = (
                "The trusted compiler changed this executable action. The "
                "original semantic claims are quarantined and post-compilation "
                "re-grounding did not complete; reason only from the exact "
                "compiled action and strictly-prior evidence."
            )
            neutral = _neutral_forecasts(request.required_metric_ids)
            claims = tuple(
                (index, 0.5, neutral, fallback_rationale)
                for index in projected_indices
            )
            failure_kind = type(error).__name__
            decision_record = {
                "schema_version": 1,
                "source_request_sha256": request.request_sha256,
                "call_id": call_id.value,
                "status": "quarantined_fallback",
                "failure_kind": failure_kind,
                "claims": [
                    {
                        "member_index": index,
                        **_claims_record(
                            probability_valid=probability_valid,
                            forecasts=forecasts,
                            rationale=rationale,
                        ),
                    }
                    for index, probability_valid, forecasts, rationale in claims
                ],
            }
            result = PostCompilationSemanticRegrounding(
                status="quarantined_fallback",
                call_id=call_id,
                decision_sha256=_hash(
                    _POSTCOMPILE_DECISION_DOMAIN,
                    decision_record,
                ),
                claims_by_member_index=claims,
                telemetry=None,
                regrounding_rationale=None,
                failure_kind=failure_kind,
                unordered_forecast_count=0,
            )
            result.__post_init__()
            return result

    async def select(
        self,
        request: HierarchicalResidualProposalRequest,
    ) -> HierarchicalResidualProposalSelection:
        request.__post_init__()
        output_type = reconciled_residual_preference_output_type(request)
        prompt = "\n".join(
            (
                render_hierarchical_residual_prompt(request),
                "",
                "VALIDITY OWNERSHIP",
                "Express the best semantic parent/action preferences. Select "
                "distinct parent-relative patch loci whenever the radius is "
                "two. Trusted engine code independently projects any locally "
                "incompatible member onto the nearest exact finite-contract "
                "action; one local mistake will not discard the full slate.",
            )
        )
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
            raise TypeError("response value differs from requested output type")
        value = cast(Any, response.value)
        reconciliations = tuple(
            reconcile_component_preferences(
                schema=request.action_schema,
                parent_candidate_id=member.parent_candidate_id,
                requested_option_ids=tuple(member.component_option_ids),
            )
            for member in value.members
        )
        canonical_predictions = tuple(
            _canonical_forecasts(member.effect_predictions)
            for member in value.members
        )
        regrounding = await self._reground_projected_members(
            request=request,
            members=value.members,
            reconciliations=reconciliations,
            canonical_predictions=canonical_predictions,
        )
        final_probabilities = [
            float(member.probability_valid) for member in value.members
        ]
        final_predictions = list(canonical_predictions)
        final_rationales = [
            str(member.interaction_rationale) for member in value.members
        ]
        for (
            member_index,
            probability_valid,
            predictions,
            rationale,
        ) in regrounding.claims_by_member_index:
            final_probabilities[member_index - 1] = probability_valid
            final_predictions[member_index - 1] = predictions
            final_rationales[member_index - 1] = rationale
        if regrounding.status == "regrounded":
            final_slate_rationale = cast(
                str,
                regrounding.regrounding_rationale,
            )
        elif regrounding.status == "quarantined_fallback":
            final_slate_rationale = (
                "One or more actions were changed by trusted compilation. "
                "Their original slate-level semantic claims are quarantined; "
                "reason only from exact compiled actions and strictly-prior "
                "evidence."
            )
        else:
            final_slate_rationale = str(value.slate_rationale)
        final_members = [
            {
                "parent_candidate_id": member.parent_candidate_id,
                "component_option_ids": list(
                    reconciliation.reconciled_component_option_ids
                ),
                "role": member.role,
                "probability_valid": probability_valid,
                "effect_predictions": [
                    {
                        "metric_id": prediction.metric_id,
                        "p10_delta": prediction.p10_delta,
                        "p50_delta": prediction.p50_delta,
                        "p90_delta": prediction.p90_delta,
                        "confidence": prediction.confidence,
                    }
                    for prediction in predictions
                ],
                "interaction_rationale": rationale,
            }
            for (
                member,
                reconciliation,
                probability_valid,
                predictions,
                rationale,
            ) in zip(
                value.members,
                reconciliations,
                final_probabilities,
                final_predictions,
                final_rationales,
                strict=True,
            )
        ]
        decision_sha256 = _hash(
            _DECISION_DOMAIN,
            {
                "schema_version": 2,
                "request_sha256": request.request_sha256,
                "postcompile_decision_sha256": (
                    regrounding.decision_sha256
                ),
                "members": final_members,
                "slate_rationale": final_slate_rationale,
            },
        )
        plans = tuple(
            HierarchicalResidualPlan(
                parent_candidate_id=next(
                    binding.parent_candidate_id
                    for binding in request.action_schema.bindings
                    if binding.parent_candidate_id.value
                    == member.parent_candidate_id
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
                component_option_ids=(
                    reconciliation.reconciled_component_option_ids
                ),
                role=ResidualProposalRole(member.role),
                expert_id=request.expert_id,
                expert_definition_sha256=request.expert_definition_sha256,
                native_rank=rank,
                decision_receipt_sha256=decision_sha256,
            )
            for rank, (member, reconciliation) in enumerate(
                zip(value.members, reconciliations, strict=True),
                start=1,
            )
        )
        result = HierarchicalResidualProposalSelection(
            request_sha256=request.request_sha256,
            decision_sha256=decision_sha256,
            plans=plans,
            rationales=tuple(final_rationales),
            probability_valid=tuple(final_probabilities),
            effect_predictions=tuple(final_predictions),
            slate_rationale=final_slate_rationale,
            telemetry=_telemetry(response, attempt_count),
        )
        result.__post_init__()
        if self.reconciliation_sink is not None:
            unordered_forecast_count = sum(
                (
                    float(prediction.p10_delta),
                    float(prediction.p50_delta),
                    float(prediction.p90_delta),
                )
                != tuple(
                    sorted(
                        (
                            float(prediction.p10_delta),
                            float(prediction.p50_delta),
                            float(prediction.p90_delta),
                        )
                    )
                )
                for member in value.members
                for prediction in member.effect_predictions
            )
            member_records: list[dict[str, object]] = []
            for member_index, (
                member,
                reconciliation,
                original_predictions,
                final_probability,
                allocation_predictions,
                allocation_rationale,
            ) in enumerate(
                zip(
                    value.members,
                    reconciliations,
                    canonical_predictions,
                    final_probabilities,
                    final_predictions,
                    final_rationales,
                    strict=True,
                ),
                start=1,
            ):
                original_claims = _claims_record(
                    probability_valid=float(member.probability_valid),
                    forecasts=original_predictions,
                    rationale=str(member.interaction_rationale),
                )
                allocation_claims = _claims_record(
                    probability_valid=final_probability,
                    forecasts=allocation_predictions,
                    rationale=allocation_rationale,
                )
                member_records.append(
                    {
                        **reconciliation.to_record(),
                        "member_index": member_index,
                        "semantic_fidelity": (
                            "exact_claims_preserved"
                            if not reconciliation.reason_codes
                            else (
                                "projected_claims_regrounded"
                                if regrounding.status == "regrounded"
                                else "projected_claims_quarantined_fallback"
                            )
                        ),
                        "original_claims": original_claims,
                        "original_claims_sha256": _hash(
                            _POSTCOMPILE_DECISION_DOMAIN,
                            original_claims,
                        ),
                        "allocation_claims": allocation_claims,
                        "allocation_claims_sha256": _hash(
                            _POSTCOMPILE_DECISION_DOMAIN,
                            allocation_claims,
                        ),
                    }
                )
            record: dict[str, object] = {
                "schema_version": 2,
                "policy_id": RECONCILED_RESIDUAL_PROPOSAL_POLICY_ID,
                "policy_version": (
                    RECONCILED_RESIDUAL_PROPOSAL_POLICY_VERSION
                ),
                "policy_definition_sha256": (
                    RECONCILED_RESIDUAL_PROPOSAL_POLICY_DEFINITION_SHA256
                ),
                "call_id": request.call_id.value,
                "request_sha256": request.request_sha256,
                "decision_sha256": decision_sha256,
                "members": member_records,
                "projected_member_count": sum(
                    bool(value.reason_codes)
                    for value in reconciliations
                ),
                "unordered_forecast_count": unordered_forecast_count,
                "postcompile_regrounding": {
                    "status": regrounding.status,
                    "call_id": (
                        None
                        if regrounding.call_id is None
                        else regrounding.call_id.value
                    ),
                    "decision_sha256": regrounding.decision_sha256,
                    "telemetry": regrounding.telemetry,
                    "failure_kind": regrounding.failure_kind,
                    "unordered_forecast_count": (
                        regrounding.unordered_forecast_count
                    ),
                },
                "original_slate_rationale": str(value.slate_rationale),
                "allocation_slate_rationale": final_slate_rationale,
                "whole_slate_retry_avoided": any(
                    value.reason_codes for value in reconciliations
                )
                or unordered_forecast_count > 0
                or regrounding.unordered_forecast_count > 0,
            }
            record["receipt_sha256"] = _hash(
                _RECONCILIATION_DOMAIN,
                record,
            )
            self.reconciliation_sink(record)
        return result


__all__ = [
    "PydanticAIReconciledHierarchicalResidualProposalPolicy",
    "PostCompilationSemanticRegrounding",
    "RECONCILED_RESIDUAL_PROPOSAL_POLICY_DEFINITION_SHA256",
    "RECONCILED_RESIDUAL_PROPOSAL_POLICY_ID",
    "RECONCILED_RESIDUAL_PROPOSAL_POLICY_VERSION",
    "ResidualMemberReconciliation",
    "reconcile_component_preferences",
    "postcompile_semantic_regrounding_output_type",
    "reconciled_residual_preference_output_type",
]
