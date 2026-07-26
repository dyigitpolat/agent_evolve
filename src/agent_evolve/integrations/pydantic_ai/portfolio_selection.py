"""Pydantic-AI adapter for one-call ranked portfolio selection.

This adapter owns only prompt/schema construction and translation.  The
injected low-level runner may be the existing queued runner, so concurrency,
retry, exponential backoff, and provider policy remain outside this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    create_model,
    model_validator,
)

from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
    LowLevelRunner,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioMemberDraft,
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
    resolve_ranked_portfolio_decision,
    validate_pairwise_disjoint_parent_patch_selection,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)


PORTFOLIO_SELECTION_TOOL_NAME = "select_ranked_portfolio"
PORTFOLIO_SELECTION_POLICY_ID = "pydantic_ai_ranked_portfolio"
PORTFOLIO_SELECTION_POLICY_VERSION = 1
PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-ranked-portfolio:v1:"
    b"one-call-exact-k-sealed-options-card-attribution-metric-predictions"
).hexdigest()
PORTFOLIO_SELECTION_DISJOINT_POLICY_VERSION = 2
PORTFOLIO_SELECTION_DISJOINT_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-ranked-portfolio:v2:"
    b"one-call-exact-k-sealed-options-card-attribution-metric-predictions;"
    b"optional-trusted-prefix-aware-pairwise-parent-patch-disjointness"
).hexdigest()

_STRICT_CONFIG = ConfigDict(
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
_Direction = Literal["decrease", "increase", "unchanged", "unknown"]


class _PortfolioMemberOutputBase(BaseModel):
    model_config = _STRICT_CONFIG

    allowed_card_keys: ClassVar[frozenset[str]] = frozenset()
    required_metric_ids: ClassVar[tuple[str, ...]] = ()
    require_supporting_cards: ClassVar[bool] = True

    @model_validator(mode="after")
    def _validate_member(self) -> "_PortfolioMemberOutputBase":
        card_keys = tuple(cast(Any, self).supporting_card_keys)
        if len(set(card_keys)) != len(card_keys):
            raise ValueError("supporting_card_keys cannot contain duplicates")
        if not set(card_keys).issubset(type(self).allowed_card_keys):
            raise ValueError("supporting_card_keys escape the request snapshot")
        if type(self).require_supporting_cards and not card_keys:
            raise ValueError("this request requires supporting-card attribution")
        predictions = tuple(cast(Any, self).effect_predictions)
        metric_ids = tuple(item.metric_id for item in predictions)
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("effect_predictions cannot repeat a metric")
        if set(metric_ids) != set(type(self).required_metric_ids):
            raise ValueError(
                "effect_predictions must cover the exact requested metrics"
            )
        return self


class _PortfolioOutputBase(BaseModel):
    model_config = _STRICT_CONFIG

    portfolio_size: ClassVar[int] = 1
    option_family_by_id: ClassVar[dict[str, str]] = {}
    min_distinct_families: ClassVar[int | None] = None
    require_pairwise_disjoint_parent_patches: ClassVar[bool] = False
    finite_variation_contract: ClassVar[Any] = None

    @model_validator(mode="after")
    def _validate_portfolio(self) -> "_PortfolioOutputBase":
        members = tuple(cast(Any, self).members)
        if len(members) != type(self).portfolio_size:
            raise ValueError("members must contain exactly portfolio_size entries")
        option_ids = tuple(member.option_id for member in members)
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("members cannot repeat a finite option")
        minimum = type(self).min_distinct_families
        if minimum is not None and len(
            {type(self).option_family_by_id[option_id] for option_id in option_ids}
        ) < minimum:
            raise ValueError("members violate min_distinct_families")
        if type(self).require_pairwise_disjoint_parent_patches:
            validate_pairwise_disjoint_parent_patch_selection(
                type(self).finite_variation_contract,
                option_ids,
            )
        return self


def _portfolio_output_type(request: PortfolioSelectionRequest) -> type[BaseModel]:
    request.__post_init__()
    option_ids = tuple(
        option.option_id for option in request.finite_variation_contract.options
    )
    card_keys = tuple(card.card_key for card in request.cards)
    metric_ids = request.required_metric_ids
    option_literal = Literal.__getitem__(option_ids)
    card_literal = Literal.__getitem__(card_keys)
    metric_literal = Literal.__getitem__(metric_ids)

    prediction_type = create_model(
        "PortfolioMetricPrediction",
        __config__=_STRICT_CONFIG,
        __module__=__name__,
        metric_id=(metric_literal, ...),
        direction=(_Direction, ...),
    )
    member_type = create_model(
        "PortfolioMemberSelection",
        __base__=_PortfolioMemberOutputBase,
        __module__=__name__,
        option_id=(
            option_literal,
            Field(description="One selected immutable finite option ID."),
        ),
        supporting_card_keys=(
            list[card_literal],
            Field(
                description=(
                    "Opaque keys of cards that materially support this action."
                ),
                max_length=len(card_keys),
            ),
        ),
        effect_predictions=(
            list[prediction_type],
            Field(
                description=(
                    "One direction prediction for every required metric."
                ),
                min_length=len(metric_ids),
                max_length=len(metric_ids),
            ),
        ),
        design_rationale=(
            _Rationale,
            Field(description="Concise reason for this ranked action."),
        ),
    )
    member_type.allowed_card_keys = frozenset(card_keys)
    member_type.required_metric_ids = metric_ids
    member_type.require_supporting_cards = request.require_supporting_cards

    output_type = create_model(
        "RankedPortfolioSelection",
        __base__=_PortfolioOutputBase,
        __module__=__name__,
        members=(
            list[member_type],
            Field(
                description="Actions ordered best-first; list position is rank.",
                min_length=request.portfolio_size,
                max_length=request.portfolio_size,
            ),
        ),
    )
    output_type.portfolio_size = request.portfolio_size
    output_type.option_family_by_id = {
        option.option_id: option.family
        for option in request.finite_variation_contract.options
    }
    output_type.min_distinct_families = request.min_distinct_families
    output_type.require_pairwise_disjoint_parent_patches = (
        request.require_pairwise_disjoint_parent_patches
    )
    output_type.finite_variation_contract = request.finite_variation_contract
    return output_type


def render_portfolio_selection_prompt(request: PortfolioSelectionRequest) -> str:
    """Render only prompt-safe card views and sealed option descriptions."""

    if type(request) is not PortfolioSelectionRequest:
        raise TypeError("request must be an exact PortfolioSelectionRequest")
    request.__post_init__()
    contract = request.finite_variation_contract
    machine_contract = {
        "schema_version": 1,
        "request_sha256": request.request_sha256,
        "context_sha256": request.context_sha256,
        "context": cast(Any, request.context),
        "finite_variation_contract": {
            "catalog_id": contract.catalog_id,
            "catalog_version": contract.catalog_version,
            "catalog_definition_sha256": contract.catalog_definition_sha256,
            "parent_configuration_sha256": contract.parent_configuration_sha256,
            "contract_identity_sha256": contract.identity_sha256,
        },
        "ordered_options": list(contract.prompt_records()),
        "cards": [card.prompt_record() for card in request.cards],
        "portfolio_constraints": {
            "portfolio_size": request.portfolio_size,
            "distinct_option_ids": True,
            "min_distinct_families": request.min_distinct_families,
            "require_supporting_cards": request.require_supporting_cards,
            **(
                {
                    "require_pairwise_disjoint_parent_patches": True,
                    "selector_policy_version": (
                        PORTFOLIO_SELECTION_DISJOINT_POLICY_VERSION
                    ),
                    "selector_policy_definition_sha256": (
                        PORTFOLIO_SELECTION_DISJOINT_POLICY_DEFINITION_SHA256
                    ),
                }
                if request.require_pairwise_disjoint_parent_patches
                else {}
            ),
            "required_metric_ids": list(request.required_metric_ids),
        },
    }
    # Frozen typed-JSON containers are not directly JSON serializable.  The
    # context is intentionally projected through its public thaw operation;
    # card prompt records already return detached Python JSON values.
    from agent_evolve.domain.typed_json import thaw_json

    machine_contract["context"] = thaw_json(request.context)
    encoded = json.dumps(
        machine_contract,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "\n".join(
        (
            request.instruction,
            "",
            "RANKED PORTFOLIO SELECTION CONTRACT",
            encoded,
            "Return exactly the requested number of distinct option IDs in "
            "best-first order. For each member, cite only supplied card_key "
            "values and predict every required metric exactly once. Do not "
            "return candidate configurations.",
        )
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
            "low-level response value does not match the portfolio output type"
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


@dataclass(slots=True)
class PydanticAIPortfolioSelectionPolicy:
    """Translate one queued structured call into a fully resolved decision."""

    generate_once: LowLevelRunner

    policy_id: ClassVar[str] = PORTFOLIO_SELECTION_POLICY_ID
    policy_version: ClassVar[int] = PORTFOLIO_SELECTION_POLICY_VERSION
    policy_definition_sha256: ClassVar[str] = (
        PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult:
        if type(request) is not PortfolioSelectionRequest:
            raise TypeError("request must be an exact PortfolioSelectionRequest")
        request.__post_init__()
        output_type = _portfolio_output_type(request)
        low_level_request = StructuredGenerationRequest(
            call_id=request.call_id,
            operation=request.operation,
            prompt=render_portfolio_selection_prompt(request),
            output_type=output_type,
            output_tool_name=PORTFOLIO_SELECTION_TOOL_NAME,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
        )
        raw = await self.generate_once(low_level_request)
        response, attempt_count = _validated_response(
            raw,
            output_type=output_type,
        )
        value = cast(Any, response.value)
        drafts = tuple(
            PortfolioMemberDraft(
                option_id=member.option_id,
                supporting_card_keys=tuple(sorted(member.supporting_card_keys)),
                effect_predictions=tuple(
                    sorted(
                        (
                            MetricEffectPrediction(
                                metric_id=prediction.metric_id,
                                direction=MetricEffectDirection(
                                    prediction.direction
                                ),
                            )
                            for prediction in member.effect_predictions
                        ),
                        key=lambda prediction: prediction.metric_id,
                    )
                ),
                design_rationale=member.design_rationale,
            )
            for member in value.members
        )
        decision = resolve_ranked_portfolio_decision(
            request,
            drafts,
            policy_id=self.policy_id,
            policy_version=(
                PORTFOLIO_SELECTION_DISJOINT_POLICY_VERSION
                if request.require_pairwise_disjoint_parent_patches
                else self.policy_version
            ),
            policy_definition_sha256=(
                PORTFOLIO_SELECTION_DISJOINT_POLICY_DEFINITION_SHA256
                if request.require_pairwise_disjoint_parent_patches
                else self.policy_definition_sha256
            ),
        )
        return PortfolioSelectionResult(
            decision=decision,
            telemetry=_telemetry(response, attempt_count=attempt_count),
        )


__all__ = [
    "PORTFOLIO_SELECTION_POLICY_DEFINITION_SHA256",
    "PORTFOLIO_SELECTION_POLICY_ID",
    "PORTFOLIO_SELECTION_POLICY_VERSION",
    "PORTFOLIO_SELECTION_DISJOINT_POLICY_DEFINITION_SHA256",
    "PORTFOLIO_SELECTION_DISJOINT_POLICY_VERSION",
    "PORTFOLIO_SELECTION_TOOL_NAME",
    "PydanticAIPortfolioSelectionPolicy",
    "render_portfolio_selection_prompt",
]
