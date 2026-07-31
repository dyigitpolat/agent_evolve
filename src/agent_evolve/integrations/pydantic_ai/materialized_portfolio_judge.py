"""Generic Pydantic-AI allocation policy over a sealed proposal population.

The application boundary is intentionally split in two:

* a workload adapter projects prior evidence into a cognitive representation;
* this adapter compares the complete sealed action universe and returns an
  authenticated required set to the workload-blind broker.

The injected low-level runner owns provider setup, queuing, retry, exponential
backoff, and durable attempt evidence.  This module contains no workload,
model, or provider branch.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from agent_evolve.application.materialized_action_broker import (
    MaterializedActionAllocationRequirement,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionProposalBatch,
    ResidualPortfolioDecisionRequest,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    AttemptedStructuredGenerationResponse,
    LowLevelRunner,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationRequest,
    StructuredGenerationResponse,
)


MATERIALIZED_PORTFOLIO_JUDGE_TOOL_NAME = "select_materialized_portfolio"
MATERIALIZED_PORTFOLIO_JUDGE_POLICY_ID = (
    "pydantic_ai_outcome_blind_materialized_portfolio_nominator"
)
MATERIALIZED_PORTFOLIO_JUDGE_POLICY_VERSION = 2
MATERIALIZED_PORTFOLIO_JUDGE_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:pydantic-ai-outcome-blind-materialized-portfolio-judge:v2;"
    b"input=prior-cutoff-request+complete-sealed-proposal-union+injected-cognitive-projection;"
    b"output=bounded-nomination-action-identities+rationale+risk-pattern;"
    b"nomination-width=injected-positive-integer-at-most-evaluation-capacity;"
    b"remaining-capacity=owned-by-downstream-broker-or-other-allocators;"
    b"candidate-outcomes=false;async=true;"
    b"workload-model-provider-branches=false"
).hexdigest()

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_PROJECTION_DOMAIN = (
    b"agent-evolve:materialized-portfolio-judge-prompt-projection:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


@dataclass(frozen=True, slots=True)
class MaterializedPortfolioJudgePromptProjection:
    """Workload-owned cognitive view of one complete sealed action universe."""

    projection_id: str
    projection_version: int
    projection_definition_sha256: str
    action_sha256s: tuple[str, ...]
    instruction: str
    payload: FrozenJsonObject
    candidate_outcomes_observed: bool = False
    projection_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.projection_id, name="projection_id")
        if (
            type(self.projection_version) is not int
            or self.projection_version <= 0
        ):
            raise ValueError("projection_version must be positive")
        require_sha256(
            self.projection_definition_sha256,
            "projection_definition_sha256",
        )
        if type(self.action_sha256s) is not tuple or not self.action_sha256s:
            raise ValueError("action_sha256s must be a non-empty exact tuple")
        if self.action_sha256s != tuple(sorted(set(self.action_sha256s))):
            raise ValueError("action_sha256s must be unique and canonical")
        for value in self.action_sha256s:
            require_sha256(value, "action_sha256")
        if type(self.instruction) is not str or not self.instruction.strip():
            raise ValueError("instruction must be a non-empty exact string")
        if type(self.candidate_outcomes_observed) is not bool:
            raise TypeError("candidate_outcomes_observed must be an exact bool")
        if self.candidate_outcomes_observed:
            raise ValueError("a judge projection cannot contain candidate outcomes")
        if (
            type(self.payload) is not FrozenJsonObject
            or freeze_json(self.payload) is not self.payload
        ):
            raise TypeError("payload must be an exact frozen JSON object")
        object.__setattr__(
            self,
            "projection_sha256",
            hashlib.sha256(
                _PROJECTION_DOMAIN + _canonical_json(self._unsigned_record())
            ).hexdigest(),
        )

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "projection": {
                "projection_id": self.projection_id,
                "projection_version": self.projection_version,
                "definition_sha256": self.projection_definition_sha256,
            },
            "action_sha256s": list(self.action_sha256s),
            "instruction_sha256": hashlib.sha256(
                self.instruction.encode("utf-8", errors="strict")
            ).hexdigest(),
            "payload_sha256": typed_json_sha256(self.payload),
            "candidate_outcomes_observed": self.candidate_outcomes_observed,
        }

    def to_record(self, *, include_payload: bool = False) -> dict[str, object]:
        self.__post_init__()
        record = {
            **self._unsigned_record(),
            "projection_sha256": self.projection_sha256,
        }
        if include_payload:
            record["payload"] = thaw_json(self.payload)
        return record


@runtime_checkable
class MaterializedPortfolioJudgePromptProjectionPort(Protocol):
    """Project workload evidence without granting it selection authority."""

    projection_id: str
    projection_version: int
    definition_sha256: str

    def project(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedPortfolioJudgePromptProjection: ...


class _JudgeOutputBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    selection_count: ClassVar[int] = 1

    @model_validator(mode="after")
    def _validate_selection(self) -> "_JudgeOutputBase":
        selected = tuple(cast(Any, self).selected_action_sha256s)
        if len(selected) != type(self).selection_count:
            raise ValueError("selected_action_sha256s has the wrong exact length")
        if len(set(selected)) != len(selected):
            raise ValueError("selected_action_sha256s must be distinct")
        return self


def _judge_output_type(
    action_sha256s: tuple[str, ...],
    selection_count: int,
) -> type[BaseModel]:
    literal = Literal.__getitem__(action_sha256s)
    result = create_model(
        "OutcomeBlindMaterializedPortfolioSelection",
        __base__=_JudgeOutputBase,
        __module__=__name__,
        selected_action_sha256s=(
            list[literal],
            Field(min_length=selection_count, max_length=selection_count),
        ),
        decision_rationale=(
            str,
            Field(min_length=1, max_length=16_384),
        ),
        rejected_high_risk_pattern=(
            str,
            Field(min_length=1, max_length=8_192),
        ),
    )
    result.selection_count = selection_count
    return result


def _action_identity_record(action: Any) -> dict[str, object]:
    return {
        "action_sha256": action.action_sha256,
        "configuration_sha256": action.configuration_sha256,
        "configuration": thaw_json(action.configuration),
        "expert_id": action.expert_id,
        "native_rank": action.native_rank,
        "parent_candidate_ids": [value.value for value in action.parent_ids],
        "operator_id": action.operator_id,
        "role_id": action.role_id,
        "reference_action": action.reference_action,
    }


def render_materialized_portfolio_judge_prompt(
    request: ResidualPortfolioDecisionRequest,
    proposals: tuple[MaterializedActionProposalBatch, ...],
    projection: MaterializedPortfolioJudgePromptProjection,
    *,
    selection_count: int | None = None,
) -> str:
    """Render an audit-bound identity envelope plus cognitive workload view."""

    if type(request) is not ResidualPortfolioDecisionRequest:
        raise TypeError("request must be exact")
    request.__post_init__()
    if type(proposals) is not tuple or not proposals:
        raise ValueError("proposals must be a non-empty exact tuple")
    for proposal in proposals:
        if type(proposal) is not MaterializedActionProposalBatch:
            raise TypeError("proposals must contain exact batches")
        proposal.__post_init__()
        proposal.require_request(request)
    if type(projection) is not MaterializedPortfolioJudgePromptProjection:
        raise TypeError("projection must be exact")
    projection.__post_init__()
    actions = tuple(
        action for proposal in proposals for action in proposal.actions
    )
    action_sha256s = tuple(
        sorted(value.action_sha256 for value in actions)
    )
    if projection.action_sha256s != action_sha256s:
        raise ValueError("prompt projection does not cover the sealed action union")
    count = request.evaluation_slots if selection_count is None else selection_count
    if type(count) is not int or not 1 <= count <= request.evaluation_slots:
        raise ValueError("selection_count must fit evaluation capacity")
    envelope = {
        "schema_version": 1,
        "residual_request_sha256": request.request_sha256,
        "prior_state_sha256": request.prior_state_sha256,
        "decision_index": request.decision_index,
        "phase": request.phase.value,
        "evaluation_slots": request.evaluation_slots,
        "proposal_sha256s": sorted(
            value.proposal_sha256 for value in proposals
        ),
        "sealed_actions": [
            _action_identity_record(value)
            for value in sorted(actions, key=lambda item: item.action_sha256)
        ],
        "cognitive_projection": {
            **projection.to_record(),
            "payload": thaw_json(projection.payload),
        },
        "candidate_outcomes_observed": False,
    }
    encoded = json.dumps(
        envelope,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "\n".join(
        (
            projection.instruction,
            "",
            "OUTCOME-BLIND MATERIALIZED PORTFOLIO CONTRACT",
            encoded,
            "Compare the complete sealed population and nominate exactly "
            f"{count} distinct action_sha256 values. "
            "Maximize expected joint archive improvement, not agreement with "
            "native ranks. The downstream broker or other authenticated "
            "allocators own every unreserved evaluation slot; do not assume "
            "your nominations are the complete final slate. Candidate outcomes "
            "are unavailable. Numeric values "
            "in the cognitive projection are ordinary decimal quantities; "
            "hashes and audit encodings are identities, not magnitudes.",
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
        raise TypeError("low-level response value does not match judge output")
    return response, attempt_count


def _telemetry_record(
    response: StructuredGenerationResponse[Any],
    *,
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


@dataclass(slots=True)
class PydanticAIMaterializedPortfolioJudge:
    """Translate one queued structured call into an allocation requirement."""

    generate_once: LowLevelRunner
    prompt_projection: MaterializedPortfolioJudgePromptProjectionPort
    call_id_factory: Callable[
        [ResidualPortfolioDecisionRequest],
        LLMCallId,
    ]
    max_output_tokens: int = 131_072
    temperature: float | None = None
    nomination_slots: int | None = None

    policy_id: ClassVar[str] = MATERIALIZED_PORTFOLIO_JUDGE_POLICY_ID
    policy_version: ClassVar[int] = MATERIALIZED_PORTFOLIO_JUDGE_POLICY_VERSION
    definition_sha256: ClassVar[str] = (
        MATERIALIZED_PORTFOLIO_JUDGE_POLICY_DEFINITION_SHA256
    )

    def __post_init__(self) -> None:
        if not callable(self.generate_once):
            raise TypeError("generate_once must be callable")
        if not isinstance(
            self.prompt_projection,
            MaterializedPortfolioJudgePromptProjectionPort,
        ):
            raise TypeError(
                "prompt_projection must implement its projection port"
            )
        _require_token(
            self.prompt_projection.projection_id,
            name="projection_id",
        )
        if (
            type(self.prompt_projection.projection_version) is not int
            or self.prompt_projection.projection_version <= 0
        ):
            raise ValueError("projection_version must be positive")
        require_sha256(
            self.prompt_projection.definition_sha256,
            "projection definition_sha256",
        )
        if not callable(self.call_id_factory):
            raise TypeError("call_id_factory must be callable")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.nomination_slots is not None and (
            type(self.nomination_slots) is not int
            or self.nomination_slots <= 0
        ):
            raise ValueError("nomination_slots must be positive or None")

    async def require(
        self,
        request: ResidualPortfolioDecisionRequest,
        proposals: tuple[MaterializedActionProposalBatch, ...],
    ) -> MaterializedActionAllocationRequirement:
        self.__post_init__()
        if type(request) is not ResidualPortfolioDecisionRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        projection = self.prompt_projection.project(request, proposals)
        if type(projection) is not MaterializedPortfolioJudgePromptProjection:
            raise TypeError("prompt projection returned a foreign value")
        projection.__post_init__()
        if (
            projection.projection_id,
            projection.projection_version,
            projection.projection_definition_sha256,
        ) != (
            self.prompt_projection.projection_id,
            self.prompt_projection.projection_version,
            self.prompt_projection.definition_sha256,
        ):
            raise ValueError("prompt projection has a foreign identity")
        selection_count = (
            request.evaluation_slots
            if self.nomination_slots is None
            else self.nomination_slots
        )
        if selection_count > request.evaluation_slots:
            raise ValueError(
                "nomination_slots cannot exceed evaluation capacity"
            )
        prompt = render_materialized_portfolio_judge_prompt(
            request,
            proposals,
            projection,
            selection_count=selection_count,
        )
        output_type = _judge_output_type(
            projection.action_sha256s,
            selection_count,
        )
        call_id = self.call_id_factory(request)
        if type(call_id) is not LLMCallId:
            raise TypeError("call_id_factory must return an exact LLMCallId")
        low_level_request = StructuredGenerationRequest(
            call_id=call_id,
            operation="materialized_portfolio_judging",
            prompt=prompt,
            output_type=output_type,
            output_tool_name=MATERIALIZED_PORTFOLIO_JUDGE_TOOL_NAME,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )
        raw = await self.generate_once(low_level_request)
        response, attempt_count = _validated_response(
            raw,
            output_type=output_type,
        )
        value = cast(Any, response.value)
        ordered_selection = tuple(value.selected_action_sha256s)
        return MaterializedActionAllocationRequirement(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            residual_request_sha256=request.request_sha256,
            proposal_sha256s=tuple(
                sorted(item.proposal_sha256 for item in proposals)
            ),
            required_action_sha256s=tuple(sorted(ordered_selection)),
            candidate_outcomes_observed=False,
            evidence=freeze_json(
                {
                    "call_id": call_id.value,
                    "prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8", errors="strict")
                    ).hexdigest(),
                    "projection": projection.to_record(),
                    "ordered_selected_action_sha256s": list(
                        ordered_selection
                    ),
                    "nomination_slots": selection_count,
                    "evaluation_slots": request.evaluation_slots,
                    "downstream_unreserved_slots": (
                        request.evaluation_slots - selection_count
                    ),
                    "decision_rationale": value.decision_rationale,
                    "rejected_high_risk_pattern": (
                        value.rejected_high_risk_pattern
                    ),
                    "telemetry": _telemetry_record(
                        response,
                        attempt_count=attempt_count,
                    ),
                    "candidate_outcomes_observed": False,
                }
            ),
        )


__all__ = [
    "MATERIALIZED_PORTFOLIO_JUDGE_POLICY_DEFINITION_SHA256",
    "MATERIALIZED_PORTFOLIO_JUDGE_POLICY_ID",
    "MATERIALIZED_PORTFOLIO_JUDGE_POLICY_VERSION",
    "MATERIALIZED_PORTFOLIO_JUDGE_TOOL_NAME",
    "MaterializedPortfolioJudgePromptProjection",
    "MaterializedPortfolioJudgePromptProjectionPort",
    "PydanticAIMaterializedPortfolioJudge",
    "render_materialized_portfolio_judge_prompt",
]
