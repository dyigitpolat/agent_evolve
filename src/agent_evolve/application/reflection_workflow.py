"""Provider-neutral orchestration strategies for structured reflection calls.

The evolutionary engine owns evidence construction and memory publication.
This module owns only how an already-frozen reflection batch is divided into
logical model calls.  Keeping that seam in the application layer lets an
experiment change reflection topology without teaching the provider adapter or
any benchmark about the change.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agent_evolve.domain.ids import LLMCallId
from agent_evolve.ports.agentic_generator import (
    AgenticGenerator,
    InsightDraft,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    ReflectionInsightContract,
    validate_reflection_insight_draft,
)
from agent_evolve.ports.generation_failure import (
    GenerationFailureDisposition,
    classify_generation_failure,
)
from agent_evolve.ports.id_factory import IdFactory


_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPACE = re.compile(r"\s+")
ReflectionCallPlannedSink = Callable[
    ["PlannedReflectionCall | PlannedReflectionBatchCall"], None
]


@dataclass(frozen=True, slots=True)
class ReflectionPromptShard:
    """One exact engine-derived contrast and its self-contained prompt."""

    contrast_id: str
    prompt: str

    def __post_init__(self) -> None:
        if (
            type(self.contrast_id) is not str
            or _LOWER_SHA256.fullmatch(self.contrast_id) is None
        ):
            raise ValueError("contrast_id must be a lowercase SHA-256 digest")
        if type(self.prompt) is not str or not self.prompt.strip():
            raise ValueError("reflection shard prompt must be non-empty")


@dataclass(frozen=True, slots=True)
class ReflectionWorkflowRequest:
    """A complete staged reflection batch before any call ID is allocated."""

    operation: str
    shards: tuple[ReflectionPromptShard, ...]
    max_output_tokens: int
    temperature: float | None = None
    insight_contract: ReflectionInsightContract | None = None
    batch_prompt: str | None = None

    def __post_init__(self) -> None:
        if type(self.operation) is not str or not self.operation.strip():
            raise ValueError("reflection operation must be non-empty")
        if type(self.shards) is not tuple or not self.shards:
            raise ValueError("reflection workflow requires at least one shard")
        for shard in self.shards:
            if type(shard) is not ReflectionPromptShard:
                raise TypeError("shards must contain exact ReflectionPromptShard values")
            ReflectionPromptShard.__post_init__(shard)
        contrast_ids = tuple(shard.contrast_id for shard in self.shards)
        if len(set(contrast_ids)) != len(contrast_ids):
            raise ValueError("reflection workflow contrast IDs must be unique")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or float(self.temperature) < 0
        ):
            raise ValueError("temperature must be finite and non-negative or None")
        if self.insight_contract is not None:
            if type(self.insight_contract) is not ReflectionInsightContract:
                raise TypeError(
                    "insight_contract must be an exact ReflectionInsightContract"
                )
            ReflectionInsightContract.__post_init__(self.insight_contract)
        if self.batch_prompt is not None and (
            type(self.batch_prompt) is not str or not self.batch_prompt.strip()
        ):
            raise ValueError("batch_prompt must be non-empty exact text or None")


@dataclass(frozen=True, slots=True)
class PlannedReflectionCall:
    """One deterministic logical call allocated before concurrent execution."""

    contrast_id: str
    request: ReflectionGenerationRequest

    def __post_init__(self) -> None:
        if (
            type(self.contrast_id) is not str
            or _LOWER_SHA256.fullmatch(self.contrast_id) is None
        ):
            raise ValueError("contrast_id must be a lowercase SHA-256 digest")
        if type(self.request) is not ReflectionGenerationRequest:
            raise TypeError("request must be an exact ReflectionGenerationRequest")
        ReflectionGenerationRequest.__post_init__(self.request)
        if self.request.available_contrast_ids != (self.contrast_id,):
            raise ValueError("a sharded call must expose exactly its assigned contrast")
        if self.request.min_insights != 1 or self.request.max_insights != 1:
            raise ValueError("a sharded call must require exactly one insight")

    @property
    def call_id(self) -> LLMCallId:
        return self.request.call_id


@dataclass(frozen=True, slots=True)
class PlannedReflectionBatchCall:
    """One deterministic logical call covering an exact contrast batch."""

    contrast_ids: tuple[str, ...]
    request: ReflectionGenerationRequest

    def __post_init__(self) -> None:
        if type(self.contrast_ids) is not tuple or not self.contrast_ids:
            raise ValueError("a batched reflection call requires contrast IDs")
        if self.contrast_ids != tuple(sorted(set(self.contrast_ids))):
            raise ValueError("batched contrast IDs must be unique and canonical")
        if any(_LOWER_SHA256.fullmatch(item) is None for item in self.contrast_ids):
            raise ValueError("batched contrast IDs must be lowercase SHA-256 digests")
        if type(self.request) is not ReflectionGenerationRequest:
            raise TypeError("request must be an exact ReflectionGenerationRequest")
        ReflectionGenerationRequest.__post_init__(self.request)
        if self.request.available_contrast_ids != self.contrast_ids:
            raise ValueError("batched request exposes the wrong contrast set")
        expected_count = len(self.contrast_ids)
        if (
            self.request.min_insights != expected_count
            or self.request.max_insights != expected_count
        ):
            raise ValueError("batched request must require one card per contrast")

    @property
    def call_id(self) -> LLMCallId:
        return self.request.call_id


@dataclass(frozen=True, slots=True)
class ReflectionShardResult:
    """One contrast-bound card and the logical call that produced it.

    A contrast-sharded workflow has one singleton generation result per card.
    A strict batched workflow deliberately shares one multi-card generation
    result across every returned card.  In both cases the exact draft must be
    present in the bound result and must cite only this contrast.
    """

    contrast_id: str
    call_id: LLMCallId
    draft: InsightDraft
    generation_result: ReflectionGenerationResult

    def __post_init__(self) -> None:
        if (
            type(self.contrast_id) is not str
            or _LOWER_SHA256.fullmatch(self.contrast_id) is None
        ):
            raise ValueError("contrast_id must be a lowercase SHA-256 digest")
        if not isinstance(self.call_id, LLMCallId):
            raise TypeError("call_id must be an LLMCallId")
        if type(self.draft) is not InsightDraft:
            raise TypeError("draft must be an exact InsightDraft")
        InsightDraft.__post_init__(self.draft)
        if self.draft.evidence_contrast_ids != (self.contrast_id,):
            raise ValueError("accepted shard card must cite exactly its origin contrast")
        if type(self.generation_result) is not ReflectionGenerationResult:
            raise TypeError(
                "generation_result must be an exact ReflectionGenerationResult"
            )
        if self.draft not in self.generation_result.insights:
            raise ValueError("generation result does not contain its accepted card")


@dataclass(frozen=True, slots=True)
class ReflectionWorkflowResult:
    """A complete validated batch, still staged outside durable memory."""

    shards: tuple[ReflectionShardResult, ...]

    def __post_init__(self) -> None:
        if type(self.shards) is not tuple or not self.shards:
            raise ValueError("reflection workflow result requires accepted shards")
        if any(type(shard) is not ReflectionShardResult for shard in self.shards):
            raise TypeError("shards must contain exact ReflectionShardResult values")
        for shard in self.shards:
            ReflectionShardResult.__post_init__(shard)
        contrast_ids = tuple(shard.contrast_id for shard in self.shards)
        if contrast_ids != tuple(sorted(set(contrast_ids))):
            raise ValueError("accepted shards must use unique canonical contrast order")

    @property
    def logical_llm_calls_used(self) -> int:
        return len({shard.call_id for shard in self.shards})

    @property
    def call_ids(self) -> tuple[LLMCallId, ...]:
        """Return unique call identities in first-card canonical order."""

        return tuple(dict.fromkeys(shard.call_id for shard in self.shards))


@dataclass(frozen=True, slots=True)
class ReflectionShardFailure:
    """Content-free terminal status retained after all sibling calls settle."""

    contrast_id: str
    call_id: LLMCallId
    error_type: str
    disposition: GenerationFailureDisposition

    def __post_init__(self) -> None:
        if (
            type(self.contrast_id) is not str
            or _LOWER_SHA256.fullmatch(self.contrast_id) is None
        ):
            raise ValueError("contrast_id must be a lowercase SHA-256 digest")
        if not isinstance(self.call_id, LLMCallId):
            raise TypeError("call_id must be an LLMCallId")
        if type(self.error_type) is not str or not self.error_type:
            raise ValueError("error_type must be non-empty")
        if type(self.disposition) is not GenerationFailureDisposition:
            raise TypeError("disposition must be a GenerationFailureDisposition")


class ReflectionCardContractError(ValueError):
    """A terminal model card escaped a shard's exact typed contract."""

    @property
    def generation_failure_disposition(self) -> GenerationFailureDisposition:
        return GenerationFailureDisposition.MODEL_OR_SCHEMA_FAILURE


class ReflectionWorkflowExecutionError(RuntimeError):
    """One or more calls failed after every concurrently submitted shard settled."""

    def __init__(
        self,
        *,
        failures: tuple[ReflectionShardFailure, ...],
        completed: tuple[ReflectionShardResult, ...],
    ) -> None:
        if not failures:
            raise ValueError("workflow execution error requires at least one failure")
        self.failures = failures
        self.completed = completed
        super().__init__("contrast-sharded reflection did not yield a complete batch")

    @property
    def generation_failure_disposition(self) -> GenerationFailureDisposition:
        if all(
            failure.disposition
            is GenerationFailureDisposition.MODEL_OR_SCHEMA_FAILURE
            for failure in self.failures
        ):
            return GenerationFailureDisposition.MODEL_OR_SCHEMA_FAILURE
        return GenerationFailureDisposition.INFRASTRUCTURE_FAILURE


@runtime_checkable
class ReflectionWorkflow(Protocol):
    """Application seam for changing reflection topology independently of a domain."""

    async def run(
        self,
        request: ReflectionWorkflowRequest,
        *,
        generator: AgenticGenerator,
        id_factory: IdFactory,
        call_planned_sink: ReflectionCallPlannedSink | None = None,
    ) -> ReflectionWorkflowResult: ...


class ContrastShardedReflectionWorkflow:
    """Generate one strict card per contrast with deterministic concurrent fan-out."""

    policy_id = "contrast_sharded_reflection"
    policy_version = 1

    async def run(
        self,
        request: ReflectionWorkflowRequest,
        *,
        generator: AgenticGenerator,
        id_factory: IdFactory,
        call_planned_sink: ReflectionCallPlannedSink | None = None,
    ) -> ReflectionWorkflowResult:
        if type(request) is not ReflectionWorkflowRequest:
            raise TypeError("request must be an exact ReflectionWorkflowRequest")
        ReflectionWorkflowRequest.__post_init__(request)
        if not isinstance(generator, AgenticGenerator):
            raise TypeError("generator must implement AgenticGenerator")
        if not isinstance(id_factory, IdFactory):
            raise TypeError("id_factory must implement IdFactory")
        if call_planned_sink is not None and not callable(call_planned_sink):
            raise TypeError("call_planned_sink must be callable or None")

        planned: list[PlannedReflectionCall] = []
        for shard in sorted(request.shards, key=lambda value: value.contrast_id):
            call = PlannedReflectionCall(
                contrast_id=shard.contrast_id,
                request=ReflectionGenerationRequest(
                    call_id=id_factory.new_llm_call_id(),
                    operation=request.operation,
                    prompt=shard.prompt,
                    max_insights=1,
                    min_insights=1,
                    max_output_tokens=request.max_output_tokens,
                    temperature=request.temperature,
                    available_contrast_ids=(shard.contrast_id,),
                    insight_contract=request.insight_contract,
                ),
            )
            planned.append(call)
            if call_planned_sink is not None:
                call_planned_sink(call)

        raw_results = await asyncio.gather(
            *(generator.reflect(call.request) for call in planned),
            return_exceptions=True,
        )
        completed: list[ReflectionShardResult] = []
        failures: list[ReflectionShardFailure] = []
        for call, raw in zip(planned, raw_results, strict=True):
            if isinstance(raw, asyncio.CancelledError):
                raise raw
            if isinstance(raw, BaseException):
                failures.append(
                    ReflectionShardFailure(
                        contrast_id=call.contrast_id,
                        call_id=call.call_id,
                        error_type=type(raw).__name__,
                        disposition=classify_generation_failure(raw),
                    )
                )
                continue
            try:
                if type(raw) is not ReflectionGenerationResult:
                    raise TypeError(
                        "generator returned a non-reflection result for a shard"
                    )
                if len(raw.insights) != 1:
                    raise ReflectionCardContractError(
                        "a contrast shard must return exactly one insight"
                    )
                draft = raw.insights[0]
                if type(draft) is not InsightDraft:
                    raise TypeError("generator returned a non-InsightDraft card")
                InsightDraft.__post_init__(draft)
                if draft.evidence_contrast_ids != (call.contrast_id,):
                    raise ReflectionCardContractError(
                        "a contrast shard must cite exactly its assigned contrast"
                    )
                if request.insight_contract is not None:
                    try:
                        validate_reflection_insight_draft(
                            draft,
                            request.insight_contract,
                        )
                    except (TypeError, ValueError) as exc:
                        raise ReflectionCardContractError(
                            "a contrast shard violated the insight contract"
                        ) from exc
                completed.append(
                    ReflectionShardResult(
                        contrast_id=call.contrast_id,
                        call_id=call.call_id,
                        draft=draft,
                        generation_result=raw,
                    )
                )
            except BaseException as error:
                if isinstance(error, asyncio.CancelledError):
                    raise
                failures.append(
                    ReflectionShardFailure(
                        contrast_id=call.contrast_id,
                        call_id=call.call_id,
                        error_type=type(error).__name__,
                        disposition=classify_generation_failure(error),
                    )
                )

        canonical_completed = tuple(
            sorted(completed, key=lambda value: value.contrast_id)
        )
        if failures:
            raise ReflectionWorkflowExecutionError(
                failures=tuple(sorted(failures, key=lambda value: value.contrast_id)),
                completed=canonical_completed,
            )
        normalized_claims = tuple(
            _SPACE.sub(" ", shard.draft.claim.strip().casefold())
            for shard in canonical_completed
        )
        if len(set(normalized_claims)) != len(normalized_claims):
            raise ReflectionCardContractError(
                "contrast shards produced duplicate normalized claims"
            )
        return ReflectionWorkflowResult(canonical_completed)


class StrictBatchedReflectionWorkflow:
    """Generate one exact card per contrast in one atomic logical call.

    This topology is intended for evaluator-limited loops where a separate
    provider call per contrast would dominate the resource budget.  It accepts
    a batch only when the model returns exactly one card for every contrast,
    every card cites exactly one distinct available contrast, all cards satisfy
    the shared insight contract, and normalized claims are unique.  Nothing is
    returned to durable memory on partial coverage.
    """

    policy_id = "strict_batched_reflection"
    policy_version = 1

    async def run(
        self,
        request: ReflectionWorkflowRequest,
        *,
        generator: AgenticGenerator,
        id_factory: IdFactory,
        call_planned_sink: ReflectionCallPlannedSink | None = None,
    ) -> ReflectionWorkflowResult:
        if type(request) is not ReflectionWorkflowRequest:
            raise TypeError("request must be an exact ReflectionWorkflowRequest")
        ReflectionWorkflowRequest.__post_init__(request)
        if not isinstance(generator, AgenticGenerator):
            raise TypeError("generator must implement AgenticGenerator")
        if not isinstance(id_factory, IdFactory):
            raise TypeError("id_factory must implement IdFactory")
        if call_planned_sink is not None and not callable(call_planned_sink):
            raise TypeError("call_planned_sink must be callable or None")
        if request.batch_prompt is None:
            raise ValueError("strict batched reflection requires batch_prompt")

        contrast_ids = tuple(sorted(shard.contrast_id for shard in request.shards))
        batched_request = ReflectionGenerationRequest(
            call_id=id_factory.new_llm_call_id(),
            operation=request.operation,
            prompt=request.batch_prompt,
            max_insights=len(contrast_ids),
            min_insights=len(contrast_ids),
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
            available_contrast_ids=contrast_ids,
            insight_contract=request.insight_contract,
        )
        call = PlannedReflectionBatchCall(
            contrast_ids=contrast_ids,
            request=batched_request,
        )
        if call_planned_sink is not None:
            call_planned_sink(call)

        try:
            raw = await generator.reflect(batched_request)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            raise ReflectionWorkflowExecutionError(
                failures=tuple(
                    ReflectionShardFailure(
                        contrast_id=contrast_id,
                        call_id=batched_request.call_id,
                        error_type=type(error).__name__,
                        disposition=classify_generation_failure(error),
                    )
                    for contrast_id in contrast_ids
                ),
                completed=(),
            ) from error

        if type(raw) is not ReflectionGenerationResult:
            raise ReflectionCardContractError(
                "generator returned a non-reflection batched result"
            )
        if len(raw.insights) != len(contrast_ids):
            raise ReflectionCardContractError(
                "batched reflection must return exactly one card per contrast"
            )

        by_contrast: dict[str, InsightDraft] = {}
        for draft in raw.insights:
            if type(draft) is not InsightDraft:
                raise ReflectionCardContractError(
                    "batched reflection returned a non-InsightDraft card"
                )
            InsightDraft.__post_init__(draft)
            if len(draft.evidence_contrast_ids) != 1:
                raise ReflectionCardContractError(
                    "every batched card must cite exactly one contrast"
                )
            contrast_id = draft.evidence_contrast_ids[0]
            if contrast_id not in contrast_ids or contrast_id in by_contrast:
                raise ReflectionCardContractError(
                    "batched reflection has foreign or duplicate contrast coverage"
                )
            if request.insight_contract is not None:
                try:
                    validate_reflection_insight_draft(
                        draft,
                        request.insight_contract,
                    )
                except (TypeError, ValueError) as exc:
                    raise ReflectionCardContractError(
                        "a batched card violated the insight contract"
                    ) from exc
            by_contrast[contrast_id] = draft

        if tuple(sorted(by_contrast)) != contrast_ids:
            raise ReflectionCardContractError(
                "batched reflection did not cover the exact contrast set"
            )
        normalized_claims = tuple(
            _SPACE.sub(" ", by_contrast[item].claim.strip().casefold())
            for item in contrast_ids
        )
        if len(set(normalized_claims)) != len(normalized_claims):
            raise ReflectionCardContractError(
                "batched reflection produced duplicate normalized claims"
            )
        return ReflectionWorkflowResult(
            tuple(
                ReflectionShardResult(
                    contrast_id=contrast_id,
                    call_id=batched_request.call_id,
                    draft=by_contrast[contrast_id],
                    generation_result=raw,
                )
                for contrast_id in contrast_ids
            )
        )


__all__ = [
    "ContrastShardedReflectionWorkflow",
    "PlannedReflectionCall",
    "PlannedReflectionBatchCall",
    "ReflectionCallPlannedSink",
    "ReflectionCardContractError",
    "ReflectionPromptShard",
    "ReflectionShardFailure",
    "ReflectionShardResult",
    "ReflectionWorkflow",
    "ReflectionWorkflowExecutionError",
    "ReflectionWorkflowRequest",
    "ReflectionWorkflowResult",
    "StrictBatchedReflectionWorkflow",
]
