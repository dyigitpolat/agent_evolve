"""Provider-neutral ports consumed by the asynchronous LLM task queue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, Protocol, TypeVar, runtime_checkable

from agent_evolve.domain.llm_task_queue import (
    AttemptRequestEvidence,
    LLMAttemptContext,
    RetryClassification,
)


RequestT_contra = TypeVar("RequestT_contra", contravariant=True)
ResponseT_co = TypeVar("ResponseT_co", covariant=True)
AwaitedT = TypeVar("AwaitedT")
PreparedResponseT = TypeVar("PreparedResponseT")


class ExecutorRetiredError(Exception):
    """Marker for an attempt failure that permanently retired its executor.

    The queue uses this benchmark- and provider-neutral signal to close
    admission and cancel sibling work instead of promoting or retrying tasks
    against a transport that can no longer execute requests.
    """


@dataclass(frozen=True, slots=True)
class PreparedLLMAttempt(Generic[PreparedResponseT]):
    """One already-derived attempt plus content-free request evidence."""

    execute_once: Callable[[], Awaitable[PreparedResponseT]]
    request_evidence: AttemptRequestEvidence

    def __post_init__(self) -> None:
        if not callable(self.execute_once):
            raise TypeError("execute_once must be callable")
        if type(self.request_evidence) is not AttemptRequestEvidence:
            raise TypeError("request_evidence must be an AttemptRequestEvidence")


@runtime_checkable
class AsyncLLMTaskExecutor(Protocol[RequestT_contra, ResponseT_co]):
    async def execute(
        self,
        request: RequestT_contra,
        *,
        context: LLMAttemptContext,
    ) -> ResponseT_co: ...


@runtime_checkable
class AttemptPreparingExecutor(Protocol[RequestT_contra, ResponseT_co]):
    def prepare_attempt(
        self,
        request: RequestT_contra,
        *,
        context: LLMAttemptContext,
    ) -> PreparedLLMAttempt[ResponseT_co]: ...


@runtime_checkable
class RetryClassifier(Protocol):
    def classify(
        self,
        error: Exception,
        *,
        context: LLMAttemptContext,
    ) -> RetryClassification: ...


@runtime_checkable
class BackoffPolicy(Protocol):
    def delay_ns(
        self,
        *,
        task_id: str,
        failed_attempt_number: int,
        classification: RetryClassification,
    ) -> int: ...


@runtime_checkable
class AsyncRuntime(Protocol):
    async def sleep(self, delay_ns: int) -> None: ...

    async def wait_for(
        self,
        awaitable: Awaitable[AwaitedT],
        timeout_ns: int | None,
    ) -> AwaitedT: ...
