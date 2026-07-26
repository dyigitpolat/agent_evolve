"""Bounded, cancellation-safe asynchronous scheduling for provider-neutral LLM tasks.

This application service owns concurrency, attempt budgets, timeout enforcement,
and retry sleeps. Provider adapters only execute one attempt and classify their
exceptions; they never sleep or retry.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Deque, Dict, Generic, Optional, TypeVar, cast

from agent_evolve.domain.llm_task_queue import (
    AttemptRequestEvidence,
    AttemptStatus,
    AttemptTelemetry,
    CancellationReason,
    LLMAttemptContext,
    LLMTask,
    LLMTaskOutcome,
    RetryBudgetPartition,
    RetryBudgetUsage,
    QueueSnapshot,
    RetryClassification,
    RetryDisposition,
    RetryReason,
    SanitizedAttemptFailure,
    TaskOutcomeStatus,
    TaskTelemetry,
    retry_budget_partition,
)
from agent_evolve.ports.clock import Clock
from agent_evolve.ports.llm_task_queue import (
    AsyncLLMTaskExecutor,
    AsyncRuntime,
    AttemptPreparingExecutor,
    BackoffPolicy,
    ExecutorRetiredError,
    PreparedLLMAttempt,
    RetryClassifier,
)


RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")
CancellationOutcomeSink = Callable[[LLMTaskOutcome[ResponseT]], None]


class LLMTaskQueueError(RuntimeError):
    """Base for admission/lifecycle failures before a task is accepted."""


class LLMTaskQueueFullError(LLMTaskQueueError):
    pass


class LLMTaskQueueClosedError(LLMTaskQueueError):
    pass


class DuplicateLLMTaskError(LLMTaskQueueError):
    pass


@dataclass(slots=True)
class _Entry(Generic[RequestT, ResponseT]):
    task: LLMTask[RequestT]
    submitted_ns: int
    future: "asyncio.Future[LLMTaskOutcome[ResponseT]]"
    runner: Optional["asyncio.Task[None]"] = None
    cancellation_reason: Optional[CancellationReason] = None
    first_started_ns: Optional[int] = None


class AsyncLLMTaskQueue(Generic[RequestT, ResponseT]):
    """Run accepted LLM tasks with hard in-flight and pending bounds.

    ``submit`` admits atomically. If all execution slots and pending slots are
    occupied, it raises :class:`LLMTaskQueueFullError` instead of retaining an
    unbounded population of blocked producers. Once accepted, a task receives a
    typed outcome unless its own submitter is cancelled.
    """

    def __init__(
        self,
        *,
        executor: AsyncLLMTaskExecutor[RequestT, ResponseT],
        retry_classifier: RetryClassifier,
        backoff_policy: BackoffPolicy,
        clock: Clock,
        max_in_flight: int,
        max_pending: int,
        attempt_timeout_ns: int | None,
        runtime: AsyncRuntime,
    ) -> None:
        if not isinstance(executor, AsyncLLMTaskExecutor):
            raise TypeError("executor must implement AsyncLLMTaskExecutor")
        if not isinstance(retry_classifier, RetryClassifier):
            raise TypeError("retry_classifier must implement RetryClassifier")
        if not isinstance(backoff_policy, BackoffPolicy):
            raise TypeError("backoff_policy must implement BackoffPolicy")
        if not isinstance(clock, Clock):
            raise TypeError("clock must implement Clock")
        if type(max_in_flight) is not int or max_in_flight < 1:
            raise ValueError("max_in_flight must be a positive integer")
        if type(max_pending) is not int or max_pending < 0:
            raise ValueError("max_pending must be a non-negative integer")
        if attempt_timeout_ns is not None and (
            type(attempt_timeout_ns) is not int or attempt_timeout_ns <= 0
        ):
            raise ValueError(
                "attempt_timeout_ns must be a positive integer or None"
            )
        if not isinstance(runtime, AsyncRuntime):
            raise TypeError("runtime must implement AsyncRuntime")

        self._executor = executor
        self._retry_classifier = retry_classifier
        self._backoff_policy = backoff_policy
        self._clock = clock
        self._max_in_flight = max_in_flight
        self._max_pending = max_pending
        self._attempt_timeout_ns = attempt_timeout_ns
        self._runtime = runtime

        self._lock = asyncio.Lock()
        self._pending: Deque[_Entry[RequestT, ResponseT]] = deque()
        self._active: Dict[str, _Entry[RequestT, ResponseT]] = {}
        self._live_task_ids: set[str] = set()
        self._closed = False
        self._executor_retirement_owner: Optional[str] = None

    async def __aenter__(self) -> "AsyncLLMTaskQueue[RequestT, ResponseT]":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        await self.aclose()

    async def snapshot(self) -> QueueSnapshot:
        async with self._lock:
            return QueueSnapshot(
                max_in_flight=self._max_in_flight,
                max_pending=self._max_pending,
                in_flight=len(self._active),
                pending=len(self._pending),
                closed=self._closed,
            )

    async def submit(
        self,
        task: LLMTask[RequestT],
        *,
        cancellation_outcome_sink: CancellationOutcomeSink[ResponseT] | None = None,
    ) -> LLMTaskOutcome[ResponseT]:
        """Atomically admit and await one task.

        Cancelling this coroutine removes a pending task or cancels its active
        provider awaitable, then waits for scheduler cleanup before propagating
        ``CancelledError``.  If supplied, ``cancellation_outcome_sink`` receives
        the entry's one terminal outcome after that cleanup has drained and
        before cancellation propagates.  This narrow receipt hook lets an outer
        runner durably account for work whose submitter cannot consume the
        normal return value; it is never called on the ordinary return path.
        """

        if type(task) is not LLMTask:
            raise TypeError("task must be an exact LLMTask")
        if cancellation_outcome_sink is not None and not callable(
            cancellation_outcome_sink
        ):
            raise TypeError("cancellation_outcome_sink must be callable or None")
        loop = asyncio.get_running_loop()
        submitted_ns = self._clock.monotonic_ns()
        entry = _Entry(
            task=task,
            submitted_ns=submitted_ns,
            future=loop.create_future(),
        )

        async with self._lock:
            if self._closed:
                raise LLMTaskQueueClosedError("the LLM task queue is closed")
            if task.task_id in self._live_task_ids:
                raise DuplicateLLMTaskError(f"task {task.task_id!r} is already live")
            if len(self._active) < self._max_in_flight:
                self._live_task_ids.add(task.task_id)
                self._start_locked(entry)
            elif len(self._pending) < self._max_pending:
                self._live_task_ids.add(task.task_id)
                self._pending.append(entry)
            else:
                raise LLMTaskQueueFullError(
                    "all in-flight and pending LLM task slots are occupied"
                )

        try:
            return await asyncio.shield(entry.future)
        except asyncio.CancelledError:
            outcome = await self._cancel_for_submitter(entry)
            if cancellation_outcome_sink is not None:
                try:
                    cancellation_outcome_sink(outcome)
                except Exception:
                    # Recorder failures must not replace submitter cancellation.
                    # A higher layer that requires publication can surface a
                    # typed ``CancelledError`` subclass instead.
                    pass
            raise

    async def aclose(self) -> None:
        """Close admission and cancel all accepted work with typed outcomes."""

        runners: list[asyncio.Task[None]] = []
        async with self._lock:
            self._closed = True
            now_ns = self._clock.monotonic_ns()
            while self._pending:
                entry = self._pending.popleft()
                entry.cancellation_reason = CancellationReason.QUEUE_CLOSED
                self._resolve_pending_cancellation(entry, now_ns)
                self._live_task_ids.discard(entry.task.task_id)
            for entry in tuple(self._active.values()):
                entry.cancellation_reason = CancellationReason.QUEUE_CLOSED
                if entry.runner is not None and not entry.runner.done():
                    entry.runner.cancel()
                    runners.append(entry.runner)

        if runners:
            await asyncio.gather(*runners, return_exceptions=True)

    def _start_locked(self, entry: _Entry[RequestT, ResponseT]) -> None:
        task_id = entry.task.task_id
        if task_id in self._active or entry.runner is not None:
            raise RuntimeError("queue attempted to start an entry twice")
        self._active[task_id] = entry
        entry.runner = asyncio.create_task(
            self._run_entry(entry),
            name=f"agent-evolve-llm-{task_id}",
        )

    async def _cancel_for_submitter(
        self,
        entry: _Entry[RequestT, ResponseT],
    ) -> LLMTaskOutcome[ResponseT]:
        runner: Optional[asyncio.Task[None]] = None
        async with self._lock:
            if entry.task.task_id in self._live_task_ids:
                entry.cancellation_reason = CancellationReason.SUBMITTER_CANCELLED
                try:
                    self._pending.remove(entry)
                except ValueError:
                    runner = entry.runner
                    if runner is not None and not runner.done():
                        runner.cancel()
                else:
                    completed_ns = self._clock.monotonic_ns()
                    self._resolve_pending_cancellation(entry, completed_ns)
                    self._live_task_ids.discard(entry.task.task_id)

        if runner is not None:
            await asyncio.gather(runner, return_exceptions=True)
        # ``entry.future`` is shielded from submitter cancellation.  Pending
        # cancellation resolves it above; active cancellation resolves it only
        # after the provider awaitable and queue runner have fully drained.
        return await asyncio.shield(entry.future)

    def _safe_elapsed(self, later_ns: int, earlier_ns: int) -> int:
        elapsed = later_ns - earlier_ns
        if elapsed < 0:
            raise RuntimeError("injected monotonic clock moved backwards")
        return elapsed

    def _classify(
        self,
        error: Exception,
        *,
        context: LLMAttemptContext,
    ) -> tuple[RetryClassification, str]:
        try:
            classification = self._retry_classifier.classify(error, context=context)
            if type(classification) is not RetryClassification:
                raise TypeError("retry classifier returned a non-classification")
            return classification, type(error).__name__
        except Exception as classifier_error:
            return (
                RetryClassification(
                    disposition=RetryDisposition.FAIL,
                    reason=RetryReason.INTERNAL,
                ),
                type(classifier_error).__name__,
            )

    def _policy_delay(
        self,
        *,
        entry: _Entry[RequestT, ResponseT],
        attempt_number: int,
        classification: RetryClassification,
    ) -> tuple[Optional[int], Optional[str]]:
        try:
            delay = self._backoff_policy.delay_ns(
                task_id=entry.task.task_id,
                failed_attempt_number=attempt_number,
                classification=classification,
            )
            if type(delay) is not int or delay < 0:
                raise ValueError("backoff policy returned an invalid delay")
            return delay, None
        except Exception as policy_error:
            return None, type(policy_error).__name__

    async def _run_entry(self, entry: _Entry[RequestT, ResponseT]) -> None:
        attempts: list[AttemptTelemetry] = []
        previous_end_ns = entry.submitted_ns
        active_attempt_number: Optional[int] = None
        active_attempt_start_ns: Optional[int] = None
        active_request_evidence: Optional[AttemptRequestEvidence] = None
        outcome: Optional[LLMTaskOutcome[ResponseT]] = None
        previous_failure: Optional[SanitizedAttemptFailure] = None
        active_output_failure: Optional[SanitizedAttemptFailure] = None
        retry_budget_usage = (
            RetryBudgetUsage() if entry.task.retry_budget is not None else None
        )

        try:
            for attempt_number in range(1, entry.task.max_attempts + 1):
                active_attempt_number = attempt_number
                attempt_start_ns = self._clock.monotonic_ns()
                active_attempt_start_ns = attempt_start_ns
                if entry.first_started_ns is None:
                    entry.first_started_ns = attempt_start_ns
                wait_time_ns = self._safe_elapsed(attempt_start_ns, previous_end_ns)
                timeout_ns = (
                    entry.task.attempt_timeout_ns
                    if entry.task.attempt_timeout_ns is not None
                    else self._attempt_timeout_ns
                )
                context = LLMAttemptContext(
                    task_id=entry.task.task_id,
                    attempt_number=attempt_number,
                    attempt_timeout_ns=timeout_ns,
                    previous_failure=previous_failure,
                    active_output_failure=active_output_failure,
                    retry_budget_usage=retry_budget_usage,
                )

                try:
                    request_evidence: Optional[AttemptRequestEvidence] = None
                    if isinstance(self._executor, AttemptPreparingExecutor):
                        prepared = self._executor.prepare_attempt(
                            entry.task.request,
                            context=context,
                        )
                        if type(prepared) is not PreparedLLMAttempt:
                            raise TypeError(
                                "attempt preparing executor returned an invalid value"
                            )
                        PreparedLLMAttempt.__post_init__(prepared)
                        request_evidence = prepared.request_evidence
                        attempt_awaitable = prepared.execute_once()
                    else:
                        attempt_awaitable = self._executor.execute(
                            entry.task.request,
                            context=context,
                        )
                    active_request_evidence = request_evidence
                    response = await self._runtime.wait_for(
                        attempt_awaitable,
                        timeout_ns,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    attempt_end_ns = self._clock.monotonic_ns()
                    service_time_ns = self._safe_elapsed(
                        attempt_end_ns, attempt_start_ns
                    )
                    previous_end_ns = attempt_end_ns
                    classification, error_type = self._classify(error, context=context)
                    executor_retired = isinstance(error, ExecutorRetiredError)
                    if executor_retired and (
                        classification.disposition is not RetryDisposition.FAIL
                        or classification.retry_after is not None
                    ):
                        classification = RetryClassification(
                            disposition=RetryDisposition.FAIL,
                            reason=classification.reason,
                            sanitized_failure=classification.sanitized_failure,
                        )
                    timed_out = isinstance(error, TimeoutError)
                    attempts_remain = attempt_number < entry.task.max_attempts
                    budget_partition: Optional[RetryBudgetPartition] = None
                    budget_allows_retry = True
                    if (
                        classification.disposition is RetryDisposition.RETRY
                        and entry.task.retry_budget is not None
                    ):
                        try:
                            budget_partition = retry_budget_partition(
                                classification.reason
                            )
                        except (TypeError, ValueError):
                            classification = RetryClassification(
                                disposition=RetryDisposition.FAIL,
                                reason=RetryReason.INTERNAL,
                                sanitized_failure=(
                                    classification.sanitized_failure
                                ),
                            )
                            error_type = "InvalidRetryBudgetClassification"
                        else:
                            if retry_budget_usage is None:
                                raise AssertionError(
                                    "partitioned retry budget has no usage ledger"
                                )
                            budget_allows_retry = retry_budget_usage.used(
                                budget_partition
                            ) < entry.task.retry_budget.limit(budget_partition)
                    will_retry = (
                        classification.disposition is RetryDisposition.RETRY
                        and attempts_remain
                        and budget_allows_retry
                    )
                    policy_delay_ns = 0
                    retry_after_ns = (
                        classification.retry_after.delay_ns
                        if classification.retry_after is not None
                        else 0
                    )
                    scheduled_delay_ns = 0

                    if will_retry:
                        policy_delay, policy_error_type = self._policy_delay(
                            entry=entry,
                            attempt_number=attempt_number,
                            classification=classification,
                        )
                        if policy_delay is None:
                            classification = RetryClassification(
                                disposition=RetryDisposition.FAIL,
                                reason=RetryReason.INTERNAL,
                                sanitized_failure=(classification.sanitized_failure),
                            )
                            error_type = cast(str, policy_error_type)
                            retry_after_ns = 0
                            will_retry = False
                        else:
                            policy_delay_ns = policy_delay
                            scheduled_delay_ns = max(policy_delay_ns, retry_after_ns)

                    if will_retry and budget_partition is not None:
                        if retry_budget_usage is None:
                            raise AssertionError(
                                "partitioned retry budget has no usage ledger"
                            )
                        retry_budget_usage = retry_budget_usage.consume(
                            budget_partition
                        )

                    attempts.append(
                        AttemptTelemetry(
                            attempt_number=attempt_number,
                            status=(
                                AttemptStatus.TIMED_OUT
                                if timed_out
                                else (
                                    AttemptStatus.RETRYABLE_FAILURE
                                    if classification.disposition
                                    is RetryDisposition.RETRY
                                    else AttemptStatus.TERMINAL_FAILURE
                                )
                            ),
                            wait_time_ns=wait_time_ns,
                            service_time_ns=service_time_ns,
                            will_retry=will_retry,
                            policy_backoff_ns=policy_delay_ns,
                            retry_after_ns=retry_after_ns,
                            scheduled_delay_ns=scheduled_delay_ns,
                            classification=classification,
                            error_type=error_type,
                            request_evidence=request_evidence,
                        )
                    )
                    active_attempt_number = None
                    active_attempt_start_ns = None
                    active_request_evidence = None

                    if executor_retired:
                        await self._retire_executor(entry)

                    if not will_retry:
                        completed_ns = self._clock.monotonic_ns()
                        outcome = self._failure_outcome(
                            entry,
                            attempts,
                            completed_ns=completed_ns,
                            exhausted=(
                                classification.disposition is RetryDisposition.RETRY
                                and (
                                    not attempts_remain
                                    or not budget_allows_retry
                                )
                            ),
                        )
                        break

                    current_failure = classification.sanitized_failure
                    if (
                        current_failure is not None
                        and current_failure.kind == "output_invalid"
                        and current_failure.retryable
                    ):
                        active_output_failure = current_failure
                    previous_failure = current_failure
                    await self._runtime.sleep(scheduled_delay_ns)
                    continue

                attempt_end_ns = self._clock.monotonic_ns()
                attempts.append(
                    AttemptTelemetry(
                        attempt_number=attempt_number,
                        status=AttemptStatus.SUCCEEDED,
                        wait_time_ns=wait_time_ns,
                        service_time_ns=self._safe_elapsed(
                            attempt_end_ns,
                            attempt_start_ns,
                        ),
                        will_retry=False,
                        request_evidence=request_evidence,
                    )
                )
                active_attempt_number = None
                active_attempt_start_ns = None
                active_request_evidence = None
                outcome = self._success_outcome(
                    entry,
                    attempts,
                    response=response,
                    completed_ns=attempt_end_ns,
                )
                break

        except asyncio.CancelledError:
            completed_ns = self._clock.monotonic_ns()
            if (
                active_attempt_number is not None
                and active_attempt_start_ns is not None
            ):
                attempts.append(
                    AttemptTelemetry(
                        attempt_number=active_attempt_number,
                        status=AttemptStatus.CANCELLED,
                        wait_time_ns=self._safe_elapsed(
                            active_attempt_start_ns,
                            previous_end_ns,
                        ),
                        service_time_ns=self._safe_elapsed(
                            completed_ns,
                            active_attempt_start_ns,
                        ),
                        will_retry=False,
                        error_type="CancelledError",
                        request_evidence=active_request_evidence,
                    )
                )
            outcome = self._cancelled_outcome(
                entry, attempts, completed_ns=completed_ns
            )
        except Exception as internal_error:
            completed_ns = self._clock.monotonic_ns()
            internal_classification = RetryClassification(
                disposition=RetryDisposition.FAIL,
                reason=RetryReason.INTERNAL,
            )
            if (
                active_attempt_number is not None
                and active_attempt_start_ns is not None
            ):
                attempts.append(
                    AttemptTelemetry(
                        attempt_number=active_attempt_number,
                        status=AttemptStatus.TERMINAL_FAILURE,
                        wait_time_ns=self._safe_elapsed(
                            active_attempt_start_ns,
                            previous_end_ns,
                        ),
                        service_time_ns=self._safe_elapsed(
                            completed_ns,
                            active_attempt_start_ns,
                        ),
                        will_retry=False,
                        classification=internal_classification,
                        error_type=type(internal_error).__name__,
                        request_evidence=active_request_evidence,
                    )
                )
            elif attempts:
                last = attempts[-1]
                last_failure = (
                    None
                    if last.classification is None
                    else last.classification.sanitized_failure
                )
                attempts[-1] = replace(
                    last,
                    status=AttemptStatus.TERMINAL_FAILURE,
                    will_retry=False,
                    policy_backoff_ns=0,
                    retry_after_ns=0,
                    scheduled_delay_ns=0,
                    classification=RetryClassification(
                        disposition=internal_classification.disposition,
                        reason=internal_classification.reason,
                        sanitized_failure=last_failure,
                    ),
                    error_type=type(internal_error).__name__,
                )
            else:
                if entry.first_started_ns is None:
                    entry.first_started_ns = completed_ns
                attempts.append(
                    AttemptTelemetry(
                        attempt_number=1,
                        status=AttemptStatus.TERMINAL_FAILURE,
                        wait_time_ns=self._safe_elapsed(
                            entry.first_started_ns,
                            entry.submitted_ns,
                        ),
                        service_time_ns=self._safe_elapsed(
                            completed_ns,
                            entry.first_started_ns,
                        ),
                        will_retry=False,
                        classification=internal_classification,
                        error_type=type(internal_error).__name__,
                    )
                )
            outcome = self._failure_outcome(
                entry,
                attempts,
                completed_ns=completed_ns,
                exhausted=False,
            )
        finally:
            await self._finish_entry(entry)
            if outcome is not None and not entry.future.done():
                entry.future.set_result(outcome)

    def _telemetry(
        self,
        entry: _Entry[RequestT, ResponseT],
        attempts: list[AttemptTelemetry],
        *,
        completed_ns: int,
    ) -> TaskTelemetry:
        if entry.first_started_ns is None:
            queue_time_ns = self._safe_elapsed(completed_ns, entry.submitted_ns)
            service_time_ns = 0
        else:
            queue_time_ns = self._safe_elapsed(
                entry.first_started_ns, entry.submitted_ns
            )
            service_time_ns = self._safe_elapsed(completed_ns, entry.first_started_ns)
        return TaskTelemetry(
            task_id=entry.task.task_id,
            queue_time_ns=queue_time_ns,
            service_time_ns=service_time_ns,
            total_time_ns=self._safe_elapsed(completed_ns, entry.submitted_ns),
            attempts=tuple(attempts),
        )

    def _success_outcome(
        self,
        entry: _Entry[RequestT, ResponseT],
        attempts: list[AttemptTelemetry],
        *,
        response: ResponseT,
        completed_ns: int,
    ) -> LLMTaskOutcome[ResponseT]:
        return LLMTaskOutcome(
            status=TaskOutcomeStatus.SUCCEEDED,
            response=response,
            telemetry=self._telemetry(entry, attempts, completed_ns=completed_ns),
        )

    def _failure_outcome(
        self,
        entry: _Entry[RequestT, ResponseT],
        attempts: list[AttemptTelemetry],
        *,
        completed_ns: int,
        exhausted: bool,
    ) -> LLMTaskOutcome[ResponseT]:
        return LLMTaskOutcome(
            status=(
                TaskOutcomeStatus.ATTEMPTS_EXHAUSTED
                if exhausted
                else TaskOutcomeStatus.TERMINAL_FAILURE
            ),
            telemetry=self._telemetry(entry, attempts, completed_ns=completed_ns),
        )

    def _cancelled_outcome(
        self,
        entry: _Entry[RequestT, ResponseT],
        attempts: list[AttemptTelemetry],
        *,
        completed_ns: int,
    ) -> LLMTaskOutcome[ResponseT]:
        return LLMTaskOutcome(
            status=TaskOutcomeStatus.CANCELLED,
            cancellation_reason=(
                entry.cancellation_reason or CancellationReason.SUBMITTER_CANCELLED
            ),
            telemetry=self._telemetry(entry, attempts, completed_ns=completed_ns),
        )

    def _resolve_pending_cancellation(
        self,
        entry: _Entry[RequestT, ResponseT],
        completed_ns: int,
    ) -> None:
        if not entry.future.done():
            entry.future.set_result(
                self._cancelled_outcome(entry, [], completed_ns=completed_ns)
            )

    async def _finish_entry(self, entry: _Entry[RequestT, ResponseT]) -> None:
        async with self._lock:
            task_id = entry.task.task_id
            if self._active.get(task_id) is entry:
                del self._active[task_id]
            self._live_task_ids.discard(task_id)
            if not self._closed and self._pending:
                promoted = self._pending.popleft()
                self._start_locked(promoted)

    async def _retire_executor(self, source: _Entry[RequestT, ResponseT]) -> None:
        """Fail closed after an attempt permanently retires shared execution."""

        sibling_runners: list[asyncio.Task[None]] = []
        async with self._lock:
            # Only one runner may coordinate retirement.  A cancelled sibling
            # can still finish its transport-abort path and arrive here; if it
            # also cancelled and awaited the first runner, the two Tasks would
            # form a cancellation cycle inside asyncio.gather.
            if self._executor_retirement_owner is not None:
                return
            self._executor_retirement_owner = source.task.task_id
            self._closed = True
            now_ns = self._clock.monotonic_ns()
            while self._pending:
                entry = self._pending.popleft()
                entry.cancellation_reason = CancellationReason.EXECUTOR_RETIRED
                self._resolve_pending_cancellation(entry, now_ns)
                self._live_task_ids.discard(entry.task.task_id)
            for entry in tuple(self._active.values()):
                if entry is source:
                    continue
                entry.cancellation_reason = CancellationReason.EXECUTOR_RETIRED
                if entry.runner is not None and not entry.runner.done():
                    entry.runner.cancel()
                    sibling_runners.append(entry.runner)

        if sibling_runners:
            await asyncio.gather(*sibling_runners, return_exceptions=True)
