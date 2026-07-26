"""Deterministic, provider-free tests for the asynchronous LLM task queue."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Any, Awaitable, Deque, Iterable, Optional, TypeVar

import pytest

from agent_evolve.application.llm_task_queue import (
    AsyncLLMTaskQueue,
    DuplicateLLMTaskError,
    LLMTaskQueueClosedError,
    LLMTaskQueueFullError,
)
from agent_evolve.domain.llm_task_queue import (
    AttemptStatus,
    CancellationReason,
    LLMAttemptContext,
    LLMTask,
    PartitionedRetryBudget,
    RetryAfter,
    RetryAfterSource,
    RetryClassification,
    RetryDisposition,
    RetryReason,
    SanitizedAttemptFailure,
    TaskOutcomeStatus,
    parse_retry_after,
)
from agent_evolve.infrastructure.asyncio_runtime import (
    AsyncioRuntime,
    TransportAbortedTimeoutError,
)
from agent_evolve.infrastructure.clock import FakeClock
from agent_evolve.policies.llm_backoff import (
    DeterministicHashJitter,
    ExponentialBackoff,
    FullJitter,
    NoJitter,
)
from agent_evolve.ports.llm_task_queue import ExecutorRetiredError


T = TypeVar("T")


def run(awaitable: Awaitable[T]) -> T:
    return asyncio.run(awaitable)


async def eventually(predicate, *, turns: int = 100) -> None:
    for _ in range(turns):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


async def eventually_snapshot(
    queue: AsyncLLMTaskQueue[Any, Any],
    *,
    in_flight: int,
    pending: int,
    turns: int = 100,
) -> None:
    for _ in range(turns):
        snapshot = await queue.snapshot()
        if snapshot.in_flight == in_flight and snapshot.pending == pending:
            return
        await asyncio.sleep(0)
    raise AssertionError("queue did not reach the expected state")


class RateLimited(Exception):
    pass


class TransientFailure(Exception):
    pass


class OutputInvalidFailure(Exception):
    pass


class PermanentFailure(Exception):
    pass


class RetiredFailure(ExecutorRetiredError):
    pass


class TypedClassifier:
    def __init__(self, *, retry_after_ns: int = 0) -> None:
        self.retry_after_ns = retry_after_ns
        self.seen: list[tuple[type[Exception], LLMAttemptContext]] = []

    def classify(
        self,
        error: Exception,
        *,
        context: LLMAttemptContext,
    ) -> RetryClassification:
        self.seen.append((type(error), context))
        if isinstance(error, RateLimited):
            retry_after = (
                RetryAfter(
                    delay_ns=self.retry_after_ns,
                    source=RetryAfterSource.DELAY_SECONDS,
                )
                if self.retry_after_ns
                else None
            )
            return RetryClassification(
                disposition=RetryDisposition.RETRY,
                reason=RetryReason.RATE_LIMIT,
                retry_after=retry_after,
            )
        if isinstance(error, (TransientFailure, TimeoutError)):
            return RetryClassification(
                disposition=RetryDisposition.RETRY,
                reason=(
                    RetryReason.TIMEOUT
                    if isinstance(error, TimeoutError)
                    else RetryReason.TRANSIENT
                ),
            )
        if isinstance(error, OutputInvalidFailure):
            return RetryClassification(
                disposition=RetryDisposition.RETRY,
                reason=RetryReason.OUTPUT_INVALID,
                sanitized_failure=SanitizedAttemptFailure(
                    kind="output_invalid",
                    retryable=True,
                    safe_message="structured output was invalid",
                ),
            )
        return RetryClassification(
            disposition=RetryDisposition.FAIL,
            reason=RetryReason.PERMANENT,
        )


class DeterministicRuntime:
    """Advance FakeClock for sleeps and selected timeout calls."""

    def __init__(
        self,
        clock: FakeClock,
        *,
        timeout_calls: Iterable[int] = (),
    ) -> None:
        self.clock = clock
        self.timeout_calls = frozenset(timeout_calls)
        self.wait_calls: list[int | None] = []
        self.sleep_calls: list[int] = []

    async def sleep(self, delay_ns: int) -> None:
        self.sleep_calls.append(delay_ns)
        self.clock.advance_ns(delay_ns)
        await asyncio.sleep(0)

    async def wait_for(
        self,
        awaitable: Awaitable[T],
        timeout_ns: int | None,
    ) -> T:
        self.wait_calls.append(timeout_ns)
        call_number = len(self.wait_calls)
        if call_number not in self.timeout_calls:
            return await awaitable
        assert timeout_ns is not None

        child = asyncio.create_task(awaitable)
        await asyncio.sleep(0)
        self.clock.advance_ns(timeout_ns)
        child.cancel()
        await asyncio.gather(child, return_exceptions=True)
        raise TimeoutError("deterministic attempt timeout")


class ScriptedExecutor:
    def __init__(
        self,
        clock: FakeClock,
        scripts: Iterable[object],
        *,
        service_ns: int = 0,
    ) -> None:
        self.clock = clock
        self.scripts: Deque[object] = deque(scripts)
        self.service_ns = service_ns
        self.contexts: list[LLMAttemptContext] = []

    async def execute(self, request: str, *, context: LLMAttemptContext) -> str:
        del request
        self.contexts.append(context)
        self.clock.advance_ns(self.service_ns)
        scripted = self.scripts.popleft()
        if isinstance(scripted, Exception):
            raise scripted
        return str(scripted)


class GateExecutor:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.releases: dict[str, asyncio.Event] = {}
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.active = 0
        self.max_active = 0
        self.service_ns: dict[str, int] = {}

    def release(self, request: str) -> None:
        self.releases.setdefault(request, asyncio.Event()).set()

    async def execute(self, request: str, *, context: LLMAttemptContext) -> str:
        del context
        self.started.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await self.releases.setdefault(request, asyncio.Event()).wait()
            self.clock.advance_ns(self.service_ns.get(request, 0))
            return f"response:{request}"
        except asyncio.CancelledError:
            self.cancelled.append(request)
            raise
        finally:
            self.active -= 1


class NeverExecutor:
    def __init__(self) -> None:
        self.contexts: list[LLMAttemptContext] = []
        self.cancelled = 0

    async def execute(self, request: str, *, context: LLMAttemptContext) -> str:
        del request
        self.contexts.append(context)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        raise AssertionError("unreachable")


class RetiringExecutor:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.retire = asyncio.Event()

    async def execute(self, request: str, *, context: LLMAttemptContext) -> str:
        del context
        self.started.append(request)
        if request == "source":
            await self.retire.wait()
            raise RetiredFailure
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.append(request)
            raise
        raise AssertionError("unreachable")


class SimultaneousTransportAbortExecutor:
    """Surface two completed transport aborts despite sibling cancellation.

    A transport abort is already terminal when queue-level retirement starts.
    Converting the sibling cancellation into the same terminal error models two
    provider attempts crossing their hard deadline in the same event-loop turn.
    """

    def __init__(self) -> None:
        self.started: list[str] = []
        self.release = asyncio.Event()
        self.cancelled_during_abort: list[str] = []

    async def execute(self, request: str, *, context: LLMAttemptContext) -> str:
        del context
        self.started.append(request)
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled_during_abort.append(request)
            raise TransportAbortedTimeoutError from None
        raise TransportAbortedTimeoutError


class RecoveryTopologyExecutor:
    """Independent permanent failure followed by two crossing hard aborts."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.release_terminal = asyncio.Event()
        self.release_aborts = asyncio.Event()
        self.cancelled: list[str] = []

    async def execute(self, request: str, *, context: LLMAttemptContext) -> str:
        del context
        self.started.append(request)
        if request == "independent":
            await self.release_terminal.wait()
            raise PermanentFailure
        if request.startswith("abort"):
            try:
                await self.release_aborts.wait()
            except asyncio.CancelledError:
                self.cancelled.append(request)
                raise TransportAbortedTimeoutError from None
            raise TransportAbortedTimeoutError
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.append(request)
            raise
        raise AssertionError("unreachable")


class BlockingSleepRuntime(DeterministicRuntime):
    def __init__(self, clock: FakeClock) -> None:
        super().__init__(clock)
        self.sleep_started = asyncio.Event()
        self.sleep_cancelled = False

    async def sleep(self, delay_ns: int) -> None:
        self.sleep_calls.append(delay_ns)
        self.sleep_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.sleep_cancelled = True
            raise


class ScriptedRandom:
    def __init__(self, values: Iterable[int]) -> None:
        self.values = deque(values)
        self.stops: list[int] = []

    def randrange(self, stop: int) -> int:
        self.stops.append(stop)
        return self.values.popleft()


def make_queue(
    *,
    executor: Any,
    clock: FakeClock,
    runtime: Optional[Any] = None,
    classifier: Optional[TypedClassifier] = None,
    backoff: Optional[Any] = None,
    max_in_flight: int = 1,
    max_pending: int = 1,
    timeout_ns: int | None = 100,
) -> AsyncLLMTaskQueue[str, str]:
    return AsyncLLMTaskQueue(
        executor=executor,
        retry_classifier=classifier or TypedClassifier(),
        backoff_policy=backoff or ExponentialBackoff(0, 0),
        clock=clock,
        runtime=runtime or DeterministicRuntime(clock),
        max_in_flight=max_in_flight,
        max_pending=max_pending,
        attempt_timeout_ns=timeout_ns,
    )


def test_retry_after_parser_supports_delta_and_http_date_without_side_effects():
    now = datetime(2026, 7, 13, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after("17", now_utc=now) == RetryAfter(
        17_000_000_000,
        RetryAfterSource.DELAY_SECONDS,
    )
    assert parse_retry_after(
        "Mon, 13 Jul 2026 12:00:05 GMT",
        now_utc=now,
    ) == RetryAfter(5_000_000_000, RetryAfterSource.HTTP_DATE)
    assert parse_retry_after(
        "Mon, 13 Jul 2026 11:59:00 GMT",
        now_utc=now,
    ) == RetryAfter(0, RetryAfterSource.HTTP_DATE)
    assert parse_retry_after("1.5", now_utc=now) is None
    assert parse_retry_after("not-a-date", now_utc=now) is None
    assert parse_retry_after(object(), now_utc=now) is None


def test_exponential_backoff_supports_exact_and_injected_full_jitter():
    classification = RetryClassification(
        disposition=RetryDisposition.RETRY,
        reason=RetryReason.TRANSIENT,
    )
    exact = ExponentialBackoff(10, 25, NoJitter())
    assert [
        exact.delay_ns(
            task_id="task",
            failed_attempt_number=n,
            classification=classification,
        )
        for n in range(1, 5)
    ] == [10, 20, 25, 25]

    random = ScriptedRandom([3, 20])
    jittered = ExponentialBackoff(10, 25, FullJitter(random))
    assert (
        jittered.delay_ns(
            task_id="task",
            failed_attempt_number=1,
            classification=classification,
        )
        == 3
    )
    assert (
        jittered.delay_ns(
            task_id="task",
            failed_attempt_number=2,
            classification=classification,
        )
        == 20
    )
    assert random.stops == [11, 21]

    rate_limited = RetryClassification(
        disposition=RetryDisposition.RETRY,
        reason=RetryReason.RATE_LIMIT,
    )
    floor_random = ScriptedRandom([0, 4])
    floored = ExponentialBackoff(
        10,
        25,
        FullJitter(floor_random),
        rate_limit_floor_ns=7,
    )
    assert floored.delay_ns(
        task_id="task",
        failed_attempt_number=1,
        classification=rate_limited,
    ) == 7
    assert floored.delay_ns(
        task_id="task",
        failed_attempt_number=2,
        classification=classification,
    ) == 4


def test_exponential_backoff_rejects_rate_limit_floor_above_cap() -> None:
    with pytest.raises(ValueError, match="rate_limit_floor_ns"):
        ExponentialBackoff(1, 10, rate_limit_floor_ns=11)


def test_deterministic_hash_jitter_is_task_keyed_and_matches_frozen_framing():
    jitter = DeterministicHashJitter(
        seed=20260714,
        domain="boils-shadow-jitter-v1",
    )
    assert (
        jitter.apply(
            1_000_000_000,
            task_id="call_a",
            failed_attempt_number=1,
        )
        == 348_744_833
    )
    assert (
        jitter.apply(
            2_000_000_000,
            task_id="call_a",
            failed_attempt_number=2,
        )
        == 1_390_739_801
    )
    assert (
        jitter.apply(
            1_000_000_000,
            task_id="call_b",
            failed_attempt_number=1,
        )
        == 816_642_073
    )
    assert (
        jitter.apply(
            2_000_000_000,
            task_id="call_b",
            failed_attempt_number=2,
        )
        == 1_458_681_908
    )
    assert (
        jitter.apply(
            1_000_000_000,
            task_id="call_a",
            failed_attempt_number=1,
        )
        == 348_744_833
    )


def test_hard_in_flight_and_pending_bounds_reject_excess_admission():
    async def scenario() -> None:
        clock = FakeClock()
        executor = GateExecutor(clock)
        queue = make_queue(
            executor=executor,
            clock=clock,
            max_in_flight=2,
            max_pending=1,
        )
        first = asyncio.create_task(queue.submit(LLMTask("first", "first", 1)))
        second = asyncio.create_task(queue.submit(LLMTask("second", "second", 1)))
        await eventually(lambda: len(executor.started) == 2)
        third = asyncio.create_task(queue.submit(LLMTask("third", "third", 1)))
        await eventually_snapshot(queue, in_flight=2, pending=1)
        assert await queue.snapshot() == queue_snapshot(2, 1, 2, 1, False)

        with pytest.raises(LLMTaskQueueFullError):
            await queue.submit(LLMTask("fourth", "fourth", 1))

        executor.release("first")
        await eventually(lambda: "third" in executor.started)
        executor.release("second")
        executor.release("third")
        outcomes = await asyncio.gather(first, second, third)
        assert [outcome.status for outcome in outcomes] == [
            TaskOutcomeStatus.SUCCEEDED,
            TaskOutcomeStatus.SUCCEEDED,
            TaskOutcomeStatus.SUCCEEDED,
        ]
        assert executor.max_active == 2
        assert (await queue.snapshot()).in_flight == 0
        await queue.aclose()

    run(scenario())


def queue_snapshot(
    max_in_flight: int,
    max_pending: int,
    in_flight: int,
    pending: int,
    closed: bool,
):
    from agent_evolve.domain.llm_task_queue import QueueSnapshot

    return QueueSnapshot(max_in_flight, max_pending, in_flight, pending, closed)


def test_queue_and_service_time_are_measured_from_the_injected_clock():
    async def scenario() -> None:
        clock = FakeClock()
        executor = GateExecutor(clock)
        executor.service_ns["second"] = 7
        queue = make_queue(executor=executor, clock=clock)

        first = asyncio.create_task(queue.submit(LLMTask("first", "first", 1)))
        await eventually(lambda: executor.started == ["first"])
        second = asyncio.create_task(queue.submit(LLMTask("second", "second", 1)))
        await eventually_snapshot(queue, in_flight=1, pending=1)
        clock.advance_ns(50)
        executor.release("first")
        await eventually(lambda: executor.started == ["first", "second"])
        executor.release("second")
        first_outcome, second_outcome = await asyncio.gather(first, second)

        assert first_outcome.telemetry.queue_time_ns == 0
        assert first_outcome.telemetry.service_time_ns == 50
        assert second_outcome.telemetry.queue_time_ns == 50
        assert second_outcome.telemetry.service_time_ns == 7
        assert second_outcome.telemetry.total_time_ns == 57
        assert second_outcome.telemetry.attempts[0].wait_time_ns == 50
        assert second_outcome.telemetry.attempts[0].service_time_ns == 7
        await queue.aclose()

    run(scenario())


def test_retry_budget_backoff_and_retry_after_produce_typed_telemetry():
    async def scenario() -> None:
        clock = FakeClock()
        runtime = DeterministicRuntime(clock)
        executor = ScriptedExecutor(
            clock,
            [RateLimited("limited"), TransientFailure("transient"), "ok"],
            service_ns=5,
        )
        classifier = TypedClassifier(retry_after_ns=25)
        queue = make_queue(
            executor=executor,
            clock=clock,
            runtime=runtime,
            classifier=classifier,
            backoff=ExponentialBackoff(10, 100),
        )

        outcome = await queue.submit(LLMTask("retry-task", "request", 3))
        assert outcome.status is TaskOutcomeStatus.SUCCEEDED
        assert outcome.response == "ok"
        assert runtime.sleep_calls == [25, 20]
        assert runtime.wait_calls == [100, 100, 100]
        assert [attempt.status for attempt in outcome.telemetry.attempts] == [
            AttemptStatus.RETRYABLE_FAILURE,
            AttemptStatus.RETRYABLE_FAILURE,
            AttemptStatus.SUCCEEDED,
        ]
        first, second, third = outcome.telemetry.attempts
        assert (
            first.policy_backoff_ns,
            first.retry_after_ns,
            first.scheduled_delay_ns,
        ) == (
            10,
            25,
            25,
        )
        assert (second.policy_backoff_ns, second.scheduled_delay_ns) == (20, 20)
        assert [attempt.wait_time_ns for attempt in (first, second, third)] == [
            0,
            25,
            20,
        ]
        assert [attempt.service_time_ns for attempt in (first, second, third)] == [
            5,
            5,
            5,
        ]
        assert outcome.telemetry.queue_time_ns == 0
        assert outcome.telemetry.service_time_ns == 60
        assert outcome.telemetry.total_time_ns == 60
        await queue.aclose()

    run(scenario())


def test_terminal_failure_does_not_retry_and_retryable_failure_exhausts_exact_budget():
    async def scenario() -> None:
        terminal_clock = FakeClock()
        terminal_executor = ScriptedExecutor(
            terminal_clock,
            [PermanentFailure("bad request")],
        )
        terminal_queue = make_queue(executor=terminal_executor, clock=terminal_clock)
        terminal = await terminal_queue.submit(LLMTask("terminal", "request", 5))
        assert terminal.status is TaskOutcomeStatus.TERMINAL_FAILURE
        assert len(terminal.telemetry.attempts) == 1
        assert terminal.telemetry.attempts[0].status is AttemptStatus.TERMINAL_FAILURE
        assert terminal.telemetry.attempts[0].error_type == "PermanentFailure"

        retry_clock = FakeClock()
        retry_runtime = DeterministicRuntime(retry_clock)
        retry_executor = ScriptedExecutor(
            retry_clock,
            [TransientFailure("one"), TransientFailure("two")],
        )
        retry_queue = make_queue(
            executor=retry_executor,
            clock=retry_clock,
            runtime=retry_runtime,
            backoff=ExponentialBackoff(10, 10),
        )
        exhausted = await retry_queue.submit(LLMTask("exhausted", "request", 2))
        assert exhausted.status is TaskOutcomeStatus.ATTEMPTS_EXHAUSTED
        assert len(exhausted.telemetry.attempts) == 2
        assert retry_runtime.sleep_calls == [10]
        assert exhausted.telemetry.attempts[-1].will_retry is False
        assert exhausted.telemetry.attempts[-1].scheduled_delay_ns == 0
        await terminal_queue.aclose()
        await retry_queue.aclose()

    run(scenario())


def test_partitioned_retry_budget_reserves_independent_failure_allowances():
    async def scenario() -> None:
        clock = FakeClock()
        executor = ScriptedExecutor(
            clock,
            [
                OutputInvalidFailure(),
                TransientFailure(),
                OutputInvalidFailure(),
                TransientFailure(),
                "success",
            ],
        )
        queue = make_queue(executor=executor, clock=clock)
        outcome = await queue.submit(
            LLMTask(
                "partitioned",
                "request",
                5,
                retry_budget=PartitionedRetryBudget(
                    output_invalid_retries=2,
                    transport_retries=2,
                ),
            )
        )

        assert outcome.status is TaskOutcomeStatus.SUCCEEDED
        assert [
            (
                context.retry_budget_usage.output_invalid_retries,
                context.retry_budget_usage.transport_retries,
            )
            for context in executor.contexts
        ] == [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]
        await queue.aclose()

    run(scenario())


def test_partitioned_retry_budget_does_not_borrow_from_another_partition():
    async def scenario() -> None:
        clock = FakeClock()
        executor = ScriptedExecutor(
            clock,
            [OutputInvalidFailure(), OutputInvalidFailure(), "must-not-run"],
        )
        queue = make_queue(executor=executor, clock=clock)
        outcome = await queue.submit(
            LLMTask(
                "no-borrowing",
                "request",
                5,
                retry_budget=PartitionedRetryBudget(
                    output_invalid_retries=1,
                    transport_retries=3,
                ),
            )
        )

        assert outcome.status is TaskOutcomeStatus.ATTEMPTS_EXHAUSTED
        assert len(outcome.telemetry.attempts) == 2
        assert outcome.telemetry.attempts[-1].will_retry is False
        assert len(executor.contexts) == 2
        await queue.aclose()

    run(scenario())


def test_timeout_cancels_each_attempt_and_obeys_the_same_attempt_budget():
    async def scenario() -> None:
        clock = FakeClock()
        runtime = DeterministicRuntime(clock, timeout_calls=(1, 2))
        executor = NeverExecutor()
        queue = make_queue(
            executor=executor,
            clock=clock,
            runtime=runtime,
            backoff=ExponentialBackoff(0, 0),
            timeout_ns=100,
        )

        outcome = await queue.submit(LLMTask("timeout", "request", 2))
        assert outcome.status is TaskOutcomeStatus.ATTEMPTS_EXHAUSTED
        assert [attempt.status for attempt in outcome.telemetry.attempts] == [
            AttemptStatus.TIMED_OUT,
            AttemptStatus.TIMED_OUT,
        ]
        assert [attempt.service_time_ns for attempt in outcome.telemetry.attempts] == [
            100,
            100,
        ]
        assert [context.attempt_number for context in executor.contexts] == [1, 2]
        assert executor.cancelled == 2
        assert runtime.sleep_calls == [0]
        assert outcome.telemetry.service_time_ns == 200
        await queue.aclose()

    run(scenario())


def test_asyncio_runtime_timeout_cancels_the_underlying_awaitable():
    async def scenario() -> None:
        cancelled = asyncio.Event()

        async def never() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with pytest.raises(TimeoutError):
            await AsyncioRuntime().wait_for(never(), 1)
        assert cancelled.is_set()

    run(scenario())


def test_asyncio_runtime_hard_timeout_aborts_transport_and_drains_attempt():
    async def scenario() -> None:
        cancellation_seen = asyncio.Event()
        transport_closed = asyncio.Event()
        attempt_finished = asyncio.Event()
        abort_calls = 0

        async def cancellation_resistant_attempt() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                # Model a provider stack whose shielded cancellation cleanup
                # cannot finish until its transport owner closes the socket.
                await transport_closed.wait()
                raise
            finally:
                attempt_finished.set()

        async def abort_transport() -> None:
            nonlocal abort_calls
            abort_calls += 1
            transport_closed.set()

        runtime = AsyncioRuntime(timeout_abort=abort_transport)
        with pytest.raises(TransportAbortedTimeoutError):
            await runtime.wait_for(cancellation_resistant_attempt(), 1)

        assert abort_calls == 1
        assert cancellation_seen.is_set()
        assert transport_closed.is_set()
        assert attempt_finished.is_set()
        current = asyncio.current_task()
        assert all(
            task is current or task.done()
            for task in asyncio.all_tasks()
        )

    run(scenario())


def test_submitter_cancellation_removes_pending_and_releases_active_capacity():
    async def scenario() -> None:
        clock = FakeClock()
        executor = GateExecutor(clock)
        queue = make_queue(executor=executor, clock=clock)

        active = asyncio.create_task(queue.submit(LLMTask("active", "active", 1)))
        await eventually(lambda: executor.started == ["active"])
        removed = asyncio.create_task(queue.submit(LLMTask("removed", "removed", 1)))
        await eventually_snapshot(queue, in_flight=1, pending=1)
        removed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await removed
        assert (await queue.snapshot()).pending == 0

        promoted = asyncio.create_task(queue.submit(LLMTask("promoted", "promoted", 1)))
        await eventually_snapshot(queue, in_flight=1, pending=1)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        await eventually(lambda: "promoted" in executor.started)
        assert executor.cancelled == ["active"]
        executor.release("promoted")
        outcome = await promoted
        assert outcome.status is TaskOutcomeStatus.SUCCEEDED
        snapshot = await queue.snapshot()
        assert (snapshot.in_flight, snapshot.pending) == (0, 0)
        await queue.aclose()

    run(scenario())


def test_submitter_cancellation_receipts_cover_pending_and_active_once_after_drain():
    async def scenario() -> None:
        clock = FakeClock()
        executor = GateExecutor(clock)
        queue = make_queue(executor=executor, clock=clock)
        receipts = []

        active = asyncio.create_task(
            queue.submit(
                LLMTask("active-receipt", "active-receipt", 1),
                cancellation_outcome_sink=receipts.append,
            )
        )
        await eventually(lambda: executor.started == ["active-receipt"])
        pending = asyncio.create_task(
            queue.submit(
                LLMTask("pending-receipt", "pending-receipt", 1),
                cancellation_outcome_sink=receipts.append,
            )
        )
        await eventually_snapshot(queue, in_flight=1, pending=1)

        clock.advance_ns(11)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert [item.telemetry.task_id for item in receipts] == ["pending-receipt"]
        pending_receipt = receipts[0]
        assert pending_receipt.status is TaskOutcomeStatus.CANCELLED
        assert (
            pending_receipt.cancellation_reason
            is CancellationReason.SUBMITTER_CANCELLED
        )
        assert pending_receipt.telemetry.attempts == ()
        assert pending_receipt.telemetry.queue_time_ns == 11
        assert pending_receipt.telemetry.service_time_ns == 0

        clock.advance_ns(7)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert executor.cancelled == ["active-receipt"]
        assert [item.telemetry.task_id for item in receipts] == [
            "pending-receipt",
            "active-receipt",
        ]
        active_receipt = receipts[1]
        assert active_receipt.status is TaskOutcomeStatus.CANCELLED
        assert (
            active_receipt.cancellation_reason
            is CancellationReason.SUBMITTER_CANCELLED
        )
        assert len(active_receipt.telemetry.attempts) == 1
        attempt = active_receipt.telemetry.attempts[0]
        assert attempt.status is AttemptStatus.CANCELLED
        assert attempt.service_time_ns == 18
        assert active_receipt.telemetry.service_time_ns == 18

        await queue.aclose()
        assert len(receipts) == 2
        assert await queue.snapshot() == queue_snapshot(1, 1, 0, 0, True)

    run(scenario())


def test_cancellation_receipt_sink_failure_cannot_replace_submitter_cancellation():
    async def scenario() -> None:
        clock = FakeClock()
        executor = GateExecutor(clock)
        queue = make_queue(executor=executor, clock=clock)
        sink_calls = 0

        def failing_sink(outcome) -> None:
            nonlocal sink_calls
            sink_calls += 1
            assert outcome.status is TaskOutcomeStatus.CANCELLED
            raise OSError("recorder unavailable SECRET_PATH")

        submitted = asyncio.create_task(
            queue.submit(
                LLMTask("failed-receipt", "failed-receipt", 1),
                cancellation_outcome_sink=failing_sink,
            )
        )
        await eventually(lambda: executor.started == ["failed-receipt"])
        submitted.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await submitted

        assert type(caught.value) is asyncio.CancelledError
        assert submitted.cancelled()
        assert sink_calls == 1
        assert executor.cancelled == ["failed-receipt"]
        await queue.aclose()
        assert sink_calls == 1

    run(scenario())


def test_submitter_cancellation_during_backoff_cleans_up_without_an_extra_attempt():
    async def scenario() -> None:
        clock = FakeClock()
        runtime = BlockingSleepRuntime(clock)
        executor = ScriptedExecutor(
            clock,
            [TransientFailure("retry"), "must-not-run"],
        )
        queue = make_queue(
            executor=executor,
            clock=clock,
            runtime=runtime,
            backoff=ExponentialBackoff(50, 50),
        )
        submitted = asyncio.create_task(
            queue.submit(LLMTask("backoff-cancel", "request", 2))
        )
        await runtime.sleep_started.wait()
        submitted.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submitted
        assert runtime.sleep_cancelled is True
        assert len(executor.contexts) == 1
        assert await queue.snapshot() == queue_snapshot(1, 1, 0, 0, False)
        await queue.aclose()

    run(scenario())


def test_close_resolves_active_and_pending_tasks_and_blocks_new_admission():
    async def scenario() -> None:
        clock = FakeClock()
        executor = GateExecutor(clock)
        queue = make_queue(executor=executor, clock=clock)
        active = asyncio.create_task(queue.submit(LLMTask("active", "active", 1)))
        await eventually(lambda: executor.started == ["active"])
        pending = asyncio.create_task(queue.submit(LLMTask("pending", "pending", 1)))
        await eventually_snapshot(queue, in_flight=1, pending=1)

        await queue.aclose()
        active_outcome, pending_outcome = await asyncio.gather(active, pending)
        assert active_outcome.status is TaskOutcomeStatus.CANCELLED
        assert pending_outcome.status is TaskOutcomeStatus.CANCELLED
        assert active_outcome.cancellation_reason is CancellationReason.QUEUE_CLOSED
        assert pending_outcome.cancellation_reason is CancellationReason.QUEUE_CLOSED
        assert pending_outcome.telemetry.attempts == ()
        assert executor.cancelled == ["active"]
        assert await queue.snapshot() == queue_snapshot(1, 1, 0, 0, True)
        with pytest.raises(LLMTaskQueueClosedError):
            await queue.submit(LLMTask("late", "late", 1))

    run(scenario())


def test_executor_retirement_closes_admission_and_cancels_all_sibling_work():
    async def scenario() -> None:
        clock = FakeClock()
        executor = RetiringExecutor()
        queue = make_queue(
            executor=executor,
            clock=clock,
            max_in_flight=2,
            max_pending=1,
        )
        sibling = asyncio.create_task(
            queue.submit(LLMTask("sibling", "sibling", 1))
        )
        source = asyncio.create_task(queue.submit(LLMTask("source", "source", 2)))
        await eventually(lambda: len(executor.started) == 2)
        pending = asyncio.create_task(queue.submit(LLMTask("pending", "pending", 1)))
        await eventually_snapshot(queue, in_flight=2, pending=1)

        executor.retire.set()
        source_outcome, sibling_outcome, pending_outcome = await asyncio.gather(
            source,
            sibling,
            pending,
        )

        assert source_outcome.status is TaskOutcomeStatus.TERMINAL_FAILURE
        assert source_outcome.telemetry.attempts[0].will_retry is False
        assert sibling_outcome.status is TaskOutcomeStatus.CANCELLED
        assert pending_outcome.status is TaskOutcomeStatus.CANCELLED
        assert sibling_outcome.cancellation_reason is CancellationReason.EXECUTOR_RETIRED
        assert pending_outcome.cancellation_reason is CancellationReason.EXECUTOR_RETIRED
        assert executor.cancelled == ["sibling"]
        assert await queue.snapshot() == queue_snapshot(2, 1, 0, 0, True)
        with pytest.raises(LLMTaskQueueClosedError):
            await queue.submit(LLMTask("late", "late", 1))

    run(scenario())


def test_simultaneous_transport_abort_retirement_has_one_owner_and_terminates():
    async def scenario() -> None:
        clock = FakeClock()
        executor = SimultaneousTransportAbortExecutor()
        queue = make_queue(
            executor=executor,
            clock=clock,
            max_in_flight=2,
            max_pending=1,
        )
        first = asyncio.create_task(queue.submit(LLMTask("first", "first", 2)))
        second = asyncio.create_task(queue.submit(LLMTask("second", "second", 2)))
        await eventually(lambda: len(executor.started) == 2)
        pending = asyncio.create_task(queue.submit(LLMTask("pending", "pending", 1)))
        await eventually_snapshot(queue, in_flight=2, pending=1)

        executor.release.set()
        first_outcome, second_outcome, pending_outcome = await asyncio.wait_for(
            asyncio.gather(first, second, pending),
            timeout=1.0,
        )

        for outcome in (first_outcome, second_outcome):
            assert outcome.status is TaskOutcomeStatus.TERMINAL_FAILURE
            assert len(outcome.telemetry.attempts) == 1
            attempt = outcome.telemetry.attempts[0]
            assert attempt.status is AttemptStatus.TIMED_OUT
            assert attempt.will_retry is False
            assert attempt.error_type == "TransportAbortedTimeoutError"
        assert len(executor.cancelled_during_abort) == 1
        assert pending_outcome.status is TaskOutcomeStatus.CANCELLED
        assert (
            pending_outcome.cancellation_reason
            is CancellationReason.EXECUTOR_RETIRED
        )
        assert await queue.snapshot() == queue_snapshot(2, 1, 0, 0, True)
        current = asyncio.current_task()
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not current
            and not task.done()
            and task.get_name().startswith("agent-evolve-llm-")
        ]

    run(scenario())


def test_recovery_topology_three_active_terminal_then_two_transport_aborts():
    async def scenario() -> None:
        clock = FakeClock()
        executor = RecoveryTopologyExecutor()
        queue = make_queue(
            executor=executor,
            clock=clock,
            max_in_flight=3,
            max_pending=1,
        )
        abort_a = asyncio.create_task(
            queue.submit(LLMTask("abort-a", "abort-a", 2))
        )
        independent = asyncio.create_task(
            queue.submit(LLMTask("independent", "independent", 2))
        )
        abort_b = asyncio.create_task(
            queue.submit(LLMTask("abort-b", "abort-b", 2))
        )
        await eventually(lambda: len(executor.started) == 3)
        pending = asyncio.create_task(
            queue.submit(LLMTask("pending", "pending", 1))
        )
        await eventually_snapshot(queue, in_flight=3, pending=1)

        executor.release_terminal.set()
        independent_outcome = await asyncio.wait_for(independent, timeout=1.0)
        assert independent_outcome.status is TaskOutcomeStatus.TERMINAL_FAILURE
        assert len(independent_outcome.telemetry.attempts) == 1
        await eventually(lambda: "pending" in executor.started)
        executor.release_aborts.set()
        abort_a_outcome, abort_b_outcome, pending_outcome = await asyncio.wait_for(
            asyncio.gather(abort_a, abort_b, pending),
            timeout=1.0,
        )

        for outcome in (abort_a_outcome, abort_b_outcome):
            assert outcome.status is TaskOutcomeStatus.TERMINAL_FAILURE
            assert len(outcome.telemetry.attempts) == 1
            assert (
                outcome.telemetry.attempts[0].error_type
                == "TransportAbortedTimeoutError"
            )
        assert pending_outcome.status is TaskOutcomeStatus.CANCELLED
        assert (
            pending_outcome.cancellation_reason
            is CancellationReason.EXECUTOR_RETIRED
        )
        assert len({
            independent_outcome.telemetry.task_id,
            abort_a_outcome.telemetry.task_id,
            abort_b_outcome.telemetry.task_id,
            pending_outcome.telemetry.task_id,
        }) == 4
        await asyncio.wait_for(queue.aclose(), timeout=1.0)
        await asyncio.wait_for(queue.aclose(), timeout=1.0)
        assert await queue.snapshot() == queue_snapshot(3, 1, 0, 0, True)
        current = asyncio.current_task()
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not current
            and not task.done()
            and task.get_name().startswith("agent-evolve-llm-")
        ]

    run(scenario())


def test_duplicate_live_task_ids_are_rejected_but_can_be_reused_after_completion():
    async def scenario() -> None:
        clock = FakeClock()
        executor = GateExecutor(clock)
        queue = make_queue(executor=executor, clock=clock)
        first = asyncio.create_task(queue.submit(LLMTask("same", "first", 1)))
        await eventually(lambda: executor.started == ["first"])
        with pytest.raises(DuplicateLLMTaskError):
            await queue.submit(LLMTask("same", "duplicate", 1))
        executor.release("first")
        assert (await first).status is TaskOutcomeStatus.SUCCEEDED

        executor.release("second")
        second = await queue.submit(LLMTask("same", "second", 1))
        assert second.status is TaskOutcomeStatus.SUCCEEDED
        await queue.aclose()

    run(scenario())


def test_per_task_timeout_override_is_passed_to_executor_and_runtime():
    async def scenario() -> None:
        clock = FakeClock()
        runtime = DeterministicRuntime(clock)
        executor = ScriptedExecutor(clock, ["ok"])
        queue = make_queue(
            executor=executor,
            clock=clock,
            runtime=runtime,
            timeout_ns=100,
        )
        outcome = await queue.submit(
            LLMTask("override", "request", 1, attempt_timeout_ns=777)
        )
        assert outcome.status is TaskOutcomeStatus.SUCCEEDED
        assert runtime.wait_calls == [777]
        assert executor.contexts[0].attempt_timeout_ns == 777
        await queue.aclose()

    run(scenario())


def test_none_attempt_timeout_delegates_liveness_without_fixed_total_cutoff():
    async def scenario() -> None:
        clock = FakeClock()
        runtime = DeterministicRuntime(clock)
        executor = ScriptedExecutor(clock, ["ok"])
        queue = make_queue(
            executor=executor,
            clock=clock,
            runtime=runtime,
            timeout_ns=None,
        )

        outcome = await queue.submit(LLMTask("progress_owned", "request", 1))

        assert outcome.status is TaskOutcomeStatus.SUCCEEDED
        assert runtime.wait_calls == [None]
        assert executor.contexts[0].attempt_timeout_ns is None
        await queue.aclose()

    run(scenario())
