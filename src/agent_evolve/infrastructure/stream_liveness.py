"""Progress-aware, content-blind supervision for one provider stream.

The supervisor knows nothing about prompts, model output, schemas, or benchmark
semantics.  A provider adapter reports only closed progress kinds/channels.  A
first-event deadline detects requests that never begin, an idle deadline moves
forward whenever progress is observed, and an optional absolute deadline is a
separate operational fail-safe.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
import time
from typing import Any, Protocol, TypeVar, runtime_checkable

from agent_evolve.ports.structured_generator import (
    GenerationFailureKind,
    StructuredGenerationError,
    StructuredStreamChannel,
    StructuredStreamCleanupTimeoutError,
    StructuredStreamLivenessPolicy,
    StructuredStreamProgress,
    StructuredStreamProgressKind,
    StructuredStreamProgressSink,
    StructuredStreamTimeoutError,
    StructuredStreamTimeoutPhase,
    StructuredStreamRetiredError,
)


ResultT = TypeVar("ResultT")


@runtime_checkable
class StreamProgressMarker(Protocol):
    """Accept only a closed content identity, never a plaintext fragment.

    Every successful operation must finish by marking exactly one
    ``STREAM_COMPLETED`` event.  It must be the last mark and is emitted by the
    adapter only after its framework call has returned a typed result.
    """

    def __call__(
        self,
        kind: StructuredStreamProgressKind,
        channel: StructuredStreamChannel,
        *,
        event_content_utf8_bytes: int,
        cumulative_content_utf8_bytes: int,
        rolling_content_sha256: str,
    ) -> None: ...


StreamOperation = Callable[[StreamProgressMarker], Awaitable[ResultT]]
StreamRetirementOperation = Callable[[], Awaitable[None]]


@runtime_checkable
class ContentBlindStreamSupervisor(Protocol):
    """Injectable supervision port consumed by provider adapters."""

    async def run(
        self,
        operation: StreamOperation[Any],
        *,
        call_id: str,
        provider_attempt_id: str | None = None,
        policy: StructuredStreamLivenessPolicy,
        progress_sink: StructuredStreamProgressSink | None = None,
    ) -> Any: ...


class AsyncioContentBlindStreamSupervisor:
    """Supervise one streamed operation without observing its content."""

    def __init__(
        self,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        retirement_operation: StreamRetirementOperation | None = None,
    ) -> None:
        if not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable")
        if retirement_operation is not None and not callable(retirement_operation):
            raise TypeError("retirement_operation must be callable or None")
        self._monotonic_ns = monotonic_ns
        self._retirement_operation = retirement_operation
        self._retired = False

    async def run(
        self,
        operation: StreamOperation[ResultT],
        *,
        call_id: str,
        provider_attempt_id: str | None = None,
        policy: StructuredStreamLivenessPolicy,
        progress_sink: StructuredStreamProgressSink | None = None,
    ) -> ResultT:
        if not callable(operation):
            raise TypeError("operation must be callable")
        if self._retired:
            raise StructuredStreamRetiredError()
        if type(policy) is not StructuredStreamLivenessPolicy:
            raise TypeError("policy must be a StructuredStreamLivenessPolicy")
        policy.__post_init__()
        if progress_sink is not None and not callable(progress_sink):
            raise TypeError("progress_sink must be callable or None")

        started_ns = self._monotonic_ns()
        if type(started_ns) is not int or started_ns < 0:
            raise ValueError("monotonic_ns returned an invalid value")
        activity = asyncio.Event()
        sequence = 0
        first_event_ns: int | None = None
        last_event_ns: int | None = None
        last_cumulative_content_utf8_bytes = 0
        last_kind: StructuredStreamProgressKind | None = None
        stream_completed_count = 0
        progress_accepting = True

        def mark_progress(
            kind: StructuredStreamProgressKind,
            channel: StructuredStreamChannel,
            *,
            event_content_utf8_bytes: int,
            cumulative_content_utf8_bytes: int,
            rolling_content_sha256: str,
        ) -> None:
            nonlocal sequence, first_event_ns, last_event_ns, last_kind
            nonlocal last_cumulative_content_utf8_bytes
            nonlocal stream_completed_count
            if not progress_accepting:
                raise StructuredGenerationError(
                    kind=GenerationFailureKind.CANCELLED,
                    retryable=False,
                    safe_message="stream progress occurred after liveness expiry",
                )
            if type(kind) is not StructuredStreamProgressKind:
                raise TypeError("stream progress kind escaped the closed domain")
            if type(channel) is not StructuredStreamChannel:
                raise TypeError("stream channel escaped the closed domain")
            if stream_completed_count:
                raise StructuredGenerationError(
                    kind=GenerationFailureKind.UNKNOWN,
                    retryable=False,
                    safe_message=(
                        "stream progress occurred after local stream completion"
                    ),
                )
            if (
                type(event_content_utf8_bytes) is not int
                or event_content_utf8_bytes < 0
            ):
                raise ValueError("stream event byte count escaped the closed domain")
            if (
                type(cumulative_content_utf8_bytes) is not int
                or cumulative_content_utf8_bytes
                != last_cumulative_content_utf8_bytes + event_content_utf8_bytes
            ):
                raise ValueError("stream cumulative byte count is not monotonic")
            observed_ns = self._monotonic_ns()
            if type(observed_ns) is not int or observed_ns < started_ns:
                raise ValueError("monotonic stream observation moved backwards")
            sequence += 1
            if first_event_ns is None:
                first_event_ns = observed_ns
            last_event_ns = observed_ns
            progress = StructuredStreamProgress(
                call_id=call_id,
                sequence=sequence,
                kind=kind,
                channel=channel,
                elapsed_ns=observed_ns - started_ns,
                event_content_utf8_bytes=event_content_utf8_bytes,
                cumulative_content_utf8_bytes=cumulative_content_utf8_bytes,
                rolling_content_sha256=rolling_content_sha256,
                provider_attempt_id=provider_attempt_id,
            )
            if progress_sink is not None:
                # Publication is deliberately fail closed: a scientific
                # response cannot escape when required liveness evidence was
                # not accepted by its injected sink.
                progress_sink(progress)
            last_cumulative_content_utf8_bytes = cumulative_content_utf8_bytes
            last_kind = kind
            if kind is StructuredStreamProgressKind.STREAM_COMPLETED:
                stream_completed_count += 1
            activity.set()

        task = asyncio.create_task(operation(mark_progress))

        async def expire_liveness(
            phase: StructuredStreamTimeoutPhase,
        ) -> None:
            nonlocal progress_accepting
            progress_accepting = False
            settled = await _cancel_and_drain(
                task,
                timeout_ns=policy.cleanup_policy.cancel_drain_timeout_ns,
            )
            if settled:
                raise StructuredStreamTimeoutError(phase)
            # asyncio cannot kill a cancellation-resistant coroutine. Retire
            # this supervisor before returning so neither a retry nor a later
            # logical call can share the abandoned attempt's transport.
            self._retired = True
            await _retire_bounded(
                self._retirement_operation,
                timeout_ns=(
                    policy.cleanup_policy.transport_retire_timeout_ns
                ),
            )
            raise StructuredStreamCleanupTimeoutError(phase)

        try:
            while True:
                if task.done():
                    result = task.result()
                    if (
                        stream_completed_count != 1
                        or last_kind is not StructuredStreamProgressKind.STREAM_COMPLETED
                    ):
                        raise StructuredGenerationError(
                            kind=GenerationFailureKind.UNKNOWN,
                            retryable=False,
                            safe_message=(
                                "streaming operation returned without one terminal "
                                "local completion event"
                            ),
                        )
                    return result

                now_ns = self._monotonic_ns()
                deadline_ns, phase = _next_deadline(
                    started_ns=started_ns,
                    first_event_ns=first_event_ns,
                    last_event_ns=last_event_ns,
                    policy=policy,
                )
                remaining_ns = deadline_ns - now_ns
                if remaining_ns <= 0:
                    await expire_liveness(phase)

                observed_sequence = sequence
                activity.clear()
                if sequence != observed_sequence:
                    continue
                activity_waiter = asyncio.create_task(activity.wait())
                try:
                    done, _ = await asyncio.wait(
                        (task, activity_waiter),
                        timeout=remaining_ns / 1_000_000_000,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    if not activity_waiter.done():
                        activity_waiter.cancel()
                        with suppress(BaseException):
                            await activity_waiter
                if task in done:
                    continue
                if activity_waiter in done:
                    continue
                await expire_liveness(phase)
        except asyncio.CancelledError:
            progress_accepting = False
            settled = await _cancel_and_drain(
                task,
                timeout_ns=policy.cleanup_policy.cancel_drain_timeout_ns,
            )
            if not settled:
                self._retired = True
                await _retire_bounded(
                    self._retirement_operation,
                    timeout_ns=(
                        policy.cleanup_policy.transport_retire_timeout_ns
                    ),
                )
            raise


def _next_deadline(
    *,
    started_ns: int,
    first_event_ns: int | None,
    last_event_ns: int | None,
    policy: StructuredStreamLivenessPolicy,
) -> tuple[int, StructuredStreamTimeoutPhase]:
    """Return the next boundary; absolute wins deterministic deadline ties."""

    if first_event_ns is None:
        candidates = [
            (
                started_ns + policy.first_event_timeout_ns,
                1,
                StructuredStreamTimeoutPhase.FIRST_EVENT,
            )
        ]
    else:
        assert last_event_ns is not None
        candidates = [
            (
                last_event_ns + policy.idle_timeout_ns,
                1,
                StructuredStreamTimeoutPhase.IDLE,
            )
        ]
    if policy.absolute_timeout_ns is not None:
        candidates.append(
            (
                started_ns + policy.absolute_timeout_ns,
                0,
                StructuredStreamTimeoutPhase.ABSOLUTE,
            )
        )
    deadline_ns, _, phase = min(candidates)
    return deadline_ns, phase


def _consume_terminal(task: "asyncio.Task[object]") -> None:
    """Retrieve a detached task's terminal state without retaining content."""

    with suppress(BaseException):
        task.exception()


async def _cancel_and_drain(
    task: "asyncio.Task[object]",
    *,
    timeout_ns: int,
) -> bool:
    """Cancel one stream and wait at most the frozen cleanup interval."""

    if type(timeout_ns) is not int or timeout_ns <= 0:
        raise ValueError("timeout_ns must be a positive exact integer")
    if not task.done():
        task.cancel()
    done, _ = await asyncio.wait(
        (task,),
        timeout=timeout_ns / 1_000_000_000,
        return_when=asyncio.ALL_COMPLETED,
    )
    if task in done:
        _consume_terminal(task)
        return True
    task.add_done_callback(_consume_terminal)
    return False


async def _retire_bounded(
    retirement_operation: StreamRetirementOperation | None,
    *,
    timeout_ns: int,
) -> bool:
    """Invoke transport retirement without trusting close cancellation."""

    if type(timeout_ns) is not int or timeout_ns <= 0:
        raise ValueError("timeout_ns must be a positive exact integer")
    if retirement_operation is None:
        return True
    task = asyncio.create_task(retirement_operation())
    done, _ = await asyncio.wait(
        (task,),
        timeout=timeout_ns / 1_000_000_000,
        return_when=asyncio.ALL_COMPLETED,
    )
    if task in done:
        _consume_terminal(task)
        return True
    task.cancel()
    task.add_done_callback(_consume_terminal)
    return False


__all__ = [
    "AsyncioContentBlindStreamSupervisor",
    "ContentBlindStreamSupervisor",
    "StreamOperation",
    "StreamProgressMarker",
    "StreamRetirementOperation",
]
