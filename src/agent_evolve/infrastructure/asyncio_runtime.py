"""Asyncio timing adapter for the provider-neutral LLM task queue."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import TypeVar

from agent_evolve.domain.llm_task_queue import NANOSECONDS_PER_SECOND
from agent_evolve.ports.llm_task_queue import ExecutorRetiredError


AwaitedT = TypeVar("AwaitedT")


class TransportAbortedTimeoutError(TimeoutError, ExecutorRetiredError):
    """The deadline expired and the owned transport was retired cleanly.

    Unlike a plain ``TimeoutError``, this condition cannot be retried on the
    same provider executor: its transport has deliberately been closed to
    guarantee that the timed-out request is no longer live.
    """


class AsyncioRuntime:
    """Own real sleeps and timeout containment outside provider adapters.

    ``asyncio.wait_for`` is not a hard wall-clock bound: after its deadline it
    cancels the child and waits indefinitely for cancellation cleanup.  A
    provider stack may legitimately shield transport cleanup, which leaves the
    queue unable to publish a timeout.  When an owned ``timeout_abort`` is
    supplied, this runtime instead observes the deadline, cancels the attempt,
    closes its transport, and drains the attempt before returning a terminal
    :class:`TransportAbortedTimeoutError`.

    The abort callback must be idempotent and must retire every external
    operation reachable from the awaitable.  Production composition binds it
    to the owned provider client's asynchronous close operation.
    """

    def __init__(
        self,
        *,
        timeout_abort: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if timeout_abort is not None and not callable(timeout_abort):
            raise TypeError("timeout_abort must be callable or None")
        self._timeout_abort = timeout_abort

    async def sleep(self, delay_ns: int) -> None:
        if type(delay_ns) is not int or delay_ns < 0:
            raise ValueError("delay_ns must be a non-negative integer")
        await asyncio.sleep(delay_ns / NANOSECONDS_PER_SECOND)

    async def wait_for(
        self,
        awaitable: Awaitable[AwaitedT],
        timeout_ns: int | None,
    ) -> AwaitedT:
        if timeout_ns is not None and (
            type(timeout_ns) is not int or timeout_ns <= 0
        ):
            raise ValueError("timeout_ns must be a positive integer or None")
        if self._timeout_abort is None:
            if timeout_ns is None:
                return await awaitable
            return await asyncio.wait_for(
                awaitable,
                timeout=timeout_ns / NANOSECONDS_PER_SECOND,
            )

        attempt = asyncio.ensure_future(awaitable)
        try:
            if timeout_ns is None:
                return await attempt
            done, _ = await asyncio.wait(
                (attempt,),
                timeout=timeout_ns / NANOSECONDS_PER_SECOND,
                return_when=asyncio.ALL_COMPLETED,
            )
        except asyncio.CancelledError:
            await self._abort_and_drain(attempt)
            raise

        if done:
            return attempt.result()

        await self._abort_and_drain(attempt)
        raise TransportAbortedTimeoutError from None

    async def _abort_and_drain(self, attempt: "asyncio.Future[object]") -> None:
        """Retire external I/O and leave no child attempt behind."""

        attempt.cancel()
        abort_error: BaseException | None = None
        try:
            assert self._timeout_abort is not None
            await self._timeout_abort()
        except BaseException as error:  # retain cleanup, never provider text
            abort_error = error

        with suppress(BaseException):
            await attempt

        if not attempt.done():  # defensive: awaited futures must be terminal
            raise RuntimeError("timed-out attempt did not reach a terminal state")
        if abort_error is not None:
            raise RuntimeError("timed-out attempt transport abort failed") from None
