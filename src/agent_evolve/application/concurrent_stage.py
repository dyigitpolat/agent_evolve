"""Failure-contained execution for one concurrent application stage.

``asyncio.gather`` propagates the first child failure without cancelling the
remaining children.  That behavior is unsafe at an application-stage
boundary: a failed model arm can leave sibling provider calls running after
the campaign has started finalization.  This module owns the generic policy
for fail-fast sibling cancellation and bounded terminal-state retrieval.

The helper is deliberately workload- and transport-neutral.  Child
coroutines remain responsible for retiring resources they own when cancelled;
the barrier guarantees that cancellation is delivered and that every child
which reaches a terminal state is observed before the stage exception exits.
Cancellation-resistant children are observed by a terminal callback after the
bounded drain interval, so their later exception can never become an
unretrieved task exception.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable
from contextlib import suppress
from typing import TypeVar


StageResultT = TypeVar("StageResultT")

NANOSECONDS_PER_SECOND = 1_000_000_000
DEFAULT_CONCURRENT_STAGE_CANCEL_DRAIN_TIMEOUT_NS = 5 * NANOSECONDS_PER_SECOND


def _consume_terminal(task: "asyncio.Future[object]") -> None:
    """Retrieve a terminal child's exception without retaining its content."""

    if not task.done():  # pragma: no cover - callback/validated caller contract.
        return
    with suppress(BaseException):
        task.exception()


async def _cancel_and_drain(
    tasks: tuple["asyncio.Future[object]", ...],
    *,
    timeout_ns: int,
) -> None:
    """Cancel live children and observe them for at most ``timeout_ns``."""

    for task in tasks:
        if not task.done():
            task.cancel()
    if not tasks:
        return
    done, pending = await asyncio.wait(
        tasks,
        timeout=timeout_ns / NANOSECONDS_PER_SECOND,
        return_when=asyncio.ALL_COMPLETED,
    )
    for task in done:
        _consume_terminal(task)
    for task in pending:
        # A transport stack can resist cancellation while it retires a socket.
        # Do not let that defeat the campaign cleanup bound, but retain explicit
        # ownership of its eventual terminal state.
        task.add_done_callback(_consume_terminal)


async def gather_concurrent_stage(
    awaitables: Iterable[Awaitable[StageResultT]],
    *,
    cancel_drain_timeout_ns: int = (
        DEFAULT_CONCURRENT_STAGE_CANCEL_DRAIN_TIMEOUT_NS
    ),
) -> tuple[StageResultT, ...]:
    """Return ordered results or fail after cancelling and draining siblings.

    Success preserves input ordering.  On the first observed child failure,
    every still-live sibling is cancelled and terminal states are retrieved
    before that failure is re-raised.  Drain time is bounded because provider
    cancellation must never hold campaign finalization hostage indefinitely.

    If the caller itself is cancelled, the same cancel-and-drain law is applied
    before ``CancelledError`` propagates.
    """

    if (
        type(cancel_drain_timeout_ns) is not int
        or cancel_drain_timeout_ns <= 0
    ):
        raise ValueError("cancel_drain_timeout_ns must be a positive exact integer")
    tasks = tuple(asyncio.ensure_future(awaitable) for awaitable in awaitables)
    if not tasks:
        return ()

    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            failed = tuple(
                task
                for task in tasks
                if task in done
                and (
                    task.cancelled()
                    or task.exception() is not None
                )
            )
            if not failed:
                continue

            primary = failed[0]
            primary_error: BaseException
            if primary.cancelled():
                primary_error = asyncio.CancelledError()
            else:
                observed = primary.exception()
                assert observed is not None
                primary_error = observed

            # Retrieve every failure that completed in the same event-loop
            # turn, not only the deterministic first one.
            for task in done:
                _consume_terminal(task)
            await _cancel_and_drain(
                tuple(task for task in tasks if task not in done),
                timeout_ns=cancel_drain_timeout_ns,
            )
            raise primary_error
    except asyncio.CancelledError:
        await _cancel_and_drain(
            tasks,
            timeout_ns=cancel_drain_timeout_ns,
        )
        raise

    return tuple(task.result() for task in tasks)


__all__ = [
    "DEFAULT_CONCURRENT_STAGE_CANCEL_DRAIN_TIMEOUT_NS",
    "gather_concurrent_stage",
]
