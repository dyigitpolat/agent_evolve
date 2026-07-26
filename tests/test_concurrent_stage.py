"""Failure-containment proofs for generic concurrent application stages."""

from __future__ import annotations

import asyncio
import time

import pytest

from agent_evolve.application.concurrent_stage import gather_concurrent_stage


def test_success_preserves_input_order_and_waits_for_every_child() -> None:
    async def scenario() -> tuple[str, ...]:
        release_first = asyncio.Event()

        async def first() -> str:
            await release_first.wait()
            return "first"

        async def second() -> str:
            release_first.set()
            return "second"

        return await gather_concurrent_stage((first(), second()))

    assert asyncio.run(scenario()) == ("first", "second")


def test_first_failure_cancels_and_drains_sibling_before_propagation() -> None:
    async def scenario() -> None:
        sibling_started = asyncio.Event()
        sibling_terminal = asyncio.Event()

        async def failed() -> None:
            await sibling_started.wait()
            raise LookupError("primary")

        async def sibling() -> None:
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                sibling_terminal.set()
                raise RuntimeError("failure during cancellation")

        with pytest.raises(LookupError, match="primary"):
            await gather_concurrent_stage((failed(), sibling()))
        assert sibling_terminal.is_set()

    asyncio.run(scenario())


def test_cancellation_resistant_sibling_is_bounded_and_later_observed() -> None:
    async def scenario() -> tuple[float, list[dict[str, object]]]:
        sibling_started = asyncio.Event()
        release_retirement = asyncio.Event()
        sibling_terminal = asyncio.Event()
        loop = asyncio.get_running_loop()
        exception_contexts: list[dict[str, object]] = []
        prior_handler = loop.get_exception_handler()
        loop.set_exception_handler(
            lambda _loop, context: exception_contexts.append(dict(context))
        )

        async def failed() -> None:
            await sibling_started.wait()
            raise LookupError("primary")

        async def cancellation_resistant_sibling() -> None:
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Model a provider runner completing non-cancellable transport
                # retirement after the campaign's bounded sibling drain.
                await release_retirement.wait()
                sibling_terminal.set()
                raise RuntimeError("late retirement failure")

        started = time.monotonic()
        try:
            with pytest.raises(LookupError, match="primary"):
                await gather_concurrent_stage(
                    (failed(), cancellation_resistant_sibling()),
                    cancel_drain_timeout_ns=10_000_000,
                )
            elapsed = time.monotonic() - started
            release_retirement.set()
            await asyncio.wait_for(sibling_terminal.wait(), timeout=1.0)
            # Allow the barrier-installed done callback to retrieve the late
            # exception before inspecting the loop exception channel.
            await asyncio.sleep(0)
            return elapsed, exception_contexts
        finally:
            loop.set_exception_handler(prior_handler)

    elapsed, contexts = asyncio.run(scenario())
    assert elapsed < 0.25
    assert not any(
        context.get("message") == "Task exception was never retrieved"
        for context in contexts
    )


def test_caller_cancellation_cancels_and_drains_all_children() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        terminals = 0

        async def child() -> None:
            nonlocal terminals
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                terminals += 1

        stage = asyncio.create_task(gather_concurrent_stage((child(), child())))
        await started.wait()
        stage.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stage
        assert terminals == 2

    asyncio.run(scenario())
