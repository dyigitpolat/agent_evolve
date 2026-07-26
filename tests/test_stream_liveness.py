"""Provider-free tests for content-blind progress-aware stream supervision."""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from agent_evolve.infrastructure.stream_liveness import (
    AsyncioContentBlindStreamSupervisor,
    _next_deadline,
)
from agent_evolve.ports.structured_generator import (
    StructuredGenerationError,
    StructuredStreamChannel,
    StructuredStreamCleanupPolicy,
    StructuredStreamCleanupTimeoutError,
    StructuredStreamLivenessPolicy,
    StructuredStreamProgressKind,
    StructuredStreamTimeoutError,
    StructuredStreamTimeoutPhase,
    StructuredStreamRetiredError,
)


_EMPTY_ROLLING_SHA256 = hashlib.sha256(
    b"agent-evolve:test-stream-content:v1\x00"
).hexdigest()


def test_liveness_policy_is_closed_and_has_no_default_absolute_cutoff() -> None:
    policy = StructuredStreamLivenessPolicy(
        first_event_timeout_ns=10,
        idle_timeout_ns=20,
    )

    assert policy.absolute_timeout_ns is None
    assert policy.cleanup_policy.cancel_drain_timeout_ns == 5_000_000_000
    assert policy.cleanup_policy.transport_retire_timeout_ns == 5_000_000_000
    assert len(policy.cleanup_policy.configuration_sha256) == 64
    with pytest.raises(ValueError):
        StructuredStreamLivenessPolicy(0, 20)
    with pytest.raises(ValueError):
        StructuredStreamLivenessPolicy(10, 0)
    with pytest.raises(ValueError):
        StructuredStreamLivenessPolicy(10, 20, 0)
    with pytest.raises(ValueError):
        StructuredStreamCleanupPolicy(cancel_drain_timeout_ns=0)


def test_deadline_moves_with_progress_and_absolute_is_independent() -> None:
    progressing = StructuredStreamLivenessPolicy(
        first_event_timeout_ns=100,
        idle_timeout_ns=40,
    )
    assert _next_deadline(
        started_ns=1_000,
        first_event_ns=None,
        last_event_ns=None,
        policy=progressing,
    ) == (1_100, StructuredStreamTimeoutPhase.FIRST_EVENT)
    assert _next_deadline(
        started_ns=1_000,
        first_event_ns=1_080,
        last_event_ns=1_250,
        policy=progressing,
    ) == (1_290, StructuredStreamTimeoutPhase.IDLE)

    bounded = StructuredStreamLivenessPolicy(
        first_event_timeout_ns=100,
        idle_timeout_ns=40,
        absolute_timeout_ns=200,
    )
    assert _next_deadline(
        started_ns=1_000,
        first_event_ns=1_080,
        last_event_ns=1_250,
        policy=bounded,
    ) == (1_200, StructuredStreamTimeoutPhase.ABSOLUTE)


def test_first_event_timeout_cancels_and_returns_no_partial_result() -> None:
    async def scenario() -> None:
        supervisor = AsyncioContentBlindStreamSupervisor()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def operation(_mark_progress):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with pytest.raises(StructuredStreamTimeoutError) as caught:
            await supervisor.run(
                operation,
                call_id="call_stream_test_000001",
                policy=StructuredStreamLivenessPolicy(
                    first_event_timeout_ns=10_000_000,
                    idle_timeout_ns=1_000_000_000,
                ),
            )
        assert started.is_set()
        assert cancelled.is_set()
        assert caught.value.phase is StructuredStreamTimeoutPhase.FIRST_EVENT

    asyncio.run(scenario())


def test_idle_timeout_is_reset_only_by_closed_progress_events() -> None:
    async def scenario() -> None:
        supervisor = AsyncioContentBlindStreamSupervisor()
        observed = []
        cancelled = asyncio.Event()

        async def operation(mark_progress):
            mark_progress(
                StructuredStreamProgressKind.PART_DELTA,
                StructuredStreamChannel.THINKING,
                event_content_utf8_bytes=7,
                cumulative_content_utf8_bytes=7,
                rolling_content_sha256=_EMPTY_ROLLING_SHA256,
            )
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        with pytest.raises(StructuredStreamTimeoutError) as caught:
            await supervisor.run(
                operation,
                call_id="call_stream_test_000002",
                policy=StructuredStreamLivenessPolicy(
                    first_event_timeout_ns=1_000_000_000,
                    idle_timeout_ns=10_000_000,
                ),
                progress_sink=observed.append,
            )
        assert cancelled.is_set()
        assert caught.value.phase is StructuredStreamTimeoutPhase.IDLE
        assert len(observed) == 1
        assert observed[0].sequence == 1
        assert observed[0].kind is StructuredStreamProgressKind.PART_DELTA
        assert observed[0].channel is StructuredStreamChannel.THINKING
        assert set(observed[0].__dataclass_fields__) == {
            "call_id",
            "sequence",
            "kind",
            "channel",
            "elapsed_ns",
            "event_content_utf8_bytes",
            "cumulative_content_utf8_bytes",
            "rolling_content_sha256",
            "provider_attempt_id",
        }
        assert observed[0].event_content_utf8_bytes == 7
        assert observed[0].cumulative_content_utf8_bytes == 7

    asyncio.run(scenario())


def test_progressing_operation_has_no_implicit_total_deadline() -> None:
    async def scenario() -> None:
        supervisor = AsyncioContentBlindStreamSupervisor()
        release = asyncio.Event()

        async def operation(mark_progress):
            mark_progress(
                StructuredStreamProgressKind.PART_STARTED,
                StructuredStreamChannel.TOOL_CALL,
                event_content_utf8_bytes=4,
                cumulative_content_utf8_bytes=4,
                rolling_content_sha256=_EMPTY_ROLLING_SHA256,
            )
            await release.wait()
            mark_progress(
                StructuredStreamProgressKind.STREAM_COMPLETED,
                StructuredStreamChannel.OTHER,
                event_content_utf8_bytes=0,
                cumulative_content_utf8_bytes=4,
                rolling_content_sha256=_EMPTY_ROLLING_SHA256,
            )
            return "complete"

        task = asyncio.create_task(
            supervisor.run(
                operation,
                call_id="call_stream_test_000003",
                policy=StructuredStreamLivenessPolicy(
                    first_event_timeout_ns=1_000_000_000,
                    idle_timeout_ns=1_000_000_000,
                    absolute_timeout_ns=None,
                ),
            )
        )
        await asyncio.sleep(0)
        release.set()
        assert await task == "complete"

    asyncio.run(scenario())


def test_output_selection_can_precede_tail_progress_and_local_completion() -> None:
    async def scenario() -> None:
        observed = []

        async def operation(mark_progress):
            mark_progress(
                StructuredStreamProgressKind.OUTPUT_SELECTED,
                StructuredStreamChannel.OTHER,
                event_content_utf8_bytes=0,
                cumulative_content_utf8_bytes=0,
                rolling_content_sha256=_EMPTY_ROLLING_SHA256,
            )
            mark_progress(
                StructuredStreamProgressKind.PART_DELTA,
                StructuredStreamChannel.TOOL_CALL,
                event_content_utf8_bytes=4,
                cumulative_content_utf8_bytes=4,
                rolling_content_sha256=_EMPTY_ROLLING_SHA256,
            )
            mark_progress(
                StructuredStreamProgressKind.PART_ENDED,
                StructuredStreamChannel.TOOL_CALL,
                event_content_utf8_bytes=0,
                cumulative_content_utf8_bytes=4,
                rolling_content_sha256=_EMPTY_ROLLING_SHA256,
            )
            mark_progress(
                StructuredStreamProgressKind.STREAM_COMPLETED,
                StructuredStreamChannel.OTHER,
                event_content_utf8_bytes=0,
                cumulative_content_utf8_bytes=4,
                rolling_content_sha256=_EMPTY_ROLLING_SHA256,
            )
            return "typed-result"

        result = await AsyncioContentBlindStreamSupervisor().run(
            operation,
            call_id="call_stream_test_tail_after_selection_000001",
            policy=StructuredStreamLivenessPolicy(
                first_event_timeout_ns=1_000_000_000,
                idle_timeout_ns=1_000_000_000,
            ),
            progress_sink=observed.append,
        )
        assert result == "typed-result"
        assert [row.kind for row in observed] == [
            StructuredStreamProgressKind.OUTPUT_SELECTED,
            StructuredStreamProgressKind.PART_DELTA,
            StructuredStreamProgressKind.PART_ENDED,
            StructuredStreamProgressKind.STREAM_COMPLETED,
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize("violation", ["missing", "duplicate", "after"])
def test_success_requires_exactly_one_last_local_completion(violation: str) -> None:
    async def scenario() -> None:
        async def operation(mark_progress):
            mark_progress(
                StructuredStreamProgressKind.PART_STARTED,
                StructuredStreamChannel.TOOL_CALL,
                event_content_utf8_bytes=1,
                cumulative_content_utf8_bytes=1,
                rolling_content_sha256=_EMPTY_ROLLING_SHA256,
            )
            if violation == "missing":
                return "invalid"
            mark_progress(
                StructuredStreamProgressKind.STREAM_COMPLETED,
                StructuredStreamChannel.OTHER,
                event_content_utf8_bytes=0,
                cumulative_content_utf8_bytes=1,
                rolling_content_sha256=_EMPTY_ROLLING_SHA256,
            )
            if violation == "duplicate":
                mark_progress(
                    StructuredStreamProgressKind.STREAM_COMPLETED,
                    StructuredStreamChannel.OTHER,
                    event_content_utf8_bytes=0,
                    cumulative_content_utf8_bytes=1,
                    rolling_content_sha256=_EMPTY_ROLLING_SHA256,
                )
            else:
                mark_progress(
                    StructuredStreamProgressKind.PART_ENDED,
                    StructuredStreamChannel.TOOL_CALL,
                    event_content_utf8_bytes=0,
                    cumulative_content_utf8_bytes=1,
                    rolling_content_sha256=_EMPTY_ROLLING_SHA256,
                )
            return "invalid"

        with pytest.raises(StructuredGenerationError):
            await AsyncioContentBlindStreamSupervisor().run(
                operation,
                call_id=f"call_stream_test_completion_{violation}_000001",
                policy=StructuredStreamLivenessPolicy(
                    first_event_timeout_ns=1_000_000_000,
                    idle_timeout_ns=1_000_000_000,
                ),
            )

    asyncio.run(scenario())


def test_cancellation_resistant_stream_is_bounded_terminal_and_retires() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        cancellation_seen = asyncio.Event()
        operation_settled = asyncio.Event()
        transport_retired = asyncio.Event()

        async def operation(_mark_progress):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await release.wait()
                return "late-result"
            finally:
                operation_settled.set()

        async def retire_transport() -> None:
            transport_retired.set()

        supervisor = AsyncioContentBlindStreamSupervisor(
            retirement_operation=retire_transport,
        )
        policy = StructuredStreamLivenessPolicy(
            first_event_timeout_ns=5_000_000,
            idle_timeout_ns=1_000_000_000,
            cleanup_policy=StructuredStreamCleanupPolicy(
                cancel_drain_timeout_ns=10_000_000,
                transport_retire_timeout_ns=10_000_000,
            ),
        )
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(StructuredStreamCleanupTimeoutError) as caught:
            await supervisor.run(
                operation,
                call_id="call_stream_cleanup_resistant_000001",
                policy=policy,
            )
        elapsed = loop.time() - started

        assert caught.value.phase is StructuredStreamTimeoutPhase.FIRST_EVENT
        assert caught.value.retryable is False
        assert cancellation_seen.is_set()
        assert transport_retired.is_set()
        assert not operation_settled.is_set()
        assert elapsed < 0.2

        invoked = False

        async def forbidden_operation(_mark_progress):
            nonlocal invoked
            invoked = True
            return "forbidden"

        with pytest.raises(StructuredStreamRetiredError):
            await supervisor.run(
                forbidden_operation,
                call_id="call_stream_cleanup_resistant_000002",
                policy=policy,
            )
        assert invoked is False

        release.set()
        await asyncio.wait_for(operation_settled.wait(), timeout=1.0)

    asyncio.run(scenario())


def test_cancellation_resistant_transport_retirement_is_also_bounded() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        operation_cancelled = asyncio.Event()
        operation_settled = asyncio.Event()
        retirement_started = asyncio.Event()
        retirement_cancelled = asyncio.Event()
        retirement_settled = asyncio.Event()

        async def operation(_mark_progress):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                operation_cancelled.set()
                await release.wait()
            finally:
                operation_settled.set()

        async def resistant_retirement() -> None:
            retirement_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                retirement_cancelled.set()
                await release.wait()
            finally:
                retirement_settled.set()

        policy = StructuredStreamLivenessPolicy(
            first_event_timeout_ns=5_000_000,
            idle_timeout_ns=1_000_000_000,
            cleanup_policy=StructuredStreamCleanupPolicy(
                cancel_drain_timeout_ns=5_000_000,
                transport_retire_timeout_ns=5_000_000,
            ),
        )
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(StructuredStreamCleanupTimeoutError):
            await AsyncioContentBlindStreamSupervisor(
                retirement_operation=resistant_retirement,
            ).run(
                operation,
                call_id="call_stream_retirement_resistant_000001",
                policy=policy,
            )
        elapsed = loop.time() - started

        assert operation_cancelled.is_set()
        assert retirement_started.is_set()
        await asyncio.sleep(0)
        assert retirement_cancelled.is_set()
        assert elapsed < 0.2
        release.set()
        await asyncio.wait_for(operation_settled.wait(), timeout=1.0)
        await asyncio.wait_for(retirement_settled.wait(), timeout=1.0)

    asyncio.run(scenario())
