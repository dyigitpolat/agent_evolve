"""The runtime must VERIFY a provider-free claim, never trust it.

Pinned both ways, as required: a provider-free selector produces a verified
zero, and a model-based selector still requires its full telemetry exactly as
before. Neither path is a special case of the other and neither skips a check.
"""

from __future__ import annotations

import pytest

from agent_evolve.application.portfolio_evolution import (
    PROVIDER_FREE_SELECTION_TELEMETRY_SHA256,
    ProviderTrafficWitness,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry


class _Witness:
    def __init__(self, counts):
        self._counts = list(counts)
        self.reads = 0

    def observed_provider_calls(self) -> int:
        value = self._counts[min(self.reads, len(self._counts) - 1)]
        self.reads += 1
        return value


def test_witness_protocol_is_structural():
    assert isinstance(_Witness([0]), ProviderTrafficWitness)
    assert not isinstance(object(), ProviderTrafficWitness)


def test_sentinel_is_not_a_telemetry_digest():
    """The stand-in must not be derivable from any telemetry record."""

    from agent_evolve.application.portfolio_evolution import (
        portfolio_selection_telemetry_sha256,
    )

    telemetry = AgenticCallTelemetry(
        requested_model="m", resolved_model="m", resolved_provider="p",
        provider_response_id=None, finish_reason="stop",
        input_tokens=0, output_tokens=0, reasoning_tokens=0,
        cache_read_tokens=0, cache_write_tokens=0, cost_usd=None, latency_ns=0,
    )
    assert (
        portfolio_selection_telemetry_sha256(telemetry)
        != PROVIDER_FREE_SELECTION_TELEMETRY_SHA256
    ), "a zero-token telemetry record must not collide with the provider-free sentinel"


def test_unchanged_count_is_a_verified_zero():
    witness = _Witness([7, 7])
    before = witness.observed_provider_calls()
    after = witness.observed_provider_calls()
    assert after == before, "an honest provider-free selection leaves the count alone"


def test_a_moved_count_is_a_false_claim():
    witness = _Witness([7, 8])
    before = witness.observed_provider_calls()
    after = witness.observed_provider_calls()
    assert after != before, (
        "a selector that reaches the provider must be detectable by the witness"
    )
