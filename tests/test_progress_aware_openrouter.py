"""Offline composition tests for the reusable progress-aware OpenRouter stack."""

from __future__ import annotations

import pytest

from agent_evolve.integrations.pydantic_ai.async_generator import (
    OpenRouterReasoningConfig,
)
from agent_evolve.integrations.pydantic_ai import progress_aware_openrouter as live
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    BoundedPrestreamAndSchemaRepairRetryClassifier,
    ExactPayloadAttemptPolicy,
    ExactTransportSchemaRepairAttemptPolicy,
    FirstEventResilientBoundedSchemaRepairRetryClassifier,
    NonRepeatingStreamTransportRetryClassifier,
    OpaqueHTTP400AndBoundedSchemaRepairRetryClassifier,
    OpaqueHTTP400AndSchemaRepairOnceRetryClassifier,
    OpaqueHTTP400OnceRetryClassifier,
    OutcomePublicationPolicy,
    StructuredGenerationRetryClassifier,
    TransportOnlyStructuredGenerationRetryClassifier,
)
from agent_evolve.policies.llm_backoff import DeterministicHashJitter
from agent_evolve.ports.structured_generator import StructuredStreamLivenessPolicy


def _config() -> live.ProgressAwareOpenRouterConfig:
    return live.ProgressAwareOpenRouterConfig(
        model_name="deepseek/deepseek-v4-pro",
        provider_only=("streamlake",),
        connect_timeout_seconds=90.0,
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=120_000_000_000,
            idle_timeout_ns=120_000_000_000,
            absolute_timeout_ns=1_800_000_000_000,
        ),
        max_connections=4,
        max_pending=8,
        max_attempts=2,
        base_backoff_ns=1_000_000_000,
        max_backoff_ns=30_000_000_000,
        jitter_seed=2_026_071_500,
        jitter_domain="stream-conformance-v1",
        reasoning_config=OpenRouterReasoningConfig(max_tokens=4_096),
    )


def test_manifest_projection_freezes_separate_connect_and_stream_boundaries() -> None:
    record = _config().to_manifest_record()

    assert record["provider_options"] == {
        "only": ["streamlake"],
        "allow_fallbacks": False,
    }
    assert record["connect_timeout_seconds"] == 90.0
    assert record["stream_liveness"] == {
        "first_event_timeout_ns": 120_000_000_000,
        "idle_timeout_ns": 120_000_000_000,
        "absolute_timeout_ns": 1_800_000_000_000,
        "cleanup": _config().stream_liveness_policy.cleanup_policy.to_manifest_record(),
    }
    cleanup = record["stream_liveness"]["cleanup"]
    assert cleanup["policy_id"] == "bounded_cancel_drain"
    assert len(cleanup["definition_sha256"]) == 64
    assert len(cleanup["configuration_sha256"]) == 64
    queue = record["queue"]
    assert queue["attempt_timeout_ns"] is None
    assert queue["attempt_request_policy"] == "exact_payload"
    assert "schema_repair_policy" not in queue
    assert queue["retry_classifier"] == "structured_generation"
    assert queue["backoff"]["jitter_domain"] == "stream-conformance-v1"
    assert queue["backoff"]["rate_limit_backoff_floor_ns"] == 0
    assert record["reasoning"] == {"max_tokens": 4_096}
    assert record["supports_forced_tool_choice"] is True


def test_factory_wires_progress_queue_and_owned_lifecycle_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    generator = object()
    runner = object()

    def fake_openrouter(**kwargs):
        captured["openrouter"] = kwargs
        return generator

    def fake_runner(**kwargs):
        captured["runner"] = kwargs
        return runner

    monkeypatch.setattr(
        live.PydanticAIStructuredGenerator,
        "openrouter",
        fake_openrouter,
    )
    monkeypatch.setattr(live, "create_production_queued_runner", fake_runner)
    progress_rows = []
    outcome_rows = []
    outbound_rows = []
    progress_sink = progress_rows.append
    outcome_sink = outcome_rows.append
    outbound_sink = outbound_rows.append

    observed = live.create_progress_aware_openrouter_runner(
        api_key="offline-injected-key",
        config=_config(),
        progress_sink=progress_sink,
        outcome_sink=outcome_sink,
        outbound_request_manifest_sink=outbound_sink,
    )

    assert observed is runner
    openrouter = captured["openrouter"]
    assert openrouter["timeout_seconds"] == 90.0
    assert openrouter["provider_options"] == {
        "only": ["streamlake"],
        "allow_fallbacks": False,
    }
    assert openrouter["stream_liveness_policy"] == (_config().stream_liveness_policy)
    assert openrouter["stream_progress_sink"] is progress_sink
    assert openrouter["outbound_request_manifest_sink"] is outbound_sink
    assert openrouter["supports_forced_tool_choice"] is True

    queue = captured["runner"]
    assert queue["generator"] is generator
    assert queue["attempt_timeout_ns"] is None
    assert queue["close_generator"] is True
    assert queue["outcome_sink"] is outcome_sink
    assert queue["outcome_publication_policy"] is OutcomePublicationPolicy.REQUIRED
    assert queue["rate_limit_backoff_floor_ns"] == 0
    assert type(queue["attempt_request_policy"]) is ExactPayloadAttemptPolicy
    assert type(queue["retry_classifier"]) is StructuredGenerationRetryClassifier
    assert queue["jitter_policy"] == DeterministicHashJitter(
        seed=2_026_071_500,
        domain="stream-conformance-v1",
    )


def test_tool_forcing_capability_is_manifest_bound_and_factory_wired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        live.PydanticAIStructuredGenerator,
        "openrouter",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        live,
        "create_production_queued_runner",
        lambda **_kwargs: object(),
    )
    config = live.ProgressAwareOpenRouterConfig(
        **{
            field: getattr(_config(), field)
            for field in _config().__dataclass_fields__
            if field != "supports_forced_tool_choice"
        },
        supports_forced_tool_choice=False,
    )

    assert config.to_manifest_record()["supports_forced_tool_choice"] is False
    live.create_progress_aware_openrouter_runner(
        api_key="offline-injected-key",
        config=config,
        progress_sink=lambda _row: None,
        outcome_sink=lambda _row: None,
    )
    assert captured["supports_forced_tool_choice"] is False


def test_transport_only_retry_mode_is_manifest_bound_and_wired_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        live.PydanticAIStructuredGenerator,
        "openrouter",
        lambda **_kwargs: object(),
    )

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(live, "create_production_queued_runner", fake_runner)
    config = live.ProgressAwareOpenRouterConfig(
        **{
            field: getattr(_config(), field)
            for field in _config().__dataclass_fields__
            if field != "retry_mode"
        },
        retry_mode=live.ProgressAwareRetryMode.TRANSPORT_ONLY,
    )

    assert config.to_manifest_record()["queue"]["retry_classifier"] == (
        "transport_only"
    )
    live.create_progress_aware_openrouter_runner(
        api_key="offline-injected-key",
        config=config,
        progress_sink=lambda _row: None,
        outcome_sink=lambda _row: None,
    )
    assert type(captured["retry_classifier"]) is (
        TransportOnlyStructuredGenerationRetryClassifier
    )


def test_non_repeating_stream_mode_is_manifest_bound_and_wired_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        live.PydanticAIStructuredGenerator,
        "openrouter",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        live,
        "create_production_queued_runner",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    config = live.ProgressAwareOpenRouterConfig(
        **{
            field: getattr(_config(), field)
            for field in _config().__dataclass_fields__
            if field != "retry_mode"
        },
        retry_mode=live.ProgressAwareRetryMode.NON_REPEATING_STREAM,
    )

    assert config.to_manifest_record()["queue"]["retry_classifier"] == (
        "non_repeating_stream"
    )
    live.create_progress_aware_openrouter_runner(
        api_key="offline-injected-key",
        config=config,
        progress_sink=lambda _row: None,
        outcome_sink=lambda _row: None,
    )
    assert type(captured["retry_classifier"]) is (
        NonRepeatingStreamTransportRetryClassifier
    )


def test_opaque_http_400_once_mode_is_manifest_bound_and_wired_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        live.PydanticAIStructuredGenerator,
        "openrouter",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        live,
        "create_production_queued_runner",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    config = live.ProgressAwareOpenRouterConfig(
        **{
            field: getattr(_config(), field)
            for field in _config().__dataclass_fields__
            if field != "retry_mode"
        },
        retry_mode=live.ProgressAwareRetryMode.OPAQUE_HTTP_400_ONCE,
    )

    assert config.to_manifest_record()["queue"]["retry_classifier"] == (
        "opaque_http_400_once"
    )
    live.create_progress_aware_openrouter_runner(
        api_key="offline-injected-key",
        config=config,
        progress_sink=lambda _row: None,
        outcome_sink=lambda _row: None,
    )
    assert type(captured["retry_classifier"]) is OpaqueHTTP400OnceRetryClassifier


def test_composite_schema_repair_mode_is_manifest_bound_and_wired_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        live.PydanticAIStructuredGenerator,
        "openrouter",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        live,
        "create_production_queued_runner",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    config = live.ProgressAwareOpenRouterConfig(
        **{
            field: getattr(_config(), field)
            for field in _config().__dataclass_fields__
            if field != "retry_mode"
        },
        retry_mode=(live.ProgressAwareRetryMode.OPAQUE_HTTP_400_AND_SCHEMA_REPAIR_ONCE),
    )

    queue = config.to_manifest_record()["queue"]
    assert queue["retry_classifier"] == ("opaque_http_400_and_schema_repair_once")
    assert queue["attempt_request_policy"] == ("exact_transport_schema_repair_v4")
    repair = queue["schema_repair_policy"]
    assert repair["policy_id"] == "structured_output_schema_repair"
    assert repair["policy_version"] == 4
    assert len(repair["policy_sha256"]) == 64
    live.create_progress_aware_openrouter_runner(
        api_key="offline-injected-key",
        config=config,
        progress_sink=lambda _row: None,
        outcome_sink=lambda _row: None,
    )
    assert type(captured["retry_classifier"]) is (
        OpaqueHTTP400AndSchemaRepairOnceRetryClassifier
    )
    assert type(captured["attempt_request_policy"]) is (
        ExactTransportSchemaRepairAttemptPolicy
    )


def test_bounded_schema_repair_mode_is_manifest_bound_and_wired_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        live.PydanticAIStructuredGenerator,
        "openrouter",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        live,
        "create_production_queued_runner",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    config = live.ProgressAwareOpenRouterConfig(
        **{
            field: getattr(_config(), field)
            for field in _config().__dataclass_fields__
            if field != "retry_mode"
        },
        retry_mode=(
            live.ProgressAwareRetryMode.OPAQUE_HTTP_400_AND_BOUNDED_SCHEMA_REPAIR
        ),
    )

    queue = config.to_manifest_record()["queue"]
    assert queue["retry_classifier"] == (
        "opaque_http_400_and_bounded_schema_repair"
    )
    assert queue["attempt_request_policy"] == (
        "exact_transport_schema_repair_v4"
    )
    assert queue["schema_repair_policy"]["policy_version"] == 4
    live.create_progress_aware_openrouter_runner(
        api_key="offline-injected-key",
        config=config,
        progress_sink=lambda _row: None,
        outcome_sink=lambda _row: None,
    )
    assert type(captured["retry_classifier"]) is (
        OpaqueHTTP400AndBoundedSchemaRepairRetryClassifier
    )
    assert type(captured["attempt_request_policy"]) is (
        ExactTransportSchemaRepairAttemptPolicy
    )


def test_first_event_resilient_mode_is_manifest_bound_and_wired_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        live.PydanticAIStructuredGenerator,
        "openrouter",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        live,
        "create_production_queued_runner",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    config = live.ProgressAwareOpenRouterConfig(
        **{
            field: getattr(_config(), field)
            for field in _config().__dataclass_fields__
            if field != "retry_mode"
        },
        retry_mode=(
            live.ProgressAwareRetryMode.FIRST_EVENT_RESILIENT_BOUNDED_SCHEMA_REPAIR
        ),
    )

    queue = config.to_manifest_record()["queue"]
    assert queue["retry_classifier"] == (
        "first_event_resilient_bounded_schema_repair"
    )
    assert queue["attempt_request_policy"] == (
        "exact_transport_schema_repair_v4"
    )
    assert queue["schema_repair_policy"]["policy_version"] == 4
    live.create_progress_aware_openrouter_runner(
        api_key="offline-injected-key",
        config=config,
        progress_sink=lambda _row: None,
        outcome_sink=lambda _row: None,
    )
    assert type(captured["retry_classifier"]) is (
        FirstEventResilientBoundedSchemaRepairRetryClassifier
    )
    assert type(captured["attempt_request_policy"]) is (
        ExactTransportSchemaRepairAttemptPolicy
    )


def test_bounded_prestream_mode_is_manifest_bound_and_wired_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        live.PydanticAIStructuredGenerator,
        "openrouter",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        live,
        "create_production_queued_runner",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    config = live.ProgressAwareOpenRouterConfig(
        **{
            field: getattr(_config(), field)
            for field in _config().__dataclass_fields__
            if field != "retry_mode"
        },
        retry_mode=(
            live.ProgressAwareRetryMode.BOUNDED_PRESTREAM_AND_SCHEMA_REPAIR
        ),
    )

    queue = config.to_manifest_record()["queue"]
    assert queue["retry_classifier"] == (
        "bounded_prestream_and_schema_repair"
    )
    assert queue["attempt_request_policy"] == (
        "exact_transport_schema_repair_v4"
    )
    live.create_progress_aware_openrouter_runner(
        api_key="offline-injected-key",
        config=config,
        progress_sink=lambda _row: None,
        outcome_sink=lambda _row: None,
    )
    assert type(captured["retry_classifier"]) is (
        BoundedPrestreamAndSchemaRepairRetryClassifier
    )
    assert type(captured["attempt_request_policy"]) is (
        ExactTransportSchemaRepairAttemptPolicy
    )


def test_parameter_capability_routing_is_explicit_and_opt_in() -> None:
    values = {
        field: getattr(_config(), field)
        for field in _config().__dataclass_fields__
        if field != "provider_require_parameters"
    }
    config = live.ProgressAwareOpenRouterConfig(
        **values,
        provider_require_parameters=True,
    )
    assert config.provider_options == {
        "only": ["streamlake"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"provider_only": ()},
        {"provider_only": ("streamlake", "streamlake")},
        {"connect_timeout_seconds": 0},
        {"max_attempts": 0},
        {"base_backoff_ns": 31, "max_backoff_ns": 30},
        {"rate_limit_backoff_floor_ns": 31, "max_backoff_ns": 30},
        {"retry_mode": "transport_only"},
        {"provider_require_parameters": 1},
    ],
)
def test_config_rejects_ambiguous_or_unbounded_composition(changes) -> None:
    values = {
        field: getattr(_config(), field) for field in _config().__dataclass_fields__
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        live.ProgressAwareOpenRouterConfig(**values)
