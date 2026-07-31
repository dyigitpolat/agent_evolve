"""Reusable progress-aware OpenRouter queue composition.

This integration-layer factory binds the transport semantics that must stay
identical across benchmark composition roots.  It does not read credentials,
choose a model/provider route, or validate benchmark/scientific results.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from agent_evolve.domain.llm_task_queue import MAX_ATTEMPTS, PartitionedRetryBudget

from agent_evolve.integrations.pydantic_ai.async_generator import (
    OpenRouterReasoningConfig,
    OpenRouterStructuredOutputMode,
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (
    OpenRouterOutboundRequestManifestSink,
)
from agent_evolve.integrations.pydantic_ai.json_schema_dialect import (
    OpenRouterJsonSchemaDialect,
)
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
    OutcomeSink,
    QueuedStructuredGenerationRunner,
    SCHEMA_REPAIR_POLICY_MANIFEST,
    StructuredEvidencePublicationPolicy,
    StructuredGenerationRetryClassifier,
    StructuredOutputEvidenceSink,
    StructuredRequestEvidenceSink,
    TransportOnlyStructuredGenerationRetryClassifier,
    create_production_queued_runner,
)
from agent_evolve.policies.llm_backoff import DeterministicHashJitter
from agent_evolve.ports.structured_generator import (
    StructuredStreamLivenessPolicy,
    StructuredStreamProgressSink,
)


_PROVIDER_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ProgressAwareRetryMode(str, Enum):
    """Closed retry semantics selected by a benchmark composition root."""

    STRUCTURED_GENERATION = "structured_generation"
    TRANSPORT_ONLY = "transport_only"
    NON_REPEATING_STREAM = "non_repeating_stream"
    OPAQUE_HTTP_400_ONCE = "opaque_http_400_once"
    OPAQUE_HTTP_400_AND_SCHEMA_REPAIR_ONCE = "opaque_http_400_and_schema_repair_once"
    OPAQUE_HTTP_400_AND_BOUNDED_SCHEMA_REPAIR = (
        "opaque_http_400_and_bounded_schema_repair"
    )
    FIRST_EVENT_RESILIENT_BOUNDED_SCHEMA_REPAIR = (
        "first_event_resilient_bounded_schema_repair"
    )
    BOUNDED_PRESTREAM_AND_SCHEMA_REPAIR = (
        "bounded_prestream_and_schema_repair"
    )


_SCHEMA_REPAIR_RETRY_MODES = frozenset(
    {
        ProgressAwareRetryMode.OPAQUE_HTTP_400_AND_SCHEMA_REPAIR_ONCE,
        ProgressAwareRetryMode.OPAQUE_HTTP_400_AND_BOUNDED_SCHEMA_REPAIR,
        ProgressAwareRetryMode.FIRST_EVENT_RESILIENT_BOUNDED_SCHEMA_REPAIR,
        ProgressAwareRetryMode.BOUNDED_PRESTREAM_AND_SCHEMA_REPAIR,
    }
)


@dataclass(frozen=True, slots=True)
class ProgressAwareOpenRouterConfig:
    """Frozen benchmark-neutral composition parameters."""

    model_name: str
    provider_only: tuple[str, ...]
    connect_timeout_seconds: float
    stream_liveness_policy: StructuredStreamLivenessPolicy
    max_connections: int
    max_pending: int
    max_attempts: int
    base_backoff_ns: int
    max_backoff_ns: int
    jitter_seed: int
    jitter_domain: str
    rate_limit_backoff_floor_ns: int = 0
    retry_budget: PartitionedRetryBudget | None = None
    app_title: str = "AgentEvolve research"
    reasoning_config: OpenRouterReasoningConfig | None = None
    structured_output_mode: OpenRouterStructuredOutputMode = (
        OpenRouterStructuredOutputMode.TOOL
    )
    structured_output_strict: bool = False
    json_schema_dialect: OpenRouterJsonSchemaDialect = (
        OpenRouterJsonSchemaDialect.PROVIDER_DEFAULT
    )
    provider_require_parameters: bool = False
    supports_forced_tool_choice: bool = True
    retry_mode: ProgressAwareRetryMode = ProgressAwareRetryMode.STRUCTURED_GENERATION

    def __post_init__(self) -> None:
        if type(self.model_name) is not str or "/" not in self.model_name:
            raise ValueError("model_name must be an OpenRouter model slug")
        if (
            type(self.provider_only) is not tuple
            or not self.provider_only
            or any(
                type(item) is not str or _PROVIDER_SLUG.fullmatch(item) is None
                for item in self.provider_only
            )
            or len(set(self.provider_only)) != len(self.provider_only)
        ):
            raise ValueError("provider_only must be unique closed provider slugs")
        if (
            isinstance(self.connect_timeout_seconds, bool)
            or not isinstance(self.connect_timeout_seconds, (int, float))
            or not math.isfinite(float(self.connect_timeout_seconds))
            or not 1 <= float(self.connect_timeout_seconds) <= 600
        ):
            raise ValueError("connect_timeout_seconds must lie in [1,600]")
        if type(self.stream_liveness_policy) is not StructuredStreamLivenessPolicy:
            raise TypeError(
                "stream_liveness_policy must be a StructuredStreamLivenessPolicy"
            )
        self.stream_liveness_policy.__post_init__()
        if (
            type(self.max_connections) is not int
            or not 1 <= self.max_connections <= 256
        ):
            raise ValueError("max_connections must lie in [1,256]")
        if type(self.max_pending) is not int or self.max_pending < 0:
            raise ValueError("max_pending must be a non-negative exact integer")
        if (
            type(self.max_attempts) is not int
            or not 1 <= self.max_attempts <= MAX_ATTEMPTS
        ):
            raise ValueError(f"max_attempts must lie in [1,{MAX_ATTEMPTS}]")
        if self.retry_budget is not None and (
            type(self.retry_budget) is not PartitionedRetryBudget
        ):
            raise TypeError(
                "retry_budget must be a PartitionedRetryBudget or None"
            )
        if self.retry_budget is not None:
            PartitionedRetryBudget.__post_init__(self.retry_budget)
        for name in (
            "base_backoff_ns",
            "max_backoff_ns",
            "rate_limit_backoff_floor_ns",
            "jitter_seed",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.base_backoff_ns > self.max_backoff_ns:
            raise ValueError("base_backoff_ns cannot exceed max_backoff_ns")
        if self.rate_limit_backoff_floor_ns > self.max_backoff_ns:
            raise ValueError(
                "rate_limit_backoff_floor_ns cannot exceed max_backoff_ns"
            )
        # Reuse the jitter value object's closed domain validation.
        DeterministicHashJitter(
            seed=self.jitter_seed,
            domain=self.jitter_domain,
        )
        if type(self.app_title) is not str or not self.app_title.strip():
            raise ValueError("app_title must be non-empty")
        if self.reasoning_config is not None and (
            type(self.reasoning_config) is not OpenRouterReasoningConfig
        ):
            raise TypeError(
                "reasoning_config must be an OpenRouterReasoningConfig or None"
            )
        if type(self.structured_output_mode) is not OpenRouterStructuredOutputMode:
            raise TypeError(
                "structured_output_mode must be an exact "
                "OpenRouterStructuredOutputMode"
            )
        if type(self.structured_output_strict) is not bool:
            raise TypeError("structured_output_strict must be an exact bool")
        if type(self.json_schema_dialect) is not OpenRouterJsonSchemaDialect:
            raise TypeError(
                "json_schema_dialect must be an exact "
                "OpenRouterJsonSchemaDialect"
            )
        if type(self.provider_require_parameters) is not bool:
            raise TypeError("provider_require_parameters must be an exact bool")
        if type(self.supports_forced_tool_choice) is not bool:
            raise TypeError("supports_forced_tool_choice must be an exact bool")
        if type(self.retry_mode) is not ProgressAwareRetryMode:
            raise TypeError("retry_mode must be an exact ProgressAwareRetryMode")

    @property
    def provider_options(self) -> dict[str, object]:
        """Return a fresh StreamLake/provider-only, no-fallback payload."""

        values: dict[str, object] = {
            "only": list(self.provider_only),
            "allow_fallbacks": False,
        }
        if self.provider_require_parameters:
            values["require_parameters"] = True
        return values

    def to_manifest_record(self) -> dict[str, object]:
        """Project every frozen composition decision to canonical JSON values."""

        policy = self.stream_liveness_policy
        return {
            "model_name": self.model_name,
            "provider_options": self.provider_options,
            "connect_timeout_seconds": float(self.connect_timeout_seconds),
            "stream_liveness": {
                "first_event_timeout_ns": policy.first_event_timeout_ns,
                "idle_timeout_ns": policy.idle_timeout_ns,
                "absolute_timeout_ns": policy.absolute_timeout_ns,
                "cleanup": policy.cleanup_policy.to_manifest_record(),
            },
            "queue": {
                "max_in_flight": self.max_connections,
                "max_pending": self.max_pending,
                "max_attempts": self.max_attempts,
                **(
                    {
                        "retry_budget": {
                            "output_invalid_retries": (
                                self.retry_budget.output_invalid_retries
                            ),
                            "transport_retries": (
                                self.retry_budget.transport_retries
                            ),
                        }
                    }
                    if self.retry_budget is not None
                    else {}
                ),
                "attempt_timeout_ns": None,
                "attempt_request_policy": (
                    "exact_transport_schema_repair_v4"
                    if self.retry_mode in _SCHEMA_REPAIR_RETRY_MODES
                    else "exact_payload"
                ),
                **(
                    {
                        "schema_repair_policy": (
                            SCHEMA_REPAIR_POLICY_MANIFEST.to_trace_record()
                        )
                    }
                    if self.retry_mode in _SCHEMA_REPAIR_RETRY_MODES
                    else {}
                ),
                "retry_classifier": self.retry_mode.value,
                "backoff": {
                    "kind": (
                        "exponential_deterministic_task_keyed_full_jitter_"
                        "with_rate_limit_floor"
                        if self.rate_limit_backoff_floor_ns
                        else "exponential_deterministic_task_keyed_full_jitter"
                    ),
                    "base_backoff_ns": self.base_backoff_ns,
                    "max_backoff_ns": self.max_backoff_ns,
                    "rate_limit_backoff_floor_ns": (
                        self.rate_limit_backoff_floor_ns
                    ),
                    "jitter_seed": self.jitter_seed,
                    "jitter_domain": self.jitter_domain,
                },
            },
            "reasoning": (
                None
                if self.reasoning_config is None
                else self.reasoning_config.to_model_setting()
            ),
            "structured_output_mode": self.structured_output_mode.value,
            "structured_output_strict": self.structured_output_strict,
            "json_schema_dialect": self.json_schema_dialect.value,
            "supports_forced_tool_choice": self.supports_forced_tool_choice,
            "lifecycle": "queued_runner_owns_generator_and_http_client",
        }


def create_progress_aware_openrouter_runner(
    *,
    api_key: str,
    config: ProgressAwareOpenRouterConfig,
    progress_sink: StructuredStreamProgressSink,
    outcome_sink: OutcomeSink,
    request_evidence_sink: StructuredRequestEvidenceSink | None = None,
    output_evidence_sink: StructuredOutputEvidenceSink | None = None,
    outbound_request_manifest_sink: (
        OpenRouterOutboundRequestManifestSink | None
    ) = None,
    evidence_publication_policy: StructuredEvidencePublicationPolicy = (
        StructuredEvidencePublicationPolicy.BEST_EFFORT
    ),
) -> QueuedStructuredGenerationRunner:
    """Compose one owned production runner with no competing total timeout.

    The API key is an explicit composition-root input and is never loaded here.
    Both telemetry sinks are required and fail closed before a successful
    response may escape the runner.
    """

    if type(config) is not ProgressAwareOpenRouterConfig:
        raise TypeError("config must be a ProgressAwareOpenRouterConfig")
    config.__post_init__()
    if not callable(progress_sink):
        raise TypeError("progress_sink must be callable")
    if not callable(outcome_sink):
        raise TypeError("outcome_sink must be callable")
    for name, sink in (
        ("request_evidence_sink", request_evidence_sink),
        ("output_evidence_sink", output_evidence_sink),
        ("outbound_request_manifest_sink", outbound_request_manifest_sink),
    ):
        if sink is not None and not callable(sink):
            raise TypeError(f"{name} must be callable or None")
    if type(evidence_publication_policy) is not StructuredEvidencePublicationPolicy:
        raise TypeError(
            "evidence_publication_policy must be a StructuredEvidencePublicationPolicy"
        )
    if evidence_publication_policy is StructuredEvidencePublicationPolicy.REQUIRED:
        if request_evidence_sink is None or output_evidence_sink is None:
            raise ValueError("required structured evidence needs both sinks")

    generator = PydanticAIStructuredGenerator.openrouter(
        api_key=api_key,
        model_name=config.model_name,
        max_connections=config.max_connections,
        timeout_seconds=float(config.connect_timeout_seconds),
        provider_options=config.provider_options,
        reasoning_config=config.reasoning_config,
        structured_output_mode=config.structured_output_mode,
        structured_output_strict=config.structured_output_strict,
        json_schema_dialect=config.json_schema_dialect,
        supports_forced_tool_choice=config.supports_forced_tool_choice,
        app_title=config.app_title,
        stream_liveness_policy=config.stream_liveness_policy,
        stream_progress_sink=progress_sink,
        outbound_request_manifest_sink=outbound_request_manifest_sink,
    )
    return create_production_queued_runner(
        generator=generator,
        max_in_flight=config.max_connections,
        max_pending=config.max_pending,
        max_attempts=config.max_attempts,
        retry_budget=config.retry_budget,
        attempt_timeout_ns=None,
        base_backoff_ns=config.base_backoff_ns,
        max_backoff_ns=config.max_backoff_ns,
        rate_limit_backoff_floor_ns=config.rate_limit_backoff_floor_ns,
        jitter_policy=DeterministicHashJitter(
            seed=config.jitter_seed,
            domain=config.jitter_domain,
        ),
        close_generator=True,
        outcome_sink=outcome_sink,
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
        request_evidence_sink=request_evidence_sink,
        output_evidence_sink=output_evidence_sink,
        evidence_publication_policy=evidence_publication_policy,
        attempt_request_policy=(
            ExactTransportSchemaRepairAttemptPolicy()
            if config.retry_mode in _SCHEMA_REPAIR_RETRY_MODES
            else ExactPayloadAttemptPolicy()
        ),
        retry_classifier={
            ProgressAwareRetryMode.STRUCTURED_GENERATION: (
                StructuredGenerationRetryClassifier
            ),
            ProgressAwareRetryMode.TRANSPORT_ONLY: (
                TransportOnlyStructuredGenerationRetryClassifier
            ),
            ProgressAwareRetryMode.NON_REPEATING_STREAM: (
                NonRepeatingStreamTransportRetryClassifier
            ),
            ProgressAwareRetryMode.OPAQUE_HTTP_400_ONCE: (
                OpaqueHTTP400OnceRetryClassifier
            ),
            ProgressAwareRetryMode.OPAQUE_HTTP_400_AND_SCHEMA_REPAIR_ONCE: (
                OpaqueHTTP400AndSchemaRepairOnceRetryClassifier
            ),
            ProgressAwareRetryMode.OPAQUE_HTTP_400_AND_BOUNDED_SCHEMA_REPAIR: (
                OpaqueHTTP400AndBoundedSchemaRepairRetryClassifier
            ),
            ProgressAwareRetryMode.FIRST_EVENT_RESILIENT_BOUNDED_SCHEMA_REPAIR: (
                FirstEventResilientBoundedSchemaRepairRetryClassifier
            ),
            ProgressAwareRetryMode.BOUNDED_PRESTREAM_AND_SCHEMA_REPAIR: (
                BoundedPrestreamAndSchemaRepairRetryClassifier
            ),
        }[config.retry_mode](),
    )


__all__ = [
    "ProgressAwareOpenRouterConfig",
    "ProgressAwareRetryMode",
    "create_progress_aware_openrouter_runner",
]
