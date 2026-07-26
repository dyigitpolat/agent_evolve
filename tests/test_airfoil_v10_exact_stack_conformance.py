"""Offline gates for the one-call Airfoil-v10 exact-stack conformance harness."""

from __future__ import annotations

import asyncio
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from agent_evolve.domain.llm_task_queue import CanonicalProviderErrorCode
from agent_evolve.integrations.pydantic_ai.async_generator import (
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (
    OpenRouterOutboundRequestManifestPublisher,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    ExactPayloadAttemptPolicy,
    OutcomePublicationPolicy,
    StructuredEvidencePublicationPolicy,
    TransportOnlyStructuredGenerationRetryClassifier,
    create_production_queued_runner,
)
from agent_evolve.policies.llm_backoff import DeterministicHashJitter
from agent_evolve.ports.structured_generator import (
    GenerationFailureKind,
    StructuredGenerationError,
    StructuredGenerationResponse,
    StructuredStreamChannel,
    StructuredStreamProgress,
    StructuredStreamProgressKind,
)
from examples.development import run_airfoil_v10_exact_stack_conformance as harness
from examples.development.run_airfoil_v10_exact_stack_conformance import (
    EXPECTED_LOGICAL_SCHEMA_SHA256,
    EXPECTED_LOGICAL_SCHEMA_UTF8_BYTES,
    EXPECTED_OUTPUT_TOOL_NAME,
    LIVE_AUTHORIZATION,
    LOGICAL_CALL_ID,
    MAX_PHYSICAL_ATTEMPTS,
    TARGET_PROMPT_UTF8_BYTES,
    ConformanceDependencies,
    ConformanceInputs,
    build_conformance_inputs,
    build_contract,
    build_high_level_request,
    capture_low_level_request,
    _failure_diagnosis,
    _execute_live_for_testing,
    _execute_readiness_for_testing,
)


@pytest.fixture(scope="module")
def conformance_inputs() -> ConformanceInputs:
    value = build_conformance_inputs()
    assert value.evaluator_guard.calls == 0
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


class _ForbiddenAgent:
    async def run(self, *_args, **_kwargs):
        raise AssertionError("offline generator override must not dispatch an agent")


class _OfflineManifestingStreamingGenerator(PydanticAIStructuredGenerator):
    """Local deterministic attempt retaining the production queue boundaries."""

    def __init__(
        self,
        *,
        config,
        progress_sink,
        outbound_sink,
        fail_http_400: bool,
    ) -> None:
        self._test_config = config
        self._test_progress_sink = progress_sink
        self._test_publisher = OpenRouterOutboundRequestManifestPublisher(
            outbound_sink
        )
        self._test_fail_http_400 = fail_http_400
        super().__init__(
            agent=_ForbiddenAgent(),
            requested_model=config.model_name,
            provider_options=config.provider_options,
            reasoning_config=config.reasoning_config,
            stream_liveness_policy=config.stream_liveness_policy,
            stream_progress_sink=progress_sink,
            outbound_request_manifest_publisher=self._test_publisher,
        )

    def _wire_schema(self, request) -> dict[str, object]:
        logical = request.output_type.model_json_schema(mode="validation")
        detached = json.loads(_canonical(logical))
        detached.pop("description", None)
        wire = OpenAIJsonSchemaTransformer(detached, strict=False).walk()
        assert type(wire) is dict
        return wire

    async def _publish_outbound(self, request) -> None:
        body = {
            "max_completion_tokens": request.max_output_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
            "model": self._test_config.model_name,
            "provider": self._test_config.provider_options,
            "reasoning": self._test_config.reasoning_config.to_model_setting(),
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": float(request.temperature),
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": request.output_tool_name,
                        "description": "Offline exact-stack function-tool projection.",
                        "parameters": self._wire_schema(request),
                    },
                }
            ],
            "usage": {"include": True},
        }
        transport_calls = 0

        def handler(http_request: httpx.Request) -> httpx.Response:
            nonlocal transport_calls
            transport_calls += 1
            return httpx.Response(204, request=http_request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            event_hooks={"request": [self._test_publisher.httpx_request_hook]},
        ) as client:
            with self._test_publisher.bind(
                request,
                requested_model=self._test_config.model_name,
                provider=self._test_config.provider_options,
                reasoning=self._test_config.reasoning_config.to_model_setting(),
                stream=True,
            ):
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=body,
                )
                assert response.status_code == 204
        assert transport_calls == 1

    async def generate_once(self, request):
        assert request.provider_attempt_id is not None
        await self._publish_outbound(request)
        if self._test_fail_http_400:
            raise StructuredGenerationError(
                kind=GenerationFailureKind.INVALID_REQUEST,
                retryable=False,
                safe_message="provider rejected invalid request parameters",
                status_code=400,
                provider_error_code=(
                    CanonicalProviderErrorCode.INVALID_REQUEST_ERROR
                ),
                provider_error_envelope_sha256=hashlib.sha256(
                    b"offline-redacted-http-error-envelope"
                ).hexdigest(),
            )

        content = b"offline-tool-call-fragment"
        digest = hashlib.sha256(content).hexdigest()
        attempt_id = request.provider_attempt_id.value
        self._test_progress_sink(
            StructuredStreamProgress(
                call_id=request.call_id.value,
                provider_attempt_id=attempt_id,
                sequence=1,
                kind=StructuredStreamProgressKind.PART_STARTED,
                channel=StructuredStreamChannel.TOOL_CALL,
                elapsed_ns=10,
                event_content_utf8_bytes=len(content),
                cumulative_content_utf8_bytes=len(content),
                rolling_content_sha256=digest,
            )
        )
        self._test_progress_sink(
            StructuredStreamProgress(
                call_id=request.call_id.value,
                provider_attempt_id=attempt_id,
                sequence=2,
                kind=StructuredStreamProgressKind.STREAM_COMPLETED,
                channel=StructuredStreamChannel.OTHER,
                elapsed_ns=20,
                event_content_utf8_bytes=0,
                cumulative_content_utf8_bytes=len(content),
                rolling_content_sha256=digest,
            )
        )
        schema = request.output_type.model_json_schema()
        option_id = schema["properties"]["option_id"]["enum"][0]
        output = request.output_type(
            option_id=option_id,
            design_rationale="Offline transport-only conformance response.",
            claimed_insight_ids=[],
        )
        return StructuredGenerationResponse(
            value=output,
            requested_model="deepseek/deepseek-v4-pro",
            resolved_model="deepseek/deepseek-v4-pro-20260423",
            resolved_provider="StreamLake",
            provider_response_id="offline-openrouter-response-000001",
            finish_reason="tool_calls",
            input_tokens=4_321,
            output_tokens=31,
            reasoning_tokens=17,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0.003"),
            latency_ns=1_000_000,
        )


def _offline_runner_factory(*, fail_http_400: bool):
    def factory(
        *,
        api_key,
        config,
        progress_sink,
        outcome_sink,
        request_evidence_sink,
        outbound_request_manifest_sink,
        output_evidence_sink,
    ):
        assert api_key == "offline-secret-never-persist"
        generator = _OfflineManifestingStreamingGenerator(
            config=config,
            progress_sink=progress_sink,
            outbound_sink=outbound_request_manifest_sink,
            fail_http_400=fail_http_400,
        )
        return create_production_queued_runner(
            generator=generator,
            max_in_flight=config.max_connections,
            max_pending=config.max_pending,
            max_attempts=config.max_attempts,
            attempt_timeout_ns=None,
            base_backoff_ns=config.base_backoff_ns,
            max_backoff_ns=config.max_backoff_ns,
            jitter_policy=DeterministicHashJitter(
                seed=config.jitter_seed,
                domain=config.jitter_domain,
            ),
            close_generator=True,
            outcome_sink=outcome_sink,
            outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
            request_evidence_sink=request_evidence_sink,
            output_evidence_sink=output_evidence_sink,
            evidence_publication_policy=(
                StructuredEvidencePublicationPolicy.REQUIRED
            ),
            attempt_request_policy=ExactPayloadAttemptPolicy(),
            retry_classifier=TransportOnlyStructuredGenerationRetryClassifier(),
        )

    return factory


def _readiness_dependencies(conformance: ConformanceInputs):
    def no_credential() -> str:
        raise AssertionError("readiness must not read a credential")

    def no_runner(**_kwargs):
        raise AssertionError("readiness must not construct a provider runner")

    return ConformanceDependencies(
        inputs_factory=lambda: conformance,
        credential_loader=no_credential,
        runner_factory=no_runner,
    )


def test_genuine_k8_request_is_exact_and_provider_evaluator_free(
    conformance_inputs: ConformanceInputs,
) -> None:
    high = build_high_level_request(conformance_inputs.inputs)
    low = asyncio.run(capture_low_level_request(high))
    contract = asyncio.run(build_contract(conformance_inputs))

    assert len(high.prompt.encode("utf-8")) == TARGET_PROMPT_UTF8_BYTES
    assert "NON-SCIENTIFIC TRANSPORT CONFORMANCE ONLY" in high.prompt
    assert high.finite_variation_contract is not None
    assert len(high.finite_variation_contract.options) == 8
    assert len({item.option_id for item in high.finite_variation_contract.options}) == 8
    assert low.output_tool_name == EXPECTED_OUTPUT_TOOL_NAME
    assert contract["request"]["output_schema_sha256"] == (
        EXPECTED_LOGICAL_SCHEMA_SHA256
    )
    assert contract["request"]["output_schema_utf8_bytes"] == (
        EXPECTED_LOGICAL_SCHEMA_UTF8_BYTES
    )
    assert contract["transport"]["reasoning"] == {"max_tokens": 384_000}
    assert contract["transport"]["provider_options"] == {
        "only": ["streamlake"],
        "allow_fallbacks": False,
    }
    assert contract["transport"]["queue"]["max_attempts"] == (
        MAX_PHYSICAL_ATTEMPTS
    )
    assert contract["transport"]["queue"]["retry_classifier"] == (
        "transport_only"
    )
    assert contract["transport"]["stream_liveness"]["absolute_timeout_ns"] == (
        600_000_000_000
    )
    provenance = contract["v10_runtime_provenance"]
    assert provenance["production_qualification_required"] is False
    assert provenance["production_qualification_verified"] is False
    assert provenance["qualification"] is None
    assert provenance["source_sha256"] == provenance["source_closure"][
        "source_sha256"
    ]
    assert provenance["provider_configuration"]["requested_model"] == (
        "deepseek/deepseek-v4-pro"
    )
    assert provenance["provider_configuration_join_exact"] is True
    assert provenance["framework_versions_join_exact"] is True
    assert conformance_inputs.evaluator_guard.calls == 0


def test_readiness_reads_no_credential_and_calls_no_provider_or_evaluator(
    tmp_path: Path,
    conformance_inputs: ConformanceInputs,
) -> None:
    summary = asyncio.run(
        _execute_readiness_for_testing(
            "v10_exact_stack_readiness_test",
            run_root=tmp_path,
            dependencies=_readiness_dependencies(conformance_inputs),
        )
    )

    result = summary["readiness"]
    assert result["status"] == "ready_offline_test_only"
    assert result["production_stack_authenticated"] is False
    assert result["credentials_read"] is False
    assert result["provider_client_constructed"] is False
    assert result["provider_call_attempted"] is False
    assert result["evaluator_call_count"] == 0
    assert result["scientific_result_eligible"] is False
    assert result["optimization_result_eligible"] is False
    assert conformance_inputs.evaluator_guard.calls == 0


def _run_offline_live(
    *,
    tmp_path: Path,
    conformance: ConformanceInputs,
    fail_http_400: bool,
):
    readiness = asyncio.run(
        _execute_readiness_for_testing(
            "v10_exact_stack_bound_readiness",
            run_root=tmp_path,
            dependencies=_readiness_dependencies(conformance),
        )
    )
    credential_reads: list[str] = []

    def credential() -> str:
        credential_reads.append("read")
        return "offline-secret-never-persist"

    dependencies = ConformanceDependencies(
        inputs_factory=lambda: conformance,
        credential_loader=credential,
        runner_factory=_offline_runner_factory(fail_http_400=fail_http_400),
    )
    summary = asyncio.run(
        _execute_live_for_testing(
            "v10_exact_stack_live_failure" if fail_http_400 else "v10_exact_stack_live_success",
            readiness_dir=Path(readiness["run_dir"]),
            authorization=LIVE_AUTHORIZATION,
            run_root=tmp_path,
            dependencies=dependencies,
        )
    )
    assert credential_reads == ["read"]
    assert conformance.evaluator_guard.calls == 0
    return summary


def test_one_logical_live_call_joins_real_queue_stream_and_outbound_evidence(
    tmp_path: Path,
    conformance_inputs: ConformanceInputs,
) -> None:
    summary = _run_offline_live(
        tmp_path=tmp_path,
        conformance=conformance_inputs,
        fail_http_400=False,
    )
    result = summary["result"]
    run_dir = Path(summary["run_dir"])

    assert summary["failed"] is False
    assert result["status"] == "completed_offline_test_only"
    assert result["production_stack_authenticated"] is False
    assert result["precredential_source_identity_verified"] is True
    assert result["terminal_source_identity_verified"] is True
    assert result["logical_call_count"] == 1
    assert result["physical_attempt_count"] == 1
    assert result["terminal_stream_completion_observed"] is True
    assert result["response"]["reasoning_tokens"] == 17
    assert result["scientific_result_eligible"] is False
    assert result["optimization_result_eligible"] is False
    assert len((run_dir / "provider_requests.jsonl").read_text().splitlines()) == 1
    assert len(
        (run_dir / "provider_attempt_requests.jsonl").read_text().splitlines()
    ) == 1
    assert len((run_dir / "provider_outcomes.jsonl").read_text().splitlines()) == 1
    assert len((run_dir / "provider_outputs.jsonl").read_text().splitlines()) == 1
    outbound = json.loads(
        (run_dir / "provider_attempt_requests.jsonl").read_text().strip()
    )
    assert outbound["call_id"] == LOGICAL_CALL_ID
    assert outbound["settings"] == {
        "model": "deepseek/deepseek-v4-pro",
        "provider": {"only": ["streamlake"], "allow_fallbacks": False},
        "reasoning": {"max_tokens": 384_000},
        "usage": {"include": True},
        "stream": True,
        "stream_options": {"include_usage": True},
        "tool_choice": "required",
        "max_completion_tokens": 384_000,
        "temperature_hex": float(0.2).hex(),
        "response_format": None,
    }
    assert outbound["tool"]["name"] == EXPECTED_OUTPUT_TOOL_NAME
    assert outbound["request_contract"]["logical_output_schema_sha256"] == (
        EXPECTED_LOGICAL_SCHEMA_SHA256
    )
    assert all(outbound["forbidden_fields_absent"].values())
    all_artifacts = "".join(
        path.read_text(encoding="utf-8")
        for path in run_dir.iterdir()
        if path.is_file()
    )
    assert "offline-secret-never-persist" not in all_artifacts


def test_http_400_after_outbound_authentication_is_distinguished_and_sanitized(
    tmp_path: Path,
    conformance_inputs: ConformanceInputs,
) -> None:
    summary = _run_offline_live(
        tmp_path=tmp_path,
        conformance=conformance_inputs,
        fail_http_400=True,
    )
    result = summary["result"]
    diagnosis = result["diagnosis"]

    assert summary["failed"] is True
    assert result["status"] == "failed_conformance_only"
    assert result["production_stack_authenticated"] is False
    assert result["precredential_source_identity_verified"] is True
    assert result["terminal_source_identity_verified"] is True
    assert diagnosis["transport_stage"] == (
        "outbound_authenticated_before_remote_failure"
    )
    assert diagnosis["request_evidence_rows"] == 1
    assert diagnosis["outbound_manifest_rows"] == 1
    assert diagnosis["terminal_outcome_rows"] == 1
    assert diagnosis["typed_output_rows"] == 0
    assert diagnosis["last_sanitized_failure"] == {
        "kind": "invalid_request",
        "retryable": False,
        "safe_message": "provider rejected invalid request parameters",
        "status_code": 400,
        "retry_after_seconds": None,
        "provider_error_code": "invalid_request_error",
        "provider_error_envelope_sha256": hashlib.sha256(
            b"offline-redacted-http-error-envelope"
        ).hexdigest(),
        "stream_timeout_phase": None,
        "output_failure_mode": None,
        "validation_issues": [],
    }
    assert diagnosis["raw_provider_body_retained"] is False
    assert diagnosis["raw_exception_text_retained"] is False
    assert diagnosis["provider_http_diagnostics"] == {
        "status_code": 400,
        "provider_error_code": "invalid_request_error",
        "provider_error_envelope_sha256": hashlib.sha256(
            b"offline-redacted-http-error-envelope"
        ).hexdigest(),
    }
    assert diagnosis["provider_attempt_join"]["join_valid"] is True
    assert result["scientific_result_eligible"] is False
    assert result["optimization_result_eligible"] is False


def test_public_production_entry_points_ignore_rebound_default_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    conformance_inputs: ConformanceInputs,
) -> None:
    def forbidden_credential() -> str:
        raise AssertionError("rebound default credential must not be used")

    def forbidden_runner(**_kwargs):
        raise AssertionError("rebound default runner must not be used")

    rebound = ConformanceDependencies(
        inputs_factory=lambda: conformance_inputs,
        credential_loader=forbidden_credential,
        runner_factory=forbidden_runner,
    )
    monkeypatch.setattr(harness, "DEFAULT_DEPENDENCIES", rebound)
    captured: list[tuple[str, ConformanceDependencies, bool]] = []

    async def capture_readiness(_run_id: str, **kwargs):
        captured.append(
            (
                "readiness",
                kwargs["dependencies"],
                kwargs["production_stack_authenticated"],
            )
        )
        return {"captured": "readiness"}

    async def capture_live(_run_id: str, **kwargs):
        captured.append(
            (
                "live",
                kwargs["dependencies"],
                kwargs["production_stack_authenticated"],
            )
        )
        return {"captured": "live"}

    monkeypatch.setattr(harness, "_execute_readiness", capture_readiness)
    assert asyncio.run(
        harness.execute_readiness(
            "sealed_readiness",
            qualification_dir=tmp_path,
            run_root=tmp_path,
        )
    ) == {"captured": "readiness"}
    monkeypatch.setattr(harness, "_execute_live", capture_live)
    assert asyncio.run(
        harness.execute_live(
            "sealed_live",
            readiness_dir=tmp_path,
            qualification_dir=tmp_path,
            authorization=LIVE_AUTHORIZATION,
            run_root=tmp_path,
        )
    ) == {"captured": "live"}

    assert [item[0] for item in captured] == ["readiness", "live"]
    for _, dependencies, authenticated in captured:
        assert dependencies is not rebound
        assert harness._is_canonical_production_dependencies(dependencies)
        assert authenticated is True


def test_malformed_last_attempt_cannot_mask_or_leak_into_failure_diagnosis(
    conformance_inputs: ConformanceInputs,
) -> None:
    contract = asyncio.run(build_contract(conformance_inputs))
    secret = "RAW_PROVIDER_BODY_MUST_NOT_SURVIVE"
    diagnosis = _failure_diagnosis(
        contract=contract,
        request_rows=[],
        outbound_rows=[],
        progress_rows=[],
        outcome_rows=[
            {
                "schema_version": 6,
                "task_id": LOGICAL_CALL_ID,
                "status": "terminal_failure",
                "cancellation_reason": None,
                "attempts": [secret],
                "response": None,
            }
        ],
        output_rows=[],
    )

    assert diagnosis["terminal_outcome_validation_failure"] == {
        "failure_type": "ExactStackConformanceError"
    }
    assert diagnosis["last_sanitized_failure"] is None
    assert diagnosis["last_sanitized_failure_validation_failure"] == {
        "failure_type": "terminal_outcome_untrusted"
    }
    assert diagnosis["raw_provider_body_retained"] is False
    assert diagnosis["raw_exception_text_retained"] is False
    assert secret not in json.dumps(diagnosis, sort_keys=True)
