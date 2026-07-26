from __future__ import annotations

import asyncio
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from agent_evolve.domain.llm_task_queue import QueueSnapshot
from agent_evolve.integrations.pydantic_ai.async_generator import (
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (
    OpenRouterOutboundRequestManifestPublisher,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    ExactTransportSchemaRepairAttemptPolicy,
    OpaqueHTTP400AndSchemaRepairOnceRetryClassifier,
    OutcomePublicationPolicy,
    StructuredEvidencePublicationPolicy,
    create_production_queued_runner,
)
from agent_evolve.policies.llm_backoff import DeterministicHashJitter
from agent_evolve.ports.structured_generator import (
    StructuredGenerationResponse,
    StructuredStreamChannel,
    StructuredStreamProgress,
    StructuredStreamProgressKind,
)

from examples.development import run_boils_exact_stack_conformance as conformance


class _ForbiddenBoundary:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.calls += 1
        raise AssertionError(f"readiness crossed {self.name}")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


class _ForbiddenAgent:
    async def run(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("offline generator must not dispatch an agent")


class _OfflineBoilsStreamingGenerator(PydanticAIStructuredGenerator):
    """Provider-free attempt retaining production queue/HTTP evidence seams."""

    def __init__(self, *, config: object, progress_sink: object, outbound_sink: object):
        self._test_config = config
        self._test_progress_sink = progress_sink
        self._test_publisher = OpenRouterOutboundRequestManifestPublisher(
            outbound_sink
        )
        super().__init__(
            agent=_ForbiddenAgent(),
            requested_model=config.model_name,
            provider_options=config.provider_options,
            reasoning_config=config.reasoning_config,
            stream_liveness_policy=config.stream_liveness_policy,
            stream_progress_sink=progress_sink,
            outbound_request_manifest_publisher=self._test_publisher,
        )

    def _wire_schema(self, request: object) -> dict[str, object]:
        logical = request.output_type.model_json_schema(mode="validation")
        detached = json.loads(_canonical(logical))
        detached.pop("description", None)
        wire = OpenAIJsonSchemaTransformer(detached, strict=False).walk()
        assert type(wire) is dict
        return wire

    async def _publish_outbound(self, request: object) -> None:
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
                        "description": "Provider-free BOiLS conformance projection.",
                        "parameters": self._wire_schema(request),
                    },
                }
            ],
            "usage": {"include": True},
        }
        calls = 0

        def handler(http_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
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
                    "https://openrouter.ai/api/v1/chat/completions", json=body
                )
                assert response.status_code == 204
        assert calls == 1

    @staticmethod
    def _option_ids(request: object) -> tuple[str, ...]:
        schema = request.output_type.model_json_schema()
        values = schema["$defs"]["CalibratedPortfolioSlateMember"]["properties"][
            "option_id"
        ]["enum"]
        by_position: dict[str, list[str]] = {}
        for value in values:
            position = value.split(".")[1]
            by_position.setdefault(position, []).append(value)
        selected: list[str] = []
        used_families: set[str] = set()
        for position in sorted(by_position)[:8]:
            candidates = sorted(by_position[position])
            choice = next(
                (
                    value
                    for value in candidates
                    if value.rsplit(".", 1)[-1] not in used_families
                ),
                candidates[0],
            )
            selected.append(choice)
            used_families.add(choice.rsplit(".", 1)[-1])
        return tuple(selected)

    async def generate_once(self, request: object) -> StructuredGenerationResponse:
        assert request.provider_attempt_id is not None
        await self._publish_outbound(request)
        content = b"provider-free-boils-tool-fragment"
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
        members = [
            {
                "option_id": option_id,
                "supporting_card_keys": ["card.boils.bootstrap"]
                if index == 1
                else [],
                "effect_predictions": [
                    {
                        "metric_id": "total_levels",
                        "direction": "decrease",
                        "confidence": "high",
                    },
                    {
                        "metric_id": "total_lut_count",
                        "direction": "decrease",
                        "confidence": "medium",
                    },
                ],
                "role_proposal": (
                    "exploit"
                    if index <= 3
                    else "falsify"
                    if index <= 6
                    else "coverage"
                ),
                "design_rationale": f"Provider-free BOiLS member {index}.",
            }
            for index, option_id in enumerate(self._option_ids(request), start=1)
        ]
        output = request.output_type.model_validate({"members": members}, strict=True)
        return StructuredGenerationResponse(
            value=output,
            requested_model="deepseek/deepseek-v4-pro",
            resolved_model="deepseek/deepseek-v4-pro",
            resolved_provider="StreamLake",
            provider_response_id="offline-boils-response-000001",
            finish_reason="tool_calls",
            input_tokens=4_321,
            output_tokens=321,
            reasoning_tokens=117,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0.003"),
            latency_ns=1_000_000,
        )


def _offline_success_runner_factory(**kwargs: object) -> object:
    assert kwargs["api_key"] == "offline-secret-never-persist"
    config = kwargs["config"]
    generator = _OfflineBoilsStreamingGenerator(
        config=config,
        progress_sink=kwargs["progress_sink"],
        outbound_sink=kwargs["outbound_request_manifest_sink"],
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
            seed=config.jitter_seed, domain=config.jitter_domain
        ),
        close_generator=True,
        outcome_sink=kwargs["outcome_sink"],
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
        request_evidence_sink=kwargs["request_evidence_sink"],
        output_evidence_sink=kwargs["output_evidence_sink"],
        evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
        attempt_request_policy=ExactTransportSchemaRepairAttemptPolicy(),
        retry_classifier=OpaqueHTTP400AndSchemaRepairOnceRetryClassifier(),
    )


def _offline_dependencies(
    credential: _ForbiddenBoundary,
    runner: _ForbiddenBoundary,
) -> conformance.ConformanceDependencies:
    return conformance.ConformanceDependencies(
        inputs_factory=conformance.build_conformance_inputs,
        credential_loader=credential,
        runner_factory=runner,
    )


def test_readiness_is_credential_provider_and_abc_free(tmp_path: Path) -> None:
    credential = _ForbiddenBoundary("credential boundary")
    runner = _ForbiddenBoundary("provider runner boundary")
    summary = asyncio.run(
        conformance._execute_readiness_for_testing(
            "offline_readiness",
            run_root=tmp_path,
            dependencies=_offline_dependencies(credential, runner),
        )
    )

    readiness = summary["readiness"]
    contract = readiness["contract"]
    assert credential.calls == 0
    assert runner.calls == 0
    assert readiness["status"] == "ready_offline_test_only"
    assert readiness["credentials_read"] is False
    assert readiness["provider_call_attempted"] is False
    assert readiness["abc_executions"] == 0
    assert readiness["evaluator_call_count"] == 0
    assert contract["all_gates_pass"] is True
    assert all(contract["gates"].values())
    assert contract["request"]["finite_option_count"] == 200
    assert contract["request"]["proposal_width"] == 8
    assert contract["request"]["evaluation_width"] == 4
    assert contract["request"]["max_output_tokens"] == 384_000
    assert contract["request"]["output_tool_name"] == (
        "propose_calibrated_portfolio_slate"
    )
    assert contract["provider_config"]["model_name"] == (
        "deepseek/deepseek-v4-pro"
    )
    assert contract["provider_config"]["provider_options"] == {
        "only": ["streamlake"],
        "allow_fallbacks": False,
    }
    assert contract["provider_config"]["reasoning"] == {"effort": "xhigh"}
    assert "mode" not in contract["provider_config"]["reasoning"]
    assert readiness["scientific_result_eligible"] is False
    assert readiness["optimization_result_eligible"] is False
    verified = conformance.verify_finalized_run_directory(Path(summary["run_dir"]))
    assert verified["status"] == "ready_offline_test_only"


def test_test_entrypoint_rejects_production_dependencies(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="injected"):
        asyncio.run(
            conformance._execute_readiness_for_testing(
                "reject_production_dependencies",
                run_root=tmp_path,
                dependencies=conformance._production_dependencies(),
            )
        )


class _FailingRunner:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def __call__(self, request: object) -> object:
        del request
        self.calls += 1
        raise RuntimeError("SECRET_PROVIDER_BODY_MUST_NOT_BE_PERSISTED")

    async def aclose(self) -> None:
        self.closed = True

    async def snapshot(self) -> QueueSnapshot:
        return QueueSnapshot(
            max_in_flight=3,
            max_pending=8,
            in_flight=0,
            pending=0,
            closed=self.closed,
        )


class _FailingRunnerFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.runner = _FailingRunner()

    def __call__(self, **kwargs: object) -> _FailingRunner:
        assert kwargs["config"].to_manifest_record()["reasoning"] == {
            "effort": "xhigh"
        }
        self.calls += 1
        return self.runner


def test_failed_injected_live_path_is_durable_and_content_safe(
    tmp_path: Path,
) -> None:
    forbidden_readiness_credential = _ForbiddenBoundary("readiness credential")
    forbidden_readiness_runner = _ForbiddenBoundary("readiness provider")
    readiness = asyncio.run(
        conformance._execute_readiness_for_testing(
            "failure_readiness",
            run_root=tmp_path,
            dependencies=_offline_dependencies(
                forbidden_readiness_credential, forbidden_readiness_runner
            ),
        )
    )
    credential_calls = 0

    def credential() -> str:
        nonlocal credential_calls
        credential_calls += 1
        return "offline-test-key-never-sent"

    factory = _FailingRunnerFactory()
    summary = asyncio.run(
        conformance._execute_live_for_testing(
            "failure_live",
            readiness_dir=Path(readiness["run_dir"]),
            authorization=conformance.LIVE_AUTHORIZATION,
            run_root=tmp_path,
            dependencies=conformance.ConformanceDependencies(
                inputs_factory=conformance.build_conformance_inputs,
                credential_loader=credential,
                runner_factory=factory,
            ),
        )
    )

    assert credential_calls == 1
    assert factory.calls == 1
    assert factory.runner.calls == 1
    assert factory.runner.closed is True
    assert summary["failed"] is True
    result = summary["result"]
    assert result["status"] == "failed_conformance_only"
    assert result["failure_type"] == "RuntimeError"
    assert result["abc_executions"] == 0
    assert result["evaluator_call_count"] == 0
    assert result["diagnosis"]["raw_exception_text_retained"] is False
    assert result["diagnosis"]["raw_provider_body_retained"] is False
    assert (
        result["diagnosis"]["failure_schema_supports_exception_provenance_v8"]
        is True
    )
    run_dir = Path(summary["run_dir"])
    assert conformance.verify_finalized_run_directory(run_dir)["status"] == (
        "failed_conformance_only"
    )
    assert (run_dir / "provider_attempt_join.json").is_file()
    assert result["queue_cleanup"]["empty_before_close"] is True
    assert result["queue_cleanup"]["closed_and_empty_after_close"] is True
    assert (run_dir / "queue_snapshot_before_close.json").is_file()
    assert (run_dir / "queue_snapshot_after_close.json").is_file()
    retained = b"".join(path.read_bytes() for path in run_dir.rglob("*") if path.is_file())
    assert b"SECRET_PROVIDER_BODY_MUST_NOT_BE_PERSISTED" not in retained
    assert b"offline-test-key-never-sent" not in retained


def test_successful_injected_live_call_joins_every_durable_boundary(
    tmp_path: Path,
) -> None:
    readiness = asyncio.run(
        conformance._execute_readiness_for_testing(
            "success_readiness",
            run_root=tmp_path,
            dependencies=_offline_dependencies(
                _ForbiddenBoundary("readiness credential"),
                _ForbiddenBoundary("readiness provider"),
            ),
        )
    )
    credential_calls = 0

    def credential() -> str:
        nonlocal credential_calls
        credential_calls += 1
        return "offline-secret-never-persist"

    summary = asyncio.run(
        conformance._execute_live_for_testing(
            "success_live",
            readiness_dir=Path(readiness["run_dir"]),
            authorization=conformance.LIVE_AUTHORIZATION,
            run_root=tmp_path,
            dependencies=conformance.ConformanceDependencies(
                inputs_factory=conformance.build_conformance_inputs,
                credential_loader=credential,
                runner_factory=_offline_success_runner_factory,
            ),
        )
    )

    assert credential_calls == 1
    assert summary["failed"] is False
    result = summary["result"]
    assert result["status"] == "completed_offline_test_only"
    assert result["logical_call_count"] == 1
    assert result["physical_attempt_count"] == 1
    assert result["proposal_width"] == 8
    assert result["evaluation_width"] == 4
    assert len(result["selected_option_ids"]) == 4
    assert result["response"]["reasoning_tokens"] == 117
    assert result["response"]["resolved_provider"] == "StreamLake"
    assert result["provider_attempt_join"]["join_valid"] is True
    assert result["queue_cleanup"]["empty_before_close"] is True
    assert result["queue_cleanup"]["closed_and_empty_after_close"] is True
    assert result["abc_executions"] == 0
    assert result["evaluator_call_count"] == 0
    assert result["scientific_result_eligible"] is False
    assert result["optimization_result_eligible"] is False
    run_dir = Path(summary["run_dir"])
    expected_rows = {
        "provider_requests.jsonl": 1,
        "provider_attempt_requests.jsonl": 1,
        "provider_progress.jsonl": 2,
        "provider_outcomes.jsonl": 1,
        "provider_outputs.jsonl": 1,
    }
    for name, count in expected_rows.items():
        assert len((run_dir / name).read_text(encoding="utf-8").splitlines()) == count
    before_close = json.loads(
        (run_dir / "queue_snapshot_before_close.json").read_text(encoding="utf-8")
    )
    after_close = json.loads(
        (run_dir / "queue_snapshot_after_close.json").read_text(encoding="utf-8")
    )
    assert before_close["pending"] == before_close["in_flight"] == 0
    assert before_close["closed"] is False
    assert after_close["pending"] == after_close["in_flight"] == 0
    assert after_close["closed"] is True
    outbound = json.loads(
        (run_dir / "provider_attempt_requests.jsonl").read_text(encoding="utf-8")
    )
    assert outbound["settings"]["reasoning"] == {"effort": "xhigh"}
    assert outbound["settings"]["provider"] == {
        "only": ["streamlake"],
        "allow_fallbacks": False,
    }
    assert outbound["settings"]["max_completion_tokens"] == 384_000
    assert outbound["settings"]["stream"] is True
    assert outbound["settings"]["usage"] == {"include": True}
    assert outbound["settings"]["tool_choice"] == "required"
    assert conformance.verify_finalized_run_directory(run_dir)["status"] == (
        "completed_offline_test_only"
    )
    retained = b"".join(path.read_bytes() for path in run_dir.rglob("*") if path.is_file())
    assert b"offline-secret-never-persist" not in retained
