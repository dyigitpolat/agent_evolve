"""Provider-free gates for the two-parent BOiLS G1 concurrency canary."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import httpx
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.tools import GenerateToolJsonSchema
import pytest

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
    GenerationFailureKind,
    StructuredGenerationError,
    StructuredGenerationResponse,
    StructuredStreamChannel,
    StructuredStreamProgress,
    StructuredStreamProgressKind,
)
from examples.development import run_boils_g1_two_call_canary as canary


class _ForbiddenBoundary:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.calls += 1
        raise AssertionError(f"readiness crossed {self.name}")


def _dependencies(
    credential: object, runner: object
) -> canary.CanaryDependencies:
    return canary.CanaryDependencies(
        inputs_factory=canary.build_canary_inputs,
        credential_loader=credential,  # type: ignore[arg-type]
        runner_factory=runner,  # type: ignore[arg-type]
    )


def test_two_call_readiness_is_concurrent_provider_and_abc_free(
    tmp_path: Path,
) -> None:
    credential = _ForbiddenBoundary("credential boundary")
    runner = _ForbiddenBoundary("provider runner boundary")
    summary = asyncio.run(
        canary._execute_readiness_for_testing(
            "offline_two_call_readiness",
            run_root=tmp_path,
            dependencies=_dependencies(credential, runner),
        )
    )

    readiness = summary["readiness"]
    contract = readiness["contract"]
    assert credential.calls == 0
    assert runner.calls == 0
    assert readiness["status"] == "ready_offline_test_only"
    assert contract["all_gates_pass"] is True
    assert all(contract["gates"].values())
    assert contract["provider_free_peak_in_flight"] == 2
    assert [lane["call_id"] for lane in contract["lanes"]] == list(
        canary._EXPECTED_CALL_IDS
    )
    assert [lane["parent_lane_id"] for lane in contract["lanes"]] == list(
        canary._EXPECTED_LANE_IDS
    )
    assert [lane["finite_option_count"] for lane in contract["lanes"]] == [
        200,
        200,
    ]
    assert [lane["proposal_width"] for lane in contract["lanes"]] == [8, 8]
    assert [lane["evaluation_width"] for lane in contract["lanes"]] == [
        canary.campaign.PORTFOLIO_WIDTH,
        canary.campaign.PORTFOLIO_WIDTH,
    ]
    assert [lane["schema"]["logical_schema_utf8_bytes"] for lane in contract["lanes"]] == [
        6709,
        6713,
    ]
    assert [lane["prompt_utf8_bytes"] for lane in contract["lanes"]] == [
        54132,
        54089,
    ]
    assert contract["provider_config"]["model_name"] == "deepseek/deepseek-v4-pro"
    assert contract["provider_config"]["provider_options"] == {
        "only": ["streamlake"],
        "allow_fallbacks": False,
    }
    assert contract["provider_config"]["reasoning"] == {"effort": "xhigh"}
    assert "mode" not in contract["provider_config"]["reasoning"]
    constructor = contract["production_runner_constructor_binding"]
    assert constructor["runtime_module_symbol_is_captured_object"] is True
    assert constructor["captured_at_module_import"] is True
    assert len(constructor["binding_identity_sha256"]) == 64
    assert contract["credentials_read"] is False
    assert contract["provider_client_constructed"] is False
    assert contract["provider_call_attempted"] is False
    assert contract["abc_executions"] == 0
    assert contract["child_materialization_boundary"] == {
        "boundary": "selector_only_before_portfolio_evolution",
        "portfolio_evolution_constructed": False,
        "child_candidate_allocator_guard_installed": True,
        "child_candidate_allocation_attempts": 0,
        "claim_scope": (
            "canary-owned selector call graph only; no claim about external "
            "or subsequent campaign materialization"
        ),
    }
    assert contract["evaluator_call_count"] == 0
    assert contract["scientific_result_eligible"] is False
    assert contract["optimization_result_eligible"] is False
    assert contract["model_quality_result_eligible"] is False
    assert canary.verify_finalized_run_directory(Path(summary["run_dir"]))[
        "status"
    ] == "ready_offline_test_only"


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


class _RetryingOfflineBoilsGenerator(PydanticAIStructuredGenerator):
    """Two concurrent first failures, then typed successes on both retries."""

    def __init__(
        self, *, config: object, progress_sink: object, outbound_sink: object
    ) -> None:
        self._test_config = config
        self._test_progress_sink = progress_sink
        self._test_publisher = OpenRouterOutboundRequestManifestPublisher(
            outbound_sink
        )
        self._attempts: defaultdict[str, int] = defaultdict(int)
        self._first_arrivals = 0
        self._both_first_attempts = asyncio.Event()
        self.first_attempt_peak = 0
        self._first_in_flight = 0
        super().__init__(
            agent=_ForbiddenAgent(),
            requested_model=config.model_name,
            provider_options=config.provider_options,
            reasoning_config=config.reasoning_config,
            stream_liveness_policy=config.stream_liveness_policy,
            stream_progress_sink=progress_sink,
            outbound_request_manifest_publisher=self._test_publisher,
        )

    @staticmethod
    def _wire_schema(request: object) -> dict[str, object]:
        logical = request.output_type.model_json_schema(
            mode="validation",
            schema_generator=GenerateToolJsonSchema,
        )
        detached = deepcopy(logical)
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
            "tool_choice": (
                "required"
                if self._test_config.supports_forced_tool_choice
                else "auto"
            ),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": request.output_tool_name,
                        "description": "Provider-free two-lane BOiLS canary.",
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
                expected_tool_choice=(
                    "required"
                    if self._test_config.supports_forced_tool_choice
                    else "auto"
                ),
            ):
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions", json=body
                )
                assert response.status_code == 204
        assert calls == 1

    @staticmethod
    def _option_ids(request: object) -> tuple[str, ...]:
        values = request.output_type.model_json_schema()["$defs"][
            "CalibratedPortfolioSlateMember"
        ]["properties"]["option_id"]["enum"]
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
        call_id = request.call_id.value
        self._attempts[call_id] += 1
        attempt_number = self._attempts[call_id]
        await self._publish_outbound(request)
        if attempt_number == 1:
            self._first_in_flight += 1
            self.first_attempt_peak = max(
                self.first_attempt_peak, self._first_in_flight
            )
            self._first_arrivals += 1
            if self._first_arrivals == 2:
                self._both_first_attempts.set()
            try:
                await asyncio.wait_for(self._both_first_attempts.wait(), timeout=2.0)
                raise StructuredGenerationError(
                    kind=GenerationFailureKind.PROVIDER_UNAVAILABLE,
                    retryable=True,
                    safe_message=canary._INVALID_STREAM_ITEM_SAFE_MESSAGE,
                )
            finally:
                self._first_in_flight -= 1

        content = f"provider-free-{call_id}".encode("ascii")
        digest = hashlib.sha256(content).hexdigest()
        attempt_id = request.provider_attempt_id.value
        self._test_progress_sink(
            StructuredStreamProgress(
                call_id=call_id,
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
                call_id=call_id,
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
            provider_response_id=f"offline-{call_id}",
            finish_reason="tool_calls",
            input_tokens=4_321,
            output_tokens=321,
            reasoning_tokens=117,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0.003"),
            latency_ns=1_000_000,
        )


class _RetryingRunnerFactory:
    def __init__(self) -> None:
        self.generator: _RetryingOfflineBoilsGenerator | None = None

    def __call__(self, **kwargs: object) -> object:
        assert kwargs["api_key"] == "offline-secret-never-persist"
        config = kwargs["config"]
        self.generator = _RetryingOfflineBoilsGenerator(
            config=config,
            progress_sink=kwargs["progress_sink"],
            outbound_sink=kwargs["outbound_request_manifest_sink"],
        )
        return create_production_queued_runner(
            generator=self.generator,
            max_in_flight=config.max_connections,
            max_pending=config.max_pending,
            max_attempts=config.max_attempts,
            attempt_timeout_ns=None,
            base_backoff_ns=1,
            max_backoff_ns=1,
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


def test_two_call_live_retries_join_and_cleanup_are_complete(tmp_path: Path) -> None:
    readiness = asyncio.run(
        canary._execute_readiness_for_testing(
            "retry_readiness",
            run_root=tmp_path,
            dependencies=_dependencies(
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

    factory = _RetryingRunnerFactory()
    summary = asyncio.run(
        canary._execute_live_for_testing(
            "retry_live",
            readiness_dir=Path(readiness["run_dir"]),
            authorization=canary.LIVE_AUTHORIZATION,
            run_root=tmp_path,
            dependencies=canary.CanaryDependencies(
                inputs_factory=canary.build_canary_inputs,
                credential_loader=credential,
                runner_factory=factory,
            ),
        )
    )

    assert credential_calls == 1
    assert factory.generator is not None
    assert factory.generator.first_attempt_peak == 2
    assert dict(factory.generator._attempts) == {
        canary._EXPECTED_CALL_IDS[0]: 2,
        canary._EXPECTED_CALL_IDS[1]: 2,
    }
    assert summary["failed"] is False
    result = summary["result"]
    assert result["status"] == "completed_offline_test_only"
    assert result["logical_call_count"] == 2
    assert result["logical_peak_in_flight"] == 2
    assert result["physical_attempt_count"] == 4
    assert len(result["provider_attempt_ids"]) == 4
    assert len(set(result["provider_attempt_ids"])) == 4
    assert [value["physical_attempt_count"] for value in result["calls"]] == [2, 2]
    assert all(
        len(value["selected_option_ids"]) == canary.campaign.PORTFOLIO_WIDTH
        for value in result["calls"]
    )
    assert all(value["response"]["reasoning_tokens"] == 117 for value in result["calls"])
    assert result["provider_attempt_join"]["join_valid"] is True
    assert result["queue_cleanup"]["empty_before_close"] is True
    assert result["queue_cleanup"]["closed_and_empty_after_close"] is True
    assert result["abc_executions"] == 0
    assert result["child_materialization_boundary"][
        "child_candidate_allocation_attempts"
    ] == 0
    assert result["evaluator_call_count"] == 0
    assert result["scientific_result_eligible"] is False
    assert result["optimization_result_eligible"] is False
    assert result["model_quality_result_eligible"] is False

    run_dir = Path(summary["run_dir"])
    expected_rows = {
        "provider_requests.jsonl": 2,
        "provider_attempt_requests.jsonl": 4,
        "provider_progress.jsonl": 4,
        "provider_outcomes.jsonl": 2,
        "provider_outputs.jsonl": 2,
        "provider_dispatch.jsonl": 4,
        "provider_boundary_events.jsonl": 18,
    }
    for name, count in expected_rows.items():
        assert len((run_dir / name).read_text(encoding="utf-8").splitlines()) == count
    dispatches = [
        json.loads(value)
        for value in (run_dir / "provider_dispatch.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [value["phase"] for value in dispatches[:2]] == ["started", "started"]
    assert dispatches[1]["logical_peak_in_flight"] == 2
    boundary_join = result["boundary_event_join"]
    assert boundary_join["evidence_join_valid"] is True
    assert boundary_join[
        "both_first_http_outbound_before_any_progress_or_terminal_outcome"
    ] is True
    assert boundary_join["strict_sequence_order"] is True
    assert boundary_join["strict_monotonic_ns_order"] is True
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
    assert canary.verify_finalized_run_directory(run_dir)["status"] == (
        "completed_offline_test_only"
    )
    launch = json.loads((run_dir / "launch.json").read_text(encoding="utf-8"))
    path_binding = launch["bound_readiness"]["directory_binding"]
    assert path_binding == {
        "base": "injected_run_root",
        "relative_path": "retry_readiness",
        "absolute_path_persisted": False,
    }
    assert str(tmp_path) not in (run_dir / "launch.json").read_text(
        encoding="utf-8"
    )
    retained = b"".join(
        path.read_bytes() for path in run_dir.rglob("*") if path.is_file()
    )
    assert b"offline-secret-never-persist" not in retained


def _release_outcome() -> tuple[dict[str, object], dict[str, object]]:
    prompt_sha256 = "a" * 64
    logical = {"prompt_sha256": prompt_sha256}

    def evidence(number: int) -> dict[str, object]:
        return {
            "variant": "original",
            "prompt_sha256": prompt_sha256,
            "provider_attempt_id": f"attempt_release_{number:06d}",
        }

    outcome = {
        "attempts": [
            {
                "attempt_number": 1,
                "status": "retryable_failure",
                "will_retry": True,
                "error_type": "StructuredGenerationError",
                "request_evidence": evidence(1),
                "classification": {
                    "disposition": "retry",
                    "reason": "transient",
                },
                "failure": {
                    "kind": "provider_unavailable",
                    "retryable": True,
                    "safe_message": canary._INVALID_STREAM_ITEM_SAFE_MESSAGE,
                    "status_code": None,
                    "retry_after_seconds": None,
                    "stream_timeout_phase": None,
                    "output_failure_mode": None,
                    "validation_issues": [],
                    "provider_error_code": None,
                    "provider_error_envelope_sha256": None,
                    "exception_provenance": None,
                },
            },
            {
                "attempt_number": 2,
                "status": "succeeded",
                "will_retry": False,
                "error_type": None,
                "request_evidence": evidence(2),
                "classification": None,
                "failure": None,
            },
        ]
    }
    return outcome, logical


def test_release_retry_gate_rejects_every_non_boundary_retry() -> None:
    outcome, logical = _release_outcome()
    assert canary._require_release_attempt_shape(
        call_id=canary._EXPECTED_CALL_IDS[0],
        outcome=outcome,
        logical_request=logical,
        progress_rows=(),
    ) == ("attempt_release_000001", "attempt_release_000002")

    hostile: list[dict[str, object]] = []
    opaque_400 = deepcopy(outcome)
    opaque_400["attempts"][0]["failure"]["status_code"] = 400
    hostile.append(opaque_400)
    generic_transport = deepcopy(outcome)
    generic_transport["attempts"][0]["failure"]["safe_message"] = (
        "provider transport unavailable"
    )
    hostile.append(generic_transport)
    other_retry = deepcopy(outcome)
    other_retry["attempts"][0]["failure"]["kind"] = "timeout"
    other_retry["attempts"][0]["classification"]["reason"] = "timeout"
    hostile.append(other_retry)
    repaired_payload = deepcopy(outcome)
    repaired_payload["attempts"][1]["request_evidence"]["variant"] = (
        "schema_repair_v2"
    )
    hostile.append(repaired_payload)
    attempt_three = deepcopy(outcome)
    third = deepcopy(attempt_three["attempts"][1])
    third["attempt_number"] = 3
    third["request_evidence"]["provider_attempt_id"] = "attempt_release_000003"
    attempt_three["attempts"].append(third)
    hostile.append(attempt_three)

    for value in hostile:
        with pytest.raises(canary.BoilsG1TwoCallCanaryError):
            canary._require_release_attempt_shape(
                call_id=canary._EXPECTED_CALL_IDS[0],
                outcome=value,
                logical_request=logical,
                progress_rows=(),
            )
    with pytest.raises(canary.BoilsG1TwoCallCanaryError, match="progress"):
        canary._require_release_attempt_shape(
            call_id=canary._EXPECTED_CALL_IDS[0],
            outcome=outcome,
            logical_request=logical,
            progress_rows=(
                {
                    "call_id": canary._EXPECTED_CALL_IDS[0],
                    "provider_attempt_id": "attempt_release_000001",
                },
            ),
        )


def test_boundary_order_rejects_progress_before_both_first_http_hooks() -> None:
    rows = [
        canary._boundary_event_record(
            sequence=1,
            monotonic_ns=1,
            boundary="http_outbound_hook",
            call_id=canary._EXPECTED_CALL_IDS[0],
            provider_attempt_id="attempt_boundary_000001",
        ),
        canary._boundary_event_record(
            sequence=2,
            monotonic_ns=2,
            boundary="stream_progress",
            call_id=canary._EXPECTED_CALL_IDS[0],
            provider_attempt_id="attempt_boundary_000001",
        ),
        canary._boundary_event_record(
            sequence=3,
            monotonic_ns=3,
            boundary="http_outbound_hook",
            call_id=canary._EXPECTED_CALL_IDS[1],
            provider_attempt_id="attempt_boundary_000002",
        ),
    ]
    with pytest.raises(canary.BoilsG1TwoCallCanaryError, match="preceded"):
        canary._validate_boundary_event_order(rows)
    reordered_clock = deepcopy(rows)
    reordered_clock[1] = canary._boundary_event_record(
        sequence=2,
        monotonic_ns=1,
        boundary="stream_progress",
        call_id=canary._EXPECTED_CALL_IDS[0],
        provider_attempt_id="attempt_boundary_000001",
    )
    with pytest.raises(canary.BoilsG1TwoCallCanaryError, match="monotonic"):
        canary._validate_boundary_event_order(reordered_clock)


def test_production_runner_constructor_monkeypatch_fails_closed(monkeypatch) -> None:
    def fake_runner(**_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(
        canary._progress_aware_openrouter_module,
        "create_progress_aware_openrouter_runner",
        fake_runner,
    )
    with pytest.raises(canary.BoilsG1TwoCallCanaryError, match="binding drifted"):
        canary._production_dependencies()


def test_live_request_commitment_rejects_each_wire_field_drift() -> None:
    lane = {
        "call_id": canary._EXPECTED_CALL_IDS[0],
        "prompt_sha256": "a" * 64,
        "prompt_utf8_bytes": 100,
        "schema": {
            "logical_schema_sha256": "b" * 64,
            "logical_schema_utf8_bytes": 200,
        },
        "output_tool_name": "propose_calibrated_portfolio_slate",
        "temperature_hex": float(0.2).hex(),
        "max_output_tokens": 384_000,
    }
    request = {
        "call_id": lane["call_id"],
        "prompt_sha256": lane["prompt_sha256"],
        "wire_prompt_sha256": lane["prompt_sha256"],
        "prompt_utf8_bytes": lane["prompt_utf8_bytes"],
        "output_schema_sha256": lane["schema"]["logical_schema_sha256"],
        "output_schema_utf8_bytes": lane["schema"]["logical_schema_utf8_bytes"],
        "output_tool_name": lane["output_tool_name"],
        "temperature_hex": lane["temperature_hex"],
        "max_output_tokens": lane["max_output_tokens"],
    }
    canary._require_request_matches_readiness_lane(request, lane)
    for name, replacement in (
        ("prompt_sha256", "c" * 64),
        ("wire_prompt_sha256", "c" * 64),
        ("prompt_utf8_bytes", 101),
        ("output_schema_sha256", "d" * 64),
        ("output_schema_utf8_bytes", 201),
        ("output_tool_name", "fake_tool"),
        ("temperature_hex", float(0.3).hex()),
        ("max_output_tokens", 1),
    ):
        drifted = dict(request)
        drifted[name] = replacement
        with pytest.raises(canary.BoilsG1TwoCallCanaryError, match="commitment"):
            canary._require_request_matches_readiness_lane(drifted, lane)


def test_provider_response_and_exact_k8_audit_join_fail_closed() -> None:
    members = [{"option_id": f"option_{index}"} for index in range(8)]
    output = {
        "provider_response_id": "response_1",
        "typed_output": {"members": members},
    }
    response = {
        "provider_response_id": "response_1",
        "finish_reason": "tool_calls",
    }
    audit = {
        "members": [
            {"model_rank": index, **deepcopy(member)}
            for index, member in enumerate(members, start=1)
        ]
    }
    assert len(
        canary._require_k8_output_audit_join(
            output=output, response=response, original_audit=audit
        )
    ) == 8
    mismatched_response = dict(response)
    mismatched_response["provider_response_id"] = "response_2"
    with pytest.raises(canary.BoilsG1TwoCallCanaryError, match="do not join"):
        canary._require_k8_output_audit_join(
            output=output,
            response=mismatched_response,
            original_audit=audit,
        )
    mismatched_audit = deepcopy(audit)
    mismatched_audit["members"][0]["option_id"] = "foreign_option"
    with pytest.raises(canary.BoilsG1TwoCallCanaryError, match="do not join"):
        canary._require_k8_output_audit_join(
            output=output,
            response=response,
            original_audit=mismatched_audit,
        )


def test_selector_only_materialization_guard_is_observed() -> None:
    guard = canary._SelectorOnlyIdFactoryGuard(object())
    with pytest.raises(canary.BoilsG1TwoCallCanaryError, match="materialization"):
        guard.new_candidate_id()
    assert guard.to_record()["child_candidate_allocation_attempts"] == 1
