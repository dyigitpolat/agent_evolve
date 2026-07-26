#!/usr/bin/env python3
"""One-call live OpenRouter streaming-conformance harness.

``readiness`` executes every source, route, schema, request, queue-policy, and
artifact gate without reading credentials or constructing a provider client.
``live`` is the only mode that reads ``OPENROUTER_API_KEY`` and dispatches the
single logical typed call (with bounded exact-payload transport retries).
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, Literal, Protocol
import xml.etree.ElementTree as ElementTree


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from pydantic import BaseModel, ConfigDict  # noqa: E402

from agent_evolve.application import llm_task_queue as queue_application  # noqa: E402
from agent_evolve.domain.ids import LLMCallId, ProviderAttemptId  # noqa: E402
from agent_evolve.integrations.pydantic_ai import async_generator  # noqa: E402
from agent_evolve.integrations.pydantic_ai import progress_aware_openrouter  # noqa: E402
from agent_evolve.integrations.pydantic_ai import queued_runner  # noqa: E402
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    AttemptedStructuredGenerationResponse,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (  # noqa: E402
    OpenRouterReasoningConfig,
    STREAM_CONTENT_IDENTITY_ALGORITHM,
    STREAM_CONTENT_IDENTITY_DOMAIN_SHA256,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (  # noqa: E402
    ProgressAwareOpenRouterConfig,
    create_progress_aware_openrouter_runner,
)
from agent_evolve.ports import structured_generator  # noqa: E402
from agent_evolve.ports.artifact_store import (  # noqa: E402
    canonical_json_bytes,
    decode_json_bytes,
)
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    StructuredStreamChannel,
    StructuredStreamLivenessPolicy,
    StructuredStreamProgress,
    StructuredStreamProgressKind,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    BatchedDurableJsonlJournal,
    DurableJsonlJournal,
    file_identity,
    finalize_run_directory,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)
from examples.development import durable_run_artifacts  # noqa: E402
from agent_evolve.infrastructure import stream_liveness  # noqa: E402


ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
DEFAULT_RUN_ROOT = ARTIFACT_ROOT / "experiment_logs" / "openrouter_conformance"
CAPABILITY_SNAPSHOT = (
    ARTIFACT_ROOT
    / "data"
    / "openrouter_deepseek_v4_pro_streamlake_capability_snapshot_20260714.json"
)
PRICING_SNAPSHOT = (
    ARTIFACT_ROOT
    / "data"
    / "openrouter_deepseek_v4_pro_streamlake_pricing_snapshot_20260714.json"
)
CAPABILITY_SNAPSHOT_SHA256 = (
    "131d0fef27cb24350f9c067ea7407cd9279ddbe242eef77e29451390a750a671"
)
PRICING_SNAPSHOT_SHA256 = (
    "5adea5e08d7aea5eb89de010e1750890fe6b7f70a3f7fe733a08996d0b8b7204"
)
PREREGISTRATION = (
    ARTIFACT_ROOT
    / "125_progress_aware_openrouter_streaming_conformance_v2_preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "257f19cc9b0e0f5735c197bcd7427e2ca002cf04b609595b83d51da9a1f5ac6b"
)
FROZEN_LIVE_RUN_ID = "openrouter_stream_conformance_live_v2_20260715"

MODEL = "deepseek/deepseek-v4-pro"
CANONICAL_MODEL = "deepseek/deepseek-v4-pro-20260423"
ALLOWED_RESOLVED_MODELS = (MODEL, CANONICAL_MODEL)
PROVIDER_SLUG = "streamlake"
RESOLVED_PROVIDER = "StreamLake"
MAX_OUTPUT_TOKENS = 384_000
REASONING_MAX_TOKENS = 4_096
CONNECT_TIMEOUT_SECONDS = 90.0
DEFAULT_FIRST_EVENT_TIMEOUT_SECONDS = 180
DEFAULT_IDLE_TIMEOUT_SECONDS = 120
DEFAULT_ABSOLUTE_TIMEOUT_SECONDS = 0
MAX_ATTEMPTS = 2
BASE_BACKOFF_NS = 1_000_000_000
MAX_BACKOFF_NS = 30_000_000_000
JITTER_SEED = 2_026_071_500
JITTER_DOMAIN = "openrouter-streaming-conformance-v1"
PROGRESS_MAX_UNFSYNCED_ROWS = 64
CONFORMANCE_NONCE = "ae_streaming_conformance_nonce_v2_20260715"
ACKNOWLEDGEMENT = "deepseek_v4_pro_streamlake_typed_stream_complete_v2"
OUTPUT_TOOL_NAME = "return_streaming_conformance"
OPERATION = "openrouter_streaming_conformance"
MANIFEST_FRAMING = b"agent-evolve:openrouter-streaming-conformance-manifest:v2\x00"
RELEASE_GATE_FRAMING = (
    b"agent-evolve:openrouter-streaming-conformance-release-gate:v2\x00"
)
_SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")

FOCUSED_TEST_RELATIVE_PATHS = (
    "tests/test_stream_liveness.py",
    "tests/test_async_structured_generator.py",
    "tests/test_llm_task_queue.py",
    "tests/test_queued_structured_runner.py",
    "tests/test_generation_failure_classification.py",
    "tests/test_progress_aware_openrouter.py",
    "tests/test_openrouter_streaming_conformance.py",
)
FOCUSED_PYTEST_ARGUMENTS = (
    "-o",
    "addopts=",
    "-q",
    *FOCUSED_TEST_RELATIVE_PATHS,
)

PROMPT = f"""You are executing a transport conformance check, not an optimization task.
Call the {OUTPUT_TOOL_NAME} output tool exactly once with both fields below and no
additional fields:
- nonce: {CONFORMANCE_NONCE}
- acknowledgement: {ACKNOWLEDGEMENT}
Do not place commentary outside the typed tool call.
"""


class StreamingConformanceOutput(BaseModel):
    """Exact typed nonce returned by the one logical call."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    nonce: Literal["ae_streaming_conformance_nonce_v2_20260715"]
    acknowledgement: Literal[
        "deepseek_v4_pro_streamlake_typed_stream_complete_v2"
    ]


class Runner(Protocol):
    async def __aenter__(self) -> "Runner": ...

    async def __aexit__(self, *_: object) -> None: ...

    async def __call__(
        self,
        request: StructuredGenerationRequest[StreamingConformanceOutput],
    ) -> AttemptedStructuredGenerationResponse[StreamingConformanceOutput]: ...


RunnerFactory = Callable[..., Runner]


@dataclass(frozen=True, slots=True)
class ConformanceDependencies:
    credential_loader: Callable[[], str]
    runner_factory: RunnerFactory


class ConformanceRunError(RuntimeError):
    """Sanitized harness failure; durable result contains only closed evidence."""


def _environment_credential() -> str:
    value = os.environ.get("OPENROUTER_API_KEY")
    if type(value) is not str or not value:
        raise ConformanceRunError("OPENROUTER_API_KEY is unavailable")
    return value


DEFAULT_DEPENDENCIES = ConformanceDependencies(
    credential_loader=_environment_credential,
    runner_factory=create_progress_aware_openrouter_runner,
)


def _sha256(value: object, framing: bytes) -> str:
    return hashlib.sha256(framing + canonical_json_bytes(value)).hexdigest()


def _schema_contract() -> dict[str, object]:
    schema = StreamingConformanceOutput.model_json_schema()
    return {
        "output_type": "StreamingConformanceOutput",
        "output_tool_name": OUTPUT_TOOL_NAME,
        "schema_sha256": hashlib.sha256(canonical_json_bytes(schema)).hexdigest(),
        "typed_nonce": CONFORMANCE_NONCE,
        "typed_nonce_sha256": hashlib.sha256(CONFORMANCE_NONCE.encode()).hexdigest(),
        "acknowledgement": ACKNOWLEDGEMENT,
        "exact_extra_policy": "forbid",
        "strict": True,
    }


def _stream_lifecycle_contract() -> dict[str, object]:
    return {
        "pydantic_final_result_event_projection": "output_selected",
        "output_selected_is_terminal": False,
        "local_typed_completion_kind": "stream_completed",
        "successful_attempt_completion_count": 1,
        "successful_attempt_completion_position": "last",
    }


def build_request(run_id: str) -> StructuredGenerationRequest[StreamingConformanceOutput]:
    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run ID violates the closed conformance grammar")
    digest = hashlib.sha256(run_id.encode("ascii")).hexdigest()[:32]
    return StructuredGenerationRequest(
        call_id=LLMCallId(f"call_stream_conformance_{digest}"),
        operation=OPERATION,
        prompt=PROMPT,
        output_type=StreamingConformanceOutput,
        output_tool_name=OUTPUT_TOOL_NAME,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=None,
    )


def _request_contract(run_id: str) -> dict[str, object]:
    request = build_request(run_id)
    return {
        "logical_call_count": 1,
        "call_id": request.call_id.value,
        "operation": request.operation,
        "prompt_utf8_bytes": len(request.prompt.encode("utf-8")),
        "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
        "output_tool_name": request.output_tool_name,
        "max_output_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "provider_attempt_id_before_queue": None,
        "schema": _schema_contract(),
    }


def _source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__),
        Path(__file__).with_name(
            "record_openrouter_streaming_conformance_release_gate.py"
        ),
        Path(durable_run_artifacts.__file__),
        Path(progress_aware_openrouter.__file__),
        Path(async_generator.__file__),
        Path(queued_runner.__file__),
        Path(stream_liveness.__file__),
        Path(structured_generator.__file__),
        Path(queue_application.__file__),
    )


def _source_identity() -> dict[str, object]:
    return source_identity(_source_paths(), relative_to=WORKSPACE_ROOT)


def _focused_test_source_identity() -> dict[str, object]:
    return source_identity(
        tuple(AGENT_EVOLVE_ROOT / path for path in FOCUSED_TEST_RELATIVE_PATHS),
        relative_to=WORKSPACE_ROOT,
    )


def _preregistration_identity() -> dict[str, object]:
    record = file_identity(PREREGISTRATION, relative_to=WORKSPACE_ROOT)
    if record.get("sha256") != PREREGISTRATION_SHA256:
        raise ConformanceRunError("frozen conformance preregistration changed")
    return record


def _validate_frozen_config(config: ProgressAwareOpenRouterConfig) -> None:
    expected = build_config(
        first_event_timeout_seconds=DEFAULT_FIRST_EVENT_TIMEOUT_SECONDS,
        idle_timeout_seconds=DEFAULT_IDLE_TIMEOUT_SECONDS,
        absolute_timeout_seconds=DEFAULT_ABSOLUTE_TIMEOUT_SECONDS,
    )
    if config.to_manifest_record() != expected.to_manifest_record():
        raise ConformanceRunError("configuration violates the frozen protocol")


def _junit_counts(path: Path) -> dict[str, int]:
    content = path.expanduser().resolve(strict=True).read_bytes()
    if not content or len(content) > 10_000_000:
        raise ConformanceRunError("focused pytest report size is invalid")
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as error:
        raise ConformanceRunError("focused pytest report is invalid XML") from error

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    if local_name(root.tag) == "testsuite":
        suites = [root]
    elif local_name(root.tag) == "testsuites":
        suites = [child for child in root if local_name(child.tag) == "testsuite"]
    else:
        suites = []
    if not suites:
        raise ConformanceRunError("focused pytest report has no test suite")

    counts = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for name in counts:
            value = suite.get(name)
            if type(value) is not str or not value.isascii() or not value.isdecimal():
                raise ConformanceRunError("focused pytest counts are invalid")
            counts[name] += int(value)
    testcase_count = sum(
        1
        for suite in suites
        for element in suite.iter()
        if local_name(element.tag) == "testcase"
    )
    if testcase_count != counts["tests"]:
        raise ConformanceRunError("focused pytest case count is inconsistent")
    return counts


def build_provider_free_release_gate(
    *,
    config: ProgressAwareOpenRouterConfig,
    junit_report_path: Path,
    pytest_exit_code: int,
    source_identity_before: Mapping[str, object],
    focused_test_source_identity_before: Mapping[str, object],
    stdout: bytes = b"",
    stderr: bytes = b"",
) -> dict[str, object]:
    """Build a committed release record from an external provider-free test run."""

    _validate_frozen_config(config)
    if type(pytest_exit_code) is not int or pytest_exit_code != 0:
        raise ConformanceRunError("focused pytest execution did not pass")
    if type(stdout) is not bytes or type(stderr) is not bytes:
        raise TypeError("captured pytest output must be exact bytes")
    counts = _junit_counts(junit_report_path)
    if (
        counts["tests"] < 1
        or counts["failures"] != 0
        or counts["errors"] != 0
        or counts["skipped"] != 0
    ):
        raise ConformanceRunError("focused pytest report is not an all-pass run")
    current_source = _source_identity()
    current_test_source = _focused_test_source_identity()
    if (
        source_identity_before != current_source
        or focused_test_source_identity_before != current_test_source
    ):
        raise ConformanceRunError("source identity changed during focused tests")
    report_path = junit_report_path.expanduser().resolve(strict=True)
    record: dict[str, object] = {
        "schema_version": 2,
        "kind": "openrouter_streaming_conformance_provider_free_release_gate",
        "status": "provider_free_release_gate_passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_live_run_id": FROZEN_LIVE_RUN_ID,
        "provider_call_performed": False,
        "credentials_read": False,
        "scientific_result_eligible": False,
        "protocol": _preregistration_identity(),
        "target_request": _request_contract(FROZEN_LIVE_RUN_ID),
        "route": route_binding(),
        "composition": config.to_manifest_record(),
        "stream_lifecycle": _stream_lifecycle_contract(),
        "source_identity": current_source,
        "focused_test_source_identity": current_test_source,
        "test_execution": {
            "runner": "python_-m_pytest",
            "arguments": list(FOCUSED_PYTEST_ARGUMENTS),
            "exit_code": pytest_exit_code,
            "counts": counts,
            "junit_report": file_identity(
                report_path,
                relative_to=report_path.parent,
            ),
            "stdout": {
                "size_bytes": len(stdout),
                "sha256": hashlib.sha256(stdout).hexdigest(),
                "plaintext_retained": False,
            },
            "stderr": {
                "size_bytes": len(stderr),
                "sha256": hashlib.sha256(stderr).hexdigest(),
                "plaintext_retained": False,
            },
            "source_identity_stable_during_execution": True,
        },
    }
    record["release_gate_commitment_sha256"] = _sha256(
        record,
        RELEASE_GATE_FRAMING,
    )
    return record


def verify_provider_free_release_gate(
    gate_dir: Path,
    *,
    config: ProgressAwareOpenRouterConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    """Verify a finalized external provider-free test gate against current code."""

    _validate_frozen_config(config)
    root = gate_dir.expanduser().resolve(strict=True)
    try:
        finalization = verify_finalized_run_directory(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ConformanceRunError("provider-free release gate is not finalized") from error
    if finalization.get("status") != "provider_free_release_gate_passed":
        raise ConformanceRunError("provider-free release gate did not pass")
    gate_path = root / "release_gate.json"
    report_path = root / "focused_tests.junit.xml"
    try:
        value = decode_json_bytes(gate_path.read_bytes())
    except (OSError, ValueError, TypeError) as error:
        raise ConformanceRunError("provider-free release record is unreadable") from error
    if type(value) is not dict:
        raise ConformanceRunError("provider-free release record is not an object")
    observed = dict(value)
    commitment = observed.pop("release_gate_commitment_sha256", None)
    if (
        type(commitment) is not str
        or _LOWER_SHA256.fullmatch(commitment) is None
        or commitment != _sha256(observed, RELEASE_GATE_FRAMING)
    ):
        raise ConformanceRunError("provider-free release commitment is invalid")
    counts = _junit_counts(report_path)
    execution = value.get("test_execution")
    if type(execution) is not dict:
        raise ConformanceRunError("provider-free test execution is missing")
    if (
        value.get("schema_version") != 2
        or value.get("kind")
        != "openrouter_streaming_conformance_provider_free_release_gate"
        or value.get("status") != "provider_free_release_gate_passed"
        or value.get("target_live_run_id") != FROZEN_LIVE_RUN_ID
        or value.get("provider_call_performed") is not False
        or value.get("credentials_read") is not False
        or value.get("scientific_result_eligible") is not False
        or value.get("protocol") != _preregistration_identity()
        or value.get("target_request") != _request_contract(FROZEN_LIVE_RUN_ID)
        or value.get("route") != route_binding()
        or value.get("composition") != config.to_manifest_record()
        or value.get("stream_lifecycle") != _stream_lifecycle_contract()
        or value.get("source_identity") != _source_identity()
        or value.get("focused_test_source_identity")
        != _focused_test_source_identity()
        or execution.get("runner") != "python_-m_pytest"
        or execution.get("arguments") != list(FOCUSED_PYTEST_ARGUMENTS)
        or execution.get("exit_code") != 0
        or execution.get("counts") != counts
        or execution.get("junit_report")
        != file_identity(report_path, relative_to=root)
        or execution.get("source_identity_stable_during_execution") is not True
        or counts["tests"] < 1
        or counts["failures"] != 0
        or counts["errors"] != 0
        or counts["skipped"] != 0
    ):
        raise ConformanceRunError("provider-free release gate semantics changed")
    return value, finalization


def _release_gate_binding(
    gate_dir: Path,
    *,
    release_gate: Mapping[str, object],
    finalization: Mapping[str, object],
) -> dict[str, object]:
    root = gate_dir.expanduser().resolve(strict=True)
    return {
        "external_gate_dir": str(root),
        "release_gate_record": file_identity(root / "release_gate.json"),
        "junit_report": file_identity(root / "focused_tests.junit.xml"),
        "finalization_record": file_identity(root / "finalized.json"),
        "finalization_sha256": finalization.get("finalization_sha256"),
        "release_gate_commitment_sha256": release_gate.get(
            "release_gate_commitment_sha256"
        ),
        "source_identity_aggregate_sha256": release_gate.get(
            "source_identity", {}
        ).get("aggregate_sha256"),
        "focused_test_source_identity_aggregate_sha256": release_gate.get(
            "focused_test_source_identity", {}
        ).get("aggregate_sha256"),
    }


def _snapshot(path: Path, expected_sha256: str) -> Mapping[str, object]:
    content = path.expanduser().resolve(strict=True).read_bytes()
    observed = hashlib.sha256(content).hexdigest()
    if observed != expected_sha256:
        raise ConformanceRunError("frozen route snapshot identity changed")
    value = decode_json_bytes(content)
    if type(value) is not dict:
        raise ConformanceRunError("frozen route snapshot is not an object")
    return value


def route_binding() -> dict[str, object]:
    capability = _snapshot(CAPABILITY_SNAPSHOT, CAPABILITY_SNAPSHOT_SHA256)
    pricing = _snapshot(PRICING_SNAPSHOT, PRICING_SNAPSHOT_SHA256)
    endpoint = capability.get("selected_endpoint")
    pricing_model = pricing.get("model")
    pricing_endpoint = pricing.get("selected_endpoint")
    if (
        capability.get("requested_model_alias") != MODEL
        or capability.get("canonical_model_slug") != CANONICAL_MODEL
        or type(endpoint) is not dict
        or endpoint.get("provider_name") != RESOLVED_PROVIDER
        or endpoint.get("provider_request_slug") != PROVIDER_SLUG
        or endpoint.get("max_completion_tokens") != MAX_OUTPUT_TOKENS
        or "tools" not in endpoint.get("supported_parameters", [])
        or type(pricing_model) is not dict
        or pricing_model.get("requested_slug") != MODEL
        or pricing_model.get("canonical_slug") != CANONICAL_MODEL
        or pricing_model.get("max_completion_tokens") != MAX_OUTPUT_TOKENS
        or type(pricing_endpoint) is not dict
        or pricing_endpoint.get("provider_name") != RESOLVED_PROVIDER
        or pricing_endpoint.get("provider_request_slug") != PROVIDER_SLUG
    ):
        raise ConformanceRunError("frozen StreamLake route semantics changed")
    return {
        "requested_model": MODEL,
        "canonical_model": CANONICAL_MODEL,
        "allowed_resolved_models": list(ALLOWED_RESOLVED_MODELS),
        "provider_name": RESOLVED_PROVIDER,
        "provider_request_slug": PROVIDER_SLUG,
        "provider_options": {
            "only": [PROVIDER_SLUG],
            "allow_fallbacks": False,
        },
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "capability_snapshot_sha256": CAPABILITY_SNAPSHOT_SHA256,
        "pricing_snapshot_sha256": PRICING_SNAPSHOT_SHA256,
    }


def build_config(
    *,
    first_event_timeout_seconds: int,
    idle_timeout_seconds: int,
    absolute_timeout_seconds: int,
) -> ProgressAwareOpenRouterConfig:
    for name, value in (
        ("first_event_timeout_seconds", first_event_timeout_seconds),
        ("idle_timeout_seconds", idle_timeout_seconds),
    ):
        if type(value) is not int or not 1 <= value <= 3_600:
            raise ValueError(f"{name} must lie in [1,3600]")
    if (
        type(absolute_timeout_seconds) is not int
        or not 0 <= absolute_timeout_seconds <= 86_400
    ):
        raise ValueError("absolute_timeout_seconds must lie in [0,86400]")
    policy = StructuredStreamLivenessPolicy(
        first_event_timeout_ns=first_event_timeout_seconds * 1_000_000_000,
        idle_timeout_ns=idle_timeout_seconds * 1_000_000_000,
        absolute_timeout_ns=(
            None
            if absolute_timeout_seconds == 0
            else absolute_timeout_seconds * 1_000_000_000
        ),
    )
    return ProgressAwareOpenRouterConfig(
        model_name=MODEL,
        provider_only=(PROVIDER_SLUG,),
        connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS,
        stream_liveness_policy=policy,
        max_connections=1,
        max_pending=1,
        max_attempts=MAX_ATTEMPTS,
        base_backoff_ns=BASE_BACKOFF_NS,
        max_backoff_ns=MAX_BACKOFF_NS,
        jitter_seed=JITTER_SEED,
        jitter_domain=JITTER_DOMAIN,
        app_title="AgentEvolve OpenRouter streaming conformance",
        reasoning_config=OpenRouterReasoningConfig(max_tokens=REASONING_MAX_TOKENS),
    )


def build_manifest(
    *,
    mode: Literal["readiness", "live"],
    run_dir: Path,
    config: ProgressAwareOpenRouterConfig,
    release_gate_dir: Path,
    release_gate: Mapping[str, object],
    release_gate_finalization: Mapping[str, object],
) -> dict[str, object]:
    run_id = run_dir.name
    request = _request_contract(run_id)
    record: dict[str, object] = {
        "schema_version": 2,
        "kind": "openrouter_streaming_conformance",
        "mode": mode,
        "run_id": run_id,
        "run_dir": str(run_dir.expanduser().resolve(strict=False)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_dispatch_authorized": mode == "live",
        "credentials_read_during_manifest_build": False,
        "credential_variable": "OPENROUTER_API_KEY",
        "request": request,
        "route": route_binding(),
        "composition": config.to_manifest_record(),
        "protocol": _preregistration_identity(),
        "provider_free_release_gate": {
            **_release_gate_binding(
                release_gate_dir,
                release_gate=release_gate,
                finalization=release_gate_finalization,
            ),
            "bound_copy": file_identity(run_dir / "provider_free_release_gate.json"),
        },
        "stream_content_identity": {
            "algorithm": STREAM_CONTENT_IDENTITY_ALGORITHM,
            "domain_sha256": STREAM_CONTENT_IDENTITY_DOMAIN_SHA256,
            "scope": "pydantic_semantic_utf8_fragments_not_sse_wire_bytes",
            "plaintext_retained": False,
            "unsupported_dictionary_tool_fragments": "fail_closed",
        },
        "stream_lifecycle": _stream_lifecycle_contract(),
        "progress_journal": {
            "path": "stream_progress.jsonl",
            "content_blind": True,
            "max_unfsynced_rows": PROGRESS_MAX_UNFSYNCED_ROWS,
            "terminal_outcome_barrier": (
                "flush_and_fsync_progress_before_outcome_append_and_fsync"
            ),
        },
        "outcome_journal": {
            "path": "queue_outcomes.jsonl",
            "publication": "required_fsync_before_response_escape",
        },
        "partial_output_policy": {
            "persist_partial_content": False,
            "scientific_result_eligible": False,
        },
        "source_identity": _source_identity(),
    }
    record["manifest_commitment_sha256"] = _sha256(record, MANIFEST_FRAMING)
    return record


def verify_manifest(
    record: Mapping[str, object],
    *,
    mode: Literal["readiness", "live"],
    run_dir: Path,
    config: ProgressAwareOpenRouterConfig,
    release_gate_dir: Path,
    release_gate: Mapping[str, object],
    release_gate_finalization: Mapping[str, object],
) -> None:
    if type(record) is not dict:
        raise ConformanceRunError("manifest is not an exact object")
    observed = dict(record)
    commitment = observed.pop("manifest_commitment_sha256", None)
    if (
        type(commitment) is not str
        or _LOWER_SHA256.fullmatch(commitment) is None
        or commitment != _sha256(observed, MANIFEST_FRAMING)
    ):
        raise ConformanceRunError("manifest commitment is invalid")
    expected_request = _request_contract(run_dir.name)
    if (
        record.get("schema_version") != 2
        or record.get("kind") != "openrouter_streaming_conformance"
        or record.get("mode") != mode
        or record.get("run_id") != run_dir.name
        or record.get("request") != expected_request
        or record.get("route") != route_binding()
        or record.get("composition") != config.to_manifest_record()
        or record.get("stream_lifecycle") != _stream_lifecycle_contract()
        or record.get("protocol") != _preregistration_identity()
        or record.get("provider_free_release_gate")
        != {
            **_release_gate_binding(
                release_gate_dir,
                release_gate=release_gate,
                finalization=release_gate_finalization,
            ),
            "bound_copy": file_identity(
                run_dir / "provider_free_release_gate.json"
            ),
        }
        or record.get("source_identity") != _source_identity()
        or record.get("credentials_read_during_manifest_build") is not False
    ):
        raise ConformanceRunError("manifest gate changed before dispatch")


class ProgressRecorder:
    """Validate and batch-persist content-blind progress observations."""

    def __init__(self, journal: BatchedDurableJsonlJournal) -> None:
        self._journal = journal
        self._lock = threading.Lock()
        self.rows: list[dict[str, object]] = []
        self._last_by_attempt: dict[str, tuple[int, int, int]] = {}
        self._sealed = False

    def __call__(self, progress: StructuredStreamProgress) -> None:
        if type(progress) is not StructuredStreamProgress:
            raise TypeError("progress must be an exact StructuredStreamProgress")
        progress.__post_init__()
        attempt_id = progress.provider_attempt_id
        if attempt_id is None:
            raise ConformanceRunError("stream progress lacks physical attempt identity")
        with self._lock:
            if self._sealed:
                raise ConformanceRunError(
                    "stream progress arrived after terminal outcome publication"
                )
            previous = self._last_by_attempt.get(attempt_id)
            expected_sequence = 1 if previous is None else previous[0] + 1
            previous_cumulative = 0 if previous is None else previous[1]
            previous_elapsed = -1 if previous is None else previous[2]
            if (
                progress.sequence != expected_sequence
                or progress.cumulative_content_utf8_bytes
                != previous_cumulative + progress.event_content_utf8_bytes
                or progress.elapsed_ns < previous_elapsed
            ):
                raise ConformanceRunError("stream progress sequence is inconsistent")
            row = {
                "schema_version": 2,
                "call_id": progress.call_id,
                "provider_attempt_id": attempt_id,
                "sequence": progress.sequence,
                "kind": progress.kind.value,
                "channel": progress.channel.value,
                "elapsed_ns": progress.elapsed_ns,
                "event_content_utf8_bytes": progress.event_content_utf8_bytes,
                "cumulative_content_utf8_bytes": (
                    progress.cumulative_content_utf8_bytes
                ),
                "rolling_content_sha256": progress.rolling_content_sha256,
            }
            self._journal.append(row)
            self.rows.append(row)
            self._last_by_attempt[attempt_id] = (
                progress.sequence,
                progress.cumulative_content_utf8_bytes,
                progress.elapsed_ns,
            )

    def flush(self) -> None:
        self._journal.flush()

    def seal_and_flush(self) -> None:
        """Atomically forbid later progress, then cross the durability barrier."""

        with self._lock:
            self._sealed = True
            self._journal.flush()


def _progress_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        attempt_id = row.get("provider_attempt_id")
        if type(attempt_id) is not str:
            raise ConformanceRunError("progress row lacks provider attempt identity")
        grouped.setdefault(attempt_id, []).append(row)
    attempts: list[dict[str, object]] = []
    for attempt_id, attempt_rows in grouped.items():
        elapsed = [row.get("elapsed_ns") for row in attempt_rows]
        if any(type(value) is not int or value < 0 for value in elapsed):
            raise ConformanceRunError("progress row has invalid elapsed time")
        gaps = [later - earlier for earlier, later in zip(elapsed, elapsed[1:])]
        attempts.append(
            {
                "provider_attempt_id": attempt_id,
                "event_count": len(attempt_rows),
                "first_event_elapsed_ns": elapsed[0],
                "last_progress_elapsed_ns": elapsed[-1],
                "maximum_inter_event_gap_ns": max(gaps, default=0),
                "cumulative_content_utf8_bytes": attempt_rows[-1].get(
                    "cumulative_content_utf8_bytes"
                ),
                "rolling_content_sha256": attempt_rows[-1].get(
                    "rolling_content_sha256"
                ),
                "terminal_progress_kind": attempt_rows[-1].get("kind"),
            }
        )
    return {
        "physical_attempt_count_with_progress": len(attempts),
        "attempts": attempts,
    }


def validate_completed_call(
    attempted: AttemptedStructuredGenerationResponse[StreamingConformanceOutput],
    *,
    expected_call_id: str,
    outcome_rows: Sequence[Mapping[str, object]],
    progress_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    try:
        LLMCallId(expected_call_id)
    except (TypeError, ValueError) as error:
        raise ConformanceRunError("expected logical call identity is invalid") from error
    if type(attempted) is not AttemptedStructuredGenerationResponse:
        raise ConformanceRunError("runner returned an invalid attempted response")
    response = attempted.response
    if type(response) is not StructuredGenerationResponse:
        raise ConformanceRunError("runner returned an invalid structured response")
    StructuredGenerationResponse.__post_init__(response)
    if (
        type(response.value) is not StreamingConformanceOutput
        or response.value.nonce != CONFORMANCE_NONCE
        or response.value.acknowledgement != ACKNOWLEDGEMENT
    ):
        raise ConformanceRunError("typed conformance nonce is missing or invalid")
    if (
        response.requested_model != MODEL
        or response.resolved_model not in ALLOWED_RESOLVED_MODELS
        or response.resolved_provider != RESOLVED_PROVIDER
    ):
        raise ConformanceRunError("resolved route/model violates the frozen policy")
    if (
        response.provider_response_id is None
        or response.finish_reason is None
        or response.input_tokens <= 0
        or response.output_tokens <= 0
        or response.cost_usd is None
    ):
        raise ConformanceRunError("required provider telemetry is missing")
    if len(outcome_rows) != 1:
        raise ConformanceRunError("required terminal queue outcome is missing")
    outcome = outcome_rows[0]
    attempts = outcome.get("attempts")
    outcome_response = outcome.get("response")
    expected_outcome_response = {
        "requested_model": response.requested_model,
        "resolved_model": response.resolved_model,
        "resolved_provider": response.resolved_provider,
        "provider_response_id": response.provider_response_id,
        "finish_reason": response.finish_reason,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "reasoning_tokens": response.reasoning_tokens,
        "cache_read_tokens": response.cache_read_tokens,
        "cache_write_tokens": response.cache_write_tokens,
        "cost_usd": str(response.cost_usd),
        "latency_ns": response.latency_ns,
    }
    if (
        outcome.get("schema_version")
        not in queued_runner.SUPPORTED_STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSIONS
        or outcome.get("task_id") != expected_call_id
        or outcome.get("status") != "succeeded"
        or type(attempts) is not list
        or not attempts
        or attempted.attempt_count != len(attempts)
        or type(outcome_response) is not dict
        or outcome_response != expected_outcome_response
        or outcome_response.get("requested_model") != MODEL
        or outcome_response.get("resolved_model") not in ALLOWED_RESOLVED_MODELS
        or outcome_response.get("resolved_provider") != RESOLVED_PROVIDER
    ):
        raise ConformanceRunError("terminal outcome telemetry violates the contract")

    physical_attempt_ids: list[str] = []
    for expected_number, attempt in enumerate(attempts, start=1):
        if type(attempt) is not dict:
            raise ConformanceRunError("physical attempt telemetry is not an object")
        evidence = attempt.get("request_evidence")
        if (
            attempt.get("attempt_number") != expected_number
            or type(evidence) is not dict
        ):
            raise ConformanceRunError("physical attempt sequence/evidence is invalid")
        attempt_id = evidence.get("provider_attempt_id")
        if type(attempt_id) is not str:
            raise ConformanceRunError("physical attempt identity is missing")
        try:
            ProviderAttemptId(attempt_id)
        except (TypeError, ValueError) as error:
            raise ConformanceRunError("physical attempt identity is invalid") from error
        physical_attempt_ids.append(attempt_id)
    if len(set(physical_attempt_ids)) != len(physical_attempt_ids):
        raise ConformanceRunError("physical attempt identities are not unique")
    if attempts[-1].get("status") != "succeeded" or any(
        attempt.get("status") == "succeeded" for attempt in attempts[:-1]
    ):
        raise ConformanceRunError("successful physical attempt position is invalid")

    known_attempt_ids = set(physical_attempt_ids)
    progress_by_attempt: dict[str, list[Mapping[str, object]]] = {}
    for row in progress_rows:
        if type(row) is not dict:
            raise ConformanceRunError("stream progress row is not an exact object")
        attempt_id = row.get("provider_attempt_id")
        if row.get("call_id") != expected_call_id:
            raise ConformanceRunError("stream progress logical identity is invalid")
        if type(attempt_id) is not str or attempt_id not in known_attempt_ids:
            raise ConformanceRunError("stream progress has an unbound attempt identity")
        progress_by_attempt.setdefault(attempt_id, []).append(row)
    for attempt_rows in progress_by_attempt.values():
        previous_cumulative = 0
        previous_elapsed = -1
        completion_positions: list[int] = []
        for expected_sequence, row in enumerate(attempt_rows, start=1):
            event_bytes = row.get("event_content_utf8_bytes")
            cumulative_bytes = row.get("cumulative_content_utf8_bytes")
            elapsed_ns = row.get("elapsed_ns")
            digest = row.get("rolling_content_sha256")
            try:
                kind = StructuredStreamProgressKind(row.get("kind"))
                channel = StructuredStreamChannel(row.get("channel"))
            except (TypeError, ValueError) as error:
                raise ConformanceRunError(
                    "stream progress kind/channel is invalid"
                ) from error
            if (
                row.get("schema_version") != 2
                or row.get("sequence") != expected_sequence
                or type(event_bytes) is not int
                or event_bytes < 0
                or type(cumulative_bytes) is not int
                or cumulative_bytes != previous_cumulative + event_bytes
                or type(elapsed_ns) is not int
                or elapsed_ns < previous_elapsed
                or type(digest) is not str
                or _LOWER_SHA256.fullmatch(digest) is None
            ):
                raise ConformanceRunError("stream progress telemetry is inconsistent")
            if kind in {
                StructuredStreamProgressKind.OUTPUT_SELECTED,
                StructuredStreamProgressKind.STREAM_COMPLETED,
            } and (
                channel is not StructuredStreamChannel.OTHER or event_bytes != 0
            ):
                raise ConformanceRunError(
                    "content-free lifecycle progress is inconsistent"
                )
            if kind is StructuredStreamProgressKind.STREAM_COMPLETED:
                completion_positions.append(expected_sequence)
            previous_cumulative = cumulative_bytes
            previous_elapsed = elapsed_ns
        if len(completion_positions) > 1 or (
            completion_positions and completion_positions[0] != len(attempt_rows)
        ):
            raise ConformanceRunError(
                "local stream completion is not unique and terminal"
            )

    successful_attempt_id = physical_attempt_ids[-1]
    successful_progress = [
        row
        for row in progress_rows
        if row.get("provider_attempt_id") == successful_attempt_id
    ]
    completion_progress = [
        row
        for row in successful_progress
        if row.get("kind") == StructuredStreamProgressKind.STREAM_COMPLETED.value
    ]
    if (
        not successful_progress
        or successful_progress[0].get("sequence") != 1
        or len(completion_progress) != 1
        or completion_progress[0] is not successful_progress[-1]
        or type(successful_progress[-1].get("cumulative_content_utf8_bytes"))
        is not int
        or successful_progress[-1]["cumulative_content_utf8_bytes"] <= 0
        or _LOWER_SHA256.fullmatch(
            str(successful_progress[-1].get("rolling_content_sha256"))
        )
        is None
    ):
        raise ConformanceRunError("successful stream telemetry is incomplete")
    return {
        "attempt_count": attempted.attempt_count,
        "successful_provider_attempt_id": successful_attempt_id,
        "response": dict(outcome_response),
        "typed_output": response.value.model_dump(mode="json"),
        "progress": _progress_summary(progress_rows),
    }


async def _call_runner(
    runner: Runner,
    request: StructuredGenerationRequest[StreamingConformanceOutput],
) -> AttemptedStructuredGenerationResponse[StreamingConformanceOutput]:
    async with runner:
        return await runner(request)


def execute(
    *,
    mode: Literal["readiness", "live"],
    run_dir: Path,
    release_gate_dir: Path,
    config: ProgressAwareOpenRouterConfig,
    dependencies: ConformanceDependencies = DEFAULT_DEPENDENCIES,
) -> dict[str, object]:
    """Create one fresh, finalized readiness or live evidence directory."""

    if mode not in {"readiness", "live"}:
        raise ValueError("mode must be readiness or live")
    if type(dependencies) is not ConformanceDependencies:
        raise TypeError("dependencies must be ConformanceDependencies")
    _validate_frozen_config(config)
    root = run_dir.expanduser().resolve(strict=False)
    if _SAFE_RUN_ID.fullmatch(root.name) is None:
        raise ConformanceRunError("run ID violates the closed conformance grammar")
    if mode == "live" and root.name != FROZEN_LIVE_RUN_ID:
        raise ConformanceRunError("live run ID violates the frozen protocol")
    if root.exists():
        raise FileExistsError(root)
    release_gate, release_gate_finalization = verify_provider_free_release_gate(
        release_gate_dir,
        config=config,
    )
    root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(root / "provider_free_release_gate.json", release_gate)
    manifest = build_manifest(
        mode=mode,
        run_dir=root,
        config=config,
        release_gate_dir=release_gate_dir,
        release_gate=release_gate,
        release_gate_finalization=release_gate_finalization,
    )
    write_json_atomic(root / "manifest.json", manifest)
    persisted = decode_json_bytes((root / "manifest.json").read_bytes())
    verify_manifest(
        persisted,
        mode=mode,
        run_dir=root,
        config=config,
        release_gate_dir=release_gate_dir,
        release_gate=release_gate,
        release_gate_finalization=release_gate_finalization,
    )

    progress_journal = BatchedDurableJsonlJournal(
        root / "stream_progress.jsonl",
        max_unfsynced_rows=PROGRESS_MAX_UNFSYNCED_ROWS,
    )
    outcome_journal = DurableJsonlJournal(root / "queue_outcomes.jsonl")
    progress = ProgressRecorder(progress_journal)
    outcome_rows: list[dict[str, object]] = []
    credential_read_attempted = False
    credentials_read = False
    client_constructed = False
    provider_call_attempted = False
    result: dict[str, object] | None = None
    pending: BaseException | None = None
    try:
        if mode == "readiness":
            result = {
                "schema_version": 2,
                "status": "ready_provider_not_called",
                "credential_read_attempted": False,
                "credentials_read": False,
                "client_constructed": False,
                "provider_call_attempted": False,
                "non_network_gates": {
                    "preregistration_identity": "passed",
                    "provider_free_release_gate": "passed",
                    "focused_pytest_evidence": "passed",
                    "manifest_commitment": "passed",
                    "source_identity": "passed",
                    "route_snapshot": "passed",
                    "typed_nonce_schema": "passed",
                    "exact_request": "passed",
                    "progress_aware_composition": "passed",
                    "durable_journals": "passed",
                },
                "scientific_result_eligible": False,
                "partial_output_persisted": False,
            }
        else:
            # Recheck mutable source/snapshot gates immediately before the only
            # credential read. Client construction and dispatch remain after it.
            rechecked_gate, rechecked_finalization = (
                verify_provider_free_release_gate(release_gate_dir, config=config)
            )
            if (
                rechecked_gate != release_gate
                or rechecked_finalization != release_gate_finalization
            ):
                raise ConformanceRunError("provider-free release gate changed")
            verify_manifest(
                persisted,
                mode=mode,
                run_dir=root,
                config=config,
                release_gate_dir=release_gate_dir,
                release_gate=release_gate,
                release_gate_finalization=release_gate_finalization,
            )
            credential_read_attempted = True
            api_key = dependencies.credential_loader()
            credentials_read = True

            def outcome_sink(outcome: Any) -> None:
                # Required publication barrier: a successful response cannot
                # escape before every preceding progress row is durable.
                progress.seal_and_flush()
                record = queued_runner.structured_generation_outcome_record(outcome)
                outcome_journal.append(record)
                outcome_rows.append(record)

            runner = dependencies.runner_factory(
                api_key=api_key,
                config=config,
                progress_sink=progress,
                outcome_sink=outcome_sink,
            )
            client_constructed = True
            request = build_request(root.name)
            provider_call_attempted = True
            attempted = asyncio.run(_call_runner(runner, request))
            validated = validate_completed_call(
                attempted,
                expected_call_id=request.call_id.value,
                outcome_rows=outcome_rows,
                progress_rows=progress.rows,
            )
            result = {
                "schema_version": 2,
                "status": "completed_conformance_only",
                "credential_read_attempted": credential_read_attempted,
                "credentials_read": credentials_read,
                "client_constructed": client_constructed,
                "provider_call_attempted": provider_call_attempted,
                **validated,
                "scientific_result_eligible": False,
                "partial_output_persisted": False,
            }
    except BaseException as error:
        pending = error
        result = {
            "schema_version": 2,
            "status": "failed_conformance_only",
            "credential_read_attempted": credential_read_attempted,
            "credentials_read": credentials_read,
            "client_constructed": client_constructed,
            "provider_call_attempted": provider_call_attempted,
            "failure_type": type(error).__name__,
            "scientific_result_eligible": False,
            "partial_output_persisted": False,
        }
    finally:
        try:
            progress.flush()
        except BaseException as flush_error:
            if pending is None:
                pending = flush_error
                result = {
                    "schema_version": 2,
                    "status": "failed_conformance_only",
                    "credential_read_attempted": credential_read_attempted,
                    "credentials_read": credentials_read,
                    "client_constructed": client_constructed,
                    "provider_call_attempted": provider_call_attempted,
                    "failure_type": type(flush_error).__name__,
                    "scientific_result_eligible": False,
                    "partial_output_persisted": False,
                }
        progress_journal.close()
        outcome_journal.close()

    assert result is not None
    write_json_atomic(root / "result.json", result)
    finalization = finalize_run_directory(root, status=str(result["status"]))
    if pending is not None:
        raise ConformanceRunError(
            "streaming conformance run failed; inspect finalized artifacts"
        ) from None
    return {
        "run_dir": str(root),
        "result": result,
        "finalization": finalization,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("readiness", "live"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--release-gate-dir",
        type=Path,
        required=True,
        help="finalized provider-free focused-test release gate",
    )
    parser.add_argument(
        "--first-event-timeout-seconds",
        type=int,
        default=DEFAULT_FIRST_EVENT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=int,
        default=DEFAULT_IDLE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--absolute-timeout-seconds",
        type=int,
        default=DEFAULT_ABSOLUTE_TIMEOUT_SECONDS,
        help="0 disables the independent absolute operational fail-safe",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = build_config(
        first_event_timeout_seconds=args.first_event_timeout_seconds,
        idle_timeout_seconds=args.idle_timeout_seconds,
        absolute_timeout_seconds=args.absolute_timeout_seconds,
    )
    try:
        summary = execute(
            mode=args.mode,
            run_dir=args.run_dir,
            release_gate_dir=args.release_gate_dir,
            config=config,
        )
    except (ConformanceRunError, FileExistsError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_dir": summary["run_dir"],
                "status": summary["result"]["status"],
                "finalization_sha256": summary["finalization"][
                    "finalization_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
