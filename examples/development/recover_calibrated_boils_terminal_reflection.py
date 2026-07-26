#!/usr/bin/env python3
"""Replay one sealed failed reflection without reopening its evolution run.

``--verify`` is credential-free: it authenticates the source campaign, rebuilds
the exact failed engine and provider requests, and checks two independent
transient-failure controls. ``--live`` performs exactly one provider attempt and
writes it to a new quarantined, recursively finalized artifact directory.

This harness is deliberately generic at the recovery boundary.  A frozen
``ReflectionRecoverySpec`` supplies source identities and comparison controls;
all reconstruction, route validation, evidence publication, and immutability
checks operate only on sealed traces and public AgentEvolve contracts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from dotenv import load_dotenv  # noqa: E402

from agent_evolve.application.agentic_evolution import (  # noqa: E402
    ReflectionCallRequest,
)
from agent_evolve.domain.ids import (  # noqa: E402
    InsightId,
    LLMCallId,
    OperatorInvocationId,
)
from agent_evolve.domain.insight import InsightRef  # noqa: E402
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    PydanticAIAgenticGenerator,
    render_reflection_prompt,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (  # noqa: E402
    OpenRouterReasoningConfig,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (  # noqa: E402
    ProgressAwareOpenRouterConfig,
    ProgressAwareRetryMode,
    create_progress_aware_openrouter_runner,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    StructuredEvidencePublicationPolicy,
    structured_generation_outcome_record,
)
from agent_evolve.ports.agentic_generator import (  # noqa: E402
    AgenticCallTelemetry,
    ReflectionGenerationRequest,
    ReflectionInsightContract,
    validate_reflection_insight_draft,
)
from agent_evolve.ports.artifact_store import (  # noqa: E402
    canonical_json_bytes,
    decode_json_bytes,
)
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredStreamCleanupPolicy,
    StructuredStreamLivenessPolicy,
    StructuredStreamProgress,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    BatchedDurableJsonlJournal,
    DurableJsonlJournal,
    finalize_run_directory,
    read_jsonl,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
)
BOILS_ROOT = ARTIFACT_ROOT / "boils_abc/portfolio_q"
HEAT_ROOT = ARTIFACT_ROOT / "benchmark_q1/engibench_heat2d/generic_campaign"

MODEL = "deepseek/deepseek-v4-pro"
CANONICAL_MODEL = "deepseek/deepseek-v4-pro-20260423"
PROVIDER_ONLY = ("streamlake",)
RESOLVED_PROVIDER = "StreamLake"
MAX_OUTPUT_TOKENS = 384_000
MIN_INSIGHTS = 1
MAX_INSIGHTS = 2
TEMPERATURE = 0.2

# The recovery is one paid attempt, not a fresh optimization or a retry block.
MAX_ATTEMPTS = 1
FIRST_EVENT_TIMEOUT_NS = 300_000_000_000
IDLE_TIMEOUT_NS = 300_000_000_000
CLEANUP_TIMEOUT_NS = 5_000_000_000
CONNECT_TIMEOUT_SECONDS = 90.0
BASE_BACKOFF_NS = 1_000_000_000
MAX_BACKOFF_NS = 30_000_000_000
JITTER_SEED = 20_260_716_400
JITTER_DOMAIN = "calibrated-boils-terminal-reflection-recovery-v1"
REQUEST_DOMAIN = b"agent-evolve:sealed-reflection-recovery-request:v2\x00"


@dataclass(frozen=True, slots=True)
class FinalizedRunIdentity:
    path: Path
    status: str
    finalization_sha256: str
    recursive_content_sha256: str


@dataclass(frozen=True, slots=True)
class ReflectionRecoverySpec:
    source: FinalizedRunIdentity
    output_run: Path
    source_call_id: str
    source_cycle: int
    semantic_prompt_sha256: str
    wire_prompt_sha256: str
    request_sha256: str
    contract_sha256: str
    failure_envelope_sha256: str
    successful_predecessor_call_ids: tuple[str, ...]
    heat_failed: FinalizedRunIdentity
    heat_succeeded: FinalizedRunIdentity
    heat_control_call_id: str
    heat_control_wire_prompt_sha256: str
    heat_control_provider_attempt_id: str


SPEC = ReflectionRecoverySpec(
    source=FinalizedRunIdentity(
        path=BOILS_ROOT / "boilsq_calibrated_g6_live_deepseek_v4_20260716",
        status="completed_unhealthy",
        finalization_sha256=(
            "089e0a05ca9cd31b9612b1ff506c3103271d913b4d2b7a049ca46d016461918a"
        ),
        recursive_content_sha256=(
            "c6c8b0110006618805e1aba281752447251a80606c70a73b7e2a13025ec66860"
        ),
    ),
    output_run=(
        BOILS_ROOT
        / "boilsq_calibrated_g6_cycle3_reflection_recovery_deepseek_v4_20260716"
    ),
    source_call_id="call_boilsq_9ed1f1d42cbb021b_000009",
    source_cycle=3,
    semantic_prompt_sha256=(
        "627d8814154193917c6ac85f3fc25af8b8303d030b68c408157db69bd30715e0"
    ),
    wire_prompt_sha256=(
        "1c7fa407a564f5c22852e8a3a003a761806612a4ac653799d7c5ddab61918aff"
    ),
    request_sha256=(
        "3eb122d799281847eb5157eb0c768f16f8dffa23f9c72ef32b2e5bdc06eb5670"
    ),
    contract_sha256=(
        "d8d29f9dd939560bb203ea9d9260a78b9c50103fa128b6a29eae51b69d957f48"
    ),
    failure_envelope_sha256=(
        "36da5db2a9f34acc88c283124da8c17940cb95e3327832a49f9c65d4e127014e"
    ),
    successful_predecessor_call_ids=(
        "call_boilsq_9ed1f1d42cbb021b_000005",
        "call_boilsq_9ed1f1d42cbb021b_000008",
    ),
    heat_failed=FinalizedRunIdentity(
        path=HEAT_ROOT / "heat2d_generic_g3_live_deepseek_v3_20260716",
        status="failed",
        finalization_sha256=(
            "9afe264e872f03acc3b534fd5913132640e5c660ada0d99f07f087fb74dc015d"
        ),
        recursive_content_sha256=(
            "c14e152d49a9bd0744d667e210ac51f856ec39c6eaff38f65389eeaf7a60a0e3"
        ),
    ),
    heat_succeeded=FinalizedRunIdentity(
        path=HEAT_ROOT / "heat2d_generic_g3_live_deepseek_v4_20260716",
        status="completed_unhealthy",
        finalization_sha256=(
            "75a3190c6d9ce4c812de4f9aafb4018904e7fbd5e8e908d6036a0db02e462816"
        ),
        recursive_content_sha256=(
            "ff79fe3e2522f5fec9352f764658179e6177667d09fabca7cef022a3bc870fd6"
        ),
    ),
    heat_control_call_id="call_heatg3_matched_r1_000001",
    heat_control_wire_prompt_sha256=(
        "c4973a54ee04e25dc8c46be457b64977645054a23c9b241843f7c91d7fb38abb"
    ),
    heat_control_provider_attempt_id=(
        "provider_attempt_5d5c50db8cffd7f565f19162fe55c575df95f731293f6ebb0df0a1a579fb44d8"
    ),
)


@dataclass(frozen=True, slots=True)
class ReconstructedReflection:
    prompt: str
    contract: ReflectionInsightContract
    engine_request: ReflectionCallRequest
    provider_request: ReflectionGenerationRequest
    verification: dict[str, object]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _exact_object(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise RuntimeError(f"{label} is not an exact object")
    return value


def _exact_list(value: object, *, label: str) -> list[object]:
    if type(value) is not list:
        raise RuntimeError(f"{label} is not an exact list")
    return value


def _json_object(path: Path) -> dict[str, object]:
    return _exact_object(
        decode_json_bytes(path.expanduser().resolve(strict=True).read_bytes()),
        label=path.name,
    )


def _authenticated_record(row: Mapping[str, object]) -> dict[str, object]:
    value = row.get("authenticated_record", row)
    return _exact_object(value, label="authenticated JSONL record")


def _verify_identity(identity: FinalizedRunIdentity) -> dict[str, object]:
    record = verify_finalized_run_directory(identity.path)
    if (
        record.get("status") != identity.status
        or record.get("finalization_sha256") != identity.finalization_sha256
        or record.get("recursive_content_sha256")
        != identity.recursive_content_sha256
    ):
        raise RuntimeError(f"sealed run identity drifted: {identity.path.name}")
    return record


def _route_record(manifest: Mapping[str, object]) -> dict[str, object]:
    model = _exact_object(manifest.get("model"), label="model manifest")
    provider = _exact_object(
        model.get("provider_options"),
        label="provider options",
    )
    reasoning = _exact_object(model.get("reasoning"), label="reasoning settings")
    raw_temperature = model.get("temperature_hex", model.get("temperature"))
    temperature_hex = (
        str(raw_temperature)
        if isinstance(raw_temperature, str)
        else float(raw_temperature).hex()
    )
    return {
        "model_name": model.get("model_name", model.get("requested_model")),
        "provider_only": provider.get("only"),
        "allow_fallbacks": provider.get("allow_fallbacks"),
        "reasoning_effort": reasoning.get("effort"),
        "reasoning_mode": model.get("reasoning_mode"),
        "max_output_tokens": model.get("max_output_tokens"),
        "temperature_hex": temperature_hex,
    }


def _assert_frozen_route(manifest: Mapping[str, object]) -> dict[str, object]:
    observed = _route_record(manifest)
    expected = {
        "model_name": MODEL,
        "provider_only": list(PROVIDER_ONLY),
        "allow_fallbacks": False,
        "reasoning_effort": "xhigh",
        "reasoning_mode": None,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature_hex": TEMPERATURE.hex(),
    }
    if observed != expected:
        raise RuntimeError("comparison run escaped the frozen provider route")
    return observed


def _one_queue_outcome(path: Path, *, call_id: str) -> dict[str, object]:
    matches = [
        _authenticated_record(row)
        for row in read_jsonl(path)
        if _authenticated_record(row).get("task_id") == call_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"no unique queue outcome for {call_id}")
    return matches[0]


def _one_attempt(outcome: Mapping[str, object]) -> dict[str, object]:
    attempts = _exact_list(outcome.get("attempts"), label="queue attempts")
    if len(attempts) != 1:
        raise RuntimeError("recovery evidence requires exactly one provider attempt")
    return _exact_object(attempts[0], label="queue attempt")


def _reconstruct_engine_request(raw: Mapping[str, object]) -> ReflectionCallRequest:
    predecessors: list[InsightRef] = []
    for value in _exact_list(
        raw.get("revision_predecessors"),
        label="revision predecessors",
    ):
        item = _exact_object(value, label="revision predecessor")
        predecessors.append(
            InsightRef(
                InsightId(str(item.get("insight_id"))),
                int(item.get("version")),
            )
        )
    temperature_hex = raw.get("temperature_hex")
    request = ReflectionCallRequest(
        label=str(raw.get("label")),
        operation=str(raw.get("operation")),
        prompt_sha256=str(raw.get("prompt_sha256")),
        min_insights=int(raw.get("min_insights")),
        max_insights=int(raw.get("max_insights")),
        max_output_tokens=int(raw.get("max_output_tokens")),
        temperature=(
            None
            if temperature_hex is None
            else float.fromhex(str(temperature_hex))
        ),
        insight_contract_sha256=(
            None
            if raw.get("insight_contract_sha256") is None
            else str(raw.get("insight_contract_sha256"))
        ),
        revision_predecessors=tuple(predecessors),
        revision_predecessor_content_sha256s=tuple(
            str(value)
            for value in _exact_list(
                raw.get("revision_predecessor_content_sha256s"),
                label="revision predecessor content hashes",
            )
        ),
        source_receipt_sha256s=tuple(
            str(value)
            for value in _exact_list(
                raw.get("source_receipt_sha256s"),
                label="source receipt hashes",
            )
        ),
        source_operator_invocation_ids=tuple(
            OperatorInvocationId(str(value))
            for value in _exact_list(
                raw.get("source_operator_invocation_ids"),
                label="source operator invocation IDs",
            )
        ),
        source_outcome_sha256s=tuple(
            str(value)
            for value in _exact_list(
                raw.get("source_outcome_sha256s"),
                label="source outcome hashes",
            )
        ),
        available_contrast_ids=tuple(
            str(value)
            for value in _exact_list(
                raw.get("available_contrast_ids"),
                label="available contrast IDs",
            )
        ),
        request_sha256=str(raw.get("request_sha256")),
    )
    reconstructed = {**request.to_record(), "request_sha256": request.request_sha256}
    if reconstructed != dict(raw):
        raise RuntimeError("ReflectionCallRequest did not reconstruct byte semantics")
    return request


def _contract(raw: Mapping[str, object]) -> ReflectionInsightContract:
    metric_ids = _exact_list(
        raw.get("required_metric_ids"),
        label="reflection metric IDs",
    )
    families = _exact_list(
        raw.get("allowed_option_families"),
        label="reflection option families",
    )
    option_ids = _exact_list(
        raw.get("allowed_option_ids"),
        label="reflection option IDs",
    )
    if any(
        type(value) is not str
        for value in (*metric_ids, *families, *option_ids)
    ):
        raise RuntimeError("reflection contract contains a non-string vocabulary")
    value = ReflectionInsightContract(
        required_metric_ids=tuple(metric_ids),
        allowed_option_families=tuple(families),
        allowed_option_ids=tuple(option_ids),
    )
    if value.to_record() != dict(raw):
        raise RuntimeError("ReflectionInsightContract did not reconstruct exactly")
    return value


def _comparison_evidence(
    spec: ReflectionRecoverySpec,
    *,
    source_engine_rows: tuple[dict[str, object], ...],
    source_summary: Mapping[str, object],
) -> dict[str, object]:
    source_route = _assert_frozen_route(_json_object(spec.source.path / "manifest.json"))
    engine_by_call = {
        str(row.get("call_id")): row
        for row in source_engine_rows
        if row.get("event_type") == "reflection_requested"
    }
    reflections = _exact_list(
        source_summary.get("reflections"),
        label="source reflections",
    )
    summary_by_call: dict[str, dict[str, object]] = {}
    for value in reflections:
        row = _exact_object(value, label="source reflection summary")
        receipt = _exact_object(row.get("receipt"), label="reflection receipt")
        summary_by_call[str(receipt.get("call_id"))] = row

    predecessors: list[dict[str, object]] = []
    failed_prompt_bytes = len(
        str(engine_by_call[spec.source_call_id]["prompt"]).encode("utf-8")
    )
    for call_id in spec.successful_predecessor_call_ids:
        event = engine_by_call.get(call_id)
        summary_row = summary_by_call.get(call_id)
        if event is None or summary_row is None:
            raise RuntimeError("source predecessor reflection is missing")
        receipt = _exact_object(
            summary_row.get("receipt"),
            label="successful predecessor receipt",
        )
        request = _exact_object(receipt.get("request"), label="predecessor request")
        telemetry = _exact_object(receipt.get("telemetry"), label="predecessor telemetry")
        outcome = _one_queue_outcome(
            spec.source.path / "queue_outcomes.jsonl",
            call_id=call_id,
        )
        attempt = _one_attempt(outcome)
        request_evidence = _exact_object(
            attempt.get("request_evidence"),
            label="predecessor request evidence",
        )
        prompt = str(event.get("prompt"))
        wire_sha = _sha_text(render_reflection_prompt(prompt))
        prompt_bytes = len(prompt.encode("utf-8"))
        if (
            summary_row.get("event_type") != "reflection_completed_quarantined"
            or receipt.get("status") != "completed"
            or outcome.get("status") != "succeeded"
            or attempt.get("status") != "succeeded"
            or request.get("prompt_sha256") != _sha_text(prompt)
            or event.get("reflection_request_sha256")
            != request.get("request_sha256")
            or event.get("insight_contract") is None
            or _exact_object(
                event.get("insight_contract"), label="predecessor contract"
            ).get("contract_identity_sha256")
            != spec.contract_sha256
            or request_evidence.get("prompt_sha256") != wire_sha
            or telemetry.get("requested_model") != MODEL
            or telemetry.get("resolved_model") not in {MODEL, CANONICAL_MODEL}
            or telemetry.get("resolved_provider") != RESOLVED_PROVIDER
            or not isinstance(telemetry.get("reasoning_tokens"), int)
            or telemetry["reasoning_tokens"] <= 0
            or prompt_bytes <= 0
        ):
            raise RuntimeError("successful predecessor reflection evidence drifted")
        predecessors.append(
            {
                "call_id": call_id,
                "semantic_prompt_sha256": event.get("prompt_sha256"),
                "wire_prompt_sha256": wire_sha,
                "request_sha256": request.get("request_sha256"),
                "contract_sha256": spec.contract_sha256,
                "prompt_utf8_bytes": prompt_bytes,
                "status": "succeeded_first_attempt",
                "reasoning_tokens": telemetry["reasoning_tokens"],
                "resolved_provider": telemetry["resolved_provider"],
            }
        )

    _verify_identity(spec.heat_failed)
    _verify_identity(spec.heat_succeeded)
    heat_failed_route = _assert_frozen_route(
        _json_object(spec.heat_failed.path / "manifest.json")
    )
    heat_succeeded_route = _assert_frozen_route(
        _json_object(spec.heat_succeeded.path / "manifest.json")
    )
    heat_failure = _one_queue_outcome(
        spec.heat_failed.path / "queue_outcomes.jsonl",
        call_id=spec.heat_control_call_id,
    )
    heat_success = _one_queue_outcome(
        spec.heat_succeeded.path / "queue_outcomes.jsonl",
        call_id=spec.heat_control_call_id,
    )
    failed_attempt = _one_attempt(heat_failure)
    succeeded_attempt = _one_attempt(heat_success)
    failed_request = _exact_object(
        failed_attempt.get("request_evidence"),
        label="Heat v3 request evidence",
    )
    succeeded_request = _exact_object(
        succeeded_attempt.get("request_evidence"),
        label="Heat v4 request evidence",
    )
    heat_failure_detail = _exact_object(
        failed_attempt.get("failure"),
        label="Heat v3 failure",
    )
    heat_response = _exact_object(heat_success.get("response"), label="Heat v4 response")
    if (
        heat_failure.get("status") != "terminal_failure"
        or failed_attempt.get("status") != "terminal_failure"
        or heat_failure_detail.get("kind") != "invalid_request"
        or heat_failure_detail.get("status_code") != 400
        or heat_failure_detail.get("provider_error_envelope_sha256")
        != spec.failure_envelope_sha256
        or heat_success.get("status") != "succeeded"
        or succeeded_attempt.get("status") != "succeeded"
        or failed_request.get("prompt_sha256")
        != spec.heat_control_wire_prompt_sha256
        or succeeded_request.get("prompt_sha256")
        != spec.heat_control_wire_prompt_sha256
        or failed_request.get("provider_attempt_id")
        != spec.heat_control_provider_attempt_id
        or succeeded_request.get("provider_attempt_id")
        != spec.heat_control_provider_attempt_id
        or heat_response.get("requested_model") != MODEL
        or heat_response.get("resolved_model") not in {MODEL, CANONICAL_MODEL}
        or heat_response.get("resolved_provider") != RESOLVED_PROVIDER
        or source_route != heat_failed_route
        or source_route != heat_succeeded_route
    ):
        raise RuntimeError("Heat exact failure-then-success control did not authenticate")

    return {
        "schema_version": 1,
        "same_route_across_source_and_heat_controls": True,
        "route": source_route,
        "source_successful_predecessors": predecessors,
        "failed_prompt_no_larger_than_predecessors": all(
            failed_prompt_bytes <= int(value["prompt_utf8_bytes"])
            for value in predecessors
        ),
        "heat_exact_request_failure_then_success": {
            "call_id": spec.heat_control_call_id,
            "wire_prompt_sha256": spec.heat_control_wire_prompt_sha256,
            "provider_attempt_id": spec.heat_control_provider_attempt_id,
            "v3_status": "terminal_failure_http_400",
            "v4_status": "succeeded_first_attempt",
            "failure_envelope_sha256": spec.failure_envelope_sha256,
            "same_call_prompt_attempt_identity": True,
        },
        "shared_failure_envelope_with_heat_v3": True,
    }


def reconstruct(
    spec: ReflectionRecoverySpec = SPEC,
) -> ReconstructedReflection:
    """Authenticate and reconstruct the exact failed request without credentials."""

    source_finalization = _verify_identity(spec.source)
    source_engine_rows = read_jsonl(spec.source.path / "engine_events.jsonl")
    summary = _json_object(spec.source.path / "summary.json")
    events = [
        row
        for row in source_engine_rows
        if row.get("event_type") == "reflection_requested"
        and row.get("call_id") == spec.source_call_id
    ]
    if len(events) != 1:
        raise RuntimeError("source has no unique failed reflection request event")
    event = events[0]
    prompt = event.get("prompt")
    if type(prompt) is not str:
        raise RuntimeError("source reflection prompt is unavailable")
    if (
        _sha_text(prompt) != spec.semantic_prompt_sha256
        or event.get("prompt_sha256") != spec.semantic_prompt_sha256
        or _sha_text(render_reflection_prompt(prompt)) != spec.wire_prompt_sha256
    ):
        raise RuntimeError("semantic or wire prompt reconstruction drifted")

    raw_contract = _exact_object(
        event.get("insight_contract"),
        label="source reflection contract",
    )
    contract = _contract(raw_contract)
    if contract.identity_sha256 != spec.contract_sha256:
        raise RuntimeError("source reflection contract identity drifted")

    reflections = _exact_list(summary.get("reflections"), label="summary reflections")
    failed_rows: list[dict[str, object]] = []
    for value in reflections:
        row = _exact_object(value, label="summary reflection row")
        receipt = _exact_object(row.get("receipt"), label="summary reflection receipt")
        if (
            row.get("cycle") == spec.source_cycle
            and row.get("event_type") == "reflection_failed"
            and receipt.get("call_id") == spec.source_call_id
        ):
            failed_rows.append(row)
    if len(failed_rows) != 1:
        raise RuntimeError("summary has no unique cycle-3 failed reflection receipt")
    failed_receipt = _exact_object(
        failed_rows[0].get("receipt"),
        label="failed reflection receipt",
    )
    raw_request = _exact_object(
        failed_receipt.get("request"),
        label="failed engine request",
    )
    engine_request = _reconstruct_engine_request(raw_request)
    if (
        engine_request.request_sha256 != spec.request_sha256
        or engine_request.prompt_sha256 != spec.semantic_prompt_sha256
        or engine_request.insight_contract_sha256 != spec.contract_sha256
        or event.get("reflection_request_sha256") != spec.request_sha256
        or event.get("available_contrast_ids")
        != list(engine_request.available_contrast_ids)
        or event.get("source_receipt_sha256s")
        != list(engine_request.source_receipt_sha256s)
        or failed_receipt.get("status") != "failed"
        or failed_receipt.get("failure_type") != "QueuedStructuredGenerationError"
        or engine_request.min_insights != MIN_INSIGHTS
        or engine_request.max_insights != MAX_INSIGHTS
        or engine_request.max_output_tokens != MAX_OUTPUT_TOKENS
        or engine_request.temperature != TEMPERATURE
    ):
        raise RuntimeError("engine event and failed summary request do not join")

    provider_request = ReflectionGenerationRequest(
        call_id=LLMCallId(spec.source_call_id),
        operation=engine_request.operation,
        prompt=prompt,
        max_insights=engine_request.max_insights,
        min_insights=engine_request.min_insights,
        max_output_tokens=engine_request.max_output_tokens,
        temperature=engine_request.temperature,
        available_contrast_ids=engine_request.available_contrast_ids,
        insight_contract=contract,
    )
    ReflectionGenerationRequest.__post_init__(provider_request)

    failed_outcome = _one_queue_outcome(
        spec.source.path / "queue_outcomes.jsonl",
        call_id=spec.source_call_id,
    )
    failed_attempt = _one_attempt(failed_outcome)
    failure = _exact_object(failed_attempt.get("failure"), label="source 400 failure")
    request_evidence = _exact_object(
        failed_attempt.get("request_evidence"),
        label="source failed request evidence",
    )
    if (
        failed_outcome.get("status") != "terminal_failure"
        or failed_attempt.get("status") != "terminal_failure"
        or failure.get("kind") != "invalid_request"
        or failure.get("status_code") != 400
        or failure.get("provider_error_envelope_sha256")
        != spec.failure_envelope_sha256
        or request_evidence.get("prompt_sha256") != spec.wire_prompt_sha256
    ):
        raise RuntimeError("source terminal 400 premise did not authenticate")

    comparison = _comparison_evidence(
        spec,
        source_engine_rows=source_engine_rows,
        source_summary=summary,
    )
    verification = {
        "schema_version": 1,
        "verified_at_utc": _utc_now(),
        "source_run": spec.source.path.name,
        "source_status": source_finalization["status"],
        "source_finalization_sha256": spec.source.finalization_sha256,
        "source_recursive_content_sha256": spec.source.recursive_content_sha256,
        "source_call_id": spec.source_call_id,
        "source_cycle": spec.source_cycle,
        "semantic_prompt_sha256": spec.semantic_prompt_sha256,
        "wire_prompt_sha256": spec.wire_prompt_sha256,
        "source_request_sha256": spec.request_sha256,
        "contract_sha256": spec.contract_sha256,
        "failure_envelope_sha256": spec.failure_envelope_sha256,
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "available_contrast_count": len(engine_request.available_contrast_ids),
        "source_receipt_count": len(engine_request.source_receipt_sha256s),
        "source_outcome_count": len(engine_request.source_outcome_sha256s),
        "engine_summary_queue_join_verified": True,
        "exact_provider_request_reconstructed": True,
        "comparison_evidence": comparison,
    }
    return ReconstructedReflection(
        prompt=prompt,
        contract=contract,
        engine_request=engine_request,
        provider_request=provider_request,
        verification=verification,
    )


def _new_call_id(spec: ReflectionRecoverySpec = SPEC) -> LLMCallId:
    digest = _sha_text(
        f"{spec.output_run.name}\x00{spec.source_call_id}\x00single-attempt"
    )[:24]
    value = LLMCallId(f"call_recovery_{digest}_000001")
    if value.value == spec.source_call_id:
        raise RuntimeError("recovery call identity collided with the source")
    return value


def _provider_request_record(value: ReflectionGenerationRequest) -> dict[str, object]:
    return {
        "call_id": value.call_id.value,
        "operation": value.operation,
        "prompt": value.prompt,
        "semantic_prompt_sha256": _sha_text(value.prompt),
        "wire_prompt_sha256": _sha_text(render_reflection_prompt(value.prompt)),
        "max_insights": value.max_insights,
        "min_insights": value.min_insights,
        "max_output_tokens": value.max_output_tokens,
        "temperature_hex": (
            None if value.temperature is None else float(value.temperature).hex()
        ),
        "available_contrast_ids": list(value.available_contrast_ids),
        "insight_contract": (
            None if value.insight_contract is None else value.insight_contract.to_record()
        ),
    }


def recovery_request_record(
    reconstructed: ReconstructedReflection,
    *,
    call_id: LLMCallId,
    spec: ReflectionRecoverySpec = SPEC,
) -> dict[str, object]:
    replay = ReflectionGenerationRequest(
        call_id=call_id,
        operation=reconstructed.provider_request.operation,
        prompt=reconstructed.provider_request.prompt,
        max_insights=reconstructed.provider_request.max_insights,
        min_insights=reconstructed.provider_request.min_insights,
        max_output_tokens=reconstructed.provider_request.max_output_tokens,
        temperature=reconstructed.provider_request.temperature,
        available_contrast_ids=reconstructed.provider_request.available_contrast_ids,
        insight_contract=reconstructed.provider_request.insight_contract,
    )
    record: dict[str, object] = {
        "schema_version": 2,
        "status": "supplemental_quarantined_single_attempt",
        "lifecycle_publication_allowed": False,
        "source_run": spec.source.path.name,
        "source_call_id": spec.source_call_id,
        "source_engine_request": {
            **reconstructed.engine_request.to_record(),
            "request_sha256": reconstructed.engine_request.request_sha256,
        },
        "source_provider_request": _provider_request_record(
            reconstructed.provider_request
        ),
        "replay_provider_request": _provider_request_record(replay),
    }
    record["recovery_request_sha256"] = hashlib.sha256(
        REQUEST_DOMAIN + canonical_json_bytes(record)
    ).hexdigest()
    return record


def _config() -> ProgressAwareOpenRouterConfig:
    return ProgressAwareOpenRouterConfig(
        model_name=MODEL,
        provider_only=PROVIDER_ONLY,
        connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS,
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=FIRST_EVENT_TIMEOUT_NS,
            idle_timeout_ns=IDLE_TIMEOUT_NS,
            absolute_timeout_ns=None,
            cleanup_policy=StructuredStreamCleanupPolicy(
                cancel_drain_timeout_ns=CLEANUP_TIMEOUT_NS,
                transport_retire_timeout_ns=CLEANUP_TIMEOUT_NS,
            ),
        ),
        max_connections=1,
        max_pending=0,
        max_attempts=MAX_ATTEMPTS,
        base_backoff_ns=BASE_BACKOFF_NS,
        max_backoff_ns=MAX_BACKOFF_NS,
        jitter_seed=JITTER_SEED,
        jitter_domain=JITTER_DOMAIN,
        app_title="AgentEvolve sealed reflection recovery",
        reasoning_config=OpenRouterReasoningConfig(effort="xhigh"),
        retry_mode=ProgressAwareRetryMode.NON_REPEATING_STREAM,
    )


def _progress_record(value: StructuredStreamProgress) -> dict[str, object]:
    StructuredStreamProgress.__post_init__(value)
    return {
        "call_id": value.call_id,
        "provider_attempt_id": value.provider_attempt_id,
        "sequence": value.sequence,
        "kind": value.kind.value,
        "channel": value.channel.value,
        "elapsed_ns": value.elapsed_ns,
        "event_content_utf8_bytes": value.event_content_utf8_bytes,
        "cumulative_content_utf8_bytes": value.cumulative_content_utf8_bytes,
        "rolling_content_sha256": value.rolling_content_sha256,
    }


def _telemetry_record(value: AgenticCallTelemetry) -> dict[str, object]:
    AgenticCallTelemetry.__post_init__(value)
    return {
        "requested_model": value.requested_model,
        "resolved_model": value.resolved_model,
        "resolved_provider": value.resolved_provider,
        "provider_response_id": value.provider_response_id,
        "finish_reason": value.finish_reason,
        "input_tokens": value.input_tokens,
        "output_tokens": value.output_tokens,
        "reasoning_tokens": value.reasoning_tokens,
        "cache_read_tokens": value.cache_read_tokens,
        "cache_write_tokens": value.cache_write_tokens,
        "cost_usd": None if value.cost_usd is None else str(value.cost_usd),
        "latency_ns": value.latency_ns,
        "attempt_count": value.attempt_count,
    }


def _queue_snapshot_record(value: object) -> dict[str, object]:
    return {
        name: getattr(value, name)
        for name in (
            "max_in_flight",
            "max_pending",
            "in_flight",
            "pending",
            "closed",
        )
    }


def _same_source_identity(spec: ReflectionRecoverySpec) -> dict[str, object]:
    record = _verify_identity(spec.source)
    return {
        "source_run": spec.source.path.name,
        "finalization_sha256": record["finalization_sha256"],
        "recursive_content_sha256": record["recursive_content_sha256"],
        "unchanged": True,
        "verified_at_utc": _utc_now(),
    }


async def _live(
    spec: ReflectionRecoverySpec = SPEC,
) -> tuple[str, dict[str, object]]:
    reconstructed = reconstruct(spec)
    call_id = _new_call_id(spec)
    request_record = recovery_request_record(
        reconstructed,
        call_id=call_id,
        spec=spec,
    )
    config = _config()
    spec.output_run.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        spec.output_run / "source_verification_before.json",
        reconstructed.verification,
    )
    write_json_atomic(spec.output_run / "recovery_request.json", request_record)
    write_json_atomic(
        spec.output_run / "manifest.json",
        {
            "schema_version": 2,
            "created_at_utc": _utc_now(),
            "run_id": spec.output_run.name,
            "claim_boundary": {
                "supplemental_only": True,
                "quarantined": True,
                "mutates_source_run": False,
                "lifecycle_publication_allowed": False,
                "optimization_claim": False,
                "one_logical_call": True,
                "one_provider_attempt_maximum": True,
            },
            "source_identity": {
                "run_id": spec.source.path.name,
                "finalization_sha256": spec.source.finalization_sha256,
                "recursive_content_sha256": spec.source.recursive_content_sha256,
            },
            "recovery_request_sha256": request_record[
                "recovery_request_sha256"
            ],
            "provider": config.to_manifest_record(),
            "source_code": source_identity(
                (
                    Path(__file__),
                    AGENT_EVOLVE_ROOT
                    / "src/agent_evolve/integrations/pydantic_ai/progress_aware_openrouter.py",
                    AGENT_EVOLVE_ROOT
                    / "src/agent_evolve/integrations/pydantic_ai/queued_runner.py",
                    AGENT_EVOLVE_ROOT
                    / "src/agent_evolve/integrations/pydantic_ai/async_generator.py",
                    AGENT_EVOLVE_ROOT
                    / "src/agent_evolve/integrations/pydantic_ai/agentic_generator.py",
                ),
                relative_to=WORKSPACE_ROOT,
            ),
        },
    )

    progress = BatchedDurableJsonlJournal(
        spec.output_run / "stream_progress.jsonl",
        max_unfsynced_rows=32,
    )
    requests = DurableJsonlJournal(spec.output_run / "request_evidence.jsonl")
    outputs = DurableJsonlJournal(spec.output_run / "output_evidence.jsonl")
    outcomes = DurableJsonlJournal(spec.output_run / "queue_outcomes.jsonl")
    outbound = DurableJsonlJournal(spec.output_run / "outbound_requests.jsonl")
    lifecycle = DurableJsonlJournal(spec.output_run / "lifecycle.jsonl")
    journals = (progress, requests, outputs, outcomes, outbound, lifecycle)
    observed_requests: list[dict[str, object]] = []
    observed_outputs: list[dict[str, object]] = []
    observed_outcomes: list[dict[str, object]] = []
    observed_outbound: list[dict[str, object]] = []
    runner = None
    status = "failed"
    result_record: dict[str, object] = {}
    failure: BaseException | None = None

    def progress_sink(value: StructuredStreamProgress) -> None:
        progress.append(_progress_record(value))

    def request_sink(value: dict[str, object]) -> None:
        observed_requests.append(value)
        requests.append(value)

    def output_sink(value: dict[str, object]) -> None:
        observed_outputs.append(value)
        outputs.append(value)

    def outcome_sink(value: object) -> None:
        progress.flush()
        record = structured_generation_outcome_record(value)
        observed_outcomes.append(record)
        outcomes.append(record)

    def outbound_sink(value: dict[str, object]) -> None:
        observed_outbound.append(value)
        outbound.append(value)

    try:
        load_dotenv(WORKSPACE_ROOT / ".env", override=False)
        load_dotenv(AGENT_EVOLVE_ROOT / ".env", override=False)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if type(api_key) is not str or not api_key:
            raise RuntimeError("OpenRouter credential is unavailable")
        runner = create_progress_aware_openrouter_runner(
            api_key=api_key,
            config=config,
            progress_sink=progress_sink,
            outcome_sink=outcome_sink,
            request_evidence_sink=request_sink,
            output_evidence_sink=output_sink,
            outbound_request_manifest_sink=outbound_sink,
            evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
        )
        lifecycle.append({"event": "runner_opened", "at_utc": _utc_now()})
        replay_request = ReflectionGenerationRequest(
            call_id=call_id,
            operation=reconstructed.provider_request.operation,
            prompt=reconstructed.provider_request.prompt,
            max_insights=reconstructed.provider_request.max_insights,
            min_insights=reconstructed.provider_request.min_insights,
            max_output_tokens=reconstructed.provider_request.max_output_tokens,
            temperature=reconstructed.provider_request.temperature,
            available_contrast_ids=(
                reconstructed.provider_request.available_contrast_ids
            ),
            insight_contract=reconstructed.provider_request.insight_contract,
        )
        generated = await PydanticAIAgenticGenerator(runner).reflect(replay_request)
        if not MIN_INSIGHTS <= len(generated.insights) <= MAX_INSIGHTS:
            raise RuntimeError("recovered insight count violates the source request")
        for insight in generated.insights:
            validate_reflection_insight_draft(insight, reconstructed.contract)
            if not set(insight.evidence_contrast_ids).issubset(
                reconstructed.engine_request.available_contrast_ids
            ):
                raise RuntimeError("recovered insight cites a foreign contrast")
        telemetry = generated.telemetry
        if (
            telemetry.requested_model != MODEL
            or telemetry.resolved_model not in {MODEL, CANONICAL_MODEL}
            or telemetry.resolved_provider != RESOLVED_PROVIDER
            or telemetry.reasoning_tokens <= 0
            or telemetry.attempt_count != 1
        ):
            raise RuntimeError("recovered response violates route/reasoning gates")
        snapshot = await runner.snapshot()
        if snapshot.pending != 0 or snapshot.in_flight != 0:
            raise RuntimeError("recovery queue did not drain")
        if not (
            len(observed_requests)
            == len(observed_outputs)
            == len(observed_outcomes)
            == len(observed_outbound)
            == 1
        ):
            raise RuntimeError("required recovery evidence is incomplete or duplicated")
        queue_outcome = observed_outcomes[0]
        attempt = _one_attempt(queue_outcome)
        attempt_request = _exact_object(
            attempt.get("request_evidence"),
            label="recovery request evidence",
        )
        if (
            queue_outcome.get("status") != "succeeded"
            or queue_outcome.get("task_id") != call_id.value
            or attempt.get("status") != "succeeded"
            or attempt_request.get("prompt_sha256") != spec.wire_prompt_sha256
        ):
            raise RuntimeError("recovery queue evidence does not join the replay")
        result_record = {
            "schema_version": 2,
            "status": "completed",
            "epistemic_status": "supplemental_quarantined",
            "source_run_mutated": False,
            "lifecycle_publication_count": 0,
            "logical_recovery_calls": 1,
            "provider_attempts": 1,
            "call_id": call_id.value,
            "source_call_id": spec.source_call_id,
            "source_request_sha256": spec.request_sha256,
            "recovery_request_sha256": request_record[
                "recovery_request_sha256"
            ],
            "transient_route_failure_confirmed": True,
            "adjudication": (
                "The exact failed semantic prompt, rendered wire prompt, reflection "
                "contract, generation parameters, model, provider, and xhigh/no-mode "
                "route succeeded on a single replay. The original 400 envelope also "
                "matches Heat v3, whose identical request identity succeeded in "
                "Heat v4. This rules out a deterministic request-contract defect and "
                "confirms a transient route rejection for this request."
            ),
            "insight_count": len(generated.insights),
            "insights": [
                {
                    "content": insight.content_record(),
                    "content_sha256": insight.content_sha256,
                    "hypothesis_sha256": insight.hypothesis_sha256,
                    "epistemic_status": "unverified_supplemental_hypothesis",
                    "lifecycle_status": "quarantined",
                }
                for insight in generated.insights
            ],
            "telemetry": _telemetry_record(telemetry),
            "queue_before_close": _queue_snapshot_record(snapshot),
            "required_evidence_counts": {
                "request": 1,
                "output": 1,
                "outcome": 1,
                "outbound": 1,
            },
            "comparison_evidence": reconstructed.verification[
                "comparison_evidence"
            ],
        }
        status = "completed"
    except BaseException as error:  # Preserve a finalized diagnostic on all failures.
        failure = error
    finally:
        if runner is not None:
            try:
                lifecycle.append(
                    {
                        "event": "runner_close_started",
                        "at_utc": _utc_now(),
                        "queue": _queue_snapshot_record(await runner.snapshot()),
                    }
                )
                await runner.aclose()
                closed_snapshot = await runner.snapshot()
                lifecycle.append(
                    {
                        "event": "runner_closed",
                        "at_utc": _utc_now(),
                        "queue": _queue_snapshot_record(closed_snapshot),
                    }
                )
                if (
                    not closed_snapshot.closed
                    or closed_snapshot.pending != 0
                    or closed_snapshot.in_flight != 0
                ):
                    raise RuntimeError("owned recovery runner did not close cleanly")
                if result_record:
                    result_record["queue_after_close"] = _queue_snapshot_record(
                        closed_snapshot
                    )
            except BaseException as close_error:
                status = "failed"
                failure = failure or close_error
        try:
            source_after = _same_source_identity(spec)
            write_json_atomic(
                spec.output_run / "source_verification_after.json",
                source_after,
            )
            if result_record:
                result_record["source_verification_after"] = source_after
        except BaseException as source_error:
            status = "failed"
            failure = failure or source_error

        if status == "completed" and failure is None:
            write_json_atomic(spec.output_run / "result.json", result_record)
        else:
            failed_record = {
                "schema_version": 2,
                "status": "failed",
                "failure_type": (
                    "UnknownFailure" if failure is None else type(failure).__name__
                ),
                "safe_message": (
                    "sealed reflection recovery failed; inspect sanitized required "
                    "evidence without repeating the provider call"
                ),
                "failed_at_utc": _utc_now(),
                "call_id": call_id.value,
                "source_call_id": spec.source_call_id,
                "logical_recovery_calls_max": 1,
                "provider_attempts_max": 1,
                "source_run_mutated": False,
                "required_evidence_counts": {
                    "request": len(observed_requests),
                    "output": len(observed_outputs),
                    "outcome": len(observed_outcomes),
                    "outbound": len(observed_outbound),
                },
            }
            write_json_atomic(spec.output_run / "failed.json", failed_record)
        for journal in journals:
            journal.close()
        finalization = finalize_run_directory(spec.output_run, status=status)
        result_record = {
            **(result_record if status == "completed" else failed_record),
            "finalization_sha256": finalization["finalization_sha256"],
            "recursive_content_sha256": finalization[
                "recursive_content_sha256"
            ],
            "recursive_file_count": finalization["recursive_file_count"],
        }
    return status, result_record


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--live", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify:
        reconstructed = reconstruct()
        request = recovery_request_record(
            reconstructed,
            call_id=_new_call_id(),
        )
        print(
            canonical_json_bytes(
                {
                    **reconstructed.verification,
                    "recovery_call_id": request["replay_provider_request"][
                        "call_id"
                    ],
                    "recovery_request_sha256": request[
                        "recovery_request_sha256"
                    ],
                    "provider": _config().to_manifest_record(),
                    "credential_read": False,
                    "provider_calls": 0,
                }
            ).decode("ascii"),
            flush=True,
        )
        return 0
    status, result = asyncio.run(_live())
    print(
        canonical_json_bytes(
            {
                "run_dir": str(SPEC.output_run),
                "status": status,
                "call_id": result.get("call_id"),
                "provider_attempts": result.get("provider_attempts", 0),
                "transient_route_failure_confirmed": result.get(
                    "transient_route_failure_confirmed", False
                ),
                "finalization_sha256": result.get("finalization_sha256"),
                "recursive_content_sha256": result.get(
                    "recursive_content_sha256"
                ),
            }
        ).decode("ascii"),
        flush=True,
    )
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
