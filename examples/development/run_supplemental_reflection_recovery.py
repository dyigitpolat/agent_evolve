#!/usr/bin/env python3
"""Recover one finalized-run reflection as a separate quarantined artifact.

The offline ``--verify`` path authenticates and rebuilds the failed request.
The credential-reading ``--live`` path is deliberately fixed to one output
directory and one supplemental call.  It never opens the source run for write.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.settings import load_credentials  # noqa: E402

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
    / "boils_abc/portfolio_q"
)
SOURCE_RUN = ARTIFACT_ROOT / "boilsq_repaired_v2_live_deepseek_g6_20260716"
OUTPUT_RUN = (
    ARTIFACT_ROOT
    / "boilsq_repaired_v2_deepseek_g6_cycle3_reflection_recovery_20260716"
)

SOURCE_FINALIZATION_SHA256 = (
    "6e609f7ea744569bfec49a9e8ce1ae3b73c541e4582455b71b996f23bfb9b4fb"
)
SOURCE_RECURSIVE_SHA256 = (
    "498c26bff3fd7073f7196f3e8ea3677f40483e7a2e63583d54ca96ea4cd6affc"
)
SOURCE_CALL_ID = "call_boilsq_40aa88cdbedd9789_000009"
SEMANTIC_PROMPT_SHA256 = (
    "3808356e69c7a62bc1c72b299e05c8cd213a1c81fbaa3a85fad4914ba4ca87e7"
)
SOURCE_REQUEST_SHA256 = (
    "ecfeb773507a4d43a7d2c0a2be4947d65cd4020afbd6af4d584e7d4f876476a4"
)
CONTRACT_SHA256 = (
    "d8d29f9dd939560bb203ea9d9260a78b9c50103fa128b6a29eae51b69d957f48"
)
SOURCE_WIRE_PROMPT_SHA256 = (
    "2520288521d877cd7dff62128db0b9f32abb5fbfbdad0986533efc17d199b368"
)

MODEL = "deepseek/deepseek-v4-pro"
PROVIDER_ONLY = ("streamlake",)
RESOLVED_PROVIDER = "StreamLake"
MAX_OUTPUT_TOKENS = 384_000
MAX_INSIGHTS = 2
MIN_INSIGHTS = 1
TEMPERATURE = 0.2
MAX_ATTEMPTS = 3
FIRST_EVENT_TIMEOUT_NS = 300_000_000_000
IDLE_TIMEOUT_NS = 300_000_000_000
CLEANUP_TIMEOUT_NS = 5_000_000_000
CONNECT_TIMEOUT_SECONDS = 90.0
BASE_BACKOFF_NS = 1_000_000_000
MAX_BACKOFF_NS = 30_000_000_000
JITTER_SEED = 20_260_716_195
JITTER_DOMAIN = "boils-g6-terminal-reflection-recovery-v1"
REQUEST_DOMAIN = b"agent-evolve:supplemental-reflection-recovery-request:v1\x00"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _new_call_id() -> LLMCallId:
    digest = _sha_text(f"{OUTPUT_RUN.name}\x00{SOURCE_CALL_ID}")[:24]
    value = LLMCallId(f"call_supplemental_{digest}_000001")
    if value.value == SOURCE_CALL_ID:
        raise RuntimeError("supplemental call identity collides with its source")
    return value


def _exact_object(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise RuntimeError(f"{label} is not an exact object")
    return value


def _exact_list(value: object, *, label: str) -> list[object]:
    if type(value) is not list:
        raise RuntimeError(f"{label} is not an exact list")
    return value


def _load_source() -> tuple[
    str,
    ReflectionInsightContract,
    ReflectionCallRequest,
    dict[str, object],
]:
    finalization = verify_finalized_run_directory(SOURCE_RUN)
    if (
        finalization.get("finalization_sha256") != SOURCE_FINALIZATION_SHA256
        or finalization.get("recursive_content_sha256") != SOURCE_RECURSIVE_SHA256
        or finalization.get("status") != "completed_unhealthy"
    ):
        raise RuntimeError("source finalization differs from the preregistration")

    engine_rows = read_jsonl(SOURCE_RUN / "engine_events.jsonl")
    requested = [
        row
        for row in engine_rows
        if row.get("event_type") == "reflection_requested"
        and row.get("call_id") == SOURCE_CALL_ID
    ]
    if len(requested) != 1:
        raise RuntimeError("source has no unique failed reflection request event")
    event = requested[0]
    prompt = event.get("prompt")
    if type(prompt) is not str or _sha_text(prompt) != SEMANTIC_PROMPT_SHA256:
        raise RuntimeError("source semantic prompt is unavailable or changed")
    if _sha_text(render_reflection_prompt(prompt)) != SOURCE_WIRE_PROMPT_SHA256:
        raise RuntimeError("source reflection renderer no longer matches the wire hash")

    raw_contract = _exact_object(
        event.get("insight_contract"),
        label="source insight contract",
    )
    metric_ids = _exact_list(
        raw_contract.get("required_metric_ids"),
        label="source metric vocabulary",
    )
    families = _exact_list(
        raw_contract.get("allowed_option_families"),
        label="source option-family vocabulary",
    )
    option_ids = _exact_list(
        raw_contract.get("allowed_option_ids"),
        label="source option vocabulary",
    )
    if not all(type(value) is str for value in (*metric_ids, *families, *option_ids)):
        raise RuntimeError("source reflection vocabulary contains a non-string")
    contract = ReflectionInsightContract(
        required_metric_ids=tuple(metric_ids),
        allowed_option_families=tuple(families),
        allowed_option_ids=tuple(option_ids),
    )
    if contract.to_record() != raw_contract or contract.identity_sha256 != CONTRACT_SHA256:
        raise RuntimeError("source reflection contract does not authenticate")

    planner_rows = read_jsonl(SOURCE_RUN / "planner_events.jsonl")
    failures = [
        row
        for row in planner_rows
        if row.get("event_type") == "reflection_failed"
        and _exact_object(row.get("receipt"), label="planner receipt").get("call_id")
        == SOURCE_CALL_ID
    ]
    if len(failures) != 1:
        raise RuntimeError("source has no unique failed planner receipt")
    receipt = _exact_object(failures[0].get("receipt"), label="planner receipt")
    raw_request = _exact_object(receipt.get("request"), label="planner request")
    predecessors: list[InsightRef] = []
    for raw in _exact_list(
        raw_request.get("revision_predecessors"),
        label="revision predecessors",
    ):
        item = _exact_object(raw, label="revision predecessor")
        predecessors.append(
            InsightRef(
                InsightId(str(item.get("insight_id"))),
                int(item.get("version")),
            )
        )
    temperature_hex = raw_request.get("temperature_hex")
    temperature = (
        None
        if temperature_hex is None
        else float.fromhex(str(temperature_hex))
    )
    source_request = ReflectionCallRequest(
        label=str(raw_request.get("label")),
        operation=str(raw_request.get("operation")),
        prompt_sha256=str(raw_request.get("prompt_sha256")),
        min_insights=int(raw_request.get("min_insights")),
        max_insights=int(raw_request.get("max_insights")),
        max_output_tokens=int(raw_request.get("max_output_tokens")),
        temperature=temperature,
        insight_contract_sha256=(
            None
            if raw_request.get("insight_contract_sha256") is None
            else str(raw_request.get("insight_contract_sha256"))
        ),
        revision_predecessors=tuple(predecessors),
        revision_predecessor_content_sha256s=tuple(
            str(value)
            for value in _exact_list(
                raw_request.get("revision_predecessor_content_sha256s"),
                label="revision predecessor hashes",
            )
        ),
        source_receipt_sha256s=tuple(
            str(value)
            for value in _exact_list(
                raw_request.get("source_receipt_sha256s"),
                label="source receipt hashes",
            )
        ),
        source_operator_invocation_ids=tuple(
            OperatorInvocationId(str(value))
            for value in _exact_list(
                raw_request.get("source_operator_invocation_ids"),
                label="source operator IDs",
            )
        ),
        source_outcome_sha256s=tuple(
            str(value)
            for value in _exact_list(
                raw_request.get("source_outcome_sha256s"),
                label="source outcome hashes",
            )
        ),
        available_contrast_ids=tuple(
            str(value)
            for value in _exact_list(
                raw_request.get("available_contrast_ids"),
                label="source contrast IDs",
            )
        ),
        request_sha256=str(raw_request.get("request_sha256")),
    )
    reconstructed = {
        **source_request.to_record(),
        "request_sha256": source_request.request_sha256,
    }
    if reconstructed != raw_request or source_request.request_sha256 != SOURCE_REQUEST_SHA256:
        raise RuntimeError("source planner request does not authenticate exactly")
    if (
        event.get("prompt_sha256") != source_request.prompt_sha256
        or event.get("reflection_request_sha256") != source_request.request_sha256
        or event.get("available_contrast_ids")
        != list(source_request.available_contrast_ids)
        or event.get("source_receipt_sha256s")
        != list(source_request.source_receipt_sha256s)
        or len(source_request.available_contrast_ids) != 16
        or source_request.insight_contract_sha256 != contract.identity_sha256
        or source_request.min_insights != MIN_INSIGHTS
        or source_request.max_insights != MAX_INSIGHTS
        or source_request.max_output_tokens != MAX_OUTPUT_TOKENS
        or source_request.temperature != TEMPERATURE
    ):
        raise RuntimeError("engine event, planner request, and contract do not join")

    outcomes = [
        row
        for row in read_jsonl(SOURCE_RUN / "queue_outcomes.jsonl")
        if row.get("task_id") == SOURCE_CALL_ID
    ]
    if len(outcomes) != 1:
        raise RuntimeError("source has no unique terminal queue outcome")
    source_outcome = outcomes[0]
    attempts = _exact_list(source_outcome.get("attempts"), label="source attempts")
    if len(attempts) != 1:
        raise RuntimeError("source reflection had an unexpected attempt count")
    attempt = _exact_object(attempts[0], label="source attempt")
    attempt_request = _exact_object(
        attempt.get("request_evidence"),
        label="source attempt request evidence",
    )
    if (
        source_outcome.get("status") != "terminal_failure"
        or attempt.get("status") != "timed_out"
        or attempt_request.get("prompt_sha256") != SOURCE_WIRE_PROMPT_SHA256
    ):
        raise RuntimeError("source terminal failure differs from the recovery premise")

    verification = {
        "schema_version": 1,
        "verified_at_utc": _utc_now(),
        "source_run": SOURCE_RUN.name,
        "source_status": finalization["status"],
        "source_finalization_sha256": SOURCE_FINALIZATION_SHA256,
        "source_recursive_content_sha256": SOURCE_RECURSIVE_SHA256,
        "source_call_id": SOURCE_CALL_ID,
        "semantic_prompt_sha256": SEMANTIC_PROMPT_SHA256,
        "wire_prompt_sha256": SOURCE_WIRE_PROMPT_SHA256,
        "source_request_sha256": SOURCE_REQUEST_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "available_contrast_count": len(source_request.available_contrast_ids),
        "prompt_utf8_bytes": len(prompt.encode("utf-8", errors="strict")),
        "source_join_verified": True,
    }
    return prompt, contract, source_request, verification


def _request_record(
    *,
    call_id: LLMCallId,
    prompt: str,
    contract: ReflectionInsightContract,
    source_request: ReflectionCallRequest,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "status": "supplemental_quarantined",
        "lifecycle_publication_allowed": False,
        "call_id": call_id.value,
        "source_call_id": SOURCE_CALL_ID,
        "source_request_sha256": source_request.request_sha256,
        "operation": source_request.operation,
        "prompt": prompt,
        "prompt_sha256": _sha_text(prompt),
        "wire_prompt_sha256": _sha_text(render_reflection_prompt(prompt)),
        "available_contrast_ids": list(source_request.available_contrast_ids),
        "min_insights": MIN_INSIGHTS,
        "max_insights": MAX_INSIGHTS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature_hex": TEMPERATURE.hex(),
        "insight_contract": contract.to_record(),
    }
    record["recovery_request_sha256"] = hashlib.sha256(
        REQUEST_DOMAIN + canonical_json_bytes(record)
    ).hexdigest()
    return record


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


def _queue_snapshot_record(value: object) -> dict[str, object]:
    return {
        "max_in_flight": getattr(value, "max_in_flight"),
        "max_pending": getattr(value, "max_pending"),
        "in_flight": getattr(value, "in_flight"),
        "pending": getattr(value, "pending"),
        "closed": getattr(value, "closed"),
    }


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
        app_title="AgentEvolve supplemental reflection recovery",
        reasoning_config=OpenRouterReasoningConfig(effort="xhigh"),
        retry_mode=ProgressAwareRetryMode.NON_REPEATING_STREAM,
    )


async def _live() -> tuple[str, dict[str, object]]:
    prompt, contract, source_request, verification = _load_source()
    call_id = _new_call_id()
    request_record = _request_record(
        call_id=call_id,
        prompt=prompt,
        contract=contract,
        source_request=source_request,
    )
    config = _config()
    OUTPUT_RUN.mkdir(parents=True, exist_ok=False)
    write_json_atomic(OUTPUT_RUN / "source_verification.json", verification)
    write_json_atomic(OUTPUT_RUN / "recovery_request.json", request_record)
    write_json_atomic(
        OUTPUT_RUN / "manifest.json",
        {
            "schema_version": 1,
            "created_at_utc": _utc_now(),
            "run_id": OUTPUT_RUN.name,
            "claim_boundary": {
                "supplemental_only": True,
                "quarantined": True,
                "mutates_source_run": False,
                "lifecycle_publication_allowed": False,
                "optimization_claim": False,
            },
            "source_verification": verification,
            "recovery_request_sha256": request_record["recovery_request_sha256"],
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
        OUTPUT_RUN / "stream_progress.jsonl",
        max_unfsynced_rows=32,
    )
    requests = DurableJsonlJournal(OUTPUT_RUN / "request_evidence.jsonl")
    outputs = DurableJsonlJournal(OUTPUT_RUN / "output_evidence.jsonl")
    outcomes = DurableJsonlJournal(OUTPUT_RUN / "queue_outcomes.jsonl")
    outbound = DurableJsonlJournal(OUTPUT_RUN / "outbound_requests.jsonl")
    lifecycle = DurableJsonlJournal(OUTPUT_RUN / "lifecycle.jsonl")
    journals = (progress, requests, outputs, outcomes, outbound, lifecycle)
    outcome_count = 0
    request_count = 0
    output_count = 0
    outbound_count = 0
    runner = None
    status = "failed"
    summary: dict[str, object] = {}

    def progress_sink(value: StructuredStreamProgress) -> None:
        progress.append(_progress_record(value))

    def request_sink(value: dict[str, object]) -> None:
        nonlocal request_count
        requests.append(value)
        request_count += 1

    def output_sink(value: dict[str, object]) -> None:
        nonlocal output_count
        outputs.append(value)
        output_count += 1

    def outcome_sink(value: object) -> None:
        nonlocal outcome_count
        progress.flush()
        outcomes.append(structured_generation_outcome_record(value))
        outcome_count += 1

    def outbound_sink(value: dict[str, object]) -> None:
        nonlocal outbound_count
        outbound.append(value)
        outbound_count += 1

    try:
        load_credentials(WORKSPACE_ROOT / ".env", override=False, optional=True)
        load_credentials(AGENT_EVOLVE_ROOT / ".env", override=False, optional=True)
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
        generator = PydanticAIAgenticGenerator(runner)
        result = await generator.reflect(
            ReflectionGenerationRequest(
                call_id=call_id,
                operation=source_request.operation,
                prompt=prompt,
                max_insights=MAX_INSIGHTS,
                min_insights=MIN_INSIGHTS,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
                available_contrast_ids=source_request.available_contrast_ids,
                insight_contract=contract,
            )
        )
        if not MIN_INSIGHTS <= len(result.insights) <= MAX_INSIGHTS:
            raise RuntimeError("provider result violates the frozen insight count")
        for insight in result.insights:
            validate_reflection_insight_draft(insight, contract)
            if not set(insight.evidence_contrast_ids).issubset(
                source_request.available_contrast_ids
            ):
                raise RuntimeError("provider result cites a foreign contrast")
        telemetry = result.telemetry
        if (
            telemetry.requested_model != MODEL
            or telemetry.resolved_model != MODEL
            or telemetry.resolved_provider != RESOLVED_PROVIDER
            or telemetry.reasoning_tokens <= 0
        ):
            raise RuntimeError("successful response violates provider/reasoning gates")
        snapshot = await runner.snapshot()
        if snapshot.pending != 0 or snapshot.in_flight != 0:
            raise RuntimeError("queue did not drain after the terminal response")
        if not (
            outcome_count == request_count == output_count == outbound_count == 1
        ):
            raise RuntimeError("required provider evidence is incomplete or duplicated")
        summary = {
            "schema_version": 1,
            "status": "completed",
            "epistemic_status": "supplemental_quarantined",
            "lifecycle_publication_count": 0,
            "source_run_mutated": False,
            "call_id": call_id.value,
            "source_call_id": SOURCE_CALL_ID,
            "recovery_request_sha256": request_record["recovery_request_sha256"],
            "insight_count": len(result.insights),
            "insights": [
                {
                    "content": insight.content_record(),
                    "content_sha256": insight.content_sha256,
                    "hypothesis_sha256": insight.hypothesis_sha256,
                    "epistemic_status": "unverified_supplemental_hypothesis",
                    "lifecycle_status": "quarantined",
                }
                for insight in result.insights
            ],
            "telemetry": _telemetry_record(telemetry),
            "queue_before_close": _queue_snapshot_record(snapshot),
            "required_evidence_counts": {
                "request": request_count,
                "output": output_count,
                "outcome": outcome_count,
                "outbound": outbound_count,
            },
        }
        write_json_atomic(OUTPUT_RUN / "result.json", summary)
        status = "completed"
    except Exception as error:
        summary = {
            "schema_version": 1,
            "status": "failed",
            "failure_type": type(error).__name__,
            "safe_message": "supplemental reflection recovery failed; inspect sanitized required evidence",
            "failed_at_utc": _utc_now(),
            "call_id": call_id.value,
            "source_call_id": SOURCE_CALL_ID,
            "epistemic_status": "no_recovered_insight",
            "lifecycle_publication_count": 0,
            "source_run_mutated": False,
            "required_evidence_counts": {
                "request": request_count,
                "output": output_count,
                "outcome": outcome_count,
                "outbound": outbound_count,
            },
        }
        write_json_atomic(OUTPUT_RUN / "failed.json", summary)
    finally:
        if runner is not None:
            before_close_record: dict[str, object] | None = None
            try:
                before_close = await runner.snapshot()
                before_close_record = _queue_snapshot_record(before_close)
                lifecycle.append(
                    {
                        "event": "runner_close_started",
                        "at_utc": _utc_now(),
                        "queue": before_close_record,
                    }
                )
            except Exception as snapshot_error:
                lifecycle.append(
                    {
                        "event": "runner_preclose_snapshot_failed",
                        "at_utc": _utc_now(),
                        "failure_type": type(snapshot_error).__name__,
                    }
                )
            try:
                await runner.aclose()
                after_close = await runner.snapshot()
                lifecycle.append(
                    {
                        "event": "runner_closed",
                        "at_utc": _utc_now(),
                        "queue": _queue_snapshot_record(after_close),
                    }
                )
            except Exception as close_error:
                lifecycle.append(
                    {
                        "event": "runner_close_failed",
                        "at_utc": _utc_now(),
                        "failure_type": type(close_error).__name__,
                    }
                )
                status = "failed"
                if not (OUTPUT_RUN / "failed.json").exists():
                    summary = {
                        "schema_version": 1,
                        "status": "failed",
                        "failure_type": type(close_error).__name__,
                        "safe_message": "owned queue or transport did not close cleanly",
                        "failed_at_utc": _utc_now(),
                        "call_id": call_id.value,
                        "source_call_id": SOURCE_CALL_ID,
                        "lifecycle_publication_count": 0,
                        "source_run_mutated": False,
                    }
                    write_json_atomic(OUTPUT_RUN / "failed.json", summary)
        for journal in journals:
            journal.close()
        finalization = finalize_run_directory(OUTPUT_RUN, status=status)
        summary = {
            **summary,
            "finalization_sha256": finalization["finalization_sha256"],
            "recursive_content_sha256": finalization["recursive_content_sha256"],
            "recursive_file_count": finalization["recursive_file_count"],
        }
    return status, summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--live", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify:
        prompt, contract, source_request, verification = _load_source()
        request = _request_record(
            call_id=_new_call_id(),
            prompt=prompt,
            contract=contract,
            source_request=source_request,
        )
        print(
            json.dumps(
                {
                    **verification,
                    "new_call_id": request["call_id"],
                    "recovery_request_sha256": request["recovery_request_sha256"],
                    "provider": _config().to_manifest_record(),
                },
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    status, summary = asyncio.run(_live())
    print(
        json.dumps(
            {
                "run_dir": str(OUTPUT_RUN),
                "status": status,
                "call_id": summary.get("call_id"),
                "insight_count": summary.get("insight_count", 0),
                "finalization_sha256": summary.get("finalization_sha256"),
                "recursive_content_sha256": summary.get(
                    "recursive_content_sha256"
                ),
            },
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
