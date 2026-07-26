#!/usr/bin/env python3
"""Two-call live conformance canary for any registered model profile.

The calls exercise two production-shaped structured schemas concurrently: an
eight-member portfolio proposal and a two-insight reflection.  They never
materialize a candidate or invoke a workload evaluator.  Every route,
reasoning, retry, request, typed output, progress, cost, and cleanup record is
durably retained before a profile is admitted to a full evolutionary run.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Literal


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, model_validator  # noqa: E402

from agent_evolve.domain.ids import LLMCallId  # noqa: E402
from agent_evolve.integrations.pydantic_ai.model_execution_profile import (  # noqa: E402
    OpenRouterModelExecutionProfile,
    openrouter_model_execution_profile,
)
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (  # noqa: E402
    validate_openrouter_outbound_request_manifest_record,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (  # noqa: E402
    ProgressAwareOpenRouterConfig,
    ProgressAwareRetryMode,
    create_progress_aware_openrouter_runner,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    QueuedStructuredGenerationError,
    StructuredEvidencePublicationPolicy,
    structured_generation_outcome_record,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry  # noqa: E402
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    StructuredStreamCleanupPolicy,
    StructuredStreamLivenessPolicy,
    StructuredStreamProgress,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    DurableJsonlJournal,
    finalize_run_directory,
    read_jsonl,
    source_identity,
    write_json_atomic,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs/"
    "model_profile_canary"
)
AUTHORIZATION = "RUN_TWO_MODEL_PROFILE_CANARY_CALLS"
_SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_PORTFOLIO_OPTION_IDS = tuple(
    f"option_{index:03d}_finite_candidate" for index in range(1, 201)
)
_PORTFOLIO_REQUIRED_OPTION_IDS = tuple(_PORTFOLIO_OPTION_IDS[:8])
_PortfolioOptionId = Literal.__getitem__(_PORTFOLIO_OPTION_IDS)


class _Forecast(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    metric_id: Literal["objective_a", "objective_b"]
    direction: Literal["improve", "worsen", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)


class _PortfolioMember(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    rank: int = Field(ge=1, le=8)
    option_id: _PortfolioOptionId
    forecasts: list[_Forecast] = Field(min_length=2, max_length=2)
    rationale: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def _exact_metrics(self) -> "_PortfolioMember":
        if {value.metric_id for value in self.forecasts} != {
            "objective_a",
            "objective_b",
        }:
            raise ValueError("each member must forecast both objectives")
        return self


class PortfolioCanaryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    members: list[_PortfolioMember] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def _exact_members(self) -> "PortfolioCanaryOutput":
        if [value.rank for value in self.members] != list(range(1, 9)):
            raise ValueError("members must be in exact rank order")
        if tuple(value.option_id for value in self.members) != (
            _PORTFOLIO_REQUIRED_OPTION_IDS
        ):
            raise ValueError("members must cover the exact option panel")
        return self


class _ReflectionInsight(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    insight_id: str = Field(pattern=r"^hypothesis_0[12]$")
    statement: str = Field(min_length=1, max_length=768)
    evidence_ids: list[str] = Field(min_length=1, max_length=2)
    expected_effect: Literal["objective_a", "objective_b", "tradeoff"]


class ReflectionCanaryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    insights: list[_ReflectionInsight] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def _exact_insights(self) -> "ReflectionCanaryOutput":
        if [value.insight_id for value in self.insights] != [
            "hypothesis_01",
            "hypothesis_02",
        ]:
            raise ValueError("reflection must cover the exact hypothesis panel")
        available = {"contrast_01", "contrast_02", "contrast_03", "contrast_04"}
        if any(not set(value.evidence_ids) <= available for value in self.insights):
            raise ValueError("reflection cited foreign evidence")
        return self


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_record() -> dict[str, object]:
    paths = (
        Path(__file__).resolve(),
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/model_execution_profile.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/progress_aware_openrouter.py",
        AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/queued_runner.py",
    )
    return source_identity(paths, relative_to=WORKSPACE_ROOT)


def _config(profile: OpenRouterModelExecutionProfile, seed: int) -> ProgressAwareOpenRouterConfig:
    return ProgressAwareOpenRouterConfig(
        model_name=profile.requested_model,
        provider_only=profile.provider_only,
        connect_timeout_seconds=90.0,
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=300_000_000_000,
            idle_timeout_ns=300_000_000_000,
            # A conformance canary must not become an unbounded generation.
            # This is a generous wall guard, not an output-token truncation;
            # every profile retains its provider-scale completion ceiling.
            absolute_timeout_ns=900_000_000_000,
            cleanup_policy=StructuredStreamCleanupPolicy(
                cancel_drain_timeout_ns=5_000_000_000,
                transport_retire_timeout_ns=5_000_000_000,
            ),
        ),
        max_connections=profile.effective_max_connections(default=2),
        max_pending=2,
        max_attempts=3,
        base_backoff_ns=1_000_000_000,
        max_backoff_ns=30_000_000_000,
        jitter_seed=seed,
        jitter_domain=f"model-profile-canary-{profile.profile_id}",
        app_title="AgentEvolve AAAI 2027 model profile canary",
        reasoning_config=profile.reasoning_config,
        structured_output_mode=profile.structured_output_mode,
        structured_output_strict=profile.structured_output_strict,
        json_schema_dialect=profile.json_schema_dialect,
        provider_require_parameters=profile.provider_require_parameters,
        supports_forced_tool_choice=profile.supports_forced_tool_choice,
        retry_mode=(
            ProgressAwareRetryMode.OPAQUE_HTTP_400_AND_BOUNDED_SCHEMA_REPAIR
        ),
    )


def _requests(
    *,
    run_id: str,
    profile: OpenRouterModelExecutionProfile,
) -> tuple[
    StructuredGenerationRequest[PortfolioCanaryOutput],
    StructuredGenerationRequest[ReflectionCanaryOutput],
]:
    digest = hashlib.sha256(run_id.encode("ascii")).hexdigest()[:12]
    required_option_ids = ", ".join(_PORTFOLIO_REQUIRED_OPTION_IDS)
    selector = StructuredGenerationRequest(
        call_id=LLMCallId(f"call_profile_canary_{digest}_selector"),
        operation="profile_canary_portfolio",
        prompt=(
            "Return one typed eight-member optimization portfolio. Use ranks 1 "
            "through 8 in order and use these option IDs in the same order: "
            f"{required_option_ids}. The output schema contains the complete "
            "bounded 200-option finite contract. For every member forecast objective_a and "
            "objective_b exactly once, choosing improve, worsen, or unknown and "
            "a confidence in [0,1]. Add a short rationale. This is a transport "
            "canary; do not add fields or prose outside the structured result."
        ),
        output_type=PortfolioCanaryOutput,
        output_tool_name="return_profile_canary_portfolio",
        max_output_tokens=profile.max_output_tokens,
        temperature=profile.temperature,
    )
    reflection = StructuredGenerationRequest(
        call_id=LLMCallId(f"call_profile_canary_{digest}_reflection"),
        operation="profile_canary_reflection",
        prompt=(
            "Return exactly two typed hypotheses in order: hypothesis_01 and "
            "hypothesis_02. Cite one or two IDs only from contrast_01, "
            "contrast_02, contrast_03, contrast_04. State a concise reusable "
            "hypothesis and label its expected effect objective_a, objective_b, "
            "or tradeoff. This is a transport canary; do not add fields or prose "
            "outside the structured result."
        ),
        output_type=ReflectionCanaryOutput,
        output_tool_name="return_profile_canary_reflection",
        max_output_tokens=profile.max_output_tokens,
        temperature=profile.temperature,
    )
    return selector, reflection


def _progress_record(value: StructuredStreamProgress) -> dict[str, object]:
    if type(value) is not StructuredStreamProgress:
        raise TypeError("progress sink received a foreign record")
    StructuredStreamProgress.__post_init__(value)
    return {
        "call_id": value.call_id,
        "sequence": value.sequence,
        "kind": value.kind.value,
        "channel": value.channel.value,
        "elapsed_ns": value.elapsed_ns,
        "event_content_utf8_bytes": value.event_content_utf8_bytes,
        "cumulative_content_utf8_bytes": value.cumulative_content_utf8_bytes,
        "rolling_content_sha256": value.rolling_content_sha256,
        "provider_attempt_id": value.provider_attempt_id,
    }


def _telemetry(
    response: StructuredGenerationResponse[object],
    *,
    attempt_count: int,
) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model=response.requested_model,
        resolved_model=response.resolved_model,
        resolved_provider=response.resolved_provider,
        provider_response_id=response.provider_response_id,
        finish_reason=response.finish_reason,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        reasoning_tokens=response.reasoning_tokens,
        cache_read_tokens=response.cache_read_tokens,
        cache_write_tokens=response.cache_write_tokens,
        cost_usd=response.cost_usd,
        latency_ns=response.latency_ns,
        attempt_count=attempt_count,
    )


async def _live(
    *,
    run_dir: Path,
    run_id: str,
    profile: OpenRouterModelExecutionProfile,
    seed: int,
) -> dict[str, object]:
    # The research workspace owns shared provider credentials; a submodule-local
    # file remains an optional developer override source.  Neither path nor any
    # credential value crosses the durable evidence boundary.
    load_dotenv(WORKSPACE_ROOT / ".env", override=False)
    load_dotenv(AGENT_EVOLVE_ROOT / ".env", override=False)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if type(api_key) is not str or not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")
    journals = {
        name: DurableJsonlJournal(run_dir / filename)
        for name, filename in (
            ("progress", "stream_progress.jsonl"),
            ("outcome", "queue_outcomes.jsonl"),
            ("request", "request_evidence.jsonl"),
            ("output", "output_evidence.jsonl"),
            ("outbound", "outbound_requests.jsonl"),
        )
    }
    runner = create_progress_aware_openrouter_runner(
        api_key=api_key,
        config=_config(profile, seed),
        progress_sink=lambda value: journals["progress"].append(
            _progress_record(value)
        ),
        outcome_sink=lambda value: journals["outcome"].append(
            structured_generation_outcome_record(value)
        ),
        request_evidence_sink=journals["request"].append,
        output_evidence_sink=journals["output"].append,
        outbound_request_manifest_sink=journals["outbound"].append,
        evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
    )
    requests = _requests(run_id=run_id, profile=profile)
    try:
        attempted = await asyncio.gather(
            *(runner(request) for request in requests),
            return_exceptions=True,
        )
        snapshot_before_close = await runner.snapshot()
    finally:
        await runner.aclose()
        for journal in journals.values():
            journal.close()

    responses = []
    failures = []
    for request, result in zip(requests, attempted, strict=True):
        if isinstance(result, BaseException):
            failures.append(
                {
                    "call_id": request.call_id.value,
                    "operation": request.operation,
                    "failure_type": type(result).__qualname__,
                    "failure_sha256": hashlib.sha256(
                        f"{type(result).__qualname__}\x00{result}".encode(
                            "utf-8", errors="replace"
                        )
                    ).hexdigest(),
                    "queue_status": (
                        result.status.value
                        if isinstance(result, QueuedStructuredGenerationError)
                        else None
                    ),
                    "failure_disposition": (
                        result.generation_failure_disposition.value
                        if isinstance(result, QueuedStructuredGenerationError)
                        else None
                    ),
                }
            )
            continue
        response = result.response
        if type(response) is not StructuredGenerationResponse:
            raise RuntimeError("canary call did not return a typed response")
        if type(response.value) is not request.output_type:
            raise RuntimeError("canary typed output differs from its schema")
        telemetry = _telemetry(response, attempt_count=result.attempt_count)
        profile.validate_telemetry(telemetry)
        responses.append(
            {
                "call_id": request.call_id.value,
                "operation": request.operation,
                "attempt_count": result.attempt_count,
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
                "cost_usd": (
                    None if response.cost_usd is None else str(response.cost_usd)
                ),
                "latency_ns": response.latency_ns,
                "typed_output": response.value.model_dump(mode="json"),
            }
        )

    outbound = tuple(
        validate_openrouter_outbound_request_manifest_record(value)
        for value in read_jsonl(run_dir / "outbound_requests.jsonl")
    )
    expected_provider = _config(profile, seed).provider_options
    outbound_gate = bool(outbound) and all(
        value["settings"]["model"] == profile.requested_model
        and value["settings"]["provider"] == expected_provider
        and value["settings"]["reasoning"]
        == profile.outbound_reasoning_setting
        and value["settings"]["max_completion_tokens"]
        == profile.max_output_tokens
        for value in outbound
    )
    counts = {
        name: len(read_jsonl(run_dir / filename))
        for name, filename in (
            ("progress", "stream_progress.jsonl"),
            ("outcome", "queue_outcomes.jsonl"),
            ("request", "request_evidence.jsonl"),
            ("output", "output_evidence.jsonl"),
            ("outbound", "outbound_requests.jsonl"),
        )
    }
    health = {
        "two_typed_calls": len(responses) == 2,
        "portfolio_and_reflection_completed": {
            value["operation"] for value in responses
        }
        == {"profile_canary_portfolio", "profile_canary_reflection"},
        "exact_profile_telemetry": True,
        "outbound_profile_contract": outbound_gate,
        "durable_request_output_outcome_counts": (
            counts["request"] == counts["output"] == counts["outcome"] == 2
        ),
        "physical_attempts_cover_logical_calls": counts["outbound"] >= 2,
        "queue_drained_before_close": (
            getattr(snapshot_before_close, "in_flight", None) == 0
            and getattr(snapshot_before_close, "pending", None) == 0
        ),
    }
    return {
        "schema_version": 1,
        "status": "completed_healthy" if all(health.values()) else "completed_unhealthy",
        "profile": profile.to_record(),
        "health": health,
        "responses": responses,
        "failures": failures,
        "durable_counts": counts,
        "logical_provider_calls": 2,
        "physical_provider_attempts": counts["outbound"],
        "evaluator_calls": 0,
        "scientific_claim": "transport_and_typed_schema_conformance_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("readiness", "live"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=20260740)
    parser.add_argument("--authorization")
    args = parser.parse_args()
    if _SAFE_RUN_ID.fullmatch(args.run_id) is None:
        raise ValueError("run ID violates the closed canary grammar")
    profile = openrouter_model_execution_profile(args.profile)
    run_dir = (ARTIFACT_ROOT / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    source = _source_record()
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "mode": args.mode,
        "created_at_utc": _utc_now(),
        "profile": profile.to_record(),
        "config": _config(profile, args.seed).to_manifest_record(),
        "source": source,
        "planned_logical_calls": 2,
        "planned_evaluator_calls": 0,
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    try:
        if args.mode == "readiness":
            requests = _requests(run_id=args.run_id, profile=profile)
            summary = {
                "schema_version": 1,
                "status": "ready_without_credential_provider_or_evaluator",
                "profile": profile.to_record(),
                "request_schemas": [
                    {
                        "operation": value.operation,
                        "call_id": value.call_id.value,
                        "max_output_tokens": value.max_output_tokens,
                        "temperature": value.temperature,
                        "schema": value.output_type.model_json_schema(),
                    }
                    for value in requests
                ],
                "provider_calls": 0,
                "credential_read": False,
                "evaluator_calls": 0,
            }
        else:
            if args.authorization != AUTHORIZATION:
                raise RuntimeError("live authorization string is invalid")
            summary = asyncio.run(
                _live(
                    run_dir=run_dir,
                    run_id=args.run_id,
                    profile=profile,
                    seed=args.seed,
                )
            )
        write_json_atomic(run_dir / "summary.json", summary)
        finalize_run_directory(run_dir, status=str(summary["status"]))
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary["status"] != "completed_unhealthy" else 2
    except BaseException as error:
        if not (run_dir / "summary.json").exists():
            write_json_atomic(
                run_dir / "summary.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "failure_type": type(error).__qualname__,
                    "failure_sha256": hashlib.sha256(
                        f"{type(error).__qualname__}\x00{error}".encode(
                            "utf-8", errors="replace"
                        )
                    ).hexdigest(),
                },
            )
        if not (run_dir / "finalized.json").exists():
            finalize_run_directory(run_dir, status="failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
