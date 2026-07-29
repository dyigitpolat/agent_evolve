#!/usr/bin/env python3
"""Run the v6 closed-loop memory engineering-development probe.

This synthetic probe diagnoses orchestration only.  It is not a benchmark and
must not be cited as SOTA, generalization, or wall-clock evidence.  The default
mode is a provider-free readiness preview.  ``--offline`` executes the complete
mechanism with a deterministic generator; only the separately explicit
``--live`` mode may compose the frozen OpenRouter route.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.settings import load_credentials  # noqa: E402

from agent_evolve.application.agentic_evolution import (  # noqa: E402
    InvocationOutcome,
)
from agent_evolve.application.budgeted_optimizer import (  # noqa: E402
    OptimizerResult,
    OptimizerState,
    pareto_archive_snapshot_hash,
)
from agent_evolve.application.gated_agentic_generator import (  # noqa: E402
    AgenticTelemetryPolicy,
    TelemetryGatedAgenticGenerator,
)
from agent_evolve.application.insight_memory import InsightMemoryBank  # noqa: E402
from agent_evolve.domain.insight import InsightRef  # noqa: E402
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (  # noqa: E402
    OpenRouterReasoningConfig,
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    OutcomePublicationPolicy,
    SCHEMA_REPAIR_POLICY_MANIFEST,
    SchemaRepairAttemptPolicy,
    create_production_queued_runner,
    structured_generation_outcome_record,
)
from agent_evolve.policies.llm_backoff import (  # noqa: E402
    DeterministicHashJitter,
)
from agent_evolve.policies.memory.staged_causal import (  # noqa: E402
    MemoryCheckpointClosureStatus,
)
from agent_evolve.ports.agentic_generator import AgenticGenerator  # noqa: E402
from examples.development.v6_closed_loop_probe_support import (  # noqa: E402
    FULL_WAVE_WIDTH,
    MAX_OUTPUT_TOKENS,
    MODEL_WAVE_WIDTH,
    OPTIMIZER_BUDGET,
    ProbeComposition,
    canonical_record_sha256,
    closed_loop_reward,
    compose_probe,
    offline_generator_factory,
)


MODEL = "deepseek/deepseek-v4-pro"
CANONICAL_RESOLVED_MODEL = "deepseek/deepseek-v4-pro-20260423"
ALLOWED_RESOLVED_MODELS = (MODEL, CANONICAL_RESOLVED_MODEL)
PROVIDER_ONLY = ("streamlake",)
ALLOWED_RESOLVED_PROVIDERS = ("StreamLake",)
AUTHORIZED_LIVE_RUN_ID = "v6_closed_loop_live_attempt1_20260714"
PREREGISTRATION_READY_MARKER = "LIVE_EXECUTION_STATUS: `READY`"

QUEUE_MAX_IN_FLIGHT = FULL_WAVE_WIDTH
QUEUE_MAX_PENDING = 8
QUEUE_MAX_ATTEMPTS = 2
QUEUE_ATTEMPT_TIMEOUT_NS = 60_000_000_000
QUEUE_BASE_BACKOFF_NS = 1_000_000_000
QUEUE_MAX_BACKOFF_NS = 8_000_000_000
JITTER_SEED = 20_260_714
JITTER_DOMAIN = "v6-closed-loop-memory-jitter-v1"

MAX_INPUT_TOKENS = 8_000
MAX_REASONING_TOKENS = MAX_OUTPUT_TOKENS
MAX_CALL_COST_USD = Decimal("0.010")
MAX_RUN_COST_USD = MAX_CALL_COST_USD * Decimal(OPTIMIZER_BUDGET.max_logical_llm_calls)
MAX_POTENTIALLY_BILLABLE_ATTEMPT_COST_USD = Decimal("0.010")
MAX_POTENTIALLY_BILLABLE_RUN_COST_USD = (
    MAX_POTENTIALLY_BILLABLE_ATTEMPT_COST_USD
    * Decimal(OPTIMIZER_BUDGET.max_logical_llm_calls)
    * Decimal(QUEUE_MAX_ATTEMPTS)
)
PROMPT_PRICE_USD_PER_TOKEN = Decimal("0.0000007134")
COMPLETION_PRICE_USD_PER_TOKEN = Decimal("0.0000014268")
DERIVED_MAX_SUCCESSFUL_RESPONSE_COST_USD = (
    Decimal(MAX_INPUT_TOKENS) * PROMPT_PRICE_USD_PER_TOKEN
    + Decimal(MAX_OUTPUT_TOKENS + MAX_REASONING_TOKENS) * COMPLETION_PRICE_USD_PER_TOKEN
)
DERIVED_MAX_ACCEPTED_RUN_COST_USD = DERIVED_MAX_SUCCESSFUL_RESPONSE_COST_USD * Decimal(
    OPTIMIZER_BUDGET.max_logical_llm_calls
)


def run_async_sync(awaitable: Any) -> Any:
    """Run one async phase with evaluator wakeups and deterministic cleanup."""

    async def run_with_heartbeat() -> Any:
        execution = asyncio.create_task(awaitable)
        while not execution.done():
            await asyncio.sleep(0.01)
        return await execution

    loop = asyncio.new_event_loop()
    executor = ThreadPoolExecutor(
        max_workers=4,
        thread_name_prefix="v6_probe_evaluator",
    )
    loop.set_default_executor(executor)
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(run_with_heartbeat())
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        loop.close()
        asyncio.set_event_loop(None)
DERIVED_MAX_POTENTIALLY_BILLABLE_RUN_COST_USD = (
    DERIVED_MAX_ACCEPTED_RUN_COST_USD * Decimal(QUEUE_MAX_ATTEMPTS)
)

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")

ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
DEFAULT_LOG_ROOT = ARTIFACT_ROOT / "experiment_logs" / "v6_closed_loop_development"
PREREGISTRATION_PATH = (
    ARTIFACT_ROOT / "86_v6_deepseek_closed_loop_engineering_probe_preregistration.md"
)
PRICING_SNAPSHOT_PATH = (
    ARTIFACT_ROOT
    / "data"
    / "openrouter_deepseek_v4_pro_streamlake_pricing_snapshot_20260714.json"
)
CAPABILITY_SNAPSHOT_PATH = (
    ARTIFACT_ROOT
    / "data"
    / "openrouter_deepseek_v4_pro_streamlake_capability_snapshot_20260714.json"
)
EXPECTED_PRICING_SNAPSHOT_SHA256 = (
    "5adea5e08d7aea5eb89de010e1750890fe6b7f70a3f7fe733a08996d0b8b7204"
)
EXPECTED_CAPABILITY_SNAPSHOT_SHA256 = (
    "131d0fef27cb24350f9c067ea7407cd9279ddbe242eef77e29451390a750a671"
)


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    member = getattr(value, "value", None)
    if type(member) is str:
        return member
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _directory_fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DurableJsonlWriter:
    """Thread-safe fsync-before-return JSONL publication boundary."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = path.open("x", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False

    def write(self, value: Mapping[str, object]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("JSONL value must be a mapping")
        payload = _canonical_json(dict(value)) + "\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("JSONL writer is closed")
            self._stream.write(payload)
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._stream.close()
            self._closed = True


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        default=_json_default,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.replace(path)
    _directory_fsync(path.parent)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if type(value) is not dict:
            raise TypeError("JSONL row must be an object")
        rows.append(value)
    return rows


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _route_snapshot_evidence() -> dict[str, object]:
    evidence: dict[str, dict[str, object]] = {}
    decoded: dict[str, object] = {}
    for name, path, expected_sha in (
        ("pricing", PRICING_SNAPSHOT_PATH, EXPECTED_PRICING_SNAPSHOT_SHA256),
        (
            "capability",
            CAPABILITY_SNAPSHOT_PATH,
            EXPECTED_CAPABILITY_SNAPSHOT_SHA256,
        ),
    ):
        content = path.resolve(strict=True).read_bytes()
        observed_sha = hashlib.sha256(content).hexdigest()
        if observed_sha != expected_sha:
            raise RuntimeError(f"frozen {name} snapshot hash drifted")
        value = json.loads(content)
        if type(value) is not dict:
            raise TypeError(f"frozen {name} snapshot must be an object")
        decoded[name] = value
        evidence[name] = {
            "path": str(path.relative_to(ARTIFACT_ROOT)),
            "sha256": observed_sha,
            "bytes": len(content),
        }

    pricing = decoded["pricing"]
    capability = decoded["capability"]
    if type(pricing) is not dict or type(capability) is not dict:
        raise TypeError("route evidence snapshots must be exact objects")
    pricing_endpoint = pricing.get("selected_endpoint")
    capability_endpoint = capability.get("selected_endpoint")
    if type(pricing_endpoint) is not dict or type(capability_endpoint) is not dict:
        raise TypeError("route evidence lacks selected endpoint")
    prices = pricing_endpoint.get("pricing_usd_per_token")
    prompt_price = None
    completion_price = None
    if type(prices) is dict:
        try:
            prompt_price = Decimal(prices.get("prompt"))
            completion_price = Decimal(prices.get("completion"))
        except (InvalidOperation, TypeError):
            pass
    route_ok = (
        type(prices) is dict
        and prompt_price == PROMPT_PRICE_USD_PER_TOKEN
        and completion_price == COMPLETION_PRICE_USD_PER_TOKEN
        and pricing_endpoint.get("provider_name") == "StreamLake"
        and pricing_endpoint.get("provider_request_slug") == "streamlake"
        and capability_endpoint.get("provider_name") == "StreamLake"
        and capability_endpoint.get("provider_request_slug") == "streamlake"
        and capability.get("requested_model_alias") == MODEL
        and capability.get("canonical_model_slug") == CANONICAL_RESOLVED_MODEL
    )
    if not route_ok:
        raise RuntimeError("frozen StreamLake route evidence drifted")
    if DERIVED_MAX_SUCCESSFUL_RESPONSE_COST_USD != Decimal("0.0078987648"):
        raise RuntimeError("derived v6 token-price envelope drifted")
    return {
        "schema_version": 1,
        "snapshots": evidence,
        "route_validated": True,
        "derived_max_successful_response_cost_usd": str(
            DERIVED_MAX_SUCCESSFUL_RESPONSE_COST_USD
        ),
        "derived_max_accepted_run_cost_usd": str(DERIVED_MAX_ACCEPTED_RUN_COST_USD),
        "derived_max_potentially_billable_run_cost_usd": str(
            DERIVED_MAX_POTENTIALLY_BILLABLE_RUN_COST_USD
        ),
    }


def _source_paths() -> tuple[Path, ...]:
    paths = {
        *(
            path.resolve()
            for path in (AGENT_EVOLVE_ROOT / "src" / "agent_evolve").rglob("*.py")
            if path.is_file()
        ),
        (
            AGENT_EVOLVE_ROOT
            / "examples"
            / "development"
            / "run_v6_closed_loop_memory_probe.py"
        ).resolve(),
        (
            AGENT_EVOLVE_ROOT
            / "examples"
            / "development"
            / "v6_closed_loop_probe_support.py"
        ).resolve(),
        (AGENT_EVOLVE_ROOT / "pyproject.toml").resolve(),
        (AGENT_EVOLVE_ROOT / "uv.lock").resolve(),
    }
    return tuple(
        sorted(
            paths,
            key=lambda path: path.relative_to(AGENT_EVOLVE_ROOT).as_posix(),
        )
    )


def _read_source_payloads() -> tuple[
    tuple[tuple[Path, str, bytes, str], ...],
    str,
]:
    payloads: list[tuple[Path, str, bytes, str]] = []
    aggregate = hashlib.sha256(b"agent-evolve:v6-probe-source-snapshot:v1\x00")
    for source in _source_paths():
        resolved = source.resolve(strict=True)
        relative = resolved.relative_to(AGENT_EVOLVE_ROOT).as_posix()
        content = resolved.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        payloads.append((resolved, relative, content, digest))
        encoded_name = relative.encode("utf-8", errors="strict")
        aggregate.update(len(encoded_name).to_bytes(8, "big"))
        aggregate.update(encoded_name)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
    return tuple(payloads), aggregate.hexdigest()


def _source_state() -> tuple[dict[str, str], str]:
    payloads, aggregate_sha256 = _read_source_payloads()
    return (
        {relative: digest for _, relative, _, digest in payloads},
        aggregate_sha256,
    )


def _verify_source_snapshot(
    snapshot: Mapping[str, object],
    *,
    stage: str,
) -> dict[str, object]:
    expected_files = snapshot.get("files")
    expected_sha = snapshot.get("sha256")
    if type(expected_files) is not dict or type(expected_sha) is not str:
        raise TypeError("source snapshot lacks exact file/hash evidence")
    observed_files, observed_sha = _source_state()
    verified = observed_files == expected_files and observed_sha == expected_sha
    record = {
        "schema_version": 1,
        "stage": stage,
        "verified": verified,
        "expected_sha256": expected_sha,
        "observed_sha256": observed_sha,
        "expected_file_count": len(expected_files),
        "observed_file_count": len(observed_files),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not verified:
        raise RuntimeError(f"source snapshot drifted at {stage}")
    return record


def _validate_run_id(value: str) -> str:
    if (
        type(value) is not str
        or value in {".", ".."}
        or _SAFE_RUN_ID.fullmatch(value) is None
    ):
        raise ValueError(
            "run_id must be one safe path component using letters, digits, "
            "dot, underscore, or hyphen"
        )
    return value


def _validate_live_cli_freeze_requirements(
    *,
    mode: str,
    run_id: str | None,
    expected_source_sha256: str | None,
    expected_readiness_sha256: str | None,
    log_root: Path,
) -> None:
    """Reject an unfrozen live invocation before creating any run artifacts.

    This is only the syntactic half of the gate.  The supplied commitments are
    compared with freshly computed source/readiness commitments later, still
    before credential loading or construction of the provider stack.
    """

    if mode != "live":
        return
    if run_id is None:
        raise ValueError("live mode requires an explicit --run-id")
    if run_id != AUTHORIZED_LIVE_RUN_ID:
        raise ValueError(
            "live run_id is not the single prospectively authorized run: "
            f"{AUTHORIZED_LIVE_RUN_ID}"
        )
    if log_root.resolve() != DEFAULT_LOG_ROOT.resolve():
        raise ValueError(
            "live mode requires the canonical immutable log root: "
            f"{DEFAULT_LOG_ROOT.resolve()}"
        )
    for option, value in (
        ("--expected-source-sha256", expected_source_sha256),
        ("--expected-readiness-sha256", expected_readiness_sha256),
    ):
        if type(value) is not str or _LOWER_SHA256.fullmatch(value) is None:
            raise ValueError(
                f"live mode requires {option} as an exact lowercase SHA-256"
            )


def _preregistration_authorization_evidence(
    *,
    mode: str,
    run_id: str,
    expected_source_sha256: str | None,
    expected_readiness_sha256: str | None,
    path: Path = PREREGISTRATION_PATH,
) -> dict[str, object]:
    """Require prospective, external authorization before live side effects."""

    if mode != "live":
        return {
            "required_for_live": True,
            "validated": False,
            "reason": "not_live",
        }
    if expected_source_sha256 is None or expected_readiness_sha256 is None:
        raise ValueError("live preregistration lookup requires both commitments")
    resolved = path.resolve(strict=True)
    content = resolved.read_bytes()
    try:
        document = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError("live preregistration is not valid UTF-8") from exc
    required_lines = {
        PREREGISTRATION_READY_MARKER,
        f"AUTHORIZED_LIVE_RUN_ID: `{run_id}`",
        f"SOURCE_SNAPSHOT_SHA256: `{expected_source_sha256}`",
        f"READINESS_SHA256: `{expected_readiness_sha256}`",
    }
    observed_lines = {line.strip() for line in document.splitlines()}
    missing = sorted(required_lines - observed_lines)
    if missing:
        raise RuntimeError(
            "live preregistration lacks exact executable authorization lines: "
            + ", ".join(missing)
        )
    return {
        "required_for_live": True,
        "validated": True,
        "path": (
            str(resolved.relative_to(ARTIFACT_ROOT.resolve()))
            if resolved.is_relative_to(ARTIFACT_ROOT.resolve())
            else str(resolved)
        ),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "authorized_run_id": run_id,
        "source_snapshot_sha256": expected_source_sha256,
        "readiness_sha256": expected_readiness_sha256,
        "ready_marker": PREREGISTRATION_READY_MARKER,
    }


def _external_freeze_commitments(
    *,
    mode: str,
    expected_source_sha256: str | None,
    expected_readiness_sha256: str | None,
    source_snapshot: Mapping[str, object],
    readiness: Mapping[str, object],
) -> dict[str, object]:
    provided = {
        "source_snapshot_sha256": expected_source_sha256,
        "readiness_sha256": expected_readiness_sha256,
    }
    if mode == "live" and any(value is None for value in provided.values()):
        raise ValueError(
            "live mode requires both --expected-source-sha256 and "
            "--expected-readiness-sha256"
        )
    observed = {
        "source_snapshot_sha256": source_snapshot.get("sha256"),
        "readiness_sha256": readiness.get("readiness_sha256"),
    }
    matches: dict[str, bool | None] = {}
    for name, expected in provided.items():
        if expected is None:
            matches[name] = None
            continue
        if _LOWER_SHA256.fullmatch(expected) is None:
            raise ValueError(f"expected {name} must be exact lowercase SHA-256")
        if expected != observed[name]:
            raise RuntimeError(f"external freeze commitment mismatch: {name}")
        matches[name] = True
    return {
        "required_for_live": True,
        "provided": provided,
        "observed": observed,
        "matches": matches,
        "all_provided_commitments_match": all(
            value is True for value in matches.values()
        ),
    }


def _finalize_run(run_dir: Path, *, status: str) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    aggregate = hashlib.sha256(b"agent-evolve:v6-probe-finalized:v1\x00")
    paths = sorted(
        (
            path
            for path in run_dir.rglob("*")
            if path.is_file()
            and path.name != "finalized.json"
            and not path.name.endswith(".tmp")
        ),
        key=lambda item: item.relative_to(run_dir).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(run_dir).as_posix()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        record: dict[str, object] = {
            "bytes": len(content),
            "sha256": digest,
        }
        if path.suffix == ".jsonl":
            record["jsonl_lines"] = len(content.splitlines())
        files[relative] = record
        encoded = relative.encode("utf-8", errors="strict")
        aggregate.update(len(encoded).to_bytes(8, "big"))
        aggregate.update(encoded)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
    finalized = {
        "schema_version": 1,
        "status": status,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "recursive_file_count": len(files),
        "recursive_content_sha256": aggregate.hexdigest(),
        "files": files,
    }
    _write_json(run_dir / "finalized.json", finalized)
    return finalized


def _snapshot_sources(run_dir: Path) -> dict[str, object]:
    snapshot_root = run_dir / "source_snapshot"
    snapshot_root.mkdir(exist_ok=False)
    payloads, aggregate_sha256 = _read_source_payloads()
    files = {relative: digest for _, relative, _, digest in payloads}
    for _, relative, content, digest in payloads:
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise RuntimeError(
                f"copied source snapshot failed verification: {relative}"
            )
    for directory in sorted(
        {path.parent for path in snapshot_root.rglob("*") if path.is_file()},
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        _directory_fsync(directory)
    _directory_fsync(snapshot_root)
    return {
        "schema_version": 1,
        "framing": "agent-evolve:v6-probe-source-snapshot:v1",
        "sha256": aggregate_sha256,
        "file_count": len(files),
        "files": files,
        "snapshot_directory": "source_snapshot",
    }


def telemetry_policy() -> AgenticTelemetryPolicy:
    return AgenticTelemetryPolicy(
        requested_model=MODEL,
        allowed_resolved_models=ALLOWED_RESOLVED_MODELS,
        allowed_resolved_providers=ALLOWED_RESOLVED_PROVIDERS,
        max_cost_usd=MAX_CALL_COST_USD,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_reasoning_tokens=MAX_REASONING_TOKENS,
        max_attempt_count=QUEUE_MAX_ATTEMPTS,
    )


@dataclass(frozen=True, slots=True)
class LiveStack:
    runner: Any
    generator: AgenticGenerator
    telemetry_policy: AgenticTelemetryPolicy


def create_live_stack(
    *,
    api_key: str,
    queue_sink: Callable[[Any], None],
    telemetry_policy_override: AgenticTelemetryPolicy | None = None,
    attempt_timeout_ns: int = QUEUE_ATTEMPT_TIMEOUT_NS,
    reasoning_config: OpenRouterReasoningConfig | None = None,
) -> LiveStack:
    """Compose, but do not enter or call, one explicitly configured live route.

    The defaults preserve the frozen v6 probe.  Later experiment composition
    roots may inject a different hard attempt wall and a validated provider
    reasoning policy without cloning the queue/provider stack.
    """

    if QUEUE_MAX_IN_FLIGHT < FULL_WAVE_WIDTH:
        raise RuntimeError("queue concurrency is smaller than the largest wave")
    if type(attempt_timeout_ns) is not int or not (
        1_000_000_000 <= attempt_timeout_ns <= 600_000_000_000
    ):
        raise ValueError("attempt_timeout_ns must lie in [1s,600s]")
    if (
        reasoning_config is not None
        and type(reasoning_config) is not OpenRouterReasoningConfig
    ):
        raise TypeError(
            "reasoning_config must be an exact OpenRouterReasoningConfig"
        )
    openrouter_options: dict[str, Any] = {
        "api_key": api_key,
        "model_name": MODEL,
        "max_connections": QUEUE_MAX_IN_FLIGHT,
        "timeout_seconds": attempt_timeout_ns / 1_000_000_000,
        "provider_options": {"only": list(PROVIDER_ONLY)},
        "app_title": "AgentEvolve AAAI 2027 v6 engineering probe",
    }
    if reasoning_config is not None:
        openrouter_options["reasoning_config"] = reasoning_config
    structured = PydanticAIStructuredGenerator.openrouter(**openrouter_options)
    runner = create_production_queued_runner(
        generator=structured,
        max_in_flight=QUEUE_MAX_IN_FLIGHT,
        max_pending=QUEUE_MAX_PENDING,
        max_attempts=QUEUE_MAX_ATTEMPTS,
        attempt_timeout_ns=attempt_timeout_ns,
        base_backoff_ns=QUEUE_BASE_BACKOFF_NS,
        max_backoff_ns=QUEUE_MAX_BACKOFF_NS,
        jitter_policy=DeterministicHashJitter(
            seed=JITTER_SEED,
            domain=JITTER_DOMAIN,
        ),
        close_generator=True,
        outcome_sink=queue_sink,
        outcome_publication_policy=OutcomePublicationPolicy.REQUIRED,
        attempt_request_policy=SchemaRepairAttemptPolicy(),
    )
    policy = (
        telemetry_policy()
        if telemetry_policy_override is None
        else telemetry_policy_override
    )
    if type(policy) is not AgenticTelemetryPolicy:
        raise TypeError(
            "telemetry_policy_override must be an exact AgenticTelemetryPolicy"
        )
    AgenticTelemetryPolicy.__post_init__(policy)
    generator = TelemetryGatedAgenticGenerator(
        PydanticAIAgenticGenerator(runner),
        policy,
    )
    return LiveStack(runner=runner, generator=generator, telemetry_policy=policy)


async def prepare_readiness() -> dict[str, object]:
    """Prepare exact G1 assignments/prompts without provider or queue I/O."""

    composition = compose_probe(
        generator_factory=offline_generator_factory,
        trace_sink=None,
        evaluation_delay_seconds=0,
    )
    seed = await composition.engine.register_seed(
        {"a": 4, "b": 4},
        label="seed_0",
    )
    composition.archive.consider(seed)
    snapshot = composition.archive.snapshot()
    state = OptimizerState(
        generation=0,
        candidates=(seed,),
        archive=snapshot,
        archive_snapshot_hash=pareto_archive_snapshot_hash(snapshot),
        unique_evaluations=1,
        logical_llm_calls=0,
        generation_receipts=(),
    )
    plan = composition.planner.plan(state, OPTIMIZER_BUDGET)
    prepared, _ = composition.engine.prepare_invocations(
        tuple(slot.plan for slot in plan.slots),
        reward_binding=plan.reward.binding,
    )
    planner = composition.planner
    if planner.wave is None or planner.prompt_shape_sha256 is None:
        raise RuntimeError("readiness did not freeze the diagnostic wave")
    assignments = {
        item.plan.resolved_insight_assignment.assignment_sha256: {
            "assignment_sha256": (
                item.plan.resolved_insight_assignment.assignment_sha256
            ),
            "assignment": item.plan.resolved_insight_assignment.to_record(),
            "operator_invocation_id": item.operator_invocation_id.value,
            "call_id": item.call_id.value if item.call_id is not None else None,
            "candidate_id": item.candidate_id.value,
            "prepared_prompt_sha256": hashlib.sha256(
                item.prompt.encode("utf-8", errors="strict")
            ).hexdigest(),
            "prepared_prompt_utf8_bytes": len(item.prompt.encode("utf-8")),
        }
        for item in prepared
        if item.plan.resolved_insight_assignment is not None
    }
    return {
        "schema_version": 1,
        "development_only": True,
        "provider_io_performed": False,
        "queue_started": False,
        "seed_configuration": {"a": 4, "b": 4},
        "prompt_shape_sha256": planner.prompt_shape_sha256,
        "prompt_shape_source": "AgenticEvolutionEngine.prompt_shape_commitment",
        "wave": {
            **planner.wave.to_record(),
            "wave_sha256": planner.wave.wave_sha256,
        },
        "assignments_by_sha256": assignments,
        "g1_model_wave_width": len(prepared),
        "g2_model_wave_width": MODEL_WAVE_WIDTH,
        "g2_full_wave_width": FULL_WAVE_WIDTH,
        "later_waves_require_sealed_checkpoint": True,
        "recourse_pool": composition.recourse_pool.to_trace_record(),
        "readiness_sha256": canonical_record_sha256(
            {
                "prompt_shape_sha256": planner.prompt_shape_sha256,
                "wave_sha256": planner.wave.wave_sha256,
                "assignments": assignments,
                "recourse_pool_sha256": composition.recourse_pool.pool_sha256,
            }
        ),
    }


class ReadinessTraceGate:
    """Synchronously reject actual G1/G2 commitments that drift from preview."""

    def __init__(
        self,
        readiness: Mapping[str, object],
        sink: Callable[[Mapping[str, object]], None],
    ) -> None:
        assignments = readiness["assignments_by_sha256"]
        if type(assignments) is not dict:
            raise TypeError("readiness assignments must be an object")
        self.expected = assignments
        self.prompt_shape_sha256 = readiness["prompt_shape_sha256"]
        self.sink = sink
        self.seen_diagnostic: set[str] = set()
        self.expected_prompt_by_insight: dict[tuple[str, int], str] = {}
        self.seen_matched_insights: set[tuple[str, int]] = set()
        for expected in assignments.values():
            if type(expected) is not dict:
                raise TypeError("readiness assignment row must be an object")
            selected = self._selected_insight(expected.get("assignment"))
            prompt_sha = expected.get("prepared_prompt_sha256")
            if (
                type(prompt_sha) is not str
                or selected in self.expected_prompt_by_insight
            ):
                raise RuntimeError("readiness prompt/card binding is not one-to-one")
            self.expected_prompt_by_insight[selected] = prompt_sha

    @staticmethod
    def _selected_insight(value: object) -> tuple[str, int]:
        if type(value) is not dict:
            raise TypeError("assignment must be an object")
        decision = value.get("selection_decision")
        if type(decision) is not dict:
            raise TypeError("assignment selection decision must be an object")
        selected = decision.get("selected")
        if type(selected) is not list or len(selected) != 1:
            raise RuntimeError("probe assignment must select exactly one insight")
        reference = selected[0]
        if type(reference) is not dict:
            raise TypeError("selected insight reference must be an object")
        insight_id = reference.get("insight_id")
        version = reference.get("version")
        if type(insight_id) is not str or type(version) is not int:
            raise TypeError("selected insight reference is malformed")
        return insight_id, version

    def __call__(self, event: Mapping[str, object]) -> None:
        if event.get("event_type") == "assignment_committed":
            assignment_sha = event.get("assignment_sha256")
            block_id = event.get("block_id")
            if type(assignment_sha) is not str or type(block_id) is not str:
                raise RuntimeError("assignment commitment lacks identity")
            if event.get("prompt_shape_commitment_verified") is not True:
                raise RuntimeError("engine did not verify prompt-shape commitment")
            if event.get("prompt_shape_sha256") != self.prompt_shape_sha256:
                raise RuntimeError("actual prompt-shape commitment drifted")
            selected = self._selected_insight(event.get("assignment"))
            expected_prompt_sha = self.expected_prompt_by_insight.get(selected)
            if (
                expected_prompt_sha is None
                or event.get("prepared_prompt_sha256") != expected_prompt_sha
            ):
                raise RuntimeError(
                    "same-card prepared prompt drifted from readiness replay"
                )
            if block_id.startswith("v6_diagnostic_"):
                expected = self.expected.get(assignment_sha)
                if type(expected) is not dict:
                    raise RuntimeError("actual diagnostic assignment was not previewed")
                if event.get("prepared_prompt_sha256") != expected.get(
                    "prepared_prompt_sha256"
                ):
                    raise RuntimeError("actual diagnostic prompt drifted from preview")
                self.seen_diagnostic.add(assignment_sha)
            elif block_id == "v6_matched_block":
                self.seen_matched_insights.add(selected)
        self.sink(event)

    def require_complete(self) -> None:
        if self.seen_diagnostic != set(self.expected):
            raise RuntimeError("not every previewed diagnostic assignment committed")
        if self.seen_matched_insights != set(self.expected_prompt_by_insight):
            raise RuntimeError("not every same-card matched prompt replay committed")


def _telemetry_record(outcome: InvocationOutcome) -> dict[str, object] | None:
    candidate = outcome.candidate
    telemetry = None if candidate is None else candidate.call_telemetry
    if telemetry is None:
        return None
    return {
        "requested_model": telemetry.requested_model,
        "resolved_model": telemetry.resolved_model,
        "resolved_provider": telemetry.resolved_provider,
        "provider_response_id": telemetry.provider_response_id,
        "finish_reason": telemetry.finish_reason,
        "input_tokens": telemetry.input_tokens,
        "output_tokens": telemetry.output_tokens,
        "reasoning_tokens": telemetry.reasoning_tokens,
        "cost_usd": None if telemetry.cost_usd is None else str(telemetry.cost_usd),
        "latency_ns": telemetry.latency_ns,
        "attempt_count": telemetry.attempt_count,
    }


def publish_outcomes(
    result: OptimizerResult,
    writer: DurableJsonlWriter,
) -> None:
    for seed in result.seed_receipts:
        writer.write(
            {
                "schema_version": 1,
                "record_type": "seed",
                "label": seed.label,
                "candidate_id": seed.candidate.candidate_id.value,
                "configuration": seed.candidate.configuration_dict,
                "objectives": seed.candidate.objective_map,
                "valid": seed.candidate.valid,
                "receipt_hash": seed.receipt_hash,
                "unique_evaluations_after": seed.unique_evaluations_after,
            }
        )
    for receipt in result.generation_receipts:
        for slot_result in receipt.slot_results:
            outcome = slot_result.outcome
            candidate = outcome.candidate
            writer.write(
                {
                    "schema_version": 1,
                    "record_type": "generation_slot",
                    "generation": receipt.generation,
                    "generation_receipt_hash": receipt.receipt_hash,
                    "slot_id": slot_result.slot.slot_id,
                    "role": slot_result.slot.role,
                    "proposal_authority": (outcome.prepared.proposal_authority.value),
                    "operator_invocation_id": (
                        outcome.prepared.operator_invocation_id.value
                    ),
                    "call_id": (
                        None
                        if outcome.prepared.call_id is None
                        else outcome.prepared.call_id.value
                    ),
                    "candidate_id": (
                        None if candidate is None else candidate.candidate_id.value
                    ),
                    "configuration": (
                        None if candidate is None else candidate.configuration_dict
                    ),
                    "objectives": (
                        None if candidate is None else candidate.objective_map
                    ),
                    "valid": None if candidate is None else candidate.valid,
                    "operator_compliant": (
                        None if candidate is None else candidate.operator_compliant
                    ),
                    "evidence_compliant": (
                        None if candidate is None else candidate.evidence_compliant
                    ),
                    "reward_hex": outcome.reward.hex(),
                    "failure_stage": outcome.failure_stage,
                    "failure_type": outcome.call_failure_type,
                    "telemetry": _telemetry_record(outcome),
                }
            )


def _decimal_cost(event: Mapping[str, object]) -> Decimal | None:
    value = event.get("cost_usd")
    if type(value) is not str:
        return None
    try:
        result = Decimal(value)
    except InvalidOperation:
        return None
    return result if result.is_finite() and result >= 0 else None


async def mechanism_analysis(
    *,
    composition: ProbeComposition,
    result: OptimizerResult,
    events: Sequence[Mapping[str, object]],
    queue_records: Sequence[Mapping[str, object]],
    mode: str,
) -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {}

    def record(name: str, passed: bool, observed: object) -> None:
        checks[name] = {"passed": bool(passed), "observed": observed}

    planner = composition.planner
    g1, g2, g3, g4 = result.generation_receipts
    g1_configs = [
        slot.outcome.candidate.configuration_dict
        if slot.outcome.candidate is not None
        else None
        for slot in g1.slot_results
    ]
    record(
        "diagnostic_targets_realized",
        g1_configs == [{"a": 3, "b": 4}, {"a": 1, "b": 4}],
        g1_configs,
    )
    closure = planner.closure
    snapshot = None if closure is None else closure.snapshot
    scores = (
        {}
        if snapshot is None
        else {
            entry.reference.insight_id.value: {
                "effect": entry.effect_estimate,
                "retrieval_score": entry.retrieval_score,
            }
            for entry in snapshot.entries
        }
    )
    record(
        "delayed_checkpoint_sealed",
        closure is not None
        and closure.status is MemoryCheckpointClosureStatus.SEALED
        and snapshot is not None,
        {
            "status": None if closure is None else closure.status.value,
            "checkpoint_index": None if snapshot is None else snapshot.checkpoint_index,
            "scores": scores,
        },
    )
    record(
        "no_within_wave_mutable_credit",
        composition.memory.trials == (),
        {"legacy_trial_count": len(composition.memory.trials)},
    )
    adaptive = planner.adaptive_assignment
    control = planner.control_assignment
    selected = {
        "adaptive": (
            []
            if adaptive is None
            else [ref.insight_id.value for ref in adaptive.selection_decision.selected]
        ),
        "control": (
            []
            if control is None
            else [ref.insight_id.value for ref in control.selection_decision.selected]
        ),
    }
    g2_slots = {slot.slot.slot_id: slot for slot in g2.slot_results}
    adaptive_slot = g2_slots.get("G2-adaptive")
    control_slot = g2_slots.get("G2-control")
    if adaptive_slot is None or control_slot is None:
        raise RuntimeError("G2 lacks its frozen adaptive/control slots")
    adaptive_outcome = adaptive_slot.outcome
    control_outcome = control_slot.outcome
    adaptive_candidate = adaptive_outcome.candidate
    control_candidate = control_outcome.candidate
    adaptive_identity = (
        None
        if adaptive_candidate is None
        else composition.engine.identify_phenotype(adaptive_candidate)
    )
    control_identity = (
        None
        if control_candidate is None
        else composition.engine.identify_phenotype(control_candidate)
    )
    paired_reward_delta = adaptive_outcome.reward - control_outcome.reward
    adaptive_recomputed_reward = (
        None
        if adaptive_candidate is None
        else closed_loop_reward(
            adaptive_candidate,
            adaptive_outcome.prepared.plan.parents,
            composition.problem.objectives,
        )
    )
    control_recomputed_reward = (
        None
        if control_candidate is None
        else closed_loop_reward(
            control_candidate,
            control_outcome.prepared.plan.parents,
            composition.problem.objectives,
        )
    )
    paired_observed = {
        "adaptive_slot_id": adaptive_slot.slot.slot_id,
        "adaptive_slot_role": adaptive_slot.slot.role,
        "adaptive_proposal_authority": adaptive_slot.slot.proposal_authority.value,
        "control_slot_id": control_slot.slot.slot_id,
        "control_slot_role": control_slot.slot.role,
        "control_proposal_authority": control_slot.slot.proposal_authority.value,
        "selected_insights": selected,
        "adaptive_configuration": (
            None
            if adaptive_candidate is None
            else adaptive_candidate.configuration_dict
        ),
        "control_configuration": (
            None if control_candidate is None else control_candidate.configuration_dict
        ),
        "adaptive_reward_hex": adaptive_outcome.reward.hex(),
        "control_reward_hex": control_outcome.reward.hex(),
        "adaptive_minus_control_reward_hex": paired_reward_delta.hex(),
        "adaptive_recomputed_reward_hex": (
            None
            if adaptive_recomputed_reward is None
            else adaptive_recomputed_reward.hex()
        ),
        "control_recomputed_reward_hex": (
            None
            if control_recomputed_reward is None
            else control_recomputed_reward.hex()
        ),
        "adaptive_phenotype_sha256": (
            None if adaptive_identity is None else adaptive_identity.identity_sha256
        ),
        "control_phenotype_sha256": (
            None if control_identity is None else control_identity.identity_sha256
        ),
        "phenotype_equal": (
            None
            if adaptive_identity is None or control_identity is None
            else adaptive_identity == control_identity
        ),
    }
    record(
        "adaptive_control_path_realized_expected_contrast",
        adaptive is not None
        and control is not None
        and adaptive_slot.slot.role == "adaptive_memory"
        and control_slot.slot.role == "score_shuffled_control"
        and adaptive_slot.slot.proposal_authority.value == "model"
        and control_slot.slot.proposal_authority.value == "model"
        and adaptive_outcome.prepared.plan.resolved_insight_assignment == adaptive
        and control_outcome.prepared.plan.resolved_insight_assignment == control
        and adaptive.selection_decision.selected == (composition.b_ref,)
        and control.selection_decision.selected == (composition.a_ref,)
        and adaptive_candidate is not None
        and control_candidate is not None
        and adaptive_candidate.configuration_dict == {"a": 1, "b": 4}
        and control_candidate.configuration_dict == {"a": 3, "b": 4}
        and adaptive_identity is not None
        and control_identity is not None
        and adaptive_identity != control_identity
        and adaptive_recomputed_reward == adaptive_outcome.reward == 3.0
        and control_recomputed_reward == control_outcome.reward == 1.0
        and adaptive_outcome.reward > control_outcome.reward
        and paired_reward_delta == 2.0,
        paired_observed,
    )
    assignments = (*planner.diagnostic_assignments, adaptive, control)
    record(
        "engine_verified_one_prompt_shape",
        adaptive is not None
        and control is not None
        and planner.prompt_shape_sha256 is not None
        and all(
            assignment is not None
            and assignment.prompt_shape_sha256 == planner.prompt_shape_sha256
            for assignment in assignments
        ),
        {"prompt_shape_sha256": planner.prompt_shape_sha256},
    )
    decision = planner.recourse_decision
    ledger = None if decision is None else decision.ledger
    record(
        "phenotype_collision_funds_one_recourse_slot",
        ledger is not None
        and len(ledger.primary_occurrences) == 4
        and len(ledger.clusters) == 3
        and ledger.successful_primary_collision_credit == 1
        and decision.selected_entry_ids == ("orthogonal_b",)
        and g2.unique_evaluations_after - g2.unique_evaluations_before == 1,
        {
            "primary_occurrences": (
                None if ledger is None else len(ledger.primary_occurrences)
            ),
            "phenotype_clusters": None if ledger is None else len(ledger.clusters),
            "collision_credit": (
                None if ledger is None else ledger.successful_primary_collision_credit
            ),
            "selected_entry_ids": (
                [] if decision is None else list(decision.selected_entry_ids)
            ),
            "g2_unique_evaluation_delta": (
                g2.unique_evaluations_after - g2.unique_evaluations_before
            ),
            "collision_source": {
                "slot_ids": ["G2-coverage-0", "G2-coverage-1"],
                "proposal_authority": "engine",
                "configuration": {"a": 2, "b": 2},
                "scripted_fixture": True,
            },
        },
    )
    combined = planner.combined_ledger
    record(
        "recourse_is_non_chaining",
        combined is not None
        and len(combined.ignored_recourse_trial_ids) == 1
        and combined.successful_primary_collision_credit == 1
        and len(g3.slot_results) == 1,
        {
            "ignored_recourse_trial_ids": (
                []
                if combined is None
                else [value.value for value in combined.ignored_recourse_trial_ids]
            ),
            "successful_primary_collision_credit": (
                None
                if combined is None
                else combined.successful_primary_collision_credit
            ),
        },
    )
    final = g4.slot_results[0].outcome.candidate
    record(
        "disjoint_recombination_reaches_optimum",
        final is not None
        and final.configuration_dict == {"a": 1, "b": 1}
        and final.operator_compliant
        and final.evidence_compliant
        and final.preservation_verified,
        None if final is None else final.configuration_dict,
    )
    cache = await composition.engine.evaluation_cache_snapshot()
    record(
        "occurrence_vs_physical_evaluation_accounting",
        len(result.final_state.candidates) == 9
        and result.final_state.unique_evaluations == 6
        and cache["misses"] == 6
        and cache["hits"] == 2
        and cache["coalesced"] == 1,
        {
            "candidate_occurrences": len(result.final_state.candidates),
            "unique_evaluations": result.final_state.unique_evaluations,
            "cache": cache,
        },
    )
    event_list = list(events)
    committed = [
        (index, event)
        for index, event in enumerate(event_list)
        if event.get("event_type") == "assignment_committed"
    ]
    terminals = [
        (index, event)
        for index, event in enumerate(event_list)
        if event.get("event_type") == "trial_terminal"
    ]
    committed_by_assignment = {
        str(event.get("assignment_sha256")): event for _, event in committed
    }
    prompt_hashes_by_insight: dict[str, list[str | None]] = {}
    for assignment in assignments:
        if assignment is None or len(assignment.selection_decision.selected) != 1:
            continue
        insight_id = assignment.selection_decision.selected[0].insight_id.value
        event = committed_by_assignment.get(assignment.assignment_sha256)
        prompt_value = None if event is None else event.get("prepared_prompt_sha256")
        prompt_hashes_by_insight.setdefault(insight_id, []).append(
            prompt_value if type(prompt_value) is str else None
        )
    expected_prompt_cards = {
        composition.a_ref.insight_id.value,
        composition.b_ref.insight_id.value,
    }
    exact_same_card_replay = set(
        prompt_hashes_by_insight
    ) == expected_prompt_cards and all(
        len(prompt_hashes) == 2
        and prompt_hashes[0] is not None
        and prompt_hashes[0] == prompt_hashes[1]
        for prompt_hashes in prompt_hashes_by_insight.values()
    )
    record(
        "same_card_prompt_replay_is_byte_exact_across_waves",
        exact_same_card_replay,
        prompt_hashes_by_insight,
    )
    wave_order_ok = True
    for block_prefix in ("v6_diagnostic_", "v6_matched_block"):
        commit_indices = [
            index
            for index, event in committed
            if str(event.get("block_id", "")).startswith(block_prefix)
        ]
        terminal_indices = [
            index
            for index, event in terminals
            if str(event.get("block_id", "")).startswith(block_prefix)
        ]
        wave_order_ok &= (
            len(commit_indices) == 2
            and len(terminal_indices) == 2
            and max(commit_indices) < min(terminal_indices)
        )
    record(
        "concurrent_wave_commit_contract",
        wave_order_ok and QUEUE_MAX_IN_FLIGHT >= FULL_WAVE_WIDTH,
        {
            "g1_model_wave_width": MODEL_WAVE_WIDTH,
            "g2_full_wave_width": FULL_WAVE_WIDTH,
            "queue_max_in_flight": QUEUE_MAX_IN_FLIGHT,
            "all_wave_assignments_committed_before_first_terminal": wave_order_ok,
        },
    )

    if mode == "live":
        completed = [
            event
            for event in event_list
            if event.get("event_type") == "llm_call_completed"
        ]
        costs = [_decimal_cost(event) for event in completed]
        total_cost = sum(
            (cost for cost in costs if cost is not None),
            Decimal("0"),
        )
        route_ok = (
            len(completed) == OPTIMIZER_BUDGET.max_logical_llm_calls
            and all(event.get("requested_model") == MODEL for event in completed)
            and all(
                event.get("resolved_model") in ALLOWED_RESOLVED_MODELS
                for event in completed
            )
            and all(
                event.get("resolved_provider") in ALLOWED_RESOLVED_PROVIDERS
                for event in completed
            )
            and all(cost is not None and cost <= MAX_CALL_COST_USD for cost in costs)
            and total_cost <= MAX_RUN_COST_USD
        )
        record(
            "exact_live_route_and_cost_gate",
            route_ok,
            {
                "completed_calls": len(completed),
                "requested_models": sorted(
                    {str(event.get("requested_model")) for event in completed}
                ),
                "resolved_models": sorted(
                    {str(event.get("resolved_model")) for event in completed}
                ),
                "resolved_providers": sorted(
                    {str(event.get("resolved_provider")) for event in completed}
                ),
                "reported_total_cost_usd": str(total_cost),
            },
        )
        queue_ok = len(queue_records) == OPTIMIZER_BUDGET.max_logical_llm_calls and all(
            record_.get("status") == "succeeded" for record_ in queue_records
        )
        record(
            "required_queue_outcomes_published",
            queue_ok,
            {
                "terminal_outcomes": len(queue_records),
                "statuses": [record_.get("status") for record_ in queue_records],
            },
        )
        first_attempt_only = queue_ok and all(
            type(record_.get("attempts")) is list
            and len(record_["attempts"]) == 1
            and type(record_["attempts"][0]) is dict
            and record_["attempts"][0].get("status") == "succeeded"
            for record_ in queue_records
        )
        record(
            "first_attempt_only_for_paired_fixture_comparability",
            first_attempt_only,
            {
                "attempt_counts": [
                    (
                        len(record_["attempts"])
                        if type(record_.get("attempts")) is list
                        else None
                    )
                    for record_ in queue_records
                ],
                "policy": (
                    "Retries remain available for operational evidence, but any "
                    "retry makes this paired engineering smoke fail closed because "
                    "schema-repair prompts can differ across arms."
                ),
            },
        )

    overall = all(item["passed"] is True for item in checks.values())
    return {
        "schema_version": 1,
        "development_only": True,
        "evidence_class": "engineering_mechanism_probe_only",
        "benchmark_claim_allowed": False,
        "wall_clock_claim_allowed": False,
        "scripted_fixture": True,
        "fixture_contrast_only": True,
        "causal_effect_identifiable": False,
        "model_reasoning_quality_tested": False,
        "fixed_assignment_ranks": {
            "diagnostic_uniform_subset_ranks": [0, 1],
            "control_score_permutation_rank": 1,
            "diagnostic_assignment_randomized": False,
            "control_permutation_randomized": False,
        },
        "mode": mode,
        "overall_pass": overall,
        "hypothesis_outcome": (
            "engineering_fixture_path_passed"
            if overall
            else "engineering_fixture_path_failed_or_drifted"
        ),
        "checks": checks,
        "optimizer_result_hash": result.result_hash,
    }


def _live_generator_factory(generator: AgenticGenerator):
    def factory(
        memory: InsightMemoryBank,
        a_ref: InsightRef,
        b_ref: InsightRef,
    ) -> AgenticGenerator:
        del memory, a_ref, b_ref
        return generator

    return factory


def _manifest(
    *,
    run_id: str,
    mode: str,
    readiness: Mapping[str, object],
    source_snapshot: Mapping[str, object],
    external_freeze_commitments: Mapping[str, object],
    preregistration_authorization: Mapping[str, object],
    route_snapshot_evidence: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "development_only": True,
        "claim_boundary": (
            "Synthetic engineering mechanism evidence only; not a benchmark, "
            "SOTA, genericity, or wall-clock result."
        ),
        "scientific_interpretation": {
            "scripted_fixture": True,
            "fixture_contrast_only": True,
            "causal_effect_identifiable": False,
            "model_reasoning_quality_tested": False,
            "provider_integration_smoke_only_if_live": True,
            "fixed_assignment_ranks": {
                "diagnostic_uniform_subset_ranks": [0, 1],
                "control_score_permutation_rank": 1,
            },
        },
        "provider_io_authorized": mode == "live",
        "model": MODEL,
        "provider": "openrouter",
        "provider_options": {"only": list(PROVIDER_ONLY)},
        "telemetry_policy": telemetry_policy().to_trace_record(),
        "cost_exposure": {
            "pricing_derivation": {
                "max_input_tokens": MAX_INPUT_TOKENS,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "max_reasoning_tokens": MAX_REASONING_TOKENS,
                "prompt_usd_per_token": str(PROMPT_PRICE_USD_PER_TOKEN),
                "completion_usd_per_token": str(COMPLETION_PRICE_USD_PER_TOKEN),
                "reasoning_accounting": (
                    "The reasoning cap is conservatively charged once more at "
                    "the completion-token rate."
                ),
                "derived_max_successful_response_cost_usd": str(
                    DERIVED_MAX_SUCCESSFUL_RESPONSE_COST_USD
                ),
                "derived_max_accepted_run_cost_usd": str(
                    DERIVED_MAX_ACCEPTED_RUN_COST_USD
                ),
                "derived_max_potentially_billable_run_cost_usd": str(
                    DERIVED_MAX_POTENTIALLY_BILLABLE_RUN_COST_USD
                ),
                "pricing_snapshot_sha256": EXPECTED_PRICING_SNAPSHOT_SHA256,
            },
            "accepted_success_response_cap_usd": str(MAX_CALL_COST_USD),
            "accepted_success_run_cap_usd": str(MAX_RUN_COST_USD),
            "max_potentially_billable_attempt_cost_usd": str(
                MAX_POTENTIALLY_BILLABLE_ATTEMPT_COST_USD
            ),
            "max_attempts_per_logical_call": QUEUE_MAX_ATTEMPTS,
            "conservative_declared_potentially_billable_run_exposure_usd": str(
                MAX_POTENTIALLY_BILLABLE_RUN_COST_USD
            ),
            "caveat": (
                "The accepted-success gate uses post-response telemetry. Failed or "
                "retried provider attempts may still be billed without usable cost "
                "telemetry, so USD 0.080 is a conservative declared budgeting "
                "envelope, not a mechanically guaranteed provider-spend cap."
            ),
        },
        "optimizer_budget": OPTIMIZER_BUDGET.to_trace_record(),
        "expected_physical_evaluations": 6,
        "concurrency": {
            "g1_model_wave_width": MODEL_WAVE_WIDTH,
            "g2_model_wave_width": MODEL_WAVE_WIDTH,
            "g2_full_wave_width": FULL_WAVE_WIDTH,
            "queue_max_in_flight": QUEUE_MAX_IN_FLIGHT,
            "engine_evaluator_concurrency": FULL_WAVE_WIDTH,
        },
        "queue": {
            "max_in_flight": QUEUE_MAX_IN_FLIGHT,
            "max_pending": QUEUE_MAX_PENDING,
            "max_attempts": QUEUE_MAX_ATTEMPTS,
            "attempt_timeout_ns": QUEUE_ATTEMPT_TIMEOUT_NS,
            "backoff": {
                "kind": "exponential",
                "base_delay_ns": QUEUE_BASE_BACKOFF_NS,
                "max_delay_ns": QUEUE_MAX_BACKOFF_NS,
            },
            "jitter": {
                "kind": "task_keyed_sha256",
                "seed": JITTER_SEED,
                "domain": JITTER_DOMAIN,
            },
            "terminal_outcome_publication": "required_fsync_before_downstream",
            "sdk_retries": 0,
            "pydantic_ai_retries": 0,
            "schema_repair_policy": (SCHEMA_REPAIR_POLICY_MANIFEST.to_trace_record()),
        },
        "prompt_shape": {
            "source": readiness["prompt_shape_source"],
            "sha256": readiness["prompt_shape_sha256"],
        },
        "readiness_sha256": readiness["readiness_sha256"],
        "external_freeze_commitments": dict(external_freeze_commitments),
        "preregistration_authorization": dict(preregistration_authorization),
        "route_snapshot_evidence": dict(route_snapshot_evidence),
        "source_snapshot": dict(source_snapshot),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "packages": {
                name: _package_version(name)
                for name in ("pydantic", "pydantic-ai", "openai", "httpx")
            },
            "credential_variable": "OPENROUTER_API_KEY",
        },
    }


def _failure_analysis(mode: str, exc: BaseException) -> dict[str, object]:
    return {
        "schema_version": 1,
        "development_only": True,
        "evidence_class": "engineering_mechanism_probe_only",
        "benchmark_claim_allowed": False,
        "wall_clock_claim_allowed": False,
        "scripted_fixture": True,
        "causal_effect_identifiable": False,
        "model_reasoning_quality_tested": False,
        "mode": mode,
        "overall_pass": False,
        "hypothesis_outcome": "execution_failed_before_complete_analysis",
        "failure_type": type(exc).__name__,
        "safe_message": (
            str(exc)[:1_024]
            if type(exc).__module__.startswith("agent_evolve")
            or type(exc).__module__.startswith("examples")
            else "probe failed; inspect sanitized durable events"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preview", action="store_true")
    modes.add_argument("--offline", action="store_true")
    modes.add_argument("--live", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--expected-source-sha256", default=None)
    parser.add_argument("--expected-readiness-sha256", default=None)
    return parser


def _mode(args: argparse.Namespace) -> str:
    if args.live:
        return "live"
    if args.offline:
        return "offline"
    return "preview"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = _mode(args)
    _validate_live_cli_freeze_requirements(
        mode=mode,
        run_id=args.run_id,
        expected_source_sha256=args.expected_source_sha256,
        expected_readiness_sha256=args.expected_readiness_sha256,
        log_root=args.log_root,
    )
    preregistration_authorization = _preregistration_authorization_evidence(
        mode=mode,
        run_id=args.run_id or "not-live",
        expected_source_sha256=args.expected_source_sha256,
        expected_readiness_sha256=args.expected_readiness_sha256,
    )
    run_id = _validate_run_id(
        args.run_id
        or datetime.now(timezone.utc).strftime(f"v6_closed_loop_{mode}_%Y%m%dT%H%M%SZ")
    )
    run_dir = args.log_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _directory_fsync(run_dir.parent)

    analysis_path = run_dir / "mechanism_analysis.json"
    summary_path = run_dir / "summary.json"
    writers: list[DurableJsonlWriter] = []
    event_writer: DurableJsonlWriter | None = None
    outcome_writer: DurableJsonlWriter | None = None
    queue_writer: DurableJsonlWriter | None = None
    source_verification_writer: DurableJsonlWriter | None = None
    summary: dict[str, object] | None = None
    status = "failed"
    pending_error: BaseException | None = None
    pending_traceback = None
    try:
        event_writer = DurableJsonlWriter(run_dir / "events.jsonl")
        writers.append(event_writer)
        outcome_writer = DurableJsonlWriter(run_dir / "outcomes.jsonl")
        writers.append(outcome_writer)
        queue_writer = DurableJsonlWriter(run_dir / "queue_outcomes.jsonl")
        writers.append(queue_writer)
        source_verification_writer = DurableJsonlWriter(
            run_dir / "source_verifications.jsonl"
        )
        writers.append(source_verification_writer)

        source_snapshot = _snapshot_sources(run_dir)
        readiness = run_async_sync(prepare_readiness())
        route_evidence = _route_snapshot_evidence()
        _write_json(run_dir / "source_snapshot.json", source_snapshot)
        _write_json(run_dir / "readiness.json", readiness)
        external_commitments = _external_freeze_commitments(
            mode=mode,
            expected_source_sha256=args.expected_source_sha256,
            expected_readiness_sha256=args.expected_readiness_sha256,
            source_snapshot=source_snapshot,
            readiness=readiness,
        )
        _write_json(
            run_dir / "manifest.json",
            _manifest(
                run_id=run_id,
                mode=mode,
                readiness=readiness,
                source_snapshot=source_snapshot,
                external_freeze_commitments=external_commitments,
                preregistration_authorization=preregistration_authorization,
                route_snapshot_evidence=route_evidence,
            ),
        )

        if mode == "preview":
            source_verification_writer.write(
                _verify_source_snapshot(
                    source_snapshot,
                    stage="post_preview_readiness",
                )
            )
            outcome_writer.write(
                {
                    "schema_version": 1,
                    "record_type": "readiness_preview",
                    "provider_io_performed": False,
                    "readiness_sha256": readiness["readiness_sha256"],
                }
            )
            analysis = {
                "schema_version": 1,
                "development_only": True,
                "evidence_class": "engineering_mechanism_probe_only",
                "benchmark_claim_allowed": False,
                "wall_clock_claim_allowed": False,
                "mode": mode,
                "overall_pass": None,
                "hypothesis_outcome": "not_executed_readiness_only",
                "provider_io_performed": False,
                "readiness_sha256": readiness["readiness_sha256"],
            }
            summary = {
                "schema_version": 1,
                "status": "preview_ready",
                "development_only": True,
                "readiness_sha256": readiness["readiness_sha256"],
            }
        else:
            gate = ReadinessTraceGate(readiness, event_writer.write)
            if mode == "offline":
                composition = compose_probe(
                    generator_factory=offline_generator_factory,
                    trace_sink=gate,
                )
                source_verification_writer.write(
                    _verify_source_snapshot(
                        source_snapshot,
                        stage="pre_offline_execution",
                    )
                )
                result = run_async_sync(
                    composition.optimizer.run(({"a": 4, "b": 4},))
                )
                source_verification_writer.write(
                    _verify_source_snapshot(
                        source_snapshot,
                        stage="post_offline_execution",
                    )
                )
            else:
                if (
                    external_commitments.get("all_provided_commitments_match")
                    is not True
                ):
                    raise RuntimeError(
                        "live external freeze commitments were not verified"
                    )
                if route_evidence.get("route_validated") is not True:
                    raise RuntimeError("frozen live route evidence was not verified")
                source_verification_writer.write(
                    _verify_source_snapshot(
                        source_snapshot,
                        stage="pre_live_credential_load",
                    )
                )
                load_credentials(WORKSPACE_ROOT / ".env", override=False, optional=True)
                api_key = os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENROUTER_API_KEY is unavailable")
                stack = create_live_stack(
                    api_key=api_key,
                    queue_sink=lambda outcome: queue_writer.write(
                        structured_generation_outcome_record(outcome)
                    ),
                )
                composition = compose_probe(
                    generator_factory=_live_generator_factory(stack.generator),
                    trace_sink=gate,
                )

                async def run_live() -> OptimizerResult:
                    source_verification_writer.write(
                        _verify_source_snapshot(
                            source_snapshot,
                            stage="pre_queue_enter",
                        )
                    )
                    live_result: OptimizerResult | None = None
                    live_error: BaseException | None = None
                    live_traceback = None
                    try:
                        async with stack.runner:
                            live_result = await composition.optimizer.run(
                                ({"a": 4, "b": 4},)
                            )
                    except BaseException as exc:
                        live_error = exc
                        live_traceback = exc.__traceback__
                    try:
                        source_verification_writer.write(
                            _verify_source_snapshot(
                                source_snapshot,
                                stage="post_queue_exit",
                            )
                        )
                    except BaseException as verification_exc:
                        if live_error is None:
                            raise
                        live_error.add_note(
                            "post-queue source verification also failed: "
                            f"{type(verification_exc).__name__}"
                        )
                    if live_error is not None:
                        raise live_error.with_traceback(live_traceback)
                    if live_result is None:
                        raise RuntimeError("live optimizer returned no result")
                    return live_result

                result = run_async_sync(run_live())
            gate.require_complete()
            publish_outcomes(result, outcome_writer)
            analysis = run_async_sync(
                mechanism_analysis(
                    composition=composition,
                    result=result,
                    events=_read_jsonl(event_writer.path),
                    queue_records=_read_jsonl(queue_writer.path),
                    mode=mode,
                )
            )
            summary = {
                "schema_version": 1,
                "status": (
                    "engineering_fixture_path_passed"
                    if analysis["overall_pass"] is True
                    else "engineering_fixture_path_failed"
                ),
                "development_only": True,
                "mode": mode,
                "optimizer_result_hash": result.result_hash,
                "unique_evaluations": result.final_state.unique_evaluations,
                "logical_llm_calls": result.final_state.logical_llm_calls,
                "mechanism_analysis_sha256": canonical_record_sha256(analysis),
            }
        _write_json(analysis_path, analysis)
        _write_json(summary_path, summary)
        status_value = summary.get("status")
        if type(status_value) is not str:
            raise TypeError("summary status must be a string")
        status = status_value
    except BaseException as exc:
        pending_error = exc
        pending_traceback = exc.__traceback__
        status = "failed"
        try:
            failure = _failure_analysis(mode, exc)
            if not analysis_path.exists():
                _write_json(analysis_path, failure)
            _write_json(
                run_dir / "failure.json",
                {
                    "failure_type": type(exc).__name__,
                    "safe_message": failure["safe_message"],
                },
            )
        except BaseException as artifact_exc:
            exc.add_note(
                "failure-artifact publication also failed: "
                f"{type(artifact_exc).__name__}"
            )
    finally:
        for writer in reversed(writers):
            try:
                writer.close()
            except BaseException as close_exc:
                status = "failed"
                if pending_error is None:
                    pending_error = close_exc
                    pending_traceback = close_exc.__traceback__
                else:
                    pending_error.add_note(
                        f"JSONL close also failed: {type(close_exc).__name__}"
                    )
        try:
            _directory_fsync(run_dir)
            _finalize_run(run_dir, status=status)
        except BaseException as finalize_exc:
            status = "failed"
            if pending_error is None:
                pending_error = finalize_exc
                pending_traceback = finalize_exc.__traceback__
            else:
                pending_error.add_note(
                    "recursive run finalization also failed: "
                    f"{type(finalize_exc).__name__}"
                )

    if pending_error is not None:
        raise pending_error.with_traceback(pending_traceback)
    if summary is None:
        raise RuntimeError("probe completed without a summary")

    print(
        _canonical_json(
            {
                "run_dir": str(run_dir),
                "status": summary["status"],
                "development_only": True,
            }
        )
    )
    return 0 if not str(summary["status"]).endswith("_failed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
