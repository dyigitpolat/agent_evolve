#!/usr/bin/env python3
"""Provider-free gate and live composition for Airfoil-v7 AgentEvolve.

This is deliberately a composition root, not an Airfoil implementation.  The
benchmark module prepares G1 evidence and owns the sealed-oracle firewall; the
generic two-stage coordinator owns M/P/N forecasting, allocation, and
evaluation ordering; Pydantic-AI owns typed provider translation; and the
shared queued runner owns physical transport retries.  This module binds those
ports to one prospectively frozen four-call development run and persists the
inter-phase barriers.

``readiness`` never loads ``.env``, reads a credential, creates a provider
client, or dispatches a request.  ``live`` is the only CLI branch allowed to
load ``OPENROUTER_API_KEY`` and it first verifies a finalized readiness gate
against the current closed source set.  Unit tests inject all provider-facing
dependencies; this file itself never launches a provider during import.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
from typing import Any, BinaryIO, Literal, Protocol
import xml.etree.ElementTree as ElementTree


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from pydantic import BaseModel  # noqa: E402

from agent_evolve.settings import load_credentials  # noqa: E402
from agent_evolve.application.action_allocation import (  # noqa: E402
    GREEDY_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256,
    GREEDY_RISK_DIVERSITY_ALLOCATOR_ID,
    GREEDY_RISK_DIVERSITY_ALLOCATOR_VERSION,
    GreedyRiskAdjustedDiversityAllocator,
)
from agent_evolve.application.reflection_workflow import (  # noqa: E402
    PlannedReflectionBatchCall,
    ReflectionWorkflowResult,
    StrictBatchedReflectionWorkflow,
)
from agent_evolve.application.two_stage_action_evolution import (  # noqa: E402
    ActionEvaluationReuseMode,
    ActionEvaluationReusePolicyBinding,
    ActionForecastArmPlan,
    FiniteActionEvaluatorBinding,
    PreparedTwoStageActionEvolution,
    PreparedTwoStageActionEvolutionRequest,
    SCIENTIFIC_ARM_ORDER,
    TwoStageActionPhase,
    TwoStageActionPhaseCommit,
    required_scientific_phase_commit_policy,
)
from agent_evolve.domain.llm_task_queue import QueueSnapshot  # noqa: E402
from agent_evolve.domain.ids import LLMCallId, RunId  # noqa: E402
from agent_evolve.domain.typed_json import (  # noqa: E402
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)
from agent_evolve.infrastructure.ids import DeterministicIdFactory  # noqa: E402
from agent_evolve.integrations.pydantic_ai import (  # noqa: E402
    action_forecast as action_forecast_integration,
)
from agent_evolve.integrations.pydantic_ai import (  # noqa: E402
    agentic_generator as agentic_generator_integration,
)
from agent_evolve.integrations.pydantic_ai import (  # noqa: E402
    progress_aware_openrouter,
)
from agent_evolve.integrations.pydantic_ai import queued_runner  # noqa: E402
from agent_evolve.integrations.pydantic_ai.action_forecast import (  # noqa: E402
    PydanticAIActionForecastPolicy,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (  # noqa: E402
    OpenRouterReasoningConfig,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (  # noqa: E402
    ProgressAwareOpenRouterConfig,
    ProgressAwareRetryMode,
    create_progress_aware_openrouter_runner,
)
from agent_evolve.ports.action_forecast import (  # noqa: E402
    ActionForecastPolicy,
    ActionForecastResult,
    ActionForecastRequest,
)
from agent_evolve.ports.agentic_generator import (  # noqa: E402
    AgenticCallTelemetry,
    InsightDraft,
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.artifact_store import (  # noqa: E402
    canonical_json_bytes,
    decode_json_bytes,
)
from agent_evolve.ports.portfolio_selection import (  # noqa: E402
    PortfolioExperimentalArm,
)
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    StructuredStreamCleanupPolicy,
    StructuredStreamLivenessPolicy,
    StructuredStreamProgress,
)
from examples.development import (  # noqa: E402
    airfoil_v7_two_stage_agent_evolution as airfoil_preparation,
)
from examples.development.airfoil_v7_two_stage_agent_evolution import (  # noqa: E402
    EVALUATOR_DEFINITION_SHA256,
    EVALUATOR_POLICY_ID,
    EVALUATOR_POLICY_VERSION,
    G2_PORTFOLIO_SIZE,
    MAX_OUTPUT_TOKENS,
    AirfoilTwoStageForecastArms,
    PreparedAirfoilTwoStageGeneration,
    build_airfoil_v7_forecast_arms,
)
from examples.development import durable_run_artifacts  # noqa: E402
from examples.development.durable_run_artifacts import (  # noqa: E402
    BatchedDurableJsonlJournal,
    DurableJsonlJournal,
    file_identity,
    finalize_run_directory,
    read_jsonl,
    source_identity,
    verify_finalized_run_directory,
    write_bytes_atomic,
    write_json_atomic,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
DEFAULT_RUN_ROOT = ARTIFACT_ROOT / "experiment_logs" / "airfoil_v7" / "two_stage"
PREREGISTRATION = (
    ARTIFACT_ROOT
    / "132_airfoil_v7_compact_semantic_two_stage_generation_preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "364df211914ca314f7f8128a786787e02bfd5afae1696066549d7a82adb81349"
)
CAPABILITY_SNAPSHOT = (
    ARTIFACT_ROOT
    / "data"
    / "openrouter_deepseek_v4_pro_streamlake_capability_snapshot_20260714.json"
)
CAPABILITY_SNAPSHOT_SHA256 = (
    "131d0fef27cb24350f9c067ea7407cd9279ddbe242eef77e29451390a750a671"
)
PRICING_SNAPSHOT = (
    ARTIFACT_ROOT
    / "data"
    / "openrouter_deepseek_v4_pro_streamlake_pricing_snapshot_20260714.json"
)
PRICING_SNAPSHOT_SHA256 = (
    "5adea5e08d7aea5eb89de010e1750890fe6b7f70a3f7fe733a08996d0b8b7204"
)

FROZEN_LIVE_RUN_ID = "ae7_generic_two_stage_generation_v2_20260715"
MODEL = "deepseek/deepseek-v4-pro"
CANONICAL_MODEL = "deepseek/deepseek-v4-pro-20260423"
ALLOWED_RESOLVED_MODELS = (MODEL, CANONICAL_MODEL)
PROVIDER_SLUG = "streamlake"
RESOLVED_PROVIDER = "StreamLake"
CONNECT_TIMEOUT_SECONDS = 90.0
FIRST_EVENT_TIMEOUT_SECONDS = 180
IDLE_TIMEOUT_SECONDS = 120
MAX_ATTEMPTS = 2
BASE_BACKOFF_NS = 1_000_000_000
MAX_BACKOFF_NS = 30_000_000_000
JITTER_SEED = 2_026_071_501
JITTER_DOMAIN = "airfoil-v7-generic-two-stage-v2"
FORECAST_CONCURRENCY = 3
MAX_PENDING = 3
PROGRESS_MAX_UNFSYNCED_ROWS = 64
RISK_AVERSION = 0.5
DIVERSITY_WEIGHT = 0.25

# Provider-visible call IDs contain no arm names.  The mapping appears only in
# the control-plane manifest and is frozen before a provider can be created.
REFLECTION_CALL_NAMESPACE = "ae7x4v2"
OPAQUE_FORECAST_CALL_IDS: Mapping[PortfolioExperimentalArm, LLMCallId] = {
    PortfolioExperimentalArm.MEMORY: LLMCallId(
        "call_airfoil_twostage_forecast_001"
    ),
    PortfolioExperimentalArm.PERMUTED_PLACEBO: LLMCallId(
        "call_airfoil_twostage_forecast_002"
    ),
    PortfolioExperimentalArm.NEUTRAL: LLMCallId(
        "call_airfoil_twostage_forecast_003"
    ),
}

_SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FRAMING = b"agent-evolve:airfoil-v7-two-stage-manifest:v1\x00"
_RELEASE_FRAMING = b"agent-evolve:airfoil-v7-two-stage-readiness:v1\x00"
_FOCUSED_TEST_FRAMING = b"agent-evolve:focused-test-execution:v1\x00"

FOCUSED_TEST_RELATIVE_PATHS = (
    "tests/test_run_airfoil_v7_two_stage_generation.py",
    "tests/test_airfoil_v7_two_stage_agent_evolution.py",
    "tests/test_agentic_public_api.py",
    "tests/test_action_semantics.py",
    "tests/test_two_stage_action_evolution.py",
    "tests/test_action_forecast_allocation.py",
    "tests/test_insight_memory.py",
    "tests/test_portfolio_selection.py",
    "tests/test_reflection_causal_evidence.py",
    "tests/test_strict_batched_reflection_workflow.py",
    "tests/test_pydantic_agentic_generator.py",
    "tests/test_pydantic_action_forecast.py",
    "tests/test_progress_aware_openrouter.py",
    "tests/test_async_structured_generator.py",
    "tests/test_queued_structured_runner.py",
    "tests/test_stream_liveness.py",
)
FOCUSED_TEST_SUPPORT_RELATIVE_PATHS = ("tests/conftest.py",)


class AirfoilTwoStageRunError(RuntimeError):
    """Sanitized fail-closed error for this one scientific composition."""


@dataclass(frozen=True, slots=True)
class FirewalledAirfoilPreparation:
    """Small local adapter around the evolving benchmark preparation API.

    The actual G2 evaluator is deliberately not carried here.  It is created as
    a closed deferred capability only after the allocation phase is fsynced.
    """

    preparation: PreparedAirfoilTwoStageGeneration
    predecision_firewall_record: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.preparation) is not PreparedAirfoilTwoStageGeneration:
            raise TypeError("preparation must be exact PreparedAirfoilTwoStageGeneration")
        self.preparation.__post_init__()
        if not isinstance(self.predecision_firewall_record, Mapping):
            raise TypeError("predecision_firewall_record must be a mapping")
        record = dict(self.predecision_firewall_record)
        if (
            record != self.preparation.evaluator.firewall_record()
            or
            type(record.get("authenticated_seal")) is not dict
            or record.get("g1_outcomes_materialized") is not True
            or record.get("non_g1_outcomes_materialized") is not False
            or record.get("g2_opened") is not False
            or record.get("predecision_oracle_result_json_decoded") is not False
        ):
            raise AirfoilTwoStageRunError(
                "predecision Airfoil input bundle lacks a closed G1-only firewall"
            )


class Runner(Protocol):
    async def __aenter__(self) -> "Runner": ...

    async def __aexit__(self, *_: object) -> None: ...

    async def __call__(self, request: StructuredGenerationRequest[Any]) -> object: ...

    async def snapshot(self) -> object: ...


RunnerFactory = Callable[..., Runner]


@dataclass(frozen=True, slots=True)
class LiveDependencies:
    runner_factory: RunnerFactory = create_progress_aware_openrouter_runner


@dataclass(slots=True)
class ClaimedLiveRun:
    """Exclusive pre-credential ownership of the exact frozen target directory."""

    run_dir: Path
    release_gate_dir: Path
    release_gate: dict[str, object]
    release_gate_finalization: dict[str, object]
    claim_record: dict[str, object]
    _lock_stream: BinaryIO
    active: bool = True

    def close(self) -> None:
        if not self.active:
            return
        try:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_stream.close()
            self.active = False


def _sha256(value: object, framing: bytes) -> str:
    return hashlib.sha256(framing + canonical_json_bytes(value)).hexdigest()


def _frozen_object(value: Mapping[str, object]) -> FrozenJsonObject:
    frozen = freeze_json(dict(value))
    if type(frozen) is not FrozenJsonObject:
        raise AssertionError("expected a frozen JSON object")
    return frozen


def _read_bound_object(path: Path, expected_sha256: str) -> dict[str, object]:
    content = path.expanduser().resolve(strict=True).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise AirfoilTwoStageRunError(f"frozen file changed: {path.name}")
    value = decode_json_bytes(content)
    if type(value) is not dict:
        raise AirfoilTwoStageRunError(f"frozen file is not an object: {path.name}")
    return value


def preregistration_identity() -> dict[str, object]:
    content = PREREGISTRATION.expanduser().resolve(strict=True).read_bytes()
    if hashlib.sha256(content).hexdigest() != PREREGISTRATION_SHA256:
        raise AirfoilTwoStageRunError("artifact 132 identity changed")
    return file_identity(PREREGISTRATION, relative_to=WORKSPACE_ROOT)


def route_binding() -> dict[str, object]:
    capability = _read_bound_object(CAPABILITY_SNAPSHOT, CAPABILITY_SNAPSHOT_SHA256)
    pricing = _read_bound_object(PRICING_SNAPSHOT, PRICING_SNAPSHOT_SHA256)
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
        raise AirfoilTwoStageRunError("frozen StreamLake route semantics changed")
    return {
        "requested_model": MODEL,
        "canonical_model": CANONICAL_MODEL,
        "allowed_resolved_models": list(ALLOWED_RESOLVED_MODELS),
        "provider_name": RESOLVED_PROVIDER,
        "provider_request_slug": PROVIDER_SLUG,
        "provider_options": {"only": [PROVIDER_SLUG], "allow_fallbacks": False},
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "capability_snapshot_sha256": CAPABILITY_SNAPSHOT_SHA256,
        "pricing_snapshot_sha256": PRICING_SNAPSHOT_SHA256,
    }


def output_authorization_binding() -> dict[str, object]:
    """Disclose the tested provider/local structured-output trust boundary."""

    return {
        "provider_visible_tool_schema": True,
        "provider_tool_strict_flag": False,
        "provider_enforced_schema_claimed": False,
        "local_pydantic_model_config": {
            "strict": True,
            "extra": "forbid",
            "frozen": True,
            "validate_default": True,
        },
        "exact_local_scientific_validation_required": True,
        "schema_repair_authorized": False,
        "logical_rerun_authorized": False,
    }


def _invocation_python_executable() -> str:
    """Return the absolute invocation path without resolving a venv symlink."""

    value = os.path.abspath(sys.executable)
    if not Path(value).is_file():
        raise AirfoilTwoStageRunError("invocation Python executable is unavailable")
    return value


def runtime_identity() -> dict[str, object]:
    """Bind the interpreter and transport-critical installed distributions."""

    package_versions: dict[str, str] = {}
    for distribution in ("pydantic-ai", "pydantic", "openai", "httpx", "pytest"):
        try:
            package_versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as error:
            raise AirfoilTwoStageRunError(
                f"required runtime distribution is unavailable: {distribution}"
            ) from error
    invocation_executable = _invocation_python_executable()
    resolved_executable = str(Path(invocation_executable).resolve(strict=True))
    return {
        # Keep both identities: resolving the venv launcher symlink would make
        # a replay command incorrectly target the base interpreter.
        "python_executable": invocation_executable,
        "python_executable_resolved": resolved_executable,
        "python_prefix": os.path.abspath(sys.prefix),
        "python_base_prefix": os.path.abspath(sys.base_prefix),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions,
    }


def build_config() -> ProgressAwareOpenRouterConfig:
    return ProgressAwareOpenRouterConfig(
        model_name=MODEL,
        provider_only=(PROVIDER_SLUG,),
        connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS,
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=FIRST_EVENT_TIMEOUT_SECONDS * 1_000_000_000,
            idle_timeout_ns=IDLE_TIMEOUT_SECONDS * 1_000_000_000,
            absolute_timeout_ns=None,
            cleanup_policy=StructuredStreamCleanupPolicy(
                cancel_drain_timeout_ns=5_000_000_000,
                transport_retire_timeout_ns=5_000_000_000,
            ),
        ),
        max_connections=FORECAST_CONCURRENCY,
        max_pending=MAX_PENDING,
        max_attempts=MAX_ATTEMPTS,
        base_backoff_ns=BASE_BACKOFF_NS,
        max_backoff_ns=MAX_BACKOFF_NS,
        jitter_seed=JITTER_SEED,
        jitter_domain=JITTER_DOMAIN,
        app_title="AgentEvolve Airfoil-v7 generic two-stage generation",
        reasoning_config=OpenRouterReasoningConfig(effort="high"),
        retry_mode=ProgressAwareRetryMode.TRANSPORT_ONLY,
    )


def _source_paths() -> tuple[Path, ...]:
    """Conservative deterministic closure for the one-shot scientific release."""

    paths = {
        *(
            path
            for path in (AGENT_EVOLVE_ROOT / "src" / "agent_evolve").rglob("*.py")
            if path.is_file()
        ),
        *(
            path
            for path in (
                AGENT_EVOLVE_ROOT / "examples" / "benchmarks" / "engibench_airfoil"
            ).rglob("*.py")
            if path.is_file()
        ),
        Path(__file__),
        Path(__file__).resolve().parent / "__init__.py",
        Path(airfoil_preparation.__file__),
        Path(durable_run_artifacts.__file__),
        AGENT_EVOLVE_ROOT / "pyproject.toml",
        AGENT_EVOLVE_ROOT / "uv.lock",
        *(AGENT_EVOLVE_ROOT / value for value in FOCUSED_TEST_RELATIVE_PATHS),
        *(
            AGENT_EVOLVE_ROOT / value
            for value in FOCUSED_TEST_SUPPORT_RELATIVE_PATHS
        ),
    }
    return tuple(sorted(paths, key=lambda path: path.resolve().as_posix()))


def current_source_identity() -> dict[str, object]:
    return source_identity(_source_paths(), relative_to=WORKSPACE_ROOT)


def focused_test_source_identity() -> dict[str, object]:
    return source_identity(
        tuple(
            AGENT_EVOLVE_ROOT / value
            for value in (
                *FOCUSED_TEST_RELATIVE_PATHS,
                *FOCUSED_TEST_SUPPORT_RELATIVE_PATHS,
            )
        ),
        relative_to=WORKSPACE_ROOT,
    )


def _junit_counts(path: Path) -> dict[str, int]:
    content = path.expanduser().resolve(strict=True).read_bytes()
    if not content or len(content) > 20_000_000:
        raise AirfoilTwoStageRunError("focused JUnit size is invalid")
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as error:
        raise AirfoilTwoStageRunError("focused JUnit is invalid XML") from error

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    suites = (
        [root]
        if local(root.tag) == "testsuite"
        else [child for child in root if local(child.tag) == "testsuite"]
    )
    if not suites:
        raise AirfoilTwoStageRunError("focused JUnit has no test suite")
    counts = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for name in counts:
            value = suite.get(name)
            if type(value) is not str or not value.isascii() or not value.isdecimal():
                raise AirfoilTwoStageRunError("focused JUnit counts are invalid")
            counts[name] += int(value)
    if (
        counts["tests"] < 1
        or counts["failures"] != 0
        or counts["errors"] != 0
        or counts["skipped"] != 0
    ):
        raise AirfoilTwoStageRunError("focused JUnit is not an all-pass run")
    return counts


def _focused_test_environment() -> dict[str, str]:
    return {
        "HOME": os.environ.get("HOME", ""),
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": os.pathsep.join(
            (str(AGENT_EVOLVE_ROOT / "src"), str(AGENT_EVOLVE_ROOT))
        ),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TZ": "UTC",
    }


def _focused_test_command(junit_path: Path) -> list[str]:
    return [
        _invocation_python_executable(),
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "-q",
        *(str(AGENT_EVOLVE_ROOT / value) for value in FOCUSED_TEST_RELATIVE_PATHS),
        f"--junitxml={junit_path.resolve(strict=False)}",
    ]


def _focused_test_execution_record(
    *,
    run_dir: Path,
    junit_path: Path,
    source_before: Mapping[str, object],
    source_after: Mapping[str, object],
    return_code: int,
    stdout: bytes,
    stderr: bytes,
) -> dict[str, object]:
    counts = _junit_counts(junit_path)
    record: dict[str, object] = {
        "schema_version": 1,
        "kind": "agent_evolve_focused_test_execution",
        "status": "focused_tests_passed",
        "command": _focused_test_command(junit_path),
        "cwd": str(AGENT_EVOLVE_ROOT),
        "environment": _focused_test_environment(),
        "runtime_identity": runtime_identity(),
        "closed_source_before": dict(source_before),
        "closed_source_after": dict(source_after),
        "focused_test_source_identity": focused_test_source_identity(),
        "return_code": return_code,
        "counts": counts,
        "junit_report": file_identity(junit_path, relative_to=run_dir),
        "pytest_stdout": file_identity(
            run_dir / "pytest.stdout",
            relative_to=run_dir,
        ),
        "pytest_stderr": file_identity(
            run_dir / "pytest.stderr",
            relative_to=run_dir,
        ),
        "credential_read_attempted": False,
        "provider_call_attempted": False,
    }
    unsigned = dict(record)
    record["execution_commitment_sha256"] = _sha256(
        unsigned,
        _FOCUSED_TEST_FRAMING,
    )
    return record


def execute_focused_test_gate(*, run_dir: Path) -> dict[str, object]:
    """Run and finalize the exact provider-free release test command."""

    root = run_dir.expanduser().resolve(strict=False)
    if root.exists():
        raise FileExistsError(root)
    if _SAFE_RUN_ID.fullmatch(root.name) is None:
        raise AirfoilTwoStageRunError("focused test run ID violates closed grammar")
    root.mkdir(parents=True, exist_ok=False)
    junit_path = root / "focused_tests.junit.xml"
    source_before = current_source_identity()
    pending: BaseException | None = None
    return_code = -1
    stdout = b""
    stderr = b""
    try:
        completed = subprocess.run(
            _focused_test_command(junit_path),
            cwd=AGENT_EVOLVE_ROOT,
            env=_focused_test_environment(),
            capture_output=True,
            check=False,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except BaseException as error:
        pending = error
    write_bytes_atomic(root / "pytest.stdout", stdout)
    write_bytes_atomic(root / "pytest.stderr", stderr)
    source_after = current_source_identity()
    try:
        if pending is not None:
            raise pending
        if return_code != 0:
            raise AirfoilTwoStageRunError("focused release tests did not pass")
        if source_before != source_after:
            raise AirfoilTwoStageRunError(
                "closed source set changed during focused release tests"
            )
        record = _focused_test_execution_record(
            run_dir=root,
            junit_path=junit_path,
            source_before=source_before,
            source_after=source_after,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
        )
    except BaseException as error:
        pending = error
        record = {
            "schema_version": 1,
            "kind": "agent_evolve_focused_test_execution",
            "status": "focused_tests_failed",
            "return_code": return_code,
            "failure_type": type(error).__name__,
            "credential_read_attempted": False,
            "provider_call_attempted": False,
        }
    write_json_atomic(root / "focused_test_execution.json", record)
    finalization = finalize_run_directory(root, status=str(record["status"]))
    if pending is not None:
        raise AirfoilTwoStageRunError(
            "focused release test gate failed; inspect finalized artifacts"
        ) from None
    return {
        "run_dir": str(root),
        "execution": record,
        "finalization": finalization,
    }


def verify_focused_test_gate(
    gate_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    root = gate_dir.expanduser().resolve(strict=True)
    finalization = verify_finalized_run_directory(root)
    value = decode_json_bytes((root / "focused_test_execution.json").read_bytes())
    if type(value) is not dict:
        raise AirfoilTwoStageRunError("focused test execution is not an object")
    observed = dict(value)
    commitment = observed.pop("execution_commitment_sha256", None)
    junit_path = root / "focused_tests.junit.xml"
    if (
        finalization.get("status") != "focused_tests_passed"
        or value.get("status") != "focused_tests_passed"
        or type(commitment) is not str
        or commitment != _sha256(observed, _FOCUSED_TEST_FRAMING)
        or value.get("command") != _focused_test_command(junit_path)
        or value.get("cwd") != str(AGENT_EVOLVE_ROOT)
        or value.get("environment") != _focused_test_environment()
        or value.get("runtime_identity") != runtime_identity()
        or value.get("closed_source_before") != current_source_identity()
        or value.get("closed_source_after") != current_source_identity()
        or value.get("focused_test_source_identity")
        != focused_test_source_identity()
        or value.get("return_code") != 0
        or value.get("counts") != _junit_counts(junit_path)
        or value.get("junit_report")
        != file_identity(junit_path, relative_to=root)
        or value.get("pytest_stdout")
        != file_identity(root / "pytest.stdout", relative_to=root)
        or value.get("pytest_stderr")
        != file_identity(root / "pytest.stderr", relative_to=root)
        or value.get("credential_read_attempted") is not False
        or value.get("provider_call_attempted") is not False
    ):
        raise AirfoilTwoStageRunError("focused test execution gate is not current")
    return value, finalization


def _embedded_focused_test_is_current(
    value: object,
    *,
    junit_path: Path,
    readiness_root: Path,
) -> bool:
    if type(value) is not dict:
        return False
    observed = dict(value)
    commitment = observed.pop("execution_commitment_sha256", None)
    command = value.get("command")
    expected = _focused_test_command(junit_path)
    return (
        type(commitment) is str
        and commitment == _sha256(observed, _FOCUSED_TEST_FRAMING)
        and value.get("status") == "focused_tests_passed"
        and type(command) is list
        and len(command) == len(expected)
        and command[:-1] == expected[:-1]
        and type(command[-1]) is str
        and command[-1].endswith("/focused_tests.junit.xml")
        and value.get("cwd") == str(AGENT_EVOLVE_ROOT)
        and value.get("environment") == _focused_test_environment()
        and value.get("runtime_identity") == runtime_identity()
        and value.get("closed_source_before") == current_source_identity()
        and value.get("closed_source_after") == current_source_identity()
        and value.get("focused_test_source_identity")
        == focused_test_source_identity()
        and value.get("return_code") == 0
        and value.get("counts") == _junit_counts(junit_path)
        and value.get("junit_report")
        == file_identity(junit_path, relative_to=readiness_root)
        and value.get("pytest_stdout")
        == file_identity(
            readiness_root / "pytest.stdout",
            relative_to=readiness_root,
        )
        and value.get("pytest_stderr")
        == file_identity(
            readiness_root / "pytest.stderr",
            relative_to=readiness_root,
        )
        and value.get("credential_read_attempted") is False
        and value.get("provider_call_attempted") is False
    )


def _fixture_draft(preparation: PreparedAirfoilTwoStageGeneration, index: int) -> InsightDraft:
    observation = preparation.observations[index]
    predictions = []
    for metric in observation.evaluation.metrics:
        if metric.delta < 0.0:
            direction = MetricEffectDirection.DECREASE
        elif metric.delta > 0.0:
            direction = MetricEffectDirection.INCREASE
        else:
            direction = MetricEffectDirection.UNCHANGED
        predictions.append(MetricEffectPrediction(metric.metric_id, direction))
    return InsightDraft(
        claim=f"Fixture mechanism {index + 1} follows one measured intervention.",
        trigger="The bounded intervention remains legal for the current parent.",
        mechanism="A bounded geometry or trim intervention changes coupled response.",
        affected_paths=(
            "$.alpha_deg"
            if observation.family == "trim_only"
            else "$.upper_coefficients"
        ,),
        evidence_summary="One exact child-minus-parent diagnostic supports this card.",
        confidence=0.5,
        evidence_contrast_ids=(observation.contrast_id,),
        effect_predictions=tuple(sorted(predictions, key=lambda value: value.metric_id)),
        recommended_option_families=(observation.family,),
        recommended_option_ids=(observation.option_id,),
        action_template="Apply the cited bounded action as a measured trial.",
        falsification_condition="Reject if a repeated diagnostic reverses its directions.",
    )


def _response(value: BaseModel, *, suffix: str) -> StructuredGenerationResponse[Any]:
    return StructuredGenerationResponse(
        value=value,
        requested_model="offline/fake",
        resolved_model="offline/fake",
        resolved_provider="offline",
        provider_response_id=f"offline-{suffix}",
        finish_reason="tool_call",
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
    )


def _draft_payload(draft: InsightDraft) -> dict[str, object]:
    return {
        "claim": draft.claim,
        "trigger": draft.trigger,
        "mechanism": draft.mechanism,
        "affected_paths": list(draft.affected_paths),
        "evidence_summary": draft.evidence_summary,
        "confidence": draft.confidence,
        "evidence_contrast_ids": list(draft.evidence_contrast_ids),
        "effect_predictions": [
            {"metric_id": value.metric_id, "direction": value.direction.value}
            for value in draft.effect_predictions
        ],
        "recommended_option_families": list(draft.recommended_option_families),
        "recommended_option_ids": list(draft.recommended_option_ids),
        "action_template": draft.action_template,
        "falsification_condition": draft.falsification_condition,
    }


def structured_request_contract(
    request: StructuredGenerationRequest[Any],
) -> dict[str, object]:
    request.__post_init__()
    schema = request.output_type.model_json_schema()
    return {
        "call_id": request.call_id.value,
        "operation": request.operation,
        "prompt_utf8_bytes": len(request.prompt.encode("utf-8")),
        "prompt_sha256": hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
        "output_type": request.output_type.__name__,
        "output_tool_name": request.output_tool_name,
        "schema_sha256": hashlib.sha256(canonical_json_bytes(schema)).hexdigest(),
        "max_output_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "provider_attempt_id": None,
    }


def _planned_request_payload(
    request: StructuredGenerationRequest[Any],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "call_contract": structured_request_contract(request),
        "prompt": request.prompt,
        "output_json_schema": request.output_type.model_json_schema(),
    }


def _plan_action_forecast_request(
    request: ActionForecastRequest,
) -> StructuredGenerationRequest[Any]:
    """Pure provider-boundary plan matching the production forecast adapter."""

    return action_forecast_integration.plan_action_forecast_request(request)


def _schema_driven_action_forecast_payload(
    output_type: type[BaseModel],
) -> dict[str, object]:
    """Construct one neutral fake response from the provider-visible schema.

    Development fakes deliberately know neither benchmark dimensions nor the
    adapter's private Python annotations.  They consume the same closed enum
    and exact-length JSON Schema that the provider sees, which keeps this seam
    honest when the generic forecast wire changes.
    """

    schema = output_type.model_json_schema()
    properties = schema.get("properties")
    required = schema.get("required")
    if type(properties) is not dict or type(required) is not list:
        raise AssertionError("forecast output schema is not a closed object")

    def enum_values(spec: object) -> tuple[str, ...]:
        if type(spec) is not dict:
            raise AssertionError("forecast code schema must be an object")
        values = spec.get("enum")
        if type(values) is not list or not values or any(
            type(value) is not str for value in values
        ):
            raise AssertionError("forecast code schema must expose a string enum")
        return tuple(values)

    def exact_length(spec: object) -> int:
        if type(spec) is not dict:
            raise AssertionError("forecast array schema must be an object")
        minimum = spec.get("minItems")
        maximum = spec.get("maxItems")
        if type(minimum) is not int or minimum <= 0 or maximum != minimum:
            raise AssertionError("forecast arrays must expose one exact length")
        return minimum

    def vector(field_name: str) -> list[str]:
        spec = properties.get(field_name)
        length = exact_length(spec)
        assert type(spec) is dict
        values = enum_values(spec.get("items"))
        return [values[len(values) // 2] for _ in range(length)]

    def matrix(field_name: str) -> list[list[str]]:
        spec = properties.get(field_name)
        row_count = exact_length(spec)
        assert type(spec) is dict
        row_spec = spec.get("items")
        column_count = exact_length(row_spec)
        assert type(row_spec) is dict
        values = enum_values(row_spec.get("items"))
        code = values[len(values) // 2]
        return [[code for _ in range(column_count)] for _ in range(row_count)]

    expected_fields = {
        "probability_valid_codes",
        "median_effect_codes",
        "lower_uncertainty_codes",
        "upper_uncertainty_codes",
    }
    optional_evidence = "evidence_slot_codes" in properties
    if set(properties) != expected_fields | (
        {"evidence_slot_codes"} if optional_evidence else set()
    ):
        raise AssertionError("unexpected action forecast wire fields")
    if set(required) != set(properties):
        raise AssertionError("every action forecast wire field must be required")

    payload: dict[str, object] = {
        "probability_valid_codes": vector("probability_valid_codes"),
        "median_effect_codes": matrix("median_effect_codes"),
        "lower_uncertainty_codes": matrix("lower_uncertainty_codes"),
        "upper_uncertainty_codes": matrix("upper_uncertainty_codes"),
    }
    if optional_evidence:
        payload["evidence_slot_codes"] = matrix("evidence_slot_codes")
    return payload


def _validate_provider_boundary_blinding(
    requests: Sequence[StructuredGenerationRequest[Any]],
) -> None:
    if len(requests) != 3:
        raise AirfoilTwoStageRunError("forecast wave must contain exactly three calls")
    forbidden = (
        "permuted_placebo",
        "control_arm",
        "portfolioexperimentalarm",
        '"arm"',
    )
    schemas: list[dict[str, object]] = []
    for request in requests:
        request.__post_init__()
        schema = request.output_type.model_json_schema()
        schemas.append(schema)
        provider_visible = "\n".join(
            (
                request.call_id.value,
                request.operation,
                request.output_tool_name,
                request.prompt,
                json.dumps(schema, sort_keys=True, separators=(",", ":")),
            )
        ).casefold()
        if any(value in provider_visible for value in forbidden):
            raise AirfoilTwoStageRunError(
                "provider-visible forecast boundary leaks control-plane identity"
            )
        lowered_call_id = request.call_id.value.casefold()
        if (
            any(value in lowered_call_id for value in ("memory", "placebo", "neutral"))
            or lowered_call_id.endswith(("_m", "_p", "_n"))
        ):
            raise AirfoilTwoStageRunError("forecast call ID is not opaque")
    if (
        requests[0].operation != requests[1].operation
        or requests[0].output_tool_name != requests[1].output_tool_name
        or schemas[0] != schemas[1]
    ):
        raise AirfoilTwoStageRunError(
            "grounded experimental arms differ in provider operation/tool/schema"
        )


class _PreviewLowLevelRunner:
    """Exercise the exact Pydantic schemas without a client or credential."""

    def __init__(
        self,
        preparation: PreparedAirfoilTwoStageGeneration,
    ) -> None:
        self.preparation = preparation
        self.forecast_requests: dict[str, ActionForecastRequest] = {}
        self.requests: list[StructuredGenerationRequest[Any]] = []

    async def __call__(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        self.requests.append(request)
        if request.output_tool_name == agentic_generator_integration.REFLECTION_TOOL_NAME:
            payload = {
                "insights": [
                    _draft_payload(_fixture_draft(self.preparation, index))
                    for index in range(len(self.preparation.observations))
                ]
            }
            return _response(
                request.output_type.model_validate(payload),
                suffix="reflection",
            )
        if request.output_tool_name != action_forecast_integration.ACTION_FORECAST_TOOL_NAME:
            raise AssertionError("unexpected provider-free output tool")
        forecast_request = self.forecast_requests[request.call_id.value]
        payload = _schema_driven_action_forecast_payload(request.output_type)
        return _response(
            request.output_type.model_validate(payload),
            suffix=request.call_id.value,
        )


@dataclass(frozen=True, slots=True)
class ProviderFreeCallPreview:
    reflection: ReflectionWorkflowResult
    arms: AirfoilTwoStageForecastArms
    arm_requests: tuple[ActionForecastRequest, ...]
    structured_calls: tuple[dict[str, object], ...]
    planned_reflection: dict[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "logical_call_count": 4,
            "planned_reflection": self.planned_reflection,
            "arms": self.arms.to_record(),
            "structured_calls": list(self.structured_calls),
            "arm_request_mapping": [
                {
                    "arm": arm.value,
                    "call_id": request.call_id.value,
                    "request_sha256": request.request_sha256,
                }
                for arm, request in zip(
                    SCIENTIFIC_ARM_ORDER,
                    self.arm_requests,
                    strict=True,
                )
            ],
        }


def _aliased_arm_requests(
    arms: AirfoilTwoStageForecastArms,
) -> tuple[ActionForecastRequest, ...]:
    native = (
        arms.memory_request,
        arms.placebo_request,
        arms.catalog_only_request,
    )
    return tuple(
        replace(request, call_id=OPAQUE_FORECAST_CALL_IDS[arm])
        for arm, request in zip(SCIENTIFIC_ARM_ORDER, native, strict=True)
    )


def _bind_arms_to_frozen_call_ids(
    arms: AirfoilTwoStageForecastArms,
) -> AirfoilTwoStageForecastArms:
    """Return one internally consistent arm bundle using release call IDs."""

    requests = _aliased_arm_requests(arms)
    rebound = replace(
        arms,
        memory_request=requests[0],
        placebo_request=requests[1],
        catalog_only_request=requests[2],
    )
    rebound.__post_init__()
    if (
        rebound.memory_request,
        rebound.placebo_request,
        rebound.catalog_only_request,
    ) != requests:
        raise AirfoilTwoStageRunError("forecast arm call-ID rebinding failed")
    return rebound


async def _build_provider_free_preview_async(
    preparation: PreparedAirfoilTwoStageGeneration,
) -> ProviderFreeCallPreview:
    low_level = _PreviewLowLevelRunner(preparation)
    planned: list[PlannedReflectionBatchCall] = []
    reflection = await StrictBatchedReflectionWorkflow().run(
        preparation.reflection_request,
        generator=PydanticAIAgenticGenerator(low_level),
        id_factory=DeterministicIdFactory(REFLECTION_CALL_NAMESPACE),
        call_planned_sink=planned.append,
    )
    if len(planned) != 1:
        raise AirfoilTwoStageRunError("readiness did not plan exactly one reflection")
    arms = _bind_arms_to_frozen_call_ids(
        build_airfoil_v7_forecast_arms(preparation, reflection)
    )
    requests = (
        arms.memory_request,
        arms.placebo_request,
        arms.catalog_only_request,
    )
    # Readiness needs exact provider-visible prompts and schemas, not 240 fake
    # resolved forecasts.  Construct the same low-level requests the public
    # adapter constructs; this avoids quadratic port revalidation while still
    # hashing the exact dynamic schema for every arm.
    forecast_plans = tuple(_plan_action_forecast_request(request) for request in requests)
    _validate_provider_boundary_blinding(forecast_plans)
    low_level.requests.extend(forecast_plans)
    if len(low_level.requests) != 4:
        raise AirfoilTwoStageRunError("readiness preview differs from four calls")
    reflection_plan = planned[0]
    return ProviderFreeCallPreview(
        reflection=reflection,
        arms=arms,
        arm_requests=requests,
        structured_calls=tuple(
            structured_request_contract(request) for request in low_level.requests
        ),
        planned_reflection={
            "call_id": reflection_plan.call_id.value,
            "contrast_ids": list(reflection_plan.contrast_ids),
            "min_insights": reflection_plan.request.min_insights,
            "max_insights": reflection_plan.request.max_insights,
            "max_output_tokens": reflection_plan.request.max_output_tokens,
            "temperature": reflection_plan.request.temperature,
        },
    )


def build_provider_free_preview(
    preparation: PreparedAirfoilTwoStageGeneration,
) -> ProviderFreeCallPreview:
    return asyncio.run(_build_provider_free_preview_async(preparation))


def _allocator_binding() -> dict[str, object]:
    allocator = GreedyRiskAdjustedDiversityAllocator(
        risk_aversion=RISK_AVERSION,
        diversity_weight=DIVERSITY_WEIGHT,
    )
    return {
        "policy_id": GREEDY_RISK_DIVERSITY_ALLOCATOR_ID,
        "policy_version": GREEDY_RISK_DIVERSITY_ALLOCATOR_VERSION,
        "definition_sha256": GREEDY_RISK_DIVERSITY_ALLOCATOR_DEFINITION_SHA256,
        "configuration_sha256": allocator.configuration_sha256,
        "risk_aversion": RISK_AVERSION,
        "diversity_weight": DIVERSITY_WEIGHT,
        "eligible_actions": 72,
        "portfolio_size": G2_PORTFOLIO_SIZE,
        "candidate_extensions_per_arm": 72 + 71 + 70,
    }


def _readiness_record(
    *,
    run_dir: Path,
    target_live_run_dir: Path,
    bundle: FirewalledAirfoilPreparation,
    preview: ProviderFreeCallPreview,
    config: ProgressAwareOpenRouterConfig,
    source: Mapping[str, object],
    focused_junit_path: Path,
    focused_test_execution: Mapping[str, object],
    focused_test_finalization: Mapping[str, object],
) -> dict[str, object]:
    preparation_record = bundle.preparation.to_record()
    record: dict[str, object] = {
        "schema_version": 1,
        "kind": "airfoil_v7_generic_two_stage_provider_free_readiness",
        "status": "ready_provider_not_called",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "readiness_run_dir": str(run_dir),
        "target_live_run_id": target_live_run_dir.name,
        "target_live_run_dir": str(target_live_run_dir),
        "target_live_run_dir_absent": not target_live_run_dir.exists(),
        "preregistration": preregistration_identity(),
        "route": route_binding(),
        "composition": config.to_manifest_record(),
        "output_authorization": output_authorization_binding(),
        "runtime_identity": runtime_identity(),
        "closed_source_identity": dict(source),
        "focused_test_source_identity": focused_test_source_identity(),
        "focused_test_execution": dict(focused_test_execution),
        "focused_test_gate": {
            "finalization_sha256": focused_test_finalization[
                "finalization_sha256"
            ],
            "copied_junit_report": file_identity(focused_junit_path),
            "copied_pytest_stdout": file_identity(run_dir / "pytest.stdout"),
            "copied_pytest_stderr": file_identity(run_dir / "pytest.stderr"),
        },
        "predecision_firewall": dict(bundle.predecision_firewall_record),
        "preparation": preparation_record,
        "call_preview": preview.to_record(),
        "allocator": _allocator_binding(),
        "evaluation_reuse": ActionEvaluationReusePolicyBinding(
            ActionEvaluationReuseMode.PER_ARM
        ).to_record(),
        "durability": {
            "planned_calls": "planned_calls.jsonl",
            "stream_progress": "stream_progress.jsonl",
            "queue_outcomes": "queue_outcomes.jsonl",
            "queue_snapshots": "queue_snapshots.jsonl",
            "phase_commits": "phase_commits.jsonl",
            "all_jsonl_created_before_release": True,
        },
        "credential_read_attempted": False,
        "credentials_read": False,
        "client_constructed": False,
        "provider_call_attempted": False,
        "scientific_result_eligible": False,
    }
    unsigned = dict(record)
    record["readiness_commitment_sha256"] = _sha256(unsigned, _RELEASE_FRAMING)
    return record


def execute_readiness(
    *,
    run_dir: Path,
    target_live_run_dir: Path,
    bundle: FirewalledAirfoilPreparation,
    focused_test_gate_dir: Path,
) -> dict[str, object]:
    """Build and finalize a zero-credential, zero-client release candidate."""

    bundle.__post_init__()
    root = run_dir.expanduser().resolve(strict=False)
    target = target_live_run_dir.expanduser().resolve(strict=False)
    if root.exists():
        raise FileExistsError(root)
    if target.exists():
        raise FileExistsError(target)
    if target.name != FROZEN_LIVE_RUN_ID:
        raise AirfoilTwoStageRunError("target live run ID differs from artifact 132")
    if _SAFE_RUN_ID.fullmatch(root.name) is None:
        raise AirfoilTwoStageRunError("readiness run ID violates closed grammar")
    config = build_config()
    test_execution, test_finalization = verify_focused_test_gate(
        focused_test_gate_dir
    )
    test_gate_root = focused_test_gate_dir.expanduser().resolve(strict=True)
    junit_path = test_gate_root / "focused_tests.junit.xml"
    source_before = current_source_identity()
    preview = build_provider_free_preview(bundle.preparation)
    source_after = current_source_identity()
    if source_before != source_after:
        raise AirfoilTwoStageRunError("closed source set changed during readiness")

    root.mkdir(parents=True, exist_ok=False)
    junit_copy = root / "focused_tests.junit.xml"
    write_bytes_atomic(junit_copy, junit_path.read_bytes())
    write_bytes_atomic(
        root / "pytest.stdout",
        (test_gate_root / "pytest.stdout").read_bytes(),
    )
    write_bytes_atomic(
        root / "pytest.stderr",
        (test_gate_root / "pytest.stderr").read_bytes(),
    )
    planned = DurableJsonlJournal(root / "planned_calls.jsonl")
    progress = BatchedDurableJsonlJournal(
        root / "stream_progress.jsonl",
        max_unfsynced_rows=PROGRESS_MAX_UNFSYNCED_ROWS,
    )
    outcomes = DurableJsonlJournal(root / "queue_outcomes.jsonl")
    snapshots = DurableJsonlJournal(root / "queue_snapshots.jsonl")
    phases = DurableJsonlJournal(root / "phase_commits.jsonl")
    try:
        for ordinal, call in enumerate(preview.structured_calls, start=1):
            planned.append(
                {
                    "schema_version": 1,
                    "ordinal": ordinal,
                    "mode": "provider_free_schema_preview",
                    **call,
                }
            )
        snapshots.append(
            {
                "schema_version": 1,
                "stage": "readiness_no_queue_constructed",
                "max_in_flight": FORECAST_CONCURRENCY,
                "max_pending": MAX_PENDING,
                "max_attempts": MAX_ATTEMPTS,
                "client_constructed": False,
            }
        )
        phases.append(
            {
                "schema_version": 1,
                "phase": "provider_free_readiness",
                "status": "passed",
                "logical_call_count_previewed": 4,
                "provider_calls": 0,
            }
        )
        progress.flush()
    finally:
        planned.close()
        progress.close()
        outcomes.close()
        snapshots.close()
        phases.close()

    record = _readiness_record(
        run_dir=root,
        target_live_run_dir=target,
        bundle=bundle,
        preview=preview,
        config=config,
        source=source_after,
        focused_junit_path=junit_copy,
        focused_test_execution=test_execution,
        focused_test_finalization=test_finalization,
    )
    write_json_atomic(root / "readiness.json", record)
    finalization = finalize_run_directory(root, status="ready_provider_not_called")
    return {
        "run_dir": str(root),
        "readiness": record,
        "finalization": finalization,
    }


def verify_readiness_gate(
    gate_dir: Path,
    *,
    target_live_run_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    root = gate_dir.expanduser().resolve(strict=True)
    finalization = verify_finalized_run_directory(root)
    value = decode_json_bytes((root / "readiness.json").read_bytes())
    if type(value) is not dict:
        raise AirfoilTwoStageRunError("readiness record is not an object")
    observed = dict(value)
    commitment = observed.pop("readiness_commitment_sha256", None)
    target = target_live_run_dir.expanduser().resolve(strict=False)
    if (
        finalization.get("status") != "ready_provider_not_called"
        or type(commitment) is not str
        or _LOWER_SHA256.fullmatch(commitment) is None
        or commitment != _sha256(observed, _RELEASE_FRAMING)
        or value.get("status") != "ready_provider_not_called"
        or value.get("target_live_run_id") != FROZEN_LIVE_RUN_ID
        or value.get("target_live_run_dir") != str(target)
        or value.get("preregistration") != preregistration_identity()
        or value.get("route") != route_binding()
        or value.get("composition") != build_config().to_manifest_record()
        or value.get("output_authorization") != output_authorization_binding()
        or value.get("runtime_identity") != runtime_identity()
        or value.get("closed_source_identity") != current_source_identity()
        or value.get("focused_test_source_identity")
        != focused_test_source_identity()
        or not _embedded_focused_test_is_current(
            value.get("focused_test_execution"),
            junit_path=root / "focused_tests.junit.xml",
            readiness_root=root,
        )
        or type(value.get("focused_test_gate")) is not dict
        or type(value.get("focused_test_gate", {}).get("finalization_sha256"))
        is not str
        or _LOWER_SHA256.fullmatch(
            value.get("focused_test_gate", {}).get("finalization_sha256", "")
        )
        is None
        or value.get("focused_test_gate", {}).get("copied_junit_report")
        != file_identity(root / "focused_tests.junit.xml")
        or value.get("focused_test_gate", {}).get("copied_pytest_stdout")
        != file_identity(root / "pytest.stdout")
        or value.get("focused_test_gate", {}).get("copied_pytest_stderr")
        != file_identity(root / "pytest.stderr")
        or value.get("credential_read_attempted") is not False
        or value.get("credentials_read") is not False
        or value.get("client_constructed") is not False
        or value.get("provider_call_attempted") is not False
        or target.exists()
    ):
        raise AirfoilTwoStageRunError("provider-free readiness gate is not current")
    return value, finalization


def claim_live_run(
    *,
    run_dir: Path,
    release_gate_dir: Path,
    launch_command: Sequence[str],
    bundle: FirewalledAirfoilPreparation,
    post_lock_hook: Callable[[], None] | None = None,
) -> ClaimedLiveRun:
    """Exclusively claim, lock, and fsync the target before credential access."""

    bundle.__post_init__()
    root = run_dir.expanduser().resolve(strict=False)
    if root.name != FROZEN_LIVE_RUN_ID:
        raise AirfoilTwoStageRunError("live run ID differs from artifact 132")
    if isinstance(launch_command, (str, bytes)) or not launch_command or any(
        type(value) is not str or not value for value in launch_command
    ):
        raise ValueError("launch_command must be a non-empty exact string sequence")
    if post_lock_hook is not None and not callable(post_lock_hook):
        raise TypeError("post_lock_hook must be callable or None")
    gate, gate_finalization = verify_readiness_gate(
        release_gate_dir,
        target_live_run_dir=root,
    )
    if (
        gate.get("predecision_firewall")
        != dict(bundle.predecision_firewall_record)
        or gate.get("preparation") != bundle.preparation.to_record()
    ):
        raise AirfoilTwoStageRunError(
            "freshly authenticated preparation differs from readiness gate"
        )
    root.mkdir(parents=True, exist_ok=False)
    lock_path = root / "writer.lock"
    try:
        lock_stream = lock_path.open("xb")
    except BaseException:
        root.rmdir()
        raise
    lock_acquired = False
    post_lock_source: dict[str, object] | None = None
    post_lock_runtime: dict[str, object] | None = None
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_acquired = True
        lock_stream.write(b"airfoil-v7-two-stage-exclusive-writer-v1\n")
        lock_stream.flush()
        os.fsync(lock_stream.fileno())
        if post_lock_hook is not None:
            post_lock_hook()
        # This is the sole post-lock mutable-identity read.  The exact values
        # checked here are reused in the claim record; no check/store gap remains.
        post_lock_source = current_source_identity()
        post_lock_runtime = runtime_identity()
        if (
            post_lock_source != gate.get("closed_source_identity")
            or post_lock_runtime != gate.get("runtime_identity")
        ):
            raise AirfoilTwoStageRunError(
                "source or runtime changed across the exclusive claim boundary"
            )
        command = list(launch_command)
        record: dict[str, object] = {
            "schema_version": 1,
            "kind": "airfoil_v7_two_stage_precredential_claim",
            "run_id": root.name,
            "run_dir": str(root),
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
            "exclusive_directory_creation": True,
            "writer_lock_held": True,
            "writer_lock_path": "writer.lock",
            "writer_pid": os.getpid(),
            "launch_command": command,
            "launch_command_sha256": hashlib.sha256(
                canonical_json_bytes(command)
            ).hexdigest(),
            "closed_source_identity": post_lock_source,
            "runtime_identity": post_lock_runtime,
            "preregistration": gate["preregistration"],
            "route": gate["route"],
            "readiness_commitment_sha256": gate[
                "readiness_commitment_sha256"
            ],
            "readiness_finalization_sha256": gate_finalization[
                "finalization_sha256"
            ],
            "predecision_firewall": dict(bundle.predecision_firewall_record),
            "credential_read_attempted": False,
            "client_constructed": False,
            "provider_call_attempted": False,
        }
        write_json_atomic(root / "precredential_claim.json", record)
        write_json_atomic(root / "provider_free_readiness.json", gate)
        return ClaimedLiveRun(
            run_dir=root,
            release_gate_dir=release_gate_dir.expanduser().resolve(strict=True),
            release_gate=dict(gate),
            release_gate_finalization=dict(gate_finalization),
            claim_record=record,
            _lock_stream=lock_stream,
        )
    except BaseException as error:
        if not lock_acquired:
            lock_stream.close()
            try:
                lock_path.unlink()
                root.rmdir()
            except OSError:
                pass
            raise
        abort_claim = ClaimedLiveRun(
            run_dir=root,
            release_gate_dir=release_gate_dir.expanduser().resolve(strict=True),
            release_gate=dict(gate),
            release_gate_finalization=dict(gate_finalization),
            claim_record={
                "closed_source_identity": gate.get("closed_source_identity"),
                "runtime_identity": gate.get("runtime_identity"),
            },
            _lock_stream=lock_stream,
        )
        try:
            write_json_atomic(
                root / "precredential_claim_abort.json",
                {
                    "schema_version": 1,
                    "status": "pre_dispatch_infrastructure_abort",
                    "failure_stage": "precredential_claim",
                    "failure_type": type(error).__name__,
                    "expected_source_aggregate_sha256": gate.get(
                        "closed_source_identity", {}
                    ).get("aggregate_sha256"),
                    "observed_source_aggregate_sha256": (
                        None
                        if post_lock_source is None
                        else post_lock_source.get("aggregate_sha256")
                    ),
                    "expected_runtime_identity": gate.get("runtime_identity"),
                    "observed_runtime_identity": post_lock_runtime,
                    "credential_read_attempted": False,
                    "client_constructed": False,
                    "provider_call_attempted": False,
                },
            )
        except BaseException:
            # A one-shot artifact write failure must not prevent the independent
            # result/finalization path below from sealing the claimed directory.
            pass
        try:
            finalize_claimed_live_abort(
                claim=abort_claim,
                error=error,
                stage="precredential_claim",
                credential_read_attempted=False,
                credentials_read=False,
            )
        except BaseException:
            abort_claim.close()
        raise AirfoilTwoStageRunError(
            "precredential claim failed; inspect sealed target artifacts"
        ) from None


def _queue_snapshot_record(snapshot: object, *, stage: str) -> dict[str, object]:
    if type(snapshot) is not QueueSnapshot:
        raise TypeError("runner returned a non-QueueSnapshot")
    snapshot.__post_init__()
    return {
        "schema_version": 1,
        "stage": stage,
        "max_in_flight": snapshot.max_in_flight,
        "max_pending": snapshot.max_pending,
        "in_flight": snapshot.in_flight,
        "pending": snapshot.pending,
        "closed": snapshot.closed,
    }


class _ProgressRecorder:
    """Content-blind, per-attempt ordered progress with an fsync barrier."""

    def __init__(self, journal: BatchedDurableJsonlJournal) -> None:
        self._journal = journal
        self._lock = threading.Lock()
        self._last: dict[str, tuple[int, int, int]] = {}
        self._kinds: dict[str, list[str]] = {}
        self._call_by_attempt: dict[str, str] = {}
        self.rows = 0

    def __call__(self, progress: StructuredStreamProgress) -> None:
        if type(progress) is not StructuredStreamProgress:
            raise TypeError("progress must be exact StructuredStreamProgress")
        progress.__post_init__()
        attempt_id = progress.provider_attempt_id
        if attempt_id is None:
            raise AirfoilTwoStageRunError("stream progress lacks attempt identity")
        with self._lock:
            previous = self._last.get(attempt_id)
            expected_sequence = 1 if previous is None else previous[0] + 1
            previous_cumulative = 0 if previous is None else previous[1]
            previous_elapsed = -1 if previous is None else previous[2]
            if (
                progress.sequence != expected_sequence
                or progress.cumulative_content_utf8_bytes
                != previous_cumulative + progress.event_content_utf8_bytes
                or progress.elapsed_ns < previous_elapsed
            ):
                raise AirfoilTwoStageRunError("stream progress ordering is invalid")
            self._journal.append(
                {
                    "schema_version": 1,
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
            )
            self._last[attempt_id] = (
                progress.sequence,
                progress.cumulative_content_utf8_bytes,
                progress.elapsed_ns,
            )
            prior_call = self._call_by_attempt.setdefault(attempt_id, progress.call_id)
            if prior_call != progress.call_id:
                raise AirfoilTwoStageRunError(
                    "one provider attempt emitted progress for multiple calls"
                )
            self._kinds.setdefault(attempt_id, []).append(progress.kind.value)
            self.rows += 1

    def flush(self) -> None:
        self._journal.flush()

    def validate_successful_attempts(
        self,
        outcome_rows: Sequence[Mapping[str, object]],
        *,
        expected_call_ids: Sequence[str],
        expected_prompt_sha256_by_call: Mapping[str, str],
    ) -> dict[str, object]:
        """Prove terminal streams and exact-original payloads for every attempt."""

        observed_call_ids = tuple(row.get("task_id") for row in outcome_rows)
        if (
            len(set(observed_call_ids)) != len(observed_call_ids)
            or set(observed_call_ids) != set(expected_call_ids)
            or set(expected_prompt_sha256_by_call) != set(expected_call_ids)
        ):
            raise AirfoilTwoStageRunError("queue terminal ledger call set changed")
        successful_ids: list[str] = []
        all_attempt_ids: list[str] = []
        for outcome in outcome_rows:
            if outcome.get("status") != "succeeded":
                raise AirfoilTwoStageRunError("one logical call did not succeed")
            task_id = outcome.get("task_id")
            if type(task_id) is not str:
                raise AirfoilTwoStageRunError("queue task identity is malformed")
            attempts = outcome.get("attempts")
            if type(attempts) is not list:
                raise AirfoilTwoStageRunError("queue outcome lacks attempt telemetry")
            for attempt in attempts:
                if type(attempt) is not dict:
                    raise AirfoilTwoStageRunError("attempt telemetry is malformed")
                evidence = attempt.get("request_evidence")
                if (
                    type(evidence) is not dict
                    or evidence.get("variant") != "original"
                    or evidence.get("prompt_sha256")
                    != expected_prompt_sha256_by_call[task_id]
                    or type(evidence.get("provider_attempt_id")) is not str
                    or not evidence.get("provider_attempt_id")
                ):
                    raise AirfoilTwoStageRunError(
                        "physical attempt differs from exact original payload"
                    )
                if type(attempt.get("will_retry")) is not bool:
                    raise AirfoilTwoStageRunError("retry decision is malformed")
                attempt_id = str(evidence["provider_attempt_id"])
                all_attempt_ids.append(attempt_id)
                progress_call = self._call_by_attempt.get(attempt_id)
                if progress_call is not None and progress_call != task_id:
                    raise AirfoilTwoStageRunError(
                        "physical attempt progress is bound to another logical call"
                    )
                if attempt.get("status") == "succeeded":
                    successful_ids.append(attempt_id)
                    if progress_call != task_id:
                        raise AirfoilTwoStageRunError(
                            "successful progress is bound to another logical call"
                        )
        if len(set(all_attempt_ids)) != len(all_attempt_ids):
            raise AirfoilTwoStageRunError("provider attempt identities repeat")
        if not set(self._call_by_attempt).issubset(all_attempt_ids):
            raise AirfoilTwoStageRunError("progress contains an unknown attempt")
        if len(successful_ids) != len(outcome_rows):
            raise AirfoilTwoStageRunError(
                "every successful logical call needs one successful physical attempt"
            )
        for attempt_id in successful_ids:
            kinds = self._kinds.get(attempt_id, [])
            if (
                kinds.count("output_selected") != 1
                or kinds.count("stream_completed") != 1
                or not kinds
                or kinds[-1] != "stream_completed"
            ):
                raise AirfoilTwoStageRunError(
                    "successful attempt lacks unique terminal stream_completed"
                )
        return {
            "provider_attempt_ids": all_attempt_ids,
            "successful_provider_attempt_ids": successful_ids,
            "scheduled_retry_count": sum(
                1
                for outcome in outcome_rows
                for attempt in outcome["attempts"]
                if attempt["will_retry"] is True
            ),
            "exact_original_payloads": True,
        }


class _PlannedLowLevelRunner:
    """Persist exact calls and enforce a precommitted forecast wave."""

    def __init__(
        self,
        delegate: Runner,
        journal: DurableJsonlJournal,
        run_dir: Path,
    ) -> None:
        self._delegate = delegate
        self._journal = journal
        self._run_dir = run_dir
        self._lock = threading.Lock()
        self._submitted: set[str] = set()
        self._planned_payloads: dict[str, dict[str, object]] = {}
        self.count = 0

    @property
    def planned_count(self) -> int:
        return len(self._planned_payloads)

    @property
    def prompt_sha256_by_call(self) -> dict[str, str]:
        return {
            call_id: str(payload["call_contract"]["prompt_sha256"])
            for call_id, payload in self._planned_payloads.items()
        }

    def precommit_forecast_wave(
        self,
        requests: Sequence[StructuredGenerationRequest[Any]],
    ) -> None:
        """Fsync the complete three-request wave before any one can dispatch."""

        plans = tuple(_planned_request_payload(request) for request in requests)
        _validate_provider_boundary_blinding(requests)
        with self._lock:
            if self.count != 1 or len(self._planned_payloads) != 1:
                raise AirfoilTwoStageRunError(
                    "forecast wave must be frozen immediately after reflection"
                )
            call_ids = tuple(
                str(plan["call_contract"]["call_id"]) for plan in plans
            )
            if (
                len(set(call_ids)) != 3
                or any(call_id in self._planned_payloads for call_id in call_ids)
            ):
                raise AirfoilTwoStageRunError("forecast wave call IDs are invalid")
            wave_path = self._run_dir / "planned_forecast_wave.json"
            write_json_atomic(
                wave_path,
                {
                    "schema_version": 1,
                    "status": "durably_precommitted_before_forecast_dispatch",
                    "calls": list(plans),
                },
            )
            for ordinal, (call_id, plan) in enumerate(
                zip(call_ids, plans, strict=True),
                start=2,
            ):
                exact_path = self._run_dir / f"planned_call_{ordinal:02d}.json"
                write_json_atomic(exact_path, plan)
                self._journal.append(
                    {
                        "schema_version": 1,
                        "ordinal": ordinal,
                        "mode": "live_precommitted_forecast_wave",
                        "wave_file": file_identity(
                            wave_path,
                            relative_to=self._run_dir,
                        ),
                        "exact_request_file": file_identity(
                            exact_path,
                            relative_to=self._run_dir,
                        ),
                        **plan["call_contract"],
                    }
                )
                self._planned_payloads[call_id] = plan

    async def __call__(self, request: StructuredGenerationRequest[Any]) -> object:
        payload = _planned_request_payload(request)
        contract = payload["call_contract"]
        call_id = request.call_id.value
        with self._lock:
            if call_id in self._submitted:
                raise AirfoilTwoStageRunError("logical call was submitted twice")
            precommitted = self._planned_payloads.get(call_id)
            if precommitted is not None:
                if precommitted != payload:
                    raise AirfoilTwoStageRunError(
                        "dispatched forecast differs from precommitted exact payload"
                    )
            else:
                if self.count != 0 or self._planned_payloads:
                    raise AirfoilTwoStageRunError(
                        "only the reflection may plan at first submission"
                    )
                exact_path = self._run_dir / "planned_call_01.json"
                write_json_atomic(exact_path, payload)
                self._journal.append(
                    {
                        "schema_version": 1,
                        "ordinal": 1,
                        "mode": "live_exact_reflection_call",
                        "exact_request_file": file_identity(
                            exact_path,
                            relative_to=self._run_dir,
                        ),
                        **contract,
                    }
                )
                self._planned_payloads[call_id] = payload
            self._submitted.add(call_id)
            self.count += 1
        return await self._delegate(request)


def _validate_telemetry(telemetry: AgenticCallTelemetry | None) -> None:
    if type(telemetry) is not AgenticCallTelemetry:
        raise AirfoilTwoStageRunError("provider telemetry is missing")
    telemetry.__post_init__()
    if (
        telemetry.requested_model != MODEL
        or telemetry.resolved_model not in ALLOWED_RESOLVED_MODELS
        or telemetry.resolved_provider != RESOLVED_PROVIDER
        or telemetry.provider_response_id is None
        or telemetry.finish_reason is None
        or telemetry.input_tokens <= 0
        or telemetry.output_tokens <= 0
        or telemetry.cost_usd is None
        or not 1 <= telemetry.attempt_count <= MAX_ATTEMPTS
    ):
        raise AirfoilTwoStageRunError("provider route/telemetry violates artifact 132")


@dataclass(slots=True)
class _RouteValidatedForecastPolicy:
    delegate: ActionForecastPolicy
    run_dir: Path
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    async def forecast(self, request: ActionForecastRequest) -> ActionForecastResult:
        result = await self.delegate.forecast(request)
        if type(result) is not ActionForecastResult:
            raise TypeError("forecast adapter returned another result type")
        result.__post_init__()
        _validate_telemetry(result.telemetry)
        assert result.telemetry is not None
        record = {
            "schema_version": 1,
            "call_id": request.call_id.value,
            "request_sha256": request.request_sha256,
            "forecasts": result.forecasts.to_record(),
            "telemetry": _telemetry_record(result.telemetry),
        }
        with self._lock:
            write_json_atomic(
                self.run_dir / f"forecast_result_{request.call_id.value}.json",
                record,
            )
        return result


class _DeferredSelectedEvaluator:
    """Serve only the selected union eagerly opened by the durable ALLOCATE sink."""

    def __init__(self) -> None:
        self._outcomes: dict[str, object] | None = None
        self._commitment: dict[str, object] | None = None

    def open_from_allocation(
        self,
        arms: AirfoilTwoStageForecastArms,
        phase_commit: TwoStageActionPhaseCommit,
    ) -> dict[str, object]:
        if self._outcomes is not None:
            raise AirfoilTwoStageRunError("G2 capability was opened more than once")
        capability = arms.open_postdecision_evaluation(phase_commit)
        evaluations = capability.evaluate_selected()
        outcomes = {value.option_id: value for value in evaluations}
        if set(outcomes) != set(capability.commitment.selected_option_ids):
            raise AirfoilTwoStageRunError("post-decision oracle returned another union")
        self._outcomes = outcomes
        self._commitment = capability.commitment.to_record()
        return dict(self._commitment)

    async def evaluate(self, request: object) -> FrozenJsonObject:
        from agent_evolve.application.two_stage_action_evolution import (
            FiniteActionEvaluationRequest,
        )

        if type(request) is not FiniteActionEvaluationRequest:
            raise TypeError("evaluation request must be exact")
        request.__post_init__()
        if self._outcomes is None or self._commitment is None:
            raise PermissionError("G2 outcomes remain closed before allocation fsync")
        evaluation = self._outcomes.get(request.option.option_id)
        if evaluation is None:
            raise PermissionError("generic coordinator requested an uncommitted action")
        return _frozen_object(
            {
                "option_id": request.option.option_id,
                "selected_by_arms": [arm.value for arm in request.selected_by_arms],
                "allocation_commitment_sha256": self._commitment[
                    "commitment_sha256"
                ],
                "cached_development_evaluation": evaluation.to_record(),
                "new_cfd_calls": 0,
            }
        )


class _DurableScientificPhaseSink:
    """Fsync phase payloads; ALLOCATE then opens exactly its selected G2 union."""

    def __init__(
        self,
        *,
        run_dir: Path,
        journal: DurableJsonlJournal,
        arms: AirfoilTwoStageForecastArms,
        evaluator: _DeferredSelectedEvaluator,
        preallocation_gate: Callable[[], None],
    ) -> None:
        self._run_dir = run_dir
        self._journal = journal
        self._arms = arms
        self._evaluator = evaluator
        self._preallocation_gate = preallocation_gate
        self._committed: set[TwoStageActionPhase] = set()

    def commit(self, phase_commit: TwoStageActionPhaseCommit) -> None:
        if type(phase_commit) is not TwoStageActionPhaseCommit:
            raise TypeError("phase_commit must be exact TwoStageActionPhaseCommit")
        phase_commit.__post_init__()
        phase = phase_commit.receipt.phase
        if phase in self._committed:
            raise AirfoilTwoStageRunError("a scientific phase was committed twice")
        record = {
            "schema_version": 1,
            "phase_commit": phase_commit.to_record(),
            "payload": thaw_json(phase_commit.payload),
        }
        # write_json_atomic fsyncs the file and directory entry before returning.
        path = self._run_dir / f"phase_{phase.value}.json"
        write_json_atomic(path, record)
        self._journal.append(
            {
                "schema_version": 1,
                "phase": phase.value,
                "phase_commit_receipt_sha256": (
                    phase_commit.receipt.receipt_sha256
                ),
                "payload_sha256": phase_commit.to_record()["payload_sha256"],
                "durable_file": file_identity(path, relative_to=self._run_dir),
            }
        )
        self._committed.add(phase)
        if phase is TwoStageActionPhase.FORECAST:
            self._preallocation_gate()
        elif phase is TwoStageActionPhase.ALLOCATE:
            commitment = self._evaluator.open_from_allocation(
                self._arms,
                phase_commit,
            )
            write_json_atomic(
                self._run_dir / "g2_allocation_commitment.json",
                commitment,
            )


def _telemetry_record(telemetry: AgenticCallTelemetry) -> dict[str, object]:
    return {
        "requested_model": telemetry.requested_model,
        "resolved_model": telemetry.resolved_model,
        "resolved_provider": telemetry.resolved_provider,
        "provider_response_id": telemetry.provider_response_id,
        "finish_reason": telemetry.finish_reason,
        "input_tokens": telemetry.input_tokens,
        "output_tokens": telemetry.output_tokens,
        "reasoning_tokens": telemetry.reasoning_tokens,
        "cache_read_tokens": telemetry.cache_read_tokens,
        "cache_write_tokens": telemetry.cache_write_tokens,
        "cost_usd": None if telemetry.cost_usd is None else str(telemetry.cost_usd),
        "latency_ns": telemetry.latency_ns,
        "attempt_count": telemetry.attempt_count,
    }


def _reflection_result_record(reflection: ReflectionWorkflowResult) -> dict[str, object]:
    reflection.__post_init__()
    generation = reflection.shards[0].generation_result
    return {
        "schema_version": 1,
        "logical_call_ids": [value.value for value in reflection.call_ids],
        "logical_calls_used": reflection.logical_llm_calls_used,
        "telemetry": _telemetry_record(generation.telemetry),
        "cards": [
            {
                "contrast_id": shard.contrast_id,
                "call_id": shard.call_id.value,
                "draft": _draft_payload(shard.draft),
                "draft_content_sha256": shard.draft.content_sha256,
            }
            for shard in reflection.shards
        ],
    }


def _live_manifest(
    *,
    run_dir: Path,
    gate_dir: Path,
    gate: Mapping[str, object],
    gate_finalization: Mapping[str, object],
    bundle: FirewalledAirfoilPreparation,
    closed_source_identity: Mapping[str, object],
    runtime_identity_record: Mapping[str, object],
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "kind": "airfoil_v7_generic_two_stage_live",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider_dispatch_authorized": True,
        "preregistration": preregistration_identity(),
        "route": route_binding(),
        "composition": build_config().to_manifest_record(),
        "output_authorization": output_authorization_binding(),
        "runtime_identity": dict(runtime_identity_record),
        "closed_source_identity": dict(closed_source_identity),
        "provider_free_readiness": {
            "gate_dir": str(gate_dir),
            "readiness_commitment_sha256": gate[
                "readiness_commitment_sha256"
            ],
            "finalization_sha256": gate_finalization["finalization_sha256"],
            "bound_copy": "provider_free_readiness.json",
        },
        "predecision_firewall": dict(bundle.predecision_firewall_record),
        "logical_call_plan": {
            "reflection": 1,
            "concurrent_forecasts": 3,
            "total": 4,
            "no_logical_rerun": True,
            "physical_attempts_per_logical_call": [1, MAX_ATTEMPTS],
            "opaque_forecast_alias_mapping": [
                {"arm": arm.value, "call_id": OPAQUE_FORECAST_CALL_IDS[arm].value}
                for arm in SCIENTIFIC_ARM_ORDER
            ],
        },
        "evaluation_reuse": ActionEvaluationReusePolicyBinding(
            ActionEvaluationReuseMode.PER_ARM
        ).to_record(),
        "allocator": _allocator_binding(),
        "phase_commit_policy": required_scientific_phase_commit_policy().to_record(),
    }
    record["manifest_commitment_sha256"] = _sha256(record, _MANIFEST_FRAMING)
    return record


async def _run_live_async(
    *,
    run_dir: Path,
    bundle: FirewalledAirfoilPreparation,
    api_key: str,
    dependencies: LiveDependencies,
    planned_journal: DurableJsonlJournal,
    progress: _ProgressRecorder,
    outcome_journal: DurableJsonlJournal,
    snapshot_journal: DurableJsonlJournal,
    phase_journal: DurableJsonlJournal,
    expected_source_identity: Mapping[str, object],
) -> dict[str, object]:
    outcome_count = 0
    outcome_rows: list[dict[str, object]] = []

    def outcome_sink(outcome: Any) -> None:
        nonlocal outcome_count
        # Every progress row observed before this terminal callback is durable
        # before the terminal queue outcome can escape the shared runner.
        progress.flush()
        record = queued_runner.structured_generation_outcome_record(outcome)
        outcome_journal.append(record)
        outcome_rows.append(record)
        outcome_count += 1

    runner = dependencies.runner_factory(
        api_key=api_key,
        config=build_config(),
        progress_sink=progress,
        outcome_sink=outcome_sink,
    )
    async with runner:
        write_json_atomic(
            run_dir / "runner_constructed.json",
            {
                "schema_version": 1,
                "runner_constructed": True,
                "provider_call_attempted": False,
                "configuration": build_config().to_manifest_record(),
            },
        )
        snapshot_journal.append(
            _queue_snapshot_record(await runner.snapshot(), stage="before_reflection")
        )
        low_level = _PlannedLowLevelRunner(runner, planned_journal, run_dir)
        reflection = await StrictBatchedReflectionWorkflow().run(
            bundle.preparation.reflection_request,
            generator=PydanticAIAgenticGenerator(low_level),
            id_factory=DeterministicIdFactory(REFLECTION_CALL_NAMESPACE),
        )
        reflection_telemetry = reflection.shards[0].generation_result.telemetry
        _validate_telemetry(reflection_telemetry)
        write_json_atomic(
            run_dir / "reflection_result.json",
            _reflection_result_record(reflection),
        )

        arms = _bind_arms_to_frozen_call_ids(
            build_airfoil_v7_forecast_arms(bundle.preparation, reflection)
        )
        requests = (
            arms.memory_request,
            arms.placebo_request,
            arms.catalog_only_request,
        )
        forecast_plans = tuple(
            _plan_action_forecast_request(request) for request in requests
        )
        low_level.precommit_forecast_wave(forecast_plans)
        write_json_atomic(
            run_dir / "cards_views_requests.json",
            {
                "schema_version": 1,
                "arms": arms.to_record(),
                "arm_requests": [
                    {
                        "arm": arm.value,
                        "request": request.to_record(),
                        "request_sha256": request.request_sha256,
                    }
                    for arm, request in zip(
                        SCIENTIFIC_ARM_ORDER,
                        requests,
                        strict=True,
                    )
                ],
            },
        )
        snapshot_journal.append(
            _queue_snapshot_record(await runner.snapshot(), stage="before_forecast_wave")
        )

        deferred_evaluator = _DeferredSelectedEvaluator()
        expected_call_ids = (
            reflection.call_ids[0].value,
            *(value.call_id.value for value in requests),
        )

        def preallocation_gate() -> None:
            progress.flush()
            if (
                low_level.count != 4
                or low_level.planned_count != 4
                or len(outcome_rows) != 4
            ):
                raise AirfoilTwoStageRunError(
                    "four-call terminal ledger is incomplete before allocation"
                )
            attempt_binding = progress.validate_successful_attempts(
                outcome_rows,
                expected_call_ids=expected_call_ids,
                expected_prompt_sha256_by_call=(
                    low_level.prompt_sha256_by_call
                ),
            )
            forecast_paths = tuple(
                run_dir / f"forecast_result_{request.call_id.value}.json"
                for request in requests
            )
            missing = [
                request.call_id.value
                for request, path in zip(requests, forecast_paths, strict=True)
                if not path.is_file()
            ]
            if missing:
                raise AirfoilTwoStageRunError(
                    "validated forecast result files are incomplete"
                )
            current_source = current_source_identity()
            if current_source != expected_source_identity:
                raise AirfoilTwoStageRunError(
                    "closed source set changed before allocation"
                )
            write_json_atomic(
                run_dir / "preallocation_terminal_barrier.json",
                {
                    "schema_version": 1,
                    "status": "all_four_calls_terminal_and_durable",
                    "expected_call_ids": list(expected_call_ids),
                    "attempt_binding": attempt_binding,
                    "planned_prompt_sha256_by_call": (
                        low_level.prompt_sha256_by_call
                    ),
                    "planned_forecast_wave": file_identity(
                        run_dir / "planned_forecast_wave.json",
                        relative_to=run_dir,
                    ),
                    "planned_calls_prefix": file_identity(
                        run_dir / "planned_calls.jsonl",
                        relative_to=run_dir,
                    ),
                    "stream_progress_prefix": file_identity(
                        run_dir / "stream_progress.jsonl",
                        relative_to=run_dir,
                    ),
                    "queue_outcomes_prefix": file_identity(
                        run_dir / "queue_outcomes.jsonl",
                        relative_to=run_dir,
                    ),
                    "forecast_phase_commit": file_identity(
                        run_dir / f"phase_{TwoStageActionPhase.FORECAST.value}.json",
                        relative_to=run_dir,
                    ),
                    "phase_commit_journal_prefix": file_identity(
                        run_dir / "phase_commits.jsonl",
                        relative_to=run_dir,
                    ),
                    "forecast_results": [
                        file_identity(path, relative_to=run_dir)
                        for path in forecast_paths
                    ],
                    "closed_source_identity": current_source,
                },
            )

        evaluator_binding = FiniteActionEvaluatorBinding(
            evaluator=deferred_evaluator,
            evaluator_id=EVALUATOR_POLICY_ID,
            evaluator_version=EVALUATOR_POLICY_VERSION,
            definition_sha256=EVALUATOR_DEFINITION_SHA256,
        )
        request = PreparedTwoStageActionEvolutionRequest(
            run_id=RunId(f"run_{FROZEN_LIVE_RUN_ID}"),
            arm_plans=tuple(
                ActionForecastArmPlan(arm=arm, request=arm_request)
                for arm, arm_request in zip(
                    SCIENTIFIC_ARM_ORDER,
                    requests,
                    strict=True,
                )
            ),
            g1_option_ids=tuple(
                sorted(member.option_id for member in bundle.preparation.sample.members)
            ),
            portfolio_size=G2_PORTFOLIO_SIZE,
            utility=bundle.preparation.utility,
            evaluator=evaluator_binding,
            evaluation_context=_frozen_object(
                {
                    "benchmark": "airfoil_v7",
                    "development_cached_oracle": True,
                    "new_cfd_calls": 0,
                    "g1_sample_receipt_sha256": (
                        bundle.preparation.sample.receipt_sha256
                    ),
                }
            ),
            evaluation_reuse=ActionEvaluationReusePolicyBinding(
                ActionEvaluationReuseMode.PER_ARM
            ),
            phase_commit_policy=required_scientific_phase_commit_policy(),
        )
        phase_sink = _DurableScientificPhaseSink(
            run_dir=run_dir,
            journal=phase_journal,
            arms=arms,
            evaluator=deferred_evaluator,
            preallocation_gate=preallocation_gate,
        )
        coordinator = PreparedTwoStageActionEvolution(
            forecaster=_RouteValidatedForecastPolicy(
                PydanticAIActionForecastPolicy(low_level),
                run_dir,
            ),
            allocator=GreedyRiskAdjustedDiversityAllocator(
                risk_aversion=RISK_AVERSION,
                diversity_weight=DIVERSITY_WEIGHT,
            ),
        )
        result = await coordinator.run(request, phase_commit_sink=phase_sink)
        snapshot_journal.append(
            _queue_snapshot_record(await runner.snapshot(), stage="after_evaluation")
        )
        write_json_atomic(
            run_dir / "resolved_forecasts.json",
            {
                "schema_version": 1,
                "arms": [
                    {
                        "arm": execution.arm.value,
                        "forecasts": execution.result.forecasts.to_record(),
                        "telemetry": _telemetry_record(execution.result.telemetry),
                    }
                    for execution in result.forecasts
                    if execution.result.telemetry is not None
                ],
            },
        )
        if low_level.count != 4 or outcome_count != 4:
            raise AirfoilTwoStageRunError(
                "terminal ledger differs from four logical provider calls"
            )
        progress.validate_successful_attempts(
            outcome_rows,
            expected_call_ids=expected_call_ids,
            expected_prompt_sha256_by_call=low_level.prompt_sha256_by_call,
        )
        if tuple(receipt.phase for receipt in result.phase_receipts) != tuple(
            TwoStageActionPhase
        ):
            raise AirfoilTwoStageRunError("generic coordinator phase order changed")
        if any(
            allocation.result.decision.candidate_evaluations != 213
            for allocation in result.allocations
        ):
            raise AirfoilTwoStageRunError(
                "Airfoil allocation did not evaluate exactly 213 extensions"
            )
        if current_source_identity() != expected_source_identity:
            raise AirfoilTwoStageRunError("closed source set changed during live run")
        return {
            "schema_version": 1,
            "status": "completed_four_call_development_generation",
            "provider_call_attempted": True,
            "logical_call_count": low_level.count,
            "terminal_queue_outcome_count": outcome_count,
            "progress_row_count": progress.rows,
            "reflection": _reflection_result_record(reflection),
            "two_stage_result": result.to_record(),
            "scientific_scope": "development_mechanism_diagnostic_not_paper_ready",
            "new_cfd_calls": 0,
        }


def _terminal_ledger_summary(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "logical_terminal_count": 0,
            "physical_attempt_count": 0,
            "scheduled_retry_count": 0,
            "extra_physical_attempt_count": 0,
            "scheduled_backoff_ns": [],
            "terminal_statuses": [],
        }
    try:
        rows = read_jsonl(path)
    except (OSError, RuntimeError, ValueError):
        return {
            "logical_terminal_count": 0,
            "physical_attempt_count": 0,
            "scheduled_retry_count": 0,
            "extra_physical_attempt_count": 0,
            "scheduled_backoff_ns": [],
            "terminal_statuses": ["journal_unreadable"],
        }
    physical = 0
    scheduled_retries = 0
    backoffs: list[int] = []
    statuses: list[object] = []
    for row in rows:
        statuses.append(row.get("status"))
        attempts = row.get("attempts")
        if type(attempts) is not list:
            continue
        physical += len(attempts)
        for attempt in attempts:
            if type(attempt) is not dict:
                continue
            if attempt.get("will_retry") is True:
                scheduled_retries += 1
                delay = attempt.get("scheduled_delay_ns")
                if type(delay) is int and delay >= 0:
                    backoffs.append(delay)
    return {
        "logical_terminal_count": len(rows),
        "physical_attempt_count": physical,
        "scheduled_retry_count": scheduled_retries,
        "extra_physical_attempt_count": max(0, physical - len(rows)),
        "scheduled_backoff_ns": backoffs,
        "terminal_statuses": statuses,
    }


def _planned_call_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return len(read_jsonl(path))
    except (OSError, RuntimeError, ValueError):
        return 0


def _incomplete_result(
    *,
    root: Path,
    error: BaseException,
    stage: str,
    credential_read_attempted: bool,
    credentials_read: bool,
) -> dict[str, object]:
    planned_count = _planned_call_count(root / "planned_calls.jsonl")
    if planned_count == 0:
        status = "pre_dispatch_infrastructure_abort"
        scope = "provider_not_called_infrastructure_abort"
    elif not (root / "reflection_result.json").is_file():
        status = "reflection_incomplete"
        scope = "transport_or_contract_incomplete"
    else:
        status = "transport_incomplete"
        scope = "transport_or_contract_incomplete"
    return {
        "schema_version": 1,
        "status": status,
        "failure_stage": stage,
        "failure_type": type(error).__name__,
        "credential_read_attempted": credential_read_attempted,
        "credentials_read": credentials_read,
        "client_constructed": (root / "runner_constructed.json").is_file(),
        "provider_call_attempted": planned_count > 0,
        "planned_logical_call_count": planned_count,
        "scientific_scope": scope,
        "new_cfd_calls": 0,
    }


def finalize_claimed_live_abort(
    *,
    claim: ClaimedLiveRun,
    error: BaseException,
    stage: str,
    credential_read_attempted: bool,
    credentials_read: bool,
) -> dict[str, object]:
    """Durably terminate a claimed run that failed before live dispatch."""

    if type(claim) is not ClaimedLiveRun or not claim.active:
        raise AirfoilTwoStageRunError("abort finalization requires an active claim")
    root = claim.run_dir
    result = _incomplete_result(
        root=root,
        error=error,
        stage=stage,
        credential_read_attempted=credential_read_attempted,
        credentials_read=credentials_read,
    )
    try:
        write_json_atomic(
            root / "pre_dispatch_abort.json",
            {
                "schema_version": 1,
                "status": result["status"],
                "failure_stage": stage,
                "failure_type": type(error).__name__,
                "provider_call_attempted": result["provider_call_attempted"],
            },
        )
        result["terminal_ledger"] = _terminal_ledger_summary(
            root / "queue_outcomes.jsonl"
        )
        write_json_atomic(root / "result.json", result)
        finalization = finalize_run_directory(root, status=str(result["status"]))
        return {
            "run_dir": str(root),
            "result": result,
            "finalization": finalization,
        }
    finally:
        claim.close()


def execute_live(
    *,
    claim: ClaimedLiveRun,
    bundle: FirewalledAirfoilPreparation,
    api_key: str,
    dependencies: LiveDependencies = LiveDependencies(),
) -> dict[str, object]:
    """Run the one authorized reflection plus concurrent M/P/N forecast wave."""

    if type(claim) is not ClaimedLiveRun or not claim.active:
        raise AirfoilTwoStageRunError(
            "live execution requires an active pre-credential writer claim"
        )
    root = claim.run_dir
    gate = claim.release_gate
    gate_finalization = claim.release_gate_finalization
    planned: DurableJsonlJournal | None = None
    progress_journal: BatchedDurableJsonlJournal | None = None
    progress: _ProgressRecorder | None = None
    outcomes: DurableJsonlJournal | None = None
    snapshots: DurableJsonlJournal | None = None
    phases: DurableJsonlJournal | None = None
    result: dict[str, object] | None = None
    pending: BaseException | None = None
    try:
        if type(api_key) is not str or not api_key:
            raise AirfoilTwoStageRunError("live API key is unavailable")
        if type(dependencies) is not LiveDependencies:
            raise TypeError("dependencies must be exact LiveDependencies")
        bundle.__post_init__()
        predispatch_source = current_source_identity()
        predispatch_runtime = runtime_identity()
        claimed_source = claim.claim_record.get("closed_source_identity")
        claimed_runtime = claim.claim_record.get("runtime_identity")
        if (
            type(claimed_source) is not dict
            or type(claimed_runtime) is not dict
            or predispatch_source != claimed_source
            or predispatch_source != gate.get("closed_source_identity")
            or predispatch_runtime != claimed_runtime
            or predispatch_runtime != gate.get("runtime_identity")
            or claim.claim_record.get("predecision_firewall")
            != dict(bundle.predecision_firewall_record)
        ):
            raise AirfoilTwoStageRunError("pre-credential writer claim changed")
        manifest = _live_manifest(
            run_dir=root,
            gate_dir=claim.release_gate_dir,
            gate=gate,
            gate_finalization=gate_finalization,
            bundle=bundle,
            closed_source_identity=claimed_source,
            runtime_identity_record=claimed_runtime,
        )
        if (
            manifest.get("closed_source_identity") != predispatch_source
            or manifest.get("runtime_identity") != predispatch_runtime
        ):
            raise AirfoilTwoStageRunError(
                "live manifest identities differ from pre-dispatch snapshot"
            )
        write_json_atomic(root / "manifest.json", manifest)
        planned = DurableJsonlJournal(root / "planned_calls.jsonl")
        progress_journal = BatchedDurableJsonlJournal(
            root / "stream_progress.jsonl",
            max_unfsynced_rows=PROGRESS_MAX_UNFSYNCED_ROWS,
        )
        progress = _ProgressRecorder(progress_journal)
        outcomes = DurableJsonlJournal(root / "queue_outcomes.jsonl")
        snapshots = DurableJsonlJournal(root / "queue_snapshots.jsonl")
        phases = DurableJsonlJournal(root / "phase_commits.jsonl")
        result = asyncio.run(
            _run_live_async(
                run_dir=root,
                bundle=bundle,
                api_key=api_key,
                dependencies=dependencies,
                planned_journal=planned,
                progress=progress,
                outcome_journal=outcomes,
                snapshot_journal=snapshots,
                phase_journal=phases,
                expected_source_identity=claimed_source,
            )
        )
    except BaseException as error:
        pending = error
        result = _incomplete_result(
            root=root,
            error=error,
            stage="execute_live",
            credential_read_attempted=True,
            credentials_read=type(api_key) is str and bool(api_key),
        )
    finally:
        cleanup_errors: list[BaseException] = []
        if progress is not None:
            try:
                progress.flush()
            except BaseException as error:
                cleanup_errors.append(error)
        for resource in (planned, progress_journal, outcomes, snapshots, phases):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            if current_source_identity() != claim.claim_record.get(
                "closed_source_identity"
            ):
                raise AirfoilTwoStageRunError(
                    "closed source set changed before finalization"
                )
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors and pending is None:
            pending = cleanup_errors[0]
    assert result is not None
    if pending is not None and result.get("status") == (
        "completed_four_call_development_generation"
    ):
        result = {
            **result,
            "status": "transport_incomplete",
            "failure_type": type(pending).__name__,
            "scientific_scope": "transport_or_contract_incomplete",
        }
    planned_count = _planned_call_count(root / "planned_calls.jsonl")
    result["provider_call_attempted"] = planned_count > 0
    result["planned_logical_call_count"] = planned_count
    result["client_constructed"] = (root / "runner_constructed.json").is_file()
    result["credential_read_attempted"] = True
    result["credentials_read"] = type(api_key) is str and bool(api_key)
    result["terminal_ledger"] = _terminal_ledger_summary(
        root / "queue_outcomes.jsonl"
    )
    finalization: dict[str, object] | None = None
    try:
        write_json_atomic(root / "result.json", result)
        finalization = finalize_run_directory(root, status=str(result["status"]))
    except BaseException as error:
        if pending is None:
            pending = error
    finally:
        claim.close()
    if pending is not None:
        raise AirfoilTwoStageRunError(
            "Airfoil two-stage live run failed; inspect finalized artifacts"
        ) from None
    assert finalization is not None
    return {"run_dir": str(root), "result": result, "finalization": finalization}


def _load_dotenv_api_key() -> str:
    """Load one credential at the live CLI boundary only.

    Routed through ``load_credentials`` so a name declared in
    ``AGENTEVOLVE_SCRUBBED`` stays unset. Reading the file directly, as this
    once did, defeated the scrub outright -- it preferred the file's value over
    the process environment, so removing the key changed nothing.
    """

    env_path = WORKSPACE_ROOT / ".env"
    if env_path.is_file():
        load_credentials(env_path, allow_credentials=("OPENROUTER_API_KEY",))
    value = os.environ.get("OPENROUTER_API_KEY")
    if type(value) is not str or not value:
        raise AirfoilTwoStageRunError("OPENROUTER_API_KEY is unavailable")
    return value


def _default_firewalled_bundle(_oracle_dir: Path) -> FirewalledAirfoilPreparation:
    """Normalize the benchmark's G1-only preparation into the local root seam."""

    preparation = airfoil_preparation.prepare_airfoil_v7_two_stage_generation(
        _oracle_dir
    )
    return FirewalledAirfoilPreparation(
        preparation=preparation,
        predecision_firewall_record=preparation.evaluator.firewall_record(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("focused-tests", "readiness", "live"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-live-run-dir", type=Path)
    parser.add_argument("--release-gate-dir", type=Path)
    parser.add_argument("--focused-test-gate-dir", type=Path)
    parser.add_argument(
        "--oracle-dir",
        type=Path,
        default=airfoil_preparation.DEFAULT_SEALED_ORACLE_DIR,
    )
    return parser


def _live_launch_command(argv: Sequence[str] | None) -> list[str]:
    command = [
        _invocation_python_executable(),
        str(Path(__file__).resolve(strict=True)),
    ]
    command.extend(sys.argv[1:] if argv is None else argv)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "focused-tests":
        result = execute_focused_test_gate(run_dir=args.run_dir)
    elif args.mode == "readiness":
        if args.target_live_run_dir is None:
            raise AirfoilTwoStageRunError("readiness requires --target-live-run-dir")
        if args.focused_test_gate_dir is None:
            raise AirfoilTwoStageRunError(
                "readiness requires --focused-test-gate-dir"
            )
        bundle = _default_firewalled_bundle(args.oracle_dir)
        result = execute_readiness(
            run_dir=args.run_dir,
            target_live_run_dir=args.target_live_run_dir,
            bundle=bundle,
            focused_test_gate_dir=args.focused_test_gate_dir,
        )
    else:
        if args.release_gate_dir is None:
            raise AirfoilTwoStageRunError("live requires --release-gate-dir")
        # Verify mutable source/release state before the only credential read.
        bundle = _default_firewalled_bundle(args.oracle_dir)
        command = _live_launch_command(argv)
        claim = claim_live_run(
            run_dir=args.run_dir,
            release_gate_dir=args.release_gate_dir,
            bundle=bundle,
            launch_command=command,
        )
        failure_stage = "credential_load"
        try:
            api_key = _load_dotenv_api_key()
            failure_stage = "execute_live"
            result = execute_live(claim=claim, bundle=bundle, api_key=api_key)
        except BaseException as error:
            if claim.active:
                finalize_claimed_live_abort(
                    claim=claim,
                    error=error,
                    stage=failure_stage,
                    credential_read_attempted=True,
                    credentials_read=failure_stage != "credential_load",
                )
            raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_RESOLVED_MODELS",
    "AirfoilTwoStageRunError",
    "FirewalledAirfoilPreparation",
    "FROZEN_LIVE_RUN_ID",
    "LiveDependencies",
    "MODEL",
    "OPAQUE_FORECAST_CALL_IDS",
    "ProviderFreeCallPreview",
    "build_config",
    "build_provider_free_preview",
    "current_source_identity",
    "claim_live_run",
    "execute_readiness",
    "execute_focused_test_gate",
    "execute_live",
    "finalize_claimed_live_abort",
    "main",
    "preregistration_identity",
    "route_binding",
    "structured_request_contract",
    "verify_readiness_gate",
    "verify_focused_test_gate",
]
