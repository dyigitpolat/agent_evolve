"""Thin live construction and provenance boundary for the Airfoil-v7 G3 run.

Nothing in this module reads ``.env`` or starts OpenRouter/CFD during import,
manifest construction, or optimizer composition.  The production transport is
created lazily on the first model request.  Because the generic optimizer
admits and evaluates seeds before planning G1, that ordering keeps credential
access behind the preflight gate and seed evaluations.  Provider-free runners
inject a fake :class:`AgenticGenerator` into the same optimizer composition
function; there is no second scientific workflow.
"""

from __future__ import annotations

import asyncio
import ast
import json
import re
import subprocess
import sys
import threading
from types import MappingProxyType
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from examples.development.corpus_paths import resolve_corpus_path
from typing import Any, Protocol, runtime_checkable

from agent_evolve.agentic import (
    AgenticGenerator,
    AgenticOptimizerComposition,
    FixedStructuredOutputBudgetPolicy,
    G3_SCREEN_BUDGET,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    VariationGenerationRequest,
    VariationGenerationResult,
    compose_agentic_optimizer,
)
from agent_evolve.application.gated_agentic_generator import (
    AgenticTelemetryPolicy,
    TelemetryGatedAgenticGenerator,
)
from agent_evolve.application.live_runtime_manifest import (
    LiveRuntimeManifest,
    LiveRuntimeManifestError,
    RuntimeFileBinding,
    RuntimeManifestSection,
    RuntimeSourceClosure,
    build_live_runtime_manifest,
    capture_git_worktree_section,
    capture_runtime_environment_section,
    capture_runtime_file,
    capture_runtime_source_closure,
    load_live_runtime_manifest,
)
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (
    OpenRouterReasoningConfig,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (
    ProgressAwareOpenRouterConfig,
    ProgressAwareRetryMode,
    create_progress_aware_openrouter_runner,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    MAX_STRUCTURED_OUTPUT_EVIDENCE_UTF8_BYTES,
    MAX_STRUCTURED_OUTPUT_SCHEMA_UTF8_BYTES,
    STRUCTURED_OUTPUT_EVIDENCE_SCHEMA_VERSION,
    STRUCTURED_REQUEST_EVIDENCE_SCHEMA_VERSION,
    StructuredEvidencePublicationPolicy,
    StructuredOutputEvidenceSink,
    StructuredRequestEvidenceSink,
)
from agent_evolve.ports.structured_generator import (
    StructuredStreamCleanupPolicy,
    StructuredStreamLivenessPolicy,
    StructuredStreamProgressSink,
)
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    local_default_converged_settings,
)
from examples.benchmarks.engibench_airfoil.problem_def import (
    EXPECTED_DATASET_SHA256,
)
from examples.benchmarks.engibench_airfoil.v7_g3_release import (
    ABSOLUTE_Q_DEFINITION_SHA256,
    AIRFOIL_G3_ABSOLUTE_REWARD,
    AIRFOIL_G3_RUNTIME_PHASE,
    DEFAULT_CARD_BANK_PATH,
    DEFAULT_DENYLIST_PATH,
    DEFAULT_FREEZE_RECEIPT_PATH,
    DEFAULT_RELEASE_PATH,
    REQUIRED_METRIC_IDS,
    load_prelaunch_freeze_receipt,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    AIRFOIL_G3_ESTIMAND_STRATUM_SHA256,
    AirfoilG3RuntimeInputs,
)
from examples.benchmarks.engibench_airfoil.v7_readiness import (
    AIRFOIL_V7_CONFLICT_PROBE_ID,
    AIRFOIL_V7_CONFLICT_PROBE_VERSION,
    AIRFOIL_V7_RESOURCE_KEY,
)


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
RESEARCH_ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
G3_RUN_ROOT = RESEARCH_ARTIFACT_ROOT / "experiment_logs" / "airfoil_g3"
G3_WORK_ROOT = Path("/tmp/agent_evolve_airfoil_v7_g3")
DEFAULT_RESOURCE_LEASE_PATH = Path(
    "/tmp/agent_evolve_resource_locks/engibench_airfoil_machaero.lock"
)
ROUTE_DATA_ROOT = RESEARCH_ARTIFACT_ROOT / "data"
SCRIPT_ROOT = RESEARCH_ARTIFACT_ROOT / "scripts"
PRICING_SNAPSHOT_PATH = ROUTE_DATA_ROOT / (
    "openrouter_deepseek_v4_pro_streamlake_pricing_snapshot_20260715.json"
)
CAPABILITY_SNAPSHOT_PATH = ROUTE_DATA_ROOT / (
    "openrouter_deepseek_v4_pro_streamlake_capability_snapshot_20260715.json"
)
GPT56_SOL_PRICING_SNAPSHOT_PATH = ROUTE_DATA_ROOT / (
    "openrouter_gpt_5_6_sol_azure_pricing_snapshot_20260715.json"
)
GPT56_SOL_CAPABILITY_SNAPSHOT_PATH = ROUTE_DATA_ROOT / (
    "openrouter_gpt_5_6_sol_azure_capability_snapshot_20260715.json"
)
CANONICAL_LAUNCHER_PATH = (
    AGENT_EVOLVE_ROOT / "examples" / "development" / "run_airfoil_v7_g3.py"
)
CANONICAL_ANALYZER_PATH = (
    AGENT_EVOLVE_ROOT
    / "examples"
    / "benchmarks"
    / "engibench_airfoil"
    / "v7_g3_analysis.py"
)

MODEL_ALIAS = "deepseek/deepseek-v4-pro"
CANONICAL_MODEL = "deepseek/deepseek-v4-pro-20260423"
PROVIDER_SLUG = "streamlake"
RESOLVED_PROVIDER = "StreamLake"
MAX_INPUT_TOKENS = 32_000
MAX_OUTPUT_TOKENS = 384_000
MAX_REASONING_TOKENS = 384_000
CONNECT_TIMEOUT_SECONDS = 90.0
FIRST_EVENT_TIMEOUT_SECONDS = 180
IDLE_TIMEOUT_SECONDS = 120
ABSOLUTE_TIMEOUT_SECONDS = 600
MAX_IN_FLIGHT = 3
MAX_PENDING = 5
MAX_ATTEMPTS = 2
BASE_BACKOFF_NS = 1_000_000_000
MAX_BACKOFF_NS = 30_000_000_000
JITTER_SEED = 2_026_071_503
JITTER_DOMAIN = "airfoil-v7-g3-causal-screen-v1"
RUN_SEED = 2_026_071_503
EVALUATOR_CONCURRENCY = 1
TEMPERATURE = 0.2
CONTAINER_IMAGE = (
    "mdolab/public@sha256:"
    "00bcded445f533f2d876c612260ac04fb991c098d29067e141c1cea4a16ae3dc"
)
OUTPUT_BUDGET_POLICY = FixedStructuredOutputBudgetPolicy(
    proposal_max_output_tokens=MAX_OUTPUT_TOKENS,
    reflection_max_output_tokens=MAX_OUTPUT_TOKENS,
)
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
PROHIBITED_RUNTIME_MODULE_PREFIXES = (
    "examples.benchmarks.engibench_airfoil.v7_experiment_support",
    "examples.benchmarks.engibench_airfoil.v7_finite_oracle",
    "examples.benchmarks.engibench_airfoil.v7_launch",
    "examples.benchmarks.engibench_airfoil.v7_oracle_portfolio_development",
    "examples.benchmarks.engibench_airfoil.v7_oracle_portfolio_live_launch",
    "examples.benchmarks.engibench_airfoil.v7_oracle_portfolio_recovery",
)
RELEASE_PREPARATION_INPUT_PATHS = (
    DEFAULT_DENYLIST_PATH,
    DEFAULT_CARD_BANK_PATH,
    DEFAULT_RELEASE_PATH,
    DEFAULT_FREEZE_RECEIPT_PATH,
)


class AirfoilG3LiveError(RuntimeError):
    """The exact G3 live route or chronology binding has drifted."""


@dataclass(frozen=True, slots=True)
class AirfoilG3ProviderProfile:
    """One fail-closed provider route for the otherwise identical G3 workflow."""

    profile_id: str
    model_alias: str
    canonical_model: str
    provider_slug: str
    resolved_provider: str
    pricing_snapshot_path: Path
    capability_snapshot_path: Path
    max_input_tokens: int
    max_output_tokens: int
    max_reasoning_tokens: int
    first_event_timeout_seconds: int
    idle_timeout_seconds: int
    absolute_timeout_seconds: int
    reasoning_config: OpenRouterReasoningConfig
    temperature: float | None
    required_supported_parameters: tuple[str, ...]
    required_reasoning_efforts: tuple[str, ...] = ()
    provider_require_parameters: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.profile_id) is not str
            or _PROFILE_ID.fullmatch(self.profile_id) is None
        ):
            raise ValueError("profile_id must use the closed lowercase grammar")
        for name in ("model_alias", "canonical_model"):
            value = getattr(self, name)
            if type(value) is not str or "/" not in value:
                raise ValueError(f"{name} must be an OpenRouter model slug")
        for name in ("provider_slug", "resolved_provider"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        for name in ("pricing_snapshot_path", "capability_snapshot_path"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a Path")
        for name in (
            "max_input_tokens",
            "max_output_tokens",
            "max_reasoning_tokens",
            "first_event_timeout_seconds",
            "idle_timeout_seconds",
            "absolute_timeout_seconds",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if (
            self.first_event_timeout_seconds > 1_800
            or self.idle_timeout_seconds > 1_800
            or self.absolute_timeout_seconds > 1_800
        ):
            raise ValueError("stream liveness seconds must not exceed 1800")
        if self.absolute_timeout_seconds < max(
            self.first_event_timeout_seconds,
            self.idle_timeout_seconds,
        ):
            raise ValueError(
                "absolute stream deadline must cover first-event and idle deadlines"
            )
        if self.max_reasoning_tokens > self.max_output_tokens:
            raise ValueError("reasoning telemetry cap cannot exceed output cap")
        if type(self.reasoning_config) is not OpenRouterReasoningConfig:
            raise TypeError("reasoning_config must be exact OpenRouterReasoningConfig")
        self.reasoning_config.__post_init__()
        if (
            self.reasoning_config.max_tokens is not None
            and self.reasoning_config.max_tokens != self.max_reasoning_tokens
        ):
            raise ValueError("numeric reasoning request and telemetry caps differ")
        if self.temperature is not None and (
            type(self.temperature) is not float or not 0.0 <= self.temperature <= 2.0
        ):
            raise ValueError("temperature must be None or a float in [0,2]")
        for name in (
            "required_supported_parameters",
            "required_reasoning_efforts",
        ):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or any(type(value) is not str or not value for value in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(f"{name} must be unique canonical strings")
        if not self.required_supported_parameters:
            raise ValueError("required_supported_parameters cannot be empty")
        if type(self.provider_require_parameters) is not bool:
            raise TypeError("provider_require_parameters must be an exact bool")

    @property
    def allowed_resolved_models(self) -> tuple[str, ...]:
        return (self.model_alias, self.canonical_model)

    @property
    def output_budget_policy(self) -> FixedStructuredOutputBudgetPolicy:
        return FixedStructuredOutputBudgetPolicy(
            proposal_max_output_tokens=self.max_output_tokens,
            reflection_max_output_tokens=self.max_output_tokens,
        )


DEEPSEEK_G3_PROVIDER_PROFILE = AirfoilG3ProviderProfile(
    profile_id="deepseek-v4-pro-streamlake-max",
    model_alias=MODEL_ALIAS,
    canonical_model=CANONICAL_MODEL,
    provider_slug=PROVIDER_SLUG,
    resolved_provider=RESOLVED_PROVIDER,
    pricing_snapshot_path=PRICING_SNAPSHOT_PATH,
    capability_snapshot_path=CAPABILITY_SNAPSHOT_PATH,
    max_input_tokens=MAX_INPUT_TOKENS,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    max_reasoning_tokens=MAX_REASONING_TOKENS,
    first_event_timeout_seconds=FIRST_EVENT_TIMEOUT_SECONDS,
    idle_timeout_seconds=IDLE_TIMEOUT_SECONDS,
    absolute_timeout_seconds=ABSOLUTE_TIMEOUT_SECONDS,
    reasoning_config=OpenRouterReasoningConfig(max_tokens=MAX_REASONING_TOKENS),
    temperature=TEMPERATURE,
    required_supported_parameters=(
        "max_tokens",
        "reasoning",
        "response_format",
        "temperature",
        "tool_choice",
        "tools",
    ),
)
GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE = AirfoilG3ProviderProfile(
    profile_id="gpt-5.6-sol-azure-xhigh",
    model_alias="openai/gpt-5.6-sol",
    canonical_model="openai/gpt-5.6-sol-20260709",
    provider_slug="azure",
    resolved_provider="Azure",
    pricing_snapshot_path=GPT56_SOL_PRICING_SNAPSHOT_PATH,
    capability_snapshot_path=GPT56_SOL_CAPABILITY_SNAPSHOT_PATH,
    max_input_tokens=32_000,
    max_output_tokens=128_000,
    max_reasoning_tokens=128_000,
    first_event_timeout_seconds=600,
    idle_timeout_seconds=300,
    absolute_timeout_seconds=600,
    reasoning_config=OpenRouterReasoningConfig(effort="xhigh"),
    temperature=None,
    required_supported_parameters=(
        "max_completion_tokens",
        "reasoning",
        "reasoning_effort",
        "response_format",
        "structured_outputs",
        "tool_choice",
        "tools",
    ),
    required_reasoning_efforts=("xhigh",),
    provider_require_parameters=True,
)
AIRFOIL_G3_PROVIDER_PROFILES = MappingProxyType(
    {
        profile.profile_id: profile
        for profile in (
            DEEPSEEK_G3_PROVIDER_PROFILE,
            GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE,
        )
    }
)


def resolve_airfoil_g3_provider_profile(profile_id: str) -> AirfoilG3ProviderProfile:
    """Resolve only a registered, immutable G3 provider profile."""

    if type(profile_id) is not str:
        raise AirfoilG3LiveError("provider profile identity must be exact text")
    try:
        profile = AIRFOIL_G3_PROVIDER_PROFILES[profile_id]
    except KeyError as exc:
        raise AirfoilG3LiveError("unknown Airfoil G3 provider profile") from exc
    profile.__post_init__()
    return profile


def verify_airfoil_g3_manifest_chronology(
    *,
    built_at_utc: str,
    expected_freeze_receipt_sha256: str,
) -> None:
    """Require the canonical manifest to be frozen strictly after preparation."""

    freeze = load_prelaunch_freeze_receipt()
    if freeze.freeze_receipt_sha256 != expected_freeze_receipt_sha256:
        raise AirfoilG3LiveError("manifest binds a foreign prelaunch freeze receipt")
    try:
        built_at = datetime.strptime(built_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        frozen_at = datetime.strptime(
            freeze.frozen_at_utc,
            "%Y-%m-%dT%H:%M:%SZ",
        ).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError) as exc:
        raise AirfoilG3LiveError(
            "freeze/manifest chronology timestamp is malformed"
        ) from exc
    if built_at <= frozen_at:
        raise AirfoilG3LiveError(
            "runtime manifest must be built strictly after the prelaunch freeze"
        )


def verify_airfoil_g3_no_leak_gate(*, stage: str) -> dict[str, object]:
    """Reject loaded outcome authorities and name the only release inputs."""

    if type(stage) is not str or not stage or stage != stage.strip():
        raise ValueError("no-leak stage must be canonical non-empty text")
    loaded = tuple(
        sorted(
            name
            for name in sys.modules
            if any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in PROHIBITED_RUNTIME_MODULE_PREFIXES
            )
        )
    )
    record = {
        "schema_version": 1,
        "stage": stage,
        "prohibited_module_prefixes": list(PROHIBITED_RUNTIME_MODULE_PREFIXES),
        "loaded_prohibited_modules": list(loaded),
        "release_preparation_input_allowlist": [
            str(path.resolve(strict=True)) for path in RELEASE_PREPARATION_INPUT_PATHS
        ],
        "passed": not loaded,
    }
    if loaded:
        raise AirfoilG3LiveError("prohibited outcome authority is loaded")
    return record


class FrozenAirfoilG3ManifestGate:
    """Reload and reproduce the complete manifest at each costly boundary."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        inputs: AirfoilG3RuntimeInputs,
        provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
    ) -> None:
        inputs.__post_init__()
        provider_profile.__post_init__()
        if inputs.freeze_receipt_sha256 is None:
            raise AirfoilG3LiveError("manifest gate requires a frozen runtime input")
        self.manifest_path = manifest_path.expanduser().resolve(strict=True)
        self.inputs = inputs
        self.provider_profile = provider_profile
        initial = load_live_runtime_manifest(self.manifest_path)
        if initial.manifest_id != "airfoil_v7_g3_live_runtime":
            raise AirfoilG3LiveError("manifest gate received another experiment")
        experiment = next(
            (
                section.to_record()["payload"]
                for section in initial.sections
                if section.section_id == "experiment"
            ),
            None,
        )
        if type(experiment) is not dict:
            raise AirfoilG3LiveError("manifest lacks the G3 experiment section")
        run_id = experiment.get("run_id")
        if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
            raise AirfoilG3LiveError("manifest run_id is malformed")
        self.run_id = run_id
        self.expected_manifest_sha256 = initial.manifest_sha256
        freeze_sha256 = inputs.freeze_receipt_sha256
        assert freeze_sha256 is not None
        verify_airfoil_g3_manifest_chronology(
            built_at_utc=initial.built_at_utc,
            expected_freeze_receipt_sha256=freeze_sha256,
        )

    def verify(self) -> "AirfoilG3LaunchVerification":
        frozen = load_live_runtime_manifest(self.manifest_path)
        if frozen.manifest_sha256 != self.expected_manifest_sha256:
            raise AirfoilG3LiveError("live runtime manifest bytes changed")
        freeze_sha256 = self.inputs.freeze_receipt_sha256
        assert freeze_sha256 is not None
        verify_airfoil_g3_manifest_chronology(
            built_at_utc=frozen.built_at_utc,
            expected_freeze_receipt_sha256=freeze_sha256,
        )
        current = build_airfoil_g3_live_runtime_manifest(
            inputs=self.inputs,
            built_at_utc=frozen.built_at_utc,
            run_id=self.run_id,
            provider_profile=self.provider_profile,
        )
        if current.to_record() != frozen.to_record():
            raise AirfoilG3LiveError("live runtime environment differs from manifest")
        freeze_sha256 = self.inputs.freeze_receipt_sha256
        assert freeze_sha256 is not None
        return AirfoilG3LaunchVerification(
            runtime_manifest_sha256=frozen.manifest_sha256,
            freeze_receipt_sha256=freeze_sha256,
        )


@runtime_checkable
class AirfoilG3LaunchGate(Protocol):
    """Reverify a frozen manifest without credentials or evaluator access."""

    def verify(self) -> "AirfoilG3LaunchVerification": ...


@dataclass(frozen=True, slots=True)
class AirfoilG3LaunchVerification:
    """The two chronology roots returned by every content-blind recheck."""

    runtime_manifest_sha256: str
    freeze_receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("runtime_manifest_sha256", "freeze_receipt_sha256"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(part not in "0123456789abcdef" for part in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _verify_launch_roots(
    gate: AirfoilG3LaunchGate,
    *,
    expected_manifest_sha256: str,
    expected_freeze_receipt_sha256: str,
) -> AirfoilG3LaunchVerification:
    observed = gate.verify()
    if type(observed) is not AirfoilG3LaunchVerification:
        raise TypeError("launch gate must return AirfoilG3LaunchVerification")
    observed.__post_init__()
    if observed.runtime_manifest_sha256 != expected_manifest_sha256:
        raise AirfoilG3LiveError("launch gate authenticated a foreign runtime manifest")
    if observed.freeze_receipt_sha256 != expected_freeze_receipt_sha256:
        raise AirfoilG3LiveError("launch gate authenticated a foreign freeze receipt")
    return observed


@dataclass(frozen=True, slots=True)
class OwnedAgenticGenerator:
    """One lazily created generator plus its owned async transport lifecycle."""

    generator: AgenticGenerator
    close: Callable[[], Any]

    def __post_init__(self) -> None:
        if not isinstance(self.generator, AgenticGenerator):
            raise TypeError("generator must implement AgenticGenerator")
        if not callable(self.close):
            raise TypeError("close must be callable")


LiveGeneratorFactory = Callable[
    [
        AirfoilG3ProviderProfile,
        str,
        ProgressAwareOpenRouterConfig,
        "AirfoilG3LiveSinks",
    ],
    OwnedAgenticGenerator,
]


@dataclass(frozen=True, slots=True)
class AirfoilG3LiveSinks:
    """Required durability boundaries supplied by the canonical live runner.

    The provider adapter remains benchmark-independent.  This small composition-
    root value keeps Airfoil's factory API stable as auditable channels are added
    and makes it difficult to accidentally construct a live transport without
    one of its four required evidence sinks.
    """

    progress: StructuredStreamProgressSink
    outcome: Callable[[Any], None]
    request_evidence: StructuredRequestEvidenceSink
    output_evidence: StructuredOutputEvidenceSink

    def __post_init__(self) -> None:
        for name in ("progress", "outcome", "request_evidence", "output_evidence"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} sink must be callable")


def build_openrouter_config(
    provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
) -> ProgressAwareOpenRouterConfig:
    """Return the exact progress-aware transport for one authenticated profile."""

    provider_profile.__post_init__()
    return ProgressAwareOpenRouterConfig(
        model_name=provider_profile.model_alias,
        provider_only=(provider_profile.provider_slug,),
        connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS,
        stream_liveness_policy=StructuredStreamLivenessPolicy(
            first_event_timeout_ns=(
                provider_profile.first_event_timeout_seconds * 1_000_000_000
            ),
            idle_timeout_ns=(provider_profile.idle_timeout_seconds * 1_000_000_000),
            absolute_timeout_ns=(
                provider_profile.absolute_timeout_seconds * 1_000_000_000
            ),
            cleanup_policy=StructuredStreamCleanupPolicy(
                cancel_drain_timeout_ns=5_000_000_000,
                transport_retire_timeout_ns=5_000_000_000,
            ),
        ),
        max_connections=MAX_IN_FLIGHT,
        max_pending=MAX_PENDING,
        max_attempts=MAX_ATTEMPTS,
        base_backoff_ns=BASE_BACKOFF_NS,
        max_backoff_ns=MAX_BACKOFF_NS,
        jitter_seed=JITTER_SEED,
        jitter_domain=JITTER_DOMAIN,
        app_title="AgentEvolve AAAI 2027 Airfoil G3 causal screen",
        reasoning_config=provider_profile.reasoning_config,
        provider_require_parameters=provider_profile.provider_require_parameters,
        retry_mode=ProgressAwareRetryMode.TRANSPORT_ONLY,
    )


def _read_snapshot(path: Path) -> tuple[dict[str, object], RuntimeFileBinding]:
    binding = capture_runtime_file(
        # Resolved here rather than inside capture_runtime_file: that helper is
        # a general "hash one exact file" utility in the shipped package and
        # has no business knowing this repository has a research corpus.
        resolve_corpus_path(path),
        logical_path=("research_artifacts/data/" + path.name),
    )
    try:
        value = json.loads(Path(binding.resolved_path).read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AirfoilG3LiveError(f"route snapshot is malformed: {path.name}") from exc
    if type(value) is not dict:
        raise AirfoilG3LiveError("route snapshot root must be an object")
    return value, binding


def bind_provider_route(
    provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
) -> dict[str, object]:
    """Authenticate one dated alias, canonical model, route, cap, and price set."""

    provider_profile.__post_init__()
    pricing, pricing_binding = _read_snapshot(provider_profile.pricing_snapshot_path)
    capability, capability_binding = _read_snapshot(
        provider_profile.capability_snapshot_path
    )
    pricing_model = pricing.get("model")
    pricing_endpoint = pricing.get("selected_endpoint")
    capability_endpoint = capability.get("selected_endpoint")
    if not all(
        type(value) is dict
        for value in (pricing_model, pricing_endpoint, capability_endpoint)
    ):
        raise AirfoilG3LiveError("route snapshots lack selected endpoint objects")
    assert type(pricing_model) is dict
    assert type(pricing_endpoint) is dict
    assert type(capability_endpoint) is dict
    prices = pricing_endpoint.get("pricing_usd_per_token")
    long_context_overrides = pricing_endpoint.get("long_context_pricing_overrides")
    shared = (
        "endpoint_tag",
        "name",
        "provider_name",
        "provider_request_slug",
        "quantization",
    )
    supported = capability_endpoint.get("supported_parameters")
    provider_registry = capability.get("provider_registry")
    model_reasoning = capability.get("model_reasoning")
    if (
        pricing.get("schema_version") != 1
        or capability.get("schema_version") != 1
        or pricing_model.get("requested_slug") != provider_profile.model_alias
        or pricing_model.get("canonical_slug") != provider_profile.canonical_model
        or capability.get("requested_model_alias") != provider_profile.model_alias
        or capability.get("canonical_model_slug") != provider_profile.canonical_model
        or pricing_model.get("context_length")
        != capability_endpoint.get("context_length")
        or pricing_model.get("max_completion_tokens")
        != provider_profile.max_output_tokens
        or capability_endpoint.get("max_completion_tokens")
        != provider_profile.max_output_tokens
        or capability_endpoint.get("provider_name")
        != provider_profile.resolved_provider
        or capability_endpoint.get("provider_request_slug")
        != provider_profile.provider_slug
        or any(
            pricing_endpoint.get(name) != capability_endpoint.get(name)
            for name in shared
        )
        or type(supported) is not list
        or not set(provider_profile.required_supported_parameters).issubset(
            set(supported)
        )
        or type(provider_registry) is not dict
        or provider_registry.get("name") != provider_profile.resolved_provider
        or provider_registry.get("slug") != provider_profile.provider_slug
        or type(prices) is not dict
        or any(
            type(prices.get(name)) is not str
            for name in ("prompt", "completion", "input_cache_read")
        )
    ):
        raise AirfoilG3LiveError("dated snapshots do not authenticate the G3 route")
    if long_context_overrides is not None:
        if (
            type(long_context_overrides) is not list
            or not long_context_overrides
            or any(type(value) is not dict for value in long_context_overrides)
        ):
            raise AirfoilG3LiveError("long-context pricing overrides are malformed")
        thresholds = tuple(
            value.get("min_prompt_tokens") for value in long_context_overrides
        )
        if any(
            type(value) is not int or value <= 0 for value in thresholds
        ) or provider_profile.max_input_tokens >= min(thresholds):
            raise AirfoilG3LiveError(
                "configured input cap reaches an unauthenticated pricing tier"
            )
    if provider_profile.required_reasoning_efforts:
        if (
            type(model_reasoning) is not dict
            or type(model_reasoning.get("supported_efforts")) is not list
            or not set(provider_profile.required_reasoning_efforts).issubset(
                set(model_reasoning["supported_efforts"])
            )
            or provider_profile.reasoning_config.effort
            not in provider_profile.required_reasoning_efforts
        ):
            raise AirfoilG3LiveError(
                "dated snapshots do not authenticate the reasoning effort"
            )
    context_length = capability_endpoint.get("context_length")
    if (
        type(context_length) is not int
        or provider_profile.max_input_tokens + provider_profile.max_output_tokens
        > context_length
    ):
        raise AirfoilG3LiveError("G3 token envelope exceeds the authenticated context")
    try:
        prompt_price = Decimal(str(prices["prompt"]))
        completion_price = Decimal(str(prices["completion"]))
        cache_price = Decimal(str(prices["input_cache_read"]))
    except (ArithmeticError, KeyError) as exc:
        raise AirfoilG3LiveError(
            "route snapshot prices are not exact decimals"
        ) from exc
    if any(
        not value.is_finite() or value < 0
        for value in (
            prompt_price,
            completion_price,
            cache_price,
        )
    ):
        raise AirfoilG3LiveError("route snapshot contains an invalid price")
    return {
        "schema_version": 1,
        "provider_profile_id": provider_profile.profile_id,
        "pricing_snapshot": pricing_binding.to_record(),
        "capability_snapshot": capability_binding.to_record(),
        "requested_model": provider_profile.model_alias,
        "canonical_model": provider_profile.canonical_model,
        "provider_request_slug": provider_profile.provider_slug,
        "resolved_provider": provider_profile.resolved_provider,
        "endpoint_tag": capability_endpoint["endpoint_tag"],
        "quantization": capability_endpoint["quantization"],
        "context_length": context_length,
        "max_completion_tokens": provider_profile.max_output_tokens,
        "prompt_usd_per_token": str(prices["prompt"]),
        "completion_usd_per_token": str(prices["completion"]),
        "input_cache_read_usd_per_token": str(prices["input_cache_read"]),
    }


def bind_streamlake_route() -> dict[str, object]:
    """Compatibility wrapper for the default DeepSeek/StreamLake profile."""

    return bind_provider_route(DEEPSEEK_G3_PROVIDER_PROFILE)


def build_telemetry_policy(
    provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
) -> AgenticTelemetryPolicy:
    """Bind model/provider telemetry before a proposal may reach evaluation."""

    provider_profile.__post_init__()
    route = bind_provider_route(provider_profile)
    maximum_cost = Decimal(provider_profile.max_input_tokens) * Decimal(
        str(route["prompt_usd_per_token"])
    ) + Decimal(provider_profile.max_output_tokens) * Decimal(
        str(route["completion_usd_per_token"])
    )
    return AgenticTelemetryPolicy(
        requested_model=provider_profile.model_alias,
        allowed_resolved_models=provider_profile.allowed_resolved_models,
        allowed_resolved_providers=(provider_profile.resolved_provider,),
        max_cost_usd=maximum_cost,
        max_input_tokens=provider_profile.max_input_tokens,
        max_output_tokens=provider_profile.max_output_tokens,
        max_reasoning_tokens=provider_profile.max_reasoning_tokens,
        max_attempt_count=MAX_ATTEMPTS,
    )


def _production_generator_factory(
    provider_profile: AirfoilG3ProviderProfile,
    api_key: str,
    config: ProgressAwareOpenRouterConfig,
    sinks: AirfoilG3LiveSinks,
) -> OwnedAgenticGenerator:
    sinks.__post_init__()
    runner = create_progress_aware_openrouter_runner(
        api_key=api_key,
        config=config,
        progress_sink=sinks.progress,
        outcome_sink=sinks.outcome,
        request_evidence_sink=sinks.request_evidence,
        output_evidence_sink=sinks.output_evidence,
        evidence_publication_policy=StructuredEvidencePublicationPolicy.REQUIRED,
    )
    generator = TelemetryGatedAgenticGenerator(
        PydanticAIAgenticGenerator(runner),
        build_telemetry_policy(provider_profile),
    )
    return OwnedAgenticGenerator(generator=generator, close=runner.aclose)


class LazyAirfoilG3AgenticGenerator:
    """Create the credentialed transport once, after a content-blind recheck."""

    def __init__(
        self,
        *,
        launch_gate: AirfoilG3LaunchGate,
        expected_manifest_sha256: str,
        expected_freeze_receipt_sha256: str,
        credential_loader: Callable[[], str],
        progress_sink: StructuredStreamProgressSink,
        outcome_sink: Callable[[Any], None],
        request_evidence_sink: StructuredRequestEvidenceSink,
        output_evidence_sink: StructuredOutputEvidenceSink,
        provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
        factory: LiveGeneratorFactory = _production_generator_factory,
    ) -> None:
        if not isinstance(launch_gate, AirfoilG3LaunchGate):
            raise TypeError("launch_gate must implement verify")
        for name, value in (
            ("credential_loader", credential_loader),
            ("progress_sink", progress_sink),
            ("outcome_sink", outcome_sink),
            ("request_evidence_sink", request_evidence_sink),
            ("output_evidence_sink", output_evidence_sink),
            ("factory", factory),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable")
        self.launch_gate = launch_gate
        AirfoilG3LaunchVerification(
            expected_manifest_sha256,
            expected_freeze_receipt_sha256,
        ).__post_init__()
        self.expected_manifest_sha256 = expected_manifest_sha256
        self.expected_freeze_receipt_sha256 = expected_freeze_receipt_sha256
        provider_profile.__post_init__()
        self.provider_profile = provider_profile
        self.credential_loader = credential_loader
        self.sinks = AirfoilG3LiveSinks(
            progress=progress_sink,
            outcome=outcome_sink,
            request_evidence=request_evidence_sink,
            output_evidence=output_evidence_sink,
        )
        self.factory = factory
        self._owned: OwnedAgenticGenerator | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def initialized(self) -> bool:
        return self._owned is not None

    async def _generator(self) -> AgenticGenerator:
        if self._closed:
            raise AirfoilG3LiveError("lazy provider generator is closed")
        if self._owned is None:
            async with self._lock:
                if self._owned is None:
                    _verify_launch_roots(
                        self.launch_gate,
                        expected_manifest_sha256=self.expected_manifest_sha256,
                        expected_freeze_receipt_sha256=(
                            self.expected_freeze_receipt_sha256
                        ),
                    )
                    api_key = self.credential_loader()
                    if type(api_key) is not str or not api_key:
                        raise AirfoilG3LiveError(
                            "credential loader returned no API key"
                        )
                    owned = self.factory(
                        self.provider_profile,
                        api_key,
                        build_openrouter_config(self.provider_profile),
                        self.sinks,
                    )
                    if type(owned) is not OwnedAgenticGenerator:
                        raise TypeError(
                            "live generator factory returned the wrong value"
                        )
                    owned.__post_init__()
                    self._owned = owned
        assert self._owned is not None
        return self._owned.generator

    async def propose(
        self,
        request: VariationGenerationRequest,
    ) -> VariationGenerationResult:
        return await (await self._generator()).propose(request)

    async def reflect(
        self,
        request: ReflectionGenerationRequest,
    ) -> ReflectionGenerationResult:
        return await (await self._generator()).reflect(request)

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._owned is not None:
                result = self._owned.close()
                if hasattr(result, "__await__"):
                    await result


def compose_airfoil_g3_optimizer(
    inputs: AirfoilG3RuntimeInputs,
    *,
    generator: AgenticGenerator,
    provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
    engine_trace_sink=None,
    optimizer_trace_sink=None,
) -> AgenticOptimizerComposition:
    """One shared provider-free/live composition path for the exact G3 design."""

    inputs.__post_init__()
    provider_profile.__post_init__()
    if not isinstance(generator, AgenticGenerator):
        raise TypeError("generator must implement AgenticGenerator")
    return compose_agentic_optimizer(
        inputs.benchmark,
        generator=generator,
        planner_factory=inputs,
        feedback_interceptor_factory=inputs.feedback_interceptor_factory,
        budget=G3_SCREEN_BUDGET,
        seed=RUN_SEED,
        id_factory=inputs.id_factory,
        memory=inputs.memory,
        evaluator_concurrency=EVALUATOR_CONCURRENCY,
        engine_trace_sink=engine_trace_sink,
        optimizer_trace_sink=optimizer_trace_sink,
        max_output_tokens=provider_profile.max_output_tokens,
        structured_output_budget_policy=provider_profile.output_budget_policy,
        temperature=provider_profile.temperature,
    )


class AirfoilG3LiveComposition:
    """Only authorized live run boundary; it rechecks immediately before G0."""

    def __init__(
        self,
        *,
        composition: AgenticOptimizerComposition,
        inputs: AirfoilG3RuntimeInputs,
        generator: LazyAirfoilG3AgenticGenerator,
        launch_gate: AirfoilG3LaunchGate,
        expected_manifest_sha256: str,
        initial_verification: AirfoilG3LaunchVerification,
    ) -> None:
        self._composition = composition
        self._inputs = inputs
        self.generator = generator
        self.launch_gate = launch_gate
        self.expected_manifest_sha256 = expected_manifest_sha256
        self.initial_verification = initial_verification
        self._run_state = "not_started"
        self._run_state_lock = threading.Lock()

    @property
    def initialized_provider(self) -> bool:
        return self.generator.initialized

    async def run(self):
        """Reverify source/manifest, then admit exact frozen seeds and no others."""

        with self._run_state_lock:
            if self._run_state != "not_started":
                raise AirfoilG3LiveError(
                    f"live composition is single-use (state={self._run_state})"
                )
            self._run_state = "running"
        try:
            freeze_sha256 = self._inputs.freeze_receipt_sha256
            assert freeze_sha256 is not None
            _verify_launch_roots(
                self.launch_gate,
                expected_manifest_sha256=self.expected_manifest_sha256,
                expected_freeze_receipt_sha256=freeze_sha256,
            )
            result = await self._composition.optimizer.run(
                self._inputs.seed_configurations
            )
        except BaseException:
            with self._run_state_lock:
                self._run_state = "failed"
            raise
        with self._run_state_lock:
            self._run_state = "completed"
        return result

    @property
    def run_state(self) -> str:
        with self._run_state_lock:
            return self._run_state

    async def aclose(self) -> None:
        await self.generator.aclose()

    @property
    def analysis_composition(self) -> AgenticOptimizerComposition:
        """Expose completed receipts to a pure analyzer, never as a run API."""

        return self._composition


def compose_airfoil_g3_live(
    inputs: AirfoilG3RuntimeInputs,
    *,
    launch_gate: AirfoilG3LaunchGate,
    expected_manifest_sha256: str,
    credential_loader: Callable[[], str],
    progress_sink: StructuredStreamProgressSink,
    outcome_sink: Callable[[Any], None],
    request_evidence_sink: StructuredRequestEvidenceSink,
    output_evidence_sink: StructuredOutputEvidenceSink,
    provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
    generator_factory: LiveGeneratorFactory = _production_generator_factory,
    engine_trace_sink=None,
    optimizer_trace_sink=None,
) -> AirfoilG3LiveComposition:
    """Verify first, compose without credentials, then defer transport creation."""

    inputs.__post_init__()
    provider_profile.__post_init__()
    if inputs.freeze_receipt_sha256 is None:
        raise AirfoilG3LiveError("live composition requires a prelaunch freeze receipt")
    initial_verification = _verify_launch_roots(
        launch_gate,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_freeze_receipt_sha256=inputs.freeze_receipt_sha256,
    )
    lazy = LazyAirfoilG3AgenticGenerator(
        launch_gate=launch_gate,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_freeze_receipt_sha256=inputs.freeze_receipt_sha256,
        credential_loader=credential_loader,
        progress_sink=progress_sink,
        outcome_sink=outcome_sink,
        request_evidence_sink=request_evidence_sink,
        output_evidence_sink=output_evidence_sink,
        provider_profile=provider_profile,
        factory=generator_factory,
    )
    composition = compose_airfoil_g3_optimizer(
        inputs,
        generator=lazy,
        provider_profile=provider_profile,
        engine_trace_sink=engine_trace_sink,
        optimizer_trace_sink=optimizer_trace_sink,
    )
    if lazy.initialized:
        raise AirfoilG3LiveError(
            "live composition read credentials before optimizer run"
        )
    return AirfoilG3LiveComposition(
        composition=composition,
        inputs=inputs,
        generator=lazy,
        launch_gate=launch_gate,
        expected_manifest_sha256=expected_manifest_sha256,
        initial_verification=initial_verification,
    )


def _labeled(paths: Sequence[Path], *, root: Path) -> dict[str, Path]:
    values: dict[str, Path] = {}
    for path in paths:
        resolved = path.expanduser().resolve(strict=True)
        label = (
            resolved.relative_to(root).as_posix()
            if resolved.is_relative_to(root)
            else "external/" + resolved.name
        )
        if label in values and values[label] != resolved:
            raise LiveRuntimeManifestError(f"ambiguous source label: {label}")
        values[label] = resolved
    return values


def capture_airfoil_g3_source_closure(
    provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
) -> RuntimeSourceClosure:
    """Capture a conservative superset of the transitive G3 live code path."""

    provider_profile.__post_init__()
    launcher_paths = (CANONICAL_LAUNCHER_PATH,)
    analyzer_paths = (CANONICAL_ANALYZER_PATH,)
    generic = tuple(
        path
        for path in (AGENT_EVOLVE_ROOT / "src" / "agent_evolve").rglob("*.py")
        if path.is_file()
    )
    benchmark = tuple(
        path
        for path in (
            AGENT_EVOLVE_ROOT / "examples" / "benchmarks" / "engibench_airfoil"
        ).glob("*.py")
        if path.is_file()
    )
    provider = tuple(
        AGENT_EVOLVE_ROOT / relative
        for relative in (
            "src/agent_evolve/application/gated_agentic_generator.py",
            "src/agent_evolve/application/llm_task_queue.py",
            "src/agent_evolve/integrations/pydantic_ai/agentic_generator.py",
            "src/agent_evolve/integrations/pydantic_ai/async_generator.py",
            "src/agent_evolve/integrations/pydantic_ai/progress_aware_openrouter.py",
            "src/agent_evolve/integrations/pydantic_ai/queued_runner.py",
        )
    )
    evaluator = (
        SCRIPT_ROOT / "airfoil_external_panel_v1.py",
        SCRIPT_ROOT / "airfoil_external_panel_v2.py",
        SCRIPT_ROOT / "airfoil_adapter_v1.py",
        SCRIPT_ROOT / "calibration_engibench_docker.py",
        SCRIPT_ROOT / "calibration_harness.py",
        SCRIPT_ROOT / "airfoil_convergence_evidence.py",
        SCRIPT_ROOT / "airfoil_convergence_overlay_v1" / "airfoil_analysis.py",
    )
    evaluator_python_root = (
        Path.home() / ".cache" / "agent_evolve_aaai2027" / "engibench"
    )
    evaluator_environment = (
        evaluator_python_root / ".venv" / "bin" / "python",
        evaluator_python_root / ".venv" / "pyvenv.cfg",
        evaluator_python_root / "pyproject.toml",
    )
    template_root = evaluator_python_root / "engibench" / "problems" / "airfoil"
    pinned_evaluator = tuple(
        template_root / relative
        for relative in (
            "v0.py",
            "utils.py",
            "templates/__init__.py",
            "templates/airfoil_analysis.py",
            "templates/airfoil_opt.py",
            "templates/cli_interface.py",
            "templates/pre_process.py",
        )
    )
    locks = (
        AGENT_EVOLVE_ROOT / "pyproject.toml",
        AGENT_EVOLVE_ROOT / "uv.lock",
    )
    routes = (
        provider_profile.pricing_snapshot_path,
        provider_profile.capability_snapshot_path,
    )
    durable_helpers = (
        AGENT_EVOLVE_ROOT / "examples" / "development" / ("durable_run_artifacts.py")
    )
    release_inputs = {
        "research_artifacts/airfoil_g3_release/" + path.name: path
        for path in RELEASE_PREPARATION_INPUT_PATHS
    }
    return capture_runtime_source_closure(
        {
            "analyzer": _labeled(analyzer_paths, root=AGENT_EVOLVE_ROOT),
            "benchmark_runtime": _labeled(benchmark, root=AGENT_EVOLVE_ROOT),
            "dependency_lock": _labeled(locks, root=AGENT_EVOLVE_ROOT),
            "evaluator_runtime": {
                "research_artifacts/scripts/"
                + path.relative_to(SCRIPT_ROOT).as_posix(): path
                for path in evaluator
            },
            "evaluator_environment": {
                "external/evaluator_environment/" + path.name: path
                for path in evaluator_environment
            },
            "pinned_evaluator_source": {
                "external/pinned_engibench/airfoil/"
                + path.relative_to(template_root).as_posix(): path
                for path in pinned_evaluator
            },
            "generic_core": _labeled(generic, root=AGENT_EVOLVE_ROOT),
            "launcher": _labeled(launcher_paths, root=AGENT_EVOLVE_ROOT),
            "artifact_journal": _labeled(
                (durable_helpers,),
                root=AGENT_EVOLVE_ROOT,
            ),
            "provider_runtime": _labeled(provider, root=AGENT_EVOLVE_ROOT),
            "release_inputs": release_inputs,
            "route_snapshot": {
                "research_artifacts/data/" + path.name: path for path in routes
            },
        }
    )


def _external_evaluator_environment_section(
    source: RuntimeSourceClosure,
) -> RuntimeManifestSection:
    settings = local_default_converged_settings()
    executable = settings.python_executable.expanduser().resolve(strict=True)
    probe = (
        "import importlib.metadata as m,json,platform,sys;"
        "rows=sorted((d.metadata.get('Name') or d.name,d.version) "
        "for d in m.distributions());"
        "print(json.dumps({'executable':sys.executable,"
        "'implementation':platform.python_implementation(),"
        "'python_version':platform.python_version(),"
        "'distributions':rows},sort_keys=True,separators=(',',':')))"
    )
    try:
        completed = subprocess.run(
            (str(settings.python_executable), "-I", "-c", probe),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        record = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise AirfoilG3LiveError("external evaluator environment probe failed") from exc
    if (
        type(record) is not dict
        or record.get("python_version") != "3.12.3"
        or record.get("implementation") != "CPython"
        or type(record.get("distributions")) is not list
    ):
        raise AirfoilG3LiveError("external evaluator environment identity drifted")
    bindings = {
        value.logical_path: value.to_record()
        for value in source.files
        if value.logical_path.startswith("external/evaluator_environment/")
    }
    if len(bindings) != 3:
        raise AirfoilG3LiveError("external evaluator environment closure is incomplete")
    return RuntimeManifestSection.seal(
        "external_evaluator_environment",
        {
            "schema_version": 1,
            "resolved_python_executable": str(executable),
            "probe": record,
            "content_bindings": bindings,
            "probe_is_provider_and_cfd_free": True,
        },
    )


def _container_image_from_bound_v1(source: RuntimeSourceClosure) -> str:
    binding = next(
        value
        for value in source.files
        if value.logical_path.endswith("airfoil_external_panel_v1.py")
    )
    tree = ast.parse(Path(binding.resolved_path).read_text(encoding="utf-8"))
    values = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "IMAGE"
            for target in node.targets
        )
    ]
    if values != [CONTAINER_IMAGE]:
        raise AirfoilG3LiveError(
            "manifest container image differs from the bound evaluator runtime"
        )
    return CONTAINER_IMAGE


def _evaluator_section(
    source: RuntimeSourceClosure,
    *,
    settings,
) -> RuntimeManifestSection:
    dataset = capture_runtime_file(
        settings.dataset_arrow,
        logical_path="external/evaluator/airfoil_v0-train.arrow",
    )
    if dataset.sha256 != EXPECTED_DATASET_SHA256:
        raise AirfoilG3LiveError("Airfoil dataset differs from its expected digest")
    evaluator_script = next(
        value
        for value in source.files
        if value.logical_path.endswith("airfoil_external_panel_v2.py")
    )
    return RuntimeManifestSection.seal(
        "evaluator",
        {
            "schema_version": 1,
            "adapter": (
                "examples.benchmarks.engibench_airfoil.v7_problem_def."
                "AirfoilV7DetailedEvaluationAdapter"
            ),
            "raw_problem": (
                "examples.benchmarks.engibench_airfoil.converged_problem_def."
                "ConvergenceQualifiedAirfoilPanelProblem"
            ),
            "python_executable": str(settings.python_executable.resolve(strict=True)),
            "evaluator_script": evaluator_script.to_record(),
            "dataset": dataset.to_record(),
            "container_image": _container_image_from_bound_v1(source),
            "cpu_set": settings.cpu_set,
            "mpi_cores": settings.mpi_cores,
            "timeout_seconds": settings.timeout_seconds,
            "evaluator_concurrency": EVALUATOR_CONCURRENCY,
            "cache_misses": 11,
            "cache_hits": 1,
            "expected_raw_receipts": 11,
            "solver_points_per_receipt": 3,
            "expected_solver_point_calls": 33,
        },
    )


def build_airfoil_g3_live_runtime_manifest(
    *,
    inputs: AirfoilG3RuntimeInputs,
    built_at_utc: str,
    run_id: str,
    provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
) -> LiveRuntimeManifest:
    """Build but do not publish one complete prospective live commitment."""

    inputs.__post_init__()
    provider_profile.__post_init__()
    if inputs.freeze_receipt_sha256 is None:
        raise AirfoilG3LiveError("runtime manifest requires the prelaunch freeze root")
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise AirfoilG3LiveError("runtime manifest run_id is malformed")
    default_settings = local_default_converged_settings()
    expected_output_root = G3_RUN_ROOT / run_id / "cfd_receipts"
    expected_work_root = G3_WORK_ROOT / run_id
    expected_settings = type(default_settings)(
        python_executable=default_settings.python_executable,
        evaluator_script=default_settings.evaluator_script,
        dataset_arrow=default_settings.dataset_arrow,
        output_root=expected_output_root,
        work_root=expected_work_root,
        cpu_set=default_settings.cpu_set,
        mpi_cores=default_settings.mpi_cores,
        timeout_seconds=default_settings.timeout_seconds,
        expected_dataset_sha256=default_settings.expected_dataset_sha256,
    )
    raw_problem = getattr(inputs.benchmark.problem, "raw_problem", None)
    if getattr(raw_problem, "settings", None) != expected_settings:
        raise AirfoilG3LiveError(
            "runtime evaluator wiring differs from the committed run-specific settings"
        )
    source = capture_airfoil_g3_source_closure(provider_profile)
    route = bind_provider_route(provider_profile)
    config = build_openrouter_config(provider_profile)
    telemetry = build_telemetry_policy(provider_profile)
    config_record = config.to_manifest_record()
    queue_record = config_record["queue"]
    assert type(queue_record) is dict
    provider_section = RuntimeManifestSection.seal(
        "provider_route",
        {
            "schema_version": 1,
            "provider_profile_id": provider_profile.profile_id,
            "route": route,
            "composition": config_record,
            "telemetry_gate": telemetry.to_trace_record(),
            "telemetry_gate_sha256": telemetry.policy_sha256,
            "logical_call_cap": G3_SCREEN_BUDGET.max_logical_llm_calls,
            "potential_raw_attempt_cap": (
                G3_SCREEN_BUDGET.max_logical_llm_calls * MAX_ATTEMPTS
            ),
            "provider_fallbacks_allowed": False,
            "structured_evidence": {
                "publication_policy": "required",
                "request_record_schema_version": (
                    STRUCTURED_REQUEST_EVIDENCE_SCHEMA_VERSION
                ),
                "output_record_schema_version": (
                    STRUCTURED_OUTPUT_EVIDENCE_SCHEMA_VERSION
                ),
                "request_published_before_queue_admission": True,
                "typed_output_published_before_downstream_validation": True,
                "prompt_content_persisted": False,
                "semantic_and_wire_prompt_commitments_persisted": True,
                "typed_output_content_persisted": True,
                "max_output_schema_utf8_bytes": (
                    MAX_STRUCTURED_OUTPUT_SCHEMA_UTF8_BYTES
                ),
                "max_typed_output_utf8_bytes": (
                    MAX_STRUCTURED_OUTPUT_EVIDENCE_UTF8_BYTES
                ),
            },
        },
    )
    queue_section = RuntimeManifestSection.seal(
        "queue_retry",
        {
            "schema_version": 1,
            "queue": queue_record,
            "largest_concurrent_wave": 3,
            "max_in_flight_covers_largest_wave": MAX_IN_FLIGHT >= 3,
            "outcome_publication": "required_before_response_release",
            "request_evidence_publication": "required_before_queue_admission",
            "output_evidence_publication": "required_before_response_release",
            "logical_schema_rerun": False,
            "retry_scope": "transport_only_exact_payload",
        },
    )
    experiment_section = RuntimeManifestSection.seal(
        "experiment",
        {
            "schema_version": 1,
            "run_id": run_id,
            "provider_profile_id": provider_profile.profile_id,
            "release_sha256": inputs.preparation.release_sha256,
            "freeze_receipt_sha256": inputs.freeze_receipt_sha256,
            "runtime_inputs_sha256": inputs.runtime_inputs_sha256,
            "phase": AIRFOIL_G3_RUNTIME_PHASE,
            "endpoint_definition_sha256": ABSOLUTE_Q_DEFINITION_SHA256,
            "reward_binding_sha256": AIRFOIL_G3_ABSOLUTE_REWARD.binding_sha256,
            "estimand_stratum_sha256": AIRFOIL_G3_ESTIMAND_STRATUM_SHA256,
            "diagnostic_permutation_receipt_sha256": (
                inputs.diagnostic_permutation.receipt_sha256
            ),
            "budget": G3_SCREEN_BUDGET.to_trace_record(),
            "seed": RUN_SEED,
            "candidate_occurrences": 12,
            "unique_physical_evaluations": 11,
            "proposal_calls": 5,
            "postseal_revision_calls": 1,
            "required_reflection_metrics": list(REQUIRED_METRIC_IDS),
            "max_output_tokens_per_call": provider_profile.max_output_tokens,
            "max_reasoning_tokens_per_call": provider_profile.max_reasoning_tokens,
            "temperature": provider_profile.temperature,
        },
    )
    boundary_section = RuntimeManifestSection.seal(
        "execution_boundary",
        {
            "schema_version": 1,
            "shared_composition_function": (
                "examples.benchmarks.engibench_airfoil.v7_g3_live."
                "compose_airfoil_g3_optimizer"
            ),
            "provider_free_and_live_share_optimizer_composition": True,
            "manifest_verified_before_benchmark_execution": True,
            "provider_transport_created_lazily": True,
            "credential_loaded_after_seed_evaluations_on_first_model_call": True,
            "manifest_reverified_immediately_before_credential_load": True,
            "durable_request_evidence_before_provider_dispatch": True,
            "durable_typed_output_evidence_before_engine_validation": True,
            "prohibited_runtime_module_prefixes": list(
                PROHIBITED_RUNTIME_MODULE_PREFIXES
            ),
            "release_preparation_input_allowlist": [
                "research_artifacts/airfoil_g3_release/" + path.name
                for path in RELEASE_PREPARATION_INPUT_PATHS
            ],
            "no_leak_gate_required_before_g0_and_after_g3": True,
            "evaluator_concurrency": EVALUATOR_CONCURRENCY,
        },
    )
    resource_lease_section = RuntimeManifestSection.seal(
        "resource_lease",
        {
            "schema_version": 1,
            "resource_key": AIRFOIL_V7_RESOURCE_KEY,
            "lease_path": str(DEFAULT_RESOURCE_LEASE_PATH),
            "scope": "host_global_fixed_container_and_cpu_allocation",
            "fixed_container_name": "machaero",
            "container_image": CONTAINER_IMAGE,
            "cpu_set": local_default_converged_settings().cpu_set,
            "mpi_cores": local_default_converged_settings().mpi_cores,
            "conflict_probe_id": AIRFOIL_V7_CONFLICT_PROBE_ID,
            "conflict_probe_version": AIRFOIL_V7_CONFLICT_PROBE_VERSION,
            "acquisition": "nonblocking_before_benchmark_construction",
            "release": "after_transport_and_durable_journal_closure",
        },
    )
    locks = tuple(
        value
        for value in source.files
        if value.logical_path in {"pyproject.toml", "uv.lock"}
    )
    environment = capture_runtime_environment_section(
        distribution_names=("httpx", "openai", "pydantic", "pydantic-ai", "pytest"),
        dependency_locks=locks,
    )
    git = capture_git_worktree_section(
        AGENT_EVOLVE_ROOT,
        source_closure=source,
    )
    evaluator = _evaluator_section(source, settings=expected_settings)
    evaluator_payload = evaluator.to_record()["payload"]
    assert type(evaluator_payload) is dict
    evaluator = RuntimeManifestSection.seal(
        "evaluator",
        {
            **evaluator_payload,
            "output_root": str(expected_output_root),
            "work_root": str(expected_work_root),
            "fresh_run_policy": {
                "run_directory_must_not_exist": True,
                "output_root_must_not_exist_before_run": True,
                "harvest_only_fresh_evaluator_run_id": True,
                "preexisting_receipts_allowed": False,
                "reproduction_receipt_allowed": False,
            },
        },
    )
    external_environment = _external_evaluator_environment_section(source)
    sections = (
        boundary_section,
        environment,
        evaluator,
        external_environment,
        experiment_section,
        git,
        provider_section,
        queue_section,
        resource_lease_section,
    )
    return build_live_runtime_manifest(
        manifest_id="airfoil_v7_g3_live_runtime",
        manifest_version=1,
        built_at_utc=built_at_utc,
        source_closure=source,
        sections=sections,
        required_section_ids=tuple(value.section_id for value in sections),
    )


__all__ = [
    "ABSOLUTE_TIMEOUT_SECONDS",
    "AIRFOIL_G3_PROVIDER_PROFILES",
    "AirfoilG3LaunchGate",
    "AirfoilG3LiveComposition",
    "AirfoilG3LiveError",
    "AirfoilG3LiveSinks",
    "AirfoilG3ProviderProfile",
    "CANONICAL_MODEL",
    "CONTAINER_IMAGE",
    "DEEPSEEK_G3_PROVIDER_PROFILE",
    "EVALUATOR_CONCURRENCY",
    "GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE",
    "LazyAirfoilG3AgenticGenerator",
    "LiveGeneratorFactory",
    "MAX_IN_FLIGHT",
    "MAX_OUTPUT_TOKENS",
    "MAX_PENDING",
    "MAX_REASONING_TOKENS",
    "MODEL_ALIAS",
    "OUTPUT_BUDGET_POLICY",
    "OwnedAgenticGenerator",
    "PROHIBITED_RUNTIME_MODULE_PREFIXES",
    "RESOLVED_PROVIDER",
    "bind_provider_route",
    "bind_streamlake_route",
    "build_airfoil_g3_live_runtime_manifest",
    "build_openrouter_config",
    "build_telemetry_policy",
    "capture_airfoil_g3_source_closure",
    "compose_airfoil_g3_live",
    "compose_airfoil_g3_optimizer",
    "resolve_airfoil_g3_provider_profile",
    "verify_airfoil_g3_no_leak_gate",
    "verify_airfoil_g3_manifest_chronology",
]
