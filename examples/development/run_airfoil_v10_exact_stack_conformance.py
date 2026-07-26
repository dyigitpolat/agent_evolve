#!/usr/bin/env python3
"""One-call, non-scientific conformance gate for the Airfoil-v10 LLM stack.

The gate deliberately does not optimize or evaluate an Airfoil.  It takes one
of v10's frozen, provider-free K=8 finite-action contracts and sends it through
the same high-level ``PydanticAIAgenticGenerator`` and production
queued/streaming OpenRouter composition used by a real run.  Its sole purpose
is to authenticate the exact request that reaches the HTTP boundary before a
more expensive evolutionary run is launched.

``readiness`` reads no credential, constructs no provider client, performs no
provider request, and invokes no evaluator.  ``live`` is one logical DeepSeek
call with v10's exact transport-only retry policy (normally one and at most two
physical attempts).  Every artifact is permanently ineligible as scientific
or optimization evidence.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Protocol


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.domain.ids import LLMCallId, ProviderAttemptId  # noqa: E402
from agent_evolve.domain.llm_task_queue import (  # noqa: E402
    CanonicalProviderErrorCode,
    SanitizedAttemptFailure,
    SanitizedValidationIssue,
    StreamTimeoutPhase,
    StructuredOutputFailureMode,
    ValidationIssueCategory,
)
from agent_evolve.application.live_runtime_manifest import (  # noqa: E402
    runtime_source_closure_from_record,
    verify_runtime_source_closure,
)
from agent_evolve.integrations.pydantic_ai import agentic_generator  # noqa: E402
from agent_evolve.integrations.pydantic_ai import async_generator  # noqa: E402
from agent_evolve.integrations.pydantic_ai import outbound_request_manifest  # noqa: E402
from agent_evolve.integrations.pydantic_ai import progress_aware_openrouter  # noqa: E402
from agent_evolve.integrations.pydantic_ai import queued_runner  # noqa: E402
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    AttemptedStructuredGenerationResponse,
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (  # noqa: E402
    validate_openrouter_outbound_request_manifest_record,
)
from agent_evolve.integrations.pydantic_ai import provider_attempt_join  # noqa: E402
from agent_evolve.integrations.pydantic_ai.provider_attempt_join import (  # noqa: E402
    build_provider_attempt_terminal_join_receipt,
    validate_provider_attempt_terminal_join_receipt,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (  # noqa: E402
    ProgressAwareOpenRouterConfig,
    create_progress_aware_openrouter_runner,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    StructuredEvidencePublicationPolicy,
    structured_generation_outcome_record,
    structured_generation_request_evidence_record,
    validate_structured_generation_output_evidence_record,
    validate_structured_generation_request_evidence_record,
)
from agent_evolve.ports.agentic_generator import (  # noqa: E402
    FiniteVariationSelectionDraft,
    VariationGenerationRequest,
    VariationGenerationResult,
)
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredGenerationRequest,
    StructuredGenerationResponse,
    StructuredStreamProgress,
    StructuredStreamProgressKind,
)
from examples.benchmarks.engibench_airfoil import v10_multi_option_inputs  # noqa: E402
from examples.benchmarks.engibench_airfoil import v10_multi_option_live  # noqa: E402
from examples.benchmarks.engibench_airfoil import v7_g3_live  # noqa: E402
from examples.benchmarks.engibench_airfoil.v7_g3_live import (  # noqa: E402
    DEEPSEEK_G3_PROVIDER_PROFILE,
    AirfoilG3ProviderProfile,
    bind_provider_route,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (  # noqa: E402
    AirfoilV7Problem,
)
from examples.benchmarks.engibench_airfoil.v10_multi_option_inputs import (  # noqa: E402
    AirfoilV10MultiOptionInputs,
    load_frozen_airfoil_v10_multi_option_inputs,
)
from examples.benchmarks.engibench_airfoil.v10_multi_option_live import (  # noqa: E402
    build_airfoil_v10_openrouter_config,
)
from examples.benchmarks.engibench_airfoil.v10_multi_option_runner import (  # noqa: E402
    airfoil_v10_expected_outbound_transport_settings,
    airfoil_v10_provider_config_record,
)
from examples.benchmarks.engibench_airfoil.v10_qualification import (  # noqa: E402
    airfoil_v10_provider_configuration_sha256,
    verify_airfoil_v10_qualification_directory,
)
from examples.benchmarks.engibench_airfoil.v10_runtime_manifest import (  # noqa: E402
    capture_airfoil_v10_runtime_source_closure,
)
from examples.development import durable_run_artifacts  # noqa: E402
from examples.development.durable_run_artifacts import (  # noqa: E402
    BatchedDurableJsonlJournal,
    DurableJsonlJournal,
    decode_json_bytes,
    file_identity,
    finalize_run_directory,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
DEFAULT_RUN_ROOT = ARTIFACT_ROOT / "experiment_logs" / "openrouter_conformance"
LIVE_AUTHORIZATION = "AIRFOIL_V10_EXACT_STACK_CONFORMANCE_LIVE_V1"

KIND = "airfoil_v10_exact_stack_conformance"
SCHEMA_VERSION = 1
LOGICAL_CALL_ID = "call_airfoil_v10_exact_stack_conformance"
OPERATION = "typed_mutation"
TARGET_PROMPT_UTF8_BYTES = 14_325
EXPECTED_LOGICAL_SCHEMA_SHA256 = (
    "3f5a2448b1728a16efcd03648ea42ed3b3b30f68fbd38c8462432697705fdde8"
)
EXPECTED_LOGICAL_SCHEMA_UTF8_BYTES = 750
EXPECTED_OUTPUT_TOOL_NAME = "select_finite_variation_option"
EXPECTED_CARD_ROLE = "learned_v2"
EXPECTED_SEED_ROLE = "diagnostic_parent"
MAX_LOGICAL_CALLS = 1
MAX_PHYSICAL_ATTEMPTS = 2
FRAMEWORK_PACKAGES = ("httpx", "openai", "pydantic", "pydantic-ai")
_MANIFEST_DOMAIN = b"agent-evolve:airfoil-v10-exact-stack-conformance:v1\x00"
_SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExactStackConformanceError(RuntimeError):
    """A content-safe readiness or one-call conformance invariant failed."""


class FailIfCalledAirfoilEvaluator:
    """Airfoil raw boundary whose invocation makes the conformance gate fail."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate_raw(self, configuration: Mapping[str, object]) -> object:
        del configuration
        self.calls += 1
        raise ExactStackConformanceError(
            "the non-scientific conformance gate invoked an Airfoil evaluator"
        )


@dataclass(frozen=True, slots=True)
class ConformanceInputs:
    inputs: AirfoilV10MultiOptionInputs
    evaluator_guard: FailIfCalledAirfoilEvaluator

    def __post_init__(self) -> None:
        if type(self.inputs) is not AirfoilV10MultiOptionInputs:
            raise TypeError("inputs must be exact AirfoilV10MultiOptionInputs")
        self.inputs.__post_init__()
        if type(self.evaluator_guard) is not FailIfCalledAirfoilEvaluator:
            raise TypeError("evaluator_guard must be exact")
        if self.evaluator_guard.calls != 0:
            raise ExactStackConformanceError(
                "an evaluator was invoked while constructing conformance inputs"
            )


class LiveRunner(Protocol):
    async def __call__(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> AttemptedStructuredGenerationResponse[Any]: ...

    async def aclose(self) -> None: ...


RunnerFactory = Callable[..., LiveRunner]


@dataclass(frozen=True, slots=True)
class ConformanceDependencies:
    inputs_factory: Callable[[], ConformanceInputs]
    credential_loader: Callable[[], str]
    runner_factory: RunnerFactory

    def __post_init__(self) -> None:
        for name in ("inputs_factory", "credential_loader", "runner_factory"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")


def build_conformance_inputs() -> ConformanceInputs:
    """Load frozen v10 authorities around a fail-if-called evaluator boundary."""

    guard = FailIfCalledAirfoilEvaluator()
    problem = AirfoilV7Problem(raw_problem=guard)
    inputs = load_frozen_airfoil_v10_multi_option_inputs(problem=problem)
    result = ConformanceInputs(inputs=inputs, evaluator_guard=guard)
    if guard.calls != 0:
        raise ExactStackConformanceError(
            "loading frozen v10 inputs unexpectedly evaluated a candidate"
        )
    return result


def _read_openrouter_api_key() -> str:
    """Read only OPENROUTER_API_KEY without retaining its value."""

    value = os.environ.get("OPENROUTER_API_KEY")
    dotenv = WORKSPACE_ROOT / ".env"
    if not value and dotenv.is_file():
        for raw in dotenv.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, candidate = line.split("=", 1)
            if name.strip() == "OPENROUTER_API_KEY":
                value = candidate.strip().strip('"').strip("'")
                break
    if type(value) is not str or not value:
        raise ExactStackConformanceError("OPENROUTER_API_KEY is unavailable")
    return value


def _production_runner_factory(**kwargs: object) -> LiveRunner:
    kwargs["evidence_publication_policy"] = (
        StructuredEvidencePublicationPolicy.REQUIRED
    )
    return create_progress_aware_openrouter_runner(**kwargs)  # type: ignore[arg-type]


_CANONICAL_PRODUCTION_DEPENDENCY_IDENTITIES = (
    build_conformance_inputs,
    _read_openrouter_api_key,
    _production_runner_factory,
)


def _is_canonical_production_dependencies(
    dependencies: ConformanceDependencies,
) -> bool:
    """Return whether every production dependency has its captured identity."""

    return (
        type(dependencies) is ConformanceDependencies
        and dependencies.inputs_factory
        is _CANONICAL_PRODUCTION_DEPENDENCY_IDENTITIES[0]
        and dependencies.credential_loader
        is _CANONICAL_PRODUCTION_DEPENDENCY_IDENTITIES[1]
        and dependencies.runner_factory
        is _CANONICAL_PRODUCTION_DEPENDENCY_IDENTITIES[2]
    )


def _sealed_production_dependencies() -> ConformanceDependencies:
    """Construct and authenticate a fresh, non-injectable production boundary."""

    dependencies = ConformanceDependencies(
        inputs_factory=_CANONICAL_PRODUCTION_DEPENDENCY_IDENTITIES[0],
        credential_loader=_CANONICAL_PRODUCTION_DEPENDENCY_IDENTITIES[1],
        runner_factory=_CANONICAL_PRODUCTION_DEPENDENCY_IDENTITIES[2],
    )
    if not _is_canonical_production_dependencies(dependencies):
        raise ExactStackConformanceError(
            "production dependency identity authentication failed"
        )
    return dependencies


# Compatibility-only snapshot. Public production entry points never resolve this
# mutable module binding; private test entry points reject canonical identities.
DEFAULT_DEPENDENCIES = _sealed_production_dependencies()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _commitment(value: object) -> str:
    return hashlib.sha256(_MANIFEST_DOMAIN + _canonical_bytes(value)).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_run_id(run_id: str) -> str:
    if type(run_id) is not str or _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ExactStackConformanceError("run_id violates the closed grammar")
    return run_id


def build_non_scientific_prompt(inputs: AirfoilV10MultiOptionInputs) -> str:
    """Build a deterministic 14,325-byte transport prompt, never an objective."""

    if type(inputs) is not AirfoilV10MultiOptionInputs:
        raise TypeError("inputs must be exact AirfoilV10MultiOptionInputs")
    inputs.__post_init__()
    binding = inputs.authority_for(
        seed_role=EXPECTED_SEED_ROLE,
        card_role=EXPECTED_CARD_ROLE,
    )
    option_ids = tuple(
        row.option.option_id for row in binding.authority.support.options
    )
    context = {
        "authority_sha256": binding.authority.authority_sha256,
        "support_sha256": binding.authority.support.support_sha256,
        "option_ids": list(option_ids),
        "requested_option_id": option_ids[0],
    }
    prefix = (
        "NON-SCIENTIFIC TRANSPORT CONFORMANCE ONLY.\n"
        "This request must never be treated as an optimization observation, "
        "candidate evaluation, benchmark result, or paper evidence.\n"
        f"Call {EXPECTED_OUTPUT_TOOL_NAME} exactly once. Select requested_option_id "
        "from the authenticated context below, use one short transport-only "
        "rationale, and return an empty claimed_insight_ids list.\n"
        "AUTHENTICATED_FINITE_CONTRACT\n"
        + _canonical_bytes(context).decode("ascii")
        + "\nINERT_LENGTH_PADDING_BEGIN\n"
    )
    suffix = "\nINERT_LENGTH_PADDING_END\nEND_NON_SCIENTIFIC_CONFORMANCE\n"
    occupied = len(prefix.encode("utf-8")) + len(suffix.encode("utf-8"))
    if occupied >= TARGET_PROMPT_UTF8_BYTES:
        raise ExactStackConformanceError("conformance prompt prefix exceeds target")
    prompt = prefix + ("." * (TARGET_PROMPT_UTF8_BYTES - occupied)) + suffix
    if len(prompt.encode("utf-8")) != TARGET_PROMPT_UTF8_BYTES:
        raise AssertionError("conformance prompt byte length is inconsistent")
    return prompt


def build_high_level_request(
    inputs: AirfoilV10MultiOptionInputs,
) -> VariationGenerationRequest:
    """Build one genuine v10 K=8 high-level finite-choice request."""

    binding = inputs.authority_for(
        seed_role=EXPECTED_SEED_ROLE,
        card_role=EXPECTED_CARD_ROLE,
    )
    profile = DEEPSEEK_G3_PROVIDER_PROFILE
    request = VariationGenerationRequest(
        call_id=LLMCallId(LOGICAL_CALL_ID),
        operation=OPERATION,
        prompt=build_non_scientific_prompt(inputs),
        candidate_model=inputs.benchmark.problem.candidate_model,
        max_output_tokens=profile.max_output_tokens,
        temperature=profile.temperature,
        finite_variation_contract=binding.authority.support.support_contract,
    )
    request.__post_init__()
    return request


class _CaptureLowLevelRunner:
    """Provider-free adapter seam used to freeze the actual low-level request."""

    def __init__(self, selected_option_id: str) -> None:
        self.selected_option_id = selected_option_id
        self.requests: list[StructuredGenerationRequest[Any]] = []

    async def __call__(
        self,
        request: StructuredGenerationRequest[Any],
    ) -> StructuredGenerationResponse[Any]:
        self.requests.append(request)
        value = request.output_type(
            option_id=self.selected_option_id,
            design_rationale="Provider-free request-shape capture only.",
            claimed_insight_ids=[],
        )
        return StructuredGenerationResponse(
            value=value,
            requested_model="offline/request-capture",
            resolved_model="offline/request-capture",
            resolved_provider="provider-free",
            provider_response_id="offline-request-capture",
            finish_reason="tool_calls",
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            cost_usd=Decimal("0"),
            latency_ns=1,
        )


async def capture_low_level_request(
    request: VariationGenerationRequest,
) -> StructuredGenerationRequest[Any]:
    """Run the real high-level adapter without a provider to expose its request."""

    contract = request.finite_variation_contract
    if contract is None:
        raise ExactStackConformanceError("conformance request lost its K=8 contract")
    capture = _CaptureLowLevelRunner(contract.options[0].option_id)
    result = await PydanticAIAgenticGenerator(capture).propose(request)
    if (
        type(result) is not VariationGenerationResult
        or type(result.draft) is not FiniteVariationSelectionDraft
        or len(capture.requests) != 1
    ):
        raise ExactStackConformanceError(
            "provider-free high-level request capture was not exact"
        )
    return capture.requests[0]


def _profile_record(profile: AirfoilG3ProviderProfile) -> dict[str, object]:
    profile.__post_init__()
    return {
        "profile_id": profile.profile_id,
        "model_alias": profile.model_alias,
        "canonical_model": profile.canonical_model,
        "allowed_resolved_models": list(profile.allowed_resolved_models),
        "provider_slug": profile.provider_slug,
        "resolved_provider": profile.resolved_provider,
        "max_input_tokens": profile.max_input_tokens,
        "max_output_tokens": profile.max_output_tokens,
        "max_reasoning_tokens": profile.max_reasoning_tokens,
        "temperature_hex": (
            None if profile.temperature is None else profile.temperature.hex()
        ),
        "reasoning": profile.reasoning_config.to_model_setting(),
    }


def _source_paths() -> tuple[Path, ...]:
    return (
        Path(__file__),
        AGENT_EVOLVE_ROOT / "tests" / "test_airfoil_v10_exact_stack_conformance.py",
        Path(agentic_generator.__file__),
        Path(async_generator.__file__),
        Path(outbound_request_manifest.__file__),
        Path(provider_attempt_join.__file__),
        Path(progress_aware_openrouter.__file__),
        Path(queued_runner.__file__),
        Path(v10_multi_option_inputs.__file__),
        Path(v10_multi_option_live.__file__),
        Path(v7_g3_live.__file__),
        Path(durable_run_artifacts.__file__),
    )


def _source_identity() -> dict[str, object]:
    return source_identity(_source_paths(), relative_to=WORKSPACE_ROOT)


def _framework_versions() -> dict[str, str]:
    values = {name: version(name) for name in FRAMEWORK_PACKAGES}
    if any(type(value) is not str or not value for value in values.values()):
        raise ExactStackConformanceError("framework version identity is invalid")
    return values


def _build_v10_runtime_provenance(
    *,
    qualification_dir: Path | None,
    production_stack_authenticated: bool,
    framework_versions: Mapping[str, str],
) -> dict[str, object]:
    """Bind the full v10 source closure and, in production, its qualification."""

    if type(production_stack_authenticated) is not bool:
        raise TypeError("production_stack_authenticated must be exact bool")
    if qualification_dir is not None and not isinstance(qualification_dir, Path):
        raise TypeError("qualification_dir must be a Path when supplied")
    if production_stack_authenticated != (qualification_dir is not None):
        raise ExactStackConformanceError(
            "production conformance requires exactly one qualification directory"
        )
    profile = DEEPSEEK_G3_PROVIDER_PROFILE
    provider_configuration = airfoil_v10_provider_config_record(profile)
    provider_configuration_sha256 = airfoil_v10_provider_configuration_sha256(
        provider_configuration
    )
    source = capture_airfoil_v10_runtime_source_closure(profile)
    verify_runtime_source_closure(source)
    qualification_record: dict[str, object] | None = None
    if qualification_dir is not None:
        qualification = verify_airfoil_v10_qualification_directory(
            qualification_dir,
            provider_profile=profile,
            provider_record=provider_configuration,
            source_closure_factory=capture_airfoil_v10_runtime_source_closure,
        )
        qualification_record = qualification.to_record()
        installed = qualification_record.get("installed_distributions")
        qualified_frameworks = (
            None
            if type(installed) is not dict
            else {name: installed.get(name) for name in FRAMEWORK_PACKAGES}
        )
        if (
            qualification_record.get("source_sha256") != source.source_sha256
            or qualification_record.get("provider_profile_id")
            != profile.profile_id
            or qualification_record.get("provider_configuration_sha256")
            != provider_configuration_sha256
            or qualified_frameworks != dict(framework_versions)
        ):
            raise ExactStackConformanceError(
                "qualification does not join exact source, route, and framework bytes"
            )
    return {
        "schema_version": 1,
        "production_qualification_required": production_stack_authenticated,
        "production_qualification_verified": production_stack_authenticated,
        "source_closure": source.to_record(),
        "source_sha256": source.source_sha256,
        "provider_configuration": provider_configuration,
        "provider_configuration_sha256": provider_configuration_sha256,
        "qualification": qualification_record,
        "source_join_exact": True,
        "provider_configuration_join_exact": True,
        "framework_versions_join_exact": True,
    }


def _verify_source_identity(contract: Mapping[str, object]) -> None:
    if contract.get("source_identity") != _source_identity():
        raise ExactStackConformanceError("conformance source identity changed")
    provenance = contract.get("v10_runtime_provenance")
    if type(provenance) is not dict:
        raise ExactStackConformanceError("v10 runtime provenance is absent")
    source_record = provenance.get("source_closure")
    if type(source_record) is not dict:
        raise ExactStackConformanceError("v10 runtime source closure is absent")
    frozen = runtime_source_closure_from_record(source_record)
    verify_runtime_source_closure(frozen)
    current = capture_airfoil_v10_runtime_source_closure(
        DEEPSEEK_G3_PROVIDER_PROFILE
    )
    current_provider_configuration = airfoil_v10_provider_config_record(
        DEEPSEEK_G3_PROVIDER_PROFILE
    )
    if (
        current.to_record() != frozen.to_record()
        or provenance.get("source_sha256") != frozen.source_sha256
        or provenance.get("provider_configuration")
        != current_provider_configuration
        or provenance.get("provider_configuration_sha256")
        != airfoil_v10_provider_configuration_sha256(
            current_provider_configuration
        )
        or contract.get("framework_versions") != _framework_versions()
        or provenance.get("source_join_exact") is not True
        or provenance.get("provider_configuration_join_exact") is not True
        or provenance.get("framework_versions_join_exact") is not True
    ):
        raise ExactStackConformanceError(
            "full v10 runtime source/route/framework closure changed"
        )
    qualification_record = provenance.get("qualification")
    if qualification_record is None:
        if (
            provenance.get("production_qualification_required") is not False
            or provenance.get("production_qualification_verified") is not False
        ):
            raise ExactStackConformanceError(
                "offline provenance has inconsistent qualification authority"
            )
        return
    if type(qualification_record) is not dict:
        raise ExactStackConformanceError("qualification identity is malformed")
    provider_configuration = provenance.get("provider_configuration")
    if type(provider_configuration) is not dict:
        raise ExactStackConformanceError("provider configuration is malformed")
    directory = qualification_record.get("directory")
    if type(directory) is not str or not directory:
        raise ExactStackConformanceError("qualification directory identity is absent")
    verified = verify_airfoil_v10_qualification_directory(
        Path(directory),
        provider_profile=DEEPSEEK_G3_PROVIDER_PROFILE,
        provider_record=provider_configuration,
        source_closure_factory=capture_airfoil_v10_runtime_source_closure,
    )
    if (
        verified.to_record() != qualification_record
        or provenance.get("production_qualification_required") is not True
        or provenance.get("production_qualification_verified") is not True
    ):
        raise ExactStackConformanceError("bound v10 qualification identity changed")


def _validate_exact_config(config: ProgressAwareOpenRouterConfig) -> None:
    expected = build_airfoil_v10_openrouter_config(DEEPSEEK_G3_PROVIDER_PROFILE)
    if config.to_manifest_record() != expected.to_manifest_record():
        raise ExactStackConformanceError("transport differs from Airfoil-v10")
    manifest = config.to_manifest_record()
    if (
        config.model_name != "deepseek/deepseek-v4-pro"
        or config.provider_only != ("streamlake",)
        or config.max_attempts != MAX_PHYSICAL_ATTEMPTS
        or manifest["provider_options"]
        != {"only": ["streamlake"], "allow_fallbacks": False}
        or manifest["reasoning"] != {"max_tokens": 384_000}
        or manifest["queue"]["retry_classifier"] != "transport_only"
    ):
        raise ExactStackConformanceError("DeepSeek exact-stack profile drifted")


async def build_contract(
    conformance: ConformanceInputs,
    *,
    qualification_dir: Path | None = None,
    production_stack_authenticated: bool = False,
) -> dict[str, object]:
    """Build the source/profile/request contract shared by readiness and live."""

    conformance.__post_init__()
    inputs = conformance.inputs
    high = build_high_level_request(inputs)
    low = await capture_low_level_request(high)
    request_evidence = structured_generation_request_evidence_record(low)
    if (
        request_evidence["call_id"] != LOGICAL_CALL_ID
        or request_evidence["operation"] != OPERATION
        or request_evidence["prompt_utf8_bytes"] != TARGET_PROMPT_UTF8_BYTES
        or request_evidence["output_tool_name"] != EXPECTED_OUTPUT_TOOL_NAME
        or request_evidence["output_schema_sha256"]
        != EXPECTED_LOGICAL_SCHEMA_SHA256
        or request_evidence["output_schema_utf8_bytes"]
        != EXPECTED_LOGICAL_SCHEMA_UTF8_BYTES
        or request_evidence["max_output_tokens"] != 384_000
        or request_evidence["temperature_hex"] != float(0.2).hex()
    ):
        raise ExactStackConformanceError("v10 G1 low-level request contract drifted")
    binding = inputs.authority_for(
        seed_role=EXPECTED_SEED_ROLE,
        card_role=EXPECTED_CARD_ROLE,
    )
    option_ids = [option.option_id for option in high.finite_variation_contract.options]  # type: ignore[union-attr]
    if len(option_ids) != 8 or len(set(option_ids)) != 8:
        raise ExactStackConformanceError("conformance authority is not genuine K=8")
    config = build_airfoil_v10_openrouter_config(DEEPSEEK_G3_PROVIDER_PROFILE)
    _validate_exact_config(config)
    framework_versions = _framework_versions()
    runtime_provenance = _build_v10_runtime_provenance(
        qualification_dir=qualification_dir,
        production_stack_authenticated=production_stack_authenticated,
        framework_versions=framework_versions,
    )
    if conformance.evaluator_guard.calls != 0:
        raise ExactStackConformanceError("contract construction invoked an evaluator")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "scientific_result_eligible": False,
        "optimization_result_eligible": False,
        "purpose": "one_call_non_scientific_transport_conformance",
        "logical_call_count": MAX_LOGICAL_CALLS,
        "maximum_physical_attempts": MAX_PHYSICAL_ATTEMPTS,
        "evaluator_call_count": 0,
        "inputs": {
            "inputs_sha256": inputs.inputs_sha256,
            "authority_sha256": binding.authority.authority_sha256,
            "support_sha256": binding.authority.support.support_sha256,
            "finite_contract_sha256": (
                binding.authority.support.support_contract.identity_sha256
            ),
            "seed_role": EXPECTED_SEED_ROLE,
            "card_role": EXPECTED_CARD_ROLE,
            "option_ids": option_ids,
            "requested_option_id": option_ids[0],
        },
        "request": request_evidence,
        "provider_profile": _profile_record(DEEPSEEK_G3_PROVIDER_PROFILE),
        "provider_route": bind_provider_route(DEEPSEEK_G3_PROVIDER_PROFILE),
        "transport": config.to_manifest_record(),
        "v10_runtime_provenance": runtime_provenance,
        "required_artifacts": [
            "provider_requests.jsonl",
            "provider_attempt_requests.jsonl",
            "provider_progress.jsonl",
            "provider_outcomes.jsonl",
            "provider_outputs.jsonl",
            "provider_attempt_join.json",
        ],
        "framework_versions": framework_versions,
        "source_identity": _source_identity(),
    }


def _readiness_record(
    run_id: str,
    contract: Mapping[str, object],
    *,
    production_stack_authenticated: bool,
) -> dict[str, object]:
    if type(production_stack_authenticated) is not bool:
        raise TypeError("production_stack_authenticated must be exact bool")
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "mode": "readiness",
        "run_id": run_id,
        "status": (
            "ready_conformance_only"
            if production_stack_authenticated
            else "ready_offline_test_only"
        ),
        "production_stack_authenticated": production_stack_authenticated,
        "created_at_utc": _utc(),
        "credentials_read": False,
        "provider_client_constructed": False,
        "provider_call_attempted": False,
        "evaluator_call_count": 0,
        "scientific_result_eligible": False,
        "optimization_result_eligible": False,
        "contract": dict(contract),
    }
    record["readiness_commitment_sha256"] = _commitment(record)
    return record


def _verify_readiness_record(
    record: Mapping[str, object],
    *,
    expected_contract: Mapping[str, object],
    production_stack_authenticated: bool,
) -> dict[str, object]:
    if type(record) is not dict:
        raise ExactStackConformanceError("readiness record is not an exact object")
    canonical = json.loads(_canonical_bytes(record))
    supplied = canonical.pop("readiness_commitment_sha256", None)
    if (
        type(supplied) is not str
        or _LOWER_SHA256.fullmatch(supplied) is None
        or supplied != _commitment(canonical)
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("kind") != KIND
        or record.get("mode") != "readiness"
        or record.get("status")
        != (
            "ready_conformance_only"
            if production_stack_authenticated
            else "ready_offline_test_only"
        )
        or record.get("production_stack_authenticated")
        is not production_stack_authenticated
        or record.get("credentials_read") is not False
        or record.get("provider_client_constructed") is not False
        or record.get("provider_call_attempted") is not False
        or record.get("evaluator_call_count") != 0
        or record.get("scientific_result_eligible") is not False
        or record.get("optimization_result_eligible") is not False
        or record.get("contract") != expected_contract
    ):
        raise ExactStackConformanceError("readiness contract is invalid or stale")
    return dict(record)


async def _execute_readiness(
    run_id: str,
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    qualification_dir: Path | None,
    dependencies: ConformanceDependencies,
    production_stack_authenticated: bool,
) -> dict[str, object]:
    """Finalize a zero-credential, zero-provider, zero-evaluator gate."""

    canonical = _validate_run_id(run_id)
    dependencies.__post_init__()
    if production_stack_authenticated and not (
        _is_canonical_production_dependencies(dependencies)
    ):
        raise ExactStackConformanceError(
            "production readiness dependencies are not authentic"
        )
    run_dir = run_root.expanduser().resolve(strict=False) / canonical
    if run_dir.exists():
        raise FileExistsError(run_dir)
    conformance = dependencies.inputs_factory()
    if type(conformance) is not ConformanceInputs:
        raise TypeError("inputs_factory returned a foreign value")
    contract = await build_contract(
        conformance,
        qualification_dir=qualification_dir,
        production_stack_authenticated=production_stack_authenticated,
    )
    if conformance.evaluator_guard.calls != 0:
        raise ExactStackConformanceError("readiness invoked an evaluator")
    run_dir.mkdir(parents=True, exist_ok=False)
    record = _readiness_record(
        canonical,
        contract,
        production_stack_authenticated=production_stack_authenticated,
    )
    write_json_atomic(run_dir / "readiness.json", record)
    write_json_atomic(
        run_dir / "result.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "ready_conformance_only"
                if production_stack_authenticated
                else "ready_offline_test_only"
            ),
            "production_stack_authenticated": production_stack_authenticated,
            "credentials_read": False,
            "provider_call_attempted": False,
            "evaluator_call_count": 0,
            "scientific_result_eligible": False,
            "optimization_result_eligible": False,
        },
    )
    _verify_source_identity(contract)
    finalization = finalize_run_directory(run_dir, status=str(record["status"]))
    return {
        "run_dir": str(run_dir),
        "readiness": record,
        "finalization": finalization,
    }


def _load_bound_readiness(
    readiness_dir: Path,
    *,
    expected_contract: Mapping[str, object],
    production_stack_authenticated: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    root = readiness_dir.expanduser().resolve(strict=True)
    finalization = verify_finalized_run_directory(root)
    expected_status = (
        "ready_conformance_only"
        if production_stack_authenticated
        else "ready_offline_test_only"
    )
    if finalization.get("status") != expected_status:
        raise ExactStackConformanceError("bound readiness did not pass")
    value = decode_json_bytes((root / "readiness.json").read_bytes())
    if type(value) is not dict:
        raise ExactStackConformanceError("bound readiness is unreadable")
    record = _verify_readiness_record(
        value,
        expected_contract=expected_contract,
        production_stack_authenticated=production_stack_authenticated,
    )
    return record, finalization


def _progress_record(value: StructuredStreamProgress) -> dict[str, object]:
    value.__post_init__()
    return {
        "schema_version": 1,
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


def _attempt_ids(outcome: Mapping[str, object]) -> tuple[str, ...]:
    attempts = outcome.get("attempts")
    if type(attempts) is not list or not 1 <= len(attempts) <= MAX_PHYSICAL_ATTEMPTS:
        raise ExactStackConformanceError("terminal outcome attempt count is invalid")
    values: list[str] = []
    for number, attempt in enumerate(attempts, start=1):
        if type(attempt) is not dict or attempt.get("attempt_number") != number:
            raise ExactStackConformanceError("terminal attempt sequence is invalid")
        evidence = attempt.get("request_evidence")
        attempt_id = None if type(evidence) is not dict else evidence.get(
            "provider_attempt_id"
        )
        if type(attempt_id) is not str:
            raise ExactStackConformanceError("physical attempt identity is missing")
        ProviderAttemptId(attempt_id)
        values.append(attempt_id)
    if len(set(values)) != len(values):
        raise ExactStackConformanceError("physical attempt identities repeat")
    return tuple(values)


def _validate_outbound_row(
    row: Mapping[str, object],
    *,
    contract: Mapping[str, object],
) -> dict[str, object]:
    value = validate_openrouter_outbound_request_manifest_record(row)
    request = contract["request"]
    assert type(request) is dict
    settings = value["settings"]
    message = value["message"]
    tool = value["tool"]
    request_contract = value["request_contract"]
    if (
        value["call_id"] != LOGICAL_CALL_ID
        or value["operation"] != OPERATION
        or settings["model"] != "deepseek/deepseek-v4-pro"
        or settings["provider"]
        != {"only": ["streamlake"], "allow_fallbacks": False}
        or settings["reasoning"] != {"max_tokens": 384_000}
        or settings["usage"] != {"include": True}
        or settings["stream"] is not True
        or settings["stream_options"] != {"include_usage": True}
        or settings["tool_choice"] != "required"
        or settings["max_completion_tokens"] != 384_000
        or settings["response_format"] is not None
        or settings["temperature_hex"] not in (None, float(0.2).hex())
        or message["content_utf8_bytes"] != TARGET_PROMPT_UTF8_BYTES
        or message["content_sha256"] != request["prompt_sha256"]
        or tool["count"] != 1
        or tool["name"] != EXPECTED_OUTPUT_TOOL_NAME
        or tool["requested_strict"] is not False
        or tool["wire_strict"] is not None
        or request_contract["logical_output_schema_sha256"]
        != EXPECTED_LOGICAL_SCHEMA_SHA256
        or request_contract["logical_output_schema_utf8_bytes"]
        != EXPECTED_LOGICAL_SCHEMA_UTF8_BYTES
        or value["framework_versions"] != contract["framework_versions"]
        or not all(value["forbidden_fields_absent"].values())
    ):
        raise ExactStackConformanceError("outbound v10 wire contract drifted")
    return value


def _expected_outbound_transport_settings(
    contract: Mapping[str, object],
) -> dict[str, object]:
    """Project the frozen v10 route onto the generic HTTP-boundary join."""

    transport = contract.get("transport")
    if type(transport) is not dict:
        raise ExactStackConformanceError("contract transport record is absent")
    expected = airfoil_v10_expected_outbound_transport_settings(
        DEEPSEEK_G3_PROVIDER_PROFILE
    )
    if (
        expected["model"] != transport.get("model_name")
        or expected["provider"] != transport.get("provider_options")
        or expected["reasoning"] != transport.get("reasoning")
    ):
        raise ExactStackConformanceError(
            "generic outbound route projection differs from frozen transport"
        )
    return expected


def validate_completed_call(
    result: VariationGenerationResult,
    *,
    contract: Mapping[str, object],
    request_rows: Sequence[Mapping[str, object]],
    outbound_rows: Sequence[Mapping[str, object]],
    progress_rows: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
    output_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Join every durable boundary for one successful logical call."""

    if type(result) is not VariationGenerationResult or type(
        result.draft
    ) is not FiniteVariationSelectionDraft:
        raise ExactStackConformanceError("high-level result is not finite choice")
    expected_request = contract["request"]
    if (
        len(request_rows) != 1
        or validate_structured_generation_request_evidence_record(request_rows[0])
        != expected_request
        or len(outcome_rows) != 1
        or len(output_rows) != 1
    ):
        raise ExactStackConformanceError("logical request/outcome/output evidence is incomplete")
    outcome = outcome_rows[0]
    if outcome.get("task_id") != LOGICAL_CALL_ID or outcome.get("status") != "succeeded":
        raise ExactStackConformanceError("terminal queue outcome is not successful")
    attempt_ids = _attempt_ids(outcome)
    validated_outbound = tuple(
        _validate_outbound_row(row, contract=contract) for row in outbound_rows
    )
    outbound_ids = tuple(row["provider_attempt_id"] for row in validated_outbound)
    if len(validated_outbound) != len(attempt_ids) or set(outbound_ids) != set(
        attempt_ids
    ):
        raise ExactStackConformanceError("outbound rows do not join physical attempts")
    output = validate_structured_generation_output_evidence_record(
        output_rows[0],
        request_evidence=dict(request_rows[0]),
    )
    typed = output["typed_output"]
    inputs_contract = contract["inputs"]
    assert type(inputs_contract) is dict
    option_ids = inputs_contract["option_ids"]
    requested_option_id = inputs_contract["requested_option_id"]
    if (
        type(typed) is not dict
        or typed.get("option_id") != requested_option_id
        or typed.get("option_id") not in option_ids
        or result.draft.option_id != typed.get("option_id")
        or result.draft.contract_identity_sha256
        != inputs_contract["finite_contract_sha256"]
    ):
        raise ExactStackConformanceError("typed finite choice escaped its K=8 contract")
    successful_attempt = attempt_ids[-1]
    successful_progress = [
        row
        for row in progress_rows
        if row.get("provider_attempt_id") == successful_attempt
    ]
    if (
        not successful_progress
        or successful_progress[-1].get("kind")
        != StructuredStreamProgressKind.STREAM_COMPLETED.value
    ):
        raise ExactStackConformanceError("successful stream lacks local completion")
    response = outcome.get("response")
    profile = DEEPSEEK_G3_PROVIDER_PROFILE
    if (
        type(response) is not dict
        or response.get("requested_model") != profile.model_alias
        or response.get("resolved_model") not in profile.allowed_resolved_models
        or response.get("resolved_provider") != profile.resolved_provider
        or type(response.get("input_tokens")) is not int
        or response["input_tokens"] <= 0
        or type(response.get("output_tokens")) is not int
        or response["output_tokens"] <= 0
        or type(response.get("reasoning_tokens")) is not int
        or response["reasoning_tokens"] <= 0
        or response.get("cost_usd") is None
    ):
        raise ExactStackConformanceError("successful provider telemetry is incomplete")
    join_receipt = validate_provider_attempt_terminal_join_receipt(
        build_provider_attempt_terminal_join_receipt(
            logical_requests=request_rows,
            outbound_manifests=outbound_rows,
            terminal_outcomes=outcome_rows,
            progress_rows=progress_rows,
            explicit_pre_transport_failures=(),
            expected_framework_versions=contract["framework_versions"],
            expected_transport_settings=(
                _expected_outbound_transport_settings(contract)
            ),
        )
    )
    if join_receipt["join_valid"] is not True:
        raise ExactStackConformanceError(
            "generic provider-attempt terminal join is invalid"
        )
    return {
        "logical_call_count": 1,
        "physical_attempt_count": len(attempt_ids),
        "provider_attempt_ids": list(attempt_ids),
        "selected_option_id": result.draft.option_id,
        "response": dict(response),
        "request_evidence_sha256": expected_request["request_evidence_sha256"],
        "outbound_request_manifest_sha256s": [
            row["outbound_request_manifest_sha256"] for row in validated_outbound
        ],
        "terminal_stream_completion_observed": True,
        "provider_attempt_join": join_receipt,
    }


_SANITIZED_FAILURE_FIELDS = frozenset(
    {
        "kind",
        "retryable",
        "safe_message",
        "status_code",
        "retry_after_seconds",
        "provider_error_code",
        "provider_error_envelope_sha256",
        "stream_timeout_phase",
        "output_failure_mode",
        "validation_issues",
    }
)


def _validated_last_sanitized_failure(
    attempts: object,
) -> tuple[dict[str, object] | None, dict[str, str] | None]:
    """Project only a domain-valid sanitized failure from an exact last attempt."""

    if type(attempts) is not list or not attempts:
        return None, None
    last_attempt = attempts[-1]
    if type(last_attempt) is not dict:
        return None, {"failure_type": "last_attempt_not_exact_object"}
    failure = last_attempt.get("failure")
    if failure is None:
        return None, None
    if type(failure) is not dict or frozenset(failure) != _SANITIZED_FAILURE_FIELDS:
        return None, {"failure_type": "last_failure_shape_invalid"}
    try:
        issues_raw = failure["validation_issues"]
        if type(issues_raw) is not list:
            raise TypeError("validation_issues must be an exact list")
        issues: list[SanitizedValidationIssue] = []
        for issue in issues_raw:
            if type(issue) is not dict or frozenset(issue) != {
                "category",
                "location",
            }:
                raise TypeError("validation issue shape is invalid")
            location = issue["location"]
            if type(location) is not list:
                raise TypeError("validation issue location must be an exact list")
            issues.append(
                SanitizedValidationIssue(
                    category=ValidationIssueCategory(issue["category"]),
                    location=tuple(location),
                )
            )
        output_mode = failure["output_failure_mode"]
        timeout_phase = failure["stream_timeout_phase"]
        provider_code = failure["provider_error_code"]
        SanitizedAttemptFailure(
            kind=failure["kind"],
            retryable=failure["retryable"],
            safe_message=failure["safe_message"],
            status_code=failure["status_code"],
            retry_after_seconds=failure["retry_after_seconds"],
            provider_error_code=(
                None
                if provider_code is None
                else CanonicalProviderErrorCode(provider_code)
            ),
            provider_error_envelope_sha256=(
                failure["provider_error_envelope_sha256"]
            ),
            stream_timeout_phase=(
                None
                if timeout_phase is None
                else StreamTimeoutPhase(timeout_phase)
            ),
            output_failure_mode=(
                None
                if output_mode is None
                else StructuredOutputFailureMode(output_mode)
            ),
            validation_issues=tuple(issues),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, {"failure_type": type(exc).__name__}
    return json.loads(_canonical_bytes(failure)), None


def _failure_diagnosis_projection(
    *,
    contract: Mapping[str, object],
    request_rows: Sequence[Mapping[str, object]],
    outbound_rows: Sequence[Mapping[str, object]],
    progress_rows: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
    output_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    validated_outbound: list[dict[str, object]] = []
    outbound_validation_failures: list[dict[str, object]] = []
    for index, row in enumerate(outbound_rows):
        try:
            validated_outbound.append(
                _validate_outbound_row(row, contract=contract)
            )
        except Exception as exc:
            outbound_validation_failures.append(
                {"row_index": index, "failure_type": type(exc).__name__}
            )
    outbound_ids = tuple(
        str(row["provider_attempt_id"]) for row in validated_outbound
    )
    try:
        join_receipt = validate_provider_attempt_terminal_join_receipt(
            build_provider_attempt_terminal_join_receipt(
                logical_requests=request_rows,
                outbound_manifests=outbound_rows,
                terminal_outcomes=outcome_rows,
                progress_rows=progress_rows,
                explicit_pre_transport_failures=(),
                expected_framework_versions=contract["framework_versions"],
                expected_transport_settings=(
                    _expected_outbound_transport_settings(contract)
                ),
            )
        )
        join_receipt_failure = None
    except Exception as exc:
        join_receipt = None
        join_receipt_failure = {"failure_type": type(exc).__name__}
    if len(outcome_rows) == 1 and type(outcome_rows[0]) is dict:
        outcome = outcome_rows[0]
        attempts = outcome.get("attempts")
        try:
            attempt_ids = (
                _attempt_ids(outcome)
                if type(attempts) is list and attempts
                else ()
            )
            outcome_validation_failure = None
        except (ExactStackConformanceError, TypeError, ValueError) as exc:
            attempt_ids = ()
            outcome_validation_failure = {"failure_type": type(exc).__name__}
        if attempt_ids and set(outbound_ids) == set(attempt_ids):
            transport_stage = "outbound_authenticated_before_remote_failure"
        elif not outbound_ids:
            transport_stage = "pre_transport_or_outbound_publication_failure"
        else:
            transport_stage = "partially_authenticated_transport_failure"
        if outcome_validation_failure is None:
            last_failure, last_failure_validation_failure = (
                _validated_last_sanitized_failure(attempts)
            )
        else:
            # A malformed attempt sequence is not a trustworthy source of
            # nested strings or provider metadata, even for diagnostics.
            last_failure = None
            last_failure_validation_failure = {
                "failure_type": "terminal_outcome_untrusted"
            }
        provider_http_diagnostics = (
            None
            if type(last_failure) is not dict
            else {
                "status_code": last_failure.get("status_code"),
                "provider_error_code": last_failure.get(
                    "provider_error_code"
                ),
                "provider_error_envelope_sha256": last_failure.get(
                    "provider_error_envelope_sha256"
                ),
            }
        )
        status = outcome.get("status")
    else:
        attempt_ids = ()
        transport_stage = "before_terminal_queue_publication"
        last_failure = None
        provider_http_diagnostics = None
        status = None
        last_failure_validation_failure = None
        outcome_validation_failure = (
            None
            if not outcome_rows
            else {"failure_type": "terminal_outcome_cardinality_invalid"}
        )
    return {
        "transport_stage": transport_stage,
        "terminal_queue_status": status,
        "physical_attempt_ids": list(attempt_ids),
        "request_evidence_rows": len(request_rows),
        "outbound_manifest_rows": len(outbound_rows),
        "terminal_outcome_rows": len(outcome_rows),
        "typed_output_rows": len(output_rows),
        "last_sanitized_failure": last_failure,
        "last_sanitized_failure_validation_failure": (
            last_failure_validation_failure
        ),
        "provider_http_diagnostics": provider_http_diagnostics,
        "provider_attempt_join": join_receipt,
        "provider_attempt_join_validation_failure": join_receipt_failure,
        "outbound_manifest_validation_failures": outbound_validation_failures,
        "terminal_outcome_validation_failure": outcome_validation_failure,
        "raw_provider_body_retained": False,
        "raw_exception_text_retained": False,
    }


def _failure_diagnosis(
    *,
    contract: Mapping[str, object],
    request_rows: Sequence[Mapping[str, object]],
    outbound_rows: Sequence[Mapping[str, object]],
    progress_rows: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
    output_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return content-safe failure evidence without masking the primary error."""

    try:
        return _failure_diagnosis_projection(
            contract=contract,
            request_rows=request_rows,
            outbound_rows=outbound_rows,
            progress_rows=progress_rows,
            outcome_rows=outcome_rows,
            output_rows=output_rows,
        )
    except Exception as exc:
        return {
            "transport_stage": "diagnostic_projection_failed",
            "terminal_queue_status": None,
            "physical_attempt_ids": [],
            "request_evidence_rows": len(request_rows),
            "outbound_manifest_rows": len(outbound_rows),
            "terminal_outcome_rows": len(outcome_rows),
            "typed_output_rows": len(output_rows),
            "last_sanitized_failure": None,
            "last_sanitized_failure_validation_failure": None,
            "provider_http_diagnostics": None,
            "provider_attempt_join": None,
            "provider_attempt_join_validation_failure": None,
            "outbound_manifest_validation_failures": [],
            "terminal_outcome_validation_failure": None,
            "diagnostic_projection_failure_type": type(exc).__name__,
            "raw_provider_body_retained": False,
            "raw_exception_text_retained": False,
        }


async def _execute_live(
    run_id: str,
    *,
    readiness_dir: Path,
    authorization: str,
    run_root: Path = DEFAULT_RUN_ROOT,
    qualification_dir: Path | None,
    production_stack_authenticated: bool,
    dependencies: ConformanceDependencies,
) -> dict[str, object]:
    """Execute and finalize exactly one non-scientific logical provider call."""

    canonical = _validate_run_id(run_id)
    if authorization != LIVE_AUTHORIZATION:
        raise ExactStackConformanceError("live authorization string is invalid")
    dependencies.__post_init__()
    if production_stack_authenticated and not (
        _is_canonical_production_dependencies(dependencies)
    ):
        raise ExactStackConformanceError(
            "production live dependencies are not authentic"
        )
    run_dir = run_root.expanduser().resolve(strict=False) / canonical
    if run_dir.exists():
        raise FileExistsError(run_dir)
    conformance = dependencies.inputs_factory()
    if type(conformance) is not ConformanceInputs:
        raise TypeError("inputs_factory returned a foreign value")
    contract = await build_contract(
        conformance,
        qualification_dir=qualification_dir,
        production_stack_authenticated=production_stack_authenticated,
    )
    readiness, readiness_finalization = _load_bound_readiness(
        readiness_dir,
        expected_contract=contract,
        production_stack_authenticated=production_stack_authenticated,
    )
    _verify_source_identity(contract)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(run_dir / "bound_readiness.json", readiness)
    write_json_atomic(
        run_dir / "launch.json",
        {
            "schema_version": SCHEMA_VERSION,
            "kind": KIND,
            "mode": "live",
            "production_stack_authenticated": production_stack_authenticated,
            "run_id": canonical,
            "created_at_utc": _utc(),
            "bound_readiness": {
                "directory": str(readiness_dir.expanduser().resolve(strict=True)),
                "readiness": file_identity(
                    readiness_dir.expanduser().resolve(strict=True) / "readiness.json"
                ),
                "finalization_sha256": readiness_finalization.get(
                    "finalization_sha256"
                ),
            },
            "scientific_result_eligible": False,
            "optimization_result_eligible": False,
        },
    )

    progress_journal = BatchedDurableJsonlJournal(
        run_dir / "provider_progress.jsonl",
        max_unfsynced_rows=32,
    )
    outcome_journal = DurableJsonlJournal(run_dir / "provider_outcomes.jsonl")
    request_journal = DurableJsonlJournal(run_dir / "provider_requests.jsonl")
    outbound_journal = DurableJsonlJournal(
        run_dir / "provider_attempt_requests.jsonl"
    )
    output_journal = DurableJsonlJournal(run_dir / "provider_outputs.jsonl")
    progress_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    outbound_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    credential_reads = 0
    client_constructed = False
    provider_call_attempted = False
    precredential_source_identity_verified = False
    terminal_source_identity_verified = False
    terminal_source_identity_failure_type: str | None = None
    runner: LiveRunner | None = None
    pending: BaseException | None = None
    result_record: dict[str, object]

    def progress_sink(value: StructuredStreamProgress) -> None:
        row = _progress_record(value)
        progress_rows.append(row)
        progress_journal.append(row)

    def outcome_sink(value: object) -> None:
        progress_journal.flush()
        row = structured_generation_outcome_record(value)  # type: ignore[arg-type]
        outcome_rows.append(row)
        outcome_journal.append(row)

    def request_sink(value: Mapping[str, object]) -> None:
        row = validate_structured_generation_request_evidence_record(value)
        request_rows.append(row)
        request_journal.append(row)

    def outbound_sink(value: Mapping[str, object]) -> None:
        row = validate_openrouter_outbound_request_manifest_record(value)
        outbound_rows.append(row)
        outbound_journal.append(row)

    def output_sink(value: Mapping[str, object]) -> None:
        row = validate_structured_generation_output_evidence_record(value)
        output_rows.append(row)
        output_journal.append(row)

    try:
        _verify_source_identity(contract)
        precredential_source_identity_verified = True
        credential_reads += 1
        if credential_reads != 1:
            raise ExactStackConformanceError("credential read count exceeded one")
        api_key = dependencies.credential_loader()
        if type(api_key) is not str or not api_key:
            raise ExactStackConformanceError("credential loader returned no key")
        write_json_atomic(
            run_dir / "credential_access.json",
            {
                "schema_version": 1,
                "credential_name": "OPENROUTER_API_KEY",
                "read_count": 1,
                "value_persisted": False,
            },
        )
        config = build_airfoil_v10_openrouter_config(DEEPSEEK_G3_PROVIDER_PROFILE)
        _validate_exact_config(config)
        runner = dependencies.runner_factory(
            api_key=api_key,
            config=config,
            progress_sink=progress_sink,
            outcome_sink=outcome_sink,
            request_evidence_sink=request_sink,
            outbound_request_manifest_sink=outbound_sink,
            output_evidence_sink=output_sink,
        )
        client_constructed = True
        high_request = build_high_level_request(conformance.inputs)
        provider_call_attempted = True
        high_result = await PydanticAIAgenticGenerator(runner).propose(high_request)
        validated = validate_completed_call(
            high_result,
            contract=contract,
            request_rows=request_rows,
            outbound_rows=outbound_rows,
            progress_rows=progress_rows,
            outcome_rows=outcome_rows,
            output_rows=output_rows,
        )
        if conformance.evaluator_guard.calls != 0:
            raise ExactStackConformanceError("live conformance invoked an evaluator")
        result_record = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "completed_conformance_only"
                if production_stack_authenticated
                else "completed_offline_test_only"
            ),
            "production_stack_authenticated": production_stack_authenticated,
            "credentials_read": credential_reads,
            "provider_client_constructed": client_constructed,
            "provider_call_attempted": provider_call_attempted,
            "precredential_source_identity_verified": (
                precredential_source_identity_verified
            ),
            "terminal_source_identity_verified": False,
            "evaluator_call_count": 0,
            **validated,
            "scientific_result_eligible": False,
            "optimization_result_eligible": False,
        }
    except BaseException as exc:
        pending = exc
        result_record = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_conformance_only",
            "production_stack_authenticated": production_stack_authenticated,
            "credentials_read": credential_reads,
            "provider_client_constructed": client_constructed,
            "provider_call_attempted": provider_call_attempted,
            "precredential_source_identity_verified": (
                precredential_source_identity_verified
            ),
            "terminal_source_identity_verified": False,
            "failure_type": type(exc).__name__,
            "evaluator_call_count": conformance.evaluator_guard.calls,
            "diagnosis": _failure_diagnosis(
                contract=contract,
                request_rows=request_rows,
                outbound_rows=outbound_rows,
                progress_rows=progress_rows,
                outcome_rows=outcome_rows,
                output_rows=output_rows,
            ),
            "scientific_result_eligible": False,
            "optimization_result_eligible": False,
        }
    finally:
        if runner is not None:
            try:
                await runner.aclose()
            except BaseException as close_exc:
                if pending is None:
                    pending = close_exc
                    result_record = {
                        "schema_version": SCHEMA_VERSION,
                        "status": "failed_conformance_only",
                        "production_stack_authenticated": (
                            production_stack_authenticated
                        ),
                        "credentials_read": credential_reads,
                        "provider_client_constructed": client_constructed,
                        "provider_call_attempted": provider_call_attempted,
                        "precredential_source_identity_verified": (
                            precredential_source_identity_verified
                        ),
                        "terminal_source_identity_verified": False,
                        "failure_type": type(close_exc).__name__,
                        "evaluator_call_count": conformance.evaluator_guard.calls,
                        "diagnosis": _failure_diagnosis(
                            contract=contract,
                            request_rows=request_rows,
                            outbound_rows=outbound_rows,
                            progress_rows=progress_rows,
                            outcome_rows=outcome_rows,
                            output_rows=output_rows,
                        ),
                        "scientific_result_eligible": False,
                        "optimization_result_eligible": False,
                    }
        for journal in (
            progress_journal,
            outcome_journal,
            request_journal,
            outbound_journal,
            output_journal,
        ):
            journal.close()

    # Closing the queued runner may synchronously publish its final cancelled or
    # failed attempt.  Make the post-close in-memory boundary state authoritative
    # for failed-call diagnosis instead of retaining a pre-close snapshot.
    if pending is not None:
        try:
            result_record["diagnosis"] = _failure_diagnosis(
                contract=contract,
                request_rows=request_rows,
                outbound_rows=outbound_rows,
                progress_rows=progress_rows,
                outcome_rows=outcome_rows,
                output_rows=output_rows,
            )
        except BaseException as diagnosis_exc:
            # Never replace the provider/transport failure with a diagnostic
            # projection failure, and never persist exception text.
            result_record["terminal_diagnosis_refresh_failure_type"] = type(
                diagnosis_exc
            ).__name__

    try:
        _verify_source_identity(contract)
        terminal_source_identity_verified = True
    except BaseException as source_exc:
        terminal_source_identity_failure_type = type(source_exc).__name__
        if pending is None:
            pending = source_exc
            result_record = {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_conformance_only",
                "production_stack_authenticated": production_stack_authenticated,
                "credentials_read": credential_reads,
                "provider_client_constructed": client_constructed,
                "provider_call_attempted": provider_call_attempted,
                "precredential_source_identity_verified": (
                    precredential_source_identity_verified
                ),
                "terminal_source_identity_verified": False,
                "failure_type": type(source_exc).__name__,
                "evaluator_call_count": conformance.evaluator_guard.calls,
                "diagnosis": _failure_diagnosis(
                    contract=contract,
                    request_rows=request_rows,
                    outbound_rows=outbound_rows,
                    progress_rows=progress_rows,
                    outcome_rows=outcome_rows,
                    output_rows=output_rows,
                ),
                "scientific_result_eligible": False,
                "optimization_result_eligible": False,
            }
    result_record["precredential_source_identity_verified"] = (
        precredential_source_identity_verified
    )
    result_record["terminal_source_identity_verified"] = (
        terminal_source_identity_verified
    )
    if terminal_source_identity_failure_type is not None:
        result_record["terminal_source_identity_failure_type"] = (
            terminal_source_identity_failure_type
        )
    join_value = result_record.get("provider_attempt_join")
    if join_value is None and type(result_record.get("diagnosis")) is dict:
        join_value = result_record["diagnosis"].get("provider_attempt_join")
    if type(join_value) is dict:
        write_json_atomic(run_dir / "provider_attempt_join.json", join_value)
    write_json_atomic(run_dir / "result.json", result_record)
    finalization = finalize_run_directory(run_dir, status=str(result_record["status"]))
    return {
        "run_dir": str(run_dir),
        "result": result_record,
        "finalization": finalization,
        "failed": pending is not None,
    }


async def execute_live(
    run_id: str,
    *,
    readiness_dir: Path,
    qualification_dir: Path,
    authorization: str,
    run_root: Path = DEFAULT_RUN_ROOT,
) -> dict[str, object]:
    """Public live entry point sealed to the exact production dependencies."""

    return await _execute_live(
        run_id,
        readiness_dir=readiness_dir,
        qualification_dir=qualification_dir,
        authorization=authorization,
        run_root=run_root,
        dependencies=_sealed_production_dependencies(),
        production_stack_authenticated=True,
    )


async def execute_readiness(
    run_id: str,
    *,
    qualification_dir: Path,
    run_root: Path = DEFAULT_RUN_ROOT,
) -> dict[str, object]:
    """Public readiness entry point sealed to production dependencies."""

    return await _execute_readiness(
        run_id,
        run_root=run_root,
        qualification_dir=qualification_dir,
        dependencies=_sealed_production_dependencies(),
        production_stack_authenticated=True,
    )


async def _execute_readiness_for_testing(
    run_id: str,
    *,
    run_root: Path,
    dependencies: ConformanceDependencies,
) -> dict[str, object]:
    """Private injected readiness; its receipt is permanently offline-only."""

    if _is_canonical_production_dependencies(dependencies):
        raise ValueError("test readiness requires injected offline dependencies")
    return await _execute_readiness(
        run_id,
        run_root=run_root,
        qualification_dir=None,
        dependencies=dependencies,
        production_stack_authenticated=False,
    )


async def _execute_live_for_testing(
    run_id: str,
    *,
    readiness_dir: Path,
    authorization: str,
    run_root: Path,
    dependencies: ConformanceDependencies,
) -> dict[str, object]:
    """Private injected path; its artifacts can never qualify production."""

    if _is_canonical_production_dependencies(dependencies):
        raise ValueError("test execution requires injected offline dependencies")
    return await _execute_live(
        run_id,
        readiness_dir=readiness_dir,
        qualification_dir=None,
        authorization=authorization,
        run_root=run_root,
        dependencies=dependencies,
        production_stack_authenticated=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("readiness", "live"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--readiness-dir", type=Path)
    parser.add_argument("--authorization")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "readiness":
            summary = asyncio.run(
                execute_readiness(
                    args.run_id,
                    qualification_dir=args.qualification_dir,
                    run_root=args.run_root,
                )
            )
        else:
            if args.readiness_dir is None:
                raise ExactStackConformanceError(
                    "live mode requires --readiness-dir"
                )
            summary = asyncio.run(
                execute_live(
                    args.run_id,
                    readiness_dir=args.readiness_dir,
                    qualification_dir=args.qualification_dir,
                    authorization=args.authorization,
                    run_root=args.run_root,
                )
            )
    except (ExactStackConformanceError, FileExistsError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_dir": summary["run_dir"],
                "status": (
                    summary.get("result", summary.get("readiness", {})).get(
                        "status"
                    )
                ),
                "finalization_sha256": summary["finalization"][
                    "finalization_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 1 if summary.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
