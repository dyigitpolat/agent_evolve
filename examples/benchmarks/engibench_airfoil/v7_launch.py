"""Prospectively frozen, fail-closed Airfoil-v7 live launch boundary.

This domain-local module turns the provider-free v7 experiment design into one
auditable live route.  It does not read credentials on import, manifest build,
or manifest verification.  The injected OpenRouter stack is created lazily on
the first proposal, which can occur only after the generic optimizer has
accepted both real-CFD seed evaluations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from agent_evolve.agentic import (
    AgenticBenchmark,
    AgenticCallTelemetry,
    AgenticGenerator,
    AgenticTelemetryPolicy,
    DetailedEvaluationPayload,
    ExclusiveResourceLease,
    FiniteVariationSelectionDraft,
    HeldOutASNAssignmentCommitment,
    InsightTreatmentRequirement,
    InsightDraft,
    OptimizerResult,
    OutcomeRelation,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    StrictTreatmentCompliancePolicy,
    StructuredOutputRequestKind,
    TreatmentAdmissionReceipt,
    TreatmentAssignmentRole,
    TreatmentClaimMode,
    TreatmentComplianceRejected,
    TreatmentPreflightReceipt,
    VariationGenerationRequest,
    VariationGenerationResult,
    bind_finite_variation_catalog,
    freeze_json,
    render_optimization_semantics,
    resolve_structured_output_budget,
    typed_json_sha256,
    validate_generation_feedback_receipt,
)
from agent_evolve.domain.ids import validate_id_namespace
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    ADFLOW_EVALUATOR_ID,
    ConvergenceQualifiedAirfoilPanelProblem,
    EVIDENCE_CONTRACT_ID,
    V2_EVALUATOR_ID,
    local_default_converged_settings,
)
from examples.benchmarks.engibench_airfoil.problem_def import (
    EXPECTED_DATASET_SHA256,
    AirfoilPanelEvaluation,
    candidate_sha256,
    normalize_candidate,
)
from examples.benchmarks.engibench_airfoil.v7_contract import (
    AIRFOIL_V7_ARCHIVE_RELATION,
    AIRFOIL_V7_REWARD_BINDING,
    ARCHIVE_DEFINITION_SHA256,
    PHENOTYPE_DEFINITION_SHA256,
    REWARD_DEFINITION_SHA256,
    TASK_SHA256,
    AirfoilV7PhenotypeIdentityPolicy,
)
from examples.benchmarks.engibench_airfoil.v7_experiment_support import (
    EVALUATOR_CONCURRENCY,
    MAX_OUTPUT_TOKENS,
    MEMORY_CARD_BEGIN,
    MEMORY_CARD_END,
    MODEL,
    NEUTRAL_PARENT,
    OPTIMIZER_BUDGET,
    PLANNER_POLICY_ID,
    PLANNER_POLICY_VERSION,
    REFLECTION_INSIGHT_CONTRACT,
    SHAM_OPTION_ID,
    SHAPE_MUTATION_CONTRACT,
    STRUCTURED_OUTPUT_BUDGET_POLICY,
    TRIM_MUTATION_CONTRACT,
    UNION_MUTATION_CONTRACT,
    compose_airfoil_v7_experiment,
    compose_offline_experiment,
    mask_memory_card,
    materialize_held_out_parent,
    structured_output_budget_policy_record,
    validate_frozen_no_cfd_candidate,
)
from examples.benchmarks.engibench_airfoil.v7_readiness import (
    AIRFOIL_V7_CONFLICT_PROBE_ID,
    AIRFOIL_V7_CONFLICT_PROBE_VERSION,
    AIRFOIL_V7_RESOURCE_KEY,
    AirfoilV7ReadinessSpec,
    create_airfoil_v7_resource_lease,
    observe_airfoil_v7_environment,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    AIRFOIL_V7_OPTIMIZATION_SEMANTICS,
    EVALUATOR_IDENTITY,
    OBJECTIVE_NAME,
    VIOLATION_NAME,
    AirfoilV7DetailedEvaluationAdapter,
    AirfoilV7Problem,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    AirfoilV7ShapeVariationCatalog,
    AirfoilV7TrimVariationCatalog,
    AirfoilV7UnionVariationCatalog,
)


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
DEFAULT_LIVE_LOG_ROOT = ARTIFACT_ROOT / "experiment_logs" / "airfoil_v7"
DEFAULT_SEED_QUALIFICATION_ROOT = DEFAULT_LIVE_LOG_ROOT / "seed_qualifications"
DEFAULT_PROVIDER_RUN_ROOT = DEFAULT_LIVE_LOG_ROOT / "provider_runs"
DEFAULT_RESOURCE_LEASE_PATH = Path(
    "/tmp/agent_evolve_resource_locks/engibench_airfoil_machaero.lock"
)
SCRIPTS_ROOT = ARTIFACT_ROOT / "scripts"
ROUTE_DATA_ROOT = ARTIFACT_ROOT / "data"
PRICING_SNAPSHOT_NAME = (
    "openrouter_deepseek_v4_pro_streamlake_pricing_snapshot_20260714.json"
)
CAPABILITY_SNAPSHOT_NAME = (
    "openrouter_deepseek_v4_pro_streamlake_capability_snapshot_20260714.json"
)

MANIFEST_KIND = "airfoil_v7_provider_launch"
SEED_MANIFEST_KIND = "airfoil_v7_seed_qualification"
MANIFEST_SCHEMA_VERSION = 1
SOURCE_FRAMING = b"agent-evolve:airfoil-v7-live-source-snapshot:v1\x00"
MANIFEST_FRAMING = b"agent-evolve:airfoil-v7-live-launch-manifest:v1\x00"
SEED_MANIFEST_FRAMING = (
    b"agent-evolve:airfoil-v7-seed-qualification-manifest:v1\x00"
)
SEED_RESULT_FRAMING = b"agent-evolve:airfoil-v7-seed-qualification-result:v1\x00"
RUN_SEED = 20_260_714
PROVIDER = "openrouter"
PROVIDER_ONLY = ("streamlake",)
MIN_PROVIDER_CONCURRENCY = 3
MAX_INPUT_TOKENS = 32_000
MAX_PROMPT_UTF8_BYTES = 65_536
PROMPT_SCHEMA_RESERVE_TOKENS = 2_048
PROMPT_PREFLIGHT_ENCODING = "cl100k_base_proxy_not_provider_tokenizer"
MAX_PROVIDER_ATTEMPTS = 2
PROVIDER_ATTEMPT_TIMEOUT_NS = 180_000_000_000
PROVIDER_REASONING_MAX_TOKENS = 4_096
CFD_ATTEMPTS_PER_UNIQUE_CANDIDATE = 1
CLEAN_EARLY_STOP_REASON_CODES = frozenset(
    {
        "reflected_card_batch_unavailable",
        "equal_origin_scores",
        "structurally_inapplicable_assignment",
    }
)
CONTAINER_IMAGE = (
    "mdolab/public@sha256:"
    "00bcded445f533f2d876c612260ac04fb991c098d29067e141c1cea4a16ae3dc"
)


def _read_route_snapshot(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    """Read one dated public route snapshot without consulting the network."""

    try:
        content = path.resolve(strict=True).read_bytes()
        value = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} route snapshot is unavailable or malformed") from exc
    if type(value) is not dict:
        raise TypeError(f"{label} route snapshot root must be an object")
    return value, content


def _streamlake_route_snapshot_binding() -> dict[str, object]:
    """Authenticate the selected endpoint and expose only launch-critical facts."""

    pricing_path = ROUTE_DATA_ROOT / PRICING_SNAPSHOT_NAME
    capability_path = ROUTE_DATA_ROOT / CAPABILITY_SNAPSHOT_NAME
    pricing, pricing_bytes = _read_route_snapshot(pricing_path, label="pricing")
    capability, capability_bytes = _read_route_snapshot(
        capability_path,
        label="capability",
    )
    pricing_model = pricing.get("model")
    pricing_endpoint = pricing.get("selected_endpoint")
    capability_endpoint = capability.get("selected_endpoint")
    if (
        pricing.get("schema_version") != 1
        or capability.get("schema_version") != 1
        or type(pricing_model) is not dict
        or type(pricing_endpoint) is not dict
        or type(capability_endpoint) is not dict
    ):
        raise RuntimeError("StreamLake route snapshots violate their frozen schema")
    requested_alias = capability.get("requested_model_alias")
    canonical_model = capability.get("canonical_model_slug")
    context_length = capability_endpoint.get("context_length")
    max_completion_tokens = capability_endpoint.get("max_completion_tokens")
    pricing_context_length = pricing_model.get("context_length")
    pricing_max_completion_tokens = pricing_model.get("max_completion_tokens")
    supported_parameters = capability_endpoint.get("supported_parameters")
    price_record = pricing_endpoint.get("pricing_usd_per_token")
    shared_endpoint_fields = (
        "endpoint_tag",
        "name",
        "provider_name",
        "provider_request_slug",
        "quantization",
    )
    if (
        requested_alias != MODEL
        or pricing_model.get("requested_slug") != MODEL
        or canonical_model != "deepseek/deepseek-v4-pro-20260423"
        or pricing_model.get("canonical_slug") != canonical_model
        or type(context_length) is not int
        or type(max_completion_tokens) is not int
        or pricing_context_length != context_length
        or pricing_max_completion_tokens != max_completion_tokens
        or max_completion_tokens != MAX_OUTPUT_TOKENS
        or MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS > context_length
        or type(supported_parameters) is not list
        or any(type(item) is not str for item in supported_parameters)
        or not {
            "max_tokens",
            "reasoning",
            "response_format",
            "temperature",
            "tool_choice",
            "tools",
        }.issubset(set(supported_parameters))
        or type(price_record) is not dict
        or any(
            type(price_record.get(field)) is not str
            for field in ("prompt", "completion", "input_cache_read")
        )
        or any(
            pricing_endpoint.get(field) != capability_endpoint.get(field)
            for field in shared_endpoint_fields
        )
        or capability_endpoint.get("provider_name") != "StreamLake"
        or capability_endpoint.get("provider_request_slug") != "streamlake"
    ):
        raise RuntimeError("dated snapshots do not authenticate the frozen model route")
    try:
        prompt_price = Decimal(str(price_record["prompt"]))
        completion_price = Decimal(str(price_record["completion"]))
        cache_read_price = Decimal(str(price_record["input_cache_read"]))
    except (KeyError, ArithmeticError) as exc:
        raise RuntimeError("pricing snapshot lacks exact decimal token prices") from exc
    if any(
        not value.is_finite() or value < 0
        for value in (prompt_price, completion_price, cache_read_price)
    ):
        raise RuntimeError("pricing snapshot contains an invalid token price")
    return {
        "schema_version": 1,
        "pricing_snapshot": {
            "path": str(pricing_path.resolve()),
            "sha256": hashlib.sha256(pricing_bytes).hexdigest(),
            "bytes": len(pricing_bytes),
            "retrieved_at_utc": pricing.get("retrieved_at_utc"),
            "source_url": pricing.get("source_url"),
        },
        "capability_snapshot": {
            "path": str(capability_path.resolve()),
            "sha256": hashlib.sha256(capability_bytes).hexdigest(),
            "bytes": len(capability_bytes),
            "retrieved_at_utc": capability.get("retrieved_at_utc"),
            "source_urls": capability.get("source_urls"),
        },
        "selected_route": {
            "requested_model": requested_alias,
            "canonical_model": canonical_model,
            "provider_name": capability_endpoint["provider_name"],
            "provider_request_slug": capability_endpoint[
                "provider_request_slug"
            ],
            "endpoint_tag": capability_endpoint["endpoint_tag"],
            "quantization": capability_endpoint["quantization"],
            "context_length": context_length,
            "max_completion_tokens": max_completion_tokens,
            "prompt_usd_per_token": str(price_record["prompt"]),
            "completion_usd_per_token": str(price_record["completion"]),
            "input_cache_read_usd_per_token": str(
                price_record["input_cache_read"]
            ),
        },
    }


def _maximum_billable_cost_per_attempt() -> Decimal:
    """Derive the fail-closed cost ceiling from frozen route facts and caps."""

    route = _streamlake_route_snapshot_binding()["selected_route"]
    if type(route) is not dict:
        raise RuntimeError("selected route binding changed type")
    prompt_price = Decimal(str(route["prompt_usd_per_token"]))
    completion_price = Decimal(str(route["completion_usd_per_token"]))
    return (
        Decimal(MAX_INPUT_TOKENS) * prompt_price
        + Decimal(MAX_OUTPUT_TOKENS + PROVIDER_REASONING_MAX_TOKENS)
        * completion_price
    )


def _cost_envelope_record(*, logical_calls: int) -> dict[str, object]:
    """Freeze maximum exposure; the large completion cap is not a forecast."""

    if type(logical_calls) is not int or logical_calls <= 0:
        raise ValueError("logical_calls must be positive")
    route = _streamlake_route_snapshot_binding()["selected_route"]
    assert type(route) is dict
    per_attempt = _maximum_billable_cost_per_attempt()
    accepted_run = per_attempt * Decimal(logical_calls)
    raw_attempt_cap = logical_calls * MAX_PROVIDER_ATTEMPTS
    potentially_billable_run = per_attempt * Decimal(raw_attempt_cap)
    return {
        "prompt_usd_per_token": route["prompt_usd_per_token"],
        "completion_usd_per_token": route["completion_usd_per_token"],
        "max_input_tokens_per_call": MAX_INPUT_TOKENS,
        "max_output_tokens_per_call": MAX_OUTPUT_TOKENS,
        "max_reasoning_tokens_per_call": PROVIDER_REASONING_MAX_TOKENS,
        "conservative_reasoning_accounting": (
            "reasoning cap is charged once more at the completion-token rate "
            "even if provider output usage already includes it"
        ),
        "derived_max_billable_attempt": str(per_attempt),
        "derived_max_accepted_run": str(accepted_run),
        "derived_max_potentially_billable_run": str(potentially_billable_run),
        "logical_call_cap": logical_calls,
        "raw_attempt_cap": raw_attempt_cap,
        "derivation_source": (
            "dated selected-endpoint prices times v7 telemetry caps; the "
            "snapshot's older attempt3 gate derivation is not reused"
        ),
        "ceiling_semantics": "worst_case_gate_not_expected_usage",
    }

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PYTEST_PASS_LINE = re.compile(
    r"(?m)^(?P<passed>[1-9][0-9]*) passed"
    r"(?:, [1-9][0-9]* subtests passed)? in .+$"
)
FOCUSED_PYTEST_ARGV = (
    "env",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONPATH=.",
    ".venv/bin/python",
    "-m",
    "pytest",
    "tests/test_optimization_semantics.py",
    "tests/test_agentic_public_api.py",
    "tests/test_boils_agentic_public_port.py",
    "tests/test_contrast_sharded_reflection_workflow.py",
    "tests/test_pydantic_agentic_generator.py",
    "tests/test_reflective_feedback.py",
    "tests/test_treatment_compliance.py",
    "tests/test_airfoil_v7_composition.py",
    "tests/test_airfoil_convergence_adapter.py",
    "tests/test_airfoil_v7_experiment_runner.py",
    "tests/test_airfoil_v7_g2_prequeue_gate.py",
    "tests/test_airfoil_v7_live_launch.py",
    "tests/test_airfoil_v7_finite_oracle.py",
    "tests/test_airfoil_v7_readiness.py",
    "tests/test_resource_lease.py",
    "tests/test_gated_agentic_generator.py",
    "tests/test_llm_task_queue.py",
    "tests/test_run_v6_closed_loop_memory_probe_offline.py",
    "../papers/agent_evolve_aaai_2027/research_artifacts/scripts/"
    "test_airfoil_convergence_evidence.py",
)
FOCUSED_RUFF_ARGV = (
    "uvx",
    "--offline",
    "ruff@0.15.21",
    "check",
    "src/agent_evolve/core/optimization_semantics.py",
    "src/agent_evolve/agentic.py",
    "src/agent_evolve/application/agentic_evolution.py",
    "src/agent_evolve/application/budgeted_optimizer.py",
    "src/agent_evolve/application/evaluation_recourse.py",
    "src/agent_evolve/application/insight_memory.py",
    "src/agent_evolve/application/reflection_workflow.py",
    "src/agent_evolve/application/staged_memory.py",
    "src/agent_evolve/infrastructure/resource_lease.py",
    "src/agent_evolve/integrations/pydantic_ai/agentic_generator.py",
    "src/agent_evolve/policies/feedback/held_out_asn.py",
    "src/agent_evolve/policies/memory/__init__.py",
    "src/agent_evolve/policies/memory/treatment_compliance.py",
    "src/agent_evolve/ports/agentic_generator.py",
    "src/agent_evolve/ports/resource_lease.py",
    "examples/benchmarks/engibench_airfoil/v7_experiment_support.py",
    "examples/benchmarks/engibench_airfoil/v7_launch.py",
    "examples/benchmarks/engibench_airfoil/v7_finite_oracle.py",
    "examples/benchmarks/engibench_airfoil/v7_problem_def.py",
    "examples/benchmarks/engibench_airfoil/converged_problem_def.py",
    "examples/benchmarks/engibench_airfoil/v7_readiness.py",
    "examples/benchmarks/engibench_airfoil/v7_variation_catalog.py",
    "examples/development/run_airfoil_v7_reflective_feedback.py",
    "examples/development/run_airfoil_v7_finite_oracle.py",
    "examples/development/run_v6_closed_loop_memory_probe.py",
    "tests/test_optimization_semantics.py",
    "tests/test_agentic_public_api.py",
    "tests/test_boils_agentic_public_port.py",
    "tests/test_contrast_sharded_reflection_workflow.py",
    "tests/test_pydantic_agentic_generator.py",
    "tests/test_reflective_feedback.py",
    "tests/test_treatment_compliance.py",
    "tests/test_airfoil_v7_composition.py",
    "tests/test_airfoil_convergence_adapter.py",
    "tests/test_airfoil_v7_experiment_runner.py",
    "tests/test_airfoil_v7_g2_prequeue_gate.py",
    "tests/test_airfoil_v7_live_launch.py",
    "tests/test_airfoil_v7_finite_oracle.py",
    "tests/test_airfoil_v7_readiness.py",
    "tests/test_resource_lease.py",
    "../papers/agent_evolve_aaai_2027/research_artifacts/scripts/"
    "airfoil_convergence_evidence.py",
    "../papers/agent_evolve_aaai_2027/research_artifacts/scripts/"
    "airfoil_convergence_overlay_v1/airfoil_analysis.py",
    "../papers/agent_evolve_aaai_2027/research_artifacts/scripts/"
    "airfoil_external_panel_v2.py",
    "../papers/agent_evolve_aaai_2027/research_artifacts/scripts/"
    "test_airfoil_convergence_evidence.py",
)

_BLINDED_CARD_KEYS = frozenset(
    {
        "affected_paths",
        "action_template",
        "claim",
        "effect_predictions",
        "falsification_condition",
        "insight_id",
        "mechanism",
        "recommended_option_families",
        "recommended_option_ids",
        "trigger",
    }
)
_RESERVED_CARD_TERMS = (
    "adaptive",
    "arm",
    "assignment",
    "confidence",
    "control",
    "lifecycle",
    "lineage",
    "manual",
    "origin",
    "provenance",
    "quarantine",
    "retrievable",
    "score swapped",
    "sham",
)
_G2_GATE_RECORD_TYPE = "g2_prequeue_batch_gate"


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    member = getattr(value, "value", None)
    if type(member) is str:
        return member
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        default=_json_default,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_record(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


_HELD_OUT_ADJUDICATOR_DEFINITION = {
    "policy_id": "airfoil_v7_held_out_transfer_adjudicator",
    "policy_version": 4,
    "adaptive_slot_id": "A",
    "delta_f": "0.001",
    "delta_v": "0.005",
    "source_contract": (
        "research_artifact_109_exact_action_treatment_repair_protocol"
    ),
    "required_claim": "exact_assigned_insight_only",
    "required_action_family": "selected_option_family_in_card_recommendations",
    "required_action_id": "selected_option_equals_card_singleton_option_id",
    "generic_treatment_policy": {
        "policy_id": StrictTreatmentCompliancePolicy().policy_id,
        "policy_version": StrictTreatmentCompliancePolicy().policy_version,
        "definition_sha256": StrictTreatmentCompliancePolicy().definition_sha256,
    },
    "metric_adjudication_precondition": (
        "candidate_exists_and_exact_assignment_claim_action_id_family_operator_"
        "evidence_validity_and_typed_treatment_admission_all_pass"
    ),
    "nonadministration_verdict": "not_tested_noncompliance",
    "nonadministration_metric_adjudications": "empty_never_falsified",
    "treatment_rejection_semantics": (
        "typed_expected_no_yield_before_evaluator_not_infrastructure_drift"
    ),
    "prediction_vocabulary": [
        "decrease",
        "increase",
        "unchanged",
        "unknown",
    ],
    "unknown_is_inconclusive": True,
    "automatic_memory_transition": False,
    "primary_metric": "objective:normalized_multipoint_drag",
    "domain_order": "exact_violation_then_exact_drag",
    "advance_rule": (
        "all_six_artifact_91_conjuncts_plus_claim_exact_action_and_family_gates"
    ),
}
HELD_OUT_ADJUDICATOR_SHA256 = _sha256_record(
    _HELD_OUT_ADJUDICATOR_DEFINITION,
    domain=b"agent-evolve:airfoil-v7-held-out-adjudicator:v4\x00",
)


def _optimization_semantics_binding() -> dict[str, object]:
    """Bind the exact benchmark semantics record and its compact identity."""

    semantics = AIRFOIL_V7_OPTIMIZATION_SEMANTICS
    return {
        "identity": {
            "semantics_id": semantics.semantics_id,
            "semantics_version": semantics.semantics_version,
            "definition_sha256": semantics.definition_sha256,
        },
        "record": semantics.to_record(),
    }


def _treatment_compliance_policy_binding() -> dict[str, object]:
    """Bind the generic pre-evaluation treatment policy used by the engine."""

    policy = StrictTreatmentCompliancePolicy()
    return {
        "policy_id": policy.policy_id,
        "policy_version": policy.policy_version,
        "definition_sha256": policy.definition_sha256,
        "rejection_stage": "treatment_noncompliance",
        "evaluator_entered_on_rejection": False,
    }


def _directory_fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path: Path, value: object) -> None:
    """Publish one immutable JSON document with file and directory durability."""

    payload = json.dumps(
        value,
        default=_json_default,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("ascii") + b"\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.replace(path)
    _directory_fsync(path.parent)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(path)
    temporary.replace(path)
    _directory_fsync(path.parent)


class DurableJsonlWriter:
    """Thread-safe append plus fsync publication for experiment evidence."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = path.open("x", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False

    def write(self, value: Mapping[str, object]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("JSONL records must be mappings")
        payload = _canonical_bytes(dict(value)).decode("ascii") + "\n"
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


@dataclass(frozen=True, slots=True)
class SourceEntry:
    label: str
    path: Path


def _source_entries() -> tuple[SourceEntry, ...]:
    entries: dict[str, Path] = {}

    def add(label: str, path: Path) -> None:
        resolved = path.resolve(strict=True)
        prior = entries.get(label)
        if prior is not None and prior != resolved:
            raise RuntimeError(f"duplicate source label {label}")
        entries[label] = resolved

    for path in (AGENT_EVOLVE_ROOT / "src" / "agent_evolve").rglob("*.py"):
        if path.is_file():
            add(path.relative_to(AGENT_EVOLVE_ROOT).as_posix(), path)
    for path in (
        AGENT_EVOLVE_ROOT / "examples" / "benchmarks" / "engibench_airfoil"
    ).glob("*.py"):
        if path.is_file():
            add(path.relative_to(AGENT_EVOLVE_ROOT).as_posix(), path)
    for name in (
        "run_airfoil_v7_reflective_feedback.py",
        "run_airfoil_v7_finite_oracle.py",
        "run_v6_closed_loop_memory_probe.py",
        "v6_closed_loop_probe_support.py",
    ):
        path = AGENT_EVOLVE_ROOT / "examples" / "development" / name
        add(path.relative_to(AGENT_EVOLVE_ROOT).as_posix(), path)
    focused_paths = {
        token
        for token in (*FOCUSED_PYTEST_ARGV, *FOCUSED_RUFF_ARGV)
        if token.endswith(".py")
    }
    for relative in sorted(focused_paths):
        path = AGENT_EVOLVE_ROOT / relative
        resolved = path.resolve(strict=True)
        label = (
            f"research_artifacts/{resolved.relative_to(ARTIFACT_ROOT).as_posix()}"
            if resolved.is_relative_to(ARTIFACT_ROOT)
            else resolved.relative_to(AGENT_EVOLVE_ROOT).as_posix()
        )
        add(label, resolved)
    for name in ("pyproject.toml", "uv.lock"):
        add(name, AGENT_EVOLVE_ROOT / name)

    for name in (
        "airfoil_adapter_v1.py",
        "airfoil_convergence_evidence.py",
        "airfoil_external_panel_v1.py",
        "airfoil_external_panel_v2.py",
        "calibration_engibench_docker.py",
        "calibration_harness.py",
    ):
        add(f"research_artifacts/scripts/{name}", SCRIPTS_ROOT / name)
    overlay = SCRIPTS_ROOT / "airfoil_convergence_overlay_v1" / "airfoil_analysis.py"
    add(
        "research_artifacts/scripts/airfoil_convergence_overlay_v1/airfoil_analysis.py",
        overlay,
    )

    for name in (PRICING_SNAPSHOT_NAME, CAPABILITY_SNAPSHOT_NAME):
        add(f"research_artifacts/data/{name}", ROUTE_DATA_ROOT / name)

    # 93 is intentionally superseded by 94. Artifacts 100 and 101 bind the
    # independent genericity evidence and prospective live release gate; 107
    # records the treatment-fidelity defect and 109 freezes its exact-action
    # repair before another provider launch.
    for number in (91, 94, 95, 96, 97, 98, 99, 100, 101, 105, 107, 109, 112):
        matches = tuple(ARTIFACT_ROOT.glob(f"{number}_*.md"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one immutable method artifact with prefix {number}_"
            )
        add(
            f"research_artifacts/method_contracts/{matches[0].name}",
            matches[0],
        )

    # The pinned evaluator imports an editable EngiBench checkout.  Bind the
    # exact five upstream templates that v2 verifies before each CFD solve.
    template_root = (
        Path.home()
        / ".cache"
        / "agent_evolve_aaai2027"
        / "engibench"
        / "engibench"
        / "problems"
        / "airfoil"
        / "templates"
    )
    for name in (
        "__init__.py",
        "airfoil_analysis.py",
        "airfoil_opt.py",
        "cli_interface.py",
        "pre_process.py",
    ):
        add(f"pinned_engibench/airfoil/templates/{name}", template_root / name)
    engibench_airfoil_root = template_root.parent
    for name in ("v0.py", "utils.py"):
        add(
            f"pinned_engibench/airfoil/{name}",
            engibench_airfoil_root / name,
        )

    return tuple(SourceEntry(label, entries[label]) for label in sorted(entries))


def source_snapshot() -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    aggregate = hashlib.sha256(SOURCE_FRAMING)
    for entry in _source_entries():
        content = entry.path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        files[entry.label] = {
            "sha256": digest,
            "bytes": len(content),
            "path": str(entry.path),
        }
        label = entry.label.encode("utf-8", errors="strict")
        aggregate.update(len(label).to_bytes(8, "big"))
        aggregate.update(label)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
    return {
        "schema_version": 1,
        "framing": SOURCE_FRAMING[:-1].decode("ascii"),
        "sha256": aggregate.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def _proxy_input_tokens(prompt: str) -> int:
    """Return a reproducible local proxy, never claimed as provider-exact."""

    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(prompt))


def prompt_preflight(
    prompt: str,
    *,
    max_output_tokens: int,
    request_kind: StructuredOutputRequestKind,
) -> dict[str, object]:
    if type(prompt) is not str or not prompt:
        raise ValueError("provider prompt must be non-empty")
    if type(request_kind) is not StructuredOutputRequestKind:
        raise TypeError("request_kind must be an exact StructuredOutputRequestKind")
    operation = (
        "typed_mutation"
        if request_kind is StructuredOutputRequestKind.PROPOSAL
        else "extract_insights"
    )
    expected_output_tokens = resolve_structured_output_budget(
        STRUCTURED_OUTPUT_BUDGET_POLICY,
        request_kind=request_kind,
        operation=operation,
    )
    if max_output_tokens != expected_output_tokens:
        raise RuntimeError(
            "Airfoil-v7 request drifted from its structured-output budget policy"
        )
    route_binding = _streamlake_route_snapshot_binding()
    route = route_binding["selected_route"]
    assert type(route) is dict
    context_length = int(route["context_length"])
    provider_completion_cap = int(route["max_completion_tokens"])
    encoded = prompt.encode("utf-8", errors="strict")
    proxy_tokens = _proxy_input_tokens(prompt)
    with_reserve = proxy_tokens + PROMPT_SCHEMA_RESERVE_TOKENS
    record = {
        "prompt_sha256": hashlib.sha256(encoded).hexdigest(),
        "utf8_bytes": len(encoded),
        "utf8_byte_cap": MAX_PROMPT_UTF8_BYTES,
        "unicode_characters": len(prompt),
        "proxy_tokenizer": PROMPT_PREFLIGHT_ENCODING,
        "proxy_input_tokens": proxy_tokens,
        "schema_and_transport_reserve_tokens": PROMPT_SCHEMA_RESERVE_TOKENS,
        "proxy_tokens_with_reserve": with_reserve,
        "telemetry_input_token_cap": MAX_INPUT_TOKENS,
        "request_kind": request_kind.value,
        "max_output_tokens": max_output_tokens,
        "provider_context_length": context_length,
        "provider_max_completion_tokens": provider_completion_cap,
        "pricing_snapshot_sha256": route_binding["pricing_snapshot"]["sha256"],
        "capability_snapshot_sha256": route_binding["capability_snapshot"][
            "sha256"
        ],
        "cap_plausible": (
            with_reserve <= MAX_INPUT_TOKENS
            and len(encoded) <= MAX_PROMPT_UTF8_BYTES
            and max_output_tokens <= provider_completion_cap
            and with_reserve + max_output_tokens <= context_length
        ),
        "caveat": (
            "The proxy is a local pre-provider guard, not DeepSeek/OpenRouter's "
            "authoritative tokenizer. Post-response telemetry remains authoritative."
        ),
    }
    if record["cap_plausible"] is not True:
        raise RuntimeError("prompt or completion budget exceeds the frozen route cap")
    return record


async def materialize_prompt_readiness() -> dict[str, object]:
    """Execute the full provider/CFD-free path and bind all seven prompt shapes."""

    fixture = compose_offline_experiment(tie_diagnostics=False, delay_seconds=0)
    semantics = AIRFOIL_V7_OPTIMIZATION_SEMANTICS
    semantics_prompt = render_optimization_semantics(semantics)
    treatment_policy = _treatment_compliance_policy_binding()
    engine = fixture.composition.engine
    if (
        engine.optimization_semantics is not semantics
        or engine.optimization_semantics_record != semantics.to_record()
        or engine.treatment_compliance_policy.policy_id
        != treatment_policy["policy_id"]
        or engine.treatment_compliance_policy.policy_version
        != treatment_policy["policy_version"]
        or engine.treatment_compliance_policy.definition_sha256
        != treatment_policy["definition_sha256"]
    ):
        raise RuntimeError("provider-free readiness policy bindings drifted")
    execution = asyncio.create_task(
        fixture.composition.optimizer.run(
            (NEUTRAL_PARENT, fixture.held_out_parent.candidate)
        )
    )
    while not execution.done():
        await asyncio.sleep(0.01)
    result = await execution
    proposals = tuple(fixture.generator.requests)
    reflections = tuple(fixture.generator.reflection_requests)
    if (
        result.final_state.logical_llm_calls != 7
        or len(proposals) != 5
        or len(reflections) != 2
    ):
        raise RuntimeError("provider-free readiness did not materialize seven calls")
    ordered: tuple[tuple[str, VariationGenerationRequest | ReflectionGenerationRequest], ...] = (
        ("g1_diagnostic_1", proposals[0]),
        ("g1_diagnostic_2", proposals[1]),
        ("g1_reflection_1", reflections[0]),
        ("g1_reflection_2", reflections[1]),
        ("g2_held_out_1", proposals[2]),
        ("g2_held_out_2", proposals[3]),
        ("g2_held_out_3", proposals[4]),
    )
    calls: list[dict[str, object]] = []
    for ordinal, (stage, request) in enumerate(ordered, start=1):
        semantics_block_count = request.prompt.count(semantics_prompt)
        semantics_marker_count = request.prompt.count(
            "OPTIMIZATION SEMANTICS (VERSIONED, AUTHORITATIVE)"
        )
        semantics_hash_count = request.prompt.count(semantics.definition_sha256)
        if (
            semantics_block_count != 1
            or semantics_marker_count != 1
            or semantics_hash_count != 1
        ):
            raise RuntimeError(
                "each proposal/reflection prompt must contain exact semantics once"
            )
        contract = (
            request.finite_variation_contract
            if type(request) is VariationGenerationRequest
            else None
        )
        request_kind = (
            StructuredOutputRequestKind.PROPOSAL
            if type(request) is VariationGenerationRequest
            else StructuredOutputRequestKind.REFLECTION
        )
        calls.append(
            {
                "ordinal": ordinal,
                "stage": stage,
                "kind": (
                    "proposal"
                    if type(request) is VariationGenerationRequest
                    else "reflection"
                ),
                "catalog_id": None if contract is None else contract.catalog_id,
                "call_id": request.call_id.value,
                "min_insights": (
                    request.min_insights
                    if type(request) is ReflectionGenerationRequest
                    else None
                ),
                "max_insights": (
                    request.max_insights
                    if type(request) is ReflectionGenerationRequest
                    else None
                ),
                "optimization_semantics_block_count": semantics_block_count,
                "optimization_semantics_marker_count": semantics_marker_count,
                "optimization_semantics_definition_sha256_count": (
                    semantics_hash_count
                ),
                **prompt_preflight(
                    request.prompt,
                    max_output_tokens=request.max_output_tokens,
                    request_kind=request_kind,
                ),
            }
        )
    return {
        "schema_version": 1,
        "provider_io_performed": False,
        "cfd_calls": 0,
        "logical_call_count": 7,
        "structured_output_budget_policy": (
            structured_output_budget_policy_record()
        ),
        "optimization_semantics": _optimization_semantics_binding(),
        "treatment_compliance_policy": treatment_policy,
        "all_caps_plausible": all(item["cap_plausible"] is True for item in calls),
        "maximum_proxy_tokens_with_reserve": max(
            int(item["proxy_tokens_with_reserve"]) for item in calls
        ),
        "calls": calls,
    }


def materialize_prompt_readiness_sync() -> dict[str, object]:
    """Materialize prompts and synchronously retire the evaluator worker."""

    loop = asyncio.new_event_loop()
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="airfoil_v7_prompt_readiness",
    )
    loop.set_default_executor(executor)
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(materialize_prompt_readiness())
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        loop.close()
        asyncio.set_event_loop(None)


def _path_record(contract: Any) -> dict[str, object]:
    paths: list[list[object]] = []
    for path in contract.editable_paths:
        segments = []
        for segment in path.segments:
            value = segment.value
            segments.append([type(segment).__name__, value])
        paths.append(segments)
    return {
        "editable_paths": paths,
        "max_changed_paths": contract.max_changed_paths,
        "max_operations": contract.max_operations,
        "allow_abstention": contract.allow_abstention,
    }


def _external_file_binding(path: Path, *, kind: str) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    content = resolved.read_bytes()
    return {
        "kind": kind,
        "path": str(resolved),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }


def _verification_environment_record() -> dict[str, object]:
    observation = observe_airfoil_v7_environment(_airfoil_readiness_spec())
    if not observation.passed:
        raise RuntimeError("observed Airfoil-v7 environment readiness failed")
    return observation.to_record()


def _airfoil_readiness_spec() -> AirfoilV7ReadinessSpec:
    settings = local_default_converged_settings()
    return AirfoilV7ReadinessSpec(
        evaluator_python=settings.python_executable,
        evaluator_script=settings.evaluator_script,
        dataset_arrow=settings.dataset_arrow,
        expected_dataset_sha256=EXPECTED_DATASET_SHA256,
        container_image=CONTAINER_IMAGE,
        cpu_set=settings.cpu_set,
        mpi_cores=settings.mpi_cores,
    )


def _invoked_absolute_path(path: Path) -> Path:
    """Keep a virtualenv invocation path without dereferencing its symlink."""

    return Path(os.path.abspath(path.expanduser()))


def _resource_lease_manifest_record(*, phase: str) -> dict[str, object]:
    """Freeze the domain adapter plugged into the generic lease port."""

    return {
        "resource_key": AIRFOIL_V7_RESOURCE_KEY,
        "lease_path": str(DEFAULT_RESOURCE_LEASE_PATH),
        "scope": "host_global_fixed_container_and_cpu_allocation",
        "phase": phase,
        "acquisition": "nonblocking_before_benchmark_construction",
        "release": "after_costly_work_and_durable_log_closure",
        "conflict_probe_id": AIRFOIL_V7_CONFLICT_PROBE_ID,
        "conflict_probe_version": AIRFOIL_V7_CONFLICT_PROBE_VERSION,
    }


def _airfoil_v7_telemetry_policy() -> AgenticTelemetryPolicy:
    """Return the one policy shared by manifest and live composition roots."""

    max_billable_attempt = _maximum_billable_cost_per_attempt()
    return AgenticTelemetryPolicy(
        requested_model=MODEL,
        allowed_resolved_models=(
            MODEL,
            "deepseek/deepseek-v4-pro-20260423",
        ),
        allowed_resolved_providers=("StreamLake",),
        max_cost_usd=max_billable_attempt,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_reasoning_tokens=PROVIDER_REASONING_MAX_TOKENS,
        max_attempt_count=MAX_PROVIDER_ATTEMPTS,
    )


def _production_resource_lease(
    run_id: str,
    phase: str,
) -> ExclusiveResourceLease:
    return create_airfoil_v7_resource_lease(
        _airfoil_readiness_spec(),
        lease_path=DEFAULT_RESOURCE_LEASE_PATH,
        run_id=run_id,
        phase=phase,
    )


def _verification_report_binding(path: Path) -> dict[str, object]:
    """Parse and authenticate the closed prelaunch verification report."""

    binding = _external_file_binding(
        path,
        kind="focused_prelaunch_verification_report",
    )
    try:
        report = json.loads(Path(str(binding["path"])).read_bytes())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("verification report is not valid JSON") from exc
    if type(report) is not dict or set(report) != {
        "schema_version",
        "status",
        "source_snapshot_sha256",
        "commands",
        "environment",
    }:
        raise RuntimeError("verification report root violates its closed schema")
    commands = report.get("commands")
    if type(commands) is not list or len(commands) != 2:
        raise RuntimeError("verification report requires pytest and Ruff commands")
    pytest_command, ruff_command = commands
    if type(pytest_command) is not dict or set(pytest_command) != {
        "id",
        "argv",
        "exit_code",
        "passed",
        "failed",
        "errors",
        "skipped",
        "stdout",
    }:
        raise RuntimeError("verification pytest command violates its closed schema")
    if type(ruff_command) is not dict or set(ruff_command) != {
        "id",
        "argv",
        "exit_code",
        "violations",
        "stdout",
    }:
        raise RuntimeError("verification Ruff command violates its closed schema")
    stdout = pytest_command.get("stdout")
    if type(stdout) is not dict or set(stdout) != {
        "kind",
        "path",
        "sha256",
        "bytes",
    }:
        raise RuntimeError("verification stdout binding is malformed")
    rebound_stdout = _external_file_binding(
        Path(str(stdout.get("path"))),
        kind="focused_pytest_stdout",
    )
    try:
        output = Path(str(rebound_stdout["path"])).read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise RuntimeError("verification stdout is not UTF-8") from exc
    pass_lines = tuple(_PYTEST_PASS_LINE.finditer(output))
    if len(pass_lines) != 1:
        raise RuntimeError("verification stdout lacks one unambiguous pass count")
    observed_passed = int(pass_lines[0].group("passed"))
    ruff_stdout = ruff_command.get("stdout")
    if type(ruff_stdout) is not dict or set(ruff_stdout) != {
        "kind",
        "path",
        "sha256",
        "bytes",
    }:
        raise RuntimeError("verification Ruff stdout binding is malformed")
    rebound_ruff_stdout = _external_file_binding(
        Path(str(ruff_stdout.get("path"))),
        kind="focused_ruff_stdout",
    )
    try:
        ruff_output = Path(str(rebound_ruff_stdout["path"])).read_text(
            encoding="utf-8"
        )
    except UnicodeError as exc:
        raise RuntimeError("verification Ruff stdout is not UTF-8") from exc
    if (
        report.get("schema_version") != 1
        or report.get("status") != "pass"
        or report.get("source_snapshot_sha256") != source_snapshot()["sha256"]
        or pytest_command.get("id") != "focused_pytest"
        or pytest_command.get("argv") != list(FOCUSED_PYTEST_ARGV)
        or pytest_command.get("exit_code") != 0
        or pytest_command.get("passed") != observed_passed
        or pytest_command.get("failed") != 0
        or pytest_command.get("errors") != 0
        or pytest_command.get("skipped") != 0
        or stdout != rebound_stdout
        or ruff_command.get("id") != "focused_ruff"
        or ruff_command.get("argv") != list(FOCUSED_RUFF_ARGV)
        or ruff_command.get("exit_code") != 0
        or ruff_command.get("violations") != 0
        or ruff_stdout != rebound_ruff_stdout
        or ruff_output != "All checks passed!\n"
        or report.get("environment") != _verification_environment_record()
    ):
        raise RuntimeError("verification report status, source, counts, or gates drifted")
    return {**binding, "validated_report": report}


def _seed_qualification_spec(
    *,
    run_id: str,
    output_dir: Path,
    verification_report: Mapping[str, object],
) -> dict[str, object]:
    held_out = materialize_held_out_parent()
    settings = local_default_converged_settings()
    phenotype_policy = AirfoilV7PhenotypeIdentityPolicy()
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "authorization": {
            "real_cfd_seed_evaluations": 2,
            "provider_calls": 0,
            "credentials_may_be_read": False,
            "child_evaluations": 0,
        },
        "parents": {
            "diagnostic": {
                "candidate": NEUTRAL_PARENT,
                "candidate_sha256": candidate_sha256(NEUTRAL_PARENT),
                "typed_configuration_sha256": typed_json_sha256(
                    freeze_json(NEUTRAL_PARENT)
                ),
                "phenotype": phenotype_policy.identify(
                    NEUTRAL_PARENT
                ).to_trace_record(),
                "no_cfd_validation": validate_frozen_no_cfd_candidate(
                    NEUTRAL_PARENT
                ).to_record(),
            },
            "held_out": {
                **held_out.to_record(),
                "phenotype": phenotype_policy.identify(
                    held_out.candidate
                ).to_trace_record(),
            },
        },
        "evaluator": {
            "identity": EVALUATOR_IDENTITY.to_record(),
            "task_sha256": TASK_SHA256,
            "python_executable": str(
                _invoked_absolute_path(settings.python_executable)
            ),
            "evaluator_script": str(settings.evaluator_script.resolve()),
            "dataset_arrow": str(settings.dataset_arrow.resolve()),
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "container_image": CONTAINER_IMAGE,
            "receipt_root": str(output_dir / "raw_receipts"),
            "work_root": str(
                Path("/tmp") / "agent_evolve_airfoil_v7_seed" / run_id
            ),
            "cpu_set": settings.cpu_set,
            "mpi_cores": settings.mpi_cores,
            "timeout_seconds": settings.timeout_seconds,
            "attempts_per_seed": CFD_ATTEMPTS_PER_UNIQUE_CANDIDATE,
        },
        "resource_lease": _resource_lease_manifest_record(
            phase="seed_qualification"
        ),
        "policy_identities": {
            "phenotype_definition_sha256": PHENOTYPE_DEFINITION_SHA256,
            "archive_relation_definition_sha256": ARCHIVE_DEFINITION_SHA256,
            "reward_definition_sha256": REWARD_DEFINITION_SHA256,
        },
        "verification_report": dict(verification_report),
    }


def build_seed_qualification_manifest_record(
    *,
    run_id: str,
    output_dir: Path,
    verification_report_path: Path,
) -> dict[str, object]:
    run_id, output_dir = _validate_run_target(run_id, output_dir)
    record: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": SEED_MANIFEST_KIND,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "qualification": _seed_qualification_spec(
            run_id=run_id,
            output_dir=output_dir,
            verification_report=_verification_report_binding(
                verification_report_path
            ),
        ),
        "source_snapshot": source_snapshot(),
    }
    record["manifest_sha256"] = _sha256_record(
        record,
        domain=SEED_MANIFEST_FRAMING,
    )
    return record


def write_seed_qualification_manifest(
    path: Path,
    *,
    run_id: str,
    output_dir: Path,
    verification_report_path: Path,
) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    record = build_seed_qualification_manifest_record(
        run_id=run_id,
        output_dir=output_dir,
        verification_report_path=verification_report_path,
    )
    write_json_atomic(resolved, record)
    return record


@dataclass(frozen=True, slots=True)
class VerifiedSeedQualificationManifest:
    path: Path
    record: dict[str, object]
    run_id: str
    output_dir: Path
    manifest_sha256: str
    source_sha256: str


def verify_seed_qualification_manifest(
    path: Path,
    *,
    require_output_absent: bool,
    enforce_canonical_output: bool = True,
) -> VerifiedSeedQualificationManifest:
    resolved = path.expanduser().resolve(strict=True)
    try:
        record = json.loads(resolved.read_bytes())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("seed qualification manifest is not valid JSON") from exc
    if type(record) is not dict:
        raise TypeError("seed qualification manifest root must be an object")
    claimed = record.get("manifest_sha256")
    unsigned = dict(record)
    unsigned.pop("manifest_sha256", None)
    observed = _sha256_record(unsigned, domain=SEED_MANIFEST_FRAMING)
    if claimed != observed:
        raise RuntimeError("seed qualification manifest self-hash mismatch")
    if (
        record.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or record.get("kind") != SEED_MANIFEST_KIND
    ):
        raise RuntimeError("seed qualification manifest kind/schema mismatch")
    qualification = record.get("qualification")
    if type(qualification) is not dict:
        raise TypeError("qualification specification must be an object")
    run_id, output_dir = _validate_run_target(
        qualification.get("run_id"),
        Path(str(qualification.get("output_dir"))),
    )
    if (
        enforce_canonical_output
        and output_dir.parent != DEFAULT_SEED_QUALIFICATION_ROOT.resolve()
    ):
        raise RuntimeError("seed output directory is outside the canonical root")
    if require_output_absent and output_dir.exists():
        raise FileExistsError(output_dir)
    if qualification != _seed_qualification_spec(
        run_id=run_id,
        output_dir=output_dir,
        verification_report=_verification_report_binding(
            Path(str(qualification.get("verification_report", {}).get("path")))
        ),
    ):
        raise RuntimeError("seed qualification evaluator, parents, or policies drifted")
    current_source = source_snapshot()
    if record.get("source_snapshot") != current_source:
        raise RuntimeError("seed qualification source snapshot drifted")
    return VerifiedSeedQualificationManifest(
        path=resolved,
        record=record,
        run_id=run_id,
        output_dir=output_dir,
        manifest_sha256=observed,
        source_sha256=str(current_source["sha256"]),
    )


def _payload_record(payload: DetailedEvaluationPayload) -> dict[str, object]:
    return {
        "failure": (
            None
            if payload.failure is None
            else {
                "category": payload.failure.category.value,
                "code": payload.failure.code.value,
                "message": payload.failure.message,
                "retryable": payload.failure.retryable,
                "exception_type": payload.failure.exception_type,
            }
        ),
        "objectives": dict(payload.objectives),
        "violations": dict(payload.violations),
        "checks": [item.to_record() for item in payload.checks],
        "receipt": (
            None
            if payload.receipt is None
            else {
                "artifact_id": payload.receipt.artifact_id.value,
                "sha256": payload.receipt.sha256_hex,
                "media_type": payload.receipt.media_type,
                "byte_count": payload.receipt.size_bytes,
            }
        ),
        "evaluator": payload.evaluator.to_record(),
        "active_wall_seconds": payload.active_wall_seconds,
        "resource_queue_wall_seconds": payload.resource_queue_wall_seconds,
    }


def _find_one_raw_receipt(run_dir: Path, configuration: object) -> Path:
    key = candidate_sha256(configuration)
    matches = tuple((run_dir / "raw_receipts").glob(f"**/{key}.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one raw evaluator receipt for {key}, found {len(matches)}"
        )
    return matches[0].resolve(strict=True)


def _verify_raw_success_receipt(
    path: Path,
    *,
    configuration: object,
) -> dict[str, object]:
    content = path.read_bytes()
    record = json.loads(content)
    if type(record) is not dict:
        raise TypeError("raw seed receipt must be an object")
    provenance = record.get("provenance")
    dataset = provenance.get("dataset") if type(provenance) is dict else None
    source_hashes = (
        provenance.get("source_sha256") if type(provenance) is dict else None
    )
    snapshot_files = source_snapshot()["files"]
    expected_sources = {
        "this_evaluator": snapshot_files[
            "research_artifacts/scripts/airfoil_external_panel_v2.py"
        ]["sha256"],
        "inherited_external_panel_v1": snapshot_files[
            "research_artifacts/scripts/airfoil_external_panel_v1.py"
        ]["sha256"],
        "convergence_contract": snapshot_files[
            "research_artifacts/scripts/airfoil_convergence_evidence.py"
        ]["sha256"],
        "convergence_overlay_airfoil_analysis": snapshot_files[
            "research_artifacts/scripts/airfoil_convergence_overlay_v1/"
            "airfoil_analysis.py"
        ]["sha256"],
    }
    overlay = provenance.get("overlay") if type(provenance) is dict else None
    runtime_overlay = record.get("runtime_overlay")
    runtime_installed = (
        runtime_overlay.get("installed_template_sha256")
        if type(runtime_overlay) is dict
        else None
    )
    points = record.get("points")
    if (
        record.get("schema_version") != 2
        or record.get("evaluator_id") != V2_EVALUATOR_ID
        or record.get("status") != "evaluated"
        or record.get("candidate_sha256") != candidate_sha256(configuration)
        or record.get("task_sha256") != TASK_SHA256
        or record.get("evaluator_calls") != 3
        or type(provenance) is not dict
        or provenance.get("evaluator_id") != V2_EVALUATOR_ID
        or provenance.get("evidence_contract") != EVIDENCE_CONTRACT_ID
        or provenance.get("adflow_evaluator_id") != ADFLOW_EVALUATOR_ID
        or provenance.get("container_image") != CONTAINER_IMAGE
        or type(dataset) is not dict
        or dataset.get("arrow_sha256") != EXPECTED_DATASET_SHA256
        or type(source_hashes) is not dict
        or any(source_hashes.get(key) != value for key, value in expected_sources.items())
        or type(overlay) is not dict
        or overlay.get("overlay_id")
        != "engibench_airfoil_adflow_convergence_overlay_v1"
        or overlay.get("overlay_airfoil_analysis_sha256")
        != expected_sources["convergence_overlay_airfoil_analysis"]
        or overlay.get("history_extractor_module_sha256")
        != expected_sources["convergence_contract"]
        or type(runtime_overlay) is not dict
        or runtime_overlay.get("overlay_id") != overlay.get("overlay_id")
        or runtime_overlay.get("overlay_airfoil_analysis_sha256")
        != overlay.get("overlay_airfoil_analysis_sha256")
        or runtime_overlay.get("history_extractor_module_sha256")
        != overlay.get("history_extractor_module_sha256")
        or type(runtime_installed) is not dict
        or runtime_installed.get("airfoil_analysis.py")
        != expected_sources["convergence_overlay_airfoil_analysis"]
        or runtime_installed.get("airfoil_convergence_evidence.py")
        != expected_sources["convergence_contract"]
        or type(points) is not list
        or len(points) != 3
    ):
        raise RuntimeError("raw seed receipt provenance or identity mismatch")
    for index, point in enumerate(points):
        evidence = point.get("evaluator_evidence") if type(point) is dict else None
        witness = evidence.get("witness") if type(evidence) is dict else None
        status = (
            witness.get("authoritative_status")
            if type(witness) is dict
            else None
        )
        if (
            point.get("index") != index
            or type(evidence) is not dict
            or evidence.get("contract_id") != EVIDENCE_CONTRACT_ID
            or evidence.get("evaluator_id") != ADFLOW_EVALUATOR_ID
            or evidence.get("accepted") is not True
            or type(status) is not dict
            or any(
                status.get(field) is not False
                for field in (
                    "solve_failed",
                    "fatal_fail",
                    "check_solution_failure",
                )
            )
        ):
            raise RuntimeError("raw seed receipt convergence witness mismatch")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "record": record,
        "evaluator_calls": 3,
    }


def _verify_raw_failure_receipt(
    path: Path,
    *,
    configuration: object,
    payload: DetailedEvaluationPayload,
) -> dict[str, object]:
    """Bind an exact raw failure receipt to its typed adapter projection."""

    if payload.failure is None or payload.receipt is None:
        raise RuntimeError("raw failure receipt requires a typed failure and receipt")
    content = path.read_bytes()
    record = json.loads(content)
    if type(record) is not dict:
        raise TypeError("raw seed failure receipt must be an object")
    digest = hashlib.sha256(content).hexdigest()
    failure = record.get("failure")
    calls = record.get("evaluator_calls")
    failed_point = record.get("failed_point_index")
    typed_pair = (payload.failure.category.value, payload.failure.code.value)
    raw_type = failure.get("type") if type(failure) is dict else None
    expected_pairs = {
        ("infrastructure_or_evaluator_failure", "WitnessBoundaryFailure"): (
            "system",
            "evaluator_contract_violation",
        ),
        ("candidate_invalid", "CandidateSolverOutcome"): (
            "candidate",
            "numerical_nonconvergence_attributable_to_candidate",
        ),
    }
    candidate_identity = record.get("candidate_sha256")
    if (
        record.get("schema_version") != 2
        or record.get("evaluator_id") != V2_EVALUATOR_ID
        or record.get("mode") != "evaluate"
        or candidate_identity != candidate_sha256(configuration)
        or type(calls) is not int
        or not 1 <= calls <= 3
        or type(failed_point) is not int
        or failed_point not in range(3)
        or calls != failed_point + 1
        or type(failure) is not dict
        or not isinstance(failure.get("message"), str)
        or not failure["message"].strip()
        or expected_pairs.get((record.get("status"), raw_type)) != typed_pair
        or (
            raw_type == "CandidateSolverOutcome"
            and record.get("failure_classification")
            != "authoritative_solver_failure"
        )
        or payload.receipt.sha256_hex != digest
        or payload.receipt.size_bytes != len(content)
        or payload.receipt.media_type != "application/json"
        or payload.failure.diagnostics_artifact_id != payload.receipt.artifact_id
    ):
        raise RuntimeError("raw seed failure receipt/typed outcome mismatch")
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": len(content),
        "record": record,
        "evaluator_calls": calls,
    }


def _verify_raw_seed_receipt(
    path: Path,
    *,
    configuration: object,
    payload: DetailedEvaluationPayload,
) -> dict[str, object]:
    if payload.failure is None:
        return _verify_raw_success_receipt(path, configuration=configuration)
    return _verify_raw_failure_receipt(
        path,
        configuration=configuration,
        payload=payload,
    )


def create_seed_qualification_benchmark(
    run_id: str,
    run_dir: Path,
) -> AgenticBenchmark:
    settings = replace(
        local_default_converged_settings(),
        output_root=run_dir / "raw_receipts",
        work_root=Path("/tmp") / "agent_evolve_airfoil_v7_seed" / run_id,
    )
    raw = ConvergenceQualifiedAirfoilPanelProblem(settings)
    problem = AirfoilV7Problem(raw_problem=raw)
    return AgenticBenchmark(
        problem=problem,
        reward=AIRFOIL_V7_REWARD_BINDING,
        detailed_evaluator=problem.detailed_evaluator,
        outcome_relation=AIRFOIL_V7_ARCHIVE_RELATION,
        phenotype_identity=AirfoilV7PhenotypeIdentityPolicy(),
        finite_variation_catalogs=(
            AirfoilV7ShapeVariationCatalog(),
            AirfoilV7TrimVariationCatalog(),
            AirfoilV7UnionVariationCatalog(),
        ),
    )


@dataclass(frozen=True, slots=True)
class SeedQualificationDependencies:
    benchmark_factory: Callable[[str, Path], AgenticBenchmark]
    raw_receipt_locator: Callable[[Path, object], Path] = _find_one_raw_receipt
    resource_lease_factory: Callable[
        [str, str], ExclusiveResourceLease
    ] = _production_resource_lease
    enforce_canonical_output: bool = True


def execute_seed_qualification_with_dependencies(
    manifest_path: Path,
    dependencies: SeedQualificationDependencies,
) -> dict[str, object]:
    """Evaluate exactly two seeds serially with no generator or credentials."""

    verified = verify_seed_qualification_manifest(
        manifest_path,
        require_output_absent=True,
        enforce_canonical_output=dependencies.enforce_canonical_output,
    )
    run_dir = verified.output_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    _directory_fsync(run_dir.parent)
    write_bytes_atomic(run_dir / "qualification_manifest.json", verified.path.read_bytes())
    source_writer = DurableJsonlWriter(run_dir / "source_verifications.jsonl")
    status = "failed"
    pending: BaseException | None = None
    result: dict[str, object] | None = None
    resource_lease: ExclusiveResourceLease | None = None
    completed_evaluations = 0
    raw_solver_calls = 0
    try:
        resource_lease = dependencies.resource_lease_factory(
            verified.run_id,
            "seed_qualification",
        )
        lease_receipt = resource_lease.acquire()
        write_json_atomic(
            run_dir / "resource_lease_acquired.json",
            {
                "schema_version": 1,
                "phase": "seed_qualification",
                "receipt": lease_receipt.to_record(),
            },
        )
        qualification = verified.record["qualification"]
        if type(qualification) is not dict:
            raise RuntimeError("verified qualification record changed type")
        benchmark = dependencies.benchmark_factory(verified.run_id, run_dir)
        evaluator = benchmark.detailed_evaluator
        if evaluator is None:
            raise RuntimeError("seed qualification requires detailed evaluation")
        held_out = materialize_held_out_parent()
        phenotype_policy = AirfoilV7PhenotypeIdentityPolicy()
        manifest_parents = qualification.get("parents")
        if (
            type(manifest_parents) is not dict
            or manifest_parents.get("held_out")
            != {
                **held_out.to_record(),
                "phenotype": phenotype_policy.identify(
                    held_out.candidate
                ).to_trace_record(),
            }
            or type(manifest_parents.get("diagnostic")) is not dict
            or manifest_parents["diagnostic"].get("candidate") != NEUTRAL_PARENT
        ):
            raise RuntimeError("runtime seed parents drifted from qualification manifest")
        configurations = (NEUTRAL_PARENT, held_out.candidate)
        labels = ("diagnostic", "held_out")
        observations = []
        for ordinal, (label, configuration) in enumerate(
            zip(labels, configurations, strict=True),
            start=1,
        ):
            current = verify_seed_qualification_manifest(
                verified.path,
                require_output_absent=False,
                enforce_canonical_output=False,
            )
            if current.source_sha256 != verified.source_sha256:
                raise RuntimeError("seed qualification source drifted before CFD")
            source_writer.write(
                {
                    "schema_version": 1,
                    "stage": f"pre_seed_cfd_{ordinal}",
                    "source_sha256": current.source_sha256,
                    "manifest_sha256": current.manifest_sha256,
                }
            )
            payload = evaluator.evaluate_evidence(normalize_candidate(configuration))
            completed_evaluations += 1
            raw_path = dependencies.raw_receipt_locator(run_dir, configuration)
            raw = _verify_raw_seed_receipt(
                raw_path,
                configuration=configuration,
                payload=payload,
            )
            raw_solver_calls += int(raw["evaluator_calls"])
            observations.append(
                {
                    "ordinal": ordinal,
                    "label": label,
                    "configuration": normalize_candidate(configuration),
                    "candidate_sha256": candidate_sha256(configuration),
                    "typed_configuration_sha256": typed_json_sha256(
                        freeze_json(configuration)
                    ),
                    "phenotype": phenotype_policy.identify(
                        configuration
                    ).to_trace_record(),
                    "payload": _payload_record(payload),
                    "raw_receipt": {
                        "relative_path": raw_path.relative_to(run_dir).as_posix(),
                        "sha256": raw["sha256"],
                        "bytes": raw["bytes"],
                    },
                }
            )
            if payload.failure is not None:
                sealed_failure = verify_seed_qualification_manifest(
                    verified.path,
                    require_output_absent=False,
                    enforce_canonical_output=False,
                )
                if sealed_failure.source_sha256 != verified.source_sha256:
                    raise RuntimeError(
                        "seed qualification source drifted after failed CFD"
                    )
                source_writer.write(
                    {
                        "schema_version": 1,
                        "stage": f"post_seed_cfd_{ordinal}_failure_seal",
                        "source_sha256": sealed_failure.source_sha256,
                        "manifest_sha256": sealed_failure.manifest_sha256,
                    }
                )
                unsigned_failure: dict[str, object] = {
                    "schema_version": 1,
                    "status": (
                        "rejected"
                        if payload.failure.category.value == "candidate"
                        else "invalidated"
                    ),
                    "provider_io_performed": False,
                    "credentials_read": False,
                    "authorized_cfd_candidate_evaluations": 2,
                    "cfd_candidate_evaluations": completed_evaluations,
                    "authorized_raw_solver_calls": 6,
                    "raw_solver_calls": raw_solver_calls,
                    "distinct_airfoil_v7_phenotypes": False,
                    "qualification_manifest_sha256": verified.manifest_sha256,
                    "source_sha256": verified.source_sha256,
                    "seeds": observations,
                }
                unsigned_failure["qualification_sha256"] = _sha256_record(
                    unsigned_failure,
                    domain=SEED_RESULT_FRAMING,
                )
                result = unsigned_failure
                write_json_atomic(run_dir / "qualification_result.json", result)
                status = str(result["status"])
                raise RuntimeError(
                    "seed qualification stopped on authenticated typed failure"
                )
        sealed = verify_seed_qualification_manifest(
            verified.path,
            require_output_absent=False,
            enforce_canonical_output=False,
        )
        if sealed.source_sha256 != verified.source_sha256:
            raise RuntimeError(
                "seed qualification source drifted after CFD before result sealing"
            )
        source_writer.write(
            {
                "schema_version": 1,
                "stage": "post_seed_cfd_2_pre_result_seal",
                "source_sha256": sealed.source_sha256,
                "manifest_sha256": sealed.manifest_sha256,
            }
        )
        all_success = all(row["payload"]["failure"] is None for row in observations)
        distinct_phenotypes = len(
            {
                row["phenotype"]["value_sha256"] for row in observations
            }
        ) == 2
        unsigned_result: dict[str, object] = {
            "schema_version": 1,
            "status": (
                "qualified" if all_success and distinct_phenotypes else "rejected"
            ),
            "provider_io_performed": False,
            "credentials_read": False,
            "authorized_cfd_candidate_evaluations": 2,
            "cfd_candidate_evaluations": completed_evaluations,
            "authorized_raw_solver_calls": 6,
            "raw_solver_calls": raw_solver_calls,
            "distinct_airfoil_v7_phenotypes": distinct_phenotypes,
            "qualification_manifest_sha256": verified.manifest_sha256,
            "source_sha256": verified.source_sha256,
            "seeds": observations,
        }
        unsigned_result["qualification_sha256"] = _sha256_record(
            unsigned_result,
            domain=SEED_RESULT_FRAMING,
        )
        result = unsigned_result
        write_json_atomic(run_dir / "qualification_result.json", result)
        status = str(result["status"])
        if not all_success or not distinct_phenotypes:
            raise RuntimeError("frozen seeds failed success or phenotype qualification")
    except BaseException as exc:
        pending = exc
        if not (run_dir / "failure.json").exists():
            try:
                write_json_atomic(
                    run_dir / "failure.json",
                    {
                        "schema_version": 1,
                        "failure_type": type(exc).__name__,
                        "safe_message": str(exc)[:1_024],
                        "provider_io_performed": False,
                        "credentials_read": False,
                        "cfd_candidate_evaluations": completed_evaluations,
                        "raw_solver_calls": raw_solver_calls,
                    },
                )
            except BaseException as artifact_exc:
                exc.add_note(
                    "seed failure artifact also failed: "
                    f"{type(artifact_exc).__name__}"
                )
    finally:
        try:
            source_writer.close()
        except BaseException as exc:
            if pending is None:
                pending = exc
            else:
                pending.add_note(
                    f"source log close also failed: {type(exc).__name__}"
                )
        if resource_lease is not None and resource_lease.active:
            try:
                release = resource_lease.release(
                    outcome=status if pending is None else "failed",
                    failure_type=(
                        None if pending is None else type(pending).__name__
                    ),
                )
                write_json_atomic(
                    run_dir / "resource_lease_released.json",
                    {
                        "schema_version": 1,
                        "phase": "seed_qualification",
                        "release": release,
                    },
                )
            except BaseException as exc:
                if pending is None:
                    pending = exc
                else:
                    pending.add_note(
                        "resource lease release/evidence also failed: "
                        f"{type(exc).__name__}"
                    )
        try:
            _directory_fsync(run_dir)
            _finalize_run(
                run_dir,
                status=status if pending is None else "failed",
            )
        except BaseException as exc:
            if pending is None:
                pending = exc
            else:
                pending.add_note(
                    f"seed run finalization also failed: {type(exc).__name__}"
                )
    if pending is not None:
        raise pending
    if result is None:
        raise RuntimeError("seed qualification returned without a result")
    return result


def load_seed_qualification_binding(result_path: Path) -> dict[str, object]:
    """Verify and project one completed seed qualification for provider replay."""

    resolved = result_path.expanduser().resolve(strict=True)
    result = json.loads(resolved.read_bytes())
    if type(result) is not dict:
        raise TypeError("qualification result root must be an object")
    claimed = result.get("qualification_sha256")
    unsigned = dict(result)
    unsigned.pop("qualification_sha256", None)
    observed = _sha256_record(unsigned, domain=SEED_RESULT_FRAMING)
    if claimed != observed or result.get("status") != "qualified":
        raise RuntimeError("seed qualification result hash/status mismatch")
    manifest_path = resolved.parent / "qualification_manifest.json"
    verified_manifest = verify_seed_qualification_manifest(
        manifest_path,
        require_output_absent=False,
        enforce_canonical_output=False,
    )
    if (
        result.get("qualification_manifest_sha256")
        != verified_manifest.manifest_sha256
        or result.get("source_sha256") != verified_manifest.source_sha256
        or result.get("cfd_candidate_evaluations") != 2
        or result.get("raw_solver_calls") != 6
        or result.get("distinct_airfoil_v7_phenotypes") is not True
        or result.get("provider_io_performed") is not False
        or result.get("credentials_read") is not False
    ):
        raise RuntimeError("seed qualification accounting/manifest mismatch")
    qualification_spec = verified_manifest.record.get("qualification")
    if type(qualification_spec) is not dict:
        raise RuntimeError("qualified manifest lost its qualification specification")
    verification_report = qualification_spec.get("verification_report")
    if type(verification_report) is not dict:
        raise RuntimeError("qualified manifest lacks its verification report binding")
    seeds = result.get("seeds")
    if type(seeds) is not list or len(seeds) != 2:
        raise RuntimeError("qualification must bind exactly two seeds")
    bound_seeds = []
    for expected_label, row in zip(("diagnostic", "held_out"), seeds, strict=True):
        if type(row) is not dict or row.get("label") != expected_label:
            raise RuntimeError("qualification seed label/order mismatch")
        configuration = row.get("configuration")
        if row.get("candidate_sha256") != candidate_sha256(configuration):
            raise RuntimeError("qualification candidate hash mismatch")
        phenotype = AirfoilV7PhenotypeIdentityPolicy().identify(
            configuration
        ).to_trace_record()
        if row.get("phenotype") != phenotype:
            raise RuntimeError("qualification seed phenotype identity mismatch")
        payload = row.get("payload")
        if type(payload) is not dict or payload.get("failure") is not None:
            raise RuntimeError("qualification contains a failed seed")
        raw = row.get("raw_receipt")
        if type(raw) is not dict:
            raise RuntimeError("qualification seed lacks raw receipt binding")
        raw_path = (resolved.parent / str(raw.get("relative_path"))).resolve(
            strict=True
        )
        if not raw_path.is_relative_to(resolved.parent):
            raise RuntimeError("qualification raw receipt escapes its run directory")
        verified_raw = _verify_raw_success_receipt(
            raw_path,
            configuration=configuration,
        )
        if (
            raw.get("sha256") != verified_raw["sha256"]
            or raw.get("bytes") != verified_raw["bytes"]
        ):
            raise RuntimeError("qualification raw receipt content drifted")
        bound_seeds.append(
            {
                "label": expected_label,
                "configuration": configuration,
                "candidate_sha256": row["candidate_sha256"],
                "typed_configuration_sha256": row[
                    "typed_configuration_sha256"
                ],
                "phenotype": phenotype,
                "payload": payload,
                "raw_receipt_path": str(raw_path),
                "raw_receipt_sha256": raw["sha256"],
                "raw_receipt_bytes": raw["bytes"],
            }
        )
    if len({row["phenotype"]["value_sha256"] for row in bound_seeds}) != 2:
        raise RuntimeError("qualification seed phenotypes are not distinct")
    return {
        "schema_version": 1,
        "qualification_result_path": str(resolved),
        "qualification_sha256": observed,
        "qualification_manifest_sha256": verified_manifest.manifest_sha256,
        "source_sha256": verified_manifest.source_sha256,
        "verification_report": dict(verification_report),
        "seeds": bound_seeds,
    }


def _live_id_namespace(run_id: str) -> str:
    """Derive the source-bound generic-ID namespace and reject it prelaunch."""

    namespace = f"ae7_{run_id}"
    validate_id_namespace(namespace)
    return namespace


def _launch_spec(
    *,
    run_id: str,
    output_dir: Path,
    prompt_readiness: Mapping[str, object],
    seed_qualification: Mapping[str, object],
) -> dict[str, object]:
    _live_id_namespace(run_id)
    held_out = materialize_held_out_parent()
    shape_catalog = AirfoilV7ShapeVariationCatalog()
    trim_catalog = AirfoilV7TrimVariationCatalog()
    union_catalog = AirfoilV7UnionVariationCatalog()
    shape = bind_finite_variation_catalog(shape_catalog, freeze_json(NEUTRAL_PARENT))
    trim = bind_finite_variation_catalog(trim_catalog, freeze_json(NEUTRAL_PARENT))
    union = bind_finite_variation_catalog(
        union_catalog,
        freeze_json(held_out.candidate),
    )
    mutation_contracts = {
        "shape": _path_record(SHAPE_MUTATION_CONTRACT),
        "trim": _path_record(TRIM_MUTATION_CONTRACT),
        "union": _path_record(UNION_MUTATION_CONTRACT),
    }
    mutation_contracts = {
        name: {
            **record,
            "definition_sha256": _sha256_record(
                record,
                domain=(
                    b"agent-evolve:airfoil-v7-launch-mutation-contract:v1\x00"
                    + name.encode("ascii")
                    + b"\x00"
                ),
            ),
        }
        for name, record in mutation_contracts.items()
    }
    logical_calls = OPTIMIZER_BUDGET.max_logical_llm_calls
    route_snapshot_binding = _streamlake_route_snapshot_binding()
    cost_envelope = _cost_envelope_record(logical_calls=logical_calls)
    default_settings = local_default_converged_settings()
    telemetry_policy = _airfoil_v7_telemetry_policy()
    g2_gate_policy = AirfoilV7G2PrequeueGatePolicy()
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "claim_boundary": (
            "Single developmental Airfoil-v7 kill test; not paper-ready, SOTA, "
            "genericity, or replicated wall-clock evidence."
        ),
        "model_route": {
            "requested_model": MODEL,
            "provider": PROVIDER,
            "provider_options": {"only": list(PROVIDER_ONLY)},
            "allowed_resolved_providers": ["StreamLake"],
            "minimum_queue_concurrency": MIN_PROVIDER_CONCURRENCY,
            "queue_max_in_flight_from_v6_factory": 4,
            "queue_max_pending_from_v6_factory": 8,
            "max_attempts": MAX_PROVIDER_ATTEMPTS,
            "attempt_timeout_seconds": (
                PROVIDER_ATTEMPT_TIMEOUT_NS // 1_000_000_000
            ),
            "reasoning": {
                "provider_parameter": "reasoning.max_tokens",
                "max_tokens": PROVIDER_REASONING_MAX_TOKENS,
            },
            "backoff": {
                "kind": "deterministic_exponential_with_task_keyed_jitter",
                "base_seconds": 1,
                "max_seconds": 8,
            },
            "factory": (
                "examples.development.run_v6_closed_loop_memory_probe."
                "create_live_stack"
            ),
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "structured_output_budget_policy": (
                structured_output_budget_policy_record()
            ),
            "route_snapshot_binding": route_snapshot_binding,
            "telemetry_policy": telemetry_policy.to_trace_record(),
            "telemetry_policy_sha256": telemetry_policy.policy_sha256,
        },
        "cost_envelope_usd": cost_envelope,
        "budget": OPTIMIZER_BUDGET.to_trace_record(),
        "experiment": {
            "planner_policy_id": PLANNER_POLICY_ID,
            "planner_policy_version": PLANNER_POLICY_VERSION,
            "run_seed": RUN_SEED,
            "seed_count": 2,
            "g1_proposal_calls": 2,
            "reflection_calls": 2,
            "reflection_topology": "one_contrast_one_card_concurrent_shards",
            "g2_proposal_calls": 3,
            "successful_full_path_logical_calls": 7,
            "full_path_unique_evaluation_range": [4, 7],
            "full_path_delegated_child_evaluation_range": [2, 5],
            "fully_administered_distinct_full_path_unique_evaluations": 7,
            "treatment_rejection_accounting": (
                "accepted_provider_call_but_no_candidate_and_no_evaluator_entry"
            ),
            "equal_diagnostic_reward_early_stop_calls": 4,
            "equal_diagnostic_reward_early_stop_unique_evaluations": 4,
            "clean_early_stop_reason_codes": sorted(
                CLEAN_EARLY_STOP_REASON_CODES
            ),
            "evaluator_concurrency": EVALUATOR_CONCURRENCY,
            "cfd_attempts_per_unique_candidate": (
                CFD_ATTEMPTS_PER_UNIQUE_CANDIDATE
            ),
            "llm_wave_concurrency": {"g1": 2, "reflection": 2, "g2": 3},
        },
        "parents": {
            "diagnostic": {
                "candidate": NEUTRAL_PARENT,
                "candidate_sha256": candidate_sha256(NEUTRAL_PARENT),
                "typed_configuration_sha256": typed_json_sha256(
                    freeze_json(NEUTRAL_PARENT)
                ),
                "no_cfd_validation": validate_frozen_no_cfd_candidate(
                    NEUTRAL_PARENT
                ).to_record(),
            },
            "held_out": held_out.to_record(),
        },
        "seed_qualification": dict(seed_qualification),
        "policies": {
            "task_sha256": TASK_SHA256,
            "objective": OBJECTIVE_NAME,
            "violation": VIOLATION_NAME,
            "optimization_semantics": _optimization_semantics_binding(),
            "treatment_compliance": _treatment_compliance_policy_binding(),
            "archive_relation_definition_sha256": ARCHIVE_DEFINITION_SHA256,
            "local_reward_definition_sha256": REWARD_DEFINITION_SHA256,
            "phenotype_definition_sha256": PHENOTYPE_DEFINITION_SHA256,
            "evaluator": {
                "evaluator_id": EVALUATOR_IDENTITY.evaluator_id,
                "evaluator_version": EVALUATOR_IDENTITY.evaluator_version,
                "evaluator_context_sha256": (
                    EVALUATOR_IDENTITY.evaluator_context_sha256
                ),
            },
            "reflection_insight_contract": (
                REFLECTION_INSIGHT_CONTRACT.to_record()
            ),
            "g2_prequeue_gate": {
                "policy_id": g2_gate_policy.policy_id,
                "policy_version": g2_gate_policy.policy_version,
                "policy_identity_sha256": g2_gate_policy.identity_sha256,
                "expected_request_count": 3,
                "release_semantics": "durable_all_or_none_before_provider_queue",
            },
            "held_out_transfer_adjudicator": {
                **_HELD_OUT_ADJUDICATOR_DEFINITION,
                "definition_sha256": HELD_OUT_ADJUDICATOR_SHA256,
            },
            "mutation_contracts": mutation_contracts,
        },
        "catalogs": {
            "g1_shape": {
                "catalog_id": shape.catalog_id,
                "catalog_version": shape.catalog_version,
                "definition_sha256": shape.catalog_definition_sha256,
                "bound_contract_identity_sha256": shape.identity_sha256,
                "option_count": len(shape.options),
            },
            "g1_trim": {
                "catalog_id": trim.catalog_id,
                "catalog_version": trim.catalog_version,
                "definition_sha256": trim.catalog_definition_sha256,
                "bound_contract_identity_sha256": trim.identity_sha256,
                "option_count": len(trim.options),
            },
            "g2_union": {
                "catalog_id": union.catalog_id,
                "catalog_version": union.catalog_version,
                "definition_sha256": union.catalog_definition_sha256,
                "bound_contract_identity_sha256": union.identity_sha256,
                "option_count": len(union.options),
            },
        },
        "evaluator_route": {
            "python_executable": str(
                _invoked_absolute_path(default_settings.python_executable)
            ),
            "evaluator_script": str(default_settings.evaluator_script.resolve()),
            "dataset_arrow": str(default_settings.dataset_arrow.resolve()),
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "cfd_receipt_root": str(output_dir / "cfd_receipts"),
            "work_root": str(Path("/tmp") / "agent_evolve_airfoil_v7" / run_id),
            "cpu_set": default_settings.cpu_set,
            "mpi_cores": default_settings.mpi_cores,
            "timeout_seconds": default_settings.timeout_seconds,
            "retry_policy": "none_one_subprocess_per_unique_engine_cache_miss",
        },
        "resource_lease": _resource_lease_manifest_record(
            phase="provider_evolution"
        ),
        "prompt_readiness": dict(prompt_readiness),
        "durability": {
            "manifest": "atomic_write_fsync_directory",
            "json_documents": "atomic_write_fsync_directory",
            "jsonl": "append_flush_fsync_before_return",
            "required_logs": [
                "candidate_receipts.jsonl",
                "provider_queue_outcomes.jsonl",
                "prompt_response_journal.jsonl",
                "source_verifications.jsonl",
                "traces.jsonl",
                "resource_lease_acquired.json",
                "resource_lease_released.json",
                "result.json",
                "summary.json",
                "finalized.json",
            ],
        },
    }


def _validate_run_target(run_id: str, output_dir: Path) -> tuple[str, Path]:
    if type(run_id) is not str or _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be one safe path component")
    resolved = output_dir.expanduser().resolve()
    if resolved.name != run_id:
        raise ValueError("output_dir basename must equal run_id")
    return run_id, resolved


def build_launch_manifest_record(
    *,
    run_id: str,
    output_dir: Path,
    qualification_result_path: Path,
    prompt_readiness: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build, but do not publish, one prospective launch commitment."""

    run_id, output_dir = _validate_run_target(run_id, output_dir)
    readiness = (
        materialize_prompt_readiness_sync()
        if prompt_readiness is None
        else dict(prompt_readiness)
    )
    launch = _launch_spec(
        run_id=run_id,
        output_dir=output_dir,
        prompt_readiness=readiness,
        seed_qualification=load_seed_qualification_binding(
            qualification_result_path
        ),
    )
    record: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "launch": launch,
        "source_snapshot": source_snapshot(),
    }
    record["manifest_sha256"] = _sha256_record(record, domain=MANIFEST_FRAMING)
    return record


def write_launch_manifest(
    path: Path,
    *,
    run_id: str,
    output_dir: Path,
    qualification_result_path: Path,
) -> dict[str, object]:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = build_launch_manifest_record(
        run_id=run_id,
        output_dir=output_dir,
        qualification_result_path=qualification_result_path,
    )
    write_json_atomic(path, record)
    return record


@dataclass(frozen=True, slots=True)
class VerifiedLaunchManifest:
    path: Path
    record: dict[str, object]
    run_id: str
    output_dir: Path
    manifest_sha256: str
    source_sha256: str


def _validate_prompt_readiness(value: object) -> None:
    if type(value) is not dict:
        raise TypeError("prompt_readiness must be an object")
    calls = value.get("calls")
    if (
        value.get("logical_call_count") != 7
        or value.get("provider_io_performed") is not False
        or value.get("cfd_calls") != 0
        or value.get("structured_output_budget_policy")
        != structured_output_budget_policy_record()
        or value.get("optimization_semantics")
        != _optimization_semantics_binding()
        or value.get("treatment_compliance_policy")
        != _treatment_compliance_policy_binding()
        or value.get("all_caps_plausible") is not True
        or type(calls) is not list
        or len(calls) != 7
    ):
        raise RuntimeError("prompt readiness is incomplete")
    if [row.get("ordinal") for row in calls if type(row) is dict] != list(
        range(1, 8)
    ):
        raise RuntimeError("prompt readiness ordinals are not exactly 1..7")
    expected_stages = [
        "g1_diagnostic_1",
        "g1_diagnostic_2",
        "g1_reflection_1",
        "g1_reflection_2",
        "g2_held_out_1",
        "g2_held_out_2",
        "g2_held_out_3",
    ]
    if [row.get("stage") for row in calls if type(row) is dict] != expected_stages:
        raise RuntimeError("prompt readiness stages violate the 2+2+3 topology")
    route_binding = _streamlake_route_snapshot_binding()
    pricing_snapshot = route_binding["pricing_snapshot"]
    capability_snapshot = route_binding["capability_snapshot"]
    selected_route = route_binding["selected_route"]
    assert type(pricing_snapshot) is dict
    assert type(capability_snapshot) is dict
    assert type(selected_route) is dict
    for row in calls:
        if type(row) is not dict:
            raise RuntimeError("prompt readiness row drifted")
        kind = row.get("kind")
        try:
            request_kind = StructuredOutputRequestKind(str(kind))
        except ValueError as exc:
            raise RuntimeError("prompt readiness request kind drifted") from exc
        operation = (
            "typed_mutation"
            if request_kind is StructuredOutputRequestKind.PROPOSAL
            else "extract_insights"
        )
        expected_output_tokens = resolve_structured_output_budget(
            STRUCTURED_OUTPUT_BUDGET_POLICY,
            request_kind=request_kind,
            operation=operation,
        )
        expected_min_insights = (
            1
            if request_kind is StructuredOutputRequestKind.REFLECTION
            else None
        )
        expected_max_insights = expected_min_insights
        if (
            row.get("cap_plausible") is not True
            or row.get("request_kind") != request_kind.value
            or row.get("max_output_tokens") != expected_output_tokens
            or row.get("min_insights") != expected_min_insights
            or row.get("max_insights") != expected_max_insights
            or row.get("telemetry_input_token_cap") != MAX_INPUT_TOKENS
            or row.get("provider_context_length")
            != selected_route["context_length"]
            or row.get("provider_max_completion_tokens")
            != selected_route["max_completion_tokens"]
            or row.get("pricing_snapshot_sha256")
            != pricing_snapshot["sha256"]
            or row.get("capability_snapshot_sha256")
            != capability_snapshot["sha256"]
            or row.get("optimization_semantics_block_count") != 1
            or row.get("optimization_semantics_marker_count") != 1
            or row.get(
                "optimization_semantics_definition_sha256_count"
            )
            != 1
            or _LOWER_SHA256.fullmatch(str(row.get("prompt_sha256"))) is None
        ):
            raise RuntimeError("prompt readiness row drifted")


def verify_launch_manifest(
    path: Path,
    *,
    require_output_absent: bool,
    enforce_canonical_output: bool = True,
) -> VerifiedLaunchManifest:
    """Verify manifest, exact source bytes, policies, route, and run target."""

    resolved_path = path.expanduser().resolve(strict=True)
    content = resolved_path.read_bytes()
    try:
        record = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("launch manifest is not valid JSON") from exc
    if type(record) is not dict:
        raise TypeError("launch manifest root must be an object")
    claimed_hash = record.get("manifest_sha256")
    unsigned = dict(record)
    unsigned.pop("manifest_sha256", None)
    observed_hash = _sha256_record(unsigned, domain=MANIFEST_FRAMING)
    if claimed_hash != observed_hash:
        raise RuntimeError("launch manifest self-hash mismatch")
    if (
        record.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or record.get("kind") != MANIFEST_KIND
    ):
        raise RuntimeError("launch manifest kind/schema mismatch")
    launch = record.get("launch")
    if type(launch) is not dict:
        raise TypeError("launch specification must be an object")
    run_id, output_dir = _validate_run_target(
        launch.get("run_id"),
        Path(str(launch.get("output_dir"))),
    )
    if (
        enforce_canonical_output
        and output_dir.parent != DEFAULT_PROVIDER_RUN_ROOT.resolve()
    ):
        raise RuntimeError("live output directory is outside the canonical log root")
    if require_output_absent and output_dir.exists():
        raise FileExistsError(output_dir)
    readiness = launch.get("prompt_readiness")
    _validate_prompt_readiness(readiness)
    seed_binding = launch.get("seed_qualification")
    if type(seed_binding) is not dict:
        raise TypeError("provider launch lacks a seed qualification binding")
    qualification_path = seed_binding.get("qualification_result_path")
    if type(qualification_path) is not str or not qualification_path:
        raise RuntimeError("seed qualification result path is malformed")
    expected_launch = _launch_spec(
        run_id=run_id,
        output_dir=output_dir,
        prompt_readiness=readiness,
        seed_qualification=load_seed_qualification_binding(
            Path(qualification_path)
        ),
    )
    if launch != expected_launch:
        raise RuntimeError("launch policies, parents, route, or contracts drifted")
    observed_source = source_snapshot()
    if record.get("source_snapshot") != observed_source:
        raise RuntimeError("launch source snapshot drifted")
    return VerifiedLaunchManifest(
        path=resolved_path,
        record=record,
        run_id=run_id,
        output_dir=output_dir,
        manifest_sha256=observed_hash,
        source_sha256=str(observed_source["sha256"]),
    )


def reverify_launch_source(verified: VerifiedLaunchManifest) -> dict[str, object]:
    """Cheap fail-closed replay used immediately before every provider call."""

    current = verify_launch_manifest(
        verified.path,
        require_output_absent=False,
        enforce_canonical_output=False,
    )
    if (
        current.manifest_sha256 != verified.manifest_sha256
        or current.source_sha256 != verified.source_sha256
        or current.output_dir != verified.output_dir
    ):
        raise RuntimeError("verified launch identity changed during execution")
    return {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": current.manifest_sha256,
        "source_sha256": current.source_sha256,
    }


def _telemetry_record(value: AgenticCallTelemetry) -> dict[str, object]:
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


def _variation_content(result: VariationGenerationResult) -> dict[str, object]:
    draft = result.draft
    if type(draft) is not FiniteVariationSelectionDraft:
        raise TypeError("Airfoil-v7 live route accepts only finite selections")
    return {
        "kind": "finite_variation_selection",
        "option_id": draft.option_id,
        "option_identity_sha256": draft.option_identity_sha256,
        "contract_identity_sha256": draft.contract_identity_sha256,
        "design_rationale": draft.design_rationale,
        "claimed_insight_ids": list(draft.claimed_insight_ids),
    }


def _insight_content(insight: InsightDraft) -> dict[str, object]:
    return {
        "claim": insight.claim,
        "trigger": insight.trigger,
        "mechanism": insight.mechanism,
        "affected_paths": list(insight.affected_paths),
        "evidence_summary": insight.evidence_summary,
        "confidence": insight.confidence,
        "evidence_contrast_ids": list(insight.evidence_contrast_ids),
        "intervention": insight.intervention_record(),
    }


class LiveStackLike(Protocol):
    runner: Any
    generator: AgenticGenerator
    telemetry_policy: AgenticTelemetryPolicy


class G2PrequeueGateError(RuntimeError):
    """The three-request held-out batch failed a closed prequeue invariant."""

    def __init__(self, reason_code: str) -> None:
        if (
            type(reason_code) is not str
            or re.fullmatch(r"[a-z][a-z0-9_]{0,95}", reason_code) is None
        ):
            raise ValueError("reason_code must be a closed lowercase token")
        self.reason_code = reason_code
        super().__init__(f"G2 prequeue gate rejected the batch: {reason_code}")


def _utf8_digest(value: str) -> tuple[bytes, str, int]:
    if type(value) is not str:
        raise G2PrequeueGateError("non_string_prompt_component")
    encoded = value.encode("utf-8", errors="strict")
    return encoded, hashlib.sha256(encoded).hexdigest(), len(encoded)


def _card_schema(value: object) -> object:
    if type(value) is dict:
        return {
            key: _card_schema(item)
            for key, item in sorted(value.items())
        }
    if type(value) is list:
        element_schemas = {
            json.dumps(
                _card_schema(item),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            for item in value
        }
        return {"array_element_schemas": sorted(element_schemas)}
    if type(value) is str:
        return "string"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    raise G2PrequeueGateError("unsupported_card_value_type")


def _walk_card_strings(value: object) -> tuple[str, ...]:
    if type(value) is str:
        return (value,)
    if type(value) is list:
        return tuple(
            text for item in value for text in _walk_card_strings(item)
        )
    if type(value) is dict:
        return tuple(
            text for item in value.values() for text in _walk_card_strings(item)
        )
    return ()


def _contains_reserved_card_term(value: str) -> bool:
    normalized = re.sub(r"[_-]+", " ", value.casefold())
    return any(
        re.search(
            rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
            normalized,
        )
        is not None
        for term in _RESERVED_CARD_TERMS
    )


def _validate_blinded_card(record: object) -> dict[str, object]:
    if type(record) is not dict or set(record) != _BLINDED_CARD_KEYS:
        raise G2PrequeueGateError("blinded_card_schema_keys_differ")
    text_fields = (
        "action_template",
        "claim",
        "falsification_condition",
        "insight_id",
        "mechanism",
        "trigger",
    )
    if any(
        type(record[field]) is not str
        or not record[field]
        or record[field] != record[field].strip()
        for field in text_fields
    ):
        raise G2PrequeueGateError("blinded_card_text_field_invalid")
    affected_paths = record["affected_paths"]
    families = record["recommended_option_families"]
    option_ids = record["recommended_option_ids"]
    effects = record["effect_predictions"]
    if (
        type(affected_paths) is not list
        or not affected_paths
        or any(type(item) is not str or not item for item in affected_paths)
        or type(families) is not list
        or not families
        or any(type(item) is not str or not item for item in families)
        or type(option_ids) is not list
        or len(option_ids) != 1
        or type(option_ids[0]) is not str
        or not option_ids[0]
        or type(effects) is not list
        or not effects
    ):
        raise G2PrequeueGateError("blinded_card_collection_field_invalid")
    for effect in effects:
        if (
            type(effect) is not dict
            or set(effect) != {"direction", "metric_id"}
            or type(effect["direction"]) is not str
            or type(effect["metric_id"]) is not str
            or not effect["direction"]
            or not effect["metric_id"]
        ):
            raise G2PrequeueGateError("blinded_card_effect_schema_invalid")
    if any(_contains_reserved_card_term(text) for text in _walk_card_strings(record)):
        raise G2PrequeueGateError("blinded_card_contains_reserved_term")
    return dict(record)


@dataclass(frozen=True, slots=True)
class _G2PromptEnvelope:
    request: VariationGenerationRequest
    raw_prompt_sha256: str
    raw_prompt_utf8_bytes: int
    masked_prompt: bytes
    masked_prompt_sha256: str
    masked_prompt_utf8_bytes: int
    payload: dict[str, object]
    payload_sha256: str
    payload_utf8_bytes: int
    payload_schema_sha256: str
    action_palette_sha256: str
    action_palette_option_count: int
    preflight: dict[str, object]


def _g2_prompt_envelope(
    request: VariationGenerationRequest,
) -> _G2PromptEnvelope:
    if type(request) is not VariationGenerationRequest:
        raise G2PrequeueGateError("non_variation_request")
    contract = request.finite_variation_contract
    if contract is None or contract.catalog_id != "airfoil_v7_union":
        raise G2PrequeueGateError("wrong_finite_contract")
    prompt = request.prompt
    if (
        prompt.count(MEMORY_CARD_BEGIN) != 1
        or prompt.count(MEMORY_CARD_END) != 1
    ):
        raise G2PrequeueGateError("memory_card_delimiter_count_invalid")
    start = prompt.index(MEMORY_CARD_BEGIN) + len(MEMORY_CARD_BEGIN)
    end = prompt.index(MEMORY_CARD_END)
    if end <= start:
        raise G2PrequeueGateError("memory_card_delimiter_order_invalid")
    framed_payload = prompt[start:end]
    payload_text = framed_payload.strip()
    if framed_payload != f"\n{payload_text}\n" or (
        MEMORY_CARD_BEGIN in payload_text or MEMORY_CARD_END in payload_text
    ):
        raise G2PrequeueGateError("memory_card_payload_framing_invalid")
    try:
        payload_value = json.loads(payload_text)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise G2PrequeueGateError("memory_card_payload_json_invalid") from exc
    payload = _validate_blinded_card(payload_value)
    try:
        exact_option = contract.resolve(payload["recommended_option_ids"][0])
    except (TypeError, ValueError) as exc:
        raise G2PrequeueGateError(
            "recommended_option_id_outside_contract"
        ) from exc
    if exact_option.family not in payload["recommended_option_families"]:
        raise G2PrequeueGateError("recommended_option_family_mismatch")
    canonical_payload = _canonical_bytes(payload)
    try:
        raw_payload = payload_text.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise G2PrequeueGateError("memory_card_payload_not_canonical_ascii") from exc
    if raw_payload != canonical_payload:
        raise G2PrequeueGateError("memory_card_payload_not_canonical_json")
    raw_prompt, raw_hash, raw_bytes = _utf8_digest(prompt)
    del raw_prompt
    masked_prompt_text = mask_memory_card(prompt)
    masked_prompt, masked_hash, masked_bytes = _utf8_digest(masked_prompt_text)
    schema = _card_schema(payload)
    schema_hash = _sha256_record(
        schema,
        domain=b"agent-evolve:airfoil-v7-blinded-card-schema:v2\x00",
    )
    palette = contract.prompt_records()
    palette_hash = _sha256_record(
        palette,
        domain=b"agent-evolve:airfoil-v7-action-palette:v1\x00",
    )
    try:
        preflight = prompt_preflight(
            prompt,
            max_output_tokens=request.max_output_tokens,
            request_kind=StructuredOutputRequestKind.PROPOSAL,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise G2PrequeueGateError("prompt_preflight_failed") from exc
    return _G2PromptEnvelope(
        request=request,
        raw_prompt_sha256=raw_hash,
        raw_prompt_utf8_bytes=raw_bytes,
        masked_prompt=masked_prompt,
        masked_prompt_sha256=masked_hash,
        masked_prompt_utf8_bytes=masked_bytes,
        payload=payload,
        payload_sha256=hashlib.sha256(raw_payload).hexdigest(),
        payload_utf8_bytes=len(raw_payload),
        payload_schema_sha256=schema_hash,
        action_palette_sha256=palette_hash,
        action_palette_option_count=len(palette),
        preflight=preflight,
    )


@dataclass(frozen=True, slots=True)
class AirfoilV7G2PrequeueGatePolicy:
    """Pure, injectable validation policy for the preregistered G2 batch.

    Buffering, credential deferral, and concurrent release belong to the live
    generator.  Airfoil's delimiter, card vocabulary, three-arm assignment,
    and finite catalog assumptions live here.  A future generic batch barrier
    can therefore retain the orchestration and inject a different policy.
    """

    policy_id: str = "airfoil_v7_blinded_g2_prequeue_gate"
    policy_version: int = 2

    def __post_init__(self) -> None:
        if self.policy_id != "airfoil_v7_blinded_g2_prequeue_gate":
            raise ValueError("G2 gate policy ID drifted")
        if self.policy_version != 2:
            raise ValueError("G2 gate policy version drifted")

    @property
    def identity_sha256(self) -> str:
        self.__post_init__()
        definition = {
            "schema_version": 2,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "expected_request_count": 3,
            "catalog_id": "airfoil_v7_union",
            "memory_card_begin": MEMORY_CARD_BEGIN,
            "memory_card_end": MEMORY_CARD_END,
            "blinded_card_keys": sorted(_BLINDED_CARD_KEYS),
            "reserved_card_terms": list(_RESERVED_CARD_TERMS),
            "requires_exact_masked_prompt_bytes": True,
            "requires_exact_contract_and_palette": True,
            "requires_distinct_assignment_bound_insight_ids": True,
            "requires_singleton_contract_bound_option_id": True,
            "requires_three_distinct_option_ids": True,
            "prospective_sham_option_id": SHAM_OPTION_ID,
        }
        return _sha256_record(
            definition,
            domain=b"agent-evolve:airfoil-v7-g2-gate-policy:v2\x00",
        )

    def prepare(self, request: VariationGenerationRequest) -> _G2PromptEnvelope:
        self.__post_init__()
        return _g2_prompt_envelope(request)

    def validate_batch(
        self,
        envelopes: tuple[_G2PromptEnvelope, ...],
        *,
        assignment_commitment: HeldOutASNAssignmentCommitment,
    ) -> dict[str, object]:
        """Return the complete accepted receipt or raise one closed rejection."""

        self.__post_init__()
        if len(envelopes) != 3:
            raise G2PrequeueGateError("request_count_not_three")
        requests = tuple(envelope.request for envelope in envelopes)
        if len({request.call_id.value for request in requests}) != 3:
            raise G2PrequeueGateError("duplicate_call_id")
        if len({request.operation for request in requests}) != 1:
            raise G2PrequeueGateError("operation_mismatch")
        if len({request.candidate_model for request in requests}) != 1:
            raise G2PrequeueGateError("candidate_model_mismatch")
        if len({request.max_output_tokens for request in requests}) != 1 or len(
            {request.temperature for request in requests}
        ) != 1:
            raise G2PrequeueGateError("generation_parameter_mismatch")
        contracts = tuple(request.finite_variation_contract for request in requests)
        if any(contract is None for contract in contracts):
            raise G2PrequeueGateError("finite_contract_missing")
        contract_hashes = {
            contract.identity_sha256 for contract in contracts if contract is not None
        }
        if len(contract_hashes) != 1:
            raise G2PrequeueGateError("finite_contract_mismatch")
        palette_hashes = {item.action_palette_sha256 for item in envelopes}
        palette_sizes = {item.action_palette_option_count for item in envelopes}
        if len(palette_hashes) != 1 or len(palette_sizes) != 1:
            raise G2PrequeueGateError("action_palette_mismatch")
        if len({item.masked_prompt for item in envelopes}) != 1:
            raise G2PrequeueGateError("masked_prompt_bytes_mismatch")
        if len({item.payload_schema_sha256 for item in envelopes}) != 1:
            raise G2PrequeueGateError("blinded_card_schema_mismatch")
        insight_ids = tuple(str(item.payload["insight_id"]) for item in envelopes)
        if len(set(insight_ids)) != 3:
            raise G2PrequeueGateError("insight_ids_not_distinct")
        exact_option_ids = tuple(
            str(item.payload["recommended_option_ids"][0])
            for item in envelopes
        )
        if len(set(exact_option_ids)) != 3:
            raise G2PrequeueGateError("recommended_option_ids_not_distinct")
        commitment = assignment_commitment
        if type(commitment) is not HeldOutASNAssignmentCommitment:
            raise G2PrequeueGateError("assignment_commitment_unavailable")
        try:
            commitment.__post_init__()
        except (TypeError, ValueError) as exc:
            raise G2PrequeueGateError("assignment_commitment_invalid") from exc
        role_by_insight_id = {
            commitment.adaptive_reference.insight_id.value: "adaptive",
            commitment.score_swapped_reference.insight_id.value: "score_swapped",
            commitment.sham_reference.insight_id.value: "sham",
        }
        if len(role_by_insight_id) != 3 or set(insight_ids) != set(
            role_by_insight_id
        ):
            raise G2PrequeueGateError("prompt_assignment_reference_mismatch")
        option_id_by_role = {
            role_by_insight_id[str(item.payload["insight_id"])]: str(
                item.payload["recommended_option_ids"][0]
            )
            for item in envelopes
        }
        if option_id_by_role.get("sham") != SHAM_OPTION_ID:
            raise G2PrequeueGateError("prospective_sham_option_id_mismatch")
        ordered = tuple(
            sorted(envelopes, key=lambda item: item.request.call_id.value)
        )
        rows = [
            {
                "call_id": item.request.call_id.value,
                "insight_id": item.payload["insight_id"],
                "assignment_role": role_by_insight_id[
                    str(item.payload["insight_id"])
                ],
                "recommended_option_id": item.payload[
                    "recommended_option_ids"
                ][0],
                "raw_prompt_sha256": item.raw_prompt_sha256,
                "raw_prompt_utf8_bytes": item.raw_prompt_utf8_bytes,
                "masked_prompt_sha256": item.masked_prompt_sha256,
                "masked_prompt_utf8_bytes": item.masked_prompt_utf8_bytes,
                "payload_sha256": item.payload_sha256,
                "payload_utf8_bytes": item.payload_utf8_bytes,
                "payload_schema_sha256": item.payload_schema_sha256,
            }
            for item in ordered
        ]
        gate_record: dict[str, object] = {
            "schema_version": 2,
            "record_type": _G2_GATE_RECORD_TYPE,
            "status": "accepted",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_identity_sha256": self.identity_sha256,
            "request_count": 3,
            "contract_identity_sha256": next(iter(contract_hashes)),
            "action_palette_sha256": next(iter(palette_hashes)),
            "action_palette_option_count": next(iter(palette_sizes)),
            "masked_prompt_sha256": ordered[0].masked_prompt_sha256,
            "masked_prompt_utf8_bytes": ordered[0].masked_prompt_utf8_bytes,
            "assignment_sha256": commitment.assignment_sha256,
            "assignment_commitment": commitment.to_record(),
            "requests": rows,
        }
        gate_record["batch_gate_sha256"] = _sha256_record(
            gate_record,
            domain=b"agent-evolve:airfoil-v7-g2-prequeue-gate:v2\x00",
        )
        return gate_record


class G2PrequeueGatePolicy(Protocol):
    """Promotion seam for a benchmark-owned blinded batch admission policy."""

    policy_id: str
    policy_version: int

    @property
    def identity_sha256(self) -> str: ...

    def prepare(self, request: VariationGenerationRequest) -> _G2PromptEnvelope: ...

    def validate_batch(
        self,
        envelopes: tuple[_G2PromptEnvelope, ...],
        *,
        assignment_commitment: HeldOutASNAssignmentCommitment,
    ) -> dict[str, object]: ...


class DeferredJournaledLiveGenerator:
    """Start the v6 queue lazily after seeds and journal every call boundary."""

    def __init__(
        self,
        *,
        credential_loader: Callable[[], str],
        stack_factory: Callable[[str], LiveStackLike],
        pre_provider_verifier: Callable[[str], Mapping[str, object]],
        journal: DurableJsonlWriter,
        expected_telemetry_policy: Mapping[str, object],
        expected_telemetry_policy_sha256: str,
        g2_gate_policy: G2PrequeueGatePolicy | None = None,
    ) -> None:
        self._credential_loader = credential_loader
        self._stack_factory = stack_factory
        self._pre_provider_verifier = pre_provider_verifier
        self._journal = journal
        if type(expected_telemetry_policy) is not dict:
            expected_telemetry_policy = dict(expected_telemetry_policy)
        if (
            type(expected_telemetry_policy_sha256) is not str
            or _LOWER_SHA256.fullmatch(expected_telemetry_policy_sha256) is None
        ):
            raise ValueError(
                "expected telemetry policy SHA must be a lowercase SHA-256"
            )
        self._expected_telemetry_policy = dict(expected_telemetry_policy)
        self._expected_telemetry_policy_sha256 = (
            expected_telemetry_policy_sha256
        )
        self._g2_gate_policy = (
            AirfoilV7G2PrequeueGatePolicy()
            if g2_gate_policy is None
            else g2_gate_policy
        )
        if (
            not callable(getattr(self._g2_gate_policy, "prepare", None))
            or not callable(getattr(self._g2_gate_policy, "validate_batch", None))
            or type(getattr(self._g2_gate_policy, "policy_id", None)) is not str
            or type(getattr(self._g2_gate_policy, "policy_version", None)) is not int
            or _LOWER_SHA256.fullmatch(
                str(getattr(self._g2_gate_policy, "identity_sha256", ""))
            )
            is None
        ):
            raise TypeError("g2_gate_policy violates the batch-policy protocol")
        self._stack: LiveStackLike | None = None
        self._start_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._g2_gate_lock = asyncio.Lock()
        self._g2_gate_event = asyncio.Event()
        self._g2_envelopes: dict[str, _G2PromptEnvelope] = {}
        self._g2_gate_failure: G2PrequeueGateError | None = None
        self._g2_gate_released = False
        self._accepted_g2_gate_record: dict[str, object] | None = None
        self._assignment_commitment_supplier: (
            Callable[[], HeldOutASNAssignmentCommitment | None] | None
        ) = None
        self._closed = False
        self._logical_calls = 0
        self._proposal_calls = 0
        self._reflection_calls = 0
        self._accepted_responses: list[dict[str, object]] = []
        self.credentials_read = False

    @property
    def logical_calls(self) -> int:
        return self._logical_calls

    @property
    def proposal_calls(self) -> int:
        return self._proposal_calls

    @property
    def reflection_calls(self) -> int:
        return self._reflection_calls

    @property
    def accepted_response_records(self) -> tuple[dict[str, object], ...]:
        return tuple(
            dict(row)
            for row in sorted(
                self._accepted_responses,
                key=lambda item: int(item["logical_call_ordinal"]),
            )
        )

    @property
    def accepted_g2_gate_record(self) -> dict[str, object] | None:
        """Return a detached copy of the durable accepted gate receipt."""

        if self._accepted_g2_gate_record is None:
            return None
        return json.loads(_canonical_bytes(self._accepted_g2_gate_record))

    def bind_assignment_commitment_supplier(
        self,
        supplier: Callable[[], HeldOutASNAssignmentCommitment | None],
    ) -> None:
        """Bind the planner handoff once, without forcing an eager assignment.

        The assignment does not exist until G2 is planned, so the gate receives
        a lazy, read-only supplier.  Binding is intentionally write-once and
        must happen before any G2 request can enter the barrier.
        """

        if not callable(supplier):
            raise TypeError("assignment commitment supplier must be callable")
        if self._assignment_commitment_supplier is not None:
            raise RuntimeError("assignment commitment supplier was already bound")
        if self._g2_envelopes or self._g2_gate_released:
            raise RuntimeError("cannot bind assignment after G2 gate admission")
        self._assignment_commitment_supplier = supplier

    async def __aenter__(self) -> "DeferredJournaledLiveGenerator":
        if self._closed:
            raise RuntimeError("deferred live generator is already closed")
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self._closed = True
        if self._stack is not None:
            await self._stack.runner.__aexit__(None, None, None)

    async def _ensure_stack(self) -> LiveStackLike:
        async with self._start_lock:
            if self._stack is not None:
                return self._stack
            self._pre_provider_verifier("pre_live_credential_load")
            api_key = self._credential_loader()
            self.credentials_read = True
            if type(api_key) is not str or not api_key.strip():
                raise RuntimeError("OPENROUTER_API_KEY is unavailable")
            stack = self._stack_factory(api_key)
            policy = getattr(stack, "telemetry_policy", None)
            if type(policy) is not AgenticTelemetryPolicy:
                raise RuntimeError(
                    "live stack lacks an exact AgenticTelemetryPolicy"
                )
            AgenticTelemetryPolicy.__post_init__(policy)
            observed_policy = policy.to_trace_record()
            observed_sha256 = policy.policy_sha256
            if (
                observed_policy != self._expected_telemetry_policy
                or observed_sha256
                != self._expected_telemetry_policy_sha256
            ):
                raise RuntimeError(
                    "live stack telemetry policy drifted from launch manifest"
                )
            self._journal.write(
                {
                    "schema_version": 1,
                    "record_type": "live_stack_admission",
                    "status": "accepted",
                    "telemetry_policy": observed_policy,
                    "telemetry_policy_sha256": observed_sha256,
                    "provider_dispatch_performed": False,
                }
            )
            await stack.runner.__aenter__()
            self._stack = stack
            return stack

    def _assignment_commitment(self) -> HeldOutASNAssignmentCommitment:
        supplier = self._assignment_commitment_supplier
        if supplier is None:
            raise G2PrequeueGateError("assignment_supplier_unbound")
        commitment = supplier()
        if type(commitment) is not HeldOutASNAssignmentCommitment:
            raise G2PrequeueGateError("assignment_commitment_unavailable")
        try:
            commitment.__post_init__()
        except (TypeError, ValueError) as exc:
            raise G2PrequeueGateError("assignment_commitment_invalid") from exc
        return commitment

    def _validate_g2_batch(
        self,
        envelopes: tuple[_G2PromptEnvelope, ...],
    ) -> tuple[HeldOutASNAssignmentCommitment, dict[str, object]]:
        commitment = self._assignment_commitment()
        gate_record = self._g2_gate_policy.validate_batch(
            envelopes,
            assignment_commitment=commitment,
        )
        if (
            type(gate_record) is not dict
            or gate_record.get("policy_id") != self._g2_gate_policy.policy_id
            or gate_record.get("policy_version")
            != self._g2_gate_policy.policy_version
            or gate_record.get("policy_identity_sha256")
            != self._g2_gate_policy.identity_sha256
        ):
            raise G2PrequeueGateError("policy_receipt_identity_mismatch")
        return commitment, gate_record

    def _g2_rejection_record(
        self,
        failure: G2PrequeueGateError,
        *,
        request_count: int,
    ) -> dict[str, object]:
        return {
            "schema_version": 2,
            "record_type": _G2_GATE_RECORD_TYPE,
            "status": "rejected",
            "reason_code": failure.reason_code,
            "request_count": request_count,
            "policy_id": self._g2_gate_policy.policy_id,
            "policy_version": self._g2_gate_policy.policy_version,
            "policy_identity_sha256": self._g2_gate_policy.identity_sha256,
            "provider_dispatch_performed": False,
        }

    async def _reject_g2_gate(self, failure: G2PrequeueGateError) -> None:
        async with self._g2_gate_lock:
            if self._g2_gate_released or self._g2_gate_failure is not None:
                return
            self._g2_gate_failure = failure
            self._journal.write(
                self._g2_rejection_record(
                    failure,
                    request_count=len(self._g2_envelopes),
                )
            )
            self._g2_gate_event.set()

    async def _await_g2_prequeue_gate(
        self,
        request: VariationGenerationRequest,
    ) -> None:
        try:
            envelope = self._g2_gate_policy.prepare(request)
        except G2PrequeueGateError as exc:
            await self._reject_g2_gate(exc)
            raise
        async with self._g2_gate_lock:
            if self._g2_gate_failure is not None:
                raise self._g2_gate_failure
            if self._g2_gate_released:
                failure = G2PrequeueGateError("request_after_batch_release")
                self._g2_gate_failure = failure
                raise failure
            call_id = request.call_id.value
            if call_id in self._g2_envelopes:
                failure = G2PrequeueGateError("duplicate_call_id")
                self._g2_gate_failure = failure
                self._journal.write(
                    self._g2_rejection_record(
                        failure,
                        request_count=len(self._g2_envelopes),
                    )
                )
                self._g2_gate_event.set()
                raise failure
            if len(self._g2_envelopes) >= 3:
                failure = G2PrequeueGateError("request_count_exceeded")
                self._g2_gate_failure = failure
                raise failure
            self._g2_envelopes[call_id] = envelope
            if len(self._g2_envelopes) == 3:
                try:
                    _, gate_record = self._validate_g2_batch(
                        tuple(self._g2_envelopes.values())
                    )
                    # Durable publication is deliberately before event release.
                    self._journal.write(gate_record)
                    self._accepted_g2_gate_record = gate_record
                except G2PrequeueGateError as exc:
                    self._g2_gate_failure = exc
                    self._journal.write(
                        self._g2_rejection_record(exc, request_count=3)
                    )
                except BaseException as exc:
                    failure = G2PrequeueGateError("gate_journal_publication_failed")
                    self._g2_gate_failure = failure
                    self._g2_gate_event.set()
                    raise failure from exc
                else:
                    self._g2_gate_released = True
                self._g2_gate_event.set()
        try:
            await self._g2_gate_event.wait()
        except asyncio.CancelledError:
            await self._reject_g2_gate(
                G2PrequeueGateError("batch_participant_cancelled")
            )
            raise
        if self._g2_gate_failure is not None:
            raise self._g2_gate_failure
        if not self._g2_gate_released:
            raise G2PrequeueGateError("batch_release_state_invalid")

    async def _admit_request(
        self,
        request: VariationGenerationRequest | ReflectionGenerationRequest,
        *,
        kind: str,
    ) -> tuple[int, dict[str, object]]:
        if self._closed:
            raise RuntimeError("provider request attempted after queue close")
        if kind == "reflection" and (
            type(request) is not ReflectionGenerationRequest
            or request.min_insights != 1
            or request.max_insights != 1
            or len(request.available_contrast_ids) != 1
        ):
            raise RuntimeError(
                "each G1 reflection shard must bind one contrast and one insight"
            )
        preflight = prompt_preflight(
            request.prompt,
            max_output_tokens=request.max_output_tokens,
            request_kind=(
                StructuredOutputRequestKind.REFLECTION
                if kind == "reflection"
                else StructuredOutputRequestKind.PROPOSAL
            ),
        )
        async with self._state_lock:
            next_call = self._logical_calls + 1
            if next_call > OPTIMIZER_BUDGET.max_logical_llm_calls:
                raise RuntimeError("live route attempted more than seven logical calls")
            if kind == "reflection":
                if self._proposal_calls != 2 or self._reflection_calls not in {0, 1}:
                    raise RuntimeError("reflection escaped the exact 2+2+3 order")
                self._reflection_calls += 1
            else:
                contract = request.finite_variation_contract
                if contract is None:
                    raise RuntimeError("live proposal lacks a finite contract")
                if self._proposal_calls < 2:
                    if self._reflection_calls != 0:
                        raise RuntimeError("G1 proposal followed a reflection shard")
                    if contract.catalog_id not in {
                        "airfoil_v7_shape",
                        "airfoil_v7_trim",
                    }:
                        raise RuntimeError("G1 proposal used the wrong catalog")
                else:
                    if self._reflection_calls != 2:
                        raise RuntimeError(
                            "G2 proposal attempted before both reflection shards"
                        )
                    if contract.catalog_id != "airfoil_v7_union":
                        raise RuntimeError("G2 proposal used the wrong catalog")
                self._proposal_calls += 1
            self._logical_calls = next_call
        verification = dict(
            self._pre_provider_verifier(f"pre_provider_call_{next_call}")
        )
        self._journal.write(
            {
                "schema_version": 1,
                "record_type": "request",
                "logical_call_ordinal": next_call,
                "kind": kind,
                "call_id": request.call_id.value,
                "operation": request.operation,
                "preflight": preflight,
                "source_verification": verification,
                "finite_contract_identity_sha256": (
                    request.finite_variation_contract.identity_sha256
                    if type(request) is VariationGenerationRequest
                    and request.finite_variation_contract is not None
                    else None
                ),
            }
        )
        return next_call, preflight

    async def propose(
        self,
        request: VariationGenerationRequest,
    ) -> VariationGenerationResult:
        contract = request.finite_variation_contract
        if contract is not None and contract.catalog_id == "airfoil_v7_union":
            await self._await_g2_prequeue_gate(request)
        ordinal, _ = await self._admit_request(request, kind="proposal")
        stack = await self._ensure_stack()
        try:
            result = await stack.generator.propose(request)
        except BaseException as exc:
            self._journal.write(
                {
                    "schema_version": 1,
                    "record_type": "response_failure",
                    "logical_call_ordinal": ordinal,
                    "kind": "proposal",
                    "call_id": request.call_id.value,
                    "failure_type": type(exc).__name__,
                }
            )
            raise
        content = _variation_content(result)
        telemetry = _telemetry_record(result.telemetry)
        self._journal.write(
            {
                "schema_version": 1,
                "record_type": "response",
                "logical_call_ordinal": ordinal,
                "kind": "proposal",
                "call_id": request.call_id.value,
                "content": content,
                "content_sha256": _sha256_record(
                    content,
                    domain=b"agent-evolve:airfoil-v7-provider-content:v1\x00",
                ),
                "telemetry": telemetry,
            }
        )
        self._accepted_responses.append(
            {
                "logical_call_ordinal": ordinal,
                "call_id": request.call_id.value,
                "kind": "proposal",
                "telemetry": telemetry,
            }
        )
        return result

    async def reflect(
        self,
        request: ReflectionGenerationRequest,
    ) -> ReflectionGenerationResult:
        ordinal, _ = await self._admit_request(request, kind="reflection")
        stack = await self._ensure_stack()
        try:
            result = await stack.generator.reflect(request)
        except BaseException as exc:
            self._journal.write(
                {
                    "schema_version": 1,
                    "record_type": "response_failure",
                    "logical_call_ordinal": ordinal,
                    "kind": "reflection",
                    "call_id": request.call_id.value,
                    "failure_type": type(exc).__name__,
                }
            )
            raise
        content = {"insights": [_insight_content(item) for item in result.insights]}
        telemetry = _telemetry_record(result.telemetry)
        self._journal.write(
            {
                "schema_version": 1,
                "record_type": "response",
                "logical_call_ordinal": ordinal,
                "kind": "reflection",
                "call_id": request.call_id.value,
                "content": content,
                "content_sha256": _sha256_record(
                    content,
                    domain=b"agent-evolve:airfoil-v7-provider-content:v1\x00",
                ),
                "telemetry": telemetry,
            }
        )
        self._accepted_responses.append(
            {
                "logical_call_ordinal": ordinal,
                "call_id": request.call_id.value,
                "kind": "reflection",
                "telemetry": telemetry,
            }
        )
        return result


def _provider_accounting_record(
    *,
    accepted_responses: tuple[dict[str, object], ...],
    queue_outcomes: tuple[dict[str, object], ...],
    expected_logical_calls: int,
    expected_accepted_responses: int,
    allowed_terminal_failures: int,
) -> dict[str, object]:
    """Fail closed on the complete accepted-response and raw-attempt envelope."""

    if (
        type(expected_logical_calls) is not int
        or expected_logical_calls <= 0
        or type(expected_accepted_responses) is not int
        or not 0 <= expected_accepted_responses <= expected_logical_calls
        or type(allowed_terminal_failures) is not int
        or allowed_terminal_failures < 0
        or expected_accepted_responses + allowed_terminal_failures
        != expected_logical_calls
        or len(accepted_responses) != expected_accepted_responses
        or len(queue_outcomes) != expected_logical_calls
    ):
        raise RuntimeError("provider response/queue terminal count mismatch")
    maximum_billable_attempt = _maximum_billable_cost_per_attempt()
    ordinals = [row.get("logical_call_ordinal") for row in accepted_responses]
    call_ids = [row.get("call_id") for row in accepted_responses]
    if ordinals != list(range(1, expected_accepted_responses + 1)) or len(
        set(call_ids)
    ) != expected_accepted_responses:
        raise RuntimeError("provider accepted-response ordinal/identity mismatch")
    queue_by_task: dict[str, dict[str, object]] = {}
    raw_attempts = 0
    terminal_failure_tasks: list[str] = []
    for outcome in queue_outcomes:
        if type(outcome) is not dict or outcome.get("schema_version") != 4:
            raise RuntimeError("provider queue outcome schema mismatch")
        task_id = outcome.get("task_id")
        attempts = outcome.get("attempts")
        status = outcome.get("status")
        response = outcome.get("response")
        if (
            type(task_id) is not str
            or task_id in queue_by_task
            or type(attempts) is not list
            or not 1 <= len(attempts) <= MAX_PROVIDER_ATTEMPTS
        ):
            raise RuntimeError("provider queue terminal outcome rejected")
        if status == "succeeded":
            if type(response) is not dict:
                raise RuntimeError("successful queue outcome lacks response")
        elif status in {"terminal_failure", "attempts_exhausted"}:
            if response is not None:
                raise RuntimeError("failed queue outcome unexpectedly has response")
            terminal_failure_tasks.append(task_id)
        else:
            raise RuntimeError("cancelled/unknown queue terminal outcome rejected")
        queue_by_task[task_id] = outcome
        raw_attempts += len(attempts)
    succeeded_tasks = {
        task_id
        for task_id, outcome in queue_by_task.items()
        if outcome["status"] == "succeeded"
    }
    if (
        succeeded_tasks != set(call_ids)
        or len(terminal_failure_tasks) != allowed_terminal_failures
    ):
        raise RuntimeError("provider queue success/failure tasks violate path state")

    total_cost = Decimal("0")
    max_input = max_output = max_reasoning = 0
    response_ids: set[str] = set()
    accepted_rows: list[dict[str, object]] = []
    for accepted in accepted_responses:
        telemetry = accepted.get("telemetry")
        if type(telemetry) is not dict:
            raise RuntimeError("accepted response lacks telemetry")
        call_id = str(accepted["call_id"])
        queue = queue_by_task[call_id]
        queue_response = queue["response"]
        assert type(queue_response) is dict
        response_id = telemetry.get("provider_response_id")
        finish_reason = telemetry.get("finish_reason")
        cost_text = telemetry.get("cost_usd")
        try:
            cost = Decimal(str(cost_text))
        except Exception as exc:
            raise RuntimeError("provider telemetry cost is missing/malformed") from exc
        if (
            telemetry.get("requested_model") != MODEL
            or telemetry.get("resolved_model")
            not in {MODEL, "deepseek/deepseek-v4-pro-20260423"}
            or telemetry.get("resolved_provider") != "StreamLake"
            or type(response_id) is not str
            or not response_id
            or response_id in response_ids
            or type(finish_reason) is not str
            or not finish_reason
            or type(telemetry.get("input_tokens")) is not int
            or not 0 <= telemetry["input_tokens"] <= MAX_INPUT_TOKENS
            or type(telemetry.get("output_tokens")) is not int
            or not 0 <= telemetry["output_tokens"] <= MAX_OUTPUT_TOKENS
            or type(telemetry.get("reasoning_tokens")) is not int
            or not 0
            <= telemetry["reasoning_tokens"]
            <= PROVIDER_REASONING_MAX_TOKENS
            or telemetry.get("attempt_count") != len(queue["attempts"])
            or cost < 0
            or cost > maximum_billable_attempt
        ):
            raise RuntimeError("accepted provider telemetry violates frozen gates")
        for field in (
            "requested_model",
            "resolved_model",
            "resolved_provider",
            "provider_response_id",
            "finish_reason",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "cost_usd",
            "latency_ns",
        ):
            if telemetry.get(field) != queue_response.get(field):
                raise RuntimeError("queue and accepted provider telemetry differ")
        response_ids.add(response_id)
        total_cost += cost
        max_input = max(max_input, int(telemetry["input_tokens"]))
        max_output = max(max_output, int(telemetry["output_tokens"]))
        max_reasoning = max(max_reasoning, int(telemetry["reasoning_tokens"]))
        accepted_rows.append(dict(accepted))
    accepted_cap = maximum_billable_attempt * Decimal(
        expected_accepted_responses
    )
    exposure_cap = (
        maximum_billable_attempt
        * Decimal(expected_logical_calls)
        * Decimal(MAX_PROVIDER_ATTEMPTS)
    )
    exposure = maximum_billable_attempt * Decimal(raw_attempts)
    if (
        total_cost > accepted_cap
        or raw_attempts > expected_logical_calls * MAX_PROVIDER_ATTEMPTS
        or exposure > exposure_cap
    ):
        raise RuntimeError("aggregate provider cost/attempt envelope exceeded")
    return {
        "schema_version": 1,
        "status": "pass",
        "accepted_response_count": len(accepted_rows),
        "terminal_failure_count": len(terminal_failure_tasks),
        "terminal_failure_task_ids": sorted(terminal_failure_tasks),
        "queue_terminal_outcome_count": len(queue_outcomes),
        "raw_attempt_count": raw_attempts,
        "raw_attempt_cap": expected_logical_calls * MAX_PROVIDER_ATTEMPTS,
        "accepted_cost_usd": str(total_cost),
        "accepted_cost_cap_usd": str(accepted_cap),
        "maximum_billable_attempt_cost_usd": str(maximum_billable_attempt),
        "potentially_billable_exposure_usd": str(exposure),
        "potentially_billable_exposure_cap_usd": str(exposure_cap),
        "maximum_input_tokens": max_input,
        "maximum_output_tokens": max_output,
        "maximum_reasoning_tokens": max_reasoning,
        "ceiling_semantics": "worst_case_gate_not_expected_usage",
        "responses": accepted_rows,
    }


def _metric_direction(
    *,
    parent_value: float,
    child_value: float,
    threshold: float,
) -> str:
    if parent_value - child_value >= threshold:
        return "decrease"
    if child_value - parent_value >= threshold:
        return "increase"
    return "unchanged"


def _is_pre_evaluation_treatment_rejection(outcome: Any) -> bool:
    """Recognize only authenticated expected no-yield treatment rejections."""

    if (
        outcome.call_failure_type != TreatmentComplianceRejected.__name__
        or outcome.failure_stage != "treatment_noncompliance"
        or outcome.candidate is not None
        or outcome.terminal_evaluation is not None
    ):
        return False
    receipt = outcome.treatment_admission_receipt
    if type(receipt) is not TreatmentAdmissionReceipt:
        return False
    TreatmentAdmissionReceipt.__post_init__(receipt)
    return not receipt.passed and receipt.evaluator_entered is False


def _physical_evaluation_identity(candidate: Any) -> str:
    detailed = candidate.detailed_evaluation
    if detailed is None:
        raise RuntimeError("Airfoil full-path candidate lacks detailed evaluation")
    return detailed.phenotype.identity_sha256


def _execution_path_record(
    result: OptimizerResult,
    *,
    generation_widths: list[int],
    early_stop_reason_code: str | None,
    early_stop_reason: str | None,
    accepted_provider_responses: int,
    terminal_provider_failures: int,
) -> dict[str, object]:
    if (
        type(accepted_provider_responses) is not int
        or accepted_provider_responses < 0
        or type(terminal_provider_failures) is not int
        or terminal_provider_failures < 0
    ):
        raise ValueError("provider terminal counts must be non-negative integers")
    receipts = tuple(
        receipt for receipt in result.feedback_receipts if receipt.generation == 1
    )
    if len(receipts) != 1:
        raise RuntimeError("execution path lacks one authenticated G1 feedback receipt")
    receipt = receipts[0]
    validate_generation_feedback_receipt(receipt)
    metadata = dict(receipt.result_metadata)
    feedback_status = metadata.get("status")
    if generation_widths == [2, 3]:
        if (
            feedback_status != "ready"
            or metadata.get("card_count") != "2"
            or receipt.used_logical_llm_calls != 2
            or early_stop_reason_code is not None
            or early_stop_reason is not None
            or accepted_provider_responses != 7
            or terminal_provider_failures != 0
        ):
            raise RuntimeError("full path disagrees with authenticated feedback state")
        generation_one = result.generation_receipts[0].slot_results
        generation_two = result.generation_receipts[1].slot_results
        if len(generation_one) != 2 or len(generation_two) != 3:
            raise RuntimeError("full path generation receipts changed width")
        for item in generation_one:
            outcome = item.outcome
            if outcome.failure_stage is not None or outcome.candidate is None:
                raise RuntimeError("full path diagnostic candidate did not evaluate")
            _physical_evaluation_identity(outcome.candidate)

        rejected_slots: list[str] = []
        for item in generation_two:
            outcome = item.outcome
            if _is_pre_evaluation_treatment_rejection(outcome):
                rejected_slots.append(item.slot.slot_id)
                continue
            if outcome.failure_stage is not None or outcome.candidate is None:
                raise RuntimeError(
                    "G2 failure is not an authenticated treatment noncompliance"
                )
            _physical_evaluation_identity(outcome.candidate)

        seed_identities = {
            _physical_evaluation_identity(receipt.candidate)
            for receipt in result.seed_receipts
        }
        if len(seed_identities) != 2:
            raise RuntimeError("full path requires two distinct evaluated seeds")
        child_identities = {
            _physical_evaluation_identity(outcome.candidate)
            for receipt in result.generation_receipts
            for item in receipt.slot_results
            for outcome in (item.outcome,)
            if outcome.candidate is not None
            and not _is_pre_evaluation_treatment_rejection(outcome)
        }
        expected_unique = len(seed_identities | child_identities)
        expected_delegated = len(child_identities - seed_identities)
        if not (4 <= expected_unique <= 7 and 2 <= expected_delegated <= 5):
            raise RuntimeError("full-path physical evaluation accounting is impossible")
        return {
            "schema_version": 1,
            "mode": "full",
            "feedback_status": feedback_status,
            "feedback_receipt_hash": receipt.receipt_hash,
            "early_stop_reason_code": None,
            "early_stop_reason": None,
            "expected_logical_calls": 7,
            "expected_accepted_responses": 7,
            "allowed_terminal_provider_failures": 0,
            "expected_proposal_calls": 5,
            "expected_reflection_calls": 2,
            "expected_unique_evaluations": expected_unique,
            "expected_delegated_child_evaluations": expected_delegated,
            "treatment_noncompliance_slots": sorted(rejected_slots),
            "treatment_noncompliance_count": len(rejected_slots),
            "physical_evaluation_count_derivation": (
                "distinct_detailed_phenotype_identities_after_seed_replay"
            ),
        }
    if generation_widths != [2, 0]:
        raise RuntimeError("generation widths are outside the closed live state machine")
    if (
        type(early_stop_reason_code) is not str
        or early_stop_reason_code not in CLEAN_EARLY_STOP_REASON_CODES
        or type(early_stop_reason) is not str
        or not early_stop_reason
    ):
        raise RuntimeError("empty G2 lacks one closed authenticated reason")
    states = {
        "diagnostic_rejected": {
            "used": 0,
            "logical": 2,
            "accepted": 2,
            "terminal": 0,
            "reflections": 0,
            "reason_codes": {"reflected_card_batch_unavailable"},
        },
        "reflection_failed": {
            "used": 2,
            "logical": 4,
            "accepted": accepted_provider_responses,
            "terminal": terminal_provider_failures,
            "reflections": 2,
            "reason_codes": {"reflected_card_batch_unavailable"},
        },
        "reflection_rejected": {
            "used": 2,
            "logical": 4,
            "accepted": 4,
            "terminal": 0,
            "reflections": 2,
            "reason_codes": {"reflected_card_batch_unavailable"},
        },
        "ready": {
            "used": 2,
            "logical": 4,
            "accepted": 4,
            "terminal": 0,
            "reflections": 2,
            "reason_codes": {
                "equal_origin_scores",
                "structurally_inapplicable_assignment",
            },
        },
    }
    state = states.get(str(feedback_status))
    if (
        state is None
        or receipt.used_logical_llm_calls != state["used"]
        or early_stop_reason_code not in state["reason_codes"]
        or metadata.get("card_count") not in {"0", "2"}
        or accepted_provider_responses != state["accepted"]
        or terminal_provider_failures != state["terminal"]
        or state["accepted"] + state["terminal"] != state["logical"]
    ):
        raise RuntimeError("early stop disagrees with authenticated feedback state")
    if feedback_status == "ready" and metadata.get("card_count") != "2":
        raise RuntimeError("ready feedback must authenticate exactly two cards")
    if feedback_status != "ready" and metadata.get("card_count") != "0":
        raise RuntimeError("unavailable feedback must authenticate zero cards")
    return {
        "schema_version": 1,
        "mode": "clean_scientific_early_stop",
        "feedback_status": feedback_status,
        "feedback_reason": metadata.get("reason"),
        "feedback_receipt_hash": receipt.receipt_hash,
        "early_stop_reason_code": early_stop_reason_code,
        "early_stop_reason": early_stop_reason,
        "expected_logical_calls": state["logical"],
        "expected_accepted_responses": state["accepted"],
        "allowed_terminal_provider_failures": state["terminal"],
        "expected_proposal_calls": 2,
        "expected_reflection_calls": state["reflections"],
        "expected_unique_evaluations": 4,
        "expected_delegated_child_evaluations": 2,
    }


def _held_out_transfer_adjudication(
    result: OptimizerResult,
    *,
    memory_entries: tuple[Any, ...],
    g2_gate_record: Mapping[str, object] | None,
) -> dict[str, object]:
    """Adjudicate transfer only after exact treatment administration."""

    generation_two = tuple(
        receipt for receipt in result.generation_receipts if receipt.generation == 2
    )
    if not generation_two or len(generation_two[0].slot_results) == 0:
        return {
            "schema_version": 2,
            "policy_id": _HELD_OUT_ADJUDICATOR_DEFINITION["policy_id"],
            "policy_version": _HELD_OUT_ADJUDICATOR_DEFINITION["policy_version"],
            "definition_sha256": HELD_OUT_ADJUDICATOR_SHA256,
            "status": "not_applicable_clean_early_stop",
            "scientific_verdict": "not_applicable_clean_early_stop",
            "metric_adjudications": [],
            "automatic_memory_transition_performed": False,
            "promotion_eligible": False,
            "development_decision": "do_not_advance",
        }
    if len(generation_two) != 1:
        raise RuntimeError("held-out adjudicator requires exactly one G2 receipt")
    slot_by_id = {
        item.slot.slot_id: item for item in generation_two[0].slot_results
    }
    if set(slot_by_id) != {"A", "S", "N"}:
        raise RuntimeError("held-out adjudicator requires exact A/S/N slots")

    gate_commitment = (
        None
        if g2_gate_record is None
        else g2_gate_record.get("assignment_commitment")
    )
    gate_unsigned = None if g2_gate_record is None else dict(g2_gate_record)
    claimed_gate_sha256 = (
        None
        if gate_unsigned is None
        else gate_unsigned.pop("batch_gate_sha256", None)
    )
    gate_integrity = (
        type(g2_gate_record) is dict
        and g2_gate_record.get("status") == "accepted"
        and g2_gate_record.get("policy_id")
        == "airfoil_v7_blinded_g2_prequeue_gate"
        and g2_gate_record.get("policy_version") == 2
        and g2_gate_record.get("policy_identity_sha256")
        == AirfoilV7G2PrequeueGatePolicy().identity_sha256
        and gate_unsigned is not None
        and claimed_gate_sha256
        == _sha256_record(
            gate_unsigned,
            domain=b"agent-evolve:airfoil-v7-g2-prequeue-gate:v2\x00",
        )
        and type(gate_commitment) is dict
        and gate_commitment.get("assignment_sha256")
        == g2_gate_record.get("assignment_sha256")
    )
    chosen_references = (
        None
        if type(gate_commitment) is not dict
        else gate_commitment.get("chosen_references")
    )

    def assigned_reference_record(slot_id: str) -> dict[str, object] | None:
        references = (
            slot_by_id[slot_id].outcome.prepared.plan.quarantine_test_insights
        )
        if len(references) != 1:
            return None
        assigned = references[0]
        return {
            "insight_id": assigned.insight_id.value,
            "insight_version": assigned.version,
        }

    actual_slot_references = {
        slot_id: assigned_reference_record(slot_id)
        for slot_id in ("A", "S", "N")
    }
    expected_slot_references = (
        {}
        if type(chosen_references) is not dict
        else {
            "A": chosen_references.get("adaptive"),
            "S": chosen_references.get("score_swapped"),
            "N": chosen_references.get("sham"),
        }
    )
    plan_assignments_match_gate = (
        all(value is not None for value in actual_slot_references.values())
        and actual_slot_references == expected_slot_references
    )

    policy = StrictTreatmentCompliancePolicy()
    policy_identity = (
        policy.policy_id,
        policy.policy_version,
        policy.definition_sha256,
    )
    arm_records: dict[str, dict[str, object]] = {}
    arm_contexts: dict[str, dict[str, Any]] = {}
    for slot_id in ("A", "S", "N"):
        outcome = slot_by_id[slot_id].outcome
        references = outcome.prepared.plan.quarantine_test_insights
        if len(references) != 1:
            raise RuntimeError(
                f"held-out slot {slot_id} lacks one quarantine assignment"
            )
        reference = references[0]
        matching_entries = tuple(
            entry for entry in memory_entries if entry.reference == reference
        )
        if len(matching_entries) != 1:
            raise RuntimeError(
                f"held-out slot {slot_id} assigned insight is absent or ambiguous"
            )
        entry = matching_entries[0]
        card = entry.draft
        contract = outcome.prepared.plan.finite_variation_contract
        if contract is None:
            raise RuntimeError(
                f"held-out slot {slot_id} lacks its finite action contract"
            )
        candidate = outcome.candidate
        rejected_pre_evaluation = _is_pre_evaluation_treatment_rejection(outcome)
        if candidate is None and not rejected_pre_evaluation:
            raise RuntimeError(
                f"held-out slot {slot_id} missing candidate lacks authenticated "
                "treatment noncompliance"
            )
        if candidate is not None and outcome.failure_stage is not None:
            raise RuntimeError(
                f"held-out slot {slot_id} carries a failed materialized candidate"
            )

        requirement = outcome.prepared.plan.insight_treatment_requirement
        requirement_record = (
            None
            if requirement is None
            else {
                **requirement.to_record(),
                "requirement_sha256": requirement.requirement_sha256,
            }
        )
        preflight = outcome.prepared.treatment_preflight_receipt
        if type(preflight) is TreatmentPreflightReceipt:
            TreatmentPreflightReceipt.__post_init__(preflight)
            preflight_record: dict[str, object] | None = {
                **preflight.to_record(),
                "receipt_sha256": preflight.receipt_sha256,
                "passed": preflight.passed,
            }
        else:
            preflight_record = None
        admission = outcome.treatment_admission_receipt
        if type(admission) is TreatmentAdmissionReceipt:
            TreatmentAdmissionReceipt.__post_init__(admission)
            admission_record: dict[str, object] | None = {
                **admission.to_record(),
                "receipt_sha256": admission.receipt_sha256,
                "passed": admission.passed,
            }
        else:
            admission_record = None

        candidate_options = (
            ()
            if candidate is None
            else tuple(
                option
                for option in contract.options
                if option.child_configuration_sha256
                == candidate.occurrence.configuration_hash
            )
        )
        action_options = (
            ()
            if type(admission) is not TreatmentAdmissionReceipt
            else tuple(
                option
                for option in contract.options
                if option.option_id == admission.selected_action.option_id
                and option.identity_sha256
                == admission.selected_action.option_identity_sha256
                and option.family == admission.selected_action.family
            )
        )
        option = (
            candidate_options[0]
            if len(candidate_options) == 1
            else (action_options[0] if len(action_options) == 1 else None)
        )
        expected_role = (
            TreatmentAssignmentRole.SHAM_CONTROL
            if slot_id == "N"
            else TreatmentAssignmentRole.ACTIVE
        )
        requirement_gate = (
            type(requirement) is InsightTreatmentRequirement
            and requirement.required_insights == references
            and requirement.claim_mode is TreatmentClaimMode.EXACT_REQUIRED
            and requirement.assignment_role is expected_role
        )
        preflight_gate = (
            type(preflight) is TreatmentPreflightReceipt
            and preflight.passed
            and type(requirement) is InsightTreatmentRequirement
            and preflight.requirement_sha256 == requirement.requirement_sha256
            and (
                preflight.policy_id,
                preflight.policy_version,
                preflight.policy_definition_sha256,
            )
            == policy_identity
        )
        admission_gate = (
            type(admission) is TreatmentAdmissionReceipt
            and admission.passed
            and admission.evaluator_entered is False
            and type(preflight) is TreatmentPreflightReceipt
            and admission.preflight_receipt_sha256 == preflight.receipt_sha256
            and (
                admission.policy_id,
                admission.policy_version,
                admission.policy_definition_sha256,
            )
            == policy_identity
        )
        action_gate = (
            admission_gate
            and len(candidate_options) == 1
            and len(action_options) == 1
            and candidate_options[0] == action_options[0]
            and admission.selected_action in preflight.compatible_actions
        )
        insight_id = reference.insight_id.value
        claim_gate = (
            candidate is not None
            and candidate.claimed_insight_ids == (insight_id,)
            and type(admission) is TreatmentAdmissionReceipt
            and admission.claimed_insight_ids == (insight_id,)
        )
        selected_gate = (
            candidate is not None
            and candidate.selected_insight_refs == references
        )
        family_gate = (
            option is not None
            and option.family in card.recommended_option_families
        )
        exact_action_gate = (
            option is not None
            and card.recommended_option_ids == (option.option_id,)
        )
        candidate_compliance_gate = (
            candidate is not None
            and candidate.valid
            and candidate.operator_compliant
            and candidate.evidence_compliant
        )
        detailed = None if candidate is None else candidate.detailed_evaluation
        evaluation_recorded = detailed is not None
        treatment_administered = (
            not rejected_pre_evaluation
            and requirement_gate
            and preflight_gate
            and admission_gate
            and action_gate
            and claim_gate
            and selected_gate
            and family_gate
            and exact_action_gate
            and candidate_compliance_gate
            and detailed is not None
            and detailed.success
        )
        exact_patch = (
            candidate is not None
            and candidate.operator_compliant
            and len(candidate_options) == 1
        )
        phenotype = None if detailed is None else detailed.phenotype
        arm_records[slot_id] = {
            "reward": outcome.reward,
            "failure_stage": outcome.failure_stage,
            "call_failure_type": outcome.call_failure_type,
            "candidate_materialized": candidate is not None,
            "evaluation_recorded": evaluation_recorded,
            "treatment_rejected_pre_evaluation": rejected_pre_evaluation,
            "treatment_requirement": requirement_record,
            "treatment_preflight": preflight_record,
            "treatment_admission": admission_record,
            "treatment_administered": treatment_administered,
            "evaluator_entered_on_admission": (
                None if admission is None else admission.evaluator_entered
            ),
            "requirement_gate_pass": requirement_gate,
            "preflight_gate_pass": preflight_gate,
            "admission_gate_pass": admission_gate,
            "action_gate_pass": action_gate,
            "claim_gate_pass": claim_gate,
            "selected_assignment_gate_pass": selected_gate,
            "action_family_gate_pass": family_gate,
            "exact_action_gate_pass": exact_action_gate,
            "candidate_compliance_gate_pass": candidate_compliance_gate,
            "valid": None if candidate is None else candidate.valid,
            "operator_compliant": (
                None if candidate is None else candidate.operator_compliant
            ),
            "evidence_compliant": (
                None if candidate is None else candidate.evidence_compliant
            ),
            "exact_finite_patch": exact_patch,
            "selected_option_id": (
                None
                if option is None
                else option.option_id
            ),
            "selected_option_family": (
                None
                if option is None
                else option.family
            ),
            "selected_option_identity_sha256": (
                None
                if option is None
                else option.identity_sha256
            ),
            "selected_insight_ids": (
                [] if candidate is None else list(candidate.selected_insight_ids)
            ),
            "claimed_insight_ids": (
                list(admission.claimed_insight_ids)
                if candidate is None and admission is not None
                else ([] if candidate is None else list(candidate.claimed_insight_ids))
            ),
            "phenotype": (
                None if phenotype is None else phenotype.to_trace_record()
            ),
            "phenotype_identity_sha256": (
                None if phenotype is None else phenotype.identity_sha256
            ),
            "f": (
                None
                if detailed is None or not detailed.success
                else dict(detailed.objectives)[OBJECTIVE_NAME]
            ),
            "V": (
                None
                if detailed is None or not detailed.success
                else dict(detailed.violations)[VIOLATION_NAME]
            ),
        }
        arm_contexts[slot_id] = {
            "outcome": outcome,
            "candidate": candidate,
            "reference": reference,
            "card": card,
            "option": option,
            "detailed": detailed,
        }

    sham_reference_record = actual_slot_references["N"]
    sham_entries = tuple(
        entry
        for entry in memory_entries
        if sham_reference_record is not None
        and entry.reference.insight_id.value
        == sham_reference_record["insight_id"]
        and entry.reference.version
        == sham_reference_record["insight_version"]
    )
    neutral_sham = (
        len(sham_entries) == 1
        and sham_entries[0].origin.value == "manual"
        and sham_entries[0].lifecycle_state.value == "quarantined"
        and sham_entries[0].evidence_lineage is None
        and not sham_entries[0].draft.evidence_contrast_ids
        and sham_entries[0].draft.recommended_option_ids == (SHAM_OPTION_ID,)
        and all(
            prediction.direction.value == "unknown"
            for prediction in sham_entries[0].draft.effect_predictions
        )
    )
    candidate_assignment_bindings = all(
        arm_records[slot_id]["selected_assignment_gate_pass"] is True
        for slot_id in ("A", "S", "N")
    )
    as_mapping_distinct = (
        gate_integrity
        and type(chosen_references) is dict
        and chosen_references.get("adaptive")
        != chosen_references.get("score_swapped")
        and plan_assignments_match_gate
        and len(
            {
                json.dumps(value, sort_keys=True)
                for value in actual_slot_references.values()
                if value is not None
            }
        )
        == 3
        and neutral_sham
    )

    adaptive_context = arm_contexts["A"]
    adaptive = adaptive_context["outcome"]
    candidate = adaptive_context["candidate"]
    reference = adaptive_context["reference"]
    card = adaptive_context["card"]
    option = adaptive_context["option"]
    adaptive_administered = (
        arm_records["A"]["treatment_administered"] is True
    )
    metric_rows: list[dict[str, object]] = []
    if adaptive_administered:
        assert candidate is not None
        assert candidate.detailed_evaluation is not None
        parent = adaptive.prepared.plan.parents[0]
        if parent.detailed_evaluation is None:
            raise RuntimeError("held-out metric adjudication lacks parent evidence")
        parent_values = {
            "objective:normalized_multipoint_drag": dict(
                parent.detailed_evaluation.objectives
            )[OBJECTIVE_NAME],
            "violation:normalized_lift_equality": dict(
                parent.detailed_evaluation.violations
            )[VIOLATION_NAME],
        }
        child_values = {
            "objective:normalized_multipoint_drag": dict(
                candidate.detailed_evaluation.objectives
            )[OBJECTIVE_NAME],
            "violation:normalized_lift_equality": dict(
                candidate.detailed_evaluation.violations
            )[VIOLATION_NAME],
        }
        thresholds = {
            "objective:normalized_multipoint_drag": 0.001,
            "violation:normalized_lift_equality": 0.005,
        }
        predictions = {
            prediction.metric_id: prediction.direction.value
            for prediction in card.effect_predictions
        }
        if set(predictions) != set(parent_values):
            raise RuntimeError("adaptive card metric prediction set drifted")
        for metric_id in sorted(parent_values):
            observed = _metric_direction(
                parent_value=parent_values[metric_id],
                child_value=child_values[metric_id],
                threshold=thresholds[metric_id],
            )
            predicted = predictions[metric_id]
            supported = predicted != "unknown" and predicted == observed
            falsified = predicted != "unknown" and predicted != observed
            metric_rows.append(
                {
                    "metric_id": metric_id,
                    "parent_value": parent_values[metric_id],
                    "child_value": child_values[metric_id],
                    "threshold": thresholds[metric_id],
                    "predicted_direction": predicted,
                    "observed_direction": observed,
                    "supported": supported,
                    "falsified": falsified,
                }
            )

    phenotype_ids = {
        slot_id: arm_records[slot_id]["phenotype_identity_sha256"]
        for slot_id in ("A", "S", "N")
    }
    as_phenotypes_distinct = (
        phenotype_ids["A"] is not None
        and phenotype_ids["S"] is not None
        and phenotype_ids["A"] != phenotype_ids["S"]
    )
    all_arm_phenotypes_distinct = (
        None not in phenotype_ids.values()
        and len(set(phenotype_ids.values())) == 3
    )
    all_arm_treatments_administered = all(
        arm_records[slot_id]["treatment_administered"] is True
        for slot_id in ("A", "S", "N")
    )
    per_protocol_asn_eligible = (
        all_arm_treatments_administered
        and candidate_assignment_bindings
        and all_arm_phenotypes_distinct
    )
    valid_exact_as = (
        arm_records["A"]["treatment_administered"] is True
        and arm_records["S"]["treatment_administered"] is True
        and arm_records["A"]["exact_finite_patch"] is True
        and arm_records["S"]["exact_finite_patch"] is True
    )

    successful_candidates = tuple(
        context["candidate"]
        for context in arm_contexts.values()
        if context["candidate"] is not None
        and context["detailed"] is not None
        and context["detailed"].success
    )
    strict_best = False
    if len(successful_candidates) == 3:
        adaptive_candidate = arm_contexts["A"]["candidate"]
        score_swapped_candidate = arm_contexts["S"]["candidate"]
        sham_candidate = arm_contexts["N"]["candidate"]
        assert adaptive_candidate is not None
        assert score_swapped_candidate is not None
        assert sham_candidate is not None
        assert adaptive_candidate.detailed_evaluation is not None
        assert score_swapped_candidate.detailed_evaluation is not None
        assert sham_candidate.detailed_evaluation is not None
        strict_best = (
            AIRFOIL_V7_ARCHIVE_RELATION.compare(
                adaptive_candidate.detailed_evaluation,
                score_swapped_candidate.detailed_evaluation,
            )
            is OutcomeRelation.BETTER
            and AIRFOIL_V7_ARCHIVE_RELATION.compare(
                adaptive_candidate.detailed_evaluation,
                sham_candidate.detailed_evaluation,
            )
            is OutcomeRelation.BETTER
        )

    primary_row = next(
        (
            row
            for row in metric_rows
            if row["metric_id"] == "objective:normalized_multipoint_drag"
        ),
        None,
    )
    primary_direction_correct = (
        primary_row is not None and primary_row["supported"] is True
    )
    violation_row = next(
        (
            row
            for row in metric_rows
            if row["metric_id"] == "violation:normalized_lift_equality"
        ),
        None,
    )
    unpredicted_v_regression = (
        violation_row is not None
        and violation_row["observed_direction"] == "increase"
        and violation_row["predicted_direction"] != "increase"
    )
    rewards = {
        slot_id: slot_by_id[slot_id].outcome.reward for slot_id in ("A", "S", "N")
    }
    conjuncts = {
        "1_seven_calls_complete_pending_tree_finalization": (
            result.final_state.logical_llm_calls == 7
            and [len(item.slot_results) for item in result.generation_receipts]
            == [2, 3]
        ),
        "2_registered_as_mapping_only_and_distinct_cards": as_mapping_distinct,
        "3_as_valid_exact_patch_distinct_phenotypes": (
            valid_exact_as and as_phenotypes_distinct
        ),
        "4_adaptive_reward_positive_and_strict": (
            rewards["A"] == 1.0
            and rewards["A"] > max(rewards["S"], rewards["N"])
        ),
        "5_adaptive_strict_domain_native_best": strict_best,
        "6_primary_direction_correct_no_unpredicted_v_regression": (
            adaptive_administered
            and primary_direction_correct
            and not unpredicted_v_regression
        ),
    }
    exact_patch_all = all(
        arm_records[slot_id]["exact_finite_patch"] is True
        for slot_id in ("A", "S", "N")
    )
    pre_finalization_eligible = (
        all(conjuncts.values())
        and per_protocol_asn_eligible
        and exact_patch_all
    )

    if not adaptive_administered:
        status = "not_tested_noncompliance"
        scientific_verdict = "not_tested_noncompliance"
        treatment_administration_status = (
            "rejected_pre_evaluation"
            if arm_records["A"]["treatment_rejected_pre_evaluation"] is True
            else "not_demonstrated"
        )
    else:
        status = "adjudicated"
        treatment_administration_status = "administered"
        if any(row["falsified"] is True for row in metric_rows):
            scientific_verdict = "falsified"
        elif any(row["supported"] is True for row in metric_rows):
            scientific_verdict = "supported"
        else:
            scientific_verdict = "inconclusive"
    development_decision = (
        "advance_after_successful_evidence_tree_finalization"
        if pre_finalization_eligible
        else "do_not_advance"
    )
    card_record = _insight_content(card)
    insight_id = reference.insight_id.value
    selected_claims = (
        []
        if candidate is None
        else list(candidate.claimed_insight_ids)
    )
    return {
        "schema_version": 2,
        "policy_id": _HELD_OUT_ADJUDICATOR_DEFINITION["policy_id"],
        "policy_version": _HELD_OUT_ADJUDICATOR_DEFINITION["policy_version"],
        "definition_sha256": HELD_OUT_ADJUDICATOR_SHA256,
        "status": status,
        "treatment_administration_status": treatment_administration_status,
        "assigned_insight": {
            "insight_id": insight_id,
            "version": reference.version,
        },
        "assigned_card_sha256": _sha256_record(
            card_record,
            domain=b"agent-evolve:airfoil-v7-adjudicated-card:v2\x00",
        ),
        "claimed_insight_ids": selected_claims,
        "claim_gate_pass": arm_records["A"]["claim_gate_pass"],
        "selected_assignment_gate_pass": arm_records["A"][
            "selected_assignment_gate_pass"
        ],
        "selected_option_id": (
            None if option is None else option.option_id
        ),
        "selected_option_family": (
            None if option is None else option.family
        ),
        "recommended_option_families": list(
            card.recommended_option_families
        ),
        "recommended_option_ids": list(card.recommended_option_ids),
        "action_family_gate_pass": arm_records["A"][
            "action_family_gate_pass"
        ],
        "exact_action_gate_pass": arm_records["A"][
            "exact_action_gate_pass"
        ],
        "candidate_compliance_gate_pass": arm_records["A"][
            "candidate_compliance_gate_pass"
        ],
        "typed_treatment_admission_gate_pass": adaptive_administered,
        "metric_adjudication_gate_pass": adaptive_administered,
        "metric_adjudications": metric_rows,
        "scientific_verdict": scientific_verdict,
        "artifact_91_promotion_conjuncts": conjuncts,
        "g2_gate_integrity_pass": gate_integrity,
        "actual_slot_assignment_references": actual_slot_references,
        "expected_slot_assignment_references": expected_slot_references,
        "plan_assignment_bindings_pass": plan_assignments_match_gate,
        "candidate_assignment_bindings_pass": candidate_assignment_bindings,
        "neutral_sham_gate_pass": neutral_sham,
        "all_arm_treatments_administered": all_arm_treatments_administered,
        "per_protocol_asn_eligible": per_protocol_asn_eligible,
        "treatment_noncompliance_slots": [
            slot_id
            for slot_id in ("A", "S", "N")
            if arm_records[slot_id]["treatment_rejected_pre_evaluation"] is True
        ],
        "pre_finalization_promotion_eligible": pre_finalization_eligible,
        "promotion_eligible": False,
        "promotion_pending_evidence_tree_finalization": (
            pre_finalization_eligible
        ),
        "automatic_memory_transition_performed": False,
        "arm_rewards": rewards,
        "delta_AS": rewards["A"] - rewards["S"],
        "delta_AN": rewards["A"] - rewards["N"],
        "arms": arm_records,
        "phenotype_ids": phenotype_ids,
        "as_phenotypes_distinct": as_phenotypes_distinct,
        "all_arm_phenotypes_distinct": all_arm_phenotypes_distinct,
        "adaptive_strict_domain_native_best": strict_best,
        "primary_direction_correct": primary_direction_correct,
        "unpredicted_v_regression": unpredicted_v_regression,
        "development_decision": development_decision,
    }

class QualifiedSeedReplayRawProblem:
    """Replay two content-bound raw receipts, then delegate every child to CFD."""

    def __init__(
        self,
        *,
        delegate: ConvergenceQualifiedAirfoilPanelProblem,
        seed_rows: tuple[Mapping[str, object], Mapping[str, object]],
    ) -> None:
        self.delegate = delegate
        self._rows = {
            str(row["candidate_sha256"]): dict(row) for row in seed_rows
        }
        if len(self._rows) != 2:
            raise ValueError("seed replay requires two distinct candidate hashes")
        self._used: set[str] = set()
        self.replay_calls = 0
        self.delegated_calls = 0
        self.replay_payload_identities: dict[str, str] = {}

    def evaluate_raw(self, configuration: object) -> AirfoilPanelEvaluation:
        candidate = normalize_candidate(configuration)
        key = candidate_sha256(candidate)
        row = self._rows.get(key)
        if row is None:
            self.delegated_calls += 1
            return self.delegate.evaluate_raw(candidate)
        if key in self._used:
            raise RuntimeError("a qualified seed receipt was replayed more than once")
        if row.get("configuration") != candidate:
            raise RuntimeError("qualified seed configuration differs from replay input")
        path = Path(str(row.get("raw_receipt_path"))).resolve(strict=True)
        content = path.read_bytes()
        if (
            hashlib.sha256(content).hexdigest()
            != row.get("raw_receipt_sha256")
            or len(content) != row.get("raw_receipt_bytes")
        ):
            raise RuntimeError("qualified seed receipt bytes changed before replay")
        record = json.loads(content)
        _verify_raw_success_receipt(path, configuration=candidate)
        objectives = record.get("objectives")
        if type(objectives) is not dict:
            raise RuntimeError("qualified raw receipt lacks objectives")
        self._used.add(key)
        self.replay_calls += 1
        return AirfoilPanelEvaluation(
            candidate_sha256=key,
            objective_values={
                str(name): float(value) for name, value in objectives.items()
            },
            wall_seconds=float(record["wall_seconds"]),
            record_path=path,
            record=record,
        )


class QualificationBoundDetailedEvaluator:
    """Require replayed seed projections to equal Phase-A sealed payloads."""

    evaluator_identity = EVALUATOR_IDENTITY

    def __init__(
        self,
        *,
        delegate: AirfoilV7DetailedEvaluationAdapter,
        replay: QualifiedSeedReplayRawProblem,
        seed_rows: tuple[Mapping[str, object], Mapping[str, object]],
    ) -> None:
        self.delegate = delegate
        self.replay = replay
        self._expected = {
            str(row["candidate_sha256"]): row.get("payload") for row in seed_rows
        }
        if len(self._expected) != 2 or any(
            type(value) is not dict for value in self._expected.values()
        ):
            raise ValueError("projection replay requires two sealed seed payloads")

    def evaluate_evidence(
        self,
        configuration: object,
    ) -> DetailedEvaluationPayload:
        key = candidate_sha256(configuration)
        payload = self.delegate.evaluate_evidence(configuration)
        expected = self._expected.get(key)
        if expected is None:
            return payload
        observed = _payload_record(payload)
        if observed != expected:
            raise RuntimeError(
                "runtime seed projection differs from Phase-A sealed payload"
            )
        identity = _sha256_record(
            observed,
            domain=b"agent-evolve:airfoil-v7-seed-projection:v1\x00",
        )
        prior = self.replay.replay_payload_identities.setdefault(key, identity)
        if prior != identity:
            raise RuntimeError("seed projection replay identity changed")
        return payload


class SeedReplayAccounting(Protocol):
    """Narrow observable boundary for physical seed/child accounting."""

    replay_calls: int
    delegated_calls: int
    replay_payload_identities: dict[str, str]


def _require_seed_replay_accounting(
    benchmark: AgenticBenchmark,
) -> SeedReplayAccounting:
    raw_problem = getattr(benchmark.problem, "raw_problem", None)
    replay_calls = getattr(raw_problem, "replay_calls", None)
    delegated_calls = getattr(raw_problem, "delegated_calls", None)
    payload_identities = getattr(raw_problem, "replay_payload_identities", None)
    if (
        type(replay_calls) is not int
        or type(delegated_calls) is not int
        or type(payload_identities) is not dict
    ):
        raise TypeError(
            "live benchmark must expose integer seed replay/delegation accounting"
        )
    return raw_problem


def _copy_bound_seed_receipts(
    run_dir: Path,
    binding: Mapping[str, object],
) -> dict[str, object]:
    destination_root = run_dir / "bound_seed_receipts"
    destination_root.mkdir(exist_ok=False)
    seeds = binding.get("seeds")
    if type(seeds) is not list or len(seeds) != 2:
        raise RuntimeError("provider binding lacks exactly two seed receipts")
    copied = []
    for row in seeds:
        if type(row) is not dict:
            raise TypeError("bound seed row must be an object")
        source = Path(str(row.get("raw_receipt_path"))).resolve(strict=True)
        content = source.read_bytes()
        expected_sha = row.get("raw_receipt_sha256")
        if hashlib.sha256(content).hexdigest() != expected_sha:
            raise RuntimeError("bound seed receipt drifted before copy")
        destination = destination_root / f"{row['candidate_sha256']}.json"
        write_bytes_atomic(destination, content)
        copied_row = {
            **row,
            "qualification_raw_receipt_path": str(source),
            "raw_receipt_path": str(destination),
        }
        copied.append(copied_row)
    index = {
        "schema_version": 1,
        "qualification_sha256": binding["qualification_sha256"],
        "seeds": copied,
    }
    write_json_atomic(run_dir / "bound_seed_receipts.json", index)
    return index


def create_real_benchmark(
    run_id: str,
    run_dir: Path,
    seed_binding: Mapping[str, object],
) -> AgenticBenchmark:
    """Construct receipt-replayed seeds plus serial real-CFD children."""

    settings = replace(
        local_default_converged_settings(),
        output_root=run_dir / "cfd_receipts",
        work_root=Path("/tmp") / "agent_evolve_airfoil_v7" / run_id,
    )
    raw = ConvergenceQualifiedAirfoilPanelProblem(settings)
    seed_rows = seed_binding.get("seeds")
    if type(seed_rows) is not list or len(seed_rows) != 2:
        raise RuntimeError("real benchmark requires two copied seed receipts")
    replay = QualifiedSeedReplayRawProblem(
        delegate=raw,
        seed_rows=(seed_rows[0], seed_rows[1]),
    )
    problem = AirfoilV7Problem(raw_problem=replay)
    detailed_evaluator = QualificationBoundDetailedEvaluator(
        delegate=AirfoilV7DetailedEvaluationAdapter(replay),
        replay=replay,
        seed_rows=(seed_rows[0], seed_rows[1]),
    )
    return AgenticBenchmark(
        problem=problem,
        reward=AIRFOIL_V7_REWARD_BINDING,
        detailed_evaluator=detailed_evaluator,
        outcome_relation=AIRFOIL_V7_ARCHIVE_RELATION,
        phenotype_identity=AirfoilV7PhenotypeIdentityPolicy(),
        finite_variation_catalogs=(
            AirfoilV7ShapeVariationCatalog(),
            AirfoilV7TrimVariationCatalog(),
            AirfoilV7UnionVariationCatalog(),
        ),
    )


@dataclass(frozen=True, slots=True)
class LiveExecutionDependencies:
    benchmark_factory: Callable[
        [str, Path, Mapping[str, object]],
        AgenticBenchmark,
    ]
    credential_loader: Callable[[], str]
    stack_factory: Callable[[str, Callable[[Mapping[str, object]], None]], LiveStackLike]
    resource_lease_factory: Callable[
        [str, str], ExclusiveResourceLease
    ] = _production_resource_lease
    enforce_canonical_output: bool = True


def production_live_dependencies() -> LiveExecutionDependencies:
    """Return lazy production factories without reading ``.env``."""

    def load_key() -> str:
        from dotenv import load_dotenv

        load_dotenv(WORKSPACE_ROOT / ".env", override=False)
        return os.environ.get("OPENROUTER_API_KEY", "")

    def stack_factory(
        api_key: str,
        queue_sink: Callable[[Mapping[str, object]], None],
    ) -> LiveStackLike:
        from examples.development.run_v6_closed_loop_memory_probe import (
            create_live_stack,
        )
        from agent_evolve.integrations.pydantic_ai.async_generator import (
            OpenRouterReasoningConfig,
        )
        from agent_evolve.integrations.pydantic_ai.queued_runner import (
            structured_generation_outcome_record,
        )

        return create_live_stack(
            api_key=api_key,
            queue_sink=lambda outcome: queue_sink(
                structured_generation_outcome_record(outcome)
            ),
            telemetry_policy_override=_airfoil_v7_telemetry_policy(),
            attempt_timeout_ns=PROVIDER_ATTEMPT_TIMEOUT_NS,
            reasoning_config=OpenRouterReasoningConfig(
                max_tokens=PROVIDER_REASONING_MAX_TOKENS,
            ),
        )

    return LiveExecutionDependencies(
        benchmark_factory=create_real_benchmark,
        credential_loader=load_key,
        stack_factory=stack_factory,
        resource_lease_factory=_production_resource_lease,
        enforce_canonical_output=True,
    )


def _result_record(result: OptimizerResult) -> dict[str, object]:
    seeds = []
    for receipt in result.seed_receipts:
        candidate = receipt.candidate
        seeds.append(
            {
                "label": receipt.label,
                "receipt_hash": receipt.receipt_hash,
                "candidate_id": candidate.candidate_id.value,
                "configuration": candidate.configuration_dict,
                "configuration_sha256": candidate_sha256(
                    candidate.configuration_dict
                ),
                "objectives": candidate.objective_map,
                "valid": candidate.valid,
                "evidence_compliant": candidate.evidence_compliant,
                "detailed_evaluation": (
                    None
                    if candidate.detailed_evaluation is None
                    else candidate.detailed_evaluation.to_record()
                ),
            }
        )
    generations = []
    for receipt in result.generation_receipts:
        slots = []
        for item in receipt.slot_results:
            outcome = item.outcome
            candidate = outcome.candidate
            requirement = outcome.prepared.plan.insight_treatment_requirement
            preflight = outcome.prepared.treatment_preflight_receipt
            admission = outcome.treatment_admission_receipt
            slots.append(
                {
                    "slot_id": item.slot.slot_id,
                    "role": item.slot.role,
                    "call_id": (
                        None
                        if outcome.prepared.call_id is None
                        else outcome.prepared.call_id.value
                    ),
                    "reward": outcome.reward,
                    "failure_stage": outcome.failure_stage,
                    "call_failure_type": outcome.call_failure_type,
                    "treatment_requirement": (
                        None
                        if requirement is None
                        else {
                            **requirement.to_record(),
                            "requirement_sha256": requirement.requirement_sha256,
                        }
                    ),
                    "treatment_preflight": (
                        None
                        if preflight is None
                        else {
                            **preflight.to_record(),
                            "receipt_sha256": preflight.receipt_sha256,
                            "passed": preflight.passed,
                        }
                    ),
                    "treatment_admission": (
                        None
                        if admission is None
                        else {
                            **admission.to_record(),
                            "receipt_sha256": admission.receipt_sha256,
                            "passed": admission.passed,
                        }
                    ),
                    "treatment_rejected_pre_evaluation": (
                        _is_pre_evaluation_treatment_rejection(outcome)
                    ),
                    "evaluator_entered_on_treatment_admission": (
                        None
                        if admission is None
                        else admission.evaluator_entered
                    ),
                    "candidate": (
                        None
                        if candidate is None
                        else {
                            "candidate_id": candidate.candidate_id.value,
                            "configuration": candidate.configuration_dict,
                            "configuration_sha256": candidate_sha256(
                                candidate.configuration_dict
                            ),
                            "objectives": candidate.objective_map,
                            "valid": candidate.valid,
                            "operator_compliant": candidate.operator_compliant,
                            "evidence_compliant": candidate.evidence_compliant,
                            "design_rationale": candidate.design_rationale,
                            "claimed_insight_ids": list(
                                candidate.claimed_insight_ids
                            ),
                            "selected_insight_ids": list(
                                candidate.selected_insight_ids
                            ),
                            "selected_insight_refs": [
                                {
                                    "insight_id": reference.insight_id.value,
                                    "version": reference.version,
                                }
                                for reference in candidate.selected_insight_refs
                            ],
                            "detailed_evaluation": (
                                None
                                if candidate.detailed_evaluation is None
                                else candidate.detailed_evaluation.to_record()
                            ),
                        }
                    ),
                }
            )
        generations.append(
            {
                "generation": receipt.generation,
                "receipt_hash": receipt.receipt_hash,
                "slots": slots,
            }
        )
    return {
        "schema_version": 1,
        "result_hash": result.result_hash,
        "stop_reason": result.stop_reason.value,
        "final_generation": result.final_state.generation,
        "unique_evaluations": result.final_state.unique_evaluations,
        "logical_llm_calls": result.final_state.logical_llm_calls,
        "seeds": seeds,
        "generations": generations,
        "feedback_receipts": [
            {
                "generation": item.generation,
                "receipt_hash": item.receipt_hash,
                "used_logical_llm_calls": item.used_logical_llm_calls,
                "result_metadata": [list(row) for row in item.result_metadata],
            }
            for item in result.feedback_receipts
        ],
    }


def _finalize_run(
    run_dir: Path,
    *,
    status: str,
    post_finalization_decision: str | None = None,
) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    aggregate = hashlib.sha256(b"agent-evolve:airfoil-v7-live-finalized:v1\x00")
    for path in sorted(
        (
            item
            for item in run_dir.rglob("*")
            if item.is_file()
            and item.name != "finalized.json"
            and not item.name.endswith(".tmp")
        ),
        key=lambda item: item.relative_to(run_dir).as_posix(),
    ):
        relative = path.relative_to(run_dir).as_posix()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        row: dict[str, object] = {"bytes": len(content), "sha256": digest}
        if path.suffix == ".jsonl":
            row["jsonl_lines"] = len(content.splitlines())
        files[relative] = row
        encoded = relative.encode("utf-8")
        aggregate.update(len(encoded).to_bytes(8, "big"))
        aggregate.update(encoded)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
    record = {
        "schema_version": 1,
        "status": status,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "recursive_file_count": len(files),
        "recursive_content_sha256": aggregate.hexdigest(),
        "files": files,
        "post_finalization_decision": post_finalization_decision,
    }
    write_json_atomic(run_dir / "finalized.json", record)
    return record


def execute_live_with_dependencies(
    manifest_path: Path,
    dependencies: LiveExecutionDependencies,
) -> dict[str, object]:
    """Execute the real route or an explicitly injected all-double route."""

    verified = verify_launch_manifest(
        manifest_path,
        require_output_absent=True,
        enforce_canonical_output=dependencies.enforce_canonical_output,
    )
    run_dir = verified.output_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    _directory_fsync(run_dir.parent)
    write_bytes_atomic(run_dir / "launch_manifest.json", verified.path.read_bytes())

    writers = {
        name: DurableJsonlWriter(run_dir / filename)
        for name, filename in {
            "candidate": "candidate_receipts.jsonl",
            "queue": "provider_queue_outcomes.jsonl",
            "journal": "prompt_response_journal.jsonl",
            "source": "source_verifications.jsonl",
            "trace": "traces.jsonl",
        }.items()
    }
    status = "failed"
    pending: BaseException | None = None
    summary: dict[str, object] | None = None
    replay_accounting: SeedReplayAccounting | None = None
    queue_outcomes: list[dict[str, object]] = []
    queue_outcomes_lock = threading.Lock()
    resource_lease: ExclusiveResourceLease | None = None
    try:
        resource_lease = dependencies.resource_lease_factory(
            verified.run_id,
            "provider_evolution",
        )
        lease_receipt = resource_lease.acquire()
        write_json_atomic(
            run_dir / "resource_lease_acquired.json",
            {
                "schema_version": 1,
                "phase": "provider_evolution",
                "receipt": lease_receipt.to_record(),
            },
        )
        writers["source"].write(
            {
                "stage": "post_run_directory_creation",
                **reverify_launch_source(verified),
            }
        )

        def verify_before_provider(stage: str) -> Mapping[str, object]:
            if stage == "pre_live_credential_load" and (
                replay_accounting is None
                or replay_accounting.replay_calls != 2
                or replay_accounting.delegated_calls != 0
                or len(replay_accounting.replay_payload_identities) != 2
            ):
                raise RuntimeError(
                    "credentials require exactly two replayed seeds and zero child CFD"
                )
            row = {"stage": stage, **reverify_launch_source(verified)}
            writers["source"].write(row)
            return row

        def trace_sink(stream: str) -> Callable[[Mapping[str, object]], None]:
            def sink(event: Mapping[str, object]) -> None:
                row = {"stream": stream, **dict(event)}
                writers["trace"].write(row)
                if event.get("event_type") in {
                    "seed_registered",
                    "candidate_evaluated",
                }:
                    writers["candidate"].write(row)

            return sink

        launch = verified.record["launch"]
        if type(launch) is not dict:
            raise RuntimeError("verified launch record changed type")
        seed_binding = launch.get("seed_qualification")
        if type(seed_binding) is not dict:
            raise RuntimeError("verified launch lacks seed qualification binding")
        model_route = launch.get("model_route")
        if type(model_route) is not dict:
            raise RuntimeError("verified launch lacks its model-route binding")
        expected_telemetry_policy = model_route.get("telemetry_policy")
        expected_telemetry_policy_sha256 = model_route.get(
            "telemetry_policy_sha256"
        )
        if (
            type(expected_telemetry_policy) is not dict
            or type(expected_telemetry_policy_sha256) is not str
        ):
            raise RuntimeError(
                "verified launch telemetry-policy binding is malformed"
            )
        copied_seed_binding = _copy_bound_seed_receipts(run_dir, seed_binding)
        benchmark = dependencies.benchmark_factory(
            verified.run_id,
            run_dir,
            copied_seed_binding,
        )
        replay_accounting = _require_seed_replay_accounting(benchmark)

        def stack_factory(api_key: str) -> LiveStackLike:
            def record_queue_outcome(record: Mapping[str, object]) -> None:
                row = dict(record)
                writers["queue"].write(row)
                with queue_outcomes_lock:
                    queue_outcomes.append(row)

            return dependencies.stack_factory(
                api_key,
                record_queue_outcome,
            )

        generator = DeferredJournaledLiveGenerator(
            credential_loader=dependencies.credential_loader,
            stack_factory=stack_factory,
            pre_provider_verifier=verify_before_provider,
            journal=writers["journal"],
            expected_telemetry_policy=expected_telemetry_policy,
            expected_telemetry_policy_sha256=(
                expected_telemetry_policy_sha256
            ),
        )
        core = compose_airfoil_v7_experiment(
            benchmark=benchmark,
            generator=generator,
            id_namespace=_live_id_namespace(verified.run_id),
            engine_trace_sink=trace_sink("engine"),
            optimizer_trace_sink=trace_sink("optimizer"),
        )
        generator.bind_assignment_commitment_supplier(
            lambda: core.planner.held_out_assignment_commitment
        )
        held_out = core.held_out_parent
        manifest_held_out = verified.record["launch"]["parents"]["held_out"]
        if held_out.to_record() != manifest_held_out:
            raise RuntimeError("runtime held-out parent drifted from launch manifest")

        async def run() -> OptimizerResult:
            async with generator:
                return await core.composition.optimizer.run(
                    (NEUTRAL_PARENT, held_out.candidate)
                )

        async def run_with_heartbeat() -> OptimizerResult:
            execution = asyncio.create_task(run())
            while not execution.done():
                await asyncio.sleep(0.01)
            return await execution

        loop = asyncio.new_event_loop()
        executor = ThreadPoolExecutor(
            max_workers=EVALUATOR_CONCURRENCY,
            thread_name_prefix="airfoil_v7_live_evaluator",
        )
        loop.set_default_executor(executor)
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_with_heartbeat())
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            loop.close()
            asyncio.set_event_loop(None)
        writers["source"].write(
            {
                "stage": "post_queue_pre_adjudication",
                **reverify_launch_source(verified),
            }
        )
        result_record = _result_record(result)
        write_json_atomic(run_dir / "result.json", result_record)
        widths = [
            len(receipt.slot_results) for receipt in result.generation_receipts
        ]
        early_stop_reason_code = core.planner.early_stop_reason_code
        early_stop_reason = core.planner.early_stop_reason
        execution_path = _execution_path_record(
            result,
            generation_widths=widths,
            early_stop_reason_code=early_stop_reason_code,
            early_stop_reason=early_stop_reason,
            accepted_provider_responses=len(generator.accepted_response_records),
            terminal_provider_failures=sum(
                row.get("status") != "succeeded" for row in queue_outcomes
            ),
        )
        full_path = execution_path["mode"] == "full"
        clean_early_stop = (
            execution_path["mode"] == "clean_scientific_early_stop"
        )
        assignment_commitment_record: dict[str, object] | None = None
        gate_record = generator.accepted_g2_gate_record
        commitment = core.planner.held_out_assignment_commitment
        if full_path:
            if (
                type(commitment) is not HeldOutASNAssignmentCommitment
                or type(gate_record) is not dict
            ):
                raise RuntimeError("full path lacks assignment/gate commitment")
            assignment_commitment_record = commitment.to_record()
            if (
                gate_record.get("assignment_sha256")
                != commitment.assignment_sha256
                or gate_record.get("assignment_commitment")
                != assignment_commitment_record
            ):
                raise RuntimeError("G2 gate and planner assignment commitments differ")
            write_json_atomic(
                run_dir / "held_out_assignment.json",
                assignment_commitment_record,
            )
            write_json_atomic(run_dir / "g2_prequeue_gate.json", gate_record)
        elif commitment is not None or gate_record is not None:
            raise RuntimeError("early-stop path unexpectedly published a G2 assignment")
        provider_accounting = _provider_accounting_record(
            accepted_responses=generator.accepted_response_records,
            queue_outcomes=tuple(queue_outcomes),
            expected_logical_calls=int(
                execution_path["expected_logical_calls"]
            ),
            expected_accepted_responses=int(
                execution_path["expected_accepted_responses"]
            ),
            allowed_terminal_failures=int(
                execution_path["allowed_terminal_provider_failures"]
            ),
        )
        transfer_adjudication = _held_out_transfer_adjudication(
            result,
            memory_entries=core.composition.memory.entries,
            g2_gate_record=gate_record,
        )
        if (
            full_path
            and transfer_adjudication.get("status")
            not in {"adjudicated", "not_tested_noncompliance"}
        ) or (
            clean_early_stop
            and transfer_adjudication.get("status")
            != "not_applicable_clean_early_stop"
        ):
            raise RuntimeError("held-out transfer adjudication state mismatch")
        write_json_atomic(run_dir / "adjudication.json", transfer_adjudication)
        accounting_pass = (
            result.final_state.logical_llm_calls
            == execution_path["expected_logical_calls"]
            and result.final_state.unique_evaluations
            == execution_path["expected_unique_evaluations"]
            and generator.proposal_calls
            == execution_path["expected_proposal_calls"]
            and generator.reflection_calls
            == execution_path["expected_reflection_calls"]
            and replay_accounting.replay_calls == 2
            and replay_accounting.delegated_calls
            == execution_path["expected_delegated_child_evaluations"]
            and len(replay_accounting.replay_payload_identities) == 2
        )
        treatment_noncompliance_count = int(
            execution_path.get("treatment_noncompliance_count", 0)
        )
        if full_path and accounting_pass:
            if transfer_adjudication["status"] == "not_tested_noncompliance":
                status = (
                    "completed_full_seven_call_path_not_tested_noncompliance"
                )
            elif treatment_noncompliance_count:
                status = (
                    "completed_full_seven_call_path_partial_treatment_"
                    "noncompliance"
                )
            else:
                status = "completed_full_seven_call_path"
        elif clean_early_stop and accounting_pass:
            status = "completed_clean_scientific_early_stop"
        else:
            status = "completed_with_accounting_drift"
        summary = {
            "schema_version": 1,
            "status": status,
            "run_id": verified.run_id,
            "manifest_sha256": verified.manifest_sha256,
            "source_sha256": verified.source_sha256,
            "result_hash": result.result_hash,
            "generation_widths": widths,
            "logical_llm_calls": result.final_state.logical_llm_calls,
            "unique_evaluations": result.final_state.unique_evaluations,
            "proposal_calls": generator.proposal_calls,
            "reflection_calls": generator.reflection_calls,
            "credentials_read": generator.credentials_read,
            "seed_receipt_replay_calls": replay_accounting.replay_calls,
            "delegated_child_cfd_candidate_evaluations": (
                replay_accounting.delegated_calls
            ),
            "seed_projection_replay_identities": dict(
                sorted(replay_accounting.replay_payload_identities.items())
            ),
            "provider_accounting": provider_accounting,
            "execution_path_adjudication": execution_path,
            "held_out_assignment": assignment_commitment_record,
            "g2_prequeue_gate": gate_record,
            "transfer_adjudication": transfer_adjudication,
            "development_decision": transfer_adjudication[
                "development_decision"
            ],
            "full_path": full_path,
            "clean_scientific_early_stop": clean_early_stop,
            "early_stop_reason_code": early_stop_reason_code,
            "early_stop_reason": early_stop_reason,
            "accounting_pass": accounting_pass,
            "claim_boundary": verified.record["launch"]["claim_boundary"],
        }
        write_json_atomic(run_dir / "summary.json", summary)
        if not accounting_pass:
            raise RuntimeError("live execution completed with accounting drift")
    except BaseException as exc:
        pending = exc
        try:
            write_json_atomic(
                run_dir / "failure.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "failure_type": type(exc).__name__,
                    "safe_message": str(exc)[:1_024],
                },
            )
        except BaseException as artifact_exc:
            exc.add_note(
                "failure artifact publication also failed: "
                f"{type(artifact_exc).__name__}"
            )
    finally:
        for writer in reversed(tuple(writers.values())):
            try:
                writer.close()
            except BaseException as exc:
                if pending is None:
                    pending = exc
                else:
                    pending.add_note(
                        f"JSONL close also failed: {type(exc).__name__}"
                    )
        if resource_lease is not None and resource_lease.active:
            try:
                release = resource_lease.release(
                    outcome=status if pending is None else "failed",
                    failure_type=(
                        None if pending is None else type(pending).__name__
                    ),
                )
                write_json_atomic(
                    run_dir / "resource_lease_released.json",
                    {
                        "schema_version": 1,
                        "phase": "provider_evolution",
                        "release": release,
                    },
                )
            except BaseException as exc:
                if pending is None:
                    pending = exc
                else:
                    pending.add_note(
                        "resource lease release/evidence also failed: "
                        f"{type(exc).__name__}"
                    )
        try:
            _directory_fsync(run_dir)
            decision = "invalid_block"
            if pending is None and summary is not None:
                predecision = summary.get("development_decision")
                decision = (
                    "advance_to_replication"
                    if predecision
                    == "advance_after_successful_evidence_tree_finalization"
                    else "do_not_advance"
                )
            _finalize_run(
                run_dir,
                status=status,
                post_finalization_decision=decision,
            )
        except BaseException as exc:
            if pending is None:
                pending = exc
            else:
                pending.add_note(f"run finalization also failed: {type(exc).__name__}")
    if pending is not None:
        raise pending
    if summary is None:
        raise RuntimeError("live execution returned without a summary")
    return summary


__all__ = [
    "AirfoilV7G2PrequeueGatePolicy",
    "DEFAULT_LIVE_LOG_ROOT",
    "DeferredJournaledLiveGenerator",
    "DurableJsonlWriter",
    "G2PrequeueGateError",
    "G2PrequeueGatePolicy",
    "LiveExecutionDependencies",
    "VerifiedLaunchManifest",
    "build_launch_manifest_record",
    "create_real_benchmark",
    "execute_live_with_dependencies",
    "materialize_prompt_readiness",
    "materialize_prompt_readiness_sync",
    "production_live_dependencies",
    "prompt_preflight",
    "reverify_launch_source",
    "source_snapshot",
    "verify_launch_manifest",
    "write_launch_manifest",
]
