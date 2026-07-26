"""Durable launcher for the complete Airfoil v10 multi-option evolution.

The launcher is deliberately a thin experiment boundary.  Airfoil-specific
input preparation and generic evolutionary policy remain in their respective
composition modules.  This module owns only run isolation, provider selection,
lazy credential access, durable journals, resource leasing, terminal
validation, and a self-contained result projection.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from agent_evolve.agentic import ExclusiveResourceLease
from agent_evolve.application.agentic_evolution import (
    EvolutionCandidate,
    OperatorKind,
    ProposalAuthority,
)
from agent_evolve.application.budgeted_optimizer import (
    OptimizerResult,
    OptimizerStopReason,
    validate_optimizer_result_integrity,
)
from agent_evolve.application.multi_option_evolution import (
    MULTI_OPTION_EVOLUTION_BUDGET,
    MULTI_OPTION_G1_SLOT_IDS,
    MULTI_OPTION_G2_SLOT_IDS,
    MULTI_OPTION_G3_CORE_SLOT_IDS,
    MULTI_OPTION_G3_CROSSOVER_SLOT_IDS,
    MULTI_OPTION_G3_SLOT_IDS,
    MultiOptionEvolutionPlanner,
)
from agent_evolve.application.outcome_relation import OutcomeRelation
from agent_evolve.application.live_runtime_manifest import (
    LiveRuntimeManifest,
    verify_runtime_source_closure,
)
from agent_evolve.application.post_evolution_reflection import (
    PostEvolutionReflectionInterceptor,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    thaw_json,
    typed_json_equal,
    typed_json_sha256,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    structured_generation_outcome_record,
)
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (
    validate_openrouter_outbound_request_manifest_record,
)
from agent_evolve.integrations.pydantic_ai.provider_attempt_join import (
    build_provider_attempt_terminal_join_receipt,
)
from agent_evolve.policies.variation.exact_parent_crossover import (
    derive_exact_parent_crossover_contract,
    exact_parent_import_exclusions_sha256,
    materialize_exact_parent_crossover,
    resolve_exact_parent_import_for_target,
    validate_exact_parent_import_exclusions,
)
from agent_evolve.ports.structured_generator import StructuredStreamProgress
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    ConvergenceQualifiedAirfoilPanelProblem,
    local_default_converged_settings,
)
from examples.benchmarks.engibench_airfoil.problem_def import (
    EXPECTED_DATASET_SHA256,
)
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    CONTAINER_IMAGE,
    DEFAULT_RESOURCE_LEASE_PATH,
    DEEPSEEK_G3_PROVIDER_PROFILE,
    GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE,
    AirfoilG3ProviderProfile,
    resolve_airfoil_g3_provider_profile,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import AirfoilV7Problem
from examples.benchmarks.engibench_airfoil.v7_readiness import (
    AirfoilV7ReadinessSpec,
    create_airfoil_v7_resource_lease,
)
from examples.benchmarks.engibench_airfoil.v8_stage_b_live import (
    AGENT_EVOLVE_ROOT,
    RESEARCH_ARTIFACT_ROOT,
)
from examples.benchmarks.engibench_airfoil.v10_multi_option_inputs import (
    AirfoilV10MultiOptionInputs,
    airfoil_v10_multi_option_readiness_record,
    load_frozen_airfoil_v10_multi_option_inputs,
)
from examples.benchmarks.engibench_airfoil.v10_multi_option_live import (
    AIRFOIL_V10_ALLOWED_PROVIDER_PROFILES,
    AirfoilV10MultiOptionLiveComposition,
    build_airfoil_v10_openrouter_config,
    compose_airfoil_v10_multi_option_live,
)
from examples.benchmarks.engibench_airfoil.v10_runtime_manifest import (
    AIRFOIL_V10_RUNTIME_MANIFEST_FILENAME,
    AIRFOIL_V10_RUNTIME_MANIFEST_ID,
    AIRFOIL_V10_RUNTIME_MANIFEST_VERSION,
    FrozenAirfoilV10RuntimeManifestGate,
    build_airfoil_v10_runtime_manifest,
    capture_airfoil_v10_runtime_source_closure,
    runtime_manifest_identity_record,
)
from examples.benchmarks.engibench_airfoil.v10_qualification import (
    airfoil_v10_provider_configuration_sha256,
    record_airfoil_v10_qualification,
    verify_airfoil_v10_qualification_directory,
)
from examples.development.durable_run_artifacts import (
    BatchedDurableJsonlJournal,
    DurableJsonlJournal,
    finalize_run_directory,
    read_jsonl,
    write_json_atomic,
)


LIVE_AUTHORIZATION = "AIRFOIL_V10_MULTI_OPTION_EVOLUTION_LIVE_V1"
RUN_ROOT = RESEARCH_ARTIFACT_ROOT / "experiment_logs" / "airfoil_v10_multi_option"
WORK_ROOT = Path("/tmp/agent_evolve_airfoil_v10_multi_option")
DEFAULT_PROVIDER_PROFILE_ID = DEEPSEEK_G3_PROVIDER_PROFILE.profile_id
GPT_XHIGH_PROVIDER_PROFILE_ID = GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE.profile_id

_PRODUCTION_EXECUTION_AUTHORITY = object()
_DEVELOPMENT_EXECUTION_AUTHORITY = object()
_PRODUCTION_READINESS_AUTHORITY = object()
_DEVELOPMENT_READINESS_AUTHORITY = object()

_READINESS_COMMITMENT_DOMAIN = b"agent-evolve:airfoil-v10-readiness:v1\x00"

_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_EXPECTED_SLOT_IDS = (
    MULTI_OPTION_G1_SLOT_IDS,
    MULTI_OPTION_G2_SLOT_IDS,
    MULTI_OPTION_G3_SLOT_IDS,
)
_EXPECTED_CANDIDATE_OCCURRENCES = 2 + sum(map(len, _EXPECTED_SLOT_IDS))


class AirfoilV10MultiOptionRunnerError(RuntimeError):
    """A durable v10 run failed; its finalized directory retains evidence."""


def _validate_run_id(value: str) -> str:
    if type(value) is not str or _RUN_ID.fullmatch(value) is None:
        raise AirfoilV10MultiOptionRunnerError("run_id uses an invalid grammar")
    return value


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_seconds() -> str:
    """Return the whole-second timestamp required by runtime manifests."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_dotenv_api_key() -> str:
    """Read only ``OPENROUTER_API_KEY`` and never persist its value."""

    path = AGENT_EVOLVE_ROOT.parent / ".env"
    value: str | None = None
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, candidate = line.split("=", 1)
            if name.strip() == "OPENROUTER_API_KEY":
                value = candidate.strip().strip('"').strip("'")
                break
    if not value:
        value = os.environ.get("OPENROUTER_API_KEY")
    if type(value) is not str or not value:
        raise AirfoilV10MultiOptionRunnerError("OPENROUTER_API_KEY is unavailable")
    return value


def _readiness_spec() -> AirfoilV7ReadinessSpec:
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


def _lease(run_id: str) -> ExclusiveResourceLease:
    return create_airfoil_v7_resource_lease(
        _readiness_spec(),
        lease_path=DEFAULT_RESOURCE_LEASE_PATH,
        run_id=run_id,
        phase="airfoil_v10_multi_option_evolution",
    )


def _problem(
    run_id: str,
    run_dir: Path,
    work_root: Path = WORK_ROOT,
) -> tuple[AirfoilV7Problem, ConvergenceQualifiedAirfoilPanelProblem]:
    settings = replace(
        local_default_converged_settings(),
        output_root=run_dir / "cfd_receipts",
        work_root=work_root / run_id,
    )
    raw = ConvergenceQualifiedAirfoilPanelProblem(settings)
    return AirfoilV7Problem(raw_problem=raw), raw


def _resolve_profile(profile_id: str) -> AirfoilG3ProviderProfile:
    profile = resolve_airfoil_g3_provider_profile(profile_id)
    if profile not in AIRFOIL_V10_ALLOWED_PROVIDER_PROFILES:
        raise AirfoilV10MultiOptionRunnerError(
            "provider profile is registered but not admitted by Airfoil v10"
        )
    return profile


def _forbidden_reasoning_paths(
    value: object,
    *,
    path: str = "$",
) -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str:
                raise AirfoilV10MultiOptionRunnerError(
                    "provider manifest contains a non-string key"
                )
            if key.casefold() in {"mode", "pro"}:
                found.append(f"{path}.{key}")
            found.extend(_forbidden_reasoning_paths(child, path=f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_forbidden_reasoning_paths(child, path=f"{path}[{index}]"))
    return tuple(found)


def airfoil_v10_provider_config_record(
    profile: AirfoilG3ProviderProfile,
) -> dict[str, object]:
    """Return the public, secret-free provider projection shared by v10 gates."""

    config = build_airfoil_v10_openrouter_config(profile)
    transport = config.to_manifest_record()
    reasoning = transport["reasoning"]
    record: dict[str, object] = {
        "schema_version": 1,
        "profile_id": profile.profile_id,
        "requested_model": profile.model_alias,
        "canonical_model": profile.canonical_model,
        "provider_slug": profile.provider_slug,
        "resolved_provider": profile.resolved_provider,
        "max_input_tokens": profile.max_input_tokens,
        "max_output_tokens": profile.max_output_tokens,
        "max_reasoning_tokens": profile.max_reasoning_tokens,
        "temperature": profile.temperature,
        "reasoning": reasoning,
        "reasoning_mode_or_pro_fields_absent": True,
        "artificial_output_cap": False,
        "transport": transport,
    }
    forbidden = _forbidden_reasoning_paths(record)
    if forbidden:
        raise AirfoilV10MultiOptionRunnerError(
            "provider configuration contains reasoning mode/pro fields"
        )
    if profile is GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE and reasoning != {
        "effort": "xhigh"
    }:
        raise AirfoilV10MultiOptionRunnerError(
            "GPT Sol profile must request xhigh effort and no reasoning mode"
        )
    return record


# Compatibility alias for existing qualification tests and downstream research
# scripts. New callers should use the public projection above.
_provider_config_record = airfoil_v10_provider_config_record


def airfoil_v10_expected_outbound_transport_settings(
    profile: AirfoilG3ProviderProfile,
) -> dict[str, object]:
    """Project the selected profile onto exact HTTP manifest setting fields."""

    config = build_airfoil_v10_openrouter_config(profile)
    return {
        "model": config.model_name,
        "provider": config.provider_options,
        "reasoning": (
            None
            if config.reasoning_config is None
            else config.reasoning_config.to_model_setting()
        ),
        "usage": {"include": True},
        "stream": True,
        "stream_options": {"include_usage": True},
        "tool_choice": "required",
        "response_format": None,
    }


def _qualification_framework_versions(
    qualification_record: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if qualification_record is None:
        return None
    installed = qualification_record.get("installed_distributions")
    expected_names = ("httpx", "openai", "pydantic", "pydantic-ai")
    if type(installed) is not dict or any(
        type(installed.get(name)) is not str or not installed[name]
        for name in expected_names
    ):
        raise AirfoilV10MultiOptionRunnerError(
            "qualification lacks exact outbound framework versions"
        )
    return {name: installed[name] for name in expected_names}


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


def _telemetry_record(value: object | None) -> dict[str, object] | None:
    if value is None:
        return None
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


def _candidate_record(candidate: EvolutionCandidate | None) -> object:
    if candidate is None:
        return None
    candidate.__post_init__()
    detailed = candidate.detailed_evaluation
    return {
        "candidate_id": candidate.candidate_id.value,
        "configuration_sha256": candidate.occurrence.configuration_hash,
        "configuration": thaw_json(candidate.configuration),
        "generation": candidate.generation,
        "label": candidate.label,
        "operator_kind": (
            None if candidate.operator_kind is None else candidate.operator_kind.value
        ),
        "parent_candidate_ids": [value.value for value in candidate.parent_ids],
        "common_ancestor_candidate_id": (
            None
            if candidate.common_ancestor_id is None
            else candidate.common_ancestor_id.value
        ),
        "valid": candidate.valid,
        "objectives_hex": {name: value.hex() for name, value in candidate.objectives},
        "operator_compliant": candidate.operator_compliant,
        "operator_failure": candidate.operator_failure,
        "evidence_compliant": candidate.evidence_compliant,
        "evidence_failure": candidate.evidence_failure,
        "parent_patch_sha256s": list(candidate.parent_patch_hashes),
        "preservation_verified": candidate.preservation_verified,
        "selected_insights": [
            {
                "insight_id": value.insight_id.value,
                "version": value.version,
            }
            for value in candidate.selected_insight_refs
        ],
        "source_attribution": [
            {"path": value.path, "source": value.source}
            for value in candidate.source_attribution
        ],
        "design_rationale": candidate.design_rationale,
        "call_telemetry": _telemetry_record(candidate.call_telemetry),
        "detailed_evaluation": (
            None
            if detailed is None
            else {
                "evidence_sha256": detailed.evidence_sha256,
                "phenotype_identity_sha256": detailed.phenotype.identity_sha256,
                "violations_hex": {
                    name: value.hex() for name, value in detailed.violations
                },
                "timings": detailed.timings.to_record(),
                "receipt": (
                    None
                    if detailed.receipt is None
                    else {
                        "artifact_id": detailed.receipt.artifact_id.value,
                        "sha256": detailed.receipt.sha256_hex,
                        "size_bytes": detailed.receipt.size_bytes,
                    }
                ),
            }
        ),
    }


def _verified_better_relation_any_parent(
    engine_trace_rows: tuple[dict[str, object], ...],
    *,
    operator_invocation_id: str,
    parent_candidate_ids: tuple[str, ...],
) -> bool:
    """Re-derive the parent-relative quality flag from engine relation evidence."""

    completed_rows = tuple(
        row
        for row in engine_trace_rows
        if row.get("event_type") == "invocation_completed"
        and row.get("operator_invocation_id") == operator_invocation_id
    )
    if len(completed_rows) != 1:
        raise AirfoilV10MultiOptionRunnerError(
            "slot outcome must join exactly one completed invocation"
        )
    completed = completed_rows[0]
    raw_relations = completed.get("parent_outcome_relations")
    if (
        completed.get("parent_ids") != list(parent_candidate_ids)
        or type(raw_relations) is not list
        or len(raw_relations) != len(parent_candidate_ids)
    ):
        raise AirfoilV10MultiOptionRunnerError(
            "completed invocation parent-relation evidence is incomplete"
        )
    relations: list[OutcomeRelation] = []
    for parent_candidate_id, relation_record in zip(
        parent_candidate_ids,
        raw_relations,
        strict=True,
    ):
        if (
            type(relation_record) is not dict
            or set(relation_record) != {"parent_candidate_id", "candidate_relation"}
            or relation_record.get("parent_candidate_id") != parent_candidate_id
            or type(relation_record.get("candidate_relation")) is not str
        ):
            raise AirfoilV10MultiOptionRunnerError(
                "completed invocation parent relation is malformed"
            )
        try:
            relations.append(OutcomeRelation(relation_record["candidate_relation"]))
        except ValueError as exc:
            raise AirfoilV10MultiOptionRunnerError(
                "completed invocation parent relation is unknown"
            ) from exc
    derived = OutcomeRelation.BETTER in relations
    if completed.get("better_relation_any_parent") is not derived:
        raise AirfoilV10MultiOptionRunnerError(
            "completed invocation better-parent relation flag is inconsistent"
        )
    return derived


def _slot_record(
    value: object,
    engine_trace_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    outcome = value.outcome
    prepared = outcome.prepared
    plan = value.slot.plan
    parent_candidate_ids = tuple(parent.candidate_id.value for parent in plan.parents)
    better_relation_any_parent = _verified_better_relation_any_parent(
        engine_trace_rows,
        operator_invocation_id=prepared.operator_invocation_id.value,
        parent_candidate_ids=parent_candidate_ids,
    )
    decision = outcome.finite_action_decision
    return {
        "slot_id": value.slot.slot_id,
        "role": value.slot.role,
        "proposal_authority": value.slot.proposal_authority.value,
        "operator_invocation_id": prepared.operator_invocation_id.value,
        "llm_call_id": None if prepared.call_id is None else prepared.call_id.value,
        "proposal_sequence": prepared.proposal_sequence,
        "operator_kind": plan.operator_kind.value,
        "phase": plan.phase,
        "parent_candidate_ids": list(parent_candidate_ids),
        "common_ancestor_candidate_id": (
            None
            if plan.common_ancestor is None
            else plan.common_ancestor.candidate_id.value
        ),
        "reward_hex": outcome.reward.hex(),
        "failure_stage": outcome.failure_stage,
        "call_failure_type": outcome.call_failure_type,
        "dominates_any_parent": outcome.dominates_any_parent,
        "better_than_any_parent": better_relation_any_parent,
        "finite_action_decision": (
            None
            if decision is None
            else {**decision.to_record(), "decision_sha256": decision.decision_sha256}
        ),
        "candidate": _candidate_record(outcome.candidate),
    }


def _memory_record(planner: MultiOptionEvolutionPlanner) -> dict[str, object]:
    closure = planner.closure
    if planner.wave is None or planner.genesis is None or closure is None:
        raise AirfoilV10MultiOptionRunnerError(
            "planner lacks its sealed G1 memory wave"
        )
    snapshot = closure.snapshot
    if snapshot is None:
        raise AirfoilV10MultiOptionRunnerError("planner lacks its G1 score checkpoint")
    ranking = sorted(
        snapshot.entries,
        key=lambda value: (
            -value.retrieval_score,
            value.reference.insight_id.value,
            value.reference.version,
        ),
    )
    return {
        "wave": {**planner.wave.to_record(), "wave_sha256": planner.wave.wave_sha256},
        "genesis": {
            **planner.genesis.to_record(),
            "snapshot_sha256": planner.genesis.snapshot_sha256,
        },
        "closure": {
            "status": closure.status.value,
            "wave_sha256": closure.wave_sha256,
            "receipts": [value.to_record() for value in closure.receipts],
            "observations": [value.to_record() for value in closure.observations],
            "snapshot": {
                **snapshot.to_record(),
                "snapshot_sha256": snapshot.snapshot_sha256,
            },
        },
        "ranking": [
            {"rank": index, **value.to_record()}
            for index, value in enumerate(ranking, start=1)
        ],
        "g1_assignments": [value.to_record() for value in planner.g1_assignments],
        "g2_assignments": [value.to_record() for value in planner.g2_assignments],
        "adaptive_reference": {
            "insight_id": planner.adaptive_reference.insight_id.value,
            "version": planner.adaptive_reference.version,
        },
    }


def _provider_summary(
    rows: tuple[dict[str, object], ...],
    profile: AirfoilG3ProviderProfile,
) -> dict[str, object]:
    if len(rows) != MULTI_OPTION_EVOLUTION_BUDGET.max_logical_llm_calls:
        raise AirfoilV10MultiOptionRunnerError(
            "provider outcome journal does not contain the exact seven calls"
        )
    responses: list[dict[str, object]] = []
    for row in rows:
        if row.get("status") != "succeeded" or type(row.get("response")) is not dict:
            raise AirfoilV10MultiOptionRunnerError(
                "completed chronology contains an unsuccessful provider call"
            )
        response = row["response"]
        assert type(response) is dict
        if (
            response.get("requested_model") != profile.model_alias
            or response.get("resolved_model") not in profile.allowed_resolved_models
            or response.get("resolved_provider") != profile.resolved_provider
        ):
            raise AirfoilV10MultiOptionRunnerError(
                "provider outcome escaped its selected route"
            )
        responses.append(response)
    reasoning_tokens = [value.get("reasoning_tokens") for value in responses]
    reasoning_verified = all(
        type(value) is int and value > 0 for value in reasoning_tokens
    )
    if profile is GPT56_SOL_AZURE_XHIGH_PROVIDER_PROFILE and not reasoning_verified:
        raise AirfoilV10MultiOptionRunnerError(
            "GPT Sol xhigh run lacks positive reasoning-token evidence for every call"
        )
    costs = [
        Decimal(str(value["cost_usd"]))
        for value in responses
        if value.get("cost_usd") is not None
    ]
    return {
        "logical_provider_calls": len(rows),
        "all_succeeded": True,
        "reasoning_tokens_by_call": reasoning_tokens,
        "positive_reasoning_tokens_every_call": reasoning_verified,
        "input_tokens_total": sum(
            int(value.get("input_tokens") or 0) for value in responses
        ),
        "output_tokens_total": sum(
            int(value.get("output_tokens") or 0) for value in responses
        ),
        "reasoning_tokens_total": sum(int(value or 0) for value in reasoning_tokens),
        "reported_cost_usd_total": None if not costs else str(sum(costs)),
        "outcomes": list(rows),
    }


def _verified_exact_parent_import_record(
    *,
    invocation: dict[str, object],
    evaluated: dict[str, object],
    slot_id: str,
    operator_invocation_id: str,
    llm_call_id: str,
    candidate_id: str,
    proposal_sequence: int,
    parent_candidate_ids: tuple[str, ...],
    parent_configuration_sha256s: tuple[str, ...],
    parent_configurations: tuple[FrozenJsonObject, FrozenJsonObject],
    known_target_configurations: tuple[FrozenJsonObject, ...],
    configuration: FrozenJsonObject,
    configuration_sha256: str,
    fail: Callable[[str], None],
    is_sha256: Callable[[object], bool],
) -> dict[str, object]:
    """Verify the bounded parent-import contract without candidate payloads."""

    policy = "bounded_exact_parent_import_v1"
    contract = invocation.get("exact_parent_crossover_contract")
    contract_sha256 = invocation.get("exact_parent_crossover_contract_sha256")
    if (
        type(contract) is not dict
        or set(contract)
        != {
            "schema_version",
            "policy",
            "max_loci",
            "base_parent_sha256",
            "donor_parent_sha256",
            "loci",
        }
        or contract.get("schema_version") != 1
        or contract.get("policy") != policy
        or type(contract.get("max_loci")) is not int
        or not 2 <= contract["max_loci"] <= 4096
        or not is_sha256(contract.get("base_parent_sha256"))
        or not is_sha256(contract.get("donor_parent_sha256"))
        or not is_sha256(contract_sha256)
    ):
        fail("exact parent crossover contract is malformed")
    encoded_contract = json.dumps(
        contract,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")
    recomputed_contract_sha256 = hashlib.sha256(
        b"agent-evolve:exact-parent-crossover-contract:v1\x00" + encoded_contract
    ).hexdigest()
    if contract_sha256 != recomputed_contract_sha256:
        fail("exact parent crossover contract digest does not verify")
    if (
        type(parent_configuration_sha256s) is not tuple
        or len(parent_configuration_sha256s) != 2
        or any(not is_sha256(value) for value in parent_configuration_sha256s)
        or parent_configuration_sha256s[0] == parent_configuration_sha256s[1]
        or contract["base_parent_sha256"] != parent_configuration_sha256s[0]
        or contract["donor_parent_sha256"] != parent_configuration_sha256s[1]
    ):
        fail("exact crossover contract is not bound to its lineage parents")
    if (
        type(parent_configurations) is not tuple
        or len(parent_configurations) != 2
        or any(type(value) is not FrozenJsonObject for value in parent_configurations)
        or type(configuration) is not FrozenJsonObject
    ):
        fail("exact crossover replay configurations are absent")
    try:
        actual_parent_sha256s = tuple(
            typed_json_sha256(value) for value in parent_configurations
        )
        actual_configuration_sha256 = typed_json_sha256(configuration)
    except (TypeError, ValueError) as exc:
        fail(f"exact crossover configuration hashing failed: {type(exc).__name__}")
    if actual_parent_sha256s != parent_configuration_sha256s:
        fail("exact crossover parent configuration hashes do not verify")
    if actual_configuration_sha256 != configuration_sha256:
        fail("exact crossover child configuration hash does not verify")
    try:
        rederived_contract = derive_exact_parent_crossover_contract(
            base=parent_configurations[0],
            donor=parent_configurations[1],
            max_loci=contract["max_loci"],
        )
    except (TypeError, ValueError) as exc:
        fail(f"exact crossover contract rederivation failed: {type(exc).__name__}")
    if (
        rederived_contract.to_record() != contract
        or rederived_contract.contract_sha256 != contract_sha256
    ):
        fail("exact crossover contract differs from its lineage parents")

    raw_forbidden = invocation.get("forbidden_exact_parent_import_sets")
    invocation_exclusions_sha256 = invocation.get(
        "exact_parent_import_exclusions_sha256"
    )
    if (
        type(raw_forbidden) is not list
        or not raw_forbidden
        or any(type(value) is not list for value in raw_forbidden)
        or not is_sha256(invocation_exclusions_sha256)
    ):
        fail("exact crossover known-child exclusions are absent or malformed")
    forbidden_import_locus_sets = tuple(tuple(value) for value in raw_forbidden)
    try:
        validate_exact_parent_import_exclusions(
            rederived_contract,
            forbidden_import_locus_sets,
        )
        recomputed_exclusions_sha256 = exact_parent_import_exclusions_sha256(
            rederived_contract,
            forbidden_import_locus_sets,
        )
    except (TypeError, ValueError) as exc:
        fail(f"exact crossover exclusion validation failed: {type(exc).__name__}")
    if invocation_exclusions_sha256 != recomputed_exclusions_sha256:
        fail("exact crossover exclusion digest does not verify")
    if (
        type(known_target_configurations) is not tuple
        or not known_target_configurations
        or any(
            type(value) is not FrozenJsonObject for value in known_target_configurations
        )
    ):
        fail("exact crossover known-target configurations are absent")
    try:
        known_target_configuration_sha256s = tuple(
            typed_json_sha256(value) for value in known_target_configurations
        )
    except (TypeError, ValueError) as exc:
        fail(f"exact crossover known-target hashing failed: {type(exc).__name__}")
    if known_target_configuration_sha256s != tuple(
        sorted(set(known_target_configuration_sha256s))
    ):
        fail("exact crossover known targets are not canonical and unique")
    try:
        expected_forbidden_import_locus_sets = tuple(
            sorted(
                {
                    resolved
                    for target in known_target_configurations
                    if (
                        resolved := resolve_exact_parent_import_for_target(
                            base=parent_configurations[0],
                            donor=parent_configurations[1],
                            contract=rederived_contract,
                            target=target,
                        )
                    )
                    is not None
                }
            )
        )
    except (TypeError, ValueError) as exc:
        fail(f"exact crossover known-target inversion failed: {type(exc).__name__}")
    if forbidden_import_locus_sets != expected_forbidden_import_locus_sets:
        fail("exact crossover exclusions are incomplete for the known targets")

    loci = contract.get("loci")
    if type(loci) is not list or not 2 <= len(loci) <= contract["max_loci"]:
        fail("exact parent crossover locus catalog is malformed")
    expected_locus_ids = [f"locus_{index:04d}" for index in range(1, len(loci) + 1)]
    for locus, expected_locus_id in zip(loci, expected_locus_ids, strict=True):
        if (
            type(locus) is not dict
            or set(locus)
            != {
                "locus_id",
                "path_text",
                "path_schema_identity",
                "base_value_sha256",
                "donor_value_sha256",
            }
            or locus.get("locus_id") != expected_locus_id
            or type(locus.get("path_text")) is not str
            or not locus["path_text"].startswith("$")
            or not is_sha256(locus.get("path_schema_identity"))
            or not is_sha256(locus.get("base_value_sha256"))
            or not is_sha256(locus.get("donor_value_sha256"))
            or locus["base_value_sha256"] == locus["donor_value_sha256"]
        ):
            fail("exact parent crossover locus evidence is malformed")

    import_ids = evaluated.get("crossover_import_locus_ids")
    if (
        type(import_ids) is not list
        or not 1 <= len(import_ids) < len(loci)
        or len(set(import_ids)) != len(import_ids)
        or import_ids
        != [value for value in expected_locus_ids if value in set(import_ids)]
    ):
        fail("exact parent crossover selection is not a proper canonical subset")
    if tuple(import_ids) in forbidden_import_locus_sets:
        fail("exact parent crossover selected a forbidden known child")
    if (
        evaluated.get("crossover_forbidden_import_locus_sets") != raw_forbidden
        or evaluated.get("crossover_import_exclusions_sha256")
        != recomputed_exclusions_sha256
    ):
        fail("exact crossover evaluated exclusions differ from invocation admission")
    plan_record = {
        "schema_version": 1,
        "policy": policy,
        "contract_sha256": contract_sha256,
        "locus_count": len(loci),
        "import_locus_ids": import_ids,
    }
    plan_sha256 = hashlib.sha256(
        b"agent-evolve:exact-parent-crossover-plan:v1\x00"
        + json.dumps(
            plan_record,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii", errors="strict")
    ).hexdigest()
    if evaluated.get("crossover_plan_sha256") != plan_sha256:
        fail("exact parent crossover plan digest does not verify")
    try:
        replayed_materialization = materialize_exact_parent_crossover(
            base=parent_configurations[0],
            donor=parent_configurations[1],
            contract=rederived_contract,
            import_locus_ids=tuple(import_ids),
        )
    except (TypeError, ValueError) as exc:
        fail(f"exact crossover replay failed: {type(exc).__name__}")
    if (
        not typed_json_equal(replayed_materialization.configuration, configuration)
        or replayed_materialization.materialized_configuration_sha256
        != configuration_sha256
    ):
        fail("exact crossover child does not replay from its lineage parents")
    replayed_source_attribution = [
        {
            "path": attribution.path_text,
            "source": "left" if attribution.source.value == "base" else "right",
        }
        for attribution in replayed_materialization.attributions
    ]
    observed_source_attribution = _exact_source_attribution_semantics(
        evaluated.get("source_attribution")
    )
    replayed_source_attribution_semantics = _exact_source_attribution_semantics(
        replayed_source_attribution
    )
    if (
        observed_source_attribution is None
        or replayed_source_attribution_semantics is None
        or observed_source_attribution != replayed_source_attribution_semantics
    ):
        fail("exact crossover source attribution differs from core replay")

    materialization = evaluated.get("crossover_materialization")
    if (
        type(materialization) is not dict
        or set(materialization)
        != {
            "schema_version",
            "policy",
            "contract_sha256",
            "plan_sha256",
            "materialized_configuration_sha256",
            "attributions",
        }
        or materialization.get("schema_version") != 1
        or materialization.get("policy") != policy
        or materialization.get("contract_sha256") != contract_sha256
        or materialization.get("plan_sha256") != plan_sha256
        or materialization.get("materialized_configuration_sha256")
        != configuration_sha256
    ):
        fail("exact parent crossover materialization is malformed")
    attributions = materialization.get("attributions")
    if type(attributions) is not list or len(attributions) != len(loci):
        fail("exact parent crossover attribution is not exhaustive")
    attribution_keys = {
        "locus_id",
        "path_text",
        "source",
        "base_value_sha256",
        "donor_value_sha256",
        "source_value_sha256",
        "materialized_value_sha256",
    }
    observed_sources: set[str] = set()
    for attribution, locus in zip(attributions, loci, strict=True):
        expected_source = "donor" if locus["locus_id"] in import_ids else "base"
        expected_source_hash = locus[
            "donor_value_sha256" if expected_source == "donor" else "base_value_sha256"
        ]
        if (
            type(attribution) is not dict
            or set(attribution) != attribution_keys
            or attribution.get("locus_id") != locus["locus_id"]
            or attribution.get("path_text") != locus["path_text"]
            or attribution.get("source") != expected_source
            or attribution.get("base_value_sha256") != locus["base_value_sha256"]
            or attribution.get("donor_value_sha256") != locus["donor_value_sha256"]
            or attribution.get("source_value_sha256") != expected_source_hash
            or attribution.get("materialized_value_sha256") != expected_source_hash
        ):
            fail("exact parent crossover attribution evidence is inconsistent")
        observed_sources.add(expected_source)
    if observed_sources != {"base", "donor"}:
        fail("exact parent crossover lacks both parent sources")

    encoded_materialization = json.dumps(
        materialization,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")
    materialization_sha256 = hashlib.sha256(
        b"agent-evolve:exact-parent-crossover-materialization:v1\x00"
        + encoded_materialization
    ).hexdigest()
    if (
        evaluated.get("crossover_materialization_sha256") != materialization_sha256
        or replayed_materialization.to_record() != materialization
        or replayed_materialization.materialization_sha256 != materialization_sha256
    ):
        fail("exact parent materialization digest does not verify")

    receipt = evaluated.get("crossover_materialization_receipt")
    if (
        type(receipt) is not dict
        or set(receipt)
        != {
            "schema_version",
            "policy",
            "max_loci",
            "base_parent_sha256",
            "donor_parent_sha256",
            "contract_sha256",
            "plan_sha256",
            "import_locus_ids",
            "materialized_configuration_sha256",
            "materialization_sha256",
            "attributions",
        }
        or receipt.get("schema_version") != 1
        or receipt.get("policy") != policy
        or receipt.get("max_loci") != contract["max_loci"]
        or receipt.get("base_parent_sha256") != contract["base_parent_sha256"]
        or receipt.get("donor_parent_sha256") != contract["donor_parent_sha256"]
        or receipt.get("contract_sha256") != contract_sha256
        or receipt.get("plan_sha256") != plan_sha256
        or receipt.get("import_locus_ids") != import_ids
        or receipt.get("materialized_configuration_sha256") != configuration_sha256
        or receipt.get("materialization_sha256") != materialization_sha256
        or receipt.get("attributions") != attributions
    ):
        fail("exact parent crossover receipt is inconsistent")
    receipt_sha256 = hashlib.sha256(
        b"agent-evolve:exact-parent-crossover-receipt:v1\x00"
        + json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii", errors="strict")
    ).hexdigest()
    if (
        evaluated.get("crossover_materialization_receipt_sha256") != receipt_sha256
        or replayed_materialization.receipt.to_record() != receipt
        or replayed_materialization.receipt.receipt_sha256 != receipt_sha256
    ):
        fail("exact parent crossover receipt digest does not verify")
    if (
        evaluated.get("crossover_contract") != contract
        or evaluated.get("crossover_contract_sha256") != contract_sha256
        or evaluated.get("crossover_materialized_configuration_hash")
        != configuration_sha256
        or evaluated.get("crossover_base_parent_candidate_id")
        != parent_candidate_ids[0]
        or evaluated.get("crossover_donor_parent_candidate_id")
        != parent_candidate_ids[1]
        or evaluated.get("source_attribution_provenance")
        != "engine_derived_exact_parent_import"
    ):
        fail("exact parent crossover trace join is inconsistent")

    return {
        "schema_version": 4,
        "evidence_protocol": "engine_exact_parent_import_trace_join_v4",
        "slot_id": slot_id,
        "operator_invocation_id": operator_invocation_id,
        "llm_call_id": llm_call_id,
        "candidate_id": candidate_id,
        "proposal_sequence": proposal_sequence,
        "parent_candidate_ids": list(parent_candidate_ids),
        "parent_configuration_sha256s": list(parent_configuration_sha256s),
        "invocation_prepared_trace_sequence": invocation["sequence"],
        "candidate_evaluated_trace_sequence": evaluated["sequence"],
        "configuration_sha256": configuration_sha256,
        "contract_sha256": contract_sha256,
        "known_target_configuration_sha256s": list(known_target_configuration_sha256s),
        "forbidden_import_locus_sets": [
            list(value) for value in forbidden_import_locus_sets
        ],
        "import_exclusions_sha256": recomputed_exclusions_sha256,
        "plan_sha256": plan_sha256,
        "materialization_sha256": materialization_sha256,
        "materialization_receipt_sha256": receipt_sha256,
        "materialization_receipt": receipt,
        "verification_facts": {
            "exact_call_operator_candidate_join": True,
            "exact_parent_identity_join": True,
            "proper_nonempty_donor_subset": True,
            "known_target_exclusions_complete": True,
            "known_target_exclusions_digest_recomputed": True,
            "selected_import_set_not_forbidden": True,
            "exact_materialized_configuration_join": True,
            "contract_plan_materialization_receipts_recomputed": True,
            "core_parent_rederivation_and_child_replay": True,
            "attribution_exhaustive_and_nonoverlapping": True,
            "both_named_parent_sources_present": True,
            "locus_count": len(loci),
            "imported_locus_count": len(import_ids),
            "retained_locus_count": len(loci) - len(import_ids),
            "known_target_count": len(known_target_configurations),
            "forbidden_import_set_count": len(forbidden_import_locus_sets),
            "model_authored_configuration_fields": 0,
            "model_authored_rationale_fields": 0,
        },
    }


def _exact_source_attribution_semantics(
    value: object,
) -> tuple[tuple[str, str], ...] | None:
    """Canonicalize exhaustive exact-crossover attribution as semantics.

    Source-attribution order is not part of candidate meaning.  New engine
    traces use exact contract order, while this projection can still verify a
    legacy trace that grouped retained and imported loci by parent.  Exact
    paths remain unique and every pair is checked against a core replay.
    """

    if type(value) is not list:
        return None
    pairs: list[tuple[str, str]] = []
    for item in value:
        if (
            type(item) is not dict
            or set(item) != {"path", "source"}
            or type(item.get("path")) is not str
            or not item["path"].startswith("$")
            or item.get("source") not in {"left", "right"}
        ):
            return None
        pairs.append((item["path"], item["source"]))
    if len({path for path, _ in pairs}) != len(pairs):
        return None
    return tuple(sorted(pairs))


def _verified_model_crossover_materialization_record(
    engine_trace_rows: tuple[dict[str, object], ...],
    *,
    slot_id: str,
    operator_invocation_id: str,
    llm_call_id: str | None,
    candidate_id: str,
    proposal_sequence: int,
    parent_candidate_ids: tuple[str, ...],
    configuration_sha256: str,
    parent_patch_sha256s: tuple[str, ...],
    source_attribution: tuple[tuple[str, str], ...],
    parent_configuration_sha256s: tuple[str, ...] | None = None,
    parent_configurations: tuple[FrozenJsonObject, FrozenJsonObject] | None = None,
    known_target_configurations: tuple[FrozenJsonObject, ...] | None = None,
    configuration: FrozenJsonObject | None = None,
) -> dict[str, object]:
    """Join one advertised model crossover to its executable engine receipt.

    The projection carries no configurations. It joins the model call to the
    reserved operator/candidate identities and then joins those identities to
    the engine's hash-only materialization receipt. Missing, duplicate,
    malformed, or mismatched evidence fails the terminal result projection.
    """

    def fail(reason: str) -> None:
        raise AirfoilV10MultiOptionRunnerError(
            f"model crossover materialization evidence failed closed: {reason}"
        )

    def is_sha256(value: object) -> bool:
        return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None

    if (
        type(slot_id) is not str
        or not slot_id
        or type(operator_invocation_id) is not str
        or not operator_invocation_id
        or type(llm_call_id) is not str
        or not llm_call_id
        or type(candidate_id) is not str
        or not candidate_id
        or type(proposal_sequence) is not int
        or proposal_sequence <= 0
        or len(parent_candidate_ids) != 2
        or len(set(parent_candidate_ids)) != 2
        or any(type(value) is not str or not value for value in parent_candidate_ids)
        or not is_sha256(configuration_sha256)
        or len(parent_patch_sha256s) != 2
        or any(not is_sha256(value) for value in parent_patch_sha256s)
        or not source_attribution
    ):
        fail("result-side crossover identity is incomplete")
    if any(
        type(path) is not str
        or not path.startswith("$")
        or source not in {"left", "right", "synthesized"}
        for path, source in source_attribution
    ) or len({path for path, _ in source_attribution}) != len(source_attribution):
        fail("result-side source attribution is malformed")
    if not {"left", "right"}.issubset({source for _, source in source_attribution}):
        fail("result-side attribution lacks both named parents")
    if type(engine_trace_rows) is not tuple or any(
        type(row) is not dict for row in engine_trace_rows
    ):
        fail("engine trace collection is not an exact tuple of records")

    invocation_rows = tuple(
        row
        for row in engine_trace_rows
        if row.get("event_type") == "invocation_prepared"
        and row.get("operator_invocation_id") == operator_invocation_id
    )
    evaluated_rows = tuple(
        row
        for row in engine_trace_rows
        if row.get("event_type") == "candidate_evaluated"
        and row.get("operator_invocation_id") == operator_invocation_id
    )
    if len(invocation_rows) != 1 or len(evaluated_rows) != 1:
        fail("operator must join to exactly one prepared and one evaluated event")
    invocation = invocation_rows[0]
    evaluated = evaluated_rows[0]
    for row in (invocation, evaluated):
        if (
            row.get("schema_version") != 1
            or row.get("source") != "engine"
            or type(row.get("sequence")) is not int
            or row["sequence"] <= 0
        ):
            fail("joined engine event envelope is malformed")
    if evaluated["sequence"] <= invocation["sequence"]:
        fail("evaluated event does not follow its prepared invocation")

    expected_invocation = {
        "call_id": llm_call_id,
        "candidate_id": candidate_id,
        "proposal_sequence": proposal_sequence,
        "operator_kind": OperatorKind.TWO_PARENT_CROSSOVER.value,
        "proposal_authority": ProposalAuthority.MODEL.value,
        "parent_ids": list(parent_candidate_ids),
    }
    if any(invocation.get(key) != value for key, value in expected_invocation.items()):
        fail("prepared event does not match the exact call/operator/candidate slot")

    expected_attribution = [
        {"path": path, "source": source} for path, source in source_attribution
    ]
    if invocation.get("crossover_response_mode") == "exact_parent_import_v1":
        if parent_configuration_sha256s is None:
            fail("exact parent configuration identities are absent")
        if (
            parent_configurations is None
            or known_target_configurations is None
            or configuration is None
        ):
            fail("exact parent replay configurations are absent")
        if (
            invocation.get("proposal_representation") != "exact_parent_import_v1"
            or evaluated.get("candidate_id") != candidate_id
            or evaluated.get("operator_compliant") is not True
            or evaluated.get("evidence_compliant") is not True
            or evaluated.get("parent_patch_hashes") != list(parent_patch_sha256s)
            or _exact_source_attribution_semantics(evaluated.get("source_attribution"))
            != _exact_source_attribution_semantics(expected_attribution)
            or evaluated.get("target_configuration_hash") != configuration_sha256
        ):
            fail("exact parent evaluated event differs from its candidate")
        return _verified_exact_parent_import_record(
            invocation=invocation,
            evaluated=evaluated,
            slot_id=slot_id,
            operator_invocation_id=operator_invocation_id,
            llm_call_id=llm_call_id,
            candidate_id=candidate_id,
            proposal_sequence=proposal_sequence,
            parent_candidate_ids=parent_candidate_ids,
            parent_configuration_sha256s=parent_configuration_sha256s,
            parent_configurations=parent_configurations,
            known_target_configurations=known_target_configurations,
            configuration=configuration,
            configuration_sha256=configuration_sha256,
            fail=fail,
            is_sha256=is_sha256,
        )
    if (
        evaluated.get("candidate_id") != candidate_id
        or evaluated.get("operator_compliant") is not True
        or evaluated.get("evidence_compliant") is not True
        or evaluated.get("parent_patch_hashes") != list(parent_patch_sha256s)
        or evaluated.get("source_attribution") != expected_attribution
        or evaluated.get("source_attribution_provenance")
        != "engine_materialized_from_model_inheritance_plan"
        or evaluated.get("target_configuration_hash") != configuration_sha256
        or evaluated.get("crossover_materialized_configuration_hash")
        != configuration_sha256
    ):
        fail("evaluated event does not match the advertised candidate")

    materialization = evaluated.get("crossover_materialization")
    expected_receipt_keys = {
        "schema_version",
        "materialization_policy",
        "witness_consistency_policy",
        "attribution_policy",
        "draft_configuration_sha256",
        "materialized_configuration_sha256",
        "inherited_paths",
        "synthesized_paths",
    }
    if (
        type(materialization) is not dict
        or set(materialization) != expected_receipt_keys
    ):
        fail("materialization receipt is absent or has an open schema")
    if (
        materialization.get("schema_version") != 1
        or materialization.get("materialization_policy")
        != "exact_named_parent_subtree_copy_v1"
        or materialization.get("witness_consistency_policy")
        != "typed_json_exact_or_one_finite_binary64_ulp_per_float_leaf_v1"
        or materialization.get("attribution_policy")
        != "exhaustive_nonoverlapping_component_plan_v1"
        or not is_sha256(materialization.get("draft_configuration_sha256"))
        or materialization.get("materialized_configuration_sha256")
        != configuration_sha256
        or evaluated.get("crossover_draft_configuration_hash")
        != materialization.get("draft_configuration_sha256")
    ):
        fail("materialization receipt identity is inconsistent")

    inherited = materialization.get("inherited_paths")
    synthesized = materialization.get("synthesized_paths")
    if type(inherited) is not list or type(synthesized) is not list:
        fail("materialization path evidence is malformed")
    inherited_keys = {
        "path",
        "source",
        "witness_value_sha256",
        "parent_value_sha256",
        "witness_exact",
        "adjusted_float_leaf_count",
        "max_float_ulp_distance",
    }
    synthesized_keys = {
        "path",
        "witness_value_sha256",
        "left_value_sha256",
        "right_value_sha256",
    }
    observed_attribution: list[tuple[str, str]] = []
    adjusted_float_leaf_count = 0
    for item in inherited:
        if (
            type(item) is not dict
            or set(item) != inherited_keys
            or type(item.get("path")) is not str
            or item.get("source") not in {"left", "right"}
            or not is_sha256(item.get("witness_value_sha256"))
            or not is_sha256(item.get("parent_value_sha256"))
            or type(item.get("witness_exact")) is not bool
            or type(item.get("adjusted_float_leaf_count")) is not int
            or item["adjusted_float_leaf_count"] < 0
            or type(item.get("max_float_ulp_distance")) is not int
            or item["max_float_ulp_distance"] not in {0, 1}
        ):
            fail("inherited-path materialization evidence is malformed")
        if item["witness_exact"] is True and (
            item["adjusted_float_leaf_count"] != 0
            or item["max_float_ulp_distance"] != 0
        ):
            fail("exact witness reports a numeric adjustment")
        if item["witness_exact"] is False and (
            item["adjusted_float_leaf_count"] <= 0
            or item["max_float_ulp_distance"] != 1
        ):
            fail("inexact witness lacks bounded one-ULP evidence")
        observed_attribution.append((item["path"], item["source"]))
        adjusted_float_leaf_count += item["adjusted_float_leaf_count"]
    for item in synthesized:
        if (
            type(item) is not dict
            or set(item) != synthesized_keys
            or type(item.get("path")) is not str
            or not is_sha256(item.get("witness_value_sha256"))
            or (
                item.get("left_value_sha256") is not None
                and not is_sha256(item.get("left_value_sha256"))
            )
            or (
                item.get("right_value_sha256") is not None
                and not is_sha256(item.get("right_value_sha256"))
            )
        ):
            fail("synthesized-path materialization evidence is malformed")
        observed_attribution.append((item["path"], "synthesized"))
    if (
        len({path for path, _ in observed_attribution}) != len(observed_attribution)
        or set(observed_attribution) != set(source_attribution)
        or len(observed_attribution) != len(source_attribution)
    ):
        fail("materialization paths do not close the candidate attribution")
    if not {"left", "right"}.issubset({source for _, source in observed_attribution}):
        fail("materialization receipt lacks both named-parent contributions")
    if evaluated.get("crossover_adjusted_float_leaf_count") != (
        adjusted_float_leaf_count
    ):
        fail("materialization adjustment count is inconsistent")

    encoded_receipt = json.dumps(
        materialization,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")
    receipt_sha256 = hashlib.sha256(
        b"agent-evolve:crossover-inheritance-materialization:v1\x00" + encoded_receipt
    ).hexdigest()
    if evaluated.get("crossover_materialization_receipt_sha256") != receipt_sha256:
        fail("materialization receipt digest does not verify")

    return {
        "schema_version": 1,
        "evidence_protocol": "engine_crossover_materialization_trace_join_v1",
        "slot_id": slot_id,
        "operator_invocation_id": operator_invocation_id,
        "llm_call_id": llm_call_id,
        "candidate_id": candidate_id,
        "proposal_sequence": proposal_sequence,
        "parent_candidate_ids": list(parent_candidate_ids),
        "invocation_prepared_trace_sequence": invocation["sequence"],
        "candidate_evaluated_trace_sequence": evaluated["sequence"],
        "configuration_sha256": configuration_sha256,
        "materialization_receipt_sha256": receipt_sha256,
        "materialization_receipt": materialization,
        "verification_facts": {
            "exact_call_operator_candidate_join": True,
            "exact_parent_identity_join": True,
            "exact_materialized_configuration_join": True,
            "receipt_digest_recomputed": True,
            "attribution_exhaustive_and_nonoverlapping": True,
            "both_named_parent_sources_present": True,
            "inherited_path_count": len(inherited),
            "synthesized_path_count": len(synthesized),
            "adjusted_float_leaf_count": adjusted_float_leaf_count,
        },
    }


async def _result_record(
    result: OptimizerResult,
    live: AirfoilV10MultiOptionLiveComposition,
    inputs: AirfoilV10MultiOptionInputs,
    provider_outcomes: tuple[dict[str, object], ...],
    engine_trace_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    validate_optimizer_result_integrity(result)
    planner = live.composition.planner
    reflection = live.composition.feedback_interceptor
    if type(planner) is not MultiOptionEvolutionPlanner:
        raise AirfoilV10MultiOptionRunnerError("live run exposed a foreign planner")
    if type(reflection) is not PostEvolutionReflectionInterceptor:
        raise AirfoilV10MultiOptionRunnerError("live run exposed a foreign reflection")
    if (
        result.stop_reason is not OptimizerStopReason.GENERATION_LIMIT_REACHED
        or result.final_state.generation != 3
        or len(result.seed_receipts) != 2
        or len(result.generation_receipts) != 3
        or len(result.feedback_receipts) != 3
        or result.final_state.logical_llm_calls
        != MULTI_OPTION_EVOLUTION_BUDGET.max_logical_llm_calls
        or len(result.final_state.candidates) != _EXPECTED_CANDIDATE_OCCURRENCES
    ):
        raise AirfoilV10MultiOptionRunnerError(
            "completed result lacks the exact G0--G3 seven-call chronology"
        )
    for receipt, expected in zip(
        result.generation_receipts,
        _EXPECTED_SLOT_IDS,
        strict=True,
    ):
        if tuple(value.slot.slot_id for value in receipt.slot_results) != expected:
            raise AirfoilV10MultiOptionRunnerError("generation slot chronology drifted")
        for value in receipt.slot_results:
            candidate = value.outcome.candidate
            if (
                value.outcome.failure_stage is not None
                or candidate is None
                or not candidate.valid
                or not candidate.operator_compliant
                or not candidate.evidence_compliant
            ):
                raise AirfoilV10MultiOptionRunnerError(
                    "completed chronology contains an unsuccessful candidate slot"
                )
    audits = planner.effective_choice_audit_receipts
    if tuple(audits) != (
        (1, MULTI_OPTION_G1_SLOT_IDS[0]),
        (1, MULTI_OPTION_G1_SLOT_IDS[1]),
        (2, MULTI_OPTION_G2_SLOT_IDS[0]),
        (2, MULTI_OPTION_G2_SLOT_IDS[1]),
    ) or any(value.effective_cardinality != 8 for value in audits.values()):
        raise AirfoilV10MultiOptionRunnerError(
            "planner lacks four genuine K=8 effective-choice receipts"
        )
    if planner.uniform_rank is None or planner.uniform_decision is None:
        raise AirfoilV10MultiOptionRunnerError("uniform comparator was not frozen")
    if reflection.reflection_authority is None or reflection.reflection_receipt is None:
        raise AirfoilV10MultiOptionRunnerError("terminal reflection lacks receipts")
    if reflection.reflection_receipt.reflection_status != "sealed_complete":
        raise AirfoilV10MultiOptionRunnerError("terminal reflection did not complete")
    if tuple(value.used_logical_llm_calls for value in result.feedback_receipts) != (
        0,
        0,
        1,
    ):
        raise AirfoilV10MultiOptionRunnerError("reflection call chronology drifted")

    generation_rows = [
        {
            "generation": receipt.generation,
            "plan_sha256": receipt.plan_hash,
            "receipt_sha256": receipt.receipt_hash,
            "pre_archive_snapshot_sha256": receipt.pre_archive_snapshot_hash,
            "post_archive_snapshot_sha256": receipt.post_archive_snapshot_hash,
            "reward_definition_sha256": receipt.reward_definition_hash,
            "reward_snapshot_sha256": receipt.reward_snapshot_hash,
            "logical_llm_calls_before": receipt.logical_llm_calls_before,
            "logical_llm_calls_after": receipt.logical_llm_calls_after,
            "unique_evaluations_before": receipt.unique_evaluations_before,
            "unique_evaluations_after": receipt.unique_evaluations_after,
            "slots": [
                _slot_record(value, engine_trace_rows) for value in receipt.slot_results
            ],
        }
        for receipt in result.generation_receipts
    ]
    by_g3 = {
        value.slot.slot_id: value
        for value in result.generation_receipts[2].slot_results
    }
    known_targets_by_sha256 = {
        typed_json_sha256(candidate.configuration): candidate.configuration
        for candidate in result.final_state.candidates
        if candidate.generation < 3
    }
    for slot_id in MULTI_OPTION_G3_CORE_SLOT_IDS[1:]:
        known_candidate = by_g3[slot_id].outcome.candidate
        if known_candidate is None:  # pragma: no cover - validated above.
            raise AirfoilV10MultiOptionRunnerError(
                "scheduled G3 core target disappeared"
            )
        known_targets_by_sha256.setdefault(
            known_candidate.occurrence.configuration_hash,
            known_candidate.configuration,
        )
    known_target_configurations = tuple(
        known_targets_by_sha256[digest] for digest in sorted(known_targets_by_sha256)
    )
    crossover_rows: list[dict[str, object]] = []
    for crossover_slot, union_slot in zip(
        MULTI_OPTION_G3_CROSSOVER_SLOT_IDS,
        MULTI_OPTION_G3_SLOT_IDS[1:3],
        strict=True,
    ):
        crossover_result = by_g3[crossover_slot]
        crossover = crossover_result.outcome.candidate
        union = by_g3[union_slot].outcome.candidate
        assert crossover is not None and union is not None
        prepared = crossover_result.outcome.prepared
        parent_candidate_ids = tuple(
            parent.candidate_id.value for parent in crossover_result.slot.plan.parents
        )
        if (
            crossover.occurrence.operator_invocation_id
            != prepared.operator_invocation_id
            or tuple(value.value for value in crossover.parent_ids)
            != parent_candidate_ids
        ):
            raise AirfoilV10MultiOptionRunnerError(
                "model crossover candidate lineage differs from its exact slot"
            )
        materialization_evidence = _verified_model_crossover_materialization_record(
            engine_trace_rows,
            slot_id=crossover_slot,
            operator_invocation_id=prepared.operator_invocation_id.value,
            llm_call_id=(None if prepared.call_id is None else prepared.call_id.value),
            candidate_id=crossover.candidate_id.value,
            proposal_sequence=prepared.proposal_sequence,
            parent_candidate_ids=parent_candidate_ids,
            parent_configuration_sha256s=tuple(
                parent.occurrence.configuration_hash
                for parent in crossover_result.slot.plan.parents
            ),
            parent_configurations=tuple(
                parent.configuration for parent in crossover_result.slot.plan.parents
            ),
            known_target_configurations=known_target_configurations,
            configuration=crossover.configuration,
            configuration_sha256=crossover.occurrence.configuration_hash,
            parent_patch_sha256s=tuple(crossover.parent_patch_hashes),
            source_attribution=tuple(
                (value.path, value.source) for value in crossover.source_attribution
            ),
        )
        if any(
            typed_json_equal(crossover.configuration, target)
            for target in known_target_configurations
        ):
            raise AirfoilV10MultiOptionRunnerError(
                "model crossover duplicated a preexisting or scheduled core target"
            )
        crossover_rows.append(
            {
                "slot_id": crossover_slot,
                "proposal_authority": crossover_result.slot.proposal_authority.value,
                "operator_kind": crossover.operator_kind.value,
                "operator_compliant": crossover.operator_compliant,
                "evidence_compliant": crossover.evidence_compliant,
                "machine_verified_two_parent_contributions": True,
                "materialization_evidence": materialization_evidence,
                "parent_patch_sha256s": list(crossover.parent_patch_hashes),
                "source_attribution": [
                    {"path": value.path, "source": value.source}
                    for value in crossover.source_attribution
                ],
                "differs_from_each_parent": all(
                    crossover.occurrence.configuration_hash
                    != parent.occurrence.configuration_hash
                    for parent in crossover_result.slot.plan.parents
                ),
                "deterministic_union_slot_id": union_slot,
                "differs_from_deterministic_union": (
                    crossover.occurrence.configuration_hash
                    != union.occurrence.configuration_hash
                ),
                "differs_from_every_preexisting_or_scheduled_core_target": True,
            }
        )
    if any(
        row["proposal_authority"] != ProposalAuthority.MODEL.value
        or row["operator_kind"] != OperatorKind.TWO_PARENT_CROSSOVER.value
        or not row["machine_verified_two_parent_contributions"]
        or not row["differs_from_deterministic_union"]
        or not row["differs_from_every_preexisting_or_scheduled_core_target"]
        for row in crossover_rows
    ):
        raise AirfoilV10MultiOptionRunnerError(
            "terminal model crossover lacks machine compliance"
        )

    cache = await live.composition.engine.evaluation_cache_snapshot()
    profile = live.provider_profile
    provider_record = _provider_config_record(profile)
    provider_summary = _provider_summary(provider_outcomes, profile)
    reflection_receipt = reflection.reflection_receipt
    reflection_authority = reflection.reflection_authority
    return {
        "schema_version": 1,
        "claim_boundary": "authentic_live_development_run_not_independent_replication",
        "chronology_validated": True,
        "optimizer_result_sha256": result.result_hash,
        "stop_reason": result.stop_reason.value,
        "inputs_sha256": inputs.inputs_sha256,
        "counts": {
            "generations": result.final_state.generation,
            "seed_candidates": len(result.seed_receipts),
            "candidate_occurrences": len(result.final_state.candidates),
            "logical_llm_calls": result.final_state.logical_llm_calls,
            "unique_evaluations": result.final_state.unique_evaluations,
            "effective_choice_model_calls": len(audits),
            "model_crossover_calls": len(MULTI_OPTION_G3_CROSSOVER_SLOT_IDS),
            "reflection_calls": 1,
        },
        "provider_configuration": provider_record,
        "provider_call_summary": provider_summary,
        "g0_seeds": [
            {
                "label": receipt.label,
                "receipt_sha256": receipt.receipt_hash,
                "candidate": _candidate_record(receipt.candidate),
            }
            for receipt in result.seed_receipts
        ],
        "generations": generation_rows,
        "effective_choice_audits": [
            {
                "generation": generation,
                "slot_id": slot_id,
                **receipt.to_record(),
                "receipt_sha256": receipt.receipt_sha256,
            }
            for (generation, slot_id), receipt in audits.items()
        ],
        "memory": _memory_record(planner),
        "uniform_comparator": {
            "rank": {
                **planner.uniform_rank.to_record(),
                "token_sha256": planner.uniform_rank.token_sha256,
            },
            "decision": {
                **planner.uniform_decision.to_record(),
                "decision_sha256": planner.uniform_decision.decision_sha256,
            },
        },
        "g3_union_materialization_receipt_sha256s": list(
            planner.g3_union_materialization_receipt_sha256s
        ),
        "crossovers": crossover_rows,
        "evaluation_cache": cache,
        "reflection": {
            "authority": {
                **reflection_authority.to_record(),
                "authority_sha256": reflection_authority.authority_sha256,
            },
            "receipt": {
                **reflection_receipt.to_record(),
                "receipt_sha256": reflection_receipt.receipt_sha256,
            },
            "call_receipt": reflection_receipt.call_receipt.to_record(),
            "revisions": [
                value.to_record()
                for value in reflection_receipt.call_receipt.publications
            ],
        },
        "feedback_receipts": [
            {
                "generation": value.generation,
                "policy_id": value.policy_id,
                "used_logical_llm_calls": value.used_logical_llm_calls,
                "logical_llm_calls_before": value.logical_llm_calls_before,
                "logical_llm_calls_after": value.logical_llm_calls_after,
                "result_metadata": [list(item) for item in value.result_metadata],
                "receipt_sha256": value.receipt_hash,
            }
            for value in result.feedback_receipts
        ],
    }


async def _await_if_needed(value: object) -> object:
    return await value if inspect.isawaitable(value) else value


def _readiness_boundary_record(*, production: bool) -> dict[str, object]:
    return {
        "schema_version": 1,
        "entrypoint": "readiness" if production else "_readiness_development",
        "dependency_mode": (
            "sealed_production_dependencies"
            if production
            else "injected_development_dependencies"
        ),
        "production_stack_authenticated": production,
        "production_dependencies_authenticated": production,
        "qualification_route_source_authenticated": production,
        "injected_dependencies_allowed": not production,
        "live_promotion_eligible": production,
        "scientific_result_eligible": False,
        "credential_read": False,
        "provider_called": False,
        "physical_evaluator_called": False,
    }


def _readiness_commitment(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        _READINESS_COMMITMENT_DOMAIN
        + json.dumps(
            dict(value),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii", errors="strict")
    ).hexdigest()


def _bind_readiness_selected_provider(
    record: dict[str, object],
    provider_record: dict[str, object],
) -> dict[str, object]:
    """Keep ``provider`` as the active-provider alias without losing preflight data."""

    if type(record) is not dict or type(provider_record) is not dict:
        raise TypeError("readiness and provider records must be exact dictionaries")
    if "input_default_provider_preflight" in record:
        raise AirfoilV10MultiOptionRunnerError(
            "readiness payload preempted provider compatibility metadata"
        )
    if "provider" not in record:
        return dict(record)
    input_default_provider = record["provider"]
    if type(input_default_provider) is not dict:
        raise AirfoilV10MultiOptionRunnerError(
            "readiness input-default provider record is malformed"
        )
    return {
        **record,
        "input_default_provider_preflight": input_default_provider,
        "provider": provider_record,
    }


def validate_airfoil_v10_readiness_record(
    value: Mapping[str, object],
    *,
    require_live_promotable: bool = True,
) -> dict[str, object]:
    """Authenticate one readiness record and enforce its promotion authority."""

    if type(require_live_promotable) is not bool:
        raise TypeError("require_live_promotable must be exact bool")
    if not isinstance(value, Mapping):
        raise TypeError("readiness record must be a mapping")
    try:
        record = json.loads(
            json.dumps(
                dict(value),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise AirfoilV10MultiOptionRunnerError(
            "readiness record must contain canonical JSON values"
        ) from exc
    if type(record) is not dict:
        raise AirfoilV10MultiOptionRunnerError(
            "readiness record must be an exact object"
        )
    supplied = record.pop("readiness_commitment_sha256", None)
    if type(supplied) is not str or supplied != _readiness_commitment(record):
        raise AirfoilV10MultiOptionRunnerError("readiness record commitment is invalid")
    boundary = record.get("readiness_boundary")
    production_boundary = _readiness_boundary_record(production=True)
    development_boundary = _readiness_boundary_record(production=False)
    if boundary not in (production_boundary, development_boundary):
        raise AirfoilV10MultiOptionRunnerError(
            "readiness authority boundary is invalid"
        )
    if any(
        type(record.get(name)) is not dict
        for name in (
            "selected_provider",
            "offline_qualification",
            "runtime_manifest",
        )
    ):
        raise AirfoilV10MultiOptionRunnerError("readiness identity is incomplete")
    if boundary == production_boundary:
        provider = record["selected_provider"]
        qualification = record["offline_qualification"]
        runtime_manifest = record["runtime_manifest"]
        assert type(provider) is dict
        assert type(qualification) is dict
        assert type(runtime_manifest) is dict
        claim_boundary = record.get("claim_boundary")
        qualification_directory = qualification.get("directory")
        if (
            record.get("ready") is not True
            or type(claim_boundary) is not dict
            or claim_boundary.get("provider_called") is not False
            or claim_boundary.get("credentials_read") is not False
            or claim_boundary.get("physical_evaluator_called") is not False
            or type(provider.get("profile_id")) is not str
            or qualification.get("provider_profile_id") != provider.get("profile_id")
            or qualification.get("provider_configuration_sha256")
            != airfoil_v10_provider_configuration_sha256(provider)
            or runtime_manifest.get("manifest_id") != AIRFOIL_V10_RUNTIME_MANIFEST_ID
            or runtime_manifest.get("manifest_version")
            != AIRFOIL_V10_RUNTIME_MANIFEST_VERSION
            or runtime_manifest.get("filename") != AIRFOIL_V10_RUNTIME_MANIFEST_FILENAME
            or qualification.get("source_sha256")
            != runtime_manifest.get("source_sha256")
            or type(qualification_directory) is not str
            or not qualification_directory
        ):
            raise AirfoilV10MultiOptionRunnerError(
                "live-promotable readiness identity is inconsistent"
            )
        try:
            verified_qualification = verify_airfoil_v10_qualification_directory(
                Path(qualification_directory),
                provider_profile=_resolve_profile(str(provider["profile_id"])),
                provider_record=provider,
                source_closure_factory=capture_airfoil_v10_runtime_source_closure,
            )
        except Exception as exc:
            raise AirfoilV10MultiOptionRunnerError(
                "live-promotable readiness qualification is no longer authentic"
            ) from exc
        if verified_qualification.to_record() != qualification:
            raise AirfoilV10MultiOptionRunnerError(
                "live-promotable readiness qualification identity changed"
            )
    if require_live_promotable and boundary != production_boundary:
        raise AirfoilV10MultiOptionRunnerError(
            "injected readiness is not eligible for live promotion"
        )
    return {**record, "readiness_commitment_sha256": supplied}


def _readiness_impl(
    run_id: str,
    *,
    _readiness_authority: object,
    qualification_dir: Path,
    provider_profile_id: str = DEFAULT_PROVIDER_PROFILE_ID,
    problem_factory: Callable[..., object] = _problem,
    inputs_loader: Callable[..., object] = load_frozen_airfoil_v10_multi_option_inputs,
    readiness_record_factory: Callable[
        ..., object
    ] = airfoil_v10_multi_option_readiness_record,
    runtime_manifest_factory: Callable[..., LiveRuntimeManifest] = (
        build_airfoil_v10_runtime_manifest
    ),
    qualification_loader: Callable[..., object] = (
        verify_airfoil_v10_qualification_directory
    ),
    run_root: Path = RUN_ROOT,
    work_root: Path = WORK_ROOT,
) -> dict[str, object]:
    """Construct a qualification-bound readiness under an explicit authority."""

    canonical = _validate_run_id(run_id)
    profile = _resolve_profile(provider_profile_id)
    if _readiness_authority is _PRODUCTION_READINESS_AUTHORITY:
        production = True
        if not (
            problem_factory is _problem
            and inputs_loader is load_frozen_airfoil_v10_multi_option_inputs
            and readiness_record_factory is airfoil_v10_multi_option_readiness_record
            and runtime_manifest_factory is build_airfoil_v10_runtime_manifest
            and qualification_loader is verify_airfoil_v10_qualification_directory
        ):
            raise AirfoilV10MultiOptionRunnerError(
                "live-promotable readiness requires canonical production dependencies"
            )
    elif _readiness_authority is _DEVELOPMENT_READINESS_AUTHORITY:
        production = False
    else:
        raise AirfoilV10MultiOptionRunnerError("readiness authority is invalid")
    for name, dependency in (
        ("problem_factory", problem_factory),
        ("inputs_loader", inputs_loader),
        ("readiness_record_factory", readiness_record_factory),
        ("runtime_manifest_factory", runtime_manifest_factory),
        ("qualification_loader", qualification_loader),
    ):
        if not callable(dependency):
            raise TypeError(f"{name} must be callable")
    run_dir = run_root / canonical
    problem_value = problem_factory(canonical, run_dir, work_root)
    problem = problem_value[0] if type(problem_value) is tuple else problem_value
    inputs = inputs_loader(problem=problem)
    record = readiness_record_factory(inputs)
    if type(record) is not dict:
        raise TypeError("readiness record factory must return an exact dictionary")
    provider_record = _provider_config_record(profile)
    record = _bind_readiness_selected_provider(record, provider_record)
    qualification = qualification_loader(
        qualification_dir,
        provider_profile=profile,
        provider_record=provider_record,
        source_closure_factory=capture_airfoil_v10_runtime_source_closure,
    )
    qualification_to_record = getattr(qualification, "to_record", None)
    if not callable(qualification_to_record):
        raise TypeError("qualification loader result must implement to_record")
    qualification_record = qualification_to_record()
    if type(qualification_record) is not dict:
        raise TypeError("qualification record must be an exact dictionary")
    manifest = runtime_manifest_factory(
        inputs=inputs,
        built_at_utc=_utc_seconds(),
        run_id=canonical,
        provider_profile=profile,
        provider_record=provider_record,
        qualification=qualification,
        run_root=run_root,
        work_root=work_root,
    )
    if type(manifest) is not LiveRuntimeManifest:
        raise TypeError("runtime manifest factory returned a foreign value")
    verify_runtime_source_closure(manifest.source_closure)
    reserved = {
        "readiness_boundary",
        "readiness_commitment_sha256",
        "selected_provider",
        "offline_qualification",
        "runtime_manifest",
    }
    if reserved & set(record):
        raise AirfoilV10MultiOptionRunnerError(
            "readiness payload attempted to replace authority metadata"
        )
    readiness_record: dict[str, object] = {
        **record,
        "readiness_boundary": _readiness_boundary_record(production=production),
        "selected_provider": provider_record,
        "offline_qualification": qualification_record,
        "runtime_manifest": runtime_manifest_identity_record(manifest),
    }
    readiness_record["readiness_commitment_sha256"] = _readiness_commitment(
        readiness_record
    )
    return validate_airfoil_v10_readiness_record(
        readiness_record,
        require_live_promotable=production,
    )


def _readiness_development(
    run_id: str,
    **dependencies: object,
) -> dict[str, object]:
    """Private injected readiness whose record is permanently non-promotable."""

    return _readiness_impl(
        run_id,
        _readiness_authority=_DEVELOPMENT_READINESS_AUTHORITY,
        **dependencies,
    )


def readiness(
    run_id: str,
    *,
    qualification_dir: Path,
    provider_profile_id: str = DEFAULT_PROVIDER_PROFILE_ID,
    run_root: Path = RUN_ROOT,
    work_root: Path = WORK_ROOT,
) -> dict[str, object]:
    """Sealed, provider-free production readiness eligible for live promotion."""

    return _readiness_impl(
        run_id,
        _readiness_authority=_PRODUCTION_READINESS_AUTHORITY,
        qualification_dir=qualification_dir,
        provider_profile_id=provider_profile_id,
        run_root=run_root,
        work_root=work_root,
    )


async def _execute_live_impl(
    run_id: str,
    *,
    _execution_authority: object,
    qualification_dir: Path,
    provider_profile_id: str = DEFAULT_PROVIDER_PROFILE_ID,
    credential_source: Callable[[], str] = _read_dotenv_api_key,
    resource_lease_factory: Callable[[str], ExclusiveResourceLease] = _lease,
    problem_factory: Callable[..., object] = _problem,
    inputs_loader: Callable[..., object] = load_frozen_airfoil_v10_multi_option_inputs,
    readiness_record_factory: Callable[
        ..., object
    ] = airfoil_v10_multi_option_readiness_record,
    runtime_manifest_factory: Callable[..., LiveRuntimeManifest] = (
        build_airfoil_v10_runtime_manifest
    ),
    runtime_manifest_gate_factory: Callable[..., object] = (
        FrozenAirfoilV10RuntimeManifestGate
    ),
    qualification_loader: Callable[..., object] = (
        verify_airfoil_v10_qualification_directory
    ),
    live_factory: Callable[..., object] = compose_airfoil_v10_multi_option_live,
    result_record_factory: Callable[..., object] = _result_record,
    generator_factory=None,
    run_root: Path = RUN_ROOT,
    work_root: Path = WORK_ROOT,
) -> dict[str, object]:
    """Run, validate, and finalize one authentic G0--G3 evolution."""

    canonical = _validate_run_id(run_id)
    profile = _resolve_profile(provider_profile_id)
    if _execution_authority is _PRODUCTION_EXECUTION_AUTHORITY:
        production_dependencies = (
            credential_source is _read_dotenv_api_key
            and resource_lease_factory is _lease
            and problem_factory is _problem
            and inputs_loader is load_frozen_airfoil_v10_multi_option_inputs
            and readiness_record_factory is airfoil_v10_multi_option_readiness_record
            and runtime_manifest_factory is build_airfoil_v10_runtime_manifest
            and runtime_manifest_gate_factory is FrozenAirfoilV10RuntimeManifestGate
            and qualification_loader is verify_airfoil_v10_qualification_directory
            and live_factory is compose_airfoil_v10_multi_option_live
            and result_record_factory is _result_record
            and generator_factory is None
        )
        if not production_dependencies:
            raise AirfoilV10MultiOptionRunnerError(
                "scientific execution requires the sealed production dependency set"
            )
        execution_boundary = {
            "schema_version": 1,
            "entrypoint": "execute_live",
            "dependency_mode": "sealed_production_dependencies",
            "injected_dependencies_allowed": False,
            "scientific_result_eligible": True,
        }
    elif _execution_authority is _DEVELOPMENT_EXECUTION_AUTHORITY:
        execution_boundary = {
            "schema_version": 1,
            "entrypoint": "_execute_live_development",
            "dependency_mode": "injected_development_dependencies",
            "injected_dependencies_allowed": True,
            "scientific_result_eligible": False,
        }
    else:
        raise AirfoilV10MultiOptionRunnerError("execution authority is invalid")
    for name, value in (
        ("credential_source", credential_source),
        ("resource_lease_factory", resource_lease_factory),
        ("problem_factory", problem_factory),
        ("inputs_loader", inputs_loader),
        ("readiness_record_factory", readiness_record_factory),
        ("runtime_manifest_factory", runtime_manifest_factory),
        ("runtime_manifest_gate_factory", runtime_manifest_gate_factory),
        ("qualification_loader", qualification_loader),
        ("live_factory", live_factory),
        ("result_record_factory", result_record_factory),
    ):
        if not callable(value):
            raise TypeError(f"{name} must be callable")
    run_dir = run_root / canonical
    work_dir = work_root / canonical
    if run_dir.exists() or work_dir.exists():
        raise AirfoilV10MultiOptionRunnerError("run output/work root already exists")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(run_dir / "execution_boundary.json", execution_boundary)
    progress = BatchedDurableJsonlJournal(
        run_dir / "provider_progress.jsonl",
        max_unfsynced_rows=32,
    )
    outcomes = DurableJsonlJournal(run_dir / "provider_outcomes.jsonl")
    requests = DurableJsonlJournal(run_dir / "provider_requests.jsonl")
    provider_attempt_requests = DurableJsonlJournal(
        run_dir / "provider_attempt_requests.jsonl"
    )
    outputs = DurableJsonlJournal(run_dir / "provider_outputs.jsonl")
    engine_traces = DurableJsonlJournal(run_dir / "engine_traces.jsonl")
    optimizer_traces = DurableJsonlJournal(run_dir / "optimizer_traces.jsonl")
    planner_traces = DurableJsonlJournal(run_dir / "planner_traces.jsonl")
    provider_rows: list[dict[str, object]] = []
    lease: ExclusiveResourceLease | None = None
    runtime_manifest: LiveRuntimeManifest | None = None
    qualification: object | None = None
    qualification_record: dict[str, object] | None = None
    runtime_manifest_gate: object | None = None
    runtime_manifest_precomposition: object | None = None
    runtime_manifest_preexecution: object | None = None
    runtime_manifest_precredential: object | None = None
    runtime_manifest_postoptimizer: object | None = None
    runtime_manifest_terminal: object | None = None
    provider_attempt_join_receipt: dict[str, object] | None = None
    live: object | None = None
    raw: object | None = None
    pending: BaseException | None = None
    result_record: dict[str, object] | None = None
    status = "failed"
    stage = "run_directory_created"
    credential_reads = 0

    def trace_sink(journal: DurableJsonlJournal, source: str):
        return lambda row: journal.append(
            {"schema_version": 1, "source": source, **dict(row)}
        )

    def progress_sink(value: StructuredStreamProgress) -> None:
        progress.append(_progress_record(value))

    def outcome_sink(value: Any) -> None:
        progress.flush()
        row = structured_generation_outcome_record(value)
        provider_rows.append(row)
        outcomes.append(row)

    def outbound_request_manifest_sink(row: Mapping[str, object]) -> None:
        canonical_manifest = validate_openrouter_outbound_request_manifest_record(row)
        provider_attempt_requests.append(canonical_manifest)

    def persist_provider_attempt_join() -> dict[str, object]:
        nonlocal provider_attempt_join_receipt
        progress.flush()
        receipt = build_provider_attempt_terminal_join_receipt(
            logical_requests=read_jsonl(requests.path),
            outbound_manifests=read_jsonl(provider_attempt_requests.path),
            terminal_outcomes=read_jsonl(outcomes.path),
            progress_rows=read_jsonl(progress.path),
            explicit_pre_transport_failures=(),
            expected_framework_versions=_qualification_framework_versions(
                qualification_record
            ),
            expected_transport_settings=(
                airfoil_v10_expected_outbound_transport_settings(profile)
            ),
        )
        write_json_atomic(run_dir / "provider_attempt_join.json", receipt)
        provider_attempt_join_receipt = receipt
        return receipt

    def register_terminal_failure(
        exc: BaseException,
        *,
        failure_stage: str,
        secondary_note: str,
    ) -> None:
        nonlocal pending, stage
        if pending is None:
            stage = failure_stage
            pending = exc
        else:
            pending.add_note(f"{secondary_note}: {type(exc).__name__}")

    def persist_failure_if_absent() -> None:
        if pending is None:
            return
        failure_path = run_dir / "failure.json"
        if failure_path.exists():
            return
        write_json_atomic(
            failure_path,
            {
                "schema_version": 1,
                "stage": stage,
                "failure_type": type(pending).__name__,
                "credential_read_count": credential_reads,
                "credential_value_persisted": False,
                "execution_boundary": execution_boundary,
                "safe_message": "inspect finalized v10 journals",
            },
        )

    def verify_runtime_manifest(stage_name: str) -> object:
        if runtime_manifest_gate is None:
            raise AirfoilV10MultiOptionRunnerError(
                "runtime manifest gate is unavailable"
            )
        verify = getattr(runtime_manifest_gate, "verify", None)
        if not callable(verify):
            raise TypeError("runtime manifest gate must implement verify")
        receipt = verify()
        to_record = getattr(receipt, "to_record", None)
        if not callable(to_record):
            raise TypeError("runtime manifest verification lacks to_record")
        record = to_record()
        if type(record) is not dict:
            raise TypeError("runtime manifest verification record must be exact")
        write_json_atomic(
            run_dir / f"runtime_manifest_{stage_name}_verification.json",
            {
                "schema_version": 1,
                "stage": stage_name,
                "verification": record,
            },
        )
        return receipt

    def credential_loader() -> str:
        nonlocal credential_reads, runtime_manifest_precredential
        credential_reads += 1
        if credential_reads != 1:
            raise AirfoilV10MultiOptionRunnerError(
                "credential loader invoked more than once"
            )
        runtime_manifest_precredential = verify_runtime_manifest("precredential")
        write_json_atomic(
            run_dir / "credential_access.json",
            {
                "schema_version": 1,
                "credential_name": "OPENROUTER_API_KEY",
                "read_count": 1,
                "value_persisted": False,
                "stage": "first_g1_model_call_after_two_g0_seed_evaluations",
            },
        )
        return credential_source()

    try:
        stage = "frozen_inputs_and_readiness"
        problem_value = problem_factory(canonical, run_dir, work_root)
        if type(problem_value) is tuple:
            problem, raw = problem_value
        else:
            problem = problem_value
            raw = None
        inputs = inputs_loader(problem=problem)
        readiness_record = readiness_record_factory(inputs)
        if type(readiness_record) is not dict:
            raise TypeError("readiness record factory must return an exact dictionary")
        provider_record = _provider_config_record(profile)
        readiness_record = _bind_readiness_selected_provider(
            readiness_record,
            provider_record,
        )
        stage = "offline_qualification"
        qualification = qualification_loader(
            qualification_dir,
            provider_profile=profile,
            provider_record=provider_record,
            source_closure_factory=capture_airfoil_v10_runtime_source_closure,
        )
        qualification_to_record = getattr(qualification, "to_record", None)
        if not callable(qualification_to_record):
            raise TypeError("qualification loader result must implement to_record")
        qualification_value = qualification_to_record()
        if type(qualification_value) is not dict:
            raise TypeError("qualification record must be an exact dictionary")
        qualification_record = qualification_value
        stage = "runtime_manifest"
        runtime_manifest = runtime_manifest_factory(
            inputs=inputs,
            built_at_utc=_utc_seconds(),
            run_id=canonical,
            provider_profile=profile,
            provider_record=provider_record,
            qualification=qualification,
            run_root=run_root,
            work_root=work_root,
        )
        if type(runtime_manifest) is not LiveRuntimeManifest:
            raise TypeError("runtime manifest factory returned a foreign value")
        manifest_path = run_dir / AIRFOIL_V10_RUNTIME_MANIFEST_FILENAME
        write_json_atomic(manifest_path, runtime_manifest.to_record())
        runtime_manifest_gate = runtime_manifest_gate_factory(
            manifest_path=manifest_path,
            inputs=inputs,
            run_id=canonical,
            provider_profile=profile,
            provider_record=provider_record,
            qualification=qualification,
            run_root=run_root,
            work_root=work_root,
        )
        runtime_manifest_precomposition = verify_runtime_manifest("precomposition")
        write_json_atomic(
            run_dir / "readiness.json",
            {
                **readiness_record,
                "execution_boundary": execution_boundary,
                "selected_provider": provider_record,
                "offline_qualification": qualification_record,
                "runtime_manifest": runtime_manifest_identity_record(runtime_manifest),
                "runtime_manifest_precomposition_verification": (
                    runtime_manifest_precomposition.to_record()
                ),
            },
        )

        stage = "resource_lease"
        lease = resource_lease_factory(canonical)
        if not isinstance(lease, ExclusiveResourceLease):
            raise TypeError("resource lease factory returned a foreign object")
        acquired = lease.acquire()
        write_json_atomic(
            run_dir / "resource_lease_acquired.json",
            {"schema_version": 1, "receipt": acquired.to_record()},
        )

        stage = "live_composition"
        kwargs: dict[str, object] = {}
        if generator_factory is not None:
            kwargs["generator_factory"] = generator_factory
        live = live_factory(
            inputs,
            credential_loader=credential_loader,
            progress_sink=progress_sink,
            outcome_sink=outcome_sink,
            request_evidence_sink=lambda row: requests.append(dict(row)),
            output_evidence_sink=lambda row: outputs.append(dict(row)),
            outbound_request_manifest_sink=outbound_request_manifest_sink,
            provider_profile=profile,
            engine_trace_sink=trace_sink(engine_traces, "engine"),
            optimizer_trace_sink=trace_sink(optimizer_traces, "optimizer"),
            planner_trace_sink=trace_sink(planner_traces, "planner"),
            **kwargs,
        )
        if bool(getattr(live, "initialized_provider", True)):
            raise AirfoilV10MultiOptionRunnerError(
                "provider initialized before G0 seed evaluation"
            )

        stage = "pre_g0_runtime_verification"
        runtime_manifest_preexecution = verify_runtime_manifest("pre_g0")
        stage = "g0_g3_multi_option_execution"
        started_at = _utc()
        wall_start = time.perf_counter()
        result = await live.run()
        wall_seconds = time.perf_counter() - wall_start
        finished_at = _utc()
        stage = "postoptimizer_runtime_verification"
        runtime_manifest_postoptimizer = verify_runtime_manifest("postoptimizer")
        if credential_reads != 1:
            raise AirfoilV10MultiOptionRunnerError(
                "completed live run did not read its credential exactly once"
            )
        stage = "result_projection"
        projected = await _await_if_needed(
            result_record_factory(
                result,
                live,
                inputs,
                tuple(provider_rows),
                read_jsonl(engine_traces.path),
            )
        )
        if type(projected) is not dict:
            raise TypeError("result record factory must return an exact dictionary")
        result_record = projected
        result_record["execution_boundary"] = execution_boundary
        result_record["timing"] = {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "end_to_end_wall_seconds": wall_seconds,
        }
        if runtime_manifest is None:
            raise RuntimeError("completed run lost its runtime manifest")
        result_record["runtime_manifest"] = {
            **runtime_manifest_identity_record(runtime_manifest),
            "precomposition_verification": (
                runtime_manifest_precomposition.to_record()
            ),
            "precredential_verification": (runtime_manifest_precredential.to_record()),
            "pre_g0_verification": runtime_manifest_preexecution.to_record(),
            "postoptimizer_verification": (runtime_manifest_postoptimizer.to_record()),
        }
        relative_paths: list[str] = []
        if raw is not None:
            evaluator = getattr(raw, "evaluator", None)
            inventory = getattr(evaluator, "durable_receipt_paths", None)
            if callable(inventory):
                relative_paths = [
                    path.relative_to(run_dir).as_posix() for path in inventory()
                ]
        write_json_atomic(
            run_dir / "raw_receipt_inventory.json",
            {
                "schema_version": 1,
                "receipt_count": len(relative_paths),
                "relative_paths": relative_paths,
            },
        )
        stage = "transport_close_and_receipts"
    except BaseException as exc:
        pending = exc
        try:
            persist_failure_if_absent()
        except BaseException as artifact_exc:
            exc.add_note(f"failure journal also failed: {type(artifact_exc).__name__}")
    finally:
        if live is not None:
            if pending is None:
                stage = "transport_close_and_receipts"
            try:
                await live.aclose()
            except BaseException as exc:
                register_terminal_failure(
                    exc,
                    failure_stage="transport_close_and_receipts",
                    secondary_note="transport close also failed",
                )
        if runtime_manifest_gate is not None:
            if pending is None:
                stage = "terminal_runtime_verification"
            try:
                runtime_manifest_terminal = verify_runtime_manifest("terminal")
            except BaseException as verification_exc:
                try:
                    write_json_atomic(
                        run_dir / "runtime_manifest_terminal_verification_failure.json",
                        {
                            "schema_version": 1,
                            "stage": "terminal",
                            "failure_type": type(verification_exc).__name__,
                            "safe_message": (
                                "terminal runtime reconstruction failed closed"
                            ),
                            "primary_failure_preserved": pending is not None,
                        },
                    )
                except BaseException as artifact_exc:
                    verification_exc.add_note(
                        "terminal verification failure journal also failed: "
                        f"{type(artifact_exc).__name__}"
                    )
                register_terminal_failure(
                    verification_exc,
                    failure_stage="terminal_runtime_verification",
                    secondary_note="terminal runtime verification also failed",
                )
        if pending is None:
            stage = "postclose_provider_attempt_terminal_join"
        try:
            terminal_join = persist_provider_attempt_join()
            if terminal_join.get("join_valid") is not True:
                join_exc = AirfoilV10MultiOptionRunnerError(
                    "provider attempt terminal join failed closed"
                )
                if pending is None:
                    register_terminal_failure(
                        join_exc,
                        failure_stage="postclose_provider_attempt_terminal_join",
                        secondary_note="provider attempt terminal join also failed",
                    )
                else:
                    pending.add_note(
                        "provider attempt terminal join also failed closed"
                    )
        except BaseException as exc:
            register_terminal_failure(
                exc,
                failure_stage="postclose_provider_attempt_terminal_join",
                secondary_note=(
                    "provider attempt join receipt publication also failed"
                ),
            )
        for journal in (
            progress,
            outcomes,
            requests,
            provider_attempt_requests,
            outputs,
            engine_traces,
            optimizer_traces,
            planner_traces,
        ):
            if pending is None:
                stage = "journal_close"
            try:
                journal.close()
            except BaseException as exc:
                register_terminal_failure(
                    exc,
                    failure_stage="journal_close",
                    secondary_note="journal close also failed",
                )
        if lease is not None and lease.active:
            if pending is None:
                stage = "resource_lease_release"
            try:
                released = lease.release(
                    outcome="completed" if pending is None else "failed",
                    failure_type=None if pending is None else type(pending).__name__,
                )
                write_json_atomic(
                    run_dir / "resource_lease_released.json",
                    {"schema_version": 1, "release": released},
                )
            except BaseException as exc:
                register_terminal_failure(
                    exc,
                    failure_stage="resource_lease_release",
                    secondary_note="resource lease release also failed",
                )
        if pending is None:
            stage = "result_publication"
            try:
                if result_record is None:
                    raise AirfoilV10MultiOptionRunnerError(
                        "completed v10 execution produced no result record"
                    )
                if provider_attempt_join_receipt is None:
                    raise AirfoilV10MultiOptionRunnerError(
                        "completed v10 execution lost its provider attempt join"
                    )
                if runtime_manifest_terminal is None:
                    raise AirfoilV10MultiOptionRunnerError(
                        "completed v10 execution lost terminal source verification"
                    )
                result_record["provider_attempt_join"] = provider_attempt_join_receipt
                result_record["runtime_manifest"]["terminal_verification"] = (
                    runtime_manifest_terminal.to_record()
                )
                write_json_atomic(run_dir / "result.json", result_record)
                status = "completed"
            except BaseException as exc:
                register_terminal_failure(
                    exc,
                    failure_stage="result_publication",
                    secondary_note="result publication also failed",
                )
        if pending is not None:
            status = "failed"
            try:
                (run_dir / "result.json").unlink(missing_ok=True)
            except BaseException as exc:
                pending.add_note(
                    f"failed result cleanup also failed: {type(exc).__name__}"
                )
            try:
                persist_failure_if_absent()
            except BaseException as exc:
                pending.add_note(
                    f"failure journal publication also failed: {type(exc).__name__}"
                )
        try:
            finalize_run_directory(run_dir, status=status)
        except BaseException as exc:
            register_terminal_failure(
                exc,
                failure_stage="run_finalization",
                secondary_note="run finalization also failed",
            )
            status = "failed"
            try:
                (run_dir / "result.json").unlink(missing_ok=True)
                (run_dir / "finalized.json").unlink(missing_ok=True)
                persist_failure_if_absent()
                finalize_run_directory(run_dir, status="failed")
            except BaseException as recovery_exc:
                pending.add_note(
                    "failed finalization recovery also failed: "
                    f"{type(recovery_exc).__name__}"
                )

    if pending is not None:
        raise AirfoilV10MultiOptionRunnerError(
            f"v10 run failed at {stage}; inspect {run_dir}"
        ) from None
    if result_record is None:
        raise AirfoilV10MultiOptionRunnerError("v10 run produced no result record")
    return {"run_dir": str(run_dir), "result": result_record}


async def _execute_live_development(
    run_id: str,
    **dependencies: object,
) -> dict[str, object]:
    """Injected provider-free harness whose artifacts are never scientific."""

    return await _execute_live_impl(
        run_id,
        _execution_authority=_DEVELOPMENT_EXECUTION_AUTHORITY,
        **dependencies,
    )


async def execute_live(
    run_id: str,
    *,
    qualification_dir: Path,
    provider_profile_id: str = DEFAULT_PROVIDER_PROFILE_ID,
    run_root: Path = RUN_ROOT,
    work_root: Path = WORK_ROOT,
) -> dict[str, object]:
    """Sealed scientific entrypoint with no injectable behavioral dependency."""

    return await _execute_live_impl(
        run_id,
        _execution_authority=_PRODUCTION_EXECUTION_AUTHORITY,
        qualification_dir=qualification_dir,
        provider_profile_id=provider_profile_id,
        run_root=run_root,
        work_root=work_root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    choices = tuple(
        profile.profile_id for profile in AIRFOIL_V10_ALLOWED_PROVIDER_PROFILES
    )
    qualify = sub.add_parser("qualify")
    qualify.add_argument("--output-dir", type=Path, required=True)
    qualify.add_argument(
        "--provider-profile",
        choices=choices,
        default=DEFAULT_PROVIDER_PROFILE_ID,
    )
    ready = sub.add_parser("readiness")
    ready.add_argument("--run-id", required=True)
    ready.add_argument("--qualification-dir", type=Path, required=True)
    ready.add_argument(
        "--provider-profile",
        choices=choices,
        default=DEFAULT_PROVIDER_PROFILE_ID,
    )
    run = sub.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--qualification-dir", type=Path, required=True)
    run.add_argument(
        "--provider-profile",
        choices=choices,
        default=DEFAULT_PROVIDER_PROFILE_ID,
    )
    run.add_argument("--authorize-live", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "qualify":
        profile = _resolve_profile(args.provider_profile)
        qualification = record_airfoil_v10_qualification(
            args.output_dir,
            provider_profile=profile,
            provider_record=_provider_config_record(profile),
            source_closure_factory=capture_airfoil_v10_runtime_source_closure,
        )
        print(json.dumps(qualification.to_record(), sort_keys=True))
        return 0
    if args.command == "readiness":
        print(
            json.dumps(
                readiness(
                    args.run_id,
                    qualification_dir=args.qualification_dir,
                    provider_profile_id=args.provider_profile,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.authorize_live != LIVE_AUTHORIZATION:
        raise AirfoilV10MultiOptionRunnerError(
            "explicit live authorization token required"
        )
    outcome = asyncio.run(
        execute_live(
            args.run_id,
            qualification_dir=args.qualification_dir,
            provider_profile_id=args.provider_profile,
        )
    )
    print(outcome["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PROVIDER_PROFILE_ID",
    "GPT_XHIGH_PROVIDER_PROFILE_ID",
    "LIVE_AUTHORIZATION",
    "RUN_ROOT",
    "WORK_ROOT",
    "AirfoilV10MultiOptionRunnerError",
    "airfoil_v10_expected_outbound_transport_settings",
    "airfoil_v10_provider_config_record",
    "execute_live",
    "main",
    "readiness",
    "validate_airfoil_v10_readiness_record",
]
