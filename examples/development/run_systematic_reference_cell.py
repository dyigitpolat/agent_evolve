#!/usr/bin/env python3
"""Execute one validated cell from the systematic reference grid.

This orchestration layer owns no optimizer or workload behavior.  It binds a
registry cell to the existing prepare/preregister/live composition roots,
injects the workload-neutral model profile and replicate seed, and writes one
durable cross-workload receipt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from examples.development.durable_run_artifacts import (  # noqa: E402
    finalize_run_directory,
    source_identity,
    write_json_atomic,
)
from examples.development.systematic_experiment_study import (  # noqa: E402
    StudyCell,
    SystematicExperimentStudy,
)
from examples.development.systematic_workload_contract import (  # noqa: E402
    WorkloadExecutionContract,
)
from agent_evolve.infrastructure.resource_lease import (  # noqa: E402
    FileExclusiveResourceLease,
    ResourceConflictDetected,
    ResourceLeaseUnavailable,
)


AUTHORIZATION = "RUN_SYSTEMATIC_REFERENCE_CELL"
PREPARE_AUTHORIZATION = "PREPARE_SYSTEMATIC_REFERENCE_CELL"
PREPARED_AUTHORIZATION = "RUN_SYSTEMATIC_PREPARED_CELL"
RECEIPT_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs/"
    "systematic_reference_cells"
)
_TOKEN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,47}$")
_V1_REFERENCE_CONFIGURATION = {
    "acquisition_policy": "full_support_model_order",
    "candidate_occurrences": "62",
    "exploration_pressure": "current_calibrated",
    "memory_mode": "randomized_diagnostic",
    "operator_mix": "alternating_disjoint_recombination",
    "prompt_arm": "provenance_semantic",
    "proposal_oversubscription": "one_x",
    "reflection_schedule": "delayed_once",
    "scale_shape": "g6_k8_r2",
}
_V2_REFERENCE_CONFIGURATION = {
    **_V1_REFERENCE_CONFIGURATION,
    "acquisition_policy": "request_keyed_full_support_model_order",
    "feasibility_witness_mode": "request_keyed",
}
_V3_REFERENCE_CONFIGURATION = {
    "acquisition_policy": "task_keyed_common_candidate_pool_model_select",
    "archive_context_mode": "authenticated_affine_v1",
    "candidate_pool_size": "24",
    "evaluation_size": "4",
    "feasibility_witness_mode": "task_keyed_common_pool",
    "generation_count": "6",
    "memory_mode": "advisory_traceable",
    "model_selection_size": "8",
    "parent_lanes": "2",
    "planned_candidate_occurrences": "38",
    "prompt_arm": "provenance_semantic",
    "recombinations_per_parent": "2",
    "reflection_schedule": "delayed_once",
}
_V4_REFERENCE_CONFIGURATION = {
    **_V3_REFERENCE_CONFIGURATION,
    "memory_mode": "compatibility_audited_randomized_diagnostic",
}
_V5_QUALITY_REFERENCE_CONFIGURATION = {
    "acquisition_policy": "full_support_model_order",
    "archive_context_mode": "authenticated_affine_v1",
    "candidate_pool_size": "8",
    "evaluation_size": "8",
    "feasibility_witness_mode": "canonical",
    "generation_count": "6",
    "memory_mode": "randomized_active_neutral_assay",
    "model_selection_size": "8",
    "parent_lanes": "2",
    "planned_candidate_occurrences": "62",
    "prompt_arm": "provenance_semantic",
    "recombinations_per_parent": "2",
    "reflection_schedule": "delayed_once",
}
_V6_HIDDEN_CERTIFICATE_QUALITY_REFERENCE_CONFIGURATION = {
    **_V5_QUALITY_REFERENCE_CONFIGURATION,
    "acquisition_policy": "hidden_certificate_full_support_model_order",
    "feasibility_witness_mode": "hidden_certificate",
}
_V7_M24_STRUCTURAL_REFERENCE_CONFIGURATION = {
    "acquisition_mode": "calibrated_frontier",
    "acquisition_policy": "task_keyed_common_pool_model_k8_structural_k4",
    "archive_context_mode": "authenticated_affine_v1",
    "candidate_pool_size": "24",
    "evaluation_size": "4",
    "feasibility_witness_mode": "task_keyed_common_pool",
    "generation_count": "6",
    "memory_mode": "randomized_active_neutral_assay",
    "model_selection_size": "8",
    "parent_lanes": "2",
    "planned_candidate_occurrences": "38",
    "prompt_arm": "provenance_semantic",
    "recombinations_per_parent": "2",
    "reflection_schedule": "delayed_once",
    "scale_shape": "g6_p2_k4_r2",
}
_V8_FRONTIER_HIERARCHICAL_REFERENCE_CONFIGURATION = {
    "acquisition_mode": "hierarchical_support",
    "archive_context_mode": "authenticated_affine_v1",
    "candidate_pool_mode": "complete_finite_contract",
    "candidate_pool_size": "all",
    "evaluation_size": "4",
    "feasibility_witness_mode": "task_keyed_common_pool",
    "generation_count": "6",
    "memory_mode": "randomized_active_neutral_assay_no_online_credit",
    "method_definition_sha256": (
        "6295707fd949d4a2db292fa6d5da13d11b3b1a44aa7f14f0ee07c5550faaa729"
    ),
    "method_id": "agent_evolve_frontier_hierarchical_successor",
    "model_selection_size": "8",
    "parent_lanes": "2",
    "planned_candidate_occurrences": "38",
    "prompt_arm": "semantic",
    "proposal_support_reservations": "2",
    "recombinations_per_parent": "2",
    "reflection_schedule": "delayed_once",
    "scale_shape": "g6_p2_modelk8_evalk4_r2",
}
_V9_EXACT_BOUNDED_FINITE_WIRE_REFERENCE_CONFIGURATION = {
    **_V8_FRONTIER_HIERARCHICAL_REFERENCE_CONFIGURATION,
    "finite_option_inline_enum_limit_utf8_bytes": "8192",
    "finite_option_wire_mode": "complete_enum_when_bounded_local_exact_fallback",
}
_V10_OPERATOR_STRATIFIED_REFERENCE_CONFIGURATION = {
    "acquisition_mode": "operator_stratified",
    "archive_context_mode": "authenticated_affine_v1",
    "candidate_pool_mode": "complete_finite_contract",
    "candidate_pool_size": "all",
    "composite_option_count": "16",
    "evaluation_size": "4",
    "feasibility_witness_mode": "task_keyed_common_pool",
    "finite_option_inline_enum_limit_utf8_bytes": "8192",
    "finite_option_wire_mode": "complete_enum_when_bounded_local_exact_fallback",
    "generation_count": "6",
    "hierarchical_composite_proposal_minimum": "2",
    "memory_mode": (
        "exact_context_advisory_transfer_randomized_assay_no_online_credit"
    ),
    "method_definition_sha256": (
        "2aea2dcca7e7b165df27160a92a44b9fe246f44122e5e1a28b9504a49c38f5dc"
    ),
    "method_id": "agent_evolve_operator_stratified_successor",
    "model_selection_size": "8",
    "operator_assay_evaluation_minimum": "1",
    "parent_lanes": "2",
    "planned_candidate_occurrences": "38",
    "prompt_arm": "semantic",
    "recombinations_per_parent": "2",
    "reflection_schedule": "delayed_once",
    "scale_shape": "g6_p2_modelk8_evalk4_r2",
    "variation_topology": "hierarchical_r2",
}
_V11_HORIZON_BOUNDED_REFERENCE_CONFIGURATION = {
    "acquisition_mode": "horizon_bounded",
    "archive_context_mode": "authenticated_affine_v1",
    "candidate_pool_mode": "complete_finite_contract",
    "candidate_pool_size": "all",
    "composite_option_count": "16",
    "evaluation_size": "4",
    "family_exposure_schedule": "composite_r2_exact2_discovery_exact0_terminal",
    "feasibility_witness_mode": "task_keyed_common_pool",
    "finite_option_inline_enum_limit_utf8_bytes": "8192",
    "finite_option_wire_mode": "complete_enum_when_bounded_local_exact_fallback",
    "generation_count": "6",
    "hierarchical_composite_proposal_minimum": "2",
    "memory_mode": (
        "exact_context_advisory_transfer_randomized_assay_no_online_credit"
    ),
    "method_definition_sha256": (
        "c62fccb80610222cfdef8bffe265cd70f103123b253fd1a8c6ad975a136a3fc4"
    ),
    "method_id": "agent_evolve_horizon_bounded_successor",
    "model_selection_size": "8",
    "parent_lanes": "2",
    "planned_candidate_occurrences": "38",
    "prompt_arm": "semantic",
    "recombinations_per_parent": "2",
    "reflection_schedule": "delayed_once",
    "scale_shape": "g6_p2_modelk8_evalk4_r2",
    "variation_topology": "hierarchical_r2",
}
_IMPLEMENTED_REFERENCE_CONFIGURATIONS = (
    _V1_REFERENCE_CONFIGURATION,
    _V2_REFERENCE_CONFIGURATION,
    _V3_REFERENCE_CONFIGURATION,
    _V4_REFERENCE_CONFIGURATION,
    _V5_QUALITY_REFERENCE_CONFIGURATION,
    _V6_HIDDEN_CERTIFICATE_QUALITY_REFERENCE_CONFIGURATION,
    _V7_M24_STRUCTURAL_REFERENCE_CONFIGURATION,
    _V8_FRONTIER_HIERARCHICAL_REFERENCE_CONFIGURATION,
    _V9_EXACT_BOUNDED_FINITE_WIRE_REFERENCE_CONFIGURATION,
    _V10_OPERATOR_STRATIFIED_REFERENCE_CONFIGURATION,
    _V11_HORIZON_BOUNDED_REFERENCE_CONFIGURATION,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"{path.name} must contain one JSON object")
    return value


def _cell(study: SystematicExperimentStudy, cell_id: str) -> StudyCell:
    matches = tuple(value for value in study.reference_cells() if value.cell_id == cell_id)
    if len(matches) != 1:
        raise ValueError("cell ID does not resolve to exactly one reference cell")
    cell = matches[0]
    if dict(cell.configuration) not in _IMPLEMENTED_REFERENCE_CONFIGURATIONS:
        raise ValueError("reference cell configuration is not implemented exactly")
    return cell


def _workload_record(
    study: SystematicExperimentStudy,
    workload_id: str,
) -> dict[str, object]:
    matches = tuple(
        value
        for value in study.record["workloads"]
        if value.get("workload_id") == workload_id
    )
    if len(matches) != 1 or type(matches[0]) is not dict:
        raise ValueError("workload ID does not resolve uniquely")
    return dict(matches[0])


def _artifact_root(workload_id: str, arm: str) -> Path:
    relative = {
        ("boils_abc", "treatment"): "boils_abc/generic_campaign",
        ("boils_abc", "control"): "boils_abc/generic_campaign",
        ("heat2d", "treatment"): (
            "benchmark_q1/engibench_heat2d/generic_campaign"
        ),
        ("heat2d", "control"): (
            "benchmark_q1/engibench_heat2d/generic_campaign_control"
        ),
        ("timeloop_v2", "treatment"): (
            "benchmark_q1/timeloop_codesign/full_support_g6"
        ),
        ("timeloop_v2", "control"): (
            "benchmark_q1/timeloop_codesign/full_support_g6/"
            "matched_uniform_control"
        ),
    }[(workload_id, arm)]
    return (
        WORKSPACE_ROOT
        / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
        / relative
    )


def _commands(
    *,
    cell: StudyCell,
    runner: Path,
    attempt_id: str,
) -> tuple[list[str], list[str], Path, Path]:
    short_model = cell.model_profile or "shared"
    stem = (
        f"grid_{cell.workload_id}_{short_model}_s{cell.replicate_seed}_"
        f"{attempt_id}"
    )
    prepare_id = f"{stem}_prepare"
    live_id = f"{stem}_live"
    root = _artifact_root(cell.workload_id, cell.arm)
    prepare_dir = root / prepare_id
    live_dir = root / live_id
    prepare = [sys.executable, str(runner), "prepare", "--run-id", prepare_id]
    if cell.arm == "treatment":
        live = [sys.executable, str(runner), "live", "--run-id", live_id]
        if cell.workload_id == "boils_abc":
            prereg = prepare_dir / "preregistration_template.json"
        elif cell.workload_id == "heat2d":
            prereg = prepare_dir / "manifest.json"
        elif cell.workload_id == "timeloop_v2":
            seed = str(cell.replicate_seed)
            prepare.extend(("--replicate-seed", seed))
            live.extend(("--replicate-seed", seed))
            prereg = prepare_dir / "preregistration_template.json"
        else:  # pragma: no cover - registry validation closes this set.
            raise AssertionError("unsupported reference workload")
        live.extend(("--prereg", str(prereg)))
        return prepare, live, prepare_dir, live_dir

    if cell.workload_id == "boils_abc":
        live = [sys.executable, str(runner), "control", "--run-id", live_id]
    elif cell.workload_id == "heat2d":
        live = [sys.executable, str(runner), "live", "--run-id", live_id]
        live.extend(("--prereg", str(prepare_dir / "manifest.json")))
    elif cell.workload_id == "timeloop_v2":
        seed = str(cell.replicate_seed)
        prepare.extend(("--replicate-seed", seed))
        live = [
            sys.executable,
            str(runner),
            "run",
            "--run-id",
            live_id,
            "--replicate-seed",
            seed,
            "--prereg",
            str(prepare_dir / "preregistration_template.json"),
        ]
    else:  # pragma: no cover - registry validation closes this set.
        raise AssertionError("unsupported reference workload")
    return prepare, live, prepare_dir, live_dir


def _plan(
    *,
    study: SystematicExperimentStudy,
    cell: StudyCell,
    attempt_id: str,
    prepared_attempt_id: str | None = None,
) -> dict[str, object]:
    workload = _workload_record(study, cell.workload_id)
    execution_contract: WorkloadExecutionContract | None = None
    if workload.get("execution_contract") is not None:
        execution_contract = WorkloadExecutionContract.load(
            WORKSPACE_ROOT / str(workload["execution_contract"]),
            workspace_root=WORKSPACE_ROOT,
            expected_workload_id=cell.workload_id,
            expected_sha256=str(workload["execution_contract_sha256"]),
        )
        short_model = cell.model_profile or "shared"
        live_stem = (
            f"grid_{cell.workload_id}_{short_model}_s{cell.replicate_seed}_"
            f"{attempt_id}"
        )
        prepare_stem = (
            live_stem
            if prepared_attempt_id is None
            else (
                f"grid_{cell.workload_id}_{short_model}_s{cell.replicate_seed}_"
                f"{prepared_attempt_id}"
            )
        )
        prepare, live, prepare_dir, live_dir, _ = execution_contract.commands(
            arm_name=cell.arm,
            prepare_run_id=f"{prepare_stem}_prepare",
            live_run_id=f"{live_stem}_live",
            replicate_seed=cell.replicate_seed,
        )
        runner = execution_contract.arm(cell.arm).runner
    else:
        if prepared_attempt_id is not None:
            raise ValueError("prepared execution requires a workload-owned contract")
        runner_key = (
            "treatment_runner" if cell.arm == "treatment" else "control_runner"
        )
        runner = (WORKSPACE_ROOT / str(workload[runner_key])).resolve()
        if AGENT_EVOLVE_ROOT not in runner.parents or not runner.is_file():
            raise ValueError("treatment runner is outside AgentEvolve or missing")
        prepare, live, prepare_dir, live_dir = _commands(
            cell=cell,
            runner=runner,
            attempt_id=attempt_id,
        )
    configuration = dict(cell.configuration)
    environment_configuration = {
        **(
            {
                "AGENT_EVOLVE_FEASIBILITY_WITNESS_MODE": configuration[
                    "feasibility_witness_mode"
                ]
            }
            if "feasibility_witness_mode" in configuration
            else {}
        ),
        **(
            {
                "AGENT_EVOLVE_COMMON_CANDIDATE_POOL_SIZE": configuration[
                    "candidate_pool_size"
                ]
            }
            if "candidate_pool_size" in configuration
            else {}
        ),
        **(
            {
                "AGENT_EVOLVE_ARCHIVE_CONTEXT_MODE": configuration[
                    "archive_context_mode"
                ]
            }
            if "archive_context_mode" in configuration
            else {}
        ),
        **(
            {
                "AGENT_EVOLVE_ACQUISITION_MODE": configuration[
                    "acquisition_mode"
                ]
            }
            if "acquisition_mode" in configuration
            else {}
        ),
        **(
            {
                "AGENT_EVOLVE_VARIATION_TOPOLOGY": configuration[
                    "variation_topology"
                ]
            }
            if "variation_topology" in configuration
            else {}
        ),
        **(
            {
                "AGENT_EVOLVE_COMPOSITE_OPTION_COUNT": configuration[
                    "composite_option_count"
                ]
            }
            if "composite_option_count" in configuration
            else {}
        ),
        **(
            {
                "AGENT_EVOLVE_REQUIRED_COMPOSITE_PROPOSALS": configuration[
                    "hierarchical_composite_proposal_minimum"
                ]
            }
            if "hierarchical_composite_proposal_minimum" in configuration
            else {}
        ),
        **(
            {
                "AGENT_EVOLVE_OPERATOR_ASSAY_MINIMUM": configuration[
                    "operator_assay_evaluation_minimum"
                ]
            }
            if "operator_assay_evaluation_minimum" in configuration
            else {}
        ),
    }
    return {
        "schema_version": 1,
        "study_id": study.record["study_id"],
        "study_sha256": study.study_sha256,
        "cell": cell.to_record(),
        "attempt_id": attempt_id,
        "prepared_attempt_id": prepared_attempt_id,
        "runner": runner.relative_to(WORKSPACE_ROOT).as_posix(),
        "execution_contract": (
            None if execution_contract is None else execution_contract.identity()
        ),
        "prepare_command": prepare,
        "live_command": live,
        "prepare_run_dir": prepare_dir.relative_to(WORKSPACE_ROOT).as_posix(),
        "live_run_dir": live_dir.relative_to(WORKSPACE_ROOT).as_posix(),
        "environment": {
            **(
                {"AGENT_EVOLVE_MODEL_PROFILE": cell.model_profile}
                if cell.model_profile is not None
                else {}
            ),
            "AGENT_EVOLVE_REPLICATE_SEED": str(cell.replicate_seed),
            "AGENT_EVOLVE_STUDY_CELL_ID": cell.cell_id,
            **environment_configuration,
        },
    }


def _resource_lease_for_plan(
    plan: dict[str, object],
) -> FileExclusiveResourceLease | None:
    """Compose the optional workload-declared lease without domain branching."""

    execution_contract = plan.get("execution_contract")
    if execution_contract is None:
        return None
    if type(execution_contract) is not dict:
        raise TypeError("execution_contract must be an object or None")
    resource = execution_contract.get("exclusive_resource")
    if resource is None:
        return None
    if type(resource) is not dict or set(resource) != {
        "lease_relative_path",
        "resource_key",
    }:
        raise ValueError("planned exclusive resource has an invalid field set")
    relative_value = resource.get("lease_relative_path")
    if type(relative_value) is not str or not relative_value or "\\" in relative_value:
        raise ValueError("planned resource lease path must be a POSIX relative path")
    relative_path = Path(relative_value)
    if relative_path.is_absolute() or any(
        part in ("", ".", "..") for part in relative_path.parts
    ):
        raise ValueError("planned resource lease path escapes the workspace")
    lease_path = (WORKSPACE_ROOT / relative_path).resolve()
    if WORKSPACE_ROOT.resolve() not in lease_path.parents:
        raise ValueError("planned resource lease path escapes the workspace")
    cell = plan.get("cell")
    if type(cell) is not dict or type(cell.get("cell_id")) is not str:
        raise ValueError("planned resource lease requires an exact cell identity")
    attempt_id = plan.get("attempt_id")
    if type(attempt_id) is not str:
        raise ValueError("planned resource lease requires an exact attempt identity")
    resource_key = resource.get("resource_key")
    if type(resource_key) is not str:
        raise ValueError("planned resource key must be an exact string")
    return FileExclusiveResourceLease(
        resource_key=resource_key,
        owner_id=f"{cell['cell_id']}:{attempt_id}",
        lease_path=lease_path,
        owner_metadata={
            "study_id": plan.get("study_id"),
            "cell_id": cell["cell_id"],
            "attempt_id": attempt_id,
            "live_run_dir": plan.get("live_run_dir"),
            "runner": plan.get("runner"),
        },
    )


def _lease_rejection_record(error: BaseException) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "status": "rejected_before_live_subprocess",
        "failure_type": type(error).__qualname__,
        "failure_sha256": hashlib.sha256(
            f"{type(error).__qualname__}\x00{error}".encode(
                "utf-8", errors="replace"
            )
        ).hexdigest(),
    }
    if type(error) is ResourceLeaseUnavailable:
        record.update(
            {
                "resource_key": error.resource_key,
                "lease_path": str(error.lease_path),
                "holder_record_sha256": error.holder_record_sha256,
            }
        )
    elif type(error) is ResourceConflictDetected:
        record.update(
            {
                "resource_key": error.resource_key,
                "conflict_observation": error.observation.to_record(),
            }
        )
    return record


def _run_live_command(
    plan: dict[str, object],
    receipt_dir: Path,
    environment: dict[str, str],
) -> tuple[int, dict[str, object] | None, dict[str, object] | None]:
    """Run live work under its optional nonblocking process-external lease."""

    lease = _resource_lease_for_plan(plan)
    acquisition: dict[str, object] | None = None
    release: dict[str, object] | None = None
    try:
        if lease is not None:
            acquisition = lease.acquire().to_record()
            write_json_atomic(
                receipt_dir / "resource_lease_acquisition.json",
                acquisition,
            )
        live = subprocess.run(
            plan["live_command"],
            cwd=AGENT_EVOLVE_ROOT,
            env=environment,
            check=False,
        )
    except (ResourceLeaseUnavailable, ResourceConflictDetected) as error:
        write_json_atomic(
            receipt_dir / "resource_lease_rejection.json",
            _lease_rejection_record(error),
        )
        raise
    except BaseException as error:
        if lease is not None and lease.active:
            release = lease.release(
                outcome="failed",
                failure_type=type(error).__qualname__,
            )
            write_json_atomic(receipt_dir / "resource_lease_release.json", release)
        raise
    else:
        if lease is not None:
            release = lease.release(
                outcome="completed" if live.returncode == 0 else "failed",
                failure_type=(
                    None if live.returncode == 0 else "LiveSubprocessNonzeroExit"
                ),
            )
            write_json_atomic(receipt_dir / "resource_lease_release.json", release)
        return live.returncode, acquisition, release


def _verified_prepared_summary(plan: dict[str, object]) -> dict[str, object]:
    prepared_attempt_id = plan.get("prepared_attempt_id")
    if type(prepared_attempt_id) is not str:
        raise ValueError("prepared execution lacks its source attempt ID")
    cell = plan["cell"]
    receipt = RECEIPT_ROOT / str(cell["cell_id"]) / prepared_attempt_id
    receipt_summary = _load_json(receipt / "summary.json")
    receipt_finalization = _load_json(receipt / "finalized.json")
    if (
        receipt_summary.get("status") != "prepared_healthy"
        or receipt_finalization.get("status") != "prepared_healthy"
        or receipt_summary.get("cell_id") != cell["cell_id"]
        or receipt_summary.get("cell_sha256") != cell["cell_sha256"]
    ):
        raise RuntimeError("prepared receipt is not a healthy exact-cell qualification")
    prepare_summary_path = (
        WORKSPACE_ROOT / str(plan["prepare_run_dir"]) / "summary.json"
    )
    if not prepare_summary_path.is_file():
        raise RuntimeError("prepared workload summary is missing")
    digest = hashlib.sha256(prepare_summary_path.read_bytes()).hexdigest()
    if digest != receipt_summary.get("prepare_summary_sha256"):
        raise RuntimeError("prepared workload summary drifted after qualification")
    return _load_json(prepare_summary_path)


def _execute(
    plan: dict[str, object],
    receipt_dir: Path,
    *,
    execute_live: bool,
    reuse_prepared: bool = False,
) -> int:
    environment = os.environ.copy()
    environment.update(plan["environment"])
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("systematic_workload_contract.py"),
        Path(WORKSPACE_ROOT / str(plan["runner"])),
    ]
    execution_contract = plan.get("execution_contract")
    if type(execution_contract) is dict:
        source_paths.append(WORKSPACE_ROOT / str(execution_contract["path"]))
        if execution_contract.get("exclusive_resource") is not None:
            source_paths.append(
                AGENT_EVOLVE_ROOT
                / "src/agent_evolve/infrastructure/resource_lease.py"
            )
    write_json_atomic(
        receipt_dir / "manifest.json",
        {
            **plan,
            "created_at_utc": _utc_now(),
            "source": source_identity(tuple(source_paths), relative_to=WORKSPACE_ROOT),
        },
    )
    prepare_dir = WORKSPACE_ROOT / str(plan["prepare_run_dir"])
    prepare_summary_path = prepare_dir / "summary.json"
    if reuse_prepared:
        prepare_exit_code = 0
        prepare_summary: dict[str, object] | None = _verified_prepared_summary(plan)
    else:
        prepare = subprocess.run(
            plan["prepare_command"],
            cwd=AGENT_EVOLVE_ROOT,
            env=environment,
            check=False,
        )
        prepare_exit_code = prepare.returncode
        prepare_summary = (
            _load_json(prepare_summary_path) if prepare_summary_path.is_file() else None
        )
    live_exit: int | None = None
    live_summary: dict[str, object] | None = None
    lease_acquisition: dict[str, object] | None = None
    lease_release: dict[str, object] | None = None
    if execute_live and prepare_exit_code == 0 and prepare_summary is not None:
        live_exit, lease_acquisition, lease_release = _run_live_command(
            plan,
            receipt_dir,
            environment,
        )
        live_dir = WORKSPACE_ROOT / str(plan["live_run_dir"])
        live_summary_path = live_dir / "summary.json"
        if live_summary_path.is_file():
            live_summary = _load_json(live_summary_path)
    if execute_live:
        status = (
            "completed_healthy"
            if live_exit == 0
            and live_summary is not None
            and live_summary.get("status") == "completed_healthy"
            else "completed_unhealthy"
        )
    else:
        status = (
            "prepared_healthy"
            if prepare_exit_code == 0
            and prepare_summary is not None
            and str(prepare_summary.get("status", "")).startswith("prepared")
            else "prepared_unhealthy"
        )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "cell_id": plan["cell"]["cell_id"],
        "cell_sha256": plan["cell"]["cell_sha256"],
        "prepare_exit_code": prepare_exit_code,
        "prepare_reused": reuse_prepared,
        "prepared_attempt_id": plan.get("prepared_attempt_id"),
        "live_exit_code": live_exit,
        "prepare_status": (
            None if prepare_summary is None else prepare_summary.get("status")
        ),
        "live_status": None if live_summary is None else live_summary.get("status"),
        "resource_lease_required": (
            type(execution_contract) is dict
            and execution_contract.get("exclusive_resource") is not None
        ),
        "resource_lease_acquisition": lease_acquisition,
        "resource_lease_release": lease_release,
        "prepare_summary_sha256": (
            None
            if not prepare_summary_path.is_file()
            else hashlib.sha256(prepare_summary_path.read_bytes()).hexdigest()
        ),
        "live_run_dir": plan["live_run_dir"],
    }
    write_json_atomic(receipt_dir / "summary.json", summary)
    finalize_run_directory(receipt_dir, status=status)
    print(json.dumps(summary, sort_keys=True))
    return 0 if status in ("completed_healthy", "prepared_healthy") else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "prepare", "live", "execute"))
    parser.add_argument("registry", type=Path)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--prepared-attempt-id")
    parser.add_argument("--authorization")
    args = parser.parse_args()
    if _TOKEN.fullmatch(args.attempt_id) is None:
        raise ValueError("attempt ID violates the closed token grammar")
    if args.prepared_attempt_id is not None and _TOKEN.fullmatch(
        args.prepared_attempt_id
    ) is None:
        raise ValueError("prepared attempt ID violates the closed token grammar")
    if args.mode == "execute" and args.prepared_attempt_id is None:
        raise ValueError("execute mode requires --prepared-attempt-id")
    if args.mode in ("prepare", "live") and args.prepared_attempt_id is not None:
        raise ValueError(f"{args.mode} mode does not accept --prepared-attempt-id")
    if args.attempt_id == args.prepared_attempt_id:
        raise ValueError("execution and prepared attempt IDs must differ")
    study = SystematicExperimentStudy.load(args.registry.resolve(strict=True))
    cell = _cell(study, args.cell_id)
    plan = _plan(
        study=study,
        cell=cell,
        attempt_id=args.attempt_id,
        prepared_attempt_id=args.prepared_attempt_id,
    )
    if args.mode == "plan":
        print(json.dumps(plan, sort_keys=True))
        return 0
    expected_authorization = {
        "prepare": PREPARE_AUTHORIZATION,
        "live": AUTHORIZATION,
        "execute": PREPARED_AUTHORIZATION,
    }[args.mode]
    if args.authorization != expected_authorization:
        raise RuntimeError(f"{args.mode} authorization string is invalid")
    receipt_dir = (RECEIPT_ROOT / cell.cell_id / args.attempt_id).resolve()
    receipt_dir.mkdir(parents=True, exist_ok=False)
    try:
        return _execute(
            plan,
            receipt_dir,
            execute_live=args.mode in ("live", "execute"),
            reuse_prepared=args.mode == "execute",
        )
    except BaseException as error:
        if not (receipt_dir / "summary.json").exists():
            write_json_atomic(
                receipt_dir / "summary.json",
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
        if not (receipt_dir / "finalized.json").exists():
            finalize_run_directory(receipt_dir, status="failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
