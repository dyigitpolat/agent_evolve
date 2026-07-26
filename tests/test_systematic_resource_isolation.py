"""Systematic-run resource isolation is contract-owned and workload-neutral."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from agent_evolve.infrastructure.resource_lease import (
    ResourceLeaseUnavailable,
)
from examples.development.run_systematic_reference_cell import (
    _plan,
    _resource_lease_for_plan,
    _run_live_command,
)
from examples.development.systematic_experiment_study import (
    SystematicExperimentStudy,
)
from examples.development.systematic_workload_contract import (
    WorkloadExecutionContract,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_V11 = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "systematic_experiment_registry_v11.json"
)
REGISTRY_V12 = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "systematic_experiment_registry_v12.json"
)
REGISTRY_V14 = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "systematic_experiment_registry_v14.json"
)


def test_v12_changes_execution_isolation_without_changing_study_cells() -> None:
    previous = SystematicExperimentStudy.load(REGISTRY_V11)
    isolated = SystematicExperimentStudy.load(REGISTRY_V12)

    assert tuple(cell.to_record() for cell in isolated.reference_cells()) == tuple(
        cell.to_record() for cell in previous.reference_cells()
    )
    controls = {
        cell.workload_id: cell
        for cell in isolated.reference_cells()
        if cell.arm == "control"
    }
    plans = {
        workload: _plan(study=isolated, cell=cell, attempt_id="isolation_plan")
        for workload, cell in controls.items()
    }
    heat_resource = plans["heat2d"]["execution_contract"]["exclusive_resource"]
    timeloop_resource = plans["timeloop_v2"]["execution_contract"][
        "exclusive_resource"
    ]
    assert heat_resource == timeloop_resource == {
        "resource_key": "shared_cpu8_evaluator_lane",
        "lease_relative_path": (
            "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs/"
            "systematic_resource_leases/shared_cpu8_evaluator_lane.lock"
        ),
    }
    assert plans["boils_abc"]["execution_contract"]["exclusive_resource"] is None


def test_v14_isolates_every_fixed_cpu_evaluator_lane() -> None:
    study = SystematicExperimentStudy.load(REGISTRY_V14)
    controls = {
        cell.workload_id: cell
        for cell in study.reference_cells()
        if cell.arm == "control"
    }
    plans = {
        workload: _plan(study=study, cell=cell, attempt_id="v14_isolation_plan")
        for workload, cell in controls.items()
    }
    heat_resource = plans["heat2d"]["execution_contract"]["exclusive_resource"]
    timeloop_resource = plans["timeloop_v2"]["execution_contract"][
        "exclusive_resource"
    ]
    assert heat_resource == timeloop_resource
    assert plans["boils_abc"]["execution_contract"]["exclusive_resource"] == {
        "resource_key": "boils_abc_cpu120_123_evaluator_lane",
        "lease_relative_path": (
            "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs/"
            "systematic_resource_leases/boils_abc_cpu120_123_evaluator_lane.lock"
        ),
    }


def test_schema_two_resource_path_cannot_escape_the_lease_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runner = workspace / "agent_evolve/examples/development/fake_runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# test runner\n", encoding="utf-8")
    contract_path = workspace / "agent_evolve/contracts/invalid.json"
    contract_path.parent.mkdir(parents=True)
    record = {
        "schema_version": 2,
        "workload_id": "fake_workload",
        "runtime": {"kind": "current_python"},
        "exclusive_resource": {
            "resource_key": "shared_lane",
            "lease_relative_path": "../escaped.lock",
        },
        "arms": {
            arm: {
                "runner": "agent_evolve/examples/development/fake_runner.py",
                "artifact_root": (
                    "papers/agent_evolve_aaai_2027/research_artifacts/"
                    f"experiment_logs/fake/{arm}"
                ),
                "prepare_arguments": ["prepare", "{prepare_run_id}"],
                "live_arguments": ["live", "{live_run_id}"],
                "preregistration_relative_path": None,
            }
            for arm in ("control", "treatment")
        },
    }
    payload = json.dumps(record).encode("utf-8")
    contract_path.write_bytes(payload)

    with pytest.raises(ValueError, match="lease_relative_path"):
        WorkloadExecutionContract.load(
            contract_path,
            workspace_root=workspace,
            expected_workload_id="fake_workload",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_live_subprocess_cannot_start_while_shared_lane_is_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import examples.development.run_systematic_reference_cell as launcher

    monkeypatch.setattr(launcher, "WORKSPACE_ROOT", tmp_path)
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    plan = {
        "cell": {"cell_id": "reference_fake_shared_control_s1_deadbeef00"},
        "attempt_id": "attempt_b",
        "execution_contract": {
            "exclusive_resource": {
                "resource_key": "shared_cpu8_evaluator_lane",
                "lease_relative_path": "leases/shared.lock",
            }
        },
        "live_command": ["fake-live-command"],
        "live_run_dir": "runs/fake",
    }
    holder_plan = {
        **plan,
        "attempt_id": "attempt_a",
    }
    holder = _resource_lease_for_plan(holder_plan)
    assert holder is not None
    holder.acquire()
    subprocess_calls = 0

    def forbidden_run(*args, **kwargs):
        del args, kwargs
        nonlocal subprocess_calls
        subprocess_calls += 1
        return subprocess.CompletedProcess(["fake"], 0)

    monkeypatch.setattr(launcher.subprocess, "run", forbidden_run)
    try:
        with pytest.raises(ResourceLeaseUnavailable):
            _run_live_command(plan, receipt_dir, {})
    finally:
        holder.release()

    assert subprocess_calls == 0
    rejection = json.loads(
        (receipt_dir / "resource_lease_rejection.json").read_text(encoding="utf-8")
    )
    assert rejection["failure_type"] == "ResourceLeaseUnavailable"
    assert rejection["resource_key"] == "shared_cpu8_evaluator_lane"


def test_live_subprocess_publishes_acquire_and_release_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import examples.development.run_systematic_reference_cell as launcher

    monkeypatch.setattr(launcher, "WORKSPACE_ROOT", tmp_path)
    receipt_dir = tmp_path / "receipt"
    receipt_dir.mkdir()
    plan = {
        "cell": {"cell_id": "reference_fake_shared_control_s1_deadbeef00"},
        "attempt_id": "attempt_a",
        "execution_contract": {
            "exclusive_resource": {
                "resource_key": "shared_cpu8_evaluator_lane",
                "lease_relative_path": "leases/shared.lock",
            }
        },
        "live_command": ["fake-live-command"],
        "live_run_dir": "runs/fake",
    }
    observed_active: list[bool] = []

    def successful_run(*args, **kwargs):
        del args, kwargs
        lease_record = json.loads(
            (tmp_path / "leases/shared.lock").read_text(encoding="ascii")
        )
        observed_active.append(lease_record["status"] == "acquired")
        return subprocess.CompletedProcess(["fake"], 0)

    monkeypatch.setattr(launcher.subprocess, "run", successful_run)
    exit_code, acquisition, release = _run_live_command(plan, receipt_dir, {})

    assert exit_code == 0
    assert observed_active == [True]
    assert acquisition is not None and acquisition["resource_key"] == (
        "shared_cpu8_evaluator_lane"
    )
    assert release is not None and release["outcome"] == "completed"
    assert json.loads(
        (receipt_dir / "resource_lease_acquisition.json").read_text(
            encoding="utf-8"
        )
    ) == acquisition
    assert json.loads(
        (receipt_dir / "resource_lease_release.json").read_text(encoding="utf-8")
    ) == release
