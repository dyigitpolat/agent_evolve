from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evolve.agentic import (
    FileExclusiveResourceLease,
    ResourceConflictDetected,
    ResourceConflictObservation,
    ResourceLeaseUnavailable,
    freeze_json,
)


class _Probe:
    def __init__(self, *, conflict: bool) -> None:
        self.conflict = conflict
        self.calls = 0

    def __call__(self) -> ResourceConflictObservation:
        self.calls += 1
        return ResourceConflictObservation(
            probe_id="resource_lease_test_probe",
            probe_version=1,
            conflict=self.conflict,
            facts=freeze_json({"call": self.calls}),
        )


def _lease(
    path: Path,
    *,
    owner: str,
    probe: _Probe,
) -> FileExclusiveResourceLease:
    return FileExclusiveResourceLease(
        resource_key="shared_simulator_token",
        owner_id=owner,
        lease_path=path,
        owner_metadata={"phase": "test", "slot": 1},
        conflict_probe=probe,
    )


def test_file_lease_is_nonblocking_durable_and_reacquirable(tmp_path: Path) -> None:
    path = tmp_path / "leases" / "simulator.lock"
    first_probe = _Probe(conflict=False)
    first = _lease(path, owner="run-a", probe=first_probe)

    receipt = first.acquire()
    assert first.active
    assert first_probe.calls == 1
    acquired = json.loads(path.read_text(encoding="ascii"))
    assert acquired["status"] == "acquired"
    assert acquired["owner_id"] == "run-a"
    assert acquired["acquisition_sha256"] == receipt.acquisition_sha256
    assert acquired["owner_metadata"] == {"phase": "test", "slot": 1}

    second = _lease(path, owner="run-b", probe=_Probe(conflict=False))
    with pytest.raises(ResourceLeaseUnavailable) as unavailable:
        second.acquire()
    assert unavailable.value.resource_key == "shared_simulator_token"
    assert len(unavailable.value.holder_record_sha256) == 64
    assert not second.active

    released = first.release(outcome="completed")
    assert not first.active
    assert released["status"] == "released"
    assert released["acquisition_sha256"] == receipt.acquisition_sha256
    assert json.loads(path.read_text(encoding="ascii")) == released

    third = _lease(path, owner="run-c", probe=_Probe(conflict=False))
    with third as third_receipt:
        assert third.active
        assert third_receipt.owner_id == "run-c"
    assert not third.active
    assert third.last_release_record is not None
    assert third.last_release_record["status"] == "released"


def test_external_conflict_fails_before_lease_publication(tmp_path: Path) -> None:
    path = tmp_path / "simulator.lock"
    conflict = _Probe(conflict=True)
    rejected = _lease(path, owner="rejected-run", probe=conflict)

    with pytest.raises(ResourceConflictDetected) as detected:
        rejected.acquire()
    assert detected.value.observation.conflict
    assert conflict.calls == 1
    assert not rejected.active
    record = json.loads(path.read_text(encoding="ascii"))
    assert record["status"] == "external_conflict_rejected"

    successor = _lease(path, owner="successor", probe=_Probe(conflict=False))
    successor.acquire()
    successor.release()


def test_context_failure_is_recorded_before_unlock(tmp_path: Path) -> None:
    path = tmp_path / "simulator.lock"
    lease = _lease(path, owner="failed-run", probe=_Probe(conflict=False))

    with pytest.raises(RuntimeError, match="fixture failure"):
        with lease:
            raise RuntimeError("fixture failure")

    record = json.loads(path.read_text(encoding="ascii"))
    assert record["status"] == "released"
    assert record["outcome"] == "failed"
    assert record["failure_type"] == "RuntimeError"
    assert not lease.active
