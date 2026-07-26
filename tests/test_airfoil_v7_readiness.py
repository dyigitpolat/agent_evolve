from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_evolve.agentic import ResourceConflictDetected
from examples.benchmarks.engibench_airfoil.v7_readiness import (
    AIRFOIL_V7_RESOURCE_KEY,
    AirfoilV7ConflictProbe,
    AirfoilV7ReadinessSpec,
    CommandObservation,
    create_airfoil_v7_resource_lease,
    observe_airfoil_v7_environment,
)


_DIGEST = "a" * 64
_IMAGE = f"example/cfd@sha256:{_DIGEST}"


class _Commands:
    def __init__(self, *, container_rows: str = "") -> None:
        self.container_rows = container_rows
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        argv: tuple[str, ...],
        timeout_seconds: float,
    ) -> CommandObservation:
        assert timeout_seconds == 2.0
        self.calls.append(argv)
        if argv[-1] == "--version":
            return CommandObservation(argv, 0, "Python 3.12.0", "")
        if len(argv) >= 3 and argv[1] == "-c":
            return CommandObservation(
                argv,
                0,
                json.dumps(
                    {
                        "executable": str(Path(argv[0]).resolve()),
                        "python_version": "3.12.0",
                        "modules": {
                            "engibench": {
                                "path": "/fixture/engibench/__init__.py",
                                "sha256": "c" * 64,
                                "version": "0.0.fixture",
                            },
                            "numpy": {
                                "path": "/fixture/numpy/__init__.py",
                                "sha256": "d" * 64,
                                "version": "2.0.fixture",
                            },
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "",
            )
        if argv[1:3] == ("version", "--format"):
            return CommandObservation(argv, 0, "27.1.0", "")
        if argv[1:3] == ("image", "inspect"):
            return CommandObservation(
                argv,
                0,
                f"sha256:{'b' * 64}\t[\"{_IMAGE}\"]",
                "",
            )
        if argv[1:3] == ("ps", "-a"):
            return CommandObservation(argv, 0, self.container_rows, "")
        raise AssertionError(f"unexpected readiness command: {argv!r}")


def _spec(tmp_path: Path) -> AirfoilV7ReadinessSpec:
    python = tmp_path / "evaluator-python"
    python.write_bytes(b"#!/bin/sh\n")
    python.chmod(0o700)
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text("print('fixture')\n", encoding="utf-8")
    dataset = tmp_path / "dataset.arrow"
    dataset.write_bytes(b"frozen-airfoil-fixture")
    return AirfoilV7ReadinessSpec(
        evaluator_python=python,
        evaluator_script=evaluator,
        dataset_arrow=dataset,
        expected_dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        container_image=_IMAGE,
        cpu_set="2-4,6",
        mpi_cores=4,
        command_timeout_seconds=2.0,
    )


def test_observed_readiness_binds_files_docker_cpu_and_conflicts(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    commands = _Commands()

    observation = observe_airfoil_v7_environment(
        spec,
        command_runner=commands,
        affinity_reader=lambda: {0, 1, 2, 3, 4, 5, 6, 7},
        process_scanner=lambda markers: (),
        executable_resolver=lambda name: f"/usr/bin/{name}",
    )

    assert observation.passed
    record = observation.to_record()
    assert record["passed"] is True
    assert record["dataset"]["hash_matches"] is True
    assert record["container_image_matches"] is True
    assert record["requested_cpus"] == [2, 3, 4, 6]
    assert record["conflict_observation"]["conflict"] is False
    assert len(commands.calls) == 5


def test_observed_readiness_fails_closed_on_missing_cpu_or_container(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    commands = _Commands(container_rows="abc\timage\tUp\tmachaero")

    observation = observe_airfoil_v7_environment(
        spec,
        command_runner=commands,
        affinity_reader=lambda: {2, 3},
        process_scanner=lambda markers: (),
        executable_resolver=lambda name: f"/usr/bin/{name}",
    )

    assert not observation.passed
    checks = observation.to_record()["checks"]
    assert checks["cpu_allocation"] is False
    assert checks["external_conflicts_absent"] is False


def test_airfoil_lease_injects_domain_conflict_without_core_leakage(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    commands = _Commands()
    lease = create_airfoil_v7_resource_lease(
        spec,
        lease_path=tmp_path / "machaero.lock",
        run_id="airfoil-v7-test",
        phase="seed_qualification",
        command_runner=commands,
        process_scanner=lambda markers: (),
    )

    with lease as receipt:
        assert receipt.resource_key == AIRFOIL_V7_RESOURCE_KEY
        assert receipt.owner_id == "airfoil-v7-test"
        assert receipt.conflict_observation.conflict is False

    conflicting = create_airfoil_v7_resource_lease(
        spec,
        lease_path=tmp_path / "machaero.lock",
        run_id="airfoil-v7-conflict",
        phase="provider_run",
        command_runner=_Commands(container_rows="abc\timage\tUp\tmachaero"),
        process_scanner=lambda markers: (),
    )
    with pytest.raises(ResourceConflictDetected):
        conflicting.acquire()


def test_conflict_probe_fails_closed_when_docker_state_is_unobservable(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)

    def unavailable(
        argv: tuple[str, ...],
        timeout_seconds: float,
    ) -> CommandObservation:
        del timeout_seconds
        return CommandObservation(argv, 127, "", "docker unavailable")

    observation = AirfoilV7ConflictProbe(
        spec,
        command_runner=unavailable,
        process_scanner=lambda markers: (),
    )()
    assert observation.conflict
    assert observation.to_record()["facts"]["probe_complete"] is False
