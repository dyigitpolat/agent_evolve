"""Observed environment and exclusive-resource boundary for Airfoil-v7.

The generic lease implementation knows nothing about CFD or Docker.  This
adapter supplies the benchmark-owned facts: a fixed upstream container name,
the pinned image, evaluator files, dataset bytes, and CPU allocation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from agent_evolve.agentic import (
    FileExclusiveResourceLease,
    FrozenJsonObject,
    ResourceConflictObservation,
    freeze_json,
    thaw_json,
)


AIRFOIL_V7_RESOURCE_KEY = "engibench_airfoil_machaero"
AIRFOIL_V7_CONFLICT_PROBE_ID = "airfoil_v7_external_conflict"
AIRFOIL_V7_CONFLICT_PROBE_VERSION = 1
AIRFOIL_V7_ENVIRONMENT_PROBE_ID = "airfoil_v7_observed_environment"
AIRFOIL_V7_ENVIRONMENT_PROBE_VERSION = 1

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAX_COMMAND_OUTPUT_BYTES = 16_384
_EVALUATOR_ENVIRONMENT_PROBE = """\
import hashlib, importlib, importlib.metadata, json, pathlib, sys
rows = {}
for name in ("numpy", "engibench"):
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = getattr(module, "__version__", "unknown")
    rows[name] = {
        "version": str(version),
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
print(json.dumps({
    "executable": str(pathlib.Path(sys.executable).absolute()),
    "python_version": sys.version.split()[0],
    "modules": rows,
}, sort_keys=True, separators=(",", ":")))
"""


@dataclass(frozen=True, slots=True)
class CommandObservation:
    """Bounded deterministic projection of one read-only local command."""

    argv: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False

    def __post_init__(self) -> None:
        if type(self.argv) is not tuple or not self.argv or any(
            type(value) is not str or not value for value in self.argv
        ):
            raise TypeError("argv must be a non-empty exact tuple of strings")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise TypeError("exit_code must be an exact integer or None")
        for name in ("stdout", "stderr"):
            value = getattr(self, name)
            if type(value) is not str:
                raise TypeError(f"{name} must be an exact string")
            if len(value.encode("utf-8", errors="strict")) > _MAX_COMMAND_OUTPUT_BYTES:
                raise ValueError(f"{name} exceeds the bounded observation limit")
        if type(self.timed_out) is not bool:
            raise TypeError("timed_out must be an exact bool")
        if self.timed_out and self.exit_code is not None:
            raise ValueError("a timed-out command cannot carry an exit code")

    @property
    def succeeded(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    def to_record(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "succeeded": self.succeeded,
        }


CommandRunner = Callable[[tuple[str, ...], float], CommandObservation]
AffinityReader = Callable[[], set[int]]
ProcessScanner = Callable[[tuple[str, ...]], tuple[int, ...]]
ExecutableResolver = Callable[[str], str | None]


def _bounded_text(payload: bytes) -> str:
    bounded = payload[:_MAX_COMMAND_OUTPUT_BYTES]
    return bounded.decode("utf-8", errors="replace").strip()


def run_observation_command(
    argv: tuple[str, ...],
    timeout_seconds: float,
) -> CommandObservation:
    """Run one read-only readiness command without shell interpretation."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")
    try:
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        return CommandObservation(
            argv=argv,
            exit_code=None,
            stdout=_bounded_text(exc.stdout or b""),
            stderr=_bounded_text(exc.stderr or b""),
            timed_out=True,
        )
    except OSError as exc:
        return CommandObservation(
            argv=argv,
            exit_code=127,
            stdout="",
            stderr=f"{type(exc).__name__}: {str(exc)[:512]}",
        )
    return CommandObservation(
        argv=argv,
        exit_code=completed.returncode,
        stdout=_bounded_text(completed.stdout),
        stderr=_bounded_text(completed.stderr),
    )


def _read_affinity() -> set[int]:
    return set(os.sched_getaffinity(0))


def _scan_process_markers(markers: tuple[str, ...]) -> tuple[int, ...]:
    encoded = tuple(marker.encode("utf-8", errors="strict") for marker in markers)
    matches: list[int] = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes()[:65_536]
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if command and any(marker in command for marker in encoded):
            matches.append(pid)
    return tuple(sorted(set(matches)))


def _parse_cpu_set(value: str) -> tuple[int, ...]:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError("cpu_set must be non-empty canonical text")
    cpus: set[int] = set()
    for part in value.split(","):
        if not part or part != part.strip():
            raise ValueError("cpu_set contains an empty or padded component")
        if "-" in part:
            components = part.split("-")
            if len(components) != 2 or not all(item.isdigit() for item in components):
                raise ValueError("cpu_set ranges must be decimal start-end")
            start, end = (int(item) for item in components)
            if start > end:
                raise ValueError("cpu_set range start exceeds its end")
            cpus.update(range(start, end + 1))
        elif part.isdigit():
            cpus.add(int(part))
        else:
            raise ValueError("cpu_set components must be decimal CPUs or ranges")
    if not cpus:
        raise ValueError("cpu_set must select at least one CPU")
    return tuple(sorted(cpus))


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _invoked_path(path: Path) -> Path:
    """Return an absolute path without dereferencing a virtualenv symlink."""

    return Path(os.path.abspath(os.path.expanduser(path)))


def _file_observation(path: Path, *, require_executable: bool) -> dict[str, object]:
    invoked = _invoked_path(path)
    resolved = invoked.resolve()
    exists = invoked.is_file()
    digest: str | None = None
    size: int | None = None
    if exists:
        digest, size = _sha256_file(invoked)
    executable = exists and os.access(invoked, os.X_OK)
    return {
        "path": str(invoked),
        "resolved_target": str(resolved),
        "is_symlink": invoked.is_symlink(),
        "exists": exists,
        "is_file": exists,
        "sha256": digest,
        "bytes": size,
        "executable": executable,
        "require_executable": require_executable,
        "passed": exists and (executable or not require_executable),
    }


@dataclass(frozen=True, slots=True)
class AirfoilV7ReadinessSpec:
    evaluator_python: Path
    evaluator_script: Path
    dataset_arrow: Path
    expected_dataset_sha256: str
    container_image: str
    cpu_set: str
    mpi_cores: int
    fixed_container_name: str = "machaero"
    docker_executable: str = "docker"
    process_markers: tuple[str, ...] = (
        "airfoil_external_panel_v2.py",
        "airfoil_external_panel_v1.py",
    )
    command_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        for name in ("evaluator_python", "evaluator_script", "dataset_arrow"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a Path")
        if (
            type(self.expected_dataset_sha256) is not str
            or _SHA256.fullmatch(self.expected_dataset_sha256) is None
        ):
            raise ValueError("expected_dataset_sha256 must be a lowercase SHA-256")
        if (
            type(self.container_image) is not str
            or "@sha256:" not in self.container_image
            or _SHA256.fullmatch(self.container_image.rsplit("@sha256:", 1)[1])
            is None
        ):
            raise ValueError("container_image must be pinned by a SHA-256 digest")
        _parse_cpu_set(self.cpu_set)
        if type(self.mpi_cores) is not int or self.mpi_cores <= 0:
            raise ValueError("mpi_cores must be a positive exact integer")
        if self.mpi_cores > len(_parse_cpu_set(self.cpu_set)):
            raise ValueError("mpi_cores exceeds the declared CPU set")
        if (
            type(self.fixed_container_name) is not str
            or _CONTAINER_NAME.fullmatch(self.fixed_container_name) is None
        ):
            raise ValueError("fixed_container_name has invalid syntax")
        if type(self.docker_executable) is not str or not self.docker_executable:
            raise ValueError("docker_executable must be non-empty")
        if type(self.process_markers) is not tuple or not self.process_markers or any(
            type(value) is not str or not value for value in self.process_markers
        ):
            raise TypeError("process_markers must be a non-empty tuple of strings")
        if (
            isinstance(self.command_timeout_seconds, bool)
            or not isinstance(self.command_timeout_seconds, (int, float))
            or not math.isfinite(float(self.command_timeout_seconds))
            or float(self.command_timeout_seconds) <= 0
        ):
            raise ValueError("command_timeout_seconds must be finite and positive")


def _docker_container_observation(
    spec: AirfoilV7ReadinessSpec,
    runner: CommandRunner,
) -> tuple[CommandObservation, tuple[str, ...]]:
    command = runner(
        (
            spec.docker_executable,
            "ps",
            "-a",
            "--filter",
            f"name=^/{spec.fixed_container_name}$",
            "--format",
            "{{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}",
        ),
        float(spec.command_timeout_seconds),
    )
    rows = tuple(line for line in command.stdout.splitlines() if line.strip())
    return command, rows


@dataclass(frozen=True, slots=True)
class AirfoilV7ConflictProbe:
    """Fail closed on the fixed Docker name or another Airfoil evaluator."""

    spec: AirfoilV7ReadinessSpec
    command_runner: CommandRunner = field(
        default=run_observation_command,
        repr=False,
        compare=False,
    )
    process_scanner: ProcessScanner = field(
        default=_scan_process_markers,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.spec) is not AirfoilV7ReadinessSpec:
            raise TypeError("spec must be an exact AirfoilV7ReadinessSpec")
        if not callable(self.command_runner) or not callable(self.process_scanner):
            raise TypeError("conflict probe dependencies must be callable")

    def __call__(self) -> ResourceConflictObservation:
        docker, containers = _docker_container_observation(
            self.spec,
            self.command_runner,
        )
        process_ids = self.process_scanner(self.spec.process_markers)
        if type(process_ids) is not tuple or any(
            type(value) is not int or value <= 0 for value in process_ids
        ):
            raise TypeError("process scanner must return positive PID tuples")
        probe_complete = docker.succeeded
        conflict = not probe_complete or bool(containers) or bool(process_ids)
        facts = freeze_json(
            {
                "fixed_container_name": self.spec.fixed_container_name,
                "docker_ps": docker.to_record(),
                "matching_container_rows": list(containers),
                "process_markers": list(self.spec.process_markers),
                "matching_process_ids": list(process_ids),
                "probe_complete": probe_complete,
            }
        )
        return ResourceConflictObservation(
            probe_id=AIRFOIL_V7_CONFLICT_PROBE_ID,
            probe_version=AIRFOIL_V7_CONFLICT_PROBE_VERSION,
            conflict=conflict,
            facts=facts,
        )


@dataclass(frozen=True, slots=True)
class AirfoilV7EnvironmentObservation:
    passed: bool
    record: FrozenJsonObject

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("passed must be an exact bool")
        frozen = freeze_json(self.record)
        if frozen is not self.record:
            raise TypeError("record must already be frozen typed JSON")

    def to_record(self) -> dict[str, object]:
        value = thaw_json(self.record)
        if type(value) is not dict:  # pragma: no cover - constructor closes this.
            raise AssertionError("environment record stopped being an object")
        return {"passed": self.passed, **value}


def observe_airfoil_v7_environment(
    spec: AirfoilV7ReadinessSpec,
    *,
    command_runner: CommandRunner = run_observation_command,
    affinity_reader: AffinityReader = _read_affinity,
    process_scanner: ProcessScanner = _scan_process_markers,
    executable_resolver: ExecutableResolver = shutil.which,
) -> AirfoilV7EnvironmentObservation:
    """Observe every local prerequisite without starting CFD or a provider."""

    if type(spec) is not AirfoilV7ReadinessSpec:
        raise TypeError("spec must be an exact AirfoilV7ReadinessSpec")
    if not all(
        callable(value)
        for value in (
            command_runner,
            affinity_reader,
            process_scanner,
            executable_resolver,
        )
    ):
        raise TypeError("environment observation dependencies must be callable")
    python = _file_observation(spec.evaluator_python, require_executable=True)
    evaluator = _file_observation(spec.evaluator_script, require_executable=False)
    dataset = _file_observation(spec.dataset_arrow, require_executable=False)
    dataset["expected_sha256"] = spec.expected_dataset_sha256
    dataset["hash_matches"] = dataset["sha256"] == spec.expected_dataset_sha256
    dataset["passed"] = bool(dataset["passed"] and dataset["hash_matches"])

    python_version = command_runner(
        (str(_invoked_path(spec.evaluator_python)), "--version"),
        float(spec.command_timeout_seconds),
    )
    evaluator_environment = command_runner(
        (
            str(_invoked_path(spec.evaluator_python)),
            "-c",
            _EVALUATOR_ENVIRONMENT_PROBE,
        ),
        float(spec.command_timeout_seconds),
    )
    try:
        evaluator_environment_record = json.loads(evaluator_environment.stdout)
    except json.JSONDecodeError:
        evaluator_environment_record = None
    environment_fingerprint_pass = (
        evaluator_environment.succeeded
        and type(evaluator_environment_record) is dict
        and set(evaluator_environment_record) == {
            "executable",
            "modules",
            "python_version",
        }
        and evaluator_environment_record["executable"]
        == str(_invoked_path(spec.evaluator_python))
        and type(evaluator_environment_record["modules"]) is dict
        and set(evaluator_environment_record["modules"]) == {"engibench", "numpy"}
        and all(
            type(row) is dict
            and set(row) == {"path", "sha256", "version"}
            and type(row["path"]) is str
            and bool(row["path"])
            and type(row["version"]) is str
            and bool(row["version"])
            and type(row["sha256"]) is str
            and _SHA256.fullmatch(row["sha256"]) is not None
            for row in evaluator_environment_record["modules"].values()
        )
    )
    docker_version = command_runner(
        (
            spec.docker_executable,
            "version",
            "--format",
            "{{.Server.Version}}",
        ),
        float(spec.command_timeout_seconds),
    )
    image = command_runner(
        (
            spec.docker_executable,
            "image",
            "inspect",
            "--format",
            "{{.Id}}\t{{json .RepoDigests}}",
            spec.container_image,
        ),
        float(spec.command_timeout_seconds),
    )
    image_matches = image.succeeded and spec.container_image in image.stdout
    conflict = AirfoilV7ConflictProbe(
        spec,
        command_runner=command_runner,
        process_scanner=process_scanner,
    )()
    requested_cpus = _parse_cpu_set(spec.cpu_set)
    affinity = tuple(sorted(affinity_reader()))
    if any(type(value) is not int or value < 0 for value in affinity):
        raise TypeError("affinity_reader must return non-negative CPU integers")
    cpu_available = set(requested_cpus).issubset(affinity)
    docker_path = executable_resolver(spec.docker_executable)
    if docker_path is not None and type(docker_path) is not str:
        raise TypeError("executable_resolver must return a string or None")

    checks = {
        "evaluator_python": bool(
            python["passed"]
            and python_version.succeeded
            and environment_fingerprint_pass
        ),
        "evaluator_script": bool(evaluator["passed"]),
        "dataset": bool(dataset["passed"]),
        "docker_server": docker_version.succeeded and docker_path is not None,
        "pinned_container_image": image_matches,
        "cpu_allocation": cpu_available,
        "external_conflicts_absent": not conflict.conflict,
    }
    passed = all(checks.values())
    record = freeze_json(
        {
            "probe_id": AIRFOIL_V7_ENVIRONMENT_PROBE_ID,
            "probe_version": AIRFOIL_V7_ENVIRONMENT_PROBE_VERSION,
            "checks": checks,
            "evaluator_python": python,
            "evaluator_python_version": python_version.to_record(),
            "evaluator_environment_command": evaluator_environment.to_record(),
            "evaluator_environment_fingerprint": evaluator_environment_record,
            "evaluator_script": evaluator,
            "dataset": dataset,
            "docker_executable_resolved": docker_path,
            "docker_server_version": docker_version.to_record(),
            "container_image": spec.container_image,
            "container_image_inspect": image.to_record(),
            "container_image_matches": image_matches,
            "cpu_set": spec.cpu_set,
            "requested_cpus": list(requested_cpus),
            "observed_affinity": list(affinity),
            "mpi_cores": spec.mpi_cores,
            "conflict_observation": conflict.to_record(),
        }
    )
    return AirfoilV7EnvironmentObservation(passed=passed, record=record)


def create_airfoil_v7_resource_lease(
    spec: AirfoilV7ReadinessSpec,
    *,
    lease_path: Path,
    run_id: str,
    phase: str,
    command_runner: CommandRunner = run_observation_command,
    process_scanner: ProcessScanner = _scan_process_markers,
) -> FileExclusiveResourceLease:
    """Compose the generic lease with Airfoil's fixed external-resource probe."""

    if type(phase) is not str or not phase or phase != phase.strip():
        raise ValueError("phase must be non-empty canonical text")
    return FileExclusiveResourceLease(
        resource_key=AIRFOIL_V7_RESOURCE_KEY,
        owner_id=run_id,
        lease_path=lease_path,
        owner_metadata={
            "phase": phase,
            "fixed_container_name": spec.fixed_container_name,
            "container_image": spec.container_image,
            "cpu_set": spec.cpu_set,
            "mpi_cores": spec.mpi_cores,
            "dataset_sha256": spec.expected_dataset_sha256,
        },
        conflict_probe=AirfoilV7ConflictProbe(
            spec,
            command_runner=command_runner,
            process_scanner=process_scanner,
        ),
    )


__all__ = [
    "AIRFOIL_V7_CONFLICT_PROBE_ID",
    "AIRFOIL_V7_CONFLICT_PROBE_VERSION",
    "AIRFOIL_V7_ENVIRONMENT_PROBE_ID",
    "AIRFOIL_V7_ENVIRONMENT_PROBE_VERSION",
    "AIRFOIL_V7_RESOURCE_KEY",
    "AirfoilV7ConflictProbe",
    "AirfoilV7EnvironmentObservation",
    "AirfoilV7ReadinessSpec",
    "CommandObservation",
    "create_airfoil_v7_resource_lease",
    "observe_airfoil_v7_environment",
    "run_observation_command",
]
