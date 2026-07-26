"""Validated workload-owned launch contracts for systematic campaigns.

The central grid launcher should not know a benchmark's CLI verbs, artifact
layout, preregistration filename, seed flags, or Python dependency overlay.
Those integration details live in one declarative contract owned by the
workload adapter.  This module supplies the closed, shell-free interpreter for
that contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Any


_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_PIN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}=="
    r"[A-Za-z0-9][A-Za-z0-9_.+!-]{0,95}$"
)
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
_ALLOWED_PLACEHOLDERS = frozenset(
    {"prepare_run_id", "live_run_id", "replicate_seed", "preregistration_path"}
)


def _closed_relative_path(value: object, name: str) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{name} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{name} must be a closed relative path")
    return path


def _arguments(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise ValueError(f"{name} must be a non-empty argument list")
    arguments: list[str] = []
    for argument in value:
        if type(argument) is not str or not argument or "\x00" in argument:
            raise ValueError(f"{name} contains an invalid argument")
        placeholders = set(_PLACEHOLDER.findall(argument))
        if placeholders - _ALLOWED_PLACEHOLDERS:
            raise ValueError(f"{name} uses an unsupported placeholder")
        residual = _PLACEHOLDER.sub("", argument)
        if "{" in residual or "}" in residual:
            raise ValueError(f"{name} contains malformed template syntax")
        arguments.append(argument)
    return tuple(arguments)


@dataclass(frozen=True, slots=True)
class ArmExecutionContract:
    """One treatment or control adapter's shell-free launch recipe."""

    runner: Path
    artifact_root: Path
    prepare_arguments: tuple[str, ...]
    live_arguments: tuple[str, ...]
    preregistration_relative_path: PurePosixPath | None


@dataclass(frozen=True, slots=True)
class ExclusiveResourceExecutionContract:
    """One process-external resource required only by the live phase."""

    resource_key: str
    lease_path: Path


@dataclass(frozen=True, slots=True)
class WorkloadExecutionContract:
    """A validated workload integration boundary for the grid launcher."""

    path: Path
    workspace_root: Path
    record: dict[str, Any]
    workload_id: str
    runtime_prefix: tuple[str, ...]
    arms: tuple[tuple[str, ArmExecutionContract], ...]
    exclusive_resource: ExclusiveResourceExecutionContract | None

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        workspace_root: Path,
        expected_workload_id: str,
        expected_sha256: str,
    ) -> "WorkloadExecutionContract":
        resolved_workspace = workspace_root.resolve(strict=True)
        resolved = path.expanduser().resolve(strict=True)
        if resolved_workspace not in resolved.parents:
            raise ValueError("execution contract must live inside the workspace")
        raw = resolved.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("execution contract hash drift")
        value = json.loads(raw.decode("utf-8", errors="strict"))
        if type(value) is not dict or value.get("schema_version") not in (1, 2):
            raise ValueError("unsupported execution contract schema")
        schema_version = value["schema_version"]
        expected_fields = {
            "schema_version",
            "workload_id",
            "runtime",
            "arms",
            *(("exclusive_resource",) if schema_version == 2 else ()),
        }
        if set(value) != expected_fields:
            raise ValueError("execution contract has an invalid top-level field set")
        workload_id = value.get("workload_id")
        if (
            type(workload_id) is not str
            or _TOKEN.fullmatch(workload_id) is None
            or workload_id != expected_workload_id
        ):
            raise ValueError("execution contract workload identity mismatch")
        runtime_prefix = cls._runtime_prefix(value.get("runtime"))
        exclusive_resource = (
            None
            if schema_version == 1
            else cls._exclusive_resource(
                value.get("exclusive_resource"),
                workspace_root=resolved_workspace,
            )
        )
        arm_records = value.get("arms")
        if type(arm_records) is not dict or set(arm_records) != {"control", "treatment"}:
            raise ValueError("execution contract requires treatment and control arms")
        arms = tuple(
            (
                arm,
                cls._arm(
                    arm_records[arm],
                    name=arm,
                    workspace_root=resolved_workspace,
                ),
            )
            for arm in ("control", "treatment")
        )
        contract = cls(
            path=resolved,
            workspace_root=resolved_workspace,
            record=value,
            workload_id=workload_id,
            runtime_prefix=runtime_prefix,
            arms=arms,
            exclusive_resource=exclusive_resource,
        )
        contract._validate_required_placeholders()
        return contract

    @staticmethod
    def _runtime_prefix(value: object) -> tuple[str, ...]:
        if type(value) is not dict:
            raise ValueError("runtime must be an object")
        kind = value.get("kind")
        if kind == "current_python":
            if set(value) != {"kind"}:
                raise ValueError("current_python runtime has unexpected fields")
            return (sys.executable,)
        if kind != "uv_python" or set(value) != {"kind", "requirements"}:
            raise ValueError("runtime kind is unsupported")
        requirements = value.get("requirements")
        if (
            type(requirements) is not list
            or not requirements
            or any(type(item) is not str or _PIN.fullmatch(item) is None for item in requirements)
            or requirements != sorted(set(requirements))
        ):
            raise ValueError("uv runtime requirements must be unique exact pins")
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv runtime requested but uv is unavailable")
        prefix: list[str] = [uv, "run"]
        for requirement in requirements:
            prefix.extend(("--with", requirement))
        prefix.append("python")
        return tuple(prefix)

    @staticmethod
    def _exclusive_resource(
        value: object,
        *,
        workspace_root: Path,
    ) -> ExclusiveResourceExecutionContract | None:
        if value is None:
            return None
        if type(value) is not dict or set(value) != {
            "lease_relative_path",
            "resource_key",
        }:
            raise ValueError("exclusive_resource has an invalid field set")
        resource_key = value.get("resource_key")
        if type(resource_key) is not str or _TOKEN.fullmatch(resource_key) is None:
            raise ValueError("exclusive_resource.resource_key is invalid")
        lease_relative = _closed_relative_path(
            value.get("lease_relative_path"),
            "exclusive_resource.lease_relative_path",
        )
        lease_path = (workspace_root / lease_relative).resolve()
        lease_root = (
            workspace_root
            / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs/"
            "systematic_resource_leases"
        ).resolve()
        if lease_path.parent != lease_root or lease_path.suffix != ".lock":
            raise ValueError(
                "exclusive_resource.lease_relative_path must name one .lock file "
                "inside the systematic resource-lease root"
            )
        return ExclusiveResourceExecutionContract(
            resource_key=resource_key,
            lease_path=lease_path,
        )

    @staticmethod
    def _arm(
        value: object,
        *,
        name: str,
        workspace_root: Path,
    ) -> ArmExecutionContract:
        if type(value) is not dict or set(value) != {
            "artifact_root",
            "live_arguments",
            "prepare_arguments",
            "preregistration_relative_path",
            "runner",
        }:
            raise ValueError(f"{name} arm has an invalid field set")
        runner_relative = _closed_relative_path(value.get("runner"), f"{name}.runner")
        runner = (workspace_root / runner_relative).resolve()
        if workspace_root not in runner.parents or not runner.is_file():
            raise ValueError(f"{name} runner is outside the workspace or missing")
        artifact_relative = _closed_relative_path(
            value.get("artifact_root"), f"{name}.artifact_root"
        )
        artifact_root = (workspace_root / artifact_relative).resolve()
        expected_artifact_parent = (
            workspace_root
            / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
        ).resolve()
        if expected_artifact_parent not in artifact_root.parents:
            raise ValueError(f"{name} artifact root escapes the experiment-log root")
        preregistration_value = value.get("preregistration_relative_path")
        preregistration = (
            None
            if preregistration_value is None
            else _closed_relative_path(
                preregistration_value,
                f"{name}.preregistration_relative_path",
            )
        )
        return ArmExecutionContract(
            runner=runner,
            artifact_root=artifact_root,
            prepare_arguments=_arguments(
                value.get("prepare_arguments"), f"{name}.prepare_arguments"
            ),
            live_arguments=_arguments(
                value.get("live_arguments"), f"{name}.live_arguments"
            ),
            preregistration_relative_path=preregistration,
        )

    def _validate_required_placeholders(self) -> None:
        for arm_name, arm in self.arms:
            prepare = "\x00".join(arm.prepare_arguments)
            live = "\x00".join(arm.live_arguments)
            if prepare.count("{prepare_run_id}") != 1:
                raise ValueError(f"{arm_name} prepare command must bind prepare_run_id once")
            if live.count("{live_run_id}") != 1:
                raise ValueError(f"{arm_name} live command must bind live_run_id once")
            expected_preregistration_count = (
                0 if arm.preregistration_relative_path is None else 1
            )
            if live.count("{preregistration_path}") != expected_preregistration_count:
                raise ValueError(
                    f"{arm_name} live command has an invalid preregistration binding"
                )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def arm(self, name: str) -> ArmExecutionContract:
        matches = tuple(value for arm, value in self.arms if arm == name)
        if len(matches) != 1:
            raise ValueError("unknown execution-contract arm")
        return matches[0]

    @staticmethod
    def _render(arguments: tuple[str, ...], bindings: dict[str, str]) -> list[str]:
        rendered: list[str] = []
        for argument in arguments:
            placeholders = set(_PLACEHOLDER.findall(argument))
            if not placeholders <= set(bindings):
                raise ValueError("command template has an unbound placeholder")
            rendered.append(_PLACEHOLDER.sub(lambda match: bindings[match.group(1)], argument))
        return rendered

    def commands(
        self,
        *,
        arm_name: str,
        prepare_run_id: str,
        live_run_id: str,
        replicate_seed: int,
    ) -> tuple[list[str], list[str], Path, Path, Path | None]:
        arm = self.arm(arm_name)
        prepare_dir = arm.artifact_root / prepare_run_id
        live_dir = arm.artifact_root / live_run_id
        preregistration = (
            None
            if arm.preregistration_relative_path is None
            else prepare_dir / arm.preregistration_relative_path
        )
        bindings = {
            "prepare_run_id": prepare_run_id,
            "live_run_id": live_run_id,
            "replicate_seed": str(replicate_seed),
            "preregistration_path": "" if preregistration is None else str(preregistration),
        }
        prepare = [
            *self.runtime_prefix,
            str(arm.runner),
            *self._render(arm.prepare_arguments, bindings),
        ]
        live = [
            *self.runtime_prefix,
            str(arm.runner),
            *self._render(arm.live_arguments, bindings),
        ]
        return prepare, live, prepare_dir, live_dir, preregistration

    def identity(self) -> dict[str, object]:
        return {
            "path": self.path.relative_to(self.workspace_root).as_posix(),
            "sha256": self.sha256,
            "workload_id": self.workload_id,
            "runtime_prefix": list(self.runtime_prefix),
            "exclusive_resource": (
                None
                if self.exclusive_resource is None
                else {
                    "resource_key": self.exclusive_resource.resource_key,
                    "lease_relative_path": self.exclusive_resource.lease_path.relative_to(
                        self.workspace_root
                    ).as_posix(),
                }
            ),
        }
