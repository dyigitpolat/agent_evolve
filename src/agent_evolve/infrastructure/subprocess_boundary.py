"""Deny-by-default subprocess implementation for the generic process port."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess

from agent_evolve.ports.subprocess_boundary import (
    ChildProcessPolicy,
    ChildProcessResult,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


class ExplicitEnvironmentSubprocessBoundary:
    """Run bounded commands with exact argv, cwd, and allowlisted environment."""

    def __init__(
        self,
        *,
        policy: ChildProcessPolicy,
        working_directory: Path,
        source_environment: Mapping[str, str] | None = None,
    ) -> None:
        if type(policy) is not ChildProcessPolicy:
            raise TypeError("policy must be exact ChildProcessPolicy")
        if not isinstance(working_directory, Path) or not working_directory.is_absolute():
            raise ValueError("working_directory must be an absolute pathlib.Path")
        resolved_working_directory = working_directory.resolve(strict=True)
        if not resolved_working_directory.is_dir():
            raise ValueError("working_directory must resolve to a directory")
        source = os.environ if source_environment is None else source_environment
        environment: dict[str, str] = {}
        for name in policy.inherited_environment_allowlist:
            value = source.get(name)
            if value is not None:
                if type(value) is not str or "\x00" in value:
                    raise ValueError("allowlisted environment values must be strings")
                environment[name] = value
        environment.update(dict(policy.fixed_environment))
        self._policy = policy
        self._working_directory = resolved_working_directory
        self._environment = environment
        self._record = {
            "schema_version": 1,
            "policy": {
                **policy.to_record(),
                "identity_sha256": policy.identity_sha256,
            },
            "working_directory": str(resolved_working_directory),
            "effective_environment_names": sorted(environment),
            "effective_environment_sha256": _canonical_sha256(environment),
        }
        self._identity_sha256 = _canonical_sha256(self._record)

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    def stable_record(self) -> dict[str, object]:
        return {
            **self._record,
            "identity_sha256": self._identity_sha256,
        }

    def invocation_observation(self, executable: str) -> dict[str, object]:
        if type(executable) is not str or not executable:
            raise ValueError("executable must be a non-empty string")
        invoked = Path(executable)
        if not invoked.is_absolute():
            raise ValueError("executable invocation path must be absolute")
        metadata = invoked.lstat()
        resolved = invoked.resolve(strict=True)
        return {
            "schema_version": 1,
            "invoked_path": str(invoked),
            "resolved_target": str(resolved),
            "is_symlink": invoked.is_symlink(),
            "lstat": {
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "mode_octal": oct(stat.S_IMODE(metadata.st_mode)),
                "size_bytes": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
            },
            "process_boundary_identity_sha256": self._identity_sha256,
        }

    def run(self, argv: tuple[str, ...], *, timeout_s: float) -> ChildProcessResult:
        if (
            type(argv) is not tuple
            or not argv
            or any(type(item) is not str or not item or "\x00" in item for item in argv)
        ):
            raise ValueError("argv must be a non-empty exact tuple of safe strings")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) <= 0.0
        ):
            raise ValueError("timeout_s must be finite and positive")
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            cwd=self._working_directory,
            env=dict(self._environment),
            shell=False,
        )
        return ChildProcessResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


__all__ = ["ExplicitEnvironmentSubprocessBoundary"]
