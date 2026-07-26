"""Workload-neutral contract for bounded child-process execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Protocol, runtime_checkable


EXPLICIT_ENVIRONMENT_BOUNDARY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:explicit-environment-subprocess-boundary:v1;"
    b"exact-argv;pinned-cwd;finite-timeout;capture-text;no-shell;"
    b"ambient-environment-deny-by-default;sensitive-name-deny"
).hexdigest()
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SENSITIVE_NAME = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD)$", re.IGNORECASE)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ChildProcessPolicy:
    """Stable deny-by-default environment and execution policy."""

    policy_id: str
    policy_version: int
    inherited_environment_allowlist: tuple[str, ...]
    fixed_environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.policy_id) is not str or not self.policy_id:
            raise ValueError("policy_id must be a non-empty string")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive integer")
        inherited = self.inherited_environment_allowlist
        if (
            type(inherited) is not tuple
            or inherited != tuple(sorted(set(inherited)))
            or any(type(name) is not str for name in inherited)
        ):
            raise ValueError(
                "inherited_environment_allowlist must be a sorted unique tuple"
            )
        fixed = self.fixed_environment
        if type(fixed) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            for item in fixed
        ):
            raise TypeError("fixed_environment must contain exact string pairs")
        fixed_names = tuple(name for name, _ in fixed)
        if fixed_names != tuple(sorted(set(fixed_names))):
            raise ValueError("fixed_environment names must be sorted and unique")
        all_names = inherited + fixed_names
        if any(_ENVIRONMENT_NAME.fullmatch(name) is None for name in all_names):
            raise ValueError("environment names must be canonical")
        if any(_SENSITIVE_NAME.search(name) is not None for name in all_names):
            raise ValueError("sensitive environment names cannot be allowlisted")
        if set(inherited).intersection(fixed_names):
            raise ValueError("fixed and inherited environment names must be disjoint")
        if any("\x00" in value for _, value in fixed):
            raise ValueError("fixed environment values cannot contain NUL")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "implementation_definition_sha256": (
                EXPLICIT_ENVIRONMENT_BOUNDARY_DEFINITION_SHA256
            ),
            "argv_policy": "preserve_exact_sequence_no_shell_v1",
            "working_directory_policy": "required_pinned_absolute_directory_v1",
            "timeout_policy": "required_finite_positive_seconds_v1",
            "environment_policy": "ambient_deny_explicit_allowlist_v1",
            "sensitive_name_policy": (
                "reject_names_ending_key_token_secret_password_v1"
            ),
            "inherited_environment_allowlist": list(
                self.inherited_environment_allowlist
            ),
            "fixed_environment": [
                {"name": name, "value": value}
                for name, value in self.fixed_environment
            ],
        }

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.to_record())


@dataclass(frozen=True, slots=True)
class ChildProcessResult:
    returncode: int
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        if type(self.returncode) is not int:
            raise TypeError("returncode must be an integer")
        if type(self.stdout) is not str or type(self.stderr) is not str:
            raise TypeError("captured output must be text")


@runtime_checkable
class BoundedSubprocessBoundary(Protocol):
    """Execute exact argv under one stable, explicit process policy."""

    @property
    def identity_sha256(self) -> str: ...

    def stable_record(self) -> dict[str, object]: ...

    def invocation_observation(self, executable: str) -> dict[str, object]: ...

    def run(self, argv: tuple[str, ...], *, timeout_s: float) -> ChildProcessResult: ...


__all__ = [
    "BoundedSubprocessBoundary",
    "ChildProcessPolicy",
    "ChildProcessResult",
    "EXPLICIT_ENVIRONMENT_BOUNDARY_DEFINITION_SHA256",
]
