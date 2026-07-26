"""Content-addressed, provider-free live-runtime provenance manifests.

The application layer owns the immutable record and its validation rules.  It
does not know any benchmark, provider SDK, credential source, or evaluator.
Composition roots supply closed JSON sections and an explicit role-indexed
source closure.  Local probes are deliberately content-blind: git diffs and
status are hashed but never embedded, so a manifest cannot leak source or
secrets through provenance metadata.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    thaw_json,
)
from agent_evolve.ports.artifact_store import decode_json_bytes


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_UTC_SECONDS = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_FILE_DOMAIN = b"agent-evolve:live-runtime-file:v1\x00"
_SOURCE_DOMAIN = b"agent-evolve:live-runtime-source-closure:v1\x00"
_SECTION_DOMAIN = b"agent-evolve:live-runtime-section:v1\x00"
_MANIFEST_DOMAIN = b"agent-evolve:live-runtime-manifest:v1\x00"


class LiveRuntimeManifestError(RuntimeError):
    """A prospective runtime commitment is missing or has drifted."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_json_equal(left: object, right: object) -> bool:
    """JSON equality that does not conflate bool with integer values."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        assert type(right) is dict
        return left.keys() == right.keys() and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        assert type(right) is list
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _validate_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed lowercase token grammar")


def _validate_logical_path(value: str) -> None:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("logical_path must be non-empty canonical POSIX text")
    parsed = Path(value)
    if parsed.is_absolute() or value != parsed.as_posix() or ".." in parsed.parts:
        raise ValueError("logical_path must be relative canonical POSIX text")


@dataclass(frozen=True, slots=True)
class RuntimeFileBinding:
    """Exact bytes for one source, lock, route, or evaluator dependency."""

    logical_path: str
    resolved_path: str
    size_bytes: int
    sha256: str
    binding_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_logical_path(self.logical_path)
        resolved = Path(self.resolved_path)
        if (
            type(self.resolved_path) is not str
            or not resolved.is_absolute()
            or self.resolved_path != str(resolved)
        ):
            raise ValueError("resolved_path must be an absolute normalized path")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative exact integer")
        if type(self.sha256) is not str or _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(
            self,
            "binding_sha256",
            _hash(_FILE_DOMAIN, self._identity_record()),
        )

    def _identity_record(self) -> dict[str, object]:
        return {
            "logical_path": self.logical_path,
            "resolved_path": self.resolved_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._identity_record(), "binding_sha256": self.binding_sha256}


def capture_runtime_file(path: Path, *, logical_path: str) -> RuntimeFileBinding:
    """Hash one exact file without interpreting or publishing its contents."""

    _validate_logical_path(logical_path)
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise LiveRuntimeManifestError(f"runtime dependency is not a file: {resolved}")
    content = resolved.read_bytes()
    return RuntimeFileBinding(
        logical_path=logical_path,
        resolved_path=str(resolved),
        size_bytes=len(content),
        sha256=_sha256_bytes(content),
    )


@dataclass(frozen=True, slots=True)
class RuntimeSourceClosure:
    """Role-indexed conservative closure over every live execution byte."""

    files: tuple[RuntimeFileBinding, ...]
    role_paths: tuple[tuple[str, tuple[str, ...]], ...]
    source_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.files) is not tuple
            or not self.files
            or any(type(value) is not RuntimeFileBinding for value in self.files)
        ):
            raise TypeError("files must be a non-empty exact tuple of bindings")
        for value in self.files:
            value.__post_init__()
        if self.files != tuple(sorted(self.files, key=lambda value: value.logical_path)):
            raise ValueError("source files must be ordered by logical_path")
        logical_paths = tuple(value.logical_path for value in self.files)
        if len(set(logical_paths)) != len(logical_paths):
            raise ValueError("source closure cannot repeat a logical path")
        if type(self.role_paths) is not tuple or not self.role_paths:
            raise TypeError("role_paths must be a non-empty exact tuple")
        if self.role_paths != tuple(sorted(self.role_paths, key=lambda value: value[0])):
            raise ValueError("source roles must be canonically ordered")
        observed_roles: set[str] = set()
        covered: set[str] = set()
        for item in self.role_paths:
            if type(item) is not tuple or len(item) != 2:
                raise TypeError("each source role must be an exact pair")
            role, paths = item
            _validate_token(role, name="source role")
            if role in observed_roles:
                raise ValueError("source roles cannot repeat")
            observed_roles.add(role)
            if (
                type(paths) is not tuple
                or not paths
                or paths != tuple(sorted(set(paths)))
                or any(path not in logical_paths for path in paths)
            ):
                raise ValueError("source role paths must be non-empty and canonical")
            covered.update(paths)
        if covered != set(logical_paths):
            raise ValueError("every source file must belong to at least one role")
        object.__setattr__(
            self,
            "source_sha256",
            _hash(_SOURCE_DOMAIN, self._identity_record()),
        )

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "files": [value.to_record() for value in self.files],
            "roles": {
                role: list(paths) for role, paths in self.role_paths
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._identity_record(), "source_sha256": self.source_sha256}


def capture_runtime_source_closure(
    files_by_role: Mapping[str, Mapping[str, Path]],
) -> RuntimeSourceClosure:
    """Capture a deduplicated closure from benchmark-owned semantic roles."""

    if not isinstance(files_by_role, Mapping) or not files_by_role:
        raise TypeError("files_by_role must be a non-empty mapping")
    bindings: dict[str, RuntimeFileBinding] = {}
    roles: list[tuple[str, tuple[str, ...]]] = []
    for role in sorted(files_by_role):
        _validate_token(role, name="source role")
        sources = files_by_role[role]
        if not isinstance(sources, Mapping) or not sources:
            raise ValueError(f"source role {role} must contain at least one file")
        paths: list[str] = []
        for logical_path in sorted(sources):
            captured = capture_runtime_file(
                Path(sources[logical_path]),
                logical_path=logical_path,
            )
            prior = bindings.get(logical_path)
            if prior is not None and prior != captured:
                raise ValueError(f"logical source path {logical_path} is ambiguous")
            bindings[logical_path] = captured
            paths.append(logical_path)
        roles.append((role, tuple(paths)))
    return RuntimeSourceClosure(
        files=tuple(bindings[key] for key in sorted(bindings)),
        role_paths=tuple(roles),
    )


@dataclass(frozen=True, slots=True)
class RuntimeManifestSection:
    """One immutable benchmark- or infrastructure-owned manifest section."""

    section_id: str
    payload: FrozenJsonObject
    section_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_token(self.section_id, name="section_id")
        if type(self.payload) is not FrozenJsonObject:
            raise TypeError("section payload must be an exact FrozenJsonObject")
        self.payload.__post_init__()
        object.__setattr__(
            self,
            "section_sha256",
            _hash(
                _SECTION_DOMAIN,
                {
                    "section_id": self.section_id,
                    "payload": thaw_json(self.payload),
                },
            ),
        )

    @classmethod
    def seal(
        cls,
        section_id: str,
        payload: Mapping[str, object],
    ) -> "RuntimeManifestSection":
        if not isinstance(payload, Mapping):
            raise TypeError("section payload must be a mapping")
        frozen = freeze_json(dict(payload))
        if type(frozen) is not FrozenJsonObject:
            raise TypeError("section payload did not freeze as an object")
        return cls(section_id=section_id, payload=frozen)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "section_id": self.section_id,
            "payload": thaw_json(self.payload),
            "section_sha256": self.section_sha256,
        }


def _run_git(root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LiveRuntimeManifestError(
            f"git provenance probe failed: {' '.join(args)}"
        ) from exc
    return completed.stdout


def capture_git_worktree_section(
    repo_root: Path,
    *,
    source_closure: RuntimeSourceClosure,
) -> RuntimeManifestSection:
    """Bind HEAD and dirty state while retaining only content-blind digests."""

    source_closure.__post_init__()
    root = repo_root.expanduser().resolve(strict=True)
    observed_root = Path(
        _run_git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve(strict=True)
    if observed_root != root:
        raise LiveRuntimeManifestError("repo_root is not the git worktree root")
    head = _run_git(root, "rev-parse", "HEAD").decode("ascii").strip()
    head_tree = _run_git(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    if _SHA256.fullmatch(head) is None and re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise LiveRuntimeManifestError("git HEAD is not a supported object identity")
    if _SHA256.fullmatch(head_tree) is None and re.fullmatch(r"[0-9a-f]{40}", head_tree) is None:
        raise LiveRuntimeManifestError("git HEAD tree is not a supported identity")
    try:
        branch_probe = subprocess.run(
            ("git", "symbolic-ref", "--quiet", "--short", "HEAD"),
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise LiveRuntimeManifestError("git branch provenance probe failed") from exc
    if branch_probe.returncode not in (0, 1):
        raise LiveRuntimeManifestError("git branch provenance probe failed")
    branch = (
        branch_probe.stdout.decode("utf-8").strip()
        if branch_probe.returncode == 0
        else None
    )
    # Restrict every dirty-state probe to the bound closure.  A live run writes
    # journals and evaluator receipts after the prospective manifest is built;
    # hashing full-worktree state would therefore make the gate invalidate
    # itself even though no executable input changed.
    repo_paths = tuple(
        sorted(
            {
                Path(binding.resolved_path).relative_to(root).as_posix()
                for binding in source_closure.files
                if Path(binding.resolved_path).is_relative_to(root)
            }
        )
    )
    if repo_paths:
        pathspec = ("--", *repo_paths)
        status = _run_git(
            root,
            "status",
            "--porcelain=v2",
            "-z",
            "--untracked-files=all",
            *pathspec,
        )
        staged = _run_git(
            root,
            "diff",
            "--cached",
            "--binary",
            "--no-ext-diff",
            *pathspec,
        )
        unstaged = _run_git(
            root,
            "diff",
            "--binary",
            "--no-ext-diff",
            *pathspec,
        )
        tracked = set(
            item.decode("utf-8")
            for item in _run_git(
                root,
                "ls-files",
                "-z",
                *pathspec,
            ).split(b"\x00")
            if item
        )
        untracked = set(
            item.decode("utf-8")
            for item in _run_git(
                root,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                *pathspec,
            ).split(b"\x00")
            if item
        )
    else:
        status = staged = unstaged = b""
        tracked: set[str] = set()
        untracked: set[str] = set()
    relevant_states: list[dict[str, str]] = []
    for binding in source_closure.files:
        path = Path(binding.resolved_path)
        if not path.is_relative_to(root):
            state = "external"
        else:
            relative = path.relative_to(root).as_posix()
            if relative in tracked:
                state = "tracked"
            elif relative in untracked:
                state = "untracked"
            else:
                state = "ignored_or_unindexed"
        relevant_states.append(
            {"logical_path": binding.logical_path, "git_state": state}
        )
    return RuntimeManifestSection.seal(
        "git_worktree",
        {
            "schema_version": 1,
            "repo_root": str(root),
            "head_commit": head,
            "head_tree": head_tree,
            "branch": branch,
            "head_mode": "attached" if branch is not None else "detached",
            "dirty": bool(status),
            "dirty_scope": "runtime_source_closure_repo_files_only",
            "repo_relative_source_file_count": len(repo_paths),
            "porcelain_v2_bytes": len(status),
            "porcelain_v2_sha256": _sha256_bytes(status),
            "staged_binary_diff_bytes": len(staged),
            "staged_binary_diff_sha256": _sha256_bytes(staged),
            "unstaged_binary_diff_bytes": len(unstaged),
            "unstaged_binary_diff_sha256": _sha256_bytes(unstaged),
            "relevant_source_manifest_sha256": source_closure.source_sha256,
            "relevant_source_git_states": relevant_states,
            "content_disclosure": "hashes_only_no_patch_or_source_text",
        },
    )


def capture_runtime_environment_section(
    *,
    distribution_names: Sequence[str],
    dependency_locks: Sequence[RuntimeFileBinding],
) -> RuntimeManifestSection:
    """Bind the interpreter, host ABI, installed transports, and lock bytes."""

    if (
        not isinstance(distribution_names, Sequence)
        or isinstance(distribution_names, (str, bytes))
        or not distribution_names
    ):
        raise TypeError("distribution_names must be a non-empty sequence")
    names = tuple(distribution_names)
    if (
        any(type(value) is not str or not value for value in names)
        or names != tuple(sorted(set(names)))
    ):
        raise ValueError("distribution_names must be unique and ordered")
    locks = tuple(dependency_locks)
    if not locks or any(type(value) is not RuntimeFileBinding for value in locks):
        raise TypeError("dependency_locks must contain exact file bindings")
    for value in locks:
        value.__post_init__()
    packages: dict[str, str] = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise LiveRuntimeManifestError(
                f"required runtime distribution is unavailable: {name}"
            ) from exc
    executable = Path(os.path.abspath(sys.executable))
    if not executable.is_file():
        raise LiveRuntimeManifestError("invocation Python executable is unavailable")
    return RuntimeManifestSection.seal(
        "runtime_environment",
        {
            "schema_version": 1,
            "python_executable": str(executable),
            "python_executable_resolved": str(executable.resolve(strict=True)),
            "python_prefix": os.path.abspath(sys.prefix),
            "python_base_prefix": os.path.abspath(sys.base_prefix),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "installed_distributions": packages,
            "dependency_locks": [value.to_record() for value in locks],
        },
    )


@dataclass(frozen=True, slots=True)
class LiveRuntimeManifest:
    """Prospective full-stack commitment produced before credentials or work."""

    manifest_id: str
    manifest_version: int
    built_at_utc: str
    source_closure: RuntimeSourceClosure
    sections: tuple[RuntimeManifestSection, ...]
    required_section_ids: tuple[str, ...]
    manifest_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_token(self.manifest_id, name="manifest_id")
        if type(self.manifest_version) is not int or self.manifest_version <= 0:
            raise ValueError("manifest_version must be a positive exact integer")
        if type(self.built_at_utc) is not str or _UTC_SECONDS.fullmatch(
            self.built_at_utc
        ) is None:
            raise ValueError("built_at_utc must be UTC RFC3339 at whole seconds")
        try:
            datetime.strptime(self.built_at_utc, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise ValueError("built_at_utc is not a real UTC instant") from exc
        self.source_closure.__post_init__()
        if (
            type(self.sections) is not tuple
            or not self.sections
            or any(type(value) is not RuntimeManifestSection for value in self.sections)
        ):
            raise TypeError("sections must be a non-empty exact tuple")
        for value in self.sections:
            value.__post_init__()
        if self.sections != tuple(sorted(self.sections, key=lambda value: value.section_id)):
            raise ValueError("manifest sections must be canonically ordered")
        ids = tuple(value.section_id for value in self.sections)
        if len(set(ids)) != len(ids):
            raise ValueError("manifest section IDs cannot repeat")
        if (
            type(self.required_section_ids) is not tuple
            or not self.required_section_ids
            or self.required_section_ids != tuple(sorted(set(self.required_section_ids)))
        ):
            raise ValueError("required_section_ids must be unique and ordered")
        for value in self.required_section_ids:
            _validate_token(value, name="required section ID")
        if not set(self.required_section_ids).issubset(ids):
            raise ValueError("one or more required manifest sections are missing")
        object.__setattr__(
            self,
            "manifest_sha256",
            _hash(_MANIFEST_DOMAIN, self._identity_record()),
        )

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "manifest_id": self.manifest_id,
            "manifest_version": self.manifest_version,
            "built_at_utc": self.built_at_utc,
            "claim_boundary": {
                "credentials_read": False,
                "provider_called": False,
                "physical_evaluator_called": False,
                "current_run_outcomes_observed": False,
                "meaning": "prospective provider-free runtime commitment only",
            },
            "source_closure": self.source_closure.to_record(),
            "required_section_ids": list(self.required_section_ids),
            "sections": [value.to_record() for value in self.sections],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._identity_record(), "manifest_sha256": self.manifest_sha256}


def build_live_runtime_manifest(
    *,
    manifest_id: str,
    manifest_version: int,
    built_at_utc: str,
    source_closure: RuntimeSourceClosure,
    sections: Sequence[RuntimeManifestSection],
    required_section_ids: Sequence[str],
) -> LiveRuntimeManifest:
    """Canonicalize injected sections into one self-authenticating manifest."""

    return LiveRuntimeManifest(
        manifest_id=manifest_id,
        manifest_version=manifest_version,
        built_at_utc=built_at_utc,
        source_closure=source_closure,
        sections=tuple(sorted(sections, key=lambda value: value.section_id)),
        required_section_ids=tuple(sorted(required_section_ids)),
    )


def verify_runtime_source_closure(source: RuntimeSourceClosure) -> None:
    """Fail closed if any currently installed source byte differs."""

    source.__post_init__()
    observed = tuple(
        capture_runtime_file(
            Path(value.resolved_path),
            logical_path=value.logical_path,
        )
        for value in source.files
    )
    if observed != source.files:
        raise LiveRuntimeManifestError("live runtime source closure drifted")


def runtime_file_binding_from_record(value: Mapping[str, object]) -> RuntimeFileBinding:
    """Reconstruct and authenticate one serialized file binding."""

    try:
        binding = RuntimeFileBinding(
            logical_path=str(value["logical_path"]),
            resolved_path=str(value["resolved_path"]),
            size_bytes=int(value["size_bytes"]),
            sha256=str(value["sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveRuntimeManifestError("runtime file binding is malformed") from exc
    if not _exact_json_equal(dict(value), binding.to_record()):
        raise LiveRuntimeManifestError("runtime file binding identity changed")
    return binding


def runtime_source_closure_from_record(
    value: Mapping[str, object],
) -> RuntimeSourceClosure:
    """Reconstruct and authenticate a serialized role-indexed source closure."""

    try:
        raw_files = value["files"]
        raw_roles = value["roles"]
        if type(raw_files) is not list or type(raw_roles) is not dict:
            raise TypeError("source closure files/roles have wrong types")
        files = tuple(
            runtime_file_binding_from_record(item)
            for item in raw_files
            if type(item) is dict
        )
        if len(files) != len(raw_files):
            raise TypeError("one source file binding is not an object")
        role_paths = tuple(
            (role, tuple(paths))
            for role, paths in sorted(raw_roles.items())
            if type(role) is str and type(paths) is list
        )
        if len(role_paths) != len(raw_roles):
            raise TypeError("one source role is malformed")
        closure = RuntimeSourceClosure(files=files, role_paths=role_paths)
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveRuntimeManifestError("runtime source closure is malformed") from exc
    if not _exact_json_equal(dict(value), closure.to_record()):
        raise LiveRuntimeManifestError("runtime source closure identity changed")
    return closure


def runtime_manifest_section_from_record(
    value: Mapping[str, object],
) -> RuntimeManifestSection:
    """Reconstruct and authenticate one serialized manifest section."""

    try:
        payload = value["payload"]
        if type(payload) is not dict:
            raise TypeError("section payload must be an object")
        section = RuntimeManifestSection.seal(str(value["section_id"]), payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveRuntimeManifestError("runtime manifest section is malformed") from exc
    if not _exact_json_equal(dict(value), section.to_record()):
        raise LiveRuntimeManifestError("runtime manifest section identity changed")
    return section


def live_runtime_manifest_from_record(
    value: Mapping[str, object],
) -> LiveRuntimeManifest:
    """Reconstruct a complete manifest and reject any forged self-hash."""

    try:
        raw_source = value["source_closure"]
        raw_sections = value["sections"]
        raw_required = value["required_section_ids"]
        if (
            type(raw_source) is not dict
            or type(raw_sections) is not list
            or type(raw_required) is not list
        ):
            raise TypeError("manifest containers have wrong types")
        source = runtime_source_closure_from_record(raw_source)
        sections = tuple(
            runtime_manifest_section_from_record(item)
            for item in raw_sections
            if type(item) is dict
        )
        if len(sections) != len(raw_sections):
            raise TypeError("one manifest section is not an object")
        manifest = LiveRuntimeManifest(
            manifest_id=str(value["manifest_id"]),
            manifest_version=int(value["manifest_version"]),
            built_at_utc=str(value["built_at_utc"]),
            source_closure=source,
            sections=sections,
            required_section_ids=tuple(str(item) for item in raw_required),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveRuntimeManifestError("live runtime manifest is malformed") from exc
    if not _exact_json_equal(dict(value), manifest.to_record()):
        raise LiveRuntimeManifestError("live runtime manifest identity changed")
    return manifest


def load_live_runtime_manifest(path: Path) -> LiveRuntimeManifest:
    """Load strict JSON and authenticate every nested manifest commitment."""

    resolved = path.expanduser().resolve(strict=True)
    try:
        value = decode_json_bytes(resolved.read_bytes())
    except Exception as exc:
        raise LiveRuntimeManifestError("live runtime manifest is unreadable") from exc
    if type(value) is not dict:
        raise LiveRuntimeManifestError("live runtime manifest root is not an object")
    return live_runtime_manifest_from_record(value)


__all__ = [
    "LiveRuntimeManifest",
    "LiveRuntimeManifestError",
    "RuntimeFileBinding",
    "RuntimeManifestSection",
    "RuntimeSourceClosure",
    "build_live_runtime_manifest",
    "capture_git_worktree_section",
    "capture_runtime_environment_section",
    "capture_runtime_file",
    "capture_runtime_source_closure",
    "live_runtime_manifest_from_record",
    "load_live_runtime_manifest",
    "runtime_file_binding_from_record",
    "runtime_manifest_section_from_record",
    "runtime_source_closure_from_record",
    "verify_runtime_source_closure",
]
