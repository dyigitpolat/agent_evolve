"""Persist the complete launch environment of a campaign preparation.

A sealed preparation directory already authenticates *what code ran* (the
source closure) and *what the campaign committed to* (the preregistration).
It has never authenticated *how the process was launched*.  Every runner in
``examples/development`` reads a substantial, silent configuration surface
from the ambient process: environment variables consulted at import time,
``PYTHONPATH``/``sys.path`` (which decides which ``agent_evolve`` package is
actually imported, independently of the hashed closure), the working
directory, the interpreter, ``.env`` files, and read-only inputs elsewhere on
the filesystem.  None of that was recorded, so a campaign whose code
reproduces byte-for-byte could still not be relaunched.

This module closes that hole generically.  It never inspects a workload and
never knows a variable name in advance; it observes.

Three independent layers are recorded, so that no single blind spot can lose
the configuration again:

``process_environment``
    Every variable in ``os.environ``, verbatim.  This is a superset of every
    environment input by construction: a value that was never in the process
    cannot have been read from it.  Credentials are recorded by presence and
    identity (sha256 + length) instead of value.

``declared_surface``
    A static scan of the campaign's own source closure for the environment
    names that code can read, including names bound to module-level string
    constants, together with each site's in-code default.  This is what makes
    the record *interpretable*: it says which variables are configuration.

``observed_reads``
    A recording proxy installed over ``os.environ`` before the runner's
    module body executes, which reports exactly which names were consulted,
    in order, including names assembled dynamically that no static scan can
    see.

Ambient filesystem inputs are captured the same way, with a ``sys`` audit
hook that records every file opened for reading outside the source closure
and the run directory.

Instrumentation only.  Nothing here participates in controller, allocator or
evaluator decisions.  Both observers are installed for ``prepare`` mode only
-- the provider-free, result-free phase -- so a paid ``live`` campaign runs
exactly the process it ran before.  Every observer is individually
fail-safe: if installation raises, the layer is marked unavailable and the
launch record is still written from the remaining layers.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import re
import sys
import threading
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from datetime import datetime, timezone


LAUNCH_RECORD_FILENAME = "launch_record.json"
LAUNCH_RECORD_SCHEMA_VERSION = 1
LAUNCH_RECORD_KIND = "agent-evolve:campaign-launch-record"

# Modes whose process is instrumented for its whole lifetime.  ``live`` is
# deliberately excluded: it is the phase that produces results, and it must
# remain byte-identical to the uninstrumented process.
INSTRUMENTED_MODES = ("prepare",)

# Modes that get a *reversible* startup window instead -- see
# ``instrument_startup_window``.  Credentials are read while the process is
# starting up, long before any timed work, so the window that proves a key was
# never read can be closed again before the clock starts.
CAMPAIGN_MODES = ("prepare", "live")

#: The phase during which a reversible recorder is installed.
STARTUP_PHASE = "startup"
#: The phase a reversible recorder must have been removed before.
MEASURED_PHASE = "measured"

# Bounds.  Every observer is capped so that instrumentation can never grow
# without limit inside a long campaign.
MAX_OBSERVED_ENVIRONMENT_NAMES = 4096
MAX_OBSERVED_READ_ORDER = 4096
MAX_AMBIENT_PATHS = 32_768
MAX_HASHED_AMBIENT_FILES = 2048
MAX_HASHED_AMBIENT_BYTES = 256 * 1024 * 1024

# A value is recorded by identity rather than verbatim when its *name* says
# it is a secret.  Redaction is by name only: a value is never inspected to
# decide whether to publish it, so the rule is auditable from this file
# alone.  Behaviour-affecting variables never match these patterns, and any
# redacted name that also appears in the declared surface is flagged.
CREDENTIAL_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|"
    r"AUTH|COOKIE|SIGNATURE|PRIVATE)(?:$|_)"
)
CREDENTIAL_NAME_ALLOWLIST = frozenset(
    {
        # Names that match the pattern but are pure configuration: their
        # values are addresses or layout selectors, not bearer material.
        "SSH_AUTH_SOCK",
        "GPG_AGENT_INFO",
        "KEYBOARD_LAYOUT",
        "DBUS_SESSION_BUS_ADDRESS",
    }
)

# Namespaces whose bare string literals are treated as candidate environment
# names even when the static scan cannot follow them into a read.  This is
# what catches names passed through a helper such as
# ``_bounded_integer_environment("AGENT_EVOLVE_...", 8192, ...)``.
CAMPAIGN_NAMESPACE_PREFIXES = ("AGENT_EVOLVE_", "AGENTEVOLVE_")

_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _relative_to(path: Path, root: Path | None) -> str | None:
    if root is None:
        return None
    try:
        return (
            path.resolve(strict=False)
            .relative_to(root.resolve(strict=False))
            .as_posix()
        )
    except (ValueError, OSError):
        return None


# --------------------------------------------------------------------------
# Layer 3: runtime observation of environment reads
# --------------------------------------------------------------------------


class EnvironmentReadLog:
    """Bounded, thread-safe record of which environment names were consulted."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._names: dict[str, dict[str, object]] = {}
        self._order: list[str] = []
        self._bulk: dict[str, int] = {}
        self._writes: dict[str, int] = {}
        self.truncated = False

    def record_read(self, name: object, *, present: bool, operation: str) -> None:
        if type(name) is not str:
            return
        with self._lock:
            entry = self._names.get(name)
            if entry is None:
                if len(self._names) >= MAX_OBSERVED_ENVIRONMENT_NAMES:
                    self.truncated = True
                    return
                entry = {"reads": 0, "present": present, "operations": []}
                self._names[name] = entry
                if len(self._order) < MAX_OBSERVED_READ_ORDER:
                    self._order.append(name)
                else:
                    self.truncated = True
            entry["reads"] = int(entry["reads"]) + 1
            entry["present"] = bool(entry["present"]) or present
            operations = entry["operations"]
            if type(operations) is list and operation not in operations:
                operations.append(operation)

    def record_write(self, name: object) -> None:
        if type(name) is not str:
            return
        with self._lock:
            if len(self._writes) < MAX_OBSERVED_ENVIRONMENT_NAMES:
                self._writes[name] = self._writes.get(name, 0) + 1
            else:
                self.truncated = True

    def record_bulk(self, operation: str) -> None:
        with self._lock:
            self._bulk[operation] = self._bulk.get(operation, 0) + 1

    def to_record(self) -> dict[str, object]:
        with self._lock:
            return {
                "schema_version": 1,
                "distinct_name_count": len(self._names),
                "first_read_order": list(self._order),
                "names": {
                    name: {
                        "reads": entry["reads"],
                        "present": entry["present"],
                        "operations": sorted(
                            entry["operations"]
                            if type(entry["operations"]) is list
                            else []
                        ),
                    }
                    for name, entry in sorted(self._names.items())
                },
                "mutated_names": dict(sorted(self._writes.items())),
                "bulk_enumerations": dict(sorted(self._bulk.items())),
                "truncated": self.truncated,
            }


class RecordingEnviron(MutableMapping):
    """A behaviour-preserving ``os.environ`` that reports what it is asked for.

    Every operation is delegated unchanged to the real ``os._Environ``, so
    ``putenv``/``unsetenv`` still run and child processes still inherit the
    same environment.  The only added effect is an append to a bounded log.
    """

    __slots__ = ("_target", "_log")

    def __init__(self, target: MutableMapping, log: EnvironmentReadLog) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_log", log)

    # -- read paths --------------------------------------------------------

    def __getitem__(self, key: str) -> str:
        try:
            value = self._target[key]
        except KeyError:
            self._log.record_read(key, present=False, operation="getitem")
            raise
        self._log.record_read(key, present=True, operation="getitem")
        return value

    def get(self, key: str, default: object = None) -> object:
        present = key in self._target
        self._log.record_read(key, present=present, operation="get")
        return self._target[key] if present else default

    def __contains__(self, key: object) -> bool:
        present = key in self._target
        self._log.record_read(key, present=present, operation="contains")
        return present

    def real_environment(self) -> MutableMapping:
        """The unwrapped mapping, so the proxy can be removed again."""
        return self._target

    def __iter__(self) -> Iterator[str]:
        self._log.record_bulk("iter")
        return iter(self._target)

    def __len__(self) -> int:
        return len(self._target)

    def keys(self):  # noqa: ANN201 - delegate exactly
        self._log.record_bulk("keys")
        return self._target.keys()

    def values(self):  # noqa: ANN201 - delegate exactly
        self._log.record_bulk("values")
        return self._target.values()

    def items(self):  # noqa: ANN201 - delegate exactly
        self._log.record_bulk("items")
        return self._target.items()

    def copy(self) -> dict[str, str]:
        self._log.record_bulk("copy")
        return dict(self._target)

    # -- write paths -------------------------------------------------------

    def __setitem__(self, key: str, value: str) -> None:
        self._target[key] = value
        self._log.record_write(key)

    def __delitem__(self, key: str) -> None:
        del self._target[key]
        self._log.record_write(key)

    def setdefault(self, key: str, value: str = "") -> str:  # type: ignore[override]
        present = key in self._target
        self._log.record_read(key, present=present, operation="setdefault")
        if not present:
            self._log.record_write(key)
        return self._target.setdefault(key, value)

    def pop(self, key: str, *args: object) -> object:
        self._log.record_read(key, present=key in self._target, operation="pop")
        self._log.record_write(key)
        return self._target.pop(key, *args)

    def clear(self) -> None:
        self._log.record_bulk("clear")
        self._target.clear()

    # -- passthrough -------------------------------------------------------

    def __repr__(self) -> str:
        return repr(self._target)

    def __getattr__(self, name: str) -> object:
        # ``os._Environ`` exposes encodekey/decodekey/encodevalue/decodevalue
        # and ``os.environ`` is duck-typed by third-party code; forward
        # anything this proxy does not define.
        return getattr(object.__getattribute__(self, "_target"), name)

    @property
    def unwrapped(self) -> MutableMapping:
        return object.__getattribute__(self, "_target")


# --------------------------------------------------------------------------
# Layer 4: runtime observation of ambient filesystem inputs
# --------------------------------------------------------------------------


class AmbientPathLog:
    """Bounded, reentrancy-safe record of files opened for reading."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paths: dict[str, str] = {}
        self._local = threading.local()
        self.truncated = False

    def record_open(self, path: object, mode: object) -> None:
        if type(path) is not str or not path:
            return
        if getattr(self._local, "busy", False):
            return
        self._local.busy = True
        try:
            text_mode = mode if type(mode) is str else ""
            if any(character in text_mode for character in "wxa+"):
                intent = "write"
            else:
                intent = "read"
            with self._lock:
                if path in self._paths:
                    if intent == "write":
                        self._paths[path] = "write"
                    return
                if len(self._paths) >= MAX_AMBIENT_PATHS:
                    self.truncated = True
                    return
                self._paths[path] = intent
        finally:
            self._local.busy = False

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._paths)


class LaunchRecorder:
    """Holds the installed observers for the lifetime of the process."""

    def __init__(self) -> None:
        self.environment_reads = EnvironmentReadLog()
        self.ambient_paths = AmbientPathLog()
        self.environment_proxy_installed = False
        self.audit_hook_installed = False
        self.installation_errors: list[str] = []
        self.installed_at_utc = _utc_now()
        self.uninstalled_at_utc: str | None = None
        self.mode: str | None = None

    @property
    def reversible(self) -> bool:
        """Whether every observer this recorder installed can be removed again.

        ``sys.addaudithook`` cannot be undone in CPython, so a recorder that
        installed the ambient-path hook can never restore an uninstrumented
        process. Only an environment-proxy-only install is reversible.
        """
        return not self.audit_hook_installed

    @property
    def environment_proxy_active(self) -> bool:
        """Whether the proxy is intercepting reads *right now*."""
        return self.environment_proxy_installed and self.uninstalled_at_utc is None

    @property
    def instrumented_phases(self) -> tuple[str, ...]:
        """Which phases ran under instrumentation.

        A reader checking that nothing timed was perturbed wants to see
        ``["startup"]`` alone: the recorder was installed while credentials were
        read and removed before the clock started.
        """
        if self.uninstalled_at_utc is not None and self.reversible:
            return (STARTUP_PHASE,)
        return (STARTUP_PHASE, MEASURED_PHASE)


_RECORDER: LaunchRecorder | None = None


def active_recorder() -> LaunchRecorder | None:
    """Return the installed recorder, if this process installed one."""

    return _RECORDER


def install_launch_recorder(
    *,
    argv: Sequence[str] | None = None,
    modes: Sequence[str] = INSTRUMENTED_MODES,
    force: bool = False,
    ambient_paths: bool = True,
) -> LaunchRecorder | None:
    """Observe environment and ambient-file reads for the rest of the process.

    Call this as early as possible in a runner -- before the imports whose
    module bodies read configuration -- so that import-time reads are seen.
    Returns ``None`` when the invocation is not an instrumented mode, which
    is the normal case for ``live``.

    ``ambient_paths=False`` skips the audit hook. That hook cannot be removed
    from a CPython process, so omitting it is what makes an install reversible
    -- see :func:`instrument_startup_window`.

    Never raises.  A failure to install any observer is recorded and the
    process continues exactly as it would have without instrumentation.
    """

    global _RECORDER
    if _RECORDER is not None:
        return _RECORDER

    words = list(sys.argv[1:] if argv is None else argv)
    mode = next((word for word in words if word in tuple(modes)), None)
    if mode is None and not force:
        return None

    recorder = LaunchRecorder()
    recorder.mode = mode

    try:
        target = os.environ
        if not isinstance(target, RecordingEnviron):
            os.environ = RecordingEnviron(  # type: ignore[assignment]
                target, recorder.environment_reads
            )
            recorder.environment_proxy_installed = True
    except BaseException as error:  # pragma: no cover - defensive
        recorder.installation_errors.append(
            f"environment_proxy:{type(error).__qualname__}"
        )

    if ambient_paths:
        try:
            ambient = recorder.ambient_paths

            def _hook(event: str, arguments: tuple) -> None:
                if event != "open":
                    return
                try:
                    ambient.record_open(
                        arguments[0] if len(arguments) > 0 else None,
                        arguments[1] if len(arguments) > 1 else None,
                    )
                except BaseException:  # pragma: no cover - must never propagate
                    return

            sys.addaudithook(_hook)
            recorder.audit_hook_installed = True
        except BaseException as error:  # pragma: no cover - defensive
            recorder.installation_errors.append(
                f"audit_hook:{type(error).__qualname__}"
            )

    _RECORDER = recorder
    return recorder


def instrument_startup_window(
    *,
    argv: Sequence[str] | None = None,
    modes: Sequence[str] = CAMPAIGN_MODES,
) -> LaunchRecorder | None:
    """Instrument the startup phase of any campaign mode, reversibly.

    ``live`` must remain byte-identical to the uninstrumented process while it
    produces results -- but the evidence that a credential was never read is
    worth having for a live run too, and credentials are read at startup, long
    before any timed work. So observe only the environment, which is removable,
    and call :func:`uninstall_launch_recorder` before the clock starts. The
    audit hook is deliberately not installed: CPython cannot remove it, and a
    recorder that installed it could never hand back an unmodified process.

    Returns ``None`` unless the invocation names a campaign mode, so importing a
    runner under a test harness instruments nothing.
    """

    return install_launch_recorder(argv=argv, modes=modes, ambient_paths=False)


def uninstall_launch_recorder() -> bool:
    """Restore the unmodified process before the measured phase begins.

    Returns True when the environment proxy was removed. The recorder itself
    survives -- its observations are the evidence -- but it stops observing, and
    the launch record then reports ``instrumented_phases: ["startup"]``.

    Never raises. Refuses to claim success if an irreversible observer is
    installed, so a caller cannot mistake a disarmed hook for an absent one.
    """

    recorder = _RECORDER
    if recorder is None:
        return False
    if not recorder.reversible:
        # A lifetime-instrumented mode (``prepare``) is *meant* to observe its
        # whole process, and the audit hook could not be removed anyway. Cutting
        # its window short would lose evidence and still not restore the process.
        return False

    restored = False
    try:
        current = os.environ
        if isinstance(current, RecordingEnviron):
            os.environ = current.real_environment()  # type: ignore[assignment]
            restored = True
    except BaseException as error:  # pragma: no cover - defensive
        recorder.installation_errors.append(f"uninstall:{type(error).__qualname__}")

    recorder.uninstalled_at_utc = _utc_now()
    return restored


# --------------------------------------------------------------------------
# Layer 2: static scan of the campaign's own source closure
# --------------------------------------------------------------------------


class _EnvironmentSurfaceScanner(ast.NodeVisitor):
    """Collect environment names a module can read, plus their code defaults."""

    def __init__(self, label: str, constants: Mapping[str, str]) -> None:
        self.label = label
        self.constants = constants
        self.found: dict[str, dict[str, object]] = {}
        self.dynamic_sites: list[str] = []

    def _add(
        self,
        name: object,
        *,
        line: int,
        evidence: str,
        default: object = None,
    ) -> None:
        if type(name) is not str or not _ENVIRONMENT_NAME_PATTERN.match(name):
            return
        entry = self.found.setdefault(
            name, {"evidence": set(), "sites": set(), "code_defaults": set()}
        )
        entry["evidence"].add(evidence)  # type: ignore[union-attr]
        entry["sites"].add(f"{self.label}:{line}")  # type: ignore[union-attr]
        if type(default) is str:
            entry["code_defaults"].add(default)  # type: ignore[union-attr]

    def _resolve(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and type(node.value) is str:
            return node.value
        if isinstance(node, ast.Name):
            return self.constants.get(node.id)
        if isinstance(node, ast.Attribute):
            return self.constants.get(node.attr)
        return None

    @staticmethod
    def _is_environ(node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "environb"}:
            return True
        # ``environ.get(...)`` / ``environment.get(...)`` inside a helper that
        # takes the mapping as a parameter.
        return isinstance(node, ast.Name) and node.id in {
            "environ",
            "environment",
            "env",
            "_environ",
        }

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr in {"get", "setdefault"} and self._is_environ(
                function.value
            ):
                name = self._resolve(node.args[0]) if node.args else None
                default = (
                    node.args[1].value
                    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                    else None
                )
                if name is None:
                    self.dynamic_sites.append(f"{self.label}:{node.lineno}")
                else:
                    self._add(
                        name,
                        line=node.lineno,
                        evidence="direct_read",
                        default=default,
                    )
            elif function.attr == "getenv":
                name = self._resolve(node.args[0]) if node.args else None
                default = (
                    node.args[1].value
                    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                    else None
                )
                if name is None:
                    self.dynamic_sites.append(f"{self.label}:{node.lineno}")
                else:
                    self._add(
                        name, line=node.lineno, evidence="direct_read", default=default
                    )
        # Any campaign-namespace name handed to any callable is a candidate:
        # this is how a name reaches a shared parsing helper such as
        # ``_bounded_integer_environment(NAME, default, minimum=..., ...)``,
        # which no ``environ.get`` scan can see.  Both the literal spelling
        # and a module-level constant binding are followed.
        for argument in list(node.args) + [word.value for word in node.keywords]:
            candidate: str | None = None
            evidence = "namespace_literal"
            if isinstance(argument, ast.Constant) and type(argument.value) is str:
                candidate = argument.value
            elif isinstance(argument, (ast.Name, ast.Attribute)):
                candidate = self._resolve(argument)
                evidence = "namespace_constant"
            if candidate is not None and candidate.startswith(
                CAMPAIGN_NAMESPACE_PREFIXES
            ):
                self._add(candidate, line=node.lineno, evidence=evidence)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802 - ast API
        if self._is_environ(node.value):
            name = self._resolve(node.slice)
            stored = isinstance(node.ctx, (ast.Store, ast.Del))
            if name is None:
                self.dynamic_sites.append(f"{self.label}:{node.lineno}")
            else:
                self._add(
                    name,
                    line=node.lineno,
                    evidence="assignment" if stored else "direct_read",
                )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:  # noqa: N802 - ast API
        if node.ops and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            for comparator in node.comparators:
                if self._is_environ(comparator):
                    name = self._resolve(node.left)
                    if name is not None:
                        self._add(name, line=node.lineno, evidence="membership_test")
        self.generic_visit(node)


def _module_level_string_constants(tree: ast.AST) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not isinstance(value, ast.Constant) or type(value.value) is not str:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value.value
    return constants


def scan_declared_environment_surface(
    paths: Sequence[Path],
    *,
    relative_to: Path | None = None,
) -> dict[str, object]:
    """Statically derive the environment surface of a source closure.

    Two passes: the first collects module-level string constants across the
    whole closure (so ``environ.get(SOME_ENV_CONSTANT)`` resolves), the
    second scans for reads.  Names that only appear as campaign-namespace
    string literals are reported as ``namespace_literal`` evidence, which is
    how names that reach a shared parsing helper are caught.
    """

    sources: list[tuple[str, ast.AST]] = []
    constants: dict[str, str] = {}
    unparsed: list[str] = []
    for path in paths:
        if path.suffix != ".py":
            continue
        label = _relative_to(path, relative_to) or path.name
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except (SyntaxError, OSError, ValueError):
            unparsed.append(label)
            continue
        sources.append((label, tree))
        constants.update(_module_level_string_constants(tree))

    merged: dict[str, dict[str, object]] = {}
    dynamic_sites: list[str] = []
    for label, tree in sources:
        scanner = _EnvironmentSurfaceScanner(label, constants)
        scanner.visit(tree)
        dynamic_sites.extend(scanner.dynamic_sites)
        for name, entry in scanner.found.items():
            target = merged.setdefault(
                name, {"evidence": set(), "sites": set(), "code_defaults": set()}
            )
            for key in ("evidence", "sites", "code_defaults"):
                target[key].update(entry[key])  # type: ignore[union-attr]

    return {
        "schema_version": 1,
        "scanned_python_file_count": len(sources),
        "unparsed_files": sorted(unparsed),
        "variables": {
            name: {
                "evidence": sorted(entry["evidence"]),  # type: ignore[arg-type]
                "sites": sorted(entry["sites"])[:32],  # type: ignore[arg-type]
                "site_count": len(entry["sites"]),  # type: ignore[arg-type]
                "code_defaults": sorted(entry["code_defaults"]),  # type: ignore[arg-type]
            }
            for name, entry in sorted(merged.items())
        },
        "unresolved_dynamic_read_sites": sorted(set(dynamic_sites)),
    }


# --------------------------------------------------------------------------
# Layer 1: the process environment, verbatim
# --------------------------------------------------------------------------


def _is_credential_name(name: str) -> bool:
    if name in CREDENTIAL_NAME_ALLOWLIST:
        return False
    return CREDENTIAL_NAME_PATTERN.search(name) is not None


def capture_process_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Record every environment variable, redacting credentials by identity."""

    source = os.environ if environ is None else environ
    raw = getattr(source, "unwrapped", source)
    variables: dict[str, object] = {}
    redacted_names: list[str] = []
    for name in sorted(raw.keys()):
        value = raw[name]
        if _is_credential_name(name):
            redacted_names.append(name)
            variables[name] = {
                "present": True,
                "redacted": True,
                "redaction_reason": "credential_name_pattern",
                "value_sha256": _sha256_text(value),
                "value_length": len(value),
            }
        else:
            variables[name] = {
                "present": True,
                "redacted": False,
                "value": value,
            }
    commitment = hashlib.sha256(b"agent-evolve:launch-environment:v1\x00")
    for name in sorted(raw.keys()):
        name_bytes = name.encode("utf-8", errors="surrogatepass")
        value_bytes = raw[name].encode("utf-8", errors="surrogatepass")
        commitment.update(len(name_bytes).to_bytes(8, "big"))
        commitment.update(name_bytes)
        commitment.update(len(value_bytes).to_bytes(8, "big"))
        commitment.update(value_bytes)
    return {
        "schema_version": 1,
        "variable_count": len(variables),
        "redacted_names": redacted_names,
        "redaction_rule": "credential name pattern; value identity recorded instead",
        "environment_commitment_sha256": commitment.hexdigest(),
        "variables": variables,
    }


# --------------------------------------------------------------------------
# Invocation, interpreter and ambient inputs
# --------------------------------------------------------------------------


def _shell_quote(word: str) -> str:
    if word and all(
        character.isalnum() or character in "@%+=:,./-_" for character in word
    ):
        return word
    return "'" + word.replace("'", "'\"'\"'") + "'"


def capture_invocation(
    *,
    workspace_root: Path | None,
    agent_evolve_root: Path | None,
    environment: Mapping[str, object],
) -> dict[str, object]:
    """Record everything needed to retype the command that started this run."""

    cwd = Path.cwd()
    script = Path(sys.argv[0]).resolve(strict=False) if sys.argv else None
    exported = [
        f"{name}={_shell_quote(str(entry.get('value')))}"
        for name, entry in sorted(environment.items())
        if type(entry) is dict
        and entry.get("redacted") is False
        and str(name).startswith(CAMPAIGN_NAMESPACE_PREFIXES + ("PYTHONPATH",))
    ]
    command = " ".join(
        [
            *exported,
            _shell_quote(sys.executable),
            *(_shell_quote(word) for word in sys.argv),
        ]
    )
    return {
        "schema_version": 1,
        "argv": list(sys.argv),
        "script_path": None if script is None else str(script),
        "script_relative_path": (
            None if script is None else _relative_to(script, workspace_root)
        ),
        "cwd": str(cwd),
        "cwd_relative_to_workspace": _relative_to(cwd, workspace_root),
        "workspace_root": None if workspace_root is None else str(workspace_root),
        "agent_evolve_root": (
            None if agent_evolve_root is None else str(agent_evolve_root)
        ),
        "agent_evolve_tree_name": (
            None if agent_evolve_root is None else agent_evolve_root.name
        ),
        "reconstructed_shell_command": command,
        "reconstructed_command_is_complete": not any(
            type(entry) is dict and entry.get("redacted") is True
            for entry in environment.values()
        ),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "user": (os.environ.get("USER") or os.environ.get("LOGNAME")),
        "hostname": platform.node(),
        "umask_octal": _current_umask(),
        "started_at_utc": _utc_now(),
    }


def _current_umask() -> str:
    # ``os.umask`` has no read-only form; set and restore atomically enough
    # for a single-threaded launch preamble.  Recorded because it decides the
    # permissions of every artifact the run publishes.
    try:
        value = os.umask(0o022)
        os.umask(value)
        return oct(value)
    except OSError:  # pragma: no cover - defensive
        return "unknown"


def capture_interpreter(*, workspace_root: Path | None) -> dict[str, object]:
    """Record the interpreter and the import state that resolves the code."""

    try:
        distributions = {
            name: version
            for name, version in sorted(
                (
                    (
                        distribution.metadata["Name"],
                        distribution.version,
                    )
                    for distribution in importlib.metadata.distributions()
                    if distribution.metadata["Name"]
                ),
                key=lambda item: str(item[0]).lower(),
            )
        }
    except Exception:  # pragma: no cover - defensive
        distributions = {}

    def _origin(module_name: str) -> str | None:
        try:
            specification = importlib.util.find_spec(module_name)
        except (ImportError, ValueError, AttributeError):
            return None
        return None if specification is None else specification.origin

    imported_origins = {
        module_name: (
            None
            if getattr(sys.modules.get(module_name), "__file__", None) is None
            else str(sys.modules[module_name].__file__)
        )
        for module_name in sorted(
            {"agent_evolve", "examples", "numpy", "torch", "botorch", "pydantic_ai"}
            & set(sys.modules)
        )
    }

    return {
        "schema_version": 1,
        "executable": sys.executable,
        "executable_realpath": str(Path(sys.executable).resolve(strict=False)),
        "version": sys.version,
        "version_info": list(sys.version_info[:5]),
        "implementation": sys.implementation.name,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "in_virtual_environment": sys.prefix != sys.base_prefix,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "flags_dont_write_bytecode": bool(sys.dont_write_bytecode),
        "hash_randomization_seed_env": os.environ.get("PYTHONHASHSEED"),
        "sys_path": list(sys.path),
        "sys_path_workspace_relative": [
            _relative_to(Path(entry), workspace_root) for entry in sys.path if entry
        ],
        "agent_evolve_module_origin": _origin("agent_evolve"),
        "imported_module_files": imported_origins,
        "installed_distribution_count": len(distributions),
        "installed_distributions": distributions,
        "installed_distributions_sha256": _sha256_text(
            "\n".join(f"{name}=={version}" for name, version in distributions.items())
        ),
    }


def _python_environment_identity(path: Path) -> dict[str, object] | None:
    """Identify the Python environment a recorded path belongs to, if any.

    Campaigns delegate numerical work to *other* interpreters named by
    environment variables -- the pinned qLogNEHVI scorer is one.  Recording
    only that path is not enough: the packages inside it decide the numbers
    it returns, and the campaign's own ``sys.prefix`` says nothing about
    them.  This reads the environment's identity from disk, listing
    ``*.dist-info`` directory names, and never executes anything.
    """

    # Deliberately absolutize without resolving: ``<venv>/bin/python`` is a
    # symlink to the base interpreter, and resolving it walks straight out of
    # the environment whose packages are the thing worth recording.
    current = Path(os.path.abspath(str(path.expanduser())))
    for candidate in (current, *current.parents):
        marker = candidate / "pyvenv.cfg"
        if not marker.is_file():
            continue
        distributions: dict[str, str] = {}
        for site in sorted(candidate.glob("lib/python*/site-packages")):
            for entry in sorted(site.glob("*.dist-info")):
                stem = entry.name[: -len(".dist-info")]
                name, _, version = stem.rpartition("-")
                if name:
                    distributions[name] = version
        try:
            configuration = marker.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - defensive
            configuration = ""
        return {
            "environment_root": str(candidate),
            "pyvenv_cfg": configuration,
            "installed_distribution_count": len(distributions),
            "installed_distributions": distributions,
            "installed_distributions_sha256": _sha256_text(
                "\n".join(
                    f"{name}=={version}"
                    for name, version in sorted(distributions.items())
                )
            ),
        }
    return None


def capture_referenced_python_environments(
    environ: Mapping[str, object],
) -> dict[str, object]:
    """Record the package identity of every Python environment a value names."""

    found: dict[str, object] = {}
    for name, entry in sorted(environ.items()):
        if type(entry) is not dict or entry.get("redacted"):
            continue
        value = entry.get("value")
        if type(value) is not str or not value.startswith("/"):
            continue
        if len(found) >= 8:
            break
        for part in value.split(os.pathsep):
            identity = _python_environment_identity(Path(part))
            if identity is not None:
                found[str(name)] = identity
                break
    return {
        "schema_version": 1,
        "count": len(found),
        "by_environment_variable": found,
    }


def capture_dotenv_files(paths: Sequence[Path]) -> list[dict[str, object]]:
    """Record which ``.env`` files a launch could consult, by name only.

    A ``.env`` is ambient state that no manifest has ever named.  Its *keys*
    are configuration and are recorded; its values are not.
    """

    records: list[dict[str, object]] = []
    for path in paths:
        expanded = path.expanduser()
        exists = expanded.exists()
        keys: list[str] = []
        if exists and expanded.is_file():
            try:
                for line in expanded.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key = stripped.split("=", 1)[0].strip()
                    if key.startswith("export "):
                        key = key[len("export ") :].strip()
                    if key:
                        keys.append(key)
            except OSError:
                keys = []
        records.append(
            {
                "path": str(expanded),
                "exists": exists,
                "is_symlink": expanded.is_symlink(),
                "realpath": (
                    str(expanded.resolve(strict=False)) if exists else None
                ),
                "sha256": _sha256_file(expanded) if exists else None,
                "size_bytes": (
                    expanded.stat().st_size if exists and expanded.is_file() else None
                ),
                "defined_keys": sorted(set(keys)),
                "values_recorded": False,
            }
        )
    return records


def capture_ambient_filesystem_inputs(
    recorder: LaunchRecorder | None,
    *,
    workspace_root: Path | None,
    run_dir: Path | None,
    closure_paths: Sequence[Path],
) -> dict[str, object]:
    """Classify every file this process opened, and hash the external ones.

    ``external`` is the interesting class: files that are neither the
    interpreter, nor the sealed source closure, nor this run's own output.
    Those are exactly the ambient inputs a relaunch has to reproduce.
    """

    if recorder is None or not recorder.audit_hook_installed:
        return {
            "schema_version": 1,
            "recorder_installed": False,
            "reason": (
                "audit hook not installed; this mode is not instrumented"
                if recorder is None
                else "audit hook installation failed"
            ),
        }

    closure = {str(path.resolve(strict=False)) for path in closure_paths}
    interpreter_roots = tuple(
        str(Path(root).resolve(strict=False))
        for root in (sys.prefix, sys.base_prefix, *(sys.path[-4:] or ()))
        if root
    )
    resolved_run_dir = None if run_dir is None else str(run_dir.resolve(strict=False))

    counts = {
        "source_closure": 0,
        "run_directory": 0,
        "interpreter": 0,
        "written": 0,
        "external": 0,
        "missing": 0,
    }
    external: list[dict[str, object]] = []
    hashed_bytes = 0
    truncated_hashes = False
    for raw_path, intent in sorted(recorder.ambient_paths.snapshot().items()):
        try:
            resolved = str(Path(raw_path).resolve(strict=False))
        except OSError:  # pragma: no cover - defensive
            resolved = raw_path
        if intent == "write":
            counts["written"] += 1
            continue
        if resolved in closure:
            counts["source_closure"] += 1
            continue
        if resolved_run_dir is not None and resolved.startswith(resolved_run_dir):
            counts["run_directory"] += 1
            continue
        if resolved.startswith(interpreter_roots):
            counts["interpreter"] += 1
            continue
        path = Path(resolved)
        if not path.is_file():
            counts["missing"] += 1
            continue
        counts["external"] += 1
        if len(external) >= MAX_HASHED_AMBIENT_FILES:
            truncated_hashes = True
            continue
        try:
            size = path.stat().st_size
        except OSError:  # pragma: no cover - defensive
            continue
        digest: str | None = None
        if hashed_bytes + size <= MAX_HASHED_AMBIENT_BYTES:
            digest = _sha256_file(path)
            hashed_bytes += size
        else:
            truncated_hashes = True
        external.append(
            {
                "path": resolved,
                "workspace_relative_path": _relative_to(path, workspace_root),
                "size_bytes": size,
                "sha256": digest,
            }
        )

    roots = sorted(
        {
            str(Path(str(record["path"])).parent)
            for record in external
            if record.get("workspace_relative_path") is not None
        }
    )
    return {
        "schema_version": 1,
        "recorder_installed": True,
        "observed_path_count": sum(counts.values()),
        "counts_by_class": counts,
        "external_read_files": external,
        "external_read_directories": roots[:512],
        "truncated_paths": recorder.ambient_paths.truncated,
        "truncated_hashes": truncated_hashes,
    }


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


def _producer_identity() -> dict[str, object]:
    path = Path(__file__).resolve(strict=False)
    return {
        "module_path": str(path),
        "module_sha256": _sha256_file(path),
        "schema_version": LAUNCH_RECORD_SCHEMA_VERSION,
    }


def build_launch_record(
    *,
    mode: str,
    run_id: str,
    run_dir: Path,
    workspace_root: Path | None = None,
    agent_evolve_root: Path | None = None,
    source_paths: Sequence[Path] = (),
    source_closure: Mapping[str, object] | None = None,
    dotenv_paths: Sequence[Path] = (),
    recorder: LaunchRecorder | None = None,
) -> dict[str, object]:
    """Assemble the complete, machine-readable launch record for a run."""

    if recorder is None:
        recorder = active_recorder()
    process_environment = capture_process_environment()
    variables = process_environment["variables"]
    assert type(variables) is dict

    declared = scan_declared_environment_surface(
        source_paths, relative_to=workspace_root
    )
    declared_variables = declared["variables"]
    assert type(declared_variables) is dict
    observed = (
        recorder.environment_reads.to_record()
        if recorder is not None and recorder.environment_proxy_installed
        else {
            "schema_version": 1,
            "recorder_installed": False,
            "reason": (
                "environment proxy not installed; this mode is not instrumented"
                if recorder is None
                else "environment proxy installation failed"
            ),
        }
    )
    observed_names = observed.get("names")
    observed_names = observed_names if type(observed_names) is dict else {}

    # The resolved configuration: for every name any layer believes is an
    # input, the value the process actually saw.
    resolved: dict[str, object] = {}
    for name in sorted(set(declared_variables) | set(observed_names)):
        entry = variables.get(name)
        declared_entry = declared_variables.get(name)
        declared_entry = declared_entry if type(declared_entry) is dict else {}
        defaults = declared_entry.get("code_defaults") or []
        record: dict[str, object] = {
            "present_in_process_environment": entry is not None,
            "declared_in_source_closure": name in declared_variables,
            "read_at_runtime": name in observed_names,
            "code_defaults": defaults,
            "evidence": declared_entry.get("evidence", []),
        }
        if entry is None:
            record["resolved_value"] = None
            record["resolved_from"] = "code_default" if defaults else "absent"
        elif type(entry) is dict and entry.get("redacted"):
            record["resolved_value"] = None
            record["redacted"] = True
            record["value_sha256"] = entry.get("value_sha256")
            record["value_length"] = entry.get("value_length")
            record["resolved_from"] = "process_environment"
        else:
            record["resolved_value"] = (
                entry.get("value") if type(entry) is dict else None
            )
            record["resolved_from"] = "process_environment"
        resolved[name] = record

    redacted_names = process_environment.get("redacted_names")
    redacted_names = redacted_names if type(redacted_names) is list else []
    withheld_behavioural = sorted(
        str(name)
        for name in redacted_names
        if str(name) in declared_variables or str(name) in observed_names
    )

    return {
        "schema_version": LAUNCH_RECORD_SCHEMA_VERSION,
        "kind": LAUNCH_RECORD_KIND,
        "recorded_at_utc": _utc_now(),
        "mode": mode,
        "run_id": run_id,
        "instrumentation": {
            "environment_proxy_installed": (
                recorder is not None and recorder.environment_proxy_installed
            ),
            "audit_hook_installed": (
                recorder is not None and recorder.audit_hook_installed
            ),
            "installed_at_utc": None if recorder is None else recorder.installed_at_utc,
            "installation_errors": (
                [] if recorder is None else list(recorder.installation_errors)
            ),
            "instrumented_modes": list(INSTRUMENTED_MODES),
            # A reader checking that no measured quantity was perturbed wants
            # these three: the window closed, nothing irreversible was ever
            # installed, and the measured phase ran unobserved.
            "uninstalled_at_utc": (
                None if recorder is None else recorder.uninstalled_at_utc
            ),
            "reversible": recorder is not None and recorder.reversible,
            "environment_proxy_active": (
                recorder is not None and recorder.environment_proxy_active
            ),
            "instrumented_phases": (
                [] if recorder is None else list(recorder.instrumented_phases)
            ),
        },
        "invocation": capture_invocation(
            workspace_root=workspace_root,
            agent_evolve_root=agent_evolve_root,
            environment=variables,
        ),
        "interpreter": capture_interpreter(workspace_root=workspace_root),
        "resolved_environment_inputs": resolved,
        "credentials_withheld_that_affect_behaviour": withheld_behavioural,
        "process_environment": process_environment,
        "declared_environment_surface": declared,
        "observed_environment_reads": observed,
        "referenced_python_environments": capture_referenced_python_environments(
            variables
        ),
        "dotenv_files": capture_dotenv_files(dotenv_paths),
        "ambient_filesystem_inputs": capture_ambient_filesystem_inputs(
            recorder,
            workspace_root=workspace_root,
            run_dir=run_dir,
            closure_paths=source_paths,
        ),
        "source_closure": (
            None
            if source_closure is None
            else {
                "aggregate_sha256": source_closure.get("aggregate_sha256"),
                "file_count": source_closure.get("file_count"),
            }
        ),
        "producer": _producer_identity(),
    }


def write_launch_record(run_dir: Path, record: Mapping[str, object]) -> Path:
    """Publish the launch record atomically beside the rest of the receipts."""

    from examples.development.durable_run_artifacts import write_json_atomic

    target = run_dir / LAUNCH_RECORD_FILENAME
    write_json_atomic(target, dict(record))
    return target


def record_campaign_launch(
    *,
    mode: str,
    run_id: str,
    run_dir: Path,
    workspace_root: Path | None = None,
    agent_evolve_root: Path | None = None,
    source_paths: Sequence[Path] = (),
    source_closure: Mapping[str, object] | None = None,
    dotenv_paths: Sequence[Path] = (),
) -> Path | None:
    """Build and publish the launch record; never fail a campaign for it.

    Instrumentation must not be able to abort a run.  A failure to record is
    reported as a ``launch_record_error.json`` receipt so that the omission is
    visible rather than silent, and the campaign continues.
    """

    try:
        record = build_launch_record(
            mode=mode,
            run_id=run_id,
            run_dir=run_dir,
            workspace_root=workspace_root,
            agent_evolve_root=agent_evolve_root,
            source_paths=source_paths,
            source_closure=source_closure,
            dotenv_paths=dotenv_paths,
        )
        return write_launch_record(run_dir, record)
    except BaseException as error:  # pragma: no cover - defensive
        try:
            from examples.development.durable_run_artifacts import write_json_atomic

            write_json_atomic(
                run_dir / "launch_record_error.json",
                {
                    "schema_version": 1,
                    "kind": "agent-evolve:campaign-launch-record-failure",
                    "failure_type": type(error).__qualname__,
                    "failure_text": str(error)[:2000],
                    "recorded_at_utc": _utc_now(),
                },
            )
        except BaseException:
            pass
        return None


__all__ = [
    "LAUNCH_RECORD_FILENAME",
    "LAUNCH_RECORD_KIND",
    "LAUNCH_RECORD_SCHEMA_VERSION",
    "AmbientPathLog",
    "EnvironmentReadLog",
    "LaunchRecorder",
    "RecordingEnviron",
    "active_recorder",
    "build_launch_record",
    "capture_ambient_filesystem_inputs",
    "capture_dotenv_files",
    "capture_interpreter",
    "capture_invocation",
    "capture_process_environment",
    "capture_referenced_python_environments",
    "install_launch_recorder",
    "record_campaign_launch",
    "scan_declared_environment_surface",
    "write_launch_record",
]
