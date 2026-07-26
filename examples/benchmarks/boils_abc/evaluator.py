"""Pinned, re-entrant Berkeley ABC evaluator for BOiLS action sequences."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

from .actions import config_sha256, expand_abc_commands, normalize_candidate


BOILS_SOURCE_COMMIT = "ee6112d39d1a9e9703fecaf9057193e1ec9dae72"
ABC_SOURCE_COMMIT = "bcfdf592289a408cd67ec19260f8a60a37b085b6"
EPFL_SOURCE_COMMIT = "0060e156826e733d69bf5b3322d1bdd0d03a1f9a"
CURRENT_ABC_SHA256 = (
    "21f3673079a1ea21378b817e5035a3a008ffc76e2656d8739906d059a7928232"
)
LUT_INPUTS = 6

_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "agent_evolve_aaai2027"
_DEFAULT_CIRCUIT_HASHES = {
    "log2": "c0d052af4e95de4c1327a2ceddd855518a052a8f3a3960e6d58c5b5ca65c0dde",
    "multiplier": "9f66477f16653748f444fb00ff076330d51da706607253e98d2ddfdbe69edd75",
    "sin": "872bf88733e99ca72681c22a18951a59cef99fcaa36e598dcf1fa42e384771a3",
    "sqrt": "7c5a28925fb2a6b3f1d0979ceaa93eafabea39fa418ec717e09cb4ff3b882107",
}
CURRENT_CIRCUIT_NAMES = tuple(_DEFAULT_CIRCUIT_HASHES)
_SAFE_CIRCUIT_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_STATS = re.compile(
    r"i/o\s*=\s*(?P<inputs>\d+)/\s*(?P<outputs>\d+).*?"
    r"nd\s*=\s*(?P<lut_count>\d+).*?edge\s*=\s*(?P<edge_count>\d+).*?"
    r"aig\s*=\s*(?P<aig_count>\d+).*?lev\s*=\s*(?P<levels>\d+)"
)
_ABC_ERROR = re.compile(
    r"(?:unknown\s+command|unknown\s+option|cannot\s+open|"
    r"command\s+.*?\s+failed|\*\*\s*cmd\s+error|syntax\s+error|"
    r"segmentation\s+fault)",
    flags=re.IGNORECASE,
)
_EQUIVALENCE_SUCCESS = "Networks are equivalent."


class ProvenanceMismatchError(RuntimeError):
    """A declared binary or circuit no longer matches its pinned identity."""


class AbcEvaluationError(RuntimeError):
    """ABC execution, statistics parsing, or equivalence checking failed."""

    def __init__(self, circuit_name: str, diagnostics: CircuitDiagnostics) -> None:
        super().__init__(
            f"ABC evaluation failed for {circuit_name!r}: {diagnostics.status}"
        )
        self.circuit_name = circuit_name
        self.diagnostics = diagnostics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class CircuitSpec:
    name: str
    source: Path
    expected_sha256: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or _SAFE_CIRCUIT_NAME.fullmatch(self.name) is None:
            raise ValueError("circuit name must use the closed lowercase token grammar")
        if not isinstance(self.source, Path):
            raise TypeError("circuit source must be a pathlib.Path")
        _require_sha256(self.expected_sha256, "circuit expected_sha256")


@dataclass(frozen=True, slots=True)
class AbcEvaluatorSettings:
    """Pure evaluator declaration; filesystem verification occurs at startup."""

    abc_binary: Path
    expected_abc_sha256: str
    circuits: tuple[CircuitSpec, ...]
    abc_source_identity: str | None = None
    circuit_suite_identity: str | None = None
    per_circuit_timeout_s: float = 60.0
    work_root: Path | None = None
    affinity_sets: tuple[tuple[int, ...], ...] = ()
    taskset_binary: Path = Path("/usr/bin/taskset")
    max_diagnostic_chars: int = 20_000

    def __post_init__(self) -> None:
        if not isinstance(self.abc_binary, Path):
            raise TypeError("abc_binary must be a pathlib.Path")
        _require_sha256(self.expected_abc_sha256, "expected_abc_sha256")
        if type(self.circuits) is not tuple or not self.circuits:
            raise ValueError("circuits must be a non-empty tuple")
        if any(type(item) is not CircuitSpec for item in self.circuits):
            raise TypeError("circuits must contain exact CircuitSpec values")
        names = tuple(item.name for item in self.circuits)
        if len(set(names)) != len(names):
            raise ValueError("circuit names must be unique")
        sources = tuple(item.source for item in self.circuits)
        if len(set(sources)) != len(sources):
            raise ValueError("circuit source paths must be unique")
        for name in ("abc_source_identity", "circuit_suite_identity"):
            identity = getattr(self, name)
            if identity is not None and (
                type(identity) is not str
                or not identity.strip()
                or len(identity.encode("utf-8")) > 512
            ):
                raise ValueError(f"{name} must be a bounded non-empty string or None")
        if (
            isinstance(self.per_circuit_timeout_s, bool)
            or not isinstance(self.per_circuit_timeout_s, (int, float))
            or not math.isfinite(float(self.per_circuit_timeout_s))
            or float(self.per_circuit_timeout_s) <= 0
        ):
            raise ValueError("per_circuit_timeout_s must be finite and positive")
        if self.work_root is not None and not isinstance(self.work_root, Path):
            raise TypeError("work_root must be a pathlib.Path or None")
        if type(self.affinity_sets) is not tuple or any(
            type(cpu_set) is not tuple or not cpu_set
            for cpu_set in self.affinity_sets
        ):
            raise ValueError("affinity_sets must be a tuple of non-empty CPU tuples")
        flattened: list[int] = []
        for cpu_set in self.affinity_sets:
            if any(type(cpu) is not int or cpu < 0 for cpu in cpu_set):
                raise ValueError("CPU affinity IDs must be non-negative exact integers")
            if len(set(cpu_set)) != len(cpu_set):
                raise ValueError("a CPU affinity set must not contain duplicates")
            flattened.extend(cpu_set)
        if len(set(flattened)) != len(flattened):
            raise ValueError("affinity sets must be disjoint for exclusive checkout")
        if not isinstance(self.taskset_binary, Path):
            raise TypeError("taskset_binary must be a pathlib.Path")
        if (
            type(self.max_diagnostic_chars) is not int
            or not 1_024 <= self.max_diagnostic_chars <= 1_000_000
        ):
            raise ValueError("max_diagnostic_chars must lie in [1024, 1000000]")

    @classmethod
    def current_circuit_panel(
        cls,
        *,
        circuit_names: tuple[str, ...],
        affinity_sets: tuple[tuple[int, ...], ...] = (),
        per_circuit_timeout_s: float = 60.0,
        cache_root: Path = _DEFAULT_CACHE_ROOT,
    ) -> AbcEvaluatorSettings:
        """Build a hash-pinned subset of the locally qualified Q0 panel."""

        if type(circuit_names) is not tuple or not circuit_names:
            raise ValueError("circuit_names must be a non-empty exact tuple")
        if any(type(name) is not str for name in circuit_names):
            raise TypeError("circuit_names must contain exact strings")
        if len(set(circuit_names)) != len(circuit_names):
            raise ValueError("circuit_names cannot contain duplicates")
        unknown = tuple(
            sorted(set(circuit_names).difference(_DEFAULT_CIRCUIT_HASHES))
        )
        if unknown:
            raise ValueError(
                f"unknown current-panel circuit names: {unknown!r}; "
                f"allowed={CURRENT_CIRCUIT_NAMES!r}"
            )

        circuit_root = cache_root / "lsils_benchmarks" / "arithmetic"
        circuits = tuple(
            CircuitSpec(
                name=name,
                source=circuit_root / f"{name}.blif",
                expected_sha256=digest,
            )
            for name in circuit_names
            for digest in (_DEFAULT_CIRCUIT_HASHES[name],)
        )
        return cls(
            abc_binary=cache_root / "abc" / "abc",
            expected_abc_sha256=CURRENT_ABC_SHA256,
            circuits=circuits,
            abc_source_identity=f"git:{ABC_SOURCE_COMMIT}",
            circuit_suite_identity=f"git:{EPFL_SOURCE_COMMIT}",
            per_circuit_timeout_s=per_circuit_timeout_s,
            work_root=cache_root / "runs" / "boils_abc_agentic",
            affinity_sets=affinity_sets,
        )

    @classmethod
    def current_four_circuit(
        cls,
        *,
        affinity_sets: tuple[tuple[int, ...], ...] = (),
        per_circuit_timeout_s: float = 60.0,
        cache_root: Path = _DEFAULT_CACHE_ROOT,
    ) -> AbcEvaluatorSettings:
        """Build the complete locally qualified Q0 four-circuit declaration."""

        return cls.current_circuit_panel(
            circuit_names=CURRENT_CIRCUIT_NAMES,
            affinity_sets=affinity_sets,
            per_circuit_timeout_s=per_circuit_timeout_s,
            cache_root=cache_root,
        )


@dataclass(frozen=True, slots=True)
class CircuitDiagnostics:
    status: str
    returncode: int | None
    elapsed_s: float
    timeout_s: float
    equivalent: bool
    error_signatures: tuple[str, ...]
    stdout_excerpt: str
    stderr_excerpt: str
    stdout_sha256: str
    stderr_sha256: str
    abc_program: str
    argv: tuple[str, ...]
    cpu_affinity: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class CircuitEvaluation:
    circuit_name: str
    circuit_sha256: str
    inputs: int
    outputs: int
    lut_count: int
    edge_count: int
    aig_count: int
    levels: int
    diagnostics: CircuitDiagnostics

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BoilsEvaluation:
    configuration_sha256: str
    sequence: tuple[str, ...]
    abc_binary_sha256: str
    lut_inputs: int
    circuit_results: tuple[CircuitEvaluation, ...]
    total_lut_count: int
    total_levels: int
    max_levels: int
    elapsed_s: float
    affinity_queue_wait_s: float
    cpu_affinity: tuple[int, ...] | None

    @property
    def objective_values(self) -> dict[str, float]:
        return {
            "total_lut_count": float(self.total_lut_count),
            "total_levels": float(self.total_levels),
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BoilsEvaluationFailure:
    """Immutable trace payload for an expected candidate-local ABC failure."""

    configuration_sha256: str
    sequence: tuple[str, ...]
    abc_binary_sha256: str
    failed_circuit_name: str
    completed_circuit_results: tuple[CircuitEvaluation, ...]
    diagnostics: CircuitDiagnostics
    elapsed_s: float
    affinity_queue_wait_s: float
    cpu_affinity: tuple[int, ...] | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


BoilsEvaluationObservation = BoilsEvaluation | BoilsEvaluationFailure
BoilsEvaluationObserver = Callable[[BoilsEvaluationObservation], None]


@dataclass(frozen=True, slots=True)
class _VerifiedFile:
    path: Path
    sha256: str
    stat_identity: tuple[int, int, int, int]

    def assert_unchanged(self, label: str) -> None:
        try:
            stat = self.path.stat()
        except OSError as exc:
            raise ProvenanceMismatchError(f"{label} is no longer readable") from exc
        current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if current != self.stat_identity:
            raise ProvenanceMismatchError(
                f"{label} changed after its cryptographic startup preflight"
            )


@dataclass(frozen=True, slots=True)
class _ParsedOutput:
    status: str
    stats: dict[str, int] | None
    equivalent: bool
    error_signatures: tuple[str, ...]


def parse_abc_output(stdout: str, stderr: str, returncode: int) -> _ParsedOutput:
    """Classify ABC output independently of its unreliable zero exit status."""

    combined = _ANSI_ESCAPE.sub("", stdout + "\n" + stderr)
    matches = tuple(_STATS.finditer(combined))
    stats = (
        {key: int(value) for key, value in matches[0].groupdict().items()}
        if len(matches) == 1
        else None
    )
    equivalent = _EQUIVALENCE_SUCCESS in combined
    errors = tuple(
        sorted(set(match.group(0) for match in _ABC_ERROR.finditer(combined)))
    )
    if returncode != 0:
        status = "abc_nonzero_exit"
    elif errors:
        status = "abc_reported_error"
    elif len(matches) != 1:
        status = "stats_missing_or_ambiguous"
    elif not equivalent:
        status = "cec_failed_or_missing"
    else:
        status = "passed"
    return _ParsedOutput(status, stats, equivalent, errors)


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _verify_file(path: Path, expected_sha256: str, label: str) -> _VerifiedFile:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ProvenanceMismatchError(f"{label} does not exist") from exc
    if not resolved.is_file():
        raise ProvenanceMismatchError(f"{label} is not a regular file")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ProvenanceMismatchError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return _VerifiedFile(resolved, actual, _stat_identity(resolved))


class BoilsAbcEvaluator:
    """Thread-safe evaluator with isolated workspaces and exclusive CPU leases."""

    def __init__(
        self,
        settings: AbcEvaluatorSettings,
        *,
        observer: BoilsEvaluationObserver | None = None,
    ) -> None:
        if type(settings) is not AbcEvaluatorSettings:
            raise TypeError("settings must be an exact AbcEvaluatorSettings")
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable or None")
        self.settings = settings
        self._observer = observer
        self._abc = _verify_file(
            settings.abc_binary,
            settings.expected_abc_sha256,
            "ABC binary",
        )
        if not os.access(self._abc.path, os.X_OK):
            raise ProvenanceMismatchError("ABC binary is not executable")
        self._circuits = tuple(
            (
                spec,
                _verify_file(
                    spec.source,
                    spec.expected_sha256,
                    f"circuit {spec.name!r}",
                ),
            )
            for spec in settings.circuits
        )
        resolved_circuit_paths = tuple(
            verified.path for _, verified in self._circuits
        )
        if len(set(resolved_circuit_paths)) != len(resolved_circuit_paths):
            raise ProvenanceMismatchError(
                "circuit paths must remain unique after symlink resolution"
            )
        self._taskset: Path | None = None
        self._taskset_sha256: str | None = None
        if settings.affinity_sets:
            try:
                taskset = settings.taskset_binary.resolve(strict=True)
            except OSError as exc:
                raise ProvenanceMismatchError("taskset binary does not exist") from exc
            if not taskset.is_file() or not os.access(taskset, os.X_OK):
                raise ProvenanceMismatchError("taskset binary is not executable")
            self._taskset = taskset
            self._taskset_sha256 = sha256_file(taskset)
        if settings.work_root is None:
            self._work_root = None
        else:
            settings.work_root.mkdir(parents=True, exist_ok=True)
            self._work_root = settings.work_root.resolve(strict=True)
            if not self._work_root.is_dir():
                raise ValueError("work_root must resolve to a directory")
        self._affinity_pool: queue.Queue[tuple[int, ...]] = queue.Queue()
        for cpu_set in settings.affinity_sets:
            self._affinity_pool.put(cpu_set)

    @property
    def concurrency_capacity(self) -> int | None:
        """Exclusive configured evaluator slots, or None when unpinned."""

        return len(self.settings.affinity_sets) or None

    def provenance(self) -> dict[str, Any]:
        return {
            "boils_source_commit": BOILS_SOURCE_COMMIT,
            "abc_source_identity": self.settings.abc_source_identity,
            "abc_binary": str(self._abc.path),
            "abc_binary_sha256": self._abc.sha256,
            "circuit_suite_identity": self.settings.circuit_suite_identity,
            "circuits": [
                {
                    "name": spec.name,
                    "source": str(verified.path),
                    "sha256": verified.sha256,
                }
                for spec, verified in self._circuits
            ],
            "lut_inputs": LUT_INPUTS,
            "per_circuit_timeout_s": float(
                self.settings.per_circuit_timeout_s
            ),
            "affinity_sets": [list(item) for item in self.settings.affinity_sets],
            "taskset_binary": (
                None if self._taskset is None else str(self._taskset)
            ),
            "taskset_binary_sha256": self._taskset_sha256,
        }

    def _assert_inputs_unchanged(self) -> None:
        self._abc.assert_unchanged("ABC binary")
        for spec, verified in self._circuits:
            verified.assert_unchanged(f"circuit {spec.name!r}")

    @contextmanager
    def _reserve_affinity(
        self,
    ) -> Iterator[tuple[tuple[int, ...] | None, float]]:
        if not self.settings.affinity_sets:
            yield None, 0.0
            return
        started = time.perf_counter_ns()
        cpu_set = self._affinity_pool.get()
        wait_s = (time.perf_counter_ns() - started) / 1e9
        try:
            yield cpu_set, wait_s
        finally:
            self._affinity_pool.put(cpu_set)

    def _bounded(self, value: str) -> str:
        limit = self.settings.max_diagnostic_chars
        if len(value) <= limit:
            return value
        omitted = len(value) - limit
        return f"<truncated {omitted} leading characters>\n" + value[-limit:]

    def _argv(
        self,
        abc_program: str,
        cpu_affinity: tuple[int, ...] | None,
    ) -> tuple[str, ...]:
        abc_argv = (str(self._abc.path), "-c", abc_program)
        if cpu_affinity is None:
            return abc_argv
        assert self._taskset is not None
        cpu_list = ",".join(str(cpu) for cpu in cpu_affinity)
        return (str(self._taskset), "--cpu-list", cpu_list, *abc_argv)

    def _diagnostics(
        self,
        *,
        status: str,
        returncode: int | None,
        elapsed_s: float,
        equivalent: bool,
        errors: tuple[str, ...],
        stdout: str,
        stderr: str,
        abc_program: str,
        argv: tuple[str, ...],
        cpu_affinity: tuple[int, ...] | None,
    ) -> CircuitDiagnostics:
        return CircuitDiagnostics(
            status=status,
            returncode=returncode,
            elapsed_s=elapsed_s,
            timeout_s=float(self.settings.per_circuit_timeout_s),
            equivalent=equivalent,
            error_signatures=errors,
            stdout_excerpt=self._bounded(stdout),
            stderr_excerpt=self._bounded(stderr),
            stdout_sha256=hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            stderr_sha256=hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            abc_program=abc_program,
            argv=argv,
            cpu_affinity=cpu_affinity,
        )

    def _evaluate_circuit(
        self,
        *,
        index: int,
        spec: CircuitSpec,
        verified: _VerifiedFile,
        candidate_dir: Path,
        transform_commands: tuple[str, ...],
        cpu_affinity: tuple[int, ...] | None,
    ) -> CircuitEvaluation:
        # Arbitrary source/work-root paths never enter the ABC language.  Each
        # source is copied into a generated directory and addressed by a fixed,
        # relative token, closing the command-injection boundary.
        circuit_dir = candidate_dir / f"circuit-{index:03d}"
        circuit_dir.mkdir()
        source_name = "source.blif"
        mapped_name = "mapped.blif"
        shutil.copyfile(verified.path, circuit_dir / source_name)
        commands = (
            f"read {source_name}",
            "strash",
            *transform_commands,
            f"if -K {LUT_INPUTS}",
            "print_stats",
            f"write_blif {mapped_name}",
            f"cec {source_name} {mapped_name}",
        )
        abc_program = "; ".join(commands) + ";"
        argv = self._argv(abc_program, cpu_affinity)
        started = time.perf_counter_ns()
        try:
            completed = subprocess.run(
                argv,
                cwd=circuit_dir,
                check=False,
                shell=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=float(self.settings.per_circuit_timeout_s),
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed_s = (time.perf_counter_ns() - started) / 1e9
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            diagnostics = self._diagnostics(
                status="timeout",
                returncode=None,
                elapsed_s=elapsed_s,
                equivalent=False,
                errors=(),
                stdout=stdout,
                stderr=stderr,
                abc_program=abc_program,
                argv=argv,
                cpu_affinity=cpu_affinity,
            )
            raise AbcEvaluationError(spec.name, diagnostics) from None

        elapsed_s = (time.perf_counter_ns() - started) / 1e9
        parsed = parse_abc_output(
            completed.stdout,
            completed.stderr,
            completed.returncode,
        )
        diagnostics = self._diagnostics(
            status=parsed.status,
            returncode=completed.returncode,
            elapsed_s=elapsed_s,
            equivalent=parsed.equivalent,
            errors=parsed.error_signatures,
            stdout=completed.stdout,
            stderr=completed.stderr,
            abc_program=abc_program,
            argv=argv,
            cpu_affinity=cpu_affinity,
        )
        if parsed.status != "passed" or parsed.stats is None:
            raise AbcEvaluationError(spec.name, diagnostics)
        return CircuitEvaluation(
            circuit_name=spec.name,
            circuit_sha256=verified.sha256,
            diagnostics=diagnostics,
            **parsed.stats,
        )

    def evaluate(self, config: object) -> BoilsEvaluation:
        """Evaluate one candidate synchronously; safe to call from worker threads.

        When configured, the observer is called exactly once for a completed
        evaluation or an expected :class:`AbcEvaluationError`.  Notification
        occurs synchronously after temporary workspaces and affinity leases are
        released.  Concurrent callers may invoke the observer concurrently, so
        persistence implementations must provide their own synchronization.
        Observer exceptions propagate to let audit-sensitive callers fail closed.
        """

        sequence = normalize_candidate(config)
        transform_commands = expand_abc_commands(config)
        configuration_sha256 = config_sha256(config)
        self._assert_inputs_unchanged()
        result: BoilsEvaluation | None = None
        failure: BoilsEvaluationFailure | None = None
        evaluation_error: AbcEvaluationError | None = None
        with self._reserve_affinity() as (cpu_affinity, queue_wait_s):
            started = time.perf_counter_ns()
            completed_results: list[CircuitEvaluation] = []
            try:
                with tempfile.TemporaryDirectory(
                    prefix="candidate-",
                    dir=None if self._work_root is None else str(self._work_root),
                ) as temporary:
                    candidate_dir = Path(temporary)
                    for index, (spec, verified) in enumerate(self._circuits):
                        completed_results.append(
                            self._evaluate_circuit(
                                index=index,
                                spec=spec,
                                verified=verified,
                                candidate_dir=candidate_dir,
                                transform_commands=transform_commands,
                                cpu_affinity=cpu_affinity,
                            )
                        )
            except AbcEvaluationError as exc:
                elapsed_s = (time.perf_counter_ns() - started) / 1e9
                failure = BoilsEvaluationFailure(
                    configuration_sha256=configuration_sha256,
                    sequence=tuple(sequence),
                    abc_binary_sha256=self._abc.sha256,
                    failed_circuit_name=exc.circuit_name,
                    completed_circuit_results=tuple(completed_results),
                    diagnostics=exc.diagnostics,
                    elapsed_s=elapsed_s,
                    affinity_queue_wait_s=queue_wait_s,
                    cpu_affinity=cpu_affinity,
                )
                evaluation_error = exc
            else:
                elapsed_s = (time.perf_counter_ns() - started) / 1e9
                results = tuple(completed_results)
                result = BoilsEvaluation(
                    configuration_sha256=configuration_sha256,
                    sequence=tuple(sequence),
                    abc_binary_sha256=self._abc.sha256,
                    lut_inputs=LUT_INPUTS,
                    circuit_results=results,
                    total_lut_count=sum(item.lut_count for item in results),
                    total_levels=sum(item.levels for item in results),
                    max_levels=max(item.levels for item in results),
                    elapsed_s=elapsed_s,
                    affinity_queue_wait_s=queue_wait_s,
                    cpu_affinity=cpu_affinity,
                )

        if failure is not None:
            if self._observer is not None:
                self._observer(failure)
            assert evaluation_error is not None
            raise evaluation_error
        assert result is not None
        if self._observer is not None:
            self._observer(result)
        return result

    async def evaluate_async(self, config: object) -> BoilsEvaluation:
        """Async convenience boundary for orchestrators outside AgentEvolve."""

        return await asyncio.to_thread(self.evaluate, config)


__all__ = [
    "ABC_SOURCE_COMMIT",
    "BOILS_SOURCE_COMMIT",
    "CURRENT_ABC_SHA256",
    "CURRENT_CIRCUIT_NAMES",
    "EPFL_SOURCE_COMMIT",
    "LUT_INPUTS",
    "AbcEvaluationError",
    "AbcEvaluatorSettings",
    "BoilsAbcEvaluator",
    "BoilsEvaluation",
    "BoilsEvaluationFailure",
    "BoilsEvaluationObservation",
    "BoilsEvaluationObserver",
    "CircuitDiagnostics",
    "CircuitEvaluation",
    "CircuitSpec",
    "ProvenanceMismatchError",
    "parse_abc_output",
    "sha256_file",
]
