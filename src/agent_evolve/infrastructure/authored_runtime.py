"""Bounded execution for authored artifacts: typed outcomes, two transports.

The optimizer never executes model-written source in its own process. Each
batch of calls is shipped to ``authored_worker`` through the deny-by-default
subprocess boundary -- exact argv, pinned cwd, no inherited environment --
with a wall-clock timeout enforced here and CPU/memory rlimits applied inside
the worker before the source is compiled. Every failure mode returns a
:class:`RuntimeOutcome` with a typed status; nothing raises into the loop,
because a crashing artifact is an ordinary, countable event, not an
emergency.

TWO TRANSPORTS, ONE WORKER. By default a batch gets its own process and the
process dies with it. With ``persistent=True`` one worker is kept alive per
runtime instance and every batch travels to it as one JSON line on its stdin,
answered as one JSON line on its stdout. The worker script, the import gate,
the validation and the payload that decides each status are the SAME code in
both modes (``authored_worker.execute``); only the route the bytes take
differs, which is what makes output identity a property that can be tested
rather than promised.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from agent_evolve.core.authored import AuthoredArtifact
from agent_evolve.infrastructure.subprocess_boundary import (
    ExplicitEnvironmentSubprocessBoundary,
)
from agent_evolve.ports.subprocess_boundary import ChildProcessPolicy

__all__ = ["RuntimeLimits", "RuntimeOutcome", "AuthoredRuntime"]

_DETAIL_LIMIT = 500
_READ_CHUNK = 1 << 16
_REAP_GRACE_S = 5.0

#: Statuses the runtime can report. ``ok`` carries results; everything else
#: carries a bounded detail string and empty results.
STATUSES = ("ok", "unparseable", "forbidden_import", "crash", "timeout",
            "memory", "bad_shape")


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    wall_time_s: float = 10.0
    cpu_seconds: int = 8
    memory_bytes: int = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RuntimeOutcome:
    status: str
    results: tuple = ()
    detail: str = ""
    #: Counters a harness-written prelude published inside the sandbox (see
    #: ``policies/emit_scaffold.py``). Empty whenever no prelude ran or the
    #: prelude wrote nothing serializable -- diagnostics, never results.
    notes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class AuthoredRuntime:
    """Runs authored artifacts out of process, one spawned worker per batch."""

    def __init__(
        self,
        *,
        limits: RuntimeLimits | None = None,
        python_executable: str | None = None,
        persistent: bool = False,
    ) -> None:
        """Bounded runtime for authored artifacts.

        *persistent* keeps ONE worker process alive for this instance and
        serves every batch over its stdio instead of spawning per batch. The
        motivation is measured: a screen or a generated pool costs one spawn
        per fit, per predict and per emit, and a NAS cell at B=384 makes
        thousands of them -- 15-90 minutes of a cell's wall is interpreter
        startup, not authored code. The spawn itself is already as cheap as it
        gets (922ms by module form, 21ms by path; see the launch comment
        below), so the remaining win is not making it at all.

        THE DEFAULT STAYS ONE-SHOT. Campaign runners opt in explicitly by
        constructing the runtime with ``persistent=True``; no config field
        turns it on, and it will not become the default until a byte-identity
        proof exists on the SEALED budgets on top of the wall win --
        ``tests/test_persistent_runtime.py`` proves outcome-for-outcome
        equality on a battery, which is the necessary half, not the sufficient
        one. Persistent mode needs POSIX (it selects on a pipe); the one-shot
        path runs anywhere.
        """

        self._limits = limits or RuntimeLimits()
        self._python = python_executable or sys.executable
        # The worker is launched BY PATH rather than as `-m agent_evolve...`.
        # It imports nothing from this package (ast, json, sys, traceback and
        # the interpreter), so the module form spent the whole of
        # `agent_evolve.__init__` -- pydantic included -- on every batch:
        # measured at 922ms per spawn against 21ms by path, which is 45x on
        # every screen, every operator generation and every generated pool.
        # The module form stays as a fallback for exotic installs (zipimport)
        # where the source file is not on disk.
        worker = Path(__file__).resolve().with_name("authored_worker.py")
        self._worker = str(worker) if worker.is_file() else None
        self._launch = ((self._worker,) if self._worker is not None
                        else ("-m", "agent_evolve.infrastructure.authored_worker"))
        package_root = str(Path(__file__).resolve().parents[2])
        self._policy = ChildProcessPolicy(
            policy_id="authored_artifact_runtime",
            policy_version=2,
            inherited_environment_allowlist=(),
            fixed_environment=(("PYTHONPATH", package_root),),
        )
        self._persistent = bool(persistent)
        # One request at a time down one pipe: the lock is what makes a shared
        # runtime safe to hand to two callers, not a hint that it is meant to
        # be. Uncontended in the single-threaded loop.
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._stderr = None
        self._buffer = b""
        self._home: tempfile.TemporaryDirectory | None = None
        self._spawns = 0
        self._respawns = 0

    @property
    def limits(self) -> RuntimeLimits:
        return self._limits

    @property
    def persistent(self) -> bool:
        return self._persistent

    @property
    def respawns(self) -> int:
        """Workers started after the first: timeouts, crashes and recycles.

        A run whose respawn count tracks its batch count has lost the whole
        point of the persistent transport and should be read as a defect
        report, not a curiosity.
        """

        return self._respawns

    @property
    def worker_pid(self) -> int | None:
        """The live worker's pid, or ``None``. Diagnostics and tests only."""

        process = self._process
        return None if process is None else process.pid

    def call(
        self,
        artifact: AuthoredArtifact,
        calls: Sequence[Sequence[Any]],
        *,
        prelude: Optional[str] = None,
        notes_global: Optional[str] = None,
    ) -> RuntimeOutcome:
        """Run *artifact* against every argument list in *calls*, bounded.

        One process for the whole batch: a per-call process would pay the
        interpreter startup once per candidate, and the batch is the unit the
        loop actually needs (screen a pool, vary a generation's offspring).

        *prelude* is HARNESS-written source executed into the artifact's
        namespace before it -- machinery the caller would otherwise be asking
        the model to re-derive, such as the emit scaffold that makes a
        shape error impossible to construct. It is not import-gated, because
        it is not the model's code; the artifact still is. *notes_global*
        names a variable the prelude may leave counters in, returned as
        :attr:`RuntimeOutcome.notes`.
        """

        if not calls:
            return RuntimeOutcome(status="ok", results=())
        request = {
            "schema_version": 1,
            "entry_point": artifact.entry_point,
            "source": artifact.source,
            "limits": {
                "cpu_seconds": self._limits.cpu_seconds,
                "memory_bytes": self._limits.memory_bytes,
            },
            "calls": [list(call) for call in calls],
        }
        if prelude:
            request["prelude"] = prelude
            request["notes_global"] = notes_global or ""
        if self._persistent:
            return self._call_persistent(request, len(calls))
        return self._call_one_shot(request, len(calls))

    # -- transport: one process per batch ---------------------------------

    def _call_one_shot(self, request: dict, count: int) -> RuntimeOutcome:
        with tempfile.TemporaryDirectory(prefix="agent_evolve_authored_") as scratch:
            request_path = Path(scratch) / "request.json"
            response_path = Path(scratch) / "response.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")

            boundary = ExplicitEnvironmentSubprocessBoundary(
                policy=self._policy, working_directory=Path(scratch)
            )
            argv = (self._python, *self._launch,
                    str(request_path), str(response_path))
            try:
                result = boundary.run(argv, timeout_s=self._limits.wall_time_s)
            except subprocess.TimeoutExpired:
                return self._wall_timeout()

            if not response_path.exists():
                if result.returncode != 0:
                    return self._dead_worker(result.returncode, result.stderr)
                return RuntimeOutcome(
                    status="bad_shape", detail="worker wrote no response"
                )
            try:
                text = response_path.read_text(encoding="utf-8")
            except OSError as error:
                return RuntimeOutcome(
                    status="bad_shape",
                    detail=f"unreadable response: {error}"[:_DETAIL_LIMIT],
                )
        return self._outcome_from_text(text, count)

    # -- transport: one process per runtime --------------------------------

    def _call_persistent(self, request: dict, count: int) -> RuntimeOutcome:
        # Encoded before the worker is touched, so that a caller who passes
        # non-JSON arguments gets the same TypeError from the same place it
        # already came from on the one-shot path.
        line = json.dumps(request)
        with self._lock:
            for _attempt in (0, 1):
                self._ensure_worker()
                try:
                    self._write(line)
                except (OSError, ValueError):
                    # The worker died before it could read the request, so the
                    # request never ran: one retry against a fresh worker is
                    # honest, and only the second failure is the caller's.
                    self._discard_worker()
                    continue
                kind, text = self._read_reply(
                    time.monotonic() + self._limits.wall_time_s)
                if kind == "timeout":
                    self._discard_worker()
                    return self._wall_timeout()
                if kind == "eof":
                    returncode, stderr = self._discard_worker()
                    if returncode is None:
                        return RuntimeOutcome(
                            status="crash", detail="worker vanished")
                    return self._dead_worker(returncode, stderr)
                outcome = self._outcome_from_text(text, count)
                if outcome.status == "memory":
                    # The allocation that hit RLIMIT_AS is gone, but the
                    # process that hit it is not the process the next batch
                    # deserves: freed arenas are not always returned, and the
                    # one-shot path answers the next batch from a virgin
                    # address space. Recycling buys that back for the price of
                    # one spawn per OOM, which is rare by construction.
                    self._discard_worker()
                return outcome
        return RuntimeOutcome(status="crash", detail="worker stdin closed twice")

    def _ensure_worker(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            return
        if process is not None:              # died while idle; reap, then spawn
            self._discard_worker()
        self._spawn()

    def _spawn(self) -> None:
        if self._home is None:
            self._home = tempfile.TemporaryDirectory(
                prefix="agent_evolve_authored_serve_")
        home = Path(self._home.name).resolve()
        # The boundary port speaks `run()` -- one command, one wait -- and has
        # no streaming form, so the policy is applied here by hand: exact argv,
        # pinned absolute cwd, and an environment built from the SAME policy
        # object the one-shot path hands the boundary. The policy stays the
        # single source of truth; only the waiting differs.
        environment = {
            name: os.environ[name]
            for name in self._policy.inherited_environment_allowlist
            if name in os.environ
        }
        environment.update(dict(self._policy.fixed_environment))
        # Stderr goes to a file rather than a pipe: a pipe nobody drains fills
        # at 64KB and wedges the worker mid-batch, and the parent cannot drain
        # one while it is blocked waiting for a reply on the other.
        self._stderr = open(home / f"worker.{self._spawns}.err", "w+b")
        limits = json.dumps({
            "cpu_seconds": self._limits.cpu_seconds,
            "memory_bytes": self._limits.memory_bytes,
        })
        argv = [self._python, *self._launch, "--serve", limits]
        self._process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            cwd=str(home),
            env=environment,
            bufsize=0,
            close_fds=True,
            shell=False,
        )
        self._buffer = b""
        if self._spawns:
            self._respawns += 1
        self._spawns += 1

    def _discard_worker(self) -> tuple[int | None, str]:
        """Kill and reap the worker; return its exit status and its stderr."""

        process, self._process = self._process, None
        stderr, self._stderr = self._stderr, None
        self._buffer = b""
        returncode: int | None = None
        if process is not None:
            returncode = process.poll()
            if returncode is None:
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                returncode = process.wait(timeout=_REAP_GRACE_S)
            except subprocess.TimeoutExpired:   # pragma: no cover - SIGKILL wins
                returncode = None
            for stream in (process.stdin, process.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
        text = ""
        if stderr is not None:
            try:
                stderr.seek(0)
                text = stderr.read().decode("utf-8", "replace")
            except (OSError, ValueError):
                text = ""
            try:
                stderr.close()
                os.unlink(stderr.name)
            except OSError:
                pass
        return returncode, text

    def _write(self, line: str) -> None:
        # Write the whole request, then read the whole reply: a batch bigger
        # than the pipe buffer would deadlock a protocol where both ends could
        # be mid-message at once, and this one cannot be -- the worker frames
        # on the newline, so it has consumed every byte of the request before
        # it writes the first byte of the answer.
        process = self._process
        if process is None or process.stdin is None:
            raise BrokenPipeError("no worker to write to")
        payload = memoryview((line + "\n").encode("utf-8"))
        while payload:
            written = process.stdin.write(payload)
            if not written:
                raise BrokenPipeError("worker stdin accepted nothing")
            payload = payload[written:]

    def _read_reply(self, deadline: float) -> tuple[str, str]:
        """One reply line, or why there was none: ``line`` / ``timeout`` / ``eof``."""

        process = self._process
        if process is None or process.stdout is None:
            return "eof", ""
        handle = process.stdout
        while True:
            index = self._buffer.find(b"\n")
            if index >= 0:
                line = self._buffer[:index]
                self._buffer = self._buffer[index + 1:]
                return "line", line.decode("utf-8", "replace")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return "timeout", ""
            try:
                ready, _w, _x = select.select([handle], [], [], remaining)
            except (OSError, ValueError):
                return "eof", ""
            if not ready:
                return "timeout", ""
            try:
                # Raw read on the fd, which is safe only because the worker is
                # spawned with bufsize=0: a buffered reader could be holding
                # bytes that select() has already reported and os.read cannot
                # see, and the reply would hang with its tail in userspace.
                chunk = os.read(handle.fileno(), _READ_CHUNK)
            except (OSError, ValueError):
                return "eof", ""
            if not chunk:
                return "eof", ""
            self._buffer += chunk

    # -- shared verdicts ---------------------------------------------------

    def _wall_timeout(self) -> RuntimeOutcome:
        return RuntimeOutcome(
            status="timeout", detail=f"wall limit {self._limits.wall_time_s}s")

    def _dead_worker(self, returncode: int, stderr: str) -> RuntimeOutcome:
        # -24 is SIGXCPU: the CPU rlimit fired, which is a timeout wearing a
        # different signal number.
        return RuntimeOutcome(
            status="timeout" if returncode == -24 else "crash",
            detail=f"rc {returncode}: {stderr[-_DETAIL_LIMIT:]}",
        )

    def _outcome_from_text(self, text: str, count: int) -> RuntimeOutcome:
        try:
            payload = json.loads(text)
        except ValueError as error:
            return RuntimeOutcome(
                status="bad_shape",
                detail=f"unreadable response: {error}"[:_DETAIL_LIMIT],
            )
        status = payload.get("status") if isinstance(payload, dict) else None
        if status not in STATUSES:
            return RuntimeOutcome(
                status="bad_shape", detail=f"unknown status {status!r}"
            )
        if status != "ok":
            return RuntimeOutcome(
                status=status, detail=str(payload.get("detail", ""))[:_DETAIL_LIMIT]
            )
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != count:
            return RuntimeOutcome(
                status="bad_shape",
                detail=(f"{len(results) if isinstance(results, list) else 'no'}"
                        f" results for {count} calls"),
            )
        notes = payload.get("notes")
        return RuntimeOutcome(
            status="ok", results=tuple(results),
            notes=dict(notes) if isinstance(notes, dict) else {})

    # -- lifetime ----------------------------------------------------------

    def close(self) -> None:
        """Terminate the worker and remove its scratch. Idempotent."""

        with self._lock:
            self._discard_worker()
            home, self._home = self._home, None
        if home is not None:
            try:
                home.cleanup()
            except OSError:
                pass

    def __enter__(self) -> "AuthoredRuntime":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:                    # pragma: no cover - shutdown
        # Best effort only: at interpreter shutdown half the module globals
        # this touches may already be None, and a runtime that fails to tidy
        # up is not worth an exception nobody can read.
        try:
            self.close()
        except Exception:
            pass
