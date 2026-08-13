"""Bounded execution for authored artifacts: one process per batch, typed outcomes.

The optimizer never executes model-written source in its own process. Each
batch of calls is shipped to ``authored_worker`` through the deny-by-default
subprocess boundary -- exact argv, pinned cwd, no inherited environment --
with a wall-clock timeout enforced here and CPU/memory rlimits applied inside
the worker before the source is compiled. Every failure mode returns a
:class:`RuntimeOutcome` with a typed status; nothing raises into the loop,
because a crashing artifact is an ordinary, countable event, not an
emergency.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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
    ) -> None:
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
        package_root = str(Path(__file__).resolve().parents[2])
        self._policy = ChildProcessPolicy(
            policy_id="authored_artifact_runtime",
            policy_version=2,
            inherited_environment_allowlist=(),
            fixed_environment=(("PYTHONPATH", package_root),),
        )

    @property
    def limits(self) -> RuntimeLimits:
        return self._limits

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
        with tempfile.TemporaryDirectory(prefix="agent_evolve_authored_") as scratch:
            request_path = Path(scratch) / "request.json"
            response_path = Path(scratch) / "response.json"
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
            request_path.write_text(json.dumps(request), encoding="utf-8")

            boundary = ExplicitEnvironmentSubprocessBoundary(
                policy=self._policy, working_directory=Path(scratch)
            )
            launch = ((self._worker,) if self._worker is not None
                      else ("-m", "agent_evolve.infrastructure.authored_worker"))
            argv = (self._python, *launch,
                    str(request_path), str(response_path))
            try:
                result = boundary.run(argv, timeout_s=self._limits.wall_time_s)
            except subprocess.TimeoutExpired:
                return RuntimeOutcome(
                    status="timeout",
                    detail=f"wall limit {self._limits.wall_time_s}s",
                )

            if not response_path.exists():
                if result.returncode != 0:
                    return RuntimeOutcome(
                        status="timeout" if result.returncode == -24 else "crash",
                        detail=(f"rc {result.returncode}: "
                                f"{result.stderr[-_DETAIL_LIMIT:]}"),
                    )
                return RuntimeOutcome(
                    status="bad_shape", detail="worker wrote no response"
                )
            try:
                payload = json.loads(response_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as error:
                return RuntimeOutcome(
                    status="bad_shape",
                    detail=f"unreadable response: {error}"[:_DETAIL_LIMIT],
                )
        status = payload.get("status")
        if status not in STATUSES:
            return RuntimeOutcome(
                status="bad_shape", detail=f"unknown status {status!r}"
            )
        if status != "ok":
            return RuntimeOutcome(
                status=status, detail=str(payload.get("detail", ""))[:_DETAIL_LIMIT]
            )
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != len(calls):
            return RuntimeOutcome(
                status="bad_shape",
                detail=(f"{len(results) if isinstance(results, list) else 'no'}"
                        f" results for {len(calls)} calls"),
            )
        notes = payload.get("notes")
        return RuntimeOutcome(
            status="ok", results=tuple(results),
            notes=dict(notes) if isinstance(notes, dict) else {})
