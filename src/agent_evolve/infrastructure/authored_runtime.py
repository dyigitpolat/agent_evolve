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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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
        # The worker resolves `agent_evolve` through this path whether the
        # package is installed or running from a source tree; pointing at the
        # installed location is harmless, pointing at src/ is load-bearing.
        package_root = str(Path(__file__).resolve().parents[2])
        self._policy = ChildProcessPolicy(
            policy_id="authored_artifact_runtime",
            policy_version=1,
            inherited_environment_allowlist=(),
            fixed_environment=(("PYTHONPATH", package_root),),
        )

    @property
    def limits(self) -> RuntimeLimits:
        return self._limits

    def call(
        self, artifact: AuthoredArtifact, calls: Sequence[Sequence[Any]]
    ) -> RuntimeOutcome:
        """Run *artifact* against every argument list in *calls*, bounded.

        One process for the whole batch: a per-call process would pay the
        interpreter startup once per candidate, and the batch is the unit the
        loop actually needs (screen a pool, vary a generation's offspring).
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
            request_path.write_text(json.dumps(request), encoding="utf-8")

            boundary = ExplicitEnvironmentSubprocessBoundary(
                policy=self._policy, working_directory=Path(scratch)
            )
            argv = (
                self._python, "-m",
                "agent_evolve.infrastructure.authored_worker",
                str(request_path), str(response_path),
            )
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
        return RuntimeOutcome(status="ok", results=tuple(results))
