"""Dependency-isolated qLogNEHVI policy for the ordinary agent runtime."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
from dataclasses import dataclass, field

from agent_evolve.infrastructure.subprocess_boundary import (
    ExplicitEnvironmentSubprocessBoundary,
)
from agent_evolve.integrations.botorch.finite_qlognehvi_identity import (
    POLICY_DEFINITION_SHA256 as UNDERLYING_DEFINITION_SHA256,
    POLICY_ID as UNDERLYING_POLICY_ID,
    POLICY_VERSION as UNDERLYING_POLICY_VERSION,
)
from agent_evolve.ports.finite_acquisition import (
    FiniteAcquisitionDecision,
    FiniteAcquisitionPolicy,
    FiniteAcquisitionRequest,
)
from agent_evolve.ports.finite_acquisition_json import (
    finite_acquisition_decision_from_record,
)
from agent_evolve.ports.subprocess_boundary import (
    BoundedSubprocessBoundary,
    ChildProcessPolicy,
)


POLICY_ID = "isolated_botorch_finite_qlognehvi"
POLICY_VERSION = 1
WORKER_MODULE = "agent_evolve.integrations.botorch.finite_qlognehvi_worker"
_DEFINITION_DOMAIN = b"agent-evolve:isolated-botorch-finite-qlognehvi:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class FiniteAcquisitionSubprocessError(RuntimeError):
    """The isolated numerical expert failed before an authenticated decision."""


@dataclass(frozen=True, slots=True)
class IsolatedBotorchQLogNehviFiniteAcquisition:
    """Run the optional numerical expert in a pinned dependency environment."""

    boundary: BoundedSubprocessBoundary
    python_executable: Path
    source_root: Path
    mc_samples: int = 128
    maximum_optimizer_batch_size: int = 2048
    timeout_s: float = 900.0
    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    definition_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, BoundedSubprocessBoundary):
            raise TypeError("boundary must implement BoundedSubprocessBoundary")
        if not isinstance(self.python_executable, Path) or not self.python_executable.is_absolute():
            raise ValueError("python_executable must be an absolute pathlib.Path")
        if not isinstance(self.source_root, Path) or not self.source_root.is_absolute():
            raise ValueError("source_root must be an absolute pathlib.Path")
        executable = self.python_executable.resolve(strict=True)
        source_root = self.source_root.resolve(strict=True)
        if not executable.is_file():
            raise ValueError("python_executable must resolve to a file")
        if not source_root.is_dir() or not (source_root / "agent_evolve").is_dir():
            raise ValueError("source_root must contain the agent_evolve package")
        if type(self.mc_samples) is not int or self.mc_samples < 16:
            raise ValueError("mc_samples must be an exact integer of at least 16")
        if (
            type(self.maximum_optimizer_batch_size) is not int
            or self.maximum_optimizer_batch_size < 1
        ):
            raise ValueError("maximum_optimizer_batch_size must be positive")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(float(self.timeout_s))
            or float(self.timeout_s) <= 0.0
        ):
            raise ValueError("timeout_s must be finite and positive")
        worker = Path(__file__).with_name("finite_qlognehvi_worker.py").resolve(
            strict=True
        )
        direct_policy = Path(__file__).with_name("finite_qlognehvi.py").resolve(
            strict=True
        )
        record = {
            "schema_version": 1,
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "underlying_policy": {
                "policy_id": UNDERLYING_POLICY_ID,
                "policy_version": UNDERLYING_POLICY_VERSION,
                "definition_sha256": UNDERLYING_DEFINITION_SHA256,
            },
            "worker_module": WORKER_MODULE,
            "worker_source_sha256": _file_sha256(worker),
            "direct_policy_source_sha256": _file_sha256(direct_policy),
            "python_executable_sha256": _file_sha256(executable),
            "python_invocation": self.boundary.invocation_observation(
                str(self.python_executable)
            ),
            "process_boundary_identity_sha256": self.boundary.identity_sha256,
            "mc_samples": self.mc_samples,
            "maximum_optimizer_batch_size": self.maximum_optimizer_batch_size,
            "timeout_s_hex": float(self.timeout_s).hex(),
            "transport": "authenticated-json-files-exact-argv-no-shell",
            "workload_model_provider_branches": False,
        }
        object.__setattr__(
            self,
            "definition_sha256",
            hashlib.sha256(_DEFINITION_DOMAIN + _canonical_json(record)).hexdigest(),
        )

    def select(self, request: FiniteAcquisitionRequest) -> FiniteAcquisitionDecision:
        self.__post_init__()
        if type(request) is not FiniteAcquisitionRequest:
            raise TypeError("request must be an exact FiniteAcquisitionRequest")
        FiniteAcquisitionRequest.__post_init__(request)
        with tempfile.TemporaryDirectory(prefix="agent-evolve-qlognehvi-") as directory:
            root = Path(directory)
            request_path = root / "request.json"
            response_path = root / "response.json"
            request_path.write_bytes(_canonical_json(request.to_record()))
            result = self.boundary.run(
                (
                    str(self.python_executable),
                    "-m",
                    WORKER_MODULE,
                    "--request",
                    str(request_path),
                    "--response",
                    str(response_path),
                    "--mc-samples",
                    str(self.mc_samples),
                    "--maximum-optimizer-batch-size",
                    str(self.maximum_optimizer_batch_size),
                ),
                timeout_s=float(self.timeout_s),
            )
            if result.returncode != 0:
                stderr = result.stderr[-8000:]
                raise FiniteAcquisitionSubprocessError(
                    f"isolated qLogNEHVI exited {result.returncode}: {stderr}"
                )
            if not response_path.is_file():
                raise FiniteAcquisitionSubprocessError(
                    "isolated qLogNEHVI produced no response"
                )
            payload = json.loads(response_path.read_text(encoding="ascii"))
        if (
            type(payload) is not dict
            or set(payload) != {"schema_version", "decision"}
            or payload["schema_version"] != 1
        ):
            raise ValueError("isolated qLogNEHVI returned a foreign response schema")
        underlying = finite_acquisition_decision_from_record(payload["decision"])
        if underlying.request_sha256 != request.request_sha256:
            raise ValueError("isolated qLogNEHVI decision targets another request")
        if (
            underlying.policy_id,
            underlying.policy_version,
            underlying.policy_definition_sha256,
        ) != (
            UNDERLYING_POLICY_ID,
            UNDERLYING_POLICY_VERSION,
            UNDERLYING_DEFINITION_SHA256,
        ):
            raise ValueError("isolated qLogNEHVI returned a foreign policy identity")
        diagnostics = dict(underlying.diagnostics)
        diagnostics.update(
            {
                "execution_boundary_sha256": self.boundary.identity_sha256,
                "underlying_decision_sha256": underlying.decision_sha256,
                "underlying_policy_definition_sha256": (
                    UNDERLYING_DEFINITION_SHA256
                ),
                "worker_stderr_sha256": hashlib.sha256(
                    result.stderr.encode("utf-8")
                ).hexdigest(),
                "worker_stdout_sha256": hashlib.sha256(
                    result.stdout.encode("utf-8")
                ).hexdigest(),
            }
        )
        return FiniteAcquisitionDecision(
            request_sha256=request.request_sha256,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            selected=underlying.selected,
            diagnostics=tuple(sorted(diagnostics.items())),
        )


def build_isolated_botorch_qlognehvi(
    *,
    python_executable: Path,
    source_root: Path,
    mc_samples: int = 128,
    maximum_optimizer_batch_size: int = 2048,
    timeout_s: float = 900.0,
) -> IsolatedBotorchQLogNehviFiniteAcquisition:
    """Build the isolated expert with a deny-by-default child environment."""

    resolved_source = source_root.resolve(strict=True)
    process_policy = ChildProcessPolicy(
        policy_id="isolated_botorch_qlognehvi_worker",
        policy_version=1,
        inherited_environment_allowlist=(),
        fixed_environment=(
            ("LANG", "C.UTF-8"),
            ("PYTHONHASHSEED", "0"),
            ("PYTHONPATH", str(resolved_source)),
        ),
    )
    boundary = ExplicitEnvironmentSubprocessBoundary(
        policy=process_policy,
        working_directory=resolved_source,
        source_environment={},
    )
    return IsolatedBotorchQLogNehviFiniteAcquisition(
        boundary=boundary,
        python_executable=python_executable,
        source_root=resolved_source,
        mc_samples=mc_samples,
        maximum_optimizer_batch_size=maximum_optimizer_batch_size,
        timeout_s=timeout_s,
    )


__all__ = [
    "FiniteAcquisitionSubprocessError",
    "IsolatedBotorchQLogNehviFiniteAcquisition",
    "POLICY_ID",
    "POLICY_VERSION",
    "build_isolated_botorch_qlognehvi",
]
