"""Dependency-isolated qLogNEHVI scorer for fixed finite slates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import tempfile

from agent_evolve.infrastructure.subprocess_boundary import (
    ExplicitEnvironmentSubprocessBoundary,
)
from agent_evolve.integrations.botorch.finite_qlognehvi_batch_identity import (
    POLICY_DEFINITION_SHA256 as UNDERLYING_DEFINITION_SHA256,
    POLICY_ID as UNDERLYING_POLICY_ID,
    POLICY_VERSION as UNDERLYING_POLICY_VERSION,
)
from agent_evolve.ports.finite_acquisition_batch import (
    FiniteAcquisitionBatchScoreDecision,
    FiniteAcquisitionBatchScorePolicy,
    FiniteAcquisitionBatchScoreRequest,
    validate_finite_acquisition_batch_score_decision,
)
from agent_evolve.ports.finite_acquisition_batch_json import (
    finite_acquisition_batch_score_decision_from_record,
)
from agent_evolve.ports.subprocess_boundary import (
    BoundedSubprocessBoundary,
    ChildProcessPolicy,
)


POLICY_ID = "isolated_botorch_finite_qlognehvi_batch_score"
POLICY_VERSION = 1
WORKER_MODULE = "agent_evolve.integrations.botorch.finite_qlognehvi_batch_worker"
_DEFINITION_DOMAIN = b"agent-evolve:isolated-botorch-qlognehvi-batch-score:v1\x00"


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


class FiniteAcquisitionBatchScoreSubprocessError(RuntimeError):
    """The isolated scorer failed before producing an authenticated decision."""


@dataclass(frozen=True, slots=True)
class IsolatedBotorchQLogNehviFiniteBatchScorePolicy:
    """Score sealed slates in the pinned optional-dependency environment."""

    boundary: BoundedSubprocessBoundary
    python_executable: Path
    source_root: Path
    mc_samples: int = 128
    maximum_score_batch_size: int = 512
    timeout_s: float = 900.0
    policy_id: str = field(init=False, default=POLICY_ID)
    policy_version: int = field(init=False, default=POLICY_VERSION)
    definition_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, BoundedSubprocessBoundary):
            raise TypeError("boundary must implement BoundedSubprocessBoundary")
        if not isinstance(self.python_executable, Path) or not (
            self.python_executable.is_absolute()
        ):
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
            type(self.maximum_score_batch_size) is not int
            or self.maximum_score_batch_size < 1
        ):
            raise ValueError("maximum_score_batch_size must be positive")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not math.isfinite(float(self.timeout_s))
            or float(self.timeout_s) <= 0.0
        ):
            raise ValueError("timeout_s must be finite and positive")
        package = Path(__file__).resolve().parent
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
            "worker_source_sha256": _file_sha256(
                package / "finite_qlognehvi_batch_worker.py"
            ),
            "direct_policy_source_sha256": _file_sha256(
                package / "finite_qlognehvi_batch.py"
            ),
            "python_executable_sha256": _file_sha256(executable),
            "python_invocation": self.boundary.invocation_observation(
                str(self.python_executable)
            ),
            "process_boundary_identity_sha256": self.boundary.identity_sha256,
            "mc_samples": self.mc_samples,
            "maximum_score_batch_size": self.maximum_score_batch_size,
            "timeout_s_hex": float(self.timeout_s).hex(),
            "transport": "authenticated-json-files-exact-argv-no-shell",
            "common_fit_and_sampler_per_request": True,
            "workload_model_provider_branches": False,
        }
        object.__setattr__(
            self,
            "definition_sha256",
            hashlib.sha256(_DEFINITION_DOMAIN + _canonical_json(record)).hexdigest(),
        )

    def score(
        self,
        request: FiniteAcquisitionBatchScoreRequest,
    ) -> FiniteAcquisitionBatchScoreDecision:
        self.__post_init__()
        if type(request) is not FiniteAcquisitionBatchScoreRequest:
            raise TypeError("request must be an exact batch-score request")
        request.__post_init__()
        with tempfile.TemporaryDirectory(
            prefix="agent-evolve-qlognehvi-batch-score-"
        ) as directory:
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
                    "--maximum-score-batch-size",
                    str(self.maximum_score_batch_size),
                ),
                timeout_s=float(self.timeout_s),
            )
            if result.returncode != 0:
                raise FiniteAcquisitionBatchScoreSubprocessError(
                    "isolated qLogNEHVI batch scorer exited "
                    f"{result.returncode}: {result.stderr[-8000:]}"
                )
            if not response_path.is_file():
                raise FiniteAcquisitionBatchScoreSubprocessError(
                    "isolated qLogNEHVI batch scorer produced no response"
                )
            payload = json.loads(response_path.read_text(encoding="ascii"))
        if (
            type(payload) is not dict
            or set(payload) != {"schema_version", "decision"}
            or payload["schema_version"] != 1
        ):
            raise ValueError("isolated qLogNEHVI scorer returned a foreign schema")
        underlying = finite_acquisition_batch_score_decision_from_record(
            payload["decision"]
        )
        validate_finite_acquisition_batch_score_decision(request, underlying)
        if (
            underlying.policy_id,
            underlying.policy_version,
            underlying.policy_definition_sha256,
        ) != (
            UNDERLYING_POLICY_ID,
            UNDERLYING_POLICY_VERSION,
            UNDERLYING_DEFINITION_SHA256,
        ):
            raise ValueError("isolated qLogNEHVI scorer returned a foreign identity")
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
        return FiniteAcquisitionBatchScoreDecision(
            request_sha256=request.request_sha256,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            scores=underlying.scores,
            diagnostics=tuple(sorted(diagnostics.items())),
        )


def build_isolated_botorch_qlognehvi_batch_score(
    *,
    python_executable: Path,
    source_root: Path,
    mc_samples: int = 128,
    maximum_score_batch_size: int = 512,
    timeout_s: float = 900.0,
) -> IsolatedBotorchQLogNehviFiniteBatchScorePolicy:
    """Build the scorer with a deny-by-default child environment."""

    resolved_source = source_root.resolve(strict=True)
    process_policy = ChildProcessPolicy(
        policy_id="isolated_botorch_qlognehvi_batch_score_worker",
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
    policy = IsolatedBotorchQLogNehviFiniteBatchScorePolicy(
        boundary=boundary,
        python_executable=python_executable,
        source_root=resolved_source,
        mc_samples=mc_samples,
        maximum_score_batch_size=maximum_score_batch_size,
        timeout_s=timeout_s,
    )
    if not isinstance(policy, FiniteAcquisitionBatchScorePolicy):
        raise AssertionError("isolated batch scorer does not satisfy its port")
    return policy


__all__ = [
    "FiniteAcquisitionBatchScoreSubprocessError",
    "IsolatedBotorchQLogNehviFiniteBatchScorePolicy",
    "POLICY_ID",
    "POLICY_VERSION",
    "build_isolated_botorch_qlognehvi_batch_score",
]
