"""Non-circular offline qualification for Airfoil v10 live runs.

The qualification directory is an external, finalized input to a live run.  It
binds an exact provider-free pytest invocation to the same runtime source
closure used by the live manifest.  The source closure is captured before and
after pytest and must remain byte-identical.  Qualification output is not part
of that closure, so the receipt never needs to contain its own file identity.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from agent_evolve.application.live_runtime_manifest import RuntimeSourceClosure
from agent_evolve.ports.artifact_store import canonical_json_bytes, decode_json_bytes
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    AirfoilG3ProviderProfile,
)
from examples.development.durable_run_artifacts import (
    file_identity,
    finalize_run_directory,
    verify_finalized_run_directory,
    write_bytes_atomic,
    write_json_atomic,
)


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[3]
AIRFOIL_V10_QUALIFICATION_ID = "airfoil_v10_runtime_offline_qualification"
AIRFOIL_V10_QUALIFICATION_VERSION = 2
AIRFOIL_V10_QUALIFICATION_STATUS = "provider_free_qualification_passed"
AIRFOIL_V10_QUALIFICATION_RECEIPT_FILENAME = "qualification_receipt.json"
AIRFOIL_V10_QUALIFICATION_JUNIT_FILENAME = "focused_tests.junit.xml"
# Helper and fixture files used by these modules are separately bound by the
# manifest's verification_tests source role.  The number of collected cases is
# deliberately observed from JUnit rather than frozen in code: adding a test
# changes the source closure and therefore requires a fresh receipt, but it
# does not require a second count constant to be updated in lockstep.
AIRFOIL_V10_QUALIFICATION_TEST_PATHS = (
    "tests/test_agentic_evolution_engine.py",
    "tests/test_agentic_pipeline_v2_replay.py",
    "tests/test_airfoil_v10_exact_stack_conformance.py",
    "tests/test_airfoil_v10_multi_option_inputs.py",
    "tests/test_airfoil_v10_multi_option_live.py",
    "tests/test_airfoil_v10_multi_option_runner.py",
    "tests/test_airfoil_v10_runtime_manifest.py",
    "tests/test_async_structured_generator.py",
    "tests/test_budgeted_optimizer.py",
    "tests/test_exact_parent_crossover.py",
    "tests/test_matched_finite_action_block.py",
    "tests/test_multi_option_evolution.py",
    "tests/test_multi_option_model_crossover.py",
    "tests/test_openrouter_outbound_request_manifest.py",
    "tests/test_post_evolution_reflection.py",
    "tests/test_progress_aware_openrouter.py",
    "tests/test_prompt_shape_commitment.py",
    "tests/test_provider_attempt_join.py",
    "tests/test_pydantic_agentic_generator.py",
    "tests/test_queued_structured_runner.py",
    "tests/test_reflective_feedback.py",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC_SECONDS = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_RECEIPT_DOMAIN = b"agent-evolve:airfoil-v10-offline-qualification:v1\x00"
_PROVIDER_CONFIGURATION_DOMAIN = (
    b"agent-evolve:airfoil-v10-provider-configuration:v1\x00"
)
_MAX_JUNIT_BYTES = 10_000_000
AIRFOIL_V10_QUALIFICATION_DISTRIBUTIONS = (
    "httpx",
    "openai",
    "pydantic",
    "pydantic-ai",
    "pytest",
)


class AirfoilV10QualificationError(RuntimeError):
    """The exact offline qualification is absent, failed, or stale."""


AirfoilV10QualificationSourceFactory = Callable[
    [AirfoilG3ProviderProfile], RuntimeSourceClosure
]


def _utc_seconds() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_sha256(value: str, *, name: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_utc_seconds(value: str, *, name: str) -> None:
    if type(value) is not str or _UTC_SECONDS.fullmatch(value) is None:
        raise ValueError(f"{name} must be UTC RFC3339 at whole seconds")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(f"{name} is not a real UTC instant") from exc


def _qualification_sha256(value: object) -> str:
    return hashlib.sha256(_RECEIPT_DOMAIN + canonical_json_bytes(value)).hexdigest()


def _exact_json_equal(left: object, right: object) -> bool:
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


def airfoil_v10_provider_configuration_sha256(
    provider_record: Mapping[str, object],
) -> str:
    """Commit the complete composition-owned provider configuration."""

    if type(provider_record) is not dict:
        raise TypeError("provider_record must be an exact dictionary")
    try:
        canonical = canonical_json_bytes(provider_record)
        decoded = decode_json_bytes(canonical)
    except (TypeError, ValueError) as exc:
        raise ValueError("provider_record must contain exact JSON values") from exc
    if type(decoded) is not dict or not _exact_json_equal(decoded, provider_record):
        raise ValueError("provider_record is not an exact canonical JSON object")
    return hashlib.sha256(_PROVIDER_CONFIGURATION_DOMAIN + canonical).hexdigest()


def _installed_distribution_versions() -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = []
    for name in AIRFOIL_V10_QUALIFICATION_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise AirfoilV10QualificationError(
                f"qualification runtime distribution is unavailable: {name}"
            ) from exc
        if type(version) is not str or not version or version != version.strip():
            raise AirfoilV10QualificationError(
                f"qualification runtime distribution version is invalid: {name}"
            )
        values.append((name, version))
    return tuple(values)


def _junit_counts(payload: bytes) -> dict[str, int]:
    if type(payload) is not bytes or not payload or len(payload) > _MAX_JUNIT_BYTES:
        raise AirfoilV10QualificationError("qualification JUnit size is invalid")
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise AirfoilV10QualificationError(
            "qualification JUnit is invalid XML"
        ) from exc

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    if local_name(root.tag) == "testsuite":
        suites = (root,)
    elif local_name(root.tag) == "testsuites":
        suites = tuple(child for child in root if local_name(child.tag) == "testsuite")
    else:
        suites = ()
    if not suites:
        raise AirfoilV10QualificationError("qualification JUnit contains no test suite")
    counts = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for name in counts:
            value = suite.get(name)
            if type(value) is not str or not value.isascii() or not value.isdecimal():
                raise AirfoilV10QualificationError(
                    "qualification JUnit counts are invalid"
                )
            counts[name] += int(value)
    testcase_count = sum(
        1
        for suite in suites
        for element in suite.iter()
        if local_name(element.tag) == "testcase"
    )
    if testcase_count != counts["tests"]:
        raise AirfoilV10QualificationError(
            "qualification JUnit testcase count is inconsistent"
        )
    return counts


@dataclass(frozen=True, slots=True)
class AirfoilV10QualificationReceipt:
    """Content-authenticated result of one exact provider-free pytest suite."""

    source_sha256: str
    provider_profile_id: str
    provider_configuration_sha256: str
    python_executable: str
    python_version: str
    installed_distributions: tuple[tuple[str, str], ...]
    started_at_utc: str
    finished_at_utc: str
    tests: int
    failures: int
    errors: int
    skipped: int
    junit_size_bytes: int
    junit_sha256: str
    stdout_size_bytes: int
    stdout_sha256: str
    stderr_size_bytes: int
    stderr_sha256: str
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_sha256",
            "provider_configuration_sha256",
            "junit_sha256",
            "stdout_sha256",
            "stderr_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if (
            type(self.provider_profile_id) is not str
            or not self.provider_profile_id
            or self.provider_profile_id != self.provider_profile_id.strip()
        ):
            raise ValueError("provider_profile_id must be canonical non-empty text")
        executable = Path(self.python_executable)
        if (
            type(self.python_executable) is not str
            or not executable.is_absolute()
            or self.python_executable != str(executable)
        ):
            raise ValueError("python_executable must be an absolute normalized path")
        for name in ("python_version",):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be canonical non-empty text")
        if (
            type(self.installed_distributions) is not tuple
            or tuple(name for name, _ in self.installed_distributions)
            != AIRFOIL_V10_QUALIFICATION_DISTRIBUTIONS
        ):
            raise ValueError(
                "installed_distributions must bind the exact qualification stack"
            )
        for item in self.installed_distributions:
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not str
                or type(item[1]) is not str
                or not item[1]
                or item[1] != item[1].strip()
            ):
                raise ValueError("installed distribution versions must be canonical")
        _require_utc_seconds(self.started_at_utc, name="started_at_utc")
        _require_utc_seconds(self.finished_at_utc, name="finished_at_utc")
        for name in (
            "tests",
            "failures",
            "errors",
            "skipped",
            "junit_size_bytes",
            "stdout_size_bytes",
            "stderr_size_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if (
            self.tests < 1
            or self.failures != 0
            or self.errors != 0
            or self.skipped != 0
            or self.junit_size_bytes < 1
        ):
            raise ValueError("qualification outcome is not the exact all-pass suite")
        object.__setattr__(
            self,
            "receipt_sha256",
            _qualification_sha256(self._identity_record()),
        )

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": AIRFOIL_V10_QUALIFICATION_VERSION,
            "qualification_id": AIRFOIL_V10_QUALIFICATION_ID,
            "source_sha256": self.source_sha256,
            "provider_profile_id": self.provider_profile_id,
            "provider_configuration_sha256": self.provider_configuration_sha256,
            "runner": {
                "python_executable": self.python_executable,
                "python_version": self.python_version,
                "module_invocation": "python_-m_pytest",
                "cache_provider_disabled": True,
            },
            "suite": {
                "test_paths": list(AIRFOIL_V10_QUALIFICATION_TEST_PATHS),
                "test_count_source": "parsed_junit_testcases",
                "junit_enabled": True,
            },
            "outcome": {
                "exit_code": 0,
                "tests": self.tests,
                "failures": self.failures,
                "errors": self.errors,
                "skipped": self.skipped,
            },
            "artifacts": {
                "junit": {
                    "path": AIRFOIL_V10_QUALIFICATION_JUNIT_FILENAME,
                    "size_bytes": self.junit_size_bytes,
                    "sha256": self.junit_sha256,
                },
                "stdout": {
                    "retained": False,
                    "size_bytes": self.stdout_size_bytes,
                    "sha256": self.stdout_sha256,
                },
                "stderr": {
                    "retained": False,
                    "size_bytes": self.stderr_size_bytes,
                    "sha256": self.stderr_sha256,
                },
            },
            "environment": {
                "openrouter_api_key_removed": True,
                "python_dont_write_bytecode": True,
                "pytest_plugin_autoload_disabled": True,
                "installed_distributions": dict(self.installed_distributions),
            },
            "authorization": {
                "provider_calls_authorized": 0,
                "physical_evaluator_calls_authorized": 0,
                "scientific_result_eligible": False,
            },
            "source_stable_during_execution": True,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._identity_record(), "receipt_sha256": self.receipt_sha256}

    @classmethod
    def from_record(cls, value: object) -> "AirfoilV10QualificationReceipt":
        if type(value) is not dict:
            raise AirfoilV10QualificationError(
                "qualification receipt root must be an exact object"
            )
        record = dict(value)
        claimed = record.pop("receipt_sha256", None)
        if (
            type(claimed) is not str
            or _SHA256.fullmatch(claimed) is None
            or claimed != _qualification_sha256(record)
        ):
            raise AirfoilV10QualificationError(
                "qualification receipt commitment is invalid"
            )
        if set(record) != {
            "schema_version",
            "qualification_id",
            "source_sha256",
            "provider_profile_id",
            "provider_configuration_sha256",
            "runner",
            "suite",
            "outcome",
            "artifacts",
            "environment",
            "authorization",
            "source_stable_during_execution",
            "started_at_utc",
            "finished_at_utc",
        }:
            raise AirfoilV10QualificationError(
                "qualification receipt root violates its closed schema"
            )
        runner = record.get("runner")
        suite = record.get("suite")
        outcome = record.get("outcome")
        artifacts = record.get("artifacts")
        environment = record.get("environment")
        authorization = record.get("authorization")
        if (
            type(record.get("schema_version")) is not int
            or record.get("schema_version") != AIRFOIL_V10_QUALIFICATION_VERSION
            or record.get("qualification_id") != AIRFOIL_V10_QUALIFICATION_ID
            or type(runner) is not dict
            or set(runner)
            != {
                "python_executable",
                "python_version",
                "module_invocation",
                "cache_provider_disabled",
            }
            or runner.get("module_invocation") != "python_-m_pytest"
            or runner.get("cache_provider_disabled") is not True
            or type(suite) is not dict
            or not _exact_json_equal(
                suite,
                {
                    "test_paths": list(AIRFOIL_V10_QUALIFICATION_TEST_PATHS),
                    "test_count_source": "parsed_junit_testcases",
                    "junit_enabled": True,
                },
            )
            or type(outcome) is not dict
            or set(outcome) != {"exit_code", "tests", "failures", "errors", "skipped"}
            or type(outcome.get("exit_code")) is not int
            or outcome.get("exit_code") != 0
            or type(artifacts) is not dict
            or set(artifacts) != {"junit", "stdout", "stderr"}
            or type(environment) is not dict
            or set(environment)
            != {
                "openrouter_api_key_removed",
                "python_dont_write_bytecode",
                "pytest_plugin_autoload_disabled",
                "installed_distributions",
            }
            or environment.get("openrouter_api_key_removed") is not True
            or environment.get("python_dont_write_bytecode") is not True
            or environment.get("pytest_plugin_autoload_disabled") is not True
            or type(environment.get("installed_distributions")) is not dict
            or tuple(environment["installed_distributions"])
            != AIRFOIL_V10_QUALIFICATION_DISTRIBUTIONS
            or not _exact_json_equal(
                authorization,
                {
                    "provider_calls_authorized": 0,
                    "physical_evaluator_calls_authorized": 0,
                    "scientific_result_eligible": False,
                },
            )
            or record.get("source_stable_during_execution") is not True
        ):
            raise AirfoilV10QualificationError(
                "qualification receipt contract is invalid"
            )
        junit = artifacts.get("junit")
        stdout = artifacts.get("stdout")
        stderr = artifacts.get("stderr")
        if (
            type(junit) is not dict
            or set(junit) != {"path", "size_bytes", "sha256"}
            or junit.get("path") != AIRFOIL_V10_QUALIFICATION_JUNIT_FILENAME
            or type(stdout) is not dict
            or set(stdout) != {"retained", "size_bytes", "sha256"}
            or stdout.get("retained") is not False
            or type(stderr) is not dict
            or set(stderr) != {"retained", "size_bytes", "sha256"}
            or stderr.get("retained") is not False
        ):
            raise AirfoilV10QualificationError(
                "qualification artifact contract is invalid"
            )
        try:
            receipt = cls(
                source_sha256=record["source_sha256"],
                provider_profile_id=record["provider_profile_id"],
                provider_configuration_sha256=record["provider_configuration_sha256"],
                python_executable=runner["python_executable"],
                python_version=runner["python_version"],
                installed_distributions=tuple(
                    (name, environment["installed_distributions"][name])
                    for name in AIRFOIL_V10_QUALIFICATION_DISTRIBUTIONS
                ),
                started_at_utc=record["started_at_utc"],
                finished_at_utc=record["finished_at_utc"],
                tests=outcome["tests"],
                failures=outcome["failures"],
                errors=outcome["errors"],
                skipped=outcome["skipped"],
                junit_size_bytes=junit["size_bytes"],
                junit_sha256=junit["sha256"],
                stdout_size_bytes=stdout["size_bytes"],
                stdout_sha256=stdout["sha256"],
                stderr_size_bytes=stderr["size_bytes"],
                stderr_sha256=stderr["sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AirfoilV10QualificationError(
                "qualification receipt values are invalid"
            ) from exc
        if receipt.receipt_sha256 != claimed or not _exact_json_equal(
            receipt.to_record(), value
        ):
            raise AirfoilV10QualificationError("qualification receipt is not canonical")
        return receipt


@dataclass(frozen=True, slots=True)
class VerifiedAirfoilV10Qualification:
    """Verified external qualification directory admitted to a live manifest."""

    directory: Path
    receipt: AirfoilV10QualificationReceipt
    finalization_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.directory, Path) or not self.directory.is_absolute():
            raise ValueError("qualification directory must be an absolute Path")
        if type(self.receipt) is not AirfoilV10QualificationReceipt:
            raise TypeError("receipt must be an exact qualification receipt")
        self.receipt.__post_init__()
        _require_sha256(self.finalization_sha256, name="finalization_sha256")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "directory": str(self.directory),
            "receipt_sha256": self.receipt.receipt_sha256,
            "source_sha256": self.receipt.source_sha256,
            "provider_profile_id": self.receipt.provider_profile_id,
            "provider_configuration_sha256": (
                self.receipt.provider_configuration_sha256
            ),
            "installed_distributions": dict(self.receipt.installed_distributions),
            "test_count": self.receipt.tests,
            "junit_sha256": self.receipt.junit_sha256,
            "finalization_sha256": self.finalization_sha256,
            "non_circular_external_receipt": True,
        }


def verify_airfoil_v10_qualification_directory(
    directory: Path,
    *,
    provider_profile: AirfoilG3ProviderProfile,
    provider_record: Mapping[str, object],
    source_closure_factory: AirfoilV10QualificationSourceFactory,
) -> VerifiedAirfoilV10Qualification:
    """Authenticate a finalized receipt against current source and route bytes."""

    if type(provider_profile) is not AirfoilG3ProviderProfile:
        raise TypeError("provider_profile must be exact")
    provider_profile.__post_init__()
    if not callable(source_closure_factory):
        raise TypeError("source_closure_factory must be callable")
    root = directory.expanduser().resolve(strict=True)
    try:
        finalization = verify_finalized_run_directory(root)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise AirfoilV10QualificationError(
            "qualification directory is not durably finalized"
        ) from exc
    if finalization.get("status") != AIRFOIL_V10_QUALIFICATION_STATUS or set(
        finalization.get("files", {})
    ) != {
        AIRFOIL_V10_QUALIFICATION_JUNIT_FILENAME,
        AIRFOIL_V10_QUALIFICATION_RECEIPT_FILENAME,
    }:
        raise AirfoilV10QualificationError(
            "qualification directory membership or status is invalid"
        )
    try:
        receipt_value = decode_json_bytes(
            (root / AIRFOIL_V10_QUALIFICATION_RECEIPT_FILENAME).read_bytes()
        )
        receipt = AirfoilV10QualificationReceipt.from_record(receipt_value)
    except AirfoilV10QualificationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise AirfoilV10QualificationError(
            "qualification receipt is unreadable"
        ) from exc
    source = source_closure_factory(provider_profile)
    if type(source) is not RuntimeSourceClosure:
        raise TypeError("source_closure_factory returned a foreign value")
    source.__post_init__()
    provider_configuration_sha256 = airfoil_v10_provider_configuration_sha256(
        provider_record
    )
    installed_distributions = _installed_distribution_versions()
    if (
        receipt.source_sha256 != source.source_sha256
        or receipt.provider_profile_id != provider_profile.profile_id
        or receipt.provider_configuration_sha256 != provider_configuration_sha256
        or receipt.python_executable != str(Path(sys.executable).absolute())
        or receipt.python_version != platform.python_version()
        or receipt.installed_distributions != installed_distributions
    ):
        raise AirfoilV10QualificationError(
            "qualification receipt belongs to different source, route, or runtime"
        )
    junit_path = root / AIRFOIL_V10_QUALIFICATION_JUNIT_FILENAME
    junit = file_identity(junit_path, relative_to=root)
    if junit != {
        "path": AIRFOIL_V10_QUALIFICATION_JUNIT_FILENAME,
        "size_bytes": receipt.junit_size_bytes,
        "sha256": receipt.junit_sha256,
    } or _junit_counts(junit_path.read_bytes()) != {
        "tests": receipt.tests,
        "failures": receipt.failures,
        "errors": receipt.errors,
        "skipped": receipt.skipped,
    }:
        raise AirfoilV10QualificationError(
            "qualification JUnit differs from its receipt"
        )
    finalization_sha256 = finalization.get("finalization_sha256")
    if type(finalization_sha256) is not str:
        raise AirfoilV10QualificationError(
            "qualification finalization identity is absent"
        )
    return VerifiedAirfoilV10Qualification(
        directory=root,
        receipt=receipt,
        finalization_sha256=finalization_sha256,
    )


def record_airfoil_v10_qualification(
    output_dir: Path,
    *,
    provider_profile: AirfoilG3ProviderProfile,
    provider_record: Mapping[str, object],
    source_closure_factory: AirfoilV10QualificationSourceFactory,
    timeout_seconds: int = 900,
) -> VerifiedAirfoilV10Qualification:
    """Run and durably seal the exact credential-stripped selected suite."""

    if type(provider_profile) is not AirfoilG3ProviderProfile:
        raise TypeError("provider_profile must be exact")
    provider_profile.__post_init__()
    if not callable(source_closure_factory):
        raise TypeError("source_closure_factory must be callable")
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError("timeout_seconds must be a positive exact integer")
    root = output_dir.expanduser().resolve(strict=False)
    if root.exists():
        raise FileExistsError(root)
    source_before = source_closure_factory(provider_profile)
    if type(source_before) is not RuntimeSourceClosure:
        raise TypeError("source_closure_factory returned a foreign value")
    source_before.__post_init__()
    provider_configuration_sha256 = airfoil_v10_provider_configuration_sha256(
        provider_record
    )
    distributions_before = _installed_distribution_versions()
    environment = dict(os.environ)
    environment.pop("OPENROUTER_API_KEY", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    started_at = _utc_seconds()
    with tempfile.TemporaryDirectory(prefix="airfoil_v10_qualification_") as temp:
        junit_temporary = Path(temp) / AIRFOIL_V10_QUALIFICATION_JUNIT_FILENAME
        command = (
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            *AIRFOIL_V10_QUALIFICATION_TEST_PATHS,
            "--junitxml",
            str(junit_temporary),
        )
        completed = subprocess.run(
            command,
            cwd=AGENT_EVOLVE_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
        finished_at = _utc_seconds()
        source_after = source_closure_factory(provider_profile)
        if type(source_after) is not RuntimeSourceClosure:
            raise TypeError("source_closure_factory returned a foreign value")
        source_after.__post_init__()
        if source_after.to_record() != source_before.to_record():
            raise AirfoilV10QualificationError(
                "runtime source closure changed during qualification"
            )
        if _installed_distribution_versions() != distributions_before:
            raise AirfoilV10QualificationError(
                "installed qualification runtime changed during execution"
            )
        if completed.returncode != 0 or not junit_temporary.is_file():
            raise AirfoilV10QualificationError(
                "exact provider-free qualification suite did not pass"
            )
        junit_payload = junit_temporary.read_bytes()
        counts = _junit_counts(junit_payload)
        if (
            counts["tests"] < 1
            or counts["failures"] != 0
            or counts["errors"] != 0
            or counts["skipped"] != 0
        ):
            raise AirfoilV10QualificationError(
                "qualification suite count or all-pass status drifted"
            )

    root.mkdir(parents=True, exist_ok=False)
    junit_path = root / AIRFOIL_V10_QUALIFICATION_JUNIT_FILENAME
    write_bytes_atomic(junit_path, junit_payload)
    receipt = AirfoilV10QualificationReceipt(
        source_sha256=source_before.source_sha256,
        provider_profile_id=provider_profile.profile_id,
        provider_configuration_sha256=provider_configuration_sha256,
        python_executable=str(Path(sys.executable).absolute()),
        python_version=platform.python_version(),
        installed_distributions=distributions_before,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        tests=counts["tests"],
        failures=counts["failures"],
        errors=counts["errors"],
        skipped=counts["skipped"],
        junit_size_bytes=len(junit_payload),
        junit_sha256=hashlib.sha256(junit_payload).hexdigest(),
        stdout_size_bytes=len(completed.stdout),
        stdout_sha256=hashlib.sha256(completed.stdout).hexdigest(),
        stderr_size_bytes=len(completed.stderr),
        stderr_sha256=hashlib.sha256(completed.stderr).hexdigest(),
    )
    write_json_atomic(
        root / AIRFOIL_V10_QUALIFICATION_RECEIPT_FILENAME,
        receipt.to_record(),
    )
    finalize_run_directory(root, status=AIRFOIL_V10_QUALIFICATION_STATUS)
    return verify_airfoil_v10_qualification_directory(
        root,
        provider_profile=provider_profile,
        provider_record=provider_record,
        source_closure_factory=source_closure_factory,
    )


__all__ = [
    "AIRFOIL_V10_QUALIFICATION_ID",
    "AIRFOIL_V10_QUALIFICATION_DISTRIBUTIONS",
    "AIRFOIL_V10_QUALIFICATION_JUNIT_FILENAME",
    "AIRFOIL_V10_QUALIFICATION_RECEIPT_FILENAME",
    "AIRFOIL_V10_QUALIFICATION_STATUS",
    "AIRFOIL_V10_QUALIFICATION_TEST_PATHS",
    "AIRFOIL_V10_QUALIFICATION_VERSION",
    "AirfoilV10QualificationError",
    "AirfoilV10QualificationReceipt",
    "AirfoilV10QualificationSourceFactory",
    "VerifiedAirfoilV10Qualification",
    "airfoil_v10_provider_configuration_sha256",
    "record_airfoil_v10_qualification",
    "verify_airfoil_v10_qualification_directory",
]
