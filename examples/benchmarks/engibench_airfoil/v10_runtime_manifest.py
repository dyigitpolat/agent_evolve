"""Prospective full-stack provenance for Airfoil v10 paid evolution runs.

The frozen v10 scientific inputs intentionally retain their G3 release roots;
those roots do not identify the executable AgentEvolve method.  This module
therefore binds a conservative source closure, selected provider route,
runtime environment, evaluator wiring, and exact v10 experiment identity in a
self-authenticating manifest.  The gate reconstructs the manifest from live
bytes and fails closed on any drift.  Construction and verification read no
credential, contact no provider, and execute no physical evaluator.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_evolve.application.live_runtime_manifest import (
    LiveRuntimeManifest,
    RuntimeManifestSection,
    RuntimeSourceClosure,
    build_live_runtime_manifest,
    capture_git_worktree_section,
    capture_runtime_environment_section,
    capture_runtime_file,
    capture_runtime_source_closure,
    load_live_runtime_manifest,
    verify_runtime_source_closure,
)
from agent_evolve.policies.memory.prompt_shape import (
    DefaultEvidencePromptShapePolicyV3,
)
from agent_evolve.ports.agentic_generator import (
    CANDIDATE_COMPONENT_PATH_CONTRACT,
    TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT,
)
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    local_default_converged_settings,
)
from examples.benchmarks.engibench_airfoil.problem_def import (
    EXPECTED_DATASET_SHA256,
)
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    CONTAINER_IMAGE,
    EVALUATOR_CONCURRENCY,
    AirfoilG3ProviderProfile,
    capture_airfoil_g3_source_closure,
)
from examples.benchmarks.engibench_airfoil.v10_multi_option_inputs import (
    AirfoilV10MultiOptionInputs,
)
from examples.benchmarks.engibench_airfoil.v10_qualification import (
    AIRFOIL_V10_QUALIFICATION_TEST_PATHS,
    VerifiedAirfoilV10Qualification,
    airfoil_v10_provider_configuration_sha256,
    verify_airfoil_v10_qualification_directory,
)


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[3]
AIRFOIL_V10_RUNTIME_MANIFEST_ID = "airfoil_v10_multi_option_runtime"
AIRFOIL_V10_RUNTIME_MANIFEST_VERSION = 3
AIRFOIL_V10_RUNTIME_MANIFEST_FILENAME = "runtime_manifest.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_VERIFICATION_DOMAIN = b"agent-evolve:airfoil-v10-runtime-verification:v1\x00"
_CONTRACT_DOMAIN = b"agent-evolve:airfoil-v10-runtime-contract:v1\x00"

_VERIFICATION_TEST_PATHS = (
    "tests/_canned.py",
    "tests/conftest.py",
    "tests/fakes.py",
    "tests/fixtures/agentic_pipeline_v2_replay.json",
    *AIRFOIL_V10_QUALIFICATION_TEST_PATHS,
)

_EXACT_STACK_PROBE_PATHS = (
    "examples/development/run_airfoil_v10_exact_stack_conformance.py",
)

_REQUIRED_SOURCE_PATHS = (
    "examples/benchmarks/engibench_airfoil/v10_multi_option_inputs.py",
    "examples/benchmarks/engibench_airfoil/v10_multi_option_live.py",
    "examples/benchmarks/engibench_airfoil/v10_multi_option_runner.py",
    "examples/benchmarks/engibench_airfoil/v10_qualification.py",
    "examples/benchmarks/engibench_airfoil/v10_runtime_manifest.py",
    "examples/development/durable_run_artifacts.py",
    *_EXACT_STACK_PROBE_PATHS,
    "src/agent_evolve/application/agentic_evolution.py",
    "src/agent_evolve/application/live_runtime_manifest.py",
    "src/agent_evolve/application/materialized_variation.py",
    "src/agent_evolve/application/multi_option_evolution.py",
    "src/agent_evolve/application/post_evolution_reflection.py",
    "src/agent_evolve/integrations/pydantic_ai/agentic_generator.py",
    "src/agent_evolve/integrations/pydantic_ai/async_generator.py",
    "src/agent_evolve/integrations/pydantic_ai/outbound_request_manifest.py",
    "src/agent_evolve/integrations/pydantic_ai/progress_aware_openrouter.py",
    "src/agent_evolve/integrations/pydantic_ai/provider_attempt_join.py",
    "src/agent_evolve/integrations/pydantic_ai/queued_runner.py",
    "src/agent_evolve/policies/memory/prompt_shape.py",
    "src/agent_evolve/policies/variation/exact_parent_crossover.py",
    "src/agent_evolve/ports/agentic_generator.py",
    *_VERIFICATION_TEST_PATHS,
)

_REQUIRED_SOURCE_ROLES = (
    "artifact_journal",
    "benchmark_runtime",
    "dependency_lock",
    "evaluator_runtime",
    "exact_stack_probe",
    "generic_core",
    "provider_runtime",
    "route_snapshot",
    "verification_tests",
)


class AirfoilV10RuntimeManifestError(RuntimeError):
    """The prospective v10 executable closure is missing or changed."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _contract_sha256(value: object) -> str:
    return hashlib.sha256(_CONTRACT_DOMAIN + _canonical_bytes(value)).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _files_for_roles(
    source: RuntimeSourceClosure,
) -> dict[str, dict[str, Path]]:
    """Reproject a validated closure to the generic capture API."""

    source.__post_init__()
    by_path = {value.logical_path: Path(value.resolved_path) for value in source.files}
    return {
        role: {logical_path: by_path[logical_path] for logical_path in paths}
        for role, paths in source.role_paths
    }


def capture_airfoil_v10_runtime_source_closure(
    provider_profile: AirfoilG3ProviderProfile,
) -> RuntimeSourceClosure:
    """Capture v7's conservative live closure plus v10 verification sources."""

    if type(provider_profile) is not AirfoilG3ProviderProfile:
        raise TypeError("provider_profile must be exact")
    provider_profile.__post_init__()
    files_by_role = _files_for_roles(
        capture_airfoil_g3_source_closure(provider_profile)
    )
    files_by_role["verification_tests"] = {
        relative: AGENT_EVOLVE_ROOT / relative for relative in _VERIFICATION_TEST_PATHS
    }
    files_by_role["exact_stack_probe"] = {
        relative: AGENT_EVOLVE_ROOT / relative for relative in _EXACT_STACK_PROBE_PATHS
    }
    source = capture_runtime_source_closure(files_by_role)
    roles = {role for role, _ in source.role_paths}
    missing_roles = set(_REQUIRED_SOURCE_ROLES) - roles
    paths = {value.logical_path for value in source.files}
    missing_paths = set(_REQUIRED_SOURCE_PATHS) - paths
    if missing_roles or missing_paths:
        raise AirfoilV10RuntimeManifestError("v10 runtime source closure is incomplete")
    return source


AirfoilV10SourceClosureFactory = Callable[
    [AirfoilG3ProviderProfile], RuntimeSourceClosure
]


def _source_contract_section(
    source: RuntimeSourceClosure,
) -> RuntimeManifestSection:
    policy = DefaultEvidencePromptShapePolicyV3()
    paths = {value.logical_path for value in source.files}
    roles = {role for role, _ in source.role_paths}
    if not set(_REQUIRED_SOURCE_PATHS).issubset(paths):
        raise AirfoilV10RuntimeManifestError(
            "runtime closure omitted a required v10 source"
        )
    if not set(_REQUIRED_SOURCE_ROLES).issubset(roles):
        raise AirfoilV10RuntimeManifestError(
            "runtime closure omitted a required semantic role"
        )
    prompt_contract = {
        "prompt_shape_policy_id": policy.policy_id,
        "prompt_shape_policy_version": policy.policy_version,
        "renderer_policy_id": policy.renderer_policy_id,
        "renderer_policy_version": policy.renderer_policy_version,
        "candidate_component_path_contract_sha256": _contract_sha256(
            CANDIDATE_COMPONENT_PATH_CONTRACT
        ),
        "two_parent_crossover_evidence_contract_sha256": _contract_sha256(
            TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT
        ),
    }
    return RuntimeManifestSection.seal(
        "source_contract",
        {
            "schema_version": 1,
            "source_sha256": source.source_sha256,
            "required_roles": list(_REQUIRED_SOURCE_ROLES),
            "required_paths": list(_REQUIRED_SOURCE_PATHS),
            "prompt_and_schema_contract": prompt_contract,
            "prompt_and_schema_contract_sha256": _contract_sha256(prompt_contract),
            "tests_are_provenance_not_scientific_evidence": True,
        },
    )


def _evaluator_section(
    *,
    run_id: str,
    run_root: Path,
    work_root: Path,
) -> RuntimeManifestSection:
    settings = local_default_converged_settings()
    dataset = capture_runtime_file(
        settings.dataset_arrow,
        logical_path="external/evaluator/airfoil_v0-train.arrow",
    )
    if dataset.sha256 != EXPECTED_DATASET_SHA256:
        raise AirfoilV10RuntimeManifestError(
            "Airfoil dataset differs from its expected digest"
        )
    return RuntimeManifestSection.seal(
        "evaluator",
        {
            "schema_version": 1,
            "python_executable": str(
                settings.python_executable.expanduser().resolve(strict=True)
            ),
            "evaluator_script": str(
                settings.evaluator_script.expanduser().resolve(strict=True)
            ),
            "dataset": dataset.to_record(),
            "container_image": CONTAINER_IMAGE,
            "cpu_set": settings.cpu_set,
            "mpi_cores": settings.mpi_cores,
            "timeout_seconds": settings.timeout_seconds,
            "evaluator_concurrency": EVALUATOR_CONCURRENCY,
            "output_root": str((run_root / run_id / "cfd_receipts").resolve()),
            "work_root": str((work_root / run_id).resolve()),
            "physical_evaluator_called_during_manifest": False,
        },
    )


def build_airfoil_v10_runtime_manifest(
    *,
    inputs: AirfoilV10MultiOptionInputs,
    built_at_utc: str,
    run_id: str,
    provider_profile: AirfoilG3ProviderProfile,
    provider_record: Mapping[str, object],
    qualification: VerifiedAirfoilV10Qualification,
    run_root: Path,
    work_root: Path,
    source_closure_factory: AirfoilV10SourceClosureFactory = (
        capture_airfoil_v10_runtime_source_closure
    ),
) -> LiveRuntimeManifest:
    """Build one exact provider/evaluator-free prospective v10 commitment."""

    if type(inputs) is not AirfoilV10MultiOptionInputs:
        raise TypeError("inputs must be exact AirfoilV10MultiOptionInputs")
    inputs.__post_init__()
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise AirfoilV10RuntimeManifestError("run_id uses an invalid grammar")
    if type(provider_profile) is not AirfoilG3ProviderProfile:
        raise TypeError("provider_profile must be exact")
    provider_profile.__post_init__()
    if not isinstance(provider_record, Mapping):
        raise TypeError("provider_record must be a mapping")
    if type(qualification) is not VerifiedAirfoilV10Qualification:
        raise TypeError("qualification must be exact")
    qualification.__post_init__()
    if not callable(source_closure_factory):
        raise TypeError("source_closure_factory must be callable")
    source = source_closure_factory(provider_profile)
    if type(source) is not RuntimeSourceClosure:
        raise TypeError("source closure factory returned a foreign value")
    source.__post_init__()
    verified_qualification = verify_airfoil_v10_qualification_directory(
        qualification.directory,
        provider_profile=provider_profile,
        provider_record=dict(provider_record),
        source_closure_factory=source_closure_factory,
    )
    provider_configuration_sha256 = airfoil_v10_provider_configuration_sha256(
        dict(provider_record)
    )
    if (
        verified_qualification.receipt.source_sha256 != source.source_sha256
        or verified_qualification.receipt.provider_configuration_sha256
        != provider_configuration_sha256
        or verified_qualification.to_record() != qualification.to_record()
    ):
        raise AirfoilV10RuntimeManifestError(
            "qualification identity changed before manifest construction"
        )
    locks = tuple(
        value
        for value in source.files
        if value.logical_path in {"pyproject.toml", "uv.lock"}
    )
    if len(locks) != 2:
        raise AirfoilV10RuntimeManifestError(
            "runtime source closure does not bind both dependency locks"
        )
    experiment = RuntimeManifestSection.seal(
        "experiment",
        {
            "schema_version": 1,
            "run_id": run_id,
            "inputs_sha256": inputs.inputs_sha256,
            "task_sha256": inputs.task_sha256,
            "schedule_sha256": inputs.schedule_sha256,
            "pre_outcome_commit_sha256": inputs.pre_outcome_commit_sha256,
            "source_runtime_inputs_sha256": inputs.source_runtime_inputs_sha256,
            "source_release_sha256": inputs.source_release_sha256,
            "phase": inputs.phase,
            "provider_profile_id": provider_profile.profile_id,
            "logical_provider_call_cap": 7,
            "candidate_occurrence_count": 14,
            "claim_boundary": "development_workflow_evidence_not_paper_efficacy",
        },
    )
    provider = RuntimeManifestSection.seal(
        "provider_route",
        {
            "schema_version": 1,
            "profile_id": provider_profile.profile_id,
            "configuration": dict(provider_record),
            "configuration_sha256": provider_configuration_sha256,
            "qualification_configuration_join_exact": True,
            "credential_read": False,
            "provider_called": False,
        },
    )
    boundary = RuntimeManifestSection.seal(
        "execution_boundary",
        {
            "schema_version": 1,
            "manifest_written_before_live_composition": True,
            "manifest_verified_before_live_composition": True,
            "source_authenticated_before_resource_lease": True,
            "qualification_verified_before_resource_lease": True,
            "manifest_reverified_immediately_before_credential_access": True,
            "manifest_reverified_after_optimizer_completion": True,
            "credential_and_provider_access_during_manifest": False,
            "physical_evaluator_access_during_manifest": False,
            "drift_policy": "fail_closed",
        },
    )
    sections = (
        boundary,
        _evaluator_section(
            run_id=run_id,
            run_root=run_root,
            work_root=work_root,
        ),
        experiment,
        capture_git_worktree_section(
            AGENT_EVOLVE_ROOT,
            source_closure=source,
        ),
        RuntimeManifestSection.seal(
            "offline_qualification",
            {
                **verified_qualification.to_record(),
                "provider_configuration_join_exact": True,
                "qualification_output_is_outside_source_closure": True,
                "receipt_is_non_circular": True,
                "provider_or_physical_evaluator_authorized": False,
            },
        ),
        provider,
        capture_runtime_environment_section(
            distribution_names=(
                "httpx",
                "openai",
                "pydantic",
                "pydantic-ai",
                "pytest",
            ),
            dependency_locks=locks,
        ),
        _source_contract_section(source),
    )
    return build_live_runtime_manifest(
        manifest_id=AIRFOIL_V10_RUNTIME_MANIFEST_ID,
        manifest_version=AIRFOIL_V10_RUNTIME_MANIFEST_VERSION,
        built_at_utc=built_at_utc,
        source_closure=source,
        sections=sections,
        required_section_ids=tuple(value.section_id for value in sections),
    )


@dataclass(frozen=True, slots=True)
class AirfoilV10RuntimeManifestVerification:
    """Self-authenticating receipt for one exact reconstruction check."""

    manifest_sha256: str
    source_sha256: str
    inputs_sha256: str
    provider_profile_id: str

    def __post_init__(self) -> None:
        for name in ("manifest_sha256", "source_sha256", "inputs_sha256"):
            _require_sha256(getattr(self, name), name)
        if (
            type(self.provider_profile_id) is not str
            or not self.provider_profile_id
            or self.provider_profile_id != self.provider_profile_id.strip()
        ):
            raise ValueError("provider_profile_id must be canonical non-empty text")

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "manifest_sha256": self.manifest_sha256,
            "source_sha256": self.source_sha256,
            "inputs_sha256": self.inputs_sha256,
            "provider_profile_id": self.provider_profile_id,
            "verified_without_credentials_provider_or_evaluator": True,
        }

    @property
    def verification_sha256(self) -> str:
        return hashlib.sha256(
            _VERIFICATION_DOMAIN + _canonical_bytes(self._identity_record())
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            **self._identity_record(),
            "verification_sha256": self.verification_sha256,
        }


class FrozenAirfoilV10RuntimeManifestGate:
    """Reload and exactly reconstruct the v10 manifest at costly boundaries."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        inputs: AirfoilV10MultiOptionInputs,
        run_id: str,
        provider_profile: AirfoilG3ProviderProfile,
        provider_record: Mapping[str, object],
        qualification: VerifiedAirfoilV10Qualification,
        run_root: Path,
        work_root: Path,
        source_closure_factory: AirfoilV10SourceClosureFactory = (
            capture_airfoil_v10_runtime_source_closure
        ),
    ) -> None:
        if type(inputs) is not AirfoilV10MultiOptionInputs:
            raise TypeError("inputs must be exact AirfoilV10MultiOptionInputs")
        inputs.__post_init__()
        initial = load_live_runtime_manifest(manifest_path)
        if (
            initial.manifest_id != AIRFOIL_V10_RUNTIME_MANIFEST_ID
            or initial.manifest_version != AIRFOIL_V10_RUNTIME_MANIFEST_VERSION
        ):
            raise AirfoilV10RuntimeManifestError(
                "runtime manifest belongs to another experiment"
            )
        experiment = next(
            (
                section.to_record()["payload"]
                for section in initial.sections
                if section.section_id == "experiment"
            ),
            None,
        )
        if type(experiment) is not dict or (
            experiment.get("run_id") != run_id
            or experiment.get("inputs_sha256") != inputs.inputs_sha256
            or experiment.get("provider_profile_id") != provider_profile.profile_id
        ):
            raise AirfoilV10RuntimeManifestError(
                "runtime manifest experiment identity differs"
            )
        self.manifest_path = manifest_path.expanduser().resolve(strict=True)
        self.inputs = inputs
        self.run_id = run_id
        self.provider_profile = provider_profile
        self.provider_record = dict(provider_record)
        if type(qualification) is not VerifiedAirfoilV10Qualification:
            raise TypeError("qualification must be exact")
        qualification.__post_init__()
        self.qualification = qualification
        self.run_root = run_root
        self.work_root = work_root
        self.source_closure_factory = source_closure_factory
        self.expected_manifest_sha256 = initial.manifest_sha256
        self.expected_source_sha256 = initial.source_closure.source_sha256

    def verify(self) -> AirfoilV10RuntimeManifestVerification:
        frozen = load_live_runtime_manifest(self.manifest_path)
        if frozen.manifest_sha256 != self.expected_manifest_sha256:
            raise AirfoilV10RuntimeManifestError(
                "runtime manifest bytes or identity changed"
            )
        verify_runtime_source_closure(frozen.source_closure)
        current = build_airfoil_v10_runtime_manifest(
            inputs=self.inputs,
            built_at_utc=frozen.built_at_utc,
            run_id=self.run_id,
            provider_profile=self.provider_profile,
            provider_record=self.provider_record,
            qualification=self.qualification,
            run_root=self.run_root,
            work_root=self.work_root,
            source_closure_factory=self.source_closure_factory,
        )
        if current.to_record() != frozen.to_record():
            raise AirfoilV10RuntimeManifestError(
                "live runtime environment differs from frozen manifest"
            )
        return AirfoilV10RuntimeManifestVerification(
            manifest_sha256=frozen.manifest_sha256,
            source_sha256=frozen.source_closure.source_sha256,
            inputs_sha256=self.inputs.inputs_sha256,
            provider_profile_id=self.provider_profile.profile_id,
        )


def runtime_manifest_identity_record(
    manifest: LiveRuntimeManifest,
) -> dict[str, object]:
    """Return the compact identity embedded in readiness and result records."""

    if type(manifest) is not LiveRuntimeManifest:
        raise TypeError("manifest must be an exact LiveRuntimeManifest")
    manifest.__post_init__()
    return {
        "manifest_id": manifest.manifest_id,
        "manifest_version": manifest.manifest_version,
        "manifest_sha256": manifest.manifest_sha256,
        "source_sha256": manifest.source_closure.source_sha256,
        "built_at_utc": manifest.built_at_utc,
        "filename": AIRFOIL_V10_RUNTIME_MANIFEST_FILENAME,
    }


__all__ = [
    "AIRFOIL_V10_RUNTIME_MANIFEST_FILENAME",
    "AIRFOIL_V10_RUNTIME_MANIFEST_ID",
    "AIRFOIL_V10_RUNTIME_MANIFEST_VERSION",
    "AirfoilV10RuntimeManifestError",
    "AirfoilV10RuntimeManifestVerification",
    "AirfoilV10SourceClosureFactory",
    "FrozenAirfoilV10RuntimeManifestGate",
    "build_airfoil_v10_runtime_manifest",
    "capture_airfoil_v10_runtime_source_closure",
    "runtime_manifest_identity_record",
]
