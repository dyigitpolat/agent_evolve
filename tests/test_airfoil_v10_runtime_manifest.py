"""Provider/evaluator-free provenance gates for the Airfoil v10 runner."""

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import platform
import sys
from functools import cache
from pathlib import Path

import pytest

from agent_evolve.application.live_runtime_manifest import (
    LiveRuntimeManifestError,
    capture_runtime_source_closure,
)
from examples.benchmarks.engibench_airfoil import v7_g3_release as release
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    DEEPSEEK_G3_PROVIDER_PROFILE,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    compose_airfoil_g3_runtime_inputs,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import AirfoilV7Problem
from examples.benchmarks.engibench_airfoil.v10_multi_option_inputs import (
    compose_airfoil_v10_multi_option_inputs,
)
from examples.benchmarks.engibench_airfoil.v10_multi_option_runner import (
    _provider_config_record,
)
from examples.benchmarks.engibench_airfoil.v10_runtime_manifest import (
    AIRFOIL_V10_RUNTIME_MANIFEST_VERSION,
    FrozenAirfoilV10RuntimeManifestGate,
    build_airfoil_v10_runtime_manifest,
    capture_airfoil_v10_runtime_source_closure,
)
from examples.benchmarks.engibench_airfoil.v10_qualification import (
    AIRFOIL_V10_QUALIFICATION_DISTRIBUTIONS,
    AIRFOIL_V10_QUALIFICATION_JUNIT_FILENAME,
    AIRFOIL_V10_QUALIFICATION_RECEIPT_FILENAME,
    AIRFOIL_V10_QUALIFICATION_STATUS,
    AirfoilV10QualificationReceipt,
    airfoil_v10_provider_configuration_sha256,
    verify_airfoil_v10_qualification_directory,
)
from examples.development.durable_run_artifacts import (
    finalize_run_directory,
    write_bytes_atomic,
    write_json_atomic,
)


class _NoRawCFD:
    def evaluate_raw(self, configuration):
        del configuration
        raise AssertionError("runtime manifest construction must not invoke CFD")


@cache
def _inputs():
    preparation = release.prepare_release()
    permutation, _, _ = release.freeze_diagnostic_permutation(preparation)
    source = compose_airfoil_g3_runtime_inputs(
        problem=AirfoilV7Problem(raw_problem=_NoRawCFD()),
        preparation=preparation,
        diagnostic_permutation=permutation,
    )
    return compose_airfoil_v10_multi_option_inputs(source)


def _qualification(tmp_path: Path, *, source_factory):
    profile = DEEPSEEK_G3_PROVIDER_PROFILE
    provider_record = _provider_config_record(profile)
    source = source_factory(profile)
    root = tmp_path / "qualification"
    root.mkdir()
    test_count = 3
    cases = b"".join(
        f'<testcase classname="offline" name="case_{index}"/>'.encode("ascii")
        for index in range(test_count)
    )
    junit = (
        f'<testsuite tests="{test_count}" failures="0" errors="0" skipped="0">'.encode(
            "ascii"
        )
        + cases
        + b"</testsuite>"
    )
    write_bytes_atomic(root / AIRFOIL_V10_QUALIFICATION_JUNIT_FILENAME, junit)
    receipt = AirfoilV10QualificationReceipt(
        source_sha256=source.source_sha256,
        provider_profile_id=profile.profile_id,
        provider_configuration_sha256=(
            airfoil_v10_provider_configuration_sha256(provider_record)
        ),
        python_executable=str(Path(sys.executable).absolute()),
        python_version=platform.python_version(),
        installed_distributions=tuple(
            (name, importlib.metadata.version(name))
            for name in AIRFOIL_V10_QUALIFICATION_DISTRIBUTIONS
        ),
        started_at_utc="2026-07-16T02:00:00Z",
        finished_at_utc="2026-07-16T02:00:01Z",
        tests=test_count,
        failures=0,
        errors=0,
        skipped=0,
        junit_size_bytes=len(junit),
        junit_sha256=hashlib.sha256(junit).hexdigest(),
        stdout_size_bytes=0,
        stdout_sha256=hashlib.sha256(b"").hexdigest(),
        stderr_size_bytes=0,
        stderr_sha256=hashlib.sha256(b"").hexdigest(),
    )
    write_json_atomic(
        root / AIRFOIL_V10_QUALIFICATION_RECEIPT_FILENAME,
        receipt.to_record(),
    )
    finalize_run_directory(root, status=AIRFOIL_V10_QUALIFICATION_STATUS)
    return verify_airfoil_v10_qualification_directory(
        root,
        provider_profile=profile,
        provider_record=provider_record,
        source_closure_factory=source_factory,
    )


def _manifest(
    tmp_path: Path, *, source_factory=capture_airfoil_v10_runtime_source_closure
):
    profile = DEEPSEEK_G3_PROVIDER_PROFILE
    qualification = _qualification(tmp_path, source_factory=source_factory)
    return build_airfoil_v10_runtime_manifest(
        inputs=_inputs(),
        built_at_utc="2026-07-16T02:00:00Z",
        run_id="v10-runtime-manifest-test",
        provider_profile=profile,
        provider_record=_provider_config_record(profile),
        qualification=qualification,
        run_root=tmp_path / "runs",
        work_root=tmp_path / "work",
        source_closure_factory=source_factory,
    )


def test_manifest_binds_v3_prompt_schema_runner_core_and_tests(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    record = manifest.to_record()
    source = record["source_closure"]
    assert type(source) is dict
    paths = {value["logical_path"] for value in source["files"]}
    roles = source["roles"]

    assert {
        "examples/development/durable_run_artifacts.py",
        "examples/development/run_airfoil_v10_exact_stack_conformance.py",
        "src/agent_evolve/application/agentic_evolution.py",
        "src/agent_evolve/integrations/pydantic_ai/agentic_generator.py",
        "src/agent_evolve/integrations/pydantic_ai/async_generator.py",
        "src/agent_evolve/integrations/pydantic_ai/outbound_request_manifest.py",
        "src/agent_evolve/integrations/pydantic_ai/provider_attempt_join.py",
        "src/agent_evolve/policies/memory/prompt_shape.py",
        "src/agent_evolve/policies/variation/exact_parent_crossover.py",
        "examples/benchmarks/engibench_airfoil/v10_multi_option_runner.py",
        "examples/benchmarks/engibench_airfoil/v10_runtime_manifest.py",
        "tests/test_airfoil_v10_runtime_manifest.py",
        "tests/test_exact_parent_crossover.py",
        "tests/test_multi_option_model_crossover.py",
        "tests/test_openrouter_outbound_request_manifest.py",
        "tests/test_provider_attempt_join.py",
    }.issubset(paths)
    assert record["manifest_version"] == AIRFOIL_V10_RUNTIME_MANIFEST_VERSION == 3
    assert "artifact_journal" in roles
    assert "exact_stack_probe" in roles
    assert "generic_core" in roles
    assert "verification_tests" in roles
    qualification = next(
        value["payload"]
        for value in record["sections"]
        if value["section_id"] == "offline_qualification"
    )
    assert qualification["test_count"] == 3
    assert qualification["receipt_is_non_circular"] is True
    assert qualification["provider_configuration_join_exact"] is True
    assert set(qualification["installed_distributions"]) == set(
        AIRFOIL_V10_QUALIFICATION_DISTRIBUTIONS
    )
    provider = next(
        value["payload"]
        for value in record["sections"]
        if value["section_id"] == "provider_route"
    )
    assert provider["qualification_configuration_join_exact"] is True
    assert (
        provider["configuration_sha256"]
        == qualification["provider_configuration_sha256"]
    )
    contract = next(
        value["payload"]
        for value in record["sections"]
        if value["section_id"] == "source_contract"
    )
    prompt = contract["prompt_and_schema_contract"]
    assert prompt["prompt_shape_policy_version"] == 3
    assert prompt["renderer_policy_version"] == 3
    assert record["claim_boundary"] == {
        "credentials_read": False,
        "provider_called": False,
        "physical_evaluator_called": False,
        "current_run_outcomes_observed": False,
        "meaning": "prospective provider-free runtime commitment only",
    }


def test_gate_reconstructs_exactly_and_fails_closed_on_source_drift(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "sentinel.py"
    sentinel.write_text("VALUE = 1\n", encoding="utf-8")

    def source_factory(profile):
        base = capture_airfoil_v10_runtime_source_closure(profile)
        by_path = {
            value.logical_path: Path(value.resolved_path) for value in base.files
        }
        roles = {
            role: {path: by_path[path] for path in paths}
            for role, paths in base.role_paths
        }
        roles["verification_tests"]["external/v10_manifest_sentinel.py"] = sentinel
        return capture_runtime_source_closure(roles)

    manifest = _manifest(tmp_path, source_factory=source_factory)
    path = tmp_path / "runtime_manifest.json"
    write_json_atomic(path, manifest.to_record())
    profile = DEEPSEEK_G3_PROVIDER_PROFILE
    gate = FrozenAirfoilV10RuntimeManifestGate(
        manifest_path=path,
        inputs=_inputs(),
        run_id="v10-runtime-manifest-test",
        provider_profile=profile,
        provider_record=_provider_config_record(profile),
        qualification=verify_airfoil_v10_qualification_directory(
            tmp_path / "qualification",
            provider_profile=profile,
            provider_record=_provider_config_record(profile),
            source_closure_factory=source_factory,
        ),
        run_root=tmp_path / "runs",
        work_root=tmp_path / "work",
        source_closure_factory=source_factory,
    )
    receipt = gate.verify()
    assert receipt.manifest_sha256 == manifest.manifest_sha256
    assert receipt.source_sha256 == manifest.source_closure.source_sha256

    sentinel.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(LiveRuntimeManifestError, match="source closure drifted"):
        gate.verify()

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["manifest_sha256"] == manifest.manifest_sha256


def test_qualification_rejects_same_profile_id_with_changed_configuration(
    tmp_path: Path,
) -> None:
    profile = DEEPSEEK_G3_PROVIDER_PROFILE
    source_factory = capture_airfoil_v10_runtime_source_closure
    _qualification(tmp_path, source_factory=source_factory)
    altered = _provider_config_record(profile)
    altered["temperature"] = 0.0

    with pytest.raises(
        RuntimeError,
        match="different source, route, or runtime",
    ):
        verify_airfoil_v10_qualification_directory(
            tmp_path / "qualification",
            provider_profile=profile,
            provider_record=altered,
            source_closure_factory=source_factory,
        )
