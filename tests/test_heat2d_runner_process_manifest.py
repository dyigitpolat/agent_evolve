"""Provider- and PDE-free checks for Heat runner process evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

import pytest

from examples.development import run_heat2d_generic_campaign as agentic
from examples.development import run_heat2d_generic_uniform_control as control
from examples.development import uniform_feasible_portfolio_control as uniform


def _source() -> dict[str, object]:
    return {"aggregate_sha256": "a" * 64, "file_count": 1, "files": []}


def test_outer_manifests_record_transient_invocation_outside_stable_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "importlib.metadata.version", lambda distribution: f"test-{distribution}"
    )
    manifests = (
        agentic._manifest("agent-manifest-test", "prepare", _source()),
        control._manifest(
            run_id="control-manifest-test",
            mode="prepare",
            source=_source(),
            source_snapshot=_source(),
            preregistration=None,
        ),
    )
    for manifest in manifests:
        observation = manifest["evaluator_process_invocation_observation"]
        environment = manifest["environment"]
        assert observation["invoked_path"] == str(Path(sys.executable).absolute())
        assert observation["resolved_target"] == str(
            Path(sys.executable).resolve(strict=True)
        )
        assert environment["python_prefix"] == str(Path(sys.prefix).absolute())
        assert environment["python_base_prefix"] == str(
            Path(sys.base_prefix).absolute()
        )
        assert (
            "evaluator_process_invocation_observation"
            not in manifest["source_identity"]
        )
    treatment_manifest, control_manifest = manifests
    assert (
        control_manifest["utility_reference_qualification"]
        == treatment_manifest["utility_reference_qualification"]
    )
    assert (
        control_manifest["one_factor_match"]["objective_resolution"]
        == treatment_manifest["objective_resolution"]
    )


def test_control_source_snapshot_copies_exact_launch_bytes(tmp_path: Path) -> None:
    paths = (
        Path(control.__file__).resolve(strict=True),
        Path(uniform.__file__).resolve(strict=True),
    )
    identity = control.source_identity(paths, relative_to=control.WORKSPACE_ROOT)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = control._snapshot_sources(run_dir, paths)

    assert snapshot["aggregate_sha256"] == identity["aggregate_sha256"]
    assert snapshot["files"] == identity["files"]
    for path in paths:
        label = path.relative_to(control.WORKSPACE_ROOT)
        assert (run_dir / "source_snapshot" / label).read_bytes() == path.read_bytes()


def test_agentic_affine_reference_is_bound_to_qualified_all_void_support() -> None:
    spec = agentic._affine_spec()
    axes = {value.metric_id: value for value in spec.axes}
    assert axes[agentic.THERMAL_OBJECTIVE_NAME].reference == 0.005
    assert axes[agentic.MATERIAL_OBJECTIVE_NAME].reference == 0.61
    assert (
        agentic.QUALIFIED_ALL_VOID_THERMAL_TERM
        == 0.004492585018256053
        < axes[agentic.THERMAL_OBJECTIVE_NAME].reference
    )
    assert agentic.QUALIFIED_ALL_VOID_MANIFEST_SHA256 in spec.reference_provenance
    assert agentic.QUALIFIED_ALL_VOID_THERMAL_TERM.hex() in spec.reference_provenance


def test_agentic_manifest_freezes_next_generation_reflection_admission() -> None:
    manifest = agentic._manifest("reflection-admission-test", "prepare", _source())

    assert manifest["protocol"]["reflection_promotion_block_pairs"] == 1
    assert manifest["protocol"]["reflection_source_generations"] == [2]
    assert manifest["protocol"]["reflection_admission_generations"] == [4]
    assert manifest["protocol"]["first_reflection_consumer_generation"] == 5
    assert manifest["protocol"]["terminal_reflection"] is False
    assert manifest["protocol"]["planned_logical_llm_calls"] == 7
    assert agentic.PROTOCOL_ID.endswith("delayed_identifiable_v5")


def test_preregistration_excludes_machine_timing_diagnostic() -> None:
    parent = {
        "seed_id": "seed_1",
        "binding_object_reused": True,
        "raw_option_count": 10,
        "eligible_option_count": 10,
        "known_excluded_option_count": 0,
        "semantic_alias_count": 0,
        "context_sha256": "a" * 64,
        "card_sha256s": ["b" * 64],
    }
    readiness = {
        "status": "provider_and_pde_free_semantic_readiness",
        "resolution": 1001,
        "provider_calls": 0,
        "pde_solves": 0,
        "semantic_readiness_process_cpu_limit_s": 300.0,
        "gate_under_semantic_readiness_process_cpu_limit_each": True,
        "gate_under_60_process_cpu_s_each": False,
        "parents": [parent],
    }
    slow_projection = agentic._readiness_contract_projection(readiness)
    readiness["gate_under_60_process_cpu_s_each"] = True
    fast_projection = agentic._readiness_contract_projection(readiness)

    assert slow_projection == fast_projection
    assert "gate_under_60_process_cpu_s_each" not in slow_projection


@pytest.mark.parametrize("module", (agentic, control))
def test_manifest_failure_after_run_directory_creation_is_durably_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: object,
) -> None:
    artifact_root = tmp_path / "runs"
    monkeypatch.setattr(module, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(module, "source_identity", lambda *args, **kwargs: _source())
    monkeypatch.setattr(module, "_snapshot_sources", lambda *args, **kwargs: _source())

    def fail_manifest(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("manifest observation failed")

    monkeypatch.setattr(module, "_manifest", fail_manifest)
    args = argparse.Namespace(mode="prepare", run_id="durability-test", prereg=None)

    with pytest.raises(RuntimeError, match="manifest observation failed"):
        asyncio.run(module._main_async(args))

    run_dir = artifact_root / "durability-test"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    finalized = json.loads((run_dir / "finalized.json").read_text(encoding="utf-8"))
    assert summary["status"] in {"failed", "failed_before_live_execution"}
    assert finalized["status"] == "failed"
