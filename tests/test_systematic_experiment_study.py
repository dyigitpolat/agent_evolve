"""Provider/evaluator-free checks for the systematic experiment registry."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from examples.development.systematic_experiment_study import (
    SystematicExperimentStudy,
)
from examples.development.run_systematic_reference_cell import (
    _cell,
    _plan,
    _verified_prepared_summary,
)


REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "systematic_experiment_grid_v1.json"
)
REGISTRY_V3 = (
    Path(__file__).resolve().parents[2]
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "systematic_experiment_registry_v3.json"
)
REGISTRY_V4 = (
    Path(__file__).resolve().parents[2]
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "systematic_experiment_registry_v4.json"
)
REGISTRY_V10 = (
    Path(__file__).resolve().parents[2]
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "systematic_experiment_registry_v10.json"
)
REGISTRY_V11 = (
    Path(__file__).resolve().parents[2]
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "systematic_experiment_registry_v11.json"
)
REGISTRY_V13 = (
    Path(__file__).resolve().parents[2]
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "systematic_experiment_registry_v13.json"
)
REGISTRY_V14 = (
    Path(__file__).resolve().parents[2]
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "systematic_experiment_registry_v14.json"
)
REGISTRY_V15 = (
    Path(__file__).resolve().parents[2]
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "systematic_experiment_registry_v15.json"
)
REGISTRY_V16 = (
    Path(__file__).resolve().parents[2]
    / "papers/agent_evolve_aaai_2027/research_artifacts/data/"
    "systematic_experiment_registry_v16.json"
)


def test_registry_crosses_six_models_three_workloads_and_shared_controls() -> None:
    study = SystematicExperimentStudy.load(REGISTRY)
    assert study.summary() == {
        "schema_version": 1,
        "study_id": "agentevolve_cross_model_cross_domain_scale_v1",
        "study_sha256": study.study_sha256,
        "model_count": 6,
        "workload_count": 3,
        "reference_treatment_cells": 18,
        "reference_shared_control_cells": 3,
        "mechanism_factorial_cells": 54,
        "total_initial_cells": 75,
    }
    cells = study.reference_cells()
    assert len({cell.cell_id for cell in cells}) == 21
    assert len({cell.cell_sha256 for cell in cells}) == 21


def test_l18_factorial_is_equal_budget_and_pairwise_orthogonal() -> None:
    study = SystematicExperimentStudy.load(REGISTRY)
    cells = study.factorial_cells()
    assert len(cells) == 18 * 3
    assert {cell.model_profile for cell in cells} == {"deepseek"}
    assert {cell.workload_id for cell in cells} == set(study.workload_ids)
    assert all(
        dict(cell.configuration)["scale_shape"]
        in {"g4_k8_r7", "g6_k8_r2", "g10_k4_r2"}
        for cell in cells
    )
    assert len({dict(cell.configuration)["design_row"] for cell in cells}) == 18


def test_registry_fails_closed_on_profile_or_orthogonal_array_drift(
    tmp_path: Path,
) -> None:
    record = json.loads(REGISTRY.read_text(encoding="utf-8"))
    profile_drift = deepcopy(record)
    profile_drift["models"][0]["profile_sha256"] = "0" * 64
    profile_path = tmp_path / "profile_drift.json"
    profile_path.write_text(json.dumps(profile_drift), encoding="utf-8")
    with pytest.raises(ValueError, match="profile hash drift"):
        SystematicExperimentStudy.load(profile_path)

    design_drift = deepcopy(record)
    design_drift["mechanism_factorial"]["orthogonal_array"][0][0] = 2
    design_path = tmp_path / "design_drift.json"
    design_path.write_text(json.dumps(design_drift), encoding="utf-8")
    with pytest.raises(ValueError, match="not pairwise orthogonal"):
        SystematicExperimentStudy.load(design_path)

    scale_drift = deepcopy(record)
    scale_factor = next(
        value
        for value in scale_drift["mechanism_factorial"]["factors"]
        if value["factor_id"] == "scale_shape"
    )
    scale_factor["levels"][0] = "g4_k14_r1"
    scale_path = tmp_path / "scale_drift.json"
    scale_path.write_text(json.dumps(scale_drift), encoding="utf-8")
    with pytest.raises(ValueError, match="executable presets"):
        SystematicExperimentStudy.load(scale_path)


def test_shared_control_cells_resolve_to_real_model_free_runners() -> None:
    study = SystematicExperimentStudy.load(REGISTRY)
    controls = {
        cell.workload_id: cell
        for cell in study.reference_cells()
        if cell.arm == "control"
    }
    assert set(controls) == {"boils_abc", "heat2d", "timeloop_v2"}
    plans = {
        workload: _plan(study=study, cell=cell, attempt_id="offline_plan")
        for workload, cell in controls.items()
    }
    assert all(
        "AGENT_EVOLVE_MODEL_PROFILE" not in plan["environment"]
        for plan in plans.values()
    )
    assert plans["boils_abc"]["live_command"][2] == "control"
    assert plans["heat2d"]["live_command"][2] == "live"
    assert plans["timeloop_v2"]["live_command"][2] == "run"
    assert all(
        plan["environment"]["AGENT_EVOLVE_REPLICATE_SEED"] == "20260740"
        for plan in plans.values()
    )


def test_v3_grid_is_common_universe_frontier_and_not_fake_factorial() -> None:
    study = SystematicExperimentStudy.load(REGISTRY_V3)
    assert study.summary() == {
        "schema_version": 3,
        "study_id": "agentevolve_common_universe_frontier_cross_domain_v3",
        "study_sha256": study.study_sha256,
        "model_count": 6,
        "workload_count": 3,
        "reference_treatment_cells": 18,
        "reference_shared_control_cells": 3,
        "mechanism_factorial_cells": 0,
        "total_initial_cells": 21,
    }
    assert study.factorial_cells() == ()
    assert {
        dict(cell.configuration)["feasibility_witness_mode"]
        for cell in study.reference_cells()
    } == {"task_keyed_common_pool"}
    assert {
        dict(cell.configuration)["archive_context_mode"]
        for cell in study.reference_cells()
    } == {"authenticated_affine_v1"}


def test_v4_changes_only_deepseek_transport_profile_identity() -> None:
    v3 = SystematicExperimentStudy.load(REGISTRY_V3)
    v4 = SystematicExperimentStudy.load(REGISTRY_V4)
    assert v4.summary()["model_count"] == 6
    assert set(v4.model_names) == (set(v3.model_names) - {"deepseek"}) | {
        "deepseek_json"
    }
    assert v4.workload_ids == v3.workload_ids
    v3_configuration = {
        tuple(sorted(dict(cell.configuration).items()))
        for cell in v3.reference_cells()
    }
    v4_configuration = {
        tuple(sorted(dict(cell.configuration).items()))
        for cell in v4.reference_cells()
    }
    assert v4_configuration == v3_configuration


def test_v11_changes_only_the_oss20_route_capacity_profile_identity() -> None:
    v10 = SystematicExperimentStudy.load(REGISTRY_V10)
    v11 = SystematicExperimentStudy.load(REGISTRY_V11)

    assert set(v11.model_names) == (set(v10.model_names) - {"gpt_oss_20b"}) | {
        "gpt_oss_20b_serial"
    }
    assert v11.workload_ids == v10.workload_ids
    assert v11.summary()["reference_treatment_cells"] == 18
    serial = next(
        cell
        for cell in v11.reference_cells()
        if cell.arm == "treatment"
        and cell.model_profile == "gpt_oss_20b_serial"
        and cell.workload_id == "boils_abc"
    )
    plan = _plan(study=v11, cell=serial, attempt_id="serial_plan")
    assert plan["environment"]["AGENT_EVOLVE_MODEL_PROFILE"] == (
        "gpt_oss_20b_serial"
    )


def test_v3_workload_contract_removes_central_runtime_and_cli_assumptions() -> None:
    from examples.development.run_timeloop_v2_frontier_probe_live import (
        ARTIFACT_ROOT as TIMELOOP_ARTIFACT_ROOT,
    )

    study = SystematicExperimentStudy.load(REGISTRY_V3)
    treatments = {
        cell.workload_id: cell
        for cell in study.reference_cells()
        if cell.arm == "treatment" and cell.model_profile == "qwen"
    }
    plans = {
        workload: _plan(study=study, cell=cell, attempt_id="contract_plan")
        for workload, cell in treatments.items()
    }
    assert all(plan["execution_contract"] is not None for plan in plans.values())
    assert plans["heat2d"]["prepare_command"][1:5] == [
        "run",
        "--with",
        "numpy==2.3.5",
        "python",
    ]
    assert "--replicate-seed" in plans["timeloop_v2"]["prepare_command"]
    assert "--replicate-seed" not in plans["boils_abc"]["prepare_command"]
    workspace_root = Path(__file__).resolve().parents[2]
    planned_timeloop_root = (
        workspace_root / plans["timeloop_v2"]["prepare_run_dir"]
    ).parent
    assert planned_timeloop_root == TIMELOOP_ARTIFACT_ROOT
    assert all(
        plan["environment"]["AGENT_EVOLVE_FEASIBILITY_WITNESS_MODE"]
        == "task_keyed_common_pool"
        for plan in plans.values()
    )
    assert all(
        plan["environment"]["AGENT_EVOLVE_COMMON_CANDIDATE_POOL_SIZE"]
        == "24"
        for plan in plans.values()
    )
    assert all(
        plan["environment"]["AGENT_EVOLVE_ARCHIVE_CONTEXT_MODE"]
        == "authenticated_affine_v1"
        for plan in plans.values()
    )


def test_v3_contract_hash_drift_fails_closed(tmp_path: Path) -> None:
    record = json.loads(REGISTRY_V3.read_text(encoding="utf-8"))
    record["workloads"][0]["execution_contract_sha256"] = "0" * 64
    path = tmp_path / "contract_drift.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="contract hash drift"):
        SystematicExperimentStudy.load(path)


def test_v10_executes_current_m24_structural_treatment_on_every_domain() -> None:
    study = SystematicExperimentStudy.load(REGISTRY_V10)
    assert study.summary() == {
        "schema_version": 3,
        "study_id": "agentevolve_m24_structural_cross_model_cross_domain_v10",
        "study_sha256": study.study_sha256,
        "model_count": 6,
        "workload_count": 3,
        "reference_treatment_cells": 18,
        "reference_shared_control_cells": 3,
        "mechanism_factorial_cells": 0,
        "total_initial_cells": 21,
    }
    cells = {
        cell.workload_id: cell
        for cell in study.reference_cells()
        if cell.arm == "treatment" and cell.model_profile == "qwen"
    }
    assert set(cells) == {"boils_abc", "heat2d", "timeloop_v2"}
    plans = {
        workload: _plan(study=study, cell=cell, attempt_id="v10_plan")
        for workload, cell in cells.items()
    }
    for plan in plans.values():
        assert plan["environment"] | {
            "AGENT_EVOLVE_ACQUISITION_MODE": "calibrated_frontier",
            "AGENT_EVOLVE_ARCHIVE_CONTEXT_MODE": "authenticated_affine_v1",
            "AGENT_EVOLVE_COMMON_CANDIDATE_POOL_SIZE": "24",
            "AGENT_EVOLVE_FEASIBILITY_WITNESS_MODE": "task_keyed_common_pool",
            "AGENT_EVOLVE_MODEL_PROFILE": "qwen",
            "AGENT_EVOLVE_REPLICATE_SEED": "20260770",
        } == plan["environment"]


def test_v13_executes_exact_frozen_frontier_successor_on_every_domain() -> None:
    study = SystematicExperimentStudy.load(REGISTRY_V13)
    assert study.summary() == {
        "schema_version": 3,
        "study_id": "agentevolve_frontier_successor_cross_model_cross_domain_v13",
        "study_sha256": study.study_sha256,
        "model_count": 6,
        "workload_count": 3,
        "reference_treatment_cells": 18,
        "reference_shared_control_cells": 3,
        "mechanism_factorial_cells": 0,
        "total_initial_cells": 21,
    }
    for cell in study.reference_cells():
        assert _cell(study, cell.cell_id) == cell

    cells = {
        cell.workload_id: cell
        for cell in study.reference_cells()
        if cell.arm == "treatment" and cell.model_profile == "qwen"
    }
    assert set(cells) == {"boils_abc", "heat2d", "timeloop_v2"}
    for workload, cell in cells.items():
        plan = _plan(study=study, cell=cell, attempt_id="v13_plan")
        assert plan["environment"] == {
            "AGENT_EVOLVE_ACQUISITION_MODE": "hierarchical_support",
            "AGENT_EVOLVE_ARCHIVE_CONTEXT_MODE": "authenticated_affine_v1",
            "AGENT_EVOLVE_COMMON_CANDIDATE_POOL_SIZE": "all",
            "AGENT_EVOLVE_FEASIBILITY_WITNESS_MODE": "task_keyed_common_pool",
            "AGENT_EVOLVE_MODEL_PROFILE": "qwen",
            "AGENT_EVOLVE_REPLICATE_SEED": "20260780",
            "AGENT_EVOLVE_STUDY_CELL_ID": cell.cell_id,
        }
        assert plan["cell"]["configuration"]["method_definition_sha256"] == (
            "6295707fd949d4a2db292fa6d5da13d11b3b1a44aa7f14f0ee07c5550faaa729"
        )
        assert plan["cell"]["configuration"]["candidate_pool_mode"] == (
            "complete_finite_contract"
        )
        assert plan["cell"]["workload_id"] == workload
        preregistration_index = plan["live_command"].index("--prereg") + 1
        assert plan["live_command"][preregistration_index].endswith(
            "/preregistration_template.json"
        )

    heat_control = next(
        cell
        for cell in study.reference_cells()
        if cell.arm == "control" and cell.workload_id == "heat2d"
    )
    heat_control_plan = _plan(
        study=study,
        cell=heat_control,
        attempt_id="v13_control_plan",
    )
    control_preregistration_index = (
        heat_control_plan["live_command"].index("--prereg") + 1
    )
    assert heat_control_plan["live_command"][
        control_preregistration_index
    ].endswith("/manifest.json")


def test_v14_preserves_frozen_cells_and_declares_boils_cpu_lease() -> None:
    previous = SystematicExperimentStudy.load(REGISTRY_V13)
    isolated = SystematicExperimentStudy.load(REGISTRY_V14)

    assert isolated.summary() == {
        "schema_version": 3,
        "study_id": (
            "agentevolve_frontier_successor_cross_model_cross_domain_v14"
        ),
        "study_sha256": isolated.study_sha256,
        "model_count": 6,
        "workload_count": 3,
        "reference_treatment_cells": 18,
        "reference_shared_control_cells": 3,
        "mechanism_factorial_cells": 0,
        "total_initial_cells": 21,
    }
    assert tuple(cell.to_record() for cell in isolated.reference_cells()) == tuple(
        cell.to_record() for cell in previous.reference_cells()
    )
    boils = next(
        cell
        for cell in isolated.reference_cells()
        if cell.arm == "treatment"
        and cell.model_profile == "qwen"
        and cell.workload_id == "boils_abc"
    )
    plan = _plan(study=isolated, cell=boils, attempt_id="v14_plan")
    assert plan["execution_contract"]["exclusive_resource"] == {
        "resource_key": "boils_abc_cpu120_123_evaluator_lane",
        "lease_relative_path": (
            "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs/"
            "systematic_resource_leases/boils_abc_cpu120_123_evaluator_lane.lock"
        ),
    }
    assert plan["environment"]["AGENT_EVOLVE_COMMON_CANDIDATE_POOL_SIZE"] == (
        "all"
    )
    assert plan["cell"]["configuration"]["method_definition_sha256"] == (
        "6295707fd949d4a2db292fa6d5da13d11b3b1a44aa7f14f0ee07c5550faaa729"
    )


def test_v15_changes_only_the_bounded_finite_wire_treatment_identity() -> None:
    previous = SystematicExperimentStudy.load(REGISTRY_V14)
    repaired = SystematicExperimentStudy.load(REGISTRY_V15)

    assert repaired.summary() == {
        "schema_version": 3,
        "study_id": "agentevolve_exact_finite_wire_cross_model_cross_domain_v15",
        "study_sha256": repaired.study_sha256,
        "model_count": 6,
        "workload_count": 3,
        "reference_treatment_cells": 18,
        "reference_shared_control_cells": 3,
        "mechanism_factorial_cells": 0,
        "total_initial_cells": 21,
    }
    assert repaired.model_names == previous.model_names
    assert repaired.workload_ids == previous.workload_ids
    previous_configuration = dict(previous.reference_cells()[0].configuration)
    repaired_configuration = dict(repaired.reference_cells()[0].configuration)
    assert repaired_configuration == previous_configuration | {
        "finite_option_inline_enum_limit_utf8_bytes": "8192",
        "finite_option_wire_mode": (
            "complete_enum_when_bounded_local_exact_fallback"
        ),
    }
    assert {
        cell.cell_id for cell in repaired.reference_cells()
    }.isdisjoint(cell.cell_id for cell in previous.reference_cells())
    qwen_heat = next(
        cell
        for cell in repaired.reference_cells()
        if cell.arm == "treatment"
        and cell.model_profile == "qwen"
        and cell.workload_id == "heat2d"
    )
    assert _cell(repaired, qwen_heat.cell_id) == qwen_heat
    plan = _plan(study=repaired, cell=qwen_heat, attempt_id="v15_plan")
    assert plan["environment"] == {
        "AGENT_EVOLVE_ACQUISITION_MODE": "hierarchical_support",
        "AGENT_EVOLVE_ARCHIVE_CONTEXT_MODE": "authenticated_affine_v1",
        "AGENT_EVOLVE_COMMON_CANDIDATE_POOL_SIZE": "all",
        "AGENT_EVOLVE_FEASIBILITY_WITNESS_MODE": "task_keyed_common_pool",
        "AGENT_EVOLVE_MODEL_PROFILE": "qwen",
        "AGENT_EVOLVE_REPLICATE_SEED": "20260780",
        "AGENT_EVOLVE_STUDY_CELL_ID": qwen_heat.cell_id,
    }


def test_v16_executes_operator_stratified_v6_on_all_models_and_domains() -> None:
    study = SystematicExperimentStudy.load(REGISTRY_V16)
    assert study.summary() == {
        "schema_version": 3,
        "study_id": "agentevolve_operator_stratified_cross_model_cross_domain_v16",
        "study_sha256": study.study_sha256,
        "model_count": 6,
        "workload_count": 3,
        "reference_treatment_cells": 18,
        "reference_shared_control_cells": 3,
        "mechanism_factorial_cells": 0,
        "total_initial_cells": 21,
    }
    deepseek_cells = {
        cell.workload_id: cell
        for cell in study.reference_cells()
        if cell.arm == "treatment" and cell.model_profile == "deepseek"
    }
    assert set(deepseek_cells) == {"boils_abc", "heat2d", "timeloop_v2"}
    for cell in deepseek_cells.values():
        assert _cell(study, cell.cell_id) == cell
        plan = _plan(study=study, cell=cell, attempt_id="v16_plan")
        assert plan["environment"] == {
            "AGENT_EVOLVE_ACQUISITION_MODE": "operator_stratified",
            "AGENT_EVOLVE_ARCHIVE_CONTEXT_MODE": "authenticated_affine_v1",
            "AGENT_EVOLVE_COMMON_CANDIDATE_POOL_SIZE": "all",
            "AGENT_EVOLVE_COMPOSITE_OPTION_COUNT": "16",
            "AGENT_EVOLVE_FEASIBILITY_WITNESS_MODE": "task_keyed_common_pool",
            "AGENT_EVOLVE_MODEL_PROFILE": "deepseek",
            "AGENT_EVOLVE_OPERATOR_ASSAY_MINIMUM": "1",
            "AGENT_EVOLVE_REPLICATE_SEED": "20260782",
            "AGENT_EVOLVE_REQUIRED_COMPOSITE_PROPOSALS": "2",
            "AGENT_EVOLVE_STUDY_CELL_ID": cell.cell_id,
            "AGENT_EVOLVE_VARIATION_TOPOLOGY": "hierarchical_r2",
        }
        assert plan["cell"]["configuration"]["method_definition_sha256"] == (
            "2aea2dcca7e7b165df27160a92a44b9fe246f44122e5e1a28b9504a49c38f5dc"
        )


def test_prepared_execution_plan_reuses_qualified_preregistration() -> None:
    study = SystematicExperimentStudy.load(REGISTRY_V3)
    cell = next(
        value
        for value in study.reference_cells()
        if value.workload_id == "heat2d" and value.model_profile == "qwen"
    )
    plan = _plan(
        study=study,
        cell=cell,
        attempt_id="v2wave1",
        prepared_attempt_id="v2q0",
    )
    assert "grid_heat2d_qwen_s20260750_v2q0_prepare" in plan["prepare_run_dir"]
    assert "grid_heat2d_qwen_s20260750_v2wave1_live" in plan["live_run_dir"]
    prereg_index = plan["live_command"].index("--prereg") + 1
    assert "grid_heat2d_qwen_s20260750_v2q0_prepare" in plan["live_command"][
        prereg_index
    ]


def test_prepared_execution_fails_closed_on_summary_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import examples.development.run_systematic_reference_cell as launcher

    workspace = tmp_path / "workspace"
    receipt_root = workspace / "receipts"
    prepare_dir = workspace / "prepared"
    receipt = receipt_root / "cell_a" / "v2q0"
    receipt.mkdir(parents=True)
    prepare_dir.mkdir(parents=True)
    prepare_summary = prepare_dir / "summary.json"
    prepare_summary.write_text('{"status":"prepared_test"}', encoding="utf-8")
    digest = hashlib.sha256(prepare_summary.read_bytes()).hexdigest()
    (receipt / "summary.json").write_text(
        json.dumps(
            {
                "status": "prepared_healthy",
                "cell_id": "cell_a",
                "cell_sha256": "a" * 64,
                "prepare_summary_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    (receipt / "finalized.json").write_text(
        '{"status":"prepared_healthy"}', encoding="utf-8"
    )
    monkeypatch.setattr(launcher, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(launcher, "RECEIPT_ROOT", receipt_root)
    plan = {
        "prepared_attempt_id": "v2q0",
        "cell": {"cell_id": "cell_a", "cell_sha256": "a" * 64},
        "prepare_run_dir": "prepared",
    }
    assert _verified_prepared_summary(plan) == {"status": "prepared_test"}
    prepare_summary.write_text('{"status":"tampered"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="drifted"):
        _verified_prepared_summary(plan)
