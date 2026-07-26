"""Provider-free host-contract tests for the Timeloop v2 evaluator."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from examples.benchmarks.timeloop_codesign.v2 import DEFAULT_CANDIDATE
from examples.benchmarks.timeloop_codesign.v2.container_runner import (
    MAX_CONSECUTIVE_INVALID_MAPPINGS,
    SEARCH_SIZE,
)
from examples.benchmarks.timeloop_codesign.v2.evaluator import (
    LayerContainerResult,
    TimeloopV2DockerEvaluator,
    TimeloopV2EvaluatorPort,
    TimeloopV2ContractError,
    TimeloopV2CandidateInfeasibleError,
    TimeloopV2StaticInfeasibleEvaluation,
    TimeloopV2Settings,
    analyze_static_mapspace_feasibility,
    build_evaluation_bundle,
    canonical_evaluation_bundle_bytes,
    _require_complete_mapping_budget,
)
from examples.benchmarks.timeloop_codesign.v2.frozen_panels import (
    FROZEN_PANEL_SHA256,
    frozen_network_panel,
)


def test_bundle_binds_compiler_panel_manifests_and_frozen_mapper_protocol() -> None:
    panel = frozen_network_panel("resnet50")
    first = build_evaluation_bundle(DEFAULT_CANDIDATE, panel)
    second = build_evaluation_bundle(DEFAULT_CANDIDATE, panel)
    assert first == second
    assert canonical_evaluation_bundle_bytes(
        first
    ) == canonical_evaluation_bundle_bytes(second)
    assert first.panel_sha256 == FROZEN_PANEL_SHA256["resnet50"]
    assert tuple(item.medoid_ordinal for item in first.layer_manifests) == (0, 1, 2)
    assert sum(item.layer_multiplicity for item in first.layer_manifests) == 53
    assert first.protocol.search_size == SEARCH_SIZE == 400
    assert (
        first.protocol.max_consecutive_invalid_mappings
        == MAX_CONSECUTIVE_INVALID_MAPPINGS
        == 100_000
    )
    assert first.protocol.mapper_threads == 1
    assert first.protocol.mapper_algorithm == "random_pruned"


def test_settings_and_evaluator_expose_a_closed_serial_port(tmp_path: Path) -> None:
    settings = TimeloopV2Settings(output_root=tmp_path)
    evaluator = TimeloopV2DockerEvaluator(settings, frozen_network_panel("resnet50"))
    assert isinstance(evaluator, TimeloopV2EvaluatorPort)
    assert evaluator.evaluator_concurrency == 1
    assert evaluator.panel is frozen_network_panel("resnet50")
    with pytest.raises(ValueError, match="protocol is frozen"):
        TimeloopV2Settings(output_root=tmp_path, search_size=399)
    with pytest.raises(ValueError, match="invalid-mapping limit is frozen"):
        TimeloopV2Settings(
            output_root=tmp_path,
            max_consecutive_invalid_mappings=10_000,
        )
    with pytest.raises(ValueError, match="serialized"):
        TimeloopV2Settings(output_root=tmp_path, external_concurrency=2)


def test_static_empty_mapspace_is_proven_before_native_simulator_invocation(
    tmp_path: Path,
) -> None:
    candidate = {
        **DEFAULT_CANDIDATE,
        "pe_mesh_x": 4,
        "policy_cluster_1": {
            **DEFAULT_CANDIDATE["policy_cluster_1"],
            "primary_spatial_axis": "Q",
            "spatial_utilization": "full",
        },
    }
    bundle = build_evaluation_bundle(candidate, frozen_network_panel("resnet50"))
    infeasibility = analyze_static_mapspace_feasibility(bundle)

    assert type(infeasibility) is TimeloopV2StaticInfeasibleEvaluation
    assert len(infeasibility.witnesses) == 1
    witness = infeasibility.witnesses[0]
    assert witness.medoid_ordinal == 1
    assert witness.primary_axis == "Q"
    assert witness.axis_extent == 14
    assert (witness.minimum_parallelism, witness.maximum_parallelism) == (4, 4)
    assert witness.admissible_spatial_factors == ()

    output_root = tmp_path / "must-not-exist"
    evaluator = TimeloopV2DockerEvaluator(
        TimeloopV2Settings(output_root=output_root),
        frozen_network_panel("resnet50"),
    )
    with pytest.raises(TimeloopV2CandidateInfeasibleError) as captured:
        evaluator.evaluate(candidate)
    assert captured.value.observation == infeasibility
    assert not output_root.exists()


def test_layer_termination_receipts_cannot_claim_inconsistent_budget_health() -> None:
    common = {
        "medoid_ordinal": 0,
        "layer_multiplicity": 1,
        "layer_manifest_sha256": "a" * 64,
        "energy_joules": 1.0,
        "latency_seconds": 1.0,
        "area_square_meters": 1.0,
        "cycles": 1,
        "computes": 1,
        "requested_valid_mapping_count": SEARCH_SIZE,
        "elapsed_s": 1.0,
        "mapping_sha256": "b" * 64,
        "processed_input_sha256": "c" * 64,
        "output_subdirectory": "medoid-0",
        "front_end_projection_exact": True,
    }
    complete = LayerContainerResult(
        **common,
        reported_valid_mapping_count=SEARCH_SIZE,
        consecutive_invalid_mapping_count=0,
        mapping_budget_complete=True,
        termination_reason="valid_mapping_target",
    )
    assert complete.mapping_budget_complete is True
    with pytest.raises(ValidationError, match="completed mapping-budget"):
        LayerContainerResult(
            **common,
            reported_valid_mapping_count=SEARCH_SIZE - 1,
            consecutive_invalid_mapping_count=0,
            mapping_budget_complete=True,
            termination_reason="valid_mapping_target",
        )
    with pytest.raises(ValidationError, match="invalid-limit receipt"):
        LayerContainerResult(
            **common,
            reported_valid_mapping_count=None,
            consecutive_invalid_mapping_count=0,
            mapping_budget_complete=False,
            termination_reason="consecutive_invalid_limit",
        )

    incomplete = LayerContainerResult(
        **common,
        reported_valid_mapping_count=None,
        consecutive_invalid_mapping_count=100_000,
        mapping_budget_complete=False,
        termination_reason="consecutive_invalid_limit",
    )
    with pytest.raises(TimeloopV2ContractError, match="did not complete"):
        _require_complete_mapping_budget(incomplete)
