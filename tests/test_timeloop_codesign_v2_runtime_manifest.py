"""Pure tests for the compiled-plan to Timeloop-v4 translation boundary."""

from __future__ import annotations

import json

from agent_evolve.agentic import freeze_json, thaw_json
from examples.benchmarks.timeloop_codesign.v2 import (
    DEFAULT_CANDIDATE,
    TimeloopV2Compiler,
    TimeloopV2FiniteVariationCatalog,
    compile_runtime_layer_manifests,
    frozen_network_panel,
    runtime_layer_manifest_sha256,
)
from examples.benchmarks.timeloop_codesign.v2.runtime_manifest import (
    RuntimeDataspaceConstraint,
    RuntimeSpatialConstraint,
    RuntimeTemporalConstraint,
    canonical_runtime_layer_manifest_bytes,
)


def test_default_compilation_translates_all_actual_medoids_without_repair() -> None:
    panel = frozen_network_panel("resnet50")
    compilation = TimeloopV2Compiler.compile(DEFAULT_CANDIDATE, panel)
    manifests = compile_runtime_layer_manifests(compilation)
    assert tuple(item.medoid_ordinal for item in manifests) == (0, 1, 2)
    assert sum(item.layer_multiplicity for item in manifests) == 53
    assert len({runtime_layer_manifest_sha256(item) for item in manifests}) == 3

    for manifest in manifests:
        assert manifest.compiled_plan_sha256 == compilation.compiled_plan_sha256
        assert manifest.candidate_sha256 == compilation.candidate_sha256
        assert manifest.panel_sha256 == compilation.panel_sha256
        assert manifest.problem_instance["G"] == 1
        assert manifest.problem_instance["Hpad"] == 0
        assert manifest.problem_instance["Wpad"] == 0
        local, spatial, global_temporal, dataspace = manifest.constraints
        assert isinstance(local, RuntimeTemporalConstraint)
        assert local.target == "reg"
        assert isinstance(spatial, RuntimeSpatialConstraint)
        assert spatial.permutation[0] == "M"
        assert spatial.minimum_parallelism == 4
        assert spatial.maximum_parallelism == 8
        assert spatial.factors[-2:] == ("M>=4", "M<=8")
        assert spatial.factors[-3] == "G=1"
        assert isinstance(global_temporal, RuntimeTemporalConstraint)
        assert global_temporal.target == "buffer"
        assert isinstance(dataspace, RuntimeDataspaceConstraint)
        assert dataspace.keep == ("Inputs", "Weights", "Outputs")
        assert dataspace.bypass == ()
        decoded = json.loads(
            canonical_runtime_layer_manifest_bytes(manifest).decode("ascii")
        )
        assert decoded["constraints"][1]["type"] == "spatial"
        assert decoded["constraints"][1]["minimum_parallelism"] == 4
        assert decoded["constraints"][1]["maximum_parallelism"] == 8


def test_register_disabled_uses_existing_inter_pe_temporal_storage_level() -> None:
    candidate = {**DEFAULT_CANDIDATE, "register_enabled": False}
    compilation = TimeloopV2Compiler.compile(
        candidate,
        frozen_network_panel("googlenet"),
    )
    for manifest in compile_runtime_layer_manifests(compilation):
        local = manifest.constraints[0]
        assert isinstance(local, RuntimeTemporalConstraint)
        assert local.target == "PE"
        assert manifest.architecture.register_enabled is False


def test_every_parent_local_choice_has_a_closed_three_layer_runtime_projection() -> (
    None
):
    panel = frozen_network_panel("resnet50")
    options = TimeloopV2FiniteVariationCatalog(panel).options(
        freeze_json(DEFAULT_CANDIDATE)
    )
    manifest_hash_sets: set[tuple[str, str, str]] = set()
    for option in options:
        compilation = TimeloopV2Compiler.compile(
            thaw_json(option.child_configuration),
            panel,
        )
        manifests = compile_runtime_layer_manifests(compilation)
        manifest_hash_sets.add(
            tuple(runtime_layer_manifest_sha256(item) for item in manifests)
        )
        for manifest in manifests:
            assert len(manifest.constraints) == 4
            assert manifest.constraints[1].type == "spatial"
    assert len(manifest_hash_sets) == 61
