"""Provider-free qualification of the immutable Timeloop v2 boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_evolve.agentic import FiniteVariationCatalog, freeze_json, thaw_json
from examples.benchmarks.timeloop_codesign.v2.candidate import (
    DEFAULT_CANDIDATE,
    POLICY_BLOCK_FIELDS,
    POLICY_FIELD_GRIDS,
    CandidateConfig,
    architecture_cardinality,
    candidate_cardinality,
    candidate_sha256,
    normalize_candidate,
    policy_cardinality,
)
from examples.benchmarks.timeloop_codesign.v2.compiler import (
    CompiledDataspaceConstraint,
    CompiledSpatialConstraint,
    CompiledTemporalConstraint,
    TimeloopV2Compiler,
    audit_compiler_injectivity,
    canonical_compiled_plan_bytes,
)
from examples.benchmarks.timeloop_codesign.v2.finite_variation_catalog import (
    TimeloopV2FiniteVariationCatalog,
)
from examples.benchmarks.timeloop_codesign.v2.network_panel import (
    NETWORK_ASSETS,
    ConvLayerShape,
    LayerMedoid,
    NetworkLayerPanel,
    panel_sha256,
    verify_network_asset,
)


def _shape(c: int, m: int, r: int, s: int, p: int, q: int) -> ConvLayerShape:
    return ConvLayerShape(
        channels_in=c,
        channels_out=m,
        filter_height=r,
        filter_width=s,
        output_height=p,
        output_width=q,
    )


def _panel(**overrides: object) -> NetworkLayerPanel:
    payload: dict[str, object] = {
        "panel_id": "test.resnet50.medoid3",
        "network_id": "resnet50",
        "role": "calibration",
        "source_asset_sha256": NETWORK_ASSETS["resnet50"].sha256,
        "extraction_definition_sha256": "a" * 64,
        "clustering_definition_sha256": "b" * 64,
        "supported_conv_layer_count": 53,
        "medoid_0": LayerMedoid(
            source_node_id="Conv_1",
            shape=_shape(64, 64, 3, 3, 56, 56),
            multiplicity=20,
        ),
        "medoid_1": LayerMedoid(
            source_node_id="Conv_2",
            shape=_shape(64, 256, 1, 1, 56, 56),
            multiplicity=16,
        ),
        "medoid_2": LayerMedoid(
            source_node_id="Conv_3",
            shape=_shape(256, 512, 3, 3, 28, 28),
            multiplicity=17,
        ),
    }
    payload.update(overrides)
    return NetworkLayerPanel.model_validate(payload, strict=True)


def _flatten(value: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in value.items():
        if type(item) is dict:
            for nested_key, nested_item in item.items():
                result[f"{key}.{nested_key}"] = nested_item
        else:
            result[key] = item
    return result


def test_candidate_is_closed_strict_immutable_and_order_canonical() -> None:
    candidate = normalize_candidate(DEFAULT_CANDIDATE)
    reversed_payload = dict(reversed(list(DEFAULT_CANDIDATE.items())))
    assert candidate_sha256(reversed_payload) == candidate_sha256(candidate)
    with pytest.raises(ValidationError):
        candidate.global_buffer_depth = 1024  # type: ignore[misc]

    invalid = (
        {**DEFAULT_CANDIDATE, "pe_mesh_x": "8"},
        {**DEFAULT_CANDIDATE, "pe_mesh_x": True},
        {**DEFAULT_CANDIDATE, "register_enabled": 1},
        {**DEFAULT_CANDIDATE, "policy_cluster_0": {"dataflow": "row"}},
        {**DEFAULT_CANDIDATE, "command": "timeloop-mapper /etc/passwd"},
    )
    for value in invalid:
        with pytest.raises(ValidationError):
            normalize_candidate(value)


def test_panel_binds_frozen_split_and_exact_cluster_coverage() -> None:
    panel = _panel()
    assert panel.supported_conv_layer_count == sum(
        item.multiplicity for item in panel.medoids()
    )
    assert len(panel_sha256(panel)) == 64

    with pytest.raises(ValidationError, match="frozen split"):
        _panel(role="validation")
    with pytest.raises(ValidationError, match="frozen network"):
        _panel(source_asset_sha256="0" * 64)
    with pytest.raises(ValidationError, match="cover every supported"):
        _panel(supported_conv_layer_count=54)
    with pytest.raises(ValidationError, match="operationally distinct"):
        _panel(
            medoid_1=LayerMedoid(
                source_node_id="Conv_copy",
                shape=_shape(64, 64, 3, 3, 56, 56),
                multiplicity=20,
            ),
            supported_conv_layer_count=57,
        )


def test_compiler_has_no_repair_and_uses_only_operational_plan_data() -> None:
    result = TimeloopV2Compiler.compile(DEFAULT_CANDIDATE, _panel())
    assert result.repair_count == 0
    assert len(result.compiled_plan_sha256) == 64
    encoded = canonical_compiled_plan_bytes(result.plan)
    assert b"row_stationary" not in encoded
    assert b"medium" not in encoded
    assert b"balanced" not in encoded
    assert b"channel_then_spatial" not in encoded
    assert b"source_node_id" not in encoded
    assert b"panel_id" not in encoded


def test_raw_and_provenance_only_aliases_coalesce_before_evaluation() -> None:
    first = _panel()
    second = _panel(
        panel_id="test.resnet50.renamed",
        extraction_definition_sha256="c" * 64,
        clustering_definition_sha256="d" * 64,
        medoid_0=first.medoid_0.model_copy(update={"source_node_id": "RenamedConv_1"}),
        medoid_1=first.medoid_1.model_copy(update={"source_node_id": "RenamedConv_2"}),
        medoid_2=first.medoid_2.model_copy(update={"source_node_id": "RenamedConv_3"}),
    )
    assert panel_sha256(first) != panel_sha256(second)
    first_result = TimeloopV2Compiler.compile(DEFAULT_CANDIDATE, first)
    second_result = TimeloopV2Compiler.compile(DEFAULT_CANDIDATE, second)
    assert first_result.compiled_plan_sha256 == second_result.compiled_plan_sha256


def test_semantically_inert_constraint_order_aliases_are_rejected() -> None:
    with pytest.raises(ValidationError, match="canonical dataspace order"):
        CompiledTemporalConstraint(
            target_role="global_buffer",
            permutation="NMCPQRS",
            no_reuse=("Outputs", "Inputs"),
        )
    with pytest.raises(ValidationError, match="canonical dimension order"):
        CompiledSpatialConstraint(
            permutation="CNMRSPQ",
            unit_factors=("Q=1", "P=1", "S=1", "R=1", "M=1", "N=1"),
            minimum_parallelism_numerator=1,
            minimum_parallelism_denominator=4,
        )
    with pytest.raises(ValidationError, match="canonical order"):
        CompiledDataspaceConstraint(
            keep=("Outputs", "Inputs"),
            bypass=("Weights",),
        )


def test_exact_component_proof_establishes_large_compiled_space() -> None:
    proof = audit_compiler_injectivity(_panel())
    assert architecture_cardinality() == 192
    assert policy_cardinality() == 1_536
    assert candidate_cardinality() == 695_784_701_952
    assert proof["exact_compiled_record_cardinality"] == candidate_cardinality()
    assert proof["architecture_records_checked"] == 192
    assert proof["policy_contexts_checked"] == 24
    assert set(proof["policy_records_checked_per_context"].values()) == {1_536}
    assert proof["whole_space_enumerated"] is False
    assert proof["timeloop_runtime_feasibility_proven"] is False
    assert len(proof["proof_sha256"]) == 64


def test_parent_local_catalog_has_61_novel_single_locus_phenotypes() -> None:
    panel = _panel()
    catalog = TimeloopV2FiniteVariationCatalog(panel)
    assert isinstance(catalog, FiniteVariationCatalog)
    parent = freeze_json(DEFAULT_CANDIDATE)
    options = catalog.options(parent)
    assert len(options) == 61
    assert len({option.child_configuration_sha256 for option in options}) == 61
    compiled_hashes = {
        dict(option.metadata)["compiled_plan_sha256"] for option in options
    }
    assert len(compiled_hashes) == 61

    parent_flat = _flatten(normalize_candidate(DEFAULT_CANDIDATE).model_dump())
    expected_loci = {
        "global_buffer_depth",
        "global_buffer_width",
        "pe_mesh_x",
        "datawidth_bits",
        "register_enabled",
        *(
            f"{block}.{field}"
            for block in POLICY_BLOCK_FIELDS
            for field, _, _ in POLICY_FIELD_GRIDS
        ),
    }
    observed_loci: set[str] = set()
    for option in options:
        child = normalize_candidate(thaw_json(option.child_configuration))
        changed = {
            locus
            for locus, value in _flatten(child.model_dump()).items()
            if value != parent_flat[locus]
        }
        assert changed == {dict(option.metadata)["locus"]}
        observed_loci.update(changed)
    assert observed_loci == expected_loci


def test_artifact_188_local_onnx_assets_match_the_frozen_hashes() -> None:
    workload_root = (
        Path(__file__).resolve().parents[2]
        / "medea_agent_evolve_wip"
        / "experiments"
        / "workloads"
    )
    for network_id, asset in NETWORK_ASSETS.items():
        assert verify_network_asset(workload_root / asset.filename, network_id) == (
            asset.sha256
        )


def test_policy_blocks_are_explicit_and_not_an_unbounded_sequence() -> None:
    schema = CandidateConfig.model_json_schema()
    assert set(POLICY_BLOCK_FIELDS).issubset(schema["properties"])
    assert "mapping_policies" not in schema["properties"]
