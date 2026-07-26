"""Provider-free qualification of Timeloop v2's frozen network panels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_evolve.agentic import freeze_json
from examples.benchmarks.timeloop_codesign.v2 import (
    DEFAULT_CANDIDATE,
    FROZEN_EXTRACTION_RECEIPT_SHA256,
    FROZEN_NETWORK_PANELS,
    FROZEN_PANEL_SHA256,
    TimeloopV2Compiler,
    TimeloopV2FiniteVariationCatalog,
    extract_network_panel,
    frozen_network_panel,
)
from examples.benchmarks.timeloop_codesign.v2.panel_extractor import (
    PanelExtractionResult,
    canonical_extraction_receipt_bytes,
)


_EXPECTED_COUNTS = {
    "resnet50": (122, 53, 23),
    "googlenet": (139, 57, 49),
    "yolov3": (304, 75, 23),
}


def _workload_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "medea_agent_evolve_wip"
        / "experiments"
        / "workloads"
    )


def test_frozen_panels_are_immutable_complete_and_compiler_ready() -> None:
    assert set(FROZEN_NETWORK_PANELS) == {"resnet50", "googlenet", "yolov3"}
    for network_id, panel in FROZEN_NETWORK_PANELS.items():
        assert frozen_network_panel(network_id) is panel
        assert sum(medoid.multiplicity for medoid in panel.medoids()) == (
            panel.supported_conv_layer_count
        )
        assert len({medoid.shape for medoid in panel.medoids()}) == 3
        compilation = TimeloopV2Compiler.compile(DEFAULT_CANDIDATE, panel)
        assert compilation.panel_sha256 == FROZEN_PANEL_SHA256[network_id]
        assert compilation.repair_count == 0
        options = TimeloopV2FiniteVariationCatalog(panel).options(
            freeze_json(DEFAULT_CANDIDATE)
        )
        assert len(options) == 61
        assert len({option.child_configuration_sha256 for option in options}) == 61

    with pytest.raises(TypeError):
        FROZEN_NETWORK_PANELS["resnet50"] = frozen_network_panel("resnet50")  # type: ignore[index]
    with pytest.raises(ValueError, match="frozen v2 split"):
        frozen_network_panel("unknown")


def test_pinned_onnx_assets_reproduce_every_frozen_panel_and_receipt() -> None:
    pytest.importorskip("onnx", minversion="1.18.0")
    root = _workload_root()
    for network_id, panel in FROZEN_NETWORK_PANELS.items():
        result = extract_network_panel(root / f"{network_id}.onnx", network_id)
        graph_nodes, supported, unique_shapes = _EXPECTED_COUNTS[network_id]
        assert result.panel == panel
        assert result.panel_sha256 == FROZEN_PANEL_SHA256[network_id]
        assert result.receipt_sha256 == FROZEN_EXTRACTION_RECEIPT_SHA256[network_id]
        assert result.receipt.graph_node_count == graph_nodes
        assert result.receipt.conv_node_count == supported
        assert result.receipt.supported_conv_layer_count == supported
        assert result.receipt.unique_supported_shape_count == unique_shapes
        assert result.receipt.excluded_conv_nodes == ()
        assert len(result.receipt.memberships) == supported


def test_extraction_is_byte_repeatable_and_contains_no_simulator_outcome() -> None:
    pytest.importorskip("onnx", minversion="1.18.0")
    path = _workload_root() / "resnet50.onnx"
    first = extract_network_panel(path, "resnet50")
    second = extract_network_panel(path, "resnet50")
    assert first == second
    assert canonical_extraction_receipt_bytes(first.receipt) == (
        canonical_extraction_receipt_bytes(second.receipt)
    )
    keys = set(
        _walk_keys(
            json.loads(
                canonical_extraction_receipt_bytes(first.receipt).decode("ascii")
            )
        )
    )
    assert keys.isdisjoint(
        {
            "area",
            "area_square_meters",
            "cycles",
            "edp",
            "energy",
            "energy_joules",
            "latency",
            "latency_seconds",
            "objective",
            "reward",
        }
    )

    tampered_receipt = first.receipt.model_copy(update={"panel_sha256": "0" * 64})
    with pytest.raises(ValueError, match="bind the panel"):
        PanelExtractionResult(
            panel=first.panel,
            receipt=tampered_receipt,
            panel_sha256=first.panel_sha256,
            receipt_sha256=first.receipt_sha256,
        )


def _walk_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(value) + tuple(
            key for item in value.values() for key in _walk_keys(item)
        )
    if isinstance(value, list):
        return tuple(key for item in value for key in _walk_keys(item))
    return ()
