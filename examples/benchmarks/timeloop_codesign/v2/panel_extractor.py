"""Outcome-blind ONNX Conv extraction and exact three-medoid clustering.

This module is a benchmark-owned preprocessing boundary.  It deliberately
does not import ONNX at module import time: the core AgentEvolve package does
not depend on ONNX, while the one-shot panel-freezing command pins its parser
version explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .network_panel import (
    NETWORK_ASSETS,
    ConvLayerShape,
    LayerMedoid,
    NetworkLayerPanel,
    panel_sha256,
    verify_network_asset,
)


EXTRACTOR_ID = "timeloop_v2_onnx_conv_extractor"
EXTRACTOR_VERSION = 1
CLUSTERING_ID = "timeloop_v2_exact_relative_l1_kmedoids"
CLUSTERING_VERSION = 1
PINNED_ONNX_VERSION = "1.18.0"
MEDOID_COUNT = 3

_EXTRACTION_HASH_DOMAIN = b"agent-evolve:timeloop-v2-extractor-definition:v1\x00"
_CLUSTERING_HASH_DOMAIN = b"agent-evolve:timeloop-v2-clustering-definition:v1\x00"
_RECEIPT_HASH_DOMAIN = b"agent-evolve:timeloop-v2-panel-extraction-receipt:v1\x00"

_EXTRACTION_DEFINITION = {
    "extractor_id": EXTRACTOR_ID,
    "extractor_version": EXTRACTOR_VERSION,
    "onnx_version": PINNED_ONNX_VERSION,
    "operator": "Conv",
    "layout": "NCHW",
    "supported_rank": 4,
    "supported_group": 1,
    "static_positive_dimensions_required": True,
    "weight_input_shape_required": True,
    "output_shape_required": True,
    "attribute_defaults": {
        "group": 1,
        "strides": [1, 1],
        "dilations": [1, 1],
    },
    "consistency_checks": [
        "input_batch_equals_output_batch",
        "input_channels_equals_weight_channels",
        "output_channels_equals_weight_outputs",
        "kernel_shape_equals_weight_kernel",
    ],
    "excluded_operator_policy": "ignore_non_conv_and_receipt_every_excluded_conv",
}

_CLUSTERING_DEFINITION = {
    "clustering_id": CLUSTERING_ID,
    "clustering_version": CLUSTERING_VERSION,
    "k": MEDOID_COUNT,
    "unit": "unique_operational_conv_shape",
    "weight": "number_of_supported_onnx_nodes_with_shape",
    "feature_order": [
        "batch",
        "channels_in",
        "channels_out",
        "filter_height",
        "filter_width",
        "output_height",
        "output_width",
        "stride_height",
        "stride_width",
        "dilation_height",
        "dilation_width",
        "mac_count",
    ],
    "per_feature_distance": "abs(a-b)/max(a,b)",
    "aggregate_distance": "exact_unweighted_l1_sum",
    "search": "exact_exhaustive_k_combination",
    "assignment_tie_break": "lowest_canonical_medoid_shape",
    "solution_tie_break": "lexicographically_lowest_canonical_medoid_shapes",
    "source_node_tie_break": "lexicographically_lowest_node_id_for_medoid_shape",
    "slot_order": "canonical_medoid_shape",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


EXTRACTION_DEFINITION_SHA256 = hashlib.sha256(
    _EXTRACTION_HASH_DOMAIN + _canonical_bytes(_EXTRACTION_DEFINITION)
).hexdigest()
CLUSTERING_DEFINITION_SHA256 = hashlib.sha256(
    _CLUSTERING_HASH_DOMAIN + _canonical_bytes(_CLUSTERING_DEFINITION)
).hexdigest()


SafeId = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=192,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$",
    ),
]
PositiveStrictInt = Annotated[int, Field(strict=True, ge=1)]
NonnegativeStrictInt = Annotated[int, Field(strict=True, ge=0)]


class _ClosedModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class ExcludedConvNode(_ClosedModel):
    source_node_id: str
    reason: Literal[
        "duplicate_node_id",
        "missing_static_tensor_shape",
        "non_positive_or_symbolic_dimension",
        "rank_not_four",
        "unsupported_group",
        "malformed_attribute",
        "inconsistent_tensor_shapes",
        "unsafe_node_id",
    ]


class ClusterMembership(_ClosedModel):
    source_node_id: SafeId
    shape: ConvLayerShape
    medoid_ordinal: Literal[0, 1, 2]
    distance_numerator: NonnegativeStrictInt
    distance_denominator: PositiveStrictInt

    @model_validator(mode="after")
    def _canonical_fraction(self) -> "ClusterMembership":
        fraction = Fraction(self.distance_numerator, self.distance_denominator)
        if (
            fraction.numerator != self.distance_numerator
            or fraction.denominator != self.distance_denominator
        ):
            raise ValueError("membership distance must be a reduced fraction")
        return self


class PanelExtractionReceipt(_ClosedModel):
    schema_version: Literal[1] = 1
    extractor_id: Literal[EXTRACTOR_ID] = EXTRACTOR_ID
    extractor_version: Literal[EXTRACTOR_VERSION] = EXTRACTOR_VERSION
    extraction_definition_sha256: Literal[EXTRACTION_DEFINITION_SHA256] = (
        EXTRACTION_DEFINITION_SHA256
    )
    clustering_id: Literal[CLUSTERING_ID] = CLUSTERING_ID
    clustering_version: Literal[CLUSTERING_VERSION] = CLUSTERING_VERSION
    clustering_definition_sha256: Literal[CLUSTERING_DEFINITION_SHA256] = (
        CLUSTERING_DEFINITION_SHA256
    )
    onnx_library_version: Literal[PINNED_ONNX_VERSION] = PINNED_ONNX_VERSION
    network_id: Literal["resnet50", "googlenet", "yolov3"]
    role: Literal["calibration", "validation", "held_out_test"]
    source_asset_sha256: str
    onnx_ir_version: PositiveStrictInt
    onnx_opsets: tuple[str, ...]
    graph_node_count: PositiveStrictInt
    conv_node_count: PositiveStrictInt
    supported_conv_layer_count: PositiveStrictInt
    unique_supported_shape_count: PositiveStrictInt
    excluded_conv_nodes: tuple[ExcludedConvNode, ...]
    clustering_objective_numerator: NonnegativeStrictInt
    clustering_objective_denominator: PositiveStrictInt
    memberships: tuple[ClusterMembership, ...]
    panel_sha256: str

    @model_validator(mode="after")
    def _validate_counts_and_order(self) -> "PanelExtractionReceipt":
        if self.supported_conv_layer_count != len(self.memberships):
            raise ValueError("supported count must equal membership count")
        if self.conv_node_count != len(self.memberships) + len(
            self.excluded_conv_nodes
        ):
            raise ValueError("every Conv node must be supported or excluded")
        if self.graph_node_count < self.conv_node_count:
            raise ValueError("graph node count cannot be smaller than Conv count")
        membership_ids = tuple(item.source_node_id for item in self.memberships)
        if membership_ids != tuple(sorted(membership_ids)):
            raise ValueError("memberships must use canonical node-id order")
        excluded_keys = tuple(
            (item.source_node_id, item.reason) for item in self.excluded_conv_nodes
        )
        if excluded_keys != tuple(sorted(excluded_keys)):
            raise ValueError("excluded nodes must use canonical order")
        shape_records = {_shape_key(item.shape) for item in self.memberships}
        if self.unique_supported_shape_count != len(shape_records):
            raise ValueError("unique supported shape count mismatch")
        objective = Fraction(
            self.clustering_objective_numerator,
            self.clustering_objective_denominator,
        )
        if (
            objective.numerator != self.clustering_objective_numerator
            or objective.denominator != self.clustering_objective_denominator
        ):
            raise ValueError("clustering objective must be a reduced fraction")
        if len(self.panel_sha256) != 64:
            raise ValueError("panel digest must be SHA-256")
        return self


def canonical_extraction_receipt_bytes(receipt: PanelExtractionReceipt) -> bytes:
    if type(receipt) is not PanelExtractionReceipt:
        raise TypeError("receipt must be an exact PanelExtractionReceipt")
    return _canonical_bytes(receipt.model_dump(mode="python"))


def extraction_receipt_sha256(receipt: PanelExtractionReceipt) -> str:
    return hashlib.sha256(
        _RECEIPT_HASH_DOMAIN + canonical_extraction_receipt_bytes(receipt)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PanelExtractionResult:
    panel: NetworkLayerPanel
    receipt: PanelExtractionReceipt
    panel_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if type(self.panel) is not NetworkLayerPanel:
            raise TypeError("panel must be an exact NetworkLayerPanel")
        if type(self.receipt) is not PanelExtractionReceipt:
            raise TypeError("receipt must be an exact PanelExtractionReceipt")
        if self.panel_sha256 != panel_sha256(self.panel):
            raise ValueError("panel digest mismatch")
        if self.receipt.panel_sha256 != self.panel_sha256:
            raise ValueError("receipt does not bind the panel")
        if self.receipt_sha256 != extraction_receipt_sha256(self.receipt):
            raise ValueError("extraction receipt digest mismatch")


def _shape_key(shape: ConvLayerShape) -> tuple[int, ...]:
    return (
        shape.batch,
        shape.channels_in,
        shape.channels_out,
        shape.filter_height,
        shape.filter_width,
        shape.output_height,
        shape.output_width,
        shape.stride_height,
        shape.stride_width,
        shape.dilation_height,
        shape.dilation_width,
    )


def _shape_features(shape: ConvLayerShape) -> tuple[int, ...]:
    dimensions = _shape_key(shape)
    mac_count = (
        shape.batch
        * shape.channels_in
        * shape.channels_out
        * shape.filter_height
        * shape.filter_width
        * shape.output_height
        * shape.output_width
    )
    return (*dimensions, mac_count)


def _shape_distance(left: ConvLayerShape, right: ConvLayerShape) -> Fraction:
    return sum(
        (
            Fraction(abs(a - b), max(a, b))
            for a, b in zip(
                _shape_features(left),
                _shape_features(right),
                strict=True,
            )
        ),
        start=Fraction(0, 1),
    )


def _tensor_shapes(graph: Any) -> dict[str, tuple[int, ...] | None]:
    result: dict[str, tuple[int, ...] | None] = {}
    for value in (*graph.input, *graph.value_info, *graph.output):
        dimensions = tuple(
            int(item.dim_value) for item in value.type.tensor_type.shape.dim
        )
        shape = (
            dimensions if dimensions and all(item > 0 for item in dimensions) else None
        )
        if value.name in result and result[value.name] != shape:
            raise ValueError(f"conflicting tensor-shape declarations: {value.name}")
        result[value.name] = shape
    return result


def _attributes(node: Any, onnx_module: Any) -> dict[str, object]:
    result: dict[str, object] = {}
    for attribute in node.attribute:
        if attribute.name in result:
            raise ValueError("duplicate Conv attribute")
        value = onnx_module.helper.get_attribute_value(attribute)
        if isinstance(value, bytes):
            value = value.decode("ascii", errors="strict")
        if isinstance(value, list):
            value = tuple(value)
        result[attribute.name] = value
    return result


def _positive_pair(value: object, name: str) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise ValueError(f"{name} must be two positive integers")
    return value


def _extract_conv_shape(
    node: Any,
    shapes: dict[str, tuple[int, ...] | None],
    onnx_module: Any,
) -> ConvLayerShape:
    if len(node.input) < 2 or len(node.output) != 1:
        raise LookupError("missing_static_tensor_shape")
    input_shape = shapes.get(node.input[0])
    weight_shape = shapes.get(node.input[1])
    output_shape = shapes.get(node.output[0])
    if input_shape is None or weight_shape is None or output_shape is None:
        raise LookupError("missing_static_tensor_shape")
    if any(
        item <= 0
        for shape in (input_shape, weight_shape, output_shape)
        for item in shape
    ):
        raise LookupError("non_positive_or_symbolic_dimension")
    if any(len(shape) != 4 for shape in (input_shape, weight_shape, output_shape)):
        raise LookupError("rank_not_four")
    try:
        attributes = _attributes(node, onnx_module)
        group = attributes.get("group", 1)
        if type(group) is not int or group <= 0:
            raise ValueError("group must be a positive integer")
        if group != 1:
            raise LookupError("unsupported_group")
        strides = _positive_pair(attributes.get("strides", (1, 1)), "strides")
        dilations = _positive_pair(attributes.get("dilations", (1, 1)), "dilations")
        kernel_shape = _positive_pair(
            attributes.get("kernel_shape", tuple(weight_shape[2:])),
            "kernel_shape",
        )
    except LookupError:
        raise
    except (TypeError, ValueError, UnicodeError) as error:
        raise LookupError("malformed_attribute") from error

    if (
        input_shape[0] != output_shape[0]
        or input_shape[1] != weight_shape[1]
        or output_shape[1] != weight_shape[0]
        or kernel_shape != tuple(weight_shape[2:])
    ):
        raise LookupError("inconsistent_tensor_shapes")
    return ConvLayerShape(
        batch=input_shape[0],
        channels_in=input_shape[1],
        channels_out=output_shape[1],
        filter_height=kernel_shape[0],
        filter_width=kernel_shape[1],
        output_height=output_shape[2],
        output_width=output_shape[3],
        stride_height=strides[0],
        stride_width=strides[1],
        dilation_height=dilations[0],
        dilation_width=dilations[1],
    )


@dataclass(frozen=True, slots=True)
class _ShapeRecord:
    shape: ConvLayerShape
    source_node_ids: tuple[str, ...]

    @property
    def weight(self) -> int:
        return len(self.source_node_ids)


def _shape_records(
    layers: tuple[tuple[str, ConvLayerShape], ...],
) -> tuple[_ShapeRecord, ...]:
    grouped: dict[tuple[int, ...], list[tuple[str, ConvLayerShape]]] = {}
    for source_node_id, shape in layers:
        grouped.setdefault(_shape_key(shape), []).append((source_node_id, shape))
    return tuple(
        _ShapeRecord(
            shape=sorted(items, key=lambda item: item[0])[0][1],
            source_node_ids=tuple(sorted(item[0] for item in items)),
        )
        for _, items in sorted(grouped.items())
    )


def _exact_medoids(
    records: tuple[_ShapeRecord, ...],
) -> tuple[tuple[int, int, int], tuple[int, ...], Fraction]:
    if len(records) < MEDOID_COUNT:
        raise ValueError(
            "at least three operationally distinct Conv shapes are required"
        )
    distances = tuple(
        tuple(_shape_distance(left.shape, right.shape) for right in records)
        for left in records
    )
    best_medoids: tuple[int, int, int] | None = None
    best_assignments: tuple[int, ...] | None = None
    best_objective: Fraction | None = None
    for medoids in combinations(range(len(records)), MEDOID_COUNT):
        assignments = tuple(
            min(
                range(MEDOID_COUNT),
                key=lambda ordinal: distances[index][medoids[ordinal]],
            )
            for index in range(len(records))
        )
        objective = sum(
            (
                records[index].weight * distances[index][medoids[assignments[index]]]
                for index in range(len(records))
            ),
            start=Fraction(0, 1),
        )
        if best_objective is None or objective < best_objective:
            best_medoids = medoids
            best_assignments = assignments
            best_objective = objective
    assert best_medoids is not None
    assert best_assignments is not None
    assert best_objective is not None
    return best_medoids, best_assignments, best_objective


def _opsets(model: Any) -> tuple[str, ...]:
    records = tuple(
        sorted((str(item.domain), int(item.version)) for item in model.opset_import)
    )
    return tuple(f"{domain}:{version}" for domain, version in records)


def extract_network_panel(asset_path: Path, network_id: str) -> PanelExtractionResult:
    """Extract and exactly cluster one hash-pinned ONNX network.

    The function is deterministic, outcome-blind, and fail-closed.  It uses no
    Timeloop objective and performs no model/provider call.
    """

    if not isinstance(asset_path, Path):
        raise TypeError("asset_path must be a pathlib.Path")
    if network_id not in NETWORK_ASSETS:
        raise ValueError("network_id is not in the frozen v2 split")
    source_digest = verify_network_asset(asset_path, network_id)
    try:
        import onnx  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            f"panel extraction requires onnx=={PINNED_ONNX_VERSION}"
        ) from error
    if onnx.__version__ != PINNED_ONNX_VERSION:
        raise RuntimeError(
            f"panel extraction requires onnx=={PINNED_ONNX_VERSION}; "
            f"observed {onnx.__version__}"
        )
    model = onnx.load(asset_path, load_external_data=False)
    shapes = _tensor_shapes(model.graph)

    conv_nodes = tuple(node for node in model.graph.node if node.op_type == "Conv")
    seen_ids: set[str] = set()
    layers: list[tuple[str, ConvLayerShape]] = []
    excluded: list[ExcludedConvNode] = []
    for ordinal, node in enumerate(conv_nodes):
        source_node_id = str(node.name)
        if not source_node_id:
            source_node_id = f"unnamed_conv_{ordinal}"
        if source_node_id in seen_ids:
            excluded.append(
                ExcludedConvNode(
                    source_node_id=source_node_id,
                    reason="duplicate_node_id",
                )
            )
            continue
        seen_ids.add(source_node_id)
        try:
            # Validate against the same closed ID grammar as LayerMedoid.
            LayerMedoid(
                source_node_id=source_node_id,
                shape=ConvLayerShape(
                    channels_in=1,
                    channels_out=1,
                    filter_height=1,
                    filter_width=1,
                    output_height=1,
                    output_width=1,
                ),
                multiplicity=1,
            )
        except Exception:
            excluded.append(
                ExcludedConvNode(
                    source_node_id=source_node_id,
                    reason="unsafe_node_id",
                )
            )
            continue
        try:
            shape = _extract_conv_shape(node, shapes, onnx)
        except LookupError as error:
            reason = str(error)
            excluded.append(
                ExcludedConvNode(source_node_id=source_node_id, reason=reason)
            )
            continue
        layers.append((source_node_id, shape))

    canonical_layers = tuple(sorted(layers, key=lambda item: item[0]))
    records = _shape_records(canonical_layers)
    medoid_indices, assignments, objective = _exact_medoids(records)
    cluster_multiplicities = tuple(
        sum(
            records[index].weight
            for index, assignment in enumerate(assignments)
            if assignment == ordinal
        )
        for ordinal in range(MEDOID_COUNT)
    )
    medoids = tuple(
        LayerMedoid(
            source_node_id=records[index].source_node_ids[0],
            shape=records[index].shape,
            multiplicity=cluster_multiplicities[ordinal],
        )
        for ordinal, index in enumerate(medoid_indices)
    )
    asset = NETWORK_ASSETS[network_id]
    panel = NetworkLayerPanel(
        panel_id=(f"timeloop.v2.{network_id}.conv-medoid3.exact-relative-l1.v1"),
        network_id=network_id,
        role=asset.role,
        source_asset_sha256=source_digest,
        extraction_definition_sha256=EXTRACTION_DEFINITION_SHA256,
        clustering_definition_sha256=CLUSTERING_DEFINITION_SHA256,
        supported_conv_layer_count=len(canonical_layers),
        medoid_0=medoids[0],
        medoid_1=medoids[1],
        medoid_2=medoids[2],
    )
    digest = panel_sha256(panel)

    record_by_key = {
        _shape_key(record.shape): index for index, record in enumerate(records)
    }
    memberships = tuple(
        ClusterMembership(
            source_node_id=source_node_id,
            shape=shape,
            medoid_ordinal=assignments[record_by_key[_shape_key(shape)]],
            distance_numerator=_shape_distance(
                shape,
                records[
                    medoid_indices[assignments[record_by_key[_shape_key(shape)]]]
                ].shape,
            ).numerator,
            distance_denominator=_shape_distance(
                shape,
                records[
                    medoid_indices[assignments[record_by_key[_shape_key(shape)]]]
                ].shape,
            ).denominator,
        )
        for source_node_id, shape in canonical_layers
    )
    receipt = PanelExtractionReceipt(
        network_id=network_id,
        role=asset.role,
        source_asset_sha256=source_digest,
        onnx_ir_version=int(model.ir_version),
        onnx_opsets=_opsets(model),
        graph_node_count=len(model.graph.node),
        conv_node_count=len(conv_nodes),
        supported_conv_layer_count=len(canonical_layers),
        unique_supported_shape_count=len(records),
        excluded_conv_nodes=tuple(
            sorted(excluded, key=lambda item: (item.source_node_id, item.reason))
        ),
        clustering_objective_numerator=objective.numerator,
        clustering_objective_denominator=objective.denominator,
        memberships=memberships,
        panel_sha256=digest,
    )
    return PanelExtractionResult(
        panel=panel,
        receipt=receipt,
        panel_sha256=digest,
        receipt_sha256=extraction_receipt_sha256(receipt),
    )


__all__ = [
    "CLUSTERING_DEFINITION_SHA256",
    "CLUSTERING_ID",
    "CLUSTERING_VERSION",
    "EXTRACTION_DEFINITION_SHA256",
    "EXTRACTOR_ID",
    "EXTRACTOR_VERSION",
    "MEDOID_COUNT",
    "PINNED_ONNX_VERSION",
    "ClusterMembership",
    "ExcludedConvNode",
    "PanelExtractionReceipt",
    "PanelExtractionResult",
    "canonical_extraction_receipt_bytes",
    "extract_network_panel",
    "extraction_receipt_sha256",
]
