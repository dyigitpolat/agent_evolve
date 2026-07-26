"""Frozen network-medoid instance types and local ONNX provenance checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HASH_DOMAIN = b"agent-evolve:timeloop-network-medoid-panel:v1\x00"

SafeId = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=192,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$",
    ),
]
Sha256Hex = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
PositiveStrictInt = Annotated[int, Field(strict=True, ge=1)]


class _ClosedModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class NetworkAsset(_ClosedModel):
    filename: SafeId
    sha256: Sha256Hex
    role: Literal["calibration", "validation", "held_out_test"]


NETWORK_ASSETS: dict[str, NetworkAsset] = {
    "resnet50": NetworkAsset(
        filename="resnet50.onnx",
        sha256="f05ceac0a3461de57db1b13b11456d294ff6b3c00057cf66f3347e034090244d",
        role="calibration",
    ),
    "googlenet": NetworkAsset(
        filename="googlenet.onnx",
        sha256="f321864a4edce8fe29b8fff25e0028ee16c337f6fe0c8742e14f4479802c1155",
        role="validation",
    ),
    "yolov3": NetworkAsset(
        filename="yolov3.onnx",
        sha256="f3296442e88e38f7abe9e5dc586277d4a275ec86b6cf42b09e3f5ccb6a59abc5",
        role="held_out_test",
    ),
}


class ConvLayerShape(_ClosedModel):
    """One canonical Timeloop convolution instance."""

    batch: PositiveStrictInt = 1
    channels_in: PositiveStrictInt
    channels_out: PositiveStrictInt
    filter_height: PositiveStrictInt
    filter_width: PositiveStrictInt
    output_height: PositiveStrictInt
    output_width: PositiveStrictInt
    stride_height: PositiveStrictInt = 1
    stride_width: PositiveStrictInt = 1
    dilation_height: PositiveStrictInt = 1
    dilation_width: PositiveStrictInt = 1

    def timeloop_instance(self) -> dict[str, int]:
        return {
            "N": self.batch,
            "C": self.channels_in,
            "M": self.channels_out,
            "R": self.filter_height,
            "S": self.filter_width,
            "P": self.output_height,
            "Q": self.output_width,
            "Hstride": self.stride_height,
            "Wstride": self.stride_width,
            "Hdilation": self.dilation_height,
            "Wdilation": self.dilation_width,
        }

    def spatial_extent(self, axis: Literal["C", "M", "P", "Q"]) -> int:
        return self.timeloop_instance()[axis]


class LayerMedoid(_ClosedModel):
    source_node_id: SafeId
    shape: ConvLayerShape
    multiplicity: PositiveStrictInt


class NetworkLayerPanel(_ClosedModel):
    """Three outcome-blind medoids and exact cluster multiplicities."""

    schema_version: Literal[1] = 1
    panel_id: SafeId
    network_id: Literal["resnet50", "googlenet", "yolov3"]
    role: Literal["calibration", "validation", "held_out_test"]
    source_asset_sha256: Sha256Hex
    extraction_definition_sha256: Sha256Hex
    clustering_definition_sha256: Sha256Hex
    supported_conv_layer_count: PositiveStrictInt
    medoid_0: LayerMedoid
    medoid_1: LayerMedoid
    medoid_2: LayerMedoid

    @model_validator(mode="after")
    def _validate_frozen_instance(self) -> "NetworkLayerPanel":
        asset = NETWORK_ASSETS[self.network_id]
        if self.source_asset_sha256 != asset.sha256:
            raise ValueError("source_asset_sha256 does not match the frozen network")
        if self.role != asset.role:
            raise ValueError("network role does not match the frozen split")
        medoids = self.medoids()
        if (
            sum(item.multiplicity for item in medoids)
            != self.supported_conv_layer_count
        ):
            raise ValueError(
                "medoid multiplicities must cover every supported Conv layer"
            )
        operational = {
            _canonical_bytes(
                {
                    "shape": item.shape.model_dump(mode="python"),
                    "multiplicity": item.multiplicity,
                }
            )
            for item in medoids
        }
        if len(operational) != 3:
            raise ValueError("medoid slots must be operationally distinct")
        if len({item.source_node_id for item in medoids}) != 3:
            raise ValueError("medoid source nodes must be distinct")
        return self

    def medoids(self) -> tuple[LayerMedoid, LayerMedoid, LayerMedoid]:
        return (self.medoid_0, self.medoid_1, self.medoid_2)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def canonical_panel_bytes(value: object) -> bytes:
    panel = (
        value
        if isinstance(value, NetworkLayerPanel)
        else NetworkLayerPanel.model_validate(value, strict=True)
    )
    return _canonical_bytes(panel.model_dump(mode="python"))


def panel_sha256(value: object) -> str:
    return hashlib.sha256(_HASH_DOMAIN + canonical_panel_bytes(value)).hexdigest()


def verify_network_asset(path: Path, network_id: str) -> str:
    """Fail closed unless one local asset has the artifact-188 identity."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if network_id not in NETWORK_ASSETS:
        raise ValueError("network_id is not in the frozen v2 split")
    expected = NETWORK_ASSETS[network_id]
    if path.name != expected.filename:
        raise ValueError("asset filename does not match the frozen network")
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if _SHA256.fullmatch(observed) is None or observed != expected.sha256:
        raise ValueError("local ONNX asset hash does not match artifact 188")
    return observed


__all__ = [
    "NETWORK_ASSETS",
    "ConvLayerShape",
    "LayerMedoid",
    "NetworkAsset",
    "NetworkLayerPanel",
    "canonical_panel_bytes",
    "panel_sha256",
    "verify_network_asset",
]
