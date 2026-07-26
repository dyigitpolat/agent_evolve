"""Outcome-blind network panels frozen from the hash-pinned ONNX assets."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .network_panel import NetworkLayerPanel, panel_sha256


_PANEL_PAYLOADS: dict[str, dict[str, object]] = {
    "resnet50": {
        "panel_id": "timeloop.v2.resnet50.conv-medoid3.exact-relative-l1.v1",
        "network_id": "resnet50",
        "role": "calibration",
        "source_asset_sha256": (
            "f05ceac0a3461de57db1b13b11456d294ff6b3c00057cf66f3347e034090244d"
        ),
        "extraction_definition_sha256": (
            "a68a98366ce5225b09c3b15588aa5d09d15782cf35822af8f8c5dced70fcd4ff"
        ),
        "clustering_definition_sha256": (
            "01f0a427e0db44de3a7e2628b9c676b6ab618e37ddaca02edd367dcfbc4ee552"
        ),
        "supported_conv_layer_count": 53,
        "medoid_0": {
            "source_node_id": "Conv_15",
            "shape": {
                "batch": 1,
                "channels_in": 64,
                "channels_out": 256,
                "filter_height": 1,
                "filter_width": 1,
                "output_height": 56,
                "output_width": 56,
                "stride_height": 1,
                "stride_width": 1,
                "dilation_height": 1,
                "dilation_width": 1,
            },
            "multiplicity": 19,
        },
        "medoid_1": {
            "source_node_id": "Conv_64",
            "shape": {
                "batch": 1,
                "channels_in": 256,
                "channels_out": 256,
                "filter_height": 3,
                "filter_width": 3,
                "output_height": 14,
                "output_width": 14,
                "stride_height": 1,
                "stride_width": 1,
                "dilation_height": 1,
                "dilation_width": 1,
            },
            "multiplicity": 14,
        },
        "medoid_2": {
            "source_node_id": "Conv_58",
            "shape": {
                "batch": 1,
                "channels_in": 256,
                "channels_out": 1024,
                "filter_height": 1,
                "filter_width": 1,
                "output_height": 14,
                "output_width": 14,
                "stride_height": 1,
                "stride_width": 1,
                "dilation_height": 1,
                "dilation_width": 1,
            },
            "multiplicity": 20,
        },
    },
    "googlenet": {
        "panel_id": "timeloop.v2.googlenet.conv-medoid3.exact-relative-l1.v1",
        "network_id": "googlenet",
        "role": "validation",
        "source_asset_sha256": (
            "f321864a4edce8fe29b8fff25e0028ee16c337f6fe0c8742e14f4479802c1155"
        ),
        "extraction_definition_sha256": (
            "a68a98366ce5225b09c3b15588aa5d09d15782cf35822af8f8c5dced70fcd4ff"
        ),
        "clustering_definition_sha256": (
            "01f0a427e0db44de3a7e2628b9c676b6ab618e37ddaca02edd367dcfbc4ee552"
        ),
        "supported_conv_layer_count": 57,
        "medoid_0": {
            "source_node_id": "Conv_41",
            "shape": {
                "batch": 1,
                "channels_in": 96,
                "channels_out": 208,
                "filter_height": 3,
                "filter_width": 3,
                "output_height": 14,
                "output_width": 14,
                "stride_height": 1,
                "stride_width": 1,
                "dilation_height": 1,
                "dilation_width": 1,
            },
            "multiplicity": 18,
        },
        "medoid_1": {
            "source_node_id": "Conv_33",
            "shape": {
                "batch": 1,
                "channels_in": 256,
                "channels_out": 64,
                "filter_height": 1,
                "filter_width": 1,
                "output_height": 28,
                "output_width": 28,
                "stride_height": 1,
                "stride_width": 1,
                "dilation_height": 1,
                "dilation_width": 1,
            },
            "multiplicity": 11,
        },
        "medoid_2": {
            "source_node_id": "Conv_65",
            "shape": {
                "batch": 1,
                "channels_in": 512,
                "channels_out": 128,
                "filter_height": 1,
                "filter_width": 1,
                "output_height": 14,
                "output_width": 14,
                "stride_height": 1,
                "stride_width": 1,
                "dilation_height": 1,
                "dilation_width": 1,
            },
            "multiplicity": 28,
        },
    },
    "yolov3": {
        "panel_id": "timeloop.v2.yolov3.conv-medoid3.exact-relative-l1.v1",
        "network_id": "yolov3",
        "role": "held_out_test",
        "source_asset_sha256": (
            "f3296442e88e38f7abe9e5dc586277d4a275ec86b6cf42b09e3f5ccb6a59abc5"
        ),
        "extraction_definition_sha256": (
            "a68a98366ce5225b09c3b15588aa5d09d15782cf35822af8f8c5dced70fcd4ff"
        ),
        "clustering_definition_sha256": (
            "01f0a427e0db44de3a7e2628b9c676b6ab618e37ddaca02edd367dcfbc4ee552"
        ),
        "supported_conv_layer_count": 75,
        "medoid_0": {
            "source_node_id": "Conv_237",
            "shape": {
                "batch": 1,
                "channels_in": 128,
                "channels_out": 256,
                "filter_height": 3,
                "filter_width": 3,
                "output_height": 52,
                "output_width": 52,
                "stride_height": 1,
                "stride_width": 1,
                "dilation_height": 1,
                "dilation_width": 1,
            },
            "multiplicity": 18,
        },
        "medoid_1": {
            "source_node_id": "Conv_103",
            "shape": {
                "batch": 1,
                "channels_in": 256,
                "channels_out": 512,
                "filter_height": 3,
                "filter_width": 3,
                "output_height": 26,
                "output_width": 26,
                "stride_height": 1,
                "stride_width": 1,
                "dilation_height": 1,
                "dilation_width": 1,
            },
            "multiplicity": 20,
        },
        "medoid_2": {
            "source_node_id": "Conv_100",
            "shape": {
                "batch": 1,
                "channels_in": 512,
                "channels_out": 256,
                "filter_height": 1,
                "filter_width": 1,
                "output_height": 26,
                "output_width": 26,
                "stride_height": 1,
                "stride_width": 1,
                "dilation_height": 1,
                "dilation_width": 1,
            },
            "multiplicity": 37,
        },
    },
}

FROZEN_PANEL_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "resnet50": (
            "c6932b95c682ef80e33d702d18bb8f5a5e04e762a2ed02ebce95722b127a76ea"
        ),
        "googlenet": (
            "7a94282a58eeec54c1f773c6a477e6c2f9eb807485017b60774bfde4a0f94b91"
        ),
        "yolov3": ("8862fcc863e4c80d638b28ca106de9747b029b3a21078845a6a2ade3de89f383"),
    }
)

FROZEN_EXTRACTION_RECEIPT_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "resnet50": (
            "3c6e726e4f92cfd5e21e719fe9f5f6e74be81771631270cc8448ee6e72509ab5"
        ),
        "googlenet": (
            "f98be4831af79c07d9c35fa1a112956a40ee39a3dfb0d9b5e05364352cb57848"
        ),
        "yolov3": ("ab4a3cbbd9c58b0491ebe37271b074556126237bd69841f740282cec2eb0d91e"),
    }
)

_PANELS = {
    network_id: NetworkLayerPanel.model_validate(payload, strict=True)
    for network_id, payload in _PANEL_PAYLOADS.items()
}
for _network_id, _panel in _PANELS.items():
    if panel_sha256(_panel) != FROZEN_PANEL_SHA256[_network_id]:
        raise RuntimeError(f"frozen Timeloop panel digest drift: {_network_id}")

FROZEN_NETWORK_PANELS: Mapping[str, NetworkLayerPanel] = MappingProxyType(_PANELS)


def frozen_network_panel(network_id: str) -> NetworkLayerPanel:
    """Return one immutable, outcome-blind network panel by split ID."""

    if network_id not in FROZEN_NETWORK_PANELS:
        raise ValueError("network_id is not in the frozen v2 split")
    return FROZEN_NETWORK_PANELS[network_id]


__all__ = [
    "FROZEN_EXTRACTION_RECEIPT_SHA256",
    "FROZEN_NETWORK_PANELS",
    "FROZEN_PANEL_SHA256",
    "frozen_network_panel",
]
