"""Frozen Airfoil-v7 shape, trim, and union finite-variation catalogs."""

from __future__ import annotations

from itertools import product
import hashlib
import json

from agent_evolve.agentic import (
    FiniteVariationOption,
    FrozenJsonObject,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from examples.benchmarks.engibench_airfoil.problem_def import normalize_candidate
from examples.benchmarks.engibench_airfoil.v7_contract import TASK_SHA256


SHAPE_MODES = (
    "camber_front",
    "camber_aft",
    "thickness_front",
    "thickness_aft",
)
SHAPE_AMPLITUDES = (-0.0030, -0.0015, 0.0015, 0.0030)
TRIM_DELTAS_DEG = (-0.50, -0.25, 0.25, 0.50)

_SHAPE_INDICES = {
    "camber_front": (1, 2, 3, 4),
    "camber_aft": (5, 6, 7, 8),
    "thickness_front": (1, 2, 3, 4),
    "thickness_aft": (5, 6, 7, 8),
}
_SHAPE_AMPLITUDE_IDS = {
    -0.0030: "n0030",
    -0.0015: "n0015",
    0.0015: "p0015",
    0.0030: "p0030",
}
_TRIM_DELTA_IDS = {
    -0.50: "n050",
    -0.25: "n025",
    0.25: "p025",
    0.50: "p050",
}

PRESENTATION_POLICY_ID = "airfoil_v7_task_keyed_sha256"
PRESENTATION_POLICY_VERSION = 1
_PRESENTATION_KEY_DOMAIN = b"agent-evolve:airfoil-v7-presentation-key:v1\x00"


def task_keyed_presentation_sha256(
    *,
    task_sha256: str,
    catalog_id: str,
    family: str,
    option_id: str,
) -> str:
    """Return the frozen, task-keyed presentation key for one opaque option.

    The task digest is deliberately part of presentation only. It does not
    alter the option vocabulary or any engine-owned child configuration.
    """

    if (
        type(task_sha256) is not str
        or len(task_sha256) != 64
        or any(character not in "0123456789abcdef" for character in task_sha256)
    ):
        raise ValueError("task_sha256 must be one lowercase SHA-256 digest")
    values = (catalog_id, family, option_id)
    if any(type(value) is not str or not value for value in values):
        raise ValueError("catalog_id, family, and option_id must be non-empty")
    payload = b"\x00".join(
        value.encode("ascii", errors="strict")
        for value in (task_sha256, catalog_id, family, option_id)
    )
    return hashlib.sha256(_PRESENTATION_KEY_DOMAIN + payload).hexdigest()


def _present(
    catalog_id: str,
    options: tuple[FiniteVariationOption, ...],
) -> tuple[FiniteVariationOption, ...]:
    return tuple(
        sorted(
            options,
            key=lambda option: (
                task_keyed_presentation_sha256(
                    task_sha256=TASK_SHA256,
                    catalog_id=catalog_id,
                    family=option.family,
                    option_id=option.option_id,
                ),
                option.option_id,
            ),
        )
    )


def _presentation_definition(catalog_id: str) -> dict[str, object]:
    return {
        "catalog_id": catalog_id,
        "domain_ascii": _PRESENTATION_KEY_DOMAIN.decode("ascii"),
        "key_fields": (
            "task_sha256",
            "catalog_id",
            "family",
            "option_id",
        ),
        "policy_id": PRESENTATION_POLICY_ID,
        "policy_version": PRESENTATION_POLICY_VERSION,
        "task_sha256": TASK_SHA256,
        "task_sha256_encoding": "lowercase_hex_ascii",
        "tie_breaker": "option_id_ascii",
    }


def _definition_hash(domain: bytes, value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(domain + encoded).hexdigest()


_SHAPE_DEFINITION = {
    "kind": "parent_relative_shape_mode",
    "mode_order": SHAPE_MODES,
    "amplitude_order": SHAPE_AMPLITUDES,
    "indices": _SHAPE_INDICES,
    "upper_delta": "+a_for_every_mode",
    "lower_delta": {
        "camber_front": "+a",
        "camber_aft": "+a",
        "thickness_front": "-a",
        "thickness_aft": "-a",
    },
    "clipping": False,
    "out_of_bounds_child": "materialized_invalid_candidate",
    "presentation": _presentation_definition("airfoil_v7_shape"),
}
SHAPE_CATALOG_DEFINITION_SHA256 = _definition_hash(
    b"agent-evolve:airfoil-v7-shape-catalog:v2\x00",
    _SHAPE_DEFINITION,
)

_TRIM_DEFINITION = {
    "kind": "parent_relative_pointwise_trim_vector",
    "delta_order": TRIM_DELTAS_DEG,
    "cartesian_product_order": "point_0_major_then_point_1_then_point_2",
    "broadcasting": False,
    "clipping": False,
    "out_of_bounds_child": "materialized_invalid_candidate",
    "presentation": _presentation_definition("airfoil_v7_trim"),
}
TRIM_CATALOG_DEFINITION_SHA256 = _definition_hash(
    b"agent-evolve:airfoil-v7-trim-catalog:v2\x00",
    _TRIM_DEFINITION,
)

UNION_CATALOG_DEFINITION_SHA256 = _definition_hash(
    b"agent-evolve:airfoil-v7-union-catalog:v2\x00",
    {
        "option_set": "all_16_shape_union_all_64_trim",
        "presentation": _presentation_definition("airfoil_v7_union"),
        "shape_definition_sha256": SHAPE_CATALOG_DEFINITION_SHA256,
        "trim_definition_sha256": TRIM_CATALOG_DEFINITION_SHA256,
    },
)


def _parent(parent_configuration: FrozenJsonObject) -> tuple[dict, str]:
    if type(parent_configuration) is not FrozenJsonObject:
        raise TypeError("Airfoil finite catalog requires an exact FrozenJsonObject")
    parent = normalize_candidate(thaw_json(parent_configuration))
    return parent, typed_json_sha256(parent_configuration)


def _shape_options(
    parent_configuration: FrozenJsonObject,
) -> tuple[FiniteVariationOption, ...]:
    parent, parent_sha256 = _parent(parent_configuration)
    options: list[FiniteVariationOption] = []
    for mode in SHAPE_MODES:
        indices = _SHAPE_INDICES[mode]
        for amplitude in SHAPE_AMPLITUDES:
            child = {
                "representation_id": parent["representation_id"],
                "upper_coefficients": list(parent["upper_coefficients"]),
                "lower_coefficients": list(parent["lower_coefficients"]),
                "alpha_deg": list(parent["alpha_deg"]),
            }
            lower_delta = amplitude if mode.startswith("camber_") else -amplitude
            for index in indices:
                child["upper_coefficients"][index] += amplitude
                child["lower_coefficients"][index] += lower_delta
            amplitude_text = f"{amplitude:+.4f}"
            options.append(
                FiniteVariationOption(
                    option_id=f"shape.{mode}.{_SHAPE_AMPLITUDE_IDS[amplitude]}",
                    parent_configuration_sha256=parent_sha256,
                    child_configuration=freeze_json(child),
                    family="shape_only",
                    description=(
                        f"Apply {mode} with amplitude {amplitude_text} to the "
                        "frozen Bernstein coefficients."
                    ),
                    metadata=(
                        ("amplitude", amplitude_text),
                        ("coordinate_count", "8"),
                        ("mode", mode),
                    ),
                )
            )
    return tuple(options)


def _trim_options(
    parent_configuration: FrozenJsonObject,
) -> tuple[FiniteVariationOption, ...]:
    parent, parent_sha256 = _parent(parent_configuration)
    options: list[FiniteVariationOption] = []
    for deltas in product(TRIM_DELTAS_DEG, repeat=3):
        child = {
            "representation_id": parent["representation_id"],
            "upper_coefficients": list(parent["upper_coefficients"]),
            "lower_coefficients": list(parent["lower_coefficients"]),
            "alpha_deg": [
                value + delta
                for value, delta in zip(parent["alpha_deg"], deltas, strict=True)
            ],
        }
        delta_text = ",".join(f"{value:+.2f}" for value in deltas)
        delta_id = ".".join(_TRIM_DELTA_IDS[value] for value in deltas)
        options.append(
            FiniteVariationOption(
                option_id=f"trim.{delta_id}",
                parent_configuration_sha256=parent_sha256,
                child_configuration=freeze_json(child),
                family="trim_only",
                description=(
                    f"Add pointwise angle deltas ({delta_text}) degrees without "
                    "broadcasting."
                ),
                metadata=(("delta_alpha_deg", delta_text),),
            )
        )
    return tuple(options)


class AirfoilV7ShapeVariationCatalog:
    catalog_id = "airfoil_v7_shape"
    catalog_version = 2
    definition_sha256 = SHAPE_CATALOG_DEFINITION_SHA256
    option_families = ("shape_only",)

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        return _present(
            AirfoilV7ShapeVariationCatalog.catalog_id,
            _shape_options(parent_configuration),
        )


class AirfoilV7TrimVariationCatalog:
    catalog_id = "airfoil_v7_trim"
    catalog_version = 2
    definition_sha256 = TRIM_CATALOG_DEFINITION_SHA256
    option_families = ("trim_only",)

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        return _present(
            AirfoilV7TrimVariationCatalog.catalog_id,
            _trim_options(parent_configuration),
        )


class AirfoilV7UnionVariationCatalog:
    catalog_id = "airfoil_v7_union"
    catalog_version = 2
    definition_sha256 = UNION_CATALOG_DEFINITION_SHA256
    option_families = ("shape_only", "trim_only")

    @staticmethod
    def options(
        parent_configuration: FrozenJsonObject,
    ) -> tuple[FiniteVariationOption, ...]:
        return _present(
            AirfoilV7UnionVariationCatalog.catalog_id,
            _shape_options(parent_configuration) + _trim_options(parent_configuration),
        )


__all__ = [
    "AirfoilV7ShapeVariationCatalog",
    "AirfoilV7TrimVariationCatalog",
    "AirfoilV7UnionVariationCatalog",
    "PRESENTATION_POLICY_ID",
    "PRESENTATION_POLICY_VERSION",
    "SHAPE_AMPLITUDES",
    "SHAPE_CATALOG_DEFINITION_SHA256",
    "SHAPE_MODES",
    "TRIM_CATALOG_DEFINITION_SHA256",
    "TRIM_DELTAS_DEG",
    "UNION_CATALOG_DEFINITION_SHA256",
    "task_keyed_presentation_sha256",
]
