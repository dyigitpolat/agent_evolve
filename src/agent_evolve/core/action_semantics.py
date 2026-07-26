"""Immutable, prompt-safe semantics for benchmark-owned action axes.

Finite variation catalogs remain the executable authority for which children an
optimizer may evaluate.  This module supplies only a descriptive, hash-bound
companion: benchmarks can state what action coordinates mean, how coordinates
are coupled, and which tempting interpretations are explicitly incorrect.

The value layer deliberately accepts catalog identities and option-family
vocabularies as primitives.  It therefore has no dependency on a particular
variation implementation and can be bound by any benchmark adapter at its
composition boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from agent_evolve.domain.patch import require_sha256


CatalogIdentity = tuple[str, int, str]

_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_CANONICAL_JSON_PATH = re.compile(
    r"^\$\.[^.\[\]\s]+(?:\.[^.\[\]\s]+|\[(?:0|[1-9][0-9]*)\])*$"
)
_MAX_PROMPT_TEXT_UTF8_BYTES = 4_096
_SEMANTICS_HASH_DOMAIN = b"agent-evolve:action-space-semantics:v1\x00"


def _token(value: object, name: str) -> str:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed lowercase token grammar")
    return value


def _prompt_text(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be canonical non-empty text")
    if len(value.encode("utf-8", errors="strict")) > _MAX_PROMPT_TEXT_UTF8_BYTES:
        raise ValueError(f"{name} exceeds its UTF-8 byte limit")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _canonical_tokens(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if type(values) is not tuple or any(type(value) is not str for value in values):
        raise TypeError(f"{name} must be an exact tuple of strings")
    if not values:
        raise ValueError(f"{name} must be non-empty")
    for value in values:
        _token(value, f"{name} entry")
    canonical = tuple(sorted(set(values)))
    if values != canonical:
        raise ValueError(f"{name} must be unique and canonically sorted")
    return values


def _canonical_paths(values: tuple[str, ...]) -> tuple[str, ...]:
    if type(values) is not tuple or any(type(value) is not str for value in values):
        raise TypeError("configuration_paths must be an exact tuple of strings")
    if not values:
        raise ValueError("configuration_paths must be non-empty")
    if any(_CANONICAL_JSON_PATH.fullmatch(value) is None for value in values):
        raise ValueError(
            "configuration_paths must contain canonical JSON paths rooted at '$.'"
        )
    canonical = tuple(sorted(set(values)))
    if values != canonical:
        raise ValueError(
            "configuration_paths must be unique and canonically sorted"
        )
    return values


def _canonical_exclusions(values: tuple[str, ...]) -> tuple[str, ...]:
    if type(values) is not tuple or any(type(value) is not str for value in values):
        raise TypeError("excluded_interpretations must be an exact tuple of strings")
    for index, value in enumerate(values):
        _prompt_text(value, f"excluded_interpretations[{index}]")
    canonical = tuple(sorted(set(values)))
    if values != canonical:
        raise ValueError(
            "excluded_interpretations must be unique and canonically sorted"
        )
    return values


def _catalog_identity(value: object, name: str) -> CatalogIdentity:
    if type(value) is not tuple or len(value) != 3:
        raise TypeError(f"{name} must be an exact (catalog_id, version, hash) tuple")
    catalog_id, version, definition_sha256 = value
    _token(catalog_id, f"{name} catalog_id")
    if type(version) is not int or version <= 0:
        raise ValueError(f"{name} version must be a positive exact integer")
    require_sha256(definition_sha256, f"{name} definition_sha256")
    return catalog_id, version, definition_sha256


def _canonical_catalog_identities(
    values: tuple[CatalogIdentity, ...],
    *,
    require_canonical_order: bool,
) -> tuple[CatalogIdentity, ...]:
    if type(values) is not tuple:
        raise TypeError("catalog_identities must be an exact tuple")
    if not values:
        raise ValueError("catalog_identities must be non-empty")
    validated = tuple(
        _catalog_identity(value, f"catalog_identities[{index}]")
        for index, value in enumerate(values)
    )
    catalog_ids = tuple(value[0] for value in validated)
    if len(set(catalog_ids)) != len(catalog_ids):
        raise ValueError("catalog_identities cannot repeat a catalog_id")
    canonical = tuple(sorted(validated, key=lambda value: value[0]))
    if require_canonical_order and validated != canonical:
        raise ValueError("catalog_identities must use canonical catalog_id order")
    return canonical


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class ActionAxisCoordinateSemantics:
    """Meaning of one ordered coordinate within a benchmark action axis."""

    index: int
    label: str
    definition: str

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("coordinate index must be a non-negative exact integer")
        _prompt_text(self.label, "coordinate label")
        _prompt_text(self.definition, "coordinate definition")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "index": self.index,
            "label": self.label,
            "definition": self.definition,
        }


@dataclass(frozen=True, slots=True)
class ActionAxisSemantics:
    """Prompt-visible meaning and coupling rules for one action-space axis."""

    axis_id: str
    configuration_paths: tuple[str, ...]
    option_families: tuple[str, ...]
    definition: str
    independence: str
    unit: str | None = None
    coordinates: tuple[ActionAxisCoordinateSemantics, ...] = ()
    excluded_interpretations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _token(self.axis_id, "axis_id")
        _canonical_paths(self.configuration_paths)
        _canonical_tokens(self.option_families, "option_families")
        _prompt_text(self.definition, "axis definition")
        _prompt_text(self.independence, "axis independence")
        if self.unit is not None:
            _prompt_text(self.unit, "axis unit")
        if type(self.coordinates) is not tuple or any(
            type(value) is not ActionAxisCoordinateSemantics
            for value in self.coordinates
        ):
            raise TypeError(
                "coordinates must be an exact tuple of "
                "ActionAxisCoordinateSemantics values"
            )
        for coordinate in self.coordinates:
            ActionAxisCoordinateSemantics.__post_init__(coordinate)
        indices = tuple(value.index for value in self.coordinates)
        if indices and indices != tuple(range(len(indices))):
            raise ValueError(
                "coordinate indices must be contiguous and ordered from zero"
            )
        labels = tuple(value.label for value in self.coordinates)
        if len(set(labels)) != len(labels):
            raise ValueError("coordinate labels must be unique within an axis")
        _canonical_exclusions(self.excluded_interpretations)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "axis_id": self.axis_id,
            "configuration_paths": list(self.configuration_paths),
            "option_families": list(self.option_families),
            "definition": self.definition,
            "independence": self.independence,
            "unit": self.unit,
            "coordinates": [value.to_record() for value in self.coordinates],
            "excluded_interpretations": list(self.excluded_interpretations),
        }


@dataclass(frozen=True, slots=True)
class ActionSpaceSemantics:
    """Versioned action glossary bound to exact finite-catalog identities."""

    semantics_id: str
    semantics_version: int
    catalog_identities: tuple[CatalogIdentity, ...]
    axes: tuple[ActionAxisSemantics, ...]
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _token(self.semantics_id, "semantics_id")
        if type(self.semantics_version) is not int or self.semantics_version <= 0:
            raise ValueError(
                "semantics_version must be a positive exact integer"
            )
        _canonical_catalog_identities(
            self.catalog_identities,
            require_canonical_order=True,
        )
        if type(self.axes) is not tuple or not self.axes:
            raise ValueError("axes must be a non-empty exact tuple")
        if any(type(value) is not ActionAxisSemantics for value in self.axes):
            raise TypeError("axes must contain exact ActionAxisSemantics values")
        for axis in self.axes:
            ActionAxisSemantics.__post_init__(axis)
        axis_ids = tuple(value.axis_id for value in self.axes)
        if axis_ids != tuple(sorted(set(axis_ids))):
            raise ValueError("axes must use unique canonical axis_id order")
        paths = tuple(
            path for axis in self.axes for path in axis.configuration_paths
        )
        if len(paths) != len(set(paths)):
            raise ValueError(
                "configuration paths cannot be described by multiple action axes"
            )
        object.__setattr__(
            self,
            "definition_sha256",
            hashlib.sha256(
                _SEMANTICS_HASH_DOMAIN + _canonical_json(self._definition_record())
            ).hexdigest(),
        )

    @property
    def declared_option_families(self) -> tuple[str, ...]:
        """Return the complete family vocabulary covered by at least one axis."""

        return tuple(
            sorted(
                {
                    family
                    for axis in self.axes
                    for family in axis.option_families
                }
            )
        )

    def _definition_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "semantics_id": self.semantics_id,
            "semantics_version": self.semantics_version,
            "catalog_identities": [
                {
                    "catalog_id": catalog_id,
                    "catalog_version": version,
                    "definition_sha256": definition_sha256,
                }
                for catalog_id, version, definition_sha256 in self.catalog_identities
            ],
            "declared_option_families": list(self.declared_option_families),
            "axes": [axis.to_record() for axis in self.axes],
        }

    @property
    def identity(self) -> tuple[str, int, str]:
        self.__post_init__()
        return self.semantics_id, self.semantics_version, self.definition_sha256

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            **self._definition_record(),
            "definition_sha256": self.definition_sha256,
        }

    def validate_catalog_binding(
        self,
        catalog_identities: tuple[CatalogIdentity, ...],
        option_families: tuple[str, ...],
    ) -> None:
        """Bind prose to exact catalogs and their complete family vocabulary.

        Catalog declaration order is not semantically meaningful at this seam,
        but every ID, version, and definition hash must match.  Likewise, axes
        must cover every executable family and may not invent a prose-only one.
        """

        self.__post_init__()
        observed_catalogs = _canonical_catalog_identities(
            catalog_identities,
            require_canonical_order=False,
        )
        if observed_catalogs != self.catalog_identities:
            raise ValueError(
                "action semantics catalog identities differ from the bound catalogs"
            )
        if type(option_families) is not tuple or not option_families:
            raise ValueError("option_families must be a non-empty exact tuple")
        if any(type(value) is not str for value in option_families):
            raise TypeError("option_families must contain exact strings")
        for value in option_families:
            _token(value, "option_families entry")
        observed_families = tuple(sorted(set(option_families)))
        if observed_families != self.declared_option_families:
            missing = tuple(
                sorted(set(observed_families) - set(self.declared_option_families))
            )
            prose_only = tuple(
                sorted(set(self.declared_option_families) - set(observed_families))
            )
            details = []
            if missing:
                details.append("uncovered executable families=" + ",".join(missing))
            if prose_only:
                details.append(
                    "non-executable declared families=" + ",".join(prose_only)
                )
            raise ValueError(
                "action semantics option-family coverage differs from the bound "
                "catalogs: " + "; ".join(details)
            )

    def validate_contract_binding(
        self,
        catalog_identity: CatalogIdentity,
        option_families: tuple[str, ...],
    ) -> None:
        """Validate one parent-bound contract against this larger action space.

        Forecast requests commonly carry one catalog contract while a benchmark
        owns semantics for several catalogs, including union catalogs.  This
        narrower check therefore requires the contract's exact catalog identity
        to be a member of the semantics snapshot and every materialized option
        family to be described, without incorrectly requiring one contract to
        exercise the benchmark's complete family vocabulary.
        """

        self.__post_init__()
        observed_catalog = _catalog_identity(
            catalog_identity,
            "catalog_identity",
        )
        if observed_catalog not in self.catalog_identities:
            raise ValueError(
                "finite contract catalog identity is absent from action semantics"
            )
        if type(option_families) is not tuple or not option_families:
            raise ValueError("option_families must be a non-empty exact tuple")
        if any(type(value) is not str for value in option_families):
            raise TypeError("option_families must contain exact strings")
        for value in option_families:
            _token(value, "option_families entry")
        foreign = tuple(
            sorted(set(option_families) - set(self.declared_option_families))
        )
        if foreign:
            raise ValueError(
                "finite contract contains option families absent from action "
                "semantics: " + ",".join(foreign)
            )


def render_action_space_semantics(semantics: ActionSpaceSemantics) -> str:
    """Render the canonical authoritative block for provider prompts."""

    if type(semantics) is not ActionSpaceSemantics:
        raise TypeError("semantics must be an exact ActionSpaceSemantics")
    ActionSpaceSemantics.__post_init__(semantics)
    payload = _canonical_json(semantics.to_record()).decode("ascii")
    return "\n".join(
        (
            "ACTION-SPACE SEMANTICS (VERSIONED, AUTHORITATIVE)",
            "Use these exact coordinate meanings and coupling rules; do not "
            "infer geometry, topology, time, or spatial meaning from names.",
            payload,
        )
    )


__all__ = [
    "ActionAxisCoordinateSemantics",
    "ActionAxisSemantics",
    "ActionSpaceSemantics",
    "CatalogIdentity",
    "render_action_space_semantics",
]
