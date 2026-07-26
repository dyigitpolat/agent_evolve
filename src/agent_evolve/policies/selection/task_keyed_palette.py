"""Task-keyed atomic palettes with complete, replayable selection facts.

The policy deliberately owns two choices which otherwise tend to leak into a
benchmark runner: which path to expose and which finite replacements to show.
An unforced path is selected from the minimum-exposure feasible paths with a
cryptographic task-key tie-break.  Within that exact path, required options are
included first as constraints, the remaining capacity is filled by exposure
and an independent task-key tie-break, and the final provider-visible order is
another independent task-key permutation.

No incoming catalog, mapping, or concurrent-completion order participates in
any decision.  The returned frozen value contains every path and option row
needed to audit or replay the choice.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Sequence

from agent_evolve.domain.patch import (
    ArrayIndex,
    JsonPath,
    ObjectKey,
    canonical_path_bytes,
    require_sha256,
    validate_json_path,
)
from agent_evolve.domain.typed_json import canonical_typed_json_bytes
from agent_evolve.domain.variation_space import (
    AtomicEditOption,
    validate_atomic_edit_option,
)


POLICY_ID = "task_keyed_atomic_palette"
POLICY_VERSION = 1
_TASK_ORDER_DOMAIN = b"agent-evolve:task-keyed-atomic-palette-order:v1\x00"
_CATALOG_DOMAIN = b"agent-evolve:task-keyed-atomic-palette-catalog:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:task-keyed-atomic-palette-decision:v1\x00"
_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_MAX_TASK_KEY_BYTES = 1024
_MAX_EXPOSURE = (1 << 63) - 1
_MAX_SEED = (1 << 64) - 1


def _frame(value: bytes) -> bytes:
    if type(value) is not bytes:
        raise TypeError("framed values must be exact bytes")
    return len(value).to_bytes(8, "big", signed=False) + value


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed <= _MAX_SEED:
        raise ValueError("seed must be an exact uint64 integer")


def _task_key_bytes(task_key: str) -> bytes:
    if type(task_key) is not str or not task_key or "\x00" in task_key:
        raise ValueError("task_key must be non-empty and NUL-free")
    try:
        encoded = task_key.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("task_key must be strict UTF-8") from exc
    if len(encoded) > _MAX_TASK_KEY_BYTES:
        raise ValueError("task_key exceeds its byte limit")
    return encoded


def _task_order_sha256(
    *,
    seed: int,
    task_key: str,
    purpose: str,
    payload: bytes,
) -> str:
    _validate_seed(seed)
    task_bytes = _task_key_bytes(task_key)
    if type(purpose) is not str or not purpose.isascii() or not purpose:
        raise ValueError("task-order purpose must be non-empty ASCII")
    digest = hashlib.sha256()
    digest.update(_TASK_ORDER_DOMAIN)
    digest.update(seed.to_bytes(8, "big", signed=False))
    digest.update(_frame(task_bytes))
    digest.update(_frame(purpose.encode("ascii")))
    digest.update(_frame(payload))
    return digest.hexdigest()


def _validate_family(value: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError("family must use the closed lowercase token grammar")


def _validate_count(value: int, *, name: str) -> None:
    if type(value) is not int or not 0 <= value <= _MAX_EXPOSURE:
        raise ValueError(f"{name} must be an exact non-negative int63")


def _path_record(path: JsonPath) -> dict[str, object]:
    validate_json_path(path)
    segments: list[dict[str, object]] = []
    for segment in path.segments:
        if type(segment) is ObjectKey:
            segments.append({"kind": "object_key", "value": segment.value})
        elif type(segment) is ArrayIndex:
            segments.append({"kind": "array_index", "value": segment.value})
        else:  # pragma: no cover - JsonPath validation closes the union.
            raise AssertionError("unsupported JSON-path segment")
    return {"schema_identity": path.schema_identity, "segments": segments}


def _option_record(option: AtomicEditOption) -> dict[str, object]:
    validate_atomic_edit_option(option)
    return {
        "option_id": option.option_id,
        "option_identity_sha256": option.identity_sha256,
        "path": _path_record(option.path),
        "replacement": option.replacement,
        "family": option.family,
        "metadata": [
            {
                "key": key,
                # Metadata may bind source commands or other artifact content;
                # its value belongs outside a durable inline event.  The exact
                # bytes remain transitively bound by option_identity_sha256.
                "value_sha256": hashlib.sha256(
                    value.encode("utf-8", errors="strict")
                ).hexdigest(),
            }
            for key, value in option.metadata
        ],
    }


def _canonical_catalog(
    options: Sequence[AtomicEditOption],
) -> tuple[AtomicEditOption, ...]:
    if isinstance(options, (str, bytes)):
        raise TypeError("options must be a finite option sequence")
    values = tuple(options)
    if not values:
        raise ValueError("options cannot be empty")
    for option in values:
        if type(option) is not AtomicEditOption:
            raise TypeError("options must contain exact AtomicEditOption values")
        validate_atomic_edit_option(option)

    option_ids = [option.option_id for option in values]
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("catalog option_id values must be globally unique")
    identities = [option.identity_sha256 for option in values]
    if len(set(identities)) != len(identities):
        raise ValueError("catalog option identities must be globally unique")

    replacements: set[tuple[bytes, bytes]] = set()
    for option in values:
        key = (
            canonical_path_bytes(option.path),
            canonical_typed_json_bytes(option.replacement),
        )
        if key in replacements:
            raise ValueError(
                "catalog cannot contain duplicate replacements at one path"
            )
        replacements.add(key)
    return tuple(
        sorted(
            values,
            key=lambda option: (
                canonical_path_bytes(option.path),
                option.option_id,
                option.identity_sha256,
            ),
        )
    )


def _catalog_sha256(options: tuple[AtomicEditOption, ...]) -> str:
    canonical = _canonical_catalog(options)
    digest = hashlib.sha256()
    digest.update(_CATALOG_DOMAIN)
    digest.update(len(canonical).to_bytes(8, "big", signed=False))
    for option in canonical:
        digest.update(bytes.fromhex(option.identity_sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class PathFamilyExposure:
    """An append-only count for one exact path/family exposure cell."""

    path: JsonPath
    family: str
    count: int

    def __post_init__(self) -> None:
        if type(self.path) is not JsonPath:
            raise TypeError("path must be an exact JsonPath")
        validate_json_path(self.path)
        _validate_family(self.family)
        _validate_count(self.count, name="exposure count")

    def revalidate(self) -> None:
        if type(self) is not PathFamilyExposure:
            raise TypeError("exposure must be an exact PathFamilyExposure")
        PathFamilyExposure.__post_init__(self)

    def _key(self) -> tuple[bytes, str, int]:
        self.revalidate()
        return canonical_path_bytes(self.path), self.family, self.count

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is PathFamilyExposure
            and type(other) is PathFamilyExposure
            and self._key() == other._key()
        )

    def __hash__(self) -> int:
        return hash((PathFamilyExposure, self._key()))

    def to_trace_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "path": _path_record(self.path),
            "family": self.family,
            "count": self.count,
        }


def _canonical_exposures(
    exposures: Sequence[PathFamilyExposure],
) -> tuple[PathFamilyExposure, ...]:
    if isinstance(exposures, (str, bytes)):
        raise TypeError("exposures must be a finite exposure sequence")
    values = tuple(exposures)
    for exposure in values:
        if type(exposure) is not PathFamilyExposure:
            raise TypeError("exposures must contain exact PathFamilyExposure values")
        exposure.revalidate()
    keys = [
        (canonical_path_bytes(exposure.path), exposure.family) for exposure in values
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("exposure cells must be unique")
    return tuple(
        sorted(
            values,
            key=lambda exposure: (
                canonical_path_bytes(exposure.path),
                exposure.family,
            ),
        )
    )


class PathSelectionMode(str, Enum):
    """Why the exact atomic path was fixed for this palette."""

    FORCED = "forced"
    REQUIRED_OPTIONS = "required_options"
    MINIMUM_EXPOSURE_TASK_KEYED = "minimum_exposure_task_keyed"


@dataclass(frozen=True, slots=True, eq=False)
class PalettePathRow:
    """Every fact used to admit and rank one catalog path."""

    path: JsonPath
    option_count: int
    family_capacity: int
    path_exposure: int
    feasible_for_size: bool
    eligible_for_choice: bool
    task_order_sha256: str

    def __post_init__(self) -> None:
        if type(self.path) is not JsonPath:
            raise TypeError("path must be an exact JsonPath")
        validate_json_path(self.path)
        if type(self.option_count) is not int or self.option_count <= 0:
            raise ValueError("option_count must be a positive exact integer")
        if (
            type(self.family_capacity) is not int
            or not 0 < self.family_capacity <= self.option_count
        ):
            raise ValueError("family_capacity must lie within the path's option count")
        _validate_count(self.path_exposure, name="path_exposure")
        if type(self.feasible_for_size) is not bool:
            raise TypeError("feasible_for_size must be an exact bool")
        if type(self.eligible_for_choice) is not bool:
            raise TypeError("eligible_for_choice must be an exact bool")
        if self.eligible_for_choice and not self.feasible_for_size:
            raise ValueError("an ineligible-size path cannot be chosen")
        require_sha256(self.task_order_sha256, "task_order_sha256")

    def revalidate(self) -> None:
        if type(self) is not PalettePathRow:
            raise TypeError("path row must be an exact PalettePathRow")
        PalettePathRow.__post_init__(self)

    def _key(self) -> tuple[bytes, int, int, int, bool, bool, str]:
        self.revalidate()
        return (
            canonical_path_bytes(self.path),
            self.option_count,
            self.family_capacity,
            self.path_exposure,
            self.feasible_for_size,
            self.eligible_for_choice,
            self.task_order_sha256,
        )

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is PalettePathRow
            and type(other) is PalettePathRow
            and self._key() == other._key()
        )

    __hash__ = None

    def to_trace_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            "path": _path_record(self.path),
            "option_count": self.option_count,
            "family_capacity": self.family_capacity,
            "path_exposure": self.path_exposure,
            "feasible_for_size": self.feasible_for_size,
            "eligible_for_choice": self.eligible_for_choice,
            "task_order_sha256": self.task_order_sha256,
        }


@dataclass(frozen=True, slots=True, eq=False)
class PaletteOptionRow:
    """Complete catalog row, including exposure, tie-break, and disposition."""

    option: AtomicEditOption
    cell_exposure: int
    family_exposure: int
    path_exposure: int
    selection_order_sha256: str
    presentation_order_sha256: str
    on_chosen_path: bool
    required: bool
    palette_position: int | None

    def __post_init__(self) -> None:
        if type(self.option) is not AtomicEditOption:
            raise TypeError("option must be an exact AtomicEditOption")
        validate_atomic_edit_option(self.option)
        _validate_count(self.cell_exposure, name="cell_exposure")
        _validate_count(self.family_exposure, name="family_exposure")
        _validate_count(self.path_exposure, name="path_exposure")
        require_sha256(self.selection_order_sha256, "selection_order_sha256")
        require_sha256(
            self.presentation_order_sha256,
            "presentation_order_sha256",
        )
        if type(self.on_chosen_path) is not bool:
            raise TypeError("on_chosen_path must be an exact bool")
        if type(self.required) is not bool:
            raise TypeError("required must be an exact bool")
        if self.required and not self.on_chosen_path:
            raise ValueError("required options must be on the chosen path")
        if self.palette_position is not None and (
            type(self.palette_position) is not int or self.palette_position < 0
        ):
            raise ValueError("palette_position must be None or non-negative")
        if self.palette_position is not None and not self.on_chosen_path:
            raise ValueError("selected options must be on the chosen path")
        if self.required and self.palette_position is None:
            raise ValueError("every required option must enter the palette")

    def revalidate(self) -> None:
        if type(self) is not PaletteOptionRow:
            raise TypeError("option row must be an exact PaletteOptionRow")
        PaletteOptionRow.__post_init__(self)

    def _key(self) -> tuple[object, ...]:
        self.revalidate()
        return (
            self.option.identity_sha256,
            self.cell_exposure,
            self.family_exposure,
            self.path_exposure,
            self.selection_order_sha256,
            self.presentation_order_sha256,
            self.on_chosen_path,
            self.required,
            self.palette_position,
        )

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is PaletteOptionRow
            and type(other) is PaletteOptionRow
            and self._key() == other._key()
        )

    __hash__ = None

    def to_trace_record(self) -> dict[str, object]:
        self.revalidate()
        return {
            **_option_record(self.option),
            "cell_exposure": self.cell_exposure,
            "family_exposure": self.family_exposure,
            "path_exposure": self.path_exposure,
            "selection_order_sha256": self.selection_order_sha256,
            "presentation_order_sha256": self.presentation_order_sha256,
            "on_chosen_path": self.on_chosen_path,
            "required": self.required,
            "palette_position": self.palette_position,
        }


@dataclass(frozen=True, slots=True)
class _PaletteParts:
    chosen_path: JsonPath
    path_selection_mode: PathSelectionMode
    exposure_snapshot: tuple[PathFamilyExposure, ...]
    path_rows: tuple[PalettePathRow, ...]
    option_rows: tuple[PaletteOptionRow, ...]
    palette: tuple[AtomicEditOption, ...]
    catalog_sha256: str


def _build_parts(
    *,
    seed: int,
    task_key: str,
    options: Sequence[AtomicEditOption],
    palette_size: int,
    max_options_per_family: int | None,
    exposures: Sequence[PathFamilyExposure],
    forced_path: JsonPath | None,
    required_option_ids: tuple[str, ...],
) -> _PaletteParts:
    _validate_seed(seed)
    _task_key_bytes(task_key)
    catalog = _canonical_catalog(options)
    exposure_snapshot = _canonical_exposures(exposures)
    if type(palette_size) is not int or palette_size <= 0:
        raise ValueError("palette_size must be a positive exact integer")
    if max_options_per_family is not None and (
        type(max_options_per_family) is not int or max_options_per_family <= 0
    ):
        raise ValueError(
            "max_options_per_family must be None or a positive exact integer"
        )
    if type(required_option_ids) is not tuple:
        raise TypeError("required_option_ids must be an exact tuple")
    if any(type(option_id) is not str for option_id in required_option_ids):
        raise TypeError("required_option_ids must contain exact strings")
    if tuple(sorted(required_option_ids)) != required_option_ids:
        raise ValueError("required_option_ids must use canonical sorted order")
    if len(set(required_option_ids)) != len(required_option_ids):
        raise ValueError("required_option_ids cannot contain duplicates")
    if len(required_option_ids) > palette_size:
        raise ValueError("required option count cannot exceed palette_size")
    if forced_path is not None:
        if type(forced_path) is not JsonPath:
            raise TypeError("forced_path must be an exact JsonPath or None")
        validate_json_path(forced_path)

    by_id = {option.option_id: option for option in catalog}
    missing = tuple(
        option_id for option_id in required_option_ids if option_id not in by_id
    )
    if missing:
        raise ValueError(
            f"required option IDs are absent from the catalog: {missing!r}"
        )
    required = tuple(by_id[option_id] for option_id in required_option_ids)
    required_paths = {canonical_path_bytes(option.path) for option in required}
    if len(required_paths) > 1:
        raise ValueError("all required options must share one exact path")
    required_path = required[0].path if required else None
    if (
        forced_path is not None
        and required_path is not None
        and forced_path != required_path
    ):
        raise ValueError("required options do not belong to the forced path")
    if max_options_per_family is not None:
        required_family_counts: dict[str, int] = {}
        for option in required:
            required_family_counts[option.family] = (
                required_family_counts.get(option.family, 0) + 1
            )
        if any(
            count > max_options_per_family for count in required_family_counts.values()
        ):
            raise ValueError("required options exceed max_options_per_family")

    groups: dict[bytes, list[AtomicEditOption]] = {}
    paths: dict[bytes, JsonPath] = {}
    for option in catalog:
        path_key = canonical_path_bytes(option.path)
        groups.setdefault(path_key, []).append(option)
        paths[path_key] = option.path

    exposure_by_cell = {
        (canonical_path_bytes(exposure.path), exposure.family): exposure.count
        for exposure in exposure_snapshot
    }
    path_exposure: dict[bytes, int] = {}
    family_exposure: dict[str, int] = {}
    for exposure in exposure_snapshot:
        path_key = canonical_path_bytes(exposure.path)
        path_exposure[path_key] = path_exposure.get(path_key, 0) + exposure.count
        family_exposure[exposure.family] = (
            family_exposure.get(exposure.family, 0) + exposure.count
        )
        _validate_count(path_exposure[path_key], name="aggregated path exposure")
        _validate_count(
            family_exposure[exposure.family],
            name="aggregated family exposure",
        )

    if forced_path is not None:
        forced_key = canonical_path_bytes(forced_path)
        if forced_key not in groups:
            raise ValueError("forced_path has no catalog options")
        eligible_path_key = forced_key
        mode = PathSelectionMode.FORCED
    elif required_path is not None:
        eligible_path_key = canonical_path_bytes(required_path)
        mode = PathSelectionMode.REQUIRED_OPTIONS
    else:
        eligible_path_key = None
        mode = PathSelectionMode.MINIMUM_EXPOSURE_TASK_KEYED

    path_rows: list[PalettePathRow] = []
    for path_key in sorted(groups):
        family_counts: dict[str, int] = {}
        for option in groups[path_key]:
            family_counts[option.family] = family_counts.get(option.family, 0) + 1
        family_capacity = (
            len(groups[path_key])
            if max_options_per_family is None
            else sum(
                min(count, max_options_per_family) for count in family_counts.values()
            )
        )
        feasible = family_capacity >= palette_size
        eligible = feasible and (
            eligible_path_key is None or path_key == eligible_path_key
        )
        path_rows.append(
            PalettePathRow(
                path=paths[path_key],
                option_count=len(groups[path_key]),
                family_capacity=family_capacity,
                path_exposure=path_exposure.get(path_key, 0),
                feasible_for_size=feasible,
                eligible_for_choice=eligible,
                task_order_sha256=_task_order_sha256(
                    seed=seed,
                    task_key=task_key,
                    purpose="path",
                    payload=path_key,
                ),
            )
        )
    eligible_rows = tuple(row for row in path_rows if row.eligible_for_choice)
    if not eligible_rows:
        if eligible_path_key is not None:
            raise ValueError(
                "the constrained path cannot satisfy palette_size under the "
                "family-cap constraint"
            )
        raise ValueError("no catalog path has enough options for the palette")
    chosen_path_row = min(
        eligible_rows,
        key=lambda row: (
            row.path_exposure,
            row.task_order_sha256,
            canonical_path_bytes(row.path),
        ),
    )
    chosen_path = chosen_path_row.path
    chosen_key = canonical_path_bytes(chosen_path)
    chosen_options = tuple(groups[chosen_key])

    selection_hashes = {
        option.option_id: _task_order_sha256(
            seed=seed,
            task_key=task_key,
            purpose="option-selection",
            payload=bytes.fromhex(option.identity_sha256),
        )
        for option in catalog
    }
    presentation_hashes = {
        option.option_id: _task_order_sha256(
            seed=seed,
            task_key=task_key,
            purpose="option-presentation",
            payload=bytes.fromhex(option.identity_sha256),
        )
        for option in catalog
    }
    required_ids = set(required_option_ids)
    remaining = tuple(
        option for option in chosen_options if option.option_id not in required_ids
    )
    ranked_remaining = tuple(
        sorted(
            remaining,
            key=lambda option: (
                exposure_by_cell.get((chosen_key, option.family), 0),
                family_exposure.get(option.family, 0),
                selection_hashes[option.option_id],
                option.option_id,
            ),
        )
    )
    selected_set = set(required_option_ids)
    selected_family_counts: dict[str, int] = {}
    for option in required:
        selected_family_counts[option.family] = (
            selected_family_counts.get(option.family, 0) + 1
        )
    for option in ranked_remaining:
        if len(selected_set) == palette_size:
            break
        if (
            max_options_per_family is not None
            and selected_family_counts.get(option.family, 0) >= max_options_per_family
        ):
            continue
        selected_set.add(option.option_id)
        selected_family_counts[option.family] = (
            selected_family_counts.get(option.family, 0) + 1
        )
    if len(selected_set) != palette_size:  # pragma: no cover - size gate above.
        raise RuntimeError("palette fill did not produce the requested size")
    palette = tuple(
        sorted(
            (option for option in chosen_options if option.option_id in selected_set),
            key=lambda option: (
                presentation_hashes[option.option_id],
                option.option_id,
            ),
        )
    )
    positions = {option.option_id: index for index, option in enumerate(palette)}

    option_rows = tuple(
        PaletteOptionRow(
            option=option,
            cell_exposure=exposure_by_cell.get(
                (canonical_path_bytes(option.path), option.family),
                0,
            ),
            family_exposure=family_exposure.get(option.family, 0),
            path_exposure=path_exposure.get(canonical_path_bytes(option.path), 0),
            selection_order_sha256=selection_hashes[option.option_id],
            presentation_order_sha256=presentation_hashes[option.option_id],
            on_chosen_path=canonical_path_bytes(option.path) == chosen_key,
            required=option.option_id in required_ids,
            palette_position=positions.get(option.option_id),
        )
        for option in catalog
    )
    return _PaletteParts(
        chosen_path=chosen_path,
        path_selection_mode=mode,
        exposure_snapshot=exposure_snapshot,
        path_rows=tuple(path_rows),
        option_rows=option_rows,
        palette=palette,
        catalog_sha256=_catalog_sha256(catalog),
    )


@dataclass(frozen=True, slots=True, eq=False)
class AtomicPaletteDecision:
    """Frozen path, palette, full candidate table, and exact tie-break law."""

    seed: int
    task_key: str
    palette_size: int
    max_options_per_family: int | None
    forced_path: JsonPath | None
    chosen_path: JsonPath
    path_selection_mode: PathSelectionMode
    required_option_ids: tuple[str, ...]
    exposure_snapshot: tuple[PathFamilyExposure, ...]
    path_rows: tuple[PalettePathRow, ...]
    option_rows: tuple[PaletteOptionRow, ...]
    palette: tuple[AtomicEditOption, ...]
    catalog_sha256: str

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION

    def __post_init__(self) -> None:
        if type(self.path_selection_mode) is not PathSelectionMode:
            raise TypeError("path_selection_mode must be a PathSelectionMode")
        if type(self.exposure_snapshot) is not tuple:
            raise TypeError("exposure_snapshot must be an exact tuple")
        if type(self.path_rows) is not tuple or any(
            type(row) is not PalettePathRow for row in self.path_rows
        ):
            raise TypeError("path_rows must contain exact PalettePathRow values")
        if type(self.option_rows) is not tuple or any(
            type(row) is not PaletteOptionRow for row in self.option_rows
        ):
            raise TypeError("option_rows must contain exact PaletteOptionRow values")
        if type(self.palette) is not tuple or any(
            type(option) is not AtomicEditOption for option in self.palette
        ):
            raise TypeError("palette must contain exact AtomicEditOption values")
        require_sha256(self.catalog_sha256, "catalog_sha256")
        expected = _build_parts(
            seed=self.seed,
            task_key=self.task_key,
            options=tuple(row.option for row in self.option_rows),
            palette_size=self.palette_size,
            max_options_per_family=self.max_options_per_family,
            exposures=self.exposure_snapshot,
            forced_path=self.forced_path,
            required_option_ids=self.required_option_ids,
        )
        if self.chosen_path != expected.chosen_path:
            raise ValueError("chosen_path does not match the policy law")
        if self.path_selection_mode is not expected.path_selection_mode:
            raise ValueError("path_selection_mode does not match the constraints")
        if self.exposure_snapshot != expected.exposure_snapshot:
            raise ValueError("exposure_snapshot must use canonical order")
        if self.path_rows != expected.path_rows:
            raise ValueError("path_rows do not match the complete selection facts")
        if self.option_rows != expected.option_rows:
            raise ValueError("option_rows do not match the complete selection facts")
        if self.palette != expected.palette:
            raise ValueError("palette does not match the task-keyed policy law")
        if self.catalog_sha256 != expected.catalog_sha256:
            raise ValueError("catalog_sha256 does not bind the complete catalog")

    def revalidate(self) -> None:
        if type(self) is not AtomicPaletteDecision:
            raise TypeError("decision must be an exact AtomicPaletteDecision")
        AtomicPaletteDecision.__post_init__(self)

    @property
    def task_key_sha256(self) -> str:
        self.revalidate()
        return hashlib.sha256(_frame(_task_key_bytes(self.task_key))).hexdigest()

    def _trace_payload(self) -> dict[str, object]:
        return {
            "event_type": "atomic_palette_selected",
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "seed": self.seed,
            "task_key_sha256": hashlib.sha256(
                _frame(_task_key_bytes(self.task_key))
            ).hexdigest(),
            "palette_size": self.palette_size,
            "max_options_per_family": self.max_options_per_family,
            "forced_path": (
                None if self.forced_path is None else _path_record(self.forced_path)
            ),
            "chosen_path": _path_record(self.chosen_path),
            "path_selection_mode": self.path_selection_mode.value,
            "required_option_ids": list(self.required_option_ids),
            "catalog_sha256": self.catalog_sha256,
            "exposure_snapshot": [
                exposure.to_trace_record() for exposure in self.exposure_snapshot
            ],
            "path_rows": [row.to_trace_record() for row in self.path_rows],
            "option_rows": [row.to_trace_record() for row in self.option_rows],
            "palette_option_ids": [option.option_id for option in self.palette],
            "palette_option_identity_sha256s": [
                option.identity_sha256 for option in self.palette
            ],
            "tie_break_law": {
                "path": (
                    "minimum path exposure, then SHA-256(seed, task key, path), "
                    "then canonical typed path"
                ),
                "fill": (
                    "minimum path/family-cell exposure, then family exposure, "
                    "then SHA-256(seed, task key, option), then option ID; "
                    "skip saturated families under max_options_per_family"
                ),
                "presentation": ("SHA-256(seed, task key, option), then option ID"),
            },
        }

    @property
    def decision_sha256(self) -> str:
        self.revalidate()
        encoded = json.dumps(
            self._trace_payload(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(_DECISION_DOMAIN + encoded).hexdigest()

    def to_trace_record(self) -> dict[str, object]:
        """Return a fresh JSON-safe record binding every candidate and tie-break."""

        self.revalidate()
        return {**self._trace_payload(), "decision_sha256": self.decision_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is AtomicPaletteDecision
            and type(other) is AtomicPaletteDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class TaskKeyedPalettePolicy:
    """Select exact atomic path/palette choices without mutable RNG state."""

    seed: int

    policy_id: ClassVar[str] = POLICY_ID
    policy_version: ClassVar[int] = POLICY_VERSION

    def __post_init__(self) -> None:
        _validate_seed(self.seed)

    def select(
        self,
        *,
        task_key: str,
        options: Sequence[AtomicEditOption],
        palette_size: int,
        max_options_per_family: int | None = None,
        exposures: Sequence[PathFamilyExposure] = (),
        path: JsonPath | None = None,
        required_option_ids: Sequence[str] = (),
    ) -> AtomicPaletteDecision:
        """Choose one path and exact provider order from immutable inputs.

        ``required_option_ids`` is a constraint, never an ordering hint.  Its
        members must be unique, share the selected path, and fit within the
        requested palette.  ``max_options_per_family`` is enforced during both
        path admission and greedy fill.  The returned tuple order remains
        task-keyed.
        """

        _validate_seed(self.seed)
        if isinstance(required_option_ids, (str, bytes)):
            raise TypeError("required_option_ids must be a finite string sequence")
        supplied_required = tuple(required_option_ids)
        if any(type(option_id) is not str for option_id in supplied_required):
            raise TypeError("required_option_ids must contain exact strings")
        if len(set(supplied_required)) != len(supplied_required):
            raise ValueError("required_option_ids cannot contain duplicates")
        canonical_required = tuple(sorted(supplied_required))
        parts = _build_parts(
            seed=self.seed,
            task_key=task_key,
            options=options,
            palette_size=palette_size,
            max_options_per_family=max_options_per_family,
            exposures=exposures,
            forced_path=path,
            required_option_ids=canonical_required,
        )
        return AtomicPaletteDecision(
            seed=self.seed,
            task_key=task_key,
            palette_size=palette_size,
            max_options_per_family=max_options_per_family,
            forced_path=path,
            chosen_path=parts.chosen_path,
            path_selection_mode=parts.path_selection_mode,
            required_option_ids=canonical_required,
            exposure_snapshot=parts.exposure_snapshot,
            path_rows=parts.path_rows,
            option_rows=parts.option_rows,
            palette=parts.palette,
            catalog_sha256=parts.catalog_sha256,
        )


__all__ = [
    "AtomicPaletteDecision",
    "POLICY_ID",
    "POLICY_VERSION",
    "PaletteOptionRow",
    "PalettePathRow",
    "PathFamilyExposure",
    "PathSelectionMode",
    "TaskKeyedPalettePolicy",
]
