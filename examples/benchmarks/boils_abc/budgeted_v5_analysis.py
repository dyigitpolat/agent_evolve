"""Pure, post-run analysis for the BOiLS budgeted-v5 development experiment.

This module is intentionally outside the live optimization path.  It performs
no file or network I/O, imports neither the provider stack nor the v5 planner,
and never discovers an oracle implicitly.  A caller must explicitly supply a
hash-verified oracle summary and the narrow, durable run projection defined
below.  This keeps sealed outcomes unavailable to planning while making the
post-run arithmetic deterministic and independently testable.

All objectives are exact BOiLS integer counts and both objectives are
minimized.  Hypervolume uses a fixed two-dimensional reference point.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction


ANALYSIS_SCHEMA_ID = "boils_abc_budgeted_v5_analysis_v1"
RUN_INPUT_SCHEMA_ID = "boils_abc_budgeted_v5_analysis_input_v2"
EXPECTED_AGENTIC_MODEL_ID = "deepseek/deepseek-v4-pro"
QUALITY_ONLY_TIMING_EXCLUSION_REASON = (
    "Shared-host scheduler contention is recorded but not controlled; this run "
    "may inform optimizer quality and mechanism diagnosis only."
)
QUALITY_ONLY_EXECUTION_CONTRACT: dict[str, object] = {
    "evidence_class": "shared_host_quality_only",
    "authorized_scope": "objective_quality_only",
    "shared_host": True,
    "timing_data_role": "operational_observability_only",
    "timing_comparison_claim_authorized": False,
    "wall_clock_claim_authorized": False,
    "wall_clock_claims_allowed": False,
    "wall_clock_dominance_claim_authorized": False,
    "timing_exclusion_reason": QUALITY_ONLY_TIMING_EXCLUSION_REASON,
}
PROTOCOL_CORRECTION_ID = (
    "boils_v5_u_extended_family_after_matched_support_inspection_v1"
)
PROTOCOL_CORRECTION_TRIGGER = "sealed_matched_random_support_median_equalled_maximum"
KNOWN_LOCAL_ORACLE_V1_SUMMARY_SHA256 = (
    "63e144b597f662b606ea4272e9816a3a1ff8e5c7962685d6751e2d9dcc040b0d"
)

G1_SLOT_ORDER = ("G1-A1", "G1-A2", "G1-D1", "G1-D2", "G1-U", "G1-X")
G2_SLOT_ORDER = ("G2-E", "G2-X")
MODEL_G1_SLOTS = frozenset(G1_SLOT_ORDER[:-1])
ENGINE_G1_SLOTS = frozenset(("G1-X",))
_ACTION_RE = re.compile(r"[a-z][a-z0-9_]*\Z")


class BoilsV5AnalysisError(ValueError):
    """An analysis input cannot support the claimed exact comparison."""


@dataclass(frozen=True, slots=True)
class QualityOnlyExecutionContract:
    """Exact machine-readable denial of timing claims for shared-host evidence."""

    @classmethod
    def from_record(cls, value: object) -> QualityOnlyExecutionContract:
        if value != QUALITY_ONLY_EXECUTION_CONTRACT:
            raise BoilsV5AnalysisError("quality-only execution contract changed")
        return cls()

    def to_record(self) -> dict[str, object]:
        return dict(QUALITY_ONLY_EXECUTION_CONTRACT)


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BoilsV5AnalysisError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_action(value: object, *, name: str) -> str:
    if type(value) is not str or _ACTION_RE.fullmatch(value) is None:
        raise BoilsV5AnalysisError(f"{name} must be a canonical action identifier")
    return value


def _require_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise BoilsV5AnalysisError(f"{name} must be an exact string")
    return value


def _require_exact_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise BoilsV5AnalysisError(f"{name} must be an exact integer")
    return value


def _fraction_record(value: Fraction) -> dict[str, object]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "fraction": f"{value.numerator}/{value.denominator}",
        "value": float(value),
    }


@dataclass(frozen=True, slots=True, order=True)
class ObjectiveVector:
    """Exact BOiLS objective vector in canonical (LUTs, levels) order."""

    total_lut_count: int
    total_levels: int

    def __post_init__(self) -> None:
        _require_exact_int(self.total_lut_count, name="total_lut_count")
        _require_exact_int(self.total_levels, name="total_levels")
        if self.total_lut_count < 0 or self.total_levels < 0:
            raise BoilsV5AnalysisError("objective counts cannot be negative")

    @classmethod
    def from_mapping(cls, value: object, *, name: str) -> ObjectiveVector:
        if not isinstance(value, Mapping) or set(value) != {
            "total_lut_count",
            "total_levels",
        }:
            raise BoilsV5AnalysisError(
                f"{name} must contain exactly the two BOiLS objectives"
            )
        return cls(
            _require_exact_int(
                value["total_lut_count"], name=f"{name}.total_lut_count"
            ),
            _require_exact_int(value["total_levels"], name=f"{name}.total_levels"),
        )

    def to_record(self) -> dict[str, int]:
        return {
            "total_lut_count": self.total_lut_count,
            "total_levels": self.total_levels,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveDelta:
    total_lut_count: int
    total_levels: int

    def __post_init__(self) -> None:
        _require_exact_int(self.total_lut_count, name="delta.total_lut_count")
        _require_exact_int(self.total_levels, name="delta.total_levels")

    def to_record(self) -> dict[str, int]:
        return {
            "total_lut_count": self.total_lut_count,
            "total_levels": self.total_levels,
        }


@dataclass(frozen=True, slots=True, order=True)
class SingleEditKey:
    index: int
    replacement: str

    def __post_init__(self) -> None:
        _require_exact_int(self.index, name="edit.index")
        if self.index < 0:
            raise BoilsV5AnalysisError("edit.index cannot be negative")
        _require_action(self.replacement, name="edit.replacement")

    def to_record(self) -> dict[str, object]:
        return {"index": self.index, "replacement": self.replacement}


@dataclass(frozen=True, slots=True)
class OracleSealExpectation:
    """External identity and complete-key contract for one sealed oracle file."""

    summary_sha256: str
    parent_boils_configuration_sha256: str
    parent_typed_json_configuration_sha256: str
    parent_objectives: ObjectiveVector
    reference_point: ObjectiveVector
    parent_actions: tuple[tuple[int, str], ...]
    expected_replacements: tuple[tuple[int, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        _require_sha256(self.summary_sha256, name="summary_sha256")
        _require_sha256(
            self.parent_boils_configuration_sha256,
            name="parent_boils_configuration_sha256",
        )
        _require_sha256(
            self.parent_typed_json_configuration_sha256,
            name="parent_typed_json_configuration_sha256",
        )
        if type(self.parent_objectives) is not ObjectiveVector:
            raise TypeError("parent_objectives must be an ObjectiveVector")
        if type(self.reference_point) is not ObjectiveVector:
            raise TypeError("reference_point must be an ObjectiveVector")
        if (
            type(self.parent_actions) is not tuple
            or type(self.expected_replacements) is not tuple
        ):
            raise TypeError("oracle key contracts must be exact tuples")
        parent_by_index: dict[int, str] = {}
        for row in self.parent_actions:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError("parent_actions rows must be two-item tuples")
            index = _require_exact_int(row[0], name="parent action index")
            action = _require_action(row[1], name="parent action")
            if index < 0 or index in parent_by_index:
                raise BoilsV5AnalysisError("parent action indices must be unique")
            parent_by_index[index] = action
        if tuple(sorted(parent_by_index.items())) != self.parent_actions:
            raise BoilsV5AnalysisError("parent_actions must be sorted by index")

        replacements_by_index: dict[int, tuple[str, ...]] = {}
        for row in self.expected_replacements:
            if type(row) is not tuple or len(row) != 2 or type(row[1]) is not tuple:
                raise TypeError(
                    "expected_replacements rows must pair an index with a tuple"
                )
            index = _require_exact_int(row[0], name="replacement index")
            replacements = tuple(
                _require_action(item, name="expected replacement") for item in row[1]
            )
            if (
                index not in parent_by_index
                or index in replacements_by_index
                or not replacements
                or len(set(replacements)) != len(replacements)
                or tuple(sorted(replacements)) != replacements
                or parent_by_index[index] in replacements
            ):
                raise BoilsV5AnalysisError(
                    "expected replacement sets must be sorted, unique, nonempty, "
                    "and exclude the parent action"
                )
            replacements_by_index[index] = replacements
        if set(replacements_by_index) != set(parent_by_index):
            raise BoilsV5AnalysisError(
                "parent action and expected replacement indices must match"
            )
        if tuple(sorted(replacements_by_index.items())) != self.expected_replacements:
            raise BoilsV5AnalysisError("expected_replacements must be index-sorted")

    @property
    def expected_keys(self) -> frozenset[SingleEditKey]:
        return frozenset(
            SingleEditKey(index, replacement)
            for index, replacements in self.expected_replacements
            for replacement in replacements
        )


def known_local_oracle_v1_expectation() -> OracleSealExpectation:
    """Return the frozen identity/key contract for the checked-in v1 oracle.

    This contains identities and legal support only, never oracle outcomes.
    Calling it performs no I/O.
    """

    all_actions = frozenset(
        (
            "rewrite",
            "rewrite_z",
            "refactor",
            "refactor_z",
            "resub",
            "resub_z",
            "balance",
            "fraig",
            "sopb",
            "blut",
            "dsdb",
        )
    )
    parent_actions = (
        (1, "rewrite"),
        (7, "refactor_z"),
        (12, "refactor"),
        (18, "rewrite_z"),
    )
    return OracleSealExpectation(
        summary_sha256=KNOWN_LOCAL_ORACLE_V1_SUMMARY_SHA256,
        parent_boils_configuration_sha256=(
            "e954b02443e92dbed5cc7aa21b8d452531400017d602bf5dcdc938fb84e5237e"
        ),
        parent_typed_json_configuration_sha256=(
            "75451fb03ed5b60faa40eb1e956cc2ef86d9f8692e7f55b94ef054b4aab4012a"
        ),
        parent_objectives=ObjectiveVector(7_944, 69),
        reference_point=ObjectiveVector(8_028, 71),
        parent_actions=parent_actions,
        expected_replacements=tuple(
            (index, tuple(sorted(all_actions - {current})))
            for index, current in parent_actions
        ),
    )


@dataclass(frozen=True, slots=True)
class OracleEntry:
    key: SingleEditKey
    objectives: ObjectiveVector
    typed_json_configuration_sha256: str
    boils_configuration_sha256: str
    frozen_order: int

    def __post_init__(self) -> None:
        if type(self.key) is not SingleEditKey:
            raise TypeError("key must be a SingleEditKey")
        if type(self.objectives) is not ObjectiveVector:
            raise TypeError("objectives must be an ObjectiveVector")
        _require_sha256(
            self.typed_json_configuration_sha256,
            name="typed_json_configuration_sha256",
        )
        _require_sha256(
            self.boils_configuration_sha256,
            name="boils_configuration_sha256",
        )
        _require_exact_int(self.frozen_order, name="frozen_order")
        if self.frozen_order <= 0:
            raise BoilsV5AnalysisError("child frozen_order must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            **self.key.to_record(),
            "objectives": self.objectives.to_record(),
            "typed_json_configuration_sha256": self.typed_json_configuration_sha256,
            "boils_configuration_sha256": self.boils_configuration_sha256,
            "frozen_order": self.frozen_order,
        }


@dataclass(frozen=True, slots=True)
class SealedSingleEditOracle:
    source_summary_sha256: str
    parent_objectives: ObjectiveVector
    reference_point: ObjectiveVector
    parent_typed_json_configuration_sha256: str
    parent_boils_configuration_sha256: str
    entries: tuple[OracleEntry, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.source_summary_sha256, name="source_summary_sha256")
        if (
            type(self.parent_objectives) is not ObjectiveVector
            or type(self.reference_point) is not ObjectiveVector
        ):
            raise TypeError(
                "oracle parent and reference must be ObjectiveVector values"
            )
        _require_sha256(
            self.parent_typed_json_configuration_sha256,
            name="parent_typed_json_configuration_sha256",
        )
        _require_sha256(
            self.parent_boils_configuration_sha256,
            name="parent_boils_configuration_sha256",
        )
        if type(self.entries) is not tuple or not self.entries:
            raise BoilsV5AnalysisError("oracle entries must be a nonempty tuple")
        if tuple(sorted(self.entries, key=lambda item: item.key)) != self.entries:
            raise BoilsV5AnalysisError("oracle entries must be key-sorted")
        keys = tuple(item.key for item in self.entries)
        if len(set(keys)) != len(keys):
            raise BoilsV5AnalysisError("oracle edit keys cannot repeat")
        typed_hashes = tuple(
            item.typed_json_configuration_sha256 for item in self.entries
        )
        boils_hashes = tuple(item.boils_configuration_sha256 for item in self.entries)
        if len(set(typed_hashes)) != len(typed_hashes) or len(set(boils_hashes)) != len(
            boils_hashes
        ):
            raise BoilsV5AnalysisError("oracle child configurations must be unique")

    def entry(self, key: SingleEditKey) -> OracleEntry:
        if type(key) is not SingleEditKey:
            raise TypeError("key must be a SingleEditKey")
        for item in self.entries:
            if item.key == key:
                return item
        raise BoilsV5AnalysisError(
            f"sealed oracle has no outcome for index={key.index}, "
            f"replacement={key.replacement}"
        )

    def to_manifest_record(self) -> dict[str, object]:
        return {
            "source_summary_sha256": self.source_summary_sha256,
            "parent_objectives": self.parent_objectives.to_record(),
            "reference_point": self.reference_point.to_record(),
            "parent_typed_json_configuration_sha256": (
                self.parent_typed_json_configuration_sha256
            ),
            "parent_boils_configuration_sha256": (
                self.parent_boils_configuration_sha256
            ),
            "entry_count": len(self.entries),
        }


def parse_sealed_single_edit_oracle(
    summary: Mapping[str, object],
    *,
    source_summary_sha256: str,
    expectation: OracleSealExpectation,
) -> SealedSingleEditOracle:
    """Validate an explicitly supplied sealed summary and build an exact lookup.

    The byte digest is supplied separately because re-serializing parsed JSON
    would not prove the identity of the durable source file.
    """

    if not isinstance(summary, Mapping):
        raise TypeError("summary must be a mapping")
    source_digest = _require_sha256(source_summary_sha256, name="source_summary_sha256")
    if source_digest != expectation.summary_sha256:
        raise BoilsV5AnalysisError(
            "sealed oracle summary digest differs from expectation"
        )
    if summary.get("schema_version") != 1:
        raise BoilsV5AnalysisError("sealed oracle schema_version must be 1")
    if (
        summary.get("status") != "succeeded"
        or summary.get("development_only") is not True
    ):
        raise BoilsV5AnalysisError(
            "sealed oracle must be a succeeded development-only artifact"
        )

    hypervolume = summary.get("hypervolume")
    if not isinstance(hypervolume, Mapping):
        raise BoilsV5AnalysisError("sealed oracle lacks hypervolume metadata")
    if hypervolume.get("objective_direction") != "minimize_both":
        raise BoilsV5AnalysisError("sealed oracle objective direction changed")
    reference = ObjectiveVector.from_mapping(
        hypervolume.get("reference_point"), name="hypervolume.reference_point"
    )
    if reference != expectation.reference_point:
        raise BoilsV5AnalysisError("sealed oracle reference point changed")

    outcomes = summary.get("outcomes_frozen_order")
    if type(outcomes) is not list or not outcomes:
        raise BoilsV5AnalysisError("sealed oracle outcomes must be a nonempty list")
    if any(not isinstance(row, Mapping) for row in outcomes):
        raise BoilsV5AnalysisError("sealed oracle outcome rows must be mappings")
    frozen_orders = tuple(row.get("frozen_order") for row in outcomes)
    if frozen_orders != tuple(range(len(outcomes))):
        raise BoilsV5AnalysisError(
            "sealed oracle outcomes must follow complete contiguous frozen order"
        )

    parent_rows = tuple(
        row
        for row in outcomes
        if row.get("index") is None and row.get("replacement") is None
    )
    if len(parent_rows) != 1 or parent_rows[0] is not outcomes[0]:
        raise BoilsV5AnalysisError(
            "sealed oracle must start with exactly one parent row"
        )
    parent = parent_rows[0]
    _validate_successful_oracle_row(parent, name="parent")
    parent_objectives = ObjectiveVector.from_mapping(
        parent.get("objectives"), name="parent.objectives"
    )
    if parent_objectives != expectation.parent_objectives:
        raise BoilsV5AnalysisError("sealed oracle parent objectives changed")
    parent_typed = _require_sha256(
        parent.get("typed_json_configuration_sha256"),
        name="parent.typed_json_configuration_sha256",
    )
    parent_boils = _require_sha256(
        parent.get("boils_configuration_sha256"),
        name="parent.boils_configuration_sha256",
    )
    if (
        parent_typed != expectation.parent_typed_json_configuration_sha256
        or parent_boils != expectation.parent_boils_configuration_sha256
    ):
        raise BoilsV5AnalysisError(
            "sealed oracle parent configuration identity changed"
        )

    entries: list[OracleEntry] = []
    for ordinal, row in enumerate(outcomes[1:], start=1):
        _validate_successful_oracle_row(row, name=f"outcomes[{ordinal}]")
        index = _require_exact_int(row.get("index"), name=f"outcomes[{ordinal}].index")
        replacement = _require_action(
            row.get("replacement"), name=f"outcomes[{ordinal}].replacement"
        )
        entries.append(
            OracleEntry(
                key=SingleEditKey(index, replacement),
                objectives=ObjectiveVector.from_mapping(
                    row.get("objectives"), name=f"outcomes[{ordinal}].objectives"
                ),
                typed_json_configuration_sha256=_require_sha256(
                    row.get("typed_json_configuration_sha256"),
                    name=f"outcomes[{ordinal}].typed_json_configuration_sha256",
                ),
                boils_configuration_sha256=_require_sha256(
                    row.get("boils_configuration_sha256"),
                    name=f"outcomes[{ordinal}].boils_configuration_sha256",
                ),
                frozen_order=ordinal,
            )
        )

    keys = frozenset(item.key for item in entries)
    if keys != expectation.expected_keys or len(keys) != len(entries):
        missing = sorted(expectation.expected_keys - keys)
        extra = sorted(keys - expectation.expected_keys)
        raise BoilsV5AnalysisError(
            f"sealed oracle edit-key support differs: missing={missing}, extra={extra}"
        )
    parent_hv = _require_exact_int(
        hypervolume.get("parent_c"), name="hypervolume.parent_c"
    )
    if parent_hv != fixed_reference_hypervolume((parent_objectives,), reference):
        raise BoilsV5AnalysisError("sealed oracle parent hypervolume is inconsistent")
    terminal_hv = _require_exact_int(
        hypervolume.get("terminal_local_oracle"),
        name="hypervolume.terminal_local_oracle",
    )
    recomputed_terminal = fixed_reference_hypervolume(
        (parent_objectives, *(item.objectives for item in entries)), reference
    )
    if terminal_hv != recomputed_terminal:
        raise BoilsV5AnalysisError("sealed oracle terminal hypervolume is inconsistent")
    return SealedSingleEditOracle(
        source_summary_sha256=source_digest,
        parent_objectives=parent_objectives,
        reference_point=reference,
        parent_typed_json_configuration_sha256=parent_typed,
        parent_boils_configuration_sha256=parent_boils,
        entries=tuple(sorted(entries, key=lambda item: item.key)),
    )


def _validate_successful_oracle_row(row: Mapping[str, object], *, name: str) -> None:
    if (
        row.get("status") != "succeeded"
        or row.get("valid") is not True
        or row.get("cec_passed") is not True
        or row.get("candidate_local_failure_status") is not None
    ):
        raise BoilsV5AnalysisError(
            f"{name} is not a successful, valid, CEC-passing evaluation"
        )


def dominates(left: ObjectiveVector, right: ObjectiveVector) -> bool:
    return (
        left.total_lut_count <= right.total_lut_count
        and left.total_levels <= right.total_levels
        and left != right
    )


def nondominated_vectors(
    points: Sequence[ObjectiveVector],
) -> tuple[ObjectiveVector, ...]:
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise TypeError("points must be a sequence")
    if any(type(point) is not ObjectiveVector for point in points):
        raise TypeError("points must contain exact ObjectiveVector values")
    unique = tuple(sorted(set(points)))
    return tuple(
        point
        for point in unique
        if not any(dominates(other, point) for other in unique if other != point)
    )


def fixed_reference_hypervolume(
    points: Sequence[ObjectiveVector], reference_point: ObjectiveVector
) -> int:
    """Exact fixed-reference hypervolume for two minimized integer objectives."""

    if type(reference_point) is not ObjectiveVector:
        raise TypeError("reference_point must be an ObjectiveVector")
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise TypeError("points must be a sequence")
    if any(type(point) is not ObjectiveVector for point in points):
        raise TypeError("points must contain exact ObjectiveVector values")
    eligible = sorted(
        set(
            point
            for point in points
            if point.total_lut_count < reference_point.total_lut_count
            and point.total_levels < reference_point.total_levels
        )
    )
    area = 0
    incumbent_levels = reference_point.total_levels
    for point in eligible:
        if point.total_levels < incumbent_levels:
            area += (reference_point.total_lut_count - point.total_lut_count) * (
                incumbent_levels - point.total_levels
            )
            incumbent_levels = point.total_levels
    return area


@dataclass(frozen=True, slots=True)
class G1PaletteSpec:
    """The exact five random choices plus the deterministic engine choice."""

    area_index: int
    area_replacements: tuple[str, str, str]
    depth_index: int
    depth_replacements: tuple[str, str, str]
    uncertainty_index: int
    uncertainty_replacements: tuple[str, str, str]
    coverage_index: int
    coverage_replacement: str

    def __post_init__(self) -> None:
        indices = (
            self.area_index,
            self.depth_index,
            self.uncertainty_index,
            self.coverage_index,
        )
        if any(type(index) is not int or index < 0 for index in indices):
            raise BoilsV5AnalysisError("palette indices must be nonnegative integers")
        if len(set(indices)) != 4:
            raise BoilsV5AnalysisError("the four v5 palette paths must be distinct")
        for name, replacements in (
            ("area", self.area_replacements),
            ("depth", self.depth_replacements),
            ("uncertainty", self.uncertainty_replacements),
        ):
            if type(replacements) is not tuple or len(replacements) != 3:
                raise BoilsV5AnalysisError(f"{name} palette must contain exactly three")
            if len(set(replacements)) != 3:
                raise BoilsV5AnalysisError(f"{name} palette options must be unique")
            for replacement in replacements:
                _require_action(replacement, name=f"{name} replacement")
        _require_action(self.coverage_replacement, name="coverage replacement")

    def allowed_keys_by_slot(self) -> dict[str, frozenset[SingleEditKey]]:
        return {
            "G1-A1": frozenset(
                SingleEditKey(self.area_index, value)
                for value in self.area_replacements
            ),
            "G1-A2": frozenset(
                SingleEditKey(self.area_index, value)
                for value in self.area_replacements
            ),
            "G1-D1": frozenset(
                SingleEditKey(self.depth_index, value)
                for value in self.depth_replacements
            ),
            "G1-D2": frozenset(
                SingleEditKey(self.depth_index, value)
                for value in self.depth_replacements
            ),
            "G1-U": frozenset(
                SingleEditKey(self.uncertainty_index, value)
                for value in self.uncertainty_replacements
            ),
            "G1-X": frozenset(
                (SingleEditKey(self.coverage_index, self.coverage_replacement),)
            ),
        }

    def to_record(self) -> dict[str, object]:
        return {
            "area": {
                "index": self.area_index,
                "replacements": list(self.area_replacements),
            },
            "depth": {
                "index": self.depth_index,
                "replacements": list(self.depth_replacements),
            },
            "uncertainty": {
                "index": self.uncertainty_index,
                "replacements": list(self.uncertainty_replacements),
            },
            "coverage": {
                "index": self.coverage_index,
                "replacement": self.coverage_replacement,
            },
        }

    @classmethod
    def from_record(cls, value: object) -> G1PaletteSpec:
        if not isinstance(value, Mapping):
            raise BoilsV5AnalysisError("palette_spec must be a mapping")
        area = value.get("area")
        depth = value.get("depth")
        uncertainty = value.get("uncertainty")
        coverage = value.get("coverage")
        if any(
            not isinstance(item, Mapping)
            for item in (area, depth, uncertainty, coverage)
        ):
            raise BoilsV5AnalysisError("palette_spec role records are missing")

        def triple(role: Mapping[str, object], *, name: str) -> tuple[str, str, str]:
            replacements = role.get("replacements")
            if type(replacements) is not list or any(
                type(item) is not str for item in replacements
            ):
                raise BoilsV5AnalysisError(
                    f"palette_spec.{name}.replacements must be a list"
                )
            return tuple(replacements)  # type: ignore[return-value]

        assert isinstance(area, Mapping)
        assert isinstance(depth, Mapping)
        assert isinstance(uncertainty, Mapping)
        assert isinstance(coverage, Mapping)
        return cls(
            area_index=_require_exact_int(
                area.get("index"), name="palette_spec.area.index"
            ),
            area_replacements=triple(area, name="area"),
            depth_index=_require_exact_int(
                depth.get("index"), name="palette_spec.depth.index"
            ),
            depth_replacements=triple(depth, name="depth"),
            uncertainty_index=_require_exact_int(
                uncertainty.get("index"), name="palette_spec.uncertainty.index"
            ),
            uncertainty_replacements=triple(uncertainty, name="uncertainty"),
            coverage_index=_require_exact_int(
                coverage.get("index"), name="palette_spec.coverage.index"
            ),
            coverage_replacement=_require_action(
                coverage.get("replacement"),
                name="palette_spec.coverage.replacement",
            ),
        )


@dataclass(frozen=True, slots=True)
class PortfolioAssignment:
    ordinal: int
    slot_edits: tuple[tuple[str, SingleEditKey], ...]
    unique_edit_keys: tuple[SingleEditKey, ...]
    child_physical_evaluation_count: int
    child_cache_hit_count: int
    archive_hypervolume: int

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise BoilsV5AnalysisError("portfolio ordinal cannot be negative")
        if tuple(slot for slot, _ in self.slot_edits) != G1_SLOT_ORDER:
            raise BoilsV5AnalysisError("portfolio slots must use exact G1 order")
        expected_unique = tuple(dict.fromkeys(key for _, key in self.slot_edits))
        if self.unique_edit_keys != expected_unique:
            raise BoilsV5AnalysisError("portfolio unique-edit accounting changed")
        if self.child_physical_evaluation_count != len(expected_unique):
            raise BoilsV5AnalysisError("portfolio physical-evaluation count changed")
        if self.child_cache_hit_count != len(G1_SLOT_ORDER) - len(expected_unique):
            raise BoilsV5AnalysisError("portfolio cache-hit count changed")
        if self.archive_hypervolume < 0:
            raise BoilsV5AnalysisError("portfolio hypervolume cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "slot_edits": [
                {"slot_id": slot_id, **key.to_record()}
                for slot_id, key in self.slot_edits
            ],
            "unique_edit_keys": [key.to_record() for key in self.unique_edit_keys],
            "child_physical_evaluation_count": self.child_physical_evaluation_count,
            "child_cache_hit_count": self.child_cache_hit_count,
            "archive_hypervolume": self.archive_hypervolume,
        }


def _type7_quantile(
    sorted_values: Sequence[int], numerator: int, denominator: int
) -> Fraction:
    if not sorted_values:
        raise BoilsV5AnalysisError("quantile input cannot be empty")
    if denominator <= 0 or not 0 <= numerator <= denominator:
        raise BoilsV5AnalysisError("quantile probability must be in [0,1]")
    position = Fraction((len(sorted_values) - 1) * numerator, denominator)
    lower = position.numerator // position.denominator
    upper = math.ceil(position)
    weight = position - lower
    return (
        Fraction(sorted_values[lower]) * (1 - weight)
        + Fraction(sorted_values[upper]) * weight
    )


@dataclass(frozen=True, slots=True)
class MatchedRandomDistribution:
    oracle_summary_sha256: str
    palette_spec: G1PaletteSpec
    reference_point: ObjectiveVector
    assignments: tuple[PortfolioAssignment, ...]
    support: tuple[tuple[int, int], ...]
    mean: Fraction
    first_quartile_type7: Fraction
    median_type7: Fraction
    third_quartile_type7: Fraction

    def __post_init__(self) -> None:
        _require_sha256(self.oracle_summary_sha256, name="oracle_summary_sha256")
        if type(self.palette_spec) is not G1PaletteSpec:
            raise TypeError("palette_spec must be a G1PaletteSpec")
        if type(self.reference_point) is not ObjectiveVector:
            raise TypeError("reference_point must be an ObjectiveVector")
        if type(self.assignments) is not tuple or type(self.support) is not tuple:
            raise TypeError("distribution assignments and support must be tuples")
        if len(self.assignments) != 243:
            raise BoilsV5AnalysisError(
                "matched v5 distribution must contain 3^5 portfolios"
            )
        if tuple(item.ordinal for item in self.assignments) != tuple(range(243)):
            raise BoilsV5AnalysisError("portfolio ordinals must be contiguous")
        counts = Counter(item.archive_hypervolume for item in self.assignments)
        if self.support != tuple(sorted(counts.items())):
            raise BoilsV5AnalysisError("distribution support does not match portfolios")
        if sum(count for _, count in self.support) != 243:
            raise BoilsV5AnalysisError("distribution support mass changed")
        sorted_hv = sorted(item.archive_hypervolume for item in self.assignments)
        if self.mean != Fraction(sum(sorted_hv), len(sorted_hv)):
            raise BoilsV5AnalysisError("distribution mean does not match portfolios")
        if (
            self.first_quartile_type7 != _type7_quantile(sorted_hv, 1, 4)
            or self.median_type7 != _type7_quantile(sorted_hv, 1, 2)
            or self.third_quartile_type7 != _type7_quantile(sorted_hv, 3, 4)
        ):
            raise BoilsV5AnalysisError("distribution quantiles do not match portfolios")

    def to_record(self, *, include_assignments: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_id": ANALYSIS_SCHEMA_ID,
            "comparison": "exact_uniform_matched_g1_palette_replay",
            "oracle_summary_sha256": self.oracle_summary_sha256,
            "palette_spec": self.palette_spec.to_record(),
            "policy_count": len(self.assignments),
            "uniform_assignment_probability": _fraction_record(Fraction(1, 243)),
            "reference_point": self.reference_point.to_record(),
            "hypervolume": {
                "support": [
                    {"hypervolume": value, "count": count}
                    for value, count in self.support
                ],
                "mean": _fraction_record(self.mean),
                "first_quartile_type7": _fraction_record(self.first_quartile_type7),
                "median_type7": _fraction_record(self.median_type7),
                "third_quartile_type7": _fraction_record(self.third_quartile_type7),
            },
            "cache_semantics": {
                "six_candidate_occurrences_per_portfolio": True,
                "same_parent_path_replacement_is_one_physical_configuration": True,
                "cache_hit_count_support": [
                    {"child_cache_hit_count": value, "portfolio_count": count}
                    for value, count in sorted(
                        Counter(
                            item.child_cache_hit_count for item in self.assignments
                        ).items()
                    )
                ],
            },
            "claim_boundary": (
                "Exact post-hoc development replay over already sealed single-edit "
                "outcomes; it is not a live baseline run or confirmatory evidence."
            ),
        }
        if include_assignments:
            record["assignments"] = [item.to_record() for item in self.assignments]
        return record


def enumerate_matched_random_portfolios(
    oracle: SealedSingleEditOracle,
    palette_spec: G1PaletteSpec,
) -> MatchedRandomDistribution:
    """Enumerate the five independent uniform choices and fixed coverage edit."""

    if type(oracle) is not SealedSingleEditOracle:
        raise TypeError("oracle must be a SealedSingleEditOracle")
    if type(palette_spec) is not G1PaletteSpec:
        raise TypeError("palette_spec must be a G1PaletteSpec")
    allowed = palette_spec.allowed_keys_by_slot()
    for keys in allowed.values():
        for key in keys:
            oracle.entry(key)

    assignments: list[PortfolioAssignment] = []
    choices = itertools.product(
        palette_spec.area_replacements,
        palette_spec.area_replacements,
        palette_spec.depth_replacements,
        palette_spec.depth_replacements,
        palette_spec.uncertainty_replacements,
    )
    for ordinal, (a1, a2, d1, d2, uncertainty) in enumerate(choices):
        slot_edits = (
            ("G1-A1", SingleEditKey(palette_spec.area_index, a1)),
            ("G1-A2", SingleEditKey(palette_spec.area_index, a2)),
            ("G1-D1", SingleEditKey(palette_spec.depth_index, d1)),
            ("G1-D2", SingleEditKey(palette_spec.depth_index, d2)),
            (
                "G1-U",
                SingleEditKey(palette_spec.uncertainty_index, uncertainty),
            ),
            (
                "G1-X",
                SingleEditKey(
                    palette_spec.coverage_index,
                    palette_spec.coverage_replacement,
                ),
            ),
        )
        unique_keys = tuple(dict.fromkeys(key for _, key in slot_edits))
        points = (
            oracle.parent_objectives,
            *(oracle.entry(key).objectives for key in unique_keys),
        )
        assignments.append(
            PortfolioAssignment(
                ordinal=ordinal,
                slot_edits=slot_edits,
                unique_edit_keys=unique_keys,
                child_physical_evaluation_count=len(unique_keys),
                child_cache_hit_count=len(G1_SLOT_ORDER) - len(unique_keys),
                archive_hypervolume=fixed_reference_hypervolume(
                    points, oracle.reference_point
                ),
            )
        )
    if len(assignments) != 3**5:  # pragma: no cover - product invariant.
        raise RuntimeError("matched random enumeration cardinality changed")
    sorted_hv = sorted(item.archive_hypervolume for item in assignments)
    support = tuple(sorted(Counter(sorted_hv).items()))
    return MatchedRandomDistribution(
        oracle_summary_sha256=oracle.source_summary_sha256,
        palette_spec=palette_spec,
        reference_point=oracle.reference_point,
        assignments=tuple(assignments),
        support=support,
        mean=Fraction(sum(sorted_hv), len(sorted_hv)),
        first_quartile_type7=_type7_quantile(sorted_hv, 1, 4),
        median_type7=_type7_quantile(sorted_hv, 1, 2),
        third_quartile_type7=_type7_quantile(sorted_hv, 3, 4),
    )


@dataclass(frozen=True, slots=True)
class G1SlotObservation:
    slot_id: str
    proposal_authority: str
    edit: SingleEditKey
    objectives: ObjectiveVector
    typed_json_configuration_sha256: str

    def __post_init__(self) -> None:
        if self.slot_id not in G1_SLOT_ORDER:
            raise BoilsV5AnalysisError("unknown G1 slot_id")
        expected = "model" if self.slot_id in MODEL_G1_SLOTS else "engine"
        if self.proposal_authority != expected:
            raise BoilsV5AnalysisError("G1 proposal authority differs from protocol")
        if type(self.edit) is not SingleEditKey:
            raise TypeError("edit must be a SingleEditKey")
        if type(self.objectives) is not ObjectiveVector:
            raise TypeError("objectives must be an ObjectiveVector")
        _require_sha256(
            self.typed_json_configuration_sha256,
            name="typed_json_configuration_sha256",
        )

    @classmethod
    def from_record(cls, value: object, *, name: str) -> G1SlotObservation:
        if not isinstance(value, Mapping):
            raise BoilsV5AnalysisError(f"{name} must be a mapping")
        edit = value.get("edit")
        if not isinstance(edit, Mapping):
            raise BoilsV5AnalysisError(f"{name}.edit must be a mapping")
        return cls(
            slot_id=_require_string(value.get("slot_id"), name=f"{name}.slot_id"),
            proposal_authority=_require_string(
                value.get("proposal_authority"),
                name=f"{name}.proposal_authority",
            ),
            edit=SingleEditKey(
                _require_exact_int(edit.get("index"), name=f"{name}.edit.index"),
                _require_action(
                    edit.get("replacement"), name=f"{name}.edit.replacement"
                ),
            ),
            objectives=ObjectiveVector.from_mapping(
                value.get("objectives"), name=f"{name}.objectives"
            ),
            typed_json_configuration_sha256=_require_sha256(
                value.get("typed_json_configuration_sha256"),
                name=f"{name}.typed_json_configuration_sha256",
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "proposal_authority": self.proposal_authority,
            "edit": self.edit.to_record(),
            "objectives": self.objectives.to_record(),
            "typed_json_configuration_sha256": self.typed_json_configuration_sha256,
        }


@dataclass(frozen=True, slots=True)
class SlotTreatmentAssignment:
    stratum_id: str
    slot_id: str
    treatment: str

    def __post_init__(self) -> None:
        if type(self.stratum_id) is not str or not self.stratum_id.strip():
            raise BoilsV5AnalysisError("stratum_id must be nonempty")
        if self.slot_id not in {"G1-A1", "G1-A2", "G1-D1", "G1-D2"}:
            raise BoilsV5AnalysisError("treatments are allowed only on paired G1 slots")
        if self.treatment not in {"real", "placebo"}:
            raise BoilsV5AnalysisError("treatment must be real or placebo")

    @classmethod
    def from_record(cls, value: object, *, name: str) -> SlotTreatmentAssignment:
        if not isinstance(value, Mapping):
            raise BoilsV5AnalysisError(f"{name} must be a mapping")
        return cls(
            stratum_id=_require_string(
                value.get("stratum_id"), name=f"{name}.stratum_id"
            ),
            slot_id=_require_string(value.get("slot_id"), name=f"{name}.slot_id"),
            treatment=_require_string(value.get("treatment"), name=f"{name}.treatment"),
        )

    def to_record(self) -> dict[str, str]:
        return {
            "stratum_id": self.stratum_id,
            "slot_id": self.slot_id,
            "treatment": self.treatment,
        }


@dataclass(frozen=True, slots=True)
class G2SlotObservation:
    slot_id: str
    branch_slot_ids: tuple[str, ...]
    objectives: ObjectiveVector | None
    typed_json_configuration_sha256: str | None
    branch_preservation_verified: bool
    provider_telemetry_present: bool
    skipped: bool = False

    def __post_init__(self) -> None:
        if self.slot_id not in G2_SLOT_ORDER:
            raise BoilsV5AnalysisError("unknown G2 slot_id")
        if (
            type(self.branch_preservation_verified) is not bool
            or type(self.provider_telemetry_present) is not bool
        ):
            raise TypeError("G2 verification flags must be booleans")
        if type(self.skipped) is not bool:
            raise TypeError("skipped must be a boolean")
        if self.skipped:
            if (
                self.branch_slot_ids != ()
                or self.objectives is not None
                or self.typed_json_configuration_sha256 is not None
                or self.branch_preservation_verified
            ):
                raise BoilsV5AnalysisError("skipped G2 slot cannot carry a candidate")
        else:
            if (
                type(self.branch_slot_ids) is not tuple
                or len(self.branch_slot_ids) != 2
                or len(set(self.branch_slot_ids)) != 2
                or any(slot_id not in G1_SLOT_ORDER for slot_id in self.branch_slot_ids)
            ):
                raise BoilsV5AnalysisError(
                    "G2 branches must name two distinct G1 slots"
                )
            if type(self.objectives) is not ObjectiveVector:
                raise TypeError("non-skipped G2 slot requires objectives")
            _require_sha256(
                self.typed_json_configuration_sha256,
                name="G2 typed_json_configuration_sha256",
            )
        if self.provider_telemetry_present:
            raise BoilsV5AnalysisError(
                "engine-only G2 slot cannot have provider telemetry"
            )

    @classmethod
    def from_record(cls, value: object, *, name: str) -> G2SlotObservation:
        if not isinstance(value, Mapping):
            raise BoilsV5AnalysisError(f"{name} must be a mapping")
        branches = value.get("branch_slot_ids")
        if type(branches) is not list or any(
            type(item) is not str for item in branches
        ):
            raise BoilsV5AnalysisError(f"{name}.branch_slot_ids must be a list")
        objectives_raw = value.get("objectives")
        digest_raw = value.get("typed_json_configuration_sha256")
        return cls(
            slot_id=_require_string(value.get("slot_id"), name=f"{name}.slot_id"),
            branch_slot_ids=tuple(branches),  # type: ignore[arg-type]
            objectives=(
                None
                if objectives_raw is None
                else ObjectiveVector.from_mapping(
                    objectives_raw, name=f"{name}.objectives"
                )
            ),
            typed_json_configuration_sha256=(
                None
                if digest_raw is None
                else _require_sha256(
                    digest_raw, name=f"{name}.typed_json_configuration_sha256"
                )
            ),
            branch_preservation_verified=value.get("branch_preservation_verified"),  # type: ignore[arg-type]
            provider_telemetry_present=value.get("provider_telemetry_present"),  # type: ignore[arg-type]
            skipped=value.get("skipped"),  # type: ignore[arg-type]
        )

    def to_record(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "branch_slot_ids": list(self.branch_slot_ids),
            "objectives": None
            if self.objectives is None
            else self.objectives.to_record(),
            "typed_json_configuration_sha256": self.typed_json_configuration_sha256,
            "branch_preservation_verified": self.branch_preservation_verified,
            "provider_telemetry_present": self.provider_telemetry_present,
            "skipped": self.skipped,
        }


@dataclass(frozen=True, slots=True)
class ProtocolCorrectionDisclosure:
    """Mandatory claim boundary for the outcome-aware U-palette correction."""

    correction_id: str
    trigger: str
    outcome_aware_design: bool
    outcome_aware_after_sealed_distribution_inspection: bool
    matched_random_support_source: str
    frozen_before_live_calls: bool
    correction_specific_inspected_outcome_facts_injected_into_uncertainty_prompt: bool
    development_only: bool
    confirmatory: bool
    required_action: str
    required_family: str

    def __post_init__(self) -> None:
        if self.correction_id != PROTOCOL_CORRECTION_ID:
            raise BoilsV5AnalysisError("protocol correction identity changed")
        if self.trigger != PROTOCOL_CORRECTION_TRIGGER:
            raise BoilsV5AnalysisError("protocol correction trigger changed")
        flags = (
            self.outcome_aware_design,
            self.outcome_aware_after_sealed_distribution_inspection,
            self.frozen_before_live_calls,
            self.correction_specific_inspected_outcome_facts_injected_into_uncertainty_prompt,
            self.development_only,
            self.confirmatory,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("protocol correction flags must be exact booleans")
        if flags != (True, True, True, False, True, False):
            raise BoilsV5AnalysisError(
                "protocol correction disclosure weakens its frozen claim boundary"
            )
        if self.required_action != "dsdb" or self.required_family != "gia_dsd_balance":
            raise BoilsV5AnalysisError("protocol correction obligation changed")
        if self.matched_random_support_source != "computed_from_sealed_local_oracle":
            raise BoilsV5AnalysisError(
                "protocol correction must disclose its sealed-oracle support source"
            )

    @classmethod
    def frozen_v1(cls) -> ProtocolCorrectionDisclosure:
        return cls(
            correction_id=PROTOCOL_CORRECTION_ID,
            trigger=PROTOCOL_CORRECTION_TRIGGER,
            outcome_aware_design=True,
            outcome_aware_after_sealed_distribution_inspection=True,
            matched_random_support_source="computed_from_sealed_local_oracle",
            frozen_before_live_calls=True,
            correction_specific_inspected_outcome_facts_injected_into_uncertainty_prompt=False,
            development_only=True,
            confirmatory=False,
            required_action="dsdb",
            required_family="gia_dsd_balance",
        )

    @classmethod
    def from_record(cls, value: object) -> ProtocolCorrectionDisclosure:
        if not isinstance(value, Mapping):
            raise BoilsV5AnalysisError("protocol_correction must be a mapping")
        return cls(
            correction_id=_require_string(
                value.get("correction_id"), name="protocol_correction.correction_id"
            ),
            trigger=_require_string(
                value.get("trigger"), name="protocol_correction.trigger"
            ),
            outcome_aware_design=value.get("outcome_aware_design"),  # type: ignore[arg-type]
            outcome_aware_after_sealed_distribution_inspection=value.get(
                "outcome_aware_after_sealed_distribution_inspection"
            ),  # type: ignore[arg-type]
            matched_random_support_source=_require_string(
                value.get("matched_random_support_source"),
                name="protocol_correction.matched_random_support_source",
            ),
            frozen_before_live_calls=value.get("frozen_before_live_calls"),  # type: ignore[arg-type]
            correction_specific_inspected_outcome_facts_injected_into_uncertainty_prompt=value.get(
                "correction_specific_inspected_outcome_facts_injected_into_uncertainty_prompt"
            ),  # type: ignore[arg-type]
            development_only=value.get("development_only"),  # type: ignore[arg-type]
            confirmatory=value.get("confirmatory"),  # type: ignore[arg-type]
            required_action=_require_string(
                value.get("required_action"),
                name="protocol_correction.required_action",
            ),
            required_family=_require_string(
                value.get("required_family"),
                name="protocol_correction.required_family",
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "correction_id": self.correction_id,
            "trigger": self.trigger,
            "outcome_aware_design": self.outcome_aware_design,
            "outcome_aware_after_sealed_distribution_inspection": (
                self.outcome_aware_after_sealed_distribution_inspection
            ),
            "matched_random_support_source": self.matched_random_support_source,
            "frozen_before_live_calls": self.frozen_before_live_calls,
            "correction_specific_inspected_outcome_facts_injected_into_uncertainty_prompt": (
                self.correction_specific_inspected_outcome_facts_injected_into_uncertainty_prompt
            ),
            "development_only": self.development_only,
            "confirmatory": self.confirmatory,
            "required_action": self.required_action,
            "required_family": self.required_family,
        }


@dataclass(frozen=True, slots=True)
class BoilsV5RunAnalysisInput:
    """Narrow durable projection the live runner may publish for later scoring."""

    agentic_model_id: str
    development_only: bool
    protocol_acceptance_passed: bool
    post_hoc_development_protocol_correction: bool
    execution_contract: QualityOnlyExecutionContract
    protocol_correction: ProtocolCorrectionDisclosure
    palette_spec: G1PaletteSpec
    g1_slots: tuple[G1SlotObservation, ...]
    treatment_assignments: tuple[SlotTreatmentAssignment, ...]
    g2_slots: tuple[G2SlotObservation, ...]

    def __post_init__(self) -> None:
        if self.agentic_model_id != EXPECTED_AGENTIC_MODEL_ID:
            raise BoilsV5AnalysisError("agentic model differs from the frozen v5 model")
        if self.development_only is not True:
            raise BoilsV5AnalysisError("v5 analysis is development-only")
        if type(self.protocol_acceptance_passed) is not bool:
            raise TypeError("protocol_acceptance_passed must be a boolean")
        if self.post_hoc_development_protocol_correction is not True:
            raise BoilsV5AnalysisError(
                "run input must disclose the post-hoc development correction"
            )
        if type(self.execution_contract) is not QualityOnlyExecutionContract:
            raise TypeError("execution_contract must deny shared-host timing claims")
        if type(self.protocol_correction) is not ProtocolCorrectionDisclosure:
            raise TypeError(
                "protocol_correction must be a ProtocolCorrectionDisclosure"
            )
        if type(self.palette_spec) is not G1PaletteSpec:
            raise TypeError("palette_spec must be a G1PaletteSpec")
        if tuple(slot.slot_id for slot in self.g1_slots) != G1_SLOT_ORDER:
            raise BoilsV5AnalysisError("run input must contain six G1 slots in order")
        if tuple(slot.slot_id for slot in self.g2_slots) != G2_SLOT_ORDER:
            raise BoilsV5AnalysisError("run input must contain two G2 slots in order")
        active_g2 = tuple(slot.slot_id for slot in self.g2_slots if not slot.skipped)
        if active_g2 not in {(), ("G2-E",), G2_SLOT_ORDER}:
            raise BoilsV5AnalysisError(
                "active G2 slots must be an exact exploit-first prefix"
            )
        if self.protocol_acceptance_passed and any(
            not slot.skipped and not slot.branch_preservation_verified
            for slot in self.g2_slots
        ):
            raise BoilsV5AnalysisError(
                "an accepted run cannot contain an unverified G2 materialization"
            )
        if (
            type(self.treatment_assignments) is not tuple
            or len(self.treatment_assignments) != 4
        ):
            raise BoilsV5AnalysisError("run input requires four treatment assignments")
        assigned_slots = tuple(item.slot_id for item in self.treatment_assignments)
        if set(assigned_slots) != {"G1-A1", "G1-A2", "G1-D1", "G1-D2"}:
            raise BoilsV5AnalysisError("each paired slot needs exactly one treatment")
        strata: dict[str, list[SlotTreatmentAssignment]] = {}
        for item in self.treatment_assignments:
            strata.setdefault(item.stratum_id, []).append(item)
        if len(strata) != 2 or any(
            len(rows) != 2 or {row.treatment for row in rows} != {"real", "placebo"}
            for rows in strata.values()
        ):
            raise BoilsV5AnalysisError(
                "treatments must form two exact real/placebo strata"
            )

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> BoilsV5RunAnalysisInput:
        if not isinstance(value, Mapping):
            raise TypeError("run analysis input must be a mapping")
        if value.get("schema_id") != RUN_INPUT_SCHEMA_ID:
            raise BoilsV5AnalysisError("run analysis input schema changed")
        g1 = value.get("g1_slots")
        treatments = value.get("treatment_assignments")
        g2 = value.get("g2_slots")
        if type(g1) is not list or type(treatments) is not list or type(g2) is not list:
            raise BoilsV5AnalysisError("run analysis input arrays are missing")
        return cls(
            agentic_model_id=_require_string(
                value.get("agentic_model_id"), name="agentic_model_id"
            ),
            development_only=value.get("development_only"),  # type: ignore[arg-type]
            protocol_acceptance_passed=value.get("protocol_acceptance_passed"),  # type: ignore[arg-type]
            post_hoc_development_protocol_correction=value.get(
                "post_hoc_development_protocol_correction"
            ),  # type: ignore[arg-type]
            execution_contract=QualityOnlyExecutionContract.from_record(
                value.get("execution_contract")
            ),
            protocol_correction=ProtocolCorrectionDisclosure.from_record(
                value.get("protocol_correction")
            ),
            palette_spec=G1PaletteSpec.from_record(value.get("palette_spec")),
            g1_slots=tuple(
                G1SlotObservation.from_record(row, name=f"g1_slots[{index}]")
                for index, row in enumerate(g1)
            ),
            treatment_assignments=tuple(
                SlotTreatmentAssignment.from_record(
                    row, name=f"treatment_assignments[{index}]"
                )
                for index, row in enumerate(treatments)
            ),
            g2_slots=tuple(
                G2SlotObservation.from_record(row, name=f"g2_slots[{index}]")
                for index, row in enumerate(g2)
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_id": RUN_INPUT_SCHEMA_ID,
            "agentic_model_id": self.agentic_model_id,
            "development_only": self.development_only,
            "protocol_acceptance_passed": self.protocol_acceptance_passed,
            "post_hoc_development_protocol_correction": (
                self.post_hoc_development_protocol_correction
            ),
            "execution_contract": self.execution_contract.to_record(),
            "protocol_correction": self.protocol_correction.to_record(),
            "palette_spec": self.palette_spec.to_record(),
            "g1_slots": [slot.to_record() for slot in self.g1_slots],
            "treatment_assignments": [
                item.to_record() for item in self.treatment_assignments
            ],
            "g2_slots": [slot.to_record() for slot in self.g2_slots],
        }


@dataclass(frozen=True, slots=True)
class ExactRankComparison:
    observed_hypervolume: int
    strictly_below_count: int
    equal_count: int
    strictly_above_count: int
    denominator: int

    def __post_init__(self) -> None:
        if (
            min(
                self.observed_hypervolume,
                self.strictly_below_count,
                self.equal_count,
                self.strictly_above_count,
            )
            < 0
            or self.denominator <= 0
        ):
            raise BoilsV5AnalysisError("rank comparison counts are invalid")
        if (
            self.strictly_below_count + self.equal_count + self.strictly_above_count
            != self.denominator
        ):
            raise BoilsV5AnalysisError("rank comparison does not conserve mass")

    @property
    def strict_percentile(self) -> Fraction:
        return Fraction(self.strictly_below_count, self.denominator)

    @property
    def strict_upper_tail(self) -> Fraction:
        return Fraction(self.strictly_above_count, self.denominator)

    @property
    def matching_or_exceeding_tail(self) -> Fraction:
        return Fraction(self.equal_count + self.strictly_above_count, self.denominator)

    def to_record(self) -> dict[str, object]:
        return {
            "observed_hypervolume": self.observed_hypervolume,
            "strictly_below": {
                "count": self.strictly_below_count,
                **_fraction_record(self.strict_percentile),
            },
            "equal": {
                "count": self.equal_count,
                **_fraction_record(Fraction(self.equal_count, self.denominator)),
            },
            "strictly_above": {
                "count": self.strictly_above_count,
                **_fraction_record(self.strict_upper_tail),
            },
            "matching_or_exceeding": {
                "count": self.equal_count + self.strictly_above_count,
                **_fraction_record(self.matching_or_exceeding_tail),
            },
            "denominator": self.denominator,
        }


@dataclass(frozen=True, slots=True)
class G1SlotScore:
    slot_id: str
    edit: SingleEditKey
    objectives: ObjectiveVector
    marginal_hypervolume_against_a0: int
    normalized_reward_against_a0: Fraction

    def to_record(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "edit": self.edit.to_record(),
            "objectives": self.objectives.to_record(),
            "marginal_hypervolume_against_a0": self.marginal_hypervolume_against_a0,
            "normalized_reward_against_a0": _fraction_record(
                self.normalized_reward_against_a0
            ),
        }


@dataclass(frozen=True, slots=True)
class GenerationOneScore:
    archive_hypervolume: int
    unique_child_physical_evaluations: int
    child_cache_hits: int
    slot_scores: tuple[G1SlotScore, ...]
    matched_random_rank: ExactRankComparison

    def to_record(self) -> dict[str, object]:
        return {
            "archive_hypervolume": self.archive_hypervolume,
            "unique_child_physical_evaluations": self.unique_child_physical_evaluations,
            "child_cache_hits": self.child_cache_hits,
            "slot_scores": [item.to_record() for item in self.slot_scores],
            "matched_random_rank": self.matched_random_rank.to_record(),
        }


def score_generation_one(
    run_input: BoilsV5RunAnalysisInput,
    oracle: SealedSingleEditOracle,
    distribution: MatchedRandomDistribution,
) -> GenerationOneScore:
    """Validate live G1 outcomes against the seal and score exact archive quality."""

    if distribution.oracle_summary_sha256 != oracle.source_summary_sha256:
        raise BoilsV5AnalysisError("distribution and oracle identities differ")
    if run_input.palette_spec != distribution.palette_spec:
        raise BoilsV5AnalysisError("run input and matched replay palette specs differ")
    allowed = distribution.palette_spec.allowed_keys_by_slot()
    points = [oracle.parent_objectives]
    base_hv = fixed_reference_hypervolume(points, oracle.reference_point)
    slot_scores: list[G1SlotScore] = []
    unique_hashes: dict[str, tuple[SingleEditKey, ObjectiveVector]] = {}
    for slot in run_input.g1_slots:
        if slot.edit not in allowed[slot.slot_id]:
            raise BoilsV5AnalysisError(
                f"{slot.slot_id} edit is outside its frozen palette"
            )
        sealed = oracle.entry(slot.edit)
        if (
            slot.objectives != sealed.objectives
            or slot.typed_json_configuration_sha256
            != sealed.typed_json_configuration_sha256
        ):
            raise BoilsV5AnalysisError(
                f"{slot.slot_id} live outcome differs from its sealed single-edit outcome"
            )
        previous = unique_hashes.setdefault(
            slot.typed_json_configuration_sha256, (slot.edit, slot.objectives)
        )
        if previous != (slot.edit, slot.objectives):
            raise BoilsV5AnalysisError(
                "one G1 configuration hash maps to conflicting facts"
            )
        points.append(slot.objectives)
        augmented = fixed_reference_hypervolume(
            (oracle.parent_objectives, slot.objectives), oracle.reference_point
        )
        marginal = augmented - base_hv
        slot_scores.append(
            G1SlotScore(
                slot_id=slot.slot_id,
                edit=slot.edit,
                objectives=slot.objectives,
                marginal_hypervolume_against_a0=marginal,
                normalized_reward_against_a0=Fraction(marginal, max(base_hv, 1)),
            )
        )
    observed_hv = fixed_reference_hypervolume(points, oracle.reference_point)
    random_hvs = tuple(item.archive_hypervolume for item in distribution.assignments)
    below = sum(value < observed_hv for value in random_hvs)
    equal = sum(value == observed_hv for value in random_hvs)
    above = len(random_hvs) - below - equal
    return GenerationOneScore(
        archive_hypervolume=observed_hv,
        unique_child_physical_evaluations=len(unique_hashes),
        child_cache_hits=len(G1_SLOT_ORDER) - len(unique_hashes),
        slot_scores=tuple(slot_scores),
        matched_random_rank=ExactRankComparison(
            observed_hypervolume=observed_hv,
            strictly_below_count=below,
            equal_count=equal,
            strictly_above_count=above,
            denominator=len(random_hvs),
        ),
    )


@dataclass(frozen=True, slots=True)
class PairedCardContrast:
    stratum_id: str
    real_slot_id: str
    placebo_slot_id: str
    real_edit: SingleEditKey
    placebo_edit: SingleEditKey
    edit_changed: bool
    real_marginal_hypervolume: int
    placebo_marginal_hypervolume: int
    marginal_hypervolume_difference: int
    normalized_reward_difference: Fraction

    def to_record(self) -> dict[str, object]:
        return {
            "stratum_id": self.stratum_id,
            "real_slot_id": self.real_slot_id,
            "placebo_slot_id": self.placebo_slot_id,
            "real_edit": self.real_edit.to_record(),
            "placebo_edit": self.placebo_edit.to_record(),
            "edit_changed": self.edit_changed,
            "real_marginal_hypervolume": self.real_marginal_hypervolume,
            "placebo_marginal_hypervolume": self.placebo_marginal_hypervolume,
            "marginal_hypervolume_difference": self.marginal_hypervolume_difference,
            "normalized_reward_difference": _fraction_record(
                self.normalized_reward_difference
            ),
        }


def paired_card_contrasts(
    run_input: BoilsV5RunAnalysisInput,
    g1_score: GenerationOneScore,
) -> tuple[PairedCardContrast, ...]:
    """Compute exact real-minus-placebo contrasts from explicit assignments."""

    observations = {item.slot_id: item for item in run_input.g1_slots}
    scores = {item.slot_id: item for item in g1_score.slot_scores}
    strata: dict[str, dict[str, str]] = {}
    for assignment in run_input.treatment_assignments:
        strata.setdefault(assignment.stratum_id, {})[assignment.treatment] = (
            assignment.slot_id
        )
    output = []
    for stratum_id in sorted(strata):
        real_slot = strata[stratum_id]["real"]
        placebo_slot = strata[stratum_id]["placebo"]
        real_observation = observations[real_slot]
        placebo_observation = observations[placebo_slot]
        if real_observation.edit.index != placebo_observation.edit.index:
            raise BoilsV5AnalysisError(
                "paired treatment slots do not share one edit path"
            )
        real_score = scores[real_slot]
        placebo_score = scores[placebo_slot]
        difference = (
            real_score.marginal_hypervolume_against_a0
            - placebo_score.marginal_hypervolume_against_a0
        )
        output.append(
            PairedCardContrast(
                stratum_id=stratum_id,
                real_slot_id=real_slot,
                placebo_slot_id=placebo_slot,
                real_edit=real_observation.edit,
                placebo_edit=placebo_observation.edit,
                edit_changed=real_observation.edit != placebo_observation.edit,
                real_marginal_hypervolume=(real_score.marginal_hypervolume_against_a0),
                placebo_marginal_hypervolume=(
                    placebo_score.marginal_hypervolume_against_a0
                ),
                marginal_hypervolume_difference=difference,
                normalized_reward_difference=(
                    real_score.normalized_reward_against_a0
                    - placebo_score.normalized_reward_against_a0
                ),
            )
        )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class G2SlotScore:
    slot_id: str
    skipped: bool
    marginal_hypervolume_against_complete_g1: int
    on_individual_augmented_front: bool
    on_terminal_front: bool
    unique_terminal_front_vector: bool
    interaction_available: bool
    interaction_residual: ObjectiveDelta | None
    interaction_unavailable_reason: str | None

    def to_record(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "skipped": self.skipped,
            "marginal_hypervolume_against_complete_g1": (
                self.marginal_hypervolume_against_complete_g1
            ),
            "on_individual_augmented_front": self.on_individual_augmented_front,
            "on_terminal_front": self.on_terminal_front,
            "unique_terminal_front_vector": self.unique_terminal_front_vector,
            "interaction": {
                "available": self.interaction_available,
                "residual": (
                    None
                    if self.interaction_residual is None
                    else self.interaction_residual.to_record()
                ),
                "unavailable_reason": self.interaction_unavailable_reason,
                "definition": "union - left - right + parent_c",
                "interpretation": (
                    "Exact descriptive same-run arithmetic; not a randomized "
                    "factorial interaction estimate."
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class GenerationTwoScore:
    generation_one_hypervolume: int
    terminal_hypervolume: int
    generation_two_marginal_hypervolume: int
    generation_one_front: tuple[ObjectiveVector, ...]
    terminal_front: tuple[ObjectiveVector, ...]
    slot_scores: tuple[G2SlotScore, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "generation_one_hypervolume": self.generation_one_hypervolume,
            "terminal_hypervolume": self.terminal_hypervolume,
            "generation_two_marginal_hypervolume": (
                self.generation_two_marginal_hypervolume
            ),
            "generation_one_front": [
                item.to_record() for item in self.generation_one_front
            ],
            "terminal_front": [item.to_record() for item in self.terminal_front],
            "slot_scores": [item.to_record() for item in self.slot_scores],
        }


def score_generation_two(
    run_input: BoilsV5RunAnalysisInput,
    oracle: SealedSingleEditOracle,
    g1_score: GenerationOneScore,
) -> GenerationTwoScore:
    """Score no-recombination controls and exact descriptive interactions."""

    g1_by_slot = {item.slot_id: item for item in run_input.g1_slots}
    g1_points = (
        oracle.parent_objectives,
        *(item.objectives for item in run_input.g1_slots),
    )
    g1_front = nondominated_vectors(g1_points)
    g2_points = tuple(
        item.objectives
        for item in run_input.g2_slots
        if not item.skipped and item.objectives is not None
    )
    terminal_points = (*g1_points, *g2_points)
    terminal_front = nondominated_vectors(terminal_points)
    terminal_hv = fixed_reference_hypervolume(terminal_points, oracle.reference_point)
    slot_scores: list[G2SlotScore] = []
    for slot in run_input.g2_slots:
        if slot.skipped:
            slot_scores.append(
                G2SlotScore(
                    slot_id=slot.slot_id,
                    skipped=True,
                    marginal_hypervolume_against_complete_g1=0,
                    on_individual_augmented_front=False,
                    on_terminal_front=False,
                    unique_terminal_front_vector=False,
                    interaction_available=False,
                    interaction_residual=None,
                    interaction_unavailable_reason="typed_skipped_slot",
                )
            )
            continue
        assert slot.objectives is not None
        individual_front = nondominated_vectors((*g1_points, slot.objectives))
        individual_hv = fixed_reference_hypervolume(
            (*g1_points, slot.objectives), oracle.reference_point
        )
        marginal = individual_hv - g1_score.archive_hypervolume
        left = g1_by_slot[slot.branch_slot_ids[0]]
        right = g1_by_slot[slot.branch_slot_ids[1]]
        if left.edit.index == right.edit.index:
            interaction_available = False
            residual = None
            unavailable = "branches_are_not_disjoint_single_edit_paths"
        elif not slot.branch_preservation_verified:
            interaction_available = False
            residual = None
            unavailable = "branch_preservation_not_verified"
        else:
            interaction_available = True
            residual = ObjectiveDelta(
                slot.objectives.total_lut_count
                - left.objectives.total_lut_count
                - right.objectives.total_lut_count
                + oracle.parent_objectives.total_lut_count,
                slot.objectives.total_levels
                - left.objectives.total_levels
                - right.objectives.total_levels
                + oracle.parent_objectives.total_levels,
            )
            unavailable = None
        slot_scores.append(
            G2SlotScore(
                slot_id=slot.slot_id,
                skipped=False,
                marginal_hypervolume_against_complete_g1=marginal,
                on_individual_augmented_front=slot.objectives in individual_front,
                on_terminal_front=slot.objectives in terminal_front,
                unique_terminal_front_vector=(
                    slot.objectives in terminal_front
                    and slot.objectives not in g1_front
                ),
                interaction_available=interaction_available,
                interaction_residual=residual,
                interaction_unavailable_reason=unavailable,
            )
        )
    return GenerationTwoScore(
        generation_one_hypervolume=g1_score.archive_hypervolume,
        terminal_hypervolume=terminal_hv,
        generation_two_marginal_hypervolume=(
            terminal_hv - g1_score.archive_hypervolume
        ),
        generation_one_front=g1_front,
        terminal_front=terminal_front,
        slot_scores=tuple(slot_scores),
    )


@dataclass(frozen=True, slots=True)
class MechanismGateAssessment:
    g1_strictly_above_matched_random_median: bool
    model_g1_has_positive_a0_marginal_hv: bool
    g2_has_positive_g1_marginal_or_unique_front_vector: bool
    card_delivery_changes_edit_and_reward_in_at_least_one_pair: bool
    both_real_cards_lower_than_placebos: bool
    protocol_acceptance_passed: bool
    mechanisms_advance: bool
    next_step: str

    def to_record(self) -> dict[str, object]:
        return {
            "artifact_71_mechanism_gates": {
                "g1_strictly_above_matched_random_median": (
                    self.g1_strictly_above_matched_random_median
                ),
                "model_g1_has_positive_a0_marginal_hv": (
                    self.model_g1_has_positive_a0_marginal_hv
                ),
                "g2_has_positive_g1_marginal_or_unique_front_vector": (
                    self.g2_has_positive_g1_marginal_or_unique_front_vector
                ),
                "card_delivery_changes_edit_and_reward_in_at_least_one_pair": (
                    self.card_delivery_changes_edit_and_reward_in_at_least_one_pair
                ),
                "not_both_real_cards_lower_than_placebos": (
                    not self.both_real_cards_lower_than_placebos
                ),
                "protocol_acceptance_passed": self.protocol_acceptance_passed,
                "all_pass": self.mechanisms_advance,
            },
            "both_real_cards_lower_than_placebos": (
                self.both_real_cards_lower_than_placebos
            ),
            "next_step": self.next_step,
            "claim_status": {
                "evidence_class": "shared_host_quality_only",
                "outcome_aware_design": True,
                "matched_random_support_source": ("computed_from_sealed_local_oracle"),
                "protocol_frozen_before_live_calls": True,
                "correction_specific_inspected_outcome_facts_injected_into_uncertainty_prompt": False,
                "confirmatory_evidence": False,
                "held_out_result": False,
                "genericity_claim_authorized": False,
                "sota_claim_authorized": False,
                "timing_comparison_claim_authorized": False,
                "wall_clock_claim_authorized": False,
                "wall_clock_dominance_claim_authorized": False,
                "only_authorized_pass_consequence": (
                    "freeze the same policies on an unopened circuit and compare "
                    "under a full matched budget"
                ),
            },
        }


def assess_artifact_71_mechanism_gates(
    run_input: BoilsV5RunAnalysisInput,
    distribution: MatchedRandomDistribution,
    g1_score: GenerationOneScore,
    card_contrasts: Sequence[PairedCardContrast],
    g2_score: GenerationTwoScore,
) -> MechanismGateAssessment:
    """Apply the frozen development gates without upgrading their claim class."""

    if len(card_contrasts) != 2:
        raise BoilsV5AnalysisError("artifact-71 requires exactly two paired contrasts")
    g1_above = Fraction(g1_score.archive_hypervolume) > distribution.median_type7
    model_slots = {
        item.slot_id
        for item in run_input.g1_slots
        if item.proposal_authority == "model"
    }
    model_positive = any(
        score.slot_id in model_slots and score.marginal_hypervolume_against_a0 > 0
        for score in g1_score.slot_scores
    )
    g2_positive_or_novel = any(
        item.interaction_available
        and (
            item.marginal_hypervolume_against_complete_g1 > 0
            or item.unique_terminal_front_vector
        )
        for item in g2_score.slot_scores
        if not item.skipped
    )
    card_active = any(
        item.edit_changed and item.marginal_hypervolume_difference != 0
        for item in card_contrasts
    )
    both_real_lower = all(
        item.marginal_hypervolume_difference < 0 for item in card_contrasts
    )
    advance = all(
        (
            run_input.protocol_acceptance_passed,
            g1_above,
            model_positive,
            g2_positive_or_novel,
            card_active,
            not both_real_lower,
        )
    )
    if advance:
        next_step = "freeze_same_policies_for_unopened_circuit_matched_budget_only"
    elif both_real_lower:
        next_step = (
            "retire_performance_card_retrieval_keep_uncertainty_coverage_logging"
        )
    elif not card_active:
        next_step = "remove_behaviorally_inert_card_delivery_before_more_experiments"
    else:
        next_step = "do_not_advance_failed_mechanism_gates"
    return MechanismGateAssessment(
        g1_strictly_above_matched_random_median=g1_above,
        model_g1_has_positive_a0_marginal_hv=model_positive,
        g2_has_positive_g1_marginal_or_unique_front_vector=g2_positive_or_novel,
        card_delivery_changes_edit_and_reward_in_at_least_one_pair=card_active,
        both_real_cards_lower_than_placebos=both_real_lower,
        protocol_acceptance_passed=run_input.protocol_acceptance_passed,
        mechanisms_advance=advance,
        next_step=next_step,
    )


def analyze_budgeted_v5_run(
    run_input: BoilsV5RunAnalysisInput,
    oracle: SealedSingleEditOracle,
    distribution: MatchedRandomDistribution,
) -> dict[str, object]:
    """Return the complete pure analysis record for one accepted or failed run."""

    g1 = score_generation_one(run_input, oracle, distribution)
    cards = paired_card_contrasts(run_input, g1)
    g2 = score_generation_two(run_input, oracle, g1)
    gates = assess_artifact_71_mechanism_gates(run_input, distribution, g1, cards, g2)
    return {
        "schema_id": ANALYSIS_SCHEMA_ID,
        "development_only": True,
        "post_hoc_development_protocol_correction": True,
        "protocol_correction": run_input.protocol_correction.to_record(),
        "oracle": oracle.to_manifest_record(),
        "matched_random_distribution": distribution.to_record(
            include_assignments=False
        ),
        "generation_one": g1.to_record(),
        "paired_card_contrasts": [item.to_record() for item in cards],
        "generation_two": g2.to_record(),
        "mechanism_assessment": gates.to_record(),
    }
