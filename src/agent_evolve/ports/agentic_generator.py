"""High-level agentic generation port used by the evolutionary workflow."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol, Tuple, runtime_checkable

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
    validate_finite_variation_contract,
)
from agent_evolve.domain.ids import LLMCallId
from agent_evolve.domain.patch import (
    ArrayIndex,
    JsonPath,
    ObjectKey,
    require_sha256,
    validate_json_path,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonArray,
    FrozenJsonObject,
    FrozenJsonValue,
    canonical_typed_json_bytes,
    freeze_json,
    is_json_scalar,
    typed_json_equal,
)


_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FINITE_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_CROSSOVER_LOCUS_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_REFLECTION_METRIC_ID = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_FINITE_OPTION_FAMILY = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_REFLECTION_SEMANTIC_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_REFLECTION_DECISION_PATH = re.compile(r"^\$([.\[].*)?$")
_REFLECTION_INSIGHT_CONTRACT_DOMAIN = b"agent-evolve:reflection-insight-contract:v2\x00"
_REFLECTION_INSIGHT_CONTRACT_V3_DOMAIN = (
    b"agent-evolve:reflection-insight-contract:v3\x00"
)
_REFLECTION_EVIDENCE_CATALOG_DOMAIN = b"agent-evolve:reflection-evidence-catalog:v1\x00"
_INSIGHT_DRAFT_CONTENT_DOMAIN = b"agent-evolve:insight-draft-content:v1\x00"
_INSIGHT_DRAFT_HYPOTHESIS_DOMAIN = (
    b"agent-evolve:insight-draft-unverified-hypothesis:v1\x00"
)

# Shared by the prompt policy and the provider-visible structured-output schema.
# Keeping this at the port boundary prevents adapters from inventing different
# meanings for the same candidate-relative evidence paths.
CANDIDATE_COMPONENT_PATH_CONTRACT = (
    "For every intended_changes entry and source_attribution.path, '$' denotes "
    "the returned configuration value itself, not the enclosing proposal and "
    "not a parent-evidence record. For example, when configuration is "
    '{"gene":1,"settings":{"rate":2}}, use "$.gene" and '
    '"$.settings.rate". For that example, "$.configuration.gene", '
    '"$.parents[0].configuration.gene", and "$" are invalid component paths. '
    "Never copy an enclosing configuration or parents wrapper into a path; "
    '"$.configuration..." is valid only when "configuration" is genuinely a '
    "top-level component inside the candidate configuration."
)

# Shared by the generic two-parent prompt and its provider-visible output
# schema.  Left/right claims are executable provenance: the engine verifies the
# witness and copies the exact immutable subtree from the named parent.
TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT = (
    "For a two-parent crossover, source='left' or source='right' is valid at a "
    "path only when the intended child matches that named parent and differs "
    "from the other parent at the same path. The engine treats each left/right "
    "claim as an executable inheritance instruction and copies the exact "
    "immutable parent subtree; the returned configuration is a consistency "
    "witness, not authority for inherited values. Its structure and nonnumeric "
    "values must match exactly; each finite float leaf may differ by at most "
    "one binary64 ULP to accommodate decimal serialization drift. A path where "
    "both parents have "
    "identical values proves neither side and must not be attributed to left "
    "or right. If the child preserves that shared value, omit the path from "
    "both source_attribution and intended_changes; if the child changes it, "
    "attribute the genuinely new value as synthesized. Synthesized values "
    "remain model-authored and must differ from both parents at their explicit "
    "paths. A structural deletion must be attributed as synthesized at the "
    "smallest retained containing object or array whose complete child value "
    "reflects the deletion; an omitted path cannot itself be claimed. Claims "
    "must be unique, non-overlapping, and collectively account for every "
    "parent/child difference. Any unclaimed value shared by both parents must "
    "remain present and exactly equal in the child. Include at least one "
    "discriminating left contribution and at least one discriminating right "
    "contribution."
)

MAX_REFLECTION_EVIDENCE_CATALOG_ENTRIES = 256
MAX_REFLECTION_SEMANTIC_VOCABULARY_ENTRIES = 256
MAX_REFLECTION_DECISION_PATH_CHARS = 4_096
_REFLECTION_EVIDENCE_CITATION_KEY = re.compile(r"^e[0-9]{4}$")


def _validate_contrast_ids(values: tuple[str, ...], *, name: str) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or _LOWER_SHA256.fullmatch(value) is None
        for value in values
    ):
        raise TypeError(f"{name} must be an exact tuple of lowercase SHA-256 IDs")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonically sorted")


def _validate_canonical_tokens(
    values: tuple[str, ...],
    *,
    name: str,
    pattern: re.Pattern[str],
) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or pattern.fullmatch(value) is None for value in values
    ):
        raise TypeError(f"{name} must be an exact tuple of canonical identifiers")
    if not values:
        raise ValueError(f"{name} must be non-empty")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonically sorted")


def _validate_canonical_enum_values(
    values: tuple[Enum, ...],
    *,
    name: str,
    enum_type: type[Enum],
    allow_empty: bool = False,
) -> None:
    if type(values) is not tuple or any(
        type(value) is not enum_type for value in values
    ):
        raise TypeError(f"{name} must be an exact tuple of {enum_type.__name__} values")
    if not values and not allow_empty:
        raise ValueError(f"{name} must be non-empty")
    canonical = tuple(sorted(set(values), key=lambda value: str(value.value)))
    if values != canonical:
        raise ValueError(f"{name} must be unique and canonically sorted")


def _validate_decision_paths(values: tuple[str, ...], *, name: str) -> None:
    if type(values) is not tuple or any(type(value) is not str for value in values):
        raise TypeError(f"{name} must be an exact tuple of strings")
    if not values:
        raise ValueError(f"{name} must be non-empty")
    if any(
        len(value) > MAX_REFLECTION_DECISION_PATH_CHARS
        or _REFLECTION_DECISION_PATH.fullmatch(value) is None
        for value in values
    ):
        raise ValueError(f"{name} must contain bounded rooted decision paths")
    if len(values) > MAX_REFLECTION_SEMANTIC_VOCABULARY_ENTRIES:
        raise ValueError(f"{name} exceeds MAX_REFLECTION_SEMANTIC_VOCABULARY_ENTRIES")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonically sorted")


@dataclass(frozen=True, slots=True)
class ReflectionEvidenceCatalogEntry:
    """One request-local short citation bound to a canonical contrast ID."""

    citation_key: str
    contrast_id: str

    def __post_init__(self) -> None:
        if (
            type(self.citation_key) is not str
            or _REFLECTION_EVIDENCE_CITATION_KEY.fullmatch(self.citation_key) is None
        ):
            raise ValueError("citation_key must use the eNNNN grammar")
        if (
            type(self.contrast_id) is not str
            or _LOWER_SHA256.fullmatch(self.contrast_id) is None
        ):
            raise ValueError("contrast_id must be a lowercase SHA-256 ID")

    def to_record(self) -> dict[str, str]:
        ReflectionEvidenceCatalogEntry.__post_init__(self)
        return {
            "citation_key": self.citation_key,
            "contrast_id": self.contrast_id,
        }


@dataclass(frozen=True, slots=True)
class ReflectionEvidenceCatalog:
    """Authenticated deterministic short keys for one reflection request.

    The catalog does not alter the evidence universe.  It is derived from the
    request's already-canonical contrast IDs and merely replaces error-prone
    model reproduction of 64-character hashes with closed ``eNNNN`` literals.
    Resolution is exact: prefixes, fuzzy matches, and foreign keys are never
    accepted.
    """

    entries: Tuple[ReflectionEvidenceCatalogEntry, ...]
    catalog_identity_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or not self.entries:
            raise ValueError("entries must be a non-empty exact tuple")
        if len(self.entries) > MAX_REFLECTION_EVIDENCE_CATALOG_ENTRIES:
            raise ValueError("entries exceed MAX_REFLECTION_EVIDENCE_CATALOG_ENTRIES")
        if any(
            type(entry) is not ReflectionEvidenceCatalogEntry for entry in self.entries
        ):
            raise TypeError(
                "entries must contain exact ReflectionEvidenceCatalogEntry values"
            )
        for entry in self.entries:
            ReflectionEvidenceCatalogEntry.__post_init__(entry)
        contrast_ids = tuple(entry.contrast_id for entry in self.entries)
        _validate_contrast_ids(contrast_ids, name="catalog contrast_ids")
        expected_keys = tuple(
            f"e{index:04d}" for index in range(1, len(self.entries) + 1)
        )
        if self.citation_keys != expected_keys:
            raise ValueError(
                "citation keys must be contiguous e0001..eNNNN in contrast-ID order"
            )
        object.__setattr__(
            self,
            "catalog_identity_sha256",
            hashlib.sha256(
                _REFLECTION_EVIDENCE_CATALOG_DOMAIN
                + canonical_typed_json_bytes(freeze_json(self._unsigned_record()))
            ).hexdigest(),
        )

    @classmethod
    def from_contrast_ids(
        cls,
        contrast_ids: Tuple[str, ...],
    ) -> "ReflectionEvidenceCatalog":
        """Derive the only canonical key assignment for an evidence universe."""

        _validate_contrast_ids(contrast_ids, name="contrast_ids")
        if not contrast_ids:
            raise ValueError("a reflection evidence catalog cannot be empty")
        if len(contrast_ids) > MAX_REFLECTION_EVIDENCE_CATALOG_ENTRIES:
            raise ValueError(
                "contrast_ids exceed MAX_REFLECTION_EVIDENCE_CATALOG_ENTRIES"
            )
        return cls(
            tuple(
                ReflectionEvidenceCatalogEntry(
                    citation_key=f"e{index:04d}",
                    contrast_id=contrast_id,
                )
                for index, contrast_id in enumerate(contrast_ids, start=1)
            )
        )

    @property
    def citation_keys(self) -> Tuple[str, ...]:
        return tuple(entry.citation_key for entry in self.entries)

    @property
    def contrast_ids(self) -> Tuple[str, ...]:
        return tuple(entry.contrast_id for entry in self.entries)

    def citation_key_for_contrast_id(self, contrast_id: str) -> str:
        """Return the exact request-local citation key for one full ID.

        Keeping this lookup on the authenticated catalog prevents composition
        roots from duplicating its field names or rebuilding a parallel map.
        Unknown, malformed, and non-string IDs fail closed.
        """

        ReflectionEvidenceCatalog.__post_init__(self)
        if type(contrast_id) is not str or _LOWER_SHA256.fullmatch(contrast_id) is None:
            raise ValueError("contrast_id must be a lowercase SHA-256 ID")
        for entry in self.entries:
            if entry.contrast_id == contrast_id:
                return entry.citation_key
        raise ValueError("contrast_id is not present in this evidence catalog")

    def resolve_citation_keys(
        self,
        citation_keys: Tuple[str, ...],
    ) -> Tuple[str, ...]:
        """Resolve exact keys to canonical full IDs without corrective matching."""

        ReflectionEvidenceCatalog.__post_init__(self)
        if type(citation_keys) is not tuple or any(
            type(value) is not str for value in citation_keys
        ):
            raise TypeError("citation_keys must be an exact tuple of strings")
        if not citation_keys:
            raise ValueError("citation_keys cannot be empty")
        if len(set(citation_keys)) != len(citation_keys):
            raise ValueError("citation_keys cannot contain duplicates")
        lookup = {entry.citation_key: entry.contrast_id for entry in self.entries}
        if not set(citation_keys).issubset(lookup):
            raise ValueError("citation_keys contain an unknown or foreign key")
        return tuple(sorted(lookup[key] for key in citation_keys))

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "entries": [entry.to_record() for entry in self.entries],
        }

    def to_record(self) -> dict[str, object]:
        ReflectionEvidenceCatalog.__post_init__(self)
        return {
            **self._unsigned_record(),
            "catalog_identity_sha256": self.catalog_identity_sha256,
        }


class MetricEffectDirection(str, Enum):
    """Closed, testable prediction for one explicitly named numeric metric."""

    DECREASE = "decrease"
    INCREASE = "increase"
    UNCHANGED = "unchanged"
    UNKNOWN = "unknown"


class ReflectionInsightKind(str, Enum):
    """Closed epistemic category for a model-authored reflection insight.

    Kind deliberately does not encode where an insight may be consumed.  That
    separate scope prevents a mechanistic conjecture, for example, from
    silently becoming authorization for every evolutionary operator.
    """

    EMPIRICAL_PREDICTIVE_RULE = "empirical_predictive_rule"
    MECHANISTIC_CONJECTURE = "mechanistic_conjecture"
    SEARCH_HEURISTIC = "search_heuristic"
    CONTRACT_INVARIANT = "contract_invariant"


class ReflectionConsumerScope(str, Enum):
    """Closed evolutionary decisions that may consume an insight."""

    MUTATION_SELECTION = "mutation_selection"
    RECOMBINATION_SELECTION = "recombination_selection"
    PARENT_SELECTION = "parent_selection"


class MetricComparisonAnchorKind(str, Enum):
    """Closed reference point for interpreting one metric direction."""

    CURRENT_PARENT = "current_parent"
    NAMED_SOURCE_ROLE = "named_source_role"
    COMMON_ANCESTOR = "common_ancestor"
    FROZEN_ARCHIVE_INCUMBENT = "frozen_archive_incumbent"


@dataclass(frozen=True, slots=True)
class MetricComparisonAnchor:
    """Typed comparison reference, optionally bound to an adapter-owned role."""

    kind: MetricComparisonAnchorKind
    source_role_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not MetricComparisonAnchorKind:
            raise TypeError("kind must be an exact MetricComparisonAnchorKind")
        if self.kind is MetricComparisonAnchorKind.NAMED_SOURCE_ROLE:
            if (
                type(self.source_role_id) is not str
                or _REFLECTION_SEMANTIC_TOKEN.fullmatch(self.source_role_id) is None
            ):
                raise ValueError(
                    "named_source_role anchors require a canonical source_role_id"
                )
        elif self.source_role_id is not None:
            raise ValueError(
                "source_role_id is valid only for named_source_role anchors"
            )

    def to_record(self) -> dict[str, str | None]:
        MetricComparisonAnchor.__post_init__(self)
        return {
            "kind": self.kind.value,
            "source_role_id": self.source_role_id,
        }


@dataclass(frozen=True, slots=True)
class MetricEffectPrediction:
    """One direction prediction, independent of any domain's goal semantics."""

    metric_id: str
    direction: MetricEffectDirection
    comparison_anchor: MetricComparisonAnchor | None = None

    def __post_init__(self) -> None:
        if (
            type(self.metric_id) is not str
            or _REFLECTION_METRIC_ID.fullmatch(self.metric_id) is None
        ):
            raise ValueError(
                "metric_id must use the closed reflection identifier grammar"
            )
        if type(self.direction) is not MetricEffectDirection:
            raise TypeError("direction must be an exact MetricEffectDirection")
        if self.comparison_anchor is not None:
            if type(self.comparison_anchor) is not MetricComparisonAnchor:
                raise TypeError(
                    "comparison_anchor must be an exact MetricComparisonAnchor or None"
                )
            MetricComparisonAnchor.__post_init__(self.comparison_anchor)

    def to_record(self) -> dict[str, object]:
        MetricEffectPrediction.__post_init__(self)
        record: dict[str, object] = {
            "metric_id": self.metric_id,
            "direction": self.direction.value,
        }
        if self.comparison_anchor is not None:
            record["comparison_anchor"] = self.comparison_anchor.to_record()
        return record


@dataclass(frozen=True, slots=True)
class SourceAttribution:
    path: str
    source: str

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path.strip():
            raise ValueError("attribution path must be non-empty")
        if self.source not in {"ancestor", "left", "right", "synthesized", "mutation"}:
            raise ValueError("unsupported attribution source")


@dataclass(frozen=True, slots=True)
class ConflictResolutionDraft:
    relation_id: str
    choice: str
    explanation: str

    def __post_init__(self) -> None:
        if type(self.relation_id) is not str or not self.relation_id.strip():
            raise ValueError("relation_id must be non-empty")
        if self.choice not in {
            "choose_left",
            "choose_right",
            "synthesize",
            "drop_both",
        }:
            raise ValueError("unsupported conflict resolution choice")
        if type(self.explanation) is not str or not self.explanation.strip():
            raise ValueError("resolution explanation must be non-empty")


@dataclass(frozen=True, slots=True)
class CandidateDraft:
    configuration: dict[str, Any]
    design_rationale: str
    intended_changes: Tuple[str, ...] = ()
    source_attribution: Tuple[SourceAttribution, ...] = ()
    claimed_insight_ids: Tuple[str, ...] = ()
    claimed_preservation_obligation_ids: Tuple[str, ...] = ()
    conflict_resolutions: Tuple[ConflictResolutionDraft, ...] = ()

    def __post_init__(self) -> None:
        if type(self.configuration) is not dict:
            raise TypeError("configuration must be an exact dict")
        if type(self.design_rationale) is not str or not self.design_rationale.strip():
            raise ValueError("design_rationale must be non-empty")
        for name in (
            "intended_changes",
            "claimed_insight_ids",
            "claimed_preservation_obligation_ids",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not str or not value.strip() for value in values
            ):
                raise TypeError(f"{name} must be an exact tuple of non-empty strings")
        if type(self.source_attribution) is not tuple or any(
            type(value) is not SourceAttribution for value in self.source_attribution
        ):
            raise TypeError("source_attribution must contain exact values")
        if type(self.conflict_resolutions) is not tuple or any(
            type(value) is not ConflictResolutionDraft
            for value in self.conflict_resolutions
        ):
            raise TypeError("conflict_resolutions must contain exact values")


def _frozen_value_at_path(
    value: FrozenJsonValue,
    path: JsonPath,
) -> FrozenJsonValue:
    """Resolve an already-frozen value without introducing a policy dependency."""

    current = value
    for segment in path.segments:
        if type(segment) is ObjectKey:
            if type(current) is not FrozenJsonObject:
                raise ValueError("atomic mutation path reaches a non-object")
            matches = tuple(item for key, item in current.items if key == segment.value)
            if len(matches) != 1:
                raise ValueError("atomic mutation path does not exist")
            current = matches[0]
        elif type(segment) is ArrayIndex:
            if type(current) is not FrozenJsonArray:
                raise ValueError("atomic mutation path reaches a non-array")
            if segment.value >= len(current.items):
                raise ValueError("atomic mutation path is out of bounds")
            current = current.items[segment.value]
        else:  # pragma: no cover - JsonPath closes the segment union.
            raise AssertionError("unsupported path segment")
    return current


@dataclass(frozen=True, slots=True)
class AtomicMutationOutputContract:
    """Immutable parent and exact scalar leaf exposed to one patch-native call."""

    parent_configuration: FrozenJsonObject
    editable_path: JsonPath
    replacement_options: Tuple[FrozenJsonValue, ...] = ()

    def __post_init__(self) -> None:
        if type(self.parent_configuration) is not FrozenJsonObject:
            raise TypeError("parent_configuration must be an exact FrozenJsonObject")
        if freeze_json(self.parent_configuration) is not self.parent_configuration:
            raise TypeError("parent_configuration must already be frozen typed JSON")
        if type(self.editable_path) is not JsonPath:
            raise TypeError("editable_path must be an exact JsonPath")
        validate_json_path(self.editable_path)
        if not self.editable_path.segments:
            raise ValueError("atomic mutation cannot replace the candidate root")
        current = _frozen_value_at_path(
            self.parent_configuration,
            self.editable_path,
        )
        if not is_json_scalar(current):
            raise ValueError("atomic mutation editable_path must resolve to a scalar")
        if type(self.replacement_options) is not tuple:
            raise TypeError("replacement_options must be an exact tuple")
        canonical_options: list[bytes] = []
        for option in self.replacement_options:
            if freeze_json(option) is not option or not is_json_scalar(option):
                raise TypeError(
                    "replacement_options must contain frozen typed-JSON scalars"
                )
            if typed_json_equal(current, option):
                raise ValueError("replacement_options must exclude the parent value")
            canonical_options.append(canonical_typed_json_bytes(option))
        if len(set(canonical_options)) != len(canonical_options):
            raise ValueError("replacement_options cannot contain duplicates")


@dataclass(frozen=True, slots=True)
class AtomicMutationDraft:
    """One model-authored scalar replacement; the engine owns materialization."""

    path: JsonPath
    replacement: FrozenJsonValue
    design_rationale: str
    claimed_insight_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.path) is not JsonPath:
            raise TypeError("path must be an exact JsonPath")
        validate_json_path(self.path)
        if not self.path.segments:
            raise ValueError("an atomic mutation path cannot be the root")
        frozen_replacement = freeze_json(self.replacement)
        if frozen_replacement is not self.replacement:
            raise TypeError("replacement must already be frozen typed JSON")
        if not is_json_scalar(self.replacement):
            raise TypeError("replacement must be an exact typed-JSON scalar")
        if type(self.design_rationale) is not str or not self.design_rationale.strip():
            raise ValueError("design_rationale must be non-empty")
        if type(self.claimed_insight_ids) is not tuple or any(
            type(value) is not str or not value.strip()
            for value in self.claimed_insight_ids
        ):
            raise TypeError(
                "claimed_insight_ids must be an exact tuple of non-empty strings"
            )


@dataclass(frozen=True, slots=True)
class ExactParentCrossoverOutputContract:
    """Request-scoped finite vocabulary for an executable parent-import plan.

    The engine owns parent configurations, locus paths, and materialization.
    The provider sees only these opaque locus identifiers and selects a proper,
    nonempty donor subset.  Keeping parent payloads out of this port contract
    prevents an integration adapter from becoming a second crossover engine.
    """

    contract_identity_sha256: str
    locus_ids: Tuple[str, ...]
    claimable_insight_ids: Tuple[str, ...] = ()
    forbidden_import_locus_sets: Tuple[Tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        require_sha256(
            self.contract_identity_sha256,
            "contract_identity_sha256",
        )
        if type(self.locus_ids) is not tuple or any(
            type(value) is not str or _CROSSOVER_LOCUS_ID.fullmatch(value) is None
            for value in self.locus_ids
        ):
            raise TypeError("locus_ids must be an exact tuple of canonical identifiers")
        if len(self.locus_ids) < 2:
            raise ValueError("exact parent crossover requires at least two loci")
        if self.locus_ids != tuple(sorted(set(self.locus_ids))):
            raise ValueError("locus_ids must be unique and canonically sorted")
        if type(self.claimable_insight_ids) is not tuple or any(
            type(value) is not str or not value.strip()
            for value in self.claimable_insight_ids
        ):
            raise TypeError(
                "claimable_insight_ids must be an exact tuple of non-empty strings"
            )
        if self.claimable_insight_ids != tuple(sorted(set(self.claimable_insight_ids))):
            raise ValueError(
                "claimable_insight_ids must be unique and canonically sorted"
            )
        if type(self.forbidden_import_locus_sets) is not tuple or any(
            type(value) is not tuple for value in self.forbidden_import_locus_sets
        ):
            raise TypeError(
                "forbidden_import_locus_sets must be an exact tuple of tuples"
            )
        allowed = set(self.locus_ids)
        for forbidden in self.forbidden_import_locus_sets:
            if not forbidden or len(forbidden) >= len(self.locus_ids):
                raise ValueError(
                    "each forbidden import locus set must be a proper nonempty subset"
                )
            if forbidden != tuple(sorted(set(forbidden))):
                raise ValueError(
                    "forbidden import locus sets must use canonical unique IDs"
                )
            if not set(forbidden).issubset(allowed):
                raise ValueError(
                    "forbidden import locus sets cannot escape the locus catalog"
                )
        if self.forbidden_import_locus_sets != tuple(
            sorted(set(self.forbidden_import_locus_sets))
        ):
            raise ValueError(
                "forbidden import locus sets must be unique and canonically sorted"
            )
        if len(self.forbidden_import_locus_sets) == (1 << len(self.locus_ids)) - 2:
            raise ValueError(
                "forbidden import locus sets exhaust the exact crossover action space"
            )


@dataclass(frozen=True, slots=True)
class ExactParentCrossoverDraft:
    """One bounded model choice; the engine owns the exact child and evidence."""

    contract_identity_sha256: str
    import_locus_ids: Tuple[str, ...]
    claimed_insight_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_sha256(
            self.contract_identity_sha256,
            "contract_identity_sha256",
        )
        if type(self.import_locus_ids) is not tuple or any(
            type(value) is not str or _CROSSOVER_LOCUS_ID.fullmatch(value) is None
            for value in self.import_locus_ids
        ):
            raise TypeError(
                "import_locus_ids must be an exact tuple of canonical identifiers"
            )
        if not self.import_locus_ids:
            raise ValueError("a crossover plan must import at least one donor locus")
        if self.import_locus_ids != tuple(sorted(set(self.import_locus_ids))):
            raise ValueError("import_locus_ids must be unique and canonically sorted")
        if type(self.claimed_insight_ids) is not tuple or any(
            type(value) is not str or not value.strip()
            for value in self.claimed_insight_ids
        ):
            raise TypeError(
                "claimed_insight_ids must be an exact tuple of non-empty strings"
            )


@dataclass(frozen=True, slots=True)
class FiniteVariationSelectionDraft:
    """A model-selected ID bound to the exact presealed option and palette."""

    option_id: str
    option_identity_sha256: str
    contract_identity_sha256: str
    design_rationale: str
    claimed_insight_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.option_id) is not str
            or _FINITE_OPTION_ID.fullmatch(self.option_id) is None
        ):
            raise ValueError(
                "option_id must use the closed lowercase identifier grammar"
            )
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        require_sha256(self.contract_identity_sha256, "contract_identity_sha256")
        if type(self.design_rationale) is not str or not self.design_rationale.strip():
            raise ValueError("design_rationale must be non-empty")
        if type(self.claimed_insight_ids) is not tuple or any(
            type(value) is not str or not value.strip()
            for value in self.claimed_insight_ids
        ):
            raise TypeError(
                "claimed_insight_ids must be an exact tuple of non-empty strings"
            )


def resolve_finite_variation_selection(
    contract: FiniteVariationContract,
    draft: FiniteVariationSelectionDraft,
) -> FiniteVariationOption:
    """Verify selection receipts and return the one sealed full child."""

    validate_finite_variation_contract(contract)
    if type(draft) is not FiniteVariationSelectionDraft:
        raise TypeError("draft must be an exact FiniteVariationSelectionDraft")
    FiniteVariationSelectionDraft.__post_init__(draft)
    if draft.contract_identity_sha256 != contract.identity_sha256:
        raise ValueError("selection draft is bound to a different finite contract")
    option = contract.resolve(draft.option_id)
    if draft.option_identity_sha256 != option.identity_sha256:
        raise ValueError("selection draft is bound to a different finite option")
    return option


@dataclass(frozen=True, slots=True)
class InsightDraft:
    claim: str
    trigger: str
    mechanism: str
    affected_paths: Tuple[str, ...]
    evidence_summary: str
    confidence: float
    evidence_contrast_ids: Tuple[str, ...] = ()
    effect_predictions: Tuple[MetricEffectPrediction, ...] = ()
    recommended_option_families: Tuple[str, ...] = ()
    recommended_option_ids: Tuple[str, ...] = ()
    action_template: str | None = None
    falsification_condition: str | None = None
    insight_kind: ReflectionInsightKind | None = None
    consumer_scopes: Tuple[ReflectionConsumerScope, ...] = ()
    factor_capabilities: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("claim", "trigger", "mechanism", "evidence_summary"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if type(self.affected_paths) is not tuple or any(
            type(value) is not str or not value.strip() for value in self.affected_paths
        ):
            raise TypeError("affected_paths must be a tuple of non-empty strings")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0 <= float(self.confidence) <= 1
        ):
            raise ValueError("confidence must be finite in [0,1]")
        _validate_contrast_ids(
            self.evidence_contrast_ids,
            name="evidence_contrast_ids",
        )
        semantic_fields_present = (
            self.insight_kind is not None
            or bool(self.consumer_scopes)
            or bool(self.factor_capabilities)
            or any(
                prediction.comparison_anchor is not None
                for prediction in self.effect_predictions
                if type(prediction) is MetricEffectPrediction
            )
        )
        if semantic_fields_present:
            if type(self.insight_kind) is not ReflectionInsightKind:
                raise TypeError(
                    "semantic insights require an exact ReflectionInsightKind"
                )
            _validate_canonical_enum_values(
                self.consumer_scopes,
                name="consumer_scopes",
                enum_type=ReflectionConsumerScope,
            )
            if self.factor_capabilities:
                _validate_canonical_tokens(
                    self.factor_capabilities,
                    name="factor_capabilities",
                    pattern=_REFLECTION_SEMANTIC_TOKEN,
                )
            if (
                self.insight_kind is ReflectionInsightKind.SEARCH_HEURISTIC
                and self.effect_predictions
            ):
                raise ValueError(
                    "search_heuristic insights cannot carry causal metric predictions"
                )
        elif type(self.consumer_scopes) is not tuple:
            raise TypeError("consumer_scopes must be an exact tuple")
        if type(self.factor_capabilities) is not tuple:
            raise TypeError("factor_capabilities must be an exact tuple")
        advanced_fields_present = (
            bool(self.effect_predictions)
            or bool(self.recommended_option_families)
            or bool(self.recommended_option_ids)
            or self.action_template is not None
            or self.falsification_condition is not None
        )
        if not advanced_fields_present:
            return
        if type(self.effect_predictions) is not tuple or any(
            type(value) is not MetricEffectPrediction
            for value in self.effect_predictions
        ):
            raise TypeError(
                "effect_predictions must contain exact MetricEffectPrediction values"
            )
        if not self.effect_predictions:
            raise ValueError("advanced insights require effect_predictions")
        for prediction in self.effect_predictions:
            MetricEffectPrediction.__post_init__(prediction)
        if semantic_fields_present and any(
            prediction.comparison_anchor is None
            for prediction in self.effect_predictions
        ):
            raise ValueError(
                "semantic insight effect predictions require comparison anchors"
            )
        metric_ids = tuple(
            prediction.metric_id for prediction in self.effect_predictions
        )
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError(
                "effect_predictions must be unique and ordered by metric_id"
            )
        _validate_canonical_tokens(
            self.recommended_option_families,
            name="recommended_option_families",
            pattern=_FINITE_OPTION_FAMILY,
        )
        if self.recommended_option_ids:
            _validate_canonical_tokens(
                self.recommended_option_ids,
                name="recommended_option_ids",
                pattern=_FINITE_OPTION_ID,
            )
        for name in ("action_template", "falsification_condition"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip() or value != value.strip():
                raise ValueError(
                    f"advanced insights require canonical non-empty {name}"
                )

    @property
    def has_intervention_contract(self) -> bool:
        """Whether the opt-in actionable/falsifiable fields are populated."""

        return bool(self.effect_predictions)

    @property
    def has_semantic_contract(self) -> bool:
        """Whether the v3 epistemic/scope vocabulary is populated."""

        return self.insight_kind is not None

    def intervention_record(self) -> dict[str, object] | None:
        """Return a detached JSON-ready projection only for advanced insights."""

        InsightDraft.__post_init__(self)
        if not self.has_intervention_contract:
            return None
        assert self.action_template is not None
        assert self.falsification_condition is not None
        record: dict[str, object] = {
            "effect_predictions": [
                prediction.to_record() for prediction in self.effect_predictions
            ],
            "recommended_option_families": list(self.recommended_option_families),
            "recommended_option_ids": list(self.recommended_option_ids),
            "action_template": self.action_template,
            "falsification_condition": self.falsification_condition,
        }
        semantic = self.semantic_record()
        if semantic is not None:
            record.update(semantic)
        return record

    def semantic_record(self) -> dict[str, object] | None:
        """Return v3 semantics independently of intervention actionability.

        Search heuristics and other outcome schemas may deliberately carry a
        semantic scope without an intervention contract.  Keeping this
        projection separate prevents prompt adapters from silently dropping
        the card kind, authorized consumers, or required factor capabilities.
        """

        InsightDraft.__post_init__(self)
        if not self.has_semantic_contract:
            return None
        assert self.insight_kind is not None
        return {
            "insight_kind": self.insight_kind.value,
            "consumer_scopes": [scope.value for scope in self.consumer_scopes],
            "factor_capabilities": list(self.factor_capabilities),
        }

    def content_record(self) -> dict[str, object]:
        """Return the complete immutable card content used by treatment binding."""

        InsightDraft.__post_init__(self)
        record: dict[str, object] = {
            "schema_version": 2 if self.has_semantic_contract else 1,
            "claim": self.claim,
            "trigger": self.trigger,
            "mechanism": self.mechanism,
            "affected_paths": list(self.affected_paths),
            "evidence_summary": self.evidence_summary,
            "confidence": float(self.confidence),
            "evidence_contrast_ids": list(self.evidence_contrast_ids),
            "effect_predictions": [
                prediction.to_record() for prediction in self.effect_predictions
            ],
            "recommended_option_families": list(self.recommended_option_families),
            "recommended_option_ids": list(self.recommended_option_ids),
            "action_template": self.action_template,
            "falsification_condition": self.falsification_condition,
        }
        if self.has_semantic_contract:
            assert self.insight_kind is not None
            record.update(
                {
                    "insight_kind": self.insight_kind.value,
                    "consumer_scopes": [scope.value for scope in self.consumer_scopes],
                    "factor_capabilities": list(self.factor_capabilities),
                }
            )
        return record

    def hypothesis_record(self) -> dict[str, object]:
        """Project model-authored content with its epistemic status explicit.

        This projection deliberately does not relabel ``evidence_summary`` as an
        observation.  It is the model's interpretation of evidence; trusted
        empirical facts belong in an engine-issued evidence snapshot.
        """

        InsightDraft.__post_init__(self)
        record: dict[str, object] = {
            "schema_version": 2 if self.has_semantic_contract else 1,
            "epistemic_status": "unverified_hypothesis",
            "claim": self.claim,
            "trigger": self.trigger,
            "mechanism_hypothesis": self.mechanism,
            "affected_paths": list(self.affected_paths),
            "evidence_interpretation": self.evidence_summary,
            "confidence": float(self.confidence),
            "evidence_contrast_ids": list(self.evidence_contrast_ids),
            "effect_predictions": [
                prediction.to_record() for prediction in self.effect_predictions
            ],
            "recommended_option_families": list(self.recommended_option_families),
            "recommended_option_ids": list(self.recommended_option_ids),
            "action_template": self.action_template,
            "falsification_condition": self.falsification_condition,
        }
        if self.has_semantic_contract:
            assert self.insight_kind is not None
            record.update(
                {
                    "insight_kind": self.insight_kind.value,
                    "consumer_scopes": [scope.value for scope in self.consumer_scopes],
                    "factor_capabilities": list(self.factor_capabilities),
                }
            )
        return record

    @property
    def hypothesis_sha256(self) -> str:
        """Bind the explicitly unverified hypothesis projection."""

        return hashlib.sha256(
            _INSIGHT_DRAFT_HYPOTHESIS_DOMAIN
            + canonical_typed_json_bytes(freeze_json(self.hypothesis_record()))
        ).hexdigest()

    @property
    def content_sha256(self) -> str:
        """Bind prose, evidence, predictions, and exact finite actions."""

        return hashlib.sha256(
            _INSIGHT_DRAFT_CONTENT_DOMAIN
            + canonical_typed_json_bytes(freeze_json(self.content_record()))
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ReflectionInsightContract:
    """Request-scoped vocabulary for actionable, falsifiable reflection.

    Metric identifiers are benchmark-owned and name quantities whose numeric
    direction can be adjudicated later. Option families are drawn from the
    finite variation vocabulary and remain stable when parent-specific option
    IDs change. Keeping this contract request-scoped avoids imposing any one
    benchmark's metrics or action taxonomy on the core workflow.
    """

    required_metric_ids: Tuple[str, ...]
    allowed_option_families: Tuple[str, ...]
    allowed_option_ids: Tuple[str, ...] = ()
    allowed_decision_paths: Tuple[str, ...] = ()
    allowed_insight_kinds: Tuple[ReflectionInsightKind, ...] = ()
    allowed_consumer_scopes: Tuple[ReflectionConsumerScope, ...] = ()
    allowed_comparison_anchor_kinds: Tuple[MetricComparisonAnchorKind, ...] = ()
    allowed_factor_capabilities: Tuple[str, ...] = ()
    allowed_source_role_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_canonical_tokens(
            self.required_metric_ids,
            name="required_metric_ids",
            pattern=_REFLECTION_METRIC_ID,
        )
        _validate_canonical_tokens(
            self.allowed_option_families,
            name="allowed_option_families",
            pattern=_FINITE_OPTION_FAMILY,
        )
        if self.allowed_option_ids:
            _validate_canonical_tokens(
                self.allowed_option_ids,
                name="allowed_option_ids",
                pattern=_FINITE_OPTION_ID,
            )
        for name in (
            "allowed_decision_paths",
            "allowed_insight_kinds",
            "allowed_consumer_scopes",
            "allowed_comparison_anchor_kinds",
            "allowed_factor_capabilities",
            "allowed_source_role_ids",
        ):
            if type(getattr(self, name)) is not tuple:
                raise TypeError(f"{name} must be an exact tuple")
        semantic_fields_present = any(
            (
                self.allowed_decision_paths,
                self.allowed_insight_kinds,
                self.allowed_consumer_scopes,
                self.allowed_comparison_anchor_kinds,
                self.allowed_factor_capabilities,
                self.allowed_source_role_ids,
            )
        )
        if not semantic_fields_present:
            return
        _validate_decision_paths(
            self.allowed_decision_paths,
            name="allowed_decision_paths",
        )
        _validate_canonical_enum_values(
            self.allowed_insight_kinds,
            name="allowed_insight_kinds",
            enum_type=ReflectionInsightKind,
        )
        _validate_canonical_enum_values(
            self.allowed_consumer_scopes,
            name="allowed_consumer_scopes",
            enum_type=ReflectionConsumerScope,
        )
        _validate_canonical_enum_values(
            self.allowed_comparison_anchor_kinds,
            name="allowed_comparison_anchor_kinds",
            enum_type=MetricComparisonAnchorKind,
        )
        if ReflectionInsightKind.SEARCH_HEURISTIC in self.allowed_insight_kinds:
            raise ValueError(
                "actionable reflection contracts cannot admit search_heuristic "
                "without a separate outcome schema"
            )
        if ReflectionInsightKind.CONTRACT_INVARIANT in self.allowed_insight_kinds:
            raise ValueError(
                "model-authored actionable reflection contracts cannot admit "
                "contract_invariant"
            )
        if self.allowed_factor_capabilities:
            _validate_canonical_tokens(
                self.allowed_factor_capabilities,
                name="allowed_factor_capabilities",
                pattern=_REFLECTION_SEMANTIC_TOKEN,
            )
            if (
                len(self.allowed_factor_capabilities)
                > MAX_REFLECTION_SEMANTIC_VOCABULARY_ENTRIES
            ):
                raise ValueError(
                    "allowed_factor_capabilities exceeds the semantic vocabulary cap"
                )
        elif type(self.allowed_factor_capabilities) is not tuple:
            raise TypeError("allowed_factor_capabilities must be an exact tuple")
        if self.allowed_source_role_ids:
            _validate_canonical_tokens(
                self.allowed_source_role_ids,
                name="allowed_source_role_ids",
                pattern=_REFLECTION_SEMANTIC_TOKEN,
            )
            if (
                len(self.allowed_source_role_ids)
                > MAX_REFLECTION_SEMANTIC_VOCABULARY_ENTRIES
            ):
                raise ValueError(
                    "allowed_source_role_ids exceeds the semantic vocabulary cap"
                )
        elif type(self.allowed_source_role_ids) is not tuple:
            raise TypeError("allowed_source_role_ids must be an exact tuple")
        named_role_allowed = (
            MetricComparisonAnchorKind.NAMED_SOURCE_ROLE
            in self.allowed_comparison_anchor_kinds
        )
        if named_role_allowed and not self.allowed_source_role_ids:
            raise ValueError(
                "named_source_role anchors require adapter-owned source role IDs"
            )
        if self.allowed_source_role_ids and not named_role_allowed:
            raise ValueError(
                "source role IDs require named_source_role in the anchor vocabulary"
            )

    @property
    def is_semantic_v3(self) -> bool:
        """Whether this request opts into the fail-closed semantic contract."""

        return bool(self.allowed_decision_paths)

    def to_record(self) -> dict[str, object]:
        ReflectionInsightContract.__post_init__(self)
        record: dict[str, object] = {
            "schema_version": 3 if self.is_semantic_v3 else 2,
            "contract_identity_sha256": self.identity_sha256,
            "required_metric_ids": list(self.required_metric_ids),
            "allowed_option_families": list(self.allowed_option_families),
            "allowed_option_ids": list(self.allowed_option_ids),
            "direction_vocabulary": [
                direction.value for direction in MetricEffectDirection
            ],
        }
        if self.is_semantic_v3:
            record.update(
                {
                    "allowed_decision_paths": list(self.allowed_decision_paths),
                    "allowed_insight_kinds": [
                        kind.value for kind in self.allowed_insight_kinds
                    ],
                    "allowed_consumer_scopes": [
                        scope.value for scope in self.allowed_consumer_scopes
                    ],
                    "allowed_comparison_anchor_kinds": [
                        kind.value for kind in self.allowed_comparison_anchor_kinds
                    ],
                    "allowed_factor_capabilities": list(
                        self.allowed_factor_capabilities
                    ),
                    "allowed_source_role_ids": list(self.allowed_source_role_ids),
                }
            )
        return record

    @property
    def identity_sha256(self) -> str:
        """Bind exact ordered metric/action vocabularies for replay."""

        ReflectionInsightContract.__post_init__(self)
        record: dict[str, object] = {
            "schema_version": 3 if self.is_semantic_v3 else 2,
            "required_metric_ids": list(self.required_metric_ids),
            "allowed_option_families": list(self.allowed_option_families),
            "allowed_option_ids": list(self.allowed_option_ids),
            "direction_vocabulary": [
                direction.value for direction in MetricEffectDirection
            ],
        }
        if self.is_semantic_v3:
            record.update(
                {
                    "allowed_decision_paths": list(self.allowed_decision_paths),
                    "allowed_insight_kinds": [
                        kind.value for kind in self.allowed_insight_kinds
                    ],
                    "allowed_consumer_scopes": [
                        scope.value for scope in self.allowed_consumer_scopes
                    ],
                    "allowed_comparison_anchor_kinds": [
                        kind.value for kind in self.allowed_comparison_anchor_kinds
                    ],
                    "allowed_factor_capabilities": list(
                        self.allowed_factor_capabilities
                    ),
                    "allowed_source_role_ids": list(self.allowed_source_role_ids),
                }
            )
        payload = freeze_json(record)
        return hashlib.sha256(
            (
                _REFLECTION_INSIGHT_CONTRACT_V3_DOMAIN
                if self.is_semantic_v3
                else _REFLECTION_INSIGHT_CONTRACT_DOMAIN
            )
            + canonical_typed_json_bytes(payload)
        ).hexdigest()


def validate_reflection_insight_draft(
    draft: InsightDraft,
    contract: ReflectionInsightContract,
    *,
    allow_all_unknown: bool = False,
    allow_missing_evidence: bool = False,
) -> None:
    """Verify exact metric coverage and finite-vocabulary actionability."""

    if type(allow_all_unknown) is not bool:
        raise TypeError("allow_all_unknown must be an exact boolean")
    if type(allow_missing_evidence) is not bool:
        raise TypeError("allow_missing_evidence must be an exact boolean")
    if type(draft) is not InsightDraft:
        raise TypeError("draft must be an exact InsightDraft")
    InsightDraft.__post_init__(draft)
    if type(contract) is not ReflectionInsightContract:
        raise TypeError("contract must be an exact ReflectionInsightContract")
    ReflectionInsightContract.__post_init__(contract)
    if not draft.has_intervention_contract:
        raise ValueError("insight is missing the advanced intervention contract")
    if contract.is_semantic_v3:
        if not draft.has_semantic_contract:
            raise ValueError("v3 insight is missing its semantic contract")
        assert draft.insight_kind is not None
        if draft.insight_kind is ReflectionInsightKind.CONTRACT_INVARIANT:
            raise ValueError(
                "model-authored v3 reflections cannot assert contract_invariant"
            )
        if not draft.affected_paths or len(set(draft.affected_paths)) != len(
            draft.affected_paths
        ):
            raise ValueError("v3 affected paths must be nonempty and unique")
        if draft.insight_kind not in contract.allowed_insight_kinds:
            raise ValueError("insight kind escapes the request vocabulary")
        if not set(draft.consumer_scopes).issubset(contract.allowed_consumer_scopes):
            raise ValueError("consumer scopes escape the request vocabulary")
        if not set(draft.affected_paths).issubset(contract.allowed_decision_paths):
            raise ValueError(
                "affected paths escape the adapter-owned decision-path vocabulary"
            )
        if not set(draft.factor_capabilities).issubset(
            contract.allowed_factor_capabilities
        ):
            raise ValueError(
                "factor capabilities escape the adapter-owned capability vocabulary"
            )
        for prediction in draft.effect_predictions:
            if prediction.direction is MetricEffectDirection.UNKNOWN:
                raise ValueError(
                    "v3 lifecycle hypotheses require adjudicable metric directions"
                )
            anchor = prediction.comparison_anchor
            if anchor is None:
                raise ValueError(
                    "v3 effect predictions require an explicit comparison anchor"
                )
            if anchor.kind not in contract.allowed_comparison_anchor_kinds:
                raise ValueError(
                    "comparison anchor kind escapes the request vocabulary"
                )
            if (
                anchor.source_role_id is not None
                and anchor.source_role_id not in contract.allowed_source_role_ids
            ):
                raise ValueError(
                    "comparison source role escapes the request vocabulary"
                )
    elif draft.has_semantic_contract:
        raise ValueError("a v3 semantic insight requires a v3 reflection contract")
    if not allow_missing_evidence and not draft.evidence_contrast_ids:
        raise ValueError(
            "an outcome-grounded intervention must cite at least one evidence contrast"
        )
    predicted_metric_ids = tuple(
        prediction.metric_id for prediction in draft.effect_predictions
    )
    if predicted_metric_ids != contract.required_metric_ids:
        raise ValueError(
            "effect predictions must cover the exact required metric identifiers"
        )
    if not set(draft.recommended_option_families).issubset(
        contract.allowed_option_families
    ):
        raise ValueError(
            "recommended option families escape the finite action vocabulary"
        )
    if contract.allowed_option_ids:
        if not draft.recommended_option_ids:
            raise ValueError(
                "an exact-action insight must recommend at least one option ID"
            )
        if not set(draft.recommended_option_ids).issubset(contract.allowed_option_ids):
            raise ValueError(
                "recommended option IDs escape the finite action vocabulary"
            )
    elif draft.recommended_option_ids:
        raise ValueError(
            "recommended option IDs require a request-scoped exact-action vocabulary"
        )
    if not allow_all_unknown and all(
        prediction.direction is MetricEffectDirection.UNKNOWN
        for prediction in draft.effect_predictions
    ):
        raise ValueError(
            "an outcome-grounded intervention must make a directional prediction"
        )


@dataclass(frozen=True, slots=True)
class AgenticCallTelemetry:
    requested_model: str
    resolved_model: str
    resolved_provider: str
    provider_response_id: str | None
    finish_reason: str | None
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: Decimal | None
    latency_ns: int
    attempt_count: int = 1

    def __post_init__(self) -> None:
        for name in ("requested_model", "resolved_model", "resolved_provider"):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "latency_ns",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.attempt_count) is not int or self.attempt_count <= 0:
            raise ValueError("attempt_count must be positive")


@dataclass(frozen=True, slots=True)
class VariationGenerationRequest:
    call_id: LLMCallId
    operation: str
    prompt: str
    candidate_model: type
    max_output_tokens: int = 2_048
    temperature: float | None = None
    atomic_mutation_contract: AtomicMutationOutputContract | None = None
    finite_variation_contract: FiniteVariationContract | None = None
    exact_parent_crossover_contract: ExactParentCrossoverOutputContract | None = None

    def __post_init__(self) -> None:
        contracts = tuple(
            value
            for value in (
                self.atomic_mutation_contract,
                self.finite_variation_contract,
                self.exact_parent_crossover_contract,
            )
            if value is not None
        )
        if len(contracts) > 1:
            raise ValueError("variation output contracts are mutually exclusive")
        if self.atomic_mutation_contract is not None:
            if type(self.atomic_mutation_contract) is not AtomicMutationOutputContract:
                raise TypeError(
                    "atomic_mutation_contract must be an exact "
                    "AtomicMutationOutputContract"
                )
            AtomicMutationOutputContract.__post_init__(self.atomic_mutation_contract)
        if self.finite_variation_contract is not None:
            validate_finite_variation_contract(self.finite_variation_contract)
        if self.exact_parent_crossover_contract is not None:
            if (
                type(self.exact_parent_crossover_contract)
                is not ExactParentCrossoverOutputContract
            ):
                raise TypeError("exact_parent_crossover_contract must be exact")
            ExactParentCrossoverOutputContract.__post_init__(
                self.exact_parent_crossover_contract
            )
            if self.operation != "two_parent_crossover":
                raise ValueError(
                    "exact parent crossover output is restricted to "
                    "two_parent_crossover"
                )


@dataclass(frozen=True, slots=True)
class ReflectionGenerationRequest:
    call_id: LLMCallId
    operation: str
    prompt: str
    max_insights: int = 4
    min_insights: int = 0
    max_output_tokens: int = 2_048
    temperature: float | None = None
    available_contrast_ids: Tuple[str, ...] = ()
    insight_contract: ReflectionInsightContract | None = None
    evidence_catalog: ReflectionEvidenceCatalog | None = None

    def __post_init__(self) -> None:
        if type(self.max_insights) is not int or not 1 <= self.max_insights <= 16:
            raise ValueError("max_insights must lie in [1,16]")
        if (
            type(self.min_insights) is not int
            or not 0 <= self.min_insights <= self.max_insights
        ):
            raise ValueError("min_insights must lie in [0,max_insights]")
        _validate_contrast_ids(
            self.available_contrast_ids,
            name="available_contrast_ids",
        )
        if self.evidence_catalog is not None:
            if type(self.evidence_catalog) is not ReflectionEvidenceCatalog:
                raise TypeError(
                    "evidence_catalog must be an exact ReflectionEvidenceCatalog or None"
                )
            ReflectionEvidenceCatalog.__post_init__(self.evidence_catalog)
            if self.evidence_catalog.contrast_ids != self.available_contrast_ids:
                raise ValueError(
                    "evidence_catalog must bind the exact available_contrast_ids"
                )
        if self.insight_contract is not None:
            if type(self.insight_contract) is not ReflectionInsightContract:
                raise TypeError(
                    "insight_contract must be an exact ReflectionInsightContract"
                )
            ReflectionInsightContract.__post_init__(self.insight_contract)


@dataclass(frozen=True, slots=True)
class VariationGenerationResult:
    draft: (
        CandidateDraft
        | AtomicMutationDraft
        | FiniteVariationSelectionDraft
        | ExactParentCrossoverDraft
    )
    telemetry: AgenticCallTelemetry


@dataclass(frozen=True, slots=True)
class ReflectionGenerationResult:
    insights: Tuple[InsightDraft, ...]
    telemetry: AgenticCallTelemetry
    evidence_catalog_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.evidence_catalog_identity_sha256 is not None:
            require_sha256(
                self.evidence_catalog_identity_sha256,
                "evidence_catalog_identity_sha256",
            )


def validate_reflection_evidence_catalog_result(
    request: ReflectionGenerationRequest,
    result: ReflectionGenerationResult,
) -> None:
    """Bind resolved full-ID citations and catalog identity to one request."""

    if type(request) is not ReflectionGenerationRequest:
        raise TypeError("request must be an exact ReflectionGenerationRequest")
    ReflectionGenerationRequest.__post_init__(request)
    if type(result) is not ReflectionGenerationResult:
        raise TypeError("result must be an exact ReflectionGenerationResult")
    ReflectionGenerationResult.__post_init__(result)
    if type(result.insights) is not tuple or any(
        type(insight) is not InsightDraft for insight in result.insights
    ):
        raise TypeError("result insights must contain exact InsightDraft values")
    available = set(request.available_contrast_ids)
    for insight in result.insights:
        InsightDraft.__post_init__(insight)
        if not set(insight.evidence_contrast_ids).issubset(available):
            raise ValueError("resolved reflection citations escaped the request")
    expected_identity = (
        None
        if request.evidence_catalog is None
        else request.evidence_catalog.catalog_identity_sha256
    )
    if result.evidence_catalog_identity_sha256 != expected_identity:
        raise ValueError("reflection result has a foreign evidence catalog identity")


@runtime_checkable
class AgenticGenerator(Protocol):
    async def propose(
        self, request: VariationGenerationRequest
    ) -> VariationGenerationResult: ...

    async def reflect(
        self, request: ReflectionGenerationRequest
    ) -> ReflectionGenerationResult: ...


__all__ = [
    "AgenticCallTelemetry",
    "AgenticGenerator",
    "AtomicMutationDraft",
    "AtomicMutationOutputContract",
    "CANDIDATE_COMPONENT_PATH_CONTRACT",
    "TWO_PARENT_CROSSOVER_EVIDENCE_CONTRACT",
    "CandidateDraft",
    "ConflictResolutionDraft",
    "ExactParentCrossoverDraft",
    "ExactParentCrossoverOutputContract",
    "FiniteVariationSelectionDraft",
    "InsightDraft",
    "MetricComparisonAnchor",
    "MetricComparisonAnchorKind",
    "MetricEffectDirection",
    "MetricEffectPrediction",
    "MAX_REFLECTION_EVIDENCE_CATALOG_ENTRIES",
    "MAX_REFLECTION_SEMANTIC_VOCABULARY_ENTRIES",
    "MAX_REFLECTION_DECISION_PATH_CHARS",
    "ReflectionEvidenceCatalog",
    "ReflectionEvidenceCatalogEntry",
    "ReflectionGenerationRequest",
    "ReflectionGenerationResult",
    "ReflectionConsumerScope",
    "ReflectionInsightContract",
    "ReflectionInsightKind",
    "resolve_finite_variation_selection",
    "SourceAttribution",
    "VariationGenerationRequest",
    "VariationGenerationResult",
    "validate_reflection_insight_draft",
    "validate_reflection_evidence_catalog_result",
]
