"""Executable, benchmark-neutral two-parent crossover inheritance.

Model-authored crossover output contains two different kinds of information:

* an inheritance plan (candidate-relative paths attributed to the left or
  right parent), and
* explicitly synthesized values.

An inherited value is not model authority.  This service verifies the plan and
copies the exact immutable subtree from the named parent.  The model's value at
that path is retained only as a consistency witness.  Exact typed-JSON equality
is required except for finite binary64 leaves that differ by at most one ULP,
which covers decimal spelling versus arithmetic-rounding drift without turning
the boundary into an optimization tolerance.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from enum import Enum

from agent_evolve.domain.patch import (
    ArrayIndex,
    JsonPath,
    ObjectKey,
    canonical_path_bytes,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonArray,
    FrozenJsonObject,
    FrozenJsonValue,
    freeze_json,
    typed_json_equal,
    typed_json_sha256,
)
from agent_evolve.policies.variation.typed_patch import (
    replace_existing_path,
    value_at_path,
)


_RECEIPT_DOMAIN = b"agent-evolve:crossover-inheritance-materialization:v1\x00"
_MISSING = object()
_BINARY64_SIGN_BIT = 1 << 63
_BINARY64_MASK = (1 << 64) - 1


class CrossoverInheritanceSource(str, Enum):
    """Closed source vocabulary for executable crossover claims."""

    LEFT = "left"
    RIGHT = "right"
    SYNTHESIZED = "synthesized"


@dataclass(frozen=True, slots=True)
class CrossoverInheritanceClaim:
    """One exact candidate-relative component claim."""

    path: str
    source: CrossoverInheritanceSource

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path.startswith("$."):
            raise ValueError("crossover claim path must be a non-root canonical path")
        if type(self.source) is not CrossoverInheritanceSource:
            raise TypeError("crossover claim source must be closed and exact")


@dataclass(frozen=True, slots=True)
class InheritedPathMaterializationEvidence:
    """Hash-only evidence for one exact parent-subtree copy."""

    path: str
    source: CrossoverInheritanceSource
    witness_value_sha256: str
    parent_value_sha256: str
    witness_exact: bool
    adjusted_float_leaf_count: int
    max_float_ulp_distance: int

    def to_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "source": self.source.value,
            "witness_value_sha256": self.witness_value_sha256,
            "parent_value_sha256": self.parent_value_sha256,
            "witness_exact": self.witness_exact,
            "adjusted_float_leaf_count": self.adjusted_float_leaf_count,
            "max_float_ulp_distance": self.max_float_ulp_distance,
        }


@dataclass(frozen=True, slots=True)
class SynthesizedPathEvidence:
    """Hash-only evidence for one explicitly model-authored component."""

    path: str
    witness_value_sha256: str
    left_value_sha256: str | None
    right_value_sha256: str | None

    def to_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "witness_value_sha256": self.witness_value_sha256,
            "left_value_sha256": self.left_value_sha256,
            "right_value_sha256": self.right_value_sha256,
        }


@dataclass(frozen=True, slots=True)
class CrossoverInheritanceMaterialization:
    """Exact child plus replay-visible materialization evidence."""

    configuration: FrozenJsonObject
    draft_configuration_sha256: str
    materialized_configuration_sha256: str
    inherited_paths: tuple[InheritedPathMaterializationEvidence, ...]
    synthesized_paths: tuple[SynthesizedPathEvidence, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "materialization_policy": "exact_named_parent_subtree_copy_v1",
            "witness_consistency_policy": (
                "typed_json_exact_or_one_finite_binary64_ulp_per_float_leaf_v1"
            ),
            "attribution_policy": "exhaustive_nonoverlapping_component_plan_v1",
            "draft_configuration_sha256": self.draft_configuration_sha256,
            "materialized_configuration_sha256": (
                self.materialized_configuration_sha256
            ),
            "inherited_paths": [item.to_record() for item in self.inherited_paths],
            "synthesized_paths": [
                item.to_record() for item in self.synthesized_paths
            ],
        }

    @property
    def receipt_sha256(self) -> str:
        encoded = json.dumps(
            self.to_record(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii", errors="strict")
        return hashlib.sha256(_RECEIPT_DOMAIN + encoded).hexdigest()


def _path_text(path: JsonPath) -> str:
    parts = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            parts.append(f".{segment.value}")
        elif type(segment) is ArrayIndex:
            parts.append(f"[{segment.value}]")
        else:  # pragma: no cover - JsonPath closes the union.
            raise AssertionError("unsupported path segment")
    return "".join(parts)


def _path_index(
    root: FrozenJsonValue,
) -> tuple[dict[str, JsonPath], frozenset[str]]:
    """Index canonical display paths and surface any ambiguous spelling."""

    values: dict[str, JsonPath] = {}
    ambiguous: set[str] = set()

    def visit(value: FrozenJsonValue, path: JsonPath) -> None:
        if path.segments:
            text = _path_text(path)
            prior = values.get(text)
            if prior is not None and canonical_path_bytes(prior) != canonical_path_bytes(
                path
            ):
                ambiguous.add(text)
            else:
                values[text] = path
        if type(value) is FrozenJsonObject:
            for key, child in value.items:
                visit(child, path.child_key(key))
        elif type(value) is FrozenJsonArray:
            for index, child in enumerate(value.items):
                visit(child, path.child_index(index))

    visit(root, JsonPath())
    return values, frozenset(ambiguous)


def _witness_equivalence(
    witness: FrozenJsonValue,
    parent: FrozenJsonValue,
) -> tuple[bool, int, int]:
    """Return exact/one-ULP structural consistency evidence.

    The integer fields are adjusted float-leaf count and maximum ULP distance.
    They are zero for exact typed-JSON equality and at most one otherwise.
    """

    if typed_json_equal(witness, parent):
        return True, 0, 0
    if type(witness) is not type(parent):
        return False, 0, 0
    if type(witness) is float:
        assert type(parent) is float
        # Finite values are guaranteed by the frozen typed-JSON boundary.  Do
        # not collapse signed zero: it has an intentional exact typed identity.
        if witness == 0.0 and parent == 0.0:
            return False, 0, 0
        witness_bits = int.from_bytes(struct.pack(">d", witness), "big")
        parent_bits = int.from_bytes(struct.pack(">d", parent), "big")

        def ordered(bits: int) -> int:
            if bits & _BINARY64_SIGN_BIT:
                return (~bits) & _BINARY64_MASK
            return bits | _BINARY64_SIGN_BIT

        if abs(ordered(witness_bits) - ordered(parent_bits)) == 1:
            return True, 1, 1
        return False, 0, 0
    if type(witness) is FrozenJsonArray:
        assert type(parent) is FrozenJsonArray
        if len(witness.items) != len(parent.items):
            return False, 0, 0
        adjusted = 0
        maximum = 0
        for witness_item, parent_item in zip(
            witness.items, parent.items, strict=True
        ):
            equivalent, item_adjusted, item_maximum = _witness_equivalence(
                witness_item, parent_item
            )
            if not equivalent:
                return False, 0, 0
            adjusted += item_adjusted
            maximum = max(maximum, item_maximum)
        return True, adjusted, maximum
    if type(witness) is FrozenJsonObject:
        assert type(parent) is FrozenJsonObject
        if tuple(key for key, _ in witness.items) != tuple(
            key for key, _ in parent.items
        ):
            return False, 0, 0
        adjusted = 0
        maximum = 0
        for (_, witness_item), (_, parent_item) in zip(
            witness.items, parent.items, strict=True
        ):
            equivalent, item_adjusted, item_maximum = _witness_equivalence(
                witness_item, parent_item
            )
            if not equivalent:
                return False, 0, 0
            adjusted += item_adjusted
            maximum = max(maximum, item_maximum)
        return True, adjusted, maximum
    return False, 0, 0


def _optional_equal(left: object, right: object) -> bool:
    if left is _MISSING or right is _MISSING:
        return left is right
    return typed_json_equal(left, right)


def _audit_exhaustive_plan(
    left: FrozenJsonObject,
    right: FrozenJsonObject,
    child: FrozenJsonObject,
    claims_by_path: dict[bytes, CrossoverInheritanceClaim],
) -> None:
    """Require every parent/child difference to have one exact source claim."""

    consumed: set[bytes] = set()

    def visit(
        left_value: FrozenJsonValue | object,
        right_value: FrozenJsonValue | object,
        child_value: FrozenJsonValue | object,
        path: JsonPath,
    ) -> None:
        encoded_path = canonical_path_bytes(path)
        claim = claims_by_path.get(encoded_path)
        if claim is not None:
            consumed.add(encoded_path)
            if child_value is _MISSING:
                raise ValueError("crossover claim path is absent from the child")
            if claim.source is CrossoverInheritanceSource.LEFT:
                if left_value is _MISSING or right_value is _MISSING:
                    raise ValueError("left inheritance path is absent from a parent")
                if _optional_equal(left_value, right_value) or not _optional_equal(
                    child_value, left_value
                ):
                    raise ValueError("left inheritance claim is not discriminating")
            elif claim.source is CrossoverInheritanceSource.RIGHT:
                if left_value is _MISSING or right_value is _MISSING:
                    raise ValueError("right inheritance path is absent from a parent")
                if _optional_equal(left_value, right_value) or not _optional_equal(
                    child_value, right_value
                ):
                    raise ValueError("right inheritance claim is not discriminating")
            else:
                if _optional_equal(child_value, left_value) or _optional_equal(
                    child_value, right_value
                ):
                    raise ValueError(
                        "synthesized claim must differ from both parent values"
                    )
            return

        if child_value is _MISSING:
            # Absence is a child value, not preservation.  In particular, an
            # identically shared parent member cannot silently disappear just
            # because neither parent discriminates it.  Since claims name
            # paths that exist in the child, a structural deletion must be
            # claimed as synthesis at a retained containing object/array.
            if _optional_equal(left_value, right_value):
                raise ValueError(
                    "crossover omits a shared component without synthesized "
                    "container attribution"
                )
            raise ValueError("crossover omits a discriminating component without a claim")

        if (
            type(left_value) is FrozenJsonObject
            and type(right_value) is FrozenJsonObject
            and type(child_value) is FrozenJsonObject
        ):
            left_items = dict(left_value.items)
            right_items = dict(right_value.items)
            child_items = dict(child_value.items)
            for key in sorted(
                set(left_items) | set(right_items) | set(child_items),
                key=lambda item: item.encode("utf-8", errors="strict"),
            ):
                visit(
                    left_items.get(key, _MISSING),
                    right_items.get(key, _MISSING),
                    child_items.get(key, _MISSING),
                    path.child_key(key),
                )
            return

        if (
            type(left_value) is FrozenJsonArray
            and type(right_value) is FrozenJsonArray
            and type(child_value) is FrozenJsonArray
            and len(left_value.items)
            == len(right_value.items)
            == len(child_value.items)
        ):
            for index, (left_item, right_item, child_item) in enumerate(
                zip(
                    left_value.items,
                    right_value.items,
                    child_value.items,
                    strict=True,
                )
            ):
                visit(
                    left_item,
                    right_item,
                    child_item,
                    path.child_index(index),
                )
            return

        if _optional_equal(left_value, right_value):
            if not _optional_equal(child_value, left_value):
                raise ValueError(
                    "crossover synthesized a component without explicit attribution"
                )
            return
        if _optional_equal(child_value, left_value):
            raise ValueError("crossover inherited a left component without attribution")
        if _optional_equal(child_value, right_value):
            raise ValueError("crossover inherited a right component without attribution")
        raise ValueError("crossover synthesized a component without explicit attribution")

    visit(left, right, child, JsonPath())
    if consumed != set(claims_by_path):
        raise ValueError("crossover plan contains an unreachable component claim")


def materialize_crossover_inheritance(
    *,
    left: object,
    right: object,
    draft: object,
    claims: tuple[CrossoverInheritanceClaim, ...],
) -> CrossoverInheritanceMaterialization:
    """Verify an executable inheritance plan and materialize its exact child."""

    frozen_left = freeze_json(left)
    frozen_right = freeze_json(right)
    frozen_draft = freeze_json(draft)
    if any(
        type(value) is not FrozenJsonObject
        for value in (frozen_left, frozen_right, frozen_draft)
    ):
        raise TypeError("two-parent crossover roots must be exact JSON objects")
    assert type(frozen_left) is FrozenJsonObject
    assert type(frozen_right) is FrozenJsonObject
    assert type(frozen_draft) is FrozenJsonObject
    if type(claims) is not tuple or any(
        type(claim) is not CrossoverInheritanceClaim for claim in claims
    ):
        raise TypeError("crossover claims must be an exact tuple of exact claims")
    for claim in claims:
        CrossoverInheritanceClaim.__post_init__(claim)
    if not claims:
        raise ValueError("two-parent crossover requires an executable source plan")

    draft_index, draft_ambiguous = _path_index(frozen_draft)
    left_index, left_ambiguous = _path_index(frozen_left)
    right_index, right_ambiguous = _path_index(frozen_right)
    if any(claim.path in draft_ambiguous for claim in claims):
        raise ValueError("crossover claim has an ambiguous candidate path spelling")

    resolved: list[tuple[CrossoverInheritanceClaim, JsonPath]] = []
    seen_text: set[str] = set()
    for claim in claims:
        if claim.path in seen_text:
            raise ValueError("crossover claim paths must be unique")
        seen_text.add(claim.path)
        path = draft_index.get(claim.path)
        if path is None:
            raise ValueError("crossover claim path does not exist in the draft")
        resolved.append((claim, path))

    for index, (_, path) in enumerate(resolved):
        for _, other in resolved[index + 1 :]:
            if path.is_prefix_of(other) or other.is_prefix_of(path):
                raise ValueError("crossover claim paths must not overlap")

    inherited: list[InheritedPathMaterializationEvidence] = []
    synthesized: list[SynthesizedPathEvidence] = []
    materialized: FrozenJsonValue = frozen_draft
    source_counts = {
        CrossoverInheritanceSource.LEFT: 0,
        CrossoverInheritanceSource.RIGHT: 0,
    }
    claims_by_path: dict[bytes, CrossoverInheritanceClaim] = {}

    for claim, draft_path in resolved:
        encoded_path = canonical_path_bytes(draft_path)
        claims_by_path[encoded_path] = claim
        witness_value = value_at_path(frozen_draft, draft_path)
        if claim.source in (
            CrossoverInheritanceSource.LEFT,
            CrossoverInheritanceSource.RIGHT,
        ):
            if claim.path in left_ambiguous or claim.path in right_ambiguous:
                raise ValueError("inherited claim is ambiguous in a parent")
            left_path = left_index.get(claim.path)
            right_path = right_index.get(claim.path)
            if left_path is None or right_path is None:
                raise ValueError("inherited claim path must exist in both parents")
            if (
                canonical_path_bytes(left_path) != encoded_path
                or canonical_path_bytes(right_path) != encoded_path
            ):
                raise ValueError("inherited claim path differs across candidate trees")
            left_value = value_at_path(frozen_left, left_path)
            right_value = value_at_path(frozen_right, right_path)
            if typed_json_equal(left_value, right_value):
                raise ValueError("inherited claim path is not parent-discriminating")
            parent_value = (
                left_value
                if claim.source is CrossoverInheritanceSource.LEFT
                else right_value
            )
            equivalent, adjusted, maximum = _witness_equivalence(
                witness_value, parent_value
            )
            if not equivalent:
                raise ValueError(
                    "inherited witness differs materially from its named parent"
                )
            materialized = replace_existing_path(
                materialized,
                draft_path,
                parent_value,
            )
            source_counts[claim.source] += 1
            inherited.append(
                InheritedPathMaterializationEvidence(
                    path=claim.path,
                    source=claim.source,
                    witness_value_sha256=typed_json_sha256(witness_value),
                    parent_value_sha256=typed_json_sha256(parent_value),
                    witness_exact=typed_json_equal(witness_value, parent_value),
                    adjusted_float_leaf_count=adjusted,
                    max_float_ulp_distance=maximum,
                )
            )
        else:
            left_path = None if claim.path in left_ambiguous else left_index.get(claim.path)
            right_path = (
                None if claim.path in right_ambiguous else right_index.get(claim.path)
            )
            if claim.path in left_ambiguous or claim.path in right_ambiguous:
                raise ValueError("synthesized claim is ambiguous in a parent")
            left_value = (
                _MISSING
                if left_path is None
                else value_at_path(frozen_left, left_path)
            )
            right_value = (
                _MISSING
                if right_path is None
                else value_at_path(frozen_right, right_path)
            )
            if _optional_equal(witness_value, left_value) or _optional_equal(
                witness_value, right_value
            ):
                raise ValueError(
                    "synthesized claim must author a value distinct from both parents"
                )
            synthesized.append(
                SynthesizedPathEvidence(
                    path=claim.path,
                    witness_value_sha256=typed_json_sha256(witness_value),
                    left_value_sha256=(
                        None
                        if left_value is _MISSING
                        else typed_json_sha256(left_value)
                    ),
                    right_value_sha256=(
                        None
                        if right_value is _MISSING
                        else typed_json_sha256(right_value)
                    ),
                )
            )

    if not all(source_counts.values()):
        raise ValueError(
            "two-parent crossover requires a discriminating claim from both parents"
        )
    if type(materialized) is not FrozenJsonObject:  # pragma: no cover - root forbidden.
        raise AssertionError("non-root inheritance changed the candidate root type")
    _audit_exhaustive_plan(
        frozen_left,
        frozen_right,
        materialized,
        claims_by_path,
    )

    result = CrossoverInheritanceMaterialization(
        configuration=materialized,
        draft_configuration_sha256=typed_json_sha256(frozen_draft),
        materialized_configuration_sha256=typed_json_sha256(materialized),
        inherited_paths=tuple(sorted(inherited, key=lambda item: item.path)),
        synthesized_paths=tuple(sorted(synthesized, key=lambda item: item.path)),
    )
    # Exercise the receipt codec at the trust boundary before returning it.
    result.receipt_sha256
    return result


__all__ = [
    "CrossoverInheritanceClaim",
    "CrossoverInheritanceMaterialization",
    "CrossoverInheritanceSource",
    "InheritedPathMaterializationEvidence",
    "SynthesizedPathEvidence",
    "materialize_crossover_inheritance",
]
