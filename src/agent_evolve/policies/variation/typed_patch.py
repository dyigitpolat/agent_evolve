"""Deterministic derivation, replay, inversion, and three-way classification.

No function in this module calls a model or performs a merge.  Three-way
classification produces explicit evidence obligations; a later operator may
choose or synthesize resolutions, but it cannot make conflicts disappear.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.lineage import (
    MAX_PRESERVATION_OBLIGATIONS,
    AbsenceContextKind,
    AbsenceFailureKind,
    CandidateOccurrence,
    PreservationClaim,
    PreservationExpectation,
    PreservationObligation,
    PreservationSource,
    VariationCase,
    VariationKind,
    validate_variation_case,
)
from agent_evolve.domain.patch import (
    DEFAULT_PATCH_LIMITS,
    MAX_PATCH_OPERATIONS,
    ArrayIndex,
    DeleteSequenceItem,
    InsertSequenceItem,
    JsonPath,
    ObjectKey,
    PatchLimits,
    PatchOperation,
    PermuteSequence,
    ReplaceScalar,
    ReplaceSubtree,
    TypedPatch,
    canonical_path_bytes,
    operation_effect_bytes,
    operation_effect_sha256,
    operation_occurrence_bytes,
    operation_kind,
    operation_sort_key,
    require_sha256,
    validate_json_path,
    validate_patch_limits,
    validate_patch_operation,
    validate_semantic_component,
    validate_typed_patch,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonArray,
    FrozenJsonObject,
    FrozenJsonValue,
    TypedJsonLimits,
    canonical_typed_json_bytes,
    freeze_json,
    is_json_scalar,
    typed_json_equal,
    typed_json_sha256,
    validate_typed_json_limits,
)


class PatchPreconditionError(ValueError):
    """The supplied base or an operation-local old value was stale."""


class PatchPostconditionError(ValueError):
    """An operation or whole patch did not produce its declared endpoint."""


class PreservationError(ValueError):
    """A parent-use or preservation claim was absent, fabricated, or false."""


@dataclass(frozen=True, slots=True, eq=False)
class ComponentTagAssignment:
    """Benchmark-supplied semantic component for one exact operation path."""

    path: JsonPath
    component: str

    def __post_init__(self) -> None:
        validate_json_path(self.path)
        if type(self.component) is not str:
            raise TypeError("component tag must be an exact string")
        validate_semantic_component(self.component)

    def __eq__(self, other: object) -> bool:
        if (
            type(self) is not ComponentTagAssignment
            or type(other) is not ComponentTagAssignment
        ):
            return False
        ComponentTagAssignment.__post_init__(self)
        ComponentTagAssignment.__post_init__(other)
        return (
            canonical_path_bytes(self.path),
            self.component,
        ) == (
            canonical_path_bytes(other.path),
            other.component,
        )

    __hash__ = None


def _component_assignments(
    values: tuple[ComponentTagAssignment, ...],
    *,
    max_assignments: int,
) -> dict[bytes, str]:
    if type(values) is not tuple:
        raise TypeError(
            "component_tags must be an exact tuple of ComponentTagAssignment values"
        )
    if len(values) > max_assignments:
        raise ValueError("component_tags exceeds the patch max_operations bound")
    if any(type(value) is not ComponentTagAssignment for value in values):
        raise TypeError(
            "component_tags must be an exact tuple of ComponentTagAssignment values"
        )
    for value in values:
        ComponentTagAssignment.__post_init__(value)
    canonical = tuple(sorted(values, key=lambda value: canonical_path_bytes(value.path)))
    if values != canonical:
        raise ValueError("component_tags must use canonical typed-path order")
    result: dict[bytes, str] = {}
    for value in values:
        key = canonical_path_bytes(value.path)
        if key in result:
            raise ValueError("component_tags cannot duplicate an operation path")
        result[key] = value.component
    return result


def _object_dict(value: FrozenJsonObject) -> dict[str, FrozenJsonValue]:
    return {key: item for key, item in value.items}


def _stable_permutation(
    before: FrozenJsonArray,
    after: FrozenJsonArray,
    *,
    limits: TypedJsonLimits,
) -> tuple[int, ...] | None:
    if len(before.items) != len(after.items):
        return None
    positions: dict[bytes, list[int]] = {}
    for index, item in enumerate(before.items):
        item_bytes = canonical_typed_json_bytes(item, limits=limits)
        positions.setdefault(item_bytes, []).append(index)
    cursors: dict[bytes, int] = {}
    result: list[int] = []
    for item in after.items:
        item_bytes = canonical_typed_json_bytes(item, limits=limits)
        cursor = cursors.get(item_bytes, 0)
        candidates = positions.get(item_bytes)
        if candidates is None or cursor >= len(candidates):
            return None
        result.append(candidates[cursor])
        cursors[item_bytes] = cursor + 1
    if any(cursors.get(key, 0) != len(indices) for key, indices in positions.items()):
        return None
    return tuple(result)


def _item_bytes(
    sequence: FrozenJsonArray,
    *,
    limits: TypedJsonLimits,
) -> tuple[bytes, ...]:
    return tuple(
        canonical_typed_json_bytes(item, limits=limits) for item in sequence.items
    )


def _single_insertion_index(
    before: FrozenJsonArray,
    after: FrozenJsonArray,
    *,
    limits: TypedJsonLimits,
) -> int | None:
    """Find the lowest valid insertion index in linear comparisons."""

    if len(after.items) != len(before.items) + 1:
        return None
    before_items = _item_bytes(before, limits=limits)
    after_items = _item_bytes(after, limits=limits)
    size = len(before_items)
    suffix_ok = [False] * (size + 1)
    suffix_ok[size] = True
    for index in range(size - 1, -1, -1):
        suffix_ok[index] = (
            before_items[index] == after_items[index + 1]
            and suffix_ok[index + 1]
        )
    prefix_ok = True
    for index in range(size + 1):
        if prefix_ok and suffix_ok[index]:
            return index
        if index < size:
            prefix_ok = prefix_ok and before_items[index] == after_items[index]
    return None


def _single_deletion_index(
    before: FrozenJsonArray,
    after: FrozenJsonArray,
    *,
    limits: TypedJsonLimits,
) -> int | None:
    """Find the lowest valid deletion index in linear comparisons."""

    if len(before.items) != len(after.items) + 1:
        return None
    before_items = _item_bytes(before, limits=limits)
    after_items = _item_bytes(after, limits=limits)
    size = len(after_items)
    suffix_ok = [False] * (size + 1)
    suffix_ok[size] = True
    for index in range(size - 1, -1, -1):
        suffix_ok[index] = (
            before_items[index + 1] == after_items[index]
            and suffix_ok[index + 1]
        )
    prefix_ok = True
    for index in range(size + 1):
        if prefix_ok and suffix_ok[index]:
            return index
        if index < size:
            prefix_ok = prefix_ok and before_items[index] == after_items[index]
    return None


def derive_patch(
    base: object,
    target: object,
    *,
    base_candidate_id: CandidateId,
    target_candidate_id: CandidateId,
    component_tags: tuple[ComponentTagAssignment, ...] = (),
    limits: PatchLimits = DEFAULT_PATCH_LIMITS,
) -> TypedPatch:
    """Derive one deterministic bounded patch between exact typed trees.

    Object key-set changes are represented as an explicit subtree replacement;
    the algebra has no implicit or path-created object insertion.  A sequence
    gets a dedicated operation for a pure permutation or exactly one insertion
    or deletion.  More complex structural edits fall back to an explicit
    subtree replacement rather than an ambiguous order-dependent script.
    """

    if type(base_candidate_id) is not CandidateId or type(
        target_candidate_id
    ) is not CandidateId:
        raise TypeError("patch endpoint IDs must be exact CandidateId values")
    CandidateId.__post_init__(base_candidate_id)
    CandidateId.__post_init__(target_candidate_id)
    validate_patch_limits(limits)
    frozen_base = freeze_json(base, limits=limits.json_limits)
    frozen_target = freeze_json(target, limits=limits.json_limits)
    tag_by_path = _component_assignments(
        component_tags,
        max_assignments=limits.max_operations,
    )
    used_tag_paths: set[bytes] = set()
    operations: list[PatchOperation] = []

    def tag(path: JsonPath) -> str | None:
        path_key = canonical_path_bytes(path)
        value = tag_by_path.get(path_key)
        if value is not None:
            used_tag_paths.add(path_key)
        return value

    def add(operation: PatchOperation) -> None:
        operations.append(operation)
        if len(operations) > limits.max_operations:
            raise ValueError("derived patch exceeds max_operations")

    def replace_at(
        path: JsonPath,
        before: FrozenJsonValue,
        after: FrozenJsonValue,
    ) -> None:
        component = tag(path)
        if is_json_scalar(before) and is_json_scalar(after):
            add(
                ReplaceScalar(
                    path,
                    before,
                    after,
                    base_candidate_id,
                    component,
                )
            )
        else:
            add(
                ReplaceSubtree(
                    path,
                    before,
                    after,
                    base_candidate_id,
                    component,
                )
            )

    def visit(
        before: FrozenJsonValue,
        after: FrozenJsonValue,
        path: JsonPath,
    ) -> None:
        if typed_json_equal(before, after, limits=limits.json_limits):
            return
        if len(path.segments) >= limits.max_path_segments:
            replace_at(path, before, after)
            return

        if type(before) is FrozenJsonObject and type(after) is FrozenJsonObject:
            before_items = _object_dict(before)
            after_items = _object_dict(after)
            if before_items.keys() != after_items.keys():
                replace_at(path, before, after)
                return
            changed_children: list[
                tuple[str, FrozenJsonValue, FrozenJsonValue, JsonPath]
            ] = []
            for key, before_item in before.items:
                after_item = after_items[key]
                if typed_json_equal(
                    before_item,
                    after_item,
                    limits=limits.json_limits,
                ):
                    continue
                try:
                    child_path = path.child_key(key)
                except ValueError:
                    # The typed-JSON domain intentionally admits longer keys
                    # than one path segment.  Preserve derivation totality by
                    # bubbling the edit to the nearest representable parent.
                    replace_at(path, before, after)
                    return
                changed_children.append((key, before_item, after_item, child_path))
            for _, before_item, after_item, child_path in changed_children:
                visit(before_item, after_item, child_path)
            return

        if type(before) is FrozenJsonArray and type(after) is FrozenJsonArray:
            permutation = _stable_permutation(
                before,
                after,
                limits=limits.json_limits,
            )
            if permutation is not None and permutation != tuple(range(len(before.items))):
                add(
                    PermuteSequence(
                        path,
                        permutation,
                        before,
                        after,
                        base_candidate_id,
                        tag(path),
                    )
                )
                return

            if len(after.items) == len(before.items) + 1:
                index = _single_insertion_index(
                    before,
                    after,
                    limits=limits.json_limits,
                )
                if index is not None:
                    add(
                        InsertSequenceItem(
                            path,
                            index,
                            after.items[index],
                            before,
                            after,
                            base_candidate_id,
                            tag(path),
                        )
                    )
                    return
                replace_at(path, before, after)
                return

            if len(before.items) == len(after.items) + 1:
                index = _single_deletion_index(
                    before,
                    after,
                    limits=limits.json_limits,
                )
                if index is not None:
                    add(
                        DeleteSequenceItem(
                            path,
                            index,
                            before.items[index],
                            before,
                            after,
                            base_candidate_id,
                            tag(path),
                        )
                    )
                    return
                replace_at(path, before, after)
                return

            if len(before.items) == len(after.items):
                for index, (before_item, after_item) in enumerate(
                    zip(before.items, after.items)
                ):
                    visit(before_item, after_item, path.child_index(index))
                return

            replace_at(path, before, after)
            return

        replace_at(path, before, after)

    visit(frozen_base, frozen_target, JsonPath())
    unused_tags = set(tag_by_path) - used_tag_paths
    if unused_tags:
        raise ValueError("component_tags contains a path that is not a derived operation")
    ordered = tuple(sorted(operations, key=operation_sort_key))
    return TypedPatch(
        base_candidate_id=base_candidate_id,
        target_candidate_id=target_candidate_id,
        base_hash=typed_json_sha256(frozen_base, limits=limits.json_limits),
        target_hash=typed_json_sha256(frozen_target, limits=limits.json_limits),
        operations=ordered,
        limits=limits,
    )


def value_at_path(value: FrozenJsonValue, path: JsonPath) -> FrozenJsonValue:
    """Resolve an existing typed path without invoking container subclasses."""

    validate_json_path(path)
    current = freeze_json(value)
    if current is not value:
        raise TypeError("value_at_path requires an already frozen typed-JSON value")
    for segment in path.segments:
        if type(segment) is ObjectKey:
            if type(current) is not FrozenJsonObject:
                raise PatchPreconditionError("object-key segment reached a non-object")
            found = False
            for key, item in current.items:
                if key == segment.value:
                    current = item
                    found = True
                    break
            if not found:
                raise PatchPreconditionError("object-key path does not exist")
        elif type(segment) is ArrayIndex:
            if type(current) is not FrozenJsonArray:
                raise PatchPreconditionError("array-index segment reached a non-array")
            if segment.value >= len(current.items):
                raise PatchPreconditionError("array-index path is out of bounds")
            current = current.items[segment.value]
        else:  # pragma: no cover - JsonPath closes this union.
            raise AssertionError("unsupported path segment")
    return current


def _replace_existing_path(
    value: FrozenJsonValue,
    path: JsonPath,
    replacement: FrozenJsonValue,
) -> FrozenJsonValue:
    if not path.segments:
        return replacement
    head = path.segments[0]
    tail = JsonPath(path.segments[1:])
    if type(head) is ObjectKey:
        if type(value) is not FrozenJsonObject:
            raise PatchPreconditionError("object-key replacement reached a non-object")
        found = False
        updated: list[tuple[str, FrozenJsonValue]] = []
        for key, item in value.items:
            if key == head.value:
                found = True
                updated.append((key, _replace_existing_path(item, tail, replacement)))
            else:
                updated.append((key, item))
        if not found:
            raise PatchPreconditionError(
                "object-key replacement cannot insert a missing key"
            )
        return FrozenJsonObject(tuple(updated))
    if type(head) is ArrayIndex:
        if type(value) is not FrozenJsonArray:
            raise PatchPreconditionError("array-index replacement reached a non-array")
        if head.value >= len(value.items):
            raise PatchPreconditionError("array-index replacement is out of bounds")
        items = list(value.items)
        items[head.value] = _replace_existing_path(items[head.value], tail, replacement)
        return FrozenJsonArray(tuple(items))
    raise AssertionError("unsupported path segment")


def replace_existing_path(
    value: FrozenJsonValue,
    path: JsonPath,
    replacement: FrozenJsonValue,
) -> FrozenJsonValue:
    """Return an immutable tree with one existing typed path replaced.

    This is the narrow construction primitive used before deriving a canonical
    patch.  It cannot insert a key or grow an array, and it accepts only values
    that are already inside the frozen typed-JSON boundary.
    """

    validate_json_path(path)
    frozen_value = freeze_json(value)
    if frozen_value is not value:
        raise TypeError("value must already be frozen typed JSON")
    frozen_replacement = freeze_json(replacement)
    if frozen_replacement is not replacement:
        raise TypeError("replacement must already be frozen typed JSON")
    # Resolve first so every missing/stale non-root path fails before any
    # provisional target is constructed (the root is itself an existing path).
    value_at_path(value, path)
    return _replace_existing_path(value, path, replacement)


def apply_patch(base: object, patch: TypedPatch) -> FrozenJsonValue:
    """Apply a patch only when every local and global precondition is exact."""

    validate_typed_patch(patch)
    current = freeze_json(base, limits=patch.limits.json_limits)
    if typed_json_sha256(current, limits=patch.limits.json_limits) != patch.base_hash:
        raise PatchPreconditionError("base hash does not match patch.base_hash")

    for operation in patch.operations:
        observed = value_at_path(current, operation.path)
        if type(operation) in (ReplaceScalar, ReplaceSubtree):
            expected_before = operation.old_value
            replacement = operation.new_value
        else:
            expected_before = operation.before_sequence
            if type(observed) is not FrozenJsonArray:
                raise PatchPreconditionError("sequence operation reached a non-array")
            if type(operation) is InsertSequenceItem:
                replacement = FrozenJsonArray(
                    observed.items[: operation.index]
                    + (operation.item,)
                    + observed.items[operation.index :]
                )
            elif type(operation) is DeleteSequenceItem:
                if operation.index >= len(observed.items):
                    raise PatchPreconditionError("delete index is stale")
                if not typed_json_equal(
                    observed.items[operation.index],
                    operation.item,
                    limits=patch.limits.json_limits,
                ):
                    raise PatchPreconditionError("deleted item precondition is stale")
                replacement = FrozenJsonArray(
                    observed.items[: operation.index]
                    + observed.items[operation.index + 1 :]
                )
            elif type(operation) is PermuteSequence:
                if len(observed.items) != len(operation.permutation):
                    raise PatchPreconditionError("permutation length is stale")
                replacement = FrozenJsonArray(
                    tuple(observed.items[index] for index in operation.permutation)
                )
            else:  # pragma: no cover - TypedPatch closes operation types.
                raise AssertionError("unsupported patch operation")
        if not typed_json_equal(
            observed,
            expected_before,
            limits=patch.limits.json_limits,
        ):
            raise PatchPreconditionError("operation old-value precondition is stale")
        expected_after = (
            operation.new_value
            if type(operation) in (ReplaceScalar, ReplaceSubtree)
            else operation.after_sequence
        )
        if not typed_json_equal(
            replacement,
            expected_after,
            limits=patch.limits.json_limits,
        ):
            raise PatchPostconditionError("operation result violates its postcondition")
        current = _replace_existing_path(current, operation.path, replacement)
        if not typed_json_equal(
            value_at_path(current, operation.path),
            expected_after,
            limits=patch.limits.json_limits,
        ):
            raise PatchPostconditionError("operation path was not replaced exactly")

    actual_target_hash = typed_json_sha256(current, limits=patch.limits.json_limits)
    if actual_target_hash != patch.target_hash:
        raise PatchPostconditionError("patch result does not match target_hash")
    return current


def invert_patch(patch: TypedPatch) -> TypedPatch:
    """Build the canonical exact inverse; applying it restores the base tree."""

    validate_typed_patch(patch)
    inverse: list[PatchOperation] = []
    source = patch.target_candidate_id
    for operation in patch.operations:
        if type(operation) is ReplaceScalar:
            inverse.append(
                ReplaceScalar(
                    operation.path,
                    operation.new_value,
                    operation.old_value,
                    source,
                    operation.semantic_component,
                )
            )
        elif type(operation) is ReplaceSubtree:
            inverse.append(
                ReplaceSubtree(
                    operation.path,
                    operation.new_value,
                    operation.old_value,
                    source,
                    operation.semantic_component,
                )
            )
        elif type(operation) is InsertSequenceItem:
            inverse.append(
                DeleteSequenceItem(
                    operation.path,
                    operation.index,
                    operation.item,
                    operation.after_sequence,
                    operation.before_sequence,
                    source,
                    operation.semantic_component,
                )
            )
        elif type(operation) is DeleteSequenceItem:
            inverse.append(
                InsertSequenceItem(
                    operation.path,
                    operation.index,
                    operation.item,
                    operation.after_sequence,
                    operation.before_sequence,
                    source,
                    operation.semantic_component,
                )
            )
        elif type(operation) is PermuteSequence:
            inverse_permutation = _stable_permutation(
                operation.after_sequence,
                operation.before_sequence,
                limits=patch.limits.json_limits,
            )
            if inverse_permutation is None:  # pragma: no cover - constructor proves it.
                raise AssertionError("validated permutation had no inverse")
            inverse.append(
                PermuteSequence(
                    operation.path,
                    inverse_permutation,
                    operation.after_sequence,
                    operation.before_sequence,
                    source,
                    operation.semantic_component,
                )
            )
        else:  # pragma: no cover - TypedPatch closes operation types.
            raise AssertionError("unsupported patch operation")
    return TypedPatch(
        base_candidate_id=patch.target_candidate_id,
        target_candidate_id=patch.base_candidate_id,
        base_hash=patch.target_hash,
        target_hash=patch.base_hash,
        operations=tuple(sorted(inverse, key=operation_sort_key)),
        limits=patch.limits,
    )


class ThreeWayRelationKind(str, Enum):
    IDENTICAL = "identical"
    DISJOINT = "disjoint"
    COMPATIBLE_SAME_COMPONENT = "compatible_same_component"
    CONFLICT = "conflict"
    INVALIDATED = "invalidated"


@dataclass(frozen=True, slots=True)
class _CrossPathIndex:
    equal_pairs: tuple[tuple[int, int], ...]
    strict_edges: tuple[tuple[int, int], ...]
    work_units: int


def _validate_operation_tuple(
    operations: tuple[PatchOperation, ...],
    *,
    name: str,
) -> None:
    if type(operations) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if len(operations) > MAX_PATCH_OPERATIONS:
        raise ValueError(f"{name} exceeds MAX_PATCH_OPERATIONS")
    for operation in operations:
        validate_patch_operation(operation)
    if operations != tuple(sorted(operations, key=operation_sort_key)):
        raise ValueError(f"{name} must use canonical operation order")
    terminal = object()
    root: dict[object, object] = {}
    for operation in operations:
        node = root
        for segment in operation.path.segments:
            if terminal in node:
                raise ValueError(f"{name} contains overlapping operation paths")
            child = node.setdefault(segment, {})
            if type(child) is not dict:  # pragma: no cover - local trie invariant.
                raise AssertionError("invalid relation-side path trie")
            node = child
        if terminal in node or node:
            raise ValueError(f"{name} contains overlapping operation paths")
        node[terminal] = True


def _index_cross_path_relations(
    left: tuple[PatchOperation, ...],
    right: tuple[PatchOperation, ...],
) -> _CrossPathIndex:
    """Index equal and strict-prefix pairs in linear trie work plus output edges.

    Each side is already internally prefix-free.  Consequently the descendant
    subtrees visited for distinct right-side terminals are disjoint, so the
    accumulated ``work_units`` is linear in total path length and emitted
    strict-overlap edges rather than the Cartesian product of branch sizes.
    """

    _validate_operation_tuple(left, name="left_operations")
    _validate_operation_tuple(right, name="right_operations")
    terminal = object()
    root: dict[object, object] = {}
    work_units = 0
    for index, operation in enumerate(left):
        node = root
        for segment in operation.path.segments:
            work_units += 1
            child = node.setdefault(segment, {})
            if type(child) is not dict:  # pragma: no cover - local trie invariant.
                raise AssertionError("invalid cross-branch path trie")
            node = child
        node[terminal] = index

    equal_pairs: list[tuple[int, int]] = []
    strict_edges: list[tuple[int, int]] = []
    for right_index, operation in enumerate(right):
        node = root
        found = True
        for segment in operation.path.segments:
            work_units += 1
            left_ancestor = node.get(terminal)
            if type(left_ancestor) is int:
                strict_edges.append((left_ancestor, right_index))
            child = node.get(segment)
            if type(child) is not dict:
                found = False
                break
            node = child
        if not found:
            continue
        exact = node.get(terminal)
        if type(exact) is int:
            equal_pairs.append((exact, right_index))
            continue
        stack = [
            child
            for key, child in node.items()
            if key is not terminal and type(child) is dict
        ]
        while stack:
            descendant = stack.pop()
            work_units += 1
            left_descendant = descendant.get(terminal)
            if type(left_descendant) is int:
                strict_edges.append((left_descendant, right_index))
            stack.extend(
                child
                for key, child in descendant.items()
                if key is not terminal and type(child) is dict
            )
    return _CrossPathIndex(
        tuple(sorted(equal_pairs)),
        tuple(sorted(set(strict_edges))),
        work_units,
    )


@dataclass(frozen=True, slots=True, eq=False)
class PatchRelation:
    kind: ThreeWayRelationKind
    left_operations: tuple[PatchOperation, ...]
    right_operations: tuple[PatchOperation, ...]
    semantic_component: str | None = None

    def __post_init__(self) -> None:
        # Validate before any equality or encoding.  A string-like object with
        # hostile __eq__/encode methods must never influence canonical replay or
        # relation identity.
        validate_semantic_component(self.semantic_component)
        if type(self.kind) is not ThreeWayRelationKind:
            raise TypeError("kind must be a ThreeWayRelationKind")
        for name, operations in (
            ("left_operations", self.left_operations),
            ("right_operations", self.right_operations),
        ):
            _validate_operation_tuple(operations, name=name)
        left = self.left_operations
        right = self.right_operations
        if self.kind is ThreeWayRelationKind.IDENTICAL:
            if len(left) != 1 or len(right) != 1:
                raise ValueError("identical relations require one operation per branch")
            if operation_effect_bytes(left[0]) != operation_effect_bytes(right[0]):
                raise ValueError("identical relation operations are not exact effects")
        elif self.kind is ThreeWayRelationKind.CONFLICT:
            if len(left) != 1 or len(right) != 1 or left[0].path != right[0].path:
                raise ValueError("conflicts require different effects at one exact path")
            if operation_effect_bytes(left[0]) == operation_effect_bytes(right[0]):
                raise ValueError("identical effects cannot be classified as conflicts")
        elif self.kind is ThreeWayRelationKind.INVALIDATED:
            if not left or not right:
                raise ValueError("invalidated relations require operations on both branches")
            indexed = _index_cross_path_relations(left, right)
            if indexed.equal_pairs:
                raise ValueError("invalidated relation cannot contain equal paths")
            incident_left = {left_index for left_index, _ in indexed.strict_edges}
            incident_right = {right_index for _, right_index in indexed.strict_edges}
            if incident_left != set(range(len(left))) or incident_right != set(
                range(len(right))
            ):
                raise ValueError("invalidated relation lacks strict prefix invalidation")
            graph: dict[tuple[str, int], set[tuple[str, int]]] = {}
            for left_index, right_index in indexed.strict_edges:
                left_node = ("left", left_index)
                right_node = ("right", right_index)
                graph.setdefault(left_node, set()).add(right_node)
                graph.setdefault(right_node, set()).add(left_node)
            visited: set[tuple[str, int]] = set()
            stack = [next(iter(graph))]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                stack.extend(graph[node] - visited)
            if len(visited) != len(left) + len(right):
                raise ValueError("invalidated relation must be one connected component")
        elif self.kind is ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT:
            if not left or not right or self.semantic_component is None:
                raise ValueError(
                    "compatible same-component relations require both branches and a tag"
                )
            if any(
                operation.semantic_component != self.semantic_component
                for operation in left + right
            ):
                raise ValueError("compatible relation component tags do not agree")
            indexed = _index_cross_path_relations(left, right)
            if indexed.equal_pairs or indexed.strict_edges:
                raise ValueError("overlapping paths are not compatible disjoint edits")
        elif self.kind is ThreeWayRelationKind.DISJOINT:
            if len(left) + len(right) != 1:
                raise ValueError("a disjoint relation contains one unmatched branch edit")
        else:  # pragma: no cover - enum closes cases.
            raise AssertionError("unsupported relation kind")
        if self.kind is not ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT and (
            self.semantic_component is not None
        ):
            raise ValueError("semantic_component is only stored on compatible relations")

    def __eq__(self, other: object) -> bool:
        if type(self) is not PatchRelation or type(other) is not PatchRelation:
            return False
        return _relation_fingerprint(self) == _relation_fingerprint(other)

    __hash__ = None

    @property
    def relation_id(self) -> str:
        validate_patch_relation(self)
        digest = hashlib.sha256()
        digest.update(b"agent-evolve:three-way-relation:v1\x00")
        digest.update(self.kind.value.encode("ascii", errors="strict"))
        component = (
            b""
            if self.semantic_component is None
            else self.semantic_component.encode("utf-8", errors="strict")
        )
        digest.update(len(component).to_bytes(8, "big", signed=False))
        digest.update(component)
        for marker, operations in (
            (b"L", self.left_operations),
            (b"R", self.right_operations),
        ):
            digest.update(marker)
            digest.update(len(operations).to_bytes(8, "big", signed=False))
            for operation in operations:
                occurrence = operation_occurrence_bytes(operation)
                digest.update(len(occurrence).to_bytes(8, "big", signed=False))
                digest.update(occurrence)
        return digest.hexdigest()


def validate_patch_relation(relation: PatchRelation) -> None:
    """Revalidate an exact relation and every nested operation."""

    if type(relation) is not PatchRelation:
        raise TypeError("relation must be an exact PatchRelation")
    PatchRelation.__post_init__(relation)


def _relation_sort_key(relation: PatchRelation) -> tuple[bytes, str, str]:
    paths = [
        canonical_path_bytes(operation.path)
        for operation in relation.left_operations + relation.right_operations
    ]
    return min(paths), relation.kind.value, relation.relation_id


def _derive_three_way_relations(
    left_patch: TypedPatch,
    right_patch: TypedPatch,
) -> tuple[PatchRelation, ...]:
    """Derive the one canonical global relation partition from exact patches."""

    validate_typed_patch(left_patch)
    validate_typed_patch(right_patch)
    left = left_patch.operations
    right = right_patch.operations
    indexed = _index_cross_path_relations(left, right)
    relations: list[PatchRelation] = []
    consumed_left: set[int] = set()
    consumed_right: set[int] = set()

    for left_index, right_index in indexed.equal_pairs:
        left_operation = left[left_index]
        right_operation = right[right_index]
        kind = (
            ThreeWayRelationKind.IDENTICAL
            if operation_effect_bytes(left_operation)
            == operation_effect_bytes(right_operation)
            else ThreeWayRelationKind.CONFLICT
        )
        relations.append(PatchRelation(kind, (left_operation,), (right_operation,)))
        consumed_left.add(left_index)
        consumed_right.add(right_index)

    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for left_index, right_index in indexed.strict_edges:
        left_node = ("left", left_index)
        right_node = ("right", right_index)
        adjacency.setdefault(left_node, set()).add(right_node)
        adjacency.setdefault(right_node, set()).add(left_node)
    visited: set[tuple[str, int]] = set()
    for seed in sorted(adjacency):
        if seed in visited:
            continue
        stack = [seed]
        component_nodes: set[tuple[str, bytes]] = set()
        while stack:
            node = stack.pop()
            if node in component_nodes:
                continue
            component_nodes.add(node)
            stack.extend(adjacency.get(node, ()))
        visited.update(component_nodes)
        left_indices = sorted(index for side, index in component_nodes if side == "left")
        right_indices = sorted(index for side, index in component_nodes if side == "right")
        consumed_left.update(left_indices)
        consumed_right.update(right_indices)
        relations.append(
            PatchRelation(
                ThreeWayRelationKind.INVALIDATED,
                tuple(left[index] for index in left_indices),
                tuple(right[index] for index in right_indices),
            )
        )

    left_components: dict[str, list[int]] = {}
    right_components: dict[str, list[int]] = {}
    for index, operation in enumerate(left):
        if index not in consumed_left and operation.semantic_component is not None:
            left_components.setdefault(operation.semantic_component, []).append(index)
    for index, operation in enumerate(right):
        if index not in consumed_right and operation.semantic_component is not None:
            right_components.setdefault(operation.semantic_component, []).append(index)
    for component in sorted(set(left_components) & set(right_components)):
        left_group = sorted(left_components[component])
        right_group = sorted(right_components[component])
        relations.append(
            PatchRelation(
                ThreeWayRelationKind.COMPATIBLE_SAME_COMPONENT,
                tuple(left[index] for index in left_group),
                tuple(right[index] for index in right_group),
                component,
            )
        )
        consumed_left.update(left_group)
        consumed_right.update(right_group)

    for index, operation in enumerate(left):
        if index not in consumed_left:
            relations.append(
                PatchRelation(ThreeWayRelationKind.DISJOINT, (operation,), ())
            )
    for index, operation in enumerate(right):
        if index not in consumed_right:
            relations.append(
                PatchRelation(ThreeWayRelationKind.DISJOINT, (), (operation,))
            )
    return tuple(sorted(relations, key=_relation_sort_key))


def _relation_fingerprint(relation: PatchRelation) -> tuple[object, ...]:
    validate_patch_relation(relation)
    return (
        relation.kind.value,
        relation.semantic_component,
        tuple(operation_occurrence_bytes(value) for value in relation.left_operations),
        tuple(operation_occurrence_bytes(value) for value in relation.right_operations),
    )


def _assert_replayed_canonical_classification(
    ancestor: FrozenJsonValue,
    left_patch: TypedPatch,
    right_patch: TypedPatch,
    relations: tuple[PatchRelation, ...],
) -> None:
    validate_typed_patch(left_patch)
    validate_typed_patch(right_patch)
    if type(relations) is not tuple:
        raise TypeError("relations must be an exact tuple")
    for relation in relations:
        validate_patch_relation(relation)
    if left_patch.limits != right_patch.limits:
        raise ValueError("three-way branch patches must share exact algebra limits")
    if (
        left_patch.base_candidate_id != right_patch.base_candidate_id
        or left_patch.base_hash != right_patch.base_hash
    ):
        raise ValueError("three-way branch patches must share the exact ancestor")
    if left_patch.target_candidate_id == right_patch.target_candidate_id:
        raise ValueError("three-way branches must target distinct occurrences")
    apply_patch(ancestor, left_patch)
    apply_patch(ancestor, right_patch)
    expected = _derive_three_way_relations(left_patch, right_patch)
    if tuple(map(_relation_fingerprint, relations)) != tuple(
        map(_relation_fingerprint, expected)
    ):
        raise ValueError(
            "relations are not the canonical global classification of the replayed patches"
        )


@dataclass(frozen=True, slots=True, eq=False)
class ThreeWayPatchClassification:
    ancestor: FrozenJsonValue
    ancestor_candidate_id: CandidateId
    ancestor_hash: str
    left_patch_hash: str
    right_patch_hash: str
    relations: tuple[PatchRelation, ...]
    left_patch: TypedPatch
    right_patch: TypedPatch

    def __post_init__(self) -> None:
        if type(self.left_patch) is not TypedPatch or type(
            self.right_patch
        ) is not TypedPatch:
            raise TypeError("classification patches must be exact TypedPatch values")
        validate_typed_patch(self.left_patch)
        validate_typed_patch(self.right_patch)
        frozen_ancestor = freeze_json(
            self.ancestor,
            limits=self.left_patch.limits.json_limits,
        )
        object.__setattr__(self, "ancestor", frozen_ancestor)
        if type(self.ancestor_candidate_id) is not CandidateId:
            raise TypeError("ancestor_candidate_id must be an exact CandidateId")
        CandidateId.__post_init__(self.ancestor_candidate_id)
        for value, name in (
            (self.ancestor_hash, "ancestor_hash"),
            (self.left_patch_hash, "left_patch_hash"),
            (self.right_patch_hash, "right_patch_hash"),
        ):
            require_sha256(value, name)
        if (
            self.left_patch.base_candidate_id != self.ancestor_candidate_id
            or self.right_patch.base_candidate_id != self.ancestor_candidate_id
            or self.left_patch.base_hash != self.ancestor_hash
            or self.right_patch.base_hash != self.ancestor_hash
            or self.left_patch.patch_hash != self.left_patch_hash
            or self.right_patch.patch_hash != self.right_patch_hash
        ):
            raise ValueError("classification endpoint or patch hashes do not bind")
        if type(self.relations) is not tuple:
            raise TypeError("relations must be an exact tuple of PatchRelation values")
        relation_bound = len(self.left_patch.operations) + len(
            self.right_patch.operations
        )
        if len(self.relations) > relation_bound:
            raise ValueError("relations exceeds the branch-operation bound")
        if any(type(relation) is not PatchRelation for relation in self.relations):
            raise TypeError("relations must be an exact tuple of PatchRelation values")
        for relation in self.relations:
            validate_patch_relation(relation)
        if self.relations != tuple(sorted(self.relations, key=_relation_sort_key)):
            raise ValueError("relations must use canonical order")
        if len({relation.relation_id for relation in self.relations}) != len(
            self.relations
        ):
            raise ValueError("classification contains duplicate relation identities")

        expected: dict[tuple[str, bytes], int] = {}
        for side, patch in (("left", self.left_patch), ("right", self.right_patch)):
            for operation in patch.operations:
                key = (side, operation_occurrence_bytes(operation))
                expected[key] = expected.get(key, 0) + 1
        observed: dict[tuple[str, bytes], int] = {}
        for relation in self.relations:
            for side, operations in (
                ("left", relation.left_operations),
                ("right", relation.right_operations),
            ):
                for operation in operations:
                    if operation.source_candidate_id != self.ancestor_candidate_id:
                        raise ValueError(
                            "relation operation does not bind the common ancestor"
                        )
                    key = (side, operation_occurrence_bytes(operation))
                    observed[key] = observed.get(key, 0) + 1
        if observed != expected:
            raise ValueError("relations do not partition the two branch patches exactly")
        _assert_replayed_canonical_classification(
            self.ancestor,
            self.left_patch,
            self.right_patch,
            self.relations,
        )

    def revalidate(self) -> None:
        """Replay both branches and recompute the complete global partition."""

        if type(self) is not ThreeWayPatchClassification:
            raise TypeError("classification must be exact ThreeWayPatchClassification")
        ThreeWayPatchClassification.__post_init__(self)

    def __eq__(self, other: object) -> bool:
        if (
            type(self) is not ThreeWayPatchClassification
            or type(other) is not ThreeWayPatchClassification
        ):
            return False
        self.revalidate()
        other.revalidate()
        left = (
            canonical_typed_json_bytes(
                self.ancestor,
                limits=self.left_patch.limits.json_limits,
            ),
            self.ancestor_candidate_id.value,
            self.ancestor_hash,
            self.left_patch_hash,
            self.right_patch_hash,
            tuple(relation.relation_id for relation in self.relations),
        )
        right = (
            canonical_typed_json_bytes(
                other.ancestor,
                limits=other.left_patch.limits.json_limits,
            ),
            other.ancestor_candidate_id.value,
            other.ancestor_hash,
            other.left_patch_hash,
            other.right_patch_hash,
            tuple(relation.relation_id for relation in other.relations),
        )
        return left == right

    __hash__ = None

    def of_kind(self, kind: ThreeWayRelationKind) -> tuple[PatchRelation, ...]:
        if type(kind) is not ThreeWayRelationKind:
            raise TypeError("kind must be a ThreeWayRelationKind")
        self.revalidate()
        return tuple(relation for relation in self.relations if relation.kind is kind)


def classify_three_way_patches(
    ancestor: object,
    left_patch: TypedPatch,
    right_patch: TypedPatch,
) -> ThreeWayPatchClassification:
    """Verify and partition two ancestor-derived patches without merging."""

    validate_typed_patch(left_patch)
    validate_typed_patch(right_patch)
    if (
        left_patch.base_candidate_id != right_patch.base_candidate_id
        or left_patch.base_hash != right_patch.base_hash
    ):
        raise ValueError("three-way branch patches must share the exact ancestor")
    if left_patch.limits != right_patch.limits:
        raise ValueError("three-way branch patches must share exact algebra limits")
    if left_patch.target_candidate_id == right_patch.target_candidate_id:
        raise ValueError("three-way branches must target distinct occurrences")
    frozen_ancestor = freeze_json(ancestor, limits=left_patch.limits.json_limits)
    ordered = _derive_three_way_relations(left_patch, right_patch)
    return ThreeWayPatchClassification(
        ancestor=frozen_ancestor,
        ancestor_candidate_id=left_patch.base_candidate_id,
        ancestor_hash=left_patch.base_hash,
        left_patch_hash=left_patch.patch_hash,
        right_patch_hash=right_patch.patch_hash,
        relations=ordered,
        left_patch=left_patch,
        right_patch=right_patch,
    )


@dataclass(frozen=True, slots=True, eq=False)
class PreservationObligationRequest:
    """One requested path within a replay-verified three-way relation effect."""

    relation_id: str
    source: PreservationSource
    path: JsonPath

    def __post_init__(self) -> None:
        require_sha256(self.relation_id, "relation_id")
        if type(self.source) is not PreservationSource:
            raise TypeError("source must be a PreservationSource")
        validate_json_path(self.path)

    def __eq__(self, other: object) -> bool:
        if (
            type(self) is not PreservationObligationRequest
            or type(other) is not PreservationObligationRequest
        ):
            return False
        PreservationObligationRequest.__post_init__(self)
        PreservationObligationRequest.__post_init__(other)
        return (
            self.relation_id,
            self.source.value,
            canonical_path_bytes(self.path),
        ) == (
            other.relation_id,
            other.source.value,
            canonical_path_bytes(other.path),
        )

    __hash__ = None


_ABSENCE_CONTEXT_SHAPE_DOMAIN = b"agent-evolve:absence-context-shape:v1\x00"


def _context_shape_hash(value: FrozenJsonValue) -> str:
    digest = hashlib.sha256()
    digest.update(_ABSENCE_CONTEXT_SHAPE_DOMAIN)
    if type(value) is FrozenJsonObject:
        digest.update(b"object")
        digest.update(len(value.items).to_bytes(8, "big"))
        for key, _ in value.items:
            encoded = key.encode("utf-8", errors="strict")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    elif type(value) is FrozenJsonArray:
        digest.update(b"array")
        digest.update(len(value.items).to_bytes(8, "big"))
    else:
        raise TypeError("absence context must be an object or array")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _ObservedPathState:
    state: PreservationExpectation
    value: FrozenJsonValue | None
    value_hash: str | None
    absence_context_path: JsonPath | None = None
    absence_context_kind: AbsenceContextKind | None = None
    absence_context_shape_hash: str | None = None
    absence_failure_kind: AbsenceFailureKind | None = None


def _path_state(
    value: FrozenJsonValue,
    path: JsonPath,
    *,
    limits: TypedJsonLimits,
) -> _ObservedPathState:
    validate_json_path(path)
    frozen = freeze_json(value, limits=limits)
    if frozen is not value:
        raise TypeError("path-state inspection requires a frozen typed-JSON value")
    current = frozen
    traversed: list[ObjectKey | ArrayIndex] = []
    for segment in path.segments:
        context_path = JsonPath(tuple(traversed))
        if type(segment) is ObjectKey:
            if type(current) is not FrozenJsonObject:
                return _ObservedPathState(PreservationExpectation.ABSENT, None, None)
            found = False
            for key, item in current.items:
                if key == segment.value:
                    current = item
                    found = True
                    break
            if not found:
                return _ObservedPathState(
                    PreservationExpectation.ABSENT,
                    None,
                    None,
                    context_path,
                    AbsenceContextKind.OBJECT,
                    _context_shape_hash(current),
                    AbsenceFailureKind.MISSING_OBJECT_KEY,
                )
        elif type(segment) is ArrayIndex:
            if type(current) is not FrozenJsonArray:
                return _ObservedPathState(PreservationExpectation.ABSENT, None, None)
            if segment.value >= len(current.items):
                return _ObservedPathState(
                    PreservationExpectation.ABSENT,
                    None,
                    None,
                    context_path,
                    AbsenceContextKind.ARRAY,
                    _context_shape_hash(current),
                    AbsenceFailureKind.ARRAY_INDEX_OUT_OF_BOUNDS,
                )
            current = current.items[segment.value]
        else:  # pragma: no cover - validate_json_path closes the union.
            raise AssertionError("unsupported path segment")
        traversed.append(segment)
    return _ObservedPathState(
        PreservationExpectation.PRESENT,
        current,
        typed_json_sha256(current, limits=limits),
    )


def _state_identity(
    state: _ObservedPathState,
) -> tuple[
    PreservationExpectation,
    str | None,
    JsonPath | None,
    AbsenceContextKind | None,
    str | None,
    AbsenceFailureKind | None,
]:
    return (
        state.state,
        state.value_hash,
        state.absence_context_path,
        state.absence_context_kind,
        state.absence_context_shape_hash,
        state.absence_failure_kind,
    )


def _states_equal(
    left: _ObservedPathState,
    right: _ObservedPathState,
    *,
    limits: TypedJsonLimits,
) -> bool:
    if left.state is not right.state:
        return False
    if left.state is PreservationExpectation.ABSENT:
        return True
    if left.value is None or right.value is None:  # pragma: no cover - state invariant.
        raise AssertionError("present state lacked a typed value")
    return typed_json_equal(left.value, right.value, limits=limits)


def _operation_prefix_index(
    operations: tuple[PatchOperation, ...],
) -> dict[object, object]:
    terminal = object()
    root: dict[object, object] = {"__terminal_marker__": terminal}
    for operation in operations:
        node = root
        for segment in operation.path.segments:
            child = node.setdefault(segment, {})
            if type(child) is not dict:  # pragma: no cover - prefix-free invariant.
                raise AssertionError("invalid operation-prefix trie")
            node = child
        node[terminal] = operation
    return root


def _covering_effect(
    index: dict[object, object],
    path: JsonPath,
) -> PatchOperation:
    terminal = index["__terminal_marker__"]
    node = index
    match: PatchOperation | None = None
    for segment in path.segments:
        candidate = node.get(terminal)
        if type(candidate) in (
            ReplaceScalar,
            ReplaceSubtree,
            InsertSequenceItem,
            DeleteSequenceItem,
            PermuteSequence,
        ):
            match = candidate
            break
        child = node.get(segment)
        if type(child) is not dict:
            break
        node = child
    else:
        candidate = node.get(terminal)
        if type(candidate) in (
            ReplaceScalar,
            ReplaceSubtree,
            InsertSequenceItem,
            DeleteSequenceItem,
            PermuteSequence,
        ):
            match = candidate
    if match is None:
        raise ValueError(
            "obligation path must lie within exactly one source-branch operation effect"
        )
    return match


def derive_preservation_obligations(
    classification: ThreeWayPatchClassification,
    requests: tuple[PreservationObligationRequest, ...],
) -> tuple[PreservationObligation, ...]:
    """Derive immutable obligations only from replayed branch innovations.

    A branch-specific request must identify a value/presence change that differs
    from both the common ancestor and the other branch.  An identical edit may
    be requested only once as neutral evidence and never yields parent credit.
    """

    if type(classification) is not ThreeWayPatchClassification:
        raise TypeError("classification must be exact ThreeWayPatchClassification")
    classification.revalidate()
    if type(requests) is not tuple:
        raise TypeError(
            "requests must be an exact tuple of PreservationObligationRequest values"
        )
    if len(requests) > MAX_PRESERVATION_OBLIGATIONS:
        raise ValueError("requests exceeds MAX_PRESERVATION_OBLIGATIONS")
    if any(type(request) is not PreservationObligationRequest for request in requests):
        raise TypeError(
            "requests must be an exact tuple of PreservationObligationRequest values"
        )
    for request in requests:
        PreservationObligationRequest.__post_init__(request)
    request_keys = tuple(
        (
            request.relation_id,
            request.source.value,
            canonical_path_bytes(request.path),
        )
        for request in requests
    )
    if len(set(request_keys)) != len(request_keys):
        raise ValueError("preservation requests cannot duplicate an exact source/path")

    relations = {relation.relation_id: relation for relation in classification.relations}
    relation_indexes = {
        relation.relation_id: (
            _operation_prefix_index(relation.left_operations),
            _operation_prefix_index(relation.right_operations),
        )
        for relation in classification.relations
    }
    limits = classification.left_patch.limits.json_limits
    left_target = apply_patch(classification.ancestor, classification.left_patch)
    right_target = apply_patch(classification.ancestor, classification.right_patch)
    obligations: list[PreservationObligation] = []

    for request in requests:
        relation = relations.get(request.relation_id)
        if relation is None:
            raise ValueError("preservation request names an unknown relation identity")
        ancestor_state = _path_state(
            classification.ancestor,
            request.path,
            limits=limits,
        )

        if request.source is PreservationSource.IDENTICAL_NEUTRAL:
            if relation.kind is not ThreeWayRelationKind.IDENTICAL:
                raise ValueError("only an identical relation may create a neutral obligation")
            left_effect = _covering_effect(
                relation_indexes[relation.relation_id][0], request.path
            )
            right_effect = _covering_effect(
                relation_indexes[relation.relation_id][1], request.path
            )
            left_state = _path_state(left_target, request.path, limits=limits)
            right_state = _path_state(right_target, request.path, limits=limits)
            if not _states_equal(left_state, right_state, limits=limits):
                raise ValueError("identical relation branches disagree at obligation path")
            expected_state = left_state
            source_parent_ids = (
                classification.left_patch.target_candidate_id,
                classification.right_patch.target_candidate_id,
            )
            patch_hashes = (
                classification.left_patch_hash,
                classification.right_patch_hash,
            )
            effect_hashes = (
                operation_effect_sha256(left_effect, limits=limits),
                operation_effect_sha256(right_effect, limits=limits),
            )
        else:
            if relation.kind is ThreeWayRelationKind.IDENTICAL:
                raise ValueError("identical edits are neutral and cannot earn branch credit")
            if request.source is PreservationSource.LEFT_BRANCH:
                operation_index = relation_indexes[relation.relation_id][0]
                target = left_target
                other_target = right_target
                patch = classification.left_patch
            elif request.source is PreservationSource.RIGHT_BRANCH:
                operation_index = relation_indexes[relation.relation_id][1]
                target = right_target
                other_target = left_target
                patch = classification.right_patch
            else:  # pragma: no cover - enum closes sources.
                raise AssertionError("unsupported preservation source")
            effect = _covering_effect(operation_index, request.path)
            expected_state = _path_state(target, request.path, limits=limits)
            other_state = _path_state(other_target, request.path, limits=limits)
            if _states_equal(expected_state, ancestor_state, limits=limits):
                raise ValueError(
                    "branch-specific obligation is not an actual change from the ancestor"
                )
            if _states_equal(expected_state, other_state, limits=limits):
                raise ValueError(
                    "branch-specific obligation is not discriminative from the other branch"
                )
            source_parent_ids = (patch.target_candidate_id,)
            patch_hashes = (patch.patch_hash,)
            effect_hashes = (operation_effect_sha256(effect, limits=limits),)

        if (
            expected_state.state is PreservationExpectation.ABSENT
            and expected_state.absence_context_path is None
        ):
            raise ValueError(
                "absence obligations require a missing key/index in a surviving container"
            )

        obligations.append(
            PreservationObligation(
                source=request.source,
                source_parent_candidate_ids=source_parent_ids,
                branch_patch_hashes=patch_hashes,
                operation_effect_hashes=effect_hashes,
                relation_id=relation.relation_id,
                path=request.path,
                expected_state=expected_state.state,
                expected_value_hash=expected_state.value_hash,
                ancestor_state=ancestor_state.state,
                ancestor_value_hash=ancestor_state.value_hash,
                absence_context_path=expected_state.absence_context_path,
                absence_context_kind=expected_state.absence_context_kind,
                absence_context_shape_hash=expected_state.absence_context_shape_hash,
                absence_failure_kind=expected_state.absence_failure_kind,
            )
        )
    ordered = tuple(sorted(obligations, key=lambda value: value.obligation_id))
    if len({value.obligation_id for value in ordered}) != len(ordered):
        raise ValueError("derived preservation obligations have duplicate identities")
    return ordered


class ResolutionChoice(str, Enum):
    CHOOSE_LEFT = "choose_left"
    CHOOSE_RIGHT = "choose_right"
    SYNTHESIZE = "synthesize"
    DROP_BOTH = "drop_both"


@dataclass(frozen=True, slots=True, eq=False)
class PatchResolution:
    relation_id: str
    choice: ResolutionChoice
    synthesized_result_hash: str | None = None

    def __post_init__(self) -> None:
        require_sha256(self.relation_id, "relation_id")
        if type(self.choice) is not ResolutionChoice:
            raise TypeError("choice must be a ResolutionChoice")
        if self.choice is ResolutionChoice.SYNTHESIZE:
            if self.synthesized_result_hash is None:
                raise ValueError("synthesize resolution requires a result hash")
            require_sha256(
                self.synthesized_result_hash,
                "synthesized_result_hash",
            )
        elif self.synthesized_result_hash is not None:
            raise ValueError("only synthesize resolutions may carry a result hash")

    def __eq__(self, other: object) -> bool:
        if type(self) is not PatchResolution or type(other) is not PatchResolution:
            return False
        PatchResolution.__post_init__(self)
        PatchResolution.__post_init__(other)
        return (
            self.relation_id,
            self.choice.value,
            self.synthesized_result_hash,
        ) == (
            other.relation_id,
            other.choice.value,
            other.synthesized_result_hash,
        )

    __hash__ = None


def validate_three_way_resolutions(
    classification: ThreeWayPatchClassification,
    resolutions: tuple[PatchResolution, ...],
) -> tuple[PatchResolution, ...]:
    """Require exactly one declared choice for every conflict/invalidation."""

    if type(classification) is not ThreeWayPatchClassification:
        raise TypeError("classification must be exact ThreeWayPatchClassification")
    classification.revalidate()
    if type(resolutions) is not tuple:
        raise TypeError("resolutions must be an exact tuple of PatchResolution values")
    if len(resolutions) > len(classification.left_patch.operations) + len(
        classification.right_patch.operations
    ):
        raise ValueError("resolutions exceeds the branch-operation bound")
    if any(type(resolution) is not PatchResolution for resolution in resolutions):
        raise TypeError("resolutions must be an exact tuple of PatchResolution values")
    for resolution in resolutions:
        PatchResolution.__post_init__(resolution)
    if resolutions != tuple(sorted(resolutions, key=lambda value: value.relation_id)):
        raise ValueError("resolutions must use canonical relation-id order")
    ids = tuple(resolution.relation_id for resolution in resolutions)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate or conflicting resolutions are prohibited")
    required = {
        relation.relation_id
        for relation in classification.relations
        if relation.kind
        in (ThreeWayRelationKind.CONFLICT, ThreeWayRelationKind.INVALIDATED)
    }
    supplied = set(ids)
    if supplied != required:
        missing = required - supplied
        unknown = supplied - required
        reason = "missing" if missing else "unknown"
        raise ValueError(f"resolution set contains {reason} relation identities")
    return resolutions


@dataclass(frozen=True, slots=True, eq=False)
class ParentConfiguration:
    occurrence: CandidateOccurrence
    configuration: FrozenJsonValue

    def __post_init__(self) -> None:
        if type(self.occurrence) is not CandidateOccurrence:
            raise TypeError("occurrence must be an exact CandidateOccurrence")
        CandidateOccurrence.__post_init__(self.occurrence)
        frozen = freeze_json(self.configuration)
        if frozen is not self.configuration:
            raise TypeError("configuration must already be a frozen typed-JSON value")
        if typed_json_sha256(frozen) != self.occurrence.configuration_hash:
            raise ValueError("configuration does not match its occurrence hash")

    def __eq__(self, other: object) -> bool:
        if type(self) is not ParentConfiguration or type(other) is not ParentConfiguration:
            return False
        ParentConfiguration.__post_init__(self)
        ParentConfiguration.__post_init__(other)
        return (
            self.occurrence._validated_values(),
            canonical_typed_json_bytes(self.configuration),
        ) == (
            other.occurrence._validated_values(),
            canonical_typed_json_bytes(other.configuration),
        )

    __hash__ = None


def bind_parent_configuration(
    occurrence: CandidateOccurrence,
    configuration: object,
    *,
    limits: TypedJsonLimits,
) -> ParentConfiguration:
    if type(occurrence) is not CandidateOccurrence:
        raise TypeError("occurrence must be an exact CandidateOccurrence")
    CandidateOccurrence.__post_init__(occurrence)
    frozen = freeze_json(configuration, limits=limits)
    if typed_json_sha256(frozen, limits=limits) != occurrence.configuration_hash:
        raise ValueError("configuration does not match its occurrence hash")
    return ParentConfiguration(occurrence, frozen)


@dataclass(frozen=True, slots=True, eq=False)
class PreservationVerification:
    child_hash: str
    verified_claims: tuple[PreservationClaim, ...]
    discriminatively_used_parent_ids: tuple[CandidateId, ...]

    def __post_init__(self) -> None:
        require_sha256(self.child_hash, "child_hash")
        if type(self.verified_claims) is not tuple:
            raise TypeError("verified_claims must contain exact PreservationClaim values")
        if not 2 <= len(self.verified_claims) <= MAX_PRESERVATION_OBLIGATIONS:
            raise ValueError(
                "verified_claims must contain 2 to "
                f"{MAX_PRESERVATION_OBLIGATIONS} claims"
            )
        if any(type(claim) is not PreservationClaim for claim in self.verified_claims):
            raise TypeError("verified_claims must contain exact PreservationClaim values")
        for claim in self.verified_claims:
            PreservationClaim.__post_init__(claim)
        claim_ids = tuple(claim.obligation_id for claim in self.verified_claims)
        if claim_ids != tuple(sorted(claim_ids)):
            raise ValueError("verified_claims must use canonical obligation-id order")
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("verified_claims cannot duplicate an obligation identity")
        if type(self.discriminatively_used_parent_ids) is not tuple:
            raise TypeError("used parent IDs must be an exact CandidateId tuple")
        if len(self.discriminatively_used_parent_ids) != 2:
            raise ValueError(
                "three-way preservation requires exactly two used parent IDs"
            )
        if any(
            type(value) is not CandidateId
            for value in self.discriminatively_used_parent_ids
        ):
            raise TypeError("used parent IDs must be an exact CandidateId tuple")
        for value in self.discriminatively_used_parent_ids:
            CandidateId.__post_init__(value)
        if len(set(self.discriminatively_used_parent_ids)) != 2:
            raise ValueError("used parent IDs must be distinct")

    def __eq__(self, other: object) -> bool:
        if (
            type(self) is not PreservationVerification
            or type(other) is not PreservationVerification
        ):
            return False
        validate_preservation_verification(self)
        validate_preservation_verification(other)
        left = (
            self.child_hash,
            tuple(claim.obligation_id for claim in self.verified_claims),
            tuple(value.value for value in self.discriminatively_used_parent_ids),
        )
        right = (
            other.child_hash,
            tuple(claim.obligation_id for claim in other.verified_claims),
            tuple(value.value for value in other.discriminatively_used_parent_ids),
        )
        return left == right

    __hash__ = None


def validate_preservation_verification(value: PreservationVerification) -> None:
    """Recursively validate one non-authoritative preservation receipt.

    Structural validity does not prove that the receipt came from
    :func:`verify_preservation_claims`; a future durable evidence graph must
    bind it to the exact variation case, classification, child, and claims.
    """

    if type(value) is not PreservationVerification:
        raise TypeError("value must be an exact PreservationVerification")
    PreservationVerification.__post_init__(value)


def verify_preservation_claims(
    variation_case: VariationCase,
    classification: ThreeWayPatchClassification,
    parent_configurations: tuple[ParentConfiguration, ...],
    child: object,
    *,
    claims: tuple[PreservationClaim, ...],
    limits: TypedJsonLimits,
) -> PreservationVerification:
    """Replay and verify exact predeclared three-way innovation obligations."""

    validate_variation_case(variation_case)
    if variation_case.variation_kind is not VariationKind.THREE_WAY_RECOMBINATION:
        raise ValueError(
            "this preservation verifier accepts only verified three-way cases"
        )
    if type(classification) is not ThreeWayPatchClassification:
        raise TypeError("classification must be exact ThreeWayPatchClassification")
    classification.revalidate()
    if type(limits) is not TypedJsonLimits:
        raise TypeError("limits must be an exact TypedJsonLimits value")
    validate_typed_json_limits(limits)
    if limits != classification.left_patch.limits.json_limits:
        raise ValueError("verification limits must match the branch patch algebra")
    if variation_case.common_ancestor is None:
        raise ValueError("three-way preservation requires a common ancestor")
    if (
        classification.ancestor_candidate_id
        != variation_case.common_ancestor.candidate_id
        or classification.ancestor_hash
        != variation_case.common_ancestor.configuration_hash
        or tuple(
            patch.patch_hash for patch in variation_case.ancestor_to_parent_patches
        )
        != (classification.left_patch_hash, classification.right_patch_hash)
    ):
        raise ValueError("classification does not bind the exact variation case")
    if type(parent_configurations) is not tuple:
        raise TypeError(
            "parent_configurations must be an exact tuple of ParentConfiguration values"
        )
    if len(parent_configurations) != 2:
        raise ValueError("three-way preservation requires exactly two parent configurations")
    if any(type(value) is not ParentConfiguration for value in parent_configurations):
        raise TypeError(
            "parent_configurations must be an exact tuple of ParentConfiguration values"
        )
    for value in parent_configurations:
        ParentConfiguration.__post_init__(value)
    if type(claims) is not tuple:
        raise TypeError("claims must be an exact tuple of PreservationClaim values")
    if len(claims) > MAX_PRESERVATION_OBLIGATIONS:
        raise ValueError("claims exceeds MAX_PRESERVATION_OBLIGATIONS")
    if any(type(claim) is not PreservationClaim for claim in claims):
        raise TypeError("claims must be an exact tuple of PreservationClaim values")
    for claim in claims:
        PreservationClaim.__post_init__(claim)
    claim_ids = tuple(claim.obligation_id for claim in claims)
    if claim_ids != tuple(sorted(claim_ids)):
        raise ValueError("claims must use canonical obligation-id order")
    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("claims cannot duplicate an obligation identity")
    expected_ids = tuple(
        parent.occurrence.candidate_id for parent in variation_case.parents
    )
    actual_ids = tuple(value.occurrence.candidate_id for value in parent_configurations)
    if actual_ids != expected_ids:
        raise ValueError("parent configurations must follow exact variation-case order")
    for binding, value in zip(variation_case.parents, parent_configurations):
        if value.occurrence != binding.occurrence:
            raise ValueError("parent configuration occurrence does not match the case")
    replayed_targets = (
        apply_patch(classification.ancestor, classification.left_patch),
        apply_patch(classification.ancestor, classification.right_patch),
    )
    for supplied, replayed in zip(parent_configurations, replayed_targets):
        if not typed_json_equal(supplied.configuration, replayed, limits=limits):
            raise ValueError("parent configuration does not match its replayed branch patch")

    requests = tuple(
        PreservationObligationRequest(
            obligation.relation_id,
            obligation.source,
            obligation.path,
        )
        for obligation in variation_case.preservation_obligations
    )
    rederived_obligations = derive_preservation_obligations(classification, requests)
    if rederived_obligations != variation_case.preservation_obligations:
        raise PreservationError(
            "predeclared obligations are not exact replay-derived branch effects"
        )
    child_value = freeze_json(child, limits=limits)
    by_id = {
        value.occurrence.candidate_id: value.configuration
        for value in parent_configurations
    }
    required_ids = expected_ids
    obligations = {
        obligation.obligation_id: obligation
        for obligation in variation_case.preservation_obligations
    }
    if set(claim_ids) != set(obligations):
        raise PreservationError(
            "preservation claims must cover every predeclared obligation exactly once"
        )
    discriminative: set[CandidateId] = set()
    for claim in claims:
        obligation = obligations.get(claim.obligation_id)
        if obligation is None:  # pragma: no cover - exact set check above.
            raise PreservationError("claim does not name a predeclared obligation")
        child_state = _path_state(child_value, obligation.path, limits=limits)
        if _state_identity(child_state) != (
            obligation.expected_state,
            obligation.expected_value_hash,
            obligation.absence_context_path,
            obligation.absence_context_kind,
            obligation.absence_context_shape_hash,
            obligation.absence_failure_kind,
        ):
            raise PreservationError(
                "child does not preserve the obligation's exact presence-aware state"
            )
        if obligation.expected_state is PreservationExpectation.PRESENT:
            source_parent = by_id[obligation.source_parent_candidate_ids[0]]
            expected_value = value_at_path(source_parent, obligation.path)
            if child_state.value is None or not typed_json_equal(
                child_state.value,
                expected_value,
                limits=limits,
            ):
                raise PreservationError("child value does not match the replayed source branch")
        if obligation.source is not PreservationSource.IDENTICAL_NEUTRAL:
            discriminative.add(obligation.source_parent_candidate_ids[0])

    if set(required_ids) != discriminative:
        raise PreservationError(
            "one or more branch parents have no verified innovation-preservation claim"
        )
    return PreservationVerification(
        child_hash=typed_json_sha256(child_value, limits=limits),
        verified_claims=claims,
        discriminatively_used_parent_ids=tuple(
            candidate_id for candidate_id in required_ids if candidate_id in discriminative
        ),
    )


__all__ = [
    "ComponentTagAssignment",
    "ParentConfiguration",
    "PatchPostconditionError",
    "PatchPreconditionError",
    "PatchRelation",
    "PatchResolution",
    "PreservationObligationRequest",
    "PreservationError",
    "PreservationVerification",
    "ResolutionChoice",
    "ThreeWayPatchClassification",
    "ThreeWayRelationKind",
    "apply_patch",
    "bind_parent_configuration",
    "classify_three_way_patches",
    "derive_patch",
    "derive_preservation_obligations",
    "invert_patch",
    "replace_existing_path",
    "validate_three_way_resolutions",
    "validate_patch_relation",
    "validate_preservation_verification",
    "value_at_path",
    "verify_preservation_claims",
]
