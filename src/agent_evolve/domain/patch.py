"""Immutable value objects for typed, reversible candidate patches."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Union

from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.typed_json import (
    DEFAULT_TYPED_JSON_LIMITS,
    FrozenJsonArray,
    FrozenJsonObject,
    FrozenJsonValue,
    TypedJsonLimits,
    canonical_typed_json_bytes,
    freeze_json,
    is_frozen_json_value,
    is_json_scalar,
    typed_json_equal,
    typed_json_sha256,
    validate_typed_json_limits,
)


_LOWER_HEX = frozenset("0123456789abcdef")
_PATCH_HASH_DOMAIN = b"agent-evolve:typed-patch:v1\x00"
_PATH_HASH_DOMAIN = b"agent-evolve:typed-path:v1\x00"
_OPERATION_EFFECT_HASH_DOMAIN = b"agent-evolve:patch-operation-effect:v1\x00"
MAX_PATH_SEGMENTS = 64
MAX_PATH_KEY_BYTES = 4096
MAX_COMPONENT_TAG_BYTES = 256
MAX_PATCH_OPERATIONS = 4096
MAX_PATCH_BYTES = 67_108_864


def require_sha256(value: str, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_candidate_id(value: CandidateId, name: str) -> None:
    if type(value) is not CandidateId:
        raise TypeError(f"{name} must be an exact CandidateId")
    CandidateId.__post_init__(value)


def _strict_utf8(value: str, *, name: str, max_bytes: int) -> bytes:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    # UTF-8 never uses fewer bytes than Python code points.  Rejecting this
    # cheap lower bound first avoids encoding an attacker-sized exact string
    # solely to discover that it exceeds a small field cap.
    if len(value) > max_bytes:
        raise ValueError(f"{name} exceeds its byte limit")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} is not valid UTF-8 text") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{name} exceeds its byte limit")
    return encoded


def _frame(value: bytes) -> bytes:
    if type(value) is not bytes:
        raise TypeError("framed values must be exact bytes")
    return len(value).to_bytes(8, "big", signed=False) + value


@dataclass(frozen=True, slots=True, eq=False)
class ObjectKey:
    """An object-key path segment; never interchangeable with an array index."""

    value: str

    def __post_init__(self) -> None:
        _strict_utf8(self.value, name="object-key path segment", max_bytes=MAX_PATH_KEY_BYTES)

    def __eq__(self, other: object) -> bool:
        if type(self) is not ObjectKey or type(other) is not ObjectKey:
            return False
        ObjectKey.__post_init__(self)
        ObjectKey.__post_init__(other)
        return self.value == other.value

    def __hash__(self) -> int:
        if type(self) is not ObjectKey:
            raise TypeError("object-key path segments must be exact ObjectKey values")
        ObjectKey.__post_init__(self)
        return hash((ObjectKey, self.value))


@dataclass(frozen=True, slots=True, eq=False)
class ArrayIndex:
    """An array-index path segment; never interchangeable with a string key."""

    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int:
            raise TypeError("array-index path segments must be exact integers")
        if self.value < 0 or self.value > (1 << 63) - 1:
            raise ValueError("array-index path segment is outside uint63 range")

    def __eq__(self, other: object) -> bool:
        if type(self) is not ArrayIndex or type(other) is not ArrayIndex:
            return False
        ArrayIndex.__post_init__(self)
        ArrayIndex.__post_init__(other)
        return self.value == other.value

    def __hash__(self) -> int:
        if type(self) is not ArrayIndex:
            raise TypeError("array-index path segments must be exact ArrayIndex values")
        ArrayIndex.__post_init__(self)
        return hash((ArrayIndex, self.value))


PathSegment = Union[ObjectKey, ArrayIndex]


def canonical_path_bytes(path: "JsonPath") -> bytes:
    validate_json_path(path)
    chunks = [len(path.segments).to_bytes(8, "big", signed=False)]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            key = _strict_utf8(
                segment.value,
                name="object-key path segment",
                max_bytes=MAX_PATH_KEY_BYTES,
            )
            chunks.extend((b"k", _frame(key)))
        elif type(segment) is ArrayIndex:
            chunks.extend((b"i", segment.value.to_bytes(8, "big", signed=False)))
        else:  # pragma: no cover - JsonPath validation closes the union.
            raise AssertionError("unsupported path segment")
    return b"".join(chunks)


@dataclass(frozen=True, slots=True, eq=False)
class JsonPath:
    """A typed path through objects and arrays; the empty path denotes root."""

    segments: tuple[PathSegment, ...] = ()

    def __post_init__(self) -> None:
        if type(self.segments) is not tuple:
            raise TypeError("JsonPath.segments must be an exact tuple")
        if len(self.segments) > MAX_PATH_SEGMENTS:
            raise ValueError("path exceeds MAX_PATH_SEGMENTS")
        for segment in self.segments:
            if type(segment) is ObjectKey:
                ObjectKey.__post_init__(segment)
            elif type(segment) is ArrayIndex:
                ArrayIndex.__post_init__(segment)
            else:
                raise TypeError(
                    "paths contain only exact ObjectKey or ArrayIndex segments"
                )

    def __eq__(self, other: object) -> bool:
        if type(self) is not JsonPath or type(other) is not JsonPath:
            return False
        return canonical_path_bytes(self) == canonical_path_bytes(other)

    def __hash__(self) -> int:
        return hash((JsonPath, canonical_path_bytes(self)))

    def child_key(self, key: str) -> "JsonPath":
        validate_json_path(self)
        return JsonPath(self.segments + (ObjectKey(key),))

    def child_index(self, index: int) -> "JsonPath":
        validate_json_path(self)
        return JsonPath(self.segments + (ArrayIndex(index),))

    def is_prefix_of(self, other: "JsonPath") -> bool:
        validate_json_path(self)
        validate_json_path(other)
        if len(self.segments) > len(other.segments):
            return False
        return self.segments == other.segments[: len(self.segments)]

    @property
    def schema_identity(self) -> str:
        digest = hashlib.sha256()
        digest.update(_PATH_HASH_DOMAIN)
        digest.update(canonical_path_bytes(self))
        return digest.hexdigest()


def validate_json_path(path: JsonPath) -> None:
    """Recursively validate an exact immutable typed path."""

    if type(path) is not JsonPath:
        raise TypeError("path must be an exact JsonPath")
    JsonPath.__post_init__(path)
    for segment in path.segments:
        if type(segment) is ObjectKey:
            ObjectKey.__post_init__(segment)
        elif type(segment) is ArrayIndex:
            ArrayIndex.__post_init__(segment)
        else:  # pragma: no cover - JsonPath.__post_init__ rejects this first.
            raise TypeError("path contains an unsupported segment")


ROOT_PATH = JsonPath()


@dataclass(frozen=True, slots=True, eq=False)
class PatchLimits:
    """Replay-visible bounds for one typed patch."""

    json_limits: TypedJsonLimits = DEFAULT_TYPED_JSON_LIMITS
    max_operations: int = 1024
    max_path_segments: int = 32
    max_patch_bytes: int = 33_554_432

    def __post_init__(self) -> None:
        validate_typed_json_limits(self.json_limits)
        if type(self.max_operations) is not int:
            raise TypeError("max_operations must be an exact integer")
        if self.max_operations <= 0 or self.max_operations > MAX_PATCH_OPERATIONS:
            raise ValueError(f"max_operations must lie in [1, {MAX_PATCH_OPERATIONS}]")
        if type(self.max_path_segments) is not int:
            raise TypeError("max_path_segments must be an exact integer")
        if self.max_path_segments <= 0 or self.max_path_segments > MAX_PATH_SEGMENTS:
            raise ValueError(
                f"max_path_segments must lie in [1, {MAX_PATH_SEGMENTS}]"
            )
        if type(self.max_patch_bytes) is not int:
            raise TypeError("max_patch_bytes must be an exact integer")
        if self.max_patch_bytes <= 0 or self.max_patch_bytes > MAX_PATCH_BYTES:
            raise ValueError(f"max_patch_bytes must lie in [1, {MAX_PATCH_BYTES}]")

    def _validated_values(self) -> tuple[TypedJsonLimits, int, int, int]:
        if type(self) is not PatchLimits:
            raise TypeError("limits must be an exact PatchLimits value")
        PatchLimits.__post_init__(self)
        return (
            self.json_limits,
            self.max_operations,
            self.max_path_segments,
            self.max_patch_bytes,
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not PatchLimits or type(other) is not PatchLimits:
            return False
        return self._validated_values() == other._validated_values()

    def __hash__(self) -> int:
        return hash((PatchLimits, self._validated_values()))


DEFAULT_PATCH_LIMITS = PatchLimits()


def validate_patch_limits(limits: PatchLimits) -> None:
    """Revalidate an exact patch-limit graph, including nested JSON limits."""

    if type(limits) is not PatchLimits:
        raise TypeError("limits must be an exact PatchLimits value")
    PatchLimits.__post_init__(limits)


def validate_semantic_component(value: str | None) -> None:
    """Validate one exact bounded semantic-component tag.

    This is public because both patch operations and replay-derived relation
    values carry the same identity-bearing tag.  Validation must happen before
    equality, hashing, or encoding so string-like objects cannot execute code or
    supply a different digest preimage.
    """

    if value is None:
        return
    encoded = _strict_utf8(
        value,
        name="semantic_component",
        max_bytes=MAX_COMPONENT_TAG_BYTES,
    )
    if not encoded:
        raise ValueError("semantic_component cannot be empty")


def _validate_operation_common(
    path: JsonPath,
    source_candidate_id: CandidateId,
    semantic_component: str | None,
) -> None:
    validate_json_path(path)
    _validate_candidate_id(source_candidate_id, "source_candidate_id")
    validate_semantic_component(semantic_component)


def _canonical_permutation(
    before: FrozenJsonArray,
    after: FrozenJsonArray,
    *,
    limits: TypedJsonLimits = DEFAULT_TYPED_JSON_LIMITS,
) -> tuple[int, ...] | None:
    """Return the stable lowest-source-index matching permutation, if any."""

    if type(before) is not FrozenJsonArray or type(after) is not FrozenJsonArray:
        raise TypeError("permutation operands must be exact FrozenJsonArray values")
    freeze_json(before, limits=limits)
    freeze_json(after, limits=limits)
    if len(before.items) != len(after.items):
        return None
    positions: dict[bytes, list[int]] = {}
    for index, item in enumerate(before.items):
        key = canonical_typed_json_bytes(item, limits=limits)
        positions.setdefault(key, []).append(index)
    consumed: dict[bytes, int] = {}
    result: list[int] = []
    for item in after.items:
        key = canonical_typed_json_bytes(item, limits=limits)
        cursor = consumed.get(key, 0)
        candidates = positions.get(key)
        if candidates is None or cursor >= len(candidates):
            return None
        result.append(candidates[cursor])
        consumed[key] = cursor + 1
    if any(consumed.get(key, 0) != len(indices) for key, indices in positions.items()):
        return None
    return tuple(result)


class _CanonicalPatchOperation:
    """Type-sensitive comparison for immutable operation occurrences."""

    __hash__ = None

    def __eq__(self, other: object) -> bool:
        if type(self) not in _OPERATION_KINDS or type(other) is not type(self):
            return False
        return operation_occurrence_bytes(self) == operation_occurrence_bytes(other)


@dataclass(frozen=True, slots=True, eq=False)
class ReplaceScalar(_CanonicalPatchOperation):
    path: JsonPath
    old_value: FrozenJsonValue
    new_value: FrozenJsonValue
    source_candidate_id: CandidateId
    semantic_component: str | None = None

    def __post_init__(self) -> None:
        _validate_operation_common(
            self.path, self.source_candidate_id, self.semantic_component
        )
        if not is_json_scalar(self.old_value) or not is_json_scalar(self.new_value):
            raise TypeError("replace_scalar requires exact typed-JSON scalar values")
        if typed_json_equal(self.old_value, self.new_value):
            raise ValueError("replace_scalar cannot encode a no-op")

    @property
    def old_value_hash(self) -> str:
        return typed_json_sha256(self.old_value)

    @property
    def new_value_hash(self) -> str:
        return typed_json_sha256(self.new_value)

    @property
    def schema_field_identity(self) -> str:
        return self.path.schema_identity


@dataclass(frozen=True, slots=True, eq=False)
class ReplaceSubtree(_CanonicalPatchOperation):
    path: JsonPath
    old_value: FrozenJsonValue
    new_value: FrozenJsonValue
    source_candidate_id: CandidateId
    semantic_component: str | None = None

    def __post_init__(self) -> None:
        _validate_operation_common(
            self.path, self.source_candidate_id, self.semantic_component
        )
        if not is_frozen_json_value(self.old_value) or not is_frozen_json_value(
            self.new_value
        ):
            raise TypeError("replace_subtree requires frozen typed-JSON values")
        if is_json_scalar(self.old_value) and is_json_scalar(self.new_value):
            raise TypeError("scalar-to-scalar edits must use replace_scalar")
        if typed_json_equal(self.old_value, self.new_value):
            raise ValueError("replace_subtree cannot encode a no-op")

    @property
    def old_value_hash(self) -> str:
        return typed_json_sha256(self.old_value)

    @property
    def new_value_hash(self) -> str:
        return typed_json_sha256(self.new_value)

    @property
    def schema_field_identity(self) -> str:
        return self.path.schema_identity


@dataclass(frozen=True, slots=True, eq=False)
class InsertSequenceItem(_CanonicalPatchOperation):
    path: JsonPath
    index: int
    item: FrozenJsonValue
    before_sequence: FrozenJsonArray
    after_sequence: FrozenJsonArray
    source_candidate_id: CandidateId
    semantic_component: str | None = None

    def __post_init__(self) -> None:
        _validate_operation_common(
            self.path, self.source_candidate_id, self.semantic_component
        )
        if type(self.index) is not int:
            raise TypeError("insert index must be an exact integer")
        if type(self.before_sequence) is not FrozenJsonArray or type(
            self.after_sequence
        ) is not FrozenJsonArray:
            raise TypeError("insert_sequence_item requires frozen arrays")
        if not is_frozen_json_value(self.item):
            raise TypeError("inserted item must be a frozen typed-JSON value")
        freeze_json(self.before_sequence)
        freeze_json(self.after_sequence)
        freeze_json(self.item)
        if self.index < 0 or self.index > len(self.before_sequence.items):
            raise ValueError("insert index is outside the source array")
        expected = FrozenJsonArray(
            self.before_sequence.items[: self.index]
            + (self.item,)
            + self.before_sequence.items[self.index :]
        )
        if not typed_json_equal(expected, self.after_sequence):
            raise ValueError("after_sequence is not the declared insertion result")

    @property
    def old_value_hash(self) -> str:
        return typed_json_sha256(self.before_sequence)

    @property
    def new_value_hash(self) -> str:
        return typed_json_sha256(self.after_sequence)

    @property
    def schema_field_identity(self) -> str:
        return self.path.schema_identity


@dataclass(frozen=True, slots=True, eq=False)
class DeleteSequenceItem(_CanonicalPatchOperation):
    path: JsonPath
    index: int
    item: FrozenJsonValue
    before_sequence: FrozenJsonArray
    after_sequence: FrozenJsonArray
    source_candidate_id: CandidateId
    semantic_component: str | None = None

    def __post_init__(self) -> None:
        _validate_operation_common(
            self.path, self.source_candidate_id, self.semantic_component
        )
        if type(self.index) is not int:
            raise TypeError("delete index must be an exact integer")
        if type(self.before_sequence) is not FrozenJsonArray or type(
            self.after_sequence
        ) is not FrozenJsonArray:
            raise TypeError("delete_sequence_item requires frozen arrays")
        if not is_frozen_json_value(self.item):
            raise TypeError("deleted item must be a frozen typed-JSON value")
        freeze_json(self.before_sequence)
        freeze_json(self.after_sequence)
        freeze_json(self.item)
        if self.index < 0 or self.index >= len(self.before_sequence.items):
            raise ValueError("delete index is outside the source array")
        if not typed_json_equal(self.before_sequence.items[self.index], self.item):
            raise ValueError("deleted item does not match the source array")
        expected = FrozenJsonArray(
            self.before_sequence.items[: self.index]
            + self.before_sequence.items[self.index + 1 :]
        )
        if not typed_json_equal(expected, self.after_sequence):
            raise ValueError("after_sequence is not the declared deletion result")

    @property
    def old_value_hash(self) -> str:
        return typed_json_sha256(self.before_sequence)

    @property
    def new_value_hash(self) -> str:
        return typed_json_sha256(self.after_sequence)

    @property
    def schema_field_identity(self) -> str:
        return self.path.schema_identity


@dataclass(frozen=True, slots=True, eq=False)
class PermuteSequence(_CanonicalPatchOperation):
    path: JsonPath
    permutation: tuple[int, ...]
    before_sequence: FrozenJsonArray
    after_sequence: FrozenJsonArray
    source_candidate_id: CandidateId
    semantic_component: str | None = None

    def __post_init__(self) -> None:
        _validate_operation_common(
            self.path, self.source_candidate_id, self.semantic_component
        )
        if type(self.before_sequence) is not FrozenJsonArray or type(
            self.after_sequence
        ) is not FrozenJsonArray:
            raise TypeError("permute_sequence requires frozen arrays")
        freeze_json(self.before_sequence)
        freeze_json(self.after_sequence)
        size = len(self.before_sequence.items)
        if type(self.permutation) is not tuple:
            raise TypeError("permutation must be an exact tuple of exact integers")
        if len(self.permutation) != size:
            raise ValueError("permutation must contain every source index exactly once")
        if any(type(index) is not int for index in self.permutation):
            raise TypeError("permutation must be an exact tuple of exact integers")
        if set(self.permutation) != set(range(size)):
            raise ValueError("permutation must contain every source index exactly once")
        canonical = _canonical_permutation(
            self.before_sequence,
            self.after_sequence,
        )
        if canonical is None:
            raise ValueError("before_sequence and after_sequence are not permutations")
        if self.permutation != canonical:
            raise ValueError(
                "permutation is not the deterministic lowest-source-index mapping"
            )
        if self.permutation == tuple(range(size)):
            raise ValueError("permute_sequence cannot encode an identity permutation")

    @property
    def old_value_hash(self) -> str:
        return typed_json_sha256(self.before_sequence)

    @property
    def new_value_hash(self) -> str:
        return typed_json_sha256(self.after_sequence)

    @property
    def schema_field_identity(self) -> str:
        return self.path.schema_identity


PatchOperation = Union[
    ReplaceScalar,
    ReplaceSubtree,
    InsertSequenceItem,
    DeleteSequenceItem,
    PermuteSequence,
]

_OPERATION_KINDS = {
    ReplaceScalar: b"replace_scalar",
    ReplaceSubtree: b"replace_subtree",
    InsertSequenceItem: b"insert_sequence_item",
    DeleteSequenceItem: b"delete_sequence_item",
    PermuteSequence: b"permute_sequence",
}


def validate_patch_operation(operation: PatchOperation) -> None:
    """Re-run the complete invariant set for one exact operation value."""

    operation_type = type(operation)
    if operation_type not in _OPERATION_KINDS:
        raise TypeError("unsupported patch operation type")
    operation_type.__post_init__(operation)


def operation_kind(operation: PatchOperation) -> str:
    validate_patch_operation(operation)
    kind = _OPERATION_KINDS.get(type(operation))
    if kind is None:  # pragma: no cover - validator closes the mapping.
        raise AssertionError("validated operation lacked a kind")
    return kind.decode("ascii")


def _operation_sort_key_unchecked(operation: PatchOperation) -> tuple[bytes, bytes]:
    return canonical_path_bytes(operation.path), _OPERATION_KINDS[type(operation)]


def operation_sort_key(operation: PatchOperation) -> tuple[bytes, bytes]:
    validate_patch_operation(operation)
    return _operation_sort_key_unchecked(operation)


def _operation_values(operation: PatchOperation) -> tuple[FrozenJsonValue, FrozenJsonValue]:
    if type(operation) in (ReplaceScalar, ReplaceSubtree):
        return operation.old_value, operation.new_value
    if type(operation) in (InsertSequenceItem, DeleteSequenceItem, PermuteSequence):
        return operation.before_sequence, operation.after_sequence
    raise TypeError("unsupported patch operation type")


def operation_effect_bytes(
    operation: PatchOperation,
    *,
    include_component: bool = True,
    limits: TypedJsonLimits = DEFAULT_TYPED_JSON_LIMITS,
) -> bytes:
    """Canonical operation effect, excluding branch-specific occurrence IDs."""

    if type(include_component) is not bool:
        raise TypeError("include_component must be an exact Boolean")
    validate_patch_operation(operation)
    validate_typed_json_limits(limits)
    old_value, new_value = _operation_values(operation)
    component = (
        b""
        if not include_component or operation.semantic_component is None
        else _strict_utf8(
            operation.semantic_component,
            name="semantic_component",
            max_bytes=MAX_COMPONENT_TAG_BYTES,
        )
    )
    parts = [
        _frame(_OPERATION_KINDS[type(operation)]),
        _frame(canonical_path_bytes(operation.path)),
        _frame(component),
        _frame(canonical_typed_json_bytes(old_value, limits=limits)),
        _frame(canonical_typed_json_bytes(new_value, limits=limits)),
    ]
    if type(operation) in (InsertSequenceItem, DeleteSequenceItem):
        parts.append(operation.index.to_bytes(8, "big", signed=False))
        parts.append(
            _frame(canonical_typed_json_bytes(operation.item, limits=limits))
        )
    elif type(operation) is PermuteSequence:
        parts.append(len(operation.permutation).to_bytes(8, "big", signed=False))
        parts.extend(
            index.to_bytes(8, "big", signed=False)
            for index in operation.permutation
        )
    return b"".join(parts)


def operation_occurrence_bytes(
    operation: PatchOperation,
    *,
    limits: TypedJsonLimits = DEFAULT_TYPED_JSON_LIMITS,
) -> bytes:
    """Canonical operation occurrence, including its exact source identity."""

    validate_patch_operation(operation)
    source = operation.source_candidate_id.value.encode("ascii", errors="strict")
    return _frame(source) + _frame(operation_effect_bytes(operation, limits=limits))


def operation_effect_sha256(
    operation: PatchOperation,
    *,
    limits: TypedJsonLimits = DEFAULT_TYPED_JSON_LIMITS,
) -> str:
    """Hash one exact operation effect independently of its branch occurrence."""

    digest = hashlib.sha256()
    digest.update(_OPERATION_EFFECT_HASH_DOMAIN)
    digest.update(operation_effect_bytes(operation, limits=limits))
    return digest.hexdigest()


def _validate_non_overlapping_operation_paths(
    operations: tuple[PatchOperation, ...],
    *,
    work_counter: list[int] | None = None,
) -> None:
    """Reject equal/prefix paths in linear total path length using a trie."""

    terminal = object()
    root: dict[object, object] = {}
    for operation in operations:
        node = root
        for segment in operation.path.segments:
            if work_counter is not None:
                work_counter[0] += 1
            if terminal in node:
                raise ValueError("patch operation paths cannot overlap")
            child = node.setdefault(segment, {})
            if type(child) is not dict:  # pragma: no cover - local trie invariant.
                raise AssertionError("invalid patch-path trie")
            node = child
        if terminal in node or node:
            raise ValueError("patch operation paths cannot overlap")
        node[terminal] = True


def _validate_operation_limits(operation: PatchOperation, limits: PatchLimits) -> None:
    validate_patch_operation(operation)
    validate_patch_limits(limits)
    if len(operation.path.segments) > limits.max_path_segments:
        raise ValueError("operation path exceeds patch max_path_segments")
    old_value, new_value = _operation_values(operation)
    canonical_typed_json_bytes(old_value, limits=limits.json_limits)
    canonical_typed_json_bytes(new_value, limits=limits.json_limits)


@dataclass(frozen=True, slots=True, eq=False)
class TypedPatch:
    """One canonical, bounded transformation between exact candidate contents."""

    base_candidate_id: CandidateId
    target_candidate_id: CandidateId
    base_hash: str
    target_hash: str
    operations: tuple[PatchOperation, ...]
    limits: PatchLimits = DEFAULT_PATCH_LIMITS
    schema_version: str = "typed_json_patch_v1"

    def __post_init__(self) -> None:
        _validate_candidate_id(self.base_candidate_id, "base_candidate_id")
        _validate_candidate_id(self.target_candidate_id, "target_candidate_id")
        if self.base_candidate_id == self.target_candidate_id:
            raise ValueError("patch endpoints must be distinct candidate occurrences")
        require_sha256(self.base_hash, "base_hash")
        require_sha256(self.target_hash, "target_hash")
        if type(self.operations) is not tuple:
            raise TypeError("patch operations must be an exact tuple")
        validate_patch_limits(self.limits)
        if type(self.schema_version) is not str or self.schema_version != "typed_json_patch_v1":
            raise ValueError("unsupported typed-patch schema_version")
        if len(self.operations) > self.limits.max_operations:
            raise ValueError("patch exceeds max_operations")
        for operation in self.operations:
            validate_patch_operation(operation)
        canonical = tuple(sorted(self.operations, key=_operation_sort_key_unchecked))
        if self.operations != canonical:
            raise ValueError("patch operations must use canonical path/kind order")
        for operation in self.operations:
            if operation.source_candidate_id != self.base_candidate_id:
                raise ValueError("every operation must bind the patch base occurrence")
            _validate_operation_limits(operation, self.limits)
        # Bound the exact framed preimage consumed by ``patch_hash`` (apart
        # from the fixed domain tag, which is process-constant).
        total_patch_bytes = (
            8
            + len(self.schema_version.encode("ascii", errors="strict"))
            + 8
            + len(self.base_candidate_id.value.encode("ascii", errors="strict"))
            + 8
            + len(self.target_candidate_id.value.encode("ascii", errors="strict"))
            + 64  # two raw SHA-256 endpoint digests
            + 9 * 8  # replay limits
            + 8  # operation count
        )
        if total_patch_bytes > self.limits.max_patch_bytes:
            raise ValueError("patch exceeds max_patch_bytes")
        for operation in self.operations:
            total_patch_bytes += 8 + len(
                operation_effect_bytes(operation, limits=self.limits.json_limits)
            )
            if total_patch_bytes > self.limits.max_patch_bytes:
                raise ValueError("patch exceeds max_patch_bytes")
        _validate_non_overlapping_operation_paths(self.operations)
        if not self.operations and self.base_hash != self.target_hash:
            raise ValueError("an empty patch requires identical base and target hashes")
        if self.operations and self.base_hash == self.target_hash:
            raise ValueError("a non-empty patch cannot claim identical endpoint hashes")

    def __eq__(self, other: object) -> bool:
        if type(self) is not TypedPatch or type(other) is not TypedPatch:
            return False
        return self.patch_hash == other.patch_hash

    __hash__ = None

    @property
    def patch_hash(self) -> str:
        validate_typed_patch(self)
        digest = hashlib.sha256()
        digest.update(_PATCH_HASH_DOMAIN)
        digest.update(_frame(self.schema_version.encode("ascii", errors="strict")))
        digest.update(_frame(self.base_candidate_id.value.encode("ascii", errors="strict")))
        digest.update(_frame(self.target_candidate_id.value.encode("ascii", errors="strict")))
        digest.update(bytes.fromhex(self.base_hash))
        digest.update(bytes.fromhex(self.target_hash))
        limit_values = (
            self.limits.max_operations,
            self.limits.max_path_segments,
            self.limits.max_patch_bytes,
            self.limits.json_limits.max_depth,
            self.limits.json_limits.max_nodes,
            self.limits.json_limits.max_container_items,
            self.limits.json_limits.max_string_bytes,
            self.limits.json_limits.max_integer_bits,
            self.limits.json_limits.max_canonical_bytes,
        )
        for value in limit_values:
            digest.update(value.to_bytes(8, "big", signed=False))
        digest.update(len(self.operations).to_bytes(8, "big", signed=False))
        for operation in self.operations:
            digest.update(
                _frame(
                    operation_effect_bytes(
                        operation,
                        limits=self.limits.json_limits,
                    )
                )
            )
        return digest.hexdigest()


def validate_typed_patch(patch: TypedPatch) -> None:
    """Revalidate a complete exact patch graph at a consuming boundary."""

    if type(patch) is not TypedPatch:
        raise TypeError("patch must be an exact TypedPatch")
    TypedPatch.__post_init__(patch)


__all__ = [
    "ArrayIndex",
    "DEFAULT_PATCH_LIMITS",
    "DeleteSequenceItem",
    "InsertSequenceItem",
    "JsonPath",
    "ObjectKey",
    "PatchLimits",
    "PatchOperation",
    "PermuteSequence",
    "ROOT_PATH",
    "ReplaceScalar",
    "ReplaceSubtree",
    "TypedPatch",
    "canonical_path_bytes",
    "operation_effect_bytes",
    "operation_effect_sha256",
    "operation_occurrence_bytes",
    "operation_kind",
    "operation_sort_key",
    "require_sha256",
    "validate_json_path",
    "validate_patch_limits",
    "validate_patch_operation",
    "validate_semantic_component",
    "validate_typed_patch",
]
