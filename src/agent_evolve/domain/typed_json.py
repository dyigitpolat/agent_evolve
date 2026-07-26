"""Immutable, type-sensitive JSON values and canonical hashing.

The standard-library JSON codec intentionally treats several Python values as
interchangeable (notably ``True`` and ``1``) and offers no immutable container
representation.  Evolutionary lineage needs stricter evidence: hashes and
patch preconditions must identify the exact typed tree that was observed.

This module is an inward domain codec.  It accepts only exact built-in JSON
types, freezes objects/arrays, rejects cycles and non-finite numbers, and emits
a small length-framed binary canonical form.  It is not an interchange JSON
serializer and it has no filesystem or framework dependency.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Union


_TYPED_JSON_HASH_DOMAIN = b"agent-evolve:typed-json:v1\x00"
_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True, slots=True, eq=False)
class TypedJsonLimits:
    """Hard resource bounds for validation and canonicalization.

    User-provided limits may only tighten these process-independent ceilings.
    This keeps a serialized patch from claiming effectively unbounded limits.
    """

    max_depth: int = 64
    max_nodes: int = 50_000
    max_container_items: int = 10_000
    max_string_bytes: int = 1_048_576
    max_integer_bits: int = 4096
    max_canonical_bytes: int = 8_388_608

    def __post_init__(self) -> None:
        ceilings = {
            "max_depth": 64,
            "max_nodes": 50_000,
            "max_container_items": 10_000,
            "max_string_bytes": 1_048_576,
            "max_integer_bits": 4096,
            "max_canonical_bytes": 8_388_608,
        }
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if value <= 0 or value > ceiling:
                raise ValueError(f"{name} must lie in [1, {ceiling}]")

    def _validated_values(self) -> tuple[int, int, int, int, int, int]:
        if type(self) is not TypedJsonLimits:
            raise TypeError("limits must be an exact TypedJsonLimits value")
        TypedJsonLimits.__post_init__(self)
        return (
            self.max_depth,
            self.max_nodes,
            self.max_container_items,
            self.max_string_bytes,
            self.max_integer_bits,
            self.max_canonical_bytes,
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not TypedJsonLimits or type(other) is not TypedJsonLimits:
            return False
        return self._validated_values() == other._validated_values()

    def __hash__(self) -> int:
        return hash((TypedJsonLimits, self._validated_values()))


DEFAULT_TYPED_JSON_LIMITS = TypedJsonLimits()


def validate_typed_json_limits(limits: TypedJsonLimits) -> None:
    """Revalidate an exact limits value at every public trust boundary.

    Frozen dataclasses prevent ordinary assignment, but callers may still hand
    a domain service an instance that did not pass its generated constructor
    (for example, a value reconstructed by a custom codec).  Consumers must
    therefore validate the complete value rather than relying on construction
    having happened elsewhere.
    """

    if type(limits) is not TypedJsonLimits:
        raise TypeError("limits must be an exact TypedJsonLimits value")
    TypedJsonLimits.__post_init__(limits)


def _utf8_bytes(value: str, *, limits: TypedJsonLimits, name: str) -> bytes:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if len(value) > limits.max_string_bytes:
        raise ValueError(f"{name} exceeds max_string_bytes")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} is not valid UTF-8 text") from exc
    if len(encoded) > limits.max_string_bytes:
        raise ValueError(f"{name} exceeds max_string_bytes")
    return encoded


def _is_scalar(value: object) -> bool:
    return value is None or type(value) in (bool, int, float, str)


def _validate_scalar(value: object, *, limits: TypedJsonLimits) -> None:
    value_type = type(value)
    if value is None or value_type is bool:
        return
    if value_type is int:
        if value.bit_length() > limits.max_integer_bits:
            raise ValueError("integer exceeds max_integer_bits")
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("floating-point values must be finite")
        return
    if value_type is str:
        _utf8_bytes(value, limits=limits, name="string value")
        return
    raise TypeError("value is not an exact typed-JSON scalar")


@dataclass(frozen=True, slots=True, eq=False)
class FrozenJsonArray:
    """An immutable typed-JSON array."""

    items: tuple["FrozenJsonValue", ...]

    def __post_init__(self) -> None:
        if type(self.items) is not tuple:
            raise TypeError("FrozenJsonArray.items must be an exact tuple")
        _validate_frozen(self, limits=DEFAULT_TYPED_JSON_LIMITS)

    def __eq__(self, other: object) -> bool:
        if type(self) is not FrozenJsonArray or type(other) is not FrozenJsonArray:
            return False
        return typed_json_equal(self, other)

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class FrozenJsonObject:
    """An immutable typed-JSON object in canonical UTF-8 key order."""

    items: tuple[tuple[str, "FrozenJsonValue"], ...]

    def __post_init__(self) -> None:
        if type(self.items) is not tuple:
            raise TypeError("FrozenJsonObject.items must be an exact tuple")
        _validate_frozen(self, limits=DEFAULT_TYPED_JSON_LIMITS)

    def __eq__(self, other: object) -> bool:
        if type(self) is not FrozenJsonObject or type(other) is not FrozenJsonObject:
            return False
        return typed_json_equal(self, other)

    __hash__ = None


FrozenJsonValue = Union[
    None,
    bool,
    int,
    float,
    str,
    FrozenJsonArray,
    FrozenJsonObject,
]


def is_frozen_json_value(value: object) -> bool:
    """Return whether *value* has an exact frozen typed-JSON runtime type."""

    return _is_scalar(value) or type(value) in (FrozenJsonArray, FrozenJsonObject)


@dataclass(slots=True)
class _ValidationState:
    nodes: int = 0
    text_bytes: int = 0
    active_container_ids: set[int] = field(default_factory=set)


def _count_text_bytes(
    state: _ValidationState | "_FreezeState",
    encoded: bytes,
    *,
    limits: TypedJsonLimits,
) -> None:
    state.text_bytes += len(encoded)
    if state.text_bytes > limits.max_canonical_bytes:
        raise ValueError("typed-JSON text exceeds max_canonical_bytes")


def _validate_frozen(
    value: FrozenJsonValue,
    *,
    limits: TypedJsonLimits,
    depth: int = 0,
    state: _ValidationState | None = None,
) -> None:
    if state is None:
        # Limits are immutable and shared through the complete traversal.  A
        # public boundary validates them once; recursive nodes must not repeat
        # the same six-field validation thousands of times.
        validate_typed_json_limits(limits)
        state = _ValidationState()
    if depth > limits.max_depth:
        raise ValueError("typed-JSON value exceeds max_depth")
    state.nodes += 1
    if state.nodes > limits.max_nodes:
        raise ValueError("typed-JSON value exceeds max_nodes")

    if _is_scalar(value):
        _validate_scalar(value, limits=limits)
        if type(value) is str:
            _count_text_bytes(
                state,
                _utf8_bytes(value, limits=limits, name="string value"),
                limits=limits,
            )
        return

    if type(value) is FrozenJsonArray:
        if type(value.items) is not tuple:
            raise TypeError("FrozenJsonArray.items must be an exact tuple")
        if len(value.items) > limits.max_container_items:
            raise ValueError("array exceeds max_container_items")
        identity = id(value)
        if identity in state.active_container_ids:
            raise ValueError("frozen typed-JSON values cannot contain cycles")
        state.active_container_ids.add(identity)
        try:
            for item in value.items:
                if not is_frozen_json_value(item):
                    raise TypeError(
                        "frozen arrays may contain only frozen typed-JSON values"
                    )
                _validate_frozen(
                    item,
                    limits=limits,
                    depth=depth + 1,
                    state=state,
                )
        finally:
            state.active_container_ids.remove(identity)
        return

    if type(value) is FrozenJsonObject:
        if type(value.items) is not tuple:
            raise TypeError("FrozenJsonObject.items must be an exact tuple")
        if len(value.items) > limits.max_container_items:
            raise ValueError("object exceeds max_container_items")
        identity = id(value)
        if identity in state.active_container_ids:
            raise ValueError("frozen typed-JSON values cannot contain cycles")
        state.active_container_ids.add(identity)
        try:
            previous_key_bytes: bytes | None = None
            for entry in value.items:
                if type(entry) is not tuple or len(entry) != 2:
                    raise TypeError(
                        "frozen object entries must be exact (key, value) tuples"
                    )
                key, item = entry
                key_bytes = _utf8_bytes(key, limits=limits, name="object key")
                _count_text_bytes(state, key_bytes, limits=limits)
                if previous_key_bytes is not None and key_bytes <= previous_key_bytes:
                    raise ValueError(
                        "frozen object keys must be unique and in canonical UTF-8 order"
                    )
                previous_key_bytes = key_bytes
                if not is_frozen_json_value(item):
                    raise TypeError(
                        "frozen objects may contain only frozen typed-JSON values"
                    )
                _validate_frozen(
                    item,
                    limits=limits,
                    depth=depth + 1,
                    state=state,
                )
        finally:
            state.active_container_ids.remove(identity)
        return

    raise TypeError("value is not a frozen typed-JSON value")


@dataclass(slots=True)
class _FreezeState:
    active_container_ids: set[int]
    nodes: int = 0
    text_bytes: int = 0


def freeze_json(
    value: object,
    *,
    limits: TypedJsonLimits = DEFAULT_TYPED_JSON_LIMITS,
) -> FrozenJsonValue:
    """Validate and deeply freeze an exact built-in JSON-like value.

    ``dict`` and ``list`` subclasses are rejected even if they appear benign:
    their iteration and indexing methods are executable behavior.  Tuples are
    not accepted as raw arrays; tuples are reserved for the immutable internal
    representation.
    """

    validate_typed_json_limits(limits)
    if is_frozen_json_value(value):
        _validate_frozen(value, limits=limits)
        return value

    state = _FreezeState(active_container_ids=set())

    def visit(current: object, depth: int) -> FrozenJsonValue:
        if depth > limits.max_depth:
            raise ValueError("typed-JSON value exceeds max_depth")
        state.nodes += 1
        if state.nodes > limits.max_nodes:
            raise ValueError("typed-JSON value exceeds max_nodes")

        if _is_scalar(current):
            _validate_scalar(current, limits=limits)
            if type(current) is str:
                _count_text_bytes(
                    state,
                    _utf8_bytes(current, limits=limits, name="string value"),
                    limits=limits,
                )
            return current  # type: ignore[return-value]

        current_type = type(current)
        if current_type not in (dict, list):
            raise TypeError(
                "typed-JSON values require exact dict/list containers and exact scalars"
            )
        if len(current) > limits.max_container_items:
            raise ValueError("container exceeds max_container_items")
        identity = id(current)
        if identity in state.active_container_ids:
            raise ValueError("typed-JSON values cannot contain cycles")
        state.active_container_ids.add(identity)
        try:
            if current_type is list:
                frozen_items = tuple(visit(item, depth + 1) for item in current)
                # The raw tree has been checked incrementally and the complete
                # frozen root is checked once below.  Calling the public
                # dataclass constructor here would recursively revalidate each
                # just-built child subtree at every ancestor.
                frozen_array = object.__new__(FrozenJsonArray)
                object.__setattr__(frozen_array, "items", frozen_items)
                return frozen_array

            encoded_entries: list[tuple[bytes, str, FrozenJsonValue]] = []
            for key, item in current.items():
                key_bytes = _utf8_bytes(key, limits=limits, name="object key")
                _count_text_bytes(state, key_bytes, limits=limits)
                encoded_entries.append((key_bytes, key, visit(item, depth + 1)))
            encoded_entries.sort(key=lambda entry: entry[0])
            frozen_object = object.__new__(FrozenJsonObject)
            object.__setattr__(
                frozen_object,
                "items",
                tuple((key, item) for _, key, item in encoded_entries),
            )
            return frozen_object
        finally:
            state.active_container_ids.remove(identity)

    frozen = visit(value, 0)
    _validate_frozen(frozen, limits=limits)
    return frozen


def thaw_json(value: FrozenJsonValue) -> object:
    """Return a fresh exact ``dict``/``list`` representation."""

    _validate_frozen(value, limits=DEFAULT_TYPED_JSON_LIMITS)

    def visit(current: FrozenJsonValue) -> object:
        # The complete immutable graph was validated above.  Recursing through
        # this private visitor preserves that one trust-boundary check without
        # revalidating every descendant subtree before thawing it.
        if _is_scalar(current):
            return current
        if type(current) is FrozenJsonArray:
            return [visit(item) for item in current.items]
        if type(current) is FrozenJsonObject:
            return {key: visit(item) for key, item in current.items}
        raise AssertionError("validated frozen value had an impossible type")

    return visit(value)


def _uint64(value: int) -> bytes:
    if type(value) is not int or value < 0 or value > _UINT64_MAX:
        raise ValueError("canonical length is outside uint64 range")
    return value.to_bytes(8, "big", signed=False)


@dataclass(slots=True)
class _CanonicalWriter:
    limit: int
    chunks: list[bytes]
    size: int = 0

    def add(self, value: bytes) -> None:
        if type(value) is not bytes:
            raise TypeError("canonical chunks must be exact bytes")
        self.size += len(value)
        if self.size > self.limit:
            raise ValueError("typed-JSON canonical form exceeds max_canonical_bytes")
        self.chunks.append(value)


def canonical_typed_json_bytes(
    value: object,
    *,
    limits: TypedJsonLimits = DEFAULT_TYPED_JSON_LIMITS,
) -> bytes:
    """Encode *value* in the version-1 type-sensitive canonical form."""

    frozen = freeze_json(value, limits=limits)
    writer = _CanonicalWriter(limits.max_canonical_bytes, [])

    def encode(current: FrozenJsonValue) -> None:
        if current is None:
            writer.add(b"n")
        elif type(current) is bool:
            writer.add(b"b1" if current else b"b0")
        elif type(current) is int:
            integer_bytes = str(current).encode("ascii", errors="strict")
            writer.add(b"i")
            writer.add(_uint64(len(integer_bytes)))
            writer.add(integer_bytes)
        elif type(current) is float:
            writer.add(b"f")
            writer.add(struct.pack(">d", current))
        elif type(current) is str:
            string_bytes = _utf8_bytes(current, limits=limits, name="string value")
            writer.add(b"s")
            writer.add(_uint64(len(string_bytes)))
            writer.add(string_bytes)
        elif type(current) is FrozenJsonArray:
            writer.add(b"a")
            writer.add(_uint64(len(current.items)))
            for item in current.items:
                encode(item)
        elif type(current) is FrozenJsonObject:
            writer.add(b"o")
            writer.add(_uint64(len(current.items)))
            for key, item in current.items:
                key_bytes = _utf8_bytes(key, limits=limits, name="object key")
                writer.add(_uint64(len(key_bytes)))
                writer.add(key_bytes)
                encode(item)
        else:  # pragma: no cover - freeze_json closes the union.
            raise AssertionError("unsupported frozen typed-JSON value")

    encode(frozen)
    return b"".join(writer.chunks)


def typed_json_sha256(
    value: object,
    *,
    limits: TypedJsonLimits = DEFAULT_TYPED_JSON_LIMITS,
) -> str:
    """Hash the exact typed tree with an explicit versioned domain tag."""

    payload = canonical_typed_json_bytes(value, limits=limits)
    return _typed_json_sha256_canonical_bytes(payload)


def _typed_json_sha256_canonical_bytes(payload: bytes) -> str:
    """Hash canonical bytes already produced by this module's encoder.

    This is an internal de-duplication seam for trusted domain validators that
    need both canonical bytes and their typed-JSON digest.  Public callers must
    continue through :func:`typed_json_sha256`, which validates and encodes.
    """

    if type(payload) is not bytes:
        raise TypeError("canonical typed-JSON payload must be exact bytes")
    digest = hashlib.sha256()
    digest.update(_TYPED_JSON_HASH_DOMAIN)
    digest.update(payload)
    return digest.hexdigest()


def typed_json_equal(
    left: object,
    right: object,
    *,
    limits: TypedJsonLimits = DEFAULT_TYPED_JSON_LIMITS,
) -> bool:
    """Type-sensitive structural equality for validated values."""

    return canonical_typed_json_bytes(left, limits=limits) == canonical_typed_json_bytes(
        right,
        limits=limits,
    )


def is_json_scalar(value: object) -> bool:
    """Validate the exact scalar type without accepting numeric subclasses."""

    if not _is_scalar(value):
        return False
    _validate_scalar(value, limits=DEFAULT_TYPED_JSON_LIMITS)
    return True


__all__ = [
    "DEFAULT_TYPED_JSON_LIMITS",
    "FrozenJsonArray",
    "FrozenJsonObject",
    "FrozenJsonValue",
    "TypedJsonLimits",
    "canonical_typed_json_bytes",
    "freeze_json",
    "is_frozen_json_value",
    "is_json_scalar",
    "thaw_json",
    "typed_json_equal",
    "typed_json_sha256",
    "validate_typed_json_limits",
]
