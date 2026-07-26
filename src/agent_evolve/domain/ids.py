"""Runtime-distinct, serialization-stable identifiers.

``typing.NewType`` is deliberately avoided: event-store validation must be able to
distinguish a run ID from a candidate ID at runtime, not only in a type checker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Type, TypeVar

from agent_evolve.domain.durable_text import (
    contains_credential_shape,
    contains_identifier_content_marker,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_STABLE_ID_LENGTH = 128
MAX_ID_NAMESPACE_LENGTH = 48


def _is_safe_identifier_text(value: str, *, max_length: int) -> bool:
    return (
        type(value) is str
        and 0 < len(value) <= max_length
        and _SAFE_ID.fullmatch(value) is not None
        and not contains_credential_shape(value)
        and not contains_identifier_content_marker(value)
    )


def validate_id_namespace(value: str) -> None:
    """Validate a bounded non-content namespace used in deterministic IDs."""

    if not _is_safe_identifier_text(value, max_length=MAX_ID_NAMESPACE_LENGTH):
        raise ValueError("namespace violates the durable identifier policy")


@dataclass(frozen=True, slots=True, order=True)
class StableId:
    """Base value object for an ID that is safe in JSON and file names."""

    value: str
    PREFIX: ClassVar[str] = "id"

    def __post_init__(self) -> None:
        if type(self.value) is not str or not self.value:
            raise ValueError(f"{type(self).__name__} value must be a non-empty string")
        if not self.value.startswith(f"{self.PREFIX}_"):
            raise ValueError(
                f"{type(self).__name__} must start with {self.PREFIX!r} followed by '_'"
            )
        if not _is_safe_identifier_text(
            self.value,
            max_length=MAX_STABLE_ID_LENGTH,
        ):
            raise ValueError(
                f"{type(self).__name__} contains content unsafe for durable storage"
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class RunId(StableId):
    PREFIX: ClassVar[str] = "run"


@dataclass(frozen=True, slots=True, order=True)
class EventId(StableId):
    PREFIX: ClassVar[str] = "event"


@dataclass(frozen=True, slots=True, order=True)
class GenerationId(StableId):
    PREFIX: ClassVar[str] = "generation"


@dataclass(frozen=True, slots=True, order=True)
class CandidateId(StableId):
    PREFIX: ClassVar[str] = "candidate"


@dataclass(frozen=True, slots=True, order=True)
class InsightId(StableId):
    PREFIX: ClassVar[str] = "insight"


@dataclass(frozen=True, slots=True, order=True)
class OperatorInvocationId(StableId):
    PREFIX: ClassVar[str] = "operator"


@dataclass(frozen=True, slots=True, order=True)
class LLMCallId(StableId):
    PREFIX: ClassVar[str] = "call"


@dataclass(frozen=True, slots=True, order=True)
class ProviderAttemptId(StableId):
    PREFIX: ClassVar[str] = "provider_attempt"


@dataclass(frozen=True, slots=True, order=True)
class EvaluationId(StableId):
    PREFIX: ClassVar[str] = "evaluation"


@dataclass(frozen=True, slots=True, order=True)
class EvaluationAttemptId(StableId):
    PREFIX: ClassVar[str] = "evaluation_attempt"


@dataclass(frozen=True, slots=True, order=True)
class CorrelationId(StableId):
    PREFIX: ClassVar[str] = "correlation"


@dataclass(frozen=True, slots=True, order=True)
class ArtifactId(StableId):
    PREFIX: ClassVar[str] = "artifact"

    def __post_init__(self) -> None:
        if (
            type(self.value) is not str
            or not self.value
            or not self.value.startswith(f"{self.PREFIX}_")
        ):
            StableId.__post_init__(self)
        digest = self.value[len(f"{self.PREFIX}_") :]
        if _LOWER_SHA256.fullmatch(digest) is None:
            raise ValueError(
                "ArtifactId must contain a lowercase 64-hex identity digest"
            )
        StableId.__post_init__(self)


StableIdT = TypeVar("StableIdT", bound=StableId)


def parse_stable_id(id_type: Type[StableIdT], value: str) -> StableIdT:
    """Construct *id_type* from its serialized representation."""

    return id_type(value)


ID_TYPES = (
    RunId,
    EventId,
    GenerationId,
    CandidateId,
    InsightId,
    OperatorInvocationId,
    LLMCallId,
    ProviderAttemptId,
    EvaluationId,
    EvaluationAttemptId,
    CorrelationId,
    ArtifactId,
)
