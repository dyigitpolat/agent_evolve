"""Provider-neutral terminal classification for agentic generation failures.

Application services must not infer causal failure semantics from exception class
names or provider-specific status text. Adapters may expose this deliberately
small projection; every absent, malformed, or hostile projection fails closed as
infrastructure. Only an explicit structured/model-output failure is eligible for
the preregistered no-yield intention-to-treat path.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable


class GenerationFailureDisposition(str, Enum):
    MODEL_OR_SCHEMA_FAILURE = "model_or_schema_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


@runtime_checkable
class ClassifiedGenerationFailure(Protocol):
    @property
    def generation_failure_disposition(self) -> GenerationFailureDisposition: ...


def classify_generation_failure(error: BaseException) -> GenerationFailureDisposition:
    """Return a closed terminal class, defaulting every untyped failure to infra."""

    try:
        disposition = getattr(error, "generation_failure_disposition", None)
    except Exception:
        return GenerationFailureDisposition.INFRASTRUCTURE_FAILURE
    if type(disposition) is not GenerationFailureDisposition:
        return GenerationFailureDisposition.INFRASTRUCTURE_FAILURE
    return disposition


__all__ = [
    "ClassifiedGenerationFailure",
    "GenerationFailureDisposition",
    "classify_generation_failure",
]
