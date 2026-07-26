from __future__ import annotations

import pytest

from agent_evolve.ports.generation_failure import (
    GenerationFailureDisposition,
    classify_generation_failure,
)
from agent_evolve.ports.structured_generator import (
    GenerationFailureKind,
    StructuredGenerationError,
)


@pytest.mark.parametrize(
    ("kind", "expected"),
    (
        (
            GenerationFailureKind.OUTPUT_INVALID,
            GenerationFailureDisposition.MODEL_OR_SCHEMA_FAILURE,
        ),
        (
            GenerationFailureKind.CONTENT_REJECTED,
            GenerationFailureDisposition.MODEL_OR_SCHEMA_FAILURE,
        ),
        (
            GenerationFailureKind.RATE_LIMITED,
            GenerationFailureDisposition.INFRASTRUCTURE_FAILURE,
        ),
        (
            GenerationFailureKind.TIMEOUT,
            GenerationFailureDisposition.INFRASTRUCTURE_FAILURE,
        ),
        (
            GenerationFailureKind.PROVIDER_UNAVAILABLE,
            GenerationFailureDisposition.INFRASTRUCTURE_FAILURE,
        ),
        (
            GenerationFailureKind.AUTHENTICATION,
            GenerationFailureDisposition.INFRASTRUCTURE_FAILURE,
        ),
        (
            GenerationFailureKind.PAYMENT_REQUIRED,
            GenerationFailureDisposition.INFRASTRUCTURE_FAILURE,
        ),
        (
            GenerationFailureKind.UNKNOWN,
            GenerationFailureDisposition.INFRASTRUCTURE_FAILURE,
        ),
    ),
)
def test_explicit_structured_failure_projection(kind, expected) -> None:
    error = StructuredGenerationError(
        kind=kind,
        retryable=False,
        safe_message="sanitized test failure",
    )
    assert classify_generation_failure(error) is expected


def test_untyped_and_hostile_failure_projections_fail_closed_as_infrastructure() -> (
    None
):
    class WrongProjection(RuntimeError):
        generation_failure_disposition = "model_or_schema_failure"

    class RaisingProjection(RuntimeError):
        @property
        def generation_failure_disposition(self):
            raise RuntimeError("hostile property")

    for error in (RuntimeError("untyped"), WrongProjection(), RaisingProjection()):
        assert (
            classify_generation_failure(error)
            is GenerationFailureDisposition.INFRASTRUCTURE_FAILURE
        )
