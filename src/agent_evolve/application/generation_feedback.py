"""Budgeted inter-generation feedback contracts.

The optimizer owns admission and accounting, while an injected interceptor owns
the feedback behavior (for example, trace reflection or memory curation).  A
reservation is frozen before a generation starts.  The interceptor is invoked
only after that generation's receipt has been sealed, so the next planner call
can observe its effects without allowing feedback to alter in-flight evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent_evolve.application.budgeted_optimizer import (
        GenerationPlan,
        GenerationReceipt,
        OptimizerState,
    )


_POLICY_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_HASH_DOMAIN = b"agent-evolve:generation-feedback:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _record_hash(kind: str, value: object) -> str:
    return hashlib.sha256(
        _HASH_DOMAIN + kind.encode("ascii") + b"\x00" + _canonical_json(value)
    ).hexdigest()


def _validate_metadata(value: tuple[tuple[str, str], ...], *, name: str) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or any(type(part) is not str or not part for part in item)
        ):
            raise TypeError(f"{name} must contain non-empty exact string pairs")
    if value != tuple(sorted(set(value))):
        raise ValueError(f"{name} must be unique and canonically sorted")


@dataclass(frozen=True, slots=True)
class GenerationFeedbackReservation:
    """Pre-generation upper bound for one interceptor invocation."""

    policy_id: str
    policy_version: int
    logical_llm_calls: int
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.policy_id) is not str
            or _POLICY_ID.fullmatch(self.policy_id) is None
        ):
            raise ValueError("policy_id must use the closed lowercase token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        if type(self.logical_llm_calls) is not int or self.logical_llm_calls < 0:
            raise ValueError("logical_llm_calls must be a non-negative exact integer")
        _validate_metadata(self.metadata, name="metadata")

    def to_record(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "logical_llm_calls": self.logical_llm_calls,
            "metadata": [list(item) for item in self.metadata],
        }

    @property
    def reservation_hash(self) -> str:
        return _record_hash("reservation", self.to_record())


@dataclass(frozen=True, slots=True)
class GenerationFeedbackContext:
    """Immutable post-generation evidence exposed to an interceptor."""

    state: OptimizerState
    plan: GenerationPlan
    generation_receipt: GenerationReceipt
    reservation: GenerationFeedbackReservation

    def __post_init__(self) -> None:
        if type(self.reservation) is not GenerationFeedbackReservation:
            raise TypeError(
                "reservation must be an exact GenerationFeedbackReservation"
            )
        if self.state.generation != self.plan.generation:
            raise ValueError("feedback state and plan generations differ")
        if self.generation_receipt.generation != self.plan.generation:
            raise ValueError("feedback receipt and plan generations differ")
        if not self.state.generation_receipts:
            raise ValueError("feedback state has no sealed generation receipt")
        if self.state.generation_receipts[-1] != self.generation_receipt:
            raise ValueError(
                "feedback must observe the latest sealed generation receipt"
            )


@dataclass(frozen=True, slots=True)
class GenerationFeedbackResult:
    """Interceptor-declared consumption and content-free result metadata."""

    logical_llm_calls_used: int
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.logical_llm_calls_used) is not int
            or self.logical_llm_calls_used < 0
        ):
            raise ValueError(
                "logical_llm_calls_used must be a non-negative exact integer"
            )
        _validate_metadata(self.metadata, name="metadata")


@dataclass(frozen=True, slots=True)
class GenerationFeedbackReceipt:
    """Authenticated accounting record for one completed interceptor call."""

    generation: int
    policy_id: str
    policy_version: int
    reservation_hash: str
    generation_receipt_hash: str
    logical_llm_calls_before: int
    logical_llm_calls_after: int
    reserved_logical_llm_calls: int
    used_logical_llm_calls: int
    result_metadata: tuple[tuple[str, str], ...]
    receipt_hash: str

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive exact integer")
        if (
            type(self.policy_id) is not str
            or _POLICY_ID.fullmatch(self.policy_id) is None
        ):
            raise ValueError("policy_id must use the closed lowercase token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        for name in ("reservation_hash", "generation_receipt_hash", "receipt_hash"):
            value = getattr(self, name)
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in (
            "logical_llm_calls_before",
            "logical_llm_calls_after",
            "reserved_logical_llm_calls",
            "used_logical_llm_calls",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")
        if self.used_logical_llm_calls > self.reserved_logical_llm_calls:
            raise ValueError("used feedback calls cannot exceed the reservation")
        if self.logical_llm_calls_after - self.logical_llm_calls_before != (
            self.used_logical_llm_calls
        ):
            raise ValueError("feedback counters differ from declared consumption")
        _validate_metadata(self.result_metadata, name="result_metadata")


def _receipt_record(receipt: GenerationFeedbackReceipt) -> dict[str, object]:
    return {
        "generation": receipt.generation,
        "policy_id": receipt.policy_id,
        "policy_version": receipt.policy_version,
        "reservation_hash": receipt.reservation_hash,
        "generation_receipt_hash": receipt.generation_receipt_hash,
        "logical_llm_calls_before": receipt.logical_llm_calls_before,
        "logical_llm_calls_after": receipt.logical_llm_calls_after,
        "reserved_logical_llm_calls": receipt.reserved_logical_llm_calls,
        "used_logical_llm_calls": receipt.used_logical_llm_calls,
        "result_metadata": [list(item) for item in receipt.result_metadata],
    }


def generation_feedback_receipt_hash(receipt: GenerationFeedbackReceipt) -> str:
    """Recompute the canonical feedback receipt identity."""

    if type(receipt) is not GenerationFeedbackReceipt:
        raise TypeError("receipt must be an exact GenerationFeedbackReceipt")
    return _record_hash("receipt", _receipt_record(receipt))


def validate_generation_feedback_receipt(
    receipt: GenerationFeedbackReceipt,
) -> None:
    """Fail closed unless a feedback receipt authenticates its contents."""

    if type(receipt) is not GenerationFeedbackReceipt:
        raise TypeError("receipt must be an exact GenerationFeedbackReceipt")
    GenerationFeedbackReceipt.__post_init__(receipt)
    if generation_feedback_receipt_hash(receipt) != receipt.receipt_hash:
        raise ValueError("feedback receipt hash does not authenticate its contents")


def seal_generation_feedback(
    *,
    context: GenerationFeedbackContext,
    result: GenerationFeedbackResult,
) -> GenerationFeedbackReceipt:
    """Validate exact consumption and seal the feedback accounting record."""

    if type(context) is not GenerationFeedbackContext:
        raise TypeError("context must be an exact GenerationFeedbackContext")
    GenerationFeedbackContext.__post_init__(context)
    if type(result) is not GenerationFeedbackResult:
        raise TypeError("result must be an exact GenerationFeedbackResult")
    GenerationFeedbackResult.__post_init__(result)
    reservation = context.reservation
    if result.logical_llm_calls_used > reservation.logical_llm_calls:
        raise ValueError("feedback consumption exceeds its pre-generation reservation")
    before = context.state.logical_llm_calls
    after = before + result.logical_llm_calls_used
    provisional = GenerationFeedbackReceipt(
        generation=context.plan.generation,
        policy_id=reservation.policy_id,
        policy_version=reservation.policy_version,
        reservation_hash=reservation.reservation_hash,
        generation_receipt_hash=context.generation_receipt.receipt_hash,
        logical_llm_calls_before=before,
        logical_llm_calls_after=after,
        reserved_logical_llm_calls=reservation.logical_llm_calls,
        used_logical_llm_calls=result.logical_llm_calls_used,
        result_metadata=result.metadata,
        receipt_hash="0" * 64,
    )
    return GenerationFeedbackReceipt(
        generation=provisional.generation,
        policy_id=provisional.policy_id,
        policy_version=provisional.policy_version,
        reservation_hash=provisional.reservation_hash,
        generation_receipt_hash=provisional.generation_receipt_hash,
        logical_llm_calls_before=provisional.logical_llm_calls_before,
        logical_llm_calls_after=provisional.logical_llm_calls_after,
        reserved_logical_llm_calls=provisional.reserved_logical_llm_calls,
        used_logical_llm_calls=provisional.used_logical_llm_calls,
        result_metadata=provisional.result_metadata,
        receipt_hash=_record_hash("receipt", _receipt_record(provisional)),
    )


@runtime_checkable
class GenerationFeedbackInterceptor(Protocol):
    """Optional feedback behavior injected between sealed generations."""

    def reserve(
        self,
        *,
        state: OptimizerState,
        plan: GenerationPlan,
    ) -> GenerationFeedbackReservation: ...

    async def after_generation(
        self,
        context: GenerationFeedbackContext,
    ) -> GenerationFeedbackResult: ...


__all__ = [
    "GenerationFeedbackContext",
    "GenerationFeedbackInterceptor",
    "GenerationFeedbackReceipt",
    "GenerationFeedbackReservation",
    "GenerationFeedbackResult",
    "generation_feedback_receipt_hash",
    "seal_generation_feedback",
    "validate_generation_feedback_receipt",
]
