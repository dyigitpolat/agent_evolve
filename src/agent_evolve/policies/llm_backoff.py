"""Injectable exponential backoff and jitter policies for LLM task retries."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol, runtime_checkable

from agent_evolve.domain.llm_task_queue import RetryClassification, RetryReason


@runtime_checkable
class JitterPolicy(Protocol):
    def apply(
        self,
        upper_bound_ns: int,
        *,
        task_id: str,
        failed_attempt_number: int,
    ) -> int: ...


@runtime_checkable
class RandomRange(Protocol):
    def randrange(self, stop: int) -> int: ...


@dataclass(frozen=True, slots=True)
class NoJitter:
    """Use the capped exponential delay exactly."""

    def apply(
        self,
        upper_bound_ns: int,
        *,
        task_id: str,
        failed_attempt_number: int,
    ) -> int:
        del task_id, failed_attempt_number
        if type(upper_bound_ns) is not int or upper_bound_ns < 0:
            raise ValueError("upper_bound_ns must be a non-negative integer")
        return upper_bound_ns


@dataclass(frozen=True, slots=True)
class FullJitter:
    """Choose uniformly from the inclusive interval ``[0, upper_bound]``."""

    random: RandomRange

    def __post_init__(self) -> None:
        if not isinstance(self.random, RandomRange):
            raise TypeError("random must implement RandomRange")

    def apply(
        self,
        upper_bound_ns: int,
        *,
        task_id: str,
        failed_attempt_number: int,
    ) -> int:
        del task_id, failed_attempt_number
        if type(upper_bound_ns) is not int or upper_bound_ns < 0:
            raise ValueError("upper_bound_ns must be a non-negative integer")
        sampled = self.random.randrange(upper_bound_ns + 1)
        if type(sampled) is not int or not 0 <= sampled <= upper_bound_ns:
            raise ValueError("random source returned a value outside the jitter interval")
        return sampled


@dataclass(frozen=True, slots=True)
class DeterministicHashJitter:
    """Task-keyed full jitter independent of concurrent completion order.

    The domain string makes the byte framing experiment- or application-specific,
    while the task ID and failed-attempt number prevent a shared mutable RNG from
    coupling otherwise independent requests.
    """

    seed: int
    domain: str = "agent-evolve-jitter-v1"

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative exact integer")
        if (
            type(self.domain) is not str
            or not self.domain
            or "\x00" in self.domain
            or not self.domain.isascii()
        ):
            raise ValueError("domain must be non-empty NUL-free ASCII")

    def apply(
        self,
        upper_bound_ns: int,
        *,
        task_id: str,
        failed_attempt_number: int,
    ) -> int:
        if type(upper_bound_ns) is not int or upper_bound_ns < 0:
            raise ValueError("upper_bound_ns must be a non-negative integer")
        if type(task_id) is not str or not task_id or "\x00" in task_id:
            raise ValueError("task_id must be non-empty and NUL-free")
        if type(failed_attempt_number) is not int or failed_attempt_number < 1:
            raise ValueError("failed_attempt_number must be a positive integer")
        framed = b"\x00".join(
            (
                self.domain.encode("ascii"),
                str(self.seed).encode("ascii"),
                task_id.encode("utf-8", errors="strict"),
                str(failed_attempt_number).encode("ascii"),
            )
        )
        value = int.from_bytes(hashlib.sha256(framed).digest(), "big")
        return value % (upper_bound_ns + 1)


@dataclass(frozen=True, slots=True)
class ExponentialBackoff:
    """Capped exponential backoff with jitter and a rate-limit floor.

    Full jitter is useful for ordinary transient failures, but a sampled delay
    close to zero defeats provider recovery after an HTTP 429.  The optional
    floor is classification-aware: it applies only to ``RATE_LIMIT`` retries,
    leaving timeout, transient, and structured-output behavior unchanged.
    """

    base_delay_ns: int
    max_delay_ns: int
    jitter: JitterPolicy = NoJitter()
    rate_limit_floor_ns: int = 0

    def __post_init__(self) -> None:
        if type(self.base_delay_ns) is not int or self.base_delay_ns < 0:
            raise ValueError("base_delay_ns must be a non-negative integer")
        if type(self.max_delay_ns) is not int or self.max_delay_ns < 0:
            raise ValueError("max_delay_ns must be a non-negative integer")
        if self.base_delay_ns > self.max_delay_ns:
            raise ValueError("base_delay_ns cannot exceed max_delay_ns")
        if (
            type(self.rate_limit_floor_ns) is not int
            or self.rate_limit_floor_ns < 0
            or self.rate_limit_floor_ns > self.max_delay_ns
        ):
            raise ValueError(
                "rate_limit_floor_ns must lie in [0, max_delay_ns]"
            )
        if not isinstance(self.jitter, JitterPolicy):
            raise TypeError("jitter must implement JitterPolicy")

    def delay_ns(
        self,
        *,
        task_id: str,
        failed_attempt_number: int,
        classification: RetryClassification,
    ) -> int:
        if type(task_id) is not str or not task_id:
            raise ValueError("task_id must be a non-empty string")
        if type(failed_attempt_number) is not int or failed_attempt_number < 1:
            raise ValueError("failed_attempt_number must be a positive integer")
        if type(classification) is not RetryClassification:
            raise TypeError("classification must be a RetryClassification")

        if self.base_delay_ns == 0 or self.max_delay_ns == 0:
            upper_bound = 0
        else:
            shift = failed_attempt_number - 1
            # Avoid constructing an enormous intermediate integer for a bad
            # attempt value while retaining the exact capped result.
            if shift >= self.max_delay_ns.bit_length():
                upper_bound = self.max_delay_ns
            else:
                upper_bound = min(self.base_delay_ns << shift, self.max_delay_ns)
        delay = self.jitter.apply(
            upper_bound,
            task_id=task_id,
            failed_attempt_number=failed_attempt_number,
        )
        if classification.reason is RetryReason.RATE_LIMIT:
            return max(delay, self.rate_limit_floor_ns)
        return delay
