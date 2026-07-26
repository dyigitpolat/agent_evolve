"""Provider- and method-neutral contracts for paired benchmark block schedules.

The request surface is deliberately closed.  It binds only benchmark/task
identity, a finite contract and partition layout, an exact eligibility mask,
public seed material, canonical block identities, and one replicate index.
There is no metadata escape hatch through which a method, treatment, provider,
prompt, or observed outcome can influence assignment.

The application layer owns the deterministic hash ranking.  These value
objects authenticate the complete no-replacement schedule separately from the
selected block, so every replicate can share one schedule receipt while
retaining its own replayable selection receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from agent_evolve.domain.patch import require_sha256


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
ELIGIBLE_ROW_MASK_DOMAIN = "agent-evolve:finite-contract-eligible-mask:v1"
_ELIGIBLE_ROW_MASK_FRAMING = ELIGIBLE_ROW_MASK_DOMAIN.encode("ascii") + b"\x00"
_POLICY_BINDING_FRAMING = b"agent-evolve:paired-block-schedule-policy:v1\x00"
_BLOCK_IDENTITY_FRAMING = b"agent-evolve:paired-block-identity:v1\x00"
_REQUEST_FRAMING = b"agent-evolve:paired-block-schedule-request:v2\x00"
_FULL_SCHEDULE_FRAMING = b"agent-evolve:paired-block-full-schedule:v2\x00"
_SELECTED_BLOCK_FRAMING = (
    b"agent-evolve:paired-block-selection-receipt:v2\x00"
)
_UINT64_MAX = (1 << 64) - 1


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _require_identifier(value: object, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must use the canonical lowercase identifier grammar")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class ExactEligibleRowMask:
    """Exact global finite-contract rows eligible for downstream selection."""

    finite_contract_identity_sha256: str
    eligible_global_row_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        require_sha256(
            self.finite_contract_identity_sha256,
            "finite_contract_identity_sha256",
        )
        if type(self.eligible_global_row_indices) is not tuple:
            raise TypeError("eligible_global_row_indices must be an exact tuple")
        if any(type(value) is not int for value in self.eligible_global_row_indices):
            raise TypeError("eligible global rows must be exact integers")
        if any(value < 0 for value in self.eligible_global_row_indices):
            raise ValueError("eligible global rows cannot be negative")
        if self.eligible_global_row_indices != tuple(
            sorted(set(self.eligible_global_row_indices))
        ):
            raise ValueError("eligible global rows must be unique and canonically sorted")

    def payload_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "finite_contract_identity_sha256": (
                self.finite_contract_identity_sha256
            ),
            "eligible_global_row_indices": list(
                self.eligible_global_row_indices
            ),
        }

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload_record())).hexdigest()

    @property
    def mask_sha256(self) -> str:
        return _hash(_ELIGIBLE_ROW_MASK_FRAMING, self.payload_record())

    def to_record(self) -> dict[str, object]:
        return {
            "domain": ELIGIBLE_ROW_MASK_DOMAIN,
            "payload": self.payload_record(),
            "payload_sha256": self.payload_sha256,
            "digest_sha256": self.mask_sha256,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is ExactEligibleRowMask
            and self.mask_sha256 == other.mask_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class PairedBlockSchedulePolicyBinding:
    """Exact public algorithm and rank-domain identity used by a schedule."""

    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    rank_domain: str

    def __post_init__(self) -> None:
        _require_identifier(self.policy_id, "policy_id")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")
        _require_identifier(self.rank_domain, "rank_domain")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            "rank_domain": self.rank_domain,
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_POLICY_BINDING_FRAMING, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is PairedBlockSchedulePolicyBinding
            and self.binding_sha256 == other.binding_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class CanonicalBlockIdentity:
    """One canonical, contiguous block in a complete benchmark partition."""

    block_index: int
    block_spec_sha256: str
    global_row_start: int
    global_row_stop: int

    def __post_init__(self) -> None:
        if type(self.block_index) is not int or self.block_index < 0:
            raise ValueError("block_index must be a non-negative exact integer")
        require_sha256(self.block_spec_sha256, "block_spec_sha256")
        if type(self.global_row_start) is not int or self.global_row_start < 0:
            raise ValueError(
                "global_row_start must be a non-negative exact integer"
            )
        if (
            type(self.global_row_stop) is not int
            or self.global_row_stop <= self.global_row_start
        ):
            raise ValueError("global_row_stop must be greater than global_row_start")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "block_index": self.block_index,
            "block_spec_sha256": self.block_spec_sha256,
            "global_row_start": self.global_row_start,
            "global_row_stop": self.global_row_stop,
        }

    @property
    def identity_sha256(self) -> str:
        return _hash(_BLOCK_IDENTITY_FRAMING, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "block_identity_sha256": self.identity_sha256,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is CanonicalBlockIdentity
            and self.identity_sha256 == other.identity_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True, eq=False)
class PairedBlockScheduleRequest:
    """Closed, method-independent input for one replicate's block assignment."""

    benchmark_id: str
    task_sha256: str
    finite_contract_identity_sha256: str
    partition_layout_sha256: str
    policy: PairedBlockSchedulePolicyBinding
    eligible_mask: ExactEligibleRowMask
    public_seed: int
    public_seed_source_sha256: str
    blocks: tuple[CanonicalBlockIdentity, ...]
    replicate_index: int

    def __post_init__(self) -> None:
        _require_identifier(self.benchmark_id, "benchmark_id")
        require_sha256(self.task_sha256, "task_sha256")
        require_sha256(
            self.finite_contract_identity_sha256,
            "finite_contract_identity_sha256",
        )
        require_sha256(self.partition_layout_sha256, "partition_layout_sha256")
        if type(self.policy) is not PairedBlockSchedulePolicyBinding:
            raise TypeError(
                "policy must be an exact PairedBlockSchedulePolicyBinding"
            )
        self.policy.__post_init__()
        if type(self.eligible_mask) is not ExactEligibleRowMask:
            raise TypeError("eligible_mask must be an exact ExactEligibleRowMask")
        self.eligible_mask.__post_init__()
        if (
            self.eligible_mask.finite_contract_identity_sha256
            != self.finite_contract_identity_sha256
        ):
            raise ValueError("eligible mask names another finite contract")
        if (
            type(self.public_seed) is not int
            or self.public_seed < 0
            or self.public_seed > _UINT64_MAX
        ):
            raise ValueError("public_seed must be an unsigned 64-bit exact integer")
        require_sha256(
            self.public_seed_source_sha256,
            "public_seed_source_sha256",
        )
        if type(self.blocks) is not tuple or not self.blocks:
            raise ValueError("blocks must be a non-empty exact tuple")
        if any(type(value) is not CanonicalBlockIdentity for value in self.blocks):
            raise TypeError("blocks must contain exact CanonicalBlockIdentity values")
        for value in self.blocks:
            value.__post_init__()

        canonical = tuple(sorted(self.blocks, key=lambda value: value.block_index))
        if len({value.identity_sha256 for value in canonical}) != len(canonical):
            raise ValueError("canonical block identities cannot repeat")
        cursor = 0
        for expected_index, block in enumerate(canonical):
            if block.block_index != expected_index:
                raise ValueError("block indices must be contiguous canonical positions")
            if block.global_row_start != cursor:
                raise ValueError("blocks must provide gap-free, overlap-free coverage")
            cursor = block.global_row_stop
        object.__setattr__(self, "blocks", canonical)

        if any(
            value >= cursor
            for value in self.eligible_mask.eligible_global_row_indices
        ):
            raise ValueError("eligible mask contains a row outside the block partition")
        if (
            type(self.replicate_index) is not int
            or self.replicate_index < 0
            or self.replicate_index >= len(canonical)
        ):
            raise ValueError(
                "replicate_index must be an exact integer within the no-replacement schedule"
            )

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    def schedule_basis_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 2,
            "benchmark_id": self.benchmark_id,
            "task_sha256": self.task_sha256,
            "finite_contract_identity_sha256": (
                self.finite_contract_identity_sha256
            ),
            "partition_layout_sha256": self.partition_layout_sha256,
            "eligible_mask_sha256": self.eligible_mask.mask_sha256,
            "public_seed": self.public_seed,
            "block_count": self.block_count,
        }

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "policy": self.policy.to_record(),
            "schedule_basis": self.schedule_basis_record(),
            "eligible_mask": self.eligible_mask.to_record(),
            "public_seed_source_sha256": self.public_seed_source_sha256,
            "canonical_blocks": [value.to_record() for value in self.blocks],
            "replicate_index": self.replicate_index,
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_REQUEST_FRAMING, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is PairedBlockScheduleRequest
            and self.request_sha256 == other.request_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class RankedBlockIdentity:
    """One block's immutable position in a hash-ranked full schedule."""

    schedule_position: int
    block: CanonicalBlockIdentity
    rank_digest_sha256: str

    def __post_init__(self) -> None:
        if type(self.schedule_position) is not int or self.schedule_position < 0:
            raise ValueError(
                "schedule_position must be a non-negative exact integer"
            )
        if type(self.block) is not CanonicalBlockIdentity:
            raise TypeError("block must be an exact CanonicalBlockIdentity")
        self.block.__post_init__()
        require_sha256(self.rank_digest_sha256, "rank_digest_sha256")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "schedule_position": self.schedule_position,
            "block": self.block.to_record(),
            "rank_digest_sha256": self.rank_digest_sha256,
        }


@dataclass(frozen=True, slots=True, eq=False)
class PairedBlockScheduleResult:
    """Authenticated complete schedule and one replicate's selected block."""

    request: PairedBlockScheduleRequest
    ranked_blocks: tuple[RankedBlockIdentity, ...]

    def __post_init__(self) -> None:
        if type(self.request) is not PairedBlockScheduleRequest:
            raise TypeError("request must be an exact PairedBlockScheduleRequest")
        self.request.__post_init__()
        if type(self.ranked_blocks) is not tuple or len(self.ranked_blocks) != (
            self.request.block_count
        ):
            raise ValueError("ranked_blocks must cover the complete block schedule")
        if any(type(value) is not RankedBlockIdentity for value in self.ranked_blocks):
            raise TypeError("ranked_blocks must contain exact RankedBlockIdentity values")
        for expected_position, value in enumerate(self.ranked_blocks):
            value.__post_init__()
            if value.schedule_position != expected_position:
                raise ValueError("ranked blocks must use canonical schedule positions")
        if tuple(
            sorted(value.block.identity_sha256 for value in self.ranked_blocks)
        ) != tuple(sorted(value.identity_sha256 for value in self.request.blocks)):
            raise ValueError("ranked blocks differ from the complete request partition")
        if tuple(
            (value.rank_digest_sha256, value.block.block_index)
            for value in self.ranked_blocks
        ) != tuple(
            sorted(
                (
                    value.rank_digest_sha256,
                    value.block.block_index,
                )
                for value in self.ranked_blocks
            )
        ):
            raise ValueError("ranked blocks are not in canonical hash-rank order")

    @property
    def block_schedule(self) -> tuple[int, ...]:
        return tuple(value.block.block_index for value in self.ranked_blocks)

    def full_schedule_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 2,
            "policy": self.request.policy.to_record(),
            "schedule_basis": self.request.schedule_basis_record(),
            "public_seed_source_sha256": (
                self.request.public_seed_source_sha256
            ),
            "ranked_blocks": [value.to_record() for value in self.ranked_blocks],
            "block_schedule": list(self.block_schedule),
        }

    @property
    def full_schedule_sha256(self) -> str:
        return _hash(_FULL_SCHEDULE_FRAMING, self.full_schedule_record())

    @property
    def selected_ranked_block(self) -> RankedBlockIdentity:
        return self.ranked_blocks[self.request.replicate_index]

    def selected_block_record(self) -> dict[str, object]:
        selected = self.selected_ranked_block
        return {
            "schema_version": 2,
            "request_sha256": self.request.request_sha256,
            "full_schedule_sha256": self.full_schedule_sha256,
            "replicate_index": self.request.replicate_index,
            "schedule_position": selected.schedule_position,
            "selected_block": selected.block.to_record(),
            "rank_digest_sha256": selected.rank_digest_sha256,
        }

    @property
    def selected_block_receipt_sha256(self) -> str:
        return _hash(_SELECTED_BLOCK_FRAMING, self.selected_block_record())

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "request": self.request.to_record(),
            "full_schedule": self.full_schedule_record(),
            "full_schedule_sha256": self.full_schedule_sha256,
            "selection": self.selected_block_record(),
            "selected_block_receipt_sha256": (
                self.selected_block_receipt_sha256
            ),
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is PairedBlockScheduleResult
            and self.selected_block_receipt_sha256
            == other.selected_block_receipt_sha256
        )

    __hash__ = None


__all__ = [
    "CanonicalBlockIdentity",
    "ELIGIBLE_ROW_MASK_DOMAIN",
    "ExactEligibleRowMask",
    "PairedBlockSchedulePolicyBinding",
    "PairedBlockScheduleRequest",
    "PairedBlockScheduleResult",
    "RankedBlockIdentity",
]
