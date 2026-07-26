"""Pure hash-ranked paired block scheduling over benchmark-neutral ports."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

from agent_evolve.ports.paired_block_schedule import (
    PairedBlockScheduleRequest,
    PairedBlockSchedulePolicyBinding,
    PairedBlockScheduleResult,
    RankedBlockIdentity,
)


PAIRED_BLOCK_SCHEDULE_POLICY_ID = "paired_benchmark_block_schedule"
PAIRED_BLOCK_SCHEDULE_POLICY_VERSION = 2
PAIRED_BLOCK_SCHEDULE_RANK_DOMAIN = (
    "agent-evolve:paired-benchmark-block-schedule:v2"
)
PAIRED_BLOCK_SCHEDULE_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:paired-benchmark-block-schedule-policy:v2;"
    b"inputs=benchmark,task,finite-contract,partition-layout,exact-eligible-mask,"
    b"public-uint64-seed,public-seed-source-sha256,complete-canonical-block-"
    b"identities,replicate-index;public-seed-source=authenticated-provenance-"
    b"not-rank-entropy;"
    b"rank=sha256(domain,schedule-basis,block-index,block-spec-sha256);"
    b"sort=rank-digest-then-block-index;complete-no-replacement-schedule;"
    b"replicate-index=exact-schedule-position;runtime-rng=forbidden;"
    b"method,treatment,provider,prompt,outcome-identities=absent-by-construction"
).hexdigest()


class PairedBlockScheduleVerificationError(RuntimeError):
    """A supplied paired schedule differs from deterministic public replay."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def paired_block_schedule_policy() -> PairedBlockSchedulePolicyBinding:
    """Return the identified benchmark-neutral no-replacement schedule policy."""

    return PairedBlockSchedulePolicyBinding(
        policy_id=PAIRED_BLOCK_SCHEDULE_POLICY_ID,
        policy_version=PAIRED_BLOCK_SCHEDULE_POLICY_VERSION,
        policy_definition_sha256=(
            PAIRED_BLOCK_SCHEDULE_POLICY_DEFINITION_SHA256
        ),
        rank_domain=PAIRED_BLOCK_SCHEDULE_RANK_DOMAIN,
    )


def rank_paired_benchmark_blocks(
    request: PairedBlockScheduleRequest,
) -> PairedBlockScheduleResult:
    """Hash-rank every block once and select the replicate's schedule position."""

    if type(request) is not PairedBlockScheduleRequest:
        raise TypeError("request must be an exact PairedBlockScheduleRequest")
    request.__post_init__()
    if request.policy != paired_block_schedule_policy():
        raise ValueError("request policy differs from this identified scheduler")
    domain = request.policy.rank_domain.encode("ascii") + b"\x00"
    basis = request.schedule_basis_record()
    ranked: list[tuple[str, int, object]] = []
    for block in request.blocks:
        rank_payload = {
            **basis,
            "block_index": block.block_index,
            "block_spec_sha256": block.block_spec_sha256,
        }
        digest = hashlib.sha256(domain + _canonical_json(rank_payload)).hexdigest()
        ranked.append((digest, block.block_index, block))
    ranked.sort(key=lambda value: (value[0], value[1]))
    return PairedBlockScheduleResult(
        request=request,
        ranked_blocks=tuple(
            RankedBlockIdentity(
                schedule_position=position,
                block=block,  # type: ignore[arg-type]
                rank_digest_sha256=digest,
            )
            for position, (digest, _block_index, block) in enumerate(ranked)
        ),
    )


def verify_paired_block_schedule(
    request: PairedBlockScheduleRequest,
    result: PairedBlockScheduleResult | Mapping[str, object],
) -> PairedBlockScheduleResult:
    """Fail closed unless an object or canonical record equals public replay."""

    expected = rank_paired_benchmark_blocks(request)
    if type(result) is PairedBlockScheduleResult:
        result.__post_init__()
        matches = result == expected and result.to_record() == expected.to_record()
    elif isinstance(result, Mapping):
        matches = type(result) is dict and result == expected.to_record()
    else:
        raise TypeError("result must be an exact schedule result or canonical dict")
    if not matches:
        raise PairedBlockScheduleVerificationError(
            "paired block schedule differs from deterministic public replay"
        )
    return expected


__all__ = [
    "PAIRED_BLOCK_SCHEDULE_POLICY_DEFINITION_SHA256",
    "PAIRED_BLOCK_SCHEDULE_POLICY_ID",
    "PAIRED_BLOCK_SCHEDULE_POLICY_VERSION",
    "PAIRED_BLOCK_SCHEDULE_RANK_DOMAIN",
    "PairedBlockScheduleVerificationError",
    "paired_block_schedule_policy",
    "rank_paired_benchmark_blocks",
    "verify_paired_block_schedule",
]
