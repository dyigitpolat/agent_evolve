from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import fields, replace
import hashlib

import pytest

from agent_evolve.application.paired_block_schedule import (
    PairedBlockScheduleVerificationError,
    paired_block_schedule_policy,
    rank_paired_benchmark_blocks,
    verify_paired_block_schedule,
)
from agent_evolve.ports.paired_block_schedule import (
    CanonicalBlockIdentity,
    ExactEligibleRowMask,
    PairedBlockSchedulePolicyBinding,
    PairedBlockScheduleRequest,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _blocks() -> tuple[CanonicalBlockIdentity, ...]:
    return tuple(
        CanonicalBlockIdentity(
            block_index=index,
            block_spec_sha256=_digest(f"block-spec-{index}"),
            global_row_start=index * 5,
            global_row_stop=(index + 1) * 5,
        )
        for index in range(4)
    )


def _request(
    *,
    replicate_index: int = 0,
    public_seed: int = 20_260_715,
    blocks: tuple[CanonicalBlockIdentity, ...] | None = None,
    eligible_rows: tuple[int, ...] = tuple(range(20)),
    policy: PairedBlockSchedulePolicyBinding | None = None,
    seed_source: str | None = None,
) -> PairedBlockScheduleRequest:
    contract = _digest("contract")
    return PairedBlockScheduleRequest(
        benchmark_id="generic_benchmark_v1",
        task_sha256=_digest("task"),
        finite_contract_identity_sha256=contract,
        partition_layout_sha256=_digest("layout"),
        policy=paired_block_schedule_policy() if policy is None else policy,
        eligible_mask=ExactEligibleRowMask(
            finite_contract_identity_sha256=contract,
            eligible_global_row_indices=eligible_rows,
        ),
        public_seed=public_seed,
        public_seed_source_sha256=(
            _digest("prior-public-seed-protocol")
            if seed_source is None
            else seed_source
        ),
        blocks=_blocks() if blocks is None else blocks,
        replicate_index=replicate_index,
    )


def test_hash_rank_schedule_is_input_permutation_and_concurrency_invariant() -> None:
    canonical = _request()
    reversed_input = _request(blocks=tuple(reversed(_blocks())))
    assert reversed_input.blocks == canonical.blocks
    assert reversed_input.request_sha256 == canonical.request_sha256

    expected = rank_paired_benchmark_blocks(canonical)
    assert rank_paired_benchmark_blocks(reversed_input).to_record() == (
        expected.to_record()
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = tuple(
            executor.map(
                lambda _index: rank_paired_benchmark_blocks(
                    reversed_input if _index % 2 else canonical
                ).to_record(),
                range(64),
            )
        )
    assert all(record == expected.to_record() for record in records)


def test_replicates_cover_one_complete_no_replacement_schedule() -> None:
    results = tuple(
        rank_paired_benchmark_blocks(_request(replicate_index=index))
        for index in range(4)
    )
    assert len({value.full_schedule_sha256 for value in results}) == 1
    assert len({value.selected_block_receipt_sha256 for value in results}) == 4
    assert tuple(
        value.selected_ranked_block.block.block_index for value in results
    ) == results[0].block_schedule
    assert sorted(results[0].block_schedule) == [0, 1, 2, 3]

    for invalid in (-1, 4, True):
        with pytest.raises(ValueError, match="replicate_index"):
            _request(replicate_index=invalid)  # type: ignore[arg-type]


def test_public_seed_replay_and_prior_source_are_independently_bound() -> None:
    original = rank_paired_benchmark_blocks(_request())
    replay = rank_paired_benchmark_blocks(_request())
    assert replay.to_record() == original.to_record()

    changed_seed = rank_paired_benchmark_blocks(_request(public_seed=20_260_716))
    assert tuple(
        value.rank_digest_sha256 for value in changed_seed.ranked_blocks
    ) != tuple(value.rank_digest_sha256 for value in original.ranked_blocks)
    assert changed_seed.full_schedule_sha256 != original.full_schedule_sha256

    changed_source = rank_paired_benchmark_blocks(
        _request(seed_source=_digest("another-prior-public-seed-protocol"))
    )
    assert changed_source.block_schedule == original.block_schedule
    assert changed_source.full_schedule_sha256 != original.full_schedule_sha256
    assert changed_source.selected_block_receipt_sha256 != (
        original.selected_block_receipt_sha256
    )

    for invalid in (-1, 1 << 64, True):
        with pytest.raises(ValueError, match="public_seed"):
            _request(public_seed=invalid)  # type: ignore[arg-type]


def test_closed_request_schema_excludes_treatment_and_runtime_identities() -> None:
    assert tuple(value.name for value in fields(PairedBlockScheduleRequest)) == (
        "benchmark_id",
        "task_sha256",
        "finite_contract_identity_sha256",
        "partition_layout_sha256",
        "policy",
        "eligible_mask",
        "public_seed",
        "public_seed_source_sha256",
        "blocks",
        "replicate_index",
    )
    request = _request()
    assert set(request.schedule_basis_record()) == {
        "schema_version",
        "benchmark_id",
        "task_sha256",
        "finite_contract_identity_sha256",
        "partition_layout_sha256",
        "eligible_mask_sha256",
        "public_seed",
        "block_count",
    }
    banned = {
        "method",
        "treatment",
        "provider",
        "prompt",
        "forecast",
        "health",
        "outcome",
    }
    assert not any(
        token in field.name
        for field in fields(PairedBlockScheduleRequest)
        for token in banned
    )
    with pytest.raises(TypeError):
        PairedBlockScheduleRequest(  # type: ignore[call-arg]
            **{
                field.name: getattr(request, field.name)
                for field in fields(PairedBlockScheduleRequest)
            },
            treatment_id="forbidden",
        )


def test_policy_identity_and_all_scientific_inputs_are_receipt_bound() -> None:
    request = _request()
    original = rank_paired_benchmark_blocks(request)

    changed_policy = replace(
        request.policy,
        policy_definition_sha256=_digest("changed-policy-definition"),
    )
    with pytest.raises(ValueError, match="identified scheduler"):
        rank_paired_benchmark_blocks(replace(request, policy=changed_policy))

    assert rank_paired_benchmark_blocks(
        replace(request, task_sha256=_digest("changed-task"))
    ).full_schedule_sha256 != original.full_schedule_sha256
    assert rank_paired_benchmark_blocks(
        replace(request, partition_layout_sha256=_digest("changed-layout"))
    ).full_schedule_sha256 != original.full_schedule_sha256

    changed_contract = _digest("changed-contract")
    contract_result = rank_paired_benchmark_blocks(
        replace(
            request,
            finite_contract_identity_sha256=changed_contract,
            eligible_mask=ExactEligibleRowMask(
                finite_contract_identity_sha256=changed_contract,
                eligible_global_row_indices=(0, 1, 2),
            ),
        )
    )
    assert contract_result.full_schedule_sha256 != original.full_schedule_sha256

    mask_result = rank_paired_benchmark_blocks(
        replace(
            request,
            eligible_mask=ExactEligibleRowMask(
                finite_contract_identity_sha256=(
                    request.finite_contract_identity_sha256
                ),
                eligible_global_row_indices=tuple(range(19)),
            ),
        )
    )
    assert mask_result.full_schedule_sha256 != original.full_schedule_sha256

    changed_blocks = list(request.blocks)
    changed_blocks[0] = replace(
        changed_blocks[0],
        block_spec_sha256=_digest("changed-block-spec"),
    )
    block_result = rank_paired_benchmark_blocks(
        replace(request, blocks=tuple(changed_blocks))
    )
    assert block_result.full_schedule_sha256 != original.full_schedule_sha256

    range_blocks = list(request.blocks)
    range_blocks[0] = replace(range_blocks[0], global_row_stop=4)
    range_blocks[1] = replace(range_blocks[1], global_row_start=4)
    range_result = rank_paired_benchmark_blocks(
        replace(request, blocks=tuple(range_blocks))
    )
    assert range_result.full_schedule_sha256 != original.full_schedule_sha256

    replicate_result = rank_paired_benchmark_blocks(
        replace(request, replicate_index=1)
    )
    assert replicate_result.full_schedule_sha256 == original.full_schedule_sha256
    assert replicate_result.selected_block_receipt_sha256 != (
        original.selected_block_receipt_sha256
    )


def test_exact_mask_and_partition_validation_fail_closed() -> None:
    contract = _digest("contract")
    with pytest.raises(ValueError, match="unique and canonically sorted"):
        ExactEligibleRowMask(contract, (0, 2, 1))
    with pytest.raises(ValueError, match="unique and canonically sorted"):
        ExactEligibleRowMask(contract, (0, 1, 1))
    with pytest.raises(ValueError, match="outside the block partition"):
        _request(eligible_rows=(0, 20))

    broken = list(_blocks())
    broken[1] = replace(broken[1], global_row_start=6)
    with pytest.raises(ValueError, match="gap-free"):
        _request(blocks=tuple(broken))
    with pytest.raises(ValueError, match="contiguous canonical positions"):
        _request(blocks=(_blocks()[0], _blocks()[2]))


def test_serialized_and_object_tampering_is_rejected_by_public_replay() -> None:
    request = _request()
    result = rank_paired_benchmark_blocks(request)
    assert verify_paired_block_schedule(request, result) == result
    assert verify_paired_block_schedule(request, result.to_record()) == result

    forged = deepcopy(result.to_record())
    forged["selected_block_receipt_sha256"] = _digest("forged-selection")
    with pytest.raises(PairedBlockScheduleVerificationError):
        verify_paired_block_schedule(request, forged)

    forged = deepcopy(result.to_record())
    forged["full_schedule"]["ranked_blocks"][0]["rank_digest_sha256"] = (
        _digest("forged-rank")
    )
    with pytest.raises(PairedBlockScheduleVerificationError):
        verify_paired_block_schedule(request, forged)
