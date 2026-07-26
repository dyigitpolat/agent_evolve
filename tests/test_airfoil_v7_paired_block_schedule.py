from __future__ import annotations

from pathlib import Path
import sys

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from examples.development import airfoil_v7_two_stage_agent_evolution as legacy
from examples.development.airfoil_v7_paired_block_schedule import (
    prepare_airfoil_v7_paired_block_assignment,
)


EXPECTED_PAYLOAD_SHA256 = (
    "bebab3e13dc274b6806aa38a5a48a9c8182635bdaa2cbac503d1fab93216e964"
)
EXPECTED_ASSIGNMENT_RECEIPT_SHA256 = (
    "504c9f1a009c93deecaa398b344ff795d71203ab750a5d84b8debdaeec3c5807"
)
EXPECTED_LAYOUT_SHA256 = (
    "fd823ec5d0ba9505bd110c2c1ed30c0982c5a9d5f05b422f27908838e2bf4abe"
)
EXPECTED_MASK_PAYLOAD_SHA256 = (
    "21c3612457c00a25e406efe02020e3b50b7337e46e48541de59357fde0a56e6d"
)
EXPECTED_MASK_SHA256 = (
    "17a858293c39f6025feace7157414099b4379b8dbcd91c54ad8f81d598941d97"
)
EXPECTED_RANKS = (
    (
        2,
        "5cd6dd0415b5a72a9cc41907c6b92164d4b54e8c9a63d96599737c8a63c534bf",
    ),
    (
        0,
        "bc845aa3fad7f66d62476153e3caa5ea3659da638842052ad259e7bc6e403920",
    ),
    (
        3,
        "c40a2e6efcaa056b99d8ebf8cd07e4a171b0f3c8d870f58f8ef436084dc8bd53",
    ),
    (
        1,
        "d1ca891a747b23fdb87d2f274fa73455afb5fa4de18ac0331db6ee0b20d12f15",
    ),
)
EXPECTED_BLOCK2_OPTION_IDS = (
    "trim.p025.p025.p050",
    "trim.p050.p050.n050",
    "trim.n025.n025.p050",
    "trim.p050.p050.p050",
    "shape.thickness_aft.p0015",
    "trim.n025.n050.n025",
    "trim.p050.n025.p050",
    "trim.n050.p025.p025",
    "trim.n025.p050.p025",
    "trim.p025.p050.p025",
    "trim.n050.p050.n025",
    "trim.n025.p050.n050",
    "trim.p025.n050.p050",
    "trim.p025.n050.n050",
    "trim.p050.n050.p050",
    "trim.p050.n025.n025",
    "trim.p050.p025.n050",
    "shape.thickness_aft.n0015",
    "trim.n025.n025.p025",
)


def test_airfoil_default_assignment_reproduces_the_frozen_v2_proposal() -> None:
    assignment = prepare_airfoil_v7_paired_block_assignment()
    request = assignment.schedule.request
    payload = assignment.payload_record()

    assert assignment.layout.layout_sha256 == EXPECTED_LAYOUT_SHA256
    assert request.eligible_mask.payload_sha256 == EXPECTED_MASK_PAYLOAD_SHA256
    assert request.eligible_mask.mask_sha256 == EXPECTED_MASK_SHA256
    assert assignment.schedule.block_schedule == (2, 0, 3, 1)
    assert tuple(
        (value.block.block_index, value.rank_digest_sha256)
        for value in assignment.schedule.ranked_blocks
    ) == EXPECTED_RANKS
    assert assignment.payload_sha256 == EXPECTED_PAYLOAD_SHA256
    assert assignment.assignment_receipt_sha256 == (
        EXPECTED_ASSIGNMENT_RECEIPT_SHA256
    )
    assert payload["block_rank_digests"] == [
        {"block_index": block_index, "rank_digest_sha256": digest}
        for block_index, digest in EXPECTED_RANKS
    ]


def test_airfoil_block2_exact_mask_excludes_g1_without_outcome_access() -> None:
    assignment = prepare_airfoil_v7_paired_block_assignment()
    rows_by_id = {
        option.option_id: index
        for index, option in enumerate(assignment.contract.options)
    }
    assert sorted(
        rows_by_id[member.option_id] for member in assignment.g1_sample.members
    ) == [1, 7, 11, 30, 37, 42, 71, 79]
    assert assignment.eligible_global_row_indices == (
        40,
        41,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
    )
    assert assignment.excluded_g1_global_row_indices == (42,)
    payload = assignment.payload_record()
    assert payload["selected_block"] == {
        "block_index": 2,
        "block_spec_sha256": (
            "e74beed09b645c4ef76b8c5c102d986975ce9386c7b5dcd840db995ac9f6c446"
        ),
        "global_row_start": 40,
        "global_row_stop": 60,
    }
    assert payload["common_g2_subset"] == {
        "count": 19,
        "global_row_indices": list(assignment.eligible_global_row_indices),
        "option_ids": list(EXPECTED_BLOCK2_OPTION_IDS),
        "excluded_g1_global_row_indices": [42],
        "excluded_g1_option_ids": ["shape.camber_aft.p0015"],
    }
    assert payload["derived_budget"] == {
        "portfolio_size": 3,
        "candidate_scores_per_arm": 54,
        "candidate_scores_total_mpn": 162,
        "logical_evaluator_slots_per_arm": 3,
        "logical_evaluator_slots_total": 9,
        "unique_cached_reads_min": 3,
        "unique_cached_reads_max": 9,
        "postdecision_exact_three_set_count": 969,
    }
    assert assignment.schedule.request.public_seed_source_sha256 == (
        assignment.g1_sample.receipt_sha256
    )


def test_airfoil_schedule_balances_every_block_and_derives_each_exact_budget() -> None:
    assignments = tuple(
        prepare_airfoil_v7_paired_block_assignment(replicate_index=index)
        for index in range(4)
    )
    assert tuple(value.selected_block.block_index for value in assignments) == (
        2,
        0,
        3,
        1,
    )
    assert tuple(len(value.eligible_global_row_indices) for value in assignments) == (
        19,
        17,
        18,
        18,
    )
    assert len({value.schedule.full_schedule_sha256 for value in assignments}) == 1
    assert len(
        {value.schedule.selected_block_receipt_sha256 for value in assignments}
    ) == 4
    with pytest.raises(ValueError, match="replicate_index"):
        prepare_airfoil_v7_paired_block_assignment(replicate_index=4)


def test_airfoil_adapter_does_not_open_or_decode_any_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("outcome-bearing path was accessed")

    monkeypatch.setattr(legacy, "prepare_airfoil_v7_two_stage_generation", forbidden)
    monkeypatch.setattr(legacy, "verify_airfoil_v7_predecision_oracle", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assignment = prepare_airfoil_v7_paired_block_assignment()
    record = assignment.to_record()
    assert record["outcomes_read"] is False
    assert record["provider_calls"] == 0
    assert record["credentials_read"] is False
    assert record["generic_schedule"]["request"]["policy"] == (
        assignment.schedule.request.policy.to_record()
    )
    assert record["generic_schedule"]["request"][
        "public_seed_source_sha256"
    ] == assignment.g1_sample.receipt_sha256
