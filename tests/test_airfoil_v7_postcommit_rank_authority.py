from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from agent_evolve.ports.postcommit_rank_authority import RankReferenceObservation
from examples.development import (
    run_airfoil_v7_v5_postcommit_rank_authority as runner,
)
from examples.development.durable_run_artifacts import (
    finalize_run_directory,
    verify_finalized_run_directory,
    write_json_atomic,
)


class _FakeReader:
    def __init__(self, request, *, target: Path) -> None:
        assert (target / "rank_authority_prerelease.json").is_file()
        assert (target / "rank_authorization.json").is_file()
        self._request = request
        self._seen: list[str] = []

    def __call__(self, item_id: str) -> RankReferenceObservation:
        self._seen.append(item_id)
        return RankReferenceObservation(
            item_id=item_id,
            endpoint_component=float(len(self._seen)),
            source_receipt_sha256=runner.hashlib.sha256(
                f"fake-{item_id}".encode("ascii")
            ).hexdigest(),
        )

    def audit_record(self):
        return {
            "schema_version": 1,
            "exact_eligible_cached_terminal_decode_count": len(self._seen),
            "eligible_item_count": len(self._request.eligible_item_ids),
            "terminal_receipt_sequence_sha256": runner.hashlib.sha256(
                b"fake-receipts"
            ).hexdigest(),
            "raw_unselected_outcomes_returned": False,
            "new_cfd_calls": 0,
            "provider_calls": 0,
        }


def test_frozen_live_source_binds_both_methods_and_completed_selected_union() -> None:
    source = runner.verify_paired_source(runner.DEFAULT_SOURCE_LIVE_RUN)
    assert source.finalization["status"] == runner.COMPLETED_SOURCE_STATUS
    assert source.paired_comparison["logical_slot_count"] == 18
    assert source.airfoil_commitment["selected_option_ids"] == [
        "shape.thickness_aft.n0015",
        "trim.p025.p025.p050",
        "trim.p025.p050.p025",
        "trim.p050.n025.p050",
        "trim.p050.n050.p050",
        "trim.p050.p050.p050",
    ]
    assert source.selected_union["unique_cached_read_count"] == 6
    assert source.selected_union["unselected_outcomes_exposed"] is False


def test_injected_reader_cannot_emit_authoritative_completion(tmp_path: Path) -> None:
    target = tmp_path / "test-only-rank-extension"

    def factory(request, _authorization, _oracle):
        return _FakeReader(request, target=target)

    execution = runner._execute_rank_extension_with_reader_for_test(
        source_live_run=runner.DEFAULT_SOURCE_LIVE_RUN,
        oracle_dir=runner.DEFAULT_ORACLE_DIR,
        run_dir=target,
        reader_factory=factory,
    )
    assert execution["result"]["status"] == runner.TEST_EXTENSION_STATUS
    assert execution["result"]["authoritative_scientific_release"] is False
    assert execution["finalization"]["status"] == runner.TEST_EXTENSION_STATUS
    assert verify_finalized_run_directory(target) == execution["finalization"]
    assert execution["release"]["exact_reference_read_count"] == 19
    assert execution["release"]["exact_portfolio_count"] == 969


def test_public_entry_has_no_reader_injection_and_internal_release_rejects_it(
    tmp_path: Path,
) -> None:
    assert "reader_factory" not in inspect.signature(
        runner.execute_rank_extension
    ).parameters

    def factory(request, _authorization, _oracle):
        return _FakeReader(request, target=tmp_path / "must-not-exist")

    with pytest.raises(
        runner.AirfoilRankExtensionError,
        match="exact production reader factory",
    ):
        runner._execute_rank_extension_common(
            source_live_run=runner.DEFAULT_SOURCE_LIVE_RUN,
            oracle_dir=runner.DEFAULT_ORACLE_DIR,
            run_dir=tmp_path / "must-not-exist",
            reader_factory=factory,
            release_mode=True,
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_incomplete_source_is_rejected_before_target_or_reference_reader(
    tmp_path: Path,
) -> None:
    source = tmp_path / "incomplete-source"
    source.mkdir()
    write_json_atomic(source / "result.json", {"status": "incomplete"})
    finalize_run_directory(source, status="incomplete")
    target = tmp_path / "target"
    with pytest.raises(runner.AirfoilRankExtensionError, match="not the completed"):
        runner.execute_rank_extension(
            source_live_run=source,
            oracle_dir=runner.DEFAULT_ORACLE_DIR,
            run_dir=target,
        )
    assert not target.exists()


def test_target_cannot_mutate_or_nest_inside_frozen_source() -> None:
    with pytest.raises(runner.AirfoilRankExtensionError, match="cannot live inside"):
        runner.execute_rank_extension(
            source_live_run=runner.DEFAULT_SOURCE_LIVE_RUN,
            oracle_dir=runner.DEFAULT_ORACLE_DIR,
            run_dir=runner.DEFAULT_SOURCE_LIVE_RUN / "forbidden-analysis",
        )
