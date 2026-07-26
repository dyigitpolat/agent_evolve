#!/usr/bin/env python3
"""Release exact-set ranks for the finalized fresh Airfoil paired trial.

This is a separate analysis extension.  It never writes into the frozen live
directory, dispatches no provider or CFD work, and decodes exactly the 19
cached eligible terminal records only after the v2/v3 commits and selected
union have been recursively verified and a rank prerelease has been fsynced.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
SRC_ROOT = AGENT_EVOLVE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.application.postcommit_rank_authority import (  # noqa: E402
    PostcommitRankOnlyAuthority,
    validate_postcommit_rank_release_bindings,
)
from agent_evolve.ports.artifact_store import (  # noqa: E402
    canonical_json_bytes,
    decode_json_bytes,
)
from agent_evolve.ports.postcommit_rank_authority import (  # noqa: E402
    PostcommitRankAuthorization,
    PostcommitRankRequest,
    RankReferenceObservation,
)
from examples.development import (  # noqa: E402
    airfoil_v7_postcommit_rank_authority as adapter,
)
from examples.development import (  # noqa: E402
    airfoil_v7_two_stage_agent_evolution as airfoil,
)
from examples.development import (  # noqa: E402
    run_airfoil_v7_v5_paired_causal_trial as trial,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    file_identity,
    finalize_run_directory,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)


ARTIFACT_ROOT = (
    WORKSPACE_ROOT / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
)
PAIRED_ROOT = (
    ARTIFACT_ROOT
    / "experiment_logs"
    / "airfoil_v7"
    / "paired_causal_v1"
)
DEFAULT_SOURCE_LIVE_RUN = (
    PAIRED_ROOT / "ae7_v5_paired_causal_live_20260715T1611SGT"
)
DEFAULT_ORACLE_DIR = airfoil.DEFAULT_SEALED_ORACLE_DIR
DEFAULT_ANALYSIS_ROOT = PAIRED_ROOT / "analysis"
COMPLETED_SOURCE_STATUS = "completed_paired_selected_union_primary_endpoint"
COMPLETED_EXTENSION_STATUS = "completed_postcommit_rank_only_release"
TEST_EXTENSION_STATUS = "provider_free_injected_test_rank_release"
_PRERELEASE_DOMAIN = b"agent-evolve:airfoil-v7-rank-prerelease:v1\x00"


class AirfoilRankExtensionError(RuntimeError):
    """The source run or rank-only extension escaped its frozen contract."""


def _load_object(path: Path) -> dict[str, object]:
    value = decode_json_bytes(path.expanduser().resolve(strict=True).read_bytes())
    if type(value) is not dict:
        raise AirfoilRankExtensionError(f"{path.name} must contain one object")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.expanduser().resolve(strict=True).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedPairedSource:
    run_dir: Path
    finalization: Mapping[str, object]
    result: Mapping[str, object]
    paired_comparison: Mapping[str, object]
    airfoil_commitment: Mapping[str, object]
    selected_union: Mapping[str, object]
    v2_commit: Mapping[str, object]
    v3_commit: Mapping[str, object]
    selected_union_file_sha256: str


def verify_paired_source(run_dir: Path) -> VerifiedPairedSource:
    """Authenticate the frozen source and its full commit-before-union seam."""

    root = run_dir.expanduser().resolve(strict=True)
    finalization = verify_finalized_run_directory(root)
    if finalization.get("status") != COMPLETED_SOURCE_STATUS:
        raise AirfoilRankExtensionError("source run is not the completed paired release")
    result = _load_object(root / "result.json")
    # Reuse the source harness's exact cross-file validator in addition to the
    # recursive seal; this verifies both phase commits and selected-union order.
    trial._validate_release_completion(root=root, result=result)
    paired = _load_object(root / "paired_allocation_comparison_commitment.json")
    benchmark = _load_object(root / "airfoil_paired_allocation_commitment.json")
    selected_union = _load_object(root / "selected_union_outcomes.json")
    v2_commit = _load_object(root / "durable_allocation_v2_commit.json")
    v3_commit = _load_object(root / "durable_allocation_v3_commit.json")
    files = finalization.get("files")
    if type(files) is not dict:
        raise AirfoilRankExtensionError("source finalization lacks file bindings")
    selected_binding = files.get("selected_union_outcomes.json")
    if type(selected_binding) is not dict:
        raise AirfoilRankExtensionError("source lacks selected-union binding")
    selected_sha256 = selected_binding.get("sha256")
    if (
        type(selected_sha256) is not str
        or selected_sha256 != _sha256_file(root / "selected_union_outcomes.json")
        or selected_union.get("raw_outcome_authority")
        != "committed_selected_union_only"
        or selected_union.get("unselected_outcomes_exposed") is not False
        or selected_union.get("unique_cached_read_count") != 6
    ):
        raise AirfoilRankExtensionError("source selected-union release changed")
    return VerifiedPairedSource(
        run_dir=root,
        finalization=finalization,
        result=result,
        paired_comparison=paired,
        airfoil_commitment=benchmark,
        selected_union=selected_union,
        v2_commit=v2_commit,
        v3_commit=v3_commit,
        selected_union_file_sha256=selected_sha256,
    )


def _source_paths() -> tuple[Path, ...]:
    paths = set(trial._source_paths())
    paths.update(
        (
            AGENT_EVOLVE_ROOT
            / "src"
            / "agent_evolve"
            / "ports"
            / "postcommit_rank_authority.py",
            AGENT_EVOLVE_ROOT
            / "src"
            / "agent_evolve"
            / "application"
            / "postcommit_rank_authority.py",
            AGENT_EVOLVE_ROOT
            / "examples"
            / "development"
            / "airfoil_v7_postcommit_rank_authority.py",
            AGENT_EVOLVE_ROOT
            / "examples"
            / "development"
            / "airfoil_v7_two_stage_agent_evolution.py",
            AGENT_EVOLVE_ROOT
            / "examples"
            / "development"
            / "run_airfoil_v7_v5_paired_causal_trial.py",
            Path(__file__),
        )
    )
    test_path = (
        AGENT_EVOLVE_ROOT
        / "tests"
        / "test_airfoil_v7_postcommit_rank_authority.py"
    )
    if test_path.is_file():
        paths.add(test_path)
    return tuple(sorted(paths, key=lambda value: value.resolve().as_posix()))


def _prerelease_record(
    *,
    request: PostcommitRankRequest,
    source: VerifiedPairedSource,
    source_code: Mapping[str, object],
    release_mode: bool,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": 1,
        "status": "authorized_before_reference_outcome_decode",
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": request.to_record(),
        "source_live_run_dir": str(source.run_dir),
        "source_run_finalization_sha256": source.finalization[
            "finalization_sha256"
        ],
        "both_method_commits_recursively_verified": True,
        "paired_comparison_commitment_verified": True,
        "selected_union_release_recursively_verified": True,
        "selected_union_release_sha256": source.selected_union_file_sha256,
        "source_code_identity": dict(source_code),
        "eligible_cached_terminal_decode_count_before_prerelease": 0,
        "raw_unselected_outcomes_read_before_prerelease": False,
        "provider_calls_before_prerelease": 0,
        "new_cfd_calls_before_prerelease": 0,
        "authority_scope": adapter.AUTHORIZATION_SCOPE,
        "authoritative_scientific_release": release_mode,
        "chronology_boundary": (
            "digest alone does not prove chronology; composing harness fsync and "
            "exact read-back are required before reader construction"
        ),
    }
    record["prerelease_record_sha256"] = hashlib.sha256(
        _PRERELEASE_DOMAIN + canonical_json_bytes(record)
    ).hexdigest()
    return record


ReaderFactory = Callable[
    [PostcommitRankRequest, PostcommitRankAuthorization, object],
    Callable[[str], RankReferenceObservation],
]


def _production_reader_factory(
    request: PostcommitRankRequest,
    authorization: PostcommitRankAuthorization,
    oracle: object,
) -> adapter.AirfoilRankReferenceReader:
    if type(oracle) is not airfoil.VerifiedAirfoilPredecisionOracle:
        raise TypeError("oracle must be exact")
    return adapter.AirfoilRankReferenceReader(
        request=request,
        authorization=authorization,
        oracle=oracle,
    )


_PRODUCTION_READER_FACTORY = _production_reader_factory
_PRODUCTION_READER_TYPE = adapter.AirfoilRankReferenceReader
_PRODUCTION_READER_CALL = adapter.AirfoilRankReferenceReader.__call__
_PRODUCTION_READER_AUDIT = adapter.AirfoilRankReferenceReader.audit_record


def _walk_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []
    if type(value) is dict:
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(child))
    elif type(value) is list:
        for child in value:
            keys.extend(_walk_keys(child))
    return tuple(keys)


def _validate_extension_completion(
    *,
    root: Path,
    source: VerifiedPairedSource,
    request: PostcommitRankRequest,
    authorization: PostcommitRankAuthorization,
    prerelease: Mapping[str, object],
    release_record: Mapping[str, object],
    execution_audit: Mapping[str, object],
    result: Mapping[str, object],
    source_code: Mapping[str, object],
    release_mode: bool,
) -> None:
    """Cross-bind every public record before recursive finalization."""

    expected_status = (
        COMPLETED_EXTENSION_STATUS if release_mode else TEST_EXTENSION_STATUS
    )
    if result.get("status") != expected_status or execution_audit.get(
        "status"
    ) != expected_status:
        raise AirfoilRankExtensionError("rank extension completion status changed")
    if _load_object(root / "rank_request.json") != request.to_record():
        raise AirfoilRankExtensionError("durable rank request differs")
    if _load_object(root / "rank_authority_prerelease.json") != dict(prerelease):
        raise AirfoilRankExtensionError("durable rank prerelease differs")
    if (
        authorization.prerelease_file_sha256
        != _sha256_file(root / "rank_authority_prerelease.json")
        or _load_object(root / "rank_authorization.json")
        != authorization.to_record()
    ):
        raise AirfoilRankExtensionError("durable rank authorization differs")
    if _load_object(root / "rank_only_release.json") != dict(release_record):
        raise AirfoilRankExtensionError("durable rank release differs")
    if _load_object(root / "execution_audit.json") != dict(execution_audit):
        raise AirfoilRankExtensionError("durable execution audit differs")
    if _load_object(root / "result.json") != dict(result):
        raise AirfoilRankExtensionError("durable rank result differs")
    if source_identity(_source_paths(), relative_to=WORKSPACE_ROOT) != source_code:
        raise AirfoilRankExtensionError("rank authority source changed during execution")
    source_replay = verify_paired_source(source.run_dir)
    if source_replay.finalization != source.finalization:
        raise AirfoilRankExtensionError("source live run changed during extension")

    reader = execution_audit.get("reader")
    release_sha256 = release_record.get("release_sha256")
    selected_ranks = release_record.get("selected_ranks")
    if (
        type(reader) is not dict
        or reader.get("exact_eligible_cached_terminal_decode_count") != 19
        or reader.get("raw_unselected_outcomes_returned") is not False
        or reader.get("new_cfd_calls") != 0
        or reader.get("provider_calls") != 0
        or release_record.get("status")
        != "completed_postcommit_rank_only_release"
        or release_record.get("request_sha256") != request.request_sha256
        or release_record.get("authorization_sha256")
        != authorization.authorization_sha256
        or release_record.get("exact_reference_read_count") != 19
        or release_record.get("exact_portfolio_count") != 969
        or release_record.get("raw_reference_values_returned") is not False
        or release_record.get("unselected_item_values_returned") is not False
        or release_record.get("unselected_portfolio_endpoints_returned") is not False
        or release_record.get("provider_calls") != 0
        or release_record.get("new_candidate_evaluations") != 0
        or execution_audit.get("rank_release_sha256") != release_sha256
        or result.get("rank_release_sha256") != release_sha256
        or result.get("rank_request_sha256") != request.request_sha256
        or result.get("rank_authorization_sha256")
        != authorization.authorization_sha256
        or result.get("exact_reference_cached_read_count") != 19
        or result.get("exact_portfolio_denominator") != 969
        or result.get("selected_ranks") != selected_ranks
        or result.get("raw_unselected_outcomes_returned") is not False
        or result.get("new_provider_calls") != 0
        or result.get("new_cfd_calls") != 0
    ):
        raise AirfoilRankExtensionError("rank release/audit/result bindings disagree")
    banned_public_keys = {
        "endpoint_component",
        "metric_deltas",
        "metrics",
        "delta_f",
        "delta_v",
        "raw_receipt",
        "unique_evaluations",
        "unselected_outcomes",
    }
    public_keys = set(
        _walk_keys(
            {
                "release": dict(release_record),
                "audit": dict(execution_audit),
                "result": dict(result),
            }
        )
    )
    if public_keys.intersection(banned_public_keys):
        raise AirfoilRankExtensionError("public rank records expose reference values")
    if release_mode and (
        type(reader.get("terminal_receipt_sequence_sha256")) is not str
        or result.get("status") != COMPLETED_EXTENSION_STATUS
    ):
        raise AirfoilRankExtensionError("authoritative reader receipt is absent")


def _execute_rank_extension_common(
    *,
    source_live_run: Path,
    oracle_dir: Path,
    run_dir: Path,
    reader_factory: ReaderFactory,
    release_mode: bool,
) -> dict[str, object]:
    """Execute and recursively finalize one separate rank-only extension."""

    if release_mode and reader_factory is not _PRODUCTION_READER_FACTORY:
        raise AirfoilRankExtensionError(
            "authoritative release requires the exact production reader factory"
        )
    source = verify_paired_source(source_live_run)
    target = run_dir.expanduser().resolve(strict=False)
    if target.exists():
        raise FileExistsError(target)
    if target == source.run_dir or source.run_dir in target.parents:
        raise AirfoilRankExtensionError("rank extension cannot live inside source run")
    target.mkdir(parents=True, exist_ok=False)

    # Structural oracle verification authenticates all sealed bytes but decodes
    # no outcome-bearing terminal.  Exact terminal decoding begins only below.
    oracle = airfoil.verify_airfoil_v7_predecision_oracle(oracle_dir)
    request = adapter.build_rank_request(
        source_finalization=source.finalization,
        paired_comparison=source.paired_comparison,
        airfoil_commitment=source.airfoil_commitment,
        selected_union_release_sha256=source.selected_union_file_sha256,
        v2_commit=source.v2_commit,
        v3_commit=source.v3_commit,
        oracle=oracle,
    )
    source_code = source_identity(_source_paths(), relative_to=WORKSPACE_ROOT)
    write_json_atomic(target / "rank_request.json", request.to_record())
    prerelease = _prerelease_record(
        request=request,
        source=source,
        source_code=source_code,
        release_mode=release_mode,
    )
    prerelease_path = target / "rank_authority_prerelease.json"
    write_json_atomic(prerelease_path, prerelease)
    if _load_object(prerelease_path) != prerelease:
        raise AirfoilRankExtensionError("rank prerelease read-back changed")
    authorization = PostcommitRankAuthorization(
        request_sha256=request.request_sha256,
        prerelease_file_sha256=_sha256_file(prerelease_path),
        authorization_scope=adapter.AUTHORIZATION_SCOPE,
    )
    write_json_atomic(target / "rank_authorization.json", authorization.to_record())
    if _load_object(target / "rank_authorization.json") != authorization.to_record():
        raise AirfoilRankExtensionError("rank authorization read-back changed")

    # The factory is deliberately invoked only after both durable read-backs.
    reader = reader_factory(request, authorization, oracle)
    if release_mode and (
        type(reader) is not _PRODUCTION_READER_TYPE
        or type(reader).__call__ is not _PRODUCTION_READER_CALL
        or type(reader).audit_record is not _PRODUCTION_READER_AUDIT
    ):
        raise AirfoilRankExtensionError(
            "authoritative release requires the exact production reader type"
        )
    authority = PostcommitRankOnlyAuthority(
        request=request,
        authorization=authorization,
        reader=reader,
    )
    release = authority.release()
    validate_postcommit_rank_release_bindings(
        request=request,
        authorization=authorization,
        release=release,
    )
    reader_audit_method = getattr(reader, "audit_record", None)
    if not callable(reader_audit_method):
        raise AirfoilRankExtensionError("Airfoil reader lacks exact audit record")
    reader_audit = reader_audit_method()
    if (
        type(reader_audit) is not dict
        or reader_audit.get("exact_eligible_cached_terminal_decode_count") != 19
        or reader_audit.get("raw_unselected_outcomes_returned") is not False
    ):
        raise AirfoilRankExtensionError("rank reader cardinality or leakage changed")

    release_record = release.to_record()
    write_json_atomic(target / "rank_only_release.json", release_record)
    if _load_object(target / "rank_only_release.json") != release_record:
        raise AirfoilRankExtensionError("rank-only release read-back changed")
    completion_status = (
        COMPLETED_EXTENSION_STATUS if release_mode else TEST_EXTENSION_STATUS
    )
    execution_audit = {
        "schema_version": 1,
        "status": completion_status,
        "rank_release_sha256": release.release_sha256,
        "request_sha256": request.request_sha256,
        "authorization_sha256": authorization.authorization_sha256,
        "source_live_finalization_sha256": source.finalization[
            "finalization_sha256"
        ],
        "reader": reader_audit,
        "in_memory_exact_portfolio_enumeration_count": (
            release.exact_portfolio_count
        ),
        "eligible_cached_terminal_decode_count": (
            release.exact_reference_read_count
        ),
        "selected_union_cached_terminal_decode_count_in_source_run": 6,
        "raw_unselected_outcomes_returned": False,
        "unselected_portfolio_endpoints_returned": False,
        "new_provider_calls": 0,
        "new_cfd_calls": 0,
        "source_live_directory_mutated": False,
        "authoritative_scientific_release": release_mode,
    }
    write_json_atomic(target / "execution_audit.json", execution_audit)
    result = {
        "schema_version": 1,
        "status": completion_status,
        "source_live_run_dir": str(source.run_dir),
        "source_live_finalization_sha256": source.finalization[
            "finalization_sha256"
        ],
        "rank_request_sha256": request.request_sha256,
        "rank_authorization_sha256": authorization.authorization_sha256,
        "rank_release_sha256": release.release_sha256,
        "exact_reference_cached_read_count": release.exact_reference_read_count,
        "exact_portfolio_denominator": release.exact_portfolio_count,
        "selected_ranks": [value.to_record() for value in release.selected_ranks],
        "aggregate_diagnostics": release_record["aggregate_diagnostics"],
        "raw_unselected_outcomes_returned": False,
        "new_provider_calls": 0,
        "new_cfd_calls": 0,
        "authoritative_scientific_release": release_mode,
    }
    write_json_atomic(target / "result.json", result)
    _validate_extension_completion(
        root=target,
        source=source,
        request=request,
        authorization=authorization,
        prerelease=prerelease,
        release_record=release_record,
        execution_audit=execution_audit,
        result=result,
        source_code=source_code,
        release_mode=release_mode,
    )
    finalization = finalize_run_directory(
        target,
        status=completion_status,
    )
    verified = verify_finalized_run_directory(target)
    if verified != finalization:
        raise AirfoilRankExtensionError("rank extension finalization replay changed")
    return {
        "result": result,
        "release": release_record,
        "finalization": finalization,
        "run_dir": str(target),
    }


def execute_rank_extension(
    *,
    source_live_run: Path,
    oracle_dir: Path,
    run_dir: Path,
) -> dict[str, object]:
    """Authoritative entry: dependency injection is intentionally unavailable."""

    return _execute_rank_extension_common(
        source_live_run=source_live_run,
        oracle_dir=oracle_dir,
        run_dir=run_dir,
        reader_factory=_PRODUCTION_READER_FACTORY,
        release_mode=True,
    )


def _execute_rank_extension_with_reader_for_test(
    *,
    source_live_run: Path,
    oracle_dir: Path,
    run_dir: Path,
    reader_factory: ReaderFactory,
) -> dict[str, object]:
    """Injected seam that can never finalize an authoritative success status."""

    return _execute_rank_extension_common(
        source_live_run=source_live_run,
        oracle_dir=oracle_dir,
        run_dir=run_dir,
        reader_factory=reader_factory,
        release_mode=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-live-run", type=Path, default=DEFAULT_SOURCE_LIVE_RUN)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    execution = execute_rank_extension(
        source_live_run=arguments.source_live_run,
        oracle_dir=arguments.oracle_dir,
        run_dir=arguments.run_dir,
    )
    return 0 if execution["result"]["status"] == COMPLETED_EXTENSION_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
