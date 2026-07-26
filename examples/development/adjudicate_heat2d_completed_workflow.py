#!/usr/bin/env python3
"""Read-only adjudication of a sealed Heat2D generic-campaign run.

The live Heat harness deliberately finalizes failed run directories.  That is
important when a failure happens after the scientific workflow has already
finished: the immutable journals can then be adjudicated without repeating a
provider call or a PDE evaluation.  This program verifies the sealed input,
reconstructs the scientific result from durable records, and writes one new
JSON artifact *outside* the source run directory.

The adjudicator never opens a source-run file for writing.  A legacy R4
progress row omitted only ``schema_version``; the program reports that defect
and may add the literal value ``1`` to an in-memory projection so the current
public provider-attempt join validator can verify the otherwise exact stream.
The immutable journal is not repaired or relabelled.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
ARTIFACT_ROOT = WORKSPACE_ROOT / "papers/agent_evolve_aaai_2027/research_artifacts"
DEFAULT_RUN_DIR = (
    ARTIFACT_ROOT
    / "experiment_logs/benchmark_q1/engibench_heat2d/generic_campaign"
    / "heat_production_learning_g6_live_deepseek_r4_20260717"
)
DEFAULT_OUTPUT = (
    ARTIFACT_ROOT
    / "analysis/heat_r4_completed_workflow_offline_adjudication_20260717.json"
)


def _preparse_run_dir(argv: list[str]) -> Path:
    """Resolve the input early enough to import its sealed validator sources."""

    value: str | None = None
    for index, argument in enumerate(argv):
        if argument == "--run-dir":
            if index + 1 >= len(argv):
                raise SystemExit("--run-dir requires a path")
            value = argv[index + 1]
        elif argument.startswith("--run-dir="):
            value = argument.partition("=")[2]
    return Path(value).expanduser().resolve(strict=False) if value else DEFAULT_RUN_DIR


VALIDATOR_RUN_DIR = _preparse_run_dir(sys.argv[1:])
SEALED_SOURCE_ROOT = VALIDATOR_RUN_DIR / "source_snapshot"
SEALED_AGENT_EVOLVE_ROOT = SEALED_SOURCE_ROOT / "agent_evolve"
SEALED_AGENT_EVOLVE_SRC = SEALED_AGENT_EVOLVE_ROOT / "src"
if not SEALED_AGENT_EVOLVE_SRC.is_dir():
    raise SystemExit(
        f"sealed AgentEvolve source snapshot is unavailable: {SEALED_AGENT_EVOLVE_SRC}"
    )

# Importing source from an immutable run must not create __pycache__ files there.
sys.dont_write_bytecode = True
sys.path.insert(0, str(SEALED_AGENT_EVOLVE_ROOT))
sys.path.insert(0, str(SEALED_AGENT_EVOLVE_SRC))

from agent_evolve.application.campaign_execution import (  # noqa: E402
    CampaignExecutionEvent,
    CampaignExecutionEventKind,
)
from agent_evolve.application.portfolio_campaign_runtime import (  # noqa: E402
    CampaignPortfolioWavePreparationReceipt,
)
from agent_evolve.domain.typed_json import freeze_json  # noqa: E402
from agent_evolve.integrations.pydantic_ai.outbound_request_manifest import (  # noqa: E402
    validate_openrouter_outbound_request_manifest_record,
)
from agent_evolve.integrations.pydantic_ai.provider_attempt_join import (  # noqa: E402
    build_provider_attempt_terminal_join_receipt,
    validate_structured_generation_outcome_record,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    validate_structured_generation_output_evidence_record,
    validate_structured_generation_request_evidence_record,
)
from agent_evolve.policies.reward.affine_hypervolume import (  # noqa: E402
    AffineHypervolume2DSpec,
    AffineHypervolumeSnapshot2D,
    AffineObjectiveAxis,
)
from agent_evolve.ports.artifact_store import canonical_json_bytes  # noqa: E402
from agent_evolve.ports.structured_generator import (  # noqa: E402
    StructuredStreamChannel,
    StructuredStreamProgress,
    StructuredStreamProgressKind,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    read_jsonl,
    verify_finalized_run_directory,
    write_json_atomic,
)


ADJUDICATION_DOMAIN = b"agent-evolve:heat2d-offline-adjudication:v1\x00"
SOURCE_SET_DOMAIN = b"agent-evolve:source-set:v1\x00"
REPORT_FAILURE_MESSAGE = "'InsightMemoryEntry' object has no attribute 'to_record'"

# Every imported validator below was in the sealed source closure.  Requiring
# byte equality prevents a newer local decoder from silently changing the
# interpretation of an old run.
VALIDATOR_SOURCE_MODULES = (
    (
        "agent_evolve/examples/development/durable_run_artifacts.py",
        "examples.development.durable_run_artifacts",
    ),
    (
        "agent_evolve/src/agent_evolve/application/campaign_execution.py",
        "agent_evolve.application.campaign_execution",
    ),
    (
        "agent_evolve/src/agent_evolve/application/portfolio_campaign_runtime.py",
        "agent_evolve.application.portfolio_campaign_runtime",
    ),
    (
        "agent_evolve/src/agent_evolve/integrations/pydantic_ai/queued_runner.py",
        "agent_evolve.integrations.pydantic_ai.queued_runner",
    ),
    (
        "agent_evolve/src/agent_evolve/integrations/pydantic_ai/provider_attempt_join.py",
        "agent_evolve.integrations.pydantic_ai.provider_attempt_join",
    ),
    (
        "agent_evolve/src/agent_evolve/integrations/pydantic_ai/"
        "outbound_request_manifest.py",
        "agent_evolve.integrations.pydantic_ai.outbound_request_manifest",
    ),
    (
        "agent_evolve/src/agent_evolve/policies/reward/affine_hypervolume.py",
        "agent_evolve.policies.reward.affine_hypervolume",
    ),
    (
        "agent_evolve/src/agent_evolve/ports/structured_generator.py",
        "agent_evolve.ports.structured_generator",
    ),
)


def _object(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise RuntimeError(f"{label} is not an exact JSON object")
    return dict(value)


def _array(value: object, *, label: str) -> list[object]:
    if type(value) is not list:
        raise RuntimeError(f"{label} is not an exact JSON array")
    return list(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _workspace_label(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    try:
        return resolved.relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _observed_records(
    path: Path,
    *,
    record_key: str = "authenticated_record",
) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    previous_observation_ns = -1
    for index, row in enumerate(read_jsonl(path)):
        if set(row) != {"observation", record_key}:
            raise RuntimeError(f"{path.name} row {index} has unexpected wrapper fields")
        observation = _object(row["observation"], label="journal observation")
        if set(observation) != {
            "monotonic_ns_since_execution_start",
            "observed_at_utc",
        }:
            raise RuntimeError(f"{path.name} row {index} has malformed observation")
        monotonic_ns = observation["monotonic_ns_since_execution_start"]
        if type(monotonic_ns) is not int or monotonic_ns < previous_observation_ns:
            raise RuntimeError(f"{path.name} observation time is not monotonic")
        if type(observation["observed_at_utc"]) is not str:
            raise RuntimeError(f"{path.name} observation has no UTC text")
        previous_observation_ns = monotonic_ns
        records.append(_object(row[record_key], label=f"{path.name} record"))
    return tuple(records)


def _source_snapshot_evidence(
    run_dir: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    source_identity = _object(
        manifest.get("source_identity"), label="manifest source identity"
    )
    source_snapshot = _object(
        manifest.get("source_snapshot"), label="manifest source snapshot"
    )
    snapshot_identity = {
        key: value
        for key, value in source_snapshot.items()
        if key != "snapshot_directory"
    }
    if source_identity != snapshot_identity:
        raise RuntimeError("launch source identity differs from snapshot identity")
    raw_files = _array(source_snapshot.get("files"), label="source snapshot files")
    if source_snapshot.get("schema_version") not in (None, 1):
        raise RuntimeError("source snapshot schema version is unsupported")
    if source_snapshot.get("file_count") != len(raw_files) or not raw_files:
        raise RuntimeError("source snapshot file count is invalid")

    snapshot_root = (run_dir / "source_snapshot").resolve(strict=True)
    expected_labels: list[str] = []
    aggregate = hashlib.sha256(SOURCE_SET_DOMAIN)
    source_by_label: dict[str, dict[str, object]] = {}
    for raw in raw_files:
        record = _object(raw, label="source file identity")
        label = record.get("path")
        if (
            type(label) is not str
            or not label
            or Path(label).is_absolute()
            or ".." in Path(label).parts
            or label in source_by_label
        ):
            raise RuntimeError("source snapshot contains an unsafe or duplicate path")
        path = (snapshot_root / label).resolve(strict=True)
        if snapshot_root not in path.parents:
            raise RuntimeError("source snapshot path escaped its root")
        content = path.read_bytes()
        observed = {
            "path": label,
            "size_bytes": len(content),
            "sha256": _sha256_bytes(content),
        }
        if record != observed:
            raise RuntimeError(f"source snapshot file identity changed: {label}")
        label_bytes = label.encode("utf-8", errors="strict")
        aggregate.update(len(label_bytes).to_bytes(8, "big"))
        aggregate.update(label_bytes)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
        expected_labels.append(label)
        source_by_label[label] = record

    actual_labels = sorted(
        path.relative_to(snapshot_root).as_posix()
        for path in snapshot_root.rglob("*")
        if path.is_file()
    )
    if actual_labels != sorted(expected_labels):
        raise RuntimeError("source snapshot directory membership changed")
    if aggregate.hexdigest() != source_snapshot.get("aggregate_sha256"):
        raise RuntimeError("source snapshot aggregate identity is invalid")

    validator_files: list[dict[str, object]] = []
    for label, module_name in VALIDATOR_SOURCE_MODULES:
        expected = source_by_label.get(label)
        if expected is None:
            raise RuntimeError(f"sealed source omitted adjudication validator: {label}")
        module = sys.modules.get(module_name)
        module_file = None if module is None else getattr(module, "__file__", None)
        if type(module_file) is not str:
            raise RuntimeError(
                f"adjudication validator was not imported: {module_name}"
            )
        loaded = Path(module_file).resolve(strict=True)
        sealed = (snapshot_root / label).resolve(strict=True)
        if loaded != sealed:
            raise RuntimeError(
                f"adjudication validator was not loaded from sealed source: {label}"
            )
        observed_sha256 = _sha256_file(loaded)
        if observed_sha256 != expected["sha256"]:
            raise RuntimeError(f"loaded validator differs from sealed source: {label}")
        validator_files.append(
            {
                "path": label,
                "sealed_sha256": expected["sha256"],
                "loaded_sha256": observed_sha256,
                "loaded_from_sealed_snapshot": True,
            }
        )

    return {
        "launch_and_snapshot_identity_exact": True,
        "aggregate_sha256": source_snapshot["aggregate_sha256"],
        "file_count": len(raw_files),
        "snapshot_directory_membership_exact": True,
        "validator_sources_loaded_from_sealed_snapshot": True,
        "validator_files": validator_files,
        "postrun_live_workspace_identity_durably_published": False,
        "postrun_identity_limitation": (
            "The live workflow computed a postrun source identity before report "
            "assembly, but the report projection failed before that identity was "
            "published. The sealed launch snapshot remains complete and exact."
        ),
    }


def _preregistration_evidence(run_dir: Path) -> dict[str, object]:
    identity = _object(
        json.loads((run_dir / "preregistration_identity.json").read_bytes()),
        label="preregistration identity",
    )
    label = identity.get("path")
    if type(label) is not str or Path(label).is_absolute() or ".." in Path(label).parts:
        raise RuntimeError("preregistration path is unsafe")
    path = (WORKSPACE_ROOT / label).resolve(strict=False)
    available = path.is_file()
    matches = (
        available
        and {
            "path": label,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        == identity
    )
    return {
        "identity": identity,
        "current_file_available": available,
        "current_file_matches_sealed_identity": matches,
    }


def _campaign_evidence(
    run_dir: Path,
    manifest: Mapping[str, object],
    preparation: Mapping[str, object],
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    rows = read_jsonl(run_dir / "campaign_events.jsonl")
    events: list[dict[str, object]] = []
    previous_hash: str | None = None
    previous_observation_ns = -1
    preparation_sha256 = preparation.get("preparation_sha256")
    for index, row in enumerate(rows, start=1):
        if set(row) != {"observation", "authenticated_campaign_event"}:
            raise RuntimeError("campaign event wrapper has unexpected fields")
        observation = _object(row["observation"], label="campaign observation")
        observed_ns = observation.get("monotonic_ns_since_execution_start")
        if type(observed_ns) is not int or observed_ns < previous_observation_ns:
            raise RuntimeError("campaign observations are not monotonic")
        previous_observation_ns = observed_ns
        record = _object(row["authenticated_campaign_event"], label="campaign event")
        if record.get("sequence") != index:
            raise RuntimeError("campaign event sequence is not contiguous")
        if record.get("previous_event_sha256") != previous_hash:
            raise RuntimeError("campaign event predecessor chain is invalid")
        if record.get("preparation_sha256") != preparation_sha256:
            raise RuntimeError("campaign event names a different preparation")
        event = CampaignExecutionEvent(
            preparation_sha256=str(record["preparation_sha256"]),
            sequence=int(record["sequence"]),
            kind=CampaignExecutionEventKind(str(record["kind"])),
            previous_event_sha256=record["previous_event_sha256"],
            payload=freeze_json(record["payload"]),
        )
        if event.event_sha256 != record.get("event_sha256"):
            raise RuntimeError("campaign event hash is invalid")
        previous_hash = event.event_sha256
        events.append(record)

    protocol = _object(manifest.get("protocol"), label="manifest protocol")
    generation_count = protocol.get("generations")
    if type(generation_count) is not int or generation_count < 1:
        raise RuntimeError("manifest generation count is invalid")
    expected_kinds = ["execution_started"]
    reflection_generations = set(protocol.get("recombination_generations", []))
    for generation in range(1, generation_count + 1):
        expected_kinds.extend(("archive_utility_frozen", "stage_sealed"))
        if generation in reflection_generations:
            expected_kinds.extend(
                (
                    "reflection_launched",
                    "reflection_completed",
                    "reflection_admitted_for_testing",
                )
            )
    expected_kinds.extend(("execution_finalized", "runtime_cleaned"))
    observed_kinds = [str(event["kind"]) for event in events]
    if observed_kinds != expected_kinds:
        raise RuntimeError("campaign event chronology differs from the frozen schedule")

    stage_events = tuple(event for event in events if event["kind"] == "stage_sealed")
    schedule = _object(preparation.get("schedule"), label="prepared schedule")
    steps = _array(schedule.get("steps"), label="prepared schedule steps")
    if len(stage_events) != generation_count or len(steps) != generation_count:
        raise RuntimeError("sealed stage count differs from the prepared schedule")
    stage_summaries: list[dict[str, object]] = []
    for expected_generation, (event, raw_step) in enumerate(
        zip(stage_events, steps, strict=True), start=1
    ):
        payload = _object(event["payload"], label="stage event payload")
        receipt = _object(payload.get("stage_receipt"), label="stage receipt")
        step = _object(raw_step, label="prepared schedule step")
        if (
            receipt.get("generation") != expected_generation
            or receipt.get("kind") != step.get("kind")
            or receipt.get("candidate_occurrence_count")
            != step.get("planned_candidate_evaluations")
            or receipt.get("unique_evaluation_count")
            != step.get("planned_candidate_evaluations")
        ):
            raise RuntimeError("stage receipt differs from its prepared step")
        counters = _object(payload.get("counters"), label="stage counters")
        stage_summaries.append(
            {
                "generation": expected_generation,
                "kind": receipt["kind"],
                "candidate_occurrences": receipt["candidate_occurrence_count"],
                "unique_evaluations": receipt["unique_evaluation_count"],
                "stage_receipt_sha256": receipt["receipt_sha256"],
                "cumulative_counters": counters,
            }
        )

    finalized_event = events[-2]
    cleaned_event = events[-1]
    finalization_receipt = _object(
        _object(finalized_event["payload"], label="finalized payload").get(
            "finalization_receipt"
        ),
        label="campaign finalization receipt",
    )
    cleanup_receipt = _object(
        _object(cleaned_event["payload"], label="cleanup payload").get(
            "cleanup_receipt"
        ),
        label="campaign cleanup receipt",
    )
    last_stage_counters = stage_summaries[-1]["cumulative_counters"]
    post_last_stage_reflection_count = sum(
        1
        for event in events
        if event["kind"] == "reflection_completed"
        and int(event["sequence"]) > int(stage_events[-1]["sequence"])
    )
    if (
        finalization_receipt.get("status") != "completed"
        or cleanup_receipt.get("released") is not True
        or _object(cleanup_receipt.get("evidence"), label="cleanup evidence").get(
            "runtime_closed"
        )
        is not True
    ):
        raise RuntimeError(
            "campaign did not publish completed finalization and cleanup"
        )

    return (
        {
            "event_count": len(events),
            "event_chain_valid": True,
            "last_event_sha256": previous_hash,
            "observed_kinds": observed_kinds,
            "stage_summaries": stage_summaries,
            "last_stage_seal_counters": last_stage_counters,
            "last_stage_seal_counter_scope": (
                "cumulative through G6 stage_sealed; excludes reflection calls "
                "completed after that seal"
            ),
            "post_last_stage_reflection_count": post_last_stage_reflection_count,
            "campaign_finalization_status": finalization_receipt["status"],
            "campaign_finalization_receipt_sha256": finalization_receipt[
                "receipt_sha256"
            ],
            "cleanup_released": cleanup_receipt["released"],
            "cleanup_receipt_sha256": cleanup_receipt["receipt_sha256"],
            "runtime_closed": True,
            "observed_elapsed_ns_to_cleanup": rows[-1]["observation"][
                "monotonic_ns_since_execution_start"
            ],
        },
        stage_events,
    )


def _engine_evidence(run_dir: Path, *, planned_evaluations: int) -> dict[str, object]:
    records = _observed_records(run_dir / "engine_events.jsonl")
    if [record.get("sequence") for record in records] != list(
        range(1, len(records) + 1)
    ):
        raise RuntimeError("engine event sequence is not contiguous")
    offsets = [record.get("monotonic_offset_ns") for record in records]
    if any(type(value) is not int for value in offsets) or offsets != sorted(offsets):
        raise RuntimeError("engine event offsets are not monotonic")

    counts = Counter(str(record.get("event_type")) for record in records)
    cache_events = [
        record
        for record in records
        if record.get("event_type") == "evaluation_cache_event"
    ]
    seeds = [
        record for record in records if record.get("event_type") == "seed_registered"
    ]
    candidates = [
        record
        for record in records
        if record.get("event_type") == "candidate_evaluated"
    ]
    detailed = [
        record
        for record in records
        if record.get("event_type") == "detailed_evaluation_completed"
    ]
    prepared = [
        record
        for record in records
        if record.get("event_type") == "invocation_prepared"
    ]
    completed = [
        record
        for record in records
        if record.get("event_type") == "invocation_completed"
    ]
    candidate_ids = [
        str(record.get("candidate_id")) for record in (*seeds, *candidates)
    ]
    if (
        len(cache_events) != planned_evaluations
        or any(record.get("cache_event_type") != "miss" for record in cache_events)
        or len(detailed) != planned_evaluations
        or len(candidate_ids) != planned_evaluations
        or len(set(candidate_ids)) != planned_evaluations
        or len(seeds) != 2
        or len(prepared) != len(candidates)
        or len(completed) != len(candidates)
        or {record.get("operator_invocation_id") for record in prepared}
        != {record.get("operator_invocation_id") for record in completed}
    ):
        raise RuntimeError("engine event accounting is incomplete")
    if any(
        record.get("valid") is not True or record.get("failure") is not None
        for record in (*seeds, *candidates)
    ):
        raise RuntimeError("engine history contains a failed or invalid candidate")
    if any(
        record.get("operator_compliant") is not True
        or record.get("evidence_compliant") is not True
        for record in candidates
    ):
        raise RuntimeError("engine history contains noncompliant candidates")
    if any(
        _object(record.get("detailed_evaluation"), label="detailed evaluation").get(
            "failure"
        )
        is not None
        for record in detailed
    ):
        raise RuntimeError("detailed evaluation history contains a failure")

    detailed_wall = [
        float(
            _object(
                _object(record["detailed_evaluation"], label="detailed evaluation").get(
                    "timings"
                ),
                label="detailed evaluation timings",
            )["total_wall_seconds"]
        )
        for record in detailed
    ]
    return {
        "event_count": len(records),
        "event_sequence_valid": True,
        "event_type_counts": dict(sorted(counts.items())),
        "candidate_count": len(candidate_ids),
        "unique_candidate_ids": len(set(candidate_ids)),
        "all_evaluations_were_cache_misses": True,
        "all_candidates_valid_and_compliant": True,
        "all_detailed_evaluations_succeeded": True,
        "detailed_evaluation_total_wall_s": {
            "min": min(detailed_wall),
            "median": statistics.median(detailed_wall),
            "mean": statistics.fmean(detailed_wall),
            "max": max(detailed_wall),
            "sum": sum(detailed_wall),
        },
    }


def _validate_legacy_progress_record(record: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "call_id",
        "provider_attempt_id",
        "sequence",
        "kind",
        "channel",
        "elapsed_ns",
        "event_content_utf8_bytes",
        "cumulative_content_utf8_bytes",
        "rolling_content_sha256",
    }
    if set(record) != expected:
        raise RuntimeError("legacy progress row differs from the closed R4 shape")
    text_fields = ("call_id", "provider_attempt_id", "kind", "channel")
    integer_fields = (
        "sequence",
        "elapsed_ns",
        "event_content_utf8_bytes",
        "cumulative_content_utf8_bytes",
    )
    if any(type(record[name]) is not str or not record[name] for name in text_fields):
        raise RuntimeError("legacy progress text fields are not exact nonempty strings")
    if (
        any(type(record[name]) is not int for name in integer_fields)
        or record["sequence"] < 1
        or any(record[name] < 0 for name in integer_fields[1:])
        or type(record["rolling_content_sha256"]) is not str
    ):
        raise RuntimeError("legacy progress numeric or digest fields are malformed")
    progress = StructuredStreamProgress(
        call_id=record["call_id"],
        provider_attempt_id=record["provider_attempt_id"],
        sequence=record["sequence"],
        kind=StructuredStreamProgressKind(record["kind"]),
        channel=StructuredStreamChannel(record["channel"]),
        elapsed_ns=record["elapsed_ns"],
        event_content_utf8_bytes=record["event_content_utf8_bytes"],
        cumulative_content_utf8_bytes=record["cumulative_content_utf8_bytes"],
        rolling_content_sha256=record["rolling_content_sha256"],
    )
    progress.__post_init__()
    return {"schema_version": 1, **dict(record)}


def _provider_evidence(
    run_dir: Path,
    manifest: Mapping[str, object],
) -> tuple[
    dict[str, object], dict[str, dict[str, object]], dict[str, dict[str, object]]
]:
    request_rows = _observed_records(run_dir / "request_evidence.jsonl")
    output_rows = _observed_records(run_dir / "output_evidence.jsonl")
    outcome_rows = _observed_records(run_dir / "queue_outcomes.jsonl")
    outbound_rows = _observed_records(run_dir / "outbound_requests.jsonl")
    progress_rows = _observed_records(run_dir / "stream_progress.jsonl")

    requests = tuple(
        validate_structured_generation_request_evidence_record(row)
        for row in request_rows
    )
    request_by_call = {str(row["call_id"]): row for row in requests}
    if len(request_by_call) != len(requests):
        raise RuntimeError("logical request call IDs are not unique")
    outputs: list[dict[str, object]] = []
    for row in output_rows:
        call_id = row.get("call_id")
        request = request_by_call.get(str(call_id))
        if request is None:
            raise RuntimeError("output evidence has no logical request")
        outputs.append(
            validate_structured_generation_output_evidence_record(
                row, request_evidence=request
            )
        )
    output_by_call = {str(row["call_id"]): row for row in outputs}
    if len(output_by_call) != len(outputs):
        raise RuntimeError("output evidence call IDs are not unique")

    outcomes = tuple(
        validate_structured_generation_outcome_record(row) for row in outcome_rows
    )
    outcome_by_call = {str(row["task_id"]): row for row in outcomes}
    outbound = tuple(
        validate_openrouter_outbound_request_manifest_record(row)
        for row in outbound_rows
    )
    outbound_by_call = {str(row["call_id"]): row for row in outbound}
    call_ids = set(request_by_call)
    if not (
        call_ids == set(output_by_call) == set(outcome_by_call) == set(outbound_by_call)
        and len(outcome_by_call) == len(outcomes)
        and len(outbound_by_call) == len(outbound)
    ):
        raise RuntimeError("request/output/outcome/outbound call joins are incomplete")

    raw_progress_public_schema = all("schema_version" in row for row in progress_rows)
    if raw_progress_public_schema:
        normalized_progress = list(progress_rows)
        progress_normalization: dict[str, object] = {
            "applied": False,
            "operation": "none",
            "in_memory_only": True,
            "source_run_mutated": False,
        }
    else:
        normalized_progress = [
            _validate_legacy_progress_record(row) for row in progress_rows
        ]
        progress_normalization = {
            "applied": True,
            "operation": "add_missing_literal_field",
            "field": "schema_version",
            "value": 1,
            "in_memory_only": True,
            "source_run_mutated": False,
        }

    first_outbound = outbound[0]
    framework_versions = _object(
        first_outbound["framework_versions"], label="framework versions"
    )
    settings = _object(first_outbound["settings"], label="outbound settings")
    transport_names = (
        "model",
        "provider",
        "reasoning",
        "usage",
        "stream",
        "stream_options",
        "tool_choice",
        "response_format",
    )
    expected_transport = {name: settings[name] for name in transport_names}
    join = build_provider_attempt_terminal_join_receipt(
        logical_requests=requests,
        outbound_manifests=outbound,
        terminal_outcomes=outcomes,
        progress_rows=normalized_progress,
        expected_framework_versions=framework_versions,
        expected_transport_settings=expected_transport,
    )
    if join.get("join_valid") is not True:
        raise RuntimeError("provider-attempt terminal join is invalid")

    model_manifest = _object(manifest.get("model"), label="manifest model")
    progress_by_attempt: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in normalized_progress:
        progress_by_attempt[str(row["provider_attempt_id"])].append(row)

    calls: list[dict[str, object]] = []
    total_cost = Decimal("0")
    for call_id in sorted(call_ids):
        request = request_by_call[call_id]
        output = output_by_call[call_id]
        outcome = outcome_by_call[call_id]
        physical = outbound_by_call[call_id]
        attempts = _array(outcome.get("attempts"), label="queue outcome attempts")
        response = _object(outcome.get("response"), label="queue outcome response")
        if len(attempts) != 1:
            raise RuntimeError("successful R4 call did not have exactly one attempt")
        attempt = _object(attempts[0], label="queue attempt")
        attempt_request = _object(
            attempt.get("request_evidence"), label="attempt request evidence"
        )
        provider_attempt_id = str(physical["provider_attempt_id"])
        call_progress = progress_by_attempt.get(provider_attempt_id, [])
        if (
            outcome.get("status") != "succeeded"
            or attempt.get("status") != "succeeded"
            or attempt.get("attempt_number") != 1
            or attempt_request.get("provider_attempt_id") != provider_attempt_id
            or attempt_request.get("prompt_sha256") != request["wire_prompt_sha256"]
            or output.get("provider_response_id")
            != response.get("provider_response_id")
            or response.get("requested_model") != model_manifest.get("model_name")
            or response.get("resolved_model") != model_manifest.get("model_name")
            or response.get("resolved_provider") != "StreamLake"
            or response.get("finish_reason") != "tool_call"
            or type(response.get("reasoning_tokens")) is not int
            or response["reasoning_tokens"] <= 0
            or physical["settings"]["max_completion_tokens"]
            != model_manifest.get("max_output_tokens")
            or physical["settings"]["reasoning"] != model_manifest.get("reasoning")
            or physical["settings"]["provider"]
            != {
                "only": model_manifest["provider_options"]["only"],
                "allow_fallbacks": model_manifest["provider_options"][
                    "allow_fallbacks"
                ],
            }
            or not all(physical["forbidden_fields_absent"].values())
        ):
            raise RuntimeError("provider call does not join the frozen route")
        if [row["sequence"] for row in call_progress] != list(
            range(1, len(call_progress) + 1)
        ):
            raise RuntimeError("provider progress sequence is not contiguous")
        if (
            not call_progress
            or call_progress[-1]["kind"] != "stream_completed"
            or sum(row["kind"] == "stream_completed" for row in call_progress) != 1
        ):
            raise RuntimeError("provider progress has no unique terminal row")
        cost = response.get("cost_usd")
        if cost is not None:
            total_cost += Decimal(str(cost))
        calls.append(
            {
                "call_id": call_id,
                "operation": request["operation"],
                "provider_attempt_id": provider_attempt_id,
                "request_evidence_sha256": request["request_evidence_sha256"],
                "output_evidence_sha256": output["output_evidence_sha256"],
                "typed_output_sha256": output["typed_output_sha256"],
                "provider_response_id": response["provider_response_id"],
                "attempt_count": len(attempts),
                "queue_time_ns": outcome["queue_time_ns"],
                "service_time_ns": outcome["service_time_ns"],
                "total_time_ns": outcome["total_time_ns"],
                "latency_ns": response["latency_ns"],
                "input_tokens": response["input_tokens"],
                "output_tokens": response["output_tokens"],
                "reasoning_tokens": response["reasoning_tokens"],
                "cache_read_tokens": response["cache_read_tokens"],
                "cache_write_tokens": response["cache_write_tokens"],
                "cost_usd": response["cost_usd"],
                "finish_reason": response["finish_reason"],
                "progress_row_count": len(call_progress),
                "terminal_progress_sequence": call_progress[-1]["sequence"],
            }
        )

    expected_calls = int(
        _object(manifest.get("protocol"), label="manifest protocol")[
            "planned_logical_llm_calls"
        ]
    )
    operation_counts = Counter(str(row["operation"]) for row in requests)
    if len(calls) != expected_calls:
        raise RuntimeError("provider call count differs from the frozen protocol")
    return (
        {
            "logical_call_count": len(calls),
            "operation_counts": dict(sorted(operation_counts.items())),
            "all_calls_succeeded_first_attempt": True,
            "all_calls_have_positive_reasoning_tokens": True,
            "requested_and_resolved_model": model_manifest["model_name"],
            "resolved_provider": "StreamLake",
            "reasoning": model_manifest["reasoning"],
            "reasoning_mode_absent": model_manifest.get("reasoning_mode") is None,
            "known_cost_usd": str(total_cost),
            "token_totals": {
                name: sum(int(call[name]) for call in calls)
                for name in (
                    "input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                )
            },
            "latency_ns": {
                "min": min(int(call["latency_ns"]) for call in calls),
                "median": statistics.median(int(call["latency_ns"]) for call in calls),
                "max": max(int(call["latency_ns"]) for call in calls),
                "sum": sum(int(call["latency_ns"]) for call in calls),
            },
            "durable_journal_counts": {
                "request": len(requests),
                "output": len(outputs),
                "outcome": len(outcomes),
                "outbound_physical_attempt": len(outbound),
                "stream_progress": len(progress_rows),
            },
            "public_terminal_join": {
                "join_valid": join["join_valid"],
                "join_receipt_sha256": join["join_receipt_sha256"],
                "source_counts": join["source_counts"],
                "defects": join["defects"],
                "invariants": join["invariants"],
            },
            "raw_progress_public_schema_conformant": raw_progress_public_schema,
            "progress_normalization": progress_normalization,
            "progress_schema_limitation": (
                None
                if raw_progress_public_schema
                else "R4 raw progress records omit schema_version. Exact legacy "
                "field/type validation passed; adding only schema_version=1 to an "
                "in-memory projection makes the full public terminal join pass."
            ),
            "calls": calls,
        },
        request_by_call,
        output_by_call,
    )


def _wave_receipt_from_record(
    record: Mapping[str, object],
) -> CampaignPortfolioWavePreparationReceipt:
    if record.get("schema_version") != 1:
        raise RuntimeError("wave preparation schema version is unsupported")
    kwargs: dict[str, Any] = {
        name: value
        for name, value in record.items()
        if name not in {"schema_version", "receipt_sha256"}
    }
    for name in (
        "card_records",
        "card_reference_mapping",
        "test_eligible_reflection_receipts",
    ):
        kwargs[name] = tuple(freeze_json(value) for value in kwargs[name])
    for name in ("context_projection_identity", "memory_credit_identity"):
        if kwargs[name] is not None:
            kwargs[name] = freeze_json(kwargs[name])
    return CampaignPortfolioWavePreparationReceipt(**kwargs)


def _wave_evidence(
    run_dir: Path,
    manifest: Mapping[str, object],
    stage_events: Sequence[Mapping[str, object]],
    output_by_call: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    records = _observed_records(
        run_dir / "wave_preparations.jsonl",
        record_key="authenticated_wave_preparation",
    )
    by_call: dict[str, dict[str, object]] = {}
    for record in records:
        receipt = _wave_receipt_from_record(record)
        if receipt.receipt_sha256 != record.get("receipt_sha256"):
            raise RuntimeError("wave preparation receipt hash is invalid")
        call_id = str(record["selector_call_id"])
        if call_id in by_call:
            raise RuntimeError("wave preparation selector call ID is duplicated")
        by_call[call_id] = record

    selection = _object(manifest.get("portfolio_selection"), label="selection manifest")
    allocation = _object(selection.get("allocator"), label="allocator manifest")
    allocation_config = _object(
        allocation.get("configuration"), label="allocator configuration"
    )
    expected_proposal = int(allocation_config["slate_size"])
    expected_selected = int(allocation_config["portfolio_size"])
    minimum_families = int(
        _object(
            selection.get("hard_allocation_contract_rendered_preprovider"),
            label="hard allocation contract",
        )["minimum_distinct_families"]
    )

    summaries: list[dict[str, object]] = []
    selector_call_ids: list[str] = []
    for event in stage_events:
        receipt = _object(
            _object(event["payload"], label="stage payload").get("stage_receipt"),
            label="stage receipt",
        )
        if receipt.get("kind") != "portfolio":
            continue
        result = _object(receipt.get("result"), label="portfolio stage result")
        waves = _array(
            result.get("portfolio_wave_receipts"), label="portfolio wave receipts"
        )
        for raw_wave in waves:
            wave = _object(raw_wave, label="portfolio wave receipt")
            call_id = str(wave["selection_call_id"])
            selector_call_ids.append(call_id)
            prepared = by_call.get(call_id)
            output = output_by_call.get(call_id)
            if prepared is None or output is None:
                raise RuntimeError(
                    "portfolio wave lacks preparation or output evidence"
                )
            proposal = _object(output["typed_output"], label="selector typed output")
            proposed_members = [
                _object(value, label="proposed slate member")
                for value in _array(proposal.get("members"), label="proposed slate")
            ]
            attributions = [
                _object(value, label="action attribution")
                for value in _array(
                    wave.get("action_attributions"), label="action attributions"
                )
            ]
            selected: list[dict[str, object]] = []
            selected_option_ids: list[str] = []
            changed_path_sets: list[set[str]] = []
            selected_families: set[str] = set()
            proposed_by_option = {
                str(value["option_id"]): value for value in proposed_members
            }
            assigned_card_keys = {
                str(_object(value, label="card record")["card_key"])
                for value in _array(
                    prepared.get("card_records"), label="prepared card records"
                )
            }
            for attribution in attributions:
                member = _object(
                    attribution.get("selected_member"), label="selected member"
                )
                option_id = str(member["option_id"])
                proposal_member = proposed_by_option.get(option_id)
                if proposal_member is None:
                    raise RuntimeError("selected option is absent from the model slate")
                if member.get("supporting_card_keys") != proposal_member.get(
                    "supporting_card_keys"
                ) or not set(member.get("supporting_card_keys", [])).issubset(
                    assigned_card_keys
                ):
                    raise RuntimeError(
                        "selected card attribution differs from the slate"
                    )
                matching_wave_member = next(
                    (
                        _object(value, label="wave member")
                        for value in _array(wave.get("members"), label="wave members")
                        if _object(value, label="wave member").get("candidate_id")
                        == attribution.get("candidate_id")
                    ),
                    None,
                )
                if matching_wave_member is None:
                    raise RuntimeError("selected attribution has no evaluated member")
                materialization = _object(
                    matching_wave_member.get("materialization"),
                    label="member materialization",
                )
                changed_paths = [
                    str(value) for value in materialization["changed_paths"]
                ]
                changed_path_sets.append(set(changed_paths))
                selected_families.add(str(member["family"]))
                selected_option_ids.append(option_id)
                selected.append(
                    {
                        "rank": member["rank"],
                        "option_id": option_id,
                        "family": member["family"],
                        "changed_paths": changed_paths,
                        "supporting_card_keys": member["supporting_card_keys"],
                        "candidate_id": attribution["candidate_id"],
                        "candidate_valid": matching_wave_member["candidate_valid"],
                        "reward_hex": matching_wave_member["reward_hex"],
                    }
                )
            if (
                len(proposed_members) != expected_proposal
                or len(proposed_by_option) != expected_proposal
                or len(selected) != expected_selected
                or len(set(selected_option_ids)) != expected_selected
                or any(
                    left & right
                    for index, left in enumerate(changed_path_sets)
                    for right in changed_path_sets[index + 1 :]
                )
                or len(selected_families) < minimum_families
                or not all(
                    any(
                        key in member.get("supporting_card_keys", [])
                        for member in proposed_members
                    )
                    for key in assigned_card_keys
                )
            ):
                raise RuntimeError(
                    "portfolio wave violates its frozen K8-to-K4 contract"
                )
            summaries.append(
                {
                    "generation": wave["generation"],
                    "parent_candidate_id": wave["parent_candidate_id"],
                    "parent_lane_id": prepared["parent_lane_id"],
                    "selector_call_id": call_id,
                    "wave_preparation_receipt_sha256": prepared["receipt_sha256"],
                    "proposal_width": len(proposed_members),
                    "evaluation_width": len(selected),
                    "assigned_card_references": prepared["card_reference_mapping"],
                    "test_eligible_reflections": prepared[
                        "test_eligible_reflection_receipts"
                    ],
                    "proposal": [
                        {
                            "rank": index,
                            "option_id": value["option_id"],
                            "role_proposal": value["role_proposal"],
                            "supporting_card_keys": value["supporting_card_keys"],
                        }
                        for index, value in enumerate(proposed_members, start=1)
                    ],
                    "selected": selected,
                    "distinct_selected_families": sorted(selected_families),
                    "pairwise_disjoint_selected_paths": True,
                    "memory_reward_hex": _object(
                        wave.get("memory_credit"), label="wave memory credit"
                    )["reward_hex"],
                }
            )

    expected_wave_count = len(
        _object(manifest.get("protocol"), label="manifest protocol").get(
            "portfolio_generations", []
        )
    ) * int(
        _object(manifest.get("protocol"), label="manifest protocol")[
            "parents_per_portfolio"
        ]
    )
    if len(summaries) != expected_wave_count or set(selector_call_ids) != set(by_call):
        raise RuntimeError(
            "wave preparation journal does not cover every selector wave"
        )
    return {
        "prepared_wave_count": len(records),
        "receipt_hashes_valid": True,
        "all_preparations_join_sealed_selector_waves": True,
        "all_k8_to_k4_contracts_pass": True,
        "waves": summaries,
    }


def _resolved_reflection_insights(
    typed_output: Mapping[str, object],
    evidence_catalog: Mapping[str, object],
) -> list[dict[str, object]]:
    mapping = {
        str(_object(value, label="evidence catalog entry")["citation_key"]): str(
            _object(value, label="evidence catalog entry")["contrast_id"]
        )
        for value in _array(evidence_catalog.get("entries"), label="catalog entries")
    }
    resolved: list[dict[str, object]] = []
    for raw in _array(typed_output.get("insights"), label="typed reflection insights"):
        value = _object(raw, label="typed reflection insight")
        keys = _array(
            value.pop("evidence_citation_keys", None),
            label="reflection evidence citation keys",
        )
        if any(type(key) is not str or key not in mapping for key in keys):
            raise RuntimeError("reflection output cites an unknown evidence key")
        value["evidence_contrast_ids"] = sorted(mapping[str(key)] for key in keys)
        value["schema_version"] = 2
        resolved.append(value)
    return resolved


def _reflection_and_memory_evidence_from_events(
    campaign_events: Sequence[Mapping[str, object]],
    stage_events: Sequence[Mapping[str, object]],
    output_by_call: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    admission_events = [
        event
        for event in campaign_events
        if event.get("kind") == "reflection_admitted_for_testing"
    ]
    admissions_by_reflection: dict[str, dict[str, object]] = {}
    for event in admission_events:
        receipt = _object(
            _object(event["payload"], label="admission payload").get(
                "test_admission_receipt"
            ),
            label="test admission receipt",
        )
        for item in _array(
            _object(receipt.get("evidence"), label="admission evidence").get(
                "reflection_completion_learning"
            ),
            label="reflection completion learning",
        ):
            learning = _object(item, label="reflection completion learning item")
            admissions_by_reflection[str(learning["reflection_receipt_sha256"])] = {
                "barrier_generation": receipt["barrier_generation"],
                "lifecycle_promoted": receipt["lifecycle_promoted"],
                "normal_retrieval_mutated": _object(
                    receipt["evidence"], label="admission evidence"
                )["normal_retrieval_mutated"],
                "registration": learning["evidence"],
            }

    reflections: list[dict[str, object]] = []
    for event in campaign_events:
        if event.get("kind") != "reflection_completed":
            continue
        receipt = _object(
            _object(event["payload"], label="reflection payload").get(
                "reflection_receipt"
            ),
            label="reflection receipt",
        )
        result = _object(
            receipt.get("quarantined_result"), label="quarantined reflection result"
        )
        call_id = str(result["call_id"])
        output = output_by_call.get(call_id)
        if output is None:
            raise RuntimeError("reflection completion has no output evidence")
        resolved = _resolved_reflection_insights(
            _object(output["typed_output"], label="reflection typed output"),
            _object(result["evidence_catalog"], label="reflection evidence catalog"),
        )
        campaign_insights = [
            _object(value, label="campaign insight")
            for value in _array(result.get("insights"), label="campaign insights")
        ]
        if resolved != campaign_insights:
            raise RuntimeError(
                "resolved reflection output differs from campaign insight"
            )
        admission = admissions_by_reflection.get(str(receipt["receipt_sha256"]), {})
        reflections.append(
            {
                "source_generation": receipt["source_generation"],
                "call_id": call_id,
                "reflection_receipt_sha256": receipt["receipt_sha256"],
                "reflection_result_sha256": _object(
                    result["campaign_reflection_learning"],
                    label="campaign reflection learning",
                )["record_sha256"],
                "status": receipt["status"],
                "quarantined": result["quarantined"],
                "lifecycle_promoted": result["lifecycle_promoted"],
                "evidence_catalog_identity_sha256": result[
                    "evidence_catalog_identity_sha256"
                ],
                "available_contrast_count": len(result["available_contrast_ids"]),
                "admission": admission,
                "insights": campaign_insights,
            }
        )

    trials: list[dict[str, object]] = []
    lifecycle_transitions: list[dict[str, object]] = []
    semantic_audits: list[dict[str, object]] = []
    observation_sha256s: list[str] = []
    for event in stage_events:
        receipt = _object(
            _object(event["payload"], label="stage payload").get("stage_receipt"),
            label="stage receipt",
        )
        if receipt.get("kind") != "portfolio":
            continue
        generation = int(receipt["generation"])
        result = _object(receipt["result"], label="portfolio result")
        credit_batch = _object(
            result.get("memory_credit_batch"), label="memory credit batch"
        )
        credits = _array(credit_batch.get("credits"), label="memory credits")
        for raw_credit in credits:
            credit = _object(raw_credit, label="memory credit")
            trials.append(
                {
                    "generation": generation,
                    "credit_unit_id": credit["credit_unit_id"],
                    "candidate_ids": credit["candidate_ids"],
                    "reward_hex": credit["reward_hex"],
                    "selection_decision_sha256": credit["selection_decision_sha256"],
                    "treatment_binding_sha256": credit["treatment_binding_sha256"],
                    "receipt_sha256": credit["receipt_sha256"],
                }
            )
        learning = _object(result["closed_loop_learning"], label="closed loop learning")
        evidence = _object(learning["evidence"], label="closed loop evidence")
        audit_preparation = _object(
            evidence["generation_audit_preparation"],
            label="generation audit preparation",
        )
        observation_sha256s.extend(
            str(value) for value in audit_preparation["observation_sha256s"]
        )
        coordinator = evidence.get("coordinator_preparation")
        if coordinator is None:
            continue
        coordinator_record = _object(coordinator, label="diagnostic coordinator")
        for raw_transition in coordinator_record["lifecycle_requests"]:
            transition = _object(raw_transition, label="lifecycle request")
            lifecycle_transitions.append({"generation": generation, **transition})
        projection = _object(
            audit_preparation["projection"], label="diagnostic audit projection"
        )
        for raw_audit in projection["audits"]:
            audit = _object(raw_audit, label="semantic audit")
            request = _object(audit["request"], label="semantic audit request")
            audit_receipt = _object(audit["receipt"], label="semantic audit receipt")
            semantic_audits.append(
                {
                    "generation": generation,
                    "reference": request["reference"],
                    "verdict": audit_receipt["verdict"],
                    "lifecycle_decision": audit_receipt["lifecycle_decision"],
                    "raw_support_count": audit_receipt["raw_support_count"],
                    "effective_support_cluster_count": audit_receipt[
                        "effective_support_cluster_count"
                    ],
                    "counterexample_count": len(audit_receipt["counterexample_ids"]),
                    "mechanism_identified": audit_receipt["mechanism_identified"],
                    "coverage_gaps": audit_receipt["coverage_gaps"],
                    "audit_receipt_sha256": audit_receipt["audit_receipt_sha256"],
                }
            )

    reflection_count = len(reflections)
    insight_count = sum(len(value["insights"]) for value in reflections)
    if (
        reflection_count != 3
        or insight_count != 6
        or len(trials) != 6
        or len(observation_sha256s) != 24
        or len(set(observation_sha256s)) != 24
        or len(lifecycle_transitions) != 4
        or any(value["new_state"] != "deprecated" for value in lifecycle_transitions)
        or any(value["status"] != "completed" for value in reflections)
        or any(value["quarantined"] is not True for value in reflections)
    ):
        raise RuntimeError("reflection or memory lifecycle accounting is incomplete")
    return {
        "reflection_count": reflection_count,
        "reflected_insight_count": insight_count,
        "all_reflections_completed_and_quarantined": True,
        "memory_trial_count": len(trials),
        "authenticated_action_observation_count": len(observation_sha256s),
        "lifecycle_transition_count": len(lifecycle_transitions),
        "deprecated_after_global_semantic_counterexample_count": len(
            lifecycle_transitions
        ),
        "adaptive_score_consumption": False,
        "causal_claim_allowed": False,
        "trials": trials,
        "semantic_audits": semantic_audits,
        "lifecycle_transitions": lifecycle_transitions,
        "reflections": reflections,
    }


def _pde_evidence(run_dir: Path, *, expected_count: int) -> dict[str, object]:
    paths = sorted((run_dir / "pde/evaluations").glob("direct-v3-*/manifest.json"))
    rows: list[dict[str, object]] = []
    for path in paths:
        value = _object(json.loads(path.read_bytes()), label="PDE manifest")
        container = _object(value.get("container_result"), label="container result")
        measurement = _object(
            container.get("resource_measurement"), label="resource measurement"
        )
        result = _object(container.get("result"), label="container PDE result")
        exact_volume = _object(
            container.get("exact_volume_contract"), label="exact volume contract"
        )
        checks = _object(value.get("checks"), label="PDE manifest checks")
        volume = _object(value.get("volume_agreement"), label="volume agreement")
        candidate = _object(value.get("candidate"), label="PDE candidate")
        elapsed = float(value.get("elapsed_s", math.inf))
        peak_rss = measurement.get("peak_rss_bytes_by_linux_kib_convention")
        numerator = exact_volume.get("exact_scaled_numerator_decimal")
        denominator = exact_volume.get("mesh_mass_denominator")
        exponent = exact_volume.get("binary64_common_denominator_exponent")
        exact_material: float | None = None
        if (
            type(numerator) is str
            and numerator.isdecimal()
            and type(denominator) is int
            and denominator > 0
            and type(exponent) is int
            and exponent >= 0
        ):
            exact_material = float(
                Fraction(int(numerator), denominator * (1 << exponent))
            )
        stdout_path = path.parent / "stdout.txt"
        stderr_path = path.parent / "stderr.txt"
        scientific_pass = (
            value.get("schema_version") == 3
            and value.get("evaluator_id") == "engibench-heatconduction2d-direct-v3"
            and value.get("all_checks_pass") is True
            and value.get("returncode") == 0
            and value.get("full_pde_solve_count") == 1
            and checks
            and all(item is True for item in checks.values())
            and checks.get("exact_cross_runtime_fe_volume_identity_matches") is True
            and volume.get("exact_identity_matches") is True
            and _sha256_file(stdout_path) == value.get("stdout_sha256")
            and _sha256_file(stderr_path) == value.get("stderr_sha256")
            and exact_material is not None
            and math.isfinite(float(result["thermal_term"]))
        )
        resource_pass = (
            math.isfinite(elapsed)
            and elapsed < 45.0
            and type(peak_rss) is int
            and 0 < peak_rss < 3 * 1024**3
        )
        rows.append(
            {
                "relative_manifest": path.relative_to(run_dir).as_posix(),
                "manifest_sha256": _sha256_file(path),
                "raw_array_sha256": candidate["raw_array_sha256"],
                "exact_volume_contract_sha256": exact_volume["contract_sha256"],
                "elapsed_s": elapsed,
                "peak_rss_bytes": peak_rss,
                "objectives": {
                    "material_fraction": exact_material,
                    "thermal_term": result["thermal_term"],
                },
                "scientific_contract_pass": scientific_pass,
                "resource_gate_pass": resource_pass,
            }
        )
    if (
        len(rows) != expected_count
        or not all(row["scientific_contract_pass"] for row in rows)
        or not all(row["resource_gate_pass"] for row in rows)
    ):
        raise RuntimeError("PDE evidence does not pass the frozen scientific gates")
    elapsed_values = [float(row["elapsed_s"]) for row in rows]
    slowest = max(rows, key=lambda row: float(row["elapsed_s"]))
    return {
        "manifest_count": len(rows),
        "one_full_pde_solve_per_unique_candidate": True,
        "all_scientific_contracts_pass": True,
        "all_under_45_s_and_3_gib": True,
        "elapsed_s": {
            "min": min(elapsed_values),
            "median": statistics.median(elapsed_values),
            "mean": statistics.fmean(elapsed_values),
            "max": max(elapsed_values),
            "sum": sum(elapsed_values),
        },
        "peak_rss_bytes_max": max(int(row["peak_rss_bytes"]) for row in rows),
        "slowest_evaluation": slowest,
        "rows": rows,
    }


def _affine_spec(manifest: Mapping[str, object]) -> AffineHypervolume2DSpec:
    utility = _object(manifest.get("utility"), label="utility manifest")
    axes = tuple(
        AffineObjectiveAxis(
            metric_id=str(axis["metric_id"]),
            goal=str(axis["goal"]),
            ideal=float.fromhex(str(axis["ideal_hex"])),
            reference=float.fromhex(str(axis["reference_hex"])),
        )
        for axis in (
            _object(value, label="affine objective axis")
            for value in _array(utility.get("axes"), label="affine axes")
        )
    )
    if len(axes) != 2:
        raise RuntimeError("affine utility does not contain two axes")
    spec = AffineHypervolume2DSpec(
        axes=(axes[0], axes[1]),
        reference_provenance=str(utility["reference_provenance"]),
    )
    if spec.to_record() != utility or spec.definition_sha256 != manifest.get(
        "utility_definition_sha256"
    ):
        raise RuntimeError("affine utility differs from its frozen definition")
    return spec


def _front_points(front: Sequence[object]) -> tuple[dict[str, float], ...]:
    points: list[dict[str, float]] = []
    for raw_candidate in front:
        candidate = _object(raw_candidate, label="archive front candidate")
        objectives = _array(candidate.get("objectives"), label="candidate objectives")
        points.append(
            {
                str(_object(value, label="candidate objective")["metric_id"]): (
                    float.fromhex(
                        str(_object(value, label="candidate objective")["value_hex"])
                    )
                )
                for value in objectives
            }
        )
    return tuple(points)


def _optimization_evidence(
    manifest: Mapping[str, object],
    campaign_events: Sequence[Mapping[str, object]],
    stage_events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    spec = _affine_spec(manifest)
    cutoffs = {
        int(
            _object(event["payload"], label="archive payload")["archive_cutoff"][
                "generation"
            ]
        ): event
        for event in campaign_events
        if event.get("kind") == "archive_utility_frozen"
    }
    first_cutoff = _object(cutoffs[1]["payload"], label="initial archive payload")
    seed_front = _object(first_cutoff["archive_cutoff"], label="initial cutoff")[
        "archive"
    ]["front_candidates"]
    fronts: list[list[object]] = [list(seed_front)]
    for event in stage_events:
        receipt = _object(
            _object(event["payload"], label="stage payload")["stage_receipt"],
            label="stage receipt",
        )
        archive_after = _object(
            _object(receipt["result"], label="stage result")["archive_after"],
            label="archive after stage",
        )
        fronts.append(_array(archive_after["front_candidates"], label="archive front"))

    trajectory: list[dict[str, object]] = []
    previous_hv: float | None = None
    for generation, front in enumerate(fronts):
        snapshot = AffineHypervolumeSnapshot2D.create(
            spec=spec,
            archive_points=_front_points(front),
        )
        gain = 0.0 if previous_hv is None else snapshot.base_hypervolume - previous_hv
        trajectory.append(
            {
                "generation": generation,
                "front_size": len(front),
                "normalized_hypervolume": snapshot.base_hypervolume,
                "normalized_hypervolume_hex": snapshot.base_hypervolume.hex(),
                "raw_oriented_hypervolume_hex": (
                    snapshot.raw_oriented_base_hypervolume.hex()
                ),
                "gain_from_previous_generation": gain,
                "gain_from_previous_generation_hex": gain.hex(),
                "snapshot_sha256": snapshot.snapshot_sha256,
                "front_candidate_ids": [
                    _object(value, label="front candidate")["candidate_id"]
                    for value in front
                ],
            }
        )
        previous_hv = snapshot.base_hypervolume

    # Every next-generation cutoff authenticates the preceding stage's front
    # and base hypervolume. The terminal G6 value has no later cutoff and is
    # recomputed from the final sealed archive.
    for generation in range(1, len(trajectory)):
        cutoff_event = cutoffs.get(generation + 1)
        if cutoff_event is None:
            continue
        archive_utility = _object(
            _object(cutoff_event["payload"], label="archive payload")[
                "archive_utility"
            ],
            label="archive utility snapshot",
        )
        snapshot_receipt = _object(
            archive_utility["snapshot_receipt"], label="utility snapshot receipt"
        )
        if (
            snapshot_receipt["base_hypervolume_hex"]
            != trajectory[generation]["normalized_hypervolume_hex"]
        ):
            raise RuntimeError("recomputed hypervolume differs from the next cutoff")

    final_front = fronts[-1]
    return {
        "utility_definition_sha256": spec.definition_sha256,
        "trajectory": trajectory,
        "final_normalized_hypervolume": trajectory[-1]["normalized_hypervolume"],
        "final_normalized_hypervolume_hex": trajectory[-1][
            "normalized_hypervolume_hex"
        ],
        "final_raw_oriented_hypervolume_hex": trajectory[-1][
            "raw_oriented_hypervolume_hex"
        ],
        "final_front": [
            {
                "candidate_id": _object(value, label="front candidate")["candidate_id"],
                "generation": _object(value, label="front candidate")["generation"],
                "configuration_sha256": _object(value, label="front candidate")[
                    "configuration_sha256"
                ],
                "objectives": _object(value, label="front candidate")["objectives"],
            }
            for value in final_front
        ],
        "matched_baseline_in_this_run": False,
        "efficacy_claim_allowed": False,
    }


def _report_projection_failure(
    run_dir: Path,
    finalization: Mapping[str, object],
    campaign: Mapping[str, object],
) -> dict[str, object]:
    summary = _object(
        json.loads((run_dir / "summary.json").read_bytes()), label="source summary"
    )
    expected_digest = _sha256_bytes(
        b"AttributeError\x00" + REPORT_FAILURE_MESSAGE.encode("utf-8")
    )
    harness_path = (
        run_dir / "source_snapshot/agent_evolve/examples/development/"
        "run_heat2d_generic_campaign.py"
    )
    memory_path = (
        run_dir / "source_snapshot/agent_evolve/src/agent_evolve/application/"
        "insight_memory.py"
    )
    harness_source = harness_path.read_text(encoding="utf-8")
    memory_source = memory_path.read_text(encoding="utf-8")
    memory_tree = ast.parse(memory_source)
    memory_class = next(
        (
            node
            for node in memory_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "InsightMemoryEntry"
        ),
        None,
    )
    class_methods = (
        []
        if memory_class is None
        else [
            node.name
            for node in memory_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    )
    failing_lines = [
        index
        for index, line in enumerate(harness_source.splitlines(), start=1)
        if "value.to_record() for value in reflected_memory_entries" in line
    ]
    exact_cause_proven = (
        summary.get("status") == "failed"
        and summary.get("failure_type") == "AttributeError"
        and summary.get("failure_digest_sha256") == expected_digest
        and memory_class is not None
        and "to_record" not in class_methods
        and len(failing_lines) == 1
        and campaign.get("campaign_finalization_status") == "completed"
        and campaign.get("cleanup_released") is True
        and finalization.get("status") == "failed"
    )
    if not exact_cause_proven:
        raise RuntimeError("post-cleanup report projection failure was not proven")
    return {
        "source_run_recorded_status": summary["status"],
        "source_run_finalization_status": finalization["status"],
        "failure_type": summary["failure_type"],
        "failure_digest_sha256": summary["failure_digest_sha256"],
        "failure_message_recovered_from_digest_and_sealed_source": (
            REPORT_FAILURE_MESSAGE
        ),
        "sealed_harness_line": failing_lines[0],
        "insight_memory_entry_has_to_record_in_sealed_source": False,
        "exact_failure_cause_proven": True,
        "failure_boundary": (
            "All six stages, all three reflections, campaign finalization, and "
            "runtime cleanup completed. The harness then raised AttributeError "
            "while projecting reflected InsightMemoryEntry values into the final "
            "summary; no provider call or PDE evaluation failed."
        ),
        "source_run_relabelled_or_mutated": False,
    }


def _campaign_records(run_dir: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        _object(row["authenticated_campaign_event"], label="campaign event")
        for row in read_jsonl(run_dir / "campaign_events.jsonl")
    )


def adjudicate(run_dir: Path) -> dict[str, object]:
    """Adjudicate one immutable source run without publishing any file."""

    source = run_dir.expanduser().resolve(strict=True)
    finalization = verify_finalized_run_directory(source)
    manifest = _object(
        json.loads((source / "manifest.json").read_bytes()), label="manifest"
    )
    if manifest.get("mode") != "live":
        raise RuntimeError("adjudication requires a sealed live campaign")
    workload = _object(manifest.get("workload"), label="workload manifest")
    if workload.get("problem_workload_id") != (
        "engibench-heatconduction2d-constructive-pareto-v1"
    ):
        raise RuntimeError("adjudication input is not the Heat2D campaign")
    preparation_rows = read_jsonl(source / "preparation.jsonl")
    if len(preparation_rows) != 1:
        raise RuntimeError("campaign must contain exactly one preparation record")
    preparation = _object(preparation_rows[0], label="campaign preparation")

    source_snapshot = _source_snapshot_evidence(source, manifest)
    preregistration = _preregistration_evidence(source)
    campaign, stage_events = _campaign_evidence(source, manifest, preparation)
    engine = _engine_evidence(
        source,
        planned_evaluations=int(
            _object(manifest["protocol"], label="manifest protocol")[
                "planned_unique_evaluations"
            ]
        ),
    )
    provider, _, output_by_call = _provider_evidence(source, manifest)
    wave = _wave_evidence(source, manifest, stage_events, output_by_call)
    campaign_events = _campaign_records(source)
    memory = _reflection_and_memory_evidence_from_events(
        campaign_events, stage_events, output_by_call
    )
    pde = _pde_evidence(
        source,
        expected_count=int(
            _object(manifest["protocol"], label="manifest protocol")[
                "planned_unique_evaluations"
            ]
        ),
    )
    optimization = _optimization_evidence(manifest, campaign_events, stage_events)
    report_failure = _report_projection_failure(source, finalization, campaign)

    protocol = _object(manifest["protocol"], label="manifest protocol")
    last_stage_counters = _object(
        campaign["last_stage_seal_counters"], label="last stage-seal counters"
    )
    last_stage_sequence = int(stage_events[-1]["sequence"])
    terminal_reflection_call_count = sum(
        1
        for event in campaign_events
        if event.get("kind") == "reflection_completed"
        and int(event["sequence"]) > last_stage_sequence
    )
    scientific_gates = {
        "campaign_finalized_completed": (
            campaign["campaign_finalization_status"] == "completed"
        ),
        "runtime_cleanup_released": campaign["cleanup_released"] is True,
        "exact_generation_count": last_stage_counters["generations_completed"]
        == protocol["generations"],
        "exact_unique_evaluations": last_stage_counters["unique_evaluations"]
        == protocol["planned_unique_evaluations"],
        "exact_candidate_occurrences": last_stage_counters["candidate_occurrences"]
        == protocol["planned_unique_evaluations"],
        "exact_total_logical_llm_calls": provider["logical_call_count"]
        == protocol["planned_logical_llm_calls"],
        "stage_counter_reconciles_terminal_reflection": (
            last_stage_counters["logical_agent_calls"] + terminal_reflection_call_count
            == protocol["planned_logical_llm_calls"]
        ),
        "all_llm_calls_succeeded_first_attempt": provider[
            "all_calls_succeeded_first_attempt"
        ],
        "provider_attempt_join_valid": provider["public_terminal_join"]["join_valid"],
        "all_wave_preparations_authenticated": wave["receipt_hashes_valid"],
        "all_k8_to_k4_contracts_pass": wave["all_k8_to_k4_contracts_pass"],
        "all_candidates_valid_and_compliant": engine[
            "all_candidates_valid_and_compliant"
        ],
        "exact_pde_manifest_count": pde["manifest_count"]
        == protocol["planned_unique_evaluations"],
        "all_pde_scientific_contracts_pass": pde["all_scientific_contracts_pass"],
        "all_pde_resource_gates_pass": pde["all_under_45_s_and_3_gib"],
        "three_completed_quarantined_reflections": memory["reflection_count"] == 3,
        "six_memory_trials": memory["memory_trial_count"] == 6,
        "source_snapshot_exact": source_snapshot["launch_and_snapshot_identity_exact"],
    }
    scientific_workflow_completed = all(scientific_gates.values())
    if not scientific_workflow_completed:
        failed_gates = sorted(
            name for name, passed in scientific_gates.items() if not passed
        )
        raise RuntimeError(
            "scientific workflow did not pass offline adjudication: "
            + ", ".join(failed_gates)
        )

    file_records = _object(finalization["files"], label="finalized files")
    journal_names = (
        "campaign_events.jsonl",
        "engine_events.jsonl",
        "request_evidence.jsonl",
        "output_evidence.jsonl",
        "queue_outcomes.jsonl",
        "outbound_requests.jsonl",
        "stream_progress.jsonl",
        "wave_preparations.jsonl",
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "adjudicated_scientific_workflow_completed_report_projection_failed",
        "adjudication_mode": "strictly_read_only_offline",
        "provider_calls_repeated": False,
        "pde_evaluations_repeated": False,
        "source_run_mutated": False,
        "source_run": {
            "run_id": manifest["run_id"],
            "path": _workspace_label(source),
            "recorded_status": finalization["status"],
            "finalization_sha256": finalization["finalization_sha256"],
            "recursive_content_sha256": finalization["recursive_content_sha256"],
            "recursive_file_count": finalization["recursive_file_count"],
            "journal_identities": {name: file_records[name] for name in journal_names},
        },
        "workflow_classification": {
            "scientific_workflow_completed": scientific_workflow_completed,
            "scientific_execution_health": "healthy",
            "final_report_projection_health": "failed_after_cleanup",
            "immutable_run_health": "recorded_failed_as_required",
            "paper_ready_efficacy_result": False,
            "reason_not_paper_ready": (
                "This development run has no matched baseline and the manifest "
                "forbids an efficacy or reflection-causal claim. It establishes "
                "end-to-end workflow function and exposes trace-level mechanisms."
            ),
        },
        "scientific_gates": scientific_gates,
        "claim_boundary": manifest["claim_boundary"],
        "report_projection_failure": report_failure,
        "source_snapshot": source_snapshot,
        "preregistration": preregistration,
        "campaign": campaign,
        "engine": engine,
        "provider": provider,
        "portfolio_waves": wave,
        "memory_and_reflection": memory,
        "optimization": optimization,
        "pde": pde,
        "limitations": [
            "No matched baseline was executed in this run; no efficacy claim is allowed.",
            "The four reflected cards tested at G3/G5 were deprecated after global "
            "semantic counterexamples; positive search reward is not semantic truth.",
            "The terminal G6 reflection was admitted only for controlled future "
            "testing and was not evaluated in this run.",
            "Raw R4 stream-progress rows omit schema_version; an explicitly reported "
            "in-memory normalization is required by the public join decoder.",
            "The postrun live-workspace source check was computed but not durably "
            "published because final summary projection failed.",
        ],
        "adjudicator": {
            "script": _workspace_label(Path(__file__)),
            "script_sha256": _sha256_file(Path(__file__)),
            "output_is_outside_source_run_required": True,
        },
    }
    result["adjudication_sha256"] = _sha256_bytes(
        ADJUDICATION_DOMAIN + canonical_json_bytes(result)
    )
    return result


def _safe_output_path(run_dir: Path, output: Path) -> Path:
    source = run_dir.expanduser().resolve(strict=True)
    target = output.expanduser().resolve(strict=False)
    if target == source or source in target.parents:
        raise RuntimeError(
            "adjudication output must be outside the immutable source run"
        )
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify and print the result commitment without writing output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = _safe_output_path(args.run_dir, args.output)
    result = adjudicate(args.run_dir)
    if not args.check_only:
        write_json_atomic(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "source_run": result["source_run"]["run_id"],
                "source_run_mutated": result["source_run_mutated"],
                "provider_calls_repeated": result["provider_calls_repeated"],
                "pde_evaluations_repeated": result["pde_evaluations_repeated"],
                "scientific_workflow_completed": result["workflow_classification"][
                    "scientific_workflow_completed"
                ],
                "adjudication_sha256": result["adjudication_sha256"],
                "output": None if args.check_only else _workspace_label(output),
            },
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
