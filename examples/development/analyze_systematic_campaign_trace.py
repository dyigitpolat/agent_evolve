#!/usr/bin/env python3
"""Normalize heterogeneous AgentEvolve campaign traces into one analysis row.

The analyzer depends only on authenticated campaign/engine/queue records and
the workload-neutral affine archive contract embedded in them.  It does not
import a benchmark adapter or know any workload metric names.  This keeps the
behavioral endpoints comparable as new domains and model routes are added.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any

from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json
from agent_evolve.policies.reward import hypervolume_2d, hypervolume_3d
from agent_evolve.policies.variation.exact_composition_capacity import (
    ExactKCompositionCapacityProjection,
)
from agent_evolve.ports.frontier_target import CampaignPortfolioFrontierTarget


_RANK = re.compile(r"\.rank_0*([1-9][0-9]*)$")
_GENERATION = re.compile(r"g([0-9]{2})(?:_|p)")
_PARENT_SLOT = re.compile(r"p0*([1-9][0-9]*)\.rank_")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(type(value) is not dict for value in rows):
        raise TypeError(f"{path} contains a non-object row")
    return rows


def _unwrap(value: dict[str, Any]) -> dict[str, Any]:
    """Remove any number of durable authentication envelopes."""

    current = value
    while True:
        nested = next(
            (
                current[key]
                for key in ("authenticated_record", "authenticated_campaign_event")
                if type(current.get(key)) is dict
            ),
            None,
        )
        if nested is None:
            return current
        current = nested


def _durable_completion_evidence(
    campaign_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Separate campaign completion from convenience-summary completion.

    A reporting-layer exception can occur after the campaign has sealed every
    stage, finalized its archive, and released runtime resources.  The durable
    campaign journal is authoritative for those lifecycle facts; ``summary``
    remains authoritative for its own status and health projection.  Keeping
    both avoids laundering a reporting failure into ``completed_healthy`` while
    retaining scientifically usable trace evidence.
    """

    events = [_unwrap(value) for value in campaign_rows]
    finalizations = [
        value for value in events if value.get("kind") == "execution_finalized"
    ]
    cleanups = [value for value in events if value.get("kind") == "runtime_cleaned"]
    finalization_receipt = (
        finalizations[-1].get("payload", {}).get("finalization_receipt")
        if finalizations
        else None
    )
    cleanup_receipt = (
        cleanups[-1].get("payload", {}).get("cleanup_receipt") if cleanups else None
    )
    evolution_completed = (
        type(finalization_receipt) is dict
        and finalization_receipt.get("status") == "completed"
    )
    runtime_released = (
        type(cleanup_receipt) is dict and cleanup_receipt.get("released") is True
    )

    raw_wall_s = summary.get("wall_s")
    if (
        type(raw_wall_s) in (int, float)
        and not isinstance(raw_wall_s, bool)
        and math.isfinite(float(raw_wall_s))
        and float(raw_wall_s) >= 0.0
    ):
        wall_s = float(raw_wall_s)
        wall_time_source = "summary.wall_s"
    else:
        offsets = [
            value.get("observation", {}).get("monotonic_ns_since_execution_start")
            for value in campaign_rows
            if type(value.get("observation")) is dict
        ]
        valid_offsets = [
            value
            for value in offsets
            if type(value) is int and not isinstance(value, bool) and value >= 0
        ]
        if not valid_offsets:
            raise ValueError(
                "campaign lacks both summary wall time and durable monotonic offsets"
            )
        wall_s = max(valid_offsets) / 1e9
        wall_time_source = (
            "campaign_events.observation.max_monotonic_ns_since_execution_start"
        )

    health = summary.get("health")
    summary_health_available = type(health) is dict and bool(health)
    return {
        "summary_status": summary.get("status"),
        "summary_health_available": summary_health_available,
        "evolution_completed": evolution_completed,
        "runtime_released": runtime_released,
        "durable_campaign_complete": evolution_completed and runtime_released,
        "wall_s": wall_s,
        "wall_time_source": wall_time_source,
    }


def _hex_or_number(value: object) -> float:
    if type(value) is str:
        return float.fromhex(value)
    if type(value) in (int, float) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result):
            return result
    raise ValueError("objective value is not a finite number or binary64 hex string")


def _objective_map(candidate: dict[str, Any]) -> dict[str, float]:
    raw = candidate.get("objectives")
    result: dict[str, float] = {}
    if type(raw) is list:
        for item in raw:
            if type(item) is not dict or type(item.get("metric_id")) is not str:
                raise ValueError("list objective row is malformed")
            source = item.get("value_hex", item.get("value"))
            result[item["metric_id"]] = _hex_or_number(source)
    elif type(raw) is dict:
        for metric_id, item in raw.items():
            if type(metric_id) is not str:
                raise ValueError("objective metric ID is malformed")
            source = (
                item.get("value_hex", item.get("value")) if type(item) is dict else item
            )
            result[metric_id] = _hex_or_number(source)
    else:
        raise ValueError("candidate objectives are absent")
    return result


def _normalize(
    candidate: dict[str, Any],
    axes: list[dict[str, Any]],
) -> tuple[float, ...]:
    objectives = _objective_map(candidate)
    point: list[float] = []
    for axis in axes:
        metric_id = axis["metric_id"]
        value = objectives[metric_id]
        ideal = float.fromhex(axis["ideal_hex"])
        reference = float.fromhex(axis["reference_hex"])
        if axis["goal"] == "min":
            point.append((value - ideal) / (reference - ideal))
        elif axis["goal"] == "max":
            point.append((ideal - value) / (ideal - reference))
        else:
            raise ValueError("unknown affine objective goal")
    return tuple(point)


def _hypervolume(points: list[tuple[float, ...]], dimension: int) -> float:
    if dimension == 2:
        return hypervolume_2d(points, (1.0, 1.0))  # type: ignore[arg-type]
    if dimension == 3:
        return hypervolume_3d(points, (1.0, 1.0, 1.0))  # type: ignore[arg-type]
    raise ValueError("systematic analysis supports affine 2-D or 3-D endpoints")


def _physical_evaluation_trajectory(
    engine: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    *,
    seed_unique_evaluations: int,
    seed_archive_points: list[tuple[float, ...]],
    allowed_generations: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Return anytime HV after the seed batch and each later physical call.

    ``allowed_generations`` separates the durable sealed endpoint from work
    completed inside a generation that later failed.  The latter is useful
    behavioral evidence, but it must not silently inflate a censored campaign
    endpoint or its authenticated evaluation budget.
    """

    dimension = len(axes)
    if type(seed_unique_evaluations) is not int or seed_unique_evaluations <= 0:
        raise ValueError("seed_unique_evaluations must be positive")
    seen_phenotypes: set[str] = set()
    scored_points = list(seed_archive_points)
    rows: list[dict[str, Any]] = [
        {
            "physical_evaluation": seed_unique_evaluations,
            "physical_evaluation_span": seed_unique_evaluations,
            "candidate_id": None,
            "generation": 0,
            "label": "seed_batch_complete",
            "valid": None,
            "phenotype_identity_sha256": None,
            "hypervolume": _hypervolume(scored_points, dimension),
        }
    ]
    cache_events = [
        event for event in engine if event.get("event_type") == "evaluation_cache_event"
    ]
    cache_event_types: dict[str, set[str]] = {}
    for event in cache_events:
        phenotype = event.get("phenotype_identity")
        phenotype_sha256 = (
            phenotype.get("identity_sha256") if type(phenotype) is dict else None
        )
        cache_event_type = event.get("cache_event_type")
        if (
            type(phenotype_sha256) is not str
            or len(phenotype_sha256) != 64
            or type(cache_event_type) is not str
        ):
            raise ValueError("evaluation-cache event is malformed")
        cache_event_types.setdefault(phenotype_sha256, set()).add(cache_event_type)
    physical_phenotypes = {
        phenotype_sha256
        for phenotype_sha256, event_types in cache_event_types.items()
        if "miss" in event_types
    }
    for event in engine:
        if event.get("event_type") != "candidate_evaluated":
            continue
        generation_match = _GENERATION.search(str(event.get("label", "")))
        generation = 0 if generation_match is None else int(generation_match.group(1))
        if allowed_generations is not None and generation not in allowed_generations:
            continue
        detailed = event.get("detailed_evaluation")
        phenotype = detailed.get("phenotype") if type(detailed) is dict else None
        phenotype_sha256 = (
            phenotype.get("identity_sha256") if type(phenotype) is dict else None
        )
        if type(phenotype_sha256) is not str or len(phenotype_sha256) != 64:
            raise ValueError("evaluated candidate lacks a typed phenotype identity")
        if cache_events:
            if phenotype_sha256 not in cache_event_types:
                raise ValueError(
                    "evaluated candidate lacks an authenticated cache disposition"
                )
            if phenotype_sha256 not in physical_phenotypes:
                continue
        if phenotype_sha256 in seen_phenotypes:
            continue
        seen_phenotypes.add(phenotype_sha256)
        if event.get("valid") is True:
            scored_points.append(_normalize(event, axes))
        else:
            failure = detailed.get("failure") if type(detailed) is dict else None
            if type(failure) is not dict or failure.get("category") != "candidate":
                raise ValueError(
                    "invalid physical evaluation lacks candidate-infeasibility evidence"
                )
        rows.append(
            {
                "physical_evaluation": rows[-1]["physical_evaluation"] + 1,
                "physical_evaluation_span": 1,
                "candidate_id": str(event["candidate_id"]),
                "generation": generation,
                "label": str(event.get("label", "")),
                "valid": event.get("valid") is True,
                "phenotype_identity_sha256": phenotype_sha256,
                "hypervolume": _hypervolume(scored_points, dimension),
            }
        )
    return rows


def _campaign_trace(
    run_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    events = [_unwrap(value) for value in _jsonl(run_dir / "campaign_events.jsonl")]
    cutoffs = [
        value for value in events if value.get("kind") == "archive_utility_frozen"
    ]
    stages = [value for value in events if value.get("kind") == "stage_sealed"]
    if not cutoffs:
        raise ValueError("campaign trace lacks an archive cutoff")
    stages.sort(key=lambda value: value["payload"]["stage_receipt"]["generation"])
    cutoffs.sort(key=lambda value: value["payload"]["archive_utility"]["generation"])
    cutoff_generations = [
        int(value["payload"]["archive_utility"]["generation"]) for value in cutoffs
    ]
    stage_generations = [
        int(value["payload"]["stage_receipt"]["generation"]) for value in stages
    ]
    if cutoff_generations[: len(stage_generations)] != stage_generations:
        raise ValueError("campaign archive cutoffs do not align with sealed stages")
    if len(cutoffs) not in (len(stages), len(stages) + 1):
        raise ValueError("campaign trace has more than one unsealed archive cutoff")
    if stages:
        final_front = stages[-1]["payload"]["stage_receipt"]["result"]["archive_after"][
            "front_candidates"
        ]
    else:
        failures = [
            value for value in events if value.get("kind") == "execution_failed"
        ]
        if len(failures) != 1 or len(cutoffs) != 1:
            raise ValueError(
                "a stage-free trace requires one failure and one seed cutoff"
            )
        final_front = cutoffs[0]["payload"]["archive_cutoff"]["archive"][
            "front_candidates"
        ]
    return cutoffs, stages, final_front


def _failure_endpoint(
    campaign_rows: list[dict[str, Any]],
    queue_rows: list[dict[str, Any]],
    cutoffs: list[dict[str, Any]],
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project a censored campaign endpoint without inventing final quality.

    Failed long runs are first-class behavioral observations.  The last sealed
    archive remains a valid anytime endpoint, while the failed generation and
    retry sequence explain why it must not be reported as a completed result.
    """

    events = [_unwrap(value) for value in campaign_rows]
    failures = [value for value in events if value.get("kind") == "execution_failed"]
    failure = failures[-1] if failures else None
    stage_generations = {
        int(value["payload"]["stage_receipt"]["generation"]) for value in stages
    }
    cutoff_generations = {
        int(value["payload"]["archive_utility"]["generation"]) for value in cutoffs
    }
    unsealed = sorted(cutoff_generations - stage_generations)

    queue = [_unwrap(value) for value in queue_rows]
    task_status_counts = Counter(str(value.get("status", "unknown")) for value in queue)
    attempt_failure_kinds: Counter[str] = Counter()
    attempt_reason_counts: Counter[str] = Counter()
    request_variant_counts: Counter[str] = Counter()
    failed_tasks: list[dict[str, Any]] = []
    for task in queue:
        attempts = task.get("attempts", [])
        if type(attempts) is not list or any(
            type(value) is not dict for value in attempts
        ):
            raise TypeError("queue task attempts must contain objects")
        normalized_attempts: list[dict[str, Any]] = []
        for attempt in attempts:
            raw_failure = attempt.get("failure")
            attempt_failure = raw_failure if type(raw_failure) is dict else {}
            classification = attempt.get("classification")
            classification = classification if type(classification) is dict else {}
            request_evidence = attempt.get("request_evidence")
            request_evidence = (
                request_evidence if type(request_evidence) is dict else {}
            )
            kind = attempt_failure.get("kind")
            reason = classification.get("reason")
            variant = request_evidence.get("variant")
            if type(kind) is str:
                attempt_failure_kinds[kind] += 1
            if type(reason) is str:
                attempt_reason_counts[reason] += 1
            if type(variant) is str:
                request_variant_counts[variant] += 1
            normalized_attempts.append(
                {
                    "attempt_number": attempt.get("attempt_number"),
                    "status": attempt.get("status"),
                    "failure_kind": kind,
                    "classification_reason": reason,
                    "request_variant": variant,
                    "status_code": attempt_failure.get("status_code"),
                    "output_failure_mode": attempt_failure.get("output_failure_mode"),
                    "validation_reason_codes": sorted(
                        {
                            str(value["reason_code"])
                            for value in attempt_failure.get("validation_issues", [])
                            if type(value) is dict
                            and type(value.get("reason_code")) is str
                        }
                    ),
                    "will_retry": attempt.get("will_retry"),
                }
            )
        if task.get("status") != "succeeded":
            failed_tasks.append(
                {
                    "task_id": task.get("task_id"),
                    "status": task.get("status"),
                    "cancellation_reason": task.get("cancellation_reason"),
                    "attempts": normalized_attempts,
                }
            )

    payload = failure.get("payload", {}) if type(failure) is dict else {}
    counters = payload.get("counters") if type(payload) is dict else None
    return {
        "campaign_failed": failure is not None,
        "campaign_failure_type": (
            payload.get("failure_type") if type(payload) is dict else None
        ),
        "campaign_failure_digest_sha256": (
            payload.get("failure_digest_sha256") if type(payload) is dict else None
        ),
        "failure_counters": counters if type(counters) is dict else None,
        "last_sealed_generation": max(stage_generations, default=0),
        "unsealed_cutoff_generations": unsealed,
        "failed_generation": unsealed[0] if len(unsealed) == 1 else None,
        "queue_task_status_counts": dict(sorted(task_status_counts.items())),
        "queue_attempt_failure_kind_counts": dict(
            sorted(attempt_failure_kinds.items())
        ),
        "queue_attempt_classification_reason_counts": dict(
            sorted(attempt_reason_counts.items())
        ),
        "queue_request_variant_counts": dict(sorted(request_variant_counts.items())),
        "failed_queue_tasks": failed_tasks,
    }


def _authenticated_evaluation_accounting(
    run_dir: Path,
    summary: dict[str, Any],
) -> dict[str, int]:
    """Normalize accounting from the common authenticated campaign event law.

    Workload summaries intentionally retain domain-owned schemas.  The campaign
    event stream is the invariant integration surface: execution start seals
    seed accounting, and the last sealed stage records final counters.
    """

    events = [_unwrap(value) for value in _jsonl(run_dir / "campaign_events.jsonl")]
    starts = [value for value in events if value.get("kind") == "execution_started"]
    stages = [value for value in events if value.get("kind") == "stage_sealed"]
    if len(starts) != 1:
        raise ValueError("campaign accounting requires one start")
    stages.sort(key=lambda value: value["payload"]["stage_receipt"]["generation"])
    seed = starts[0]["payload"]["start_receipt"]["seed_accounting"]
    if stages:
        counters = stages[-1]["payload"]["counters"]
    else:
        failures = [
            value for value in events if value.get("kind") == "execution_failed"
        ]
        if (
            len(failures) != 1
            or type(failures[0]["payload"].get("counters")) is not dict
        ):
            raise ValueError("stage-free campaign accounting requires failure counters")
        counters = failures[0]["payload"]["counters"]
    candidate_occurrences = int(counters["candidate_occurrences"])
    unique_evaluations = int(counters["unique_evaluations"])
    result = {
        "candidate_occurrences": candidate_occurrences,
        "unique_evaluations": unique_evaluations,
        "cache_reuse_occurrences": candidate_occurrences - unique_evaluations,
        "seed_occurrences": int(seed["occurrences"]),
        "seed_unique_evaluations": int(seed["unique_evaluations"]),
    }
    if result["cache_reuse_occurrences"] < 0:
        raise ValueError("unique evaluations exceed candidate occurrences")
    summary_accounting = summary.get("evaluation_accounting")
    if type(summary_accounting) is dict:
        for key, observed in result.items():
            if key in summary_accounting and int(summary_accounting[key]) != observed:
                raise ValueError(
                    f"summary and authenticated campaign accounting disagree: {key}"
                )
    return result


def _provider_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = [_unwrap(value) for value in rows]
    responses = [
        value["response"] for value in records if type(value.get("response")) is dict
    ]
    attempts = [item for value in records for item in value.get("attempts", [])]
    costs = [Decimal(value["cost_usd"]) for value in responses]
    latencies = [int(value["latency_ns"]) for value in responses]
    return {
        "logical_calls": len(records),
        "successful_logical_calls": sum(
            value.get("status") == "succeeded" for value in records
        ),
        "failed_logical_calls": sum(
            value.get("status") == "attempts_exhausted" for value in records
        ),
        "cancelled_logical_calls": sum(
            value.get("status") == "cancelled" for value in records
        ),
        "logical_call_status_counts": dict(
            sorted(
                Counter(
                    str(value.get("status", "unknown")) for value in records
                ).items()
            )
        ),
        "physical_attempts": len(attempts),
        "retry_attempts": max(0, len(attempts) - len(records)),
        "input_tokens": sum(int(value["input_tokens"]) for value in responses),
        "output_tokens": sum(int(value["output_tokens"]) for value in responses),
        "reasoning_tokens": sum(int(value["reasoning_tokens"]) for value in responses),
        "cost_usd": str(sum(costs, Decimal(0))),
        "provider_latency_s": sum(latencies) / 1e9,
        "mean_provider_latency_s": (
            None if not latencies else sum(latencies) / len(latencies) / 1e9
        ),
        "requested_models": sorted({value["requested_model"] for value in responses}),
        "resolved_models": sorted({value["resolved_model"] for value in responses}),
        "resolved_providers": sorted(
            {value["resolved_provider"] for value in responses}
        ),
        "finish_reasons": sorted({value["finish_reason"] for value in responses}),
    }


def _action_forecast_information(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure how much discrimination the model puts into action forecasts.

    Forecast calibration is only observable after an action is evaluated.  This
    complementary diagnostic is available immediately from authenticated
    provider outputs and detects a different failure mode: a valid structured
    response that assigns nearly the same consequence distribution to every
    option.  All statistics are workload-neutral and retain the enum codes so
    runs using different metric scales remain comparable.
    """

    records = [
        _unwrap(value)
        for value in rows
        if _unwrap(value).get("operation") == "forecast_target_realization"
    ]
    effect_counts: Counter[str] = Counter()
    effect_sign_counts: Counter[str] = Counter()
    validity_counts: Counter[str] = Counter()
    uncertainty_counts: Counter[str] = Counter()
    signature_counts: Counter[tuple[str, ...]] = Counter()
    per_call: list[dict[str, Any]] = []
    action_count = 0
    metric_cell_count = 0
    asymmetric_uncertainty_count = 0
    all_zero_call_count = 0
    constant_effect_call_count = 0

    def require_rows(
        value: object,
        *,
        field: str,
        expected_action_count: int,
    ) -> list[list[str]]:
        if type(value) is not list or len(value) != expected_action_count:
            raise ValueError(f"{field} must have one row per forecast action")
        result: list[list[str]] = []
        for row in value:
            if (
                type(row) is not list
                or not row
                or any(type(item) is not str for item in row)
            ):
                raise ValueError(f"{field} must contain non-empty string rows")
            result.append(row)
        return result

    for record in records:
        typed = record.get("typed_output")
        if type(typed) is not dict:
            raise ValueError("action forecast output lacks a typed output object")
        validity = typed.get("probability_valid_codes")
        if (
            type(validity) is not list
            or not validity
            or any(type(item) is not str for item in validity)
        ):
            raise ValueError(
                "action forecast validity codes must be a non-empty string list"
            )
        call_action_count = len(validity)
        medians = require_rows(
            typed.get("median_effect_codes"),
            field="median_effect_codes",
            expected_action_count=call_action_count,
        )
        lower = require_rows(
            typed.get("lower_uncertainty_codes"),
            field="lower_uncertainty_codes",
            expected_action_count=call_action_count,
        )
        upper = require_rows(
            typed.get("upper_uncertainty_codes"),
            field="upper_uncertainty_codes",
            expected_action_count=call_action_count,
        )
        metric_count = len(medians[0])
        if any(
            len(row) != metric_count
            for matrix in (medians, lower, upper)
            for row in matrix
        ):
            raise ValueError("action forecast matrices have inconsistent dimensions")

        call_effects = [item for row in medians for item in row]
        call_lower = [item for row in lower for item in row]
        call_upper = [item for row in upper for item in row]
        effect_counts.update(call_effects)
        validity_counts.update(validity)
        uncertainty_counts.update(call_lower)
        uncertainty_counts.update(call_upper)
        effect_sign_counts.update(
            (
                "zero"
                if item == "z"
                else ("negative" if item.startswith("n") else "positive")
            )
            for item in call_effects
        )
        asymmetric_uncertainty_count += sum(
            low != high
            for low_row, high_row in zip(lower, upper, strict=True)
            for low, high in zip(low_row, high_row, strict=True)
        )
        for index, effect_row in enumerate(medians):
            signature_counts[
                (
                    validity[index],
                    *effect_row,
                    *lower[index],
                    *upper[index],
                )
            ] += 1

        all_zero = all(item == "z" for item in call_effects)
        constant = len(set(call_effects)) == 1
        all_zero_call_count += all_zero
        constant_effect_call_count += constant
        action_count += call_action_count
        metric_cell_count += len(call_effects)
        per_call.append(
            {
                "call_id": record.get("call_id"),
                "action_count": call_action_count,
                "metric_count": metric_count,
                "distinct_effect_code_count": len(set(call_effects)),
                "zero_effect_cell_rate": sum(item == "z" for item in call_effects)
                / len(call_effects),
                "all_zero_effects": all_zero,
                "constant_effect_code": constant,
                "high_validity_action_rate": sum(item == "p0_95" for item in validity)
                / call_action_count,
            }
        )

    def entropy(counts: Counter[Any]) -> float | None:
        total = sum(counts.values())
        if total == 0:
            return None
        return -sum(
            (count / total) * math.log(count / total)
            for count in counts.values()
            if count
        )

    uncertainty_cell_count = 2 * metric_cell_count
    most_common_signature_count = max(signature_counts.values(), default=0)
    return {
        "schema_version": 1,
        "forecast_call_count": len(records),
        "action_count": action_count,
        "metric_cell_count": metric_cell_count,
        "effect_code_counts": dict(sorted(effect_counts.items())),
        "effect_sign_counts": dict(sorted(effect_sign_counts.items())),
        "distinct_effect_code_count": len(effect_counts),
        "effect_entropy_nats": entropy(effect_counts),
        "zero_effect_cell_rate": (
            None if metric_cell_count == 0 else effect_counts["z"] / metric_cell_count
        ),
        "all_zero_call_rate": (
            None if not records else all_zero_call_count / len(records)
        ),
        "constant_effect_call_rate": (
            None if not records else constant_effect_call_count / len(records)
        ),
        "validity_code_counts": dict(sorted(validity_counts.items())),
        "validity_entropy_nats": entropy(validity_counts),
        "high_validity_action_rate": (
            None if action_count == 0 else validity_counts["p0_95"] / action_count
        ),
        "uncertainty_code_counts": dict(sorted(uncertainty_counts.items())),
        "zero_uncertainty_cell_rate": (
            None
            if uncertainty_cell_count == 0
            else uncertainty_counts["u0"] / uncertainty_cell_count
        ),
        "asymmetric_uncertainty_cell_rate": (
            None
            if metric_cell_count == 0
            else asymmetric_uncertainty_count / metric_cell_count
        ),
        "distinct_full_action_signature_count": len(signature_counts),
        "full_action_signature_entropy_nats": entropy(signature_counts),
        "most_common_full_action_signature_rate": (
            None if action_count == 0 else most_common_signature_count / action_count
        ),
        "per_call": per_call,
    }


def _reflection_schema(summary: dict[str, Any]) -> dict[str, Any]:
    versions: list[int] = []
    decimal_effects = 0
    effect_count = 0
    for record in summary.get("reflection_records", []):
        learning = record.get("campaign_reflection_learning", {})
        for evidence in learning.get("empirical_evidence", []):
            versions.append(int(evidence["fact_schema_version"]))
            for effect in evidence.get("facts", {}).get("observed_metric_effects", []):
                effect_count += 1
                decimal_effects += "delta_decimal" in effect
    return {
        "fact_schema_versions": sorted(set(versions)),
        "observed_metric_effect_count": effect_count,
        "observed_metric_effects_with_decimal": decimal_effects,
        "decimal_magnitude_prompt_contract": bool(versions) and min(versions) >= 2,
        "hex_exponent_interpretation_risk": bool(versions) and min(versions) < 2,
    }


def _matched_memory_controls(stages: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize arm-aware memory assays from the generic stage contract."""

    rows: list[dict[str, Any]] = []
    for event in stages:
        receipt = event["payload"]["stage_receipt"]
        result = receipt["result"]
        closed_loop = result.get("closed_loop_learning")
        evidence = closed_loop.get("evidence") if type(closed_loop) is dict else None
        preparation = (
            evidence.get("generation_audit_preparation")
            if type(evidence) is dict
            else None
        )
        raw_outcomes = (
            preparation.get("matched_memory_control_outcomes", [])
            if type(preparation) is dict
            else []
        )
        if type(raw_outcomes) is not list or any(
            type(value) is not dict for value in raw_outcomes
        ):
            raise TypeError("matched memory control outcomes must contain objects")
        for outcome in raw_outcomes:
            active = _hex_or_number(outcome["active_wave_reward_hex"])
            neutral = _hex_or_number(outcome["neutral_wave_reward_hex"])
            observed = _hex_or_number(outcome["observed_active_minus_neutral_hex"])
            if not math.isclose(
                observed,
                active - neutral,
                rel_tol=0.0,
                abs_tol=64 * math.ulp(max(1.0, abs(active), abs(neutral))),
            ):
                raise ValueError("matched memory outcome arithmetic is inconsistent")
            reference = outcome.get("reference")
            if type(reference) is not dict:
                raise ValueError("matched memory outcome lacks an insight reference")
            rows.append(
                {
                    "generation": int(outcome["generation"]),
                    "insight_id": str(reference["insight_id"]),
                    "insight_version": int(reference["version"]),
                    "active_result_receipt_sha256": str(
                        outcome["active_result_receipt_sha256"]
                    ),
                    "neutral_result_receipt_sha256": str(
                        outcome["neutral_result_receipt_sha256"]
                    ),
                    "active_wave_reward": active,
                    "neutral_wave_reward": neutral,
                    "observed_active_minus_neutral": observed,
                    "active_better_than_neutral": observed > 0.0,
                    "single_block_card_effect_identified": bool(
                        outcome.get("single_block_card_effect_identified")
                    ),
                    "online_score_update_allowed": bool(
                        outcome.get("online_score_update_allowed")
                    ),
                    "analysis_scope": str(outcome.get("analysis_scope", "unknown")),
                    "outcome_sha256": str(outcome["outcome_sha256"]),
                }
            )
    return {
        "matched_memory_control_block_count": len(rows),
        "matched_memory_active_better_count": sum(
            value["active_better_than_neutral"] for value in rows
        ),
        "matched_memory_identified_effect_count": sum(
            value["single_block_card_effect_identified"] for value in rows
        ),
        "matched_memory_online_score_update_allowed_count": sum(
            value["online_score_update_allowed"] for value in rows
        ),
        "matched_memory_total_active_reward": sum(
            value["active_wave_reward"] for value in rows
        ),
        "matched_memory_total_neutral_reward": sum(
            value["neutral_wave_reward"] for value in rows
        ),
        "matched_memory_total_active_minus_neutral": sum(
            value["observed_active_minus_neutral"] for value in rows
        ),
        "matched_memory_control_rows": rows,
        "matched_memory_causal_claim_allowed": False,
    }


def _memory_lifecycle(
    summary: dict[str, Any],
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recover workload-neutral memory testing, credit, and falsification facts.

    Utility credit and semantic validity are intentionally separate endpoints:
    a memory assignment can accompany a useful candidate wave while the card's
    predictive claim is contradicted by newly observed evidence.
    """

    raw_memory = summary.get("memory", {})
    memory = raw_memory if type(raw_memory) is dict else {}
    raw_entries = memory.get("reflected_entries", [])
    entries = raw_entries if type(raw_entries) is list else []
    entry_source = "summary.memory.reflected_entries"
    if not entries:
        # Newer delayed-identifiable campaigns intentionally keep reflected
        # insights quarantined from normal retrieval and expose them only via
        # an explicit future-testing lane.  Those runs do not project the
        # legacy ``summary.memory`` object, but their typed reflection records
        # remain authoritative evidence that the insights were generated.
        # Preserve the lifecycle distinction: test-only cards are counted as
        # reflected/quarantined, never as normally retrievable memory.
        raw_reflections = summary.get("reflection_records", [])
        reflections = raw_reflections if type(raw_reflections) is list else []
        synthetic_entries: list[dict[str, Any]] = []
        for record in reflections:
            if type(record) is not dict:
                raise TypeError("reflection record must be an object")
            raw_insights = record.get("insights")
            if raw_insights is None:
                declared_count = int(record.get("insight_count", 0))
                insights = [{} for _ in range(declared_count)]
            elif type(raw_insights) is list:
                insights = raw_insights
                if any(type(value) is not dict for value in insights):
                    raise TypeError("reflection record insights must contain objects")
                declared_count = int(record.get("insight_count", len(insights)))
                if declared_count != len(insights):
                    raise ValueError("reflection insight count disagrees with insights")
            else:
                raise TypeError("reflection record insights must be a list or null")
            state = "quarantined" if record.get("quarantined") is True else "unknown"
            synthetic_entries.extend(
                {"lifecycle_state": state, "retrievable": False} for _ in insights
            )
        if synthetic_entries:
            entries = synthetic_entries
            entry_source = "summary.reflection_records"
    lifecycle_states: Counter[str] = Counter()
    retrievable_count = 0
    for entry in entries:
        if type(entry) is not dict:
            raise TypeError("reflected memory entry must be an object")
        lifecycle_states[str(entry.get("lifecycle_state", "unknown"))] += 1
        retrievable_count += entry.get("retrievable") is True

    raw_scores = memory.get("score_evidence_postrun_diagnostic_only", [])
    scores = raw_scores if type(raw_scores) is list else []
    if any(type(value) is not dict for value in scores):
        raise TypeError("post-run memory score evidence must contain objects")

    audit_verdicts: Counter[str] = Counter()
    lifecycle_requests: Counter[str] = Counter()
    publication_scopes: Counter[str] = Counter()
    rewards: list[float] = []
    credit_count = 0
    credit_by_generation: list[dict[str, Any]] = []
    for event in stages:
        result = event["payload"]["stage_receipt"]["result"]
        credit_batch = result.get("memory_credit_batch")
        if type(credit_batch) is dict:
            credits = credit_batch.get("credits", [])
            if type(credits) is not list or any(
                type(value) is not dict for value in credits
            ):
                raise TypeError("memory credit batch must contain object credits")
            declared_count = int(credit_batch.get("credit_count", len(credits)))
            if declared_count != len(credits):
                raise ValueError("memory credit batch count disagrees with credits")
            credit_count += len(credits)
            batch_rewards = [_hex_or_number(value["reward_hex"]) for value in credits]
            rewards.extend(batch_rewards)
            credit_by_generation.append(
                {
                    "generation": int(credit_batch["generation"]),
                    "credit_count": len(credits),
                    "positive_wave_reward_count": sum(
                        value > 0.0 for value in batch_rewards
                    ),
                    "zero_wave_reward_count": sum(
                        value == 0.0 for value in batch_rewards
                    ),
                    "negative_wave_reward_count": sum(
                        value < 0.0 for value in batch_rewards
                    ),
                    "total_wave_reward": sum(batch_rewards),
                }
            )
            publication_scopes[
                str(credit_batch.get("publication_scope", "unknown"))
            ] += 1

        closed_loop = result.get("closed_loop_learning")
        evidence = closed_loop.get("evidence") if type(closed_loop) is dict else None
        if type(evidence) is not dict:
            continue
        coordinator = evidence.get("coordinator_preparation")
        if type(coordinator) is dict:
            requests = coordinator.get("lifecycle_requests", [])
            if type(requests) is not list or any(
                type(value) is not dict for value in requests
            ):
                raise TypeError("memory lifecycle requests must contain objects")
            lifecycle_requests.update(
                str(value.get("new_state", "unknown")) for value in requests
            )
        audit_preparation = evidence.get("generation_audit_preparation")
        projection = (
            audit_preparation.get("projection")
            if type(audit_preparation) is dict
            else None
        )
        audits = projection.get("audits", []) if type(projection) is dict else []
        if type(audits) is not list or any(type(value) is not dict for value in audits):
            raise TypeError("semantic audit projection must contain objects")
        for audit in audits:
            receipt = audit.get("receipt")
            if type(receipt) is dict:
                audit_verdicts[str(receipt.get("verdict", "unknown"))] += 1

    trial_count = memory.get("trial_count")
    return {
        "reflected_entry_count": len(entries),
        "reflected_entry_source": entry_source,
        "reflected_retrievable_count": retrievable_count,
        "reflected_lifecycle_state_counts": dict(sorted(lifecycle_states.items())),
        "memory_trial_count": (
            None if type(trial_count) is not int else int(trial_count)
        ),
        "adaptive_score_consumption": memory.get("adaptive_score_consumption"),
        "causal_claim_allowed": memory.get("causal_claim_allowed"),
        "postrun_score_entry_count": len(scores),
        "postrun_identified_score_count": sum(
            value.get("identified") is True for value in scores
        ),
        "memory_assignment_credit_count": credit_count,
        "memory_assignment_positive_wave_reward_count": sum(
            value > 0.0 for value in rewards
        ),
        "memory_assignment_zero_wave_reward_count": sum(
            value == 0.0 for value in rewards
        ),
        "memory_assignment_negative_wave_reward_count": sum(
            value < 0.0 for value in rewards
        ),
        "memory_assignment_total_wave_reward": sum(rewards),
        "memory_assignment_credit_by_generation": sorted(
            credit_by_generation, key=lambda value: value["generation"]
        ),
        "memory_credit_publication_scope_counts": dict(
            sorted(publication_scopes.items())
        ),
        "semantic_audit_verdict_counts": dict(sorted(audit_verdicts.items())),
        "lifecycle_request_state_counts": dict(sorted(lifecycle_requests.items())),
        **_advisory_memory_exposure(summary),
        **_matched_memory_controls(stages),
        **_memory_action_performance(summary, stages),
    }


def _advisory_memory_exposure(summary: dict[str, Any]) -> dict[str, Any]:
    """Normalize test-only memory exposures without inventing causal credit."""

    raw_waves = summary.get("wave_records", [])
    waves = raw_waves if type(raw_waves) is list else []
    statuses: Counter[str] = Counter()
    lane_count = 0
    selected_card_count = 0
    exact_parent_match_count = 0
    exact_replay_authorized_count = 0
    for wave in waves:
        if type(wave) is not dict:
            raise TypeError("summary wave record must be an object")
        diagnostic = wave.get("diagnostic")
        if type(diagnostic) is not dict:
            continue
        status = diagnostic.get("status")
        if type(status) is not str or "advisory" not in status:
            continue
        lane_count += 1
        statuses[status] += 1
        resolution = diagnostic.get("lane_support_resolution")
        if (
            type(resolution) is dict
            and type(resolution.get("selected_card_key")) is str
        ):
            selected_card_count += 1
        transfer = diagnostic.get("memory_context_transfer_assessment")
        if type(transfer) is dict:
            exact_parent_match_count += (
                transfer.get("exact_source_parent_match") is True
            )
            exact_replay_authorized_count += (
                transfer.get("exact_action_replay_authorized") is True
            )
    return {
        "advisory_memory_lane_count": lane_count,
        "advisory_memory_selected_card_count": selected_card_count,
        "advisory_memory_exact_parent_match_count": exact_parent_match_count,
        "advisory_memory_exact_replay_authorized_count": (
            exact_replay_authorized_count
        ),
        "advisory_memory_status_counts": dict(sorted(statuses.items())),
        "advisory_memory_causal_claim_allowed": False,
    }


def _memory_action_performance(
    summary: dict[str, Any],
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Separate supported-action performance from whole-wave memory credit.

    A prompt-visible card can support one evaluated action while an uncited
    member of the same wave creates all frontier gain.  It can also recommend
    an exact option that another prompt-wide lane independently selects.  Both
    facts are post-treatment diagnostics, not causal effects, but they prevent
    us from interpreting a positive whole-wave reward as evidence that the
    cited card or action caused that reward.
    """

    raw_summary_waves = summary.get("wave_records", [])
    summary_waves = raw_summary_waves if type(raw_summary_waves) is list else []
    advisory_request_sha256s: set[str] = set()
    for wave in summary_waves:
        if type(wave) is not dict:
            raise TypeError("summary wave record must be an object")
        diagnostic = wave.get("diagnostic")
        status = diagnostic.get("status") if type(diagnostic) is dict else None
        request_sha256 = wave.get("selection_request_sha256")
        if type(status) is str and "advisory" in status and type(request_sha256) is str:
            advisory_request_sha256s.add(request_sha256)

    rows: list[dict[str, Any]] = []
    for event in stages:
        result = event["payload"]["stage_receipt"]["result"]
        generation = int(event["payload"]["stage_receipt"]["generation"])
        matched_rows = _matched_memory_controls([event])["matched_memory_control_rows"]
        matched_rewards: dict[str, tuple[float, str]] = {}
        for matched in matched_rows:
            matched_rewards[str(matched["active_result_receipt_sha256"])] = (
                float(matched["active_wave_reward"]),
                "active",
            )
            matched_rewards[str(matched["neutral_result_receipt_sha256"])] = (
                float(matched["neutral_wave_reward"]),
                "neutral",
            )
        raw_waves = result.get("portfolio_wave_receipts", [])
        waves = raw_waves if type(raw_waves) is list else []
        normalized_waves: list[dict[str, Any]] = []
        for wave in waves:
            if type(wave) is not dict:
                continue
            receipt_sha256 = str(wave.get("receipt_sha256", ""))
            memory_credit = wave.get("memory_credit")
            if type(memory_credit) is dict:
                joint_wave_reward = _hex_or_number(memory_credit["reward_hex"])
                reward_source = "legacy_memory_credit"
                assay_arm = "legacy_assigned"
            elif receipt_sha256 in matched_rewards:
                joint_wave_reward, assay_arm = matched_rewards[receipt_sha256]
                reward_source = "matched_memory_control_outcome"
            else:
                if str(wave.get("request_sha256", "")) not in (
                    advisory_request_sha256s
                ):
                    continue
                joint_wave_reward = None
                reward_source = "advisory_exposure_no_wave_credit"
                assay_arm = "optimization_advisory_exposure"
            members = wave.get("members", [])
            attributions = wave.get("action_attributions", [])
            if (
                type(members) is not list
                or type(attributions) is not list
                or len(members) != len(attributions)
            ):
                raise ValueError(
                    "memory action performance requires a complete attribution join"
                )
            member_by_candidate = {
                value["candidate_id"]: value
                for value in members
                if type(value) is dict and type(value.get("candidate_id")) is str
            }
            if len(member_by_candidate) != len(members):
                raise ValueError("memory wave members lack unique candidate IDs")
            actions: list[dict[str, Any]] = []
            for attribution in attributions:
                if type(attribution) is not dict:
                    raise TypeError("memory action attribution must be an object")
                candidate_id = attribution.get("candidate_id")
                member = member_by_candidate.get(candidate_id)
                selected = attribution.get("selected_member")
                supporting = attribution.get("supporting_cards", [])
                if (
                    type(member) is not dict
                    or type(selected) is not dict
                    or type(supporting) is not list
                ):
                    raise ValueError("memory action attribution join is incomplete")
                actions.append(
                    {
                        "candidate_id": candidate_id,
                        "rank": int(selected["rank"]),
                        "option_id": str(selected["option_id"]),
                        "option_identity_sha256": str(
                            selected["option_identity_sha256"]
                        ),
                        "supporting_cards": supporting,
                        "engine_reward": _hex_or_number(
                            member.get("engine_reward_hex", member.get("reward_hex"))
                        ),
                        "better_than_any_parent": bool(
                            member.get("better_than_any_parent")
                        ),
                        "dominates_any_parent": bool(
                            member.get("dominates_any_parent")
                        ),
                    }
                )
            if joint_wave_reward is None and not any(
                action["supporting_cards"] for action in actions
            ):
                continue
            normalized_waves.append(
                {
                    "request_sha256": str(wave["request_sha256"]),
                    "parent_candidate_id": str(wave["parent_candidate_id"]),
                    "joint_wave_reward": joint_wave_reward,
                    "reward_source": reward_source,
                    "assay_arm": assay_arm,
                    "actions": actions,
                }
            )

        for lane_index, wave in enumerate(normalized_waves):
            other_actions = [
                action
                for other_index, other in enumerate(normalized_waves)
                if other_index != lane_index
                for action in other["actions"]
            ]
            supported_actions = [
                action for action in wave["actions"] if action["supporting_cards"]
            ]
            unsupported_positive = [
                action
                for action in wave["actions"]
                if not action["supporting_cards"] and action["engine_reward"] > 0.0
            ]
            for action in supported_actions:
                spillovers = [
                    other
                    for other in other_actions
                    if other["option_id"] == action["option_id"]
                ]
                for card in action["supporting_cards"]:
                    if (
                        type(card) is not dict
                        or type(card.get("reference")) is not dict
                    ):
                        raise ValueError("supporting card lacks an exact reference")
                    reference = card["reference"]
                    card_reward = float(action["engine_reward"])
                    raw_lane_reward = wave["joint_wave_reward"]
                    lane_reward = (
                        None if raw_lane_reward is None else float(raw_lane_reward)
                    )
                    rows.append(
                        {
                            "generation": generation,
                            "request_sha256": wave["request_sha256"],
                            "parent_candidate_id": wave["parent_candidate_id"],
                            "card_key": str(card["card_key"]),
                            "insight_id": str(reference["insight_id"]),
                            "insight_version": int(reference["version"]),
                            "candidate_id": action["candidate_id"],
                            "rank": action["rank"],
                            "option_id": action["option_id"],
                            "option_identity_sha256": action["option_identity_sha256"],
                            "candidate_engine_reward": card_reward,
                            "candidate_positive_reward": card_reward > 0.0,
                            "candidate_better_than_parent": action[
                                "better_than_any_parent"
                            ],
                            "candidate_dominates_parent": action[
                                "dominates_any_parent"
                            ],
                            "joint_wave_reward": lane_reward,
                            "joint_wave_reward_source": wave["reward_source"],
                            "assay_arm": wave["assay_arm"],
                            "joint_wave_positive_reward": (
                                None if lane_reward is None else lane_reward > 0.0
                            ),
                            "unsupported_positive_candidate_count": len(
                                unsupported_positive
                            ),
                            "positive_wave_with_nonpositive_supported_action": (
                                lane_reward is not None
                                and lane_reward > 0.0
                                and card_reward <= 0.0
                            ),
                            "cross_lane_same_option_count": len(spillovers),
                            "cross_lane_same_option_candidate_ids": sorted(
                                str(value["candidate_id"]) for value in spillovers
                            ),
                            "cross_lane_action_spillover": bool(spillovers),
                            "attribution_scope": (
                                "post_treatment_descriptive_not_causal_credit"
                            ),
                        }
                    )
    return {
        "memory_supported_action_count": len(rows),
        "memory_supported_action_positive_reward_count": sum(
            value["candidate_positive_reward"] for value in rows
        ),
        "memory_positive_wave_nonpositive_supported_action_count": sum(
            value["positive_wave_with_nonpositive_supported_action"] for value in rows
        ),
        "memory_cross_lane_action_spillover_count": sum(
            value["cross_lane_action_spillover"] for value in rows
        ),
        "memory_action_performance_rows": rows,
        "memory_action_performance_causal_claim_allowed": False,
    }


def _evaluator_latency(engine: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize physical evaluator latency from generic detailed receipts."""

    values: list[float] = []
    for event in engine:
        if event.get("event_type") != "detailed_evaluation_completed":
            continue
        detailed = event.get("detailed_evaluation")
        timings = detailed.get("timings") if type(detailed) is dict else None
        raw = timings.get("total_wall_seconds") if type(timings) is dict else None
        if type(raw) not in (int, float) or isinstance(raw, bool):
            continue
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "physical evaluator latency must be finite and nonnegative"
            )
        values.append(value)
    return {
        "physical_evaluator_latency_count": len(values),
        "physical_evaluator_latency_min_s": None if not values else min(values),
        "physical_evaluator_latency_median_s": None if not values else median(values),
        "physical_evaluator_latency_mean_s": (
            None if not values else sum(values) / len(values)
        ),
        "physical_evaluator_latency_max_s": None if not values else max(values),
    }


def _observed_direction(parent: float, child: float) -> str:
    if child < parent:
        return "decrease"
    if child > parent:
        return "increase"
    return "unchanged"


def _forecast_confidence_label(value: float) -> str:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError("forecast confidence must be a finite canonical float")
    if not 0.0 <= value <= 1.0:
        raise ValueError("forecast confidence must lie in [0, 1]")
    if value >= 0.75:
        return "high"
    if value >= 0.4:
        return "medium"
    if value > 0.0:
        return "low"
    return "unknown"


def _selector_forecast_response_members(
    response: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalize legacy model slates and trusted all-action decisions.

    Outcome-conditioned selection has no second free-form selector response:
    the model emits numeric forecasts for every action and trusted code emits
    the evaluated K-set.  Its durable response therefore stores members under
    ``ranked_decision`` and continuous confidence in ``selected_forecasts``.
    This decoder preserves that distinction while presenting the common
    selected effect rows needed for realized forecast calibration.
    """

    if type(response) is not dict:
        raise TypeError("selector response projection must be an object")
    supplemental = response.get("supplemental_selector_audit")
    payload = supplemental.get("payload") if type(supplemental) is dict else None
    audit_kind = supplemental.get("audit_kind") if type(supplemental) is dict else None
    if audit_kind != "outcome_conditioned_expert_portfolio":
        original = (
            payload.get("original_k8_response") if type(payload) is dict else None
        )
        raw_members = (
            original.get("members")
            if type(original) is dict
            else response.get("members")
        )
        if type(raw_members) is not list or any(
            type(value) is not dict for value in raw_members
        ):
            raise ValueError("selector forecast response lacks object members")
        return raw_members

    ranked = response.get("ranked_decision")
    raw_members = ranked.get("members") if type(ranked) is dict else None
    if (
        type(payload) is not dict
        or type(raw_members) is not list
        or any(type(value) is not dict for value in raw_members)
    ):
        raise ValueError("outcome-conditioned response lacks ranked members")
    aliases = payload.get("metric_aliases")
    selected_forecasts = payload.get("selected_forecasts")
    if type(aliases) is not list or type(selected_forecasts) is not list:
        raise ValueError("outcome-conditioned response lacks forecast decoding")
    target_by_forecast: dict[str, str] = {}
    for alias in aliases:
        if (
            type(alias) is not dict
            or type(alias.get("forecast_metric_id")) is not str
            or type(alias.get("target_metric_id")) is not str
        ):
            raise ValueError("outcome-conditioned metric alias is malformed")
        target_by_forecast[str(alias["forecast_metric_id"])] = str(
            alias["target_metric_id"]
        )
    confidence_by_cell: dict[tuple[str, str], str] = {}
    for forecast in selected_forecasts:
        metrics = forecast.get("metric_forecasts") if type(forecast) is dict else None
        option_id = forecast.get("option_id") if type(forecast) is dict else None
        if type(option_id) is not str or type(metrics) is not list:
            raise ValueError("outcome-conditioned selected forecast is malformed")
        for metric in metrics:
            if (
                type(metric) is not dict
                or type(metric.get("metric_id")) is not str
                or type(metric.get("confidence_hex")) is not str
            ):
                raise ValueError("outcome-conditioned forecast metric is malformed")
            forecast_metric_id = str(metric["metric_id"])
            target_metric_id = target_by_forecast.get(forecast_metric_id)
            if target_metric_id is None:
                raise ValueError("selected forecast uses an unknown metric alias")
            confidence_by_cell[(option_id, target_metric_id)] = (
                _forecast_confidence_label(float.fromhex(metric["confidence_hex"]))
            )

    normalized: list[dict[str, Any]] = []
    for member in raw_members:
        option_id = member.get("option_id")
        effects = member.get("effect_predictions")
        if (
            type(option_id) is not str
            or type(effects) is not list
            or any(type(value) is not dict for value in effects)
        ):
            raise ValueError("outcome-conditioned ranked member is malformed")
        normalized_effects: list[dict[str, Any]] = []
        for effect in effects:
            metric_id = effect.get("metric_id")
            if type(metric_id) is not str:
                raise ValueError("ranked effect prediction lacks a metric ID")
            confidence = confidence_by_cell.get((option_id, metric_id))
            if confidence is None:
                raise ValueError("ranked effect lacks its selected confidence")
            normalized_effects.append({**effect, "confidence": confidence})
        normalized.append({**member, "effect_predictions": normalized_effects})
    if {str(value["option_id"]) for value in normalized} != {
        option_id for option_id, _ in confidence_by_cell
    }:
        raise ValueError("ranked decision and selected forecasts disagree")
    return normalized


def _forecast_calibration(
    stages: list[dict[str, Any]],
    engine: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score model-authored effect directions against realized parent contrasts."""

    # A seed can remain on the Pareto frontier and be selected as a parent in
    # any later portfolio generation.  Seed registrations carry the same
    # typed objective payload needed for an exact parent-child contrast, so
    # omitting them falsely censors otherwise scorable forecasts whenever an
    # evolved child is rooted directly at a seed.
    evaluated = {
        str(value["candidate_id"]): value
        for value in engine
        if value.get("event_type") in ("seed_registered", "candidate_evaluated")
        and type(value.get("candidate_id")) is str
    }
    rows: list[dict[str, Any]] = []
    member_count = 0
    unscorable_member_count = 0
    unscorable_missing_event_member_count = 0
    unscorable_invalid_candidate_member_count = 0
    unscorable_objective_payload_member_count = 0
    for event in stages:
        receipt = event["payload"]["stage_receipt"]
        generation = int(receipt["generation"])
        response_members_by_request: dict[str, dict[str, dict[str, Any]]] = {}
        for audit in receipt.get("selector_audits") or []:
            plaintext = audit.get("plaintext_audit")
            if type(plaintext) is not dict:
                raise ValueError("forecast calibration lacks selector plaintext")
            response = json.loads(plaintext["response_text"])
            if type(response) is not dict:
                raise TypeError("selector response projection must be an object")
            raw_members = _selector_forecast_response_members(response)
            members = {
                str(value["option_id"]): value
                for value in raw_members
                if type(value.get("option_id")) is str
            }
            if len(members) != len(raw_members):
                raise ValueError("selector forecast members lack unique option IDs")
            response_members_by_request[str(audit["request_sha256"])] = members

        raw_waves = receipt["result"].get("portfolio_wave_receipts", [])
        waves = raw_waves if type(raw_waves) is list else []
        for wave in waves:
            if type(wave) is not dict:
                raise TypeError("portfolio wave receipt must be an object")
            request_sha256 = str(wave["request_sha256"])
            forecasts = response_members_by_request.get(request_sha256)
            if forecasts is None:
                raise ValueError("evaluated wave lacks its selector forecast response")
            parent_id = str(wave["parent_candidate_id"])
            parent_event = evaluated.get(parent_id)
            raw_attributions = wave.get("action_attributions", [])
            if type(raw_attributions) is not list:
                raise TypeError("wave action attributions must be a list")
            for attribution in raw_attributions:
                if type(attribution) is not dict:
                    raise TypeError("wave action attribution must be an object")
                selected = attribution.get("selected_member")
                if type(selected) is not dict:
                    raise ValueError("wave attribution lacks its selected member")
                option_id = str(selected["option_id"])
                forecast = forecasts.get(option_id)
                if forecast is None:
                    raise ValueError("evaluated option lacks its model forecast")
                candidate_id = str(attribution["candidate_id"])
                child_event = evaluated.get(candidate_id)
                member_count += 1
                if parent_event is None or child_event is None:
                    unscorable_member_count += 1
                    unscorable_missing_event_member_count += 1
                    continue
                if (
                    parent_event.get("valid") is not True
                    or child_event.get("valid") is not True
                ):
                    unscorable_member_count += 1
                    unscorable_invalid_candidate_member_count += 1
                    continue
                try:
                    parent_objectives = _objective_map(parent_event)
                    child_objectives = _objective_map(child_event)
                except ValueError:
                    unscorable_member_count += 1
                    unscorable_objective_payload_member_count += 1
                    continue
                raw_predictions = forecast.get("effect_predictions", [])
                if type(raw_predictions) is not list or any(
                    type(value) is not dict for value in raw_predictions
                ):
                    raise TypeError("effect predictions must contain objects")
                for prediction in raw_predictions:
                    metric_id = str(prediction["metric_id"])
                    if (
                        metric_id not in parent_objectives
                        or metric_id not in child_objectives
                    ):
                        raise ValueError("forecast metric lacks a realized objective")
                    predicted = str(prediction["direction"])
                    actual = _observed_direction(
                        parent_objectives[metric_id], child_objectives[metric_id]
                    )
                    confidence = str(prediction.get("confidence", "unspecified"))
                    known = predicted != "unknown"
                    rows.append(
                        {
                            "generation": generation,
                            "request_sha256": request_sha256,
                            "parent_candidate_id": parent_id,
                            "candidate_id": candidate_id,
                            "option_id": option_id,
                            "metric_id": metric_id,
                            "confidence": confidence,
                            "predicted_direction": predicted,
                            "observed_direction": actual,
                            "known_direction_forecast": known,
                            "direction_correct": known and predicted == actual,
                            "adjudication_policy": "binary64_exact_sign_v1",
                        }
                    )

    known_rows = [value for value in rows if value["known_direction_forecast"]]
    high_rows = [value for value in known_rows if value["confidence"] == "high"]
    by_confidence: list[dict[str, Any]] = []
    for confidence in sorted({str(value["confidence"]) for value in rows}):
        members = [value for value in rows if value["confidence"] == confidence]
        known = [value for value in members if value["known_direction_forecast"]]
        by_confidence.append(
            {
                "confidence": confidence,
                "prediction_count": len(members),
                "known_direction_count": len(known),
                "unknown_direction_count": len(members) - len(known),
                "direction_accuracy": (
                    None
                    if not known
                    else sum(value["direction_correct"] for value in known) / len(known)
                ),
            }
        )
    by_generation: list[dict[str, Any]] = []
    for generation in sorted({int(value["generation"]) for value in rows}):
        members = [value for value in rows if value["generation"] == generation]
        known = [value for value in members if value["known_direction_forecast"]]
        by_generation.append(
            {
                "generation": generation,
                "prediction_count": len(members),
                "known_direction_count": len(known),
                "direction_accuracy": (
                    None
                    if not known
                    else sum(value["direction_correct"] for value in known) / len(known)
                ),
            }
        )
    return {
        "evaluated_forecast_member_count": member_count,
        "unscorable_forecast_member_count": unscorable_member_count,
        "unscorable_missing_event_member_count": (
            unscorable_missing_event_member_count
        ),
        "unscorable_invalid_candidate_member_count": (
            unscorable_invalid_candidate_member_count
        ),
        "unscorable_objective_payload_member_count": (
            unscorable_objective_payload_member_count
        ),
        "effect_prediction_count": len(rows),
        "known_direction_forecast_count": len(known_rows),
        "unknown_direction_forecast_count": len(rows) - len(known_rows),
        "unknown_direction_forecast_rate": (
            None if not rows else (len(rows) - len(known_rows)) / len(rows)
        ),
        "direction_accuracy": (
            None
            if not known_rows
            else sum(value["direction_correct"] for value in known_rows)
            / len(known_rows)
        ),
        "high_confidence_known_direction_count": len(high_rows),
        "high_confidence_direction_error_count": sum(
            not value["direction_correct"] for value in high_rows
        ),
        "high_confidence_direction_accuracy": (
            None
            if not high_rows
            else sum(value["direction_correct"] for value in high_rows) / len(high_rows)
        ),
        "by_confidence": by_confidence,
        "by_generation": by_generation,
        "rows": rows,
    }


def _prompt_machine_contract(prompt: str) -> dict[str, Any]:
    """Extract the one canonical JSON machine contract from a selector prompt."""

    if type(prompt) is not str:
        raise TypeError("selector request_text must be an exact string")
    for line in prompt.splitlines():
        if not line.startswith("{"):
            continue
        value = json.loads(line)
        if type(value) is dict and type(value.get("proposal_constraints")) is dict:
            return value
    raise ValueError("selector prompt lacks its machine contract")


def _prompt_finite_contract(prompt: str) -> dict[str, Any]:
    """Extract a finite-option contract without requiring a live-model suffix."""

    if type(prompt) is not str:
        raise TypeError("selector request_text must be an exact string")
    for line in prompt.splitlines():
        if not line.startswith("{"):
            continue
        value = json.loads(line)
        if type(value) is dict and type(value.get("ordered_options")) is list:
            return value
    raise ValueError("selector prompt lacks its finite-option contract")


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _rationale_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_.$-]+", value.lower()))


def _pairwise_token_jaccards(values: list[str]) -> list[float]:
    rows: list[float] = []
    token_sets = [_rationale_tokens(value) for value in values]
    for left_index, left in enumerate(token_sets):
        for right in token_sets[left_index + 1 :]:
            rows.append(_jaccard(left, right))
    return rows


def _frontier_target_projection(raw: object) -> dict[str, Any] | None:
    """Authenticate one workload-neutral lane target embedded in a prompt."""

    if raw is None:
        return None
    if type(raw) is not dict:
        raise TypeError("campaign frontier target must be an object")
    if raw.get("schema_version") != 1:
        raise ValueError("campaign frontier target has an unknown schema")
    allocator = raw.get("allocator")
    payload = raw.get("payload")
    if type(allocator) is not dict or type(payload) is not dict:
        raise ValueError("campaign frontier target lacks allocator or payload")
    frozen_payload = freeze_json(payload)
    if type(frozen_payload) is not FrozenJsonObject:
        raise TypeError("campaign frontier target payload must freeze to an object")
    target = CampaignPortfolioFrontierTarget(
        allocator_id=str(allocator["allocator_id"]),
        allocator_version=int(allocator["allocator_version"]),
        definition_sha256=str(allocator["definition_sha256"]),
        archive_utility_snapshot_sha256=str(raw["archive_utility_snapshot_sha256"]),
        lane_id=str(raw["lane_id"]),
        parent_configuration_sha256=str(raw["parent_configuration_sha256"]),
        direction_id=str(raw["direction_id"]),
        opportunity_rank=int(raw["opportunity_rank"]),
        payload=frozen_payload,
    )
    if target.to_record() != raw:
        raise ValueError("campaign frontier target authentication failed")

    cutoff = payload.get("epistemic_cutoff")
    direction = payload.get("target_direction")
    parent = payload.get("assigned_parent")
    instruction = payload.get("acquisition_instruction")
    scalar = payload.get("achievement_scalar")
    if any(
        type(value) is not dict
        for value in (cutoff, direction, parent, instruction, scalar)
    ):
        raise ValueError("campaign frontier target lacks generic target geometry")
    assert type(cutoff) is dict
    assert type(direction) is dict
    assert type(parent) is dict
    assert type(instruction) is dict
    assert type(scalar) is dict
    raw_weights = direction.get("normalized_weights_decimal")
    if type(raw_weights) is not list or not raw_weights:
        raise ValueError("campaign frontier target lacks direction weights")
    weights = tuple(float(str(value)) for value in raw_weights)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("campaign frontier target weights must be finite/nonnegative")
    if not any(value > 0 for value in weights):
        raise ValueError("campaign frontier target must activate an axis")
    if str(direction["direction_id"]) != target.direction_id:
        raise ValueError("campaign frontier target direction identity drifted")
    if int(direction["opportunity_rank"]) != target.opportunity_rank:
        raise ValueError("campaign frontier target opportunity rank drifted")
    if (
        str(cutoff["archive_utility_snapshot_sha256"])
        != target.archive_utility_snapshot_sha256
    ):
        raise ValueError("campaign frontier target archive cutoff drifted")

    raw_parent_point = parent.get("normalized_point_decimal")
    if type(raw_parent_point) is not list or len(raw_parent_point) != len(weights):
        raise ValueError("campaign frontier target lacks its normalized parent")
    parent_point = tuple(float(str(value)) for value in raw_parent_point)
    if any(not math.isfinite(value) for value in parent_point):
        raise ValueError("campaign frontier target parent must be finite")

    residual = payload.get("residual_frontier_cell")
    normalized_aspiration: tuple[float, ...] | None = None
    residual_potential: float | None = None
    residual_cell_sha256: str | None = None
    if residual is not None:
        if type(residual) is not dict:
            raise TypeError("residual frontier cell must be an object")
        raw_aspiration = residual.get("normalized_aspiration_point_decimal")
        if type(raw_aspiration) is not list or len(raw_aspiration) != len(weights):
            raise ValueError("residual frontier cell lacks its aspiration")
        normalized_aspiration = tuple(float(str(value)) for value in raw_aspiration)
        if any(not math.isfinite(value) for value in normalized_aspiration):
            raise ValueError("residual frontier aspiration must be finite")
        residual_potential = float(str(residual["potential_hypervolume_gain_decimal"]))
        if not math.isfinite(residual_potential) or residual_potential <= 0.0:
            raise ValueError("residual frontier potential must be positive")
        residual_cell_sha256 = str(residual["cell_sha256"])

    objective_space_target = payload.get("objective_space_target")
    raw_target_axes: list[dict[str, Any]] | None = None
    if objective_space_target is not None:
        if type(objective_space_target) is not dict:
            raise TypeError("objective-space target must be an object")
        raw_axes = objective_space_target.get("axes")
        if type(raw_axes) is not list or len(raw_axes) != len(weights):
            raise ValueError("objective-space target axes differ from its geometry")
        raw_target_axes = []
        for axis in raw_axes:
            if type(axis) is not dict:
                raise TypeError("objective-space target axis must be an object")
            row = {
                "metric_id": str(axis["metric_id"]),
                "goal": str(axis["goal"]),
                "ideal": float(str(axis["ideal_decimal"])),
                "reference": float(str(axis["reference_decimal"])),
                "parent_value": float(str(axis["parent_value_decimal"])),
                "aspiration_value": float(str(axis["aspiration_value_decimal"])),
                "signed_parent_to_aspiration_delta": float(
                    str(axis["signed_parent_to_aspiration_delta_decimal"])
                ),
                "improving_raw_delta_sign": str(axis["improving_raw_delta_sign"]),
            }
            if row["goal"] not in {"min", "max"}:
                raise ValueError("objective-space target axis has an unknown goal")
            if row["improving_raw_delta_sign"] != (
                "negative" if row["goal"] == "min" else "positive"
            ):
                raise ValueError("objective-space target improving sign drifted")
            if any(
                not math.isfinite(float(row[key]))
                for key in (
                    "ideal",
                    "reference",
                    "parent_value",
                    "aspiration_value",
                    "signed_parent_to_aspiration_delta",
                )
            ):
                raise ValueError("objective-space target axis must be finite")
            raw_target_axes.append(row)

    return {
        "target_sha256": target.target_sha256,
        "allocator_id": target.allocator_id,
        "allocator_version": target.allocator_version,
        "allocator_definition_sha256": target.definition_sha256,
        "archive_utility_snapshot_sha256": (target.archive_utility_snapshot_sha256),
        "lane_id": target.lane_id,
        "parent_configuration_sha256": target.parent_configuration_sha256,
        "direction_id": target.direction_id,
        "opportunity_rank": target.opportunity_rank,
        "normalized_weights": list(weights),
        "normalized_parent_point": list(parent_point),
        "normalized_aspiration_point": (
            None if normalized_aspiration is None else list(normalized_aspiration)
        ),
        "residual_cell_sha256": residual_cell_sha256,
        "residual_potential_hypervolume_gain": residual_potential,
        "objective_space_target_axes": raw_target_axes,
        "archive_best_achievement": float(
            str(direction["archive_best_achievement_decimal"])
        ),
        "opportunity_from_ideal": float(
            str(direction["opportunity_from_ideal_decimal"])
        ),
        "parent_achievement": float(str(parent["achievement_decimal"])),
        "parent_regret_above_archive_best": float(
            str(parent["regret_above_archive_best_decimal"])
        ),
        "objective": str(instruction["objective"]),
        "tradeoffs_can_be_frontier_improving": bool(
            instruction["tradeoffs_can_be_frontier_improving"]
        ),
        "simultaneous_improvement_on_every_axis_required": bool(
            instruction["simultaneous_improvement_on_every_axis_required"]
        ),
        "evaluator_outcomes_remain_unknown": bool(
            instruction["evaluator_outcomes_remain_unknown"]
        ),
        "achievement_kind": str(scalar["kind"]),
        "generation": int(cutoff["generation"]),
        "future_outcomes_consulted": bool(
            cutoff["current_or_future_candidate_outcomes_consulted"]
        ),
        "workload_identifiers_consulted": bool(
            payload["workload_identifiers_consulted"]
        ),
        "model_or_provider_fields_consulted": bool(
            payload["model_or_provider_fields_consulted"]
        ),
    }


def _augmented_chebyshev(point: tuple[float, ...], weights: tuple[float, ...]) -> float:
    active = tuple(
        (value, weight)
        for value, weight in zip(point, weights, strict=True)
        if weight > 0
    )
    if not active:
        raise ValueError("target achievement requires one active axis")
    maximum = max(weight * value for value, weight in active)
    weighted_mean = sum(weight * value for value, weight in active) / sum(
        weight for _, weight in active
    )
    return maximum + 0.05 * weighted_mean


def _prompt_evidence_and_frontier_context(
    machine: dict[str, Any],
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Recover empirical cards, affine context, and a coordinated lane target."""

    raw_cards = machine.get("cards", [])
    if type(raw_cards) is not list:
        raise TypeError("selector prompt cards must be a list")
    card_targets: dict[str, tuple[str, ...]] = {}
    for card in raw_cards:
        if type(card) is not dict or type(card.get("card_key")) is not str:
            raise ValueError("selector prompt card is malformed")
        card_key = str(card["card_key"])
        if card_key in card_targets:
            raise ValueError("selector prompt contains a duplicate card key")
        evidence = card.get("finite_action_evidence", [])
        if type(evidence) is not list:
            raise TypeError("finite-action card evidence must be a list")
        targets = tuple(
            sorted(
                {
                    str(value["option_id"])
                    for value in evidence
                    if type(value) is dict and type(value.get("option_id")) is str
                }
            )
        )
        card_targets[card_key] = targets

    raw_context = machine.get("context", {})
    if type(raw_context) is not dict:
        raise TypeError("selector prompt context must be an object")
    frontier_target = _frontier_target_projection(
        raw_context.get("campaign_frontier_target")
    )
    raw_frontier = raw_context.get("campaign_archive_context")
    if raw_frontier is None:
        return card_targets, None, frontier_target
    if type(raw_frontier) is not dict:
        raise TypeError("campaign archive context must be an object")
    payload = raw_frontier.get("payload")
    projector = raw_frontier.get("projector")
    if type(payload) is not dict or type(projector) is not dict:
        raise ValueError("campaign archive context lacks payload or projector")
    optimization = payload.get("optimization_frame")
    archive = payload.get("archive")
    parent = payload.get("parent")
    if (
        type(optimization) is not dict
        or type(archive) is not dict
        or type(parent) is not dict
    ):
        raise ValueError("campaign archive context lacks generic geometry")
    dimension = int(optimization["dimension"])
    axes = optimization.get("axes")
    points = archive.get("normalized_points_decimal")
    if (
        dimension not in (2, 3)
        or type(axes) is not list
        or len(axes) != dimension
        or type(points) is not list
        or int(archive["point_count"]) != len(points)
        or any(type(point) is not list or len(point) != dimension for point in points)
    ):
        raise ValueError("campaign archive context has inconsistent dimensions")
    projection_sha256 = raw_frontier.get("projection_sha256")
    definition_sha256 = projector.get("definition_sha256")
    if type(projection_sha256) is not str or type(definition_sha256) is not str:
        raise ValueError("campaign archive context lacks authenticated identities")
    return (
        card_targets,
        {
            "projection_sha256": projection_sha256,
            "projector_id": str(projector["projector_id"]),
            "projector_version": int(projector["projector_version"]),
            "projector_definition_sha256": definition_sha256,
            "dimension": dimension,
            "archive_point_count": len(points),
            "parent_dominated_by_archive": bool(
                parent["dominated_by_an_archive_point"]
            ),
            "base_hypervolume_decimal": str(optimization["base_hypervolume_decimal"]),
            "reference_direction_count": len(optimization["reference_directions"]),
            "future_outcomes_consulted": bool(
                payload["epistemic_cutoff"][
                    "current_or_future_candidate_outcomes_consulted"
                ]
            ),
        },
        frontier_target,
    )


def _proposal_support_projection(
    *,
    machine: dict[str, Any],
    supplemental_payload: dict[str, Any] | None,
    option_ids: tuple[str, ...],
    evaluated_option_ids: tuple[str, ...],
    common_pool: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Authenticate and normalize proposal-only structural reservations.

    The prompt deliberately exposes only the membership constraint.  A
    provider-backed supplemental receipt retains the complete pre-dispatch
    structural decision.  Joining both views here lets analysis distinguish a
    proposal reservation from an *observed* downstream evaluator preference;
    it does not infer that the reservation policy forced an evaluator slot.
    """

    constraints = machine.get("proposal_constraints")
    if constraints is None and supplemental_payload is None:
        # Legacy provider-free finite contracts predate proposal constraints.
        return None
    if type(constraints) is not dict:
        raise TypeError("selector machine contract lacks proposal constraints")
    prompt = constraints.get("proposal_support")
    audit = (
        supplemental_payload.get("proposal_support")
        if type(supplemental_payload) is dict
        else None
    )
    if (prompt is None) != (audit is None) and supplemental_payload is not None:
        raise ValueError("proposal-support prompt and supplemental audit disagree")
    if prompt is None:
        return None
    if type(prompt) is not dict:
        raise TypeError("proposal-support prompt projection must be an object")

    raw_required = prompt.get("required_option_ids")
    if type(raw_required) is not list or any(
        type(value) is not str or not value for value in raw_required
    ):
        raise ValueError("proposal-support required option IDs are malformed")
    required = tuple(str(value) for value in raw_required)
    if len(required) == 0 or len(set(required)) != len(required):
        raise ValueError("proposal-support required option IDs must be unique")
    missing_required = tuple(
        option_id for option_id in required if option_id not in option_ids
    )
    if missing_required:
        reconciliation = (
            supplemental_payload.get("semantic_reconciliation")
            if type(supplemental_payload) is dict
            else None
        )
        deferred = (
            reconciliation.get("deferred_proposal_support_option_ids")
            if type(reconciliation) is dict
            else None
        )
        if type(deferred) is not list or not set(missing_required).issubset(
            {str(value) for value in deferred}
        ):
            raise ValueError(
                "reconciled proposal omitted support without authenticated deferral"
            )
    if (
        prompt.get("reservations_are_quality_rankings") is not False
        or prompt.get("reservations_force_evaluator_slots") is not False
        or prompt.get("model_may_rank_reserved_options_anywhere") is not True
    ):
        raise ValueError("proposal-support prompt semantics drifted")

    evaluated_rank_by_option: dict[str, int] = {}
    allocator_role_by_option: dict[str, str] = {}
    if common_pool is not None:
        evaluated = common_pool.get("evaluated_option_ids")
        roles = common_pool.get("allocator_roles")
        if not (
            type(evaluated) is list
            and type(roles) is list
            and len(evaluated) == len(roles)
        ):
            raise ValueError("proposal-support join lacks a resolved allocation")
        for rank, (option_id, role) in enumerate(
            zip(evaluated, roles, strict=True), start=1
        ):
            evaluated_rank_by_option[str(option_id)] = rank
            allocator_role_by_option[str(option_id)] = str(role)

    reservations: list[dict[str, Any]] = []
    prompt_matches_audit: bool | None = None
    decision_sha256 = prompt.get("decision_sha256")
    membership_constraint_effective: bool | None = None
    policy: dict[str, Any] | None = None
    if audit is not None:
        if type(audit) is not dict:
            raise TypeError("proposal-support supplemental audit must be an object")
        raw_audit_required = audit.get("required_option_ids")
        raw_reservations = audit.get("reservations")
        raw_candidates = audit.get("candidates")
        if (
            type(raw_audit_required) is not list
            or type(raw_reservations) is not list
            or type(raw_candidates) is not list
        ):
            raise ValueError("proposal-support supplemental audit is malformed")
        audit_required = tuple(str(value) for value in raw_audit_required)
        if set(audit_required) != set(required):
            raise ValueError("proposal-support prompt membership differs from audit")
        if audit.get("decision_sha256") != decision_sha256:
            raise ValueError("proposal-support decision identity differs from prompt")
        if (
            audit.get("evaluator_slot_authority") is not False
            or audit.get("objective_or_outcome_values_consulted") is not False
            or audit.get("workload_or_model_fields_consulted") is not False
        ):
            raise ValueError("proposal-support epistemic contract drifted")
        reservation_ids: list[str] = []
        for value in raw_reservations:
            candidate = value.get("candidate") if type(value) is dict else None
            if (
                type(value) is not dict
                or type(candidate) is not dict
                or type(value.get("role")) is not str
                or type(candidate.get("option_id")) is not str
            ):
                raise ValueError("proposal-support reservation is malformed")
            option_id = str(candidate["option_id"])
            reservation_ids.append(option_id)
            reservations.append(
                {
                    "role": str(value["role"]),
                    "option_id": option_id,
                    "original_model_rank": (
                        None
                        if option_id not in option_ids
                        else option_ids.index(option_id) + 1
                    ),
                    "evaluated": option_id in evaluated_option_ids,
                    "allocation_rank": evaluated_rank_by_option.get(option_id),
                    "allocator_role": allocator_role_by_option.get(option_id),
                    "archive_novelty_score": _hex_or_number(
                        candidate["archive_novelty_score_hex"]
                    ),
                    "structural_coverage_score": _hex_or_number(
                        candidate["structural_coverage_score_hex"]
                    ),
                }
            )
        if set(reservation_ids) != set(required):
            raise ValueError("proposal-support reservations differ from membership")
        raw_roles = prompt.get("reservation_roles")
        if type(raw_roles) is not list or tuple(raw_roles) != tuple(
            value["role"] for value in reservations
        ):
            raise ValueError("proposal-support roles differ from prompt")
        model_selection_size = audit.get("model_selection_size")
        if type(model_selection_size) is not int:
            raise ValueError("proposal-support audit lacks model selection size")
        membership_constraint_effective = len(raw_candidates) > model_selection_size
        raw_policy = audit.get("policy")
        if type(raw_policy) is not dict:
            raise ValueError("proposal-support audit lacks policy identity")
        policy = dict(raw_policy)
        prompt_matches_audit = True
    else:
        # Provider-free controls retain the prompt-visible membership contract
        # even when they do not carry a provider supplemental audit.
        reservations = [
            {
                "role": None,
                "option_id": option_id,
                "original_model_rank": (
                    None
                    if option_id not in option_ids
                    else option_ids.index(option_id) + 1
                ),
                "evaluated": option_id in evaluated_option_ids,
                "allocation_rank": evaluated_rank_by_option.get(option_id),
                "allocator_role": allocator_role_by_option.get(option_id),
                "archive_novelty_score": None,
                "structural_coverage_score": None,
            }
            for option_id in required
        ]

    return {
        "decision_sha256": decision_sha256,
        "policy": policy,
        "required_option_ids": list(required),
        "reservation_count": len(reservations),
        "selected_inclusion_count": len(required) - len(missing_required),
        "missing_required_option_ids": list(missing_required),
        "authenticated_deferral_count": len(missing_required),
        "reconciled_membership_exact": not missing_required,
        "evaluated_reservation_count": sum(
            bool(value["evaluated"]) for value in reservations
        ),
        "all_reservations_evaluated": all(
            bool(value["evaluated"]) for value in reservations
        ),
        "prompt_projection_matches_audit": prompt_matches_audit,
        "membership_constraint_effective": membership_constraint_effective,
        "reservations_force_evaluator_slots": False,
        "reservations_are_quality_rankings": False,
        "reservations": reservations,
    }


def _semantic_reconciliation_projection(
    payload: dict[str, Any],
    *,
    evaluated_option_ids: tuple[str, ...],
) -> dict[str, Any] | None:
    """Authenticate the V9 semantic-slate projection without workload knowledge.

    The semantic-slate family lets the model express local preferences while
    deterministic code owns all cross-member constraints.  The resulting K8
    therefore has two distinct provenances: retained model suggestions and
    engine insertions.  Conflating the reconciled rank with the original model
    rank would make a healthy repair look like model search quality, so recover
    both layers explicitly here.
    """

    raw_receipt = payload.get("semantic_reconciliation")
    if raw_receipt is None:
        return None
    if type(raw_receipt) is not dict:
        raise TypeError("semantic reconciliation receipt must be an object")
    original = payload.get("original_model_response")
    reconciled = payload.get("original_k8_response")
    if type(original) is not dict or type(reconciled) is not dict:
        raise ValueError("semantic reconciliation lacks original/reconciled slates")
    original_members = original.get("members")
    reconciled_members = reconciled.get("members")
    receipt_members = raw_receipt.get("members")
    if not all(
        type(value) is list
        for value in (original_members, reconciled_members, receipt_members)
    ):
        raise TypeError("semantic reconciliation member projections must be lists")

    original_ids = tuple(str(value["option_id"]) for value in original_members)
    reconciled_ids = tuple(str(value["option_id"]) for value in reconciled_members)
    receipt_ids = tuple(str(value["option_id"]) for value in receipt_members)
    if receipt_ids != reconciled_ids:
        raise ValueError("semantic reconciliation receipt differs from K8 ordering")
    if len(set(reconciled_ids)) != len(reconciled_ids):
        raise ValueError("semantic reconciliation retained duplicate K8 members")
    duplicate_count = len(original_ids) - len(set(original_ids))
    if raw_receipt.get("duplicate_model_member_count") != duplicate_count:
        raise ValueError("semantic reconciliation duplicate count is unauthenticated")
    if tuple(int(value["reconciled_rank"]) for value in receipt_members) != tuple(
        range(1, len(receipt_members) + 1)
    ):
        raise ValueError("semantic reconciliation ranks are not contiguous")
    if not set(evaluated_option_ids).issubset(reconciled_ids):
        raise ValueError("semantic reconciliation omits an evaluated option")

    original_first_rank: dict[str, int] = {}
    for ordinal, member in enumerate(original_members, start=1):
        model_rank = int(member.get("model_rank", ordinal))
        original_first_rank.setdefault(str(member["option_id"]), model_rank)
    allowed_origins = {
        "model",
        "engine_required_support",
        "engine_memory_dose",
        "engine_feasibility",
        "engine_refill",
        "engine_global_coverage",
        "engine_contextual_allocation",
    }
    member_rows: list[dict[str, Any]] = []
    for value in receipt_members:
        origin = str(value["origin"])
        if origin not in allowed_origins:
            raise ValueError("semantic reconciliation has an unknown origin")
        option_id = str(value["option_id"])
        original_rank = value.get("original_model_rank")
        if origin == "model":
            if original_rank != original_first_rank.get(option_id):
                raise ValueError("retained model member lost its original rank")
        elif original_rank is not None:
            raise ValueError("engine-inserted member claims an original model rank")
        original_cards = tuple(
            str(card) for card in value.get("original_supporting_card_keys", [])
        )
        reconciled_cards = tuple(
            str(card) for card in value.get("reconciled_supporting_card_keys", [])
        )
        reasons = tuple(str(reason) for reason in value.get("reasons", []))
        member_rows.append(
            {
                "option_id": option_id,
                "origin": origin,
                "original_model_rank": original_rank,
                "reconciled_rank": int(value["reconciled_rank"]),
                "evaluated": option_id in evaluated_option_ids,
                "card_attribution_rewritten": original_cards != reconciled_cards,
                "original_supporting_card_keys": list(original_cards),
                "reconciled_supporting_card_keys": list(reconciled_cards),
                "reasons": list(reasons),
            }
        )

    model_members = [value for value in member_rows if value["origin"] == "model"]
    engine_members = [value for value in member_rows if value["origin"] != "model"]
    evaluated_members = [value for value in member_rows if value["evaluated"]]
    evaluated_engine = [
        value for value in evaluated_members if value["origin"] != "model"
    ]
    original_unique_ids = tuple(dict.fromkeys(original_ids))
    retained_model_ids = {value["option_id"] for value in model_members}

    raw_contextual_projection = raw_receipt.get("contextual_allocation_projection")
    contextual_projection: dict[str, Any] | None = None
    if raw_contextual_projection is not None:
        if type(raw_contextual_projection) is not dict:
            raise TypeError("contextual allocation projection must be an object")

        def exact_counts(
            field: str,
            expected_arms: tuple[str, ...] | None,
        ) -> tuple[tuple[str, int], ...]:
            raw_counts = raw_contextual_projection.get(field)
            if type(raw_counts) is not list:
                raise TypeError(f"{field} must be a list")
            counts: list[tuple[str, int]] = []
            for raw_count in raw_counts:
                if (
                    type(raw_count) is not list
                    or len(raw_count) != 2
                    or type(raw_count[0]) is not str
                    or type(raw_count[1]) is not int
                    or raw_count[1] < 0
                ):
                    raise TypeError(f"{field} has a malformed count")
                counts.append((raw_count[0], raw_count[1]))
            result = tuple(counts)
            observed_arms = tuple(value[0] for value in result)
            if observed_arms != tuple(sorted(set(observed_arms))):
                raise ValueError(f"{field} uses noncanonical or duplicate arms")
            if expected_arms is not None and observed_arms != expected_arms:
                raise ValueError(f"{field} uses unknown or noncanonical arms")
            if sum(value[1] for value in result) != len(evaluated_option_ids):
                raise ValueError(f"{field} does not cover the evaluated slate")
            return result

        requested_source = exact_counts("requested_source_target_counts", None)
        realized_source = exact_counts(
            "realized_source_target_counts",
            tuple(value[0] for value in requested_source),
        )
        requested_operator = exact_counts(
            "requested_operator_target_counts", ("atomic", "composite")
        )
        realized_operator = exact_counts(
            "realized_operator_target_counts", ("atomic", "composite")
        )

        def l1(
            requested: tuple[tuple[str, int], ...],
            realized: tuple[tuple[str, int], ...],
        ) -> int:
            return sum(
                abs(left[1] - right[1])
                for left, right in zip(requested, realized, strict=True)
            )

        source_l1 = l1(requested_source, realized_source)
        operator_l1 = l1(requested_operator, realized_operator)
        projected_ids = raw_contextual_projection.get("evaluation_option_ids")
        if (
            type(projected_ids) is not list
            or len(projected_ids) != len(evaluated_option_ids)
            or any(type(value) is not str for value in projected_ids)
            or len(set(projected_ids)) != len(projected_ids)
            or set(projected_ids) != set(evaluated_option_ids)
        ):
            raise ValueError(
                "contextual allocation projection differs from evaluated slate"
            )
        if (
            raw_contextual_projection.get("source_l1_deviation") != source_l1
            or raw_contextual_projection.get("operator_l1_deviation") != operator_l1
            or raw_contextual_projection.get("exact")
            is not (source_l1 == 0 and operator_l1 == 0)
        ):
            raise ValueError(
                "contextual allocation projection has inconsistent recourse"
            )
        if (
            raw_contextual_projection.get("objective_values_consulted") is not False
            or raw_contextual_projection.get("workload_identifiers_consulted")
            is not False
        ):
            raise ValueError(
                "contextual allocation projection is not workload/outcome blind"
            )
        if raw_contextual_projection.get("contract_sha256") != (
            raw_receipt.get("contextual_allocation_contract_sha256")
        ):
            raise ValueError(
                "contextual allocation projection contract identity differs"
            )
        contextual_projection = {
            "policy_id": str(raw_contextual_projection["policy_id"]),
            "policy_version": int(raw_contextual_projection["policy_version"]),
            "policy_definition_sha256": str(
                raw_contextual_projection["policy_definition_sha256"]
            ),
            "projection_sha256": str(raw_contextual_projection["projection_sha256"]),
            "contract_sha256": str(raw_contextual_projection["contract_sha256"]),
            "exact": source_l1 == 0 and operator_l1 == 0,
            "source_l1_deviation": source_l1,
            "operator_l1_deviation": operator_l1,
            "requested_source_target_counts": [
                list(value) for value in requested_source
            ],
            "realized_source_target_counts": [list(value) for value in realized_source],
            "requested_operator_target_counts": [
                list(value) for value in requested_operator
            ],
            "realized_operator_target_counts": [
                list(value) for value in realized_operator
            ],
            "evaluation_option_ids": sorted(projected_ids),
            "objective_values_consulted": False,
            "workload_identifiers_consulted": False,
        }
    raw_composition_projection = raw_receipt.get("composition_capacity_projection")
    composition_projection: dict[str, Any] | None = None
    if raw_composition_projection is not None:
        if type(raw_composition_projection) is not dict:
            raise TypeError("composition capacity projection must be an object")
        projection = ExactKCompositionCapacityProjection(
            proposal_size=int(raw_composition_projection["proposal_size"]),
            preferred_composite_count=int(
                raw_composition_projection["preferred_composite_count"]
            ),
            mandatory_atomic_count=int(
                raw_composition_projection["mandatory_atomic_count"]
            ),
            mandatory_composite_count=int(
                raw_composition_projection["mandatory_composite_count"]
            ),
            selectable_atomic_count=int(
                raw_composition_projection["selectable_atomic_count"]
            ),
            selectable_composite_count=int(
                raw_composition_projection["selectable_composite_count"]
            ),
            feasible_minimum_composite_count=int(
                raw_composition_projection["feasible_minimum_composite_count"]
            ),
            feasible_maximum_composite_count=int(
                raw_composition_projection["feasible_maximum_composite_count"]
            ),
            effective_composite_count=int(
                raw_composition_projection["effective_composite_count"]
            ),
        )
        expected_projection = projection.to_record()
        if raw_composition_projection != expected_projection:
            raise ValueError("composition capacity projection is unauthenticated")
        composition_projection = expected_projection
    return {
        "policy_id": str(raw_receipt["policy_id"]),
        "policy_version": int(raw_receipt["policy_version"]),
        "receipt_sha256": str(raw_receipt["receipt_sha256"]),
        "objective_values_consulted": raw_receipt.get("objective_values_consulted"),
        "workload_identifiers_consulted": raw_receipt.get(
            "workload_identifiers_consulted"
        ),
        "original_member_count": len(original_ids),
        "original_unique_member_count": len(original_unique_ids),
        "duplicate_model_member_count": duplicate_count,
        "reconciled_member_count": len(reconciled_ids),
        "retained_model_member_count": len(model_members),
        "engine_inserted_member_count": len(engine_members),
        "retained_model_member_rate": (
            None
            if not original_unique_ids
            else len(model_members) / len(original_unique_ids)
        ),
        "engine_insertion_rate": (
            None if not reconciled_ids else len(engine_members) / len(reconciled_ids)
        ),
        "dropped_original_option_ids": [
            option_id
            for option_id in original_unique_ids
            if option_id not in retained_model_ids
        ],
        "engine_inserted_option_ids": [value["option_id"] for value in engine_members],
        "evaluated_member_count": len(evaluated_members),
        "evaluated_model_member_count": (
            len(evaluated_members) - len(evaluated_engine)
        ),
        "evaluated_engine_member_count": len(evaluated_engine),
        "evaluated_engine_member_rate": (
            None
            if not evaluated_members
            else len(evaluated_engine) / len(evaluated_members)
        ),
        "model_card_attribution_rewrite_count": sum(
            bool(value["card_attribution_rewritten"]) for value in model_members
        ),
        "origin_counts": dict(
            sorted(Counter(value["origin"] for value in member_rows).items())
        ),
        "reason_counts": dict(
            sorted(
                Counter(
                    reason for value in member_rows for reason in value["reasons"]
                ).items()
            )
        ),
        "evaluation_feasibility_witness": [
            str(value)
            for value in raw_receipt.get("evaluation_feasibility_witness", [])
        ],
        "original_option_ids": list(original_ids),
        "reconciled_option_ids": list(reconciled_ids),
        "evaluated_option_ids": list(evaluated_option_ids),
        "contextual_allocation_projection": contextual_projection,
        "composition_capacity_projection": composition_projection,
        "members": member_rows,
    }


def _outcome_conditioned_selector_projection(
    *,
    response: dict[str, Any],
    selector_audit: dict[str, Any],
    generation: int,
) -> dict[str, Any]:
    """Normalize one trusted all-action K-set without inventing a model slate."""

    supplemental = response.get("supplemental_selector_audit")
    if (
        type(supplemental) is not dict
        or supplemental.get("audit_kind") != "outcome_conditioned_expert_portfolio"
    ):
        raise ValueError("response is not an outcome-conditioned selector audit")
    payload = supplemental.get("payload")
    ranked = response.get("ranked_decision")
    if type(payload) is not dict or type(ranked) is not dict:
        raise ValueError("outcome-conditioned selector lacks payload or decision")
    members = _selector_forecast_response_members(response)
    option_ids = tuple(str(value["option_id"]) for value in members)
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("outcome-conditioned decision repeats an option")
    allocation = payload.get("allocation")
    allocation_members = allocation.get("members") if type(allocation) is dict else None
    if type(allocation_members) is not list:
        raise ValueError("outcome-conditioned selector lacks trusted allocation")
    allocation_ids = tuple(str(value["option_id"]) for value in allocation_members)
    if allocation_ids != option_ids:
        raise ValueError("trusted allocation and ranked decision disagree")

    health = payload.get("forecast_health")
    assessments = health.get("metric_assessments") if type(health) is dict else None
    if type(assessments) is not list or not assessments:
        raise ValueError("outcome-conditioned selector lacks forecast health")
    row_counts = {
        int(value["row_count"])
        for value in assessments
        if type(value) is dict and type(value.get("row_count")) is int
    }
    if len(row_counts) != 1 or len(assessments) != len(
        [value for value in assessments if type(value) is dict]
    ):
        raise ValueError("forecast health has inconsistent universe sizes")
    forecast_universe_size = next(iter(row_counts))
    if forecast_universe_size < len(option_ids):
        raise ValueError("forecast universe is smaller than the evaluated slate")
    candidate_evaluations = allocation.get("candidate_evaluations")
    physical_call_count = payload.get("physical_call_count")
    if type(candidate_evaluations) is not int or candidate_evaluations <= 0:
        raise ValueError("trusted allocation lacks candidate-evaluation accounting")
    if type(physical_call_count) is not int or physical_call_count <= 0:
        raise ValueError("outcome-conditioned selector lacks physical call accounting")

    action_records = [
        {
            "model_rank": None,
            "allocation_rank": rank,
            "option_id": option_id,
            "action_kind": "sealed_action_without_hierarchy_projection",
            "component_option_ids": [],
            "evaluated": True,
        }
        for rank, option_id in enumerate(option_ids, start=1)
    ]
    phenotypes = tuple(str(value["child_configuration_sha256"]) for value in members)
    if len(set(phenotypes)) != len(phenotypes):
        collision_count = len(phenotypes) - len(set(phenotypes))
    else:
        collision_count = 0
    rationales = [str(value.get("design_rationale", "")) for value in members]
    rationale_jaccards = _pairwise_token_jaccards(rationales)
    member_card_attributions = []
    for member in members:
        cited = tuple(str(value) for value in member.get("supporting_card_keys", []))
        member_card_attributions.append(
            {
                "option_id": str(member["option_id"]),
                "evaluated": True,
                "cited_card_keys": list(cited),
                # The compact outcome-conditioned audit authenticates selected
                # card keys but deliberately does not reproduce the full prompt
                # card/action evidence.  Keep exact-target joins unidentified.
                "card_exact_finite_target": None,
                "card_citation_without_exact_finite_target": None,
                "cited_empirical_card_keys": [],
                "empirical_card_exact_target": None,
                "empirical_card_cross_target_generalization": None,
            }
        )
    cited_member_count = sum(
        bool(value["cited_card_keys"]) for value in member_card_attributions
    )
    empirical_card_attribution = {
        "available_card_count": None,
        "selected_card_citation_member_count": cited_member_count,
        "evaluated_card_citation_member_count": cited_member_count,
        "selected_card_citation_without_exact_finite_target_count": 0,
        "evaluated_card_citation_without_exact_finite_target_count": 0,
        "selected_citation_member_count": 0,
        "evaluated_citation_member_count": 0,
        "selected_exact_target_member_count": 0,
        "evaluated_exact_target_member_count": 0,
        "selected_cross_target_generalization_member_count": 0,
        "evaluated_cross_target_generalization_member_count": 0,
        "exact_card_target_projection_identified": False,
        "members": member_card_attributions,
    }
    role = f"trusted_all_action:{payload.get('target_kind', 'unknown')}"
    return {
        "generation": generation,
        "parent_slot": int(selector_audit["parent_slot"]),
        "request_sha256": str(selector_audit["request_sha256"]),
        "prompt_definition_sha256": str(payload["policy_definition_sha256"]),
        "selection_mode": "outcome_conditioned_trusted_all_action",
        "forecast_universe_size": forecast_universe_size,
        "allocator_candidate_evaluations": candidate_evaluations,
        "physical_forecast_call_count": physical_call_count,
        "role_label": role,
        "option_ids": list(option_ids),
        "variation_actions": action_records,
        "hierarchical_composition": None,
        "witness_mode": "trusted_outcome_conditioned_no_model_slate",
        "witness_option_ids": [],
        "hidden_feasibility_certificate": None,
        "witness_overlap_count": 0,
        "witness_overlap_fraction": None,
        "exact_ordered_witness_copy": False,
        "exact_set_witness_copy": False,
        "unique_option_count": len(option_ids),
        "unique_phenotype_count": len(set(phenotypes)),
        "within_call_phenotype_collision_count": collision_count,
        "unique_design_rationale_count": len(set(rationales)),
        "exact_duplicate_design_rationale_occurrence_count": (
            len(rationales) - len(set(rationales))
        ),
        "mean_pairwise_rationale_token_jaccard": (
            None
            if not rationale_jaccards
            else sum(rationale_jaccards) / len(rationale_jaccards)
        ),
        "card_attributed_member_count": cited_member_count,
        "empirical_card_attribution": empirical_card_attribution,
        "frontier_context": None,
        "frontier_target": None,
        "common_candidate_pool": None,
        "proposal_support": None,
        "semantic_reconciliation": None,
        "members": members,
        "phenotypes": list(phenotypes),
    }


def _selector_behavior(
    stages: list[dict[str, Any]],
    *,
    provider_backed: bool,
) -> dict[str, Any]:
    """Recover workload-neutral proposal diversity and witness-copy endpoints."""

    calls: list[dict[str, Any]] = []
    option_counts: Counter[str] = Counter()
    phenotype_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    prompt_definition_counts: Counter[str] = Counter()
    witness_mode_counts: Counter[str] = Counter()
    action_kind_counts: Counter[str] = Counter()
    evaluated_action_kind_counts: Counter[str] = Counter()
    composite_model_rank_counts: Counter[str] = Counter()
    catalogue_ids: set[str] = set()
    common_universe_ids: set[str] = set()
    by_generation: dict[int, list[dict[str, Any]]] = {}

    for event in stages:
        receipt = event["payload"]["stage_receipt"]
        generation = int(receipt["generation"])
        for selector_audit in receipt.get("selector_audits") or []:
            plaintext = selector_audit.get("plaintext_audit")
            if type(plaintext) is not dict:
                raise ValueError("selector audit lacks its plaintext projection")
            response = json.loads(plaintext["response_text"])
            if type(response) is not dict:
                raise TypeError("selector response projection must be an object")
            supplemental_projection = response.get("supplemental_selector_audit")
            if (
                provider_backed
                and type(supplemental_projection) is dict
                and supplemental_projection.get("audit_kind")
                == "outcome_conditioned_expert_portfolio"
            ):
                call = _outcome_conditioned_selector_projection(
                    response=response,
                    selector_audit=selector_audit,
                    generation=generation,
                )
                members = call["members"]
                option_ids = tuple(str(value) for value in call["option_ids"])
                option_counts.update(option_ids)
                prompt_definition_counts[call["prompt_definition_sha256"]] += 1
                witness_mode_counts[call["witness_mode"]] += 1
                role_counts[call["role_label"]] += len(members)
                for action in call["variation_actions"]:
                    action_kind_counts[str(action["action_kind"])] += 1
                    if action["evaluated"]:
                        evaluated_action_kind_counts[str(action["action_kind"])] += 1
                for member in members:
                    for prediction in member.get("effect_predictions", []):
                        confidence_counts[str(prediction["confidence"])] += 1
                        direction_counts[str(prediction["direction"])] += 1
                phenotype_counts.update(str(value) for value in call["phenotypes"])
                calls.append(call)
                by_generation.setdefault(generation, []).append(call)
                continue
            evaluated_option_ids: tuple[str, ...] = ()
            hidden_feasibility_certificate: dict[str, Any] | None = None
            payload: dict[str, Any] | None = None
            if provider_backed:
                machine = _prompt_machine_contract(plaintext["request_text"])
                supplemental = response.get("supplemental_selector_audit")
                if type(supplemental) is not dict:
                    raise ValueError("selector response lacks its supplemental audit")
                payload = supplemental.get("payload")
                if type(payload) is not dict:
                    raise ValueError("selector supplemental audit lacks its payload")
                original = payload.get("original_k8_response")
                if (
                    type(original) is not dict
                    or type(original.get("members")) is not list
                ):
                    raise ValueError(
                        "selector supplemental audit lacks original members"
                    )
                members = original["members"]
                proposal_constraints = machine["proposal_constraints"]
                prompt_common_pool = proposal_constraints.get(
                    "task_keyed_common_candidate_pool"
                )
                audit_common_pool = payload.get("common_candidate_pool")
                if (prompt_common_pool is None) != (audit_common_pool is None):
                    raise ValueError(
                        "common-pool prompt and supplemental audit disagree"
                    )
                common_pool: dict[str, Any] | None = None
                evaluated_model_ranks: tuple[int, ...] = ()
                allocator_roles: tuple[str, ...] = ()
                allocator_replacement_count: int | None = None
                if prompt_common_pool is not None:
                    if (
                        type(prompt_common_pool) is not dict
                        or type(audit_common_pool) is not dict
                    ):
                        raise TypeError("common-pool evidence must contain objects")
                    pool_ids = tuple(
                        str(value) for value in audit_common_pool.get("option_ids", [])
                    )
                    prompt_pool_ids = tuple(
                        str(value) for value in prompt_common_pool.get("option_ids", [])
                    )
                    task = audit_common_pool.get("task")
                    if type(task) is not dict:
                        raise ValueError("common-pool audit lacks its task record")
                    candidate_universe_size = int(task["candidate_pool_size"])
                    model_selection_size = int(task["model_selection_size"])
                    evaluation_size = int(task["evaluation_size"])
                    if (
                        len(pool_ids) != candidate_universe_size
                        or len(set(pool_ids)) != candidate_universe_size
                        or prompt_pool_ids != pool_ids
                        or prompt_common_pool.get("candidate_pool_size")
                        != candidate_universe_size
                        or prompt_common_pool.get("model_selection_size")
                        != model_selection_size
                        or prompt_common_pool.get("decision_sha256")
                        != audit_common_pool.get("decision_sha256")
                        or prompt_common_pool.get("task_identity_sha256")
                        != audit_common_pool.get("task_identity_sha256")
                    ):
                        raise ValueError(
                            "common-pool prompt is not an exact projection of its audit"
                        )
                    option_ids_for_validation = tuple(
                        str(value["option_id"]) for value in members
                    )
                    if (
                        len(option_ids_for_validation) != model_selection_size
                        or len(set(option_ids_for_validation)) != model_selection_size
                        or not set(option_ids_for_validation).issubset(pool_ids)
                    ):
                        raise ValueError(
                            "model selection is not an exact common-universe subset"
                        )
                    selected_role_join = payload.get("selected_role_join")
                    if type(selected_role_join) is not list:
                        raise ValueError(
                            "common-pool audit lacks the evaluated-role join"
                        )
                    evaluated_option_ids = tuple(
                        str(value["option_id"]) for value in selected_role_join
                    )
                    evaluated_model_ranks = tuple(
                        int(value["model_rank"]) for value in selected_role_join
                    )
                    allocator_roles = tuple(
                        str(value["role"]) for value in selected_role_join
                    )
                    if (
                        len(evaluated_option_ids) != evaluation_size
                        or len(set(evaluated_option_ids)) != evaluation_size
                        or not set(evaluated_option_ids).issubset(
                            option_ids_for_validation
                        )
                    ):
                        raise ValueError(
                            "evaluated allocation is not an exact selected subset"
                        )
                    resolved = payload.get("resolved_k4_decision")
                    resolved_members = (
                        resolved.get("members") if type(resolved) is dict else None
                    )
                    if (
                        type(resolved_members) is not list
                        or tuple(str(value["option_id"]) for value in resolved_members)
                        != evaluated_option_ids
                    ):
                        raise ValueError(
                            "evaluated-role join differs from the resolved decision"
                        )
                    witness_ids = tuple(
                        str(value)
                        for value in audit_common_pool.get(
                            "feasibility_witness_option_ids", []
                        )
                    )
                    if len(witness_ids) != evaluation_size or not set(
                        witness_ids
                    ).issubset(pool_ids):
                        raise ValueError(
                            "common-pool feasibility certificate is malformed"
                        )
                    allocator_replacement_count = sum(
                        role != "model_anchor" for role in allocator_roles
                    )
                    common_universe_ids.update(pool_ids)
                    common_pool = {
                        "candidate_universe_option_ids": list(pool_ids),
                        "candidate_universe_size": candidate_universe_size,
                        "model_selection_size": model_selection_size,
                        "evaluation_size": evaluation_size,
                        "evaluated_option_ids": list(evaluated_option_ids),
                        "evaluated_model_ranks": list(evaluated_model_ranks),
                        "allocator_roles": list(allocator_roles),
                        "allocator_replacement_count": (allocator_replacement_count),
                        "selection_fraction_of_universe": (
                            model_selection_size / candidate_universe_size
                        ),
                        "evaluation_fraction_of_universe": (
                            evaluation_size / candidate_universe_size
                        ),
                        "common_pool_decision_sha256": str(
                            audit_common_pool["decision_sha256"]
                        ),
                        "common_pool_task_identity_sha256": str(
                            audit_common_pool["task_identity_sha256"]
                        ),
                        "common_pool_policy_id": str(
                            audit_common_pool["policy"]["policy_id"]
                        ),
                        "common_pool_policy_version": int(
                            audit_common_pool["policy"]["policy_version"]
                        ),
                        "prompt_projection_matches_audit": True,
                        "model_or_provider_fields_consulted": (
                            audit_common_pool.get("model_or_provider_fields_consulted")
                        ),
                        "objective_or_outcome_values_consulted": (
                            audit_common_pool.get(
                                "objective_or_outcome_values_consulted"
                            )
                        ),
                        "hidden_feasibility_witness_absent_from_prompt": (
                            "feasibility_witness_option_ids" not in prompt_common_pool
                            and "engine_verified_feasible_option_id_witness"
                            not in proposal_constraints
                        ),
                        "model_selection_is_exact_common_pool_subset": True,
                        "evaluated_is_exact_model_selection_subset": True,
                        "literal_model_top_evaluation_size_preserved": (
                            allocator_replacement_count == 0
                            and evaluated_model_ranks
                            == tuple(range(1, evaluation_size + 1))
                        ),
                    }
                    witness_mode = "task_keyed_common_candidate_pool"
                else:
                    evaluated_option_ids = tuple(
                        str(value["option_id"]) for value in members
                    )
                    raw_hidden_certificate = proposal_constraints.get(
                        "engine_verified_feasibility_certificate"
                    )
                    if raw_hidden_certificate is not None:
                        if not (
                            type(raw_hidden_certificate) is dict
                            and raw_hidden_certificate.get("feasible_subset_exists")
                            is True
                            and raw_hidden_certificate.get("member_option_ids_rendered")
                            is False
                            and raw_hidden_certificate.get("objective_values_consulted")
                            is False
                            and type(raw_hidden_certificate.get("certificate_sha256"))
                            is str
                        ):
                            raise ValueError(
                                "hidden feasibility certificate is malformed"
                            )
                        hidden_feasibility_certificate = dict(raw_hidden_certificate)
                        witness_ids = ()
                        witness_mode = "hidden_certificate"
                    else:
                        witness_ids = tuple(
                            str(value)
                            for value in proposal_constraints.get(
                                "engine_verified_feasible_option_id_witness",
                                [],
                            )
                        )
                        witness_mode = (
                            "request_keyed"
                            if "witness_ordering_policy" in proposal_constraints
                            else "canonical"
                        )
                prompt_definition = str(payload["prompt_definition_sha256"])
                slate = payload.get("calibrated_slate", {})
                slate_members = slate.get("members", []) if type(slate) is dict else []
                phenotypes = tuple(
                    str(value["phenotype_identity_sha256"])
                    for value in slate_members
                    if type(value) is dict
                    and type(value.get("phenotype_identity_sha256")) is str
                )
            else:
                machine = _prompt_finite_contract(plaintext["request_text"])
                members = response.get("members")
                if type(members) is not list:
                    raise ValueError(
                        "provider-free selector decision lacks direct members"
                    )
                witness_ids = ()
                witness_mode = "provider_free_no_witness"
                prompt_definition = str(response["policy_definition_sha256"])
                phenotypes = tuple(
                    str(value["child_configuration_sha256"])
                    for value in members
                    if type(value) is dict
                    and type(value.get("child_configuration_sha256")) is str
                )
                common_pool = None
                evaluated_option_ids = tuple(
                    str(value["option_id"]) for value in members
                )

            semantic_reconciliation = (
                None
                if payload is None
                else _semantic_reconciliation_projection(
                    payload,
                    evaluated_option_ids=evaluated_option_ids,
                )
            )
            (
                card_targets,
                frontier_context,
                frontier_target,
            ) = _prompt_evidence_and_frontier_context(machine)
            option_ids = tuple(str(value["option_id"]) for value in members)
            action_records: list[dict[str, Any]] = []
            for model_rank, member in enumerate(members, start=1):
                option_id = str(member["option_id"])
                raw_action = member.get("hierarchical_action")
                if raw_action is None:
                    action_kind = (
                        "legacy_flat_composite"
                        if option_id.startswith("compose.")
                        else "atomic"
                    )
                    component_option_ids: tuple[str, ...] = ()
                else:
                    if type(raw_action) is not dict:
                        raise TypeError(
                            "hierarchical action projection must be an object"
                        )
                    action_kind = str(raw_action.get("action_kind"))
                    if action_kind not in {"atomic", "compose_r2"}:
                        raise ValueError(
                            "hierarchical action projection has an unknown kind"
                        )
                    raw_components = raw_action.get("component_option_ids", [])
                    if type(raw_components) is not list:
                        raise TypeError(
                            "hierarchical component projection must be a list"
                        )
                    component_option_ids = tuple(str(value) for value in raw_components)
                    if action_kind == "atomic" and component_option_ids:
                        raise ValueError("atomic hierarchy member declares components")
                    if action_kind == "compose_r2" and (
                        len(component_option_ids) != 2
                        or len(set(component_option_ids)) != 2
                    ):
                        raise ValueError(
                            "radius-two hierarchy member lacks two source atoms"
                        )
                evaluated = option_id in evaluated_option_ids
                action_kind_counts[action_kind] += 1
                if evaluated:
                    evaluated_action_kind_counts[action_kind] += 1
                if action_kind in {"compose_r2", "legacy_flat_composite"}:
                    composite_model_rank_counts[str(model_rank)] += 1
                action_records.append(
                    {
                        "model_rank": model_rank,
                        "option_id": option_id,
                        "action_kind": action_kind,
                        "component_option_ids": list(component_option_ids),
                        "evaluated": evaluated,
                    }
                )
            hierarchy_projection = None
            if provider_backed:
                raw_hierarchy = machine["proposal_constraints"].get(
                    "hierarchical_composition"
                )
                if raw_hierarchy is not None:
                    if type(raw_hierarchy) is not dict:
                        raise TypeError(
                            "hierarchical prompt projection must be an object"
                        )
                    required_composites = int(
                        raw_hierarchy["required_composite_proposals"]
                    )
                    observed_composites = sum(
                        value["action_kind"] == "compose_r2" for value in action_records
                    )
                    if observed_composites != required_composites:
                        raise ValueError(
                            "hierarchical proposal count differs from its prompt"
                        )
                    hierarchy_projection = {
                        "required_composite_proposals": required_composites,
                        "observed_composite_proposals": observed_composites,
                        "exact_required_count": True,
                    }
            proposal_support = _proposal_support_projection(
                machine=machine,
                supplemental_payload=payload,
                option_ids=option_ids,
                evaluated_option_ids=evaluated_option_ids,
                common_pool=common_pool,
            )
            rationales = [str(value.get("design_rationale", "")) for value in members]
            rationale_jaccards = _pairwise_token_jaccards(rationales)
            available_card_keys = set(card_targets)
            empirical_card_keys = {
                key for key, targets in card_targets.items() if targets
            }
            member_card_attributions: list[dict[str, Any]] = []
            for member in members:
                option_id = str(member["option_id"])
                cited_keys = tuple(
                    str(value) for value in member.get("supporting_card_keys", [])
                )
                if not set(cited_keys).issubset(available_card_keys):
                    raise ValueError("selector cited a card absent from its prompt")
                cited_empirical = tuple(
                    key for key in cited_keys if key in empirical_card_keys
                )
                exact_finite_target = any(
                    option_id in card_targets[key] for key in cited_keys
                )
                exact_target = any(
                    option_id in card_targets[key] for key in cited_empirical
                )
                member_card_attributions.append(
                    {
                        "option_id": option_id,
                        "evaluated": option_id in evaluated_option_ids,
                        "cited_card_keys": list(cited_keys),
                        "card_exact_finite_target": exact_finite_target,
                        "card_citation_without_exact_finite_target": bool(cited_keys)
                        and not exact_finite_target,
                        "cited_empirical_card_keys": list(cited_empirical),
                        "empirical_card_exact_target": exact_target,
                        "empirical_card_cross_target_generalization": bool(
                            cited_empirical
                        )
                        and not exact_target,
                    }
                )
            option_counts.update(option_ids)
            catalogue_ids.update(
                str(value["option_id"]) for value in machine["ordered_options"]
            )
            prompt_definition_counts[prompt_definition] += 1
            witness_mode_counts[witness_mode] += 1
            for member in members:
                role_counts[
                    str(member.get("role_proposal", "provider_free_control"))
                ] += 1
                for prediction in member.get("effect_predictions", []):
                    if "confidence" in prediction:
                        confidence_counts[str(prediction["confidence"])] += 1
                    direction_counts[str(prediction["direction"])] += 1
            phenotype_counts.update(phenotypes)
            option_set = set(option_ids)
            witness_set = set(witness_ids)
            overlap = len(option_set & witness_set)
            call = {
                "generation": generation,
                "parent_slot": int(selector_audit["parent_slot"]),
                "request_sha256": str(selector_audit["request_sha256"]),
                "prompt_definition_sha256": prompt_definition,
                "selection_mode": (
                    "legacy_provider_backed_free_form"
                    if provider_backed
                    else "provider_free_direct"
                ),
                "forecast_universe_size": None,
                "allocator_candidate_evaluations": None,
                "physical_forecast_call_count": None,
                "witness_mode": witness_mode,
                "option_ids": list(option_ids),
                "variation_actions": action_records,
                "hierarchical_composition": hierarchy_projection,
                "witness_option_ids": list(witness_ids),
                "hidden_feasibility_certificate": (hidden_feasibility_certificate),
                "witness_overlap_count": overlap,
                "witness_overlap_fraction": (
                    None if not witness_ids else overlap / len(witness_ids)
                ),
                "exact_ordered_witness_copy": bool(witness_ids)
                and option_ids == witness_ids,
                "exact_set_witness_copy": bool(witness_ids)
                and option_set == witness_set,
                "unique_option_count": len(option_set),
                "unique_phenotype_count": len(set(phenotypes)),
                "within_call_phenotype_collision_count": (
                    len(phenotypes) - len(set(phenotypes))
                ),
                "unique_design_rationale_count": len(set(rationales)),
                "exact_duplicate_design_rationale_occurrence_count": (
                    len(rationales) - len(set(rationales))
                ),
                "mean_pairwise_rationale_token_jaccard": (
                    None
                    if not rationale_jaccards
                    else sum(rationale_jaccards) / len(rationale_jaccards)
                ),
                "card_attributed_member_count": sum(
                    bool(value.get("supporting_card_keys")) for value in members
                ),
                "empirical_card_attribution": {
                    "available_card_count": len(empirical_card_keys),
                    "selected_card_citation_member_count": sum(
                        bool(value["cited_card_keys"])
                        for value in member_card_attributions
                    ),
                    "evaluated_card_citation_member_count": sum(
                        bool(value["cited_card_keys"]) and bool(value["evaluated"])
                        for value in member_card_attributions
                    ),
                    "selected_card_citation_without_exact_finite_target_count": sum(
                        bool(value["card_citation_without_exact_finite_target"])
                        for value in member_card_attributions
                    ),
                    "evaluated_card_citation_without_exact_finite_target_count": sum(
                        bool(value["card_citation_without_exact_finite_target"])
                        and bool(value["evaluated"])
                        for value in member_card_attributions
                    ),
                    "selected_citation_member_count": sum(
                        bool(value["cited_empirical_card_keys"])
                        for value in member_card_attributions
                    ),
                    "evaluated_citation_member_count": sum(
                        bool(value["cited_empirical_card_keys"])
                        and bool(value["evaluated"])
                        for value in member_card_attributions
                    ),
                    "selected_exact_target_member_count": sum(
                        bool(value["empirical_card_exact_target"])
                        for value in member_card_attributions
                    ),
                    "evaluated_exact_target_member_count": sum(
                        bool(value["empirical_card_exact_target"])
                        and bool(value["evaluated"])
                        for value in member_card_attributions
                    ),
                    "selected_cross_target_generalization_member_count": sum(
                        bool(value["empirical_card_cross_target_generalization"])
                        for value in member_card_attributions
                    ),
                    "evaluated_cross_target_generalization_member_count": sum(
                        bool(value["empirical_card_cross_target_generalization"])
                        and bool(value["evaluated"])
                        for value in member_card_attributions
                    ),
                    "members": member_card_attributions,
                },
                "frontier_context": frontier_context,
                "frontier_target": frontier_target,
                "common_candidate_pool": common_pool,
                "proposal_support": proposal_support,
                "semantic_reconciliation": semantic_reconciliation,
            }
            calls.append(call)
            by_generation.setdefault(generation, []).append(call)

    lane_rows: list[dict[str, Any]] = []
    all_lane_jaccards: list[float] = []
    for generation, generation_calls in sorted(by_generation.items()):
        pair_jaccards: list[float] = []
        for left_index, left in enumerate(generation_calls):
            for right in generation_calls[left_index + 1 :]:
                value = _jaccard(set(left["option_ids"]), set(right["option_ids"]))
                pair_jaccards.append(value)
                all_lane_jaccards.append(value)
        lane_rows.append(
            {
                "generation": generation,
                "lane_count": len(generation_calls),
                "lane_pair_count": len(pair_jaccards),
                "mean_lane_option_jaccard": (
                    None
                    if not pair_jaccards
                    else sum(pair_jaccards) / len(pair_jaccards)
                ),
                "minimum_lane_option_jaccard": (
                    None if not pair_jaccards else min(pair_jaccards)
                ),
                "maximum_lane_option_jaccard": (
                    None if not pair_jaccards else max(pair_jaccards)
                ),
                "lane_union_option_count": len(
                    set().union(
                        *(set(value["option_ids"]) for value in generation_calls)
                    )
                ),
            }
        )

    option_entropy = _entropy(option_counts)
    effective_option_count = math.exp(option_entropy) if option_counts else 0.0
    role_entropy = _entropy(role_counts)
    proposal_count = sum(option_counts.values())
    composite_proposal_count = sum(
        action_kind_counts.get(value, 0)
        for value in ("compose_r2", "legacy_flat_composite")
    )
    composite_evaluated_count = sum(
        evaluated_action_kind_counts.get(value, 0)
        for value in ("compose_r2", "legacy_flat_composite")
    )
    witness_calls = [value for value in calls if value["witness_option_ids"]]
    common_pool_calls = [
        value for value in calls if type(value["common_candidate_pool"]) is dict
    ]
    common_pool_records = [
        value["common_candidate_pool"] for value in common_pool_calls
    ]
    common_pool_evaluated_count = sum(
        int(value["evaluation_size"]) for value in common_pool_records
    )
    common_pool_allocator_replacements = sum(
        int(value["allocator_replacement_count"]) for value in common_pool_records
    )
    frontier_records = [
        value["frontier_context"]
        for value in calls
        if type(value["frontier_context"]) is dict
    ]
    frontier_target_records = [
        value["frontier_target"]
        for value in calls
        if type(value["frontier_target"]) is dict
    ]
    empirical_card_records = [value["empirical_card_attribution"] for value in calls]
    proposal_support_records = [
        value["proposal_support"]
        for value in calls
        if type(value["proposal_support"]) is dict
    ]
    proposal_support_reservations = [
        reservation
        for value in proposal_support_records
        for reservation in value["reservations"]
    ]
    semantic_reconciliation_records = [
        value["semantic_reconciliation"]
        for value in calls
        if type(value["semantic_reconciliation"]) is dict
    ]
    contextual_allocation_projections = [
        value["contextual_allocation_projection"]
        for value in semantic_reconciliation_records
        if type(value["contextual_allocation_projection"]) is dict
    ]
    composition_capacity_projections = [
        value["composition_capacity_projection"]
        for value in semantic_reconciliation_records
        if type(value["composition_capacity_projection"]) is dict
    ]

    def aggregate_contextual_counts(field: str) -> dict[str, int]:
        return dict(
            sorted(
                sum(
                    (
                        Counter(dict(value[field]))
                        for value in contextual_allocation_projections
                    ),
                    Counter(),
                ).items()
            )
        )

    semantic_reconciled_member_count = sum(
        int(value["reconciled_member_count"])
        for value in semantic_reconciliation_records
    )
    semantic_evaluated_member_count = sum(
        int(value["evaluated_member_count"])
        for value in semantic_reconciliation_records
    )
    semantic_engine_member_count = sum(
        int(value["engine_inserted_member_count"])
        for value in semantic_reconciliation_records
    )
    semantic_evaluated_engine_count = sum(
        int(value["evaluated_engine_member_count"])
        for value in semantic_reconciliation_records
    )
    selection_mode_counts = Counter(str(value["selection_mode"]) for value in calls)
    outcome_conditioned_calls = [
        value
        for value in calls
        if value["selection_mode"] == "outcome_conditioned_trusted_all_action"
    ]
    return {
        "selector_call_count": len(calls),
        "selection_mode_counts": dict(sorted(selection_mode_counts.items())),
        "outcome_conditioned_call_count": len(outcome_conditioned_calls),
        "outcome_conditioned_forecast_universe_row_count_total": sum(
            int(value["forecast_universe_size"]) for value in outcome_conditioned_calls
        ),
        "outcome_conditioned_forecast_universe_size_counts": dict(
            sorted(
                Counter(
                    str(value["forecast_universe_size"])
                    for value in outcome_conditioned_calls
                ).items()
            )
        ),
        "outcome_conditioned_allocator_candidate_evaluations_total": sum(
            int(value["allocator_candidate_evaluations"])
            for value in outcome_conditioned_calls
        ),
        "outcome_conditioned_physical_forecast_call_count": sum(
            int(value["physical_forecast_call_count"])
            for value in outcome_conditioned_calls
        ),
        "proposal_member_count_semantics": (
            "mixed_see_selection_mode_counts"
            if len(selection_mode_counts) > 1
            else (
                "trusted_evaluated_k_set"
                if outcome_conditioned_calls
                else "model_or_control_proposal_members"
            )
        ),
        "proposal_member_count": proposal_count,
        "variation_action_kind_counts": dict(sorted(action_kind_counts.items())),
        "evaluated_variation_action_kind_counts": dict(
            sorted(evaluated_action_kind_counts.items())
        ),
        "hierarchical_call_count": sum(
            value["hierarchical_composition"] is not None for value in calls
        ),
        "hierarchical_exact_required_count_rate": (
            None
            if not any(value["hierarchical_composition"] is not None for value in calls)
            else sum(
                bool(value["hierarchical_composition"]["exact_required_count"])
                for value in calls
                if value["hierarchical_composition"] is not None
            )
            / sum(value["hierarchical_composition"] is not None for value in calls)
        ),
        "composite_proposal_count": composite_proposal_count,
        "composite_proposal_share": (
            None if proposal_count == 0 else composite_proposal_count / proposal_count
        ),
        "composite_evaluated_count": composite_evaluated_count,
        "composite_proposal_evaluation_rate": (
            None
            if composite_proposal_count == 0
            else composite_evaluated_count / composite_proposal_count
        ),
        "composite_model_rank_counts": dict(
            sorted(composite_model_rank_counts.items(), key=lambda item: int(item[0]))
        ),
        "unique_option_count": len(option_counts),
        "catalogue_union_option_count": len(catalogue_ids),
        "catalogue_coverage_fraction": (
            None if not catalogue_ids else len(option_counts) / len(catalogue_ids)
        ),
        "option_entropy_nats": option_entropy,
        "effective_option_count": effective_option_count,
        "effective_option_fraction_of_slots": (
            None if proposal_count == 0 else effective_option_count / proposal_count
        ),
        "normalized_option_entropy_to_catalogue": (
            None
            if len(catalogue_ids) <= 1
            else option_entropy / math.log(len(catalogue_ids))
        ),
        "role_entropy_nats": role_entropy,
        "normalized_role_entropy_to_observed_roles": (
            None if len(role_counts) <= 1 else role_entropy / math.log(len(role_counts))
        ),
        "exact_duplicate_design_rationale_occurrence_count": sum(
            int(value["exact_duplicate_design_rationale_occurrence_count"])
            for value in calls
        ),
        "mean_within_call_rationale_token_jaccard": (
            None
            if not calls
            else sum(
                float(value["mean_pairwise_rationale_token_jaccard"])
                for value in calls
                if value["mean_pairwise_rationale_token_jaccard"] is not None
            )
            / sum(
                value["mean_pairwise_rationale_token_jaccard"] is not None
                for value in calls
            )
            if any(
                value["mean_pairwise_rationale_token_jaccard"] is not None
                for value in calls
            )
            else None
        ),
        "unique_phenotype_count": len(phenotype_counts),
        "phenotype_collision_occurrence_count": (
            sum(phenotype_counts.values()) - len(phenotype_counts)
        ),
        "exact_ordered_witness_copy_count": sum(
            value["exact_ordered_witness_copy"] for value in witness_calls
        ),
        "exact_ordered_witness_copy_rate": (
            None
            if not witness_calls
            else sum(value["exact_ordered_witness_copy"] for value in witness_calls)
            / len(witness_calls)
        ),
        "exact_set_witness_copy_count": sum(
            value["exact_set_witness_copy"] for value in witness_calls
        ),
        "exact_set_witness_copy_rate": (
            None
            if not witness_calls
            else sum(value["exact_set_witness_copy"] for value in witness_calls)
            / len(witness_calls)
        ),
        "mean_witness_overlap_fraction": (
            None
            if not witness_calls
            else sum(value["witness_overlap_fraction"] for value in witness_calls)
            / len(witness_calls)
        ),
        "mean_cross_lane_option_jaccard": (
            None
            if not all_lane_jaccards
            else sum(all_lane_jaccards) / len(all_lane_jaccards)
        ),
        "role_counts": dict(sorted(role_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "direction_counts": dict(sorted(direction_counts.items())),
        "prompt_definition_counts": dict(sorted(prompt_definition_counts.items())),
        "witness_mode_counts": dict(sorted(witness_mode_counts.items())),
        "hidden_feasibility_certificate_call_count": witness_mode_counts.get(
            "hidden_certificate", 0
        ),
        "common_pool_call_count": len(common_pool_calls),
        "common_universe_union_option_count": len(common_universe_ids),
        "common_pool_candidate_universe_size_counts": dict(
            sorted(
                Counter(
                    str(value["candidate_universe_size"])
                    for value in common_pool_records
                ).items()
            )
        ),
        "common_pool_model_selection_size_counts": dict(
            sorted(
                Counter(
                    str(value["model_selection_size"]) for value in common_pool_records
                ).items()
            )
        ),
        "common_pool_evaluation_size_counts": dict(
            sorted(
                Counter(
                    str(value["evaluation_size"]) for value in common_pool_records
                ).items()
            )
        ),
        "mean_common_universe_selection_fraction": (
            None
            if not common_pool_records
            else sum(
                float(value["selection_fraction_of_universe"])
                for value in common_pool_records
            )
            / len(common_pool_records)
        ),
        "mean_common_universe_evaluation_fraction": (
            None
            if not common_pool_records
            else sum(
                float(value["evaluation_fraction_of_universe"])
                for value in common_pool_records
            )
            / len(common_pool_records)
        ),
        "common_pool_allocator_replacement_count": (common_pool_allocator_replacements),
        "common_pool_allocator_replacement_rate": (
            None
            if common_pool_evaluated_count == 0
            else common_pool_allocator_replacements / common_pool_evaluated_count
        ),
        "common_pool_literal_model_top_evaluation_size_preserved_rate": (
            None
            if not common_pool_records
            else sum(
                bool(value["literal_model_top_evaluation_size_preserved"])
                for value in common_pool_records
            )
            / len(common_pool_records)
        ),
        "common_pool_prompt_projection_match_rate": (
            None
            if not common_pool_records
            else sum(
                bool(value["prompt_projection_matches_audit"])
                for value in common_pool_records
            )
            / len(common_pool_records)
        ),
        "common_pool_model_provider_blind_rate": (
            None
            if not common_pool_records
            else sum(
                value["model_or_provider_fields_consulted"] is False
                for value in common_pool_records
            )
            / len(common_pool_records)
        ),
        "common_pool_outcome_blind_rate": (
            None
            if not common_pool_records
            else sum(
                value["objective_or_outcome_values_consulted"] is False
                for value in common_pool_records
            )
            / len(common_pool_records)
        ),
        "common_pool_hidden_witness_rate": (
            None
            if not common_pool_records
            else sum(
                bool(value["hidden_feasibility_witness_absent_from_prompt"])
                for value in common_pool_records
            )
            / len(common_pool_records)
        ),
        "proposal_support_call_count": len(proposal_support_records),
        "proposal_support_reservation_count": len(proposal_support_reservations),
        "proposal_support_selected_inclusion_rate": (
            None
            if not proposal_support_records
            else sum(
                int(value["selected_inclusion_count"])
                for value in proposal_support_records
            )
            / sum(int(value["reservation_count"]) for value in proposal_support_records)
        ),
        "proposal_support_authenticated_deferral_count": sum(
            int(value["authenticated_deferral_count"])
            for value in proposal_support_records
        ),
        "proposal_support_reconciled_membership_exact_call_rate": (
            None
            if not proposal_support_records
            else sum(
                bool(value["reconciled_membership_exact"])
                for value in proposal_support_records
            )
            / len(proposal_support_records)
        ),
        "proposal_support_evaluated_reservation_count": sum(
            bool(value["evaluated"]) for value in proposal_support_reservations
        ),
        "proposal_support_reservation_evaluation_rate": (
            None
            if not proposal_support_reservations
            else sum(
                bool(value["evaluated"]) for value in proposal_support_reservations
            )
            / len(proposal_support_reservations)
        ),
        "proposal_support_evaluator_slot_share": (
            None
            if common_pool_evaluated_count == 0
            else sum(
                bool(value["evaluated"]) for value in proposal_support_reservations
            )
            / common_pool_evaluated_count
        ),
        "proposal_support_all_reservations_evaluated_call_rate": (
            None
            if not proposal_support_records
            else sum(
                bool(value["all_reservations_evaluated"])
                for value in proposal_support_records
            )
            / len(proposal_support_records)
        ),
        "proposal_support_original_model_rank_counts": dict(
            sorted(
                Counter(
                    str(value["original_model_rank"])
                    for value in proposal_support_reservations
                ).items()
            )
        ),
        "proposal_support_allocator_role_counts": dict(
            sorted(
                Counter(
                    str(value["allocator_role"])
                    for value in proposal_support_reservations
                    if value["allocator_role"] is not None
                ).items()
            )
        ),
        "proposal_support_prompt_projection_match_rate": (
            None
            if not [
                value
                for value in proposal_support_records
                if value["prompt_projection_matches_audit"] is not None
            ]
            else sum(
                value["prompt_projection_matches_audit"] is True
                for value in proposal_support_records
                if value["prompt_projection_matches_audit"] is not None
            )
            / sum(
                value["prompt_projection_matches_audit"] is not None
                for value in proposal_support_records
            )
        ),
        "semantic_reconciliation": {
            "call_count": len(semantic_reconciliation_records),
            "call_coverage_rate": (
                None if not calls else len(semantic_reconciliation_records) / len(calls)
            ),
            "original_member_count": sum(
                int(value["original_member_count"])
                for value in semantic_reconciliation_records
            ),
            "original_unique_member_count": sum(
                int(value["original_unique_member_count"])
                for value in semantic_reconciliation_records
            ),
            "duplicate_model_member_count": sum(
                int(value["duplicate_model_member_count"])
                for value in semantic_reconciliation_records
            ),
            "reconciled_member_count": semantic_reconciled_member_count,
            "retained_model_member_count": sum(
                int(value["retained_model_member_count"])
                for value in semantic_reconciliation_records
            ),
            "engine_inserted_member_count": semantic_engine_member_count,
            "engine_insertion_rate": (
                None
                if semantic_reconciled_member_count == 0
                else semantic_engine_member_count / semantic_reconciled_member_count
            ),
            "evaluated_member_count": semantic_evaluated_member_count,
            "evaluated_model_member_count": (
                semantic_evaluated_member_count - semantic_evaluated_engine_count
            ),
            "evaluated_engine_member_count": semantic_evaluated_engine_count,
            "evaluated_engine_member_rate": (
                None
                if semantic_evaluated_member_count == 0
                else semantic_evaluated_engine_count / semantic_evaluated_member_count
            ),
            "model_card_attribution_rewrite_count": sum(
                int(value["model_card_attribution_rewrite_count"])
                for value in semantic_reconciliation_records
            ),
            "origin_counts": dict(
                sorted(
                    sum(
                        (
                            Counter(value["origin_counts"])
                            for value in semantic_reconciliation_records
                        ),
                        Counter(),
                    ).items()
                )
            ),
            "reason_counts": dict(
                sorted(
                    sum(
                        (
                            Counter(value["reason_counts"])
                            for value in semantic_reconciliation_records
                        ),
                        Counter(),
                    ).items()
                )
            ),
            "objective_blind_call_rate": (
                None
                if not semantic_reconciliation_records
                else sum(
                    value["objective_values_consulted"] is False
                    for value in semantic_reconciliation_records
                )
                / len(semantic_reconciliation_records)
            ),
            "workload_identifier_blind_call_rate": (
                None
                if not semantic_reconciliation_records
                else sum(
                    value["workload_identifiers_consulted"] is False
                    for value in semantic_reconciliation_records
                )
                / len(semantic_reconciliation_records)
            ),
            "contextual_allocation_projection": {
                "call_count": len(contextual_allocation_projections),
                "call_coverage_rate": (
                    None
                    if not semantic_reconciliation_records
                    else len(contextual_allocation_projections)
                    / len(semantic_reconciliation_records)
                ),
                "exact_call_count": sum(
                    value["exact"] is True
                    for value in contextual_allocation_projections
                ),
                "exact_call_rate": (
                    None
                    if not contextual_allocation_projections
                    else sum(
                        value["exact"] is True
                        for value in contextual_allocation_projections
                    )
                    / len(contextual_allocation_projections)
                ),
                "source_l1_deviation": sum(
                    int(value["source_l1_deviation"])
                    for value in contextual_allocation_projections
                ),
                "operator_l1_deviation": sum(
                    int(value["operator_l1_deviation"])
                    for value in contextual_allocation_projections
                ),
                "requested_source_target_counts": aggregate_contextual_counts(
                    "requested_source_target_counts"
                ),
                "realized_source_target_counts": aggregate_contextual_counts(
                    "realized_source_target_counts"
                ),
                "requested_operator_target_counts": (
                    aggregate_contextual_counts("requested_operator_target_counts")
                ),
                "realized_operator_target_counts": (
                    aggregate_contextual_counts("realized_operator_target_counts")
                ),
                "objective_blind_call_rate": (
                    None
                    if not contextual_allocation_projections
                    else sum(
                        value["objective_values_consulted"] is False
                        for value in contextual_allocation_projections
                    )
                    / len(contextual_allocation_projections)
                ),
                "workload_identifier_blind_call_rate": (
                    None
                    if not contextual_allocation_projections
                    else sum(
                        value["workload_identifiers_consulted"] is False
                        for value in contextual_allocation_projections
                    )
                    / len(contextual_allocation_projections)
                ),
            },
            "composition_capacity_projection": {
                "call_count": len(composition_capacity_projections),
                "call_coverage_rate": (
                    None
                    if not semantic_reconciliation_records
                    else len(composition_capacity_projections)
                    / len(semantic_reconciliation_records)
                ),
                "capacity_projected_call_count": sum(
                    bool(value["capacity_projected"])
                    for value in composition_capacity_projections
                ),
                "capacity_projected_call_rate": (
                    None
                    if not composition_capacity_projections
                    else sum(
                        bool(value["capacity_projected"])
                        for value in composition_capacity_projections
                    )
                    / len(composition_capacity_projections)
                ),
                "preferred_composite_count_total": sum(
                    int(value["preferred_composite_count"])
                    for value in composition_capacity_projections
                ),
                "effective_composite_count_total": sum(
                    int(value["effective_composite_count"])
                    for value in composition_capacity_projections
                ),
                "absolute_projection_distance_total": sum(
                    abs(
                        int(value["preferred_composite_count"])
                        - int(value["effective_composite_count"])
                    )
                    for value in composition_capacity_projections
                ),
                "projections": composition_capacity_projections,
            },
            "calls": [
                {
                    "generation": value["generation"],
                    "parent_slot": value["parent_slot"],
                    **value["semantic_reconciliation"],
                }
                for value in calls
                if type(value["semantic_reconciliation"]) is dict
            ],
        },
        "card_attributed_member_count": sum(
            value["card_attributed_member_count"] for value in calls
        ),
        "evaluated_card_citation_member_count": sum(
            int(value["evaluated_card_citation_member_count"])
            for value in empirical_card_records
        ),
        "evaluated_card_citation_without_exact_finite_target_count": sum(
            int(value["evaluated_card_citation_without_exact_finite_target_count"])
            for value in empirical_card_records
        ),
        "empirical_card_available_call_count": sum(
            type(value["available_card_count"]) is int
            and int(value["available_card_count"]) > 0
            for value in empirical_card_records
        ),
        "empirical_card_selected_citation_member_count": sum(
            int(value["selected_citation_member_count"])
            for value in empirical_card_records
        ),
        "empirical_card_evaluated_citation_member_count": sum(
            int(value["evaluated_citation_member_count"])
            for value in empirical_card_records
        ),
        "empirical_card_selected_exact_target_member_count": sum(
            int(value["selected_exact_target_member_count"])
            for value in empirical_card_records
        ),
        "empirical_card_evaluated_exact_target_member_count": sum(
            int(value["evaluated_exact_target_member_count"])
            for value in empirical_card_records
        ),
        "empirical_card_selected_cross_target_generalization_member_count": sum(
            int(value["selected_cross_target_generalization_member_count"])
            for value in empirical_card_records
        ),
        "empirical_card_evaluated_cross_target_generalization_member_count": sum(
            int(value["evaluated_cross_target_generalization_member_count"])
            for value in empirical_card_records
        ),
        "frontier_context_call_count": len(frontier_records),
        "frontier_context_enabled_rate": (
            None if not calls else len(frontier_records) / len(calls)
        ),
        "frontier_context_distinct_projection_count": len(
            {value["projection_sha256"] for value in frontier_records}
        ),
        "frontier_context_dimension_counts": dict(
            sorted(
                Counter(str(value["dimension"]) for value in frontier_records).items()
            )
        ),
        "frontier_context_projector_counts": dict(
            sorted(
                Counter(
                    f"{value['projector_id']}:v{value['projector_version']}"
                    for value in frontier_records
                ).items()
            )
        ),
        "frontier_context_parent_dominated_count": sum(
            bool(value["parent_dominated_by_archive"]) for value in frontier_records
        ),
        "frontier_context_future_outcome_leak_count": sum(
            bool(value["future_outcomes_consulted"]) for value in frontier_records
        ),
        "frontier_target_call_count": len(frontier_target_records),
        "frontier_target_enabled_rate": (
            None if not calls else len(frontier_target_records) / len(calls)
        ),
        "frontier_target_distinct_target_count": len(
            {value["target_sha256"] for value in frontier_target_records}
        ),
        "frontier_target_allocator_counts": dict(
            sorted(
                Counter(
                    f"{value['allocator_id']}:v{value['allocator_version']}"
                    for value in frontier_target_records
                ).items()
            )
        ),
        "frontier_target_direction_counts": dict(
            sorted(
                Counter(
                    str(value["direction_id"]) for value in frontier_target_records
                ).items()
            )
        ),
        "frontier_target_opportunity_rank_counts": dict(
            sorted(
                Counter(
                    str(value["opportunity_rank"]) for value in frontier_target_records
                ).items()
            )
        ),
        "frontier_target_lane_counts": dict(
            sorted(
                Counter(
                    str(value["lane_id"]) for value in frontier_target_records
                ).items()
            )
        ),
        "frontier_target_weight_counts": dict(
            sorted(
                Counter(
                    json.dumps(value["normalized_weights"], separators=(",", ":"))
                    for value in frontier_target_records
                ).items()
            )
        ),
        "frontier_target_mean_opportunity_from_ideal": (
            None
            if not frontier_target_records
            else sum(
                float(value["opportunity_from_ideal"])
                for value in frontier_target_records
            )
            / len(frontier_target_records)
        ),
        "frontier_target_mean_parent_regret_above_archive_best": (
            None
            if not frontier_target_records
            else sum(
                float(value["parent_regret_above_archive_best"])
                for value in frontier_target_records
            )
            / len(frontier_target_records)
        ),
        "frontier_target_future_outcome_leak_count": sum(
            bool(value["future_outcomes_consulted"])
            for value in frontier_target_records
        ),
        "frontier_target_workload_identifier_consulted_count": sum(
            bool(value["workload_identifiers_consulted"])
            for value in frontier_target_records
        ),
        "frontier_target_model_or_provider_consulted_count": sum(
            bool(value["model_or_provider_fields_consulted"])
            for value in frontier_target_records
        ),
        "generation_lane_diversity": lane_rows,
        "calls": calls,
    }


def _original_model_rank_and_allocator_calibration(
    candidate_rows: list[dict[str, Any]],
    selector_behavior: dict[str, Any],
) -> dict[str, Any]:
    """Join evaluated candidates to pre-allocation K8 ranks and K4 roles.

    Candidate labels encode the position in the *resolved* K4 slate.  That
    position is not the original model rank when the workload-neutral
    structural allocator replaces or reorders the literal model top-K.  The
    authenticated ``selected_role_join`` recovered by ``_selector_behavior``
    is the authority for the original K8 rank and allocator role.  This join
    deliberately uses only generation, parent slot, and resolved-slate
    position, so it remains independent of workload option semantics.
    """

    allocation_by_slot: dict[tuple[int, int, int], dict[str, Any]] = {}
    support_by_option: dict[tuple[int, int, str], dict[str, Any]] = {}
    for call in selector_behavior["calls"]:
        common = call.get("common_candidate_pool")
        if type(common) is not dict:
            continue
        option_ids = common.get("evaluated_option_ids")
        model_ranks = common.get("evaluated_model_ranks")
        roles = common.get("allocator_roles")
        if not (
            type(option_ids) is list
            and type(model_ranks) is list
            and type(roles) is list
            and len(option_ids) == len(model_ranks) == len(roles)
        ):
            raise ValueError("common-pool evaluated-role join is malformed")
        generation = int(call["generation"])
        parent_slot = int(call["parent_slot"])
        proposal_support = call.get("proposal_support")
        if type(proposal_support) is dict:
            for reservation in proposal_support.get("reservations", []):
                if type(reservation) is not dict:
                    raise ValueError("proposal-support reservation join is malformed")
                support_key = (
                    generation,
                    parent_slot,
                    str(reservation["option_id"]),
                )
                if support_key in support_by_option:
                    raise ValueError("proposal-support reservation join is duplicated")
                support_by_option[support_key] = reservation
        for allocation_rank, (option_id, model_rank, role) in enumerate(
            zip(option_ids, model_ranks, roles, strict=True),
            start=1,
        ):
            key = (generation, parent_slot, allocation_rank)
            if key in allocation_by_slot:
                raise ValueError("common-pool evaluated-role join is duplicated")
            allocation_by_slot[key] = {
                "option_id": str(option_id),
                "original_model_rank": int(model_rank),
                "allocator_role": str(role),
            }

    joined_rows: list[dict[str, Any]] = []
    eligible_label_count = 0
    for candidate in candidate_rows:
        allocation_rank = candidate.get("allocation_rank")
        parent_slot = candidate.get("parent_slot")
        if allocation_rank is None or parent_slot is None:
            continue
        eligible_label_count += 1
        key = (
            int(candidate["generation"]),
            int(parent_slot),
            int(allocation_rank),
        )
        allocation = allocation_by_slot.get(key)
        if allocation is None:
            continue
        support = support_by_option.get(
            (
                int(candidate["generation"]),
                int(parent_slot),
                str(allocation["option_id"]),
            )
        )
        joined_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "generation": candidate["generation"],
                "parent_slot": parent_slot,
                "allocation_rank": allocation_rank,
                **allocation,
                "proposal_support_reserved": support is not None,
                "proposal_support_role": (
                    None if support is None else support.get("role")
                ),
                "scored": candidate["scored"],
                "typed_candidate_infeasible": candidate["typed_candidate_infeasible"],
                "normalized_point": candidate["normalized_point"],
                "individual_marginal_hypervolume": candidate[
                    "individual_marginal_hypervolume"
                ],
                "positive_individual_marginal": candidate[
                    "positive_individual_marginal"
                ],
                "admitted_to_stage_front": candidate["admitted_to_stage_front"],
                "admitted_to_final_front": candidate["admitted_to_final_front"],
            }
        )

    joined_by_option = {
        (
            int(value["generation"]),
            int(value["parent_slot"]),
            str(value["option_id"]),
        ): value
        for value in joined_rows
    }
    proposal_support_rows: list[dict[str, Any]] = []
    for (generation, parent_slot, option_id), reservation in sorted(
        support_by_option.items()
    ):
        outcome = joined_by_option.get((generation, parent_slot, option_id))
        proposal_support_rows.append(
            {
                "generation": generation,
                "parent_slot": parent_slot,
                **reservation,
                "scored": None if outcome is None else outcome["scored"],
                "typed_candidate_infeasible": (
                    None if outcome is None else outcome["typed_candidate_infeasible"]
                ),
                "individual_marginal_hypervolume": (
                    None
                    if outcome is None
                    else outcome["individual_marginal_hypervolume"]
                ),
                "positive_individual_marginal": (
                    None if outcome is None else outcome["positive_individual_marginal"]
                ),
                "admitted_to_stage_front": (
                    None if outcome is None else outcome["admitted_to_stage_front"]
                ),
                "admitted_to_final_front": (
                    None if outcome is None else outcome["admitted_to_final_front"]
                ),
            }
        )

    selected_proposal_count = sum(
        len(value["option_ids"])
        for value in selector_behavior["calls"]
        if type(value.get("proposal_support")) is dict
    )
    reserved_count = len(proposal_support_rows)
    evaluated_support = [value for value in proposal_support_rows if value["evaluated"]]
    evaluated_non_support = [
        value
        for value in joined_rows
        if not value["proposal_support_reserved"]
        and (value["generation"], value["parent_slot"])
        in {(row["generation"], row["parent_slot"]) for row in proposal_support_rows}
    ]
    nonreserved_selected_count = selected_proposal_count - reserved_count
    support_marginals = [
        float(value["individual_marginal_hypervolume"])
        for value in evaluated_support
        if value["individual_marginal_hypervolume"] is not None
    ]
    nonsupport_marginals = [
        float(value["individual_marginal_hypervolume"])
        for value in evaluated_non_support
        if value["individual_marginal_hypervolume"] is not None
    ]
    support_evaluation_rate = (
        None if reserved_count == 0 else len(evaluated_support) / reserved_count
    )
    nonsupport_evaluation_rate = (
        None
        if nonreserved_selected_count == 0
        else len(evaluated_non_support) / nonreserved_selected_count
    )

    def support_role_rows() -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for role in sorted({str(value["role"]) for value in proposal_support_rows}):
            members = [
                value for value in proposal_support_rows if str(value["role"]) == role
            ]
            evaluated_members = [value for value in members if value["evaluated"]]
            marginals = [
                float(value["individual_marginal_hypervolume"])
                for value in evaluated_members
                if value["individual_marginal_hypervolume"] is not None
            ]
            result.append(
                {
                    "role": role,
                    "reservation_count": len(members),
                    "evaluated_count": len(evaluated_members),
                    "positive_individual_marginal_count": sum(
                        value["positive_individual_marginal"] is True
                        for value in evaluated_members
                    ),
                    "stage_front_admission_count": sum(
                        value["admitted_to_stage_front"] is True
                        for value in evaluated_members
                    ),
                    "final_front_admission_count": sum(
                        value["admitted_to_final_front"] is True
                        for value in evaluated_members
                    ),
                    "median_individual_marginal_hypervolume": (
                        None if not marginals else median(marginals)
                    ),
                }
            )
        return result

    def aggregate(key: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for value in sorted({row[key] for row in joined_rows}, key=str):
            members = [row for row in joined_rows if row[key] == value]
            marginals = [
                row["individual_marginal_hypervolume"]
                for row in members
                if row["individual_marginal_hypervolume"] is not None
            ]
            result.append(
                {
                    key: value,
                    "candidate_count": len(members),
                    "scored_candidate_count": sum(row["scored"] for row in members),
                    "typed_candidate_infeasible_count": sum(
                        row["typed_candidate_infeasible"] for row in members
                    ),
                    "positive_individual_marginal_count": sum(
                        row["positive_individual_marginal"] for row in members
                    ),
                    "stage_front_admission_count": sum(
                        row["admitted_to_stage_front"] for row in members
                    ),
                    "final_front_admission_count": sum(
                        row["admitted_to_final_front"] for row in members
                    ),
                    "median_individual_marginal_hypervolume": (
                        None if not marginals else median(marginals)
                    ),
                }
            )
        return result

    return {
        "join_semantics": (
            "authenticated_generation_parent_resolved_slot_to_original_k8_rank"
        ),
        "allocated_slot_count": len(allocation_by_slot),
        "eligible_candidate_label_count": eligible_label_count,
        "joined_candidate_count": len(joined_rows),
        "unjoined_candidate_label_count": eligible_label_count - len(joined_rows),
        "original_model_rank_rows": aggregate("original_model_rank"),
        "allocator_role_rows": aggregate("allocator_role"),
        "proposal_support_calibration": {
            "semantics": (
                "authenticated_proposal_membership_joined_to_observed_"
                "downstream_allocation_and_candidate_outcome"
            ),
            "reservation_count": reserved_count,
            "evaluated_reservation_count": len(evaluated_support),
            "reservation_evaluation_rate": support_evaluation_rate,
            "selected_nonreservation_count": nonreserved_selected_count,
            "evaluated_nonreservation_count": len(evaluated_non_support),
            "nonreservation_evaluation_rate": nonsupport_evaluation_rate,
            "reservation_to_nonreservation_evaluation_rate_ratio": (
                None
                if support_evaluation_rate is None
                or nonsupport_evaluation_rate in (None, 0.0)
                else support_evaluation_rate / nonsupport_evaluation_rate
            ),
            "evaluator_slot_share": (
                None if not joined_rows else len(evaluated_support) / len(joined_rows)
            ),
            "positive_individual_marginal_count": sum(
                value["positive_individual_marginal"] is True
                for value in evaluated_support
            ),
            "positive_individual_marginal_rate": (
                None
                if not evaluated_support
                else sum(
                    value["positive_individual_marginal"] is True
                    for value in evaluated_support
                )
                / len(evaluated_support)
            ),
            "nonreservation_positive_individual_marginal_count": sum(
                value["positive_individual_marginal"] is True
                for value in evaluated_non_support
            ),
            "nonreservation_positive_individual_marginal_rate": (
                None
                if not evaluated_non_support
                else sum(
                    value["positive_individual_marginal"] is True
                    for value in evaluated_non_support
                )
                / len(evaluated_non_support)
            ),
            "median_individual_marginal_hypervolume": (
                None if not support_marginals else median(support_marginals)
            ),
            "nonreservation_median_individual_marginal_hypervolume": (
                None if not nonsupport_marginals else median(nonsupport_marginals)
            ),
            "stage_front_admission_count": sum(
                value["admitted_to_stage_front"] is True for value in evaluated_support
            ),
            "final_front_admission_count": sum(
                value["admitted_to_final_front"] is True for value in evaluated_support
            ),
            "role_rows": support_role_rows(),
            "rows": proposal_support_rows,
        },
        "joined_candidates": joined_rows,
    }


def _semantic_reconciliation_outcomes(
    rank_role_join: dict[str, Any],
    selector_behavior: dict[str, Any],
) -> dict[str, Any]:
    """Join V9 model/engine origins to real downstream candidate outcomes."""

    reconciliation = selector_behavior.get("semantic_reconciliation")
    calls = reconciliation.get("calls", []) if type(reconciliation) is dict else []
    member_by_option: dict[tuple[int, int, str], dict[str, Any]] = {}
    call_keys: set[tuple[int, int]] = set()
    for call in calls:
        generation = int(call["generation"])
        parent_slot = int(call["parent_slot"])
        call_key = (generation, parent_slot)
        if call_key in call_keys:
            raise ValueError("semantic reconciliation call is duplicated")
        call_keys.add(call_key)
        for member in call["members"]:
            key = (generation, parent_slot, str(member["option_id"]))
            if key in member_by_option:
                raise ValueError("semantic reconciliation member join is duplicated")
            member_by_option[key] = member

    rows: list[dict[str, Any]] = []
    eligible_count = 0
    for candidate in rank_role_join["joined_candidates"]:
        call_key = (int(candidate["generation"]), int(candidate["parent_slot"]))
        if call_key not in call_keys:
            continue
        eligible_count += 1
        member = member_by_option.get((*call_key, str(candidate["option_id"])))
        if member is None or member.get("evaluated") is not True:
            raise ValueError(
                "evaluated candidate lacks its semantic reconciliation origin"
            )
        origin = str(member["origin"])
        rows.append(
            {
                **candidate,
                # The pre-existing field comes from selected_role_join and is
                # the rank in the reconciled K8.  Preserve it under an honest
                # name and expose the pre-reconciliation model rank separately.
                "reconciled_k8_rank": candidate["original_model_rank"],
                "semantic_origin": origin,
                "semantic_origin_group": ("model" if origin == "model" else "engine"),
                "semantic_original_model_rank": member["original_model_rank"],
                "reconciliation_reasons": member["reasons"],
            }
        )
    if len(rows) != eligible_count:
        raise ValueError("semantic reconciliation outcome join is incomplete")

    def aggregate(key: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        values = sorted(
            {value[key] for value in rows if value[key] is not None},
            key=str,
        )
        for group in values:
            members = [value for value in rows if value[key] == group]
            marginals = [
                float(value["individual_marginal_hypervolume"])
                for value in members
                if value["individual_marginal_hypervolume"] is not None
            ]
            positive = sum(
                value["positive_individual_marginal"] is True for value in members
            )
            result.append(
                {
                    key: group,
                    "candidate_count": len(members),
                    "positive_individual_marginal_count": positive,
                    "positive_individual_marginal_rate": (
                        None if not members else positive / len(members)
                    ),
                    "stage_front_admission_count": sum(
                        value["admitted_to_stage_front"] is True for value in members
                    ),
                    "final_front_admission_count": sum(
                        value["admitted_to_final_front"] is True for value in members
                    ),
                    "median_individual_marginal_hypervolume": (
                        None if not marginals else median(marginals)
                    ),
                    # Individual marginals overlap and are not additive causal
                    # credit.  The explicit name prevents this diagnostic from
                    # being mistaken for a Shapley or ablation attribution.
                    "descriptive_sum_individual_marginal_hypervolume": sum(marginals),
                }
            )
        return result

    return {
        "semantics": (
            "authenticated_reconciled_member_origin_joined_to_downstream_"
            "candidate_outcome;individual_marginal_sums_are_descriptive_"
            "nonadditive_not_causal_credit"
        ),
        "reconciliation_call_count": len(calls),
        "eligible_candidate_count": eligible_count,
        "joined_candidate_count": len(rows),
        "origin_group_rows": aggregate("semantic_origin_group"),
        "origin_rows": aggregate("semantic_origin"),
        "semantic_original_model_rank_rows": aggregate("semantic_original_model_rank"),
        "rows": rows,
    }


def _frontier_target_outcomes(
    rank_role_join: dict[str, Any],
    selector_behavior: dict[str, Any],
) -> dict[str, Any]:
    """Join authenticated lane targets to realized generic frontier outcomes."""

    target_by_call: dict[tuple[int, int], dict[str, Any]] = {}
    for call in selector_behavior.get("calls", []):
        target = call.get("frontier_target")
        if type(target) is not dict:
            continue
        key = (int(call["generation"]), int(call["parent_slot"]))
        if key in target_by_call:
            raise ValueError("frontier target call is duplicated")
        target_by_call[key] = target

    rows: list[dict[str, Any]] = []
    for candidate in rank_role_join.get("joined_candidates", []):
        key = (int(candidate["generation"]), int(candidate["parent_slot"]))
        target = target_by_call.get(key)
        if target is None:
            continue
        raw_point = candidate.get("normalized_point")
        if raw_point is None:
            achievement = None
            target_improvement = None
            improves_parent = False
            beats_archive_best = False
        else:
            if type(raw_point) is not list:
                raise TypeError("joined candidate normalized point must be a list")
            point = tuple(float(value) for value in raw_point)
            weights = tuple(float(value) for value in target["normalized_weights"])
            if len(point) != len(weights):
                raise ValueError("frontier target and candidate dimensions differ")
            achievement = _augmented_chebyshev(point, weights)
            parent_achievement = float(target["parent_achievement"])
            archive_best = float(target["archive_best_achievement"])
            target_improvement = parent_achievement - achievement
            tolerance = 64 * math.ulp(
                max(1.0, abs(parent_achievement), abs(archive_best), abs(achievement))
            )
            improves_parent = target_improvement > tolerance
            beats_archive_best = archive_best - achievement > tolerance
        aspiration = target.get("normalized_aspiration_point")
        parent_point = target.get("normalized_parent_point")
        if raw_point is None or aspiration is None:
            aspiration_shortfall_l1 = None
            aspiration_linf_distance = None
            parent_aspiration_shortfall_l1 = None
            aspiration_shortfall_reduction = None
            attains_or_dominates_aspiration = False
        else:
            assert type(raw_point) is list
            if type(aspiration) is not list or type(parent_point) is not list:
                raise TypeError("residual target points must be lists")
            point = tuple(float(value) for value in raw_point)
            target_point = tuple(float(value) for value in aspiration)
            source_point = tuple(float(value) for value in parent_point)
            if not len(point) == len(target_point) == len(source_point):
                raise ValueError("residual target and candidate dimensions differ")
            aspiration_shortfall_l1 = sum(
                max(0.0, value - target_value)
                for value, target_value in zip(point, target_point, strict=True)
            )
            aspiration_linf_distance = max(
                abs(value - target_value)
                for value, target_value in zip(point, target_point, strict=True)
            )
            parent_aspiration_shortfall_l1 = sum(
                max(0.0, value - target_value)
                for value, target_value in zip(
                    source_point,
                    target_point,
                    strict=True,
                )
            )
            aspiration_shortfall_reduction = (
                parent_aspiration_shortfall_l1 - aspiration_shortfall_l1
            )
            aspiration_tolerance = 64 * math.ulp(
                max(
                    1.0,
                    *(abs(value) for value in point),
                    *(abs(value) for value in target_point),
                )
            )
            attains_or_dominates_aspiration = (
                aspiration_shortfall_l1 <= aspiration_tolerance
            )
        rows.append(
            {
                **candidate,
                "target_sha256": target["target_sha256"],
                "target_lane_id": target["lane_id"],
                "target_direction_id": target["direction_id"],
                "target_opportunity_rank": target["opportunity_rank"],
                "target_normalized_weights": target["normalized_weights"],
                "target_opportunity_from_ideal": target["opportunity_from_ideal"],
                "target_parent_achievement": target["parent_achievement"],
                "target_archive_best_achievement": target["archive_best_achievement"],
                "candidate_target_achievement": achievement,
                "target_achievement_improvement_over_parent": target_improvement,
                "improves_assigned_parent_target_achievement": improves_parent,
                "beats_prior_archive_best_target_achievement": beats_archive_best,
                "target_normalized_aspiration_point": aspiration,
                "target_residual_potential_hypervolume_gain": target.get(
                    "residual_potential_hypervolume_gain"
                ),
                "candidate_aspiration_shortfall_l1": aspiration_shortfall_l1,
                "candidate_aspiration_linf_distance": aspiration_linf_distance,
                "parent_aspiration_shortfall_l1": parent_aspiration_shortfall_l1,
                "aspiration_shortfall_reduction_over_parent": (
                    aspiration_shortfall_reduction
                ),
                "attains_or_dominates_residual_aspiration": (
                    attains_or_dominates_aspiration
                ),
            }
        )

    def aggregate(key: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for group in sorted({value[key] for value in rows}, key=str):
            members = [value for value in rows if value[key] == group]
            scored = [
                value
                for value in members
                if value["candidate_target_achievement"] is not None
            ]
            improvements = [
                float(value["target_achievement_improvement_over_parent"])
                for value in scored
            ]
            result.append(
                {
                    key: group,
                    "candidate_count": len(members),
                    "scored_candidate_count": len(scored),
                    "improves_assigned_parent_count": sum(
                        bool(value["improves_assigned_parent_target_achievement"])
                        for value in scored
                    ),
                    "improves_assigned_parent_rate": (
                        None
                        if not scored
                        else sum(
                            bool(value["improves_assigned_parent_target_achievement"])
                            for value in scored
                        )
                        / len(scored)
                    ),
                    "beats_prior_archive_best_count": sum(
                        bool(value["beats_prior_archive_best_target_achievement"])
                        for value in scored
                    ),
                    "positive_individual_marginal_count": sum(
                        value["positive_individual_marginal"] is True
                        for value in members
                    ),
                    "stage_front_admission_count": sum(
                        value["admitted_to_stage_front"] is True for value in members
                    ),
                    "final_front_admission_count": sum(
                        value["admitted_to_final_front"] is True for value in members
                    ),
                    "median_target_achievement_improvement_over_parent": (
                        None if not improvements else median(improvements)
                    ),
                }
            )
        return result

    return {
        "semantics": (
            "authenticated_prior_only_lane_target_joined_to_real_candidate;"
            "positive_target_achievement_improvement_reduces_assigned_"
            "augmented_chebyshev_scalar;descriptive_not_randomized_causal_credit"
        ),
        "target_call_count": len(target_by_call),
        "joined_candidate_count": len(rows),
        "scored_candidate_count": sum(
            value["candidate_target_achievement"] is not None for value in rows
        ),
        "improves_assigned_parent_count": sum(
            bool(value["improves_assigned_parent_target_achievement"]) for value in rows
        ),
        "beats_prior_archive_best_count": sum(
            bool(value["beats_prior_archive_best_target_achievement"]) for value in rows
        ),
        "residual_target_joined_candidate_count": sum(
            value["target_normalized_aspiration_point"] is not None for value in rows
        ),
        "attains_or_dominates_residual_aspiration_count": sum(
            bool(value["attains_or_dominates_residual_aspiration"]) for value in rows
        ),
        "reduces_residual_aspiration_shortfall_count": sum(
            value["aspiration_shortfall_reduction_over_parent"] is not None
            and float(value["aspiration_shortfall_reduction_over_parent"]) > 0.0
            for value in rows
        ),
        "median_residual_aspiration_shortfall_reduction_over_parent": (
            None
            if not (
                reductions := [
                    float(value["aspiration_shortfall_reduction_over_parent"])
                    for value in rows
                    if value["aspiration_shortfall_reduction_over_parent"] is not None
                ]
            )
            else median(reductions)
        ),
        "direction_rows": aggregate("target_direction_id"),
        "lane_rows": aggregate("target_lane_id"),
        "opportunity_rank_rows": aggregate("target_opportunity_rank"),
        "rows": rows,
    }


def _contextual_search_behavior(summary: dict[str, Any]) -> dict[str, Any]:
    """Recover prospective source/operator decisions and their realized rewards.

    Source IDs are deliberately opaque workload-owned tokens.  In older traces
    they describe semantic model/engine origin; current traces describe the
    sealed finite variation source (for example, primary or global restart).
    The analyzer never hard-codes either vocabulary.
    """

    def records(name: str) -> list[dict[str, Any]]:
        raw = summary.get(name, [])
        if raw is None:
            return []
        if type(raw) is not list or any(type(value) is not dict for value in raw):
            raise TypeError(f"{name} must contain JSON objects")
        return raw

    observations = records("contextual_search_observations")
    plans = records("contextual_search_plans")
    delayed_credits = records("contextual_search_delayed_credits")
    if not observations and not plans and not delayed_credits:
        return {
            "enabled": False,
            "observation_count": 0,
            "plan_count": 0,
            "delayed_credit_count": 0,
            "source_rows": [],
            "operator_rows": [],
            "wave_rows": [],
            "plan_rows": [],
            "source_probability_trajectories": {},
            "operator_probability_trajectories": {},
            "exact_source_realization_wave_rate": None,
            "exact_operator_realization_wave_rate": None,
            "delayed_credit": {
                "stage_front_persisted_count": 0,
                "useful_descendant_true_count": 0,
                "useful_descendant_false_count": 0,
                "useful_descendant_unresolved_count": 0,
                "final_front_persisted_count": 0,
            },
        }

    def aggregate_observations(field: str) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for observation in observations:
            arm = observation.get(field)
            if type(arm) is not str or not arm:
                raise TypeError(f"contextual observation omits {field}")
            grouped.setdefault(arm, []).append(observation)
        result = []
        for arm, rows in sorted(grouped.items()):
            utilities = [
                _hex_or_number(value["normalized_marginal_utility_hex"])
                for value in rows
            ]
            shares = [
                _hex_or_number(value["marginal_utility_share_hex"]) for value in rows
            ]
            result.append(
                {
                    "arm_id": arm,
                    "observation_count": len(rows),
                    "feasible_count": sum(
                        value.get("feasible") is True for value in rows
                    ),
                    "positive_marginal_utility_count": sum(
                        value.get("positive_marginal_utility") is True for value in rows
                    ),
                    "positive_marginal_utility_rate": sum(
                        value.get("positive_marginal_utility") is True for value in rows
                    )
                    / len(rows),
                    "mean_normalized_marginal_utility": sum(utilities) / len(rows),
                    "sum_normalized_marginal_utility": sum(utilities),
                    "mean_marginal_utility_share": sum(shares) / len(rows),
                    "wave_counts": dict(
                        sorted(
                            Counter(
                                str(int(value["wave_index"])) for value in rows
                            ).items()
                        )
                    ),
                }
            )
        return result

    plan_rows: list[dict[str, Any]] = []
    trajectories: dict[str, dict[str, list[dict[str, Any]]]] = {
        "source": {},
        "operator": {},
    }
    plan_by_wave: dict[int, dict[str, Any]] = {}
    for plan in plans:
        generation = int(plan["campaign_generation"])
        contracts = plan.get("contracts")
        if (
            type(contracts) is not list
            or not contracts
            or any(type(value) is not dict for value in contracts)
        ):
            raise TypeError("contextual plan omits its lane contracts")
        controller_waves = {int(value["controller_wave_index"]) for value in contracts}
        if len(controller_waves) != 1:
            raise ValueError("contextual plan lane contracts disagree on wave index")
        controller_wave = controller_waves.pop()
        stage = plan.get("stage_allocation")
        if type(stage) is not dict or type(stage.get("decision")) is not dict:
            raise TypeError("contextual plan omits its stage allocation decision")
        decision = stage["decision"]
        phase = str(decision["phase"])
        parsed: dict[str, list[dict[str, Any]]] = {}
        for kind in ("source", "operator"):
            raw_allocations = decision.get(f"{kind}_allocations")
            if type(raw_allocations) is not list or any(
                type(value) is not dict for value in raw_allocations
            ):
                raise TypeError(f"contextual plan omits {kind} allocations")
            allocations = []
            for allocation in raw_allocations:
                arm_id = allocation.get("arm_id")
                if type(arm_id) is not str or not arm_id:
                    raise TypeError("contextual allocation omits an arm ID")
                row = {
                    "arm_id": arm_id,
                    "target_slots": int(allocation["target_slots"]),
                    "allocation_probability": _hex_or_number(
                        allocation["allocation_probability_hex"]
                    ),
                    "score": _hex_or_number(allocation["score_hex"]),
                    "exploration_slot": allocation.get("exploration_slot") is True,
                }
                allocations.append(row)
                trajectories[kind].setdefault(arm_id, []).append(
                    {"generation": generation, "phase": phase, **row}
                )
            parsed[kind] = allocations
        row = {
            "generation": generation,
            "controller_wave_index": controller_wave,
            "phase": phase,
            "decision_sha256": str(decision["decision_sha256"]),
            "source_allocations": parsed["source"],
            "operator_allocations": parsed["operator"],
        }
        if controller_wave in plan_by_wave:
            raise ValueError("contextual plans contain a duplicate controller wave")
        plan_by_wave[controller_wave] = row
        plan_rows.append(row)
    plan_rows.sort(key=lambda value: value["generation"])

    observations_by_wave: dict[int, list[dict[str, Any]]] = {}
    for observation in observations:
        wave = int(observation["wave_index"])
        observations_by_wave.setdefault(wave, []).append(observation)
    wave_rows = []
    for wave in sorted(set(plan_by_wave) | set(observations_by_wave)):
        plan = plan_by_wave.get(wave)
        rows = observations_by_wave.get(wave, [])

        def planned_counts(kind: str) -> dict[str, int]:
            if plan is None:
                return {}
            return {
                str(value["arm_id"]): int(value["target_slots"])
                for value in plan[f"{kind}_allocations"]
            }

        observed_source = dict(
            sorted(Counter(str(value["source_id"]) for value in rows).items())
        )
        observed_operator = dict(
            sorted(Counter(str(value["operator_id"]) for value in rows).items())
        )
        source_plan = planned_counts("source")
        operator_plan = planned_counts("operator")
        wave_rows.append(
            {
                "wave_index": wave,
                "phase": None if plan is None else plan["phase"],
                "observation_count": len(rows),
                "planned_source_counts": source_plan,
                "observed_source_counts": observed_source,
                "source_realization_exact": source_plan == observed_source,
                "planned_operator_counts": operator_plan,
                "observed_operator_counts": observed_operator,
                "operator_realization_exact": operator_plan == observed_operator,
            }
        )
    comparable_waves = [value for value in wave_rows if value["phase"] is not None]

    return {
        "enabled": True,
        "observation_count": len(observations),
        "plan_count": len(plans),
        "delayed_credit_count": len(delayed_credits),
        "source_rows": aggregate_observations("source_id"),
        "operator_rows": aggregate_observations("operator_id"),
        "wave_rows": wave_rows,
        "plan_rows": plan_rows,
        "source_probability_trajectories": dict(sorted(trajectories["source"].items())),
        "operator_probability_trajectories": dict(
            sorted(trajectories["operator"].items())
        ),
        "exact_source_realization_wave_rate": (
            None
            if not comparable_waves
            else sum(value["source_realization_exact"] for value in comparable_waves)
            / len(comparable_waves)
        ),
        "exact_operator_realization_wave_rate": (
            None
            if not comparable_waves
            else sum(value["operator_realization_exact"] for value in comparable_waves)
            / len(comparable_waves)
        ),
        "delayed_credit": {
            "stage_front_persisted_count": sum(
                value.get("stage_front_persisted") is True for value in delayed_credits
            ),
            "useful_descendant_true_count": sum(
                value.get("useful_descendant_observed") is True
                for value in delayed_credits
            ),
            "useful_descendant_false_count": sum(
                value.get("useful_descendant_observed") is False
                for value in delayed_credits
            ),
            "useful_descendant_unresolved_count": sum(
                value.get("useful_descendant_observed") is None
                for value in delayed_credits
            ),
            "final_front_persisted_count": sum(
                value.get("final_front_persisted") is True for value in delayed_credits
            ),
        },
    }


def analyze_run(
    run_dir: Path,
    *,
    workload_id: str,
    model_profile: str,
    replicate_seed: int,
    arm: str = "treatment",
) -> dict[str, Any]:
    """Return one generic behavioral/quality row from a finalized campaign."""

    root = run_dir.expanduser().resolve(strict=True)
    summary = _json(root / "summary.json")
    campaign_rows = _jsonl(root / "campaign_events.jsonl")
    queue_rows = _jsonl(root / "queue_outcomes.jsonl")
    output_evidence_rows = _jsonl(root / "output_evidence.jsonl")
    completion = _durable_completion_evidence(campaign_rows, summary)
    cutoffs, stages, final_front = _campaign_trace(root)
    failure_endpoint = _failure_endpoint(campaign_rows, queue_rows, cutoffs, stages)
    engine = [_unwrap(value) for value in _jsonl(root / "engine_events.jsonl")]
    first_snapshot = cutoffs[0]["payload"]["archive_utility"]["snapshot_receipt"]
    axes = first_snapshot["spec"]["axes"]
    dimension = len(axes)
    before_hv: dict[int, float] = {}
    before_points: dict[int, list[tuple[float, ...]]] = {}
    for event in cutoffs:
        utility = event["payload"]["archive_utility"]
        generation = int(utility["generation"])
        snapshot = utility["snapshot_receipt"]
        before_hv[generation] = float.fromhex(snapshot["base_hypervolume_hex"])
        before_points[generation] = [
            tuple(float.fromhex(item) for item in point)
            for point in snapshot["normalized_archive_points"]
        ]
    final_points = [_normalize(candidate, axes) for candidate in final_front]
    final_hv = _hypervolume(final_points, dimension)
    accounting = _authenticated_evaluation_accounting(root, summary)

    stage_rows: list[dict[str, Any]] = []
    stage_front_ids: dict[int, set[str]] = {}
    stage_kind: dict[int, str] = {}
    for index, event in enumerate(stages):
        receipt = event["payload"]["stage_receipt"]
        generation = int(receipt["generation"])
        kind = str(receipt["kind"])
        after = final_hv if index == len(stages) - 1 else before_hv[generation + 1]
        front = receipt["result"]["archive_after"]["front_candidates"]
        stage_front_ids[generation] = {value["candidate_id"] for value in front}
        stage_kind[generation] = kind
        stage_rows.append(
            {
                "generation": generation,
                "operator_stage": kind,
                "candidate_occurrences": int(receipt["candidate_occurrence_count"]),
                "hypervolume_before": before_hv[generation],
                "hypervolume_after": after,
                "hypervolume_gain": after - before_hv[generation],
                "front_size_after": len(front),
            }
        )

    physical_trajectory = _physical_evaluation_trajectory(
        engine,
        axes,
        seed_unique_evaluations=int(accounting["seed_unique_evaluations"]),
        seed_archive_points=before_points[min(before_points)],
        allowed_generations=set(stage_kind),
    )
    observed_physical_trajectory = _physical_evaluation_trajectory(
        engine,
        axes,
        seed_unique_evaluations=int(accounting["seed_unique_evaluations"]),
        seed_archive_points=before_points[min(before_points)],
    )
    evaluated = {
        value["candidate_id"]: value
        for value in engine
        if value.get("event_type") == "candidate_evaluated"
    }
    completed = {
        value["candidate_id"]: value
        for value in engine
        if value.get("event_type") == "invocation_completed"
    }
    final_ids = {value["candidate_id"] for value in final_front}
    candidate_rows: list[dict[str, Any]] = []
    for candidate_id, event in evaluated.items():
        label = str(event.get("label", ""))
        generation_match = _GENERATION.search(label)
        if generation_match is None:
            continue
        generation = int(generation_match.group(1))
        if generation not in before_points:
            continue
        objectives = _objective_map(event)
        point: tuple[float, ...] | None = None
        missing_metric_ids = tuple(
            str(axis["metric_id"])
            for axis in axes
            if axis["metric_id"] not in objectives
        )
        if missing_metric_ids:
            detailed = event.get("detailed_evaluation")
            failure = detailed.get("failure") if type(detailed) is dict else None
            if (
                event.get("valid") is not False
                or type(failure) is not dict
                or failure.get("category") != "candidate"
            ):
                raise ValueError(
                    "candidate lacks decision objectives without typed "
                    "candidate-infeasibility evidence"
                )
            marginal = None
            positive_marginal = False
        else:
            point = _normalize(event, axes)
            marginal = (
                _hypervolume([*before_points[generation], point], dimension)
                - before_hv[generation]
            )
            positive_marginal = marginal > 64 * math.ulp(
                max(1.0, abs(before_hv[generation]))
            )
        rank_match = _RANK.search(label)
        parent_slot_match = _PARENT_SLOT.search(label)
        terminal = completed.get(candidate_id, {})
        sealed_stage_kind = stage_kind.get(generation)
        sealed_stage_front = stage_front_ids.get(generation)
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "generation": generation,
                "operator_kind": event.get("operator_kind"),
                # A provider failure can occur after one concurrent lane has
                # already produced and evaluated candidates but before the
                # generation is sealed.  Keep those rank/forecast outcomes as
                # censored behavioral evidence without pretending that they
                # belong to a completed operator stage.
                "operator_stage": (
                    sealed_stage_kind
                    if sealed_stage_kind is not None
                    else "unsealed_partial_stage"
                ),
                "stage_sealed": sealed_stage_kind is not None,
                # Backward-compatible field: historically this was named
                # model_rank, but the label encodes the resolved K4 allocation
                # position.  The authenticated original K8 rank is recovered
                # separately below.
                "model_rank": None if rank_match is None else int(rank_match.group(1)),
                "allocation_rank": (
                    None if rank_match is None else int(rank_match.group(1))
                ),
                "parent_slot": (
                    None
                    if parent_slot_match is None
                    else int(parent_slot_match.group(1)) - 1
                ),
                "scored": marginal is not None,
                "typed_candidate_infeasible": marginal is None,
                "normalized_point": None if point is None else list(point),
                "individual_marginal_hypervolume": marginal,
                "positive_individual_marginal": positive_marginal,
                "admitted_to_stage_front": (
                    sealed_stage_front is not None
                    and candidate_id in sealed_stage_front
                ),
                "admitted_to_final_front": candidate_id in final_ids,
                "dominates_any_parent": terminal.get("dominates_any_parent"),
                "better_relation_any_parent": terminal.get(
                    "better_relation_any_parent"
                ),
                "claimed_insight_count": len(event.get("claimed_insight_ids", [])),
                "insight_credit_status": terminal.get("insight_credit_status"),
            }
        )

    rank_rows: list[dict[str, Any]] = []
    for rank in sorted(
        {
            value["model_rank"]
            for value in candidate_rows
            if value["model_rank"] is not None
        }
    ):
        members = [value for value in candidate_rows if value["model_rank"] == rank]
        marginals = [
            value["individual_marginal_hypervolume"]
            for value in members
            if value["individual_marginal_hypervolume"] is not None
        ]
        rank_rows.append(
            {
                "rank": rank,
                "candidate_count": len(members),
                "scored_candidate_count": sum(value["scored"] for value in members),
                "typed_candidate_infeasible_count": sum(
                    value["typed_candidate_infeasible"] for value in members
                ),
                "positive_individual_marginal_count": sum(
                    value["positive_individual_marginal"] for value in members
                ),
                "stage_front_admission_count": sum(
                    value["admitted_to_stage_front"] for value in members
                ),
                "final_front_admission_count": sum(
                    value["admitted_to_final_front"] for value in members
                ),
                "median_individual_marginal_hypervolume": (
                    None if not marginals else median(marginals)
                ),
            }
        )

    if physical_trajectory[-1]["physical_evaluation"] != int(
        accounting["unique_evaluations"]
    ):
        raise ValueError(
            "physical-evaluation trajectory disagrees with finalized accounting"
        )
    if not physical_trajectory:
        raise ValueError("completed campaign has no physical evaluations")
    observed_physical_evaluations = int(
        observed_physical_trajectory[-1]["physical_evaluation"]
    )
    unsealed_physical_evaluations = observed_physical_evaluations - int(
        accounting["unique_evaluations"]
    )
    if unsealed_physical_evaluations < 0:
        raise ValueError(
            "observed physical evaluations are below sealed campaign accounting"
        )
    raw_health = summary.get("health")
    health = raw_health if type(raw_health) is dict else {}
    operator_totals = []
    for kind in sorted({value["operator_stage"] for value in stage_rows}):
        relevant_stages = [
            value for value in stage_rows if value["operator_stage"] == kind
        ]
        relevant_candidates = [
            value for value in candidate_rows if value["operator_stage"] == kind
        ]
        operator_totals.append(
            {
                "operator_stage": kind,
                "stage_count": len(relevant_stages),
                "candidate_count": len(relevant_candidates),
                "scored_candidate_count": sum(
                    value["scored"] for value in relevant_candidates
                ),
                "typed_candidate_infeasible_count": sum(
                    value["typed_candidate_infeasible"] for value in relevant_candidates
                ),
                "total_stage_hypervolume_gain": sum(
                    value["hypervolume_gain"] for value in relevant_stages
                ),
                "positive_individual_marginal_count": sum(
                    value["positive_individual_marginal"]
                    for value in relevant_candidates
                ),
                "stage_front_admission_count": sum(
                    value["admitted_to_stage_front"] for value in relevant_candidates
                ),
                "final_front_admission_count": sum(
                    value["admitted_to_final_front"] for value in relevant_candidates
                ),
            }
        )
    provider = _provider_summary(queue_rows)
    selector_behavior = _selector_behavior(
        stages,
        provider_backed=provider["logical_calls"] > 0,
    )
    original_rank_and_role_calibration = _original_model_rank_and_allocator_calibration(
        candidate_rows,
        selector_behavior,
    )
    semantic_reconciliation_outcomes = _semantic_reconciliation_outcomes(
        original_rank_and_role_calibration,
        selector_behavior,
    )
    frontier_target_outcomes = _frontier_target_outcomes(
        original_rank_and_role_calibration,
        selector_behavior,
    )
    seed_hv = before_hv[min(before_hv)]
    candidate_accounting = summary.get("candidate_outcome_accounting", {})
    evaluator_accounting = summary.get("evaluator", {})
    engine_terminal_events = [
        value
        for value in engine
        if value.get("event_type") in ("seed_registered", "candidate_evaluated")
    ]
    sealed_engine_terminal_events = []
    for value in engine_terminal_events:
        if value.get("event_type") == "seed_registered":
            sealed_engine_terminal_events.append(value)
            continue
        generation_match = _GENERATION.search(str(value.get("label", "")))
        if (
            generation_match is not None
            and int(generation_match.group(1)) in stage_kind
        ):
            sealed_engine_terminal_events.append(value)
    sealed_candidate_rows = [value for value in candidate_rows if value["stage_sealed"]]
    observed_scored_candidates = sum(
        value.get("valid") is True for value in engine_terminal_events
    )
    sealed_scored_candidates = int(
        candidate_accounting.get(
            "scored_count",
            sum(value.get("valid") is True for value in sealed_engine_terminal_events),
        )
    )
    observed_typed_candidate_infeasible = sum(
        value["typed_candidate_infeasible"] for value in candidate_rows
    )
    sealed_typed_candidate_infeasible = sum(
        value["typed_candidate_infeasible"] for value in sealed_candidate_rows
    )
    runtime_failures = int(
        candidate_accounting.get(
            "runtime_failure_count",
            (
                evaluator_accounting.get("runtime_failures", 0)
                if type(evaluator_accounting) is dict
                else 0
            ),
        )
    )
    return {
        "schema_version": 1,
        "run_id": root.name,
        "run_dir": str(root),
        "workload_id": workload_id,
        "model_profile": model_profile,
        "replicate_seed": replicate_seed,
        "arm": arm,
        "status": completion["summary_status"],
        "health_all_true": (
            all(value is True for value in health.values())
            if completion["summary_health_available"]
            else None
        ),
        "wall_s": completion["wall_s"],
        "completion_evidence": completion,
        "failure_endpoint": failure_endpoint,
        "evaluation": {
            "candidate_occurrences": int(accounting["candidate_occurrences"]),
            "unique_evaluations": int(accounting["unique_evaluations"]),
            "cache_reuse_occurrences": int(accounting["cache_reuse_occurrences"]),
            "scored_candidates": sealed_scored_candidates,
            "observed_scored_candidates": observed_scored_candidates,
            "unsealed_scored_candidates": (
                observed_scored_candidates - sealed_scored_candidates
            ),
            "typed_candidate_infeasible": sealed_typed_candidate_infeasible,
            "observed_typed_candidate_infeasible": (
                observed_typed_candidate_infeasible
            ),
            "unsealed_typed_candidate_infeasible": (
                observed_typed_candidate_infeasible - sealed_typed_candidate_infeasible
            ),
            "runtime_failures": runtime_failures,
            "generations": len(stages),
            "observed_physical_evaluations": observed_physical_evaluations,
            "unsealed_physical_evaluations": unsealed_physical_evaluations,
            **_evaluator_latency(engine),
        },
        "provider": provider,
        "action_forecast_information": _action_forecast_information(
            output_evidence_rows
        ),
        "quality": {
            "is_final": completion["durable_campaign_complete"],
            "endpoint_kind": (
                "completed_campaign_final_archive"
                if completion["durable_campaign_complete"]
                else (
                    "censored_latest_sealed_archive"
                    if stages
                    else "censored_pre_stage_seed_archive"
                )
            ),
            "indicator_dimension": dimension,
            "seed_hypervolume": seed_hv,
            "final_hypervolume": final_hv,
            "absolute_gain": final_hv - seed_hv,
            "relative_gain": None if seed_hv == 0 else (final_hv - seed_hv) / seed_hv,
            "final_front_size": len(final_front),
            "mean_hypervolume_over_physical_evaluations": sum(
                value["hypervolume"] * value["physical_evaluation_span"]
                for value in physical_trajectory
            )
            / int(accounting["unique_evaluations"]),
            "physical_evaluation_trajectory": physical_trajectory,
            "unsealed_observed_hypervolume": (
                None
                if unsealed_physical_evaluations == 0
                else observed_physical_trajectory[-1]["hypervolume"]
            ),
            "unsealed_observed_hypervolume_gain_over_sealed_endpoint": (
                None
                if unsealed_physical_evaluations == 0
                else observed_physical_trajectory[-1]["hypervolume"] - final_hv
            ),
            "trajectory": stage_rows,
        },
        "operators": operator_totals,
        "model_rank_calibration_semantics": (
            "legacy_resolved_k4_candidate_label_rank_not_original_k8_model_rank"
        ),
        "model_rank_calibration": rank_rows,
        "allocation_rank_calibration": rank_rows,
        "original_model_rank_calibration": original_rank_and_role_calibration[
            "original_model_rank_rows"
        ],
        "allocator_role_calibration": original_rank_and_role_calibration[
            "allocator_role_rows"
        ],
        "rank_role_join": original_rank_and_role_calibration,
        "semantic_reconciliation_outcomes": semantic_reconciliation_outcomes,
        "frontier_target_outcomes": frontier_target_outcomes,
        "proposal_support_calibration": original_rank_and_role_calibration[
            "proposal_support_calibration"
        ],
        "contextual_search_controller": _contextual_search_behavior(summary),
        "selector_behavior": selector_behavior,
        "forecast_calibration": _forecast_calibration(stages, engine),
        "memory_and_reflection": {
            "candidate_outputs_claiming_insights": sum(
                value["claimed_insight_count"] > 0 for value in candidate_rows
            ),
            "insight_credit_status_counts": {
                str(status): sum(
                    value["insight_credit_status"] == status for value in candidate_rows
                )
                for status in sorted(
                    {value["insight_credit_status"] for value in candidate_rows},
                    key=str,
                )
            },
            **_reflection_schema(summary),
            **_memory_lifecycle(summary, stages),
        },
    }


def _flat_row(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": value["run_id"],
        "workload_id": value["workload_id"],
        "model_profile": value["model_profile"],
        "replicate_seed": value["replicate_seed"],
        "arm": value["arm"],
        "status": value["status"],
        "health_all_true": value["health_all_true"],
        "wall_s": value["wall_s"],
        "wall_time_source": value["completion_evidence"]["wall_time_source"],
        "summary_health_available": value["completion_evidence"][
            "summary_health_available"
        ],
        "evolution_completed": value["completion_evidence"]["evolution_completed"],
        "runtime_released": value["completion_evidence"]["runtime_released"],
        "durable_campaign_complete": value["completion_evidence"][
            "durable_campaign_complete"
        ],
        "campaign_failed": value["failure_endpoint"]["campaign_failed"],
        "campaign_failure_type": value["failure_endpoint"]["campaign_failure_type"],
        "failed_generation": value["failure_endpoint"]["failed_generation"],
        "last_sealed_generation": value["failure_endpoint"]["last_sealed_generation"],
        **{f"evaluation_{key}": item for key, item in value["evaluation"].items()},
        "logical_calls": value["provider"]["logical_calls"],
        "successful_logical_calls": value["provider"]["successful_logical_calls"],
        "failed_logical_calls": value["provider"]["failed_logical_calls"],
        "cancelled_logical_calls": value["provider"]["cancelled_logical_calls"],
        "physical_attempts": value["provider"]["physical_attempts"],
        "retry_attempts": value["provider"]["retry_attempts"],
        "input_tokens": value["provider"]["input_tokens"],
        "output_tokens": value["provider"]["output_tokens"],
        "reasoning_tokens": value["provider"]["reasoning_tokens"],
        "cost_usd": value["provider"]["cost_usd"],
        "provider_latency_s": value["provider"]["provider_latency_s"],
        "action_forecast_call_count": value["action_forecast_information"][
            "forecast_call_count"
        ],
        "action_forecast_action_count": value["action_forecast_information"][
            "action_count"
        ],
        "action_forecast_distinct_effect_code_count": value[
            "action_forecast_information"
        ]["distinct_effect_code_count"],
        "action_forecast_effect_entropy_nats": value["action_forecast_information"][
            "effect_entropy_nats"
        ],
        "action_forecast_zero_effect_cell_rate": value["action_forecast_information"][
            "zero_effect_cell_rate"
        ],
        "action_forecast_high_validity_action_rate": value[
            "action_forecast_information"
        ]["high_validity_action_rate"],
        "action_forecast_most_common_signature_rate": value[
            "action_forecast_information"
        ]["most_common_full_action_signature_rate"],
        "action_forecast_information_json": json.dumps(
            value["action_forecast_information"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "selector_unique_option_count": value["selector_behavior"][
            "unique_option_count"
        ],
        "selector_selection_mode_counts_json": json.dumps(
            value["selector_behavior"]["selection_mode_counts"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "selector_outcome_conditioned_call_count": value["selector_behavior"][
            "outcome_conditioned_call_count"
        ],
        "selector_outcome_conditioned_forecast_universe_row_count_total": value[
            "selector_behavior"
        ]["outcome_conditioned_forecast_universe_row_count_total"],
        "selector_outcome_conditioned_allocator_candidate_evaluations_total": value[
            "selector_behavior"
        ]["outcome_conditioned_allocator_candidate_evaluations_total"],
        "selector_effective_option_count": value["selector_behavior"][
            "effective_option_count"
        ],
        "selector_exact_set_witness_copy_rate": value["selector_behavior"][
            "exact_set_witness_copy_rate"
        ],
        "selector_mean_cross_lane_option_jaccard": value["selector_behavior"][
            "mean_cross_lane_option_jaccard"
        ],
        "selector_common_pool_call_count": value["selector_behavior"][
            "common_pool_call_count"
        ],
        "selector_mean_common_universe_selection_fraction": value["selector_behavior"][
            "mean_common_universe_selection_fraction"
        ],
        "selector_mean_common_universe_evaluation_fraction": value["selector_behavior"][
            "mean_common_universe_evaluation_fraction"
        ],
        "selector_common_pool_allocator_replacement_rate": value["selector_behavior"][
            "common_pool_allocator_replacement_rate"
        ],
        "selector_proposal_support_reservation_evaluation_rate": value[
            "selector_behavior"
        ]["proposal_support_reservation_evaluation_rate"],
        "selector_proposal_support_evaluator_slot_share": value["selector_behavior"][
            "proposal_support_evaluator_slot_share"
        ],
        "selector_proposal_support_authenticated_deferral_count": value[
            "selector_behavior"
        ]["proposal_support_authenticated_deferral_count"],
        "selector_proposal_support_reconciled_membership_exact_call_rate": value[
            "selector_behavior"
        ]["proposal_support_reconciled_membership_exact_call_rate"],
        "selector_frontier_context_enabled_rate": value["selector_behavior"][
            "frontier_context_enabled_rate"
        ],
        "selector_frontier_context_distinct_projection_count": value[
            "selector_behavior"
        ]["frontier_context_distinct_projection_count"],
        "selector_frontier_target_enabled_rate": value["selector_behavior"][
            "frontier_target_enabled_rate"
        ],
        "selector_frontier_target_distinct_target_count": value["selector_behavior"][
            "frontier_target_distinct_target_count"
        ],
        "selector_frontier_target_direction_counts_json": json.dumps(
            value["selector_behavior"]["frontier_target_direction_counts"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "selector_frontier_target_mean_opportunity_from_ideal": value[
            "selector_behavior"
        ]["frontier_target_mean_opportunity_from_ideal"],
        "selector_frontier_target_mean_parent_regret_above_archive_best": value[
            "selector_behavior"
        ]["frontier_target_mean_parent_regret_above_archive_best"],
        "selector_empirical_card_evaluated_citation_member_count": value[
            "selector_behavior"
        ]["empirical_card_evaluated_citation_member_count"],
        "selector_evaluated_card_citation_member_count": value["selector_behavior"][
            "evaluated_card_citation_member_count"
        ],
        "selector_evaluated_card_citation_without_exact_finite_target_count": value[
            "selector_behavior"
        ]["evaluated_card_citation_without_exact_finite_target_count"],
        "selector_empirical_card_evaluated_exact_target_member_count": value[
            "selector_behavior"
        ]["empirical_card_evaluated_exact_target_member_count"],
        (
            "selector_empirical_card_evaluated_cross_target_generalization_member_count"
        ): value["selector_behavior"][
            "empirical_card_evaluated_cross_target_generalization_member_count"
        ],
        "selector_role_entropy_nats": value["selector_behavior"]["role_entropy_nats"],
        "selector_mean_within_call_rationale_token_jaccard": value["selector_behavior"][
            "mean_within_call_rationale_token_jaccard"
        ],
        "selector_semantic_reconciliation_call_count": value["selector_behavior"][
            "semantic_reconciliation"
        ]["call_count"],
        "selector_semantic_engine_insertion_rate": value["selector_behavior"][
            "semantic_reconciliation"
        ]["engine_insertion_rate"],
        "selector_semantic_evaluated_engine_member_rate": value["selector_behavior"][
            "semantic_reconciliation"
        ]["evaluated_engine_member_rate"],
        "selector_semantic_reconciliation_objective_blind_call_rate": value[
            "selector_behavior"
        ]["semantic_reconciliation"]["objective_blind_call_rate"],
        "selector_contextual_projection_call_count": value["selector_behavior"][
            "semantic_reconciliation"
        ]["contextual_allocation_projection"]["call_count"],
        "selector_contextual_projection_exact_call_rate": value["selector_behavior"][
            "semantic_reconciliation"
        ]["contextual_allocation_projection"]["exact_call_rate"],
        "selector_contextual_projection_source_l1_deviation": value[
            "selector_behavior"
        ]["semantic_reconciliation"]["contextual_allocation_projection"][
            "source_l1_deviation"
        ],
        "selector_contextual_projection_operator_l1_deviation": value[
            "selector_behavior"
        ]["semantic_reconciliation"]["contextual_allocation_projection"][
            "operator_l1_deviation"
        ],
        "selector_composition_capacity_projection_call_count": value[
            "selector_behavior"
        ]["semantic_reconciliation"]["composition_capacity_projection"]["call_count"],
        "selector_composition_capacity_projected_call_rate": value["selector_behavior"][
            "semantic_reconciliation"
        ]["composition_capacity_projection"]["capacity_projected_call_rate"],
        "selector_composition_capacity_absolute_projection_distance_total": value[
            "selector_behavior"
        ]["semantic_reconciliation"]["composition_capacity_projection"][
            "absolute_projection_distance_total"
        ],
        "semantic_reconciliation_outcomes_json": json.dumps(
            value["semantic_reconciliation_outcomes"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "frontier_target_joined_candidate_count": value["frontier_target_outcomes"][
            "joined_candidate_count"
        ],
        "frontier_target_improves_assigned_parent_count": value[
            "frontier_target_outcomes"
        ]["improves_assigned_parent_count"],
        "frontier_target_beats_prior_archive_best_count": value[
            "frontier_target_outcomes"
        ]["beats_prior_archive_best_count"],
        "frontier_target_residual_joined_candidate_count": value[
            "frontier_target_outcomes"
        ]["residual_target_joined_candidate_count"],
        "frontier_target_attains_or_dominates_aspiration_count": value[
            "frontier_target_outcomes"
        ]["attains_or_dominates_residual_aspiration_count"],
        "frontier_target_reduces_aspiration_shortfall_count": value[
            "frontier_target_outcomes"
        ]["reduces_residual_aspiration_shortfall_count"],
        "frontier_target_outcomes_json": json.dumps(
            value["frontier_target_outcomes"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "model_rank_calibration_semantics": value["model_rank_calibration_semantics"],
        "rank_role_allocated_slot_count": value["rank_role_join"][
            "allocated_slot_count"
        ],
        "rank_role_joined_candidate_count": value["rank_role_join"][
            "joined_candidate_count"
        ],
        "rank_role_unjoined_candidate_label_count": value["rank_role_join"][
            "unjoined_candidate_label_count"
        ],
        "proposal_support_calibration_json": json.dumps(
            value["proposal_support_calibration"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "contextual_search_enabled": value["contextual_search_controller"]["enabled"],
        "contextual_search_observation_count": value["contextual_search_controller"][
            "observation_count"
        ],
        "contextual_search_plan_count": value["contextual_search_controller"][
            "plan_count"
        ],
        "contextual_search_exact_source_realization_wave_rate": value[
            "contextual_search_controller"
        ]["exact_source_realization_wave_rate"],
        "contextual_search_exact_operator_realization_wave_rate": value[
            "contextual_search_controller"
        ]["exact_operator_realization_wave_rate"],
        "contextual_search_controller_json": json.dumps(
            value["contextual_search_controller"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "original_model_rank_calibration_json": json.dumps(
            value["original_model_rank_calibration"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "allocator_role_calibration_json": json.dumps(
            value["allocator_role_calibration"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "forecast_direction_accuracy": value["forecast_calibration"][
            "direction_accuracy"
        ],
        "forecast_high_confidence_direction_accuracy": value["forecast_calibration"][
            "high_confidence_direction_accuracy"
        ],
        "forecast_high_confidence_direction_error_count": value["forecast_calibration"][
            "high_confidence_direction_error_count"
        ],
        "forecast_unknown_direction_rate": value["forecast_calibration"][
            "unknown_direction_forecast_rate"
        ],
        "forecast_unscorable_member_count": value["forecast_calibration"][
            "unscorable_forecast_member_count"
        ],
        "forecast_unscorable_missing_event_member_count": value["forecast_calibration"][
            "unscorable_missing_event_member_count"
        ],
        "forecast_unscorable_invalid_candidate_member_count": value[
            "forecast_calibration"
        ]["unscorable_invalid_candidate_member_count"],
        "forecast_unscorable_objective_payload_member_count": value[
            "forecast_calibration"
        ]["unscorable_objective_payload_member_count"],
        "quality_is_final": value["quality"]["is_final"],
        "quality_endpoint_kind": value["quality"]["endpoint_kind"],
        "seed_hypervolume": value["quality"]["seed_hypervolume"],
        "final_hypervolume": value["quality"]["final_hypervolume"],
        "absolute_gain": value["quality"]["absolute_gain"],
        "relative_gain": value["quality"]["relative_gain"],
        "final_front_size": value["quality"]["final_front_size"],
        "mean_hypervolume_over_physical_evaluations": value["quality"][
            "mean_hypervolume_over_physical_evaluations"
        ],
        "decimal_magnitude_prompt_contract": value["memory_and_reflection"][
            "decimal_magnitude_prompt_contract"
        ],
        "hex_exponent_interpretation_risk": value["memory_and_reflection"][
            "hex_exponent_interpretation_risk"
        ],
        "memory_reflected_entry_count": value["memory_and_reflection"][
            "reflected_entry_count"
        ],
        "memory_reflected_retrievable_count": value["memory_and_reflection"][
            "reflected_retrievable_count"
        ],
        "advisory_memory_lane_count": value["memory_and_reflection"][
            "advisory_memory_lane_count"
        ],
        "advisory_memory_selected_card_count": value["memory_and_reflection"][
            "advisory_memory_selected_card_count"
        ],
        "advisory_memory_exact_parent_match_count": value["memory_and_reflection"][
            "advisory_memory_exact_parent_match_count"
        ],
        "advisory_memory_exact_replay_authorized_count": value["memory_and_reflection"][
            "advisory_memory_exact_replay_authorized_count"
        ],
        "memory_assignment_credit_count": value["memory_and_reflection"][
            "memory_assignment_credit_count"
        ],
        "memory_assignment_total_wave_reward": value["memory_and_reflection"][
            "memory_assignment_total_wave_reward"
        ],
        "matched_memory_control_block_count": value["memory_and_reflection"][
            "matched_memory_control_block_count"
        ],
        "matched_memory_total_active_minus_neutral": value["memory_and_reflection"][
            "matched_memory_total_active_minus_neutral"
        ],
        "memory_supported_action_count": value["memory_and_reflection"][
            "memory_supported_action_count"
        ],
        "memory_supported_action_positive_reward_count": value["memory_and_reflection"][
            "memory_supported_action_positive_reward_count"
        ],
        "memory_positive_wave_nonpositive_supported_action_count": value[
            "memory_and_reflection"
        ]["memory_positive_wave_nonpositive_supported_action_count"],
        "memory_cross_lane_action_spillover_count": value["memory_and_reflection"][
            "memory_cross_lane_action_spillover_count"
        ],
        "memory_action_performance_causal_claim_allowed": value[
            "memory_and_reflection"
        ]["memory_action_performance_causal_claim_allowed"],
        "memory_semantic_audit_contradicted_count": value["memory_and_reflection"][
            "semantic_audit_verdict_counts"
        ].get("contradicted", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    spec = _json(args.spec.resolve(strict=True))
    runs = spec.get("runs")
    if type(runs) is not list or not runs:
        raise ValueError("analysis spec requires a non-empty runs list")
    rows = [
        analyze_run(
            Path(value["run_dir"]),
            workload_id=value["workload_id"],
            model_profile=value["model_profile"],
            replicate_seed=int(value["replicate_seed"]),
            arm=value.get("arm", "treatment"),
        )
        for value in runs
    ]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"schema_version": 1, "runs": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    flat = [_flat_row(value) for value in rows]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(
        json.dumps(
            {
                "run_count": len(rows),
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
