"""Canonical, provider-free analyzer for one sealed Airfoil-v7 G3 result."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Sequence

from agent_evolve.agentic import AgenticOptimizerComposition
from agent_evolve.application.agentic_evolution import ReflectionCallStatus
from agent_evolve.application.budgeted_optimizer import OptimizerResult
from agent_evolve.application.g3_causal_screen import G3CausalScreenPlanner
from agent_evolve.application.g3_postseal_curation import (
    G3PostsealCurationInterceptor,
)
from agent_evolve.application.g3_causal_validation import (
    validate_g3_causal_screen_result,
    validate_g3_terminal_state,
)
from agent_evolve.domain.artifact import artifact_ref_for_bytes
from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json, thaw_json
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    FINITE_VARIATION_SELECTION_TOOL_NAME,
    REFLECTION_PROMPT_RENDERER_DEFINITION_SHA256,
    REFLECTION_PROMPT_RENDERER_ID,
    REFLECTION_PROMPT_RENDERER_REVISION,
    REFLECTION_TOOL_NAME,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    SUPPORTED_STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSIONS,
    validate_structured_generation_output_evidence_record,
    validate_structured_generation_request_evidence_record,
)
from agent_evolve.ports.agentic_generator import AgenticCallTelemetry
from agent_evolve.ports.artifact_store import JSON_MEDIA_TYPE, decode_json_bytes
from agent_evolve.ports.structured_generator import (
    IDENTITY_PROMPT_RENDERER_DEFINITION_SHA256,
    IDENTITY_PROMPT_RENDERER_ID,
    IDENTITY_PROMPT_RENDERER_REVISION,
)
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    ADFLOW_EVALUATOR_ID,
    EVIDENCE_CONTRACT_ID,
    V2_EVALUATOR_ID,
)
from examples.benchmarks.engibench_airfoil.problem_def import candidate_sha256
from examples.benchmarks.engibench_airfoil.v7_g3_live import (
    DEEPSEEK_G3_PROVIDER_PROFILE,
    AirfoilG3ProviderProfile,
    build_telemetry_policy,
)
from examples.benchmarks.engibench_airfoil.v7_g3_runtime import (
    AirfoilG3RuntimeInputs,
)


_ANALYSIS_DOMAIN = b"agent-evolve:airfoil-v7-g3-live-analysis:v2\x00"


class AirfoilG3AnalysisError(RuntimeError):
    """The live result cannot support the preregistered G3 claim."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _telemetry_record(
    values: Sequence[AgenticCallTelemetry],
    *,
    provider_profile: AirfoilG3ProviderProfile,
) -> dict[str, object]:
    policy = build_telemetry_policy(provider_profile)
    for value in values:
        policy.validate(value)
    costs = tuple(value.cost_usd for value in values)
    if any(type(value) is not Decimal for value in costs):
        raise AirfoilG3AnalysisError("one live call lacks exact cost telemetry")
    response_ids = tuple(value.provider_response_id for value in values)
    if any(type(value) is not str or not value for value in response_ids) or len(
        set(response_ids)
    ) != len(response_ids):
        raise AirfoilG3AnalysisError("provider response identities are absent or repeated")
    complete = len(values) == 6
    observed_input = sum(value.input_tokens for value in values)
    observed_output = sum(value.output_tokens for value in values)
    observed_reasoning = sum(value.reasoning_tokens for value in values)
    observed_cost = sum(costs, Decimal("0"))
    return {
        "logical_calls_attempted": 6,
        "successful_response_telemetry_rows": len(values),
        "accounting_complete": complete,
        "requested_models": sorted({value.requested_model for value in values}),
        "resolved_models": sorted({value.resolved_model for value in values}),
        "resolved_providers": sorted({value.resolved_provider for value in values}),
        "provider_response_ids": [value.provider_response_id for value in values],
        "attempt_counts": [value.attempt_count for value in values],
        "observed_success_input_tokens": observed_input,
        "observed_success_output_tokens": observed_output,
        "observed_success_reasoning_tokens": observed_reasoning,
        "observed_success_reported_cost_usd": str(observed_cost),
        "total_input_tokens": observed_input if complete else None,
        "total_output_tokens": observed_output if complete else None,
        "total_reasoning_tokens": observed_reasoning if complete else None,
        "reported_total_cost_usd": str(observed_cost) if complete else None,
        "provider_latency_seconds_observed_success_sum": (
            sum(value.latency_ns for value in values) / 1_000_000_000.0
        ),
        "provider_latency_seconds_observed_success_max": (
            max(value.latency_ns for value in values) / 1_000_000_000.0
        ),
        "unknown_accounting_reason": (
            None
            if complete
            else "one_or_more_logical_tasks_lack_success_telemetry"
        ),
        "telemetry_policy_sha256": policy.policy_sha256,
    }


@dataclass(frozen=True, slots=True)
class _ExpectedProviderCall:
    operation: str
    semantic_prompt_sha256: str
    renderer_id: str
    renderer_revision: str
    renderer_definition_sha256: str
    output_tool_name: str
    telemetry: AgenticCallTelemetry | None
    is_reflection: bool


def _telemetry_response_record(value: AgenticCallTelemetry) -> dict[str, object]:
    value.__post_init__()
    return {
        "requested_model": value.requested_model,
        "resolved_model": value.resolved_model,
        "resolved_provider": value.resolved_provider,
        "provider_response_id": value.provider_response_id,
        "finish_reason": value.finish_reason,
        "input_tokens": value.input_tokens,
        "output_tokens": value.output_tokens,
        "reasoning_tokens": value.reasoning_tokens,
        "cache_read_tokens": value.cache_read_tokens,
        "cache_write_tokens": value.cache_write_tokens,
        "cost_usd": None if value.cost_usd is None else str(value.cost_usd),
        "latency_ns": value.latency_ns,
    }


def _request_evidence_record(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_calls: Mapping[str, _ExpectedProviderCall],
    provider_profile: AirfoilG3ProviderProfile,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    if len(rows) != 6:
        raise AirfoilG3AnalysisError("request-evidence ledger must contain six rows")
    by_call: dict[str, dict[str, object]] = {}
    for raw in rows:
        try:
            row = validate_structured_generation_request_evidence_record(raw)
        except (TypeError, ValueError) as exc:
            raise AirfoilG3AnalysisError(
                "one structured request-evidence row is invalid"
            ) from exc
        call_id = str(row["call_id"])
        if call_id in by_call or call_id not in expected_calls:
            raise AirfoilG3AnalysisError(
                "request-evidence call identities are repeated or foreign"
            )
        expected = expected_calls[call_id]
        if (
            row["operation"] != expected.operation
            or row["semantic_prompt_sha256"]
            != expected.semantic_prompt_sha256
            or row["prompt_renderer_id"] != expected.renderer_id
            or row["prompt_renderer_revision"] != expected.renderer_revision
            or row["prompt_renderer_definition_sha256"]
            != expected.renderer_definition_sha256
            or row["output_tool_name"] != expected.output_tool_name
            or row["max_output_tokens"] != provider_profile.max_output_tokens
            or row["temperature_hex"]
            != (
                None
                if provider_profile.temperature is None
                else float(provider_profile.temperature).hex()
            )
        ):
            raise AirfoilG3AnalysisError(
                "structured request evidence differs from its engine call"
            )
        by_call[call_id] = row
    if set(by_call) != set(expected_calls):
        raise AirfoilG3AnalysisError(
            "request-evidence call IDs differ from the exact 5+1 topology"
        )
    ordered = sorted(by_call)
    return (
        {
            "request_records": 6,
            "request_evidence_sha256_by_call": {
                call_id: by_call[call_id]["request_evidence_sha256"]
                for call_id in ordered
            },
            "wire_prompt_sha256_by_call": {
                call_id: by_call[call_id]["wire_prompt_sha256"]
                for call_id in ordered
            },
            "semantic_prompt_sha256_by_call": {
                call_id: by_call[call_id]["semantic_prompt_sha256"]
                for call_id in ordered
            },
            "all_engine_prompt_and_renderer_joins_exact": True,
        },
        by_call,
    )


def _provider_attempt_record(
    rows: Sequence[Mapping[str, object]],
    *,
    curation_status: str,
    reflection_call_id: str,
    expected_calls: Mapping[str, _ExpectedProviderCall],
    requests_by_call: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    if len(rows) != 6:
        raise AirfoilG3AnalysisError("provider ledger must contain six logical tasks")
    rows_by_call: dict[str, dict[str, object]] = {}
    attempt_count = 0
    response_ids: dict[str, str] = {}
    terminal_failure_statuses = {
        "terminal_failure",
        "attempts_exhausted",
        "cancelled",
    }
    for raw in rows:
        row = dict(raw)
        task_id = row.get("task_id")
        status = row.get("status")
        attempts = row.get("attempts")
        response = row.get("response")
        if (
            row.get("schema_version")
            not in SUPPORTED_STRUCTURED_GENERATION_OUTCOME_SCHEMA_VERSIONS
            or type(task_id) is not str
            or not task_id
            or task_id in rows_by_call
            or task_id not in expected_calls
            or status not in {"succeeded", *terminal_failure_statuses}
            or type(attempts) is not list
            or not 1 <= len(attempts) <= 2
        ):
            raise AirfoilG3AnalysisError("provider ledger row is malformed")
        rows_by_call[task_id] = row
        attempt_count += len(attempts)
        for number, attempt in enumerate(attempts, start=1):
            evidence = attempt.get("request_evidence") if type(attempt) is dict else None
            if (
                type(attempt) is not dict
                or attempt.get("attempt_number") != number
                or type(evidence) is not dict
                or evidence.get("variant") != "original"
                or type(evidence.get("prompt_sha256")) is not str
                or type(evidence.get("provider_attempt_id")) is not str
                or evidence.get("prompt_sha256")
                != requests_by_call[task_id]["wire_prompt_sha256"]
            ):
                raise AirfoilG3AnalysisError(
                    "provider physical attempt is not exact-payload authenticated"
                )
        expected = expected_calls[task_id]
        if not expected.is_reflection and status != "succeeded":
            raise AirfoilG3AnalysisError("one optimization proposal call failed")
        if status == "succeeded":
            telemetry = expected.telemetry
            if telemetry is None or type(response) is not dict or response != (
                _telemetry_response_record(telemetry)
            ) or len(attempts) != telemetry.attempt_count:
                raise AirfoilG3AnalysisError(
                    "provider response differs from exact engine telemetry"
                )
            response_id = response.get("provider_response_id")
            if type(response_id) is not str or not response_id:
                raise AirfoilG3AnalysisError("provider response identity is absent")
            response_ids[task_id] = response_id
        elif (
            task_id != reflection_call_id
            or curation_status != "incomplete"
            or response is not None
            or expected.telemetry is not None
        ):
            raise AirfoilG3AnalysisError(
                "only isolated postseal curation may end as provider failure"
            )
    if set(rows_by_call) != set(expected_calls):
        raise AirfoilG3AnalysisError("provider task IDs differ from engine call IDs")
    if len(set(response_ids.values())) != len(response_ids):
        raise AirfoilG3AnalysisError("provider response IDs are not unique")
    return (
        {
            "logical_tasks": 6,
            "physical_attempts": attempt_count,
            "maximum_physical_attempts": 12,
            "statuses_by_call": {
                call_id: rows_by_call[call_id]["status"]
                for call_id in sorted(rows_by_call)
            },
            "provider_response_ids_by_call": {
                call_id: response_ids[call_id] for call_id in sorted(response_ids)
            },
            "call_ids": sorted(rows_by_call),
            "all_attempts_exact_original_payload": True,
            "all_successes_join_engine_telemetry_row_by_row": True,
            "only_isolated_curation_failure_allowed": True,
        },
        rows_by_call,
    )


def _output_evidence_record(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_calls: Mapping[str, _ExpectedProviderCall],
    requests_by_call: Mapping[str, Mapping[str, object]],
    provider_rows_by_call: Mapping[str, Mapping[str, object]],
    reflection_call_id: str,
    reflection_status: ReflectionCallStatus,
    reflection_failure_type: str | None,
    reflection_publications: int,
    curation_status: str,
) -> tuple[dict[str, object], str]:
    successful_calls = {
        call_id
        for call_id, row in provider_rows_by_call.items()
        if row["status"] == "succeeded"
    }
    if len(rows) != len(successful_calls):
        raise AirfoilG3AnalysisError(
            "typed-output ledger cardinality differs from provider successes"
        )
    by_call: dict[str, dict[str, object]] = {}
    for raw in rows:
        raw_call_id = raw.get("call_id")
        if type(raw_call_id) is not str or raw_call_id not in requests_by_call:
            raise AirfoilG3AnalysisError("typed-output evidence has a foreign call")
        try:
            row = validate_structured_generation_output_evidence_record(
                raw,
                request_evidence=requests_by_call[raw_call_id],
            )
        except (TypeError, ValueError) as exc:
            raise AirfoilG3AnalysisError(
                "one typed-output evidence row is invalid"
            ) from exc
        call_id = str(row["call_id"])
        if call_id in by_call or call_id not in successful_calls:
            raise AirfoilG3AnalysisError(
                "typed-output evidence identities are repeated or non-successful"
            )
        provider_response = provider_rows_by_call[call_id]["response"]
        assert type(provider_response) is dict
        if (
            row["provider_response_id"]
            != provider_response["provider_response_id"]
            or row["operation"] != expected_calls[call_id].operation
        ):
            raise AirfoilG3AnalysisError(
                "typed-output evidence differs from its provider/engine call"
            )
        by_call[call_id] = row
    if set(by_call) != successful_calls:
        raise AirfoilG3AnalysisError(
            "typed-output evidence does not cover every provider success exactly"
        )

    reflection_provider_succeeded = reflection_call_id in successful_calls
    reflection_output = by_call.get(reflection_call_id)
    if not reflection_provider_succeeded:
        if (
            reflection_output is not None
            or reflection_status is not ReflectionCallStatus.FAILED
            or curation_status != "incomplete"
        ):
            raise AirfoilG3AnalysisError("provider failure classification is inconsistent")
        classification = "provider_failure"
        reflection_insight_count = None
        reflection_output_sha256 = None
    else:
        if reflection_output is None:
            raise AirfoilG3AnalysisError("successful reflection lacks typed output")
        typed_output = reflection_output["typed_output"]
        if type(typed_output) is not dict or type(typed_output.get("insights")) is not list:
            raise AirfoilG3AnalysisError("reflection typed output lacks exact insights")
        reflection_insight_count = len(typed_output["insights"])
        reflection_output_sha256 = reflection_output["output_evidence_sha256"]
        if reflection_insight_count == 0:
            if (
                reflection_status is not ReflectionCallStatus.COMPLETED
                or reflection_failure_type is not None
                or reflection_publications != 0
                or curation_status != "sealed_complete"
            ):
                raise AirfoilG3AnalysisError("true abstention evidence is inconsistent")
            classification = "true_abstention"
        elif reflection_status is ReflectionCallStatus.COMPLETED:
            if (
                reflection_failure_type is not None
                or reflection_publications != 1
                or curation_status != "sealed_complete"
            ):
                raise AirfoilG3AnalysisError("accepted revision evidence is inconsistent")
            classification = "accepted_revision"
        else:
            if (
                reflection_failure_type != "ReflectionCardContractError"
                or reflection_publications != 0
                or curation_status != "incomplete"
            ):
                raise AirfoilG3AnalysisError(
                    "non-empty rejected reflection evidence is inconsistent"
                )
            classification = "rejected_nonempty"
    return (
        {
            "output_records": len(by_call),
            "output_evidence_sha256_by_call": {
                call_id: by_call[call_id]["output_evidence_sha256"]
                for call_id in sorted(by_call)
            },
            "typed_output_sha256_by_call": {
                call_id: by_call[call_id]["typed_output_sha256"]
                for call_id in sorted(by_call)
            },
            "all_provider_successes_have_exactly_one_typed_output": True,
            "reflection_output_evidence_sha256": reflection_output_sha256,
            "reflection_insight_count": reflection_insight_count,
            "curation_classification": classification,
        },
        classification,
    )


def _raw_evaluator_record(
    paths: Sequence[Path],
    *,
    expected_artifact_candidates: Mapping[str, str],
) -> dict[str, object]:
    if len(paths) != 11:
        raise AirfoilG3AnalysisError("raw evaluator ledger must contain 11 receipts")
    candidate_sha256s: set[str] = set()
    artifact_ids: set[str] = set()
    file_sha256s: list[str] = []
    solver_points = 0
    for path in paths:
        content = path.expanduser().resolve(strict=True).read_bytes()
        value = decode_json_bytes(content)
        if type(value) is not dict:
            raise AirfoilG3AnalysisError("raw evaluator receipt is not an object")
        points = value.get("points")
        candidate_sha256 = value.get("candidate_sha256")
        if (
            value.get("schema_version") != 2
            or value.get("evaluator_id") != V2_EVALUATOR_ID
            or value.get("status") != "evaluated"
            or value.get("evaluator_calls") != 3
            or type(candidate_sha256) is not str
            or len(candidate_sha256) != 64
            or type(points) is not list
            or len(points) != 3
        ):
            raise AirfoilG3AnalysisError("raw evaluator receipt contract drifted")
        for index, point in enumerate(points):
            evidence = point.get("evaluator_evidence") if type(point) is dict else None
            if (
                type(point) is not dict
                or point.get("index") != index
                or type(evidence) is not dict
                or evidence.get("contract_id") != EVIDENCE_CONTRACT_ID
                or evidence.get("evaluator_id") != ADFLOW_EVALUATOR_ID
                or evidence.get("accepted") is not True
            ):
                raise AirfoilG3AnalysisError("one solver point lacks accepted evidence")
        candidate_sha256s.add(candidate_sha256)
        artifact_id = artifact_ref_for_bytes(
            content,
            media_type=JSON_MEDIA_TYPE,
        ).artifact_id.value
        artifact_ids.add(artifact_id)
        if expected_artifact_candidates.get(artifact_id) != candidate_sha256:
            raise AirfoilG3AnalysisError(
                "raw receipt candidate identity differs from its engine occurrence"
            )
        file_sha256s.append(hashlib.sha256(content).hexdigest())
        solver_points += len(points)
    if (
        len(candidate_sha256s) != 11
        or artifact_ids != set(expected_artifact_candidates)
        or solver_points != 33
    ):
        raise AirfoilG3AnalysisError(
            "raw receipts do not prove 11 physical evaluations / 33 solver points"
        )
    return {
        "receipt_count": 11,
        "unique_candidate_sha256s": sorted(candidate_sha256s),
        "receipt_file_sha256s": sorted(file_sha256s),
        "artifact_ids": sorted(artifact_ids),
        "solver_point_calls": 33,
        "reproduction_raw_receipt_created": False,
    }


@dataclass(frozen=True, slots=True)
class AirfoilG3AnalysisReceipt:
    runtime_manifest_sha256: str
    freeze_receipt_sha256: str
    runtime_inputs_sha256: str
    optimizer_result_sha256: str
    result_validation_receipt_sha256: str
    terminal_validation_receipt_sha256: str
    reflection_call_id: str
    reflection_call_receipt_sha256: str
    reflection_call_status: str
    reflection_failure_type: str | None
    curation_status: str
    curation_classification: str
    mechanism_decision: FrozenJsonObject
    cache_evidence: FrozenJsonObject
    telemetry: FrozenJsonObject
    provider_attempts: FrozenJsonObject
    structured_evidence: FrozenJsonObject
    raw_evaluator: FrozenJsonObject
    physical_receipt_artifact_ids: tuple[str, ...]
    run_started_at_utc: str
    run_finished_at_utc: str
    end_to_end_wall_seconds: float
    unique_evaluator_wall_seconds_sum: float
    unique_evaluator_wall_seconds_max: float
    replication_recommended: bool
    analysis_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "runtime_manifest_sha256",
            "freeze_receipt_sha256",
            "runtime_inputs_sha256",
            "optimizer_result_sha256",
            "result_validation_receipt_sha256",
            "terminal_validation_receipt_sha256",
            "reflection_call_receipt_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(part not in "0123456789abcdef" for part in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.curation_status not in {"sealed_complete", "incomplete"}:
            raise ValueError("curation_status is not closed")
        if self.curation_classification not in {
            "provider_failure",
            "true_abstention",
            "accepted_revision",
            "rejected_nonempty",
        }:
            raise ValueError("curation_classification is not closed")
        if type(self.reflection_call_id) is not str or not self.reflection_call_id:
            raise ValueError("reflection_call_id must be non-empty")
        if self.reflection_call_status not in {"completed", "failed"}:
            raise ValueError("reflection_call_status is not closed")
        if self.reflection_call_status == "completed":
            if self.reflection_failure_type is not None:
                raise ValueError("completed reflection cannot carry a failure")
        elif type(self.reflection_failure_type) is not str or not self.reflection_failure_type:
            raise ValueError("failed reflection requires its typed failure")
        for name in (
            "mechanism_decision",
            "cache_evidence",
            "telemetry",
            "provider_attempts",
            "structured_evidence",
            "raw_evaluator",
        ):
            value = getattr(self, name)
            if type(value) is not FrozenJsonObject:
                raise TypeError(f"{name} must be an exact FrozenJsonObject")
            value.__post_init__()
        if (
            type(self.physical_receipt_artifact_ids) is not tuple
            or len(self.physical_receipt_artifact_ids) != 11
            or len(set(self.physical_receipt_artifact_ids)) != 11
        ):
            raise ValueError("analysis requires exactly 11 physical receipt artifacts")
        if type(self.replication_recommended) is not bool:
            raise TypeError("replication_recommended must be exact bool")
        for name in (
            "run_started_at_utc",
            "run_finished_at_utc",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "end_to_end_wall_seconds",
            "unique_evaluator_wall_seconds_sum",
            "unique_evaluator_wall_seconds_max",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        object.__setattr__(
            self,
            "analysis_sha256",
            hashlib.sha256(_ANALYSIS_DOMAIN + _canonical_bytes(self._record())).hexdigest(),
        )

    def _record(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "freeze_receipt_sha256": self.freeze_receipt_sha256,
            "runtime_inputs_sha256": self.runtime_inputs_sha256,
            "optimizer_result_sha256": self.optimizer_result_sha256,
            "result_validation_receipt_sha256": (
                self.result_validation_receipt_sha256
            ),
            "terminal_validation_receipt_sha256": (
                self.terminal_validation_receipt_sha256
            ),
            "reflection_call_id": self.reflection_call_id,
            "reflection_call_receipt_sha256": self.reflection_call_receipt_sha256,
            "reflection_call_status": self.reflection_call_status,
            "reflection_failure_type": self.reflection_failure_type,
            "curation_status": self.curation_status,
            "curation_classification": self.curation_classification,
            "mechanism_decision": thaw_json(self.mechanism_decision),
            "cache_evidence": thaw_json(self.cache_evidence),
            "telemetry": thaw_json(self.telemetry),
            "provider_attempts": thaw_json(self.provider_attempts),
            "structured_evidence": thaw_json(self.structured_evidence),
            "raw_evaluator": thaw_json(self.raw_evaluator),
            "physical_receipt_artifact_ids": list(
                self.physical_receipt_artifact_ids
            ),
            "timing": {
                "run_started_at_utc": self.run_started_at_utc,
                "run_finished_at_utc": self.run_finished_at_utc,
                "end_to_end_wall_seconds": self.end_to_end_wall_seconds.hex(),
                "unique_evaluator_wall_seconds_sum": (
                    self.unique_evaluator_wall_seconds_sum.hex()
                ),
                "unique_evaluator_wall_seconds_max": (
                    self.unique_evaluator_wall_seconds_max.hex()
                ),
            },
            "claim_boundary": {
                "preregistered_g3_screen_result": True,
                "paper_ready_sota_claim": False,
                "genericity_claim": False,
                "replication_required": self.replication_recommended,
            },
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._record(), "analysis_sha256": self.analysis_sha256}


async def analyze_airfoil_g3_live_result(
    *,
    composition: AgenticOptimizerComposition,
    inputs: AirfoilG3RuntimeInputs,
    result: OptimizerResult,
    runtime_manifest_sha256: str,
    provider_outcomes: Sequence[Mapping[str, object]],
    provider_requests: Sequence[Mapping[str, object]],
    provider_outputs: Sequence[Mapping[str, object]],
    raw_evaluator_receipt_paths: Sequence[Path],
    run_started_at_utc: str,
    run_finished_at_utc: str,
    end_to_end_wall_seconds: float,
    provider_profile: AirfoilG3ProviderProfile = DEEPSEEK_G3_PROVIDER_PROFILE,
) -> AirfoilG3AnalysisReceipt:
    """Authenticate endpoints, route telemetry, cache, and 11 raw receipts."""

    inputs.__post_init__()
    provider_profile.__post_init__()
    if inputs.freeze_receipt_sha256 is None:
        raise AirfoilG3AnalysisError("live analysis requires a freeze receipt")
    if composition.benchmark is not inputs.benchmark:
        raise AirfoilG3AnalysisError("analysis received a foreign composition")
    planner = composition.planner
    if type(planner) is not G3CausalScreenPlanner:
        raise AirfoilG3AnalysisError("analysis requires the exact G3 planner")
    curation = composition.feedback_interceptor
    if type(curation) is not G3PostsealCurationInterceptor:
        raise AirfoilG3AnalysisError("analysis requires the generic G3 curation policy")
    if curation.curation_authority is None or curation.curation_receipt is None:
        raise AirfoilG3AnalysisError("G3 curation lacks typed authority/receipt evidence")
    cache = await composition.engine.evaluation_cache_snapshot()
    result_validation = validate_g3_causal_screen_result(
        result,
        planner=planner,
        evaluation_cache_snapshot=cache,
        curation_spec=curation.spec,
        curation_authority=curation.curation_authority,
        curation_receipt=curation.curation_receipt,
    )
    pre_curation = replace(
        result.final_state,
        logical_llm_calls=5,
        feedback_receipts=result.final_state.feedback_receipts[:2],
    )
    terminal = validate_g3_terminal_state(
        state=pre_curation,
        planner=planner,
        evaluation_cache_snapshot=cache,
    )
    proposal_telemetry = tuple(
        candidate.call_telemetry
        for candidate in result.final_state.candidates
        if candidate.call_telemetry is not None
    )
    if len(proposal_telemetry) != 5:
        raise AirfoilG3AnalysisError("G3 result lacks exactly five proposal telemetry rows")
    reflection_receipts = composition.engine.reflection_call_receipts
    if len(reflection_receipts) != 1:
        raise AirfoilG3AnalysisError("G3 result lacks its one curation call receipt")
    reflection = reflection_receipts[0]
    reflection.__post_init__()
    if reflection != curation.curation_receipt.call_receipt:
        raise AirfoilG3AnalysisError(
            "engine reflection receipt differs from curation evidence"
        )
    if reflection.telemetry is not None:
        all_telemetry = (*proposal_telemetry, reflection.telemetry)
    else:
        if (
            reflection.status is ReflectionCallStatus.COMPLETED
            or result_validation.curation_status != "incomplete"
        ):
            raise AirfoilG3AnalysisError(
                "curation telemetry is absent outside an isolated provider failure"
            )
        all_telemetry = proposal_telemetry
    telemetry = _telemetry_record(
        all_telemetry,
        provider_profile=provider_profile,
    )
    if reflection.status is ReflectionCallStatus.COMPLETED and len(all_telemetry) != 6:
        raise AirfoilG3AnalysisError("completed G3 run lacks six route telemetry rows")

    expected_calls: dict[str, _ExpectedProviderCall] = {}
    for generation in result.generation_receipts:
        for slot in generation.slot_results:
            outcome = slot.outcome
            call_id = outcome.prepared.call_id
            if call_id is None:
                continue
            candidate = outcome.candidate
            telemetry_row = None if candidate is None else candidate.call_telemetry
            if telemetry_row is None or call_id.value in expected_calls:
                raise AirfoilG3AnalysisError(
                    "proposal call topology or telemetry is incomplete"
                )
            prompt = outcome.prepared.prompt
            expected_calls[call_id.value] = _ExpectedProviderCall(
                operation=outcome.prepared.plan.operator_kind.value,
                semantic_prompt_sha256=hashlib.sha256(
                    prompt.encode("utf-8", errors="strict")
                ).hexdigest(),
                renderer_id=IDENTITY_PROMPT_RENDERER_ID,
                renderer_revision=IDENTITY_PROMPT_RENDERER_REVISION,
                renderer_definition_sha256=(
                    IDENTITY_PROMPT_RENDERER_DEFINITION_SHA256
                ),
                output_tool_name=FINITE_VARIATION_SELECTION_TOOL_NAME,
                telemetry=telemetry_row,
                is_reflection=False,
            )
    proposal_call_ids = set(expected_calls)
    if len(proposal_call_ids) != 5 or reflection.call_id.value in proposal_call_ids:
        raise AirfoilG3AnalysisError("engine call IDs do not prove exact 5+1 topology")
    expected_calls[reflection.call_id.value] = _ExpectedProviderCall(
        operation=reflection.request.operation,
        semantic_prompt_sha256=reflection.request.prompt_sha256,
        renderer_id=REFLECTION_PROMPT_RENDERER_ID,
        renderer_revision=REFLECTION_PROMPT_RENDERER_REVISION,
        renderer_definition_sha256=REFLECTION_PROMPT_RENDERER_DEFINITION_SHA256,
        output_tool_name=REFLECTION_TOOL_NAME,
        telemetry=reflection.telemetry,
        is_reflection=True,
    )
    receipt_ids: set[str] = set()
    receipt_candidates: dict[str, str] = {}
    duration_by_receipt: dict[str, float] = {}
    for candidate in result.final_state.candidates:
        detailed = candidate.detailed_evaluation
        receipt = None if detailed is None else detailed.payload.receipt
        if receipt is None:
            raise AirfoilG3AnalysisError("one G3 occurrence lacks a physical receipt")
        receipt_ids.add(receipt.artifact_id.value)
        configuration = thaw_json(candidate.configuration)
        expected_candidate_sha256 = candidate_sha256(configuration)
        previous_candidate = receipt_candidates.setdefault(
            receipt.artifact_id.value,
            expected_candidate_sha256,
        )
        if previous_candidate != expected_candidate_sha256:
            raise AirfoilG3AnalysisError(
                "one physical receipt is attached to different candidates"
            )
        assert detailed is not None
        duration_by_receipt.setdefault(
            receipt.artifact_id.value,
            float(detailed.timings.total_wall_seconds),
        )
    if len(receipt_ids) != 11:
        raise AirfoilG3AnalysisError(
            "physical receipt cardinality is not exact 11 MISS / one HIT"
        )
    request_evidence, requests_by_call = _request_evidence_record(
        provider_requests,
        expected_calls=expected_calls,
        provider_profile=provider_profile,
    )
    provider_attempts, provider_rows_by_call = _provider_attempt_record(
        provider_outcomes,
        curation_status=result_validation.curation_status,
        reflection_call_id=reflection.call_id.value,
        expected_calls=expected_calls,
        requests_by_call=requests_by_call,
    )
    output_evidence, curation_classification = _output_evidence_record(
        provider_outputs,
        expected_calls=expected_calls,
        requests_by_call=requests_by_call,
        provider_rows_by_call=provider_rows_by_call,
        reflection_call_id=reflection.call_id.value,
        reflection_status=reflection.status,
        reflection_failure_type=reflection.failure_type,
        reflection_publications=len(reflection.publications),
        curation_status=result_validation.curation_status,
    )
    if set(telemetry["provider_response_ids"]) != set(
        provider_attempts["provider_response_ids_by_call"].values()
    ):
        raise AirfoilG3AnalysisError(
            "engine telemetry and durable provider outcomes disagree"
        )
    raw_evaluator = _raw_evaluator_record(
        raw_evaluator_receipt_paths,
        expected_artifact_candidates=receipt_candidates,
    )
    decision = terminal.mechanism_decision.to_record()
    frozen_values = []
    for value in (
        decision,
        dict(terminal.cache_evidence),
        telemetry,
        provider_attempts,
        {
            "request": request_evidence,
            "output": output_evidence,
        },
        raw_evaluator,
    ):
        frozen = freeze_json(value)
        if type(frozen) is not FrozenJsonObject:
            raise TypeError("analysis record did not freeze as an object")
        frozen_values.append(frozen)
    return AirfoilG3AnalysisReceipt(
        runtime_manifest_sha256=runtime_manifest_sha256,
        freeze_receipt_sha256=inputs.freeze_receipt_sha256,
        runtime_inputs_sha256=inputs.runtime_inputs_sha256,
        optimizer_result_sha256=result.result_hash,
        result_validation_receipt_sha256=result_validation.receipt_sha256,
        terminal_validation_receipt_sha256=terminal.receipt_sha256,
        reflection_call_id=reflection.call_id.value,
        reflection_call_receipt_sha256=reflection.receipt_sha256,
        reflection_call_status=reflection.status.value,
        reflection_failure_type=reflection.failure_type,
        curation_status=result_validation.curation_status,
        curation_classification=curation_classification,
        mechanism_decision=frozen_values[0],
        cache_evidence=frozen_values[1],
        telemetry=frozen_values[2],
        provider_attempts=frozen_values[3],
        structured_evidence=frozen_values[4],
        raw_evaluator=frozen_values[5],
        physical_receipt_artifact_ids=tuple(sorted(receipt_ids)),
        run_started_at_utc=run_started_at_utc,
        run_finished_at_utc=run_finished_at_utc,
        end_to_end_wall_seconds=float(end_to_end_wall_seconds),
        unique_evaluator_wall_seconds_sum=float(sum(duration_by_receipt.values())),
        unique_evaluator_wall_seconds_max=float(max(duration_by_receipt.values())),
        replication_recommended=bool(decision["advance_to_replication"]),
    )


__all__ = [
    "AirfoilG3AnalysisError",
    "AirfoilG3AnalysisReceipt",
    "analyze_airfoil_g3_live_result",
]
