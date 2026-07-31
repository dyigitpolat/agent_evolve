#!/usr/bin/env python3
"""Run a capped, fully traced AgentEvolve development probe.

This runner exercises the real OpenRouter/Pydantic-AI path on cheap synthetic
problems whose useful branch recombination is known in advance.  Its outputs
debug the agentic workflow; they are explicitly not benchmark or wall-clock
evidence for the paper.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timezone
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.settings import load_credentials

from agent_evolve.application.agentic_evolution import (
    AgenticEvolutionEngine,
    EvolutionCandidate,
    InvocationOutcome,
    InvocationPlan,
    OperatorKind,
)
from agent_evolve.application.insight_memory import (
    InsightMemoryBank,
    context_stratum_hash,
)
from agent_evolve.domain.llm_task_queue import LLMTaskOutcome
from agent_evolve.infrastructure.ids import DeterministicIdFactory
from agent_evolve.integrations.pydantic_ai.agentic_generator import (
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (
    create_production_queued_runner,
)
from agent_evolve.ports.agentic_generator import InsightDraft

from examples.development.dag_dispatch_codesign import problem_def as dag_problem
from examples.development.pipeline_codesign import problem_def as pipeline_problem
from examples.development.corpus_paths import resolve_corpus_path  # noqa: E402


MODEL = "deepseek/deepseek-v4-pro"
PROVIDER_ORDER = ("together", "parasail", "wandb")
DEFAULT_LOG_ROOT = (
    WORKSPACE_ROOT
    / "papers"
    / "agent_evolve_aaai_2027"
    / "research_artifacts"
    / "experiment_logs"
    / "agentic_development"
)


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "value") and type(getattr(value, "value")) is str:
        return getattr(value, "value")
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class JsonlWriter:
    """Small synchronous trace interceptor with one-line atomic records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = path.open("x", encoding="utf-8")

    def write(self, value: Mapping[str, object]) -> None:
        self._stream.write(_canonical_json(dict(value)) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        self._stream.close()


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sha256(path: Path) -> str:
    # Resolves across the 2026-07-28 archive split. Safe because the digest is
    # of the bytes: an archived copy either hashes to the sealed value or is
    # caught by whatever compares against it.
    return hashlib.sha256(resolve_corpus_path(path).read_bytes()).hexdigest()


def _source_snapshot(paths: Sequence[Path]) -> dict[str, object]:
    files = tuple(
        sorted(
            {
                path.resolve()
                for root in paths
                for path in root.rglob("*.py")
                if path.is_file()
            },
            key=lambda path: path.relative_to(AGENT_EVOLVE_ROOT).as_posix(),
        )
    )
    framed = hashlib.sha256(b"agent-evolve:development-source-snapshot:v1\x00")
    per_file: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(AGENT_EVOLVE_ROOT).as_posix()
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        per_file[relative] = digest
        name = relative.encode("utf-8", errors="strict")
        framed.update(len(name).to_bytes(8, "big"))
        framed.update(name)
        framed.update(len(content).to_bytes(8, "big"))
        framed.update(content)
    return {
        "framing": "agent-evolve:development-source-snapshot:v1",
        "file_count": len(files),
        "sha256": framed.hexdigest(),
        "files": per_file,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _queue_outcome_record(outcome: LLMTaskOutcome[Any]) -> dict[str, object]:
    attempts = []
    for attempt in outcome.telemetry.attempts:
        classification = attempt.classification
        attempts.append(
            {
                "attempt_number": attempt.attempt_number,
                "status": attempt.status.value,
                "wait_time_ns": attempt.wait_time_ns,
                "service_time_ns": attempt.service_time_ns,
                "will_retry": attempt.will_retry,
                "policy_backoff_ns": attempt.policy_backoff_ns,
                "retry_after_ns": attempt.retry_after_ns,
                "scheduled_delay_ns": attempt.scheduled_delay_ns,
                "error_type": attempt.error_type,
                "classification": (
                    None
                    if classification is None
                    else {
                        "disposition": classification.disposition.value,
                        "reason": classification.reason.value,
                    }
                ),
            }
        )
    return {
        "task_id": outcome.telemetry.task_id,
        "status": outcome.status.value,
        "cancellation_reason": (
            None
            if outcome.cancellation_reason is None
            else outcome.cancellation_reason.value
        ),
        "queue_time_ns": outcome.telemetry.queue_time_ns,
        "service_time_ns": outcome.telemetry.service_time_ns,
        "total_time_ns": outcome.telemetry.total_time_ns,
        "attempts": attempts,
    }


def _value_counts(values: Sequence[str]) -> dict[str, int]:
    return {
        value: sum(item == value for item in values)
        for value in sorted(set(values))
    }


def _call_summary(
    events: Sequence[Mapping[str, object]],
    *,
    expected_logical_calls: int,
) -> dict[str, object]:
    completed = [
        event
        for event in events
        if event.get("event_type")
        in {"llm_call_completed", "reflection_completed"}
    ]
    failed = [
        event
        for event in events
        if event.get("event_type") in {"llm_call_failed", "reflection_failed"}
    ]
    observed = len(completed) + len(failed)
    if observed != expected_logical_calls:
        raise RuntimeError(
            "trace call accounting mismatch: "
            f"expected {expected_logical_calls}, observed {observed}"
        )
    known_costs = [
        Decimal(str(event["cost_usd"]))
        for event in completed
        if event.get("cost_usd") is not None
    ]
    return {
        "expected_logical_calls": expected_logical_calls,
        "successful_logical_calls": len(completed),
        "failed_logical_calls": len(failed),
        "successful_attempts_reported": sum(
            int(event["attempt_count"]) for event in completed
        ),
        "input_tokens_successful_responses": sum(
            int(event["input_tokens"]) for event in completed
        ),
        "output_tokens_successful_responses": sum(
            int(event["output_tokens"]) for event in completed
        ),
        "reasoning_tokens_successful_responses": sum(
            int(event["reasoning_tokens"]) for event in completed
        ),
        "cost_usd_successful_responses": str(sum(known_costs, Decimal(0))),
        "responses_without_reported_cost": len(completed) - len(known_costs),
        "cost_scope": (
            "Successful responses only; failed attempts can be billable and are "
            "not included unless the provider returned usage telemetry."
        ),
        "requested_models": _value_counts(
            [str(event["requested_model"]) for event in completed]
        ),
        "resolved_models": _value_counts(
            [str(event["resolved_model"]) for event in completed]
        ),
        "resolved_providers": _value_counts(
            [str(event["resolved_provider"]) for event in completed]
        ),
        "failure_types": _value_counts(
            [str(event["failure_type"]) for event in failed]
        ),
    }


def _queue_log_summary(path: Path) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    attempts = [attempt for record in records for attempt in record["attempts"]]
    retry_reasons = [
        attempt["classification"]["reason"]
        for attempt in attempts
        if attempt["classification"] is not None and attempt["will_retry"]
    ]
    return {
        "terminal_outcomes": len(records),
        "attempts": len(attempts),
        "retried_attempts": sum(bool(attempt["will_retry"]) for attempt in attempts),
        "terminal_statuses": _value_counts(
            [str(record["status"]) for record in records]
        ),
        "retry_reasons": _value_counts([str(value) for value in retry_reasons]),
        "queue_time_ns": sum(int(record["queue_time_ns"]) for record in records),
        "service_time_ns": sum(
            int(record["service_time_ns"]) for record in records
        ),
    }


def _aggregate_call_summaries(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    additive = (
        "expected_logical_calls",
        "successful_logical_calls",
        "failed_logical_calls",
        "successful_attempts_reported",
        "input_tokens_successful_responses",
        "output_tokens_successful_responses",
        "reasoning_tokens_successful_responses",
        "responses_without_reported_cost",
    )

    def merged_counts(field: str) -> dict[str, int]:
        keys = sorted(
            {
                key
                for summary in summaries
                for key in dict(summary[field])
            }
        )
        return {
            key: sum(int(dict(summary[field]).get(key, 0)) for summary in summaries)
            for key in keys
        }

    return {
        **{
            field: sum(int(summary[field]) for summary in summaries)
            for field in additive
        },
        "cost_usd_successful_responses": str(
            sum(
                (
                    Decimal(str(summary["cost_usd_successful_responses"]))
                    for summary in summaries
                ),
                Decimal(0),
            )
        ),
        "cost_scope": (
            "Successful responses only; failed attempts can be billable and are "
            "not included unless the provider returned usage telemetry."
        ),
        "requested_models": merged_counts("requested_models"),
        "resolved_models": merged_counts("resolved_models"),
        "resolved_providers": merged_counts("resolved_providers"),
        "failure_types": merged_counts("failure_types"),
    }


@dataclass(frozen=True, slots=True)
class DomainDefinition:
    name: str
    problem_factory: Callable[[], Any]
    base: dict[str, Any]
    left: dict[str, Any]
    right: dict[str, Any]
    known_recombination_target: dict[str, Any]
    mutation_scope: tuple[str, ...]
    initial_insights: tuple[InsightDraft, ...]


PIPELINE = DomainDefinition(
    name="pipeline_codesign",
    problem_factory=pipeline_problem.PipelineCoDesignProblem,
    base=pipeline_problem.BASE_CONFIG,
    left=pipeline_problem.DEVELOPMENT_BRANCH_LEFT,
    right=pipeline_problem.DEVELOPMENT_BRANCH_RIGHT,
    known_recombination_target=pipeline_problem.DEVELOPMENT_RECOMBINATION_TARGET,
    mutation_scope=("runtime",),
    initial_insights=(
        InsightDraft(
            claim="Compose disjoint compiler and runtime branch patches instead of copying one parent.",
            trigger="one branch changes compiler components and the other changes only runtime",
            mechanism="typed three-way replay preserves both innovations without overwriting either patch",
            affected_paths=("$.passes", "$.frontend", "$.backend", "$.runtime"),
            evidence_summary="mechanistic development prior; not yet supported by a live generation",
            confidence=0.65,
        ),
        InsightDraft(
            claim="Moderate prefetch distance is preferable to maximal lookahead for threaded execution.",
            trigger="runtime threads exceed one",
            mechanism="moderate lookahead can hide latency without excess traffic",
            affected_paths=("$.runtime.threads", "$.runtime.prefetch_distance"),
            evidence_summary="synthetic evaluator structure disclosed for workflow debugging",
            confidence=0.55,
        ),
        InsightDraft(
            claim="Vectorized passes benefit from a structure-of-arrays data layout.",
            trigger="the pass sequence includes vectorize",
            mechanism="contiguous vector lanes avoid gather overhead",
            affected_paths=("$.passes", "$.runtime.data_layout"),
            evidence_summary="synthetic evaluator structure disclosed for workflow debugging",
            confidence=0.6,
        ),
        InsightDraft(
            claim="Always maximize threads and prefetch distance.",
            trigger="runtime tuning is allowed",
            mechanism="more parallelism and lookahead should always improve speed",
            affected_paths=("$.runtime.threads", "$.runtime.prefetch_distance"),
            evidence_summary="deliberately questionable hypothesis included as a negative memory control",
            confidence=0.35,
        ),
    ),
)


DAG = DomainDefinition(
    name="dag_dispatch_codesign",
    problem_factory=dag_problem.DagDispatchCoDesignProblem,
    base=dag_problem.BASE_CONFIG,
    left=dag_problem.DEVELOPMENT_BRANCH_LEFT,
    right=dag_problem.DEVELOPMENT_BRANCH_RIGHT,
    known_recombination_target=dag_problem.DEVELOPMENT_RECOMBINATION_TARGET,
    mutation_scope=("dispatch_order",),
    initial_insights=(
        InsightDraft(
            claim="Combine accelerator placement with a topological order that makes fusible tasks adjacent.",
            trigger="placement co-locates encode and compress while order is mutable",
            mechanism="adjacency activates fusion without discarding placement gains",
            affected_paths=("$.assignments", "$.dispatch_order"),
            evidence_summary="mechanistic development prior; not yet supported by a live generation",
            confidence=0.7,
        ),
        InsightDraft(
            claim="Preserve canonical task ordering inside the assignments array.",
            trigger="changing worker placement",
            mechanism="the strict candidate contract identifies assignment entries by canonical task position",
            affected_paths=("$.assignments",),
            evidence_summary="known feasibility contract for the development evaluator",
            confidence=0.65,
        ),
        InsightDraft(
            claim="Accelerators should be reserved for the longest compatible tasks.",
            trigger="GPU or NPU placement capacity is available",
            mechanism="critical-path service reductions can outweigh transfer overhead",
            affected_paths=("$.assignments",),
            evidence_summary="testable scheduling hypothesis",
            confidence=0.55,
        ),
        InsightDraft(
            claim="Put every task on a different worker whenever possible.",
            trigger="multiple workers are available",
            mechanism="maximum distribution should always minimize the critical path",
            affected_paths=("$.assignments",),
            evidence_summary="deliberately questionable hypothesis included as a negative memory control",
            confidence=0.3,
        ),
    ),
)


def _candidate_record(candidate: EvolutionCandidate | None) -> dict[str, object] | None:
    if candidate is None:
        return None
    telemetry = candidate.call_telemetry
    return {
        "candidate_id": candidate.candidate_id.value,
        "label": candidate.label,
        "configuration": candidate.configuration_dict,
        "objectives": candidate.objective_map,
        "valid": candidate.valid,
        "operator_compliant": candidate.operator_compliant,
        "operator_failure": candidate.operator_failure,
        "evidence_compliant": candidate.evidence_compliant,
        "evidence_failure": candidate.evidence_failure,
        "preservation_verified": candidate.preservation_verified,
        "selected_insight_ids": list(candidate.selected_insight_ids),
        "claimed_insight_ids": list(candidate.claimed_insight_ids),
        "source_attribution": [
            {"path": item.path, "source": item.source}
            for item in candidate.source_attribution
        ],
        "conflict_resolutions": [
            {
                "relation_id": item.relation_id,
                "choice": item.choice,
                "explanation": item.explanation,
            }
            for item in candidate.conflict_resolutions
        ],
        "design_rationale": candidate.design_rationale,
        "call": (
            None
            if telemetry is None
            else {
                "requested_model": telemetry.requested_model,
                "resolved_model": telemetry.resolved_model,
                "resolved_provider": telemetry.resolved_provider,
                "finish_reason": telemetry.finish_reason,
                "input_tokens": telemetry.input_tokens,
                "output_tokens": telemetry.output_tokens,
                "reasoning_tokens": telemetry.reasoning_tokens,
                "cost_usd": (
                    None if telemetry.cost_usd is None else str(telemetry.cost_usd)
                ),
                "latency_ns": telemetry.latency_ns,
                "attempt_count": telemetry.attempt_count,
            }
        ),
    }


def _outcome_record(
    outcome: InvocationOutcome,
    *,
    known_target: Mapping[str, object],
) -> dict[str, object]:
    candidate = outcome.candidate
    return {
        "label": outcome.prepared.plan.label,
        "operator_kind": outcome.prepared.plan.operator_kind.value,
        "operator_invocation_id": outcome.prepared.operator_invocation_id.value,
        "call_failure_type": outcome.call_failure_type,
        "reward": outcome.reward,
        "dominates_any_parent": outcome.dominates_any_parent,
        "better_than_any_parent": outcome.better_than_any_parent,
        "selected_insight_ids": [
            ref.insight_id.value
            for ref in outcome.prepared.variation_case.selected_insights
        ],
        "known_recombination_target_match": bool(
            outcome.prepared.plan.operator_kind
            is OperatorKind.THREE_WAY_RECOMBINATION
            and
            candidate is not None
            and candidate.configuration_dict == dict(known_target)
        ),
        "candidate": _candidate_record(candidate),
    }


def _best_parent(
    outcomes: Sequence[InvocationOutcome],
    fallback: EvolutionCandidate,
) -> EvolutionCandidate:
    eligible = [
        outcome
        for outcome in outcomes
        if outcome.candidate is not None
        and outcome.candidate.valid
        and outcome.candidate.operator_compliant
        and outcome.dominates_any_parent
    ]
    if not eligible:
        return fallback
    return max(eligible, key=lambda outcome: outcome.reward).candidate  # type: ignore[return-value]


def _verified_recombination_parent(
    outcomes: Sequence[InvocationOutcome],
) -> EvolutionCandidate:
    eligible = [
        outcome.candidate
        for outcome in outcomes
        if outcome.prepared.plan.operator_kind
        is OperatorKind.THREE_WAY_RECOMBINATION
        and outcome.candidate is not None
        and outcome.candidate.valid
        and outcome.candidate.operator_compliant
        and outcome.candidate.preservation_verified is True
    ]
    if len(eligible) != 1:
        raise RuntimeError(
            "development workflow requires exactly one valid, operator-compliant, "
            "preservation-verified recombination parent"
        )
    return eligible[0]


async def _run_domain(
    definition: DomainDefinition,
    *,
    generator: PydanticAIAgenticGenerator,
    seed: int,
    event_writer: JsonlWriter,
    max_output_tokens: int,
    temperature: float,
) -> dict[str, object]:
    ids = DeterministicIdFactory(f"live_{definition.name}_{seed}")
    memory = InsightMemoryBank(
        id_factory=ids,
        exploration_probability=Fraction(1, 1),
        shrinkage_effective_sample_size=4.0,
    )
    memory.extend(definition.initial_insights)
    trace_events: list[dict[str, object]] = []

    def record_event(event: Mapping[str, object]) -> None:
        record = dict(event)
        trace_events.append(record)
        event_writer.write({"domain": definition.name, **record})

    engine = AgenticEvolutionEngine(
        problem=definition.problem_factory(),
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=seed,
        evaluator_concurrency=4,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        trace_sink=record_event,
    )
    base, left, right = await asyncio.gather(
        engine.register_seed(definition.base, label="base"),
        engine.register_seed(definition.left, label="left_branch"),
        engine.register_seed(definition.right, label="right_branch"),
    )

    generation_one = await engine.run_invocations(
        (
            InvocationPlan(
                OperatorKind.TWO_PARENT_CROSSOVER,
                (left, right),
                generation=1,
                label="g1_crossover_no_memory",
            ),
            InvocationPlan(
                OperatorKind.THREE_WAY_RECOMBINATION,
                (left, right),
                generation=1,
                label="g1_recombine_no_memory",
                common_ancestor=base,
            ),
            InvocationPlan(
                OperatorKind.REPRODUCTION,
                (left,),
                generation=1,
                label="g1_reproduction_control",
            ),
        )
    )
    first_reflection = await engine.reflect(
        generation_one,
        label="after_generation_one",
        max_insights=3,
    )
    if not first_reflection:
        raise RuntimeError("generation-one reflection produced no new insight")

    # The adaptation stage deliberately begins from the verified two-branch
    # composition.  Fitness-based selection here would confound recombination
    # with crossover or a one-parent mutation.
    mutation_parent = _verified_recombination_parent(generation_one)
    generation_two = await engine.run_invocations(
        (
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (mutation_parent,),
                generation=2,
                label="g2_mutation_no_memory",
                allowed_top_level=definition.mutation_scope,
                phase="g2_same_parent_discovery",
            ),
            *tuple(
                InvocationPlan(
                    OperatorKind.TYPED_MUTATION,
                    (mutation_parent,),
                    generation=2,
                    label=f"g2_mutation_random_memory_{index}",
                    allowed_top_level=definition.mutation_scope,
                    use_memory=True,
                    memory_subset_size=2,
                    memory_exploration_probability=Fraction(1, 1),
                    phase="g2_same_parent_discovery",
                )
                for index in range(1, 7)
            ),
        )
    )
    mutation_context = context_stratum_hash(
        problem_id=engine.problem_id,
        operator_kind=OperatorKind.TYPED_MUTATION.value,
        phase="g2_same_parent_discovery",
    )
    policy_snapshot_before_generation_three = {
        "context_hash": mutation_context,
        "entry_count": len(memory.entries),
        "score_evidence": list(memory.score_evidence(mutation_context)),
    }
    exploitation_parent = _best_parent(generation_two, mutation_parent)
    generation_three = await engine.run_invocations(
        (
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (exploitation_parent,),
                generation=3,
                label="g3_mutation_no_memory_control",
                allowed_top_level=definition.mutation_scope,
                phase="g3_no_memory_control",
            ),
            *tuple(
                InvocationPlan(
                    OperatorKind.TYPED_MUTATION,
                    (exploitation_parent,),
                    generation=3,
                    label=f"g3_mutation_score_exploit_{index}",
                    allowed_top_level=definition.mutation_scope,
                    use_memory=True,
                    memory_subset_size=2,
                    memory_exploration_probability=Fraction(0, 1),
                    memory_score_phase="g2_same_parent_discovery",
                    phase="g3_score_exploitation",
                )
                for index in range(1, 3)
            ),
            *tuple(
                InvocationPlan(
                    OperatorKind.TYPED_MUTATION,
                    (exploitation_parent,),
                    generation=3,
                    label=f"g3_mutation_uniform_holdout_{index}",
                    allowed_top_level=definition.mutation_scope,
                    use_memory=True,
                    memory_subset_size=2,
                    memory_exploration_probability=Fraction(1, 1),
                    memory_score_phase="g2_same_parent_discovery",
                    phase="g3_uniform_holdout",
                )
                for index in range(1, 3)
            ),
        )
    )
    final_reflection = await engine.reflect(
        (*generation_two, *generation_three),
        label="after_discovery_and_exploitation",
        max_insights=3,
    )
    if not final_reflection:
        raise RuntimeError("final reflection produced no new insight")

    all_outcomes = (*generation_one, *generation_two, *generation_three)
    expected_logical_calls = (
        sum(
            outcome.prepared.call_id is not None
            for outcome in all_outcomes
        )
        + 2
    )
    return {
        "domain": definition.name,
        "development_only": True,
        "seed_candidates": {
            "base": _candidate_record(base),
            "left": _candidate_record(left),
            "right": _candidate_record(right),
        },
        "mutation_parent_id": mutation_parent.candidate_id.value,
        "exploitation_parent_id": exploitation_parent.candidate_id.value,
        "generation_one": [
            _outcome_record(
                outcome,
                known_target=definition.known_recombination_target,
            )
            for outcome in generation_one
        ],
        "generation_two": [
            _outcome_record(
                outcome,
                known_target=definition.known_recombination_target,
            )
            for outcome in generation_two
        ],
        "generation_three": [
            _outcome_record(
                outcome,
                known_target=definition.known_recombination_target,
            )
            for outcome in generation_three
        ],
        "memory": {
            "entry_count": len(memory.entries),
            "trial_count": len(memory.trials),
            "policy_snapshot_before_generation_three": (
                policy_snapshot_before_generation_three
            ),
            "final_mutation_score_evidence": list(
                memory.score_evidence(mutation_context)
            ),
        },
        "provider_calls": _call_summary(
            trace_events,
            expected_logical_calls=expected_logical_calls,
        ),
        "counts": {
            "logical_variation_invocations": len(all_outcomes),
            "candidate_outputs": sum(
                outcome.candidate is not None for outcome in all_outcomes
            ),
            "valid_candidates": sum(
                outcome.candidate is not None and outcome.candidate.valid
                for outcome in all_outcomes
            ),
            "operator_compliant_candidates": sum(
                outcome.candidate is not None
                and outcome.candidate.operator_compliant
                for outcome in all_outcomes
            ),
            "evidence_compliant_candidates": sum(
                outcome.candidate is not None
                and outcome.candidate.evidence_compliant
                for outcome in all_outcomes
            ),
            "positive_reward_candidates": sum(
                outcome.reward > 0 for outcome in all_outcomes
            ),
            "verified_recombination_target_matches": sum(
                outcome.prepared.plan.operator_kind
                is OperatorKind.THREE_WAY_RECOMBINATION
                and outcome.candidate is not None
                and outcome.candidate.configuration_dict
                == definition.known_recombination_target
                for outcome in all_outcomes
            ),
        },
    }


def _manifest(args: argparse.Namespace, run_id: str) -> dict[str, object]:
    sources = {
        "engine": AGENT_EVOLVE_ROOT
        / "src"
        / "agent_evolve"
        / "application"
        / "agentic_evolution.py",
        "memory": AGENT_EVOLVE_ROOT
        / "src"
        / "agent_evolve"
        / "application"
        / "insight_memory.py",
        "queue_domain": AGENT_EVOLVE_ROOT
        / "src"
        / "agent_evolve"
        / "domain"
        / "llm_task_queue.py",
        "queue": AGENT_EVOLVE_ROOT
        / "src"
        / "agent_evolve"
        / "application"
        / "llm_task_queue.py",
        "backoff_policy": AGENT_EVOLVE_ROOT
        / "src"
        / "agent_evolve"
        / "policies"
        / "llm_backoff.py",
        "memory_selector": AGENT_EVOLVE_ROOT
        / "src"
        / "agent_evolve"
        / "policies"
        / "memory"
        / "randomized_subset.py",
        "typed_patch_policy": AGENT_EVOLVE_ROOT
        / "src"
        / "agent_evolve"
        / "policies"
        / "variation"
        / "typed_patch.py",
        "queued_runner": AGENT_EVOLVE_ROOT
        / "src"
        / "agent_evolve"
        / "integrations"
        / "pydantic_ai"
        / "queued_runner.py",
        "provider_adapter": AGENT_EVOLVE_ROOT
        / "src"
        / "agent_evolve"
        / "integrations"
        / "pydantic_ai"
        / "async_generator.py",
        "agentic_adapter": AGENT_EVOLVE_ROOT
        / "src"
        / "agent_evolve"
        / "integrations"
        / "pydantic_ai"
        / "agentic_generator.py",
        "pipeline_problem": AGENT_EVOLVE_ROOT
        / "examples"
        / "development"
        / "pipeline_codesign"
        / "problem_def.py",
        "dag_problem": AGENT_EVOLVE_ROOT
        / "examples"
        / "development"
        / "dag_dispatch_codesign"
        / "problem_def.py",
        "probe": Path(__file__).resolve(),
    }
    return {
        "schema_version": 1,
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "claim_boundary": (
            "Synthetic workflow-debugging evidence only; not a benchmark, "
            "SOTA result, or wall-clock-dominance result."
        ),
        "model": args.model,
        "provider": "openrouter",
        "provider_options": {
            "order": list(PROVIDER_ORDER),
            "allow_fallbacks": False,
            "require_parameters": False,
        },
        "domain_selection": args.domain,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "queue": {
            "max_in_flight": args.max_in_flight,
            "max_pending": args.max_pending,
            "max_attempts": args.max_attempts,
            "attempt_timeout_ns": args.attempt_timeout_seconds * 1_000_000_000,
            "base_backoff_ns": args.base_backoff_seconds * 1_000_000_000,
            "max_backoff_ns": args.max_backoff_seconds * 1_000_000_000,
            "retry_owner": "AsyncLLMTaskQueue",
            "sdk_retries": 0,
            "pydantic_ai_retries": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "packages": {
                name: _package_version(name)
                for name in ("pydantic", "pydantic-ai", "openai", "httpx")
            },
            "credential_variable": "OPENROUTER_API_KEY",
        },
        "source_sha256": {
            name: _sha256(path) for name, path in sources.items()
        },
        "python_source_snapshot": _source_snapshot(
            (
                AGENT_EVOLVE_ROOT / "src",
                AGENT_EVOLVE_ROOT / "examples" / "development",
            )
        ),
    }


async def _run(args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")

    event_writer = JsonlWriter(run_dir / "events.jsonl")
    queue_writer = JsonlWriter(run_dir / "queue_outcomes.jsonl")
    started_ns = time.perf_counter_ns()
    structured = PydanticAIStructuredGenerator.openrouter(
        api_key=api_key,
        model_name=args.model,
        max_connections=args.max_in_flight,
        timeout_seconds=float(args.attempt_timeout_seconds),
        provider_options={
            "order": list(PROVIDER_ORDER),
            "allow_fallbacks": False,
        },
        app_title="AgentEvolve AAAI 2027 development probe",
    )
    runner = create_production_queued_runner(
        generator=structured,
        max_in_flight=args.max_in_flight,
        max_pending=args.max_pending,
        max_attempts=args.max_attempts,
        attempt_timeout_ns=args.attempt_timeout_seconds * 1_000_000_000,
        base_backoff_ns=args.base_backoff_seconds * 1_000_000_000,
        max_backoff_ns=args.max_backoff_seconds * 1_000_000_000,
        close_generator=True,
        outcome_sink=lambda outcome: queue_writer.write(
            _queue_outcome_record(outcome)
        ),
    )
    generator = PydanticAIAgenticGenerator(runner)
    definitions = {
        "pipeline": (PIPELINE,),
        "dag": (DAG,),
        "both": (PIPELINE, DAG),
    }[args.domain]
    results: list[dict[str, object]] = []
    try:
        async with runner:
            for offset, definition in enumerate(definitions):
                results.append(
                    await _run_domain(
                        definition,
                        generator=generator,
                        seed=args.seed + offset,
                        event_writer=event_writer,
                        max_output_tokens=args.max_output_tokens,
                        temperature=args.temperature,
                    )
                )
    finally:
        event_writer.close()
        queue_writer.close()
    provider_calls = _aggregate_call_summaries(
        [dict(result["provider_calls"]) for result in results]
    )
    queue = _queue_log_summary(run_dir / "queue_outcomes.jsonl")
    if queue["terminal_outcomes"] != provider_calls["expected_logical_calls"]:
        raise RuntimeError(
            "queue outcome accounting mismatch: "
            f"expected {provider_calls['expected_logical_calls']}, "
            f"observed {queue['terminal_outcomes']}"
        )
    return {
        "schema_version": 1,
        "development_only": True,
        "elapsed_ns": time.perf_counter_ns() - started_ns,
        "provider_calls": provider_calls,
        "queue": queue,
        "domains": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("pipeline", "dag", "both"), default="pipeline")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--max-in-flight", type=int, default=4)
    parser.add_argument("--max-pending", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--attempt-timeout-seconds", type=int, default=90)
    parser.add_argument("--base-backoff-seconds", type=int, default=1)
    parser.add_argument("--max-backoff-seconds", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=1_600)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.model != MODEL:
        raise SystemExit(f"development probe is frozen to {MODEL}")
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "probe_%Y%m%dT%H%M%SZ"
    )
    run_dir = args.log_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    load_credentials(WORKSPACE_ROOT / ".env", override=False, optional=True)
    _write_json(run_dir / "manifest.json", _manifest(args, run_id))
    try:
        summary = asyncio.run(_run(args, run_dir))
    except BaseException as exc:
        _write_json(
            run_dir / "failure.json",
            {
                "failure_type": type(exc).__name__,
                "safe_message": (
                    str(exc)
                    if type(exc).__module__.startswith("agent_evolve")
                    else "development probe failed; inspect sanitized trace evidence"
                ),
            },
        )
        raise
    _write_json(run_dir / "summary.json", summary)
    print(_canonical_json({"run_dir": str(run_dir), "status": "succeeded"}))


if __name__ == "__main__":
    main()
