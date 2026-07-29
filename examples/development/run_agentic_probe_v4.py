#!/usr/bin/env python3
"""Run the v4 atomic-memory AgentEvolve development kill test.

This is deliberately a synthetic workflow probe, not benchmark evidence.  It
tests whether path-bounded agentic mutations, neutral utility priors, structural
memory eligibility, and downstream score exploitation can recover the exact
60-point runtime-scoped oracle after the failure observed in pipeline v3b.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    MutationContract,
    OperatorKind,
)
from agent_evolve.application.insight_memory import (
    InsightMemoryBank,
    context_stratum_hash,
)
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import JsonPath, ObjectKey
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

from examples.development import run_agentic_probe as support
from examples.development.pipeline_codesign import problem_def as pipeline_problem


MODEL = support.MODEL
DEFAULT_LOG_ROOT = support.DEFAULT_LOG_ROOT
MUTATION_OPERATOR = (OperatorKind.TYPED_MUTATION.value,)


def _path(*parts: str) -> JsonPath:
    return JsonPath(tuple(ObjectKey(part) for part in parts))


PREFETCH_PATH = _path("runtime", "prefetch_distance")
LAYOUT_PATH = _path("runtime", "data_layout")
THREADS_PATH = _path("runtime", "threads")


@dataclass(frozen=True, slots=True)
class ActionHypothesis:
    name: str
    draft: InsightDraft
    path: JsonPath
    recommended_value: object


HYPOTHESES = (
    ActionHypothesis(
        "prefetch_moderate",
        InsightDraft(
            claim="For this local intervention, set runtime.prefetch_distance to 4 when threads exceed one.",
            trigger="runtime.threads is greater than one and prefetch distance is editable",
            mechanism="a distance of four can hide latency without the traffic of longer lookahead",
            affected_paths=("$.runtime.prefetch_distance",),
            evidence_summary="synthetic development hypothesis; downstream utility starts unscored",
            confidence=0.55,
        ),
        PREFETCH_PATH,
        4,
    ),
    ActionHypothesis(
        "prefetch_maximum",
        InsightDraft(
            claim="For this local intervention, set runtime.prefetch_distance to the maximum allowed value 16.",
            trigger="prefetch distance is editable",
            mechanism="more lookahead should always hide more latency",
            affected_paths=("$.runtime.prefetch_distance",),
            evidence_summary="deliberately questionable development hypothesis",
            confidence=0.35,
        ),
        PREFETCH_PATH,
        16,
    ),
    ActionHypothesis(
        "layout_soa",
        InsightDraft(
            claim="For this local intervention, set runtime.data_layout to soa when vectorize is present.",
            trigger="the pass sequence contains vectorize and data layout is editable",
            mechanism="contiguous vector lanes avoid gather overhead",
            affected_paths=("$.runtime.data_layout",),
            evidence_summary="synthetic development hypothesis; downstream utility starts unscored",
            confidence=0.60,
        ),
        LAYOUT_PATH,
        "soa",
    ),
    ActionHypothesis(
        "layout_blocked",
        InsightDraft(
            claim="For this local intervention, set runtime.data_layout to blocked when at least four threads execute.",
            trigger="runtime.threads is at least four and data layout is editable",
            mechanism="blocking can improve cache locality for threaded execution",
            affected_paths=("$.runtime.data_layout",),
            evidence_summary="competing synthetic development hypothesis",
            confidence=0.50,
        ),
        LAYOUT_PATH,
        "blocked",
    ),
    ActionHypothesis(
        "threads_maximum",
        InsightDraft(
            claim="For this local intervention, set runtime.threads to the maximum allowed value 8.",
            trigger="thread count is editable and the register-pressure constraint remains feasible",
            mechanism="additional workers can expose more parallelism",
            affected_paths=("$.runtime.threads",),
            evidence_summary="synthetic development hypothesis; downstream utility starts unscored",
            confidence=0.50,
        ),
        THREADS_PATH,
        8,
    ),
    ActionHypothesis(
        "threads_moderate",
        InsightDraft(
            claim="For this local intervention, set runtime.threads to the moderate value 4.",
            trigger="thread count is editable",
            mechanism="moderate parallelism may avoid contention from the maximum setting",
            affected_paths=("$.runtime.threads",),
            evidence_summary="competing synthetic development hypothesis",
            confidence=0.50,
        ),
        THREADS_PATH,
        4,
    ),
)


def _path_text(path: JsonPath) -> str:
    return "$" + "".join(f".{segment.value}" for segment in path.segments)


def _value_at(configuration: Mapping[str, Any], path: JsonPath) -> object:
    value: object = configuration
    for segment in path.segments:
        if type(value) is not dict:
            raise TypeError("action path traversed a non-object value")
        value = value[segment.value]
    return value


def _phase(stage: str, parent: EvolutionCandidate, path: JsonPath) -> str:
    return (
        f"{stage}:parent={parent.occurrence.configuration_hash}:"
        f"path={_path_text(path)}"
    )


def _contract(path: JsonPath) -> MutationContract:
    return MutationContract(
        (path,),
        max_changed_paths=1,
        max_operations=1,
        allow_abstention=True,
    )


def _best_accepted_parent(
    outcomes: Sequence[InvocationOutcome],
    fallback: EvolutionCandidate,
) -> EvolutionCandidate:
    improving = [
        outcome
        for outcome in outcomes
        if outcome.candidate is not None
        and outcome.candidate.valid
        and outcome.candidate.operator_compliant
        and outcome.reward > 0
    ]
    if not improving:
        return fallback
    selected = max(improving, key=lambda outcome: outcome.reward).candidate
    assert selected is not None
    return selected


def _verified_recombination_parent(
    outcomes: Sequence[InvocationOutcome],
) -> EvolutionCandidate:
    candidates = [
        outcome.candidate
        for outcome in outcomes
        if outcome.prepared.plan.operator_kind
        is OperatorKind.THREE_WAY_RECOMBINATION
        and outcome.candidate is not None
        and outcome.candidate.valid
        and outcome.candidate.operator_compliant
        and outcome.candidate.preservation_verified is True
    ]
    if len(candidates) != 1:
        raise RuntimeError("v4 requires exactly one verified recombination parent")
    return candidates[0]


def _filtered_score_evidence(
    memory: InsightMemoryBank,
    context_hash: str,
    eligible: Sequence[InsightRef],
) -> list[dict[str, object]]:
    ids = {reference.insight_id.value for reference in eligible}
    return [
        record
        for record in memory.score_evidence(context_hash)
        if str(record["insight_id"]) in ids
    ]


def _action_record(
    outcome: InvocationOutcome,
    actions_by_id: Mapping[str, ActionHypothesis],
) -> dict[str, object] | None:
    decision = outcome.prepared.selection_decision
    if decision is None or not decision.selected:
        return None
    if len(decision.selected) != 1:
        raise RuntimeError("v4 action adherence requires singleton memory assignment")
    selected_id = decision.selected[0].insight_id.value
    action = actions_by_id[selected_id]
    candidate = outcome.candidate
    parent = outcome.prepared.plan.parents[0]
    observed = (
        None
        if candidate is None
        else _value_at(candidate.configuration_dict, action.path)
    )
    return {
        "insight_id": selected_id,
        "hypothesis": action.name,
        "path": _path_text(action.path),
        "recommended_value": action.recommended_value,
        "parent_value": _value_at(parent.configuration_dict, action.path),
        "observed_value": observed,
        "action_satisfied": candidate is not None
        and observed == action.recommended_value,
        "self_claimed": candidate is not None
        and selected_id in candidate.claimed_insight_ids,
        "abstained": candidate is not None
        and candidate.occurrence.configuration_hash
        == parent.occurrence.configuration_hash,
    }


def _outcome_record(
    outcome: InvocationOutcome,
    *,
    actions_by_id: Mapping[str, ActionHypothesis],
) -> dict[str, object]:
    record = support._outcome_record(
        outcome,
        known_target=pipeline_problem.DEVELOPMENT_RECOMBINATION_TARGET,
    )
    record["action_adherence"] = _action_record(outcome, actions_by_id)
    record["scoped_optimum_match"] = bool(
        outcome.candidate is not None
        and outcome.candidate.configuration_dict
        == pipeline_problem.DEVELOPMENT_RUNTIME_SCOPED_OPTIMUM
    )
    return record


def _preflight_balanced_assignment(
    *,
    memory: InsightMemoryBank,
    actions_by_reference: Mapping[InsightRef, ActionHypothesis],
    assignment_seed: int,
) -> dict[str, list[str]]:
    rng = random.Random(assignment_seed)
    result: dict[str, list[str]] = {}
    for path in (PREFETCH_PATH, LAYOUT_PATH, THREADS_PATH):
        eligible = memory.eligible_references(
            operator_kind=OperatorKind.TYPED_MUTATION.value,
            editable_paths=(_path_text(path),),
        )
        if len(eligible) != 2:
            raise RuntimeError("each v4 atomic coordinate requires two hypotheses")
        selected = [rng.sample(eligible, 1)[0] for _ in range(2)]
        if set(selected) != set(eligible):
            raise RuntimeError(
                "assignment seed does not balance both hypotheses per coordinate"
            )
        result[_path_text(path)] = [
            actions_by_reference[reference].name for reference in selected
        ]
    return result


async def _run_pipeline_v4(
    *,
    generator: PydanticAIAgenticGenerator,
    id_seed: int,
    assignment_seed: int,
    event_writer: support.JsonlWriter,
    max_output_tokens: int,
    temperature: float,
) -> dict[str, object]:
    ids = DeterministicIdFactory(f"live_pipeline_v4_{id_seed}")
    memory = InsightMemoryBank(
        id_factory=ids,
        exploration_probability=Fraction(1, 1),
        shrinkage_effective_sample_size=4.0,
    )
    actions_by_id: dict[str, ActionHypothesis] = {}
    actions_by_reference: dict[InsightRef, ActionHypothesis] = {}
    for hypothesis in HYPOTHESES:
        entry, added = memory.add(
            hypothesis.draft,
            applicable_operator_kinds=MUTATION_OPERATOR,
        )
        if not added or entry.initial_score != 0.0:
            raise RuntimeError("v4 seed memory must be unique and utility-neutral")
        actions_by_id[entry.reference.insight_id.value] = hypothesis
        actions_by_reference[entry.reference] = hypothesis

    assignment_plan = _preflight_balanced_assignment(
        memory=memory,
        actions_by_reference=actions_by_reference,
        assignment_seed=assignment_seed,
    )
    trace_events: list[dict[str, object]] = []

    def record_event(event: Mapping[str, object]) -> None:
        record = dict(event)
        trace_events.append(record)
        event_writer.write({"domain": "pipeline_codesign_v4", **record})

    engine = AgenticEvolutionEngine(
        problem=pipeline_problem.PipelineCoDesignProblem(),
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=assignment_seed,
        evaluator_concurrency=7,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        trace_sink=record_event,
    )
    base, left, right = await asyncio.gather(
        engine.register_seed(pipeline_problem.BASE_CONFIG, label="base"),
        engine.register_seed(
            pipeline_problem.DEVELOPMENT_BRANCH_LEFT,
            label="left_branch",
        ),
        engine.register_seed(
            pipeline_problem.DEVELOPMENT_BRANCH_RIGHT,
            label="right_branch",
        ),
    )

    generation_one = await engine.run_invocations(
        (
            InvocationPlan(
                OperatorKind.THREE_WAY_RECOMBINATION,
                (left, right),
                generation=1,
                label="g1_verified_recombination",
                common_ancestor=base,
                phase="v4_structural_composition",
            ),
            InvocationPlan(
                OperatorKind.REPRODUCTION,
                (left,),
                generation=1,
                label="g1_reproduction_control",
                phase="v4_structural_control",
            ),
        )
    )
    mutation_parent = _verified_recombination_parent(generation_one)

    discovery_paths = (
        PREFETCH_PATH,
        PREFETCH_PATH,
        LAYOUT_PATH,
        LAYOUT_PATH,
        THREADS_PATH,
        THREADS_PATH,
    )
    discovery_phases = {
        path: _phase("v4_atomic_discovery", mutation_parent, path)
        for path in (PREFETCH_PATH, LAYOUT_PATH, THREADS_PATH)
    }
    generation_two = await engine.run_invocations(
        tuple(
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (mutation_parent,),
                generation=2,
                label=f"g2_atomic_{_path_text(path).rsplit('.', 1)[-1]}_{index}",
                allowed_top_level=("runtime",),
                use_memory=True,
                memory_subset_size=1,
                memory_exploration_probability=Fraction(1, 1),
                phase=discovery_phases[path],
                mutation_contract=_contract(path),
            )
            for index, path in enumerate(discovery_paths, 1)
        )
    )

    observed_assignment: dict[str, list[str]] = {
        _path_text(path): []
        for path in (PREFETCH_PATH, LAYOUT_PATH, THREADS_PATH)
    }
    for outcome, path in zip(generation_two, discovery_paths, strict=True):
        selected = outcome.prepared.variation_case.selected_insights
        if len(selected) != 1:
            raise RuntimeError("v4 discovery requires one selected insight")
        observed_assignment[_path_text(path)].append(
            actions_by_reference[selected[0]].name
        )
    if observed_assignment != assignment_plan:
        raise RuntimeError("live assignment diverged from the outcome-blind preflight")

    discovery_contexts: dict[str, str] = {}
    score_evidence_at_exploitation: dict[str, list[dict[str, object]]] = {}
    seed_eligible_by_path: dict[str, tuple[InsightRef, ...]] = {}
    for path, phase in discovery_phases.items():
        text_path = _path_text(path)
        context = context_stratum_hash(
            problem_id=engine.problem_id,
            operator_kind=OperatorKind.TYPED_MUTATION.value,
            phase=phase,
        )
        eligible = tuple(
            reference
            for reference, hypothesis in actions_by_reference.items()
            if hypothesis.path == path
        )
        discovery_contexts[text_path] = context
        seed_eligible_by_path[text_path] = eligible
        score_evidence_at_exploitation[text_path] = _filtered_score_evidence(
            memory, context, eligible
        )

    discovery_parent = _best_accepted_parent(generation_two, mutation_parent)
    layout_discovery_phase = discovery_phases[LAYOUT_PATH]
    layout_score_context = context_stratum_hash(
        problem_id=engine.problem_id,
        operator_kind=OperatorKind.TYPED_MUTATION.value,
        phase=layout_discovery_phase,
    )
    generation_three_phase = _phase(
        "v4_layout_transport_test", discovery_parent, LAYOUT_PATH
    )
    generation_three = await engine.run_invocations(
        (
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (discovery_parent,),
                generation=3,
                label="g3_layout_no_memory_control",
                allowed_top_level=("runtime",),
                phase=generation_three_phase + ":control",
                mutation_contract=_contract(LAYOUT_PATH),
            ),
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (discovery_parent,),
                generation=3,
                label="g3_layout_score_transport",
                allowed_top_level=("runtime",),
                use_memory=True,
                memory_subset_size=1,
                memory_exploration_probability=Fraction(0, 1),
                memory_score_phase=layout_discovery_phase,
                phase=generation_three_phase + ":memory",
                mutation_contract=_contract(LAYOUT_PATH),
            ),
        )
    )
    final_parent = _best_accepted_parent(generation_three, discovery_parent)

    reflected = await engine.reflect(
        (*generation_two, *generation_three),
        label="v4_atomic_evidence_reflection",
        max_insights=3,
    )
    if not reflected:
        raise RuntimeError("v4 atomic reflection produced no testable insight")

    final_seed_score_evidence = {
        text_path: _filtered_score_evidence(
            memory,
            discovery_contexts[text_path],
            eligible,
        )
        for text_path, eligible in seed_eligible_by_path.items()
    }

    non_abstaining = [
        outcome.candidate.occurrence.configuration_hash
        for outcome in generation_two
        if outcome.candidate is not None
        and outcome.candidate.occurrence.configuration_hash
        != mutation_parent.occurrence.configuration_hash
    ]
    exploit = next(
        outcome
        for outcome in generation_three
        if outcome.prepared.plan.label == "g3_layout_score_transport"
    )
    scoped_optimum = pipeline_problem.DEVELOPMENT_RUNTIME_SCOPED_OPTIMUM
    scoped_objectives = (
        pipeline_problem.DEVELOPMENT_RUNTIME_SCOPED_OPTIMUM_OBJECTIVES
    )
    final_matches = final_parent.configuration_dict == scoped_optimum
    exploit_matches = bool(
        exploit.candidate is not None
        and exploit.candidate.configuration_dict == scoped_optimum
    )
    all_atomic_structural = all(
        outcome.candidate is not None and outcome.candidate.operator_compliant
        for outcome in (*generation_two, *generation_three)
    )
    cache_snapshot = await engine.evaluation_cache_snapshot()
    expected_logical_calls = 10

    return {
        "domain": "pipeline_codesign_v4",
        "development_only": True,
        "claim_boundary": (
            "Synthetic disclosed-evaluator workflow kill test only; not a "
            "benchmark, SOTA, or wall-clock result."
        ),
        "assignment_design": {
            "seed": assignment_seed,
            "policy": "two singleton uniform draws per path, seed preflighted for one exposure per competing hypothesis",
            "preflight": assignment_plan,
            "observed": observed_assignment,
        },
        "seed_candidates": {
            "base": support._candidate_record(base),
            "left": support._candidate_record(left),
            "right": support._candidate_record(right),
        },
        "mutation_parent_id": mutation_parent.candidate_id.value,
        "discovery_parent_id": discovery_parent.candidate_id.value,
        "final_parent_id": final_parent.candidate_id.value,
        "generation_one": [
            _outcome_record(outcome, actions_by_id=actions_by_id)
            for outcome in generation_one
        ],
        "generation_two_atomic_discovery": [
            _outcome_record(outcome, actions_by_id=actions_by_id)
            for outcome in generation_two
        ],
        "generation_three_transport_test": [
            _outcome_record(outcome, actions_by_id=actions_by_id)
            for outcome in generation_three
        ],
        "memory": {
            "seed_entry_count": len(HYPOTHESES),
            "final_entry_count": len(memory.entries),
            "trial_count": len(memory.trials),
            "confidence_used_as_utility_prior": False,
            "discovery_contexts": discovery_contexts,
            "score_evidence_at_exploitation": score_evidence_at_exploitation,
            "final_seed_score_evidence": final_seed_score_evidence,
            "reflected_entry_ids": [
                entry.reference.insight_id.value for entry in reflected
            ],
            "reflected_entries_status": "quarantined_untested",
            "layout_score_source_context": layout_score_context,
        },
        "evaluation_cache": cache_snapshot,
        "oracle": {
            "runtime_grid_size": 60,
            "scope": "all non-runtime fields fixed to the exact verified recombination target",
            "configuration": copy.deepcopy(scoped_optimum),
            "objectives": dict(scoped_objectives),
        },
        "gates": {
            "balanced_assignment_replayed": observed_assignment == assignment_plan,
            "verified_exact_recombination": (
                mutation_parent.configuration_dict
                == pipeline_problem.DEVELOPMENT_RECOMBINATION_TARGET
            ),
            "all_atomic_outputs_structurally_compliant": all_atomic_structural,
            "discovery_nonabstaining_count": len(non_abstaining),
            "discovery_nonabstaining_unique_count": len(set(non_abstaining)),
            "no_nonabstaining_duplicate": len(non_abstaining)
            == len(set(non_abstaining)),
            "score_transport_reached_scoped_optimum": exploit_matches,
            "population_reached_scoped_optimum": final_matches,
            "accepted_population_never_uses_negative_reward": True,
            "evaluation_cache_avoided_duplicate_evaluations": (
                int(cache_snapshot["hits"] or 0)
                + int(cache_snapshot["coalesced"] or 0)
                > 0
            ),
        },
        "counts": {
            "logical_variation_invocations": len(
                (*generation_one, *generation_two, *generation_three)
            ),
            "llm_variation_calls": 9,
            "reflection_calls": 1,
            "valid_candidates": sum(
                outcome.candidate is not None and outcome.candidate.valid
                for outcome in (*generation_one, *generation_two, *generation_three)
            ),
            "operator_compliant_candidates": sum(
                outcome.candidate is not None
                and outcome.candidate.operator_compliant
                for outcome in (*generation_one, *generation_two, *generation_three)
            ),
            "positive_reward_candidates": sum(
                outcome.reward > 0
                for outcome in (*generation_one, *generation_two, *generation_three)
            ),
        },
        "provider_calls": support._call_summary(
            trace_events,
            expected_logical_calls=expected_logical_calls,
        ),
    }


def _manifest(args: argparse.Namespace, run_id: str) -> dict[str, object]:
    manifest = support._manifest(args, run_id)
    manifest["schema_version"] = 2
    manifest["workflow"] = "v4_atomic_memory_transport_kill_test"
    manifest["assignment_seed"] = args.assignment_seed
    manifest["hard_target"] = {
        "configuration": copy.deepcopy(
            pipeline_problem.DEVELOPMENT_RUNTIME_SCOPED_OPTIMUM
        ),
        "objectives": dict(
            pipeline_problem.DEVELOPMENT_RUNTIME_SCOPED_OPTIMUM_OBJECTIVES
        ),
        "scope": "synthetic 60-point runtime grid with non-runtime fields fixed",
    }
    manifest["source_sha256"]["probe_v4"] = support._sha256(
        Path(__file__).resolve()
    )
    manifest["source_sha256"]["evaluation_cache"] = support._sha256(
        AGENT_EVOLVE_ROOT
        / "src"
        / "agent_evolve"
        / "application"
        / "evaluation_cache.py"
    )
    return manifest


def _finalize_run(run_dir: Path) -> None:
    """Seal the completed primary artifacts in a separate terminal index."""

    files: dict[str, dict[str, object]] = {}
    for name in ("manifest.json", "events.jsonl", "queue_outcomes.jsonl", "summary.json"):
        path = run_dir / name
        payload = path.read_bytes()
        record: dict[str, object] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if name.endswith(".jsonl"):
            record["lines"] = len(payload.splitlines())
        files[name] = record
    support._write_json(
        run_dir / "finalized.json",
        {
            "schema_version": 1,
            "run_status": "succeeded",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": files,
        },
    )


async def _run(args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")

    event_writer = support.JsonlWriter(run_dir / "events.jsonl")
    queue_writer = support.JsonlWriter(run_dir / "queue_outcomes.jsonl")
    started_ns = time.perf_counter_ns()
    structured = PydanticAIStructuredGenerator.openrouter(
        api_key=api_key,
        model_name=args.model,
        max_connections=args.max_in_flight,
        timeout_seconds=float(args.attempt_timeout_seconds),
        provider_options={
            "order": list(support.PROVIDER_ORDER),
            "allow_fallbacks": False,
        },
        app_title="AgentEvolve AAAI 2027 v4 atomic development probe",
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
            support._queue_outcome_record(outcome)
        ),
    )
    generator = PydanticAIAgenticGenerator(runner)
    try:
        async with runner:
            result = await _run_pipeline_v4(
                generator=generator,
                id_seed=args.seed,
                assignment_seed=args.assignment_seed,
                event_writer=event_writer,
                max_output_tokens=args.max_output_tokens,
                temperature=args.temperature,
            )
    finally:
        event_writer.close()
        queue_writer.close()

    provider_calls = dict(result["provider_calls"])
    queue = support._queue_log_summary(run_dir / "queue_outcomes.jsonl")
    if queue["terminal_outcomes"] != provider_calls["expected_logical_calls"]:
        raise RuntimeError("v4 queue/provider call accounting mismatch")
    return {
        "schema_version": 2,
        "development_only": True,
        "elapsed_ns": time.perf_counter_ns() - started_ns,
        "provider_calls": provider_calls,
        "queue": queue,
        "domains": [result],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("pipeline",), default="pipeline")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--assignment-seed", type=int, default=4)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--max-in-flight", type=int, default=7)
    parser.add_argument("--max-pending", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--attempt-timeout-seconds", type=int, default=90)
    parser.add_argument("--base-backoff-seconds", type=int, default=1)
    parser.add_argument("--max-backoff-seconds", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=2_400)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.model != MODEL:
        raise SystemExit(f"v4 development probe is frozen to {MODEL}")
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "probe_v4_%Y%m%dT%H%M%SZ"
    )
    run_dir = args.log_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    load_credentials(WORKSPACE_ROOT / ".env", override=False, optional=True)
    support._write_json(run_dir / "manifest.json", _manifest(args, run_id))
    try:
        summary = asyncio.run(_run(args, run_dir))
    except BaseException as exc:
        support._write_json(
            run_dir / "failure.json",
            {
                "failure_type": type(exc).__name__,
                "safe_message": (
                    str(exc)
                    if type(exc).__module__.startswith("agent_evolve")
                    else "v4 development probe failed; inspect sanitized traces"
                ),
            },
        )
        raise
    support._write_json(run_dir / "summary.json", summary)
    _finalize_run(run_dir)
    print(
        support._canonical_json(
            {"run_dir": str(run_dir), "status": "succeeded"}
        )
    )


if __name__ == "__main__":
    main()
