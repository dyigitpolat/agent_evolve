#!/usr/bin/env python3
"""Run the first bounded real-evaluator AgentEvolve BOiLS development pilot.

This is a workflow-debugging experiment, not paper benchmark evidence.  It uses
one calibrated 10--30 second BOiLS/ABC task to exercise genuine agentic
reproduction, crossover, ancestor-aware recombination, atomic mutation,
reflection, and an isolated quarantined-insight test with complete provenance.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.settings import load_credentials  # noqa: E402

from agent_evolve.application.agentic_evolution import (  # noqa: E402
    AgenticEvolutionEngine,
    EvolutionCandidate,
    InsightAssignmentKind,
    InvocationOutcome,
    InvocationPlan,
    MutationContract,
    OperatorKind,
    PreparedInvocation,
    default_evidence_prompt,
)
from agent_evolve.application.insight_memory import (  # noqa: E402
    InsightLifecycleState,
    InsightMemoryBank,
    InsightMemoryEntry,
)
from agent_evolve.application.pareto_archive import ParetoArchive  # noqa: E402
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey  # noqa: E402
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256  # noqa: E402
from agent_evolve.infrastructure.ids import DeterministicIdFactory  # noqa: E402
from agent_evolve.integrations.pydantic_ai.agentic_generator import (  # noqa: E402
    PydanticAIAgenticGenerator,
)
from agent_evolve.integrations.pydantic_ai.async_generator import (  # noqa: E402
    PydanticAIStructuredGenerator,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    create_production_queued_runner,
)

from examples.benchmarks.boils_abc.actions import (  # noqa: E402
    ACTION_IDS,
    DEFAULT_ACTION_SEQUENCE,
    SEQUENCE_LENGTH,
    config_sha256,
)
from examples.benchmarks.boils_abc.evaluator import (  # noqa: E402
    AbcEvaluatorSettings,
    BoilsAbcEvaluator,
    BoilsEvaluation,
    BoilsEvaluationFailure,
    BoilsEvaluationObservation,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem  # noqa: E402
from examples.development import run_agentic_probe as support  # noqa: E402


MODEL = "deepseek/deepseek-v4-pro"
PROVIDER_ORDER = support.PROVIDER_ORDER
PILOT_CIRCUITS = ("log2",)
PILOT_CPUS = (8, 9, 10, 11)
ATOMIC_MUTATION_INDICES = (1, 7, 12, 18)
EXPECTED_PILOT_LLM_CALLS = 9
DEFAULT_LOG_ROOT = (
    WORKSPACE_ROOT
    / "papers"
    / "agent_evolve_aaai_2027"
    / "research_artifacts"
    / "experiment_logs"
    / "boils_agentic_development"
)
_SEQUENCE_PATH = re.compile(r"^\$\.sequence(?:\[(?P<index>\d+)\])?$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return support._canonical_json(value)


class DurableJsonlWriter:
    """Thread-safe, fail-closed, fsync-on-record JSONL interceptor."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._stream = path.open("x", encoding="utf-8")
        self._closed = False

    def write(self, value: Mapping[str, object]) -> None:
        payload = _canonical_json(dict(value)) + "\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot write to a closed JSONL interceptor")
            self._stream.write(payload)
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._stream.close()
            self._closed = True


class EvaluationObservationRecorder:
    """Serialize evaluator callbacks that can arrive from worker threads."""

    def __init__(self, writer: DurableJsonlWriter) -> None:
        self._writer = writer
        self._lock = threading.Lock()
        self._sequence = 0

    @property
    def count(self) -> int:
        with self._lock:
            return self._sequence

    def __call__(self, observation: BoilsEvaluationObservation) -> None:
        if type(observation) is BoilsEvaluation:
            kind = "succeeded"
        elif type(observation) is BoilsEvaluationFailure:
            kind = "candidate_local_failure"
        else:  # pragma: no cover - closed evaluator observation union.
            raise TypeError("unknown BOiLS evaluator observation")
        with self._lock:
            self._sequence += 1
            self._writer.write(
                {
                    "schema_version": 1,
                    "observation_sequence": self._sequence,
                    "recorded_at_utc": _utc_now(),
                    "status": kind,
                    "observation": observation.as_dict(),
                }
            )


class TraceRecorder:
    """Add a runner-global ordering layer around engine and archive events."""

    def __init__(self, writer: DurableJsonlWriter) -> None:
        self._writer = writer
        self._sequence = 0
        self.events: list[dict[str, object]] = []

    def emit(self, event: Mapping[str, object]) -> None:
        self._sequence += 1
        record = dict(event)
        engine_sequence = record.pop("sequence", None)
        ordered = {
            "stream_sequence": self._sequence,
            "domain": "boils_abc_log2_length20",
            "engine_sequence": engine_sequence,
            **record,
        }
        self.events.append(ordered)
        self._writer.write(ordered)


def _seed_configurations() -> dict[str, dict[str, Any]]:
    base = list(DEFAULT_ACTION_SEQUENCE)
    left = base.copy()
    right = base.copy()
    composed = base.copy()
    left[4] = "fraig"
    right[15] = "resub_z"
    composed[4] = "fraig"
    composed[15] = "resub_z"
    return {
        "ancestor_a": {"sequence": base},
        "left_l": {"sequence": left},
        "right_r": {"sequence": right},
        "expected_composition_c": {"sequence": composed},
    }


SEEDS = _seed_configurations()
EXPECTED_CONFIG_HASHES = {
    "ancestor_a": "2f1b2c40172a4dd83e8d056a2b6581948ea0983055fea63d930791108509eef4",
    "left_l": "3c20d80b43bdf0e0842f8bc02d5739a156d14b110299806c18ceb0a58876b871",
    "right_r": "6ed5bba484b7f15c610fa8bddd25e101217a63ffc133af68b6ec5908e1626dac",
    "expected_composition_c": "e954b02443e92dbed5cc7aa21b8d452531400017d602bf5dcdc938fb84e5237e",
}
EXPECTED_TYPED_JSON_HASHES = {
    "ancestor_a": "91e8e9756403130ae67423409d6e40860228a8adc4a72bce68ac97a41530f878",
    "left_l": "755ece9e18a1262016da1b152a482f39ad9cac61535bc771e6fde6ba1a630604",
    "right_r": "33ac4888e43f39d4f1f18af97f00d8a4d32c1afc9d087b34b354831bdc886942",
    "expected_composition_c": "75451fb03ed5b60faa40eb1e956cc2ef86d9f8692e7f55b94ef054b4aab4012a",
}
if {
    name: config_sha256(configuration) for name, configuration in SEEDS.items()
} != EXPECTED_CONFIG_HASHES:  # pragma: no cover - frozen source invariant.
    raise RuntimeError("frozen BOiLS pilot candidate identities changed")
if {
    name: typed_json_sha256(freeze_json(configuration))
    for name, configuration in SEEDS.items()
} != EXPECTED_TYPED_JSON_HASHES:  # pragma: no cover - frozen source invariant.
    raise RuntimeError("frozen BOiLS pilot typed-JSON identities changed")


def _sequence_path(index: int) -> JsonPath:
    if type(index) is not int or not 0 <= index < SEQUENCE_LENGTH:
        raise ValueError("sequence index is outside the frozen BOiLS schema")
    return JsonPath((ObjectKey("sequence"), ArrayIndex(index)))


def _atomic_contract(index: int) -> MutationContract:
    return MutationContract(
        (_sequence_path(index),),
        max_changed_paths=1,
        max_operations=1,
        allow_abstention=False,
    )


def boils_prompt_builder(
    problem_description: str,
    prepared: PreparedInvocation,
    selected_insights: tuple[dict[str, object], ...],
) -> str:
    """Add a machine-derived sequence diff without changing engine semantics."""

    prompt = default_evidence_prompt(
        problem_description,
        prepared,
        selected_insights,
    )
    plan = prepared.plan
    if plan.operator_kind is not OperatorKind.TWO_PARENT_CROSSOVER:
        return prompt
    left = plan.parents[0].configuration_dict["sequence"]
    right = plan.parents[1].configuration_dict["sequence"]
    differences = [
        {
            "path": f"$.sequence[{index}]",
            "left": left_action,
            "right": right_action,
        }
        for index, (left_action, right_action) in enumerate(zip(left, right, strict=True))
        if left_action != right_action
    ]
    return "\n".join(
        (
            prompt,
            "",
            "MACHINE-DERIVED BOILS CROSSOVER DIFF",
            "Parent order is exactly left then right. The rows below are the only "
            "parent disagreements; they are evidence, not a prescribed child.",
            _canonical_json(differences),
            "For truthful source attribution, cite the exact $.sequence[index] "
            "path of each inherited disagreement and its left/right source. The "
            "system will independently reject a child without a contribution "
            "from both parents.",
        )
    )


def _candidate_record(candidate: EvolutionCandidate | None) -> dict[str, object] | None:
    record = support._candidate_record(candidate)
    if record is None or candidate is None:
        return None
    record.update(
        {
            "typed_json_configuration_sha256": candidate.occurrence.configuration_hash,
            "boils_schema_configuration_sha256": config_sha256(
                candidate.configuration_dict
            ),
            "selected_insight_refs": [
                {
                    "insight_id": reference.insight_id.value,
                    "version": reference.version,
                }
                for reference in candidate.selected_insight_refs
            ],
            "insight_assignment_kind": (
                None
                if candidate.insight_assignment_kind is None
                else candidate.insight_assignment_kind.value
            ),
        }
    )
    return record


def _outcome_record(outcome: InvocationOutcome) -> dict[str, object]:
    plan = outcome.prepared.plan
    contract = plan.mutation_contract
    return {
        "label": plan.label,
        "operator_kind": plan.operator_kind.value,
        "operator_invocation_id": outcome.prepared.operator_invocation_id.value,
        "call_id": (
            None
            if outcome.prepared.call_id is None
            else outcome.prepared.call_id.value
        ),
        "phase": plan.phase,
        "parents": [parent.candidate_id.value for parent in plan.parents],
        "common_ancestor_id": (
            None
            if plan.common_ancestor is None
            else plan.common_ancestor.candidate_id.value
        ),
        "mutation_contract": (
            None
            if contract is None
            else {
                "editable_paths": [
                    f"$.sequence[{path.segments[1].value}]"
                    for path in contract.editable_paths
                ],
                "max_changed_paths": contract.max_changed_paths,
                "max_operations": contract.max_operations,
                "allow_abstention": contract.allow_abstention,
            }
        ),
        "selected_insight_refs": [
            {
                "insight_id": reference.insight_id.value,
                "version": reference.version,
            }
            for reference in outcome.prepared.variation_case.selected_insights
        ],
        "assignment_kind": (
            None
            if outcome.prepared.insight_assignment_kind is None
            else outcome.prepared.insight_assignment_kind.value
        ),
        "call_failure_type": outcome.call_failure_type,
        "reward": outcome.reward,
        "dominates_any_parent": outcome.dominates_any_parent,
        "better_than_any_parent": outcome.better_than_any_parent,
        "candidate": _candidate_record(outcome.candidate),
    }


def _consider(
    archive: ParetoArchive,
    candidate: EvolutionCandidate | None,
    trace: TraceRecorder,
) -> None:
    if candidate is None:
        return
    for decision in archive.consider(candidate):
        trace.emit(decision.to_trace_record())


def _select_parent(
    archive: ParetoArchive,
    *,
    baseline: EvolutionCandidate,
    stage: str,
    trace: TraceRecorder,
) -> tuple[EvolutionCandidate, dict[str, object]]:
    candidates = archive.front or (baseline,)
    baseline_values = baseline.objective_map

    def key(candidate: EvolutionCandidate) -> tuple[float, float, str]:
        ratios = tuple(
            candidate.objective_map[objective.name]
            / baseline_values[objective.name]
            for objective in archive.objectives
        )
        return (sum(ratios), max(ratios), candidate.occurrence.configuration_hash)

    selected = min(candidates, key=key)
    record = {
        "event_type": "parent_selected",
        "stage": stage,
        "rule": (
            "minimum sum of raw objective ratios to ancestor A over the "
            "admissibility-gated Pareto front; then maximum ratio; then typed-JSON hash"
        ),
        "candidate_id": selected.candidate_id.value,
        "configuration_hash": selected.occurrence.configuration_hash,
        "boils_configuration_hash": config_sha256(selected.configuration_dict),
        "objectives": selected.objective_map,
        "baseline_candidate_id": baseline.candidate_id.value,
        "baseline_objectives": baseline_values,
    }
    trace.emit(record)
    return selected, record


def _reflection_record(entry: InsightMemoryEntry) -> dict[str, object]:
    lineage = entry.evidence_lineage
    return {
        "insight_id": entry.reference.insight_id.value,
        "version": entry.reference.version,
        "claim": entry.draft.claim,
        "trigger": entry.draft.trigger,
        "mechanism": entry.draft.mechanism,
        "affected_paths": list(entry.draft.affected_paths),
        "evidence_summary": entry.draft.evidence_summary,
        "confidence": entry.draft.confidence,
        "lifecycle_state": entry.lifecycle_state.value,
        "retrievable": entry.retrievable,
        "origin": entry.origin.value,
        "evidence_lineage": (
            None
            if lineage is None
            else {
                "reflection_call_id": lineage.reflection_call_id.value,
                "source_operator_invocation_ids": [
                    value.value for value in lineage.source_operator_invocation_ids
                ],
                "source_candidate_ids": [
                    value.value for value in lineage.source_candidate_ids
                ],
                "available_contrast_ids": list(lineage.available_contrast_ids),
                "cited_contrast_ids": list(lineage.cited_contrast_ids),
            }
        ),
    }


def _choose_quarantine_test(
    entries: Sequence[InsightMemoryEntry],
) -> tuple[InsightMemoryEntry, int, str]:
    sequence_wide: tuple[InsightMemoryEntry, str] | None = None
    for entry in entries:
        if entry.lifecycle_state is not InsightLifecycleState.QUARANTINED:
            continue
        for path in entry.draft.affected_paths:
            match = _SEQUENCE_PATH.fullmatch(path)
            if match is None:
                continue
            index_text = match.group("index")
            if index_text is None:
                if sequence_wide is None:
                    sequence_wide = (entry, path)
                continue
            index = int(index_text)
            if 0 <= index < SEQUENCE_LENGTH:
                return entry, index, path
    if sequence_wide is not None:
        entry, path = sequence_wide
        return entry, ATOMIC_MUTATION_INDICES[0], path
    raise RuntimeError(
        "reflection produced no quarantined insight applicable to $.sequence"
    )


async def _register_seeds(
    engine: AgenticEvolutionEngine,
    archive: ParetoArchive,
    trace: TraceRecorder,
) -> tuple[EvolutionCandidate, EvolutionCandidate, EvolutionCandidate]:
    ancestor, left, right = await asyncio.gather(
        engine.register_seed(copy.deepcopy(SEEDS["ancestor_a"]), label="ancestor_a"),
        engine.register_seed(copy.deepcopy(SEEDS["left_l"]), label="left_l"),
        engine.register_seed(copy.deepcopy(SEEDS["right_r"]), label="right_r"),
    )
    for candidate in (ancestor, left, right):
        _consider(archive, candidate, trace)
    return ancestor, left, right


async def run_workflow(
    *,
    problem: Any,
    generator: Any,
    id_seed: int,
    event_writer: DurableJsonlWriter,
    evaluator_concurrency: int,
    max_output_tokens: int,
    temperature: float,
) -> dict[str, object]:
    """Execute the frozen pilot against injected real or offline boundaries."""

    ids = DeterministicIdFactory(f"boils_agentic_pilot_v1_{id_seed}")
    memory = InsightMemoryBank(
        id_factory=ids,
        exploration_probability=Fraction(1, 1),
        shrinkage_effective_sample_size=4.0,
    )
    trace = TraceRecorder(event_writer)
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=id_seed,
        evaluator_concurrency=evaluator_concurrency,
        trace_sink=trace.emit,
        prompt_builder=boils_prompt_builder,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    archive = ParetoArchive(engine.objectives)
    ancestor, left, right = await _register_seeds(engine, archive, trace)

    generation_one = await engine.run_invocations(
        (
            InvocationPlan(
                OperatorKind.THREE_WAY_RECOMBINATION,
                (left, right),
                generation=1,
                label="g1_ancestor_aware_recombination",
                common_ancestor=ancestor,
                phase="boils_v1_structural_comparison",
            ),
            InvocationPlan(
                OperatorKind.TWO_PARENT_CROSSOVER,
                (left, right),
                generation=1,
                label="g1_ordinary_two_parent_crossover",
                phase="boils_v1_structural_comparison",
            ),
            InvocationPlan(
                OperatorKind.REPRODUCTION,
                (ancestor,),
                generation=1,
                label="g1_exact_reproduction_control",
                phase="boils_v1_structural_comparison",
            ),
        )
    )
    for outcome in generation_one:
        _consider(archive, outcome.candidate, trace)

    mutation_parent, mutation_selection = _select_parent(
        archive,
        baseline=ancestor,
        stage="before_atomic_discovery",
        trace=trace,
    )
    generation_two = await engine.run_invocations(
        tuple(
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (mutation_parent,),
                generation=2,
                label=f"g2_atomic_sequence_index_{index}",
                allowed_top_level=("sequence",),
                phase="boils_v1_atomic_discovery",
                mutation_contract=_atomic_contract(index),
            )
            for index in ATOMIC_MUTATION_INDICES
        )
    )
    for outcome in generation_two:
        _consider(archive, outcome.candidate, trace)

    reflected = await engine.reflect(
        generation_two,
        label="boils_v1_atomic_evidence_reflection",
        max_insights=3,
    )
    if not reflected:
        raise RuntimeError("atomic reflection returned no new testable insight")
    test_insight, test_index, matched_affected_path = _choose_quarantine_test(
        reflected
    )
    pair_parent, pair_selection = _select_parent(
        archive,
        baseline=ancestor,
        stage="before_quarantine_pair",
        trace=trace,
    )
    trials_before_pair = len(memory.trials)
    generation_three = await engine.run_invocations(
        (
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (pair_parent,),
                generation=3,
                label=f"g3_pair_control_index_{test_index}",
                allowed_top_level=("sequence",),
                phase="boils_v1_quarantine_pair",
                mutation_contract=_atomic_contract(test_index),
            ),
            InvocationPlan(
                OperatorKind.TYPED_MUTATION,
                (pair_parent,),
                generation=3,
                label=f"g3_pair_quarantine_test_index_{test_index}",
                allowed_top_level=("sequence",),
                quarantine_test_insights=(test_insight.reference,),
                phase="boils_v1_quarantine_pair",
                mutation_contract=_atomic_contract(test_index),
            ),
        )
    )
    for outcome in generation_three:
        _consider(archive, outcome.candidate, trace)
    trials_after_pair = len(memory.trials)

    snapshot = archive.snapshot()
    trace.emit(snapshot.to_trace_record())
    cache_snapshot = await engine.evaluation_cache_snapshot()
    all_outcomes = (*generation_one, *generation_two, *generation_three)
    recombination = generation_one[0].candidate
    crossover = generation_one[1].candidate
    reproduction = generation_one[2].candidate
    quarantine_candidate = generation_three[1].candidate
    tested_entry_after_pair = next(
        entry
        for entry in memory.entries
        if entry.reference == test_insight.reference
    )
    provider_calls = support._call_summary(
        trace.events,
        expected_logical_calls=EXPECTED_PILOT_LLM_CALLS,
    )

    return {
        "schema_version": 1,
        "status": "succeeded",
        "completed_at_utc": _utc_now(),
        "development_only": True,
        "claim_boundary": (
            "Single-task agentic workflow/debugging evidence on calibrated BOiLS "
            "log2 only; not a benchmark comparison, SOTA result, statistical "
            "memory-effect claim, or wall-clock-dominance claim."
        ),
        "task": {
            "domain": "boils_abc",
            "circuit_panel": list(PILOT_CIRCUITS),
            "sequence_length": SEQUENCE_LENGTH,
            "objectives": [
                {"name": objective.name, "goal": objective.goal}
                for objective in archive.objectives
            ],
        },
        "seeds": {
            "ancestor_a": _candidate_record(ancestor),
            "left_l": _candidate_record(left),
            "right_r": _candidate_record(right),
            "expected_composition_c": copy.deepcopy(
                SEEDS["expected_composition_c"]
            ),
            "expected_composition_c_boils_sha256": EXPECTED_CONFIG_HASHES[
                "expected_composition_c"
            ],
        },
        "generation_one_structural_comparison": [
            _outcome_record(outcome) for outcome in generation_one
        ],
        "mutation_parent_selection": mutation_selection,
        "generation_two_atomic_discovery": [
            _outcome_record(outcome) for outcome in generation_two
        ],
        "reflection": {
            "entries": [_reflection_record(entry) for entry in reflected],
            "selected_test_insight": {
                "insight_id": test_insight.reference.insight_id.value,
                "version": test_insight.reference.version,
                "matched_affected_path": matched_affected_path,
                "test_index": test_index,
            },
        },
        "quarantine_pair_parent_selection": pair_selection,
        "generation_three_quarantine_pair": [
            _outcome_record(outcome) for outcome in generation_three
        ],
        "memory": {
            "entry_count": len(memory.entries),
            "trial_count_before_pair": trials_before_pair,
            "trial_count_after_pair": trials_after_pair,
            "lifecycle_transitions": [
                {
                    "sequence": transition.sequence,
                    "insight_id": transition.reference.insight_id.value,
                    "version": transition.reference.version,
                    "prior_state": transition.prior_state.value,
                    "new_state": transition.new_state.value,
                    "reason": transition.reason,
                    "supporting_evidence": list(
                        transition.supporting_evidence
                    ),
                }
                for transition in memory.transitions
            ],
        },
        "pareto_archive": snapshot.to_trace_record(),
        "pareto_front": [_candidate_record(item) for item in snapshot.front_candidates],
        "evaluation_cache": cache_snapshot,
        "provider_calls": provider_calls,
        "counts": {
            "logical_variation_invocations": len(all_outcomes),
            "llm_variation_calls": EXPECTED_PILOT_LLM_CALLS - 1,
            "reflection_calls": 1,
            "evaluated_candidate_occurrences": sum(
                outcome.candidate is not None for outcome in all_outcomes
            )
            + 3,
            "valid_variation_candidates": sum(
                outcome.candidate is not None and outcome.candidate.valid
                for outcome in all_outcomes
            ),
            "operator_compliant_variation_candidates": sum(
                outcome.candidate is not None
                and outcome.candidate.operator_compliant
                for outcome in all_outcomes
            ),
            "evidence_compliant_variation_candidates": sum(
                outcome.candidate is not None
                and outcome.candidate.evidence_compliant
                for outcome in all_outcomes
            ),
        },
        "gates": {
            "all_three_seeds_valid": all(
                candidate.valid for candidate in (ancestor, left, right)
            ),
            "reproduction_exact": bool(
                reproduction is not None
                and reproduction.configuration_dict == ancestor.configuration_dict
                and reproduction.operator_compliant
            ),
            "ancestor_aware_recombination_exact_c": bool(
                recombination is not None
                and recombination.configuration_dict
                == SEEDS["expected_composition_c"]
                and recombination.operator_compliant
                and recombination.preservation_verified is True
            ),
            "ordinary_crossover_operator_compliant": bool(
                crossover is not None and crossover.operator_compliant
            ),
            "all_atomic_candidates_returned": all(
                outcome.candidate is not None for outcome in generation_two
            ),
            "all_atomic_candidates_operator_compliant": all(
                outcome.candidate is not None
                and outcome.candidate.operator_compliant
                for outcome in generation_two
            ),
            "atomic_nonabstention_enforced": all(
                outcome.candidate is not None
                and outcome.candidate.occurrence.configuration_hash
                != mutation_parent.occurrence.configuration_hash
                for outcome in generation_two
            ),
            "reflection_entries_all_quarantined": all(
                entry.lifecycle_state is InsightLifecycleState.QUARANTINED
                for entry in reflected
            ),
            "reflection_entries_all_nonretrievable": all(
                not entry.retrievable for entry in reflected
            ),
            "quarantine_test_assignment_recorded": bool(
                quarantine_candidate is not None
                and quarantine_candidate.insight_assignment_kind
                is InsightAssignmentKind.QUARANTINE_TEST
                and quarantine_candidate.selected_insight_refs
                == (test_insight.reference,)
            ),
            "quarantine_test_created_no_retrieval_credit": (
                trials_after_pair == trials_before_pair
            ),
            "tested_insight_remained_quarantined": (
                tested_entry_after_pair.lifecycle_state
                is InsightLifecycleState.QUARANTINED
            ),
            "tested_insight_not_auto_promoted": (
                not tested_entry_after_pair.retrievable
            ),
        },
    }


class _NoCallGenerator:
    """Structural port used for real-evaluator seed preflight only."""

    async def propose(self, request: object) -> object:  # pragma: no cover
        del request
        raise AssertionError("seed preflight must not request a proposal")

    async def reflect(self, request: object) -> object:  # pragma: no cover
        del request
        raise AssertionError("seed preflight must not request reflection")


async def run_seed_preflight(
    *,
    problem: Any,
    id_seed: int,
    event_writer: DurableJsonlWriter,
    evaluator_concurrency: int,
) -> dict[str, object]:
    ids = DeterministicIdFactory(f"boils_agentic_seed_preflight_{id_seed}")
    memory = InsightMemoryBank(id_factory=ids)
    trace = TraceRecorder(event_writer)
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=_NoCallGenerator(),
        id_factory=ids,
        memory=memory,
        seed=id_seed,
        evaluator_concurrency=evaluator_concurrency,
        trace_sink=trace.emit,
    )
    archive = ParetoArchive(engine.objectives)
    ancestor, left, right = await _register_seeds(engine, archive, trace)
    snapshot = archive.snapshot()
    trace.emit(snapshot.to_trace_record())
    return {
        "schema_version": 1,
        "status": "succeeded",
        "completed_at_utc": _utc_now(),
        "development_only": True,
        "claim_boundary": (
            "Fixed-seed real-evaluator preflight only; no LLM, optimizer, "
            "baseline comparison, SOTA result, or wall-clock claim."
        ),
        "mode": "seed_preflight",
        "seeds": {
            "ancestor_a": _candidate_record(ancestor),
            "left_l": _candidate_record(left),
            "right_r": _candidate_record(right),
        },
        "pareto_archive": snapshot.to_trace_record(),
        "pareto_front": [_candidate_record(item) for item in snapshot.front_candidates],
        "evaluation_cache": await engine.evaluation_cache_snapshot(),
        "provider_calls": {
            "expected_logical_calls": 0,
            "successful_logical_calls": 0,
            "failed_logical_calls": 0,
        },
        "gates": {
            "all_three_seeds_valid": all(
                candidate.valid for candidate in (ancestor, left, right)
            ),
            "all_seed_hashes_match_frozen_plan": all(
                config_sha256(candidate.configuration_dict)
                == EXPECTED_CONFIG_HASHES[label]
                for label, candidate in (
                    ("ancestor_a", ancestor),
                    ("left_l", left),
                    ("right_r", right),
                )
            ),
        },
    }


async def _run_live_pilot(
    *,
    args: argparse.Namespace,
    problem: BoilsAbcProblem,
    event_writer: DurableJsonlWriter,
    queue_writer: DurableJsonlWriter,
) -> dict[str, object]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")
    structured = PydanticAIStructuredGenerator.openrouter(
        api_key=api_key,
        model_name=args.model,
        max_connections=args.max_in_flight,
        timeout_seconds=float(args.attempt_timeout_seconds),
        provider_options={
            "order": list(PROVIDER_ORDER),
            "allow_fallbacks": False,
        },
        app_title="AgentEvolve AAAI 2027 BOiLS agentic pilot v1",
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
    async with runner:
        return await run_workflow(
            problem=problem,
            generator=generator,
            id_seed=args.seed,
            event_writer=event_writer,
            evaluator_concurrency=len(PILOT_CPUS),
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
        )


def _source_hashes() -> dict[str, str]:
    sources = {
        "runner": Path(__file__).resolve(),
        "actions": AGENT_EVOLVE_ROOT
        / "examples/benchmarks/boils_abc/actions.py",
        "evaluator": AGENT_EVOLVE_ROOT
        / "examples/benchmarks/boils_abc/evaluator.py",
        "problem": AGENT_EVOLVE_ROOT
        / "examples/benchmarks/boils_abc/problem_def.py",
        "engine": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/agentic_evolution.py",
        "memory": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/insight_memory.py",
        "pareto_archive": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/pareto_archive.py",
        "queue": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/llm_task_queue.py",
        "queue_domain": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/domain/llm_task_queue.py",
        "backoff": AGENT_EVOLVE_ROOT / "src/agent_evolve/policies/llm_backoff.py",
        "typed_patch": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/policies/variation/typed_patch.py",
        "queued_runner": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/queued_runner.py",
        "provider_adapter": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/async_generator.py",
        "agentic_adapter": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/agentic_generator.py",
    }
    return {name: support._sha256(path) for name, path in sources.items()}


def _manifest(
    args: argparse.Namespace,
    *,
    run_id: str,
    evaluator: BoilsAbcEvaluator,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "started_at_utc": _utc_now(),
        "mode": args.mode,
        "development_only": True,
        "claim_boundary": (
            "BOiLS/log2 workflow-development experiment only; not final "
            "benchmark evidence, SOTA, or wall-clock dominance."
        ),
        "frozen_workflow": {
            "seeds": copy.deepcopy(SEEDS),
            "boils_configuration_sha256": dict(EXPECTED_CONFIG_HASHES),
            "typed_json_configuration_sha256": dict(
                EXPECTED_TYPED_JSON_HASHES
            ),
            "generation_one": [
                "ancestor_aware_recombination(left_l,right_r|ancestor_a)",
                "ordinary_two_parent_crossover(left_l,right_r)",
                "exact_reproduction(ancestor_a)",
            ],
            "atomic_mutation_indices": list(ATOMIC_MUTATION_INDICES),
            "reflection_max_insights": 3,
            "quarantine_test": (
                "paired no-memory and exact-version quarantine assignment on "
                "the same selected parent and exact index; no automatic credit or promotion"
            ),
            "prompt_policy": (
                "default_evidence_prompt plus machine-derived BOiLS crossover "
                "parent-difference rows"
            ),
            "expected_pilot_llm_calls": EXPECTED_PILOT_LLM_CALLS,
        },
        "task": {
            "circuits": list(PILOT_CIRCUITS),
            "sequence_length": SEQUENCE_LENGTH,
            "allowed_actions": list(ACTION_IDS),
            "raw_objectives": ["total_lut_count", "total_levels"],
            "calibration_artifact": (
                "length20_panel_calibration_v3_log2_20260713"
            ),
            "calibrated_median_seconds": 18.692041769,
        },
        "evaluator_provenance": evaluator.provenance(),
        "model": args.model,
        "provider": "openrouter",
        "provider_options": {
            "order": list(PROVIDER_ORDER),
            "allow_fallbacks": False,
        },
        "queue": {
            "enabled": args.mode == "pilot",
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
        "seed": args.seed,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "source_sha256": _source_hashes(),
        "python_source_snapshot": support._source_snapshot(
            (
                AGENT_EVOLVE_ROOT / "src",
                AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc",
                AGENT_EVOLVE_ROOT / "examples/development",
            )
        ),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
            "process_affinity_at_start": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "credential_variable": "OPENROUTER_API_KEY",
            "packages": {
                name: support._package_version(name)
                for name in ("pydantic", "pydantic-ai", "openai", "httpx")
            },
        },
    }


def _evaluation_log_summary(path: Path) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    statuses = [str(record["status"]) for record in records]
    observations = [dict(record["observation"]) for record in records]
    elapsed = [float(record["elapsed_s"]) for record in observations]
    affinities = [
        json.dumps(record.get("cpu_affinity"), separators=(",", ":"))
        for record in observations
    ]
    hashes = [str(record["configuration_sha256"]) for record in observations]
    return {
        "observations": len(records),
        "statuses": {
            status: statuses.count(status) for status in sorted(set(statuses))
        },
        "unique_boils_configuration_sha256": len(set(hashes)),
        "total_evaluator_elapsed_s": sum(elapsed),
        "max_evaluator_elapsed_s": max(elapsed, default=0.0),
        "cpu_affinity_counts": {
            value: affinities.count(value) for value in sorted(set(affinities))
        },
    }


def _finalize(run_dir: Path, status: str) -> None:
    names = (
        "manifest.json",
        "runner_source.py",
        "events.jsonl",
        "queue_outcomes.jsonl",
        "evaluations.jsonl",
        "summary.json",
        "failure.json",
    )
    files: dict[str, dict[str, object]] = {}
    for name in names:
        path = run_dir / name
        if not path.exists():
            continue
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
            "status": status,
            "completed_at_utc": _utc_now(),
            "files": files,
        },
    )


def _parse_cpus(value: str) -> tuple[int, ...]:
    try:
        cpus = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("CPUs must be comma-separated integers") from exc
    if len(cpus) != 4 or len(set(cpus)) != 4 or any(cpu < 0 for cpu in cpus):
        raise argparse.ArgumentTypeError("exactly four distinct non-negative CPUs required")
    return cpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "seed_preflight"), default="pilot")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--cpus", type=_parse_cpus, default=PILOT_CPUS)
    parser.add_argument("--per-circuit-timeout-s", type=float, default=60.0)
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
        raise SystemExit(f"BOiLS pilot is frozen to {MODEL}")
    if args.seed < 0:
        raise SystemExit("seed must be non-negative")
    if args.per_circuit_timeout_s <= 0:
        raise SystemExit("per-circuit timeout must be positive")
    if not 1 <= args.max_in_flight <= 64:
        raise SystemExit("max-in-flight must lie in [1,64]")
    if args.max_pending < 0:
        raise SystemExit("max-pending must be non-negative")

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        f"boils_agentic_{args.mode}_v1_%Y%m%dT%H%M%SZ"
    )
    run_dir = args.log_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__).resolve(), run_dir / "runner_source.py")

    event_writer = DurableJsonlWriter(run_dir / "events.jsonl")
    evaluation_writer = DurableJsonlWriter(run_dir / "evaluations.jsonl")
    queue_writer: DurableJsonlWriter | None = None
    load_credentials(WORKSPACE_ROOT / ".env", override=False, optional=True)
    started_ns = time.perf_counter_ns()
    status = "failed"
    try:
        settings = AbcEvaluatorSettings.current_circuit_panel(
            circuit_names=PILOT_CIRCUITS,
            affinity_sets=tuple((cpu,) for cpu in args.cpus),
            per_circuit_timeout_s=args.per_circuit_timeout_s,
        )
        observation_recorder = EvaluationObservationRecorder(evaluation_writer)
        evaluator = BoilsAbcEvaluator(settings, observer=observation_recorder)
        problem = BoilsAbcProblem(settings, evaluator=evaluator)
        support._write_json(
            run_dir / "manifest.json",
            _manifest(args, run_id=run_id, evaluator=evaluator),
        )

        if args.mode == "seed_preflight":
            summary = asyncio.run(
                run_seed_preflight(
                    problem=problem,
                    id_seed=args.seed,
                    event_writer=event_writer,
                    evaluator_concurrency=len(args.cpus),
                )
            )
        else:
            queue_writer = DurableJsonlWriter(run_dir / "queue_outcomes.jsonl")
            summary = asyncio.run(
                _run_live_pilot(
                    args=args,
                    problem=problem,
                    event_writer=event_writer,
                    queue_writer=queue_writer,
                )
            )
        summary["runner_elapsed_ns"] = time.perf_counter_ns() - started_ns
        summary["evaluator_observations"] = _evaluation_log_summary(
            run_dir / "evaluations.jsonl"
        )
        if args.mode == "pilot":
            assert queue_writer is not None
            queue_writer.close()
            queue_writer = None
            queue_summary = support._queue_log_summary(
                run_dir / "queue_outcomes.jsonl"
            )
            expected = int(summary["provider_calls"]["expected_logical_calls"])
            if queue_summary["terminal_outcomes"] != expected:
                raise RuntimeError("queue/provider logical call accounting mismatch")
            summary["queue"] = queue_summary
        support._write_json(run_dir / "summary.json", summary)
        status = "succeeded"
    except BaseException as exc:
        support._write_json(
            run_dir / "failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "failure_type": type(exc).__name__,
                "safe_message": (
                    str(exc)
                    if type(exc).__module__.startswith("agent_evolve")
                    or type(exc).__module__.startswith("examples")
                    else "BOiLS agentic pilot failed; inspect sanitized traces"
                ),
            },
        )
        raise
    finally:
        if queue_writer is not None:
            queue_writer.close()
        event_writer.close()
        evaluation_writer.close()
        _finalize(run_dir, status)

    print(_canonical_json({"run_dir": str(run_dir), "status": status}))


if __name__ == "__main__":
    main()
