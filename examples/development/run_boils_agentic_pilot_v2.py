#!/usr/bin/env python3
"""Run the frozen BOiLS patch-native AgentEvolve development pilot v2.

This bounded mechanism experiment evaluates one frozen parent, requests four
concurrent scalar replacements, and reflects over the complete fixed block.
It is not benchmark, memory-utility, SOTA, or wall-clock-dominance evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from dotenv import load_dotenv  # noqa: E402

from agent_evolve.application.agentic_evolution import (  # noqa: E402
    AgenticEvolutionEngine,
    EvolutionCandidate,
    InvocationOutcome,
    InvocationPlan,
    MutationContract,
    MutationResponseMode,
    OperatorKind,
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
    SEQUENCE_LENGTH,
    config_sha256,
)
from examples.benchmarks.boils_abc.evaluator import (  # noqa: E402
    AbcEvaluatorSettings,
    BoilsAbcEvaluator,
)
from examples.benchmarks.boils_abc.problem_def import BoilsAbcProblem  # noqa: E402
from examples.development import run_agentic_probe as support  # noqa: E402
from examples.development import run_boils_agentic_pilot as v1  # noqa: E402


MODEL = "deepseek/deepseek-v4-pro"
PROVIDER_ORDER = ("together", "parasail", "wandb")
ALLOWED_RESOLVED_PROVIDERS = frozenset(("Together", "Parasail", "W&B"))
PILOT_CIRCUITS = ("log2",)
PILOT_CPUS = (8, 9, 10, 11)
MUTATION_INDICES = (1, 7, 12, 18)
EXPECTED_VARIATION_CALLS = 4
EXPECTED_REFLECTION_CALLS = 1
EXPECTED_TOTAL_CALLS = 5
EXPECTED_SEED_OBJECTIVES = {
    "total_lut_count": 7_944.0,
    "total_levels": 69.0,
}
EXPECTED_ABC_SHA256 = (
    "21f3673079a1ea21378b817e5035a3a008ffc76e2656d8739906d059a7928232"
)
EXPECTED_CIRCUIT_SHA256 = (
    "c0d052af4e95de4c1327a2ceddd855518a052a8f3a3960e6d58c5b5ca65c0dde"
)
EXPECTED_PARENT_BOILS_SHA256 = (
    "e954b02443e92dbed5cc7aa21b8d452531400017d602bf5dcdc938fb84e5237e"
)
EXPECTED_PARENT_TYPED_SHA256 = (
    "75451fb03ed5b60faa40eb1e956cc2ef86d9f8692e7f55b94ef054b4aab4012a"
)
EXPECTED_LEGAL_FILE_SHA256 = (
    "49f14616ecc4c931a41f6fa43e6dfb31ce79747f661613865f2dd2aab38fb9e9"
)
MAX_SUCCESSFUL_RESPONSE_COST_USD = Decimal("0.05")

QUEUE_MAX_IN_FLIGHT = 4
QUEUE_MAX_PENDING = 8
QUEUE_MAX_ATTEMPTS = 2
QUEUE_ATTEMPT_TIMEOUT_SECONDS = 60
QUEUE_BASE_BACKOFF_SECONDS = 1
QUEUE_MAX_BACKOFF_SECONDS = 8
MAX_OUTPUT_TOKENS = 2_400
TEMPERATURE = 0.2

DEFAULT_LOG_ROOT = (
    WORKSPACE_ROOT
    / "papers"
    / "agent_evolve_aaai_2027"
    / "research_artifacts"
    / "experiment_logs"
    / "boils_agentic_development"
)
LEGAL_CHILD_PATH = (
    WORKSPACE_ROOT
    / "papers"
    / "agent_evolve_aaai_2027"
    / "research_artifacts"
    / "data"
    / "boils_v2_patch_native_legal_children.json"
)

PARENT_C: dict[str, Any] = {
    "sequence": [
        "balance",
        "rewrite",
        "refactor",
        "balance",
        "fraig",
        "rewrite_z",
        "balance",
        "refactor_z",
        "rewrite_z",
        "balance",
        "balance",
        "rewrite",
        "refactor",
        "balance",
        "rewrite",
        "resub_z",
        "balance",
        "refactor_z",
        "rewrite_z",
        "balance",
    ]
}

FACTORIAL_REPLAY_CELLS: dict[str, dict[str, object]] = {
    "parent_c": {
        "boils_configuration_sha256": EXPECTED_PARENT_BOILS_SHA256,
        "objectives": (7_944, 69),
    },
    "index_1_only": {
        "boils_configuration_sha256": (
            "bd71137843f397e063798cb94ca6ec4cb34e565ce9c2ad0c7ddba5f592016372"
        ),
        "objectives": (7_935, 69),
    },
    "index_12_only": {
        "boils_configuration_sha256": (
            "5fb1adfa2cb0aeeacbfefa1a9f5aace3a838ec01f350934c248321e066fb3378"
        ),
        "objectives": (7_931, 69),
    },
    "both": {
        "boils_configuration_sha256": (
            "df54c93433c38c2b2d839f9947631459d73a4995d27f617c5d7729bd45ce1609"
        ),
        "objectives": (7_918, 70),
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return support._canonical_json(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sequence_path(index: int) -> JsonPath:
    if type(index) is not int or index not in MUTATION_INDICES:
        raise ValueError("index is outside the frozen v2 mutation matrix")
    return JsonPath((ObjectKey("sequence"), ArrayIndex(index)))


def _path_text(index: int) -> str:
    return f"$.sequence[{index}]"


def _atomic_contract(index: int) -> MutationContract:
    return MutationContract(
        (_sequence_path(index),),
        max_changed_paths=1,
        max_operations=1,
        allow_abstention=False,
    )


def _materialize_child(index: int, replacement: str) -> dict[str, Any]:
    child = copy.deepcopy(PARENT_C)
    child["sequence"][index] = replacement
    return child


def _load_and_validate_legal_universe() -> tuple[bytes, dict[str, object]]:
    payload = LEGAL_CHILD_PATH.read_bytes()
    if _sha256_bytes(payload) != EXPECTED_LEGAL_FILE_SHA256:
        raise RuntimeError("frozen BOiLS v2 legal-child file hash changed")
    parsed = json.loads(payload)
    if type(parsed) is not dict or set(parsed) != {
        "schema_version",
        "parent_boils_configuration_sha256",
        "parent_typed_json_configuration_sha256",
        "indices",
    }:
        raise RuntimeError("frozen legal-child document has an unexpected shape")
    if parsed["schema_version"] != 1:
        raise RuntimeError("frozen legal-child schema version changed")
    if parsed["parent_boils_configuration_sha256"] != EXPECTED_PARENT_BOILS_SHA256:
        raise RuntimeError("legal-child BOiLS parent identity changed")
    if parsed["parent_typed_json_configuration_sha256"] != EXPECTED_PARENT_TYPED_SHA256:
        raise RuntimeError("legal-child typed parent identity changed")
    indices = parsed["indices"]
    if type(indices) is not dict or set(indices) != {
        str(index) for index in MUTATION_INDICES
    }:
        raise RuntimeError("legal-child index set changed")

    row_count = 0
    for index in MUTATION_INDICES:
        block = indices[str(index)]
        if type(block) is not dict or set(block) != {
            "parent_value",
            "path",
            "legal_children",
        }:
            raise RuntimeError(f"legal-child block {index} has an unexpected shape")
        parent_value = PARENT_C["sequence"][index]
        if block["parent_value"] != parent_value or block["path"] != _path_text(index):
            raise RuntimeError(f"legal-child block {index} changed its parent or path")
        rows = block["legal_children"]
        if type(rows) is not list or len(rows) != len(ACTION_IDS) - 1:
            raise RuntimeError(f"legal-child block {index} must contain ten rows")
        replacements: set[str] = set()
        for row in rows:
            if type(row) is not dict or set(row) != {
                "replacement",
                "boils_configuration_sha256",
                "typed_json_configuration_sha256",
            }:
                raise RuntimeError("legal-child row has an unexpected shape")
            replacement = row["replacement"]
            if type(replacement) is not str:
                raise RuntimeError("legal-child replacement must be a string")
            replacements.add(replacement)
            child = _materialize_child(index, replacement)
            if config_sha256(child) != row["boils_configuration_sha256"]:
                raise RuntimeError("legal-child BOiLS hash does not materialize")
            if typed_json_sha256(freeze_json(child)) != row[
                "typed_json_configuration_sha256"
            ]:
                raise RuntimeError("legal-child typed hash does not materialize")
            row_count += 1
        if replacements != set(ACTION_IDS) - {parent_value}:
            raise RuntimeError(f"legal-child replacement set changed at index {index}")
    if row_count != 40:
        raise RuntimeError("frozen legal-child universe must contain exactly 40 rows")
    return payload, parsed


LEGAL_CHILD_BYTES, LEGAL_CHILD_UNIVERSE = _load_and_validate_legal_universe()

if config_sha256(PARENT_C) != EXPECTED_PARENT_BOILS_SHA256:
    raise RuntimeError("frozen BOiLS v2 parent identity changed")
if typed_json_sha256(freeze_json(PARENT_C)) != EXPECTED_PARENT_TYPED_SHA256:
    raise RuntimeError("frozen BOiLS v2 typed parent identity changed")


def legal_child_rows() -> tuple[dict[str, object], ...]:
    """Return fresh rows in the frozen archive/report order."""

    return tuple(
        {
            "index": index,
            "path": _path_text(index),
            **copy.deepcopy(row),
        }
        for index in MUTATION_INDICES
        for row in LEGAL_CHILD_UNIVERSE["indices"][str(index)]["legal_children"]
    )


def factorial_replay() -> dict[str, object]:
    """Reproduce the frozen v1 two-by-two local interaction arithmetically."""

    baseline = FACTORIAL_REPLAY_CELLS["parent_c"]["objectives"]
    index_1 = FACTORIAL_REPLAY_CELLS["index_1_only"]["objectives"]
    index_12 = FACTORIAL_REPLAY_CELLS["index_12_only"]["objectives"]
    both = FACTORIAL_REPLAY_CELLS["both"]["objectives"]
    assert type(baseline) is tuple
    assert type(index_1) is tuple
    assert type(index_12) is tuple
    assert type(both) is tuple
    effect_1 = tuple(value - base for value, base in zip(index_1, baseline, strict=True))
    effect_12 = tuple(
        value - base for value, base in zip(index_12, baseline, strict=True)
    )
    observed_joint = tuple(
        value - base for value, base in zip(both, baseline, strict=True)
    )
    additive_expected = tuple(
        left + right for left, right in zip(effect_1, effect_12, strict=True)
    )
    interaction = tuple(
        observed - additive
        for observed, additive in zip(observed_joint, additive_expected, strict=True)
    )
    return {
        "cells": copy.deepcopy(FACTORIAL_REPLAY_CELLS),
        "index_1_effect": effect_1,
        "index_12_effect": effect_12,
        "observed_joint_effect": observed_joint,
        "additive_expected_joint_effect": additive_expected,
        "interaction": interaction,
        "interpretation": (
            "exact local epistasis evidence and a counterexample to context-free "
            "additive transfer; not generalized outside this neighborhood"
        ),
    }


def _legal_child_match(
    index: int,
    candidate: EvolutionCandidate,
) -> dict[str, object]:
    configuration = candidate.configuration_dict
    sequence = configuration.get("sequence")
    if type(sequence) is not list or len(sequence) != SEQUENCE_LENGTH:
        raise RuntimeError("v2 child escaped the BOiLS sequence schema")
    if any(
        value != PARENT_C["sequence"][position]
        for position, value in enumerate(sequence)
        if position != index
    ):
        raise RuntimeError("v2 child changed a path outside its assigned scalar")
    replacement = sequence[index]
    rows = LEGAL_CHILD_UNIVERSE["indices"][str(index)]["legal_children"]
    matching = [row for row in rows if row["replacement"] == replacement]
    if len(matching) != 1:
        raise RuntimeError("v2 replacement is outside its preregistered legal set")
    row = matching[0]
    boils_hash = config_sha256(configuration)
    typed_hash = candidate.occurrence.configuration_hash
    if boils_hash != row["boils_configuration_sha256"]:
        raise RuntimeError("live child does not match its preregistered BOiLS hash")
    if typed_hash != row["typed_json_configuration_sha256"]:
        raise RuntimeError("live child does not match its preregistered typed hash")
    return {
        "index": index,
        "path": _path_text(index),
        "parent_value": PARENT_C["sequence"][index],
        **copy.deepcopy(row),
    }


def _candidate_record(candidate: EvolutionCandidate | None) -> dict[str, object] | None:
    return v1._candidate_record(candidate)


def _reflection_record(entry: InsightMemoryEntry) -> dict[str, object]:
    record = v1._reflection_record(entry)
    record["evidence_contrast_ids"] = list(entry.draft.evidence_contrast_ids)
    return record


def _call_summary(
    events: Sequence[Mapping[str, object]],
    *,
    expected_executed_calls: int,
) -> dict[str, object]:
    summary = support._call_summary(
        events,
        expected_logical_calls=expected_executed_calls,
    )
    attempts = summary.pop("successful_attempts_reported")
    summary["total_attempts_for_successful_logical_calls"] = attempts
    summary["preregistered_logical_calls"] = EXPECTED_TOTAL_CALLS
    summary["executed_logical_calls"] = expected_executed_calls
    summary["reflection_skipped_calls"] = EXPECTED_TOTAL_CALLS - expected_executed_calls
    return summary


def _reflection_rows(events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    requested = [event for event in events if event.get("event_type") == "reflection_requested"]
    if not requested:
        return []
    if len(requested) != 1:
        raise RuntimeError("v2 must have at most one reflection request")
    prompt = str(requested[0]["prompt"])
    marker = "\nEVALUATED TRACE\n"
    suffix = "\n\nReturn at most "
    if marker not in prompt or suffix not in prompt:
        raise RuntimeError("reflection prompt framing changed")
    payload = prompt.split(marker, 1)[1].rsplit(suffix, 1)[0]
    rows = json.loads(payload)
    if type(rows) is not list:
        raise RuntimeError("reflection trace must be a JSON array")
    return rows


def _atomic_event_map(
    events: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        str(event["candidate_id"]): dict(event)
        for event in events
        if event.get("event_type") == "candidate_evaluated"
        and event.get("mutation_response_mode")
        == MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1.value
    }


def _atomic_trace_gate(
    *,
    index: int,
    candidate: EvolutionCandidate,
    event: Mapping[str, object],
    legal: Mapping[str, object],
) -> bool:
    path = _path_text(index)
    replacement = legal["replacement"]
    return bool(
        event.get("atomic_submitted_path") == path
        and event.get("atomic_old_value_hash")
        == typed_json_sha256(freeze_json(PARENT_C["sequence"][index]))
        and event.get("atomic_new_value_hash")
        == typed_json_sha256(freeze_json(replacement))
        and event.get("parent_configuration_hash") == EXPECTED_PARENT_TYPED_SHA256
        and event.get("target_configuration_hash")
        == candidate.occurrence.configuration_hash
        and event.get("source_attribution_provenance") == "system_derived"
        and event.get("intended_changes") == [path]
        and event.get("source_attribution") == [{"path": path, "source": "mutation"}]
        and type(event.get("materialized_patch_hash")) is str
        and event.get("parent_patch_hashes") == [event.get("materialized_patch_hash")]
        and candidate.operator_compliant
        and candidate.evidence_compliant
        and candidate.source_attribution
        and len(candidate.source_attribution) == 1
        and candidate.source_attribution[0].path == path
        and candidate.source_attribution[0].source == "mutation"
    )


def _reflection_machine_fact_gate(
    rows: Sequence[Mapping[str, object]],
    outcomes: Sequence[InvocationOutcome],
) -> bool:
    if len(rows) != len(MUTATION_INDICES):
        return False
    for index, row, outcome in zip(MUTATION_INDICES, rows, outcomes, strict=True):
        candidate = outcome.candidate
        if candidate is None:
            return False
        contrasts = row.get("machine_derived_contrasts")
        if type(contrasts) is not list or len(contrasts) != 1:
            return False
        contrast = contrasts[0]
        operations = contrast.get("system_derived_operations")
        if type(operations) is not list or len(operations) != 1:
            return False
        operation = operations[0]
        replacement = candidate.configuration_dict["sequence"][index]
        patch_hashes = candidate.parent_patch_hashes
        if not (
            type(contrast.get("contrast_id")) is str
            and len(contrast["contrast_id"]) == 64
            and contrast.get("parent_configuration_hash")
            == EXPECTED_PARENT_TYPED_SHA256
            and contrast.get("child_configuration_hash")
            == candidate.occurrence.configuration_hash
            and len(patch_hashes) == 1
            and contrast.get("derived_patch_hash") == patch_hashes[0]
            and contrast.get("changed_paths") == [_path_text(index)]
            and contrast.get("patch_operation_count") == 1
            and contrast.get("contrast_scope") == "single_operation"
            and operation.get("operation_kind") == "replace_scalar"
            and operation.get("path") == _path_text(index)
            and operation.get("old_value") == PARENT_C["sequence"][index]
            and operation.get("new_value") == replacement
            and operation.get("old_value_hash")
            == typed_json_sha256(freeze_json(PARENT_C["sequence"][index]))
            and operation.get("new_value_hash")
            == typed_json_sha256(freeze_json(replacement))
        ):
            return False
    return True


def _citation_lineage_gate(
    entries: Sequence[InsightMemoryEntry],
    rows: Sequence[Mapping[str, object]],
) -> bool:
    contrast_lineage: dict[str, tuple[str, frozenset[str]]] = {}
    for row in rows:
        contrasts = row["machine_derived_contrasts"]
        for contrast in contrasts:
            contrast_lineage[str(contrast["contrast_id"])] = (
                str(row["operator_invocation_id"]),
                frozenset(
                    (
                        str(contrast["parent_candidate_id"]),
                        str(contrast["child_candidate_id"]),
                    )
                ),
            )
    available = frozenset(contrast_lineage)
    if not entries or len(available) != len(MUTATION_INDICES):
        return False
    for entry in entries:
        lineage = entry.evidence_lineage
        if lineage is None:
            return False
        cited = frozenset(entry.draft.evidence_contrast_ids)
        if not cited or not cited <= available:
            return False
        expected_operators = frozenset(contrast_lineage[value][0] for value in cited)
        expected_candidates = frozenset(
            candidate_id
            for value in cited
            for candidate_id in contrast_lineage[value][1]
        )
        if (
            frozenset(lineage.available_contrast_ids) != available
            or frozenset(lineage.cited_contrast_ids) != cited
            or frozenset(
                value.value for value in lineage.source_operator_invocation_ids
            )
            != expected_operators
            or frozenset(value.value for value in lineage.source_candidate_ids)
            != expected_candidates
        ):
            return False
    return True


async def run_workflow(
    *,
    problem: Any,
    generator: Any,
    id_seed: int,
    event_writer: v1.DurableJsonlWriter,
    evaluator_concurrency: int,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    temperature: float = TEMPERATURE,
) -> dict[str, object]:
    """Execute the frozen v2 workflow against injected live or offline ports."""

    ids = DeterministicIdFactory(f"boils_patch_native_pilot_v2_{id_seed}")
    memory = InsightMemoryBank(id_factory=ids)
    trace = v1.TraceRecorder(event_writer)
    engine = AgenticEvolutionEngine(
        problem=problem,
        generator=generator,
        id_factory=ids,
        memory=memory,
        seed=id_seed,
        evaluator_concurrency=evaluator_concurrency,
        trace_sink=trace.emit,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    archive = ParetoArchive(engine.objectives)

    # This is intentionally the sole evaluation before the fixed paid batch.
    parent = await engine.register_seed(copy.deepcopy(PARENT_C), label="parent_c")
    if (
        not parent.valid
        or parent.configuration_dict != PARENT_C
        or config_sha256(parent.configuration_dict) != EXPECTED_PARENT_BOILS_SHA256
        or parent.occurrence.configuration_hash != EXPECTED_PARENT_TYPED_SHA256
        or parent.objective_map != EXPECTED_SEED_OBJECTIVES
    ):
        raise RuntimeError(
            "frozen parent C failed identity, validity, or exact-objective preflight"
        )
    v1._consider(archive, parent, trace)
    trace.emit(
        {
            "event_type": "paid_batch_seed_gate_passed",
            "candidate_id": parent.candidate_id.value,
            "boils_configuration_sha256": EXPECTED_PARENT_BOILS_SHA256,
            "typed_json_configuration_sha256": EXPECTED_PARENT_TYPED_SHA256,
            "objectives": parent.objective_map,
        }
    )

    plans = tuple(
        InvocationPlan(
            OperatorKind.TYPED_MUTATION,
            (parent,),
            generation=1,
            label=f"g1_patch_native_sequence_index_{index}",
            allowed_top_level=("sequence",),
            phase="boils_v2_patch_native_fixed_block",
            mutation_contract=_atomic_contract(index),
            mutation_response_mode=(
                MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1
            ),
        )
        for index in MUTATION_INDICES
    )
    outcomes = await engine.run_invocations(plans)
    if tuple(outcome.prepared.plan.label for outcome in outcomes) != tuple(
        plan.label for plan in plans
    ):
        raise RuntimeError("variation completion altered frozen report order")

    legal_matches: list[dict[str, object] | None] = []
    for index, outcome in zip(MUTATION_INDICES, outcomes, strict=True):
        candidate = outcome.candidate
        legal = None if candidate is None else _legal_child_match(index, candidate)
        legal_matches.append(legal)
        # Completion order never controls this consideration order.
        v1._consider(archive, candidate, trace)

    reflection_skipped = any(outcome.candidate is None for outcome in outcomes)
    reflection_failure_type: str | None = None
    reflected: tuple[InsightMemoryEntry, ...] = ()
    if reflection_skipped:
        trace.emit(
            {
                "event_type": "reflection_skipped",
                "label": "boils_v2_fixed_block_reflection",
                "reason": "provider_or_patch_boundary_candidate_missing",
                "missing_indices": [
                    index
                    for index, outcome in zip(MUTATION_INDICES, outcomes, strict=True)
                    if outcome.candidate is None
                ],
            }
        )
    else:
        try:
            reflected = await engine.reflect(
                outcomes,
                label="boils_v2_fixed_block_reflection",
                max_insights=3,
            )
        except Exception as exc:  # The one frozen call remains a negative result.
            reflection_failure_type = type(exc).__name__

    snapshot = archive.snapshot()
    trace.emit(snapshot.to_trace_record())
    cache_snapshot = await engine.evaluation_cache_snapshot()
    executed_calls = EXPECTED_VARIATION_CALLS + (0 if reflection_skipped else 1)
    provider_calls = _call_summary(
        trace.events,
        expected_executed_calls=executed_calls,
    )
    atomic_events = _atomic_event_map(trace.events)
    atomic_trace_gates: list[bool] = []
    for index, outcome, legal in zip(
        MUTATION_INDICES, outcomes, legal_matches, strict=True
    ):
        candidate = outcome.candidate
        atomic_trace_gates.append(
            bool(
                candidate is not None
                and legal is not None
                and candidate.candidate_id.value in atomic_events
                and _atomic_trace_gate(
                    index=index,
                    candidate=candidate,
                    event=atomic_events[candidate.candidate_id.value],
                    legal=legal,
                )
            )
        )
    reflection_rows = _reflection_rows(trace.events)
    reflected_entries = tuple(reflected)
    successful_calls = int(provider_calls["successful_logical_calls"])
    resolved_providers = set(dict(provider_calls["resolved_providers"]))
    cost = Decimal(str(provider_calls["cost_usd_successful_responses"]))
    no_credit_events = not any(
        event.get("event_type")
        in {
            "insight_credit_updated",
            "insight_credit_censored",
            "insight_lifecycle_transition",
        }
        for event in trace.events
    )
    no_filter_events = not any(
        event.get("event_type")
        == "reflection_evidence_contrast_ids_filtered"
        for event in trace.events
    )
    child_hashes = [
        outcome.candidate.occurrence.configuration_hash
        for outcome in outcomes
        if outcome.candidate is not None
    ]
    gates: dict[str, bool] = {
        "seed_gate_passed_before_proposals": True,
        "all_four_variation_calls_succeeded": (
            all(outcome.candidate is not None for outcome in outcomes)
            and int(provider_calls["failed_logical_calls"]) == 0
        ),
        "all_calls_requested_exact_model": dict(provider_calls["requested_models"])
        == {MODEL: successful_calls},
        "all_calls_resolved_exact_model": dict(provider_calls["resolved_models"])
        == {MODEL: successful_calls},
        "resolved_providers_within_frozen_set": bool(resolved_providers)
        and resolved_providers <= ALLOWED_RESOLVED_PROVIDERS,
        "successful_response_cost_within_budget": cost
        <= MAX_SUCCESSFUL_RESPONSE_COST_USD,
        "all_four_children_distinct": len(child_hashes) == 4
        and len(set(child_hashes)) == 4,
        "all_children_in_preregistered_legal_universe": all(
            legal is not None for legal in legal_matches
        ),
        "all_atomic_traces_exact_and_system_attributed": all(atomic_trace_gates),
        "archive_considered_parent_and_fixed_children_once": (
            snapshot.consideration_count
            == 1 + sum(outcome.candidate is not None for outcome in outcomes)
        ),
        "reflection_was_not_skipped": not reflection_skipped,
        "reflection_call_succeeded": reflection_failure_type is None
        and not reflection_skipped,
        "reflection_created_at_least_one_entry": bool(reflected_entries),
        "reflection_machine_facts_complete": _reflection_machine_fact_gate(
            reflection_rows, outcomes
        ),
        "reflection_citations_exact_and_lineage_narrow": _citation_lineage_gate(
            reflected_entries, reflection_rows
        ),
        "reflection_had_no_filtered_ids": no_filter_events,
        "reflection_entries_all_quarantined": bool(reflected_entries)
        and all(
            entry.lifecycle_state is InsightLifecycleState.QUARANTINED
            for entry in reflected_entries
        ),
        "reflection_entries_all_nonretrievable": bool(reflected_entries)
        and all(not entry.retrievable for entry in reflected_entries),
        "no_memory_trials_or_transitions": not memory.trials
        and not memory.transitions
        and no_credit_events,
    }
    acceptance_passed = all(gates.values()) and executed_calls == EXPECTED_TOTAL_CALLS

    return {
        "schema_version": 1,
        "status": "succeeded",
        "completed_at_utc": _utc_now(),
        "development_only": True,
        "acceptance_passed": acceptance_passed,
        "claim_boundary": (
            "One frozen four-call BOiLS mechanism block: patch-native intent/edit "
            "identity and exact structured reflection citations only. No memory "
            "utility, optimizer, genericity, SOTA, or wall-clock claim."
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
        "parent_c": _candidate_record(parent),
        "fixed_mutation_order": list(MUTATION_INDICES),
        "generation_one_patch_native": [
            {
                **v1._outcome_record(outcome),
                "mutation_response_mode": (
                    outcome.prepared.plan.mutation_response_mode.value
                ),
                "legal_child": legal,
                "atomic_trace_gate": atomic_gate,
            }
            for outcome, legal, atomic_gate in zip(
                outcomes, legal_matches, atomic_trace_gates, strict=True
            )
        ],
        "reflection": {
            "attempted": not reflection_skipped,
            "skipped": reflection_skipped,
            "failure_type": reflection_failure_type,
            "max_insights": 3,
            "entries": [_reflection_record(entry) for entry in reflected_entries],
            "machine_derived_rows": reflection_rows,
        },
        "memory": {
            "entry_count": len(memory.entries),
            "trial_count": len(memory.trials),
            "lifecycle_transition_count": len(memory.transitions),
        },
        "pareto_archive": snapshot.to_trace_record(),
        "pareto_front": [
            _candidate_record(candidate) for candidate in snapshot.front_candidates
        ],
        "evaluation_cache": cache_snapshot,
        "provider_calls": provider_calls,
        "factorial_replay": factorial_replay(),
        "counts": {
            "seed_evaluations_requested": 1,
            "variation_invocations": len(outcomes),
            "reflection_calls_attempted": int(not reflection_skipped),
            "valid_variation_candidates": sum(
                outcome.candidate is not None and outcome.candidate.valid
                for outcome in outcomes
            ),
        },
        "gates": gates,
    }


async def _run_live_pilot(
    *,
    args: argparse.Namespace,
    problem: BoilsAbcProblem,
    event_writer: v1.DurableJsonlWriter,
    queue_writer: v1.DurableJsonlWriter,
) -> dict[str, object]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unavailable")
    structured = PydanticAIStructuredGenerator.openrouter(
        api_key=api_key,
        model_name=MODEL,
        max_connections=QUEUE_MAX_IN_FLIGHT,
        timeout_seconds=float(QUEUE_ATTEMPT_TIMEOUT_SECONDS),
        provider_options={
            "order": list(PROVIDER_ORDER),
            "allow_fallbacks": False,
        },
        app_title="AgentEvolve AAAI 2027 BOiLS patch-native pilot v2",
    )
    runner = create_production_queued_runner(
        generator=structured,
        max_in_flight=QUEUE_MAX_IN_FLIGHT,
        max_pending=QUEUE_MAX_PENDING,
        max_attempts=QUEUE_MAX_ATTEMPTS,
        attempt_timeout_ns=QUEUE_ATTEMPT_TIMEOUT_SECONDS * 1_000_000_000,
        base_backoff_ns=QUEUE_BASE_BACKOFF_SECONDS * 1_000_000_000,
        max_backoff_ns=QUEUE_MAX_BACKOFF_SECONDS * 1_000_000_000,
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
        )


def _assert_evaluator_provenance(evaluator: BoilsAbcEvaluator) -> None:
    provenance = evaluator.provenance()
    if provenance.get("abc_binary_sha256") != EXPECTED_ABC_SHA256:
        raise RuntimeError("ABC executable does not match the frozen v2 plan")
    circuits = provenance.get("circuits")
    if not (
        type(circuits) is list
        and len(circuits) == 1
        and circuits[0].get("name") == "log2"
        and circuits[0].get("sha256") == EXPECTED_CIRCUIT_SHA256
    ):
        raise RuntimeError("log2 circuit does not match the frozen v2 plan")
    if provenance.get("lut_inputs") != 6:
        raise RuntimeError("v2 is frozen to LUT-6 mapping")
    if provenance.get("affinity_sets") != [[cpu] for cpu in PILOT_CPUS]:
        raise RuntimeError("evaluator affinity leases do not match the frozen v2 plan")


def _source_hashes() -> dict[str, str]:
    sources = {
        "runner": Path(__file__).resolve(),
        "legal_children": LEGAL_CHILD_PATH,
        "v1_durable_helpers": Path(v1.__file__).resolve(),
        "actions": AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/actions.py",
        "evaluator": AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/evaluator.py",
        "problem": AGENT_EVOLVE_ROOT / "examples/benchmarks/boils_abc/problem_def.py",
        "engine": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/agentic_evolution.py",
        "memory": AGENT_EVOLVE_ROOT / "src/agent_evolve/application/insight_memory.py",
        "pareto_archive": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/application/pareto_archive.py",
        "generator_port": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/ports/agentic_generator.py",
        "queue": AGENT_EVOLVE_ROOT / "src/agent_evolve/application/llm_task_queue.py",
        "backoff": AGENT_EVOLVE_ROOT / "src/agent_evolve/policies/llm_backoff.py",
        "agentic_adapter": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/agentic_generator.py",
        "provider_adapter": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/async_generator.py",
        "queued_runner": AGENT_EVOLVE_ROOT
        / "src/agent_evolve/integrations/pydantic_ai/queued_runner.py",
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
        "development_only": True,
        "claim_boundary": (
            "BOiLS/log2 patch-native mechanism experiment only; no memory utility, "
            "optimizer, genericity, SOTA, or wall-clock claim."
        ),
        "frozen_workflow": {
            "parent_c": copy.deepcopy(PARENT_C),
            "parent_boils_configuration_sha256": EXPECTED_PARENT_BOILS_SHA256,
            "parent_typed_json_configuration_sha256": EXPECTED_PARENT_TYPED_SHA256,
            "required_parent_objectives": dict(EXPECTED_SEED_OBJECTIVES),
            "mutation_indices_zero_based": list(MUTATION_INDICES),
            "mutation_response_mode": (
                MutationResponseMode.ATOMIC_SCALAR_REPLACEMENT_V1.value
            ),
            "variation_calls_concurrent": EXPECTED_VARIATION_CALLS,
            "fixed_archive_and_report_order": list(MUTATION_INDICES),
            "reflection_max_insights": 3,
            "expected_total_logical_calls": EXPECTED_TOTAL_CALLS,
            "reflection_skip_rule": (
                "skip exactly when at least one fixed-block outcome has candidate None"
            ),
            "factorial_replay": factorial_replay(),
        },
        "legal_child_universe": {
            "source": str(LEGAL_CHILD_PATH),
            "sha256": EXPECTED_LEGAL_FILE_SHA256,
            "row_count": 40,
            "document": copy.deepcopy(LEGAL_CHILD_UNIVERSE),
        },
        "task": {
            "circuits": list(PILOT_CIRCUITS),
            "circuit_sha256": EXPECTED_CIRCUIT_SHA256,
            "abc_sha256": EXPECTED_ABC_SHA256,
            "sequence_length": SEQUENCE_LENGTH,
            "allowed_actions": list(ACTION_IDS),
            "raw_objectives": ["total_lut_count", "total_levels"],
            "mapping": "LUT-6 followed by mandatory CEC",
        },
        "evaluator_provenance": evaluator.provenance(),
        "model": MODEL,
        "provider": "openrouter",
        "provider_options": {
            "order": list(PROVIDER_ORDER),
            "allow_fallbacks": False,
        },
        "queue": {
            "enabled": True,
            "max_in_flight": QUEUE_MAX_IN_FLIGHT,
            "max_pending": QUEUE_MAX_PENDING,
            "max_attempts": QUEUE_MAX_ATTEMPTS,
            "attempt_timeout_ns": QUEUE_ATTEMPT_TIMEOUT_SECONDS * 1_000_000_000,
            "base_backoff_ns": QUEUE_BASE_BACKOFF_SECONDS * 1_000_000_000,
            "max_backoff_ns": QUEUE_MAX_BACKOFF_SECONDS * 1_000_000_000,
            "retry_owner": "AsyncLLMTaskQueue",
            "sdk_retries": 0,
            "pydantic_ai_retries": 0,
        },
        "budget": {
            "successful_response_cost_usd": str(MAX_SUCCESSFUL_RESPONSE_COST_USD),
            "missingness_policy": "record failure; never selectively cancel or rerun an arm",
        },
        "seed": args.seed,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
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


def _finalize(run_dir: Path, status: str) -> None:
    names = (
        "manifest.json",
        "runner_source.py",
        "legal_children.json",
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
            "sha256": _sha256_bytes(payload),
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--cpus", default=",".join(str(cpu) for cpu in PILOT_CPUS))
    parser.add_argument("--max-in-flight", type=int, default=QUEUE_MAX_IN_FLIGHT)
    parser.add_argument("--max-pending", type=int, default=QUEUE_MAX_PENDING)
    parser.add_argument("--max-attempts", type=int, default=QUEUE_MAX_ATTEMPTS)
    parser.add_argument(
        "--attempt-timeout-seconds",
        type=int,
        default=QUEUE_ATTEMPT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--base-backoff-seconds",
        type=int,
        default=QUEUE_BASE_BACKOFF_SECONDS,
    )
    parser.add_argument(
        "--max-backoff-seconds",
        type=int,
        default=QUEUE_MAX_BACKOFF_SECONDS,
    )
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    return parser


def _assert_frozen_cli(args: argparse.Namespace) -> None:
    expected = {
        "model": MODEL,
        "cpus": ",".join(str(cpu) for cpu in PILOT_CPUS),
        "max_in_flight": QUEUE_MAX_IN_FLIGHT,
        "max_pending": QUEUE_MAX_PENDING,
        "max_attempts": QUEUE_MAX_ATTEMPTS,
        "attempt_timeout_seconds": QUEUE_ATTEMPT_TIMEOUT_SECONDS,
        "base_backoff_seconds": QUEUE_BASE_BACKOFF_SECONDS,
        "max_backoff_seconds": QUEUE_MAX_BACKOFF_SECONDS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise SystemExit(f"BOiLS patch-native pilot v2 freezes --{name.replace('_', '-')}={value}")
    if args.seed < 0:
        raise SystemExit("seed must be non-negative")


def main() -> None:
    args = _parser().parse_args()
    _assert_frozen_cli(args)
    if _sha256_bytes(LEGAL_CHILD_PATH.read_bytes()) != EXPECTED_LEGAL_FILE_SHA256:
        raise SystemExit("frozen legal-child source changed after import")

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "boils_patch_native_pilot_v2_%Y%m%dT%H%M%SZ"
    )
    run_dir = args.log_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__).resolve(), run_dir / "runner_source.py")
    shutil.copyfile(LEGAL_CHILD_PATH, run_dir / "legal_children.json")
    if support._sha256(run_dir / "legal_children.json") != EXPECTED_LEGAL_FILE_SHA256:
        raise RuntimeError("durable legal-child copy failed its hash gate")

    event_writer = v1.DurableJsonlWriter(run_dir / "events.jsonl")
    evaluation_writer = v1.DurableJsonlWriter(run_dir / "evaluations.jsonl")
    queue_writer = v1.DurableJsonlWriter(run_dir / "queue_outcomes.jsonl")
    load_dotenv(WORKSPACE_ROOT / ".env", override=False)
    started_ns = time.perf_counter_ns()
    status = "failed"
    try:
        settings = AbcEvaluatorSettings.current_circuit_panel(
            circuit_names=PILOT_CIRCUITS,
            affinity_sets=tuple((cpu,) for cpu in PILOT_CPUS),
            per_circuit_timeout_s=60.0,
        )
        observation_recorder = v1.EvaluationObservationRecorder(evaluation_writer)
        evaluator = BoilsAbcEvaluator(settings, observer=observation_recorder)
        _assert_evaluator_provenance(evaluator)
        problem = BoilsAbcProblem(settings, evaluator=evaluator)
        support._write_json(
            run_dir / "manifest.json",
            _manifest(args, run_id=run_id, evaluator=evaluator),
        )
        summary = asyncio.run(
            _run_live_pilot(
                args=args,
                problem=problem,
                event_writer=event_writer,
                queue_writer=queue_writer,
            )
        )
        summary["runner_elapsed_ns"] = time.perf_counter_ns() - started_ns
        summary["evaluator_observations"] = v1._evaluation_log_summary(
            run_dir / "evaluations.jsonl"
        )
        queue_writer.close()
        queue_summary = support._queue_log_summary(run_dir / "queue_outcomes.jsonl")
        expected_terminal = int(summary["provider_calls"]["executed_logical_calls"])
        if queue_summary["terminal_outcomes"] != expected_terminal:
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
                    else "BOiLS patch-native pilot v2 failed; inspect sanitized traces"
                ),
            },
        )
        raise
    finally:
        queue_writer.close()
        event_writer.close()
        evaluation_writer.close()
        _finalize(run_dir, status)

    print(_canonical_json({"run_dir": str(run_dir), "status": status}))


if __name__ == "__main__":
    main()
