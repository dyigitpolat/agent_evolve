#!/usr/bin/env python3
"""Evaluate the unselected half of a finalized BOiLS calibrated K8 slate.

This is a post-hoc mechanism diagnostic, not a campaign counterfactual.  It
holds each recorded parent and pre-wave archive cutoff fixed, evaluates only
the four previously unselected proposals, and compares equal-size K4
allocations within the exact model-proposed K8 support.  No provider, API key,
memory update, parent transition, or source-run file is touched.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from agent_evolve.core.problem import ObjectiveSpec  # noqa: E402
from agent_evolve.domain.typed_json import freeze_json, thaw_json  # noqa: E402
from agent_evolve.domain.typed_json import typed_json_sha256  # noqa: E402
from agent_evolve.policies.reward.frozen_wave_archive import (  # noqa: E402
    FrozenArchiveWaveSnapshot2D,
)
from agent_evolve.ports.variation_catalog import (  # noqa: E402
    bind_finite_variation_catalog,
)
from examples.benchmarks.boils_abc.evaluator import (  # noqa: E402
    AbcEvaluatorSettings,
    BoilsAbcEvaluator,
)
from examples.benchmarks.boils_abc.actions import config_sha256  # noqa: E402
from examples.benchmarks.boils_abc.finite_variation_catalog import (  # noqa: E402
    BoilsFiniteVariationCatalog,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    finalize_run_directory,
    read_jsonl,
    source_identity,
    verify_finalized_run_directory,
    write_json_atomic,
)


DEFAULT_SOURCE_RUN = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "boils_abc/portfolio_q/boilsq_calibrated_g6_live_deepseek_v4_20260716"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "boils_abc/portfolio_q"
)
OBJECTIVES = (
    ObjectiveSpec("total_lut_count", "min"),
    ObjectiveSpec("total_levels", "min"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _point_from_record(value: object) -> dict[str, float]:
    if type(value) is not list:
        raise TypeError("objective point record must be a list")
    result: dict[str, float] = {}
    for item in value:
        if type(item) is not list or len(item) != 2:
            raise TypeError("objective point cells must be two-item lists")
        name, encoded = item
        if type(name) is not str or type(encoded) is not str:
            raise TypeError("objective point cells must contain strings")
        result[name] = float.fromhex(encoded)
    return result


def _snapshot(value: dict[str, Any]) -> FrozenArchiveWaveSnapshot2D:
    snapshot = FrozenArchiveWaveSnapshot2D.create(
        objectives=OBJECTIVES,
        reference_point=_point_from_record(value["reference_point"]),
        archive_points=tuple(
            _point_from_record(point) for point in value["archive_points"]
        ),
    )
    if snapshot.snapshot_hash != value["snapshot_hash"]:
        raise RuntimeError("reconstructed archive snapshot differs from source trace")
    return snapshot


def _candidate_index(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = summary.get("candidates")
    if type(candidates) is not list:
        raise RuntimeError("source summary omitted candidates")
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if type(candidate) is not dict or type(candidate.get("candidate_id")) is not str:
            raise RuntimeError("source summary contains a malformed candidate")
        candidate_id = candidate["candidate_id"]
        if candidate_id in by_id:
            raise RuntimeError("source summary repeats a candidate ID")
        by_id[candidate_id] = candidate
    return by_id


def _build_legacy_plan(
    *,
    source_run: Path,
    summary: dict[str, Any],
    wave_rows: tuple[dict[str, object], ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audits = summary.get("portfolio_selection_audits")
    if type(audits) is not list or len(audits) != 6:
        raise RuntimeError("expected exactly six calibrated selector audits")
    candidates = _candidate_index(summary)
    wave_by_request: dict[str, dict[str, Any]] = {}
    for row in wave_rows:
        binding = row.get("calibrated_input_binding")
        if type(binding) is not dict or type(binding.get("request_sha256")) is not str:
            raise RuntimeError("wave trace omitted calibrated request identity")
        request_sha256 = binding["request_sha256"]
        if request_sha256 in wave_by_request:
            raise RuntimeError("wave trace repeats a selector request")
        wave_by_request[request_sha256] = dict(row)

    evaluation_plan: list[dict[str, Any]] = []
    waves: list[dict[str, Any]] = []
    seen_configurations: set[str] = set()
    for audit_ordinal, audit in enumerate(audits, start=1):
        if type(audit) is not dict:
            raise RuntimeError("selector audit must be an object")
        request_sha256 = audit.get("wave_request_sha256")
        if type(request_sha256) is not str:
            raise RuntimeError("selector audit omitted request identity")
        trace = wave_by_request.get(request_sha256)
        if trace is None:
            raise RuntimeError("selector audit has no exact wave trace")
        parent_id = trace.get("parent_candidate_id")
        if type(parent_id) is not str or parent_id not in candidates:
            raise RuntimeError("wave trace names an unknown parent")
        parent_record = candidates[parent_id]
        parent_configuration = parent_record.get("configuration")
        parent_frozen = freeze_json(parent_configuration)
        contract = bind_finite_variation_catalog(
            BoilsFiniteVariationCatalog(),
            parent_frozen,
        )

        decision = audit.get("decision")
        if type(decision) is not dict:
            raise RuntimeError("selector audit omitted decision")
        supplemental = decision.get("supplemental_selector_audit")
        if type(supplemental) is not dict:
            raise RuntimeError("selector audit omitted calibrated evidence")
        payload = supplemental.get("payload")
        allocation = None if type(payload) is not dict else payload.get("allocation")
        request = None if type(allocation) is not dict else allocation.get("request")
        slate = None if type(request) is not dict else request.get("slate")
        members = None if type(slate) is not dict else slate.get("members")
        selected = None if type(allocation) is not dict else allocation.get("selected")
        if type(members) is not list or len(members) != 8:
            raise RuntimeError("calibrated audit must contain exactly eight members")
        if type(selected) is not list or len(selected) != 4:
            raise RuntimeError("calibrated audit must contain exactly four selections")
        selected_ids = {item["option_id"] for item in selected}

        feedback = audit.get("outcome_feedback")
        actions = None if type(feedback) is not dict else feedback.get("actions")
        if type(actions) is not list or len(actions) != 4:
            raise RuntimeError("selector audit omitted complete selected outcomes")
        selected_outcomes: dict[str, dict[str, float]] = {}
        for action in actions:
            if type(action) is not dict:
                raise RuntimeError("feedback action must be an object")
            child = candidates.get(action.get("candidate_id"))
            if child is None or type(child.get("objectives")) is not dict:
                raise RuntimeError("feedback action has no exact source candidate")
            selected_outcomes[action["option_id"]] = {
                name: float(value) for name, value in child["objectives"].items()
            }
        if set(selected_outcomes) != selected_ids:
            raise RuntimeError("selected outcomes differ from calibrated selection")

        ordered: list[dict[str, Any]] = []
        for member in sorted(members, key=lambda item: item["model_rank"]):
            option_id = member["option_id"]
            option = contract.resolve(option_id)
            child_sha256 = option.child_configuration_sha256
            if child_sha256 != member["phenotype_identity_sha256"]:
                raise RuntimeError("current catalog differs from recorded phenotype")
            if child_sha256 in seen_configurations:
                raise RuntimeError("K8 diagnostic plan repeats a child configuration")
            seen_configurations.add(child_sha256)
            row = {
                "wave_ordinal": audit_ordinal,
                "cycle": audit["cycle"],
                "request_sha256": request_sha256,
                "parent_candidate_id": parent_id,
                "model_rank": member["model_rank"],
                "option_id": option_id,
                "option_identity_sha256": member["option_identity_sha256"],
                "configuration_sha256": child_sha256,
                "configuration": thaw_json(option.child_configuration),
                "calibrated_selected": option_id in selected_ids,
                "source_objectives": selected_outcomes.get(option_id),
            }
            row["evaluator_configuration_sha256"] = config_sha256(
                row["configuration"]
            )
            ordered.append(row)
            if option_id not in selected_ids:
                evaluation_plan.append(row)
        reward_snapshot = trace.get("archive_reward_snapshot")
        if type(reward_snapshot) is not dict:
            raise RuntimeError("wave trace omitted frozen archive reward snapshot")
        _snapshot(reward_snapshot)
        waves.append(
            {
                "wave_ordinal": audit_ordinal,
                "cycle": audit["cycle"],
                "request_sha256": request_sha256,
                "parent_candidate_id": parent_id,
                "archive_reward_snapshot": reward_snapshot,
                "members": ordered,
            }
        )
    if len(evaluation_plan) != 24 or len(seen_configurations) != 48:
        raise RuntimeError("expected 24 fresh rejected actions in 48 unique K8 actions")
    source_hashes = {
        candidate["configuration_sha256"] for candidate in candidates.values()
    }
    if any(row["configuration_sha256"] in source_hashes for row in evaluation_plan):
        raise RuntimeError("a rejected proposal was already evaluated in the source run")
    return evaluation_plan, waves


def _objective_map_from_campaign_candidate(
    candidate: dict[str, Any],
) -> dict[str, float]:
    values = candidate.get("objectives")
    if type(values) is not list:
        raise RuntimeError("campaign candidate omitted objective observations")
    result: dict[str, float] = {}
    for value in values:
        if type(value) is not dict:
            raise RuntimeError("campaign candidate objective must be an object")
        metric_id = value.get("metric_id")
        value_hex = value.get("value_hex")
        if type(metric_id) is not str or type(value_hex) is not str:
            raise RuntimeError("campaign candidate objective is malformed")
        result[metric_id] = float.fromhex(value_hex)
    expected = {objective.name for objective in OBJECTIVES}
    if set(result) != expected:
        raise RuntimeError("campaign candidate objective set differs from protocol")
    return result


def _current_source_observations(
    rows: tuple[dict[str, object], ...],
) -> dict[str, dict[str, Any]]:
    """Index real source outcomes by AgentEvolve's typed configuration hash."""

    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        evaluation = row.get("evaluation")
        if type(evaluation) is not dict:
            raise RuntimeError("real evaluator observation omitted evaluation")
        sequence = evaluation.get("sequence")
        if type(sequence) is not list:
            raise RuntimeError("real evaluator observation omitted BOiLS sequence")
        configuration = {"sequence": sequence}
        frozen = freeze_json(configuration)
        configuration_sha256 = typed_json_sha256(frozen)
        objectives = {
            "total_lut_count": float(evaluation["total_lut_count"]),
            "total_levels": float(evaluation["total_levels"]),
        }
        existing = indexed.get(configuration_sha256)
        record = {
            "configuration": configuration,
            "objectives": objectives,
            "evaluation": evaluation,
        }
        if existing is not None and existing != record:
            raise RuntimeError("source observations disagree for one phenotype")
        indexed[configuration_sha256] = record
    return indexed


def _current_reference_point(manifest: dict[str, Any]) -> dict[str, float]:
    utility = manifest.get("utility")
    axes = None if type(utility) is not dict else utility.get("axes")
    if type(axes) is not list:
        raise RuntimeError("campaign manifest omitted fixed utility axes")
    result: dict[str, float] = {}
    for axis in axes:
        if type(axis) is not dict:
            raise RuntimeError("campaign utility axis must be an object")
        metric_id = axis.get("metric_id")
        reference_hex = axis.get("reference_hex")
        if type(metric_id) is not str or type(reference_hex) is not str:
            raise RuntimeError("campaign utility axis is malformed")
        result[metric_id] = float.fromhex(reference_hex)
    if set(result) != {objective.name for objective in OBJECTIVES}:
        raise RuntimeError("campaign utility axes differ from diagnostic objectives")
    return result


def _authenticated_campaign_events(
    rows: tuple[dict[str, object], ...],
) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = []
    for row in rows:
        event = row.get("authenticated_campaign_event")
        if type(event) is not dict:
            raise RuntimeError("campaign journal row omitted authenticated event")
        events.append(event)
    return tuple(events)


def _build_current_campaign_plan(
    *,
    manifest: dict[str, Any],
    campaign_rows: tuple[dict[str, object], ...],
    output_rows: tuple[dict[str, object], ...],
    observation_rows: tuple[dict[str, object], ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct K8 supports from the authenticated generic campaign trace."""

    events = _authenticated_campaign_events(campaign_rows)
    stage_by_generation: dict[int, dict[str, Any]] = {}
    cutoff_by_generation: dict[int, dict[str, Any]] = {}
    for event in events:
        kind = event.get("kind")
        payload = event.get("payload")
        if type(payload) is not dict:
            raise RuntimeError("authenticated campaign event omitted payload")
        if kind == "stage_sealed":
            receipt = payload.get("stage_receipt")
            if type(receipt) is not dict or type(receipt.get("generation")) is not int:
                raise RuntimeError("sealed stage omitted generation receipt")
            generation = receipt["generation"]
            if generation in stage_by_generation:
                raise RuntimeError("campaign repeats a sealed generation")
            stage_by_generation[generation] = receipt
        elif kind == "archive_utility_frozen":
            cutoff = payload.get("archive_cutoff")
            if type(cutoff) is not dict or type(cutoff.get("generation")) is not int:
                raise RuntimeError("archive cutoff omitted generation")
            generation = cutoff["generation"]
            if generation in cutoff_by_generation:
                raise RuntimeError("campaign repeats a generation archive cutoff")
            cutoff_by_generation[generation] = cutoff

    output_by_call: dict[str, dict[str, Any]] = {}
    for row in output_rows:
        record = row.get("authenticated_record")
        if type(record) is not dict or record.get("operation") != "select_portfolio":
            continue
        call_id = record.get("call_id")
        typed_output = record.get("typed_output")
        if type(call_id) is not str or type(typed_output) is not dict:
            raise RuntimeError("portfolio output evidence is malformed")
        if call_id in output_by_call:
            raise RuntimeError("portfolio output evidence repeats a call ID")
        output_by_call[call_id] = typed_output

    if set(stage_by_generation) != set(range(1, 7)):
        raise RuntimeError("current diagnostic requires one complete G6 campaign")
    if set(cutoff_by_generation) != set(range(1, 7)):
        raise RuntimeError("current diagnostic requires six frozen archive cutoffs")
    if len(output_by_call) != 6:
        raise RuntimeError("expected six authenticated portfolio outputs")

    source_observations = _current_source_observations(observation_rows)
    if len(source_observations) != 38:
        raise RuntimeError("expected 38 unique real source observations")
    reference_point = _current_reference_point(manifest)
    evaluation_plan: list[dict[str, Any]] = []
    waves: list[dict[str, Any]] = []
    support_occurrence_count = 0
    unique_support_configurations: set[str] = set()
    selected_count = 0

    for generation in (1, 3, 5):
        stage = stage_by_generation[generation]
        result = stage.get("result")
        if type(result) is not dict:
            raise RuntimeError("sealed portfolio stage omitted result")
        candidates = result.get("candidates")
        receipts = result.get("portfolio_wave_receipts")
        if type(candidates) is not list or type(receipts) is not list or len(receipts) != 2:
            raise RuntimeError("portfolio stage must contain two complete lane receipts")
        candidate_by_id = {
            candidate["candidate_id"]: candidate
            for candidate in candidates
            if type(candidate) is dict and type(candidate.get("candidate_id")) is str
        }
        if len(candidate_by_id) != len(candidates):
            raise RuntimeError("portfolio stage contains malformed or duplicate candidates")

        cutoff = cutoff_by_generation[generation]
        archive = cutoff.get("archive")
        front = None if type(archive) is not dict else archive.get("front_candidates")
        if type(front) is not list:
            raise RuntimeError("frozen archive cutoff omitted front candidates")
        snapshot = FrozenArchiveWaveSnapshot2D.create(
            objectives=OBJECTIVES,
            reference_point=reference_point,
            archive_points=tuple(
                _objective_map_from_campaign_candidate(candidate)
                for candidate in front
            ),
        )
        reward_snapshot = snapshot.to_record()

        for receipt in receipts:
            if type(receipt) is not dict:
                raise RuntimeError("portfolio wave receipt must be an object")
            call_id = receipt.get("selection_call_id")
            request_sha256 = receipt.get("request_sha256")
            parent_id = receipt.get("parent_candidate_id")
            parent_sha256 = receipt.get("parent_configuration_sha256")
            if not all(
                type(value) is str
                for value in (call_id, request_sha256, parent_id, parent_sha256)
            ):
                raise RuntimeError("portfolio wave receipt identity is malformed")
            typed_output = output_by_call.get(call_id)
            members = None if typed_output is None else typed_output.get("members")
            if type(members) is not list or len(members) != 8:
                raise RuntimeError("portfolio output must contain exactly eight members")
            option_ids = [member.get("option_id") for member in members]
            if any(type(option_id) is not str for option_id in option_ids):
                raise RuntimeError("portfolio output member omitted option ID")
            if len(set(option_ids)) != 8:
                raise RuntimeError("portfolio output repeats an option ID")

            parent_observation = source_observations.get(parent_sha256)
            if parent_observation is None:
                raise RuntimeError("portfolio parent has no exact real source observation")
            parent_frozen = freeze_json(parent_observation["configuration"])
            if typed_json_sha256(parent_frozen) != parent_sha256:
                raise RuntimeError("reconstructed parent differs from campaign identity")
            contract = bind_finite_variation_catalog(
                BoilsFiniteVariationCatalog(),
                parent_frozen,
            )

            selected_members = receipt.get("members")
            if type(selected_members) is not list or len(selected_members) != 4:
                raise RuntimeError("portfolio receipt must contain four executed members")
            selected_by_option: dict[str, dict[str, Any]] = {}
            for selected in selected_members:
                if type(selected) is not dict:
                    raise RuntimeError("executed portfolio member must be an object")
                materialization = selected.get("materialization")
                if type(materialization) is not dict:
                    raise RuntimeError("executed member omitted materialization")
                option_id = materialization.get("option_id")
                if type(option_id) is not str or option_id in selected_by_option:
                    raise RuntimeError("executed portfolio option identity is invalid")
                selected_by_option[option_id] = selected
            if not set(selected_by_option).issubset(set(option_ids)):
                raise RuntimeError("executed option was absent from authenticated K8 output")
            selected_count += len(selected_by_option)

            ordered: list[dict[str, Any]] = []
            for model_rank, output_member in enumerate(members, start=1):
                if type(output_member) is not dict:
                    raise RuntimeError("portfolio output member must be an object")
                option_id = output_member["option_id"]
                option = contract.resolve(option_id)
                child_sha256 = option.child_configuration_sha256
                support_occurrence_count += 1
                unique_support_configurations.add(child_sha256)
                selected = selected_by_option.get(option_id)
                source_objectives = None
                if selected is not None:
                    materialization = selected["materialization"]
                    executed_portfolio_rank = materialization.get("rank")
                    if type(executed_portfolio_rank) is not int:
                        raise RuntimeError("executed member omitted portfolio rank")
                    if materialization.get("child_configuration_sha256") != child_sha256:
                        raise RuntimeError("executed child differs from current finite catalog")
                    if materialization.get("option_identity_sha256") != option.identity_sha256:
                        raise RuntimeError("executed option identity differs from catalog")
                    candidate = candidate_by_id.get(selected.get("candidate_id"))
                    if candidate is None:
                        raise RuntimeError("executed member has no sealed source candidate")
                    if candidate.get("configuration_sha256") != child_sha256:
                        raise RuntimeError("sealed source candidate differs from catalog child")
                    source_objectives = _objective_map_from_campaign_candidate(candidate)
                elif child_sha256 in source_observations:
                    # A rejected member may have been evaluated in a later source wave.
                    # Its deterministic real outcome remains valid for this fixed-parent
                    # local diagnostic and avoids a redundant simulator call.
                    source_objectives = source_observations[child_sha256]["objectives"]

                row = {
                    "wave_ordinal": len(waves) + 1,
                    "cycle": (generation + 1) // 2,
                    "generation": generation,
                    "request_sha256": request_sha256,
                    "selection_call_id": call_id,
                    "parent_candidate_id": parent_id,
                    "model_rank": model_rank,
                    "executed_portfolio_rank": (
                        None
                        if selected is None
                        else selected["materialization"]["rank"]
                    ),
                    "model_role_proposal": output_member.get("role_proposal"),
                    "effect_predictions": output_member.get("effect_predictions"),
                    "option_id": option_id,
                    "option_family": option.family,
                    "option_identity_sha256": option.identity_sha256,
                    "configuration_sha256": child_sha256,
                    "configuration": thaw_json(option.child_configuration),
                    "calibrated_selected": selected is not None,
                    "source_objectives": source_objectives,
                }
                row["evaluator_configuration_sha256"] = config_sha256(
                    row["configuration"]
                )
                ordered.append(row)
                if selected is None and source_objectives is None:
                    evaluation_plan.append(row)
            waves.append(
                {
                    "wave_ordinal": len(waves) + 1,
                    "cycle": (generation + 1) // 2,
                    "generation": generation,
                    "request_sha256": request_sha256,
                    "parent_candidate_id": parent_id,
                    "archive_reward_snapshot": reward_snapshot,
                    "members": ordered,
                }
            )

    if len(waves) != 6 or selected_count != 24:
        raise RuntimeError("expected six K8 waves and 24 executed members")
    if support_occurrence_count != 48:
        raise RuntimeError("expected 48 proposal occurrences across current K8 supports")
    if len(unique_support_configurations) < 24:
        raise RuntimeError("current K8 supports contain implausibly few unique children")
    if len(evaluation_plan) > 24:
        raise RuntimeError("current K8 completion exceeds 24 rejected members")
    return evaluation_plan, waves


async def _evaluate_rejected(
    plan: list[dict[str, Any]],
    *,
    affinity_sets: tuple[tuple[int, ...], ...],
) -> tuple[list[dict[str, Any]], float]:
    evaluator = BoilsAbcEvaluator(
        AbcEvaluatorSettings.current_circuit_panel(
            circuit_names=("log2",),
            affinity_sets=affinity_sets,
        )
    )
    started = time.perf_counter()

    async def evaluate(row: dict[str, Any]) -> dict[str, Any]:
        result = await evaluator.evaluate_async(row["configuration"])
        if (
            result.configuration_sha256
            != row["evaluator_configuration_sha256"]
        ):
            raise RuntimeError("evaluator returned a foreign configuration")
        return {
            "wave_ordinal": row["wave_ordinal"],
            "request_sha256": row["request_sha256"],
            "model_rank": row["model_rank"],
            "option_id": row["option_id"],
            "option_identity_sha256": row["option_identity_sha256"],
            "configuration_sha256": row["configuration_sha256"],
            "evaluator_configuration_sha256": row[
                "evaluator_configuration_sha256"
            ],
            "objectives": result.objective_values,
            "evaluation": result.as_dict(),
        }

    results = await asyncio.gather(*(evaluate(row) for row in plan))
    return list(results), time.perf_counter() - started


def _allocation_analysis(
    waves: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    new_outcomes = {
        (row["request_sha256"], row["option_id"]): row["objectives"]
        for row in evaluations
    }
    wave_results: list[dict[str, Any]] = []
    for wave in waves:
        request_sha256 = wave["request_sha256"]
        snapshot = _snapshot(wave["archive_reward_snapshot"])
        members = wave["members"]
        outcomes: dict[str, dict[str, float]] = {}
        for member in members:
            value = member["source_objectives"]
            if value is None:
                value = new_outcomes[(request_sha256, member["option_id"])]
            outcomes[member["option_id"]] = value
        calibrated = tuple(
            member["option_id"] for member in members if member["calibrated_selected"]
        )
        direct = tuple(member["option_id"] for member in members[:4])
        feasible_subset_values = wave.get("feasible_k4_option_id_sets")
        feasible_subsets: set[frozenset[str]] | None = None
        if feasible_subset_values is not None:
            if type(feasible_subset_values) is not list:
                raise RuntimeError("feasible K4 subsets must be an exact list")
            feasible_subsets = set()
            for value in feasible_subset_values:
                if (
                    type(value) is not list
                    or len(value) != 4
                    or any(type(option_id) is not str for option_id in value)
                ):
                    raise RuntimeError("feasible K4 subset row is malformed")
                feasible_subsets.add(frozenset(value))
            if not feasible_subsets:
                raise RuntimeError("K8 support contains no feasible K4 subset")
        subset_rows = []
        for subset in combinations(tuple(outcomes), 4):
            if (
                feasible_subsets is not None
                and frozenset(subset) not in feasible_subsets
            ):
                continue
            augmented = snapshot.augmented_hypervolume(
                tuple(outcomes[option_id] for option_id in subset)
            )
            subset_rows.append(
                {
                    "option_ids": list(subset),
                    "augmented_hypervolume": augmented,
                    "gain": max(0.0, augmented - snapshot.base_hypervolume),
                }
            )
        subset_rows.sort(key=lambda row: (-row["gain"], row["option_ids"]))
        gains = [row["gain"] for row in subset_rows]

        def describe(option_ids: tuple[str, ...]) -> dict[str, Any]:
            feasible = (
                True
                if feasible_subsets is None
                else frozenset(option_ids) in feasible_subsets
            )
            augmented = snapshot.augmented_hypervolume(
                tuple(outcomes[option_id] for option_id in option_ids)
            )
            gain = max(0.0, augmented - snapshot.base_hypervolume)
            better = sum(value > gain for value in gains)
            ties = sum(value == gain for value in gains)
            return {
                "option_ids": list(option_ids),
                "augmented_hypervolume": augmented,
                "gain": gain,
                "feasible": feasible,
                "rank_min": better + 1,
                "rank_max": better + ties,
                "strictly_better_than_uniform_fraction": (
                    sum(gain > value for value in gains) / len(gains)
                ),
            }

        wave_results.append(
            {
                "wave_ordinal": wave["wave_ordinal"],
                "cycle": wave["cycle"],
                "request_sha256": request_sha256,
                "parent_candidate_id": wave["parent_candidate_id"],
                "base_hypervolume": snapshot.base_hypervolume,
                "calibrated_k4": describe(calibrated),
                "direct_model_top4": describe(direct),
                "uniform_k4": {
                    "conditioned_on_hard_feasibility": feasible_subsets is not None,
                    "support_size": len(gains),
                    "expected_gain": fmean(gains),
                    "minimum_gain": min(gains),
                    "maximum_gain": max(gains),
                    "zero_gain_fraction": sum(value == 0.0 for value in gains)
                    / len(gains),
                },
                "oracle_k4": subset_rows[0],
                "all_k8_outcomes": [
                    {
                        "model_rank": member["model_rank"],
                        "option_id": member["option_id"],
                        "calibrated_selected": member["calibrated_selected"],
                        "outcomes": outcomes[member["option_id"]],
                    }
                    for member in members
                ],
            }
        )
    calibrated_gains = [row["calibrated_k4"]["gain"] for row in wave_results]
    direct_gains = [row["direct_model_top4"]["gain"] for row in wave_results]
    uniform_gains = [row["uniform_k4"]["expected_gain"] for row in wave_results]
    oracle_gains = [row["oracle_k4"]["gain"] for row in wave_results]
    return {
        "schema_version": 1,
        "claim_scope": (
            "fixed_parent_fixed_k8_local_allocation_mechanism_diagnostic_not_"
            "campaign_counterfactual_or_efficacy_claim"
        ),
        "wave_count": len(wave_results),
        "waves": wave_results,
        "aggregate": {
            "calibrated_gain_sum": sum(calibrated_gains),
            "direct_model_top4_gain_sum": sum(direct_gains),
            "uniform_expected_gain_sum": sum(uniform_gains),
            "oracle_gain_sum": sum(oracle_gains),
            "calibrated_minus_direct_gain_sum": sum(calibrated_gains)
            - sum(direct_gains),
            "calibrated_minus_uniform_expected_gain_sum": sum(calibrated_gains)
            - sum(uniform_gains),
            "calibrated_wins_vs_direct": sum(
                left > right for left, right in zip(calibrated_gains, direct_gains)
            ),
            "calibrated_ties_vs_direct": sum(
                left == right for left, right in zip(calibrated_gains, direct_gains)
            ),
            "calibrated_losses_vs_direct": sum(
                left < right for left, right in zip(calibrated_gains, direct_gains)
            ),
            "calibrated_wins_vs_uniform_expectation": sum(
                left > right for left, right in zip(calibrated_gains, uniform_gains)
            ),
        },
    }


async def _main_async(args: argparse.Namespace) -> int:
    source_run = args.source_run.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_finalization = verify_finalized_run_directory(source_run)
    summary = _read_json(source_run / "summary.json")
    if (source_run / "wave_requests.jsonl").is_file():
        source_format = "portfolio_q_calibrated_trace_v1"
        evaluation_plan, waves = _build_legacy_plan(
            source_run=source_run,
            summary=summary,
            wave_rows=read_jsonl(source_run / "wave_requests.jsonl"),
        )
    elif all(
        (source_run / name).is_file()
        for name in (
            "manifest.json",
            "campaign_events.jsonl",
            "output_evidence.jsonl",
            "real_evaluator_observations.jsonl",
        )
    ):
        source_format = "generic_authenticated_campaign_trace_v1"
        evaluation_plan, waves = _build_current_campaign_plan(
            manifest=_read_json(source_run / "manifest.json"),
            campaign_rows=read_jsonl(source_run / "campaign_events.jsonl"),
            output_rows=read_jsonl(source_run / "output_evidence.jsonl"),
            observation_rows=read_jsonl(
                source_run / "real_evaluator_observations.jsonl"
            ),
        )
    else:
        raise RuntimeError("source run has no supported authenticated K8 trace")
    recorded_outcome_count = sum(
        member["source_objectives"] is not None
        for wave in waves
        for member in wave["members"]
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "running",
        "diagnostic": "complete_recorded_k8_support_with_rejected_action_outcomes",
        "source_run": {
            "path": source_run.relative_to(WORKSPACE_ROOT).as_posix(),
            "format": source_format,
            "finalization_sha256": source_finalization["finalization_sha256"],
            "recursive_content_sha256": source_finalization[
                "recursive_content_sha256"
            ],
            "summary_sha256": _sha256_file(source_run / "summary.json"),
            "mutated": False,
        },
        "workload": {
            "id": "boils_abc_log2_portfolio_q",
            "fresh_evaluation_count": len(evaluation_plan),
            "recorded_outcome_count": recorded_outcome_count,
            "recorded_selected_outcome_count": 24,
            "completed_k8_outcome_count": 48,
            "affinity_sets": [list(value) for value in args.affinity_sets],
        },
        "claim_boundary": {
            "provider_calls": 0,
            "api_key_reads": 0,
            "campaign_counterfactual": False,
            "fixed_parent_local_allocation_diagnostic": True,
            "paper_ready_efficacy": False,
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
        },
        "source_identity": source_identity(
            (
                Path(__file__),
                AGENT_EVOLVE_ROOT
                / "examples/benchmarks/boils_abc/evaluator.py",
                AGENT_EVOLVE_ROOT
                / "examples/benchmarks/boils_abc/finite_variation_catalog.py",
                AGENT_EVOLVE_ROOT
                / "src/agent_evolve/policies/reward/frozen_wave_archive.py",
                AGENT_EVOLVE_ROOT / "examples/development/durable_run_artifacts.py",
            ),
            relative_to=WORKSPACE_ROOT,
        ),
    }
    write_json_atomic(output_dir / "manifest.json", manifest)
    write_json_atomic(
        output_dir / "evaluation_plan.json",
        {
            "schema_version": 1,
            "fresh_evaluation_count": len(evaluation_plan),
            "rows": evaluation_plan,
        },
    )
    evaluations, wall_s = await _evaluate_rejected(
        evaluation_plan,
        affinity_sets=args.affinity_sets,
    )
    analysis = _allocation_analysis(waves, evaluations)
    write_json_atomic(
        output_dir / "evaluations.json",
        {
            "schema_version": 1,
            "fresh_evaluation_count": len(evaluations),
            "batch_wall_s": wall_s,
            "mean_evaluation_elapsed_s": fmean(
                row["evaluation"]["elapsed_s"] for row in evaluations
            ),
            "rows": evaluations,
        },
    )
    write_json_atomic(output_dir / "analysis.json", analysis)
    result = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": _utc_now(),
        "source_run_mutated": False,
        "provider_calls": 0,
        "api_key_reads": 0,
        "fresh_evaluation_count": len(evaluations),
        "batch_wall_s": wall_s,
        "aggregate": analysis["aggregate"],
        "claim_scope": analysis["claim_scope"],
    }
    write_json_atomic(output_dir / "result.json", result)
    finalization = finalize_run_directory(output_dir, status="completed")
    print(
        json.dumps(
            {
                **result,
                "output_dir": str(output_dir),
                "finalization_sha256": finalization["finalization_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_affinity_sets(value: str) -> tuple[tuple[int, ...], ...]:
    cpus = tuple(int(item) for item in value.split(",") if item)
    if not cpus or len(set(cpus)) != len(cpus) or any(cpu < 0 for cpu in cpus):
        raise argparse.ArgumentTypeError("affinity CPUs must be unique nonnegative IDs")
    return tuple((cpu,) for cpu in cpus)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--affinity-sets",
        type=_parse_affinity_sets,
        default=_parse_affinity_sets(",".join(str(value) for value in range(16))),
        help="comma-separated CPU IDs; each becomes one exclusive evaluator slot",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_main_async(args))
    except BaseException as exc:
        output_dir = args.output_dir.expanduser().resolve(strict=False)
        manifest = output_dir / "manifest.json"
        finalized = output_dir / "finalized.json"
        if manifest.is_file() and not finalized.exists():
            failed = output_dir / "failed.json"
            if not failed.exists():
                write_json_atomic(
                    failed,
                    {
                        "schema_version": 1,
                        "status": "failed_harness",
                        "failure_type": type(exc).__name__,
                        "safe_message": (
                            "counterfactual diagnostic failed; inspect the "
                            "authenticated source and harness"
                        ),
                        "provider_calls": 0,
                        "api_key_reads": 0,
                        "source_run_mutated": False,
                        "scientific_result_available": False,
                    },
                )
            finalize_run_directory(output_dir, status="failed_harness")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
