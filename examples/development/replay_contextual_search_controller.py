#!/usr/bin/env python3
"""Replay the generic contextual controller over authenticated trace analysis.

This is a retrospective behavioral qualification, not a counterfactual search
result.  It reconstructs source and operator observations that were actually
evaluated, exposes only observations from earlier portfolio waves to each
decision, and records the allocation the controller *would* request next.  It
does not claim that unevaluated candidates would have had the observed yield.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from agent_evolve.application.contextual_search_controller import (
    ContextualSearchDelayedCredit,
    ContextualSearchLedger,
    ContextualSearchObservation,
    ContextualSearchQuery,
    PhaseAwareContextualSearchController,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict or type(value.get("runs")) is not list:
        raise ValueError(f"{path} is not a systematic trace-analysis object")
    return value


def _treatment_run(value: dict[str, Any]) -> dict[str, Any]:
    runs = value["runs"]
    healthy = [
        run
        for run in runs
        if type(run) is dict
        and run.get("status") == "completed_healthy"
        and run.get("model_profile") != "provider_free_conditional_uniform"
    ]
    if len(healthy) != 1:
        raise ValueError("each replay input must identify one healthy treatment run")
    return healthy[0]


def _source_by_evaluated_cell(run: dict[str, Any]) -> dict[tuple[int, int, str], str]:
    reconciliation = run.get("selector_behavior", {}).get(
        "semantic_reconciliation", {}
    )
    calls = reconciliation.get("calls")
    if type(calls) is not list:
        raise ValueError("trace analysis omitted semantic reconciliation calls")
    result: dict[tuple[int, int, str], str] = {}
    for call in calls:
        if type(call) is not dict:
            raise ValueError("semantic reconciliation call is malformed")
        generation = call.get("generation")
        parent_slot = call.get("parent_slot")
        members = call.get("members")
        if type(generation) is not int or type(parent_slot) is not int:
            raise ValueError("semantic reconciliation call omitted its coordinates")
        if type(members) is not list:
            raise ValueError("semantic reconciliation call omitted members")
        for member in members:
            if type(member) is not dict or member.get("evaluated") is not True:
                continue
            option_id = member.get("option_id")
            origin = member.get("origin")
            if type(option_id) is not str or type(origin) is not str:
                raise ValueError("evaluated reconciliation member is malformed")
            source = "model" if origin == "model" else "engine"
            key = (generation, parent_slot, option_id)
            if key in result:
                raise ValueError("evaluated source cell is not unique")
            result[key] = source
    return result


def _portfolio_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    joined = run.get("rank_role_join", {}).get("joined_candidates")
    if type(joined) is not list or not joined:
        raise ValueError("trace analysis omitted joined portfolio candidates")
    return [row for row in joined if type(row) is dict]


def _operator(option_id: str) -> str:
    return "composite" if option_id.startswith("compose.") else "atomic"


def _unwrap(value: dict[str, Any]) -> dict[str, Any]:
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


def _recombination_descendant_labels(run: dict[str, Any]) -> dict[int, dict[str, bool]]:
    run_dir = run.get("run_dir")
    if type(run_dir) is not str:
        raise ValueError("trace analysis omitted its raw run directory")
    path = Path(run_dir) / "campaign_events.jsonl"
    rows = [
        _unwrap(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels: dict[int, dict[str, bool]] = {}
    for event in rows:
        if event.get("kind") != "stage_sealed":
            continue
        receipt = event.get("payload", {}).get("stage_receipt")
        if type(receipt) is not dict or receipt.get("kind") != "recombination":
            continue
        generation = receipt.get("generation")
        result = receipt.get("result")
        if type(generation) is not int or type(result) is not dict:
            raise ValueError("recombination stage receipt is malformed")
        candidates = result.get("candidates")
        front = result.get("archive_after", {}).get("summary", {}).get("front")
        if type(candidates) is not list or type(front) is not list:
            raise ValueError("recombination stage omitted candidate/front evidence")
        front_ids = {
            value.get("candidate_id") for value in front if type(value) is dict
        }
        exposed: dict[str, list[bool]] = {}
        for candidate in candidates:
            if type(candidate) is not dict:
                raise ValueError("recombination candidate is malformed")
            candidate_id = candidate.get("candidate_id")
            parent_ids = candidate.get("parent_ids")
            if type(candidate_id) is not str or type(parent_ids) is not list:
                raise ValueError("recombination candidate omitted its lineage")
            useful = candidate_id in front_ids
            for parent_id in parent_ids:
                if type(parent_id) is not str:
                    raise ValueError("recombination parent identity is malformed")
                exposed.setdefault(parent_id, []).append(useful)
        labels[generation] = {
            parent_id: any(values) for parent_id, values in exposed.items()
        }
    return labels


def _allocation_record(values: tuple[Any, ...]) -> dict[str, int]:
    return {value.arm_id: value.target_slots for value in values}


def _replay_run(run: dict[str, Any], *, input_path: Path) -> dict[str, Any]:
    run_id = run.get("run_id")
    workload_id = run.get("workload_id")
    model_profile = run.get("model_profile")
    if not all(
        type(value) is str and value
        for value in (run_id, workload_id, model_profile)
    ):
        raise ValueError("trace run identity is incomplete")
    source_by_cell = _source_by_evaluated_cell(run)
    joined = _portfolio_rows(run)
    generations = tuple(sorted({row.get("generation") for row in joined}))
    if not generations or any(type(value) is not int for value in generations):
        raise ValueError("joined candidate generations are malformed")
    generation_to_wave = {
        generation: ordinal
        for ordinal, generation in enumerate(generations, start=1)
    }
    campaign_scope = _sha(f"retrospective-controller-replay:{run_id}")
    final_rows = {
        row.get("candidate_id")
        for row in joined
        if row.get("admitted_to_final_front") is True
    }
    observations_by_wave: dict[int, list[ContextualSearchObservation]] = {
        wave: [] for wave in generation_to_wave.values()
    }
    raw_rows_by_wave: dict[int, list[dict[str, Any]]] = {
        wave: [] for wave in generation_to_wave.values()
    }
    observation_by_candidate_id: dict[str, ContextualSearchObservation] = {}
    for generation in generations:
        wave = generation_to_wave[generation]
        rows = [row for row in joined if row.get("generation") == generation]
        positive_total = sum(
            max(0.0, float(row.get("individual_marginal_hypervolume", 0.0)))
            for row in rows
        )
        for row in rows:
            option_id = row.get("option_id")
            parent_slot = row.get("parent_slot")
            candidate_id = row.get("candidate_id")
            if (
                type(option_id) is not str
                or type(parent_slot) is not int
                or type(candidate_id) is not str
            ):
                raise ValueError("joined candidate row omitted identity coordinates")
            source = source_by_cell.get((generation, parent_slot, option_id))
            if source is None:
                raise ValueError(
                    "joined candidate lacks an authenticated reconciliation source: "
                    f"{run_id}/{generation}/{parent_slot}/{option_id}"
                )
            raw_gain = max(
                0.0,
                float(row.get("individual_marginal_hypervolume", 0.0)),
            )
            share = 0.0 if positive_total == 0.0 else raw_gain / positive_total
            observation = ContextualSearchObservation(
                campaign_scope_sha256=campaign_scope,
                wave_index=wave,
                source_id=source,
                operator_id=_operator(option_id),
                option_identity_sha256=_sha(
                    f"retrospective-option:{run_id}:{generation}:{parent_slot}:"
                    f"{option_id}"
                ),
                parent_context_sha256=_sha(
                    f"retrospective-parent:{run_id}:{generation}:{parent_slot}"
                ),
                feasible=row.get("typed_candidate_infeasible") is not True,
                positive_marginal_utility=raw_gain > 0.0,
                normalized_marginal_utility=float(raw_gain),
                marginal_utility_share=float(share),
                # Final-front membership is only known after all later waves.
                # Keep it in the retrospective diagnostic row below, but never
                # expose it to a prior-only controller decision.
                final_front_persisted=None,
                useful_descendant_observed=None,
                source_distance=0.0,
            )
            observations_by_wave[wave].append(observation)
            observation_by_candidate_id[candidate_id] = observation
            raw_rows_by_wave[wave].append(
                {
                    "candidate_id": candidate_id,
                    "generation": generation,
                    "parent_slot": parent_slot,
                    "option_id": option_id,
                    "source_id": source,
                    "operator_id": _operator(option_id),
                    "individual_marginal_hypervolume": raw_gain,
                    "marginal_utility_share": share,
                    "final_front_persisted": candidate_id in final_rows,
                    "observation_sha256": observation.observation_sha256,
                }
            )

    descendant_labels = _recombination_descendant_labels(run)
    credit_fields: dict[tuple[int, str], dict[str, bool]] = {}
    for recombination_generation, labels in descendant_labels.items():
        source_generation = recombination_generation - 1
        source_wave = generation_to_wave.get(source_generation)
        if source_wave is None:
            continue
        available_at = source_wave + 1
        for candidate_id, useful in labels.items():
            observation = observation_by_candidate_id.get(candidate_id)
            if observation is None:
                raise ValueError(
                    "recombination lineage cites an unjoined portfolio parent"
                )
            credit_fields.setdefault(
                (available_at, observation.observation_sha256), {}
            )["useful_descendant_observed"] = useful
    final_credit_wave = len(generations) + 1
    for candidate_id, observation in observation_by_candidate_id.items():
        credit_fields.setdefault(
            (final_credit_wave, observation.observation_sha256), {}
        )["final_front_persisted"] = candidate_id in final_rows
    credits_by_wave: dict[int, list[ContextualSearchDelayedCredit]] = {}
    for (available_at, observation_sha256), fields in credit_fields.items():
        credit = ContextualSearchDelayedCredit(
            campaign_scope_sha256=campaign_scope,
            source_observation_sha256=observation_sha256,
            available_at_wave_index=available_at,
            final_front_persisted=fields.get("final_front_persisted"),
            useful_descendant_observed=fields.get("useful_descendant_observed"),
        )
        credits_by_wave.setdefault(available_at, []).append(credit)

    ledger = ContextualSearchLedger()
    controller = PhaseAwareContextualSearchController()
    decisions: list[dict[str, Any]] = []
    prior_positive_atomic = 0
    prior_stage_gains: list[float] = []
    for wave in range(1, len(generations) + 1):
        query = ContextualSearchQuery(
            campaign_scope_sha256=campaign_scope,
            wave_index=wave,
            total_portfolio_waves=len(generations),
            real_evaluation_slots=len(raw_rows_by_wave[wave]),
            available_source_ids=("engine", "model"),
            available_operator_ids=("atomic", "composite"),
            incumbent_source_id="model",
            incumbent_operator_id="atomic",
            archive_front_size=max(
                1,
                int(run.get("quality", {}).get("final_front_size", 1)),
            ),
            recent_normalized_archive_gains=tuple(prior_stage_gains[-4:]),
            composition_evidence_available=prior_positive_atomic >= 2,
        )
        snapshot = ledger.snapshot(
            campaign_scope_sha256=campaign_scope,
            cutoff_wave_index_exclusive=wave,
            available_source_ids=query.available_source_ids,
            available_operator_ids=query.available_operator_ids,
        )
        decision = controller.decide(query, snapshot)
        decisions.append(
            {
                "wave_index": wave,
                "historical_generation": generations[wave - 1],
                "phase": decision.phase.value,
                "source_allocation": _allocation_record(decision.source_allocations),
                "operator_allocation": _allocation_record(
                    decision.operator_allocations
                ),
                "source_scores": {
                    value.arm_id: value.score for value in decision.source_allocations
                },
                "source_allocation_probabilities": {
                    value.arm_id: value.allocation_probability
                    for value in decision.source_allocations
                },
                "operator_scores": {
                    value.arm_id: value.score for value in decision.operator_allocations
                },
                "operator_allocation_probabilities": {
                    value.arm_id: value.allocation_probability
                    for value in decision.operator_allocations
                },
                "prior_observation_count": len(snapshot.observation_sha256s),
                "prior_delayed_credit_count": len(
                    snapshot.delayed_credit_sha256s
                ),
                "decision_sha256": decision.decision_sha256,
                "observed_historical_composition": {
                    "source_counts": {
                        source: sum(
                            row["source_id"] == source
                            for row in raw_rows_by_wave[wave]
                        )
                        for source in ("engine", "model")
                    },
                    "operator_counts": {
                        operator: sum(
                            row["operator_id"] == operator
                            for row in raw_rows_by_wave[wave]
                        )
                        for operator in ("atomic", "composite")
                    },
                },
            }
        )
        current = tuple(
            sorted(
                observations_by_wave[wave],
                key=lambda value: value.observation_sha256,
            )
        )
        ledger.append_batch(current)
        delayed = tuple(
            sorted(
                credits_by_wave.get(wave + 1, []),
                key=lambda value: value.credit_sha256,
            )
        )
        if delayed:
            ledger.append_delayed_credit_batch(delayed)
        prior_positive_atomic += sum(
            value.operator_id == "atomic" and value.positive_marginal_utility
            for value in current
        )
        prior_stage_gains.append(
            min(
                1.0,
                sum(
                    row["individual_marginal_hypervolume"]
                    for row in raw_rows_by_wave[wave]
                ),
            )
        )

    final_snapshot = ledger.snapshot(
        campaign_scope_sha256=campaign_scope,
        cutoff_wave_index_exclusive=len(generations) + 1,
        available_source_ids=("engine", "model"),
        available_operator_ids=("atomic", "composite"),
    )
    return {
        "run_id": run_id,
        "workload_id": workload_id,
        "model_profile": model_profile,
        "source_trace_analysis": str(input_path),
        "campaign_scope_sha256": campaign_scope,
        "portfolio_wave_count": len(generations),
        "observation_count": len(ledger.observations),
        "decisions": decisions,
        "final_snapshot": final_snapshot.to_record(),
        "observations": [
            row
            for wave in range(1, len(generations) + 1)
            for row in raw_rows_by_wave[wave]
        ],
        "delayed_credits": [
            value.to_record()
            for value in sorted(
                ledger.delayed_credits,
                key=lambda value: (
                    value.available_at_wave_index,
                    value.credit_sha256,
                ),
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [
        _replay_run(_treatment_run(_load(path)), input_path=path)
        for path in args.inputs
    ]
    record = {
        "schema_version": 1,
        "analysis_kind": "retrospective_prior_only_contextual_controller_replay",
        "claim_boundary": {
            "counterfactual_performance_identified": False,
            "causal_quality_effect_identified": False,
            "provider_calls": 0,
            "real_evaluator_calls": 0,
            "purpose": (
                "behavioral qualification of prior-only phase/source/operator "
                "decisions on already observed campaign outcomes"
            ),
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
