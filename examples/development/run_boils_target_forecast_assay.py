#!/usr/bin/env python3
"""Run a real BOiLS action-consequence and allocation mechanism assay.

The workload adapter contributes campaign construction and real ABC evaluation
only.  Target projection, all-action forecasting, partitioned provider
execution, forecast health, archive-aligned allocation, feasibility, and
receipts are the same generic services used by the Heat assay.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = AGENT_EVOLVE_ROOT.parent
if str(AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_EVOLVE_ROOT))

from dotenv import load_dotenv  # noqa: E402

from agent_evolve.application.action_forecast_partitioning import (  # noqa: E402
    ConcurrentActionForecastWave,
    assess_resolved_action_forecast_health,
    build_action_forecast_block_requests,
    build_action_forecast_partition_layout,
    lenient_action_forecast_health_policy,
)
from agent_evolve.application.archive_conditioned_action_target import (  # noqa: E402
    bind_archive_conditioned_affine_action_target,
)
from agent_evolve.application.target_conditioned_action_forecast import (  # noqa: E402
    allocate_target_conditioned_actions,
    audit_target_conditioned_role_allocation,
    build_target_conditioned_action_forecast_plan,
)
from agent_evolve.domain.ids import LLMCallId  # noqa: E402
from agent_evolve.domain.typed_json import thaw_json  # noqa: E402
from agent_evolve.integrations.pydantic_ai.action_forecast import (  # noqa: E402
    PydanticAIActionForecastBlockPolicy,
    plan_action_forecast_block_request,
)
from agent_evolve.integrations.pydantic_ai.progress_aware_openrouter import (  # noqa: E402
    create_progress_aware_openrouter_runner,
)
from agent_evolve.integrations.pydantic_ai.queued_runner import (  # noqa: E402
    StructuredEvidencePublicationPolicy,
    structured_generation_outcome_record,
)
from agent_evolve.ports.action_forecast import (  # noqa: E402
    ActionForecastPartitionPolicyBinding,
)
from agent_evolve.ports.portfolio_selection import (  # noqa: E402
    validate_pairwise_disjoint_parent_patch_selection,
)
from examples.development.durable_run_artifacts import (  # noqa: E402
    BatchedDurableJsonlJournal,
    DurableJsonlJournal,
    finalize_run_directory,
    source_identity,
    write_json_atomic,
)
from examples.development.sealed_assay_replay import (  # noqa: E402
    FinalizedAssayReplay,
    build_finalized_assay_replay,
)
from examples.development import run_boils_generic_campaign as boils  # noqa: E402


ARTIFACT_ROOT = (
    WORKSPACE_ROOT
    / "papers/agent_evolve_aaai_2027/research_artifacts/experiment_logs"
    / "boils_abc/target_forecast_assay"
)
PARTITION_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:boils-target-forecast-assay-partition:v1;"
    b"generic-contiguous-blocks;max-rows=32;max-metric-cells=64"
).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _actual_target_audit(plan: Any, objectives: dict[str, float]) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    l1 = 0.0
    linf = 0.0
    for axis in plan.objective_target.axes:
        child = float(objectives[axis.metric_id])
        shortfall = max(0.0, axis.normalize(child) - axis.aspiration_normalized)
        l1 += shortfall
        linf = max(linf, shortfall)
        cells.append(
            {
                "metric_id": axis.metric_id,
                "parent_value_hex": axis.parent_value.hex(),
                "aspiration_value_hex": axis.aspiration_value.hex(),
                "child_value_hex": child.hex(),
                "signed_parent_to_child_delta_hex": (
                    child - axis.parent_value
                ).hex(),
                "normalized_shortfall_hex": shortfall.hex(),
            }
        )
    return {
        "axes": cells,
        "normalized_shortfall_l1_hex": l1.hex(),
        "normalized_shortfall_linf_hex": linf.hex(),
        "attains_or_dominates_aspiration": linf == 0.0,
    }


async def _run(args: argparse.Namespace) -> int:
    run_dir = (ARTIFACT_ROOT / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    status = "failed"
    preparation = DurableJsonlJournal(run_dir / "preparation.jsonl")
    evaluator_journal = DurableJsonlJournal(run_dir / "evaluator_evidence.jsonl")
    journals: dict[str, Any] = {}
    runner: Any | None = None
    replay: FinalizedAssayReplay | None = None
    try:
        source = source_identity(
            (*boils._source_paths(), Path(__file__).resolve()),
            relative_to=WORKSPACE_ROOT,
        )
        write_json_atomic(
            run_dir / "manifest.json",
            {
                "schema_version": 1,
                "run_id": args.run_id,
                "created_at_utc": _utc_now(),
                "mechanism": "partitioned_target_conditioned_action_forecast",
                "workload": "boils_abc_log2",
                "model_execution_profile": boils.MODEL_EXECUTION_PROFILE.to_record(),
                "source_identity": source,
                "wave_index": args.wave_index,
                "block_rows": args.block_rows,
                "max_concurrency": args.max_concurrency,
                "allocation_utility": args.allocation_utility,
                "factorial_allocation_assay": args.factorial_allocation_assay,
                "physical_evaluation_enabled": not args.no_evaluate,
                "prepare_only": args.prepare_only,
                "replay_source_run_dir": (
                    None
                    if args.replay_source_run_dir is None
                    else str(args.replay_source_run_dir.expanduser().resolve())
                ),
            },
        )
        bundle = boils._prepare_bundle(
            run_dir=run_dir,
            preparation_journal=preparation,
            evaluator_journal=evaluator_journal,
            source_closure_sha256=str(source["aggregate_sha256"]),
            arm="live",
        )
        waves: list[Any] = []
        wave_contexts: list[Any] = []
        probe = boils._all_wave_probe(
            bundle,
            wave_sink=waves.append,
            wave_context_sink=lambda _wave, context: wave_contexts.append(context),
        )
        write_json_atomic(run_dir / "all_wave_probe.json", probe)
        if len(waves) != len(wave_contexts):
            raise RuntimeError("wave/context capture cardinality differs")
        if not 0 <= args.wave_index < len(waves):
            raise ValueError("wave_index is outside the constructed wave set")
        wave = waves[args.wave_index]
        wave_context = wave_contexts[args.wave_index]
        affine_snapshot = bundle.utility.require_snapshot(
            wave_context.stage_request.archive_utility
        )
        selection_request, archive_target = (
            bind_archive_conditioned_affine_action_target(
                selection_request=wave.selection_request,
                archive_utility=wave_context.stage_request.archive_utility,
                affine_snapshot=affine_snapshot,
                parent_objectives=dict(wave.parent.objectives),
                lane_id=wave_context.parent_lane.lane_id,
            )
        )
        option_count = len(selection_request.finite_variation_contract.options)
        if (
            args.expected_portfolio_size is not None
            and selection_request.portfolio_size != args.expected_portfolio_size
        ):
            raise RuntimeError(
                "constructed portfolio size differs from the preregistered value"
            )
        if (
            args.expected_finite_options is not None
            and option_count != args.expected_finite_options
        ):
            raise RuntimeError(
                "constructed finite-option count differs from the preregistered value"
            )
        forecast_identity_seed = (
            args.run_id
            if args.replay_source_run_dir is None
            else args.replay_source_run_dir.expanduser().resolve(strict=True).name
        )
        call_digest = hashlib.sha256(
            forecast_identity_seed.encode("utf-8")
        ).hexdigest()[:24]
        plan = build_target_conditioned_action_forecast_plan(
            selection_request=selection_request,
            optimization_semantics=bundle.benchmark.optimization_semantics,
            call_id=LLMCallId(f"call_target_forecast_{call_digest}"),
            max_output_tokens=boils.MAX_OUTPUT_TOKENS,
            temperature=boils.TEMPERATURE,
        )
        partition_policy = ActionForecastPartitionPolicyBinding(
            policy_id="target_forecast_contiguous_blocks",
            policy_version=1,
            policy_definition_sha256=PARTITION_POLICY_DEFINITION_SHA256,
            max_rows_per_block=args.block_rows,
            max_metric_cells_per_block=(
                args.block_rows * len(plan.request.required_metric_ids)
            ),
        )
        layout = build_action_forecast_partition_layout(
            plan.request,
            partition_policy,
        )
        block_requests = build_action_forecast_block_requests(plan.request, layout)
        prompt_sizes = [
            len(plan_action_forecast_block_request(value).prompt.encode("utf-8"))
            for value in block_requests
        ]
        write_json_atomic(
            run_dir / "forecast_plan.json",
            {
                "schema_version": 1,
                "campaign_target": plan.campaign_target.to_record(),
                "archive_conditioned_target": archive_target.to_record(),
                "objective_target": plan.objective_target.to_record(),
                "residual_cell": (
                    None
                    if plan.residual_cell is None
                    else plan.residual_cell.to_record()
                ),
                "forecast_request": plan.request.to_record(),
                "partition_layout": layout.to_record(),
                "block_prompt_utf8_bytes": prompt_sizes,
                "total_prompt_utf8_bytes": sum(prompt_sizes),
                "finite_option_count": len(
                    plan.request.finite_variation_contract.options
                ),
                "min_distinct_families": plan.min_distinct_families,
                "require_pairwise_disjoint_parent_patches": (
                    plan.require_pairwise_disjoint_parent_patches
                ),
            },
        )
        if args.prepare_only:
            status = "prepared_provider_credential_and_evaluation_free"
            write_json_atomic(
                run_dir / "summary.json",
                {
                    "schema_version": 1,
                    "status": status,
                    "provider_calls": 0,
                    "abc_evaluations": 0,
                    "finite_option_count": len(
                        plan.request.finite_variation_contract.options
                    ),
                    "partition_block_count": layout.block_count,
                    "residual_cell_available": plan.residual_cell is not None,
                    "hard_constraints": {
                        "min_distinct_families": plan.min_distinct_families,
                        "require_pairwise_disjoint_parent_patches": (
                            plan.require_pairwise_disjoint_parent_patches
                        ),
                    },
                },
            )
            return 0

        started_ns = time.perf_counter_ns()

        def observed(value: dict[str, object]) -> dict[str, object]:
            return {
                "observation": {
                    "monotonic_ns_since_execution_start": (
                        time.perf_counter_ns() - started_ns
                    ),
                    "observed_at_utc": _utc_now(),
                },
                "authenticated_record": value,
            }

        if args.replay_source_run_dir is None:
            load_dotenv(WORKSPACE_ROOT / ".env", override=False)
            load_dotenv(AGENT_EVOLVE_ROOT / ".env", override=False)
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if type(api_key) is not str or not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is unavailable")
            journals = {
                "requests": DurableJsonlJournal(
                    run_dir / "request_evidence.jsonl"
                ),
                "outputs": DurableJsonlJournal(run_dir / "output_evidence.jsonl"),
                "outcomes": DurableJsonlJournal(run_dir / "queue_outcomes.jsonl"),
                "outbound": DurableJsonlJournal(
                    run_dir / "outbound_requests.jsonl"
                ),
                "progress": BatchedDurableJsonlJournal(
                    run_dir / "stream_progress.jsonl", max_unfsynced_rows=32
                ),
            }

            def progress_sink(value: Any) -> None:
                journals["progress"].append(observed(boils._progress_record(value)))

            def outcome_sink(value: object) -> None:
                journals["progress"].flush()
                journals["outcomes"].append(
                    observed(structured_generation_outcome_record(value))
                )

            runner = create_progress_aware_openrouter_runner(
                api_key=api_key,
                config=boils._provider_config(),
                progress_sink=progress_sink,
                outcome_sink=outcome_sink,
                request_evidence_sink=lambda value: journals["requests"].append(
                    observed(value)
                ),
                output_evidence_sink=lambda value: journals["outputs"].append(
                    observed(value)
                ),
                outbound_request_manifest_sink=lambda value: journals[
                    "outbound"
                ].append(observed(value)),
                evidence_publication_policy=(
                    StructuredEvidencePublicationPolicy.REQUIRED
                ),
            )
        else:
            journals = {
                "replay_decisions": DurableJsonlJournal(
                    run_dir / "sealed_replay_decisions.jsonl"
                )
            }
            replay = build_finalized_assay_replay(
                source_run_dir=args.replay_source_run_dir,
                requested_model=boils.MODEL,
                decision_receipt_sink=lambda value: journals[
                    "replay_decisions"
                ].append(observed(value)),
            )
            if replay.source.accepted_output_count != layout.block_count:
                raise RuntimeError(
                    "sealed replay output count differs from the forecast layout"
                )
            write_json_atomic(
                run_dir / "sealed_replay_source.json",
                {
                    "source_run_dir": str(replay.source_run_dir),
                    "source_finalization_sha256": (
                        replay.source_finalization_sha256
                    ),
                    "source_receipt": replay.source.source_receipt(),
                },
            )
            runner = replay.runner
        forecast_started = time.perf_counter()
        result = await ConcurrentActionForecastWave(
            block_policy=PydanticAIActionForecastBlockPolicy(runner),
            max_concurrency=args.max_concurrency,
        ).forecast_partitioned(plan.request, layout)
        if replay is not None and replay.runner.remaining_entry_count != 0:
            raise RuntimeError("sealed replay did not consume every accepted output")
        forecast_wall_s = time.perf_counter() - forecast_started
        health = assess_resolved_action_forecast_health(
            plan.request,
            result.forecasts,
            member_id="target_forecast_assay",
            health_policy=lenient_action_forecast_health_policy(),
        )
        utility_modes = [args.allocation_utility]
        if args.factorial_allocation_assay:
            utility_modes.extend(
                value
                for value in (
                    "target_closure",
                    "expected_hypervolume",
                    "reliability_adjusted_expected_hypervolume",
                    "role_factorized",
                )
                if value not in utility_modes
            )
        allocations: dict[str, Any] = {}
        role_assignment_audits: dict[str, list[dict[str, object]]] = {}
        for utility_mode in utility_modes:
            allocation = allocate_target_conditioned_actions(
                plan=plan,
                forecasts=result.forecasts,
                portfolio_size=selection_request.portfolio_size,
                beam_width=args.beam_width,
                utility_mode=utility_mode,
            )
            selected = tuple(
                value.option_id for value in allocation.decision.members
            )
            if plan.require_pairwise_disjoint_parent_patches:
                validate_pairwise_disjoint_parent_patch_selection(
                    plan.request.finite_variation_contract,
                    selected,
                )
            if plan.min_distinct_families is not None and len(
                {value.family for value in allocation.decision.members}
            ) < plan.min_distinct_families:
                raise RuntimeError("BOiLS allocation violated family coverage")
            allocations[utility_mode] = allocation
            if utility_mode == "role_factorized":
                role_assignment_audits[utility_mode] = [
                    value.to_record()
                    for value in audit_target_conditioned_role_allocation(
                        plan=plan,
                        forecasts=result.forecasts,
                        decision=allocation.decision,
                    )
                ]
        primary = allocations[args.allocation_utility]
        selected_ids: list[str] = []
        for allocation in allocations.values():
            for member in allocation.decision.members:
                if member.option_id not in selected_ids:
                    selected_ids.append(member.option_id)
        realization = plan.assess(result.forecasts)
        write_json_atomic(run_dir / "forecast_receipt.json", result.forecasts.to_record())
        write_json_atomic(
            run_dir / "forecast_blocks.json",
            {
                "partitioned_receipt": result.to_record(),
                "blocks": [
                    {
                        "forecast": value.forecasts.to_record(),
                        "telemetry": (
                            None
                            if value.telemetry is None
                            else boils._telemetry_record(value.telemetry)
                        ),
                    }
                    for value in result.block_results
                ],
            },
        )
        write_json_atomic(
            run_dir / "target_realization_forecasts.json",
            {
                "schema_version": 1,
                "rows": [value.to_record() for value in realization],
            },
        )
        write_json_atomic(run_dir / "allocation.json", primary.decision.to_record())
        write_json_atomic(
            run_dir / "allocations.json",
            {
                "schema_version": 1,
                "primary_utility": args.allocation_utility,
                "arms": {
                    key: value.decision.to_record()
                    for key, value in allocations.items()
                },
            },
        )
        write_json_atomic(
            run_dir / "role_assignment_audits.json",
            {"schema_version": 1, "arms": role_assignment_audits},
        )

        physical: list[dict[str, object]] = []
        if not args.no_evaluate:
            options = {
                value.option_id: value
                for value in plan.request.finite_variation_contract.options
            }
            memberships = {
                option_id: [
                    {"utility": key, "rank": member.rank}
                    for key, allocation in allocations.items()
                    for member in allocation.decision.members
                    if member.option_id == option_id
                ]
                for option_id in selected_ids
            }
            for option_id in selected_ids:
                option = options[option_id]
                evaluation_started = time.perf_counter()
                objectives = bundle.benchmark.problem.evaluate(
                    thaw_json(option.child_configuration)
                )
                evaluation_wall_s = time.perf_counter() - evaluation_started
                physical.append(
                    {
                        "option_id": option.option_id,
                        "option_identity_sha256": option.identity_sha256,
                        "child_configuration_sha256": (
                            option.child_configuration_sha256
                        ),
                        "family": option.family,
                        "allocation_memberships": memberships[option_id],
                        "objectives": {
                            key: {
                                "value": float(value),
                                "value_hex": float(value).hex(),
                            }
                            for key, value in sorted(objectives.items())
                        },
                        "target_audit": _actual_target_audit(plan, objectives),
                        "evaluation_wall_s": evaluation_wall_s,
                    }
                )
                write_json_atomic(
                    run_dir / "physical_evaluations_partial.json",
                    {"schema_version": 1, "rows": physical},
                )
        write_json_atomic(
            run_dir / "physical_evaluations.json",
            {"schema_version": 1, "rows": physical},
        )

        telemetry = tuple(
            value.telemetry
            for value in result.block_results
            if value.telemetry is not None
        )
        for value in telemetry:
            boils.MODEL_EXECUTION_PROFILE.validate_telemetry(value)
        await runner.aclose()
        snapshot = await runner.snapshot()
        runner = None
        status = "completed_healthy" if health.passes else "completed_unhealthy"
        summary = {
            "schema_version": 1,
            "status": status,
            "health_pass": health.passes,
            "workload": "boils_abc_log2",
            "model_profile": boils.MODEL_PROFILE_NAME,
            "requested_model": boils.MODEL,
            "resolved_models": sorted({value.resolved_model for value in telemetry}),
            "resolved_providers": sorted(
                {value.resolved_provider for value in telemetry}
            ),
            "reasoning_tokens": sum(value.reasoning_tokens for value in telemetry),
            "input_tokens": sum(value.input_tokens for value in telemetry),
            "output_tokens": sum(value.output_tokens for value in telemetry),
            "cache_read_tokens": sum(value.cache_read_tokens for value in telemetry),
            "cost_usd": str(
                sum(
                    (
                        value.cost_usd
                        for value in telemetry
                        if value.cost_usd is not None
                    ),
                    start=0,
                )
            ),
            "forecast_wall_s": forecast_wall_s,
            "physical_provider_call_count": (
                len(result.block_results) if replay is None else 0
            ),
            "historical_provider_output_count_replayed": (
                0 if replay is None else len(result.block_results)
            ),
            "replay_source": (
                None
                if replay is None
                else {
                    "source_id": replay.source.source_id,
                    "source_identity_sha256": (
                        replay.source.source_identity_sha256
                    ),
                    "source_finalization_sha256": (
                        replay.source_finalization_sha256
                    ),
                    "telemetry_origin": "historical_source_attempt",
                }
            ),
            "finite_option_count": len(result.forecasts.forecasts),
            "partition_block_count": layout.block_count,
            "forecast_health": health.to_record(),
            "allocation_utility": args.allocation_utility,
            "factorial_allocation_assay": args.factorial_allocation_assay,
            "allocation": primary.decision.to_record(),
            "allocations": {
                key: value.decision.to_record()
                for key, value in allocations.items()
            },
            "role_assignment_audits": role_assignment_audits,
            "physical_evaluations": physical,
            "hard_feasibility": {
                "min_distinct_families": plan.min_distinct_families,
                "require_pairwise_disjoint_parent_patches": (
                    plan.require_pairwise_disjoint_parent_patches
                ),
                "passes": True,
            },
            "queue_snapshot_after_close": {
                "closed": snapshot.closed,
                "in_flight": snapshot.in_flight,
                "pending": snapshot.pending,
                "max_in_flight": snapshot.max_in_flight,
                "max_pending": snapshot.max_pending,
            },
        }
        write_json_atomic(run_dir / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True))
        return 0
    except BaseException as exc:
        write_json_atomic(
            run_dir / "summary.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
            },
        )
        raise
    finally:
        if runner is not None:
            await runner.aclose()
        preparation.close()
        evaluator_journal.close()
        for journal in journals.values():
            journal.close()
        finalize_run_directory(run_dir, status=status)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--wave-index", type=int, default=0)
    parser.add_argument("--block-rows", type=int, default=32)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--expected-portfolio-size", type=int)
    parser.add_argument("--expected-finite-options", type=int)
    parser.add_argument(
        "--allocation-utility",
        choices=(
            "target_closure",
            "expected_hypervolume",
            "reliability_adjusted_expected_hypervolume",
            "role_factorized",
        ),
        default="expected_hypervolume",
    )
    parser.add_argument("--factorial-allocation-assay", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--replay-source-run-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
