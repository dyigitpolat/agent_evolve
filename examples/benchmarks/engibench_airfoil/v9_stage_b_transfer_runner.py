"""Durable CLI for outcome-blind fresh-parent Stage-B transfer blocks."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable

from agent_evolve.agentic import ExclusiveResourceLease
from examples.benchmarks.engibench_airfoil.v8_stage_b_runner import (
    AirfoilV8StageBRunnerError,
    _lease,
    _read_dotenv_api_key,
    _result_record,
    execute_live as execute_stage_b_live,
    readiness as stage_b_readiness,
)
from examples.benchmarks.engibench_airfoil.v9_stage_b_transfer import (
    airfoil_v9_stage_b_transfer_readiness_record,
    compose_airfoil_v9_stage_b_transfer_inputs,
)


LIVE_AUTHORIZATION = "AIRFOIL_V9_STAGE_B_FRESH_TRANSFER_LIVE_V1"


def _inputs_factory(panel_index: int):
    return lambda source: compose_airfoil_v9_stage_b_transfer_inputs(
        source,
        panel_index=panel_index,
    )


def _readiness_factory(panel_index: int):
    return lambda source, inputs: airfoil_v9_stage_b_transfer_readiness_record(
        source,
        inputs,
        panel_index=panel_index,
    )


def _result_factory(panel_index: int):
    def build(result, live, source, inputs) -> dict[str, object]:
        record = _result_record(result, live)
        readiness = airfoil_v9_stage_b_transfer_readiness_record(
            source,
            inputs,
            panel_index=panel_index,
        )
        uniform = live.composition.planner.uniform_decision
        prospective = readiness["prospective_uniform"]
        if (
            uniform is None
            or uniform.selected_ordinal != prospective["selected_ordinal"]
            or uniform.option_id != prospective["option_id"]
        ):
            raise AirfoilV8StageBRunnerError(
                "live U decision differs from its pre-outcome readiness seal"
            )
        record.update(
            {
                "schema_version": 2,
                "claim_boundary": (
                    "fresh_parent_single_block_development_not_replicated_"
                    "paper_evidence"
                ),
                "transfer_parent": readiness["transfer_parent"],
                "prospective_uniform": prospective,
                "selector_evidence_scope": {
                    "current_parent_outcomes_in_learned_card": False,
                    "source_parent_and_source_action_outcomes_in_learned_card": True,
                    "support_is_card_sign_pattern_local_neighbourhood": True,
                    "global_uninformed_support_comparator_present": False,
                },
            }
        )
        return record

    return build


def readiness(run_id: str, *, panel_index: int) -> dict[str, object]:
    return stage_b_readiness(
        run_id,
        inputs_factory=_inputs_factory(panel_index),
        readiness_record_factory=_readiness_factory(panel_index),
    )


async def execute_live(
    run_id: str,
    *,
    panel_index: int,
    credential_source: Callable[[], str] = _read_dotenv_api_key,
    resource_lease_factory: Callable[[str], ExclusiveResourceLease] = _lease,
    generator_factory=None,
) -> dict[str, object]:
    return await execute_stage_b_live(
        run_id,
        credential_source=credential_source,
        resource_lease_factory=resource_lease_factory,
        generator_factory=generator_factory,
        inputs_factory=_inputs_factory(panel_index),
        readiness_record_factory=_readiness_factory(panel_index),
        result_record_factory=_result_factory(panel_index),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ready = sub.add_parser("readiness")
    ready.add_argument("--run-id", required=True)
    ready.add_argument("--panel-index", required=True, type=int)
    run = sub.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--panel-index", required=True, type=int)
    run.add_argument("--authorize-live", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "readiness":
        print(
            json.dumps(
                readiness(args.run_id, panel_index=args.panel_index),
                sort_keys=True,
            )
        )
        return 0
    if args.authorize_live != LIVE_AUTHORIZATION:
        raise AirfoilV8StageBRunnerError("explicit live authorization token required")
    outcome = asyncio.run(
        execute_live(args.run_id, panel_index=args.panel_index)
    )
    print(outcome["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LIVE_AUTHORIZATION",
    "execute_live",
    "main",
    "readiness",
]
