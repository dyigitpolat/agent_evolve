#!/usr/bin/env python3
"""Measure decision parity and port pressure for a materialized-action slate.

The benchmark deliberately uses only public broker types and counting ports.
Run it against two source roots via ``PYTHONPATH`` to compare implementations
without copying either implementation into the assay.
"""

from __future__ import annotations

import hashlib
import json
import time

from agent_evolve.application.contextual_search_controller import SearchPhase
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionBrokerRequest,
    MaterializedActionContext,
    MaterializedActionDescriptor,
    MaterializedActionEvidenceLedger,
    RegretBrokeredMaterializedActionPolicy,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.typed_json import freeze_json


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


class CountingFeasibility:
    definition_sha256 = _sha("counting-feasibility-v1")

    def __init__(self) -> None:
        self.calls = 0

    def permits(self, actions: tuple[MaterializedActionDescriptor, ...]) -> bool:
        self.calls += 1
        return True


class CountingValue:
    definition_sha256 = _sha("counting-value-v1")

    def __init__(self) -> None:
        self.calls = 0

    def value(self, actions: tuple[MaterializedActionDescriptor, ...]) -> float:
        self.calls += 1
        # A deterministic non-additive objective makes decision parity more
        # informative than a constant or independent score.
        indices = tuple(
            int(dict(action.configuration.items)["candidate_index"])
            for action in actions
        )
        coverage = len({value % 8 for value in indices}) / 8.0
        rank = sum((64 - value) / 64.0 for value in indices) / len(indices)
        return min(1.0, 0.55 * coverage + 0.45 * rank)


def main() -> None:
    context = MaterializedActionContext(
        campaign_scope_sha256=_sha("cache-benchmark-campaign"),
        decision_index=1,
        phase=SearchPhase.BASIN_EXPANSION,
        remaining_decisions=3,
        remaining_evaluations=12,
        residual_frontier_cell="frontier_gap",
        parent_position_cell="parent_edge",
        archive_relation_cell="nondominated_near",
        structural_signature_sha256=_sha("cache-benchmark-structure"),
        patch_compatibility_cell="compatible",
        forecast_calibration_cell="medium_support",
        source_distance_bin=1,
        memory_dose_bin=0,
    )
    actions = tuple(
        MaterializedActionDescriptor(
            context=context,
            configuration=freeze_json({"candidate_index": index}),
            phenotype_identity_sha256=_sha(f"phenotype:{index}"),
            expert_id=("numerical" if index % 3 == 0 else "semantic"),
            native_rank=index + 1,
            parent_ids=(CandidateId(f"candidate_parent_{index}"),),
            operator_id="mutation",
            target_candidate_id=CandidateId(f"candidate_target_{index}"),
            role_id="frontier_candidate",
            normalized_evaluation_cost=0.5,
        )
        for index in range(64)
    )
    feasibility = CountingFeasibility()
    value = CountingValue()
    policy = RegretBrokeredMaterializedActionPolicy(
        MaterializedActionEvidenceLedger(),
        exact_combination_limit=1,
        beam_width=512,
    )
    started = time.perf_counter()
    decision = policy.select(
        MaterializedActionBrokerRequest(
            actions=actions,
            evaluation_slots=4,
            slate_value=value,
            slate_feasibility=feasibility,
            reference_escrow_slots=0,
        )
    )
    print(
        json.dumps(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "feasibility_calls": feasibility.calls,
                "slate_value_calls": value.calls,
                "decision_sha256": decision.decision_sha256,
                "selected_action_sha256s": [
                    action.action_sha256 for action in decision.selected_actions
                ],
                "search_mode": decision.search_mode,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
