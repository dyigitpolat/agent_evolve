"""Thin Timeloop-v2 ports for the generic residual portfolio runtime.

The benchmark adapter owns compiled-plan phenotype identity and the qualified
Timeloop evidence call.  Search allocation, proposal generation, residual
scoring, and portfolio selection remain workload-neutral injected policies.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Callable

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.detailed_evaluation import (
    DetailedEvaluation,
    EvaluationTimings,
)
from agent_evolve.application.materialized_action_broker import (
    MaterializedActionDescriptor,
)
from agent_evolve.application.residual_portfolio_evolution import (
    MaterializedActionEvaluation,
)
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    canonical_typed_json_bytes,
    thaw_json,
)
from agent_evolve.integrations.pydantic_ai.materialized_hierarchical_residual_expert import (
    MaterializedPhenotypeProjectionPort,
    SelectedMaterializedActionEvaluationPort,
)
from agent_evolve.policies.selection.phenotype_recourse import (
    PhenotypeIdentity,
)

from .compiler import COMPILER_DEFINITION_SHA256, TimeloopV2Compiler
from .detailed_evaluation import TimeloopV2DetailedEvaluationAdapter
from .network_panel import panel_sha256
from .problem_def import TimeloopV2CoDesignProblem


TIMELOOP_V2_RESIDUAL_PHENOTYPE_PROJECTION_ID = (
    "timeloop_v2_compiled_plan_identity"
)
TIMELOOP_V2_RESIDUAL_PHENOTYPE_PROJECTION_VERSION = 1
TIMELOOP_V2_SELECTED_ACTION_EVALUATOR_ID = (
    "timeloop_v2_selected_materialized_action"
)
TIMELOOP_V2_SELECTED_ACTION_EVALUATOR_VERSION = 1

_PROJECTION_DEFINITION_DOMAIN = (
    b"agent-evolve:timeloop-v2:residual-phenotype-projection:v1\x00"
)
_EVALUATOR_DEFINITION_DOMAIN = (
    b"agent-evolve:timeloop-v2:selected-action-evaluator:v1\x00"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def timeloop_v2_residual_phenotype_projection_definition_sha256(
    problem: TimeloopV2CoDesignProblem,
) -> str:
    if type(problem) is not TimeloopV2CoDesignProblem:
        raise TypeError("problem must be the exact Timeloop-v2 problem")
    return hashlib.sha256(
        _PROJECTION_DEFINITION_DOMAIN
        + _canonical_json(
            {
                "schema_version": 1,
                "identity": "compiled_plan_sha256",
                "compiler_definition_sha256": COMPILER_DEFINITION_SHA256,
                "panel_sha256": panel_sha256(problem.panel),
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class TimeloopV2ResidualPhenotypeProjection:
    """Use the exact compiler output digest consumed by Timeloop as identity."""

    problem: TimeloopV2CoDesignProblem = field(repr=False, compare=False)
    projection_id: str = TIMELOOP_V2_RESIDUAL_PHENOTYPE_PROJECTION_ID
    projection_version: int = TIMELOOP_V2_RESIDUAL_PHENOTYPE_PROJECTION_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.problem) is not TimeloopV2CoDesignProblem:
            raise TypeError("problem must be the exact Timeloop-v2 problem")
        object.__setattr__(
            self,
            "definition_sha256",
            timeloop_v2_residual_phenotype_projection_definition_sha256(
                self.problem
            ),
        )

    def project(self, configuration: FrozenJsonObject) -> str:
        if type(configuration) is not FrozenJsonObject:
            raise TypeError("configuration must be a frozen object")
        return TimeloopV2Compiler.compile(
            thaw_json(configuration),
            self.problem.panel,
        ).compiled_plan_sha256

    def identify(self, configuration: FrozenJsonObject) -> PhenotypeIdentity:
        return PhenotypeIdentity(
            policy_id=self.projection_id,
            policy_version=self.projection_version,
            value_sha256=self.project(configuration),
        )


def _selected_evaluator_definition_sha256(
    *,
    detailed_adapter: TimeloopV2DetailedEvaluationAdapter,
    phenotype_projection: TimeloopV2ResidualPhenotypeProjection,
) -> str:
    return hashlib.sha256(
        _EVALUATOR_DEFINITION_DOMAIN
        + _canonical_json(
            {
                "schema_version": 1,
                "detailed_evaluator": (
                    detailed_adapter.evaluator_identity.to_record()
                ),
                "phenotype_projection": {
                    "projection_id": phenotype_projection.projection_id,
                    "projection_version": (
                        phenotype_projection.projection_version
                    ),
                    "definition_sha256": (
                        phenotype_projection.definition_sha256
                    ),
                },
                "candidate_projection": "raw_objectives_no_resolution",
                "failure_projection": (
                    "candidate_invalid_with_authenticated_detailed_evidence"
                ),
            }
        )
    ).hexdigest()


@dataclass(slots=True)
class TimeloopV2SelectedMaterializedActionEvaluator:
    """Serialize selected actions through the qualified Timeloop-v2 adapter."""

    detailed_adapter: TimeloopV2DetailedEvaluationAdapter = field(
        repr=False,
        compare=False,
    )
    phenotype_projection: MaterializedPhenotypeProjectionPort = field(
        repr=False,
        compare=False,
    )
    observation_sink: Callable[[dict[str, object]], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    proposal_sequence_start: int = 0
    evaluator_id: str = field(
        init=False,
        default=TIMELOOP_V2_SELECTED_ACTION_EVALUATOR_ID,
    )
    evaluator_version: int = field(
        init=False,
        default=TIMELOOP_V2_SELECTED_ACTION_EVALUATOR_VERSION,
    )
    definition_sha256: str = field(init=False, default="")
    _lock: asyncio.Lock = field(
        init=False,
        default_factory=asyncio.Lock,
        repr=False,
    )
    _evaluation_count: int = field(init=False, default=0)
    _evaluated_action_sha256s: set[str] = field(
        init=False,
        default_factory=set,
        repr=False,
    )

    def __post_init__(self) -> None:
        if type(self.detailed_adapter) is not TimeloopV2DetailedEvaluationAdapter:
            raise TypeError(
                "detailed_adapter must be the qualified Timeloop-v2 adapter"
            )
        if not isinstance(
            self.phenotype_projection,
            MaterializedPhenotypeProjectionPort,
        ):
            raise TypeError(
                "phenotype_projection must implement its runtime port"
            )
        expected = TimeloopV2ResidualPhenotypeProjection(
            self.detailed_adapter.problem
        )
        if (
            self.phenotype_projection.projection_id
            != expected.projection_id
            or self.phenotype_projection.projection_version
            != expected.projection_version
            or self.phenotype_projection.definition_sha256
            != expected.definition_sha256
        ):
            raise ValueError(
                "Timeloop evaluator requires its compiled-plan projection"
            )
        if self.observation_sink is not None and not callable(
            self.observation_sink
        ):
            raise TypeError("observation_sink must be callable or None")
        if (
            type(self.proposal_sequence_start) is not int
            or self.proposal_sequence_start < 0
        ):
            raise ValueError("proposal_sequence_start must be non-negative")
        self.definition_sha256 = _selected_evaluator_definition_sha256(
            detailed_adapter=self.detailed_adapter,
            phenotype_projection=expected,
        )

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    async def evaluate(
        self,
        action: MaterializedActionDescriptor,
    ) -> MaterializedActionEvaluation:
        if type(action) is not MaterializedActionDescriptor:
            raise TypeError("action must be exact")
        action.__post_init__()
        async with self._lock:
            if action.action_sha256 in self._evaluated_action_sha256s:
                raise ValueError(
                    "one materialized action can be evaluated only once"
                )
            phenotype = PhenotypeIdentity(
                policy_id=self.phenotype_projection.projection_id,
                policy_version=self.phenotype_projection.projection_version,
                value_sha256=self.phenotype_projection.project(
                    action.configuration
                ),
            )
            if (
                phenotype.value_sha256
                != action.phenotype_identity_sha256
            ):
                raise RuntimeError(
                    "sealed action differs from compiled Timeloop phenotype"
                )

            started = time.perf_counter_ns()
            payload = await asyncio.to_thread(
                self.detailed_adapter.evaluate_evidence,
                thaw_json(action.configuration),
            )
            elapsed_s = (time.perf_counter_ns() - started) / 1e9
            detailed = DetailedEvaluation(
                phenotype=phenotype,
                payload=payload,
                timings=EvaluationTimings(
                    total_wall_seconds=float(elapsed_s),
                    active_wall_seconds=payload.active_wall_seconds,
                    resource_queue_wall_seconds=(
                        payload.resource_queue_wall_seconds
                    ),
                ),
            )
            self._evaluation_count += 1
            valid = detailed.success
            candidate = EvolutionCandidate(
                occurrence=CandidateOccurrence(
                    candidate_id=action.target_candidate_id,
                    configuration_hash=action.configuration_sha256,
                    configuration_artifact_hash=hashlib.sha256(
                        canonical_typed_json_bytes(action.configuration)
                    ).hexdigest(),
                    proposal_sequence=(
                        self.proposal_sequence_start
                        + self._evaluation_count
                    ),
                ),
                configuration=action.configuration,
                objectives=detailed.objectives if valid else (),
                valid=valid,
                generation=action.context.decision_index,
                label=(
                    f"timeloop_residual_d{action.context.decision_index:04d}_"
                    f"{self._evaluation_count:04d}"
                ),
                parent_ids=action.parent_ids,
                design_rationale=(
                    "Trusted materialization selected by the generic residual "
                    "portfolio broker."
                ),
                failure_message=(
                    None
                    if payload.failure is None
                    else payload.failure.message
                ),
                detailed_evaluation=detailed,
            )
            evaluated = MaterializedActionEvaluation(
                action=action,
                candidate=candidate,
                evaluator_receipt_sha256=detailed.evidence_sha256,
            )
            self._evaluated_action_sha256s.add(action.action_sha256)
            if self.observation_sink is not None:
                action_record = action.to_record(
                    include_configuration=True
                )
                action_record["configuration"] = thaw_json(
                    action.configuration
                )
                self.observation_sink(
                    {
                        "schema_version": 1,
                        "action": action_record,
                        "candidate_id": candidate.candidate_id.value,
                        "valid": candidate.valid,
                        "failure": (
                            None
                            if payload.failure is None
                            else {
                                "category": payload.failure.category.value,
                                "code": payload.failure.code.value,
                                "message": payload.failure.message,
                            }
                        ),
                        "objectives": [
                            {
                                "metric_id": metric_id,
                                "value_hex": value.hex(),
                            }
                            for metric_id, value in candidate.objectives
                        ],
                        "phenotype_identity_sha256": (
                            phenotype.value_sha256
                        ),
                        "detailed_evaluation_sha256": (
                            detailed.evidence_sha256
                        ),
                        "receipt_artifact_id": (
                            None
                            if payload.receipt is None
                            else payload.receipt.artifact_id.value
                        ),
                        "total_wall_seconds": float(elapsed_s),
                        "selected_only_authoritative_evaluation": True,
                    }
                )
            return evaluated


__all__ = [
    "TIMELOOP_V2_RESIDUAL_PHENOTYPE_PROJECTION_ID",
    "TIMELOOP_V2_RESIDUAL_PHENOTYPE_PROJECTION_VERSION",
    "TIMELOOP_V2_SELECTED_ACTION_EVALUATOR_ID",
    "TIMELOOP_V2_SELECTED_ACTION_EVALUATOR_VERSION",
    "TimeloopV2ResidualPhenotypeProjection",
    "TimeloopV2SelectedMaterializedActionEvaluator",
    "timeloop_v2_residual_phenotype_projection_definition_sha256",
]
