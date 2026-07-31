"""Thin Heat2D ports for the generic residual portfolio runtime.

This module contains the benchmark-specific side of the inverted API.  The
generic optimizer never imports it.  The adapter owns only evaluator-semantic
phenotype identity, the qualified direct-v3 truth call, fixed-grid objective
resolution, and optional durable observation publication.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json
import time
from typing import Callable

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.detailed_evaluation import (
    DetailedEvaluation,
    DetailedEvaluationPayload,
    EvaluationTimings,
    EvaluatorIdentity,
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
from agent_evolve.policies.objective_resolution.fixed_grid import (
    FixedGridMetricSpec,
    FixedGridObjectiveResolution,
    FixedGridRoundingLaw,
)
from agent_evolve.ports.objective_resolution import (
    ObjectiveResolutionRequest,
    resolve_objectives,
)

from .multiobjective_v1 import (
    FORMULATION_DEFINITION_SHA256,
    MATERIAL_OBJECTIVE_NAME,
    THERMAL_OBJECTIVE_NAME,
    WORKLOAD_ID,
    Heat2DMultiObjectiveV1Problem,
)
from .phenotype_identity import Heat2DPhenotypeIdentityPolicy


HEAT2D_RESIDUAL_PHENOTYPE_PROJECTION_ID = (
    "heat2d_direct_v3_field_value_identity"
)
HEAT2D_RESIDUAL_PHENOTYPE_PROJECTION_VERSION = 1
HEAT2D_SELECTED_ACTION_EVALUATOR_ID = (
    "heat2d_constructive_pareto_direct_v3_selected_action"
)
HEAT2D_SELECTED_ACTION_EVALUATOR_VERSION = 1
HEAT2D_OBJECTIVE_RESOLUTION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:heat2d-residual-objective-resolution:v1;"
    b"metrics=material-fraction-plus-thermal-term;"
    b"decimal-quantum=1e-12;rounding=nearest-ties-to-even"
).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def heat2d_residual_phenotype_projection_definition_sha256(
    resolution: int,
) -> str:
    """Bind decoded-field identity to the exact direct-v3 mesh."""

    if type(resolution) is not int or resolution < 3:
        raise ValueError("resolution must be at least three")
    return hashlib.sha256(
        b"agent-evolve:heat2d-residual-phenotype-projection:v1\x00"
        + _canonical_json(
            {
                "identity": (
                    "qualified-decoded-dense-field-value-sha256"
                ),
                "formulation_definition_sha256": (
                    FORMULATION_DEFINITION_SHA256
                ),
                "resolution": resolution,
            }
        )
    ).hexdigest()


HEAT2D_RESIDUAL_PHENOTYPE_PROJECTION_DEFINITION_SHA256 = (
    heat2d_residual_phenotype_projection_definition_sha256(1001)
)


def heat2d_objective_resolution() -> FixedGridObjectiveResolution:
    """Return the qualified public raw-to-decision objective policy."""

    return FixedGridObjectiveResolution(
        metric_specs=tuple(
            FixedGridMetricSpec(
                metric_id=metric_id,
                decimal_origin=Decimal("0"),
                decimal_quantum=Decimal("0.000000000001"),
                rounding_law=FixedGridRoundingLaw.NEAREST_TIES_TO_EVEN,
            )
            for metric_id in sorted(
                (THERMAL_OBJECTIVE_NAME, MATERIAL_OBJECTIVE_NAME)
            )
        )
    )


def _evaluator_definition_sha256(
    problem: Heat2DMultiObjectiveV1Problem,
) -> str:
    settings = problem.settings
    return hashlib.sha256(
        b"agent-evolve:heat2d-selected-materialized-action-evaluator:v1\x00"
        + _canonical_json(
            {
                "workload_id": WORKLOAD_ID,
                "formulation_definition_sha256": (
                    FORMULATION_DEFINITION_SHA256
                ),
                "resolution": settings.resolution,
                "required_numpy_version": settings.required_numpy_version,
                "external_concurrency": settings.external_concurrency,
                "objective_resolution_definition_sha256": (
                    HEAT2D_OBJECTIVE_RESOLUTION_DEFINITION_SHA256
                ),
                "selected_subset_only": True,
                "candidate_generation_authority": False,
            }
        )
    ).hexdigest()


def _evaluator_identity(
    problem: Heat2DMultiObjectiveV1Problem,
) -> EvaluatorIdentity:
    return EvaluatorIdentity(
        evaluator_id=HEAT2D_SELECTED_ACTION_EVALUATOR_ID,
        evaluator_version=HEAT2D_SELECTED_ACTION_EVALUATOR_VERSION,
        evaluator_context_sha256=_evaluator_definition_sha256(problem),
    )


@dataclass(frozen=True, slots=True)
class Heat2DResidualPhenotypeProjection:
    """Use the exact field digest consumed by direct-v3 as phenotype identity."""

    resolution: int = 1001
    projection_id: str = HEAT2D_RESIDUAL_PHENOTYPE_PROJECTION_ID
    projection_version: int = (
        HEAT2D_RESIDUAL_PHENOTYPE_PROJECTION_VERSION
    )
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.resolution) is not int or self.resolution < 3:
            raise ValueError("resolution must be at least three")
        object.__setattr__(
            self,
            "definition_sha256",
            heat2d_residual_phenotype_projection_definition_sha256(
                self.resolution
            ),
        )

    def project(self, configuration: FrozenJsonObject) -> str:
        if type(configuration) is not FrozenJsonObject:
            raise TypeError("configuration must be a frozen object")
        identity = Heat2DPhenotypeIdentityPolicy(
            resolution=self.resolution
        ).identify(thaw_json(configuration))
        return identity.value_sha256


@dataclass(slots=True)
class Heat2DSelectedMaterializedActionEvaluator:
    """Serialize broker-selected candidates through qualified direct-v3."""

    problem: Heat2DMultiObjectiveV1Problem = field(
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
        default=HEAT2D_SELECTED_ACTION_EVALUATOR_ID,
    )
    evaluator_version: int = field(
        init=False,
        default=HEAT2D_SELECTED_ACTION_EVALUATOR_VERSION,
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
        if type(self.problem) is not Heat2DMultiObjectiveV1Problem:
            raise TypeError("problem must be the exact Heat2D Pareto problem")
        if not isinstance(
            self.phenotype_projection,
            MaterializedPhenotypeProjectionPort,
        ):
            raise TypeError(
                "phenotype_projection must implement its runtime port"
            )
        if (
            self.phenotype_projection.projection_id
            != HEAT2D_RESIDUAL_PHENOTYPE_PROJECTION_ID
            or self.phenotype_projection.projection_version
            != HEAT2D_RESIDUAL_PHENOTYPE_PROJECTION_VERSION
            or self.phenotype_projection.definition_sha256
            != heat2d_residual_phenotype_projection_definition_sha256(
                self.problem.settings.resolution
            )
        ):
            raise ValueError(
                "Heat evaluator requires its qualified phenotype projection"
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
        self.definition_sha256 = _evaluator_definition_sha256(
            self.problem
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
            started = time.perf_counter_ns()
            configuration = thaw_json(action.configuration)
            result = await asyncio.to_thread(
                self.problem.evaluate_detailed,
                configuration,
            )
            elapsed_s = (time.perf_counter_ns() - started) / 1e9
            raw_objectives = tuple(
                (
                    objective.name,
                    float(result.objective_values[objective.name]),
                )
                for objective in self.problem.objectives
            )
            resolution = resolve_objectives(
                heat2d_objective_resolution(),
                ObjectiveResolutionRequest(
                    configuration=action.configuration,
                    objectives=self.problem.objectives,
                    raw_objectives=raw_objectives,
                ),
            )
            phenotype = Heat2DPhenotypeIdentityPolicy(
                resolution=self.problem.settings.resolution
            ).identify(configuration)
            if (
                phenotype.value_sha256
                != action.phenotype_identity_sha256
            ):
                raise RuntimeError(
                    "evaluated Heat phenotype differs from the sealed action"
                )
            detailed = DetailedEvaluation(
                phenotype=phenotype,
                payload=DetailedEvaluationPayload(
                    failure=None,
                    objectives=raw_objectives,
                    violations=(),
                    checks=(),
                    receipt=None,
                    evaluator=_evaluator_identity(self.problem),
                ),
                timings=EvaluationTimings(
                    total_wall_seconds=float(elapsed_s)
                ),
            )
            self._evaluation_count += 1
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
                objectives=resolution.decision_objectives,
                valid=True,
                generation=action.context.decision_index,
                label=(
                    f"heat_residual_d{action.context.decision_index:04d}_"
                    f"{self._evaluation_count:04d}"
                ),
                parent_ids=action.parent_ids,
                design_rationale=(
                    "Trusted materialization selected by the generic residual "
                    "portfolio broker."
                ),
                detailed_evaluation=detailed,
                objective_resolution_receipt=resolution,
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
                # Observation sinks are an interchange boundary.  Domain
                # records retain FrozenJsonObject internally, but a sink must
                # receive an exact JSON tree so it can durably encode the
                # selected evaluation without workload-specific codecs.
                action_record["configuration"] = thaw_json(
                    action.configuration
                )
                self.observation_sink(
                    {
                        "schema_version": 1,
                        "action": action_record,
                        "candidate_id": candidate.candidate_id.value,
                        "objectives": [
                            {
                                "metric_id": metric_id,
                                "value_hex": value.hex(),
                            }
                            for metric_id, value in candidate.objectives
                        ],
                        "raw_objectives": [
                            {
                                "metric_id": metric_id,
                                "value_hex": value.hex(),
                            }
                            for metric_id, value in raw_objectives
                        ],
                        "objective_resolution": resolution.to_record(),
                        "phenotype_identity_sha256": (
                            phenotype.value_sha256
                        ),
                        "detailed_evaluation_sha256": (
                            detailed.evidence_sha256
                        ),
                        "direct_v3_manifest": result.direct_v3.manifest,
                        "total_wall_seconds": float(elapsed_s),
                        "selected_only_authoritative_evaluation": True,
                    }
                )
            return evaluated


__all__ = [
    "HEAT2D_OBJECTIVE_RESOLUTION_DEFINITION_SHA256",
    "HEAT2D_RESIDUAL_PHENOTYPE_PROJECTION_DEFINITION_SHA256",
    "HEAT2D_RESIDUAL_PHENOTYPE_PROJECTION_ID",
    "HEAT2D_RESIDUAL_PHENOTYPE_PROJECTION_VERSION",
    "HEAT2D_SELECTED_ACTION_EVALUATOR_ID",
    "HEAT2D_SELECTED_ACTION_EVALUATOR_VERSION",
    "Heat2DResidualPhenotypeProjection",
    "Heat2DSelectedMaterializedActionEvaluator",
    "heat2d_objective_resolution",
    "heat2d_residual_phenotype_projection_definition_sha256",
]
