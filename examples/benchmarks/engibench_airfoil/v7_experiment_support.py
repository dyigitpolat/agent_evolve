"""Domain-local orchestration for the Airfoil-v7 seven-call kill test.

The generic engine knows nothing about Airfoil.  This module supplies only the
benchmark-owned parent materialization, finite catalogs, planner, prompt
renderer, and provider-free verification doubles needed by artifact 95.

No function in this module reads credentials or launches CFD.  The live runner
remains fail-closed until a separately frozen launch manifest authorizes it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import struct
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from agent_evolve.agentic import (
    AgenticBenchmark,
    AgenticCallTelemetry,
    AgenticOptimizerComposition,
    ArrayIndex,
    ContrastShardedReflectionWorkflow,
    DetailedEvaluationPayload,
    DeterministicIdFactory,
    EvaluationCheck,
    EvaluationCheckStatus,
    EvaluatorIdentity,
    FiniteVariationContract,
    FiniteVariationSelectionDraft,
    FixedStructuredOutputBudgetPolicy,
    FrozenWaveReward,
    G1ReflectionFeedbackInterceptor,
    GenerationPlan,
    HeldOutASNAssignmentCommitment,
    HeldOutASNPlannerAdapter,
    HeldOutAssignmentUnavailable,
    HeldOutAssignmentUnavailableReason,
    InsightDraft,
    InsightMemoryBank,
    InvocationPlan,
    JsonPath,
    MetricEffectDirection,
    MetricEffectPrediction,
    MutationContract,
    MutationResponseMode,
    ObjectKey,
    OperatorKind,
    OptimizerBudget,
    OptimizerResult,
    OptimizerSlot,
    OptimizerState,
    ReflectionGenerationRequest,
    ReflectionGenerationResult,
    ReflectionInsightContract,
    ReflectionRowProjectionBinding,
    ReflectedCardMailbox,
    VariationGenerationRequest,
    VariationGenerationResult,
    artifact_ref_for_bytes,
    compose_agentic_optimizer,
    default_evidence_prompt,
    freeze_json,
    register_neutral_sham_card,
    typed_json_sha256,
)
from examples.benchmarks.engibench_airfoil.problem_def import (
    candidate_sha256,
    normalize_candidate,
)
from examples.benchmarks.engibench_airfoil.v7_contract import (
    AIRFOIL_V7_ARCHIVE_RELATION,
    AIRFOIL_V7_REWARD_BINDING,
    ARCHIVE_DEFINITION_SHA256,
    REWARD_DEFINITION_SHA256,
    TASK_SHA256,
    AirfoilV7PhenotypeIdentityPolicy,
    decoded_float32_le_bytes,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    OBJECTIVE_NAME,
    VIOLATION_NAME,
    AirfoilV7Problem,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    SHAPE_CATALOG_DEFINITION_SHA256,
    TRIM_CATALOG_DEFINITION_SHA256,
    UNION_CATALOG_DEFINITION_SHA256,
    AirfoilV7ShapeVariationCatalog,
    AirfoilV7TrimVariationCatalog,
    AirfoilV7UnionVariationCatalog,
)


MODEL = "deepseek/deepseek-v4-pro"
PLANNER_POLICY_ID = "airfoil_v7_seven_call_sharded_asn"
PLANNER_POLICY_VERSION = 3
DIAGNOSTIC_SLOT_IDS = ("D-S", "D-T")
HELD_OUT_SLOT_IDS = ("A", "S", "N")
# This is the selected route's published completion ceiling, not an expected
# response length.  The live launch independently binds and verifies the
# dated StreamLake capability snapshot before any provider dispatch.
MAX_OUTPUT_TOKENS = 384_000
STRUCTURED_OUTPUT_BUDGET_POLICY = FixedStructuredOutputBudgetPolicy(
    proposal_max_output_tokens=MAX_OUTPUT_TOKENS,
    reflection_max_output_tokens=MAX_OUTPUT_TOKENS,
)
# The deployed external evaluator uses a fixed container/receipt boundary and
# is not safe for parallel CFD. Provider proposals within each wave still run
# concurrently before they queue at this single evaluator slot.
EVALUATOR_CONCURRENCY = 1
DIAGNOSTIC_SHAPE_OPTION_ID = "shape.camber_aft.p0015"
DIAGNOSTIC_TRIM_OPTION_ID = "trim.p050.n025.n050"
SHAM_OPTION_ID = "trim.p025.n025.p050"
OPTIMIZER_BUDGET = OptimizerBudget(
    max_unique_evaluations=7,
    max_logical_llm_calls=7,
    max_generations=2,
)
REFLECTION_INSIGHT_CONTRACT = ReflectionInsightContract(
    required_metric_ids=(
        "objective:normalized_multipoint_drag",
        "violation:normalized_lift_equality",
    ),
    allowed_option_families=("shape_only", "trim_only"),
    allowed_option_ids=(
        DIAGNOSTIC_SHAPE_OPTION_ID,
        DIAGNOSTIC_TRIM_OPTION_ID,
    ),
)
SHAM_INSIGHT_CONTRACT = ReflectionInsightContract(
    required_metric_ids=REFLECTION_INSIGHT_CONTRACT.required_metric_ids,
    allowed_option_families=("trim_only",),
    allowed_option_ids=(SHAM_OPTION_ID,),
)


def structured_output_budget_policy_record() -> dict[str, object]:
    """Return the frozen provider-neutral output allocation for this study."""

    return {
        "policy_id": STRUCTURED_OUTPUT_BUDGET_POLICY.policy_id,
        "policy_version": STRUCTURED_OUTPUT_BUDGET_POLICY.policy_version,
        "proposal_max_output_tokens": (
            STRUCTURED_OUTPUT_BUDGET_POLICY.proposal_max_output_tokens
        ),
        "reflection_max_output_tokens": (
            STRUCTURED_OUTPUT_BUDGET_POLICY.reflection_max_output_tokens
        ),
        "ceiling_semantics": "provider_maximum_not_expected_usage",
    }

MEMORY_CARD_BEGIN = "<MEMORY_CARD>"
MEMORY_CARD_END = "</MEMORY_CARD>"
MEMORY_CARD_MASK = "<ONE_MEMORY_CARD>"

NEUTRAL_PARENT: dict[str, object] = {
    "representation_id": "external_bernstein_y_panel_v1",
    "upper_coefficients": [0.0] * 10,
    "lower_coefficients": [0.0] * 10,
    "alpha_deg": [2.5, 2.5, 2.5],
}
HELD_OUT_PARENT_NONCE = 0
HELD_OUT_PARENT_CANDIDATE_SHA256 = (
    "4e27383154ae5f2ff63c79c9dd9ff57a62031b5898978ea20a86e8b455a4955a"
)
HELD_OUT_PARENT_TYPED_SHA256 = (
    "61503bf31edfaa87b473184553fd245ea18747fd33c89361f94fd27038b67506"
)

_HELD_OUT_DOMAIN = b"agent-evolve:airfoil-v7-heldout-parent:v1\x00"
_INITIAL_AREA = 0.04632803061919573
_AREA_RATIO_BOUNDS = (0.8873697327569672, 1.2)
_RAW_X_BOUNDS = (-1.0e-3, 1.001)
_RAW_CHORD_BOUNDS = (0.99, 1.01)
_PREPROCESSED_Y_BOUNDS = (-0.25, 0.25)
_GEOMETRY_ATOL = 1.0e-10
_MIN_SEGMENT_LENGTH = 1.0e-8
_MIN_AREA = 1.0e-8

_OFFLINE_EVALUATOR_CONTEXT = hashlib.sha256(
    b"agent-evolve:airfoil-v7-offline-evaluator:v1"
).hexdigest()
OFFLINE_EVALUATOR_IDENTITY = EvaluatorIdentity(
    evaluator_id="airfoil_v7_provider_free_fixture",
    evaluator_version=1,
    evaluator_context_sha256=_OFFLINE_EVALUATOR_CONTEXT,
)


def _canonical_sha256(domain: bytes, value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(domain + encoded).hexdigest()


_REFLECTION_PROJECTION_DEFINITION = {
    "kind": "remove_prior_model_rationale_only",
    "removed_path": "$.candidate.design_rationale",
    "preserved_machine_derived_contrasts": True,
    "preserved_invocation_identity": True,
}
REFLECTION_PROJECTION_DEFINITION_SHA256 = _canonical_sha256(
    b"agent-evolve:airfoil-v7-reflection-projection:v1\x00",
    _REFLECTION_PROJECTION_DEFINITION,
)


def _project_airfoil_reflection_row(
    row: Mapping[str, object],
) -> Mapping[str, object]:
    projected = dict(row)
    candidate = projected.get("candidate")
    if type(candidate) is dict:
        candidate_record = dict(candidate)
        candidate_record.pop("design_rationale", None)
        projected["candidate"] = candidate_record
    return projected


AIRFOIL_V7_REFLECTION_PROJECTION = ReflectionRowProjectionBinding(
    project=_project_airfoil_reflection_row,
    policy_id="airfoil_v7_remove_prior_rationale",
    policy_version=1,
    definition_sha256=REFLECTION_PROJECTION_DEFINITION_SHA256,
)


def _path(field: str, index: int) -> JsonPath:
    return JsonPath((ObjectKey(field), ArrayIndex(index)))


SHAPE_MUTATION_CONTRACT = MutationContract(
    editable_paths=tuple(
        _path(field, index)
        for field in ("lower_coefficients", "upper_coefficients")
        for index in range(1, 9)
    ),
    max_changed_paths=8,
    max_operations=8,
    allow_abstention=False,
)
TRIM_MUTATION_CONTRACT = MutationContract(
    editable_paths=tuple(_path("alpha_deg", index) for index in range(3)),
    max_changed_paths=3,
    max_operations=3,
    allow_abstention=False,
)
UNION_MUTATION_CONTRACT = MutationContract(
    editable_paths=(
        *TRIM_MUTATION_CONTRACT.editable_paths,
        *SHAPE_MUTATION_CONTRACT.editable_paths,
    ),
    max_changed_paths=8,
    max_operations=8,
    allow_abstention=False,
)


class NoCFDValidationError(ValueError):
    """A deterministic representation or geometry check rejected a parent."""


@dataclass(frozen=True, slots=True)
class NoCFDValidation:
    candidate_sha256: str
    decoded_coords_sha256: str
    area: float
    area_ratio: float
    preprocessed_area: float
    minimum_segment_length: float
    checks: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_sha256": self.candidate_sha256,
            "decoded_coords_sha256": self.decoded_coords_sha256,
            "area": self.area,
            "area_ratio": self.area_ratio,
            "preprocessed_area": self.preprocessed_area,
            "minimum_segment_length": self.minimum_segment_length,
            "checks": list(self.checks),
        }


@dataclass(frozen=True, slots=True)
class HeldOutParentMaterialization:
    nonce: int
    candidate: dict[str, object]
    candidate_sha256: str
    typed_configuration_sha256: str
    validation: NoCFDValidation
    rejected_nonces: tuple[tuple[int, str], ...]

    def to_record(self) -> dict[str, object]:
        return {
            "nonce": self.nonce,
            "candidate": self.candidate,
            "candidate_sha256": self.candidate_sha256,
            "typed_configuration_sha256": self.typed_configuration_sha256,
            "validation": self.validation.to_record(),
            "rejected_nonces": [list(item) for item in self.rejected_nonces],
        }


def _held_out_sign(nonce: int, field: str, index: int) -> float:
    digest = hashlib.sha256(
        _HELD_OUT_DOMAIN
        + TASK_SHA256.encode("ascii")
        + b"\x00"
        + str(nonce).encode("ascii")
        + b"\x00"
        + field.encode("ascii")
        + b"\x00"
        + str(index).encode("ascii")
    ).digest()
    return 1.0 if digest[0] & 1 else -1.0


def held_out_candidate_for_nonce(nonce: int) -> dict[str, object]:
    """Materialize artifact-95's outcome-blind held-out parent rule."""

    if type(nonce) is not int or nonce < 0:
        raise ValueError("nonce must be a non-negative exact integer")
    upper = [0.0] * 10
    lower = [0.0] * 10
    for index in range(1, 9):
        upper[index] = _held_out_sign(nonce, "upper", index) * 0.0015
        lower[index] = _held_out_sign(nonce, "lower", index) * 0.0015
    alpha = [2.5 + _held_out_sign(nonce, "alpha", index) * 0.25 for index in range(3)]
    return normalize_candidate(
        {
            "representation_id": "external_bernstein_y_panel_v1",
            "upper_coefficients": upper,
            "lower_coefficients": lower,
            "alpha_deg": alpha,
        }
    )


def _shoelace(x_values: list[float], y_values: list[float]) -> float:
    return (
        abs(
            sum(
                x_values[index] * y_values[(index + 1) % len(x_values)]
                - y_values[index] * x_values[(index + 1) % len(x_values)]
                for index in range(len(x_values))
            )
        )
        / 2.0
    )


def _is_blunted(x_values: list[float], tolerance: float = 1.0e-5) -> bool:
    x_gate = max(x_values) * 0.99
    matches: set[int] = set()
    size = len(x_values)
    for index, value in enumerate(x_values):
        if abs(value - x_values[(index + 1) % size]) < tolerance:
            matches.add(index)
        if abs(value - x_values[(index - 1) % size]) < tolerance:
            matches.add(index)
    return len(tuple(index for index in matches if x_values[index] >= x_gate)) > 1


def _trailing_edge_indices(
    x_values: list[float],
    tolerance: float,
) -> list[int]:
    x_gate = max(x_values) * 0.99
    matches: set[int] = set()
    size = len(x_values)
    for index, value in enumerate(x_values):
        if abs(value - x_values[(index + 1) % size]) < tolerance:
            matches.add(index)
        if abs(value - x_values[(index - 1) % size]) < tolerance:
            matches.add(index)
    return sorted(index for index in matches if x_values[index] >= x_gate)


def _cross(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _subtract(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float]:
    return left[0] - right[0], left[1] - right[1]


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    if abs(_cross(_subtract(end, start), _subtract(point, start))) > _GEOMETRY_ATOL:
        return False
    return all(
        min(start[axis], end[axis]) - _GEOMETRY_ATOL
        <= point[axis]
        <= max(start[axis], end[axis]) + _GEOMETRY_ATOL
        for axis in (0, 1)
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> bool:
    first_direction = _subtract(first_end, first_start)
    second_direction = _subtract(second_end, second_start)
    o1 = _cross(first_direction, _subtract(second_start, first_start))
    o2 = _cross(first_direction, _subtract(second_end, first_start))
    o3 = _cross(second_direction, _subtract(first_start, second_start))
    o4 = _cross(second_direction, _subtract(first_end, second_start))
    first_straddles = (o1 > _GEOMETRY_ATOL and o2 < -_GEOMETRY_ATOL) or (
        o1 < -_GEOMETRY_ATOL and o2 > _GEOMETRY_ATOL
    )
    second_straddles = (o3 > _GEOMETRY_ATOL and o4 < -_GEOMETRY_ATOL) or (
        o3 < -_GEOMETRY_ATOL and o4 > _GEOMETRY_ATOL
    )
    return (
        (first_straddles and second_straddles)
        or (
            abs(o1) <= _GEOMETRY_ATOL
            and _point_on_segment(second_start, first_start, first_end)
        )
        or (
            abs(o2) <= _GEOMETRY_ATOL
            and _point_on_segment(second_end, first_start, first_end)
        )
        or (
            abs(o3) <= _GEOMETRY_ATOL
            and _point_on_segment(first_start, second_start, second_end)
        )
        or (
            abs(o4) <= _GEOMETRY_ATOL
            and _point_on_segment(first_end, second_start, second_end)
        )
    )


def _first_self_intersection(
    points: list[tuple[float, float]],
) -> tuple[int, int] | None:
    if math.dist(points[0], points[-1]) <= _GEOMETRY_ATOL:
        points = points[:-1]
    count = len(points)
    for first in range(count):
        for second in range(first + 1, count):
            if second == first + 1 or (first == 0 and second == count - 1):
                continue
            if _segments_intersect(
                points[first],
                points[(first + 1) % count],
                points[second],
                points[(second + 1) % count],
            ):
                return first, second
    return None


def validate_frozen_no_cfd_candidate(configuration: object) -> NoCFDValidation:
    """Mirror the frozen v1 representation/geometry checks without CFD.

    The decoder bytes are the same bytes bound by the v7 phenotype policy.  The
    checks mirror adapter-v1's raw range, EngiBench preprocessing, closure,
    nondegenerate segment, nonintersection, positive-area, and task area-ratio
    gates.  No task outcome or stored optimum is available here.
    """

    candidate = normalize_candidate(configuration)
    raw = decoded_float32_le_bytes(candidate)
    values = struct.unpack("<384f", raw)
    x_values = list(values[:192])
    y_values = list(values[192:])
    if not all(math.isfinite(value) for value in values):
        raise NoCFDValidationError("decoded coordinates are non-finite")
    x_min = min(x_values)
    x_max = max(x_values)
    if x_min < _RAW_X_BOUNDS[0] or x_max > _RAW_X_BOUNDS[1]:
        raise NoCFDValidationError("raw x range is outside adapter bounds")
    chord = x_max - x_min
    if not _RAW_CHORD_BOUNDS[0] <= chord <= _RAW_CHORD_BOUNDS[1]:
        raise NoCFDValidationError("raw chord is outside adapter bounds")

    area = _shoelace(x_values, y_values)
    ratio = area / _INITIAL_AREA
    if not _AREA_RATIO_BOUNDS[0] <= ratio <= _AREA_RATIO_BOUNDS[1]:
        raise NoCFDValidationError("decoded task area ratio is outside bounds")

    blunted = _is_blunted(x_values)
    leading = min(range(len(x_values)), key=x_values.__getitem__)
    xcut = 0.99 if blunted else 1.0
    processed_x = [xcut * (value - x_min) / chord for value in x_values]
    processed_y = [value - y_values[leading] for value in y_values]
    processed_x[0] = xcut
    processed_x[-1] = xcut
    processed_y[-1] = processed_y[0]
    if blunted:
        trailing = _trailing_edge_indices(processed_x, 1.0e-5)
        tolerance = 1.0e-4
        while len(trailing) < 6:
            trailing = _trailing_edge_indices(processed_x, tolerance)
            tolerance *= 1.5
            if tolerance > 1.0e-3:
                break
        deleted = set(trailing[1:-1])
        if deleted:
            # Adapter v1 rejects the resulting cardinality before any CFD.
            raise NoCFDValidationError("blunted preprocessing changes cardinality")
    if (
        min(processed_y) < _PREPROCESSED_Y_BOUNDS[0]
        or max(processed_y) > _PREPROCESSED_Y_BOUNDS[1]
    ):
        raise NoCFDValidationError("preprocessed y range is outside bounds")
    points = list(zip(processed_x, processed_y, strict=True))
    closure_gap = math.dist(points[0], points[-1])
    if closure_gap > _GEOMETRY_ATOL:
        raise NoCFDValidationError("preprocessed curve is not closed")
    segment_lengths = [
        math.dist(points[index], points[index + 1]) for index in range(len(points) - 1)
    ]
    minimum_segment = min(segment_lengths)
    if minimum_segment < _MIN_SEGMENT_LENGTH:
        raise NoCFDValidationError("preprocessed curve has a degenerate segment")
    intersection = _first_self_intersection(points)
    if intersection is not None:
        raise NoCFDValidationError(
            "preprocessed curve self-intersects at "
            f"edges {intersection[0]} and {intersection[1]}"
        )
    processed_area = _shoelace(processed_x, processed_y)
    if not math.isfinite(processed_area) or processed_area <= _MIN_AREA:
        raise NoCFDValidationError("preprocessed area is nonpositive")
    return NoCFDValidation(
        candidate_sha256=candidate_sha256(candidate),
        decoded_coords_sha256=hashlib.sha256(raw).hexdigest(),
        area=area,
        area_ratio=ratio,
        preprocessed_area=processed_area,
        minimum_segment_length=minimum_segment,
        checks=(
            "exact_representation_and_bounds",
            "decoded_shape_and_finiteness",
            "raw_x_range",
            "raw_chord_span",
            "task_area_ratio",
            "preprocessed_y_range",
            "preprocessed_closure",
            "preprocessed_segment_length",
            "preprocessed_no_self_intersection",
            "preprocessed_positive_area",
        ),
    )


def materialize_held_out_parent(
    *,
    validator: Callable[[object], NoCFDValidation] = validate_frozen_no_cfd_candidate,
    max_nonces: int = 10_000,
) -> HeldOutParentMaterialization:
    """Return the first artifact-95 nonce passing the outcome-blind validator."""

    if not callable(validator):
        raise TypeError("validator must be callable")
    if type(max_nonces) is not int or max_nonces <= 0:
        raise ValueError("max_nonces must be a positive exact integer")
    neutral_hash = candidate_sha256(NEUTRAL_PARENT)
    rejected: list[tuple[int, str]] = []
    for nonce in range(max_nonces):
        candidate = held_out_candidate_for_nonce(nonce)
        if candidate_sha256(candidate) == neutral_hash:
            rejected.append((nonce, "equals_diagnostic_parent"))
            continue
        try:
            validation = validator(candidate)
        except (TypeError, ValueError) as exc:
            rejected.append((nonce, type(exc).__name__))
            continue
        frozen = freeze_json(candidate)
        return HeldOutParentMaterialization(
            nonce=nonce,
            candidate=candidate,
            candidate_sha256=candidate_sha256(candidate),
            typed_configuration_sha256=typed_json_sha256(frozen),
            validation=validation,
            rejected_nonces=tuple(rejected),
        )
    raise NoCFDValidationError("no held-out parent passed within max_nonces")


def _memory_card_payload(prompt: str) -> str | None:
    start = prompt.find(MEMORY_CARD_BEGIN)
    end = prompt.find(MEMORY_CARD_END)
    if start < 0 and end < 0:
        return None
    if start < 0 or end < 0 or end <= start:
        raise ValueError("prompt has a malformed MEMORY_CARD payload")
    payload_start = start + len(MEMORY_CARD_BEGIN)
    if prompt.find(MEMORY_CARD_BEGIN, payload_start) >= 0:
        raise ValueError("prompt has multiple MEMORY_CARD payloads")
    if prompt.find(MEMORY_CARD_END, end + len(MEMORY_CARD_END)) >= 0:
        raise ValueError("prompt has multiple MEMORY_CARD payloads")
    return prompt[payload_start:end].strip()


def mask_memory_card(prompt: str) -> str:
    """Replace the complete delimited card with artifact-95's sentinel."""

    payload = _memory_card_payload(prompt)
    if payload is None:
        return prompt
    start = prompt.index(MEMORY_CARD_BEGIN)
    end = prompt.index(MEMORY_CARD_END) + len(MEMORY_CARD_END)
    return prompt[:start] + MEMORY_CARD_MASK + prompt[end:]


def airfoil_v7_prompt_builder(
    problem_description: str,
    prepared: Any,
    selected_insights: tuple[dict[str, object], ...],
) -> str:
    """Wrap the generic renderer with one exactly delimited held-out card."""

    base = default_evidence_prompt(problem_description, prepared, ())
    diagnostic_target = {
        "airfoil_v7_g1_shape": DIAGNOSTIC_SHAPE_OPTION_ID,
        "airfoil_v7_g1_trim": DIAGNOSTIC_TRIM_OPTION_ID,
    }.get(prepared.plan.label)
    if diagnostic_target is not None:
        contract = prepared.plan.finite_variation_contract
        if contract is None:
            raise RuntimeError("targeted Airfoil diagnostic lacks a finite contract")
        contract.resolve(diagnostic_target)
        base = "\n".join(
            (
                base,
                "",
                "PROSPECTIVELY TARGETED DIAGNOSTIC ACTION",
                f"Select exact option_id {diagnostic_target}. The complete sealed "
                "palette above remains visible and unchanged; this exact target "
                "is the preregistered diagnostic intervention for this slot.",
            )
        )
    requirement = prepared.plan.insight_treatment_requirement
    if requirement is not None:
        # The generic renderer exposes the exact requirement, compatible
        # families, and receipt hashes.  Airfoil-v7 deliberately blinds those
        # arm-specific values: the assigned card already carries its exact ID
        # and recommended families inside the maskable payload, while the
        # shared administration instruction below remains byte-identical.
        treatment_heading = "\nASSIGNED INSIGHT TREATMENT CONTRACT"
        mutation_heading = "\nMACHINE MUTATION CONTRACT"
        if (
            base.count(treatment_heading) != 1
            or base.count(mutation_heading) != 1
        ):
            raise RuntimeError("generic prompt renderer treatment seam changed")
        treatment_start = base.index(treatment_heading)
        mutation_start = base.index(mutation_heading, treatment_start)
        base = base[:treatment_start] + base[mutation_start:]
    if not selected_insights:
        return base
    if len(selected_insights) != 1:
        raise ValueError("Airfoil-v7 held-out plans require exactly one card")
    source = selected_insights[0]
    required = {
        "insight_id",
        "claim",
        "trigger",
        "mechanism",
        "affected_paths",
        "effect_predictions",
        "recommended_option_families",
        "recommended_option_ids",
        "action_template",
        "falsification_condition",
    }
    if not required.issubset(source):
        raise ValueError("held-out card lacks the blinded substantive schema")
    # Deliberately omit origin, lifecycle, retrievability, confidence, evidence
    # presence/lineage, semantic relations, and assignment-arm metadata.  Those
    # remain in durable engine traces but cannot reveal A/S/N arm identity to
    # the proposal model.
    blinded_card = {
        key: source[key]
        for key in (
            "insight_id",
            "claim",
            "trigger",
            "mechanism",
            "affected_paths",
            "effect_predictions",
            "recommended_option_families",
            "recommended_option_ids",
            "action_template",
            "falsification_condition",
        )
    }
    needle = "SELECTED MEMORY HYPOTHESES\nNone. Set claimed_insight_ids to []."
    if base.count(needle) != 1:
        raise RuntimeError("generic prompt renderer memory seam changed")
    payload = json.dumps(
        blinded_card,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    administration_instruction = (
        "This is an enforced isolated transfer treatment. The origin trigger "
        "scopes its source evidence; preflight guarantees a compatible action "
        "on the current parent. Select the exact singleton option ID named in "
        "recommended_option_ids; the assigned hypothesis must influence that "
        "selection, and claimed_insight_ids must contain its exact insight_id "
        "to record administration."
        if requirement is not None
        and requirement.claim_mode.value == "exact_required"
        else (
            "Use this one assigned hypothesis only when its trigger applies; "
            "report its insight_id only if it affected the selected option."
        )
    )
    replacement = "\n".join(
        (
            "MEMORY_CARD",
            MEMORY_CARD_BEGIN,
            payload,
            MEMORY_CARD_END,
            administration_instruction,
        )
    )
    return base.replace(needle, replacement)


def _wave_reward(
    state: OptimizerState,
    *,
    generation: int,
    parent_configuration_sha256: str,
) -> FrozenWaveReward:
    snapshot = _canonical_sha256(
        b"agent-evolve:airfoil-v7-wave-reward:v1\x00",
        {
            "generation": generation,
            "archive_snapshot_sha256": state.archive_snapshot_hash,
            "parent_configuration_sha256": parent_configuration_sha256,
            "local_reward_definition_sha256": REWARD_DEFINITION_SHA256,
            "archive_relation_definition_sha256": ARCHIVE_DEFINITION_SHA256,
        },
    )
    return FrozenWaveReward(
        binding=AIRFOIL_V7_REWARD_BINDING,
        archive_snapshot_hash=state.archive_snapshot_hash,
        reward_snapshot_hash=snapshot,
    )


def _finite_plan(
    *,
    parent: Any,
    generation: int,
    label: str,
    allowed_top_level: tuple[str, ...],
    mutation_contract: MutationContract,
    finite_contract: FiniteVariationContract,
) -> InvocationPlan:
    return InvocationPlan(
        operator_kind=OperatorKind.TYPED_MUTATION,
        parents=(parent,),
        generation=generation,
        label=label,
        allowed_top_level=allowed_top_level,
        phase="airfoil_v7_reflective_feedback",
        mutation_contract=mutation_contract,
        mutation_response_mode=MutationResponseMode.FINITE_OPTION_SELECTION_V1,
        finite_variation_contract=finite_contract,
    )


@dataclass(slots=True)
class AirfoilV7SevenCallPlanner:
    """Precommit P_D/P_H and issue exactly the artifact-95 G1 and G2 waves."""

    benchmark: AgenticBenchmark
    held_out_adapter: HeldOutASNPlannerAdapter
    diagnostic_parent_sha256: str
    held_out_parent_sha256: str
    early_stop_reason: str | None = None
    early_stop_reason_code: str | None = None
    held_out_assignment_commitment: HeldOutASNAssignmentCommitment | None = None

    def _parent(self, state: OptimizerState, candidate_hash: str):
        matches = tuple(
            candidate
            for candidate in state.candidates
            if candidate_sha256(candidate.configuration_dict) == candidate_hash
            and candidate.generation == 0
        )
        if len(matches) != 1:
            raise ValueError("planner state lacks one exact frozen seed parent")
        return matches[0]

    def plan(
        self,
        state: OptimizerState,
        budget: OptimizerBudget,
    ) -> GenerationPlan:
        if budget != OPTIMIZER_BUDGET:
            raise ValueError("Airfoil-v7 planner received a different hard budget")
        if state.generation == 0:
            parent = self._parent(state, self.diagnostic_parent_sha256)
            shape = self.benchmark.bind_finite_variation(
                "airfoil_v7_shape", parent.configuration
            )
            trim = self.benchmark.bind_finite_variation(
                "airfoil_v7_trim", parent.configuration
            )
            slots = (
                OptimizerSlot.model(
                    slot_id="D-S",
                    role="diagnostic_shape_only",
                    plan=_finite_plan(
                        parent=parent,
                        generation=1,
                        label="airfoil_v7_g1_shape",
                        allowed_top_level=(
                            "lower_coefficients",
                            "upper_coefficients",
                        ),
                        mutation_contract=SHAPE_MUTATION_CONTRACT,
                        finite_contract=shape,
                    ),
                ),
                OptimizerSlot.model(
                    slot_id="D-T",
                    role="diagnostic_trim_only",
                    plan=_finite_plan(
                        parent=parent,
                        generation=1,
                        label="airfoil_v7_g1_trim",
                        allowed_top_level=("alpha_deg",),
                        mutation_contract=TRIM_MUTATION_CONTRACT,
                        finite_contract=trim,
                    ),
                ),
            )
            return GenerationPlan(
                generation=1,
                slots=slots,
                reward=_wave_reward(
                    state,
                    generation=1,
                    parent_configuration_sha256=parent.occurrence.configuration_hash,
                ),
                planner_policy_id=PLANNER_POLICY_ID,
                planner_policy_version=PLANNER_POLICY_VERSION,
                metadata=tuple(
                    sorted(
                        (
                            ("diagnostic_parent", self.diagnostic_parent_sha256),
                            ("shape_contract", shape.identity_sha256),
                            ("trim_contract", trim.identity_sha256),
                            ("wave", "g1_diagnostic"),
                        )
                    )
                ),
            )
        if state.generation != 1:
            raise ValueError("Airfoil-v7 planner supports exactly two generations")
        parent = self._parent(state, self.held_out_parent_sha256)
        union = self.benchmark.bind_finite_variation(
            "airfoil_v7_union", parent.configuration
        )
        bases = tuple(
            _finite_plan(
                parent=parent,
                generation=2,
                label=label,
                allowed_top_level=(
                    "alpha_deg",
                    "lower_coefficients",
                    "upper_coefficients",
                ),
                mutation_contract=UNION_MUTATION_CONTRACT,
                finite_contract=union,
            )
            for label in (
                "airfoil_v7_g2_adaptive",
                "airfoil_v7_g2_score_swapped",
                "airfoil_v7_g2_sham",
            )
        )
        try:
            bound = self.held_out_adapter.bind_plans(
                state,
                adaptive_base=bases[0],
                score_swapped_base=bases[1],
                sham_base=bases[2],
            )
        except HeldOutAssignmentUnavailable as exc:
            self.held_out_assignment_commitment = None
            self.early_stop_reason = str(exc)
            self.early_stop_reason_code = exc.reason.value
            return GenerationPlan(
                generation=2,
                slots=(),
                reward=_wave_reward(
                    state,
                    generation=2,
                    parent_configuration_sha256=parent.occurrence.configuration_hash,
                ),
                planner_policy_id=PLANNER_POLICY_ID,
                planner_policy_version=PLANNER_POLICY_VERSION,
                metadata=tuple(
                    sorted(
                        (
                            ("early_stop", exc.reason.value),
                            ("union_contract", union.identity_sha256),
                            ("wave", "g2_held_out"),
                        )
                    )
                ),
            )
        assignments = bound.assignments
        if self.held_out_assignment_commitment is not None:
            raise RuntimeError("held-out assignment commitment was already published")
        self.held_out_assignment_commitment = bound.assignment_commitment
        slots = tuple(
            OptimizerSlot.model(slot_id=slot_id, role=role, plan=plan)
            for slot_id, role, plan in zip(
                HELD_OUT_SLOT_IDS,
                ("adaptive", "score_swapped", "sham"),
                (bound.adaptive, bound.score_swapped, bound.sham),
                strict=True,
            )
        )
        assignment_metadata = (
            ("adaptive_insight", assignments.adaptive.reference.insight_id.value),
            (
                "score_swapped_insight",
                assignments.score_swapped.reference.insight_id.value,
            ),
            ("sham_insight", assignments.sham.reference.insight_id.value),
        )
        return GenerationPlan(
            generation=2,
            slots=slots,
            reward=_wave_reward(
                state,
                generation=2,
                parent_configuration_sha256=parent.occurrence.configuration_hash,
            ),
            planner_policy_id=PLANNER_POLICY_ID,
            planner_policy_version=PLANNER_POLICY_VERSION,
            metadata=tuple(
                sorted(
                    (
                        *assignment_metadata,
                        (
                            "held_out_assignment_sha256",
                            bound.assignment_commitment.assignment_sha256,
                        ),
                        ("held_out_parent", self.held_out_parent_sha256),
                        ("union_contract", union.identity_sha256),
                        ("wave", "g2_held_out"),
                    )
                )
            ),
        )


class _ForbiddenRawProblem:
    def evaluate_raw(self, config: object) -> None:
        del config
        raise AssertionError("offline composition must not enter the raw CFD port")


class OfflineAirfoilV7Evaluator:
    """Deterministic detailed-evidence double with observable concurrency."""

    evaluator_identity = OFFLINE_EVALUATOR_IDENTITY

    def __init__(self, *, tie_diagnostics: bool, delay_seconds: float = 0.01) -> None:
        self.tie_diagnostics = tie_diagnostics
        self.delay_seconds = delay_seconds
        self.calls = 0
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()
        self.intervals: list[tuple[int, int, str]] = []

    def evaluate_evidence(self, configuration: object) -> DetailedEvaluationPayload:
        candidate = normalize_candidate(configuration)
        key = candidate_sha256(candidate)
        with self._lock:
            self.calls += 1
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        started = time.monotonic_ns()
        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            if self.tie_diagnostics:
                f_value, v_value = 1.0, 0.52
            else:
                coefficients = tuple(candidate["upper_coefficients"]) + tuple(
                    candidate["lower_coefficients"]
                )
                coefficient_sum = sum(float(value) for value in coefficients)
                coefficient_energy = sum(float(value) ** 2 for value in coefficients)
                alpha_offset = (
                    sum(float(value) for value in candidate["alpha_deg"]) - 7.5
                )
                v_value = 0.52 - 4.0 * coefficient_sum - 0.02 * alpha_offset
                f_value = 1.0 + 0.5 * coefficient_energy - 0.002 * alpha_offset
            finished = time.monotonic_ns()
            receipt_bytes = json.dumps(
                {
                    "candidate_sha256": key,
                    "cfd_calls": 0,
                    "status": "provider_free_fixture",
                },
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            return DetailedEvaluationPayload(
                failure=None,
                objectives=((OBJECTIVE_NAME, float(f_value)),),
                violations=((VIOLATION_NAME, float(v_value)),),
                checks=(
                    EvaluationCheck(
                        name="provider_free_fixture",
                        status=EvaluationCheckStatus.PASS,
                        observed_value=freeze_json(
                            {"candidate_sha256": key, "cfd_calls": 0}
                        ),
                        receipt_locator="$.status",
                    ),
                ),
                receipt=artifact_ref_for_bytes(
                    receipt_bytes,
                    media_type="application/json",
                ),
                evaluator=self.evaluator_identity,
                active_wall_seconds=(finished - started) / 1_000_000_000,
            )
        finally:
            finished = time.monotonic_ns()
            with self._lock:
                self._in_flight -= 1
                self.intervals.append((started, finished, key))


def _offline_telemetry(kind: str, sequence: int) -> AgenticCallTelemetry:
    return AgenticCallTelemetry(
        requested_model="offline/airfoil-v7-fixture",
        resolved_model="offline/airfoil-v7-fixture",
        resolved_provider="provider-free",
        provider_response_id=f"offline-{kind}-{sequence}",
        finish_reason="fixture",
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=Decimal("0"),
        latency_ns=1,
        attempt_count=1,
    )


class OfflineAirfoilV7Generator:
    """Select sealed IDs and produce exactly cited cards without provider I/O."""

    def __init__(self, *, delay_seconds: float = 0.01) -> None:
        self.delay_seconds = delay_seconds
        self.propose_calls = 0
        self.reflect_calls = 0
        self.max_propose_in_flight = 0
        self._propose_in_flight = 0
        self._lock = asyncio.Lock()
        self.requests: list[VariationGenerationRequest] = []
        self.reflection_requests: list[ReflectionGenerationRequest] = []
        self.propose_intervals: list[tuple[int, int, str]] = []

    @staticmethod
    def _union_option(prompt: str) -> str:
        payload = _memory_card_payload(prompt)
        if payload is None:
            raise ValueError("union fixture request lacks one MEMORY_CARD")
        record = json.loads(payload)
        option_ids = record.get("recommended_option_ids")
        if (
            type(option_ids) is not list
            or len(option_ids) != 1
            or type(option_ids[0]) is not str
        ):
            raise ValueError("offline fixture requires one exact held-out option")
        return option_ids[0]

    async def propose(
        self,
        request: VariationGenerationRequest,
    ) -> VariationGenerationResult:
        contract = request.finite_variation_contract
        if contract is None:
            raise ValueError("Airfoil-v7 fixture accepts only finite contracts")
        async with self._lock:
            self.propose_calls += 1
            sequence = self.propose_calls
            self._propose_in_flight += 1
            self.max_propose_in_flight = max(
                self.max_propose_in_flight,
                self._propose_in_flight,
            )
            self.requests.append(request)
        started = time.monotonic_ns()
        try:
            if self.delay_seconds:
                await asyncio.sleep(self.delay_seconds)
            if contract.catalog_id == "airfoil_v7_shape":
                option_id = DIAGNOSTIC_SHAPE_OPTION_ID
            elif contract.catalog_id == "airfoil_v7_trim":
                option_id = DIAGNOSTIC_TRIM_OPTION_ID
            elif contract.catalog_id == "airfoil_v7_union":
                option_id = self._union_option(request.prompt)
            else:
                raise ValueError("fixture received an unknown finite catalog")
            option = contract.resolve(option_id)
            payload = _memory_card_payload(request.prompt)
            claimed: tuple[str, ...] = ()
            if payload is not None:
                insight_id = json.loads(payload).get("insight_id")
                if type(insight_id) is str and insight_id:
                    claimed = (insight_id,)
            return VariationGenerationResult(
                draft=FiniteVariationSelectionDraft(
                    option_id=option.option_id,
                    option_identity_sha256=option.identity_sha256,
                    contract_identity_sha256=contract.identity_sha256,
                    design_rationale=(
                        "Select the presealed option assigned by the provider-free "
                        "orchestration fixture."
                    ),
                    claimed_insight_ids=claimed,
                ),
                telemetry=_offline_telemetry("proposal", sequence),
            )
        finally:
            finished = time.monotonic_ns()
            async with self._lock:
                self._propose_in_flight -= 1
                self.propose_intervals.append((started, finished, contract.catalog_id))

    async def reflect(
        self,
        request: ReflectionGenerationRequest,
    ) -> ReflectionGenerationResult:
        self.reflect_calls += 1
        sequence = self.reflect_calls
        self.reflection_requests.append(request)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if (
            len(request.available_contrast_ids) != 1
            or request.min_insights != 1
            or request.max_insights != 1
        ):
            raise ValueError("offline reflection requires one singleton contrast")
        if sequence not in {1, 2}:
            raise ValueError("offline reflection expects exactly two shard calls")
        if request.insight_contract != REFLECTION_INSIGHT_CONTRACT:
            raise ValueError("offline reflection received the wrong insight contract")
        (contrast_id,) = request.available_contrast_ids
        paths = ("$.alpha_deg", "$.lower_coefficients", "$.upper_coefficients")
        first = sequence == 1
        direction = (
            MetricEffectDirection.DECREASE
            if first
            else MetricEffectDirection.INCREASE
        )
        predictions = tuple(
            MetricEffectPrediction(metric_id, direction)
            for metric_id in REFLECTION_INSIGHT_CONTRACT.required_metric_ids
        )
        ordinal = "first" if first else "second"
        family = "shape_only" if first else "trim_only"
        option_id = (
            DIAGNOSTIC_SHAPE_OPTION_ID if first else DIAGNOSTIC_TRIM_OPTION_ID
        )
        expected_direction = "decrease" if first else "increase"
        return ReflectionGenerationResult(
            insights=(
                InsightDraft(
                    claim=f"The {ordinal} diagnostic intervention may transfer.",
                    trigger="The held-out parent admits the same finite action palette.",
                    mechanism="Select one legal coordinated option under the frozen contract.",
                    affected_paths=paths,
                    evidence_summary=(
                        f"Fixture card citing only the {ordinal} full contrast."
                    ),
                    confidence=0.5,
                    evidence_contrast_ids=(contrast_id,),
                    effect_predictions=predictions,
                    recommended_option_families=(family,),
                    recommended_option_ids=(option_id,),
                    action_template=(
                        f"Select exact option_id {option_id} from the frozen "
                        "held-out palette."
                    ),
                    falsification_condition=(
                        "Falsified if the selected held-out child does not "
                        f"{expected_direction} "
                        "both named metrics relative to its held-out parent."
                    ),
                ),
            ),
            telemetry=_offline_telemetry("reflection", self.reflect_calls),
        )


class _ReflectorProxy:
    """Break the composition/interceptor construction cycle without mutation in core."""

    def __init__(self) -> None:
        self.engine: Any | None = None

    def bind(self, engine: Any) -> None:
        if self.engine is not None:
            raise RuntimeError("reflector proxy is write-once")
        self.engine = engine

    async def reflect(self, *args: Any, **kwargs: Any):
        if self.engine is None:
            raise RuntimeError("reflector proxy is unbound")
        return await self.engine.reflect(*args, **kwargs)

    def identify_phenotype(self, configuration: object):
        if self.engine is None:
            raise RuntimeError("reflector proxy is unbound")
        return self.engine.identify_phenotype(configuration)


@dataclass(slots=True)
class AirfoilV7ExperimentComposition:
    """Shared domain orchestration around any injected benchmark/generator.

    The benchmark owns evaluator semantics and the generator owns provider
    semantics.  This helper owns only the frozen Airfoil-v7 experiment design;
    neither the generic AgentEvolve core nor the live launcher needs to clone
    planner, memory, or reflection wiring.
    """

    composition: AgenticOptimizerComposition
    planner: AirfoilV7SevenCallPlanner
    held_out_parent: HeldOutParentMaterialization


def compose_airfoil_v7_experiment(
    *,
    benchmark: AgenticBenchmark,
    generator: Any,
    id_namespace: str,
    engine_trace_sink: Callable[[Mapping[str, object]], None] | None = None,
    optimizer_trace_sink: Callable[[Mapping[str, object]], None] | None = None,
) -> AirfoilV7ExperimentComposition:
    """Bind the exact 2+2+3 design to injected generic ports."""

    if type(benchmark) is not AgenticBenchmark:
        raise TypeError("benchmark must be an exact AgenticBenchmark")
    if type(id_namespace) is not str or not id_namespace:
        raise ValueError("id_namespace must be non-empty")
    held_out = materialize_held_out_parent()
    ids = DeterministicIdFactory(id_namespace)
    memory = InsightMemoryBank(id_factory=ids)
    sham_reference = register_neutral_sham_card(
        memory=memory,
        affected_paths=(
            "$.alpha_deg",
            "$.lower_coefficients",
            "$.upper_coefficients",
        ),
        applicable_operator_kinds=(OperatorKind.TYPED_MUTATION.value,),
        insight_contract=SHAM_INSIGHT_CONTRACT,
    )
    mailbox = ReflectedCardMailbox()
    held_out_adapter = HeldOutASNPlannerAdapter(
        mailbox=mailbox,
        memory=memory,
        sham_reference=sham_reference,
    )
    planner = AirfoilV7SevenCallPlanner(
        benchmark=benchmark,
        held_out_adapter=held_out_adapter,
        diagnostic_parent_sha256=candidate_sha256(NEUTRAL_PARENT),
        held_out_parent_sha256=held_out.candidate_sha256,
    )
    proxy = _ReflectorProxy()
    feedback = G1ReflectionFeedbackInterceptor(
        engine=proxy,
        mailbox=mailbox,
        diagnostic_slot_ids=DIAGNOSTIC_SLOT_IDS,
        required_metric_ids=REFLECTION_INSIGHT_CONTRACT.required_metric_ids,
        allowed_option_families=(REFLECTION_INSIGHT_CONTRACT.allowed_option_families),
        allowed_option_ids=REFLECTION_INSIGHT_CONTRACT.allowed_option_ids,
        reflection_logical_calls=2,
    )
    composition = compose_agentic_optimizer(
        benchmark,
        generator=generator,
        planner=planner,
        budget=OPTIMIZER_BUDGET,
        seed=20_260_714,
        id_factory=ids,
        memory=memory,
        evaluator_concurrency=EVALUATOR_CONCURRENCY,
        engine_trace_sink=engine_trace_sink,
        optimizer_trace_sink=optimizer_trace_sink,
        prompt_builder=airfoil_v7_prompt_builder,
        reflection_row_projection=AIRFOIL_V7_REFLECTION_PROJECTION,
        reflection_workflow=ContrastShardedReflectionWorkflow(),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        structured_output_budget_policy=STRUCTURED_OUTPUT_BUDGET_POLICY,
        temperature=0.2,
        feedback_interceptor=feedback,
    )
    proxy.bind(composition.engine)
    return AirfoilV7ExperimentComposition(
        composition=composition,
        planner=planner,
        held_out_parent=held_out,
    )


@dataclass(slots=True)
class OfflineExperimentComposition:
    composition: AgenticOptimizerComposition
    planner: AirfoilV7SevenCallPlanner
    generator: OfflineAirfoilV7Generator
    evaluator: OfflineAirfoilV7Evaluator
    held_out_parent: HeldOutParentMaterialization
    engine_events: list[dict[str, object]]
    optimizer_events: list[dict[str, object]]


def compose_offline_experiment(
    *,
    tie_diagnostics: bool = False,
    delay_seconds: float = 0.01,
) -> OfflineExperimentComposition:
    """Compose the actual generic loop with deterministic provider/CFD doubles."""

    problem = AirfoilV7Problem(raw_problem=_ForbiddenRawProblem())
    evaluator = OfflineAirfoilV7Evaluator(
        tie_diagnostics=tie_diagnostics,
        delay_seconds=delay_seconds,
    )
    shape = AirfoilV7ShapeVariationCatalog()
    trim = AirfoilV7TrimVariationCatalog()
    union = AirfoilV7UnionVariationCatalog()
    benchmark = AgenticBenchmark(
        problem=problem,
        reward=AIRFOIL_V7_REWARD_BINDING,
        detailed_evaluator=evaluator,
        outcome_relation=AIRFOIL_V7_ARCHIVE_RELATION,
        phenotype_identity=AirfoilV7PhenotypeIdentityPolicy(),
        finite_variation_catalogs=(shape, trim, union),
    )
    generator = OfflineAirfoilV7Generator(delay_seconds=delay_seconds)
    engine_events: list[dict[str, object]] = []
    optimizer_events: list[dict[str, object]] = []
    core = compose_airfoil_v7_experiment(
        benchmark=benchmark,
        generator=generator,
        id_namespace=(
            "airfoil_v7_offline_tie" if tie_diagnostics else "airfoil_v7_offline"
        ),
        engine_trace_sink=lambda event: engine_events.append(dict(event)),
        optimizer_trace_sink=lambda event: optimizer_events.append(dict(event)),
    )
    return OfflineExperimentComposition(
        composition=core.composition,
        planner=core.planner,
        generator=generator,
        evaluator=evaluator,
        held_out_parent=core.held_out_parent,
        engine_events=engine_events,
        optimizer_events=optimizer_events,
    )


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _offline_record(
    fixture: OfflineExperimentComposition,
    result: OptimizerResult,
) -> dict[str, object]:
    generation_widths = [
        len(receipt.slot_results) for receipt in result.generation_receipts
    ]
    g1_rewards = [
        item.outcome.reward for item in result.generation_receipts[0].slot_results
    ]
    union_requests = tuple(
        request
        for request in fixture.generator.requests
        if request.finite_variation_contract is not None
        and request.finite_variation_contract.catalog_id == "airfoil_v7_union"
    )
    union_contract_sha256s = tuple(
        request.finite_variation_contract.identity_sha256
        for request in union_requests
        if request.finite_variation_contract is not None
    )
    raw_prompt_hashes = tuple(_hash_text(request.prompt) for request in union_requests)
    masked_prompt_hashes = tuple(
        _hash_text(mask_memory_card(request.prompt)) for request in union_requests
    )
    prepared = tuple(
        event
        for event in fixture.engine_events
        if event.get("event_type") == "invocation_prepared"
    )
    evaluated = tuple(
        event
        for event in fixture.engine_events
        if event.get("event_type") == "candidate_evaluated"
    )
    reflection_requests = tuple(
        event
        for event in fixture.engine_events
        if event.get("event_type") == "reflection_requested"
    )
    expected_full = fixture.planner.early_stop_reason is None
    assignment_commitment = fixture.planner.held_out_assignment_commitment
    evaluated_by_label = {
        str(event["label"]): event
        for event in evaluated
        if type(event.get("label")) is str
    }
    held_out_evaluated = tuple(
        event
        for label, event in evaluated_by_label.items()
        if label.startswith("airfoil_v7_g2_")
    )
    trace_checks = {
        "all_prepared_bind_finite_contract": bool(prepared)
        and all("finite_variation_contract_sha256" in event for event in prepared),
        "all_evaluated_bind_option": bool(evaluated)
        and all("finite_option_id" in event for event in evaluated),
        "all_evaluated_bind_catalog": bool(evaluated)
        and all("finite_catalog_definition_sha256" in event for event in evaluated),
        "held_out_assignment_is_quarantine_test": all(
            event.get("assignment_kind") == "quarantine_test"
            for event in prepared
            if event.get("label")
            in {
                "airfoil_v7_g2_adaptive",
                "airfoil_v7_g2_score_swapped",
                "airfoil_v7_g2_sham",
            }
        ),
        "diagnostics_execute_prospective_exact_actions": (
            evaluated_by_label.get("airfoil_v7_g1_shape", {}).get(
                "finite_option_id"
            )
            == DIAGNOSTIC_SHAPE_OPTION_ID
            and evaluated_by_label.get("airfoil_v7_g1_trim", {}).get(
                "finite_option_id"
            )
            == DIAGNOSTIC_TRIM_OPTION_ID
        ),
        "reflection_binds_metric_family_and_exact_action_contract": (
            len(reflection_requests) == 2
            and all(
                event.get("insight_contract")
                == REFLECTION_INSIGHT_CONTRACT.to_record()
                and len(event.get("available_contrast_ids", ())) == 1
                for event in reflection_requests
            )
            and len(
                {
                    event["available_contrast_ids"][0]
                    for event in reflection_requests
                }
            )
            == 2
        ),
        "held_out_arms_share_one_union_contract": (
            len(union_contract_sha256s) == 3 and len(set(union_contract_sha256s)) == 1
            if expected_full
            else not union_contract_sha256s
        ),
        "held_out_actions_equal_card_exact_option_ids": (
            len(held_out_evaluated) == 3
            and all(
                type(event.get("selected_insight_records")) is list
                and len(event["selected_insight_records"]) == 1
                and event["selected_insight_records"][0].get(
                    "recommended_option_ids"
                )
                == [event.get("finite_option_id")]
                for event in held_out_evaluated
            )
            if expected_full
            else not held_out_evaluated
        ),
    }
    accounting = {
        "seed_evaluations": len(result.seed_receipts),
        "unique_evaluations": result.final_state.unique_evaluations,
        "logical_llm_calls": result.final_state.logical_llm_calls,
        "proposal_calls": fixture.generator.propose_calls,
        "reflection_calls": fixture.generator.reflect_calls,
        "generation_widths": generation_widths,
        "feedback_calls_by_generation": [
            receipt.used_logical_llm_calls for receipt in result.feedback_receipts
        ],
    }
    if expected_full:
        accounting_pass = accounting == {
            "seed_evaluations": 2,
            "unique_evaluations": 7,
            "logical_llm_calls": 7,
            "proposal_calls": 5,
            "reflection_calls": 2,
            "generation_widths": [2, 3],
            "feedback_calls_by_generation": [2, 0],
        }
    else:
        accounting_pass = accounting == {
            "seed_evaluations": 2,
            "unique_evaluations": 4,
            "logical_llm_calls": 4,
            "proposal_calls": 2,
            "reflection_calls": 2,
            "generation_widths": [2, 0],
            "feedback_calls_by_generation": [2, 0],
        }
    prompt_invariant = (
        (
            len(union_requests) == 3
            and len(set(raw_prompt_hashes)) == 3
            and len(set(masked_prompt_hashes)) == 1
        )
        if expected_full
        else not union_requests
    )
    return {
        "schema_version": 1,
        "mode": "offline",
        "provider_io_performed": False,
        "cfd_calls": 0,
        "tie_diagnostics": fixture.evaluator.tie_diagnostics,
        "early_stop_reason": fixture.planner.early_stop_reason,
        "early_stop_reason_code": fixture.planner.early_stop_reason_code,
        "accounting": accounting,
        "accounting_pass": accounting_pass,
        "g1_rewards": g1_rewards,
        "g1_reward_contrast_pass": (
            sorted(g1_rewards) == [-1.0, 1.0]
            if expected_full
            else g1_rewards == [0.0, 0.0]
        ),
        "concurrency": {
            "planned_g1_width": 2,
            "planned_g2_width": 3 if expected_full else 0,
            "max_generator_in_flight": fixture.generator.max_propose_in_flight,
            "max_evaluator_in_flight": fixture.evaluator.max_in_flight,
            "concurrency_ready": (
                fixture.generator.max_propose_in_flight >= (3 if expected_full else 2)
                and fixture.evaluator.max_in_flight == 1
            ),
        },
        "prompt_difference": {
            "raw_prompt_sha256s": list(raw_prompt_hashes),
            "masked_prompt_sha256s": list(masked_prompt_hashes),
            "mask_sentinel": MEMORY_CARD_MASK,
            "invariant_pass": prompt_invariant,
        },
        "policy_separation": {
            "archive_relation_definition_sha256": ARCHIVE_DEFINITION_SHA256,
            "local_reward_definition_sha256": REWARD_DEFINITION_SHA256,
            "distinct": ARCHIVE_DEFINITION_SHA256 != REWARD_DEFINITION_SHA256,
        },
        "reflection_insight_contract": REFLECTION_INSIGHT_CONTRACT.to_record(),
        "held_out_assignment_commitment": (
            assignment_commitment.to_record()
            if assignment_commitment is not None
            else None
        ),
        "catalogs": {
            "shape_definition_sha256": SHAPE_CATALOG_DEFINITION_SHA256,
            "trim_definition_sha256": TRIM_CATALOG_DEFINITION_SHA256,
            "union_definition_sha256": UNION_CATALOG_DEFINITION_SHA256,
            "invocation_contract_sha256s": [
                event.get("finite_variation_contract_sha256") for event in prepared
            ],
            "held_out_union_contract_sha256s": list(union_contract_sha256s),
        },
        "trace_checks": trace_checks,
        "trace_checks_pass": all(trace_checks.values()),
        "held_out_parent": fixture.held_out_parent.to_record(),
        "result_hash": result.result_hash,
        "overall_pass": bool(
            accounting_pass
            and prompt_invariant
            and all(trace_checks.values())
            and (fixture.generator.max_propose_in_flight >= (3 if expected_full else 2))
            and fixture.evaluator.max_in_flight == 1
            and ARCHIVE_DEFINITION_SHA256 != REWARD_DEFINITION_SHA256
        ),
    }


async def run_offline_scenario(*, tie_diagnostics: bool) -> dict[str, object]:
    fixture = compose_offline_experiment(tie_diagnostics=tie_diagnostics)
    execution = asyncio.create_task(
        fixture.composition.optimizer.run(
            (NEUTRAL_PARENT, fixture.held_out_parent.candidate)
        )
    )
    # Some restricted CI/sandbox event loops do not promptly observe
    # ``to_thread`` completion when no other timer or file descriptor is live.
    # This provider-free heartbeat changes no ordering or evidence; it merely
    # keeps the loop responsive while the real engine's blocking-evaluator port
    # completes. A live queued provider supplies its own I/O wakeups.
    while not execution.done():
        await asyncio.sleep(0.01)
    result = await execution
    return _offline_record(fixture, result)


async def run_offline_verification() -> dict[str, object]:
    """Execute full and tied-score paths through the real generic optimizer."""

    full = await run_offline_scenario(tie_diagnostics=False)
    tied = await run_offline_scenario(tie_diagnostics=True)
    return {
        "schema_version": 1,
        "mode": "offline_verification",
        "provider_io_performed": False,
        "cfd_calls": 0,
        "full_seven_call_path": full,
        "equal_score_early_stop_path": tied,
        "overall_pass": full["overall_pass"] is True and tied["overall_pass"] is True,
    }


def run_offline_verification_sync() -> dict[str, object]:
    """Run offline verification and synchronously retire its evaluator thread.

    Python 3.11's ``asyncio.run`` shuts its default executor through another
    thread and waits for an event-loop wakeup. Some restricted execution hosts
    delay that wakeup indefinitely even after the worker is idle. Owning the
    one-worker executor here matches the serialized evaluator contract and
    makes shutdown explicit without changing optimizer scheduling.
    """

    loop = asyncio.new_event_loop()
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="airfoil_v7_offline_evaluator",
    )
    loop.set_default_executor(executor)
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(run_offline_verification())
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        loop.close()
        asyncio.set_event_loop(None)


def validation_record() -> dict[str, object]:
    """Materialize parents and all parent-bound catalogs without external I/O."""

    held_out = materialize_held_out_parent()
    if (
        held_out.nonce != HELD_OUT_PARENT_NONCE
        or held_out.candidate_sha256 != HELD_OUT_PARENT_CANDIDATE_SHA256
        or held_out.typed_configuration_sha256 != HELD_OUT_PARENT_TYPED_SHA256
    ):
        raise RuntimeError("materialized held-out parent differs from its freeze")
    shape = AirfoilV7ShapeVariationCatalog()
    trim = AirfoilV7TrimVariationCatalog()
    union = AirfoilV7UnionVariationCatalog()
    benchmark = AgenticBenchmark(
        problem=AirfoilV7Problem(raw_problem=_ForbiddenRawProblem()),
        reward=AIRFOIL_V7_REWARD_BINDING,
        detailed_evaluator=OfflineAirfoilV7Evaluator(
            tie_diagnostics=False,
            delay_seconds=0.0,
        ),
        outcome_relation=AIRFOIL_V7_ARCHIVE_RELATION,
        phenotype_identity=AirfoilV7PhenotypeIdentityPolicy(),
        finite_variation_catalogs=(shape, trim, union),
    )
    contracts = {
        "diagnostic_shape": benchmark.bind_finite_variation(
            "airfoil_v7_shape", NEUTRAL_PARENT
        ),
        "diagnostic_trim": benchmark.bind_finite_variation(
            "airfoil_v7_trim", NEUTRAL_PARENT
        ),
        "held_out_union": benchmark.bind_finite_variation(
            "airfoil_v7_union", held_out.candidate
        ),
    }
    record = {
        "schema_version": 1,
        "mode": "validate",
        "provider_io_performed": False,
        "cfd_calls": 0,
        "credentials_read": False,
        "live_authorized": False,
        "model": MODEL,
        "structured_output_budget_policy": (
            structured_output_budget_policy_record()
        ),
        "budget": OPTIMIZER_BUDGET.to_trace_record(),
        "parents": {
            "diagnostic": {
                "candidate": NEUTRAL_PARENT,
                "candidate_sha256": candidate_sha256(NEUTRAL_PARENT),
                "validation": validate_frozen_no_cfd_candidate(
                    NEUTRAL_PARENT
                ).to_record(),
            },
            "held_out": held_out.to_record(),
        },
        "catalogs": {
            name: {
                "catalog_id": contract.catalog_id,
                "catalog_definition_sha256": (contract.catalog_definition_sha256),
                "contract_identity_sha256": contract.identity_sha256,
                "option_count": len(contract.options),
            }
            for name, contract in contracts.items()
        },
        "waves": {
            "g1": {
                "slot_ids": list(DIAGNOSTIC_SLOT_IDS),
                "prospective_option_ids": [
                    DIAGNOSTIC_SHAPE_OPTION_ID,
                    DIAGNOSTIC_TRIM_OPTION_ID,
                ],
                "provider_proposals_concurrent": True,
                "evaluator_serialized": True,
                "logical_calls": 2,
                "unique_evaluation_reservation": 2,
            },
            "feedback": {
                "logical_calls": 2,
                "workflow": "contrast_sharded_reflection",
                "calls_concurrent": True,
            },
            "g2": {
                "slot_ids": list(HELD_OUT_SLOT_IDS),
                "provider_proposals_concurrent": True,
                "evaluator_serialized": True,
                "logical_calls": 3,
                "unique_evaluation_reservation": 3,
                "clean_early_stop_reason_codes": [
                    reason.value for reason in HeldOutAssignmentUnavailableReason
                ],
                "equal_origin_scores_stop_before_wave": True,
                "prospective_sham_option_id": SHAM_OPTION_ID,
            },
        },
        "policy_separation": {
            "archive_relation_definition_sha256": ARCHIVE_DEFINITION_SHA256,
            "local_reward_definition_sha256": REWARD_DEFINITION_SHA256,
            "distinct": ARCHIVE_DEFINITION_SHA256 != REWARD_DEFINITION_SHA256,
        },
        "reflection_insight_contract": REFLECTION_INSIGHT_CONTRACT.to_record(),
        "sham_insight_contract": SHAM_INSIGHT_CONTRACT.to_record(),
    }
    return {
        **record,
        "validation_sha256": _canonical_sha256(
            b"agent-evolve:airfoil-v7-validation-record:v1\x00", record
        ),
    }


__all__ = [
    "AIRFOIL_V7_REFLECTION_PROJECTION",
    "AirfoilV7ExperimentComposition",
    "AirfoilV7SevenCallPlanner",
    "DIAGNOSTIC_SHAPE_OPTION_ID",
    "DIAGNOSTIC_SLOT_IDS",
    "DIAGNOSTIC_TRIM_OPTION_ID",
    "EVALUATOR_CONCURRENCY",
    "HELD_OUT_SLOT_IDS",
    "HELD_OUT_PARENT_CANDIDATE_SHA256",
    "HELD_OUT_PARENT_NONCE",
    "HELD_OUT_PARENT_TYPED_SHA256",
    "HeldOutParentMaterialization",
    "MAX_OUTPUT_TOKENS",
    "MEMORY_CARD_BEGIN",
    "MEMORY_CARD_END",
    "MEMORY_CARD_MASK",
    "MODEL",
    "NEUTRAL_PARENT",
    "NoCFDValidation",
    "NoCFDValidationError",
    "OPTIMIZER_BUDGET",
    "OfflineAirfoilV7Evaluator",
    "OfflineAirfoilV7Generator",
    "OfflineExperimentComposition",
    "REFLECTION_INSIGHT_CONTRACT",
    "REFLECTION_PROJECTION_DEFINITION_SHA256",
    "STRUCTURED_OUTPUT_BUDGET_POLICY",
    "SHAM_INSIGHT_CONTRACT",
    "SHAM_OPTION_ID",
    "airfoil_v7_prompt_builder",
    "compose_airfoil_v7_experiment",
    "compose_offline_experiment",
    "held_out_candidate_for_nonce",
    "mask_memory_card",
    "materialize_held_out_parent",
    "structured_output_budget_policy_record",
    "run_offline_scenario",
    "run_offline_verification",
    "run_offline_verification_sync",
    "validate_frozen_no_cfd_candidate",
    "validation_record",
]
