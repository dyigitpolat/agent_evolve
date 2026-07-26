"""Frozen Airfoil-v7 semantic identities, relation, and local reward.

This module contains only Airfoil policy.  It consumes generic AgentEvolve
value objects and never teaches the engine about aerodynamic fields.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
import hashlib
import json
import math
import struct
import zlib

from agent_evolve.agentic import (
    DetailedEvaluation,
    EvolutionCandidate,
    ObjectiveSpec,
    OutcomeRelation,
    OutcomeRelationPolicyBinding,
    PhenotypeIdentity,
    RewardPolicyBinding,
    thaw_json,
)
from examples.benchmarks.engibench_airfoil.problem_def import (
    REPRESENTATION_ID,
    normalize_candidate,
)


TASK_SHA256 = "069a017de1cef519616d793955aeef8a98af7acc240c36f88909b38b894cd8ea"
EXTERNAL_DECODER_EVALUATOR_SHA256 = (
    "1d1a2e1a8d2a8e13b9c90e363f01c03707a9d0a653dbe657f1ee5117de20f3b2"
)
COMPATIBILITY_ADAPTER_SHA256 = (
    "53d069ef8d3391a6146868fc47fcde44ac19413e8f538c422331859b1db1d5cc"
)
BASE_COORDS_FLOAT64_LE_SHA256 = (
    "73e0e0e929f01edf9a5f010a8e12f2a93dcaf897933e3929c26bbe482ab280eb"
)
NEUTRAL_DECODED_FLOAT32_LE_SHA256 = (
    "586a9f32b89c14a859f8004e0e12bffe73607f3f522b2ce0d61f43c2f5a3c717"
)
SHAPE_CONTROL_DECODED_FLOAT32_LE_SHA256 = (
    "de9e4e184cb46e1cd6225334fb0152db6a9833029baa7ebaa7f41663fb8abf53"
)
NEUTRAL_POINT_DRAGS = (
    0.010000531374545594,
    0.010012891582960827,
    0.01049422167237048,
)
LIFT_TARGET = 1.1391966821121162
DELTA_F = 0.001
DELTA_V = 0.005

PHENOTYPE_POLICY_ID = "airfoil_v7_decoded_panel"
PHENOTYPE_POLICY_VERSION = 1
ARCHIVE_POLICY_ID = "airfoil_v7_exact_violation_then_drag"
ARCHIVE_POLICY_VERSION = 1
_ARCHIVE_HASH_DOMAIN = b"agent-evolve:airfoil-v7-archive-relation:v1\x00"
_REWARD_HASH_DOMAIN = b"agent-evolve:airfoil-v7-local-parent-reward:v1\x00"

_BASE_COORDS_ZLIB_BASE64 = (
    "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"
)


def _canonical_hash(domain: bytes, record: object) -> str:
    encoded = json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(domain + encoded).hexdigest()


def _base_coordinates() -> tuple[tuple[float, ...], tuple[float, ...]]:
    raw = zlib.decompress(base64.b64decode(_BASE_COORDS_ZLIB_BASE64))
    if len(raw) != 384 * 8 or hashlib.sha256(raw).hexdigest() != BASE_COORDS_FLOAT64_LE_SHA256:
        raise RuntimeError("frozen Airfoil base-coordinate bytes failed identity validation")
    values = struct.unpack("<384d", raw)
    return tuple(values[:192]), tuple(values[192:])


_BASE_X, _BASE_Y = _base_coordinates()


def _bernstein_displacement(coefficients: tuple[float, ...], chord_x: float) -> float:
    degree = len(coefficients) - 1
    blend = sum(
        coefficient
        * math.comb(degree, index)
        * chord_x**index
        * (1.0 - chord_x) ** (degree - index)
        for index, coefficient in enumerate(coefficients)
    )
    return 4.0 * chord_x * (1.0 - chord_x) * blend


def decoded_float32_le_bytes(configuration: object) -> bytes:
    """Reproduce the frozen external-v1 decoder without NumPy."""

    try:
        configuration = thaw_json(configuration)
    except TypeError:
        pass
    candidate = normalize_candidate(configuration)
    upper = tuple(float(item) for item in candidate["upper_coefficients"])
    lower = tuple(float(item) for item in candidate["lower_coefficients"])
    x_min = min(_BASE_X)
    chord = max(_BASE_X) - x_min
    if not chord > 0.0:
        raise RuntimeError("frozen Airfoil chord is nonpositive")
    leading_index = min(range(len(_BASE_X)), key=_BASE_X.__getitem__)
    if leading_index != 96:
        raise RuntimeError("frozen Airfoil leading-edge index changed")
    decoded_y: list[float] = []
    for index, (x_value, y_value) in enumerate(zip(_BASE_X, _BASE_Y, strict=True)):
        chord_x = min(1.0, max(0.0, (x_value - x_min) / chord))
        coefficients = upper if index <= leading_index else lower
        displacement = _bernstein_displacement(coefficients, chord_x)
        if index in (0, leading_index, len(_BASE_X) - 1):
            displacement = 0.0
        decoded_y.append(y_value + displacement)
    return struct.pack("<384f", *_BASE_X, *decoded_y)


_PHENOTYPE_DEFINITION = {
    "representation_id": REPRESENTATION_ID,
    "decoder": "degree9_bernstein_per_surface_with_4x1minusx_envelope",
    "decoder_evaluator_sha256": EXTERNAL_DECODER_EVALUATOR_SHA256,
    "compatibility_adapter_sha256": COMPATIBILITY_ADAPTER_SHA256,
    "task_sha256": TASK_SHA256,
    "base_coords_float64_le_sha256": BASE_COORDS_FLOAT64_LE_SHA256,
    "decoded_coordinate_encoding": "2x192_float32_little_endian_c_order",
    "angle_encoding": "3x_float64_little_endian",
}
PHENOTYPE_DEFINITION_SHA256 = _canonical_hash(
    b"agent-evolve:airfoil-v7-phenotype-definition:v1\x00",
    _PHENOTYPE_DEFINITION,
)


class AirfoilV7PhenotypeIdentityPolicy:
    """Identity of the exact decoded geometry and three operating-point angles."""

    policy_id = PHENOTYPE_POLICY_ID
    policy_version = PHENOTYPE_POLICY_VERSION

    def identify(self, configuration: object) -> PhenotypeIdentity:
        try:
            candidate_value = thaw_json(configuration)
        except TypeError:
            candidate_value = configuration
        candidate = normalize_candidate(candidate_value)
        coordinates = decoded_float32_le_bytes(candidate)
        angles = struct.pack("<3d", *(float(item) for item in candidate["alpha_deg"]))
        digest = hashlib.sha256()
        digest.update(b"agent-evolve:airfoil-v7-phenotype-value:v1\x00")
        digest.update(bytes.fromhex(PHENOTYPE_DEFINITION_SHA256))
        digest.update(len(coordinates).to_bytes(8, "big"))
        digest.update(coordinates)
        digest.update(len(angles).to_bytes(8, "big"))
        digest.update(angles)
        return PhenotypeIdentity(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            value_sha256=digest.hexdigest(),
        )


def _metrics(evaluation: DetailedEvaluation) -> tuple[float, float]:
    if type(evaluation) is not DetailedEvaluation:
        raise TypeError("Airfoil outcome policy requires exact DetailedEvaluation values")
    if not evaluation.success:
        raise ValueError("Airfoil outcome policy compares successful evaluations only")
    objectives = dict(evaluation.objectives)
    violations = dict(evaluation.violations)
    if set(objectives) != {"normalized_multipoint_drag"}:
        raise ValueError("Airfoil v7 objective projection mismatch")
    if set(violations) != {"normalized_lift_equality"}:
        raise ValueError("Airfoil v7 violation projection mismatch")
    return objectives["normalized_multipoint_drag"], violations["normalized_lift_equality"]


def compare_exact_violation_then_drag(
    left: DetailedEvaluation,
    right: DetailedEvaluation,
) -> OutcomeRelation:
    """Transitive archive order: exact V first, then exact f on equal V."""

    left_f, left_v = _metrics(left)
    right_f, right_v = _metrics(right)
    if left_v < right_v:
        return OutcomeRelation.BETTER
    if left_v > right_v:
        return OutcomeRelation.WORSE
    if left_f < right_f:
        return OutcomeRelation.BETTER
    if left_f > right_f:
        return OutcomeRelation.WORSE
    return OutcomeRelation.EQUIVALENT


_ARCHIVE_DEFINITION = {
    "order": "exact_violation_first_then_exact_drag",
    "violation": "normalized_lift_equality",
    "objective": "normalized_multipoint_drag",
    "scientific_feasibility_claim": False,
    "equivalent_only_when_both_values_exactly_equal": True,
}
ARCHIVE_DEFINITION_SHA256 = _canonical_hash(
    _ARCHIVE_HASH_DOMAIN,
    _ARCHIVE_DEFINITION,
)
AIRFOIL_V7_ARCHIVE_RELATION = OutcomeRelationPolicyBinding(
    compare=compare_exact_violation_then_drag,
    policy_id=ARCHIVE_POLICY_ID,
    policy_version=ARCHIVE_POLICY_VERSION,
    definition_sha256=ARCHIVE_DEFINITION_SHA256,
)


def local_delta_parent_feedback(
    child: DetailedEvaluation,
    parent: DetailedEvaluation,
) -> float:
    """Return contextual parent feedback, never an archive-ordering result.

    The dead bands make this utility intentionally nontransitive: its meaning
    is restricted to one child and its actual parent.  Callers must not use it
    for archive admission, survivor selection, or pairwise population sorting.
    """

    child_f, child_v = _metrics(child)
    parent_f, parent_v = _metrics(parent)
    violation_improvement = parent_v - child_v
    if violation_improvement >= DELTA_V:
        return 1.0
    if violation_improvement <= -DELTA_V:
        return -1.0
    drag_improvement = parent_f - child_f
    if drag_improvement >= DELTA_F:
        return 1.0
    if drag_improvement <= -DELTA_F:
        return -1.0
    return 0.0


def airfoil_v7_local_parent_reward(
    child: EvolutionCandidate,
    parents: tuple[EvolutionCandidate, ...],
    objectives: Sequence[ObjectiveSpec],
) -> float:
    """Contextual, nontransitive parent utility; never an archive relation."""

    if tuple((item.name, item.goal) for item in objectives) != (
        ("normalized_multipoint_drag", "min"),
    ):
        raise ValueError("Airfoil v7 reward received the wrong objective declaration")
    if len(parents) != 1:
        raise ValueError("Airfoil v7 local reward requires exactly one actual parent")
    if (
        not child.valid
        or not child.operator_compliant
        or not child.evidence_compliant
    ):
        return -1.0
    parent = parents[0]
    if not parent.valid:
        return -1.0
    if child.detailed_evaluation is None or parent.detailed_evaluation is None:
        raise ValueError("Airfoil v7 reward requires detailed evidence")
    return local_delta_parent_feedback(
        child.detailed_evaluation,
        parent.detailed_evaluation,
    )


_REWARD_DEFINITION = {
    "kind": "local_parent_feedback_not_archive_admission",
    "contextual_pair": "one_child_and_its_actual_parent",
    "nontransitive_contextual_utility": True,
    "must_not_control_archive_admission_or_ordering": True,
    "delta_v": DELTA_V,
    "delta_f": DELTA_F,
    "priority": "violation_then_drag_within_violation_resolution",
    "mapping": {"better": 1.0, "worse": -1.0, "unresolved": 0.0},
    "candidate_operator_or_evidence_failure_reward": -1.0,
    "required_parent_count": 1,
    "archive_relation_definition_sha256": ARCHIVE_DEFINITION_SHA256,
}
REWARD_DEFINITION_SHA256 = _canonical_hash(
    _REWARD_HASH_DOMAIN,
    _REWARD_DEFINITION,
)
AIRFOIL_V7_REWARD_BINDING = RewardPolicyBinding(
    score=airfoil_v7_local_parent_reward,
    definition_hash=REWARD_DEFINITION_SHA256,
)


__all__ = [
    "AIRFOIL_V7_ARCHIVE_RELATION",
    "AIRFOIL_V7_REWARD_BINDING",
    "ARCHIVE_DEFINITION_SHA256",
    "AirfoilV7PhenotypeIdentityPolicy",
    "BASE_COORDS_FLOAT64_LE_SHA256",
    "DELTA_F",
    "DELTA_V",
    "LIFT_TARGET",
    "NEUTRAL_DECODED_FLOAT32_LE_SHA256",
    "NEUTRAL_POINT_DRAGS",
    "PHENOTYPE_DEFINITION_SHA256",
    "REWARD_DEFINITION_SHA256",
    "SHAPE_CONTROL_DECODED_FLOAT32_LE_SHA256",
    "TASK_SHA256",
    "airfoil_v7_local_parent_reward",
    "compare_exact_violation_then_drag",
    "decoded_float32_le_bytes",
    "local_delta_parent_feedback",
]
