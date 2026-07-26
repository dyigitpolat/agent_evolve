"""Authenticated post-decision support oracle for the fresh T000 block.

This module is benchmark-local.  It authenticates the finalized T000 A/U run,
recompiles its exact generic :class:`FiniteActionSetAuthority`, and exploits a
property of this particular Airfoil support: the eight children are the full
Cartesian product of two independent angle choices at each of three operating
points.  One new child supplies the only point-factor level absent from T000.

All eight aggregate outcomes may be reconstructed only after exact repeated
point-witness identities and the Airfoil aggregate equations pass.  Callers
must fall back to direct evaluation of all six previously unseen children when
that certificate fails; the generic AgentEvolve core is deliberately unaware
of this benchmark-specific factorization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import math
from pathlib import Path

from agent_evolve.agentic import (
    CandidateId,
    FiniteActionSetAuthority,
    OperatorKind,
    thaw_json,
)
from agent_evolve.ports.artifact_store import canonical_json_bytes, decode_json_bytes
from examples.benchmarks.engibench_airfoil.problem_def import (
    candidate_sha256,
    normalize_candidate,
)
from examples.benchmarks.engibench_airfoil.v7_contract import (
    LIFT_TARGET,
    NEUTRAL_POINT_DRAGS,
)
from examples.benchmarks.engibench_airfoil.v7_g3_release import (
    ABSOLUTE_Q_DEFINITION_SHA256,
    absolute_airfoil_q,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    OBJECTIVE_NAME,
    VIOLATION_NAME,
    replay_airfoil_v7_durable_receipt,
)
from examples.benchmarks.engibench_airfoil.v8_stage_b_live import (
    AIRFOIL_V8_STAGE_B_CATALOG_ID,
    AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
    AirfoilV8StageBInputs,
    RESEARCH_ARTIFACT_ROOT,
)
from examples.development.durable_run_artifacts import (
    read_jsonl,
    verify_finalized_run_directory,
)


SOURCE_RUN_ID = "airfoil_v9_stage_b_t000_20260715t1403z"
SOURCE_RUN_DIR = (
    RESEARCH_ARTIFACT_ROOT
    / "experiment_logs"
    / "airfoil_stage_b_development"
    / SOURCE_RUN_ID
)
SOURCE_FINALIZATION_SHA256 = (
    "227611d7939eb2d7a11a74a8d64cfceb4c6ed3fa30b30c18e9fc0bbe5a854bdd"
)
SOURCE_RECURSIVE_CONTENT_SHA256 = (
    "31d2bafc43277c8f9cfe3c0c6045aaf1da5fa4782df3d0b15e565664a70f48fd"
)
SOURCE_AUTHORITY_SHA256 = (
    "c3ef366400f0f8df005c566dabe865e2e0d24a0cf815ce2e9c08639d5147df43"
)
SOURCE_SUPPORT_SHA256 = (
    "28ed93f7372dc19da8458b9a588079aefc12a5b4fb5941261423d05bf64383c2"
)
SOURCE_PARENT_CONFIGURATION_SHA256 = (
    "bb67993ad0f9c2724cc2958b99c2974603931605e8d2ac75f86c4ba245db93c7"
)
SOURCE_PARENT_CANDIDATE_ID = "candidate_airfoil_g3_runtime_000001"
ADAPTIVE_OPTION_ID = "trim.p025.n025.p050"
UNIFORM_OPTION_ID = "trim.p050.n025.p025"
FACTOR_PROBE_OPTION_ID = "trim.p025.n050.p025"
FACTOR_PROBE_ORDINAL = 0

SUPPORT_ORACLE_SCHEMA_VERSION = 1
SUPPORT_ORACLE_RESULT_FRAMING = (
    b"agent-evolve:airfoil-v9-stage-b-t000-support-oracle-result:v1\x00"
)
FACTORIZATION_POLICY_ID = "airfoil_v9_exact_point_factorization"
FACTORIZATION_POLICY_VERSION = 1
_FACTORIZATION_DEFINITION = {
    "schema_version": 1,
    "policy_id": FACTORIZATION_POLICY_ID,
    "policy_version": FACTORIZATION_POLICY_VERSION,
    "support": "exact_two_by_two_by_two_point_specific_alpha_cartesian_product",
    "point_independence": (
        "one shared immutable geometry; each external RANS call consumes only its "
        "own operating point and point-specific alpha"
    ),
    "objective": "arithmetic_mean_of_three_point_cd_over_neutral_cd",
    "violation": "sum_of_three_point_abs_cl_minus_target_over_abs_target",
    "certificate": [
        "all_six_point_factor_levels_observed",
        "every_repeated_authoritative_point_witness_matches_exactly",
        "every_direct_aggregate_recomputes_bit_exactly_from_its_point_factors",
    ],
    "failure_action": "directly_evaluate_all_six_non_A_U_options",
}
FACTORIZATION_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:airfoil-v9-stage-b-factorization:def:v1\x00"
    + canonical_json_bytes(_FACTORIZATION_DEFINITION)
).hexdigest()


class AirfoilV9SupportOracleError(RuntimeError):
    """T000 provenance, support, receipt, or factorization evidence is invalid."""


def _load_json(path: Path) -> dict[str, object]:
    value = decode_json_bytes(path.expanduser().resolve(strict=True).read_bytes())
    if type(value) is not dict:
        raise AirfoilV9SupportOracleError(f"{path.name} is not an exact object")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise AirfoilV9SupportOracleError(f"{name} must be an exact object")
    return value


def _list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise AirfoilV9SupportOracleError(f"{name} must be an exact list")
    return value


def _hex_metric(value: object, *, name: str) -> float:
    if type(value) is not str:
        raise AirfoilV9SupportOracleError(f"{name} must use hexadecimal float text")
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise AirfoilV9SupportOracleError(f"{name} is not a hexadecimal float") from exc
    if not math.isfinite(number):
        raise AirfoilV9SupportOracleError(f"{name} must be finite")
    return number


def _one_metric(values: Sequence[tuple[str, float]], *, name: str) -> float:
    record = dict(values)
    if set(record) != {name}:
        raise AirfoilV9SupportOracleError(f"typed evaluation lacks exact {name}")
    value = float(record[name])
    if not math.isfinite(value):
        raise AirfoilV9SupportOracleError(f"typed {name} is non-finite")
    return value


def _receipt_binding(path: Path, *, root: Path, artifact_id: str) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    content = resolved.read_bytes()
    return {
        "relative_path": resolved.relative_to(root.resolve(strict=True)).as_posix(),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "artifact_id": artifact_id,
    }


@dataclass(frozen=True, slots=True)
class PointFactorObservation:
    point_index: int
    alpha_deg: float
    cd: float
    cl: float
    witness_sha256: str

    def key(self) -> tuple[int, str]:
        return self.point_index, self.alpha_deg.hex()

    def to_record(self) -> dict[str, object]:
        return {
            "point_index": self.point_index,
            "alpha_deg_hex": self.alpha_deg.hex(),
            "cd_hex": self.cd.hex(),
            "cl_hex": self.cl.hex(),
            "witness_sha256": self.witness_sha256,
        }


@dataclass(frozen=True, slots=True)
class SupportObservation:
    ordinal: int
    option_id: str
    configuration_sha256: str
    raw_candidate_sha256: str
    objective: float
    violation: float
    q_value: float
    receipt_path: Path
    receipt: Mapping[str, object]
    point_factors: tuple[PointFactorObservation, ...]
    evidence_mode: str

    def to_record(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "option_id": self.option_id,
            "configuration_sha256": self.configuration_sha256,
            "raw_candidate_sha256": self.raw_candidate_sha256,
            "objective_hex": self.objective.hex(),
            "violation_hex": self.violation.hex(),
            "q_hex": self.q_value.hex(),
            "receipt": dict(self.receipt),
            "point_factors": [value.to_record() for value in self.point_factors],
            "evidence_mode": self.evidence_mode,
        }


@dataclass(frozen=True, slots=True)
class T000SupportOracleContext:
    authority: FiniteActionSetAuthority
    source_run_dir: Path
    source_finalization: Mapping[str, object]
    source_result: Mapping[str, object]
    known_observations: tuple[SupportObservation, SupportObservation]
    support_levels: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]

    @property
    def known_by_option(self) -> dict[str, SupportObservation]:
        return {value.option_id: value for value in self.known_observations}


def compile_t000_authority(inputs: AirfoilV8StageBInputs) -> FiniteActionSetAuthority:
    """Recompile the exact generic authority used by the finalized T000 run."""

    if type(inputs) is not AirfoilV8StageBInputs:
        raise TypeError("inputs must be exact AirfoilV8StageBInputs")
    inputs.__post_init__()
    compiled = inputs.benchmark.compile_registered_hypothesis_treatment(
        catalog_id=AIRFOIL_V8_STAGE_B_CATALOG_ID,
        parent_candidate_id=CandidateId(SOURCE_PARENT_CANDIDATE_ID),
        parent_configuration=inputs.seed_configuration,
        entry=inputs.learned_card,
        requested_operator_kind=OperatorKind.TYPED_MUTATION.value,
        context_projection_sha256=inputs.planner_factory.context_projection_sha256,
        endpoint_definition_sha256=inputs.planner_factory.endpoint_definition_sha256,
    )
    authority, _ = inputs.benchmark.compile_finite_action_set(
        compiled_anchor=compiled,
        required_cardinality=AIRFOIL_V8_STAGE_B_REQUIRED_CARDINALITY,
        source_mode=inputs.planner_factory.source_mode,
    )
    return authority


def _cartesian_support_levels(
    authority: FiniteActionSetAuthority,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    first_child = normalize_candidate(
        thaw_json(authority.support.options[0].option.child_configuration)
    )
    # Use the first child only as the immutable shape reference.  Every support
    # child must share it; the parent hash itself intentionally has no decoder.
    reference = dict(first_child)
    reference.pop("alpha_deg")
    combinations: set[tuple[float, float, float]] = set()
    levels: list[set[float]] = [set(), set(), set()]
    for row in authority.support.options:
        child = normalize_candidate(thaw_json(row.option.child_configuration))
        shape = dict(child)
        alpha = tuple(float(value) for value in shape.pop("alpha_deg"))
        if shape != reference:
            raise AirfoilV9SupportOracleError(
                "T000 support changes geometry as well as point-specific alpha"
            )
        combinations.add(alpha)
        for index, value in enumerate(alpha):
            levels[index].add(value)
    canonical_levels = tuple(tuple(sorted(value)) for value in levels)
    if any(len(value) != 2 for value in canonical_levels):
        raise AirfoilV9SupportOracleError("T000 support is not binary at every point")
    expected = set(itertools.product(*canonical_levels))
    if len(combinations) != 8 or combinations != expected:
        raise AirfoilV9SupportOracleError(
            "T000 support is not the complete two-by-two-by-two Cartesian product"
        )
    return canonical_levels  # type: ignore[return-value]


def observation_from_receipt(
    *,
    authority: FiniteActionSetAuthority,
    ordinal: int,
    receipt_path: Path,
    receipt_root: Path,
    evidence_mode: str,
) -> SupportObservation:
    """Authenticate one raw receipt and project exact point-factor evidence."""

    if type(ordinal) is not int or not 0 <= ordinal < authority.support.cardinality:
        raise AirfoilV9SupportOracleError("support ordinal lies outside the authority")
    option = authority.support.options[ordinal]
    child = normalize_candidate(thaw_json(option.option.child_configuration))
    payload = replay_airfoil_v7_durable_receipt(receipt_path, child)
    if payload.failure is not None or payload.receipt is None:
        raise AirfoilV9SupportOracleError("support-oracle receipt is not successful")
    objective = _one_metric(payload.objectives, name=OBJECTIVE_NAME)
    violation = _one_metric(payload.violations, name=VIOLATION_NAME)
    q_value = absolute_airfoil_q(
        normalized_multipoint_drag=objective,
        normalized_lift_equality=violation,
        valid=True,
    )
    raw = _load_json(receipt_path)
    if normalize_candidate(raw.get("candidate")) != child:
        raise AirfoilV9SupportOracleError("raw receipt candidate differs from authority child")
    expected_raw_sha = candidate_sha256(child)
    if raw.get("candidate_sha256") != expected_raw_sha:
        raise AirfoilV9SupportOracleError("raw receipt candidate identity changed")
    raw_points = _list(raw.get("points"), name="raw receipt points")
    if len(raw_points) != 3:
        raise AirfoilV9SupportOracleError("raw receipt lacks exactly three points")
    factors: list[PointFactorObservation] = []
    for index, raw_point_value in enumerate(raw_points):
        raw_point = _mapping(raw_point_value, name=f"raw point {index}")
        evidence = _mapping(
            raw_point.get("evaluator_evidence"),
            name=f"raw point {index} evaluator evidence",
        )
        witness = _mapping(evidence.get("witness"), name=f"raw point {index} witness")
        outputs = _mapping(witness.get("outputs"), name=f"raw point {index} outputs")
        point_basis = _mapping(witness.get("point"), name=f"raw point {index} basis")
        try:
            alpha = float(raw_point.get("alpha_deg"))
            cd = float(raw_point.get("cd"))
            cl = float(raw_point.get("cl"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise AirfoilV9SupportOracleError("raw point metrics are malformed") from exc
        if not all(math.isfinite(value) for value in (alpha, cd, cl)):
            raise AirfoilV9SupportOracleError("raw point metrics are non-finite")
        if (
            raw_point.get("index") != index
            or alpha != child["alpha_deg"][index]
            or point_basis.get("alpha_deg") != alpha
            or outputs.get("cd") != cd
            or outputs.get("cl") != cl
        ):
            raise AirfoilV9SupportOracleError(
                "authoritative point witness differs from raw point or child"
            )
        factors.append(
            PointFactorObservation(
                point_index=index,
                alpha_deg=alpha,
                cd=cd,
                cl=cl,
                witness_sha256=hashlib.sha256(canonical_json_bytes(dict(witness))).hexdigest(),
            )
        )
    projected_f, projected_v = aggregate_point_factors(tuple(factors))
    if projected_f.hex() != objective.hex() or projected_v.hex() != violation.hex():
        raise AirfoilV9SupportOracleError(
            "typed aggregate is not bit-exact under the Airfoil factor equations"
        )
    artifact_id = payload.receipt.artifact_id.value
    return SupportObservation(
        ordinal=ordinal,
        option_id=option.option.option_id,
        configuration_sha256=option.option.child_configuration_sha256,
        raw_candidate_sha256=expected_raw_sha,
        objective=objective,
        violation=violation,
        q_value=q_value,
        receipt_path=receipt_path.expanduser().resolve(strict=True),
        receipt=_receipt_binding(
            receipt_path,
            root=receipt_root,
            artifact_id=artifact_id,
        ),
        point_factors=tuple(factors),
        evidence_mode=evidence_mode,
    )


def aggregate_point_factors(
    factors: Sequence[PointFactorObservation],
) -> tuple[float, float]:
    if len(factors) != 3:
        raise AirfoilV9SupportOracleError("exactly three point factors are required")
    if tuple(value.point_index for value in factors) != (0, 1, 2):
        raise AirfoilV9SupportOracleError("point factors must be ordered 0,1,2")
    objective = sum(
        value.cd / neutral
        for value, neutral in zip(factors, NEUTRAL_POINT_DRAGS, strict=True)
    ) / 3.0
    violation = sum(
        abs(value.cl - LIFT_TARGET) / abs(LIFT_TARGET) for value in factors
    )
    return float(objective), float(violation)


def _find_source_receipt(
    source_root: Path,
    relative_paths: Sequence[object],
    *,
    raw_candidate_sha256: str,
) -> Path:
    matches: list[Path] = []
    for value in relative_paths:
        if type(value) is not str:
            raise AirfoilV9SupportOracleError("source receipt inventory is malformed")
        path = (source_root / value).resolve(strict=True)
        raw = _load_json(path)
        if raw.get("candidate_sha256") == raw_candidate_sha256:
            matches.append(path)
    if len(matches) != 1:
        raise AirfoilV9SupportOracleError(
            "source run does not contain exactly one matching A/U raw receipt"
        )
    return matches[0]


def load_t000_support_oracle_context(
    inputs: AirfoilV8StageBInputs,
    *,
    source_run_dir: Path = SOURCE_RUN_DIR,
) -> T000SupportOracleContext:
    """Authenticate T000 and bind its two measured rows to the exact authority."""

    root = source_run_dir.expanduser().resolve(strict=True)
    finalized = verify_finalized_run_directory(root)
    if (
        finalized.get("status") != "completed"
        or finalized.get("finalization_sha256") != SOURCE_FINALIZATION_SHA256
        or finalized.get("recursive_content_sha256")
        != SOURCE_RECURSIVE_CONTENT_SHA256
    ):
        raise AirfoilV9SupportOracleError("source T000 finalization identity changed")
    authority = compile_t000_authority(inputs)
    if (
        authority.authority_sha256 != SOURCE_AUTHORITY_SHA256
        or authority.support.support_sha256 != SOURCE_SUPPORT_SHA256
        or authority.support.parent_configuration_sha256
        != SOURCE_PARENT_CONFIGURATION_SHA256
        or authority.support.cardinality != 8
    ):
        raise AirfoilV9SupportOracleError("recompiled T000 authority/hash mismatch")
    levels = _cartesian_support_levels(authority)

    result = _load_json(root / "result.json")
    if (
        result.get("authority_sha256") != authority.authority_sha256
        or result.get("support_sha256") != authority.support.support_sha256
        or result.get("support_cardinality") != 8
        or result.get("claim_boundary")
        != "fresh_parent_single_block_development_not_replicated_paper_evidence"
    ):
        raise AirfoilV9SupportOracleError("source result differs from exact T000 block")
    traces = read_jsonl(root / "execution_traces.jsonl")
    prepared = [
        value
        for value in traces
        if value.get("event_type") == "invocation_prepared"
        and value.get("finite_action_set_authority") is not None
    ]
    expected_authority_record = {
        **authority.to_record(),
        "authority_sha256": authority.authority_sha256,
    }
    if (
        len(prepared) != 1
        or prepared[0].get("finite_action_set_authority")
        != expected_authority_record
    ):
        raise AirfoilV9SupportOracleError("source trace authority bytes do not replay")

    inventory = _load_json(root / "raw_receipt_inventory.json")
    relative_paths = _list(
        inventory.get("relative_paths"),
        name="source raw receipt inventory",
    )
    if inventory.get("receipt_count") != len(relative_paths):
        raise AirfoilV9SupportOracleError("source raw receipt inventory count changed")
    arms = _mapping(result.get("arms"), name="source result arms")
    expected_options = {"A": ADAPTIVE_OPTION_ID, "U": UNIFORM_OPTION_ID}
    observations: list[SupportObservation] = []
    for arm_name in ("A", "U"):
        arm = _mapping(arms.get(arm_name), name=f"source arm {arm_name}")
        option_id = expected_options[arm_name]
        if arm.get("option_id") != option_id:
            raise AirfoilV9SupportOracleError(f"source {arm_name} option changed")
        ordinal = arm.get("selected_ordinal")
        if type(ordinal) is not int or not 0 <= ordinal < 8:
            raise AirfoilV9SupportOracleError(f"source {arm_name} ordinal changed")
        option = authority.support.options[ordinal]
        if option.option.option_id != option_id:
            raise AirfoilV9SupportOracleError(f"source {arm_name} ordinal/ID mismatch")
        candidate = _mapping(arm.get("candidate"), name=f"source {arm_name} candidate")
        if candidate.get("configuration_hash") != option.option.child_configuration_sha256:
            raise AirfoilV9SupportOracleError(f"source {arm_name} child hash mismatch")
        child = normalize_candidate(thaw_json(option.option.child_configuration))
        raw_path = _find_source_receipt(
            root,
            relative_paths,
            raw_candidate_sha256=candidate_sha256(child),
        )
        observation = observation_from_receipt(
            authority=authority,
            ordinal=ordinal,
            receipt_path=raw_path,
            receipt_root=root,
            evidence_mode=f"reused_source_{arm_name}",
        )
        objectives = _mapping(candidate.get("objectives"), name=f"source {arm_name} objectives")
        violations = _mapping(candidate.get("violations"), name=f"source {arm_name} violations")
        if (
            _hex_metric(objectives.get(OBJECTIVE_NAME), name=f"source {arm_name} objective").hex()
            != observation.objective.hex()
            or _hex_metric(violations.get(VIOLATION_NAME), name=f"source {arm_name} violation").hex()
            != observation.violation.hex()
            or candidate.get("evaluation_receipt")
            != observation.receipt["artifact_id"]
            or _hex_metric(arm.get("reward_hex"), name=f"source {arm_name} reward").hex()
            != observation.q_value.hex()
        ):
            raise AirfoilV9SupportOracleError(f"source {arm_name} metrics/receipt mismatch")
        candidate_trace = [
            value
            for value in traces
            if value.get("event_type") == "candidate_evaluated"
            and value.get("candidate_id") == candidate.get("candidate_id")
        ]
        if len(candidate_trace) != 1:
            raise AirfoilV9SupportOracleError(f"source {arm_name} candidate trace missing")
        detailed = _mapping(
            candidate_trace[0].get("detailed_evaluation"),
            name=f"source {arm_name} detailed trace",
        )
        trace_objectives = _mapping(detailed.get("objectives"), name="trace objectives")
        trace_violations = _mapping(detailed.get("violations"), name="trace violations")
        trace_receipt = _mapping(detailed.get("receipt"), name="trace receipt")
        if (
            trace_objectives.get(OBJECTIVE_NAME) != observation.objective
            or trace_violations.get(VIOLATION_NAME) != observation.violation
            or trace_receipt.get("artifact_id") != observation.receipt["artifact_id"]
            or trace_receipt.get("sha256_hex") != observation.receipt["sha256"]
        ):
            raise AirfoilV9SupportOracleError(f"source {arm_name} trace evidence mismatch")
        observations.append(observation)

    context = T000SupportOracleContext(
        authority=authority,
        source_run_dir=root,
        source_finalization=finalized,
        source_result=result,
        known_observations=(observations[0], observations[1]),
        support_levels=levels,
    )
    # Two rows can never certify all six factor levels.  A disagreement in their
    # repeated central point is not a provenance failure: it disables the cheap
    # factorized path and forces direct evaluation of all six unseen options.
    passed, _, _ = factorization_certificate(context, context.known_observations)
    if passed:
        raise AirfoilV9SupportOracleError(
            "two source rows unexpectedly suffice for the complete factor certificate"
        )
    return context


def support_oracle_readiness_record(
    context: T000SupportOracleContext,
) -> dict[str, object]:
    authority = context.authority
    probe = authority.support.options[FACTOR_PROBE_ORDINAL]
    if probe.option.option_id != FACTOR_PROBE_OPTION_ID:
        raise AirfoilV9SupportOracleError("factor probe ordinal/ID changed")
    known = context.known_by_option
    if set(known) != {ADAPTIVE_OPTION_ID, UNIFORM_OPTION_ID}:
        raise AirfoilV9SupportOracleError("known T000 option set changed")
    unseen = [
        row.option.option_id
        for row in authority.support.options
        if row.option.option_id not in known
    ]
    _, source_certificate, _ = factorization_certificate(
        context,
        context.known_observations,
    )
    repeated_checks = source_certificate["repeated_witness_checks"]
    existing_repeat_exact = bool(
        type(repeated_checks) is list
        and len(repeated_checks) == 1
        and repeated_checks[0].get("point_index") == 1
        and repeated_checks[0].get("exact") is True
        and source_certificate["repeated_witnesses_exact"] is True
    )
    return {
        "schema_version": SUPPORT_ORACLE_SCHEMA_VERSION,
        "ready": True,
        "claim_boundary": "postdecision_support_diagnostic_not_method_comparison",
        "provider_calls_planned": 0,
        "credentials_read": False,
        "source": {
            "run_id": SOURCE_RUN_ID,
            "finalization_sha256": context.source_finalization["finalization_sha256"],
            "recursive_content_sha256": context.source_finalization[
                "recursive_content_sha256"
            ],
            "authority_sha256": authority.authority_sha256,
            "support_sha256": authority.support.support_sha256,
            "parent_configuration_sha256": authority.support.parent_configuration_sha256,
            "reused_options": [
                value.to_record() for value in context.known_observations
            ],
        },
        "factorization": {
            "policy_id": FACTORIZATION_POLICY_ID,
            "policy_version": FACTORIZATION_POLICY_VERSION,
            "definition_sha256": FACTORIZATION_DEFINITION_SHA256,
            "support_levels_hex": [
                [level.hex() for level in values] for values in context.support_levels
            ],
            "existing_repeated_point_1_witness_exact": existing_repeat_exact,
            "primary_factor_probe_eligible": existing_repeat_exact,
            "execution_path": (
                "one_factor_probe_then_certify"
                if existing_repeat_exact
                else "direct_full_six_required"
            ),
            "probe": {
                "ordinal": FACTOR_PROBE_ORDINAL,
                "option_id": FACTOR_PROBE_OPTION_ID,
                "configuration_sha256": probe.option.child_configuration_sha256,
            },
            "primary_new_evaluations": 1,
            "fallback_new_evaluations": 6,
            "fallback_policy": "direct_full_six_on_any_factor_certificate_failure",
            "unseen_option_ids": unseen,
        },
        "ranking": {
            "endpoint_definition_sha256": ABSOLUTE_Q_DEFINITION_SHA256,
            "larger_q_is_better": True,
            "uniform_expectation_denominator": 8,
        },
    }


def factorization_certificate(
    context: T000SupportOracleContext,
    observations: Sequence[SupportObservation],
) -> tuple[
    bool,
    dict[str, object],
    dict[tuple[int, str], PointFactorObservation],
]:
    """Return an exact witness/aggregate certificate and one factor bank."""

    by_key: dict[tuple[int, str], list[tuple[str, PointFactorObservation]]] = {}
    direct_checks: list[dict[str, object]] = []
    for observation in observations:
        projected_f, projected_v = aggregate_point_factors(observation.point_factors)
        direct_checks.append(
            {
                "option_id": observation.option_id,
                "objective_exact": projected_f.hex() == observation.objective.hex(),
                "violation_exact": projected_v.hex() == observation.violation.hex(),
            }
        )
        for factor in observation.point_factors:
            by_key.setdefault(factor.key(), []).append((observation.option_id, factor))
    repeated: list[dict[str, object]] = []
    repeated_exact = True
    for key, values in sorted(by_key.items()):
        if len(values) < 2:
            continue
        witness_hashes = sorted({value.witness_sha256 for _, value in values})
        exact = len(witness_hashes) == 1
        repeated_exact = repeated_exact and exact
        repeated.append(
            {
                "point_index": key[0],
                "alpha_deg_hex": key[1],
                "source_option_ids": sorted(option_id for option_id, _ in values),
                "witness_sha256s": witness_hashes,
                "exact": exact,
            }
        )
    required_keys = {
        (point_index, value.hex())
        for point_index, levels in enumerate(context.support_levels)
        for value in levels
    }
    observed_keys = set(by_key)
    bank = {key: values[0][1] for key, values in by_key.items()}
    direct_exact = all(
        value["objective_exact"] and value["violation_exact"]
        for value in direct_checks
    )
    passed = (
        observed_keys == required_keys
        and repeated_exact
        and direct_exact
        and len(repeated) >= 3
    )
    return (
        passed,
        {
            "schema_version": 1,
            "policy_id": FACTORIZATION_POLICY_ID,
            "policy_version": FACTORIZATION_POLICY_VERSION,
            "definition_sha256": FACTORIZATION_DEFINITION_SHA256,
            "required_factor_count": 6,
            "observed_factor_count": len(observed_keys),
            "all_factor_levels_observed": observed_keys == required_keys,
            "repeated_witness_checks": repeated,
            "repeated_witnesses_exact": repeated_exact,
            "direct_aggregate_checks": direct_checks,
            "direct_aggregates_exact": direct_exact,
            "passed": passed,
        },
        bank,
    )


def build_support_oracle_result(
    context: T000SupportOracleContext,
    *,
    new_observations: Sequence[SupportObservation],
    fallback_used: bool,
) -> dict[str, object]:
    """Compute all-eight Q rows, ranks, and exact-uniform summaries."""

    all_observations = (*context.known_observations, *tuple(new_observations))
    if len({value.option_id for value in all_observations}) != len(all_observations):
        raise AirfoilV9SupportOracleError("support observation option IDs collide")
    direct_by_id = {value.option_id: value for value in all_observations}
    passed, certificate, factor_bank = factorization_certificate(
        context,
        all_observations,
    )
    if fallback_used:
        if set(direct_by_id) != {
            row.option.option_id for row in context.authority.support.options
        }:
            raise AirfoilV9SupportOracleError(
                "fallback result does not directly cover all eight options"
            )
        result_mode = "direct_full_support_fallback"
    else:
        if not passed:
            raise AirfoilV9SupportOracleError(
                "factorized result requested without a passing certificate"
            )
        if len(new_observations) != 1 or (
            new_observations[0].option_id != FACTOR_PROBE_OPTION_ID
        ):
            raise AirfoilV9SupportOracleError(
                "factorized result must use exactly the sealed one-option probe"
            )
        result_mode = "certified_factorized_support_oracle"

    rows: list[dict[str, object]] = []
    q_by_id: dict[str, float] = {}
    for ordinal, option_row in enumerate(context.authority.support.options):
        option_id = option_row.option.option_id
        child = normalize_candidate(thaw_json(option_row.option.child_configuration))
        direct = direct_by_id.get(option_id)
        if direct is not None:
            objective = direct.objective
            violation = direct.violation
            q_value = direct.q_value
            evidence_mode = direct.evidence_mode
            receipt: object = dict(direct.receipt)
            factors = direct.point_factors
        else:
            factors = tuple(
                factor_bank[(index, float(alpha).hex())]
                for index, alpha in enumerate(child["alpha_deg"])
            )
            objective, violation = aggregate_point_factors(factors)  # type: ignore[arg-type]
            q_value = absolute_airfoil_q(
                normalized_multipoint_drag=objective,
                normalized_lift_equality=violation,
                valid=True,
            )
            evidence_mode = "factorized_reconstruction"
            receipt = None
        q_by_id[option_id] = q_value
        rows.append(
            {
                "ordinal": ordinal,
                "option_id": option_id,
                "configuration_sha256": option_row.option.child_configuration_sha256,
                "objective_hex": objective.hex(),
                "violation_hex": violation.hex(),
                "q_hex": q_value.hex(),
                "evidence_mode": evidence_mode,
                "receipt": receipt,
                "factor_witness_sha256s": [
                    value.witness_sha256 for value in factors
                ],
            }
        )
    for row in rows:
        q_value = q_by_id[str(row["option_id"])]
        row["rank"] = 1 + sum(value > q_value for value in q_by_id.values())
        row["rank_tie_count"] = sum(value == q_value for value in q_by_id.values())
    rows.sort(key=lambda value: int(value["ordinal"]))
    a_q = q_by_id[ADAPTIVE_OPTION_ID]
    a_row = next(value for value in rows if value["option_id"] == ADAPTIVE_OPTION_ID)
    uniform_sum = sum(q_by_id.values())
    uniform_mean = uniform_sum / 8.0
    beats_a = sum(value > a_q for value in q_by_id.values())
    equals_a = sum(value == a_q for value in q_by_id.values())
    unsigned: dict[str, object] = {
        "schema_version": SUPPORT_ORACLE_SCHEMA_VERSION,
        "claim_boundary": "postdecision_support_diagnostic_not_method_comparison",
        "status": result_mode,
        "provider_calls": 0,
        "credentials_read": False,
        "source": {
            "run_id": SOURCE_RUN_ID,
            "finalization_sha256": context.source_finalization[
                "finalization_sha256"
            ],
            "recursive_content_sha256": context.source_finalization[
                "recursive_content_sha256"
            ],
            "authority_sha256": context.authority.authority_sha256,
            "support_sha256": context.authority.support.support_sha256,
            "reused_evaluation_count": 2,
            "reused_options": [
                value.to_record() for value in context.known_observations
            ],
        },
        "new_evaluation_count": len(new_observations),
        "fallback_used": fallback_used,
        "factorization_certificate": certificate,
        "support_results": rows,
        "ranking": {
            "semantics": "larger_absolute_q_is_better; competition_rank_1_is_best",
            "adaptive_option_id": ADAPTIVE_OPTION_ID,
            "adaptive_rank": a_row["rank"],
            "adaptive_q_hex": a_q.hex(),
            "best_option_ids": [
                str(value["option_id"]) for value in rows if value["rank"] == 1
            ],
            "exact_uniform": {
                "support_size": 8,
                "q_sum_hex": uniform_sum.hex(),
                "expected_q_hex": uniform_mean.hex(),
                "adaptive_minus_expected_q_hex": (a_q - uniform_mean).hex(),
                "strictly_beats_adaptive": [beats_a, 8],
                "equals_adaptive": [equals_a, 8],
                "worse_than_adaptive": [8 - beats_a - equals_a, 8],
            },
        },
    }
    return unsigned


def seal_support_oracle_result(record: Mapping[str, object]) -> dict[str, object]:
    unsigned = dict(record)
    if "result_sha256" in unsigned:
        raise AirfoilV9SupportOracleError("support-oracle result is already sealed")
    return {
        **unsigned,
        "result_sha256": hashlib.sha256(
            SUPPORT_ORACLE_RESULT_FRAMING + canonical_json_bytes(unsigned)
        ).hexdigest(),
    }


__all__ = [
    "ADAPTIVE_OPTION_ID",
    "AirfoilV9SupportOracleError",
    "FACTORIZATION_DEFINITION_SHA256",
    "FACTORIZATION_POLICY_ID",
    "FACTOR_PROBE_OPTION_ID",
    "FACTOR_PROBE_ORDINAL",
    "SOURCE_AUTHORITY_SHA256",
    "SOURCE_RUN_DIR",
    "SOURCE_RUN_ID",
    "SOURCE_SUPPORT_SHA256",
    "SUPPORT_ORACLE_SCHEMA_VERSION",
    "PointFactorObservation",
    "SupportObservation",
    "T000SupportOracleContext",
    "aggregate_point_factors",
    "build_support_oracle_result",
    "compile_t000_authority",
    "factorization_certificate",
    "load_t000_support_oracle_context",
    "observation_from_receipt",
    "seal_support_oracle_result",
    "support_oracle_readiness_record",
]
