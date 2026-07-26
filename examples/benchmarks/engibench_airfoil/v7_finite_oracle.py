"""Provider-free exhaustive oracle for the frozen Airfoil-v7 finite catalog.

The module is benchmark-local by design.  It consumes AgentEvolve's public
``AgenticBenchmark`` and finite-catalog interfaces, while Airfoil owns the
external evaluator, raw-receipt replay, and host-global resource lease.

Execution is deliberately conservative: every one of the 80 options gets
exactly one evaluator attempt.  Each attempt is charged by an immutable
``started.json`` before evaluator entry and committed by an immutable
``terminal.json`` only after any raw receipt has been fsynced and replayed.
Resume accepts a verified successful prefix plus at most one open start.  An
open start is recoverable only from one matching authenticated raw receipt;
otherwise the run is invalidated rather than silently retrying.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import time

from agent_evolve.agentic import (
    AgenticBenchmark,
    DetailedEvaluation,
    DetailedEvaluationPayload,
    EvaluationTimings,
    ExclusiveResourceLease,
    FileExclusiveResourceLease,
    FiniteVariationContract,
    FiniteVariationOption,
    OutcomeRelation,
    bind_finite_variation_catalog,
    freeze_json,
    thaw_json,
)
from examples.benchmarks.engibench_airfoil.converged_problem_def import (
    local_default_converged_settings,
)
from examples.benchmarks.engibench_airfoil.problem_def import (
    candidate_sha256,
    normalize_candidate,
)
from examples.benchmarks.engibench_airfoil.v7_experiment_support import (
    HELD_OUT_PARENT_CANDIDATE_SHA256,
    HELD_OUT_PARENT_TYPED_SHA256,
    materialize_held_out_parent,
)
from examples.benchmarks.engibench_airfoil.v7_contract import (
    AIRFOIL_V7_ARCHIVE_RELATION,
    AIRFOIL_V7_REWARD_BINDING,
    ARCHIVE_DEFINITION_SHA256,
    AirfoilV7PhenotypeIdentityPolicy,
    DELTA_F,
    DELTA_V,
    PHENOTYPE_DEFINITION_SHA256,
    REWARD_DEFINITION_SHA256,
    local_delta_parent_feedback,
)
from examples.benchmarks.engibench_airfoil.v7_launch import (
    DEFAULT_LIVE_LOG_ROOT,
    _production_resource_lease,
    _resource_lease_manifest_record,
    _verification_report_binding,
    create_seed_qualification_benchmark,
    source_snapshot,
    write_bytes_atomic,
    write_json_atomic,
)
from examples.benchmarks.engibench_airfoil.v7_problem_def import (
    AIRFOIL_V7_OPTIMIZATION_SEMANTICS,
    EVALUATOR_IDENTITY,
    OBJECTIVE_NAME,
    VIOLATION_NAME,
    replay_airfoil_v7_durable_receipt,
)
from examples.benchmarks.engibench_airfoil.v7_variation_catalog import (
    AirfoilV7UnionVariationCatalog,
)


ORACLE_KIND = "airfoil_v7_finite_catalog_oracle"
ORACLE_SCHEMA_VERSION = 1
ORACLE_MANIFEST_FRAMING = b"agent-evolve:airfoil-v7-finite-oracle-manifest:v1\x00"
ORACLE_RECORD_FRAMING = b"agent-evolve:airfoil-v7-finite-oracle-record:v1\x00"
ORACLE_RESULT_FRAMING = b"agent-evolve:airfoil-v7-finite-oracle-result:v1\x00"
ORACLE_FINALIZATION_FRAMING = (
    b"agent-evolve:airfoil-v7-finite-oracle-finalization:v1\x00"
)
DEFAULT_ORACLE_ROOT = DEFAULT_LIVE_LOG_ROOT / "finite_oracles"
DEFAULT_PRIOR_RUN_DIR = (
    DEFAULT_LIVE_LOG_ROOT
    / "provider_runs"
    / "ae7_exact_action_full_0715_0046"
)
DEFAULT_SECOND_PRIOR_RUN_DIR = (
    DEFAULT_LIVE_LOG_ROOT / "provider_runs" / "ae7_sharded_full_0714_2330"
)
EXPECTED_OPTION_COUNT = 80
EXPECTED_RANS_CALLS = 240
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

PARENT_METRICS = {
    "normalized_multipoint_drag": 1.0015506698765686,
    "normalized_lift_equality": 0.4974082147422454,
}

KNOWN_ACTIONS: tuple[dict[str, object], ...] = (
    {
        "arm": "A",
        "option_id": "shape.camber_aft.p0015",
        "normalized_multipoint_drag": 1.0086768952206244,
        "normalized_lift_equality": 0.4688452250407644,
    },
    {
        "arm": "S",
        "option_id": "trim.p050.n025.n050",
        "normalized_multipoint_drag": 0.9925322559379826,
        "normalized_lift_equality": 0.5367465525554247,
    },
    {
        "arm": "N",
        "option_id": "trim.p025.n025.p050",
        "normalized_multipoint_drag": 1.027073908240716,
        "normalized_lift_equality": 0.4344998536733178,
    },
    {
        "arm": None,
        "option_id": "trim.p050.n025.n025",
        "normalized_multipoint_drag": 0.999521050019065,
        "normalized_lift_equality": 0.5037446062382724,
    },
    {
        "arm": None,
        "option_id": "trim.p050.n050.n025",
        "normalized_multipoint_drag": 0.993398830197159,
        "normalized_lift_equality": 0.5342328531511289,
    },
)


def _parent_reward_from_metrics(*, objective: float, violation: float) -> float:
    """Project frozen scalar metrics through the preregistered parent reward."""

    violation_improvement = PARENT_METRICS[VIOLATION_NAME] - violation
    if violation_improvement >= DELTA_V:
        return 1.0
    if violation_improvement <= -DELTA_V:
        return -1.0
    objective_improvement = PARENT_METRICS[OBJECTIVE_NAME] - objective
    if objective_improvement >= DELTA_F:
        return 1.0
    if objective_improvement <= -DELTA_F:
        return -1.0
    return 0.0


def _resolved_direction(delta: float, threshold: float) -> str:
    if delta >= threshold:
        return "increase"
    if delta <= -threshold:
        return "decrease"
    return "unresolved"


class OracleContractError(RuntimeError):
    """The manifest, journal, source, or receipt graph is inconsistent."""


class OracleRunInvalidated(RuntimeError):
    """One-attempt completeness became impossible; partial evidence remains."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _self_hash(value: Mapping[str, object], *, framing: bytes) -> str:
    return hashlib.sha256(framing + _canonical_bytes(dict(value))).hexdigest()


def _file_binding(path: Path, *, kind: str) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    content = resolved.read_bytes()
    return {
        "kind": kind,
        "path": str(resolved),
        "sha256": _sha256_bytes(content),
        "bytes": len(content),
    }


def _verify_file_binding(value: object, *, kind: str) -> Path:
    if type(value) is not dict or set(value) != {"kind", "path", "sha256", "bytes"}:
        raise OracleContractError(f"{kind} binding has invalid fields")
    if value.get("kind") != kind:
        raise OracleContractError(f"{kind} binding kind changed")
    path = Path(str(value.get("path"))).expanduser().resolve(strict=True)
    content = path.read_bytes()
    if value.get("sha256") != _sha256_bytes(content) or value.get("bytes") != len(
        content
    ):
        raise OracleContractError(f"{kind} binding bytes changed")
    return path


def _validation_report_binding(path: Path, source_sha256: str) -> dict[str, object]:
    try:
        binding = _verification_report_binding(path)
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
        raise OracleContractError("strict verification report validation failed") from exc
    report = binding.get("validated_report")
    if type(report) is not dict or report.get("source_snapshot_sha256") != source_sha256:
        raise OracleContractError(
            "verification report does not pass for the current source snapshot"
        )
    return binding


def _contract_binding() -> tuple[FiniteVariationContract, dict[str, object]]:
    held_out = materialize_held_out_parent()
    if (
        held_out.candidate_sha256 != HELD_OUT_PARENT_CANDIDATE_SHA256
        or held_out.typed_configuration_sha256 != HELD_OUT_PARENT_TYPED_SHA256
    ):
        raise OracleContractError("frozen held-out parent identity changed")
    contract = bind_finite_variation_catalog(
        AirfoilV7UnionVariationCatalog(),
        freeze_json(held_out.candidate),
    )
    if len(contract.options) != EXPECTED_OPTION_COUNT:
        raise OracleContractError("held-out union catalog does not contain 80 options")
    order = []
    for ordinal, option in enumerate(contract.options, start=1):
        child = thaw_json(option.child_configuration)
        order.append(
            {
                "ordinal": ordinal,
                "option_id": option.option_id,
                "family": option.family,
                "option_identity_sha256": option.identity_sha256,
                "typed_child_configuration_sha256": (
                    option.child_configuration_sha256
                ),
                "raw_candidate_sha256": candidate_sha256(child),
                "child_configuration": child,
            }
        )
    if len({row["raw_candidate_sha256"] for row in order}) != EXPECTED_OPTION_COUNT:
        raise OracleContractError("union catalog raw candidates are not distinct")
    return contract, {
        "parent": held_out.to_record(),
        "contract": contract.evidence_record(),
        "evaluation_order": order,
    }


def _evaluator_binding() -> dict[str, object]:
    settings = local_default_converged_settings()
    return {
        "identity": EVALUATOR_IDENTITY.to_record(),
        "python_executable": str(settings.python_executable.absolute()),
        "evaluator_script": str(settings.evaluator_script.resolve(strict=True)),
        "dataset_arrow": str(settings.dataset_arrow.resolve(strict=True)),
        "expected_dataset_sha256": settings.expected_dataset_sha256,
        "cpu_set": settings.cpu_set,
        "mpi_cores": settings.mpi_cores,
        "candidate_timeout_seconds": settings.timeout_seconds,
        "candidate_concurrency": 1,
        "provider_calls": 0,
        "credentials_read": False,
    }


def _execution_binding() -> dict[str, object]:
    return {
        "ordered_candidate_attempts": EXPECTED_OPTION_COUNT,
        "expected_full_rans_calls": EXPECTED_RANS_CALLS,
        "attempts_per_option": 1,
        "candidate_concurrency": 1,
        "attempt_charge": "immutable_started_record_before_evaluator_entry",
        "terminal_commit": "fsynced_receipt_then_authenticated_typed_projection",
        "resume": "verified_success_prefix_and_at_most_one_open_start",
        "open_start_without_receipt": "invalidate_no_retry",
        "any_failed_payload": "invalidate_complete_ranking",
        "provider_calls": 0,
        "credentials_read": False,
    }


def _analysis_binding() -> dict[str, object]:
    return {
        "status": "post_hoc_exploratory_diagnostic",
        "claim_boundary": (
            "cannot replace the sealed ae7_exact_action_full_0715_0046 verdict "
            "or establish AgentEvolve efficacy"
        ),
        "known_before_oracle": [dict(row) for row in KNOWN_ACTIONS],
        "held_out_parent_metrics": dict(PARENT_METRICS),
        "ranking": {
            "eligibility": "all_80_options_require_successful_typed_evaluation",
            "primary": f"violation:{VIOLATION_NAME}:ascending",
            "secondary": f"objective:{OBJECTIVE_NAME}:ascending",
            "scientific_rank": "one_plus_count_strictly_better_exact_ties_share_rank",
            "display_only_tie_order": "option_id_unsigned_ascii_ascending",
        },
        "planned_outputs": [
            "complete_order",
            "one_based_rank",
            "family_rank",
            "better_equivalent_worse_counts",
            "contextual_reward_mass_plus_zero_minus",
            "exact_uniform_one_action_mass",
            "uniform_one_action_rank_and_reward_mass_overall_and_by_family",
            "exact_all_82160_three_action_portfolios",
            "tie_explicit_observed_asn_portfolio_percentile",
            "ordinary_two_objective_nondominated_set",
            "known_action_ranks",
            "known_action_percentiles",
            "five_known_action_fresh_minus_prior_deltas",
            "thresholded_a_s_direction_and_reward_stability_decisions",
            "prospective_rank_band_and_random_median_decisions",
        ],
        "portfolio_quantiles": "nearest_rank_at_25_50_75_percent",
        "observed_asn_percentile": {
            "definition": "fraction_with_strictly_better_best_rank",
            "orientation": "zero_is_best",
            "tie_mass_reported_separately": True,
        },
        "fresh_repeat_adjudication": {
            "direction_delta_reference": "fresh_or_prior_minus_frozen_parent",
            "direction_resolution": {
                "f": DELTA_F,
                "v": DELTA_V,
            },
            "A_expected": {"f": "increase", "v": "decrease"},
            "S_expected": {"f": "decrease", "v": "increase"},
            "kill_if_direction_or_contextual_reward_changes": True,
        },
        "prospective_decisions": {
            "adaptive_rank_1_20": "retain_top_quartile_selection_hypothesis",
            "adaptive_rank_21_40": "require_selector_expansion_or_redesign",
            "adaptive_rank_41_80": "kill_unchanged_single_card_selection_rule",
            "reject_three_arm_competitiveness_if": (
                "observed_asn_best_rank_strictly_worse_than_random_three_action_median"
            ),
        },
        "outcome_relation": {
            "policy_id": AIRFOIL_V7_ARCHIVE_RELATION.policy_id,
            "policy_version": AIRFOIL_V7_ARCHIVE_RELATION.policy_version,
            "definition_sha256": ARCHIVE_DEFINITION_SHA256,
        },
        "contextual_reward": {
            "definition_sha256": REWARD_DEFINITION_SHA256,
            "bound_definition_sha256": AIRFOIL_V7_REWARD_BINDING.definition_hash,
            "delta_f": DELTA_F,
            "delta_v": DELTA_V,
        },
        "phenotype_policy_definition_sha256": PHENOTYPE_DEFINITION_SHA256,
        "optimization_semantics": AIRFOIL_V7_OPTIMIZATION_SEMANTICS.to_record(),
    }


def _prior_run_binding(
    path: Path,
    *,
    expected_slots: Mapping[str, str],
) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise OracleContractError("prior run binding is not a directory")
    bindings = {
        name: _file_binding(resolved / name, kind=f"prior_run_{name}")
        for name in ("result.json", "adjudication.json", "finalized.json")
    }
    result = json.loads((resolved / "result.json").read_bytes())
    if type(result) is not dict:
        raise OracleContractError("prior run result root is not an object")
    seeds = result.get("seeds")
    held_out_rows = (
        []
        if type(seeds) is not list
        else [
            row
            for row in seeds
            if type(row) is dict
            and row.get("configuration_sha256") == HELD_OUT_PARENT_CANDIDATE_SHA256
        ]
    )
    if len(held_out_rows) != 1:
        raise OracleContractError("prior run does not bind the frozen held-out parent")

    def successful_metrics(candidate: object, *, label: str) -> tuple[float, float]:
        if type(candidate) is not dict:
            raise OracleContractError(f"{label} candidate is malformed")
        detailed = candidate.get("detailed_evaluation")
        if (
            candidate.get("valid") is not True
            or candidate.get("evidence_compliant") is not True
            or type(detailed) is not dict
            or detailed.get("failure") is not None
            or set(detailed.get("objectives", {})) != {OBJECTIVE_NAME}
            or set(detailed.get("violations", {})) != {VIOLATION_NAME}
        ):
            raise OracleContractError(f"{label} is not one successful exact evaluation")
        objective = detailed["objectives"][OBJECTIVE_NAME]
        violation = detailed["violations"][VIOLATION_NAME]
        if (
            isinstance(objective, bool)
            or not isinstance(objective, (int, float))
            or isinstance(violation, bool)
            or not isinstance(violation, (int, float))
            or not math.isfinite(float(objective))
            or not math.isfinite(float(violation))
        ):
            raise OracleContractError(f"{label} metrics are malformed")
        return float(objective), float(violation)

    parent_f, parent_v = successful_metrics(
        held_out_rows[0],
        label="prior held-out parent",
    )
    if (parent_f, parent_v) != (
        PARENT_METRICS[OBJECTIVE_NAME],
        PARENT_METRICS[VIOLATION_NAME],
    ):
        raise OracleContractError("prior held-out parent metrics changed")
    _, catalog = _contract_binding()
    child_to_option = {
        str(row["raw_candidate_sha256"]): str(row["option_id"])
        for row in catalog["evaluation_order"]
    }
    expected_metrics = {
        str(item["option_id"]): (
            float(item[OBJECTIVE_NAME]),
            float(item[VIOLATION_NAME]),
        )
        for item in KNOWN_ACTIONS
    }
    observed: dict[str, str] = {}
    generations = result.get("generations")
    if type(generations) is list:
        for generation in generations:
            if type(generation) is not dict or generation.get("generation") != 2:
                continue
            slots = generation.get("slots")
            if type(slots) is not list:
                continue
            for slot in slots:
                if type(slot) is not dict:
                    continue
                candidate = slot.get("candidate")
                if type(candidate) is dict and type(slot.get("slot_id")) is str:
                    candidate_hash = candidate.get("configuration_sha256")
                    option_id = child_to_option.get(str(candidate_hash))
                    if option_id is not None:
                        slot_id = str(slot["slot_id"])
                        if slot_id in observed:
                            raise OracleContractError("prior run repeats a generation-2 slot")
                        observed[slot_id] = option_id
                        metrics = successful_metrics(
                            candidate,
                            label=f"prior slot {slot_id}",
                        )
                        if metrics != expected_metrics.get(option_id):
                            raise OracleContractError(
                                f"prior slot {slot_id} numerical facts changed"
                            )
                        expected_reward = _parent_reward_from_metrics(
                            objective=metrics[0],
                            violation=metrics[1],
                        )
                        if slot.get("reward") != expected_reward:
                            raise OracleContractError(
                                f"prior slot {slot_id} contextual reward changed"
                            )
    expected = dict(expected_slots)
    if observed != expected:
        raise OracleContractError("prior run A/S/N exact actions changed")
    return {
        "run_id": resolved.name,
        "run_dir": str(resolved),
        "files": bindings,
    }


def _validate_run_target(run_id: object, output_dir: Path) -> tuple[str, Path]:
    if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
        raise OracleContractError("oracle run_id has invalid syntax")
    resolved = output_dir.expanduser().resolve()
    if resolved.name != run_id:
        raise OracleContractError("oracle output directory must end in run_id")
    return run_id, resolved


def write_oracle_manifest(
    path: Path,
    *,
    run_id: str,
    output_dir: Path,
    verification_report_path: Path,
    prior_run_dir: Path = DEFAULT_PRIOR_RUN_DIR,
    second_prior_run_dir: Path = DEFAULT_SECOND_PRIOR_RUN_DIR,
    source_snapshot_factory: Callable[[], dict[str, object]] = source_snapshot,
    enforce_canonical_output: bool = True,
) -> dict[str, object]:
    """Build and atomically publish a zero-provider oracle manifest."""

    run_id, output_dir = _validate_run_target(run_id, output_dir)
    if enforce_canonical_output and output_dir.parent != DEFAULT_ORACLE_ROOT.resolve():
        raise OracleContractError("oracle output directory is outside its canonical root")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    snapshot = source_snapshot_factory()
    if type(snapshot) is not dict or _SHA256.fullmatch(
        str(snapshot.get("sha256"))
    ) is None:
        raise OracleContractError("source snapshot is malformed")
    _, catalog = _contract_binding()
    oracle = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "verification_report": _validation_report_binding(
            verification_report_path,
            str(snapshot["sha256"]),
        ),
        "prior_runs": [
            _prior_run_binding(
                prior_run_dir,
                expected_slots={
                    "A": "shape.camber_aft.p0015",
                    "S": "trim.p050.n025.n050",
                    "N": "trim.p025.n025.p050",
                },
            ),
            _prior_run_binding(
                second_prior_run_dir,
                expected_slots={
                    "A": "trim.p050.n025.n025",
                    "S": "trim.p050.n025.n050",
                    "N": "trim.p050.n050.n025",
                },
            ),
        ],
        "catalog": catalog,
        "evaluator": _evaluator_binding(),
        "resource_lease": _resource_lease_manifest_record(
            phase="finite_catalog_oracle"
        ),
        "execution": _execution_binding(),
        "analysis": _analysis_binding(),
    }
    unsigned: dict[str, object] = {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "kind": ORACLE_KIND,
        "built_at_utc": _utc_now(),
        "source_snapshot": snapshot,
        "oracle": oracle,
    }
    record = {
        **unsigned,
        "manifest_sha256": _self_hash(
            unsigned,
            framing=ORACLE_MANIFEST_FRAMING,
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, record)
    return record


@dataclass(frozen=True, slots=True)
class VerifiedOracleManifest:
    path: Path
    record: dict[str, object]
    run_id: str
    output_dir: Path
    manifest_sha256: str
    source_sha256: str
    contract: FiniteVariationContract


def verify_oracle_manifest(
    path: Path,
    *,
    require_output_absent: bool | None,
    source_snapshot_factory: Callable[[], dict[str, object]] = source_snapshot,
    enforce_canonical_output: bool = True,
) -> VerifiedOracleManifest:
    """Authenticate a manifest and recompute every deterministic binding."""

    if require_output_absent is not None and type(require_output_absent) is not bool:
        raise TypeError("require_output_absent must be an exact bool or None")
    resolved = path.expanduser().resolve(strict=True)
    try:
        record = json.loads(resolved.read_bytes())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OracleContractError("oracle manifest is not valid JSON") from exc
    if type(record) is not dict or set(record) != {
        "schema_version",
        "kind",
        "built_at_utc",
        "source_snapshot",
        "oracle",
        "manifest_sha256",
    }:
        raise OracleContractError("oracle manifest fields changed")
    claimed = record.get("manifest_sha256")
    unsigned = dict(record)
    unsigned.pop("manifest_sha256")
    if (
        record.get("schema_version") != ORACLE_SCHEMA_VERSION
        or record.get("kind") != ORACLE_KIND
        or claimed != _self_hash(unsigned, framing=ORACLE_MANIFEST_FRAMING)
    ):
        raise OracleContractError("oracle manifest self-hash or identity failed")
    oracle = record.get("oracle")
    if type(oracle) is not dict or set(oracle) != {
        "run_id",
        "output_dir",
        "verification_report",
        "prior_runs",
        "catalog",
        "evaluator",
        "resource_lease",
        "execution",
        "analysis",
    }:
        raise OracleContractError("oracle specification fields changed")
    run_id, output_dir = _validate_run_target(
        oracle.get("run_id"),
        Path(str(oracle.get("output_dir"))),
    )
    if enforce_canonical_output and output_dir.parent != DEFAULT_ORACLE_ROOT.resolve():
        raise OracleContractError("oracle output directory is outside its canonical root")
    if require_output_absent is True and output_dir.exists():
        raise FileExistsError(output_dir)
    if require_output_absent is False and not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    current_source = source_snapshot_factory()
    if record.get("source_snapshot") != current_source:
        raise OracleContractError("oracle source snapshot changed")
    source_sha256 = str(current_source.get("sha256"))
    report_value = oracle.get("verification_report")
    if type(report_value) is not dict or type(report_value.get("path")) is not str:
        raise OracleContractError("oracle verification report binding is malformed")
    report_path = Path(str(report_value["path"])).resolve(strict=True)
    if report_value != _validation_report_binding(report_path, source_sha256):
        raise OracleContractError("oracle verification report binding changed")
    prior = oracle.get("prior_runs")
    if type(prior) is not list or len(prior) != 2:
        raise OracleContractError("oracle prior-run list changed")
    expected_prior = [
        _prior_run_binding(
            Path(str(prior[0].get("run_dir"))),
            expected_slots={
                "A": "shape.camber_aft.p0015",
                "S": "trim.p050.n025.n050",
                "N": "trim.p025.n025.p050",
            },
        ),
        _prior_run_binding(
            Path(str(prior[1].get("run_dir"))),
            expected_slots={
                "A": "trim.p050.n025.n025",
                "S": "trim.p050.n025.n050",
                "N": "trim.p050.n050.n025",
            },
        ),
    ]
    if prior != expected_prior:
        raise OracleContractError("oracle prior-run binding changed")
    contract, catalog = _contract_binding()
    if oracle.get("catalog") != catalog:
        raise OracleContractError("oracle held-out catalog binding changed")
    if (
        oracle.get("evaluator") != _evaluator_binding()
        or oracle.get("resource_lease")
        != _resource_lease_manifest_record(phase="finite_catalog_oracle")
        or oracle.get("execution") != _execution_binding()
        or oracle.get("analysis") != _analysis_binding()
    ):
        raise OracleContractError("oracle evaluator, execution, or analysis changed")
    return VerifiedOracleManifest(
        path=resolved,
        record=record,
        run_id=run_id,
        output_dir=output_dir,
        manifest_sha256=str(claimed),
        source_sha256=source_sha256,
        contract=contract,
    )


def _payload_record(payload: DetailedEvaluationPayload) -> dict[str, object]:
    failure = payload.failure
    return {
        "failure": (
            None
            if failure is None
            else {
                "category": failure.category.value,
                "code": failure.code.value,
                "message": failure.message,
                "retryable": failure.retryable,
                "exception_type": failure.exception_type,
                "diagnostics_artifact_id": (
                    None
                    if failure.diagnostics_artifact_id is None
                    else failure.diagnostics_artifact_id.value
                ),
            }
        ),
        "objectives": dict(payload.objectives),
        "violations": dict(payload.violations),
        "checks": [check.to_record() for check in payload.checks],
        "receipt": (
            None
            if payload.receipt is None
            else {
                "artifact_id": payload.receipt.artifact_id.value,
                "sha256": payload.receipt.sha256_hex,
                "bytes": payload.receipt.size_bytes,
                "media_type": payload.receipt.media_type,
            }
        ),
        "evaluator": payload.evaluator.to_record(),
        "active_wall_seconds": payload.active_wall_seconds,
        "resource_queue_wall_seconds": payload.resource_queue_wall_seconds,
    }


def _sealed_record(unsigned: Mapping[str, object]) -> dict[str, object]:
    return {
        **dict(unsigned),
        "record_sha256": _self_hash(unsigned, framing=ORACLE_RECORD_FRAMING),
    }


def _read_sealed_record(path: Path, *, kind: str) -> dict[str, object]:
    try:
        record = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OracleContractError(f"{kind} record is unreadable") from exc
    if type(record) is not dict or record.get("kind") != kind:
        raise OracleContractError(f"{kind} record identity changed")
    claimed = record.get("record_sha256")
    unsigned = dict(record)
    unsigned.pop("record_sha256", None)
    if claimed != _self_hash(unsigned, framing=ORACLE_RECORD_FRAMING):
        raise OracleContractError(f"{kind} record self-hash failed")
    return record


def _option_directory(run_dir: Path, ordinal: int, option_id: str) -> Path:
    return run_dir / "options" / f"{ordinal:03d}-{option_id}"


def _option_identity(
    option: FiniteVariationOption,
    *,
    ordinal: int,
) -> dict[str, object]:
    child = thaw_json(option.child_configuration)
    return {
        "ordinal": ordinal,
        "option_id": option.option_id,
        "family": option.family,
        "option_identity_sha256": option.identity_sha256,
        "typed_child_configuration_sha256": option.child_configuration_sha256,
        "raw_candidate_sha256": candidate_sha256(child),
    }


def _write_started(
    option_dir: Path,
    *,
    verified: VerifiedOracleManifest,
    invocation_id: str,
    option: FiniteVariationOption,
    ordinal: int,
) -> dict[str, object]:
    option_dir.mkdir(parents=False, exist_ok=False)
    _fsync_directory(option_dir.parent)
    record = _sealed_record(
        {
            "schema_version": ORACLE_SCHEMA_VERSION,
            "kind": "oracle_option_started",
            "run_id": verified.run_id,
            "manifest_sha256": verified.manifest_sha256,
            "source_sha256": verified.source_sha256,
            "invocation_id": invocation_id,
            "attempt": 1,
            **_option_identity(option, ordinal=ordinal),
            "started_at_utc": _utc_now(),
            "attempt_charged_before_evaluator_entry": True,
        }
    )
    write_json_atomic(option_dir / "started.json", record)
    return record


def _validate_started(
    record: Mapping[str, object],
    *,
    verified: VerifiedOracleManifest,
    option: FiniteVariationOption,
    ordinal: int,
) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "run_id",
        "manifest_sha256",
        "source_sha256",
        "invocation_id",
        "attempt",
        "ordinal",
        "option_id",
        "family",
        "option_identity_sha256",
        "typed_child_configuration_sha256",
        "raw_candidate_sha256",
        "started_at_utc",
        "attempt_charged_before_evaluator_entry",
        "record_sha256",
    }
    if (
        set(record) != expected_keys
        or record.get("schema_version") != ORACLE_SCHEMA_VERSION
        or record.get("run_id") != verified.run_id
        or record.get("manifest_sha256") != verified.manifest_sha256
        or record.get("source_sha256") != verified.source_sha256
        or record.get("attempt") != 1
        or record.get("attempt_charged_before_evaluator_entry") is not True
        or type(record.get("invocation_id")) is not str
        or re.fullmatch(r"[0-9]{4}", str(record.get("invocation_id"))) is None
        or type(record.get("started_at_utc")) is not str
        or not str(record.get("started_at_utc"))
        or any(
            record.get(key) != value
            for key, value in _option_identity(option, ordinal=ordinal).items()
        )
    ):
        raise OracleContractError("option started record changed")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_receipt(path: Path, *, run_dir: Path) -> None:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(run_dir.resolve()):
        raise OracleContractError("raw oracle receipt escapes its run directory")
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(resolved.parent)


def _receipt_binding(
    path: Path,
    *,
    run_dir: Path,
    expected_candidate_sha256: str,
) -> dict[str, object]:
    _fsync_receipt(path, run_dir=run_dir)
    resolved = path.resolve(strict=True)
    content = resolved.read_bytes()
    try:
        raw = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OracleContractError("raw oracle receipt is not valid JSON") from exc
    if (
        type(raw) is not dict
        or raw.get("candidate_sha256") != expected_candidate_sha256
        or type(raw.get("evaluator_calls")) is not int
        or not 0 <= int(raw["evaluator_calls"]) <= 3
    ):
        raise OracleContractError("raw oracle receipt candidate/accounting changed")
    return {
        "relative_path": resolved.relative_to(run_dir.resolve()).as_posix(),
        "sha256": _sha256_bytes(content),
        "bytes": len(content),
        "raw_candidate_sha256": expected_candidate_sha256,
        "raw_solver_calls": int(raw["evaluator_calls"]),
    }


def _verify_receipt_binding(
    value: object,
    *,
    run_dir: Path,
    expected_candidate_sha256: str,
) -> Path:
    if type(value) is not dict or set(value) != {
        "relative_path",
        "sha256",
        "bytes",
        "raw_candidate_sha256",
        "raw_solver_calls",
    }:
        raise OracleContractError("terminal raw-receipt binding fields changed")
    path = (run_dir / str(value.get("relative_path"))).resolve(strict=True)
    if not path.is_relative_to(run_dir.resolve()):
        raise OracleContractError("terminal raw receipt escapes the run directory")
    observed = _receipt_binding(
        path,
        run_dir=run_dir,
        expected_candidate_sha256=expected_candidate_sha256,
    )
    if value != observed:
        raise OracleContractError("terminal raw-receipt binding changed")
    return path


def _find_uncommitted_receipts(
    run_dir: Path,
    *,
    referenced: set[str],
) -> tuple[Path, ...]:
    root = run_dir / "raw_receipts"
    if not root.exists():
        return ()
    rows = []
    for path in root.rglob("*.json"):
        relative = path.resolve().relative_to(run_dir.resolve()).as_posix()
        if relative not in referenced:
            rows.append(path.resolve())
    return tuple(sorted(rows, key=lambda item: item.as_posix()))


def _locate_current_receipt(
    run_dir: Path,
    *,
    referenced: set[str],
    expected_candidate_sha256: str,
) -> Path | None:
    uncommitted = _find_uncommitted_receipts(run_dir, referenced=referenced)
    matching = tuple(
        path for path in uncommitted if path.name == f"{expected_candidate_sha256}.json"
    )
    if len(matching) > 1 or len(uncommitted) != len(matching):
        raise OracleContractError("raw receipt set is not one canonical open attempt")
    return None if not matching else matching[0]


def _terminal_record(
    *,
    verified: VerifiedOracleManifest,
    option: FiniteVariationOption,
    ordinal: int,
    started: Mapping[str, object],
    payload: DetailedEvaluationPayload,
    receipt: dict[str, object] | None,
    outer_wall_seconds: float | None,
    recovered_after_interruption: bool,
    post_evaluation_source: Mapping[str, object],
) -> dict[str, object]:
    if payload.evaluator.to_record() != _evaluator_binding()["identity"]:
        raise OracleContractError("oracle payload evaluator identity changed")
    if outer_wall_seconds is not None and (
        not math.isfinite(outer_wall_seconds) or outer_wall_seconds < 0
    ):
        raise OracleContractError("oracle outer wall time is invalid")
    if type(recovered_after_interruption) is not bool:
        raise TypeError("recovered_after_interruption must be an exact bool")
    if (
        post_evaluation_source.get("source_sha256") != verified.source_sha256
        or type(post_evaluation_source.get("observed_at_utc")) is not str
    ):
        raise OracleContractError("post-evaluation source verification is malformed")
    disposition = "success" if payload.failure is None else "failed_invalidate"
    return _sealed_record(
        {
            "schema_version": ORACLE_SCHEMA_VERSION,
            "kind": "oracle_option_terminal",
            "run_id": verified.run_id,
            "manifest_sha256": verified.manifest_sha256,
            **_option_identity(option, ordinal=ordinal),
            "attempt": 1,
            "started_record_sha256": started["record_sha256"],
            "invocation_id": started["invocation_id"],
            "source_sha256": verified.source_sha256,
            "post_evaluation_source": dict(post_evaluation_source),
            "disposition": disposition,
            "payload": _payload_record(payload),
            "raw_receipt": receipt,
            "outer_wall_seconds": outer_wall_seconds,
            "recovered_after_interruption": recovered_after_interruption,
            "finished_at_utc": _utc_now(),
        }
    )


def _write_terminal(
    option_dir: Path,
    *,
    verified: VerifiedOracleManifest,
    option: FiniteVariationOption,
    ordinal: int,
    started: Mapping[str, object],
    payload: DetailedEvaluationPayload,
    receipt: dict[str, object] | None,
    outer_wall_seconds: float | None,
    recovered_after_interruption: bool,
    post_evaluation_source: Mapping[str, object],
) -> dict[str, object]:
    record = _terminal_record(
        verified=verified,
        option=option,
        ordinal=ordinal,
        started=started,
        payload=payload,
        receipt=receipt,
        outer_wall_seconds=outer_wall_seconds,
        recovered_after_interruption=recovered_after_interruption,
        post_evaluation_source=post_evaluation_source,
    )
    write_json_atomic(option_dir / "terminal.json", record)
    return record


def _validate_terminal(
    record: Mapping[str, object],
    *,
    verified: VerifiedOracleManifest,
    option: FiniteVariationOption,
    ordinal: int,
    started: Mapping[str, object],
    run_dir: Path,
    receipt_replayer: Callable[[Path, object], DetailedEvaluationPayload],
) -> tuple[DetailedEvaluationPayload, str | None]:
    expected_keys = {
        "schema_version",
        "kind",
        "run_id",
        "manifest_sha256",
        "ordinal",
        "option_id",
        "family",
        "option_identity_sha256",
        "typed_child_configuration_sha256",
        "raw_candidate_sha256",
        "attempt",
        "started_record_sha256",
        "invocation_id",
        "source_sha256",
        "post_evaluation_source",
        "disposition",
        "payload",
        "raw_receipt",
        "outer_wall_seconds",
        "recovered_after_interruption",
        "finished_at_utc",
        "record_sha256",
    }
    if (
        set(record) != expected_keys
        or record.get("schema_version") != ORACLE_SCHEMA_VERSION
        or record.get("run_id") != verified.run_id
        or record.get("manifest_sha256") != verified.manifest_sha256
        or record.get("attempt") != 1
        or record.get("started_record_sha256") != started.get("record_sha256")
        or record.get("invocation_id") != started.get("invocation_id")
        or record.get("source_sha256") != verified.source_sha256
        or type(record.get("recovered_after_interruption")) is not bool
        or type(record.get("finished_at_utc")) is not str
        or not str(record.get("finished_at_utc"))
        or any(
            record.get(key) != value
            for key, value in _option_identity(option, ordinal=ordinal).items()
        )
    ):
        raise OracleContractError("option terminal record changed")
    outer = record.get("outer_wall_seconds")
    if outer is not None and (
        isinstance(outer, bool)
        or not isinstance(outer, (int, float))
        or not math.isfinite(float(outer))
        or float(outer) < 0
    ):
        raise OracleContractError("option terminal outer wall time is invalid")
    post_source = record.get("post_evaluation_source")
    if (
        type(post_source) is not dict
        or post_source.get("source_sha256") != verified.source_sha256
        or type(post_source.get("observed_at_utc")) is not str
    ):
        raise OracleContractError("terminal post-evaluation source check changed")
    raw_binding = record.get("raw_receipt")
    if raw_binding is None:
        raise OracleContractError("oracle terminal record lacks a raw receipt")
    child = normalize_candidate(thaw_json(option.child_configuration))
    path = _verify_receipt_binding(
        raw_binding,
        run_dir=run_dir,
        expected_candidate_sha256=candidate_sha256(child),
    )
    payload = receipt_replayer(path, child)
    if record.get("payload") != _payload_record(payload):
        raise OracleContractError("terminal typed payload differs from its raw receipt")
    expected_disposition = "success" if payload.failure is None else "failed_invalidate"
    if record.get("disposition") != expected_disposition:
        raise OracleContractError("terminal disposition differs from its typed payload")
    relative = path.relative_to(run_dir.resolve()).as_posix()
    return payload, relative


@dataclass(frozen=True, slots=True)
class OracleJournalState:
    terminal_records: tuple[dict[str, object], ...]
    terminal_payloads: tuple[DetailedEvaluationPayload, ...]
    next_ordinal: int
    open_started: dict[str, object] | None
    referenced_receipts: frozenset[str]
    failed_ordinal: int | None


def _read_journal(
    run_dir: Path,
    *,
    verified: VerifiedOracleManifest,
    receipt_replayer: Callable[[Path, object], DetailedEvaluationPayload],
) -> OracleJournalState:
    options_root = run_dir / "options"
    if not options_root.is_dir():
        raise OracleContractError("oracle options journal root is missing")
    expected_names = {
        _option_directory(run_dir, ordinal, option.option_id).name
        for ordinal, option in enumerate(verified.contract.options, start=1)
    }
    entries = tuple(options_root.iterdir())
    if any(not path.is_dir() for path in entries):
        raise OracleContractError("oracle options journal contains non-directories")
    actual_names = {path.name for path in entries}
    if not actual_names.issubset(expected_names):
        raise OracleContractError("oracle option journal contains unknown directories")
    terminals: list[dict[str, object]] = []
    payloads: list[DetailedEvaluationPayload] = []
    referenced: set[str] = set()
    open_started: dict[str, object] | None = None
    failed_ordinal: int | None = None
    next_ordinal = 1
    for ordinal, option in enumerate(verified.contract.options, start=1):
        option_dir = _option_directory(run_dir, ordinal, option.option_id)
        later = actual_names - {
            _option_directory(run_dir, prior, prior_option.option_id).name
            for prior, prior_option in enumerate(
                verified.contract.options[: ordinal - 1],
                start=1,
            )
        }
        if not option_dir.exists():
            if later:
                raise OracleContractError("oracle option journal is not a prefix")
            next_ordinal = ordinal
            break
        started_path = option_dir / "started.json"
        allowed_names = {
            "started.json",
            "source_verified.json",
            "post_evaluation_source_verified.json",
            "terminal.json",
        }
        if any(path.name not in allowed_names for path in option_dir.iterdir()):
            raise OracleContractError("oracle option directory contains unknown files")
        if not started_path.is_file():
            raise OracleContractError("oracle option directory lacks started.json")
        started = _read_sealed_record(started_path, kind="oracle_option_started")
        _validate_started(
            started,
            verified=verified,
            option=option,
            ordinal=ordinal,
        )
        terminal_path = option_dir / "terminal.json"
        if not terminal_path.exists():
            if later != {option_dir.name}:
                raise OracleContractError("open oracle option is not the journal tail")
            open_started = started
            next_ordinal = ordinal
            break
        terminal = _read_sealed_record(
            terminal_path,
            kind="oracle_option_terminal",
        )
        payload, receipt_relative = _validate_terminal(
            terminal,
            verified=verified,
            option=option,
            ordinal=ordinal,
            started=started,
            run_dir=run_dir,
            receipt_replayer=receipt_replayer,
        )
        terminals.append(terminal)
        payloads.append(payload)
        if receipt_relative is not None:
            if receipt_relative in referenced:
                raise OracleContractError("raw receipt is committed more than once")
            referenced.add(receipt_relative)
        if payload.failure is not None:
            failed_ordinal = ordinal
            next_ordinal = ordinal
            if later != {option_dir.name}:
                raise OracleContractError("failed oracle option is not the journal tail")
            break
        next_ordinal = ordinal + 1
    else:
        next_ordinal = EXPECTED_OPTION_COUNT + 1
    return OracleJournalState(
        terminal_records=tuple(terminals),
        terminal_payloads=tuple(payloads),
        next_ordinal=next_ordinal,
        open_started=open_started,
        referenced_receipts=frozenset(referenced),
        failed_ordinal=failed_ordinal,
    )


@dataclass(frozen=True, slots=True)
class OracleExecutionDependencies:
    benchmark_factory: Callable[[str, Path], AgenticBenchmark]
    resource_lease_factory: Callable[[str, str], ExclusiveResourceLease]
    source_snapshot_factory: Callable[[], dict[str, object]] = source_snapshot
    receipt_replayer: Callable[
        [Path, object], DetailedEvaluationPayload
    ] = replay_airfoil_v7_durable_receipt
    monotonic_ns: Callable[[], int] = time.monotonic_ns
    after_raw_receipt: Callable[[Path, FiniteVariationOption], None] | None = None
    enforce_canonical_output: bool = True

    def __post_init__(self) -> None:
        for name in (
            "benchmark_factory",
            "resource_lease_factory",
            "source_snapshot_factory",
            "receipt_replayer",
            "monotonic_ns",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        if self.after_raw_receipt is not None and not callable(
            self.after_raw_receipt
        ):
            raise TypeError("after_raw_receipt must be callable or None")
        if type(self.enforce_canonical_output) is not bool:
            raise TypeError("enforce_canonical_output must be an exact bool")


def production_oracle_dependencies() -> OracleExecutionDependencies:
    """Return the zero-provider production composition for the oracle."""

    return OracleExecutionDependencies(
        benchmark_factory=create_seed_qualification_benchmark,
        resource_lease_factory=_production_resource_lease,
    )


def _run_lock_path(verified: VerifiedOracleManifest) -> Path:
    return (
        verified.output_dir.parent
        / ".finite_oracle_run_locks"
        / f"{verified.run_id}.lock"
    )


def _new_run_lock(verified: VerifiedOracleManifest) -> FileExclusiveResourceLease:
    return FileExclusiveResourceLease(
        resource_key=f"airfoil_v7_finite_oracle_run:{verified.run_id}",
        owner_id=f"{verified.run_id}.pid-{os.getpid()}",
        lease_path=_run_lock_path(verified),
        owner_metadata={
            "manifest_sha256": verified.manifest_sha256,
            "output_dir": str(verified.output_dir),
            "scope": "run_mutation_through_closure",
        },
    )


def _next_invocation_dir(run_dir: Path) -> tuple[str, Path]:
    root = run_dir / "invocations"
    root.mkdir(exist_ok=True)
    ordinals = []
    for path in root.iterdir():
        if not path.is_dir() or re.fullmatch(r"[0-9]{4}", path.name) is None:
            raise OracleContractError("oracle invocation directory set is malformed")
        ordinals.append(int(path.name))
    ordinal = max(ordinals, default=0) + 1
    invocation_id = f"{ordinal:04d}"
    path = root / invocation_id
    path.mkdir(exist_ok=False)
    _fsync_directory(root)
    return invocation_id, path


def _verify_current_source(
    verified: VerifiedOracleManifest,
    dependencies: OracleExecutionDependencies,
    *,
    stage: str,
) -> dict[str, object]:
    current = dependencies.source_snapshot_factory()
    if current != verified.record["source_snapshot"]:
        raise OracleContractError(f"oracle source changed at {stage}")
    return {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "stage": stage,
        "source_sha256": verified.source_sha256,
        "observed_at_utc": _utc_now(),
    }


def _write_invalidation(
    run_dir: Path,
    *,
    verified: VerifiedOracleManifest,
    reason: str,
    ordinal: int | None,
    option_id: str | None,
) -> dict[str, object]:
    path = run_dir / "invalidation.json"
    if path.exists():
        return _read_sealed_record(path, kind="oracle_run_invalidated")
    record = _sealed_record(
        {
            "schema_version": ORACLE_SCHEMA_VERSION,
            "kind": "oracle_run_invalidated",
            "run_id": verified.run_id,
            "manifest_sha256": verified.manifest_sha256,
            "source_sha256": verified.source_sha256,
            "reason": reason,
            "ordinal": ordinal,
            "option_id": option_id,
            "complete_ranking_available": False,
            "provider_calls": 0,
            "credentials_read": False,
            "invalidated_at_utc": _utc_now(),
        }
    )
    write_json_atomic(path, record)
    return record


def _result_record(
    verified: VerifiedOracleManifest,
    state: OracleJournalState,
) -> dict[str, object]:
    if (
        len(state.terminal_records) != EXPECTED_OPTION_COUNT
        or len(state.terminal_payloads) != EXPECTED_OPTION_COUNT
        or any(payload.failure is not None for payload in state.terminal_payloads)
    ):
        raise OracleContractError("complete oracle result requires 80 successes")
    phenotype = AirfoilV7PhenotypeIdentityPolicy()

    def detailed(configuration: object, payload: DetailedEvaluationPayload) -> DetailedEvaluation:
        return DetailedEvaluation(
            phenotype=phenotype.identify(configuration),
            payload=payload,
            timings=EvaluationTimings(
                total_wall_seconds=0.0,
                active_wall_seconds=payload.active_wall_seconds,
                resource_queue_wall_seconds=payload.resource_queue_wall_seconds,
            ),
        )

    parent_payload = DetailedEvaluationPayload(
        failure=None,
        objectives=((OBJECTIVE_NAME, float(PARENT_METRICS[OBJECTIVE_NAME])),),
        violations=((VIOLATION_NAME, float(PARENT_METRICS[VIOLATION_NAME])),),
        checks=(),
        receipt=None,
        evaluator=EVALUATOR_IDENTITY,
        active_wall_seconds=None,
        resource_queue_wall_seconds=None,
    )
    parent_configuration = materialize_held_out_parent().candidate
    parent_detailed = detailed(parent_configuration, parent_payload)
    detailed_rows = tuple(
        detailed(thaw_json(option.child_configuration), payload)
        for option, payload in zip(
            verified.contract.options,
            state.terminal_payloads,
            strict=True,
        )
    )
    metrics = tuple(
        (
            dict(payload.violations)[VIOLATION_NAME],
            dict(payload.objectives)[OBJECTIVE_NAME],
        )
        for payload in state.terminal_payloads
    )
    if any(not all(math.isfinite(value) for value in row) for row in metrics):
        raise OracleContractError("oracle payload metrics are not finite")
    relations = tuple(
        tuple(
            AIRFOIL_V7_ARCHIVE_RELATION.relate(left, right)
            for right in detailed_rows
        )
        for left in detailed_rows
    )
    rank_by_ordinal = {
        ordinal: 1
        + sum(
            relations[other - 1][ordinal - 1] is OutcomeRelation.BETTER
            for other in range(1, EXPECTED_OPTION_COUNT + 1)
        )
        for ordinal in range(1, EXPECTED_OPTION_COUNT + 1)
    }
    family_rank: dict[int, int] = {}
    for ordinal, option in enumerate(verified.contract.options, start=1):
        peers = [
            index
            for index, peer in enumerate(verified.contract.options, start=1)
            if peer.family == option.family
        ]
        family_rank[ordinal] = 1 + sum(
            relations[peer - 1][ordinal - 1] is OutcomeRelation.BETTER
            for peer in peers
        )
    display_order = sorted(
        range(1, EXPECTED_OPTION_COUNT + 1),
        key=lambda ordinal: (
            metrics[ordinal - 1][0],
            metrics[ordinal - 1][1],
            verified.contract.options[ordinal - 1].option_id.encode("ascii"),
        ),
    )
    rewards = tuple(
        local_delta_parent_feedback(child, parent_detailed) for child in detailed_rows
    )
    rows = []
    for ordinal, (option, payload, terminal) in enumerate(
        zip(
            verified.contract.options,
            state.terminal_payloads,
            state.terminal_records,
            strict=True,
        ),
        start=1,
    ):
        rank = rank_by_ordinal[ordinal]
        better = sum(
            relation is OutcomeRelation.WORSE
            for relation in relations[ordinal - 1]
        )
        equivalent = sum(
            relation is OutcomeRelation.EQUIVALENT
            for relation in relations[ordinal - 1]
        )
        worse = EXPECTED_OPTION_COUNT - better - equivalent
        best_ordinal = display_order[0]
        rows.append(
            {
                **_option_identity(option, ordinal=ordinal),
                "objectives": dict(payload.objectives),
                "violations": dict(payload.violations),
                "rank": rank,
                "display_order": display_order.index(ordinal) + 1,
                "rank_percentile_0_best": (rank - 1) / EXPECTED_OPTION_COUNT,
                "family_rank": family_rank[ordinal],
                "rank_regret": rank - 1,
                "better_equivalent_worse": {
                    "better": better,
                    "equivalent": equivalent,
                    "worse": worse,
                },
                "contextual_parent_reward": rewards[ordinal - 1],
                "difference_from_oracle_best": {
                    "normalized_lift_equality": (
                        metrics[ordinal - 1][0] - metrics[best_ordinal - 1][0]
                    ),
                    "normalized_multipoint_drag": (
                        metrics[ordinal - 1][1] - metrics[best_ordinal - 1][1]
                    ),
                },
                "terminal_record_sha256": terminal["record_sha256"],
            }
        )
    by_id = {str(row["option_id"]): row for row in rows}
    known = []
    for item in KNOWN_ACTIONS:
        current = by_id[str(item["option_id"])]
        prior_f = float(item[OBJECTIVE_NAME])
        prior_v = float(item[VIOLATION_NAME])
        known.append(
            {
                "arm": item["arm"],
                "option_id": item["option_id"],
                "prior_objectives": {OBJECTIVE_NAME: prior_f},
                "prior_violations": {VIOLATION_NAME: prior_v},
                "prior_contextual_parent_reward": _parent_reward_from_metrics(
                    objective=prior_f,
                    violation=prior_v,
                ),
                "fresh_objectives": dict(current["objectives"]),
                "fresh_violations": dict(current["violations"]),
                "fresh_contextual_parent_reward": current[
                    "contextual_parent_reward"
                ],
                "rank": current["rank"],
                "rank_percentile_0_best": current["rank_percentile_0_best"],
                "family_rank": current["family_rank"],
            }
        )
    raw_solver_calls = sum(
        int(terminal["raw_receipt"]["raw_solver_calls"])
        for terminal in state.terminal_records
        if type(terminal.get("raw_receipt")) is dict
    )
    if raw_solver_calls != EXPECTED_RANS_CALLS:
        raise OracleContractError("complete oracle does not account for 240 RANS calls")

    def uniform_record(target: DetailedEvaluation, family: str | None = None) -> dict[str, object]:
        indices = [
            index
            for index, option in enumerate(verified.contract.options)
            if family is None or option.family == family
        ]
        target_relations = [
            AIRFOIL_V7_ARCHIVE_RELATION.relate(detailed_rows[index], target)
            for index in indices
        ]
        denominator = len(indices)
        counts = {
            relation.value: sum(item is relation for item in target_relations)
            for relation in (
                OutcomeRelation.BETTER,
                OutcomeRelation.EQUIVALENT,
                OutcomeRelation.WORSE,
            )
        }
        return {
            "denominator": denominator,
            "counts": counts,
            "probabilities": {
                name: count / denominator for name, count in counts.items()
            },
        }

    uniform = {
        "parent": {
            "overall": uniform_record(parent_detailed),
            "shape_only": uniform_record(parent_detailed, "shape_only"),
            "trim_only": uniform_record(parent_detailed, "trim_only"),
        }
    }
    for item in KNOWN_ACTIONS[:3]:
        target_ordinal = int(by_id[str(item["option_id"])]["ordinal"])
        target = detailed_rows[target_ordinal - 1]
        uniform[str(item["arm"])] = {
            "overall": uniform_record(target),
            "shape_only": uniform_record(target, "shape_only"),
            "trim_only": uniform_record(target, "trim_only"),
        }

    def one_action_mass(family: str | None = None) -> dict[str, object]:
        ordinals = [
            ordinal
            for ordinal, option in enumerate(verified.contract.options, start=1)
            if family is None or option.family == family
        ]
        denominator = len(ordinals)
        rank_counts = {
            str(rank): sum(rank_by_ordinal[ordinal] == rank for ordinal in ordinals)
            for rank in sorted({rank_by_ordinal[ordinal] for ordinal in ordinals})
        }
        family_rank_counts = {
            str(rank): sum(family_rank[ordinal] == rank for ordinal in ordinals)
            for rank in sorted({family_rank[ordinal] for ordinal in ordinals})
        }
        reward_counts = {
            label: sum(rewards[ordinal - 1] == value for ordinal in ordinals)
            for label, value in (
                ("plus_one", 1.0),
                ("zero", 0.0),
                ("minus_one", -1.0),
            )
        }
        return {
            "denominator": denominator,
            "overall_scientific_rank": {
                "counts": rank_counts,
                "probabilities": {
                    rank: count / denominator for rank, count in rank_counts.items()
                },
            },
            "conditional_family_rank": {
                "counts": family_rank_counts,
                "probabilities": {
                    rank: count / denominator
                    for rank, count in family_rank_counts.items()
                },
            },
            "contextual_parent_reward": {
                "counts": reward_counts,
                "probabilities": {
                    label: count / denominator
                    for label, count in reward_counts.items()
                },
            },
        }

    uniform["mass_functions"] = {
        "overall": one_action_mass(),
        "shape_only": one_action_mass("shape_only"),
        "trim_only": one_action_mass("trim_only"),
    }

    portfolio_best_ranks: list[int] = []
    parent_better_portfolios = 0
    for combination in itertools.combinations(range(EXPECTED_OPTION_COUNT), 3):
        best = min(combination, key=lambda index: (metrics[index], verified.contract.options[index].option_id.encode("ascii")))
        portfolio_best_ranks.append(rank_by_ordinal[best + 1])
        if any(
            AIRFOIL_V7_ARCHIVE_RELATION.relate(
                detailed_rows[index], parent_detailed
            )
            is OutcomeRelation.BETTER
            for index in combination
        ):
            parent_better_portfolios += 1
    if len(portfolio_best_ranks) != 82_160:
        raise OracleContractError("three-action portfolio census is incomplete")
    sorted_portfolio = sorted(portfolio_best_ranks)

    def nearest_rank_quantile(numerator: int, denominator: int) -> int:
        position = math.ceil(len(sorted_portfolio) * numerator / denominator)
        return sorted_portfolio[position - 1]

    rank_mass = {
        str(rank): portfolio_best_ranks.count(rank)
        for rank in sorted(set(portfolio_best_ranks))
    }
    portfolio_targets = {}
    for label in ("A", "N"):
        target_rank = int(
            by_id[next(str(item["option_id"]) for item in KNOWN_ACTIONS if item["arm"] == label)]["rank"]
        )
        counts = {
            "better": sum(rank < target_rank for rank in portfolio_best_ranks),
            "tie": sum(rank == target_rank for rank in portfolio_best_ranks),
            "worse": sum(rank > target_rank for rank in portfolio_best_ranks),
        }
        portfolio_targets[label] = {
            "target_rank": target_rank,
            "denominator": 82_160,
            "counts": counts,
            "probabilities": {
                name: count / 82_160 for name, count in counts.items()
            },
        }
    observed_ids = {
        str(item["option_id"]) for item in KNOWN_ACTIONS if item["arm"] in {"A", "S", "N"}
    }
    observed_best_rank = min(int(by_id[option_id]["rank"]) for option_id in observed_ids)
    observed_counts = {
        "better": sum(rank < observed_best_rank for rank in portfolio_best_ranks),
        "tie": sum(rank == observed_best_rank for rank in portfolio_best_ranks),
        "worse": sum(rank > observed_best_rank for rank in portfolio_best_ranks),
    }

    nondominated = []
    for ordinal, (violation, drag) in enumerate(metrics, start=1):
        dominated = any(
            other_v <= violation
            and other_f <= drag
            and (other_v < violation or other_f < drag)
            for index, (other_v, other_f) in enumerate(metrics, start=1)
            if index != ordinal
        )
        if not dominated:
            nondominated.append(verified.contract.options[ordinal - 1].option_id)

    fresh_repeat = []
    card_stability: dict[str, dict[str, object]] = {}
    for prior in KNOWN_ACTIONS:
        current = by_id[str(prior["option_id"])]
        fresh_f = float(current["objectives"][OBJECTIVE_NAME])
        fresh_v = float(current["violations"][VIOLATION_NAME])
        prior_f = float(prior[OBJECTIVE_NAME])
        prior_v = float(prior[VIOLATION_NAME])
        prior_reward = _parent_reward_from_metrics(
            objective=prior_f,
            violation=prior_v,
        )
        fresh_reward = float(current["contextual_parent_reward"])
        prior_f_direction = _resolved_direction(
            prior_f - PARENT_METRICS[OBJECTIVE_NAME], DELTA_F
        )
        prior_v_direction = _resolved_direction(
            prior_v - PARENT_METRICS[VIOLATION_NAME], DELTA_V
        )
        fresh_f_direction = _resolved_direction(
            fresh_f - PARENT_METRICS[OBJECTIVE_NAME], DELTA_F
        )
        fresh_v_direction = _resolved_direction(
            fresh_v - PARENT_METRICS[VIOLATION_NAME], DELTA_V
        )
        row = {
                "arm": prior["arm"],
                "option_id": prior["option_id"],
                "prior_f": prior_f,
                "prior_v": prior_v,
                "prior_contextual_reward": prior_reward,
                "fresh_f": fresh_f,
                "fresh_v": fresh_v,
                "fresh_minus_prior_f": fresh_f - prior_f,
                "fresh_minus_prior_v": fresh_v - prior_v,
                "fresh_minus_parent_f": fresh_f
                - PARENT_METRICS[OBJECTIVE_NAME],
                "fresh_minus_parent_v": fresh_v
                - PARENT_METRICS[VIOLATION_NAME],
                "prior_resolved_directions": {
                    "f": prior_f_direction,
                    "v": prior_v_direction,
                },
                "fresh_resolved_directions": {
                    "f": fresh_f_direction,
                    "v": fresh_v_direction,
                },
                "fresh_contextual_reward": fresh_reward,
                "contextual_reward_reproduced": fresh_reward == prior_reward,
            }
        fresh_repeat.append(row)
        arm = prior["arm"]
        if arm in {"A", "S"}:
            expected = (
                {"f": "increase", "v": "decrease"}
                if arm == "A"
                else {"f": "decrease", "v": "increase"}
            )
            directions_reproduced = (
                fresh_f_direction == prior_f_direction == expected["f"]
                and fresh_v_direction == prior_v_direction == expected["v"]
            )
            reward_reproduced = fresh_reward == prior_reward
            card_stability[str(arm)] = {
                "expected_directions": expected,
                "prior_resolved_directions": row["prior_resolved_directions"],
                "fresh_resolved_directions": row["fresh_resolved_directions"],
                "directions_reproduced": directions_reproduced,
                "prior_contextual_reward": prior_reward,
                "fresh_contextual_reward": fresh_reward,
                "contextual_reward_reproduced": reward_reproduced,
                "local_stability_retained": directions_reproduced
                and reward_reproduced,
                "decision": (
                    "retain_local_stability_statement"
                    if directions_reproduced and reward_reproduced
                    else "kill_local_stability_statement"
                ),
            }

    adaptive_rank = int(by_id["shape.camber_aft.p0015"]["rank"])
    if adaptive_rank <= 20:
        adaptive_decision = "retain_top_quartile_selection_hypothesis"
    elif adaptive_rank <= 40:
        adaptive_decision = "require_selector_expansion_or_redesign"
    else:
        adaptive_decision = "kill_unchanged_single_card_selection_rule"
    portfolio_median = nearest_rank_quantile(1, 2)
    reject_three_arm_competitiveness = observed_best_rank > portfolio_median

    unsigned: dict[str, object] = {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "kind": "airfoil_v7_finite_oracle_result",
        "run_id": verified.run_id,
        "manifest_sha256": verified.manifest_sha256,
        "source_sha256": verified.source_sha256,
        "status": "completed_80_action_oracle",
        "complete_ranking_available": True,
        "provider_calls": 0,
        "credentials_read": False,
        "candidate_attempts": EXPECTED_OPTION_COUNT,
        "successful_candidates": EXPECTED_OPTION_COUNT,
        "raw_solver_calls": raw_solver_calls,
        "expected_full_rans_calls": EXPECTED_RANS_CALLS,
        "ranking_semantics": _analysis_binding()["ranking"],
        "results": rows,
        "known_action_results": known,
        "held_out_parent_metrics": dict(PARENT_METRICS),
        "contextual_reward_mass": {
            "plus_one": rewards.count(1.0),
            "zero": rewards.count(0.0),
            "minus_one": rewards.count(-1.0),
            "policy_definition_sha256": REWARD_DEFINITION_SHA256,
        },
        "exact_uniform_one_action": uniform,
        "three_action_portfolios": {
            "combination_count": 82_160,
            "best_rank_mass": rank_mass,
            "q25_best_rank": nearest_rank_quantile(1, 4),
            "median_best_rank": nearest_rank_quantile(1, 2),
            "q75_best_rank": nearest_rank_quantile(3, 4),
            "mean_best_rank": sum(portfolio_best_ranks) / 82_160,
            "comparisons": portfolio_targets,
            "at_least_one_beats_parent": {
                "count": parent_better_portfolios,
                "denominator": 82_160,
                "probability": parent_better_portfolios / 82_160,
            },
            "observed_asn_best_rank": observed_best_rank,
            "observed_asn_comparison": {
                "denominator": 82_160,
                "counts": observed_counts,
                "probabilities": {
                    name: count / 82_160
                    for name, count in observed_counts.items()
                },
            },
            "observed_asn_percentile_definition": (
                "fraction_of_portfolios_with_strictly_better_best_rank;"
                "zero_is_best"
            ),
            "observed_asn_percentile_0_best": sum(
                rank < observed_best_rank for rank in portfolio_best_ranks
            )
            / 82_160,
        },
        "nondominated_option_ids": nondominated,
        "fresh_repeat_audit": fresh_repeat,
        "prospective_decisions": {
            "adaptive_rank": adaptive_rank,
            "adaptive_rank_decision": adaptive_decision,
            "card_local_stability": card_stability,
            "random_three_action_median_best_rank": portfolio_median,
            "observed_asn_best_rank": observed_best_rank,
            "observed_asn_worse_than_random_median": (
                reject_three_arm_competitiveness
            ),
            "reject_three_arm_competitiveness_claim": (
                reject_three_arm_competitiveness
            ),
        },
    }
    return {
        **unsigned,
        "result_sha256": _self_hash(unsigned, framing=ORACLE_RESULT_FRAMING),
    }


def _write_or_verify_result(
    run_dir: Path,
    *,
    verified: VerifiedOracleManifest,
    state: OracleJournalState,
) -> dict[str, object]:
    result = _result_record(verified, state)
    result_path = run_dir / "oracle_result.json"
    journal_path = run_dir / "option_results.jsonl"
    journal_bytes = b"".join(
        _canonical_bytes(record) + b"\n" for record in state.terminal_records
    )
    if result_path.exists():
        if json.loads(result_path.read_bytes()) != result:
            raise OracleContractError("existing oracle result differs from journal")
    else:
        write_json_atomic(result_path, result)
    if journal_path.exists():
        if journal_path.read_bytes() != journal_bytes:
            raise OracleContractError("existing aggregate journal differs from options")
    else:
        write_bytes_atomic(journal_path, journal_bytes)
    return result


def _recursive_content_binding(
    run_dir: Path,
) -> tuple[dict[str, dict[str, object]], str]:
    files: dict[str, dict[str, object]] = {}
    aggregate = hashlib.sha256(ORACLE_FINALIZATION_FRAMING)
    for path in sorted(
        (
            item
            for item in run_dir.rglob("*")
            if item.is_file() and item != run_dir / "finalized.json"
        ),
        key=lambda item: item.relative_to(run_dir).as_posix(),
    ):
        if path.is_symlink():
            raise OracleContractError("oracle run contains a symbolic-link file")
        relative = path.relative_to(run_dir).as_posix()
        content = path.read_bytes()
        files[relative] = {
            "sha256": _sha256_bytes(content),
            "bytes": len(content),
        }
        encoded = relative.encode("utf-8")
        aggregate.update(len(encoded).to_bytes(8, "big"))
        aggregate.update(encoded)
        aggregate.update(len(content).to_bytes(8, "big"))
        aggregate.update(content)
    return files, aggregate.hexdigest()


def _finalize_run(run_dir: Path, *, status: str) -> dict[str, object]:
    finalized_path = run_dir / "finalized.json"
    files, aggregate_sha256 = _recursive_content_binding(run_dir)
    if finalized_path.exists():
        existing = _read_sealed_record(finalized_path, kind="oracle_run_finalized")
        if set(existing) != {
            "schema_version",
            "kind",
            "status",
            "recursive_file_count",
            "recursive_content_sha256",
            "files",
            "finalized_at_utc",
            "record_sha256",
        }:
            raise OracleContractError("oracle finalization fields changed")
        if (
            existing.get("schema_version") != ORACLE_SCHEMA_VERSION
            or existing.get("status") != status
            or existing.get("recursive_file_count") != len(files)
            or existing.get("recursive_content_sha256") != aggregate_sha256
            or existing.get("files") != files
        ):
            raise OracleContractError(
                "oracle finalized recursive content or accounting changed"
            )
        return existing
    record = _sealed_record(
        {
            "schema_version": ORACLE_SCHEMA_VERSION,
            "kind": "oracle_run_finalized",
            "status": status,
            "recursive_file_count": len(files),
            "recursive_content_sha256": aggregate_sha256,
            "files": files,
            "finalized_at_utc": _utc_now(),
        }
    )
    write_json_atomic(finalized_path, record)
    return record


def _initialize_run(verified: VerifiedOracleManifest) -> None:
    run_dir = verified.output_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(run_dir.parent)
    write_bytes_atomic(run_dir / "oracle_manifest.json", verified.path.read_bytes())
    write_json_atomic(
        run_dir / "catalog_contract.json",
        verified.record["oracle"]["catalog"],
    )
    (run_dir / "options").mkdir()
    (run_dir / "invocations").mkdir()
    _fsync_directory(run_dir)


def _verify_initialized_run(verified: VerifiedOracleManifest) -> None:
    run_dir = verified.output_dir
    copied = run_dir / "oracle_manifest.json"
    if not copied.is_file() or copied.read_bytes() != verified.path.read_bytes():
        raise OracleContractError("copied oracle manifest differs from launch bytes")
    contract_path = run_dir / "catalog_contract.json"
    if (
        not contract_path.is_file()
        or json.loads(contract_path.read_bytes())
        != verified.record["oracle"]["catalog"]
    ):
        raise OracleContractError("copied oracle catalog contract changed")


def _recover_or_invalidate_open_start(
    *,
    verified: VerifiedOracleManifest,
    dependencies: OracleExecutionDependencies,
    state: OracleJournalState,
) -> OracleJournalState:
    if state.open_started is None:
        return state
    ordinal = state.next_ordinal
    option = verified.contract.options[ordinal - 1]
    expected_raw_sha = candidate_sha256(thaw_json(option.child_configuration))
    receipt_path = _locate_current_receipt(
        verified.output_dir,
        referenced=set(state.referenced_receipts),
        expected_candidate_sha256=expected_raw_sha,
    )
    if receipt_path is None:
        _write_invalidation(
            verified.output_dir,
            verified=verified,
            reason="open_started_attempt_has_no_durable_raw_receipt",
            ordinal=ordinal,
            option_id=option.option_id,
        )
        raise OracleRunInvalidated(
            "open oracle attempt has no durable receipt; one-attempt run invalidated"
        )
    child = normalize_candidate(thaw_json(option.child_configuration))
    receipt = _receipt_binding(
        receipt_path,
        run_dir=verified.output_dir,
        expected_candidate_sha256=expected_raw_sha,
    )
    payload = dependencies.receipt_replayer(receipt_path, child)
    post_source = _verify_current_source(
        verified,
        dependencies,
        stage=f"recovered_option_{ordinal:03d}_pre_terminal",
    )
    write_json_atomic(
        _option_directory(verified.output_dir, ordinal, option.option_id)
        / "post_evaluation_source_verified.json",
        post_source,
    )
    _write_terminal(
        _option_directory(verified.output_dir, ordinal, option.option_id),
        verified=verified,
        option=option,
        ordinal=ordinal,
        started=state.open_started,
        payload=payload,
        receipt=receipt,
        outer_wall_seconds=None,
        recovered_after_interruption=True,
        post_evaluation_source=post_source,
    )
    state = _read_journal(
        verified.output_dir,
        verified=verified,
        receipt_replayer=dependencies.receipt_replayer,
    )
    if payload.failure is not None:
        _write_invalidation(
            verified.output_dir,
            verified=verified,
            reason=f"recovered_{payload.failure.category.value}_failure",
            ordinal=ordinal,
            option_id=option.option_id,
        )
        raise OracleRunInvalidated("recovered oracle receipt is a failed evaluation")
    return state


def _execute_oracle_locked(
    verified: VerifiedOracleManifest,
    *,
    resume: bool,
    dependencies: OracleExecutionDependencies,
) -> dict[str, object]:
    """Mutate one authenticated oracle run while its run lock is held."""

    run_dir = verified.output_dir
    if resume:
        _verify_initialized_run(verified)
        if (run_dir / "finalized.json").exists():
            raise OracleContractError("finalized oracle runs are immutable")
        if (run_dir / "invalidation.json").exists():
            raise OracleRunInvalidated("invalidated oracle runs cannot resume")
    else:
        _initialize_run(verified)

    invocation_id, invocation_dir = _next_invocation_dir(run_dir)
    write_json_atomic(
        invocation_dir / "started.json",
        _sealed_record(
            {
                "schema_version": ORACLE_SCHEMA_VERSION,
                "kind": "oracle_invocation_started",
                "run_id": verified.run_id,
                "invocation_id": invocation_id,
                "resume": resume,
                "manifest_sha256": verified.manifest_sha256,
                "started_at_utc": _utc_now(),
            }
        ),
    )
    source_record = _verify_current_source(
        verified,
        dependencies,
        stage=f"invocation_{invocation_id}_pre_lease",
    )
    write_json_atomic(invocation_dir / "source_verified.json", source_record)

    lease: ExclusiveResourceLease | None = None
    pending: BaseException | None = None
    result: dict[str, object] | None = None
    outcome = "failed_before_completion"
    try:
        lease = dependencies.resource_lease_factory(
            verified.run_id,
            "finite_catalog_oracle",
        )
        lease_receipt = lease.acquire()
        write_json_atomic(
            invocation_dir / "resource_lease_acquired.json",
            {
                "schema_version": ORACLE_SCHEMA_VERSION,
                "receipt": lease_receipt.to_record(),
            },
        )
        benchmark = dependencies.benchmark_factory(verified.run_id, run_dir)
        if type(benchmark) is not AgenticBenchmark:
            raise OracleContractError("oracle benchmark factory returned the wrong type")
        runtime_contract = benchmark.bind_finite_variation(
            "airfoil_v7_union",
            materialize_held_out_parent().candidate,
        )
        if runtime_contract != verified.contract:
            raise OracleContractError("runtime oracle catalog differs from manifest")
        evaluator = benchmark.detailed_evaluator
        if evaluator is None or evaluator.evaluator_identity != EVALUATOR_IDENTITY:
            raise OracleContractError("runtime oracle evaluator identity changed")

        state = _read_journal(
            run_dir,
            verified=verified,
            receipt_replayer=dependencies.receipt_replayer,
        )
        if state.failed_ordinal is not None:
            option = verified.contract.options[state.failed_ordinal - 1]
            _write_invalidation(
                run_dir,
                verified=verified,
                reason="journal_contains_failed_typed_evaluation",
                ordinal=state.failed_ordinal,
                option_id=option.option_id,
            )
            raise OracleRunInvalidated("oracle journal contains a failed evaluation")
        state = _recover_or_invalidate_open_start(
            verified=verified,
            dependencies=dependencies,
            state=state,
        )

        while state.next_ordinal <= EXPECTED_OPTION_COUNT:
            ordinal = state.next_ordinal
            option = verified.contract.options[ordinal - 1]
            current_source = _verify_current_source(
                verified,
                dependencies,
                stage=f"pre_option_{ordinal:03d}",
            )
            option_dir = _option_directory(run_dir, ordinal, option.option_id)
            started = _write_started(
                option_dir,
                verified=verified,
                invocation_id=invocation_id,
                option=option,
                ordinal=ordinal,
            )
            write_json_atomic(option_dir / "source_verified.json", current_source)
            child = normalize_candidate(thaw_json(option.child_configuration))
            started_ns = dependencies.monotonic_ns()
            try:
                payload = evaluator.evaluate_evidence(child)
            except Exception as exc:
                _write_invalidation(
                    run_dir,
                    verified=verified,
                    reason=f"evaluator_raised_{type(exc).__name__}",
                    ordinal=ordinal,
                    option_id=option.option_id,
                )
                raise OracleRunInvalidated(
                    "oracle evaluator raised after its single charged attempt"
                ) from exc
            finished_ns = dependencies.monotonic_ns()
            if (
                type(started_ns) is not int
                or type(finished_ns) is not int
                or finished_ns < started_ns
            ):
                raise OracleContractError("oracle monotonic clock is invalid")
            expected_raw_sha = candidate_sha256(child)
            receipt_path = _locate_current_receipt(
                run_dir,
                referenced=set(state.referenced_receipts),
                expected_candidate_sha256=expected_raw_sha,
            )
            if receipt_path is None:
                _write_invalidation(
                    run_dir,
                    verified=verified,
                    reason="evaluator_returned_without_durable_raw_receipt",
                    ordinal=ordinal,
                    option_id=option.option_id,
                )
                raise OracleRunInvalidated(
                    "oracle evaluator returned without a durable raw receipt"
                )
            receipt = _receipt_binding(
                receipt_path,
                run_dir=run_dir,
                expected_candidate_sha256=expected_raw_sha,
            )
            replayed = dependencies.receipt_replayer(receipt_path, child)
            if _payload_record(replayed) != _payload_record(payload):
                _write_invalidation(
                    run_dir,
                    verified=verified,
                    reason="live_payload_differs_from_durable_receipt_replay",
                    ordinal=ordinal,
                    option_id=option.option_id,
                )
                raise OracleRunInvalidated(
                    "live oracle payload differs from durable receipt replay"
                )
            if dependencies.after_raw_receipt is not None:
                dependencies.after_raw_receipt(receipt_path, option)
            post_source = _verify_current_source(
                verified,
                dependencies,
                stage=f"post_option_{ordinal:03d}_pre_terminal",
            )
            write_json_atomic(
                option_dir / "post_evaluation_source_verified.json",
                post_source,
            )
            _write_terminal(
                option_dir,
                verified=verified,
                option=option,
                ordinal=ordinal,
                started=started,
                payload=payload,
                receipt=receipt,
                outer_wall_seconds=(finished_ns - started_ns) / 1_000_000_000,
                recovered_after_interruption=False,
                post_evaluation_source=post_source,
            )
            state = _read_journal(
                run_dir,
                verified=verified,
                receipt_replayer=dependencies.receipt_replayer,
            )
            if payload.failure is not None:
                _write_invalidation(
                    run_dir,
                    verified=verified,
                    reason=f"typed_{payload.failure.category.value}_failure",
                    ordinal=ordinal,
                    option_id=option.option_id,
                )
                raise OracleRunInvalidated(
                    "one oracle option produced a failed typed evaluation"
                )

        _verify_current_source(
            verified,
            dependencies,
            stage="post_option_080_pre_result",
        )
        result = _write_or_verify_result(
            run_dir,
            verified=verified,
            state=state,
        )
        outcome = "completed_80_action_oracle"
    except BaseException as exc:
        pending = exc
        if isinstance(exc, OracleRunInvalidated):
            outcome = "invalidated_partial_oracle"
        elif isinstance(exc, OracleContractError) and any(
            (run_dir / "options").iterdir()
        ):
            _write_invalidation(
                run_dir,
                verified=verified,
                reason=f"post_initialization_contract_failure_{type(exc).__name__}",
                ordinal=None,
                option_id=None,
            )
            outcome = "invalidated_partial_oracle"
    finally:
        if lease is not None and lease.active:
            try:
                release = lease.release(
                    outcome=outcome,
                    failure_type=None if pending is None else type(pending).__name__,
                )
                write_json_atomic(
                    invocation_dir / "resource_lease_released.json",
                    {
                        "schema_version": ORACLE_SCHEMA_VERSION,
                        "release": (
                            release.to_record()
                            if callable(getattr(release, "to_record", None))
                            else release
                        ),
                    },
                )
            except BaseException as exc:
                if pending is None:
                    pending = exc
                else:
                    pending.add_note(
                        f"resource lease release also failed: {type(exc).__name__}"
                    )
        try:
            write_json_atomic(
                invocation_dir / "outcome.json",
                {
                    "schema_version": ORACLE_SCHEMA_VERSION,
                    "outcome": outcome,
                    "failure_type": None if pending is None else type(pending).__name__,
                    "finished_at_utc": _utc_now(),
                },
            )
        except BaseException as exc:
            if pending is None:
                pending = exc
            else:
                pending.add_note(
                    f"invocation outcome publication also failed: {type(exc).__name__}"
                )
    if outcome == "completed_80_action_oracle" and pending is None:
        try:
            _finalize_run(run_dir, status=outcome)
        except BaseException as exc:
            if pending is None:
                pending = exc
            else:
                pending.add_note(
                    f"oracle finalization also failed: {type(exc).__name__}"
                )
    if pending is not None:
        raise pending
    if result is None:
        raise OracleContractError("oracle execution returned without a result")
    return result


def execute_oracle(
    manifest_path: Path,
    *,
    resume: bool,
    dependencies: OracleExecutionDependencies | None = None,
) -> dict[str, object]:
    """Execute or explicitly resume the frozen provider-free 80-action oracle."""

    if type(resume) is not bool:
        raise TypeError("resume must be an exact bool")
    dependencies = dependencies or production_oracle_dependencies()
    observed = verify_oracle_manifest(
        manifest_path,
        require_output_absent=None,
        source_snapshot_factory=dependencies.source_snapshot_factory,
        enforce_canonical_output=dependencies.enforce_canonical_output,
    )
    run_lock = _new_run_lock(observed)
    run_lock.acquire()
    pending: BaseException | None = None
    result: dict[str, object] | None = None
    try:
        # Recheck the manifest and output precondition after acquiring the lock.
        # This closes the race between an initial existence check and mutation.
        verified = verify_oracle_manifest(
            manifest_path,
            require_output_absent=not resume,
            source_snapshot_factory=dependencies.source_snapshot_factory,
            enforce_canonical_output=dependencies.enforce_canonical_output,
        )
        result = _execute_oracle_locked(
            verified,
            resume=resume,
            dependencies=dependencies,
        )
    except BaseException as exc:
        pending = exc
    finally:
        try:
            run_lock.release(
                outcome="completed" if pending is None else "failed",
                failure_type=None if pending is None else type(pending).__name__,
            )
        except BaseException as exc:
            if pending is None:
                pending = exc
            else:
                pending.add_note(
                    f"run-level lock release also failed: {type(exc).__name__}"
                )
    if pending is not None:
        raise pending
    if result is None:  # pragma: no cover - defensive control-flow invariant.
        raise OracleContractError("locked oracle execution returned no result")
    return result


__all__ = [
    "DEFAULT_ORACLE_ROOT",
    "DEFAULT_PRIOR_RUN_DIR",
    "EXPECTED_OPTION_COUNT",
    "EXPECTED_RANS_CALLS",
    "KNOWN_ACTIONS",
    "ORACLE_KIND",
    "OracleContractError",
    "OracleExecutionDependencies",
    "OracleRunInvalidated",
    "VerifiedOracleManifest",
    "execute_oracle",
    "production_oracle_dependencies",
    "verify_oracle_manifest",
    "write_oracle_manifest",
]
